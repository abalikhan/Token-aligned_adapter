# evaluate.py (FINAL — PCK aligned with SOTA: normalized by predicted bbox diagonal)
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
import csv
from collections import defaultdict

from torch.utils.data import DataLoader
from utils_new import (
    collate_temporal_chunks,
    save_kp_overlay_bt,
    save_heatmap_overlay_bt,
    _format_row_4dp,
    save_token_similarity_diagnostics_bt,
)

from misc import (
    create_pose_estimator,
    estimate_pose_from_keypoints,
    compute_speed_metrics_batch
)

from losses import softargmax_2d_from_logits
from dataset import SPADESDataset, SPARKDataset
from model_hm_dsnt_2 import SpacecraftHMDSNT


# -------------------------
# Helpers
# -------------------------
import random

@torch.no_grad()
def forward_microbatch(
    model,
    video: torch.Tensor,
    micro_bs: int = 16,
    keep_keys=("heatmaps", "keypoints", "hm_prob", "kp_scores"),
):
    """
    Run the model in micro-batches to avoid OOM, and only keep required outputs.

    video: (B, T, C, H, W)

    Returns a dict with a subset of keys, concatenated on dim=0 (batch dim).
    """
    B = video.shape[0]
    kept_chunks = {k: [] for k in keep_keys}

    for s in range(0, B, micro_bs):
        e = min(B, s + micro_bs)

        out = model(video[s:e])
        if not isinstance(out, dict):
            raise TypeError(f"Expected dict output, got {type(out)}")

        # keep only what eval uses
        for k in keep_keys:
            if k in out and out[k] is not None:
                if not torch.is_tensor(out[k]):
                    raise TypeError(f"Expected tensor for key '{k}', got {type(out[k])}")
                kept_chunks[k].append(out[k])

        # aggressively drop references to big tensors (adapter_tokens/itoken/etc.)
        # so they can be freed before the next microbatch
        del out

    # Merge (cat) only if we have chunks for that key
    merged = {}
    for k, chunks in kept_chunks.items():
        if len(chunks) == 0:
            merged[k] = None
        else:
            merged[k] = torch.cat(chunks, dim=0)

    return merged

def _kp_scores_from_hm_logits(hm_logits_bt: torch.Tensor) -> torch.Tensor:
    """
    hm_logits_bt: (B,T,K,Hm,Wm)
    returns:      (B,T,K) scores (higher is better)
    NOTE: For ranking, max logit is enough and avoids extra softmax cost.
    """
    return hm_logits_bt.amax(dim=(-2, -1))


def is_main_process():
    try:
        import torch.distributed as dist
        return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    except Exception:
        return True

def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model

@torch.no_grad()
def _forward_model(model, video):
    real = _unwrap_model(model)
    return real(video)

def _quat_xyzw_to_wxyz(q_xyzw: torch.Tensor) -> torch.Tensor:
    return torch.stack([q_xyzw[..., 3], q_xyzw[..., 0], q_xyzw[..., 1], q_xyzw[..., 2]], dim=-1)

def _normalize_quat_safe(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)

def _norm_hm_to_hm_px(kps_norm_hm: torch.Tensor, heatmap_hw):
    """
    kps_norm_hm: (...,2) in [-1,1] w.r.t heatmap grid
    returns: (...,2) in heatmap pixel coords [0..Wm-1],[0..Hm-1]
    """
    Hm, Wm = heatmap_hw
    x = (kps_norm_hm[..., 0] + 1.0) * 0.5 * (Wm - 1.0)
    y = (kps_norm_hm[..., 1] + 1.0) * 0.5 * (Hm - 1.0)
    return torch.stack([x, y], dim=-1)

def _hm_px_to_crop_px(kps_hm_px: torch.Tensor, input_size: int, heatmap_hw):
    """
    Invert dataset mapping (typical):
      hm_px = crop_px * ((Wm-1)/(inp-1))  =>  crop_px = hm_px * ((inp-1)/(Wm-1))
    crop_px lives in [0..input_size-1]
    """
    Hm, Wm = heatmap_hw
    den_w = float(max(Wm - 1, 1))
    den_h = float(max(Hm - 1, 1))
    num = float(max(input_size - 1, 1))
    x_crop = kps_hm_px[..., 0] * (num / den_w)
    y_crop = kps_hm_px[..., 1] * (num / den_h)
    return torch.stack([x_crop, y_crop], dim=-1)

def _hm_norm_to_input_px(kps_norm_hm: torch.Tensor, input_size: int, heatmap_hw):
    """
    kps_norm_hm: (...,2) in [-1,1] w.r.t heatmap grid
    returns: (...,2) in input image pixel coords [0..input_size-1]
    """
    Hm, Wm = heatmap_hw
    kps_hm_px = _norm_hm_to_hm_px(kps_norm_hm, heatmap_hw=(Hm, Wm))
    kps_in_px = _hm_px_to_crop_px(kps_hm_px, input_size=input_size, heatmap_hw=(Hm, Wm))
    return kps_in_px

def pck_on_input_px_from_hm_norm(kps_pred_norm, kps_gt_norm, vis, input_size, heatmap_hw, thresh_px):
    pred_in = _hm_norm_to_input_px(kps_pred_norm, input_size=input_size, heatmap_hw=heatmap_hw)
    gt_in   = _hm_norm_to_input_px(kps_gt_norm,   input_size=input_size, heatmap_hw=heatmap_hw)
    d = torch.norm(pred_in - gt_in, dim=-1)
    m = (vis > 0.5)
    return ((d < thresh_px) & m).float().sum() / m.float().sum().clamp_min(1)

def pck_normalized_by_pred_bbox_diag(pred_xy, gt_xy, vis, thr_frac: float, eps: float = 1e-8):
    """
    pred_xy, gt_xy: (B,T,K,2) in SAME pixel coord space (full px recommended)
    vis:            (B,T,K) float/bool visibility mask
    thr_frac:       e.g. 0.01, 0.05, 0.10
    returns: scalar PCK over all visible points (float)
    """
    d = torch.norm(pred_xy - gt_xy, dim=-1)  # (B,T,K)
    m = (vis > 0.5)

    big = 1e9
    px = pred_xy[..., 0]
    py = pred_xy[..., 1]

    px_min = torch.where(m, px, torch.full_like(px, big)).amin(dim=-1)   # (B,T)
    px_max = torch.where(m, px, torch.full_like(px, -big)).amax(dim=-1)  # (B,T)
    py_min = torch.where(m, py, torch.full_like(py, big)).amin(dim=-1)
    py_max = torch.where(m, py, torch.full_like(py, -big)).amax(dim=-1)

    diag = torch.sqrt((px_max - px_min).clamp_min(0)**2 + (py_max - py_min).clamp_min(0)**2).clamp_min(eps)  # (B,T)
    thr = (thr_frac * diag).unsqueeze(-1)  # (B,T,1)

    correct = (d <= thr) & m
    denom = m.sum().clamp_min(1)
    return (correct.float().sum() / denom).item()

def pck_norm_counts_bt(pred_xy, gt_xy, vis, thr_frac, eps=1e-8):
    """
    Returns (num_correct, num_visible) over all (B,T,K), using
    threshold = thr_frac * diag(pred_bbox) per (B,T).
    """
    d = torch.norm(pred_xy - gt_xy, dim=-1)  # (B,T,K)
    m = (vis > 0.5)

    big = 1e9
    px = pred_xy[..., 0]; py = pred_xy[..., 1]
    px_min = torch.where(m, px, torch.full_like(px, big)).amin(dim=-1)
    px_max = torch.where(m, px, torch.full_like(px, -big)).amax(dim=-1)
    py_min = torch.where(m, py, torch.full_like(py, big)).amin(dim=-1)
    py_max = torch.where(m, py, torch.full_like(py, -big)).amax(dim=-1)

    diag = torch.sqrt((px_max - px_min).clamp_min(0)**2 + (py_max - py_min).clamp_min(0)**2).clamp_min(eps)  # (B,T)
    thr = (thr_frac * diag).unsqueeze(-1)  # (B,T,1)

    correct = (d <= thr) & m
    return correct.float().sum().item(), m.float().sum().item()

def _kps_normhm_to_full_px(
    kps_norm_hm: torch.Tensor,     # (B,T,K,2) [-1,1] on heatmap grid
    crop_xywh_norm: torch.Tensor,  # (B,T,4) normalized (x0/W, y0/H, w/W, h/H)
    original_size_wh,
    input_size: int,
    heatmap_hw,
):
    """
    Unified conversion for BOTH GT and DSNT preds (since both are heatmap-normalized coords).
    """
    W_full, H_full = original_size_wh
    Hm, Wm = heatmap_hw

    kps_hm_px = _norm_hm_to_hm_px(kps_norm_hm, heatmap_hw=(Hm, Wm))   # (B,T,K,2)
    kps_crop  = _hm_px_to_crop_px(kps_hm_px, input_size=input_size, heatmap_hw=(Hm, Wm))

    x0 = crop_xywh_norm[..., 0] * W_full
    y0 = crop_xywh_norm[..., 1] * H_full
    w  = crop_xywh_norm[..., 2] * W_full
    h  = crop_xywh_norm[..., 3] * H_full

    x0 = x0.unsqueeze(-1); y0 = y0.unsqueeze(-1)
    w  = w.unsqueeze(-1);  h  = h.unsqueeze(-1)

    den = float(max(input_size - 1, 1))
    x_full = x0 + (kps_crop[..., 0] / den) * w
    y_full = y0 + (kps_crop[..., 1] / den) * h
    return torch.stack([x_full, y_full], dim=-1)

def _heatmap_logits_to_full_px(
    hm_logits_bt: torch.Tensor,    # (B,T,K,Hm,Wm)
    crop_xywh_norm: torch.Tensor,
    original_size_wh,
    input_size: int,
    tau_now: float,
):
    """
    softargmax on heatmap logits -> full px.
    Matches training softargmax convention (tau).
    """
    B, T, K, Hm, Wm = hm_logits_bt.shape
    kps_hm_px = softargmax_2d_from_logits(hm_logits_bt.float(), tau=tau_now)  # (B,T,K,2) hm px
    kps_crop  = _hm_px_to_crop_px(kps_hm_px, input_size=input_size, heatmap_hw=(Hm, Wm))

    W_full, H_full = original_size_wh
    x0 = crop_xywh_norm[..., 0] * W_full
    y0 = crop_xywh_norm[..., 1] * H_full
    w  = crop_xywh_norm[..., 2] * W_full
    h  = crop_xywh_norm[..., 3] * H_full

    x0 = x0.unsqueeze(-1); y0 = y0.unsqueeze(-1)
    w  = w.unsqueeze(-1);  h  = h.unsqueeze(-1)

    den = float(max(input_size - 1, 1))
    x_full = x0 + (kps_crop[..., 0] / den) * w
    y_full = y0 + (kps_crop[..., 1] / den) * h
    return torch.stack([x_full, y_full], dim=-1)
def _kps_normhm_to_full_px_letterbox(kps_norm_hm, letterbox_meta, input_size, heatmap_hw):
    """
    letterbox_meta: (B,T,5) = [scale, pad_l, pad_t, Wfull, Hfull]
    """
    kps_in_px = _hm_norm_to_input_px(kps_norm_hm, input_size=input_size, heatmap_hw=heatmap_hw)

    scale = letterbox_meta[..., 0]  # (B,T)
    pad_l = letterbox_meta[..., 1]
    pad_t = letterbox_meta[..., 2]
    Wfull = letterbox_meta[..., 3]
    Hfull = letterbox_meta[..., 4]

    x_unpad = kps_in_px[..., 0] - pad_l[..., None]
    y_unpad = kps_in_px[..., 1] - pad_t[..., None]

    x_full = x_unpad / scale[..., None].clamp_min(1e-8)
    y_full = y_unpad / scale[..., None].clamp_min(1e-8)

    # tensor bounds for clamp (avoid clamp(Tensor, float, Tensor) error)
    x_max = (Wfull[..., None] - 1.0).clamp_min(0.0)
    y_max = (Hfull[..., None] - 1.0).clamp_min(0.0)
    x_min = torch.zeros_like(x_full)
    y_min = torch.zeros_like(y_full)

    x_full = torch.minimum(torch.maximum(x_full, x_min), x_max)
    y_full = torch.minimum(torch.maximum(y_full, y_min), y_max)

    return torch.stack([x_full, y_full], dim=-1)


def _heatmap_logits_to_full_px_letterbox(
    hm_logits_bt: torch.Tensor,     # (B,T,K,Hm,Wm)
    letterbox_meta: torch.Tensor,   # (B,T,5)
    input_size: int,
    heatmap_hw,
    tau_now: float,
):
    """
    softargmax on logits -> hm px -> input px -> original full px (letterbox inverse)
    """
    B, T, K, Hm, Wm = hm_logits_bt.shape
    kps_hm_px = softargmax_2d_from_logits(hm_logits_bt.float(), tau=tau_now)  # (B,T,K,2)
    kps_in_px = _hm_px_to_crop_px(kps_hm_px, input_size=input_size, heatmap_hw=(Hm, Wm))

    scale = letterbox_meta[..., 0]
    pad_l = letterbox_meta[..., 1]
    pad_t = letterbox_meta[..., 2]
    Wfull = letterbox_meta[..., 3]
    Hfull = letterbox_meta[..., 4]

    x_unpad = kps_in_px[..., 0] - pad_l[..., None]
    y_unpad = kps_in_px[..., 1] - pad_t[..., None]

    x_full = x_unpad / scale[..., None].clamp_min(1e-8)
    y_full = y_unpad / scale[..., None].clamp_min(1e-8)

    # tensor bounds for clamp (avoid clamp(Tensor, float, Tensor) error)
    x_max = (Wfull[..., None] - 1.0).clamp_min(0.0)
    y_max = (Hfull[..., None] - 1.0).clamp_min(0.0)
    x_min = torch.zeros_like(x_full)
    y_min = torch.zeros_like(y_full)

    x_full = torch.minimum(torch.maximum(x_full, x_min), x_max)
    y_full = torch.minimum(torch.maximum(y_full, y_min), y_max)

    return torch.stack([x_full, y_full], dim=-1)


def _best_wxyz_from_pnp(q_raw: torch.Tensor, gt_wxyz: torch.Tensor):
    """
    Robustly align PnP quaternion ordering:
    - try treating q_raw as wxyz
    - try treating q_raw as xyzw -> wxyz
    pick the one with higher |dot| to GT; also flip sign to maximize dot.
    """
    q_raw = q_raw.reshape(4)
    gt = gt_wxyz.reshape(4)

    cand1 = _normalize_quat_safe(q_raw)                 # assume wxyz
    cand2 = _normalize_quat_safe(_quat_xyzw_to_wxyz(q_raw))  # assume xyzw

    d1 = torch.abs(torch.dot(cand1, gt))
    d2 = torch.abs(torch.dot(cand2, gt))

    q = cand1 if d1 >= d2 else cand2
    if torch.dot(q, gt) < 0:
        q = -q
    return q


# -------------------------
# Main evaluation
# -------------------------
@torch.no_grad()
def evaluate_model(model, val_loader, eval_cfg, epoch: int):
    device = eval_cfg.get("device", "cuda")
    data_cfg = eval_cfg["data"]

    H_full, W_full = data_cfg.get("original_size", [720, 1280])
    input_size = int(data_cfg.get("input_size", 224))

    cam_path = data_cfg["cam_intrinsics_json"]

    ds_name = data_cfg.get("name", "").lower()
    if ds_name == "spades":
        points_path = data_cfg["root"].rstrip("/") + "/points.json"
    elif ds_name == "spark":
        points_path = data_cfg["root"].rstrip("/") + "/points_spark.json"
    elif ds_name == "vbn":
        points_path = data_cfg.get("object_points_json")
    else:
        points_path = data_cfg["root"].rstrip("/") + "/points.json"

    print(f"[debug] camera_path: {cam_path}", flush=True)
    print(f"[debug] points_path: {points_path}", flush=True)

    pose_estimator, pnp_config, scale = create_pose_estimator(
        pnp_kpts=data_cfg["num_kp"],
        camera_path=cam_path,
        points_path=points_path
    )

    debug_dir = eval_cfg.get("debug_dir", "./eval_debug")
    os.makedirs(debug_dir, exist_ok=True)

    debug_batches = int(eval_cfg.get("debug_batches", 1))
    debug_frames  = int(eval_cfg.get("debug_frames", 2))
    do_debug_viz  = bool(eval_cfg.get("debug_viz", True))
    do_debug_print = bool(eval_cfg.get("debug_print", True))

    tau_now = float(eval_cfg.get("tau_now", eval_cfg.get("tau", 1.0)))
    tau_now = max(tau_now, 1e-4)

    kp_source = str(eval_cfg.get("kp_source", "dsnt")).lower()
    assert kp_source in ("dsnt", "heatmap"), f"kp_source must be dsnt|heatmap, got {kp_source}"

    # Max sub-batch size for GPU-safe forward passes
    max_sub_batch = 16

    # ---- Global micro-averaged PCK accumulators ----
    global_pck_corr1  = 0.0
    global_pck_corr5  = 0.0
    global_pck_corr10 = 0.0
    global_pck_vis    = 0.0

    # SPEED buffers (per-frame values for global micro-average)
    all_Epose, all_Et, all_Eqdeg, all_EtOver, all_EposeStar = [], [], [], [], []

    seq_pose_sum = defaultdict(lambda: {
        "E_pose": 0.0, "E_pose_star": 0.0, "E_t": 0.0,
        "E_q_deg": 0.0, "E_t_over": 0.0, "n_frames": 0
    })

    seq_pck_sum = defaultdict(lambda: {"corr1": 0.0, "corr5": 0.0, "corr10": 0.0, "vis": 0.0})

    seq_valid_sum = defaultdict(lambda: {"valid": 0, "total": 0})
    valid_global = 0
    total_global = 0

    use_speedplus_gate = bool(eval_cfg.get("use_speedplus_gate", False))

    def _extend_finite(dst, arr):
        a = np.asarray(arr).reshape(-1)
        a = a[np.isfinite(a)]
        dst.extend(list(a))

    model.eval()

    global_batch_idx = 0

    for loader_idx, data in enumerate(tqdm(val_loader, desc="Eval", leave=False)):

        # Unpack the full (potentially huge) batch from collate
        video_all, kps_gt_norm_all, quat_gt_xyzw_all, trans_gt_all, vis_all, \
            hm_gt_all, frame_ids_all, crop_xywh_all, geom_all, seq_name_all = data

        if isinstance(seq_name_all, (list, tuple)):
            seq = seq_name_all[0]
        else:
            seq = seq_name_all

        B_total = video_all.shape[0]

        if loader_idx == 0 and is_main_process():
            print(f"[Eval] Collated batch has B={B_total} chunks "
                  f"(will sub-chunk with max_sub_batch={max_sub_batch})")

        # ================================================================
        # Sub-chunk loop: process max_sub_batch chunks at a time
        # ================================================================
        for sub_start in range(0, B_total, max_sub_batch):
            sub_end = min(sub_start + max_sub_batch, B_total)
            idx = global_batch_idx

            video        = video_all[sub_start:sub_end].to(device, non_blocking=True)
            kps_gt_norm  = kps_gt_norm_all[sub_start:sub_end].to(device, non_blocking=True)
            quat_gt_xyzw = quat_gt_xyzw_all[sub_start:sub_end].to(device, non_blocking=True)
            trans_gt     = trans_gt_all[sub_start:sub_end].to(device, non_blocking=True)
            vis          = vis_all[sub_start:sub_end].to(device, non_blocking=True)
            crop_xywh    = crop_xywh_all[sub_start:sub_end].to(device, non_blocking=True)
            geom         = geom_all[sub_start:sub_end].to(device, non_blocking=True)

            # detect geometry type
            use_letterbox = torch.isfinite(geom[..., 0]).all()
            use_letterbox = bool(use_letterbox.item())
            letterbox_meta = geom if use_letterbox else None

            out = _forward_model(model, video)

            hm_logits = out["heatmaps"]
            kps_pred_norm = out.get("keypoints", None)
            B, T, K, Hm, Wm = hm_logits.shape

            if idx == 0 and is_main_process():
                print(f"[Eval] hm_logits: {tuple(hm_logits.shape)}")
                print(f"[Eval] kp_source={kp_source}, tau_now={tau_now}")
                print(f"[Eval] kps_gt_norm: {tuple(kps_gt_norm.shape)} "
                      f"scale={letterbox_meta[0,0,0].item() if use_letterbox else None}")
                if kps_pred_norm is not None:
                    print(f"[Eval] kps_pred_norm: {tuple(kps_pred_norm.shape)}")

            # -------- FULL image pixel coords (pred) --------
            if kp_source == "dsnt":
                if kps_pred_norm is None:
                    raise KeyError("Model output missing 'keypoints' but kp_source='dsnt'.")
                if use_letterbox:
                    pred_kpts_full = _kps_normhm_to_full_px_letterbox(
                        kps_norm_hm=kps_pred_norm, letterbox_meta=letterbox_meta,
                        input_size=input_size, heatmap_hw=(Hm, Wm))
                else:
                    pred_kpts_full = _kps_normhm_to_full_px(
                        kps_norm_hm=kps_pred_norm, crop_xywh_norm=crop_xywh,
                        original_size_wh=(W_full, H_full), input_size=input_size,
                        heatmap_hw=(Hm, Wm))
            else:
                if use_letterbox:
                    pred_kpts_full = _heatmap_logits_to_full_px_letterbox(
                        hm_logits_bt=hm_logits, letterbox_meta=letterbox_meta,
                        input_size=input_size, heatmap_hw=(Hm, Wm), tau_now=tau_now)
                else:
                    pred_kpts_full = _heatmap_logits_to_full_px(
                        hm_logits_bt=hm_logits, crop_xywh_norm=crop_xywh,
                        original_size_wh=(W_full, H_full), input_size=input_size,
                        tau_now=tau_now)

            # -------- FULL image pixel coords (gt) --------
            if use_letterbox:
                gt_kpts_full = _kps_normhm_to_full_px_letterbox(
                    kps_norm_hm=kps_gt_norm, letterbox_meta=letterbox_meta,
                    input_size=input_size, heatmap_hw=(Hm, Wm))
            else:
                gt_kpts_full = _kps_normhm_to_full_px(
                    kps_norm_hm=kps_gt_norm, crop_xywh_norm=crop_xywh,
                    original_size_wh=(W_full, H_full), input_size=input_size,
                    heatmap_hw=(Hm, Wm))

            # -------- Normalized PCK counts (micro-averaged) --------
            corr1,  visn1  = pck_norm_counts_bt(pred_kpts_full, gt_kpts_full, vis, 0.01)
            corr5,  visn5  = pck_norm_counts_bt(pred_kpts_full, gt_kpts_full, vis, 0.05)
            corr10, visn10 = pck_norm_counts_bt(pred_kpts_full, gt_kpts_full, vis, 0.10)

            visn = max(visn1, visn5, visn10)

            # Accumulate into per-sequence counters
            seq_pck_sum[seq]["corr1"]  += corr1
            seq_pck_sum[seq]["corr5"]  += corr5
            seq_pck_sum[seq]["corr10"] += corr10
            seq_pck_sum[seq]["vis"]    += visn

            # Accumulate into global counters for micro-averaged PCK
            global_pck_corr1  += corr1
            global_pck_corr5  += corr5
            global_pck_corr10 += corr10
            global_pck_vis    += visn

            # debug checks
            if idx == 0:
                d_full = torch.norm(pred_kpts_full - gt_kpts_full, dim=-1)
                m = (vis > 0.5)
                print("mean full px err:",
                      (d_full[m].mean().item() if m.any() else None))
                if kps_pred_norm is not None:
                    pred_in = _hm_norm_to_input_px(
                        kps_pred_norm, input_size=input_size, heatmap_hw=(Hm, Wm))
                    gt_in = _hm_norm_to_input_px(
                        kps_gt_norm, input_size=input_size, heatmap_hw=(Hm, Wm))
                    d_in = torch.norm(pred_in - gt_in, dim=-1)
                    print("mean input px err:",
                          (d_in[m].mean().item() if m.any() else None))
                if use_letterbox:
                    s  = letterbox_meta[0, 0, 0].item()
                    pl = letterbox_meta[0, 0, 1].item()
                    pt = letterbox_meta[0, 0, 2].item()
                    Wf = letterbox_meta[0, 0, 3].item()
                    Hf = letterbox_meta[0, 0, 4].item()
                    print(f"[dbg] letterbox_meta: scale={s:.6f}, "
                          f"pad_l={pl:.3f}, pad_t={pt:.3f}, W={Wf}, H={Hf}")

            if kps_pred_norm is not None:
                pck10_in = pck_on_input_px_from_hm_norm(
                    kps_pred_norm, kps_gt_norm, vis, input_size, (Hm, Wm), 10.0
                ).item()
                pck5_in = pck_on_input_px_from_hm_norm(
                    kps_pred_norm, kps_gt_norm, vis, input_size, (Hm, Wm), 5.0
                ).item()
                if idx == 0 and is_main_process():
                    print(f"[dbg] INPUT-PCK@10(px)={pck10_in:.4f} "
                          f"INPUT-PCK@5(px)={pck5_in:.4f}")

            # -------- GT quats --------
            quat_gt_wxyz = _normalize_quat_safe(_quat_xyzw_to_wxyz(quat_gt_xyzw))

            # -------- PnP --------
            kp_scores = out.get("kp_scores", None)
            if kp_scores is None:
                kp_scores = _kp_scores_from_hm_logits(hm_logits)
            else:
                kp_scores = kp_scores.to(device=device)

            pnp_topk = int(eval_cfg.get("pnp_topk", 8))

            q_list, r_list, valid_list = [], [], []
            for b in range(B):
                for t in range(T):
                    mask = (vis[b, t] > 0.5)
                    idx_all = torch.where(mask)[0]

                    if idx_all.numel() < 4:
                        q_list.append(torch.zeros(4, device=device))
                        r_list.append(torch.zeros(3, device=device))
                        valid_list.append(False)
                        continue

                    scores_bt = kp_scores[b, t]
                    idx_sorted = idx_all[
                        torch.argsort(scores_bt[idx_all], descending=True)
                    ]
                    vis_idx = idx_sorted[:min(pnp_topk, idx_sorted.numel())]

                    if vis_idx.numel() < 4:
                        q_list.append(torch.zeros(4, device=device))
                        r_list.append(torch.zeros(3, device=device))
                        valid_list.append(False)
                        continue

                    q, r, ok_list_pnp = estimate_pose_from_keypoints(
                        pred_kpts_full[b, t].unsqueeze(0),
                        vis_idx.unsqueeze(0),
                        pose_estimator,
                        pnp_config
                    )
                    ok = bool(ok_list_pnp[0])

                    if not ok:
                        q_list.append(torch.zeros(4, device=device))
                        r_list.append(torch.zeros(3, device=device))
                        valid_list.append(False)
                        continue

                    # ---- FIX: deterministic scipy (x,y,z,w) → (w,x,y,z) ----
                    q_raw_xyzw = torch.tensor(
                        q[0], device=device, dtype=torch.float32
                    )  # scipy convention: (x, y, z, w)
                    q_wxyz = torch.stack([
                        q_raw_xyzw[3], q_raw_xyzw[0],
                        q_raw_xyzw[1], q_raw_xyzw[2]
                    ])  # → (w, x, y, z)
                    q_wxyz = _normalize_quat_safe(q_wxyz)
                    # Standard hemisphere alignment (quaternion double-cover):
                    if torch.dot(q_wxyz, quat_gt_wxyz[b, t]) < 0:
                        q_wxyz = -q_wxyz

                    # ---- FIX: single append (was duplicated before) ----
                    q_list.append(q_wxyz)
                    r_list.append(
                        torch.tensor(r[0], device=device, dtype=torch.float32)
                    )
                    valid_list.append(True)

            q_pred = torch.stack(q_list).view(B, T, 4)
            r_pred = torch.stack(r_list).view(B, T, 3)
            q_pred = _normalize_quat_safe(q_pred)
            r_pred = torch.nan_to_num(r_pred, nan=0.0, posinf=0.0, neginf=0.0)

            valid = np.array(valid_list, dtype=bool).reshape(B, T)

            v = int(valid.sum())
            n = int(valid.size)
            valid_global += v
            total_global += n
            seq_valid_sum[seq]["valid"] += v
            seq_valid_sum[seq]["total"] += n

            # -------- Debug viz --------
            if is_main_process() and do_debug_viz and idx < debug_batches:
                viz_root = os.path.join(debug_dir, f"epoch_{epoch:04d}")
                os.makedirs(viz_root, exist_ok=True)
                b0 = 0
                for t0 in range(min(T, debug_frames)):
                    kp_path = os.path.join(viz_root, f"b{idx}_t{t0}_kp.png")
                    hm_path = os.path.join(viz_root, f"b{idx}_t{t0}_hm.png")
                    save_kp_overlay_bt(
                        frames=video, kps_gt_norm=kps_gt_norm,
                        hm_logits=hm_logits, vis=vis,
                        cfg={"data": {"input_size": input_size}},
                        save_path=kp_path, b=b0, t=t0)
                    try:
                        save_heatmap_overlay_bt(
                            frames=video, hm_logits=hm_logits, vis=vis,
                            cfg={"data": {"input_size": input_size}},
                            save_path=hm_path, b=b0, t=t0,
                            reduce="max", use_vis=True, tau=1.0)
                    except Exception:
                        pass

            # -------- Pose metrics --------
            E_q_rad, E_q_deg, E_t, E_t_over, E_pose, E_pose_star = \
                compute_speed_metrics_batch(
                    q_gt_wxyz=quat_gt_wxyz, t_gt=trans_gt,
                    q_pred_wxyz=q_pred, t_pred=r_pred,
                    use_speedplus_gate=use_speedplus_gate)

            E_pose      = np.where(valid, E_pose, np.nan)
            E_pose_star = np.where(valid, E_pose_star, np.nan)
            E_t         = np.where(valid, E_t, np.nan)
            E_q_deg     = np.where(valid, E_q_deg, np.nan)
            E_t_over    = np.where(valid, E_t_over, np.nan)

            _extend_finite(all_Epose, E_pose)
            _extend_finite(all_EposeStar, E_pose_star)
            _extend_finite(all_Et, E_t)
            _extend_finite(all_Eqdeg, E_q_deg)
            _extend_finite(all_EtOver, E_t_over)

            epose_flat = np.asarray(E_pose).reshape(-1)
            et_flat = np.asarray(E_t).reshape(-1)
            eqdeg_flat = np.asarray(E_q_deg).reshape(-1)
            etover_flat = np.asarray(E_t_over).reshape(-1)
            epose_star_flat = np.asarray(E_pose_star).reshape(-1)

            n_valid_frames = int(np.isfinite(epose_flat).sum())
            seq_pose_sum[seq]["E_pose"]      += float(np.nansum(epose_flat))
            seq_pose_sum[seq]["E_pose_star"]  += float(np.nansum(epose_star_flat))
            seq_pose_sum[seq]["E_t"]          += float(np.nansum(et_flat))
            seq_pose_sum[seq]["E_q_deg"]      += float(np.nansum(eqdeg_flat))
            seq_pose_sum[seq]["E_t_over"]     += float(np.nansum(etover_flat))
            seq_pose_sum[seq]["n_frames"]     += n_valid_frames

            if is_main_process() and do_debug_print and idx < debug_batches:
                # Compute sub-batch micro-averaged PCK for debug print
                _dbg_pck10 = corr10 / max(1.0, visn)
                _dbg_pck5  = corr5  / max(1.0, visn)
                _dbg_pck1  = corr1  / max(1.0, visn)
                print(f"[Eval dbg] nPCK@10={_dbg_pck10:.4f} "
                      f"nPCK@5={_dbg_pck5:.4f} nPCK@1={_dbg_pck1:.4f}")
                print(f"[Eval dbg] {ds_name}(valid): "
                      f"E_pose mean={float(np.nanmean(epose_flat)):.4f} | "
                      f"E_t mean={float(np.nanmean(et_flat)):.4f} | "
                      f"E_q(deg) mean={float(np.nanmean(eqdeg_flat)):.4f}")
                print("GT t norm mean:", trans_gt.norm(dim=-1).mean().item())
                print("PNP t norm mean:", r_pred.norm(dim=-1).mean().item())
                print("GT t example:", trans_gt[0, 0].detach().cpu().numpy())
                print("PNP t example:", r_pred[0, 0].detach().cpu().numpy())
                print("scale:", scale)

            # Free sub-chunk GPU tensors
            del video, kps_gt_norm, quat_gt_xyzw, trans_gt, vis, crop_xywh, geom
            del hm_logits, kps_pred_norm, out, pred_kpts_full, gt_kpts_full
            del q_pred, r_pred, kp_scores
            if letterbox_meta is not None:
                del letterbox_meta
            torch.cuda.empty_cache()

            global_batch_idx += 1

    # -------------------------
    # Aggregate results
    # -------------------------
    def _mean(x):
        return float(np.mean(x)) if len(x) else float("inf")

    # ---- FIX: global PCK via micro-averaging (counts), not macro-averaging ----
    avg_pck_1  = global_pck_corr1  / max(1.0, global_pck_vis)
    avg_pck_5  = global_pck_corr5  / max(1.0, global_pck_vis)
    avg_pck_10 = global_pck_corr10 / max(1.0, global_pck_vis)

    avg_Epose     = _mean(all_Epose)
    avg_Et        = _mean(all_Et)
    avg_Eqdeg     = _mean(all_Eqdeg)
    avg_EtOver    = _mean(all_EtOver)
    avg_EposeStar = _mean(all_EposeStar) if len(all_EposeStar) else avg_Epose

    def _pct(a, b):
        return 100.0 * float(a) / max(1.0, float(b))

    overall_valid_pct = _pct(valid_global, total_global)

    if is_main_process():
        print(f"[Eval] PnP valid frames: {valid_global}/{total_global} "
              f"= {overall_valid_pct:.2f}%")
        seq_items = sorted(
            seq_valid_sum.items(),
            key=lambda kv: _pct(kv[1]["valid"], kv[1]["total"])
        )
        for s, d in seq_items:
            vp = _pct(d["valid"], d["total"])
            print(f"[Eval]   {s}: {d['valid']}/{d['total']} = {vp:.2f}%")

    # -------------------------
    # Write per-sequence CSV
    # -------------------------
    csv_path = eval_cfg.get("save_per_sequence_csv", None)
    if csv_path is not None:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        rows = []
        seq_names = sorted(
            set(list(seq_pose_sum.keys()) + list(seq_pck_sum.keys()))
        )

        for s in seq_names:
            n_frames = int(seq_pose_sum[s]["n_frames"])
            epose      = seq_pose_sum[s]["E_pose"]      / max(1, n_frames)
            epose_star = seq_pose_sum[s]["E_pose_star"]  / max(1, n_frames)
            et         = seq_pose_sum[s]["E_t"]          / max(1, n_frames)
            eqdeg      = seq_pose_sum[s]["E_q_deg"]      / max(1, n_frames)
            etover     = seq_pose_sum[s]["E_t_over"]     / max(1, n_frames)

            visn  = float(seq_pck_sum[s]["vis"])
            pck1  = float(seq_pck_sum[s]["corr1"])  / max(1.0, visn)
            pck5  = float(seq_pck_sum[s]["corr5"])  / max(1.0, visn)
            pck10 = float(seq_pck_sum[s]["corr10"]) / max(1.0, visn)

            d = seq_valid_sum[s]
            valid_pct = 100.0 * float(d["valid"]) / max(1.0, float(d["total"]))

            rows.append({
                "Seq_name": s,
                "Valid_%": valid_pct,
                "Total_frames": int(d["total"]),
                "E_Pose": epose_star if use_speedplus_gate else epose,
                "E_T": et,
                "E_R_deg": eqdeg,
                "PCK@1": pck1,
                "PCK@5": pck5,
                "PCK@10": pck10,
            })

        # Avg row uses micro-averaged values
        rows.append({
            "Seq_name": "Avg",
            "Valid_%": overall_valid_pct,
            "Total_frames": int(total_global),
            "E_Pose": avg_EposeStar if use_speedplus_gate else avg_Epose,
            "E_T": avg_Et,
            "E_R_deg": avg_Eqdeg,
            "PCK@1": avg_pck_1,
            "PCK@5": avg_pck_5,
            "PCK@10": avg_pck_10,
        })

        float_cols = [
            "Valid_%", "Total_frames",
            "E_Pose", "E_T", "E_R_deg",
            "PCK@1", "PCK@5", "PCK@10"
        ]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Seq_name", "Total_frames",
                    "Valid_%",
                    "E_Pose", "E_T", "E_R_deg",
                    "PCK@1", "PCK@5", "PCK@10"
                ]
            )
            writer.writeheader()
            writer.writerows(
                [_format_row_4dp(r, float_keys=float_cols) for r in rows]
            )

        if is_main_process():
            print(f"[Eval] Saved per-sequence results CSV -> {csv_path}")

    avg_pose_out = avg_EposeStar if use_speedplus_gate else avg_Epose
    return avg_pose_out, avg_Et, avg_Eqdeg, avg_EtOver, avg_pck_10, avg_pck_5, avg_pck_1
# -------------------------
# Standalone runner
# -------------------------

def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg.setdefault("train", {})
    cfg.setdefault("loss", {})
    cfg.setdefault("eval", {})
    return cfg

def make_dataloaders(cfg):
    ds_name = cfg["data"].get("name", "").lower()
    mode = cfg["eval"].get("mode", "test")

    base_kwargs = dict(
        gt_folder=cfg["data"]["root"],
        short_mem=int(cfg["data"]["short_mem"]),
        input_size=int(cfg["data"]["input_size"]),
        sampling_rate=int(cfg["data"]["sampling_rate"]),
        clip_stride=int(cfg["data"].get("clip_stride", 0)),
        crop=bool(cfg["data"].get("crop", True)),
        eval_use_crop=cfg["data"].get("eval_use_crop", None),
        gaussian_sigma_px=float(cfg["data"].get("gaussian_sigma_px", 2.0)),
        heatmap_out=tuple(cfg["model"].get("heatmap_out", [56, 56])),
        spatial_transforms=None,
    )

    print (f"[debug] mode is {mode}", flush=True)
    if ds_name == "spades":
        test_set = SPADESDataset(mode=mode, **base_kwargs)
    elif ds_name == "spark":
        test_set = SPARKDataset(mode=mode, **base_kwargs)
    else:
        raise ValueError("data.name must be 'spades' or 'spark'")

    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_temporal_chunks(b, int(cfg["data"]["test_T_chunk"]))
    )
    return test_loader

def build_model_for_eval(cfg: dict, ckpt: dict | None = None):
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    T = int(data_cfg["test_T_chunk"])
    input_size = int(data_cfg["input_size"])
    num_kp = int(data_cfg["num_kp"])

    patch_size = int(model_cfg.get("patch_size", 16))
    Hp = input_size // patch_size
    Wp = input_size // patch_size
    grid_hw = (Hp, Wp)

    heatmap_hw = tuple(model_cfg.get("heatmap_out", [256, 256]))
    hm_mid_c = model_cfg.get("hm_mid_c", 128)
    hf_name = model_cfg.get("hf_name", "facebook/dinov3-vitl16-pretrain-lvd1689m")

    adapter_bottleneck = int(model_cfg.get("adapter_bottleneck", 64))
    adapter_drop = float(model_cfg.get("adapter_drop", 0.1))
    adapter_in_last = model_cfg.get("adapter_in_last", None)
    if adapter_in_last is not None:
        adapter_in_last = int(adapter_in_last)

    fuse = model_cfg.get("fuse", "sum")

    n_register_tokens = model_cfg.get("n_register_tokens", None)
    if n_register_tokens is not None:
        n_register_tokens = int(n_register_tokens)

    model = SpacecraftHMDSNT(
        num_kp=num_kp,
        num_frames=T,
        img_hw=input_size,
        hf_name=hf_name,
        adapter_bottleneck=adapter_bottleneck,
        adapter_drop=adapter_drop,
        adapter_in_last=adapter_in_last,
        heatmap_hw=heatmap_hw,
        grid_hw=grid_hw,
        n_register_tokens=n_register_tokens,
        ks=3, kt=3,
        fuse=fuse,
        hm_mid_c=hm_mid_c,
    )
    return model

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_cfg("./test_config.yaml")

    ckpt_path = cfg["eval"].get("load_from", None)
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint path invalid: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    model = build_model_for_eval(cfg, ckpt=ckpt)

    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"[Eval] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if len(unexpected) > 0:
        print("  unexpected keys (first 20):", unexpected[:20])
    if len(missing) > 0:
        print("  missing keys (first 20):", missing[:20])

    model.to(device).eval()
    test_loader = make_dataloaders(cfg)

    eval_cfg = cfg.get("eval", {})
    eval_cfg["device"] = device
    eval_cfg["data"] = cfg["data"]
    eval_cfg["tau_now"] = float(cfg.get("loss", {}).get("tau", 0.65))

    eval_cfg["save_per_sequence_csv"] = os.path.join(
        cfg["eval"].get("save_results_to", "./eval_results"),
        cfg["eval"].get("model_type", "mhsa_mlp") + "_" + ckpt_path.split("/")[-1].rstrip(".pt") + "_per_sequence.csv"
    )

    eval_cfg["debug_print"] = True
    eval_cfg["debug_viz"]   = True
    eval_cfg["debug_dir"]   = cfg["eval"].get("debug_dir", "./eval_debug")
    eval_cfg["debug_batches"] = 2
    eval_cfg["debug_frames"]  = 4

    print(f"checking kp_source={eval_cfg.get('kp_source', 'dsnt')}")

    avg_pose, avg_Et, avg_Eqdeg, avg_EtOver, avg_pck_10, avg_pck_5, avg_pck_1 = evaluate_model(
        model, test_loader, eval_cfg, epoch=int(ckpt.get("epoch", 0))
    )

    print(
        f"Eval Results - SPEED E_Pose: {avg_pose:.4f}, E_T: {avg_Et:.4f}, E_R(deg): {avg_Eqdeg:.4f}, "
        f"E_t/||t||: {avg_EtOver:.6f}, PCK@10: {avg_pck_10:.4f}, PCK@5: {avg_pck_5:.4f}, PCK@1: {avg_pck_1:.4f}"
    )