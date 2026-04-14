# evaluate.py (FINAL — aligned with SpacecraftHMDSNT + dataset.py + utils_new + misc.py)
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
import json

from utils_new import (
    collate_temporal_chunks,
    save_kp_overlay_bt,
    save_heatmap_overlay_bt,   # NEW
    _format_row_4dp,
)

from misc import (
    create_pose_estimator,
    estimate_pose_from_keypoints,
    compute_speed_metrics_batch
)

from losses import softargmax_2d_from_logits
from torch.utils.data import DataLoader
from dataset import SPADESDataset, SPARKDataset
from model_hm_dsnt import SpacecraftHMDSNT  # change import path if your file name differs
import csv
from collections import defaultdict
# -------------------------
# Helpers
# -------------------------

# def pck_on_crop_norm(kps_pred_norm, kps_gt_norm, vis, input_size, thresh_px):
#     # kps_*: (B,T,K,2) in [-1,1]
#     den = float(max(input_size - 1, 1))
#     pred_px = (kps_pred_norm + 1) * 0.5 * den
#     gt_px   = (kps_gt_norm   + 1) * 0.5 * den
#     d = torch.norm(pred_px - gt_px, dim=-1)  # (B,T,K)
#     m = (vis > 0.5)
#     return ((d < thresh_px) & m).float().sum() / m.float().sum().clamp_min(1)

def pck_on_input_px_from_hm_norm(kps_pred_norm, kps_gt_norm, vis, input_size, heatmap_hw, thresh_px):
    pred_in = _hm_norm_to_input_px(kps_pred_norm, input_size=input_size, heatmap_hw=heatmap_hw)
    gt_in   = _hm_norm_to_input_px(kps_gt_norm,   input_size=input_size, heatmap_hw=heatmap_hw)
    d = torch.norm(pred_in - gt_in, dim=-1)
    m = (vis > 0.5)
    return ((d < thresh_px) & m).float().sum() / m.float().sum().clamp_min(1)

def is_main_process():
    # local eval is single-process by default; safe if later DDP-eval
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

    # crop box normalized -> absolute full image
    x0 = crop_xywh_norm[..., 0] * W_full
    y0 = crop_xywh_norm[..., 1] * H_full
    w  = crop_xywh_norm[..., 2] * W_full
    h  = crop_xywh_norm[..., 3] * H_full

    x0 = x0.unsqueeze(-1); y0 = y0.unsqueeze(-1)
    w  = w.unsqueeze(-1);  h  = h.unsqueeze(-1)

    # IMPORTANT: crop_px is in [0..input_size-1], so divide by (input_size-1)
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
    Ablation path: softargmax on heatmap logits -> full px.
    Matches training softargmax convention (tau).
    """
    B, T, K, Hm, Wm = hm_logits_bt.shape
    kps_hm_px = softargmax_2d_from_logits(hm_logits_bt.float(), tau=tau_now)  # (B,T,K,2) in hm px
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

def _pck_counts_bt(pred_full, gt_full, vis, thresh_px: float):
    d = torch.norm(pred_full - gt_full, dim=-1)  # (B,T,K)
    vis_m = (vis > 0.5)
    corr = ((d < thresh_px) & vis_m).float().sum().item()
    visn = vis_m.float().sum().item()
    return corr, visn

def _pck_bt(pred_full, gt_full, vis, thresh_px: float):
    """
    Sequence-level PCK across all (T,K) visible points for B=1 (and also works for B>1).
    """
    d = torch.norm(pred_full - gt_full, dim=-1)   # (B,T,K)
    vis_m = (vis > 0.5)
    correct = (d < thresh_px) & vis_m
    denom = vis_m.sum().clamp_min(1)
    return (correct.float().sum() / denom).item()

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
    # resolve sign ambiguity to match GT
    if torch.dot(q, gt) < 0:
        q = -q
    return q

# ------------------------------
# Helpsers for Full frame inputs 
# ------------------------------

def _hm_norm_to_input_px(kps_norm_hm: torch.Tensor, input_size: int, heatmap_hw):
    """
    kps_norm_hm: (...,2) in [-1,1] w.r.t heatmap grid
    returns: (...,2) in input image pixel coords [0..input_size-1]
    """
    Hm, Wm = heatmap_hw
    # norm -> hm px
    kps_hm_px = _norm_hm_to_hm_px(kps_norm_hm, heatmap_hw=(Hm, Wm))
    # hm px -> input px (inverse of dataset scaling)
    kps_in_px = _hm_px_to_crop_px(kps_hm_px, input_size=input_size, heatmap_hw=(Hm, Wm))
    return kps_in_px


def _kps_normhm_to_full_px_letterbox(kps_norm_hm, letterbox_meta, input_size, heatmap_hw):
    kps_in_px = _hm_norm_to_input_px(kps_norm_hm, input_size=input_size, heatmap_hw=heatmap_hw)

    scale = letterbox_meta[..., 0]  # (B,T)
    pad_l = letterbox_meta[..., 1]
    pad_t = letterbox_meta[..., 2]
    Wfull = letterbox_meta[..., 3]
    Hfull = letterbox_meta[..., 4]

    # (B,T,K)
    x_unpad = kps_in_px[..., 0] - pad_l[..., None]
    y_unpad = kps_in_px[..., 1] - pad_t[..., None]

    x_full = x_unpad / scale[..., None].clamp_min(1e-8)
    y_full = y_unpad / scale[..., None].clamp_min(1e-8)

    x_max = (Wfull[..., None] - 1.0)
    y_max = (Hfull[..., None] - 1.0)

    x_full = torch.maximum(x_full, torch.zeros_like(x_full))
    y_full = torch.maximum(y_full, torch.zeros_like(y_full))

    x_full = torch.minimum(x_full, x_max)
    y_full = torch.minimum(y_full, y_max)

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

    # (B,T,K,2) hm pixel coords
    kps_hm_px = softargmax_2d_from_logits(hm_logits_bt.float(), tau=tau_now)

    # hm px -> input px
    kps_in_px = _hm_px_to_crop_px(kps_hm_px, input_size=input_size, heatmap_hw=(Hm, Wm))

    scale = letterbox_meta[..., 0].unsqueeze(-1).unsqueeze(-1)
    pad_l = letterbox_meta[..., 1].unsqueeze(-1).unsqueeze(-1)
    pad_t = letterbox_meta[..., 2].unsqueeze(-1).unsqueeze(-1)
    Wfull = letterbox_meta[..., 3].unsqueeze(-1).unsqueeze(-1)
    Hfull = letterbox_meta[..., 4].unsqueeze(-1).unsqueeze(-1)

    x_unpad = kps_in_px[..., 0] - pad_l.squeeze(-1)
    y_unpad = kps_in_px[..., 1] - pad_t.squeeze(-1)

    s = scale.squeeze(-1).squeeze(-1).clamp_min(1e-8)
    x_full = x_unpad / s.unsqueeze(-1)
    y_full = y_unpad / s.unsqueeze(-1)

    x_max = (Wfull[..., None] - 1.0)
    y_max = (Hfull[..., None] - 1.0)

    x_full = torch.maximum(x_full, torch.zeros_like(x_full))
    y_full = torch.maximum(y_full, torch.zeros_like(y_full))
    x_full = torch.minimum(x_full, x_max)
    y_full = torch.minimum(y_full, y_max)

    return torch.stack([x_full, y_full], dim=-1)

# -------------------------
# Main evaluation
# -------------------------

@torch.no_grad()
def evaluate_model(model, val_loader, eval_cfg, epoch: int):
    device = eval_cfg.get("device", "cuda")
    data_cfg = eval_cfg["data"]

    H_full, W_full = data_cfg.get("original_size", [720, 1280]) # H, W
    input_size = int(data_cfg.get("input_size", 224))

    cam_path = data_cfg["cam_intrinsics_json"]
    points_path = data_cfg["root"].rstrip("/") + "/points.json"
    pose_estimator, pnp_config, scale = create_pose_estimator(
        pnp_kpts=data_cfg["num_kp"],
        camera_path=cam_path,
        points_path=points_path
    )

    max_eval_batches = int(eval_cfg.get("max_eval_batches", 0) or 0)
    pck_thresh_5, pck_thresh_10 = eval_cfg.get("pck_thresh_px", [5, 10])

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

    # all_ori, all_loc, all_range, all_esa = [], [], [], []
    pck10_vals, pck5_vals = [], []

    all_Epose, all_Et, all_Eqdeg, all_EtOver = [], [], [], []
    all_EposeStar = []

    model.eval()

    # -------------------------
    # Per-sequence metrics accumulators
    # -------------------------
    # Pose metrics: accumulate sums over frames, plus frame counts
    seq_pose_sum = defaultdict(lambda: {"esa": 0.0, "loc": 0.0, "ori": 0.0, "range": 0.0, "n_frames": 0})

    # PCK: accumulate counts of correct/visible keypoints (more correct than averaging chunk-PCK)
    seq_pck_sum = defaultdict(lambda: {"corr5": 0.0, "corr10": 0.0, "vis": 0.0})

    for idx, data in enumerate(tqdm(val_loader, desc="Eval", leave=False)):
        if 0 < max_eval_batches <= idx:
            break

        # dataset returns:
        # frames, kps_norm, gt_q, gt_r, vis, hm, delta_Rt_gt, frame_ids, crop_xywh
        # video, kps_gt_norm, quat_gt_xyzw, trans_gt, vis, hm_gt, delta_Rt, frame_ids, crop_xywh, meta_data = data

        # video = video.to(device, non_blocking=True)
        # kps_gt_norm = kps_gt_norm.to(device, non_blocking=True)
        # quat_gt_xyzw = quat_gt_xyzw.to(device, non_blocking=True)
        # trans_gt = trans_gt.to(device, non_blocking=True)
        # vis = vis.to(device, non_blocking=True)
        # crop_xywh = crop_xywh.to(device, non_blocking=True)

        # frames, kps_norm, gt_q, gt_r, vis, hm, delta_Rt_gt, frame_ids, (crop_xywh OR letterbox_meta)
        video, kps_gt_norm, quat_gt_xyzw, trans_gt, vis, hm_gt, delta_Rt, frame_ids, crop_xywh, geom, seq_name = data

        # seq_name: list[str] length B OR str
        if isinstance(seq_name, (list, tuple)):
            seq = seq_name[0]
            assert all(s == seq for s in seq_name), f"Mixed seq batch: {seq_name}"
        else:
            seq = seq_name

        video = video.to(device, non_blocking=True)
        kps_gt_norm = kps_gt_norm.to(device, non_blocking=True)
        quat_gt_xyzw = quat_gt_xyzw.to(device, non_blocking=True)
        trans_gt = trans_gt.to(device, non_blocking=True)
        vis = vis.to(device, non_blocking=True)
        crop_xywh = crop_xywh.to(device, non_blocking=True)

        geom = geom.to(device, non_blocking=True)

        # detect geometry type
        # use_letterbox = (geom.shape[-1] == 5)  # (B,T,5) => letterbox meta
        # crop_xywh = geom if (not use_letterbox) else None
        use_letterbox = torch.isfinite(geom[..., 0]).all()
        # if you ever mix modes in a batch, you must handle per-sample; but batch_size=1 is OK.
        use_letterbox = bool(use_letterbox.item())
        letterbox_meta = geom if use_letterbox else None

        out = _forward_model(model, video)

        hm_logits = out["heatmaps"]                    # (B,T,K,Hm,Wm)
        kps_pred_norm = out.get("keypoints", None)     # (B,T,K,2) in [-1,1] (heatmap norm)
        B, T, K, Hm, Wm = hm_logits.shape

        if idx == 0 and is_main_process():
            print(f"[Eval] hm_logits: {tuple(hm_logits.shape)}")
            print(f"[Eval] kp_source={kp_source}, tau_now={tau_now}")
            print(f"[Eval] kps_gt_norm: {tuple(kps_gt_norm.shape)} scale={letterbox_meta[0,0,0].item() if use_letterbox else None}")
            if kps_pred_norm is not None:
                print(f"[Eval] kps_pred_norm: {tuple(kps_pred_norm.shape)}")

        # -------- FULL image pixel coords (pred) --------
        # if kp_source == "dsnt":
        #     if kps_pred_norm is None:
        #         raise KeyError("Model output missing 'keypoints' but kp_source='dsnt'.")
        #     pred_kpts_full = _kps_normhm_to_full_px(
        #         kps_norm_hm=kps_pred_norm,
        #         crop_xywh_norm=crop_xywh,
        #         original_size_wh=(W_full, H_full),
        #         input_size=input_size,
        #         heatmap_hw=(Hm, Wm),
        #     )
        # else:
        #     pred_kpts_full = _heatmap_logits_to_full_px(
        #         hm_logits_bt=hm_logits,
        #         crop_xywh_norm=crop_xywh,
        #         original_size_wh=(W_full, H_full),
        #         input_size=input_size,
        #         tau_now=tau_now,
        #     )

        # # -------- FULL image pixel coords (gt) --------
        # gt_kpts_full = _kps_normhm_to_full_px(
        #     kps_norm_hm=kps_gt_norm,
        #     crop_xywh_norm=crop_xywh,
        #     original_size_wh=(W_full, H_full),
        #     input_size=input_size,
        #     heatmap_hw=(Hm, Wm),
        # )
        # -------- FULL image pixel coords (pred) --------
        if kp_source == "dsnt":
            if kps_pred_norm is None:
                raise KeyError("Model output missing 'keypoints' but kp_source='dsnt'.")

            if use_letterbox:
                pred_kpts_full = _kps_normhm_to_full_px_letterbox(
                    kps_norm_hm=kps_pred_norm,
                    letterbox_meta=letterbox_meta,
                    input_size=input_size,
                    heatmap_hw=(Hm, Wm),
                )
            else:
                pred_kpts_full = _kps_normhm_to_full_px(
                    kps_norm_hm=kps_pred_norm,
                    crop_xywh_norm=crop_xywh,
                    original_size_wh=(W_full, H_full),
                    input_size=input_size,
                    heatmap_hw=(Hm, Wm),
                )
        else:
            # heatmap-softargmax ablation
            if use_letterbox:
                pred_kpts_full = _heatmap_logits_to_full_px_letterbox(
                    hm_logits_bt=hm_logits,
                    letterbox_meta=letterbox_meta,
                    input_size=input_size,
                    heatmap_hw=(Hm, Wm),
                    tau_now=tau_now,
                )
            else:
                pred_kpts_full = _heatmap_logits_to_full_px(
                    hm_logits_bt=hm_logits,
                    crop_xywh_norm=crop_xywh,
                    original_size_wh=(W_full, H_full),
                    input_size=input_size,
                    tau_now=tau_now,
                )

        # -------- FULL image pixel coords (gt) --------
        if use_letterbox:
            gt_kpts_full = _kps_normhm_to_full_px_letterbox(
                kps_norm_hm=kps_gt_norm,
                letterbox_meta=letterbox_meta,
                input_size=input_size,
                heatmap_hw=(Hm, Wm),
            )
        else:
            gt_kpts_full = _kps_normhm_to_full_px(
                kps_norm_hm=kps_gt_norm,
                crop_xywh_norm=crop_xywh,
                original_size_wh=(W_full, H_full),
                input_size=input_size,
                heatmap_hw=(Hm, Wm),
            )

        # -------- PCK counts (per-sequence) --------
        corr10, visn10 = _pck_counts_bt(pred_kpts_full, gt_kpts_full, vis, float(pck_thresh_10))
        corr5,  visn5  = _pck_counts_bt(pred_kpts_full, gt_kpts_full, vis, float(pck_thresh_5))

        # visn10 and visn5 should be identical (same vis mask), but keep robust
        seq_pck_sum[seq]["corr10"] += corr10
        seq_pck_sum[seq]["corr5"]  += corr5
        seq_pck_sum[seq]["vis"]    += max(visn10, visn5)

        # debugging 
        if idx == 0:
            # mean pixel error in full frame
            d_full = torch.norm(pred_kpts_full - gt_kpts_full, dim=-1)  # (B,T,K)
            m = (vis > 0.5)
            print("mean full px err:", (d_full[m].mean().item() if m.any() else None))

            # mean pixel error in input space
            pred_in = _hm_norm_to_input_px(kps_pred_norm, input_size=input_size, heatmap_hw=(Hm, Wm))
            gt_in   = _hm_norm_to_input_px(kps_gt_norm,   input_size=input_size, heatmap_hw=(Hm, Wm))
            d_in = torch.norm(pred_in - gt_in, dim=-1)
            print("mean input px err:", (d_in[m].mean().item() if m.any() else None))

            if idx == 0 and use_letterbox:
                s  = letterbox_meta[0,0,0].item()
                pl = letterbox_meta[0,0,1].item()
                pt = letterbox_meta[0,0,2].item()
                Wf = letterbox_meta[0,0,3].item()
                Hf = letterbox_meta[0,0,4].item()
                print(f"[dbg] letterbox_meta: scale={s:.6f}, pad_l={pl:.3f}, pad_t={pt:.3f}, W={Wf}, H={Hf}")

            if use_letterbox:
                s = float(letterbox_meta[0,0,0].item())
                print("approx ratio full/input:", (1.0 / s))

        # in eval loop, after you have kps_pred_norm/kps_gt_norm/vis:
        # ----- crop/input-space PCK (debug only) -----
        if kps_pred_norm is not None:
            pck10_in = pck_on_input_px_from_hm_norm(kps_pred_norm, kps_gt_norm, vis, input_size, (Hm, Wm), 10.0).item()
            pck5_in  = pck_on_input_px_from_hm_norm(kps_pred_norm, kps_gt_norm, vis, input_size, (Hm, Wm), 5.0).item()
            if idx == 0 and is_main_process():
                print(f"[dbg] INPUT-PCK@10={pck10_in:.4f} INPUT-PCK@5={pck5_in:.4f}")

        # -------- PCK (sequence-level) --------
        pck10_vals.append(_pck_bt(pred_kpts_full, gt_kpts_full, vis, float(pck_thresh_10)))
        pck5_vals.append(_pck_bt(pred_kpts_full, gt_kpts_full, vis, float(pck_thresh_5)))

        # -------- Prepare GT quats (wxyz) for robust PnP quat alignment --------
        quat_gt_wxyz = _normalize_quat_safe(_quat_xyzw_to_wxyz(quat_gt_xyzw))  # (B,T,4)

        # -------- PnP pose per frame --------
        q_list, r_list = [], []
        for b in range(B):
            for t in range(T):
                vis_idx = (vis[b, t] > 0.5).nonzero(as_tuple=False).squeeze(1)
                if vis_idx.numel() < 4:
                    q_list.append(torch.zeros(4, device=device))
                    r_list.append(torch.zeros(3, device=device))
                    continue

                q, r, _ = estimate_pose_from_keypoints(
                    pred_kpts_full[b, t].unsqueeze(0),   # (1,K,2) full px
                    vis_idx.unsqueeze(0),                # (1,n_vis)
                    pose_estimator,
                    pnp_config
                )

                q_raw = torch.tensor(q[0], device=device, dtype=torch.float32)
                q_wxyz = _best_wxyz_from_pnp(q_raw, quat_gt_wxyz[b, t])
                q_list.append(q_wxyz)

                r_list.append(torch.tensor(r[0], device=device, dtype=torch.float32))

        q_pred = torch.stack(q_list).view(B, T, 4)
        r_pred = torch.stack(r_list).view(B, T, 3)
        q_pred = _normalize_quat_safe(q_pred)
        r_pred = torch.nan_to_num(r_pred, nan=0.0, posinf=0.0, neginf=0.0)



        # -------- Debug viz --------
        if is_main_process() and do_debug_viz and idx < debug_batches:
            viz_root = os.path.join(debug_dir, f"epoch_{epoch:04d}")
            os.makedirs(viz_root, exist_ok=True)

            b0 = 0
            for t0 in range(min(T, debug_frames)):
                kp_path = os.path.join(viz_root, f"b{idx}_t{t0}_kp.png")
                # tok_path = os.path.join(viz_root, f"b{idx}_t{t0}_tok.png")
                hm_path = os.path.join(viz_root, f"b{idx}_t{t0}_hm.png")



                save_kp_overlay_bt(
                    frames=video,
                    kps_gt_norm=kps_gt_norm,
                    hm_logits=hm_logits,
                    vis=vis,
                    cfg={"data": {"input_size": input_size}},
                    save_path=kp_path,
                    b=b0,
                    t=t0,
                )
                try:
                    save_heatmap_overlay_bt(
                    frames=video,
                    hm_logits=hm_logits,
                    vis=vis,
                    cfg={"data": {"input_size": input_size}},
                    save_path=hm_path,
                    b=b0,
                    t=t0,
                    reduce="max",      # try "sum" too
                    use_vis=True,
                    tau=1.0,           # purely for viz; not your DSNT tau
                )
                    # save_token_similarity_overlay_bt(
                    #     out=out, frames=video, kps_gt=kps_gt_norm, vis=vis,
                    #     cfg={"data": {"input_size": input_size}},
                    #     save_path=tok_path, b=b0, t=t0,
                    #     sigma_patch=1.25,
                    # )
                except Exception:
                    pass

        # -------- Pose metrics --------
        # batch_for_scores = {
        #     "quats": quat_gt_wxyz,  # (B,T,4) wxyz
        #     "rotations": trans_gt,  # (B,T,3) translation (legacy naming)
        # }

        # ori_deg, loc_err, range_error, esa_score = compute_pose_scores_batch(
        #     batch_for_scores, r_pred, q_pred, scale=scale
        # )

        # --- SPEED metrics (exact) ---
        use_speedplus_gate = bool(eval_cfg.get("use_speedplus_gate", False))

        E_q_rad, E_q_deg, E_t, E_t_over, E_pose, E_pose_star = compute_speed_metrics_batch(
            q_gt_wxyz=quat_gt_wxyz,     # (B,T,4)
            t_gt=trans_gt,              # (B,T,3)
            q_pred_wxyz=q_pred,         # (B,T,4)
            t_pred=r_pred,              # (B,T,3)
            use_speedplus_gate=use_speedplus_gate,
        )

        # If you want to match the paper’s reporting:
        #   E_T   = mean(E_t)
        #   E_R   = mean(E_q_deg)  (most papers show degrees)
        #   E_Pose = mean(E_pose)  (radians + normalized translation)
        # Or, for SPEED+ gated: mean(E_pose_star)

        all_Epose.extend(list(E_pose.reshape(-1)))
        all_Et.extend(list(E_t.reshape(-1)))
        all_Eqdeg.extend(list(E_q_deg.reshape(-1)))
        all_EtOver.extend(list(E_t_over.reshape(-1)))

        # (optional) if using gate
        all_EposeStar.extend(list(E_pose_star.reshape(-1)))

        # all_ori.extend(list(np.array(ori_deg).reshape(-1)))
        # all_loc.extend(list(np.array(loc_err).reshape(-1)))
        # all_range.extend(list(np.array(range_error).reshape(-1)))
        # all_esa.extend(list(np.array(esa_score).reshape(-1)))

        # -------- Per-sequence pose accumulation --------
        # Convert to flat float arrays
        # ori_flat   = np.array(ori_deg).reshape(-1)
        # loc_flat   = np.array(loc_err).reshape(-1)
        # range_flat = np.array(range_error).reshape(-1)
        # esa_flat   = np.array(esa_score).reshape(-1)
        avg_Epose = float(np.mean(all_Epose)) if len(all_Epose) else float("inf")
        avg_Et    = float(np.mean(all_Et))    if len(all_Et)    else float("inf")
        avg_Eqdeg = float(np.mean(all_Eqdeg)) if len(all_Eqdeg) else float("inf")
        avg_EtOver= float(np.mean(all_EtOver))if len(all_EtOver)else float("inf")

        avg_EposeStar = float(np.mean(all_EposeStar)) if len(all_EposeStar) else avg_Epose

        n_frames = int(esa_flat.size)  # typically B*T

        seq_pose_sum[seq]["esa"]   += float(np.sum(esa_flat))
        seq_pose_sum[seq]["loc"]   += float(np.sum(loc_flat))
        seq_pose_sum[seq]["ori"]   += float(np.sum(ori_flat))
        seq_pose_sum[seq]["range"] += float(np.sum(range_flat))
        seq_pose_sum[seq]["n_frames"] += n_frames


        if is_main_process() and do_debug_print and idx < debug_batches:
            print(f"[Eval dbg] PCK@10={pck10_vals[-1]:.4f} PCK@5={pck5_vals[-1]:.4f}")
            print(f"[Eval dbg] Ori(deg) mean={float(np.mean(ori_deg)):.3f} | Loc mean={float(np.mean(loc_err)):.3f}")

            print("GT t norm mean:", trans_gt.norm(dim=-1).mean().item())
            print("PNP t norm mean:", r_pred.norm(dim=-1).mean().item())
            print("GT t example:", trans_gt[0,0].detach().cpu().numpy())
            print("PNP t example:", r_pred[0,0].detach().cpu().numpy())
            print("scale:", scale)


    def _mean(x):
        return float(np.mean(x)) if len(x) else float("inf")

    avg_pck_10 = _mean(pck10_vals)
    avg_pck_5  = _mean(pck5_vals)
    avg_ori    = _mean(all_ori)
    avg_loc    = _mean(all_loc)
    avg_range  = _mean(all_range)
    avg_esa    = _mean(all_esa)

    # -------------------------
    # Write per-sequence CSV
    # -------------------------
    csv_path = eval_cfg.get("save_per_sequence_csv", None)
    if csv_path is not None:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        rows = []
        seq_names = sorted(set(list(seq_pose_sum.keys()) + list(seq_pck_sum.keys())))

        for s in seq_names:
            n_frames = seq_pose_sum[s]["n_frames"]
            esa   = seq_pose_sum[s]["esa"]   / max(1, n_frames)
            loc   = seq_pose_sum[s]["loc"]   / max(1, n_frames)
            ori   = seq_pose_sum[s]["ori"]   / max(1, n_frames)
            rng   = seq_pose_sum[s]["range"] / max(1, n_frames)

            visn  = seq_pck_sum[s]["vis"]
            pck5  = seq_pck_sum[s]["corr5"]  / max(1.0, visn)
            pck10 = seq_pck_sum[s]["corr10"] / max(1.0, visn)

            rows.append({
                "Seq_name": s,
                "E_Pose": esa,
                "E_T": loc,
                "E_R": ori,
                "Range": rng,
                "PCK@5": pck5,
                "PCK@10": pck10,
            })

        # Append overall row
        rows.append({
            "Seq_name": "__ALL__",
            "E_Pose": avg_esa,
            "E_T":    avg_loc,
            "E_R":    avg_ori,
            "Range": avg_range,
            "PCK@5":  avg_pck_5,
            "PCK@10": avg_pck_10,
        })

        # Write CSV
        float_cols = ["E_Pose", "E_T", "E_R", "Range", "PCK@5", "PCK@10"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["Seq_name", "E_Pose", "E_T", "E_R", "Range", "PCK@5", "PCK@10"]
            )
            writer.writeheader()
            writer.writerows([_format_row_4dp(r, float_keys=float_cols) for r in rows])

        if is_main_process():
            print(f"[Eval] Saved per-sequence results CSV -> {csv_path}")

    return avg_esa, avg_loc, avg_ori, avg_range, avg_pck_10, avg_pck_5


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

    if ds_name == "spades":
        test_set = SPADESDataset(mode="test", **base_kwargs)
    elif ds_name == "spark":
        test_set = SPARKDataset(mode="val", **base_kwargs)
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
    """
    Build SpacecraftHMDSNT directly for evaluation.
    Uses cfg['data'], cfg['model'] and optionally ckpt metadata.
    """

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    # -------- core sizes --------
    T = int(data_cfg["test_T_chunk"])           # for eval we run chunk length
    input_size = int(data_cfg["input_size"])    # e.g., 512
    num_kp = int(data_cfg["num_kp"])

    # ViT patch size: from cfg (preferred), fallback to 16
    patch_size = int(model_cfg.get("patch_size", 16))

    # grid_hw implied by input_size/patch_size (ViT-*/16 => 512 -> 32x32)
    Hp = input_size // patch_size
    Wp = input_size // patch_size
    grid_hw = (Hp, Wp)

    heatmap_hw = tuple(model_cfg.get("heatmap_out", [256, 256]))
    hf_name = model_cfg.get("hf_name", "facebook/dinov3-vitl16-pretrain-lvd1689m")

    adapter_bottleneck = int(model_cfg.get("adapter_bottleneck", 64))
    adapter_drop = float(model_cfg.get("adapter_drop", 0.1))
    adapter_in_last = model_cfg.get("adapter_in_last", None)
    if adapter_in_last is not None:
        adapter_in_last = int(adapter_in_last)

    fuse = model_cfg.get("fuse", "sum")

    # register tokens: safest is auto-infer (None), unless you KNOW the exact count
    n_register_tokens = model_cfg.get("n_register_tokens", None)
    if n_register_tokens is not None:
        n_register_tokens = int(n_register_tokens)

    # Optional: if ckpt stored num_frames / grid, you can override, but not required.
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
        n_register_tokens=n_register_tokens,  # None => auto infer
        ks=3, kt=3,
        fuse=fuse,
    )
    return model

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_cfg("./test_config.yaml")

    ckpt_path = cfg["eval"].get("load_from", None)
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        print(f"Checkpoint path invalid: {ckpt_path}")
        raise FileNotFoundError(f"Checkpoint path invalid: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")  # load on CPU first

    model = build_model_for_eval(cfg, ckpt=ckpt)

    # Load weights
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
    # where to save per-seq csv
    eval_cfg["save_per_sequence_csv"] = os.path.join(
        cfg["eval"].get("save_results_to", "./eval_results"),
        cfg["eval"].get("model_type", "mhsa_mlp")+"_"+ckpt_path.split("/")[-1].rstrip(".pt") + "_per_sequence.csv"
    )
    # eval_cfg["kp_source"] = "dsnt"

    eval_cfg["debug_print"] = True
    eval_cfg["debug_viz"]   = True
    eval_cfg["debug_dir"]   = cfg["eval"].get("debug_dir", "./eval_debug")
    eval_cfg["debug_batches"] = 1
    eval_cfg["debug_frames"]  = 2

    print(f"checking kp_source={eval_cfg['kp_source']}")
    avg_esa, avg_loc, avg_ori, avg_range, avg_pck_10, avg_pck_5 = evaluate_model(
        model, test_loader, eval_cfg, epoch=int(ckpt.get("epoch", 0)))
    print(
        f"Eval Results - ESA/E_Pose: {avg_esa:.4f}, Loc/E_T: {avg_loc:.4f}, Ori/E_R: {avg_ori:.4f}, "
        f"Range: {avg_range:.4f}, PCK@10: {avg_pck_10:.4f}, PCK@5: {avg_pck_5:.4f}"
    )
    # if save_path is not None:
    #     results_dict = {
    #         "avg_esa/E_pose": avg_esa,
    #         "avg_loc/E_T": avg_loc,
    #         "avg_ori/E_R": avg_ori,
    #         "avg_range": avg_range,
    #         "avg_pck_10": avg_pck_10,
    #         "avg_pck_5": avg_pck_5,
    #     }
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     with open(save_path, "w") as f:
    #         json.dump(results_dict, f, indent=4)
    #     print(f"Saved eval results to {save_path}")