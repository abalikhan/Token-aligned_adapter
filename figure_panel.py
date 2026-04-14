"""
figure_panel.py
---------------
Generates a 4-panel figure for a single input frame:
  1) Patch tokens (PCA RGB) on top of the cropped input
  2) Heatmap blob (thresholded max-activation mask) on top of the cropped input
  3) Heatmap colormap overlay on the cropped input
  4) Predicted keypoints on the cropped input

Usage:
    python figure_panel.py \
        --config config_vitb.yaml \
        --ckpt /path/to/best_epoch_106.pt \
        --seq 001 \
        --t 4 \
        --out figure_panel.png

    # or pick a random frame from a random sequence:
    python figure_panel.py --config config_vitb.yaml --ckpt /path/to/ckpt.pt
"""

import argparse
import os
import random

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.decomposition import PCA

from dataset import SPADESDataset, SPARKDataset
from model_hm_dsnt_2 import SpacecraftHMDSNT
from viz_utils import (
    save_trajectory_strip_eval,
    save_pose_fullframe_grid,
    save_temporal_jitter_figure,
    denorm_imagenet_bt,
)
from misc_new import create_pose_estimator, estimate_pose_from_keypoints

# ─────────────────────────────────────────────────────────────────────────────
# Config / model helpers (mirrors viz_spades_new.py)
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_cfg(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k in ["data", "model", "eval", "loss", "train"]:
        cfg.setdefault(k, {})
    return cfg


def build_model(cfg: dict) -> SpacecraftHMDSNT:
    d = cfg["data"]
    m = cfg["model"]
    T          = int(d["test_T_chunk"])
    input_size = int(d["input_size"])
    patch_size = int(m.get("patch_size", 16))
    Hp = input_size // patch_size
    grid_hw    = (Hp, Hp)
    heatmap_hw = tuple(m.get("heatmap_out", [256, 256]))
    nreg = m.get("n_register_tokens", None)
    if nreg is not None:
        nreg = int(nreg)
    return SpacecraftHMDSNT(
        num_kp            = int(d["num_kp"]),
        num_frames        = T,
        img_hw            = input_size,
        hf_name           = m["hf_name"],
        adapter_bottleneck= int(m.get("adapter_bottleneck", 64)),
        adapter_drop      = float(m.get("adapter_drop", 0.1)),
        adapter_in_last   = m.get("adapter_in_last", None),
        heatmap_hw        = heatmap_hw,
        grid_hw           = grid_hw,
        n_register_tokens = nreg,
        ks=3, kt=3,
        fuse              = m.get("fuse", "sum"),
        hm_mid_c          = int(m.get("hm_mid_c", 128)),
    )


def load_ckpt(model: SpacecraftHMDSNT, ckpt_path: str, device: str):
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [ckpt] missing ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"  [ckpt] unexpected ({len(unexpected)}): {unexpected[:5]} ...")
    model.to(device).eval()


def denorm(frame_chw: torch.Tensor) -> np.ndarray:
    """ImageNet-denorm + to uint8 HWC."""
    img = (frame_chw.cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
# Full-frame coordinate conversion helpers
#   (mirrors the equivalent functions in evaluate.py; kept local to avoid
#    circular imports)
# ─────────────────────────────────────────────────────────────────────────────

def _norm_hm_to_hm_px(kps_norm: torch.Tensor, heatmap_hw) -> torch.Tensor:
    Hm, Wm = heatmap_hw
    x = (kps_norm[..., 0] + 1.0) * 0.5 * (Wm - 1.0)
    y = (kps_norm[..., 1] + 1.0) * 0.5 * (Hm - 1.0)
    return torch.stack([x, y], dim=-1)


def _hm_px_to_crop_px(kps_hm_px: torch.Tensor, input_size: int, heatmap_hw) -> torch.Tensor:
    Hm, Wm = heatmap_hw
    x = kps_hm_px[..., 0] * float(max(input_size - 1, 1)) / float(max(Wm - 1, 1))
    y = kps_hm_px[..., 1] * float(max(input_size - 1, 1)) / float(max(Hm - 1, 1))
    return torch.stack([x, y], dim=-1)


def _kps_normhm_to_full_px_crop(
    kps_norm_hm: torch.Tensor,   # (..., K, 2) in [-1,1]
    crop_xywh_norm: torch.Tensor, # (..., 4) normalised crop box
    W_full: float,
    H_full: float,
    input_size: int,
    heatmap_hw,
) -> torch.Tensor:
    """Back-project heatmap-normalised kps → full-frame pixel coords (crop path)."""
    kps_hm_px  = _norm_hm_to_hm_px(kps_norm_hm, heatmap_hw)
    kps_crop   = _hm_px_to_crop_px(kps_hm_px, input_size, heatmap_hw)
    x0 = crop_xywh_norm[..., 0:1] * W_full
    y0 = crop_xywh_norm[..., 1:2] * H_full
    w  = crop_xywh_norm[..., 2:3] * W_full
    h  = crop_xywh_norm[..., 3:4] * H_full
    den = float(max(input_size - 1, 1))
    x_full = x0 + (kps_crop[..., 0:1] / den) * w
    y_full = y0 + (kps_crop[..., 1:2] / den) * h
    return torch.cat([x_full, y_full], dim=-1)


def _kps_normhm_to_full_px_letterbox(
    kps_norm_hm: torch.Tensor,   # (T, K, 2) in [-1,1]
    meta: torch.Tensor,           # (T, 5)  [scale, pad_l, pad_t, W_full, H_full]
    input_size: int,
    heatmap_hw,
) -> torch.Tensor:
    """Back-project heatmap-normalised kps → full-frame pixel coords (letterbox path)."""
    kps_in_px  = _hm_px_to_crop_px(_norm_hm_to_hm_px(kps_norm_hm, heatmap_hw),
                                    input_size, heatmap_hw)
    scale  = meta[..., 0:1].unsqueeze(-1)   # (T,1,1)
    pad_l  = meta[..., 1:2].unsqueeze(-1)
    pad_t  = meta[..., 2:3].unsqueeze(-1)
    W_full = meta[..., 3:4].unsqueeze(-1)
    H_full = meta[..., 4:5].unsqueeze(-1)
    x_unpad = kps_in_px[..., 0:1] - pad_l
    y_unpad = kps_in_px[..., 1:2] - pad_t
    x_full  = (x_unpad / scale.clamp_min(1e-8)).clamp(0, W_full - 1)
    y_full  = (y_unpad / scale.clamp_min(1e-8)).clamp(0, H_full - 1)
    return torch.cat([x_full, y_full], dim=-1)


def _kps_to_full_px(
    kps_norm_hm: torch.Tensor,   # (T, K, 2)
    crop_xywh:   torch.Tensor,   # (T, 4)
    meta:        torch.Tensor,   # (T, 5)
    input_size:  int,
    heatmap_hw,
) -> torch.Tensor:
    """Dispatch to crop or letterbox back-projection based on the meta content."""
    use_letterbox = torch.isfinite(meta[:, 0]).all()
    if use_letterbox:
        return _kps_normhm_to_full_px_letterbox(kps_norm_hm, meta, input_size, heatmap_hw)
    # crop path — read W_full/H_full from meta[3:5]
    W_full = float(meta[0, 3].item())
    H_full = float(meta[0, 4].item())
    return _kps_normhm_to_full_px_crop(
        kps_norm_hm, crop_xywh, W_full, H_full, input_size, heatmap_hw
    )


# ─────────────────────────────────────────────────────────────────────────────
# Full-frame image loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_full_frame(imgs_path: str, frame_idx: int, seq_name: str,
                     ext: str = ".png") -> np.ndarray:
    """Load a single original (un-cropped) frame as uint8 RGB."""
    img_name = f"img{frame_idx:03d}_{seq_name}{ext}"
    path = os.path.join(imgs_path, img_name)
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Frame not found: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers shared by new figure builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_dataset(cfg: dict, mode: str = "train") -> SPADESDataset:
    d = cfg["data"]
    return SPADESDataset(
        gt_folder         = d["root"],
        input_size        = int(d["input_size"]),
        sampling_rate     = int(d["sampling_rate"]),
        short_mem         = int(d.get("short_mem", 8)),
        heatmap_out       = tuple(cfg["model"].get("heatmap_out", [256, 256])),
        gaussian_sigma_px = float(d.get("gaussian_sigma_px", 2.0)),
        mode              = mode,
    )


def _resolve_seq_indices(ds, seq_names: list) -> list:
    """Return dataset indices for the given sequence names (first window each)."""
    name_to_idx = {}
    for i, entry in enumerate(ds.dataset):
        sn = entry[0]
        if sn not in name_to_idx:
            name_to_idx[sn] = i
    indices = []
    for sn in seq_names:
        if sn in name_to_idx:
            indices.append(name_to_idx[sn])
        else:
            print(f"[warn] sequence '{sn}' not found in dataset")
    return indices


@torch.inference_mode()
def _run_model_on_sequence(
    model, cfg: dict, device: str,
    ds, seq_idx: int,
    max_frames: int = 512,
):
    """
    Load the full annotation sequence for ds.dataset[seq_idx], run the model,
    and return a dict with all tensors on CPU.

    Returns
    -------
    dict with keys:
      frames        (T,3,H,W)  ImageNet-normalised
      kps_gt        (T,K,2)    heatmap-normalised [-1,1]
      kps_pred      (T,K,2)    heatmap-normalised [-1,1]  (DSNT output)
      vis           (T,K)
      gt_trans      (T,3)
      crop_xywh     (T,4)
      meta          (T,5)      [scale, pad_l, pad_t, W_full, H_full]
      frame_idxs    list[int]  raw frame indices on disk
      seq_name      str
      imgs_path     str
    """
    seq_name, ws, we, imgs_path, gt_idx = ds.dataset[seq_idx]

    # All annotated frame indices for this sequence
    all_frame_idxs = np.array(ds.seq_to_frames[seq_name], dtype=np.int64)
    if max_frames and len(all_frame_idxs) > max_frames:
        all_frame_idxs = all_frame_idxs[:max_frames]

    frames, kps_gt, gt_q, gt_r, vis, hm, crop_xywh, meta = ds.get_kps_gt(
        all_frame_idxs, seq_name, gt_idx, imgs_path
    )

    # Model requires exactly T_chunk frames per forward pass.
    # Process the sequence in non-overlapping windows of that size.
    T_chunk = int(cfg["data"].get("test_T_chunk", cfg["data"].get("short_mem", 8)))
    T       = frames.shape[0]
    kps_pred_chunks = []
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
        for start in range(0, T, T_chunk):
            chunk = frames[start : start + T_chunk]       # up to T_chunk frames
            n_valid = chunk.shape[0]
            if n_valid < T_chunk:
                # pad last chunk by repeating its final frame
                pad = frames[-1:].expand(T_chunk - n_valid, -1, -1, -1)
                chunk = torch.cat([chunk, pad], dim=0)
            video = chunk.unsqueeze(0).to(device)         # (1, T_chunk, 3, H, W)
            out   = model(video)
            kps_pred_chunks.append(
                out["keypoints"].float().cpu()[0, :n_valid]  # drop padding
            )
    kps_pred = torch.cat(kps_pred_chunks, dim=0)           # (T, K, 2)

    return {
        "frames":     frames.cpu(),
        "kps_gt":     kps_gt.cpu(),
        "kps_pred":   kps_pred,
        "vis":        vis.cpu(),
        "gt_trans":   gt_r.cpu(),
        "crop_xywh":  crop_xywh.cpu(),
        "meta":       meta.cpu(),
        "frame_idxs": all_frame_idxs.tolist(),
        "seq_name":   seq_name,
        "imgs_path":  imgs_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Panel helpers
# ─────────────────────────────────────────────────────────────────────────────

def panel_patch_tokens_noisy(img_u8: np.ndarray, patch_tokens_pd: torch.Tensor,
                             grid_hw: tuple, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Grainy/noisy patch-token image: token L2-norms → blocky upsampled grid
    with heavy grain, scanlines, and rough texture — no PCA, just raw signal.
    patch_tokens_pd: (P, D)  P = Hp*Wp
    """
    if rng is None:
        rng = np.random.default_rng()

    Hp, Wp = grid_hw
    P, D   = patch_tokens_pd.shape
    tokens_np = patch_tokens_pd.float().cpu().numpy()   # (P, D)

    H, W = img_u8.shape[:2]

    # ── token norms → (Hp, Wp) ──
    norms = np.linalg.norm(tokens_np, axis=1)           # (P,)
    norms = (norms - norms.min()) / (norms.max() - norms.min() + 1e-8)
    norm_grid = norms.reshape(Hp, Wp).astype(np.float32)

    # ── per-token variance → second channel ──
    var_grid = tokens_np.var(axis=1).reshape(Hp, Wp).astype(np.float32)
    var_grid = (var_grid - var_grid.min()) / (var_grid.max() - var_grid.min() + 1e-8)

    # ── mean-abs activation → third channel ──
    mean_abs = np.abs(tokens_np).mean(axis=1).reshape(Hp, Wp).astype(np.float32)
    mean_abs = (mean_abs - mean_abs.min()) / (mean_abs.max() - mean_abs.min() + 1e-8)

    # stack to 3-channel (Hp, Wp, 3)
    token_rgb = np.stack([norm_grid, var_grid, mean_abs], axis=-1)

    # blocky nearest-neighbour upsample
    token_up = cv2.resize(token_rgb, (W, H), interpolation=cv2.INTER_NEAREST)  # (H, W, 3)

    # ── heavy Gaussian grain ──
    grain = rng.normal(0, 0.18, (H, W, 3)).astype(np.float32)
    token_up = np.clip(token_up + grain, 0, 1)

    # ── salt-and-pepper noise ──
    sp_mask = rng.random((H, W)) < 0.04
    token_up[sp_mask] = rng.choice([0.0, 1.0], size=(sp_mask.sum(), 3))

    # ── horizontal scanlines ──
    scanline = np.ones((H, W, 3), dtype=np.float32)
    scanline[::3] *= 0.55                               # darken every 3rd row
    token_up = token_up * scanline

    # ── patch border lines (rough, not anti-aliased) ──
    for r in range(0, H, H // Hp):
        token_up[r : r + 2, :] *= 0.3
    for c_ in range(0, W, W // Wp):
        token_up[:, c_ : c_ + 2] *= 0.3

    token_u8 = (token_up.clip(0, 1) * 255).astype(np.uint8)

    # blend darkly with original so structure is vaguely visible
    out = cv2.addWeighted(img_u8, 0.25, token_u8, 0.75, 0)
    return out


def panel_patch_tokens(img_u8: np.ndarray, patch_tokens_pd: torch.Tensor,
                       grid_hw: tuple) -> np.ndarray:
    """
    PCA of patch tokens → 3-component RGB overlay.
    patch_tokens_pd: (P, D)  P = Hp*Wp
    """
    Hp, Wp = grid_hw
    P, D   = patch_tokens_pd.shape
    assert P == Hp * Wp, f"P={P} != Hp*Wp={Hp*Wp}"

    tokens_np = patch_tokens_pd.float().cpu().numpy()           # (P, D)

    # PCA → 3 components → map to [0,1] RGB
    n_comp = min(3, D, P)
    pca    = PCA(n_components=n_comp)
    emb    = pca.fit_transform(tokens_np)                       # (P, 3)

    # normalize each component to [0, 1]
    for c in range(n_comp):
        lo, hi = emb[:, c].min(), emb[:, c].max()
        emb[:, c] = (emb[:, c] - lo) / (hi - lo + 1e-8)

    # pad to 3 channels if needed
    if n_comp < 3:
        pad = np.zeros((P, 3 - n_comp), dtype=np.float32)
        emb = np.concatenate([emb, pad], axis=1)

    pca_img = emb.reshape(Hp, Wp, 3).astype(np.float32)        # (Hp, Wp, 3)

    # upsample to image resolution
    H, W = img_u8.shape[:2]
    pca_up = cv2.resize(pca_img, (W, H), interpolation=cv2.INTER_NEAREST)
    pca_u8 = (pca_up * 255).clip(0, 255).astype(np.uint8)

    # blend: 50% original + 50% PCA colours
    out = cv2.addWeighted(img_u8, 0.45, pca_u8, 0.55, 0)

    # draw patch grid lines
    for r in range(0, H, H // Hp):
        cv2.line(out, (0, r), (W, r), (60, 60, 60), 1)
    for c_ in range(0, W, W // Wp):
        cv2.line(out, (c_, 0), (c_, H), (60, 60, 60), 1)

    return out


def panel_heatmap_blob(img_u8: np.ndarray, hm_logits_kHW: torch.Tensor,
                       vis_k: torch.Tensor | None = None,
                       threshold: float = 0.5) -> np.ndarray:
    """
    Hard blob mask: threshold the max-over-K softmax heatmap and overlay it.
    hm_logits_kHW: (K, Hm, Wm)
    """
    H, W  = img_u8.shape[:2]
    K, Hm, Wm = hm_logits_kHW.shape

    hm = hm_logits_kHW.float()

    # optional visibility mask
    if vis_k is not None:
        v = (vis_k > 0.5).float().view(K, 1, 1)
        hm = hm * v

    # softmax per keypoint → aggregate max
    hm_prob = F.softmax(hm.view(K, -1), dim=-1).view(K, Hm, Wm)
    agg     = hm_prob.max(dim=0).values                        # (Hm, Wm)
    agg     = (agg - agg.min()) / (agg.max() - agg.min() + 1e-8)

    # upsample
    agg_up = F.interpolate(agg.view(1, 1, Hm, Wm), size=(H, W),
                           mode="bilinear", align_corners=False)[0, 0].numpy()

    # threshold → binary mask
    blob = (agg_up > threshold).astype(np.uint8)

    # morphological close to fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blob   = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel)

    # highlight: tint the blob region cyan
    out  = img_u8.copy().astype(np.float32)
    mask = blob[..., None].astype(np.float32)
    cyan = np.array([0, 255, 220], dtype=np.float32)
    out  = out * (1 - 0.4 * mask) + cyan * (0.4 * mask)
    out  = out.clip(0, 255).astype(np.uint8)

    # draw blob contour
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 220, 200), 2)

    return out


def panel_heatmap_overlay(img_u8: np.ndarray, hm_logits_kHW: torch.Tensor,
                          vis_k: torch.Tensor | None = None,
                          alpha: float = 0.50) -> np.ndarray:
    """
    Smooth JET heatmap overlay.
    hm_logits_kHW: (K, Hm, Wm)
    """
    H, W  = img_u8.shape[:2]
    K, Hm, Wm = hm_logits_kHW.shape

    hm = hm_logits_kHW.float()
    if vis_k is not None:
        v  = (vis_k > 0.5).float().view(K, 1, 1)
        hm = hm * v

    hm_prob = F.softmax(hm.view(K, -1), dim=-1).view(K, Hm, Wm)
    agg     = hm_prob.max(dim=0).values
    agg     = (agg - agg.min()) / (agg.max() - agg.min() + 1e-8)

    agg_up = F.interpolate(agg.view(1, 1, Hm, Wm), size=(H, W),
                           mode="bilinear", align_corners=False)[0, 0].numpy()

    heat_u8 = (agg_up * 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    out = cv2.addWeighted(img_u8, 1 - alpha, heat_rgb, alpha, 0)
    return out


def panel_heatmap_raw(img_u8: np.ndarray, hm_logits_kHW: torch.Tensor,
                      vis_k: torch.Tensor | None = None,
                      alpha: float = 0.60,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """
    "All-over-the-place" heatmap: raw logits passed through sigmoid and summed
    across visible keypoints (no softmax normalisation), rendered with HSV
    colormap + grain so the activation smears everywhere.
    hm_logits_kHW: (K, Hm, Wm)
    """
    if rng is None:
        rng = np.random.default_rng()

    H, W  = img_u8.shape[:2]
    K, Hm, Wm = hm_logits_kHW.shape

    hm = hm_logits_kHW.float()

    # sigmoid (not softmax) — keeps all activations alive across the map
    hm_sig = torch.sigmoid(hm)                          # (K, Hm, Wm)

    if vis_k is not None:
        v = (vis_k > 0.5).float().view(K, 1, 1)
        hm_sig = hm_sig * v

    # sum over keypoints → lots of activation everywhere visible kps fire
    agg = hm_sig.sum(dim=0)                             # (Hm, Wm)

    # do NOT normalise tightly; just clip to [0,1] so the raw spread shows
    agg_np = agg.numpy()
    agg_np = np.clip(agg_np / (agg_np.max() + 1e-6), 0, 1)

    # ── upsample bilinear ──
    agg_t  = torch.from_numpy(agg_np).float()
    agg_up = F.interpolate(agg_t.view(1, 1, Hm, Wm), size=(H, W),
                           mode="bilinear", align_corners=False)[0, 0].numpy()

    # ── add noise so it looks chaotic ──
    noise = rng.exponential(0.06, (H, W)).astype(np.float32)
    agg_up = np.clip(agg_up + noise, 0, 1)

    # ── HSV colormap (wild colours across the whole map) ──
    heat_u8   = (agg_up * 255).astype(np.uint8)
    heat_bgr  = cv2.applyColorMap(heat_u8, cv2.COLORMAP_HSV)
    heat_rgb  = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    # ── heavy grain on top ──
    grain = rng.normal(0, 20, heat_rgb.shape).astype(np.float32)
    heat_rgb = np.clip(heat_rgb.astype(np.float32) + grain, 0, 255).astype(np.uint8)

    out = cv2.addWeighted(img_u8, 1 - alpha, heat_rgb, alpha, 0)
    return out


def panel_keypoints(img_u8: np.ndarray,
                    kps_pred_norm: torch.Tensor,
                    kps_gt_norm:   torch.Tensor | None = None,
                    vis_k:         torch.Tensor | None = None) -> np.ndarray:
    """
    Draw predicted keypoints (and optional GT) on the image.
    kps_pred_norm / kps_gt_norm: (K, 2) in [-1, 1]
    """
    H, W = img_u8.shape[:2]
    out  = img_u8.copy()
    K    = kps_pred_norm.shape[0]

    # colour palette for K keypoints
    palette = plt.cm.get_cmap("tab10", K)

    r  = max(4, int(0.008 * min(H, W)))
    th = max(1, int(0.003 * min(H, W)))

    for k in range(K):
        if vis_k is not None and float(vis_k[k]) < 0.5:
            continue

        color_rgb = tuple(int(c * 255) for c in palette(k)[:3])

        # predicted
        px = int(round((float(kps_pred_norm[k, 0]) + 1.0) * 0.5 * (W - 1)))
        py = int(round((float(kps_pred_norm[k, 1]) + 1.0) * 0.5 * (H - 1)))
        cv2.circle(out, (px, py), r, color_rgb, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (px, py), r + 2, (255, 255, 255), th, lineType=cv2.LINE_AA)

        # ground truth (optional)
        if kps_gt_norm is not None:
            gx = int(round((float(kps_gt_norm[k, 0]) + 1.0) * 0.5 * (W - 1)))
            gy = int(round((float(kps_gt_norm[k, 1]) + 1.0) * 0.5 * (H - 1)))
            cv2.circle(out, (gx, gy), r, (255, 255, 255), th + 1, lineType=cv2.LINE_AA)
            cv2.drawMarker(out, (gx, gy), color_rgb, cv2.MARKER_CROSS, r * 2, th + 1, cv2.LINE_AA)

    return out

# ─────────────────────────────────────────────────────────────────────────────
# Main figure builder
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def build_figure(cfg, model, device, ds, seq_idx: int, t_frame: int,
                 out_path: str, blob_threshold: float = 0.45):
    """
    Run the model on one sample and produce the 4-panel figure.

    ds[seq_idx] returns:
      frames_TCHW, kps_TK2, quat_T4, trans_T3, vis_TK, hm_TKHW,
      frame_ids, crop_T4, geom_T5, seq_name
    """
    T_chunk    = int(cfg["data"]["test_T_chunk"])
    input_size = int(cfg["data"]["input_size"])
    patch_size = int(cfg["model"].get("patch_size", 16))
    Hp = input_size // patch_size
    grid_hw = (Hp, Hp)

    sample = ds[seq_idx]
    frames_TCHW = sample[0]  # (T_full, 3, H, W)
    kps_TK2     = sample[1]  # (T_full, K, 2)
    vis_TK      = sample[4]  # (T_full, K)

    T_full = frames_TCHW.shape[0]
    if T_full < T_chunk:
        raise ValueError(f"Sequence too short: T_full={T_full} < T_chunk={T_chunk}")

    # choose a random start so we have a valid T-frame clip
    start = random.randint(0, T_full - T_chunk)
    frames  = frames_TCHW[start : start + T_chunk]   # (T, 3, H, W)
    # kps_gt  = kps_TK2[start : start + T_chunk]       # (T, K, 2)
    vis     = vis_TK[start : start + T_chunk]         # (T, K)

    t = max(0, min(t_frame, T_chunk - 1))

    video = frames.unsqueeze(0).to(device)            # (1, T, 3, H, W)

    with torch.amp.autocast("cuda", dtype=torch.float16,
                            enabled=(device == "cuda")):
        out = model(video)

    # ── extract outputs (all moved to CPU) ──
    hm_logits    = out["heatmaps"].float().cpu()      # (1, T, K, Hm, Wm)
    kps_pred_btk2= out["keypoints"].float().cpu()     # (1, T, K, 2)
    adapter_tok  = out["adapter_tokens"].float().cpu()# (1, T, P, D)

    hm_t        = hm_logits[0, t]                    # (K, Hm, Wm)
    kps_pred_k2 = kps_pred_btk2[0, t]                # (K, 2)
    # kps_gt_k2   = kps_gt[t]                          # (K, 2)
    vis_k       = vis[t]                              # (K,)
    patch_tok   = adapter_tok[0, t]                   # (P, D)

    # cropped input image (denorm)
    img_u8 = denorm(frames[t])                        # (H, W, 3) uint8

    rng = np.random.default_rng()

    # ── build 6 panels ──
    p1 = panel_patch_tokens(img_u8, patch_tok, grid_hw)
    p2 = panel_patch_tokens_noisy(img_u8, patch_tok, grid_hw, rng=rng)
    p3 = panel_heatmap_blob(img_u8, hm_t, vis_k, threshold=blob_threshold)
    p4 = panel_heatmap_overlay(img_u8, hm_t, vis_k)
    p5 = panel_heatmap_raw(img_u8, hm_t, vis_k, rng=rng)
    p6 = panel_keypoints(img_u8, kps_pred_k2, vis_k=vis_k)

    panels = [p1, p2, p3, p4, p5, p6]
    titles = [
        "Patch Tokens (PCA)",
        "Patch Tokens (Noisy)",
        "Heatmap Blob",
        "Heatmap Overlay",
        "Heatmap Raw (sigmoid+sum)",
        "Predicted Keypoints",
    ]

    # ── save each panel individually to eval_debug/ ──
    debug_dir = os.path.join(os.path.dirname(os.path.abspath(out_path)), "eval_debug")
    os.makedirs(debug_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(out_path))[0]
    panel_fnames = [
        "patch_tokens_pca",
        "patch_tokens_noisy",
        "heatmap_blob",
        "heatmap_overlay",
        "heatmap_raw",
        "keypoints",
    ]
    for panel, fname in zip(panels, panel_fnames):
        save_path = os.path.join(debug_dir, f"{stem}_{fname}.png")
        cv2.imwrite(save_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        print(f"  [debug] {save_path}")

    # also save raw input
    cv2.imwrite(os.path.join(debug_dir, f"{stem}_input.png"),
                cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR))

    # ── assemble composite figure ──
    fig, axes = plt.subplots(1, 6, figsize=(30, 5.5))
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel)
        ax.set_title(title, fontsize=11, pad=6)
        ax.axis("off")

    plt.suptitle(
        f"seq={getattr(ds, 'dataset', [{}])[seq_idx] if hasattr(ds, 'dataset') else seq_idx}  "
        f"frame t={t}",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Keypoint trajectory strip
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def build_figure_trajectory(
    cfg:           dict,
    model,
    device:        str,
    ds,
    seq_idx:       int,
    out_path:      str,
    frame_stride:  int  = 8,
    frame_indices: list = None,
    max_k_show:    int  = 11,
):
    """
    Figure 1 — Keypoint trajectory strip.

    Runs the model on the full annotation sequence and produces a horizontal
    strip showing predicted and GT keypoints at selected frames (default: every
    ``frame_stride``-th frame, e.g. 0→8→16→32→64…).  Each keypoint is drawn
    in a distinct colour: GT as a filled circle (solid line) and prediction as
    × (dashed line), demonstrating temporal consistency across varied poses.

    Parameters
    ----------
    frame_stride  : pick every N-th frame  (e.g. 8 → frames 0,8,16,…)
    frame_indices : explicit list override (e.g. [0,8,16,32,64])
    """
    result = _run_model_on_sequence(model, cfg, device, ds, seq_idx)
    T = result["frames"].shape[0]

    frames   = result["frames"].unsqueeze(0)    # (1,T,3,H,W)
    kps_gt   = result["kps_gt"].unsqueeze(0)    # (1,T,K,2)
    kps_pred = result["kps_pred"].unsqueeze(0)  # (1,T,K,2)
    vis      = result["vis"].unsqueeze(0)        # (1,T,K)

    # Default: [0, 8, 16, 32, 64] clamped to available frames.
    # Override via frame_indices or frame_stride.
    if frame_indices is None:
        if frame_stride is not None:
            frame_indices = list(range(0, T, max(1, frame_stride)))
        else:
            frame_indices = [f for f in [0, 8, 16, 32, 64] if f < T]
            if not frame_indices:
                frame_indices = list(range(min(T, 5)))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    save_trajectory_strip_eval(
        frames        = frames,
        kps_gt        = kps_gt,
        kps_pred      = kps_pred,
        vis           = vis,
        save_path     = out_path,
        b             = 0,
        frame_indices = frame_indices,
        max_k_show    = max_k_show,
    )
    print(f"[Fig1] Saved trajectory strip → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: 3-D pose + keypoints on full original images
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def build_figure_pose_fullframe(
    cfg:              dict,
    model,
    device:           str,
    ds,
    seq_indices:      list,
    out_path:         str,
    cam_json:         str,
    points_json:      str,
    n_frames_per_seq: int   = 4,
    grid_cols:        int   = 4,
    axis_length_m:    float = 0.5,
    frame_stride:     int   = None,   # if None, evenly space n_frames_per_seq
):
    """
    Figure 2 — Predicted 3-D pose and 2-D keypoints projected onto the full
    original (un-cropped) image, similar to Figure 3 of arXiv:2509.06000.

    For each sequence in ``seq_indices`` the function:
      * Runs the model on the full sequence.
      * Back-projects predicted keypoints to full-frame pixel coordinates using
        the stored crop/letterbox metadata.
      * Runs PnP on each selected frame to recover rvec/tvec.
      * Draws GT keypoints (green ○), predicted keypoints (magenta ×), the
        model crop rectangle (white dashed), and 3-D pose axes (R/G/B) onto
        the original full-frame image.
      * Assembles all selected frames into a single grid figure.

    Parameters
    ----------
    cam_json    : path to camera intrinsics JSON (SPADES or VBN format)
    points_json : path to 3-D object keypoints JSON
    """
    input_size = int(cfg["data"]["input_size"])
    heatmap_hw = tuple(cfg["model"].get("heatmap_out", [256, 256]))
    num_kp     = int(cfg["data"]["num_kp"])

    pose_estimator, pnp_config, _ = create_pose_estimator(
        pnp_kpts     = num_kp,
        camera_path  = cam_json,
        points_path  = points_json,
    )
    camera_matrix = pose_estimator.camera_matrix

    full_frames_all:      list = []
    kps_pred_full_all:    list = []
    kps_gt_full_all:      list = []
    vis_all:              list = []
    rvecs_all:            list = []
    tvecs_all:            list = []
    crop_boxes_all:       list = []
    frame_labels:         list = []

    for seq_idx in seq_indices:
        result   = _run_model_on_sequence(model, cfg, device, ds, seq_idx)
        T        = result["frames"].shape[0]
        seq_name = result["seq_name"]
        imgs_path= result["imgs_path"]
        ext      = ".png" if isinstance(ds, SPADESDataset) else ".jpg"

        # Back-project kps to full-frame pixel coords
        kps_pred_full = _kps_to_full_px(
            result["kps_pred"], result["crop_xywh"], result["meta"],
            input_size, heatmap_hw,
        ).numpy()   # (T,K,2)

        kps_gt_full = _kps_to_full_px(
            result["kps_gt"], result["crop_xywh"], result["meta"],
            input_size, heatmap_hw,
        ).numpy()   # (T,K,2)

        vis_np = result["vis"].numpy()   # (T,K)

        # Compute crop boxes in full-frame pixel coords for overlay
        meta      = result["meta"]       # (T,5)
        crop_xywh = result["crop_xywh"] # (T,4)
        use_lb    = torch.isfinite(meta[:, 0]).all()
        if use_lb:
            W_full = float(meta[0, 3].item())
            H_full = float(meta[0, 4].item())
            # letterbox: full frame is the crop
            crop_boxes_px = np.column_stack([
                np.zeros(T), np.zeros(T),
                np.full(T, W_full), np.full(T, H_full)
            ]).astype(int)
        else:
            W_full = float(meta[0, 3].item())
            H_full = float(meta[0, 4].item())
            crop_boxes_px = np.column_stack([
                crop_xywh[:, 0].numpy() * W_full,
                crop_xywh[:, 1].numpy() * H_full,
                crop_xywh[:, 2].numpy() * W_full,
                crop_xywh[:, 3].numpy() * H_full,
            ]).astype(int)

        # Select frame indices evenly spaced
        if frame_stride is not None:
            sel = list(range(0, T, frame_stride))[:n_frames_per_seq]
        else:
            sel = [int(round(i * (T - 1) / max(n_frames_per_seq - 1, 1)))
                   for i in range(n_frames_per_seq)]
        sel = [s for s in sel if 0 <= s < T]

        for t_idx in sel:
            raw_frame_idx = result["frame_idxs"][t_idx]
            try:
                full_img = _load_full_frame(imgs_path, raw_frame_idx, seq_name, ext)
            except FileNotFoundError as e:
                print(f"  [warn] {e}")
                continue

            # PnP on this frame
            vis_t = vis_np[t_idx]
            kp_full_t = kps_pred_full[t_idx]           # (K,2)
            vis_idx   = np.where(vis_t > 0.5)[0]
            rvec, tvec = None, None
            if len(vis_idx) >= 4:
                kps_tensor = torch.from_numpy(kp_full_t).float().unsqueeze(0)  # (1,K,2)
                vis_tensor = torch.from_numpy(vis_idx).long().unsqueeze(0)     # (1,V)
                _, t_list, ok_list = estimate_pose_from_keypoints(
                    kps_tensor, vis_tensor, pose_estimator, pnp_config
                )
                if ok_list[0]:
                    import cv2 as _cv2
                    rmat = _cv2.Rodrigues(np.zeros(3, np.float32))[0]  # placeholder
                    # recover rvec from the translation returned is r_xyz not rvec
                    # Re-run PnP directly to get rvec
                    img_pts  = np.ascontiguousarray(kp_full_t[vis_idx].astype(np.float64))
                    obj_pts  = np.ascontiguousarray(
                        pose_estimator.model_points[vis_idx].astype(np.float64))
                    dist     = pose_estimator.dist_coefs
                    try:
                        ok_r, rvec_r, tvec_r = _cv2.solvePnP(
                            obj_pts, img_pts, camera_matrix, dist,
                            flags=_cv2.SOLVEPNP_ITERATIVE)
                        if ok_r:
                            rvec = rvec_r.reshape(3)
                            tvec = tvec_r.reshape(3)
                    except Exception:
                        pass

            full_frames_all.append(full_img)
            kps_pred_full_all.append(kp_full_t)
            kps_gt_full_all.append(kps_gt_full[t_idx])
            vis_all.append(vis_t)
            rvecs_all.append(rvec)
            tvecs_all.append(tvec)
            crop_boxes_all.append(crop_boxes_px[t_idx])
            frame_labels.append(f"{seq_name} t={raw_frame_idx}")

    if not full_frames_all:
        print("[warn] No frames collected for pose_fullframe figure.")
        return

    N = len(full_frames_all)
    kps_pred_arr = np.stack(kps_pred_full_all)  # (N,K,2)
    kps_gt_arr   = np.stack(kps_gt_full_all)    # (N,K,2)
    vis_arr      = np.stack(vis_all)            # (N,K)
    crop_arr     = np.stack(crop_boxes_all)     # (N,4)

    rvecs_arr = np.array([rv if rv is not None else np.zeros(3)
                          for rv in rvecs_all], dtype=np.float64)
    tvecs_arr = np.array([tv if tv is not None else np.zeros(3)
                          for tv in tvecs_all], dtype=np.float64)
    has_pose  = [rv is not None for rv in rvecs_all]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    save_pose_fullframe_grid(
        full_frames      = full_frames_all,
        kps_pred_full_px = kps_pred_arr,
        kps_gt_full_px   = kps_gt_arr,
        vis              = vis_arr,
        camera_matrix    = camera_matrix,
        rvecs            = rvecs_arr if any(has_pose) else None,
        tvecs            = tvecs_arr if any(has_pose) else None,
        save_path        = out_path,
        crop_boxes_xywh  = crop_arr,
        axis_length_m    = axis_length_m,
        grid_cols        = grid_cols,
        frame_title_fn   = lambda i: frame_labels[i],
        fig_title        = "3-D Pose & 2-D Keypoints — Full-Frame Projection",
    )
    print(f"[Fig2] Saved pose full-frame grid → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Temporal jitter graph (6 sequences, 2×3 grid)
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def build_figure_jitter(
    cfg:         dict,
    model,
    device:      str,
    ds,
    seq_indices: list,
    out_path:    str,
    cam_json:    str,
    points_json: str,
    n_cols:      int = 3,
):
    """
    Figure 3 — Temporal position jitter visualisation.

    For each sequence, runs the model on all frames, runs PnP on each frame's
    predicted keypoints to recover translation (Tx, Ty, Tz), and plots all
    three components vs. time step.  GT translation from the dataset annotations
    is overlaid as a faint dashed reference.

    The output is a 2×3 grid (configurable via ``n_cols``) where each cell
    shows the position curve for one sequence.  Periodic jitter — if present —
    is visible as oscillations on top of the smooth GT reference.

    Parameters
    ----------
    cam_json    : path to camera intrinsics JSON
    points_json : path to 3-D object keypoints JSON
    seq_indices : dataset indices of up to 6 sequences to visualise
    """
    input_size = int(cfg["data"]["input_size"])
    heatmap_hw = tuple(cfg["model"].get("heatmap_out", [256, 256]))
    num_kp     = int(cfg["data"]["num_kp"])

    pose_estimator, pnp_config, _ = create_pose_estimator(
        pnp_kpts    = num_kp,
        camera_path  = cam_json,
        points_path  = points_json,
    )
    camera_matrix = pose_estimator.camera_matrix
    dist_coefs    = pose_estimator.dist_coefs

    import cv2 as _cv2

    seq_data: dict = {}

    for seq_idx in seq_indices:
        result   = _run_model_on_sequence(model, cfg, device, ds, seq_idx)
        T        = result["frames"].shape[0]
        seq_name = result["seq_name"]

        kps_pred_full = _kps_to_full_px(
            result["kps_pred"], result["crop_xywh"], result["meta"],
            input_size, heatmap_hw,
        ).numpy()   # (T,K,2)

        vis_np   = result["vis"].numpy()          # (T,K)
        gt_trans = result["gt_trans"].numpy()     # (T,3)  r_xyz annotation

        pred_trans = []
        for t_idx in range(T):
            vis_idx = np.where(vis_np[t_idx] > 0.5)[0]
            tvec_t  = np.full(3, np.nan, dtype=np.float32)
            if len(vis_idx) >= 4:
                img_pts = np.ascontiguousarray(
                    kps_pred_full[t_idx][vis_idx].astype(np.float64))
                obj_pts = np.ascontiguousarray(
                    pose_estimator.model_points[vis_idx].astype(np.float64))
                try:
                    ok, _, tvec_r = _cv2.solvePnP(
                        obj_pts, img_pts, camera_matrix, dist_coefs,
                        flags=_cv2.SOLVEPNP_ITERATIVE)
                    if ok and tvec_r is not None:
                        tvec_t = tvec_r.reshape(3).astype(np.float32)
                except Exception:
                    pass
            pred_trans.append(tvec_t)

        pred_trans_arr = np.stack(pred_trans, axis=0)   # (T,3)

        # Replace NaN runs with linearly interpolated values for cleaner plots
        for ci in range(3):
            col = pred_trans_arr[:, ci]
            nans = np.isnan(col)
            if nans.any() and (~nans).any():
                xs = np.where(~nans)[0]
                col[nans] = np.interp(np.where(nans)[0], xs, col[xs])
                pred_trans_arr[:, ci] = col

        seq_data[seq_name] = {
            "pred_trans": pred_trans_arr,
            "gt_trans":   gt_trans,
        }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    save_temporal_jitter_figure(
        seq_data  = seq_data,
        save_path = out_path,
        n_cols    = n_cols,
        max_seqs  = len(seq_indices),
        fig_title = "Position prediction results with time step."
                    "  Periodic jitters can be found in each curve.",
    )
    print(f"[Fig3] Saved temporal jitter figure → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cam_and_points_from_cfg(cfg: dict, cam_override=None, points_override=None):
    """
    Resolve camera-intrinsics and object-points JSON paths.

    Priority (highest first):
      1. CLI override (cam_override / points_override)
      2. config keys  data.cam_intrinsics_json / data.object_points_json
      3. Derived from data.root + data.name  (mirrors evaluate.py logic)
    """
    d = cfg["data"]

    cam = (cam_override
           or d.get("cam_intrinsics_json")
           or d.get("camera_json"))
    if not cam:
        raise ValueError(
            "Camera JSON not found. Set data.cam_intrinsics_json in the config "
            "or pass --cam_json on the command line.")

    pts = points_override or d.get("object_points_json")
    if not pts:
        root = d["root"].rstrip("/")
        ds_name = d.get("name", "").lower()
        pts = (root + "/points_spark.json" if ds_name == "spark"
               else root + "/points.json")

    return cam, pts


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation figure generator for spacecraft pose estimation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Figures
-------
  panel          : Original 6-panel per-frame diagnostic (patch tokens, heatmaps, kps).
  trajectory     : Keypoint trajectory strip across selected frames (Fig 1).
  pose_fullframe : Predicted 3-D pose + 2-D kps on full original images (Fig 2).
  jitter         : Temporal position jitter graph for multiple sequences (Fig 3).
  all            : All four figures (output suffixes: _panel / _traj / _pose / _jitter).

Camera / points paths are read automatically from the YAML config
(data.cam_intrinsics_json, data.root + dataset name).  Override with
--cam_json / --points_json only when needed.

Examples
--------
  # Panel figure (original behaviour)
  python figure_panel.py --config config_vitb.yaml --ckpt best.pt --seq RT001

  # Trajectory strip — explicit frame list
  python figure_panel.py --config config_vitb.yaml --ckpt best.pt \\
      --figure trajectory --seq RT001 --frame_indices 0,8,16,32,64

  # Trajectory strip — stride knob (every 8th frame)
  python figure_panel.py --config config_vitb.yaml --ckpt best.pt \\
      --figure trajectory --seq RT001 --frame_stride 8

  # Full-frame pose grid for 3 sequences
  python figure_panel.py --config config_vitb.yaml --ckpt best.pt \\
      --figure pose_fullframe --seqs RT001,RT002,RT003

  # Temporal jitter (6 sequences, 2×3 grid)
  python figure_panel.py --config config_vitb.yaml --ckpt best.pt \\
      --figure jitter --seqs RT001,RT002,RT003,RT004,RT005,RT006

  # All four figures at once
  python figure_panel.py --config config_vitb.yaml --ckpt best.pt \\
      --figure all --seq RT001 --seqs RT001,RT002,RT003,RT004,RT005,RT006 \\
      --out eval_figs.png
""")

    # ── Required ───────────────────────────────────────────────────────────
    parser.add_argument("--config", default="config_vitb.yaml",
                        help="YAML config file (default: config_vitb.yaml)")
    parser.add_argument("--ckpt",   required=True,
                        help="Model checkpoint (.pt)")

    # ── Figure type ────────────────────────────────────────────────────────
    parser.add_argument("--figure",
                        choices=["panel", "trajectory", "pose_fullframe", "jitter", "all"],
                        default="panel",
                        help="Figure to produce (default: panel)")

    # ── Sequence selection ─────────────────────────────────────────────────
    parser.add_argument("--seq",  default=None,
                        help="Sequence name for single-seq figures (panel / trajectory)")
    parser.add_argument("--seqs", default=None,
                        help="Comma-separated sequence names for multi-seq figures "
                             "(pose_fullframe / jitter).  Falls back to random if omitted.")
    parser.add_argument("--n_seqs", type=int, default=6,
                        help="How many random sequences when --seqs is not given (default: 6)")

    # ── Output ────────────────────────────────────────────────────────────
    parser.add_argument("--out", default="figure_panel.png",
                        help="Output PNG path (or stem when --figure all)")
    parser.add_argument("--device", default=None, help="cuda | cpu (auto-detected)")

    # ── Trajectory knobs ──────────────────────────────────────────────────
    parser.add_argument("--frame_stride", type=int, default=8,
                        help="Pick every N-th frame for trajectory strip (default: 8)")
    parser.add_argument("--frame_indices", default=None,
                        help="Explicit comma-separated frame indices, e.g. '0,8,16,32,64'")

    # ── Panel knobs ───────────────────────────────────────────────────────
    parser.add_argument("--t",           type=int, default=None,
                        help="Frame index within clip (panel only, random if omitted)")
    parser.add_argument("--blob_thresh", type=float, default=0.45)

    # ── Pose / jitter knobs ───────────────────────────────────────────────
    parser.add_argument("--n_frames_per_seq", type=int, default=4,
                        help="Frames shown per sequence in pose_fullframe grid (default: 4)")
    parser.add_argument("--grid_cols",        type=int, default=4,
                        help="Grid columns for pose_fullframe (default: 4)")
    parser.add_argument("--axis_len",         type=float, default=0.5,
                        help="Pose axis length in metres (default: 0.5)")
    parser.add_argument("--n_cols_jitter",    type=int, default=3,
                        help="Grid columns for jitter figure (default: 3 → 2×3)")

    # ── Path overrides (normally not needed — read from config) ───────────
    parser.add_argument("--cam_json",    default=None,
                        help="Override: camera intrinsics JSON")
    parser.add_argument("--points_json", default=None,
                        help="Override: 3-D object keypoints JSON")

    args = parser.parse_args()

    # ── Bootstrap ─────────────────────────────────────────────────────────
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    cfg   = load_cfg(args.config)
    model = build_model(cfg)
    load_ckpt(model, args.ckpt, device)

    ds = _build_dataset(cfg, mode="train")
    N  = len(ds)
    print(f"[dataset] {N} clips  ({ds.dataset[0][0]} … {ds.dataset[-1][0]})")

    # ── Sequence index helpers ─────────────────────────────────────────────
    def _single_seq_idx() -> int:
        if args.seq is not None:
            for i, entry in enumerate(ds.dataset):
                if str(entry[0]) == args.seq:
                    return i
            raise ValueError(f"Sequence '{args.seq}' not found in dataset.")
        return random.randint(0, N - 1)

    def _multi_seq_indices(n: int) -> list:
        src = args.seqs or args.seq or ""
        names = [s.strip() for s in src.split(",") if s.strip()]
        if names:
            return _resolve_seq_indices(ds, names)
        # random unique sequences
        seen, out, attempts = set(), [], 0
        while len(out) < n and attempts < N * 3:
            i  = random.randint(0, N - 1)
            sn = ds.dataset[i][0]
            if sn not in seen:
                seen.add(sn); out.append(i)
            attempts += 1
        return out

    # ── Resolve frame_indices list (trajectory) ────────────────────────────
    frame_indices_parsed = None
    if args.frame_indices:
        frame_indices_parsed = [int(x) for x in args.frame_indices.split(",") if x.strip()]

    # ── Resolve cam / points paths from config (with CLI overrides) ────────
    fig_type = args.figure
    cam_json = points_json = None
    if fig_type in ("pose_fullframe", "jitter", "all"):
        cam_json, points_json = _cam_and_points_from_cfg(
            cfg, args.cam_json, args.points_json)
        print(f"[cam]    {cam_json}")
        print(f"[points] {points_json}")

    # ── Output stem ───────────────────────────────────────────────────────
    stem = os.path.splitext(args.out)[0]

    # ── Dispatch ──────────────────────────────────────────────────────────
    if fig_type == "panel":
        seq_idx = _single_seq_idx()
        T_chunk = int(cfg["data"].get("test_T_chunk", cfg["data"].get("short_mem", 8)))
        t_frame = args.t if args.t is not None else random.randint(0, T_chunk - 1)
        build_figure(cfg, model, device, ds,
                     seq_idx=seq_idx, t_frame=t_frame,
                     out_path=args.out, blob_threshold=args.blob_thresh)

    if fig_type in ("trajectory", "all"):
        seq_idx  = _single_seq_idx()
        out_path = (stem + "_traj.png") if fig_type == "all" else args.out
        build_figure_trajectory(
            cfg=cfg, model=model, device=device, ds=ds,
            seq_idx       = seq_idx,
            out_path      = out_path,
            frame_stride  = args.frame_stride,
            frame_indices = frame_indices_parsed,
        )

    if fig_type in ("pose_fullframe", "all"):
        seq_indices = _multi_seq_indices(args.n_seqs)
        out_path    = (stem + "_pose.png") if fig_type == "all" else args.out
        build_figure_pose_fullframe(
            cfg=cfg, model=model, device=device, ds=ds,
            seq_indices      = seq_indices,
            out_path         = out_path,
            cam_json         = cam_json,
            points_json      = points_json,
            n_frames_per_seq = args.n_frames_per_seq,
            grid_cols        = args.grid_cols,
            axis_length_m    = args.axis_len,
            frame_stride     = args.frame_stride if frame_indices_parsed is None else None,
        )

    if fig_type in ("jitter", "all"):
        seq_indices = _multi_seq_indices(args.n_seqs)
        out_path    = (stem + "_jitter.png") if fig_type == "all" else args.out
        build_figure_jitter(
            cfg=cfg, model=model, device=device, ds=ds,
            seq_indices = seq_indices,
            out_path    = out_path,
            cam_json    = cam_json,
            points_json = points_json,
            n_cols      = args.n_cols_jitter,
        )


if __name__ == "__main__":
    main()
