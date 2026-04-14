"""
Keypoint overlay visualization.

Uses keypoints directly from model output (already computed via
_softmax_2d → dsnt in the forward pass). No recomputation needed.
"""

import torch
import numpy as np
import os
import cv2
import math
from typing import Tuple, Optional, List, Dict, Any
from PIL import Image
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ── ImageNet denormalization ──
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1)


@torch.no_grad()
def denorm_imagenet_bt(frames_btchw: torch.Tensor) -> torch.Tensor:
    """Denormalize ImageNet-normalized frames: (B, T, C, H, W) → [0, 1]"""
    mean = _IMAGENET_MEAN.to(frames_btchw.device)
    std = _IMAGENET_STD.to(frames_btchw.device)
    return (frames_btchw * std + mean).clamp(0.0, 1.0)


def _norm_to_px(
    coords_norm: torch.Tensor,  # (..., 2) in [-1, 1] as (x, y)
    H: int,
    W: int,
) -> torch.Tensor:
    """Convert normalized [-1, 1] coordinates to pixel coordinates."""
    px = coords_norm.clone()
    px[..., 0] = ((px[..., 0] + 1.0) * 0.5) * (W - 1.0)
    px[..., 1] = ((px[..., 1] + 1.0) * 0.5) * (H - 1.0)
    return px


def _draw_keypoints_on_frame(
    img_np: np.ndarray,
    gt_px: np.ndarray,
    pred_px: np.ndarray,
    vis: np.ndarray,
    radius: int = 3,
    thickness: int = 1,
    color_gt: Tuple[int, int, int] = (0, 255, 0),
    color_pred: Tuple[int, int, int] = (255, 0, 255),
    draw_lines: bool = True,
) -> np.ndarray:
    """Draw GT and predicted keypoints on a single frame."""
    K = gt_px.shape[0]
    for k in range(K):
        if vis[k] < 0.5:
            continue
        pt_gt = (int(round(gt_px[k, 0])), int(round(gt_px[k, 1])))
        pt_pr = (int(round(pred_px[k, 0])), int(round(pred_px[k, 1])))
        if draw_lines:
            cv2.line(img_np, pt_gt, pt_pr, (100, 100, 100), 1, lineType=cv2.LINE_AA)
        cv2.circle(img_np, pt_gt, radius, color_gt, -1, lineType=cv2.LINE_AA)
        cv2.circle(img_np, pt_pr, radius, color_pred, thickness, lineType=cv2.LINE_AA)
    return img_np


@torch.no_grad()
def save_kp_overlay_clip_grid(
    frames: torch.Tensor,          # (B, T, C, H, W)
    kps_gt_norm: torch.Tensor,     # (B, T, K, 2) in [-1, 1]
    kps_pred_norm: torch.Tensor,   # (B, T, K, 2) in [-1, 1] — directly from model
    vis: torch.Tensor,             # (B, T, K)
    save_path: str,
    b: int = 0,
    grid_cols: int = 4,
    max_frames: int = 16,
    frame_border: int = 2,
    draw_lines: bool = True,
):
    """
    Save one video clip as a grid figure with keypoint overlays.

    All T frames (up to max_frames) arranged in a grid.
    Each frame shows:
      - Green filled circles: GT keypoints
      - Magenta hollow circles: Predicted keypoints
      - Gray lines: error between GT and pred
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    B, T, C, H_img, W_img = frames.shape
    T_show = min(T, max_frames)
    grid_rows = int(math.ceil(T_show / grid_cols))

    # ── Slice batch element and move to CPU ──
    gt_b = kps_gt_norm[b, :T_show].cpu()         # (T_show, K, 2)
    pred_b = kps_pred_norm[b, :T_show].cpu()      # (T_show, K, 2)
    vis_np = vis[b, :T_show].cpu().numpy()         # (T_show, K)

    # ── Convert to pixel coordinates ──
    gt_px = _norm_to_px(gt_b, H_img, W_img)       # (T_show, K, 2)
    pred_px = _norm_to_px(pred_b, H_img, W_img)   # (T_show, K, 2)

    # ── Denormalize frames ──
    frames_vis = denorm_imagenet_bt(frames)
    frames_b = frames_vis[b, :T_show].cpu()        # (T_show, C, H, W)

    # ── Drawing parameters ──
    radius = max(2, int(0.006 * max(H_img, W_img)))
    thickness = max(1, int(0.003 * max(H_img, W_img)))

    # ── Build grid ──
    grid_h = grid_rows * H_img + (grid_rows - 1) * frame_border
    grid_w = grid_cols * W_img + (grid_cols - 1) * frame_border
    grid_img = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 40

    for t_idx in range(T_show):
        row = t_idx // grid_cols
        col = t_idx % grid_cols
        y0 = row * (H_img + frame_border)
        x0 = col * (W_img + frame_border)

        frame_np = (frames_b[t_idx].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8).copy()
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

        frame_bgr = _draw_keypoints_on_frame(
            frame_bgr,
            gt_px=gt_px[t_idx].numpy(),
            pred_px=pred_px[t_idx].numpy(),
            vis=vis_np[t_idx],
            radius=radius,
            thickness=thickness,
            draw_lines=draw_lines,
        )

        # Frame index label
        font_scale = max(0.3, 0.001 * max(H_img, W_img))
        font_th = max(1, int(0.002 * max(H_img, W_img)))
        cv2.putText(
            frame_bgr, f"t={t_idx}",
            (4, int(20 * font_scale / 0.3)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (255, 255, 255), font_th, cv2.LINE_AA,
        )

        grid_img[y0 : y0 + H_img, x0 : x0 + W_img] = frame_bgr

    try:
        cv2.imwrite(save_path, grid_img)
    except Exception:
        grid_rgb = cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB)
        Image.fromarray(grid_rgb).save(save_path)


@torch.no_grad()
def save_kp_overlay_bt(
    frames: torch.Tensor,
    kps_gt_norm: torch.Tensor,
    kps_pred_norm: torch.Tensor,
    vis: torch.Tensor,
    save_path: str,
    b: int = 0,
    t: int = 0,
):
    """Save a single frame with keypoint overlay."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    B, T, C, H_img, W_img = frames.shape

    gt_px = _norm_to_px(kps_gt_norm[b, t].cpu(), H_img, W_img).numpy()
    pred_px = _norm_to_px(kps_pred_norm[b, t].cpu(), H_img, W_img).numpy()
    vis_t = vis[b, t].cpu().numpy() if vis is not None else np.ones(gt_px.shape[0])

    frames_vis = denorm_imagenet_bt(frames)
    frame_np = (frames_vis[b, t].cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8).copy()
    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

    radius = max(2, int(0.006 * max(H_img, W_img)))
    thickness = max(1, int(0.003 * max(H_img, W_img)))

    frame_bgr = _draw_keypoints_on_frame(
        frame_bgr, gt_px, pred_px, vis_t,
        radius=radius, thickness=thickness,
    )

    try:
        cv2.imwrite(save_path, frame_bgr)
    except Exception:
        Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).save(save_path)


def _get_img_np(frames_vis: torch.Tensor, b: int, t: int) -> np.ndarray:
    """Extract single frame as uint8 numpy (H, W, 3)."""
    return (
        frames_vis[b, t].detach().cpu().permute(1, 2, 0).numpy() * 255
    ).clip(0, 255).astype(np.uint8)


def _upsample_map(arr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Upsample (Hp, Wp) map to (H, W) via bilinear interpolation."""
    t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=size, mode="bilinear", align_corners=False)
    return t[0, 0].numpy()



@torch.no_grad()
def save_token_diagnostics(
    out: Dict[str, torch.Tensor],
    frames: torch.Tensor,         # (B, T, C, H, W)
    kps_gt: torch.Tensor,         # (B, T, K, 2) in [-1, 1]
    kps_pred: torch.Tensor,       # (B, T, K, 2) in [-1, 1]
    vis: torch.Tensor,            # (B, T, K)
    cfg: dict,
    save_path: str,
    b: int = 0,
    max_k_show: int = 8,
) -> Optional[Dict[str, Any]]:
    """
    Visualizes token_spatial_alignment_loss behavior.

    For each keypoint k the loss computes:
      tgt_k(p)  ∝ exp(-||p - kp||² / 2σ²)          (Gaussian target over patches)
      proto_k    = normalize(Σ_p tgt_k(p) · tok_p)  (Gaussian-weighted prototype)
      P_k(p)    = softmax(sim(tok_p, proto_k) / τ)   (predicted distribution)
      loss      = KL(tgt_k || P_k)

    Layout:
      Rows 0..Kvis-1  Predicted similarity map P_k(p) [hot] + target Gaussian [cyan contour]
                      at selected frames. Green + = GT keypoint. Red × = DSNT predicted.
                      KL value annotated per cell.
      Row Kvis        KL(tgt || pred) per frame — loss being minimized (↓ better).
      Row Kvis+1      Temporal prototype cosine similarity per keypoint (↑ better).
      Row Kvis+2      Keypoint trajectory strip.
      Row Kvis+3      Target Gaussian tgt_k(p) overlay — what the loss drives P_k toward.
    """
    input_size  = int(cfg["data"]["input_size"])
    patch_size  = int(cfg["model"].get("patch_size", 16))
    sigma_patch = float(cfg["loss"].get("sigma_patch", 0.35))
    temp_pred   = float(cfg["loss"].get("temp_pred", 0.07))

    Hp = input_size // patch_size
    Wp = input_size // patch_size
    P  = Hp * Wp

    tok = out.get("adapter_tokens")
    if tok is None:
        return None

    B, T, P_got, D = tok.shape
    if P_got > P:
        tok = tok[:, :, -P:, :]
    elif P_got < P:
        return None

    frames_vis = denorm_imagenet_bt(frames)
    vis_bt     = vis[b].float()   # (T, K)
    K          = vis_bt.shape[1]
    device     = tok.device
    dtype      = tok.dtype

    # ── Select frames ──
    if T >= 5:
        t_indices = [0, T // 4, T // 2, 3 * T // 4, T - 1]
    elif T >= 3:
        t_indices = [0, T // 2, T - 1]
    else:
        t_indices = list(range(T))
    n_tframes = len(t_indices)

    # ── Select visible keypoints ──
    vis_mask = torch.ones(K, device=device, dtype=torch.bool)
    for ti in t_indices:
        vis_mask &= vis_bt[ti] > 0.5
    k_ids = torch.where(vis_mask)[0].tolist()[:max_k_show]
    if not k_ids:
        mid_t = t_indices[len(t_indices) // 2]
        k_ids = torch.where(vis_bt[mid_t] > 0.5)[0].tolist()[:max_k_show]
    if not k_ids:
        return None
    Kvis = len(k_ids)

    # ── Patch grid in [-1, 1] ──
    ys_g, xs_g = torch.meshgrid(
        torch.linspace(-1, 1, Hp, device=device, dtype=dtype),
        torch.linspace(-1, 1, Wp, device=device, dtype=dtype),
        indexing="ij",
    )
    patch_xy = torch.stack([xs_g, ys_g], dim=-1).view(P, 2)  # (P, 2)

    tok_b  = F.normalize(tok[b].float(), dim=-1)  # (T, P, D)
    sig2   = float(max(sigma_patch, 1e-6)) ** 2
    temp   = float(max(temp_pred, 1e-6))
    eps_kl = 1e-8

    # ── Per (t, ki_local): target, predicted map, KL, prototype ──
    tgt_maps:   Dict[Tuple[int, int], np.ndarray] = {}
    pred_maps:  Dict[Tuple[int, int], np.ndarray] = {}
    kl_vals:    Dict[Tuple[int, int], float]       = {}
    protos_all: Dict[Tuple[int, int], torch.Tensor] = {}

    for t in range(T):
        tok_t = tok_b[t]  # (P, D)
        for ki_local, k_global in enumerate(k_ids):
            if vis_bt[t, k_global] < 0.5:
                continue
            kp = kps_gt[b, t, k_global]  # (2,) in [-1,1]

            # Gaussian target distribution
            dist2      = ((patch_xy - kp.unsqueeze(0)) ** 2).sum(-1)  # (P,)
            tgt        = F.softmax(-0.5 * dist2 / sig2, dim=-1)        # (P,)

            # Gaussian-weighted prototype
            proto = F.normalize((tgt.unsqueeze(-1) * tok_t).sum(0), dim=-1)  # (D,)
            protos_all[(t, ki_local)] = proto

            # Predicted distribution P_k(p)
            sim   = tok_t @ proto                                    # (P,)
            log_p = F.log_softmax(sim / temp, dim=-1)               # (P,)
            pred_p = log_p.exp()                                     # (P,)

            # KL(tgt || pred)
            kl = (tgt * (torch.log(tgt.clamp_min(eps_kl)) - log_p)).sum().item()

            tgt_maps[(t, ki_local)]  = tgt.reshape(Hp, Wp).cpu().numpy()
            pred_maps[(t, ki_local)] = pred_p.reshape(Hp, Wp).cpu().numpy()
            kl_vals[(t, ki_local)]   = float(kl)

    # ── Temporal prototype cosine similarity ──
    cosine_curves: Dict[int, List[float]] = {}
    for ki in range(Kvis):
        curve = []
        for t in range(T - 1):
            if (t, ki) in protos_all and (t + 1, ki) in protos_all:
                cos = float((protos_all[(t, ki)] * protos_all[(t + 1, ki)]).sum().item())
                curve.append(cos)
            else:
                curve.append(float("nan"))
        cosine_curves[ki] = curve

    # ── KL per frame (mean over visible keypoints) ──
    kl_per_frame = []
    for t in range(T):
        vals = [kl_vals[(t, ki)] for ki in range(Kvis) if (t, ki) in kl_vals]
        kl_per_frame.append(float(np.mean(vals)) if vals else float("nan"))

    # ── Summary stats ──
    mean_cosines = []
    for ki in range(Kvis):
        valid = [v for v in cosine_curves[ki] if not math.isnan(v)]
        if valid:
            mean_cosines.append(np.mean(valid))
    overall_cos = float(np.mean(mean_cosines)) if mean_cosines else 0.0

    valid_kl   = [v for v in kl_per_frame if not math.isnan(v)]
    overall_kl = float(np.mean(valid_kl)) if valid_kl else 0.0

    colors = plt.cm.tab10(np.linspace(0, 1, min(10, Kvis)))

    # Derive base path (strip extension)
    base, _ = os.path.splitext(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ──────────────────────────────────────────────
    # Figure 1: Predicted similarity maps (grid Kvis × n_tframes)
    # ──────────────────────────────────────────────
    fig1 = plt.figure(figsize=(3.2 * n_tframes, 3.0 * Kvis))
    gs1  = GridSpec(Kvis, n_tframes, figure=fig1, hspace=0.30, wspace=0.0)
    fig1.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.02)

    for ki_row in range(Kvis):
        k_global = k_ids[ki_row]
        for col, t_idx in enumerate(t_indices):
            ax  = fig1.add_subplot(gs1[ki_row, col])
            img = _get_img_np(frames_vis, b, t_idx)
            ax.imshow(img)

            if (t_idx, ki_row) in pred_maps:
                pred_up = _upsample_map(pred_maps[(t_idx, ki_row)], (input_size, input_size))
                vmin, vmax = pred_up.min(), pred_up.max()
                if vmax - vmin > 1e-8:
                    pred_up = (pred_up - vmin) / (vmax - vmin)
                ax.imshow(pred_up, cmap="hot", alpha=0.5, vmin=0, vmax=1)
                if (t_idx, ki_row) in tgt_maps:
                    tgt_up   = _upsample_map(tgt_maps[(t_idx, ki_row)], (input_size, input_size))
                    tgt_norm = (tgt_up - tgt_up.min()) / (tgt_up.max() - tgt_up.min() + 1e-8)
                    ax.contour(tgt_norm, levels=[0.3, 0.6], colors=["cyan"], linewidths=0.8, alpha=0.8)

            gx_px = float((kps_gt[b, t_idx, k_global, 0].item() + 1) * 0.5 * (input_size - 1))
            gy_px = float((kps_gt[b, t_idx, k_global, 1].item() + 1) * 0.5 * (input_size - 1))
            ax.plot(gx_px, gy_px, "g+", markersize=12, markeredgewidth=2)

            px_px = float((kps_pred[b, t_idx, k_global, 0].item() + 1) * 0.5 * (input_size - 1))
            py_px = float((kps_pred[b, t_idx, k_global, 1].item() + 1) * 0.5 * (input_size - 1))
            ax.plot(px_px, py_px, "rx", markersize=10, markeredgewidth=2)

            if (t_idx, ki_row) in kl_vals:
                ax.text(4, input_size - 6, f"KL={kl_vals[(t_idx, ki_row)]:.3f}",
                        color="white", fontsize=7,
                        bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"))

            if col == 0:
                ax.set_ylabel(f"kp{k_global}", fontsize=11, fontweight="bold")
            if ki_row == 0:
                ax.set_title(f"t={t_idx}", fontsize=11)
            ax.axis("off")

    fig1.suptitle(
        f"Predicted Similarity Maps P_k(p)  |  mean KL={overall_kl:.3f}  |  "
        f"Kvis={Kvis}  |  T={T}  |  σ={sigma_patch:.2f}  τ={temp_pred:.3f}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.savefig(f"{base}_01_sim_maps.png", bbox_inches="tight", pad_inches=0.01, dpi=150)
    plt.close(fig1)

    # ──────────────────────────────────────────────
    # Figure 2: Token alignment loss (KL) per frame
    # ──────────────────────────────────────────────
    fig2, ax_kl = plt.subplots(figsize=(max(8, 0.6 * T), 4))
    t_all = list(range(T))
    ax_kl.plot(t_all, kl_per_frame, "ko-", linewidth=2, markersize=5, label="mean over kps", zorder=5)
    for ki in range(Kvis):
        k_global = k_ids[ki]
        kl_ki = [kl_vals.get((t, ki), float("nan")) for t in range(T)]
        ax_kl.plot(t_all, kl_ki, "-", color=colors[ki % len(colors)],
                   alpha=0.5, linewidth=1, label=f"kp{k_global}")
    ax_kl.set_xlabel("Frame t", fontsize=10)
    ax_kl.set_ylabel("KL(tgt || P_k)  ↓ better", fontsize=10)
    ax_kl.set_title(
        "Token Alignment Loss per Frame — KL(tgt_k || P_k)\n"
        "P_k(p) = softmax(sim(tok_p, proto_k)/τ),  tgt_k(p) ∝ Gaussian around GT kp",
        fontsize=10,
    )
    ax_kl.legend(fontsize=7, ncol=min(Kvis + 1, 6), loc="upper right")
    ax_kl.grid(True, alpha=0.3)
    fig2.tight_layout()
    plt.savefig(f"{base}_02_kl_per_frame.png", bbox_inches="tight", pad_inches=0.05, dpi=150)
    plt.close(fig2)

    # ──────────────────────────────────────────────
    # Figure 3: Temporal prototype cosine similarity
    # ──────────────────────────────────────────────
    fig3, ax_cos = plt.subplots(figsize=(max(8, 0.6 * T), 4))
    for ki in range(Kvis):
        k_global = k_ids[ki]
        curve    = cosine_curves[ki]
        valid_t  = [t for t, v in enumerate(curve) if not math.isnan(v)]
        valid_v  = [curve[t] for t in valid_t]
        ax_cos.plot(valid_t, valid_v, "o-", color=colors[ki % len(colors)],
                    label=f"kp{k_global}", linewidth=1.5, markersize=4)
    ax_cos.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, label="ref ≥ 0.9")
    ax_cos.set_xlabel("Frame transition (t → t+1)", fontsize=10)
    ax_cos.set_ylabel("proto_k cosine sim  ↑ better", fontsize=10)
    ax_cos.set_title(
        f"Temporal Prototype Consistency — cos(proto_k[t], proto_k[t+1])  |  "
        f"mean cos={overall_cos:.3f}\n"
        "(high = tokens near kp are stable across frames)",
        fontsize=10,
    )
    ax_cos.set_ylim(-0.1, 1.05)
    ax_cos.legend(fontsize=7, ncol=min(Kvis + 1, 6), loc="lower left")
    ax_cos.grid(True, alpha=0.3)
    fig3.tight_layout()
    plt.savefig(f"{base}_03_proto_cosine.png", bbox_inches="tight", pad_inches=0.05, dpi=150)
    plt.close(fig3)

    # ──────────────────────────────────────────────
    # Figure 4: Keypoint trajectory strip
    # ──────────────────────────────────────────────
    fig4, ax_traj = plt.subplots(figsize=(3.2 * n_tframes, 3.5))
    strip_imgs = [_get_img_np(frames_vis, b, ti) for ti in t_indices]
    strip = np.concatenate(strip_imgs, axis=1)
    ax_traj.imshow(strip)

    for ki in range(Kvis):
        k_global = k_ids[ki]
        color    = colors[ki % len(colors)]
        gt_xs, gt_ys, pred_xs, pred_ys = [], [], [], []
        for col, t_idx in enumerate(t_indices):
            if vis_bt[t_idx, k_global] < 0.5:
                continue
            offset_x = col * input_size
            gx = float((kps_gt[b, t_idx, k_global, 0].item() + 1) * 0.5 * (input_size - 1)) + offset_x
            gy = float((kps_gt[b, t_idx, k_global, 1].item() + 1) * 0.5 * (input_size - 1))
            gt_xs.append(gx); gt_ys.append(gy)
            px = float((kps_pred[b, t_idx, k_global, 0].item() + 1) * 0.5 * (input_size - 1)) + offset_x
            py = float((kps_pred[b, t_idx, k_global, 1].item() + 1) * 0.5 * (input_size - 1))
            pred_xs.append(px); pred_ys.append(py)
        if len(gt_xs) > 1:
            ax_traj.plot(gt_xs, gt_ys, "-", color=color, linewidth=1.5, alpha=0.6)
        if len(pred_xs) > 1:
            ax_traj.plot(pred_xs, pred_ys, "--", color=color, linewidth=1.5, alpha=0.8)
        ax_traj.scatter(gt_xs, gt_ys, c=[color], marker="+", s=80, linewidths=2, zorder=5)
        ax_traj.scatter(pred_xs, pred_ys, c=[color], marker="x", s=60, linewidths=1.5, zorder=5)

    ax_traj.set_title("Keypoint Trajectories: GT (+, solid) vs DSNT Pred (×, dashed)", fontsize=12)
    ax_traj.axis("off")
    fig4.tight_layout()
    plt.savefig(f"{base}_04_trajectory.png", bbox_inches="tight", pad_inches=0.05, dpi=150)
    plt.close(fig4)

    # ──────────────────────────────────────────────
    # Figure 5: Target Gaussian distributions
    # ──────────────────────────────────────────────
    fig5 = plt.figure(figsize=(3.2 * n_tframes, 3.5))
    gs5  = GridSpec(1, n_tframes, figure=fig5, hspace=0.05, wspace=0.0)
    fig5.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.02)

    for col, t_idx in enumerate(t_indices):
        ax  = fig5.add_subplot(gs5[0, col])
        img = _get_img_np(frames_vis, b, t_idx)
        ax.imshow(img)
        for ki in range(Kvis):
            if (t_idx, ki) in tgt_maps:
                tgt_up   = _upsample_map(tgt_maps[(t_idx, ki)], (input_size, input_size))
                tgt_norm = (tgt_up - tgt_up.min()) / (tgt_up.max() - tgt_up.min() + 1e-8)
                rgba     = np.zeros((input_size, input_size, 4), dtype=np.float32)
                rgba[..., :3] = colors[ki % len(colors)][:3]
                rgba[..., 3]  = tgt_norm * 0.65
                ax.imshow(rgba)
        if col == 0:
            ax.set_ylabel("Target tgt_k(p)\n(Gaussian)", fontsize=9, fontweight="bold")
        ax.set_title(f"t={t_idx}", fontsize=10)
        ax.axis("off")

    fig5.suptitle("Target Gaussian tgt_k(p) — what the loss drives P_k toward", fontsize=11, fontweight="bold")
    plt.savefig(f"{base}_05_target_gaussian.png", bbox_inches="tight", pad_inches=0.01, dpi=150)
    plt.close(fig5)

    return {
        "mean_proto_cosine": float(overall_cos),
        "mean_kl": float(overall_kl),
        "per_kp_cosine": {
            k_ids[ki]: float(np.nanmean(cosine_curves[ki]))
            for ki in range(Kvis)
            if cosine_curves[ki]
        },
        "Kvis": Kvis,
    }

@torch.no_grad()
def save_adapter_feature_maps(
    out: Dict[str, torch.Tensor],
    frames: torch.Tensor,         # (B, T, C, H, W)
    kps_gt: torch.Tensor,         # (B, T, K, 2) in [-1, 1]
    kps_pred: torch.Tensor,       # (B, T, K, 2) in [-1, 1]
    vis: torch.Tensor,            # (B, T, K)
    cfg: dict,
    save_path: str,
    b: int = 0,
    t_indices: Optional[List[int]] = None,
    max_k_show: int = 4,
) -> Optional[Dict[str, Any]]:
    """
    Cross-frame prototype transfer diagnostic.

    Tests whether the token alignment loss is actually working by building
    a prototype from frame t_ref and querying it against ALL other frames.
    This is NOT tautological: if the loss is working, the prototype from
    frame 0 should still peak at the correct keypoint location in frame T-1.

    For each keypoint k:
      proto_k = normalize(Σ_p tgt_k^{t_ref}(p) · tok_p^{t_ref})  (from reference frame)
      P_k(p)  = softmax(sim(tok_p^{t_query}, proto_k) / τ)        (queried on other frame)

    Layout — 2 figures per run:

    Figure 1 (grid Kvis × n_frames):
      Row 0: same-frame sim maps (tautological baseline — should always be peaked)
      Row 1: cross-frame transfer from t_ref=first frame → each query frame
      Row 2: cross-frame transfer from t_ref=last frame  → each query frame
      (for each keypoint, saved separately)

    Figure 2 (summary):
      Left:  Cross-frame transfer KL vs temporal distance |t_query - t_ref|
      Right: Peak location error (px) vs temporal distance
      (loss working → flat curves; loss not working → curves degrade with distance)
    """
    tok = out.get("adapter_tokens")
    if tok is None:
        return None

    input_size  = int(cfg["data"]["input_size"])
    patch_size  = int(cfg["model"].get("patch_size", 16))
    sigma_patch = float(cfg["loss"].get("sigma_patch", 0.35))
    temp_pred   = float(cfg["loss"].get("temp_pred", 0.07))

    Hp = input_size // patch_size
    Wp = input_size // patch_size
    P  = Hp * Wp

    B, T, P_got, D = tok.shape
    if P_got > P:
        tok = tok[:, :, -P:, :]
    elif P_got < P:
        return None

    K      = vis.shape[2]
    device = tok.device
    dtype  = tok.dtype

    # Select frames
    if t_indices is None:
        if T >= 5:
            t_indices = [0, T // 4, T // 2, 3 * T // 4, T - 1]
        elif T >= 3:
            t_indices = [0, T // 2, T - 1]
        else:
            t_indices = list(range(T))
    t_indices = [t for t in t_indices if t < T]
    n_cols = len(t_indices)

    # Select visible keypoints
    vis_bt = vis[b].float()
    vis_mask = torch.ones(K, device=vis_bt.device, dtype=torch.bool)
    for ti in t_indices:
        vis_mask &= vis_bt[ti] > 0.5
    k_ids = torch.where(vis_mask)[0].tolist()[:max_k_show]
    if not k_ids:
        mid_t = t_indices[len(t_indices) // 2]
        k_ids = torch.where(vis_bt[mid_t] > 0.5)[0].tolist()[:max_k_show]
    if not k_ids:
        return None
    Kvis = len(k_ids)

    # Patch grid in [-1, 1]
    ys_g, xs_g = torch.meshgrid(
        torch.linspace(-1, 1, Hp, device=device, dtype=dtype),
        torch.linspace(-1, 1, Wp, device=device, dtype=dtype),
        indexing="ij",
    )
    patch_xy = torch.stack([xs_g, ys_g], dim=-1).view(P, 2)

    sig2 = float(max(sigma_patch, 1e-6)) ** 2
    temp = float(max(temp_pred, 1e-6))
    eps_kl = 1e-8

    tok_b = F.normalize(tok[b].float(), dim=-1)  # (T, P, D)
    frames_vis = denorm_imagenet_bt(frames)

    base, _ = os.path.splitext(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ── Helper: build prototype from t_ref, query on t_query ──
    def _compute_transfer(ki_local, k_global, t_ref, t_query):
        """Returns (pred_map_np, tgt_map_np, kl, peak_sim, peak_err_px) or None."""
        if vis_bt[t_ref, k_global] < 0.5 or vis_bt[t_query, k_global] < 0.5:
            return None
        kp_ref   = kps_gt[b, t_ref, k_global]
        kp_query = kps_gt[b, t_query, k_global]

        # Gaussian target around ref keypoint → build prototype from ref tokens
        dist2_ref = ((patch_xy - kp_ref.unsqueeze(0)) ** 2).sum(-1)
        tgt_ref   = F.softmax(-0.5 * dist2_ref / sig2, dim=-1)
        proto     = F.normalize((tgt_ref.unsqueeze(-1) * tok_b[t_ref]).sum(0), dim=-1)

        # Query: similarity of proto against query-frame tokens
        sim   = tok_b[t_query] @ proto
        log_p = F.log_softmax(sim / temp, dim=-1)
        pred_p = log_p.exp()

        # Target for query frame (where the keypoint actually is)
        dist2_q = ((patch_xy - kp_query.unsqueeze(0)) ** 2).sum(-1)
        tgt_q   = F.softmax(-0.5 * dist2_q / sig2, dim=-1)

        # KL(tgt_query || pred_from_ref_proto)
        kl = (tgt_q * (torch.log(tgt_q.clamp_min(eps_kl)) - log_p)).sum().item()

        # Peak location error in pixels
        pred_2d  = pred_p.reshape(Hp, Wp)
        peak_idx = pred_2d.argmax()
        peak_r, peak_c = peak_idx // Wp, peak_idx % Wp
        # Convert peak patch coord to pixel
        peak_x_px = float(peak_c) / (Wp - 1) * (input_size - 1)
        peak_y_px = float(peak_r) / (Hp - 1) * (input_size - 1)
        gt_x_px = float((kp_query[0].item() + 1) * 0.5 * (input_size - 1))
        gt_y_px = float((kp_query[1].item() + 1) * 0.5 * (input_size - 1))
        peak_err = math.sqrt((peak_x_px - gt_x_px) ** 2 + (peak_y_px - gt_y_px) ** 2)

        return {
            "pred": pred_p.reshape(Hp, Wp).cpu().numpy(),
            "tgt":  tgt_q.reshape(Hp, Wp).cpu().numpy(),
            "kl":   kl,
            "peak_sim": float(sim.max().item()),
            "peak_err_px": peak_err,
        }

    # ── Reference frames: first and last of selected ──
    t_ref_first = t_indices[0]
    t_ref_last  = t_indices[-1]

    # ── Per-keypoint figure: 3 rows × n_cols ──
    #   Row 0: same-frame (tautological baseline)
    #   Row 1: cross-frame from first ref
    #   Row 2: cross-frame from last ref
    row_labels = [
        "Same-frame\n(baseline)",
        f"Proto from t={t_ref_first}\n→ query each",
        f"Proto from t={t_ref_last}\n→ query each",
    ]

    for ki_local, k_global in enumerate(k_ids):
        fig = plt.figure(figsize=(3.2 * n_cols, 3.0 * 3))
        gs  = GridSpec(3, n_cols, figure=fig, hspace=0.30, wspace=0.02)
        fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.02)

        for col, t_query in enumerate(t_indices):
            for row, t_ref in enumerate([t_query, t_ref_first, t_ref_last]):
                ax = fig.add_subplot(gs[row, col])
                img = _get_img_np(frames_vis, b, t_query)
                ax.imshow(img)

                result = _compute_transfer(ki_local, k_global, t_ref, t_query)
                if result is not None:
                    pred_up = _upsample_map(result["pred"], (input_size, input_size))
                    vmin, vmax = pred_up.min(), pred_up.max()
                    if vmax - vmin > 1e-8:
                        pred_up = (pred_up - vmin) / (vmax - vmin)
                    ax.imshow(pred_up, cmap="hot", alpha=0.55, vmin=0, vmax=1)

                    tgt_up   = _upsample_map(result["tgt"], (input_size, input_size))
                    tgt_norm = (tgt_up - tgt_up.min()) / (tgt_up.max() - tgt_up.min() + 1e-8)
                    ax.contour(tgt_norm, levels=[0.3, 0.6], colors=["cyan"],
                               linewidths=0.8, alpha=0.8)

                    label = f"KL={result['kl']:.2f}"
                    if row > 0:
                        label += f"\nerr={result['peak_err_px']:.0f}px"
                    ax.text(4, input_size - 6, label,
                            color="white", fontsize=6,
                            bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"))

                # GT marker on query frame
                gx = float((kps_gt[b, t_query, k_global, 0].item() + 1) * 0.5 * (input_size - 1))
                gy = float((kps_gt[b, t_query, k_global, 1].item() + 1) * 0.5 * (input_size - 1))
                ax.plot(gx, gy, "g+", markersize=10, markeredgewidth=2)
                px_m = float((kps_pred[b, t_query, k_global, 0].item() + 1) * 0.5 * (input_size - 1))
                py_m = float((kps_pred[b, t_query, k_global, 1].item() + 1) * 0.5 * (input_size - 1))
                ax.plot(px_m, py_m, "rx", markersize=8, markeredgewidth=2)

                if col == 0:
                    ax.set_ylabel(row_labels[row], fontsize=8, fontweight="bold")
                if row == 0:
                    ax.set_title(f"t={t_query}", fontsize=10)
                ax.axis("off")

        fig.suptitle(
            f"Cross-Frame Prototype Transfer — kp{k_global}  |  "
            f"σ={sigma_patch:.2f}  τ={temp_pred:.3f}\n"
            f"Row 0=same-frame (always peaked)  |  "
            f"Rows 1-2=cross-frame (peaked only if loss works)",
            fontsize=10, fontweight="bold",
        )
        plt.savefig(f"{base}_xfer_kp{k_global}.png",
                    bbox_inches="tight", pad_inches=0.02, dpi=150)
        plt.close(fig)

    # ── Summary: transfer quality vs temporal distance ──
    # Compute cross-frame transfer for ALL frame pairs using last adapter tokens
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, Kvis)))

    # Collect per-keypoint: list of (|dt|, kl, peak_err_px)
    xfer_data: Dict[int, List[tuple]] = {ki: [] for ki in range(Kvis)}
    for ki_local, k_global in enumerate(k_ids):
        for t_ref in range(T):
            for t_query in range(T):
                result = _compute_transfer(ki_local, k_global, t_ref, t_query)
                if result is not None:
                    dt = abs(t_query - t_ref)
                    xfer_data[ki_local].append((dt, result["kl"], result["peak_err_px"]))

    fig_sum, (ax_kl, ax_err) = plt.subplots(1, 2, figsize=(13, 5))

    for ki_local, k_global in enumerate(k_ids):
        color = colors[ki_local % len(colors)]
        data = xfer_data[ki_local]
        if not data:
            continue
        # Group by dt
        dt_groups: Dict[int, List[tuple]] = {}
        for dt, kl, err in data:
            dt_groups.setdefault(dt, []).append((kl, err))

        dts = sorted(dt_groups.keys())
        mean_kl  = [float(np.mean([v[0] for v in dt_groups[dt]])) for dt in dts]
        mean_err = [float(np.mean([v[1] for v in dt_groups[dt]])) for dt in dts]

        ax_kl.plot(dts, mean_kl, "o-", color=color, label=f"kp{k_global}",
                   linewidth=1.5, markersize=4)
        ax_err.plot(dts, mean_err, "o-", color=color, label=f"kp{k_global}",
                    linewidth=1.5, markersize=4)

    ax_kl.set_xlabel("Temporal distance |t_ref − t_query|", fontsize=10)
    ax_kl.set_ylabel("KL(tgt_query || P_k^{ref→query})  ↓ better", fontsize=10)
    ax_kl.set_title(
        "Cross-Frame Transfer KL vs Temporal Distance\n"
        "(flat = loss working, tokens consistent across time)\n"
        "(rising = tokens NOT temporally aligned)",
        fontsize=10,
    )
    ax_kl.legend(fontsize=8)
    ax_kl.grid(True, alpha=0.3)

    ax_err.set_xlabel("Temporal distance |t_ref − t_query|", fontsize=10)
    ax_err.set_ylabel("Peak location error (px)  ↓ better", fontsize=10)
    ax_err.set_title(
        "Cross-Frame Peak Error vs Temporal Distance\n"
        "(flat near 0 = loss working, prototype transfers across frames)\n"
        "(rising = prototype fails to localize across time)",
        fontsize=10,
    )
    ax_err.legend(fontsize=8)
    ax_err.grid(True, alpha=0.3)

    fig_sum.tight_layout()
    plt.savefig(f"{base}_xfer_summary.png",
                bbox_inches="tight", pad_inches=0.05, dpi=150)
    plt.close(fig_sum)

    # ── Return summary stats ──
    summary = {}
    for ki_local, k_global in enumerate(k_ids):
        data = xfer_data[ki_local]
        if not data:
            continue
        same_frame = [kl for dt, kl, err in data if dt == 0]
        cross_frame = [kl for dt, kl, err in data if dt > 0]
        far_frame = [kl for dt, kl, err in data if dt >= T // 2]
        same_err = [err for dt, kl, err in data if dt == 0]
        far_err = [err for dt, kl, err in data if dt >= T // 2]
        summary[f"kp{k_global}"] = {
            "same_frame_kl": float(np.mean(same_frame)) if same_frame else None,
            "cross_frame_kl": float(np.mean(cross_frame)) if cross_frame else None,
            "far_frame_kl": float(np.mean(far_frame)) if far_frame else None,
            "same_frame_err_px": float(np.mean(same_err)) if same_err else None,
            "far_frame_err_px": float(np.mean(far_err)) if far_err else None,
        }
    return summary


'''
┌──────────────────────────────────────────────────┐
│ Row 0: kp0 attention maps at t=0, t=2, t=4, t=6, t=7   │
│ Row 1: kp3 attention maps at t=0, t=2, t=4, t=6, t=7   │
│ Row 2: kp5 attention maps at t=0, t=2, t=4, t=6, t=7   │
│ Row 3: kp7 attention maps at t=0, t=2, t=4, t=6, t=7   │
│        (hot overlay = where each prototype attends)      │
│        (green + = GT, red × = predicted)                 │
├──────────────────────────────────────────────────┤
│ Row 4: SCTA Temporal Consistency                         │
│        (cosine similarity between consecutive protos)    │
│        (one line per keypoint, should be > 0.9)          │
├──────────────────────────────────────────────────┤
│ Row 5: Keypoint Trajectories                             │
│        (frame strip with GT and pred paths overlaid)     │
│        (drift visible as diverging dashed lines)         │
├──────────────────────────────────────────────────┤
│ Row 6: Voronoi Ownership Maps                            │
│        (each patch colored by which keypoint "owns" it)  │
│        (shows spatial specialization of tokens)          │
├──────────────────────────────────────────────────┤
│ Row 7: Equivariant Consistency Maps                      │
│        (blue = high cosine sim between t and t+1)        │
│        (red = low sim = features changed unexpectedly)   │
│        (object region should be blue if equiv loss works)│
└──────────────────────────────────────────────────┘
'''


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION FIGURES (paper-quality)
# ══════════════════════════════════════════════════════════════════════════════

# ── Figure 1: Keypoint trajectory strip with configurable frame selection ─────

@torch.no_grad()
def save_trajectory_strip_eval(
    frames: torch.Tensor,           # (B, T, C, H, W) — ImageNet-normalised
    kps_gt: torch.Tensor,           # (B, T, K, 2) in [-1, 1]
    kps_pred: torch.Tensor,         # (B, T, K, 2) in [-1, 1]
    vis: torch.Tensor,              # (B, T, K)  0/1
    save_path: str,
    b: int = 0,
    frame_stride: Optional[int] = None,
    frame_indices: Optional[List[int]] = None,
    max_k_show: int = 11,
) -> None:
    """
    Figure 1 — Keypoint trajectory strip.

    Identical visual style to save_token_diagnostics Figure 4:
      - frames laid side-by-side as a horizontal strip
      - GT keypoints: + marker, solid connecting line
      - Pred keypoints: × marker, dashed connecting line
      - one colour per keypoint (tab10)

    Frame selection (priority order):
      1. frame_indices — explicit list, e.g. [0, 8, 16, 32, 64]
      2. frame_stride  — [0, s, 2s, 3s, …] up to T
      3. default       — 5 frames: [0, T//4, T//2, 3T//4, T-1]
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    frames_vis = denorm_imagenet_bt(frames)
    B, T, C, H_img, W_img = frames.shape
    input_size = H_img

    # ── Resolve frame indices ──────────────────────────────────────────────
    if frame_indices is not None:
        t_sel = [t for t in frame_indices if 0 <= t < T]
    elif frame_stride is not None:
        t_sel = list(range(0, T, frame_stride))
    else:
        if T >= 5:
            t_sel = [0, T // 4, T // 2, 3 * T // 4, T - 1]
        elif T >= 3:
            t_sel = [0, T // 2, T - 1]
        else:
            t_sel = list(range(T))
    if not t_sel:
        t_sel = [0]
    n_tframes = len(t_sel)

    # ── Choose visible keypoints ───────────────────────────────────────────
    vis_bt = vis[b].float()   # (T, K)
    K = vis_bt.shape[1]
    vis_mask = torch.ones(K, dtype=torch.bool)
    for ti in t_sel:
        vis_mask &= vis_bt[ti] > 0.5
    k_ids = torch.where(vis_mask)[0].tolist()[:max_k_show]
    if not k_ids:
        mid = t_sel[len(t_sel) // 2]
        k_ids = torch.where(vis_bt[mid] > 0.5)[0].tolist()[:max_k_show]
    if not k_ids:
        return
    Kvis = len(k_ids)

    # ── Colours — tab10, same as save_token_diagnostics ───────────────────
    colors = plt.colormaps["tab10"](np.linspace(0, 1, min(10, Kvis)))

    # ── Build strip ────────────────────────────────────────────────────────
    strip_imgs = [_get_img_np(frames_vis, b, ti) for ti in t_sel]
    strip = np.concatenate(strip_imgs, axis=1)

    fig, ax = plt.subplots(figsize=(3.2 * n_tframes, 3.5))
    ax.imshow(strip)

    for ki in range(Kvis):
        k_global = k_ids[ki]
        color    = colors[ki % len(colors)]
        gt_xs, gt_ys, pred_xs, pred_ys = [], [], [], []

        for col, t_idx in enumerate(t_sel):
            if vis_bt[t_idx, k_global] < 0.5:
                continue
            offset_x = col * input_size
            gx = float((kps_gt[b, t_idx, k_global, 0].item() + 1) * 0.5 * (input_size - 1)) + offset_x
            gy = float((kps_gt[b, t_idx, k_global, 1].item() + 1) * 0.5 * (input_size - 1))
            gt_xs.append(gx); gt_ys.append(gy)
            px = float((kps_pred[b, t_idx, k_global, 0].item() + 1) * 0.5 * (input_size - 1)) + offset_x
            py = float((kps_pred[b, t_idx, k_global, 1].item() + 1) * 0.5 * (input_size - 1))
            pred_xs.append(px); pred_ys.append(py)

        if len(gt_xs) > 1:
            ax.plot(gt_xs, gt_ys, "-", color=color, linewidth=1.5, alpha=0.6)
        if len(pred_xs) > 1:
            ax.plot(pred_xs, pred_ys, "--", color=color, linewidth=1.5, alpha=0.8)
        ax.scatter(gt_xs, gt_ys, c=[color], marker="+", s=80, linewidths=2, zorder=5)
        ax.scatter(pred_xs, pred_ys, c=[color], marker="x", s=60, linewidths=1.5, zorder=5)

    ax.set_title("Keypoint Trajectories: GT (+, solid) vs DSNT Pred (×, dashed)", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05, dpi=150)
    plt.close(fig)


# ── Figure 2: 3-D pose + keypoints projected onto the full original image ─────

def save_pose_fullframe_grid(
    full_frames: List[np.ndarray],       # list of H_full×W_full×3 uint8 images
    kps_pred_full_px: np.ndarray,        # (N, K, 2) predicted kps in full-frame px
    kps_gt_full_px: Optional[np.ndarray],# (N, K, 2) GT kps in full-frame px, or None
    vis: np.ndarray,                     # (N, K) visibility  0/1
    camera_matrix: np.ndarray,           # (3, 3) intrinsics
    rvecs: Optional[np.ndarray],         # (N, 3) or (N, 3, 1) rodrigues rotation, or None
    tvecs: Optional[np.ndarray],         # (N, 3) or (N, 3, 1) translation (metres), or None
    save_path: str,
    crop_boxes_xywh: Optional[np.ndarray] = None,  # (N, 4) x,y,w,h in full px
    axis_length_m: float = 0.5,
    grid_cols: int = 4,
    frame_title_fn=None,                 # callable(i) -> str, or None
    fig_title: str = "3-D Pose + 2-D Keypoints on Full-Frame Images",
    kp_radius: int = 4,
    show_kp_ids: bool = False,
) -> None:
    """
    Figure 2 — 3-D pose and keypoints projected onto the full original image.

    For each selected frame the function:
      * Draws predicted 2-D keypoints (magenta ×) on the *full-frame* image.
      * Optionally draws GT 2-D keypoints (green ○) for comparison.
      * Optionally overlays a crop-box rectangle showing the model input region.
      * Projects the 3-D pose axes (X=red, Y=green, Z=blue) using rvec/tvec
        and the given camera intrinsics.

    Parameters
    ----------
    full_frames : list of uint8 arrays (H_full, W_full, 3)
        One per selected frame — the original (un-cropped) image.
    kps_pred_full_px : (N, K, 2)
        Predicted keypoint pixel coordinates in full-frame space.
    kps_gt_full_px : (N, K, 2) or None
        GT keypoint pixel coordinates in full-frame space.
    rvecs / tvecs : (N, 3) arrays or None
        Rodrigues rotation + translation from PnP (or from GT annotation).
        If None, pose axes are not drawn.
    camera_matrix : (3, 3)
        Camera intrinsic matrix (OpenCV convention).
    crop_boxes_xywh : (N, 4) or None
        If provided, a white dashed rectangle is drawn on each frame to show
        the region fed into the model.
    axis_length_m : float
        Length (metres) of the projected pose axes.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    N = len(full_frames)
    if N == 0:
        return

    grid_rows = math.ceil(N / grid_cols)
    fig, axes = plt.subplots(grid_rows, grid_cols,
                             figsize=(4.5 * grid_cols, 4.0 * grid_rows))
    if grid_rows == 1 and grid_cols == 1:
        axes = np.array([[axes]])
    elif grid_rows == 1:
        axes = axes[np.newaxis, :]
    elif grid_cols == 1:
        axes = axes[:, np.newaxis]

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    axis_pts_obj = np.float32([
        [0, 0, 0],
        [axis_length_m, 0, 0],   # X  red
        [0, axis_length_m, 0],   # Y  green
        [0, 0, axis_length_m],   # Z  blue
    ])

    for idx in range(grid_rows * grid_cols):
        row = idx // grid_cols
        col = idx % grid_cols
        ax = axes[row, col]

        if idx >= N:
            ax.axis("off")
            continue

        img = full_frames[idx].copy()
        H_f, W_f = img.shape[:2]

        K_kp = kps_pred_full_px.shape[1]

        # ── Draw crop box ──────────────────────────────────────────────────
        if crop_boxes_xywh is not None:
            bx, by, bw, bh = crop_boxes_xywh[idx].astype(int)
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh),
                          (255, 255, 255), 2, cv2.LINE_AA)

        # ── Draw GT keypoints ──────────────────────────────────────────────
        if kps_gt_full_px is not None:
            for k in range(K_kp):
                if vis[idx, k] < 0.5:
                    continue
                gx, gy = int(round(kps_gt_full_px[idx, k, 0])), int(round(kps_gt_full_px[idx, k, 1]))
                cv2.circle(img, (gx, gy), kp_radius + 1, (0, 255, 0), -1, cv2.LINE_AA)
                if show_kp_ids:
                    cv2.putText(img, str(k), (gx + 4, gy - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)

        # ── Draw predicted keypoints ───────────────────────────────────────
        for k in range(K_kp):
            if vis[idx, k] < 0.5:
                continue
            px_k = int(round(kps_pred_full_px[idx, k, 0]))
            py_k = int(round(kps_pred_full_px[idx, k, 1]))
            # Draw X marker
            d = kp_radius
            cv2.line(img, (px_k - d, py_k - d), (px_k + d, py_k + d),
                     (255, 0, 255), 2, cv2.LINE_AA)
            cv2.line(img, (px_k + d, py_k - d), (px_k - d, py_k + d),
                     (255, 0, 255), 2, cv2.LINE_AA)
            if show_kp_ids:
                cv2.putText(img, str(k), (px_k + 4, py_k - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1, cv2.LINE_AA)

        # ── Project pose axes ──────────────────────────────────────────────
        if rvecs is not None and tvecs is not None:
            rvec = np.asarray(rvecs[idx], dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(tvecs[idx], dtype=np.float64).reshape(3, 1)
            try:
                pts_2d, _ = cv2.projectPoints(
                    axis_pts_obj, rvec, tvec, camera_matrix, dist_coeffs)
                pts_2d = pts_2d.reshape(-1, 2).astype(int)
                origin = tuple(pts_2d[0])
                ax_colors = [(255, 50, 50), (50, 255, 50), (50, 100, 255)]  # X Y Z
                for ai, clr in enumerate(ax_colors):
                    tip = tuple(pts_2d[ai + 1])
                    cv2.arrowedLine(img, origin, tip, clr, 2,
                                   cv2.LINE_AA, tipLength=0.15)
            except Exception:
                pass

        ax.imshow(img)
        if frame_title_fn is not None:
            ax.set_title(frame_title_fn(idx), fontsize=9)
        ax.axis("off")

    # ── Legend patch ──────────────────────────────────────────────────────
    legend_elems = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=(0, 1, 0),
                   markersize=7, label="GT keypoint"),
        plt.Line2D([0], [0], marker="x", color=(1, 0, 1), markersize=8,
                   markeredgewidth=2, linestyle="None", label="Pred keypoint"),
    ]
    if rvecs is not None:
        legend_elems += [
            plt.Line2D([0], [0], color=(1, 0.2, 0.2), linewidth=2, label="X axis"),
            plt.Line2D([0], [0], color=(0.2, 1, 0.2), linewidth=2, label="Y axis"),
            plt.Line2D([0], [0], color=(0.2, 0.4, 1), linewidth=2, label="Z axis"),
        ]
    if crop_boxes_xywh is not None:
        legend_elems.append(
            plt.Line2D([0], [0], color="white", linewidth=2,
                       linestyle="-", label="Model crop")
        )

    fig.legend(handles=legend_elems, loc="lower center", ncol=len(legend_elems),
               fontsize=9, framealpha=0.85, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(fig_title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.08, dpi=200)
    plt.close(fig)


# ── Figure 3: Temporal jitter — position vs time step for N sequences ─────────

def save_temporal_jitter_figure(
    seq_data: Dict[str, Dict[str, np.ndarray]],
    save_path: str,
    n_cols: int = 3,
    max_seqs: int = 6,
    components: List[str] = ("Tx", "Ty", "Tz"),
    component_colors: List[str] = ("black", "red", "blue"),
    show_gt: bool = True,
    gt_alpha: float = 0.35,
    pred_alpha: float = 0.85,
    x_label: str = "time step",
    y_label: str = "Position (m)",
    fig_title: str = "Position prediction results with time step",
) -> None:
    """
    Figure 3 — Temporal jitter visualisation.

    Creates a grid of sub-figures (default 2 rows × 3 cols = 6 sequences),
    each showing the predicted translation components (Tx, Ty, Tz) over all
    frames of one sequence, mirroring the jitter plot style used in the paper.

    Parameters
    ----------
    seq_data : dict
        Keys are sequence names.  Each value is a dict with at least:
          ``"pred_trans"``  : np.ndarray (T, 3) — predicted translations
        Optionally:
          ``"gt_trans"``    : np.ndarray (T, 3) — ground-truth translations
                              (drawn as a thin horizontal / smooth reference line)
    components : sequence of str
        Labels for the three translation axes.  Default ("Tx", "Ty", "Tz").
    component_colors : sequence of str
        Matplotlib colour for each component.
    show_gt : bool
        If True and ``"gt_trans"`` is present, draw it as a faint reference.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    seq_names = list(seq_data.keys())[:max_seqs]
    N = len(seq_names)
    if N == 0:
        return

    n_rows = math.ceil(N / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.8 * n_cols, 3.4 * n_rows))
    # Normalise axes to always be 2-D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    comps = list(components)
    clrs  = list(component_colors)

    for idx in range(n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax  = axes[row, col]

        if idx >= N:
            ax.axis("off")
            continue

        seq_name = seq_names[idx]
        entry    = seq_data[seq_name]

        pred_trans = np.asarray(entry["pred_trans"], dtype=np.float32)  # (T, 3)
        T_seq = pred_trans.shape[0]
        time_steps = np.arange(T_seq)

        # GT reference (optional)
        gt_trans = None
        if show_gt and "gt_trans" in entry:
            gt_trans = np.asarray(entry["gt_trans"], dtype=np.float32)

        for ci, (comp, clr) in enumerate(zip(comps, clrs)):
            if ci >= pred_trans.shape[1]:
                continue
            # Predicted — drawn solid with full opacity
            ax.plot(time_steps, pred_trans[:, ci], "-",
                    color=clr, linewidth=1.0, alpha=pred_alpha, label=comp)
            # GT reference — drawn as a faint dashed line
            if gt_trans is not None and ci < gt_trans.shape[1]:
                ax.plot(time_steps, gt_trans[:, ci], "--",
                        color=clr, linewidth=0.8, alpha=gt_alpha)

        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel(y_label, fontsize=8)
        ax.set_title(f"({chr(ord('a') + idx)}) {seq_name}.", fontsize=9)
        ax.legend(fontsize=7, loc="upper right",
                  ncol=len(comps), framealpha=0.7)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        fig_title +
        ("\n(solid = predicted, dashed = GT)" if show_gt else ""),
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.08, dpi=200)
    plt.close(fig)
