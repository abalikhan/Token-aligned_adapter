# utils_new.py (CLEAN)
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from typing import Dict, Any, Tuple, Optional
from losses import softargmax_2d_from_logits
# ---- ImageNet denorm for visualization ----
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1)


# =========================
# utils_new.py  (ADD THIS)
# =========================

# NOTE: this function assumes you already have denorm_imagenet_bt(frames) in utils_new.py
# and that frames are (B,T,3,H,W) ImageNet-normalized float tensors.

@torch.no_grad()
def save_dino_style_kp_attention_overlays_bt(
    out,
    frames,
    kps_gt,
    vis,
    cfg,
    save_dir,
    b: int = 0,
    t: int = 0,
    sigma_patch: float = 0.35,
    temp_pred: float = 0.07,
    max_k_show: int = 8,
    keep_mass: float = 0.60,     # DINO-like: keep top cumulative mass
    alpha: float = 0.45,         # overlay alpha
):
    """
    Option A (recommended): "Keypoint-induced attention" overlays in DINO style.

    Uses adapter/patch tokens tok and keypoint locations to build per-keypoint patch
    distributions P_k(p) (softmax over similarity to kp-prototype), then upsamples
    to image resolution and overlays as attention masks.

    Saves:
      - kp attention overlays:  attn_kp{kid}.png (up to max_k_show visible kps)
      - aggregate overlay:      attn_agg.png (max over shown kps)
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- pick token tensor ---
    tok = None
    for k in ["adapter_tokens", "patch_tokens_prenorm", "patch_tokens"]:
        if isinstance(out, dict) and (k in out) and (out[k] is not None):
            tok = out[k]
            break
    if tok is None:
        return

    # Expect (B,T,P,D) or (B,T,1+nreg+P,D); enforce 4D
    if tok.dim() != 4:
        raise ValueError(f"[attn] Expected tok dim=4 (B,T,P,D), got shape={tuple(tok.shape)}")

    input_size = int(cfg["data"]["input_size"])
    patch_size = int(cfg["model"].get("patch_size", 16))
    Hp = input_size // patch_size
    Wp = input_size // patch_size
    P_expected = Hp * Wp

    B, T, P_got, D = tok.shape
    if P_got < P_expected:
        raise RuntimeError(f"[attn] Too few tokens: got P={P_got}, expected >= {P_expected}")

    # Keep only patch tokens (drop CLS/REG etc)
    if P_got > P_expected:
        tok = tok[:, :, -P_expected:, :]
        P = P_expected
    else:
        P = P_got

    # Patch grid
    Hpatch, Wpatch = Hp, Wp

    # --- image prep ---
    frames_vis = denorm_imagenet_bt(frames)  # must exist in your utils_new
    img = frames_vis[b, t].detach().cpu().permute(1, 2, 0).numpy()
    img_u8 = (img * 255.0).clip(0, 255).astype(np.uint8)

    # --- token prep ---
    tok_bt = tok[b, t]  # (P,D)
    tok_n = F.normalize(tok_bt.float(), dim=-1)  # (P,D)

    # patch coords
    yy, xx = torch.meshgrid(
        torch.arange(Hpatch, device=tok.device),
        torch.arange(Wpatch, device=tok.device),
        indexing="ij",
    )
    grid_xy = torch.stack([xx, yy], dim=-1).view(-1, 2).float()  # (P,2)

    # keypoint coords in patch grid coords
    kp_x = (kps_gt[b, t, :, 0] + 1.0) * 0.5 * (Wpatch - 1)
    kp_y = (kps_gt[b, t, :, 1] + 1.0) * 0.5 * (Hpatch - 1)
    kp_xy = torch.stack([kp_x, kp_y], dim=-1)  # (K,2)
    K = kp_xy.shape[0]
    vis_bt = vis[b, t].float()

    sig2 = float(max(sigma_patch, 1e-6)) ** 2
    temp = float(max(temp_pred, 1e-6))

    # ---- helper: keep-top-mass mask (DINO-style) ----
    def _keep_top_mass(attn_2d: np.ndarray, mass: float) -> np.ndarray:
        mass = float(np.clip(mass, 0.0, 1.0))
        if mass <= 0.0:
            return np.zeros_like(attn_2d)
        if mass >= 1.0:
            return attn_2d
        x = attn_2d.reshape(-1)
        order = np.argsort(-x)  # desc
        xs = x[order]
        csum = np.cumsum(xs)
        tot = csum[-1] + 1e-12
        keep_n = int(np.searchsorted(csum / tot, mass, side="left")) + 1
        keep_idx = order[:keep_n]
        mask = np.zeros_like(x, dtype=np.float32)
        mask[keep_idx] = 1.0
        return (attn_2d * mask.reshape(attn_2d.shape)).astype(np.float32)

    # ---- helper: overlay and save ----
    def _save_overlay(attn_img01: np.ndarray, out_path: str, title: str = ""):
        import matplotlib.pyplot as plt
        attn_img01 = np.clip(attn_img01, 0.0, 1.0)

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(img_u8)
        ax.imshow(attn_img01, cmap="jet", alpha=alpha, vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

    # ---- compute per-k attention maps ----
    attn_up_list = []
    kid_list = []

    for k in range(K):
        if float(vis_bt[k].item()) < 0.5:
            continue

        # GT Gaussian weights around kp in patch grid
        d2 = ((grid_xy - kp_xy[k:k+1]) ** 2).sum(dim=-1)  # (P,)
        w = torch.exp(-0.5 * d2 / sig2)
        w = w / (w.sum().clamp_min(1e-6))

        # kp prototype token
        proto = (w.unsqueeze(-1) * tok_n).sum(dim=0)       # (D,)
        proto = F.normalize(proto, dim=-1)

        # similarity -> distribution over patches
        sim = (tok_n @ proto)                              # (P,)
        Pk = F.softmax(sim / temp, dim=-1)                 # (P,)

        # to patch grid
        Pk_2d = Pk.view(1, 1, Hpatch, Wpatch)

        # upsample to image
        Himg = int(cfg["data"]["input_size"])
        Wimg = int(cfg["data"]["input_size"])
        Pk_up = F.interpolate(Pk_2d, size=(Himg, Wimg), mode="bilinear", align_corners=False)[0, 0]
        Pk_up = Pk_up.detach().cpu().numpy().astype(np.float32)

        # normalize to [0,1] for visualization
        Pk_up = (Pk_up - Pk_up.min()) / (Pk_up.max() - Pk_up.min() + 1e-12)

        # keep top mass (DINO style)
        Pk_up = _keep_top_mass(Pk_up, keep_mass)

        attn_up_list.append(Pk_up)
        kid_list.append(k)

        if len(attn_up_list) >= int(max_k_show):
            break

    if len(attn_up_list) == 0:
        return

    # ---- save per-k overlays ----
    # for attn_img01, kid in zip(attn_up_list, kid_list):
    #     _save_overlay(
    #         attn_img01,
    #         os.path.join(save_dir, f"t{t:03d}_attn_kp{kid:02d}.png"),
    #         title=f"DINO-style kp-attn | kp={kid} | keep_mass={keep_mass:.2f} | sigma={sigma_patch:.2f} | temp={temp_pred:.2f}",
    #     )

    # ---- aggregate overlay (max over shown kps) ----
    agg = np.max(np.stack(attn_up_list, axis=0), axis=0)
    agg = np.clip(agg, 0.0, 1.0).astype(np.float32)
    _save_overlay(
        agg,
        os.path.join(save_dir, f"t{t:03d}_attn_agg.png"),
        title=f"DINO-style kp-attn (agg=max) | Kshown={len(attn_up_list)} | keep_mass={keep_mass:.2f}",
    )

    return {
        "attn_Kshown": int(len(attn_up_list)),
        "attn_keep_mass": float(keep_mass),
        "attn_sigma_patch": float(sigma_patch),
        "attn_temp_pred": float(temp_pred),
    }

#
def _infer_patch_grid(P: int) -> int:
    g = int(round(math.sqrt(P)))
    return g if g * g == P else g

@torch.no_grad()
def denorm_imagenet_bt(frames_btchw: torch.Tensor) -> torch.Tensor:
    mean = _IMAGENET_MEAN.to(frames_btchw.device)
    std  = _IMAGENET_STD.to(frames_btchw.device)
    x = frames_btchw * std + mean
    return x.clamp(0.0, 1.0)

def build_warmup_cosine_scheduler(optimizer, epochs, steps_per_epoch, warmup_epochs=0, lr_min=1e-6):
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_iters = max(0, warmup_epochs * steps_per_epoch)
    decay_iters = max(1, total_steps - warmup_iters)

    warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_iters)
    cosine = CosineAnnealingLR(optimizer, T_max=decay_iters, eta_min=lr_min)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_iters])


def build_warmup_cosine_restart_scheduler(
    optimizer, epochs, steps_per_epoch,
    warmup_epochs=0, lr_min=1e-5,
    restart_period_epochs=25, restart_mult=1,
):
    """
    Linear warmup followed by cosine annealing with warm restarts (SGDR).

    Args:
        restart_period_epochs: Length of the first restart cycle in epochs (T_0).
        restart_mult: Factor by which each subsequent cycle is lengthened (T_mult).
        lr_min: Hard floor for the LR — the scheduler never goes below this,
                so plateau drops still have room to work.
    """
    warmup_iters = max(0, warmup_epochs * steps_per_epoch)
    T_0 = max(1, restart_period_epochs * steps_per_epoch)

    cosine_restart = CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=restart_mult, eta_min=lr_min
    )

    if warmup_iters > 0:
        warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_iters)
        return SequentialLR(optimizer, schedulers=[warmup, cosine_restart], milestones=[warmup_iters])

    return cosine_restart


def freeze_all_but_adapters_and_heads(model: nn.Module, verbose: bool = True):
    """
    Freeze everything, then unfreeze:
      - model.backbone.adapters
      - model.backbone.ffn_adapters
      - model.hm_head
      - model.gates (Parameter)
    Keeps HF DINOv3 backbone frozen.
    """

    # 1) Freeze all params
    for p in model.parameters():
        p.requires_grad = False

    # 2) Unfreeze adapters + head modules (module-based, robust)
    def unfreeze_module(m: nn.Module):
        for p in m.parameters(recurse=True):
            p.requires_grad = True

    # Adapters live inside the wrapper backbone
    if hasattr(model, "backbone"):
        bb = model.backbone

        # Your wrapper: DINOv3backbone
        # HF DINOv3 weights are in bb.backbone  --> keep frozen
        if hasattr(bb, "adapters"):
            unfreeze_module(bb.adapters)
        if hasattr(bb, "attn_adapters"):
            unfreeze_module(bb.attn_adapters)
        if hasattr(bb, "ffn_adapters"):
            unfreeze_module(bb.ffn_adapters)

        # Safety: ensure HF DINOv3 stays frozen
        if hasattr(bb, "backbone"):
            for p in bb.backbone.parameters():
                p.requires_grad = False

    # Heatmap head
    if hasattr(model, "hm_head"):
        unfreeze_module(model.hm_head)

    # Gates are a bare Parameter, not a module
    if hasattr(model, "gates") and isinstance(model.gates, nn.Parameter):
        model.gates.requires_grad = True

    # 3) Summary
    total, trainable = 0, 0
    trainable_names = []
    frozen_names = []

    for n, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            trainable_names.append(n)
        else:
            frozen_names.append(n)

    if verbose:
        print(f"[freeze] trainable {trainable/1e6:.2f}M / {total/1e6:.2f}M ({100*trainable/total:.2f}%)", flush=True)

        # quick sanity checks
        # HF backbone should be fully frozen
        hf_trainable = [n for n in trainable_names if "backbone.backbone" in n]
        if len(hf_trainable) > 0:
            print("[WARN] Some HF DINOv3 params are trainable but should be frozen:")
            for n in hf_trainable[:50]:
                print("   ", n)
            print("   ...")

        # Show buckets
        buckets = {
            "gates": [n for n in trainable_names if n.endswith("gates")],
            "hm_head": [n for n in trainable_names if "hm_head" in n],
            "mhsa_adapters": [n for n in trainable_names if "backbone.adapters" in n or "backbone.attn_adapters" in n],
            "ffn_adapters": [n for n in trainable_names if "backbone.ffn_adapters" in n],
        }
        for k, names in buckets.items():
            print(f"  - {k}: {len(names)} tensors", flush=True)

    return trainable_names


def _norm_to_px(kps_norm: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    kps_norm: (B,T,K,2) in [-1,1]
    returns:  (B,T,K,2) in pixels [0..W-1], [0..H-1]
    """
    x = (kps_norm[..., 0] + 1.0) * 0.5 * (W - 1)
    y = (kps_norm[..., 1] + 1.0) * 0.5 * (H - 1)
    return torch.stack([x, y], dim=-1)

@torch.no_grad()
def pck_from_norm(
    kps_pred_norm: torch.Tensor,
    kps_gt_norm: torch.Tensor,
    vis: torch.Tensor,
    H: int,
    W: int,
    thr_px: float,
) -> torch.Tensor:
    """
    Returns scalar PCK in [0,1] over visible keypoints.
    """
    pred_px = _norm_to_px(kps_pred_norm, H=H, W=W)
    gt_px   = _norm_to_px(kps_gt_norm,   H=H, W=W)

    err = torch.linalg.norm(pred_px - gt_px, dim=-1)  # (B,T,K)
    ok = (err <= float(thr_px)).float()               # (B,T,K)

    # vis: (B,T,K) or (B,T,K,?) -> your vis is (B,T,K)
    vis_f = vis.float()
    num = (ok * vis_f).sum()
    den = vis_f.sum().clamp_min(1.0)
    return num / den

def get_w_token(epoch: int, cfg: Dict[str, Any], total_epochs: int) -> float:
    lcfg = cfg["loss"]
    base = float(lcfg.get("w_token", 0.0))

    sched = lcfg.get("w_token_sched", None)
    if not sched:
        return base

    typ = str(sched.get("type", "none")).lower()
    if typ in ["none", "off"]:
        return base

    start = float(sched.get("start", base))
    end   = float(sched.get("end", base))
    warm  = int(sched.get("warmup_epochs", 0))

    if epoch <= warm and warm > 0:
        return start

    # progress t in [0,1] after warmup
    denom = max(1, total_epochs - warm)
    t = (epoch - warm) / float(denom)
    t = max(0.0, min(1.0, t))

    if typ == "linear":
        return start + (end - start) * t

    if typ == "cosine":
        # cosine from start -> end
        return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * t))

    if typ == "piecewise":
        # list of segments like:
        # w_token_sched:
        #   type: piecewise
        #   segments:
        #     - [1, 5, 0.2]
        #     - [6, 15, 0.1]
        #     - [16, 100, 0.05]
        segs = sched.get("segments", [])
        for a, b, v in segs:
            if int(a) <= epoch <= int(b):
                return float(v)
        return base

    return base

@torch.no_grad()
def save_heatmap_overlay_bt(
    frames: torch.Tensor,          # (B,T,3,H,W) imagenet-normalized
    hm_logits: torch.Tensor,       # (B,T,K,Hm,Wm)
    vis: torch.Tensor | None,      # (B,T,K) optional
    cfg: dict,
    save_path: str,
    b: int = 0,
    t: int = 0,
    reduce: str = "max",           # "max" | "sum" | "mean"
    use_vis: bool = True,
    tau: float = 1.0,              # optional temperature for softmax visualization
):
    """
    Overlay an aggregated heatmap (across K keypoints) on the input frame.

    - Aggregation happens in heatmap space (Hm,Wm).
    - Then upsample to input_size and overlay on RGB.
    - If use_vis=True, only visible keypoints contribute.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    frames_vis = denorm_imagenet_bt(frames)
    img = frames_vis[b, t].detach().cpu().permute(1, 2, 0).numpy()
    img_u8 = (img * 255.0).astype(np.uint8)

    hm_bt = hm_logits[b, t].detach()  # (K,Hm,Wm)
    K, Hm, Wm = hm_bt.shape

    # Optional visibility mask
    if use_vis and (vis is not None):
        v = (vis[b, t].detach() > 0.5).to(hm_bt.device).float().view(K, 1, 1)
        hm_bt = hm_bt * v

    # Convert logits -> positive map for viz
    # Using softmax over spatial dims per keypoint makes maps comparable.
    hm = hm_bt.float().view(K, -1) / max(float(tau), 1e-6)
    hm = torch.softmax(hm, dim=-1).view(K, Hm, Wm)

    # Aggregate across keypoints
    if reduce == "sum":
        agg = hm.sum(dim=0)
    elif reduce == "mean":
        denom = float(K)
        if use_vis and (vis is not None):
            denom = float((vis[b, t] > 0.5).sum().clamp_min(1).item())
        agg = hm.sum(dim=0) / max(denom, 1.0)
    else:  # "max"
        agg = hm.max(dim=0)[0]

    # Normalize
    agg = agg - agg.min()
    agg = agg / (agg.max() + 1e-6)

    # Upsample to input resolution
    H = int(cfg["data"]["input_size"])
    W = int(cfg["data"]["input_size"])
    agg_up = F.interpolate(
        agg.view(1, 1, Hm, Wm),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu().numpy()

    # Save overlay
    try:
        import cv2
        heat = (agg_up * 255.0).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        out_img = cv2.addWeighted(img_u8[..., ::-1], 0.65, heat, 0.35, 0)  # BGR
        cv2.imwrite(save_path, out_img)
    except Exception:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.imshow(agg_up, cmap="jet", alpha=0.35)
        plt.axis("off")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()


@torch.no_grad()
def save_token_similarity_diagnostics_bt(
    out,
    frames,
    kps_gt,
    vis,
    cfg,
    save_path,
    b: int = 0,
    t: int = 0,
    sigma_patch: float = 0.35,   # match loss default (your old viz used 1.25 = too broad)
    temp_pred: float = 0.07,
    max_k_show: int = 8,
):
    """
    Saves a single figure containing:
      - per-keypoint predicted token attention P_k(p) overlays (up to max_k_show visible kps)
      - aggregate max-over-k map
      - patch-wise keypoint-specialization: peak_map and entropy_map

    Predicted distribution per keypoint:
      proto_k from GT Gaussian around kp, then
      P_k(p) = softmax( sim(tok_p, proto_k) / temp_pred )

    Also computes:
      - peak_k = max_p P_k(p)
      - H_k = entropy(P_k) / log(P) in [0,1]
      - peak_map(p) = max_k q_k(p) where q_k(p) = P_k(p)/sum_j P_j(p)
      - entropy_map(p) = entropy(q(.|p)) / log(K) in [0,1]
    """

    # --- pick token tensor ---
    tok = None
    for k in ["adapter_tokens", "patch_tokens_prenorm", "patch_tokens"]:
        if k in out and out[k] is not None:
            tok = out[k]
            break
    if tok is None:
        return

    # If tokens include CLS/REG, slice to patches if needed (based on input/patch)
    # Prefer cfg to infer expected P
    input_size = int(cfg["data"]["input_size"])
    patch_size = int(cfg["model"].get("patch_size", 16))
    Hp = input_size // patch_size
    Wp = input_size // patch_size
    P_expected = Hp * Wp

    B, T, P_got, D = tok.shape
    if P_got >= P_expected:
        tok = tok[:, :, -P_expected:, :]
        P = P_expected
    else:
        P = P_got  # fallback; will infer grid from sqrt(P)

    # --- infer patch grid ---
    g = int(round(math.sqrt(P)))
    if g * g != P:
        # last resort: don't crash; just bail
        return
    Hpatch = Wpatch = g

    # --- image prep ---
    Himg = int(cfg["data"]["input_size"])
    Wimg = int(cfg["data"]["input_size"])

    # denorm_imagenet_bt must exist in your codebase
    frames_vis = denorm_imagenet_bt(frames)
    img = frames_vis[b, t].detach().cpu().permute(1, 2, 0).numpy()
    img_u8 = (img * 255.0).clip(0, 255).astype(np.uint8)

    # --- token prep ---
    tok_bt = tok[b, t]  # (P,D)
    tok_n = F.normalize(tok_bt.float(), dim=-1)  # (P,D)

    # --- patch coordinates in patch grid (not normalized) ---
    yy, xx = torch.meshgrid(
        torch.arange(Hpatch, device=tok.device),
        torch.arange(Wpatch, device=tok.device),
        indexing="ij",
    )
    grid_xy = torch.stack([xx, yy], dim=-1).view(-1, 2).float()  # (P,2)

    # --- keypoint coords in patch-grid coordinates ---
    # kps_gt is in [-1,1] (normalized), convert to [0..Wpatch-1], [0..Hpatch-1]
    kp_x = (kps_gt[b, t, :, 0] + 1.0) * 0.5 * (Wpatch - 1)
    kp_y = (kps_gt[b, t, :, 1] + 1.0) * 0.5 * (Hpatch - 1)
    kp_xy = torch.stack([kp_x, kp_y], dim=-1)  # (K,2)

    K = kp_xy.shape[0]
    vis_bt = vis[b, t].float()  # (K,)

    # --- build per-keypoint predicted maps P_k(p) ---
    P_list = []
    k_ids = []
    peak_k_list = []
    ent_k_list = []

    sig2 = float(max(sigma_patch, 1e-6)) ** 2
    temp = float(max(temp_pred, 1e-6))

    for k in range(K):
        if float(vis_bt[k].item()) < 0.5:
            continue

        # GT Gaussian weights around kp (in patch grid coords)
        d2 = ((grid_xy - kp_xy[k:k+1]) ** 2).sum(dim=-1)  # (P,)
        w = torch.exp(-0.5 * d2 / sig2)                   # (P,)
        w = w / (w.sum().clamp_min(1e-6))

        # prototype token for kp
        proto = (w.unsqueeze(-1) * tok_n).sum(dim=0)      # (D,)
        proto = F.normalize(proto, dim=-1)

        # similarity per patch
        sim = (tok_n @ proto)                             # (P,)

        # predicted distribution over patches
        logP = F.log_softmax(sim / temp, dim=-1)          # (P,)
        Pk = logP.exp()                                   # (P,)
        P_list.append(Pk)
        k_ids.append(k)

        # diagnostics
        peak_k = float(Pk.max().item())
        H_k = float((-(Pk * logP).sum() / math.log(P)).item())  # normalized entropy in [0,1]
        peak_k_list.append(peak_k)
        ent_k_list.append(H_k)

        if len(P_list) >= max_k_show:
            break

    if len(P_list) == 0:
        return

    Pk_stack = torch.stack(P_list, dim=0)   # (Kvis,P)
    Kvis = Pk_stack.shape[0]

    # --- aggregate maps over keypoints (over patches) ---
    # max-over-keypoints attention on patches
    P_max = Pk_stack.max(dim=0).values.view(1, 1, Hpatch, Wpatch)  # (1,1,g,g)

    # keypoint-specialization per patch: normalize across keypoints at each patch
    # q_k(p) = P_k(p) / sum_j P_j(p)
    denom = Pk_stack.sum(dim=0).clamp_min(1e-8)    # (P,)
    q = (Pk_stack / denom.unsqueeze(0))            # (Kvis,P)

    peak_map = q.max(dim=0).values.view(1, 1, Hpatch, Wpatch)  # (1,1,g,g)
    ent_map = (-(q * (q.clamp_min(1e-8).log())).sum(dim=0) / math.log(Kvis)).view(1, 1, Hpatch, Wpatch)

    # --- upsample maps to image size (do NOT per-map min/max normalize) ---
    def upsample01(x_11gg):
        # x already in [0,1] for peak_map/ent_map; for P maps in (0,1) but not [0,1] range-fixed.
        x_up = F.interpolate(x_11gg, size=(Himg, Wimg), mode="bilinear", align_corners=False)[0, 0]
        return x_up.detach().cpu().numpy()

    P_max_up   = upsample01(P_max)
    peak_up    = upsample01(peak_map)
    ent_up     = upsample01(ent_map)

    # Per-k overlays
    Pk_up_list = []
    for i in range(Kvis):
        Pk_i = Pk_stack[i].view(1, 1, Hpatch, Wpatch)
        Pk_up_list.append(upsample01(Pk_i))

    # --- plotting ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    import matplotlib.pyplot as plt

    # layout: keypoint maps + 3 diagnostics
    # rows: ceil((Kvis + 3) / cols)
    cols = 4
    n_panels = Kvis + 3
    rows = int(math.ceil(n_panels / cols))

    fig = plt.figure(figsize=(4 * cols, 4 * rows))

    # helper to overlay
    def _overlay(ax, heat, title):
        ax.imshow(img)
        ax.imshow(heat, cmap="jet", alpha=0.35, vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # per-k maps (title includes kp id + metrics)
    for i in range(Kvis):
        ax = fig.add_subplot(rows, cols, i + 1)
        kid = k_ids[i]
        _overlay(ax, Pk_up_list[i], f"kp{kid}  peak={peak_k_list[i]:.3f}  H={ent_k_list[i]:.3f}")

    # diagnostics
    ax = fig.add_subplot(rows, cols, Kvis + 1)
    _overlay(ax, P_max_up, "max_k Pk(p) (aggregate)")

    ax = fig.add_subplot(rows, cols, Kvis + 2)
    _overlay(ax, peak_up, f"peak_map (keypoint ownership)  Kvis={Kvis}")

    ax = fig.add_subplot(rows, cols, Kvis + 3)
    _overlay(ax, ent_up, "entropy_map over keypoints (0=sharp,1=shared)")

    # global summary in suptitle
    mean_peak_k = float(np.mean(peak_k_list))
    mean_ent_k  = float(np.mean(ent_k_list))
    mean_peak_map = float(np.mean(peak_up))
    mean_ent_map  = float(np.mean(ent_up))
    fig.suptitle(
        f"Token diagnostics | sigma_patch={sigma_patch:.2f} temp_pred={temp_pred:.2f} | "
        f"mean(peak_k)={mean_peak_k:.3f} mean(H_k)={mean_ent_k:.3f} | "
        f"mean(peak_map)={mean_peak_map:.3f} mean(ent_map)={mean_ent_map:.3f}",
        fontsize=12
    )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    # Optional: return metrics if caller wants to log them
    return {
        "tok_mean_peak_k": mean_peak_k,
        "tok_mean_ent_k": mean_ent_k,
        "tok_mean_peak_map": mean_peak_map,
        "tok_mean_ent_map": mean_ent_map,
        "tok_Kvis": int(Kvis),
    }


# --------------------------------------
# Good visualization for Proto loss usage 
# ---------------------------------------
@torch.no_grad()
def save_temporal_proto_diagnostics(
    out: dict,
    frames: torch.Tensor,       # (B,T,C,H,W)
    kps_gt: torch.Tensor,       # (B,T,K,2) in [-1,1]
    kps_pred: torch.Tensor,     # (B,T,K,2) in [-1,1] — from out["keypoints"]
    vis: torch.Tensor,          # (B,T,K)
    cfg: dict,
    save_path: str,
    b: int = 0,
    sigma_patch: float = 0.15,
    max_k_show: int = 8,
):
    """
    Comprehensive temporal proto diagnostics for paper figures.
    
    Generates a multi-panel figure showing:
      Row 1: Per-keypoint proto attention maps at 3 selected frames (early, mid, late in clip)
      Row 2: Proto cosine similarity across time (line plot) + trajectory overlay on frame strip
      Row 3: Keypoint ownership (peak_map) at the same 3 frames — showing spatial specialisation evolution
    
    This directly visualises the effect of temporal_proto_consistency_loss.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    # --- setup ---
    input_size = int(cfg["data"]["input_size"])
    patch_size = int(cfg["model"].get("patch_size", 16))
    Hp = input_size // patch_size
    Wp = input_size // patch_size
    P = Hp * Wp
    
    # Get tokens
    tok = None
    for key in ["adapter_tokens", "patch_tokens_prenorm", "patch_tokens"]:
        if key in out and out[key] is not None:
            tok = out[key]
            break
    if tok is None:
        return None
    
    B, T, P_got, D = tok.shape
    if P_got > P:
        tok = tok[:, :, -P:, :]
    elif P_got < P:
        return None
    
    # Denormalize frames for display
    frames_vis = denorm_imagenet_bt(frames)  # (B,T,C,H,W) in [0,1]
    
    vis_bt = vis[b].float()  # (T, K)
    K = vis_bt.shape[1]
    
    # Select frames to visualise: early, middle, late
    if T >= 5:
        t_indices = [0, T // 4, T // 2, 3 * T // 4, T - 1]
    elif T >= 3:
        t_indices = [0, T // 2, T - 1]
    else:
        t_indices = list(range(T))
    n_tframes = len(t_indices)
    
    # --- find visible keypoints (visible in ALL selected frames) ---
    vis_mask = torch.ones(K, device=vis.device, dtype=torch.bool)
    for ti in t_indices:
        vis_mask &= (vis_bt[ti] > 0.5)
    k_ids = torch.where(vis_mask)[0].tolist()[:max_k_show]
    
    if len(k_ids) == 0:
        # fallback: keypoints visible in at least the middle frame
        mid_t = t_indices[len(t_indices) // 2]
        k_ids = torch.where(vis_bt[mid_t] > 0.5)[0].tolist()[:max_k_show]
    
    if len(k_ids) == 0:
        return None
    
    Kvis = len(k_ids)
    
    # --- patch grid coordinates ---
    device = tok.device
    dtype = tok.dtype
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, Hp, device=device, dtype=dtype),
        torch.linspace(-1, 1, Wp, device=device, dtype=dtype),
        indexing="ij",
    )
    patch_xy = torch.stack([xs, ys], dim=-1).view(1, P, 2)  # (1,P,2)
    
    sig2 = float(max(sigma_patch, 1e-6)) ** 2
    
    # --- compute prototypes and attention maps for all frames ---
    # protos[t][k] = normalised prototype vector
    # attn_maps[t][k] = attention map (P,) as numpy
    protos_all = {}    # dict of (t, k_idx) -> (D,) tensor
    attn_maps = {}     # dict of (t, k_idx) -> (Hp, Wp) numpy
    
    for t in range(T):
        tok_t = F.normalize(tok[b, t].float(), dim=-1)  # (P, D)
        kp_t = kps_gt[b, t]  # (K, 2)
        
        for ki, k in enumerate(k_ids):
            if vis_bt[t, k] < 0.5:
                continue
            
            # Gaussian weights around GT keypoint
            kp_xy_k = kp_t[k:k+1]  # (1, 2) in [-1,1]
            # Convert to patch grid for distance
            dist2 = ((patch_xy - kp_xy_k.unsqueeze(0)) ** 2).sum(dim=-1).squeeze(0)  # (P,)
            w = torch.exp(-0.5 * dist2 / sig2)
            w = w / w.sum().clamp_min(1e-6)
            
            # Prototype
            proto = (w.unsqueeze(-1) * tok_t).sum(dim=0)  # (D,)
            proto = F.normalize(proto, dim=-1)
            protos_all[(t, ki)] = proto
            
            # Attention map: cosine similarity to prototype
            sim = tok_t @ proto  # (P,)
            attn = F.softmax(sim / 0.07, dim=-1)  # sharper temp for visualisation
            attn_maps[(t, ki)] = attn.view(Hp, Wp).cpu().numpy()
    
    # --- compute temporal proto cosine similarities ---
    # cosine_curves[ki] = list of cosine sims between consecutive frames
    cosine_curves = {}
    for ki in range(Kvis):
        curve = []
        for t in range(T - 1):
            if (t, ki) in protos_all and (t + 1, ki) in protos_all:
                cos = float((protos_all[(t, ki)] * protos_all[(t + 1, ki)]).sum().item())
                curve.append(cos)
            else:
                curve.append(float('nan'))
        cosine_curves[ki] = curve
    
    # --- prepare images ---
    def get_img(t_idx):
        img = frames_vis[b, t_idx].detach().cpu().permute(1, 2, 0).numpy()
        return (img * 255).clip(0, 255).astype(np.uint8)
    
    def upsample_attn(attn_hw, size=(input_size, input_size)):
        a = torch.from_numpy(attn_hw).float().unsqueeze(0).unsqueeze(0)
        a = F.interpolate(a, size=size, mode="bilinear", align_corners=False)
        return a[0, 0].numpy()
    
    # =====================================================
    # FIGURE LAYOUT
    # =====================================================
    # Row 1: Proto attention maps for each selected keypoint across selected frames
    #         Columns = frames, one sub-figure per keypoint
    # Row 2: (left) Cosine similarity curves, (right) trajectory overlay on frame strip
    # Row 3: Ownership maps (peak_map) at selected frames
    
    fig = plt.figure(figsize=(5 * n_tframes, 5 * (Kvis + 3)))
    gs = GridSpec(Kvis + 3, n_tframes, figure=fig, hspace=0.3, wspace=0.15)
    
    # ---- Row 1..Kvis: per-keypoint attention maps across frames ----
    for ki_row, ki in enumerate(range(Kvis)):
        k_id = k_ids[ki]
        for col, t_idx in enumerate(t_indices):
            ax = fig.add_subplot(gs[ki_row, col])
            img = get_img(t_idx)
            ax.imshow(img)
            
            if (t_idx, ki) in attn_maps:
                attn_up = upsample_attn(attn_maps[(t_idx, ki)])
                # Normalise per-map for visibility
                vmin, vmax = attn_up.min(), attn_up.max()
                if vmax - vmin > 1e-8:
                    attn_up = (attn_up - vmin) / (vmax - vmin)
                ax.imshow(attn_up, cmap="hot", alpha=0.5, vmin=0, vmax=1)
            
            # Mark GT keypoint
            kp_x = float((kps_gt[b, t_idx, k_id, 0].item() + 1) * 0.5 * (input_size - 1))
            kp_y = float((kps_gt[b, t_idx, k_id, 1].item() + 1) * 0.5 * (input_size - 1))
            ax.plot(kp_x, kp_y, 'g+', markersize=12, markeredgewidth=2)
            
            # Mark predicted keypoint
            kp_px = float((kps_pred[b, t_idx, k_id, 0].item() + 1) * 0.5 * (input_size - 1))
            kp_py = float((kps_pred[b, t_idx, k_id, 1].item() + 1) * 0.5 * (input_size - 1))
            ax.plot(kp_px, kp_py, 'rx', markersize=10, markeredgewidth=2)
            
            if col == 0:
                ax.set_ylabel(f"kp{k_id}", fontsize=11, fontweight='bold')
            if ki_row == 0:
                ax.set_title(f"t={t_idx}", fontsize=11)
            ax.axis("off")
    
    # ---- Row Kvis+1: Cosine similarity curves ----
    ax_cos = fig.add_subplot(gs[Kvis, :])
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, Kvis)))
    for ki in range(Kvis):
        k_id = k_ids[ki]
        curve = cosine_curves[ki]
        valid_t = [t for t, v in enumerate(curve) if not np.isnan(v)]
        valid_v = [curve[t] for t in valid_t]
        ax_cos.plot(valid_t, valid_v, 'o-', color=colors[ki % len(colors)],
                    label=f"kp{k_id}", linewidth=1.5, markersize=4)
    
    ax_cos.set_xlabel("Frame transition (t → t+1)", fontsize=11)
    ax_cos.set_ylabel("Proto cosine similarity", fontsize=11)
    ax_cos.set_title("Temporal Proto Consistency (higher = more stable across frames)", fontsize=12)
    ax_cos.set_ylim(-0.1, 1.05)
    ax_cos.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='target ≥ 0.9')
    ax_cos.legend(fontsize=8, ncol=min(Kvis, 4), loc='lower left')
    ax_cos.grid(True, alpha=0.3)
    
    # ---- Row Kvis+2: Keypoint trajectory overlay on frame strip ----
    ax_traj = fig.add_subplot(gs[Kvis + 1, :])
    
    # Create a horizontal strip of selected frames
    strip_imgs = [get_img(ti) for ti in t_indices]
    strip = np.concatenate(strip_imgs, axis=1)  # (H, W*n_tframes, 3)
    ax_traj.imshow(strip)
    
    for ki in range(Kvis):
        k_id = k_ids[ki]
        color = colors[ki % len(colors)]
        
        gt_xs, gt_ys = [], []
        pred_xs, pred_ys = [], []
        
        for col, t_idx in enumerate(t_indices):
            if vis_bt[t_idx, k_id] < 0.5:
                continue
            offset_x = col * input_size
            
            gx = float((kps_gt[b, t_idx, k_id, 0].item() + 1) * 0.5 * (input_size - 1)) + offset_x
            gy = float((kps_gt[b, t_idx, k_id, 1].item() + 1) * 0.5 * (input_size - 1))
            gt_xs.append(gx)
            gt_ys.append(gy)
            
            px = float((kps_pred[b, t_idx, k_id, 0].item() + 1) * 0.5 * (input_size - 1)) + offset_x
            py = float((kps_pred[b, t_idx, k_id, 1].item() + 1) * 0.5 * (input_size - 1))
            pred_xs.append(px)
            pred_ys.append(py)
        
        # Draw trajectories
        if len(gt_xs) > 1:
            ax_traj.plot(gt_xs, gt_ys, '-', color=color, linewidth=1.5, alpha=0.6)
        if len(pred_xs) > 1:
            ax_traj.plot(pred_xs, pred_ys, '--', color=color, linewidth=1.5, alpha=0.8)
        
        # Draw points
        ax_traj.scatter(gt_xs, gt_ys, c=[color], marker='+', s=80, linewidths=2, zorder=5)
        ax_traj.scatter(pred_xs, pred_ys, c=[color], marker='x', s=60, linewidths=1.5, zorder=5)
    
    ax_traj.set_title(
        "Keypoint Trajectories: GT (+, solid) vs Pred (×, dashed)",
        fontsize=12
    )
    ax_traj.axis("off")
    
    # ---- Row Kvis+3: Ownership maps at selected frames ----
    for col, t_idx in enumerate(t_indices):
        ax = fig.add_subplot(gs[Kvis + 2, col])
        img = get_img(t_idx)
        ax.imshow(img)
        
        # Build ownership map: for each patch, which keypoint "owns" it most?
        maps_at_t = []
        for ki in range(Kvis):
            if (t_idx, ki) in attn_maps:
                maps_at_t.append(attn_maps[(t_idx, ki)])
            else:
                maps_at_t.append(np.zeros((Hp, Wp)))
        
        if len(maps_at_t) > 0:
            stacked = np.stack(maps_at_t, axis=0)  # (Kvis, Hp, Wp)
            
            # Ownership: argmax over keypoints at each patch
            owner = np.argmax(stacked, axis=0)  # (Hp, Wp) values in [0, Kvis-1]
            confidence = np.max(stacked, axis=0)  # (Hp, Wp)
            
            # Normalise confidence for alpha
            conf_norm = confidence / (confidence.max() + 1e-8)
            
            # Create RGB ownership map
            owner_rgb = np.zeros((Hp, Wp, 3))
            for ki in range(Kvis):
                mask = (owner == ki)
                owner_rgb[mask] = colors[ki % len(colors)][:3]
            
            # Upsample
            owner_rgb_up = np.stack([
                upsample_attn(owner_rgb[:, :, c]) for c in range(3)
            ], axis=-1)
            conf_up = upsample_attn(conf_norm)
            
            ax.imshow(owner_rgb_up, alpha=conf_up * 0.6)
        
        ax.set_title(f"Ownership t={t_idx}", fontsize=10)
        ax.axis("off")
    
    # ---- Global summary ----
    mean_cosines = []
    for ki in range(Kvis):
        valid = [v for v in cosine_curves[ki] if not np.isnan(v)]
        if valid:
            mean_cosines.append(np.mean(valid))
    
    overall_cos = np.mean(mean_cosines) if mean_cosines else 0.0
    
    fig.suptitle(
        f"Temporal Proto Diagnostics | σ={sigma_patch:.2f} | "
        f"mean proto cosine={overall_cos:.3f} | Kvis={Kvis} | T={T}",
        fontsize=14, fontweight='bold', y=1.01
    )
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.3, dpi=150)
    plt.close(fig)
    
    return {
        "mean_proto_cosine": float(overall_cos),
        "per_kp_cosine": {k_ids[ki]: float(np.nanmean(cosine_curves[ki])) 
                          for ki in range(Kvis) if cosine_curves[ki]},
        "Kvis": Kvis,
    }

 # old version creating blobs        
# @torch.no_grad()
# def save_token_similarity_overlay_bt(out, frames, kps_gt, vis, cfg, save_path, b=0, t=0, sigma_patch=1.25):
#     # prefer adapter tokens, fallback patch tokens
#     tok = None
#     for k in ["adapter_tokens", "patch_tokens_prenorm", "patch_tokens"]:
#         if k in out:
#             tok = out[k]
#             break
#     if tok is None:
#         return

#     os.makedirs(os.path.dirname(save_path), exist_ok=True)

#     B, T, P, D = tok.shape
#     Himg = int(cfg["data"]["input_size"])
#     Wimg = int(cfg["data"]["input_size"])

#     g = _infer_patch_grid(P)
#     Hpatch = Wpatch = g

#     tok_bt = tok[b, t]                           # (P,D)
#     tok_norm = F.normalize(tok_bt.float(), dim=-1)

#     yy, xx = torch.meshgrid(
#         torch.arange(Hpatch, device=tok.device),
#         torch.arange(Wpatch, device=tok.device),
#         indexing="ij",
#     )
#     grid_xy = torch.stack([xx, yy], dim=-1).view(-1, 2).float()  # (P,2)

#     kp_x = (kps_gt[b, t, :, 0] + 1) * 0.5 * (Wpatch - 1)
#     kp_y = (kps_gt[b, t, :, 1] + 1) * 0.5 * (Hpatch - 1)
#     kp_xy = torch.stack([kp_x, kp_y], dim=-1)

#     sim_maps = []
#     K = kp_xy.shape[0]
#     for k in range(K):
#         if float(vis[b, t, k].item()) < 0.5:
#             continue
#         d2 = ((grid_xy - kp_xy[k:k+1]) ** 2).sum(dim=-1)
#         w = torch.exp(-0.5 * d2 / (sigma_patch ** 2))
#         w = w / (w.sum() + 1e-6)
#         proto = (w.unsqueeze(-1) * tok_bt).sum(dim=0)
#         proto = F.normalize(proto.float(), dim=-1)
#         sim = (tok_norm @ proto)                 # (P,)
#         sim_maps.append(sim.view(1, 1, Hpatch, Wpatch))

#     if len(sim_maps) == 0:
#         return

#     sim_map = torch.cat(sim_maps, dim=0).max(dim=0, keepdim=True)[0]  # (1,1,g,g)
#     sm = sim_map[0, 0]
#     sm = (sm - sm.min()) / (sm.max() - sm.min() + 1e-6)
#     sm = sm.view(1, 1, Hpatch, Wpatch)
#     sm_up = F.interpolate(sm, size=(Himg, Wimg), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()

#     frames_vis = denorm_imagenet_bt(frames)
#     img = frames_vis[b, t].detach().cpu().permute(1, 2, 0).numpy()
#     img_u8 = (img * 255.0).astype(np.uint8)

#     try:
#         import cv2
#         heat = (sm_up * 255.0).astype(np.uint8)
#         heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
#         out_img = cv2.addWeighted(img_u8[..., ::-1], 0.65, heat, 0.35, 0)
#         cv2.imwrite(save_path, out_img)
#     except Exception:
#         import matplotlib.pyplot as plt
#         plt.figure(figsize=(5, 5))
#         plt.imshow(img)
#         plt.imshow(sm_up, cmap="jet", alpha=0.35)
#         plt.axis("off")
#         plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
#         plt.close()

@torch.no_grad()
def save_kp_overlay_bt(frames, kps_gt_norm, hm_logits, vis, cfg, save_path, b=0, t=0):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    frames_vis = denorm_imagenet_bt(frames)
    f = frames_vis[b, t].detach().cpu()

    B, T, K, Hm, Wm = hm_logits.shape
    H = int(cfg["data"]["input_size"])
    W = int(cfg["data"]["input_size"])

    from losses import softargmax_2d_from_logits
    hm_ = hm_logits[b:b+1, t:t+1]
    pred_px_hm = softargmax_2d_from_logits(hm_)[0, 0]  # (K,2) hm px

    sx, sy = float(W) / float(Wm), float(H) / float(Hm)
    pred_px_img = pred_px_hm.clone()
    pred_px_img[:, 0] *= sx
    pred_px_img[:, 1] *= sy

    gx = ((kps_gt_norm[b, t, :, 0] + 1.0) * 0.5) * (W - 1.0)
    gy = ((kps_gt_norm[b, t, :, 1] + 1.0) * 0.5) * (H - 1.0)
    gt_px_img = torch.stack([gx, gy], dim=-1).cpu()

    vis_ = (vis[b, t].detach().cpu() > 0).float() if vis is not None else None

    img = (f.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8).copy()

    try:
        import cv2
        col_gt = (0, 255, 0)
        col_pr = (255, 0, 255)
        r = max(2, int(0.004 * max(H, W)))
        th = max(1, int(0.002 * max(H, W)))

        for i in range(gt_px_img.shape[0]):
            px_gt = (int(gt_px_img[i, 0].item()), int(gt_px_img[i, 1].item()))
            px_pr = (int(pred_px_img[i, 0].item()), int(pred_px_img[i, 1].item()))
            if vis_ is None or vis_[i] > 0.5:
                cv2.circle(img, px_gt, r, col_gt, -1, lineType=cv2.LINE_AA)
                cv2.circle(img, px_pr, r, col_pr, th, lineType=cv2.LINE_AA)
        cv2.imwrite(save_path, img)
    except Exception:
        from PIL import Image
        Image.fromarray(img).save(save_path)

def collate_temporal_chunks(batch, T_chunk: int, drop_last: bool = True):
    """
    Same behavior as your current collate_temporal_chunks but only what we need.
    """

    out_frames, out_kps, out_gt_q, out_gt_r, out_vis, out_hm = [], [], [], [], [], []
    out_bboxs, out_meta = [], []
    out_seq_names, out_frame_ids = [], []

    def _chunk_or_pad_tensor(x: torch.Tensor, T_eff: int):
        # x: (T, ...)
        T = x.shape[0]
        if drop_last:
            x = x[:T_eff]
            n_chunks = T_eff // T_chunk
            return x.view(n_chunks, T_chunk, *x.shape[1:])
        else:
            n_chunks = (T_eff + T_chunk - 1) // T_chunk
            T_pad = n_chunks * T_chunk
            if T_pad > T_eff:
                pad = T_pad - T_eff
                x = torch.cat([x, x[-1:].repeat(pad, *([1] * (x.ndim - 1)))], dim=0)
            return x.view(n_chunks, T_chunk, *x.shape[1:])

    def _chunk_or_pad_paths(paths: list, T_eff: int):
        # paths: list length T
        if drop_last:
            paths = paths[:T_eff]
            n_chunks = T_eff // T_chunk
            return [paths[i*T_chunk:(i+1)*T_chunk] for i in range(n_chunks)]
        else:
            n_chunks = (T_eff + T_chunk - 1) // T_chunk
            T_pad = n_chunks * T_chunk
            if T_pad > T_eff:
                pad = T_pad - T_eff
                paths = paths + [paths[-1]] * pad
            return [paths[i*T_chunk:(i+1)*T_chunk] for i in range(n_chunks)]
        

    for sample in batch:
        frames, kps, gt_q, gt_r, vis, hm, frame_ids, crop_xywh, meta, seq_name = sample

        # frame_ids can be [[...]]; unwrap
        if isinstance(frame_ids, (list, tuple)) and len(frame_ids) == 1 and isinstance(frame_ids[0], (list, tuple)):
            frame_ids = frame_ids[0]

        T = frames.shape[0]
        if drop_last:
            T_eff = (T // T_chunk) * T_chunk
        else:
            T_eff = T

        if T_eff == 0:
            continue

        frames_c   = _chunk_or_pad_tensor(frames,   T_eff)
        kps_c      = _chunk_or_pad_tensor(kps,      T_eff)
        q_c        = _chunk_or_pad_tensor(gt_q,     T_eff)
        r_c        = _chunk_or_pad_tensor(gt_r,     T_eff)
        vis_c      = _chunk_or_pad_tensor(vis,     T_eff)
        hm_c       = _chunk_or_pad_tensor(hm,      T_eff)
        bboxs_c    = _chunk_or_pad_tensor(crop_xywh, T_eff)
        # delta_rt_c = _chunk_or_pad_tensor(delta_Rt, T_eff)
        meta_c     = _chunk_or_pad_tensor(meta,     T_eff)

        paths_c = _chunk_or_pad_paths(frame_ids, T_eff)

        n_chunks = frames_c.shape[0]
        for i in range(n_chunks):
            out_frames.append(frames_c[i])
            out_kps.append(kps_c[i])
            out_gt_q.append(q_c[i])
            out_gt_r.append(r_c[i])
            out_vis.append(vis_c[i])
            out_hm.append(hm_c[i])
            out_bboxs.append(bboxs_c[i])
            # out_delta_Rt.append(delta_rt_c[i])

            out_meta.append(meta_c[i])     
            out_seq_names.append(seq_name)
            out_frame_ids.append(paths_c[i])

    if len(out_frames) == 0:
        raise RuntimeError("collate_temporal_chunks produced 0 chunks (check T_chunk vs sequence length).")
    
    frames_b = torch.stack(out_frames, dim=0)
    kps_b = torch.stack(out_kps, dim=0)
    gt_q_b = torch.stack(out_gt_q, dim=0)
    gt_r_b = torch.stack(out_gt_r, dim=0)
    vis_b = torch.stack(out_vis, dim=0)
    hm_b = torch.stack(out_hm, dim=0)
    bbox_b = torch.stack(out_bboxs, dim=0)
    # delta_Rt_b = torch.stack(out_delta_Rt, dim=0)
    meta_b = torch.stack(out_meta, dim=0)

    return frames_b, kps_b, gt_q_b, gt_r_b, vis_b, hm_b, out_frame_ids, bbox_b, meta_b, out_seq_names

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

# --------------------------------------------------
# for calculating loss based on PCK improvement only 
# --------------------------------------------------

@torch.no_grad()
def pck_crop_from_normhm(
    kps_pred_normhm: torch.Tensor,   # (B,T,K,2) in [-1,1] w.r.t heatmap grid
    kps_gt_normhm: torch.Tensor,     # (B,T,K,2) in [-1,1] w.r.t heatmap grid
    vis: torch.Tensor,               # (B,T,K)
    input_size: int,                 # e.g., 512
    heatmap_hw,                      # (Hm,Wm) e.g., (256,256)
    thr_px: float,
) -> torch.Tensor:
    """
    PCK in CROP-INPUT pixel coordinates (0..input_size-1), consistent for crop-only training.
    """
    pred_in = _hm_norm_to_input_px(kps_pred_normhm, input_size=input_size, heatmap_hw=heatmap_hw)
    gt_in   = _hm_norm_to_input_px(kps_gt_normhm,   input_size=input_size, heatmap_hw=heatmap_hw)

    err = torch.linalg.norm(pred_in - gt_in, dim=-1)        # (B,T,K)
    ok  = (err <= float(thr_px)).float()

    vis_f = vis.float()
    num = (ok * vis_f).sum()
    den = vis_f.sum().clamp_min(1.0)
    return num / den

def _format_row_4dp(row: dict, float_keys=None):
    if float_keys is None:
        float_keys = [k for k, v in row.items() if isinstance(v, (float, np.floating))]
    out = dict(row)
    for k in float_keys:
        v = out.get(k, "")
        if v == "" or v is None:
            continue
        out[k] = f"{float(v):.4f}"
    return out