# utils_new.py (CLEAN)
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from typing import Dict, Any, Tuple, Optional
from losses import softargmax_2d_from_logits
# ---- ImageNet denorm for visualization ----
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1)

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
        if hasattr(bb, "ffn_adapters"):
            unfreeze_module(bb.ffn_adapters)

        # Safety: ensure HF DINOv3 stays frozen
        if hasattr(bb, "backbone"):
            for p in bb.backbone.parameters():
                p.requires_grad = False

    if hasattr(model, "neck"):
        unfreeze_module(model.neck)

    if hasattr(model, "sigma_head"):
        unfreeze_module(model.sigma_head)
    # Heatmap head
    if hasattr(model, "hm_head"):
        unfreeze_module(model.hm_head)

    # # Gates are a bare Parameter, not a module
    # if hasattr(model, "gates") and isinstance(model.gates, nn.Parameter):
    #     model.gates.requires_grad = True

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
            "mhsa_adapters": [n for n in trainable_names if "backbone.adapters" in n],
            "ffn_adapters": [n for n in trainable_names if "backbone.ffn_adapters" in n],
            "neck": [n for n in trainable_names if n.startswith("neck.")],
            "sigma_head": [n for n in trainable_names if "sigma_head" in n],
            "kp_token": [n for n in trainable_names if "kp_token" in n or "kp_tokens" in n],
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


import os, math
import numpy as np
import torch
import torch.nn.functional as F

@torch.no_grad()
def save_token_similarity_diagnostics_bt(
    out,
    frames,
    cfg,
    save_path,
    crop_xywh: torch.Tensor,         # (B,T,4) normalized wrt FULL
    cam_K: torch.Tensor,             # (B,3,3)
    full_hw: tuple[int, int],        # (H_full,W_full)
    b: int = 0,
    t: int = 0,                      # kept for compatibility; we visualize t0->t1 pair
    t_pair: str = "ends",            # "ends" | "random"
    ray_bin_deg: float = 6.0,
    temp: float = 0.07,
    n_anchors_show: int = 6,
):
    """
    Ray-consistent self-supervised token diagnostics (NO GT).

    Produces:
      1) Ray-bin map overlay (patches grouped by same ray neighborhood)
      2) Positives-per-anchor histogram (multi-positive counts)
      3) Collapse indicators:
         - mean(pos sim), mean(neg sim), gap
         - row-entropy of softmax(sim/temp)
         - sharpness proxy = mean(logsumexp(sim/temp))
      4) A few anchor panels:
         - similarity map anchor->all targets at t1
         - highlight positives (same ray-bin) overlay

    Assumes tokens are patch tokens (or slices them to P_expected).
    """

    # -------------------------
    # pick token tensor
    # -------------------------
    tok = None
    for k in ["tokens_geo", "adapter_tokens", "patch_tokens_prenorm", "patch_tokens"]:
        if k in out and out[k] is not None:
            tok = out[k]
            break
    if tok is None:
        return

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
        P = P_got
        g = int(round(math.sqrt(P)))
        if g * g != P:
            return
        Hp = Wp = g

    if T < 2:
        return

    device = tok.device

    # -------------------------
    # choose time pair
    # -------------------------
    if t_pair == "random":
        t0 = 0
        t1 = int(torch.randint(low=1, high=T, size=(1,), device=device).item())
    else:
        t0, t1 = 0, T - 1

    # -------------------------
    # image prep
    # -------------------------
    Himg = int(cfg["data"]["input_size"])
    Wimg = int(cfg["data"]["input_size"])

    frames_vis = denorm_imagenet_bt(frames)
    img = frames_vis[b, t0].detach().cpu().permute(1, 2, 0).numpy()
    img_u8 = (img * 255.0).clip(0, 255).astype(np.uint8)

    # -------------------------
    # tokens: normalize
    # -------------------------
    z0 = F.normalize(tok[b, t0].float(), dim=-1)  # (P,D)
    z1 = F.normalize(tok[b, t1].float(), dim=-1)  # (P,D)

    # -------------------------
    # compute per-patch ray bins (for THIS sample b)
    # -------------------------
    # patch centers in crop px
    ys = (torch.arange(Hp, device=device, dtype=torch.float32) + 0.5) * patch_size
    xs = (torch.arange(Wp, device=device, dtype=torch.float32) + 0.5) * patch_size
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    uv_crop = torch.stack([xx, yy], dim=-1).view(P, 2)  # (P,2)

    H_full, W_full = int(full_hw[0]), int(full_hw[1])
    # crop box at (b,t0) in full px
    x0 = crop_xywh[b, t0, 0] * W_full
    y0 = crop_xywh[b, t0, 1] * H_full
    w  = crop_xywh[b, t0, 2] * W_full
    h  = crop_xywh[b, t0, 3] * H_full

    u = uv_crop[:, 0] / float(input_size)
    v = uv_crop[:, 1] / float(input_size)

    u_full = x0 + u * w  # (P,)
    v_full = y0 + v * h  # (P,)

    ones = torch.ones((P,), device=device, dtype=torch.float32)
    pix = torch.stack([u_full, v_full, ones], dim=-1)  # (P,3)

    Kinv = torch.inverse(cam_K[b].float())  # (3,3)
    dirs = (pix @ Kinv.T)                   # (P,3)
    dirs = F.normalize(dirs, dim=-1)

    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2].clamp_min(1e-8)
    az = torch.atan2(x, z)
    el = torch.atan2(y, z)

    bin_rad = math.radians(max(ray_bin_deg, 1e-3))
    az_bin = torch.floor((az + math.pi) / bin_rad).long()
    el_bin = torch.floor((el + (math.pi * 0.5)) / bin_rad).long()
    n_el = int(math.ceil(math.pi / bin_rad)) + 1
    bins = az_bin * n_el + el_bin  # (P,)

    # compact bins for colormap stability in visualization
    uniq_bins, inv_bins = torch.unique(bins, return_inverse=True)
    bins_compact = inv_bins  # (P,) values in [0..nbins-1]
    nbins = int(uniq_bins.numel())

    # -------------------------
    # build similarity matrix for this sample: (P,P) anchor->targets
    # -------------------------
    sim = (z0 @ z1.T)  # (P,P)
    sim_scaled = sim / max(float(temp), 1e-6)
    prob = F.softmax(sim_scaled, dim=-1)  # (P,P)
    logprob = torch.log(prob.clamp_min(1e-12))

    # positives mask: same ray bin
    pos = bins.unsqueeze(1).eq(bins.unsqueeze(0))  # (P,P)

    # per-anchor counts of positives (including self); report also excluding self
    pos_count = pos.sum(dim=1).float()  # (P,)
    pos_count_ex = (pos_count - 1.0).clamp_min(0.0)

    # collapse indicators
    # mean pos/neg sims
    pos_sim_mean = (sim[pos]).mean().item() if pos.any() else float("nan")
    neg_sim_mean = (sim[~pos]).mean().item() if (~pos).any() else float("nan")
    gap = pos_sim_mean - neg_sim_mean if np.isfinite(pos_sim_mean) and np.isfinite(neg_sim_mean) else float("nan")

    # row entropy of prob (normalized to [0,1])
    ent = (-(prob * logprob).sum(dim=1) / math.log(P)).detach().cpu().numpy()
    ent_mean = float(np.mean(ent))
    ent_p10 = float(np.percentile(ent, 10))
    ent_p90 = float(np.percentile(ent, 90))

    # sharpness proxy: logsumexp(sim/temp) per row (higher = peaky / large logits)
    lse = torch.logsumexp(sim_scaled, dim=1).detach().cpu().numpy()
    lse_mean = float(np.mean(lse))

    # expected positives mass under prob
    pos_mass = (prob * pos.float()).sum(dim=1).detach().cpu().numpy()
    pos_mass_mean = float(np.mean(pos_mass))

    # -------------------------
    # choose anchors to visualize (spread across bins)
    # -------------------------
    # take anchors from most frequent bins first, but diversify
    bin_ids = bins_compact.detach().cpu().numpy()
    anchors = []
    # rank bins by frequency
    freqs = np.bincount(bin_ids, minlength=max(1, nbins))
    bin_order = np.argsort(-freqs)
    for bi in bin_order:
        idxs = np.where(bin_ids == bi)[0]
        if idxs.size == 0:
            continue
        anchors.append(int(idxs[idxs.size // 2]))  # median index for stability
        if len(anchors) >= n_anchors_show:
            break
    if len(anchors) == 0:
        anchors = [P // 2]

    # -------------------------
    # helpers: upsample patch maps to image and overlay
    # -------------------------
    def upsample_map(patch_vals_1d: torch.Tensor):
        m = patch_vals_1d.view(1, 1, Hp, Wp)
        m_up = F.interpolate(m, size=(Himg, Wimg), mode="bilinear", align_corners=False)[0, 0]
        return m_up.detach().cpu().numpy()

    # ray-bin visualization: normalize bins to [0,1] for display
    ray_map = bins_compact.float()
    ray_map = (ray_map - ray_map.min()) / (ray_map.max() - ray_map.min() + 1e-8)
    ray_up = upsample_map(ray_map)

    # positives-per-anchor map: show log(count) per patch bin
    # map each patch to its bin frequency (in this crop)
    freq_per_patch = torch.tensor(freqs, device=device, dtype=torch.float32)[bins_compact]
    freq_map = torch.log1p(freq_per_patch)
    freq_map = (freq_map - freq_map.min()) / (freq_map.max() - freq_map.min() + 1e-8)
    freq_up = upsample_map(freq_map)

    # -------------------------
    # plotting
    # -------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    import matplotlib.pyplot as plt

    cols = 4
    # panels:
    # 1: ray bins, 2: bin freq map, 3: histogram, 4: entropy map
    # then anchors: 2 panels each (sim map, pos mask)
    n_anchor_panels = 2 * len(anchors)
    n_panels = 4 + n_anchor_panels
    rows = int(math.ceil(n_panels / cols))

    fig = plt.figure(figsize=(4.2 * cols, 4.2 * rows))

    def overlay(ax, heat, title, alpha=0.38, vmin=0.0, vmax=1.0):
        ax.imshow(img)
        ax.imshow(heat, cmap="jet", alpha=alpha, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # (1) ray bin map
    ax = fig.add_subplot(rows, cols, 1)
    overlay(ax, ray_up, f"ray-bin map (deg={ray_bin_deg:.1f}) | nbins={nbins}")

    # (2) bin frequency map
    ax = fig.add_subplot(rows, cols, 2)
    overlay(ax, freq_up, "log1p(bin freq) per patch")

    # (3) positives count histogram
    ax = fig.add_subplot(rows, cols, 3)
    pc = pos_count_ex.detach().cpu().numpy()
    ax.hist(pc, bins=30)
    ax.set_title(f"positives per anchor (excl self)\nmean={pc.mean():.2f}  p50={np.median(pc):.1f}  p90={np.percentile(pc,90):.1f}", fontsize=10)

    # (4) entropy map proxy: per-anchor entropy laid onto its patch location
    # just display entropy at each anchor index as a patch map (same entropy per anchor)
    ent_t = torch.tensor(ent, device=device, dtype=torch.float32)
    ent_map = (ent_t - ent_t.min()) / (ent_t.max() - ent_t.min() + 1e-8)
    ent_up_map = upsample_map(ent_map)
    ax = fig.add_subplot(rows, cols, 4)
    overlay(ax, ent_up_map, "row entropy of softmax(sim/temp) (normalized)")

    panel_i = 5

    # anchor panels
    for ai, aidx in enumerate(anchors):
        # similarity map anchor->all targets
        sim_row = sim[aidx]  # (P,)
        # normalize for display (robust percentiles)
        s = sim_row.detach().cpu().numpy()
        lo, hi = np.percentile(s, 5), np.percentile(s, 95)
        sim_disp = np.clip((s - lo) / (hi - lo + 1e-8), 0.0, 1.0)
        sim_up = upsample_map(torch.tensor(sim_disp, device=device))

        ax = fig.add_subplot(rows, cols, panel_i); panel_i += 1
        overlay(ax, sim_up, f"anchor {aidx} sim(t{t0}->t{t1})\nbin={int(bins_compact[aidx])}  pos#={int(pos_count_ex[aidx].item())}")

        # positive mask map for this anchor (same ray bin)
        pos_row = pos[aidx].float()  # (P,)
        pos_up = upsample_map(pos_row)

        ax = fig.add_subplot(rows, cols, panel_i); panel_i += 1
        overlay(ax, pos_up, f"positives mask (same ray bin)\nanchor {aidx}", alpha=0.32)

    # global summary
    fig.suptitle(
        f"Ray-consistent token diagnostics | b={b} t0={t0} t1={t1} | temp={temp:.3f} deg={ray_bin_deg:.1f}\n"
        f"mean(pos sim)={pos_sim_mean:.3f}  mean(neg sim)={neg_sim_mean:.3f}  gap={gap:.3f} | "
        f"ent(mean)={ent_mean:.3f} [p10={ent_p10:.3f}, p90={ent_p90:.3f}] | "
        f"lse(mean)={lse_mean:.3f} | pos_mass(mean)={pos_mass_mean:.3f}",
        fontsize=12
    )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    return {
        "tok_pos_sim_mean": float(pos_sim_mean),
        "tok_neg_sim_mean": float(neg_sim_mean),
        "tok_posneg_gap": float(gap),
        "tok_entropy_mean": float(ent_mean),
        "tok_entropy_p10": float(ent_p10),
        "tok_entropy_p90": float(ent_p90),
        "tok_lse_mean": float(lse_mean),
        "tok_pos_mass_mean": float(pos_mass_mean),
        "tok_pos_count_mean": float(np.mean(pc)),
        "tok_pos_count_p90": float(np.percentile(pc, 90)),
        "tok_nbins": int(nbins),
        "tok_P": int(P),
    }



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
    out = {
        "frames": [], "kps_norm": [], "kps_full_px": [],
        "gt_q": [], "gt_r": [], "vis": [], "hm": [],
        "frame_ids": [], "crop_xywh": [], "meta": [], "seq_name": []
    }

    def _chunk(x: torch.Tensor, T_eff: int):
        T = x.shape[0]
        if drop_last:
            x = x[:T_eff]
            n = T_eff // T_chunk
            return x.view(n, T_chunk, *x.shape[1:])
        n = (T_eff + T_chunk - 1) // T_chunk
        T_pad = n * T_chunk
        if T_pad > T_eff:
            pad = T_pad - T_eff
            x = torch.cat([x, x[-1:].repeat(pad, *([1] * (x.ndim - 1)))], dim=0)
        return x.view(n, T_chunk, *x.shape[1:])

    def _chunk_paths(paths: list, T_eff: int):
        if drop_last:
            paths = paths[:T_eff]
            n = T_eff // T_chunk
            return [paths[i*T_chunk:(i+1)*T_chunk] for i in range(n)]
        n = (T_eff + T_chunk - 1) // T_chunk
        T_pad = n * T_chunk
        if T_pad > T_eff:
            paths = paths + [paths[-1]] * (T_pad - T_eff)
        return [paths[i*T_chunk:(i+1)*T_chunk] for i in range(n)]

    for sample in batch:
        (
            frames, kps_norm, kps_full_px,
            gt_q, gt_r, vis, hm, frame_ids,
            crop_xywh, meta, seq_name
        ) = sample

        if isinstance(frame_ids, (list, tuple)) and len(frame_ids) == 1 and isinstance(frame_ids[0], (list, tuple)):
            frame_ids = frame_ids[0]

        T = frames.shape[0]
        T_eff = (T // T_chunk) * T_chunk if drop_last else T
        if T_eff == 0:
            continue

        frames_c = _chunk(frames, T_eff)
        kpsn_c   = _chunk(kps_norm, T_eff)
        kpsf_c   = _chunk(kps_full_px, T_eff)
        q_c      = _chunk(gt_q, T_eff)
        r_c      = _chunk(gt_r, T_eff)
        vis_c    = _chunk(vis, T_eff)
        hm_c     = _chunk(hm, T_eff)
        box_c    = _chunk(crop_xywh, T_eff)
        meta_c   = _chunk(meta, T_eff)

        paths_c = _chunk_paths(frame_ids, T_eff)

        n_chunks = frames_c.shape[0]
        for i in range(n_chunks):
            out["frames"].append(frames_c[i])
            out["kps_norm"].append(kpsn_c[i])
            out["kps_full_px"].append(kpsf_c[i])
            out["gt_q"].append(q_c[i])
            out["gt_r"].append(r_c[i])
            out["vis"].append(vis_c[i])
            out["hm"].append(hm_c[i])
            out["crop_xywh"].append(box_c[i])
            out["meta"].append(meta_c[i])
            out["frame_ids"].append(paths_c[i])
            out["seq_name"].append(seq_name)

    if len(out["frames"]) == 0:
        raise RuntimeError("collate_temporal_chunks produced 0 chunks.")

    # stack
    frames_b = torch.stack(out["frames"], dim=0)
    kpsn_b   = torch.stack(out["kps_norm"], dim=0)
    kpsf_b   = torch.stack(out["kps_full_px"], dim=0)
    gt_q_b   = torch.stack(out["gt_q"], dim=0)
    gt_r_b   = torch.stack(out["gt_r"], dim=0)
    vis_b    = torch.stack(out["vis"], dim=0)
    hm_b     = torch.stack(out["hm"], dim=0)
    box_b    = torch.stack(out["crop_xywh"], dim=0)
    meta_b   = torch.stack(out["meta"], dim=0)

    return (
        frames_b, kpsn_b, kpsf_b,
        gt_q_b, gt_r_b, vis_b, hm_b,
        out["frame_ids"], box_b, meta_b, out["seq_name"]
    )

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

# =============================
# Delta-pose helpers
# =============================

def quat_to_rotmat(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Vectorized quaternion (x,y,z,w) -> rotation matrix.
    q: (..., 4)
    returns: (..., 3, 3)
    """
    assert q.shape[-1] == 4, f"Expected (...,4), got {q.shape}"

    q = F.normalize(q, dim=-1, eps=eps)
    x, y, z, w = q.unbind(dim=-1)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    # Build matrix entries
    m00 = 1.0 - 2.0 * (yy + zz)
    m01 = 2.0 * (xy - wz)
    m02 = 2.0 * (xz + wy)

    m10 = 2.0 * (xy + wz)
    m11 = 1.0 - 2.0 * (xx + zz)
    m12 = 2.0 * (yz - wx)

    m20 = 2.0 * (xz - wy)
    m21 = 2.0 * (yz + wx)
    m22 = 1.0 - 2.0 * (xx + yy)

    R = torch.stack([
        torch.stack([m00, m01, m02], dim=-1),
        torch.stack([m10, m11, m12], dim=-1),
        torch.stack([m20, m21, m22], dim=-1),
    ], dim=-2)

    return R


# utils_geospace.py
def get_w_scheduled(epoch: int, base_w: float, sched_cfg: dict):
    if sched_cfg is None:
        return float(base_w)
    typ = str(sched_cfg.get("type", "none")).lower()
    if typ == "none":
        return float(base_w)

    if typ == "piecewise":
        for (a, b, v) in sched_cfg.get("segments", []):
            if int(a) <= int(epoch) <= int(b):
                return float(v)
        return float(base_w)

    raise ValueError(f"Unknown schedule type: {typ}")