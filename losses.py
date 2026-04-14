# losses.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# special token loss 
def token_spatial_alignment_loss_loo(
    patch_tokens: torch.Tensor,   # (B,T,P,D)
    kps_gt_norm: torch.Tensor,    # (B,T,K,2) in [-1,1]
    vis: torch.Tensor,            # (B,T,K) in {0,1}
    sigma_patch: float = 0.35,
    temp_pred: float = 0.07,
    eps: float = 1e-8,
    w_div: float = 0.05,
    detach_proto: bool = True,    # <--- ADD THIS
):
    B, T, P, D = patch_tokens.shape
    g = int(round(math.sqrt(P)))
    assert g * g == P, f"P={P} not square"

    device = patch_tokens.device
    dtype  = patch_tokens.dtype

    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, g, device=device, dtype=dtype),
        torch.linspace(-1, 1, g, device=device, dtype=dtype),
        indexing="ij",
    )
    patch_xy = torch.stack([xs, ys], dim=-1).view(1, 1, 1, P, 2)

    kp_xy = kps_gt_norm.unsqueeze(3)  # (B,T,K,1,2)
    dist2 = ((kp_xy - patch_xy) ** 2).sum(dim=-1)  # (B,T,K,P)

    sig2 = float(max(sigma_patch, 1e-6)) ** 2
    tgt_logits = -0.5 * dist2 / sig2
    tgt = F.softmax(tgt_logits, dim=-1)  # (B,T,K,P)

    tok = F.normalize(patch_tokens.float(), dim=-1)  # (B,T,P,D)

    # teacher tokens used to build proto
    tok_t = tok.detach() if detach_proto else tok  # <--- CONTROLLED HERE

    w = tgt  # (B,T,K,P)

    sum_w = w.sum(dim=-1, keepdim=True).clamp_min(eps)                         # (B,T,K,1)
    sum_wtok = (w.unsqueeze(-1) * tok_t.unsqueeze(2)).sum(dim=3)               # (B,T,K,D)

    w_p = w.unsqueeze(-1)                                                      # (B,T,K,P,1)
    tok_p = tok_t.unsqueeze(2).expand(B, T, w.shape[2], P, D)                   # (B,T,K,P,D)

    denom = (sum_w.unsqueeze(3) - w_p).clamp_min(eps)                           # (B,T,K,P,1)
    num   = (sum_wtok.unsqueeze(3) - w_p * tok_p)                               # (B,T,K,P,D)

    proto_loo = F.normalize(num / denom, dim=-1)                                # (B,T,K,P,D)

    tok_s = tok.unsqueeze(2).expand(B, T, w.shape[2], P, D)                     # (B,T,K,P,D)
    sim = (tok_s * proto_loo).sum(dim=-1)                                       # (B,T,K,P)

    temp = float(max(temp_pred, 1e-6))
    logP = F.log_softmax(sim / temp, dim=-1)

    kl = (tgt * (torch.log(tgt.clamp_min(eps)) - logP)).sum(dim=-1)             # (B,T,K)

    m = vis.float()
    loss_align = (kl * m).sum() / (m.sum().clamp_min(1.0))

    loss_div = tok.new_zeros(())
    if w_div > 0:
        P_pred = logP.exp()
        overlap = torch.einsum("btkp,btjp->btkj", P_pred, P_pred)
        eye = torch.eye(overlap.shape[-1], device=device, dtype=overlap.dtype).view(1,1,overlap.shape[-1],overlap.shape[-1])
        off = overlap * (1.0 - eye)
        vm = m.unsqueeze(-1) * m.unsqueeze(-2)
        loss_div = (off * vm).sum() / (vm.sum().clamp_min(1.0))

    return loss_align + w_div * loss_div


# new testing for token loss 
def token_spatial_alignment_loss(
    patch_tokens: torch.Tensor,   # (B,T,P,D)
    kps_gt_norm: torch.Tensor,    # (B,T,K,2) in [-1,1] (heatmap-norm coords)
    vis: torch.Tensor,            # (B,T,K) in {0,1}
    sigma_patch: float = 0.35,    # controls target/prototype locality in [-1,1] grid
    temp_pred: float = 0.07,      # softmax temperature for predicted distribution
    eps: float = 1e-8,
    detach_proto: bool = False,   # True = more stable early, False = stronger learning
    w_div: float = 0.05,          # overlap penalty weight
):
    """
    Keypoint-conditional token alignment.

    Target: tgt_k(p) ∝ exp(-||p - kp||^2 / (2*sigma^2))
    Pred:   P_k(p) ∝ exp(sim(token_p, proto_k) / temp_pred)

    proto_k is a Gaussian-weighted avg of tokens around the GT kp location.
    """
    assert patch_tokens.ndim == 4
    assert kps_gt_norm.ndim == 4
    assert vis.ndim == 3

    B, T, P, D = patch_tokens.shape
    g = int(round(math.sqrt(P)))
    assert g * g == P, f"P={P} not square (make sure you sliced to patches-only)"
    Hp = Wp = g

    device = patch_tokens.device
    dtype = patch_tokens.dtype

    # --- patch grid in [-1,1] ---
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, Hp, device=device, dtype=dtype),
        torch.linspace(-1, 1, Wp, device=device, dtype=dtype),
        indexing="ij",
    )
    patch_xy = torch.stack([xs, ys], dim=-1).view(1, 1, 1, P, 2)  # (1,1,1,P,2)

    # --- distances to keypoints ---
    kp_xy = kps_gt_norm.unsqueeze(3)  # (B,T,K,1,2)
    dist2 = ((kp_xy - patch_xy) ** 2).sum(dim=-1)  # (B,T,K,P)

    # --- target distribution (Gaussian) ---
    sig2 = float(max(sigma_patch, 1e-6)) ** 2
    tgt_logits = -0.5 * dist2 / sig2
    tgt = F.softmax(tgt_logits, dim=-1)  # (B,T,K,P)

    # --- normalize tokens ---
    tok = patch_tokens
    tok_n = F.normalize(tok.float(), dim=-1)  # (B,T,P,D)

    # --- build proto_k by Gaussian-weighted avg of tokens around kp ---
    w = tgt  # reuse tgt as weights (already local around kp)
    proto = (w.unsqueeze(-1) * tok_n.unsqueeze(2)).sum(dim=3)  # (B,T,K,D)
    proto = F.normalize(proto, dim=-1)

    if detach_proto:
        proto = proto.detach()

    # --- keypoint-conditional similarity map ---
    # sim = (tok_n · proto_k) over D -> (B,T,K,P)
    sim = torch.einsum("btpd,btkd->btkp", tok_n, proto)

    # predicted distribution
    temp = float(max(temp_pred, 1e-6))
    logP = F.log_softmax(sim / temp, dim=-1)  # (B,T,K,P)

    # --- KL(tgt || P) ---
    kl = (tgt * (torch.log(tgt.clamp_min(eps)) - logP)).sum(dim=-1)  # (B,T,K)

    m = vis.float()
    loss_align = (kl * m).sum() / (m.sum().clamp_min(1.0))

    # --- optional: diversity / low-overlap penalty between keypoints ---
    # Encourage different keypoints to attend different patches.
    # Compute overlap matrix O_ij = sum_p tgt_i(p)*tgt_j(p) (or use P instead of tgt).
    loss_div = tok.new_zeros(())
    if w_div > 0:
        # use predicted probs for diversity (more direct)
        P_pred = logP.exp()  # (B,T,K,P)
        # (B,T,K,K) overlap
        overlap = torch.einsum("btkp,btjp->btkj", P_pred, P_pred)
        # penalize off-diagonal overlaps
        eye = torch.eye(overlap.shape[-1], device=device, dtype=overlap.dtype).view(1,1,overlap.shape[-1],overlap.shape[-1])
        off = overlap * (1.0 - eye)
        # visibility mask for pairs
        vm = m.unsqueeze(-1) * m.unsqueeze(-2)  # (B,T,K,K)
        loss_div = (off * vm).sum() / (vm.sum().clamp_min(1.0))

    return loss_align + w_div * loss_div



def add_l1_pose_loss(
    R_pred: torch.Tensor,  # (B,T,3,3)
    t_pred: torch.Tensor,  # (B,T,3)
    R_gt: torch.Tensor,    # (B,T,3,3)
    t_gt: torch.Tensor,    # (B,T,3)
    pts3d_model: torch.Tensor,  # (K,3) or (1,K,3)
    frame_mask: torch.Tensor | None = None,  # (B,T) or None
    reduction: str = "mean",
):
    """
    ADD with L1 (robust):
      mean_{k} || (R_pred X + t_pred) - (R_gt X + t_gt) ||_1
    """
    if pts3d_model.dim() == 2:
        pts = pts3d_model.unsqueeze(0)  # (1,K,3)
    else:
        pts = pts3d_model
    _, K, _ = pts.shape

    B, T = t_pred.shape[:2]
    pts = pts.to(R_pred.device, R_pred.dtype)

    Rp = R_pred.reshape(B*T, 3, 3)
    tp = t_pred.reshape(B*T, 3).unsqueeze(1)            # (BT,1,3)
    Rg = R_gt.reshape(B*T, 3, 3)
    tg = t_gt.reshape(B*T, 3).unsqueeze(1)

    X = pts.expand(B*T, -1, -1)                          # (BT,K,3)

    pred = (Rp @ X.transpose(1,2)).transpose(1,2) + tp   # (BT,K,3)
    gt   = (Rg @ X.transpose(1,2)).transpose(1,2) + tg

    l1 = (pred - gt).abs().mean(dim=(1,2))               # mean over K and xyz -> (BT,)
    l1 = l1.view(B, T)

    if frame_mask is not None:
        m = frame_mask.float()
        out = (l1 * m).sum() / (m.sum().clamp_min(1.0))
    else:
        out = l1.mean()

    return out
# working toekn spatial alignment loss
# def token_spatial_alignment_loss(
#     patch_tokens: torch.Tensor,   # (B,T,P,D)
#     kps_gt_norm: torch.Tensor,    # (B,T,K,2) in [-1,1]
#     vis: torch.Tensor,            # (B,T,K) in {0,1}
#     temperature: float = 0.07,
#     eps: float = 1e-8,
# ):
#     """
#     Token ↔ keypoint spatial alignment (KL).

#     Build a target distribution over patches for each GT keypoint using a
#     distance-based softmax:  tgt(p) ∝ exp(-||p - kp||^2 / temperature)

#     Compare it to a predicted patch distribution derived from token "energy"
#     (L2 norm), but made scale-robust by centering energy per-frame:
#         energy := ||token|| ; energy_centered := energy - mean_p(energy)
#         logP := log_softmax(energy_centered)

#     Returns:
#         scalar loss (visibility-masked mean over (B,T,K))
#     """
#     assert patch_tokens.ndim == 4, f"patch_tokens must be (B,T,P,D), got {patch_tokens.shape}"
#     assert kps_gt_norm.ndim == 4, f"kps_gt_norm must be (B,T,K,2), got {kps_gt_norm.shape}"
#     assert vis.ndim == 3, f"vis must be (B,T,K), got {vis.shape}"

#     B, T, P, D = patch_tokens.shape
#     Hp = int(round(math.sqrt(P)))
#     assert Hp * Hp == P, f"P={P} is not a square — invalid ViT patch grid"
#     Wp = Hp

#     device = patch_tokens.device
#     dtype = patch_tokens.dtype

#     # ------------------------------------------------
#     # 1) Patch grid coordinates in [-1,1]
#     # ------------------------------------------------
#     ys, xs = torch.meshgrid(
#         torch.linspace(-1, 1, Hp, device=device, dtype=dtype),
#         torch.linspace(-1, 1, Wp, device=device, dtype=dtype),
#         indexing="ij",
#     )
#     patch_xy = torch.stack([xs, ys], dim=-1).view(1, 1, 1, P, 2)  # (1,1,1,P,2)

#     # ------------------------------------------------
#     # 2) Target distribution over patches for each keypoint (distance-based)
#     # ------------------------------------------------
#     kp_xy = kps_gt_norm.unsqueeze(3)  # (B,T,K,1,2)
#     dist2 = ((kp_xy - patch_xy) ** 2).sum(dim=-1)  # (B,T,K,P)

#     temp = float(max(temperature, 1e-6))
#     spatial_logits = -dist2 / temp
#     tgt = F.softmax(spatial_logits, dim=-1)  # (B,T,K,P)

#     # ------------------------------------------------
#     # 3) Predicted distribution from token energy (scale-robust)
#     # ------------------------------------------------
#     # Energy per patch
#     energy = patch_tokens.norm(dim=-1)  # (B,T,P)

#     # Center per (B,T) to reduce "just scale everything up" solutions
#     energy = energy - energy.mean(dim=-1, keepdim=True)  # (B,T,P)

#     # Broadcast to match (B,T,K,P)
#     logP = F.log_softmax(energy.unsqueeze(2), dim=-1)  # (B,T,1,P) -> broadcast

#     # ------------------------------------------------
#     # 4) KL(tgt || P) per keypoint
#     # ------------------------------------------------
#     kl = (tgt * (torch.log(tgt.clamp_min(eps)) - logP)).sum(dim=-1)  # (B,T,K)

#     # ------------------------------------------------
#     # 5) Visibility mask reduction
#     # ------------------------------------------------
#     m = vis.float()
#     loss = (kl * m).sum() / (m.sum().clamp_min(1.0))
#     return loss
# losses.py

def temporal_smoothness_l2(x: torch.Tensor, vis: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x:   (B,T,K,2)  (e.g., COM in pixel coords or normalized coords)
    vis: (B,T,K)    0/1 visibility
    Returns scalar.
    Penalizes second difference: x[t+2] - 2x[t+1] + x[t]
    """
    if x.shape[1] < 3:
        return x.new_zeros(())

    dd = x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]          # (B,T-2,K,2)
    v  = vis[:, 2:] * vis[:, 1:-1] * vis[:, :-2]          # (B,T-2,K)
    w  = v.unsqueeze(-1)                                   # (B,T-2,K,1)

    num = (dd.pow(2) * w).sum()
    den = w.sum() * dd.shape[-1] + eps
    return num / den

def temporal_contrastive_loss(z, temperature=0.1):
    """
    z: (B,T,D) normalized embeddings
    Positive: same sample, adjacent frames
    Negatives: other samples in batch
    """
    B, T, D = z.shape
    if T < 2:
        return torch.tensor(0.0, device=z.device)

    z1 = z[:, :-1]  # (B,T-1,D)
    z2 = z[:, 1:]   # (B,T-1,D)

    z1 = z1.reshape(-1, D)
    z2 = z2.reshape(-1, D)

    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0), device=z.device)

    return F.cross_entropy(logits, labels)

class TemporalProjHead(torch.nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, out_dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)
    
def frequency_jitter_loss(
    coords: torch.Tensor,     # (B, T, K, 2)
    vis: torch.Tensor,         # (B, T, K)
    cutoff_ratio: float = 0.4,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Penalize high-frequency components in keypoint trajectories along the
    temporal dimension.  Low-frequency = legitimate smooth motion;
    high-frequency = jitter/oscillation.

    Uses real FFT on the time axis, then penalizes energy above cutoff_ratio
    of the Nyquist frequency.
    """
    B, T, K, _ = coords.shape
    if T < 4:
        return coords.new_zeros(())

    # Build a per-keypoint visibility mask that requires ALL frames visible
    # (FFT needs contiguous signal; partial visibility would corrupt spectrum)
    all_vis = vis.min(dim=1).values  # (B, K) — 1 only if visible in every frame
    if all_vis.sum() < 1:
        return coords.new_zeros(())

    # FFT along time dimension: (B, T, K, 2) -> (B, F, K, 2) complex
    X = torch.fft.rfft(coords.float(), dim=1)  # (B, F, K, 2) where F = T//2 + 1
    F = X.shape[1]

    # Power spectrum
    power = X.abs().pow(2)  # (B, F, K, 2)

    # Zero out low-frequency bins (keep them), penalize high-frequency
    cutoff = max(1, int(F * cutoff_ratio))
    high_freq_power = power[:, cutoff:]  # (B, F-cutoff, K, 2)

    # Mask by all_vis: (B, K) -> (B, 1, K, 1)
    mask = all_vis.unsqueeze(1).unsqueeze(-1).float()

    # Mean high-frequency energy over visible keypoints
    loss = (high_freq_power * mask).sum() / (mask.sum() * max(1, high_freq_power.shape[1]) * 2 + eps)

    return loss


def heatmap_to_com(hm_logits: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """
    hm_logits: (B,T,K,H,W)
    returns:   (B,T,K,2) in pixel coords of heatmap
    """
    B, T, K, H, W = hm_logits.shape
    hm = torch.softmax((hm_logits / max(tau, 1e-6)).reshape(B*T*K, -1), dim=-1)
    hm = hm.view(B*T*K, H, W)

    ys = torch.linspace(0, H-1, H, device=hm.device, dtype=hm.dtype)
    xs = torch.linspace(0, W-1, W, device=hm.device, dtype=hm.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    xx = xx.reshape(1, H, W)
    yy = yy.reshape(1, H, W)

    x = (hm * xx).sum(dim=(1, 2))
    y = (hm * yy).sum(dim=(1, 2))

    coords = torch.stack([x, y], dim=-1)
    return coords.view(B, T, K, 2)

# ----------------------------
# Quaternion & translation
# ----------------------------
def geodesic_quat_loss(pred_q: torch.Tensor, gt_q: torch.Tensor, reduction: str | None = "mean"):
    """ pred_q, gt_q: (B,T,4), returns 2*acos(|<q, q̂>|) """
    eps = 1e-8
    p = pred_q / (pred_q.norm(dim=-1, keepdim=True) + eps)
    g = gt_q   / (gt_q.norm(dim=-1, keepdim=True) + eps)
    dot = (p * g).sum(dim=-1).clamp(-1.0, 1.0).abs()
    ang = 2.0 * torch.arccos(dot)  # radians, (B,T)
    if reduction == "mean": return ang.mean()
    if reduction == "sum":  return ang.sum()
    return ang

def translation_l1(pred_t, gt_t, reduction="mean"):
    if reduction in ("mean", "sum", "none"):
        return F.l1_loss(pred_t, gt_t, reduction=reduction)
    elif reduction is None:
        return (pred_t - gt_t).abs()   # raw elementwise tensor
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

# ----------------------------
# Adding Temporal Coherence to Proto Loss
# ----------------------------
def temporal_proto_consistency_loss(
    patch_tokens: torch.Tensor,   # (B, T, P, D)
    kps_gt_norm: torch.Tensor,    # (B, T, K, 2)
    vis: torch.Tensor,            # (B, T, K)
    sigma_patch: float = 0.35,
) -> torch.Tensor:
    """
    Encourage keypoint prototypes to be consistent across adjacent frames.
    If keypoint k is visible in frames t and t+1, their prototypes should be similar.
    """
    B, T, P, D = patch_tokens.shape
    if T < 2:
        return patch_tokens.new_zeros(())
    
    g = int(round(math.sqrt(P)))
    device = patch_tokens.device
    dtype = patch_tokens.dtype
    
    # Build patch grid
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, g, device=device, dtype=dtype),
        torch.linspace(-1, 1, g, device=device, dtype=dtype),
        indexing="ij",
    )
    patch_xy = torch.stack([xs, ys], dim=-1).view(1, 1, 1, P, 2)
    
    # Compute prototypes per (frame, keypoint)
    kp_xy = kps_gt_norm.unsqueeze(3)  # (B,T,K,1,2)
    dist2 = ((kp_xy - patch_xy) ** 2).sum(dim=-1)  # (B,T,K,P)
    weights = F.softmax(-0.5 * dist2 / (sigma_patch ** 2), dim=-1)  # (B,T,K,P)
    
    tok_n = F.normalize(patch_tokens.float(), dim=-1)  # (B,T,P,D)
    protos = torch.einsum("btkp,btpd->btkd", weights, tok_n)  # (B,T,K,D)
    protos = F.normalize(protos, dim=-1)
    
    # Cosine similarity between consecutive-frame prototypes
    sim = (protos[:, 1:] * protos[:, :-1]).sum(dim=-1)  # (B,T-1,K)
    
    # Mask: both frames visible
    vis_pair = vis[:, 1:] * vis[:, :-1]  # (B,T-1,K)
    
    # Loss: maximize similarity (minimize 1 - sim)
    loss = (1.0 - sim) * vis_pair
    return loss.sum() / (vis_pair.sum().clamp_min(1.0))

# ----------------------------
# Heatmap losses
# ----------------------------
# def heatmap_kl_div_from_logits_masked(
#     hm_logits: torch.Tensor,     # (B,T,K,H,W), raw logits
#     hm_target: torch.Tensor,     # (B,T,K,H,W), un/normalized
#     vis: torch.Tensor,           # (B,T,K) in {0,1}
#     tau: float = 1.0,
#     target_power: float = 1.0,
# ) -> torch.Tensor:
#     """
#     KL(target || softmax(logits/tau)), averaged **only over visible kps**.
#     Dynamic sharpening is handled by target_power.
#     """
#     assert hm_logits.shape == hm_target.shape, f"Shape mismatch: logits={hm_logits.shape}, target={hm_target.shape}"
#     B, T, K, H, W = hm_logits.shape
#     logP = torch.nn.functional.log_softmax(
#         hm_logits.reshape(B*T*K, H*W) / tau, dim=-1
#     )
#     tgt = hm_target.reshape(B*T*K, H*W)
#     tgt = tgt / (tgt.sum(dim=-1, keepdim=True) + 1e-8)

#     if target_power != 1.0:
#         tgt = tgt.clamp_min(1e-12).pow(target_power)
#         tgt = tgt / (tgt.sum(dim=-1, keepdim=True) + 1e-8)

#     kl_vec = torch.sum(tgt * (torch.log(tgt + 1e-8) - logP), dim=-1)  # (BTK,)
#     m = vis.reshape(B*T*K).float()                                     # (BTK,)
#     return (kl_vec * m).sum() / (m.sum().clamp_min(1.0))

def heatmap_temporal_smoothness(
    hm_logits: torch.Tensor,   # (B, T, K, H, W)
    vis: torch.Tensor,         # (B, T, K)
    tau: float = 1.0,
) -> torch.Tensor:
    """
    Penalize heatmap distribution changes between adjacent frames via
    Jensen-Shannon divergence. This directly regularizes what DSNT reads from,
    attacking the root cause of coordinate jitter.
    """
    B, T, K, H, W = hm_logits.shape
    if T < 2:
        return hm_logits.new_zeros(())

    # Convert to probability distributions
    hm = F.softmax(hm_logits.reshape(B, T, K, -1) / max(tau, 1e-4), dim=-1)

    hm_prev = hm[:, :-1]   # (B, T-1, K, H*W)
    hm_next = hm[:, 1:]
    hm_mid = 0.5 * (hm_prev + hm_next)

    # JS divergence (symmetric, bounded in [0, ln2])
    kl_prev = (hm_prev * (hm_prev.clamp_min(1e-8).log() - hm_mid.clamp_min(1e-8).log())).sum(-1)
    kl_next = (hm_next * (hm_next.clamp_min(1e-8).log() - hm_mid.clamp_min(1e-8).log())).sum(-1)
    jsd = 0.5 * (kl_prev + kl_next)  # (B, T-1, K)

    vis_pair = vis[:, 1:] * vis[:, :-1]
    return (jsd * vis_pair).sum() / vis_pair.sum().clamp_min(1.0)


def temporal_smoothness_loss(
    kps_pred: torch.Tensor,   # (B, T, K, 2)
    vis: torch.Tensor,        # (B, T, K)
    order: int = 2,           # 1=velocity, 2=acceleration
    beta: float = 0.005,
) -> torch.Tensor:
    """
    Penalize non-smooth keypoint trajectories.
    Order 1: penalizes velocity magnitude (encourages stationarity — use sparingly)
    Order 2: penalizes acceleration (encourages constant-velocity — usually better)
    """
    B, T, K, _ = kps_pred.shape
    
    if T < 2:
        return kps_pred.new_zeros(())
    
    # First-order differences (velocity)
    vel = kps_pred[:, 1:] - kps_pred[:, :-1]  # (B, T-1, K, 2)
    vis_pair = vis[:, 1:] * vis[:, :-1]        # both visible
    
    if order == 1:
        loss = F.smooth_l1_loss(vel, torch.zeros_like(vel), beta=beta, reduction='none')
        mask = vis_pair.unsqueeze(-1)
        return (loss * mask).sum() / (mask.sum() * 2 + 1e-6)
    
    elif order == 2 and T >= 3:
        # Second-order differences (acceleration)
        accel = vel[:, 1:] - vel[:, :-1]  # (B, T-2, K, 2)
        vis_triple = vis[:, 2:] * vis[:, 1:-1] * vis[:, :-2]
        mask = vis_triple.unsqueeze(-1)
        loss = F.smooth_l1_loss(accel, torch.zeros_like(accel), beta=beta, reduction='none')
        return (loss * mask).sum() / (mask.sum() * 2 + 1e-6)
    
    return kps_pred.new_zeros(())

def heatmap_kl_div_from_logits_masked(
    hm_logits: torch.Tensor,     # (B,T,K,H,W)
    hm_target: torch.Tensor,     # (B,T,K,H,W) >=0
    vis: torch.Tensor,           # (B,T,K)
    tau: float = 1.0,
    target_power: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    assert hm_logits.shape == hm_target.shape
    B, T, K, H, W = hm_logits.shape

    # flatten
    logits = hm_logits.reshape(B*T*K, H*W) / max(tau, 1e-6)
    tgt    = hm_target.reshape(B*T*K, H*W).clamp_min(0.0)

    # normalize target to prob simplex
    tgt = tgt / (tgt.sum(dim=-1, keepdim=True) + eps)

    # optional sharpening of GT distribution
    if target_power != 1.0:
        tgt = tgt.clamp_min(1e-12).pow(target_power)
        tgt = tgt / (tgt.sum(dim=-1, keepdim=True) + eps)

    logP = F.log_softmax(logits, dim=-1)

    # KL(tgt || P) per (BTK)
    kl = F.kl_div(logP, tgt, reduction="none").sum(dim=-1)  # (BTK,)

    m = vis.reshape(B*T*K).float()
    return (kl * m).sum() / (m.sum().clamp_min(1.0))


def heatmap_kl_div_from_logits(hm_logits: torch.Tensor, hm_target: torch.Tensor, tau: float = 1.0):
    """KL(target || softmax(logits/tau)) averaged over all keypoints."""
    B, T, K, H, W = hm_logits.shape
    logP = F.log_softmax(hm_logits.reshape(B * T * K, H * W) / tau, dim=-1)
    tgt = hm_target.reshape(B * T * K, H * W)
    tgt = tgt / (tgt.sum(dim=-1, keepdim=True) + 1e-8)
    kl = torch.sum(tgt * (torch.log(tgt + 1e-8) - logP), dim=-1)
    return kl.mean()


# ----------------------------
# Utils
# ----------------------------
def heat_px_to_norm(coords_px: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """ (B,T,K,2) pixel -> normalized [-1,1] """
    x = coords_px[..., 0]; y = coords_px[..., 1]
    x_n = 2.0 * (x / (W - 1.0)) - 1.0
    y_n = 2.0 * (y / (H - 1.0)) - 1.0
    return torch.stack([x_n, y_n], dim=-1)

def smooth_l1_vis_weighted(pred: torch.Tensor, target: torch.Tensor, vis: torch.Tensor, beta: float = 0.5, eps: float = 1e-6):
    """ SmoothL1 only where keypoints are visible. """
    loss = F.smooth_l1_loss(pred, target, beta=beta, reduction='none')  # (B,T,K,2)
    w = vis.unsqueeze(-1)  # (B,T,K,1)
    loss = (loss * w).sum() / (w.sum() * 2.0 + eps)  # *2 for x,y
    return loss

def softargmax_2d_from_logits(logits: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """ Soft-argmax on 2D logits → pixel coords. """
    B, T, K, H, W = logits.shape
    sm = F.softmax((logits.reshape(B*T*K, H*W) / max(tau, 1e-6)), dim=-1).reshape(B*T*K, H, W)
    xs = torch.linspace(0, W-1, W, device=logits.device, dtype=logits.dtype)
    ys = torch.linspace(0, H-1, H, device=logits.device, dtype=logits.dtype)
    xg = xs.view(1, 1, W).expand(B*T*K, H, W)
    yg = ys.view(1, H, 1).expand(B*T*K, H, W)
    x = (sm * xg).sum(dim=(1, 2))
    y = (sm * yg).sum(dim=(1, 2))
    return torch.stack([x, y], dim=-1).reshape(B, T, K, 2)


# ----------------------------
# Combined loss: Heatmap + Keypoint
# ----------------------------
def kp_heatmap_loss_combo(
    hm_logits: torch.Tensor,
    hm_target: torch.Tensor,
    kp_target_norm: torch.Tensor,          # (B,T,K,2) normalized coords
    vis: torch.Tensor,                     # (B,T,K) in {0,1}
    tau: float = 1.0,
    w_hm: float = 1.0,
    w_kp: float = 0.5,
    kp_space: str = "norm",
    kps_pred_norm: torch.Tensor | None = None,
    target_power: float = 1.0,
):
    """
    Heatmap KL (with dynamic target_power sharpening) + KP regression loss.
    """
    B, T, K, H, W = hm_logits.shape

    # --- Heatmap KL ---
    L_hm = heatmap_kl_div_from_logits_masked(
        hm_logits, hm_target, vis,
        tau=tau, target_power=target_power
    )

    # --- Keypoint regression ---
    if kps_pred_norm is None:
        kps_px = softargmax_2d_from_logits(hm_logits, tau=tau)  # (B,T,K,2) in px
        if kp_space == "norm":
            x_norm = 2.0 * (kps_px[..., 0] / (W - 1.0)) - 1.0
            y_norm = 2.0 * (kps_px[..., 1] / (H - 1.0)) - 1.0
            xy_pred_norm = torch.stack([x_norm, y_norm], dim=-1)
            vis_mask = vis.unsqueeze(-1).float()
            coord_diff = F.smooth_l1_loss(xy_pred_norm, kp_target_norm, reduction='none')
            L_kp = (coord_diff * vis_mask).sum() / vis_mask.sum().clamp_min(1.0)
        else:
            gx = ((kp_target_norm[..., 0] + 1.0) * 0.5) * (W - 1.0)
            gy = ((kp_target_norm[..., 1] + 1.0) * 0.5) * (H - 1.0)
            gt_px = torch.stack([gx, gy], dim=-1)
            vis_mask = vis.unsqueeze(-1).float()
            coord_diff = F.smooth_l1_loss(kps_px, gt_px, reduction='none')
            L_kp = (coord_diff * vis_mask).sum() / vis_mask.sum().clamp_min(1.0)
    else:
        xy_pred_norm = kps_pred_norm
        vis_mask = vis.unsqueeze(-1).float()
        coord_diff = F.smooth_l1_loss(xy_pred_norm, kp_target_norm, reduction='none')
        L_kp = (coord_diff * vis_mask).sum() / vis_mask.sum().clamp_min(1.0)

    total = w_hm * L_hm + w_kp * L_kp
    return total, {"hm_raw": L_hm.detach(), "kp_raw": L_kp.detach()}


# ----------------------------
# Metrics
# ----------------------------
@torch.no_grad()
def rot_err_deg_all(pred_q: torch.Tensor, gt_q: torch.Tensor):
    """Return per-frame rotation error in degrees: (B,T)."""
    ang_rad = geodesic_quat_loss(pred_q, gt_q, reduction=None)  # (B,T)
    return ang_rad * (180.0 / math.pi)



#----------------------------
# Ray-consistent token contrastive loss
#----------------------------
def _patch_centers_in_crop_px(input_size: int, patch_size: int, device, dtype):
    """Return (P,2) crop-space patch centers in pixels: (u_crop, v_crop)."""
    Hp = input_size // patch_size
    Wp = input_size // patch_size
    ys = (torch.arange(Hp, device=device, dtype=dtype) + 0.5) * patch_size
    xs = (torch.arange(Wp, device=device, dtype=dtype) + 0.5) * patch_size
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    uv = torch.stack([xx, yy], dim=-1).view(-1, 2)  # (P,2)
    return uv, Hp, Wp

def _crop_to_full_uv(
    uv_crop: torch.Tensor,              # (P,2) in [0,input_size]
    crop_xywh_bt: torch.Tensor,         # (B,T,4) normalized wrt FULL
    full_hw: tuple[int, int],           # (H_full,W_full)
    input_size: int,
):
    """
    Map crop-space uv (pixels in resized crop) -> full-image uv (pixels).
    Uses crop box: (x,y,w,h) normalized wrt FULL.
    """
    H_full, W_full = int(full_hw[0]), int(full_hw[1])

    # (B,T) crop box in full pixels
    x0 = crop_xywh_bt[..., 0] * W_full
    y0 = crop_xywh_bt[..., 1] * H_full
    w  = crop_xywh_bt[..., 2] * W_full
    h  = crop_xywh_bt[..., 3] * H_full

    # uv_crop normalized within crop
    u = uv_crop[:, 0] / float(input_size)   # (P,)
    v = uv_crop[:, 1] / float(input_size)   # (P,)

    # broadcast to (B,T,P)
    u_full = x0.unsqueeze(-1) + u.view(1, 1, -1) * w.unsqueeze(-1)
    v_full = y0.unsqueeze(-1) + v.view(1, 1, -1) * h.unsqueeze(-1)
    return u_full, v_full  # (B,T,P), (B,T,P)

def _ray_bins_from_uv(
    u_full: torch.Tensor,               # (B,T,P)
    v_full: torch.Tensor,               # (B,T,P)
    cam_K: torch.Tensor,                # (B,3,3)
    ray_bin_deg: float = 6.0,
):
    """
    Compute per-patch ray bin IDs using inv(K)[u,v,1], normalized direction,
    then quantize (azimuth,elevation).
    Returns: bins (B,P) using T=0 (rays do not depend on time for fixed pixel).
    """
    device = u_full.device
    dtype  = u_full.dtype
    B, T, P = u_full.shape

    # use t=0 for ray binning (pixel->ray is time-invariant)
    u0 = u_full[:, 0]  # (B,P)
    v0 = v_full[:, 0]  # (B,P)

    ones = torch.ones((B, P), device=device, dtype=dtype)
    pix = torch.stack([u0, v0, ones], dim=-1)  # (B,P,3)

    Kinv = torch.inverse(cam_K).to(dtype=dtype)  # (B,3,3)
    dirs = torch.einsum("bij,bpj->bpi", Kinv, pix)  # (B,P,3)
    dirs = F.normalize(dirs, dim=-1)  # (B,P,3)

    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2].clamp_min(1e-8)

    az = torch.atan2(x, z)  # [-pi, pi]
    el = torch.atan2(y, z)  # approx [-pi/2, pi/2]

    bin_rad = math.radians(max(ray_bin_deg, 1e-3))
    az_bin = torch.floor((az + math.pi) / bin_rad).long()
    el_bin = torch.floor((el + (math.pi * 0.5)) / bin_rad).long()

    # combine
    n_el = int(math.ceil(math.pi / bin_rad)) + 1
    bins = az_bin * n_el + el_bin  # (B,P)
    return bins
# ----------------------------
# Ray-consistent token contrastive loss (FIXED: rays-only, no recompute)
# ----------------------------
def ray_token_contrastive_loss(
    patch_tokens: torch.Tensor,     # (B,T,P,D)
    rays: torch.Tensor,             # (B,T,P,3) fp32 (or fp16/bf16 ok; fp32 preferred)
    ray_bin_deg: float = 6.0,
    temp: float = 0.07,
    t_pair: str = "ends",           # "ends" | "random"
    eps: float = 1e-8,
):
    """
    Self-supervised multi-positive InfoNCE using precomputed rays from the model.

    Positives: same sample AND same ray-bin (quantized by azimuth/elevation from rays).
    Negatives: everything else (other ray bins + other samples).

    Returns: scalar.
    """
    assert patch_tokens.ndim == 4, f"Expected (B,T,P,D), got {patch_tokens.shape}"
    assert rays is not None, "rays must be provided (use out['rays'])."
    assert rays.ndim == 4 and rays.shape[-1] == 3, f"Expected rays (B,T,P,3), got {rays.shape}"
    assert rays.shape[:3] == patch_tokens.shape[:3], f"rays {rays.shape[:3]} != tokens {patch_tokens.shape[:3]}"

    B, T, P, D = patch_tokens.shape
    if T < 2:
        return patch_tokens.new_zeros(())

    device = patch_tokens.device

    # pick time pair
    if t_pair == "random":
        t0 = 0
        t1 = torch.randint(low=1, high=T, size=(1,), device=device).item()
    else:
        t0, t1 = 0, T - 1

    # normalize tokens
    z0 = F.normalize(patch_tokens[:, t0].float(), dim=-1)  # (B,P,D)
    z1 = F.normalize(patch_tokens[:, t1].float(), dim=-1)  # (B,P,D)

    # ---- bin rays (use t=0; ray direction per patch is time-invariant given pixels)
    r = rays[:, 0].float()                                  # (B,P,3)
    r = F.normalize(r, dim=-1)

    x = r[..., 0]
    y = r[..., 1]
    z = r[..., 2].clamp_min(1e-8)

    az = torch.atan2(x, z)           # [-pi, pi]
    el = torch.atan2(y, z)           # approx [-pi/2, pi/2]

    bin_rad = math.radians(max(ray_bin_deg, 1e-3))
    az_bin = torch.floor((az + math.pi) / bin_rad).long()
    el_bin = torch.floor((el + (math.pi * 0.5)) / bin_rad).long()

    n_el = int(math.ceil(math.pi / bin_rad)) + 1
    bins = az_bin * n_el + el_bin                            # (B,P)

    # ---- flatten
    a = z0.reshape(B * P, D)                                 # anchors
    b = z1.reshape(B * P, D)                                 # targets
    logits = (a @ b.t()) / max(float(temp), 1e-6)            # (BP, BP)

    b_idx = torch.arange(B, device=device).view(B, 1).expand(B, P).reshape(-1)  # (BP,)
    bin_id = bins.reshape(-1)                                                    # (BP,)

    same_b   = b_idx.unsqueeze(1).eq(b_idx.unsqueeze(0))                         # (BP,BP)
    same_bin = bin_id.unsqueeze(1).eq(bin_id.unsqueeze(0))                       # (BP,BP)
    pos = same_b & same_bin                                                      # (BP,BP)

    logp = F.log_softmax(logits, dim=1)
    neg_inf = torch.finfo(logp.dtype).min
    logp_pos = torch.logsumexp(logp.masked_fill(~pos, neg_inf), dim=1)            # (BP,)
    loss01 = -logp_pos.mean()

    # symmetric
    logits_sym = (b @ a.t()) / max(float(temp), 1e-6)
    logp_sym = F.log_softmax(logits_sym, dim=1)
    logp_pos_sym = torch.logsumexp(logp_sym.masked_fill(~pos, neg_inf), dim=1)
    loss10 = -logp_pos_sym.mean()

    return 0.5 * (loss01 + loss10)

# old one having issue such as loss keep growing in negative above -8.00 
# def gaussian_nll_kp_px(
#     mu_px: torch.Tensor,      # (B,T,K,2) full px
#     gt_px: torch.Tensor,      # (B,T,K,2) full px
#     sigma: torch.Tensor,      # (B,T,K,1 or 2) px
#     vis: torch.Tensor,        # (B,T,K) 0/1
#     sigma_floor: float = 1.0,
#     full_hw=None,             # (H_full, W_full)  <-- NEW
#     norm_mode: str = "diag",  # "diag" or "max"   <-- NEW
#     nll_cap: float = 20.0,    # <-- used by Option B clamp (keep here)
# ):
#     """
#     NLL in *normalized* full-image units.
#     - Convert (mu, gt, sigma_floor, sigma) from px -> normalized by image scale.
#     - This prevents NLL from dominating when using full-res pixel space.
#     """

#     assert full_hw is not None, "Pass full_hw=(H_full,W_full) to gaussian_nll_kp_px."
#     H_full, W_full = int(full_hw[0]), int(full_hw[1])

#     if norm_mode == "diag":
#         s = math.sqrt(float(H_full * H_full + W_full * W_full))
#     elif norm_mode == "max":
#         s = float(max(H_full, W_full))
#     else:
#         raise ValueError(f"norm_mode must be 'diag' or 'max', got {norm_mode}")

#     mu = mu_px / s
#     gt = gt_px / s

#     # sigma handling: allow scalar/diag
#     if sigma.shape[-1] == 1:
#         sig = sigma.expand_as(mu[..., :1]).repeat(1, 1, 1, 2)  # (B,T,K,2)
#     else:
#         sig = sigma

#     sig = sig / s
#     sig_floor = float(sigma_floor) / s
#     sig = torch.clamp(sig, min=sig_floor)

#     # Gaussian NLL (diag)
#     diff = (mu - gt) / sig
#     # nll = 0.5 * (diff ** 2).sum(dim=-1) + torch.log(sig).sum(dim=-1)  # (B,T,K)
#     diff = (mu - gt) / sig  # (B,T,K,2)

#     LOG2PI = math.log(2.0 * math.pi)
#     # Standard diag-Gaussian NLL per keypoint (sum over x,y)
#     nll = 0.5 * (diff**2 + 2.0 * torch.log(sig) + LOG2PI).sum(dim=-1)  # (B,T,K)

#     # visible mask + mean
#     vis = vis.float()
#     nll = nll * vis
#     denom = vis.sum().clamp(min=1.0)
#     nll = nll.sum() / denom

#     # Option B clamp hook (see below) – keeps training stable early
#     if nll_cap is not None:
#         nll = torch.clamp(nll, min=-float(nll_cap), max=float(nll_cap))

#     return nll

# to reduce NLL dominance when using full-res pixel space
def gaussian_nll_kp_px(
    mu_px: torch.Tensor,          # (B,T,K,2) predicted mean in FULL px
    gt_px: torch.Tensor,          # (B,T,K,2) GT in FULL px
    sigma: torch.Tensor,          # (B,T,K,1) or (B,T,K,2) predicted sigma in px
    vis: torch.Tensor,            # (B,T,K) 0/1
    sigma_floor: float = 1.0,     # in PIXELS (>=1.0 strongly recommended)
    sigma_ceil: float = 80.0,  # optional (e.g., 80.0)
    nll_cap: float = 50.0,        # clamp very large losses for stability
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Proper 2D diagonal Gaussian NLL in PIXEL space:

      NLL = 0.5 * [ ((dx/sx)^2 + (dy/sy)^2) + 2*log(sx) + 2*log(sy) + 2*log(2π) ]

    With sigma clamped to [sigma_floor, sigma_ceil].

    This prevents the pathological negative drift from log(sigma) in normalized units.
    """

    assert mu_px.shape == gt_px.shape, f"mu_px {mu_px.shape} vs gt_px {gt_px.shape}"
    assert mu_px.shape[-1] == 2
    assert vis.shape == mu_px.shape[:-1], f"vis {vis.shape} vs {mu_px.shape[:-1]}"

    # Expand sigma to (B,T,K,2)
    if sigma.shape[-1] == 1:
        sig = sigma.expand(-1, -1, -1, 2)
    else:
        sig = sigma

    # Clamp sigma in PIXELS
    sig = sig.clamp_min(float(sigma_floor))
    if sigma_ceil is not None:
        sig = sig.clamp_max(float(sigma_ceil))

    # Error in PIXELS
    diff = (mu_px - gt_px)  # (B,T,K,2)

    # Per-dim term
    quad = (diff / (sig + eps)).pow(2)          # (B,T,K,2)
    logdet = 2.0 * torch.log(sig + eps)         # (B,T,K,2) but per-dim; we will sum over dims

    # 2D constant: log((2π)^2) = 2*log(2π)
    const = 2.0 * math.log(2.0 * math.pi)       # scalar

    nll = 0.5 * (quad.sum(dim=-1) + logdet.sum(dim=-1) + const)  # (B,T,K)

    # Safety clamps (prevents exploding gradients)
    if nll_cap is not None:
        nll = nll.clamp(min=0.0, max=float(nll_cap))  # IMPORTANT: lower-bound at 0

    m = vis.float()
    return (nll * m).sum() / (m.sum().clamp_min(1.0))

def kp_attn_kl_to_gaussian_patches(
    attn_kp2patch: torch.Tensor,   # (B,T,K,P)  (prob OR logits; see flag below)
    kps_gt_normhm: torch.Tensor,   # (B,T,K,2)  in [-1,1] w.r.t heatmap grid
    vis: torch.Tensor,             # (B,T,K)    {0,1}
    input_size: int,               # e.g. 224/448/512
    heatmap_hw: tuple[int, int],   # (Hm,Wm)
    patch_size: int,               # e.g. 16
    sigma_px: float = None,        # if None -> 0.75 * patch_size
    attn_is_logits: bool = False,  # set True if you return pre-softmax scores
    eps: float = 1e-8,
):
    """
    KL( target_gaussian || attn_dist ) averaged over visible kps.
    Target is Gaussian over patch centers in INPUT pixel coords.
    """
    assert attn_kp2patch.ndim == 4, attn_kp2patch.shape
    B, T, K, P = attn_kp2patch.shape

    Hp = input_size // patch_size
    Wp = input_size // patch_size
    P_expected = Hp * Wp
    assert P == P_expected, f"attn P={P} but expected {P_expected} (Hp={Hp},Wp={Wp})"

    Hm, Wm = heatmap_hw
    device = attn_kp2patch.device
    dtype = attn_kp2patch.dtype

    # --- convert kps_normhm [-1,1] -> heatmap px -> input px ---
    # heatmap px
    x_hm = (kps_gt_normhm[..., 0] + 1.0) * 0.5 * (Wm - 1.0)
    y_hm = (kps_gt_normhm[..., 1] + 1.0) * 0.5 * (Hm - 1.0)

    # hm px -> input px (inverse of your dataset scaling)
    # crop_px = hm_px * ((input_size - 1)/(Wm - 1)), same for y
    sx = float(max(input_size - 1, 1)) / float(max(Wm - 1, 1))
    sy = float(max(input_size - 1, 1)) / float(max(Hm - 1, 1))
    x_in = x_hm * sx
    y_in = y_hm * sy
    kp_in = torch.stack([x_in, y_in], dim=-1)  # (B,T,K,2)

    # --- patch centers in input px: (P,2) ---
    ys = (torch.arange(Hp, device=device, dtype=torch.float32) + 0.5) * patch_size
    xs = (torch.arange(Wp, device=device, dtype=torch.float32) + 0.5) * patch_size
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    centers = torch.stack([xx, yy], dim=-1).view(1, 1, 1, P, 2)  # (1,1,1,P,2)

    kp_in_f = kp_in.to(torch.float32).unsqueeze(3)  # (B,T,K,1,2)
    d2 = ((kp_in_f - centers) ** 2).sum(dim=-1)     # (B,T,K,P)

    if sigma_px is None:
        sigma_px = 0.75 * float(patch_size)
    sig2 = float(max(sigma_px, 1e-6)) ** 2

    tgt_logits = -0.5 * d2 / sig2
    tgt = F.softmax(tgt_logits, dim=-1).to(dtype)   # (B,T,K,P)

    # --- predicted distribution ---
    if attn_is_logits:
        logP = F.log_softmax(attn_kp2patch, dim=-1)
    else:
        # assume already prob
        Pp = attn_kp2patch.clamp_min(eps)
        Pp = Pp / (Pp.sum(dim=-1, keepdim=True) + eps)
        logP = torch.log(Pp)

    # KL(tgt || P)
    kl = (tgt * (torch.log(tgt.clamp_min(eps)) - logP)).sum(dim=-1)  # (B,T,K)

    m = vis.float()
    return (kl * m).sum() / (m.sum().clamp_min(1.0))