# model_geospaceformer_theseus.py
# Implements exactly:
# 0) DINOv3 backbone + YOUR existing Pre-MHSA spatiotemporal adapters (unchanged logic)
# 1) Ray-attention Neck Option A:
#    - spatial self-attn per frame with ray bias
#    - temporal attention across frames per patch index
# 2) THESEUS layer INSIDE the model to output per-frame (R,t)
#    - tracking init: pose_t initialized from pose_{t-1} (best solution)
#    - returns a temporal prior loss term to be used OUTSIDE Theseus (or later as factor)
# 3) Keep your TinyHeatmapHead for K heatmaps per frame + parallel uncertainty head (sigma)
#    - mu via DSNT from heatmaps
# 4) Coordinate consistency:
#    - DSNT -> heatmap px -> input px -> FULL image px (camera image space)
#    - camera intrinsics K are assumed in FULL image pixel space
#
# Forward signature:
#   out = model(video, cam_K, full_hw=(Hfull,Wfull), crop_xywh=optional(B,T,4) norm [0,1])
#
# Outputs include:
#   - heatmaps, hm_prob
#   - mu_norm (DSNT), mu_px_full (per-frame, per-kpt in full image pixels)
#   - sigma (per-frame, per-kpt)
#   - pred_R (B,T,3,3), pred_t (B,T,3)
#   - temporal_pose_prior_loss (scalar)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DINOv3ViTModel

# ---------------- Theseus import (required for pose head) ----------------
try:
    import theseus as th
except Exception as e:
    th = None
    _THESEUS_IMPORT_ERROR = e

# ---------------- DSNT utilities ----------------
def _softmax_2d(hm, eps=1e-8):
    # hm: (B,T,K,H,W)
    B, T, K, H, W = hm.shape
    x = hm.contiguous().view(B * T * K, H * W)
    x = F.softmax(x, dim=-1)
    return x.view(B, T, K, H, W).clamp_min(eps)

def dsnt(hm_prob):
    """
    hm_prob: (B,T,K,H,W), sums to 1 over H*W
    returns: coords in [-1,1], shape (B,T,K,2) with (x,y)
    """
    B, T, K, H, W = hm_prob.shape
    xs = torch.linspace(-1, 1, W, device=hm_prob.device, dtype=hm_prob.dtype)
    ys = torch.linspace(-1, 1, H, device=hm_prob.device, dtype=hm_prob.dtype)
    px = hm_prob.sum(dim=-2)  # (B,T,K,W)
    py = hm_prob.sum(dim=-1)  # (B,T,K,H)
    x = (px * xs).sum(dim=-1)
    y = (py * ys).sum(dim=-1)
    return torch.stack([x, y], dim=-1)  # (B,T,K,2)

def norm_to_pixel(coords_norm, hw):
    """coords_norm (...,2) in [-1,1] -> pixel coords in [0..W-1, 0..H-1]."""
    H, W = hw
    x = (coords_norm[..., 0] + 1.0) * 0.5 * (W - 1)
    y = (coords_norm[..., 1] + 1.0) * 0.5 * (H - 1)
    return torch.stack([x, y], dim=-1)

def crop_norm_to_pixel(crop_xywh_norm, full_hw):
    """
    crop_xywh_norm: (B,T,4) normalized [0,1] wrt FULL image: x,y,w,h
    returns (B,T,4) pixels in FULL space
    """
    Hf, Wf = full_hw
    x = crop_xywh_norm[..., 0] * (Wf - 1)
    y = crop_xywh_norm[..., 1] * (Hf - 1)
    w = crop_xywh_norm[..., 2] * (Wf - 1)
    h = crop_xywh_norm[..., 3] * (Hf - 1)
    return torch.stack([x, y, w, h], dim=-1)

# ---------------- Geometry helpers for rays ----------------
def make_patch_centers_in_input(grid_hw, in_hw, device, dtype):
    """
    grid_hw: (Hp,Wp), in_hw: (H_in,W_in)
    returns (P,2) centers in INPUT pixel coords (x,y)
    """
    Hp, Wp = grid_hw
    H_in, W_in = in_hw
    ys = (torch.arange(Hp, device=device, dtype=dtype) + 0.5) * (H_in / Hp)
    xs = (torch.arange(Wp, device=device, dtype=dtype) + 0.5) * (W_in / Wp)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).view(Hp * Wp, 2)

def unproject_to_rays(uv_px_full, K_px):
    """
    uv_px_full: (B,T,P,2)
    K_px: (B,3,3) or (B,T,3,3)

    returns: (B,T,P,3)
    """
    B, T, P, _ = uv_px_full.shape

    if K_px.dim() == 3:
        K_px = K_px.unsqueeze(1).expand(B, T, 3, 3)   # (B,T,3,3)
    elif K_px.dim() == 4:
        assert K_px.shape[1] == T, f"K_px has T={K_px.shape[1]} but uv has T={T}"
    else:
        raise ValueError(f"K_px must be (B,3,3) or (B,T,3,3), got {K_px.shape}")

    fx = K_px[:, :, 0, 0].view(B, T, 1, 1)
    fy = K_px[:, :, 1, 1].view(B, T, 1, 1)
    cx = K_px[:, :, 0, 2].view(B, T, 1, 1)
    cy = K_px[:, :, 1, 2].view(B, T, 1, 1)

    x = (uv_px_full[..., 0:1] - cx) / fx
    y = (uv_px_full[..., 1:2] - cy) / fy
    z = torch.ones_like(x)

    v = torch.cat([x, y, z], dim=-1)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return v

def build_patch_rays_full(B, T, grid_hw, in_hw, full_hw, cam_K, crop_xywh_norm, device, dtype):
    """
    Returns rays per patch center in FULL camera space.
    - video is (H_in,W_in) input resolution (after crop-resize if applicable)
    - crop_xywh_norm: (B,T,4) optional normalized crop wrt FULL; if None => treat as full-frame.
    """
    P = grid_hw[0] * grid_hw[1]
    centers_in = make_patch_centers_in_input(grid_hw, in_hw, device, dtype)  # (P,2) in input px
    centers_in = centers_in.view(1, 1, P, 2).expand(B, T, P, 2)              # (B,T,P,2)

    H_in, W_in = in_hw
    Hf, Wf = full_hw

    if crop_xywh_norm is None:
        # input is a resized full frame -> scale to FULL pixels
        sx = (Wf - 1) / max(W_in - 1, 1)
        sy = (Hf - 1) / max(H_in - 1, 1)
        centers_full = torch.stack([centers_in[..., 0] * sx, centers_in[..., 1] * sy], dim=-1)
    else:
        # input is resized crop -> map to FULL pixels using crop box
        crop_xywh_px = crop_norm_to_pixel(crop_xywh_norm, full_hw)  # (B,T,4)
        x0 = crop_xywh_px[..., 0:1]
        y0 = crop_xywh_px[..., 1:2]
        cw = crop_xywh_px[..., 2:3].clamp_min(1.0)
        ch = crop_xywh_px[..., 3:4].clamp_min(1.0)
        sx = cw / max(W_in - 1, 1)
        sy = ch / max(H_in - 1, 1)
        centers_full = torch.stack([x0 + centers_in[..., 0] * sx, y0 + centers_in[..., 1] * sy], dim=-1)

    rays = unproject_to_rays(centers_full, cam_K)  # (B,T,P,3)
    return rays

# ---------------- Your adapters (UNCHANGED) ----------------
class SpatioTemporal3DAdapter(nn.Module):
    def __init__(
        self,
        d: int,
        grid_hw: tuple[int, int],
        bottleneck: int = 64,
        drop: float = 0.1,
        use_ln: bool = True,
        scale_init: float = 1e-3,
        ks: int = 3,
        kt: int = 3,
        fuse: str = "sum",
    ):
        super().__init__()
        self.Hp, self.Wp = grid_hw
        self.P = self.Hp * self.Wp
        self.fuse = fuse

        self.ln = nn.LayerNorm(d) if use_ln else nn.Identity()
        self.in_proj = nn.Linear(d, bottleneck)

        pad_s = ks // 2
        pad_t = kt // 2

        self.spatial_conv = nn.Conv3d(
            bottleneck, bottleneck, kernel_size=(1, ks, ks),
            padding=(0, pad_s, pad_s), bias=False
        )
        self.temporal_conv = nn.Conv3d(
            bottleneck, bottleneck, kernel_size=(kt, 1, 1),
            padding=(pad_t, 0, 0), bias=False
        )

        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

        self.fuse_proj = nn.Conv3d(2 * bottleneck, bottleneck, kernel_size=1, bias=False) if fuse == "concat" else None
        self.out_proj = nn.Linear(bottleneck, d)
        self.scale = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))

    def forward(self, x_patch: torch.Tensor) -> torch.Tensor:
        B, T, P, D = x_patch.shape
        assert P == self.P, f"P={P} must equal Hp*Wp={self.P}"

        y = self.ln(x_patch)
        y = self.in_proj(y)  # (B,T,P,C)
        C = y.shape[-1]

        y = y.view(B, T, self.Hp, self.Wp, C).permute(0, 4, 1, 2, 3).contiguous()  # (B,C,T,H,W)

        ys = self.spatial_conv(y)
        yt = self.temporal_conv(y)

        if self.fuse == "concat":
            z = torch.cat([ys, yt], dim=1)
            z = self.fuse_proj(z)
        else:
            z = ys + yt

        z = self.act(z)
        z = self.drop(z)

        z = z.permute(0, 2, 3, 4, 1).contiguous().view(B, T, P, C)
        z = self.out_proj(z)

        return x_patch + self.scale * z

class PatchOnlyPreMHSAAdapter(nn.Module):
    def __init__(
        self,
        d: int,
        grid_hw: tuple[int, int],
        T: int,
        n_register_tokens: int | None = None,
        bottleneck: int = 64,
        drop: float = 0.1,
        use_ln: bool = True,
        scale_init: float = 1e-3,
        ks: int = 3,
        kt: int = 3,
        fuse: str = "sum",
    ):
        super().__init__()
        self.T = T
        self.n_register_tokens = n_register_tokens
        self.Hp, self.Wp = grid_hw
        self.P = self.Hp * self.Wp

        self.patch_adapter = SpatioTemporal3DAdapter(
            d=d, grid_hw=grid_hw, bottleneck=bottleneck, drop=drop,
            use_ln=use_ln, scale_init=scale_init, ks=ks, kt=kt, fuse=fuse
        )
        self.last_patch_bt = None

    def forward(self, x_btnd: torch.Tensor) -> torch.Tensor:
        BT, N, D = x_btnd.shape
        assert BT % self.T == 0, f"BT={BT} must be divisible by T={self.T}"
        B = BT // self.T

        nreg = self.n_register_tokens
        if nreg is None:
            nreg = N - 1 - self.P
            assert nreg >= 0, f"Cannot infer n_register_tokens: N={N}, P={self.P}"

        n_prefix = 1 + nreg
        prefix = x_btnd[:, :n_prefix, :]
        patch  = x_btnd[:, n_prefix:n_prefix + self.P, :]
        suffix = x_btnd[:, n_prefix + self.P:, :]

        patch = patch.view(B, self.T, self.P, D)
        patch = self.patch_adapter(patch)
        patch_bt = patch.view(BT, self.P, D)

        self.last_patch_bt = patch_bt
        return torch.cat([prefix, patch_bt, suffix], dim=1)

# ---------------- HF block discovery ----------------
def _find_attention_module(block: nn.Module):
    for name in ["attention", "self_attn", "attn", "mha", "mhsa"]:
        if hasattr(block, name):
            return getattr(block, name)
    for n, m in block.named_children():
        if "attn" in n.lower() or "attention" in n.lower() or "mhsa" in n.lower():
            return m
    return None

def _get_blocks_from_hf_model(model: nn.Module):
    candidates = [
        ("encoder", "layer"),
        ("encoder", "layers"),
        ("vision_model", "encoder", "layer"),
        ("vision_model", "encoder", "layers"),
        ("backbone", "encoder", "layer"),
        ("backbone", "encoder", "layers"),
        ("layers",),
        ("layer",),
    ]
    for path in candidates:
        m = model
        ok = True
        for name in path:
            if not hasattr(m, name):
                ok = False
                break
            m = getattr(m, name)
        if ok and isinstance(m, (nn.ModuleList, list, tuple)) and len(m) > 0:
            return m

    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            if _find_attention_module(mod[0]) is not None:
                print(f"[INFO] Using blocks from: {name}")
                return mod

    raise AttributeError("Could not find transformer blocks in DINOv3ViTModel. Update discovery.")

# ---------------- Backbone wrapper with YOUR pre-MHSA hooks ----------------
class DINOv3backbone(nn.Module):
    def __init__(
        self,
        hf_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        adapter_bottleneck=64,
        adapter_drop=0.1,
        adapter_in_last=None,
        num_frames=16,
        grid_hw=(14, 14),
        n_register_tokens: int | None = None,
        ks=3, kt=3, fuse="sum",
    ):
        super().__init__()
        self.backbone = DINOv3ViTModel.from_pretrained(hf_name)
        d = self.backbone.config.hidden_size

        for p in self.backbone.parameters():
            p.requires_grad = False

        blocks = _get_blocks_from_hf_model(self.backbone)
        n_blocks = len(blocks)

        if adapter_in_last is None:
            use_idx = set(range(n_blocks))
        else:
            use_idx = set(range(max(0, n_blocks - adapter_in_last), n_blocks))

        self.adapters = nn.ModuleList()
        for i in range(n_blocks):
            if i in use_idx:
                self.adapters.append(
                    PatchOnlyPreMHSAAdapter(
                        d=d, grid_hw=grid_hw, T=num_frames, n_register_tokens=n_register_tokens,
                        bottleneck=adapter_bottleneck, drop=adapter_drop,
                        ks=ks, kt=kt, fuse=fuse
                    )
                )
            else:
                self.adapters.append(nn.Identity())

        self._install_prehooks(blocks)

    def _install_prehooks(self, blocks):
        for i, blk in enumerate(blocks):
            attn = _find_attention_module(blk)
            if attn is None:
                raise RuntimeError(f"Could not locate attention module in block {i}.")
            adapter = self.adapters[i]

            def make_mhsa_hook(adapter_i):
                def hook(module, inputs):
                    x = inputs[0]  # (BT,N,D)
                    x2 = adapter_i(x)
                    return (x2,) + inputs[1:]
                return hook

            attn.register_forward_pre_hook(make_mhsa_hook(adapter), with_kwargs=False)

    def get_last_adapter_patch_bt(self):
        for ad in reversed(self.adapters):
            if isinstance(ad, PatchOnlyPreMHSAAdapter) and getattr(ad, "last_patch_bt", None) is not None:
                return ad.last_patch_bt
        return None

    def forward_tokens(self, x):
        out = self.backbone(pixel_values=x, return_dict=True)
        return out.last_hidden_state  # (BT,N,D)

# ---------------- Ray-attention Neck Option A ----------------
class SpatialRayBiasAttention(nn.Module):
    """
    Spatial self-attn per frame with additive bias from ray similarity.
    x:    (B,T,P,D)
    rays: (B,T,P,3)
    """
    def __init__(self, dim, num_heads=8, attn_drop=0.1, proj_drop=0.1, bias_scale=2.0, bias_mode="cos"):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.h = num_heads
        self.dh = dim // num_heads
        self.bias_scale = bias_scale
        self.bias_mode = bias_mode

        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def _bias(self, rays):
        sim = torch.einsum("btpc,btqc->btpq", rays, rays).clamp(-1.0, 1.0)  # (B,T,P,P)
        if self.bias_mode == "cos":
            b = self.bias_scale * sim
        elif self.bias_mode == "neg_l2":
            dist2 = (2.0 - 2.0 * sim).clamp_min(0.0)
            b = -self.bias_scale * dist2
        else:
            raise ValueError(self.bias_mode)
        return b.unsqueeze(2)  # (B,T,1,P,P)

    def forward(self, x, rays):
        B, T, P, D = x.shape
        qkv = self.qkv(x).view(B, T, P, 3, self.h, self.dh).permute(3, 0, 1, 4, 2, 5).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B,T,h,P,dh)
        attn = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.dh))  # (B,T,h,P,P)
        attn = attn + self._bias(rays)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)  # (B,T,h,P,dh)
        out = out.transpose(2, 3).contiguous().view(B, T, P, D)
        out = self.proj_drop(self.proj(out))
        return out

class TemporalPatchAttention(nn.Module):
    """
    Temporal attention across frames for each patch index independently.
    x: (B,T,P,D)
    """
    def __init__(self, dim, num_heads=8, drop=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=drop, batch_first=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, T, P, D = x.shape
        xp = x.permute(0, 2, 1, 3).contiguous().view(B * P, T, D)  # (B*P,T,D)
        y, _ = self.mha(xp, xp, xp, need_weights=False)
        y = self.drop(y)
        return y.view(B, P, T, D).permute(0, 2, 1, 3).contiguous()  # (B,T,P,D)

class RayAttentionNeckOptionA(nn.Module):
    def __init__(self, dim, num_heads=8, drop=0.1, mlp_ratio=4.0, bias_scale=2.0, bias_mode="cos"):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.spatial = SpatialRayBiasAttention(dim, num_heads=num_heads, attn_drop=drop, proj_drop=drop,
                                              bias_scale=bias_scale, bias_mode=bias_mode)
        self.ln2 = nn.LayerNorm(dim)
        self.temporal = TemporalPatchAttention(dim, num_heads=num_heads, drop=drop)
        self.ln3 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )
        # small residual scales for stability with frozen backbone
        self.g_s = nn.Parameter(torch.tensor(1e-3, dtype=torch.float32))
        self.g_t = nn.Parameter(torch.tensor(1e-3, dtype=torch.float32))
        self.g_m = nn.Parameter(torch.tensor(1e-3, dtype=torch.float32))

    def forward(self, x_btpd, rays_btp3):
        x = x_btpd
        x = x + self.g_s * self.spatial(self.ln1(x), rays_btp3)
        x = x + self.g_t * self.temporal(self.ln2(x))
        x = x + self.g_m * self.mlp(self.ln3(x))
        return x

# ---------------- Heatmap + uncertainty heads ----------------
class TinyHeatmapHead(nn.Module):
    def __init__(self, d, grid_hw=(14, 14), num_kp=8, out_hw=(224, 224), mid_c=128):
        super().__init__()
        self.Hp, self.Wp = grid_hw

        self.proj = nn.Linear(d, mid_c)
        self.conv = nn.Conv2d(mid_c, mid_c, 3, padding=1)
        self.act = nn.GELU()
        self.up1 = nn.ConvTranspose2d(mid_c, mid_c, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(mid_c, mid_c, 4, 2, 1)
        self.up3 = nn.ConvTranspose2d(mid_c, mid_c, 4, 2, 1)

        native_hw = (self.Hp * 8, self.Wp * 8)
        if out_hw is None or tuple(out_hw) == native_hw:
            self.final_up = nn.Identity()
        else:
            self.final_up = nn.Upsample(size=out_hw, mode="bilinear", align_corners=False)

        self.hm = nn.Conv2d(mid_c, num_kp, 1)

    def forward(self, patch_tokens):
        B, T, P, D = patch_tokens.shape
        BT = B * T
        x = self.proj(patch_tokens).view(BT, P, -1).transpose(1, 2).reshape(BT, -1, self.Hp, self.Wp)
        x = self.act(self.conv(x))
        x = self.act(self.up1(x))
        x = self.act(self.up2(x))
        x = self.act(self.up3(x))
        x = self.final_up(x)
        x = self.hm(x)
        return x.view(B, T, x.shape[1], x.shape[-2], x.shape[-1])

class UncertaintyHead(nn.Module):
    """
    Predict sigma per keypoint per frame:
      mode="scalar": (B,T,K,1)
      mode="diag":   (B,T,K,2)
    Uses pooled patch tokens per frame.
    """
    def __init__(self, d, num_kp=8, mode="diag", hidden=256, drop=0.1, sigma_min=1e-4):
        super().__init__()
        assert mode in ["scalar", "diag"]
        self.mode = mode
        self.K = num_kp
        self.out_dim = 1 if mode == "scalar" else 2
        self.sigma_min = sigma_min
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, num_kp * self.out_dim),
        )

    def forward(self, patch_tokens_btpd):
        pooled = patch_tokens_btpd.mean(dim=2)  # (B,T,D)
        raw = self.net(pooled).view(patch_tokens_btpd.shape[0], patch_tokens_btpd.shape[1], self.K, self.out_dim)
        sigma = F.softplus(raw) + self.sigma_min
        return sigma

# ---------------- Theseus reprojection error ----------------
def reprojection_error(pose: "th.SE3", pts_3d, pts_2d, weights, K):
    """
    pose: th.SE3 (B)
    pts_3d: (B,K,3)
    pts_2d: (B,K,2) in FULL pixel coords (x,y)
    weights: (B,K,2) or (B,K,1) -> we will broadcast to (B,K,2)
    K: (B,3,3) intrinsics in FULL pixel coords

    returns residuals: (B, K*2)
    """
    # transform points
    # Theseus SE3 supports transform_from / transform_to depending on version.
    # We'll use transform_from if available; otherwise fall back to pose.to_matrix().
    if hasattr(pose, "transform_from"):
        X = pose.transform_from(pts_3d)  # (B,K,3)
    else:
        T = pose.to_matrix()  # (B,4,4)
        R = T[:, :3, :3]
        t = T[:, :3, 3:4]
        X = (R @ pts_3d.transpose(1, 2) + t).transpose(1, 2)

    Xx, Xy, Xz = X[..., 0], X[..., 1], X[..., 2].clamp_min(1e-6)
    fx = K[:, 0, 0:1]
    fy = K[:, 1, 1:2]
    cx = K[:, 0, 2:3]
    cy = K[:, 1, 2:3]

    u = fx * (Xx / Xz) + cx  # (B,K)
    v = fy * (Xy / Xz) + cy  # (B,K)
    proj = torch.stack([u, v], dim=-1)  # (B,K,2)

    if weights.shape[-1] == 1:
        w = weights.expand(-1, -1, 2)
    else:
        w = weights

    err = (proj - pts_2d) * w  # weighted reprojection
    return err.view(err.shape[0], -1)  # (B, K*2)

def temporal_pose_prior_loss(poses_se3: torch.Tensor, weight=1.0, order=1):
    """
    poses_se3: (B,T,6) se(3) tangent vectors used for init/regularization (approx).
    order=1: ||xi_t - xi_{t-1}||^2
    order=2: ||xi_t - 2xi_{t-1} + xi_{t-2}||^2
    """
    if poses_se3 is None or poses_se3.shape[1] < (order + 1):
        return poses_se3.new_tensor(0.0) if poses_se3 is not None else torch.tensor(0.0)
    if order == 1:
        d = poses_se3[:, 1:, :] - poses_se3[:, :-1, :]
    elif order == 2:
        d = poses_se3[:, 2:, :] - 2.0 * poses_se3[:, 1:-1, :] + poses_se3[:, :-2, :]
    else:
        raise ValueError("order must be 1 or 2")
    return weight * d.pow(2).mean()

# ---------------- Full model: backbone+adapters + ray-neck + heatmaps+sigma + Theseus pose ----------------
class GeoSpaceFormer(nn.Module):
    def __init__(
        self,
        model_pts_3d: torch.Tensor,   # (1,K,3) or (K,3)
        num_kp=8,
        num_frames=16,
        hf_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        grid_hw=(14, 14),
        heatmap_hw=(224, 224),
        hm_mid_c=128,
        # adapters
        adapter_bottleneck=64,
        adapter_drop=0.1,
        adapter_in_last=None,
        n_register_tokens: int | None = None,
        ks=3, kt=3, fuse="sum",
        # ray neck
        neck_heads=8,
        neck_drop=0.1,
        ray_bias_scale=2.0,
        ray_bias_mode="cos",
        # uncertainty
        unc_mode="diag",   # "scalar" or "diag"
        unc_hidden=256,
        # theseus
        lm_max_iters=5,
        # pose prior outside Theseus
        pose_prior_weight=0.0,
        pose_prior_order=1,
    ):
        super().__init__()
        if th is None:
            raise ImportError(f"Theseus import failed. Install theseus-ai. Original error: {_THESEUS_IMPORT_ERROR}")

        self.num_kp = num_kp
        self.num_frames = num_frames
        self.grid_hw = grid_hw
        self.P = grid_hw[0] * grid_hw[1]
        self.heatmap_hw = tuple(heatmap_hw)
        self.n_register_tokens = n_register_tokens
        self.pose_prior_weight = float(pose_prior_weight)
        self.pose_prior_order = int(pose_prior_order)

        if model_pts_3d.dim() == 2:
            model_pts_3d = model_pts_3d.unsqueeze(0)  # (1,K,3)
        assert model_pts_3d.shape[-2:] == (num_kp, 3), f"pts_3d must be (1,{num_kp},3)"
        self.register_buffer("pts_3d", model_pts_3d)

        # 0) backbone + YOUR pre-MHSA adapters
        self.backbone = DINOv3backbone(
            hf_name=hf_name,
            adapter_bottleneck=adapter_bottleneck,
            adapter_drop=adapter_drop,
            adapter_in_last=adapter_in_last,
            num_frames=num_frames,
            grid_hw=grid_hw,
            n_register_tokens=n_register_tokens,
            ks=ks, kt=kt, fuse=fuse,
        )
        d = self.backbone.backbone.config.hidden_size

        # 1) ray-attention neck option A
        self.neck = RayAttentionNeckOptionA(
            dim=d, num_heads=neck_heads, drop=neck_drop,
            bias_scale=ray_bias_scale, bias_mode=ray_bias_mode
        )

        # 3) heatmap + sigma
        self.hm_head = TinyHeatmapHead(d, grid_hw=grid_hw, num_kp=num_kp, out_hw=heatmap_hw, mid_c=hm_mid_c)
        self.sigma_head = UncertaintyHead(d, num_kp=num_kp, mode=unc_mode, hidden=unc_hidden, drop=neck_drop)

        # 2) THESEUS setup inside model (PnP-like reprojection)
        # Variable
        self.pose = th.SE3(name="pose")

        # Objective
        objective = th.Objective()
        # aux variables are placeholders; we pass tensors through inputs dict at runtime
        cost_fn = th.AutoDiffCostFunction(
            [self.pose],
            reprojection_error,
            num_kp * 2,
            aux_vars=[
                th.Variable(torch.zeros(1, num_kp, 3), name="pts_3d"),
                th.Variable(torch.zeros(1, num_kp, 2), name="pts_2d"),
                th.Variable(torch.zeros(1, num_kp, 2), name="weights"),
                th.Variable(torch.zeros(1, 3, 3), name="K"),
            ],
            name="pnp_cost",
        )
        objective.add(cost_fn)
        optimizer = th.LevenbergMarquardt(objective, max_iterations=lm_max_iters)
        self.theseus_layer = th.TheseusLayer(optimizer)
        self.solve_dtype = torch.float32

    def forward(self, video, cam_K, full_hw=None, crop_xywh=None, track_init=True):
        """
        video: (B,T,C,H_in,W_in) (input frames, possibly crop-resized)
        cam_K: (B,3,3) or (B,T,3,3) intrinsics in FULL image pixel space
        full_hw: (H_full,W_full). If None => assume full == input (H_in,W_in).
        crop_xywh: optional (B,T,4) normalized [0,1] wrt FULL image.

        track_init:
          - if True, pose_t init = pose_{t-1} best solution (tracking)
          - if False, all frames init with identity
        """
        B, T, C, H_in, W_in = video.shape
        assert T == self.num_frames, f"T={T} != num_frames={self.num_frames}"
        if full_hw is None:
            full_hw = (H_in, W_in)

        # backbone tokens
        x = video.view(B * T, C, H_in, W_in)
        tok = self.backbone.forward_tokens(x)  # (BT,N,D)
        BT, N, D = tok.shape

        # infer register tokens if needed
        nreg = self.n_register_tokens
        if nreg is None:
            nreg = N - 1 - self.P
            assert nreg >= 0, f"Cannot infer n_register_tokens: N={N}, P={self.P}"

        tok_patch = tok[:, 1 + nreg: 1 + nreg + self.P, :]  # (BT,P,D)
        tok_patch_btpd = tok_patch.view(B, T, self.P, D)

        # rays per patch center in FULL camera space
        rays_btp3 = build_patch_rays_full(
            B=B, T=T,
            grid_hw=self.grid_hw,
            in_hw=(H_in, W_in),
            full_hw=full_hw,
            cam_K=cam_K,
            crop_xywh_norm=crop_xywh,
            device=video.device,
            dtype=video.dtype,
        )

        # 1) ray neck
        tok_geo = self.neck(tok_patch_btpd, rays_btp3)  # (B,T,P,D)

        # 3) heatmaps + DSNT mu + sigma
        hm_logits = self.hm_head(tok_geo)               # (B,T,K,Hh,Wh)
        hm_prob = _softmax_2d(hm_logits)
        mu_norm = dsnt(hm_prob)                         # (B,T,K,2) in [-1,1] on heatmap grid

        sigma = self.sigma_head(tok_geo)                # (B,T,K,1 or 2) positive
        # weights for reprojection: use inverse sigma (broadcast to 2D if needed)
        if sigma.shape[-1] == 1:
            weights = (1.0 / sigma).expand(-1, -1, -1, 2)  # (B,T,K,2)
        else:
            weights = 1.0 / sigma                           # (B,T,K,2)

        # 4) coords -> FULL pixel space
        Hh, Wh = hm_logits.shape[-2], hm_logits.shape[-1]
        mu_px_hm = norm_to_pixel(mu_norm, (Hh, Wh))  # (B,T,K,2) in heatmap px

        sx = (W_in - 1) / max(Wh - 1, 1)
        sy = (H_in - 1) / max(Hh - 1, 1)
        mu_px_in = torch.stack([mu_px_hm[..., 0] * sx, mu_px_hm[..., 1] * sy], dim=-1)  # input px

        Hf, Wf = full_hw
        if crop_xywh is None:
            # full-frame resized -> scale to full if needed
            if (Hf, Wf) == (H_in, W_in):
                mu_px_full = mu_px_in
            else:
                sx2 = (Wf - 1) / max(W_in - 1, 1)
                sy2 = (Hf - 1) / max(H_in - 1, 1)
                mu_px_full = torch.stack([mu_px_in[..., 0] * sx2, mu_px_in[..., 1] * sy2], dim=-1)
        else:
            crop_xywh_px = crop_norm_to_pixel(crop_xywh, full_hw)  # (B,T,4)
            x0 = crop_xywh_px[..., 0:1]
            y0 = crop_xywh_px[..., 1:2]
            cw = crop_xywh_px[..., 2:3].clamp_min(1.0)
            ch = crop_xywh_px[..., 3:4].clamp_min(1.0)
            sx3 = cw / max(W_in - 1, 1)
            sy3 = ch / max(H_in - 1, 1)
            mu_px_full = torch.stack([x0 + mu_px_in[..., 0] * sx3, y0 + mu_px_in[..., 1] * sy3], dim=-1)

        # 2) THESEUS per-frame solve (tracking init)
        pred_R = []
        pred_t = []
        pred_xi = []   # store se(3) tangent approx for prior outside Theseus (if you want it)

        prev_pose_obj = None
        with torch.autocast(device_type="cuda", enabled=False):
            for t in range(T):
                # pts_3d = self.pts_3d.repeat(B, 1, 1)          # (B,K,3)
                # pts_2d = mu_px_full[:, t, :, :]               # (B,K,2) FULL px
                # w2d = weights[:, t, :, :]                     # (B,K,2

                # if cam_K.dim() == 4:
                #     K_t = cam_K[:, t, :, :]
                # else:
                #     K_t = cam_K

                # inputs = {
                #     "pts_3d": pts_3d,
                #     "pts_2d": pts_2d,
                #     "weights": w2d,
                #     "K": K_t,
                # }

                # if track_init and (prev_pose_obj is not None):
                #     init_pose = prev_pose_obj
                # else:
                #     init_pose = th.SE3(torch.zeros(B, 6, device=video.device, dtype=video.dtype), name="pose")
                pts_3d = self.pts_3d.repeat(B, 1, 1).to(dtype=self.solve_dtype)

                pts_2d = mu_px_full[:, t].to(dtype=self.solve_dtype)
                w2d    = weights[:, t].to(dtype=self.solve_dtype)

                K_t = (cam_K[:, t] if cam_K.dim() == 4 else cam_K).to(dtype=self.solve_dtype)

                if track_init and (prev_pose_obj is not None):
                    init_pose = prev_pose_obj
                else:
                    init_pose = th.SE3(torch.zeros(B, 6, device=video.device, dtype=self.solve_dtype), name="pose")

                inputs = {
                        "pts_3d": pts_3d,
                        "pts_2d": pts_2d,
                        "weights": w2d,
                        "K": K_t,
                    }
                updated_inputs, info = self.theseus_layer.forward(
                    inputs,
                    optimizer_kwargs={
                        "track_best_solution": True,
                        "init_optim_vars": {"pose": init_pose},
                    },
                )

                pose_obj = updated_inputs["pose"]  # th.SE3 (B)
                prev_pose_obj = pose_obj

                # rotation + translation
                R = pose_obj.rotation().matrix()        # (B,3,3)
                tvec = pose_obj.translation().tensor()  # (B,3)

                pred_R.append(R)
                pred_t.append(tvec)

                # best-effort tangent (for prior outside Theseus)
                if hasattr(pose_obj, "log_map"):
                    xi = pose_obj.log_map()  # (B,6)
                elif hasattr(pose_obj, "to_matrix"):
                    # fallback: keep zeros (won't be used unless weight>0)
                    xi = torch.zeros(B, 6, device=video.device, dtype=video.dtype)
                else:
                    xi = torch.zeros(B, 6, device=video.device, dtype=video.dtype)
                pred_xi.append(xi)

        pred_R = torch.stack(pred_R, dim=1)  # (B,T,3,3)
        pred_t = torch.stack(pred_t, dim=1)  # (B,T,3)
        pred_xi = torch.stack(pred_xi, dim=1)  # (B,T,6)

        prior_loss = temporal_pose_prior_loss(pred_xi, weight=self.pose_prior_weight, order=self.pose_prior_order)

        return {
            "heatmaps": hm_logits,
            "hm_prob": hm_prob,
            "mu_norm": mu_norm,           # DSNT coords [-1,1] on heatmap grid
            "mu_px_full": mu_px_full,     # (B,T,K,2) FULL camera pixel coords
            "sigma": sigma,               # (B,T,K,1 or 2)
            "weights": weights,           # (B,T,K,2)
            "pred_R": pred_R,             # (B,T,3,3)
            "pred_t": pred_t,             # (B,T,3)
            "temporal_pose_prior_loss": prior_loss,
            # optional debugging
            "tokens_geo": tok_geo,
            "rays": rays_btp3,
        }


# ---------------- quick sanity test ----------------
def _smoke_test():
    if th is None:
        print("Theseus not installed; skipping smoke test.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T, C, H, W = 2, 8, 3, 512, 512
    full_hw = (720, 1280)
    K = torch.tensor([[900.0, 0.0, 640.0],
                      [0.0, 900.0, 360.0],
                      [0.0, 0.0, 1.0]], device=device).unsqueeze(0).repeat(B, 1, 1)

    crop_xywh = torch.tensor([0.25, 0.2, 0.5, 0.6], device=device).view(1, 1, 4).repeat(B, T, 1)

    pts_3d = torch.randn(1, 8, 3, device=device)  # replace with your SPADES model points

    model = GeoSpaceFormer(
        model_pts_3d=pts_3d,
        num_kp=8,
        num_frames=T,
        hf_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        grid_hw=(32, 32),
        heatmap_hw=(256, 256),
        adapter_in_last=4,
        unc_mode="diag",
        pose_prior_weight=0.1,
        pose_prior_order=1,
    ).to(device).eval()

    video = torch.randn(B, T, C, H, W, device=device)

    with torch.no_grad():
        out = model(video, cam_K=K, full_hw=full_hw, crop_xywh=crop_xywh, track_init=True)

    print("heatmaps:", out["heatmaps"].shape)
    print("mu_px_full:", out["mu_px_full"].shape)
    print("pred_R:", out["pred_R"].shape)
    print("pred_t:", out["pred_t"].shape)
    print("prior:", float(out["temporal_pose_prior_loss"]))

if __name__ == "__main__":
    _smoke_test()