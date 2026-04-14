# model_geospaceformer_rcsa.py
# Clean + corrected implementation
#
# Fixes applied vs your draft:
# 1) ✅ Correct sparse attention semantics: updates are DISJOINT.
#    - For each bin m: queries = patches in bin m
#    - keys/values = patches in (bin m + neighbor bins)
#    - scatter only to query positions → no double counting.
#
# 2) ✅ Fast grouping without per-bin argsort:
#    - per-frame, per-batch: sort once by bin id → contiguous segments per bin
#    - compute counts + offsets → gather patches by bin cheaply
#
# 3) ✅ Robust bin assignment under AMP:
#    - rays + centers quantization computed in fp32 (noisy fp16 argmax avoided)
#
# 4) ✅ Removed dead/unused helpers (TemporalPatchAttention, _pad_to_max).
#
# 5) ✅ Added small safety / sanity checks + comments.
#
# Notes:
# - This keeps a small loop over (T * B * M) to assemble variable-length lists,
#   but avoids the catastrophic per-bin argsort and maintains correct semantics.
# - If you later want *fully* fused performance, we can replace the per-bin list
#   assembly with a CUDA kernel or a precomputed index map, but this is already
#   a strong “fast + correct” PyTorch baseline.

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DINOv3ViTModel


# ---------------- DSNT utilities ----------------
def _softmax_2d(hm: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # hm: (B,T,K,H,W)
    B, T, K, H, W = hm.shape
    x = hm.contiguous().view(B * T * K, H * W)
    x = F.softmax(x, dim=-1)
    return x.view(B, T, K, H, W).clamp_min(eps)


def dsnt(hm_prob: torch.Tensor) -> torch.Tensor:
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


def norm_to_pixel(coords_norm: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    """coords_norm (...,2) in [-1,1] -> pixel coords in [0..W-1, 0..H-1]."""
    H, W = hw
    x = (coords_norm[..., 0] + 1.0) * 0.5 * (W - 1)
    y = (coords_norm[..., 1] + 1.0) * 0.5 * (H - 1)
    return torch.stack([x, y], dim=-1)


def crop_norm_to_pixel(crop_xywh_norm: torch.Tensor, full_hw: Tuple[int, int]) -> torch.Tensor:
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
def make_patch_centers_in_input(
    grid_hw: Tuple[int, int], in_hw: Tuple[int, int], device, dtype
) -> torch.Tensor:
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


def _normalize(v: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def unproject_to_rays(uv_px_full: torch.Tensor, K_px: torch.Tensor) -> torch.Tensor:
    """
    uv_px_full: (B,T,P,2)
    K_px: (B,3,3) or (B,T,3,3)
    returns: (B,T,P,3) unit rays
    """
    B, T, P, _ = uv_px_full.shape

    if K_px.dim() == 3:
        K_px = K_px.unsqueeze(1).expand(B, T, 3, 3)
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
    v = _normalize(v)
    return v


def build_patch_rays_full(
    B: int,
    T: int,
    grid_hw: Tuple[int, int],
    in_hw: Tuple[int, int],
    full_hw: Tuple[int, int],
    cam_K: torch.Tensor,
    crop_xywh_norm: Optional[torch.Tensor],
    device,
) -> torch.Tensor:
    """
    Returns rays per patch center in FULL camera space.
    Important: compute in fp32 for stable bin assignment under AMP.
    """
    H_in, W_in = in_hw
    Hf, Wf = full_hw
    P = grid_hw[0] * grid_hw[1]

    # centers in INPUT pixel coords (x,y), fp32
    centers_in = make_patch_centers_in_input(grid_hw, in_hw, device=device, dtype=torch.float32)  # (P,2)
    centers_in = centers_in.view(1, 1, P, 2).expand(B, T, P, 2)

    if crop_xywh_norm is None:
        sx = (Wf - 1) / max(W_in - 1, 1)
        sy = (Hf - 1) / max(H_in - 1, 1)
        centers_full = torch.stack([centers_in[..., 0] * sx, centers_in[..., 1] * sy], dim=-1)
    else:
        crop_xywh_px = crop_norm_to_pixel(crop_xywh_norm.float(), full_hw)  # fp32
        x0 = crop_xywh_px[..., 0:1]
        y0 = crop_xywh_px[..., 1:2]
        cw = crop_xywh_px[..., 2:3].clamp_min(1.0)
        ch = crop_xywh_px[..., 3:4].clamp_min(1.0)
        sx = cw / max(W_in - 1, 1)
        sy = ch / max(H_in - 1, 1)
        centers_full = torch.stack([x0 + centers_in[..., 0] * sx, y0 + centers_in[..., 1] * sy], dim=-1)

    rays = unproject_to_rays(centers_full, cam_K.float())  # fp32
    return rays  # fp32 on purpose


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

        y = y.view(B, T, self.Hp, self.Wp, C).permute(0, 4, 1, 2, 3).contiguous()

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
                    x = inputs[0]
                    if x.dim() != 3:
                        return inputs  # unexpected signature, do nothing
                    return (adapter_i(x),) + inputs[1:]
                return hook

            attn.register_forward_pre_hook(make_mhsa_hook(adapter), with_kwargs=False)

    def forward_tokens(self, x):
        out = self.backbone(pixel_values=x, return_dict=True)
        return out.last_hidden_state  # (BT,N,D)


# ============================================================
# A) Ray-Conditioned Sparse Attention (RCSA)
# ============================================================

def _fibonacci_sphere(M: int, device, dtype):
    i = torch.arange(M, device=device, dtype=dtype)
    phi = math.pi * (3.0 - math.sqrt(5.0))
    y = 1.0 - 2.0 * (i + 0.5) / M
    r = torch.sqrt((1.0 - y * y).clamp_min(0.0))
    theta = phi * i
    x = torch.cos(theta) * r
    z = torch.sin(theta) * r
    return _normalize(torch.stack([x, y, z], dim=-1))  # (M,3)


@torch.no_grad()
def assign_ray_bins_fp32(rays_btp3_fp32: torch.Tensor, centers_m3_fp32: torch.Tensor) -> torch.Tensor:
    # rays unit, centers unit (both fp32)
    sim = torch.einsum("btpc,mc->btpm", rays_btp3_fp32, centers_m3_fp32)  # (B,T,P,M)
    return sim.argmax(dim=-1)  # (B,T,P)


class RCSASpatialAttention(nn.Module):
    """
    Correct + fast(ish) RCSA:

    For each frame:
      1) assign each patch to a ray bin (fp32)
      2) sort patches by bin id ONCE  (per B)
      3) for each bin m:
           queries = patches in bin m (disjoint)
           keys    = union of patches in bin m + neighbor bins
           do attention Q(m) x K(neighbors)
           scatter only to Q positions

    This avoids:
      - per-bin argsort
      - double-count accumulation due to neighbor unions
      - fp16 quantization instability
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        drop: float = 0.1,
        M: int = 64,
        neighbor_bins: int = 0,
        # runtime safety:
        approx_topk_kv: Optional[int] = 64,   # cap KV set size per (bin m) union
        max_q: Optional[int] = None,          # optional cap on queries per bin (rarely needed)
        bias_scale: float = 0.0,              # optional ray similarity bias inside the selected KV set
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.h = num_heads
        self.dh = dim // num_heads
        self.M = int(M)
        self.neighbor_bins = int(neighbor_bins)
        self.approx_topk_kv = approx_topk_kv
        self.max_q = max_q
        self.bias_scale = float(bias_scale)

        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(drop)
        self.proj_drop = nn.Dropout(drop)

        self.register_buffer("_centers_fp32", torch.empty(0), persistent=False)
        self.register_buffer("_nbrs", torch.empty(0, dtype=torch.long), persistent=False)

    def _ensure_centers_and_neighbors(self, device):
        if self._centers_fp32.numel() == 0 or self._centers_fp32.device != device:
            centers = _fibonacci_sphere(self.M, device=device, dtype=torch.float32)  # fp32
            if self.neighbor_bins > 0:
                sim = centers @ centers.t()  # (M,M) fp32
                k = min(self.neighbor_bins + 1, self.M)
                nbrs = sim.topk(k=k, dim=-1).indices  # (M,k) includes self
            else:
                nbrs = torch.arange(self.M, device=device).view(self.M, 1)
            self._centers_fp32 = centers
            self._nbrs = nbrs

    @staticmethod
    def _counts_from_sorted_bins(sorted_bins: torch.Tensor, M: int) -> torch.Tensor:
        """
        sorted_bins: (P,) int64 in [0..M-1], sorted ascending
        returns counts: (M,) int64
        """
        return torch.bincount(sorted_bins, minlength=M)

    def forward(self, x_btpd: torch.Tensor, rays_btp3_fp32: torch.Tensor) -> torch.Tensor:
        """
        x_btpd: (B,T,P,D) in model dtype (fp16/bf16/fp32)
        rays_btp3_fp32: (B,T,P,3) fp32
        """
        B, T, P, D = x_btpd.shape
        device = x_btpd.device
        self._ensure_centers_and_neighbors(device)

        # bin assignment in fp32 (no grad)
        with torch.no_grad():
            bin_idx = assign_ray_bins_fp32(rays_btp3_fp32, self._centers_fp32)  # (B,T,P), int64

        # qkv computed in model dtype
        qkv = (
            self.qkv(x_btpd)
            .view(B, T, P, 3, self.h, self.dh)
            .permute(3, 0, 1, 4, 2, 5)
            .contiguous()
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B,T,h,P,dh)

        out = x_btpd.new_zeros(B, T, P, D)

        # Process per frame
        for t in range(T):
            # per batch element do a single sort by bin id
            for b in range(B):
                bidx = bin_idx[b, t]  # (P,)

                perm = torch.argsort(bidx)              # (P,)
                bidx_sorted = bidx[perm]                # (P,)
                counts = self._counts_from_sorted_bins(bidx_sorted, self.M)  # (M,)
                offsets = torch.cumsum(counts, dim=0) - counts              # (M,)

                # gather q/k/v in sorted order (still P-length)
                q_sorted = q[b, t, :, perm, :]  # (h,P,dh)
                k_sorted = k[b, t, :, perm, :]
                v_sorted = v[b, t, :, perm, :]

                # also rays in sorted order (fp32) for optional bias / topk scoring
                rays_sorted = rays_btp3_fp32[b, t, perm, :]  # (P,3) fp32

                # helper: slice [start:end)
                # each bin segment is contiguous inside sorted arrays
                for m in range(self.M):
                    q_start = int(offsets[m].item())
                    q_len = int(counts[m].item())
                    if q_len == 0:
                        continue

                    if self.max_q is not None and q_len > self.max_q:
                        q_len = int(self.max_q)

                    q_end = q_start + q_len

                    # KV union of neighbor bins
                    nbrs = self._nbrs[m]  # (kn,)
                    # collect KV indices ranges in sorted space
                    kv_ranges = []
                    kv_total = 0
                    for nb in nbrs.tolist():
                        nb = int(nb)
                        nb_start = int(offsets[nb].item())
                        nb_len = int(counts[nb].item())
                        if nb_len <= 0:
                            continue
                        kv_ranges.append((nb_start, nb_start + nb_len))
                        kv_total += nb_len
                    if kv_total == 0:
                        continue

                    # If KV too large, approximate by selecting top-k keys by ray similarity to center(m)
                    if self.approx_topk_kv is not None and kv_total > int(self.approx_topk_kv):
                        Kk = int(self.approx_topk_kv)

                        # gather rays from union ranges
                        # (kv_total,3) fp32
                        rays_kv = torch.cat([rays_sorted[s:e] for (s, e) in kv_ranges], dim=0)
                        center = self._centers_fp32[m].view(1, 3)  # fp32
                        score = (rays_kv * center).sum(dim=-1)     # (kv_total,)
                        topk_local = torch.topk(score, k=Kk, dim=0).indices  # (Kk,)

                        # Need KV tensors corresponding to these indices:
                        k_kv_all = torch.cat([k_sorted[:, s:e, :] for (s, e) in kv_ranges], dim=1)  # (h,kv_total,dh)
                        v_kv_all = torch.cat([v_sorted[:, s:e, :] for (s, e) in kv_ranges], dim=1)
                        k_kv = k_kv_all[:, topk_local, :]  # (h,Kk,dh)
                        v_kv = v_kv_all[:, topk_local, :]
                        rays_kv_sel = rays_kv[topk_local]  # (Kk,3)
                    else:
                        k_kv = torch.cat([k_sorted[:, s:e, :] for (s, e) in kv_ranges], dim=1)  # (h,kv_total,dh)
                        v_kv = torch.cat([v_sorted[:, s:e, :] for (s, e) in kv_ranges], dim=1)
                        rays_kv_sel = torch.cat([rays_sorted[s:e] for (s, e) in kv_ranges], dim=0)  # (kv_total,3)

                    # Queries: bin m only (disjoint update)
                    q_q = q_sorted[:, q_start:q_end, :]  # (h,q_len,dh)

                    # Attention: (h,q,kv)
                    attn = torch.matmul(q_q, k_kv.transpose(-2, -1)) * (1.0 / math.sqrt(self.dh))

                    if self.bias_scale != 0.0:
                        # ray bias: dot(ray_q, ray_k)
                        rays_q = rays_sorted[q_start:q_end, :]  # (q,3) fp32
                        sim = (rays_q @ rays_kv_sel.t()).clamp(-1.0, 1.0)  # (q,kv) fp32
                        # cast to attn dtype
                        attn = attn + self.bias_scale * sim.to(attn.dtype).unsqueeze(0)

                    attn = F.softmax(attn, dim=-1)
                    attn = self.attn_drop(attn)

                    y = torch.matmul(attn, v_kv)  # (h,q,dh)

                    # Scatter to original (unsorted) positions:
                    # q positions in sorted space correspond to perm[q_start:q_end] in original P
                    out_bthp = out[b, t].view(P, self.h, self.dh)  # (P,h,dh)
                    orig_idx = perm[q_start:q_end]                 # (q,)
                    # add in place (no duplicates because orig_idx disjoint across m)
                    out_bthp[orig_idx] += y.permute(1, 0, 2).contiguous()
                    out[b, t] = out_bthp.view(P, D)

        out = self.proj_drop(self.proj(out))
        return out


class RCSANeck(nn.Module):
    """
    RCSA neck:
      - sparse spatial attention (correct disjoint updates)
      - optional temporal attention per patch index (lightweight)
      - MLP
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        drop: float = 0.1,
        mlp_ratio: float = 4.0,
        M: int = 64,
        neighbor_bins: int = 0,
        use_temporal: bool = True,
        approx_topk_kv: Optional[int] = 64,
        max_q: Optional[int] = None,
        ray_bias_scale: float = 0.0,
    ):
        super().__init__()
        self.use_temporal = bool(use_temporal)

        self.ln1 = nn.LayerNorm(dim)
        self.spatial = RCSASpatialAttention(
            dim=dim,
            num_heads=num_heads,
            drop=drop,
            M=M,
            neighbor_bins=neighbor_bins,
            approx_topk_kv=approx_topk_kv,
            max_q=max_q,
            bias_scale=ray_bias_scale,
        )

        self.ln2 = nn.LayerNorm(dim)
        self.temporal = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=drop, batch_first=True)

        self.ln3 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

        self.g_s = nn.Parameter(torch.tensor(1e-3, dtype=torch.float32))
        self.g_t = nn.Parameter(torch.tensor(1e-3, dtype=torch.float32))
        self.g_m = nn.Parameter(torch.tensor(1e-3, dtype=torch.float32))

    def forward(self, x_btpd: torch.Tensor, rays_btp3_fp32: torch.Tensor) -> torch.Tensor:
        x = x_btpd
        x = x + self.g_s * self.spatial(self.ln1(x), rays_btp3_fp32)

        if self.use_temporal:
            B, T, P, D = x.shape
            xp = self.ln2(x).permute(0, 2, 1, 3).contiguous().view(B * P, T, D)
            y, _ = self.temporal(xp, xp, xp, need_weights=False)
            y = y.view(B, P, T, D).permute(0, 2, 1, 3).contiguous()
            x = x + self.g_t * y

        x = x + self.g_m * self.mlp(self.ln3(x))
        return x


# ============================================================
# B) Keypoint Pose Tokens (cross-attention with ray conditioning)
# ============================================================
class RayConditionedKPTokens(nn.Module):
    """
    Learnable K tokens z_k. For each frame, cross-attend z_k -> patch tokens.
    Ray conditioning: add bias proportional to cos(ray_patch, ray_token_k) where ray_token_k is learnable direction.
    Returns:
      - kp_tokens: (B,T,K,D)
      - attn_mean (optional): (B,T,K,P)  (mean over heads)
    """
    def __init__(
        self,
        dim: int,
        num_kp: int = 8,
        num_heads: int = 8,
        drop: float = 0.1,
        ray_bias_scale: float = 2.0
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.K = int(num_kp)
        self.h = int(num_heads)
        self.dh = dim // num_heads
        self.ray_bias_scale = float(ray_bias_scale)

        self.z = nn.Parameter(torch.randn(self.K, dim) * 0.02)
        self.ray_dir = nn.Parameter(_normalize(torch.randn(self.K, 3)))  # normalized at init

        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)

        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(drop)
        self.proj_drop = nn.Dropout(drop)

        self.ln_z = nn.LayerNorm(dim)
        self.ln_x = nn.LayerNorm(dim)

    def forward(
        self,
        patch_tokens_btpd: torch.Tensor,
        rays_btp3_fp32: torch.Tensor,
    ):
        """
        patch_tokens_btpd: (B,T,P,D)
        rays_btp3_fp32:    (B,T,P,3) fp32
        """
        B, T, P, D = patch_tokens_btpd.shape

        # (B,T,K,D)
        z = self.ln_z(self.z).view(1, 1, self.K, D).expand(B, T, self.K, D)
        x = self.ln_x(patch_tokens_btpd)

        # q: (B,T,h,K,dh), k/v: (B,T,h,P,dh)
        q = self.q(z).view(B, T, self.K, self.h, self.dh).permute(0, 1, 3, 2, 4).contiguous()
        k = self.k(x).view(B, T, P, self.h, self.dh).permute(0, 1, 3, 2, 4).contiguous()
        v = self.v(x).view(B, T, P, self.h, self.dh).permute(0, 1, 3, 2, 4).contiguous()

        scale = 1.0 / math.sqrt(self.dh)

        # attn: (B,T,h,K,P)
        attn = (q @ k.transpose(-1, -2)) * scale

        # ray bias: sim (B,T,K,P) in fp32 -> cast to attn dtype
        sim = torch.einsum(
            "kc,btpc->btkp",
            _normalize(self.ray_dir),  # (K,3) float32-ish
            rays_btp3_fp32             # (B,T,P,3) fp32
        ).clamp(-1.0, 1.0)  # (B,T,K,P) fp32

        attn = attn + self.ray_bias_scale * sim.to(attn.dtype).unsqueeze(2)  # (B,T,1,K,P) broadcast over heads

        # softmax + dropout
        attn = attn.softmax(dim=-1)          # (B,T,h,K,P)
        attn = self.attn_drop(attn)          # (B,T,h,K,P)

        # output tokens
        y = attn @ v                          # (B,T,h,K,dh)
        y = y.permute(0, 1, 3, 2, 4).contiguous().view(B, T, self.K, D)
        y = self.proj_drop(self.proj(y))      # (B,T,K,D)

        
        # mean over heads -> (B,T,K,P)
        attn_mean = attn.mean(dim=2)
        return y, attn_mean

# ---------------- Heatmap + uncertainty heads ----------------
class TinyHeatmapHead(nn.Module):
    def __init__(self, d: int, grid_hw=(14, 14), num_kp: int = 8, out_hw=(224, 224), mid_c: int = 128):
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

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
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


class UncertaintyHeadFromKPTokens(nn.Module):
    def __init__(self, d: int, num_kp: int = 8, mode: str = "diag", hidden: int = 256, drop: float = 0.1, sigma_min: float = 1e-4, sigma_init: float = 30.0):
        super().__init__()
        assert mode in ["scalar", "diag"]
        self.out_dim = 1 if mode == "scalar" else 2
        self.sigma_min = float(sigma_min)
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, self.out_dim),
        )

        with torch.no_grad():
            b = math.log(math.exp(float(sigma_init)) - 1.0)
            self.net[-1].bias.fill_(b)
            
    def forward(self, kp_tokens_btkd: torch.Tensor) -> torch.Tensor:
        raw = self.net(kp_tokens_btkd)
        return F.softplus(raw) + self.sigma_min


# ---------------- Optional Pose Head (no Theseus) ----------------
def rot6d_to_matrix(x6: torch.Tensor) -> torch.Tensor:
    a1 = x6[..., 0:3]
    a2 = x6[..., 3:6]
    b1 = _normalize(a1)
    b2 = _normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


class PoseHeadFromKPTokens(nn.Module):
    def __init__(self, d: int, hidden: int = 512, drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.rot = nn.Linear(hidden, 6)
        self.t = nn.Linear(hidden, 3)

    def forward(self, kp_tokens_btkd: torch.Tensor):
        frame = kp_tokens_btkd.mean(dim=2)  # (B,T,D)
        h = self.net(frame)
        rot6 = self.rot(h)
        t = self.t(h)
        R = rot6d_to_matrix(rot6.view(-1, 6)).view(kp_tokens_btkd.shape[0], kp_tokens_btkd.shape[1], 3, 3)
        return R, t


# ---------------- Full model ----------------
class GeoSpaceFormer(nn.Module):
    def __init__(
        self,
        model_pts_3d: Optional[torch.Tensor],
        num_kp: int = 8,
        num_frames: int = 16,
        hf_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        grid_hw: Tuple[int, int] = (14, 14),
        heatmap_hw: Tuple[int, int] = (224, 224),
        hm_mid_c: int = 128,
        # adapters
        adapter_bottleneck: int = 64,
        adapter_drop: float = 0.1,
        adapter_in_last=None,
        n_register_tokens: int | None = None,
        ks: int = 3,
        kt: int = 3,
        fuse: str = "sum",
        # RCSA
        neck_heads: int = 8,
        neck_drop: float = 0.1,
        rcsa_bins: int = 64,
        rcsa_neighbor_bins: int = 6,
        rcsa_use_temporal: bool = True,
        rcsa_topk_kv: Optional[int] = 64,
        rcsa_max_q: Optional[int] = None,
        rcsa_ray_bias_scale: float = 0.0,
        # keypoint tokens
        kp_token_heads: int = 8,
        kp_token_drop: float = 0.1,
        kp_ray_bias_scale: float = 2.0,
        # uncertainty
        unc_mode: str = "diag",
        unc_hidden: int = 256,
        # pose head (optional)
        enable_pose_head: bool = False,
        pose_hidden: int = 512,
    ):
        super().__init__()

        self.num_kp = int(num_kp)
        self.num_frames = int(num_frames)
        self.grid_hw = tuple(grid_hw)
        self.P = self.grid_hw[0] * self.grid_hw[1]
        self.heatmap_hw = tuple(heatmap_hw)
        self.n_register_tokens = n_register_tokens
        self.enable_pose_head = bool(enable_pose_head)

        if model_pts_3d is not None:
            if model_pts_3d.dim() == 2:
                model_pts_3d = model_pts_3d.unsqueeze(0)
            self.register_buffer("pts_3d", model_pts_3d, persistent=True)
        else:
            self.register_buffer("pts_3d", torch.empty(0), persistent=False)

        # 0) backbone + pre-MHSA adapters
        self.backbone = DINOv3backbone(
            hf_name=hf_name,
            adapter_bottleneck=adapter_bottleneck,
            adapter_drop=adapter_drop,
            adapter_in_last=adapter_in_last,
            num_frames=num_frames,
            grid_hw=grid_hw,
            n_register_tokens=n_register_tokens,
            ks=ks,
            kt=kt,
            fuse=fuse,
        )
        d = self.backbone.backbone.config.hidden_size

        # A) RCSA neck
        self.neck = RCSANeck(
            dim=d,
            num_heads=neck_heads,
            drop=neck_drop,
            M=rcsa_bins,
            neighbor_bins=rcsa_neighbor_bins,
            use_temporal=rcsa_use_temporal,
            approx_topk_kv=rcsa_topk_kv,
            max_q=rcsa_max_q,
            ray_bias_scale=rcsa_ray_bias_scale,
        )

        # B) keypoint tokens
        self.kp_tokens = RayConditionedKPTokens(
            dim=d,
            num_kp=num_kp,
            num_heads=kp_token_heads,
            drop=kp_token_drop,
            ray_bias_scale=kp_ray_bias_scale,
        )

        # heatmaps from patch tokens
        self.hm_head = TinyHeatmapHead(d, grid_hw=grid_hw, num_kp=num_kp, out_hw=heatmap_hw, mid_c=hm_mid_c)

        # sigma from kp tokens
        self.sigma_head = UncertaintyHeadFromKPTokens(d, num_kp=num_kp, mode=unc_mode, hidden=unc_hidden, drop=kp_token_drop)

        # optional pose head
        self.pose_head = PoseHeadFromKPTokens(d, hidden=pose_hidden, drop=kp_token_drop) if self.enable_pose_head else None

    def forward(
        self,
        video: torch.Tensor,
        cam_K: torch.Tensor,
        full_hw: Optional[Tuple[int, int]] = None,
        crop_xywh: Optional[torch.Tensor] = None,
    ):
        """
        video: (B,T,C,H_in,W_in)
        cam_K: (B,3,3) or (B,T,3,3) intrinsics in FULL image pixel space
        full_hw: (H_full,W_full). If None => assume full == input
        crop_xywh: (B,T,4) normalized [0,1] wrt FULL image
        """
        B, T, C, H_in, W_in = video.shape
        assert T == self.num_frames, f"T={T} != num_frames={self.num_frames}"
        if full_hw is None:
            full_hw = (H_in, W_in)

        # backbone tokens
        x = video.view(B * T, C, H_in, W_in)
        tok = self.backbone.forward_tokens(x)  # (BT,N,D)
        BT, N, D = tok.shape

        nreg = self.n_register_tokens
        if nreg is None:
            nreg = N - 1 - self.P
            assert nreg >= 0, f"Cannot infer n_register_tokens: N={N}, P={self.P}"

        tok_patch = tok[:, 1 + nreg: 1 + nreg + self.P, :]  # (BT,P,D)
        tok_patch_btpd = tok_patch.view(B, T, self.P, D)

        # rays (fp32) per patch center in FULL camera space
        rays_btp3_fp32 = build_patch_rays_full(
            B=B,
            T=T,
            grid_hw=self.grid_hw,
            in_hw=(H_in, W_in),
            full_hw=full_hw,
            cam_K=cam_K,
            crop_xywh_norm=crop_xywh,
            device=video.device,
        )

        # A) RCSA neck
        tok_geo = self.neck(tok_patch_btpd, rays_btp3_fp32)

        # B) keypoint tokens + attention map (kp -> patch)

        kp_out, attn_kp2patch = self.kp_tokens(tok_geo, rays_btp3_fp32)  # (B,T,K,D), (B,T,K,P)
        # else:
        #     kp_out = self.kp_tokens(tok_geo, rays_btp3_fp32, return_attn=False)  # (B,T,K,D)
        #     attn_kp2patch = None

        # heatmaps + DSNT
        hm_logits = self.hm_head(tok_geo)               # (B,T,K,Hh,Wh)
        hm_prob = _softmax_2d(hm_logits)
        mu_norm = dsnt(hm_prob)                         # (B,T,K,2)

        # sigma
        sigma = self.sigma_head(kp_out)
        weights = (1.0 / sigma).expand(-1, -1, -1, 2) if sigma.shape[-1] == 1 else (1.0 / sigma)

        # coords -> FULL pixel
        Hh, Wh = hm_logits.shape[-2], hm_logits.shape[-1]
        mu_px_hm = norm_to_pixel(mu_norm, (Hh, Wh))  # (B,T,K,2)

        sx = (W_in - 1) / max(Wh - 1, 1)
        sy = (H_in - 1) / max(Hh - 1, 1)
        mu_px_in = torch.stack([mu_px_hm[..., 0] * sx, mu_px_hm[..., 1] * sy], dim=-1)

        Hf, Wf = full_hw
        if crop_xywh is None:
            if (Hf, Wf) == (H_in, W_in):
                mu_px_full = mu_px_in
            else:
                sx2 = (Wf - 1) / max(W_in - 1, 1)
                sy2 = (Hf - 1) / max(H_in - 1, 1)
                mu_px_full = torch.stack([mu_px_in[..., 0] * sx2, mu_px_in[..., 1] * sy2], dim=-1)
        else:
            crop_xywh_px = crop_norm_to_pixel(crop_xywh, full_hw)
            x0 = crop_xywh_px[..., 0:1]
            y0 = crop_xywh_px[..., 1:2]
            cw = crop_xywh_px[..., 2:3].clamp_min(1.0)
            ch = crop_xywh_px[..., 3:4].clamp_min(1.0)
            sx3 = cw / max(W_in - 1, 1)
            sy3 = ch / max(H_in - 1, 1)
            mu_px_full = torch.stack([x0 + mu_px_in[..., 0] * sx3, y0 + mu_px_in[..., 1] * sy3], dim=-1)

        out = {
            "heatmaps": hm_logits,
            "hm_prob": hm_prob,
            "mu_norm": mu_norm,
            "mu_px_full": mu_px_full,
            "sigma": sigma,
            "weights": weights,
            "tokens_geo": tok_geo,
            "kp_tokens": kp_out,
            "rays": rays_btp3_fp32,  # fp32,
            "attn_kp2patch": attn_kp2patch,

        }

        if self.pose_head is not None:
            pred_R, pred_t = self.pose_head(kp_out)
            out["pred_R"] = pred_R
            out["pred_t"] = pred_t

        return out


# ---------------- quick sanity test ----------------
def _smoke_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T, C, H, W = 1, 4, 3, 256, 256
    full_hw = (720, 1280)

    K = torch.tensor([[900.0, 0.0, 640.0],
                      [0.0, 900.0, 360.0],
                      [0.0, 0.0, 1.0]], device=device).unsqueeze(0).repeat(B, 1, 1)

    crop_xywh = torch.tensor([0.25, 0.2, 0.5, 0.6], device=device).view(1, 1, 4).repeat(B, T, 1)

    pts_3d = torch.randn(1, 8, 3, device=device)

    model = GeoSpaceFormer(
        model_pts_3d=pts_3d,
        num_kp=8,
        num_frames=T,
        hf_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        grid_hw=(16, 16),
        heatmap_hw=(128, 128),
        adapter_in_last=2,
        unc_mode="diag",
        rcsa_bins=64,
        rcsa_neighbor_bins=6,
        rcsa_use_temporal=True,
        rcsa_topk_kv=64,
        enable_pose_head=True,
    ).to(device).eval()

    video = torch.randn(B, T, C, H, W, device=device)

    with torch.no_grad():
        out = model(video, cam_K=K, full_hw=full_hw, crop_xywh=crop_xywh)

    print("heatmaps:", out["heatmaps"].shape)
    print("mu_px_full:", out["mu_px_full"].shape)
    print("kp_tokens:", out["kp_tokens"].shape)
    print("kp_tokens:", out["kp_tokens"].shape)
    print("pred_R:", out["pred_R"].shape)
    print("pred_t:", out["pred_t"].shape)
    print("rays dtype:", out["rays"].dtype)


if __name__ == "__main__":
    _smoke_test()