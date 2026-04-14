# model.py — DINOv3 backbone + temporal heads, NO temporal stride inside the model
import math
import cv2
import numpy as np
from typing import Optional, Tuple, Literal, Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from vision_transformer import vit_large, vit_small  # Meta's package
from torchvision.ops import DeformConv2d


# ======== CONFIG: path to your compact, strictly-loaded safetensors ========
BACKBONE_CKPT = "./dinov3_vits16_meta_strict.safetensors"

# -----------------------------------------------------------------------------
# Dual Path injection
# -----------------------------------------------------------------------------

class DualPathAdapter(nn.Module):
    """
    Dual-path adapter on patch tokens: temporal DW-conv1D + feature MLP, both bottlenecked.
    Input/Output: patches (B, T, P, D), residual inside.
    """
    def __init__(self, d_model=384, r=96, kT=3, dropout=0.05) -> None:
        super().__init__()

        # temporal path (along time per patch)
        self.t_down = nn.Linear(d_model, r, bias=False)
        self.t_dw = nn.Conv1d(r, r, kernel_size=kT, padding=kT//2, groups=r, bias=False)
        self.t_up = nn.Linear(r, d_model, bias=False)

        # feature path (per frame per patch)
        self.f_down = nn.Linear(d_model, r, bias=False)
        self.f_mlp = nn.Sequential( nn.GELU(),
                            nn.Linear(r, r, bias=False), 
                            )
        self.f_up = nn.Linear(r, d_model, bias=False)

        self.drop = nn.Dropout(dropout)

        # init so it starts as near-identity
        nn.init.kaiming_uniform_(self.t_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.t_up.weight)
        nn.init.kaiming_uniform_(self.f_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.f_up.weight)

    def forward(self, patches):
        B, T, P, D = patches.shape
        x = patches

        #temporal path
        t = self.t_down(x)               # (B,T,P,r)
        t = t.permute(0, 2, 3, 1).contiguous().view(B*P, -1, T)  # (B*P,r,T)
        t = self.t_dw(t)                 # (B*P,r,T)
        t = t.view(B, P, -1, T).permute(0, 3, 1, 2)              # (B,T,P,r)
        t = self.t_up(t)                 # (B,T,P,D)

        #feature path
        f = self.f_down(x)       # (B,T,P,r)
        f = self.f_mlp(f)        # (B,T,P,r)
        f = self.f_up(f)       # (B,T,P,D)

        return x + self.drop(t + f)

# -----------------------------------------------------------------------------
# LoRA utils
# -----------------------------------------------------------------------------
# model.py

# --- model.py ---

class LoRALinear(nn.Module):
    """
    Wrap a frozen Linear W with LoRA adapters:
      y = x @ W^T + scale * ( x @ A^T @ B^T )
    """
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, lora_dropout: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.in_features  = base.in_features
        self.out_features = base.out_features
        self.base = base

        # freeze base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.r = int(r)
        self.scaling = float(alpha) / float(max(1, r))
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # ensure adapters match base device/dtype
        dev = base.weight.device
        dt  = base.weight.dtype

        # name these modules with "lora_" so your optimizer grouping picks them up
        self.lora_A = nn.Linear(self.in_features, self.r, bias=False, device=dev, dtype=dt)
        self.lora_B = nn.Linear(self.r, self.out_features, bias=False, device=dev, dtype=dt)

        # init: small A, zeros B
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        l = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return y + l

def _replace_linear_with_lora(module: nn.Module,
                              name_filter: Iterable[str],
                              r: int, alpha: int, lora_dropout: float,
                              _qualname: str = "") -> int:
    n_wrapped = 0
    for child_name, child in list(module.named_children()):
        qn = f"{_qualname}.{child_name}" if _qualname else child_name

        # Recurse first
        n_wrapped += _replace_linear_with_lora(child, name_filter, r, alpha, lora_dropout, _qualname=qn)

        # Decide if we wrap this child itself
        if isinstance(child, nn.Linear) and any(tag in qn for tag in name_filter):
            setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, lora_dropout=lora_dropout))
            n_wrapped += 1
    return n_wrapped

def apply_vit_lora(backbone: nn.Module,
                   target: str = "qkv+proj",
                   r: int = 8,
                   alpha: int = 16,
                   lora_dropout: float = 0.05) -> int:
    """
    Inject LoRA into ViT attention projections inside the given backbone.
    """
    targets = {
        "qkv+proj": ["qkv", "proj"],             # DINOv3 SelfAttentionBlock uses qkv/proj
        "qkvlproj": ["q", "k", "v", "proj"],     # separate projections
        "all_attn": ["qkv", "q", "k", "v", "proj", "out_proj"],
    }.get(target, ["qkv", "proj"])

    return _replace_linear_with_lora(backbone, targets, r=r, alpha=alpha, lora_dropout=lora_dropout)

def freeze_backbone_except_lora(backbone: nn.Module):
    """Freeze everything, then unfreeze LoRA params."""
    for p in backbone.parameters():
        p.requires_grad = False
    for m in backbone.modules():
        if isinstance(m, LoRALinear):
            for p in m.parameters():
                p.requires_grad = True

# -----------------------------------------------------------------------------
# Utils & initialization
# -----------------------------------------------------------------------------
# -- Initialization --
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def kaiming_init(m: nn.Module):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
        if getattr(m, "weight", None) is not None:
            nn.init.ones_(m.weight)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)

def soft_argmax_2d(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    heatmaps: (B,T,K,H,W) -> keypoints (B,T,K,2) in heatmap pixel coords [0..W-1], [0..H-1]
    """
    B, T, K, H, W = heatmaps.shape
    prob = F.softmax(heatmaps.reshape(B*T*K, H*W), dim=-1).reshape(B*T*K, H, W)

    xs = torch.linspace(0, W - 1, W, device=heatmaps.device, dtype=heatmaps.dtype)
    ys = torch.linspace(0, H - 1, H, device=heatmaps.device, dtype=heatmaps.dtype)
    x_grid = xs.contiguous().view(1, 1, W).expand(B * T * K, H, W)
    y_grid = ys.contiguous().view(1, H, 1).expand(B * T * K, H, W)

    x = (prob * x_grid).sum(dim=(1, 2))
    y = (prob * y_grid).sum(dim=(1, 2))
    return torch.stack([x, y], dim=-1).reshape(B, T, K, 2)

# -----------------------------------------------------------------------------
# DINOv3 ViT-S/16 backbone (Meta repo) -> tokens
# -----------------------------------------------------------------------------
class DINOv3Backbone(nn.Module):
    """
    Returns:
      cls:     (B, T, D)
      patches: (B, T, P, D) where P = (224/16)^2 = 196
    Expects ImageNet-normalized 224x224 frames.
    """
    def __init__(self, ckpt_path: str, img_size: int = 224, patch: int = 16, freeze: bool = True, cfg: Dict = {}):
        super().__init__()
        self.img_size = img_size
        self.patch = patch
        self.grid = img_size // patch
        self.num_patches = self.grid * self.grid

        # self.model = vit_small(patch_size=patch, img_size=img_size)
        self.model = vit_large(patch_size=patch, img_size=img_size, adapter=cfg["model"]["adapter"], adapter_drop=cfg["model"]["adapter_drop"], 
                            num_frames=cfg["data"]["num_frames"], adapter_in_last=cfg["model"]["adapter_in_last"])
        sd = load_file(ckpt_path)
        self.model.load_state_dict(sd, strict=False)
        # self.model.eval()

        # if freeze:
        #     for p in self.model.parameters():
        #         p.requires_grad = False

        # print("\n[DEBUG] Sample param names:")
        # for name, param in list(self.model.named_parameters()):
        #     print(name)

        self.hidden_size = self.model.embed_dim

        # add temporal positional encoding to make model aware of frame order
        self.temporal_pe = TemporalPositionalEmbedding(self.hidden_size, max_T=cfg["data"]["num_frames"])
        self.temporal_cls = LearnedAbsPosEncoding(self.hidden_size)

    def forward(self, video: torch.Tensor):
        """
        video: (B,T,C,H,W) or (T,C,H,W)
        """
        added_batch = False
        if video.dim() == 4:
            video = video.unsqueeze(0)
            added_batch = True

        B, T, C, H, W = video.shape
        assert (H, W) == (self.img_size, self.img_size), f"Expected {(self.img_size, self.img_size)}, got {(H, W)}"
        x = video.contiguous().view(B * T, C, H, W)

        feats = self.model.forward_features(x)
        cls = feats.get("x_norm_clstoken", feats.get("x_clstoken", None))
        patches = feats.get("x_norm_patchtokens", feats.get("x_patchtokens", None))
        if cls is None or patches is None:
            raise RuntimeError(f"forward_features missing expected keys. Got: {list(feats.keys())}")

        D = cls.shape[-1]
        cls = cls.contiguous().view(B, T, D)

        # if self.temporal_cls is not None:
        cls = self.temporal_cls(cls)

        patches = patches.view(B, T, self.num_patches, D)
        # if self.temporal_pe is not None:
        patches = self.temporal_pe(patches)
        return (cls[0], patches[0]) if added_batch else (cls, patches)

# -----------------------------------------------------------------------------
# CLS Fusion with Patch Tokens
# -----------------------------------------------------------------------------
class FuseCLS(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(2 * d_model, d_model)
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        self.apply(kaiming_init)

    def forward(self, pooled: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
        z = torch.cat([pooled, cls], dim=-1)
        g = self.gate(z)                         # (B,T,1)
        return (1.0 - g) * pooled + g * self.proj(z)

# -----------------------------------------------------------------------------
# Optional side features
# -----------------------------------------------------------------------------
class SideFeatureFusion(nn.Module):
    def __init__(self, d_model: int, in_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.apply(kaiming_init)

    def forward(self, x: torch.Tensor, side: Optional[torch.Tensor]) -> torch.Tensor:
        return x if side is None else x + self.proj(side)

def pack_side_inputs_vis_only(vis: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if vis is None else vis.float()

# -----------------------------------------------------------------------------
# Local fine features
# -----------------------------------------------------------------------------
class FinerCNN(nn.Module):
    def __init__(self, in_dim=384, mid_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, mid_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(mid_dim, 6, kernel_size=1)  # ΔR (as 3-vector) + Δt
        )
        self.apply(kaiming_init)

    def forward(self, x):
        # x: (B, T, D)
        x = x.permute(0, 2, 1)  # (B, D, T)
        return self.net(x).permute(0, 2, 1)  # (B, T, 6)

# -----------------------------------------------------------------------------
# Per-frame token pooling
# -----------------------------------------------------------------------------
class MeanTokenPool(nn.Module):
    def forward(self, tokens_np: torch.Tensor) -> torch.Tensor:
        return tokens_np.mean(dim=2)  # (B,T,P,D) -> (B,T,D)

class AttnTokenPool(nn.Module):
    """
    Single per-frame query attends to patch tokens -> (B,T,D).
    """
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.learned_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, patches: torch.Tensor, cls: torch.Tensor | None = None) -> torch.Tensor:
        B, T, P, D = patches.shape
        if cls is None:
            q = self.q(self.learned_query).expand(B, T, -1)          # (B,T,D)
        else:
            q = self.q(cls)                                          # (B,T,D)
        k = self.k(patches)                                          # (B,T,P,D)
        v = self.v(patches)                                          # (B,T,P,D)
        attn = torch.einsum('btd,btnd->btn', q, k) / math.sqrt(D)    # (B,T,P)
        attn = self.drop(attn.softmax(dim=-1))
        pooled = torch.einsum('btn,btnd->btd', attn, v)              # (B,T,D)
        return self.out(pooled)

# -----------------------------------------------------------------------------
# Temporal encodings / modules
# -----------------------------------------------------------------------------

class TemporalPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_T: int = 512):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_T, 1, d_model))
        nn.init.normal_(self.pe, mean=0.0, std=0.02)
        self.max_T = max_T

    def forward(self, patches: torch.Tensor):
        """
        patches: (B,T,P,D)
        returns: (B,T,P,D) with temporal encoding added
        """
        B, T, P, D = patches.shape
        if T > self.max_T:
            raise ValueError(f"T={T} > max_T={self.max_T}")
        return patches + self.pe[:, :T, :, :].to(patches.dtype)
    
class LearnedAbsPosEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, mean=0.0, std=0.02)
        self.max_len = max_len

    def forward(self, x: torch.Tensor):
        T = x.shape[1]
        if T > self.max_len:
            raise ValueError(f"T={T} > max_len {self.max_len}")
        return x + self.pe[:, :T, :]

class TimeScalarEmbedding(nn.Module):
    def __init__(self, max_len: int = 512):
        super().__init__()
        self.emb = nn.Embedding(max_len, 1)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        self.max_len = max_len

    def forward(self, B: int, T: int, device, dtype):
        if T > self.max_len:
            raise ValueError(f"T={T} > max_len {self.max_len}")
        idx = torch.arange(T, device=device, dtype=torch.long)
        return self.emb(idx).view(1, T, 1).expand(B, T, 1).to(dtype=dtype)

class RelativePositionBias1D(nn.Module):
    def __init__(self, num_heads: int, max_rel: int = 256):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_heads, 2 * max_rel - 1))
        nn.init.normal_(self.bias, mean=0.0, std=0.02)
        self.max_rel = max_rel
        self.num_heads = num_heads

    def forward(self, T: int, device=None, dtype=None) -> torch.Tensor:
        pos = torch.arange(T, device=device)
        rel = pos[None, :] - pos[:, None]
        rel = rel.clamp(-self.max_rel + 1, self.max_rel - 1)
        idx = rel + (self.max_rel - 1)
        return self.bias[:, idx].to(dtype=dtype)  # (H,T,T)

class MHSA_Temporal(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        dk = d_model // nhead
        self.nhead = nhead
        self.dk = dk
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.rpb = RelativePositionBias1D(nhead, 256)
        self.apply(kaiming_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.nhead, self.dk).transpose(1, 2)  # (B,H,T,dk)
        k = self.k(x).view(B, T, self.nhead, self.dk).transpose(1, 2)
        v = self.v(x).view(B, T, self.nhead, self.dk).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)          # (B,H,T,T)
        attn = attn + self.rpb(T, x.device, x.dtype).unsqueeze(0)      # add relative bias
        attn = self.drop(attn.softmax(dim=-1))
        out = attn @ v                                                 # (B,H,T,dk)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o(out)

class TransformerBlockTemporal(nn.Module):
    def __init__(self, d_model: int, nhead: int = 6, dim_ff: int = 768, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = MHSA_Temporal(d_model, nhead, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout)
        )
        self.apply(kaiming_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.norm1(x)))
        x = x + self.ffn(self.norm2(x))
        return x

class TemporalTransformer(nn.Module):
    def __init__(self, d_model: int,
                 nhead: int = 6,
                 dim_ff: int = 768,
                 depth: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlockTemporal(d_model, nhead, dim_ff, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

class TemporalTCN(nn.Module):
    def __init__(self,
                 d_model: int,
                 hidden: int = 384,
                 kernel: int = 3,
                 depth: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.pos = LearnedAbsPosEncoding(d_model)
        assert kernel % 2 == 1, "Use odd kernel for simple 'same' padding"

        layers = []
        in_ch = d_model
        dil = 1
        for _ in range(depth):
            pad = dil * (kernel - 1) // 2
            layers += [
                nn.Conv1d(in_ch, hidden, kernel, padding=pad, dilation=dil),
                nn.ReLU(True), nn.Dropout(dropout),
            ]
            in_ch = hidden
            dil *= 2

        layers.append(nn.Conv1d(hidden, d_model, 1))
        self.net = nn.Sequential(*layers)
        self.apply(kaiming_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos(x)
        return self.net(x.transpose(1, 2)).transpose(1, 2)

class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.apply(kaiming_init)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        Ax = torch.einsum('ij,bjd->bid', A, x)       # (B, T, Din)
        return F.relu(self.lin(Ax))

def chain_adj(T: int, device=None, dtype=None, self_loop=True) -> torch.Tensor:
    A = torch.zeros((T, T), device=device, dtype=dtype)
    for t in range(T):
        if self_loop:
            A[t, t] = 1
        if t - 1 >= 0:
            A[t, t - 1] = 1
        if t + 1 < T:
            A[t, t + 1] = 1
    deg = A.sum(1, keepdim=True).clamp_min(1.0)
    return A / deg

class TemporalGCN(nn.Module):
    def __init__(self, d_model: int, hidden: int = 384, max_len: int = 512):
        super().__init__()
        self.tscalar = TimeScalarEmbedding(max_len)
        self.g1 = GraphConv(d_model + 1, hidden)
        self.g2 = GraphConv(hidden, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        A = chain_adj(T, x.device, x.dtype)
        s = self.tscalar(B, T, x.device, x.dtype)
        h = self.g1(torch.cat([x, s], dim=-1), A)
        h = self.g2(h, A)
        return self.norm(h)

# -----------------------------------------------------------------------------
# Heads: Quaternion, Heatmaps, Keypoint residuals
# -----------------------------------------------------------------------------

# ============================
# New Hybrid Keypoint+Pose Head
# ============================
class HybridTinyPoseHead(nn.Module):
    def __init__(self, d_model, grid_hw, num_kp=8, heatmap_out=(56, 56), hidden_dim=256):
        super().__init__()
        self.hm_head = TinyHeatmapHead(
            in_dim=d_model,
            grid_hw=grid_hw,
            num_kp=num_kp,
            out_hw=heatmap_out,
            mid_c=hidden_dim,
            use_film=False
        )
        self.pose_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, 7)  # (q:4, t:3)
        )

    def forward(self, patches, temporal):
        B, T, _, _ = patches.shape

        # Heatmap + keypoints
        hm = self.hm_head(patches, temporal)
        kps = soft_argmax_2d(hm)  # (B,T,K,2)

        # Clamp keypoints for stability
        kps = torch.nan_to_num(kps, nan=0.0, posinf=1e6, neginf=-1e6)
        kps = kps.clamp(min=0, max=max(self.hm_head.out_hw))  # in pixel range

        # Clamp logits to avoid Inf/NaN
        hm = torch.nan_to_num(hm, nan=0.0, posinf=10.0, neginf=-10.0)

        # Pose
        pose = self.pose_head(temporal)  # (B,T,7)
        q = pose[..., :4]
        t = pose[..., 4:]
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)

        # Aggregation (mean)
        q_agg = q.mean(dim=1)
        t_agg = t.mean(dim=1)
        q_agg = q_agg / (q_agg.norm(dim=-1, keepdim=True) + 1e-8)
        hm_agg = hm.mean(dim=1)
        kps_agg = kps.mean(dim=1)

        return {
            "heatmaps_frame": hm,
            "keypoints_frame": kps,
            "heatmaps_agg": hm_agg,
            "keypoints_agg": kps_agg,
            "q_frame": q,
            "t_frame": t,
            "q_agg": q_agg,
            "t_agg": t_agg,
        }
    
class DeformableHeatmapHead(nn.Module):
    def __init__(self, in_dim, grid_hw, num_kp, out_hw=(224,224), mid_c=128, use_film=True, dropout=0.2):
        super().__init__()
        self.Hp, self.Wp = grid_hw
        self.K = num_kp
        self.out_hw = out_hw
        self.use_film = use_film

        # Project tokens → conv feature map
        self.proj = nn.Linear(in_dim, mid_c)
        self.drop_proj = nn.Dropout(dropout)   # <---

        # Offset generator for deformable conv
        self.offset_conv = nn.Conv2d(mid_c, 18, kernel_size=3, padding=1)

        # Deformable conv + refinement
        self.defconv = DeformConv2d(mid_c, mid_c, kernel_size=3, padding=1)
        self.conv_refine = nn.Conv2d(mid_c, mid_c, kernel_size=3, padding=1)
        self.drop_conv = nn.Dropout2d(dropout)  # <---

        # Final keypoint heatmap prediction
        self.hm_pred = nn.Conv2d(mid_c, num_kp, kernel_size=1)

        if use_film:
            self.film_gamma = nn.Linear(in_dim, mid_c)
            self.film_beta  = nn.Linear(in_dim, mid_c)

        self.apply(kaiming_init)

    def forward(self, patches, temporal=None):
        B, T, P, D = patches.shape
        x = self.proj(patches)   
        x = self.drop_proj(x)  # regularize
        x = x.view(B*T, self.Hp, self.Wp, -1).permute(0,3,1,2)

        if self.use_film and temporal is not None:
            g = self.film_gamma(temporal).reshape(B*T, -1, 1, 1)
            b = self.film_beta(temporal).reshape(B*T, -1, 1, 1)
            x = x * (1 + g) + b

        offset = self.offset_conv(x)
        x = self.defconv(x, offset)
        x = self.drop_conv(F.relu(self.conv_refine(x)))  # regularize feature maps

        hm = self.hm_pred(x)  
        hm = F.interpolate(hm, size=self.out_hw, mode="bicubic", align_corners=False)
        return hm.view(B, T, self.K, self.out_hw[0], self.out_hw[1])
    
# PATCH: QuaternionPoseHead to be crop-aware

class QuaternionPoseHead(nn.Module):
    """
    Crop-aware pose head. Takes optional crop transforms (scale, offset) and adjusts t prediction.
    Input:
      x: (B,T,D) fused features
    """
    def __init__(self, d_model: int, hidden: int = 512, dropout: float = 0.1, agg: str = "attn"):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 7)   # 4 for q, 3 for t
        )
        self.agg = agg
        self.apply(kaiming_init)

    def forward(self, x: torch.Tensor):
        y = self.mlp(x)  # (B,T,7)
        q = y[..., :4]
        t_pose = y[..., 4:]
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)


        # Aggregation
        if self.agg == "mean":
            q_agg = q.mean(dim=1)
            t_agg = t_pose.mean(dim=1)
        elif self.agg == "attn":
            w = torch.softmax((y**2).sum(-1), dim=1).unsqueeze(-1)
            q_agg = (q * w).sum(dim=1)
            t_agg = (t_pose * w).sum(dim=1)
        else:
            raise ValueError(f"Unknown agg={self.agg}")

        q_agg = q_agg / (q_agg.norm(dim=-1, keepdim=True) + 1e-8)
        return (q, t_pose), (q_agg, t_agg)

class KeypointHead(nn.Module):
    """Predict per-frame and aggregated keypoints/heatmaps."""
    def __init__(self, d_model, grid_hw, num_kp=8, heatmap_out=(56,56),
                 kind="deformable", use_residual=False, hm_mid_c=128, use_film=False, agg="mean"):
        super().__init__()
        self.use_residual = use_residual
        self.kind = kind
        self.agg = agg

        if kind == "tiny":
            self.hm_head = TinyHeatmapHead(
                in_dim=d_model, grid_hw=grid_hw, num_kp=num_kp,
                out_hw=heatmap_out, mid_c=hm_mid_c, use_film=use_film
            )
        elif kind == "deformable":
            self.hm_head = DeformableHeatmapHead(
                in_dim=d_model, grid_hw=grid_hw, num_kp=num_kp,
                out_hw=heatmap_out, mid_c=hm_mid_c, use_film=use_film
            )
        else:
            self.hm_head = HeatmapHead(
                d_model, grid_hw, num_kp=num_kp,
                upsample_to=heatmap_out, use_film=use_film
            )

        if use_residual:
            self.kp_res_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 256), nn.GELU(),
                nn.Linear(256, num_kp*2)
            )
        else:
            self.kp_res_head = None

    def forward(self, patches, temporal):
        B, T, P, D = patches.shape
        # Per-frame heatmaps + keypoints
        hm = self.hm_head(patches, temporal)   # (B,T,K,H,W)
        kps = soft_argmax_2d(hm)               # (B,T,K,2)

        if self.kp_res_head is not None:
            dK = self.kp_res_head(temporal).view(B, T, -1, 2)
            kps = kps + dK

        # Aggregated (mean or attn)
        if self.agg == "mean":
            hm_agg = hm.mean(dim=1)       # (B,K,H,W)
            kps_agg = kps.mean(dim=1)     # (B,K,2)
        else:  # attention weighting
            weights = torch.softmax(hm.view(B,T,-1).sum(-1), dim=1)  # (B,T)
            hm_agg = (hm * weights.view(B,T,1,1,1)).sum(dim=1)
            kps_agg = (kps * weights.view(B,T,1,1)).sum(dim=1)

        return (hm, kps), (hm_agg, kps_agg)

class HeatmapHead(nn.Module):
    """
    Produce K heatmaps from patch tokens. Optional FiLM from temporal features.
    Inputs:
      patches: (B,T,P,D) with P=H_p*W_p
      temporal: (B,T,D)
    Returns:
      heatmaps: (B,T,K,Hm,Wm)
    """
    def __init__(self, d_model: int, grid_hw: Tuple[int, int], num_kp: int = 8,
                 upsample_to: Optional[Tuple[int, int]] = None, use_film: bool = True):
        super().__init__()
        self.Hp, self.Wp = grid_hw
        self.K = num_kp
        self.use_film = use_film

        if use_film:
            self.gamma = nn.Linear(d_model, d_model)
            self.beta  = nn.Linear(d_model, d_model)
            kaiming_init(self.gamma)
            kaiming_init(self.beta)

        self.proj = nn.Linear(d_model, num_kp)  # per-patch logits for K heatmaps
        kaiming_init(self.proj)

        self.upsample_to = upsample_to  # e.g., (56,56) or None

    def forward(self, patches: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        B, T, P, D = patches.shape
        Hp, Wp = self.Hp, self.Wp
        x = patches.view(B, T, Hp, Wp, D)

        if self.use_film:
            g = self.gamma(temporal).unsqueeze(-2).unsqueeze(-2)  # (B,T,1,1,D)
            b = self.beta(temporal).unsqueeze(-2).unsqueeze(-2)
            x = x * (1 + g) + b

        logits = self.proj(x)                # (B,T,Hp,Wp,K)
        hm = logits.permute(0, 1, 4, 2, 3)   # (B,T,K,Hp,Wp)

        if self.upsample_to is not None:
            Hm, Wm = self.upsample_to
            hm = F.interpolate(
                hm.reshape(B * T, self.K, Hp, Wp),
                size=(Hm, Wm),
                mode='bilinear',
                align_corners=False
            ).view(B, T, self.K, Hm, Wm)
        return hm

# Small conv head (token -> convs -> upsample)
class TinyHeatmapHead(nn.Module):
    def __init__(self, in_dim, grid_hw, num_kp, out_hw=(56,56), mid_c=128, use_film=False):
        super().__init__()
        self.Hp, self.Wp = grid_hw
        self.out_hw = out_hw
        self.use_film = use_film

        self.proj = nn.Linear(in_dim, mid_c)
        self.conv1 = nn.Conv2d(mid_c, mid_c, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.10)
        self.conv2 = nn.Conv2d(mid_c, num_kp, kernel_size=1)

        if use_film:
            self.film_gamma = nn.Linear(in_dim, mid_c)
            self.film_beta  = nn.Linear(in_dim, mid_c)

        self.apply(kaiming_init)

    def forward(self, tok_bt_hw_d, global_feat=None):
        x = tok_bt_hw_d
        B = T = None
        if x.dim() == 4:  # [B,T,Hp*Wp,D] -> [BT,Hp*Wp,D]
            B, T = x.shape[:2]
            x = x.reshape(B*T, self.Hp*self.Wp, x.shape[-1])

        x = self.proj(x)                                  # [BT, Hp*Wp, C]
        x = x.transpose(1, 2).reshape(-1, x.shape[-1], self.Hp, self.Wp)  # [BT,C,Hp,Wp]

        if self.use_film and global_feat is not None:
            g = global_feat
            if g.dim() == 3:
                g = g.reshape(-1, g.shape[-1])
            gamma = self.film_gamma(g).unsqueeze(-1).unsqueeze(-1)  # [BT,C,1,1]
            beta  = self.film_beta(g).unsqueeze(-1).unsqueeze(-1)   # [BT,C,1,1]
            x = x * (1 + gamma) + beta

        x = self.act(self.conv1(x))                      # [BT,C,Hp,Wp]
        x = self.drop(x)
        x = self.conv2(x)                                # [BT,K,Hp,Wp]
        x = F.interpolate(x, size=self.out_hw, mode="bicubic", align_corners=False)  # [BT,K,H,W]
        if B is not None:
            x = x.reshape(B, T, x.shape[1], x.shape[2], x.shape[3])  # [B,T,K,H,W]
        return x

# -----------------------------------------------------------------------------
# Base model and variants (NO temporal stride inside)
# -----------------------------------------------------------------------------
class BasePoseModel(nn.Module):
    """
    - Backbone: Meta DINOv3 ViT-S/16 (tokens)
    - Per-frame pooling of patch tokens: 'cls' | 'mean' | 'attn'
    - Temporal head: 'tcn' | 'transformer' | 'gcn'
    - Output mode:
        * 'quat' -> quaternion + translation (B,T,4) & (B,T,3)
        * 'kp_heatmap' -> heatmaps (B,T,K,Hm,Wm) + keypoints (B,T,K,2)
    """
    def __init__(self,
                 temporal_kind: Literal['tcn', 'transformer', 'gcn'],
                 aggregator: Literal['cls', 'mean', 'attn'] = 'attn',
                 output_mode: Literal['quat', 'kp_heatmap', 'both', 'hybrid'] = 'hybrid',
                 ckpt_path: str = BACKBONE_CKPT,
                 freeze_backbone: bool = True,
                 fuse_side: bool = False, num_kp: int = 8, d_side: int = 0,
                 heatmap_out: Optional[Tuple[int, int]] = (56, 56),
                 cam_intrinsics=None,
                 **kwargs):
        super().__init__()
        assert isinstance(ckpt_path, str) and len(ckpt_path) > 0, "Provide ckpt_path to DINOv3 safetensors."

        # config file
        self.cfg = kwargs.get("cfg", {})
        # backbone
        self.backbone = DINOv3Backbone(ckpt_path=ckpt_path, freeze=freeze_backbone, cfg=self.cfg)

        # freeze_all_but_dualpath_adapters(self.backbone)

        # ----- LoRA wiring (optional) -----
        cfg_lora = self.cfg.get("train", {}).get("lora", {})
        if cfg_lora.get("enabled", False):
            wrapped = apply_vit_lora(
                self.backbone,                            # recursion reaches inner ViT
                target=cfg_lora.get("target", "qkv+proj"),
                r=int(cfg_lora.get("r", 8)),
                alpha=int(cfg_lora.get("alpha", 16)),
                lora_dropout=float(cfg_lora.get("dropout", 0.05)),
            )
            freeze_backbone_except_lora(self.backbone)
            if wrapped == 0:                              # <--- fix ==, not =
                print("[LoRA] WARNING: no attention linears were wrapped. (check names/target).", flush=True)
            else:
                print(f"[LoRA] Wrapped {wrapped} Linear layers in backbone.", flush=True)
        else:
            # keep original freeze behavior if LoRA disabled
            if freeze_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad = False

        D = self.backbone.hidden_size
        self.grid_hw = (self.backbone.grid, self.backbone.grid)
        self.num_kp = num_kp
        self.output_mode = output_mode
        self.aggregator_kind = aggregator
        self.fuse_cls = None
        # self.dp_adapter = None

        # -------------------------------
        # Dual-Path Adapter (optional)
        # -------------------------------
        # dp_cfg = self.cfg.get("model", {}).get("dp_adapter", {})
        # self.dp_adapter = None

        # if dp_cfg.get("enabled", False):
        #     self.dp_adapter = DualPathAdapter(
        #         d_model=D,
        #         r=int(dp_cfg.get("r", 8)),
        #         kT=int(dp_cfg.get("kT", 3)),
        #         dorpout=float(dp_cfg.get("dropout", 0.05)),
        #     )
        #     print("[DualPath] Injecting dual-path adapter with r={}, kT={}, dropout={}".format(
        #         dp_cfg.get("r", 8), dp_cfg.get("kT", 3), dp_cfg.get("dropout", 0.05)
        #     ), flush=True)

        #     # Temporal Positional Encoding for patch tokens
        #     self.temporal_pe = TemporalPositionalEmbedding(D, max_T=int(dp_cfg.get("max_T", 512)))

        #     # Optional temporal PE for CLS token
        #     if dp_cfg.get("use_cls_pe", False): 
        #         self.temporal_pe_cls = nn.Embedding(int(dp_cfg.get("max_T", 512)), D)
        #         nn.init.normal_(self.temporal_pe_cls.weight, mean=0.0, std=0.02)
        #         print("[DualPath] Using temporal PE for CLS token.", flush=True)


        # inititlize local feature CNN module (optional)
        self.use_finer_cnn = kwargs.get("use_finer_cnn", False)
        if self.use_finer_cnn:
            self.finer_cnn = FinerCNN(in_dim=D, mid_dim=128)
            self.final_projection = nn.Linear(6, D)  # 6 for ΔR (as 3-vector) + Δt
            self.fuse_mlp = nn.Sequential(
                nn.Linear(2*D, D),
                nn.GELU(),
                nn.Linear(D, D),
                nn.Dropout(0.1)
                )
            self.fuse_ln = nn.LayerNorm(D)
            self.final_projection.apply(kaiming_init)
            self.fuse_mlp.apply(kaiming_init)
            print("[FinerCNN] Using local fine feature CNN module.", flush=True)

        # Aggregators
        if aggregator == 'cls':
            self.agg = None
        elif aggregator == 'mean':
            self.agg = MeanTokenPool()
        elif aggregator == 'attn':
            self.agg = AttnTokenPool(D)
            if kwargs.get("fuse_cls", False):
                self.fuse_cls = FuseCLS(D)
        else:
            raise ValueError("aggregator must be 'cls'|'mean'|'attn'")

        # Side fusion
        self.fuse_side = fuse_side
        self.side_fusion = SideFeatureFusion(D, in_dim=num_kp) if fuse_side else None

        # Temporal head
        if temporal_kind == 'tcn':
            self.temporal = TemporalTCN(
                D,
                hidden=kwargs.get("tcn_hidden", 384),
                kernel=kwargs.get("tcn_kernel", 3),
                depth=kwargs.get("tcn_depth", 2),
                dropout=kwargs.get("tcn_dropout", 0.1),
            )
        elif temporal_kind == 'transformer':
            self.temporal = TemporalTransformer(
                D,
                nhead=kwargs.get("transformer_nhead", 6),
                dim_ff=kwargs.get("transformer_ff", 768),
                depth=kwargs.get("transformer_depth", 2),
                dropout=kwargs.get("transformer_dropout", 0.1),
            )
        elif temporal_kind == 'gcn':
            self.temporal = TemporalGCN(D)
        
        else:
            self.temporal = None
            # print("[WARNING] No temporal module used, just per-frame features.", flush=True)
            # raise ValueError("temporal_kind invalid")

        # Heads
        # pose regression
        
        if output_mode == 'hybrid':
            self.hybrid_head = HybridTinyPoseHead(
                    d_model=D,
                    grid_hw=self.grid_hw,
                    num_kp=num_kp,
                    heatmap_out=heatmap_out
                )
            
        if output_mode not in ('hybrid'):
            raise ValueError("output_mode must be 'hybrid'")
        
        # Pnp logic
        self.use_pnp = self.cfg.get("model", {}).get("use_pnp", False)
        if cam_intrinsics is not None:
            K_np = cam_intrinsics["K"]
            self.register_buffer("cam_K", torch.tensor(K_np, dtype=torch.float32))
            dist_np = np.array(cam_intrinsics.get("dist", [0, 0, 0, 0, 0]), dtype=np.float32)
            self.dist_coeffs = dist_np.reshape(-1, 1)  # (5,1) OpenCV style
        else:
            self.cam_K = None
            self.dist_coeffs = None


    def _per_frame_embed(self, patches: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
        if self.aggregator_kind == 'cls':
            return cls
        elif self.aggregator_kind == 'mean':
            return self.agg(patches)
        else:  # 'attn'
            pooled = self.agg(patches, cls)  # (B,T,D)
            return self.fuse_cls(pooled, cls) if self.fuse_cls is not None else pooled

    def forward(
        self,
        video: torch.Tensor,                      # (B,T,C,H,W), already normalized
        # keypoints: Optional[torch.Tensor] = None, # (B,T,K,2) optional
        visibility: Optional[torch.Tensor] = None,# (B,T,K) optional
        # heatmap_stats: Optional[torch.Tensor] = None,  # optional side info (unused)
        points_3d: Optional[np.ndarray] = None,
        use_ransac: bool = True
    ) -> Dict[str, object]:
        """
        NO temporal sub-sampling inside. Uses every frame it’s given.
        """
        # 1) Backbone tokens
        cls, patches = self.backbone(video)     # (B,T,D), (B,T,P,D)

        # # 2) Temporal Positional Encoding
        # patches = self.temporal_pe(patches)  # (B,T,P,D)
        # if hasattr(self, "temporal_pe_cls"):
        #     t = torch.arange(cls.shape[1], device=cls.device) #(T,)
        #     pe = self.temporal_pe_cls(t)  # (T,D)
        #     cls = cls + pe # (B,T,D)
        #     # cls = cls + self.temporal_pe_cls(cls.shape[1])  # (B,T,D)


        # # 3) Dual-path adapter
        # if self.dp_adapter is not None:
        #     patches = self.dp_adapter(patches)          # (B,T,P,D)


        # 4) Per-frame embedding -> temporal features
        per_frame = self._per_frame_embed(patches, cls)  # (B,T,D)
        if self.side_fusion is not None:
            side = pack_side_inputs_vis_only(visibility) # (B,T,K) or None
            per_frame = self.side_fusion(per_frame, side)

        temporal = self.temporal(per_frame) if isinstance(self.temporal, nn.Module) else per_frame          # (B,T,D)

        # 5) Outputs
        out: Dict[str, object] = {}

        # local features 
        if getattr(self, "use_finer_cnn", False) and hasattr(self, "finer_cnn"):
            finer_feat = self.finer_cnn(temporal)  # (B,T,6)
            out['delta_Rt'] = finer_feat[:, 1:]  # use the raw finer feature as  Visual Odometry (delta_R/t)
            finer_project = self.final_projection(finer_feat)  # (B,T,D)
            residual = temporal + finer_project                   # (B,T,D) ← raw sum
            fused_feat = torch.cat([temporal, finer_project], dim=-1)
            fused_feat = self.fuse_mlp(fused_feat) # (B, T, 2D) → (B, T, D)
            fused_feat = fused_feat + residual  # (B,T,D)
            fused_feat = self.fuse_ln(fused_feat) # (B, T, D) final normalized fused feature
        else:
            fused_feat = temporal
            
        
        if self.output_mode == 'hybrid':
            out = self.hybrid_head(patches, fused_feat)



        else:
            out['heatmaps_frame'] = out['heatmaps_agg'] = None
            out['keypoints_frame'] = out['keypoints_agg'] = None


        # Optional PnP
        if self.use_pnp and 'keypoints' in out:
            assert hasattr(self, 'cam_K') and self.cam_K is not None, "Camera intrinsics not loaded. Set `cam_intrinsics`."
            K = self.cam_K.detach().cpu().numpy()  # (3,3)
            B, T, Kpts, _ = out['keypoints'].shape
            Rs_all, ts_all = [], []
            vis_np = visibility.detach().cpu().numpy() if visibility is not None else None
            for b in range(B):
                k2d = out['keypoints'][b].detach().cpu().numpy()
                Hm, Wm = out['heatmaps'].shape[-2], out['heatmaps'].shape[-1]
                # scale to 224x224 image coords
                scale_x = self.backbone.img_size / float(Wm)
                scale_y = self.backbone.img_size / float(Hm)
                k2d_img = k2d.copy()
                k2d_img[..., 0] *= scale_x
                k2d_img[..., 1] *= scale_y
                visb = vis_np[b] if vis_np is not None else None
                Rb, tb = pnp_sequence(k2d_img, points_3d, K, visibility=visb, dist_coeffs=self.dist_coeffs,use_ransac=use_ransac)
                Rs_all.append(Rb)
                ts_all.append(tb)
            out['R'] = np.stack(Rs_all, axis=0)   # (B,T,3,3)
            out['t_pose'] = np.stack(ts_all, axis=0)   # (B,T,3)
            out['q_pnp'], out['t_pnp'] = convert_Rt_to_qt(out['R'], out['t'])
            print(f"[PnP] Completed for batch {B}, seq len {T}, {Kpts} keypoints.", flush=True)

        return out

# Concrete temporal variants
class SpacecraftPoseTCN(BasePoseModel):
    def __init__(self, **kwargs):
        super().__init__(temporal_kind='tcn', **kwargs)

class SpacecraftPoseTransformer(BasePoseModel):
    def __init__(self, **kwargs):
        super().__init__(temporal_kind='transformer', **kwargs)

class SpacecraftPoseGCN(BasePoseModel):
    def __init__(self, **kwargs):
        super().__init__(temporal_kind='gcn', **kwargs)

class SpacecraftPoseDualPath(BasePoseModel):
    def __init__(self, **kwargs):
        kwargs["temporal_kind"] = None  # disable temporal modeling
        super().__init__(**kwargs)
        # assert self.dp_adapter is not None, "DualPathAdapter is required but not enabled in config!"

        # # Enforce dual-path enabled regardless of config
        # dp_cfg = self.cfg.get("model", {}).get("dp_adapter", {})
        # self.dp_adapter = DualPathAdapter(
        #     d_model=self.backbone.hidden_size,
        #     r=dp_cfg.get("dp_r", 6),
        #     kT=dp_cfg.get("dp_kT", 2),
        #     dorpout=dp_cfg.get("dp_dropout", 0.05),
        # )

# -----------------------------------------------------------------------------
# PnP wrapper (unchanged)
# -----------------------------------------------------------------------------

def convert_Rt_to_qt(R: np.ndarray, t: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts (B,T,3,3) rotation matrices and (B,T,3) translations
    to unit quaternions (B,T,4) and translations (B,T,3), both as torch.Tensor.
    """
    import scipy.spatial.transform
    B, T = R.shape[:2]
    R_flat = R.reshape(B*T, 3, 3)
    t_flat = t.reshape(B*T, 3)

    q_all = []
    for R_mat in R_flat:
        rot = scipy.spatial.transform.Rotation.from_matrix(R_mat)
        q = rot.as_quat()  # returns [x,y,z,w]
        q_all.append(q[[3, 0, 1, 2]])  # reorder to [w,x,y,z] or [qw,qx,qy,qz]

    q_out = np.stack(q_all, axis=0).reshape(B, T, 4)
    t_out = t_flat.reshape(B, T, 3)

    q_tensor = torch.from_numpy(q_out).float()
    t_tensor = torch.from_numpy(t_out).float()
    return q_tensor, t_tensor

def pnp_sequence(
    keypoints_2d: np.ndarray,      # (T,K,2)
    points_3d: np.ndarray,         # (K,3)
    K: np.ndarray,                 # (3,3)
    visibility: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
    use_ransac: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    T_len, Kpts, _ = keypoints_2d.shape
    Rs = np.zeros((T_len, 3, 3), dtype=np.float64)
    ts = np.zeros((T_len, 3), dtype=np.float64)
    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    for t in range(T_len):
        vis = np.ones((Kpts,), bool) if visibility is None else visibility[t].astype(bool)
        pts2d = keypoints_2d[t][vis].reshape(-1, 1, 2).astype(np.float64)
        pts3d = points_3d[vis].reshape(-1, 1, 3).astype(np.float64)
        if pts2d.shape[0] < 4:
            continue
        if use_ransac:
            ok, rvec, tvec, _ = cv2.solvePnPRansac(
                pts3d, pts2d, K, dist_coeffs, flags=cv2.SOLVEPNP_EPNP,
                reprojectionError=3.0, iterationsCount=100
            )
        else:
            ok, rvec, tvec = cv2.solvePnP(pts3d, pts2d, K, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
        if ok:
            R, _ = cv2.Rodrigues(rvec)
            Rs[t] = R
            ts[t] = tvec.reshape(-1)
    return Rs, ts