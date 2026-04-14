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


# ======== CONFIG: path to compact, strictly-loaded safetensors ========
BACKBONE_CKPT = "./dinov3_vits16_meta_strict.safetensors"

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
        # if hasattr(self.model, "blocks"):
        #     for blk in self.model.blocks:
        #         if hasattr(blk, "num_frames"):
        #             blk.num_frames = T

        # ---- normalized tokens (what you already used) ----
        cls_norm = feats.get("x_norm_clstoken", None)
        patches_norm = feats.get("x_norm_patchtokens", None)

        # ---- prenorm full sequence (CLS + patches) ----
        x_prenorm = feats.get("x_prenorm", None)

        if cls_norm is None or patches_norm is None or x_prenorm is None:
            raise RuntimeError(
                f"forward_features missing expected keys. Got: {list(feats.keys())}"
            )

        D = cls_norm.shape[-1]

        # cls: (BT, D) -> (B, T, D)
        cls = cls_norm.contiguous().view(B, T, D)
        cls = self.temporal_cls(cls)

        # patches_norm: (BT, P, D) -> (B, T, P, D)
        patches = patches_norm.contiguous().view(B, T, self.num_patches, D)
        patches = self.temporal_pe(patches)

        # prenorm patches: x_prenorm is (BT, 1 + P, D)
        patches_prenorm = x_prenorm[:, 1:, :].contiguous().view(B, T, self.num_patches, D)
        patches_prenorm = self.temporal_pe(patches_prenorm)

        # return both
        if added_batch:
            return (cls[0], patches[0], patches_prenorm[0])
        else:
            return (cls, patches, patches_prenorm)

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

# -----------------------------------------------------------------------------
# Heads: Quaternion, Heatmaps, Keypoint residuals
# -----------------------------------------------------------------------------
    
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

# Small conv head (token -> convs -> upsample)
class TinyHeatmapHead(nn.Module):
    """
    Predicts FULL-RESOLUTION 224×224 heatmaps from ViT patch tokens.
    Uses a learned multi-stage upsampling head (no interpolation).
    """

    def __init__(
        self,
        in_dim,                 # ViT token dim (1024 for ViT-L/16)
        grid_hw,                # patch grid, e.g. (14,14)
        num_kp,                 # number of keypoints
        out_hw=(224, 224),      # final heatmap resolution
        mid_c=128,
        use_film=False
    ):
        super().__init__()

        self.Hp, self.Wp = grid_hw           # 14×14 grid for ViT-L/16
        self.out_hw = out_hw
        self.use_film = use_film

        # ---------------------------
        # Token → Conv feature map
        # ---------------------------
        self.proj = nn.Linear(in_dim, mid_c)

        # Shape: (BT, mid_c, 14, 14)
        # ---------------------------

        self.conv1 = nn.Conv2d(mid_c, mid_c, 3, padding=1)
        self.act = nn.GELU()

        # -----------------------------------------------
        # Multi-stage upsampling: 14 → 28 → 56 → 112 → 224
        # -----------------------------------------------
        self.up1 = nn.ConvTranspose2d(mid_c, mid_c, kernel_size=4, stride=2, padding=1)  # 14→28
        self.up2 = nn.ConvTranspose2d(mid_c, mid_c, kernel_size=4, stride=2, padding=1)  # 28→56
        self.up3 = nn.ConvTranspose2d(mid_c, mid_c, kernel_size=4, stride=2, padding=1)  # 56→112
        # self.up4 = nn.ConvTranspose2d(mid_c, num_kp, kernel_size=4, stride=2, padding=1) # 112→224
        # self.up4 = nn.Conv2d(mid_c, num_kp, kernel_size=3, padding=1)  # 112→128 (via small conv)
        self.extra_upsample = nn.Upsample(size=out_hw, mode='bilinear', align_corners=False)
        # Add this final conv after upsampling
        self.final_conv = nn.Conv2d(mid_c, num_kp, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.bn2 = nn.BatchNorm2d(mid_c)
        self.bn3 = nn.BatchNorm2d(mid_c)

        # FiLM for optional modulation
        if use_film:
            self.film_gamma = nn.Linear(in_dim, mid_c)
            self.film_beta  = nn.Linear(in_dim, mid_c)

        self.apply(kaiming_init)

    def forward(self, tokens, global_feat=None):
        """
        tokens: (B,T,P,D)
        Returns heatmaps: (B,T,K,224,224)
        """
        B, T, P, D = tokens.shape
        BT = B * T

        # ---------------------------
        # Linear projection → conv map
        # ---------------------------
        x = self.proj(tokens)                          # (B,T,P,mid_c)
        x = x.view(BT, P, -1).transpose(1, 2)          # (BT, mid_c, P)
        x = x.reshape(BT, -1, self.Hp, self.Wp)        # (BT, mid_c, 14, 14)

        # Optional FiLM conditioning
        if self.use_film and global_feat is not None:
            g = global_feat.reshape(BT, -1)
            gamma = self.film_gamma(g).unsqueeze(-1).unsqueeze(-1)
            beta  = self.film_beta(g).unsqueeze(-1).unsqueeze(-1)
            x = x * (1 + gamma) + beta

        # ---------------------------
        # Conv + learned upsampling
        # ---------------------------
        x = self.act(F.dropout(self.conv1(x), 0.1))
        
        x = self.bn1(self.act(self.up1(x)))  # 14→28
        x = self.bn2(self.act(self.up2(x)))  # 28→56
        x = self.bn3(self.act(self.up3(x)))  # 56→112

        x = self.extra_upsample(x)    # 112→224
        x = self.final_conv(x)        # final K-channel heatmap

        # ---------------------------
        # Reshape back to (B,T,K,H,W)
        # ---------------------------
        return x.view(B, T, -1, x.shape[-2], x.shape[-1])
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
                 output_mode: Literal[ 'heatmap'] = 'heatmap',
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

        D = self.backbone.hidden_size
        self.grid_hw = (self.backbone.grid, self.backbone.grid)
        self.output_mode = output_mode
        self.aggregator_kind = aggregator
        self.fuse_cls = None
        self.num_kp = num_kp
        self.use_film = self.cfg["model"].get("use_film_in_hm_head", False)

        # Heatmap head
        self.hm_head = TinyHeatmapHead(
            in_dim=D,
            grid_hw=self.grid_hw,
            num_kp=self.num_kp,
            out_hw=heatmap_out,
            mid_c=self.cfg["model"].get("hm_mid_c", 128),
            use_film=self.use_film
        )
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
        visibility: Optional[torch.Tensor] = None,# (B,T,K) optional
    ) -> Dict[str, object]:
        """
        NO temporal sub-sampling inside. Uses every frame it’s given.
        """
        # 1) Backbone tokens
        cls, patches_norm, patches_prenorm = self.backbone(video)     # (B,T,D), (B,T,P,D)

        # 4) Per-frame embedding -> temporal features
        per_frame = self._per_frame_embed(patches_norm, cls)  # (B,T,D)
        if self.side_fusion is not None:
            side = pack_side_inputs_vis_only(visibility) # (B,T,K) or None
            per_frame = self.side_fusion(per_frame, side)

        # 5) Outputs
        out: Dict[str, object] = {}

        heatmaps = self.hm_head(patches_norm)  # (B,T,K,Hm,Wm)
        out['heatmaps'] = heatmaps
        
        out["patch_tokens_norm"] = patches_norm
        out["patch_tokens_prenorm"] = patches_prenorm

        return out

# Concrete temporal variants

class SpacecraftPoseDualPath(BasePoseModel):
    def __init__(self, **kwargs):
        kwargs["temporal_kind"] = None  # disable temporal modeling
        super().__init__(**kwargs)

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