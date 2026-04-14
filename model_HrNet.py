# model_dinov3_litehrnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal, Dict, Iterable
from model import DINOv3Backbone, soft_argmax_2d, MeanTokenPool, AttnTokenPool, FuseCLS
from LiteHrNet.models.backbones.litehrnet import LiteHRNet
import numpy as np


# ======== CONFIG: path to your compact, strictly-loaded safetensors ========
BACKBONE_CKPT = "./dinov3_vits16_meta_strict.safetensors"

class DINOv3LiteHRNetPose(nn.Module):
    def __init__(self, 
                 dinov3_ckpt: str,
                 litehrnet_cfg: dict,
                 cfg: dict = {},
                 num_kp: int = 8,
                 img_size: int = 224,
                 heatmap_size: int = 56,
                 return_coords: bool = True):
        super().__init__()

        # ViT backbone (DINOv3)
        self.backbone = DINOv3Backbone(ckpt_path=dinov3_ckpt, img_size=img_size, freeze=True, cfg=cfg)
        D = self.backbone.hidden_size  # e.g. 384

        # Convert patch tokens → 2D feature map (reshaped grid)
        self.grid_hw = (img_size // self.backbone.patch, img_size // self.backbone.patch)
        self.proj = nn.Linear(D, 32)  # LiteHRNet input = 32 channels

        # LiteHRNet
        self.hrnet = LiteHRNet(**litehrnet_cfg)

        # Final head → keypoints heatmaps
        self.head = nn.Conv2d(self.hrnet.channels, num_kp, kernel_size=1)

        self.upsample = nn.Upsample(size=(heatmap_size, heatmap_size), mode="bilinear", align_corners=False)
        self.return_coords = return_coords
        self.num_kp = num_kp

    def forward(self, video: torch.Tensor):
        """
        video: (B, T, C, H, W)
        returns:
            - heatmaps: (B, T, K, Hm, Wm)
            - keypoints (optional): (B, T, K, 2)
        """
        B, T, C, H, W = video.shape
        cls, patches = self.backbone(video)  # cls: (B, T, D), patches: (B, T, P, D)

        Hp, Wp = self.grid_hw
        x = self.proj(patches).view(B*T, Hp, Wp, -1).permute(0, 3, 1, 2).contiguous()  # (BT, 32, H, W)

        # Pass through LiteHRNet
        feats = self.hrnet(x)  # (BT, C, H, W) where C = out channels (e.g., 128)

        # Predict heatmaps
        hm = self.head(feats)  # (BT, K, H, W)
        hm = self.upsample(hm)  # (BT, K, Hm, Wm)
        hm = hm.view(B, T, self.num_kp, hm.shape[2], hm.shape[3])

        if self.return_coords:
            kps = soft_argmax_2d(hm)  # (B, T, K, 2)
            return hm, kps
        else:
            return hm
        
if __name__ == "__main__":
    import yaml
    with open("configs.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    model_cfg = cfg["model"]
    litehrnet_cfg = model_cfg["litehrnet"]

    model = DINOv3LiteHRNetPose(
        dinov3_ckpt=model_cfg["dinov3_ckpt"],
        cfg = model_cfg,
        litehrnet_cfg=litehrnet_cfg,
        num_kp=8,
        img_size=224,
        heatmap_size=56,
        return_coords=True
    )

    # Input: (B, T, 3, 224, 224)
    video = torch.randn(2, 8, 3, 224, 224)
    hm, kps = model(video)  # (2,8,8,56,56), (2,8,8,2)



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
                 ckpt_path: str = BACKBONE_CKPT,
                 freeze_backbone: bool = True,
                 aggregator: Literal['cls', 'mean', 'attn'] = 'attn',
                 heatmap_out: Optional[Tuple[int, int]] = (56, 56),
                 num_kp:int = 8,
                 **kwargs):
        super().__init__()
        assert isinstance(ckpt_path, str) and len(ckpt_path) > 0, "Provide ckpt_path to DINOv3 safetensors."

        # config file
        self.cfg = kwargs.get("cfg", {})
        self.num_kp = num_kp
        # backbone
        self.backbone = DINOv3Backbone(ckpt_path=ckpt_path, freeze=freeze_backbone, cfg=self.cfg)
        D = self.backbone.hidden_size # e.g. 384

        # Convert patch tokens → 2D feature map (reshaped grid)
        self.grid_hw = (self.backbone.grid, self.backbone.grid)
        self.proj = nn.Linear(D, 32)  # LiteHRNet input = 32 channels

        # LiteHRNet
        self.hrnet = LiteHRNet(**litehrnet_cfg)

        # Final head → keypoints heatmaps
        self.head = nn.Conv2d(self.hrnet.channels, num_kp, kernel_size=1)


        self.upsample = nn.Upsample(size=heatmap_out, mode="bilinear", align_corners=False)


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

class SpacecraftPoseDualPath(BasePoseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)