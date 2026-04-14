# model_hm_dsnt.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DINOv3ViTModel

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

# ---------------- Adapters ----------------
class SpatioTemporal3DAdapter(nn.Module):
    """
    Patch-token adapter with 3D CNN branches:
      - spatial:  (1, ks, ks)  (within-frame)
      - temporal: (kt, 1, 1)  (across-frame)
    Input:  (B,T,P,D) where P=Hp*Wp
    Output: (B,T,P,D)
    """
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
        fuse: str = "sum",  # "sum" or "concat"
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

# For adpater in series with MHSA module
class PatchOnlyPreMHSAAdapter(nn.Module):
    """
    Works on flattened tokens (BT,N,D).
    Applies spatiotemporal 3D adapter ONLY to PATCH tokens.
    Leaves CLS/REG unchanged.
    Stores last adapted patch tokens as self.last_patch_bt: (BT,P,D)
    """
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

        # will be populated each forward
        self.last_patch_bt = None  # (BT,P,D)

    def forward(self, x_btnd: torch.Tensor) -> torch.Tensor:
        BT, N, D = x_btnd.shape
        assert BT % self.T == 0, f"BT={BT} must be divisible by T={self.T}"
        B = BT // self.T

        nreg = self.n_register_tokens
        if nreg is None:
            nreg = N - 1 - self.P
            assert nreg >= 0, f"Cannot infer n_register_tokens: N={N}, P={self.P}"

        n_prefix = 1 + nreg
        assert N >= n_prefix + self.P, f"N={N} too small for prefix={n_prefix} + P={self.P}"

        prefix = x_btnd[:, :n_prefix, :]
        patch  = x_btnd[:, n_prefix:n_prefix + self.P, :]
        suffix = x_btnd[:, n_prefix + self.P:, :]

        patch = patch.view(B, self.T, self.P, D)
        patch = self.patch_adapter(patch)              # (B,T,P,D)
        patch_bt = patch.view(BT, self.P, D)           # (BT,P,D)

        # STORE for token-alignment loss (keep grad!)
        self.last_patch_bt = patch_bt

        return torch.cat([prefix, patch_bt, suffix], dim=1)
    

# for adapters as in parallel with FFN module
class PatchOnlyDeltaAdapter(nn.Module):
    def __init__(self, d, grid_hw, T, n_register_tokens=None, bottleneck=64, drop=0.1,
                 use_ln=True, scale_init=1e-3, ks=3, kt=3, fuse="sum"):
        super().__init__()
        self.core = PatchOnlyPreMHSAAdapter(
            d=d, grid_hw=grid_hw, T=T, n_register_tokens=n_register_tokens,
            bottleneck=bottleneck, drop=drop, use_ln=use_ln,
            scale_init=scale_init, ks=ks, kt=kt, fuse=fuse
        )

    def forward(self, x_btnd: torch.Tensor) -> torch.Tensor:
        # core returns x + delta_on_patches; so delta = core(x) - x
        y = self.core(x_btnd)
        return y - x_btnd
    
# ---------------- HF block / FF module discory ------------------
def _find_mlp_module(block: nn.Module):
    for name in ["mlp", "ffn", "feed_forward", "feedforward"]:
        if hasattr(block, name):
            return getattr(block, name)
    for n, m in block.named_children():
        if "mlp" in n.lower() or "ffn" in n.lower() or "feed" in n.lower():
            return m
    return None

# ---------------- HF block / attention discovery ----------------
def _find_attention_module(block: nn.Module):
    # common names
    for name in ["attention", "self_attn", "attn", "mha", "mhsa"]:
        if hasattr(block, name):
            return getattr(block, name)
    # fallback by child name heuristic
    for n, m in block.named_children():
        if "attn" in n.lower() or "attention" in n.lower() or "mhsa" in n.lower():
            return m
    return None

def _get_blocks_from_hf_model(model: nn.Module):
    """
    Return a ModuleList/list of transformer blocks from HF DINOv3ViTModel.
    Tries common paths, then heuristics.
    """
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

    # heuristic scan: first ModuleList whose first element has an attention module
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            if _find_attention_module(mod[0]) is not None:
                print(f"[INFO] Using blocks from: {name}")
                return mod

    raise AttributeError("Could not find transformer blocks in DINOv3ViTModel. Print model tree and update discovery.")

# ---------------- Backbone wrapper with pre-MHSA hooks ----------------
class DINOv3backbone(nn.Module):
    def __init__(
        self,
        hf_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        adapter_bottleneck=64,
        adapter_drop=0.1,
        adapter_in_last=None,
        num_frames=16,
        grid_hw=(14, 14),
        n_register_tokens: int | None = None,  # None => infer at runtime
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
        
        # self.ffn_adapters = nn.ModuleList()
        # for i in range(n_blocks):
        #     if i in use_idx:
        #         self.ffn_adapters.append(
        #             PatchOnlyDeltaAdapter(
        #                 d=d, grid_hw=grid_hw, T=num_frames, n_register_tokens=n_register_tokens,
        #                 bottleneck=adapter_bottleneck, drop=adapter_drop,
        #                 ks=ks, kt=kt, fuse=fuse, scale_init=1e-3
        #             )
        #         )
        #     else:
        #         self.ffn_adapters.append(nn.Identity())

        self._install_prehooks(blocks)

    def _install_prehooks(self, blocks):
        for i, blk in enumerate(blocks):
            attn = _find_attention_module(blk)
            if attn is None:
                raise RuntimeError(f"Could not locate attention module in block {i}.")
            adapter = self.adapters[i]

            # mlp = _find_mlp_module(blk)
            # if mlp is None:
            #     raise RuntimeError(f"Could not locate MLP/FFN module in block {i}.")
            # ffn_adapter = self.ffn_adapters[i]

            def make_mhsa_hook(adapter_i):
                def hook(module, inputs):
                    x = inputs[0]          # (BT,N,D)
                    x2 = adapter_i(x)
                    return (x2,) + inputs[1:]
                return hook
            
            # def make_mlp_hook(ffn_adapter_i):
            #     def hook(module, inputs, output):
            #         x_in = inputs[0]          # (BT,N,D) input to MLP
            #         delta = ffn_adapter_i(x_in)
            #         return output + delta
            #     return hook

            # mlp.register_forward_hook(make_mlp_hook(ffn_adapter))

            attn.register_forward_pre_hook(make_mhsa_hook(adapter), with_kwargs=False)

    def get_last_adapter_patch_bt(self):
        """
        Returns last active adapter's stored patch tokens: (BT,P,D) or None.
        """
        for ad in reversed(self.adapters):
            if isinstance(ad, PatchOnlyPreMHSAAdapter):
                if getattr(ad, "last_patch_bt", None) is not None:
                    return ad.last_patch_bt
        return None

    def get_all_adapter_patches_bt(self):
        """
        Returns list of (layer_idx, patch_tokens) for every active adapter.
        Each patch_tokens is (BT, P, D).
        """
        results = []
        for i, ad in enumerate(self.adapters):
            if isinstance(ad, PatchOnlyPreMHSAAdapter):
                if getattr(ad, "last_patch_bt", None) is not None:
                    results.append((i, ad.last_patch_bt))
        return results

    def forward_tokens(self, x):
        out = self.backbone(pixel_values=x, return_dict=True)
        return out.last_hidden_state  # (BT,N,D)


# ---------- Modified Heatmap Head ----------
# class TinyHeatmapHead(nn.Module):
#     def __init__(self, d, grid_hw=(14, 14), num_kp=8, out_hw=(224, 224), mid_c=128):
#         super().__init__()
#         self.Hp, self.Wp = grid_hw

#         self.proj = nn.Linear(d, mid_c)
#         self.conv = nn.Conv2d(mid_c, mid_c, 3, padding=1)
#         self.act = nn.GELU()
#         self.up1 = nn.ConvTranspose2d(mid_c, mid_c, 4, 2, 1)
#         self.up2 = nn.ConvTranspose2d(mid_c, mid_c, 4, 2, 1)
#         self.up3 = nn.ConvTranspose2d(mid_c, mid_c, 4, 2, 1)

#         # refinement layer 
#         self.refine = nn.Sequential(
#             nn.Conv2d(mid_c, mid_c, 3, padding=1, groups=1),
#             nn.GELU(),
#             nn.Conv2d(mid_c, mid_c, 3, padding=1, groups=1),
#             nn.GELU(),
#             )

#         native_hw = (self.Hp * 8, self.Wp * 8)  # after up1+up2+up3
#         if out_hw is None or tuple(out_hw) == native_hw:
#             self.final_up = nn.Identity()
#         else:
#             self.final_up = nn.Upsample(size=out_hw, mode="bilinear", align_corners=False)

#         self.hm = nn.Conv2d(mid_c, num_kp, 1)

#     def forward(self, patch_tokens):
#         B, T, P, D = patch_tokens.shape
#         BT = B * T
#         x = self.proj(patch_tokens).view(BT, P, -1).transpose(1, 2).reshape(BT, -1, self.Hp, self.Wp)
#         x = self.act(self.conv(x))
#         x = self.act(self.up1(x))
#         x = self.act(self.up2(x))
#         x = self.act(self.up3(x))
#         x = self.final_up(x)
#         x = self.refine(x)
#         x = self.hm(x)
#         return x.view(B, T, x.shape[1], x.shape[-2], x.shape[-1])
    
# ---------------- Heatmap head ----------------
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

        native_hw = (self.Hp * 8, self.Wp * 8)  # after up1+up2+up3
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

# ---------------- Full model ----------------
class SpacecraftHMDSNT(nn.Module):
    def __init__(
        self,
        num_kp=8,
        num_frames=16,
        img_hw=224,
        hf_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        adapter_bottleneck=64,
        adapter_drop=0.1,
        adapter_in_last=None,
        heatmap_hw=(224, 224),
        grid_hw=(14, 14),
        n_register_tokens: int | None = None,  # None => infer at runtime
        ks=3, kt=3, fuse="sum",
        hm_mid_c=128,
    ):
        super().__init__()
        self.num_kp = num_kp
        self.num_frames = num_frames
        self.img_hw = img_hw
        self.grid_hw = grid_hw
        self.P = grid_hw[0] * grid_hw[1]
        self.n_register_tokens = n_register_tokens  # can be None

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
        self.hm_head = TinyHeatmapHead(d, grid_hw=grid_hw, num_kp=num_kp, out_hw=heatmap_hw, mid_c=hm_mid_c)

    def forward(self, video):
        # video: (B,T,C,H,W)
        B, T, C, H, W = video.shape
        assert T == self.num_frames, f"Got T={T}, expected num_frames={self.num_frames}"
        x = video.view(B * T, C, H, W)

        tok = self.backbone.forward_tokens(x)  # (BT,N,D)
        BT, N, D = tok.shape
        assert BT == B * T

        # infer register tokens if needed
        nreg = self.n_register_tokens
        if nreg is None:
            nreg = N - 1 - self.P
            assert nreg >= 0, f"Cannot infer n_register_tokens: N={N}, P={self.P}"

        tok_cls = tok[:, 0:1, :]  # (BT,1,D)

        # final backbone patch tokens (post all blocks)
        tok_patch = tok[:, 1 + nreg: 1 + nreg + self.P, :]  # (BT,P,D)
        tok_patch_btpd = tok_patch.view(B, T, self.P, D)

        # adapter tokens for alignment loss (pre-MHSA tokens from the last active adapter)
        adapter_patch_bt = self.backbone.get_last_adapter_patch_bt()  # (BT,P,D) or None
        if adapter_patch_bt is None:
            # fallback: use final patch tokens if adapters disabled or not captured
            adapter_patch_bt = tok_patch
        adapter_tokens_bt = adapter_patch_bt.view(B, T, self.P, D)

        # all adapter layer tokens for layer-wise diagnostics
        all_adapter_list = self.backbone.get_all_adapter_patches_bt()
        # list of (layer_idx, (B, T, P, D)) tuples
        all_adapter_tokens = [
            (layer_idx, pat.view(B, T, self.P, D))
            for layer_idx, pat in all_adapter_list
        ]

        # heatmaps / dsnt
        hm_logits = self.hm_head(tok_patch_btpd)        # (B,T,K,H,W)
        hm_prob = _softmax_2d(hm_logits)
        kps = dsnt(hm_prob)                              # (B,T,K,2)

        return {
            # "hm_logits": hm_logits,              # logits
            "heatmaps": hm_logits,               # alias (if your trainer expects "heatmaps")
            "hm_prob": hm_prob,                  # normalized heatmaps
            "keypoints": kps,                    # DSNT coords
            "itoken": tok_cls.view(B, T, D),     # CLS token per frame (optional)
            "adapter_tokens": adapter_tokens_bt, # (B,T,P,D) for token alignment loss
            "all_adapter_tokens": all_adapter_tokens,  # [(layer_idx, (B,T,P,D)), ...]
        }
# ---------------- Simple test main ----------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = 1
    T = 8
    C = 3
    H = W = 512
    num_kp = 8

    # For ViT-*/16 at 512 => grid=32
    grid_hw = (32, 32)

    model = SpacecraftHMDSNT(
        num_kp=num_kp,
        num_frames=T,
        img_hw=H,
        hf_name="facebook/dinov3-vitb16-pretrain-lvd1689m",  # switch to vitb16 below if desired
        # hf_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
        adapter_bottleneck=64,
        adapter_drop=0.1,
        adapter_in_last=4,
        heatmap_hw=(512, 512),
        grid_hw=grid_hw,
        n_register_tokens=None,  # AUTO-infer (recommended)
        ks=3,
        kt=3,
        fuse="sum",
    ).to(device)

    model.eval()

    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")
    exit()
    video = torch.randn(B, T, C, H, W, device=device)

    with torch.no_grad():
        out = model(video)

    print("heatmaps:", out["heatmaps"].shape)   # (B,T,K,512,512)
    print("keypoints:", out["keypoints"].shape) # (B,T,K,2)
    print("itoken:", out["itoken"].shape)       # (B,T,D)

def probe_register_tokens(
    hf_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
    img_hw=224,
    patch_size=16,
    device=None,
    ):
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Grid from image/patch (for ViT-*/16: 224->14, 512->32)
    Hp = img_hw // patch_size
    Wp = img_hw // patch_size
    P = Hp * Wp

    model = DINOv3ViTModel.from_pretrained(hf_name).to(device).eval()

    x = torch.randn(1, 3, img_hw, img_hw, device=device)  # (BT=1,C,H,W)

    with torch.no_grad():
        out = model(pixel_values=x, return_dict=True)
        tok = out.last_hidden_state  # (BT, N, D)

    BT, N, D = tok.shape
    inferred_nreg = N - 1 - P

    print("=" * 80)
    print(f"hf_name          : {hf_name}")
    print(f"input            : {img_hw}x{img_hw}, patch_size={patch_size}")
    print(f"grid_hw          : ({Hp},{Wp}) -> P={P}")
    print(f"tokens           : N={N}, D={D}")
    print(f"expected (no reg): 1 + P = {1 + P}")
    print(f"inferred_nreg    : N - 1 - P = {inferred_nreg}")

    if inferred_nreg == 0:
        print("=> No extra tokens beyond CLS+PATCH. (n_register_tokens = 0)")
    elif inferred_nreg > 0:
        print(f"=> There ARE extra tokens between CLS and PATCH. (n_register_tokens = {inferred_nreg})")
    else:
        print("=> ERROR: inferred_nreg < 0. Your grid_hw / patch_size / img_hw assumptions are wrong.")
        print("   Most common cause: wrong patch_size or wrong grid_hw for your input resolution.")

    # Optional: try to read from config if present (safe introspection, no assumptions)
    cfg = model.config
    for attr in ["num_register_tokens", "n_register_tokens", "register_tokens", "use_register_tokens"]:
        if hasattr(cfg, attr):
            print(f"config.{attr} = {getattr(cfg, attr)}")

    print("=" * 80)



if __name__ == "__main__":
    main()
    # probe_register_tokens(
    #     hf_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
    #     img_hw=224,
    #     patch_size=16,
    # )
    # probe_register_tokens(
    #     hf_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
    #     img_hw=512,
    #     patch_size=16,
    # )
# if __name__ == "__main__":
#     main()