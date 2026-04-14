# model_hm_dsnt.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DINOv3ViTModel
import dsntnn

# ========================== DSNT utilities ==========================
def _softmax_2d(hm, eps=1e-8):
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
    px = hm_prob.sum(dim=-2)
    py = hm_prob.sum(dim=-1)
    x = (px * xs).sum(dim=-1)
    y = (py * ys).sum(dim=-1)
    return torch.stack([x, y], dim=-1)


# ========================== Core 3D Adapter ==========================
class SpatioTemporal3DAdapter(nn.Module):
    """
    Patch-token adapter with factorized 3D CNN branches:
      - spatial:  (1, ks, ks)
      - temporal: (kt, 1, 1)
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
        fuse: str = "sum",
    ):
        super().__init__()
        self.Hp, self.Wp = grid_hw
        self.P = self.Hp * self.Wp
        self.fuse = fuse

        self.ln = nn.LayerNorm(d) if use_ln else nn.Identity()
        self.in_proj = nn.Linear(d, bottleneck)

        self.spatial_conv = nn.Conv3d(
            bottleneck, bottleneck, kernel_size=(1, ks, ks),
            padding=(0, ks // 2, ks // 2), bias=False
        )
        self.temporal_conv = nn.Conv3d(
            bottleneck, bottleneck, kernel_size=(kt, 1, 1),
            padding=(kt // 2, 0, 0), bias=False
        )

        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

        self.fuse_proj = (
            nn.Conv3d(2 * bottleneck, bottleneck, kernel_size=1, bias=False)
            if fuse == "concat" else None
        )
        self.out_proj = nn.Linear(bottleneck, d)
        self.scale = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))

    def forward(self, x_patch: torch.Tensor) -> torch.Tensor:
        B, T, P, D = x_patch.shape
        assert P == self.P

        y = self.ln(x_patch)
        y = self.in_proj(y)
        C = y.shape[-1]

        y = y.view(B, T, self.Hp, self.Wp, C).permute(0, 4, 1, 2, 3).contiguous()

        ys = self.spatial_conv(y)
        yt = self.temporal_conv(y)

        if self.fuse == "concat":
            z = self.fuse_proj(torch.cat([ys, yt], dim=1))
        else:
            z = ys + yt

        z = self.act(z)
        z = self.drop(z)
        z = z.permute(0, 2, 3, 4, 1).contiguous().view(B, T, P, C)
        z = self.out_proj(z)

        return x_patch + self.scale * z


# ========================== Token-level wrappers ==========================
class PatchOnlyAdapter(nn.Module):
    """
    Operates on flattened tokens (BT,N,D).
    Applies SpatioTemporal3DAdapter ONLY to patch tokens.
    CLS / register tokens pass through unchanged.

    When mode="residual" (default): returns adapted full sequence  (x + δ on patches)
    When mode="delta":              returns delta only              (0 on prefix/suffix, δ on patches)
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
        output_mode: str = "residual",  # "residual" or "delta"
    ):
        super().__init__()
        self.T = T
        self.n_register_tokens = n_register_tokens
        self.Hp, self.Wp = grid_hw
        self.P = self.Hp * self.Wp
        self.output_mode = output_mode

        self.patch_adapter = SpatioTemporal3DAdapter(
            d=d, grid_hw=grid_hw, bottleneck=bottleneck, drop=drop,
            use_ln=use_ln, scale_init=scale_init, ks=ks, kt=kt, fuse=fuse
        )

        self.last_patch_bt = None  # (BT,P,D) stored each forward for alignment loss

    def forward(self, x_btnd: torch.Tensor) -> torch.Tensor:
        BT, N, D = x_btnd.shape
        assert BT % self.T == 0
        B = BT // self.T

        nreg = self.n_register_tokens
        if nreg is None:
            nreg = N - 1 - self.P
            assert nreg >= 0, f"Cannot infer n_register_tokens: N={N}, P={self.P}"

        n_prefix = 1 + nreg
        assert N >= n_prefix + self.P

        prefix = x_btnd[:, :n_prefix, :]
        patch  = x_btnd[:, n_prefix:n_prefix + self.P, :]
        suffix = x_btnd[:, n_prefix + self.P:, :]

        patch_4d = patch.view(B, self.T, self.P, D)
        adapted  = self.patch_adapter(patch_4d)         # (B,T,P,D)  =  patch + scale*δ
        patch_bt = adapted.view(BT, self.P, D)

        self.last_patch_bt = patch_bt  # keep grad for alignment loss

        if self.output_mode == "delta":
            # Return pure delta: zeros on prefix/suffix, adapter contribution on patches
            delta_patch = patch_bt - patch                              # (BT,P,D)
            zeros_pre  = torch.zeros_like(prefix)
            zeros_suf  = torch.zeros_like(suffix)
            return torch.cat([zeros_pre, delta_patch, zeros_suf], dim=1)
        else:
            # Return full adapted sequence
            return torch.cat([prefix, patch_bt, suffix], dim=1)


# ========================== HF module discovery ==========================
def _find_mlp_module(block: nn.Module):
    for name in ["mlp", "ffn", "feed_forward", "feedforward"]:
        if hasattr(block, name):
            return getattr(block, name)
    for n, m in block.named_children():
        if "mlp" in n.lower() or "ffn" in n.lower() or "feed" in n.lower():
            return m
    return None

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
        ("encoder", "layer"), ("encoder", "layers"),
        ("vision_model", "encoder", "layer"), ("vision_model", "encoder", "layers"),
        ("backbone", "encoder", "layer"), ("backbone", "encoder", "layers"),
        ("layers",), ("layer",),
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

    raise AttributeError("Could not find transformer blocks in DINOv3ViTModel.")


# ========================== Backbone with configurable adapter placement ==========================

# Valid adapter_mode values:
#   "pre_mhsa"          — adapter before attention only              (ablation A)
#   "post_mhsa"         — adapter after attention only               (ablation B)
#   "ffn"               — adapter parallel to FFN only               (ablation C)
#   "pre_mhsa+ffn"      — both pre-MHSA and parallel-FFN            (ablation D)
#   "post_mhsa+ffn"     — both post-MHSA and parallel-FFN           (ablation E)

VALID_ADAPTER_MODES = {"pre_mhsa", "post_mhsa", "ffn", "pre_mhsa+ffn", "post_mhsa+ffn"}

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
        adapter_mode: str = "pre_mhsa",
    ):
        super().__init__()
        assert adapter_mode in VALID_ADAPTER_MODES, (
            f"adapter_mode={adapter_mode!r} not in {VALID_ADAPTER_MODES}"
        )
        self.adapter_mode = adapter_mode
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

        # --- Decide which adapter lists to build ---
        use_attn_adapter = "mhsa" in adapter_mode   # True for pre_mhsa, post_mhsa, pre_mhsa+ffn, post_mhsa+ffn
        use_ffn_adapter  = "ffn" in adapter_mode     # True for ffn, pre_mhsa+ffn, post_mhsa+ffn
        attn_is_pre      = adapter_mode.startswith("pre")  # True for pre_mhsa, pre_mhsa+ffn

        # ---- Attention adapters (pre or post MHSA) ----
        # Pre-MHSA: input is post-LN1 → use_ln=False (already normalized)
        # Post-MHSA: input is attention output (NOT normalized) → use_ln=True
        self.attn_adapters = nn.ModuleList()
        if use_attn_adapter:
            attn_use_ln = not attn_is_pre  # False for pre, True for post
            attn_output_mode = "residual"  # both pre and post replace the full sequence
            for i in range(n_blocks):
                if i in use_idx:
                    self.attn_adapters.append(
                        PatchOnlyAdapter(
                            d=d, grid_hw=grid_hw, T=num_frames,
                            n_register_tokens=n_register_tokens,
                            bottleneck=adapter_bottleneck, drop=adapter_drop,
                            use_ln=attn_use_ln,
                            scale_init=1e-3,
                            ks=ks, kt=kt, fuse=fuse,
                            output_mode=attn_output_mode,
                        )
                    )
                else:
                    self.attn_adapters.append(nn.Identity())

        # ---- FFN adapters (parallel to MLP, returns delta) ----
        # Input to MLP hook is post-LN2 → use_ln=False (already normalized by block's LN2)
        self.ffn_adapters = nn.ModuleList()
        if use_ffn_adapter:
            for i in range(n_blocks):
                if i in use_idx:
                    self.ffn_adapters.append(
                        PatchOnlyAdapter(
                            d=d, grid_hw=grid_hw, T=num_frames,
                            n_register_tokens=n_register_tokens,
                            bottleneck=adapter_bottleneck, drop=adapter_drop,
                            use_ln=False,  # input to MLP is post-LN2
                            scale_init=1e-3,
                            ks=ks, kt=kt, fuse=fuse,
                            output_mode="delta",  # only return the delta for additive hook
                        )
                    )
                else:
                    self.ffn_adapters.append(nn.Identity())

        self._install_hooks(blocks, use_attn_adapter, use_ffn_adapter, attn_is_pre)

    def _install_hooks(self, blocks, use_attn_adapter, use_ffn_adapter, attn_is_pre):
        for i, blk in enumerate(blocks):

            # ---------- Attention adapter ----------
            if use_attn_adapter:
                attn = _find_attention_module(blk)
                if attn is None:
                    raise RuntimeError(f"Could not locate attention module in block {i}.")
                adapter = self.attn_adapters[i]

                if attn_is_pre:
                    # PRE-MHSA: modify input to attention
                    def make_pre_hook(adapter_i):
                        def hook(module, inputs):
                            x = inputs[0]
                            x2 = adapter_i(x)
                            return (x2,) + inputs[1:]
                        return hook
                    attn.register_forward_pre_hook(make_pre_hook(adapter), with_kwargs=False)
                else:
                    # POST-MHSA: modify output of attention
                    def make_post_attn_hook(adapter_i):
                        def hook(module, inputs, output):
                            # output shape depends on HF impl:
                            #   could be (tensor,) tuple or just tensor
                            if isinstance(output, tuple):
                                x_out = output[0]       # (BT,N,D)
                                x_adapted = adapter_i(x_out)
                                return (x_adapted,) + output[1:]
                            else:
                                return adapter_i(output)
                        return hook
                    attn.register_forward_hook(make_post_attn_hook(adapter))

            # ---------- FFN adapter ----------
            if use_ffn_adapter:
                mlp = _find_mlp_module(blk)
                if mlp is None:
                    raise RuntimeError(f"Could not locate MLP/FFN module in block {i}.")
                ffn_adapter = self.ffn_adapters[i]

                def make_ffn_hook(ffn_adapter_i):
                    def hook(module, inputs, output):
                        x_in = inputs[0]        # (BT,N,D) — post-LN2 input to MLP
                        delta = ffn_adapter_i(x_in)  # (BT,N,D) — delta only
                        return output + delta
                    return hook
                mlp.register_forward_hook(make_ffn_hook(ffn_adapter))

    def get_last_adapter_patch_bt(self):
        """Returns last active adapter's stored patch tokens (BT,P,D), or None.
        Checks attn_adapters first (full residual representation), then falls back
        to ffn_adapters so ffn-only adapter_mode is also covered.
        """
        for ad in reversed(self.attn_adapters):
            if isinstance(ad, PatchOnlyAdapter):
                if getattr(ad, "last_patch_bt", None) is not None:
                    return ad.last_patch_bt
        for ad in reversed(self.ffn_adapters):
            if isinstance(ad, PatchOnlyAdapter):
                if getattr(ad, "last_patch_bt", None) is not None:
                    return ad.last_patch_bt
        return None

    def forward_tokens(self, x):
        out = self.backbone(pixel_values=x, return_dict=True)
        return out.last_hidden_state


# ========================== Heatmap Heads ==========================

class TemporalHeatmapHead(nn.Module):
    def __init__(self, d, grid_hw=(14, 14), num_kp=8, out_hw=(224, 224),
                 mid_c=128, temporal_kernel=3):
        super().__init__()
        self.Hp, self.Wp = grid_hw
        self.num_kp = num_kp

        self.proj = nn.Linear(d, mid_c)
        self.conv = nn.Conv2d(mid_c, mid_c, 3, padding=1)
        self.act = nn.GELU()

        self.up1 = nn.ConvTranspose2d(mid_c, mid_c, 4, 2, 1)

        self.temporal_refine = nn.Sequential(
            nn.Conv3d(mid_c, mid_c,
                      kernel_size=(temporal_kernel, 1, 1),
                      padding=(temporal_kernel // 2, 0, 0),
                      groups=mid_c, bias=False),
            nn.Conv3d(mid_c, mid_c, kernel_size=1, bias=False),
            nn.GELU(),
        )
        self.temporal_gate = nn.Parameter(torch.zeros(1))

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

        x = self.proj(patch_tokens).view(BT, P, -1)
        C = x.shape[-1]
        x = x.transpose(1, 2).reshape(BT, C, self.Hp, self.Wp)
        x = self.act(self.conv(x))
        x = self.act(self.up1(x))

        _, C, H_mid, W_mid = x.shape
        x_5d = x.view(B, T, C, H_mid, W_mid).permute(0, 2, 1, 3, 4)
        x_temp = self.temporal_refine(x_5d)
        x_5d = x_5d + torch.sigmoid(self.temporal_gate) * x_temp
        x = x_5d.permute(0, 2, 1, 3, 4).reshape(BT, C, H_mid, W_mid)

        x = self.act(self.up2(x))
        x = self.act(self.up3(x))
        x = self.final_up(x)
        x = self.hm(x)
        return x.view(B, T, self.num_kp, x.shape[-2], x.shape[-1])


# ========================== Full Model ==========================
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
        n_register_tokens: int | None = None,
        ks=3, kt=3, fuse="sum",
        hm_mid_c=128,
        temporal_kernel=3,
        adapter_mode: str = "pre_mhsa",
    ):
        super().__init__()
        self.num_kp = num_kp
        self.num_frames = num_frames
        self.img_hw = img_hw
        self.grid_hw = grid_hw
        self.P = grid_hw[0] * grid_hw[1]
        self.n_register_tokens = n_register_tokens
        self.adapter_mode = adapter_mode

        self.backbone = DINOv3backbone(
            hf_name=hf_name,
            adapter_bottleneck=adapter_bottleneck,
            adapter_drop=adapter_drop,
            adapter_in_last=adapter_in_last,
            num_frames=num_frames,
            grid_hw=grid_hw,
            n_register_tokens=n_register_tokens,
            ks=ks, kt=kt, fuse=fuse,
            adapter_mode=adapter_mode,
        )

        d = self.backbone.backbone.config.hidden_size
        self.hm_head = TemporalHeatmapHead(
            d, grid_hw=grid_hw, num_kp=num_kp,
            out_hw=heatmap_hw, mid_c=hm_mid_c,
            temporal_kernel=temporal_kernel,
        )

    def forward(self, video):
        B, T, C, H, W = video.shape
        assert T == self.num_frames
        x = video.view(B * T, C, H, W)

        tok = self.backbone.forward_tokens(x)
        BT, N, D = tok.shape
        assert BT == B * T

        nreg = self.n_register_tokens
        if nreg is None:
            nreg = N - 1 - self.P
            assert nreg >= 0

        tok_cls   = tok[:, 0:1, :]
        tok_patch = tok[:, 1 + nreg: 1 + nreg + self.P, :]
        tok_patch_btpd = tok_patch.view(B, T, self.P, D)

        hm_logits = self.hm_head(tok_patch_btpd)
        hm_prob   = _softmax_2d(hm_logits)
        kps       = dsnt(hm_prob)

        # Direct adapter-layer tokens for alignment loss: (BT,P,D) → (B,T,P,D).
        # Falls back to final backbone tokens if no adapter captured (e.g. Identity blocks only).
        adapter_bt = self.backbone.get_last_adapter_patch_bt()
        adapter_layer_tok = adapter_bt.view(B, T, self.P, D) if adapter_bt is not None else tok_patch_btpd

        return {
            "heatmaps": hm_logits,
            "hm_prob": hm_prob,
            "keypoints": kps,
            "itoken": tok_cls.view(B, T, D),
            "adapter_tokens": tok_patch_btpd,
            "adapter_layer_tokens": adapter_layer_tok,
        }


# ========================== Test / Utility ==========================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, T, C, H, W, num_kp = 1, 8, 3, 512, 512, 8
    grid_hw = (32, 32)

    # ---- Test each ablation config ----
    for mode in VALID_ADAPTER_MODES:
        print(f"\n{'='*60}")
        print(f"  adapter_mode = {mode!r}")
        print(f"{'='*60}")

        model = SpacecraftHMDSNT(
            num_kp=num_kp,
            num_frames=T,
            img_hw=H,
            hf_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
            adapter_bottleneck=64,
            adapter_drop=0.1,
            adapter_in_last=4,
            heatmap_hw=(512, 512),
            grid_hw=grid_hw,
            n_register_tokens=None,
            ks=3, kt=3, fuse="sum",
            adapter_mode=mode,
        ).to(device)

        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"  trainable: {n_train:,}  /  total: {n_total:,}  "
              f"({100*n_train/n_total:.2f}%)")

        # Quick forward test
        model.eval()
        video = torch.randn(B, T, C, H, W, device=device)
        with torch.no_grad():
            out = model(video)
        print(f"  heatmaps:  {out['heatmaps'].shape}")
        print(f"  keypoints: {out['keypoints'].shape}")

        del model
        torch.cuda.empty_cache() if device == "cuda" else None


def probe_register_tokens(
    hf_name="facebook/dinov3-vitb16-pretrain-lvd1689m",
    img_hw=224,
    patch_size=16,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Hp = img_hw // patch_size
    Wp = img_hw // patch_size
    P = Hp * Wp

    model = DINOv3ViTModel.from_pretrained(hf_name).to(device).eval()
    x = torch.randn(1, 3, img_hw, img_hw, device=device)

    with torch.no_grad():
        tok = model(pixel_values=x, return_dict=True).last_hidden_state

    BT, N, D = tok.shape
    inferred_nreg = N - 1 - P

    print("=" * 80)
    print(f"hf_name       : {hf_name}")
    print(f"input         : {img_hw}x{img_hw}, patch_size={patch_size}")
    print(f"grid_hw       : ({Hp},{Wp}) -> P={P}")
    print(f"tokens        : N={N}, D={D}")
    print(f"inferred_nreg : {inferred_nreg}")
    print("=" * 80)


if __name__ == "__main__":
    main()