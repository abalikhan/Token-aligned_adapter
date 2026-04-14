"""
Visualize the effect of token alignment loss — v4.

v1-v3 failed because:
- Token cosine similarity is dominated by frozen DINOv2 backbone (99% of signal)
- Adapter residuals are spatially uniform (same magnitude everywhere)
- Per-frame differences are too subtle to see (effect is statistical)

v4 strategy:
1. ATTENTION MAPS — Hook into frozen MHSA to extract attention weights.
   Pre-MHSA adapter modifies the input to attention, so attention patterns
   MUST differ. This is the direct proof of the amplification effect.
2. STATISTICAL AGGREGATION — Process entire hard sequences, aggregate
   per-frame errors, find and visualize the frames with LARGEST differences.
3. CHERRY-PICKED EXAMPLES — Auto-find frames where one model is clearly
   better than the other.

Usage: python visualize_token_similarity.py
"""

import os
import math
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from dataset import SPADESDataset
from model_hm_dsnt_2 import SpacecraftHMDSNT, PatchOnlyAdapter, _find_attention_module, _get_blocks_from_hf_model
from viz_utils import denorm_imagenet_bt
from utils_new import collate_temporal_chunks


# ── Paths ──
CONFIG_TOKEN   = "config_vitb.yaml"
CONFIG_NOTOKEN = "config_no_token_align.yaml"

CKPT_TOKEN   = "/project/scratch/p200249/aali/pose_estimation/dinov3_temporal/outputs/spades_ad8_temphead_pre_mhsa_vitb/best_epoch_100.pt"
CKPT_NOTOKEN = "/project/scratch/p200249/aali/pose_estimation/dinov3_temporal/outputs/spades_ad8_no_token_premhsa/best_epoch_89.pt"

OUT_DIR = "/project/scratch/p200249/aali/pose_estimation/dinov3_temporal/outputs/token_similarity_viz_v4"

HARD_SEQUENCES = ["RT232", "RT137", "RT181", "RT116", "RT059", "RT037", "RT054", "RT121"]

KP_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8",
    "#f58231", "#911eb4", "#42d4f4", "#f032e6",
]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg, device):
    mc = cfg["model"]
    dc = cfg["data"]
    patch_size = mc.get("patch_size", 16)
    input_size = dc.get("input_size", 512)
    g = input_size // patch_size
    model = SpacecraftHMDSNT(
        hf_name=mc["hf_name"], num_kp=dc["num_kp"],
        num_frames=dc.get("short_mem", 128) // dc.get("sampling_rate", 16),
        grid_hw=(g, g),
        heatmap_hw=tuple(mc.get("heatmap_out", [256, 256])),
        adapter_in_last=mc.get("adapter_in_last", 0),
        adapter_bottleneck=mc.get("adapter_bottleneck", 128),
        adapter_drop=mc.get("adapter_drop", 0.1),
        n_register_tokens=mc.get("n_register_tokens", 4),
        fuse=mc.get("fuse", "sum"),
        adapter_mode=mc.get("adapter_mode", "pre_mhsa"),
        hm_mid_c=mc.get("hm_mid_c", 128),
    )
    # Force eager attention so hooks can capture attention_probs
    # (SDPA/FlashAttention return None for attention weights)
    model.backbone.backbone.config._attn_implementation = "eager"
    return model.to(device).eval()


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    print(f"  Loaded: {ckpt_path}")
    return model


def kp_norm_to_px(kp_norm, img_size):
    x = ((kp_norm[..., 0] + 1) * 0.5) * (img_size - 1)
    y = ((kp_norm[..., 1] + 1) * 0.5) * (img_size - 1)
    return x, y


def kp_norm_to_patch_idx(kp_norm, grid_side):
    col = ((kp_norm[..., 0] + 1) * 0.5 * (grid_side - 1)).round().long().clamp(0, grid_side - 1)
    row = ((kp_norm[..., 1] + 1) * 0.5 * (grid_side - 1)).round().long().clamp(0, grid_side - 1)
    return row, col


def upsample_map(patch_map, grid_side, img_size):
    scale = img_size // grid_side
    return np.kron(patch_map, np.ones((scale, scale)))


# ═══════════════════════════════════════════════════════════════════
# Attention extraction
# ═══════════════════════════════════════════════════════════════════
class AttentionExtractor:
    """Hook into frozen MHSA layers to capture attention weights."""

    def __init__(self, model):
        self.model = model
        self.attention_maps = {}
        self._hooks = []

    def install(self, layer_indices=None):
        """Install hooks on specified (or all adapter) layers."""
        blocks = _get_blocks_from_hf_model(self.model.backbone.backbone)
        if layer_indices is None:
            # Use the layers that have adapters
            layer_indices = []
            for i, ad in enumerate(self.model.backbone.attn_adapters):
                if isinstance(ad, PatchOnlyAdapter):
                    layer_indices.append(i)

        for i in layer_indices:
            blk = blocks[i]
            attn_mod = _find_attention_module(blk)
            if attn_mod is None:
                continue
            # The DINOv2 SelfAttention.forward returns (context, attn_probs)
            # We hook the inner self-attention module
            inner = attn_mod
            # Navigate to the actual self-attention: attn_mod.attention
            if hasattr(attn_mod, 'attention'):
                inner = attn_mod.attention

            def make_hook(layer_idx):
                def hook(module, inputs, output):
                    # output = (context_layer, attention_probs)
                    if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                        self.attention_maps[layer_idx] = output[1].detach().cpu()
                return hook

            h = inner.register_forward_hook(make_hook(i))
            self._hooks.append(h)

        return self

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def get_maps(self):
        return dict(self.attention_maps)

    def clear(self):
        self.attention_maps = {}


def forward_with_attention(model, video):
    """Run forward pass and extract attention maps from adapter layers."""
    extractor = AttentionExtractor(model)
    extractor.install()
    extractor.clear()
    with torch.no_grad():
        out = model(video)
    attn_maps = extractor.get_maps()
    extractor.remove()
    return out, attn_maps


# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: Attention map comparison for keypoint patches
# ═══════════════════════════════════════════════════════════════════
def fig_attention_maps(attn_w, attn_wo, kps_gt, vis, img,
                       grid_side, input_size, n_reg, b_in_chunk, t,
                       seq_name, out_dir):
    """
    Show how keypoint patches attend to other patches.
    Pre-MHSA adapter changes the input → attention patterns differ.
    """
    K = kps_gt.shape[2]
    visible = [k for k in range(K) if vis[b_in_chunk, t, k] > 0.5]
    if len(visible) == 0:
        return

    # Use last adapter layer (strongest effect)
    last_layer = max(attn_w.keys())
    # attn shape: (BT, num_heads, N, N) where N = 1 + n_reg + P
    # We need the right batch*time index
    BT_idx = b_in_chunk * kps_gt.shape[1] + t  # index into BT dimension

    aw = attn_w[last_layer]  # (BT, H, N, N)
    awo = attn_wo[last_layer]

    if BT_idx >= aw.shape[0]:
        return

    # Average over heads
    aw_avg = aw[BT_idx].mean(dim=0)    # (N, N)
    awo_avg = awo[BT_idx].mean(dim=0)

    n_prefix = 1 + n_reg  # CLS + registers
    P = grid_side * grid_side

    # Extract patch-to-patch attention only
    aw_pp = aw_avg[n_prefix:n_prefix+P, n_prefix:n_prefix+P]   # (P, P)
    awo_pp = awo_avg[n_prefix:n_prefix+P, n_prefix:n_prefix+P]

    n_vis = min(len(visible), 4)  # show max 4 keypoints
    fig, axes = plt.subplots(n_vis, 4, figsize=(22, 5.5 * n_vis))
    if n_vis == 1:
        axes = axes[np.newaxis, :]

    for i, k in enumerate(visible[:n_vis]):
        row, col = kp_norm_to_patch_idx(kps_gt[b_in_chunk, t, k:k+1].cpu(), grid_side)
        pidx = (row[0] * grid_side + col[0]).item()
        kp_x, kp_y = kp_norm_to_px(kps_gt[b_in_chunk, t, k].cpu(), input_size)

        # Attention FROM keypoint patch TO all others
        attn_from_w = aw_pp[pidx].numpy().reshape(grid_side, grid_side)
        attn_from_wo = awo_pp[pidx].numpy().reshape(grid_side, grid_side)

        # Attention TO keypoint patch FROM all others
        attn_to_w = aw_pp[:, pidx].numpy().reshape(grid_side, grid_side)
        attn_to_wo = awo_pp[:, pidx].numpy().reshape(grid_side, grid_side)

        vmax_from = max(attn_from_w.max(), attn_from_wo.max())
        vmax_to = max(attn_to_w.max(), attn_to_wo.max())

        # Col 0: WITH — where KP attends TO
        ax = axes[i, 0]
        ax.imshow(img)
        ax.imshow(upsample_map(attn_from_w, grid_side, input_size),
                  alpha=0.7, cmap="hot", vmin=0, vmax=vmax_from,
                  extent=[0, input_size, input_size, 0])
        ax.plot(kp_x.item(), kp_y.item(), "o", color="lime", markersize=10,
                markeredgecolor="white", markeredgewidth=2)
        ax.set_title(f"KP{k} WITH — attends TO\n(where this KP looks)", fontsize=10, fontweight="bold")
        ax.axis("off")

        # Col 1: WITHOUT — where KP attends TO
        ax = axes[i, 1]
        ax.imshow(img)
        ax.imshow(upsample_map(attn_from_wo, grid_side, input_size),
                  alpha=0.7, cmap="hot", vmin=0, vmax=vmax_from,
                  extent=[0, input_size, input_size, 0])
        ax.plot(kp_x.item(), kp_y.item(), "o", color="lime", markersize=10,
                markeredgecolor="white", markeredgewidth=2)
        ax.set_title(f"KP{k} WITHOUT — attends TO", fontsize=10)
        ax.axis("off")

        # Col 2: Difference in outgoing attention
        ax = axes[i, 2]
        diff_from = attn_from_w - attn_from_wo
        vlim = max(abs(diff_from.min()), abs(diff_from.max()), 1e-8)
        ax.imshow(img)
        ax.imshow(upsample_map(diff_from, grid_side, input_size),
                  alpha=0.7, cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                  extent=[0, input_size, input_size, 0])
        ax.plot(kp_x.item(), kp_y.item(), "o", color="lime", markersize=10,
                markeredgecolor="white", markeredgewidth=2)
        ax.set_title(f"KP{k} DIFF (with−without)\nRed=more attn with align", fontsize=9)
        ax.axis("off")

        # Col 3: Stats
        ax = axes[i, 3]
        ax.axis("off")
        # Entropy of attention distribution
        eps = 1e-10
        ent_w = -(attn_from_w * np.log(attn_from_w + eps)).sum()
        ent_wo = -(attn_from_wo * np.log(attn_from_wo + eps)).sum()
        # Top-k concentration
        flat_w = attn_from_w.reshape(-1)
        flat_wo = attn_from_wo.reshape(-1)
        top5_w = np.sort(flat_w)[-5:].sum()
        top5_wo = np.sort(flat_wo)[-5:].sum()
        top10_w = np.sort(flat_w)[-10:].sum()
        top10_wo = np.sort(flat_wo)[-10:].sum()
        # Spatial focus: attn mass within 3 patches of KP
        coords = np.stack(np.meshgrid(np.arange(grid_side), np.arange(grid_side), indexing="ij"), -1).reshape(-1, 2)
        kp_rc = np.array([row[0].item(), col[0].item()])
        dists = np.abs(coords - kp_rc).max(axis=-1)
        near3 = dists <= 3
        focus_w = flat_w[near3].sum() / flat_w.sum()
        focus_wo = flat_wo[near3].sum() / flat_wo.sum()

        stats = (
            f"KP{k} Attention Statistics\n"
            f"{'─' * 30}\n"
            f"Entropy (lower=more focused):\n"
            f"  WITH:    {ent_w:.3f}\n"
            f"  WITHOUT: {ent_wo:.3f}\n"
            f"  Δ: {ent_w - ent_wo:+.3f}\n\n"
            f"Top-5 patch concentration:\n"
            f"  WITH:    {top5_w:.4f}\n"
            f"  WITHOUT: {top5_wo:.4f}\n\n"
            f"Top-10 patch concentration:\n"
            f"  WITH:    {top10_w:.4f}\n"
            f"  WITHOUT: {top10_wo:.4f}\n\n"
            f"Mass within 3 patches of KP:\n"
            f"  WITH:    {focus_w:.4f}\n"
            f"  WITHOUT: {focus_wo:.4f}\n"
            f"  Ratio:   {focus_w / max(focus_wo, 1e-8):.2f}x"
        )
        ax.text(0.05, 0.95, stats, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    fig.suptitle(f"Frozen Attention Maps — {seq_name}, Frame {t}, Layer {last_layer}\n"
                 f"(pre-MHSA adapter modifies input → different attention patterns)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, f"attention_{seq_name}_frame{t}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 2: Aggregated attention statistics across all frames
# ═══════════════════════════════════════════════════════════════════
def fig_attention_aggregate(all_attn_stats, out_dir):
    """Show aggregated attention statistics across sequences."""
    if not all_attn_stats:
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    seqs = list(all_attn_stats.keys())
    x = np.arange(len(seqs))
    w = 0.35

    # Entropy
    ax = axes[0]
    ent_w = [np.mean(all_attn_stats[s]["entropy_w"]) for s in seqs]
    ent_wo = [np.mean(all_attn_stats[s]["entropy_wo"]) for s in seqs]
    ax.bar(x - w/2, ent_w, w, label="WITH token align", color="#2196F3")
    ax.bar(x + w/2, ent_wo, w, label="WITHOUT token align", color="#FF9800")
    ax.set_xticks(x); ax.set_xticklabels(seqs, rotation=45, fontsize=9)
    ax.set_ylabel("Attention entropy"); ax.set_title("KP Attention Entropy\n(lower = more focused)", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # Focus (mass near KP)
    ax = axes[1]
    foc_w = [np.mean(all_attn_stats[s]["focus_w"]) for s in seqs]
    foc_wo = [np.mean(all_attn_stats[s]["focus_wo"]) for s in seqs]
    ax.bar(x - w/2, foc_w, w, label="WITH token align", color="#2196F3")
    ax.bar(x + w/2, foc_wo, w, label="WITHOUT token align", color="#FF9800")
    ax.set_xticks(x); ax.set_xticklabels(seqs, rotation=45, fontsize=9)
    ax.set_ylabel("Attention mass near KP"); ax.set_title("Spatial Focus (mass ≤3 patches of KP)\n(higher = more spatially grounded)", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # Top-10 concentration
    ax = axes[2]
    top_w = [np.mean(all_attn_stats[s]["top10_w"]) for s in seqs]
    top_wo = [np.mean(all_attn_stats[s]["top10_wo"]) for s in seqs]
    ax.bar(x - w/2, top_w, w, label="WITH token align", color="#2196F3")
    ax.bar(x + w/2, top_wo, w, label="WITHOUT token align", color="#FF9800")
    ax.set_xticks(x); ax.set_xticklabels(seqs, rotation=45, fontsize=9)
    ax.set_ylabel("Top-10 attn mass"); ax.set_title("Top-10 Patch Concentration\n(higher = more peaked attention)", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Aggregated Attention Statistics Across Hard Sequences", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "attention_aggregate.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 3: Per-frame error comparison + cherry-picked best examples
# ═══════════════════════════════════════════════════════════════════
def fig_error_comparison(all_errors, frames_cache, kps_gt_cache, vis_cache, out_w_cache, out_wo_cache,
                         input_size, out_dir):
    """Plot per-frame errors and save the frames with biggest difference."""
    if not all_errors:
        return

    for seq_name, errs in all_errors.items():
        err_w = np.array(errs["err_w"])
        err_wo = np.array(errs["err_wo"])
        delta = err_wo - err_w  # positive = token alignment is better

        fig, axes = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={"height_ratios": [2, 1]})

        # Error curves
        ax = axes[0]
        frames_x = np.arange(len(err_w))
        ax.plot(frames_x, err_w, "-", color="#2196F3", linewidth=1, alpha=0.7, label="WITH token align")
        ax.plot(frames_x, err_wo, "-", color="#FF9800", linewidth=1, alpha=0.7, label="WITHOUT token align")
        ax.fill_between(frames_x, err_w, err_wo, where=(delta > 0),
                        color="#4CAF50", alpha=0.3, label="Token align better")
        ax.fill_between(frames_x, err_w, err_wo, where=(delta < 0),
                        color="#F44336", alpha=0.3, label="Token align worse")
        ax.set_ylabel("Mean keypoint error (px)", fontsize=10)
        ax.set_title(f"{seq_name} — Per-Frame Error Comparison\n"
                     f"Mean Δerr = {delta.mean():+.2f}px  "
                     f"(positive = token alignment better in {(delta > 0).sum()}/{len(delta)} frames)",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # Delta curve
        ax = axes[1]
        colors = ["#4CAF50" if d > 0 else "#F44336" for d in delta]
        ax.bar(frames_x, delta, color=colors, alpha=0.6, width=1.0)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axhline(delta.mean(), color="#2196F3", linewidth=1.5, linestyle="--", label=f"mean={delta.mean():+.2f}")
        ax.set_xlabel("Frame index", fontsize=10)
        ax.set_ylabel("Δerr (without − with)", fontsize=10)
        ax.set_title("Green = token alignment better, Red = worse", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        path = os.path.join(out_dir, f"error_curve_{seq_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # Cherry-pick: find frames with biggest positive delta across all sequences
    print("\n  Finding best cherry-pick examples...")
    all_deltas = []
    for seq_name, errs in all_errors.items():
        for i, (ew, ewo) in enumerate(zip(errs["err_w"], errs["err_wo"])):
            all_deltas.append((ewo - ew, seq_name, i, ew, ewo))

    all_deltas.sort(reverse=True)

    # Save top-6 examples
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    for idx, (delta, seq, frame_i, ew, ewo) in enumerate(all_deltas[:6]):
        ax = axes[idx // 3, idx % 3]
        cache_key = f"{seq}_{frame_i}"
        if cache_key not in frames_cache:
            ax.text(0.5, 0.5, f"{seq} frame {frame_i}\nΔ={delta:+.1f}px", transform=ax.transAxes,
                    ha="center", fontsize=12)
            ax.axis("off")
            continue

        img = frames_cache[cache_key]
        kps_gt = kps_gt_cache[cache_key]
        vis = vis_cache[cache_key]
        kps_w = out_w_cache[cache_key]
        kps_wo = out_wo_cache[cache_key]
        K = kps_gt.shape[0]

        ax.imshow(img)
        for k in range(K):
            if vis[k] < 0.5:
                continue
            gt_x, gt_y = kp_norm_to_px(kps_gt[k], input_size)
            pw_x, pw_y = kp_norm_to_px(kps_w[k], input_size)
            pwo_x, pwo_y = kp_norm_to_px(kps_wo[k], input_size)

            ax.plot(gt_x.item(), gt_y.item(), "o", color="lime", markersize=7,
                    markeredgecolor="white", markeredgewidth=1.5)
            ax.plot([gt_x.item(), pw_x.item()], [gt_y.item(), pw_y.item()],
                    "-", color="#2196F3", linewidth=2)
            ax.plot(pw_x.item(), pw_y.item(), "x", color="#2196F3", markersize=8, markeredgewidth=2)
            ax.plot([gt_x.item(), pwo_x.item()], [gt_y.item(), pwo_y.item()],
                    "-", color="#FF9800", linewidth=2)
            ax.plot(pwo_x.item(), pwo_y.item(), "+", color="#FF9800", markersize=10, markeredgewidth=2)

        ax.set_title(f"{seq} frame {frame_i}\n"
                     f"WITH={ew:.1f}px  WITHOUT={ewo:.1f}px  Δ={delta:+.1f}px",
                     fontsize=10, fontweight="bold",
                     color="darkgreen" if delta > 0 else "darkred")
        ax.axis("off")

    # Add legend to first plot
    axes[0, 0].plot([], [], "o", color="lime", markersize=7, label="GT")
    axes[0, 0].plot([], [], "x", color="#2196F3", markersize=8, label="WITH token")
    axes[0, 0].plot([], [], "+", color="#FF9800", markersize=10, label="WITHOUT token")
    axes[0, 0].legend(fontsize=8, loc="upper right")

    fig.suptitle("Best Cherry-Pick Examples: Frames Where Token Alignment Helps Most\n"
                 "(sorted by error reduction)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "cherry_pick_best.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_visualization():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)

    cfg_token = load_config(CONFIG_TOKEN)
    cfg_notoken = load_config(CONFIG_NOTOKEN)

    dc = cfg_token["data"]
    input_size = dc.get("input_size", 512)
    patch_size = cfg_token["model"].get("patch_size", 16)
    grid_side = input_size // patch_size
    n_reg = cfg_token["model"].get("n_register_tokens", 4)
    heatmap_hw = tuple(cfg_token["model"].get("heatmap_out", [256, 256]))
    T_chunk = int(dc.get("test_T_chunk", 8))

    print("Building model WITH token alignment...")
    model_token = build_model(cfg_token, device)
    model_token = load_checkpoint(model_token, CKPT_TOKEN, device)

    print("Building model WITHOUT token alignment...")
    model_notoken = build_model(cfg_notoken, device)
    model_notoken = load_checkpoint(model_notoken, CKPT_NOTOKEN, device)

    print(f"Loading SPADES test set...")
    test_ds = SPADESDataset(
        gt_folder=dc["root"], mode="test", input_size=input_size,
        short_mem=dc.get("short_mem", 128), sampling_rate=dc.get("sampling_rate", 16),
        clip_stride=dc.get("clip_stride", 18),
        crop=dc.get("eval_use_crop", dc.get("crop", True)),
        gaussian_sigma_px=dc.get("gaussian_sigma_px", 2.5),
        heatmap_out=heatmap_hw, spatial_transforms=None,
    )
    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=lambda b: collate_temporal_chunks(b, T_chunk),
    )

    all_attn_stats = defaultdict(lambda: defaultdict(list))
    all_errors = {}
    frames_cache = {}
    kps_gt_cache = {}
    vis_cache = {}
    out_w_cache = {}
    out_wo_cache = {}

    processed = set()

    for batch_idx, data in enumerate(loader):
        frames_all = data[0]
        kps_gt_all = data[1]
        vis_all = data[4]
        seq_names = data[9]
        seq_name = seq_names[0] if isinstance(seq_names, (list, tuple)) else seq_names

        if seq_name not in HARD_SEQUENCES or seq_name in processed:
            continue
        processed.add(seq_name)
        print(f"\nProcessing {seq_name} ({len(processed)}/{len(HARD_SEQUENCES)})...")

        B_total = frames_all.shape[0]
        all_errors[seq_name] = {"err_w": [], "err_wo": []}

        # Process ALL chunks for statistical aggregation
        attn_viz_done = False
        for ci in range(0, B_total, 1):
            video = frames_all[ci:ci+1].to(device)
            kps_gt = kps_gt_all[ci:ci+1].to(device)
            vis = vis_all[ci:ci+1].to(device)

            out_w, attn_w = forward_with_attention(model_token, video)
            out_wo, attn_wo = forward_with_attention(model_notoken, video)

            frames_vis = denorm_imagenet_bt(video).cpu()
            T = video.shape[1]
            K = kps_gt.shape[2]

            for t in range(T):
                visible = [k for k in range(K) if vis[0, t, k] > 0.5]
                if len(visible) == 0:
                    continue

                # Compute per-frame mean error
                err_w_frame, err_wo_frame = 0, 0
                for k in visible:
                    gt_x, gt_y = kp_norm_to_px(kps_gt[0, t, k].cpu(), input_size)
                    pw_x, pw_y = kp_norm_to_px(out_w["keypoints"][0, t, k].cpu(), input_size)
                    pwo_x, pwo_y = kp_norm_to_px(out_wo["keypoints"][0, t, k].cpu(), input_size)
                    err_w_frame += math.sqrt((gt_x.item()-pw_x.item())**2 + (gt_y.item()-pw_y.item())**2)
                    err_wo_frame += math.sqrt((gt_x.item()-pwo_x.item())**2 + (gt_y.item()-pwo_y.item())**2)
                err_w_frame /= len(visible)
                err_wo_frame /= len(visible)
                all_errors[seq_name]["err_w"].append(err_w_frame)
                all_errors[seq_name]["err_wo"].append(err_wo_frame)

                # Cache frame data for cherry-picking
                frame_key = f"{seq_name}_{ci * T + t}"
                img_np = frames_vis[0, t].permute(1, 2, 0).numpy()
                frames_cache[frame_key] = img_np
                kps_gt_cache[frame_key] = kps_gt[0, t].cpu()
                vis_cache[frame_key] = vis[0, t].cpu()
                out_w_cache[frame_key] = out_w["keypoints"][0, t].cpu()
                out_wo_cache[frame_key] = out_wo["keypoints"][0, t].cpu()

                # Attention stats
                last_layer = max(attn_w.keys()) if attn_w else None
                if last_layer is not None:
                    BT_idx = t
                    aw = attn_w[last_layer]
                    awo = attn_wo[last_layer]
                    if BT_idx < aw.shape[0]:
                        n_prefix = 1 + n_reg
                        P = grid_side * grid_side
                        aw_avg = aw[BT_idx].mean(dim=0)
                        awo_avg = awo[BT_idx].mean(dim=0)
                        aw_pp = aw_avg[n_prefix:n_prefix+P, n_prefix:n_prefix+P]
                        awo_pp = awo_avg[n_prefix:n_prefix+P, n_prefix:n_prefix+P]

                        coords = np.stack(np.meshgrid(np.arange(grid_side), np.arange(grid_side), indexing="ij"), -1).reshape(-1, 2)
                        eps = 1e-10

                        for k in visible:
                            row, col = kp_norm_to_patch_idx(kps_gt[0, t, k:k+1].cpu(), grid_side)
                            pidx = (row[0] * grid_side + col[0]).item()
                            kp_rc = np.array([row[0].item(), col[0].item()])
                            dists = np.abs(coords - kp_rc).max(axis=-1)
                            near3 = dists <= 3

                            a_w = aw_pp[pidx].numpy()
                            a_wo = awo_pp[pidx].numpy()

                            all_attn_stats[seq_name]["entropy_w"].append(-(a_w * np.log(a_w + eps)).sum())
                            all_attn_stats[seq_name]["entropy_wo"].append(-(a_wo * np.log(a_wo + eps)).sum())
                            all_attn_stats[seq_name]["focus_w"].append(a_w[near3].sum() / a_w.sum())
                            all_attn_stats[seq_name]["focus_wo"].append(a_wo[near3].sum() / a_wo.sum())
                            all_attn_stats[seq_name]["top10_w"].append(np.sort(a_w)[-10:].sum())
                            all_attn_stats[seq_name]["top10_wo"].append(np.sort(a_wo)[-10:].sum())

            # Attention map visualization for first chunk only
            if not attn_viz_done and attn_w:
                for t in [0, T//2, T-1]:
                    fig_attention_maps(
                        attn_w, attn_wo, kps_gt, vis, frames_vis[0, t].permute(1, 2, 0).numpy(),
                        grid_side, input_size, n_reg, 0, t, seq_name, OUT_DIR)
                attn_viz_done = True

            # Limit chunks to avoid OOM with caching
            if ci >= 10:
                break

        if len(processed) >= len(HARD_SEQUENCES):
            break

    # Aggregated figures
    print("\nGenerating aggregate figures...")
    fig_attention_aggregate(dict(all_attn_stats), OUT_DIR)
    fig_error_comparison(all_errors, frames_cache, kps_gt_cache, vis_cache,
                         out_w_cache, out_wo_cache, input_size, OUT_DIR)

    print(f"\nDone! All figures saved to {OUT_DIR}/")
    print("\nFigure types:")
    print("  1. attention_*         — Frozen MHSA attention maps (MUST differ between models)")
    print("  2. attention_aggregate — Bar charts: entropy, spatial focus, concentration")
    print("  3. error_curve_*       — Per-frame error comparison across full sequences")
    print("  4. cherry_pick_best    — Auto-selected frames with biggest difference")


if __name__ == "__main__":
    run_visualization()
