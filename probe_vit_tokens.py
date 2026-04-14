import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import yaml
import cv2
import os

from model import SpacecraftPoseDualPath
from dataset import SPADESDataset


# ============================================================
# Heatmap utilities (FIXED)
# ============================================================

def logits_to_prob_heatmaps(hm_logits: np.ndarray, tau: float = 0.4) -> np.ndarray:
    """
    Convert heatmap logits to probability heatmaps via softmax.

    hm_logits: (K, H, W)
    returns:   (K, H, W)
    """
    K, H, W = hm_logits.shape
    hm = torch.from_numpy(hm_logits).float()
    hm = hm.view(K, -1) / tau
    hm = torch.softmax(hm, dim=-1)
    return hm.view(K, H, W).cpu().numpy()


def normalize_heatmap(hm: np.ndarray) -> np.ndarray:
    """
    Robust per-keypoint normalization using percentile clipping.
    """
    p1, p99 = np.percentile(hm, [1, 99])
    hm = np.clip(hm, p1, p99)
    hm = (hm - hm.min()) / (hm.max() + 1e-6)
    return hm


# ============================================================
# Visualization
# ============================================================

def visualize_heatmap_grid(
    frame: np.ndarray,          # (H,W,3) uint8
    heatmaps: np.ndarray,       # (K,H,W) probabilities
    kps_norm: np.ndarray = None,# (K,2) in [-1,1]
    cols: int = 8,
    alpha: float = 0.4,
    save_path: str = "./debug_vis/debug_full_heatmap_grid.png",
):
    """
    Grid visualization of all keypoint heatmaps.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    K, Hm, Wm = heatmaps.shape
    H, W = frame.shape[:2]
    rows = int(np.ceil(K / cols))

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(2.4 * cols, 2.4 * rows),
        squeeze=False
    )

    for k in range(rows * cols):
        ax = axes[k // cols, k % cols]
        ax.axis("off")

        if k >= K:
            continue

        hm = normalize_heatmap(heatmaps[k])
        hm = (hm * 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_TURBO)
        hm_color = cv2.resize(hm_color, (W, H))

        overlay = cv2.addWeighted(frame, 1 - alpha, hm_color, alpha, 0)

        # draw GT keypoint
        if kps_norm is not None:
            x = int((kps_norm[k, 0] + 1) * 0.5 * (W - 1))
            y = int((kps_norm[k, 1] + 1) * 0.5 * (H - 1))
            cv2.circle(overlay, (x, y), 4, (0, 255, 0), -1)

        ax.imshow(overlay)
        ax.set_title(f"K{k}", fontsize=10)

    plt.tight_layout(pad=0.4)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Saved] {save_path}")


# ============================================================
# Model loading
# ============================================================

@torch.no_grad()
def load_frozen_model(ckpt_path, cfg, device="cuda"):
    model = SpacecraftPoseDualPath(
        aggregator=cfg["model"]["aggregator"],
        ckpt_path=cfg["model"]["backbone_ckpt"],
        heatmap_out=tuple(cfg["model"]["heatmap_out"]),
        cfg=cfg,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model


# ============================================================
# Probing
# ============================================================

@torch.no_grad()
def extract_tokens_kps_heatmaps(model, dataset, idx=0, device="cuda"):
    frames, kps_gt, hm_gt, *_ = dataset[idx]
    frames = frames.unsqueeze(0).to(device)  # (1,T,C,H,W)

    out = model(frames)

    T = frames.shape[1]
    t = T // 2

    # patch tokens
    patches = model.backbone(frames)[1][0, t]  # (P,D)

    # GT keypoints
    kps = kps_gt[t].cpu().numpy()  # (K,2)

    # FIX: logits -> probability heatmaps
    hm_logits = out["heatmaps"][0, t].cpu().numpy()
    hm_pred = logits_to_prob_heatmaps(
        hm_logits,
        tau=cfg["loss"].get("tau", 0.4),
    )

    frame = frames[0, t].permute(1, 2, 0).cpu().numpy()
    frame = (frame * 255).astype(np.uint8)

    return patches.cpu().numpy(), kps, hm_pred, frame


# ============================================================
# Token analysis
# ============================================================

def pca_patch_visualization(patches, kps_norm, grid_hw=(14,14)):
    pca = PCA(n_components=2)
    emb = pca.fit_transform(patches)

    H, W = grid_hw
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    xs, ys = xs.flatten(), ys.flatten()

    plt.figure(figsize=(6,6))
    plt.scatter(xs, ys, c=emb[:,0], cmap="coolwarm", s=25)

    kps_px = (kps_norm + 1) / 2 * (W - 1)
    plt.scatter(kps_px[:,0], kps_px[:,1], c="lime", s=80, marker="x")

    plt.title("PCA of ViT patch tokens (PC1)")
    plt.gca().invert_yaxis()
    plt.savefig("./debug_vis/pca_patch_tokens_stage0.png", dpi=300)
    plt.close()


def cosine_affinity_test(patches, kps_norm, grid_hw=(14,14)):
    patches = patches / np.linalg.norm(patches, axis=1, keepdims=True)
    anchor = patches.mean(axis=0, keepdims=True)
    sims = (patches @ anchor.T).squeeze()

    H, W = grid_hw
    max_idx = sims.argmax()
    y, x = divmod(max_idx, W)

    kps_px = (kps_norm + 1) / 2 * (W - 1)
    dists = np.linalg.norm(kps_px - np.array([[x,y]]), axis=1)
    print(f"Nearest-token → GT mean dist: {dists.mean():.2f} patches")


# ============================================================
# Config + runner
# ============================================================

def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    for k in ["data","model","train","loss","eval"]:
        cfg.setdefault(k, {})
    return cfg


def run_probe(cfg, ckpt_path):
    device = "cuda"
    model = load_frozen_model(ckpt_path, cfg, device)

    ds = SPADESDataset(
        mode="val",
        gt_folder=cfg["data"]["root"],
        short_mem=cfg["data"]["short_mem"],
        input_size=cfg["data"]["input_size"],
        sampling_rate=cfg["data"]["sampling_rate"],
        crop=False,
        heatmap_out=tuple(cfg["model"]["heatmap_out"]),
    )

    patches, kps, hm_pred, frame = extract_tokens_kps_heatmaps(
        model, ds, idx=0, device=device
    )

    pca_patch_visualization(patches, kps)
    cosine_affinity_test(patches, kps)

    visualize_heatmap_grid(
        frame=frame,
        heatmaps=hm_pred,
        kps_norm=kps,
        cols=hm_pred.shape[0],
        save_path="./debug_vis/debug_full_heatmap_grid_stage0.png",
    )


# ============================================================
if __name__ == "__main__":
    cfg = load_cfg("./config.yaml")
    ckpt_path = "./outputs/spades_dinov3L_dp/stage0_anchor.pt"
    run_probe(cfg, ckpt_path)