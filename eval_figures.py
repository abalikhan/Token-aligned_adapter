"""
eval_figures.py
---------------
Generates exactly three paper-quality evaluation figures:

  1) fig1_trajectory.png  — keypoint trajectory strip (6 frames at stride 8)
  2) fig2_pose_grid.png   — 3×3 full-frame grid with GT + pred 3-D bbox / keypoints
  3) fig3_jitter.png      — GT (solid) vs predicted (dashed) position trajectories

Usage
-----
    python eval_figures.py \
        --config config_vitb.yaml \
        --ckpt   /path/to/best.pt \
        --seq    RT001 \
        --seqs   RT001 RT002 RT003 RT004 \
        --out_dir ./eval_figs

    # --seq  : main sequence used for Fig 1 and Fig 2
    # --seqs : sequences used for Fig 3 (defaults to first 4 in dataset)
    # Camera / points paths are derived automatically from the config.
"""

import argparse
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R_scipy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import SPADESDataset
from model_hm_dsnt_2 import SpacecraftHMDSNT
from misc_new import create_pose_estimator

# ---------------------------------------------------------------------------
# ImageNet constants
# ---------------------------------------------------------------------------
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ===========================================================================
# Config / model helpers
# ===========================================================================

def load_cfg(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k in ("data", "model", "eval", "loss", "train"):
        cfg.setdefault(k, {})
    return cfg


def build_model(cfg: dict) -> SpacecraftHMDSNT:
    d, m   = cfg["data"], cfg["model"]
    T      = int(d["test_T_chunk"])
    isz    = int(d["input_size"])
    psz    = int(m.get("patch_size", 16))
    Hp     = isz // psz
    hm_hw  = tuple(m.get("heatmap_out", [256, 256]))
    nreg   = m.get("n_register_tokens", None)
    if nreg is not None:
        nreg = int(nreg)
    return SpacecraftHMDSNT(
        num_kp             = int(d["num_kp"]),
        num_frames         = T,
        img_hw             = isz,
        hf_name            = m["hf_name"],
        adapter_bottleneck = int(m.get("adapter_bottleneck", 64)),
        adapter_drop       = float(m.get("adapter_drop", 0.1)),
        adapter_in_last    = m.get("adapter_in_last", None),
        heatmap_hw         = hm_hw,
        grid_hw            = (Hp, Hp),
        n_register_tokens  = nreg,
        ks=3, kt=3,
        fuse               = m.get("fuse", "sum"),
        hm_mid_c           = int(m.get("hm_mid_c", 128)),
    )


def load_ckpt(model: SpacecraftHMDSNT, ckpt_path: str, device: str):
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [ckpt] missing ({len(missing)}): {missing[:3]} …")
    if unexpected:
        print(f"  [ckpt] unexpected ({len(unexpected)}): {unexpected[:3]} …")
    model.to(device).eval()
    print(f"  [ckpt] loaded from {ckpt_path}")


# ===========================================================================
# Dataset helper
# ===========================================================================

def build_dataset(cfg: dict, mode: str = "val") -> SPADESDataset:
    d = cfg["data"]
    return SPADESDataset(
        gt_folder         = d["root"],
        input_size        = int(d["input_size"]),
        sampling_rate     = int(d["sampling_rate"]),
        short_mem         = int(d.get("short_mem", 8)),
        heatmap_out       = tuple(cfg["model"].get("heatmap_out", [256, 256])),
        gaussian_sigma_px = float(d.get("gaussian_sigma_px", 2.0)),
        mode              = mode,
    )


# ===========================================================================
# Coordinate conversion — strictly 2-D (T, K), no shape promotion
# ===========================================================================

def kps_norm_to_full_px(
    kps_norm:   torch.Tensor,   # (T, K, 2)  heatmap-normalised [-1, 1]
    crop_xywh:  torch.Tensor,   # (T, 4)     normalised crop box
    meta:       torch.Tensor,   # (T, 5)     [scale, pad_l, pad_t, W_full, H_full]
    input_size: int,
    hm_hw,                      # (Hm, Wm)
) -> torch.Tensor:              # (T, K, 2)  full-frame pixel coords
    Hm, Wm = hm_hw

    # Step 1: heatmap-norm → heatmap pixels   (T, K)
    hm_x = (kps_norm[:, :, 0] + 1.0) * 0.5 * float(Wm - 1)
    hm_y = (kps_norm[:, :, 1] + 1.0) * 0.5 * float(Hm - 1)

    # Step 2: heatmap px → input-size px   (T, K)
    sx = float(input_size - 1) / float(max(Wm - 1, 1))
    sy = float(input_size - 1) / float(max(Hm - 1, 1))
    inp_x = hm_x * sx
    inp_y = hm_y * sy

    # Step 3: input-size px → full-frame px
    use_letterbox = torch.isfinite(meta[:, 0]).all()

    if use_letterbox:
        scale  = meta[:, 0].unsqueeze(1).clamp_min(1e-8)   # (T, 1)
        pad_l  = meta[:, 1].unsqueeze(1)
        pad_t  = meta[:, 2].unsqueeze(1)
        W_full = meta[:, 3].unsqueeze(1)
        H_full = meta[:, 4].unsqueeze(1)
        full_x = ((inp_x - pad_l) / scale).clamp(0.0, (W_full - 1).clamp_min(0.0))
        full_y = ((inp_y - pad_t) / scale).clamp(0.0, (H_full - 1).clamp_min(0.0))
    else:
        # crop path — W_full, H_full in meta columns 3, 4
        W_full_f = float(meta[0, 3].item())
        H_full_f = float(meta[0, 4].item())
        den      = float(max(input_size - 1, 1))

        x0 = (crop_xywh[:, 0] * W_full_f).unsqueeze(1)   # (T, 1)
        y0 = (crop_xywh[:, 1] * H_full_f).unsqueeze(1)
        w  = (crop_xywh[:, 2] * W_full_f).unsqueeze(1)
        h  = (crop_xywh[:, 3] * H_full_f).unsqueeze(1)

        full_x = x0 + (inp_x / den) * w   # (T, K)
        full_y = y0 + (inp_y / den) * h

    return torch.stack([full_x, full_y], dim=-1)   # (T, K, 2)


# ===========================================================================
# Misc helpers
# ===========================================================================

def denorm(frame_chw: torch.Tensor) -> np.ndarray:
    """ImageNet-denorm → uint8 HWC RGB numpy array."""
    img = (frame_chw.cpu() * _STD + _MEAN).clamp(0.0, 1.0)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def load_full_frame(imgs_path: str, frame_idx: int, seq_name: str,
                    ext: str = ".png") -> np.ndarray:
    """Load an original (un-cropped) frame as uint8 RGB."""
    name    = f"img{frame_idx:03d}_{seq_name}{ext}"
    path    = os.path.join(imgs_path, name)
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Frame not found: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def cam_and_points_from_cfg(cfg: dict):
    """Resolve camera JSON and object-points JSON paths from the config."""
    d   = cfg["data"]
    cam = d.get("cam_intrinsics_json") or d.get("camera_json")
    if not cam:
        raise ValueError("Set data.cam_intrinsics_json in config.")
    pts = d.get("object_points_json")
    if not pts:
        root    = d["root"].rstrip("/")
        ds_name = d.get("name", "").lower()
        pts     = root + ("/points_spark.json" if ds_name == "spark"
                          else "/points.json")
    return cam, pts


# ===========================================================================
# 3-D bounding box helpers
# ===========================================================================

def bbox_corners_from_pts(obj_pts: np.ndarray) -> np.ndarray:
    """8 corners of the axis-aligned bounding box of the object keypoints."""
    lo = obj_pts.min(axis=0)
    hi = obj_pts.max(axis=0)
    c  = np.array([
        [lo[0], lo[1], lo[2]],
        [hi[0], lo[1], lo[2]],
        [hi[0], hi[1], lo[2]],
        [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]],
        [hi[0], lo[1], hi[2]],
        [hi[0], hi[1], hi[2]],
        [lo[0], hi[1], hi[2]],
    ], dtype=np.float64)
    return c


_BBOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # back face
    (4, 5), (5, 6), (6, 7), (7, 4),   # front face
    (0, 4), (1, 5), (2, 6), (3, 7),   # side edges
]


def draw_bbox_on_img(
    img: np.ndarray,            # uint8 HWC RGB  (will be modified in-place)
    corners_2d: np.ndarray,     # (8, 2) projected corners
    color_rgb: tuple,           # (R, G, B)
    thickness: int = 2,
) -> np.ndarray:
    """Draw a 3-D bounding box wireframe on a uint8 RGB image."""
    # cv2 works on BGR but since img is an RGB numpy array we pass RGB colours
    # directly — the final imshow in matplotlib will still look correct.
    pts = corners_2d.astype(np.int32)
    H, W = img.shape[:2]
    for i, j in _BBOX_EDGES:
        p1 = tuple(int(x) for x in np.clip(pts[i], -W, 2 * W))
        p2 = tuple(int(x) for x in np.clip(pts[j], -H, 2 * H))
        cv2.line(img, p1, p2, color_rgb, thickness, cv2.LINE_AA)
    return img


def project_pts(pts_3d, rvec, tvec, cam_mat, dist):
    """Project 3-D points. Returns (N, 2) float64 or None on failure."""
    try:
        proj, _ = cv2.projectPoints(
            pts_3d.astype(np.float64),
            rvec.astype(np.float64),
            tvec.astype(np.float64),
            cam_mat, dist,
        )
        return proj.reshape(-1, 2)
    except Exception as e:
        print(f"    [warn] projectPoints failed: {e}")
        return None


# ===========================================================================
# PnP per-frame  (returns array of translations, shape (T, 3))
# ===========================================================================

def pnp_per_frame(
    kps_full_px: np.ndarray,   # (T, K, 2)
    vis:         np.ndarray,   # (T, K)
    obj_pts:     np.ndarray,   # (K, 3)
    cam_mat:     np.ndarray,
    dist:        np.ndarray,
) -> np.ndarray:               # (T, 3) predicted translation, NaN where failed
    T, K, _ = kps_full_px.shape
    tvecs    = np.full((T, 3), np.nan, dtype=np.float64)

    for t in range(T):
        mask = vis[t] > 0.5
        if mask.sum() < 4:
            continue
        img_pts = kps_full_px[t][mask].astype(np.float64).reshape(-1, 1, 2)
        obj_sel = obj_pts[mask].astype(np.float64)
        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj_sel, img_pts, cam_mat, dist,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok:
                tvecs[t] = tvec.ravel()
        except Exception:
            pass

    return tvecs


# ===========================================================================
# Chunked model inference on a full sequence
# ===========================================================================

@torch.inference_mode()
def run_model_on_sequence(
    model, cfg: dict, device: str, ds: SPADESDataset, seq_name: str
) -> dict:
    """Run model on every annotated frame of *seq_name*."""
    seq_entry = None
    for entry in ds.dataset:
        if entry[0] == seq_name:
            seq_entry = entry
            break
    if seq_entry is None:
        raise ValueError(f"Sequence '{seq_name}' not found in the dataset.")

    _, ws, we, imgs_path, gt_idx = seq_entry
    all_frame_idxs = np.array(ds.seq_to_frames[seq_name], dtype=np.int64)
    print(f"    seq={seq_name}  n_frames={len(all_frame_idxs)}  path={imgs_path}")

    frames, kps_gt, gt_q, gt_r, vis, hm, crop_xywh, meta = ds.get_kps_gt(
        all_frame_idxs, seq_name, gt_idx, imgs_path
    )

    T_chunk = int(cfg["data"].get("test_T_chunk", 8))
    T       = frames.shape[0]
    chunks  = []

    use_amp = device == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        for start in range(0, T, T_chunk):
            chunk   = frames[start : start + T_chunk]
            n_valid = chunk.shape[0]
            if n_valid < T_chunk:
                pad   = frames[-1:].expand(T_chunk - n_valid, -1, -1, -1)
                chunk = torch.cat([chunk, pad], dim=0)
            out = model(chunk.unsqueeze(0).to(device))
            chunks.append(out["keypoints"].float().cpu()[0, :n_valid])

    kps_pred = torch.cat(chunks, dim=0)   # (T, K, 2)

    return {
        "frames":     frames.cpu(),
        "kps_gt":     kps_gt.cpu(),
        "kps_pred":   kps_pred,
        "vis":        vis.cpu(),
        "gt_trans":   gt_r.cpu(),
        "gt_q":       gt_q.cpu(),
        "crop_xywh":  crop_xywh.cpu(),
        "meta":       meta.cpu(),
        "frame_idxs": all_frame_idxs.tolist(),
        "seq_name":   seq_name,
        "imgs_path":  imgs_path,
    }


# ===========================================================================
# Figure 1 — Keypoint trajectory strip
# ===========================================================================

def figure_1_trajectory(sd: dict, cfg: dict, out_path: str,
                         stride: int = 8, n_frames: int = 6):
    """
    Six frames at stride *stride*.  GT: + / solid line.  Pred: × / dashed.
    Lines cross subplot boundaries using ConnectionPatch.
    """
    frames   = sd["frames"]    # (T, 3, H, W)
    kps_gt   = sd["kps_gt"]    # (T, K, 2) heatmap-norm
    kps_pred = sd["kps_pred"]  # (T, K, 2)

    T = frames.shape[0]
    H_img, W_img = frames.shape[2], frames.shape[3]
    K = kps_gt.shape[1]

    hm_hw = tuple(cfg["model"].get("heatmap_out", [256, 256]))
    isz   = int(cfg["data"]["input_size"])
    Hm, Wm = hm_hw

    # Selected frame indices (clamped to available)
    fi_list = [min(i * stride, T - 1) for i in range(n_frames)]

    # heatmap-norm → input-size px  (T, K)  — stays 2-D throughout
    def _to_input_px(kps_norm):
        hx = (kps_norm[:, :, 0] + 1.0) * 0.5 * float(Wm - 1)
        hy = (kps_norm[:, :, 1] + 1.0) * 0.5 * float(Hm - 1)
        sx = float(isz - 1) / float(max(Wm - 1, 1))
        sy = float(isz - 1) / float(max(Hm - 1, 1))
        return hx * sx, hy * sy   # both (T, K)

    gt_px_x,   gt_px_y   = _to_input_px(kps_gt)
    pred_px_x, pred_px_y = _to_input_px(kps_pred)

    K_vis  = min(K, 10)
    colors = plt.colormaps["tab10"](np.linspace(0, 1, 10))[:K_vis]

    n_show = len(fi_list)
    fig, axes = plt.subplots(1, n_show, figsize=(3.2 * n_show, 3.5),
                             squeeze=False)
    axes = axes[0]

    # ── Draw frame images + per-frame markers ─────────────────────────────
    for col, fi in enumerate(fi_list):
        ax = axes[col]
        ax.imshow(denorm(frames[fi]))
        ax.set_xlim(0, W_img - 1)
        ax.set_ylim(H_img - 1, 0)
        ax.axis("off")

        for k in range(K_vis):
            color = colors[k]
            # GT marker (+)
            ax.scatter(
                [gt_px_x[fi, k].item()], [gt_px_y[fi, k].item()],
                c=[color], marker="+", s=100, linewidths=2, zorder=6,
            )
            # Pred marker (×)
            ax.scatter(
                [pred_px_x[fi, k].item()], [pred_px_y[fi, k].item()],
                c=[color], marker="x", s=70, linewidths=1.8, zorder=6,
            )

    # ── Draw trajectory lines crossing subplot boundaries ─────────────────
    plt.tight_layout(pad=0.2)   # must call before add_artist for correct coords

    for col in range(1, n_show):
        fi_prev = fi_list[col - 1]
        fi_curr = fi_list[col]
        ax_prev = axes[col - 1]
        ax_curr = axes[col]

        for k in range(K_vis):
            color = colors[k]

            # GT line (solid), from prev subplot to current subplot
            con_gt = ConnectionPatch(
                xyA=(gt_px_x[fi_prev, k].item(), gt_px_y[fi_prev, k].item()),
                xyB=(gt_px_x[fi_curr, k].item(), gt_px_y[fi_curr, k].item()),
                coordsA="data", coordsB="data",
                axesA=ax_prev, axesB=ax_curr,
                color=color, linestyle="-", linewidth=1.2, alpha=0.85,
                zorder=10,
            )
            fig.add_artist(con_gt)

            # Pred line (dashed)
            con_pred = ConnectionPatch(
                xyA=(pred_px_x[fi_prev, k].item(), pred_px_y[fi_prev, k].item()),
                xyB=(pred_px_x[fi_curr, k].item(), pred_px_y[fi_curr, k].item()),
                coordsA="data", coordsB="data",
                axesA=ax_prev, axesB=ax_curr,
                color=color, linestyle="--", linewidth=1.2, alpha=0.85,
                zorder=10,
            )
            fig.add_artist(con_pred)

    fig.suptitle(
        "Keypoint Trajectories: GT (+, solid) vs DSNT Pred (×, dashed)",
        fontsize=12, y=1.01,
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig1] saved → {out_path}")


# ===========================================================================
# Figure 2 — 3×3 full-frame pose grid
# ===========================================================================

def figure_2_pose_grid(seq_data_list: list, cfg: dict,
                        cam_json: str, points_json: str, out_path: str):
    """
    3×3 grid of full-resolution frames.
    GT:   green wireframe 3-D bounding box + green keypoint circles.
    Pred: pink  wireframe 3-D bounding box + pink  keypoint circles.
    """
    d, m   = cfg["data"], cfg["model"]
    isz    = int(d["input_size"])
    hm_hw  = tuple(m.get("heatmap_out", [256, 256]))
    num_kp = int(d["num_kp"])

    pe, pnp_cfg, _scale = create_pose_estimator(num_kp, cam_json, points_json)
    cam_mat = np.asarray(pe.camera_matrix, dtype=np.float64)
    dist    = np.asarray(pe.dist_coefs,    dtype=np.float64)
    obj_pts = np.asarray(pe.model_points,  dtype=np.float64)   # (K, 3)

    bbox_3d = bbox_corners_from_pts(obj_pts)   # (8, 3)

    # GT colour (pure green RGB), Pred colour (hot-pink RGB)
    GT_RGB   = (0,   220, 0)
    PRED_RGB = (255, 20, 147)

    # ── Collect 9 samples across sequences ────────────────────────────────
    samples = []
    for sd in seq_data_list:
        T          = sd["frames"].shape[0]
        frame_idxs = sd["frame_idxs"]

        kps_pred_full = kps_norm_to_full_px(
            sd["kps_pred"], sd["crop_xywh"], sd["meta"], isz, hm_hw
        ).numpy()   # (T, K, 2)
        kps_gt_full   = kps_norm_to_full_px(
            sd["kps_gt"],   sd["crop_xywh"], sd["meta"], isz, hm_hw
        ).numpy()   # (T, K, 2)

        # How many frames to sample from this sequence
        n_want = max(1, 9 // len(seq_data_list))
        t_idxs = np.linspace(0, T - 1, n_want, dtype=int)

        for t in t_idxs:
            fi = frame_idxs[int(t)]
            try:
                full_frame = load_full_frame(sd["imgs_path"], fi, sd["seq_name"])
            except FileNotFoundError as e:
                print(f"  [warn] {e}")
                continue

            samples.append({
                "img":      full_frame,
                "gt_kps":   kps_gt_full[t],            # (K, 2) full-frame px
                "pred_kps": kps_pred_full[t],           # (K, 2) full-frame px
                "gt_trans": sd["gt_trans"][t].numpy(),  # (3,)
                "gt_q":     sd["gt_q"][t].numpy(),      # (4,) xyzw
                "vis":      sd["vis"][t].numpy(),        # (K,)
            })
            if len(samples) >= 9:
                break
        if len(samples) >= 9:
            break

    while len(samples) < 9:
        samples.append(samples[-1])
    samples = samples[:9]

    # ── Draw 3×3 grid ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    axes = axes.ravel()

    for i, s in enumerate(samples):
        ax  = axes[i]
        img = s["img"].copy()   # uint8 RGB
        H, W = img.shape[:2]

        gt_kps   = s["gt_kps"].astype(np.float64)
        pred_kps = s["pred_kps"].astype(np.float64)
        vis_mask = s["vis"] > 0.5

        # ── GT pose: quaternion xyzw → rvec, tvec ─────────────────────────
        try:
            rot_mat   = R_scipy.from_quat(s["gt_q"]).as_matrix()
            rvec_gt,_ = cv2.Rodrigues(rot_mat)
            tvec_gt   = s["gt_trans"].reshape(3, 1).astype(np.float64)

            corners_gt = project_pts(bbox_3d, rvec_gt, tvec_gt, cam_mat, dist)
            if corners_gt is not None:
                draw_bbox_on_img(img, corners_gt, GT_RGB, thickness=2)
        except Exception as e:
            print(f"  [warn] GT pose draw failed sample {i}: {e}")

        # ── Pred pose via PnP ─────────────────────────────────────────────
        if vis_mask.sum() >= 4:
            try:
                ok, rvec_p, tvec_p = cv2.solvePnP(
                    obj_pts[vis_mask].astype(np.float64),
                    pred_kps[vis_mask].reshape(-1, 1, 2),
                    cam_mat, dist,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if ok:
                    corners_pred = project_pts(bbox_3d, rvec_p, tvec_p, cam_mat, dist)
                    if corners_pred is not None:
                        draw_bbox_on_img(img, corners_pred, PRED_RGB, thickness=2)
            except Exception as e:
                print(f"  [warn] pred PnP failed sample {i}: {e}")

        # ── Keypoints ─────────────────────────────────────────────────────
        r = max(3, H // 150)   # circle radius scales with image size
        for k in range(len(gt_kps)):
            gx, gy = int(round(gt_kps[k, 0])),    int(round(gt_kps[k, 1]))
            px, py = int(round(pred_kps[k, 0])),  int(round(pred_kps[k, 1]))
            if 0 <= gx < W and 0 <= gy < H:
                cv2.circle(img, (gx, gy), r, GT_RGB,   -1, cv2.LINE_AA)
            if 0 <= px < W and 0 <= py < H:
                cv2.circle(img, (px, py), r, PRED_RGB, -1, cv2.LINE_AA)

        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout(pad=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig2] saved → {out_path}")


# ===========================================================================
# Figure 3 — Temporal position jitter: GT (solid) vs Pred (dashed)
# ===========================================================================

def figure_3_jitter(seq_data_list: list, cfg: dict,
                     cam_json: str, points_json: str, out_path: str):
    """
    1×N subplots (N ≤ 4).
    Tx/Ty/Tz of GT (solid) and predicted (dashed) translations vs time step.
    Predicted translation is obtained via per-frame PnP on the predicted keypoints.
    """
    d, m   = cfg["data"], cfg["model"]
    isz    = int(d["input_size"])
    hm_hw  = tuple(m.get("heatmap_out", [256, 256]))
    num_kp = int(d["num_kp"])

    pe, pnp_cfg, _scale = create_pose_estimator(num_kp, cam_json, points_json)
    cam_mat = np.asarray(pe.camera_matrix, dtype=np.float64)
    dist    = np.asarray(pe.dist_coefs,    dtype=np.float64)
    obj_pts = np.asarray(pe.model_points,  dtype=np.float64)   # (K, 3)

    n    = min(len(seq_data_list), 4)
    seqs = seq_data_list[:n]

    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3.5), squeeze=False)
    axes = axes[0]

    for i, sd in enumerate(seqs):
        ax = axes[i]

        gt_trans = sd["gt_trans"].numpy()    # (T, 3)
        T        = gt_trans.shape[0]
        t_axis   = np.arange(T)

        # ── GT trajectories (solid) ────────────────────────────────────────
        ax.plot(t_axis, gt_trans[:, 0], color="black", linewidth=0.9, label="Tx GT")
        ax.plot(t_axis, gt_trans[:, 1], color="red",   linewidth=0.9, label="Ty GT")
        ax.plot(t_axis, gt_trans[:, 2], color="blue",  linewidth=0.9, label="Tz GT")

        # ── Predicted trajectories via per-frame PnP (dashed) ─────────────
        kps_pred_full = kps_norm_to_full_px(
            sd["kps_pred"], sd["crop_xywh"], sd["meta"], isz, hm_hw
        ).numpy()   # (T, K, 2)

        pred_tvecs = pnp_per_frame(
            kps_pred_full, sd["vis"].numpy(), obj_pts, cam_mat, dist
        )   # (T, 3) with NaN where PnP failed

        # Only plot where we have valid predictions
        valid = np.isfinite(pred_tvecs[:, 0])
        if valid.sum() > 0:
            t_v = t_axis[valid]
            ax.plot(t_v, pred_tvecs[valid, 0], color="black", linewidth=0.7,
                    linestyle="--", alpha=0.7, label="Tx pred")
            ax.plot(t_v, pred_tvecs[valid, 1], color="red",   linewidth=0.7,
                    linestyle="--", alpha=0.7, label="Ty pred")
            ax.plot(t_v, pred_tvecs[valid, 2], color="blue",  linewidth=0.7,
                    linestyle="--", alpha=0.7, label="Tz pred")

        ax.set_xlabel("Time step", fontsize=9)
        ax.set_title(sd["seq_name"], fontsize=9)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.set_ylabel("Translation (m)", fontsize=9)
            # Compact legend: solid=GT, dashed=pred
            from matplotlib.lines import Line2D
            handles = [
                Line2D([0], [0], color="black", lw=1.0, label="Tx"),
                Line2D([0], [0], color="red",   lw=1.0, label="Ty"),
                Line2D([0], [0], color="blue",  lw=1.0, label="Tz"),
                Line2D([0], [0], color="gray",  lw=1.0, ls="-",  label="GT"),
                Line2D([0], [0], color="gray",  lw=1.0, ls="--", label="Pred"),
            ]
            ax.legend(handles=handles, fontsize=6, loc="upper left", ncol=2)

    plt.suptitle("Position Trajectories: GT (solid) vs Predicted (dashed)",
                 fontsize=11, y=1.02)
    plt.tight_layout(pad=0.4)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3] saved → {out_path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate 3 paper-quality evaluation figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config",  required=True)
    parser.add_argument("--ckpt",    required=True)
    parser.add_argument("--seq",     default=None,
                        help="Main sequence for Fig 1 & Fig 2")
    parser.add_argument("--seqs",    nargs="+", default=None,
                        help="Up to 4 sequences for Fig 3")
    parser.add_argument("--out_dir", default=".")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg    = load_cfg(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}")

    print("[setup] building model …")
    model = build_model(cfg)
    load_ckpt(model, args.ckpt, device)

    print("[setup] building dataset (val mode) …")
    ds = build_dataset(cfg, mode="val")

    all_seqs_in_ds  = list(dict.fromkeys(e[0] for e in ds.dataset))
    seq_main        = args.seq  or all_seqs_in_ds[0]
    seqs_for_jitter = (args.seqs or all_seqs_in_ds[:4])[:4]
    needed_seqs     = list(dict.fromkeys([seq_main] + seqs_for_jitter))

    print(f"[inference] running on {len(needed_seqs)} sequence(s) …")
    seq_data = {}
    for sn in needed_seqs:
        print(f"  → {sn}")
        seq_data[sn] = run_model_on_sequence(model, cfg, device, ds, sn)

    cam_json, points_json = cam_and_points_from_cfg(cfg)
    print(f"[paths] cam    = {cam_json}")
    print(f"[paths] points = {points_json}")

    # ── Figure 1 ─────────────────────────────────────────────────────────
    out1 = os.path.join(args.out_dir, "fig1_trajectory.png")
    figure_1_trajectory(seq_data[seq_main], cfg, out1, stride=8, n_frames=6)

    # ── Figure 2 ─────────────────────────────────────────────────────────
    out2 = os.path.join(args.out_dir, "fig2_pose_grid.png")
    figure_2_pose_grid(
        seq_data_list = list(seq_data.values()),
        cfg           = cfg,
        cam_json      = cam_json,
        points_json   = points_json,
        out_path      = out2,
    )

    # ── Figure 3 ─────────────────────────────────────────────────────────
    out3 = os.path.join(args.out_dir, "fig3_jitter.png")
    figure_3_jitter(
        seq_data_list = [seq_data[sn] for sn in seqs_for_jitter],
        cfg           = cfg,
        cam_json      = cam_json,
        points_json   = points_json,
        out_path      = out3,
    )

    print("\n[done] All three figures written to", args.out_dir)


if __name__ == "__main__":
    main()
