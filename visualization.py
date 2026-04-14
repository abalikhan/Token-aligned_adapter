# visualization.py
import os
import re
import yaml
import cv2
import torch
import numpy as np
import torch.nn.functional as F

from dataset import SPADESDataset
from model_hm_dsnt_2 import SpacecraftHMDSNT
import json 
import math

from utils_new import (
    save_kp_overlay_bt,
    save_heatmap_overlay_bt,
    save_token_similarity_diagnostics_bt,
    save_dino_style_kp_attention_overlays_bt,
    _kps_normhm_to_full_px_letterbox,
    _kps_normhm_to_full_px,
    _heatmap_logits_to_full_px_letterbox,
    _heatmap_logits_to_full_px
)

from misc import create_pose_estimator, estimate_pose_from_keypoints
# -------------------------
# Config / IO
# -------------------------
def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg.setdefault("eval", {})
    cfg.setdefault("loss", {})
    return cfg


def _get_cuboid_corners_from_points(pts_3d: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    pts_3d: (N,3) object-frame points (e.g., mesh points or keypoints)
    returns: (8,3) cuboid corners in object frame
    """
    pts = np.asarray(pts_3d, dtype=np.float64).reshape(-1, 3) * float(scale)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)

    x0, y0, z0 = mn
    x1, y1, z1 = mx

    # 8 corners
    corners = np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ], dtype=np.float64)
    return corners


def get_pose_estimator_cuboid(pose_estimator) -> np.ndarray:
    """
    Prefer dense mesh points if present; fallback to model_points (keypoints).
    """
    if hasattr(pose_estimator, "mesh_points") and pose_estimator.mesh_points is not None:
        pts = pose_estimator.mesh_points
    elif hasattr(pose_estimator, "mesh") and pose_estimator.mesh is not None:
        pts = pose_estimator.mesh  # in case you stored points here
    else:
        pts = pose_estimator.model_points
    return _get_cuboid_corners_from_points(pts)

def draw_3d_cuboid_on_original(
    img_rgb: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    q_wxyz: np.ndarray,
    t_xyz: np.ndarray,
    cuboid_corners_3d: np.ndarray,  # (8,3) in object coords
    color_bgr=(255, 0, 255),
    thickness: int = 1,
):
    """
    Draw a 3D cuboid projected onto the image using (R,t).
    """
    out = img_rgb.copy()
    uv = _project_points_cv(cuboid_corners_3d, K, dist, q_wxyz, t_xyz)  # (8,2)

    pts2d = np.round(uv).astype(int)

    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    # edges: bottom face 0-1-2-3, top face 4-5-6-7, verticals 0-4,1-5,2-6,3-7
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for i, j in edges:
        p1 = tuple(pts2d[i].tolist())
        p2 = tuple(pts2d[j].tolist())
        cv2.line(bgr, p1, p2, color_bgr, thickness, lineType=cv2.LINE_AA)

    out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return out

# ============================================================
# visualization.py  (ADD) — 6DoF pose overlay in ORIGINAL space
# Prediction-only (PnP from predicted keypoints) + draw axes.
# ============================================================

# -------------------------
# Camera / Pose helpers
# -------------------------

def _load_camera_from_json(cam_json_path: str):
    """
    Returns:
      K: (3,3) float64
      dist: (N,1) float64  (OpenCV expects (N,1))
    """
    with open(cam_json_path, "r") as f:
        dt = json.load(f)
    K = np.asarray(dt["cameraMatrix"], dtype=np.float64)
    dist = dt.get("distCoeffs", [0, 0, 0, 0, 0])
    dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)
    return K, dist


def _quat_wxyz_to_R(q_wxyz: np.ndarray) -> np.ndarray:
    """
    q_wxyz: (4,) -> R (3,3)
    """
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    w, x, y, z = q
    n = math.sqrt(w*w + x*x + y*y + z*z) + 1e-12
    w, x, y, z = w/n, x/n, y/n, z/n

    # standard quaternion->rotation
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return R


def _project_points_cv(pts_3d: np.ndarray, K: np.ndarray, dist: np.ndarray, q_wxyz: np.ndarray, t_xyz: np.ndarray):
    """
    pts_3d: (N,3) in object coords
    q_wxyz: (4,)
    t_xyz:  (3,) in camera coords (OpenCV convention)
    returns: (N,2) float32
    """
    R = _quat_wxyz_to_R(q_wxyz)
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    tvec = np.asarray(t_xyz, dtype=np.float64).reshape(3, 1)

    pts_3d = np.asarray(pts_3d, dtype=np.float64).reshape(-1, 3)
    proj, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, dist)
    return proj.reshape(-1, 2).astype(np.float32)


def draw_6dof_axes_on_original(
    img_rgb: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    q_wxyz: np.ndarray,
    t_xyz: np.ndarray,
    axis_len: float = 0.20,   # in SAME units as points.json keypoints (after its scale)
    thickness: int = 2,
    draw_origin: bool = True,
):
    """
    Draw X/Y/Z axes using the predicted pose on ORIGINAL RGB image.
    Colors (OpenCV BGR):
      X = red, Y = green, Z = blue
    """
    out = img_rgb.copy()

    # origin + axis endpoints in object frame
    pts = np.array([
        [0.0, 0.0, 0.0],               # origin
        [axis_len, 0.0, 0.0],          # +X
        [0.0, axis_len, 0.0],          # +Y
        [0.0, 0.0, axis_len],          # +Z
    ], dtype=np.float64)

    uv = _project_points_cv(pts, K, dist, q_wxyz, t_xyz)  # (4,2)
    o = tuple(np.round(uv[0]).astype(int).tolist())
    x = tuple(np.round(uv[1]).astype(int).tolist())
    y = tuple(np.round(uv[2]).astype(int).tolist())
    z = tuple(np.round(uv[3]).astype(int).tolist())

    # safety: if origin is wildly off-image, just return
    H, W = out.shape[:2]
    if not (-W <= o[0] <= 2*W and -H <= o[1] <= 2*H):
        return out

    # draw axes (convert RGB->BGR for cv2 drawing, then back)
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    if draw_origin:
        cv2.circle(bgr, o, max(2, thickness + 1), (255, 255, 255), -1, lineType=cv2.LINE_AA)

    cv2.line(bgr, o, x, (0, 0, 255), thickness, lineType=cv2.LINE_AA)   # X red
    cv2.line(bgr, o, y, (0, 255, 0), thickness, lineType=cv2.LINE_AA)   # Y green
    cv2.line(bgr, o, z, (255, 0, 0), thickness, lineType=cv2.LINE_AA)   # Z blue

    out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return out


def draw_projected_model_keypoints_on_original(
    img_rgb: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    model_pts_3d: np.ndarray,   # (K,3) from points.json (already scaled)
    q_wxyz: np.ndarray,
    t_xyz: np.ndarray,
    color_rgb=(0, 255, 255),
    radius: int = 3,
):
    """
    Projects the 3D model keypoints with the predicted pose and draws them.
    (Useful sanity-check: should align with visible spacecraft corners/features.)
    """
    out = img_rgb.copy()
    uv = _project_points_cv(model_pts_3d, K, dist, q_wxyz, t_xyz)
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    col_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))

    for p in uv:
        x, y = int(round(p[0])), int(round(p[1]))
        cv2.circle(bgr, (x, y), radius, col_bgr, -1, lineType=cv2.LINE_AA)

    out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return out

def build_model_for_vis(cfg: dict):
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    T = int(data_cfg["test_T_chunk"])  # we will feed exactly T frames
    input_size = int(data_cfg["input_size"])
    num_kp = int(data_cfg["num_kp"])

    patch_size = int(model_cfg.get("patch_size", 16))
    Hp = input_size // patch_size
    Wp = input_size // patch_size
    grid_hw = (Hp, Wp)

    heatmap_hw = tuple(model_cfg.get("heatmap_out", [256, 256]))
    hm_mid_c = int(model_cfg.get("hm_mid_c", 128))
    hf_name = model_cfg.get("hf_name", "facebook/dinov3-vitl16-pretrain-lvd1689m")

    adapter_bottleneck = int(model_cfg.get("adapter_bottleneck", 64))
    adapter_drop = float(model_cfg.get("adapter_drop", 0.1))
    adapter_in_last = model_cfg.get("adapter_in_last", None)
    if adapter_in_last is not None:
        adapter_in_last = int(adapter_in_last)

    fuse = model_cfg.get("fuse", "sum")

    n_register_tokens = model_cfg.get("n_register_tokens", None)
    if n_register_tokens is not None:
        n_register_tokens = int(n_register_tokens)

    model = SpacecraftHMDSNT(
        num_kp=num_kp,
        num_frames=T,
        img_hw=input_size,
        hf_name=hf_name,
        adapter_bottleneck=adapter_bottleneck,
        adapter_drop=adapter_drop,
        adapter_in_last=adapter_in_last,
        heatmap_hw=heatmap_hw,
        grid_hw=grid_hw,
        n_register_tokens=n_register_tokens,
        ks=3,
        kt=3,
        fuse=fuse,
        hm_mid_c=hm_mid_c,
    )
    return model


def load_ckpt(model, ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[VIS] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        print("  unexpected (first 20):", unexpected[:20])
    if missing:
        print("  missing (first 20):", missing[:20])
    model.to(device).eval()
    return ckpt


# -------------------------
# Dataset helpers
# -------------------------
def build_spades_test_dataset(cfg: dict) -> SPADESDataset:
    d = cfg["data"]
    ds = SPADESDataset(
        gt_folder=d["root"],
        short_mem=int(d["short_mem"]),
        input_size=int(d["input_size"]),
        sampling_rate=int(d["sampling_rate"]),
        clip_stride=int(d.get("clip_stride", 0)),
        crop=bool(d.get("crop", False)),
        mode=cfg["eval"].get("mode", "test"),
        heatmap_out=tuple(cfg["model"].get("heatmap_out", [256, 256])),
        gaussian_sigma_px=float(d.get("gaussian_sigma_px", 2.0)),
        eval_use_crop=d.get("eval_use_crop", None),
        spatial_transforms=None,
    )
    return ds


def build_seq_index_map(ds: SPADESDataset) -> dict:
    seq_to_i = {}
    for i, entry in enumerate(ds.dataset):
        seq = entry[0]
        seq_to_i[seq] = i
    return seq_to_i


def slice_random_clip_from_full_sequence(
    frames_TCHW: torch.Tensor,
    kps_norm_TK2: torch.Tensor,
    quat_T4: torch.Tensor,
    trans_T3: torch.Tensor,
    vis_TK: torch.Tensor,
    hm_TKHW: torch.Tensor,
    frame_ids,
    crop_xywh_T4: torch.Tensor,
    geom_T5: torch.Tensor,
    T_chunk: int,
    rng: np.random.RandomState,
):
    T_full = int(frames_TCHW.shape[0])
    if T_full < T_chunk:
        raise RuntimeError(f"[VIS] Sequence too short: T_full={T_full} < T_chunk={T_chunk}")

    start = int(rng.randint(0, T_full - T_chunk + 1))
    end = start + T_chunk

    if isinstance(frame_ids, (list, tuple)) and len(frame_ids) == 1 and isinstance(frame_ids[0], (list, tuple)):
        frame_ids = frame_ids[0]

    return dict(
        frames=frames_TCHW[start:end],
        kps=kps_norm_TK2[start:end],
        quat=quat_T4[start:end],
        trans=trans_T3[start:end],
        vis=vis_TK[start:end],
        hm=hm_TKHW[start:end],
        frame_ids=frame_ids[start:end],
        crop=crop_xywh_T4[start:end],
        geom=geom_T5[start:end],
        start=start,
        end=end,
    )


# -------------------------
# Original-space helpers
# -------------------------
_FRAME_RE = re.compile(r"^(RT\d+)[_\-](\d+)$")


def _parse_frame_id(fid: str):
    m = _FRAME_RE.match(str(fid))
    if m is None:
        raise ValueError(f"[VIS] Unexpected frame_id format: {fid}")
    return m.group(1), int(m.group(2))


def _resolve_imgs_path(root: str, seq_name: str):
    p1 = os.path.join(root, "images", seq_name)
    if os.path.isdir(p1):
        return p1
    return os.path.join(root, seq_name)


def _read_original_rgb(root: str, seq_name: str, frame_idx: int, ext: str = ".png"):
    imgs_path = _resolve_imgs_path(root, seq_name)
    img_name = f"img{str(frame_idx).zfill(3)}_{seq_name}{ext}"
    p = os.path.join(imgs_path, img_name)
    bgr = cv2.imread(p, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(p)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, p


def _kps_norm_to_input_px(kps_norm_bt: torch.Tensor, input_size: int, heatmap_hw):
    Hm, Wm = int(heatmap_hw[0]), int(heatmap_hw[1])
    k = kps_norm_bt[0]  # (T,K,2)
    x_hm = (k[..., 0] + 1.0) * 0.5 * (Wm - 1.0)
    y_hm = (k[..., 1] + 1.0) * 0.5 * (Hm - 1.0)

    x_in = x_hm * ((input_size - 1.0) / max(Wm - 1.0, 1.0))
    y_in = y_hm * ((input_size - 1.0) / max(Hm - 1.0, 1.0))
    return torch.stack([x_in, y_in], dim=-1)  # (T,K,2)


def _map_input_px_to_original(
    kps_in_px_tk2: np.ndarray,
    crop_xywh_norm_t4: np.ndarray,
    meta_t5: np.ndarray,
    input_size: int,
    fallback_hw: tuple,  # (H_orig, W_orig)
):
    H_orig, W_orig = int(fallback_hw[0]), int(fallback_hw[1])

    crop = crop_xywh_norm_t4
    use_crop = (
        np.isfinite(crop).all()
        and (abs(crop[0]) > 1e-6 or abs(crop[1]) > 1e-6 or abs(crop[2] - 1.0) > 1e-6 or abs(crop[3] - 1.0) > 1e-6)
    )

    if use_crop:
        x0 = float(crop[0]) * W_orig
        y0 = float(crop[1]) * H_orig
        cw = float(crop[2]) * W_orig
        ch = float(crop[3]) * H_orig
        x = kps_in_px_tk2[:, 0] / max(input_size - 1.0, 1.0) * (cw - 1.0) + x0
        y = kps_in_px_tk2[:, 1] / max(input_size - 1.0, 1.0) * (ch - 1.0) + y0
        return np.stack([x, y], axis=-1)

    scale = float(meta_t5[0]) if np.isfinite(meta_t5[0]) else None
    pad_l = float(meta_t5[1]) if np.isfinite(meta_t5[1]) else 0.0
    pad_t = float(meta_t5[2]) if np.isfinite(meta_t5[2]) else 0.0

    if scale is None or scale <= 0:
        x = kps_in_px_tk2[:, 0] / max(input_size - 1.0, 1.0) * (W_orig - 1.0)
        y = kps_in_px_tk2[:, 1] / max(input_size - 1.0, 1.0) * (H_orig - 1.0)
        return np.stack([x, y], axis=-1)

    x = (kps_in_px_tk2[:, 0] - pad_l) / scale
    y = (kps_in_px_tk2[:, 1] - pad_t) / scale
    return np.stack([x, y], axis=-1)


def _heatmap_to_original_mask(
    hm_logits_bt: torch.Tensor,  # (1,T,K,Hm,Wm)
    t0: int,
    input_size: int,
    crop_xywh_norm_t4: np.ndarray,
    meta_t5: np.ndarray,
    orig_hw: tuple,              # (H_orig, W_orig)
):
    H_orig, W_orig = int(orig_hw[0]), int(orig_hw[1])

    hm = hm_logits_bt[0, t0]  # (K,Hm,Wm)
    hm = hm.max(dim=0).values.unsqueeze(0).unsqueeze(0).float()  # (1,1,Hm,Wm)

    hm_in = F.interpolate(hm, size=(input_size, input_size), mode="bilinear", align_corners=False)[0, 0]
    hm_in = hm_in.clamp_min(0.0)
    hm_in = (hm_in / hm_in.max().clamp_min(1e-6)).clamp(0, 1)
    hm_in_np = hm_in.detach().cpu().numpy().astype(np.float32)  # (S,S)

    crop = crop_xywh_norm_t4
    use_crop = (
        np.isfinite(crop).all()
        and (abs(crop[0]) > 1e-6 or abs(crop[1]) > 1e-6 or abs(crop[2] - 1.0) > 1e-6 or abs(crop[3] - 1.0) > 1e-6)
    )

    if use_crop:
        x0 = int(round(float(crop[0]) * W_orig))
        y0 = int(round(float(crop[1]) * H_orig))
        cw = int(round(float(crop[2]) * W_orig))
        ch = int(round(float(crop[3]) * H_orig))
        cw = max(1, cw)
        ch = max(1, ch)

        hm_crop = cv2.resize(hm_in_np, (cw, ch), interpolation=cv2.INTER_LINEAR)

        mask = np.zeros((H_orig, W_orig), dtype=np.float32)
        x1 = min(W_orig, x0 + cw)
        y1 = min(H_orig, y0 + ch)
        if x0 < W_orig and y0 < H_orig and x1 > 0 and y1 > 0:
            mask[y0:y1, x0:x1] = hm_crop[: (y1 - y0), : (x1 - x0)]
        return mask

    # Full-frame letterbox
    scale = float(meta_t5[0]) if np.isfinite(meta_t5[0]) else None
    pad_l = float(meta_t5[1]) if np.isfinite(meta_t5[1]) else 0.0
    pad_t = float(meta_t5[2]) if np.isfinite(meta_t5[2]) else 0.0

    if scale is None or scale <= 0:
        return cv2.resize(hm_in_np, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    content_w = int(round(scale * max(W_orig - 1.0, 1.0)))
    content_h = int(round(scale * max(H_orig - 1.0, 1.0)))
    x0 = int(round(pad_l))
    y0 = int(round(pad_t))
    x1 = min(input_size, x0 + content_w)
    y1 = min(input_size, y0 + content_h)

    if x1 <= x0 or y1 <= y0:
        return cv2.resize(hm_in_np, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    hm_content = hm_in_np[y0:y1, x0:x1]
    hm_orig = cv2.resize(hm_content, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
    return hm_orig.astype(np.float32)


def _save_keypoints_on_original(
    img_rgb: np.ndarray,
    kp_gt_xy: np.ndarray,
    kp_pr_xy: np.ndarray,
    vis_k: np.ndarray,
    save_path: str,
    title: str = "",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    out = img_rgb.copy()

    for k in range(kp_gt_xy.shape[0]):
        if float(vis_k[k]) < 0.5:
            continue
        gx, gy = kp_gt_xy[k]
        px, py = kp_pr_xy[k]
        cv2.circle(out, (int(round(gx)), int(round(gy))), 4, (0, 255, 0), -1)   # GT
        cv2.circle(out, (int(round(px)), int(round(py))), 4, (255, 0, 0), -1)   # Pred
        cv2.line(out, (int(round(gx)), int(round(gy))), (int(round(px)), int(round(py))), (255, 255, 255), 1)

    if title:
        cv2.putText(out, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(save_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def _save_heatmap_on_original(
    img_rgb: np.ndarray,
    hm_mask01: np.ndarray,   # (H,W) in [0,1]
    save_path: str,
    title: str = "",
    alpha: float = 0.40,
    thresh: float = 0.03,
    blur_ksize: int = 0,
    gamma: float = 0.7,
):
    """
    FIXED: removes the crop "box" by using per-pixel alpha derived from hm_mask01.
    - Outside crop -> hm_mask ~ 0 -> alpha_map ~ 0 -> NO overlay.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    hm = hm_mask01.astype(np.float32)
    hm = np.clip(hm, 0.0, 1.0)

    # suppress low background so it doesn't tint large regions
        # Hard gate: below thresh -> zero alpha (prevents tinting the whole crop)
    if thresh > 0:
        hm[hm < thresh] = 0.0

    # soften edges (helps if crop boundary is sharp)
    if blur_ksize and blur_ksize > 1:
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        hm = cv2.GaussianBlur(hm, (blur_ksize, blur_ksize), 0)

    # alpha map (per-pixel), optionally non-linear
    hm_alpha = np.clip(hm, 0.0, 1.0) ** float(gamma)
    alpha_map = float(alpha) * hm_alpha  # (H,W) in [0,alpha]

    # colormap from heat intensity (not from alpha)
    hm_u8 = (hm * 255.0).clip(0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    img_f = img_rgb.astype(np.float32)
    a3 = alpha_map[..., None].astype(np.float32)

    out = (img_f * (1.0 - a3) + heat_rgb * a3).clip(0, 255).astype(np.uint8)

    if title:
        cv2.putText(out, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(save_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


# -------------------------
# Visualization core
# -------------------------
@torch.inference_mode()
def visualize_one_sequence(cfg, model, device: str, ds: SPADESDataset, seq_to_i: dict,
                           seq: str, tag: str, rng: np.random.RandomState,
                           pose_estimator, pnp_config, K_cam, dist_cam,
                           cuboid_corners_3d):
    eval_cfg = cfg["eval"]
    debug_dir = eval_cfg.get("debug_dir", "./vis_debug")
    os.makedirs(debug_dir, exist_ok=True)

    if seq not in seq_to_i:
        raise RuntimeError(f"[VIS] seq={seq} not found in dataset split.")

    full = ds[seq_to_i[seq]]
    frames_TCHW, kps_TK2, quat_T4, trans_T3, vis_TK, hm_TKHW, frame_ids, crop_T4, geom_T5, seq_name = full

    T_chunk = int(cfg["data"]["test_T_chunk"])
    clip = slice_random_clip_from_full_sequence(
        frames_TCHW, kps_TK2, quat_T4, trans_T3, vis_TK, hm_TKHW,
        frame_ids, crop_T4, geom_T5, T_chunk=T_chunk, rng=rng,
    )

    t0 = int(rng.randint(0, T_chunk))

    video_cpu = clip["frames"].unsqueeze(0)     # (1,T,3,S,S)
    kps_gt_norm = clip["kps"].unsqueeze(0)      # (1,T,K,2)
    vis = clip["vis"].unsqueeze(0)              # (1,T,K)
    crop_cpu = clip["crop"]                     # (T,4)
    geom_cpu = clip["geom"]                     # (T,5)
    fid = clip["frame_ids"][t0]

    data_root = cfg["data"]["root"]
    seq_fid, frame_idx = _parse_frame_id(fid)
    img_rgb, img_path = _read_original_rgb(data_root, seq_fid, frame_idx, ext=".png")
    H0, W0 = img_rgb.shape[:2]
    orig_hw = (H0, W0)


    video = video_cpu.to(device, non_blocking=True)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        out = model(video)

    hm_logits = out["heatmaps"].float().cpu()
    kps_pred_norm = out.get("keypoints", None)
    if kps_pred_norm is not None:
        kps_pred_norm = kps_pred_norm.float().cpu()

    adapter_tokens = out.get("adapter_tokens", None)
    out_tok = None
    if adapter_tokens is not None:
        tok = adapter_tokens
        if tok.dim() == 3:
            T_expected = int(cfg["data"]["test_T_chunk"])
            if tok.shape[0] == T_expected:
                tok = tok.unsqueeze(0)
            else:
                tok = tok.unsqueeze(0).unsqueeze(0)

        input_size = int(cfg["data"]["input_size"])
        patch_size = int(cfg["model"].get("patch_size", 16))
        Hp = input_size // patch_size
        Wp = input_size // patch_size
        P_expected = Hp * Wp

        if tok.shape[2] < P_expected:
            raise RuntimeError(f"[VIS] adapter_tokens too few tokens: got P={tok.shape[2]} expected >= {P_expected}")
        if tok.shape[2] > P_expected:
            tok = tok[:, :, -P_expected:, :]

        out_tok = {"adapter_tokens": tok.float().cpu()}

    del out, video
    torch.cuda.empty_cache()

    seq_dir = os.path.join(debug_dir, f"vis_{tag}_{seq}")
    os.makedirs(seq_dir, exist_ok=True)

    # Input-space outputs (unchanged)
    save_kp_overlay_bt(
        frames=video_cpu,
        kps_gt_norm=kps_gt_norm,
        hm_logits=hm_logits,
        vis=vis,
        cfg={"data": {"input_size": int(cfg["data"]["input_size"])}},
        save_path=os.path.join(seq_dir, f"t{t0:03d}_kp.png"),
        b=0, t=t0,
    )

    try:
        save_heatmap_overlay_bt(
            frames=video_cpu,
            hm_logits=hm_logits,
            vis=vis,
            cfg={"data": {"input_size": int(cfg["data"]["input_size"])}},
            save_path=os.path.join(seq_dir, f"t{t0:03d}_hm.png"),
            b=0, t=t0,
            reduce="max",
            use_vis=True,
            tau=1.0,
        )
    except Exception as e:
        print(f"[VIS] heatmap overlay failed for {seq}: {repr(e)}")

    if out_tok is not None:
        try:
            save_token_similarity_diagnostics_bt(
                out=out_tok,
                frames=video_cpu,
                kps_gt=kps_gt_norm,
                vis=vis,
                cfg={"data": {"input_size": int(cfg["data"]["input_size"])}, "model": cfg.get("model", {})},
                save_path=os.path.join(seq_dir, f"t{t0:03d}_tok.png"),
                b=0, t=t0,
                sigma_patch=float(eval_cfg.get("tok_sigma_patch", 0.35)),
                temp_pred=float(eval_cfg.get("tok_temp_pred", 0.07)),
                max_k_show=int(eval_cfg.get("tok_max_k_show", 8)),
            )
        except Exception as e:
            print(f"[VIS] token diagnostics failed for {seq}: {repr(e)}")

        try:
            save_dino_style_kp_attention_overlays_bt(
                out=out_tok,
                frames=video_cpu,
                kps_gt=kps_gt_norm,
                vis=vis,
                cfg={"data": {"input_size": int(cfg["data"]["input_size"])},
                     "model": cfg.get("model", {})},
                save_dir=seq_dir,
                b=0,
                t=t0,
                sigma_patch=float(eval_cfg.get("tok_sigma_patch", 0.35)),
                temp_pred=float(eval_cfg.get("tok_temp_pred", 0.07)),
                max_k_show=int(eval_cfg.get("tok_max_k_show", 8)),
                keep_mass=float(eval_cfg.get("attn_keep_mass", 0.60)),
                alpha=float(eval_cfg.get("attn_alpha", 0.45)),
            )
        except Exception as e:
            print(f"[VIS] DINO-style kp-attn failed for {seq}: {repr(e)}")

    # Original-space: KP and HM separately
    try:
        input_size = int(cfg["data"]["input_size"])
        heatmap_hw = tuple(cfg["model"].get("heatmap_out", [256, 256]))

        if kps_pred_norm is not None:
            kps_gt_in = _kps_norm_to_input_px(kps_gt_norm, input_size=input_size, heatmap_hw=heatmap_hw)
            kps_pr_in = _kps_norm_to_input_px(kps_pred_norm, input_size=input_size, heatmap_hw=heatmap_hw)

            crop_t4 = crop_cpu[t0].numpy()
            meta_t5 = geom_cpu[t0].numpy()
            vis_k = vis[0, t0].numpy()

            kps_gt_orig = _map_input_px_to_original(kps_gt_in[t0].numpy(), crop_t4, meta_t5, input_size, fallback_hw=orig_hw)
            kps_pr_orig = _map_input_px_to_original(kps_pr_in[t0].numpy(), crop_t4, meta_t5, input_size, fallback_hw=orig_hw)

            _save_keypoints_on_original(
                img_rgb=img_rgb,
                kp_gt_xy=kps_gt_orig,
                kp_pr_xy=kps_pr_orig,
                vis_k=vis_k,
                save_path=os.path.join(seq_dir, f"t{t0:03d}_orig_kp.png"),
                title=f"{seq} {tag} | frame={frame_idx} | KP (GT green, Pred red)",
            )

        crop_t4 = crop_cpu[t0].numpy()
        meta_t5 = geom_cpu[t0].numpy()

        hm_mask_orig = _heatmap_to_original_mask(
            hm_logits_bt=hm_logits,
            t0=t0,
            input_size=input_size,
            crop_xywh_norm_t4=crop_t4,
            meta_t5=meta_t5,
            orig_hw=orig_hw,
        )
        if hm_mask_orig.shape[:2] != (H0, W0):
            hm_mask_orig = cv2.resize(hm_mask_orig, (W0, H0), interpolation=cv2.INTER_LINEAR).astype(np.float32)

        # _save_heatmap_on_original(
        #     img_rgb=img_rgb,
        #     hm_mask01=hm_mask_orig,
        #     save_path=os.path.join(seq_dir, f"t{t0:03d}_orig_hm.png"),
        #     title=f"{seq} {tag} | frame={frame_idx} | HM overlay",
        #     alpha=float(eval_cfg.get("orig_hm_alpha", 0.40)),
        #     thresh=float(eval_cfg.get("orig_hm_thresh", 0.03)),
        #     blur_ksize=int(eval_cfg.get("orig_hm_blur", 0)),
        #     gamma=float(eval_cfg.get("orig_hm_gamma", 0.7)),
        # )

        # ============================================================
        # 6DoF visualization (PREDICTION ONLY) in ORIGINAL image space
        #   - compute pred 2D keypoints in FULL px (same as evaluate.py)
        #   - run PnP (top-k by kp score)
        #   - draw axes + projected 3D keypoints
        # ============================================================

        try:
            input_size = int(cfg["data"]["input_size"])
            heatmap_hw = tuple(cfg["model"].get("heatmap_out", [256, 256]))
            Hm, Wm = int(heatmap_hw[0]), int(heatmap_hw[1])

            # Decide kp source for pose: DSNT keypoints if present, else softargmax(heatmaps)
            kp_source = str(eval_cfg.get("kp_source", "dsnt")).lower()
            tau_now = float(eval_cfg.get("tau_now", eval_cfg.get("tau", 1.0)))
            tau_now = max(tau_now, 1e-4)

            crop_t4 = crop_cpu[t0].unsqueeze(0).unsqueeze(0)  # (1,1,4) torch
            geom_t5 = geom_cpu[t0].unsqueeze(0).unsqueeze(0)  # (1,1,5) torch  (letterbox_meta OR nan)

            # detect letterbox meta vs crop (same rule as eval)
            use_letterbox = torch.isfinite(geom_t5[..., 0]).all().item()
            letterbox_meta = geom_t5 if use_letterbox else None

            # ---- full-px predicted keypoints (1,1,K,2) ----
            if (kp_source == "dsnt") and (kps_pred_norm is not None):
                # kps_pred_norm is (1,T,K,2) heatmap-normalized, take t0
                kps_pred_1t = kps_pred_norm[:, t0:t0+1]  # (1,1,K,2)
                if use_letterbox:
                    pred_full = _kps_normhm_to_full_px_letterbox(
                        kps_norm_hm=kps_pred_1t,
                        letterbox_meta=letterbox_meta,
                        input_size=input_size,
                        heatmap_hw=(Hm, Wm),
                    )
                else:
                    pred_full = _kps_normhm_to_full_px(
                        kps_norm_hm=kps_pred_1t,
                        crop_xywh_norm=crop_t4,
                        original_size_wh=(W0, H0),  # (W_full,H_full)
                        input_size=input_size,
                        heatmap_hw=(Hm, Wm),
                    )
            else:
                hm_1t = hm_logits[:, t0:t0+1]  # (1,1,K,Hm,Wm)
                if use_letterbox:
                    pred_full = _heatmap_logits_to_full_px_letterbox(
                        hm_logits_bt=hm_1t,
                        letterbox_meta=letterbox_meta,
                        input_size=input_size,
                        heatmap_hw=(Hm, Wm),
                        tau_now=tau_now,
                    )
                else:
                    pred_full = _heatmap_logits_to_full_px(
                        hm_logits_bt=hm_1t,
                        crop_xywh_norm=crop_t4,
                        original_size_wh=(W0, H0),
                        input_size=input_size,
                        tau_now=tau_now,
                    )

            # ---- visibility + kp scores (1,1,K) ----
            vis_1t = vis[:, t0:t0+1].float()  # (1,1,K)
            kp_scores_1t = hm_logits[:, t0:t0+1].amax(dim=(-2, -1))  # (1,1,K)

            # select visible indices, then sort by score desc, take topk (same as eval)
            mask = (vis_1t[0, 0] > 0.5)
            idx_all = torch.where(mask)[0]
            pnp_topk = int(eval_cfg.get("pnp_topk", 8))

            if idx_all.numel() >= 4:
                scores_k = kp_scores_1t[0, 0]  # (K,)
                idx_sorted = idx_all[torch.argsort(scores_k[idx_all], descending=True)]
                vis_idx = idx_sorted[: min(pnp_topk, idx_sorted.numel())]

                if vis_idx.numel() >= 4:
                    q_list, t_list, ok_list = estimate_pose_from_keypoints(
                        points=pred_full[0, 0].unsqueeze(0),      # (1,K,2) full px
                        array=vis_idx.unsqueeze(0),               # (1,n)
                        pe=pose_estimator,
                        pnpconfig=pnp_config,
                    )
                    ok = bool(ok_list[0])

                    if ok:
                        q_xyzw = np.asarray(q_list[0], dtype=np.float64).reshape(4)  # (x,y,z,w)
                        t_xyz  = np.asarray(t_list[0], dtype=np.float64).reshape(3)

                        # xyzw -> wxyz
                        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)

                        out_pose = draw_3d_cuboid_on_original(
                            img_rgb=img_rgb,
                            K=K_cam,
                            dist=dist_cam,
                            q_wxyz=q_wxyz,
                            t_xyz=t_xyz,
                            cuboid_corners_3d=cuboid_corners_3d,
                            color_bgr=(255, 0, 255),
                            thickness=int(eval_cfg.get("vis_box_th", 1)),
                        )

                        cv2.imwrite(
                            os.path.join(seq_dir, f"t{t0:03d}_orig_6dof_box.png"),
                            cv2.cvtColor(out_pose, cv2.COLOR_RGB2BGR),
                        )

                        # optional: project & draw the 3D model keypoints (from pose_estimator)
                        if bool(eval_cfg.get("vis_proj_model_kps", True)):
                            out_pose = draw_projected_model_keypoints_on_original(
                                img_rgb=out_pose,
                                K=K_cam,
                                dist=dist_cam,
                                model_pts_3d=pose_estimator.model_points,  # (K,3), already scaled
                                q_wxyz=q_wxyz,
                                t_xyz=t_xyz,
                                color_rgb=(0, 255, 255),
                                radius=int(eval_cfg.get("vis_proj_kp_r", 3)),
                            )

                        cv2.imwrite(
                            os.path.join(seq_dir, f"t{t0:03d}_orig_6dof.png"),
                            cv2.cvtColor(out_pose, cv2.COLOR_RGB2BGR),
                        )
                    else:
                        print(f"[VIS] PnP failed for {seq} frame={frame_idx} (not saving 6DoF overlay).")
            else:
                print(f"[VIS] Not enough visible keypoints for PnP: n={idx_all.numel()} (need >=4).")

        except Exception as e:
            print(f"[VIS] 6DoF overlay failed for {seq}: {repr(e)}")

    except Exception as e:
        print(f"[VIS] original-space overlays failed for {seq}: {repr(e)}")

    print(f"[VIS] {seq} {tag}: full_T={frames_TCHW.shape[0]}, clip=[{clip['start']}:{clip['end']}], t0={t0} -> {seq_dir}")



def main(cfg_path: str):
    cfg = load_cfg(cfg_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TOP5 = ["RT037", "RT060", "RT068", "RT019", "RT007"] #sequences for ad8_crop_vitL and no crop_vitL
    # BOT5 = ["RT165", "RT180", "RT232", "RT235", "RT137"]

    # TOP5 = ["RT037", "RT007", "RT082", "RT044", "RT110"] # for ad12_no_crop_vitL
    # BOT5 = ["RT165", "RT180", "RT235", "RT121", "RT232"]

    TOP5 = ["RT237", "RT179", "RT060", "RT250", "RT037"] # for 8 adapters_vits+_crop
    BOT5 = ["RT180", "RT165", "RT121", "RT023", "RT043"]
    seed = int(cfg["eval"].get("vis_seed", 256))
    rng = np.random.RandomState(seed)

    ds = build_spades_test_dataset(cfg)
    seq_to_i = build_seq_index_map(ds)

    model = build_model_for_vis(cfg)
    ckpt_path = cfg["eval"]["load_from"]
    load_ckpt(model, ckpt_path, device=device)

    # ------------------------------------------------------------
    # Camera + PoseEstimator (needed for 6DoF visualization)
    # ------------------------------------------------------------
    cam_json = cfg["data"].get("cam_intrinsics_json", None)
    if cam_json is None:
        raise KeyError("cfg['data']['cam_intrinsics_json'] is required for 6DoF visualization.")

    points_path = os.path.join(cfg["data"]["root"], "points.json")

    pose_estimator, pnp_config, scale = create_pose_estimator(
        pnp_kpts=int(cfg["data"]["num_kp"]),
        camera_path=cam_json,
        points_path=points_path,
    )

    cuboid_corners_3d = get_pose_estimator_cuboid(pose_estimator)

    K_cam, dist_cam = _load_camera_from_json(cam_json)

    for s in TOP5:
        visualize_one_sequence(cfg, model, device, ds, seq_to_i, seq=s, tag="TOP", rng=rng,
                      pose_estimator=pose_estimator, pnp_config=pnp_config, K_cam=K_cam, dist_cam=dist_cam,
                      cuboid_corners_3d=cuboid_corners_3d)

    for s in BOT5:
        visualize_one_sequence(cfg, model, device, ds, seq_to_i, seq=s, tag="BOT", rng=rng,
                      pose_estimator=pose_estimator, pnp_config=pnp_config, K_cam=K_cam, dist_cam=dist_cam,
                      cuboid_corners_3d=cuboid_corners_3d)


if __name__ == "__main__":
    main("./test_config.yaml")