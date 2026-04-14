# vis_video_keypoints_fullspace.py
# ------------------------------------------------------------
# Input: a SPADES-style sequence (so we have GT trajectories + crop/letterbox meta)
# Output: a VIDEO where each original frame is overlaid with:
#   - GT keypoints (green)
#   - Pred keypoints (red)
# in FULL IMAGE COORDINATES (original resolution)
#
# Robustness:
# - frame_ids can be nested => _flatten_frame_ids()
# - original frames can be .png/.jpg/.jpeg (any case) => _read_original_bgr() tries all
# ------------------------------------------------------------

import os
import re
import yaml
import cv2
import torch
import numpy as np
from scipy.ndimage import uniform_filter1d

from dataset import SPADESDataset, SPARKDataset
from model_hm_dsnt_2 import SpacecraftHMDSNT

from utils_new import (
    _kps_normhm_to_full_px_letterbox,
    _kps_normhm_to_full_px,
    _heatmap_logits_to_full_px_letterbox,
    _heatmap_logits_to_full_px,
)

# -------------------------
# Config
# -------------------------
def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg.setdefault("eval", {})
    cfg.setdefault("loss", {})
    return cfg


# -------------------------
# Model (same as your vis)
# -------------------------
def build_model_for_vis(cfg: dict):
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    T = int(data_cfg["test_T_chunk"])  # model forward expects exactly this T
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
    adapter_mode = model_cfg.get("adapter_mode", "pre_mhsa")

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
        adapter_mode=adapter_mode,
    )
    return model


def load_ckpt(model, ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt

    # Remap checkpoint keys to match current model naming
    remapped = {}
    for k, v in state.items():
        if k.startswith("backbone.adapters."):
            new_k = "backbone.attn_adapters." + k[len("backbone.adapters."):]
            remapped[new_k] = v
        elif k.startswith("backbone.ffn_adapters."):
            suffix = k[len("backbone.ffn_adapters."):]
            suffix = suffix.replace(".core.", ".", 1)
            remapped["backbone.ffn_adapters." + suffix] = v
        else:
            remapped[k] = v
    state = remapped

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[VIDVIS] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        print("  unexpected (first 20):", unexpected[:20])
    if missing:
        print("  missing (first 20):", missing[:20])
    model.to(device).eval()
    return ckpt


# -------------------------
# Dataset access (full sequence)
# -------------------------
def build_dataset(cfg: dict):
    d = cfg["data"]
    ds_name = d.get("name", "").lower()
    base_kwargs = dict(
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
    if ds_name == "spark":
        return SPARKDataset(**base_kwargs)
    return SPADESDataset(**base_kwargs)


def build_seq_index_map(ds) -> dict:
    seq_to_i = {}
    for i, entry in enumerate(ds.dataset):
        seq_to_i[entry[0]] = i
    return seq_to_i


# -------------------------
# Original image reader
# -------------------------
_FRAME_RE = re.compile(r"^(RT\d+)[_\-](\d+)$")

# IMPORTANT: must be a tuple, not a string
_IMG_EXTS = (".jpg",)  # SPARK images are .jpg


def _parse_frame_id(fid: str):
    m = _FRAME_RE.match(str(fid))
    if m is None:
        raise ValueError(f"[VIDVIS] Unexpected frame_id format: {fid}")
    return m.group(1), int(m.group(2))


def _resolve_imgs_path(root: str, seq_name: str):
    p1 = os.path.join(root, "images", seq_name)
    if os.path.isdir(p1):
        return p1
    return os.path.join(root, seq_name)


def _read_bgr_anyext(path_no_ext: str):
    """
    Try reading an image given a path WITHOUT extension.
    Returns (bgr, used_path).
    """
    tried = []
    for ext in _IMG_EXTS:
        p = path_no_ext + ext
        tried.append(p)
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is not None:
            return bgr, p
    raise FileNotFoundError("Could not read image. Tried:\n" + "\n".join(tried))


def _read_original_bgr(root: str, seq_name: str, frame_idx: int):
    imgs_path = _resolve_imgs_path(root, seq_name)
    stem = f"img{str(frame_idx).zfill(3)}_{seq_name}"
    path_no_ext = os.path.join(imgs_path, stem)
    return _read_bgr_anyext(path_no_ext)

def _flatten_frame_ids(frame_ids):
    """
    SPADES sometimes returns frame_ids as:
      - list[str]
      - tuple[str]
      - [list[str]]  (nested once)
      - (list[str],) (nested once)
    This returns a flat list[str].
    """
    if isinstance(frame_ids, (list, tuple)) and len(frame_ids) == 1 and isinstance(frame_ids[0], (list, tuple)):
        frame_ids = frame_ids[0]
    return list(frame_ids)


# -------------------------
# Post-hoc temporal smoothing
# -------------------------
def smooth_keypoints(kps_np: np.ndarray, vis_np: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Apply 1D uniform filter to predicted keypoint trajectories.
    kps_np: (T, K, 2) full-pixel coordinates
    vis_np: (T, K) visibility mask
    window: smoothing window size (odd recommended)
    Returns: smoothed (T, K, 2)
    """
    if window <= 1:
        return kps_np
    smoothed = np.copy(kps_np)
    T, K, _ = kps_np.shape
    for k in range(K):
        # Only smooth segments where the keypoint is visible
        vis_k = vis_np[:, k] > 0.5
        if vis_k.sum() < window:
            continue
        for d in range(2):
            traj = kps_np[:, k, d].copy()
            # Replace invisible frames with nearest visible value to avoid boundary artifacts
            for t in range(T):
                if not vis_k[t]:
                    # find nearest visible frame
                    dists = np.abs(np.arange(T) - t)
                    dists[~vis_k] = T + 1
                    traj[t] = traj[dists.argmin()]
            filtered = uniform_filter1d(traj, size=window, mode='nearest')
            # Only write back visible frames
            smoothed[vis_k, k, d] = filtered[vis_k]
    return smoothed


# -------------------------
# Drawing (full image)
# -------------------------
def draw_kps_fullspace_bgr(
    bgr: np.ndarray,
    gt_xy: np.ndarray,   # (K,2) full px
    pr_xy: np.ndarray,   # (K,2) full px
    vis_k: np.ndarray,   # (K,) {0,1}
    r: int = 4,
    th: int = 1,
):
    """
    GT: green, Pred: red, line GT->Pred white.
    """
    out = bgr.copy()
    K = gt_xy.shape[0]

    for k in range(K):
        if float(vis_k[k]) < 0.5:
            continue

        gx, gy = float(gt_xy[k, 0]), float(gt_xy[k, 1])
        px, py = float(pr_xy[k, 0]), float(pr_xy[k, 1])

        cv2.circle(out, (int(round(gx)), int(round(gy))), r, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (int(round(px)), int(round(py))), r, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.line(
            out,
            (int(round(gx)), int(round(gy))),
            (int(round(px)), int(round(py))),
            (255, 255, 255),
            th,
            lineType=cv2.LINE_AA,
        )

    return out


# -------------------------
# Full-space mapping using your existing utilities
# -------------------------
@torch.no_grad()
def _kps_normhm_to_full_px_bt(
    kps_norm_hm_bt: torch.Tensor,     # (1,T,K,2) in [-1,1] heatmap-norm coords
    crop_xywh_norm_bt: torch.Tensor,  # (1,T,4) normalized crop (if crop pipeline) OR values that indicate no-crop
    geom_bt: torch.Tensor,            # (1,T,5) letterbox meta OR NaNs
    original_size_wh: tuple,          # (W0,H0)
    input_size: int,
    heatmap_hw: tuple,                # (Hm,Wm)
    kp_source: str,
    hm_logits_bt: torch.Tensor = None,  # (1,T,K,Hm,Wm) if kp_source != dsnt
    tau_now: float = 1.0,
):
    """
    Returns: full_px (1,T,K,2) torch in full-image pixels
    """
    T = int(kps_norm_hm_bt.shape[1])
    W0, H0 = int(original_size_wh[0]), int(original_size_wh[1])
    Hm, Wm = int(heatmap_hw[0]), int(heatmap_hw[1])

    out_full = []
    for t in range(T):
        crop_t = crop_xywh_norm_bt[:, t:t+1]  # (1,1,4)
        geom_t = geom_bt[:, t:t+1]            # (1,1,5)

        use_letterbox = torch.isfinite(geom_t[..., 0]).all().item()
        letterbox_meta = geom_t if use_letterbox else None

        if (kp_source == "dsnt") and (kps_norm_hm_bt is not None):
            kps_t = kps_norm_hm_bt[:, t:t+1]  # (1,1,K,2)
            if use_letterbox:
                full_t = _kps_normhm_to_full_px_letterbox(
                    kps_norm_hm=kps_t,
                    letterbox_meta=letterbox_meta,
                    input_size=input_size,
                    heatmap_hw=(Hm, Wm),
                )
            else:
                full_t = _kps_normhm_to_full_px(
                    kps_norm_hm=kps_t,
                    crop_xywh_norm=crop_t,
                    original_size_wh=(W0, H0),
                    input_size=input_size,
                    heatmap_hw=(Hm, Wm),
                )
        else:
            if hm_logits_bt is None:
                raise ValueError("hm_logits_bt is required when kp_source != 'dsnt'")
            hm_t = hm_logits_bt[:, t:t+1]  # (1,1,K,Hm,Wm)
            if use_letterbox:
                full_t = _heatmap_logits_to_full_px_letterbox(
                    hm_logits_bt=hm_t,
                    letterbox_meta=letterbox_meta,
                    input_size=input_size,
                    heatmap_hw=(Hm, Wm),
                    tau_now=max(float(tau_now), 1e-4),
                )
            else:
                full_t = _heatmap_logits_to_full_px(
                    hm_logits_bt=hm_t,
                    crop_xywh_norm=crop_t,
                    original_size_wh=(W0, H0),
                    input_size=input_size,
                    tau_now=max(float(tau_now), 1e-4),
                )

        out_full.append(full_t[:, 0])  # (1,K,2)

    out_full = torch.stack(out_full, dim=1)  # (1,T,K,2)
    return out_full


# -------------------------
# Main video rendering
# -------------------------
@torch.inference_mode()
def render_sequence_to_video(cfg: dict, model, ds, seq_name: str, save_path: str):
    seq_to_i = build_seq_index_map(ds)
    if seq_name not in seq_to_i:
        raise RuntimeError(f"[VIDVIS] seq={seq_name} not found in dataset split ({cfg['eval'].get('mode','test')}).")

    device = next(model.parameters()).device
    droot = cfg["data"]["root"]
    input_size = int(cfg["data"]["input_size"])
    T_chunk = int(cfg["data"]["test_T_chunk"])
    heatmap_hw = tuple(cfg["model"].get("heatmap_out", [256, 256]))
    Hm, Wm = int(heatmap_hw[0]), int(heatmap_hw[1])

    eval_cfg = cfg["eval"]
    kp_source = str(eval_cfg.get("kp_source", "dsnt")).lower()
    tau_now = float(eval_cfg.get("tau_now", eval_cfg.get("tau", 1.0)))
    r = int(eval_cfg.get("video_kp_radius", 4))
    th = int(eval_cfg.get("video_kp_line_th", 1))

    # load full sequence tensors from dataset
    full = ds[seq_to_i[seq_name]]
    frames_TCHW, kps_gt_norm_TK2, quat_T4, trans_T3, vis_TK, hm_TKHW, frame_ids, crop_T4, geom_T5, _ = full
    frame_ids = _flatten_frame_ids(frame_ids)

    T_full = int(frames_TCHW.shape[0])
    if T_full < 1:
        raise RuntimeError("[VIDVIS] Empty sequence")

    # Read first original frame to get W0,H0 + initialize writer (robust ext)
    seq_fid0, frame_idx0 = _parse_frame_id(frame_ids[0])
    bgr0, used0 = _read_original_bgr(droot, seq_fid0, frame_idx0)
    H0, W0 = bgr0.shape[:2]

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fps = float(eval_cfg.get("video_fps", 30.0))
    fourcc = cv2.VideoWriter_fourcc(*str(eval_cfg.get("video_fourcc", "mp4v")))
    writer = cv2.VideoWriter(save_path, fourcc, fps, (W0, H0))
    if not writer.isOpened():
        raise RuntimeError(f"[VIDVIS] Could not open VideoWriter at: {save_path}")

    smooth_window = int(eval_cfg.get("smooth_window", 5))
    print(f"[VIDVIS] Writing: {save_path} | seq={seq_name} | frames={T_full} | size={W0}x{H0} | fps={fps} | smooth={smooth_window}")
    print(f"[VIDVIS] First frame resolved: {used0}")

    def _pad_to_T(x: torch.Tensor, T: int, dim0_time: int = 0):
        tcur = x.shape[dim0_time]
        if tcur == T:
            return x
        if tcur > T:
            return x.narrow(dim0_time, 0, T)
        pad_n = T - tcur
        last = x.narrow(dim0_time, tcur - 1, 1)
        reps = [1] * x.dim()
        reps[dim0_time] = pad_n
        pad = last.repeat(*reps)
        return torch.cat([x, pad], dim=dim0_time)

    # --- Pass 1: collect all GT/pred keypoints in full-pixel space ---
    all_gt = []    # list of (tlen, K, 2) arrays
    all_pr = []
    all_vis = []

    for start in range(0, T_full, T_chunk):
        end = min(T_full, start + T_chunk)
        tlen = end - start

        video_chunk = frames_TCHW[start:end]
        kps_gt_chunk = kps_gt_norm_TK2[start:end]
        vis_chunk = vis_TK[start:end]
        crop_chunk = crop_T4[start:end]
        geom_chunk = geom_T5[start:end]
        fids_chunk = frame_ids[start:end]

        video_in = _pad_to_T(video_chunk, T_chunk, dim0_time=0).unsqueeze(0)
        kps_gt_in = _pad_to_T(kps_gt_chunk, T_chunk, dim0_time=0).unsqueeze(0)
        vis_in = _pad_to_T(vis_chunk, T_chunk, dim0_time=0).unsqueeze(0)
        crop_in = _pad_to_T(crop_chunk, T_chunk, dim0_time=0).unsqueeze(0)
        geom_in = _pad_to_T(geom_chunk, T_chunk, dim0_time=0).unsqueeze(0)

        video_in = video_in.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(video_in)

        hm_logits_bt = out["heatmaps"].float()
        kps_pred_norm_bt = out.get("keypoints", None)
        if kps_pred_norm_bt is not None:
            kps_pred_norm_bt = kps_pred_norm_bt.float()

        # confirm original size is stable (robust ext)
        seq_fid, frame_idx = _parse_frame_id(fids_chunk[0])
        bgr_first, _ = _read_original_bgr(droot, seq_fid, frame_idx)
        Hc, Wc = bgr_first.shape[:2]
        assert (Hc, Wc) == (H0, W0), "[VIDVIS] Original frame size changed within sequence."

        gt_full = _kps_normhm_to_full_px_bt(
            kps_norm_hm_bt=kps_gt_in.cpu(),
            crop_xywh_norm_bt=crop_in.cpu(),
            geom_bt=geom_in.cpu(),
            original_size_wh=(W0, H0),
            input_size=input_size,
            heatmap_hw=(Hm, Wm),
            kp_source="dsnt",
            hm_logits_bt=None,
            tau_now=tau_now,
        )

        pred_full = _kps_normhm_to_full_px_bt(
            kps_norm_hm_bt=kps_pred_norm_bt.cpu() if kps_pred_norm_bt is not None else None,
            crop_xywh_norm_bt=crop_in.cpu(),
            geom_bt=geom_in.cpu(),
            original_size_wh=(W0, H0),
            input_size=input_size,
            heatmap_hw=(Hm, Wm),
            kp_source=kp_source,
            hm_logits_bt=hm_logits_bt.cpu(),
            tau_now=tau_now,
        )

        all_gt.append(gt_full[0, :tlen].numpy())
        all_pr.append(pred_full[0, :tlen].numpy())
        all_vis.append(vis_in[0, :tlen].cpu().numpy())

        del out, hm_logits_bt, kps_pred_norm_bt
        torch.cuda.empty_cache()

    # Concatenate across all chunks: (T_full, K, 2) and (T_full, K)
    all_gt_np = np.concatenate(all_gt, axis=0)
    all_pr_np = np.concatenate(all_pr, axis=0)
    all_vis_np = np.concatenate(all_vis, axis=0)

    # --- Apply temporal smoothing to predictions ---
    if smooth_window > 1:
        all_pr_np = smooth_keypoints(all_pr_np, all_vis_np, window=smooth_window)
        print(f"[VIDVIS] Applied temporal smoothing (window={smooth_window})")

    # --- Pass 2: render video ---
    for t in range(T_full):
        seq_fid_t, frame_idx_t = _parse_frame_id(frame_ids[t])
        bgr, _ = _read_original_bgr(droot, seq_fid_t, frame_idx_t)

        out_bgr = draw_kps_fullspace_bgr(
            bgr=bgr,
            gt_xy=all_gt_np[t],
            pr_xy=all_pr_np[t],
            vis_k=all_vis_np[t],
            r=r,
            th=th,
        )
        writer.write(out_bgr)

    writer.release()
    print(f"[VIDVIS] Done: {save_path}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="./test_spark.yaml", help="path to config yaml")
    args = ap.parse_args()

    # Example list (edit as needed)
    Spark_real = ["RT001", "RT002", "RT003", "RT004"]

    cfg = load_cfg(args.cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = build_dataset(cfg)

    model = build_model_for_vis(cfg)
    ckpt_path = cfg["eval"]["load_from"]
    load_ckpt(model, ckpt_path, device=device)

    for seq_name in Spark_real:
        out_path = os.path.join("./eval_debug", f"{seq_name}_ViT-L_kp_vis.mp4")
        render_sequence_to_video(cfg, model, ds, seq_name=seq_name, save_path=out_path)


if __name__ == "__main__":
    main()