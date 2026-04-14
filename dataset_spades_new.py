# dataset_spades_new.py  –  Test-only dataset for SPADES trajectories
#
# Directory layout expected:
#   gt_folder/
#     001/  002/  ...  012/          ← image folders
#     ground_truth.json              ← single annotation file
#
# ground_truth.json format (list of entries):
#   [
#     {
#       "filename": "001/1685017634_484836341.png",
#       "keypoints": [x0,y0, x1,y1, ...],   # flat, 8 kps → 16 values (no visibility)
#       "r_xyz":  [...],
#       "q_xyzw": [...]
#     }, ...
#   ]

import json
import os
import os.path as osp
from typing import Tuple, List, Optional

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import albumentations as A
import albumentations.augmentations.geometric.resize as albu_resize

# ──────────────────────────────────────────────────────────────────────────────
# Shared utilities (kept self-contained)
# ──────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

def to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def imagenet_normalize_tensor(t: torch.Tensor) -> torch.Tensor:
    return (t - IMAGENET_MEAN.to(t.device)) / IMAGENET_STD.to(t.device)

def make_gaussian_heatmaps(
    kps_hw: torch.Tensor, H: int = 56, W: int = 56, sigma_px: float = 1.8
) -> torch.Tensor:
    """kps_hw: (T,K,2) in heatmap pixel coords → (T,K,H,W)"""
    device = kps_hw.device
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    maps = []
    for t in range(kps_hw.shape[0]):
        frame_maps = []
        for k in range(kps_hw.shape[1]):
            x, y = kps_hw[t, k, 0], kps_hw[t, k, 1]
            g = torch.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2.0 * sigma_px ** 2))
            frame_maps.append(g / (g.sum() + 1e-8))
        maps.append(torch.stack(frame_maps, dim=0))
    return torch.stack(maps, dim=0)  # (T,K,H,W)

def clamp_bbox(xmin, ymin, xmax, ymax, W, H):
    xmin = max(0, min(int(xmin), W - 1))
    ymin = max(0, min(int(ymin), H - 1))
    xmax = max(0, min(int(xmax), W - 1))
    ymax = max(0, min(int(ymax), H - 1))
    if xmax <= xmin: xmax = min(W - 1, xmin + 1)
    if ymax <= ymin: ymax = min(H - 1, ymin + 1)
    return xmin, ymin, xmax, ymax

def bbox_from_keypoints(
    kps_xy: np.ndarray,
    img_w: int,
    img_h: int,
    padding_abs: int = 20,       # absolute pixel padding on each side
    padding_rel: float = 0.15,   # relative padding as fraction of kp span
    make_square: bool = True,
    min_size: int = 32,
) -> Tuple[int, int, int, int]:
    """
    Derive a crop bounding box purely from keypoints.

    padding_abs  – minimum pixel margin added on every side
    padding_rel  – additional margin = padding_rel * max(kp_span_x, kp_span_y)
    """
    if kps_xy.size == 0:
        return clamp_bbox(0, 0, img_w - 1, img_h - 1, img_w, img_h)

    kx, ky = kps_xy[:, 0], kps_xy[:, 1]
    x1_k, x2_k = float(np.min(kx)), float(np.max(kx))
    y1_k, y2_k = float(np.min(ky)), float(np.max(ky))

    span = max(x2_k - x1_k, y2_k - y1_k, 1.0)
    pad = padding_abs + padding_rel * span

    x1 = x1_k - pad
    y1 = y1_k - pad
    x2 = x2_k + pad
    y2 = y2_k + pad

    bw = max(x2 - x1, float(min_size))
    bh = max(y2 - y1, float(min_size))

    if make_square:
        side = max(bw, bh)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        x1, y1 = cx - side / 2, cy - side / 2
        x2, y2 = cx + side / 2, cy + side / 2

    return clamp_bbox(x1, y1, x2, y2, img_w, img_h)


# ──────────────────────────────────────────────────────────────────────────────
# Main dataset
# ──────────────────────────────────────────────────────────────────────────────

class SpadesNewDataset(data.Dataset):
    """
    Test-only dataset for SPADES trajectories 001-012.

    Args:
        gt_folder        : root folder containing trajectory sub-dirs and
                           ground_truth.json
        input_size       : spatial resolution fed to the model (square)
        sampling_rate    : keep every N-th annotated frame per trajectory
                           (1 = every frame, 2 = every other frame, …)
        heatmap_out      : (H, W) of output heatmap heads
        gaussian_sigma_px: sigma for Gaussian heatmap (in heatmap pixels)
        bbox_pad_abs     : absolute pixel padding when building bbox from kps
        bbox_pad_rel     : relative padding fraction when building bbox from kps
    """

    def __init__(
        self,
        gt_folder: str,
        input_size: int = 224,
        sampling_rate: int = 1,
        heatmap_out: Tuple[int, int] = (56, 56),
        gaussian_sigma_px: float = 2.5,
        bbox_pad_abs: int = 20,
        bbox_pad_rel: float = 0.15,
    ):
        self.gt_folder        = gt_folder
        self.input_size       = int(input_size)
        self.sampling_rate    = max(1, int(sampling_rate))
        self.hm_size          = tuple(heatmap_out)
        self.hm_sigma_px      = float(gaussian_sigma_px)
        self.bbox_pad_abs     = int(bbox_pad_abs)
        self.bbox_pad_rel     = float(bbox_pad_rel)

        # Load and index annotations
        ann_path = osp.join(gt_folder, "ground_truth_gt1-12.json")
        with open(ann_path, "r") as f:
            raw = json.load(f)

        # Build per-trajectory ordered annotation lists
        # traj_id → [ {filename, keypoints, r_xyz, q_xyzw}, … ]  (sorted by filename)
        # Trajectories we need 
        TRAJ_IDS = [f"{i:03d}" for i in range(1, 13)]   # "001" … "012"
        self._traj_entries: dict = {tid: [] for tid in TRAJ_IDS}
        for entry in raw:
            fname = entry["filename"]                 # e.g. "001/1685017634_484836341.png"
            tid   = fname.split("/")[0]               # "001"
            if tid in self._traj_entries:
                self._traj_entries[tid].append(entry)

        for tid in TRAJ_IDS:
            self._traj_entries[tid].sort(key=lambda e: e["filename"])

        # Build per-trajectory dataset index: list of (traj_id, [entries])
        # applying sampling_rate per trajectory
        self.dataset: List[Tuple[str, List[dict]]] = []
        for tid in TRAJ_IDS:
            entries = self._traj_entries[tid]
            sampled = entries[:: self.sampling_rate]
            if sampled:
                self.dataset.append((tid, sampled))

    # ------------------------------------------------------------------ helpers

    def _read_frame(self, path: str) -> np.ndarray:
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return to_rgb(img_bgr)

    def _parse_kps(self, entry: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          xy  : (K, 2) float32 keypoint pixel coordinates
          vis : (K,)   float32 – all 1.0 (no visibility flag in this format)
        """
        flat = np.asarray(entry["keypoints"], dtype=np.float32)
        # flat has 16 values → 8 keypoints (x,y) pairs
        K = flat.shape[0] // 2
        xy  = flat.reshape(K, 2)
        vis = np.ones(K, dtype=np.float32)
        return xy, vis

    def _preprocess(
        self,
        frame_rgb: np.ndarray,
        kp_xy: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[float], List[float]]:
        """
        1. Build bbox from keypoints.
        2. Crop to bbox.
        3. Resize to (input_size, input_size).

        Returns:
          img_r          : (input_size, input_size, 3) uint8
          kp_r           : (K, 2) float32 in input_size pixel space
          crop_xywh_norm : [x0/W, y0/H, w/W, h/H] normalised to original frame
          meta           : [nan, nan, nan, W, H]  (scale/pad unused for crop mode)
        """
        H, W, _ = frame_rgb.shape

        x1, y1, x2, y2 = bbox_from_keypoints(
            kp_xy, W, H,
            padding_abs=self.bbox_pad_abs,
            padding_rel=self.bbox_pad_rel,
            make_square=True,
        )

        transform = A.Compose(
            [
                A.Crop(x_min=x1, y_min=y1, x_max=x2, y_max=y2),
                albu_resize.Resize(self.input_size, self.input_size, p=1.0),
            ],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )

        out  = transform(image=frame_rgb, keypoints=kp_xy)
        img_r = out["image"]
        kp_r  = np.asarray(out["keypoints"], dtype=np.float32)

        crop_w = x2 - x1
        crop_h = y2 - y1
        crop_xywh_norm = [x1 / W, y1 / H, crop_w / W, crop_h / H]
        meta = [float("nan"), float("nan"), float("nan"), float(W), float(H)]

        return img_r, kp_r, crop_xywh_norm, meta

    # ------------------------------------------------------------------ Dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """
        Returns a *full-trajectory* item so that collate_temporal_chunks can
        chunk it into windows of size T_chunk.

        Batch fields:
          frames        : (T, 3, H, W)   ImageNet-normalised float32
          kps_norm      : (T, K, 2)      normalised to [-1, 1] in heatmap coords
          gt_q          : (T, 4)         quaternion xyzw
          gt_r          : (T, 3)         translation xyz
          vis           : (T, K)         visibility (all 1.0)
          hm_target     : (T, K, Hm, Wm) Gaussian heatmaps
          frame_ids     : [str]          list length T of "{traj}_{filename_stem}"
          crop_xywh     : (T, 4)         normalised crop box
          meta          : (T, 5)         [scale, pad_l, pad_t, W_orig, H_orig]
          seq_name      : str            trajectory id e.g. "001"
        """
        tid, entries = self.dataset[idx]

        Hm, Wm = self.hm_size
        inp = float(self.input_size)

        all_frames, all_kps_norm, all_gt_q, all_gt_r = [], [], [], []
        all_vis, all_hm, all_crop_xywh, all_meta, all_frame_ids = [], [], [], [], []

        for entry in entries:
            fname = entry["filename"]                        # "001/1685017634_….png"
            traj_dir, img_name = fname.split("/", 1)
            img_path = osp.join(self.gt_folder, traj_dir, "images", img_name)
            frame_rgb = self._read_frame(img_path)

            kp_xy, vis_np = self._parse_kps(entry)
            img_r, kp_r, crop_xywh_norm, meta = self._preprocess(frame_rgb, kp_xy)

            img_t = torch.from_numpy(img_r).permute(2, 0, 1).float() / 255.0
            img_t = imagenet_normalize_tensor(img_t)                            # (3,H,W)

            gt_q_t = torch.tensor(entry["q_xyzw"], dtype=torch.float32)        # (4,)
            gt_r_t = torch.tensor(entry["r_xyz"],  dtype=torch.float32)        # (3,)
            vis_t  = torch.from_numpy(vis_np)                                   # (K,)

            crop_t = torch.tensor(crop_xywh_norm, dtype=torch.float32)         # (4,)
            meta_t = torch.tensor(meta,            dtype=torch.float32)         # (5,)

            kps_img = torch.from_numpy(kp_r).float()                           # (K,2)
            kps_hm  = kps_img.clone()
            kps_hm[..., 0] = kps_hm[..., 0] * ((Wm - 1.0) / (inp - 1.0))
            kps_hm[..., 1] = kps_hm[..., 1] * ((Hm - 1.0) / (inp - 1.0))

            hm_t = make_gaussian_heatmaps(
                kps_hm.unsqueeze(0), H=Hm, W=Wm, sigma_px=self.hm_sigma_px
            ).squeeze(0)                                                        # (K,Hm,Wm)

            x_norm = 2.0 * (kps_hm[..., 0] / max(Wm - 1.0, 1.0)) - 1.0
            y_norm = 2.0 * (kps_hm[..., 1] / max(Hm - 1.0, 1.0)) - 1.0
            kps_norm_t = torch.stack([x_norm, y_norm], dim=-1)                 # (K,2)

            stem = osp.splitext(osp.basename(fname))[0]

            all_frames.append(img_t)
            all_kps_norm.append(kps_norm_t)
            all_gt_q.append(gt_q_t)
            all_gt_r.append(gt_r_t)
            all_vis.append(vis_t)
            all_hm.append(hm_t)
            all_crop_xywh.append(crop_t)
            all_meta.append(meta_t)
            all_frame_ids.append(f"{tid}_{stem}")

        frames    = torch.stack(all_frames,    dim=0)   # (T,3,H,W)
        kps_norm  = torch.stack(all_kps_norm,  dim=0)   # (T,K,2)
        gt_q      = torch.stack(all_gt_q,      dim=0)   # (T,4)
        gt_r      = torch.stack(all_gt_r,      dim=0)   # (T,3)
        vis       = torch.stack(all_vis,        dim=0)   # (T,K)
        hm_target = torch.stack(all_hm,         dim=0)   # (T,K,Hm,Wm)
        crop_xywh = torch.stack(all_crop_xywh,  dim=0)   # (T,4)
        meta_out  = torch.stack(all_meta,        dim=0)   # (T,5)

        return frames, kps_norm, gt_q, gt_r, vis, hm_target, all_frame_ids, crop_xywh, meta_out, tid


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--gt_folder", required=True)
    p.add_argument("--sampling_rate", type=int, default=5)
    p.add_argument("--input_size",    type=int, default=224)
    args = p.parse_args()

    ds = SpadesNewDataset(
        gt_folder     = args.gt_folder,
        input_size    = args.input_size,
        sampling_rate = args.sampling_rate,
        heatmap_out   = (56, 56),
        gaussian_sigma_px = 2.5,
        bbox_pad_abs  = 20,
        bbox_pad_rel  = 0.15,
    )

    print(f"Total samples (after sampling_rate={args.sampling_rate}): {len(ds)}")

    frames, kps_norm, gt_q, gt_r, vis, hm, frame_ids, crop_xywh, meta, seq = ds[0]
    print(f"  seq          : {seq}")
    print(f"  frames       : {frames.shape}   dtype={frames.dtype}")
    print(f"  kps_norm     : {kps_norm.shape}")
    print(f"  gt_q / gt_r  : {gt_q.shape} / {gt_r.shape}")
    print(f"  vis          : {vis.shape}")
    print(f"  hm_target    : {hm.shape}")
    print(f"  crop_xywh    : {crop_xywh}")
    print(f"  frame_ids    : {frame_ids}")

    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    for batch in loader:
        frames, kps_norm, gt_q, gt_r, vis, hm, frame_ids, crop_xywh, meta, seq_name = batch
        print("batch frames:", frames.shape)
        break
