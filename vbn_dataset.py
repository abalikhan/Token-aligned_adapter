# vbn_dataset.py
# VBN dataset reader aligned to your SPADES/SPARK sliding-window concept.
# Key idea: VBN frames are indexed by POSITION in the per-sequence entry list (0..length-1),
# not by numeric id in the filename, because files can be missing / non-contiguous.

import os
import os.path as osp
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch

from dataset import (
    _BaseSpaceDataset,
    imagenet_normalize_tensor,
    make_gaussian_heatmaps,
    recenter_bbox_with_kps,
    clamp_bbox,
)

VBN_H = 1024
VBN_W = 1024


def _compute_bbox_from_kps_xy(
    kps_xy: np.ndarray,
    vis: np.ndarray,
    W: int,
    H: int,
    expand: float = 1.20,
    min_size: int = 32,
) -> np.ndarray:
    """bbox from visible kps, no square, clamp."""
    if kps_xy.size == 0:
        return np.array([0, 0, W - 1, H - 1], dtype=np.float32)

    if vis is None:
        mask = np.ones((kps_xy.shape[0],), dtype=bool)
    else:
        mask = vis.astype(np.float32) > 0.5

    if not np.any(mask):
        return np.array([0, 0, W - 1, H - 1], dtype=np.float32)

    pts = kps_xy[mask]
    x1, y1 = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
    x2, y2 = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))

    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    bw = max(float(min_size), bw * expand)
    bh = max(float(min_size), bh * expand)

    x1n, y1n = cx - bw / 2.0, cy - bh / 2.0
    x2n, y2n = cx + bw / 2.0, cy + bh / 2.0
    x1i, y1i, x2i, y2i = clamp_bbox(x1n, y1n, x2n, y2n, W, H)
    return np.array([x1i, y1i, x2i, y2i], dtype=np.float32)


class VBNDataset(_BaseSpaceDataset):
    """
    Expects split jsons at:
      {gt_folder}/{split}.json
    where split ∈ {train_synth, val_synth, test_real, test_real_L1, test_real_L2}

    JSON entry fields expected:
      - filename: "MAN-DATA-S2/batch06/image02956.png"
      - keypoints: flat [x,y,vis]*30
      - q_xyzw: (4,)
      - r_xyz:  (3,)
    """
    def __init__(
        self,
        gt_folder: str,
        short_mem: int,
        input_size: int,
        sampling_rate: int,
        clip_stride: int = 0,
        crop: bool = True,
        spatial_transforms=None,
        mode: str = "train",        # controls augmentation + crop decision like your base
        split: str = "train_synth", # chooses JSON file
        heatmap_out: Tuple[int, int] = (56, 56),
        gaussian_sigma_px: float = 2.5,
        eval_use_crop: Optional[bool] = None,
        num_kp: int = 30,
        make_square_crop: bool = False,  # per your request
    ):
        super().__init__(
            gt_folder=gt_folder,
            short_mem=short_mem,
            input_size=input_size,
            sampling_rate=sampling_rate,
            clip_stride=clip_stride,
            crop=crop,
            spatial_transforms=spatial_transforms,
            mode=mode,
            heatmap_out=heatmap_out,
            gaussian_sigma_px=gaussian_sigma_px,
            eval_use_crop=eval_use_crop,
        )
        self.split = str(split)
        self.num_kp = int(num_kp)
        self.make_square_crop = bool(make_square_crop)

        if self.split.startswith("test"):
            self.make_full_sequence_test_dataset(gt_folder)
        else:
            # IMPORTANT: mode passed here controls the seed behavior (train vs val)
            self._make_dataset(gt_folder, mode=("train" if self.split.startswith("train") else "val"))

    # -------------------------
    # Naming / JSON
    # -------------------------
    def _json_filename(self) -> str:
        return f"{self.split}.json"

    def _image_ext(self) -> str:
        return ".png"

    def _seq_from_filename(self, fname_rel: str) -> str:
        # "MAN-DATA-S2/batch06/image02956.png" -> "MAN-DATA-S2_batch06"
        domain, batch, _ = fname_rel.split("/")
        return f"{domain}_{batch}"

    def _image_path_from_entry(self, entry: dict) -> str:
        rel = entry["filename"]
        domain, batch, img = rel.split("/")
        return osp.join(self.gt_folder, domain, "images", batch, img)

    def _parse_entry_xy_vis(self, entry: dict):
        kpts = np.asarray(entry["keypoints"], dtype=np.float32).reshape(-1, 3)
        kpts = kpts[: self.num_kp]
        xy = kpts[:, :2].copy()
        vis = kpts[:, 2].copy()
        return xy, vis

    # -------------------------
    # Build seq maps (VBN-specific)
    # -------------------------
    def _build_seq_maps(self):
        """
        Build:
          - self.seq_to_entries[seq] : list of entries sorted by filename (position index)
          - self.seq_to_len[seq]     : len(list)
        Filter missing image files here so dataloader never hits dead paths.
        """
        self.seq_to_entries: Dict[str, List[dict]] = {}
        missing = 0

        for e in self.data:
            p = self._image_path_from_entry(e)
            if not osp.isfile(p):
                missing += 1
                continue
            seq = self._seq_from_filename(e["filename"])
            self.seq_to_entries.setdefault(seq, []).append(e)

        for seq in list(self.seq_to_entries.keys()):
            self.seq_to_entries[seq].sort(key=lambda x: x["filename"])

        self.seq_to_len = {k: len(v) for k, v in self.seq_to_entries.items()}

        if missing > 0:
            print(f"[VBN] filtered missing images: {missing}")

    # -------------------------
    # SPADES-aligned sliding windows
    # -------------------------
    def _make_dataset(self, gt_folder: str, mode: str):
        """
        Dense sliding windows (SPADES-compatible):
        - ws is the "raw" start index (here: POSITION in entry list)
        - we is bookkeeping (ws + short_mem capped by length)
        - step = clip_stride
        - train: random seed in [0, step)
        - Guarantee no padding: ws <= length - required_raw
        """
        self.dataset.clear()
        self.data = self._load_json(osp.join(gt_folder, self._json_filename()))
        self._build_seq_maps()

        # mimic first_idx + _build_split interface (even if gt_idx unused)
        first_idx = {}
        for i, e in enumerate(self.data):
            seq = self._seq_from_filename(e["filename"])
            if seq not in first_idx:
                first_idx[seq] = i

        entries = self._build_split(first_idx)

        # how many frames the model will consume per clip
        T_out = int(np.ceil(self.short_mem / self.sampling_rate))
        required_raw = 1 + (T_out - 1) * self.sampling_rate

        for seq_name, gt_idx in entries:
            length = int(self.seq_to_len.get(seq_name, 0))
            if length <= 0:
                continue

            last_ws = length - required_raw
            if last_ws < 0:
                continue

            step = max(1, int(self.clip_stride))
            # seed = int(np.random.randint(step)) if mode == "train" else 0
            seed = (abs(hash(seq_name)) % step) if mode == "train" else 0

            ws = seed
            while ws <= last_ws:
                we = min(ws + self.short_mem, length)  # bookkeeping only (like SPADES)
                # imgs_path/gt_idx are not needed for VBN, but keep structure identical
                self.dataset.append([seq_name, ws, we, None, gt_idx])
                ws += step

    def make_full_sequence_test_dataset(self, gt_folder: str):
        self.dataset.clear()
        self.data = self._load_json(osp.join(gt_folder, self._json_filename()))
        self._build_seq_maps()

        # one item per sequence
        for seq_name, length in self.seq_to_len.items():
            if length <= 0:
                continue
            self.dataset.append([seq_name, 0, length, None, 0])

    # -------------------------
    # Crop behavior (no square if you don't want it)
    # -------------------------
    def _preprocess_single(self, frame_rgb, kp_xy, bbox_xyxy, use_crop: bool):
        if not use_crop:
            return super()._preprocess_single(frame_rgb, kp_xy, bbox_xyxy, use_crop=False)

        H, W, _ = frame_rgb.shape
        x1, y1, x2, y2 = recenter_bbox_with_kps(
            bbox_xyxy, kp_xy, W, H,
            expand=1.2, center_thresh=0.15, min_size=32,
            make_square=self.make_square_crop,  # False by default
        )
        bbox2 = np.array([x1, y1, x2, y2], dtype=np.float32)
        return super()._preprocess_single(frame_rgb, kp_xy, bbox2, use_crop=True)

    # -------------------------
    # Fetch GT for a clip (position-based)
    # -------------------------
    def get_kps_gt(self, work_positions: np.ndarray, seq_name: str):
        frames_resized = []
        kps_xy_resized = []
        gt_q = []
        gt_r = []
        vis_list = []
        crop_xywh_list = []
        meta_list = []

        use_crop_clip = self._decide_use_crop_for_clip()
        entries = self.seq_to_entries[seq_name]
        n = len(entries)

        for pos in work_positions.tolist():
            pos = int(pos)
            if pos < 0 or pos >= n:
                raise IndexError(f"[VBN] pos={pos} out of range for seq={seq_name}, n={n}")

            entry = entries[pos]
            im_pth = self._image_path_from_entry(entry)
            img_rgb = self._read_frame(im_pth)

            xy, vis = self._parse_entry_xy_vis(entry)

            # bbox from keypoints in ORIGINAL 1024x1024
            bbox = _compute_bbox_from_kps_xy(xy, vis, W=VBN_W, H=VBN_H, expand=1.2, min_size=32)

            img_r, kp_r, crop_xywh_norm, meta = self._preprocess_single(
                img_rgb, xy, bbox, use_crop=use_crop_clip
            )

            frames_resized.append(img_r)
            kps_xy_resized.append(kp_r)
            crop_xywh_list.append(torch.tensor(crop_xywh_norm, dtype=torch.float32))
            meta_list.append(torch.tensor(meta, dtype=torch.float32))

            gt_q.append(torch.tensor(entry["q_xyzw"], dtype=torch.float32))
            gt_r.append(torch.tensor(entry["r_xyz"], dtype=torch.float32))
            vis_list.append(torch.from_numpy(vis.astype(np.float32)))

        T = len(frames_resized)

        # train-only photometric aug
        if self.mode == "train" and T > 0 and self.spatial_transforms is not None:
            frames_resized, kps_xy_resized = self.spatial_transforms(
                frames_resized, kps_xy_resized
            )

        # to torch + imagenet norm
        frames = []
        for t in range(T):
            img = torch.from_numpy(frames_resized[t]).permute(2, 0, 1).float() / 255.0
            img = imagenet_normalize_tensor(img)
            frames.append(img)
        frames = torch.stack(frames, dim=0)  # (T,3,H,W)

        gt_q = torch.stack(gt_q, dim=0)
        gt_r = torch.stack(gt_r, dim=0)
        vis  = torch.stack(vis_list, dim=0)
        crop_xywh = torch.stack(crop_xywh_list, dim=0)
        meta_tensor = torch.stack(meta_list, dim=0)

        # heatmaps + normalized keypoints in HM coords
        Hm, Wm = self.hm_size
        inp = float(self.input_size)

        kps_img = torch.from_numpy(np.stack(kps_xy_resized, axis=0)).float()  # (T,K,2)
        kps_hm = kps_img.clone()
        kps_hm[..., 0] = kps_hm[..., 0] * ((Wm - 1.0) / (inp - 1.0))
        kps_hm[..., 1] = kps_hm[..., 1] * ((Hm - 1.0) / (inp - 1.0))

        hm_target = make_gaussian_heatmaps(kps_hm, H=Hm, W=Wm, sigma_px=self.hm_sigma_px)

        x_norm = 2.0 * (kps_hm[..., 0] / max(Wm - 1.0, 1.0)) - 1.0
        y_norm = 2.0 * (kps_hm[..., 1] / max(Hm - 1.0, 1.0)) - 1.0
        kp_target_norm = torch.stack([x_norm, y_norm], dim=-1)

        if not use_crop_clip:
            crop_xywh = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32).view(1, 4).repeat(T, 1)

        return frames, kp_target_norm, gt_q, gt_r, vis, hm_target, crop_xywh, meta_tensor

    # -------------------------
    # __getitem__ (SPADES-compatible)
    # -------------------------
    def __getitem__(self, idx):
        seq_name, work_start, work_end, _imgs_path, _gt_idx = self.dataset[idx]

        # TEST: return full sequence
        if self.split.startswith("test") or self.mode in ("test", "test_real"):
            n = self.seq_to_len[seq_name]
            work_positions = np.arange(n, dtype=np.int64)

            frames, kps_norm, gt_q, gt_r, vis, hm, crop_xywh, meta = self.get_kps_gt(
                work_positions, seq_name
            )
            # ids are stable filenames
            entries = self.seq_to_entries[seq_name]
            frame_ids = [[f"{seq_name}_{osp.splitext(osp.basename(e['filename']))[0]}" for e in entries]]
            return frames, kps_norm, gt_q, gt_r, vis, hm, frame_ids, crop_xywh, meta, seq_name

        # TRAIN/VAL: fixed-length clip (NO padding)
        T_out = int(np.ceil(self.short_mem / self.sampling_rate))
        work_positions = work_start + np.arange(T_out, dtype=np.int64) * int(self.sampling_rate)

        frames, kps_norm, gt_q, gt_r, vis, hm, crop_xywh, _meta = self.get_kps_gt(
            work_positions, seq_name
        )

        entries = self.seq_to_entries[seq_name]
        frame_ids = [[
            f"{seq_name}_{osp.splitext(osp.basename(entries[int(p)]['filename']))[0]}"
            for p in work_positions.tolist()
        ]]
        return frames, kps_norm, gt_q, gt_r, vis, hm, frame_ids, crop_xywh
    
if __name__ == "__main__":
    # quick test of dataset loading
    gt_folder = "/project/scratch/p200249/space_data/VBN/"
    dataset = VBNDataset(
        gt_folder=gt_folder,
        short_mem=8,
        input_size=256,
        sampling_rate=1,
        clip_stride=4,
        crop=True,
        spatial_transforms=None,
        mode="test",
        split="test_real",
        heatmap_out=(56, 56),
        gaussian_sigma_px=2.5,
        eval_use_crop=None,
        num_kp=30,
        make_square_crop=False,
    )
    print(f"dataset length: {len(dataset)}")