# dataset.py (CLEAN, no replay)
import json
import os
import os.path as osp
from typing import Tuple, List, Optional

import cv2
import numpy as np
from PIL import Image  # noqa: F401

import torch
import torch.utils.data as data
import torch.nn.functional as F

import albumentations as A
import albumentations.augmentations.geometric.resize as albu_resize


# =============================
# Delta-pose helpers
# =============================

def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    q: (4,) as (x,y,z,w) — convert to (3,3)
    """
    q = F.normalize(q, dim=0)
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = torch.stack([
        torch.stack([1 - 2*(yy + zz),     2*(xy - wz),     2*(xz + wy)], dim=0),
        torch.stack([    2*(xy + wz), 1 - 2*(xx + zz),     2*(yz - wx)], dim=0),
        torch.stack([    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx + yy)], dim=0)
    ], dim=0)
    return R  # (3,3)

def rotmat_to_logvec(R: torch.Tensor) -> torch.Tensor:
    """
    R: (B,3,3) — returns (B,3) log vector
    """
    B = R.shape[0]
    cos_theta = ((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]) - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)

    log_R = torch.zeros(B, 3, device=R.device)
    zero = torch.tensor(0.0, device=R.device)
    for i in range(B):
        t = theta[i]
        if torch.isclose(t, zero):
            continue
        skew = (R[i] - R[i].T) / (2 * torch.sin(t))
        log_R[i, 0] = t * skew[2, 1]
        log_R[i, 1] = t * skew[0, 2]
        log_R[i, 2] = t * skew[1, 0]
    return log_R


# =============================
# Image helpers
# =============================
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

def to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def imagenet_normalize_tensor(img_chw_float01: torch.Tensor) -> torch.Tensor:
    return (img_chw_float01 - IMAGENET_MEAN.to(img_chw_float01.device)) / IMAGENET_STD.to(img_chw_float01.device)


# =============================
# Gaussian heatmaps (pixel-coord version)
# =============================
def make_gaussian_heatmaps(
    kps_hw: torch.Tensor, H: int = 56, W: int = 56, sigma_px: float = 1.8
) -> torch.Tensor:
    """
    kps_hw: (T,K,2) in heatmap pixel coords (x=cols, y=rows)
    returns: (T,K,H,W) with each (K,H,W) normalized to sum=1
    """
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
            x = kps_hw[t, k, 0]
            y = kps_hw[t, k, 1]
            g = torch.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2.0 * sigma_px ** 2))
            g = g / (g.sum() + 1e-8)
            frame_maps.append(g)
        maps.append(torch.stack(frame_maps, dim=0))
    return torch.stack(maps, dim=0)  # (T,K,H,W)


# =============================
# BBox helpers
# =============================
def clamp_bbox(xmin, ymin, xmax, ymax, W, H):
    xmin = max(0, min(int(xmin), W - 1))
    ymin = max(0, min(int(ymin), H - 1))
    xmax = max(0, min(int(xmax), W - 1))
    ymax = max(0, min(int(ymax), H - 1))
    if xmax <= xmin:
        xmax = min(W - 1, xmin + 1)
    if ymax <= ymin:
        ymax = min(H - 1, ymin + 1)
    return xmin, ymin, xmax, ymax

def recenter_bbox_with_kps(
    bbox: np.ndarray,
    kps_xy: np.ndarray,
    img_w: int,
    img_h: int,
    expand: float = 1.2,
    center_thresh: float = 0.15,
    min_size: int = 32,
    make_square: bool = True,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(float, bbox)
    bx_c = 0.5 * (x1 + x2)
    by_c = 0.5 * (y1 + y2)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    if kps_xy.size == 0:
        return clamp_bbox(x1, y1, x2, y2, img_w, img_h)

    kx = kps_xy[:, 0]
    ky = kps_xy[:, 1]
    kx_min, kx_max = float(np.min(kx)), float(np.max(kx))
    ky_min, ky_max = float(np.min(ky)), float(np.max(ky))
    k_cx = 0.5 * (kx_min + kx_max)
    k_cy = 0.5 * (ky_min + ky_max)
    kw = max(1.0, kx_max - kx_min)
    kh = max(1.0, ky_max - ky_min)

    if abs(k_cx - bx_c) > center_thresh * bw or abs(k_cy - by_c) > center_thresh * bh:
        bx_c, by_c = k_cx, k_cy

    box_w = max(bw, kw * expand)
    box_h = max(bh, kh * expand)
    box_w = max(box_w, float(min_size))
    box_h = max(box_h, float(min_size))

    if make_square:
        side = max(box_w, box_h)
        box_w = box_h = side

    x1_new = bx_c - box_w / 2.0
    y1_new = by_c - box_h / 2.0
    x2_new = bx_c + box_w / 2.0
    y2_new = by_c + box_h / 2.0

    return clamp_bbox(x1_new, y1_new, x2_new, y2_new, img_w, img_h)


# =============================
# Base dataset
# =============================
class _BaseSpaceDataset(data.Dataset):
    def __init__(
        self,
        gt_folder: str,
        short_mem: int,
        input_size: int,
        sampling_rate: int,
        clip_stride: int = 0,
        crop: bool = True,
        spatial_transforms: Optional[A.BasicTransform] = None,
        mode: str = "train",
        heatmap_out: Tuple[int, int] = (56, 56),
        gaussian_sigma_px: float = 1.8,
        eval_use_crop: Optional[bool] = None,
        # spatial_transforms_full: Optional[A.BasicTransform] = None,
    ):
        self.gt_folder = gt_folder
        self.spatial_transforms = spatial_transforms
        # self.spatial_transforms_full = spatial_transforms_full
        self.mode = mode

        self.crop = bool(crop)
        # self.crop_prob = float(crop_prob)
        self.eval_use_crop = eval_use_crop

        self.short_mem = int(short_mem)
        self.input_size = int(input_size)
        self.sampling_rate = int(sampling_rate)
        self.hm_size = tuple(heatmap_out)
        self.hm_sigma_px = float(gaussian_sigma_px)

        # passage-style stride
        self.clip_stride = int(clip_stride) if int(clip_stride) > 0 else int(short_mem)

        # Random temporal stride: during training, __getitem__ picks sr from
        # this list per clip, giving varied temporal spans within each batch.
        # _make_dataset uses self.sampling_rate (should be the max of the range)
        # to size windows, so all strides fit safely.
        self._sampling_rate_range = None

        self.data: List[dict] = []
        self.dataset: List[list] = []

    def set_sampling_rate_range(self, sr_range):
        """Set list of sampling rates to randomly choose from during training."""
        if sr_range is not None:
            sr_range = sorted(int(s) for s in sr_range)
            assert sr_range[-1] <= self.sampling_rate, (
                f"Max of sampling_rate_range ({sr_range[-1]}) must be <= "
                f"sampling_rate ({self.sampling_rate}) used for window sizing"
            )
            self._sampling_rate_range = sr_range

    def _json_filename(self) -> str:
        return f"{self.mode}.json"

    def _image_ext(self) -> str:
        return ".jpg"

    def _build_split(self, all_seq_first_index: dict):
        return list(all_seq_first_index.items())

    def _load_json(self, path: str):
        with open(path, "r") as f:
            return json.load(f)

    def make_full_sequence_test_dataset(self, gt_folder: str):
        self.dataset.clear()
        self.data = self._load_json(os.path.join(gt_folder, self._json_filename()))
        self._build_seq_maps()

        first_idx = {}
        for i, e in enumerate(self.data):
            seq = e["filename"].split(".")[0].split("_")[1]
            if seq not in first_idx:
                first_idx[seq] = i

        for seq_name, gt_idx in first_idx.items():
            imgs_path = os.path.join(self.gt_folder, "images", seq_name)
            if not os.path.isdir(imgs_path):
                imgs_path = os.path.join(self.gt_folder, seq_name)
            # length = len([n for n in os.listdir(imgs_path) if n.endswith(self._image_ext())])
            frames_avail = self.seq_to_frames[seq_name]
            length = (max(frames_avail) + 1) if len(frames_avail) else 0
            self.dataset.append([seq_name, 0, length, imgs_path, gt_idx])

    def _make_dataset(self, gt_folder: str, mode: str):
        """
        Dense sliding windows:
        - raw window start = ws
        - we is just bookkeeping (ws + short_mem capped by length)
        - step = clip_stride (raw-frame stride between windows)
        - train: random seed offset in [0, step)
        - IMPORTANT: we only create windows where sampling yields >= T_out frames,
            so __getitem__ never needs to pad.
        """
        self.dataset.clear()
        self.data = self._load_json(osp.join(gt_folder, self._json_filename()))
        self._build_seq_maps()

        first_idx = {}
        for i, e in enumerate(self.data):
            seq = e["filename"].split(".")[0].split("_")[1]
            if seq not in first_idx:
                first_idx[seq] = i

        entries = self._build_split(first_idx)

        # how many frames the model will consume per clip
        T_out = int(np.ceil(self.short_mem / self.sampling_rate))
        # minimum raw length needed so raw[::sr] gives T_out samples
        required_raw = 1 + (T_out - 1) * self.sampling_rate

        for seq_name, gt_idx in entries:
            imgs_path = osp.join(self.gt_folder, "images", seq_name)
            if not osp.isdir(imgs_path):
                imgs_path = osp.join(self.gt_folder, seq_name)

            length = len([n for n in os.listdir(imgs_path) if n.endswith(self._image_ext())])
            if length <= 0:
                continue

            # last start index that still allows T_out sampled frames
            last_ws = length - required_raw
            if last_ws < 0:
                continue

            step = max(1, int(self.clip_stride))
            seed = int(np.random.randint(step)) if mode == "train" else 0

            ws = seed
            while ws <= last_ws:
                we = min(ws + self.short_mem, length)  # bookkeeping only
                self.dataset.append([seq_name, ws, we, imgs_path, gt_idx])
                ws += step

    def _read_frame(self, path: str) -> np.ndarray:
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(path)
        return to_rgb(img_bgr)

    def _parse_entry_xy_vis(self, entry: dict):
        kpts = np.asarray(entry["keypoints"], dtype=np.float32).reshape(-1, 3)
        kpts = kpts[:8]
        xy = kpts[:, :2].copy()
        vis = kpts[:, 2].copy()
        return xy, vis

    def _decide_use_crop_for_clip(self) -> bool:
        # deterministic behavior (no crop_prob)
        if self.mode == "train":
            return bool(self.crop)
        # val/test: allow explicit override
        if self.eval_use_crop is None:
            return bool(self.crop)
        return bool(self.eval_use_crop)

    def _preprocess_single(
        self,
        frame_rgb: np.ndarray,
        kp_xy: np.ndarray,
        bbox_xyxy: np.ndarray,
        use_crop: bool,
    ):
        """
        Returns:
          img_r: resized RGB (input_size,input_size,3)
          kp_r:  resized keypoints in image pixel coords (input_size space)
          crop_xywh_norm: crop box normalized w.r.t ORIGINAL full frame
                          (x0/W_full, y0/H_full, w/W_full, h/H_full)
        """
        H, W, _ = frame_rgb.shape

        # -------- default meta (used for crop path or ignored) --------
        meta = [float("nan"), float("nan"), float("nan"), float(W), float(H)]
        if use_crop:
            x1, y1, x2, y2 = recenter_bbox_with_kps(
                bbox_xyxy, kp_xy, W, H,
                expand=1.2, center_thresh=0.15, min_size=32, make_square=True
            )
            crop_xywh = [x1, y1, x2 - x1, y2 - y1]
            transform = A.Compose(
                [
                    A.Crop(x_min=int(x1), y_min=int(y1), x_max=int(x2), y_max=int(y2)),
                    albu_resize.Resize(self.input_size, self.input_size, p=1.0),
                ],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            )
        else:
            # FULL FRAME, NO CROP, PRESERVE ASPECT RATIO via letterbox
            crop_xywh = [0, 0, W, H]  # still full frame

            transform = A.Compose(
                [
                    A.LongestMaxSize(max_size=self.input_size, interpolation=cv2.INTER_AREA, p=1.0),
                    A.PadIfNeeded(
                        min_height=self.input_size,
                        min_width=self.input_size,
                        border_mode=cv2.BORDER_CONSTANT,
                        fill=0,
                        position="center",
                        p=1.0,
                    ),
                ],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            )

            # ---- exact meta from transformed corners (no rounding mismatch) ----
            # Add 2 sentinel keypoints: top-left and bottom-right corners in original pixel coords.
            # Use (W-1,H-1) because your kp math uses [0..W-1], [0..H-1] convention.
            corners = np.array([[0.0, 0.0], [float(W - 1), float(H - 1)]], dtype=np.float32)

            kp_aug = np.concatenate([kp_xy.astype(np.float32), corners], axis=0)  # (K+2,2)

            out = transform(image=frame_rgb, keypoints=kp_aug)
            img_r = out["image"]
            kp_r_aug = np.asarray(out["keypoints"], dtype=np.float32)

            kp_r = kp_r_aug[:-2]            # (K,2) in input_size px
            c_tl = kp_r_aug[-2]             # transformed (0,0)
            c_br = kp_r_aug[-1]             # transformed (W-1,H-1)

            # pad is exactly where the top-left corner ended up
            pad_left = float(c_tl[0])
            pad_top  = float(c_tl[1])

            # effective isotropic scale from mapped width/height
            # width in input px corresponds to (W-1) in original; same for height
            s_x = (float(c_br[0]) - float(c_tl[0])) / max(float(W - 1), 1.0)
            s_y = (float(c_br[1]) - float(c_tl[1])) / max(float(H - 1), 1.0)

            # They should be ~equal (isotropic). Average for stability.
            scale = 0.5 * (s_x + s_y)

            meta = [float(scale), float(pad_left), float(pad_top), float(W), float(H)]

            # IMPORTANT: return here because we already ran transform
            crop_xywh_norm = [0.0, 0.0, 1.0, 1.0]
            return img_r, kp_r, crop_xywh_norm, meta

        crop_xywh_norm = [
            crop_xywh[0] / W,
            crop_xywh[1] / H,
            crop_xywh[2] / W,
            crop_xywh[3] / H,
        ]

        out = transform(image=frame_rgb, keypoints=kp_xy)
        img_r = out["image"]
        kp_r = np.asarray(out["keypoints"], dtype=np.float32)
        return img_r, kp_r, crop_xywh_norm, meta

    def _build_seq_maps(self):
        # seq -> {frame_idx:int -> entry:dict}
        self.seq_to_frame_entry = {}
        # seq -> sorted list of available frame indices
        self.seq_to_frames = {}

        for e in self.data:
            fname = e["filename"]  # e.g. "img005_RT590.jpg"
            base = os.path.basename(fname)
            parts = base.split("_")
            frame_str = parts[0]          # "img005"
            seq_name = parts[1].split(".")[0]  # "RT590"
            frame_idx = int(frame_str.replace("img", ""))

            self.seq_to_frame_entry.setdefault(seq_name, {})[frame_idx] = e

        for seq, mp in self.seq_to_frame_entry.items():
            self.seq_to_frames[seq] = sorted(mp.keys())

    def get_kps_gt(self, work_indices, seq_name, gt_idx, imgs_path):
        frames_resized: List[np.ndarray] = []
        kps_xy_resized: List[np.ndarray] = []
        gt_q: List[torch.Tensor] = []
        gt_r: List[torch.Tensor] = []
        vis_list: List[torch.Tensor] = []
        crop_xywh_list: List[torch.Tensor] = []
        meta_list: List[torch.Tensor] = []

        use_crop_clip = self._decide_use_crop_for_clip()

        for idx in work_indices:
            img_name = f"img{str(idx).zfill(3)}_{seq_name}{self._image_ext()}"
            im_pth = osp.join(imgs_path, img_name)
            img_rgb = self._read_frame(im_pth)

            # entry = self.data[gt_idx + idx]
            entry = self.seq_to_frame_entry[seq_name].get(int(idx), None)
            if entry is None:
                # If you want hard-fail (recommended during debugging):
                raise KeyError(f"Missing annotation for seq={seq_name}, frame={int(idx)}")

            bbox = np.asarray(entry["bbox"], dtype=np.float32)
            xy, vis = self._parse_entry_xy_vis(entry)

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

        # ---- augmentation (geo clip-consistent + photometric per-frame) ----
        if self.mode == "train" and T > 0 and self.spatial_transforms is not None:
            frames_resized, kps_xy_resized = self.spatial_transforms(
                frames_resized, kps_xy_resized
            )
            # crop_xywh stays as bbox-crop in original frame; do NOT modify.

        # ---- to torch, ImageNet norm ----
        frames = []
        for t in range(T):
            img = torch.from_numpy(frames_resized[t]).permute(2, 0, 1).float() / 255.0
            img = imagenet_normalize_tensor(img)
            frames.append(img)
        frames = torch.stack(frames, dim=0)  # (T,3,H,W)

        gt_q = torch.stack(gt_q, dim=0)      # (T,4)
        gt_r = torch.stack(gt_r, dim=0)      # (T,3)
        vis  = torch.stack(vis_list, dim=0)  # (T,K)
        crop_xywh = torch.stack(crop_xywh_list, dim=0)  # (T,4)
        meta_tensor = torch.stack(meta_list, dim=0)  # (T,5)

        # ---- heatmap + normalized keypoints in HEATMAP coordinates ----
        Hm, Wm = self.hm_size
        inp = float(self.input_size)

        kps_img = torch.from_numpy(np.stack(kps_xy_resized, axis=0)).float()  # (T,K,2) in input_size px

        kps_hm = kps_img.clone()
        kps_hm[..., 0] = kps_hm[..., 0] * ((Wm - 1.0) / (inp - 1.0))
        kps_hm[..., 1] = kps_hm[..., 1] * ((Hm - 1.0) / (inp - 1.0))

        hm_target = make_gaussian_heatmaps(kps_hm, H=Hm, W=Wm, sigma_px=self.hm_sigma_px)  # (T,K,Hm,Wm)

        x_norm = 2.0 * (kps_hm[..., 0] / max(Wm - 1.0, 1.0)) - 1.0
        y_norm = 2.0 * (kps_hm[..., 1] / max(Hm - 1.0, 1.0)) - 1.0
        kp_target_norm = torch.stack([x_norm, y_norm], dim=-1)  # (T,K,2) in [-1,1]

        # if not cropping, enforce [0,0,1,1]
        if not use_crop_clip:
            crop_xywh = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32).view(1, 4).repeat(T, 1)

        return frames, kp_target_norm, gt_q, gt_r, vis, hm_target, crop_xywh, meta_tensor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        seq_name, work_start, work_end, imgs_path, gt_idx = self.dataset[idx]

        # -------------------------
        # TEST: return full sampled sequence (variable T)
        # -------------------------
        if self.mode in ("test", "val-syn"):
            # raw = np.arange(work_start, work_end, dtype=np.int64)  # full seq range
            # work_indices = raw[:: int(self.sampling_rate)]
            work_indices = np.array(self.seq_to_frames[seq_name], dtype=np.int64)
            if work_indices.size == 0:
                raise RuntimeError(f"No annotations for seq {seq_name} in {self._json_filename()}")

            # work_indices = frames_avail #[:: int(self.sampling_rate)]
            if work_indices.size == 0:
                work_indices = np.array([work_start], dtype=np.int64)

            frames, kps_norm, gt_q, gt_r, vis, hm, crop_xywh, meta_tensor = self.get_kps_gt(
                work_indices, seq_name, gt_idx, imgs_path
            )


            frame_ids = [[f"{seq_name}_{int(t)}" for t in work_indices.tolist()]]
            return frames, kps_norm, gt_q, gt_r, vis, hm, frame_ids, crop_xywh, meta_tensor, seq_name

        # -------------------------
        # TRAIN/VAL: fixed-length clip (no padding)
        # -------------------------
        # T is always determined by the config sampling_rate (max stride)
        T_out = int(np.ceil(self.short_mem / self.sampling_rate))

        # During training, randomly pick stride from the range for temporal diversity
        if self.mode == "train" and self._sampling_rate_range is not None:
            sr = int(np.random.choice(self._sampling_rate_range))
        else:
            sr = int(self.sampling_rate)

        work_indices = work_start + np.arange(T_out, dtype=np.int64) * sr

        frames, kps_norm, gt_q, gt_r, vis, hm, crop_xywh, _ = self.get_kps_gt(
            work_indices, seq_name, gt_idx, imgs_path
        )

        frame_ids = [[f"{seq_name}_{int(t)}" for t in work_indices.tolist()]]
        return frames, kps_norm, gt_q, gt_r, vis, hm, frame_ids, crop_xywh

# =============================
# SPADES dataset
# =============================
class SPADESDataset(_BaseSpaceDataset):
    def __init__(
        self,
        gt_folder, short_mem, input_size, sampling_rate,
        clip_stride: int = 0,
        crop=True,
        spatial_transforms=None,
        mode="train",
        heatmap_out=(56, 56),
        gaussian_sigma_px: float = 2.5,
        eval_use_crop: Optional[bool] = None,
        # spatial_transforms_full: Optional[A.BasicTransform] = None,
    ):
        super().__init__(
            gt_folder, short_mem, input_size, sampling_rate, clip_stride,
            crop, spatial_transforms, mode, heatmap_out, gaussian_sigma_px,
            eval_use_crop=eval_use_crop,
            # spatial_transforms_full=spatial_transforms_full
        )

        if mode == "test":
            self.make_full_sequence_test_dataset(gt_folder)
        else:
            self._make_dataset(gt_folder, mode)

    def _json_filename(self) -> str:
        return f"{self.mode}.json"

    def _image_ext(self) -> str:
        return ".png"


# =============================
# SPARK dataset
# =============================
class SPARKDataset(_BaseSpaceDataset):
    VAL_SEQS = {"RT500", "RT540", "RT586", "RT565"}

    def __init__(
        self,
        gt_folder, short_mem, input_size, sampling_rate,
        clip_stride: int = 0,
        crop=True,
        spatial_transforms=None,
        mode="train",
        heatmap_out=(56, 56),
        gaussian_sigma_px: float = 1.8,
        eval_use_crop: Optional[bool] = None,
        # spatial_transforms_full: Optional[A.BasicTransform] = None,
    ):
        super().__init__(
            gt_folder, short_mem, input_size, sampling_rate, clip_stride,
            crop, spatial_transforms, mode, heatmap_out, gaussian_sigma_px,
            eval_use_crop=eval_use_crop,
            # spatial_transforms_full=spatial_transforms_full
        )

        # SPARK uses a single train.json for both train and val splits
        if mode == 'test' or mode == 'val-syn':
            self.make_full_sequence_test_dataset(gt_folder)
        else:  # "train" or "val" — _build_split filters by VAL_SEQS
            self._make_dataset(gt_folder, mode)
            

    def _json_filename(self):
        if self.mode in ("train", "val"):
            return "train.json"
        return f"{self.mode}.json"

    def _image_ext(self) -> str:
        return ".jpg"

    def _build_split(self, all_seq_first_index: dict):
        if self.mode == "train":
            return [(k, v) for k, v in all_seq_first_index.items() if k not in self.VAL_SEQS]
        else:
            return [(k, v) for k, v in all_seq_first_index.items() if k in self.VAL_SEQS]


# =============================
# Quick smoke test
# =============================
if __name__ == "__main__":
    from utils_new import collate_temporal_chunks
    gt_folder = "/project/scratch/p200249/space_data/spark-2024-stream-2"
    ds = SPARKDataset(
        gt_folder=gt_folder,
        input_size=512,
        sampling_rate =16,       # max stride — used for window sizing & model T (ceil(128/16)=8)
        short_mem = 128,          # wider window to accommodate large strides
        clip_stride = 18,         # adjusted to keep ~same clip count per sequence
        crop = True,
        eval_use_crop = True, 
        gaussian_sigma_px = 2.5, #from 2.5
    )
    # print("num clips:", len(ds))
    # x = ds[0]
    # print("frames:", x[0].shape, "kps_norm:", x[1].shape, "hm:", x[5].shape, "crop:", x[-1].shape)

    dloader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, num_workers=8, collate_fn=lambda b: collate_temporal_chunks(b, 8))
    for batch in dloader:
        video, kps_gt_norm, quat_gt_xyzw, trans_gt, vis, hm_gt, frame_ids, crop_xywh, geom, seq_name = batch
        print("frames:", video.shape)
        print("kps_norm:", kps_gt_norm.shape)
