# dataset.py (final)
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
# Delta-pose helpers (logSO3, quat2rot, etc)
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
            continue  # near-zero rotation
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
def make_gaussian_heatmaps(kps_hw: torch.Tensor, H: int = 56, W: int = 56, sigma_px: float = 1.8) -> torch.Tensor:
    """
    kps_hw: (T,K,2) in heatmap pixel coords (x=cols, y=rows)
    returns: (T,K,H,W) with each (K,H,W) normalized to sum=1
    """
    device = kps_hw.device
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
    )  # (H,W)

    maps = []
    for t in range(kps_hw.shape[0]):
        frame_maps = []
        for k in range(kps_hw.shape[1]):
            x = kps_hw[t, k, 0]
            y = kps_hw[t, k, 1]
            g = torch.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2.0 * sigma_px ** 2))
            g = g / (g.sum() + 1e-8)  # sum-to-1
            frame_maps.append(g)
        maps.append(torch.stack(frame_maps, dim=0))  # (K,H,W)
    return torch.stack(maps, dim=0)  # (T,K,H,W)


# =============================
# Function to update crop_xywh based on replay dict from albumentations
# =============================
def update_crop_xywh_from_replay(crop_xywh, replay_dict, img_size):
    for op in replay_dict.get('transforms', []):
        if op['__class_fullname__'].endswith('RandomResizedCrop') and op.get('applied', False):
            params = op.get('params', {})
            if not params:
                continue

            w = params.get('width', params.get('crop_width', img_size))
            h = params.get('height', params.get('crop_height', img_size))
            x = params.get('x_min', params.get('x', 0))
            y = params.get('y_min', params.get('y', 0))

            scale_x = w / img_size
            scale_y = h / img_size
            shift_x = x / img_size
            shift_y = y / img_size

            crop_xywh[0] = crop_xywh[0] * scale_x + shift_x
            crop_xywh[1] = crop_xywh[1] * scale_y + shift_y
            crop_xywh[2] *= scale_x
            crop_xywh[3] *= scale_y
    return crop_xywh


# =============================
# BBox helpers
# =============================
def clamp_bbox(xmin, ymin, xmax, ymax, W, H):
    xmin = max(0, min(int(xmin), W - 1))
    ymin = max(0, min(int(ymin), H - 1))
    xmax = max(0, min(int(xmax), W - 1))
    ymax = max(0, min(int(ymax), H - 1))
    if xmax <= xmin: xmax = min(W - 1, xmin + 1)
    if ymax <= ymin: ymax = min(H - 1, ymin + 1)
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

    kx = kps_xy[:, 0]; ky = kps_xy[:, 1]
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
        spatial_transforms: Optional[A.ReplayCompose] = None,
        mode: str = "train",
        heatmap_out: Tuple[int, int] = (56, 56),
        gaussian_sigma_px: float = 1.8,
        crop_prob: float = 0.5,
        eval_use_crop: Optional[bool] = None,
        spatial_transforms_full: Optional[A.ReplayCompose] = None,
    ):
        self.gt_folder = gt_folder
        self.spatial_transforms = spatial_transforms
        self.spatial_transforms_full = spatial_transforms_full
        self.mode = mode
        self.crop = bool(crop)
        self.crop_prob = float(crop_prob)
        self.eval_use_crop = eval_use_crop

        self.short_mem = int(short_mem)
        self.input_size = int(input_size)
        self.sampling_rate = int(sampling_rate)
        self.hm_size = tuple(heatmap_out)
        self.hm_sigma_px = float(gaussian_sigma_px)

        # IMPORTANT: passage-style stride
        # - if clip_stride <= 0: default to non-overlapping windows (short_mem)
        # - else: dense sliding windows with overlap (passage behavior)
        self.clip_stride = int(clip_stride) if int(clip_stride) > 0 else int(short_mem)

        self.data: List[dict] = []
        self.dataset: List[list] = []
        self.hm_std = torch.tensor([0.015, 0.015], dtype=torch.float32)

    def _json_filename(self) -> str: return f"{self.mode}.json"
    def _image_ext(self) -> str:     return ".png"

    def _build_split(self, all_seq_first_index: dict):
        return list(all_seq_first_index.items())

    def _load_json(self, path: str):
        with open(path, "r") as f:
            return json.load(f)

    def make_full_sequence_test_dataset(self, gt_folder: str):
        self.dataset.clear()
        self.data = self._load_json(os.path.join(gt_folder, self._json_filename()))

        first_idx = {}
        for i, e in enumerate(self.data):
            seq = e["filename"].split(".")[0].split("_")[1]
            if seq not in first_idx:
                first_idx[seq] = i

        for seq_name, gt_idx in first_idx.items():
            imgs_path = os.path.join(self.gt_folder, "images", seq_name)
            if not os.path.isdir(imgs_path):
                imgs_path = os.path.join(self.gt_folder, seq_name)

            length = len([n for n in os.listdir(imgs_path) if n.endswith(self._image_ext())])
            self.dataset.append([seq_name, 0, length, imgs_path, gt_idx])

    def _make_dataset(self, gt_folder: str, mode: str):
        """
        PASSAGE-STYLE clip generation (dense sliding windows):
          - window length = short_mem
          - step = clip_stride (passage)
          - train: random seed offset for diversity
          - include last partial window (your __getitem__ will pad frames anyway)
        """
        self.dataset.clear()
        self.data = self._load_json(osp.join(gt_folder, self._json_filename()))

        first_idx = {}
        for i, e in enumerate(self.data):
            seq = e["filename"].split(".")[0].split("_")[1]
            if seq not in first_idx:
                first_idx[seq] = i

        entries = self._build_split(first_idx)

        for seq_name, gt_idx in entries:
            imgs_path = osp.join(self.gt_folder, "images", seq_name)
            if not osp.isdir(imgs_path):
                imgs_path = osp.join(self.gt_folder, seq_name)

            length = len([n for n in os.listdir(imgs_path) if n.endswith(self._image_ext())])
            if length <= 0:
                continue

            passage = max(1, int(self.clip_stride))

            # reference behavior: random seed in train to diversify boundaries
            if mode == "train":
                seed = int(np.random.randint(passage))
            else:
                seed = 0

            ws = seed
            while ws < length:
                we = min(ws + self.short_mem, length)   # allow last partial
                self.dataset.append([seq_name, ws, we, imgs_path, gt_idx])
                ws += passage

    def _read_frame(self, path: str) -> np.ndarray:
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(path)
        return to_rgb(img_bgr)

    def _preprocess_single(
        self,
        frame_rgb: np.ndarray,
        kp_xy: np.ndarray,
        bbox_xyxy: np.ndarray,
        use_crop: bool
    ):
        H, W, _ = frame_rgb.shape

        if use_crop:
            x1, y1, x2, y2 = recenter_bbox_with_kps(
                bbox_xyxy, kp_xy, W, H,
                expand=1.2, center_thresh=0.15, min_size=32, make_square=True
            )
            crop_xywh = [x1, y1, x2 - x1, y2 - y1]
            transform = A.Compose(
                [A.Crop(x_min=int(x1), y_min=int(y1), x_max=int(x2), y_max=int(y2)),
                 albu_resize.Resize(self.input_size, self.input_size, p=1.0)],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            )
        else:
            crop_xywh = [0, 0, W, H]
            transform = A.Compose(
                [albu_resize.Resize(self.input_size, self.input_size, p=1.0)],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            )

        norm_crop_xywh = [c / d for c, d in zip(crop_xywh, [W, H, W, H])]

        out = transform(image=frame_rgb, keypoints=kp_xy)
        img_r = out["image"]
        kp_r  = np.asarray(out["keypoints"], dtype=np.float32)
        return img_r, kp_r, norm_crop_xywh

    def _parse_entry_xy_vis(self, entry: dict):
        kpts = np.asarray(entry["keypoints"], dtype=np.float32).reshape(-1, 3)
        kpts = kpts[:8]
        xy   = kpts[:, :2].copy()
        vis  = kpts[:,  2].copy()
        return xy, vis

    def _decide_use_crop_for_clip(self) -> bool:
        if self.mode == "train":
            if not self.crop:
                return False
            return bool(np.random.rand() < self.crop_prob)
        else:
            if self.eval_use_crop is None:
                return self.crop
            return bool(self.eval_use_crop)

    def get_kps_gt(self, work_indices, seq_name, gt_idx, imgs_path):
        frames_resized, kps_xy_resized = [], []
        gt_q, gt_r, vis_list = [], [], []
        crop_xywh_list = []

        use_crop_clip = self._decide_use_crop_for_clip()

        for idx in work_indices:
            img_name = f"img{str(idx).zfill(3)}_{seq_name}{self._image_ext()}"
            im_pth = osp.join(imgs_path, img_name)
            img_rgb = self._read_frame(im_pth)

            entry = self.data[gt_idx + idx]
            bbox = np.asarray(entry["bbox"], dtype=np.float32)
            xy, vis = self._parse_entry_xy_vis(entry)

            img_r, kp_r, crop_xywh = self._preprocess_single(img_rgb, xy, bbox, use_crop=use_crop_clip)
            crop_xywh_list.append(crop_xywh)
            frames_resized.append(img_r)
            kps_xy_resized.append(kp_r)

            gt_q.append(torch.tensor(entry["q_xyzw"], dtype=torch.float32))
            gt_r.append(torch.tensor(entry["r_xyz"],   dtype=torch.float32))
            vis_list.append(torch.from_numpy(vis.astype(np.float32)))

        T = len(frames_resized)

        if self.mode == "train" and T > 0:
            aug = self.spatial_transforms if use_crop_clip else (self.spatial_transforms_full or self.spatial_transforms)
            if aug is not None:
                out0 = aug(image=frames_resized[0], keypoints=kps_xy_resized[0])
                replay = out0["replay"]
                frames_resized[0] = out0["image"]
                kps_xy_resized[0] = np.asarray(out0["keypoints"], dtype=np.float32)

                crop_xywh_list[0] = update_crop_xywh_from_replay(crop_xywh_list[0], replay, self.input_size)

                for t in range(1, T):
                    outt = A.ReplayCompose.replay(replay, image=frames_resized[t], keypoints=kps_xy_resized[t])
                    frames_resized[t] = outt["image"]
                    kps_xy_resized[t] = np.asarray(outt["keypoints"], dtype=np.float32)
                    crop_xywh_list[t] = update_crop_xywh_from_replay(crop_xywh_list[t], replay, self.input_size)

        frames = []
        for t in range(T):
            img = torch.from_numpy(frames_resized[t]).permute(2, 0, 1).float() / 255.0
            img = imagenet_normalize_tensor(img)
            frames.append(img)
        frames = torch.stack(frames)

        gt_q = torch.stack(gt_q)
        gt_r = torch.stack(gt_r)
        vis  = torch.stack(vis_list)

        Hm, Wm = self.hm_size
        inp = float(self.input_size)

        kps_img = torch.from_numpy(np.stack(kps_xy_resized, axis=0)).float()

        kps_hm = kps_img.clone()
        kps_hm[..., 0] = kps_hm[..., 0] * ((Wm - 1.0) / (inp - 1.0))
        kps_hm[..., 1] = kps_hm[..., 1] * ((Hm - 1.0) / (inp - 1.0))

        hm_target = make_gaussian_heatmaps(kps_hm, H=Hm, W=Wm, sigma_px=self.hm_sigma_px)

        x_norm = 2.0 * (kps_hm[..., 0] / max(Wm - 1.0, 1.0)) - 1.0
        y_norm = 2.0 * (kps_hm[..., 1] / max(Hm - 1.0, 1.0)) - 1.0
        kp_target_norm = torch.stack([x_norm, y_norm], dim=-1)

        if use_crop_clip:
            croped_xywh_tensor = torch.stack([torch.tensor(x, dtype=torch.float32) for x in crop_xywh_list])
        else:
            dummy = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
            croped_xywh_tensor = dummy.repeat(len(frames_resized), 1)

        return frames, kp_target_norm, gt_q, gt_r, vis, hm_target, croped_xywh_tensor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        seq_name, work_start, work_end, imgs_path, gt_idx = self.dataset[idx]

        raw = np.arange(work_start, work_end)
        sampled = raw[:: self.sampling_rate]

        T_out = int(np.ceil(self.short_mem / self.sampling_rate))

        if len(sampled) == 0:
            sampled = np.array([work_start], dtype=np.int64)

        if len(sampled) < T_out:
            pad = np.full((T_out - len(sampled),), sampled[-1], dtype=sampled.dtype)
            work_indices = np.concatenate([sampled, pad], axis=0)
        else:
            work_indices = sampled[:T_out]

        frames, kps_norm, gt_q, gt_r, vis, hm, crop_xywh = self.get_kps_gt(
            work_indices, seq_name, gt_idx, imgs_path
        )

        R_seq = torch.stack([quat_to_rotmat(q) for q in gt_q])
        if R_seq.shape[0] >= 2:
            delta_R = torch.matmul(R_seq[1:], R_seq[:-1].transpose(1, 2))
            delta_t = gt_r[1:] - gt_r[:-1]
            delta_R_vec = rotmat_to_logvec(delta_R)
            delta_Rt_gt = torch.cat([delta_R_vec, delta_t], dim=-1)
        else:
            delta_Rt_gt = torch.zeros((0, 6), dtype=torch.float32)

        frame_ids = [[f"{seq_name}_{int(t)}" for t in work_indices.tolist()]]

        return frames, kps_norm, gt_q, gt_r, vis, hm, delta_Rt_gt, frame_ids, crop_xywh


# =============================
# SPADES dataset
# =============================
class SPADESDataset(_BaseSpaceDataset):
    def __init__(self, gt_folder, short_mem, input_size, sampling_rate,
                 clip_stride: int = 0,
                 crop=True,
                 spatial_transforms=None,
                 mode="train",
                 heatmap_out=(56, 56),
                 gaussian_sigma_px: float = 2.5,
                 crop_prob: float = 0.5,
                 eval_use_crop: Optional[bool] = None,
                 spatial_transforms_full: Optional[A.ReplayCompose] = None):
        super().__init__(gt_folder, short_mem, input_size, sampling_rate, clip_stride,
                         crop, spatial_transforms, mode, heatmap_out, gaussian_sigma_px,
                         crop_prob=crop_prob, eval_use_crop=eval_use_crop,
                         spatial_transforms_full=spatial_transforms_full)

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

    def __init__(self, gt_folder, short_mem, input_size, sampling_rate,
                 clip_stride: int = 0,
                 crop=True,
                 spatial_transforms=None,
                 mode="train",
                 heatmap_out=(56, 56),
                 gaussian_sigma_px: float = 1.8,
                 crop_prob: float = 0.5,
                 eval_use_crop: Optional[bool] = None,
                 spatial_transforms_full: Optional[A.ReplayCompose] = None):
        super().__init__(gt_folder, short_mem, input_size, sampling_rate, clip_stride,
                         crop, spatial_transforms, mode, heatmap_out, gaussian_sigma_px,
                         crop_prob=crop_prob, eval_use_crop=eval_use_crop,
                         spatial_transforms_full=spatial_transforms_full)

        # SPARK uses a single train.json
        self.data = self._load_json(osp.join(gt_folder, "train.json"))
        self._make_dataset(gt_folder, mode)

    def _json_filename(self) -> str:
        return "train.json"

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
    gt_folder = "/project/scratch/p200249/space_data/SPADES_RGB"
    ds = SPADESDataset(
        gt_folder=gt_folder,
        short_mem=96,
        input_size=224,
        sampling_rate=6,
        clip_stride=48,   # passage
        crop=True,
        crop_prob=1.0,
        mode="train",
        heatmap_out=(56, 56),
        gaussian_sigma_px=1.2,
        eval_use_crop=True,
    )
    print("num clips:", len(ds))