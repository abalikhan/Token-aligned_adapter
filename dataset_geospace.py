# dataset_geospace.py  (CLEAN, Theseus-free, GeoSpaceFormer-aligned)
import json
import os
import os.path as osp
from typing import Tuple, List, Optional

import cv2
import numpy as np
import torch
import torch.utils.data as data

import albumentations as A
import albumentations.augmentations.geometric.resize as albu_resize


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
# Gaussian heatmaps
# =============================
def make_gaussian_heatmaps(
    kps_hw: torch.Tensor, H: int = 56, W: int = 56, sigma_px: float = 1.8
) -> torch.Tensor:
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
        cam_intrinsics_json: Optional[str] = None,   # loads FULL K once (used by train_val via get_K_full_from_loader)
    ):
        self.gt_folder = gt_folder
        self.spatial_transforms = spatial_transforms
        self.mode = mode

        self.crop = bool(crop)
        self.eval_use_crop = eval_use_crop

        self.short_mem = int(short_mem)
        self.input_size = int(input_size)
        self.sampling_rate = int(sampling_rate)
        self.hm_size = tuple(heatmap_out)
        self.hm_sigma_px = float(gaussian_sigma_px)
        self.clip_stride = int(clip_stride) if int(clip_stride) > 0 else int(short_mem)

        self.data: List[dict] = []
        self.dataset: List[list] = []

        # FULL intrinsics cached for training loop usage
        self.K_full = None
        if cam_intrinsics_json is not None and str(cam_intrinsics_json) != "":
            with open(cam_intrinsics_json, "r") as f:
                c = json.load(f)
            K = np.array(c["cameraMatrix"], dtype=np.float32)
            self.K_full = torch.from_numpy(K)  # (3,3)

    def _json_filename(self) -> str:
        return f"{self.mode}.json"

    def _image_ext(self) -> str:
        return ".png"

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
        self.dataset.clear()
        self.data = self._load_json(osp.join(gt_folder, self._json_filename()))

        first_idx = {}
        for i, e in enumerate(self.data):
            seq = e["filename"].split(".")[0].split("_")[1]
            if seq not in first_idx:
                first_idx[seq] = i

        entries = self._build_split(first_idx)

        T_out = int(np.ceil(self.short_mem / self.sampling_rate))
        required_raw = 1 + (T_out - 1) * self.sampling_rate

        for seq_name, gt_idx in entries:
            imgs_path = osp.join(self.gt_folder, "images", seq_name)
            if not osp.isdir(imgs_path):
                imgs_path = osp.join(self.gt_folder, seq_name)

            length = len([n for n in os.listdir(imgs_path) if n.endswith(self._image_ext())])
            if length <= 0:
                continue

            last_ws = length - required_raw
            if last_ws < 0:
                continue

            step = max(1, int(self.clip_stride))
            seed = int(np.random.randint(step)) if mode == "train" else 0

            ws = seed
            while ws <= last_ws:
                we = min(ws + self.short_mem, length)
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
        xy = kpts[:, :2].copy()     # FULL px
        vis = kpts[:, 2].copy()
        return xy, vis

    def _decide_use_crop_for_clip(self) -> bool:
        if self.mode == "train":
            return bool(self.crop)
        if self.eval_use_crop is None:
            return bool(self.crop)
        return bool(self.eval_use_crop)

    def _preprocess_single(
        self,
        frame_rgb: np.ndarray,
        kp_xy_full: np.ndarray,     # FULL px
        bbox_xyxy: np.ndarray,
        use_crop: bool,
    ):
        """
        Returns:
          img_r: resized RGB (input_size,input_size,3)
          kp_r:  resized keypoints in INPUT px (input_size space)
          crop_xywh_norm: crop box normalized w.r.t FULL frame (x/W, y/H, w/W, h/H)
          meta: [scale, pad_left, pad_top, W_full, H_full] (letterbox) OR NaNs (crop)
        """
        H, W, _ = frame_rgb.shape
        meta = [float("nan"), float("nan"), float("nan"), float(W), float(H)]

        if use_crop:
            x1, y1, x2, y2 = recenter_bbox_with_kps(
                bbox_xyxy, kp_xy_full, W, H,
                expand=1.2, center_thresh=0.15, min_size=32, make_square=True
            )

            transform = A.Compose(
                [
                    A.Crop(x_min=int(x1), y_min=int(y1), x_max=int(x2), y_max=int(y2)),
                    albu_resize.Resize(self.input_size, self.input_size, p=1.0),
                ],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            )

            out = transform(image=frame_rgb, keypoints=kp_xy_full)
            img_r = out["image"]
            kp_r = np.asarray(out["keypoints"], dtype=np.float32)  # INPUT px

            crop_xywh_norm = [x1 / W, y1 / H, (x2 - x1) / W, (y2 - y1) / H]
            return img_r, kp_r, crop_xywh_norm, meta

        # -------- FULL FRAME letterbox path --------
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

        corners = np.array([[0.0, 0.0], [float(W - 1), float(H - 1)]], dtype=np.float32)
        kp_aug = np.concatenate([kp_xy_full.astype(np.float32), corners], axis=0)

        out = transform(image=frame_rgb, keypoints=kp_aug)
        img_r = out["image"]
        kp_r_aug = np.asarray(out["keypoints"], dtype=np.float32)

        kp_r = kp_r_aug[:-2]
        c_tl = kp_r_aug[-2]
        c_br = kp_r_aug[-1]

        pad_left = float(c_tl[0])
        pad_top  = float(c_tl[1])

        s_x = (float(c_br[0]) - float(c_tl[0])) / max(float(W - 1), 1.0)
        s_y = (float(c_br[1]) - float(c_tl[1])) / max(float(H - 1), 1.0)
        scale = 0.5 * (s_x + s_y)

        meta = [float(scale), float(pad_left), float(pad_top), float(W), float(H)]
        crop_xywh_norm = [0.0, 0.0, 1.0, 1.0]
        return img_r, kp_r, crop_xywh_norm, meta

    def get_kps_gt(self, work_indices, seq_name, gt_idx, imgs_path):
        frames_resized: List[np.ndarray] = []
        kps_xy_resized: List[np.ndarray] = []
        kps_xy_full_list: List[torch.Tensor] = []

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

            entry = self.data[gt_idx + idx]
            bbox = np.asarray(entry["bbox"], dtype=np.float32)

            xy_full, vis = self._parse_entry_xy_vis(entry)   # FULL px

            img_r, kp_r, crop_xywh_norm, meta = self._preprocess_single(
                img_rgb, xy_full, bbox, use_crop=use_crop_clip
            )

            frames_resized.append(img_r)
            kps_xy_resized.append(kp_r)

            kps_xy_full_list.append(torch.tensor(xy_full, dtype=torch.float32))
            crop_xywh_list.append(torch.tensor(crop_xywh_norm, dtype=torch.float32))
            meta_list.append(torch.tensor(meta, dtype=torch.float32))

            gt_q.append(torch.tensor(entry["q_xyzw"], dtype=torch.float32))  # xyzw
            gt_r.append(torch.tensor(entry["r_xyz"], dtype=torch.float32))
            vis_list.append(torch.from_numpy(vis.astype(np.float32)))

        T = len(frames_resized)

        # photometric aug only (geometry unchanged)
        if self.mode == "train" and T > 0:
            aug = self.spatial_transforms
            if aug is not None:
                for t in range(T):
                    outt = aug(image=frames_resized[t], keypoints=kps_xy_resized[t])
                    frames_resized[t] = outt["image"]
                    kps_xy_resized[t] = np.asarray(outt["keypoints"], dtype=np.float32)

        # to torch, ImageNet norm
        frames = []
        for t in range(T):
            img = torch.from_numpy(frames_resized[t]).permute(2, 0, 1).float() / 255.0
            img = imagenet_normalize_tensor(img)
            frames.append(img)
        frames = torch.stack(frames, dim=0)  # (T,3,H,W)

        gt_q = torch.stack(gt_q, dim=0)      # (T,4)
        gt_r = torch.stack(gt_r, dim=0)      # (T,3)
        vis  = torch.stack(vis_list, dim=0)  # (T,K)
        crop_xywh = torch.stack(crop_xywh_list, dim=0)      # (T,4) normalized wrt FULL
        meta_tensor = torch.stack(meta_list, dim=0)         # (T,5) [scale,pad_l,pad_t,W_full,H_full]

        kps_full_px = torch.stack(kps_xy_full_list, dim=0)  # (T,K,2) full px

        # heatmap + normalized keypoints in HEATMAP coordinates
        Hm, Wm = self.hm_size
        inp = float(self.input_size)

        kps_img = torch.from_numpy(np.stack(kps_xy_resized, axis=0)).float()  # (T,K,2) input px

        kps_hm = kps_img.clone()
        kps_hm[..., 0] = kps_hm[..., 0] * ((Wm - 1.0) / (inp - 1.0))
        kps_hm[..., 1] = kps_hm[..., 1] * ((Hm - 1.0) / (inp - 1.0))

        hm_target = make_gaussian_heatmaps(kps_hm, H=Hm, W=Wm, sigma_px=self.hm_sigma_px)  # (T,K,Hm,Wm)

        x_norm = 2.0 * (kps_hm[..., 0] / max(Wm - 1.0, 1.0)) - 1.0
        y_norm = 2.0 * (kps_hm[..., 1] / max(Hm - 1.0, 1.0)) - 1.0
        kp_target_norm = torch.stack([x_norm, y_norm], dim=-1)  # (T,K,2) in [-1,1] on heatmap grid

        if not use_crop_clip:
            crop_xywh = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32).view(1, 4).repeat(T, 1)

        return (
            frames,
            kp_target_norm,
            kps_full_px,
            gt_q,
            gt_r,
            vis,
            hm_target,
            crop_xywh,
            meta_tensor,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        seq_name, work_start, work_end, imgs_path, gt_idx = self.dataset[idx]

        # TEST: return full sampled sequence (variable T)
        if self.mode == "test":
            raw = np.arange(work_start, work_end, dtype=np.int64)
            work_indices = raw[:: int(self.sampling_rate)]
            if work_indices.size == 0:
                work_indices = np.array([work_start], dtype=np.int64)

            frames, kps_norm, kps_full_px, gt_q, gt_r, vis, hm, crop_xywh, meta_tensor = \
                self.get_kps_gt(work_indices, seq_name, gt_idx, imgs_path)

            frame_ids = [[f"{seq_name}_{int(t)}" for t in work_indices.tolist()]]
            return (
                frames, kps_norm, kps_full_px, gt_q, gt_r, vis, hm,
                frame_ids, crop_xywh, meta_tensor, seq_name
            )

        # TRAIN/VAL: fixed-length clip
        T_out = int(np.ceil(self.short_mem / self.sampling_rate))
        work_indices = work_start + np.arange(T_out, dtype=np.int64) * int(self.sampling_rate)

        frames, kps_norm, kps_full_px, gt_q, gt_r, vis, hm, crop_xywh, meta_tensor = \
            self.get_kps_gt(work_indices, seq_name, gt_idx, imgs_path)

        frame_ids = [[f"{seq_name}_{int(t)}" for t in work_indices.tolist()]]

        return (
            frames, kps_norm, kps_full_px, gt_q, gt_r, vis, hm,
            frame_ids, crop_xywh, meta_tensor, seq_name
        )


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
        cam_intrinsics_json: Optional[str] = None,
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
            cam_intrinsics_json=cam_intrinsics_json,
        )

        if mode == "test":
            self.make_full_sequence_test_dataset(gt_folder)
        else:
            self._make_dataset(gt_folder, mode)

    def _image_ext(self) -> str:
        return ".png"


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
        cam_intrinsics_json: Optional[str] = None,
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
            cam_intrinsics_json=cam_intrinsics_json,
        )

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
