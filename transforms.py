# transforms.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import albumentations as A
import cv2
# from albumentations.augmentations.geometric.resize import Resize
from albumentations.core.transforms_interface import BasicTransform
from albumentations.core.composition import BaseCompose

# Albumentations stubs use something like TransformType = BasicTransform | BaseCompose
TransformType = BasicTransform | BaseCompose


class ClipAugment:
    """
    Single augmentation object for temporal clips.

    Pass 1 — geometric (clip-consistent):
        Sampled once from frame 0 via ReplayCompose, then replayed identically
        on all subsequent frames so the spatial structure stays coherent across
        the clip (no random jumps in viewpoint between frames).

    Pass 2 — photometric (frame-independent):
        Re-sampled independently for every frame, simulating realistic
        per-frame lighting variation (exposure flicker, shadow edges, etc.).

    Usage in dataset:
        frames, keypoints = self.spatial_transforms(frames, keypoints)
    """

    def __init__(self, geo: A.ReplayCompose, photo: A.Compose):
        self.geo   = geo
        self.photo = photo

    def __call__(
        self,
        frames:    List[np.ndarray],   # list[T] of (H,W,3) uint8
        keypoints: List[np.ndarray],   # list[T] of (K,2) float32
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        T = len(frames)
        if T == 0:
            return frames, keypoints

        # ---- Pass 1: geometric — same params replayed across all frames ----
        out0 = self.geo(image=frames[0], keypoints=keypoints[0].tolist())
        frames[0]    = out0["image"]
        keypoints[0] = np.asarray(out0["keypoints"], dtype=np.float32)
        replay = out0["replay"]
        for t in range(1, T):
            outt = A.ReplayCompose.replay(
                replay, image=frames[t], keypoints=keypoints[t].tolist()
            )
            frames[t]    = outt["image"]
            keypoints[t] = np.asarray(outt["keypoints"], dtype=np.float32)

        # ---- Pass 2: photometric — independent per frame ----
        for t in range(T):
            outt = self.photo(image=frames[t], keypoints=keypoints[t].tolist())
            frames[t]    = outt["image"]
            keypoints[t] = np.asarray(outt["keypoints"], dtype=np.float32)

        return frames, keypoints


# def build_transforms(img_size: int = 512) -> A.Compose:
#     transforms: list[TransformType] = [
#         # A.Resize(img_size, img_size, interpolation=cv2.INTER_AREA),

#         # A.ShiftScaleRotate(
#         #     shift_limit=0.03,
#         #     scale_limit=0.05,
#         #     rotate_limit=5,
#         #     border_mode=cv2.BORDER_CONSTANT,
#         #     fill=0,     # prefer fill (newer API); if your version only supports value, use value=0
#         #     p=0.5,
#         # ),

#         A.RandomBrightnessContrast(0.3, 0.3, p=0.5),
#         A.RandomGamma(gamma_limit=(70, 130), p=0.4),
#         A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
#         A.HueSaturationValue(5, 10, 10, p=0.25),
#     ]

#     return A.Compose(
#         transforms=transforms,
#         keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
#     )

def build_transforms(img_size=224):
    return A.Compose(
        [
            # Asymmetric brightness: bias heavily toward darkening to simulate
            # space shadow / eclipse frames (spacecraft entering Earth's umbra).
            # Old: symmetric ±0.4. New: (-0.7, +0.2) — 3× more dark headroom.
            A.RandomBrightnessContrast(
                brightness_limit=(-0.7, 0.2),
                contrast_limit=(-0.4, 0.4),
                p=0.6,
            ),
            # Gamma >100 darkens, <100 brightens. Old range (60–140) spent half
            # its probability brightening. New range (80–220) spends ~2/3 of it
            # darkening, covering extreme shadow conditions.
            A.RandomGamma(gamma_limit=(80, 220), p=0.5),
            # CLAHE: normalises local contrast; critical for extracting features
            # from near-black shadow frames. Raised from p=0.2 → p=0.5.
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.5),
            # HSV: mild colour jitter unchanged — space frames are near-greyscale
            # in shadow anyway so this has small effect there.
            A.HueSaturationValue(5, 10, 10, p=0.25),
            # Gaussian noise: shot/read noise increases significantly in low-light.
            # Borrowed from DINOv3 augmentation philosophy (add signal-level noise).
            A.GaussNoise(var_limit=(10.0, 60.0), p=0.4),
            # Gaussian blur: cameras auto-extend exposure in dark → motion blur.
            # Directly borrowed from dinov3/dinov3/data/transforms.py GaussianBlur.
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def build_clip_augmentation(img_size: int = 224) -> ClipAugment:
    """
    Returns a ClipAugment that applies:
      - Geometric transforms identically across all frames in a clip (replay).
      - Photometric transforms independently per frame.

    No HorizontalFlip: spacecraft asymmetry means flipping invalidates GT keypoints.
    """
    geo = A.ReplayCompose(
        [
            # Small translation/scale/rotation: realistic orbital viewpoint changes.
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.10,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=0.6,
            ),
            # Mild perspective warp: simulates slight camera tilt variation.
            A.Perspective(scale=(0.02, 0.06), p=0.3),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

    photo = build_transforms(img_size=img_size)

    return ClipAugment(geo=geo, photo=photo)


# def build_transforms(aug_mode: str = "photometric",
#                      allow_tiny_crop: bool = False,
#                      img_size: int = 224):
#     """
#     Returns an Albumentations ReplayCompose suitable for per-frame use.
#     """
#     # ---- default crop/resize ----
#     # tiny_geo = []
#     # if allow_tiny_crop:
#     #     tiny_geo = [A.RandomResizedCrop(size=(img_size, img_size),
#     #                                     scale=(0.95, 1.0),
#     #                                     ratio=(0.99, 1.01),
#     #                                     p=0.5)]
#     # else:
#     #     tiny_geo = [A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_AREA)]

#     # ---- photometric only ----
#     photometric = [
#         A.RandomBrightnessContrast(brightness_limit=0.40, contrast_limit=0.40, p=0.5),
#         A.RandomGamma(gamma_limit=(60, 140), p=0.4),
#         A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.20),
#         A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.25),
#     ]

#     # ---- geometry + photometric (recommended for KP supervision) ----
#     kp_geo = [
#         A.RandomResizedCrop(size=(img_size, img_size),
#                             scale=(0.85, 1.0),  # stronger than tiny
#                             ratio=(0.9, 1.1),
#                             p=0.8),
#         A.ShiftScaleRotate(shift_limit=0.05,   # ±5% shift
#                            scale_limit=0.1,    # ±10% zoom
#                            rotate_limit=10,    # ±10° rotation
#                            border_mode=cv2.BORDER_CONSTANT,
#                            fill=(0, 0, 0),
#                            fill_mask=0,
#                            p=0.6),
#         A.HorizontalFlip(p=0.5),   # spacecraft are usually symmetric; safe flip
#         A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
#         A.RandomGamma(gamma_limit=(70, 130), p=0.4),
#         A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
#     ]

#     if aug_mode == "kp_geo":
#         pipeline = [t for t in kp_geo if not isinstance(t, A.HorizontalFlip)]
#     else:
#         pipeline = photometric

#     return pipeline
#     # return A.ReplayCompose(
#     #     pipeline,
#     #     # keypoint_params=A.KeypointParams(format="xy", remove_invisible=False)
#     # )