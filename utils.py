import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
import json
from torch import nn
import torch
from losses import softargmax_2d_from_logits
from typing import Dict, Tuple, Optional, List, Any
# from pytorch3d.transforms import quaternion_to_matrix 
import torch.nn.functional as F
import torch.distributed as dist
#from torch._six import inf
import random


import os
import math
import numpy as np
import torch
import torch.nn.functional as F

def _infer_patch_grid(P: int) -> int:
    g = int(round(math.sqrt(P)))
    return g if g * g == P else g

# ---- ImageNet denorm for visualization ----
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1)

@torch.no_grad()
def denorm_imagenet_bt(frames_btchw: torch.Tensor) -> torch.Tensor:
    """
    frames_btchw: (B,T,3,H,W) ImageNet-normalized
    returns: (B,T,3,H,W) in [0,1]
    """
    mean = _IMAGENET_MEAN.to(frames_btchw.device)
    std  = _IMAGENET_STD.to(frames_btchw.device)
    x = frames_btchw * std + mean
    return x.clamp(0.0, 1.0)

@torch.no_grad()
def save_token_similarity_overlay_bt(
    out: dict,
    frames: torch.Tensor,     # (B,T,3,H,W) in [0,1] ideally
    kps_gt: torch.Tensor,     # (B,T,K,2) in [-1,1]
    vis: torch.Tensor,        # (B,T,K) in {0,1}
    cfg: dict,
    save_path: str,
    b: int = 0,
    t: int = 0,
    sigma_patch: float = 1.25,   # radius in patch units
):
    """
    Saves an overlay that shows where patch tokens are most similar to GT keypoint regions.
    Requires out["patch_tokens_prenorm"] (B,T,P,D).
    """
    if "patch_tokens_prenorm" not in out:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    tok = out["patch_tokens_prenorm"]  # (B,T,P,D)
    B, T, P, D = tok.shape

    Himg = int(cfg["data"]["input_size"])
    Wimg = int(cfg["data"]["input_size"])

    # infer patch grid
    g = _infer_patch_grid(P)
    Hpatch = Wpatch = g

    tok_bt = tok[b, t]  # (P,D)
    tok_norm = F.normalize(tok_bt.float(), dim=-1)  # (P,D)

    # patch grid coordinates
    yy, xx = torch.meshgrid(
        torch.arange(Hpatch, device=tok.device),
        torch.arange(Wpatch, device=tok.device),
        indexing="ij",
    )
    grid_xy = torch.stack([xx, yy], dim=-1).view(-1, 2).float()  # (P,2)

    # kp norm -> patch coords
    kp_x = (kps_gt[b, t, :, 0] + 1) * 0.5 * (Wpatch - 1)
    kp_y = (kps_gt[b, t, :, 1] + 1) * 0.5 * (Hpatch - 1)
    kp_xy = torch.stack([kp_x, kp_y], dim=-1)  # (K,2)

    K = kp_xy.shape[0]
    sim_maps = []

    for k in range(K):
        if float(vis[b, t, k].item()) < 0.5:
            continue

        d2 = ((grid_xy - kp_xy[k:k+1]) ** 2).sum(dim=-1)      # (P,)
        w = torch.exp(-0.5 * d2 / (sigma_patch ** 2))          # (P,)
        w = w / (w.sum() + 1e-6)

        proto = (w.unsqueeze(-1) * tok_bt).sum(dim=0)          # (D,)
        proto = F.normalize(proto.float(), dim=-1)

        sim = (tok_norm @ proto)                                # (P,)
        sim_map = sim.view(1, 1, Hpatch, Wpatch)                # (1,1,g,g)
        sim_maps.append(sim_map)

    if len(sim_maps) == 0:
        return

    # aggregate: max over K visible keypoints
    sim_map = torch.cat(sim_maps, dim=0).max(dim=0, keepdim=True)[0]  # (1,1,g,g)

    # normalize to [0,1]
    sm = sim_map[0, 0]
    sm = (sm - sm.min()) / (sm.max() - sm.min() + 1e-6)
    sm = sm.view(1, 1, Hpatch, Wpatch)

    # upsample to image size
    sm_up = F.interpolate(sm, size=(Himg, Wimg), mode="bilinear", align_corners=False)  # (1,1,H,W)
    sm_up = sm_up[0, 0].detach().cpu().numpy()

    # base image
    # img = frames[b, t].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    frames_vis = denorm_imagenet_bt(frames)
    img = frames_vis[b, t].detach().cpu().permute(1, 2, 0).numpy()
    img_u8 = (img * 255.0).astype(np.uint8)

    # overlay with OpenCV colormap
    try:
        import cv2
        heat = (sm_up * 255.0).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)  # (H,W,3) BGR
        out_img = cv2.addWeighted(img_u8[..., ::-1], 0.65, heat, 0.35, 0)  # convert RGB->BGR for cv2
        cv2.imwrite(save_path, out_img)
    except Exception:
        # fallback: save heatmap only
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.imshow(sm_up, cmap="jet", alpha=0.35)
        plt.axis("off")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()

@torch.no_grad()
def save_kp_overlay_bt(frames, kps_gt_norm, hm_logits, vis, cfg, save_path, b=0, t=0):
    """
    Same idea as save_kp_overlay, but lets you pick (b,t).
    frames: (B,T,3,H,W) in [0,1] ideally
    kps_gt_norm: (B,T,K,2) in [-1,1]
    hm_logits: (B,T,K,Hm,Wm)
    vis: (B,T,K)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # f = frames[b, t].detach().cpu().clamp(0, 1)  # (3,H,W)
    frames_vis = denorm_imagenet_bt(frames)
    f = frames_vis[b, t].detach().cpu()   # already [0,1]
    B, T, K, Hm, Wm = hm_logits.shape
    H = int(cfg["data"]["input_size"])
    W = int(cfg["data"]["input_size"])

    from losses import softargmax_2d_from_logits
    hm_ = hm_logits[b:b+1, t:t+1]  # (1,1,K,Hm,Wm)
    pred_px_hm = softargmax_2d_from_logits(hm_)[0, 0]  # (K,2) in hm px

    sx, sy = float(W) / float(Wm), float(H) / float(Hm)
    pred_px_img = pred_px_hm.clone()
    pred_px_img[:, 0] = pred_px_hm[:, 0] * sx
    pred_px_img[:, 1] = pred_px_hm[:, 1] * sy

    gx = ((kps_gt_norm[b, t, :, 0] + 1.0) * 0.5) * (W - 1.0)
    gy = ((kps_gt_norm[b, t, :, 1] + 1.0) * 0.5) * (H - 1.0)
    gt_px_img = torch.stack([gx, gy], dim=-1).detach().cpu()  # (K,2)

    vis_ = None
    if vis is not None and vis.numel() > 0:
        vis_ = (vis[b, t].detach().cpu() > 0).float()

    img = (f.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8).copy()

    try:
        import cv2
        col_gt = (0, 255, 0)       # green
        col_pred = (255, 0, 255)   # magenta
        r = max(2, int(0.004 * max(H, W)))
        th = max(1, int(0.002 * max(H, W)))

        for i in range(gt_px_img.shape[0]):
            px_gt  = (int(gt_px_img[i, 0].item()),  int(gt_px_img[i, 1].item()))
            px_prd = (int(pred_px_img[i, 0].item()), int(pred_px_img[i, 1].item()))
            if vis_ is None or vis_[i] > 0.5:
                cv2.circle(img, px_gt,  r, col_gt,  -1, lineType=cv2.LINE_AA)
                cv2.circle(img, px_prd, r, col_pred, th, lineType=cv2.LINE_AA)
            else:
                cv2.circle(img, px_gt,  r, (0,128,0), -1, lineType=cv2.LINE_AA)
                cv2.circle(img, px_prd, r, (128,0,128), th, lineType=cv2.LINE_AA)

        cv2.imwrite(save_path, img)
    except Exception:
        from PIL import Image, ImageDraw
        pimg = Image.fromarray(img)
        draw = ImageDraw.Draw(pimg)
        r = max(2, int(0.004 * max(H, W)))

        def _disk(xy, color, fill=True):
            x, y = xy
            bbox = [x - r, y - r, x + r, y + r]
            if fill:
                draw.ellipse(bbox, fill=color)
            else:
                draw.ellipse(bbox, outline=color, width=max(1, r//2))

        for i in range(gt_px_img.shape[0]):
            px_gt  = (float(gt_px_img[i, 0].item()),  float(gt_px_img[i, 1].item()))
            px_prd = (float(pred_px_img[i, 0].item()), float(pred_px_img[i, 1].item()))
            if vis_ is None or vis_[i] > 0.5:
                _disk(px_gt,  (0,255,0),  fill=True)
                _disk(px_prd, (255,0,255), fill=False)
            else:
                _disk(px_gt,  (0,128,0),  fill=True)
                _disk(px_prd, (128,0,128), fill=False)

        pimg.save(save_path)

# --------------------------------
# Scheduler helper for Token loss
# --------------------------------
def get_w_token_for_epoch(cfg, epoch: int) -> float:
    sched = cfg.get("loss", {}).get("w_token_schedule", None)
    base  = float(cfg.get("loss", {}).get("w_token", 0.0))

    if not sched:
        return base

    start = float(sched.get("start", base))
    end   = float(sched.get("end", base))
    e0    = int(sched.get("start_epoch", 0))
    e1    = int(sched.get("end_epoch", 0))

    if epoch <= e0:
        return start
    if epoch >= e1:
        return end

    # linear interpolation
    alpha = (epoch - e0) / max(1, (e1 - e0))
    return start + alpha * (end - start)
# refine_kps_eval.py
# Complete utility to refine keypoints from heatmaps (cropped), remap to full image using bbox, and get final 2D keypoints

from scipy.spatial.transform import Rotation as R


def draw_kps_on_frame(frame: np.ndarray, kps: np.ndarray, conf: np.ndarray = None, radius=4, color=(0, 0, 255), thickness=-1):
    """
    Draw keypoints on an image.
    
    Args:
        frame (np.ndarray): RGB image as numpy array (H, W, 3)
        kps (np.ndarray): Keypoints (K, 2) in image coordinates
        conf (np.ndarray, optional): Keypoint confidences (K,)
        radius (int): Circle radius
        color (tuple): BGR color tuple
        thickness (int): Circle thickness (-1 = filled)
    
    Returns:
        np.ndarray: Image with keypoints drawn
    """
    img = frame.copy()
    K = kps.shape[0]
    for i in range(K):
        x, y = int(kps[i, 0]), int(kps[i, 1])
        if conf is not None and conf[i] < 0.1:
            continue  # Skip low-confidence keypoints
        if x < 0 or y < 0 or x >= img.shape[1] or y >= img.shape[0]:
            continue  # Skip invalid keypoints
        cv2.circle(img, (x, y), radius, color, thickness)
    return img

class PoseEstimator:
    def __init__(self, model_coordinates, camera_matrix):
        # 3D model points.
        self.model_points = model_coordinates

        # Camera intrinsics
        # cam = Camera()
        self.camera_matrix = camera_matrix

        # Assuming no lens distortion
        self.dist_coefs = np.zeros((4, 1))

        # Rotation vector and translation vector
        self.r_vec = None
        self.t_vec = None

    def solve_pose(self, image_points, array, pnpconfig, flag=0):
        """
        Solve pose from image points
        Return (rotation_vector and translation_vector) as output.
        """
        if (pnpconfig.pnp_solver == 'ransac_pnp'):
            _, rvec, tvec, inliers = cv2.solvePnPRansac(
                self.model_points[array],
                image_points[array],
                self.camera_matrix,
                self.dist_coefs,
                flags=0)
        
        elif (pnpconfig.pnp_solver == 'ransac_epnp'):
            _, rvec, tvec, inliers = cv2.solvePnPRansac(
                self.model_points[array],
                image_points[array],
                self.camera_matrix,
                self.dist_coefs,
                flags=3)
        else:
            data_out = cv2.solvePnP(
                self.model_points[array],
                image_points[array],
                self.camera_matrix,
                self.dist_coefs,
                flags=0)
            _, rvec, tvec = data_out

        if (pnpconfig.refinement_algorithm == 'iterative'):
            pnp_flag, rvec, tvec, inliers = cv2.solvePnPRansac(
                self.model_points[array],
                image_points[array],
                self.camera_matrix,
                self.dist_coefs,
                tvec=tvec,
                rvec=rvec,
                useExtrinsicGuess=True,
                flags=0)
        
        elif (pnpconfig.refinement_algorithm == 'LM'):
            rvec, tvec = cv2.solvePnPRefineLM(
                self.model_points[array],
                image_points[array],
                self.camera_matrix,
                self.dist_coefs,
                tvec=tvec,
                rvec=rvec)
        
        elif (pnpconfig.refinement_algorithm == 'VVS'):
            rvec, tvec = cv2.solvePnPRefineVVS(
                self.model_points[array],
                image_points[array],
                self.camera_matrix,
                self.dist_coefs,
                tvec=tvec,
                rvec=rvec)
        else:
            pass

        rmat, _ = cv2.Rodrigues(rvec)
        rot_m = R.from_matrix(rmat)
        q = rot_m.as_quat()  # (x, y, z, w) format.
        r = tvec.squeeze()
        
        return q, r, flag

class PnPConfig:
    def __init__(self, pnp_solver='ransac_pnp',
                 refinement_algorithm='iterative'):
        
        # Specify the PnP solver to use
        self.pnp_solver = pnp_solver
        print('Pose solver:', self.pnp_solver)

        # Specify the refinment algorithm to use
        self.refinement_algorithm = refinement_algorithm
        print('Refinement Algorithm:', self.refinement_algorithm)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def round_up_to_even(f):
    return round(f / 2.0) * 2

def cal_sq_coord(xc, yc, length, imw, imh):
    x1 = max(0, xc - length / 2)
    y1 = max(0, yc - length / 2)
    x2 = min(imw, xc + length / 2)
    y2 = min(imh, yc + length / 2)
    return x1, x2, y1, y2

def adjust_boxes(boxes, adj, imw, imh, square=False):
    xmin, ymin, xmax, ymax = boxes.tolist()
    w = abs(xmax - xmin)
    h = abs(ymax - ymin)
    adj_w = adj * w
    adj_h = adj * h
    x1 = (xmin - adj_w) if (xmin - adj_w) > 0 else 0
    y1 = (ymin - adj_h) if (ymin - adj_h) > 0 else 0
    x2 = (xmax + adj_w) if (xmax + adj_w) < imw else imw
    y2 = (ymax + adj_h) if (ymax + adj_h) < imh else imh

    xc = int(round(x1 + (x2 - x1) / 2))
    yc = int(round(y1 + (y2 - y1) / 2))

    if square:
        correction_factor = (x2 - x1) / (y2 - y1)
        if correction_factor > 1:
            x_length = int(round_up_to_even(x2 - x1))
            x1, x2, y1, y2 = cal_sq_coord(xc, yc, x_length, imw, imh)
        elif correction_factor < 1:
            y_length = int(round_up_to_even(y2 - y1))
            x1, x2, y1, y2 = cal_sq_coord(xc, yc, y_length, imw, imh)
        box = np.array([x1, y1, x2, y2], dtype=int)
    else:
        box = np.array([x1, y1, x2, y2], dtype=int)
    return box

def cal_padding(box):
    xmin, ymin, xmax, ymax = box
    xl = xmax - xmin
    yl = ymax - ymin
    xpad = ypad = 0
    if xl > yl:
        ypad = (xl - yl) / 2
    elif yl > xl:
        xpad = (yl - xl) / 2
    return xpad, ypad

def get_max_preds(batch_heatmaps):
    assert isinstance(batch_heatmaps, torch.Tensor)
    B, K, H, W = batch_heatmaps.shape
    heatmaps_reshaped = batch_heatmaps.view(B, K, -1)
    maxvals, idx = torch.max(heatmaps_reshaped, dim=2)
    maxvals = maxvals.unsqueeze(-1)
    idx = idx.float()
    preds = torch.zeros((B, K, 2), device=batch_heatmaps.device)
    preds[..., 0] = idx % W
    preds[..., 1] = torch.floor(idx / W)
    pred_mask = torch.gt(maxvals, 0.0).repeat(1, 1, 2).float().to(batch_heatmaps.device)
    preds *= pred_mask
    return preds, maxvals

def refine_keypoints(heatmaps, keypoints):
    B, K, H, W = heatmaps.shape
    refined = keypoints.clone().cpu().numpy()
    heatmaps = heatmaps.cpu().numpy()
    for b in range(B):
        for j in range(K):
            x, y = refined[b, j].astype(int)
            if 1 <= x < W-1 and 1 <= y < H-1:
                patch = heatmaps[b, j, y-1:y+2, x-1:x+2]
                if patch.sum() > 1e-6:
                    dx = 0.5 * (patch[1, 2] - patch[1, 0])
                    dy = 0.5 * (patch[2, 1] - patch[0, 1])
                    dxx = patch[1, 2] + patch[1, 0] - 2 * patch[1, 1]
                    dyy = patch[2, 1] + patch[0, 1] - 2 * patch[1, 1]
                    dxy = 0.25 * (patch[2, 2] + patch[0, 0] - patch[0, 2] - patch[2, 0])
                    hessian = np.array([[dxx, dxy], [dxy, dyy]])
                    grad = np.array([-dx, -dy])
                    if np.linalg.det(hessian) > 1e-6:
                        shift = np.linalg.inv(hessian).dot(grad)
                        refined[b, j] += shift
    return torch.from_numpy(refined).float()

def map_kps_to_full_image(kps, scores, bbox, output_size, kp_min_count=5, det_thresh=0.2, full_im_size=224):
    xpad, ypad = cal_padding(bbox)
    count = 0
    max_count = len([s for s in scores if 0.0 < s < 1.0])
    kp_min_count = min(kp_min_count, max_count)
    thresh = det_thresh
    points = np.zeros((kps.shape[0], 3), dtype=np.float32)

    while count < kp_min_count and thresh > 0.0:
        thresh *= 0.9
        count = 0
        for i in range(len(kps)):
            points[i, 0] = (kps[i, 0] * full_im_size / output_size) - xpad
            points[i, 1] = (kps[i, 1] * full_im_size / output_size) - ypad
            points[i, 2] = scores[i]
            if thresh < scores[i] < 1.0:
                count += 1
    return points, thresh

def convert_to_image_coords(points, thresh, bbox, num_kp):
    xmin, ymin, _, _ = bbox
    pts = np.zeros((num_kp, 2), dtype=np.float32)
    vis = []
    for i in range(num_kp):
        if thresh < points[i, 2] < 1.0:
            pts[i, 0] = xmin + points[i, 0]
            pts[i, 1] = ymin + points[i, 1]
            vis.append(i)
        else:
            pts[i, 0] = 0.0
            pts[i, 1] = 0.0
    return pts, vis

# ========== MAIN WRAPPER FUNCTION ==========
def extract_refined_keypoints_from_heatmaps(hm, bbox, output_size=56, full_im_size=224, kp_min_count=5, kp_det_thresh=0.2):
    """
    Main function: from heatmaps + bbox → get keypoints in full image coords
    Args:
        hm: (B, K, H, W) heatmaps
        bbox: (B, 4) XYXY crop boxes
    Returns:
        - keypoints: (B, K, 2)
        - visibility_mask: list of indices per frame
    """
    B, K, Hm, Wm = hm.shape
    preds, scores = get_max_preds(hm)
    refined = refine_keypoints(hm, preds)

    kps_all = []
    vis_all = []
    for b in range(B):
        kp_adj, thresh = map_kps_to_full_image(refined[b], scores[b], bbox[b], output_size, kp_min_count, kp_det_thresh, full_im_size)
        kps_img, vis = convert_to_image_coords(kp_adj, thresh, bbox[b], K)
        kps_all.append(kps_img)
        vis_all.append(vis)

    return np.stack(kps_all), vis_all
    
# =============================
# Function for crop-aware translation scaling
# =============================
def scale_translation_to_full_image(translations, crop_xywh, intrinsics, img_size=(640, 480)):
    """
    Convert predicted translations from crop-space to full-frame coordinates.
    translations: (B*T, 3)
    crop_xywh:    (B*T, 4) normalized [x,y,w,h] relative to full image
    intrinsics:   (B*T, 3, 3) full-frame camera intrinsics
    """
    W, H = img_size
    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 1, 1]
    cx, cy = intrinsics[:, 0, 2], intrinsics[:, 1, 2]

    x_c = crop_xywh[:, 0] * W
    y_c = crop_xywh[:, 1] * H
    w_c = crop_xywh[:, 2] * W
    h_c = crop_xywh[:, 3] * H

    center_x = x_c + 0.5 * w_c
    center_y = y_c + 0.5 * h_c

    t_x = translations[:, 0] + translations[:, 2] * (center_x - cx) / fx
    t_y = translations[:, 1] + translations[:, 2] * (center_y - cy) / fy
    t_z = translations[:, 2]

    return torch.stack([t_x, t_y, t_z], dim=-1)


def quat_inverse(q: torch.Tensor) -> torch.Tensor:
    """Inverse of quaternion (B,T,4)."""
    q_inv = q.clone()
    q_inv[..., 1:] *= -1
    return q_inv / (q.norm(dim=-1, keepdim=True) + 1e-8)

def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion multiplication."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack((w,x,y,z), dim=-1)

def quat_relative(q2: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    """Relative quaternion q_rel = q1^{-1} * q2.
       q1, q2: (...,4) unit quats
    """
    return quat_mul(quat_inverse(q1), q2)


# ----------- Temporal consistency loss to align T features within a video -----
# ------------------------------------------------------------------------------
def temporal_heatmap_loss(hm_logits):
    # hm_logits: (B, T, K, H, W)
    return F.mse_loss(
        hm_logits[:, 1:], 
        hm_logits[:, :-1]
    )
def temporal_consistency_loss(kps_pred, weight=1.0):
    """
    kps_pred: (B, T, N, 2) predicted keypoints (normalized coords)
    weight: scaling factor for loss
    """
    diffs = kps_pred[:, 1:] - kps_pred[:, :-1]        # (B, T-1, N, 2)
    loss = torch.mean(torch.norm(diffs, dim=-1))      # L2 distance across frames
    return weight * loss

# ----- Heatmap Scheduler ------------------------------------------------------
def cosine_schedule(start, end, start_epoch, end_epoch, cur_epoch):
    """Cosine interpolation between start and end weights."""
    if cur_epoch < start_epoch:
        return start
    if cur_epoch > end_epoch:
        return end
    progress = (cur_epoch - start_epoch) / float(end_epoch - start_epoch)
    weight = end + 0.5 * (start - end) * (1 + math.cos(math.pi * progress))
    return weight

# ---- tau scheduling ----------------------------------------------------------
# --- tau schedule helper ---
def get_tau_for_epoch(cfg: Dict[str, Any], epoch: int) -> float:
    loss_cfg = cfg.get("loss", {})
    base_tau = float(loss_cfg.get("tau", 1.0))
    sched = loss_cfg.get("tau_schedule", None)
    if not sched:
        return base_tau

    typ = str(sched.get("type", "cosine")).lower()
    start = float(sched.get("start", base_tau))
    end = float(sched.get("end",  start))
    e0 = int(sched.get("start_epoch", 1))
    e1 = int(sched.get("end_epoch",   e0))

    if epoch <= e0: return start
    if epoch >= e1: return end

    # progress in [0,1]
    t = (epoch - e0) / max(1, (e1 - e0))
    if typ == "linear":
        val = start + t * (end - start)
    else:  # cosine (default)
        import math
        val = end + (start - end) * 0.5 * (1 + math.cos(math.pi * t))
    return float(val)

# heatmap_power scheduling
def get_heatmap_power_for_epoch(cfg, epoch: int) -> float:
    loss_cfg = cfg.get("loss", {})
    base_pow = float(loss_cfg.get("heatmap_target_power", 1.0))
    sched = loss_cfg.get("heatmap_power_schedule", None)
    if not sched:
        return base_pow

    typ = str(sched.get("type", "linear")).lower()
    start = float(sched.get("start", base_pow))
    end   = float(sched.get("end", base_pow))
    e0    = int(sched.get("start_epoch", 1))
    e1    = int(sched.get("end_epoch", e0))

    if epoch <= e0: return start
    if epoch >= e1: return end

    t = (epoch - e0) / max(1, (e1 - e0))
    if typ == "linear":
        return start + t * (end - start)
    else:  # cosine
        import math
        return end + (start - end) * 0.5 * (1 + math.cos(math.pi * t))
# ----------------------------------------------------------------------------- 

def project_keypoints(points_3d: torch.Tensor,
                      q: torch.Tensor,
                      t: torch.Tensor,
                      cam_K: torch.Tensor) -> torch.Tensor:
    """
    Project 3D object keypoints into 2D image coordinates.

    Args:
        points_3d: (K,3) tensor of object keypoints in object frame
        q: (B,T,4) predicted quaternions (wxyz)
        t: (B,T,3) predicted translations in camera frame
        cam_K: (3,3) or (B,3,3) intrinsics matrix

    Returns:
        pts_2d: (B,T,K,2) keypoints in pixel coordinates
    """
    B, T = q.shape[:2]
    K = points_3d.shape[0]

    # Normalize quaternion and convert to rotation matrix
    q = F.normalize(q, dim=-1)
    R = quat_to_rotmat(q)  # (B,T,3,3)

    # Apply pose to transform object points into camera coords
    pts_obj = points_3d.to(q.device)  # (K,3)
    pts_obj = pts_obj.unsqueeze(0).unsqueeze(0).expand(B, T, K, 3)  # (B,T,K,3)
    pts_cam = torch.einsum('btij,btkj->btki', R, pts_obj) + t.unsqueeze(2)  # (B,T,K,3)

    # Perspective projection: (x/z, y/z)
    x = pts_cam[..., 0] / pts_cam[..., 2]
    y = pts_cam[..., 1] / pts_cam[..., 2]

    # Intrinsics
    if cam_K.ndim == 2:  # (3,3)
        fx, fy = cam_K[0,0], cam_K[1,1]
        cx, cy = cam_K[0,2], cam_K[1,2]
    else:  # (B,3,3)
        fx, fy = cam_K[:,0,0], cam_K[:,1,1]
        cx, cy = cam_K[:,0,2], cam_K[:,1,2]

    # Broadcast intrinsics
    pts_2d_x = fx * x + cx
    pts_2d_y = fy * y + cy

    pts_2d = torch.stack([pts_2d_x, pts_2d_y], dim=-1)  # (B,T,K,2)
    return pts_2d


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Quaternion (w,x,y,z) → rotation matrix."""
    w, x, y, z = q.unbind(-1)
    B, T = q.shape[:2]

    R = torch.empty((B,T,3,3), device=q.device, dtype=q.dtype)
    R[...,0,0] = 1 - 2*y*y - 2*z*z
    R[...,0,1] = 2*x*y - 2*z*w
    R[...,0,2] = 2*x*z + 2*y*w
    R[...,1,0] = 2*x*y + 2*z*w
    R[...,1,1] = 1 - 2*x*x - 2*z*z
    R[...,1,2] = 2*y*z - 2*x*w
    R[...,2,0] = 2*x*z - 2*y*w
    R[...,2,1] = 2*y*z + 2*x*w
    R[...,2,2] = 1 - 2*x*x - 2*y*y
    return R

def hm_to_crop_px(x_hm: torch.Tensor, Wm: int, inp: int) -> torch.Tensor:
    # heatmap pixel coord [0..Wm-1] -> crop-plane pixel [0..inp-1]
    return x_hm * ((float(inp) - 1.0) / float(max(Wm - 1, 1)))

def crop_to_hm_px(x_crop: torch.Tensor, Wm: int, inp: int) -> torch.Tensor:
    # crop-plane pixel [0..inp-1] -> heatmap pixel [0..Wm-1]
    return x_crop * ((float(Wm) - 1.0) / float(max(inp - 1, 1)))

def freeze_all_but_dualpath_and_heads(
    model: nn.Module, 
    unfreeze_last_n_blocks: int = 2
):
    """
    Freezes all parameters in the model except:
    - Dual-path adapters (spatial/temporal attn/mlp adapters)
    - Downstream heads (pose, heatmap, residuals, finerCNN, etc.)
    - Patch embedding projection
    - Final norm
    - Last N transformer blocks
    """
    total, trainable = 0, 0

    # Keywords to allow training
    adapter_keys = (
        "temporal_attn_adapter", "temporal_mlp_adapter", 
        "spatial_attn_adapter", "spatial_mlp_adapter", #"adapter"
    )
    head_keys = (
        "pose_head", "hm_head", "kp_res_head", "finer_cnn",
        "final_projection", "fuse_mlp", "fuse_ln"
    )
    # extra_keys = ("patch_embed.proj", "norm")
    extra_keys = ()


    # Find how many blocks exist
    num_blocks = len(getattr(model.backbone.model, "blocks", []))
    last_block_idxs = set(range(max(0, num_blocks - unfreeze_last_n_blocks), num_blocks))

    for name, param in model.named_parameters():
        total += param.numel()

        # Always allow adapters & heads
        if any(k in name for k in adapter_keys + head_keys):
            param.requires_grad = True
            trainable += param.numel()
            continue

        # Allow patch embed + final norm
        if any(k in name for k in extra_keys):
            param.requires_grad = True
            trainable += param.numel()
            continue

        # Allow last N blocks
        if "blocks." in name:
            block_id = int(name.split("blocks.")[1].split(".")[0])
            if block_id in last_block_idxs:
                param.requires_grad = True
                trainable += param.numel()
                continue

        # Otherwise freeze
        param.requires_grad = False

    print(f"[freeze] Trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M "
          f"({100 * trainable / total:.2f}%) | "
          f"last {unfreeze_last_n_blocks} blocks + Adapters + patch_embed + norm unfrozen + heads.")
    return {
        "Total Params": total,
        "Trainable Params": trainable,
        "Trainable %": round(100 * trainable / total, 2)
    }

def unnormalize_keypoints(normalized_keypoints, image_size):
    """
    Unnormalize keypoints from [-1, 1] range back to pixel coordinates.
    
    Args:
        normalized_keypoints (torch.Tensor or np.ndarray): Normalized keypoints (batch_size, num_keypoints, 2) in range [-1, 1].
        image_size (tuple): The (width, height) of the original image.
    
    Returns:
        torch.Tensor or np.ndarray: Un-normalized keypoints in pixel coordinates.
    """
    width, height = image_size
    
    # Unnormalize keypoints from [-1, 1] back to pixel coordinates
    keypoints_x = ((normalized_keypoints[:, 0] + 1) / 2) * width
    keypoints_y = ((normalized_keypoints[:, 1] + 1) / 2) * height
    
    unnormalized_keypoints = torch.stack([keypoints_x, keypoints_y], dim=-1)
    
    return unnormalized_keypoints


def load_camera_intrinsics(json_path: str) -> dict:
    with open(json_path, 'r') as f:
        cam_data = json.load(f)
    K = np.array(cam_data["cameraMatrix"], dtype=np.float32)  # (3,3)
    dist = np.array(cam_data["distCoeffs"], dtype=np.float32)  # (5,) or (1,5)
    return {"K": K, "dist": dist}


def plot_keypoints2(frames, ground_truth_keypoints, predicted_keypoints, phase, epoch, figsize=(8, 8), image_name="image.png"):
    """
    Plots a sequence of frames with ground truth and predicted keypoints.
    Args:
        frames (torch.Tensor): (T, 3, H, W)
        ground_truth_keypoints (torch.Tensor): (T, K, 2)
        predicted_keypoints (torch.Tensor): (T, K, 2)
    """
    assert frames.dim() == 4 and frames.shape[1] == 3, \
        f"Expected (T, 3, H, W), got {frames.shape}"

    save_dir = f'keypoints_images_{phase}_{epoch}'
    os.makedirs(save_dir, exist_ok=True)

    for t in range(frames.shape[0]):
        image_tensor = frames[t]  # (3, H, W)
        file_path = os.path.join(save_dir, f"./keypoints_time{t}.png")

        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

        plt.figure(figsize=figsize)
        plt.imshow(image_np)

        # Plot GT keypoints
        if ground_truth_keypoints is not None:
            gp = unnormalize_keypoints(
                ground_truth_keypoints[t], (224,224)
            ).detach().cpu().numpy()
            plt.scatter(gp[:, 0], gp[:, 1], c='green', s=60, marker='o', label='Ground Truth')

        if predicted_keypoints is not None:
            pp = unnormalize_keypoints(
                predicted_keypoints[t], (224,224)
            ).detach().cpu().numpy()
            plt.scatter(pp[:, 0], pp[:, 1], c='red', s=60, marker='x', label='Predicted')

        plt.legend()
        plt.axis('off')
        plt.savefig(file_path)
        plt.close()


def plot_heatmaps(heatmaps,epoch,phase):
    save_dir = 'heatmap_images'+'_'+phase+'_'+str(epoch)
    os.makedirs(save_dir, exist_ok=True)
    B,T,N,H,W = heatmaps.size()
    heatmaps = heatmaps.cpu().detach()
    B = 1
    # Loop through the batch, time steps, and classes, and save each heatmap
    for b in range(B):
        for t in range(T):
            for n in range(N):
                heatmap = heatmaps[b, t, n, :, :]

                # Define the file path for saving
                file_name = f"heatmap_batch{b}_time{t}_class{n}.png"
                file_path = os.path.join(save_dir, file_name)

                # Plot and save the heatmap
                plt.imshow(heatmap, cmap='hot', interpolation='nearest')
                plt.colorbar()
                plt.title(f"Heatmap for Batch {b}, Time Step {t}, Class {n}")
                plt.axis('off')  # Hide axes

                # Save the image
                plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
                plt.close()  # Close the plot to avoid displaying it
                #print(f"Saved heatmap to {file_path}")


def _apply_temporal_stride(frames, kps_gt, gt_q, gt_r, vis, hm_gt, stride: int):
    if stride <= 1:
        return frames, kps_gt, gt_q, gt_r, vis, hm_gt
    sl = slice(None, None, stride)  # ::stride
    # each is (B, T, ...)
    return (
        frames[:, sl],
        kps_gt[:, sl],
        gt_q[:, sl],
        gt_r[:, sl],
        vis[:, sl],
        hm_gt[:, sl],
    )

from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

def build_warmup_cosine_scheduler(optimizer, epochs, steps_per_epoch,
                                   warmup_epochs=0, warmup_steps=None,
                                   lr_min=1e-6):
    """
    PyTorch native warmup + cosine decay scheduler.
    Returns per-step SequentialLR:
        - Linear warmup: 0 → base_lr
        - Cosine decay: base_lr → lr_min
    """
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_iters = warmup_steps if warmup_steps is not None else warmup_epochs * steps_per_epoch
    warmup_iters = max(0, warmup_iters)
    decay_iters = max(1, total_steps - warmup_iters)

    # Linear warmup from near-zero to base_lr
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_iters
    )

    # Cosine annealing from base_lr to lr_min
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=decay_iters,
        eta_min=lr_min
    )

    # Compose into sequential scheduler
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_iters]
    )
    return scheduler

@torch.no_grad()
def trans_err_l2_all(pred_t: torch.Tensor, gt_t: torch.Tensor):
    """Return per-frame translation L2 error in meters: (B,T)."""
    return torch.norm(pred_t - gt_t, dim=-1)  # (B,T)

@torch.no_grad()
def nanmean(x: torch.Tensor) -> torch.Tensor:
    return torch.nanmean(x)

@torch.no_grad()
def nanmedian(x: torch.Tensor) -> torch.Tensor:
    return torch.nanmedian(x)

# def _forward_safely(model, **kwargs):
#     """
#     Use the wrapped module's signature (if DDP) so we keep only supported kwargs.
#     Always pass `video` positionally to satisfy DDP.
#     """
#     if "video" not in kwargs:
#         raise KeyError("`video` must be provided to model.forward(...)")

#     # unwrap if DDP
#     real_model = model.module if isinstance(model, DDP) else model

#     # filter kwargs by the real model's forward signature
#     sig_params = inspect.signature(real_model.forward).parameters
#     filtered = {k: v for k, v in kwargs.items() if k in sig_params}

#     # pop video and pass it positionally
#     video = filtered.pop("video")
#     return model(video, **filtered)

# def _extract_frame_ids_if_present(batch_tail: Any) -> Optional[List[List[Any]]]:
#     """
#     If your dataset returns frame_ids as the last element (List[List[anything]]),
#     detect and return them; otherwise None.
#     """
#     if isinstance(batch_tail, list) and batch_tail and isinstance(batch_tail[0], list):
#         return batch_tail
#     return None

@torch.no_grad()
def save_kp_overlay(frames, kps_gt_norm, hm_logits, vis, cfg, save_path):
    """
    Overlays GT keypoints (green) and predicted softargmax (magenta)
    on the first sample / first timestep image and saves PNG.

    frames:   (B,T,3,H,W) in [0,1] (after transforms)
    kps_gt_norm: (B,T,K,2) in [-1,1] normalized image coords
    hm_logits:   (B,T,K,Hm,Wm) raw logits
    vis:         (B,T,K) in {0,1}
    cfg:         full cfg (for input size)
    save_path:   str, path to write a .png
    """
    # pick first sample/time
    f = frames[0, 0].detach().cpu().clamp(0, 1)  # (3,H,W)
    B, T, K, Hm, Wm = hm_logits.shape
    H, W = int(cfg["data"]["input_size"]), int(cfg["data"]["input_size"])

    # ---- pred: softargmax in heatmap px -> upsample to image px ----
    from losses import softargmax_2d_from_logits
    hm_ = hm_logits[0:1, 0:1]  # (1,1,K,Hm,Wm)
    pred_px_hm = softargmax_2d_from_logits(hm_)  # (1,1,K,2) in [0..Wm-1],[0..Hm-1]
    pred_px_hm = pred_px_hm[0, 0]  # (K,2)
    sx, sy = float(W) / float(Wm), float(H) / float(Hm)
    pred_px_img = pred_px_hm.clone()
    pred_px_img[:, 0] = pred_px_hm[:, 0] * sx
    pred_px_img[:, 1] = pred_px_hm[:, 1] * sy

    # ---- GT: norm [-1,1] -> image px ----
    gx = ((kps_gt_norm[0, 0, :, 0] + 1.0) * 0.5) * (W - 1.0)
    gy = ((kps_gt_norm[0, 0, :, 1] + 1.0) * 0.5) * (H - 1.0)
    gt_px_img = torch.stack([gx, gy], dim=-1)  # (K,2)

    # ---- visibility mask (optional dimming of invisible KPs) ----
    vis_ = None
    if vis is not None and vis.numel() > 0:
        vis_ = (vis[0, 0].detach().cpu() > 0).float()  # (K,)

    # ---- convert frame to numpy HxWx3 uint8
    import numpy as np
    img = (f.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8).copy()

    # Try OpenCV for drawing, otherwise fall back to PIL
    try:
        import cv2
        # colors: BGR
        col_gt = (0, 255, 0)       # green
        col_pred = (255, 0, 255)   # magenta
        r = max(2, int(0.004 * max(H, W)))  # radius ~0.4% of size
        th = max(1, int(0.002 * max(H, W)))

        for i in range(gt_px_img.shape[0]):
            px_gt  = (int(gt_px_img[i, 0].item()),  int(gt_px_img[i, 1].item()))
            px_prd = (int(pred_px_img[i, 0].item()), int(pred_px_img[i, 1].item()))
            if vis_ is None or vis_[i] > 0.5:
                cv2.circle(img, px_gt,  r, col_gt,  -1, lineType=cv2.LINE_AA)
                cv2.circle(img, px_prd, r, col_pred, th, lineType=cv2.LINE_AA)
            else:
                # draw faint if not visible
                cv2.circle(img, px_gt,  r, (0,128,0), -1, lineType=cv2.LINE_AA)
                cv2.circle(img, px_prd, r, (128,0,128), th, lineType=cv2.LINE_AA)

        cv2.imwrite(save_path, img)
    except Exception:
        # PIL fallback
        from PIL import Image, ImageDraw
        pimg = Image.fromarray(img)
        draw = ImageDraw.Draw(pimg)
        r = max(2, int(0.004 * max(H, W)))

        def _disk(xy, color, fill=True):
            x, y = xy
            bbox = [x - r, y - r, x + r, y + r]
            if fill:
                draw.ellipse(bbox, fill=color)
            else:
                draw.ellipse(bbox, outline=color, width=max(1, r//2))

        for i in range(gt_px_img.shape[0]):
            px_gt  = (float(gt_px_img[i, 0].item()),  float(gt_px_img[i, 1].item()))
            px_prd = (float(pred_px_img[i, 0].item()), float(pred_px_img[i, 1].item()))
            if vis_ is None or vis_[i] > 0.5:
                _disk(px_gt,  (0,255,0),  fill=True)      # GT solid green
                _disk(px_prd, (255,0,255), fill=False)     # Pred magenta ring
            else:
                _disk(px_gt,  (0,128,0),  fill=True)
                _disk(px_prd, (128,0,128), fill=False)

        pimg.save(save_path)

    print(f"[SANITY] saved keypoint overlay -> {save_path}", flush=True)


def debug_check_heatmap_alignment(frames, hm_target, kp_target_norm, epoch=0, step=0, save_dir="./debug"):
    """
    Quick sanity check: visualize GT frame, overlay GT heatmap max, and kp_target_norm.

    Args:
      frames: (B,T,C,H,W) tensor in [0,1].
      hm_target: (B,T,K,Hm,Wm) GT heatmaps (sum=1 per heatmap).
      kp_target_norm: (B,T,K,2) GT normalized coords in [-1,1].
    """
    os.makedirs(save_dir, exist_ok=True)
    B, T, C, H, W = frames.shape
    _, _, K, Hm, Wm = hm_target.shape

    for b in range(B):
        # pick first frame
        frame = frames[b, 0].detach().cpu()  # (C,H,W)
        img = frame.permute(1,2,0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        # sum heatmaps over K
        hm = hm_target[b, 0].sum(0).detach().cpu().numpy()  # (Hm,Wm)
        hm_up = cv2.resize(hm, (W, H), interpolation=cv2.INTER_CUBIC)

        # denormalize keypoints
        gx = ((kp_target_norm[b,0,:,0].detach().cpu().numpy() + 1.0) * 0.5) * (Wm - 1)
        gy = ((kp_target_norm[b,0,:,1].detach().cpu().numpy() + 1.0) * 0.5) * (Hm - 1)
        gx = gx * (W / Wm)
        gy = gy * (H / Hm)

        # plot
        plt.figure(figsize=(5,5))
        plt.imshow(img)
        plt.imshow(hm_up, alpha=0.4, cmap="jet")
        plt.scatter(gx, gy, c="lime", s=40, marker="x")
        plt.title(f"GT Heatmap + KPs (epoch {epoch}, step {step}, batch {b})")

        path = os.path.join(save_dir, f"debug_hm_e{epoch}_s{step}_b{b}.png")
        plt.savefig(path); plt.close()
        print(f"[DEBUG] saved {path}", flush=True)

def debug_check_pred_alignment(frames, out,
                               epoch=0, step=0, batch_idx=0, save_dir="./debug_pred"):
    """
    Visualize predicted frame with predicted heatmaps + keypoints from model out.
    Args:
      frames: (B,T,C,H,W) or (T,C,H,W) tensor in [0,1].
      out: dict from model forward (must contain 'heatmaps' and 'keypoints').
    """
    heatmaps = out["heatmaps"]     # (B,T,K,Hm,Wm)
    if heatmaps is None:
        return
    os.makedirs(save_dir, exist_ok=True)

    if frames.ndim == 4:  # (T,C,H,W) without batch
        frames = frames.unsqueeze(0)

    B, T, C, H, W = frames.shape
    # heatmaps = out["heatmaps"]     # (B,T,K,Hm,Wm)
    keypoints = out["keypoints"]   # (B,T,K,2)

    _, _, K, Hm, Wm = heatmaps.shape

    # ---- pick first sample and frame ----
    frame = frames[0, 0].detach().cpu()
    img = frame.permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)

    hm_pred = torch.softmax(heatmaps[0, 0], dim=-1).detach().cpu().numpy()  # (K,Hm,Wm)
    hm_sum = hm_pred.sum(0)  # aggregate across K

    # upscale to input resolution
    hm_up = cv2.resize(hm_sum, (W, H), interpolation=cv2.INTER_CUBIC)

    # convert keypoints (predicted are already normalized to image grid of heatmap)
    kps = keypoints[0, 0].detach().cpu().numpy()  # (K,2)
    gx = (kps[:, 0] / (Wm - 1)) * W
    gy = (kps[:, 1] / (Hm - 1)) * H

    # ---- plot ----
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.imshow(hm_up, alpha=0.4, cmap="jet")
    plt.scatter(gx, gy, c="yellow", s=40, marker="x")
    plt.title(f"Pred Heatmap + KPs (epoch {epoch}, step {step}, batch {batch_idx})")
    path = os.path.join(save_dir, f"pred_hm_epoch{epoch}_s{step}_b{batch_idx}.png")
    plt.savefig(path); plt.close()
    print(f"[DEBUG] saved {path}")


def collate_temporal_chunks(batch, T_chunk: int):
    """
    Each batch item:
      frames:     (T,C,H,W)
      kps:        (T,K,2)
      gt_q:       (T,4)
      gt_r:       (T,3)
      vis:        (T,K)
      hm:         (T,K,Hm,Wm)
      delta_Rt:   (T-1,6)
      frame_ids
      crop_xywh:  (T,4)

    Returns:
      All tensors chunked to (B_chunks, T_chunk, ...)
    """
    tensor_out = [[] for _ in range(7)]  # frames..hm
    frame_ids_out = []
    crop_out = []

    for sample in batch:
        frames, kps, gt_q, gt_r, vis, hm, delta_Rt, frame_ids, crop_xywh = sample

        T = frames.shape[0]
        T_eff = (T // T_chunk) * T_chunk
        if T_eff == 0:
            continue

        frames    = frames[:T_eff]
        kps       = kps[:T_eff]
        gt_q      = gt_q[:T_eff]
        gt_r      = gt_r[:T_eff]
        vis       = vis[:T_eff]
        hm        = hm[:T_eff]
        crop_xywh = crop_xywh[:T_eff]

        n_chunks = T_eff // T_chunk

        frames    = frames.view(n_chunks, T_chunk, *frames.shape[1:])
        kps       = kps.view(n_chunks, T_chunk, *kps.shape[1:])
        gt_q      = gt_q.view(n_chunks, T_chunk, *gt_q.shape[1:])
        gt_r      = gt_r.view(n_chunks, T_chunk, *gt_r.shape[1:])
        vis       = vis.view(n_chunks, T_chunk, *vis.shape[1:])
        hm        = hm.view(n_chunks, T_chunk, *hm.shape[1:])
        crop_xywh = crop_xywh.view(n_chunks, T_chunk, 4)

        for i in range(n_chunks):
            tensor_out[0].append(frames[i])
            tensor_out[1].append(kps[i])
            tensor_out[2].append(gt_q[i])
            tensor_out[3].append(gt_r[i])
            tensor_out[4].append(vis[i])
            tensor_out[5].append(hm[i])
            tensor_out[6].append(delta_Rt)   # untouched
            frame_ids_out.append(frame_ids)
            crop_out.append(crop_xywh[i])

    tensors = [torch.stack(x) for x in tensor_out]
    crop_xywh = torch.stack(crop_out)

    return (*tensors, frame_ids_out, crop_xywh)

def collate_variable_length(batch):
    """
    Pads variable-length sequences and returns a valid-frame mask.
    Mask MUST be used for per-frame metrics (PCK, errors).
    """

    frames, kps, gt_q, gt_r, vis, hm, dRt, fids, crop_xywh = zip(*batch)

    lengths = torch.tensor([x.shape[0] for x in frames], dtype=torch.long)
    T_max = lengths.max().item()

    def pad_T(x, pad_value=0.0):
        T = x.shape[0]
        if T == T_max:
            return x
        pad_shape = (T_max - T, *x.shape[1:])
        pad = torch.full(pad_shape, pad_value, dtype=x.dtype)
        return torch.cat([x, pad], dim=0)

    frames    = torch.stack([pad_T(x) for x in frames])
    kps       = torch.stack([pad_T(x) for x in kps])
    gt_q      = torch.stack([pad_T(x) for x in gt_q])
    gt_r      = torch.stack([pad_T(x) for x in gt_r])
    vis       = torch.stack([pad_T(x) for x in vis])
    hm        = torch.stack([pad_T(x) for x in hm])
    crop_xywh = torch.stack([pad_T(x) for x in crop_xywh])

    # ΔRt has T-1
    dRt_max = max(x.shape[0] for x in dRt)
    def pad_dRt(x):
        if x.shape[0] == dRt_max:
            return x
        pad = torch.zeros((dRt_max - x.shape[0], x.shape[1]), dtype=x.dtype)
        return torch.cat([x, pad], dim=0)

    dRt = torch.stack([pad_dRt(x) for x in dRt])

    # -------------------------
    # VALID FRAME MASK
    # -------------------------
    valid_mask = torch.arange(T_max)[None, :] < lengths[:, None]  # (B,T)

    return (
        frames, kps, gt_q, gt_r, vis, hm,
        dRt, fids, crop_xywh, valid_mask
    )