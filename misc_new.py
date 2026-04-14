# misc.py (FINAL — auto-detect VBN vs SPADES camera/points format)
import os
import json
import wandb
import torch
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
import cv2
from scipy.spatial.transform import Rotation as R


# -------------------------
# Camera JSON loader (auto-detect format)
# -------------------------

def _load_camera_matrix_and_dist(camera_path: str):
    """
    Auto-detect camera JSON format:
      - SPADES format: has "cameraMatrix" (3x3 list) and "distCoeffs" (list)
      - VBN format: has "focalLengthX", "focalLengthY", "principalPointX", "principalPointY"
    
    VBN imageCoordinateOrigin convention:
      - "corner": (0,0) is at the CORNER of the top-left pixel
                  → OpenCV uses center-of-pixel, so subtract 0.5
      - "center": (0,0) is at the CENTER of the top-left pixel
                  → Already matches OpenCV convention, use directly
    
    NOTE: "center"/"corner" refers to sub-pixel origin (corner vs center
    of the top-left pixel), NOT "relative to image center". The principal
    point values (~503, ~499) are always near the image center for a
    1024×1024 sensor.
    
    Returns:
      CMat:   np.ndarray (3,3) float64 — in OpenCV convention (center-of-pixel)
      dist:   np.ndarray (N,1) float64
      img_wh: (W, H) or None
    """
    with open(camera_path, "r") as f:
        dt = json.load(f)

    # --- SPADES format ---
    if "cameraMatrix" in dt:
        CMat = np.array(dt["cameraMatrix"], dtype=np.float64)
        dist = dt.get("distCoeffs", [0, 0, 0, 0, 0])
        dist = np.array(dist, dtype=np.float64).reshape(-1, 1)
        W = dt.get("Nu", None)
        H = dt.get("Nv", None)
        img_wh = (int(W), int(H)) if (W is not None and H is not None) else None
        return CMat, dist, img_wh

    # --- VBN format ---
    if "focalLengthX" in dt:
        fx = float(dt["focalLengthX"])
        fy = float(dt["focalLengthY"])
        cx = float(dt["principalPointX"])
        cy = float(dt["principalPointY"])
        W = int(dt.get("imageWidth", 1024))
        H = int(dt.get("imageHeight", 1024))

        origin = dt.get("imageCoordinateOrigin", "corner").lower()
        
        if origin == "corner":
            # "corner": (0,0) is at top-left corner of pixel
            # OpenCV convention is center-of-pixel, so shift by -0.5
            cx = cx - 0.5
            cy = cy - 0.5
        # "center": already in center-of-pixel convention (matches OpenCV)
        # → use cx, cy directly

        CMat = np.array([
            [fx,  0.0, cx],
            [0.0, fy,  cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Distortion
        k1 = float(dt.get("distortionParamK1", 0.0))
        k2 = float(dt.get("distortionParamK2", 0.0))
        p1 = float(dt.get("distortionParamP1", 0.0))
        p2 = float(dt.get("distortionParamP2", 0.0))
        k3 = float(dt.get("distortionParamK3", 0.0))
        dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64).reshape(-1, 1)

        img_wh = (W, H)
        
        print(f"[Camera] origin='{origin}', cx={cx:.4f}, cy={cy:.4f}, "
              f"fx={fx:.2f}, fy={fy:.2f}, img={W}x{H}")
        
        return CMat, dist, img_wh

    raise ValueError(
        f"Cannot parse camera JSON: {camera_path}\n"
        f"Expected either 'cameraMatrix' (SPADES) or 'focalLengthX' (VBN) keys."
    )

def _load_camera_matrix_and_dist_spades(camera_path: str):
    """
    Auto-detect camera JSON format:
      - SPADES format: has "cameraMatrix" (3x3 list) and "distCoeffs" (list)
      - VBN format: has "focalLengthX", "focalLengthY", "principalPointX", "principalPointY"
                    and optionally "imageCoordinateOrigin": "center"
    Returns:
      CMat:  np.ndarray (3,3) float64
      dist:  np.ndarray (N,1) float64
      img_wh: (W, H) or None
    """
    with open(camera_path, "r") as f:
        dt = json.load(f)

    # --- SPADES format ---
    if "cameraMatrix" in dt:
        CMat = np.array(dt["cameraMatrix"], dtype=np.float64)
        dist = dt.get("distCoeffs", [0, 0, 0, 0, 0])
        dist = np.array(dist, dtype=np.float64).reshape(-1, 1)
        W = dt.get("Nu", None)
        H = dt.get("Nv", None)
        img_wh = (int(W), int(H)) if (W is not None and H is not None) else None
        return CMat, dist, img_wh

    # --- VBN format ---
    if "focalLengthX" in dt:
        fx = float(dt["focalLengthX"])
        fy = float(dt["focalLengthY"])
        cx = float(dt["principalPointX"])
        cy = float(dt["principalPointY"])
        W = int(dt.get("imageWidth", 1024))
        H = int(dt.get("imageHeight", 1024))

        # VBN "center" origin means (cx, cy) are offsets from image center
        # Convert to standard top-left pixel origin
        origin = dt.get("imageCoordinateOrigin", "top-left").lower()
        if origin == "center":
            cx = cx + W / 2.0
            cy = cy + H / 2.0

        CMat = np.array([
            [fx,  0.0, cx],
            [0.0, fy,  cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Distortion
        k1 = float(dt.get("distortionParamK1", 0.0))
        k2 = float(dt.get("distortionParamK2", 0.0))
        p1 = float(dt.get("distortionParamP1", 0.0))
        p2 = float(dt.get("distortionParamP2", 0.0))
        k3 = float(dt.get("distortionParamK3", 0.0))
        dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64).reshape(-1, 1)

        img_wh = (W, H)
        return CMat, dist, img_wh

    raise ValueError(
        f"Cannot parse camera JSON: {camera_path}\n"
        f"Expected either 'cameraMatrix' (SPADES) or 'focalLengthX' (VBN) keys."
    )


# -------------------------
# Points JSON loader (auto-detect format)
# -------------------------

def _load_object_points(points_path: str):
    """
    Auto-detect object points format:
      - SPADES format: dict with "keypoints" (list of [x,y,z]) and optional "mockup_scale_ratio"
      - VBN format: list of dicts with "id", "x", "y", "z"
    Returns:
      ref_keypoints: np.ndarray (K,3) float64
      scale: float
    """
    with open(points_path, "r") as f:
        dt = json.load(f)

    # --- SPADES format (dict with "keypoints") ---
    if isinstance(dt, dict) and "keypoints" in dt:
        scale = float(dt.get("mockup_scale_ratio", 1.0))
        ref_keypoints = np.array(dt["keypoints"], dtype=np.float64) * scale
        return ref_keypoints, scale

    # --- VBN format (list of dicts with "id", "x", "y", "z") ---
    if isinstance(dt, list) and len(dt) > 0 and "x" in dt[0]:
        # Sort by id to ensure consistent ordering
        dt_sorted = sorted(dt, key=lambda p: int(p["id"]))
        ref_keypoints = np.array(
            [[p["x"], p["y"], p["z"]] for p in dt_sorted],
            dtype=np.float64
        )
        scale = 1.0  # VBN points are already in real units (meters)
        return ref_keypoints, scale

    raise ValueError(
        f"Cannot parse points JSON: {points_path}\n"
        f"Expected either dict with 'keypoints' (SPADES) or list of {{id,x,y,z}} dicts (VBN)."
    )


# -------------------------
# Pose Estimator Factory
# -------------------------

def create_pose_estimator(pnp_kpts, camera_path, points_path):
    """
    Unified factory: auto-detects SPADES vs VBN format for both camera and points.
    """
    CMat, dist, img_wh = _load_camera_matrix_and_dist(camera_path)
    ref_keypoints, scale = _load_object_points(points_path)

    print(f"[PoseEstimator] Camera matrix:\n{CMat}")
    print(f"[PoseEstimator] Distortion: {dist.ravel()}")
    if img_wh is not None:
        print(f"[PoseEstimator] Image size (W×H): {img_wh[0]}×{img_wh[1]}")
    print(f"[PoseEstimator] Object points shape: {ref_keypoints.shape}, scale: {scale}")
    print(f"[PoseEstimator] Computing pose with top {pnp_kpts} keypoints")

    pose_estimator = PoseEstimator(ref_keypoints, CMat, dist)
    pnp_config = PnPConfig()
    return pose_estimator, pnp_config, scale


class PnPConfig:
    def __init__(
        self,
        pnp_solver="ransac_ap3p_then_iterative",
        refinement_algorithm="lm",
        ransac_reproj_error_px=6.0,
        ransac_iters=2000,
        ransac_conf=0.999,
        min_inliers=4,
    ):
        self.pnp_solver = str(pnp_solver)
        self.refinement_algorithm = str(refinement_algorithm)
        self.ransac_reproj_error_px = float(ransac_reproj_error_px)
        self.ransac_iters = int(ransac_iters)
        self.ransac_conf = float(ransac_conf)
        self.min_inliers = int(min_inliers)

        print("Pose solver:", self.pnp_solver)
        print("Refinement Algorithm:", self.refinement_algorithm)


class PoseEstimator:
    def __init__(self, model_coordinates, camera_matrix, dist_coefs):
        self.model_points = np.asarray(model_coordinates, dtype=np.float64)  # (K,3)
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)     # (3,3)
        self.dist_coefs = np.asarray(dist_coefs, dtype=np.float64)           # (N,1)

    @staticmethod
    def _reproj_error(obj_pts, img_pts, K, dist, rvec, tvec):
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)
        err = np.linalg.norm(proj - img_pts, axis=1)
        return err, proj

    def solve_pose(self, image_points, array, pnpconfig, flag=0):
        """
        image_points: (K,2) full-image pixel coords
        array: indices of visible points (1D)
        Returns:
          q_xyzw: (4,) float32
          t:      (3,) float32
          ok:     bool
        """
        idx = np.asarray(array, dtype=np.int64).reshape(-1)
        if idx.size < pnpconfig.min_inliers:
            return np.zeros(4, np.float32), np.zeros(3, np.float32), False

        img_pts = np.ascontiguousarray(image_points[idx].astype(np.float64))
        obj_pts = np.ascontiguousarray(self.model_points[idx].astype(np.float64))

        ap3p_flag = cv2.SOLVEPNP_AP3P if hasattr(cv2, "SOLVEPNP_AP3P") else cv2.SOLVEPNP_EPNP

        # 1) RANSAC init
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_pts, img_pts,
                self.camera_matrix, self.dist_coefs,
                iterationsCount=pnpconfig.ransac_iters,
                reprojectionError=pnpconfig.ransac_reproj_error_px,
                confidence=pnpconfig.ransac_conf,
                flags=ap3p_flag,
            )
        except cv2.error:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_pts, img_pts,
                self.camera_matrix, self.dist_coefs,
                flags=ap3p_flag,
            )

        # 1b) Fallback to ITERATIVE
        if (not ok) or (rvec is None) or (tvec is None) or (inliers is None) or (len(inliers) < pnpconfig.min_inliers):
            try:
                okF, rvecF, tvecF = cv2.solvePnP(
                    obj_pts, img_pts,
                    self.camera_matrix, self.dist_coefs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if okF and (rvecF is not None) and (tvecF is not None):
                    ok, rvec, tvec = okF, rvecF, tvecF
                    inliers = np.arange(obj_pts.shape[0], dtype=np.int32).reshape(-1, 1)
                else:
                    return np.zeros(4, np.float32), np.zeros(3, np.float32), False
            except cv2.error:
                return np.zeros(4, np.float32), np.zeros(3, np.float32), False

        inl = inliers.reshape(-1)
        obj_in = obj_pts[inl]
        img_in = img_pts[inl]

        # 2) Refit on inliers
        refit_flag = cv2.SOLVEPNP_SQPNP if hasattr(cv2, "SOLVEPNP_SQPNP") else cv2.SOLVEPNP_ITERATIVE
        try:
            ok2, rvec2, tvec2 = cv2.solvePnP(
                obj_in, img_in,
                self.camera_matrix, self.dist_coefs,
                rvec=rvec, tvec=tvec,
                useExtrinsicGuess=True,
                flags=refit_flag
            )
            if ok2 and (rvec2 is not None) and (tvec2 is not None):
                rvec, tvec = rvec2, tvec2
        except cv2.error:
            pass

        # 3) Refinement (LM / VVS)
        algo = pnpconfig.refinement_algorithm.lower()
        if algo == "lm":
            try:
                rvec, tvec = cv2.solvePnPRefineLM(
                    obj_in, img_in, self.camera_matrix, self.dist_coefs,
                    rvec=rvec, tvec=tvec
                )
            except cv2.error:
                pass
        elif algo == "vvs":
            try:
                rvec, tvec = cv2.solvePnPRefineVVS(
                    obj_in, img_in, self.camera_matrix, self.dist_coefs,
                    rvec=rvec, tvec=tvec
                )
            except cv2.error:
                pass

        # 4) Reprojection error gate
        try:
            err, _ = self._reproj_error(
                obj_in, img_in, self.camera_matrix, self.dist_coefs, rvec, tvec
            )
            if (not np.isfinite(err).all()) or (float(np.mean(err)) > 50.0):
                return np.zeros(4, np.float32), np.zeros(3, np.float32), False
        except cv2.error:
            pass

        # 5) Convert to quaternion (xyzw) + translation
        rmat, _ = cv2.Rodrigues(rvec)
        q_xyzw = R.from_matrix(rmat).as_quat().astype(np.float32)
        t = np.asarray(tvec, dtype=np.float32).reshape(-1)

        tn = float(np.linalg.norm(t))
        if (not np.isfinite(tn)) or (tn < 1e-6) or (tn > 200.0):
            return np.zeros(4, np.float32), np.zeros(3, np.float32), False

        return q_xyzw, t, True


# -------------------------
# Heatmap -> Keypoints helpers
# -------------------------

def cal_padding(box):
    xmin, ymin, xmax, ymax = box
    xl = xmax - xmin
    yl = ymax - ymin
    xpad = ypad = 0.0
    if xl > yl:
        ypad = (xl - yl) / 2.0
    elif yl > xl:
        xpad = (yl - xl) / 2.0
    return xpad, ypad


def map_kps_to_full_image(kps, scores, bbox, output_size, kp_min_count=5, det_thresh=0.2, full_im_size=224):
    if isinstance(full_im_size, (tuple, list, np.ndarray)):
        crop_w = float(full_im_size[0])
        crop_h = float(full_im_size[1])
    else:
        crop_w = crop_h = float(full_im_size)

    if isinstance(output_size, (tuple, list, np.ndarray)):
        out_h = float(output_size[0])
        out_w = float(output_size[1])
    else:
        out_h = out_w = float(output_size)

    xpad, ypad = cal_padding(bbox)

    points = np.zeros((kps.shape[0], 3), dtype=np.float32)
    thresh = float(det_thresh)

    max_count = len([s for s in scores if 0.0 < float(s) < 1.0])
    kp_min_count = min(int(kp_min_count), max_count)

    count = 0
    while count < kp_min_count and thresh > 0.0:
        thresh *= 0.9
        count = 0
        for i in range(len(kps)):
            points[i, 0] = (kps[i, 0] * (crop_w / out_w)) - xpad
            points[i, 1] = (kps[i, 1] * (crop_h / out_h)) - ypad
            points[i, 2] = float(scores[i])
            if thresh < points[i, 2] < 1.0:
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


def refine_keypoints(heatmaps, keypoints):
    B, K, H, W = heatmaps.shape
    refined = keypoints.copy()

    for b in range(B):
        for j in range(K):
            x, y = keypoints[b, j].astype(int)
            if 1 <= x < W - 1 and 1 <= y < H - 1:
                patch = heatmaps[b, j, y-1:y+2, x-1:x+2]
                if patch.sum() > 1e-6:
                    dx = 0.5 * (patch[1, 2] - patch[1, 0])
                    dy = 0.5 * (patch[2, 1] - patch[0, 1])
                    dxx = patch[1, 2] + patch[1, 0] - 2 * patch[1, 1]
                    dyy = patch[2, 1] + patch[0, 1] - 2 * patch[1, 1]
                    dxy = 0.25 * (patch[2, 2] + patch[0, 0] - patch[0, 2] - patch[2, 0])
                    hessian = np.array([[dxx, dxy], [dxy, dyy]], dtype=np.float64)
                    grad = np.array([-dx, -dy], dtype=np.float64)
                    if np.linalg.det(hessian) > 1e-6:
                        shift = np.linalg.inv(hessian) @ grad
                        refined[b, j] += shift.astype(np.float32)

    return refined


def get_max_preds(batch_heatmaps):
    assert isinstance(batch_heatmaps, np.ndarray)
    assert batch_heatmaps.ndim == 4

    B, K, H, W = batch_heatmaps.shape
    reshaped = batch_heatmaps.reshape((B, K, -1))
    idx = np.argmax(reshaped, axis=2)
    maxvals = np.amax(reshaped, axis=2).reshape((B, K, 1))

    preds = np.zeros((B, K, 2), dtype=np.float32)
    preds[:, :, 0] = idx % W
    preds[:, :, 1] = idx // W

    refined = refine_keypoints(batch_heatmaps, preds)
    return refined, maxvals


def extract_refined_keypoints_from_heatmaps(
    hm, bbox, output_size=(56, 56), full_im_size=(224, 224), kp_min_count=5, kp_det_thresh=0.2
):
    B, K, Hm, Wm = hm.shape

    preds, scores = get_max_preds(hm)
    refined = preds

    kps_all, vis_all = [], []
    for b in range(B):
        pts_adj, thresh = map_kps_to_full_image(
            refined[b], scores[b], bbox[b],
            output_size=(Hm, Wm),
            kp_min_count=kp_min_count,
            det_thresh=kp_det_thresh,
            full_im_size=full_im_size,
        )
        kps_img, vis = convert_to_image_coords(pts_adj, thresh, bbox[b], K)
        kps_all.append(kps_img)
        vis_all.append(vis)

    return np.stack(kps_all, axis=0), vis_all


# -------------------------
# Visualization / Logging helpers
# -------------------------

def vis_frame_heatmap_wandb(frame, heatmaps, plot_caption):
    frame_ = frame.permute(1, 2, 0)
    heatmaps_ = F.interpolate(heatmaps.sum(0).unsqueeze(0).unsqueeze(0), size=(224, 224), mode="bilinear")

    fig, ax = plt.subplots()
    ax.imshow(frame_.cpu().numpy())
    ax.imshow(heatmaps_.squeeze(0).squeeze(0).cpu().numpy(), cmap="jet", alpha=0.6)
    plot = wandb.Image(fig, caption=plot_caption)
    plt.close(fig)
    return plot


# -------------------------
# PnP Pose estimation wrapper
# -------------------------

def estimate_pose_from_keypoints(points, array, pe, pnpconfig):
    B, K, _ = points.size()
    q_list, r_list, ok_list = [], [], []

    for b in range(B):
        q1, r1, ok = pe.solve_pose(
            np.array(points[b].detach().cpu().numpy()),
            array[b].detach().cpu().numpy(),
            pnpconfig,
            flag=1,
        )
        q_list.append(q1)
        r_list.append(r1)
        ok_list.append(bool(ok))

    return q_list, r_list, ok_list


# -------------------------
# Pose scoring
# -------------------------

def compute_pose_scores_batch(batch, r_pred, q_pred, scale=1.0, eps=1e-8):
    B, T = r_pred.shape[:2]

    ori_deg   = np.zeros((B, T), dtype=np.float32)
    loc_err   = np.zeros((B, T), dtype=np.float32)
    range_err = np.zeros((B, T), dtype=np.float32)
    esa_score = np.zeros((B, T), dtype=np.float32)

    for b in range(B):
        for t in range(T):
            gt_q = batch["quats"][b, t].detach().cpu().numpy().astype(np.float64)
            gt_r = batch["rotations"][b, t].detach().cpu().numpy().astype(np.float64)

            pq = q_pred[b, t].detach().cpu().numpy().astype(np.float64)
            pr = r_pred[b, t].detach().cpu().numpy().astype(np.float64)

            gt_q = gt_q / (np.linalg.norm(gt_q) + eps)
            pq   = pq   / (np.linalg.norm(pq)   + eps)

            mmp = float(np.inner(pq, gt_q))
            mmp = np.clip(mmp, -1.0, 1.0)
            ang_rad = 2.0 * np.arccos(abs(mmp))
            ang_deg = ang_rad * 180.0 / np.pi

            gt_r_s = gt_r * scale
            pr_s   = pr   * scale

            loc = float(np.linalg.norm(pr_s - gt_r_s))
            rng = float(np.linalg.norm(gt_r_s) + eps)
            rerr = loc / rng

            esa = rerr + ang_rad

            ori_deg[b, t]   = ang_deg
            loc_err[b, t]   = loc
            range_err[b, t] = rerr
            esa_score[b, t] = esa

    return ori_deg, loc_err, range_err, esa_score


def _to_float(x):
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def update_global_metrics(global_poses_errors, batch_p_errors):
    for seq, vals in batch_p_errors.items():
        if seq not in global_poses_errors:
            global_poses_errors[seq] = {
                "esa": [], "loc_error": [], "ori_error": [], "range_error": [], "pck": []
            }
        global_poses_errors[seq]["pck"].append(_to_float(vals["pck"]))
        global_poses_errors[seq]["esa"].append(_to_float(vals["esa"]))
        global_poses_errors[seq]["loc_error"].append(_to_float(vals["loc_error"]))
        global_poses_errors[seq]["ori_error"].append(_to_float(vals["ori_error"]))
        global_poses_errors[seq]["range_error"].append(_to_float(vals["range_error"]))


def compute_speed_metrics_batch(
    q_gt_wxyz: torch.Tensor,
    t_gt: torch.Tensor,
    q_pred_wxyz: torch.Tensor,
    t_pred: torch.Tensor,
    eps: float = 1e-8,
    use_speedplus_gate: bool = False,
    theta_q_deg: float = 0.169,
    theta_t_mm_per_m: float = 2.173,
):
    qg = q_gt_wxyz / (q_gt_wxyz.norm(dim=-1, keepdim=True).clamp_min(eps))
    qp = q_pred_wxyz / (q_pred_wxyz.norm(dim=-1, keepdim=True).clamp_min(eps))

    dot = torch.sum(qp * qg, dim=-1).abs().clamp(0.0, 1.0)
    E_q_rad = 2.0 * torch.acos(dot)
    E_q_deg = E_q_rad * (180.0 / np.pi)

    diff = t_pred - t_gt
    E_t = torch.linalg.norm(diff, dim=-1)

    t_norm = torch.linalg.norm(t_gt, dim=-1).clamp_min(eps)
    E_t_over = E_t / t_norm

    E_pose = E_q_rad + E_t_over

    if use_speedplus_gate:
        theta_q_rad = float(theta_q_deg) * (np.pi / 180.0)
        theta_t = float(theta_t_mm_per_m) / 1000.0
        ok = (E_q_rad < theta_q_rad) & (E_t_over < theta_t)
        E_pose_star = torch.where(ok, torch.zeros_like(E_pose), E_pose)
    else:
        E_pose_star = E_pose

    return (
        E_q_rad.detach().cpu().numpy(),
        E_q_deg.detach().cpu().numpy(),
        E_t.detach().cpu().numpy(),
        E_t_over.detach().cpu().numpy(),
        E_pose.detach().cpu().numpy(),
        E_pose_star.detach().cpu().numpy(),
    )