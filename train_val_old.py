# train_val.py
import os
import math
import time
import warnings
from typing import Dict, Tuple, Optional, List, Any
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler
import numpy as np
import datetime

# plot in tensorboard
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F

# ---- Silence the albumentations ShiftScaleRotate warning (true fix is in transforms.py) ----
warnings.filterwarnings(
    "ignore",
    message="ShiftScaleRotate is a special case of Affine transform. Please use Affine transform instead."
)

# ---- DDP imports ----
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from model import SpacecraftPoseTransformer, SpacecraftPoseTCN, SpacecraftPoseGCN, SpacecraftPoseDualPath

# ---- dataset imports  ----
from dataset import SPADESDataset, SPARKDataset
from transforms import build_transforms
from utils import unnormalize_keypoints, save_kp_overlay, build_warmup_cosine_scheduler, load_camera_intrinsics, freeze_all_but_dualpath_and_heads
from utils import project_keypoints, debug_check_heatmap_alignment, debug_check_pred_alignment, get_heatmap_power_for_epoch, get_tau_for_epoch
from utils import cosine_schedule, temporal_consistency_loss, quat_inverse, quat_mul, quat_relative, scale_translation_to_full_image
from torch.utils.data import Subset

# ---- perf toggles (safe even on CPU; they only take effect on CUDA) ----
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1

def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(backend="nccl"):
    """
    Initializes default process group using env:// (works with torchrun + Slurm).
    Sets current CUDA device from LOCAL_RANK.
    """
    if is_dist_avail_and_initialized():
        return  # already done
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend=backend, init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    else:
        # Single-process fallback (no DDP)
        pass


# ---------- Optimizer Helper (head LR multiplier) -------
def split_params_for_optimizer(
    model,
    base_lr,
    weight_decay=0.05,
    head_lr_mult=5.0,
    block_lr_mult=0.5,
):
    real = model.module if hasattr(model, "module") else model

    adapter_params, head_params, block_params = [], [], []

    def no_weight_decay(n):
        return n.endswith("bias") or "bn" in n.lower() or "norm" in n.lower()

    for n, p in real.named_parameters():
        if not p.requires_grad:
            continue

        # Adapters
        if "adapter" in n or "dualpath" in n:
            adapter_params.append((n, p))

        # Heads
        elif any(k in n for k in ["hm_head", "keypoint_head", "pose_head", "finer_cnn", "pnp", "kp_res_head"]):
            head_params.append((n, p))

        # Patch embed / norm / last blocks
        elif any(k in n for k in ["patch_embed", "norm", "blocks."]):
            block_params.append((n, p))

    def make_split(named_params, lr):
        decay, no_decay = [], []
        for n, p in named_params:
            (no_decay if no_weight_decay(n) else decay).append(p)
        groups = []
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
        return groups

    groups = []
    groups += make_split(adapter_params, base_lr)
    groups += make_split(block_params, base_lr * block_lr_mult)
    groups += make_split(head_params, base_lr * head_lr_mult)

    # Debug info
    print(f"[opt] LR groups:")
    for g in groups:
        n_params = sum(p.numel() for p in g["params"])
        print(f"  - {n_params} params @ lr={g['lr']}, wd={g['weight_decay']}")

    return groups

# ---------- Subset helpers ----------
def _subset_indices(n_total: int, n_take: int, mode: str = "first", seed: int = 0) -> List[int]:
    n_take = int(min(max(0, n_take), n_total))
    if n_take <= 0 or n_take == n_total:
        return list(range(n_total))
    if mode == "first":
        return list(range(n_take))
    # random but deterministic
    import numpy as _np
    rng = _np.random.RandomState(seed)
    idx = _np.arange(n_total)
    rng.shuffle(idx)
    return idx[:n_take].tolist()

def _broadcast_indices(idxs: List[int]) -> List[int]:
    if not is_dist_avail_and_initialized():
        return idxs
    obj_list = [idxs]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]
# ------------------------------------

def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()

def seed_everything(seed: int):
    random.seed(seed); np = __import__("numpy"); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    # Determinism off (we want speed); set if you need exact reproducibility:
    torch.backends.cudnn.deterministic = False

# ---- loss suite (final choice we agreed on) ----
from losses import (
    geodesic_quat_loss, translation_l1,
    kp_heatmap_loss_combo,
    heatmap_kl_div_from_logits, softargmax_2d_from_logits,
    heatmap_kl_div_from_logits_masked,
)

def log(s: str):
    print(s, flush=True)

def enable_tf32():
    """Optional convenience; harmless to call more than once."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def _sync_and_trim_dataset(ds, name: str):
    """Make sure all ranks see identical dataset length by trimming to the global MIN length."""
    local_len = len(ds)
    if is_dist_avail_and_initialized():
        lens = [None for _ in range(get_world_size())]
        dist.all_gather_object(lens, local_len)
        gmin, gmax = min(lens), max(lens)
        if is_main_process() and gmin != gmax:
            log(f"[WARN] {name} length mismatch across ranks: {lens} -> trimming all to {gmin}")
        if local_len != gmin:
            from torch.utils.data import Subset
            ds = Subset(ds, range(gmin))
    return ds

# ----------------------------
# Metrics + SOTA helpers
# ----------------------------
@torch.no_grad()
def rot_err_deg(pred_q: torch.Tensor, gt_q: torch.Tensor) -> torch.Tensor:
    # geodesic_quat_loss returns radians (B,T) with reduction=None
    ang = geodesic_quat_loss(pred_q, gt_q, reduction=None)
    return ang * (180.0 / math.pi)

@torch.no_grad()
def quat_ori_err_rad(pred_q: torch.Tensor, gt_q: torch.Tensor) -> torch.Tensor:
    """(B,T,4) -> radians, shape (B,T)."""
    eps = 1e-8
    p = pred_q / (pred_q.norm(dim=-1, keepdim=True) + eps)
    g = gt_q   / (gt_q.norm(dim=-1, keepdim=True) + eps)
    dot = (p * g).sum(dim=-1).clamp(-1.0, 1.0).abs()
    return 2.0 * torch.arccos(dot)  # radians

@torch.no_grad()
def trans_mae(pred_t: torch.Tensor, gt_t: torch.Tensor) -> torch.Tensor:
    return (pred_t - gt_t).abs().mean(dim=-1)  # (B,T)

@torch.no_grad()
def upsample_argmax_kps(hm_logits: torch.Tensor, out_hw: Tuple[int,int]) -> torch.Tensor:
    """
    hm_logits: (B,T,K,Hm,Wm). Returns (B,T,K,2) in image pixels using argmax in heatmap space, then scale.
    """
    B,T,K,Hm,Wm = hm_logits.shape
    probs = torch.softmax(hm_logits.reshape(B*T*K, Hm*Wm), dim=-1).reshape(B*T*K, Hm, Wm)
    flat_idx = probs.reshape(B*T*K, -1).argmax(dim=-1)
    ys = (flat_idx // Wm).float()
    xs = (flat_idx %  Wm).float()
    kps_hm = torch.stack([xs, ys], dim=-1).reshape(B, T, K, 2)  # heatmap px

    Hi, Wi = out_hw
    sx = float(Wi) / float(Wm)
    sy = float(Hi) / float(Hm)
    kps_img = kps_hm.clone()
    kps_img[..., 0] = kps_hm[..., 0] * sx
    kps_img[..., 1] = kps_hm[..., 1] * sy
    return kps_img

@torch.no_grad()
def pck_from_img_px(pred_xy: torch.Tensor, gt_norm_xy: torch.Tensor, out_hw: Tuple[int,int], thresh_px: float, vis: Optional[torch.Tensor]) -> float:
    """
    pred_xy: (B,T,K,2) in image pixels
    gt_norm_xy: (B,T,K,2) in [-1,1] normalized image coords (dataset format)
    vis: (B,T,K) or None
    """
    Hi, Wi = out_hw
    gx = ((gt_norm_xy[..., 0] + 1.0) * 0.5) * (Wi - 1.0)
    gy = ((gt_norm_xy[..., 1] + 1.0) * 0.5) * (Hi - 1.0)
    gt_px = torch.stack([gx, gy], dim=-1)
    err = (pred_xy - gt_px).norm(dim=-1)  # (B,T,K)
    err = err.reshape(-1)   # all joints in batch
    if vis is not None:
        m = (vis > 0).float()
        m = m.reshape(-1)
        correct = ((err <= thresh_px) * m).sum() / m.sum().clamp_min(1.0)
    else:
        correct = (err <= thresh_px).float().mean()

    # if vis is not None:
    #     m = (vis > 0).float()
    #     denom = m.sum().clamp_min(1.0)
    #     correct = ((err <= float(thresh_px)).float() * m).sum() / denom
    # else:
    #     correct = (err < float(thresh_px)).float().mean()
    return float(correct.item())

@torch.no_grad()
def keypoint_mae_px_from_logits(hm_logits: torch.Tensor, gt_kps_norm: torch.Tensor) -> torch.Tensor:
    """
    hm_logits: (B,T,K,H,W) — raw logits
    gt_kps_norm: (B,T,K,2) — normalized [-1,1] keypoints (your dataset format)
    Returns mean absolute error in *heatmap pixels* averaged over (K,2,B,T).
    """
    B, T, K, H, W = hm_logits.shape
    pred_px = softargmax_2d_from_logits(hm_logits)  # (B,T,K,2) in [0..W-1],[0..H-1]
    gx = ((gt_kps_norm[..., 0] + 1.0) * 0.5) * (W - 1.0)
    gy = ((gt_kps_norm[..., 1] + 1.0) * 0.5) * (H - 1.0)
    gt_px = torch.stack([gx, gy], dim=-1)
    return (pred_px - gt_px).abs().mean()  # scalar

# --------------------------------
# Scheduler helper for Heatmap
# --------------------------------

def get_w_hm_for_epoch(cfg, epoch: int) -> float:
    sched = cfg.get("loss", {}).get("w_hm_schedule", None)
    base = float(cfg.get("loss", {}).get("w_hm", 1.0))
    if not sched:
        return base
    return cosine_schedule(
        start=sched.get("start", base),
        end=sched.get("end", base),
        start_epoch=sched.get("start_epoch", 0),
        end_epoch=sched.get("end_epoch", 0),
        cur_epoch=epoch,
    )

# ----------------------------
# Model factory
# ----------------------------
def build_model(cfg) -> nn.Module:
    common = dict(
        aggregator=cfg["model"].get("aggregator", "attn"),
        output_mode=cfg["model"]["output_mode"],
        ckpt_path=cfg["model"]["backbone_ckpt"],
        freeze_backbone=cfg["model"].get("freeze_backbone", True),
        fuse_side=cfg["model"].get("fuse_side", False),
        num_kp=cfg["data"]["num_kp"],
        d_side=cfg["model"].get("d_side", 0),
        heatmap_out=tuple(cfg["model"].get("heatmap_out", [56, 56])),


        # camera intrinsics for PnP (if used)
        cam_params = load_camera_intrinsics(cfg["data"]["cam_intrinsics_json"]) if cfg["model"].get("use_pnp", False) else None,
        

        # NEW: heatmap head choice + params
        hm_head=cfg["model"].get("hm_head", "film"),
        hm_mid_c=cfg["model"].get("hm_mid_c", 128),
        hm_use_film=cfg["model"].get("hm_use_film", False),
        use_kp_residual=cfg["model"].get("use_residual", False),

        # NEW: TCN knobs
        tcn_hidden=cfg["model"].get("tcn_hidden", 384),
        tcn_kernel=cfg["model"].get("tcn_kernel", 3),
        tcn_depth=cfg["model"].get("tcn_depth", 2),
        tcn_dropout=cfg["model"].get("tcn_dropout", 0.1),

        # Finer_CNN and Pnp
        use_finer_cnn=cfg["model"].get("finer_cnn", False),
        return_pnp=cfg["model"].get("use_pnp", False),

        # use_kp_residuals=cfg["model"].get("use_kp_residuals", False),
        cfg = cfg,
    )
    temporal = cfg["model"]["temporal"]
    if   temporal == "transformer": return SpacecraftPoseTransformer(**common)
    elif temporal == "tcn":         return SpacecraftPoseTCN(**common)
    elif temporal == "gcn":         return SpacecraftPoseGCN(**common)
    elif temporal in [None, "", "none"]: return SpacecraftPoseDualPath(**common)  # dual_path is a variant of transformer with dual-path adapter
    else: raise ValueError(f"Unknown temporal kind: {temporal}")

# ----------------------------
# Dataloaders
# ----------------------------
def make_dataloaders(cfg):
    ds_name = cfg["data"].get("name", "").lower()

    # Build the transform separately
    # Build the crop + full-frame transforms separately
    tfm_crop = build_transforms(
        aug_mode=cfg["train"].get("aug_mode", "kp_geo"),
        allow_tiny_crop=bool(cfg["train"].get("aug_allow_tiny_crop", False)),
        img_size=int(cfg["data"]["input_size"]),
    )
    tfm_full = build_transforms(
        aug_mode=cfg["train"].get("aug_mode_full", "photometric"),
        allow_tiny_crop=False,
        img_size=int(cfg["data"]["input_size"]),
    )
    # ONLY include kwargs your dataset __init__ supports
    base_kwargs = dict(
        gt_folder=cfg["data"]["root"],
        short_mem=cfg["data"]["short_mem"],
        input_size=cfg["data"]["input_size"],
        sampling_rate=cfg["data"]["sampling_rate"],
        crop=cfg["data"].get("crop", True),
        crop_prob = cfg['data'].get('crop_prob', 0.0),
        eval_use_crop = cfg['data'].get('eval_use_crop', False),
        heatmap_out=tuple(cfg["model"].get("heatmap_out", [56, 56])),
    )

    if ds_name == "spades":
        train_set = SPADESDataset(mode="train", spatial_transforms=tfm_crop, spatial_transforms_full=tfm_full, **base_kwargs)
        val_mode  = "val" if os.path.exists(os.path.join(cfg["data"]["root"], "val.json")) else "test"
        val_set   = SPADESDataset(mode=val_mode, spatial_transforms=None, spatial_transforms_full=None, **base_kwargs)

    elif ds_name == "spark":
        train_set = SPARKDataset(mode="train", spatial_transforms=tfm_crop, spatial_transforms_full=tfm_full, **base_kwargs)
        val_set   = SPARKDataset(mode="val",   spatial_transforms=None, spatial_transforms_full=None, **base_kwargs)

    else:
        raise ValueError("data.name must be 'spades' or 'spark'")

    # ==== Optional debugging subsets ====
    train_take = int(cfg["data"].get("train_subset", 0) or 0)
    val_take   = int(cfg["data"].get("val_subset", 0) or 0)
    sub_mode   = str(cfg["data"].get("subset_mode", "first")).lower()
    sub_seed   = int(cfg["data"].get("subset_seed", 0) or 0)

    if train_take > 0:
        if is_main_process():
            idxs = _subset_indices(len(train_set), train_take, mode=sub_mode, seed=sub_seed)
        else:
            idxs = None
        # broadcast to all ranks (including single-process no-op)
        idxs = _broadcast_indices(idxs if idxs is not None else [])
        train_set = Subset(train_set, idxs)
        if is_main_process():
            log(f"[Subset] train -> {len(idxs)} samples (mode={sub_mode}, seed={sub_seed})")

    if val_take > 0:
        if is_main_process():
            idxs = _subset_indices(len(val_set), val_take, mode=sub_mode, seed=sub_seed + 1)
        else:
            idxs = None
        idxs = _broadcast_indices(idxs if idxs is not None else [])
        val_set = Subset(val_set, idxs)
        if is_main_process():
            log(f"[Subset] val   -> {len(idxs)} samples (mode={sub_mode}, seed={sub_seed + 1})")

    # ==== NEW: force identical lengths across ranks ====
    train_set = _sync_and_trim_dataset(train_set, "train_set")
    val_set   = _sync_and_trim_dataset(val_set,   "val_set")

    # ==== Distributed samplers ====
    world = get_world_size()
    rank  = get_rank()
    train_sampler = DistributedSampler(
        train_set, num_replicas=world, rank=rank, shuffle=True, drop_last=True
    ) if world > 1 else None

    val_sampler = DistributedSampler(
        val_set, num_replicas=world, rank=rank, shuffle=False, drop_last=False
    ) if world > 1 else None

    nw = int(cfg["train"].get("num_workers", 4))
    bs = int(cfg["train"]["batch_size"])
    vbs = int(cfg["train"].get("val_batch_size", bs))

    train_loader = DataLoader(
        train_set,
        batch_size=bs,
        shuffle=(train_sampler is None),
        num_workers=nw,
        pin_memory=True,
        drop_last=True,                 # keep True for strict equal steps
        sampler=train_sampler,
        persistent_workers=(nw > 0),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=vbs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        drop_last=False,
        sampler=val_sampler,
        persistent_workers=(nw > 0),
    )

    # ==== Sanity checks (visible in logs) ====
    if is_dist_avail_and_initialized():
        tb = len(train_loader); vb = len(val_loader)
        tsl = len(train_sampler) if train_sampler is not None else len(train_set)
        vsl = len(val_sampler)   if val_sampler   is not None else len(val_set)

        all_tb = [None]*world; dist.all_gather_object(all_tb, tb)
        all_vb = [None]*world; dist.all_gather_object(all_vb, vb)
        all_tsl = [None]*world; dist.all_gather_object(all_tsl, tsl)
        all_vsl = [None]*world; dist.all_gather_object(all_vsl, vsl)

        if is_main_process():
            log(f"[check] train_batches per rank: {all_tb}")
            log(f"[check] val_batches   per rank: {all_vb}")
            log(f"[check] train_sampler_len per rank: {all_tsl}")
            log(f"[check] val_sampler_len   per rank: {all_vsl}")

        # hard assert: they must match now
        if len(set(all_tb)) != 1 or len(set(all_tsl)) != 1:
            raise RuntimeError(f"Train loader/sampler mismatch across ranks: "
                               f"batches={all_tb}, sampler_len={all_tsl}")

    return train_loader, val_loader, train_sampler, val_sampler


# helper formatters
def fmt(x):
    return "-" if x is None else f"{(x.item() if torch.is_tensor(x) else float(x)):.3f}"
def fval(x):
    return None if x is None else (x.item() if torch.is_tensor(x) else float(x))

# ---------------------------
# Unified loss/metric function (training objective)
#----------------------------

# =========================
# Compute Losses
# =========================
def compute_losses(
    out: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
    kps_gt: torch.Tensor,
    hm_gt: torch.Tensor,
    gt_q: torch.Tensor,
    gt_r: torch.Tensor,
    vis: torch.Tensor,
    delta_Rt_gt: torch.Tensor,
    crop_xywh: Optional[torch.Tensor] = None,  
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, float]]:

    output_mode = cfg["model"]["output_mode"]
    W, H = cfg["data"].get("original_size", [1280, 720])

    # weights
    w_rot   = float(cfg["loss"].get("w_rot",   1.0))
    w_trans = float(cfg["loss"].get("w_trans", 1.0))
    tau     = float(cfg["loss"].get("tau_now", cfg["loss"].get("tau", 1.0)))
    w_hm    = float(cfg["loss"].get("w_hm",    1.0))
    w_kp    = float(cfg["loss"].get("w_kp",    0.5))
    w_vo    = float(cfg["loss"].get("w_vo",    1.0))
    w_cons  = float(cfg["loss"].get("w_cons",  0.0))   # consistency weight
    w_temp  = float(cfg["loss"].get("w_temp",  0.0))

    device = gt_q.device
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    parts: Dict[str, torch.Tensor] = {}
    metrics: Dict[str, float] = {}

    # ---------------- Pose branch ----------------
    if output_mode in ("quat", "both"):
        q_frame = out["q_frame"].float()  # (B,T,4)
        t_frame = out["t_frame"].float()  # (B,T,3)
        B, T, _ = t_frame.shape
        K = load_camera_intrinsics(cfg["data"]["cam_intrinsics_json"])["K"]
        K_torch = torch.from_numpy(K).to(device).unsqueeze(0).repeat(B*T, 1, 1)



        # ---- crop-aware scaling for predicted translations ----
        if crop_xywh is not None:
            # B, T, _ = t_frame.shape
            t_frame_flat = t_frame.view(B*T, 3)
            crop_flat = crop_xywh.view(B*T, 4)
            t_frame_scaled = scale_translation_to_full_image(
                                    t_frame_flat, crop_flat, 
                                    intrinsics=K_torch, 
                                    img_size=(W,H))
            
            t_frame = t_frame_scaled.view(B, T, 3)

        # per-frame loss (optimized)
        loss_rot = geodesic_quat_loss(q_frame, gt_q.float(), reduction="mean")
        loss_trn = translation_l1(t_frame, gt_r.float(), reduction="mean")
        parts["rot_frame"] = loss_rot
        parts["trn_frame"] = loss_trn
        total_loss = total_loss + w_rot * loss_rot + w_trans * loss_trn

        # aggregated (monitor only)
        if "q_agg" in out and "t_agg" in out:
            gt_q_last = gt_q[:, -1]
            gt_r_last = gt_r[:, -1]
            parts["rot_agg"] = geodesic_quat_loss(out["q_agg"].float(), gt_q_last, reduction="mean")
            parts["trn_agg"] = translation_l1(out["t_agg"].float(), gt_r_last, reduction="mean")

        # consistency loss (pose deltas)
        if w_cons > 0:
            # relative quats
            q_rel_pred = quat_relative(q_frame[:, 1:], q_frame[:, :-1])   # (B,T-1,4)
            q_rel_gt   = quat_relative(gt_q[:, 1:], gt_q[:, :-1])
            L_q_cons = geodesic_quat_loss(q_rel_pred, q_rel_gt, reduction="mean")

            # relative translations
            t_rel_pred = t_frame[:, 1:] - t_frame[:, :-1]
            t_rel_gt   = gt_r[:, 1:] - gt_r[:, :-1]
            L_t_cons   = F.smooth_l1_loss(t_rel_pred, t_rel_gt, reduction="mean")

            L_cons = L_q_cons + L_t_cons
            total_loss = total_loss + w_cons * L_cons
            parts["cons"] = L_cons

        with torch.no_grad():
            ori_err_rad = quat_ori_err_rad(q_frame, gt_q)  # (B,T)
            metrics.update(dict(
                rot_mae_deg=float((ori_err_rad.mean() * 180.0 / math.pi).item()),
                trans_mae=float((t_frame - gt_r).norm(dim=-1).mean().item()),
            ))

    # -------------- VO branch --------------
    if "delta_Rt" in out and out["delta_Rt"] is not None:
        L_vo = F.smooth_l1_loss(out["delta_Rt"], delta_Rt_gt, reduction="mean")
        parts["vo"] = L_vo
        total_loss = total_loss + w_vo * L_vo

    # -------------- Heatmap / KP branch --------------
    if output_mode in ("kp_heatmap", "both"):
        hm_logits = out["heatmaps_frame"].float()  # (B,T,K,H,W)
        B,T,K,Hm_pred,Wm_pred = hm_logits.shape

        # per-frame keypoints
        kps_pred_norm = None
        if out["keypoints_frame"] is not None:
            kps_px = out["keypoints_frame"].float()
            x = 2.0 * (kps_px[..., 0] / (Wm_pred - 1.0)) - 1.0
            y = 2.0 * (kps_px[..., 1] / (Hm_pred - 1.0)) - 1.0
            kps_pred_norm = torch.stack([x, y], dim=-1)

        hm_kp_total, dbg = kp_heatmap_loss_combo(
            hm_logits=hm_logits,
            hm_target=hm_gt.float(),
            kp_target_norm=kps_gt.float(),
            vis=vis.float(),
            tau=tau,
            w_hm=w_hm,
            w_kp=w_kp,
            kp_space="norm",
            kps_pred_norm=kps_pred_norm,
            target_power=float(cfg["loss"].get("heatmap_target_power", 1.0)),
        )

        L_reg = 1e-6 * (hm_logits**2).mean()
        total_loss = total_loss + hm_kp_total + L_reg
        parts["hm"] = dbg.get("hm_raw", hm_kp_total)
        if "kp_raw" in dbg: parts["kp"] = dbg["kp_raw"]
        parts["hm_reg"] = L_reg

        # temporal smoothness on keypoints
        if kps_pred_norm is not None and w_temp > 0:
            L_temp = temporal_consistency_loss(kps_pred_norm)
            total_loss = total_loss + w_temp * L_temp
            parts["temp"] = L_temp

        # aggregated (monitor only)
        if "heatmaps_agg" in out and out["heatmaps_agg"] is not None:
            hm_agg = out["heatmaps_agg"].float()
            kps_agg = out["keypoints_agg"].float()
            hm_gt_agg = hm_gt.mean(dim=1)
            kps_gt_agg = kps_gt.mean(dim=1)
            parts["hm_agg"] = F.mse_loss(hm_agg, hm_gt_agg, reduction="mean")
            parts["kp_agg"] = F.smooth_l1_loss(kps_agg, kps_gt_agg, reduction="mean")

    return total_loss, parts, metrics

# =========================
# Train One Epoch
# =========================
def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, cfg, epoch, global_step, writer):
    model.train()
    max_norm = cfg["train"].get("grad_clip", 1.0)
    use_amp = bool(cfg["train"].get("amp", True))

    running = {}
    n_batches = len(loader)

    dev_type = "cuda" if device.startswith("cuda") else "cpu"
    autocast_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float16

    for it, batch in enumerate(loader):
        frames, kps_gt, gt_q, gt_r, vis, hm_gt, delta_Rt_gt, _, crop_xywh = batch
        frames, kps_gt, gt_q, gt_r, vis, hm_gt, delta_Rt_gt, crop_xywh = [
            x.to(device, non_blocking=True) for x in [frames, kps_gt, gt_q, gt_r, vis, hm_gt, delta_Rt_gt, crop_xywh]
        ]

        tau_now = get_tau_for_epoch(cfg, epoch)
        cfg.setdefault("loss", {})["tau_now"] = float(tau_now)
        cfg["loss"]["heatmap_target_power"] = float(get_heatmap_power_for_epoch(cfg, epoch))
        cfg["loss"]["w_hm"] = float(get_w_hm_for_epoch(cfg, epoch))

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=dev_type, dtype=autocast_dtype, enabled=use_amp):
            out = model(frames, visibility=vis)

        with torch.autocast(device_type=dev_type, enabled=False):
            total_loss, parts, metrics = compute_losses(out, cfg, kps_gt, hm_gt, gt_q, gt_r, vis, delta_Rt_gt, crop_xywh=crop_xywh)

        scaler.scale(total_loss).backward()
        if max_norm and max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()
        if scheduler: scheduler.step()

        global_step += 1

        # accumulate
        for k,v in [("total", total_loss)] + list(parts.items()):
            running[k] = running.get(k,0.0) + float(v.item())
        if "esa_score" in metrics:
            running["esa"] = running.get("esa",0.0) + metrics["esa_score"]

    # averages
    avg = {k: v / n_batches for k,v in running.items()}
    if is_main_process():
        log(
            f"Train Epoch {epoch}: "
            f"loss={avg.get('total',0):.4f} "
            f"| rotF={avg.get('rot_frame',0):.4f} | rotA={avg.get('rot_agg',0):.4f} "
            f"| trnF={avg.get('trn_frame',0):.4f} | trnA={avg.get('trn_agg',0):.4f} "
            f"| hmF={avg.get('hm',0):.4f} | hmA={avg.get('hm_agg',0):.4f} "
            f"| kpF={avg.get('kp',0):.4f} | kpA={avg.get('kp_agg',0):.4f} "
            f"| esa={avg.get('esa',0):.4f}"
        )
        writer.add_scalar('Train/Loss', avg.get("total",0), epoch)
        # per-frame pose
        writer.add_scalar('Train/Loss_Rot_Frame', avg.get("rot_frame",0), epoch)
        writer.add_scalar('Train/Loss_Trn_Frame', avg.get("trn_frame",0), epoch)
        # agg pose
        writer.add_scalar('Train/Loss_Rot_Agg', avg.get("rot_agg",0), epoch)
        writer.add_scalar('Train/Loss_Trn_Agg', avg.get("trn_agg",0), epoch)
        # per-frame hm/kp
        writer.add_scalar('Train/Loss_HM_Frame', avg.get("hm",0), epoch)
        if cfg["model"].get("use_residual", False):
            writer.add_scalar('Train/Loss_KP_Frame', avg.get("kp",0), epoch)
        # agg hm/kp
        writer.add_scalar('Train/Loss_HM_Agg', avg.get("hm_agg",0), epoch)
        if cfg["model"].get("use_residual", False):
            writer.add_scalar('Train/Loss_KP_Agg', avg.get("kp_agg",0), epoch)
        # extras
        writer.add_scalar('Train/ESA', avg.get("esa",0), epoch)
        writer.add_scalar("Train/Loss_Temp", avg.get("temp",0), epoch)
    return avg

# =========================
# Validation
# =========================
@torch.no_grad()
def validate(model, loader, device, cfg, epoch, writer, global_step):
    if is_dist_avail_and_initialized() and not is_main_process():
        return {}

    model.eval()
    running = {
        k: 0.0 for k in [
            "total","rot_frame","rot_agg","trn_frame","trn_agg",
            "hm","kp","esa","pck@5_frame","pck@10_frame","pck@5_agg","pck@10_agg"
        ]
    }
    n_batches = len(loader)
    output_mode = cfg["model"].get("output_mode", "both")

    for it, batch in enumerate(loader):
        frames, kps_gt, gt_q, gt_r, vis, hm_gt, delta_Rt_gt, _, crop_xywh = batch
        frames = frames.to(device, non_blocking=True)
        kps_gt = kps_gt.to(device, non_blocking=True)
        gt_q   = gt_q.to(device, non_blocking=True)
        gt_r   = gt_r.to(device, non_blocking=True)
        vis    = vis.to(device, non_blocking=True)
        hm_gt  = hm_gt.to(device, non_blocking=True)
        delta_Rt_gt = delta_Rt_gt.to(device, non_blocking=True)
        crop_xywh = crop_xywh.to(device, non_blocking=True)

        out = model(frames, visibility=vis)
        tau_now = get_tau_for_epoch(cfg, epoch)
        cfg_loss_over = dict(cfg.get("loss", {})); cfg_loss_over["tau"] = float(tau_now)
        cfg_over = dict(cfg); cfg_over["loss"] = cfg_loss_over

        val_loss, parts, metrics = compute_losses(
            out, cfg_over, kps_gt, hm_gt, gt_q, gt_r, vis, delta_Rt_gt, crop_xywh=crop_xywh
        )

        # --- PCK evaluation ---
        if output_mode in ("kp_heatmap", "both"):
            img_hw = (int(cfg["data"]["input_size"]), int(cfg["data"]["input_size"]))

            # per-frame PCK
            if "heatmaps_frame" in out and out["heatmaps_frame"] is not None:
                kps_img_frame = upsample_argmax_kps(out["heatmaps_frame"], img_hw)  # (B,T,K,2)
                pck5F  = pck_from_img_px(kps_img_frame, kps_gt, img_hw, thresh_px=5.0,  vis=vis)
                pck10F = pck_from_img_px(kps_img_frame, kps_gt, img_hw, thresh_px=10.0, vis=vis)
                metrics["pck@5_frame"]  = pck5F
                metrics["pck@10_frame"] = pck10F

            # aggregated PCK
            if "heatmaps_agg" in out and out["heatmaps_agg"] is not None:
                kps_img_agg = upsample_argmax_kps(out["heatmaps_agg"], img_hw)   # (B,K,2)
                kps_gt_agg  = kps_gt.mean(dim=1)                                 # (B,K,2)
                vis_agg     = vis.mean(dim=1)                                    # (B,K)
                pck5A  = pck_from_img_px(kps_img_agg, kps_gt_agg, img_hw, thresh_px=5.0,  vis=vis_agg)
                pck10A = pck_from_img_px(kps_img_agg, kps_gt_agg, img_hw, thresh_px=10.0, vis=vis_agg)
                metrics["pck@5_agg"]  = pck5A
                metrics["pck@10_agg"] = pck10A

        global_step += 1

        running["total"] += float(val_loss.item())
        for key in ["rot_frame","rot_agg","trn_frame","trn_agg","hm"]:
            if key in parts:
                running[key] = running.get(key,0.0) + float(parts[key].item())
        if cfg["model"].get("use_residual", False) and "kp" in parts:
            running["kp"] = running.get("kp",0.0) + float(parts["kp"].item())
        for key in ["pck@5_frame","pck@10_frame","pck@5_agg","pck@10_agg"]:
            if key in metrics:
                running[key] += float(metrics[key])
        if "esa_score" in metrics:
            running["esa"] += metrics["esa_score"]

    # averages
    avg = {k: v / n_batches for k,v in running.items()}
    if is_main_process():
        log(
            f"Val Epoch {epoch}: "
            f"loss={avg['total']:.4f} | "
            f"rotF={avg.get('rot_frame',0):.4f} | rotA={avg.get('rot_agg',0):.4f} "
            f"| trnF={avg.get('trn_frame',0):.4f} | trnA={avg.get('trn_agg',0):.4f} "
            f"| hm={avg['hm']:.4f} | esa={avg['esa']:.4f}"
            + (f" | kp={avg['kp']:.4f}" if cfg['model'].get('use_residual', False) else "")
            + f" | pck5F={avg.get('pck@5_frame',0):.3f} | pck10F={avg.get('pck@10_frame',0):.3f} "
            + f"| pck5A={avg.get('pck@5_agg',0):.3f} | pck10A={avg.get('pck@10_agg',0):.3f}"
        )

        # TensorBoard logging
        writer.add_scalar('Val/Loss', avg["total"], epoch)
        writer.add_scalar('Val/Loss_Rot_Frame', avg.get("rot_frame",0), epoch)
        writer.add_scalar('Val/Loss_Trn_Frame', avg.get("trn_frame",0), epoch)
        writer.add_scalar('Val/Loss_Rot_Agg', avg.get("rot_agg",0), epoch)
        writer.add_scalar('Val/Loss_Trn_Agg', avg.get("trn_agg",0), epoch)
        writer.add_scalar('Val/Loss_HM', avg["hm"], epoch)
        if cfg["model"].get("use_residual", False):
            writer.add_scalar('Val/Loss_KP', avg["kp"], epoch)

        # PCK frame
        writer.add_scalar('Val/PCK@5_Frame', avg.get("pck@5_frame",0), epoch)
        writer.add_scalar('Val/PCK@10_Frame', avg.get("pck@10_frame",0), epoch)
        # PCK agg
        writer.add_scalar('Val/PCK@5_Agg', avg.get("pck@5_agg",0), epoch)
        writer.add_scalar('Val/PCK@10_Agg', avg.get("pck@10_agg",0), epoch)

    return avg
# ----------------------------
# Orchestration
# ----------------------------

def run_training(cfg, logger=log):
    setup_distributed(backend=cfg["train"].get("ddp_backend", "nccl"))
    device = cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if device.startswith("cuda"):
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"

    seed = int(cfg["train"].get("seed", 42)) + get_rank()
    seed_everything(seed)

    train_loader, val_loader, train_sampler, val_sampler = make_dataloaders(cfg)

    model = build_model(cfg).to(device)
    freeze_all_but_dualpath_and_heads(model)

    if get_world_size() > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    # sanity check debugging----------------------------------
    # real = model.module if isinstance(model, DDP) else model
    # for n, p in real.named_parameters():
    #     if p.requires_grad:
    #         print(f"[trainable] {n} {p.shape}")

    # --------------------------------------------------------

    scaler = GradScaler(device="cuda" if torch.cuda.is_available() else "cpu",
                        enabled=cfg["train"].get("amp", True))

    optim_name = cfg["train"].get("optimizer", "adamw").lower()
    lr = float(cfg["train"].get("lr", 3e-4))
    wd = float(cfg["train"].get("weight_decay", 0.05))
    head_lr_mult = float(cfg["train"].get("head_lr_mult", 5.0))
    block_lr_mult = float(cfg["train"].get("block_lr_mult", 0.5))

    param_groups = split_params_for_optimizer(model, base_lr=lr, weight_decay=wd, head_lr_mult=head_lr_mult, block_lr_mult=block_lr_mult)
    if optim_name == "adamw":
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))
    else:
        optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999))

    warmup_epochs = int(cfg["train"].get("warmup_epochs", 0))
    warmup_steps  = cfg["train"].get("warmup_steps", None)
    lr_min        = float(cfg["train"].get("lr_min", 1e-6))
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        epochs=cfg["train"]["epochs"],
        steps_per_epoch=len(train_loader),
        warmup_epochs=warmup_epochs,
        warmup_steps=warmup_steps,
        lr_min=lr_min,
    )

    best_metric = float("inf")
    out_dir = cfg["train"].get("out_dir", "./outputs")
    os.makedirs(out_dir, exist_ok=True)

    #tensorboard
    run_name = f"exp_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    logdir = os.path.join(cfg['train']['tb_logdir'], run_name)
    writer = SummaryWriter(log_dir=logdir)

    global_step = 0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        if isinstance(val_sampler, DistributedSampler):
            val_sampler.set_epoch(epoch)

        tr_stats = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, cfg, epoch, global_step, writer)
        cfg["_epoch_for_val"] = epoch

        if is_main_process():
            model.eval()
            val_stats = validate(model, val_loader, device, cfg, epoch, writer, global_step)
            cfg.pop("_epoch_for_val", None)

            # simple model selection: smallest val loss
            score = val_stats["total"]
            if score < best_metric:
                best_metric = score
                best_path = os.path.join(out_dir, f"best_epoch_{epoch}.pt")
                torch.save(
                    {"cfg": cfg, "epoch": epoch,
                     "model": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
                     "optimizer": optimizer.state_dict(),
                     "best_metric": best_metric},
                    best_path)
                logger(f"✓ Saved best checkpoint to {best_path} (score={best_metric:.4f})")

        if is_dist_avail_and_initialized():
            dist.barrier()

    if is_main_process():
        logger(f"Training done. Best score={best_metric:.4f}")
    cleanup_distributed()

# def run_training(cfg, logger=log):
#     # 1) distributed setup
#     setup_distributed(backend=cfg["train"].get("ddp_backend", "nccl"))

#     # 2) device per rank
#     device = cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
#     local_rank = int(os.environ.get("LOCAL_RANK", 0))
#     if device.startswith("cuda"):
#         torch.cuda.set_device(local_rank)
#         device = f"cuda:{local_rank}"
    
#     # 3) seed (optionally offset by rank)
#     seed = int(cfg["train"].get("seed", 42)) + get_rank()
#     seed_everything(seed)

#     # 4) data
#     train_loader, val_loader, train_sampler, val_sampler = make_dataloaders(cfg)

#     if is_dist_avail_and_initialized():
#         logger(f"[rank {get_rank()}] train_batches={len(train_loader)} "
#                f"val_batches={len(val_loader)} "
#                f"drop_last(train)={getattr(train_loader, 'drop_last', None)} "
#                f"drop_last(val)={getattr(val_loader, 'drop_last', None)}")
#         if isinstance(train_sampler, DistributedSampler):
#             logger(f"[rank {get_rank()}] train_sampler_len={len(train_sampler)}")
#         if isinstance(val_sampler, DistributedSampler):
#             logger(f"[rank {get_rank()}] val_sampler_len={len(val_sampler)}")

#     # 5) model
#     model = build_model(cfg).to(device)

#     freeze_all_but_dualpath_and_heads(model)

#     #Debugging
#     # for name, p in model.named_parameters():
#     #     if p.requires_grad:
#     #         print(f"[Debugging] Trainable: {name}")

#     # Print adapter trainability
#     if hasattr(model, "dp_adapter") and model.dp_adapter is not None:
#         print("[check] DualPathAdapter trainable:", any(p.requires_grad for p in model.dp_adapter.parameters()))

    

#     # Inspect kp_res_head presence
#     real = model  # not DDP yet here
#     if cfg["model"]["output_mode"] == "kp_heatmap":
#             if hasattr(real, "pose_head") and real.pose_head is not None:
#                 for p in real.pose_head.parameters():
#                     p.requires_grad = False

#             # If you temporarily are NOT using kp_res_head in the loss path,
#             # freeze it as well to avoid DDP unused-param trips.
#             use_kp_res = bool(cfg["model"].get("use_kp_residuals", False))
#             if not use_kp_res and hasattr(real, "kp_res_head") and real.kp_res_head is not None:
#                 for p in real.kp_res_head.parameters():
#                     p.requires_grad = False
#     if hasattr(real, "kp_res_head") and (real.kp_res_head is not None):
#         n_params = sum(p.numel() for p in real.kp_res_head.parameters())
#         n_train  = sum(p.numel() for p in real.kp_res_head.parameters() if p.requires_grad)
#         print(f"[check] kp_res_head params: total={n_params}, trainable={n_train}", flush=True)

#     # Zero-init kp residual head so softargmax path is the starting point
#     if getattr(model, "kp_res_head", None) is not None:
#         last = model.kp_res_head.mlp[-1]
#         if isinstance(last, nn.Linear):
#             nn.init.zeros_(last.weight)
#             if last.bias is not None:
#                 nn.init.zeros_(last.bias)

#     # ema config (supports both keys)
#     use_ema = cfg["train"].get("use_ema", cfg["train"].get("ema", False))
#     ema_decay = float(cfg["train"].get("ema_decay", 0.99))

#     # wrap in DDP if multi-GPU
#     if get_world_size() > 1:
#         model = DDP(
#             model,
#             device_ids=[local_rank],
#             output_device=local_rank,
#             broadcast_buffers=False,
#             find_unused_parameters=True,
#             gradient_as_bucket_view=False,
#         )
#     # AMP scaler with device argument
#     scaler = GradScaler(device="cuda" if torch.cuda.is_available() else "cpu",
#                         enabled=cfg["train"].get("amp", True))

#     if use_ema:
#         from torch.optim.swa_utils import AveragedModel
#         _track = model.module if isinstance(model, DDP) else model
#         ema_model = AveragedModel(_track, avg_fn=lambda ema_v, model_v, _: ema_decay*ema_v + (1.0-ema_decay)*model_v)
#         # tiny helper so train_one_epoch can find it
#         class _Wrap: pass
#         w = _Wrap(); w.update_parameters = ema_model.update_parameters
#         model._ema_hook = w
#     else:
#         ema_model = None

#     # log model size
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     logger(f"Model total parameters: {total_params:,} ({total_params/1e6:.2f}M) | "
#            f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

#     # Optimizer: param groups (higher LR for heads)
#     optim_name = cfg["train"].get("optimizer", "adamw").lower()
#     lr = float(cfg["train"].get("lr", 3e-4))  # this MUST be 3e-4
#     wd = float(cfg["train"].get("weight_decay", 0.05))
#     head_lr_mult = float(cfg["train"].get("head_lr_mult", 5.0))

#     param_groups = split_params_for_optimizer(model,
#                                           base_lr=lr,            # 3e-4
#                                           weight_decay=wd,
#                                           head_lr_mult=head_lr_mult)
    
#     print(f"[DEBUG] base_lr passed to optimizer: {lr}")

#     if optim_name == "adamw":
#         optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))
#     elif optim_name == "adam":
#         optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999))
#     else:
#         raise ValueError("optimizer must be adam or adamw")

#     for i, group in enumerate(optimizer.param_groups):
#         print(f"[DEBUG] Optim group {i} initial LR: {group['lr']} | params={sum(p.numel() for p in group['params'])}")
#     # per-step warmup  + cosine scheduler
#     warmup_epochs = int(cfg["train"].get("warmup_epochs", 0))
#     warmup_steps  = cfg["train"].get("warmup_steps", None)
#     lr_min        = float(cfg["train"].get("lr_min", 1e-6))
#     scheduler = build_warmup_cosine_scheduler(
#         optimizer,
#         epochs=cfg["train"]["epochs"],
#         steps_per_epoch=len(train_loader),
#         warmup_epochs=warmup_epochs,
#         warmup_steps=warmup_steps,
#         lr_min=lr_min,
#     )

#     # debuggin 
#     print("\n[SANITY CHECK] Scheduler preview (initial LRs):")
#     for i, group in enumerate(optimizer.param_groups):
#         print(f"  Group {i}: initial LR = {group['lr']}")
#     # check optimizer
#     pg_sizes = [sum(p.numel() for p in g["params"]) for g in optimizer.param_groups]
#     if is_main_process():
#         log(f"[optim] groups (lr_base, lr_heads): sizes={pg_sizes}, "
#             f"lrs={[g['lr'] for g in optimizer.param_groups]}")

#     # 6) training loop
#     best_metric = float("inf")
#     best_path = None
#     out_dir = cfg["train"].get("out_dir", "./outputs")
#     os.makedirs(out_dir, exist_ok=True)
#     global_step = 0

#     # debug through what is being trained and what is not 
#     # for name, p in model.named_parameters():
#     #     if p.requires_grad:
#     #         print(f"{name}: {p.shape}")

#     # ----------------------------------------------------
#     for i, g in enumerate(optimizer.param_groups):
#         log(f"[DEBUG] group {i} lr={g['lr']}, params={sum(p.numel() for p in g['params'])}")

#     for epoch in range(1, cfg["train"]["epochs"] + 1):
#         if isinstance(train_sampler, DistributedSampler):
#             train_sampler.set_epoch(epoch)
#         if isinstance(val_sampler, DistributedSampler):
#             val_sampler.set_epoch(epoch)

#         tr_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, cfg, epoch, global_step)

        
#         cfg["_epoch_for_val"] = epoch

#         # only rank 0 runs validation & checkpoint logic
#         if is_main_process():
#             if use_ema:
#                 # Swap to EMA weights for eval
#                 real = model.module if isinstance(model, DDP) else model
#                 ema_state = ema_model.state_dict()
#                 real_state = real.state_dict()
#                 real.load_state_dict(ema_state, strict=False)

#             val_metrics = validate(model, val_loader, device, cfg, epoch)
#             cfg.pop("_epoch_for_val", None)

#             if use_ema:
#                 # restore original (non-EMA) weights
#                 real.load_state_dict(real_state, strict=False)

#             sel = cfg["train"].get("val_metric", None)
#             if sel is not None:
#                 score = val_metrics.get(sel, float("inf"))
#             else:
#                 if cfg["model"]["output_mode"] == "quat":
#                     score = val_metrics.get("rot_mae_deg", 0.0) + val_metrics.get("trans_mae", 0.0)
#                 elif cfg["model"]["output_mode"] == "kp_heatmap":
#                     score = val_metrics.get("val_loss_hm", float("inf")) \
#                         + 0.1 * val_metrics.get("rot_mae_deg", 0.0) \
#                         + val_metrics.get("trans_mae", 0.0)
#                     # score = val_metrics.get("val_loss_hm", float("inf"))
#                 else:  # both
#                     score = val_metrics.get("kp_mae_px", 0.0) \
#                             + 0.1 * val_metrics.get("rot_mae_deg", 0.0) \
#                             + val_metrics.get("trans_mae", 0.0)

#             if score < best_metric:
#                 best_metric = score
#                 best_path = os.path.join(out_dir, f"best_{cfg['model']['output_mode']}.pt")
#                 torch.save(
#                     {
#                         "cfg": cfg,
#                         "epoch": epoch,
#                         "model": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
#                         "optimizer": optimizer.state_dict(),
#                         "best_metric": best_metric,
#                     },
#                     best_path
#                 )
#                 logger(f"✓ Saved best checkpoint to {best_path} (score={best_metric:.4f})")

#             if epoch % cfg["train"].get("ckpt_interval", 5) == 0:
#                 p = os.path.join(out_dir, f"epoch_{epoch}.pt")
#                 torch.save(
#                     {"cfg": cfg, "epoch": epoch,
#                      "model": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict())},
#                     p
#                 )
#                 logger(f"Checkpoint saved: {p}")

#         # sync ranks before next epoch
#         if is_dist_avail_and_initialized():
#             dist.barrier()

#     if is_main_process():
#         logger(f"Training done. Best score={best_metric:.4f}, best path={best_path}")

#     # cleanup
#     cleanup_distributed()



# train_val.py
# import os
# import math
# import time
# import warnings
# from typing import Dict, Tuple, Optional, List, Any
# import random
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torch.amp import GradScaler
# import numpy as np
# import datetime

# # plot in tensorboard
# from torch.utils.tensorboard import SummaryWriter
# from torch.nn import functional as F
# from misc import extract_refined_keypoints_from_heatmaps
# from evaluate import evaluate_model

# # ---- Silence the albumentations ShiftScaleRotate warning (true fix is in transforms.py) ----
# warnings.filterwarnings(
#     "ignore",
#     message="ShiftScaleRotate is a special case of Affine transform. Please use Affine transform instead."
# )

# # ---- DDP imports ----
# from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.distributed as dist

# from model import SpacecraftPoseDualPath
# import cv2

# # ---- dataset imports  ----
# from dataset import SPADESDataset, SPARKDataset
# from transforms import build_transforms
# from utils import build_warmup_cosine_scheduler, load_camera_intrinsics, freeze_all_but_dualpath_and_heads
# from utils import project_keypoints, debug_check_heatmap_alignment, debug_check_pred_alignment, get_heatmap_power_for_epoch, get_tau_for_epoch
# from utils import cosine_schedule, temporal_consistency_loss, quat_inverse, quat_mul, quat_relative, scale_translation_to_full_image
# from torch.utils.data import Subset
# from utils import collate_variable_length, temporal_heatmap_loss, collate_temporal_chunks, draw_kps_on_frame, get_w_token_for_epoch
# from utils import hm_to_crop_px, crop_to_hm_px

# from losses import heatmap_to_com, temporal_contrastive_loss, smooth_l1_vis_weighted
# from losses import token_spatial_alignment_loss, temporal_smoothness_l2

# # ---- perf toggles (safe even on CPU; they only take effect on CUDA) ----
# torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True


# def is_dist_avail_and_initialized() -> bool:
#     return dist.is_available() and dist.is_initialized()

# def get_rank() -> int:
#     return dist.get_rank() if is_dist_avail_and_initialized() else 0

# def get_world_size() -> int:
#     return dist.get_world_size() if is_dist_avail_and_initialized() else 1

# def is_main_process() -> bool:
#     return get_rank() == 0


# def setup_distributed(backend="nccl"):
#     """
#     Initializes default process group using env:// (works with torchrun + Slurm).
#     Sets current CUDA device from LOCAL_RANK.
#     """
#     if is_dist_avail_and_initialized():
#         return  # already done
#     if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
#         dist.init_process_group(backend=backend, init_method="env://")
#         local_rank = int(os.environ.get("LOCAL_RANK", 0))
#         torch.cuda.set_device(local_rank)
#     else:
#         # Single-process fallback (no DDP)
#         pass


# # ---------- Optimizer Helper (head LR multiplier) -------
# def split_params_for_optimizer(
#     model,
#     base_lr,
#     weight_decay=0.05,
#     head_lr_mult=5.0,
#     block_lr_mult=0.5,
# ):
#     real = model.module if hasattr(model, "module") else model

#     adapter_params, head_params, block_params = [], [], []

#     def no_weight_decay(n):
#         return n.endswith("bias") or "bn" in n.lower() or "norm" in n.lower()

#     for n, p in real.named_parameters():
#         if not p.requires_grad:
#             continue

#         # Adapters
#         if "adapter" in n or "dualpath" in n:
#             adapter_params.append((n, p))

#         # Heads
#         elif any(k in n for k in ["hm_head", "keypoint_head", "pose_head", "finer_cnn", "pnp", "kp_res_head"]):
#             head_params.append((n, p))

#         # Patch embed / norm / last blocks
#         elif any(k in n for k in ["patch_embed", "norm", "blocks."]):
#             block_params.append((n, p))

#     def make_split(named_params, lr):
#         decay, no_decay = [], []
#         for n, p in named_params:
#             (no_decay if no_weight_decay(n) else decay).append(p)
#         groups = []
#         if decay:
#             groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
#         if no_decay:
#             groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
#         return groups

#     groups = []
#     groups += make_split(adapter_params, base_lr)
#     groups += make_split(block_params, base_lr * block_lr_mult)
#     groups += make_split(head_params, base_lr * head_lr_mult)

#     # Debug info
#     print(f"[opt] LR groups:")
#     for g in groups:
#         n_params = sum(p.numel() for p in g["params"])
#         print(f"  - {n_params} params @ lr={g['lr']}, wd={g['weight_decay']}")

#     return groups

# # ---------- Subset helpers ----------
# def _subset_indices(n_total: int, n_take: int, mode: str = "first", seed: int = 0) -> List[int]:
#     n_take = int(min(max(0, n_take), n_total))
#     if n_take <= 0 or n_take == n_total:
#         return list(range(n_total))
#     if mode == "first":
#         return list(range(n_take))
#     # random but deterministic
#     import numpy as _np
#     rng = _np.random.RandomState(seed)
#     idx = _np.arange(n_total)
#     rng.shuffle(idx)
#     return idx[:n_take].tolist()

# def _broadcast_indices(idxs: List[int]) -> List[int]:
#     if not is_dist_avail_and_initialized():
#         return idxs
#     obj_list = [idxs]
#     dist.broadcast_object_list(obj_list, src=0)
#     return obj_list[0]
# # ------------------------------------

# def cleanup_distributed():
#     if is_dist_avail_and_initialized():
#         dist.barrier()
#         dist.destroy_process_group()

# def seed_everything(seed: int):
#     random.seed(seed); np = __import__("numpy"); np.random.seed(seed)
#     torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
#     # Determinism off (we want speed); set if you need exact reproducibility:
#     torch.backends.cudnn.deterministic = False

# # ---- loss suite (final choice we agreed on) ----
# from losses import (
#     geodesic_quat_loss, translation_l1,
#     kp_heatmap_loss_combo,
#     heatmap_kl_div_from_logits, softargmax_2d_from_logits,
#     heatmap_kl_div_from_logits_masked,
# )

# def log(s: str):
#     print(s, flush=True)

# def enable_tf32():
#     """Optional convenience; harmless to call more than once."""
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True

# def _sync_and_trim_dataset(ds, name: str):
#     """Make sure all ranks see identical dataset length by trimming to the global MIN length."""
#     local_len = len(ds)
#     if is_dist_avail_and_initialized():
#         lens = [None for _ in range(get_world_size())]
#         dist.all_gather_object(lens, local_len)
#         gmin, gmax = min(lens), max(lens)
#         if is_main_process() and gmin != gmax:
#             log(f"[WARN] {name} length mismatch across ranks: {lens} -> trimming all to {gmin}")
#         if local_len != gmin:
#             from torch.utils.data import Subset
#             ds = Subset(ds, range(gmin))
#     return ds


# # --------------------------------
# # Scheduler helper for Heatmap
# # --------------------------------

# def get_w_hm_for_epoch(cfg, epoch: int) -> float:
#     sched = cfg.get("loss", {}).get("w_hm_schedule", None)
#     base = float(cfg.get("loss", {}).get("w_hm", 1.0))
#     if not sched:
#         return base
#     return cosine_schedule(
#         start=sched.get("start", base),
#         end=sched.get("end", base),
#         start_epoch=sched.get("start_epoch", 0),
#         end_epoch=sched.get("end_epoch", 0),
#         cur_epoch=epoch,
#     )

# # ----------------------------
# # Model factory
# # ----------------------------
# def build_model(cfg) -> nn.Module:
#     common = dict(
#         aggregator=cfg["model"].get("aggregator", "attn"),
#         ckpt_path=cfg["model"]["backbone_ckpt"],
#         d_side=cfg["model"].get("d_side", 0),
#         heatmap_out=tuple(cfg["model"].get("heatmap_out", [56, 56])),


#         # camera intrinsics for PnP (if used)
#         cam_params = load_camera_intrinsics(cfg["data"]["cam_intrinsics_json"]) if cfg["model"].get("use_pnp", False) else None,
        

#         # NEW: heatmap head choice + params
#         hm_head=cfg["model"].get("hm_head", "film"),
#         hm_mid_c=cfg["model"].get("hm_mid_c", 128),
#         hm_use_film=cfg["model"].get("hm_use_film", False),
#         use_kp_residual=cfg["model"].get("use_residual", False),

#         # use_kp_residuals=cfg["model"].get("use_kp_residuals", False),
#         cfg = cfg,
#     )
#     return SpacecraftPoseDualPath(**common)  # dual_path is a variant of transformer with dual-path adapter

# # ----------------------------
# # Dataloaders
# # ----------------------------
# def make_dataloaders(cfg):
#     ds_name = cfg["data"].get("name", "").lower()

#     # Build the transform separately
#     # Build the crop + full-frame transforms separately
#     tfm_crop = build_transforms(
#         # aug_mode=cfg["train"].get("aug_mode", "kp_geo"),
#         # allow_tiny_crop=bool(cfg["train"].get("aug_allow_tiny_crop", False)),
#         img_size=int(cfg["data"]["input_size"]),
#     )
#     tfm_full = build_transforms(
#         # aug_mode=cfg["train"].get("aug_mode_full", "photometric"),
#         # allow_tiny_crop=False,
#         img_size=int(cfg["data"]["input_size"]),
#     )
#     # ONLY include kwargs dataset __init__ supports
#     base_kwargs = dict(
#         gt_folder=cfg["data"]["root"],
#         short_mem=cfg["data"]["short_mem"],
#         input_size=cfg["data"]["input_size"],
#         sampling_rate=cfg["data"]["sampling_rate"],
#         clip_stride = cfg["data"].get("clip_stride"),
#         crop=cfg["data"].get("crop", True),
#         crop_prob = cfg['data'].get('crop_prob', 0.0),
#         eval_use_crop = cfg['data'].get('eval_use_crop', False),
#         gaussian_sigma_px=cfg["data"].get("gaussian_sigma_px", 2.0),
#         heatmap_out=tuple(cfg["model"].get("heatmap_out", [56, 56])),
#     )

#     if ds_name == "spades":
#         train_set = SPADESDataset(mode="train", spatial_transforms=tfm_crop, spatial_transforms_full=tfm_full, **base_kwargs)
#         # val_mode  = "val" if os.path.exists(os.path.join(cfg["data"]["root"], "val.json")) else "test"
#         val_set   = SPADESDataset(mode="val", spatial_transforms=None, spatial_transforms_full=None, **base_kwargs)

#         test_set = SPADESDataset(mode="test", spatial_transforms=None, spatial_transforms_full=None, **base_kwargs)

#     elif ds_name == "spark":
#         train_set = SPARKDataset(mode="train", spatial_transforms=tfm_crop, spatial_transforms_full=tfm_full, **base_kwargs)
#         val_set   = SPARKDataset(mode="val",   spatial_transforms=None, spatial_transforms_full=None, **base_kwargs)

#     else:
#         raise ValueError("data.name must be 'spades' or 'spark'")

#     # ==== Optional debugging subsets ====
#     train_take = int(cfg["data"].get("train_subset", 0) or 0)
#     val_take   = int(cfg["data"].get("val_subset", 0) or 0)
#     sub_mode   = str(cfg["data"].get("subset_mode", "first")).lower()
#     sub_seed   = int(cfg["data"].get("subset_seed", 0) or 0)

#     if train_take > 0:
#         if is_main_process():
#             idxs = _subset_indices(len(train_set), train_take, mode=sub_mode, seed=sub_seed)
#         else:
#             idxs = None
#         # broadcast to all ranks (including single-process no-op)
#         idxs = _broadcast_indices(idxs if idxs is not None else [])
#         train_set = Subset(train_set, idxs)
#         if is_main_process():
#             log(f"[Subset] train -> {len(idxs)} samples (mode={sub_mode}, seed={sub_seed})")

#     if val_take > 0:
#         if is_main_process():
#             idxs = _subset_indices(len(val_set), val_take, mode=sub_mode, seed=sub_seed + 1)
#         else:
#             idxs = None
#         idxs = _broadcast_indices(idxs if idxs is not None else [])
#         val_set = Subset(val_set, idxs)
#         if is_main_process():
#             log(f"[Subset] val   -> {len(idxs)} samples (mode={sub_mode}, seed={sub_seed + 1})")

#     # ==== NEW: force identical lengths across ranks ====
#     train_set = _sync_and_trim_dataset(train_set, "train_set")
#     val_set   = _sync_and_trim_dataset(val_set,   "val_set")

#     # ==== Distributed samplers ====
#     world = get_world_size()
#     rank  = get_rank()
#     train_sampler = DistributedSampler(
#         train_set, num_replicas=world, rank=rank, shuffle=True, drop_last=True
#     ) if world > 1 else None

#     val_sampler = DistributedSampler(
#         val_set, num_replicas=world, rank=rank, shuffle=False, drop_last=False
#     ) if world > 1 else None

#     nw = int(cfg["train"].get("num_workers", 4))
#     bs = int(cfg["train"]["batch_size"])
#     vbs = int(cfg["train"].get("val_batch_size", bs))

#     train_loader = DataLoader(
#         train_set,
#         batch_size=bs,
#         shuffle=(train_sampler is None),
#         num_workers=nw,
#         pin_memory=True,
#         drop_last=True,                 # keep True for strict equal steps
#         sampler=train_sampler,
#         persistent_workers=(nw > 0),
#     )
#     val_loader = DataLoader(
#         val_set,
#         batch_size=vbs,
#         shuffle=False,
#         num_workers=nw,
#         pin_memory=True,
#         drop_last=False,
#         sampler=val_sampler,
#         persistent_workers=(nw > 0),
#     )

#     test_loader = DataLoader(
#         test_set,
#         batch_size=1,
#         shuffle=False,
#         num_workers=0,
#         collate_fn=lambda b: collate_temporal_chunks(
#             b, cfg["data"]["test_T_chunk"]
#         )
#         )

#     # ==== Sanity checks (visible in logs) ====
#     if is_dist_avail_and_initialized():
#         tb = len(train_loader); vb = len(val_loader)
#         tsl = len(train_sampler) if train_sampler is not None else len(train_set)
#         vsl = len(val_sampler)   if val_sampler   is not None else len(val_set)

#         all_tb = [None]*world; dist.all_gather_object(all_tb, tb)
#         all_vb = [None]*world; dist.all_gather_object(all_vb, vb)
#         all_tsl = [None]*world; dist.all_gather_object(all_tsl, tsl)
#         all_vsl = [None]*world; dist.all_gather_object(all_vsl, vsl)

#         if is_main_process():
#             log(f"[check] train_batches per rank: {all_tb}")
#             log(f"[check] val_batches   per rank: {all_vb}")
#             log(f"[check] train_sampler_len per rank: {all_tsl}")
#             log(f"[check] val_sampler_len   per rank: {all_vsl}")

#         # hard assert: they must match now
#         if len(set(all_tb)) != 1 or len(set(all_tsl)) != 1:
#             raise RuntimeError(f"Train loader/sampler mismatch across ranks: "
#                                f"batches={all_tb}, sampler_len={all_tsl}")

#     return train_loader, val_loader, train_sampler, val_sampler, test_loader


# # helper formatters
# def fmt(x):
#     return "-" if x is None else f"{(x.item() if torch.is_tensor(x) else float(x)):.3f}"
# def fval(x):
#     return None if x is None else (x.item() if torch.is_tensor(x) else float(x))


# # =========================
# # Compute Losses (Updated for HybridTinyPoseHead)
# # =========================
# def compute_losses(
#     out: Dict[str, torch.Tensor],
#     cfg: Dict[str, Any],
#     kps_gt: torch.Tensor,    # (B,T,K,2) in [-1,1]
#     hm_gt: torch.Tensor,     # (B,T,K,H,W)
#     vis: torch.Tensor,       # (B,T,K) 0/1
#     phase: str = "Train",
# ):
#     tau = float(cfg["loss"].get("tau", 1.0))

#     # weights
#     w_hm_frame = float(cfg["loss"].get("w_hm_frame", 1.0))
#     w_hm_agg   = float(cfg["loss"].get("w_hm_agg",   cfg["loss"].get("w_hm", 0.0)))
#     w_token    = float(cfg["loss"].get("w_token", 0.0))
#     w_temp     = float(cfg["loss"].get("w_temp", 0.0))
#     target_pow = float(cfg["loss"].get("heatmap_target_power", 1.0))
#     w_kp = float(cfg["loss"].get("w_kp", 1.0))

#     hm_logits = out["heatmaps"]  # (B,T,K,H,W)
#     B, T, K, Hm, Wm = hm_logits.shape

#     total_loss = hm_logits.new_zeros(())
#     parts: Dict[str, torch.Tensor] = {}

#     # -----------------------------------
#     # 1) Heatmap KL per-frame (masked)
#     # -----------------------------------
#     L_hm_frame = heatmap_kl_div_from_logits_masked(
#         hm_logits=hm_logits,
#         hm_target=hm_gt,
#         vis=vis,
#         tau=tau,
#         target_power=target_pow,
#     )
#     total_loss += w_hm_frame * L_hm_frame
#     parts["hm_frame"] = L_hm_frame

#     # -----------------------------------
#     # 2) Heatmap KL aggregated over time
#     # -----------------------------------
#     hm_logits_agg = hm_logits.mean(dim=1)        # (B,K,H,W)
#     hm_gt_agg     = hm_gt.mean(dim=1)            # (B,K,H,W)
#     vis_agg       = (vis.sum(dim=1) > 0).float() # (B,K)

#     L_hm_agg = heatmap_kl_div_from_logits_masked(
#         hm_logits=hm_logits_agg.unsqueeze(1),    # (B,1,K,H,W)
#         hm_target=hm_gt_agg.unsqueeze(1),
#         vis=vis_agg.unsqueeze(1),
#         tau=tau,
#         target_power=target_pow,
#     )
#     total_loss += w_hm_agg * L_hm_agg
#     parts["hm_agg"] = L_hm_agg

#     # -----------------------------------
#     # 3) Token spatial alignment loss
#     # -----------------------------------
#     if w_token > 0 and "patch_tokens_prenorm" in out:
#         L_token = token_spatial_alignment_loss(
#             patch_tokens=out["patch_tokens_prenorm"],  # (B,T,P,D)
#             kps_gt_norm=kps_gt,                         # (B,T,K,2)
#             vis=vis,                                    # (B,T,K)
#             temperature=0.07,
#         )
#         total_loss += w_token * L_token
#         parts["token_spatial"] = L_token

#     # ======================================================
#     # 3. Differentiable Keypoint Loss (optional)
#     # ======================================================
#     if w_kp > 0:
#         # soft-argmax → pixel coords
#         kps_px = softargmax_2d_from_logits(hm_logits, tau=tau)  # (B,T,K,2)

#         # pixel → normalized [-1,1]
#         x_norm = 2.0 * (kps_px[..., 0] / (Wm - 1.0)) - 1.0
#         y_norm = 2.0 * (kps_px[..., 1] / (Hm - 1.0)) - 1.0
#         kps_pred_norm = torch.stack([x_norm, y_norm], dim=-1)

#         L_kp = smooth_l1_vis_weighted(
#             kps_pred_norm,
#             kps_gt,
#             vis,
#         )

#         total_loss += w_kp * L_kp
#         parts["kp_frame"] = L_kp
#     else:
#         kps_pred_norm = None

#     # -----------------------------------
#     # 4) Temporal smoothness on COM
#     # -----------------------------------
#     if w_temp > 0 and phase == "Train":
#         com_px = heatmap_to_com(hm_logits, tau=tau)  # (B,T,K,2) in px
#         x = 2.0 * (com_px[..., 0] / (Wm - 1.0)) - 1.0
#         y = 2.0 * (com_px[..., 1] / (Hm - 1.0)) - 1.0
#         com_norm = torch.stack([x, y], dim=-1)

#         L_temp_raw = temporal_smoothness_l2(com_norm, vis)
#         total_loss += w_temp * L_temp_raw
#         parts["temp_smooth_raw"] = L_temp_raw
#         parts["temp_smooth"]     = w_temp * L_temp_raw

#     return total_loss, parts

# # =========================
# # Train One Epoch
# # =========================
# def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, cfg, epoch, global_step, writer):
#     phase = "Train"
#     model.train()
#     max_norm = cfg["train"].get("grad_clip", 1.0)
#     use_amp = bool(cfg["train"].get("amp", True))

#     # running = {}
#     running = {"total": 0.0}
#     n_batches = len(loader)

#     dev_type = "cuda" if device.startswith("cuda") else "cpu"
#     autocast_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float16

#     for it, batch in enumerate(loader):
#         frames, kps_gt, gt_q, gt_r, vis, hm_gt, delta_Rt_gt, _, crop_xywh = batch
#         frames, kps_gt, gt_q, gt_r, vis, hm_gt, delta_Rt_gt, crop_xywh = [
#             x.to(device, non_blocking=True) for x in [frames, kps_gt, gt_q, gt_r, vis, hm_gt, delta_Rt_gt, crop_xywh]
#         ]

#         tau_now = get_tau_for_epoch(cfg, epoch)
#         tau_now = max(get_tau_for_epoch(cfg, epoch), 0.05)  # avoid very sharp maps
#         cfg["loss"]["tau_now"] = tau_now
#         cfg["loss"]["heatmap_target_power"] = float(get_heatmap_power_for_epoch(cfg, epoch))
#         cfg["loss"]["w_hm_frame"] = float(get_w_hm_for_epoch(cfg, epoch))
#         cfg["loss"]["w_hm_agg"] = float(get_w_hm_for_epoch(cfg, epoch))
#         cfg["loss"]["w_token"] = float(get_w_token_for_epoch(cfg, epoch))
#         # cfg["loss"]["w_hm"] = min(cfg["loss"].get("w_hm", 1.0), 2.0)  # avoid oversized weights

#         optimizer.zero_grad(set_to_none=True)

#         with torch.autocast(device_type=dev_type, dtype=autocast_dtype, enabled=use_amp):
#             out = model(frames, visibility=vis)

#         with torch.autocast(device_type=dev_type, enabled=False):
#             total_loss, parts = compute_losses(
#                                         out, cfg,
#                                         kps_gt=kps_gt,
#                                         hm_gt=hm_gt,
#                                         vis=vis,
#                                         phase="Train",
#                                     )

#         # Debug visualization for GT vs predicted heatmaps
#         # Create debug output
#         # if is_main_process() and epoch % 1 == 0 and it == 0:
#         #     os.makedirs("debug_vis", exist_ok=True)

#         #     B, T, K, Hm, Wm = out["heatmaps"].shape
#         #     frames_np = frames[0].detach().cpu().numpy()  # (T,3,H,W)
#         #     heatmaps_pred = out["heatmaps"][0].detach().cpu().to(torch.float32)  # (T,K,H,W)
#         #     heatmaps_gt = hm_gt[0].detach().cpu()  # (T,K,H,W)
#         #     kps_gt_np = kps_gt[0].detach().cpu().numpy()  # (T,K,2)

#         #     for t in range(T):
#         #         frame = frames_np[t].transpose(1, 2, 0)  # (H,W,3)
#         #         frame = (frame * 255).astype(np.uint8).copy()

#         #         # GT Keypoints
#         #         img_overlay = draw_kps_on_frame(frame.copy(), kps_gt_np[t], radius=2, color=(0,255,0))  # green

#         #         # Predicted Keypoints
#         #         kps_pred_np = kps_pred[0, t].detach().cpu().numpy()  # (K,2)
#         #         img_overlay = draw_kps_on_frame(img_overlay, kps_pred_np, radius=2, color=(0,0,255))  # red

#         #         # GT Heatmap
#         #         hm_gt_t = heatmaps_gt[t].numpy()
#         #         hm_gt_vis = (np.mean(hm_gt_t, axis=0) - 0).clip(min=0)
#         #         hm_gt_vis = (hm_gt_vis / (hm_gt_vis.max() + 1e-6) * 255).astype(np.uint8)
#         #         hm_gt_color = cv2.applyColorMap(hm_gt_vis, cv2.COLORMAP_JET)
#         #         hm_gt_color = cv2.resize(hm_gt_color, (frame.shape[1], frame.shape[0]))
#         #         img_gt_hm = cv2.addWeighted(frame.copy(), 0.6, hm_gt_color, 0.4, 0)

#         #         # Pred Heatmap
#         #         hm_pred_t = heatmaps_pred[t].numpy()
#         #         hm_pred_vis = (np.mean(hm_pred_t, axis=0) - 0).clip(min=0)
#         #         hm_pred_vis = (hm_pred_vis / (hm_pred_vis.max() + 1e-6) * 255).astype(np.uint8)
#         #         hm_pred_color = cv2.applyColorMap(hm_pred_vis, cv2.COLORMAP_JET)
#         #         hm_pred_color = cv2.resize(hm_pred_color, (frame.shape[1], frame.shape[0]))
#         #         img_pred_hm = cv2.addWeighted(frame.copy(), 0.6, hm_pred_color, 0.4, 0)

#         #         # Save everything
#         #         cv2.imwrite(f"./debug_vis/epoch{epoch}_t{t:02d}_overlay.png", img_overlay)
#         #         cv2.imwrite(f"./debug_vis/epoch{epoch}_t{t:02d}_gt_heatmap.png", img_gt_hm)
#         #         cv2.imwrite(f"./debug_vis/epoch{epoch}_t{t:02d}_pred_heatmap.png", img_pred_hm)

#         #     exit()

#         scaler.scale(total_loss).backward()
#         if max_norm and max_norm > 0:
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
#         scaler.step(optimizer)
#         scaler.update()
#         if scheduler: scheduler.step()

#         global_step += 1

#         # accumulate
#         for k,v in [("total", total_loss)] + list(parts.items()):
#             val = v.item() if torch.is_tensor(v) else float(v)
#             if math.isnan(val) or math.isinf(val):
#                 print(f"[WARN] Skipping {k} = {val} (NaN or Inf)")
#                 continue
#             running[k] = running.get(k, 0.0) + val
#         for key in [
#             "rot_frame","rot_agg","trn_frame","trn_agg",
#             "hm_frame","hm_agg","kp_frame","kp_agg", "temp_smooth",
#         ]:
#             if key in parts:
#                 running[key] = running.get(key, 0.0) + float(parts[key].item())

#         # if "esa_score" in metrics:
#         #     running["esa"] = running.get("esa", 0.0) + metrics["esa_score"]

#     # averages
#     avg = {k: v / n_batches for k,v in running.items()}
#     if is_main_process():
#         log(
#             f"{phase} Epoch {epoch}: "
#             f"loss={avg['total']:.4f} "
#             # f"| rotF={avg.get('rot_frame',0):.4f} | rotA={avg.get('rot_agg',0):.4f} "
#             # f"| trnF={avg.get('trn_frame',0):.4f} | trnA={avg.get('trn_agg',0):.4f} "
#             f"| hmF={avg.get('hm_frame',0):.4f} | hmA={avg.get('hm_agg',0):.4f} "
#             f"| kpF={avg.get('kp_frame',0):.4f} | kpA={avg.get('kp_agg',0):.4f} "
#             f"| temp_loss_raw={avg.get('temp_smooth_raw',0):.4f}"
#             f"| temp_loss_weighted={avg.get('temp_smooth',0):.4f}"
#             f"| token={avg.get('token_spatial',0):.4f} "
#             # f"| esa={avg.get('esa',0):.4f}"
#         )
#         writer.add_scalar(f'{phase}/Loss', avg.get("total",0), epoch)
#         # writer.add_scalar(f'{phase}/Loss_Rot_Frame', avg.get("rot_frame",0), epoch)
#         # writer.add_scalar(f'{phase}/Loss_Rot_Agg',   avg.get("rot_agg",0), epoch)
#         # writer.add_scalar(f'{phase}/Loss_Trn_Frame', avg.get("trn_frame",0), epoch)
#         # writer.add_scalar(f'{phase}/Loss_Trn_Agg',   avg.get("trn_agg",0), epoch)
#         writer.add_scalar(f'{phase}/Loss_HM_Frame',  avg.get("hm_frame",0), epoch)
#         writer.add_scalar(f'{phase}/Loss_HM_Agg',    avg.get("hm_agg",0), epoch)
#         writer.add_scalar(f'{phase}/Loss_KP_Frame',  avg.get("kp_frame",0), epoch)
#         writer.add_scalar(f'{phase}/Loss_KP_Agg',    avg.get("kp_agg",0), epoch)
#         writer.add_scalar(f'{phase}/Loss_Temp',      avg.get("temp_smooth",0), epoch)
#         writer.add_scalar("Train/w_token", cfg["loss"]["w_token"], epoch)
#         # writer.add_scalar(f'{phase}/ESA',            avg.get("esa",0), epoch)
#     return avg

# # =========================
# # Validation
# # =========================
# @torch.no_grad()
# def validate(model, loader, device, cfg, epoch, writer, global_step):
#     if is_dist_avail_and_initialized() and not is_main_process():
#         return {}

#     model.eval()
#     running = {
#         "total": 0.0,
#         "hm_frame": 0.0,
#         "hm_agg": 0.0,
#         "kp_frame": 0.0,
#         "kp_agg": 0.0,
#         "token_spatial": 0.0,
#     }
#     n_batches = len(loader)
#     output_mode = cfg["model"].get("output_mode", "both")

#     for it, batch in enumerate(loader):
#         frames, kps_gt, gt_q, gt_r, vis, hm_gt, delta_Rt_gt, _, crop_xywh = batch
#         frames = frames.to(device, non_blocking=True)
#         kps_gt = kps_gt.to(device, non_blocking=True)
#         gt_q   = gt_q.to(device, non_blocking=True)
#         gt_r   = gt_r.to(device, non_blocking=True)
#         vis    = vis.to(device, non_blocking=True)
#         hm_gt  = hm_gt.to(device, non_blocking=True)
#         delta_Rt_gt = delta_Rt_gt.to(device, non_blocking=True)
#         crop_xywh = crop_xywh.to(device, non_blocking=True)

#         out = model(frames, visibility=vis)
#         tau_now = get_tau_for_epoch(cfg, epoch)
#         cfg_loss_over = dict(cfg.get("loss", {})); cfg_loss_over["tau"] = float(tau_now)
#         cfg_over = dict(cfg); cfg_over["loss"] = cfg_loss_over

#         val_loss, parts = compute_losses(
#                                                 out, cfg_over,
#                                                 kps_gt=kps_gt,
#                                                 hm_gt=hm_gt,
#                                                 vis=vis,
#                                                 # crop_xywh=crop_xywh,
#                                                 # temporal_proj=None,
#                                                 phase="Val",
#                                                 )

#         running["total"] += float(val_loss.item())
#         for key in ["hm_frame", "hm_agg", "kp_frame", "kp_agg", "token_spatial"]:
#             if key in parts:
#                 running[key] += float(parts[key].item())
#         # if cfg["model"].get("use_residual", False) and "kp" in parts:
#         #     running["kp"] = running.get("kp",0.0) + float(parts["kp"].item())
#         # for key in ["pck@5_frame","pck@10_frame","pck@5_agg","pck@10_agg"]:
#             # if key in metrics:
#             #     running[key] += float(metrics[key])
#         # if "esa_score" in metrics:
#         #     running["esa"] += metrics["esa_score"]

#     # averages
#     avg = {k: v / n_batches for k,v in running.items()}
#     if is_main_process():
#         log(
#             f"Val Epoch {epoch}: "
#             f"loss={avg['total']:.4f} | "
#             f"hmF={avg['hm_frame']:.4f} | "
#             f"hmA={avg['hm_agg']:.4f}"
#             f"| kpF={avg['kp_frame']:.4f} | "
#             f"kpA={avg['kp_agg']:.4f}"
#             f"| token={avg.get('token_spatial',0):.4f}"
#         )  # + f"| pck5A={avg.get('pck@5_agg',0):.3f} | pck10A={avg.get('pck@10_agg',0):.3f}"

#         # TensorBoard logging
#         writer.add_scalar('Val/Loss', avg["total"], epoch)
#         # writer.add_scalar('Val/Loss_Rot_Frame', avg.get("rot_frame",0), epoch)
#         # writer.add_scalar('Val/Loss_Trn_Frame', avg.get("trn_frame",0), epoch)
#         # writer.add_scalar('Val/Loss_Rot_Agg', avg.get("rot_agg",0), epoch)
#         # writer.add_scalar('Val/Loss_Trn_Agg', avg.get("trn_agg",0), epoch)
#         # writer.add_scalar('Val/Loss_HM', avg["hm"], epoch)
#         writer.add_scalar('Val/HM_Frame', avg['hm_frame'], epoch)
#         writer.add_scalar('Val/HM_Agg',   avg['hm_agg'], epoch)
#         writer.add_scalar('Val/KP_Frame', avg['kp_frame'], epoch)
#         writer.add_scalar('Val/KP_Agg',   avg['kp_agg'], epoch)
#         # if cfg["model"].get("use_residual", False):
#             # writer.add_scalar('Val/Loss_KP', avg["kp"], epoch)

#         # PCK frame
#         # writer.add_scalar('Val/PCK@5_Frame', avg.get("pck@5_frame",0), epoch)
#         # writer.add_scalar('Val/PCK@10_Frame', avg.get("pck@10_frame",0), epoch)
#         # PCK agg
#         # writer.add_scalar('Val/PCK@5_Agg', avg.get("pck@5_agg",0), epoch)
#         # writer.add_scalar('Val/PCK@10_Agg', avg.get("pck@10_agg",0), epoch)

#     return avg

# # ----------------------------
# # Orchestration
# # ----------------------------

# def run_training(cfg, logger=log):
#     setup_distributed(backend=cfg["train"].get("ddp_backend", "nccl"))
#     device = cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
#     local_rank = int(os.environ.get("LOCAL_RANK", 0))
#     if device.startswith("cuda"):
#         torch.cuda.set_device(local_rank)
#         device = f"cuda:{local_rank}"

#     seed = int(cfg["train"].get("seed", 42)) + get_rank()
#     seed_everything(seed)

#     train_loader, val_loader, train_sampler, val_sampler, test_loader = make_dataloaders(cfg)

#     # temporal infoNCE loss
#     # global / cached
#     # temporal_proj = TemporalProjHead(in_dim=cfg["data"]["num_kp"] * 2).to(device)

#     model = build_model(cfg).to(device)
#     # -------------------------------------------------
#     # LOAD STAGE-0 ANCHOR CHECKPOINT
#     # -------------------------------------------------
#     if cfg.get("train", {}).get("resume_from", None):
#         ckpt_path = cfg["train"]["resume_from"]
#         assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"

#         ckpt = torch.load(ckpt_path, map_location=device)
#         state = ckpt["model"]

#         missing, unexpected = model.load_state_dict(state, strict=False)

#         if is_main_process():
#             print(f"[Resume] Loaded Stage-0 checkpoint: {ckpt_path}")
#             print(f"[Resume] Missing keys: {missing}")
#             print(f"[Resume] Unexpected keys: {unexpected}")

#     freeze_all_but_dualpath_and_heads(model, unfreeze_last_n_blocks=0)
#     # for n, p in model.named_parameters():
#     #     if "hm_head" in n:
#     #         p.requires_grad = True
#     #         print(f"[Freeze] {n} frozen.")

#     # -------------------------------------------------
#     # Stage-0: damp spatial adapter strength
#     # -------------------------------------------------
#     for name, module in model.named_modules():
#         if "spatial_" in name and hasattr(module, "scale"):
#             module.scale.data.fill_(0.08)

#     if get_world_size() > 1:
#         model = DDP(
#             model,
#             device_ids=[local_rank],
#             output_device=local_rank,
#             broadcast_buffers=False,
#             find_unused_parameters=True,
#         )

#     # sanity check debugging----------------------------------
#     # real = model.module if isinstance(model, DDP) else model
#     # for n, p in real.named_parameters():
#     #     if p.requires_grad:
#     #         print(f"[trainable] {n} {p.shape}")

#     # --------------------------------------------------------

#     scaler = GradScaler(device="cuda" if torch.cuda.is_available() else "cpu",
#                         enabled=cfg["train"].get("amp", True))

#     optim_name = cfg["train"].get("optimizer", "adamw").lower()
#     lr = float(cfg["train"].get("lr", 3e-4))
#     wd = float(cfg["train"].get("weight_decay", 0.05))
#     head_lr_mult = float(cfg["train"].get("head_lr_mult", 5.0))
#     block_lr_mult = float(cfg["train"].get("block_lr_mult", 0.5))

#     param_groups = split_params_for_optimizer(model, base_lr=lr, weight_decay=wd, head_lr_mult=head_lr_mult, block_lr_mult=block_lr_mult)
#     if optim_name == "adamw":
#         optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))
#     else:
#         optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999))

#     warmup_epochs = int(cfg["train"].get("warmup_epochs", 0))
#     warmup_steps  = cfg["train"].get("warmup_steps", None)
#     lr_min        = float(cfg["train"].get("lr_min", 1e-6))
#     scheduler = build_warmup_cosine_scheduler(
#         optimizer,
#         epochs=cfg["train"]["epochs"],
#         steps_per_epoch=len(train_loader),
#         warmup_epochs=warmup_epochs,
#         warmup_steps=warmup_steps,
#         lr_min=lr_min,
#     )

#     # best_metric = float("inf")
#     cfg["_best_metric"] = float("inf")
#     out_dir = cfg["train"].get("out_dir", "./outputs")
#     os.makedirs(out_dir, exist_ok=True)

#     #tensorboard
#     run_name = f"exp_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
#     logdir = os.path.join(cfg['train']['tb_logdir'], run_name)
#     writer = SummaryWriter(log_dir=logdir)

#     eval_configs = cfg.get("eval", {})
#     eval_configs["device"] = device
#     eval_configs["data"] = cfg["data"]  # inject data-level info

#     global_step = 0
#     for epoch in range(1, cfg["train"]["epochs"] + 1):
#         if isinstance(train_sampler, DistributedSampler):
#             train_sampler.set_epoch(epoch)
#         if isinstance(val_sampler, DistributedSampler):
#             val_sampler.set_epoch(epoch)
        
#         # if is_main_process():
#         #     avg_esa, avg_loc, avg_ori, avg_range, avg_pck_10, avg_pack_5 = evaluate_model(model, test_loader, eval_configs, 42)  # evaluate at the start of epoch
#         #     print("[Testing] ESA Avg: {:.3f} | Loc Avg: {:.3f} | Ori Avg: {:.3f} | Range Avg: {:.3f} | PCK Avg@10: {:.2f}% | PCK Avg@5: {:.2f}%".format(
#         #                 avg_esa, avg_loc, avg_ori, avg_range, avg_pck_10, avg_pack_5
#         #                 ))
#         # exit()
#         _ = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, cfg, epoch, global_step, writer)

#         cfg["_epoch_for_val"] = epoch

    
#         model.eval()
#         val_stats = validate(model, val_loader, device, cfg, epoch, writer, global_step)

#         if is_main_process():
#             score = val_stats["total"]
#             # simple model selection: smallest val loss
#             # if "total" in val_stats:
#             #     score = val_stats["total"]   # LOWER is better
#             # else:
#             #     score = val_stats["total"]

#             if score < cfg["_best_metric"]:
#                 cfg["_best_metric"] = score
#                 best_path = os.path.join(out_dir, f"best_epoch_{epoch}.pt")
#                 torch.save(
#                     {
#                         "cfg": cfg,
#                         "epoch": epoch,
#                         "model": (model.module.state_dict() if isinstance(model, DDP)
#                                 else model.state_dict()),
#                         "optimizer": optimizer.state_dict(),
#                         "best_metric": score,
#                     },
#                     best_path,
#                 )
#                 logger(f"✓ Saved best checkpoint to {best_path} (score={score:.4f})")
#         if is_main_process():
#             if epoch % 5 == 0 : #or (epoch == cfg["train"]["epochs"] - 1):
#                 with torch.no_grad():
#                     avg_esa, avg_loc, avg_ori, avg_range, avg_pck_10, avg_pack_5 = evaluate_model(model, test_loader, eval_configs, epoch)            
#                 print("[Testing] ESA Avg: {:.3f} | Loc Avg: {:.3f} | Ori Avg: {:.3f} | Range Avg: {:.3f} | PCK Avg@10: {:.2f}% | PCK Avg@5: {:.2f}%".format(
#                         avg_esa, avg_loc, avg_ori, avg_range, avg_pck_10, avg_pack_5
#                         ))

#         if is_dist_avail_and_initialized():
#             dist.barrier()

#     if is_main_process():
#         b_m = cfg["_best_metric"]
#         logger(f"Training done. Best score={b_m:.4f}")
#     cleanup_distributed()
