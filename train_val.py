# train_val.py
import os
import math
import time
import warnings
import random
import datetime
from typing import Dict, Optional, List, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import numpy as np

# ---- Silence the albumentations ShiftScaleRotate warning (true fix is in transforms.py) ----
warnings.filterwarnings(
    "ignore",
    message="ShiftScaleRotate is a special case of Affine transform. Please use Affine transform instead."
)

from model import SpacecraftPoseDualPath
from dataset import SPADESDataset, SPARKDataset
from transforms import build_transforms

from utils import (
    build_warmup_cosine_scheduler,
    load_camera_intrinsics,
    freeze_all_but_dualpath_and_heads,
    cosine_schedule,
    get_heatmap_power_for_epoch,
    get_tau_for_epoch,
    get_w_token_for_epoch,
    collate_temporal_chunks,
)

from utils import save_kp_overlay_bt, save_token_similarity_overlay_bt

from losses import (
    heatmap_to_com,
    smooth_l1_vis_weighted,
    token_spatial_alignment_loss,
    temporal_smoothness_l2,
    heatmap_kl_div_from_logits_masked,
    softargmax_2d_from_logits,
)

from evaluate import evaluate_model

# ---- perf toggles (safe even on CPU; they only take effect on CUDA) ----
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# =========================
# DDP helpers
# =========================
def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1

def is_main_process() -> bool:
    return get_rank() == 0

def log(s: str, flush: bool = True, **kwargs):
    print(s, flush=flush, **kwargs)

def setup_distributed(backend="nccl"):
    """
    Initializes default process group using env:// (works with torchrun + Slurm).
    Sets current CUDA device from LOCAL_RANK.
    """
    if is_dist_avail_and_initialized():
        return

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend=backend, init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False


# =========================
# Optimizer param groups
# =========================
def split_params_for_optimizer(
    model,
    base_lr,
    weight_decay=0.05,
    head_lr_mult=5.0,
    block_lr_mult=0.5,
):
    real = model.module if hasattr(model, "module") else model

    adapter_params, head_params, block_params = [], [], []

    def no_weight_decay(n: str) -> bool:
        return n.endswith("bias") or "bn" in n.lower() or "norm" in n.lower()

    for n, p in real.named_parameters():
        if not p.requires_grad:
            continue

        if "adapter" in n or "dualpath" in n:
            adapter_params.append((n, p))
        elif any(k in n for k in ["hm_head", "keypoint_head", "pose_head", "finer_cnn", "pnp", "kp_res_head"]):
            head_params.append((n, p))
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

    if is_main_process():
        log(f"[opt] LR groups:")
        for g in groups:
            n_params = sum(p.numel() for p in g["params"])
            log(f"  - {n_params} params @ lr={g['lr']}, wd={g['weight_decay']}")

    return groups


# =========================
# Subset helpers
# =========================
def _subset_indices(n_total: int, n_take: int, mode: str = "first", seed: int = 0) -> List[int]:
    n_take = int(min(max(0, n_take), n_total))
    if n_take <= 0 or n_take == n_total:
        return list(range(n_total))
    if mode == "first":
        return list(range(n_take))
    rng = np.random.RandomState(seed)
    idx = np.arange(n_total)
    rng.shuffle(idx)
    return idx[:n_take].tolist()

def _broadcast_indices(idxs: List[int]) -> List[int]:
    if not is_dist_avail_and_initialized():
        return idxs
    obj_list = [idxs]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]

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
            ds = Subset(ds, range(gmin))
    return ds


# =========================
# Loss schedule helper
# =========================
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

def apply_epoch_loss_schedules(cfg: Dict[str, Any], epoch: int):
    """
    One place to set all per-epoch scheduled knobs.
    This avoids 'tau_now not in config.yml' issues and avoids per-iteration cfg mutation.
    """
    cfg.setdefault("loss", {})
    tau_now = float(get_tau_for_epoch(cfg, epoch))
    tau_now = max(tau_now, 0.05)  # avoid extremely sharp maps early
    cfg["loss"]["tau_now"] = tau_now

    cfg["loss"]["heatmap_target_power"] = float(get_heatmap_power_for_epoch(cfg, epoch))

    # heatmap weights (if schedule is null, this returns base)
    w_hm_now = float(get_w_hm_for_epoch(cfg, epoch))
    cfg["loss"]["w_hm_frame"] = float(cfg["loss"].get("w_hm_frame", 1.0)) if cfg["loss"].get("w_hm_schedule", None) is None else w_hm_now
    cfg["loss"]["w_hm_agg"]   = float(cfg["loss"].get("w_hm_agg", 0.1))   if cfg["loss"].get("w_hm_schedule", None) is None else w_hm_now

    cfg["loss"]["w_token"] = float(get_w_token_for_epoch(cfg, epoch))


# =========================
# Model factory
# =========================
def build_model(cfg) -> nn.Module:
    cam_params = None
    if cfg["model"].get("use_pnp", False):
        cam_params = load_camera_intrinsics(cfg["data"]["cam_intrinsics_json"])

    common = dict(
        aggregator=cfg["model"].get("aggregator", "attn"),
        ckpt_path=cfg["model"]["backbone_ckpt"],
        d_side=cfg["model"].get("d_side", 0),
        heatmap_out=tuple(cfg["model"].get("heatmap_out", [56, 56])),

        cam_params=cam_params,

        hm_head=cfg["model"].get("hm_head", "film"),
        hm_mid_c=cfg["model"].get("hm_mid_c", 128),
        hm_use_film=cfg["model"].get("hm_use_film", False),
        use_kp_residual=cfg["model"].get("use_residual", False),

        cfg=cfg,
    )
    return SpacecraftPoseDualPath(**common)


# =========================
# Dataloaders
# =========================
def make_dataloaders(cfg):
    ds_name = cfg["data"].get("name", "").lower()

    tfm_crop = build_transforms(img_size=int(cfg["data"]["input_size"]))
    tfm_full = build_transforms(img_size=int(cfg["data"]["input_size"]))

    base_kwargs = dict(
        gt_folder=cfg["data"]["root"],
        short_mem=cfg["data"]["short_mem"],
        input_size=cfg["data"]["input_size"],
        sampling_rate=cfg["data"]["sampling_rate"],
        clip_stride=cfg["data"].get("clip_stride", None),
        crop=cfg["data"].get("crop", True),
        crop_prob=cfg["data"].get("crop_prob", 0.0),
        eval_use_crop=cfg["data"].get("eval_use_crop", False),
        gaussian_sigma_px=cfg["data"].get("gaussian_sigma_px", 2.0),
        heatmap_out=tuple(cfg["model"].get("heatmap_out", [56, 56])),
    )

    test_set = None

    if ds_name == "spades":
        train_set = SPADESDataset(mode="train", spatial_transforms=tfm_crop, spatial_transforms_full=tfm_full, **base_kwargs)
        val_set   = SPADESDataset(mode="val", spatial_transforms=None, spatial_transforms_full=None, **base_kwargs)
        test_set  = SPADESDataset(mode="test", spatial_transforms=None, spatial_transforms_full=None, **base_kwargs)

    elif ds_name == "spark":
        train_set = SPARKDataset(mode="train", spatial_transforms=tfm_crop, spatial_transforms_full=tfm_full, **base_kwargs)
        val_set   = SPARKDataset(mode="val", spatial_transforms=None, spatial_transforms_full=None, **base_kwargs)
        # If your SPARK dataset has no explicit "test", just reuse val for testing hooks:
        test_set  = SPARKDataset(mode="val", spatial_transforms=None, spatial_transforms_full=None, **base_kwargs)

    else:
        raise ValueError("data.name must be 'spades' or 'spark'")

    # ---- Optional debugging subsets ----
    train_take = int(cfg["data"].get("train_subset", 0) or 0)
    val_take   = int(cfg["data"].get("val_subset", 0) or 0)
    sub_mode   = str(cfg["data"].get("subset_mode", "first")).lower()
    sub_seed   = int(cfg["data"].get("subset_seed", 0) or 0)

    if train_take > 0:
        idxs = _subset_indices(len(train_set), train_take, mode=sub_mode, seed=sub_seed) if is_main_process() else []
        idxs = _broadcast_indices(idxs)
        train_set = Subset(train_set, idxs)
        if is_main_process():
            log(f"[Subset] train -> {len(idxs)} samples (mode={sub_mode}, seed={sub_seed})")

    if val_take > 0:
        idxs = _subset_indices(len(val_set), val_take, mode=sub_mode, seed=sub_seed + 1) if is_main_process() else []
        idxs = _broadcast_indices(idxs)
        val_set = Subset(val_set, idxs)
        if is_main_process():
            log(f"[Subset] val   -> {len(idxs)} samples (mode={sub_mode}, seed={sub_seed + 1})")

    # ---- Ensure identical lengths across ranks ----
    train_set = _sync_and_trim_dataset(train_set, "train_set")
    val_set   = _sync_and_trim_dataset(val_set, "val_set")

    # ---- Distributed samplers ----
    world = get_world_size()
    rank  = get_rank()

    train_sampler = DistributedSampler(train_set, num_replicas=world, rank=rank, shuffle=True, drop_last=True) if world > 1 else None
    val_sampler   = DistributedSampler(val_set,   num_replicas=world, rank=rank, shuffle=False, drop_last=False) if world > 1 else None

    nw = int(cfg["train"].get("num_workers", 4))
    bs = int(cfg["train"]["batch_size"])
    vbs = int(cfg["train"].get("val_batch_size", bs))

    train_loader = DataLoader(
        train_set,
        batch_size=bs,
        shuffle=(train_sampler is None),
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
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

    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_temporal_chunks(b, cfg["data"]["test_T_chunk"]),
    )

    # ---- Sanity checks for DDP ----
    if is_dist_avail_and_initialized():
        tb = len(train_loader)
        tsl = len(train_sampler) if train_sampler is not None else len(train_set)
        all_tb = [None] * world
        all_tsl = [None] * world
        dist.all_gather_object(all_tb, tb)
        dist.all_gather_object(all_tsl, tsl)
        if is_main_process():
            log(f"[check] train_batches per rank: {all_tb}")
            log(f"[check] train_sampler_len per rank: {all_tsl}")
        if len(set(all_tb)) != 1 or len(set(all_tsl)) != 1:
            raise RuntimeError(f"Train loader/sampler mismatch across ranks: batches={all_tb}, sampler_len={all_tsl}")

    return train_loader, val_loader, train_sampler, val_sampler, test_loader


# =========================
# Compute losses
# =========================
def compute_losses(
    out: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
    kps_gt: torch.Tensor,    # (B,T,K,2) in [-1,1]
    hm_gt: torch.Tensor,     # (B,T,K,H,W)
    vis: torch.Tensor,       # (B,T,K) 0/1
    phase: str = "Train",
):
    loss_cfg = cfg.get("loss", {})
    tau = float(loss_cfg.get("tau_now", loss_cfg.get("tau", 1.0)))

    # weights
    w_hm_frame = float(loss_cfg.get("w_hm_frame", 1.0))
    w_hm_agg   = float(loss_cfg.get("w_hm_agg", loss_cfg.get("w_hm", 0.0)))
    w_token    = float(loss_cfg.get("w_token", 0.0))
    w_temp     = float(loss_cfg.get("w_temp", 0.0))
    target_pow = float(loss_cfg.get("heatmap_target_power", 1.0))
    w_kp       = float(loss_cfg.get("w_kp", 1.0))

    hm_logits = out["heatmaps"]  # (B,T,K,H,W)
    B, T, K, Hm, Wm = hm_logits.shape

    total_loss = hm_logits.new_zeros(())
    parts: Dict[str, torch.Tensor] = {}

    # 1) Heatmap KL per-frame (masked)
    L_hm_frame = heatmap_kl_div_from_logits_masked(
        hm_logits=hm_logits,
        hm_target=hm_gt,
        vis=vis,
        tau=tau,
        target_power=target_pow,
    )
    total_loss += w_hm_frame * L_hm_frame
    parts["hm_frame"] = L_hm_frame

    # 2) Heatmap KL aggregated over time
    hm_logits_agg = hm_logits.mean(dim=1)        # (B,K,H,W)
    hm_gt_agg     = hm_gt.mean(dim=1)            # (B,K,H,W)
    vis_agg       = (vis.sum(dim=1) > 0).float() # (B,K)

    L_hm_agg = heatmap_kl_div_from_logits_masked(
        hm_logits=hm_logits_agg.unsqueeze(1),    # (B,1,K,H,W)
        hm_target=hm_gt_agg.unsqueeze(1),
        vis=vis_agg.unsqueeze(1),
        tau=tau,
        target_power=target_pow,
    )
    total_loss += w_hm_agg * L_hm_agg
    parts["hm_agg"] = L_hm_agg

    # 3) Token spatial alignment loss
    if w_token > 0 and "patch_tokens_prenorm" in out:
        L_token = token_spatial_alignment_loss(
            patch_tokens=out["patch_tokens_prenorm"],  # (B,T,P,D)
            kps_gt_norm=kps_gt,
            vis=vis,
            temperature=0.07,
        )
        total_loss += w_token * L_token
        parts["token_spatial"] = L_token

    # 4) Differentiable keypoint loss (soft-argmax)
    if w_kp > 0:
        kps_px = softargmax_2d_from_logits(hm_logits, tau=tau)  # (B,T,K,2)
        x_norm = 2.0 * (kps_px[..., 0] / (Wm - 1.0)) - 1.0
        y_norm = 2.0 * (kps_px[..., 1] / (Hm - 1.0)) - 1.0
        kps_pred_norm = torch.stack([x_norm, y_norm], dim=-1)

        L_kp = smooth_l1_vis_weighted(kps_pred_norm, kps_gt, vis)
        total_loss += w_kp * L_kp
        parts["kp_frame"] = L_kp
    else:
        kps_pred_norm = None  # noqa

    # 5) Temporal smoothness on COM (train only)
    if w_temp > 0 and phase == "Train":
        com_px = heatmap_to_com(hm_logits, tau=tau)  # (B,T,K,2) in px
        x = 2.0 * (com_px[..., 0] / (Wm - 1.0)) - 1.0
        y = 2.0 * (com_px[..., 1] / (Hm - 1.0)) - 1.0
        com_norm = torch.stack([x, y], dim=-1)

        L_temp_raw = temporal_smoothness_l2(com_norm, vis)
        total_loss += w_temp * L_temp_raw
        parts["temp_smooth_raw"] = L_temp_raw
        parts["temp_smooth"] = w_temp * L_temp_raw

    return total_loss, parts


# =========================
# Train One Epoch
# =========================
def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, cfg, epoch, global_step, writer):
    phase = "Train"
    model.train()

    # Apply schedules ONCE per epoch
    apply_epoch_loss_schedules(cfg, epoch)

    max_norm = float(cfg["train"].get("grad_clip", 1.0))
    use_amp = bool(cfg["train"].get("amp", True))

    running = {"total": 0.0}
    n_batches = len(loader)

    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
    autocast_dtype = torch.bfloat16 if dev_type == "cuda" else torch.bfloat16

    for it, batch in enumerate(loader):
        frames, kps_gt, gt_q, gt_r, vis, hm_gt, delta_Rt_gt, _, crop_xywh = batch

        frames = frames.to(device, non_blocking=True)
        kps_gt = kps_gt.to(device, non_blocking=True)
        vis    = vis.to(device, non_blocking=True)
        hm_gt  = hm_gt.to(device, non_blocking=True)
        # gt_q, gt_r, delta_Rt_gt, crop_xywh kept for future use:
        gt_q = gt_q.to(device, non_blocking=True)
        gt_r = gt_r.to(device, non_blocking=True)
        delta_Rt_gt = delta_Rt_gt.to(device, non_blocking=True)
        crop_xywh = crop_xywh.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=dev_type, dtype=autocast_dtype, enabled=use_amp):
            out = model(frames, visibility=vis)


        # ==========================
        # SAVE VISUALS (disk) — every 5 epochs
        # 4 samples × 2 images (kp overlay + token map)
        # only on main rank, only first iter to keep cheap
        # ==========================
        if is_main_process() and (epoch % 5 == 0) and (it == 0):
            viz_root = os.path.join(cfg["train"].get("out_dir", "./outputs"), "train_viz_tokens")
            ep_dir = os.path.join(viz_root, f"epoch_{epoch:04d}")
            os.makedirs(ep_dir, exist_ok=True)

            B, T = frames.shape[0], frames.shape[1]
            t_sel = T // 2  # "mid" frame

            n_show = min(4, B)
            for b_idx in range(n_show):
                kp_path = os.path.join(ep_dir, f"b{b_idx}_t{t_sel}_kp_overlay.png")
                tok_path = os.path.join(ep_dir, f"b{b_idx}_t{t_sel}_token_sim.png")

                save_kp_overlay_bt(
                    frames=frames,
                    kps_gt_norm=kps_gt,
                    hm_logits=out["heatmaps"],
                    vis=vis,
                    cfg=cfg,
                    save_path=kp_path,
                    b=b_idx,
                    t=t_sel,
                )

                save_token_similarity_overlay_bt(
                    out=out,
                    frames=frames,
                    kps_gt=kps_gt,
                    vis=vis,
                    cfg=cfg,
                    save_path=tok_path,
                    b=b_idx,
                    t=t_sel,
                    sigma_patch=1.25,
                )

            log(f"[viz] saved token/kp visuals -> {ep_dir}", flush=True)
        # compute loss in fp32 for stability
        with torch.autocast(device_type=dev_type, enabled=False):
            total_loss, parts = compute_losses(
                out, cfg,
                kps_gt=kps_gt,
                hm_gt=hm_gt,
                vis=vis,
                phase="Train",
            )

        scaler.scale(total_loss).backward()

        if max_norm and max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()
        if scheduler:
            scheduler.step()

        global_step += 1

        # accumulate
        for k, v in [("total", total_loss)] + list(parts.items()):
            val = float(v.item()) if torch.is_tensor(v) else float(v)
            if math.isnan(val) or math.isinf(val):
                if is_main_process():
                    log(f"[WARN] Skipping {k} = {val} (NaN/Inf)")
                continue
            running[k] = running.get(k, 0.0) + val

    avg = {k: v / n_batches for k, v in running.items()}

    if is_main_process():
        log(
            f"{phase} Epoch {epoch}: "
            f"loss={avg['total']:.4f} "
            f"| hmF={avg.get('hm_frame',0):.4f} | hmA={avg.get('hm_agg',0):.4f} "
            f"| kpF={avg.get('kp_frame',0):.4f} "
            f"| temp_raw={avg.get('temp_smooth_raw',0):.4f} "
            f"| temp_w={avg.get('temp_smooth',0):.4f} "
            f"| token={avg.get('token_spatial',0):.4f} "
            f"| tau_now={cfg['loss'].get('tau_now', cfg['loss'].get('tau', 0)):.3f}"
        )
        writer.add_scalar(f'{phase}/Loss', avg.get("total", 0), epoch)
        writer.add_scalar(f'{phase}/Loss_HM_Frame', avg.get("hm_frame", 0), epoch)
        writer.add_scalar(f'{phase}/Loss_HM_Agg',   avg.get("hm_agg", 0), epoch)
        writer.add_scalar(f'{phase}/Loss_KP_Frame', avg.get("kp_frame", 0), epoch)
        writer.add_scalar(f'{phase}/Loss_Temp',     avg.get("temp_smooth", 0), epoch)
        writer.add_scalar("Train/w_token", float(cfg["loss"].get("w_token", 0.0)), epoch)
        writer.add_scalar("Train/tau_now", float(cfg["loss"].get("tau_now", cfg["loss"].get("tau", 1.0))), epoch)

    return avg


# =========================
# Validation
# =========================
@torch.no_grad()
def validate(model, loader, device, cfg, epoch, writer, global_step):
    # Only run validation on main process (common DDP pattern)
    if is_dist_avail_and_initialized() and not is_main_process():
        return {}

    model.eval()

    # Apply schedules ONCE per epoch (use same schedule as train)
    apply_epoch_loss_schedules(cfg, epoch)

    running = {
        "total": 0.0,
        "hm_frame": 0.0,
        "hm_agg": 0.0,
        "kp_frame": 0.0,
        "token_spatial": 0.0,
    }
    n_batches = len(loader)

    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
    autocast_dtype = torch.bfloat16 if dev_type == "cuda" else torch.bfloat16

    for it, batch in enumerate(loader):
        frames, kps_gt, gt_q, gt_r, vis, hm_gt, delta_Rt_gt, _, crop_xywh = batch

        frames = frames.to(device, non_blocking=True)
        kps_gt = kps_gt.to(device, non_blocking=True)
        vis    = vis.to(device, non_blocking=True)
        hm_gt  = hm_gt.to(device, non_blocking=True)

        # unused currently but keep tensors moved (future-proof)
        gt_q = gt_q.to(device, non_blocking=True)
        gt_r = gt_r.to(device, non_blocking=True)
        delta_Rt_gt = delta_Rt_gt.to(device, non_blocking=True)
        crop_xywh = crop_xywh.to(device, non_blocking=True)

        with torch.autocast(device_type=dev_type, dtype=autocast_dtype, enabled=False):
            out = model(frames, visibility=vis)

        with torch.autocast(device_type=dev_type, enabled=False):
            val_loss, parts = compute_losses(
                out, cfg,
                kps_gt=kps_gt,
                hm_gt=hm_gt,
                vis=vis,
                phase="Val",
            )

        running["total"] += float(val_loss.item())
        for key in ["hm_frame", "hm_agg", "kp_frame", "token_spatial"]:
            if key in parts:
                running[key] += float(parts[key].item())

    avg = {k: v / n_batches for k, v in running.items()}

    if is_main_process():
        log(
            f"Val Epoch {epoch}: "
            f"loss={avg['total']:.4f} | "
            f"hmF={avg['hm_frame']:.4f} | "
            f"hmA={avg['hm_agg']:.4f} | "
            f"kpF={avg['kp_frame']:.4f} | "
            f"token={avg.get('token_spatial',0):.4f} "
            f"| tau_now={cfg['loss'].get('tau_now', cfg['loss'].get('tau', 0)):.3f}"
        )

        writer.add_scalar('Val/Loss', avg["total"], epoch)
        writer.add_scalar('Val/HM_Frame', avg['hm_frame'], epoch)
        writer.add_scalar('Val/HM_Agg',   avg['hm_agg'], epoch)
        writer.add_scalar('Val/KP_Frame', avg['kp_frame'], epoch)
        writer.add_scalar('Val/Token',    avg.get('token_spatial', 0.0), epoch)
        writer.add_scalar("Val/tau_now",  float(cfg["loss"].get("tau_now", cfg["loss"].get("tau", 1.0))), epoch)

    return avg


# =========================
# Orchestration
# =========================
def run_training(cfg, logger=log):
    setup_distributed(backend=cfg["train"].get("ddp_backend", "nccl"))

    device = cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"

    seed = int(cfg["train"].get("seed", 42)) + get_rank()
    seed_everything(seed)

    train_loader, val_loader, train_sampler, val_sampler, test_loader = make_dataloaders(cfg)

    model = build_model(cfg).to(device)

    # ---- Resume ----
    if cfg.get("train", {}).get("resume_from", None):
        ckpt_path = cfg["train"]["resume_from"]
        assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if is_main_process():
            log(f"[Resume] Loaded: {ckpt_path}")
            log(f"[Resume] Missing keys: {missing}")
            log(f"[Resume] Unexpected keys: {unexpected}")

    freeze_all_but_dualpath_and_heads(model, unfreeze_last_n_blocks=0)

    # # ---- Damp spatial adapter strength (stage-0) ----
    # for name, module in model.named_modules():
    #     if "spatial_" in name and hasattr(module, "scale"):
    #         try:
    #             module.scale.data.fill_(0.08)
    #         except Exception:
    #             pass

    if get_world_size() > 1:
        model = DDP(
            model,
            device_ids=[local_rank] if str(device).startswith("cuda") else None,
            output_device=local_rank if str(device).startswith("cuda") else None,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    # ---- AMP scaler (torch.amp.GradScaler expects device_type) ----
    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
    scaler = torch.amp.GradScaler(device="cuda" if torch.cuda.is_available() else "cpu",
                        enabled=cfg["train"].get("amp", True))

    optim_name = cfg["train"].get("optimizer", "adamw").lower()
    lr = float(cfg["train"].get("lr", 3e-4))
    wd = float(cfg["train"].get("weight_decay", 0.05))
    head_lr_mult = float(cfg["train"].get("head_lr_mult", 5.0))
    block_lr_mult = float(cfg["train"].get("block_lr_mult", 0.5))

    param_groups = split_params_for_optimizer(
        model,
        base_lr=lr,
        weight_decay=wd,
        head_lr_mult=head_lr_mult,
        block_lr_mult=block_lr_mult,
    )

    if optim_name == "adamw":
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))
    else:
        optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.999))

    warmup_epochs = int(cfg["train"].get("warmup_epochs", 0))
    warmup_steps  = cfg["train"].get("warmup_steps", None)
    lr_min        = float(cfg["train"].get("lr_min", 1e-6))

    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        epochs=int(cfg["train"]["epochs"]),
        steps_per_epoch=len(train_loader),
        warmup_epochs=warmup_epochs,
        warmup_steps=warmup_steps,
        lr_min=lr_min,
    )

    cfg["_best_metric"] = float("inf")
    out_dir = cfg["train"].get("out_dir", "./outputs")
    os.makedirs(out_dir, exist_ok=True)

    run_name = f"exp_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    logdir = os.path.join(cfg['train']['tb_logdir'], run_name)
    writer = SummaryWriter(log_dir=logdir)

    eval_configs = cfg.get("eval", {})
    eval_configs["device"] = device
    eval_configs["data"] = cfg["data"]

    global_step = 0
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        if isinstance(val_sampler, DistributedSampler):
            val_sampler.set_epoch(epoch)

        _ = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, cfg, epoch, global_step, writer)
        val_stats = validate(model, val_loader, device, cfg, epoch, writer, global_step)

        if is_main_process():
            score = float(val_stats.get("total", 1e9))
            if score < cfg["_best_metric"]:
                cfg["_best_metric"] = score
                best_path = os.path.join(out_dir, f"best_epoch_{epoch}.pt")
                torch.save(
                    {
                        "cfg": cfg,
                        "epoch": epoch,
                        "model": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
                        "optimizer": optimizer.state_dict(),
                        "best_metric": score,
                    },
                    best_path,
                )
                logger(f"✓ Saved best checkpoint to {best_path} (score={score:.4f})")

            if epoch % 5 == 0:
                with torch.no_grad():
                    avg_esa, avg_loc, avg_ori, avg_range, avg_pck_10, avg_pack_5 = evaluate_model(
                        model, test_loader, eval_configs, epoch
                    )
                log(
                    "[Testing] ESA Avg: {:.3f} | Loc Avg: {:.3f} | Ori Avg: {:.3f} | Range Avg: {:.3f} | "
                    "PCK Avg@10: {:.2f}% | PCK Avg@5: {:.2f}%".format(
                        avg_esa, avg_loc, avg_ori, avg_range, avg_pck_10, avg_pack_5
                    )
                )

        if is_dist_avail_and_initialized():
            dist.barrier()

    if is_main_process():
        logger(f"Training done. Best score={cfg['_best_metric']:.4f}")
    cleanup_distributed()