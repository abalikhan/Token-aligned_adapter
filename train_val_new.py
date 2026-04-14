# train_val.py (CLEAN, DDP-safe, main-rank logging/checkpointing only)
import os
import math
import random
import datetime
import warnings
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings(
    "ignore",
    message="ShiftScaleRotate is a special case of Affine transform. Please use Affine transform instead."
)

from dataset import SPADESDataset, SPARKDataset
from vbn_dataset import VBNDataset
from transforms import build_clip_augmentation

from model_hm_dsnt_2 import SpacecraftHMDSNT
# from model_hm_dsnt_mhsa_mlp import SpacecraftHMDSNT

from losses import (
    heatmap_kl_div_from_logits_masked,
    smooth_l1_vis_weighted,
    token_spatial_alignment_loss,
    heatmap_temporal_smoothness,
    temporal_smoothness_loss,
    frequency_jitter_loss,
)

from utils_new import (
    build_warmup_cosine_scheduler,
    build_warmup_cosine_restart_scheduler,
    freeze_all_but_adapters_and_heads,
    collate_temporal_chunks,
    # save_kp_overlay_bt,
    # save_token_similarity_overlay_bt,
    # pck_from_norm,
    get_w_token,
    # save_token_similarity_diagnostics_bt,
    pck_crop_from_normhm
)

from viz_utils import (
    save_kp_overlay_clip_grid,
    save_token_diagnostics,
    save_adapter_feature_maps,
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# --- add near imports ---
from typing import Callable

# --- keep helpers as-is, but make log use an overridable logger ---
def _default_logger(s: str):
    if is_main_process():
        print(s, flush=True)

# -------------------------
# DDP helpers
# -------------------------
def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def _subset_indices(n: int, take: int, mode: str = "random", seed: int = 0):
    """
    Returns a Python list of indices length=take.
    mode:
      - "first": take first N
      - "random": random sample without replacement (deterministic by seed)
    """
    take = int(take)
    if take <= 0 or take >= n:
        return list(range(n))

    mode = str(mode).lower()
    if mode == "first":
        return list(range(take))

    # default random
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return idx[:take].tolist()

def _broadcast_indices(idxs):
    """
    Make sure all ranks see the exact same subset indices.
    """
    if not is_dist_avail_and_initialized():
        return idxs
    obj_list = [idxs if get_rank() == 0 else None]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]

def _sync_min_len(n: int) -> int:
    """
    Return the minimum of n across all ranks.
    Used to ensure subset indices are always in range on every rank.
    """
    if not is_dist_avail_and_initialized():
        return n
    lens = [None] * get_world_size()
    dist.all_gather_object(lens, n)
    return min(lens)

def _sync_and_trim_dataset(ds, name: str):
    """
    Ensures every rank has the same dataset length.
    If mismatch happens (usually due to file listing differences), trims to min length.
    Works for Dataset or Subset.
    """
    if not is_dist_avail_and_initialized():
        return ds

    local_len = len(ds)
    lens = [None] * get_world_size()
    dist.all_gather_object(lens, local_len)

    if len(set(lens)) == 1:
        return ds

    # If mismatch: trim to min length deterministically.
    min_len = min(lens)

    if is_main_process():
        log(f"[DDP-sync] {name} length mismatch across ranks: {lens} -> trimming to {min_len}")

    # Trim by taking first min_len samples of current ds ordering.
    if isinstance(ds, Subset):
        new_indices = list(ds.indices)[:min_len]
        return Subset(ds.dataset, new_indices)
    else:
        return Subset(ds, list(range(min_len)))
    
def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0

def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1

def is_main_process() -> bool:
    return get_rank() == 0

def log(s: str):
    # Only main prints to avoid multi-rank spam
    if is_main_process():
        print(s, flush=True)

def setup_distributed(backend: str = "nccl"):
    """
    Initializes torch.distributed if launched with torchrun (env://).
    Safe to call in single-GPU as well.
    """
    if is_dist():
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend=backend, init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

def cleanup_distributed():
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sync_float(x: float) -> float:
    """Broadcast a float from rank0 to all ranks."""
    if not is_dist_avail_and_initialized():
        return float(x)
    obj = [float(x) if is_main_process() else None]
    dist.broadcast_object_list(obj, src=0)
    return float(obj[0])

def _apply_lr_mult_from_scheduler(optimizer, scheduler, lr_mult: float):
    """
    IMPORTANT:
      - scheduler sets a 'base' LR each step
      - we then overwrite optimizer LR = base_lr * lr_mult
    """
    if scheduler is None:
        # no scheduler => just scale current lr once (rarely used here)
        for g in optimizer.param_groups:
            g["lr"] = float(g["lr"]) * float(lr_mult)
        return

    base_lrs = scheduler.get_last_lr()  # list with len == num param groups
    for g, base in zip(optimizer.param_groups, base_lrs):
        g["lr"] = float(base) * float(lr_mult)

def make_trainable_params_contiguous(m: nn.Module):
    for p in m.parameters():
        if p.requires_grad and not p.is_contiguous():
            p.data = p.data.contiguous()
# -------------------------
# Model / loaders
# -------------------------
def build_model(cfg) -> nn.Module:
    dcfg = cfg["data"]
    mcfg = cfg.get("model", {})

    num_kp = int(dcfg.get("num_kp", 8))

    short_mem = int(dcfg["short_mem"])
    sampling_rate = int(dcfg["sampling_rate"])
    # IMPORTANT: int, not float
    num_frames = int(math.ceil(short_mem / sampling_rate))

    img_hw = int(dcfg.get("input_size", 224))
    hf_name = mcfg.get("hf_name", "facebook/dinov3-vitl16-pretrain-lvd1689m")

    nreg = int(mcfg.get("n_register_tokens", 4))

    heatmap_hw = tuple(mcfg.get("heatmap_out", [img_hw, img_hw]))
    hm_mid_c = int(mcfg.get("hm_mid_c", 128))

    adapter_bottleneck = int(mcfg.get("adapter_bottleneck", 64))
    adapter_drop = float(mcfg.get("adapter_drop", 0.1))
    adapter_in_last = mcfg.get("adapter_in_last", None)
    adapter_in_last = int(adapter_in_last) if adapter_in_last is not None else None
    adapter_mode = str(mcfg.get("adapter_mode", "pre_mhsa"))

    patch = int(mcfg.get("patch_size", 16))
    grid_hw = tuple(mcfg.get("grid_hw", [img_hw // patch, img_hw // patch]))

    ks = int(mcfg.get("ks", 3))
    kt = int(mcfg.get("kt", 3))
    fuse = mcfg.get("fuse", "sum")

    model = SpacecraftHMDSNT(
        num_kp=num_kp,
        num_frames=num_frames,
        img_hw=img_hw,
        hf_name=hf_name,
        adapter_bottleneck=adapter_bottleneck,
        adapter_drop=adapter_drop,
        adapter_in_last=adapter_in_last,
        heatmap_hw=heatmap_hw,
        grid_hw=grid_hw,
        n_register_tokens=nreg,
        ks=ks,
        kt=kt,
        fuse=fuse,
        hm_mid_c=hm_mid_c,
        adapter_mode=adapter_mode,
    )
    return model

# def _subset(ds, take: int, seed: int = 0):
#     if take <= 0 or take >= len(ds):
#         return ds
#     rng = np.random.RandomState(seed)
#     idx = np.arange(len(ds))
#     rng.shuffle(idx)
#     idx = idx[:take].tolist()
#     return Subset(ds, idx)

def make_dataloaders(cfg):
    ds_name = str(cfg["data"]["name"]).lower()
    img_size = int(cfg["data"]["input_size"])

    tfm_crop = build_clip_augmentation(img_size=img_size)
    # tfm_full = build_clip_augmentation(img_size=img_size)

    eval_use_crop = cfg["data"].get("eval_use_crop", None)
    if eval_use_crop is not None:
        eval_use_crop = bool(eval_use_crop)

    base_kwargs = dict(
        gt_folder=cfg["data"]["root"],
        short_mem=int(cfg["data"]["short_mem"]),
        input_size=img_size,
        sampling_rate=int(cfg["data"]["sampling_rate"]),
        clip_stride=int(cfg["data"].get("clip_stride", 0)),
        crop=bool(cfg["data"].get("crop", True)),
        eval_use_crop=eval_use_crop,
        gaussian_sigma_px=float(cfg["data"].get("gaussian_sigma_px", 1.2)),
        heatmap_out=tuple(cfg["model"]["heatmap_out"]),
    )

    if ds_name == "spades":
        train_set = SPADESDataset(mode="train", spatial_transforms=tfm_crop, **base_kwargs)
        val_set   = SPADESDataset(mode="val",  spatial_transforms=None,   **base_kwargs)
        test_set  = SPADESDataset(mode="test", spatial_transforms=None,   **base_kwargs)
    elif ds_name == "spark":
        train_set = SPARKDataset(mode="train", spatial_transforms=tfm_crop, **base_kwargs)
        val_set   = SPARKDataset(mode="val",  spatial_transforms=None,   **base_kwargs)
        test_set  = SPARKDataset(mode="val",  spatial_transforms=None,   **base_kwargs)
    elif ds_name == "vbn":
        train_set = VBNDataset(mode="train", split="train_synth", spatial_transforms=tfm_crop, **base_kwargs)
        val_set   = VBNDataset(mode="val",  split="val_synth", spatial_transforms=None,   **base_kwargs)
        test_set  = VBNDataset(mode="test", split="test_real",  spatial_transforms=None,   **base_kwargs)
    else:
        raise ValueError("data.name must be 'spades', 'vbn', or 'spark'")

    # ---- Random temporal stride (train only) ----
    sr_range = cfg["data"].get("sampling_rate_range", None)
    if sr_range is not None:
        train_set.set_sampling_rate_range(sr_range)
        if is_main_process():
            log(f"[data] sampling_rate_range={sr_range} (train only)")

    # ---- Optional debugging subsets (SYNCED) ----
    train_take = int(cfg["data"].get("train_subset", 0) or 0)
    val_take   = int(cfg["data"].get("val_subset", 0) or 0)
    sub_mode   = str(cfg["data"].get("subset_mode", "random")).lower()  # "random" or "first"
    sub_seed   = int(cfg["data"].get("subset_seed", int(cfg["train"].get("seed", 42))) or 0)

    if train_take > 0:
        # Sync min length first so indices are always in range on every rank
        safe_n = _sync_min_len(len(train_set))
        idxs = _subset_indices(safe_n, train_take, mode=sub_mode, seed=sub_seed) if is_main_process() else []
        idxs = _broadcast_indices(idxs)
        train_set = Subset(train_set, idxs)
        if is_main_process():
            log(f"[Subset] train -> {len(idxs)} samples (mode={sub_mode}, seed={sub_seed}, safe_n={safe_n})")

    if val_take > 0:
        # Sync min length first so indices are always in range on every rank
        safe_n = _sync_min_len(len(val_set))
        idxs = _subset_indices(safe_n, val_take, mode=sub_mode, seed=sub_seed + 1) if is_main_process() else []
        idxs = _broadcast_indices(idxs)
        val_set = Subset(val_set, idxs)
        if is_main_process():
            log(f"[Subset] val   -> {len(idxs)} samples (mode={sub_mode}, seed={sub_seed + 1}, safe_n={safe_n})")

    # ---- Ensure identical lengths across ranks ----
    train_set = _sync_and_trim_dataset(train_set, "train_set")
    val_set   = _sync_and_trim_dataset(val_set,   "val_set")

    world = get_world_size()
    rank  = get_rank()

    train_sampler = DistributedSampler(train_set, num_replicas=world, rank=rank, shuffle=True,  drop_last=True)  if world > 1 else None
    val_sampler   = DistributedSampler(val_set,   num_replicas=world, rank=rank, shuffle=False, drop_last=False) if world > 1 else None

    nw  = int(cfg["train"].get("num_workers", 4))
    bs  = int(cfg["train"]["batch_size"])
    vbs = int(cfg["train"].get("val_batch_size", bs))

    train_loader = DataLoader(
        train_set,
        batch_size=bs,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(nw > 0),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=vbs,
        shuffle=False,
        sampler=val_sampler,
        num_workers=nw,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(nw > 0),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_temporal_chunks(b, int(cfg["data"]["test_T_chunk"])),
    )

    # ---- DDP sanity checks ----
    if is_dist_avail_and_initialized():
        tb = len(train_loader)
        tsl = len(train_sampler) if train_sampler is not None else len(train_set)

        all_tb  = [None] * world
        all_tsl = [None] * world
        dist.all_gather_object(all_tb, tb)
        dist.all_gather_object(all_tsl, tsl)

        if is_main_process():
            log(f"[check] train_batches per rank: {all_tb}")
            log(f"[check] train_sampler_len per rank: {all_tsl}")

        if len(set(all_tb)) != 1 or len(set(all_tsl)) != 1:
            raise RuntimeError(f"Train loader/sampler mismatch across ranks: batches={all_tb}, sampler_len={all_tsl}")

        # ---- Val loader sanity check ----
        vb  = len(val_loader)
        vsl = len(val_sampler) if val_sampler is not None else len(val_set)

        all_vb  = [None] * world
        all_vsl = [None] * world
        dist.all_gather_object(all_vb,  vb)
        dist.all_gather_object(all_vsl, vsl)

        if is_main_process():
            log(f"[check] val_batches per rank: {all_vb}")
            log(f"[check] val_sampler_len per rank: {all_vsl}")

        if len(set(all_vb)) != 1 or len(set(all_vsl)) != 1:
            raise RuntimeError(f"Val loader/sampler mismatch across ranks: batches={all_vb}, sampler_len={all_vsl}")

    print(f"Train loader: {len(train_loader)} batches, Val loader: {len(val_loader)} batches")

    return train_loader, val_loader, train_sampler, val_sampler, test_loader
# -------------------------
# Loss
# -------------------------
def compute_losses(
    out: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
    kps_gt: torch.Tensor,
    hm_gt: torch.Tensor,
    vis: torch.Tensor,
    epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    lcfg = cfg["loss"]
    tau = float(lcfg["tau"])
    target_pow = float(lcfg["heatmap_target_power"])

    w_hm_frame = float(lcfg["w_hm_frame"])
    w_hm_agg   = float(lcfg["w_hm_agg"])
    w_kp       = float(lcfg["w_kp"])
    # w_token    = float(lcfg["w_token"])
    w_token = float(get_w_token(epoch, cfg, total_epochs=int(cfg["train"]["epochs"])))

    # token alignement hyperparams 
    sigma_patch = float(lcfg["sigma_patch"])
    temp_pred = float(lcfg["temp_pred"])
    w_div = float(lcfg["w_div"])


    hm_logits = out["heatmaps"]  # (B,T,K,Hm,Wm)

    parts: Dict[str, torch.Tensor] = {}
    total = hm_logits.new_zeros(())

    # 1) Heatmap KL per-frame
    L_hm_frame = heatmap_kl_div_from_logits_masked(
        hm_logits=hm_logits, hm_target=hm_gt, vis=vis, tau=tau, target_power=target_pow
    )
    total += w_hm_frame * L_hm_frame
    parts["hm_frame"] = L_hm_frame

    # 2) Heatmap KL aggregated across time
    hm_logits_agg = hm_logits.mean(dim=1)         # (B,K,H,W)
    hm_gt_agg     = hm_gt.mean(dim=1)             # (B,K,H,W)
    vis_agg       = (vis.sum(dim=1) > 0).float()  # (B,K)

    L_hm_agg = heatmap_kl_div_from_logits_masked(
        hm_logits=hm_logits_agg.unsqueeze(1),
        hm_target=hm_gt_agg.unsqueeze(1),
        vis=vis_agg.unsqueeze(1),
        tau=tau, target_power=target_pow
    )
    total += w_hm_agg * L_hm_agg
    parts["hm_agg"] = L_hm_agg

    # 3) Keypoints (DSNT output already in [-1,1])
    kps_pred_norm = out["keypoints"]
    assert kps_pred_norm.shape == kps_gt.shape, f"{kps_pred_norm.shape=} vs {kps_gt.shape=}"

    L_kp = smooth_l1_vis_weighted(kps_pred_norm, kps_gt, vis)
    total += w_kp * L_kp
    parts["kp_frame"] = L_kp

    # 4a) Heatmap temporal smoothness (JSD between adjacent-frame heatmaps)
    w_hm_temp = float(lcfg.get("w_hm_temp", 0.0))
    if w_hm_temp > 0:
        L_hm_temp = heatmap_temporal_smoothness(
            hm_logits=hm_logits, vis=vis, tau=tau,
        )
        total += w_hm_temp * L_hm_temp
        parts["hm_temp"] = L_hm_temp
    else:
        parts["hm_temp"] = hm_logits.new_zeros(())

    # 4b) Temporal smoothness on keypoint trajectories (penalize acceleration)
    w_temp = float(lcfg.get("w_temp", 0.0))
    if w_temp > 0:
        L_temp = temporal_smoothness_loss(
            kps_pred_norm, vis,
            order=int(lcfg.get("temporal_order", 2)),
            beta=0.005,
        )
        total += w_temp * L_temp
        parts["temporal"] = L_temp
    else:
        parts["temporal"] = hm_logits.new_zeros(())

    # 4c) Frequency jitter loss on keypoint trajectories
    w_freq = float(lcfg.get("w_freq_jitter", 0.0))
    if w_freq > 0:
        L_freq = frequency_jitter_loss(
            kps_pred_norm, vis,
            cutoff_ratio=float(lcfg.get("freq_cutoff_ratio", 0.4)),
        )
        total += w_freq * L_freq
        parts["freq_jitter"] = L_freq
    else:
        parts["freq_jitter"] = hm_logits.new_zeros(())

    # 4) Token alignment — use direct adapter-layer output (shorter gradient path)
    # "adapter_layer_tokens" is the last adapter's output before frozen layers process it further.
    # Falls back to "adapter_tokens" (final backbone output) only if the key is absent.
    tok = out.get("adapter_layer_tokens", out.get("adapter_tokens", None))
    if w_token > 0 and tok is not None:
        input_size = int(cfg["data"]["input_size"])
        patch_size = int(cfg["model"].get("patch_size", 16))
        Hp = input_size // patch_size
        Wp = input_size // patch_size
        P_expected = Hp * Wp

        P_got = tok.shape[2]
        if P_got < P_expected:
            raise RuntimeError(f"adapter_layer_tokens has too few tokens: got P={P_got}, expected >= {P_expected}")

        if P_got > P_expected:
            # assume [CLS][REG...][PATCH...], keep patches
            tok = tok[:, :, -P_expected:, :]

        detach_proto = (epoch <= 12)
        L_tok = token_spatial_alignment_loss(
            patch_tokens=tok,
            kps_gt_norm=kps_gt,
            vis=vis,
            temp_pred=temp_pred,
            detach_proto=detach_proto,
            sigma_patch=sigma_patch,
            w_div=w_div,      # (optional) slightly lower than 0.05 to reduce interference
        )
        total += w_token * L_tok
        parts["token"] = L_tok
    else:
        parts["token"] = hm_logits.new_zeros(())

    return total, parts



# -------------------------
# Resume helper 
# -------------------------

def load_checkpoint(path: str, model, optimizer, scheduler, scaler, device: str):
    ckpt = torch.load(path, map_location="cpu")

    target = model.module if isinstance(model, DDP) else model
    missing, unexpected = target.load_state_dict(ckpt["model"], strict=False)
    if is_main_process():
        print(f"[resume] missing: {missing}")
        print(f"[resume] unexpected: {unexpected}")


    if optimizer is not None and ckpt.get("optimizer") is not None and len(missing)==0 and len(unexpected)==0:
        optimizer.load_state_dict(ckpt["optimizer"])
        # move optimizer tensors to device
        for st in optimizer.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(device)

    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_ckpt = float(ckpt.get("best_metric", float("inf")))

    plateau = ckpt.get("plateau", {}) or {}
    best_plateau = float(plateau.get("best_plateau", best_ckpt))
    lr_mult = float(plateau.get("lr_mult", 1.0))
    bad_epochs = int(plateau.get("bad_epochs", 0))
    n_drops = int(plateau.get("n_drops", 0))

    ckpt_metric_name = str(ckpt.get("cfg", {}).get("train", {}).get("save_metric", "total"))
    ckpt_metric_mode = str(ckpt.get("cfg", {}).get("train", {}).get("save_metric_mode", "min")).lower()
    return start_epoch, best_ckpt, best_plateau, lr_mult, bad_epochs, n_drops, ckpt_metric_name, ckpt_metric_mode

# -------------------------
# Train / Val  (CORRECT)
# -------------------------
def train_one_epoch(
    model, loader, optimizer, scheduler, scaler, device, cfg, epoch,
    writer: Optional[SummaryWriter], lr_mult: float = 1.0
):
    model.train()

    max_norm = float(cfg["train"].get("grad_clip", 0.0))
    use_amp = bool(cfg["train"].get("amp", True))
    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
    autocast_dtype = torch.bfloat16  # OK on Ampere+; keep as you had

    running = {"total": 0.0, "hm_frame": 0.0, "hm_agg": 0.0, "kp_frame": 0.0, "token": 0.0, "hm_temp": 0.0, "temporal": 0.0, "freq_jitter": 0.0}
    n_batches = len(loader)

    for it, batch in enumerate(loader):
        frames, kps_gt, gt_q, gt_r, vis, hm_gt,  _, crop_xywh = batch

        frames = frames.to(device, non_blocking=True)
        kps_gt = kps_gt.to(device, non_blocking=True)
        vis    = vis.to(device, non_blocking=True)
        hm_gt  = hm_gt.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=dev_type, dtype=autocast_dtype, enabled=use_amp):
            out = model(frames)

        # loss in fp32
        with torch.autocast(device_type=dev_type, enabled=False):
            total_loss, parts = compute_losses(out, cfg, kps_gt=kps_gt, hm_gt=hm_gt, vis=vis, epoch=epoch)

        if is_main_process() and epoch == 1 and it == 0:
            if "adapter_tokens" in out:
                print("adapter_tokens shape:", tuple(out["adapter_tokens"].shape))
            input_size = int(cfg["data"]["input_size"])
            patch_size = int(cfg["model"].get("patch_size", 16))
            print("P_expected:", (input_size // patch_size) * (input_size // patch_size))

        scaler.scale(total_loss).backward()

        if max_norm and max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # ---- SINGLE optimizer step ----
        scaler.step(optimizer)
        scaler.update()

        # ---- SINGLE scheduler step ----
        if scheduler is not None:
            scheduler.step()
            _apply_lr_mult_from_scheduler(optimizer, scheduler, lr_mult)

        running["total"] += float(total_loss.item())
        running["hm_frame"] += float(parts["hm_frame"].item())
        running["hm_agg"] += float(parts["hm_agg"].item())
        running["kp_frame"] += float(parts["kp_frame"].item())
        running["token"] += float(parts["token"].item())
        running["hm_temp"] += float(parts["hm_temp"].item())
        running["temporal"] += float(parts["temporal"].item())
        running["freq_jitter"] += float(parts["freq_jitter"].item())

        # Visualizations (main process, every 5 epochs, first batch)
        if is_main_process() and (epoch % 5 == 0) and (it == 0):
            viz_root = os.path.join(cfg["train"]["out_dir"], "viz_tokens")
            ep_dir = os.path.join(viz_root, f"epoch_{epoch:04d}")
            os.makedirs(ep_dir, exist_ok=True)

            B_viz = min(2, frames.shape[0])

            for b_idx in range(B_viz):
                # Full clip grid (all T frames in one figure)
                grid_path = os.path.join(ep_dir, f"b{b_idx}_clip_grid.png")
                save_kp_overlay_clip_grid(
                    frames=frames,
                    kps_gt_norm=kps_gt,
                    kps_pred_norm=out["keypoints"],  # directly from model
                    vis=vis,
                    save_path=grid_path,
                    b=b_idx,
                    grid_cols=4,
                    max_frames=8,
                    draw_lines=True,
                )


                # Token diagnostics
                diag_path = os.path.join(ep_dir, f"b{b_idx}_token_diag.png")
                metrics = save_token_diagnostics(
                    out=out,
                    frames=frames,
                    kps_gt=kps_gt,
                    kps_pred=out["keypoints"],
                    vis=vis,
                    cfg=cfg,
                    save_path=diag_path,
                    b=b_idx,
                    max_k_show=8,
                )

                if metrics is not None:
                    log(
                        f"  [diag] b={b_idx} "
                        f"proto_cos={metrics['mean_proto_cosine']:.3f} "
                        f"mean_kl={metrics['mean_kl']:.3f}"
                    )

                # Cross-frame prototype transfer diagnostic
                xfer_path = os.path.join(ep_dir, f"b{b_idx}_xfer.png")
                xfer_metrics = save_adapter_feature_maps(
                    out=out,
                    frames=frames,
                    kps_gt=kps_gt,
                    kps_pred=out["keypoints"],
                    vis=vis,
                    cfg=cfg,
                    save_path=xfer_path,
                    b=b_idx,
                    max_k_show=4,
                )
                if xfer_metrics is not None:
                    for kp_name, kp_m in xfer_metrics.items():
                        same_kl = kp_m.get("same_frame_kl")
                        far_kl = kp_m.get("far_frame_kl")
                        far_err = kp_m.get("far_frame_err_px")
                        log(
                            f"  [xfer] b={b_idx} {kp_name}: "
                            f"same_KL={same_kl:.3f} "
                            f"far_KL={far_kl:.3f} "
                            f"far_err={far_err:.1f}px"
                        ) if same_kl is not None and far_kl is not None and far_err is not None else None

            log(f"  [viz] saved → {ep_dir}")

    avg = {k: v / max(1, n_batches) for k, v in running.items()}

    if is_main_process():
        w_tok_now = get_w_token(epoch, cfg, total_epochs=int(cfg["train"]["epochs"]))
        log(
            f"Train Epoch {epoch}: "
            f"loss={avg['total']:.4f} | hmF={avg['hm_frame']:.4f} | hmA={avg['hm_agg']:.4f} "
            f"| kp={avg['kp_frame']:.4f}"
            f"| token={avg['token']:.4f} | w_token={w_tok_now:.4f}"
            f" | hmT={avg['hm_temp']:.4f} | temp={avg['temporal']:.4f}"
            f" | freq={avg['freq_jitter']:.4f}"
        )

        if writer is not None:
            if scheduler is not None:
                writer.add_scalar("LR/lr0", float(optimizer.param_groups[0]["lr"]), epoch)
            writer.add_scalar("Train/total", avg["total"], epoch)
            writer.add_scalar("Train/hm_frame", avg["hm_frame"], epoch)
            writer.add_scalar("Train/hm_agg", avg["hm_agg"], epoch)
            writer.add_scalar("Train/kp", avg["kp_frame"], epoch)
            writer.add_scalar("Train/token", avg["token"], epoch)
            writer.add_scalar("Train/hm_temp", avg["hm_temp"], epoch)
            writer.add_scalar("Train/temporal", avg["temporal"], epoch)
            writer.add_scalar("Train/freq_jitter", avg["freq_jitter"], epoch)

    return avg

@torch.no_grad()
def validate(model, loader, device, cfg, epoch, writer: Optional[SummaryWriter]):
    model.eval()
    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"

    running = {"total": 0.0, "hm_frame": 0.0, "hm_agg": 0.0, "kp_frame": 0.0, "token": 0.0, "hm_temp": 0.0, "temporal": 0.0, "freq_jitter": 0.0}
    n_batches = len(loader)

    # NEW: accumulate PCK
    pck5_sum = 0.0
    pck10_sum = 0.0

    input_size = int(cfg["data"]["input_size"])  # 512
    H_in, W_in = input_size, input_size


    for batch in loader:
        frames, kps_gt, gt_q, gt_r, vis, hm_gt,  _, crop_xywh = batch
        frames = frames.to(device, non_blocking=True)
        kps_gt = kps_gt.to(device, non_blocking=True)
        vis    = vis.to(device, non_blocking=True)
        hm_gt  = hm_gt.to(device, non_blocking=True)

        out = model(frames)

        assert tuple(cfg["model"]["heatmap_out"]) == (out["heatmaps"].shape[-2], out["heatmaps"].shape[-1])

        with torch.autocast(device_type=dev_type, enabled=False):
            val_loss, parts = compute_losses(out, cfg, kps_gt=kps_gt, hm_gt=hm_gt, vis=vis, epoch=epoch)

        running["total"] += float(val_loss.item())
        running["hm_frame"] += float(parts["hm_frame"].item())
        running["hm_agg"] += float(parts["hm_agg"].item())
        running["kp_frame"] += float(parts["kp_frame"].item())
        running["token"] += float(parts["token"].item())
        running["hm_temp"] += float(parts["hm_temp"].item())
        running["temporal"] += float(parts["temporal"].item())
        running["freq_jitter"] += float(parts["freq_jitter"].item())

        # NEW: PCK metrics (crop-input px), consistent with heatmap-normalized coords
        kps_pred_normhm = out["keypoints"]  # DSNT: [-1,1] on heatmap grid
        heatmap_hw = tuple(cfg["model"]["heatmap_out"])
        input_size = int(cfg["data"]["input_size"])

        pck5  = pck_crop_from_normhm(kps_pred_normhm, kps_gt, vis, input_size, heatmap_hw, thr_px=5.0)
        pck10 = pck_crop_from_normhm(kps_pred_normhm, kps_gt, vis, input_size, heatmap_hw, thr_px=10.)

        pck5_sum  += float(pck5.item())
        pck10_sum += float(pck10.item())

    # ---- Aggregate metrics across all ranks ----
    if is_dist_avail_and_initialized():
        keys_order = ["total", "hm_frame", "hm_agg", "kp_frame", "token", "hm_temp", "temporal", "freq_jitter"]
        arr = torch.tensor(
            [running[k] for k in keys_order] + [pck5_sum, pck10_sum, float(n_batches)],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(arr, op=dist.ReduceOp.SUM)
        arr = arr.tolist()
        for i, k in enumerate(keys_order):
            running[k] = arr[i]
        pck5_sum  = arr[len(keys_order)]
        pck10_sum = arr[len(keys_order) + 1]
        n_batches = max(1, int(round(arr[len(keys_order) + 2])))

    avg = {k: v / max(1, n_batches) for k, v in running.items()}

    # NEW: averaged over batches
    avg["pck5_crop"]  = pck5_sum / max(1, n_batches)
    avg["pck10_crop"] = pck10_sum / max(1, n_batches)

    if is_main_process():
        log(
            f"Val   Epoch {epoch}: "
            f"loss={avg['total']:.4f} | hmF={avg['hm_frame']:.4f} | hmA={avg['hm_agg']:.4f} "
            f"| kp={avg['kp_frame']:.4f} | token={avg['token']:.4f} "
            f"| hmT={avg['hm_temp']:.4f} | temp={avg['temporal']:.4f} "
            f"| freq={avg['freq_jitter']:.4f} "
            f"| PCK@10(crop)={avg['pck10_crop']:.4f} PCK@5(crop)={avg['pck5_crop']:.4f}"
        )
        if writer is not None:
            writer.add_scalar("Val/PCK10_crop", avg["pck10_crop"], epoch)
            writer.add_scalar("Val/PCK5_crop",  avg["pck5_crop"], epoch)
            writer.add_scalar("Val/total", avg["total"], epoch)
            writer.add_scalar("Val/hm_frame", avg["hm_frame"], epoch)
            writer.add_scalar("Val/hm_agg", avg["hm_agg"], epoch)
            writer.add_scalar("Val/kp", avg["kp_frame"], epoch)
            writer.add_scalar("Val/token", avg["token"], epoch)
            writer.add_scalar("Val/hm_temp", avg["hm_temp"], epoch)
            writer.add_scalar("Val/temporal", avg["temporal"], epoch)
            writer.add_scalar("Val/freq_jitter", avg["freq_jitter"], epoch)

    return avg


# -------------------------
# Run  (CORRECT plateau + best)
# -------------------------
def run_training(cfg, logger: Optional[Callable[[str], None]] = None):
    if logger is None:
        logger = _default_logger

    def _log(s: str):
        if is_main_process():
            logger(s)

    setup_distributed(backend=cfg["train"].get("ddp_backend", "nccl"))

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"

    seed_everything(int(cfg["train"].get("seed", 42)) + get_rank())

    train_loader, val_loader, train_sampler, val_sampler, _ = make_dataloaders(cfg)

    model = build_model(cfg).to(device)
    freeze_all_but_adapters_and_heads(model)
    make_trainable_params_contiguous(model)

    if get_world_size() > 1:
        use_cuda = str(device).startswith("cuda")
        model = DDP(
            model,
            device_ids=[local_rank] if use_cuda else None,
            output_device=local_rank if use_cuda else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    lr = float(cfg["train"]["lr"])
    wd = float(cfg["train"].get("weight_decay", 0.05))
    params = [p for p in (model.module.parameters() if isinstance(model, DDP) else model.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.999))

    lr_schedule = str(cfg["train"].get("lr_schedule", "cosine")).lower()
    if lr_schedule == "cosine_restart":
        scheduler = build_warmup_cosine_restart_scheduler(
            optimizer=optimizer,
            epochs=int(cfg["train"]["epochs"]),
            steps_per_epoch=len(train_loader),
            warmup_epochs=int(cfg["train"].get("warmup_epochs", 0)),
            lr_min=float(cfg["train"].get("lr_min", 1e-5)),
            restart_period_epochs=int(cfg["train"].get("lr_restart_period", 25)),
            restart_mult=int(cfg["train"].get("lr_restart_mult", 1)),
        )
    else:
        scheduler = build_warmup_cosine_scheduler(
            optimizer=optimizer,
            epochs=int(cfg["train"]["epochs"]),
            steps_per_epoch=len(train_loader),
            warmup_epochs=int(cfg["train"].get("warmup_epochs", 0)),
            lr_min=float(cfg["train"].get("lr_min", 1e-6)),
        )

    scaler = torch.amp.GradScaler(
        device="cuda" if torch.cuda.is_available() else "cpu",
        enabled=bool(cfg["train"].get("amp", True))
    )

    writer: Optional[SummaryWriter] = None
    out_dir = cfg["train"]["out_dir"]
    tb_root = cfg["train"]["tb_logdir"]
    metric_mode = str(cfg["train"].get("save_metric_mode", "min")).lower()
    metric_name = str(cfg["train"].get("save_metric", "total"))


    # --- in run_training() ---
    resume_path = cfg["train"].get("resume", None)
    start_epoch = 1

    if resume_path:
        assert os.path.isfile(resume_path), f"Resume path not found: {resume_path}"
        if is_main_process():
            _log(f"[resume] loading {resume_path}")

        start_epoch, best_ckpt, best_plateau, lr_mult, bad_epochs, n_drops, ckpt_mname, ckpt_mmode = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device
        )

        # IMPORTANT: scheduler is already restored; just re-apply lr_mult
        _apply_lr_mult_from_scheduler(optimizer, scheduler, lr_mult)

        if is_main_process():
            _log(
                f"[resume] start_epoch={start_epoch} best_ckpt={best_ckpt:.6f} "
                f"best_plateau={best_plateau:.6f} lr_mult={lr_mult:.6f} "
                f"bad_epochs={bad_epochs} drops={n_drops}"
            )

        if (ckpt_mname != metric_name) or (ckpt_mmode != metric_mode):
            best_ckpt = -float("inf") if metric_mode == "max" else float("inf")
            best_plateau = best_ckpt
            bad_epochs = 0
            n_drops = 0
            lr_mult = 1.0
            _log(f"[resume] metric changed ({ckpt_mname}/{ckpt_mmode} -> {metric_name}/{metric_mode}); reset best baselines.")
    else:
        best_ckpt = -float("inf") if metric_mode == "max" else float("inf")
        best_plateau = best_ckpt
        lr_mult = 1.0
        bad_epochs = 0
        n_drops = 0

    # sync to all ranks
    lr_mult = _sync_float(lr_mult)

    ft = cfg["train"].get("finetune", {}) or {}
    ft_enabled = bool(ft.get("enabled", False))
    ft_start_epoch = int(ft.get("start_epoch", 10**9))

    # Optional overrides
    ft_lr = ft.get("lr_override", None)
    ft_wd = ft.get("wd_override", None)
    ft_disable_plateau = bool(ft.get("disable_plateau", True))
    ft_disable_scheduler = bool(ft.get("disable_scheduler", True))

    if is_main_process():
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(tb_root, exist_ok=True)
        run_name = f"exp_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        tb_dir = os.path.join(tb_root, run_name)
        writer = SummaryWriter(tb_dir)
        _log(f"[tb] logging -> {tb_dir}")

    if is_dist():
        dist.barrier()

    # ---- Bests & plateau state ----
    assert metric_mode in ("min", "max")

    # NOTE: lr_mult / bad_epochs / n_drops are set by the resume block above (or
    # by the else-branch for fresh runs). Do NOT reset them here — that was a bug
    # that discarded the plateau state loaded from a checkpoint.

    plateau_patience  = int(cfg["train"].get("plateau_patience", 4))
    plateau_factor    = float(cfg["train"].get("plateau_factor", 0.3))
    plateau_min_delta = float(cfg["train"].get("plateau_min_delta", 1e-4))
    plateau_max_drops = int(cfg["train"].get("plateau_max_drops", 2))


    for epoch in range(start_epoch, int(cfg["train"]["epochs"]) + 1):
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)

        # ---- Fine-tune mode switch ----
        if ft_enabled and epoch >= ft_start_epoch:
            # 1) disable plateau drops
            if ft_disable_plateau:
                plateau_max_drops = 0

            # 2) disable scheduler & use constant lr
            if ft_disable_scheduler and (scheduler is not None):
                scheduler = None
                if is_main_process():
                    _log(f"[finetune] scheduler disabled at epoch {epoch}")

            # 3) override lr / wd (once is enough, but safe to re-apply)
            if ft_lr is not None:
                for g in optimizer.param_groups:
                    g["lr"] = float(ft_lr) * float(lr_mult)
            if ft_wd is not None:
                for g in optimizer.param_groups:
                    g["weight_decay"] = float(ft_wd)

        train_stats = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, cfg, epoch, writer,
            lr_mult=lr_mult
        )
        val_stats = validate(model, val_loader, device, cfg, epoch, writer)

        # ---- Rank0: checkpoint + plateau ----
        _stop = False
        if is_main_process():
            # score = float(val_stats.get("total", 1e9))
            default_score = -1e9 if metric_mode == "max" else 1e9
            score = float(val_stats.get(metric_name, default_score))
            # Save-best logic
            if metric_mode == "max":
                is_better = score > (best_ckpt + 1e-12)
            else:
                is_better = score < (best_ckpt - 1e-12)

            # (A) Save best checkpoint (exact)
            if is_better and epoch > 5:
                best_ckpt = score
                path = os.path.join(out_dir, f"best_epoch_{epoch}.pt")
                torch.save(
                    {
                        "cfg": cfg,
                        "epoch": epoch,
                        "model": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "scaler": scaler.state_dict() if scaler is not None else None,
                        "best_metric": best_ckpt,
                        "plateau": {
                            "best_plateau": best_plateau,
                            "lr_mult": lr_mult,
                            "bad_epochs": bad_epochs,
                            "n_drops": n_drops,
                        },
                    },
                    path,
                )
                _log(f"✓ Saved best -> {path} (score={best_ckpt:.4f})")

            if epoch > 5:
                # (B) Plateau detection (min-delta aware)
                if metric_mode == "max":
                    improved = score > (best_plateau + plateau_min_delta)
                else:
                    improved = score < (best_plateau - plateau_min_delta)

                if improved:
                    best_plateau = score
                    bad_epochs = 0
                else:
                    bad_epochs += 1

                if (bad_epochs >= plateau_patience) and (n_drops < plateau_max_drops):
                    lr_mult *= plateau_factor
                    n_drops += 1
                    bad_epochs = 0
                    _log(f"[plateau] no improve for {plateau_patience} ep -> lr_mult={lr_mult:.6f} (drops={n_drops})")

            # ---- Early stopping ----
            early_stop_patience = int(cfg["train"].get("early_stop_patience", 0))
            early_stop_min_lr   = float(cfg["train"].get("early_stop_min_lr", 0.0))
            if early_stop_patience > 0 and n_drops >= plateau_max_drops and bad_epochs >= early_stop_patience:
                _log(f"[early_stop] No improvement for {bad_epochs} epochs after all LR drops → stopping at epoch {epoch}.")
                _stop = True
            if not _stop and early_stop_min_lr > 0:
                eff_lr = min(g["lr"] for g in optimizer.param_groups)
                if eff_lr < early_stop_min_lr:
                    _log(f"[early_stop] Effective LR {eff_lr:.2e} < min {early_stop_min_lr:.2e} → stopping at epoch {epoch}.")
                    _stop = True

        # ---- Sync lr_mult and stop flag to all ranks ----
        lr_mult = _sync_float(lr_mult)
        if is_dist_avail_and_initialized():
            _stop_list = [_stop]
            dist.broadcast_object_list(_stop_list, src=0)
            _stop = _stop_list[0]

        if is_dist():
            dist.barrier()

        if _stop:
            break

    if is_main_process():
        _log(f"Training done. Best={best_ckpt:.4f}")
        if writer is not None:
            writer.flush()
            writer.close()

    cleanup_distributed()

# Reuse if things went south
# # -------------------------
# # Train / Val
# # -------------------------
# def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, cfg, epoch, writer: Optional[SummaryWriter]):
#     model.train()

#     max_norm = float(cfg["train"].get("grad_clip", 0.0))
#     use_amp = bool(cfg["train"].get("amp", True))
#     dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
#     autocast_dtype = torch.bfloat16 if dev_type == "cuda" else torch.bfloat16

#     running = {"total": 0.0, "hm_frame": 0.0, "hm_agg": 0.0, "kp_frame": 0.0, "token": 0.0}
#     n_batches = len(loader)

#     for it, batch in enumerate(loader):
#         frames, kps_gt, gt_q, gt_r, vis, hm_gt,  _, crop_xywh = batch

#         frames = frames.to(device, non_blocking=True)
#         kps_gt = kps_gt.to(device, non_blocking=True)
#         vis    = vis.to(device, non_blocking=True)
#         hm_gt  = hm_gt.to(device, non_blocking=True)

#         optimizer.zero_grad(set_to_none=True)

#         with torch.autocast(device_type=dev_type, dtype=autocast_dtype, enabled=use_amp):
#             out = model(frames)

#         with torch.autocast(device_type=dev_type, enabled=False):
#             total_loss, parts = compute_losses(out, cfg, kps_gt=kps_gt, hm_gt=hm_gt, vis=vis)

#         if is_main_process() and epoch == 1 and it == 0:
#             print("adapter_tokens shape:", tuple(out["adapter_tokens"].shape))
#             input_size = int(cfg["data"]["input_size"])
#             patch_size = int(cfg["model"].get("patch_size", 16))
#             print("P_expected:", (input_size//patch_size)*(input_size//patch_size))

#         scaler.scale(total_loss).backward()

#         if max_norm and max_norm > 0:
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

#         scaler.step(optimizer)
#         scaler.update()
#         if scheduler is not None:
#             scheduler.step()

#         running["total"] += float(total_loss.item())
#         running["hm_frame"] += float(parts["hm_frame"].item())
#         running["hm_agg"] += float(parts["hm_agg"].item())
#         running["kp_frame"] += float(parts["kp_frame"].item())
#         running["token"] += float(parts["token"].item())

#         # visualizations: ONLY main
#         if is_main_process() and (epoch % 5 == 0) and (it == 0):
#             viz_root = os.path.join(cfg["train"]["out_dir"], "viz_tokens")
#             ep_dir = os.path.join(viz_root, f"epoch_{epoch:04d}")
#             os.makedirs(ep_dir, exist_ok=True)

#             B, T = frames.shape[0], frames.shape[1]
#             t_sel = T // 2
#             n_show = min(4, B)

#             for b_idx in range(n_show):
#                 kp_path = os.path.join(ep_dir, f"b{b_idx}_t{t_sel}_kp.png")
#                 tok_path = os.path.join(ep_dir, f"b{b_idx}_t{t_sel}_tok.png")

#                 save_kp_overlay_bt(
#                     frames=frames,
#                     kps_gt_norm=kps_gt,
#                     hm_logits=out["heatmaps"],
#                     vis=vis,
#                     cfg=cfg,
#                     save_path=kp_path,
#                     b=b_idx,
#                     t=t_sel,
#                 )
#                 save_token_similarity_overlay_bt(
#                     out=out, frames=frames, kps_gt=kps_gt, vis=vis,
#                     cfg=cfg, save_path=tok_path, b=b_idx, t=t_sel,
#                     sigma_patch=1.25,
#                 )
#             log(f"[viz] saved -> {ep_dir}")

#     avg = {k: v / max(1, n_batches) for k, v in running.items()}

#     if is_main_process():
#         log(
#             f"Train Epoch {epoch}: "
#             f"loss={avg['total']:.4f} | hmF={avg['hm_frame']:.4f} | hmA={avg['hm_agg']:.4f} "
#             f"| kp={avg['kp_frame']:.4f} | token={avg['token']:.4f}"
#         )
#         if writer is not None:
#             writer.add_scalar("Train/total", avg["total"], epoch)
#             writer.add_scalar("Train/hm_frame", avg["hm_frame"], epoch)
#             writer.add_scalar("Train/hm_agg", avg["hm_agg"], epoch)
#             writer.add_scalar("Train/kp", avg["kp_frame"], epoch)
#             writer.add_scalar("Train/token", avg["token"], epoch)

#     return avg


# @torch.no_grad()
# def validate(model, loader, device, cfg, epoch, writer: Optional[SummaryWriter]):
#     # Only main runs validation to avoid duplicated work/logs.
#     if is_dist() and not is_main_process():
#         return {}

#     model.eval()
#     dev_type = "cuda" if str(device).startswith("cuda") else "cpu"

#     running = {"total": 0.0, "hm_frame": 0.0, "hm_agg": 0.0, "kp_frame": 0.0, "token": 0.0}
#     n_batches = len(loader)

#     for batch in loader:
#         frames, kps_gt, gt_q, gt_r, vis, hm_gt,  _, crop_xywh = batch
#         frames = frames.to(device, non_blocking=True)
#         kps_gt = kps_gt.to(device, non_blocking=True)
#         vis    = vis.to(device, non_blocking=True)
#         hm_gt  = hm_gt.to(device, non_blocking=True)

#         out = model(frames)

#         with torch.autocast(device_type=dev_type, enabled=False):
#             val_loss, parts = compute_losses(out, cfg, kps_gt=kps_gt, hm_gt=hm_gt, vis=vis)

#         running["total"] += float(val_loss.item())
#         running["hm_frame"] += float(parts["hm_frame"].item())
#         running["hm_agg"] += float(parts["hm_agg"].item())
#         running["kp_frame"] += float(parts["kp_frame"].item())
#         running["token"] += float(parts["token"].item())

#     avg = {k: v / max(1, n_batches) for k, v in running.items()}

#     if is_main_process():
#         log(
#             f"Val   Epoch {epoch}: "
#             f"loss={avg['total']:.4f} | hmF={avg['hm_frame']:.4f} | hmA={avg['hm_agg']:.4f} "
#             f"| kp={avg['kp_frame']:.4f} | token={avg['token']:.4f}"
#         )
#         if writer is not None:
#             writer.add_scalar("Val/total", avg["total"], epoch)
#             writer.add_scalar("Val/hm_frame", avg["hm_frame"], epoch)
#             writer.add_scalar("Val/hm_agg", avg["hm_agg"], epoch)
#             writer.add_scalar("Val/kp", avg["kp_frame"], epoch)
#             writer.add_scalar("Val/token", avg["token"], epoch)

#     return avg

            
# # -------------------------
# # Run
# # -------------------------
# def run_training(cfg, logger: Optional[Callable[[str], None]] = None):
#     """
#     logger: callable that takes a string. If None, prints only on rank0.
#     """
#     if logger is None:
#         logger = _default_logger

#     # override log() usage by closing over logger
#     def _log(s: str):
#         if is_main_process():
#             logger(s)

#     setup_distributed(backend=cfg["train"].get("ddp_backend", "nccl"))

#     local_rank = int(os.environ.get("LOCAL_RANK", 0))
#     device = cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
#     if str(device).startswith("cuda") and torch.cuda.is_available():
#         torch.cuda.set_device(local_rank)
#         device = f"cuda:{local_rank}"

            
#     seed_everything(int(cfg["train"].get("seed", 42)) + get_rank())

#     train_loader, val_loader, train_sampler, val_sampler, _ = make_dataloaders(cfg)

#     model = build_model(cfg).to(device)
#     freeze_all_but_adapters_and_heads(model)
#     make_trainable_params_contiguous(model)

#     if get_world_size() > 1:
#         use_cuda = str(device).startswith("cuda")
#         model = DDP(
#             model,
#             device_ids=[local_rank] if use_cuda else None,
#             output_device=local_rank if use_cuda else None,
#             broadcast_buffers=False,
#             find_unused_parameters=False,
#         )

#     lr = float(cfg["train"]["lr"])
#     wd = float(cfg["train"].get("weight_decay", 0.05))

#     params = [p for p in (model.module.parameters() if isinstance(model, DDP) else model.parameters()) if p.requires_grad]
#     optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.999))

#     scheduler = build_warmup_cosine_scheduler(
#         optimizer=optimizer,
#         epochs=int(cfg["train"]["epochs"]),
#         steps_per_epoch=len(train_loader),
#         warmup_epochs=int(cfg["train"].get("warmup_epochs", 0)),
#         lr_min=float(cfg["train"].get("lr_min", 1e-6)),
#     )

#     scaler = torch.amp.GradScaler(
#         device="cuda" if torch.cuda.is_available() else "cpu",
#         enabled=bool(cfg["train"].get("amp", True))
#     )

#     writer: Optional[SummaryWriter] = None
#     out_dir = cfg["train"]["out_dir"]
#     tb_root = cfg["train"]["tb_logdir"]

#     if is_main_process():
#         os.makedirs(out_dir, exist_ok=True)
#         os.makedirs(tb_root, exist_ok=True)

#         run_name = f"exp_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
#         tb_dir = os.path.join(tb_root, run_name)
#         writer = SummaryWriter(tb_dir)
#         _log(f"[tb] logging -> {tb_dir}")

#     if is_dist():
#         dist.barrier()

#     best = float("inf")

#     for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
#         if isinstance(train_sampler, DistributedSampler):
#             train_sampler.set_epoch(epoch)

#         train_stats = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, cfg, epoch, writer)
#         val_stats = validate(model, val_loader, device, cfg, epoch, writer)

#         if is_main_process():
#             score = float(val_stats.get("total", 1e9))
#             if score < best:
#                 best = score
#                 path = os.path.join(out_dir, f"best_epoch_{epoch}.pt")
#                 torch.save(
#                     {
#                         "cfg": cfg,
#                         "epoch": epoch,
#                         "model": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
#                         "optimizer": optimizer.state_dict(),
#                         "best_metric": best,
#                     },
#                     path,
#                 )
#                 _log(f"✓ Saved best -> {path} (score={best:.4f})")

#         if is_dist():
#             dist.barrier()

#     if is_main_process():
#         _log(f"Training done. Best={best:.4f}")
#         if writer is not None:
#             writer.flush()
#             writer.close()

#     cleanup_distributed()