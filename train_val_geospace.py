# train_val.py  (CLEAN, DDP-safe, GeoSpaceFormer-aligned, Theseus-free)
import os
import math
import random
import datetime
import warnings
from typing import Dict, Any, Tuple, Optional, Callable

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

from dataset_geospace import SPADESDataset, SPARKDataset
from transforms import build_transforms
from model_video_geospace import GeoSpaceFormer

from losses import (
    heatmap_kl_div_from_logits_masked,
    smooth_l1_vis_weighted,
    # token_spatial_alignment_loss,
    ray_token_contrastive_loss,
    gaussian_nll_kp_px,
    add_l1_pose_loss,
    kp_attn_kl_to_gaussian_patches,
)

from utils_geospace import (
    build_warmup_cosine_scheduler,
    freeze_all_but_adapters_and_heads,
    collate_temporal_chunks,
    save_kp_overlay_bt,
    get_w_token,
    save_token_similarity_diagnostics_bt,
    pck_crop_from_normhm,
    quat_to_rotmat,
    get_w_scheduled,
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# -------------------------
# DDP helpers
# -------------------------
def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0

def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1

def is_main_process() -> bool:
    return get_rank() == 0

def log(s: str):
    if is_main_process():
        print(s, flush=True)

def setup_distributed(backend: str = "nccl"):
    if is_dist():
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend=backend, init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

def _tb_add_dict(writer, prefix: str, d: dict, step: int):
    if writer is None or d is None:
        return
    for k, v in d.items():
        if v is None:
            continue
        try:
            writer.add_scalar(f"{prefix}/{k}", float(v), step)
        except Exception:
            # Skip non-scalar / non-castable values
            pass

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
    if not is_dist_avail_and_initialized():
        return float(x)
    obj = [float(x) if is_main_process() else None]
    dist.broadcast_object_list(obj, src=0)
    return float(obj[0])

def _apply_lr_mult_from_scheduler(optimizer, scheduler, lr_mult: float):
    if scheduler is None:
        for g in optimizer.param_groups:
            g["lr"] = float(g["lr"]) * float(lr_mult)
        return
    base_lrs = scheduler.get_last_lr()
    for g, base in zip(optimizer.param_groups, base_lrs):
        g["lr"] = float(base) * float(lr_mult)

def make_trainable_params_contiguous(m: nn.Module):
    for p in m.parameters():
        if p.requires_grad and not p.is_contiguous():
            p.data = p.data.contiguous()


# -------------------------
# Subset helpers (DDP-safe)
# -------------------------
def _subset_indices(n: int, take: int, mode: str = "random", seed: int = 0):
    take = int(take)
    if take <= 0 or take >= n:
        return list(range(n))
    mode = str(mode).lower()
    if mode == "first":
        return list(range(take))
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return idx[:take].tolist()

def _broadcast_indices(idxs):
    if not is_dist_avail_and_initialized():
        return idxs
    obj_list = [idxs if get_rank() == 0 else None]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]

def _sync_and_trim_dataset(ds, name: str):
    if not is_dist_avail_and_initialized():
        return ds
    local_len = len(ds)
    lens = [None] * get_world_size()
    dist.all_gather_object(lens, local_len)
    if len(set(lens)) == 1:
        return ds
    min_len = min(lens)
    if is_main_process():
        log(f"[DDP-sync] {name} length mismatch across ranks: {lens} -> trimming to {min_len}")
    if isinstance(ds, Subset):
        new_indices = list(ds.indices)[:min_len]
        return Subset(ds.dataset, new_indices)
    return Subset(ds, list(range(min_len)))


# -------------------------
# Model pts3d loader
# -------------------------
import json

def _load_model_keypoints_3d(cfg) -> torch.Tensor:
    pts_path = cfg["data"]["root"].rstrip("/") + "/points.json"
    with open(pts_path, "r") as f:
        j = json.load(f)
    pts = torch.tensor(j["keypoints"], dtype=torch.float32)  # (K,3)
    return pts


def build_model(cfg) -> nn.Module:
    dcfg = cfg["data"]
    mcfg = cfg.get("model", {})

    num_kp = int(dcfg.get("num_kp", 8))
    short_mem = int(dcfg["short_mem"])
    sampling_rate = int(dcfg["sampling_rate"])
    num_frames = int(math.ceil(short_mem / sampling_rate))

    img_hw = int(dcfg.get("input_size", 224))
    hf_name = mcfg.get("hf_name", "facebook/dinov3-vitl16-pretrain-lvd1689m")

    patch = int(mcfg.get("patch_size", 16))
    grid_hw = tuple(mcfg.get("grid_hw", [img_hw // patch, img_hw // patch]))
    nreg = mcfg.get("n_register_tokens", None)
    nreg = int(nreg) if nreg is not None else None

    heatmap_hw = tuple(mcfg.get("heatmap_out", [56, 56]))
    hm_mid_c = int(mcfg.get("hm_mid_c", 128))

    neck_heads = int(mcfg.get("neck_heads", mcfg.get("num_heads", 8)))
    neck_drop = float(mcfg.get("neck_drop", 0.1))

    adapter_bottleneck = int(mcfg.get("adapter_bottleneck", 64))
    adapter_drop = float(mcfg.get("adapter_drop", 0.1))
    adapter_in_last = mcfg.get("adapter_in_last", None)
    adapter_in_last = int(adapter_in_last) if adapter_in_last is not None else None
    ks = int(mcfg.get("ks", 3))
    kt = int(mcfg.get("kt", 3))
    fuse = str(mcfg.get("fuse", "sum"))

    pts3d = _load_model_keypoints_3d(cfg)

    rcsa_ray_bias_scale = float(mcfg.get("rcsa_ray_bias_scale", 0.0))
    kp_ray_bias_scale = float(mcfg.get("kp_ray_bias_scale", 2.0))

    rcsa_bins = int(mcfg.get("rcsa_bins", 64))
    rcsa_neighbor_bins = int(mcfg.get("rcsa_neighbor_bins", 6))
    rcsa_use_temporal = bool(mcfg.get("rcsa_use_temporal", True))
    rcsa_topk_kv = mcfg.get("rcsa_topk_kv", 64)
    rcsa_topk_kv = int(rcsa_topk_kv) if rcsa_topk_kv is not None else None
    rcsa_max_q = mcfg.get("rcsa_max_q", None)
    rcsa_max_q = int(rcsa_max_q) if rcsa_max_q is not None else None

    kp_token_heads = int(mcfg.get("kp_token_heads", 8))
    kp_token_drop = float(mcfg.get("kp_token_drop", 0.1))

    #pose head configs 
    enable_pose_head = bool(mcfg.get("enable_pose_head", False))
    pose_hidden = int(mcfg.get("pose_hidden", 512))
    unc_mode = str(mcfg.get("unc_mode", "diag"))
    unc_hidden = int(mcfg.get("unc_hidden", 256))    

    model = GeoSpaceFormer(
        model_pts_3d=pts3d,
        num_kp=num_kp,
        num_frames=num_frames,
        hf_name=hf_name,
        grid_hw=grid_hw,
        heatmap_hw=heatmap_hw,
        hm_mid_c=hm_mid_c,

        adapter_bottleneck=adapter_bottleneck,
        adapter_drop=adapter_drop,
        adapter_in_last=adapter_in_last,
        n_register_tokens=nreg,
        ks=ks, kt=kt, fuse=fuse,

        neck_heads=neck_heads,
        neck_drop=neck_drop,
        rcsa_bins=rcsa_bins,
        rcsa_neighbor_bins=rcsa_neighbor_bins,
        rcsa_use_temporal=rcsa_use_temporal,
        rcsa_topk_kv=rcsa_topk_kv,
        rcsa_max_q=rcsa_max_q,
        rcsa_ray_bias_scale=rcsa_ray_bias_scale,

        kp_token_heads=kp_token_heads,
        kp_token_drop=kp_token_drop,
        kp_ray_bias_scale=kp_ray_bias_scale,

        unc_mode=unc_mode,
        unc_hidden=unc_hidden,
        enable_pose_head=enable_pose_head,
        pose_hidden=pose_hidden,
    )
    return model


# -------------------------
# FULL-K fetch helper
# -------------------------
def _unwrap_dataset(ds):
    while isinstance(ds, Subset):
        ds = ds.dataset
    return ds

def get_K_full_from_loader(loader) -> torch.Tensor:
    base = _unwrap_dataset(loader.dataset)
    assert hasattr(base, "K_full") and (base.K_full is not None), \
        "Dataset has no K_full. Did you pass cam_intrinsics_json in make_dataloaders?"
    return base.K_full.float()  # (3,3)


# -------------------------
# DataLoaders
# -------------------------
def make_dataloaders(cfg):
    ds_name = str(cfg["data"]["name"]).lower()
    img_size = int(cfg["data"]["input_size"])

    tfm_crop = build_transforms(img_size=img_size)

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
        cam_intrinsics_json=str(cfg["data"].get("cam_intrinsics_json", "")) or None,
    )

    if ds_name == "spades":
        train_set = SPADESDataset(mode="train", spatial_transforms=tfm_crop, **base_kwargs)
        val_set   = SPADESDataset(mode="val",   spatial_transforms=None,   **base_kwargs)
        test_set  = SPADESDataset(mode="test",  spatial_transforms=None,   **base_kwargs)
    elif ds_name == "spark":
        train_set = SPARKDataset(mode="train", spatial_transforms=tfm_crop, **base_kwargs)
        val_set   = SPARKDataset(mode="val",   spatial_transforms=None,   **base_kwargs)
        test_set  = SPARKDataset(mode="val",   spatial_transforms=None,   **base_kwargs)
    else:
        raise ValueError("data.name must be 'spades' or 'spark'")

    # optional debugging subsets (SYNCED)
    train_take = int(cfg["data"].get("train_subset", 0) or 0)
    val_take   = int(cfg["data"].get("val_subset", 0) or 0)
    sub_mode   = str(cfg["data"].get("subset_mode", "random")).lower()
    sub_seed   = int(cfg["data"].get("subset_seed", int(cfg["train"].get("seed", 42))) or 0)

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

    train_set = _sync_and_trim_dataset(train_set, "train_set")
    val_set   = _sync_and_trim_dataset(val_set,   "val_set")

    world = get_world_size()
    rank  = get_rank()

    train_sampler = DistributedSampler(train_set, num_replicas=world, rank=rank, shuffle=True, drop_last=True) if world > 1 else None
    val_sampler   = None

    if world > 1 and not is_main_process():
        val_set = Subset(val_set, [])

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

    if is_main_process():
        log(f"Train loader: {len(train_loader)} batches, Val loader: {len(val_loader)} batches")

    return train_loader, val_loader, train_sampler, val_sampler, test_loader


# -------------------------
# Loss
# -------------------------
# -------------------------
# Loss (FIXED: always use out["rays"], no recomputation)
# -------------------------
def compute_losses(
    out: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
    kps_gt: torch.Tensor,            # (B,T,K,2) [-1,1] heatmap-norm
    hm_gt: torch.Tensor,             # (B,T,K,Hm,Wm)
    vis: torch.Tensor,               # (B,T,K)
    epoch: int,
    cam_K: Optional[torch.Tensor] = None,            # (B,3,3)  (kept for pose/add if needed elsewhere)
    crop_xywh: Optional[torch.Tensor] = None,        # (B,T,4)  (kept for model forward; NOT for token loss)
    full_hw: Optional[Tuple[int, int]] = None,       # (H_full,W_full)
    kps_gt_full_px: Optional[torch.Tensor] = None,   # (B,T,K,2) full px
    R_gt: Optional[torch.Tensor] = None,             # (B,T,3,3)
    t_gt: Optional[torch.Tensor] = None,             # (B,T,3)
    pts3d_model: Optional[torch.Tensor] = None,      # (K,3)
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    lcfg = cfg["loss"]
    tau = float(lcfg["tau"])
    target_pow = float(lcfg["heatmap_target_power"])

    w_hm_frame = float(lcfg["w_hm_frame"])
    w_hm_agg   = float(lcfg["w_hm_agg"])
    w_kp       = float(lcfg["w_kp"])
    w_token    = float(get_w_token(epoch, cfg, total_epochs=int(cfg["train"]["epochs"])))

    base_w_nll = float(lcfg.get("w_nll", 1.0))
    w_nll = float(get_w_scheduled(epoch, base_w_nll, lcfg.get("w_nll_sched", None)))
    w_add   = float(lcfg.get("w_add", 5.0))
    sigma_floor = float(get_w_scheduled(
        epoch,
        float(lcfg.get("sigma_floor_px", 1.0)),
        lcfg.get("sigma_floor_sched", None)
    ))

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

    # 3) KP regression (optional)
    kps_pred_norm = out.get("mu_norm")
    kps_pred_norm = out.get("keypoints") if kps_pred_norm is None else kps_pred_norm
    if kps_pred_norm is not None:
        L_kp = smooth_l1_vis_weighted(kps_pred_norm, kps_gt, vis)
        total += w_kp * L_kp
        parts["kp_frame"] = L_kp
    else:
        parts["kp_frame"] = hm_logits.new_zeros(())

    # 4) Probabilistic NLL in FULL pixel coords (if model provides)
    if (w_nll > 0) and ("mu_px_full" in out) and ("sigma" in out) and (kps_gt_full_px is not None):
        L_nll = gaussian_nll_kp_px(
            mu_px=out["mu_px_full"],
            gt_px=kps_gt_full_px,
            sigma=out["sigma"],
            vis=vis,
            sigma_floor=sigma_floor,
            # full_hw=full_hw,  # <-- IMPORTANT (you already pass full_hw into compute_losses)
            # norm_mode=str(lcfg.get("nll_norm_mode", "diag")),
            nll_cap=float(lcfg.get("nll_cap", 20.0)),
        )
        total += w_nll * L_nll
        parts["nll_kp"] = L_nll
    else:
        parts["nll_kp"] = hm_logits.new_zeros(())

    # 5) ADD pose loss (if model provides pred_R/pred_t)
    if (w_add > 0) and ("pred_R" in out) and ("pred_t" in out) and (R_gt is not None) and (t_gt is not None) and (pts3d_model is not None):
        frame_mask = (vis.sum(dim=2) > 0).float()  # (B,T)
        L_add = add_l1_pose_loss(
            R_pred=out["pred_R"], t_pred=out["pred_t"],
            R_gt=R_gt, t_gt=t_gt,
            pts3d_model=pts3d_model,
            frame_mask=frame_mask,
        )
        total += w_add * L_add
        parts["pose_add"] = L_add
    else:
        parts["pose_add"] = hm_logits.new_zeros(())

    
    # 6) KP->patch attention alignment (task-coupled)
    base_w_kp_attn = float(lcfg.get("w_kp_attn", 0.0))
    w_kp_attn = float(get_w_scheduled(epoch, base_w_kp_attn, lcfg.get("w_kp_attn_sched", None)))

    if (w_kp_attn > 0) and ("attn_kp2patch" in out) and (out["attn_kp2patch"] is not None):
        input_size = int(cfg["data"]["input_size"])
        patch_size = int(cfg["model"].get("patch_size", 16))
        heatmap_hw = tuple(cfg["model"]["heatmap_out"])

        L_kp_attn = kp_attn_kl_to_gaussian_patches(
            attn_kp2patch=out["attn_kp2patch"],   # (B,T,K,P)
            kps_gt_normhm=kps_gt,                 # (B,T,K,2) [-1,1] hm-norm
            vis=vis,                              # (B,T,K)
            input_size=input_size,
            heatmap_hw=heatmap_hw,
            patch_size=patch_size,
            sigma_px=float(lcfg.get("kp_attn_sigma_px", 0.75 * patch_size)),
            attn_is_logits=bool(lcfg.get("kp_attn_is_logits", False)),
        )
        total += w_kp_attn * L_kp_attn
        parts["kp_attn"] = L_kp_attn
    else:
        parts["kp_attn"] = hm_logits.new_zeros(())

    # 7) Token loss (FIXED: ALWAYS uses out["rays"], no geometry recompute)
    tok  = out["tokens_geo"]                     # (B,T,P,D)
    rays = out["rays"]                           # (B,T,P,3) fp32
    if w_token > 0:
        input_size = int(cfg["data"]["input_size"])
        patch_size = int(cfg["model"].get("patch_size", 16))
        Hp = input_size // patch_size
        Wp = input_size // patch_size
        P_expected = Hp * Wp
        assert tok.shape[2] == P_expected, f"tokens_geo P={tok.shape[2]} expected {P_expected}"
        assert rays is not None and rays.shape[:3] == tok.shape[:3], \
            f"out['rays'] must be (B,T,P,3), got {None if rays is None else rays.shape}"

        L_tok = ray_token_contrastive_loss(
            patch_tokens=tok,                          # (B,T,P,D)
            rays=rays,                                 # (B,T,P,3) fp32
            ray_bin_deg=float(lcfg.get("ray_bin_deg", 6.0)),
            temp=float(lcfg.get("token_temp", 0.07)),
            t_pair=str(lcfg.get("token_t_pair", "ends")),
        )
        total += w_token * L_tok
        parts["token"] = L_tok
    else:
        parts["token"] = hm_logits.new_zeros(())

    parts["loss"] = total
    return total, parts

# -------------------------
# Resume helper (minimal)
# -------------------------
def load_checkpoint_minimal(path: str, model, optimizer, device: str):
    ckpt = torch.load(path, map_location="cpu")
    target = model.module if isinstance(model, DDP) else model
    target.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        for st in optimizer.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(device)
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_ckpt = float(ckpt.get("best_metric", float("inf")))
    return start_epoch, best_ckpt


# -------------------------
# Train / Val
# -------------------------
def train_one_epoch(
    model, loader, optimizer, scheduler, scaler, device, cfg, epoch,
    writer: Optional[SummaryWriter], lr_mult: float = 1.0,
    K_full_33: Optional[torch.Tensor] = None,
):
    model.train()

    max_norm = float(cfg["train"].get("grad_clip", 0.0))
    use_amp = bool(cfg["train"].get("amp", True))
    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
    autocast_dtype = torch.bfloat16

    running = {
        "total": 0.0, "hm_frame": 0.0, "hm_agg": 0.0, "kp_frame": 0.0, "token": 0.0, "kp_attn": 0.0,
        "nll_kp": 0.0, "pose_add": 0.0,
    }
    n_batches = len(loader)

    assert K_full_33 is not None, "Pass K_full_33=get_K_full_from_loader(train_loader) from run_training()."
    K_full_33 = K_full_33.to(device, non_blocking=True)

    target = model.module if isinstance(model, DDP) else model
    pts3d_model = target.pts_3d.squeeze(0)  # (K,3)

    for it, batch in enumerate(loader):
        (
            frames,                 # (B,T,3,H,W)
            kps_gt_normhm,          # (B,T,K,2) [-1,1] hm-norm
            kps_gt_full_px,         # (B,T,K,2) full px
            gt_q,                   # (B,T,4) xyzw
            gt_r,                   # (B,T,3)
            vis,                    # (B,T,K)
            hm_gt,                  # (B,T,K,Hm,Wm)
            frame_ids,              # unused here
            crop_xywh,              # (B,T,4) normalized wrt FULL
            meta,                   # (B,T,5) [scale,pad_l,pad_t,W_full,H_full]
            seq_name,               # unused here
        ) = batch

        frames = frames.to(device, non_blocking=True)
        kps_gt = kps_gt_normhm.to(device, non_blocking=True)
        kps_gt_full_px = kps_gt_full_px.to(device, non_blocking=True)
        vis = vis.to(device, non_blocking=True)
        hm_gt = hm_gt.to(device, non_blocking=True)
        crop_xywh = crop_xywh.to(device, non_blocking=True)
        gt_q = gt_q.to(device, non_blocking=True)
        gt_r = gt_r.to(device, non_blocking=True)

        B, T = frames.shape[0], frames.shape[1]

        # full size from meta
        W_full = int(float(meta[0, 0, 3].item()))
        H_full = int(float(meta[0, 0, 4].item()))
        full_hw = (H_full, W_full)

        cam_K = K_full_33.unsqueeze(0).expand(B, -1, -1)  # (B,3,3)

        R_gt = quat_to_rotmat(gt_q)  # (B,T,3,3)
        t_gt = gt_r

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=dev_type, dtype=autocast_dtype, enabled=use_amp):
            out = model(
                frames,
                cam_K=cam_K,
                full_hw=full_hw,
                crop_xywh=crop_xywh,
            )
        assert out["heatmaps"].dim() == 5
        assert out["mu_norm"].shape[-1] == 2
        assert out["tokens_geo"].dim() == 4

        assert out["attn_kp2patch"] is None or out["attn_kp2patch"].dim() == 4, \
        f"attn_kp2patch must be (B,T,K,P), got {None if out['attn_kp2patch'] is None else out['attn_kp2patch'].shape}"

        #debugging for coordinate system 
        # if epoch == 1 and it == 0:
        #     mu = out["mu_px_full"]          # (B,T,K,2)
        #     gt = kps_gt_full_px             # (B,T,K,2)
        #     sig = out["sigma"]              # (B,T,K,1/2)

        #     print("mu range:", mu.min().item(), mu.max().item())
        #     print("gt range:", gt.min().item(), gt.max().item())
        #     print("abs err mean:", (mu-gt).abs().mean().item(), "px")

        #     H_full, W_full = full_hw
        #     s = (H_full**2 + W_full**2) ** 0.5
        #     print("abs err mean (px):", (mu-gt).abs().mean().item())
        #     print("abs err mean (norm):", ((mu-gt)/s).abs().mean().item())



        with torch.autocast(device_type=dev_type, enabled=False):
            total_loss, parts = compute_losses(
                out, cfg,
                kps_gt=kps_gt,
                hm_gt=hm_gt,
                vis=vis,
                epoch=epoch,
                cam_K=cam_K,
                crop_xywh=crop_xywh,
                full_hw=full_hw,
                kps_gt_full_px=kps_gt_full_px,
                R_gt=R_gt,
                t_gt=t_gt,
                pts3d_model=pts3d_model,
            )

        if is_main_process() and epoch in [6, 12, 20] and it == 0:
            mu = out["mu_px_full"]
            gt = kps_gt_full_px
            sig = out["sigma"]
            vis_f = vis.float()

            err = (mu - gt).norm(dim=-1)  # (B,T,K)
            err_mean = (err * vis_f).sum() / vis_f.sum().clamp_min(1.0)

            if sig.shape[-1] == 1:
                sig2 = sig.expand(-1, -1, -1, 2)
            else:
                sig2 = sig
            sig_mean = (sig2.mean(dim=-1) * vis_f).sum() / vis_f.sum().clamp_min(1.0)

            print(f"[dbg] err_mean_px={float(err_mean):.3f}  sigma_mean_px={float(sig_mean):.3f}  sigma_min={float(sig2.min()):.3f}")
        # if epoch == 6 and it == 0:
        #     # ---- debug: consistent err + effective sigma ----
        #     mu = out["mu_px_full"]          # (B,T,K,2)
        #     gt = kps_gt_full_px             # (B,T,K,2)
        #     vis_f = vis.float()

        #     # L2 error per kp (px)
        #     err_k = (mu - gt).norm(dim=-1)  # (B,T,K)
        #     err_mean = (err_k * vis_f).sum() / vis_f.sum().clamp_min(1.0)

        #     # raw sigma from model
        #     sig = out["sigma"]              # (B,T,K,1/2)

        #     # effective sigma in px (apply sigma_floor_px exactly like loss does)
        #     sigma_floor = float(cfg["loss"].get("sigma_floor_px", 1.0))
        #     if sig.shape[-1] == 1:
        #         sig2 = sig.expand(-1, -1, -1, 2)
        #     else:
        #         sig2 = sig
        #     sig2_eff = sig2.clamp_min(sigma_floor)  # (B,T,K,2)

        #     sig_eff_mean = (sig2_eff.mean(dim=-1) * vis_f).sum() / vis_f.sum().clamp_min(1.0)
        #     sig_raw_mean = (sig2.mean(dim=-1) * vis_f).sum() / vis_f.sum().clamp_min(1.0)

        #     print("mean_err_px(L2):", float(err_mean))
        #     print("sigma_raw_mean_px:", float(sig_raw_mean), "sigma_raw_min_px:", float(sig.min()))
        #     print("sigma_eff_mean_px:", float(sig_eff_mean), "sigma_eff_min_px:", float(sig2_eff.min()))
                    
        scaler.scale(total_loss).backward()

        if max_norm and max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()
            _apply_lr_mult_from_scheduler(optimizer, scheduler, lr_mult)

        running["total"] += float(parts["loss"].item())
        for k in ["hm_frame","hm_agg","kp_frame","token","nll_kp", "kp_attn" , "pose_add"]:
            running[k] += float(parts[k].item())

        # optional viz (main only)
        if is_main_process() and (epoch % 5 == 0) and (it == 0):
            viz_root = os.path.join(cfg["train"]["out_dir"], "viz_tokens")
            ep_dir = os.path.join(viz_root, f"epoch_{epoch:04d}")
            os.makedirs(ep_dir, exist_ok=True)

            # Use consistent time selection for overlays; token diagnostics uses t0/t1 internally.
            t_sel = T // 2
            n_show = min(4, B)

            for b_idx in range(n_show):
                kp_path = os.path.join(ep_dir, f"b{b_idx}_t{t_sel}_kp.png")
                tok_path = os.path.join(ep_dir, f"b{b_idx}_tok.png")

                # Keep your KP overlay (supervised) as-is
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

                # Self-supervised ray-consistent token diagnostics (NO GT used inside)
                tok_metrics = save_token_similarity_diagnostics_bt(
                    out=out,
                    frames=frames,
                    cfg=cfg,
                    save_path=tok_path,
                    crop_xywh=crop_xywh,
                    cam_K=cam_K,
                    full_hw=full_hw,
                    b=b_idx,
                    t=t_sel,  # kept for compatibility; function decides t0/t1 using t_pair
                    t_pair=str(cfg["loss"].get("token_t_pair", "ends")),
                    ray_bin_deg=float(cfg["loss"].get("ray_bin_deg", 6.0)),
                    temp=float(cfg["loss"].get("token_temp", 0.07)),
                    n_anchors_show=int(cfg["loss"].get("tok_viz_n_anchors", 6)),
                )

                # Log per-sample metrics (namespaced by sample index)
                _tb_add_dict(writer, prefix=f"TokDiag/b{b_idx}", d=tok_metrics, step=epoch)

            log(f"[viz] saved -> {ep_dir}")


    avg = {k: v / max(1, n_batches) for k, v in running.items()}

    if is_main_process():
        w_tok_now = get_w_token(epoch, cfg, total_epochs=int(cfg["train"]["epochs"]))
        log(
            f"Train Epoch {epoch}: "
            f"loss={avg['total']:.4f} | hmF={avg['hm_frame']:.4f} | hmA={avg['hm_agg']:.4f} "
            f"| kp={avg['kp_frame']:.4f} | nll={avg['nll_kp']:.4f} | add={avg['pose_add']:.4f} "
            f"| token={avg['token']:.4f} | w_token={w_tok_now:.4f}"
            f"| kp_attn={avg['kp_attn']:.4f}"
        )
        if writer is not None:
            if scheduler is not None:
                writer.add_scalar("LR/lr0", float(optimizer.param_groups[0]["lr"]), epoch)
            writer.add_scalar("Train/total", avg["total"], epoch)
            writer.add_scalar("Train/hm_frame", avg["hm_frame"], epoch)
            writer.add_scalar("Train/hm_agg", avg["hm_agg"], epoch)
            writer.add_scalar("Train/kp", avg["kp_frame"], epoch)
            writer.add_scalar("Train/nll_kp", avg["nll_kp"], epoch)
            writer.add_scalar("Train/pose_add", avg["pose_add"], epoch)
            writer.add_scalar("Train/token", avg["token"], epoch)

    return avg


@torch.no_grad()
def validate(
    model, loader, device, cfg, epoch, writer: Optional[SummaryWriter],
    K_full_33: Optional[torch.Tensor] = None,
):
    if is_dist() and not is_main_process():
        return {}

    model.eval()
    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"

    running = {
        "total": 0.0, "hm_frame": 0.0, "hm_agg": 0.0, "kp_frame": 0.0, "token": 0.0, "kp_attn": 0.0,
        "nll_kp": 0.0, "pose_add": 0.0,
    }
    n_batches = len(loader)

    pck5_sum = 0.0
    pck10_sum = 0.0

    assert K_full_33 is not None, "Pass K_full_33=get_K_full_from_loader(val_loader) from run_training()."
    K_full_33 = K_full_33.to(device, non_blocking=True)

    target = model.module if isinstance(model, DDP) else model
    pts3d_model = target.pts_3d.squeeze(0)

    heatmap_hw = tuple(cfg["model"]["heatmap_out"])
    input_size = int(cfg["data"]["input_size"])

    for batch in loader:
        (
            frames,
            kps_gt_normhm,
            kps_gt_full_px,
            gt_q,
            gt_r,
            vis,
            hm_gt,
            frame_ids,
            crop_xywh,
            meta,
            seq_name,
        ) = batch

        frames = frames.to(device, non_blocking=True)
        kps_gt = kps_gt_normhm.to(device, non_blocking=True)
        kps_gt_full_px = kps_gt_full_px.to(device, non_blocking=True)
        vis = vis.to(device, non_blocking=True)
        hm_gt = hm_gt.to(device, non_blocking=True)
        crop_xywh = crop_xywh.to(device, non_blocking=True)
        gt_q = gt_q.to(device, non_blocking=True)
        gt_r = gt_r.to(device, non_blocking=True)

        B, T = frames.shape[0], frames.shape[1]

        W_full = int(float(meta[0, 0, 3].item()))
        H_full = int(float(meta[0, 0, 4].item()))
        full_hw = (H_full, W_full)

        cam_K = K_full_33.unsqueeze(0).expand(B, -1, -1)

        R_gt = quat_to_rotmat(gt_q)
        t_gt = gt_r

        out = model(
            frames,
            cam_K=cam_K,
            full_hw=full_hw,
            crop_xywh=crop_xywh,
        )

        assert out["heatmaps"].dim() == 5
        assert out["mu_norm"].shape[-1] == 2
        assert out["tokens_geo"].dim() == 4

        assert out["attn_kp2patch"] is None or out["attn_kp2patch"].dim() == 4, \
        f"attn_kp2patch must be (B,T,K,P), got {None if out['attn_kp2patch'] is None else out['attn_kp2patch'].shape}"

        with torch.autocast(device_type=dev_type, enabled=False):
            val_loss, parts = compute_losses(
                out, cfg,
                kps_gt=kps_gt,
                hm_gt=hm_gt,
                vis=vis,
                epoch=epoch,
                cam_K=cam_K,
                crop_xywh=crop_xywh,
                full_hw=full_hw,
                kps_gt_full_px=kps_gt_full_px,
                R_gt=R_gt,
                t_gt=t_gt,
                pts3d_model=pts3d_model,
            )

        running["total"] += float(parts["loss"].item())
        for k in ["hm_frame","hm_agg","kp_frame","token","nll_kp", "kp_attn", "pose_add"]:
            running[k] += float(parts[k].item())

        # kps_pred_normhm = out.get("mu_norm", None) or out.get("keypoints", None)
        kps_pred_normhm = out["mu_norm"]
        if kps_pred_normhm is not None:
            pck5  = pck_crop_from_normhm(kps_pred_normhm, kps_gt, vis, input_size, heatmap_hw, thr_px=5.0)
            pck10 = pck_crop_from_normhm(kps_pred_normhm, kps_gt, vis, input_size, heatmap_hw, thr_px=10.0)
            pck5_sum  += float(pck5.item())
            pck10_sum += float(pck10.item())

    avg = {k: v / max(1, n_batches) for k, v in running.items()}
    avg["pck5_crop"] = pck5_sum / max(1, n_batches)
    avg["pck10_crop"] = pck10_sum / max(1, n_batches)

    if is_main_process():
        log(
            f"Val   Epoch {epoch}: "
            f"loss={avg['total']:.4f} | hmF={avg['hm_frame']:.4f} | hmA={avg['hm_agg']:.4f} "
            f"| kp={avg['kp_frame']:.4f} | nll={avg['nll_kp']:.4f} | add={avg['pose_add']:.4f} "
            f"| token={avg['token']:.4f} "
            f"| PCK@10(crop)={avg['pck10_crop']:.4f} PCK@5(crop)={avg['pck5_crop']:.4f}"
            f"| kp_attn={avg['kp_attn']:.4f}"
        )
        if writer is not None:
            writer.add_scalar("Val/total", avg["total"], epoch)
            writer.add_scalar("Val/hm_frame", avg["hm_frame"], epoch)
            writer.add_scalar("Val/hm_agg", avg["hm_agg"], epoch)
            writer.add_scalar("Val/kp", avg["kp_frame"], epoch)
            writer.add_scalar("Val/nll_kp", avg["nll_kp"], epoch)
            writer.add_scalar("Val/pose_add", avg["pose_add"], epoch)
            writer.add_scalar("Val/token", avg["token"], epoch)
            writer.add_scalar("Val/PCK10_crop", avg["pck10_crop"], epoch)
            writer.add_scalar("Val/PCK5_crop", avg["pck5_crop"], epoch)

    return avg


# -------------------------
# Run (plateau + best)
# -------------------------
def run_training(cfg, logger: Optional[Callable[[str], None]] = None):
    if logger is None:
        logger = lambda s: print(s, flush=True) if is_main_process() else None

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

    train_loader, val_loader, train_sampler, _, _ = make_dataloaders(cfg)

    K_full_train = get_K_full_from_loader(train_loader)
    K_full_val   = get_K_full_from_loader(val_loader)

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
            find_unused_parameters=True,
        )

    lr = float(cfg["train"]["lr"])
    wd = float(cfg["train"].get("weight_decay", 0.05))
    params = [p for p in (model.module.parameters() if isinstance(model, DDP) else model.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.999))

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

    resume_path = cfg["train"].get("resume", None)
    start_epoch = 1

    metric_name = str(cfg["train"].get("save_metric", "total"))
    metric_mode = str(cfg["train"].get("save_metric_mode", "min")).lower()
    assert metric_mode in ("min", "max")

    best_ckpt = -float("inf") if metric_mode == "max" else float("inf")

    if resume_path:
        assert os.path.isfile(resume_path), f"Resume path not found: {resume_path}"
        _log(f"[resume] loading: {resume_path}")
        start_epoch, best_ckpt = load_checkpoint_minimal(resume_path, model, optimizer, device)

        # fast-forward scheduler to match completed steps
        steps_per_epoch = len(train_loader)
        completed_epochs = start_epoch - 1
        n_steps = completed_epochs * steps_per_epoch
        for _ in range(n_steps):
            scheduler.step()
        _apply_lr_mult_from_scheduler(optimizer, scheduler, lr_mult=1.0)
        _log(f"[resume] start_epoch={start_epoch} best_ckpt={best_ckpt:.6f} sched_steps={n_steps}")

    if is_main_process():
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(tb_root, exist_ok=True)
        run_name = f"exp_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        tb_dir = os.path.join(tb_root, run_name)
        writer = SummaryWriter(tb_dir)
        _log(f"[tb] logging -> {tb_dir}")

    if is_dist():
        dist.barrier()

    best_plateau = best_ckpt
    lr_mult = 1.0
    bad_epochs = 0
    n_drops = 0

    plateau_patience  = int(cfg["train"].get("plateau_patience", 4))
    plateau_factor    = float(cfg["train"].get("plateau_factor", 0.3))
    plateau_min_delta = float(cfg["train"].get("plateau_min_delta", 1e-4))
    plateau_max_drops = int(cfg["train"].get("plateau_max_drops", 2))

    for epoch in range(start_epoch, int(cfg["train"]["epochs"]) + 1):
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)

        _ = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, cfg, epoch, writer,
            lr_mult=lr_mult, K_full_33=K_full_train
        )
        val_stats = validate(
            model, val_loader, device, cfg, epoch, writer,
            K_full_33=K_full_val
        )

        if is_main_process():
            default_score = -1e9 if metric_mode == "max" else 1e9
            score = float(val_stats.get(metric_name, default_score))

            if metric_mode == "max":
                is_better = score > (best_ckpt + 1e-12)
            else:
                is_better = score < (best_ckpt - 1e-12)

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
                _log(f"✓ Saved best -> {path} (score={best_ckpt:.6f})")

            if epoch > 5:
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

        lr_mult = _sync_float(lr_mult)
        if is_dist():
            dist.barrier()

    if is_main_process():
        _log(f"Training done. Best={best_ckpt:.6f}")
        if writer is not None:
            writer.flush()
            writer.close()

    cleanup_distributed()
