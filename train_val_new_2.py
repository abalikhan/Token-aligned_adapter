# train_val.py (FINAL — CLEAN, DDP-safe, main-rank logging/checkpointing only)
import os
import math
import random
import datetime
import warnings
from typing import Dict, Any, Tuple, Optional, Callable, List

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
from transforms import build_transforms
from model_hm_dsnt_2 import SpacecraftHMDSNT

from losses import (
    heatmap_kl_div_from_logits_masked,
    smooth_l1_vis_weighted,
    token_spatial_alignment_loss_loo,
)

from utils_new import (
    build_warmup_cosine_scheduler,
    freeze_all_but_adapters_and_heads,
    collate_temporal_chunks,
    save_kp_overlay_bt,
    get_w_token,
    save_token_similarity_diagnostics_bt,
    pck_crop_from_normhm,
    save_dino_style_kp_attention_overlays_bt,
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

def _default_logger(s: str):
    log(s)

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
    obj_list = [idxs if is_main_process() else None]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]

def _finite(x): 
                return torch.isfinite(x).all().item()

def _sync_and_trim_dataset(ds, name: str):
    """
    Ensures every rank has the same dataset length.
    If mismatch happens, trims to min length deterministically.
    Works for Dataset or Subset.
    """
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
    else:
        return Subset(ds, list(range(min_len)))

def make_trainable_params_contiguous(m: nn.Module):
    for p in m.parameters():
        if p.requires_grad and not p.is_contiguous():
            p.data = p.data.contiguous()


# -------------------------
# LR helpers
# -------------------------
def _apply_lr_mult_from_scheduler(optimizer, scheduler, lr_mult: float):
    """
    scheduler sets base LR each step; we overwrite optimizer LR = base_lr * lr_mult
    """
    if scheduler is None:
        raise RuntimeError("_apply_lr_mult_from_scheduler called with scheduler=None")
    base_lrs = scheduler.get_last_lr()
    for g, base in zip(optimizer.param_groups, base_lrs):
        g["lr"] = float(base) * float(lr_mult)

def _set_lr_from_base(optimizer, base_lrs: List[float], lr_mult: float):
    """When scheduler is disabled, enforce lr = base_lr * lr_mult (no accumulation)."""
    assert len(base_lrs) == len(optimizer.param_groups)
    for g, b in zip(optimizer.param_groups, base_lrs):
        g["lr"] = float(b) * float(lr_mult)


# -------------------------
# Model
# -------------------------
def build_model(cfg) -> nn.Module:
    dcfg = cfg["data"]
    mcfg = cfg.get("model", {})

    num_kp = int(dcfg.get("num_kp", 8))
    short_mem = int(dcfg["short_mem"])
    sampling_rate = int(dcfg["sampling_rate"])
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

    patch = int(mcfg.get("patch_size", 16))
    grid_hw = tuple(mcfg.get("grid_hw", [img_hw // patch, img_hw // patch]))

    ks = int(mcfg.get("ks", 3))
    kt = int(mcfg.get("kt", 3))
    fuse = mcfg.get("fuse", "sum")

    return SpacecraftHMDSNT(
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
    )


# -------------------------
# Loaders
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
    )

    if ds_name == "spades":
        train_set = SPADESDataset(mode="train", spatial_transforms=tfm_crop, **base_kwargs)
        val_set   = SPADESDataset(mode="val",  spatial_transforms=None,   **base_kwargs)
        test_set  = SPADESDataset(mode="test", spatial_transforms=None,   **base_kwargs)
    elif ds_name == "spark":
        train_set = SPARKDataset(mode="train", spatial_transforms=tfm_crop, **base_kwargs)
        val_set   = SPARKDataset(mode="val",  spatial_transforms=None,   **base_kwargs)
        test_set  = SPARKDataset(mode="val",  spatial_transforms=None,   **base_kwargs)
    else:
        raise ValueError("data.name must be 'spades' or 'spark'")

    # ---- Optional debugging subsets (SYNCED) ----
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

    # ---- Ensure identical lengths across ranks ----
    train_set = _sync_and_trim_dataset(train_set, "train_set")
    val_set   = _sync_and_trim_dataset(val_set,   "val_set")

    world = get_world_size()
    rank  = get_rank()

    train_sampler = DistributedSampler(train_set, num_replicas=world, rank=rank, shuffle=True, drop_last=True) if world > 1 else None
    val_sampler   = None

    # Only main validates; others get empty val_set
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

    if is_main_process():
        print(f"Train loader: {len(train_loader)} batches, Val loader: {len(val_loader)} batches", flush=True)

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
    w_token    = float(get_w_token(epoch, cfg, total_epochs=int(cfg["train"]["epochs"])))

    sigma_patch = float(lcfg["sigma_patch"])
    temp_pred   = float(lcfg["temp_pred"])
    w_div       = float(lcfg["w_div"])

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

    # 3) Keypoints
    kps_pred_norm = out["keypoints"]
    assert kps_pred_norm.shape == kps_gt.shape, f"{kps_pred_norm.shape=} vs {kps_gt.shape=}"

    L_kp = smooth_l1_vis_weighted(kps_pred_norm, kps_gt, vis)
    total += w_kp * L_kp
    parts["kp_frame"] = L_kp

    # 4) Token alignment
    tok = out.get("adapter_tokens", None)
    if w_token > 0 and tok is not None:
        input_size = int(cfg["data"]["input_size"])
        patch_size = int(cfg["model"].get("patch_size", 16))
        Hp = input_size // patch_size
        Wp = input_size // patch_size
        P_expected = Hp * Wp

        P_got = tok.shape[2]
        if P_got < P_expected:
            raise RuntimeError(f"adapter_tokens has too few tokens: got P={P_got}, expected >= {P_expected}")
        if P_got > P_expected:
            tok = tok[:, :, -P_expected:, :]

        detach_proto = (epoch <= 20)
        L_tok = token_spatial_alignment_loss_loo(
            patch_tokens=tok,
            kps_gt_norm=kps_gt,
            vis=vis,
            temp_pred=temp_pred,
            sigma_patch=sigma_patch,
            detach_proto=detach_proto,
            w_div=w_div,
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
        print(f"[resume] missing: {missing}", flush=True)
        print(f"[resume] unexpected: {unexpected}", flush=True)

    if optimizer is not None and ckpt.get("optimizer") is not None and len(missing) == 0 and len(unexpected) == 0:
        optimizer.load_state_dict(ckpt["optimizer"])
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
# Train / Val
# -------------------------
def train_one_epoch(
    model, loader, optimizer, scheduler, scaler, device, cfg, epoch,
    writer: Optional[SummaryWriter], lr_mult: float, base_lrs_no_sched: Optional[List[float]]
):
    model.train()

    max_norm = float(cfg["train"].get("grad_clip", 0.0))
    use_amp  = bool(cfg["train"].get("amp", True))
    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
    autocast_dtype = torch.bfloat16  # keep as you had

    #parameteres for token loss viz
    sigma_patch = float(cfg["loss"].get("sigma_patch", 0.35))
    temp_pred   = float(cfg["loss"].get("temp_pred", 0.07))

    running = {"total": 0.0, "hm_frame": 0.0, "hm_agg": 0.0, "kp_frame": 0.0, "token": 0.0}
    n_batches = len(loader)

    for it, batch in enumerate(loader):
        frames, kps_gt, gt_q, gt_r, vis, hm_gt, _, crop_xywh = batch

        frames = frames.to(device, non_blocking=True)
        kps_gt = kps_gt.to(device, non_blocking=True)
        vis    = vis.to(device, non_blocking=True)
        hm_gt  = hm_gt.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=dev_type, dtype=autocast_dtype, enabled=use_amp):
            out = model(frames)

            # debugging for non-fininte at the overflow iteration 
            if epoch == 1 and it in [0, 1, 2, 9, 112, 133]:
                hm = out["heatmaps"]
                kp = out["keypoints"]
                print("finite heatmaps:", _finite(hm), "min/max:", float(hm.min()), float(hm.max()))
                print("finite keypoints:", _finite(kp), "min/max:", float(kp.min()), float(kp.max()))


        with torch.autocast(device_type=dev_type, enabled=False):
            total_loss, parts = compute_losses(out, cfg, kps_gt=kps_gt, hm_gt=hm_gt, vis=vis, epoch=epoch)
            if not torch.isfinite(total_loss):
                print("NON-FINITE total_loss", total_loss.item(), {k: float(v.item()) for k,v in parts.items()})
                raise RuntimeError("non-finite loss")

        # with torch.autocast(device_type=dev_type, enabled=False):
        #     total_loss, parts = compute_losses(out, cfg, kps_gt=kps_gt, hm_gt=hm_gt, vis=vis, epoch=epoch)

        # Debug shapes once
        if is_main_process() and epoch == 1 and it == 0:
            if "adapter_tokens" in out:
                print("adapter_tokens shape:", tuple(out["adapter_tokens"].shape), flush=True)
            input_size = int(cfg["data"]["input_size"])
            patch_size = int(cfg["model"].get("patch_size", 16))
            print("P_expected:", (input_size // patch_size) * (input_size // patch_size), flush=True)

        scale_before = float(scaler.get_scale()) if scaler is not None else 1.0

        scaler.scale(total_loss).backward()

        if max_norm and max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # Optimizer step (maybe skipped on overflow)
        scaler.step(optimizer)
        scaler.update()

        scale_after = float(scaler.get_scale()) if scaler is not None else 1.0
        overflow = (scale_after < scale_before)

        if is_main_process() and overflow:
            print(
                f"[AMP] overflow detected, optimizer step skipped at epoch={epoch} it={it} "
                f"(scale {scale_before}->{scale_after})",
                flush=True
            )

        # Scheduler step MUST be skipped if overflow (no optimizer step happened)
        if (scheduler is not None) and (not overflow):
            scheduler.step()
            _apply_lr_mult_from_scheduler(optimizer, scheduler, lr_mult)

        # If scheduler disabled, enforce lr from base each iter (so plateau lr_mult is effective)
        if scheduler is None:
            assert base_lrs_no_sched is not None
            _set_lr_from_base(optimizer, base_lrs_no_sched, lr_mult)

        running["total"]    += float(total_loss.item())
        running["hm_frame"] += float(parts["hm_frame"].item())
        running["hm_agg"]   += float(parts["hm_agg"].item())
        running["kp_frame"] += float(parts["kp_frame"].item())
        running["token"]    += float(parts["token"].item())

        if is_main_process() and epoch == 1 and it in [0, 1, 2, 9, 112, 133]:
            m = model.module if isinstance(model, DDP) else model
            cnt = 0
            for n, p in m.named_parameters():
                if p.requires_grad:
                    print(n, float(p.detach().abs().mean()), float(p.detach().abs().max()))
                    cnt += 1
                    if cnt == 5: break
            # print(f"[CHK] epoch={epoch} train first-param absmean={s:.8f}", flush=True)

        # Viz: only main
        if is_main_process() and (epoch % 10 == 0) and (it == 0):
            viz_root = os.path.join(cfg["train"]["out_dir"], "viz_tokens")
            ep_dir = os.path.join(viz_root, f"epoch_{epoch:04d}")
            os.makedirs(ep_dir, exist_ok=True)

            B, T = frames.shape[0], frames.shape[1]
            t_sel = T // 2
            n_show = min(4, B)

            for b_idx in range(n_show):
                kp_path  = os.path.join(ep_dir, f"b{b_idx}_t{t_sel}_kp.png")
                tok_path = os.path.join(ep_dir, f"b{b_idx}_t{t_sel}_tok.png")

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
                save_token_similarity_diagnostics_bt(
                    out=out,
                    frames=frames,
                    kps_gt=kps_gt,
                    vis=vis,
                    cfg=cfg,
                    save_path=tok_path,
                    b=b_idx,
                    t=t_sel,
                    sigma_patch=sigma_patch,
                    temp_pred=temp_pred,
                    max_k_show=8,
                )
                save_dino_style_kp_attention_overlays_bt(
                    out=out,
                    frames=frames,
                    kps_gt=kps_gt,
                    vis=vis,
                    cfg=cfg,
                    save_dir=tok_path.replace(".png", "_"),
                    b=b_idx,
                    t=t_sel,
                    sigma_patch=sigma_patch,
                )
            log(f"[viz] saved -> {ep_dir}")

    avg = {k: v / max(1, n_batches) for k, v in running.items()}

    if is_main_process():
        w_tok_now = float(get_w_token(epoch, cfg, total_epochs=int(cfg["train"]["epochs"])))
        log(
            f"Train Epoch {epoch}: "
            f"loss={avg['total']:.4f} | hmF={avg['hm_frame']:.4f} | hmA={avg['hm_agg']:.4f} "
            f"| kp={avg['kp_frame']:.4f} | token={avg['token']:.4f} | w_token={w_tok_now:.4f}"
        )

        if writer is not None:
            writer.add_scalar("LR/lr0", float(optimizer.param_groups[0]["lr"]), epoch)
            writer.add_scalar("Train/total", avg["total"], epoch)
            writer.add_scalar("Train/hm_frame", avg["hm_frame"], epoch)
            writer.add_scalar("Train/hm_agg", avg["hm_agg"], epoch)
            writer.add_scalar("Train/kp", avg["kp_frame"], epoch)
            writer.add_scalar("Train/token", avg["token"], epoch)

    return avg


@torch.no_grad()
def validate(model, loader, device, cfg, epoch, writer: Optional[SummaryWriter]):
    if is_dist() and not is_main_process():
        return {}

    model.eval()
    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"

    running = {"total": 0.0, "hm_frame": 0.0, "hm_agg": 0.0, "kp_frame": 0.0, "token": 0.0}
    n_batches = len(loader)

    pck5_sum = 0.0
    pck10_sum = 0.0

    for batch in loader:
        frames, kps_gt, gt_q, gt_r, vis, hm_gt, _, crop_xywh = batch
        frames = frames.to(device, non_blocking=True)
        kps_gt = kps_gt.to(device, non_blocking=True)
        vis    = vis.to(device, non_blocking=True)
        hm_gt  = hm_gt.to(device, non_blocking=True)

        out = model(frames)

        assert tuple(cfg["model"]["heatmap_out"]) == (out["heatmaps"].shape[-2], out["heatmaps"].shape[-1])

        with torch.autocast(device_type=dev_type, enabled=False):
            val_loss, parts = compute_losses(out, cfg, kps_gt=kps_gt, hm_gt=hm_gt, vis=vis, epoch=epoch)

        running["total"]    += float(val_loss.item())
        running["hm_frame"] += float(parts["hm_frame"].item())
        running["hm_agg"]   += float(parts["hm_agg"].item())
        running["kp_frame"] += float(parts["kp_frame"].item())
        running["token"]    += float(parts["token"].item())

        kps_pred_normhm = out["keypoints"]
        heatmap_hw = tuple(cfg["model"]["heatmap_out"])
        input_size = int(cfg["data"]["input_size"])

        pck5  = pck_crop_from_normhm(kps_pred_normhm, kps_gt, vis, input_size, heatmap_hw, thr_px=5.0)
        pck10 = pck_crop_from_normhm(kps_pred_normhm, kps_gt, vis, input_size, heatmap_hw, thr_px=10.0)

        pck5_sum  += float(pck5.item())
        pck10_sum += float(pck10.item())

    avg = {k: v / max(1, n_batches) for k, v in running.items()}
    avg["pck5_crop"]  = pck5_sum / max(1, n_batches)
    avg["pck10_crop"] = pck10_sum / max(1, n_batches)

    if is_main_process() and n_batches > 0:
        m = model.module if isinstance(model, DDP) else model
        s = 0.0
        for _, p in m.named_parameters():
            if p.requires_grad:
                s = p.detach().float().abs().mean().item()
                break
        print(f"[CHK] epoch={epoch} val first-param absmean={s:.8f}", flush=True)

    if is_main_process():
        log(
            f"Val   Epoch {epoch}: "
            f"loss={avg['total']:.4f} | hmF={avg['hm_frame']:.4f} | hmA={avg['hm_agg']:.4f} "
            f"| kp={avg['kp_frame']:.4f} | token={avg['token']:.4f} "
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

    return avg


# -------------------------
# Run (plateau + best)
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

    scheduler = build_warmup_cosine_scheduler(
        optimizer=optimizer,
        epochs=int(cfg["train"]["epochs"]),
        steps_per_epoch=len(train_loader),
        warmup_epochs=int(cfg["train"].get("warmup_epochs", 0)),
        lr_min=float(cfg["train"].get("lr_min", 1e-6)),
    )

    if is_main_process():
        n_train = sum(p.numel() for p in params)
        print(f"[opt] trainable params: {n_train/1e6:.2f}M", flush=True)

    scaler = torch.amp.GradScaler(
        device="cuda" if torch.cuda.is_available() else "cpu",
        enabled=bool(cfg["train"].get("amp", True))
    )

    writer: Optional[SummaryWriter] = None
    out_dir = cfg["train"]["out_dir"]
    tb_root = cfg["train"]["tb_logdir"]
    metric_mode = str(cfg["train"].get("save_metric_mode", "min")).lower()
    metric_name = str(cfg["train"].get("save_metric", "total"))

    resume_path = cfg["train"].get("resume", None)

    # ---- init/resume state (rank0 decides, then broadcast lr_mult) ----
    start_epoch = 1
    if resume_path:
        assert os.path.isfile(resume_path), f"Resume path not found: {resume_path}"
        if is_main_process():
            _log(f"[resume] loading {resume_path}")

        start_epoch, best_ckpt, best_plateau, lr_mult, bad_epochs, n_drops, ckpt_mname, ckpt_mmode = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device
        )

        if (ckpt_mname != metric_name) or (ckpt_mmode != metric_mode):
            best_ckpt = -float("inf") if metric_mode == "max" else float("inf")
            best_plateau = best_ckpt
            bad_epochs = 0
            n_drops = 0
            lr_mult = 1.0
            _log(f"[resume] metric changed ({ckpt_mname}/{ckpt_mmode} -> {metric_name}/{metric_mode}); reset best baselines.")

        if is_main_process():
            _log(
                f"[resume] start_epoch={start_epoch} best_ckpt={best_ckpt:.6f} "
                f"best_plateau={best_plateau:.6f} lr_mult={lr_mult:.6f} "
                f"bad_epochs={bad_epochs} drops={n_drops}"
            )
    else:
        best_ckpt = -float("inf") if metric_mode == "max" else float("inf")
        best_plateau = best_ckpt
        lr_mult = 1.0
        bad_epochs = 0
        n_drops = 0

    # sync lr_mult to all ranks BEFORE training starts
    lr_mult = _sync_float(lr_mult)

    # Apply lr_mult consistently (scheduler mode uses scheduler base; no-scheduler uses captured base_lrs)
    if scheduler is not None:
        _apply_lr_mult_from_scheduler(optimizer, scheduler, lr_mult)
        base_lrs_no_sched = None
    else:
        base_lrs_no_sched = [float(g["lr"]) / float(lr_mult) for g in optimizer.param_groups]
        _set_lr_from_base(optimizer, base_lrs_no_sched, lr_mult)

    # Fine-tune controls
    ft = cfg["train"].get("finetune", {}) or {}
    ft_enabled = bool(ft.get("enabled", False))
    ft_start_epoch = int(ft.get("start_epoch", 10**9))

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

    assert metric_mode in ("min", "max")

    plateau_patience  = int(cfg["train"].get("plateau_patience", 4))
    plateau_factor    = float(cfg["train"].get("plateau_factor", 0.3))
    plateau_min_delta = float(cfg["train"].get("plateau_min_delta", 1e-4))
    plateau_max_drops = int(cfg["train"].get("plateau_max_drops", 2))

    for epoch in range(start_epoch, int(cfg["train"]["epochs"]) + 1):
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)

        # ---- Fine-tune mode switch ----
        if ft_enabled and epoch >= ft_start_epoch:
            if ft_disable_plateau:
                plateau_max_drops = 0

            # Disable scheduler if requested
            if ft_disable_scheduler and (scheduler is not None):
                scheduler = None
                # Capture base_lrs for no-scheduler mode (use current LR / lr_mult)
                base_lrs_no_sched = [float(g["lr"]) / float(lr_mult) for g in optimizer.param_groups]
                if is_main_process():
                    _log(f"[finetune] scheduler disabled at epoch {epoch}")

            # Override lr/wd if provided
            if ft_lr is not None:
                if scheduler is not None:
                    # scheduler will overwrite anyway; keep it consistent via lr_mult
                    for g in optimizer.param_groups:
                        g["lr"] = float(ft_lr) * float(lr_mult)
                else:
                    base_lrs_no_sched = [float(ft_lr)] * len(optimizer.param_groups)
                    _set_lr_from_base(optimizer, base_lrs_no_sched, lr_mult)
            if ft_wd is not None:
                for g in optimizer.param_groups:
                    g["weight_decay"] = float(ft_wd)

        train_stats = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, cfg, epoch, writer,
            lr_mult=lr_mult,
            base_lrs_no_sched=base_lrs_no_sched,
        )
        val_stats = validate(model, val_loader, device, cfg, epoch, writer)

        # ---- Rank0: checkpoint + plateau ----
        if is_main_process():
            default_score = -1e9 if metric_mode == "max" else 1e9
            score = float(val_stats.get(metric_name, default_score))

            is_better = (score > (best_ckpt + 1e-12)) if metric_mode == "max" else (score < (best_ckpt - 1e-12))

            # Save best
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

            # Plateau detection
            if epoch > 5:
                improved = (score > (best_plateau + plateau_min_delta)) if metric_mode == "max" else (score < (best_plateau - plateau_min_delta))
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

        # ---- Sync lr_mult to all ranks ----
        lr_mult = _sync_float(lr_mult)

        # Apply lr_mult immediately after sync (so next epoch starts consistent)
        if scheduler is not None:
            _apply_lr_mult_from_scheduler(optimizer, scheduler, lr_mult)
        else:
            assert base_lrs_no_sched is not None
            _set_lr_from_base(optimizer, base_lrs_no_sched, lr_mult)

        if is_dist():
            dist.barrier()

    if is_main_process():
        _log(f"Training done. Best={best_ckpt:.4f}")
        if writer is not None:
            writer.flush()
            writer.close()

    cleanup_distributed()