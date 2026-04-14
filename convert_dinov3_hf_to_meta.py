# convert_dinov3_hf_to_meta.py
import os
import math
import re
import torch
from safetensors.torch import load_file, save_file
from dinov3.models.vision_transformer import vit_large

# ---- paths ----
ROOT = os.path.abspath(os.path.dirname(__file__))
HF_WEIGHTS = os.path.join(ROOT, "model_weights/dinov3_vit_l_16.pth")
# HF_WEIGHTS = os.path.join(HF_DIR, "model.safetensors")
OUT_WEIGHTS = os.path.join(ROOT, "model_weights/dinov3_vit_l_16_meta_converted.safetensors")

# ---- target model (must match HF config) ----
IMG_SIZE = 224
PATCH = 16
model = vit_large(patch_size=PATCH, img_size=IMG_SIZE)  # ViT-S/16, no register tokens
model.eval()
model_sd = model.state_dict()

print(f"Target model params: {len(model_sd)} tensors")

# ---- load HF weights ----
if not os.path.isfile(HF_WEIGHTS):
    raise FileNotFoundError(f"Could not find HF weights at {HF_WEIGHTS}")
# hf_sd = load_file(HF_WEIGHTS)
hf_sd = torch.load(HF_WEIGHTS, map_location="cpu")
print(f"HF checkpoint tensors: {len(hf_sd)}")

# ---- helpers ----
def strip_prefixes(k: str):
    # Common wrappers
    for p in ("model.", "backbone.", "module.", "encoder."):
        if k.startswith(p): return k[len(p):]
    return k

def rename_basic(k: str):
    # Normalize common names
    k = k.replace("cls_token", "class_token")  # some repos use class_token
    k = k.replace("positional_embedding", "pos_embed")
    k = k.replace("position_embedding", "pos_embed")
    k = k.replace("patch_embeddings", "patch_embed")
    k = k.replace("embed_proj", "proj")
    return k

def maybe_merge_qkv(src_sd, base_name):
    """
    Some checkpoints store separate q, k, v; others store fused qkv.
    Try to build fused tensors if target expects qkv.* and we have q.*,k.*,v.*.
    """
    Wq = src_sd.get(base_name + "q.weight", None)
    Wk = src_sd.get(base_name + "k.weight", None)
    Wv = src_sd.get(base_name + "v.weight", None)
    bq = src_sd.get(base_name + "q.bias", None)
    bk = src_sd.get(base_name + "k.bias", None)
    bv = src_sd.get(base_name + "v.bias", None)
    if Wq is not None and Wk is not None and Wv is not None:
        W = torch.cat([Wq, Wk, Wv], dim=0)
        out = {base_name + "qkv.weight": W}
        if bq is not None and bk is not None and bv is not None:
            b = torch.cat([bq, bk, bv], dim=0)
            out[base_name + "qkv.bias"] = b
        return out
    return {}

def maybe_split_qkv(src_sd, base_name):
    """
    Opposite direction: if src has fused qkv and target expects q,k,v.
    """
    Wqkv = src_sd.get(base_name + "qkv.weight", None)
    bqkv = src_sd.get(base_name + "qkv.bias", None)
    if Wqkv is None:
        return {}
    # infer heads from shapes by 3-way split
    split = Wqkv.shape[0] // 3
    Wq, Wk, Wv = Wqkv[:split], Wqkv[split:2*split], Wqkv[2*split:]
    out = {
        base_name + "q.weight": Wq,
        base_name + "k.weight": Wk,
        base_name + "v.weight": Wv,
    }
    if bqkv is not None:
        bq, bk, bv = bqkv[:split], bqkv[split:2*split], bqkv[2*split:]
        out[base_name + "q.bias"] = bq
        out[base_name + "k.bias"] = bk
        out[base_name + "v.bias"] = bv
    return out

def interpolate_pos_embed(pe_src: torch.Tensor, pe_tgt: torch.Tensor):
    """
    Interpolate 2D spatial part if needed. Supports only no-register ViT-S.
    Shapes: src (1, 1+P, D), tgt (1, 1+P, D) or (1, P+1, D)
    """
    if pe_src.shape == pe_tgt.shape:
        return pe_src
    # split cls and grid
    cls_src, grid_src = pe_src[:, :1, :], pe_src[:, 1:, :]
    cls_tgt, grid_tgt = pe_tgt[:, :1, :], pe_tgt[:, 1:, :]
    D = grid_src.shape[-1]
    gs_src = int(math.sqrt(grid_src.shape[1]))
    gs_tgt = int(math.sqrt(grid_tgt.shape[1]))
    grid_src = grid_src.reshape(1, gs_src, gs_src, D).permute(0,3,1,2)  # (1,D,H,W)
    grid_src = torch.nn.functional.interpolate(grid_src, size=(gs_tgt, gs_tgt), mode="bicubic", align_corners=False)
    grid_src = grid_src.permute(0,2,3,1).reshape(1, gs_tgt*gs_tgt, D)
    pe_out = torch.cat([cls_src, grid_src], dim=1)
    return pe_out

# ---- build a remapped dict ----
remapped = {}
skipped = set()
src_sd = {rename_basic(strip_prefixes(k)): v for k, v in hf_sd.items()}

# 1) Remove classification head if present
for k in list(src_sd.keys()):
    if k.startswith("head.") or k.startswith("fc.") or k.startswith("mlp_head."):
        src_sd.pop(k)

# 2) Handle qkv naming differences layer by layer
# Try to detect if target uses fused qkv:
uses_fused_qkv = any("attn.qkv.weight" in k for k in model_sd.keys())
uses_split_qkv = any("attn.q.weight" in k for k in model_sd.keys())

# prepare a working copy to add synthesized entries
work_sd = dict(src_sd)

num_blocks = len({re.match(r"blocks\.(\d+)\.", k).group(1)
                  for k in model_sd.keys() if k.startswith("blocks.")})

for i in range(num_blocks):
    base_src = f"blocks.{i}.attn."
    if uses_fused_qkv:
        # If source has separate q/k/v, synthesize fused
        synth = maybe_merge_qkv(work_sd, base_src)
        work_sd.update(synth)
    elif uses_split_qkv:
        # If source has fused, synthesize splits
        synth = maybe_split_qkv(work_sd, base_src)
        work_sd.update(synth)

# 3) Direct assign with best-effort key normalization
for k_tgt, tensor_tgt in model_sd.items():
    k_src = k_tgt  # start with same
    k_src = rename_basic(strip_prefixes(k_src))

    # slight differences between repos:
    k_src = k_src.replace("patch_embed.proj.", "patch_embed.proj.")
    k_src = k_src.replace("norm1", "norm1")
    k_src = k_src.replace("norm2", "norm2")

    if k_src in work_sd and work_sd[k_src].shape == tensor_tgt.shape:
        remapped[k_tgt] = work_sd[k_src]
    else:
        # try some special cases
        # qkv
        if "attn.qkv." in k_tgt and uses_fused_qkv and (k_src.replace("attn.qkv.","attn.qkv.") in work_sd):
            v = work_sd[k_src]
            if v.shape == tensor_tgt.shape:
                remapped[k_tgt] = v
                continue
        # split q/k/v
        if any(x in k_tgt for x in ["attn.q.", "attn.k.", "attn.v."]) and uses_split_qkv:
            if k_src in work_sd and work_sd[k_src].shape == tensor_tgt.shape:
                remapped[k_tgt] = work_sd[k_src]
                continue

        # pos_embed interpolation
        if k_tgt == "pos_embed" and k_src in work_sd:
            pe = work_sd[k_src]
            pe_interp = interpolate_pos_embed(pe, tensor_tgt)
            if pe_interp.shape == tensor_tgt.shape:
                remapped[k_tgt] = pe_interp
                continue

        # cls/class token rename
        if k_tgt == "class_token":
            cand = work_sd.get("class_token", None) or work_sd.get("cls_token", None)
            if cand is not None and cand.shape == tensor_tgt.shape:
                remapped[k_tgt] = cand
                continue

        # fallback: mark skipped
        skipped.add((k_tgt, tensor_tgt.shape, work_sd.get(k_src, None).shape if k_src in work_sd else None))

# 4) Report coverage
ok = 0
for k in model_sd.keys():
    if k in remapped:
        ok += 1
print(f"Mapped {ok}/{len(model_sd)} tensors to target.")

if skipped:
    print(f"Unmapped target tensors: {len(skipped)} (showing up to 10)")
    for i,(k, s_t, s_s) in enumerate(list(skipped)[:10]):
        print(f"  - {k}: target {tuple(s_t)}  source {tuple(s_s) if s_s else None}")

# 5) Try strict load with remapped + keep original for leftovers that match
final_sd = {}
final_sd.update(remapped)

missing = [k for k in model_sd.keys() if k not in final_sd]
print(f"Attempt strict load with remapped tensors; missing in remap: {len(missing)}")

# (Optional) If you *really* want zero-missing, you can stop here and inspect the unmapped list
# and add more rules. For now, try strict load and see if zero.
try:
    model.load_state_dict(final_sd, strict=True)
    print("Strict load succeeded with remapped weights only.")
except Exception as e:
    print("Strict load failed with remapped only:", e)
    # fall back to non-strict to see residual stats
    missing_keys, unexpected_keys = model.load_state_dict(final_sd, strict=False)
    print(f"Non-strict load -> missing: {len(missing_keys)}, unexpected: {len(unexpected_keys)}")
    if missing_keys:
        print("  First few missing:", missing_keys[:10])
    if unexpected_keys:
        print("  First few unexpected:", unexpected_keys[:10])

# 6) Save converted checkpoint for future direct loads (safetensors)
save_file({k: v.cpu() for k,v in final_sd.items()}, OUT_WEIGHTS)
print(f"Saved converted weights to: {OUT_WEIGHTS}")