#!/usr/bin/env python3
import io, os, sys, math, zipfile, pickle
import numpy as np

try:
    import torch
except Exception:
    torch = None

import matplotlib.pyplot as plt

def as_numpy(x):
    if x is None: return None
    if hasattr(x, "detach"): x = x.detach()
    if hasattr(x, "cpu"): x = x.cpu()
    try:
        return np.asarray(x)
    except Exception:
        return np.array(x)

def stats(name, arr):
    if arr is None:
        print(f"{name}: <missing>")
        return
    a = as_numpy(arr)
    print(f"{name}: shape={a.shape} dtype={a.dtype} "
          f"min={a.min():.6f} max={a.max():.6f} mean={a.mean():.6f} std={a.std():.6f}")

def try_load_flat(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    keys = set(z.files)
    need = {"hm_gt", "hm_pred", "vis", "kps_gt", "kps_pred", "img_hw"}
    if need & keys:
        # looks like a flat npz
        out = {k: (z[k] if k in z.files else None) for k in need}
        if out["img_hw"] is not None:
            out["img_hw"] = tuple(out["img_hw"].tolist())
        return out
    return None

def try_load_nested_with_pickle(npz_path):
    # NPZ is a ZIP; look for something like "debug_hm_dump_first_batch/data.pkl"
    with zipfile.ZipFile(npz_path, "r") as zf:
        pkl_name = None
        for n in zf.namelist():
            if n.endswith(".pkl"):
                pkl_name = n
                break
        if pkl_name is None:
            return None
        raw = zf.read(pkl_name)
    # Attempt torch.load first (best for tensors); fallback to pickle
    payload = None
    if torch is not None:
        try:
            payload = torch.load(io.BytesIO(raw), map_location="cpu")
        except Exception:
            payload = None
    if payload is None:
        # plain pickle fallback
        payload = pickle.loads(raw)
    # Normalize output dictionary
    out = {
        "hm_gt":   payload.get("hm_gt"),
        "hm_pred": payload.get("hm_pred"),
        "vis":     payload.get("vis"),
        "kps_gt":  payload.get("kps_gt"),
        "kps_pred":payload.get("kps_pred"),
        "img_hw":  tuple(payload.get("img_hw")) if payload.get("img_hw") is not None else None,
    }
    return out

def main(npz_path):
    assert os.path.exists(npz_path), f"Not found: {npz_path}"

    bundle = try_load_flat(npz_path)
    if bundle is None:
        bundle = try_load_nested_with_pickle(npz_path)
    if bundle is None:
        raise RuntimeError("Could not decode dump. (Expected flat npz keys or an embedded data.pkl)")

    hm_gt   = bundle.get("hm_gt")
    hm_pred = bundle.get("hm_pred")
    vis     = bundle.get("vis")
    kps_gt  = bundle.get("kps_gt")
    kps_pred= bundle.get("kps_pred")
    img_hw  = bundle.get("img_hw")

    print("=== Sanity numbers (first batch) ===")
    stats("hm_gt", hm_gt)
    stats("hm_pred", hm_pred)
    stats("vis", vis)
    stats("kps_gt", kps_gt)
    stats("kps_pred", kps_pred)
    print("img_hw:", img_hw)

    # Quick checks
    if kps_gt is not None:
        a = as_numpy(kps_gt)
        out_of_range = np.logical_or(a < -1.01, a > 1.01).any()
        print("kps_gt outside [-1,1]? ->", bool(out_of_range))

    if vis is not None:
        print("vis mean:", float(as_numpy(vis).astype(np.float32).mean()))

    # ----- Plots -----
    out_dir = os.path.dirname(os.path.abspath(npz_path))
    paths = []

    # 1) histograms
    if (hm_gt is not None) and (hm_pred is not None):
        g = as_numpy(hm_gt).ravel()
        p = as_numpy(hm_pred).ravel()
        fig = plt.figure(figsize=(8,4))
        ax1 = fig.add_subplot(1,2,1); ax1.hist(g, bins=100); ax1.set_title("hm_gt histogram")
        ax2 = fig.add_subplot(1,2,2); ax2.hist(p, bins=100); ax2.set_title("hm_pred histogram")
        fig.suptitle("Heatmap Value Distributions")
        fig.tight_layout()
        f = os.path.join(out_dir, "hm_histograms.png"); fig.savefig(f, dpi=140); plt.close(fig); paths.append(f)

    # helper: reduce to (K,H,W)
    def first_triplet(hm):
        a = as_numpy(hm)
        while a.ndim > 3:
            a = a[0]
        return a

    # 2) GT vs Pred grids
    if (hm_gt is not None) and (hm_pred is not None):
        g = first_triplet(hm_gt)  # (K,H,W)
        p = first_triplet(hm_pred)
        K = int(min(g.shape[0], p.shape[0], 6))
        fig = plt.figure(figsize=(10, 2*K))
        for i in range(K):
            axg = fig.add_subplot(K,2,2*i+1); axg.imshow(g[i]); axg.set_axis_off(); axg.set_title(f"GT k={i}")
            axp = fig.add_subplot(K,2,2*i+2); axp.imshow(p[i]); axp.set_axis_off(); axp.set_title(f"Pred k={i}")
        fig.suptitle("GT vs Pred Heatmaps (first sample/time)")
        fig.tight_layout()
        f = os.path.join(out_dir, "hm_gt_pred_grid.png"); fig.savefig(f, dpi=140); plt.close(fig); paths.append(f)

    # 3) KP scatter in image pixels
    if (kps_gt is not None) and (kps_pred is not None) and (img_hw is not None):
        H, W = img_hw
        g = as_numpy(kps_gt)[0,0]  # (K,2) in [-1,1]
        p = as_numpy(kps_pred)[0,0] if as_numpy(kps_pred).ndim >= 4 else as_numpy(kps_pred)[0]
        gx = ((g[...,0] + 1.0) * 0.5) * (W - 1.0); gy = ((g[...,1] + 1.0) * 0.5) * (H - 1.0)
        px = ((p[...,0] + 1.0) * 0.5) * (W - 1.0); py = ((p[...,1] + 1.0) * 0.5) * (H - 1.0)

        fig = plt.figure(figsize=(5,5))
        plt.scatter(gx, gy, label="GT", marker="o")
        plt.scatter(px, py, label="Pred", marker="x")
        plt.gca().invert_yaxis(); plt.legend(); plt.title("Keypoints (first B,T) in image pixels")
        f = os.path.join(out_dir, "kp_scatter.png"); fig.savefig(f, dpi=140); plt.close(fig); paths.append(f)

    print("\nSaved:", *paths, sep="\n  ")
    return paths

if __name__ == "__main__":
    npz = sys.argv[1] if len(sys.argv) > 1 else "debug_hm_dump_first_batch.npz"
    main(npz)