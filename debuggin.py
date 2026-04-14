# debuggin.py
import sys
import os
import numpy as np

# default path (you can also pass a path: python debuggin.py /path/to/file.npz)
PATH = sys.argv[1] if len(sys.argv) > 1 else "/home/users/u103361/temporal_correlation/debug/VAL_debug_hm_dump_first_batch.npz"

def softargmax_xy_from_probs(P):  # P: (K,H,W) or (T,K,H,W)
    if P.ndim == 3:
        P = P[None, ...]  # -> (1,K,H,W)
    T, K, H, W = P.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    xx = xx.astype(np.float32); yy = yy.astype(np.float32)
    num_x = (P * xx[None, None]).sum(axis=(-2, -1))
    num_y = (P * yy[None, None]).sum(axis=(-2, -1))
    # denom is 1 if probs are normalized; still safe to compute
    # denom = P.sum(axis=(-2, -1))
    xy = np.stack([num_x, num_y], axis=-1)  # (T,K,2) in heatmap pixels
    return xy

def summarize_npz(p):
    d = np.load(p)
    print("npz keys:", list(d.files))

    # ---- New-format dump (what your training is writing now) ----
    if {"hm_gt", "hm_pred_logits", "hm_pred_probs", "pred_argmax_xy"}.issubset(d.files):
        hm_gt   = d["hm_gt"]              # (T,K,H,W)  (probabilities)
        pred_lp = d["hm_pred_logits"]     # (T,K,H,W)  (logits)
        pred_pp = d["hm_pred_probs"]      # (T,K,H,W)  (probabilities)
        arg_xy  = d["pred_argmax_xy"]     # (T,K,2)    (x,y in heatmap px)

        T, K, H, W = hm_gt.shape
        t0 = 0

        # Soft-argmax GT in heatmap space (center of mass)
        gt_xy = softargmax_xy_from_probs(hm_gt)  # (T,K,2)

        # Errors (heatmap pixels) at t0
        err_px = np.linalg.norm(arg_xy - gt_xy, axis=-1)  # (T,K)
        print(f"[shape] T={T} K={K} HxW={H}x{W}")
        print("pred_probs  t0: min/mean/max/std =", float(pred_pp[t0].min()), float(pred_pp[t0].mean()),
              float(pred_pp[t0].max()), float(pred_pp[t0].std()))
        print("gt_probs    t0: min/mean/max/std =", float(hm_gt[t0].min()), float(hm_gt[t0].mean()),
              float(hm_gt[t0].max()), float(hm_gt[t0].std()))
        print("argmax vs GT (heatmap px)  t0: mean={:.3f}  median={:.3f}".format(err_px[t0].mean(), np.median(err_px[t0])))
        for r in (1, 2, 3, 5):
            print(f"PCK@{r}px (t0): {(err_px[t0] < r).mean():.3f}")
        return True

    # ---- Old-format dump (earlier suggestion) ----
    expected_old = {"hm_gt", "hm_pred", "vis", "kps_gt", "kps_pred_norm", "kps_pred_px"}
    if expected_old.issubset(d.files):
        hm_gt   = d["hm_gt"]          # (K,H,W)
        hm_pred = d["hm_pred"]        # (K,H,W) logits or probs
        vis     = d["vis"]            # (K,)
        kps_gt  = d["kps_gt"]         # (K,2) normalized [-1,1]
        kps_pn  = d["kps_pred_norm"]  # (K,2) normalized [-1,1]
        kps_ppx = d["kps_pred_px"]    # (K,2) in heatmap px
        print("hm_gt   stats:", hm_gt.min(), hm_gt.max(), hm_gt.mean())
        print("hm_pred stats:", hm_pred.min(), hm_pred.max(), hm_pred.mean(), hm_pred.std())
        print("vis mean:", vis.mean())
        print("kps_gt range:", kps_gt.min(), kps_gt.max())
        return True

    # Unknown layout
    print("Unrecognized NPZ layout. Available keys:", list(d.files))
    return False

def summarize_pt(p):
    import torch
    d = torch.load(p, map_location="cpu")
    print("pt keys:", list(d.keys()))
    hm_logits = d["hm_logits"].numpy()     # (K,H,W)
    probs     = d["probs"].numpy()         # (K,H,W)
    argmax_xy = d["argmax_xy"].numpy()     # (K,2)
    soft_xy   = d["softargmax_xy"].numpy() # (K,2)
    gt_xy_hm  = d["gt_xy_heatmap"].numpy() # (K,2)
    print("hm_logits stats:", hm_logits.min(), hm_logits.max(), hm_logits.mean())
    print("probs stats:", probs.min(), probs.max(), probs.mean(), probs.std())
    print("argmax_xy[0:3]:", argmax_xy[:3])
    print("softargmax_xy[0:3]:", soft_xy[:3])
    print("gt_xy_heatmap[0:3]:", gt_xy_hm[:3])
    return True

def main():
    ok = False
    if os.path.isfile(PATH) and PATH.endswith(".npz"):
        try:
            ok = summarize_npz(PATH)
        except Exception as e:
            print("npz read failed:", e)

    if not ok:
        # try a sibling .pt
        pt_path = os.path.splitext(PATH)[0] + ".pt"
        try:
            ok = summarize_pt(pt_path)
        except Exception as e:
            print("pt read failed:", e)

    if not ok:
        print("Could not read either npz or pt dump. Check the file path and that the first-iter dump ran.")

if __name__ == "__main__":
    main()