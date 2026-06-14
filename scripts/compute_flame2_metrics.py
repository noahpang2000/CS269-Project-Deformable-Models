"""Authoritative metric table for the FLAME-2 and combined (FLAME 1+2+3) reruns.

Computes IoU / Dice / Boundary-F1@2px on the held-out test split for every
headline method we can run in this environment:
  deep:      unet, dals, deep_snake_simple   (checkpoints models/<ds>_<m>.pt)
  classical: color floor, Kass (oracle), GAC (oracle)
Deep Snake (paper) is omitted -- its CenterNet detector needs mmdet, which is not
installed here.

Frames are scored at a capped resolution (--max-side, default 1024) so FLAME-1's
4K frames in the combined set stay tractable; the same cap is applied to every
method for a fair comparison. Writes results/<dataset>_metrics.csv.

Run:  python scripts/compute_flame2_metrics.py --dataset flame2
      python scripts/compute_flame2_metrics.py --dataset combined
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from flame.data import load_frame
from flame.splits import make_splits
from flame.metrics import iou as iou_fn, dice as dice_fn
from flame.baselines import DEFAULT_TAU, color_threshold_mask
from flame.kass import KassConfig, run_kass
from flame.gac import GACConfig, run_gac
import run_deep as R

SIZE, DEVICE = 512, "cuda" if torch.cuda.is_available() else "cpu"
DEEP_METHODS = ["unet", "dals", "deep_snake_simple"]


def boundary_f1(pred, gt, tol=2):
    """F1 of boundary pixels within `tol` px -- identical to compute_paper_metrics."""
    p = (pred > 0).astype(np.uint8); g = (gt > 0).astype(np.uint8)
    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    if p.sum() == 0 or g.sum() == 0:
        return 0.0
    def boundary(m):
        return m - cv2.erode(m, np.ones((3, 3), np.uint8))
    pb, gb = boundary(p), boundary(g)
    k = 2 * tol + 1
    gb_d = cv2.dilate(gb, np.ones((k, k), np.uint8))
    pb_d = cv2.dilate(pb, np.ones((k, k), np.uint8))
    prec = (pb * gb_d).sum() / (pb.sum() + 1e-9)
    rec = (gb * pb_d).sum() / (gb.sum() + 1e-9)
    return 0.0 if prec + rec == 0 else float(2 * prec * rec / (prec + rec))


def score(frames, predict):
    iou_l, dice_l, bf_l = [], [], []
    for fr in frames:
        pred = predict(fr)
        iou_l.append(iou_fn(pred, fr.gt_mask))
        dice_l.append(dice_fn(pred, fr.gt_mask))
        bf_l.append(boundary_f1(pred, fr.gt_mask))
    return float(np.mean(iou_l)), float(np.mean(dice_l)), float(np.mean(bf_l)), len(frames)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["flame2", "combined"], required=True)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--no-contour-classical", action="store_true",
                    help="skip the slow Kass/GAC oracle contours (keeps the color floor)")
    args = ap.parse_args()

    test = make_splits(dataset=args.dataset)["test"]
    # Preload frames once (every method scores the same frames at the same res).
    frames = [load_frame(f, dataset=args.dataset, max_side=args.max_side) for f in test]
    print(f"[{args.dataset}] {len(frames)} test frames @ max_side={args.max_side}")

    rows = []
    # --- deep methods ---
    for m in DEEP_METHODS:
        ckpt = ROOT / "models" / f"{args.dataset}_{m}.pt"
        if not ckpt.exists():
            print(f"  skip {m}: no checkpoint {ckpt.name}"); continue
        model = R.build_model(m).to(DEVICE).eval()
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        res = score(frames, lambda fr, m=m, model=model:
                    R.predict_native(m, model, fr, SIZE, DEVICE, conf_threshold=0.3))
        rows.append((args.dataset, m, *res))
        print(f"  {m:20s} IoU={res[0]:.3f} Dice={res[1]:.3f} BF@2={res[2]:.3f}")

    # --- Deep Snake (paper): CenterNet detector + snake head (FLAME-2 only) ---
    ckpt_txt = ROOT / "results" / f"{args.dataset}_centernet_ckpt.txt"
    snake_ckpt = ROOT / "models" / f"{args.dataset}_deep_snake_paper.pt"
    cfg = ROOT / "flame" / "deep" / f"{args.dataset}_centernet.py" \
        if args.dataset == "flame2" else None
    if snake_ckpt.exists():
        tw = torch.load(snake_ckpt, map_location=DEVICE, weights_only=True)

        # ORACLE (GT per-component boxes -> snake): isolates the snake head,
        # independent of the (failed) detector. Uses the lightweight TrainWrapper.
        wrap = R.build_model("deep_snake_paper").to(DEVICE).eval()
        wrap.load_state_dict(tw)

        def paper_oracle(fr):
            m = R._snake_predict_paper(wrap, fr, SIZE, DEVICE) * 255
            h, w = fr.gt_mask.shape
            return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        res = score(frames, paper_oracle)
        rows.append((args.dataset, "deep_snake_paper_oracle", *res))
        print(f"  {'deep_snake_paper_oracle':24s} IoU={res[0]:.3f} Dice={res[1]:.3f} BF@2={res[2]:.3f}")
        # NOTE: the detector-fed SYSTEM number is computed separately
        # (scripts/paper_system_metric.py) so the flaky mmcv init can't kill this run.

    # --- classical ---
    kass_cfg, gac_cfg = KassConfig(), GACConfig()
    classical = {"color": lambda fr: color_threshold_mask(fr.rgb, DEFAULT_TAU)}
    if not args.no_contour_classical:   # Kass/GAC oracle are slow and dataset-agnostic
        classical["kass_oracle"] = lambda fr: run_kass(fr, kass_cfg, init_mask=None)
        classical["gac_oracle"] = lambda fr: run_gac(fr, gac_cfg, init_mask=None)
    for name, fn in classical.items():
        res = score(frames, fn)
        rows.append((args.dataset, name, *res))
        print(f"  {name:20s} IoU={res[0]:.3f} Dice={res[1]:.3f} BF@2={res[2]:.3f}")

    out = ROOT / "results" / f"{args.dataset}_metrics.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["dataset", "method", "iou", "dice", "bf_2px", "n"])
        for ds, m, i, d, b, n in rows:
            w.writerow([ds, m, f"{i:.4f}", f"{d:.4f}", f"{b:.4f}", n])
    print("wrote", out)


if __name__ == "__main__":
    main()
