"""GT-threshold sensitivity for the no-training methods (color, kass, gac).

Re-derives the thermal GT at several temperatures and reports mean IoU/Dice per
method, so we can see whether the method ranking is robust or an artifact of the
150 C cutoff. Deep methods: the trained model is fixed, so just re-run
`run_deep.py --mode eval --threshold-c {T}` at each T for the same curve.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from code.flame.data import list_frame_ids, load_frame
from code.flame.baselines import DEFAULT_TAU, color_threshold_mask
from code.flame.metrics import dice, iou
from code.flame.kass import KassConfig, run_kass
from code.flame.gac import GACConfig, run_gac

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=["color", "kass", "gac"],
                    choices=["color", "kass", "gac"])
    ap.add_argument("--thresholds", nargs="+", type=float, default=[120.0, 150.0, 200.0])
    ap.add_argument("--init", choices=["oracle", "color"], default="color",
                    help="Contour init for kass/gac (color = no oracle)")
    ap.add_argument("--energy-mode", default="rg", choices=["rg", "thermal"])
    ap.add_argument("--color-tau", type=float, default=DEFAULT_TAU)
    ap.add_argument("--limit", type=int, default=None)
    return ap.parse_args()


def predict(name, frame, color_mask, args, cfg_kass, cfg_gac):
    if name == "color":
        return color_mask
    init_mask = None if args.init == "oracle" else color_mask
    if name == "kass":
        return run_kass(frame, cfg_kass, init_mask=init_mask)
    return run_gac(frame, cfg_gac, init_mask=init_mask)


def main() -> None:
    args = parse_args()
    cfg_kass = KassConfig(energy_mode=args.energy_mode)
    cfg_gac = GACConfig(energy_mode=args.energy_mode)
    frame_ids = list_frame_ids()
    if args.limit:
        frame_ids = frame_ids[: args.limit]

    rows = []
    for thr in args.thresholds:
        acc = {m: {"iou": [], "dice": []} for m in args.methods}
        for fid in frame_ids:
            frame = load_frame(fid, threshold_c=thr)
            if frame.gt_mask.max() == 0:
                continue
            color_mask = color_threshold_mask(frame.rgb, args.color_tau)
            for m in args.methods:
                pred = predict(m, frame, color_mask, args, cfg_kass, cfg_gac)
                acc[m]["iou"].append(iou(pred, frame.gt_mask))
                acc[m]["dice"].append(dice(pred, frame.gt_mask))
        for m in args.methods:
            n = len(acc[m]["iou"])
            rows.append({
                "method": m,
                "threshold_c": thr,
                "mean_iou": float(np.mean(acc[m]["iou"])) if n else float("nan"),
                "mean_dice": float(np.mean(acc[m]["dice"])) if n else float("nan"),
                "n": n,
            })

    print(f"\nGT-threshold sensitivity (init={args.init}, n frames per threshold vary):\n")
    header = "method".ljust(8) + "".join(f"{t:>10.0f}C" for t in args.thresholds)
    print("IoU")
    print(header)
    for m in args.methods:
        line = m.ljust(8)
        for t in args.thresholds:
            v = next(r["mean_iou"] for r in rows if r["method"] == m and r["threshold_c"] == t)
            line += f"{v:>11.4f}"
        print(line)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "threshold_sensitivity.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "threshold_c", "mean_iou", "mean_dice", "n"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
