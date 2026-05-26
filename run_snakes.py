"""Run the color floor and/or classical contours on FLAME-3 and score them.

For each frame the ground truth is the thermal frame thresholded at
>= --threshold-c. Methods:
  color  RGB R-G color threshold (no model, no init) -- the floor.
  kass   Kass Snakes; --init oracle (from GT) or color (from the color floor).
  gac    Geodesic Active Contours; same --init choice.
  all    color + kass + gac.

IoU and Dice are scored against the GT; per-frame scores go to
results/<method>_per_frame.csv.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from flame.data import DEFAULT_THRESHOLD_C, list_frame_ids, load_frame
from flame.baselines import DEFAULT_TAU, color_threshold_mask
from flame.metrics import dice, iou
from flame.kass import KassConfig, run_kass
from flame.gac import GACConfig, run_gac

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["color", "kass", "gac", "all"], default="all")
    ap.add_argument("--init", choices=["oracle", "color"], default="oracle",
                    help="Contour init for kass/gac: oracle (from GT) or color (RGB floor)")
    ap.add_argument("--energy-mode", default="rg", choices=["rg", "thermal"])
    ap.add_argument("--color-tau", type=float, default=DEFAULT_TAU,
                    help="R-G threshold for the color floor / color init (default 0.15)")
    ap.add_argument("--threshold-c", type=float, default=DEFAULT_THRESHOLD_C)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--frames", nargs="+", default=None)
    return ap.parse_args()


def predict(name: str, frame, args, cfg_kass, cfg_gac) -> np.ndarray:
    if name == "color":
        return color_threshold_mask(frame.rgb, args.color_tau)
    init_mask = None if args.init == "oracle" else color_threshold_mask(frame.rgb, args.color_tau)
    if name == "kass":
        return run_kass(frame, cfg_kass, init_mask=init_mask)
    return run_gac(frame, cfg_gac, init_mask=init_mask)


def run_method(name: str, frame_ids: list[str], args: argparse.Namespace) -> None:
    cfg_kass = KassConfig(energy_mode=args.energy_mode)
    cfg_gac = GACConfig(energy_mode=args.energy_mode)
    init_desc = "" if name == "color" else f", init={args.init}"
    print(f"\n=== {name.upper()}  (energy={args.energy_mode}{init_desc}, "
          f"GT threshold>={args.threshold_c:g}C) ===")

    rows: list[dict] = []
    skipped_empty = 0
    for i, fid in enumerate(frame_ids, 1):
        frame = load_frame(fid, threshold_c=args.threshold_c)
        if frame.gt_mask.max() == 0:
            skipped_empty += 1
            continue
        t0 = time.perf_counter()
        pred = predict(name, frame, args, cfg_kass, cfg_gac)
        latency = time.perf_counter() - t0
        rows.append({
            "frame": fid,
            "iou": iou(pred, frame.gt_mask),
            "dice": dice(pred, frame.gt_mask),
            "gt_px": int((frame.gt_mask > 0).sum()),
            "pred_px": int((pred > 0).sum()),
            "latency_s": latency,
        })
        if i % 25 == 0 or i == len(frame_ids):
            print(f"  {i}/{len(frame_ids)} frames processed")

    if not rows:
        print("  No non-empty frames to score.")
        return

    ious = np.array([r["iou"] for r in rows])
    dices = np.array([r["dice"] for r in rows])
    lat = np.array([r["latency_s"] for r in rows])
    print(f"  scored frames : {len(rows)}  (skipped {skipped_empty} empty-GT)")
    print(f"  IoU   mean={ious.mean():.4f}  median={np.median(ious):.4f}  std={ious.std():.4f}")
    print(f"  Dice  mean={dices.mean():.4f}  median={np.median(dices):.4f}  std={dices.std():.4f}")
    print(f"  latency median={np.median(lat):.3f} s/frame")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_csv = RESULTS_DIR / f"{name}_per_frame.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  per-frame scores -> {out_csv}")


def main() -> None:
    args = parse_args()
    frame_ids = args.frames if args.frames else list_frame_ids()
    if args.frames is None and args.limit is not None:
        frame_ids = frame_ids[: args.limit]
    print(f"Frames to process: {len(frame_ids)}")
    methods = ["color", "kass", "gac"] if args.method == "all" else [args.method]
    for name in methods:
        run_method(name, frame_ids, args)


if __name__ == "__main__":
    main()
