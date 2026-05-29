"""Recompute predictions and score boundary-aware metrics for every method
on both FLAME-3 (94 test frames) and FLAME-1 (301 test frames).

Outputs:
  results/{flame3,flame1}_<method>_boundary_per_frame.csv  (per-frame rows)
  results/boundary_summary.csv                              (one row per (dataset,method))
  stdout                                                    (markdown table)

Cost: ~3 min per method per dataset (CPU classical, GPU deep with eval_max_side).
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from flame.baselines import color_threshold_mask
from flame.boundary_metrics import (
    boundary_fscore, boundary_iou, hausdorff_and_avg,
    polygon_chamfer_and_hausdorff, polygons_from_mask,
)
from flame.data import load_frame
from flame.gac import GACConfig, run_gac
from flame.kass import KassConfig, run_kass
from flame.unet_seed import unet_seed_mask
from flame.metrics import dice, iou
from flame.splits import make_splits
from run_deep import build_model, predict_native, _prefix

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

# (dataset, method, family, ckpt_tag, max_side, scope)
# scope: "test" (deep eval split) or "all" (classical full sweep)
CONFIGS = [
    # FLAME-3
    ("flame3", "color",       "classical", "", None,  "all"),
    ("flame3", "kass_oracle", "classical", "", None,  "all"),
    ("flame3", "gac_oracle",  "classical", "", None,  "all"),
    ("flame3", "unet",        "deep",      "", None,  "test"),
    ("flame3", "gac_unet",    "hybrid",    "", None,  "test"),
    ("flame3", "dals",        "deep",      "", None,  "test"),
    ("flame3", "deep_snake",  "deep",      "", None,  "test"),
    # FLAME-1
    ("flame1", "color",       "classical", "", 768,   "all"),
    ("flame1", "kass_oracle", "classical", "", 768,   "all"),
    ("flame1", "gac_oracle",  "classical", "", 768,   "all"),
    ("flame1", "unet",        "deep",      "", 1024,  "test"),
    ("flame1", "gac_unet",    "hybrid",    "", 1024,  "test"),
    ("flame1", "dals",        "deep",      "", 1024,  "test"),
    ("flame1", "deep_snake",  "deep",      "", 1024,  "test"),
]


def predict_classical(name: str, frame) -> np.ndarray:
    if name == "color":
        return color_threshold_mask(frame.rgb)
    if name == "kass_oracle":
        return run_kass(frame, KassConfig(), init_mask=None)
    if name == "gac_oracle":
        return run_gac(frame, GACConfig(), init_mask=None)
    raise ValueError(name)


def load_deep_model(method: str, dataset: str, tag: str = ""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = MODELS / f"{_prefix(dataset)}{method}{tag}.pt"
    model = build_model(method).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    return model, device


def predict_deep(method: str, model, frame, device, size: int = 512) -> np.ndarray:
    with torch.no_grad():
        return predict_native(method, model, frame, size, device)


def predict_gac_unet(frame, dataset: str, device: str) -> np.ndarray:
    """U-Net-seeded GAC: trained detector predicts a mask, GAC refines from it.

    Same GACConfig() as gac_oracle -- the only change vs gac_oracle is the init
    source (U-Net prediction instead of the GT mask). Tests whether a classical
    level set seeded by a real detector tracks the boundary better than the
    detector's own per-pixel mask.
    """
    seed = unet_seed_mask(frame, device=device, dataset=dataset)
    if seed.max() == 0:
        # No detection -> nothing to seed the level set from; GAC would error on
        # an empty init. Return empty (scored like any other empty prediction).
        return np.zeros_like(frame.gt_mask)
    return run_gac(frame, GACConfig(), init_mask=seed)


def score_frame(pred: np.ndarray, gt: np.ndarray) -> dict:
    bf1 = boundary_fscore(pred, gt, tol_px=1.0)
    bf2 = boundary_fscore(pred, gt, tol_px=2.0)
    bf5 = boundary_fscore(pred, gt, tol_px=5.0)
    hd, avg = hausdorff_and_avg(pred, gt)
    biou2 = boundary_iou(pred, gt, band_px=2)
    # Polygon-aware metrics (set-to-set symmetric Chamfer + Hausdorff). Polygons
    # are extracted from the rasterized mask for consistency across methods --
    # the rasterization error is at most ~1px, which is small vs the boundary
    # noise the metric is meant to expose for poor methods.
    pred_polys = polygons_from_mask(pred)
    gt_polys   = polygons_from_mask(gt)
    poly_chamfer, poly_hausdorff = polygon_chamfer_and_hausdorff(pred_polys, gt_polys)
    return {
        "iou": iou(pred, gt), "dice": dice(pred, gt),
        "bf_1px": bf1, "bf_2px": bf2, "bf_5px": bf5,
        "hausdorff_px": hd, "avg_surf_px": avg,
        "biou_2px": biou2,
        "poly_chamfer_px": poly_chamfer, "poly_hausdorff_px": poly_hausdorff,
    }


def run_one(dataset: str, method: str, family: str, tag: str,
            max_side: int | None, scope: str) -> dict:
    print(f"\n=== {dataset} {method} ({family}) scope={scope} ===")
    if scope == "test":
        ids = make_splits(dataset=dataset)["test"]
    else:
        from flame.data import list_frame_ids
        ids = list_frame_ids(dataset=dataset)
    deep_model, deep_device = (None, None)
    if family == "deep":
        deep_model, deep_device = load_deep_model(method, dataset, tag)
    hybrid_device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    skipped_empty = skipped_err = 0
    t_start = time.perf_counter()
    for i, fid in enumerate(ids, 1):
        try:
            f = load_frame(fid, dataset=dataset, max_side=max_side)
            if f.gt_mask.max() == 0:
                # An empty-GT frame: only the deep/hybrid eval includes these.
                # Classical already skips them. Keep behaviour consistent so the
                # gac_unet row is comparable to the unet row (same test frames).
                if family == "classical":
                    skipped_empty += 1
                    continue
                if family == "hybrid":
                    pred = predict_gac_unet(f, dataset, hybrid_device)
                else:
                    pred = predict_deep(method, deep_model, f, deep_device)
            elif family == "deep":
                pred = predict_deep(method, deep_model, f, deep_device)
            elif family == "hybrid":
                pred = predict_gac_unet(f, dataset, hybrid_device)
            else:
                pred = predict_classical(method, f)
        except Exception as e:
            skipped_err += 1
            print(f"  WARN {fid}: {type(e).__name__}: {e}")
            continue
        s = score_frame(pred, f.gt_mask)
        s["frame"] = fid
        rows.append(s)
        if i % 50 == 0 or i == len(ids):
            elapsed = time.perf_counter() - t_start
            print(f"  {i}/{len(ids)}  elapsed {elapsed:.0f}s")

    out = RESULTS / f"{_prefix(dataset)}{method}_boundary_per_frame.csv"
    fieldnames = ["frame", "iou", "dice", "bf_1px", "bf_2px", "bf_5px",
                  "hausdorff_px", "avg_surf_px", "biou_2px",
                  "poly_chamfer_px", "poly_hausdorff_px"]
    with out.open("w", newline="") as f_out:
        w = csv.DictWriter(f_out, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out}  ({len(rows)} rows, "
          f"skipped {skipped_empty} empty-GT, {skipped_err} errors)")

    # Hausdorff can be inf (empty pred or empty gt); filter for the mean.
    def mean_finite(key):
        v = np.array([r[key] for r in rows], dtype=np.float64)
        v = v[np.isfinite(v)]
        return float(v.mean()) if len(v) else float("nan")

    return {
        "dataset": dataset, "method": method, "n": len(rows),
        "iou":    float(np.mean([r["iou"]    for r in rows])),
        "dice":   float(np.mean([r["dice"]   for r in rows])),
        "bf_1px": float(np.mean([r["bf_1px"] for r in rows])),
        "bf_2px": float(np.mean([r["bf_2px"] for r in rows])),
        "bf_5px": float(np.mean([r["bf_5px"] for r in rows])),
        "hausdorff_px": mean_finite("hausdorff_px"),
        "avg_surf_px":  mean_finite("avg_surf_px"),
        "biou_2px":     float(np.mean([r["biou_2px"] for r in rows])),
        "poly_chamfer_px":   mean_finite("poly_chamfer_px"),
        "poly_hausdorff_px": mean_finite("poly_hausdorff_px"),
    }


def main() -> None:
    summaries = []
    for cfg in CONFIGS:
        try:
            summaries.append(run_one(*cfg))
        except FileNotFoundError as e:
            print(f"\nSKIP {cfg[0]}/{cfg[1]}: {e}")

    out_csv = RESULTS / "boundary_summary.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader(); w.writerows(summaries)
    print(f"\nwrote {out_csv}")

    # Markdown table to stdout, grouped by dataset.
    print("\n## Boundary metrics summary\n")
    print("| Dataset | Method      | n   | IoU   | Dice  | BF@1px | BF@2px | BF@5px | Haus(px) | AvgSurf(px) | BIoU@2px | PolyChf(px) | PolyHaus(px) |")
    print("|---------|-------------|----:|------:|------:|-------:|-------:|-------:|---------:|------------:|---------:|------------:|-------------:|")
    for s in summaries:
        print(f"| {s['dataset']:<7} | {s['method']:<11} | {s['n']:>3} | "
              f"{s['iou']:.3f} | {s['dice']:.3f} | "
              f"{s['bf_1px']:.3f}  | {s['bf_2px']:.3f}  | {s['bf_5px']:.3f}  | "
              f"{s['hausdorff_px']:>8.1f} | {s['avg_surf_px']:>11.2f} | {s['biou_2px']:.3f}    | "
              f"{s['poly_chamfer_px']:>11.2f} | {s['poly_hausdorff_px']:>12.1f} |")


if __name__ == "__main__":
    main()
