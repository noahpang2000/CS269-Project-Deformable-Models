"""Generate all figures for the FLAME-3 experiment report.

Reads results/*.csv for quantitative plots and re-runs the pipeline on a few
selected frames for qualitative RGB -> prediction overlays. Outputs PNGs to
report/figs/.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flame.data import load_frame
from flame.baselines import color_threshold_mask, DEFAULT_TAU
from flame.kass import KassConfig, run_kass
from flame.gac import GACConfig, run_gac
from flame.metrics import iou

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGS = ROOT / "report" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.bbox": "tight"})

# Display order and labels for the headline comparison.
METHODS = [
    ("gac_oracle", "GAC (oracle)", "#1f77b4"),
    ("kass_oracle", "Kass (oracle)", "#2ca02c"),
    ("unet", "U-Net", "#d62728"),
    ("dals", "DALS", "#9467bd"),
    ("deep_snake", "Deep Snake", "#8c564b"),
    ("color", "Color floor", "#7f7f7f"),
    ("gac_color", "GAC (color init)", "#aec7e8"),
    ("kass_color", "Kass (color init)", "#98df8a"),
]


def load_csv(name: str) -> list[dict]:
    with (RESULTS / f"{name}_per_frame.csv").open() as f:
        return list(csv.DictReader(f))


def col(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows])


# ---------------------------------------------------------------- Fig 1: bars
def fig_bars() -> None:
    means_iou, means_dice, labels, colors = [], [], [], []
    for name, label, c in METHODS:
        rows = load_csv(name)
        means_iou.append(col(rows, "iou").mean())
        means_dice.append(col(rows, "dice").mean())
        labels.append(label)
        colors.append(c)
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, means_iou, w, label="IoU", color=colors, edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, means_dice, w, label="Dice", color=colors, alpha=0.55,
           edgecolor="black", linewidth=0.4, hatch="//")
    for i, (vi, vd) in enumerate(zip(means_iou, means_dice)):
        ax.text(i - w / 2, vi + 0.005, f"{vi:.3f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + w / 2, vd + 0.005, f"{vd:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean score")
    ax.set_title("Mean IoU and Dice by method (solid = IoU, hatched = Dice)")
    ax.set_ylim(0, max(means_dice) * 1.25)
    ax.legend()
    fig.savefig(FIGS / "fig_bars.png")
    plt.close(fig)


# ----------------------------------------------------------- Fig 2: boxplots
def fig_box() -> None:
    box_methods = ["gac_oracle", "kass_oracle", "unet", "dals", "deep_snake", "color"]
    data = [col(load_csv(m), "iou") for m in box_methods]
    labels = [dict(METHODS_LABEL := {n: l for n, l, _ in METHODS})[m] for m in box_methods]
    fig, ax = plt.subplots(figsize=(8, 4))
    bp = ax.boxplot(data, labels=labels, showmeans=True, patch_artist=True)
    for patch, m in zip(bp["boxes"], box_methods):
        patch.set_facecolor(dict((n, c) for n, _, c in METHODS)[m])
        patch.set_alpha(0.6)
    ax.set_ylabel("Per-frame IoU")
    ax.set_title("Per-frame IoU distribution (triangle = mean, line = median)")
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(FIGS / "fig_box.png")
    plt.close(fig)


# ------------------------------------------------- Fig 3: threshold sweep
def fig_threshold() -> None:
    rows = load_csv_named(RESULTS / "threshold_sensitivity.csv")
    methods = ["color", "kass", "gac"]
    thr = sorted({float(r["threshold_c"]) for r in rows})
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in methods:
        ys = [next(float(r["mean_iou"]) for r in rows
                   if r["method"] == m and float(r["threshold_c"]) == t) for t in thr]
        ax.plot(thr, ys, marker="o", label=m)
    ax.set_xlabel("GT thermal threshold (deg C)")
    ax.set_ylabel("Mean IoU (no-oracle init)")
    ax.set_title("GT-threshold sensitivity (no-oracle methods stay near the floor)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(FIGS / "fig_threshold.png")
    plt.close(fig)


def load_csv_named(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


# ------------------------------------------- Fig 4: qualitative overlays
def overlay(ax, rgb, gt, pred, title):
    ax.imshow(rgb)
    # GT outline in cyan, prediction filled translucent magenta.
    ax.contour(gt > 0, levels=[0.5], colors="cyan", linewidths=1.2)
    if pred is not None and (pred > 0).any():
        m = np.zeros((*pred.shape, 4))
        m[pred > 0] = [1, 0, 1, 0.35]
        ax.imshow(m)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def predict_for(method: str, frame):
    """Recompute a method's prediction for one frame (classical only)."""
    if method == "color":
        return color_threshold_mask(frame.rgb, DEFAULT_TAU)
    if method == "gac_oracle":
        return run_gac(frame, GACConfig(), init_mask=None)
    if method == "kass_oracle":
        return run_kass(frame, KassConfig(), init_mask=None)
    if method == "gac_color":
        return run_gac(frame, GACConfig(), init_mask=color_threshold_mask(frame.rgb, DEFAULT_TAU))
    if method == "kass_color":
        return run_kass(frame, KassConfig(), init_mask=color_threshold_mask(frame.rgb, DEFAULT_TAU))
    raise ValueError(method)


def pick_bmw(name: str) -> dict:
    """best / median / worst frame ids by IoU for a method's CSV."""
    rows = load_csv(name)
    rows.sort(key=lambda r: float(r["iou"]))
    n = len(rows)
    return {"worst": rows[0], "median": rows[n // 2], "best": rows[-1]}


def fig_qualitative_classical() -> None:
    """Best/median/worst overlays for the two oracle contour methods + color floor."""
    for method in ["gac_oracle", "kass_oracle", "color"]:
        sel = pick_bmw(method)
        fig, axes = plt.subplots(1, 3, figsize=(11, 4))
        for ax, kind in zip(axes, ["best", "median", "worst"]):
            r = sel[kind]
            frame = load_frame(r["frame"])
            pred = predict_for(method, frame)
            overlay(ax, frame.rgb, frame.gt_mask, pred,
                    f"{kind}: frame {r['frame']}  IoU={float(r['iou']):.3f}")
        fig.suptitle(f"{dict((n, l) for n, l, _ in METHODS)[method]} "
                     f"(cyan = GT outline, magenta = prediction)", fontsize=10)
        fig.savefig(FIGS / f"fig_qual_{method}.png")
        plt.close(fig)


def fig_qualitative_deep() -> None:
    """Real deep predictions from the saved checkpoints (best/median/worst by IoU)."""
    import torch
    from run_deep import build_model, predict_native
    from flame.deep.dataset import NET_SIZE
    device = torch.device("cpu")
    for method in ["unet", "dals", "deep_snake"]:
        ckpt = ROOT / "models" / f"{method}.pt"
        if not ckpt.exists():
            print(f"  skip {method}: no checkpoint"); continue
        model = build_model(method).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()
        sel = pick_bmw(method)
        fig, axes = plt.subplots(1, 3, figsize=(11, 4))
        for ax, kind in zip(axes, ["best", "median", "worst"]):
            r = sel[kind]
            frame = load_frame(r["frame"])
            with torch.no_grad():
                pred = predict_native(method, model, frame, NET_SIZE, device)
            overlay(ax, frame.rgb, frame.gt_mask, pred,
                    f"{kind}: frame {r['frame']}  IoU={float(r['iou']):.3f}")
        fig.suptitle(f"{dict((n, l) for n, l, _ in METHODS)[method]} "
                     f"(cyan = GT outline, magenta = prediction)", fontsize=10)
        fig.savefig(FIGS / f"fig_qual_{method}.png")
        plt.close(fig)


if __name__ == "__main__":
    print("fig 1 bars...");        fig_bars()
    print("fig 2 box...");         fig_box()
    print("fig 3 threshold...");   fig_threshold()
    print("fig 4 classical qual..."); fig_qualitative_classical()
    print("fig 5 deep qual..."); fig_qualitative_deep()
    print("done ->", FIGS)
