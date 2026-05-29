"""Figures for the FLAME-1 experiment report -> report/figs/flame1/.

Quantitative plots from results/flame1_*_per_frame.csv; qualitative overlays
re-run the trained checkpoints on selected test frames. Standalone — does not
touch FLAME-3 figures.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from flame.baselines import color_threshold_mask
from flame.data import load_frame
from flame.gac import GACConfig, run_gac
from flame.kass import KassConfig, run_kass
from flame.metrics import iou as iou_fn
from flame.splits import make_splits
from run_deep import build_model, predict_native

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
FIGS = ROOT / "report" / "figs" / "flame1"
FIGS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "savefig.bbox": "tight", "figure.dpi": 150})

# Display order for the headline comparison.
METHODS = [
    ("flame1_kass_oracle", "Kass (oracle)", "#2ca02c"),
    ("flame1_color",       "Color floor",   "#7f7f7f"),
    ("flame1_gac_oracle",  "GAC (oracle)",  "#1f77b4"),
    ("flame1_deep_snake",  "Deep Snake",    "#8c564b"),
    ("flame1_dals",        "DALS",          "#9467bd"),
    ("flame1_unet",        "U-Net",         "#d62728"),
]


def load_csv(name: str) -> list[dict]:
    with (RESULTS / f"{name}_per_frame.csv").open() as f:
        return list(csv.DictReader(f))


def col(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows])


# ---- Fig 1: headline bar chart (means) ----
def fig_bars() -> None:
    # Sort ascending so the bar chart reads bottom-to-top worst-to-best.
    sorted_methods = sorted(METHODS, key=lambda m: col(load_csv(m[0]), "iou").mean())
    labels, means_iou, means_dice, colors = [], [], [], []
    for name, label, c in sorted_methods:
        rows = load_csv(name)
        means_iou.append(col(rows, "iou").mean())
        means_dice.append(col(rows, "dice").mean())
        labels.append(label)
        colors.append(c)
    x = np.arange(len(labels))
    w = 0.4
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.barh(x - w / 2, means_iou, w, color=colors, edgecolor="black", lw=0.4,
                 label="IoU")
    b2 = ax.barh(x + w / 2, means_dice, w, color=colors, alpha=0.5, edgecolor="black",
                 lw=0.4, label="Dice")
    for bars, vals in [(b1, means_iou), (b2, means_dice)]:
        for bar, v in zip(bars, vals):
            ax.text(v + 0.008, bar.get_y() + bar.get_height() / 2, f"{v:.3f}",
                    va="center", fontsize=8)
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Score (mean)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("FLAME-1 headline: deep methods dominate the visible-flame task.\n"
                 "Color floor 0.51, oracle-GAC 0.59, U-Net 0.77.")
    fig.savefig(FIGS / "bars.png")
    plt.close(fig)


# ---- Fig 2: per-frame IoU boxplot, fire-only ----
def fig_box() -> None:
    sorted_methods = sorted(METHODS, key=lambda m: col(load_csv(m[0]), "iou").mean())
    data, labels, colors = [], [], []
    for name, label, c in sorted_methods:
        rows = [r for r in load_csv(name) if int(r["gt_px"]) > 0]
        data.append(col(rows, "iou"))
        labels.append(f"{label}\n(n={len(rows)})")
        colors.append(c)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.65,
                    flierprops=dict(marker=".", markersize=3, alpha=0.4))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
        patch.set_edgecolor("black")
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("IoU (per frame, fire-only frames)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_title("FLAME-1 per-frame IoU distribution\n"
                 "(classical n=2001, deep n=test split)")
    fig.savefig(FIGS / "box.png")
    plt.close(fig)


# ---- Fig 3: cross-dataset comparison (FLAME-3 vs FLAME-1) ----
def fig_cross_dataset() -> None:
    # FLAME-3 numbers from the prior reports (RESULTS.md / improvements.tex).
    f3 = {"Color floor": 0.004, "Kass oracle": 0.198, "GAC oracle": 0.365,
          "U-Net": 0.147, "DALS": 0.122, "Deep Snake": 0.032}
    f1 = {"Color floor": col(load_csv("flame1_color"), "iou").mean(),
          "Kass oracle": col(load_csv("flame1_kass_oracle"), "iou").mean(),
          "GAC oracle": col(load_csv("flame1_gac_oracle"), "iou").mean(),
          "U-Net": col(load_csv("flame1_unet"), "iou").mean(),
          "DALS": col(load_csv("flame1_dals"), "iou").mean(),
          "Deep Snake": col(load_csv("flame1_deep_snake"), "iou").mean()}
    methods = list(f3.keys())
    x = np.arange(len(methods))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, [f3[m] for m in methods], w, color="#1f77b4",
           edgecolor="black", lw=0.4, label="FLAME-3 (thermal GT, smoke-occluded, 305 train)")
    ax.bar(x + w / 2, [f1[m] for m in methods], w, color="#d62728",
           edgecolor="black", lw=0.4, label="FLAME-1 (visible-flame GT, 1400 train)")
    for i, m in enumerate(methods):
        ax.text(i - w / 2, f3[m] + 0.012, f"{f3[m]:.3f}", ha="center", fontsize=7)
        ax.text(i + w / 2, f1[m] + 0.012, f"{f1[m]:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Test IoU")
    ax.set_ylim(0, 0.95)
    ax.set_title("FLAME-3 vs FLAME-1 by method. Easier task + more data flips the ranking:\n"
                 "deep methods, near-zero on FLAME-3 from-scratch, dominate on FLAME-1.")
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(FIGS / "cross_dataset.png")
    plt.close(fig)


# ---- Qualitative overlays ----
def _overlay(ax, rgb, gt_mask, pred_mask, title):
    ax.imshow(rgb)
    ax.contour((gt_mask > 0).astype(np.uint8), [0.5], colors="cyan", linewidths=1.1)
    ax.contour((pred_mask > 0).astype(np.uint8), [0.5], colors="magenta", linewidths=1.0)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _select_three(csv_name: str) -> dict:
    rows = [r for r in load_csv(csv_name) if int(r["gt_px"]) > 0]
    rows.sort(key=lambda r: float(r["iou"]))
    return {"worst": rows[0], "median": rows[len(rows) // 2], "best": rows[-1]}


def _deep_pred_at_native(method: str, ckpt_tag: str, frame, size: int = 512,
                          eval_max_side: int = 1024) -> np.ndarray:
    """Re-load checkpoint and predict on a frame already capped at eval_max_side."""
    device = torch.device("cpu")
    model = build_model(method).to(device)
    ckpt = MODELS / f"flame1_{method}{ckpt_tag}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    return predict_native(method, model, frame, size, device)


def fig_qual_deep(method: str, label: str) -> None:
    sel = _select_three(f"flame1_{method}")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (lbl, row) in zip(axes, sel.items()):
        fid = row["frame"]
        f = load_frame(fid, dataset="flame1", max_side=1024)
        pred = _deep_pred_at_native(method, "", f)
        _overlay(ax, f.rgb, f.gt_mask, pred,
                 f"{lbl}: {fid}  IoU={float(row['iou']):.3f}")
    fig.suptitle(f"FLAME-1 {label}: best / median / worst test frames "
                 "(cyan=GT, magenta=pred)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / f"qual_{method}.png")
    plt.close(fig)


def fig_qual_classical(name: str, label: str, runner) -> None:
    """name = 'color' | 'kass_oracle' | 'gac_oracle'; runner returns pred mask."""
    sel = _select_three(f"flame1_{name}")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (lbl, row) in zip(axes, sel.items()):
        fid = row["frame"]
        f = load_frame(fid, dataset="flame1", max_side=768)
        pred = runner(f)
        _overlay(ax, f.rgb, f.gt_mask, pred,
                 f"{lbl}: {fid}  IoU={float(row['iou']):.3f}")
    fig.suptitle(f"FLAME-1 {label}: best / median / worst frames "
                 "(cyan=GT, magenta=pred)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / f"qual_{name}.png")
    plt.close(fig)


def main() -> None:
    print("Quantitative figures...")
    fig_bars()
    fig_box()
    fig_cross_dataset()
    print(f"  wrote {FIGS}/{{bars,box,cross_dataset}}.png")

    print("Qualitative overlays (classical)...")
    fig_qual_classical("color", "Color floor (R−G)",
                       lambda f: color_threshold_mask(f.rgb))
    fig_qual_classical("kass_oracle", "Kass (oracle init)",
                       lambda f: run_kass(f, KassConfig()))
    fig_qual_classical("gac_oracle", "GAC (oracle init)",
                       lambda f: run_gac(f, GACConfig()))

    print("Qualitative overlays (deep)...")
    fig_qual_deep("unet", "U-Net")
    fig_qual_deep("dals", "DALS")
    fig_qual_deep("deep_snake", "Deep Snake")
    print(f"  wrote 6 qual_*.png files to {FIGS}")


if __name__ == "__main__":
    main()
