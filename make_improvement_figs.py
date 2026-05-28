"""Figures for the improvement-investigation report -> report/figs/imp/."""
from __future__ import annotations
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from flame.data import load_frame
from flame.splits import make_splits
from flame.deep.dataset import NET_SIZE
from run_deep import build_model, predict_native

ROOT = Path(__file__).resolve().parent
FIGS = ROOT / "report" / "figs" / "imp"
FIGS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "savefig.bbox": "tight", "figure.dpi": 150})


def curve(path):
    xs, ys = [], []
    for line in Path(path).read_text().splitlines():
        e, v = line.split(",")
        xs.append(int(e)); ys.append(float(v))
    return xs, ys


# ---- Fig 1: attempt scoreboard ----
def fig_scoreboard():
    data = [
        ("Color floor", 0.004, "#bbbbbb"),
        ("Deep Snake (base / +aug)", 0.032, "#8c564b"),
        ("Thermal regr.", 0.068, "#d62728"),
        ("Strong aug 50ep\n(under-trained)", 0.092, "#ff9896"),
        ("DALS (baseline)", 0.122, "#9467bd"),
        ("U-Net +HSV+RG", 0.135, "#c5b0d5"),
        ("U-Net (baseline)", 0.147, "#1f77b4"),
        ("U-Net +R-G", 0.150, "#aec7e8"),
        ("U-Net focal-Tv", 0.153, "#17becf"),
        ("U-Net + medium aug", 0.179, "#2ca02c"),
        ("DALS + medium aug\n120ep (BEST)", 0.182, "#1a7a1a"),
        ("Oracle-GAC\n(location prior)", 0.365, "#ff7f0e"),
    ]
    labels = [d[0] for d in data]; vals = [d[1] for d in data]; cols = [d[2] for d in data]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.barh(range(len(data)), vals, color=cols, edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(data))); ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0.147, color="#1f77b4", ls="--", lw=1, alpha=0.7)
    for b, v in zip(bars, vals):
        ax.text(v + 0.004, b.get_y() + b.get_height()/2, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Test IoU"); ax.set_xlim(0, 0.40)
    ax.set_title("Every attempt vs. the U-Net baseline (dashed). Medium augmentation lifts\n"
                 "both U-Net and DALS; Deep Snake is detector-limited. Oracle-GAC = prior-free gap.")
    fig.savefig(FIGS / "scoreboard.png"); plt.close(fig)


# ---- Fig 5: augmentation transfer across architectures ----
def fig_transfer():
    arch = ["U-Net", "DALS", "Deep Snake"]
    base = [0.147, 0.122, 0.032]
    aug = [0.179, 0.182, 0.032]
    x = np.arange(len(arch)); w = 0.36
    fig, ax = plt.subplots(figsize=(6, 4))
    b1 = ax.bar(x - w/2, base, w, label="baseline (light aug)", color="#9ecae1", edgecolor="black", lw=0.4)
    b2 = ax.bar(x + w/2, aug, w, label="medium aug, 120ep", color="#2ca02c", edgecolor="black", lw=0.4)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.004,
                    f"{bar.get_height():.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(arch)
    ax.set_ylabel("Test IoU"); ax.set_ylim(0, 0.22)
    ax.set_title("Augmentation transfers to the per-pixel architectures\n"
                 "(U-Net, DALS) but not to detector-limited Deep Snake")
    ax.legend(fontsize=8)
    fig.savefig(FIGS / "transfer.png"); plt.close(fig)


# ---- Fig 2: learning curve ----
def fig_learning():
    rows = list(csv.DictReader(open(ROOT / "results/learning_curve.csv")))
    n = [int(r["n_train"]) for r in rows]; v = [float(r["best_val_iou"]) for r in rows]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(n, v, "o-", color="#2ca02c", lw=2, ms=7)
    for x, y in zip(n, v):
        ax.text(x, y + 0.004, f"{y:.3f}", ha="center", fontsize=8)
    ax.set_xlabel("# training frames"); ax.set_ylabel("Best val IoU")
    ax.set_title("Learning curve: val IoU still rising at 100%\nof data -> data-starved")
    ax.grid(alpha=0.3); ax.set_ylim(0.08, 0.20)
    fig.savefig(FIGS / "learning_curve.png"); plt.close(fig)


# ---- Fig 3: augmentation convergence ----
def fig_convergence():
    fig, ax = plt.subplots(figsize=(7, 4))
    for path, lbl, c in [("/tmp/curve_baseline.csv", "baseline (light aug, 50ep)", "#1f77b4"),
                         ("/tmp/curve_aug.csv", "strong aug, 50ep (under-trained)", "#d62728"),
                         ("/tmp/curve_aug2.csv", "medium aug, 120ep (win)", "#2ca02c")]:
        x, y = curve(path)
        ax.plot(x, y, lw=1.6, color=c, label=lbl, alpha=0.9)
    ax.set_xlabel("epoch"); ax.set_ylabel("val IoU")
    ax.set_title("Why the first aug attempt failed: convergence, not augmentation.\n"
                 "Strong aug @50ep was cut off mid-climb; medium aug @120ep converges higher.")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.savefig(FIGS / "convergence.png"); plt.close(fig)


# ---- Fig 4: qualitative overlays for the winning model (DALS + aug) ----
def fig_qual():
    # Show DALS+aug (the best model). Pick best/median/worst over the frames that
    # actually contain fire (GT>0), so "best" is a real segmentation, not an
    # empty-on-empty IoU=1 freebie.
    rows = [r for r in csv.DictReader(open(ROOT / "results/dals_aug2_per_frame.csv"))
            if int(r["gt_px"]) > 0]
    rows.sort(key=lambda r: float(r["iou"]))
    sel = {"worst": rows[0], "median": rows[len(rows)//2], "best": rows[-1]}
    dev = torch.device("cpu")
    model = build_model("dals").to(dev)
    model.load_state_dict(torch.load(ROOT/"models/dals_aug2.pt", map_location=dev, weights_only=True))
    model.eval()
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, kind in zip(axes, ["best", "median", "worst"]):
        r = sel[kind]; frame = load_frame(r["frame"])
        with torch.no_grad():
            pred = predict_native("dals", model, frame, NET_SIZE, dev)
        ax.imshow(frame.rgb)
        ax.contour(frame.gt_mask > 0, [0.5], colors="cyan", linewidths=1.0)
        if (pred > 0).any():
            m = np.zeros((*pred.shape, 4)); m[pred > 0] = [1, 0, 1, 0.35]; ax.imshow(m)
        ax.set_title(f"{kind}: frame {r['frame']}  IoU={float(r['iou']):.3f}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Best model (DALS + medium aug, 120ep), RGB-only, fire-containing frames. "
                 "cyan = thermal GT, magenta = prediction", fontsize=10)
    fig.savefig(FIGS / "qual_aug2.png"); plt.close(fig)


# ---- Fig 6: occlusion filter effect (fire-only IoU) ----
def fig_filter():
    models = ["U-Net\nbaseline", "DALS\nbaseline", "U-Net\n+aug", "DALS\n+aug"]
    unfilt = [0.130, 0.126, 0.152, 0.155]   # fire-only IoU, unfiltered
    filt = [0.144, 0.132, 0.153, 0.154]     # fire-only IoU, filtered
    x = np.arange(len(models)); w = 0.36
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, unfilt, w, label="unfiltered (94 test)", color="#9ecae1", edgecolor="black", lw=0.4)
    ax.bar(x + w/2, filt, w, label="filtered, smoke<0.95 (89 test)", color="#fd8d3c", edgecolor="black", lw=0.4)
    for i,(u,f) in enumerate(zip(unfilt, filt)):
        ax.text(i-w/2, u+0.003, f"{u:.3f}", ha="center", fontsize=7)
        ax.text(i+w/2, f+0.003, f"{f:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8)
    ax.set_ylabel("Fire-only IoU (GT>0)"); ax.set_ylim(0, 0.19)
    ax.set_title("Occlusion filter: helps the light-aug baselines, neutral for the\n"
                 "augmented models (which already cope with occluded frames)")
    ax.legend(fontsize=8)
    fig.savefig(FIGS / "filter.png"); plt.close(fig)


# ---- Fig 7: examples of removed (degenerate) frames ----
def fig_removed():
    import json
    occ = json.loads((ROOT / "results/occluded_frames.json").read_text())["ids"]
    show = occ[:4]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, fid in zip(axes, show):
        frame = load_frame(fid)
        ax.imshow(frame.rgb)
        ax.contour(frame.gt_mask > 0, [0.5], colors="cyan", linewidths=1.0)
        ax.set_title(f"frame {fid}  GTpx={int((frame.gt_mask>0).sum())}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Removed as degenerate (smoke-fraction >= 0.95): full-frame smoke, "
                 "yet thermal GT (cyan) still present — unrecoverable from RGB", fontsize=10)
    fig.savefig(FIGS / "removed.png"); plt.close(fig)


if __name__ == "__main__":
    fig_scoreboard(); print("scoreboard")
    fig_learning(); print("learning curve")
    fig_convergence(); print("convergence")
    fig_transfer(); print("transfer")
    fig_filter(); print("filter")
    fig_removed(); print("removed examples")
    fig_qual(); print("qualitative")
    print("->", FIGS)
