"""Figures for the boundary-metrics mini-report.

Reads results/boundary_summary.csv and produces 3 figures to report/figs/boundary/.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
FIGS = ROOT / "report" / "figs" / "boundary"
FIGS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "savefig.bbox": "tight", "figure.dpi": 150})

# Stable color per method, shared across plots.
COLORS = {
    "color": "#7f7f7f", "kass_oracle": "#2ca02c", "gac_oracle": "#1f77b4",
    "unet": "#d62728",  "dals": "#9467bd",        "deep_snake": "#8c564b",
}
LABELS = {
    "color": "Color floor", "kass_oracle": "Kass (oracle)",
    "gac_oracle": "GAC (oracle)", "unet": "U-Net",
    "dals": "DALS", "deep_snake": "Deep Snake",
}


def load_summary() -> dict[tuple[str, str], dict]:
    rows = {}
    with (ROOT / "results" / "boundary_summary.csv").open() as f:
        for r in csv.DictReader(f):
            rows[(r["dataset"], r["method"])] = r
    return rows


def fig_bf_curves(summary: dict, dataset: str) -> None:
    """BF score vs tolerance (1, 2, 5 px), one line per method."""
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    tols = [1, 2, 5]
    methods = ["gac_oracle", "unet", "dals", "color", "deep_snake", "kass_oracle"]
    for m in methods:
        r = summary[(dataset, m)]
        ys = [float(r[f"bf_{t}px"]) for t in tols]
        ax.plot(tols, ys, "o-", color=COLORS[m], lw=2, ms=7, label=LABELS[m])
        for t, y in zip(tols, ys):
            ax.text(t, y + 0.014, f"{y:.2f}", color=COLORS[m], ha="center", fontsize=7)
    ax.set_xticks(tols)
    ax.set_xlabel("Tolerance (px)")
    ax.set_ylabel("Boundary F-score")
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    label = "FLAME-3 (thermal GT)" if dataset == "flame3" else "FLAME-1 (visible GT)"
    ax.set_title(f"{label}: Boundary F-score vs tolerance.\n"
                 "Steeper rise = method places boundaries roughly right but\n"
                 "not pixel-precise; flat-high = pixel-accurate already.")
    fig.savefig(FIGS / f"{dataset}_bf_curves.png")
    plt.close(fig)


def fig_chamfer_bars(summary: dict) -> None:
    """Polygon Chamfer (lower is better) per method, both datasets side-by-side."""
    methods = ["unet", "dals", "deep_snake", "gac_oracle", "color", "kass_oracle"]
    f3 = [float(summary[("flame3", m)]["poly_chamfer_px"]) for m in methods]
    f1 = [float(summary[("flame1", m)]["poly_chamfer_px"]) for m in methods]
    x = np.arange(len(methods))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    b1 = ax.bar(x - w / 2, f3, w, color="#1f77b4", edgecolor="black", lw=0.4,
                label="FLAME-3 (n=592 / 94)")
    b2 = ax.bar(x + w / 2, f1, w, color="#d62728", edgecolor="black", lw=0.4,
                label="FLAME-1 (n=2001 / 301)")
    for bars, vals in [(b1, f3), (b2, f1)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5, f"{v:.1f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in methods], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Polygon Chamfer distance (px, lower = better)")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=8)
    ax.set_title("Boundary localization: polygon set-to-set Chamfer.\n"
                 "On FLAME-1 GAC has the most accurate boundary even though "
                 "its IoU loses to U-Net by 0.18.")
    fig.savefig(FIGS / "chamfer_bars.png")
    plt.close(fig)


def fig_iou_vs_chamfer(summary: dict, dataset: str) -> None:
    """Scatter: IoU vs PolyChamfer. Shows where region- and boundary-rankings diverge."""
    fig, ax = plt.subplots(figsize=(6.0, 4.3))
    for m, label in LABELS.items():
        r = summary[(dataset, m)]
        ax.scatter(float(r["iou"]), float(r["poly_chamfer_px"]),
                   s=140, color=COLORS[m], edgecolor="black", lw=0.5, zorder=3)
        ax.annotate(label, (float(r["iou"]), float(r["poly_chamfer_px"])),
                    xytext=(7, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Test IoU (higher = better)")
    ax.set_ylabel("Polygon Chamfer px (lower = better)")
    ax.grid(alpha=0.3, zorder=1)
    label = "FLAME-3 (thermal GT)" if dataset == "flame3" else "FLAME-1 (visible GT)"
    ax.set_title(f"{label}: region quality vs boundary quality.\n"
                 "Methods in the bottom-right corner are best on both axes.")
    # Invert y so up-and-right = better on both axes (more intuitive).
    ax.invert_yaxis()
    fig.savefig(FIGS / f"{dataset}_iou_vs_chamfer.png")
    plt.close(fig)


def main() -> None:
    summary = load_summary()
    fig_bf_curves(summary, "flame3")
    fig_bf_curves(summary, "flame1")
    fig_chamfer_bars(summary)
    fig_iou_vs_chamfer(summary, "flame3")
    fig_iou_vs_chamfer(summary, "flame1")
    print(f"wrote 5 figures to {FIGS}")


if __name__ == "__main__":
    main()
