"""Additional progress-report figures (ablations, learning curve, init study, boundary)."""
import sys
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "figs" / "progress"
OUT.mkdir(parents=True, exist_ok=True)


def miou(name):
    p = ROOT / f"results/{name}_per_frame.csv"
    r = list(csv.DictReader(open(p)))
    return np.mean([float(x["iou"]) for x in r])


# Fig: learning curve (data starvation, FLAME-3)
rows = list(csv.DictReader(open(ROOT / "results/learning_curve.csv")))
frac = [float(r["fraction"]) * 100 for r in rows]
iou = [float(r["best_val_iou"]) for r in rows]
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.plot(frac, iou, "o-", color="#48a", lw=2, ms=7)
for f, v in zip(frac, iou): ax.text(f, v + 0.004, f"{v:.3f}", ha="center", fontsize=7)
ax.set_xlabel("% of training data"); ax.set_ylabel("best val IoU")
ax.set_title("FLAME-3 is data-starved (U-Net learning curve)")
ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(OUT / "learning_curve.png", dpi=140); plt.close(fig)
print("learning_curve.png")

# Fig: contour initialization study (FLAME-3)
inits = ["color\n(no oracle)", "U-Net\nseed", "oracle\n(GT)"]
kass = [miou("kass_color"), miou("kass_unet"), miou("kass_oracle")]
gac = [miou("gac_color"), miou("gac_unet"), miou("gac_oracle")]
x = np.arange(3); w = 0.38
fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.bar(x - w/2, kass, w, label="Kass Snakes", color="#c97")
ax.bar(x + w/2, gac, w, label="GAC", color="#6a8")
ax.set_xticks(x); ax.set_xticklabels(inits, fontsize=8); ax.set_ylabel("IoU (FLAME-3)")
ax.set_title("Initialization is everything for classical contours")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
for xi, v in zip(x - w/2, kass): ax.text(xi, v + 0.005, f"{v:.2f}", ha="center", fontsize=7)
for xi, v in zip(x + w/2, gac): ax.text(xi, v + 0.005, f"{v:.2f}", ha="center", fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "init_study.png", dpi=140); plt.close(fig)
print("init_study.png")

# Fig: U-Net / DALS improvement ablations (FLAME-3)
unet_v = {"base": "unet", "+R-G": "unet_rgb_rg", "+HSV+RG": "unet_rgb_hsv_rg",
          "aug": "unet_aug", "aug2": "unet_aug2", "filt": "unet_filt",
          "filt+aug": "unet_filt_aug", "ft": "unet_ft"}
dals_v = {"base": "dals", "ft": "dals_ft", "filt": "dals_filt",
          "filt+aug": "dals_filt_aug", "aug2": "dals_aug2"}
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, title, vd, c in [(axs[0], "U-Net ablations", unet_v, "#48a"),
                         (axs[1], "DALS ablations", dals_v, "#a58")]:
    labels = list(vd.keys()); vals = [miou(vd[k]) for k in labels]
    ax.bar(range(len(labels)), vals, color=c)
    ax.axhline(vals[0], ls="--", c="#999", lw=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_title(f"{title} (FLAME-3)", fontsize=9); ax.set_ylabel("IoU"); ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(vals): ax.text(i, v + 0.003, f"{v:.2f}", ha="center", fontsize=6)
fig.tight_layout(); fig.savefig(OUT / "ablations.png", dpi=140); plt.close(fig)
print("ablations.png")

# Fig: boundary metrics -- IoU vs Boundary-F (2px), FLAME-1, to show contour quality angle
b = {}
for r in csv.DictReader(open(ROOT / "results/boundary_summary.csv")):
    if r["dataset"] == "flame1":
        b[r["method"]] = (float(r["iou"]), float(r["bf_2px"]))
order = ["color", "gac_oracle", "gac_unet", "deep_snake", "dals", "unet"]
order = [m for m in order if m in b]
iouv = [b[m][0] for m in order]; bfv = [b[m][1] for m in order]
x = np.arange(len(order)); w = 0.38
fig, ax = plt.subplots(figsize=(7.5, 3.2))
ax.bar(x - w/2, iouv, w, label="region IoU", color="#48a")
ax.bar(x + w/2, bfv, w, label="Boundary-F @2px", color="#e90")
ax.set_xticks(x); ax.set_xticklabels(order, fontsize=7, rotation=20, ha="right")
ax.set_title("Region vs boundary quality (FLAME-1)"); ax.set_ylabel("score"); ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "boundary.png", dpi=140); plt.close(fig)
print("boundary.png")
print("DONE")
