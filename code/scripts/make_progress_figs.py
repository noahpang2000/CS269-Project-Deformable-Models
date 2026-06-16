"""Generate figures for the progress report. Outputs to report/figs/progress/."""
import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "report" / "figs" / "progress"
OUT.mkdir(parents=True, exist_ok=True)

# ---- Fig 1: FLAME-3 vs FLAME-1 scoreboard (the dataset-dominates story) ----
import csv
agg = {}
for r in csv.DictReader(open(ROOT / "results/boundary_summary.csv")):
    agg[(r["dataset"], r["method"])] = float(r["iou"])
methods = ["color", "kass_oracle", "gac_oracle", "unet", "dals", "deep_snake"]
labels = ["Color\nfloor", "Kass\n(oracle)", "GAC\n(oracle)", "U-Net", "DALS", "Deep\nSnake"]
f3 = [agg.get(("flame3", m), 0) for m in methods]
f1 = [agg.get(("flame1", m), 0) for m in methods]
x = np.arange(len(methods)); w = 0.38
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.bar(x - w/2, f3, w, label="FLAME-3 (thermal GT)", color="#c44")
ax.bar(x + w/2, f1, w, label="FLAME-1 (visible flame)", color="#48a")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Test IoU"); ax.set_ylim(0, 0.85)
ax.set_title("Dataset dominates: same methods, two definitions of 'fire'")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
for xi, v in zip(x - w/2, f3): ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=6)
for xi, v in zip(x + w/2, f1): ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=6)
fig.tight_layout(); fig.savefig(OUT / "scoreboard.png", dpi=140); plt.close(fig)
print("scoreboard.png")

# ---- Fig 2: Deep Snake diagnostic progression (FLAME-1) ----
stages = ["Baseline\n(whole-box,\nconf 0.30)", "Lower conf\n0.10", "Per-instance\nboxes\n(oracle)",
          "+ stride-4\nfeats", "+ Chamfer\nloss", "Synthetic\ncontrol"]
vals = [0.050, 0.218, 0.497, 0.463, 0.463, 0.970]
colors = ["#999", "#7a9", "#5b8", "#5b8", "#5b8", "#2a6"]
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.bar(range(len(stages)), vals, color=colors)
ax.set_xticks(range(len(stages))); ax.set_xticklabels(stages, fontsize=7)
ax.set_ylabel("IoU"); ax.set_ylim(0, 1.0)
ax.axhline(0.658, ls="--", c="#48a", lw=1); ax.text(0.1, 0.67, "deep_snake_simple 0.658", fontsize=7, color="#48a")
ax.set_title("Deep Snake (paper) diagnostic: each fix, and the positive control")
for i, v in enumerate(vals): ax.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "diagnostic.png", dpi=140); plt.close(fig)
print("diagnostic.png")

# ---- Fig 3: synthetic positive control sample (RGB + GT) ----
img = cv2.cvtColor(cv2.imread(str(ROOT / "data/SYNTH/images/image_0.jpg")), cv2.COLOR_BGR2RGB)
gt = cv2.imread(str(ROOT / "data/SYNTH/Masks/image_0.png"), cv2.IMREAD_GRAYSCALE)
fig, axs = plt.subplots(1, 2, figsize=(6, 3))
axs[0].imshow(img); axs[0].set_title("Synthetic blob (RGB)", fontsize=9); axs[0].axis("off")
axs[1].imshow(gt, cmap="gray"); axs[1].set_title("GT mask", fontsize=9); axs[1].axis("off")
fig.suptitle("Positive control: 1 large smooth closed object (Deep Snake -> IoU 0.970)", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "synth_sample.png", dpi=140); plt.close(fig)
print("synth_sample.png")

# ---- Fig 4: U-Net is not spotty -- pred components match GT ----
import torch
from code.flame.deep.unet import UNet
from code.flame.data import load_frame
from code.flame.splits import make_splits
m = UNet().cuda().eval()
m.load_state_dict(torch.load(ROOT / "models/flame1_unet.pt", map_location="cuda", weights_only=True))
fid = make_splits(dataset="flame1")["test"][2]
fr = load_frame(fid, dataset="flame1")
r = cv2.resize(fr.rgb, (512, 512)); x = torch.from_numpy(r).float().permute(2, 0, 1)[None].cuda() / 255.
with torch.no_grad():
    pred = (torch.sigmoid(m(x))[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
gt = cv2.resize(fr.gt_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
fig, axs = plt.subplots(1, 3, figsize=(9, 3))
axs[0].imshow(r); axs[0].set_title("FLAME-1 RGB", fontsize=9); axs[0].axis("off")
axs[1].imshow(gt, cmap="hot"); axs[1].set_title("GT mask", fontsize=9); axs[1].axis("off")
axs[2].imshow(pred, cmap="hot"); axs[2].set_title("U-Net raw output (no postproc)", fontsize=9); axs[2].axis("off")
fig.tight_layout(); fig.savefig(OUT / "unet_output.png", dpi=140); plt.close(fig)
print("unet_output.png")
print("DONE ->", OUT)
