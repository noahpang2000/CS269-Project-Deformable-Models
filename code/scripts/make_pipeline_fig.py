"""Paper Figure 1: Experimental design overview (NO results).
Grid-aligned, numbered stages, consistent styling, real thumbnails.
Output: report/figs/paper/pipeline.png. Canvas 24 x 13.5.
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from flame.data import load_frame, smoke_fraction
from flame.splits import make_splits

plt.rcParams["font.family"] = "DejaVu Sans"

# palette
INK = "#1f2933"; SUB = "#52606d"
C_DATA = ("#e8efff", "#3b7dd8"); C_SET = ("#fff4d6", "#e0a008")
C_INIT = ("#ffe8cc", "#e0780a"); C_METH = ("#efe7ff", "#7c4dd6")
C_MET = ("#fce0f3", "#d6469b")
FAM = {"classical": ("#d9ccf7", "#7c4dd6"),
       "pixel": ("#c6f0cf", "#27a35a"),
       "contour": ("#c8f2ec", "#0c9baa")}


def rgb_thumb(fid, ds, sz=200):
    return cv2.resize(load_frame(fid, dataset=ds).rgb, (sz, sz))


def thermal_thumb(fid, ds, sz=200):
    th = load_frame(fid, dataset=ds).thermal_c
    thn = np.clip((th - th.min()) / (th.ptp() + 1e-6), 0, 1)
    cm = cv2.applyColorMap((cv2.resize(thn, (sz, sz)) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)


fig, ax = plt.subplots(figsize=(24, 12.2))
ax.set_xlim(0, 24); ax.set_ylim(0, 12.2); ax.axis("off")
# (title/subtitle removed per request; rendered on a transparent background)


def shadow_box(x, y, w, h, fill, ec, lw=2.2):
    ax.add_patch(FancyBboxPatch((x + 0.06, y - 0.08), w, h, boxstyle="round,pad=0.04,rounding_size=0.16",
                                facecolor="#00000018", edgecolor="none", zorder=1))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.16",
                                facecolor=fill, edgecolor=ec, linewidth=lw, zorder=2))


def stage_header(x, y, w, num, label, ec):
    # numbered pill + stage title at top of a stage box
    ax.add_patch(plt.Circle((x + 0.55, y - 0.55), 0.32, facecolor=ec, edgecolor="white",
                            linewidth=1.5, zorder=5))
    ax.text(x + 0.55, y - 0.56, str(num), ha="center", va="center", fontsize=14,
            weight="bold", color="white", zorder=6)
    ax.text(x + 1.05, y - 0.55, label, ha="left", va="center", fontsize=16, weight="bold",
            color=INK, zorder=6)


def text(cx, cy, s, fs=11, color=INK, ha="center", weight="normal"):
    ax.text(cx, cy, s, ha=ha, va="center", fontsize=fs, color=color, weight=weight)


def imgbox(cx, top, s, img, caption, lc):
    x = cx - s / 2
    ax.imshow(img, extent=(x, x + s, top - s, top), zorder=4, aspect="auto")
    ax.add_patch(plt.Rectangle((x, top - s), s, s, fill=False, edgecolor=lc, linewidth=1.8, zorder=5))
    ax.text(cx, top - s - 0.30, caption, ha="center", va="top", fontsize=10, color=SUB, zorder=5)


def subcard(x, y, w, h, fam, title, body):
    fill, ec = FAM[fam]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                                facecolor=fill, edgecolor=ec, linewidth=1.8, zorder=3))
    ax.text(x + 0.35, y + h - 0.40, title, ha="left", va="top", fontsize=13, weight="bold", color=INK, zorder=4)
    ax.text(x + 0.35, y + h - 1.05, body, ha="left", va="top", fontsize=10, color=SUB, zorder=4)


def arrow(x1, y1, x2, y2, c=INK, lw=2.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=24,
                                 color=c, linewidth=lw, zorder=2,
                                 connectionstyle="arc3,rad=0"))


f1 = "image_1702"
f3 = next(f for f in make_splits(dataset="flame3")["test"]
          if smoke_fraction(load_frame(f, dataset="flame3").rgb) < 0.5
          and (load_frame(f, dataset="flame3").gt_mask > 0).sum() >= 200)

TOP = 11.2  # shared top baseline for the tall stage boxes

# ===== Stage 1: Datasets =====
shadow_box(0.5, 1.0, 5.3, TOP - 1.0, *C_DATA)
stage_header(0.5, TOP, 5.3, 1, "Datasets", C_DATA[1])
imgbox(1.95, TOP - 1.1, 1.85, rgb_thumb(f1, "flame1") / 255.0, "FLAME-1 (drone), 2001 fr\nvisible flame, snow", C_DATA[1])
imgbox(4.35, TOP - 1.1, 1.85, rgb_thumb(f3, "flame3") / 255.0, "FLAME-3 (aircraft), 622 fr\nsmoke-dominated", C_INIT[1])
imgbox(3.15, TOP - 5.0, 1.85, thermal_thumb(f3, "flame3") / 255.0, "FLAME-3 thermal pair", C_INIT[1])
text(3.15, 3.55, "Ground truth", 12, INK, weight="bold")
text(3.15, 2.85, "FLAME-1: hand-labeled masks\nFLAME-3: thermal thresholded ≥ 150°C", 10, SUB)
text(3.15, 1.65, "All methods see RGB only", 10.5, C_DATA[1], weight="bold")

# ===== Stage 2: Protocol =====
shadow_box(6.5, 6.4, 3.7, TOP - 6.4, *C_SET)
stage_header(6.5, TOP, 3.7, 2, "Protocol", C_SET[1])
text(8.35, 9.0, "70 / 15 / 15 split\n(contiguous, no leakage)\n512 × 512 inputs\nfull-smoke frames dropped",
     11, SUB)

# ===== Stage 3: Initialization =====
shadow_box(6.5, 1.0, 3.7, 4.8, *C_INIT)
stage_header(6.5, 5.8, 3.7, 3, "Init (contours)", C_INIT[1])
text(8.35, 3.85, "Three seeds:\noracle (from GT)\ncolour prior (R−G)\nU-Net mask", 11, SUB)
text(8.35, 1.75, "isolates the role\nof localization", 10, C_INIT[1], weight="bold")

# ===== Stage 4: Methods =====
M_X, M_Y, M_W, M_H = 10.9, 1.0, 9.0, TOP - 1.0
shadow_box(M_X, M_Y, M_W, M_H, *C_METH)
stage_header(M_X, TOP, M_W, 4, "Methods (five variants, three families)", C_METH[1])
# Vertically center the four cards in the area below the header.
cards = [
    ("classical", "Classical contours  ·  Snakes, GAC",
     "energy-minimizing curve / level set; no learning; needs an init"),
    ("pixel", "Deep pixel / level-set  ·  U-Net, DALS",
     "U-Net: per-pixel mask.  DALS: learned Chan–Vese level set over CNN features"),
    ("contour", "Deep contour (simple)  ·  Deep Snake",
     "U-Net coarse mask seeds the contour → circular-conv vertex-offset deformation"),
    ("contour", "Deep contour (paper)  ·  Deep Snake",
     "CenterNet detector → octagon per box → vertex-offset deformation"),
]
cw, cx0 = 8.4, 11.3
card_h, gap = 1.7, 0.55
header_pad = 1.15                                   # space reserved for the header row
region_top = TOP - header_pad                        # usable area top
region_bot = M_Y + 0.45                              # usable area bottom (margin)
stack_h = len(cards) * card_h + (len(cards) - 1) * gap
y_start = (region_top + region_bot) / 2 + stack_h / 2 - card_h   # top card's y
for i, (fam, title, body) in enumerate(cards):
    subcard(cx0, y_start - i * (card_h + gap), cw, card_h, fam, title, body)

# ===== Stage 5: Metrics =====
shadow_box(20.6, 4.7, 3.0, 4.0, *C_MET)
stage_header(20.6, 8.7, 3.0, 5, "Metrics", C_MET[1])
text(22.1, 6.5, "Dice\n(region)\nBoundary-F1\n(contour)\nIoU", 11.5, SUB)

# ===== flow arrows (clean, axis-aligned where possible) =====
arrow(5.8, 8.6, 6.5, 8.6)                 # datasets -> protocol
arrow(5.8, 3.4, 6.5, 3.4, C_INIT[1])      # datasets -> init
arrow(10.2, 8.6, 10.9, 7.4)               # protocol -> methods
arrow(10.2, 3.4, 10.9, 4.4, C_INIT[1])    # init -> methods (contours)
arrow(19.9, 6.6, 20.6, 6.6)               # methods -> metrics

# ===== legend (family colour key) =====
leg = [Line2D([0],[0], marker='s', color='none', markerfacecolor=FAM['classical'][0],
              markeredgecolor=FAM['classical'][1], markersize=14, label='classical (no learning)'),
       Line2D([0],[0], marker='s', color='none', markerfacecolor=FAM['pixel'][0],
              markeredgecolor=FAM['pixel'][1], markersize=14, label='deep pixel / level-set'),
       Line2D([0],[0], marker='s', color='none', markerfacecolor=FAM['contour'][0],
              markeredgecolor=FAM['contour'][1], markersize=14, label='deep contour')]
ax.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3,
          frameon=False, fontsize=12)

out = ROOT / "report" / "figs" / "paper" / "pipeline.png"
fig.savefig(out, dpi=160, bbox_inches="tight", transparent=True)
print("wrote", out)
