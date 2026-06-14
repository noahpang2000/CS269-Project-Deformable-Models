"""Figures for the FLAME-2 and combined (FLAME 1+2+3) reruns -> report/figs/paper/.

  f2_gallery.png       FLAME-2 RGB / colorized-IR / derived GT gallery
  flame2_metrics.png   Dice + Boundary-F1 bars (from results/flame2_metrics.csv)
  f2_methods.png       qualitative U-Net / DALS / Deep Snake on FLAME-2 test frames
  combined_metrics.png Dice + Boundary-F1 bars (from results/combined_metrics.csv)
  combined_transfer.png  per-dataset IoU: dataset-specific vs combined-trained model
                         (from results/combined_transfer.csv)

Run after training + compute_flame2_metrics + combined_transfer.
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "report" / "figs" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

from flame.data import load_frame, FLAME2_IR_DIR
from flame.splits import make_splits
from flame.metrics import iou as iou_fn, dice as dice_fn
import run_deep as R

SIZE, device = 512, "cuda" if torch.cuda.is_available() else "cpu"
RED, BLUE, GAP = (0, 0, 255), (255, 90, 0), 4


# ---- montage helpers (mirrors make_paper_figs_v2) ----
def overlay(bgr, mask, color, alpha=0.45):
    out = bgr.copy(); m = (mask > 0).astype(np.uint8)
    t = np.zeros_like(out); t[m > 0] = color
    out = cv2.addWeighted(out, 1.0, t, alpha, 0)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, 2)
    return out

def col(imgs, sz):
    out = []
    for im in imgs:
        if im.ndim == 2: im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        out.append(im)
    sep = np.full((GAP, sz, 3), 255, np.uint8)
    parts = []
    for i, im in enumerate(out):
        if i: parts.append(sep)
        parts.append(im)
    return np.vstack(parts)

def hstack_sep(cols):
    sep = np.full((cols[0].shape[0], GAP, 3), 255, np.uint8)
    parts = []
    for i, c in enumerate(cols):
        if i: parts.append(sep)
        parts.append(c)
    return np.hstack(parts)


def f2_test_ids(k=4, min_gt_px=400):
    out = []
    for fid in make_splits(dataset="flame2")["test"]:
        fr = load_frame(fid, dataset="flame2")
        if (fr.gt_mask > 0).sum() >= min_gt_px:
            out.append(fid)
        if len(out) >= k:
            break
    return out


def fig_f2_gallery():
    sz = 300
    cols = []
    for fid in f2_test_ids(4):
        fr = load_frame(fid, dataset="flame2")
        bgr = cv2.cvtColor(cv2.resize(fr.rgb, (sz, sz)), cv2.COLOR_RGB2BGR)
        gt = cv2.resize(fr.gt_mask, (sz, sz), interpolation=cv2.INTER_NEAREST)
        ir = cv2.imread(str(FLAME2_IR_DIR / f"{fid}.jpg"))
        ir = cv2.resize(ir, (sz, sz)) if ir is not None else np.zeros((sz, sz, 3), np.uint8)
        cols.append(col([bgr, ir, overlay(bgr, gt, RED)], sz))
    cv2.imwrite(str(OUT / "f2_gallery.png"), hstack_sep(cols)); print("f2_gallery.png")


@torch.no_grad()
def fig_f2_methods():
    sz = 260
    un = R.build_model("unet").to(device).eval()
    un.load_state_dict(torch.load(ROOT/"models/flame2_unet.pt", map_location=device, weights_only=True))
    da = R.build_model("dals").to(device).eval()
    da.load_state_dict(torch.load(ROOT/"models/flame2_dals.pt", map_location=device, weights_only=True))
    sn = R.build_model("deep_snake_simple").to(device).eval()
    sn.load_state_dict(torch.load(ROOT/"models/flame2_deep_snake_simple.pt", map_location=device, weights_only=True))

    def predict(m, method, fr):
        return cv2.resize(R.predict_native(method, m, fr, SIZE, device, 0.3),
                          (sz, sz), interpolation=cv2.INTER_NEAREST)

    cols = []
    for fid in f2_test_ids(3):
        fr = load_frame(fid, dataset="flame2")
        bgr = cv2.cvtColor(cv2.resize(fr.rgb, (sz, sz)), cv2.COLOR_RGB2BGR)
        gt = cv2.resize(fr.gt_mask, (sz, sz), interpolation=cv2.INTER_NEAREST)
        pu, pd, ps = (predict(un, "unet", fr), predict(da, "dals", fr),
                      predict(sn, "deep_snake_simple", fr))
        cols.append(col([bgr, overlay(bgr, gt, BLUE), overlay(bgr, pu, RED),
                         overlay(bgr, pd, RED), overlay(bgr, ps, RED)], sz))
    cv2.imwrite(str(OUT / "f2_methods.png"), hstack_sep(cols)); print("f2_methods.png")


# ---- bar charts ----
PRETTY = {"unet": "U-Net", "dals": "DALS", "deep_snake_simple": "Deep Snake",
          "color": "Color floor", "kass_oracle": "Kass (orc)", "gac_oracle": "GAC (orc)"}
ORDER = ["unet", "dals", "deep_snake_simple", "gac_oracle", "kass_oracle", "color"]


def _metrics_bar(csv_path, out_png, title):
    rows = {r["method"]: r for r in csv.DictReader(open(csv_path))}
    methods = [m for m in ORDER if m in rows]
    dice = [float(rows[m]["dice"]) for m in methods]
    bf = [float(rows[m]["bf_2px"]) for m in methods]
    labels = [PRETTY.get(m, m) for m in methods]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    for ax, vals, name in zip(axes, (dice, bf), ("Dice", "Boundary-F1@2px")):
        ax.bar(range(len(vals)), vals, color="#c0392b")
        ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1); ax.set_title(name, fontsize=11)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150); plt.close(fig); print(Path(out_png).name)


def fig_combined_transfer():
    path = ROOT / "results" / "combined_transfer.csv"
    if not path.exists():
        print("skip combined_transfer.png (no csv)"); return
    rows = list(csv.DictReader(open(path)))
    lut = {(r["method"], r["dataset"]): r for r in rows}   # robust to missing cells
    methods = ["unet", "dals", "deep_snake_simple"]
    dsets = ["flame1", "flame2", "flame3"]
    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 3.3), squeeze=False)
    for ax, m in zip(axes[0], methods):
        spec = [float(lut[(m, d)]["specific_iou"]) if (m, d) in lut else 0.0 for d in dsets]
        comb = [float(lut[(m, d)]["combined_iou"]) if (m, d) in lut else 0.0 for d in dsets]
        x = np.arange(len(dsets))
        ax.bar(x - 0.2, spec, 0.4, label="dataset-specific", color="#2980b9")
        ax.bar(x + 0.2, comb, 0.4, label="combined 1+2+3", color="#c0392b")
        ax.set_xticks(x); ax.set_xticklabels([d.upper() for d in dsets], fontsize=8)
        ax.set_ylim(0, 1); ax.set_title(PRETTY.get(m, m), fontsize=11); ax.set_ylabel("test IoU")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Pooled (FLAME 1+2+3) vs dataset-specific training, per-dataset test IoU", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "combined_transfer.png", dpi=150); plt.close(fig)
    print("combined_transfer.png")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default="")
    only = ap.parse_args().only
    figs = {
        "gallery": fig_f2_gallery,
        "methods": fig_f2_methods,
        "f2metrics": lambda: _metrics_bar(ROOT/"results/flame2_metrics.csv", OUT/"flame2_metrics.png", "FLAME-2 test results"),
        "combmetrics": lambda: _metrics_bar(ROOT/"results/combined_metrics.csv", OUT/"combined_metrics.png", "Combined (FLAME 1+2+3) test results"),
        "transfer": fig_combined_transfer,
    }
    for k, fn in figs.items():
        if only and k != only: continue
        try: fn()
        except Exception as e:
            import traceback; print(f"FIG {k} FAILED:", e); traceback.print_exc()
