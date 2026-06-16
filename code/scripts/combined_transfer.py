"""Does pooling FLAME 1+2+3 help or hurt each dataset?

For every deep method, evaluate the dataset-specific checkpoint and the
combined-trained checkpoint on each dataset's own held-out test split, so we can
see whether training on the pooled set transfers (helps) or dilutes (hurts) each
individual dataset. Writes results/combined_transfer.csv.

Specific checkpoints:  flame1 -> models/flame1_<m>.pt,  flame2 -> models/flame2_<m>.pt,
                       flame3 -> models/<m>.pt (the unprefixed FLAME-3 default).
Combined checkpoint:   models/combined_<m>.pt.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from flame.data import load_frame
from flame.splits import make_splits
from flame.metrics import iou as iou_fn, dice as dice_fn
import run_deep as R

SIZE, DEVICE, MAX_SIDE = 512, "cuda" if torch.cuda.is_available() else "cpu", 1024
METHODS = ["unet", "dals", "deep_snake_simple"]
DATASETS = ["flame1", "flame2", "flame3"]


def specific_ckpt(method, ds):
    return ROOT / "models" / (f"{method}.pt" if ds == "flame3" else f"{ds}_{method}.pt")


@torch.no_grad()
def eval_model(ckpt, method, frames):
    if not ckpt.exists():
        return None
    model = R.build_model(method).to(DEVICE).eval()
    try:
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    except RuntimeError as e:
        # The original FLAME-1/3 DALS checkpoints predate the DALS rewrite
        # (mu/lam1/lam2 vs phi_head/params_head); skip those rather than crash.
        print(f"  [skip {ckpt.name}: incompatible state_dict] {str(e).splitlines()[0]}")
        return None
    ious, dices = [], []
    for fr in frames:
        pred = R.predict_native(method, model, fr, SIZE, DEVICE, conf_threshold=0.3)
        ious.append(iou_fn(pred, fr.gt_mask)); dices.append(dice_fn(pred, fr.gt_mask))
    return float(np.mean(ious)), float(np.mean(dices))


def main():
    rows = []
    for ds in DATASETS:
        test = make_splits(dataset=ds)["test"]
        frames = [load_frame(f, dataset=ds, max_side=MAX_SIDE) for f in test]
        print(f"[{ds}] {len(frames)} test frames")
        for m in METHODS:
            spec = eval_model(specific_ckpt(m, ds), m, frames)
            comb = eval_model(ROOT / "models" / f"combined_{m}.pt", m, frames)
            if spec is None or comb is None:
                print(f"  {m}: missing ckpt (spec={spec is not None}, comb={comb is not None})")
                continue
            rows.append({"method": m, "dataset": ds,
                         "specific_iou": f"{spec[0]:.4f}", "specific_dice": f"{spec[1]:.4f}",
                         "combined_iou": f"{comb[0]:.4f}", "combined_dice": f"{comb[1]:.4f}"})
            print(f"  {m:18s} specific IoU={spec[0]:.3f}  combined IoU={comb[0]:.3f}  "
                  f"(d={comb[0]-spec[0]:+.3f})")
    out = ROOT / "results" / "combined_transfer.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "dataset", "specific_iou",
                                          "specific_dice", "combined_iou", "combined_dice"])
        w.writeheader(); w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
