"""Learning-curve diagnostic: is the U-Net data-starved or information-limited?

Train on increasing fractions of the train split (val + test held FIXED) and
record best val IoU at each size. If best val IoU is still rising at 100% of the
data, more data would help (data-starved). If it has plateaued well before 100%,
the ceiling is the information in the RGB, not the sample count (information-
limited). Same U-Net, loss, epochs, and eval as run_deep.py for comparability.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from flame.splits import make_splits
from flame.deep.dataset import FlameDataset, NET_SIZE
from flame.deep.losses import bce_dice
from run_deep import build_model, evaluate_split

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def train_one(train_ids, val_ids, epochs, size, bs, lr, device):
    ds = FlameDataset(train_ids, size=size, augment=True)
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0)
    model = build_model("unet", in_ch=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best = -1.0
    for ep in range(1, epochs + 1):
        model.train()
        for batch in loader:
            opt.zero_grad()
            loss = bce_dice(model(batch["image"].to(device)), batch["mask"].to(device))
            loss.backward()
            opt.step()
        val = evaluate_split("unet", model, val_ids, 150.0, size, device)
        best = max(best, val["iou"])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--size", type=int, default=NET_SIZE)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    splits = make_splits()
    train_all, val_ids = splits["train"], splits["val"]   # val FIXED across all points
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(train_all))               # subsample a fixed shuffled prefix

    rows = []
    print(f"learning curve: {len(train_all)} max train, {len(val_ids)} val (fixed), "
          f"{args.epochs} epochs each")
    for frac in args.fractions:
        n = max(1, int(round(frac * len(train_all))))
        subset = [train_all[i] for i in order[:n]]
        best = train_one(subset, val_ids, args.epochs, args.size,
                         args.batch_size, args.lr, device)
        print(f"  frac={frac:.2f}  n_train={n:3d}  best val IoU={best:.4f}")
        rows.append({"fraction": frac, "n_train": n, "best_val_iou": best})

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "learning_curve.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fraction", "n_train", "best_val_iou"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")

    # verdict heuristic: slope over the last segment
    if len(rows) >= 2:
        d = rows[-1]["best_val_iou"] - rows[-2]["best_val_iou"]
        print(f"\nlast-segment delta val IoU = {d:+.4f}  "
              f"({'still rising -> data-starved' if d > 0.01 else 'flat -> information-limited'})")


if __name__ == "__main__":
    main()
