"""Random-search hyperparameter optimization for DALS.

Two objectives (val IoU, val boundary F-score @2px) searched separately on each
dataset. Each trial trains DALS at reduced epochs with a sampled config; the best
config per (dataset, objective) is reported and (optionally) retrained at full
epochs with a test-set eval.

    python tune_dals.py --dataset flame3 --objective iou --trials 20 --epochs 25
    python tune_dals.py --dataset flame1 --objective bf  --trials 20 --epochs 25 --retrain

Outputs: results/dals_tune_<dataset>_<objective>.csv (ranked trials).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

N_ITER_CHOICES = [3, 5, 8, 12]
LOSS_CHOICES = ["bce_dice", "focal_tversky"]


def sample_config(rng: np.random.Generator) -> dict:
    """One random DALS config. FT params sampled but only used if loss=focal_tversky."""
    beta = float(rng.uniform(0.5, 0.8))
    return {
        "n_iter": int(rng.choice(N_ITER_CHOICES)),
        "mu": round(float(rng.uniform(0.1, 1.0)), 4),
        "lam1": round(float(rng.uniform(0.5, 2.0)), 4),
        "lam2": round(float(rng.uniform(0.5, 2.0)), 4),
        "dt": round(float(rng.uniform(0.05, 0.3)), 4),
        "loss": str(rng.choice(LOSS_CHOICES)),
        "ft_beta": round(beta, 4),
        "ft_alpha": round(1.0 - beta, 4),
        "ft_gamma": round(float(rng.uniform(0.5, 1.0)), 4),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["flame3", "flame1"], required=True)
    ap.add_argument("--objective", choices=["iou", "bf"], required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap train frames per trial (smoke testing)")
    ap.add_argument("--val-limit", dest="val_limit", type=int, default=None,
                    help="Cap val frames scored per trial (speed)")
    ap.add_argument("--retrain", action="store_true",
                    help="Retrain the winning config at --full-epochs and eval test")
    ap.add_argument("--full-epochs", dest="full_epochs", type=int, default=50)
    return ap.parse_args()


import torch
from torch.utils.data import DataLoader

from flame.deep.dataset import NET_SIZE, FlameDataset
from flame.deep.val_metrics import val_iou_and_bf
from flame.data import DEFAULT_THRESHOLD_C
from flame.splits import make_splits
from run_deep import build_model, compute_loss

THRESHOLD_C = DEFAULT_THRESHOLD_C


def _eval_max_side(dataset: str) -> int | None:
    return 1024 if dataset == "flame1" else None


def _load_max_side(dataset: str) -> int | None:
    return 1024 if dataset == "flame1" else None


def run_trial(cfg: dict, dataset: str, epochs: int, device,
              limit: int | None, val_limit: int | None) -> dict:
    """Train DALS with cfg for `epochs`, return cfg + final val IoU/BF."""
    splits = make_splits(dataset=dataset)
    train_ids, val_ids = splits["train"], splits["val"]
    if limit:
        train_ids = train_ids[:limit]
    if val_limit:
        val_ids = val_ids[:val_limit]

    ds = FlameDataset(train_ids, threshold_c=THRESHOLD_C, size=NET_SIZE,
                      augment=True, in_channels="rgb", aug_mode="light",
                      dataset=dataset, load_max_side=_load_max_side(dataset))
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

    dals_cfg = {k: cfg[k] for k in ("n_iter", "mu", "lam1", "lam2", "dt")}
    model = build_model("dals", dals_cfg=dals_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    ft = {"alpha": cfg["ft_alpha"], "beta": cfg["ft_beta"], "gamma": cfg["ft_gamma"]}

    for _ in range(epochs):
        model.train()
        for batch in loader:
            opt.zero_grad()
            loss = compute_loss("dals", model, batch, device, cfg["loss"], ft=ft)
            loss.backward()
            opt.step()

    val_iou, val_bf = val_iou_and_bf("dals", model, val_ids, THRESHOLD_C, NET_SIZE,
                                     device, dataset=dataset,
                                     eval_max_side=_eval_max_side(dataset))
    return {**cfg, "val_iou": round(val_iou, 4), "val_bf": round(val_bf, 4)}
