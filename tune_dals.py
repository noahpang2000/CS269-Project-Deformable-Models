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
