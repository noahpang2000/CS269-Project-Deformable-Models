"""Validation-set boundary F-score for the hyperparameter search.

Mirrors run_deep.evaluate_split's prediction path (predict_native) but scores
the symmetric boundary F-score @2px instead of IoU/Dice. Used by tune_dals.py
as the boundary objective.
"""
from __future__ import annotations

import numpy as np
import torch

from flame.boundary_metrics import boundary_fscore
from flame.data import load_frame
from flame.metrics import iou
from run_deep import predict_native


@torch.no_grad()
def val_iou_and_bf(method: str, model, frame_ids: list[str], threshold_c: float,
                   size: int, device, dataset: str = "flame3",
                   eval_max_side: int | None = None,
                   tol_px: float = 2.0) -> tuple[float, float]:
    """Mean val IoU and mean val boundary F-score@tol_px over frame_ids."""
    model.eval()
    ious, bfs = [], []
    for fid in frame_ids:
        frame = load_frame(fid, threshold_c=threshold_c, dataset=dataset,
                           max_side=eval_max_side)
        pred = predict_native(method, model, frame, size, device)
        ious.append(iou(pred, frame.gt_mask))
        bfs.append(boundary_fscore(pred, frame.gt_mask, tol_px=tol_px))
    return float(np.mean(ious)), float(np.mean(bfs))
