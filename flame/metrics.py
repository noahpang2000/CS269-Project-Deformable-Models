"""Segmentation metrics: IoU and Dice on binary masks (empty/empty -> 1.0)."""
from __future__ import annotations

import numpy as np

def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred > 0, gt > 0
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, g).sum()) / float(union)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred > 0, gt > 0
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(p, g).sum()) / float(denom)
