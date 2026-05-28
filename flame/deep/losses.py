"""Losses shared by the deep baselines."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    num = 2 * (prob * target).sum(dim=(1, 2, 3)) + eps
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()


def bce_dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


def focal_tversky(logits: torch.Tensor, target: torch.Tensor,
                  alpha: float = 0.3, beta: float = 0.7, gamma: float = 0.75,
                  eps: float = 1.0) -> torch.Tensor:
    """Focal-Tversky loss (Abraham & Khan 2019) for small/imbalanced foreground.

    Tversky index weights false negatives (beta) vs false positives (alpha); with
    beta > alpha it penalises under-segmentation harder, which is exactly the
    failure mode on faint, fragmented fire. gamma < 1 focuses on hard regions.
    """
    prob = torch.sigmoid(logits)
    tp = (prob * target).sum(dim=(1, 2, 3))
    fp = (prob * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - prob) * target).sum(dim=(1, 2, 3))
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return ((1 - tversky) ** gamma).mean()


def bce_focal_tversky(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE for calibration + Focal-Tversky for recall on the tiny foreground."""
    return F.binary_cross_entropy_with_logits(logits, target) + focal_tversky(logits, target)


def weighted_thermal_loss(pred_norm: torch.Tensor, target_norm: torch.Tensor,
                          thr_norm: float, hot_weight: float = 50.0,
                          band: float = 0.06) -> torch.Tensor:
    """Huber regression on normalized temperature, upweighting the pixels that
    matter for the >=150C support: hot pixels (target above threshold) and the
    band straddling the threshold (where a small temp error flips the label).

    Hot pixels are ~0.8% of the image, so an unweighted loss just predicts 'cool'
    everywhere; the weight map is what makes regression recover the support.
    """
    err = F.huber_loss(pred_norm, target_norm, reduction="none", delta=0.1)
    w = torch.ones_like(target_norm)
    w = w + (hot_weight - 1.0) * (target_norm >= thr_norm).float()
    w = w + (hot_weight * 0.5) * ((target_norm - thr_norm).abs() < band).float()
    return (err * w).sum() / w.sum()


def cyclic_contour_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Smooth-L1 between contours [B,N,2], minimised over cyclic shifts of gt.

    Handles the unknown start-vertex correspondence between two closed polygons.
    """
    b, n, _ = pred.shape
    best = None
    for s in range(n):
        shifted = torch.roll(gt, shifts=s, dims=1)
        cost = F.smooth_l1_loss(pred, shifted, reduction="none").mean(dim=(1, 2))
        best = cost if best is None else torch.minimum(best, cost)
    return best.mean()
