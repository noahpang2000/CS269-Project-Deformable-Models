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
