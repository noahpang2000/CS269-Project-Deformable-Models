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


def chamfer_contour_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Symmetric Chamfer distance between two point sets [B,N,2], [B,M,2].

    For each predicted vertex, distance to the nearest GT point (pred->gt), and
    for each GT point, distance to the nearest predicted vertex (gt->pred). The
    gt->pred term is what prevents the contour from collapsing inward: a shrunken
    prediction leaves GT boundary points uncovered, which costs. Order-invariant,
    so it complements the vertex-order cyclic L1.
    """
    # pairwise squared distances [B, N, M]
    d2 = torch.cdist(pred, gt) ** 2
    pred_to_gt = d2.min(dim=2).values.mean(dim=1)   # each pred vertex -> nearest GT
    gt_to_pred = d2.min(dim=1).values.mean(dim=1)   # each GT point  -> nearest pred (anti-collapse)
    return (pred_to_gt + gt_to_pred).mean()


def snake_contour_loss(pred: torch.Tensor, gt: torch.Tensor,
                       chamfer_weight: float = 0.01) -> torch.Tensor:
    """Cyclic L1 (correspondence) + weighted Chamfer (boundary coverage).

    The Chamfer term is in squared pixels, so it is down-weighted to balance the
    per-coordinate L1 magnitude; it supplies the anti-collapse gradient the L1
    alone lacks.
    """
    return cyclic_contour_loss(pred, gt) + chamfer_weight * chamfer_contour_loss(pred, gt)
