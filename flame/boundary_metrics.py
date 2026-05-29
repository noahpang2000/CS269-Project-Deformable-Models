"""Boundary-aware segmentation metrics.

Region metrics (IoU, Dice) reward correct interior fill and are insensitive to
how well a prediction's edge tracks the GT's edge. Contour-based methods (Kass,
GAC, Deep Snake) might localize boundaries more accurately than a per-pixel
U-Net even when their IoU is lower; these metrics test that hypothesis.

All inputs are uint8 binary masks (0 or >0) at matching shape. Pred and GT may
be both empty (degenerate) -- functions return their natural extremum then.
"""
from __future__ import annotations

import cv2
import numpy as np


def _boundary_pixels(mask: np.ndarray) -> np.ndarray:
    """1-pixel-wide boundary of a binary mask. uint8 {0,1}."""
    bm = (mask > 0).astype(np.uint8)
    if bm.sum() == 0:
        return bm
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(bm, kernel, iterations=1)
    return (bm - eroded).astype(np.uint8)


def _distance_to(mask_b: np.ndarray) -> np.ndarray:
    """Per-pixel Euclidean distance to the nearest True pixel in mask_b.

    If mask_b is empty, returns +inf everywhere (no target to measure to).
    """
    if mask_b.sum() == 0:
        return np.full(mask_b.shape, np.inf, dtype=np.float32)
    # cv2.distanceTransform measures distance to the nearest ZERO pixel, so we
    # invert. DIST_L2 + DIST_MASK_PRECISE -> Euclidean (sub-pixel-accurate).
    return cv2.distanceTransform(
        (mask_b == 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )


def boundary_fscore(pred: np.ndarray, gt: np.ndarray,
                    tol_px: float = 2.0) -> float:
    """Symmetric boundary F1 with a pixel tolerance (Csurka et al. 2013).

    Precision = fraction of pred boundary within tol_px of GT boundary.
    Recall    = fraction of GT  boundary within tol_px of pred boundary.
    F1        = harmonic mean.

    Both boundaries empty -> 1.0 (perfect on a true negative).
    One empty, one not    -> 0.0 (total miss).
    """
    pb = _boundary_pixels(pred)
    gb = _boundary_pixels(gt)
    pb_n, gb_n = int(pb.sum()), int(gb.sum())
    if pb_n == 0 and gb_n == 0:
        return 1.0
    if pb_n == 0 or gb_n == 0:
        return 0.0
    d_to_g = _distance_to(gb)        # distance from every pixel to GT boundary
    d_to_p = _distance_to(pb)
    precision = float(((pb > 0) & (d_to_g <= tol_px)).sum()) / pb_n
    recall    = float(((gb > 0) & (d_to_p <= tol_px)).sum()) / gb_n
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def hausdorff_and_avg(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Symmetric Hausdorff distance and average symmetric surface distance.

    Hausdorff = max over both directions of (max distance from a boundary
    pixel to the nearest pixel in the other boundary). Pixels.
    Avg = mean of those same distances (less outlier-sensitive). Pixels.

    Both empty -> (0, 0). One empty -> (inf, inf).
    """
    pb = _boundary_pixels(pred)
    gb = _boundary_pixels(gt)
    pb_n, gb_n = int(pb.sum()), int(gb.sum())
    if pb_n == 0 and gb_n == 0:
        return 0.0, 0.0
    if pb_n == 0 or gb_n == 0:
        return float("inf"), float("inf")
    d_p_to_g = _distance_to(gb)[pb > 0]    # distances FROM pred boundary TO GT
    d_g_to_p = _distance_to(pb)[gb > 0]
    hd = max(float(d_p_to_g.max()), float(d_g_to_p.max()))
    # 95th percentile would also be sensible; mean is simpler and what
    # most boundary papers report alongside Hausdorff.
    avg = (float(d_p_to_g.mean()) + float(d_g_to_p.mean())) / 2.0
    return hd, avg


def _resample_polyline_equal_arc(pts: np.ndarray,
                                  spacing_px: float = 1.0,
                                  closed: bool = True) -> np.ndarray:
    """Resample a polyline so consecutive samples are ~`spacing_px` apart.

    Returns shape (M, 2). Used to give every polygon a sample density
    proportional to its arc length, so longer polygons contribute more points
    to the set-distance computation -- the standard convention in the
    deformable-contour metric literature.
    """
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) < 2:
        return pts
    if closed:
        seq = np.vstack([pts, pts[:1]])
    else:
        seq = pts
    seg = np.linalg.norm(np.diff(seq, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total < 1e-6:
        return pts[:1]
    n = max(2, int(round(total / spacing_px)))
    targets = np.linspace(0.0, total, n, endpoint=not closed)
    xs = np.interp(targets, cum, seq[:, 0])
    ys = np.interp(targets, cum, seq[:, 1])
    return np.stack([xs, ys], axis=1).astype(np.float32)


def polygons_from_mask(mask: np.ndarray, min_area_px: int = 5) -> list[np.ndarray]:
    """Extract per-CC outer contours as point-arrays from a binary mask.

    Used when a method only produces a mask (U-Net/DALS/GAC/color) and we want
    polygon-aware metrics for cross-method comparison. For methods with native
    polygon output (Kass, Deep Snake), feed those polygons in directly instead.
    """
    bm = (mask > 0).astype(np.uint8)
    if bm.sum() == 0:
        return []
    contours, _ = cv2.findContours(bm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area_px:
            continue
        polys.append(c[:, 0, :].astype(np.float32))   # (N, 2)
    return polys


def _polygon_set_points(polys: list[np.ndarray],
                         spacing_px: float = 1.0) -> np.ndarray:
    """Flatten a list of polygons to one (N, 2) sample array, equal-arc-resampled."""
    if not polys:
        return np.empty((0, 2), dtype=np.float32)
    parts = [_resample_polyline_equal_arc(p, spacing_px=spacing_px, closed=True)
             for p in polys]
    return np.concatenate([p for p in parts if len(p) > 0], axis=0)


def polygon_chamfer_and_hausdorff(
        pred_polys: list[np.ndarray],
        gt_polys: list[np.ndarray],
        spacing_px: float = 1.0,
        sample_cap: int | None = 4000) -> tuple[float, float]:
    """Set-to-set symmetric polygon Chamfer (mean) and Hausdorff (max), pixels.

    Both sides resampled to equal arc-length spacing then collapsed to a single
    point set. For each pred sample, find nearest GT sample; for each GT
    sample, find nearest pred sample. Chamfer = mean of both directions'
    means; Hausdorff = max of both directions' maxes.

    Standard in deformable-contour benchmarks (Peng 2020 Deep Snake;
    Ling 2019 Curve-GCN). Naturally handles multi-component cases.

    sample_cap: downsample each side to <= sample_cap points before the
    pairwise distance computation (an O(NM) cost). 4000 keeps the all-pairs
    matrix under ~64MB and metric noise under ~0.1 px on 2k-pt polygons.

    Both empty -> (0, 0). One empty -> (inf, inf).
    """
    pred_pts = _polygon_set_points(pred_polys, spacing_px=spacing_px)
    gt_pts   = _polygon_set_points(gt_polys,   spacing_px=spacing_px)
    if len(pred_pts) == 0 and len(gt_pts) == 0:
        return 0.0, 0.0
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("inf"), float("inf")
    if sample_cap is not None:
        if len(pred_pts) > sample_cap:
            idx = np.linspace(0, len(pred_pts) - 1, sample_cap).astype(int)
            pred_pts = pred_pts[idx]
        if len(gt_pts) > sample_cap:
            idx = np.linspace(0, len(gt_pts) - 1, sample_cap).astype(int)
            gt_pts = gt_pts[idx]
    # Pairwise Euclidean distance matrix [P, G].
    diff = pred_pts[:, None, :] - gt_pts[None, :, :]
    d = np.sqrt((diff * diff).sum(axis=2))
    d_p_to_g = d.min(axis=1)
    d_g_to_p = d.min(axis=0)
    chamfer = float((d_p_to_g.mean() + d_g_to_p.mean()) / 2.0)
    hausdorff = float(max(d_p_to_g.max(), d_g_to_p.max()))
    return chamfer, hausdorff


def boundary_iou(pred: np.ndarray, gt: np.ndarray,
                 band_px: int = 2) -> float:
    """Boundary IoU (Cheng et al. 2021): IoU restricted to a band around the
    GT and pred boundaries. Removes interior-fill credit.

    band_px: thickness on each side of the boundary considered.
    """
    bm_p = (pred > 0).astype(np.uint8)
    bm_g = (gt > 0).astype(np.uint8)
    if bm_p.sum() == 0 and bm_g.sum() == 0:
        return 1.0
    # Build a band by morphological gradient (dilate - erode), then take its
    # intersection with the mask to get the inner band only -- the original
    # paper's formulation (only counts the inner-band pixels, not the outer ones).
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * band_px + 1, 2 * band_px + 1))
    eroded_p = cv2.erode(bm_p, k, iterations=1)
    eroded_g = cv2.erode(bm_g, k, iterations=1)
    band_p = (bm_p & ~eroded_p).astype(bool)
    band_g = (bm_g & ~eroded_g).astype(bool)
    inter = int((band_p & band_g).sum())
    union = int((band_p | band_g).sum())
    if union == 0:
        return 1.0
    return inter / union
