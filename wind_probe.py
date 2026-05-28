"""Feasibility probe: estimate wind direction from a SINGLE RGB frame via
smoke-plume principal-axis PCA, signed by the fire->plume vector.

This is a probe, not a pipeline component. It asks one question: is there a
usable per-frame wind cue in the smoke? Output overlays -> report/figs/wind/.

Pipeline (per frame, RGB only at estimation time; thermal GT used only to
anchor the fire source for signing + as a sanity reference):
  1. Segment a smoke candidate: bright + low-saturation + low-texture pixels
     that are NOT the (dark/obscured) fire and NOT green vegetation.
  2. PCA on smoke pixel coordinates -> principal axis = wind orientation (mod 180).
  3. Sign it: vector from the fire centroid to the smoke centroid picks the
     downwind half. (Fire centroid from GT here; in a real RGB-only run it would
     come from the U-Net/coarse seg.)
  4. Overlay arrow on RGB; report axis angle and an anisotropy score (how
     elongated the plume is = how confident the orientation).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flame.data import load_frame

OUT = Path(__file__).resolve().parent / "report" / "figs" / "wind"
OUT.mkdir(parents=True, exist_ok=True)


def smoke_mask(rgb: np.ndarray) -> np.ndarray:
    """Bright, desaturated, low-texture pixels = smoke candidate."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    bright = v > 120
    desat = s < 60
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # low local texture (smoke is smooth vs. textured vegetation)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    smooth = cv2.GaussianBlur(lap, (0, 0), 3) < 8
    m = (bright & desat & smooth).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    # keep the largest blob (the main plume)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if num <= 1:
        return m
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == big).astype(np.uint8)


def pca_axis(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return None
    pts = np.stack([xs, ys], 1).astype(np.float32)
    mean = pts.mean(0)
    cov = np.cov((pts - mean).T)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, -1]                      # principal direction
    anisotropy = float(evals[-1] / (evals[0] + 1e-6))  # elongation = confidence
    return mean, axis, anisotropy


def estimate(frame):
    m = smoke_mask(frame.rgb)
    res = pca_axis(m)
    if res is None:
        return None
    smoke_c, axis, aniso = res
    # sign by fire->smoke direction
    ys, xs = np.nonzero(frame.gt_mask > 0)
    if len(xs) == 0:
        return None
    fire_c = np.array([xs.mean(), ys.mean()], np.float32)
    ref = smoke_c - fire_c
    if np.dot(axis, ref) < 0:
        axis = -axis
    angle = np.degrees(np.arctan2(axis[1], axis[0]))
    return dict(mask=m, smoke_c=smoke_c, fire_c=fire_c, axis=axis,
               aniso=aniso, angle=angle)


def main():
    frames = ["00013", "00529", "00265", "00193", "00001", "00109"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, fid in zip(axes.ravel(), frames):
        frame = load_frame(fid)
        ax.imshow(frame.rgb)
        ax.contour(frame.gt_mask > 0, [0.5], colors="cyan", linewidths=1.0)
        e = estimate(frame)
        if e is None:
            ax.set_title(f"{fid}: no plume found", fontsize=9)
            ax.axis("off"); continue
        # tint smoke mask
        tint = np.zeros((*e["mask"].shape, 4)); tint[e["mask"] > 0] = [1, 1, 0, 0.18]
        ax.imshow(tint)
        L = 130
        sc = e["smoke_c"]
        ax.arrow(sc[0], sc[1], e["axis"][0] * L, e["axis"][1] * L,
                 color="red", width=3, head_width=18, length_includes_head=True)
        ax.plot(*e["fire_c"], "c+", ms=12, mew=2)
        ax.set_title(f"{fid}: wind {e['angle']:.0f} deg, "
                     f"anisotropy {e['aniso']:.1f}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Single-frame wind probe: smoke plume (yellow) principal axis "
                 "(red arrow = est. downwind), fire centroid (cyan +)", fontsize=11)
    fig.tight_layout()
    out = OUT / "wind_probe.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)
    # text summary
    print(f"\n{'frame':8s} {'angle_deg':>10s} {'anisotropy':>11s}")
    for fid in frames:
        e = estimate(load_frame(fid))
        if e: print(f"{fid:8s} {e['angle']:>10.0f} {e['aniso']:>11.1f}")
        else: print(f"{fid:8s} {'--':>10s} {'(no plume)':>11s}")


if __name__ == "__main__":
    main()
