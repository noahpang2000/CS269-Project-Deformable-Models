"""Synthetic single-blob dataset --- a controlled diagnostic for Deep Snake.

Each image has ONE large, smooth, closed, blob-like region (a star-perturbed
ellipse) on a textured background, with an exact ground-truth mask. This is the
case offset-based contour models are designed for: a single object whose
box->octagon init is already close and needs only smooth deformation. If Deep
Snake cannot do well here, the implementation is broken; if it does well here but
not on FLAME-1, the FLAME result is a data/method mismatch (the conclusion we
reached) rather than a bug.

Layout mirrors FLAME-1 so flame.data can read it:
    data/SYNTH/images/image_<i>.jpg
    data/SYNTH/Masks/image_<i>.png   (uint8 {0,1})
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNTH_DIR = PROJECT_ROOT / "data" / "SYNTH"
IMG_DIR = SYNTH_DIR / "images"
MASK_DIR = SYNTH_DIR / "Masks"

IMG_SIZE = 512


def _blob_mask(rng: np.random.Generator, size: int) -> np.ndarray:
    """One large smooth closed blob: an ellipse whose radius is perturbed by a
    few low-frequency sinusoids (star-shaped but smooth, single closed contour)."""
    cx = rng.uniform(0.4, 0.6) * size
    cy = rng.uniform(0.4, 0.6) * size
    base_r = rng.uniform(0.18, 0.30) * size          # large: 18-30% of the frame
    ax, ay = rng.uniform(0.8, 1.2, size=2)            # mild ellipticity
    # 3 low-frequency harmonics -> smooth, non-circular but still convex-ish.
    ks = rng.integers(2, 5, size=3)
    amps = rng.uniform(0.05, 0.15, size=3) * base_r
    phs = rng.uniform(0, 2 * np.pi, size=3)

    th = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    r = base_r + sum(a * np.sin(k * th + p) for k, a, p in zip(ks, amps, phs))
    xs = cx + ax * r * np.cos(th)
    ys = cy + ay * r * np.sin(th)
    pts = np.stack([xs, ys], axis=1).round().astype(np.int32)
    mask = np.zeros((size, size), np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def _render(rng: np.random.Generator, mask: np.ndarray, size: int) -> np.ndarray:
    """Textured background + a distinctly-colored, textured foreground blob."""
    bg = rng.integers(40, 110, size=(size, size, 3), dtype=np.uint8)
    bg = cv2.GaussianBlur(bg, (0, 0), 4)
    fg_color = np.array([rng.integers(150, 255), rng.integers(60, 160),
                         rng.integers(40, 120)], dtype=np.float32)
    fg = (fg_color[None, None] + rng.normal(0, 18, (size, size, 3))).clip(0, 255).astype(np.uint8)
    img = np.where(mask[..., None] > 0, fg, bg)
    img = (img.astype(np.float32) + rng.normal(0, 6, img.shape)).clip(0, 255).astype(np.uint8)
    return img


def generate(n: int = 400, size: int = IMG_SIZE, seed: int = 0) -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        mask = _blob_mask(rng, size)
        img = _render(rng, mask, size)
        cv2.imwrite(str(IMG_DIR / f"image_{i}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(MASK_DIR / f"image_{i}.png"), mask)  # {0,1}
    print(f"wrote {n} synthetic blob frames -> {SYNTH_DIR}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    generate(args.n, seed=args.seed)
