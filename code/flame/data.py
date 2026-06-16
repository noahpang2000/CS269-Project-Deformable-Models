"""Frame loading and ground-truth generation for FLAME-3 (thermal) and FLAME-1 (hand-labeled)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tifffile

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# FLAME-3: RGB + Celsius thermal TIFF; GT is thermal >= 150C.
FIRE_DIR = PROJECT_ROOT / "data" / "FLAME3" / "Fire"
RGB_DIR = FIRE_DIR / "RGB" / "Corrected FOV"
THERMAL_DIR = FIRE_DIR / "Thermal" / "Celsius TIFF"

# FLAME-1: RGB + binary PNG mask (visible-flame, hand-labeled). No thermal.
FLAME1_DIR = PROJECT_ROOT / "data" / "FLAME1"
FLAME1_RGB_DIR = FLAME1_DIR / "images"
FLAME1_MASK_DIR = FLAME1_DIR / "Masks"

# SYNTH: synthetic single-blob diagnostic (flame/synth.py). Same layout as FLAME-1.
SYNTH_DIR = PROJECT_ROOT / "data" / "SYNTH"
SYNTH_RGB_DIR = SYNTH_DIR / "images"
SYNTH_MASK_DIR = SYNTH_DIR / "Masks"

# FLAME-2: paired RGB + colorized-IR *video*; no raw Celsius. We sample frames at
# ~1 fps (scripts/extract_flame2.py) and derive a thermal-hot GT from the IR
# palette (flame2_fire_mask below). Stored frame-wise like FLAME-1: RGB jpg +
# binary PNG mask, plus the colorized IR frame (figures only).
FLAME2_DIR = PROJECT_ROOT / "data" / "FLAME2"
FLAME2_RGB_DIR = FLAME2_DIR / "images"
FLAME2_MASK_DIR = FLAME2_DIR / "Masks"
FLAME2_IR_DIR = FLAME2_DIR / "ir"

# Datasets pooled into the "combined" set (FLAME 1+2+3). NOTE: FLAME-1 GT is
# *visible flame* (hand-labeled) while FLAME-2/3 GT is *thermal hot region*; the
# pooled set deliberately mixes these two label semantics (reported with that
# caveat).
COMBINED_DATASETS = ("flame1", "flame2", "flame3")

DEFAULT_DATASET = "flame3"
DEFAULT_THRESHOLD_C = 150.0
DEFAULT_MIN_BLOB_PX = 20


@dataclass
class Frame:
    frame_id: str
    rgb: np.ndarray
    thermal_c: np.ndarray | None   # None for FLAME-1 (no thermal channel)
    gt_mask: np.ndarray


def threshold_thermal(temp_c: np.ndarray, threshold_c: float = DEFAULT_THRESHOLD_C,
                      min_blob_px: int = DEFAULT_MIN_BLOB_PX) -> np.ndarray:
    binary = (temp_c >= threshold_c).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    if min_blob_px > 0 and binary.any():
        num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        keep = np.zeros_like(binary)
        for lbl in range(1, num):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_blob_px:
                keep[labels == lbl] = 1
        binary = keep
    return binary * 255


def flame2_fire_mask(ir_bgr: np.ndarray, min_blob_px: int = 60) -> np.ndarray:
    """Derive a thermal-hot GT mask from a FLAME-2 colorized-IR frame.

    FLAME-2 ships only palette-mapped IR video (no raw Celsius like FLAME-3), so
    we cannot threshold a temperature directly. In this palette cool background is
    dark red (R high, G/B low) and the hottest fire is bright white/cyan (G and B
    both high). The active fire shows up as *dense fine speckle* of these hot
    pixels rather than a solid blob, so 3x3 opening (as used for FLAME-3) erases
    it. Instead we measure the local *density* of hot speckle and keep the
    concentrated regions -- the analogue of FLAME-3's >=150C threshold.

    Tuned on the #6 pair; thresholds are at the IR native resolution (640x512).
    Input is BGR (OpenCV convention).
    """
    b, g, r = cv2.split(ir_bgr)
    v = cv2.cvtColor(ir_bgr, cv2.COLOR_BGR2HSV)[..., 2]
    hot = ((v > 185) & (g > 140) & (b > 140)).astype(np.float32)   # hottest white/cyan end
    density = cv2.GaussianBlur(hot, (0, 0), 9)                     # local concentration
    mask = (density > 0.06).astype(np.uint8)                       # dense-fire regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if min_blob_px > 0 and mask.any():
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros_like(mask)
        for lbl in range(1, num):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_blob_px:
                keep[labels == lbl] = 1
        mask = keep
    return mask * 255


DEFAULT_OCCLUSION_TAU = 0.95
_OCCLUSION_CACHE = PROJECT_ROOT / "results" / "occluded_frames.json"


def smoke_fraction(rgb: np.ndarray) -> float:
    """Fraction of the frame that is bright + desaturated + low-texture (smoke).

    A frame near 1.0 is essentially full-frame smoke: the fire location is not
    recoverable from RGB regardless of method (a degenerate, unsolvable input).
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    smooth = cv2.GaussianBlur(lap, (0, 0), 3) < 8
    return float(((v > 120) & (s < 60) & smooth).mean())


def occluded_frame_ids(tau: float = DEFAULT_OCCLUSION_TAU,
                       use_cache: bool = True) -> set[str]:
    """Frame ids whose smoke_fraction >= tau. Cached to results/ to avoid
    recomputing on every training run; delete the cache to force a refresh."""
    import json
    if use_cache and _OCCLUSION_CACHE.exists():
        data = json.loads(_OCCLUSION_CACHE.read_text())
        if abs(data.get("tau", -1) - tau) < 1e-9:
            return set(data["ids"])
    all_ids = sorted(p.stem for p in RGB_DIR.glob("*.JPG"))
    occ = [fid for fid in all_ids
           if smoke_fraction(cv2.cvtColor(cv2.imread(str(RGB_DIR / f"{fid}.JPG")),
                                          cv2.COLOR_BGR2RGB)) >= tau]
    _OCCLUSION_CACHE.parent.mkdir(exist_ok=True)
    _OCCLUSION_CACHE.write_text(json.dumps({"tau": tau, "ids": occ}))
    return set(occ)


def list_frame_ids(exclude_occluded: bool = False,
                   occlusion_tau: float = DEFAULT_OCCLUSION_TAU,
                   dataset: str = DEFAULT_DATASET) -> list[str]:
    if dataset == "flame3":
        if not RGB_DIR.exists():
            raise FileNotFoundError(f"RGB directory not found: {RGB_DIR}")
        ids = sorted(p.stem for p in RGB_DIR.glob("*.JPG"))
        if exclude_occluded:
            occ = occluded_frame_ids(occlusion_tau)
            ids = [i for i in ids if i not in occ]
        return ids
    if dataset == "flame1":
        # Only frames where BOTH image and mask exist (2 orphan masks in the zip).
        if not FLAME1_RGB_DIR.exists() or not FLAME1_MASK_DIR.exists():
            raise FileNotFoundError(f"FLAME-1 dirs not found under {FLAME1_DIR}")
        imgs = {p.stem for p in FLAME1_RGB_DIR.glob("image_*.jpg")}
        masks = {p.stem for p in FLAME1_MASK_DIR.glob("image_*.png")}
        paired = imgs & masks
        # Sort numerically by the index after 'image_', not lexically.
        return sorted(paired, key=lambda s: int(s.split("_")[1]))
    if dataset == "synth":
        if not SYNTH_RGB_DIR.exists():
            raise FileNotFoundError(f"SYNTH dir not found: {SYNTH_DIR} (run python -m flame.synth)")
        imgs = {p.stem for p in SYNTH_RGB_DIR.glob("image_*.jpg")}
        masks = {p.stem for p in SYNTH_MASK_DIR.glob("image_*.png")}
        return sorted(imgs & masks, key=lambda s: int(s.split("_")[1]))
    if dataset == "flame2":
        if not FLAME2_RGB_DIR.exists() or not FLAME2_MASK_DIR.exists():
            raise FileNotFoundError(
                f"FLAME-2 frames not found under {FLAME2_DIR} (run python scripts/extract_flame2.py)")
        imgs = {p.stem for p in FLAME2_RGB_DIR.glob("*.jpg")}
        masks = {p.stem for p in FLAME2_MASK_DIR.glob("*.png")}
        # ids look like v6_000090 -> sort by (video, frame index).
        return sorted(imgs & masks,
                      key=lambda s: (int(s.split("_")[0][1:]), int(s.split("_")[1])))
    if dataset == "combined":
        # Pool FLAME 1+2+3, tagging each id with its source so load_frame can
        # dispatch. Per-dataset order is preserved for a deterministic split.
        out: list[str] = []
        for ds in COMBINED_DATASETS:
            out += [f"{ds}::{fid}" for fid in list_frame_ids(dataset=ds)]
        return out
    raise ValueError(
        f"unknown dataset {dataset!r} (expected 'flame3', 'flame1', 'flame2', 'synth', or 'combined')")


def _resize_to_max_side(rgb: np.ndarray, mask: np.ndarray,
                        max_side: int) -> tuple[np.ndarray, np.ndarray]:
    """Downscale so max(H,W) == max_side, preserving aspect ratio.

    FLAME-1 is 3840x2160; running Kass/GAC and 33MP IoU eval at native is
    prohibitively slow (~18s/frame for Kass). Resizing once at load time lets
    every downstream method (classical + deep eval) see a smaller frame
    without per-method plumbing changes. Image: bilinear. Mask: nearest.
    """
    h, w = mask.shape
    side = max(h, w)
    if side <= max_side:
        return rgb, mask
    scale = max_side / side
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    rgb_s = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_s = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return rgb_s, mask_s


def load_frame(frame_id: str, threshold_c: float = DEFAULT_THRESHOLD_C,
               min_blob_px: int = DEFAULT_MIN_BLOB_PX,
               dataset: str = DEFAULT_DATASET,
               max_side: int | None = None) -> Frame:
    """Load a frame's RGB + GT mask (+ thermal on FLAME-3).

    max_side: if set, downscale RGB+mask so max(H,W) == max_side. FLAME-1 frames
    are 3840x2160; capping the working resolution here keeps Kass/GAC tractable
    and the deep eval fast without changing the FLAME-3 default (max_side=None).
    """
    if dataset == "flame3":
        bgr = cv2.imread(str(RGB_DIR / f"{frame_id}.JPG"), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(RGB_DIR / f"{frame_id}.JPG")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        thermal_c = tifffile.imread(THERMAL_DIR / f"{frame_id}.TIFF").astype(np.float32)
        gt_mask = threshold_thermal(thermal_c, threshold_c, min_blob_px)
        if max_side is not None:
            rgb, gt_mask = _resize_to_max_side(rgb, gt_mask, max_side)
            # Resize thermal with linear interp so threshold semantics stay sane.
            h, w = gt_mask.shape
            thermal_c = cv2.resize(thermal_c, (w, h), interpolation=cv2.INTER_LINEAR)
        return Frame(frame_id=frame_id, rgb=rgb, thermal_c=thermal_c, gt_mask=gt_mask)
    if dataset == "flame1":
        bgr = cv2.imread(str(FLAME1_RGB_DIR / f"{frame_id}.jpg"), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(FLAME1_RGB_DIR / f"{frame_id}.jpg")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Masks are uint8 {0,1}; rescale to 0/255 to match FLAME-3 convention.
        m = cv2.imread(str(FLAME1_MASK_DIR / f"{frame_id}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(FLAME1_MASK_DIR / f"{frame_id}.png")
        gt_mask = (m > 0).astype(np.uint8) * 255
        if max_side is not None:
            rgb, gt_mask = _resize_to_max_side(rgb, gt_mask, max_side)
        return Frame(frame_id=frame_id, rgb=rgb, thermal_c=None, gt_mask=gt_mask)
    if dataset == "synth":
        bgr = cv2.imread(str(SYNTH_RGB_DIR / f"{frame_id}.jpg"), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(SYNTH_RGB_DIR / f"{frame_id}.jpg")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        m = cv2.imread(str(SYNTH_MASK_DIR / f"{frame_id}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(SYNTH_MASK_DIR / f"{frame_id}.png")
        gt_mask = (m > 0).astype(np.uint8) * 255
        if max_side is not None:
            rgb, gt_mask = _resize_to_max_side(rgb, gt_mask, max_side)
        return Frame(frame_id=frame_id, rgb=rgb, thermal_c=None, gt_mask=gt_mask)
    if dataset == "flame2":
        # Pre-derived at extraction time: RGB jpg + binary PNG mask (resize-and-pair,
        # mask already in RGB-frame coordinates). No raw thermal channel.
        bgr = cv2.imread(str(FLAME2_RGB_DIR / f"{frame_id}.jpg"), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(FLAME2_RGB_DIR / f"{frame_id}.jpg")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        m = cv2.imread(str(FLAME2_MASK_DIR / f"{frame_id}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(FLAME2_MASK_DIR / f"{frame_id}.png")
        gt_mask = (m > 0).astype(np.uint8) * 255
        if max_side is not None:
            rgb, gt_mask = _resize_to_max_side(rgb, gt_mask, max_side)
        return Frame(frame_id=frame_id, rgb=rgb, thermal_c=None, gt_mask=gt_mask)
    if dataset == "combined":
        # frame_id is 'flame1::image_5' / 'flame2::v6_0090' / 'flame3::00012'.
        src, _, raw = frame_id.partition("::")
        return load_frame(raw, threshold_c=threshold_c, min_blob_px=min_blob_px,
                          dataset=src, max_side=max_side)
    raise ValueError(
        f"unknown dataset {dataset!r} (expected 'flame3', 'flame1', 'flame2', 'synth', or 'combined')")
