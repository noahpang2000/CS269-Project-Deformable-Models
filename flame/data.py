"""Frame loading and thermal-threshold ground-truth generation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tifffile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIRE_DIR = PROJECT_ROOT / "data" / "FLAME3" / "Fire"
RGB_DIR = FIRE_DIR / "RGB" / "Corrected FOV"
THERMAL_DIR = FIRE_DIR / "Thermal" / "Celsius TIFF"

DEFAULT_THRESHOLD_C = 150.0
DEFAULT_MIN_BLOB_PX = 20


@dataclass
class Frame:
    frame_id: str
    rgb: np.ndarray
    thermal_c: np.ndarray
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
                   occlusion_tau: float = DEFAULT_OCCLUSION_TAU) -> list[str]:
    if not RGB_DIR.exists():
        raise FileNotFoundError(f"RGB directory not found: {RGB_DIR}")
    ids = sorted(p.stem for p in RGB_DIR.glob("*.JPG"))
    if exclude_occluded:
        occ = occluded_frame_ids(occlusion_tau)
        ids = [i for i in ids if i not in occ]
    return ids


def load_frame(frame_id: str, threshold_c: float = DEFAULT_THRESHOLD_C,
               min_blob_px: int = DEFAULT_MIN_BLOB_PX) -> Frame:
    bgr = cv2.imread(str(RGB_DIR / f"{frame_id}.JPG"), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(RGB_DIR / f"{frame_id}.JPG")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    thermal_c = tifffile.imread(THERMAL_DIR / f"{frame_id}.TIFF").astype(np.float32)
    gt_mask = threshold_thermal(thermal_c, threshold_c, min_blob_px)
    return Frame(frame_id=frame_id, rgb=rgb, thermal_c=thermal_c, gt_mask=gt_mask)
