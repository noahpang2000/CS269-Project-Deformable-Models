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


def list_frame_ids() -> list[str]:
    if not RGB_DIR.exists():
        raise FileNotFoundError(f"RGB directory not found: {RGB_DIR}")
    return sorted(p.stem for p in RGB_DIR.glob("*.JPG"))


def load_frame(frame_id: str, threshold_c: float = DEFAULT_THRESHOLD_C,
               min_blob_px: int = DEFAULT_MIN_BLOB_PX) -> Frame:
    bgr = cv2.imread(str(RGB_DIR / f"{frame_id}.JPG"), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(RGB_DIR / f"{frame_id}.JPG")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    thermal_c = tifffile.imread(THERMAL_DIR / f"{frame_id}.TIFF").astype(np.float32)
    gt_mask = threshold_thermal(thermal_c, threshold_c, min_blob_px)
    return Frame(frame_id=frame_id, rgb=rgb, thermal_c=thermal_c, gt_mask=gt_mask)
