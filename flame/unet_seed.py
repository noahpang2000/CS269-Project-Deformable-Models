"""Produce a U-Net predicted mask to seed the classical contour methods.

This is the 'learned, prior-free init' alternative to oracle (GT) and color
(R-G floor) seeding: the contour starts from a real detector's output instead
of the ground truth. Torch is imported lazily so the pure-classical path
(color/oracle) never needs it.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from flame.data import Frame

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


@lru_cache(maxsize=2)
def _load(tag: str, device_str: str):
    import torch
    from run_deep import build_model
    from flame.deep.dataset import NET_SIZE
    ckpt = MODELS_DIR / f"unet{tag}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No U-Net checkpoint at {ckpt}; train it first.")
    model = build_model("unet").to(device_str)
    model.load_state_dict(torch.load(ckpt, map_location=device_str, weights_only=True))
    model.eval()
    return model, NET_SIZE


def unet_seed_mask(frame: Frame, tag: str = "", device: str = "cpu") -> np.ndarray:
    """0/255 mask from the trained U-Net, at the frame's native resolution."""
    import torch
    from run_deep import predict_native
    model, size = _load(tag, device)
    with torch.no_grad():
        return predict_native("unet", model, frame, size, device)
