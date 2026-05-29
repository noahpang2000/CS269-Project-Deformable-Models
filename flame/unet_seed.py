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


@lru_cache(maxsize=4)
def _load(tag: str, device_str: str, dataset: str):
    import torch
    from run_deep import build_model, _prefix
    from flame.deep.dataset import NET_SIZE
    ckpt = MODELS_DIR / f"{_prefix(dataset)}unet{tag}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No U-Net checkpoint at {ckpt}; train it first.")
    model = build_model("unet").to(device_str)
    model.load_state_dict(torch.load(ckpt, map_location=device_str, weights_only=True))
    model.eval()
    return model, NET_SIZE


def unet_seed_mask(frame: Frame, tag: str = "", device: str = "cpu",
                   dataset: str = "flame3") -> np.ndarray:
    """0/255 mask from the trained U-Net, at the frame's native resolution.

    dataset selects the checkpoint: 'flame3' -> unet{tag}.pt, 'flame1' ->
    flame1_unet{tag}.pt (matching run_deep._prefix), so the seed comes from the
    detector trained on the same data the frame is from.
    """
    import torch
    from run_deep import predict_native
    model, size = _load(tag, device, dataset)
    with torch.no_grad():
        return predict_native("unet", model, frame, size, device)
