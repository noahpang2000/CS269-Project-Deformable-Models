"""Engineered input channels for the deep detector.

The bottleneck on this data is detection from a weak RGB cue, not target
representation. Instead of making the network rediscover a fire cue from 435
frames, we hand it the same hand-crafted features the classical methods use
(R-G opponency, HSV) stacked onto the RGB. The network then learns how to
weight a weak cue rather than discarding it.

build_input(rgb, spec) returns a float32 [H,W,C] array in [0,1], used identically
at train time (in the dataset) and inference (in predict_native).
"""
from __future__ import annotations

import cv2
import numpy as np

# channel count per spec, so callers can set UNet in_ch
SPEC_CHANNELS = {"rgb": 3, "rgb_rg": 4, "rgb_hsv_rg": 7}


def build_input(rgb: np.ndarray, spec: str = "rgb") -> np.ndarray:
    """rgb: uint8 [H,W,3]. Returns float32 [H,W,C] in [0,1] per `spec`."""
    rgb_f = rgb.astype(np.float32) / 255.0
    if spec == "rgb":
        return rgb_f
    chans = [rgb_f]
    if spec in ("rgb_rg", "rgb_hsv_rg"):
        rg = np.clip(rgb[..., 0].astype(np.float32) - rgb[..., 1].astype(np.float32),
                     0, 255) / 255.0
        if spec == "rgb_hsv_rg":
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
            chans.append(hsv)
        chans.append(rg[..., None])
    return np.concatenate(chans, axis=-1).astype(np.float32)
