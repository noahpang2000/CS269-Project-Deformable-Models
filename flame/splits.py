"""Contiguous 70/15/15 train/val/test split over FLAME-3 frames."""
from __future__ import annotations

from flame.data import list_frame_ids


def make_splits(train: float = 0.70, val: float = 0.15) -> dict[str, list[str]]:
    ids = list_frame_ids()
    n = len(ids)
    a = int(n * train)
    b = a + int(n * val)
    return {"train": ids[:a], "val": ids[a:b], "test": ids[b:]}
