"""Contiguous 70/15/15 train/val/test split over FLAME-3 or FLAME-1 frames."""
from __future__ import annotations

from flame.data import DEFAULT_DATASET, list_frame_ids


def make_splits(train: float = 0.70, val: float = 0.15,
                exclude_occluded: bool = False,
                dataset: str = DEFAULT_DATASET) -> dict[str, list[str]]:
    ids = list_frame_ids(exclude_occluded=exclude_occluded, dataset=dataset)
    n = len(ids)
    a = int(n * train)
    b = a + int(n * val)
    return {"train": ids[:a], "val": ids[a:b], "test": ids[b:]}
