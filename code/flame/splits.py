"""Contiguous 70/15/15 train/val/test split over a FLAME dataset.

For the pooled "combined" set we split each source dataset 70/15/15 *separately*
and concatenate per split, so every dataset is represented in train/val/test (a
single contiguous split over the pooled list would otherwise dump all of FLAME-1
into train and all of FLAME-3 into test).
"""
from __future__ import annotations

from code.flame.data import DEFAULT_DATASET, COMBINED_DATASETS, list_frame_ids


def _contiguous(ids: list[str], train: float, val: float) -> dict[str, list[str]]:
    n = len(ids)
    a = int(n * train)
    b = a + int(n * val)
    return {"train": ids[:a], "val": ids[a:b], "test": ids[b:]}


def make_splits(train: float = 0.70, val: float = 0.15,
                exclude_occluded: bool = False,
                dataset: str = DEFAULT_DATASET) -> dict[str, list[str]]:
    if dataset == "combined":
        out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
        for ds in COMBINED_DATASETS:
            ids = list_frame_ids(exclude_occluded=exclude_occluded, dataset=ds)
            sp = _contiguous(ids, train, val)
            for k in out:
                out[k] += [f"{ds}::{fid}" for fid in sp[k]]
        return out
    ids = list_frame_ids(exclude_occluded=exclude_occluded, dataset=dataset)
    return _contiguous(ids, train, val)
