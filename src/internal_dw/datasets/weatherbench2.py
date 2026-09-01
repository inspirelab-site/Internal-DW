"""Memory-mapped WeatherBench-2 multi-level field dataset.

The companion ``scripts/data/prepare_weatherbench2_pilot.py`` extracts a selected
subset of the official 1.5-degree ERA5 Zarr archive into normalized ``.npy``
arrays.  Training then uses ordinary local memory maps: no cloud access or Zarr
dependency is required in DataLoader workers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from .base import SequenceDataset


class WeatherBench2Dataset(SequenceDataset):
    dataset_name = "weatherbench2"
    has_external_input = True
    task_type = "field2d"
    evaluator_name = "field2d"

    def __init__(self, root: Path, split: str, segment_length: int, stride: int):
        self.root = Path(root)
        self.split = str(split)
        self.segment_length = int(segment_length)
        self.stride = max(1, int(stride))
        if self.segment_length <= 1:
            raise ValueError("WeatherBench2 segment length must exceed one")

        metadata_path = self.root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"{metadata_path} is missing; run scripts/data/prepare_weatherbench2_pilot.py"
            )
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        split_meta = self.metadata["splits"].get(self.split)
        if split_meta is None:
            raise KeyError(f"split {self.split!r} is absent from {metadata_path}")

        self.states = np.load(self.root / split_meta["state_file"], mmap_mode="r")
        self.time_features = np.load(
            self.root / split_meta["time_features_file"], mmap_mode="r"
        )
        if self.states.ndim != 4:
            raise ValueError(f"expected [T,C,H,W], got {self.states.shape}")
        if self.time_features.shape != (self.states.shape[0], 4):
            raise ValueError(
                f"time-feature/state mismatch: {self.time_features.shape} vs {self.states.shape}"
            )
        self.field_shape = tuple(int(x) for x in self.states.shape[1:])
        last = int(self.states.shape[0]) - self.segment_length
        if last < 0:
            raise ValueError(
                f"split {self.split} has {self.states.shape[0]} steps, "
                f"shorter than segment_length={self.segment_length}"
            )
        self.starts = np.arange(0, last + 1, self.stride, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.starts.size)

    def __getitem__(self, index):
        start = int(self.starts[index])
        stop = start + self.segment_length
        # Copy only the requested short trajectory.  Returning a view into a
        # read-only mmap makes torch warn about non-writable tensors and keeps
        # file-backed pages pinned for longer than necessary.
        state = torch.from_numpy(np.array(self.states[start:stop], copy=True))
        stim = torch.from_numpy(np.array(self.time_features[start:stop], copy=True))
        return {
            "state": state,
            "external_input": stim,
            "label": int(index),
            "metadata": {
                "dataset": self.dataset_name,
                "split": self.split,
                "index": int(index),
                "start": start,
            },
        }


def build_weatherbench2_splits(args) -> Tuple[WeatherBench2Dataset, WeatherBench2Dataset, WeatherBench2Dataset]:
    root = Path(args.data_path)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    channels = len(metadata["channel_names"])
    height = int(metadata["height"])
    width = int(metadata["width"])
    segment_length = int(getattr(args, "wb2_seg_len", 64))
    train_stride = int(getattr(args, "wb2_train_stride", segment_length))
    eval_stride = int(getattr(args, "wb2_eval_stride", segment_length))

    args.field_channels = channels
    args.field_height = height
    args.field_width = width
    args.roi_dim = channels * height * width
    args.stim_dim = 4

    train = WeatherBench2Dataset(root, "train", segment_length, train_stride)
    val = WeatherBench2Dataset(root, "val", segment_length, eval_stride)
    test = WeatherBench2Dataset(root, "test", segment_length, eval_stride)
    print(
        f"[WB2] C={channels}, grid={height}x{width}, segment={segment_length}, "
        f"train/val/test samples={len(train)}/{len(val)}/{len(test)}; "
        f"train_stride={train_stride}, eval_stride={eval_stride}",
        flush=True,
    )
    return train, val, test

