"""Streaming SEVIR VIL sequences for autoregressive field prediction.

The loader reads the official VIL HDF5 files in place.  It deliberately does
not materialize a second copy of the roughly 137 GiB archive.  Samples follow
the repository-wide field convention ``[T, C, H, W]`` and are scaled from the
native uint8 VIL range to ``[0, 1]``.  ``native_scale`` in the metadata records
the inverse transform needed by threshold-based SEVIR scores.

The official SEVIR test boundary is 2019-06-01.  We retain it and carve a
chronological validation split from the preceding data:

  train: time < 2019-01-01
  val:   2019-01-01 <= time < 2019-06-01
  test:  time >= 2019-06-01

This prevents storm frames (or duplicate catalogue identifiers) from crossing
the split through random sampling.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .base import SequenceDataset

try:  # optional dependency, required only for dataset=sevir
    import h5py
except Exception:  # pragma: no cover
    h5py = None


SEVIR_NATIVE_SCALE = 255.0
SEVIR_FRAME_MINUTES = 5
SEVIR_VIL_THRESHOLDS = (16, 74, 133, 160, 181, 219)


def _resolve_catalog(root: Path) -> Path:
    for candidate in (root / "CATALOG.csv", root / "catalog.csv"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"SEVIR catalogue not found below {root}")


def _resolve_h5(root: Path, relative: str) -> Path:
    relative_path = Path(str(relative).replace("\\", "/"))
    for candidate in (root / relative_path, root / "data" / relative_path):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"catalogue entry {relative!r} was not found below {root} or {root / 'data'}"
    )


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(str(value).strip())


def _split_contains(split: str, time: datetime, val_start: datetime, test_start: datetime) -> bool:
    if split == "train":
        return time < val_start
    if split in ("val", "valid", "validation"):
        return val_start <= time < test_start
    if split == "test":
        return time >= test_start
    raise ValueError(f"unknown SEVIR split {split!r}")


def _evenly_cap(records: List[dict], maximum: int) -> List[dict]:
    """Deterministically retain events across the entire chronological split."""
    if maximum <= 0 or len(records) <= maximum:
        return records
    indices = np.linspace(0, len(records) - 1, num=int(maximum)).round().astype(np.int64)
    return [records[int(index)] for index in indices]


class SEVIRVILDataset(SequenceDataset):
    """Official SEVIR VIL events streamed from HDF5.

    Each catalogue row is one 49-frame, five-minute VIL event.  HDF5 handles
    are opened lazily inside each DataLoader worker and cached per process.
    """

    dataset_name = "sevir"
    has_external_input = False
    task_type = "field2d"
    evaluator_name = "field2d"

    def __init__(
        self,
        root: str | Path,
        split: str,
        sequence_length: int = 49,
        spatial_subsample: int = 3,
        max_events: int = 0,
        val_start: str = "2019-01-01",
        test_start: str = "2019-06-01",
        native_scale: float = SEVIR_NATIVE_SCALE,
    ):
        if h5py is None:
            raise ImportError("dataset=sevir requires h5py (`pip install h5py`)")
        self.root = Path(root).expanduser().resolve()
        self.split = "val" if str(split) in ("valid", "validation") else str(split)
        self.sequence_length = int(sequence_length)
        self.spatial_subsample = max(1, int(spatial_subsample))
        self.max_events = int(max_events)
        self.val_start = _parse_time(val_start)
        self.test_start = _parse_time(test_start)
        self.native_scale = float(native_scale)
        if not (1 < self.sequence_length <= 49):
            raise ValueError("SEVIR sequence_length must be in [2, 49]")
        if self.native_scale <= 0:
            raise ValueError("SEVIR native_scale must be positive")

        catalog = _resolve_catalog(self.root)
        records: List[dict] = []
        resolved_paths: Dict[str, str] = {}
        with catalog.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("img_type", "")).strip().lower() != "vil":
                    continue
                time = _parse_time(row["time_utc"])
                if not _split_contains(self.split, time, self.val_start, self.test_start):
                    continue
                relative = str(row["file_name"])
                if relative not in resolved_paths:
                    resolved_paths[relative] = str(_resolve_h5(self.root, relative))
                records.append(
                    {
                        "id": str(row["id"]),
                        "path": resolved_paths[relative],
                        "file_index": int(row["file_index"]),
                        "time_utc": time.isoformat(sep=" "),
                        "event_type": str(row.get("event_type", "")),
                    }
                )
        records.sort(key=lambda item: (item["time_utc"], item["path"], item["file_index"]))
        self.records = _evenly_cap(records, self.max_events)
        if not self.records:
            raise RuntimeError(f"no VIL events found for SEVIR split={self.split} below {self.root}")

        # HDF5 objects cannot be pickled safely.  Every DataLoader worker gets
        # its own lazily populated handle map via __getstate__ below.
        self._handles: Dict[str, object] = {}
        output_side = (384 + self.spatial_subsample - 1) // self.spatial_subsample
        self.field_shape: Tuple[int, int, int] = (1, output_side, output_side)

    def __len__(self) -> int:
        return len(self.records)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handles"] = {}
        return state

    def _handle(self, path: str):
        handle = self._handles.get(path)
        if handle is None:
            handle = h5py.File(path, "r")
            self._handles[path] = handle
        return handle

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()

    def __del__(self):  # pragma: no cover - interpreter shutdown is best effort
        self.close()

    def __getitem__(self, index: int):
        record = self.records[int(index)]
        dataset = self._handle(record["path"])["vil"]
        # Native layout is [H,W,T].  A centre-pixel spatial stride preserves
        # the native VIL scale (unlike averaging, which would change the
        # threshold semantics) while making 384x384 pilots inexpensive.
        offset = self.spatial_subsample // 2
        raw = np.asarray(
            dataset[
                int(record["file_index"]),
                offset:: self.spatial_subsample,
                offset:: self.spatial_subsample,
                : self.sequence_length,
            ],
            dtype=np.float32,
        )
        state = np.transpose(raw, (2, 0, 1))[:, None] / self.native_scale
        state_tensor = torch.from_numpy(np.ascontiguousarray(state, dtype=np.float32))
        return {
            "state": state_tensor,
            "external_input": None,
            "label": int(index),
            "metadata": {
                "dataset": self.dataset_name,
                "split": self.split,
                "event_id": record["id"],
                "time_utc": record["time_utc"],
                "event_type": record["event_type"],
                "path": record["path"],
                "file_index": int(record["file_index"]),
                "sequence_length": self.sequence_length,
                "frame_minutes": SEVIR_FRAME_MINUTES,
                "spatial_subsample": self.spatial_subsample,
                "native_scale": self.native_scale,
                "state_format": "TCHW",
            },
        }


def build_sevir_splits(args):
    common = dict(
        root=args.data_path,
        sequence_length=int(getattr(args, "sevir_sequence_length", 49)),
        spatial_subsample=int(getattr(args, "sevir_spatial_subsample", 3)),
        val_start=str(getattr(args, "sevir_val_start", "2019-01-01")),
        test_start=str(getattr(args, "sevir_test_start", "2019-06-01")),
        native_scale=float(getattr(args, "sevir_native_scale", SEVIR_NATIVE_SCALE)),
    )
    train = SEVIRVILDataset(
        split="train", max_events=int(getattr(args, "sevir_max_train_events", 0)), **common
    )
    val = SEVIRVILDataset(
        split="val", max_events=int(getattr(args, "sevir_max_val_events", 0)), **common
    )
    test = SEVIRVILDataset(
        split="test", max_events=int(getattr(args, "sevir_max_test_events", 0)), **common
    )
    channels, height, width = train.field_shape
    args.field_channels = channels
    args.field_height = height
    args.field_width = width
    args.roi_dim = channels * height * width
    args.stim_dim = 1
    print(
        f"[SEVIR] VIL/255; grid={height}x{width}; sequence={train.sequence_length}; "
        f"train/val/test events={len(train)}/{len(val)}/{len(test)}; "
        f"time split: <{common['val_start']} / <{common['test_start']} / test; "
        f"spatial_subsample={train.spatial_subsample}",
        flush=True,
    )
    return train, val, test

