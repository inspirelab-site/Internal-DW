"""Subject-disjoint KTH Actions clips for autoregressive video prediction."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from .base import SequenceDataset

try:  # optional, required only for dataset=kth_actions
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


KTH_SUBJECT_SPLITS = {
    "train": {11, 12, 13, 14, 15, 16, 17, 18},
    "val": {19, 20, 21, 23, 24, 25, 1, 4},
    "test": {22, 2, 3, 5, 6, 7, 8, 9, 10},
}

_LINE = re.compile(r"^(person(?P<subject>\d+)_[a-z]+_d\d)\s+frames\s+(?P<ranges>.+)$")
_RANGE = re.compile(r"(\d+)\s*-\s*(\d+)")


def _video_map(root: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for path in root.rglob("*.avi"):
        key = path.stem
        if key.endswith("_uncomp"):
            key = key[: -len("_uncomp")]
        mapping[key.lower()] = str(path.resolve())
    return mapping


def _cap_evenly(records: List[tuple], maximum: int) -> List[tuple]:
    if maximum <= 0 or len(records) <= maximum:
        return records
    indices = np.linspace(0, len(records) - 1, num=maximum).round().astype(np.int64)
    return [records[int(index)] for index in indices]


class KTHActionsDataset(SequenceDataset):
    dataset_name = "kth_actions"
    has_external_input = False
    task_type = "field2d"
    evaluator_name = "field2d"

    def __init__(
        self,
        root: str | Path,
        split: str,
        sequence_length: int = 32,
        sequence_stride: int = 16,
        height: int = 64,
        width: int = 80,
        temporal_subsample: int = 1,
        max_samples: int = 0,
    ):
        if cv2 is None:
            raise ImportError("dataset=kth_actions requires opencv-python")
        self.root = Path(root).expanduser().resolve()
        self.split = "val" if str(split) in ("valid", "validation") else str(split)
        if self.split not in KTH_SUBJECT_SPLITS:
            raise ValueError(f"unknown KTH split {split!r}")
        self.sequence_length = int(sequence_length)
        self.sequence_stride = max(1, int(sequence_stride))
        self.height = int(height)
        self.width = int(width)
        self.temporal_subsample = max(1, int(temporal_subsample))
        self.max_samples = int(max_samples)
        if self.sequence_length <= 1 or self.height <= 0 or self.width <= 0:
            raise ValueError("invalid KTH sequence/grid configuration")

        sequence_file = self.root / "00sequences.txt"
        if not sequence_file.is_file():
            raise FileNotFoundError(f"missing {sequence_file}; run scripts/download_kth_actions.sh")
        videos = _video_map(self.root / "videos")
        if not videos:
            raise FileNotFoundError(f"no AVI files below {self.root / 'videos'}")

        raw_span = (self.sequence_length - 1) * self.temporal_subsample + 1
        records: List[Tuple[str, int, int, str, int]] = []
        for text in sequence_file.read_text(encoding="utf-8", errors="replace").splitlines():
            match = _LINE.match(text.strip())
            if match is None:
                continue
            subject = int(match.group("subject"))
            if subject not in KTH_SUBJECT_SPLITS[self.split]:
                continue
            key = match.group(1).lower()
            path = videos.get(key)
            if path is None:
                continue
            action = key.split("_")[1]
            for first, last in _RANGE.findall(match.group("ranges")):
                # Annotation indices are one-based and inclusive.
                clip_first = int(first) - 1
                clip_last = int(last) - 1
                final_start = clip_last - raw_span + 1
                for start in range(clip_first, final_start + 1, self.sequence_stride):
                    records.append((path, int(start), raw_span, action, subject))
        records.sort(key=lambda item: (item[4], item[0], item[1]))
        self.records = _cap_evenly(records, self.max_samples)
        if not self.records:
            raise RuntimeError(f"no feasible KTH clips for split={self.split}")
        self.field_shape = (1, self.height, self.width)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        path, start, raw_span, action, subject = self.records[int(index)]
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise OSError(f"could not open KTH video {path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        frames = []
        wanted_raw = set(range(0, raw_span, self.temporal_subsample))
        for offset in range(raw_span):
            ok, frame = capture.read()
            if not ok:
                capture.release()
                raise OSError(f"short read in {path} at frame {start + offset}")
            if offset not in wanted_raw:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
            frames.append(gray)
        capture.release()
        state = np.stack(frames, axis=0).astype(np.float32)[:, None] / 255.0
        if state.shape[0] != self.sequence_length:
            raise RuntimeError(f"decoded {state.shape[0]} KTH frames, expected {self.sequence_length}")
        return {
            "state": torch.from_numpy(np.ascontiguousarray(state)),
            "external_input": None,
            "label": int(index),
            "metadata": {
                "dataset": self.dataset_name,
                "split": self.split,
                "path": path,
                "start": int(start),
                "action": action,
                "subject": int(subject),
                "sequence_length": self.sequence_length,
                "temporal_subsample": self.temporal_subsample,
                "state_format": "TCHW",
            },
        }


def build_kth_actions_splits(args):
    common = dict(
        root=args.data_path,
        sequence_length=int(getattr(args, "kth_sequence_length", 32)),
        sequence_stride=int(getattr(args, "kth_sequence_stride", 16)),
        height=int(getattr(args, "kth_height", 64)),
        width=int(getattr(args, "kth_width", 80)),
        temporal_subsample=int(getattr(args, "kth_time_subsample", 1)),
    )
    train = KTHActionsDataset(
        split="train", max_samples=int(getattr(args, "kth_max_train_samples", 0)), **common
    )
    val = KTHActionsDataset(
        split="val", max_samples=int(getattr(args, "kth_max_val_samples", 0)), **common
    )
    test = KTHActionsDataset(
        split="test", max_samples=int(getattr(args, "kth_max_test_samples", 0)), **common
    )
    args.field_channels = 1
    args.field_height = common["height"]
    args.field_width = common["width"]
    args.roi_dim = args.field_height * args.field_width
    args.stim_dim = 1
    print(
        f"[KTH] subject-disjoint clips; grid={args.field_height}x{args.field_width}; "
        f"sequence={common['sequence_length']}; train/val/test={len(train)}/{len(val)}/{len(test)}",
        flush=True,
    )
    return train, val, test

