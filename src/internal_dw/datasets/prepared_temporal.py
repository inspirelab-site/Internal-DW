"""Prepared vector time series used by the temporal-regime follow-up.

The archive is produced by ``prepare_temporal_candidate_screen_data.py`` and
contains fixed train/validation/test chunks.  Reusing it here guarantees that
the regime screen and the end-to-end comparison use identical raw examples and
that dataset selection never inspects the official test split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .base import SequenceDataset


class _PreparedTemporalBase(SequenceDataset):
    task_type = "sequence_vector"
    evaluator_name = "generic"

    def __init__(
        self,
        states: np.ndarray,
        drives: np.ndarray | None,
        split: str,
        source_name: str,
        state_mean: np.ndarray,
        state_scale: np.ndarray,
        drive_mean: np.ndarray | None,
        drive_scale: np.ndarray | None,
    ) -> None:
        self.states = np.ascontiguousarray(states, dtype=np.float32)
        self.drives = None if drives is None else np.ascontiguousarray(drives, dtype=np.float32)
        self.split = str(split)
        self.source_name = str(source_name)
        self.state_mean = np.asarray(state_mean, dtype=np.float32)
        self.state_scale = np.asarray(state_scale, dtype=np.float32)
        self.drive_mean = None if drive_mean is None else np.asarray(drive_mean, dtype=np.float32)
        self.drive_scale = None if drive_scale is None else np.asarray(drive_scale, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __getitem__(self, index: int) -> dict:
        drive = None if self.drives is None else torch.from_numpy(self.drives[index])
        return {
            "state": torch.from_numpy(self.states[index]),
            "external_input": drive,
            "label": int(index),
            "metadata": {
                "dataset": self.source_name,
                "split": self.split,
                "index": int(index),
            },
        }


class PreparedTemporalAutonomousDataset(_PreparedTemporalBase):
    dataset_name = "prepared_temporal_autonomous"
    has_external_input = False


class PreparedTemporalDrivenDataset(_PreparedTemporalBase):
    dataset_name = "prepared_temporal_driven"
    has_external_input = True


def _safe_scale(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = array.mean(axis=(0, 1), keepdims=True, dtype=np.float64).astype(np.float32)
    scale = array.std(axis=(0, 1), keepdims=True, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, np.float32(1e-5))
    return mean, scale


def build_prepared_temporal_splits(args, expect_external: bool):
    path = Path(str(getattr(args, "prepared_temporal_npz", "")))
    if not path.is_file():
        raise FileNotFoundError(
            f"--prepared_temporal_npz must point to a prepared archive, got {path}"
        )
    with np.load(path, allow_pickle=False) as archive:
        states = {
            split: np.asarray(archive[f"{split}_state"], dtype=np.float32)
            for split in ("train", "validation", "test")
        }
        drive_keys = [f"{split}_drive" in archive for split in ("train", "validation", "test")]
        if any(drive_keys) != bool(expect_external) or (any(drive_keys) and not all(drive_keys)):
            raise ValueError(
                f"dataset external-input contract mismatch for {path}: "
                f"expected={expect_external}, keys={drive_keys}"
            )
        drives = {
            split: (
                np.asarray(archive[f"{split}_drive"], dtype=np.float32)
                if expect_external
                else None
            )
            for split in ("train", "validation", "test")
        }
        metadata = {}
        if "metadata_json" in archive:
            metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))

    reference_shape = states["train"].shape[2:]
    if len(reference_shape) != 1:
        raise ValueError(f"prepared temporal state must be [S,T,D], got {states['train'].shape}")
    for split in ("train", "validation", "test"):
        if states[split].ndim != 3 or states[split].shape[2:] != reference_shape:
            raise ValueError(f"inconsistent {split} state shape {states[split].shape}")
        if expect_external and (
            drives[split].ndim != 3 or drives[split].shape[:2] != states[split].shape[:2]
        ):
            raise ValueError(
                f"{split} drive {drives[split].shape} does not align with state {states[split].shape}"
            )

    standardize = bool(int(getattr(args, "prepared_temporal_standardize", 1)))
    if standardize:
        state_mean, state_scale = _safe_scale(states["train"])
        states = {
            split: ((array - state_mean) / state_scale).astype(np.float32)
            for split, array in states.items()
        }
        if expect_external:
            drive_mean, drive_scale = _safe_scale(drives["train"])
            drives = {
                split: ((array - drive_mean) / drive_scale).astype(np.float32)
                for split, array in drives.items()
            }
        else:
            drive_mean = drive_scale = None
    else:
        state_mean = np.zeros((1, 1, reference_shape[0]), dtype=np.float32)
        state_scale = np.ones_like(state_mean)
        if expect_external:
            drive_dim = int(drives["train"].shape[-1])
            drive_mean = np.zeros((1, 1, drive_dim), dtype=np.float32)
            drive_scale = np.ones_like(drive_mean)
        else:
            drive_mean = drive_scale = None

    args.roi_dim = int(reference_shape[0])
    if expect_external:
        args.stim_dim = int(drives["train"].shape[-1])
    source_name = str(metadata.get("dataset", path.stem))
    dataset_class = PreparedTemporalDrivenDataset if expect_external else PreparedTemporalAutonomousDataset
    result = tuple(
        dataset_class(
            states[split], drives[split], split, source_name,
            state_mean, state_scale, drive_mean, drive_scale,
        )
        for split in ("train", "validation", "test")
    )
    print(
        f"[prepared-temporal] source={source_name} external={int(expect_external)} "
        f"state_dim={args.roi_dim} stim_dim={getattr(args, 'stim_dim', 0) if expect_external else 0} "
        f"train/val/test={len(result[0])}/{len(result[1])}/{len(result[2])} "
        f"chunk={states['train'].shape[1]} standardized={int(standardize)}",
        flush=True,
    )
    return result
