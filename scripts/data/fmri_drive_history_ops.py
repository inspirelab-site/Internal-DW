"""Small array helpers for the subject-disjoint fMRI regime diagnostic."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def collect_subjects(dataset, label: str) -> Tuple[np.ndarray, list[str]]:
    states: list[np.ndarray] = []
    subjects: list[str] = []
    for index in range(len(dataset)):
        item = dataset[index]
        state = item["state"].numpy().astype(np.float32, copy=False)
        if state.ndim != 2:
            raise ValueError(f"expected [T,D], got {state.shape}")
        states.append(state)
        subjects.append(str(item["metadata"]["subject_id"]))
        if index == 0 or index + 1 == len(dataset) or (index + 1) % 20 == 0:
            print(
                f"[collect:{label}] {index + 1}/{len(dataset)} "
                f"subject={subjects[-1]} shape={state.shape}",
                flush=True,
            )
    if not states:
        raise ValueError(f"empty {label} split")
    dimensions = {row.shape[1] for row in states}
    if len(dimensions) != 1:
        raise ValueError(f"inconsistent ROI dimensions in {label}: {dimensions}")
    time = min(row.shape[0] for row in states)
    result = np.stack([row[:time] for row in states]).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"non-finite values in {label} split")
    result -= result.mean(axis=1, keepdims=True)
    return result, subjects


def common_time(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    time = min(array.shape[1] for array in arrays)
    dimensions = {array.shape[2] for array in arrays}
    if len(dimensions) != 1:
        raise ValueError(f"split ROI dimensions differ: {dimensions}")
    return tuple(np.ascontiguousarray(array[:, :time]) for array in arrays)


def valid_starts(
    time: int,
    window: int,
    maximum_horizon: int,
    stride: int,
) -> np.ndarray:
    values = np.arange(
        int(window), int(time) - int(maximum_horizon) + 1, max(1, int(stride))
    )
    if values.size == 0:
        raise ValueError(
            f"no valid starts for T={time}, W={window}, K={maximum_horizon}"
        )
    return values


def future_targets(
    residual: np.ndarray,
    starts: np.ndarray,
    horizons: Sequence[int],
    shifts: np.ndarray | None = None,
) -> np.ndarray:
    subject_count = residual.shape[0]
    output = np.empty(
        (subject_count, len(starts), len(horizons), residual.shape[-1]),
        dtype=np.float32,
    )
    for subject in range(subject_count):
        shift = 0 if shifts is None else int(shifts[subject])
        for column, horizon in enumerate(horizons):
            indices = (starts + int(horizon) - 1 + shift) % residual.shape[1]
            output[subject, :, column] = residual[subject, indices]
    return output


def mse_by_subject(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.square(prediction - target, dtype=np.float64).mean(axis=(1, 3))
