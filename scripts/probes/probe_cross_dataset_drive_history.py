#!/usr/bin/env python3
"""Unified nested-prediction drive--history diagnostic.

For every dataset, compare a drive-only predictor D with the same predictor
augmented by a state-history representation D+H:

    history_value(k) = 1 - MSE(D+H) / MSE(D)
    drive_value(k)   = 1 - MSE(D)   / MSE(mean)

Two model classes are reported independently: ridge regression and nonlinear
local analogs.  A training-history permutation preserves the drive/target pair
but destroys their association with history.  History-window selection uses
validation trajectories; test trajectories are scored only after selection.

Datasets without an exposed external input use the training mean as D, so
drive_value is exactly zero and history_value measures predictive history over
the unconditional baseline.  For WeatherBench-2 and NARMA, D observes the full
known future-drive path up to the scored horizon.  High-dimensional fields are
evaluated on a fixed channel-stratified coordinate sample, an unbiased Monte
Carlo approximation to coordinate-averaged standardized MSE.

No forecasting-model weights are used.  A checkpoint is accepted only to
recover the exact dataset construction and split arguments of the main run.

The output also records H-only predictions and the two-player Shapley
allocation from the four coalitions empty, H, D, and H+D.  Repeating the H and
H+D coalitions with permuted history gives the finite-readout null contribution
and the corrected quantity Delta phi_H = phi_H - phi_H_null.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from internal_dw.datasets.registry import build_dataloaders  # noqa: E402


class _ArraySequenceDataset:
    """Minimal in-memory dataset for checkpoint-free mechanism screening."""

    task_type = "sequence_vector"
    evaluator_name = "generic"

    def __init__(
        self,
        states: np.ndarray,
        drives: np.ndarray | None,
        dataset_name: str,
        split: str,
    ) -> None:
        states = np.asarray(states, dtype=np.float32)
        if states.ndim < 3:
            raise ValueError(
                f"{split} states must have shape [sequence,time,...], got {states.shape}"
            )
        if drives is not None:
            drives = np.asarray(drives, dtype=np.float32)
            if drives.ndim != 3 or drives.shape[:2] != states.shape[:2]:
                raise ValueError(
                    f"{split} drives must be [sequence,time,D] and align with states; "
                    f"got states={states.shape}, drives={drives.shape}"
                )
        self.states = np.ascontiguousarray(states)
        self.drives = None if drives is None else np.ascontiguousarray(drives)
        self.dataset_name = str(dataset_name)
        self.split = str(split)
        self.has_external_input = self.drives is not None

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __getitem__(self, index: int) -> dict:
        drive = None if self.drives is None else torch.from_numpy(self.drives[index])
        return {
            "state": torch.from_numpy(self.states[index]),
            "external_input": drive,
            "label": int(index),
            "metadata": {
                "dataset": self.dataset_name,
                "split": self.split,
                "index": int(index),
            },
        }


def _array_datasets(path: Path, dataset_name: str):
    with np.load(path, allow_pickle=False) as archive:
        required = ("train_state", "validation_state", "test_state")
        missing = [key for key in required if key not in archive]
        if missing:
            raise KeyError(f"{path} is missing required arrays {missing}")
        states = {split: np.asarray(archive[f"{split}_state"], dtype=np.float32)
                  for split in ("train", "validation", "test")}
        drive_keys = [f"{split}_drive" in archive for split in ("train", "validation", "test")]
        if any(drive_keys) and not all(drive_keys):
            raise KeyError(f"{path} must contain all three *_drive arrays or none")
        drives = {
            split: (
                np.asarray(archive[f"{split}_drive"], dtype=np.float32)
                if all(drive_keys)
                else None
            )
            for split in ("train", "validation", "test")
        }
        metadata = {}
        if "metadata_json" in archive:
            metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
    datasets = tuple(
        _ArraySequenceDataset(states[split], drives[split], dataset_name, split)
        for split in ("train", "validation", "test")
    )
    return datasets, metadata


def _parse_ints(text: str) -> list[int]:
    values = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    if not values or min(values) <= 0:
        raise ValueError(f"invalid positive integer list: {text!r}")
    return values


def _checkpoint_args(path: Path) -> SimpleNamespace:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    raw = checkpoint.get("args")
    if raw is None:
        raise KeyError(f"checkpoint has no args: {path}")
    values = dict(raw if isinstance(raw, dict) else vars(raw))
    values["num_workers"] = 0
    values["local_batch_size"] = 1
    values["batch_size"] = 1
    return SimpleNamespace(**values)


def _select_coordinates(
    shape: Sequence[int],
    maximum: int,
    seed: int,
    field_channels: Sequence[int] | None = None,
) -> np.ndarray:
    flat = int(np.prod(shape))
    if field_channels is None and (maximum <= 0 or flat <= maximum):
        return np.arange(flat, dtype=np.int64)
    rng = np.random.default_rng(int(seed) + 101)
    # Field tensors conventionally use [C,H,W].  Sample each channel equally.
    if len(shape) >= 3 and int(shape[0]) > 1:
        total_channels = int(shape[0])
        channels = (
            list(range(total_channels))
            if field_channels is None
            else [int(channel) for channel in field_channels]
        )
        if not channels or min(channels) < 0 or max(channels) >= total_channels:
            raise ValueError(
                f"field channels {channels} are invalid for shape {tuple(shape)}"
            )
        per_channel = (
            int(np.prod(shape[1:]))
            if maximum <= 0
            else max(1, int(maximum) // len(channels))
        )
        spatial = int(np.prod(shape[1:]))
        selected = []
        for channel in channels:
            count = min(per_channel, spatial)
            positions = rng.choice(spatial, size=count, replace=False)
            selected.extend((channel * spatial + positions).tolist())
        return np.asarray(sorted(selected), dtype=np.int64)
    if field_channels is not None:
        raise ValueError("--field-channels requires a state shaped [T,C,...]")
    return np.sort(rng.choice(flat, size=int(maximum), replace=False)).astype(np.int64)


def _dataset_indices(length: int, maximum: int, seed: int) -> np.ndarray:
    if maximum <= 0 or length <= maximum:
        return np.arange(length, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(length, size=int(maximum), replace=False)).astype(np.int64)


def _collect(
    dataset,
    indices: np.ndarray,
    coordinates: np.ndarray,
    external: bool,
    label: str,
) -> Tuple[np.ndarray, np.ndarray | None]:
    states = []
    drives = []
    for position, index in enumerate(indices):
        # WB2 exposes mmap-backed arrays.  Slice sampled coordinates directly
        # instead of copying all 29 x 121 x 240 values for every segment.
        if all(
            hasattr(dataset, name)
            for name in ("states", "starts", "segment_length", "time_features")
        ):
            begin = int(dataset.starts[int(index)])
            stop = begin + int(dataset.segment_length)
            state_view = dataset.states[begin:stop]
            flat = state_view.reshape(state_view.shape[0], -1)
            if int(coordinates.max(initial=-1)) >= flat.shape[1]:
                raise ValueError("coordinate sample exceeds flattened state dimension")
            states.append(
                np.ascontiguousarray(flat[:, coordinates], dtype=np.float32)
            )
            state_shape = tuple(int(value) for value in state_view.shape)
            if external:
                drive = np.asarray(dataset.time_features[begin:stop], dtype=np.float32)
                drives.append(np.ascontiguousarray(drive.reshape(drive.shape[0], -1)))
        else:
            item = dataset[int(index)]
            state = item["state"].detach().cpu().numpy().astype(np.float32, copy=False)
            if state.ndim < 2:
                raise ValueError(f"state must be [T,...], got {state.shape}")
            flat = state.reshape(state.shape[0], -1)
            if int(coordinates.max(initial=-1)) >= flat.shape[1]:
                raise ValueError("coordinate sample exceeds flattened state dimension")
            states.append(np.ascontiguousarray(flat[:, coordinates]))
            state_shape = tuple(int(value) for value in state.shape)
            if external:
                drive = item.get("external_input")
                if drive is None:
                    raise ValueError(f"{label} declares external input but item has None")
                drive = drive.detach().cpu().numpy().astype(np.float32, copy=False)
                drives.append(np.ascontiguousarray(drive.reshape(drive.shape[0], -1)))
        if position == 0 or position + 1 == len(indices) or (position + 1) % 16 == 0:
            print(
                f"[collect:{label}] {position + 1}/{len(indices)} "
                f"state={state_shape} sampled_D={len(coordinates)}",
                flush=True,
            )
    time = min(row.shape[0] for row in states)
    state_array = np.stack([row[:time] for row in states]).astype(np.float32)
    drive_array = None
    if external:
        time = min(time, min(row.shape[0] for row in drives))
        state_array = state_array[:, :time]
        drive_array = np.stack([row[:time] for row in drives]).astype(np.float32)
    return state_array, drive_array


def _standardize_state(
    train: np.ndarray, validation: np.ndarray, test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=(0, 1), keepdims=True)
    scale = train.std(axis=(0, 1), keepdims=True)
    scale = np.maximum(scale, 1e-5)
    return (
        ((train - mean) / scale).astype(np.float32),
        ((validation - mean) / scale).astype(np.float32),
        ((test - mean) / scale).astype(np.float32),
        mean.reshape(-1),
        scale.reshape(-1),
    )


def _standardize_drive(
    train: np.ndarray | None,
    validation: np.ndarray | None,
    test: np.ndarray | None,
) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if train is None:
        return None, None, None
    mean = train.mean(axis=(0, 1), keepdims=True)
    scale = np.maximum(train.std(axis=(0, 1), keepdims=True), 1e-5)
    return (
        ((train - mean) / scale).astype(np.float32),
        ((validation - mean) / scale).astype(np.float32),
        ((test - mean) / scale).astype(np.float32),
    )


def _pca(
    train: np.ndarray,
    rank: int,
    maximum_frames: int,
    seed: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    flat = train.reshape(-1, train.shape[-1])
    rng = np.random.default_rng(int(seed) + 211)
    if maximum_frames > 0 and flat.shape[0] > maximum_frames:
        chosen = rng.choice(flat.shape[0], size=int(maximum_frames), replace=False)
        flat = flat[chosen]
    matrix = torch.from_numpy(np.ascontiguousarray(flat)).to(device)
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    q = min(int(rank), int(matrix.shape[0]) - 1, int(matrix.shape[1]))
    if q <= 0:
        raise ValueError("not enough frames/coordinates for PCA")
    _, singular, vectors = torch.pca_lowrank(matrix, q=q, center=False, niter=3)
    projection = vectors[:, :q].cpu().numpy().astype(np.float32)
    eigenvalues = torch.square(singular).cpu().numpy().astype(np.float64)
    del matrix, singular, vectors
    return projection, eigenvalues


def _project(state: np.ndarray, projection: np.ndarray) -> np.ndarray:
    return np.einsum("std,dr->str", state, projection, optimize=True).astype(np.float32)


def _pairs(
    sequences: int,
    time: int,
    maximum_window: int,
    maximum_horizon: int,
    stride: int,
    maximum: int,
    seed: int,
) -> np.ndarray:
    starts = np.arange(
        int(maximum_window),
        int(time) - int(maximum_horizon) + 1,
        max(1, int(stride)),
    )
    if starts.size == 0:
        raise ValueError(
            f"no starts for T={time}, W={maximum_window}, K={maximum_horizon}"
        )
    rows = np.asarray(
        [(sequence, int(start)) for sequence in range(sequences) for start in starts],
        dtype=np.int64,
    )
    if maximum > 0 and len(rows) > maximum:
        rng = np.random.default_rng(int(seed))
        rows = rows[np.sort(rng.choice(len(rows), size=int(maximum), replace=False))]
    return rows


def _history(projected: np.ndarray, pairs: np.ndarray, window: int) -> np.ndarray:
    rows = []
    for sequence, start in pairs:
        rows.append(projected[int(sequence), int(start) - int(window) : int(start)].reshape(-1))
    return np.asarray(rows, dtype=np.float32)


def _future_drive(
    drive: np.ndarray | None, pairs: np.ndarray, horizon: int
) -> np.ndarray:
    if drive is None:
        return np.empty((len(pairs), 0), dtype=np.float32)
    rows = []
    for sequence, start in pairs:
        rows.append(drive[int(sequence), int(start) : int(start) + int(horizon)].reshape(-1))
    return np.asarray(rows, dtype=np.float32)


def _target(state: np.ndarray, pairs: np.ndarray, horizon: int) -> np.ndarray:
    return np.asarray(
        [state[int(sequence), int(start) + int(horizon) - 1] for sequence, start in pairs],
        dtype=np.float32,
    )


def _feature_scale(fit: np.ndarray, query: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if fit.shape[1] == 0:
        return fit, query
    mean = fit.mean(axis=0, keepdims=True)
    scale = np.maximum(fit.std(axis=0, keepdims=True), 1e-5)
    return ((fit - mean) / scale).astype(np.float32), ((query - mean) / scale).astype(np.float32)


def _ridge(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    query_x: np.ndarray,
    ridge: float,
    device: torch.device,
) -> np.ndarray:
    target_mean = fit_y.mean(axis=0, keepdims=True).astype(np.float32)
    if fit_x.shape[1] == 0:
        return np.repeat(target_mean, len(query_x), axis=0)
    x, q = _feature_scale(fit_x, query_x)
    tx = torch.from_numpy(x).to(device)
    tq = torch.from_numpy(q).to(device)
    ty = torch.from_numpy(fit_y).to(device)
    x_mean = tx.mean(dim=0, keepdim=True)
    y_mean = ty.mean(dim=0, keepdim=True)
    xc = tx - x_mean
    gram = xc.T @ xc / float(len(tx))
    scale = torch.trace(gram) / max(1, int(gram.shape[0]))
    gram = gram + float(ridge) * scale.clamp_min(1e-12) * torch.eye(
        gram.shape[0], device=device, dtype=gram.dtype
    )
    cross = xc.T @ (ty - y_mean) / float(len(tx))
    weight = torch.linalg.solve(gram, cross)
    prediction = ((tq - x_mean) @ weight + y_mean).cpu().numpy().astype(np.float32)
    del tx, tq, ty, xc, gram, cross, weight
    return prediction


def _knn(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    query_x: np.ndarray,
    neighbors: int,
    chunk: int,
    device: torch.device,
) -> np.ndarray:
    if fit_x.shape[1] == 0:
        return np.repeat(fit_y.mean(axis=0, keepdims=True), len(query_x), axis=0)
    fit_x, query_x = _feature_scale(fit_x, query_x)
    tx = torch.from_numpy(fit_x).to(device)
    tq = torch.from_numpy(query_x).to(device)
    ty = torch.from_numpy(fit_y).to(device)
    fit_norm = torch.square(tx).sum(dim=1).unsqueeze(0)
    count = min(int(neighbors), int(len(tx)))
    output = []
    for begin in range(0, len(tq), int(chunk)):
        current = tq[begin : begin + int(chunk)]
        distance = (
            torch.square(current).sum(dim=1, keepdim=True)
            + fit_norm
            - 2.0 * current @ tx.T
        )
        indices = torch.topk(distance, k=count, largest=False).indices
        output.append(ty[indices].mean(dim=1).cpu())
    prediction = torch.cat(output, dim=0).numpy().astype(np.float32)
    del tx, tq, ty, fit_norm
    return prediction


def _mse_by_sequence(
    prediction: np.ndarray, target: np.ndarray, pairs: np.ndarray, count: int
) -> np.ndarray:
    point = np.square(prediction - target, dtype=np.float64).mean(axis=1)
    output = np.full(count, np.nan, dtype=np.float64)
    for sequence in range(count):
        selected = pairs[:, 0] == sequence
        if selected.any():
            output[sequence] = point[selected].mean()
    if not np.isfinite(output).all():
        raise ValueError("one or more evaluation sequences received no sampled starts")
    return output


def _pooled_ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    return float(numerator.sum() / max(float(denominator.sum()), 1e-30))


def _ci(
    numerator: np.ndarray,
    denominator: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    values = []
    for _ in range(int(repeats)):
        chosen = rng.integers(0, len(numerator), size=len(numerator))
        values.append(1.0 - _pooled_ratio(numerator[chosen], denominator[chosen]))
    return {
        "low": float(np.quantile(values, 0.025)),
        "high": float(np.quantile(values, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument(
        "--input-npz",
        type=Path,
        help=(
            "Checkpoint-free arrays with train/validation/test_state and, "
            "optionally, matching *_drive keys."
        ),
    )
    parser.add_argument(
        "--input-dataset-name",
        default="candidate",
        help="Dataset label stored when --input-npz is used.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--horizons", required=True)
    parser.add_argument("--history-windows", default="1,2,4,8,16")
    parser.add_argument("--long-horizon-min", type=int, default=8)
    parser.add_argument("--max-coordinates", type=int, default=4096)
    parser.add_argument(
        "--field-channels",
        default="",
        help="Optional comma-separated zero-based state channels to score.",
    )
    parser.add_argument(
        "--screen-on-validation",
        action="store_true",
        help=(
            "Keep the official test split untouched: use the earlier half of "
            "validation for window selection and the later half for scoring."
        ),
    )
    parser.add_argument("--pca-rank", type=int, default=32)
    parser.add_argument("--pca-frames", type=int, default=2048)
    parser.add_argument("--max-train-sequences", type=int, default=64)
    parser.add_argument("--max-val-sequences", type=int, default=32)
    parser.add_argument("--max-test-sequences", type=int, default=32)
    parser.add_argument("--fit-samples", type=int, default=2048)
    parser.add_argument("--val-samples", type=int, default=1024)
    parser.add_argument("--test-samples", type=int, default=1024)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--knn-chunk", type=int, default=32)
    parser.add_argument("--null-repeats", type=int, default=8)
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    horizons = _parse_ints(args.horizons)
    windows = _parse_ints(args.history_windows)
    maximum_horizon, maximum_window = max(horizons), max(windows)
    device = torch.device(args.device)
    input_metadata = {}
    if args.input_npz is not None:
        (train_set, validation_set, test_set), input_metadata = _array_datasets(
            args.input_npz, args.input_dataset_name
        )
        dataset_name = str(args.input_dataset_name)
        dataset_args_source = str(args.input_npz)
    else:
        checkpoint_args = _checkpoint_args(args.checkpoint)
        train_loader, validation_loader, test_loader = build_dataloaders(
            checkpoint_args, rank=0, world_size=1
        )
        train_set, validation_set, test_set = (
            train_loader.dataset,
            validation_loader.dataset,
            test_loader.dataset,
        )
        dataset_name = str(checkpoint_args.dataset)
        dataset_args_source = str(args.checkpoint)
    first = train_set[0]["state"]
    field_channels = (
        [int(part.strip()) for part in args.field_channels.split(",") if part.strip()]
        if args.field_channels.strip()
        else None
    )
    coordinates = _select_coordinates(
        tuple(first.shape[1:]),
        int(args.max_coordinates),
        int(args.seed),
        field_channels,
    )
    external = bool(getattr(train_set, "has_external_input", False))
    train_indices = _dataset_indices(
        len(train_set), int(args.max_train_sequences), int(args.seed) + 1
    )
    if args.screen_on_validation:
        # A chronological split makes variable/dataset screening independent of
        # the official forecasting test set.  The earlier validation segments
        # select W; the later segments provide the screening score.
        all_validation = np.arange(len(validation_set), dtype=np.int64)
        if int(args.max_val_sequences) > 0 and len(all_validation) > int(args.max_val_sequences):
            all_validation = _dataset_indices(
                len(validation_set), int(args.max_val_sequences), int(args.seed) + 2
            )
        midpoint = len(all_validation) // 2
        if midpoint < 1 or len(all_validation) - midpoint < 1:
            raise ValueError("--screen-on-validation needs at least two validation sequences")
        val_indices = all_validation[:midpoint]
        test_indices = all_validation[midpoint:]
        score_set = validation_set
        score_label = "validation-screen"
    else:
        val_indices = _dataset_indices(
            len(validation_set), int(args.max_val_sequences), int(args.seed) + 2
        )
        test_indices = _dataset_indices(
            len(test_set), int(args.max_test_sequences), int(args.seed) + 3
        )
        score_set = test_set
        score_label = "test"
    train, train_drive = _collect(train_set, train_indices, coordinates, external, "train")
    validation, val_drive = _collect(
        validation_set, val_indices, coordinates, external, "validation"
    )
    test, test_drive = _collect(
        score_set, test_indices, coordinates, external, score_label
    )
    time = min(train.shape[1], validation.shape[1], test.shape[1])
    train, validation, test = train[:, :time], validation[:, :time], test[:, :time]
    if external:
        train_drive, val_drive, test_drive = (
            train_drive[:, :time], val_drive[:, :time], test_drive[:, :time]
        )
    train, validation, test, state_mean, state_scale = _standardize_state(
        train, validation, test
    )
    train_drive, val_drive, test_drive = _standardize_drive(
        train_drive, val_drive, test_drive
    )
    projection, spectrum = _pca(
        train,
        int(args.pca_rank),
        int(args.pca_frames),
        int(args.seed),
        device,
    )
    train_z, val_z, test_z = (
        _project(train, projection),
        _project(validation, projection),
        _project(test, projection),
    )
    train_pairs = _pairs(
        len(train), time, maximum_window, maximum_horizon, args.sample_stride,
        args.fit_samples, args.seed + 11,
    )
    val_pairs = _pairs(
        len(validation), time, maximum_window, maximum_horizon, args.sample_stride,
        args.val_samples, args.seed + 13,
    )
    test_pairs = _pairs(
        len(test), time, maximum_window, maximum_horizon, args.sample_stride,
        args.test_samples, args.seed + 17,
    )

    # Select W using only validation trajectories and a linear nested comparison.
    selection = []
    for window in windows:
        train_history = _history(train_z, train_pairs, window)
        val_history = _history(val_z, val_pairs, window)
        ratios = []
        for horizon in horizons:
            dy_train = _future_drive(train_drive, train_pairs, horizon)
            dy_val = _future_drive(val_drive, val_pairs, horizon)
            target_train = _target(train, train_pairs, horizon)
            target_val = _target(validation, val_pairs, horizon)
            drive_prediction = _ridge(
                dy_train, target_train, dy_val, args.ridge, device
            )
            full_prediction = _ridge(
                np.concatenate([dy_train, train_history], axis=1),
                target_train,
                np.concatenate([dy_val, val_history], axis=1),
                args.ridge,
                device,
            )
            drive_mse = np.square(drive_prediction - target_val, dtype=np.float64).mean()
            full_mse = np.square(full_prediction - target_val, dtype=np.float64).mean()
            ratios.append(float(full_mse / max(drive_mse, 1e-30)))
        row = {"window": int(window), "mean_mse_ratio": float(np.mean(ratios)), "ratios": ratios}
        selection.append(row)
        print(f"[select:{args.label}] W={window} mean (D+H)/D={row['mean_mse_ratio']:.4f}", flush=True)
    selected_window = int(min(selection, key=lambda row: row["mean_mse_ratio"])["window"])
    print(f"[selected:{args.label}] W={selected_window}", flush=True)

    # Standard practice after validation selection: refit on train+validation.
    fit = np.concatenate([train, validation], axis=0)
    fit_z = np.concatenate([train_z, val_z], axis=0)
    fit_drive = None if train_drive is None else np.concatenate([train_drive, val_drive], axis=0)
    fit_pairs = _pairs(
        len(fit), time, maximum_window, maximum_horizon, args.sample_stride,
        args.fit_samples, args.seed + 19,
    )
    fit_history = _history(fit_z, fit_pairs, selected_window)
    test_history = _history(test_z, test_pairs, selected_window)
    rng = np.random.default_rng(int(args.seed) + 2309)
    permutation_rows = [rng.permutation(len(fit_history)) for _ in range(int(args.null_repeats))]
    rows: Dict[str, object] = {}
    method_arrays = {name: [] for name in ("linear", "nonlinear", "linear_null", "nonlinear_null")}
    history_only_arrays = {name: [] for name in ("linear", "nonlinear")}
    history_only_null_arrays = {name: [] for name in ("linear", "nonlinear")}
    drive_arrays = {name: [] for name in ("linear", "nonlinear")}
    mean_arrays = []
    for horizon in horizons:
        drive_fit = _future_drive(fit_drive, fit_pairs, horizon)
        drive_test = _future_drive(test_drive, test_pairs, horizon)
        target_fit = _target(fit, fit_pairs, horizon)
        target_test = _target(test, test_pairs, horizon)
        mean_prediction = np.repeat(target_fit.mean(axis=0, keepdims=True), len(target_test), axis=0)
        mean_mse = _mse_by_sequence(mean_prediction, target_test, test_pairs, len(test))

        linear_drive_prediction = _ridge(drive_fit, target_fit, drive_test, args.ridge, device)
        linear_prediction = _ridge(
            np.concatenate([drive_fit, fit_history], axis=1), target_fit,
            np.concatenate([drive_test, test_history], axis=1), args.ridge, device,
        )
        nonlinear_drive_prediction = _knn(
            drive_fit, target_fit, drive_test, args.neighbors, args.knn_chunk, device
        )
        nonlinear_prediction = _knn(
            np.concatenate([drive_fit, fit_history], axis=1), target_fit,
            np.concatenate([drive_test, test_history], axis=1),
            args.neighbors, args.knn_chunk, device,
        )
        linear_history_prediction = _ridge(
            fit_history, target_fit, test_history, args.ridge, device
        )
        nonlinear_history_prediction = _knn(
            fit_history,
            target_fit,
            test_history,
            args.neighbors,
            args.knn_chunk,
            device,
        )
        linear_drive_mse = _mse_by_sequence(linear_drive_prediction, target_test, test_pairs, len(test))
        nonlinear_drive_mse = _mse_by_sequence(nonlinear_drive_prediction, target_test, test_pairs, len(test))
        linear_mse = _mse_by_sequence(linear_prediction, target_test, test_pairs, len(test))
        nonlinear_mse = _mse_by_sequence(nonlinear_prediction, target_test, test_pairs, len(test))
        linear_history_mse = _mse_by_sequence(
            linear_history_prediction, target_test, test_pairs, len(test)
        )
        nonlinear_history_mse = _mse_by_sequence(
            nonlinear_history_prediction, target_test, test_pairs, len(test)
        )
        null_linear, null_nonlinear = [], []
        null_linear_history, null_nonlinear_history = [], []
        for permutation in permutation_rows:
            shuffled = fit_history[permutation]
            null_linear_prediction = _ridge(
                np.concatenate([drive_fit, shuffled], axis=1), target_fit,
                np.concatenate([drive_test, test_history], axis=1), args.ridge, device,
            )
            null_nonlinear_prediction = _knn(
                np.concatenate([drive_fit, shuffled], axis=1), target_fit,
                np.concatenate([drive_test, test_history], axis=1),
                args.neighbors, args.knn_chunk, device,
            )
            null_linear.append(_mse_by_sequence(null_linear_prediction, target_test, test_pairs, len(test)))
            null_nonlinear.append(_mse_by_sequence(null_nonlinear_prediction, target_test, test_pairs, len(test)))
            null_linear_history_prediction = _ridge(
                shuffled, target_fit, test_history, args.ridge, device
            )
            null_nonlinear_history_prediction = _knn(
                shuffled,
                target_fit,
                test_history,
                args.neighbors,
                args.knn_chunk,
                device,
            )
            null_linear_history.append(
                _mse_by_sequence(
                    null_linear_history_prediction,
                    target_test,
                    test_pairs,
                    len(test),
                )
            )
            null_nonlinear_history.append(
                _mse_by_sequence(
                    null_nonlinear_history_prediction,
                    target_test,
                    test_pairs,
                    len(test),
                )
            )
        method_arrays["linear"].append(linear_mse)
        method_arrays["nonlinear"].append(nonlinear_mse)
        method_arrays["linear_null"].append(np.median(null_linear, axis=0))
        method_arrays["nonlinear_null"].append(np.median(null_nonlinear, axis=0))
        history_only_arrays["linear"].append(linear_history_mse)
        history_only_arrays["nonlinear"].append(nonlinear_history_mse)
        history_only_null_arrays["linear"].append(
            np.median(null_linear_history, axis=0)
        )
        history_only_null_arrays["nonlinear"].append(
            np.median(null_nonlinear_history, axis=0)
        )
        drive_arrays["linear"].append(linear_drive_mse)
        drive_arrays["nonlinear"].append(nonlinear_drive_mse)
        mean_arrays.append(mean_mse)
        print(f"[score:{args.label}] H={horizon} complete", flush=True)

    mean_array = np.stack(mean_arrays, axis=1)
    method_arrays = {name: np.stack(value, axis=1) for name, value in method_arrays.items()}
    history_only_arrays = {
        name: np.stack(value, axis=1) for name, value in history_only_arrays.items()
    }
    history_only_null_arrays = {
        name: np.stack(value, axis=1)
        for name, value in history_only_null_arrays.items()
    }
    drive_arrays = {name: np.stack(value, axis=1) for name, value in drive_arrays.items()}
    bootstrap_rng = np.random.default_rng(int(args.seed) + 3203)
    methods = {}
    for name, numerator in method_arrays.items():
        base = "linear" if name.startswith("linear") else "nonlinear"
        denominator = drive_arrays[base]
        history_value = 1.0 - numerator.sum(axis=0) / np.maximum(denominator.sum(axis=0), 1e-30)
        methods[name] = {
            "history_value": history_value.tolist(),
            "history_value_ci95": [
                _ci(numerator[:, column], denominator[:, column], args.bootstraps, bootstrap_rng)
                for column in range(len(horizons))
            ],
            "mse_by_sequence_and_horizon": numerator.tolist(),
        }
    drives = {}
    for name, numerator in drive_arrays.items():
        value = 1.0 - numerator.sum(axis=0) / np.maximum(mean_array.sum(axis=0), 1e-30)
        drives[name] = {
            "drive_value": value.tolist(),
            "drive_value_ci95": [
                _ci(numerator[:, column], mean_array[:, column], args.bootstraps, bootstrap_rng)
                for column in range(len(horizons))
            ],
            "mse_by_sequence_and_horizon": numerator.tolist(),
        }
    long_columns = [index for index, horizon in enumerate(horizons) if horizon >= args.long_horizon_min]
    if not long_columns:
        raise ValueError("no horizon reaches --long-horizon-min")
    summary = {}
    shapley = {}
    for name in ("linear", "nonlinear"):
        history_value = float(
            np.mean(np.asarray(methods[name]["history_value"])[long_columns])
        )
        null_value = float(
            np.mean(
                np.asarray(methods[f"{name}_null"]["history_value"])[long_columns]
            )
        )
        summary[name] = {
            "mean_long_history_value": history_value,
            "mean_long_drive_value": float(np.mean(np.asarray(drives[name]["drive_value"])[long_columns])),
            "mean_long_null_history_value": null_value,
            "mean_long_history_beyond_null": history_value - null_value,
        }
        # Common coalition value v(X)=1-R_X/R_empty.  The two-player Shapley
        # allocation splits the H--D interaction symmetrically and is directly
        # comparable with a crossed simulator decomposition.
        empty_risk = mean_array.sum(axis=0)
        history_risk = history_only_arrays[name].sum(axis=0)
        drive_risk = drive_arrays[name].sum(axis=0)
        joint_risk = method_arrays[name].sum(axis=0)
        denominator = np.maximum(empty_risk, 1e-30)
        v_history = 1.0 - history_risk / denominator
        v_drive = 1.0 - drive_risk / denominator
        v_joint = 1.0 - joint_risk / denominator
        phi_history = 0.5 * (v_history + v_joint - v_drive)
        phi_drive = 0.5 * (v_drive + v_joint - v_history)
        null_history_risk = history_only_null_arrays[name].sum(axis=0)
        null_joint_risk = method_arrays[f"{name}_null"].sum(axis=0)
        v_null_history = 1.0 - null_history_risk / denominator
        v_null_joint = 1.0 - null_joint_risk / denominator
        phi_null_history = 0.5 * (v_null_history + v_null_joint - v_drive)
        delta_phi_history = phi_history - phi_null_history
        shapley[name] = {
            "coalition_value_empty": [0.0 for _ in horizons],
            "coalition_value_history": v_history.tolist(),
            "coalition_value_drive": v_drive.tolist(),
            "coalition_value_history_and_drive": v_joint.tolist(),
            "history_shapley": phi_history.tolist(),
            "drive_shapley": phi_drive.tolist(),
            "null_history_shapley": phi_null_history.tolist(),
            "history_shapley_beyond_null": delta_phi_history.tolist(),
            "history_only_mse_by_sequence_and_horizon": history_only_arrays[
                name
            ].tolist(),
            "null_history_only_mse_by_sequence_and_horizon": history_only_null_arrays[
                name
            ].tolist(),
            "mean_long_history_shapley": float(np.mean(phi_history[long_columns])),
            "mean_long_drive_shapley": float(np.mean(phi_drive[long_columns])),
            "mean_long_null_history_shapley": float(
                np.mean(phi_null_history[long_columns])
            ),
            "mean_long_history_shapley_beyond_null": float(
                np.mean(delta_phi_history[long_columns])
            ),
            "mean_long_explained_value": float(np.mean(v_joint[long_columns])),
        }
    result = {
        "format_version": 2,
        "experiment": "unified_nested_drive_history_map",
        "label": args.label,
        "dataset": dataset_name,
        "dataset_args_source": dataset_args_source,
        "checkpoint_args_source": None if args.checkpoint is None else str(args.checkpoint),
        "input_metadata": input_metadata,
        "uses_forecasting_model_weights": False,
        "has_external_drive": external,
        "drive_semantics": "known future external-input path" if external else "training mean (no exposed drive)",
        "horizons": horizons,
        "long_horizon_min": int(args.long_horizon_min),
        "history_windows": windows,
        "selected_history_window": selected_window,
        "window_selection": selection,
        "state_original_shape": list(first.shape[1:]),
        "field_channels": field_channels,
        "sampled_coordinate_count": int(len(coordinates)),
        "sampled_coordinate_indices": coordinates.tolist(),
        "coordinate_standardization": "training coordinate mean/std",
        "pca_rank": int(projection.shape[1]),
        "pca_spectrum": spectrum.tolist(),
        "split_sequence_counts": {
            "train": int(len(train)), "validation": int(len(validation)), "test": int(len(test))
        },
        "screening_protocol": (
            "train fit; early official-validation window selection; late "
            "official-validation scoring; official test untouched"
            if args.screen_on_validation
            else "train fit; official-validation selection; official-test scoring"
        ),
        "official_test_touched": not bool(args.screen_on_validation),
        "sample_counts": {
            "fit": int(len(fit_pairs)), "validation": int(len(val_pairs)), "test": int(len(test_pairs))
        },
        "methods": methods,
        "drives": drives,
        "shapley": shapley,
        "mean_baseline_mse_by_sequence_and_horizon": mean_array.tolist(),
        "long_horizon_summary": summary,
        "state_standardization_mean": state_mean.tolist(),
        "state_standardization_scale": state_scale.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\n{args.label} unified drive--history map (H>={args.long_horizon_min})")
    for name in ("linear", "nonlinear"):
        row = summary[name]
        print(
            f"{name:>9}: drive={row['mean_long_drive_value']:+.4f} "
            f"history={row['mean_long_history_value']:+.4f} "
            f"null={row['mean_long_null_history_value']:+.4f} "
            f"beyond={row['mean_long_history_beyond_null']:+.4f}"
        )
    print(f"[out] {args.output}")


if __name__ == "__main__":
    main()
