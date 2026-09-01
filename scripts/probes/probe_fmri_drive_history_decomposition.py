#!/usr/bin/env python3
"""Subject-disjoint drive--history decomposition for time-aligned movie fMRI.

The shared movie response is estimated only from fit subjects.  For a held-out
subject, this gives the diagnostic drive-only prediction at every future time.
We then ask whether the held-out subject's residual history adds predictive
information beyond that shared response.

To avoid concluding "no history" merely because one model is weak, two very
different residual-history predictors are evaluated:

* multivariate direct ridge regression in a training-only PCA history space;
* a nonlinear local-analog (nearest-neighbour) predictor in the same space.

The history window is selected on validation subjects, after which the shared
response, PCA, and predictors are refit on train+validation subjects and scored
once on disjoint test subjects.  Circularly shifted future residuals form a
structure-preserving null.  No forecasting checkpoint is read.

This is a repeated-subject diagnostic: the drive-only predictor uses the
time-aligned response of other subjects and is not a deployable forecaster.
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


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from internal_dw.datasets.hcp import build_hcp_splits  # noqa: E402


def _parse_ints(text: str, *, positive: bool = True) -> list[int]:
    values = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    if not values or (positive and min(values) <= 0):
        raise ValueError(f"invalid integer list: {text!r}")
    return values


def _collect(dataset, label: str) -> Tuple[np.ndarray, list[str]]:
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
    # Static subject offsets are nuisance variation, not movie drive or dynamics.
    result -= result.mean(axis=1, keepdims=True)
    return result, subjects


def _common_time(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    time = min(array.shape[1] for array in arrays)
    dimension = {array.shape[2] for array in arrays}
    if len(dimension) != 1:
        raise ValueError(f"split ROI dimensions differ: {dimension}")
    return tuple(np.ascontiguousarray(array[:, :time]) for array in arrays)


def _pca_projection(
    residual: np.ndarray, rank: int, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    flat = residual.reshape(-1, residual.shape[-1]).astype(np.float32)
    mean = flat.mean(axis=0, keepdims=True)
    centered = torch.from_numpy(np.ascontiguousarray(flat - mean)).to(device)
    covariance = centered.T @ centered / max(int(centered.shape[0]), 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    take = min(int(rank), int(eigenvectors.shape[0]))
    projection = eigenvectors[:, -take:].flip(1).cpu().numpy().astype(np.float32)
    spectrum = eigenvalues.flip(0).cpu().numpy().astype(np.float64)
    del centered, covariance, eigenvalues, eigenvectors
    return projection, spectrum


def _project(residual: np.ndarray, projection: np.ndarray) -> np.ndarray:
    return np.einsum("std,dr->str", residual, projection, optimize=True).astype(
        np.float32
    )


def _starts(time: int, window: int, maximum_horizon: int, stride: int) -> np.ndarray:
    values = np.arange(
        int(window), int(time) - int(maximum_horizon) + 1, max(1, int(stride))
    )
    if values.size == 0:
        raise ValueError(
            f"no valid starts for T={time}, W={window}, K={maximum_horizon}"
        )
    return values


def _features(projected: np.ndarray, starts: np.ndarray, window: int) -> np.ndarray:
    blocks = [projected[:, starts - window + offset] for offset in range(window)]
    return np.concatenate(blocks, axis=-1).astype(np.float32)


def _targets(
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


def _standardize(
    fit: np.ndarray, *others: np.ndarray
) -> Tuple[np.ndarray, ...]:
    flat = fit.reshape(-1, fit.shape[-1])
    mean = flat.mean(axis=0)
    scale = flat.std(axis=0)
    scale = np.maximum(scale, 1e-5)

    def transform(array: np.ndarray) -> np.ndarray:
        return ((array - mean) / scale).astype(np.float32)

    return (transform(fit),) + tuple(transform(array) for array in others)


def _ridge_predict(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    query_x: np.ndarray,
    ridge_relative: float,
    device: torch.device,
) -> np.ndarray:
    """Direct multi-horizon ridge; arrays retain [subject,start,...] axes."""

    x = torch.from_numpy(fit_x.reshape(-1, fit_x.shape[-1])).to(device)
    q = torch.from_numpy(query_x.reshape(-1, query_x.shape[-1])).to(device)
    y = torch.from_numpy(
        fit_y.reshape(-1, fit_y.shape[-2], fit_y.shape[-1])
    ).to(device)
    x_mean = x.mean(dim=0, keepdim=True)
    centered_x = x - x_mean
    gram = centered_x.T @ centered_x / float(centered_x.shape[0])
    scale = torch.trace(gram) / max(int(gram.shape[0]), 1)
    regularized = gram + (
        float(ridge_relative) * scale.clamp_min(1e-12)
        * torch.eye(gram.shape[0], device=device, dtype=gram.dtype)
    )
    cholesky = torch.linalg.cholesky(regularized)
    predictions = []
    for horizon in range(y.shape[1]):
        target = y[:, horizon]
        target_mean = target.mean(dim=0, keepdim=True)
        cross = centered_x.T @ (target - target_mean) / float(centered_x.shape[0])
        weight = torch.cholesky_solve(cross, cholesky)
        predictions.append((q - x_mean) @ weight + target_mean)
    stacked = torch.stack(predictions, dim=1).cpu().numpy().astype(np.float32)
    del x, q, y, centered_x, gram, regularized, cholesky
    return stacked.reshape(
        query_x.shape[0], query_x.shape[1], fit_y.shape[-2], fit_y.shape[-1]
    )


def _nearest_indices(
    fit_x: np.ndarray,
    query_x: np.ndarray,
    neighbors: int,
    chunk: int,
    device: torch.device,
) -> np.ndarray:
    fit = torch.from_numpy(fit_x.reshape(-1, fit_x.shape[-1])).to(device)
    query = torch.from_numpy(query_x.reshape(-1, query_x.shape[-1])).to(device)
    count = min(int(neighbors), int(fit.shape[0]))
    fit_norm = torch.square(fit).sum(dim=1).unsqueeze(0)
    pieces = []
    for begin in range(0, query.shape[0], int(chunk)):
        current = query[begin : begin + int(chunk)]
        distance = (
            torch.square(current).sum(dim=1, keepdim=True)
            + fit_norm
            - 2.0 * current @ fit.T
        )
        pieces.append(torch.topk(distance, k=count, largest=False).indices.cpu())
    indices = torch.cat(pieces, dim=0).numpy()
    del fit, query, fit_norm
    return indices


def _analog_predict(
    fit_y: np.ndarray,
    indices: np.ndarray,
    query_shape: Sequence[int],
) -> np.ndarray:
    target = fit_y.reshape(-1, fit_y.shape[-2], fit_y.shape[-1])
    prediction = target[indices].mean(axis=1, dtype=np.float64).astype(np.float32)
    return prediction.reshape(
        int(query_shape[0]),
        int(query_shape[1]),
        fit_y.shape[-2],
        fit_y.shape[-1],
    )


def _mse_by_subject(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.square(prediction - target, dtype=np.float64).mean(axis=(1, 3))


def _history_value(method_mse: np.ndarray, drive_mse: np.ndarray) -> np.ndarray:
    return 1.0 - method_mse.sum(axis=0) / np.maximum(drive_mse.sum(axis=0), 1e-30)


def _bootstrap_history_value(
    method_mse: np.ndarray,
    drive_mse: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
) -> Dict[str, list[float]]:
    values = []
    count = method_mse.shape[0]
    for _ in range(int(repeats)):
        chosen = rng.integers(0, count, size=count)
        values.append(_history_value(method_mse[chosen], drive_mse[chosen]))
    draws = np.asarray(values)
    return {
        "low": np.quantile(draws, 0.025, axis=0).tolist(),
        "high": np.quantile(draws, 0.975, axis=0).tolist(),
    }


def _json_array(values: Iterable[float]) -> list[float]:
    return np.asarray(values, dtype=np.float64).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--movie", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--roi-dim", type=int, default=400)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--horizons", default="1,2,4,8,16,24,32,48,64")
    parser.add_argument("--history-windows", default="1,2,4,8,16")
    parser.add_argument("--pca-rank", type=int, default=16)
    parser.add_argument("--ridge-relative", type=float, default=1e-2)
    parser.add_argument("--fit-stride", type=int, default=8)
    parser.add_argument("--score-stride", type=int, default=4)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--distance-chunk", type=int, default=256)
    parser.add_argument("--null-repeats", type=int, default=8)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    horizons = _parse_ints(args.horizons)
    windows = _parse_ints(args.history_windows)
    maximum_horizon = max(horizons)
    maximum_window = max(windows)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    split_args = SimpleNamespace(
        data_path=str(args.data_path),
        movie=int(args.movie),
        seed=int(args.seed),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        roi_dim=int(args.roi_dim),
        visual_only=False,
    )
    train_set, val_set, test_set = build_hcp_splits(split_args)
    train, train_subjects = _collect(train_set, "train")
    validation, validation_subjects = _collect(val_set, "validation")
    test, test_subjects = _collect(test_set, "test")
    train, validation, test = _common_time(train, validation, test)
    if set(train_subjects) & set(validation_subjects + test_subjects):
        raise AssertionError("HCP subject splits overlap")
    if set(validation_subjects) & set(test_subjects):
        raise AssertionError("HCP validation/test subject splits overlap")
    if train.shape[1] <= maximum_horizon + maximum_window:
        raise ValueError("fMRI sequence too short for requested horizon/history")

    # Validation-only selection of history length.
    shared_train = train.mean(axis=0)
    train_residual = train - shared_train[None]
    validation_residual = validation - shared_train[None]
    projection, selection_spectrum = _pca_projection(
        train_residual, int(args.pca_rank), device
    )
    train_projected = _project(train_residual, projection)
    validation_projected = _project(validation_residual, projection)
    common_fit_starts = _starts(
        train.shape[1], maximum_window, maximum_horizon, args.fit_stride
    )
    common_score_starts = _starts(
        train.shape[1], maximum_window, maximum_horizon, args.score_stride
    )
    train_targets = _targets(train_residual, common_fit_starts, horizons)
    validation_targets = _targets(
        validation_residual, common_score_starts, horizons
    )
    validation_drive_mse = _mse_by_subject(
        np.zeros_like(validation_targets), validation_targets
    )
    selection_rows = []
    for window in windows:
        fit_x = _features(train_projected, common_fit_starts, window)
        score_x = _features(validation_projected, common_score_starts, window)
        fit_x, score_x = _standardize(fit_x, score_x)
        prediction = _ridge_predict(
            fit_x,
            train_targets,
            score_x,
            float(args.ridge_relative),
            device,
        )
        mse = _mse_by_subject(prediction, validation_targets)
        relative = mse.sum(axis=0) / np.maximum(
            validation_drive_mse.sum(axis=0), 1e-30
        )
        selection_rows.append(
            {
                "window": int(window),
                "mean_relative_mse": float(relative.mean()),
                "relative_mse_by_horizon": relative.tolist(),
            }
        )
        print(
            f"[select] W={window:>2d} mean MSE/drive={relative.mean():.4f}",
            flush=True,
        )
    selected_window = int(
        min(selection_rows, key=lambda row: row["mean_relative_mse"])["window"]
    )
    print(f"[selected] history window={selected_window}", flush=True)

    # Refit every data-dependent object on train+validation, then score test once.
    fit_state = np.concatenate([train, validation], axis=0)
    fit_subjects = train_subjects + validation_subjects
    shared_fit = fit_state.mean(axis=0)
    fit_residual = fit_state - shared_fit[None]
    test_residual = test - shared_fit[None]
    projection, spectrum = _pca_projection(fit_residual, int(args.pca_rank), device)
    fit_projected = _project(fit_residual, projection)
    test_projected = _project(test_residual, projection)
    fit_starts = _starts(
        fit_state.shape[1], selected_window, maximum_horizon, args.fit_stride
    )
    score_starts = _starts(
        test.shape[1], selected_window, maximum_horizon, args.score_stride
    )
    fit_x = _features(fit_projected, fit_starts, selected_window)
    test_x = _features(test_projected, score_starts, selected_window)
    fit_x, test_x = _standardize(fit_x, test_x)
    fit_targets = _targets(fit_residual, fit_starts, horizons)
    test_targets = _targets(test_residual, score_starts, horizons)
    drive_mse = _mse_by_subject(np.zeros_like(test_targets), test_targets)
    # The states were subject-centered at collection time, so zero is the
    # subject-independent mean baseline.  This gives the second, common map
    # coordinate: how much the cross-subject shared movie response improves on
    # an unconditional prediction.
    test_state_targets = _targets(test, score_starts, horizons)
    mean_baseline_mse = _mse_by_subject(
        np.zeros_like(test_state_targets), test_state_targets
    )
    drive_value = 1.0 - drive_mse.sum(axis=0) / np.maximum(
        mean_baseline_mse.sum(axis=0), 1e-30
    )
    ridge_prediction = _ridge_predict(
        fit_x,
        fit_targets,
        test_x,
        float(args.ridge_relative),
        device,
    )
    ridge_mse = _mse_by_subject(ridge_prediction, test_targets)
    neighbor_indices = _nearest_indices(
        fit_x,
        test_x,
        int(args.neighbors),
        int(args.distance_chunk),
        device,
    )
    analog_prediction = _analog_predict(fit_targets, neighbor_indices, test_x.shape)
    analog_mse = _mse_by_subject(analog_prediction, test_targets)

    # Circular shifts keep each fit subject's residual autocorrelation and scale
    # but destroy the mapping from the observed history to its true future.
    rng = np.random.default_rng(int(args.seed) + 7027)
    guard = selected_window + maximum_horizon
    null_ridge_rows = []
    null_analog_rows = []
    for repeat in range(int(args.null_repeats)):
        if fit_state.shape[1] > 2 * guard + 1:
            shifts = rng.integers(
                guard, fit_state.shape[1] - guard, size=fit_state.shape[0]
            )
        else:
            shifts = rng.integers(1, fit_state.shape[1], size=fit_state.shape[0])
        null_targets = _targets(fit_residual, fit_starts, horizons, shifts=shifts)
        null_ridge_prediction = _ridge_predict(
            fit_x,
            null_targets,
            test_x,
            float(args.ridge_relative),
            device,
        )
        null_ridge_rows.append(_mse_by_subject(null_ridge_prediction, test_targets))
        null_analog_prediction = _analog_predict(
            null_targets, neighbor_indices, test_x.shape
        )
        null_analog_rows.append(_mse_by_subject(null_analog_prediction, test_targets))
        print(f"[null] {repeat + 1}/{args.null_repeats}", flush=True)
    null_ridge_stack = np.asarray(null_ridge_rows)
    null_analog_stack = np.asarray(null_analog_rows)
    null_ridge_mse = np.median(null_ridge_stack, axis=0)
    null_analog_mse = np.median(null_analog_stack, axis=0)

    methods = {
        "linear_ridge": ridge_mse,
        "nonlinear_local_analog": analog_mse,
        "circular_shift_linear_null": null_ridge_mse,
        "circular_shift_analog_null": null_analog_mse,
    }
    method_output: Dict[str, object] = {}
    bootstrap_rng = np.random.default_rng(int(args.seed) + 11213)
    for name, mse in methods.items():
        value = _history_value(mse, drive_mse)
        subject_values = 1.0 - mse / np.maximum(drive_mse, 1e-30)
        method_output[name] = {
            "history_value": _json_array(value),
            "history_value_ci95": _bootstrap_history_value(
                mse, drive_mse, int(args.bootstraps), bootstrap_rng
            ),
            "macro_subject_history_value": _json_array(subject_values.mean(axis=0)),
            "mse_by_test_subject_and_horizon": mse.tolist(),
        }

    explained = spectrum[: projection.shape[1]].sum() / max(spectrum.sum(), 1e-30)
    result = {
        "format_version": 1,
        "experiment": "subject_disjoint_fmri_drive_history_decomposition",
        "uses_forecasting_checkpoint": False,
        "drive_only_semantics": (
            "time-aligned shared response estimated from fit subjects; diagnostic, "
            "not a deployable forecaster"
        ),
        "history_value_definition": "1 - MSE(drive+history) / MSE(drive-only)",
        "drive_value_definition": "1 - MSE(shared movie response) / MSE(subject-centered zero mean)",
        "movie": int(args.movie),
        "horizons": [int(value) for value in horizons],
        "selected_history_window": selected_window,
        "window_selection": selection_rows,
        "train_subjects": train_subjects,
        "validation_subjects": validation_subjects,
        "test_subjects": test_subjects,
        "fit_subjects_after_selection": fit_subjects,
        "time_points": int(fit_state.shape[1]),
        "roi_dim": int(fit_state.shape[2]),
        "pca_rank": int(projection.shape[1]),
        "pca_explained_variance_fraction": float(explained),
        "ridge_relative": float(args.ridge_relative),
        "neighbors": int(args.neighbors),
        "fit_stride": int(args.fit_stride),
        "score_stride": int(args.score_stride),
        "null_repeats": int(args.null_repeats),
        "drive_only_mse_by_test_subject_and_horizon": drive_mse.tolist(),
        "mean_baseline_mse_by_test_subject_and_horizon": mean_baseline_mse.tolist(),
        "drive_value": drive_value.tolist(),
        "methods": method_output,
        "selection_pca_spectrum": selection_spectrum.tolist(),
        "final_pca_spectrum": spectrum.tolist(),
        "interpretation_boundary": (
            "A non-positive history value means no incremental history was detected "
            "by either cross-fitted predictor; it is not a proof of zero information."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("\nfMRI incremental residual-history value beyond shared movie drive")
    print(" H    linear   nonlinear   null-linear   null-nonlinear")
    values = {
        name: np.asarray(row["history_value"])
        for name, row in method_output.items()
    }
    for column, horizon in enumerate(horizons):
        print(
            f"{horizon:>2d}  {values['linear_ridge'][column]:8.4f} "
            f"{values['nonlinear_local_analog'][column]:10.4f} "
            f"{values['circular_shift_linear_null'][column]:12.4f} "
            f"{values['circular_shift_analog_null'][column]:15.4f}"
        )
    print(f"[out] {args.output}")


if __name__ == "__main__":
    main()
