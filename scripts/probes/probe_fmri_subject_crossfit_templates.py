#!/usr/bin/env python3
"""Subject-cross-fitted fMRI conditional-innovation templates.

The HCP movie subjects are time aligned because they watch the same stimulus.
For each held-out subject fold, the conditional mean is built only from the
remaining training subjects:

1. the fit-subject mean trajectory estimates the stimulus-locked response;
2. a small coordinatewise AR model predicts the remaining subject deviation;
3. the complete held-out residual trajectory is retained as one innovation
   template, preserving ROI and cross-horizon covariance.

AR order is selected by subject-cross-fitted multi-horizon MSE.  This is a
model-free probe: it reads only the HCP training split and never reads a neural
forecasting checkpoint, validation subject, or test subject.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from internal_dw.datasets.hcp import build_hcp_splits  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--movie", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--roi-dim", type=int, default=400)
    parser.add_argument("--max-horizon", type=int, default=64)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ar-orders", default="0,1,2,4,8")
    parser.add_argument("--ridge-relative", type=float, default=1e-3)
    parser.add_argument("--score-stride", type=int, default=8)
    parser.add_argument("--template-stride", type=int, default=4)
    parser.add_argument("--max-templates", type=int, default=1024)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _parse_ints(text: str, *, allow_zero: bool = False) -> List[int]:
    values = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    lower = 0 if allow_zero else 1
    if not values or min(values) < lower:
        raise ValueError(f"invalid integer list {text!r}")
    return values


def _collect_training_subjects(dataset) -> Tuple[np.ndarray, List[str]]:
    states: List[np.ndarray] = []
    subjects: List[str] = []
    for index in range(len(dataset)):
        item = dataset[index]
        state = item["state"].numpy().astype(np.float32, copy=False)
        if state.ndim != 2:
            raise ValueError(f"expected HCP state [T,D], got {state.shape}")
        states.append(state)
        subjects.append(str(item["metadata"]["subject_id"]))
        if index == 0 or (index + 1) % 20 == 0 or index + 1 == len(dataset):
            print(
                f"[collect] {index + 1}/{len(dataset)} "
                f"subject={subjects[-1]} shape={state.shape}",
                flush=True,
            )
    if len(states) < 6:
        raise ValueError(f"need at least six training subjects, found {len(states)}")
    time = min(row.shape[0] for row in states)
    dimensions = {row.shape[1] for row in states}
    if len(dimensions) != 1:
        raise ValueError(f"inconsistent ROI dimensions: {sorted(dimensions)}")
    stacked = np.stack([row[:time] for row in states]).astype(np.float32)
    if not np.isfinite(stacked).all():
        raise ValueError("HCP training states contain non-finite values")
    return stacked, subjects


def _fold_assignment(count: int, folds: int, seed: int) -> np.ndarray:
    folds = max(2, min(int(folds), int(count)))
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(count)
    assignment = np.empty(count, dtype=np.int64)
    assignment[permutation] = np.arange(count, dtype=np.int64) % folds
    return assignment


def _fit_diagonal_ar(
    sequences: np.ndarray,
    order: int,
    ridge_relative: float,
    device: torch.device,
) -> np.ndarray:
    """Fit one AR(order) per ROI to [S,T,D] sequences."""

    dimension = int(sequences.shape[-1])
    if order == 0:
        return np.zeros((dimension, 0), dtype=np.float32)
    if sequences.shape[1] <= order:
        raise ValueError(f"order {order} exceeds sequence length {sequences.shape[1]}")
    # X is [sample, ROI, lag], with lag zero denoting t-1.
    x = np.stack(
        [sequences[:, order - lag : sequences.shape[1] - lag] for lag in range(1, order + 1)],
        axis=-1,
    ).reshape(-1, dimension, order)
    y = sequences[:, order:].reshape(-1, dimension)
    tx = torch.from_numpy(np.ascontiguousarray(x)).to(device=device, dtype=torch.float32)
    ty = torch.from_numpy(np.ascontiguousarray(y)).to(device=device, dtype=torch.float32)
    gram = torch.einsum("ndp,ndq->dpq", tx, tx) / float(tx.shape[0])
    cross = torch.einsum("ndp,nd->dp", tx, ty) / float(tx.shape[0])
    scale = gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-12)
    eye = torch.eye(order, device=device, dtype=gram.dtype).unsqueeze(0)
    gram = gram + float(ridge_relative) * scale[:, None, None] * eye
    coefficients = torch.linalg.solve(gram, cross.unsqueeze(-1)).squeeze(-1)
    result = coefficients.cpu().numpy().astype(np.float32)
    del tx, ty, gram, cross, coefficients
    return _stabilize_diagonal_ar(result)


def _root_radius(coefficients: np.ndarray) -> float:
    if coefficients.size == 0:
        return 0.0
    roots = np.roots(np.r_[1.0, -coefficients.astype(np.float64)])
    return float(np.max(np.abs(roots))) if roots.size else 0.0


def _stabilize_diagonal_ar(coefficients: np.ndarray, limit: float = 0.995) -> np.ndarray:
    """Shrink only unstable scalar AR rows, using a root-radius bisection."""

    result = coefficients.copy()
    for coordinate in range(result.shape[0]):
        row = result[coordinate].astype(np.float64)
        if _root_radius(row) <= limit:
            continue
        low, high = 0.0, 1.0
        for _ in range(24):
            middle = 0.5 * (low + high)
            if _root_radius(row * middle) <= limit:
                low = middle
            else:
                high = middle
        result[coordinate] = (row * low).astype(np.float32)
    return result


def _forecast_deviation(
    deviation: np.ndarray,
    starts: np.ndarray,
    coefficients: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Forecast one subject deviation at many starts; return [N,K,D]."""

    starts = np.asarray(starts, dtype=np.int64)
    count = int(starts.size)
    dimension, order = coefficients.shape
    if order == 0:
        return np.zeros((count, horizon, dimension), dtype=np.float32)
    history = np.stack([deviation[start - order : start] for start in starts]).astype(
        np.float32
    )
    output = np.empty((count, horizon, dimension), dtype=np.float32)
    for step in range(horizon):
        # history is chronological; coefficients[:,0] multiplies t-1.
        prediction = np.einsum(
            "npd,dp->nd", history[:, ::-1, :], coefficients, optimize=True
        ).astype(np.float32)
        output[:, step] = prediction
        if order > 1:
            history[:, :-1] = history[:, 1:]
        history[:, -1] = prediction
    return output


def _horizon_indices(horizon: int) -> List[int]:
    wanted = [1, 2, 4, 8, 16, 32, 64, 96, 128]
    return [value for value in wanted if value <= horizon]


def _score_fold_order(
    centered: np.ndarray,
    fit_indices: np.ndarray,
    held_indices: np.ndarray,
    order: int,
    common_start: int,
    horizon: int,
    stride: int,
    ridge_relative: float,
    device: torch.device,
) -> Dict[str, object]:
    shared = centered[fit_indices].mean(axis=0)
    fit_deviation = centered[fit_indices] - shared[None]
    coefficients = _fit_diagonal_ar(
        fit_deviation, order, ridge_relative=ridge_relative, device=device
    )
    total_squared = np.zeros(horizon, dtype=np.float64)
    shared_squared = np.zeros(horizon, dtype=np.float64)
    target_squared = np.zeros(horizon, dtype=np.float64)
    elements = np.zeros(horizon, dtype=np.float64)
    trajectory_count = 0
    last_start = centered.shape[1] - horizon
    starts = np.arange(common_start, last_start + 1, max(1, int(stride)))
    if starts.size == 0:
        raise ValueError(
            f"no fMRI starts: T={centered.shape[1]} start={common_start} K={horizon}"
        )
    future_index = starts[:, None] + np.arange(horizon, dtype=np.int64)[None]
    for subject_index in held_indices:
        observed = centered[int(subject_index)]
        deviation = observed - shared
        deviation_prediction = _forecast_deviation(
            deviation, starts, coefficients, horizon
        )
        prediction = shared[future_index] + deviation_prediction
        target = observed[future_index]
        residual = target - prediction
        shared_residual = target - shared[future_index]
        total_squared += np.square(residual, dtype=np.float64).sum(axis=(0, 2))
        shared_squared += np.square(shared_residual, dtype=np.float64).sum(axis=(0, 2))
        target_squared += np.square(target, dtype=np.float64).sum(axis=(0, 2))
        elements += float(target.shape[0] * target.shape[2])
        trajectory_count += int(target.shape[0])
    mse = total_squared / np.maximum(elements, 1.0)
    shared_mse = shared_squared / np.maximum(elements, 1.0)
    target_mse = target_squared / np.maximum(elements, 1.0)
    return {
        "order": int(order),
        "trajectory_count": int(trajectory_count),
        "mse_by_horizon": mse.tolist(),
        "shared_only_mse_by_horizon": shared_mse.tolist(),
        "residual_energy_fraction_by_horizon": (
            total_squared / np.maximum(target_squared, 1e-30)
        ).tolist(),
        "relative_to_shared_only_by_horizon": (
            total_squared / np.maximum(shared_squared, 1e-30)
        ).tolist(),
        "mean_mse": float(mse.mean()),
        "mean_shared_only_mse": float(shared_mse.mean()),
        "mean_target_energy": float(target_mse.mean()),
        "max_root_radius": float(
            max((_root_radius(row) for row in coefficients), default=0.0)
        ),
    }


def _select_order(
    centered: np.ndarray,
    assignment: np.ndarray,
    orders: Sequence[int],
    horizon: int,
    score_stride: int,
    ridge_relative: float,
    device: torch.device,
) -> Tuple[int, Dict[str, object]]:
    folds = int(assignment.max()) + 1
    common_start = max(max(orders), 1)
    fold_rows: List[Dict[str, object]] = []
    totals = {int(order): np.zeros(horizon, dtype=np.float64) for order in orders}
    counts = {int(order): 0 for order in orders}
    for fold in range(folds):
        fit_indices = np.flatnonzero(assignment != fold)
        held_indices = np.flatnonzero(assignment == fold)
        for order in orders:
            row = _score_fold_order(
                centered,
                fit_indices,
                held_indices,
                int(order),
                common_start,
                horizon,
                score_stride,
                ridge_relative,
                device,
            )
            row["fold"] = int(fold)
            fold_rows.append(row)
            trajectories = int(row["trajectory_count"])
            totals[int(order)] += np.asarray(row["mse_by_horizon"]) * trajectories
            counts[int(order)] += trajectories
            print(
                "[score] fold=%d/%d order=%d trajectories=%d mean_mse=%.6g "
                "vs_shared=%.4f"
                % (
                    fold + 1,
                    folds,
                    order,
                    trajectories,
                    row["mean_mse"],
                    row["mean_mse"] / max(row["mean_shared_only_mse"], 1e-30),
                ),
                flush=True,
            )
    aggregate = {}
    for order in orders:
        curve = totals[int(order)] / max(counts[int(order)], 1)
        aggregate[str(order)] = {
            "mean_mse": float(curve.mean()),
            "mse_by_horizon": curve.tolist(),
            "trajectory_count": int(counts[int(order)]),
        }
    selected = min(orders, key=lambda value: aggregate[str(value)]["mean_mse"])
    shared_reference = aggregate.get("0")
    return int(selected), {
        "selection_metric": "subject_cross_fitted_equal_horizon_mean_mse",
        "common_scoring_start": int(common_start),
        "selected_relative_to_shared_only": (
            float(
                aggregate[str(selected)]["mean_mse"]
                / max(shared_reference["mean_mse"], 1e-30)
            )
            if shared_reference is not None
            else None
        ),
        "candidate_aggregate": aggregate,
        "fold_candidate_scores": fold_rows,
    }


def _build_templates(
    centered: np.ndarray,
    assignment: np.ndarray,
    order: int,
    horizon: int,
    stride: int,
    maximum: int,
    ridge_relative: float,
    device: torch.device,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    folds = int(assignment.max()) + 1
    start_min = max(int(order), 1)
    last_start = centered.shape[1] - horizon
    descriptors: List[Tuple[int, int, int]] = []
    for fold in range(folds):
        for subject_index in np.flatnonzero(assignment == fold):
            for start in range(start_min, last_start + 1, max(1, int(stride))):
                descriptors.append((fold, int(subject_index), int(start)))
    if not descriptors:
        raise ValueError("no valid held-out fMRI template trajectories")
    rng = np.random.default_rng(int(seed) + 1709)
    if maximum > 0 and len(descriptors) > maximum:
        chosen = np.sort(rng.choice(len(descriptors), size=maximum, replace=False))
        descriptors = [descriptors[int(index)] for index in chosen]

    templates: List[np.ndarray] = []
    fold_counts: Dict[str, int] = {}
    subject_counts: Dict[str, int] = {}
    for fold in range(folds):
        selected = [row for row in descriptors if row[0] == fold]
        if not selected:
            continue
        fit_indices = np.flatnonzero(assignment != fold)
        shared = centered[fit_indices].mean(axis=0)
        coefficients = _fit_diagonal_ar(
            centered[fit_indices] - shared[None],
            order,
            ridge_relative=ridge_relative,
            device=device,
        )
        by_subject: Dict[int, List[int]] = {}
        for _, subject_index, start in selected:
            by_subject.setdefault(subject_index, []).append(start)
        for subject_index, starts_list in sorted(by_subject.items()):
            starts = np.asarray(starts_list, dtype=np.int64)
            observed = centered[subject_index]
            predicted_deviation = _forecast_deviation(
                observed - shared, starts, coefficients, horizon
            )
            indices = starts[:, None] + np.arange(horizon)[None]
            prediction = shared[indices] + predicted_deviation
            templates.append((observed[indices] - prediction).astype(np.float32))
            key = str(subject_index)
            subject_counts[key] = subject_counts.get(key, 0) + int(starts.size)
        fold_counts[str(fold)] = len(selected)
        print(
            f"[templates] fold={fold + 1}/{folds} count={len(selected)}",
            flush=True,
        )
    output = np.concatenate(templates, axis=0)
    permutation = rng.permutation(output.shape[0])
    output = np.ascontiguousarray(output[permutation])
    centered_templates = output - output.mean(axis=0, keepdims=True)
    residual_variance = np.square(centered_templates, dtype=np.float64).mean(axis=(0, 2))
    lag_numerator = float(
        np.sum(centered_templates[:, :-1].astype(np.float64) * centered_templates[:, 1:])
    )
    lag_denominator = float(
        np.sqrt(
            np.sum(np.square(centered_templates[:, :-1], dtype=np.float64))
            * np.sum(np.square(centered_templates[:, 1:], dtype=np.float64))
        )
    )
    diagnostics = {
        "template_count": int(output.shape[0]),
        "fold_template_counts": fold_counts,
        "subject_template_counts": subject_counts,
        "template_variance_by_horizon": residual_variance.tolist(),
        "template_lag1_correlation": float(
            lag_numerator / max(lag_denominator, 1e-30)
        ),
    }
    return output, diagnostics


def main() -> None:
    a = parse_args()
    orders = _parse_ints(a.ar_orders, allow_zero=True)
    if a.max_horizon <= 0:
        raise ValueError("max-horizon must be positive")
    split_args = SimpleNamespace(
        data_path=str(a.data_path),
        movie=int(a.movie),
        seed=int(a.seed),
        train_ratio=float(a.train_ratio),
        val_ratio=float(a.val_ratio),
        roi_dim=int(a.roi_dim),
        visual_only=False,
    )
    train, _, _ = build_hcp_splits(split_args)
    state, subjects = _collect_training_subjects(train)
    if state.shape[1] <= int(a.max_horizon) + max(orders):
        raise ValueError(
            f"HCP T={state.shape[1]} is too short for K={a.max_horizon} "
            f"and max AR order={max(orders)}"
        )
    # Subject baselines are static nuisance offsets.  Removing them is the
    # same training-only centering used by the previous shared-response probe.
    centered = state - state.mean(axis=1, keepdims=True)
    assignment = _fold_assignment(len(subjects), int(a.folds), int(a.seed))
    device = torch.device(a.device)
    selected_order, selection = _select_order(
        centered,
        assignment,
        orders,
        int(a.max_horizon),
        int(a.score_stride),
        float(a.ridge_relative),
        device,
    )
    print(f"[selected] diagonal subject-deviation AR order={selected_order}", flush=True)
    templates, template_diagnostics = _build_templates(
        centered,
        assignment,
        selected_order,
        int(a.max_horizon),
        int(a.template_stride),
        int(a.max_templates),
        float(a.ridge_relative),
        device,
        int(a.seed),
    )
    selected_curve = selection["candidate_aggregate"][str(selected_order)][
        "mse_by_horizon"
    ]
    report_horizons = _horizon_indices(int(a.max_horizon))
    metadata = {
        "format_version": 1,
        "dataset": "hcp_movie",
        "movie": int(a.movie),
        "fit_split": "training_subjects_only_subject_cross_fitted",
        "uses_forecasting_checkpoint": False,
        "conditional_model": (
            "fit_subject_time_aligned_shared_response_plus_coordinatewise_AR_deviation"
        ),
        "innovation_sampler": "whole_heldout_residual_trajectory_bootstrap",
        "subjects": subjects,
        "subject_count": int(len(subjects)),
        "time_points": int(state.shape[1]),
        "state_dim": int(state.shape[2]),
        "folds": int(assignment.max()) + 1,
        "fold_assignment": assignment.tolist(),
        "candidate_ar_orders": [int(value) for value in orders],
        "selected_ar_order": int(selected_order),
        "max_horizon": int(a.max_horizon),
        "ridge_relative": float(a.ridge_relative),
        "score_stride": int(a.score_stride),
        "template_stride": int(a.template_stride),
        "reported_selected_mse": {
            str(h): float(selected_curve[h - 1]) for h in report_horizons
        },
        "assumption": (
            "Responses shared across time-aligned training subjects are stimulus-locked; "
            "the held-out remainder after a cross-fitted shared response and a selected "
            "short diagonal AR deviation model is the conditional innovation."
        ),
        **selection,
        **template_diagnostics,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.output,
        innovation_templates=templates.astype(np.float16),
        innovation_estimator_name=np.asarray(
            "hcp_subject_crossfit_shared_response_residual_bootstrap"
        ),
        selected_ar_order=np.asarray(selected_order, dtype=np.int64),
    )
    a.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[out] {a.output}")
    print(f"[out] {a.output.with_suffix('.json')}")
    print(
        "[HCP crossfit] subjects=%d folds=%d order=%d templates=%d K=%d "
        "lag1=%.4f cv_vs_shared=%.4f"
        % (
            len(subjects),
            int(assignment.max()) + 1,
            selected_order,
            templates.shape[0],
            templates.shape[1],
            template_diagnostics["template_lag1_correlation"],
            selection["selected_relative_to_shared_only"],
        )
    )


if __name__ == "__main__":
    main()
