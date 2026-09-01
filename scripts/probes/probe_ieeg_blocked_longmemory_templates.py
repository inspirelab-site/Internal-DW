#!/usr/bin/env python3
"""Blocked-cross-fitted long-memory innovation templates for iEEG envelopes.

The iEEG testbed contains band-amplitude envelopes, whose slow dependence is
much longer than the oscillation period of the carrier.  This probe fits a
dynamic factor conditional mean on disjoint portions of the training block:

* PCA learns a stable cross-channel subspace from fit time blocks;
* one scalar AR process per factor models its long temporal memory, while the
  orthogonal remainder is carried forward from the latest observed sample;
* rank and AR order are selected by purged blocked cross-fitted rollout MSE;
* complete held-out residual trajectories are saved for empirical bootstrap.

Only the training split is used.  No neural forecasting checkpoint, validation
segment, or test segment is read.
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

from internal_dw.datasets.ieeg import build_ieeg_splits  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/synthetic")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subject", default="P41CS")
    parser.add_argument("--task", default="enc")
    parser.add_argument("--contact", default="macro")
    parser.add_argument("--band", default="theta")
    parser.add_argument("--step-ms", type=float, default=20.0)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--max-channels", type=int, default=0)
    parser.add_argument("--ieeg-root", default="")
    parser.add_argument("--max-horizon", type=int, default=64)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--purge-gap", type=int, default=128)
    parser.add_argument("--factor-ranks", default="8,16,32,64,80")
    parser.add_argument("--ar-orders", default="8,16,32,64")
    parser.add_argument("--ridge-relative", type=float, default=1e-3)
    parser.add_argument("--score-stride", type=int, default=32)
    parser.add_argument("--template-stride", type=int, default=8)
    parser.add_argument("--max-templates", type=int, default=1024)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _parse_positive_ints(text: str) -> List[int]:
    values = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    if not values or min(values) <= 0:
        raise ValueError(f"invalid positive integer list {text!r}")
    return values


def _load_training_series(a: argparse.Namespace) -> np.ndarray:
    split_args = SimpleNamespace(
        data_path=str(a.data_path),
        seed=int(a.seed),
        train_ratio=float(a.train_ratio),
        val_ratio=float(a.val_ratio),
        ieeg_subject=str(a.subject),
        ieeg_task=str(a.task),
        ieeg_contact=str(a.contact),
        ieeg_band=str(a.band),
        ieeg_step_ms=float(a.step_ms),
        ieeg_chunk=int(a.chunk),
        ieeg_max_channels=int(a.max_channels),
        ieeg_split_gap=256,
    )
    if str(a.ieeg_root):
        split_args.ieeg_root = str(a.ieeg_root)
    train, _, _ = build_ieeg_splits(split_args)
    series = np.ascontiguousarray(train.chunks.reshape(-1, train.chunks.shape[-1]))
    if series.ndim != 2 or not np.isfinite(series).all():
        raise ValueError(f"invalid normalized iEEG training series {series.shape}")
    print(
        f"[iEEG train] shape={series.shape} mean={series.mean():.5g} "
        f"std={series.std():.5g}",
        flush=True,
    )
    return series.astype(np.float32, copy=False)


def _fold_ranges(length: int, folds: int) -> List[Tuple[int, int]]:
    folds = max(2, min(int(folds), max(2, length // 512)))
    edges = np.linspace(0, length, folds + 1, dtype=np.int64)
    return [(int(edges[index]), int(edges[index + 1])) for index in range(folds)]


def _fit_segments(
    series: np.ndarray, held_start: int, held_end: int, gap: int
) -> List[np.ndarray]:
    segments: List[np.ndarray] = []
    left_end = max(0, int(held_start) - int(gap))
    right_start = min(series.shape[0], int(held_end) + int(gap))
    if left_end > 1:
        segments.append(series[:left_end])
    if right_start < series.shape[0] - 1:
        segments.append(series[right_start:])
    return segments


def _fit_pca(
    segments: Sequence[np.ndarray], maximum_rank: int, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit = np.concatenate(segments, axis=0).astype(np.float32, copy=False)
    mean = fit.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = torch.from_numpy(np.ascontiguousarray(fit - mean)).to(device)
    covariance = centered.T @ centered / float(max(centered.shape[0] - 1, 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0)
    eigenvectors = eigenvectors[:, order]
    rank = min(int(maximum_rank), int(eigenvectors.shape[1]))
    basis = eigenvectors[:, :rank].cpu().numpy().astype(np.float32)
    values = eigenvalues.cpu().numpy().astype(np.float64)
    explained = np.cumsum(values) / max(float(values.sum()), 1e-30)
    del centered, covariance, eigenvalues, eigenvectors
    return mean, basis, explained


def _factor_segments(
    segments: Sequence[np.ndarray], mean: np.ndarray, basis: np.ndarray
) -> List[np.ndarray]:
    return [
        np.ascontiguousarray((segment - mean) @ basis, dtype=np.float32)
        for segment in segments
    ]


def _root_radius(coefficients: np.ndarray) -> float:
    roots = np.roots(np.r_[1.0, -coefficients.astype(np.float64)])
    return float(np.max(np.abs(roots))) if roots.size else 0.0


def _stabilize_factor_ar(coefficients: np.ndarray, limit: float = 0.995) -> np.ndarray:
    result = coefficients.copy()
    for factor in range(result.shape[0]):
        row = result[factor].astype(np.float64)
        if _root_radius(row) <= limit:
            continue
        low, high = 0.0, 1.0
        for _ in range(24):
            middle = 0.5 * (low + high)
            if _root_radius(row * middle) <= limit:
                low = middle
            else:
                high = middle
        result[factor] = (row * low).astype(np.float32)
    return result


def _fit_factor_ar(
    factor_segments: Sequence[np.ndarray],
    order: int,
    ridge_relative: float,
    device: torch.device,
) -> np.ndarray:
    rank = int(factor_segments[0].shape[1])
    x_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []
    for segment in factor_segments:
        if segment.shape[0] <= order:
            continue
        x_rows.append(
            np.stack(
                [segment[order - lag : segment.shape[0] - lag] for lag in range(1, order + 1)],
                axis=1,
            )
        )
        y_rows.append(segment[order:])
    if not x_rows:
        raise ValueError(f"no fit segment is longer than AR order {order}")
    x = np.concatenate(x_rows, axis=0)  # [N,p,R]
    y = np.concatenate(y_rows, axis=0)  # [N,R]
    tx = torch.from_numpy(np.ascontiguousarray(x)).to(device)
    ty = torch.from_numpy(np.ascontiguousarray(y)).to(device)
    gram = torch.einsum("npr,nqr->rpq", tx, tx) / float(tx.shape[0])
    cross = torch.einsum("npr,nr->rp", tx, ty) / float(tx.shape[0])
    scale = gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-12)
    eye = torch.eye(order, device=device, dtype=gram.dtype).unsqueeze(0)
    gram = gram + float(ridge_relative) * scale[:, None, None] * eye
    coefficients = torch.linalg.solve(gram, cross.unsqueeze(-1)).squeeze(-1)
    result = coefficients.cpu().numpy().astype(np.float32)
    del tx, ty, gram, cross, coefficients, x, y
    return _stabilize_factor_ar(result)


def _forecast_factors(
    held_factors: np.ndarray,
    starts: np.ndarray,
    coefficients: np.ndarray,
    horizon: int,
) -> np.ndarray:
    starts = np.asarray(starts, dtype=np.int64)
    rank, order = coefficients.shape
    history = np.stack(
        [held_factors[start - order : start, :rank] for start in starts]
    ).astype(np.float32)
    output = np.empty((starts.size, horizon, rank), dtype=np.float32)
    for step in range(horizon):
        prediction = np.einsum(
            "npr,rp->nr", history[:, ::-1, :], coefficients, optimize=True
        ).astype(np.float32)
        output[:, step] = prediction
        if order > 1:
            history[:, :-1] = history[:, 1:]
        history[:, -1] = prediction
    return output


def _forecast_residual(
    held: np.ndarray,
    starts: np.ndarray,
    mean: np.ndarray,
    basis: np.ndarray,
    coefficients: np.ndarray,
    rank: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    held_factors = (held - mean) @ basis
    factor_prediction = _forecast_factors(
        held_factors, starts, coefficients[:rank], horizon
    )
    # Do not erase the high-dimensional component outside the selected factor
    # subspace.  It is observable at the rollout start and is locally smooth in
    # a band envelope, so persistence is the conservative conditional mean for
    # that remainder.  Only the selected slow factors are recursively modeled.
    latest = held[starts - 1]
    latest_factor_part = held_factors[starts - 1, :rank] @ basis[:, :rank].T
    remainder = latest - mean[None] - latest_factor_part
    prediction = (
        mean[None, None]
        + factor_prediction @ basis[:, :rank].T
        + remainder[:, None, :]
    )
    indices = starts[:, None] + np.arange(horizon, dtype=np.int64)[None]
    target = held[indices]
    return (target - prediction).astype(np.float32), target.astype(np.float32)


def _score_candidates(
    series: np.ndarray,
    ranges: Sequence[Tuple[int, int]],
    ranks: Sequence[int],
    orders: Sequence[int],
    horizon: int,
    gap: int,
    stride: int,
    ridge_relative: float,
    device: torch.device,
) -> Tuple[Tuple[int, int], Dict[str, object]]:
    max_rank = min(max(ranks), series.shape[1])
    common_history = max(orders)
    sum_squared: Dict[Tuple[int, int], np.ndarray] = {
        (rank, order): np.zeros(horizon, dtype=np.float64)
        for rank in ranks
        for order in orders
        if rank <= max_rank
    }
    persistence_squared = np.zeros(horizon, dtype=np.float64)
    target_squared = np.zeros(horizon, dtype=np.float64)
    counts = np.zeros(horizon, dtype=np.float64)
    fold_rows: List[Dict[str, object]] = []
    subspaces: List[np.ndarray] = []

    for fold, (held_start, held_end) in enumerate(ranges):
        fit = _fit_segments(series, held_start, held_end, gap)
        minimum_fit = common_history + 2
        fit = [segment for segment in fit if segment.shape[0] >= minimum_fit]
        if not fit:
            raise ValueError(f"fold {fold} has no usable fit segment after purge gap")
        mean, basis, explained = _fit_pca(fit, max_rank, device)
        subspaces.append(basis[:, :max_rank])
        factor_fit = _factor_segments(fit, mean, basis)
        coefficient_by_order = {
            order: _fit_factor_ar(
                factor_fit, order, ridge_relative=ridge_relative, device=device
            )
            for order in orders
        }
        held = series[held_start:held_end]
        starts = np.arange(
            common_history,
            held.shape[0] - horizon + 1,
            max(1, int(stride)),
            dtype=np.int64,
        )
        if starts.size == 0:
            raise ValueError(
                f"held block {fold} length={held.shape[0]} has no K={horizon} starts"
            )
        indices = starts[:, None] + np.arange(horizon, dtype=np.int64)[None]
        target = held[indices]
        persistence = held[starts - 1][:, None, :]
        fold_persistence = np.square(
            target.astype(np.float64) - persistence.astype(np.float64)
        ).sum(axis=(0, 2))
        persistence_squared += fold_persistence
        target_squared += np.square(target, dtype=np.float64).sum(axis=(0, 2))
        counts += float(starts.size * held.shape[1])
        fold_scores: Dict[str, float] = {}
        held_factors = (held - mean) @ basis
        for order in orders:
            factor_prediction_all = _forecast_factors(
                held_factors, starts, coefficient_by_order[order], horizon
            )
            for rank in ranks:
                if rank > max_rank:
                    continue
                latest_factor_part = (
                    held_factors[starts - 1, :rank] @ basis[:, :rank].T
                )
                remainder = held[starts - 1] - mean[None] - latest_factor_part
                prediction = (
                    mean[None, None]
                    + factor_prediction_all[:, :, :rank] @ basis[:, :rank].T
                    + remainder[:, None, :]
                )
                squared = np.square(
                    target.astype(np.float64) - prediction.astype(np.float64)
                ).sum(axis=(0, 2))
                sum_squared[(rank, order)] += squared
                fold_scores[f"r{rank}_p{order}"] = float(
                    squared.sum() / max(float(target.size), 1.0)
                )
        fold_rows.append(
            {
                "fold": int(fold),
                "held_range": [int(held_start), int(held_end)],
                "fit_lengths": [int(segment.shape[0]) for segment in fit],
                "starts": int(starts.size),
                "persistence_mean_mse": float(
                    fold_persistence.sum() / max(float(target.size), 1.0)
                ),
                "explained_variance_at_max_rank": float(explained[max_rank - 1]),
                "candidate_mean_mse": fold_scores,
            }
        )
        best_fold = min(fold_scores, key=fold_scores.get)
        print(
            "[score] fold=%d/%d held=%d starts=%d best=%s mse=%.6g persistence=%.6g"
            % (
                fold + 1,
                len(ranges),
                held.shape[0],
                starts.size,
                best_fold,
                fold_scores[best_fold],
                fold_rows[-1]["persistence_mean_mse"],
            ),
            flush=True,
        )

    aggregate: Dict[str, object] = {}
    for (rank, order), squared in sum_squared.items():
        mse = squared / np.maximum(counts, 1.0)
        key = f"r{rank}_p{order}"
        aggregate[key] = {
            "rank": int(rank),
            "order": int(order),
            "mean_mse": float(mse.mean()),
            "mse_by_horizon": mse.tolist(),
            "relative_to_persistence": float(
                squared.sum() / max(float(persistence_squared.sum()), 1e-30)
            ),
            "residual_energy_fraction": float(
                squared.sum() / max(float(target_squared.sum()), 1e-30)
            ),
        }
    selected_key = min(aggregate, key=lambda key: aggregate[key]["mean_mse"])
    selected = (
        int(aggregate[selected_key]["rank"]),
        int(aggregate[selected_key]["order"]),
    )

    similarities: List[float] = []
    for left in range(len(subspaces)):
        for right in range(left + 1, len(subspaces)):
            singular = np.linalg.svd(
                subspaces[left].T @ subspaces[right], compute_uv=False
            )
            similarities.append(float(np.mean(np.square(singular))))
    diagnostics = {
        "selection_metric": "purged_blocked_cross_fitted_equal_horizon_mean_mse",
        "candidate_aggregate": aggregate,
        "fold_candidate_scores": fold_rows,
        "persistence_mse_by_horizon": (
            persistence_squared / np.maximum(counts, 1.0)
        ).tolist(),
        "target_energy_by_horizon": (
            target_squared / np.maximum(counts, 1.0)
        ).tolist(),
        "max_rank_subspace_similarity_mean": float(np.mean(similarities)),
        "max_rank_subspace_similarity_min": float(np.min(similarities)),
    }
    return selected, diagnostics


def _build_templates(
    series: np.ndarray,
    ranges: Sequence[Tuple[int, int]],
    rank: int,
    order: int,
    horizon: int,
    gap: int,
    stride: int,
    maximum: int,
    ridge_relative: float,
    device: torch.device,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    descriptors: List[Tuple[int, int]] = []
    for fold, (held_start, held_end) in enumerate(ranges):
        held_length = held_end - held_start
        for start in range(order, held_length - horizon + 1, max(1, int(stride))):
            descriptors.append((fold, start))
    if not descriptors:
        raise ValueError("no valid iEEG held-out template trajectories")
    rng = np.random.default_rng(int(seed) + 2207)
    if maximum > 0 and len(descriptors) > maximum:
        chosen = np.sort(rng.choice(len(descriptors), size=maximum, replace=False))
        descriptors = [descriptors[int(index)] for index in chosen]

    rows: List[np.ndarray] = []
    fold_counts: Dict[str, int] = {}
    fold_energy: Dict[str, float] = {}
    for fold, (held_start, held_end) in enumerate(ranges):
        starts = np.asarray(
            [start for candidate_fold, start in descriptors if candidate_fold == fold],
            dtype=np.int64,
        )
        if starts.size == 0:
            continue
        fit = _fit_segments(series, held_start, held_end, gap)
        fit = [segment for segment in fit if segment.shape[0] > order]
        mean, basis, _ = _fit_pca(fit, rank, device)
        coefficients = _fit_factor_ar(
            _factor_segments(fit, mean, basis),
            order,
            ridge_relative=ridge_relative,
            device=device,
        )
        residual, target = _forecast_residual(
            series[held_start:held_end],
            starts,
            mean,
            basis,
            coefficients,
            rank,
            horizon,
        )
        rows.append(residual)
        fold_counts[str(fold)] = int(starts.size)
        fold_energy[str(fold)] = float(
            np.square(residual, dtype=np.float64).sum()
            / max(float(np.square(target, dtype=np.float64).sum()), 1e-30)
        )
        print(
            f"[templates] fold={fold + 1}/{len(ranges)} count={starts.size}",
            flush=True,
        )
    output = np.concatenate(rows, axis=0)
    output = np.ascontiguousarray(output[rng.permutation(output.shape[0])])
    centered = output - output.mean(axis=0, keepdims=True)
    variance = np.square(centered, dtype=np.float64).mean(axis=(0, 2))
    numerator = float(
        np.sum(centered[:, :-1].astype(np.float64) * centered[:, 1:])
    )
    denominator = float(
        np.sqrt(
            np.square(centered[:, :-1], dtype=np.float64).sum()
            * np.square(centered[:, 1:], dtype=np.float64).sum()
        )
    )
    diagnostics = {
        "template_count": int(output.shape[0]),
        "fold_template_counts": fold_counts,
        "fold_residual_energy_fraction": fold_energy,
        "template_variance_by_horizon": variance.tolist(),
        "template_lag1_correlation": float(numerator / max(denominator, 1e-30)),
    }
    return output, diagnostics


def _horizon_indices(horizon: int) -> List[int]:
    return [value for value in (1, 2, 4, 8, 16, 32, 64, 96, 128) if value <= horizon]


def main() -> None:
    a = parse_args()
    ranks = _parse_positive_ints(a.factor_ranks)
    orders = _parse_positive_ints(a.ar_orders)
    series = _load_training_series(a)
    ranks = [rank for rank in ranks if rank <= series.shape[1]]
    if not ranks:
        raise ValueError(
            f"all requested factor ranks exceed iEEG dimension {series.shape[1]}"
        )
    if a.max_horizon <= 0:
        raise ValueError("max-horizon must be positive")
    gap = max(int(a.purge_gap), max(orders), int(a.max_horizon))
    ranges = _fold_ranges(series.shape[0], int(a.folds))
    device = torch.device(a.device)
    (selected_rank, selected_order), selection = _score_candidates(
        series,
        ranges,
        ranks,
        orders,
        int(a.max_horizon),
        gap,
        int(a.score_stride),
        float(a.ridge_relative),
        device,
    )
    print(
        f"[selected] dynamic-factor rank={selected_rank} AR order={selected_order}",
        flush=True,
    )
    templates, template_diagnostics = _build_templates(
        series,
        ranges,
        selected_rank,
        selected_order,
        int(a.max_horizon),
        gap,
        int(a.template_stride),
        int(a.max_templates),
        float(a.ridge_relative),
        device,
        int(a.seed),
    )
    selected_key = f"r{selected_rank}_p{selected_order}"
    selected_curve = selection["candidate_aggregate"][selected_key]["mse_by_horizon"]
    horizons = _horizon_indices(int(a.max_horizon))
    metadata = {
        "format_version": 1,
        "dataset": "ieeg",
        "subject": str(a.subject),
        "task": str(a.task),
        "contact": str(a.contact),
        "band": str(a.band),
        "step_ms": float(a.step_ms),
        "fit_split": "training_time_only_purged_block_cross_fitted",
        "uses_forecasting_checkpoint": False,
        "conditional_model": (
            "PCA_dynamic_factors_with_independent_long_memory_AR_and_"
            "persistent_orthogonal_remainder"
        ),
        "innovation_sampler": "whole_heldout_residual_trajectory_bootstrap",
        "training_steps": int(series.shape[0]),
        "state_dim": int(series.shape[1]),
        "fold_ranges": [[int(left), int(right)] for left, right in ranges],
        "purge_gap": int(gap),
        "candidate_factor_ranks": [int(value) for value in ranks],
        "candidate_ar_orders": [int(value) for value in orders],
        "selected_factor_rank": int(selected_rank),
        "selected_ar_order": int(selected_order),
        "max_horizon": int(a.max_horizon),
        "ridge_relative": float(a.ridge_relative),
        "score_stride": int(a.score_stride),
        "template_stride": int(a.template_stride),
        "reported_selected_mse": {
            str(h): float(selected_curve[h - 1]) for h in horizons
        },
        "assumption": (
            "Theta-band amplitude envelopes lie near a stable cross-channel factor "
            "subspace, while each factor has long scalar temporal memory.  The "
            "purged blocked-cross-fitted forecast remainder is treated as innovation."
        ),
        **selection,
        **template_diagnostics,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.output,
        innovation_templates=templates.astype(np.float16),
        innovation_estimator_name=np.asarray(
            "ieeg_blocked_crossfit_longmemory_factor_residual_bootstrap"
        ),
        selected_factor_rank=np.asarray(selected_rank, dtype=np.int64),
        selected_ar_order=np.asarray(selected_order, dtype=np.int64),
    )
    a.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[out] {a.output}")
    print(f"[out] {a.output.with_suffix('.json')}")
    print(
        "[iEEG long memory] rank=%d order=%d templates=%d K=%d lag1=%.4f "
        "cv_vs_persistence=%.4f"
        % (
            selected_rank,
            selected_order,
            templates.shape[0],
            templates.shape[1],
            template_diagnostics["template_lag1_correlation"],
            selection["candidate_aggregate"][selected_key][
                "relative_to_persistence"
            ],
        )
    )


if __name__ == "__main__":
    main()
