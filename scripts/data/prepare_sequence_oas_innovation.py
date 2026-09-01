#!/usr/bin/env python3
"""Fit a training-only lagged VAR/VARX innovation process with OAS shrinkage.

This is the deployable innovation estimate used by the OAS-initialized
Dual-Wiener runs.  It does not load a forecasting checkpoint.  The state and
external-input arrays are taken from the exact same normalized training split
as ``src/main.py``.  For driven data, the lagged drive is included in the
conditional mean before the residual covariance is estimated.

The saved process is a companion-form linear Gaussian approximation.  Its
latent state preserves the selected number of lags, while an observation
matrix maps sampled innovations back to output space.  The controller then
draws one coherent innovation path across rollout horizons.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from internal_dw.datasets.ieeg import build_ieeg_splits  # noqa: E402
from internal_dw.datasets.prepared_temporal import build_prepared_temporal_splits  # noqa: E402
from internal_dw.datasets.synthetic_memory import (  # noqa: E402
    build_mackey_glass_splits,
    build_narma_splits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=(
            "mackey_glass",
            "narma",
            "ieeg",
            "prepared_temporal_autonomous",
            "prepared_temporal_driven",
        ),
    )
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-path", default="data/synthetic")
    parser.add_argument("--seed", type=int, default=0, help="training split seed")
    parser.add_argument("--max-horizon", type=int, required=True)
    parser.add_argument("--lags", type=int, default=0, help="0 selects the dataset default")
    parser.add_argument("--calibration-transitions", type=int, default=8192)
    parser.add_argument("--calibration-seed", type=int, default=2027)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--ieeg-root", default="")
    parser.add_argument(
        "--prepared-input-root",
        default="probe_inputs/temporal_candidate_regime_v1",
    )
    parser.add_argument(
        "--ignore-prepared-drive",
        action="store_true",
        help=(
            "For a prepared_temporal_driven archive, fit the matched state-only "
            "VAR reference while still using the identical fixed splits.  This "
            "is used only for a validation-side test of whether the declared "
            "calendar drive improves the innovation model."
        ),
    )
    parser.add_argument(
        "--estimator-name",
        default="training_only_lagged_varx_oas_wiener",
    )
    return parser.parse_args()


def _dataset_args(a: argparse.Namespace) -> Tuple[SimpleNamespace, int]:
    common = dict(
        seed=int(a.seed),
        train_ratio=float(a.train_ratio),
        val_ratio=float(a.val_ratio),
        data_path=str(a.data_path),
    )
    if a.dataset == "mackey_glass":
        if not a.condition.startswith("tau"):
            raise ValueError("Mackey--Glass condition must look like tau30")
        tau = float(a.condition[3:])
        common.update(
            mg_dim=8,
            mg_tau=tau,
            mg_dt=1.0,
            mg_solver_dt=0.1,
            mg_len=2048,
            mg_traj=40,
            mg_beta=0.2,
            mg_gamma=0.1,
            mg_n=10.0,
            mg_transient=1000,
            mg_seed=0,
        )
        return SimpleNamespace(**common), max(1, int(round(tau)))
    if a.dataset == "narma":
        if not a.condition.startswith("L"):
            raise ValueError("NARMA condition must look like L5")
        order = int(a.condition[1:])
        common.update(
            narma_dim=8,
            narma_order=order,
            narma_len=2048,
            narma_traj=40,
            narma_transient=200,
            narma_u_scale=0.5,
            narma_bounded=1,
            narma_seed=0,
            narma_drive=1.5,
        )
        return SimpleNamespace(**common), max(1, order)
    if a.dataset in ("prepared_temporal_autonomous", "prepared_temporal_driven"):
        prepared_path = Path(a.prepared_input_root) / f"{a.condition}.npz"
        if not prepared_path.is_file():
            raise FileNotFoundError(f"prepared temporal archive not found: {prepared_path}")
        common.update(
            prepared_temporal_npz=str(prepared_path),
            prepared_temporal_standardize=1,
        )
        # Autonomous long-memory candidates default to a moderate history;
        # driven candidates default shorter because their calendar/input terms
        # are included explicitly.  Formal runs pass --lags explicitly.
        default_lags = 32 if a.dataset == "prepared_temporal_autonomous" else 8
        return SimpleNamespace(**common), default_lags
    if a.condition not in ("theta", "delta", "alpha", "hfb"):
        raise ValueError("iEEG condition must be theta, delta, alpha, or hfb")
    common.update(
        ieeg_root=(
            a.ieeg_root
            or "data/ieeg/preprocessed_length_matched"
        ),
        ieeg_subject="P41CS",
        ieeg_task="enc",
        ieeg_contact="macro",
        ieeg_band=a.condition,
        ieeg_step_ms=20.0,
        ieeg_chunk=1024,
        ieeg_max_channels=0,
        ieeg_split_gap=256,
    )
    # Eight 20-ms samples are enough to model short-range envelope covariance
    # without turning the nuisance fit into another horizon sweep.
    return SimpleNamespace(**common), 8


def _load_splits(a: argparse.Namespace):
    data_args, default_lags = _dataset_args(a)
    if a.dataset == "mackey_glass":
        train, validation, _ = build_mackey_glass_splits(data_args)
        train_state, val_state = train.states, validation.states
        train_input = val_input = None
    elif a.dataset == "narma":
        train, validation, _ = build_narma_splits(data_args)
        train_state, val_state = train.states, validation.states
        train_input, val_input = train.inputs, validation.inputs
    elif a.dataset in ("prepared_temporal_autonomous", "prepared_temporal_driven"):
        expect_external = a.dataset == "prepared_temporal_driven"
        train, validation, _ = build_prepared_temporal_splits(
            data_args, expect_external=expect_external
        )
        train_state, val_state = train.states, validation.states
        train_input, val_input = train.drives, validation.drives
        if bool(a.ignore_prepared_drive):
            if not expect_external:
                raise ValueError(
                    "--ignore-prepared-drive requires prepared_temporal_driven"
                )
            train_input = val_input = None
    else:
        train, validation, _ = build_ieeg_splits(data_args)
        train_state, val_state = train.chunks, validation.chunks
        train_input = val_input = None
    lags = int(a.lags) if int(a.lags) > 0 else int(default_lags)
    return (
        np.asarray(train_state, dtype=np.float64),
        None if train_input is None else np.asarray(train_input, dtype=np.float64),
        np.asarray(val_state, dtype=np.float64),
        None if val_input is None else np.asarray(val_input, dtype=np.float64),
        lags,
    )


def _lagged_design(
    state: np.ndarray, drive: Optional[np.ndarray], lags: int
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    if state.ndim != 3 or state.shape[1] <= lags:
        raise ValueError(f"state must be [N,T,D] with T>lags; got {state.shape}")
    n, time, _ = state.shape
    indices = np.arange(lags - 1, time - 1)
    state_blocks = [state[:, indices - lag, :] for lag in range(lags)]
    state_design = np.concatenate(state_blocks, axis=2).reshape(len(indices) * n, -1)
    target = state[:, indices + 1, :].reshape(len(indices) * n, -1)
    drive_design = None
    if drive is not None:
        if drive.shape[:2] != state.shape[:2]:
            raise ValueError(f"drive shape {drive.shape} is incompatible with {state.shape}")
        drive_blocks = [drive[:, indices - lag, :] for lag in range(lags)]
        drive_design = np.concatenate(drive_blocks, axis=2).reshape(len(indices) * n, -1)
    return state_design, drive_design, target


def _oas_covariance(residual: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    centered = residual - residual.mean(axis=0, keepdims=True)
    n, dimension = centered.shape
    empirical = centered.T @ centered / float(max(n, 1))
    empirical = 0.5 * (empirical + empirical.T)
    trace_mean = float(np.trace(empirical)) / dimension
    alpha = float(np.mean(empirical * empirical))
    denominator = (float(n) + 1.0) * (
        alpha - (trace_mean * trace_mean) / dimension
    )
    shrinkage = 1.0 if denominator <= 0.0 else min(
        (alpha + trace_mean * trace_mean) / denominator, 1.0
    )
    covariance = (1.0 - shrinkage) * empirical
    covariance = covariance.copy()
    covariance.flat[:: dimension + 1] += shrinkage * trace_mean
    values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    floor = max(trace_mean * 1e-8, 1e-12)
    covariance = (vectors * np.maximum(values, floor)) @ vectors.T
    return covariance, float(shrinkage), empirical


def _companion(top: np.ndarray, dimension: int, lags: int) -> np.ndarray:
    latent = dimension * lags
    matrix = np.zeros((latent, latent), dtype=np.float64)
    matrix[:dimension] = top
    if lags > 1:
        matrix[dimension:, :-dimension] = np.eye(dimension * (lags - 1))
    return matrix


def _stabilize(top: np.ndarray, dimension: int, lags: int) -> Tuple[np.ndarray, float, float]:
    transition = _companion(top, dimension, lags)
    radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
    if radius < 0.999:
        return top, radius, 1.0
    lo, hi = 0.0, 1.0
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        candidate = _companion(top * mid, dimension, lags)
        candidate_radius = float(np.max(np.abs(np.linalg.eigvals(candidate))))
        if candidate_radius < 0.999:
            lo = mid
        else:
            hi = mid
    scaled = top * lo
    radius = float(np.max(np.abs(np.linalg.eigvals(_companion(scaled, dimension, lags)))))
    return scaled, radius, float(lo)


def _gaussian_nll(residual: np.ndarray, covariance: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        return float("inf")
    solved = np.linalg.solve(covariance, residual.T).T
    quadratic = np.sum(residual * solved, axis=1)
    dimension = covariance.shape[0]
    return float(
        0.5 * np.mean(quadratic + logdet + dimension * np.log(2.0 * np.pi))
        / dimension
    )


def _lag1_rms(residual: np.ndarray) -> float:
    if residual.shape[0] < 3:
        return float("nan")
    left = residual[1:] - residual[1:].mean(axis=0, keepdims=True)
    right = residual[:-1] - residual[:-1].mean(axis=0, keepdims=True)
    cross = left.T @ right / left.shape[0]
    scale_l = np.sqrt(np.mean(left * left, axis=0)).clip(min=1e-12)
    scale_r = np.sqrt(np.mean(right * right, axis=0)).clip(min=1e-12)
    correlation = cross / np.outer(scale_l, scale_r)
    return float(np.sqrt(np.mean(correlation * correlation)))


def _propagate(
    transition: np.ndarray,
    innovation: np.ndarray,
    observation: np.ndarray,
    max_horizon: int,
) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (innovation + innovation.T))
    factor_output = vectors * np.sqrt(np.maximum(values, 0.0)).reshape(1, -1)
    factor = np.zeros((transition.shape[0], factor_output.shape[1]), dtype=np.float64)
    factor[: innovation.shape[0]] = factor_output
    accumulated = np.zeros((observation.shape[0], observation.shape[0]), dtype=np.float64)
    rows = []
    for _ in range(max_horizon):
        projected = observation @ factor
        accumulated += projected @ projected.T
        rows.append(accumulated.copy())
        factor = transition @ factor
    return np.asarray(rows)


def main() -> None:
    a = parse_args()
    if a.max_horizon <= 0 or a.calibration_transitions <= 0:
        raise ValueError("max horizon and calibration transitions must be positive")
    train_state, train_drive, val_state, val_drive, lags = _load_splits(a)
    x_state, x_drive, y = _lagged_design(train_state, train_drive, lags)
    vx_state, vx_drive, vy = _lagged_design(val_state, val_drive, lags)

    available = x_state.shape[0]
    sample_count = min(int(a.calibration_transitions), available)
    rng = np.random.default_rng(int(a.calibration_seed))
    selected = rng.choice(available, size=sample_count, replace=False)
    state_fit = x_state[selected]
    drive_fit = None if x_drive is None else x_drive[selected]
    target_fit = y[selected]
    design_fit = state_fit if drive_fit is None else np.concatenate([state_fit, drive_fit], axis=1)
    design_mean = design_fit.mean(axis=0, keepdims=True)
    target_mean = target_fit.mean(axis=0, keepdims=True)
    centered_design = design_fit - design_mean
    centered_target = target_fit - target_mean
    gram = centered_design.T @ centered_design / sample_count
    scale = max(float(np.trace(gram)) / gram.shape[0], 1e-12)
    gram.flat[:: gram.shape[0] + 1] += float(a.ridge) * scale
    cross = centered_design.T @ centered_target / sample_count
    coefficients = np.linalg.solve(gram, cross)

    dimension = target_fit.shape[1]
    state_width = dimension * lags
    top = coefficients[:state_width].T
    top, spectral_radius, stability_scale = _stabilize(top, dimension, lags)
    drive_coefficients = coefficients[state_width:]

    def predict(state_design, drive_design):
        if drive_design is None:
            design = state_design
        else:
            design = np.concatenate([state_design, drive_design], axis=1)
        centered_state = design[:, :state_width] - design_mean[:, :state_width]
        prediction = target_mean + centered_state @ top.T
        if design.shape[1] > state_width:
            prediction += (
                design[:, state_width:] - design_mean[:, state_width:]
            ) @ drive_coefficients
        return prediction

    train_residual = target_fit - predict(state_fit, drive_fit)
    val_residual = vy - predict(vx_state, vx_drive)
    covariance, shrinkage, empirical = _oas_covariance(train_residual)
    transition = _companion(top, dimension, lags)
    observation = np.zeros((dimension, dimension * lags), dtype=np.float64)
    observation[:, :dimension] = np.eye(dimension)
    horizon_covariance = _propagate(
        transition, covariance, observation, int(a.max_horizon)
    )

    metadata = {
        "format_version": 1,
        "dataset": a.dataset,
        "condition": a.condition,
        "fit_split": "train_only",
        "conditional_model": "lagged_VARX" if train_drive is not None else "lagged_VAR",
        "drive_conditioned": bool(train_drive is not None),
        "prepared_drive_ignored": bool(a.ignore_prepared_drive),
        "lags": int(lags),
        "state_dim": int(dimension),
        "latent_dim": int(dimension * lags),
        "calibration_transitions": int(sample_count),
        "available_training_transitions": int(available),
        "calibration_seed": int(a.calibration_seed),
        "split_seed": int(a.seed),
        "ridge_relative": float(a.ridge),
        "oas_shrinkage": float(shrinkage),
        "stability_top_row_scale": float(stability_scale),
        "spectral_radius": float(spectral_radius),
        "heldout_oas_nll_per_coordinate": _gaussian_nll(val_residual, covariance),
        "heldout_empirical_nll_per_coordinate": _gaussian_nll(
            val_residual,
            empirical + np.eye(dimension) * max(float(np.trace(empirical)) / dimension, 1e-12) * 1e-8,
        ),
        "heldout_innovation_lag1_correlation_rms": _lag1_rms(val_residual),
        "assumption": (
            "The lagged training process is locally linear Gaussian after conditioning "
            "on the observed drive; OAS shrinks only the one-step innovation covariance."
        ),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.output,
        innovation_variance=np.diagonal(
            horizon_covariance, axis1=1, axis2=2
        ).astype(np.float32),
        horizon_covariance=horizon_covariance.astype(np.float32),
        ar_transition_matrix=transition.astype(np.float32),
        ar_observation_matrix=observation.astype(np.float32),
        one_step_innovation_covariance=covariance.astype(np.float32),
        innovation_estimator_name=np.asarray(str(a.estimator_name)),
        calibration_transitions=np.asarray(sample_count, dtype=np.int64),
        lags=np.asarray(lags, dtype=np.int64),
    )
    a.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[out] {a.output}")
    print(f"[out] {a.output.with_suffix('.json')}")
    print(
        "[OAS] dataset=%s/%s lags=%d n=%d shrink=%.5f rho=%.5f "
        "stable_scale=%.5f val_nll=%.6f lag1_rms=%.5f"
        % (
            a.dataset,
            a.condition,
            lags,
            sample_count,
            shrinkage,
            spectral_radius,
            stability_scale,
            metadata["heldout_oas_nll_per_coordinate"],
            metadata["heldout_innovation_lag1_correlation_rms"],
        )
    )


if __name__ == "__main__":
    main()
