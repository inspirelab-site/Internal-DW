#!/usr/bin/env python3
"""Estimate fMRI innovation from repeated-subject stimulus reliability.

All training subjects watch the same movie.  Their time-aligned mean is used as
the stimulus-locked conditional component; leave-one-subject-out deviations
estimate subject-specific observation/process variation.  An AR(1) model is
then fit to those deviations so the noise probe preserves fMRI temporal
autocorrelation rather than treating TRs independently.

No forecasting checkpoint or validation/test subject is used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from internal_dw.datasets.hcp import build_hcp_splits  # noqa: E402
from prepare_sequence_oas_innovation import _gaussian_nll, _lag1_rms, _oas_covariance  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--movie", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--roi-dim", type=int, default=400)
    parser.add_argument("--max-horizon", type=int, default=64)
    parser.add_argument("--ridge-relative", type=float, default=1e-3)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _collect_aligned_training_subjects(dataset) -> tuple[np.ndarray, list[str]]:
    states, subjects = [], []
    for index in range(len(dataset)):
        item = dataset[index]
        state = item["state"].numpy().astype(np.float64, copy=False)
        states.append(state)
        subjects.append(str(item["metadata"]["subject_id"]))
        print(
            f"[collect] {index + 1}/{len(dataset)} subject={subjects[-1]} "
            f"shape={state.shape}",
            flush=True,
        )
    if len(states) < 3:
        raise ValueError("shared-response innovation needs at least three training subjects")
    time = min(state.shape[0] for state in states)
    dimension = {state.shape[1] for state in states}
    if len(dimension) != 1:
        raise ValueError(f"training subjects have inconsistent ROI dimensions: {dimension}")
    return np.stack([state[:time] for state in states]), subjects


def _fit_ar1(residual: np.ndarray, ridge_relative: float, device: torch.device):
    # residual: [S,T,D], already zero-centered across subjects at each time.
    x_np = residual[:, :-1].reshape(-1, residual.shape[-1])
    y_np = residual[:, 1:].reshape(-1, residual.shape[-1])
    x = torch.from_numpy(x_np.astype(np.float32)).to(device)
    y = torch.from_numpy(y_np.astype(np.float32)).to(device)
    gram = x.T @ x / x.shape[0]
    scale = (torch.trace(gram) / gram.shape[0]).clamp_min(1e-12)
    gram.diagonal().add_(float(ridge_relative) * scale)
    transition = torch.linalg.solve(gram, x.T @ y / x.shape[0]).T
    prediction = x @ transition.T
    innovation = (y - prediction).cpu().numpy().astype(np.float64)
    return transition.cpu().numpy().astype(np.float64), innovation


def _stabilize(transition: np.ndarray) -> tuple[np.ndarray, float, float]:
    radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
    if radius < 0.999:
        return transition, radius, 1.0
    scale = 0.999 / max(radius, 1e-12)
    stable = transition * scale
    stable_radius = float(np.max(np.abs(np.linalg.eigvals(stable))))
    return stable, stable_radius, float(scale)


def _propagate(transition: np.ndarray, covariance: np.ndarray, horizon: int) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    impulse = vectors * np.sqrt(np.maximum(values, 0.0)).reshape(1, -1)
    accumulated = np.zeros_like(covariance)
    rows = []
    for _ in range(int(horizon)):
        accumulated += impulse @ impulse.T
        rows.append(np.diag(accumulated).copy())
        impulse = transition @ impulse
    return np.asarray(rows, dtype=np.float32)


def main() -> None:
    a = parse_args()
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
    state, subjects = _collect_aligned_training_subjects(train)
    # Remove subject-specific temporal baselines: the autoregressive model can
    # infer them from history, so counting them as innovation would inflate R.
    centered = state - state.mean(axis=1, keepdims=True)
    shared = centered.mean(axis=0, keepdims=True)
    count = centered.shape[0]
    loo_mean = (centered.sum(axis=0, keepdims=True) - centered) / float(count - 1)
    loo_residual = centered - loo_mean
    # Var(x_s - mean_others)=sigma^2*S/(S-1).  Correct the finite-subject
    # inflation so the covariance estimates one subject's nuisance process.
    loo_residual *= np.sqrt(float(count - 1) / float(count))

    transition, raw_innovation = _fit_ar1(
        loo_residual, float(a.ridge_relative), torch.device(a.device)
    )
    transition, radius, stability_scale = _stabilize(transition)
    # Recompute innovations after any stability correction.
    x = loo_residual[:, :-1].reshape(-1, loo_residual.shape[-1])
    y = loo_residual[:, 1:].reshape(-1, loo_residual.shape[-1])
    innovation = y - x @ transition.T
    covariance, shrinkage, empirical = _oas_covariance(innovation)
    variance = _propagate(transition, covariance, int(a.max_horizon))

    metadata = {
        "format_version": 1,
        "dataset": "hcp_movie",
        "movie": int(a.movie),
        "fit_split": "train_subjects_only",
        "conditional_model": "time_aligned_leave_one_subject_out_shared_response",
        "nuisance_process": "subject_deviation_AR1_OAS",
        "subjects": subjects,
        "subject_count": int(count),
        "time_points": int(state.shape[1]),
        "state_dim": int(state.shape[2]),
        "ridge_relative": float(a.ridge_relative),
        "oas_shrinkage": float(shrinkage),
        "spectral_radius": float(radius),
        "stability_scale": float(stability_scale),
        "innovation_lag1_correlation_rms": _lag1_rms(innovation),
        "innovation_oas_nll_per_coordinate": _gaussian_nll(innovation, covariance),
        "innovation_empirical_nll_per_coordinate": _gaussian_nll(
            innovation,
            empirical
            + np.eye(empirical.shape[0])
            * max(float(np.trace(empirical)) / empirical.shape[0], 1e-12)
            * 1e-8,
        ),
        "shared_variance_fraction": float(
            np.mean(shared * shared)
            / max(float(np.mean(centered * centered)), 1e-30)
        ),
        "assumption": (
            "Time-aligned variation shared across training subjects is stimulus-locked "
            "signal; finite-subject-corrected deviations form an AR(1) nuisance process."
        ),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.output,
        innovation_variance=variance,
        ar_transition_matrix=transition.astype(np.float32),
        one_step_innovation_covariance=covariance.astype(np.float32),
        innovation_estimator_name=np.asarray("hcp_shared_response_ar1_oas_wiener"),
    )
    a.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[out] {a.output}")
    print(f"[out] {a.output.with_suffix('.json')}")
    print(
        "[HCP shared response] subjects=%d T=%d shared_fraction=%.5f "
        "shrink=%.5f rho=%.5f lag1=%.5f"
        % (
            count,
            state.shape[1],
            metadata["shared_variance_fraction"],
            shrinkage,
            radius,
            metadata["innovation_lag1_correlation_rms"],
        )
    )


if __name__ == "__main__":
    main()
