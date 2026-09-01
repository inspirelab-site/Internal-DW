#!/usr/bin/env python3
"""Fit the pre-specified diagonal AR(1) innovation sampler on train only.

The generated NPZ is directly consumable by ``DualWienerController``.  The
fit never reads the analytic coefficients stored in the known-SNR archive;
those coefficients are used only for explicitly labelled diagnostics in the
sidecar JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def _split_indices(count: int, seed: int, train_ratio: float, val_ratio: float):
    order = np.random.default_rng(int(seed)).permutation(int(count))
    n_train = max(1, int(round(int(count) * float(train_ratio))))
    n_val = max(1, int(round(int(count) * float(val_ratio))))
    if n_train + n_val >= count:
        n_train, n_val = max(1, count - 2), 1
    return {
        "train": order[:n_train],
        "validation": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }


def _fit_diagonal_ar1(train: np.ndarray):
    x = np.asarray(train[:, :-1], dtype=np.float64).reshape(-1, train.shape[-1])
    y = np.asarray(train[:, 1:], dtype=np.float64).reshape(-1, train.shape[-1])
    denominator = np.sum(x * x, axis=0)
    if np.any(denominator <= np.finfo(np.float64).tiny):
        raise RuntimeError("a training coordinate has zero AR denominator")
    transition = np.sum(x * y, axis=0) / denominator
    residual = y - x * transition[None, :]
    innovation_variance = np.mean(residual * residual, axis=0)
    innovation_variance = np.maximum(
        innovation_variance, np.finfo(np.float64).tiny
    )
    if not np.all(np.isfinite(transition)) or not np.all(
        np.isfinite(innovation_variance)
    ):
        raise RuntimeError("non-finite diagonal AR(1) fit")
    if np.any(np.abs(transition) >= 1.0):
        raise RuntimeError(
            "train-only diagonal AR(1) estimate is not stationary: "
            + np.array2string(transition, precision=6)
        )
    return transition, innovation_variance, int(x.shape[0])


def _propagate(transition: np.ndarray, innovation: np.ndarray, horizon: int):
    covariance = np.zeros_like(innovation, dtype=np.float64)
    rows = []
    for _ in range(int(horizon)):
        covariance = transition @ covariance @ transition.T + innovation
        covariance = 0.5 * (covariance + covariance.T)
        rows.append(covariance.copy())
    return np.asarray(rows, dtype=np.float64)


def _relative_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(estimate) - np.asarray(truth))
        / max(float(np.linalg.norm(truth)), np.finfo(np.float64).tiny)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-horizon", type=int, default=32)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    if args.max_horizon <= 0:
        raise ValueError("--max-horizon must be positive")
    with np.load(args.data, allow_pickle=False) as archive:
        if "trajs" not in archive.files:
            raise KeyError(f"{args.data} has no 'trajs' array")
        trajectories = np.asarray(archive["trajs"], dtype=np.float64)
        # Diagnostic-only: never used by the estimator or saved sampler.
        true_coefficients = (
            np.asarray(archive["coefficients"], dtype=np.float64)
            if "coefficients" in archive.files
            else None
        )
    if trajectories.ndim != 3 or trajectories.shape[0] < 3:
        raise ValueError(
            f"known-SNR trajectories must be [N,T,D] with N>=3, got {trajectories.shape}"
        )

    splits = _split_indices(
        trajectories.shape[0], args.split_seed, args.train_ratio, args.val_ratio
    )
    estimated_a, estimated_q, transitions = _fit_diagonal_ar1(
        trajectories[splits["train"]]
    )
    transition = np.diag(estimated_a)
    innovation = np.diag(estimated_q)
    horizon_covariance = _propagate(
        transition, innovation, int(args.max_horizon)
    )

    metadata = {
        "format_version": 1,
        "estimator": "train_only_zero_intercept_coordinatewise_diagonal_ar1",
        "fit_split": "train_only",
        "data": str(args.data.resolve()),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "split_seed": int(args.split_seed),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "train_trajectory_indices": splits["train"].tolist(),
        "validation_trajectory_indices_untouched": splits["validation"].tolist(),
        "test_trajectory_indices_untouched": splits["test"].tolist(),
        "training_transitions": int(transitions),
        "estimated_ar_coefficients": estimated_a.tolist(),
        "estimated_one_step_innovation_variance": estimated_q.tolist(),
        "max_horizon": int(args.max_horizon),
        "oracle_values_used_for_fit": False,
    }
    if true_coefficients is not None:
        if true_coefficients.shape != estimated_a.shape:
            raise ValueError(
                f"diagnostic coefficients have shape {true_coefficients.shape}, "
                f"expected {estimated_a.shape}"
            )
        true_q = 1.0 - true_coefficients * true_coefficients
        metadata["diagnostic_only"] = {
            "A_relative_l2_error": _relative_error(
                estimated_a, true_coefficients
            ),
            "Q_relative_l2_error": _relative_error(estimated_q, true_q),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            innovation_variance=np.diagonal(
                horizon_covariance, axis1=1, axis2=2
            ).astype(np.float32),
            ar_transition_matrix=transition.astype(np.float32),
            one_step_innovation_covariance=innovation.astype(np.float32),
            innovation_estimator_name=np.asarray(metadata["estimator"]),
            horizon_covariance=horizon_covariance.astype(np.float32),
            calibration_transitions=np.asarray(transitions, dtype=np.int64),
        )
    os.replace(temporary, args.output)
    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print("DONE train-only diagonal AR(1) sampler")
    print(f"train/validation/test={len(splits['train'])}/{len(splits['validation'])}/{len(splits['test'])}")
    print("Ahat=" + np.array2string(estimated_a, precision=6))
    print("Qhat=" + np.array2string(estimated_q, precision=6))
    if "diagnostic_only" in metadata:
        diagnostic = metadata["diagnostic_only"]
        print(
            "diagnostic-only Aerr=%.3e Qerr=%.3e"
            % (
                diagnostic["A_relative_l2_error"],
                diagnostic["Q_relative_l2_error"],
            )
        )
    print(f"out={args.output}")


if __name__ == "__main__":
    main()
