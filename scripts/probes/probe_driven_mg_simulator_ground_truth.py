#!/usr/bin/env python3
"""Simulator-level drive--history attribution for Driven Mackey--Glass.

The forecasting probes estimate information using finite readouts.  This script
instead intervenes on the known simulator.  At an intervention time it crosses
independent dynamical histories with independent future AR(1) innovation paths,
rolls out every history--drive combination, and applies the balanced two-factor
functional-ANOVA decomposition.  Half of the interaction variance is assigned
to each factor (the two-player Shapley allocation).

No forecasting checkpoint, learned representation, validation score, or test
trajectory is read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_SCALES = (0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08)


def _parse_scales(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("--scales must contain at least one value")
    return values


def _advance_sample(
    history: np.ndarray,
    position: int,
    force: np.ndarray,
    *,
    substeps: int,
    step_size: float,
    delay_steps: float,
    beta: float,
    gamma: float,
    exponent: float,
    drive_scale: float,
) -> int:
    """Advance a vectorized ensemble by one observed autoregressive step."""

    floor_delay = int(np.floor(delay_steps))
    weight = float(delay_steps - floor_delay)
    ring_size = int(history.shape[1])
    scaled_force = float(drive_scale) * np.asarray(force, dtype=np.float64)
    for _ in range(int(substeps)):
        index0 = (position - floor_delay) % ring_size
        index1 = (position - floor_delay - 1) % ring_size
        delayed = (1.0 - weight) * history[:, index0] + weight * history[:, index1]
        current = history[:, position]

        def rhs(value: np.ndarray) -> np.ndarray:
            return (
                float(beta) * delayed / (1.0 + np.power(delayed, float(exponent)))
                - float(gamma) * value
                + scaled_force
            )

        k1 = rhs(current)
        k2 = rhs(current + 0.5 * step_size * k1)
        k3 = rhs(current + 0.5 * step_size * k2)
        k4 = rhs(current + step_size * k3)
        following = current + (step_size / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        position = (position + 1) % ring_size
        history[:, position] = following
    return position


def crossed_rollouts(
    drive_scale: float,
    *,
    history_samples: int,
    drive_samples: int,
    transient: int,
    intervention_time: int,
    horizons: tuple[int, ...],
    seed: int,
    tau: float = 30.0,
    dt: float = 1.0,
    solver_dt: float = 0.1,
    beta: float = 0.2,
    gamma: float = 0.1,
    exponent: float = 10.0,
    drive_rho: float = 0.9,
    return_factors: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return crossed outcomes ``[history, future-drive, horizon]``."""

    if history_samples < 2 or drive_samples < 2:
        raise ValueError("at least two history and two drive samples are required")
    if not horizons or min(horizons) < 1:
        raise ValueError("horizons must be positive")
    if not -1.0 < float(drive_rho) < 1.0:
        raise ValueError("drive_rho must satisfy |rho| < 1")

    substeps = max(1, int(round(float(dt) / float(solver_dt))))
    step_size = float(dt) / substeps
    delay_steps = float(tau) / step_size
    ring_size = int(np.ceil(delay_steps)) + 2
    rng = np.random.default_rng(int(seed))
    history = 1.2 + 0.02 * rng.standard_normal((int(history_samples), ring_size))
    position = ring_size - 1

    # Each row is a naturally distributed history.  The current AR(1) forcing
    # state remains part of that history at the intervention boundary.
    force = rng.standard_normal(int(history_samples))
    innovation_scale = np.sqrt(1.0 - float(drive_rho) ** 2)
    pre_steps = int(transient) + int(intervention_time)
    for sample in range(pre_steps):
        if sample > 0:
            force = (
                float(drive_rho) * force
                + innovation_scale * rng.standard_normal(int(history_samples))
            )
        position = _advance_sample(
            history,
            position,
            force,
            substeps=substeps,
            step_size=step_size,
            delay_steps=delay_steps,
            beta=beta,
            gamma=gamma,
            exponent=exponent,
            drive_scale=drive_scale,
        )

    # Canonicalize the circular delay state from oldest to current before it is
    # exposed as the matched history factor.  The boundary AR(1) forcing state
    # is appended because it is required to reconstruct a continuous future
    # forcing path from innovations.
    chronological_indices = (
        np.arange(ring_size, dtype=np.int64) + position + 1
    ) % ring_size
    history_factor = np.concatenate(
        [history[:, chronological_indices], force[:, None]], axis=1
    )

    # Cross every retained latent history with every independent future
    # innovation path.  Conditional AR continuity is preserved through the
    # history-specific boundary force value.
    crossed_history = np.repeat(history[:, None, :], int(drive_samples), axis=1)
    crossed_history = crossed_history.reshape(
        int(history_samples) * int(drive_samples), ring_size
    )
    crossed_force = np.repeat(force[:, None], int(drive_samples), axis=1).reshape(-1)
    drive_factor = rng.standard_normal((int(drive_samples), max(horizons)))
    future_innovations = np.broadcast_to(
        drive_factor[None, :, :],
        (int(history_samples), int(drive_samples), max(horizons)),
    ).reshape(int(history_samples) * int(drive_samples), max(horizons))

    horizon_to_column = {int(horizon): index for index, horizon in enumerate(horizons)}
    output = np.empty(
        (int(history_samples), int(drive_samples), len(horizons)), dtype=np.float64
    )
    for step in range(1, max(horizons) + 1):
        crossed_force = (
            float(drive_rho) * crossed_force
            + innovation_scale * future_innovations[:, step - 1]
        )
        position = _advance_sample(
            crossed_history,
            position,
            crossed_force,
            substeps=substeps,
            step_size=step_size,
            delay_steps=delay_steps,
            beta=beta,
            gamma=gamma,
            exponent=exponent,
            drive_scale=drive_scale,
        )
        if step in horizon_to_column:
            values = crossed_history[:, position].reshape(
                int(history_samples), int(drive_samples)
            )
            output[:, :, horizon_to_column[step]] = values
    if not np.isfinite(output).all():
        raise FloatingPointError(f"non-finite crossed rollout at drive_scale={drive_scale}")
    if return_factors:
        return output, history_factor, drive_factor
    return output


def variance_decomposition(outcomes: np.ndarray) -> dict[str, float]:
    """Balanced two-factor ANOVA and two-player Shapley allocation."""

    values = np.asarray(outcomes, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"expected [history,drive,horizon], got {values.shape}")
    grand = values.mean(axis=(0, 1), keepdims=True)
    history_effect = values.mean(axis=1, keepdims=True) - grand
    drive_effect = values.mean(axis=0, keepdims=True) - grand
    interaction = values - grand - history_effect - drive_effect
    history_variance = float(np.square(history_effect).mean())
    drive_variance = float(np.square(drive_effect).mean())
    interaction_variance = float(np.square(interaction).mean())
    total_variance = float(np.square(values - grand).mean())
    reconstructed = history_variance + drive_variance + interaction_variance
    if not np.isclose(total_variance, reconstructed, rtol=5e-10, atol=1e-14):
        raise AssertionError(
            f"ANOVA does not close: total={total_variance} components={reconstructed}"
        )
    denominator = max(total_variance, np.finfo(np.float64).tiny)
    history_shapley = (history_variance + 0.5 * interaction_variance) / denominator
    drive_shapley = (drive_variance + 0.5 * interaction_variance) / denominator
    return {
        "total_variance": total_variance,
        "history_main_fraction": history_variance / denominator,
        "drive_main_fraction": drive_variance / denominator,
        "interaction_fraction": interaction_variance / denominator,
        "history_shapley_fraction": history_shapley,
        "drive_shapley_fraction": drive_shapley,
        "null_history_shapley_fraction": 0.0,
        "history_shapley_beyond_null": history_shapley,
        "closure_error": abs(1.0 - history_shapley - drive_shapley),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scales", default=",".join(f"{value:.2f}" for value in DEFAULT_SCALES)
    )
    parser.add_argument("--history-samples", type=int, default=32)
    parser.add_argument("--drive-samples", type=int, default=32)
    parser.add_argument("--transient", type=int, default=1000)
    parser.add_argument("--intervention-time", type=int, default=256)
    parser.add_argument("--horizons", default="8,16,32")
    parser.add_argument("--seed", type=int, default=20270821)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "probe_outputs/driven_mg_strength_sweep_v1/simulator_ground_truth.json"
        ),
    )
    args = parser.parse_args()
    scales = _parse_scales(args.scales)
    horizons = tuple(int(item.strip()) for item in args.horizons.split(",") if item.strip())
    rows = []
    for index, scale in enumerate(scales):
        outcomes = crossed_rollouts(
            scale,
            history_samples=int(args.history_samples),
            drive_samples=int(args.drive_samples),
            transient=int(args.transient),
            intervention_time=int(args.intervention_time),
            horizons=horizons,
            seed=int(args.seed),
        )
        row = {"drive_scale": float(scale), **variance_decomposition(outcomes)}
        rows.append(row)
        print(
            f"lambda={scale:.2f} history={row['history_shapley_fraction']:.4f} "
            f"drive={row['drive_shapley_fraction']:.4f} "
            f"interaction={row['interaction_fraction']:.4f}",
            flush=True,
        )
    result = {
        "experiment": "driven_mg_simulator_crossed_shapley",
        "uses_forecasting_model": False,
        "uses_probe_readout": False,
        "uses_official_test_split": False,
        "factor_definition": {
            "history": "latent delayed simulator state plus boundary AR(1) forcing state",
            "drive": "future AR(1) innovation path, independent of history",
            "interaction_allocation": "two-player Shapley: half to each factor",
        },
        "parameters": {
            "history_samples": int(args.history_samples),
            "drive_samples": int(args.drive_samples),
            "transient": int(args.transient),
            "intervention_time": int(args.intervention_time),
            "horizons": list(horizons),
            "seed": int(args.seed),
            "tau": 30.0,
            "drive_rho": 0.9,
        },
        "points": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[out] {args.output}")


if __name__ == "__main__":
    main()
