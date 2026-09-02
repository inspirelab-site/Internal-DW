#!/usr/bin/env python3
"""Write the analytic process oracle paired with a known-SNR AR archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-horizon", type=int, default=32)
    args = parser.parse_args()

    if args.max_horizon < 1:
        raise SystemExit("--max-horizon must be positive")
    with np.load(args.data, allow_pickle=False) as archive:
        coefficients = np.asarray(archive["coefficients"], dtype=np.float64)
    if coefficients.ndim != 1 or coefficients.size == 0:
        raise SystemExit(f"coefficients must be a nonempty vector, got {coefficients.shape}")
    if not np.isfinite(coefficients).all() or np.any(np.abs(coefficients) >= 1.0):
        raise SystemExit("known-SNR coefficients must be finite with |a_j| < 1")

    coefficient_power = np.square(coefficients)
    one_step_variance = np.maximum(1.0 - coefficient_power, 0.0)
    one_step_std = np.sqrt(one_step_variance)
    horizons = np.arange(1, args.max_horizon + 1, dtype=np.int64)[:, None]
    innovation_variance = np.maximum(
        1.0 - np.power(coefficient_power[None, :], horizons), 0.0
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        oracle_ar_coefficients=coefficients,
        oracle_one_step_innovation_std=one_step_std,
        innovation_variance=innovation_variance,
        horizons=np.arange(1, args.max_horizon + 1, dtype=np.int64),
    )
    metadata = {
        "format_version": 1,
        "estimator": "analytic_diagonal_ar1_oracle",
        "data": str(args.data.resolve()),
        "output": str(args.output.resolve()),
        "max_horizon": args.max_horizon,
        "state_dim": int(coefficients.size),
        "definition": "x[t+1,j]=a[j]x[t,j]+eps[t+1,j], Var(eps_j)=1-a[j]^2",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[known-SNR oracle] coefficients={coefficients.tolist()}")
    print(f"[out] {args.output}")


if __name__ == "__main__":
    main()
