#!/usr/bin/env python3
"""Generate driven Mackey--Glass splits for the drive--history probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from internal_dw.datasets.synthetic_memory import generate_driven_mackey_glass  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--tau", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--solver-dt", type=float, default=0.1)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--trajectories", type=int, default=40)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--n-exp", type=float, default=10.0)
    parser.add_argument("--transient", type=int, default=1000)
    parser.add_argument("--drive-scale", type=float, required=True)
    parser.add_argument("--drive-rho", type=float, default=0.9)
    parser.add_argument("--generator-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    args = parser.parse_args()

    states, drives = generate_driven_mackey_glass(
        n_traj=args.trajectories,
        D=args.dim,
        tau=args.tau,
        dt=args.dt,
        solver_dt=args.solver_dt,
        n_samples=args.length,
        beta=args.beta,
        gamma=args.gamma,
        n_exp=args.n_exp,
        transient=args.transient,
        seed=args.generator_seed,
        drive_scale=args.drive_scale,
        drive_rho=args.drive_rho,
    )
    rng = np.random.default_rng(int(args.split_seed))
    order = rng.permutation(int(args.trajectories))
    train_count = max(1, int(round(args.trajectories * args.train_ratio)))
    validation_count = max(1, int(round(args.trajectories * args.validation_ratio)))
    if train_count + validation_count >= args.trajectories:
        train_count, validation_count = max(1, args.trajectories - 2), 1
    split_indices = {
        "train": order[:train_count],
        "validation": order[train_count : train_count + validation_count],
        "test": order[train_count + validation_count :],
    }
    if len(split_indices["test"]) == 0:
        split_indices["test"] = split_indices["validation"]

    metadata = {
        "system": "driven_mackey_glass",
        "equation": "dx/dt=beta*x(t-tau)/(1+x(t-tau)^n)-gamma*x(t)+drive_scale*u(t)",
        "drive": "observed stationary unit-variance AR(1), piecewise constant per AR step",
        "parameters": {
            "dim": args.dim,
            "tau": args.tau,
            "dt": args.dt,
            "solver_dt": args.solver_dt,
            "length": args.length,
            "trajectories": args.trajectories,
            "beta": args.beta,
            "gamma": args.gamma,
            "n_exp": args.n_exp,
            "transient": args.transient,
            "drive_scale": args.drive_scale,
            "drive_rho": args.drive_rho,
            "generator_seed": args.generator_seed,
            "split_seed": args.split_seed,
        },
        "split_indices": {
            name: indices.tolist() for name, indices in split_indices.items()
        },
    }
    arrays = {"metadata_json": np.asarray(json.dumps(metadata))}
    for split, indices in split_indices.items():
        arrays[f"{split}_state"] = np.ascontiguousarray(states[indices])
        arrays[f"{split}_drive"] = np.ascontiguousarray(drives[indices])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(
        f"[out] {args.output} states={states.shape} drive_scale={args.drive_scale} "
        f"drive_rho={args.drive_rho}"
    )


if __name__ == "__main__":
    main()
