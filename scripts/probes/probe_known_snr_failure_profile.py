#!/usr/bin/env python3
"""Directly identify the delayed-credit failure mode on known-SNR AR data.

For a frozen forecasting checkpoint and one batch of observed rollout starts,
the simulator supplies both E[x[t+k] | x[t]] and independent future-noise
realizations.  With squared error this makes the conditional-mean parameter
gradient at every horizon observable.  The probe reports

  * per-horizon signal, innovation-noise, and total gradient energy;
  * their gradient SNR and noise fraction;
  * the bias/noise decomposition of every prefix gradient relative to the
    clean *full-K* BPTT target.

No parameter is updated and the checkpoint is never modified.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import known_snr_gradient_ops as gradient_ops  # noqa: E402
from probe_known_snr_diagonal_ar_gain import (  # noqa: E402
    _preflight_exact_checkpoint,
    _split_indices,
)
from probe_data_ops import plan_draws  # noqa: E402
from probe_setup import add_common_args, setup  # noqa: E402
from known_snr_route_ops import normalization as _normalization  # noqa: E402


def _install_trajectory_view(bundle, raw, indices, args, requested_batch):
    selected_indices = np.asarray(indices, dtype=np.int64)
    selected = np.ascontiguousarray(raw[selected_indices])
    if requested_batch > 0:
        selected = selected[: min(int(requested_batch), selected.shape[0])]
        selected_indices = selected_indices[: selected.shape[0]]
    if selected.shape[0] < 2:
        raise SystemExit("failure profile needs at least two trajectories")
    args.batch = int(selected.shape[0])
    rows, starts, axis = plan_draws(selected, args)
    bundle.xt = torch.as_tensor(selected[rows], dtype=torch.float32, device=args.device)
    bundle.ut = None
    bundle.rows_t = torch.arange(len(rows), device=args.device)
    bundle.draw_starts = starts
    bundle.axis = axis
    bundle.batch = int(len(rows))
    return selected_indices, axis


def _evaluate_profile(
    bundle,
    args,
    parameters: Sequence[torch.Tensor],
    starts,
    coefficients: torch.Tensor,
    innovation_std: torch.Tensor,
    normalization_mean: float,
    normalization_std: float,
) -> dict:
    conditional, noisy = gradient_ops.conditional_and_noisy_targets(
        bundle,
        starts,
        coefficients,
        innovation_std,
        normalization_mean,
        normalization_std,
        args.K,
        args.eval_noise_draws,
        args.seed + 510007,
    )
    predictions = gradient_ops.rollout_predictions(bundle, starts, args)
    clean = gradient_ops.gradient_matrix(
        gradient_ops.horizon_losses(predictions, conditional),
        parameters,
        retain_after=True,
    )
    clean_prefix = clean.cumsum(dim=0)
    clean_full = clean_prefix[-1]
    horizons = int(clean.shape[0])

    signal_energy = clean.double().square().mean(dim=1).cpu().numpy()
    total_energy = np.zeros(horizons, dtype=np.float64)
    noise_energy = np.zeros(horizons, dtype=np.float64)
    signal_noise_cross = np.zeros(horizons, dtype=np.float64)
    prefix_total_risk = np.zeros(horizons, dtype=np.float64)
    prefix_noise_risk = np.zeros(horizons, dtype=np.float64)

    for draw_index, targets in enumerate(noisy):
        noisy_matrix = gradient_ops.gradient_matrix(
            gradient_ops.horizon_losses(predictions, targets),
            parameters,
            retain_after=draw_index + 1 < len(noisy),
        )
        innovation = noisy_matrix - clean
        total_energy += noisy_matrix.double().square().mean(dim=1).cpu().numpy()
        noise_energy += innovation.double().square().mean(dim=1).cpu().numpy()
        signal_noise_cross += (
            (clean.double() * innovation.double()).mean(dim=1).cpu().numpy()
        )

        noisy_prefix = noisy_matrix.cumsum(dim=0)
        innovation_prefix = innovation.cumsum(dim=0)
        prefix_total_risk += (
            (noisy_prefix - clean_full).double().square().mean(dim=1).cpu().numpy()
        )
        prefix_noise_risk += (
            innovation_prefix.double().square().mean(dim=1).cpu().numpy()
        )

        del noisy_matrix, innovation, noisy_prefix, innovation_prefix

    count = max(len(noisy), 1)
    total_energy /= count
    noise_energy /= count
    signal_noise_cross /= count
    prefix_total_risk /= count
    prefix_noise_risk /= count
    prefix_bias_risk = (
        (clean_prefix - clean_full).double().square().mean(dim=1).cpu().numpy()
    )
    closure = prefix_total_risk - prefix_bias_risk - prefix_noise_risk

    eps = 1e-300
    gradient_snr = signal_energy / np.maximum(noise_energy, eps)
    noise_fraction = noise_energy / np.maximum(signal_energy + noise_energy, eps)
    rms_reference = math.sqrt(max(float(total_energy[0]), eps))
    total_rms_relative = np.sqrt(np.maximum(total_energy, 0.0)) / rms_reference
    signal_rms_relative = np.sqrt(np.maximum(signal_energy, 0.0)) / rms_reference
    noise_rms_relative = np.sqrt(np.maximum(noise_energy, 0.0)) / rms_reference

    prefix_best_index = int(np.argmin(prefix_total_risk))
    return {
        "eval_noise_draws": len(noisy),
        "horizon": list(range(1, horizons + 1)),
        "signal_energy": signal_energy.tolist(),
        "noise_energy": noise_energy.tolist(),
        "total_energy": total_energy.tolist(),
        "signal_noise_cross": signal_noise_cross.tolist(),
        "gradient_snr": gradient_snr.tolist(),
        "noise_fraction": noise_fraction.tolist(),
        "total_rms_relative_to_h1": total_rms_relative.tolist(),
        "signal_rms_relative_to_h1": signal_rms_relative.tolist(),
        "noise_rms_relative_to_h1": noise_rms_relative.tolist(),
        "prefix_bias_risk_to_full_clean_target": prefix_bias_risk.tolist(),
        "prefix_noise_risk": prefix_noise_risk.tolist(),
        "prefix_total_risk_to_full_clean_target": prefix_total_risk.tolist(),
        "prefix_decomposition_closure": closure.tolist(),
        "best_prefix_horizon": prefix_best_index + 1,
        "best_prefix_risk_over_exact": float(
            prefix_total_risk[prefix_best_index] / max(prefix_total_risk[-1], eps)
        ),
        "maximum_absolute_prefix_decomposition_error": float(
            np.max(np.abs(closure))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--oracle-file", required=True)
    parser.add_argument("--eval-noise-draws", type=int, default=64)
    parser.add_argument(
        "--trajectory-split",
        choices=("all", "train", "val", "test"),
        default="all",
        help="seed-matched trajectory split used by this frozen probe",
    )
    parser.add_argument(
        "--split-seed", type=int, default=None,
        help="split seed, separated from the Monte-Carlo replicate seed",
    )
    parser.add_argument(
        "--checkpoint-seed", type=int, default=None,
        help="expected training seed for --ckpt",
    )
    parser.add_argument(
        "--strict-exact-closure", action="store_true",
        help=(
            "require an Exact-BPTT checkpoint, raw known-SNR coordinates, "
            "and a non-all seed-matched trajectory split"
        ),
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.eval_noise_draws < 2:
        raise SystemExit("--eval-noise-draws must be at least two")
    args.draws = 1
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    split_seed = int(args.seed if args.split_seed is None else args.split_seed)
    checkpoint_seed = int(
        split_seed if args.checkpoint_seed is None else args.checkpoint_seed
    )
    with np.load(args.npz, allow_pickle=True) as archive:
        raw = np.asarray(archive[args.state_key], dtype=np.float32)
        dataset_coefficients = np.asarray(
            archive["coefficients"], dtype=np.float64
        )
    if raw.ndim != 3:
        raise SystemExit(f"known-SNR state must be [B,T,D], got {raw.shape}")
    if args.strict_exact_closure:
        if args.data_preprocess != "none":
            raise SystemExit(
                "strict Exact closure requires --data-preprocess none"
            )
        if args.trajectory_split == "all":
            raise SystemExit(
                "strict Exact closure requires train/val/test trajectory separation"
            )
        if not args.exact_checkpoint_adapter:
            raise SystemExit(
                "strict Exact closure requires --exact-checkpoint-adapter"
            )
        _preflight_exact_checkpoint(args.ckpt, checkpoint_seed)

    if args.data_preprocess == "none":
        # The known-SNR generator is analytically in stationary zero-mean,
        # unit-variance coordinates.  Re-estimating a scalar mean/std here
        # would put the conditional simulator in a different coordinate system
        # from the frozen checkpoint.
        normalization_mean, normalization_std = 0.0, 1.0
    else:
        normalization_mean, normalization_std, _ = _normalization(
            args.npz,
            args.state_key,
            split_seed,
            args.split_train_ratio,
            args.split_val_ratio,
        )

    requested_batch = int(args.batch)
    if args.trajectory_split != "all":
        args.batch = 0  # setup must expose all trajectories before splitting
    bundle = setup(args)
    if args.trajectory_split == "all":
        selected_indices = np.arange(bundle.batch, dtype=np.int64)
        selected_axis = bundle.axis
    else:
        splits = _split_indices(
            raw.shape[0], split_seed,
            args.split_train_ratio, args.split_val_ratio,
        )
        selected_indices, selected_axis = _install_trajectory_view(
            bundle, raw, splits[args.trajectory_split], args, requested_batch
        )
        print(
            f"[split] seed={split_seed} split={args.trajectory_split} "
            f"trajectories={bundle.batch} indices={selected_indices.tolist()} "
            "coordinates=analytic(mean=0,std=1)",
            flush=True,
        )
    coefficients, innovation_std = gradient_ops.load_process(
        args.oracle_file, normalization_std, args.device
    )
    if not np.allclose(
        dataset_coefficients,
        coefficients.detach().cpu().numpy(),
        atol=1e-6,
    ):
        raise SystemExit("dataset coefficients disagree with --oracle-file")
    _, parameters = gradient_ops.trainable_parameters(bundle.model)
    starts = bundle.draw_starts[0]
    open_coefficients = torch.ones_like(bundle.dw.coefficients)
    with gradient_ops.routing_operator(bundle.dw, open_coefficients):
        profile = _evaluate_profile(
            bundle,
            args,
            parameters,
            starts,
            coefficients,
            innovation_std,
            normalization_mean,
            normalization_std,
        )

    output = {
        "format_version": 2,
        "definition": {
            "signal": "per-horizon parameter gradient to E[x[t+k] | x[t]]",
            "noise": "realized per-horizon gradient minus that conditional-mean gradient",
            "failure_risk": "prefix-gradient MSE to the clean full-K BPTT target",
        },
        "checkpoint": str(Path(args.ckpt).resolve()),
        "data": str(Path(args.npz).resolve()),
        "oracle_file": str(Path(args.oracle_file).resolve()),
        "seed": int(args.seed),
        "protocol": {
            "strict_exact_closure": bool(args.strict_exact_closure),
            "checkpoint_role": "frozen_exact_bptt"
            if args.strict_exact_closure else "unspecified",
            "checkpoint_seed": checkpoint_seed,
            "monte_carlo_replicate_seed": int(args.seed),
            "data_preprocess": str(args.data_preprocess),
            "normalization_mean": float(normalization_mean),
            "normalization_std": float(normalization_std),
            "trajectory_split": str(args.trajectory_split),
            "split_seed": split_seed,
            "trajectory_indices": selected_indices.tolist(),
            "sampling_axis": selected_axis,
            "future_noise_draws": "independent diagonal-AR simulator draws",
        },
        "K": int(args.K),
        "batch": int(bundle.batch),
        "parameter_coordinates": sum(int(parameter.numel()) for parameter in parameters),
        "profile": profile,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print("=== known-SNR delayed-credit failure profile ===")
    print(" H   amp(total)    grad-SNR  noise-frac  prefix-risk/exact")
    exact_prefix = max(profile["prefix_total_risk_to_full_clean_target"][-1], 1e-300)
    for horizon in (1, 2, 4, 8, 16, 32):
        if horizon > args.K:
            continue
        index = horizon - 1
        print(
            f"{horizon:2d}  {profile['total_rms_relative_to_h1'][index]:11.4g}  "
            f"{profile['gradient_snr'][index]:10.4g}  "
            f"{profile['noise_fraction'][index]:10.4f}  "
            f"{profile['prefix_total_risk_to_full_clean_target'][index] / exact_prefix:17.4f}"
        )
    print(
        f"best prefix H={profile['best_prefix_horizon']} "
        f"risk/exact={profile['best_prefix_risk_over_exact']:.4f}"
    )
    print(f"[out] {path.resolve()}")


if __name__ == "__main__":
    main()
