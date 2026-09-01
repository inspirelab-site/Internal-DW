#!/usr/bin/env python
"""Frozen real-data mechanism audit for internal/Jacobian DW-Generic.

At one trained Internal-DW checkpoint, compare three backward-only arms:

``open``
    Set every used internal route pair to (1, 1).
``learned``
    Keep the checkpoint's learned route pairs.
``permuted``
    Permute complete (alpha, m) pairs across horizons within each layer.

All arms use identical model parameters and forward predictions.  A direction
computed on calibration example A is *always* evaluated against the ordinary
fully-open forecasting gradient/loss on disjoint example B.  Thus an arm never
defines its own evaluation target.  The primary statistics are first-order
held-out utilities.  Selected horizons also receive reversible parameter
steps with the same Euclidean update norm for all arms.

This is a frozen-checkpoint probe: it does not train, update an optimizer, or
estimate innovation SNR.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from internal_dw.datasets.registry import build_dataloaders  # noqa: E402
from internal_dw.models.registry import build_model  # noqa: E402
from internal_dw.utils import load_checkpoint, unwrap_model  # noqa: E402
from gradient_probe_ops import (  # noqa: E402
    GradientProjection,
    infer_thewell_shape as _infer_thewell_shape,
    namespace as _namespace,
)
from probe_heldout_delayed_gradient_utility import (  # noqa: E402
    _apply_step,
    _cosine,
    _full_local_projected,
    _full_local_tensors_at_horizon,
    _parameter_norm,
    _parse_float_list,
    _parse_horizons,
    _projected_dot,
    _projected_norm,
    _rollout_losses,
    _strict_candidate_pool,
    _summary,
    _tensor_dot,
    _tensor_norm,
)


ARMS = ("open", "learned", "permuted")


def _checkpoint_state(checkpoint: dict) -> dict:
    for key in ("model", "state_dict", "model_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    raise SystemExit("checkpoint does not contain a model state dictionary")


def _validate_internal_checkpoint(checkpoint: dict) -> None:
    keys = {str(key).removeprefix("module.") for key in _checkpoint_state(checkpoint)}
    if any("global_horizon_wiener" in key for key in keys):
        raise SystemExit("outer global-horizon checkpoint rejected")
    if "dual_wiener.coefficients" not in keys:
        raise SystemExit("checkpoint lacks internal dual_wiener.coefficients")


def _permuted(weights: torch.Tensor, K: int, seed: int) -> torch.Tensor:
    output = weights.detach().clone()
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 161803)
    for layer in range(int(output.shape[1])):
        order = torch.randperm(int(K), generator=generator)
        output[:K, layer] = weights[:K, layer].index_select(0, order)
    return output


def _make_arms(weights: torch.Tensor, K: int, seed: int) -> dict[str, torch.Tensor]:
    if weights.ndim != 3 or int(weights.shape[-1]) != 2:
        raise RuntimeError(
            "internal coefficients must have shape [H,depth,2], got "
            f"{tuple(weights.shape)}"
        )
    if int(K) > int(weights.shape[0]):
        raise RuntimeError(f"K={K} exceeds coefficient horizon {weights.shape[0]}")
    learned = weights.detach().clone().cpu()
    if not torch.isfinite(learned[:K]).all():
        raise RuntimeError("used learned coefficients contain non-finite values")
    if not bool(((learned[:K] >= 0.0) & (learned[:K] <= 1.0)).all()):
        raise RuntimeError("used learned coefficients are outside [0,1]")
    opened = learned.clone()
    opened[:K].fill_(1.0)
    return {
        "open": opened,
        "learned": learned,
        "permuted": _permuted(learned, K, seed),
    }


def _gain_summary(weights: torch.Tensor, K: int) -> dict[str, float]:
    used = weights[:K].double()
    return {
        "alpha_mean": float(used[..., 0].mean()),
        "alpha_min": float(used[..., 0].min()),
        "alpha_max": float(used[..., 0].max()),
        "m_mean": float(used[..., 1].mean()),
        "m_min": float(used[..., 1].min()),
        "m_max": float(used[..., 1].max()),
    }


def _router_snapshot(raw, dw) -> dict[str, Any]:
    names = (
        "resgrad_routing", "resgrad_policy", "resgrad_outer",
        "resgrad_block_gate", "resgrad_current_horizon",
        "resgrad_current_total_horizon",
    )
    return {
        "raw": {name: getattr(raw, name) for name in names if hasattr(raw, name)},
        "coefficients": dw.coefficients.detach().clone(),
        "collecting": bool(dw._collecting),
        "mode": str(dw._mode),
    }


def _restore_router(raw, dw, saved: dict[str, Any]) -> None:
    for name, value in saved["raw"].items():
        setattr(raw, name, value)
    with torch.no_grad():
        dw.coefficients.copy_(saved["coefficients"])
    dw._collecting = saved["collecting"]
    dw._mode = saved["mode"]


def _set_arm(raw, dw, weights: torch.Tensor) -> None:
    if dw.const_gain is not None:
        raise RuntimeError("DUAL_WIENER_CONST is active; learned Generic-DW is required")
    if hasattr(raw, "resgrad_routing"):
        raw.resgrad_routing = True
    if hasattr(raw, "resgrad_policy"):
        raw.resgrad_policy = "dualwiener"
    if hasattr(raw, "resgrad_outer"):
        raw.resgrad_outer = False
    if hasattr(raw, "set_resgrad_context"):
        raw.set_resgrad_context(None, None)
    dw._collecting = False
    dw._mode = "apply"
    with torch.no_grad():
        dw.coefficients.copy_(
            weights.to(device=dw.coefficients.device, dtype=dw.coefficients.dtype)
        )


def _arm_metrics(full_a, local_a, full_b_open, window_b_open, projection):
    delayed_a = [x - y for x, y in zip(full_a, local_a)]
    full_norm = np.asarray([_projected_norm(x, projection) for x in full_a])
    delayed_norm = np.asarray([_projected_norm(x, projection) for x in delayed_a])
    full_matched = np.asarray([
        _projected_dot(x, y, projection) / max(_projected_norm(x, projection), 1e-30)
        for x, y in zip(full_a, full_b_open)
    ])
    full_window = np.asarray([
        _projected_dot(x, window_b_open, projection)
        / max(_projected_norm(x, projection), 1e-30)
        for x in full_a
    ])
    delayed_matched = np.asarray([
        _projected_dot(x, y, projection) / max(_projected_norm(x, projection), 1e-30)
        for x, y in zip(delayed_a, full_b_open)
    ])
    delayed_window = np.asarray([
        _projected_dot(x, window_b_open, projection)
        / max(_projected_norm(x, projection), 1e-30)
        for x in delayed_a
    ])
    return {
        "full_norm_A": full_norm,
        "delayed_norm_A": delayed_norm,
        "delayed_fraction_A": delayed_norm / np.maximum(full_norm, 1e-30),
        "full_amplitude_H1": full_norm / max(float(full_norm[0]), 1e-30),
        "full_matched_utility_to_open_target": full_matched,
        "full_window_utility_to_open_target": full_window,
        "delayed_matched_utility_to_open_target": delayed_matched,
        "delayed_window_utility_to_open_target": delayed_window,
        "full_cosine_to_open_target": np.asarray([
            _cosine(x, y) for x, y in zip(full_a, full_b_open)
        ]),
        "delayed_cosine_to_open_target": np.asarray([
            _cosine(x, y) for x, y in zip(delayed_a, full_b_open)
        ]),
    }


def _contrast_rows(left: dict[str, np.ndarray], right: dict[str, np.ndarray]):
    epsilon = 1e-30
    return {
        "full_norm_ratio": left["full_norm_A"] / np.maximum(right["full_norm_A"], epsilon),
        "delayed_norm_ratio": left["delayed_norm_A"] / np.maximum(right["delayed_norm_A"], epsilon),
        "full_matched_utility_delta": (
            left["full_matched_utility_to_open_target"]
            - right["full_matched_utility_to_open_target"]
        ),
        "full_window_utility_delta": (
            left["full_window_utility_to_open_target"]
            - right["full_window_utility_to_open_target"]
        ),
        "delayed_matched_utility_delta": (
            left["delayed_matched_utility_to_open_target"]
            - right["delayed_matched_utility_to_open_target"]
        ),
        "delayed_window_utility_delta": (
            left["delayed_window_utility_to_open_target"]
            - right["delayed_window_utility_to_open_target"]
        ),
    }


def _snapshot_parameters(parameters: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


def _restore_parameters(parameters, values) -> None:
    with torch.no_grad():
        for parameter, value in zip(parameters, values):
            parameter.copy_(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--num_pairs", type=int, default=8)
    parser.add_argument("--finite_pairs", type=int, default=1)
    parser.add_argument("--finite_horizons", default="1,8,16,32")
    parser.add_argument("--relative_radii", default="2.5e-7,5e-7,1e-6")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--calibration_split", choices=("train", "val"), default="train")
    parser.add_argument("--evaluation_split", choices=("val", "test"), default="test")
    parser.add_argument("--loss_type", choices=("auto", "rel_l2", "mse"), default="auto")
    parser.add_argument("--coord_subsample", type=int, default=250_000)
    parser.add_argument("--start_fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    cli = parser.parse_args()

    checkpoint = torch.load(cli.ckpt, map_location="cpu", weights_only=False)
    _validate_internal_checkpoint(checkpoint)
    args = _namespace(checkpoint["args"])
    args.num_workers = 0
    args.local_batch_size = 1
    train_loader, val_loader, test_loader = build_dataloaders(args, rank=0, world_size=1)
    _infer_thewell_shape(args, train_loader)
    model = build_model(args, rank=0)
    device = torch.device(f"cuda:{int(cli.gpu)}" if torch.cuda.is_available() else "cpu")
    load_checkpoint(model, cli.ckpt, map_location=str(device), strict=True)
    raw = unwrap_model(model).to(device).eval()
    dw = getattr(raw, "dual_wiener", None)
    if dw is None:
        raise RuntimeError("loaded model has no internal DualWienerController")
    if getattr(raw, "global_horizon_wiener", None) is not None:
        raise RuntimeError("outer global-horizon controller rejected")
    recurrent = bool(getattr(raw, "is_recurrent_state_ar", False))
    parameters = [parameter for parameter in raw.parameters() if parameter.requires_grad]
    projection = GradientProjection(parameters, cli.coord_subsample, cli.seed)

    if cli.loss_type == "auto":
        candidate = (
            getattr(args, "mamba_loss_type", None)
            if recurrent else getattr(args, "bptt_loss_type", None)
        )
        loss_type = "rel_l2" if str(candidate).lower() == "rel_l2" else "mse"
    else:
        loss_type = cli.loss_type

    calibration_loader = train_loader if cli.calibration_split == "train" else val_loader
    evaluation_loader = val_loader if cli.evaluation_split == "val" else test_loader
    calibration_pool = _strict_candidate_pool(
        calibration_loader, args, recurrent, cli.K, cli.num_pairs,
        cli.seed, cli.start_fraction,
    )
    evaluation_pool = _strict_candidate_pool(
        evaluation_loader, args, recurrent, cli.K, cli.num_pairs,
        cli.seed + 1, cli.start_fraction,
    )
    num_pairs = min(cli.num_pairs, len(calibration_pool), len(evaluation_pool))
    if num_pairs < 1:
        raise RuntimeError("no feasible disjoint calibration/evaluation pair")
    pairs = list(zip(calibration_pool[:num_pairs], evaluation_pool[:num_pairs]))
    finite_horizons = _parse_horizons(cli.finite_horizons, cli.K)
    relative_radii = _parse_float_list(cli.relative_radii)

    saved_router = _router_snapshot(raw, dw)
    arm_weights = _make_arms(saved_router["coefficients"].cpu(), cli.K, cli.seed)
    arm_rows: dict[str, dict[str, list[np.ndarray]]] = {name: {} for name in ARMS}
    pair_records = []
    forward_max_diff = 0.0
    try:
        with torch.enable_grad():
            for pair_index, (calibration, evaluation) in enumerate(pairs):
                print(
                    f"  pair {pair_index + 1}/{num_pairs}: "
                    f"A=batch{calibration['batch_index']}@{calibration['start']} "
                    f"B=batch{evaluation['batch_index']}@{evaluation['start']}",
                    flush=True,
                )
                _set_arm(raw, dw, arm_weights["open"])
                full_b_open, local_b_open, loss_b_open, local_loss_b_open, fwd_b = (
                    _full_local_projected(
                        raw, parameters, projection, evaluation, cli.K, args,
                        recurrent, loss_type, device,
                    )
                )
                window_b_open = sum(full_b_open) / float(cli.K)
                pair_arm_metrics = {}
                open_loss_a = None
                open_local_loss_a = None
                for arm_name in ARMS:
                    _set_arm(raw, dw, arm_weights[arm_name])
                    full_a, local_a, loss_a, local_loss_a, fwd_a = _full_local_projected(
                        raw, parameters, projection, calibration, cli.K, args,
                        recurrent, loss_type, device,
                    )
                    if arm_name == "open":
                        open_loss_a = np.asarray(loss_a, dtype=np.float64)
                        open_local_loss_a = np.asarray(local_loss_a, dtype=np.float64)
                    else:
                        forward_max_diff = max(
                            forward_max_diff,
                            float(np.max(np.abs(np.asarray(loss_a) - open_loss_a))),
                            float(np.max(np.abs(np.asarray(local_loss_a) - open_local_loss_a))),
                        )
                    forward_max_diff = max(forward_max_diff, fwd_a, fwd_b)
                    metrics = _arm_metrics(
                        full_a, local_a, full_b_open, window_b_open, projection
                    )
                    pair_arm_metrics[arm_name] = metrics
                    for key, value in metrics.items():
                        arm_rows[arm_name].setdefault(key, []).append(value)
                    del full_a, local_a

                pair_records.append(
                    {
                        "pair_index": pair_index,
                        "A": {key: calibration[key] for key in ("batch_index", "start", "metadata")},
                        "B": {key: evaluation[key] for key in ("batch_index", "start", "metadata")},
                        "open_target_loss_B": loss_b_open,
                        "open_target_local_loss_B": local_loss_b_open,
                        "metrics_by_arm": {
                            name: {key: value.tolist() for key, value in metrics.items()}
                            for name, metrics in pair_arm_metrics.items()
                        },
                    }
                )
                del full_b_open, local_b_open, pair_arm_metrics
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        stacked = {
            arm: {key: np.stack(values) for key, values in rows.items()}
            for arm, rows in arm_rows.items()
        }
        contrast_arrays: dict[str, dict[str, np.ndarray]] = {}
        for contrast_name, left, right in (
            ("learned_minus_open", "learned", "open"),
            ("learned_minus_permuted", "learned", "permuted"),
        ):
            rows = [
                _contrast_rows(
                    {key: stacked[left][key][index] for key in stacked[left]},
                    {key: stacked[right][key][index] for key in stacked[right]},
                )
                for index in range(num_pairs)
            ]
            contrast_arrays[contrast_name] = {
                key: np.stack([row[key] for row in rows]) for key in rows[0]
            }

        finite_records = []
        baseline_parameters = _snapshot_parameters(parameters)
        model_parameter_norm = _parameter_norm(parameters)
        for pair_index, (calibration, evaluation) in enumerate(pairs[: cli.finite_pairs]):
            _set_arm(raw, dw, arm_weights["open"])
            baseline_loss = _rollout_losses(
                raw, evaluation, cli.K, args, recurrent, loss_type, device
            )
            for horizon in finite_horizons:
                if horizon == 1:
                    continue
                _set_arm(raw, dw, arm_weights["open"])
                with torch.enable_grad():
                    full_b_open, _, _, fwd_b = _full_local_tensors_at_horizon(
                        raw, parameters, evaluation, horizon, cli.K, args,
                        recurrent, loss_type, device,
                    )
                for arm_name in ARMS:
                    _restore_parameters(parameters, baseline_parameters)
                    _set_arm(raw, dw, arm_weights[arm_name])
                    with torch.enable_grad():
                        full_a, local_a, delayed_a, fwd_a = _full_local_tensors_at_horizon(
                            raw, parameters, calibration, horizon, cli.K, args,
                            recurrent, loss_type, device,
                        )
                    direction_norm = _tensor_norm(delayed_a)
                    if direction_norm <= 1e-30:
                        continue
                    unit_direction = [value / direction_norm for value in delayed_a]
                    predicted = _tensor_dot(unit_direction, full_b_open)
                    forward_max_diff = max(forward_max_diff, fwd_a, fwd_b)
                    for relative_radius in relative_radii:
                        _restore_parameters(parameters, baseline_parameters)
                        radius = float(relative_radius) * model_parameter_norm
                        _apply_step(parameters, unit_direction, radius)
                        _set_arm(raw, dw, arm_weights["open"])
                        after = _rollout_losses(
                            raw, evaluation, cli.K, args, recurrent, loss_type, device
                        )
                        finite_records.append(
                            {
                                "pair_index": pair_index,
                                "horizon": horizon,
                                "arm": arm_name,
                                "relative_radius": float(relative_radius),
                                "radius": radius,
                                "direction": "arm_temporal_delayed_full_minus_local",
                                "direction_norm_before_normalization": direction_norm,
                                "predicted_matched_utility_to_open_target": predicted,
                                "actual_matched_utility": float(
                                    (baseline_loss[horizon - 1] - after[horizon - 1])
                                    / max(radius, 1e-30)
                                ),
                                "actual_window_utility": float(
                                    (baseline_loss.mean() - after.mean())
                                    / max(radius, 1e-30)
                                ),
                                "matched_loss_before": float(baseline_loss[horizon - 1]),
                                "matched_loss_after": float(after[horizon - 1]),
                                "window_loss_before": float(baseline_loss.mean()),
                                "window_loss_after": float(after.mean()),
                            }
                        )
                    del full_a, local_a, delayed_a, unit_direction
                del full_b_open
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        _restore_parameters(parameters, baseline_parameters)

    finally:
        _restore_router(raw, dw, saved_router)

    arm_output = {}
    for arm_name in ARMS:
        arm_output[arm_name] = {
            "gain_summary": _gain_summary(arm_weights[arm_name], cli.K),
            "summary": {key: _summary(value) for key, value in stacked[arm_name].items()},
            "fraction_positive": {
                key: np.mean(value > 0.0, axis=0).tolist()
                for key, value in stacked[arm_name].items() if "utility" in key
            },
        }
    contrast_output = {
        name: {
            "summary": {key: _summary(value) for key, value in arrays.items()},
            "fraction_learned_better": {
                key: np.mean(value > 0.0, axis=0).tolist()
                for key, value in arrays.items() if key.endswith("_delta")
            },
        }
        for name, arrays in contrast_arrays.items()
    }

    output = {
        "format_version": 1,
        "semantics": {
            "scope": "frozen DW-Generic checkpoint; no training and no SNR claim",
            "arms": "open, checkpoint learned, and same-layer horizon-permuted internal Jacobian gains",
            "evaluation_target": "ordinary fully-open heldout forecasting gradient/loss for every arm",
            "delayed_component": "arm full gradient minus arm final-step local gradient",
            "finite_step": "reversible equal-Euclidean-norm SGD-shaped update for all arms",
        },
        "checkpoint": os.path.abspath(cli.ckpt),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset": str(getattr(args, "dataset", "")),
        "thewell_dataset_name": str(getattr(args, "thewell_dataset_name", "")),
        "model_name": str(getattr(args, "model_name", "")),
        "K": int(cli.K),
        "horizons": list(range(1, cli.K + 1)),
        "num_pairs": num_pairs,
        "calibration_split": cli.calibration_split,
        "evaluation_split": cli.evaluation_split,
        "shared_relative_start_fraction": float(cli.start_fraction),
        "loss_type": loss_type,
        "gradient_projection": projection.metadata(),
        "forward_arm_loss_max_diff": forward_max_diff,
        "arms": arm_output,
        "contrasts": contrast_output,
        "pair_records": pair_records,
        "finite_step_records": finite_records,
        "finite_horizons": finite_horizons,
        "relative_radii": relative_radii,
    }
    out_path = Path(cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    selected = sorted({h for h in (2, 4, 8, 16, 24, 32, 48, cli.K) if h <= cli.K})
    print("\n H  norm L/O   open Uwin  learned Uwin  permuted Uwin  learned>open")
    for horizon in selected:
        index = horizon - 1
        norm_ratio = contrast_arrays["learned_minus_open"]["delayed_norm_ratio"][:, index]
        open_u = stacked["open"]["delayed_window_utility_to_open_target"][:, index]
        learned_u = stacked["learned"]["delayed_window_utility_to_open_target"][:, index]
        permuted_u = stacked["permuted"]["delayed_window_utility_to_open_target"][:, index]
        print(
            f"{horizon:>2}  {np.median(norm_ratio):>8.3f}  "
            f"{np.median(open_u):>10.4g}  {np.median(learned_u):>12.4g}  "
            f"{np.median(permuted_u):>13.4g}  "
            f"{np.mean(learned_u > open_u):>12.3f}"
        )
    print(f"forward arm loss max diff={forward_max_diff:.3e}")
    print(f"[out] {out_path}")


if __name__ == "__main__":
    main()
