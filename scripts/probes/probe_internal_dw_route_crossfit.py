#!/usr/bin/env python
"""Strict held-out mechanism audit for the *internal-route* Dual-Wiener gate.

This probe is deliberately narrower than the forecasting experiments.  It
freezes a trained DW-Generic checkpoint and uses the identifiable linear-
Gaussian known-SNR system to test the exact local estimand implemented at each
residual merge.  In particular it reports

1. whether the checkpoint coefficients replay the shipped 2x2 Wiener solve on
   the checkpoint's saved total/noise moments;
2. route-message risk on trajectories never used to train the checkpoint and
   on start times disjoint from those used to fit the oracle gains;
3. a same-layer horizon permutation control, which preserves the gain
   distribution but destroys its route assignment;
4. descriptive gain/SNR and learned/oracle correlations; and
5. the maximum forward difference after changing only backward coefficients.

No parameter is trained or updated.  The reported risk is the local
conditional route-message MSE solved by DW, not a full parameter-gradient
risk.  The Mamba implementation also reuses ``m`` on recurrent branch-state
inputs; those extra state routes are not part of the controller's saved 2x2
token-route moments and are therefore explicitly outside this audit.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))

from probe_known_snr_matched_routes import (  # noqa: E402
    _average_maps,
    _conditional_decomposition,
    _process_tensors,
    _route_moments,
    _sample_process,
    _solve,
    _sym,
)
from probe_reach_vjp import plan_draws  # noqa: E402
from probe_setup import add_common_args, setup  # noqa: E402
from probe_wiener_oracle import RouteCapture, covector_loss, push, rollout  # noqa: E402


MatrixMap = Dict[int, np.ndarray]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks without adding a SciPy dependency to the probe."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _correlation(left: Iterable[float], right: Iterable[float], *, ranks=False) -> float:
    left = np.asarray(list(left), dtype=np.float64).reshape(-1)
    right = np.asarray(list(right), dtype=np.float64).reshape(-1)
    mask = np.isfinite(left) & np.isfinite(right)
    left, right = left[mask], right[mask]
    if left.size < 2:
        return float("nan")
    if ranks:
        left, right = _rankdata(left), _rankdata(right)
    left, right = left - left.mean(), right - right.mean()
    denominator = math.sqrt(float(left @ left) * float(right @ right))
    return float((left @ right) / denominator) if denominator > 0.0 else float("nan")


def _matrix_from_three(vector: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            torch.stack((vector[..., 0], vector[..., 1]), dim=-1),
            torch.stack((vector[..., 1], vector[..., 2]), dim=-1),
        ),
        dim=-2,
    )


def _checkpoint_solver_replay(dw, K: int) -> Dict[str, Any]:
    coefficients = dw.coefficients[:K].detach().double().cpu()
    total = _matrix_from_three(dw.total_moments[:K].detach().double().cpu())
    noise = _matrix_from_three(dw.noise_moments[:K].detach().double().cpu())
    total_updates = dw.total_updates[:K].detach().cpu()
    noise_updates = dw.noise_updates[:K].detach().cpu()
    valid = (total_updates > 0) & (noise_updates > 0)

    solved = torch.full_like(coefficients, float("nan"))
    for horizon, layer in torch.nonzero(valid, as_tuple=False).tolist():
        solved[horizon, layer] = torch.as_tensor(
            _solve(total[horizon, layer].numpy(), noise[horizon, layer].numpy()),
            dtype=solved.dtype,
        )
    errors = (solved - coefficients).abs()[valid]
    return {
        "routes_total": int(K * dw.depth),
        "routes_with_both_saved_moments": int(valid.sum().item()),
        "coefficient_vs_replayed_solve_mean_abs": (
            float(errors.mean()) if errors.numel() else float("nan")
        ),
        "coefficient_vs_replayed_solve_max_abs": (
            float(errors.max()) if errors.numel() else float("nan")
        ),
        "routes_error_gt_1e-5": int((errors > 1e-5).sum().item()),
        "saved_total_updates_minmax": [
            int(total_updates[valid].min()) if bool(valid.any()) else 0,
            int(total_updates[valid].max()) if bool(valid.any()) else 0,
        ],
        "saved_noise_updates_minmax": [
            int(noise_updates[valid].min()) if bool(valid.any()) else 0,
            int(noise_updates[valid].max()) if bool(valid.any()) else 0,
        ],
        "note": (
            "A nonzero gap can be legitimate only when a gain trust region/freeze "
            "made coefficients lag the latest saved moments."
        ),
    }


def _split_indices(count: int, seed: int, train_ratio: float, val_ratio: float):
    order = np.random.default_rng(int(seed)).permutation(int(count))
    n_train = max(1, int(round(count * train_ratio)))
    n_val = max(1, int(round(count * val_ratio)))
    if n_train + n_val >= count:
        n_train, n_val = max(1, count - 2), 1
    train = order[:n_train]
    val = order[n_train:n_train + n_val]
    test = order[n_train + n_val:]
    if test.size == 0:
        test = val
    return {"train": train, "val": val, "test": test}


def _install_heldout_split(bundle, a, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Replace setup()'s data view by the checkpoint-matched known-SNR split.

    The known-SNR training loader uses analytic mean=0 and std=1.  Treating the
    file as Mackey--Glass and empirically standardizing it silently changes the
    checkpoint's input coordinates, so this function intentionally consumes
    the raw generated trajectories.
    """
    with np.load(a.npz, allow_pickle=True) as archive:
        raw = np.asarray(archive[a.state_key], dtype=np.float32)
        coefficients = np.asarray(archive["coefficients"], dtype=np.float64)
    if raw.ndim != 3:
        raise SystemExit(f"known-SNR state must be [B,T,D], got {raw.shape}")
    indices = _split_indices(
        raw.shape[0], a.seed, a.split_train_ratio, a.split_val_ratio
    )[split]
    selected = np.ascontiguousarray(raw[indices])
    if a.batch > 0:
        selected = selected[: min(int(a.batch), selected.shape[0])]
        indices = indices[: selected.shape[0]]
    if selected.shape[0] < 2:
        raise SystemExit("held-out route audit needs at least two trajectories")

    rows, draw_starts, axis = plan_draws(selected, a)
    bundle.xt = torch.as_tensor(selected[rows], dtype=torch.float32, device=a.device)
    bundle.ut = None
    bundle.rows_t = torch.arange(len(rows), device=a.device)
    bundle.draw_starts = draw_starts
    bundle.axis = f"{split} trajectories; {axis}"
    bundle.batch = int(len(rows))
    print(
        f"[heldout] split={split} trajectories={bundle.batch} "
        f"indices={indices.tolist()} analytic normalization mean=0 std=1",
        flush=True,
    )
    return coefficients, indices


def _measure_records(bundle, a, coefficients: np.ndarray, process) -> list[Dict[str, Any]]:
    dw = bundle.dw
    saved_mode = dw._mode
    saved_collecting = dw._collecting
    saved_slot = dw._slot
    capture = RouteCapture(dw)
    records = []
    try:
        for draw_index, starts in enumerate(bundle.draw_starts):
            # Collecting=True is essential: current route_pair() inserts the
            # _DualWienerRoute recording nodes only on collecting forwards.
            dw._mode = "total"  # makes every measured route fully open
            dw._collecting = True
            dw._slot = 0
            dw._root_refs = []
            preds, roots, targets = rollout(bundle, starts, a)
            signal, _realized_noise, _total, error = _conditional_decomposition(
                bundle, starts, preds, targets, coefficients, 0.0, 1.0, a
            )
            signal_pairs = push(
                covector_loss(preds, signal), roots, capture, retain=True
            )
            P = _route_moments(signal_pairs)
            if not P:
                raise RuntimeError(
                    "no internal routes captured; collection must be enabled "
                    "during the rollout, not only during backward"
                )

            generator = torch.Generator(device="cpu").manual_seed(
                int(a.seed) + 100_003 * (draw_index + 1)
            )
            noise_maps = []
            batch, dimension = int(preds[0].shape[0]), int(preds[0].shape[1])
            for noise_index in range(int(a.noise_draws)):
                z = torch.randn(
                    (a.K, batch, dimension), generator=generator, dtype=torch.float32
                ).to(device=a.device, dtype=preds[0].dtype)
                vectors = _sample_process(z, *process)
                last = noise_index == int(a.noise_draws) - 1
                pairs = push(
                    covector_loss(preds, vectors), roots, capture, retain=not last
                )
                noise_maps.append(_route_moments(pairs))
            R = _average_maps(noise_maps)
            common = set(P).intersection(R)
            if not common:
                raise RuntimeError("signal/noise VJPs have no common captured routes")
            records.append({"P": P, "R": R, "start": np.asarray(starts).tolist()})
            print(
                f"[route] draw {draw_index + 1}/{len(bundle.draw_starts)} "
                f"routes={len(common)} noise_draws={a.noise_draws} "
                f"conditional_decomposition_max={error:.3e}",
                flush=True,
            )
            del preds, roots, targets, signal_pairs
    finally:
        capture.close()
        dw._mode = saved_mode
        dw._collecting = saved_collecting
        dw._slot = saved_slot
        dw._root_refs = []
    return records


def _risk(P: np.ndarray, R: np.ndarray, weight: np.ndarray) -> float:
    weight = np.asarray(weight, dtype=np.float64)
    delta = weight - 1.0
    return float(delta @ P @ delta + weight @ R @ weight)


def _project_psd(matrix: np.ndarray) -> np.ndarray:
    matrix = _sym(np.asarray(matrix, dtype=np.float64))
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.maximum(values, 0.0)[None, :]) @ vectors.T


def _solve_tied(total: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Solve the same projected Wiener objective under alpha=m=c."""

    total = _project_psd(total)
    noise = _project_psd(noise)
    signal = _project_psd(total - noise)
    covariance = signal + noise
    scale = float(np.trace(covariance))
    if not np.isfinite(scale) or scale <= 1e-20:
        return np.ones(2, dtype=np.float64)
    ridge = scale * 1e-6 + 1e-20
    covariance = covariance + ridge * np.eye(2, dtype=np.float64)
    one = np.ones(2, dtype=np.float64)
    numerator = float(one @ signal @ one)
    denominator = float(one @ covariance @ one)
    coefficient = np.clip(numerator / max(denominator, 1e-300), 0.0, 1.0)
    return np.full(2, coefficient, dtype=np.float64)


def _same_layer_half_shift(routes: list[int], depth: int) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for layer in range(depth):
        members = sorted(route for route in routes if route % depth == layer)
        if len(members) < 2:
            mapping.update({route: route for route in members})
            continue
        shift = max(1, len(members) // 2)
        mapping.update(
            {route: members[(index + shift) % len(members)]
             for index, route in enumerate(members)}
        )
    return mapping


def _random_same_layer_map(routes: list[int], depth: int, rng) -> Dict[int, int]:
    mapping = {}
    for layer in range(depth):
        members = np.asarray(
            sorted(route for route in routes if route % depth == layer), dtype=np.int64
        )
        if members.size < 2:
            mapping.update({int(route): int(route) for route in members})
            continue
        shuffled = rng.permutation(members)
        mapping.update({int(left): int(right) for left, right in zip(members, shuffled)})
    return mapping


def _risk_summary(values: Dict[int, float], open_values: Dict[int, float]) -> Dict[str, Any]:
    routes = sorted(set(values).intersection(open_values))
    risk = np.asarray([values[route] for route in routes], dtype=np.float64)
    opened = np.asarray([open_values[route] for route in routes], dtype=np.float64)
    valid = np.isfinite(risk) & np.isfinite(opened) & (opened > np.finfo(np.float64).tiny)
    ratio = risk[valid] / opened[valid]
    return {
        "routes": int(valid.sum()),
        "sum_mse_over_sum_open": float(risk[valid].sum() / opened[valid].sum()),
        "median_route_mse_over_open": float(np.median(ratio)),
        "p90_route_mse_over_open": float(np.percentile(ratio, 90)),
        "route_win_fraction_vs_open": float(np.mean(ratio < 1.0)),
    }


def _gain_snr_correlations(rows: list[Dict[str, Any]], arm: str) -> Dict[str, Any]:
    weight = np.asarray([row["weights"][arm] for row in rows], dtype=np.float64)
    marginal = np.asarray([row["marginal_signal_fraction"] for row in rows])
    joint = np.asarray([row["joint_signal_fraction"] for row in rows])
    return {
        "alpha_vs_identity_marginal": {
            "pearson": _correlation(weight[:, 0], marginal[:, 0]),
            "spearman": _correlation(weight[:, 0], marginal[:, 0], ranks=True),
        },
        "m_vs_branch_marginal": {
            "pearson": _correlation(weight[:, 1], marginal[:, 1]),
            "spearman": _correlation(weight[:, 1], marginal[:, 1], ranks=True),
        },
        "mean_gain_vs_joint_signal_fraction": {
            "pearson": _correlation(weight.mean(axis=1), joint),
            "spearman": _correlation(weight.mean(axis=1), joint, ranks=True),
        },
        "caveat": (
            "Marginal SNR correlations are descriptive: the 2x2 Wiener solve "
            "couples alpha and m through off-diagonal covariance."
        ),
    }


def _heldout_summary(
    records: list[Dict[str, Any]],
    fit_draws: int,
    coefficients: np.ndarray,
    plugin_total: np.ndarray,
    plugin_noise: np.ndarray,
    depth: int,
    permutations: int,
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    fit_records, eval_records = records[:fit_draws], records[fit_draws:]
    fit_starts = {
        int(start) for record in fit_records for start in record.get("start", [])
    }
    evaluation_starts = {
        int(start) for record in eval_records for start in record.get("start", [])
    }
    overlap = sorted(fit_starts.intersection(evaluation_starts))
    if overlap:
        raise RuntimeError(
            "oracle-fit and evaluation start sets overlap: "
            + ",".join(map(str, overlap[:8]))
        )
    fit_P = _average_maps(record["P"] for record in fit_records)
    fit_R = _average_maps(record["R"] for record in fit_records)
    eval_P = _average_maps(record["P"] for record in eval_records)
    eval_R = _average_maps(record["R"] for record in eval_records)
    routes = sorted(set(fit_P).intersection(fit_R, eval_P, eval_R))
    if not routes:
        raise RuntimeError("fit/evaluation route maps have no common entries")

    learned = {
        route: np.asarray(coefficients[route // depth, route % depth], dtype=np.float64)
        for route in routes
    }
    oracle = {
        route: _solve(_sym(fit_P[route]) + _sym(fit_R[route]), _sym(fit_R[route]))
        for route in routes
    }
    tied_learned = {
        route: _solve_tied(
            plugin_total[route // depth, route % depth],
            plugin_noise[route // depth, route % depth],
        )
        for route in routes
    }
    tied_oracle = {
        route: _solve_tied(
            _sym(fit_P[route]) + _sym(fit_R[route]),
            _sym(fit_R[route]),
        )
        for route in routes
    }
    open_weights = {route: np.ones(2, dtype=np.float64) for route in routes}
    permutation_map = _same_layer_half_shift(routes, depth)
    permuted = {route: learned[permutation_map[route]] for route in routes}
    weights = {
        "open": open_weights,
        "oracle_fit": oracle,
        "learned": learned,
        "tied_learned": tied_learned,
        "tied_oracle_fit": tied_oracle,
        "permuted_same_layer": permuted,
    }

    risks: Dict[str, Dict[int, float]] = {name: {} for name in weights}
    rows = []
    one = np.ones(2, dtype=np.float64)
    for route in routes:
        P, R = _sym(eval_P[route]), _sym(eval_R[route])
        for name, arm in weights.items():
            risks[name][route] = _risk(P, R, arm[route])
        marginal_p = np.maximum(np.diag(P), 0.0)
        marginal_r = np.maximum(np.diag(R), 0.0)
        marginal_fraction = marginal_p / np.maximum(marginal_p + marginal_r, 1e-300)
        joint_p = max(float(one @ P @ one), 0.0)
        joint_r = max(float(one @ R @ one), 0.0)
        rows.append({
            "route": int(route),
            "horizon": int(route // depth + 1),
            "layer": int(route % depth),
            "P": P,
            "R": R,
            "marginal_signal_fraction": marginal_fraction,
            "joint_signal_fraction": joint_p / max(joint_p + joint_r, 1e-300),
            "joint_snr": joint_p / max(joint_r, 1e-300),
            "weights": {name: arm[route] for name, arm in weights.items()},
            "mse_over_open": {
                name: risks[name][route] / max(risks["open"][route], 1e-300)
                for name in risks
            },
        })

    methods = {
        name: _risk_summary(values, risks["open"])
        for name, values in risks.items()
    }
    rng = np.random.default_rng(int(seed) + 71_119)
    random_ratios = []
    open_sum = sum(risks["open"].values())
    for _ in range(max(int(permutations), 0)):
        mapping = _random_same_layer_map(routes, depth, rng)
        random_risk = sum(
            _risk(_sym(eval_P[route]), _sym(eval_R[route]), learned[mapping[route]])
            for route in routes
        )
        random_ratios.append(random_risk / max(open_sum, 1e-300))
    random_ratios = np.asarray(random_ratios, dtype=np.float64)

    learned_array = np.asarray([learned[route] for route in routes])
    oracle_array = np.asarray([oracle[route] for route in routes])
    tied_learned_array = np.asarray([tied_learned[route] for route in routes])[:, 0]
    tied_oracle_array = np.asarray([tied_oracle[route] for route in routes])[:, 0]
    gain_difference = np.abs(learned_array - oracle_array)
    result = {
        "fit_draws": int(len(fit_records)),
        "evaluation_draws": int(len(eval_records)),
        "fit_and_evaluation_start_sets_are_disjoint": not overlap,
        "fit_t0_values": sorted(fit_starts),
        "evaluation_t0_values": sorted(evaluation_starts),
        "risk_definition": (
            "(w-1)^T P_eval (w-1) + w^T R_eval w for the local two-token-route message"
        ),
        "primary_aggregation": "sum route MSE / sum fully-open route MSE",
        "methods": methods,
        "same_layer_half_shift": {
            "mapping": permutation_map,
            "preserves": "layer and the multiset of learned (alpha,m) pairs",
        },
        "same_layer_random_permutations": {
            "count": int(random_ratios.size),
            "mse_over_open_mean": (
                float(random_ratios.mean()) if random_ratios.size else float("nan")
            ),
            "mse_over_open_sd": (
                float(random_ratios.std(ddof=1)) if random_ratios.size > 1 else 0.0
            ),
            "mse_over_open_p05_median_p95": (
                np.percentile(random_ratios, [5, 50, 95]).tolist()
                if random_ratios.size else [float("nan")] * 3
            ),
            "fraction_no_worse_than_targeted_learned": (
                float(np.mean(random_ratios <= methods["learned"]["sum_mse_over_sum_open"]))
                if random_ratios.size else float("nan")
            ),
        },
        "gain_vs_true_route_snr": {
            "learned": _gain_snr_correlations(rows, "learned"),
            "oracle_fit": _gain_snr_correlations(rows, "oracle_fit"),
        },
        "learned_vs_oracle_gain": {
            "alpha_pearson": _correlation(learned_array[:, 0], oracle_array[:, 0]),
            "alpha_spearman": _correlation(
                learned_array[:, 0], oracle_array[:, 0], ranks=True
            ),
            "m_pearson": _correlation(learned_array[:, 1], oracle_array[:, 1]),
            "m_spearman": _correlation(
                learned_array[:, 1], oracle_array[:, 1], ranks=True
            ),
            "mean_absolute_error": float(gain_difference.mean()),
            "max_absolute_error": float(gain_difference.max()),
        },
        "tied_learned_vs_oracle_gain": {
            "pearson": _correlation(tied_learned_array, tied_oracle_array),
            "spearman": _correlation(
                tied_learned_array, tied_oracle_array, ranks=True
            ),
            "mean_absolute_error": float(
                np.mean(np.abs(tied_learned_array - tied_oracle_array))
            ),
            "max_absolute_error": float(
                np.max(np.abs(tied_learned_array - tied_oracle_array))
            ),
        },
        "routes": rows,
    }

    full_shape = coefficients.shape
    coefficient_arms = {
        "open": np.ones(full_shape, dtype=np.float64),
        "learned": np.array(coefficients, copy=True),
        "oracle_fit": np.array(coefficients, copy=True),
        "tied_learned": np.array(coefficients, copy=True),
        "tied_oracle_fit": np.array(coefficients, copy=True),
        "permuted_same_layer": np.array(coefficients, copy=True),
    }
    for route in routes:
        horizon, layer = divmod(route, depth)
        coefficient_arms["oracle_fit"][horizon, layer] = oracle[route]
        coefficient_arms["tied_learned"][horizon, layer] = tied_learned[route]
        coefficient_arms["tied_oracle_fit"][horizon, layer] = tied_oracle[route]
        coefficient_arms["permuted_same_layer"][horizon, layer] = permuted[route]
    return result, coefficient_arms


def _prediction_values(bundle, starts, a, coefficients: np.ndarray, collecting: bool):
    dw = bundle.dw
    with torch.no_grad():
        dw.coefficients.copy_(
            torch.as_tensor(coefficients, device=dw.coefficients.device,
                            dtype=dw.coefficients.dtype)
        )
    dw._mode = "apply"
    dw._collecting = bool(collecting)
    dw._slot = 0
    dw._root_refs = []
    predictions, roots, targets = rollout(bundle, starts, a)
    values = [prediction.detach().cpu().clone() for prediction in predictions]
    del predictions, roots, targets
    dw._root_refs = []
    return values


def _forward_invariance(bundle, starts, a, arms: Dict[str, np.ndarray]) -> Dict[str, Any]:
    dw = bundle.dw
    saved_coefficients = dw.coefficients.detach().clone()
    saved_mode, saved_collecting, saved_slot = dw._mode, dw._collecting, dw._slot
    output = {}
    try:
        for collecting, label in ((False, "ordinary_straight_through"),
                                  (True, "collecting_custom_autograd")):
            reference = _prediction_values(bundle, starts, a, arms["open"], collecting)
            comparisons = {}
            for name, coefficients in arms.items():
                values = _prediction_values(bundle, starts, a, coefficients, collecting)
                max_abs = max(
                    float((value - base).abs().max())
                    for value, base in zip(values, reference)
                )
                numerator = sum(float((value - base).double().square().sum())
                                for value, base in zip(values, reference))
                denominator = sum(float(base.double().square().sum())
                                  for base in reference)
                comparisons[name] = {
                    "bitwise_equal_to_open": all(
                        torch.equal(value, base) for value, base in zip(values, reference)
                    ),
                    "max_absolute_difference": max_abs,
                    "relative_l2_difference": math.sqrt(
                        numerator / max(denominator, 1e-300)
                    ),
                }
            output[label] = comparisons
    finally:
        with torch.no_grad():
            dw.coefficients.copy_(saved_coefficients)
        dw._mode = saved_mode
        dw._collecting = saved_collecting
        dw._slot = saved_slot
        dw._root_refs = []
    output["all_arms_bitwise_equal"] = all(
        item["bitwise_equal_to_open"]
        for mode in ("ordinary_straight_through", "collecting_custom_autograd")
        for item in output[mode].values()
    )
    output["scope"] = "forward predictions only; backward coefficients are changed"
    return output


def _print_summary(output: Dict[str, Any]) -> None:
    replay = output["checkpoint_solver_replay"]
    print("\n=== checkpoint solver replay ===")
    print(
        f"routes={replay['routes_with_both_saved_moments']}/"
        f"{replay['routes_total']} mean|dw|="
        f"{replay['coefficient_vs_replayed_solve_mean_abs']:.3e} "
        f"max|dw|={replay['coefficient_vs_replayed_solve_max_abs']:.3e}"
    )
    heldout = output["heldout_local_route_message"]
    print("\n=== held-out local route-message MSE / open ===")
    for name, item in heldout["methods"].items():
        print(
            f"{name:>22s} pooled={item['sum_mse_over_sum_open']:.4f} "
            f"median={item['median_route_mse_over_open']:.4f} "
            f"wins={item['route_win_fraction_vs_open']:.3f}"
        )
    random_control = heldout["same_layer_random_permutations"]
    print(
        f"same-layer random permutation median="
        f"{random_control['mse_over_open_p05_median_p95'][1]:.4f}; "
        f"fraction <= learned="
        f"{random_control['fraction_no_worse_than_targeted_learned']:.3f}"
    )
    print(
        f"forward invariant (all arms, both code paths)="
        f"{output['forward_invariance']['all_arms_bitwise_equal']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--oracle-file", required=True)
    parser.add_argument("--fit-draws", type=int, default=4)
    parser.add_argument("--noise-draws", type=int, default=32)
    parser.add_argument("--evaluation-split", choices=("val", "test"), default="test")
    parser.add_argument("--permutations", type=int, default=256)
    parser.add_argument("--out", required=True)
    a = parser.parse_args()

    if not a.npz or not a.state_key:
        raise SystemExit("this probe requires --npz and --state-key")
    if a.data_preprocess != "none":
        raise SystemExit(
            "known-SNR checkpoint uses analytic mean=0,std=1; pass --data-preprocess none"
        )
    if a.t0 >= 0:
        raise SystemExit("a fixed --t0 cannot provide disjoint fit/evaluation draws")
    if a.fit_draws < 1 or a.draws <= a.fit_draws:
        raise SystemExit("--draws must exceed --fit-draws")
    if a.noise_draws < 1:
        raise SystemExit("--noise-draws must be positive")

    torch.manual_seed(int(a.seed))
    np.random.seed(int(a.seed))
    bundle = setup(a)
    if bundle.ut is not None:
        raise SystemExit("known-SNR route audit does not accept an external stimulus")
    if bundle.dw.const_gain is not None:
        raise SystemExit("DUAL_WIENER_CONST is set; a learned Generic-DW checkpoint is required")

    coefficients, heldout_indices = _install_heldout_split(
        bundle, a, a.evaluation_split
    )
    if len(bundle.draw_starts) <= a.fit_draws:
        raise SystemExit(
            f"draw planning produced only {len(bundle.draw_starts)} unique starts; "
            f"need > fit_draws={a.fit_draws}"
        )
    with np.load(a.oracle_file, allow_pickle=True) as oracle:
        oracle_coefficients = np.asarray(
            oracle["oracle_ar_coefficients"], dtype=np.float64
        )
        oracle_std = np.asarray(
            oracle["oracle_one_step_innovation_std"], dtype=np.float64
        )
    # The dataset stores coefficients as float64 while the oracle artifact was
    # serialized through float32.  The two represent the same configured AR
    # process; use the same serialization-safe tolerance as the intervention
    # probe instead of rejecting harmless ~2e-8 roundoff.
    if not np.allclose(coefficients, oracle_coefficients, atol=1e-7, rtol=0.0):
        raise SystemExit("dataset coefficients disagree with --oracle-file")
    process = _process_tensors(
        np.diag(oracle_coefficients), np.diag(oracle_std ** 2),
        a.device, torch.float32,
    )

    replay = _checkpoint_solver_replay(bundle.dw, int(a.K))
    records = _measure_records(bundle, a, coefficients, process)
    plugin_total = _matrix_from_three(
        bundle.dw.total_moments[: int(a.K)].detach().double().cpu()
    ).numpy()
    plugin_noise = _matrix_from_three(
        bundle.dw.noise_moments[: int(a.K)].detach().double().cpu()
    ).numpy()
    heldout, coefficient_arms = _heldout_summary(
        records,
        int(a.fit_draws),
        bundle.dw.coefficients.detach().cpu().numpy(),
        plugin_total,
        plugin_noise,
        int(bundle.depth),
        int(a.permutations),
        int(a.seed),
    )
    forward = _forward_invariance(
        bundle, bundle.draw_starts[int(a.fit_draws)], a, coefficient_arms
    )

    output = {
        "format_version": 2,
        "scope": "internal Jacobian/token-route DW-Generic; frozen checkpoint",
        "checkpoint": str(Path(a.ckpt).resolve()),
        "data": str(Path(a.npz).resolve()),
        "oracle_file": str(Path(a.oracle_file).resolve()),
        "K": int(a.K),
        "depth": int(bundle.depth),
        "evaluation_split": a.evaluation_split,
        "heldout_trajectory_indices": heldout_indices,
        "batch": int(bundle.batch),
        "axis": bundle.axis,
        "analytic_normalization": {"mean": 0.0, "std": 1.0},
        "checkpoint_estimator": str(getattr(bundle.dw, "noise_model", "unknown")),
        "checkpoint_solver_replay": replay,
        "heldout_local_route_message": heldout,
        "forward_invariance": forward,
        "caveats": [
            "Risk is local route-message MSE, not full parameter-gradient MSE.",
            "Mamba branch_state_input reuses m on recurrent state routes not recorded in the 2x2 token-route moments.",
            "The oracle gains are fit on held-out trajectories at calibration t0 values and evaluated at disjoint t0 values from the same held-out trajectory split.",
        ],
    }
    _print_summary(output)
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(output), indent=2, sort_keys=True), encoding="utf-8")
    print(f"\n[out] {path.resolve()}")


if __name__ == "__main__":
    main()
