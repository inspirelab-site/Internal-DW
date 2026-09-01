#!/usr/bin/env python
"""Matched route-moment audit on the identifiable known-SNR AR system.

This probe freezes one checkpoint and one set of fully-open rollout graphs.  It
measures the total route covariance ``T`` once, then changes only the estimate
of the route-noise covariance ``R``.  The synthetic AR process makes the
conditional mean and the realized future innovations observable, so the same
graphs also yield empirical oracle ``P`` and ``R``.

The comparison separates three failure sources that training curves confound:

* residual-noise identification (different output-space innovation models);
* finite-sample incompatibility of ``T`` and ``R`` (negative ``T-R`` modes);
* the online controller/EMA (not used here).

No parameter is trained and no checkpoint is modified.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable, Tuple

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from probe_setup import add_common_args, setup  # noqa: E402
from probe_wiener_oracle import (  # noqa: E402
    RouteCapture,
    covector_loss,
    push,
    risk_analytic,
    rollout,
)
from internal_dw.models.dual_wiener import solve_box_wiener_2x2  # noqa: E402


MatrixMap = Dict[int, np.ndarray]
PairMap = Dict[int, Tuple[torch.Tensor, torch.Tensor]]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _sym(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return 0.5 * (matrix + matrix.T)


def _moment(pair: Tuple[torch.Tensor, torch.Tensor], rows=None) -> np.ndarray:
    identity, branch = pair
    if rows is not None:
        index = torch.as_tensor(rows, device=identity.device, dtype=torch.long)
        identity = identity.index_select(0, index)
        branch = branch.index_select(0, index)
    vec = torch.stack((identity, branch)).reshape(2, -1)
    return ((vec @ vec.t()) / max(int(vec.shape[1]), 1)).detach().cpu().numpy()


def _cross(
    left: Tuple[torch.Tensor, torch.Tensor],
    right: Tuple[torch.Tensor, torch.Tensor],
) -> np.ndarray:
    lhs = torch.stack(left).reshape(2, -1)
    rhs = torch.stack(right).reshape(2, -1)
    return ((lhs @ rhs.t()) / max(int(lhs.shape[1]), 1)).detach().cpu().numpy()


def _route_moments(pairs: PairMap, rows=None) -> MatrixMap:
    return {int(route): _moment(pair, rows=rows) for route, pair in pairs.items()}


def _route_cross(left: PairMap, right: PairMap) -> MatrixMap:
    return {
        int(route): _cross(left[route], right[route])
        for route in sorted(set(left).intersection(right))
    }


def _average_maps(maps: Iterable[MatrixMap]) -> MatrixMap:
    total: Dict[int, np.ndarray] = {}
    count: Dict[int, int] = defaultdict(int)
    for matrix_map in maps:
        for route, matrix in matrix_map.items():
            if route not in total:
                total[route] = np.zeros((2, 2), dtype=np.float64)
            total[route] += np.asarray(matrix, dtype=np.float64)
            count[route] += 1
    return {route: value / max(count[route], 1) for route, value in total.items()}


def _geometric_median_maps(maps: Iterable[MatrixMap], iterations: int = 50) -> MatrixMap:
    """Robust convex aggregation of per-draw 2x2 covariance matrices.

    The Weiszfeld iterate is a convex combination of the input covariance
    matrices, so PSD inputs remain PSD.  It is used only as a frozen-probe
    candidate here; the training controller is not changed by this script.
    """
    maps = list(maps)
    routes = sorted(set.intersection(*(set(matrix_map) for matrix_map in maps)))
    output = {}
    for route in routes:
        samples = np.stack([_sym(matrix_map[route]) for matrix_map in maps])
        center = samples.mean(axis=0)
        for _ in range(iterations):
            distances = np.linalg.norm(samples - center[None, :, :], axis=(1, 2))
            closest = int(np.argmin(distances))
            if distances[closest] < 1e-15:
                center = samples[closest]
                break
            weights = 1.0 / np.maximum(distances, 1e-15)
            next_center = np.einsum("n,nij->ij", weights, samples) / weights.sum()
            if np.linalg.norm(next_center - center) <= 1e-10 * max(
                np.linalg.norm(center), 1.0
            ):
                center = next_center
                break
            center = next_center
        output[route] = _sym(center)
    return output


def _normalization(npz_path: str, state_key: str, seed: int, train_ratio: float,
                   val_ratio: float):
    with np.load(npz_path, allow_pickle=True) as archive:
        raw = np.asarray(archive[state_key], dtype=np.float32)
        coefficients = np.asarray(archive["coefficients"], dtype=np.float64)
    if raw.ndim != 3:
        raise SystemExit(f"known-SNR state must be [B,T,D], got {raw.shape}")
    order = np.random.default_rng(seed).permutation(raw.shape[0])
    n_train = max(1, int(round(raw.shape[0] * train_ratio)))
    n_val = max(1, int(round(raw.shape[0] * val_ratio)))
    if n_train + n_val >= raw.shape[0]:
        n_train, n_val = max(1, raw.shape[0] - 2), 1
    train = raw[order[:n_train]]
    mean = float(train.mean())
    std = float(train.std()) or 1.0
    return mean, std, coefficients


def _load_process(path: str, std: float, device: str, dtype: torch.dtype):
    with np.load(path, allow_pickle=True) as archive:
        transition = np.asarray(archive["ar_transition_matrix"], dtype=np.float64)
        covariance = np.asarray(
            archive["one_step_innovation_covariance"], dtype=np.float64
        ) / (std * std)
        name = str(archive["innovation_estimator_name"].item())
    return name, _process_tensors(transition, covariance, device, dtype)


def _process_tensors(transition: np.ndarray, covariance: np.ndarray, device: str,
                     dtype: torch.dtype):
    covariance = 0.5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(covariance)
    factor = (vectors * np.sqrt(np.maximum(values, 0.0))[None, :]) @ vectors.T
    return (
        torch.as_tensor(transition, device=device, dtype=dtype),
        torch.as_tensor(factor, device=device, dtype=dtype),
    )


def _sample_process(
    z: torch.Tensor,
    transition: torch.Tensor,
    factor: torch.Tensor,
) -> list[torch.Tensor]:
    state = torch.zeros_like(z[0])
    output = []
    for horizon in range(z.shape[0]):
        state = state @ transition.t() + z[horizon] @ factor.t()
        output.append(state)
    return output


def _conditional_decomposition(bundle, starts, preds, targets, coefficients, mean, std, a):
    start_tensor = torch.as_tensor(starts, device=a.device)
    raw_mean = bundle.xt[bundle.rows_t, start_tensor] * std + mean
    coeff = torch.as_tensor(coefficients, device=a.device, dtype=raw_mean.dtype)
    signal, noise, total = [], [], []
    for prediction, target in zip(preds, targets):
        raw_mean = raw_mean * coeff
        conditional_mean = (raw_mean - mean) / std
        signal.append((prediction - conditional_mean).detach())
        noise.append((conditional_mean - target).detach())
        total.append((prediction - target).detach())
    max_error = max(
        float((e - b - n).abs().max().detach().cpu())
        for e, b, n in zip(total, signal, noise)
    )
    return signal, noise, total, max_error


def _fit_centered_residual_variance(bundle, fit_starts, a):
    sums = [torch.zeros(bundle.state_dim, dtype=torch.float64, device=a.device)
            for _ in range(a.K)]
    sums2 = [torch.zeros_like(sums[0]) for _ in range(a.K)]
    counts = [0 for _ in range(a.K)]
    for index, starts in enumerate(fit_starts):
        preds, _roots, targets = rollout(bundle, starts, a)
        for horizon, (prediction, target) in enumerate(zip(preds, targets)):
            residual = (prediction - target).detach().double()
            sums[horizon] += residual.sum(dim=0)
            sums2[horizon] += residual.square().sum(dim=0)
            counts[horizon] += int(residual.shape[0])
        del preds, targets, _roots
        print(f"[fit] centered residual draw {index + 1}/{len(fit_starts)}", flush=True)
    variances = []
    for horizon in range(a.K):
        count = max(counts[horizon], 1)
        mu = sums[horizon] / count
        variance = (sums2[horizon] / count - mu.square()).clamp_min(1e-12)
        variances.append(variance.to(dtype=torch.float32).sqrt())
    return variances


def _solve(total: np.ndarray, noise: np.ndarray) -> np.ndarray:
    return solve_box_wiener_2x2(
        torch.as_tensor(total, dtype=torch.float64),
        torch.as_tensor(noise, dtype=torch.float64),
    ).detach().cpu().numpy().astype(np.float64)


def _negative_signal_stats(total: np.ndarray, noise: np.ndarray):
    raw = _sym(total) - _sym(noise)
    values = np.linalg.eigvalsh(raw)
    scale = max(float(np.trace(_sym(total))), 1e-30)
    return float(values.min()), float(np.maximum(-values, 0.0).sum() / scale)


def _summarize(records, depth: int, checkpoint_coefficients: np.ndarray):
    fields = ("T", "P", "R_exact", "R_exact_replay", "C", "R_true_mc", "R_centered",
              "R_autocov", "R_autocov_oas")
    aggregate = {field: _average_maps(record[field] for record in records) for field in fields}
    routes = sorted(set.intersection(*(set(aggregate[field]) for field in fields)))
    if not routes:
        counts = {field: len(aggregate[field]) for field in fields}
        raise RuntimeError(
            "matched-route audit has no common captured routes; "
            f"per-source route counts={counts}"
        )
    rows = []
    for route in routes:
        T = _sym(aggregate["T"][route])
        P = _sym(aggregate["P"][route])
        R = _sym(aggregate["R_exact"][route])
        C = np.asarray(aggregate["C"][route], dtype=np.float64)
        ideal_total = P + R
        ideal = _solve(ideal_total, R)
        denom = max(float(np.linalg.norm(T)), 1e-30)
        closure = float(np.linalg.norm(T - (P + R + C + C.T)) / denom)
        cross = float(np.linalg.norm(C) / max(math.sqrt(
            max(float(np.trace(P)), 0.0) * max(float(np.trace(R)), 0.0)
        ), 1e-30))
        row = {
            "route": int(route), "horizon": int(route // depth),
            "layer": int(route % depth), "closure_error": closure,
            "cross_ratio": cross, "ideal_gain": ideal.tolist(),
            "estimators": {},
        }
        candidates = {
            "oracle_matched_T": R,
            "exact_noise_replay": aggregate["R_exact_replay"][route],
            "true_process_mc": aggregate["R_true_mc"][route],
            "centered_residual": aggregate["R_centered"][route],
            "autocov_matched": aggregate["R_autocov"][route],
            "autocov_oas": aggregate["R_autocov_oas"][route],
        }
        ideal_risk = max(risk_analytic(P, R, ideal), 1e-30)
        for name, estimated_noise in candidates.items():
            estimated_noise = _sym(estimated_noise)
            gain = _solve(T, estimated_noise)
            risk = risk_analytic(P, R, gain)
            min_eigenvalue, negative_mass = _negative_signal_stats(T, estimated_noise)
            row["estimators"][name] = {
                "gain": gain.tolist(),
                "gain_l1_to_ideal": float(np.abs(gain - ideal).mean()),
                "R_relative_frobenius_error": float(
                    np.linalg.norm(estimated_noise - R) / max(np.linalg.norm(R), 1e-30)
                ),
                "R_trace_over_T_trace": float(
                    np.trace(estimated_noise) / max(float(np.trace(T)), 1e-30)
                ),
                "min_eigenvalue_T_minus_R": min_eigenvalue,
                "negative_signal_mass_fraction": negative_mass,
                "ideal_risk_ratio": float(risk / ideal_risk),
            }
        h, layer = divmod(route, depth)
        if h < checkpoint_coefficients.shape[0] and layer < checkpoint_coefficients.shape[1]:
            gain = np.asarray(checkpoint_coefficients[h, layer], dtype=np.float64)
            row["estimators"]["checkpoint"] = {
                "gain": gain.tolist(),
                "gain_l1_to_ideal": float(np.abs(gain - ideal).mean()),
                "ideal_risk_ratio": float(risk_analytic(P, R, gain) / ideal_risk),
            }
        rows.append(row)

    estimator_names = list(rows[0]["estimators"]) if rows else []
    summary = {
        "draws": len(records),
        "routes": len(rows),
        "closure_error_median": float(np.median([r["closure_error"] for r in rows])),
        "closure_error_max": float(max(r["closure_error"] for r in rows)),
        "cross_ratio_median": float(np.median([r["cross_ratio"] for r in rows])),
        "estimators": {},
    }
    for name in estimator_names:
        entries = [row["estimators"][name] for row in rows]
        item = {
            "alpha_mean": float(np.mean([entry["gain"][0] for entry in entries])),
            "m_mean": float(np.mean([entry["gain"][1] for entry in entries])),
            "gain_l1_to_ideal_mean": float(np.mean([
                entry["gain_l1_to_ideal"] for entry in entries
            ])),
            "ideal_risk_ratio_median": float(np.median([
                entry["ideal_risk_ratio"] for entry in entries
            ])),
            "ideal_risk_ratio_p90": float(np.percentile([
                entry["ideal_risk_ratio"] for entry in entries
            ], 90)),
        }
        if name != "checkpoint":
            item.update({
                "R_relative_frobenius_error_median": float(np.median([
                    entry["R_relative_frobenius_error"] for entry in entries
                ])),
                "R_trace_exceeds_T_fraction": float(np.mean([
                    entry["R_trace_over_T_trace"] > 1.0 for entry in entries
                ])),
                "negative_T_minus_R_fraction": float(np.mean([
                    entry["min_eigenvalue_T_minus_R"] < -1e-12 for entry in entries
                ])),
                "negative_signal_mass_median": float(np.median([
                    entry["negative_signal_mass_fraction"] for entry in entries
                ])),
            })
        summary["estimators"][name] = item

    by_horizon = {}
    for horizon in sorted({row["horizon"] for row in rows}):
        selected = [row for row in rows if row["horizon"] == horizon]
        by_horizon[str(horizon + 1)] = {
            name: {
                "alpha": float(np.mean([r["estimators"][name]["gain"][0] for r in selected])),
                "m": float(np.mean([r["estimators"][name]["gain"][1] for r in selected])),
                "ideal_risk_ratio": float(np.mean([
                    r["estimators"][name]["ideal_risk_ratio"] for r in selected
                ])),
            }
            for name in estimator_names
        }
    return summary, by_horizon, rows


def _print_summary(prefix, summary):
    print(f"\n=== matched route audit: first {prefix} measurement draw(s) ===")
    print(f"routes={summary['routes']}  closure median={summary['closure_error_median']:.3e} "
          f"max={summary['closure_error_max']:.3e}  "
          f"signal/noise cross median={summary['cross_ratio_median']:.3e}")
    header = (f"{'estimator':>20s} {'alpha':>8s} {'m':>8s} {'|gain-or|':>10s} "
              f"{'risk/or':>9s} {'Rerr':>9s} {'R>T':>7s} {'neg(T-R)':>9s}")
    print(header)
    print("-" * len(header))
    for name, item in summary["estimators"].items():
        print(f"{name:>20s} {item['alpha_mean']:8.4f} {item['m_mean']:8.4f} "
              f"{item['gain_l1_to_ideal_mean']:10.4f} "
              f"{item['ideal_risk_ratio_median']:9.4f} "
              f"{item.get('R_relative_frobenius_error_median', float('nan')):9.4f} "
              f"{item.get('R_trace_exceeds_T_fraction', float('nan')):7.1%} "
              f"{item.get('negative_T_minus_R_fraction', float('nan')):9.1%}")


def _crossfit_summarize(records, depth: int, checkpoint_coefficients: np.ndarray):
    """Leave one t0 draw out and score gains on conditional oracle risk.

    ``R_exact`` contains only the one innovation realization paired with the
    observed residual in ``T``.  It is useful for the algebraic closure check,
    but is a noisy risk target.  Cross-fitted scoring instead uses ``P`` and
    the known-process Monte-Carlo ``R_true_mc`` from the held-out graph.
    """
    if len(records) < 2:
        return None
    all_rows = []
    for heldout_index in range(len(records)):
        fit_records = [record for index, record in enumerate(records)
                       if index != heldout_index]
        heldout = records[heldout_index]
        fit = {
            field: _average_maps(record[field] for record in fit_records)
            for field in ("T", "P", "R_exact", "R_true_mc", "R_centered",
                          "R_autocov", "R_autocov_oas")
        }
        robust = {
            field: _geometric_median_maps(record[field] for record in fit_records)
            for field in ("T", "P", "R_true_mc", "R_autocov_oas")
        }
        eval_P = heldout["P"]
        eval_R = heldout["R_true_mc"]
        fields = list(fit.values()) + [eval_P, eval_R]
        routes = sorted(set.intersection(*(set(field) for field in fields)))
        for route in routes:
            P_test = _sym(eval_P[route])
            R_test = _sym(eval_R[route])
            ideal_test = _solve(P_test + R_test, R_test)
            ideal_risk = max(risk_analytic(P_test, R_test, ideal_test), 1e-30)
            candidates = {
                # Knows the conditional signal and innovation covariances on
                # fit draws; this is the stationary estimand upper bound.
                "oracle_P_and_R": _solve(
                    _sym(fit["P"][route]) + _sym(fit["R_true_mc"][route]),
                    _sym(fit["R_true_mc"][route]),
                ),
                # Knows R, but estimates total covariance from realized model
                # residuals exactly as the current controller does.
                "oracle_R_with_T": _solve(
                    _sym(fit["T"][route]), _sym(fit["R_true_mc"][route])
                ),
                "oracle_R_with_robust_T": _solve(
                    _sym(robust["T"][route]), _sym(fit["R_true_mc"][route])
                ),
                "paired_realized_R": _solve(
                    _sym(fit["T"][route]), _sym(fit["R_exact"][route])
                ),
                "centered_residual": _solve(
                    _sym(fit["T"][route]), _sym(fit["R_centered"][route])
                ),
                "autocov_matched": _solve(
                    _sym(fit["T"][route]), _sym(fit["R_autocov"][route])
                ),
                "autocov_oas": _solve(
                    _sym(fit["T"][route]), _sym(fit["R_autocov_oas"][route])
                ),
                "oas_robust_T": _solve(
                    _sym(robust["T"][route]), _sym(fit["R_autocov_oas"][route])
                ),
                "oas_robust_T_and_R": _solve(
                    _sym(robust["T"][route]), _sym(robust["R_autocov_oas"][route])
                ),
            }
            horizon, layer = divmod(route, depth)
            candidates["checkpoint"] = np.asarray(
                checkpoint_coefficients[horizon, layer], dtype=np.float64
            )
            for name, gain in candidates.items():
                all_rows.append({
                    "heldout_draw": heldout_index,
                    "route": route,
                    "horizon": horizon,
                    "layer": layer,
                    "estimator": name,
                    "gain": gain.tolist(),
                    "gain_l1_to_test_ideal": float(np.abs(gain - ideal_test).mean()),
                    "heldout_ideal_risk_ratio": float(
                        risk_analytic(P_test, R_test, gain) / ideal_risk
                    ),
                })
    names = sorted({row["estimator"] for row in all_rows})
    summary = {}
    for name in names:
        selected = [row for row in all_rows if row["estimator"] == name]
        ratios = np.asarray([
            row["heldout_ideal_risk_ratio"] for row in selected
        ], dtype=np.float64)
        errors = np.asarray([
            row["gain_l1_to_test_ideal"] for row in selected
        ], dtype=np.float64)
        fold_medians = []
        for heldout_index in range(len(records)):
            fold = [row["heldout_ideal_risk_ratio"] for row in selected
                    if row["heldout_draw"] == heldout_index]
            fold_medians.append(float(np.median(fold)))
        summary[name] = {
            "heldout_ideal_risk_ratio_median": float(np.median(ratios)),
            "heldout_ideal_risk_ratio_mean": float(np.mean(ratios)),
            "heldout_ideal_risk_ratio_p90": float(np.percentile(ratios, 90)),
            "gain_l1_to_test_ideal_mean": float(np.mean(errors)),
            "fold_risk_ratio_medians": fold_medians,
        }
    return {
        "folds": len(records), "fit_draws_per_fold": len(records) - 1,
        "heldout_truth": "conditional P plus known-process Monte-Carlo R",
        "summary": summary,
    }


def _print_crossfit(result):
    if result is None:
        return
    print(f"\n=== leave-one-draw-out held-out route risk: "
          f"{result['folds']} folds, {result['fit_draws_per_fold']} fit draws/fold ===")
    header = f"{'estimator':>22s} {'risk/or med':>12s} {'risk/or mean':>13s} {'p90':>9s} {'|gain-or|':>10s}"
    print(header)
    print("-" * len(header))
    for name, item in result["summary"].items():
        print(f"{name:>22s} {item['heldout_ideal_risk_ratio_median']:12.4f} "
              f"{item['heldout_ideal_risk_ratio_mean']:13.4f} "
              f"{item['heldout_ideal_risk_ratio_p90']:9.4f} "
              f"{item['gain_l1_to_test_ideal_mean']:10.4f}")


def _gain_stability(records, depth: int):
    """Separate trajectory-sampling noise from genuine t0-dependent gains.

    Every draw uses the same fixed trajectory halves.  ``within_t0`` compares
    the two disjoint halves at one start; ``across_t0`` compares the same half
    at two starts.  Both estimates therefore use the same number of
    trajectories.  A substantial across/within excess supports time/state
    conditioning; equality supports stronger pooling instead.
    """
    if len(records) < 2 or not records[0].get("subgroups"):
        return None
    labels = sorted(records[0]["subgroups"])
    gains = []
    for record in records:
        draw_gains = {}
        for label in labels:
            subgroup = record["subgroups"][label]
            routes = sorted(set(subgroup["P"]).intersection(subgroup["R_true_mc"]))
            draw_gains[label] = {
                route: _solve(
                    _sym(subgroup["P"][route]) + _sym(subgroup["R_true_mc"][route]),
                    _sym(subgroup["R_true_mc"][route]),
                )
                for route in routes
            }
        gains.append(draw_gains)

    within, across = [], []
    split_ids = sorted({label[:-1] for label in labels})
    for draw_index, draw_gains in enumerate(gains):
        for split_id in split_ids:
            left, right = f"{split_id}a", f"{split_id}b"
            routes = set(draw_gains[left]).intersection(draw_gains[right])
            for route in routes:
                within.append({
                    "draw": draw_index, "route": route,
                    "horizon": route // depth,
                    "distance": float(np.abs(
                        draw_gains[left][route] - draw_gains[right][route]
                    ).mean()),
                })
    for left_draw in range(len(gains)):
        for right_draw in range(left_draw + 1, len(gains)):
            for label in labels:
                routes = set(gains[left_draw][label]).intersection(gains[right_draw][label])
                for route in routes:
                    across.append({
                        "left_draw": left_draw, "right_draw": right_draw,
                        "route": route, "horizon": route // depth,
                        "distance": float(np.abs(
                            gains[left_draw][label][route]
                            - gains[right_draw][label][route]
                        ).mean()),
                    })

    def describe(rows):
        values = np.asarray([row["distance"] for row in rows], dtype=np.float64)
        return {
            "median": float(np.median(values)), "mean": float(np.mean(values)),
            "p90": float(np.percentile(values, 90)), "count": int(values.size),
        }

    within_summary, across_summary = describe(within), describe(across)
    by_horizon = {}
    for horizon in sorted({row["horizon"] for row in within}):
        within_h = [row for row in within if row["horizon"] == horizon]
        across_h = [row for row in across if row["horizon"] == horizon]
        w, b = describe(within_h), describe(across_h)
        by_horizon[str(horizon + 1)] = {
            "within_t0": w, "across_t0": b,
            "median_ratio_across_over_within": b["median"] / max(w["median"], 1e-30),
        }
    return {
        "trajectory_halves": int(len(next(iter(records[0]["subgroups"].values()))["rows"])),
        "split_repeats": len(split_ids), "within_t0": within_summary,
        "across_t0": across_summary,
        "median_ratio_across_over_within": (
            across_summary["median"] / max(within_summary["median"], 1e-30)
        ),
        "by_horizon": by_horizon,
    }


def _print_stability(result):
    if result is None:
        return
    within, across = result["within_t0"], result["across_t0"]
    print("\n=== oracle-gain stability: fixed trajectory halves ===")
    print(f"within one t0 : median={within['median']:.4f} mean={within['mean']:.4f} "
          f"p90={within['p90']:.4f}")
    print(f"across t0     : median={across['median']:.4f} mean={across['mean']:.4f} "
          f"p90={across['p90']:.4f}")
    print(f"across/within median ratio = "
          f"{result['median_ratio_across_over_within']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--fit-draws", type=int, default=2,
                        help="first disjoint draws used only to fit centered residual variance")
    parser.add_argument("--noise-draws", type=int, default=8,
                        help="common-random-number Monte Carlo draws per measurement graph")
    parser.add_argument("--oracle-file", required=True)
    parser.add_argument("--autocov-file", required=True)
    parser.add_argument("--autocov-oas-file", required=True)
    parser.add_argument("--prefixes", default="1,2,4",
                        help="measurement-draw prefixes for the finite-sample audit")
    parser.add_argument("--out", required=True)
    parser.add_argument("--stability-splits", type=int, default=4,
                        help="fixed random trajectory bipartitions for gain-stability audit")
    a = parser.parse_args()

    if not a.npz or not a.state_key:
        raise SystemExit("this probe requires --npz and --state-key")
    if a.data_preprocess not in ("mackey_glass", "narma"):
        raise SystemExit("use --data-preprocess mackey_glass for the known-SNR AR data")
    if a.fit_draws < 1 or a.draws <= a.fit_draws:
        raise SystemExit("--draws must exceed --fit-draws")
    if a.noise_draws < 1:
        raise SystemExit("--noise-draws must be positive")

    mean, std, coefficients = _normalization(
        a.npz, a.state_key, a.seed, a.split_train_ratio, a.split_val_ratio
    )
    bundle = setup(a)
    fit_starts = bundle.draw_starts[:a.fit_draws]
    measure_starts = bundle.draw_starts[a.fit_draws:]
    centered_scales = _fit_centered_residual_variance(bundle, fit_starts, a)

    split_groups = {}
    if bundle.batch >= 4 and a.stability_splits > 0:
        for split_index in range(a.stability_splits):
            permutation = np.random.default_rng(
                a.seed + 7919 * (split_index + 1)
            ).permutation(bundle.batch)
            midpoint = bundle.batch // 2
            split_groups[f"s{split_index}a"] = permutation[:midpoint]
            split_groups[f"s{split_index}b"] = permutation[midpoint:]

    with np.load(a.oracle_file, allow_pickle=True) as oracle:
        true_coefficients = np.asarray(oracle["oracle_ar_coefficients"], dtype=np.float64)
        true_std = np.asarray(oracle["oracle_one_step_innovation_std"], dtype=np.float64) / std
    if not np.allclose(coefficients, true_coefficients, atol=1e-6):
        raise SystemExit("dataset coefficients disagree with --oracle-file")
    true_transition = np.diag(true_coefficients)
    true_covariance = np.diag(true_std ** 2)
    processes = {
        "true_mc": _process_tensors(
            true_transition, true_covariance, a.device, torch.float32
        )
    }
    raw_name, processes["autocov"] = _load_process(
        a.autocov_file, std, a.device, torch.float32
    )
    oas_name, processes["autocov_oas"] = _load_process(
        a.autocov_oas_file, std, a.device, torch.float32
    )

    dw = bundle.dw
    original_mode = dw._mode
    original_collecting = dw._collecting
    capture = RouteCapture(dw)
    records = []
    decomposition_errors = []
    try:
        dw._mode = "total"
        # route_pair() inserts the two custom autograd route nodes only on a
        # collecting batch.  _mode="total" makes those nodes fully open; it is
        # independent of whether the nodes are inserted.
        dw._collecting = True
        for draw_index, starts in enumerate(measure_starts):
            dw.begin_batch()
            dw._mode = "total"
            dw._collecting = True
            dw._slot = 0
            preds, roots, targets = rollout(bundle, starts, a)
            signal, exact_noise, total, error = _conditional_decomposition(
                bundle, starts, preds, targets, coefficients, mean, std, a
            )
            decomposition_errors.append(error)

            total_pairs = push(covector_loss(preds, total), roots, capture, retain=True)
            signal_pairs = push(covector_loss(preds, signal), roots, capture, retain=True)
            exact_noise_pairs = push(
                covector_loss(preds, exact_noise), roots, capture, retain=True
            )
            record = {
                "T": _route_moments(total_pairs),
                "P": _route_moments(signal_pairs),
                "R_exact": _route_moments(exact_noise_pairs),
                "C": _route_cross(signal_pairs, exact_noise_pairs),
                "subgroups": {
                    label: {
                        "rows": np.asarray(rows, dtype=np.int64).tolist(),
                        "P": _route_moments(signal_pairs, rows=rows),
                    }
                    for label, rows in split_groups.items()
                },
            }
            base_counts = {name: len(value) for name, value in record.items()}
            if min(base_counts.values(), default=0) == 0:
                raise RuntimeError(
                    "a matched oracle VJP captured no routes; "
                    f"draw={draw_index + 1} counts={base_counts}. "
                    "The controller must be collecting while _mode='total'."
                )

            mc_maps = {name: [] for name in (
                "R_true_mc", "R_centered", "R_autocov", "R_autocov_oas"
            )}
            subgroup_true_mc_maps = {label: [] for label in split_groups}
            generator = torch.Generator(device="cpu").manual_seed(
                a.seed + 100003 * (draw_index + 1)
            )
            batch = int(preds[0].shape[0])
            dim = int(preds[0].shape[1])
            for noise_index in range(a.noise_draws):
                z = torch.randn((a.K, batch, dim), generator=generator).to(
                    device=a.device, dtype=preds[0].dtype
                )
                source_vectors = {
                    "R_true_mc": _sample_process(z, *processes["true_mc"]),
                    "R_autocov": _sample_process(z, *processes["autocov"]),
                    "R_autocov_oas": _sample_process(z, *processes["autocov_oas"]),
                    "R_centered": [
                        z[horizon] * centered_scales[horizon].to(
                            device=a.device, dtype=preds[0].dtype
                        )[None, :]
                        for horizon in range(a.K)
                    ],
                }
                for source_index, (source, vectors) in enumerate(source_vectors.items()):
                    pairs = push(
                        covector_loss(preds, vectors), roots, capture, retain=True
                    )
                    mc_maps[source].append(_route_moments(pairs))
                    if source == "R_true_mc":
                        for label, rows in split_groups.items():
                            subgroup_true_mc_maps[label].append(
                                _route_moments(pairs, rows=rows)
                            )
            for source, maps in mc_maps.items():
                record[source] = _average_maps(maps)
                if not record[source]:
                    raise RuntimeError(
                        f"noise estimator {source!r} captured no routes on "
                        f"measurement draw {draw_index + 1}"
                    )
            for label, maps in subgroup_true_mc_maps.items():
                record["subgroups"][label]["R_true_mc"] = _average_maps(maps)
            # Replaying the exact covector after all Monte-Carlo VJPs detects
            # any non-reentrant/stateful backward implementation.  It must
            # reproduce R_exact to numerical precision and is the final VJP,
            # so it can release the graph.
            replay_pairs = push(
                covector_loss(preds, exact_noise), roots, capture, retain=False
            )
            record["R_exact_replay"] = _route_moments(replay_pairs)
            if not record["R_exact_replay"]:
                raise RuntimeError(
                    f"exact-noise replay captured no routes on measurement "
                    f"draw {draw_index + 1}"
                )
            records.append(record)
            print(f"[measure] draw {draw_index + 1}/{len(measure_starts)} "
                  f"noise_draws={a.noise_draws} routes={len(record['T'])} "
                  f"decomposition_max={error:.3e}", flush=True)
            del preds, roots, targets, total_pairs, signal_pairs, exact_noise_pairs
    finally:
        capture.close()
        dw._mode = original_mode
        dw._collecting = original_collecting

    requested_prefixes = sorted({
        min(max(int(value), 1), len(records))
        for value in a.prefixes.split(",") if value.strip()
    } | {len(records)})
    checkpoint_coefficients = dw.coefficients.detach().cpu().numpy()
    output = {
        "format_version": 1,
        "checkpoint": os.path.abspath(a.ckpt),
        "data": os.path.abspath(a.npz),
        "K": int(a.K), "depth": int(bundle.depth), "batch": int(bundle.batch),
        "fit_draws": int(a.fit_draws), "measurement_draws": int(len(records)),
        "noise_draws": int(a.noise_draws), "normalization_mean": mean,
        "normalization_std": std, "conditional_decomposition_max_error": float(
            max(decomposition_errors)
        ),
        "estimators": {
            "centered_residual": "per-horizon diagonal Gaussian on disjoint fit draws",
            "autocov_matched": raw_name,
            "autocov_oas": oas_name,
            "true_process_mc": "known diagonal AR transition and innovation covariance",
        },
        "prefixes": {},
    }
    for prefix in requested_prefixes:
        summary, by_horizon, rows = _summarize(
            records[:prefix], bundle.depth, checkpoint_coefficients
        )
        output["prefixes"][str(prefix)] = {
            "summary": summary, "by_horizon": by_horizon, "routes": rows
        }
        _print_summary(prefix, summary)

    crossfit = _crossfit_summarize(records, bundle.depth, checkpoint_coefficients)
    output["crossfit"] = crossfit
    _print_crossfit(crossfit)
    stability = _gain_stability(records, bundle.depth)
    output["gain_stability"] = stability
    _print_stability(stability)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(output), handle, indent=2, sort_keys=True)
    print(f"\n[out] {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
