#!/usr/bin/env python3
"""Identify Internal-DW gains from a train-only diagonal AR(1) prior.

This is the first gate in the known-SNR closure and deliberately does no
training and no forecasting evaluation.  It attaches fully-open internal route
hooks to one frozen Exact-BPTT checkpoint, fits

    x[t+1,j] = a[j] x[t,j] + eps[t+1,j]

coordinate by coordinate on the training trajectories only, and compares the
resulting route gains with gains obtained from the true synthetic process.

The validation trajectories provide the route geometry.  Estimated and true
processes use the same starts, the same fully-open autograd graphs, and common
standard-normal draws.  The test trajectories remain untouched for the later
route-risk panel.  Two errors are reported:

``pipeline_vs_oracle`` (primary)
    The deployable solve ``solve(T_observed, R_diagonal_AR)`` versus the ideal
    conditional solve ``solve(P_true + R_true, R_true)``.

``process_only_vs_oracle`` (diagnostic)
    ``solve(P_true + R_diagonal_AR, R_diagonal_AR)`` versus the same oracle.
    This removes finite-sample total-moment error and isolates the AR prior.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch


_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "src"))

from known_snr_route_ops import (  # noqa: E402
    conditional_decomposition as _conditional_decomposition,
    process_tensors as _process_tensors,
    route_moments as _route_moments,
    sample_process as _sample_process,
    solve_gain as _solve,
    symmetrize as _sym,
)
from probe_data_ops import plan_draws  # noqa: E402
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


def _split_indices(count: int, seed: int, train_ratio: float, val_ratio: float):
    order = np.random.default_rng(int(seed)).permutation(int(count))
    n_train = max(1, int(round(int(count) * float(train_ratio))))
    n_val = max(1, int(round(int(count) * float(val_ratio))))
    if n_train + n_val >= count:
        n_train, n_val = max(1, count - 2), 1
    return {
        "train": order[:n_train],
        "val": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }


def _fit_diagonal_ar1(train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Fit the exact pre-specified zero-intercept coordinatewise AR(1)."""

    x = np.asarray(train[:, :-1], dtype=np.float64).reshape(-1, train.shape[-1])
    y = np.asarray(train[:, 1:], dtype=np.float64).reshape(-1, train.shape[-1])
    denominator = np.sum(x * x, axis=0)
    if np.any(denominator <= np.finfo(np.float64).tiny):
        raise RuntimeError("a training coordinate has zero AR denominator")
    transition = np.sum(x * y, axis=0) / denominator
    residual = y - x * transition[None, :]
    innovation_variance = np.mean(residual * residual, axis=0)
    if not np.all(np.isfinite(transition)) or not np.all(
        np.isfinite(innovation_variance)
    ):
        raise RuntimeError("non-finite diagonal AR(1) estimate")
    innovation_variance = np.maximum(
        innovation_variance, np.finfo(np.float64).tiny
    )
    return transition, innovation_variance, int(x.shape[0])


def _average_maps(maps: Iterable[MatrixMap]) -> MatrixMap:
    total: Dict[int, np.ndarray] = {}
    count: Dict[int, int] = defaultdict(int)
    for matrix_map in maps:
        for route, matrix in matrix_map.items():
            if route not in total:
                total[route] = np.zeros((2, 2), dtype=np.float64)
            total[route] += np.asarray(matrix, dtype=np.float64)
            count[route] += 1
    return {route: value / count[route] for route, value in total.items()}


def _symmetric_relative_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    estimate = np.asarray(estimate, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    numerator = float(np.linalg.norm(estimate - truth))
    denominator = float(np.linalg.norm(estimate) + np.linalg.norm(truth))
    if denominator <= np.finfo(np.float64).tiny:
        return 0.0 if numerator <= np.finfo(np.float64).tiny else 1.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _relative_frobenius(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(estimate) - np.asarray(truth))
        / max(float(np.linalg.norm(truth)), np.finfo(np.float64).tiny)
    )


def _preflight_exact_checkpoint(path: str, requested_seed: int) -> None:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = blob
    for key in ("model", "state_dict", "model_state_dict"):
        if isinstance(blob, dict) and key in blob:
            state = blob[key]
            break
    keys = {
        str(key).removeprefix("module.")
        for key in state
        if isinstance(state, dict)
    }
    if "dual_wiener.coefficients" in keys:
        raise SystemExit(
            "--ckpt contains trained Internal-DW coefficients; the gain gate "
            "requires the frozen Exact-BPTT checkpoint"
        )
    if "global_horizon_wiener.weights" in keys:
        raise SystemExit("outer global-horizon checkpoint rejected")
    arguments = blob.get("args", {}) if isinstance(blob, dict) else {}
    if isinstance(arguments, dict):
        dataset = arguments.get("dataset")
        if dataset not in (None, "known_snr_ar"):
            raise SystemExit(f"checkpoint dataset is {dataset!r}, not known_snr_ar")
        checkpoint_seed = arguments.get("seed")
        if checkpoint_seed is not None and int(checkpoint_seed) != requested_seed:
            raise SystemExit(
                f"checkpoint seed={checkpoint_seed}, requested seed={requested_seed}"
            )


def _install_validation_view(bundle, raw: np.ndarray, indices: np.ndarray, a) -> None:
    selected = np.ascontiguousarray(raw[indices])
    if a.batch > 0:
        selected = selected[: min(int(a.batch), selected.shape[0])]
        indices = indices[: selected.shape[0]]
    if selected.shape[0] < 2:
        raise SystemExit("gain identification needs at least two validation trajectories")
    rows, starts, axis = plan_draws(selected, a)
    bundle.xt = torch.as_tensor(selected[rows], dtype=torch.float32, device=a.device)
    bundle.ut = None
    bundle.rows_t = torch.arange(len(rows), device=a.device)
    bundle.draw_starts = starts
    bundle.axis = f"validation trajectories; {axis}"
    bundle.batch = int(len(rows))
    print(
        f"[calibration] validation trajectories={bundle.batch} "
        f"indices={indices.tolist()} starts={len(starts)}",
        flush=True,
    )


def _measure(
    bundle,
    a,
    true_coefficients: np.ndarray,
    estimated_process,
    true_process,
) -> list[Dict[str, Any]]:
    dw = bundle.dw
    saved_mode, saved_collecting, saved_slot = dw._mode, dw._collecting, dw._slot
    capture = RouteCapture(dw)
    records = []
    try:
        for draw_index, starts in enumerate(bundle.draw_starts):
            dw.begin_batch()
            dw._mode = "total"
            dw._collecting = True
            dw._slot = 0
            dw._root_refs = []
            predictions, roots, targets = rollout(bundle, starts, a)
            signal, _realized_noise, total, decomposition_error = (
                _conditional_decomposition(
                    bundle,
                    starts,
                    predictions,
                    targets,
                    true_coefficients,
                    0.0,
                    1.0,
                    a,
                )
            )
            signal_pairs = push(
                covector_loss(predictions, signal), roots, capture, retain=True
            )
            total_pairs = push(
                covector_loss(predictions, total), roots, capture, retain=True
            )
            record = {
                "P": _route_moments(signal_pairs),
                "T": _route_moments(total_pairs),
                "R_estimated_draws": [],
                "R_true_draws": [],
                "starts": np.asarray(starts, dtype=np.int64).tolist(),
                "decomposition_max_abs": float(decomposition_error),
            }
            if not record["P"] or not record["T"]:
                raise RuntimeError("fully-open signal/total VJP captured no routes")

            generator = torch.Generator(device="cpu").manual_seed(
                int(a.seed) + 100_003 * (draw_index + 1)
            )
            batch = int(predictions[0].shape[0])
            dimension = int(predictions[0].shape[1])
            for noise_index in range(int(a.noise_draws)):
                white = torch.randn(
                    (a.K, batch, dimension),
                    generator=generator,
                    dtype=torch.float32,
                ).to(device=a.device, dtype=predictions[0].dtype)
                estimated_vectors = _sample_process(white, *estimated_process)
                true_vectors = _sample_process(white, *true_process)
                estimated_pairs = push(
                    covector_loss(predictions, estimated_vectors),
                    roots,
                    capture,
                    retain=True,
                )
                is_last = noise_index == int(a.noise_draws) - 1
                true_pairs = push(
                    covector_loss(predictions, true_vectors),
                    roots,
                    capture,
                    retain=not is_last,
                )
                record["R_estimated_draws"].append(_route_moments(estimated_pairs))
                record["R_true_draws"].append(_route_moments(true_pairs))
            record["R_estimated"] = _average_maps(record.pop("R_estimated_draws"))
            record["R_true"] = _average_maps(record.pop("R_true_draws"))
            common = set(record["P"]).intersection(
                record["T"], record["R_estimated"], record["R_true"]
            )
            if not common:
                raise RuntimeError("route sources have no common entries")
            records.append(record)
            print(
                f"[gain] draw {draw_index + 1}/{len(bundle.draw_starts)} "
                f"routes={len(common)} common-white={a.noise_draws} "
                f"decomposition={decomposition_error:.3e}",
                flush=True,
            )
            del predictions, roots, targets, signal_pairs, total_pairs
    finally:
        capture.close()
        dw._mode = saved_mode
        dw._collecting = saved_collecting
        dw._slot = saved_slot
        dw._root_refs = []
    return records


def _describe(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def _summarize(records: list[Dict[str, Any]], K: int, depth: int):
    aggregate = {
        name: _average_maps(record[name] for record in records)
        for name in ("P", "T", "R_estimated", "R_true")
    }
    routes = sorted(set.intersection(*(set(value) for value in aggregate.values())))
    if len(routes) != int(K * depth):
        raise RuntimeError(
            f"expected {K * depth} routes, found {len(routes)} common routes"
        )

    heatmap = np.full((K, depth), np.nan, dtype=np.float64)
    process_heatmap = np.full_like(heatmap, np.nan)
    estimated_gains = np.full((K, depth, 2), np.nan, dtype=np.float64)
    oracle_gains = np.full_like(estimated_gains, np.nan)
    process_only_gains = np.full_like(estimated_gains, np.nan)
    true_process_pipeline_gains = np.full_like(estimated_gains, np.nan)
    rows = []
    for route in routes:
        horizon, layer = divmod(int(route), int(depth))
        P = _sym(aggregate["P"][route])
        T = _sym(aggregate["T"][route])
        R_estimated = _sym(aggregate["R_estimated"][route])
        R_true = _sym(aggregate["R_true"][route])
        oracle = _solve(P + R_true, R_true)
        estimated = _solve(T, R_estimated)
        process_only = _solve(P + R_estimated, R_estimated)
        true_process_pipeline = _solve(T, R_true)
        primary_error = _symmetric_relative_error(estimated, oracle)
        process_error = _symmetric_relative_error(process_only, oracle)

        heatmap[horizon, layer] = primary_error
        process_heatmap[horizon, layer] = process_error
        estimated_gains[horizon, layer] = estimated
        oracle_gains[horizon, layer] = oracle
        process_only_gains[horizon, layer] = process_only
        true_process_pipeline_gains[horizon, layer] = true_process_pipeline
        rows.append(
            {
                "route": int(route),
                "horizon": int(horizon + 1),
                "layer": int(layer + 1),
                "estimated_gain": estimated,
                "oracle_gain": oracle,
                "process_only_gain": process_only,
                "true_process_pipeline_gain": true_process_pipeline,
                "pipeline_symmetric_relative_error": primary_error,
                "process_only_symmetric_relative_error": process_error,
                "alpha_absolute_error": float(abs(estimated[0] - oracle[0])),
                "m_absolute_error": float(abs(estimated[1] - oracle[1])),
                "R_relative_frobenius_error": _relative_frobenius(
                    R_estimated, R_true
                ),
                "T_relative_frobenius_error_to_P_plus_R": _relative_frobenius(
                    T, P + R_true
                ),
            }
        )

    primary = np.asarray(
        [row["pipeline_symmetric_relative_error"] for row in rows]
    )
    process = np.asarray(
        [row["process_only_symmetric_relative_error"] for row in rows]
    )
    summary = {
        "routes": int(len(rows)),
        "pipeline_vs_oracle": _describe(primary),
        "process_only_vs_oracle": _describe(process),
        "alpha_absolute_error": _describe(
            np.asarray([row["alpha_absolute_error"] for row in rows])
        ),
        "m_absolute_error": _describe(
            np.asarray([row["m_absolute_error"] for row in rows])
        ),
        "R_relative_frobenius_error": _describe(
            np.asarray([row["R_relative_frobenius_error"] for row in rows])
        ),
        "T_relative_frobenius_error_to_P_plus_R": _describe(
            np.asarray(
                [row["T_relative_frobenius_error_to_P_plus_R"] for row in rows]
            )
        ),
        "estimated_gain_mean": estimated_gains.mean(axis=(0, 1)).tolist(),
        "oracle_gain_mean": oracle_gains.mean(axis=(0, 1)).tolist(),
    }
    arrays = {
        "pipeline_error": heatmap,
        "process_only_error": process_heatmap,
        "estimated_gains": estimated_gains,
        "oracle_gains": oracle_gains,
        "process_only_gains": process_only_gains,
        "true_process_pipeline_gains": true_process_pipeline_gains,
    }
    return summary, rows, arrays


def _plot(error: np.ndarray, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    maximum = float(np.nanmax(error))
    upper = min(1.0, max(0.05, math.ceil(maximum / 0.05) * 0.05))
    fig, axis = plt.subplots(figsize=(3.35, 3.15), constrained_layout=True)
    image = axis.imshow(
        error,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=upper,
        cmap="magma",
    )
    axis.set_xlabel("residual block $\\ell$")
    axis.set_ylabel("forecast step $k$")
    axis.set_xticks(np.arange(error.shape[1]), np.arange(1, error.shape[1] + 1))
    axis.set_yticks(
        np.asarray([0, 3, 7, 15, 23, 31]),
        np.asarray([1, 4, 8, 16, 24, 32]),
    )
    colorbar = fig.colorbar(image, ax=axis, pad=0.025)
    colorbar.set_label("symmetric relative gain error")
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--noise-draws", type=int, default=64)
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-plot", action="store_true")
    a = parser.parse_args()

    if not a.npz or not a.state_key:
        raise SystemExit("pass --npz and --state-key")
    if a.data_preprocess != "none":
        raise SystemExit("known-SNR coordinates require --data-preprocess none")
    if not a.exact_checkpoint_adapter:
        raise SystemExit("pass --exact-checkpoint-adapter for this frozen probe")
    if a.K != 32:
        raise SystemExit("the pre-registered closure uses K=32")
    if a.draws < 1 or a.noise_draws < 1:
        raise SystemExit("draw counts must be positive")
    if a.t0 >= 0:
        raise SystemExit("do not pin t0; multiple calibration starts are required")

    _preflight_exact_checkpoint(a.ckpt, int(a.seed))
    with np.load(a.npz, allow_pickle=True) as archive:
        raw = np.asarray(archive[a.state_key], dtype=np.float32)
        true_coefficients = np.asarray(archive["coefficients"], dtype=np.float64)
    if raw.ndim != 3 or true_coefficients.shape != (raw.shape[-1],):
        raise SystemExit(
            f"expected [B,T,D] trajectories and [D] coefficients, got "
            f"{raw.shape} and {true_coefficients.shape}"
        )
    splits = _split_indices(
        raw.shape[0], int(a.seed), a.split_train_ratio, a.split_val_ratio
    )
    estimated_coefficients, estimated_q, transition_count = _fit_diagonal_ar1(
        raw[splits["train"]]
    )
    true_q = 1.0 - true_coefficients * true_coefficients
    if np.any(true_q <= 0.0):
        raise SystemExit("true stationary diagonal AR process has non-positive Q")

    torch.manual_seed(int(a.seed))
    np.random.seed(int(a.seed))
    bundle = setup(a)
    if bundle.ut is not None:
        raise SystemExit("known-SNR AR does not use an external drive")
    _install_validation_view(bundle, raw, splits["val"], a)

    estimated_process = _process_tensors(
        np.diag(estimated_coefficients),
        np.diag(estimated_q),
        a.device,
        torch.float32,
    )
    true_process = _process_tensors(
        np.diag(true_coefficients),
        np.diag(true_q),
        a.device,
        torch.float32,
    )
    records = _measure(
        bundle,
        a,
        true_coefficients,
        estimated_process,
        true_process,
    )
    summary, rows, arrays = _summarize(records, int(a.K), int(bundle.depth))

    output_path = Path(a.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "format_version": 1,
        "scope": "gain identification only; frozen Exact-BPTT checkpoint",
        "checkpoint": str(Path(a.ckpt).resolve()),
        "data": str(Path(a.npz).resolve()),
        "K": int(a.K),
        "depth": int(bundle.depth),
        "seed": int(a.seed),
        "estimator": {
            "name": "train_only_coordinatewise_zero_intercept_diagonal_AR1",
            "assumption": "independent Gaussian AR(1) coordinates",
            "uses_true_A_or_Q": False,
            "training_transition_count": transition_count,
            "a_hat": estimated_coefficients,
            "q_hat": estimated_q,
            "a_relative_l2_error_diagnostic_only": _relative_frobenius(
                estimated_coefficients, true_coefficients
            ),
            "q_relative_l2_error_diagnostic_only": _relative_frobenius(
                estimated_q, true_q
            ),
        },
        "oracle": {
            "uses_true_A_and_Q": True,
            "a": true_coefficients,
            "q": true_q,
        },
        "separation": {
            "train_trajectory_indices": splits["train"],
            "validation_route_indices": splits["val"],
            "test_trajectory_indices_untouched": splits["test"],
            "route_calibration_t0_values": sorted(
                {
                    int(start)
                    for starts in bundle.draw_starts
                    for start in np.asarray(starts).reshape(-1)
                }
            ),
            "common_white_process_draws_per_t0": int(a.noise_draws),
        },
        "error_definition": (
            "||w_hat-w_star||_2 / "
            "(||w_hat||_2+||w_star||_2+epsilon); range [0,1]"
        ),
        "summary": summary,
        "routes": rows,
        "caveats": [
            "The primary pipeline error includes finite-sample observed-total-moment error.",
            "The process-only error holds the analytic signal moment fixed and isolates the diagonal-AR innovation prior.",
            "No test trajectory, training update, forecasting metric, Generic estimator, OAS estimator, or tied-gain arm is used.",
        ],
    }
    output_path.write_text(
        json.dumps(_jsonable(output), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output_path.with_suffix(".npz"),
        **arrays,
        a_hat=estimated_coefficients,
        q_hat=estimated_q,
        a_true=true_coefficients,
        q_true=true_q,
    )
    if not a.no_plot:
        _plot(arrays["pipeline_error"], output_path)

    primary = summary["pipeline_vs_oracle"]
    isolated = summary["process_only_vs_oracle"]
    print("\n=== diagonal-AR(1) gain identification ===")
    print(
        "estimated mean alpha/m="
        f"{summary['estimated_gain_mean'][0]:.4f}/"
        f"{summary['estimated_gain_mean'][1]:.4f}"
    )
    print(
        "oracle mean alpha/m="
        f"{summary['oracle_gain_mean'][0]:.4f}/"
        f"{summary['oracle_gain_mean'][1]:.4f}"
    )
    print(
        "pipeline symmetric relative error "
        f"mean/median/p90/max={primary['mean']:.4f}/"
        f"{primary['median']:.4f}/{primary['p90']:.4f}/{primary['max']:.4f}"
    )
    print(
        "process-only symmetric relative error "
        f"mean/median/p90/max={isolated['mean']:.4f}/"
        f"{isolated['median']:.4f}/{isolated['p90']:.4f}/{isolated['max']:.4f}"
    )
    print(
        f"A relative error={output['estimator']['a_relative_l2_error_diagnostic_only']:.4e} "
        f"Q relative error={output['estimator']['q_relative_l2_error_diagnostic_only']:.4e}"
    )
    print(f"[out] {output_path.resolve()}")
    print(f"[out] {output_path.with_suffix('.npz').resolve()}")
    if not a.no_plot:
        print(f"[out] {output_path.with_suffix('.png').resolve()}")
        print(f"[out] {output_path.with_suffix('.pdf').resolve()}")


if __name__ == "__main__":
    main()
