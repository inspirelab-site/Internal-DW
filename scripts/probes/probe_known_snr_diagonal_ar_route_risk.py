#!/usr/bin/env python3
"""Held-out route-risk evaluation for frozen diagonal-AR(1) DW gains.

The gains are read from ``probe_known_snr_diagonal_ar_gain.py`` and are never
refit here.  This probe attaches fully-open internal routes to the same frozen
Exact-BPTT checkpoint, installs the untouched test trajectories, and estimates
the true conditional signal/noise route moments with coherent draws from the
known process.  It compares exactly four pre-registered arms:

* Exact BPTT / open: ``(alpha,m)=(1,1)``;
* misplaced: the diagonal-AR gains circularly shifted by K/2 within each layer;
* DW: the correctly assigned diagonal-AR gains;
* oracle: the true-process gains fitted on the validation routes.

No model parameter is trained or updated and no test loss selects a gain.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch


_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "src"))

from known_snr_route_ops import (  # noqa: E402
    conditional_decomposition as _conditional_decomposition,
    forward_invariance as _forward_invariance,
    process_tensors as _process_tensors,
    route_moments as _route_moments,
    sample_process as _sample_process,
    symmetrize as _sym,
)
from probe_known_snr_diagonal_ar_gain import (  # noqa: E402
    _jsonable,
    _preflight_exact_checkpoint,
    _split_indices,
)
from probe_data_ops import plan_draws  # noqa: E402
from probe_setup import add_common_args, setup  # noqa: E402
from probe_wiener_oracle import RouteCapture, covector_loss, push, rollout  # noqa: E402


MatrixMap = Dict[int, np.ndarray]


def _average_maps(maps: Iterable[MatrixMap]) -> MatrixMap:
    total: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = defaultdict(int)
    for matrix_map in maps:
        for route, matrix in matrix_map.items():
            if route not in total:
                total[route] = np.zeros((2, 2), dtype=np.float64)
            total[route] += np.asarray(matrix, dtype=np.float64)
            counts[route] += 1
    return {route: value / counts[route] for route, value in total.items()}


def _install_test_view(bundle, raw: np.ndarray, indices: np.ndarray, a) -> None:
    selected = np.ascontiguousarray(raw[indices])
    if a.batch > 0:
        selected = selected[: min(int(a.batch), selected.shape[0])]
        indices = indices[: selected.shape[0]]
    if selected.shape[0] < 2:
        raise SystemExit("route-risk evaluation needs at least two test trajectories")
    rows, starts, axis = plan_draws(selected, a)
    bundle.xt = torch.as_tensor(selected[rows], dtype=torch.float32, device=a.device)
    bundle.ut = None
    bundle.rows_t = torch.arange(len(rows), device=a.device)
    bundle.draw_starts = starts
    bundle.axis = f"test trajectories; {axis}"
    bundle.batch = int(len(rows))
    print(
        f"[evaluation] test trajectories={bundle.batch} "
        f"indices={indices.tolist()} starts={len(starts)}",
        flush=True,
    )


def _measure_test_moments(bundle, a, coefficients: np.ndarray, true_process):
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
            signal, _realized_noise, _total, decomposition_error = (
                _conditional_decomposition(
                    bundle,
                    starts,
                    predictions,
                    targets,
                    coefficients,
                    0.0,
                    1.0,
                    a,
                )
            )
            signal_pairs = push(
                covector_loss(predictions, signal), roots, capture, retain=True
            )
            P = _route_moments(signal_pairs)
            if not P:
                raise RuntimeError("test signal VJP captured no internal routes")

            generator = torch.Generator(device="cpu").manual_seed(
                int(a.seed) + 700_001 + 100_003 * (draw_index + 1)
            )
            noise_maps = []
            batch = int(predictions[0].shape[0])
            dimension = int(predictions[0].shape[1])
            for noise_index in range(int(a.noise_draws)):
                white = torch.randn(
                    (a.K, batch, dimension),
                    generator=generator,
                    dtype=torch.float32,
                ).to(device=a.device, dtype=predictions[0].dtype)
                vectors = _sample_process(white, *true_process)
                pairs = push(
                    covector_loss(predictions, vectors),
                    roots,
                    capture,
                    retain=noise_index != int(a.noise_draws) - 1,
                )
                noise_maps.append(_route_moments(pairs))
            R = _average_maps(noise_maps)
            common = set(P).intersection(R)
            if not common:
                raise RuntimeError("test signal/noise moments have no common routes")
            records.append(
                {
                    "P": P,
                    "R": R,
                    "starts": np.asarray(starts, dtype=np.int64).tolist(),
                    "decomposition_max_abs": float(decomposition_error),
                }
            )
            print(
                f"[risk] draw {draw_index + 1}/{len(bundle.draw_starts)} "
                f"routes={len(common)} true-process-draws={a.noise_draws} "
                f"decomposition={decomposition_error:.3e}",
                flush=True,
            )
            del predictions, roots, targets, signal_pairs
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


def _risk_summary(values: np.ndarray, opened: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    opened = np.asarray(opened, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(opened) & (opened > 1e-300)
    ratio = values[valid] / opened[valid]
    return {
        "routes": int(valid.sum()),
        "sum_mse_over_sum_open": float(values[valid].sum() / opened[valid].sum()),
        "median_route_mse_over_open": float(np.median(ratio)),
        "p90_route_mse_over_open": float(np.percentile(ratio, 90)),
        "max_route_mse_over_open": float(np.max(ratio)),
        "route_win_fraction_vs_open": float(np.mean(ratio < 1.0)),
    }


def _load_gains(gain_json: Path, gain_npz: Path, a, split_indices):
    metadata = json.loads(gain_json.read_text(encoding="utf-8"))
    if int(metadata["K"]) != int(a.K):
        raise SystemExit("gain artifact K does not match route-risk K")
    if int(metadata["depth"]) != int(a.depth):
        raise SystemExit("gain artifact depth does not match route-risk depth")
    if int(metadata["seed"]) != int(a.seed):
        raise SystemExit("gain artifact seed does not match route-risk seed")
    if Path(metadata["checkpoint"]).name != Path(a.ckpt).name:
        raise SystemExit("gain artifact checkpoint filename differs")
    stored_test = np.asarray(
        metadata["separation"]["test_trajectory_indices_untouched"], dtype=np.int64
    )
    if not np.array_equal(stored_test, np.asarray(split_indices, dtype=np.int64)):
        raise SystemExit("test split differs from the untouched gain-artifact split")
    with np.load(gain_npz, allow_pickle=False) as archive:
        estimated = np.asarray(archive["estimated_gains"], dtype=np.float64)
        oracle = np.asarray(archive["oracle_gains"], dtype=np.float64)
    expected = (int(a.K), int(a.depth), 2)
    if estimated.shape != expected or oracle.shape != expected:
        raise SystemExit(
            f"gain array shape mismatch: {estimated.shape}, {oracle.shape}, expected {expected}"
        )
    if not np.all(np.isfinite(estimated)) or not np.all(np.isfinite(oracle)):
        raise SystemExit("gain artifact contains non-finite values")
    return metadata, estimated, oracle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--gain-json", required=True, type=Path)
    parser.add_argument("--gain-npz", required=True, type=Path)
    parser.add_argument("--noise-draws", type=int, default=64)
    parser.add_argument("--out", required=True, type=Path)
    a = parser.parse_args()

    if not a.npz or not a.state_key:
        raise SystemExit("pass --npz and --state-key")
    if a.data_preprocess != "none":
        raise SystemExit("known-SNR coordinates require --data-preprocess none")
    if not a.exact_checkpoint_adapter:
        raise SystemExit("pass --exact-checkpoint-adapter")
    if a.K != 32:
        raise SystemExit("the pre-registered closure uses K=32")
    if a.draws < 1 or a.noise_draws < 1 or a.t0 >= 0:
        raise SystemExit("use positive draw counts and do not pin t0")
    if not a.gain_json.is_file() or not a.gain_npz.is_file():
        raise SystemExit("completed gain-identification JSON and NPZ are required")

    _preflight_exact_checkpoint(a.ckpt, int(a.seed))
    with np.load(a.npz, allow_pickle=True) as archive:
        raw = np.asarray(archive[a.state_key], dtype=np.float32)
        coefficients = np.asarray(archive["coefficients"], dtype=np.float64)
    splits = _split_indices(
        raw.shape[0], int(a.seed), a.split_train_ratio, a.split_val_ratio
    )
    gain_metadata, estimated_gains, oracle_gains = _load_gains(
        a.gain_json, a.gain_npz, a, splits["test"]
    )

    torch.manual_seed(int(a.seed))
    np.random.seed(int(a.seed))
    bundle = setup(a)
    if bundle.ut is not None:
        raise SystemExit("known-SNR AR does not use an external drive")
    _install_test_view(bundle, raw, splits["test"], a)

    open_gains = np.ones_like(estimated_gains)
    misplaced_gains = np.roll(estimated_gains, shift=int(a.K // 2), axis=0)
    arms = {
        "exact_bptt_open": open_gains,
        "misplaced": misplaced_gains,
        "diagonal_ar_dw": estimated_gains,
        "oracle": oracle_gains,
    }
    # Run this cheap forward-only preflight before the expensive route VJPs.
    # The helper reserves the literal key ``open`` as its reference, while the
    # serialized scientific arm remains ``exact_bptt_open``.
    coefficient_arms = {
        "open": np.asarray(open_gains),
        "misplaced": np.asarray(misplaced_gains),
        "diagonal_ar_dw": np.asarray(estimated_gains),
        "oracle": np.asarray(oracle_gains),
    }
    forward = _forward_invariance(
        bundle, bundle.draw_starts[0], a, coefficient_arms
    )

    true_q = 1.0 - coefficients * coefficients
    true_process = _process_tensors(
        np.diag(coefficients), np.diag(true_q), a.device, torch.float32
    )
    records = _measure_test_moments(bundle, a, coefficients, true_process)
    P = _average_maps(record["P"] for record in records)
    R = _average_maps(record["R"] for record in records)
    routes = sorted(set(P).intersection(R))
    if len(routes) != int(a.K * bundle.depth):
        raise RuntimeError(
            f"expected {a.K * bundle.depth} routes, found {len(routes)}"
        )

    route_rows = []
    method_risks = {name: [] for name in arms}
    for route in routes:
        horizon, layer = divmod(int(route), int(bundle.depth))
        row_risks = {}
        for name, gains in arms.items():
            value = _risk(_sym(P[route]), _sym(R[route]), gains[horizon, layer])
            method_risks[name].append(value)
            row_risks[name] = value
        open_value = max(row_risks["exact_bptt_open"], 1e-300)
        route_rows.append(
            {
                "route": int(route),
                "horizon": int(horizon + 1),
                "layer": int(layer + 1),
                "risk_over_open": {
                    name: float(value / open_value)
                    for name, value in row_risks.items()
                },
            }
        )

    opened = np.asarray(method_risks["exact_bptt_open"], dtype=np.float64)
    methods = {
        name: _risk_summary(np.asarray(values, dtype=np.float64), opened)
        for name, values in method_risks.items()
    }
    output = {
        "format_version": 1,
        "scope": "held-out local route-message risk; frozen Exact-BPTT checkpoint",
        "checkpoint": str(Path(a.ckpt).resolve()),
        "data": str(Path(a.npz).resolve()),
        "gain_artifact_json": str(a.gain_json.resolve()),
        "gain_artifact_npz": str(a.gain_npz.resolve()),
        "estimator": gain_metadata["estimator"]["name"],
        "K": int(a.K),
        "depth": int(bundle.depth),
        "seed": int(a.seed),
        "test_trajectory_indices": splits["test"],
        "test_t0_values": sorted(
            {
                int(start)
                for starts in bundle.draw_starts
                for start in np.asarray(starts).reshape(-1)
            }
        ),
        "true_process_draws_per_t0": int(a.noise_draws),
        "risk_definition": "(w-1)^T P_test (w-1) + w^T R_test w",
        "primary_aggregation": "sum route MSE / sum fully-open route MSE",
        "methods": methods,
        "misplaced_control": {
            "definition": "fixed K/2 circular horizon shift within every layer",
            "shift": int(a.K // 2),
            "preserves": "each layer's multiset of estimated (alpha,m) gains",
        },
        "forward_invariance": forward,
        "routes": route_rows,
        "caveats": [
            "Risk is local route-message MSE, not full parameter-gradient MSE.",
            "All gains were fixed before this test-split evaluation.",
            "No tied gain, clipping, JReg, TBPTT, Generic estimator, or training update is included.",
        ],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(
        json.dumps(_jsonable(output), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("\n=== held-out route-gradient risk / Exact-BPTT open ===")
    for name in ("exact_bptt_open", "misplaced", "diagonal_ar_dw", "oracle"):
        item = methods[name]
        print(
            f"{name:>18s}  pooled={item['sum_mse_over_sum_open']:.4f} "
            f"median={item['median_route_mse_over_open']:.4f} "
            f"p90={item['p90_route_mse_over_open']:.4f} "
            f"wins={item['route_win_fraction_vs_open']:.3f}"
        )
    print(
        "forward bitwise invariant="
        f"{output['forward_invariance']['all_arms_bitwise_equal']}"
    )
    print(f"[out] {a.out.resolve()}")


if __name__ == "__main__":
    main()
