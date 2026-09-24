"""Aggregate repeated Known-SNR gradient-profile measurements."""
import argparse
import json
from pathlib import Path
import numpy as np

def _write_json(path, payload, *, atomic=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + '\n')

def _band(values: np.ndarray) -> dict:
    return {
        "median": np.median(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
    }


def build_panels12(args: argparse.Namespace) -> None:
    paths = [Path(path) for path in args.input]
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    reference = payloads[0]
    reference_protocol = reference.get("protocol", {})
    required = {
        "strict_exact_closure": True,
        "checkpoint_role": "frozen_exact_bptt",
        "data_preprocess": "none",
        "normalization_mean": 0.0,
        "normalization_std": 1.0,
        "trajectory_split": "test",
    }
    for index, payload in enumerate(payloads):
        protocol = payload.get("protocol", {})
        for key, expected in required.items():
            if protocol.get(key) != expected:
                raise SystemExit(
                    f"input {index} violates strict protocol: {key}="
                    f"{protocol.get(key)!r}, expected {expected!r}"
                )
        for key in ("checkpoint", "data", "oracle_file", "K"):
            if payload.get(key) != reference.get(key):
                raise SystemExit(f"input {index} differs in {key}")
        for key in ("split_seed", "trajectory_indices"):
            if protocol.get(key) != reference_protocol.get(key):
                raise SystemExit(f"input {index} differs in protocol.{key}")

    profile = lambda key: np.asarray(  # noqa: E731
        [item["profile"][key] for item in payloads], dtype=np.float64
    )
    horizon = np.asarray(reference["profile"]["horizon"], dtype=np.int64)
    exact = np.maximum(
        profile("prefix_total_risk_to_full_clean_target")[:, -1:],
        np.finfo(np.float64).tiny,
    )
    amplitude = profile("total_rms_relative_to_h1")
    snr = profile("gradient_snr")
    noise_fraction = profile("noise_fraction")
    bias = profile("prefix_bias_risk_to_full_clean_target") / exact
    noise = profile("prefix_noise_risk") / exact
    total = profile("prefix_total_risk_to_full_clean_target") / exact
    total_median = np.median(total, axis=0)
    best_index = int(np.argmin(total_median))
    output = {
        "format_version": 1,
        "panel_sources": {
            "panel_1_magnitude_and_snr": [str(path.resolve()) for path in paths],
            "panel_2_prefix_signal_noise_tradeoff": [
                str(path.resolve()) for path in paths
            ],
        },
        "protocol": {
            "checkpoint": reference["checkpoint"],
            "checkpoint_role": "frozen_exact_bptt",
            "data": reference["data"],
            "oracle_file": reference["oracle_file"],
            "K": int(reference["K"]),
            "repetitions": len(payloads),
            "monte_carlo_replicate_seeds": [int(x["seed"]) for x in payloads],
            "data_preprocess": "none",
            "normalization_mean": 0.0,
            "normalization_std": 1.0,
            "trajectory_split": "test",
            "split_seed": int(reference_protocol["split_seed"]),
            "trajectory_indices": reference_protocol["trajectory_indices"],
            "future_noise_draws": "independent diagonal-AR simulator draws",
        },
        "horizon": horizon.tolist(),
        "panel_1": {
            "total_gradient_rms_relative_to_h1": _band(amplitude),
            "gradient_snr": _band(snr),
            "noise_fraction": _band(noise_fraction),
        },
        "panel_2": {
            "omitted_signal_bias_over_exact": _band(bias),
            "innovation_risk_over_exact": _band(noise),
            "total_prefix_risk_over_exact": _band(total),
            "median_curve_best_prefix_horizon": int(horizon[best_index]),
            "median_curve_best_prefix_risk_over_exact": float(
                total_median[best_index]
            ),
            "per_repetition_best_prefix_horizon": [
                int(x["profile"]["best_prefix_horizon"]) for x in payloads
            ],
            "per_repetition_best_prefix_risk_over_exact": [
                float(x["profile"]["best_prefix_risk_over_exact"])
                for x in payloads
            ],
        },
    }
    _write_json(args.out, output, atomic=True)
    print(args.out.resolve())



if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['panels12'])
    p.add_argument('--input', action='append', required=True)
    p.add_argument('--out', type=Path, required=True)
    build_panels12(p.parse_args())
