#!/usr/bin/env python3
"""Build the four result artifacts used by the known-SNR closure.

The public command has four subcommands:

``panels12`` aggregates repeated gradient-profile probes;
``forecasting`` verifies a matched Full-BPTT/Internal-DW pair;
``controls`` verifies all forecasting baselines; and
``closure`` assembles the cumulative five-panel ledger.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


HORIZONS = (1, 2, 4, 8, 16, 32)
FORECAST_MATCHED_ARGUMENTS = (
    "seed",
    "dataset",
    "model_name",
    "simple_hidden_dim",
    "simple_depth",
    "simple_nhead",
    "simple_dropout",
    "mamba_bptt_horizon",
    "mamba_burnin",
    "mamba_train_starts_per_sequence",
    "mamba_loss_type",
    "window_size",
    "base_lr",
    "weight_decay",
    "grad_clip",
    "num_epochs",
    "early_stop_patience",
    "ar_optimizer",
    "ar_scheduler",
    "ar_min_lr",
    "snr_ar_dim",
    "snr_ar_len",
    "snr_ar_traj",
    "snr_ar_seed",
    "snr_ar_coefficients",
)
CONTROL_COMMON_ARGUMENTS = tuple(
    key for key in FORECAST_MATCHED_ARGUMENTS if key != "grad_clip"
)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _write_json(path: Path, payload: dict, *, atomic: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=atomic) + "\n"
    if not atomic:
        path.write_text(text, encoding="utf-8")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _checkpoint_arguments(path: Path) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    values = blob.get("args", {}) if isinstance(blob, dict) else {}
    if hasattr(values, "__dict__"):
        values = vars(values)
    if not isinstance(values, dict):
        raise TypeError(f"checkpoint {path} has no argument dictionary")
    return values


def _horizon_metric(eval_path: Path, horizons: tuple[int, ...] | list[int]):
    values = json.loads(eval_path.read_text(encoding="utf-8"))
    by_horizon = {}
    for horizon in horizons:
        key = f"test/horizon_{horizon}_rel_l2"
        if key not in values:
            raise KeyError(f"{eval_path} is missing {key}")
        by_horizon[str(horizon)] = float(values[key])
    return by_horizon, float(np.mean(list(by_horizon.values())))


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


def build_forecasting(args: argparse.Namespace) -> None:
    exact_checkpoint = args.exact_run / "best.pth"
    dw_checkpoint = args.dw_run / "best.pth"
    exact_eval = args.exact_run / "eval_results.json"
    dw_eval = args.dw_run / "eval_results.json"
    for path in (exact_checkpoint, dw_checkpoint, exact_eval, dw_eval, args.artifact):
        if not path.is_file():
            raise FileNotFoundError(path)

    exact_arguments = _checkpoint_arguments(exact_checkpoint)
    dw_arguments = _checkpoint_arguments(dw_checkpoint)
    mismatches = {
        key: {"exact": exact_arguments.get(key), "diagonal_ar_dw": dw_arguments.get(key)}
        for key in FORECAST_MATCHED_ARGUMENTS
        if exact_arguments.get(key) != dw_arguments.get(key)
    }
    if mismatches:
        raise RuntimeError(
            "Full BPTT and diagonal-AR DW are not matched:\n"
            + json.dumps(mismatches, indent=2, default=str)
        )
    if not bool(dw_arguments.get("resgrad_routing", False)):
        raise RuntimeError("DW checkpoint does not enable internal residual routing")
    if str(dw_arguments.get("resgrad_policy", "")) != "dualwiener":
        raise RuntimeError("DW checkpoint does not use the dualwiener policy")
    if bool(exact_arguments.get("resgrad_routing", False)):
        raise RuntimeError("Full-BPTT checkpoint unexpectedly enables routing")
    if args.dw_world_size <= 0:
        raise ValueError("--dw-world-size must be positive")
    exact_effective_batch = int(exact_arguments["local_batch_size"]) * int(
        exact_arguments["grad_accum_steps"]
    )
    dw_effective_batch = (
        int(dw_arguments["local_batch_size"])
        * int(dw_arguments["grad_accum_steps"])
        * int(args.dw_world_size)
    )
    if exact_effective_batch != dw_effective_batch:
        raise RuntimeError(
            f"effective batch mismatch: Full BPTT={exact_effective_batch}, "
            f"DW={dw_effective_batch}"
        )

    exact_by_horizon, exact_mean = _horizon_metric(exact_eval, args.horizons)
    dw_by_horizon, dw_mean = _horizon_metric(dw_eval, args.horizons)
    reduction = (exact_mean - dw_mean) / exact_mean
    metadata = _load_json(args.artifact.with_suffix(".json"))
    output = {
        "format_version": 1,
        "protocol": {
            "seed": int(exact_arguments["seed"]),
            "metric": "mean test relative L2 over pre-specified horizons",
            "horizons": args.horizons,
            "selection": "validation-selected checkpoint; test used once for reporting",
            "exact_reused": True,
            "matched_checkpoint_arguments": list(FORECAST_MATCHED_ARGUMENTS),
            "argument_mismatches": mismatches,
            "exact_world_size": 1,
            "diagonal_ar_dw_world_size": int(args.dw_world_size),
            "exact_effective_batch": exact_effective_batch,
            "diagonal_ar_dw_effective_batch": dw_effective_batch,
        },
        "exact_bptt": {
            "display_name": "Full BPTT",
            "mean_relative_l2": exact_mean,
            "relative_l2_by_horizon": exact_by_horizon,
            "run": str(args.exact_run.resolve()),
        },
        "diagonal_ar_dw": {
            "mean_relative_l2": dw_mean,
            "relative_l2_by_horizon": dw_by_horizon,
            "run": str(args.dw_run.resolve()),
            "artifact": str(args.artifact.resolve()),
            "artifact_estimator": metadata["estimator"],
            "artifact_fit_split": metadata["fit_split"],
        },
        "paired_relative_reduction": reduction,
    }
    _write_json(args.output, output)
    build_closure(args.output.parent, quiet=True)
    print("DONE known-SNR forecasting closure")
    print("metric=mean test relative-L2 at H=" + ",".join(map(str, args.horizons)))
    print(f"Full-BPTT={exact_mean:.6f}")
    print(f"diagonal-AR Internal-DW={dw_mean:.6f}")
    print(f"paired relative reduction={100.0 * reduction:+.2f}%")
    print(f"out={args.output}")


def build_controls(args: argparse.Namespace) -> None:
    runs = {
        "exact_bptt": args.exact_run,
        "clip_0p1": args.clip_run,
        "jreg": args.jreg_run,
        "tbptt8": args.tbptt_run,
        "diagonal_ar_dw": args.dw_run,
    }
    for run in runs.values():
        for filename in ("best.pth", "eval_results.json"):
            if not (run / filename).is_file():
                raise FileNotFoundError(run / filename)
    protocol = _load_json(args.protocol)
    arguments = {
        name: _checkpoint_arguments(run / "best.pth") for name, run in runs.items()
    }
    exact = arguments["exact_bptt"]
    mismatches = {}
    for name, values in arguments.items():
        different = {
            key: {"exact": exact.get(key), name: values.get(key)}
            for key in CONTROL_COMMON_ARGUMENTS
            if exact.get(key) != values.get(key)
        }
        if different:
            mismatches[name] = different
    if mismatches:
        raise RuntimeError("protocol mismatch:\n" + json.dumps(mismatches, indent=2))
    if float(arguments["clip_0p1"].get("grad_clip")) != 0.1:
        raise RuntimeError("Clip arm does not use grad_clip=0.1")
    for name in ("exact_bptt", "jreg", "tbptt8", "diagonal_ar_dw"):
        if float(arguments[name].get("grad_clip")) != 1.0:
            raise RuntimeError(f"{name} does not use common grad_clip=1.0")
    if float(arguments["jreg"].get("forward_jacobian_lambda", 0.0)) != 0.1:
        raise RuntimeError("JReg checkpoint does not record lambda=0.1")
    if bool(arguments["exact_bptt"].get("resgrad_routing", False)):
        raise RuntimeError("Full-BPTT checkpoint unexpectedly routes gradients")
    if not bool(arguments["diagonal_ar_dw"].get("resgrad_routing", False)):
        raise RuntimeError("DW checkpoint does not route gradients")

    effective = {
        "exact_bptt": int(exact["local_batch_size"]) * int(exact["grad_accum_steps"]),
        "diagonal_ar_dw": int(arguments["diagonal_ar_dw"]["local_batch_size"])
        * int(arguments["diagonal_ar_dw"]["grad_accum_steps"])
        * int(protocol.get("diagonal_ar_dw_world_size", 4)),
    }
    for name in ("clip_0p1", "jreg", "tbptt8"):
        effective[name] = (
            int(arguments[name]["local_batch_size"])
            * int(arguments[name]["grad_accum_steps"])
            * int(protocol["world_size"])
        )
    if set(effective.values()) != {32}:
        raise RuntimeError(f"effective batches are not matched: {effective}")

    results = {}
    for name, run in runs.items():
        by_horizon, mean = _horizon_metric(run / "eval_results.json", HORIZONS)
        results[name] = {
            "mean_relative_l2": mean,
            "relative_l2_by_horizon": by_horizon,
            "run": str(run.resolve()),
        }
    exact_mean = results["exact_bptt"]["mean_relative_l2"]
    for value in results.values():
        value["relative_reduction_vs_exact"] = (
            exact_mean - value["mean_relative_l2"]
        ) / exact_mean
    output = {
        "format_version": 1,
        "metric": "mean test relative-L2 at H=1,2,4,8,16,32",
        "selection": "validation-selected checkpoint; test used once",
        "horizons": list(HORIZONS),
        "protocol": protocol,
        "verified_common_arguments": list(CONTROL_COMMON_ARGUMENTS),
        "argument_mismatches": mismatches,
        "effective_batch_sizes": effective,
        "results": results,
    }
    _write_json(args.output, output)
    build_closure(args.output.parent, quiet=True)
    print("DONE matched known-SNR forecasting controls")
    print("method                   rel-L2    reduction-vs-Full")
    for name, value in results.items():
        print(
            f"{name:<22} {value['mean_relative_l2']:.6f} "
            f"{100.0 * value['relative_reduction_vs_exact']:+8.2f}%"
        )
    print(f"out={args.output}")


def build_closure(root: Path, *, quiet: bool = False) -> Path:
    paths = {
        "panels_1_2": root / "panels_1_2.json",
        "gain_identification": root / "gain_identification.json",
        "route_risk": root / "route_risk.json",
        "forecasting": root / "forecasting.json",
        "forecasting_controls": root / "forecasting_controls.json",
    }
    payloads = {name: _load_json(path) for name, path in paths.items()}
    ledger = {
        "format_version": 1,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root.resolve()),
        "stages": {},
        "panel_sources": {},
    }
    panels = payloads["panels_1_2"]
    if panels is not None:
        ledger["stages"]["panels_1_2"] = {
            "state": "DONE",
            "panel_1": panels["panel_1"],
            "panel_2": panels["panel_2"],
            "source": str(paths["panels_1_2"].resolve()),
        }
        ledger["panel_sources"]["panel_1_magnitude_and_snr"] = str(
            paths["panels_1_2"].resolve()
        )
        ledger["panel_sources"]["panel_2_prefix_signal_noise_tradeoff"] = str(
            paths["panels_1_2"].resolve()
        )
    else:
        ledger["stages"]["panels_1_2"] = {"state": "WAIT"}

    gain = payloads["gain_identification"]
    if gain is not None:
        ledger["stages"]["gain_identification"] = {
            "state": "DONE",
            "estimator": gain["estimator"]["name"],
            "pipeline_vs_oracle": gain["summary"]["pipeline_vs_oracle"],
            "process_only_vs_oracle": gain["summary"]["process_only_vs_oracle"],
            "estimated_gain_mean": gain["summary"]["estimated_gain_mean"],
            "oracle_gain_mean": gain["summary"]["oracle_gain_mean"],
            "A_relative_error": gain["estimator"]["a_relative_l2_error_diagnostic_only"],
            "Q_relative_error": gain["estimator"]["q_relative_l2_error_diagnostic_only"],
            "source": str(paths["gain_identification"].resolve()),
        }
        ledger["panel_sources"]["panel_3_gain_identification"] = str(
            paths["gain_identification"].resolve()
        )
    else:
        ledger["stages"]["gain_identification"] = {"state": "WAIT"}

    risk = payloads["route_risk"]
    if risk is not None:
        ledger["stages"]["route_risk"] = {
            "state": "DONE",
            "methods": risk["methods"],
            "forward_bitwise_invariant": risk["forward_invariance"][
                "all_arms_bitwise_equal"
            ],
            "source": str(paths["route_risk"].resolve()),
        }
        ledger["panel_sources"]["panel_4_route_gradient_risk"] = str(
            paths["route_risk"].resolve()
        )
    else:
        ledger["stages"]["route_risk"] = {"state": "WAIT"}

    forecasting = payloads["forecasting"]
    ledger["stages"]["forecasting"] = (
        {
            "state": "DONE",
            "results": forecasting,
            "source": str(paths["forecasting"].resolve()),
        }
        if forecasting is not None
        else {"state": "WAIT"}
    )
    controls = payloads["forecasting_controls"]
    if controls is not None:
        ledger["stages"]["forecasting_controls"] = {
            "state": "DONE",
            "results": controls,
            "source": str(paths["forecasting_controls"].resolve()),
        }
        ledger["panel_sources"]["panel_5_forecasting_performance"] = str(
            paths["forecasting_controls"].resolve()
        )
    else:
        ledger["stages"]["forecasting_controls"] = {"state": "WAIT"}

    output = root / "closure_summary.json"
    _write_json(output, ledger, atomic=True)
    if not quiet:
        print(output)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    panels = commands.add_parser("panels12", help="aggregate Panels 1--2 probes")
    panels.add_argument("--input", action="append", required=True)
    panels.add_argument("--out", type=Path, required=True)
    panels.set_defaults(handler=build_panels12)

    forecasting = commands.add_parser(
        "forecasting", help="verify the matched forecasting pair"
    )
    forecasting.add_argument("--exact-run", type=Path, required=True)
    forecasting.add_argument("--dw-run", type=Path, required=True)
    forecasting.add_argument("--artifact", type=Path, required=True)
    forecasting.add_argument("--output", type=Path, required=True)
    forecasting.add_argument("--dw-world-size", type=int, default=1)
    forecasting.add_argument(
        "--horizons", type=int, nargs="+", default=list(HORIZONS)
    )
    forecasting.set_defaults(handler=build_forecasting)

    controls = commands.add_parser("controls", help="verify forecasting controls")
    for name in ("exact", "dw", "clip", "tbptt", "jreg"):
        controls.add_argument(f"--{name}-run", type=Path, required=True)
    controls.add_argument("--protocol", type=Path, required=True)
    controls.add_argument("--output", type=Path, required=True)
    controls.set_defaults(handler=build_controls)

    closure = commands.add_parser("closure", help="assemble the five-panel ledger")
    closure.add_argument("--root", type=Path, required=True)
    closure.add_argument("--quiet", action="store_true")
    closure.set_defaults(
        handler=lambda values: build_closure(values.root, quiet=values.quiet)
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
