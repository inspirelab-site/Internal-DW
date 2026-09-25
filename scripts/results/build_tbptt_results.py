#!/usr/bin/env python3
"""Select TBPTT segment length on validation and aggregate its test seeds."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch


DEFAULT_RUN_ROOT = Path("experiments/tbptt_positive_sweep_v1")
DEFAULT_SELECTION = Path("probe_outputs/tbptt_positive_sweep_v1/selection.json")
DEFAULT_DENSE_ROOT = Path("probe_outputs/tbptt_positive_sweep_v1/dense")


def _checkpoint_map(root: Path) -> dict[str, dict[int, Path]]:
    return {
        "mg": {
            8: root / "mg/mackey_glass/tau30_K32/tbptt8/seed0/best.pth",
            16: root / "mg/mackey_glass/tau30_K32/tbptt16/seed0/best.pth",
        },
        "ettm1": {
            16: root
            / "ettm1/prepared_temporal_driven/ettm1_K64/tbptt16/seed0/best.pth",
            32: root
            / "ettm1/prepared_temporal_driven/ettm1_K64/tbptt32/seed0/best.pth",
        },
        "ettm2": {
            16: root
            / "ettm2/prepared_temporal_driven/ettm2_K64/tbptt16/seed0/best.pth",
            32: root
            / "ettm2/prepared_temporal_driven/ettm2_K64/tbptt32/seed0/best.pth",
        },
        "shear": {
            8: root / "shear/shear_flow/unet_b32_D4_W2_K32_ds4/tbptt8/seed0/best.pth",
            16: root
            / "shear/shear_flow/unet_b32_D4_W2_K32_ds4/tbptt16/seed0/best.pth",
        },
    }


def _best_validation(path: Path) -> tuple[float, int]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    value = float(state["best_val"])
    epoch = int(state.get("epoch", -1))
    if not (value == value and value < float("inf")):
        raise ValueError(f"invalid best_val={value!r} in {path}")
    return value, epoch


def select(args: argparse.Namespace) -> None:
    rows: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    for data, candidates in _checkpoint_map(args.root).items():
        values: dict[str, dict[str, object]] = {}
        for segment, path in candidates.items():
            if not path.is_file():
                missing.append(str(path))
                continue
            # The marker is checked for completion only. Its test values are
            # deliberately never opened during selection.
            if not any((path.parent / marker).is_file()
                       for marker in (".train_complete", "eval_results.json")):
                missing.append(f"{path} (training not complete)")
                continue
            value, epoch = _best_validation(path)
            values[str(segment)] = {
                "best_val_loss": value,
                "best_epoch": epoch,
                "checkpoint": str(path),
            }
        if len(values) == len(candidates):
            selected = min(values, key=lambda key: float(values[key]["best_val_loss"]))
            rows[data] = {
                "K": 32 if data in {"mg", "shear"} else 64,
                "selected_S": int(selected),
                "candidates": values,
            }
    if missing:
        print("[not ready] seed-0 candidate checkpoints still missing:")
        for path in missing:
            print(f"  {path}")
        raise SystemExit(1)
    payload = {
        "selection_split": "validation",
        "selection_seed": 0,
        "selection_metric": "minimum saved best_val (val/loss)",
        "test_metrics_read_during_selection": False,
        "completion_check": ".train_complete or eval_results.json existence only; contents not read",
        "datasets": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"{'data':<8} {'K':>3} {'candidate val losses':<36} {'selected S':>10}")
    for data, row in rows.items():
        candidates = ", ".join(
            f"S={segment}: {entry['best_val_loss']:.6f}"
            for segment, entry in row["candidates"].items()
        )
        print(f"{data:<8} {row['K']:>3} {candidates:<36} {row['selected_S']:>10}")
    print(f"[saved] {args.out}")


def _result_path(
    dense_root: Path, data: str, segment: int, seed: int
) -> Path:
    return dense_root / data / f"tbptt{segment}_seed{seed}.json"


def aggregate(args: argparse.Namespace) -> None:
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    missing: list[Path] = []
    rows = []
    for data, item in selection["datasets"].items():
        segment = int(item["selected_S"])
        values = []
        sources = []
        for seed in range(3):
            path = _result_path(
                args.dense_root, data, segment, seed
            )
            if not path.is_file():
                missing.append(path)
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            values.append(float(payload["summary"]["all_horizons"]["mean"]))
            sources.append(str(path.resolve()))
        if len(values) == 3:
            rows.append(
                {
                    "data": data,
                    "K": int(item["K"]),
                    "selected_S": segment,
                    "seeds": values,
                    "mean": statistics.mean(values),
                    "sample_sd": statistics.stdev(values),
                    "sources": sources,
                }
            )
    if missing:
        print("[not ready] selected dense tests still missing:")
        for path in missing:
            print(f"  {path}")
        raise SystemExit(1)
    output = {
        "selection": str(args.selection.resolve()),
        "selection_split": selection["selection_split"],
        "test_metrics_read_during_selection": selection[
            "test_metrics_read_during_selection"
        ],
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"{'data':<8} {'K':>3} {'S*':>3} {'n':>2} {'mean':>11} {'sample sd':>11}  seeds")
    for row in rows:
        values = row["seeds"]
        text = ", ".join(f"{value:.5f}" for value in values)
        print(
            f"{row['data']:<8} {row['K']:>3} {row['selected_S']:>3} "
            f"{len(values):>2} {row['mean']:>11.5f} {row['sample_sd']:>11.5f}  {text}"
        )
    print(f"[saved] {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    selection = commands.add_parser("select", help="choose S using seed-0 validation")
    selection.add_argument("--root", type=Path, default=DEFAULT_RUN_ROOT)
    selection.add_argument("--out", type=Path, default=DEFAULT_SELECTION)
    selection.set_defaults(handler=select)
    summary = commands.add_parser("aggregate", help="aggregate three test seeds")
    summary.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    summary.add_argument("--dense-root", type=Path, default=DEFAULT_DENSE_ROOT)
    summary.add_argument(
        "--out",
        type=Path,
        default=Path("probe_outputs/tbptt_positive_sweep_v1/results.json"),
    )
    summary.set_defaults(handler=aggregate)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
