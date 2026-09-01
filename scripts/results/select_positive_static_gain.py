#!/usr/bin/env python3
"""Select c in {0.3,0.6,0.9} from seed-0 validation loss only."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CANDIDATES = ("0.3", "0.6", "0.9")
DATASETS = ("mg", "ettm1", "ettm2", "shear")


def run_dir(root: Path, data: str, gain: str, seed: int = 0) -> Path:
    if data == "mg":
        return root / "mackey_glass" / "tau30_K32" / f"dwc{gain}" / f"seed{seed}"
    if data in {"ettm1", "ettm2"}:
        return root / "prepared_temporal_driven" / f"{data}_K64" / f"dwc{gain}" / f"seed{seed}"
    if data == "shear":
        return root / "shear_flow" / "unet_b32_D4_W2_K32_ds4" / f"dwc{gain}" / f"seed{seed}"
    raise ValueError(data)


def best_validation(path: Path) -> tuple[float | None, int | None]:
    log = path / "train_logs.jsonl"
    best: tuple[float, int | None] | None = None
    if not log.is_file():
        return None, None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        value = row.get("val/loss")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        epoch_value = row.get("epoch")
        epoch = int(epoch_value) if isinstance(epoch_value, (int, float)) else None
        candidate = (float(value), epoch)
        if best is None or candidate[0] < best[0]:
            best = candidate
    return best if best is not None else (None, None)


def load_report(path: Path) -> dict:
    if not path.is_file():
        return {
            "selection_metric": "minimum seed-0 val/loss in train_logs.jsonl",
            "test_metrics_read": False,
            "candidate_gains": list(CANDIDATES),
            "datasets": {},
        }
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("test_metrics_read") is not False:
        raise RuntimeError("existing selection ledger does not certify validation-only selection")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, action="append")
    parser.add_argument(
        "--root", type=Path,
        default=Path("experiments/internal_dw_static_positive_v1"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("probe_outputs/internal_dw_static_positive_v1/selection.json"),
    )
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else REPO / args.root
    out = args.out if args.out.is_absolute() else REPO / args.out
    requested = tuple(args.dataset or DATASETS)
    report = load_report(out)

    failed = False
    for data in requested:
        rows: dict[str, dict] = {}
        valid: list[tuple[float, str]] = []
        print(f"\n=== {data}: seed-0 static-gain validation sweep ===")
        for gain in CANDIDATES:
            path = run_dir(root, data, gain)
            value, epoch = best_validation(path)
            complete = (path / "eval_results.json").is_file()
            state = "DONE" if complete else "INCOMPLETE"
            if value is None or not complete:
                failed = True
                rows[gain] = {
                    "state": state,
                    "best_val_loss": value,
                    "epoch": epoch,
                    "path": str(path.relative_to(REPO)),
                }
                print(f"c={gain:<3} {state:<10} best-val={value}")
            else:
                rows[gain] = {
                    "state": "DONE",
                    "best_val_loss": value,
                    "epoch": epoch,
                    "path": str(path.relative_to(REPO)),
                }
                valid.append((value, gain))
                print(f"c={gain:<3} DONE       best-val={value:.8g} epoch={epoch}")
        if len(valid) == len(CANDIDATES):
            selected_val, selected_gain = min(valid)
            report["datasets"][data] = {
                "candidates": rows,
                "selected_c": selected_gain,
                "selected_val_loss": selected_val,
                "selection_seed": 0,
            }
            print(f"SELECT c={selected_gain} (val/loss={selected_val:.8g})")
        else:
            # Never allow a stale selection from an older/partial run to drive
            # the seed-1/2 queue after one candidate has disappeared or failed.
            report["datasets"].pop(data, None)
            print("SELECT unavailable until all three candidates finish")

    report["protocol"] = {
        "static_operator": "alpha=m=c at every internal residual route",
        "selection": "seed 0 validation only",
        "test_metrics_read": False,
        "selected_replicates": [0, 1, 2],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[out] {out.relative_to(REPO)}")
    if failed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
