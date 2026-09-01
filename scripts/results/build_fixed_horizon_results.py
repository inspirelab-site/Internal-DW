#!/usr/bin/env python3
"""Aggregate the fixed-evaluation-horizon Internal-DW K sweep.

Validation determines the discrete best K and the largest K within a relative
tolerance of the best risk.  Test values are reported at those validation-
selected K values when test JSONs are available; test never selects K.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


DATA_ORDER = ("mg", "narma", "ieeg", "fmri", "ettm1", "ettm2", "shear", "wb2")
FILE_RE = re.compile(
    r"^(?P<arm>exact|dw)_K(?P<K>\d+)_seed(?P<seed>\d+)_H(?P<H>\d+)\.json$"
)


def read_rows(root: Path) -> List[dict]:
    rows: List[dict] = []
    for split in ("val", "test"):
        split_root = root / split
        if not split_root.exists():
            continue
        for path in sorted(split_root.glob("*/*.json")):
            match = FILE_RE.match(path.name)
            if match is None:
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            data = path.parent.name
            K = int(match.group("K"))
            H = int(match.group("H"))
            summary = payload["summary"]
            payload_K = payload.get("train_horizon", summary.get("relative_l2_train_horizon"))
            payload_H = payload.get("eval_horizon", summary.get("relative_l2_eval_horizon"))
            if int(payload_K) != K or int(payload_H) != H:
                raise ValueError(f"filename/payload horizon mismatch: {path}")
            if str(payload["split"]) != split:
                raise ValueError(f"filename/payload split mismatch: {path}")
            if "all_horizons" in summary:
                risk = float(summary["all_horizons"]["mean"])
            else:
                risk = float(summary["relative_l2_all_channels_horizon_mean"])
            rows.append(
                {
                    "data": data,
                    "split": split,
                    "arm": match.group("arm"),
                    "K": K,
                    "H": H,
                    "seed": int(match.group("seed")),
                    "risk": risk,
                    "path": str(path),
                }
            )
    if not rows:
        raise FileNotFoundError(f"no fixed-horizon JSONs found below {root}")
    return rows


def aggregate(rows: Iterable[dict]) -> List[dict]:
    groups: Dict[Tuple[str, str, str, int, int], List[float]] = defaultdict(list)
    for row in rows:
        key = (row["data"], row["split"], row["arm"], row["K"], row["H"])
        groups[key].append(float(row["risk"]))
    result = []
    for (data, split, arm, K, H), values in sorted(groups.items()):
        array = np.asarray(values, dtype=np.float64)
        result.append(
            {
                "data": data,
                "split": split,
                "arm": arm,
                "K": K,
                "H": H,
                "n": int(array.size),
                "mean_risk": float(array.mean()),
                "sd_across_seeds": (
                    float(array.std(ddof=1)) if array.size > 1 else float("nan")
                ),
            }
        )
    return result


def curve(rows: Sequence[dict], data: str, split: str, arm: str) -> List[dict]:
    return sorted(
        [r for r in rows if r["data"] == data and r["split"] == split and r["arm"] == arm],
        key=lambda r: r["K"],
    )


def select_k(points: Sequence[dict], tolerance: float) -> Optional[dict]:
    if not points:
        return None
    best = min(points, key=lambda row: (row["mean_risk"], row["K"]))
    threshold = float(best["mean_risk"]) * (1.0 + float(tolerance))
    near = [row for row in points if row["mean_risk"] <= threshold]
    return {
        "best_K": int(best["K"]),
        "best_risk": float(best["mean_risk"]),
        "near_optimal_tolerance": float(tolerance),
        "largest_near_optimal_K": int(max(row["K"] for row in near)),
        "threshold": threshold,
    }


def risk_at(points: Sequence[dict], K: int) -> Optional[float]:
    matches = [row for row in points if int(row["K"]) == int(K)]
    return float(matches[0]["mean_risk"]) if len(matches) == 1 else None


def selection_summary(rows: Sequence[dict], tolerance: float) -> dict:
    output = {}
    for data in DATA_ORDER:
        exact_val = curve(rows, data, "val", "exact")
        dw_val = curve(rows, data, "val", "dw")
        if not exact_val and not dw_val:
            continue
        exact = select_k(exact_val, tolerance)
        dw = select_k(dw_val, tolerance)
        item = {"fixed_H": int((exact_val or dw_val)[0]["H"]), "exact": exact, "dw": dw}
        if exact is not None and dw is not None:
            item["best_K_shift_dw_minus_exact"] = int(dw["best_K"] - exact["best_K"])
            item["near_optimal_max_K_shift_dw_minus_exact"] = int(
                dw["largest_near_optimal_K"] - exact["largest_near_optimal_K"]
            )
            exact_test = curve(rows, data, "test", "exact")
            dw_test = curve(rows, data, "test", "dw")
            item["test_at_validation_selected_K"] = {
                "exact": risk_at(exact_test, exact["best_K"]),
                "dw": risk_at(dw_test, dw["best_K"]),
            }
        output[data] = item
    return output


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("data", "split", "arm", "K", "H", "n", "mean_risk", "sd_across_seeds")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("probe_outputs/internal_dw_fixed_horizon_positive_v1"),
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("probe_outputs/internal_dw_fixed_horizon_positive_v1/summary"),
    )
    parser.add_argument(
        "--near-optimal-tolerance", type=float, default=0.01,
        help="relative validation-risk tolerance used for the largest near-optimal K",
    )
    args = parser.parse_args()
    if args.near_optimal_tolerance < 0:
        parser.error("near-optimal-tolerance must be nonnegative")

    raw = read_rows(args.root)
    rows = aggregate(raw)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "fixed_horizon_curves.csv", rows)
    selection = selection_summary(rows, args.near_optimal_tolerance)
    (args.out_dir / "validation_selected_horizons.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("=== validation-selected training horizons ===")
    for data in DATA_ORDER:
        if data not in selection:
            continue
        item = selection[data]
        exact, dw = item.get("exact"), item.get("dw")
        if exact is None or dw is None:
            print(f"{data}: incomplete validation pair")
            continue
        print(
            f"{data}: H={item['fixed_H']}; "
            f"best K Exact={exact['best_K']}, DW={dw['best_K']} "
            f"(shift {item['best_K_shift_dw_minus_exact']:+d}); "
            f"largest K within {100 * args.near_optimal_tolerance:.1f}% "
            f"Exact={exact['largest_near_optimal_K']}, "
            f"DW={dw['largest_near_optimal_K']} "
            f"(shift {item['near_optimal_max_K_shift_dw_minus_exact']:+d})"
        )
    print(f"[out] {args.out_dir}")


if __name__ == "__main__":
    main()
