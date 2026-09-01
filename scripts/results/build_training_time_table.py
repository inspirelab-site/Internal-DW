#!/usr/bin/env python3
"""Summarize matched Exact/Internal-DW timing runs.

Each run is reduced to the median of its post-warm-up epochs; dataset-level
overhead is then computed from paired repeat ratios on the same physical GPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


DATASET_ORDER = ["mg", "ettm1", "ettm2", "shear", "narma", "ieeg", "fmri", "wb2"]
DATASET_LABEL = {
    "mg": "Mackey--Glass",
    "ettm1": "ETTm1",
    "ettm2": "ETTm2",
    "shear": "Shear flow",
    "narma": "NARMA-5",
    "ieeg": "iEEG theta",
    "fmri": "Movie fMRI",
    "wb2": "WeatherBench-2",
}


def percentile(values: list[float], q: float) -> float:
    xs = sorted(values)
    if not xs:
        return math.nan
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def load_run(meta_path: Path, warmup_epochs: int) -> dict | None:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    log_path = meta_path.parent / "train_logs.jsonl"
    if not log_path.exists():
        return None
    rows = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(row.get("epoch", 0)) > warmup_epochs and "train/epoch_wall_seconds" in row:
            rows.append(row)
    if not rows:
        return None
    median = lambda key: statistics.median(float(r[key]) for r in rows if key in r)
    out = dict(meta)
    out.update(
        path=str(meta_path.parent),
        measured_epochs=len(rows),
        sec_per_epoch=median("train/epoch_wall_seconds"),
        sec_per_step=median("train/seconds_per_optimizer_step"),
        examples_per_sec=median("train/examples_per_second"),
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/internal_dw_compute_overhead_v2_nockpt")
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--appendix-out", default="docs/appendix_training_time.tex")
    args = ap.parse_args()
    root = Path(args.root)
    runs = [r for p in root.rglob("timing_meta.json") if (r := load_run(p, args.warmup_epochs))]
    if not runs:
        raise SystemExit(f"no completed timing logs under {root}")

    run_csv = root / "timing_runs.csv"
    with run_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0]))
        w.writeheader(); w.writerows(runs)

    datasets = sorted(
        {r["dataset"] for r in runs},
        key=lambda name: DATASET_ORDER.index(name) if name in DATASET_ORDER else len(DATASET_ORDER),
    )
    summaries = []
    for data in datasets:
        by = {(r["arm"], int(r["repeat"])): r for r in runs if r["dataset"] == data}
        repeats = sorted({rep for arm, rep in by if ("exact", rep) in by and ("dw", rep) in by})
        if not repeats:
            continue
        exact = [by[("exact", rep)] for rep in repeats]
        dw = [by[("dw", rep)] for rep in repeats]
        t_ratio = [d["sec_per_epoch"] / e["sec_per_epoch"] for e, d in zip(exact, dw)]
        row = {
            "dataset": data,
            "n_pairs": len(repeats),
            "exact_sec_per_epoch": statistics.median(r["sec_per_epoch"] for r in exact),
            "dw_sec_per_epoch": statistics.median(r["sec_per_epoch"] for r in dw),
            "time_overhead_pct_median": 100.0 * (statistics.median(t_ratio) - 1.0),
            "time_overhead_pct_q25": 100.0 * (percentile(t_ratio, .25) - 1.0),
            "time_overhead_pct_q75": 100.0 * (percentile(t_ratio, .75) - 1.0),
            "exact_ms_per_step": 1000.0 * statistics.median(r["sec_per_step"] for r in exact),
            "dw_ms_per_step": 1000.0 * statistics.median(r["sec_per_step"] for r in dw),
            "exact_examples_per_sec": statistics.median(r["examples_per_sec"] for r in exact),
            "dw_examples_per_sec": statistics.median(r["examples_per_sec"] for r in dw),
            "gpu": exact[0].get("gpu_name", ""),
        }
        summaries.append(row)

    print(f"root={root}")
    print("post-warm-up medians; paired Exact/DW repeats on the same GPU")
    print(f"{'data':8s} {'n':>2s} {'Exact s/ep':>11s} {'DW s/ep':>10s} {'time overhead [IQR]':>25s}")
    for r in summaries:
        print(
            f"{r['dataset']:8s} {r['n_pairs']:2d} {r['exact_sec_per_epoch']:11.3f} "
            f"{r['dw_sec_per_epoch']:10.3f} {r['time_overhead_pct_median']:+7.2f}% "
            f"[{r['time_overhead_pct_q25']:+.2f},{r['time_overhead_pct_q75']:+.2f}]"
        )

    if not summaries:
        raise SystemExit("timing runs exist, but no dataset has a complete matched Exact/DW pair")

    summary_csv = root / "timing_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0]))
        w.writeheader(); w.writerows(summaries)
    (root / "timing_summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")

    tex = root / "timing_summary.tex"
    with tex.open("w", encoding="utf-8") as f:
        f.write("% Same-GPU matched timing; median post-warm-up epoch per repeat.\n")
        f.write("\\begin{tabular}{lrrr}\n\\toprule\n")
        f.write("Data & Exact s/ep & DW s/ep & Overhead \\\\\n\\midrule\n")
        for r in summaries:
            f.write(
                f"{r['dataset']} & {r['exact_sec_per_epoch']:.2f} & {r['dw_sec_per_epoch']:.2f} & "
                f"{r['time_overhead_pct_median']:+.1f}\\% \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")

    appendix = Path(args.appendix_out)
    appendix.parent.mkdir(parents=True, exist_ok=True)
    with appendix.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated by scripts/results/build_training_time_table.py.\n")
        f.write("\\subsection{End-to-end training time}\n")
        f.write("\\label{app:training-time}\n\n")
        f.write(
            "We measure the practical wall-clock cost of the assigned Exact-BPTT and "
            "Internal-DW training configurations on the same NVIDIA H200 NVL.  Each "
            "matched pair uses the same dataset, model, training horizon, batch geometry, "
            "and optimizer, and is executed sequentially on the same physical GPU.  We "
            "run four epochs, discard the first epoch as CUDA and allocator warm-up, take "
            "the median of the remaining epoch times within each run, and report the "
            "median and interquartile range (IQR) over three paired repeats.  CUDA is "
            "explicitly synchronized at both timing boundaries.  The timer covers the "
            "training loop only, so validation and checkpoint I/O are excluded.\n\n"
        )
        f.write(
            "The Internal-DW measurements include its online covariance probes and "
            "small routewise solves.  Construction of any train-only domain-prior "
            "artifact is a one-time preprocessing operation and is excluded.  This is an "
            "end-to-end comparison of the implementations used in the forecasting "
            "experiments rather than an isolated solver microbenchmark.  Activation "
            "checkpointing is disabled in both arms, so the measured difference isolates "
            "the additional backward routing and online estimation used by Internal-DW.  "
            "We draw a wall-clock conclusion only and do not interpret these measurements "
            "as a memory comparison.\n\n"
        )
        f.write("\\begin{table}[t]\n")
        f.write("  \\centering\n")
        f.write("  \\caption{End-to-end training time on one NVIDIA H200 NVL.  Values are "
                "median seconds per post-warm-up epoch.  The final column is the median "
                "paired percentage change from Exact BPTT, with the paired IQR in "
                "brackets; negative values indicate faster training.}\n")
        f.write("  \\label{tab:training-time}\n")
        f.write("  \\small\n")
        f.write("  \\begin{tabular}{lrrr}\n")
        f.write("    \\toprule\n")
        f.write("    Data & Exact BPTT & Internal-DW & Change (\\%) \\\\\n")
        f.write("    \\midrule\n")
        for r in summaries:
            label = DATASET_LABEL.get(r["dataset"], r["dataset"])
            f.write(
                f"    {label} & {r['exact_sec_per_epoch']:.2f} & "
                f"{r['dw_sec_per_epoch']:.2f} & {r['time_overhead_pct_median']:+.1f} "
                f"[{r['time_overhead_pct_q25']:+.1f}, {r['time_overhead_pct_q75']:+.1f}] \\\\\n"
            )
        f.write("    \\bottomrule\n")
        f.write("  \\end{tabular}\n")
        f.write("\\end{table}\n\n")
        f.write(
            "Across the completed testbeds, Internal-DW changes median epoch time by "
            f"{min(r['time_overhead_pct_median'] for r in summaries):+.1f}\\% to "
            f"{max(r['time_overhead_pct_median'] for r in summaries):+.1f}\\%.  Thus, "
            "these matched measurements quantify the net runtime cost of online "
            "estimation and routewise gradient scaling without activation-recomputation "
            "as a confound.\n"
        )
    print(f"[out] {run_csv}")
    print(f"[out] {summary_csv}")
    print(f"[out] {tex}")
    print(f"[out] {appendix}")


if __name__ == "__main__":
    main()
