#!/usr/bin/env python3
"""Export the seed-level relative-L2 values underlying paper Figure 6."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.reproduce.train_test_ieeg import collect_cohort, result_root
OUT = ROOT / "probe_outputs" / "figure6_raw_metrics_v1"
METHOD_ORDER = ["Exact BPTT", "Clip", "JReg", "TBPTT", "Static", "Internal-DW"]
FAVORABLE = ["MG", "ETTm1", "ETTm2", "Shear"]
BOUNDARIES = ["NARMA-5", "iEEG", "fMRI", "WB2"]
DATASETS = FAVORABLE + BOUNDARIES
DISPLAY = {
    "MG": "Mackey--Glass",
    "ETTm1": "ETTm1",
    "ETTm2": "ETTm2",
    "Shear": "Shear flow",
    "NARMA-5": "NARMA-5",
    "iEEG": "iEEG theta",
    "fMRI": "Movie fMRI",
    "WB2": "WeatherBench-2",
}
SOURCES = {
    "MG": {
        "Exact BPTT": "probe_outputs/dense_multistart_rel_l2_1p5k_v1/mg/exact_seed*.json",
        "Internal-DW": "probe_outputs/dense_multistart_rel_l2_1p5k_v1/mg/dw_seed*.json",
        "Clip": "probe_outputs/internal_dw_AB_dense_1p5k_v1/A/mg/clip_K32_seed*.json",
        "JReg": "probe_outputs/internal_dw_AB_dense_1p5k_v1/A/mg/jreg_K32_seed*.json",
        "TBPTT": "probe_outputs/tbptt_dense_multistart_rel_l2_1p5k_v1/mg/tbptt8_seed*.json",
        "Static": "probe_outputs/internal_dw_static_positive_v1/dense/mg/static_c0.6_seed*.json",
    },
    "ETTm1": {
        "Exact BPTT": "probe_outputs/temporal_candidate_dense_1p5k_v1/ettm1/exact_seed*.json",
        "Internal-DW": "probe_outputs/temporal_candidate_dense_1p5k_v1/ettm1/generic_seed*.json",
        "Clip": "probe_outputs/internal_dw_AB_dense_1p5k_v1/A/ettm1/clip_K64_seed*.json",
        "JReg": "probe_outputs/internal_dw_AB_dense_1p5k_v1/A/ettm1/jreg_K64_seed*.json",
        "TBPTT": "probe_outputs/tbptt_positive_sweep_v1/dense_candidates/ettm1/tbptt16_seed*.json",
        "Static": "probe_outputs/internal_dw_static_positive_v1/dense/ettm1/static_c0.3_seed*.json",
    },
    "ETTm2": {
        "Exact BPTT": "probe_outputs/temporal_candidate_dense_1p5k_v1/ettm2/exact_seed*.json",
        "Internal-DW": "probe_outputs/temporal_candidate_dense_1p5k_v1/ettm2/generic_seed*.json",
        "Clip": "probe_outputs/internal_dw_AB_dense_1p5k_v1/A/ettm2/clip_K64_seed*.json",
        "JReg": "probe_outputs/internal_dw_AB_dense_1p5k_v1/A/ettm2/jreg_K64_seed*.json",
        "TBPTT": "probe_outputs/tbptt_positive_sweep_v1/dense_candidates/ettm2/tbptt32_seed*.json",
        "Static": "probe_outputs/internal_dw_static_positive_v1/dense/ettm2/static_c0.6_seed*.json",
    },
    "Shear": {
        "Exact BPTT": "probe_outputs/dense_multistart_rel_l2_1p5k_v1/shear/exact_seed*.json",
        "Internal-DW": "probe_outputs/dense_multistart_rel_l2_1p5k_v1/shear/dw_seed*.json",
        "Clip": "probe_outputs/internal_dw_AB_dense_1p5k_v1/A/shear/clip_K32_seed*.json",
        "JReg": "probe_outputs/internal_dw_AB_dense_1p5k_v1/A/shear/jreg_K32_seed*.json",
        "TBPTT": "probe_outputs/tbptt_dense_multistart_rel_l2_1p5k_v1/shear/tbptt8_seed*.json",
        "Static": "probe_outputs/internal_dw_static_positive_v1/dense/shear/static_c0.3_seed*.json",
    },
    "NARMA-5": {
        "Exact BPTT": "probe_outputs/dense_multistart_rel_l2_1p5k_v1/narma/exact_seed*.json",
        "Internal-DW": "probe_outputs/dense_multistart_rel_l2_1p5k_v1/narma/dw_seed*.json",
        "Clip": "probe_outputs/internal_dw_AB_dense_1p5k_v1/A/narma/clip_K32_seed*.json",
        "JReg": "probe_outputs/internal_dw_AB_dense_1p5k_v1/A/narma/jreg_K32_seed*.json",
    },
    "fMRI": {
        "Exact BPTT": "probe_outputs/dense_multistart_rel_l2_1p5k_v1/fmri/exact_seed*.json",
        "Internal-DW": "probe_outputs/dense_multistart_rel_l2_1p5k_v1/fmri/dw_seed*.json",
        "Clip": "probe_outputs/internal_dw_AB_dense_1p5k_v1/A/fmri/clip_K64_seed*.json",
        "JReg": "probe_outputs/internal_dw_AB_dense_1p5k_v1/A/fmri/jreg_K64_seed*.json",
    },
    "WB2": {
        "Exact BPTT": "probe_outputs/dense_multistart_rel_l2_1p5k_v1/wb2/exact_seed*.json",
        "Internal-DW": "probe_outputs/dense_multistart_rel_l2_1p5k_v1/wb2/dw_seed*.json",
        "Clip": "probe_outputs/internal_dw_AB_dense_1p5k_v1/A/wb2/clip_K48_seed*.json",
        "JReg": "probe_outputs/internal_dw_AB_dense_1p5k_v1/A/wb2/jreg_K48_seed*.json",
    },
}
SEED_RE = re.compile(r"seed(?P<seed>\d+)")


def read_value(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    if "all_horizons" in summary:
        return float(summary["all_horizons"]["mean"])
    return float(summary["relative_l2_all_horizons_mean"])


def read_seed_values(pattern: str) -> tuple[list[float], list[str]]:
    by_seed: dict[int, tuple[float, str]] = {}
    for path in ROOT.glob(pattern):
        match = SEED_RE.search(path.name)
        if match is None:
            raise RuntimeError(f"Cannot read seed from {path}")
        seed = int(match.group("seed"))
        if seed in by_seed:
            raise RuntimeError(f"Duplicate seed {seed} for {pattern}")
        by_seed[seed] = (read_value(path), str(path.relative_to(ROOT)))
    if sorted(by_seed) != [0, 1, 2]:
        raise RuntimeError(f"Expected seeds 0,1,2 for {pattern}; found {sorted(by_seed)}")
    return [by_seed[s][0] for s in range(3)], [by_seed[s][1] for s in range(3)]


def collect() -> list[dict]:
    rows: list[dict] = []
    for dataset in DATASETS:
        if dataset == "iEEG":
            for r in collect_cohort():
                rows.append(dict(dataset=dataset, regime="identified boundary", method=r['method'],
                                 metric="subject-mean dense 1:96 relative L2",
                                 seed0=r['per_seed'][0], seed1=r['per_seed'][1], seed2=r['per_seed'][2],
                                 mean=r['mean'], sample_sd=r['sample_sd'],
                                 figure6_percent_change_from_exact_mean=r['mean_percent'],
                                 percent_change_aggregation="mean of participant-seed paired changes",
                                 source_glob=str(result_root() / 'sub-*' / r['arm'] / 'seed*/test.json'),
                                 source_jsons=r['source_jsons']))
            continue
        exact_values, _ = read_seed_values(SOURCES[dataset]["Exact BPTT"])
        exact_mean = mean(exact_values)
        for method in METHOD_ORDER:
            if method not in SOURCES[dataset]:
                continue
            values, paths = read_seed_values(SOURCES[dataset][method])
            current_mean = mean(values)
            rows.append(
                {
                    "dataset": dataset,
                    "regime": (
                        "history-dominated, weak drive"
                        if dataset in FAVORABLE
                        else "identified boundary"
                    ),
                    "method": method,
                    "metric": "dense 1:ceil(1.5K) relative L2",
                    "seed0": values[0],
                    "seed1": values[1],
                    "seed2": values[2],
                    "mean": current_mean,
                    "sample_sd": stdev(values),
                    "figure6_percent_change_from_exact_mean": (
                        0.0
                        if method == "Exact BPTT"
                        else 100.0 * (current_mean - exact_mean) / exact_mean
                    ),
                    "source_glob": SOURCES[dataset][method],
                    "source_jsons": paths,
                }
            )
    return rows


def write_csv(rows: list[dict]) -> None:
    fields = [
        "dataset", "regime", "method", "metric", "seed0", "seed1", "seed2",
        "mean", "sample_sd", "figure6_percent_change_from_exact_mean",
        "source_glob",
    ]
    with (OUT / "figure6_seed_level_relative_l2.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def cell(row: dict | None, best: bool) -> str:
    if row is None:
        return "--"
    value = f"{row['mean']:.5f} $\\pm$ {row['sample_sd']:.5f}"
    return f"\\textbf{{{value}}}" if best else value


def panel(rows: list[dict], datasets: list[str], methods: list[str]) -> list[str]:
    lookup = {(row["dataset"], row["method"]): row for row in rows}
    lines = [
        "  Dataset & " + " & ".join(methods) + r" \\ ",
        r"  \midrule",
    ]
    for dataset in datasets:
        present = [lookup[(dataset, method)] for method in methods if (dataset, method) in lookup]
        best_mean = min(row["mean"] for row in present)
        values = [
            cell(lookup.get((dataset, method)), lookup.get((dataset, method), {}).get("mean") == best_mean)
            for method in methods
        ]
        lines.append(f"  {DISPLAY[dataset]} & " + " & ".join(values) + r" \\ ")
    return lines


def write_tex(rows: list[dict]) -> None:
    methods_a = METHOD_ORDER
    methods_b = ["Exact BPTT", "Clip", "JReg", "Internal-DW"]
    text = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\caption{\textbf{Absolute forecasting errors underlying Figure~\ref{fig:assigned-estimator-performance}.} Values are dense $1{:}\lceil1.5K\rceil$ relative $L_2$ (mean $\pm$ sample standard deviation over three matched seeds); lower is better. For iEEG, each seed averages 16 participants equally and percentage changes are paired within participant and seed; other datasets use the corresponding full-BPTT mean as reference. Bold marks the lowest mean in each row.}",
        r"\label{tab:forecasting-controls-absolute}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"\multicolumn{7}{l}{\emph{History-dominated, weak-drive regime}} \\ ",
    ]
    text.extend(panel(rows, FAVORABLE, methods_a))
    text.extend(
        [
            r"  \bottomrule",
            r"\end{tabular}",
            r"\vspace{3pt}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"\multicolumn{5}{l}{\emph{Identified boundaries}} \\ ",
        ]
    )
    text.extend(panel(rows, BOUNDARIES, methods_b))
    text.extend([r"  \bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (OUT / "figure6_absolute_relative_l2_table.tex").write_text(
        "\n".join(text), encoding="utf-8"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = collect()
    write_csv(rows)
    write_tex(rows)
    (OUT / "figure6_raw_metrics_ledger.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print("dataset     method           seed0     seed1     seed2        mean+-sd")
    for row in rows:
        print(
            f"{row['dataset']:<11} {row['method']:<14} "
            f"{row['seed0']:.5f}  {row['seed1']:.5f}  {row['seed2']:.5f}  "
            f"{row['mean']:.5f}+-{row['sample_sd']:.5f}"
        )
    print(f"[out] {OUT}")


if __name__ == "__main__":
    main()
