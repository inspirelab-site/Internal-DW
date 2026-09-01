"""Load the matched-seed forecasting changes consumed by Figure 6."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from paper_figure_style import (
    COLOR_CLIP,
    COLOR_DW,
    COLOR_JREG,
    COLOR_STATIC,
    COLOR_TBPTT,
)


ROOT = Path(__file__).resolve().parents[2]
METHODS = ["Clip", "JReg", "TBPTT", "Static", "Internal-DW"]
COLORS = {
    "Clip": COLOR_CLIP,
    "JReg": COLOR_JREG,
    "TBPTT": COLOR_TBPTT,
    "Static": COLOR_STATIC,
    "Internal-DW": COLOR_DW,
}
DATASETS = ["MG", "ETTm1", "ETTm2", "Shear", "NARMA-5", "iEEG", "fMRI", "WB2"]


def _means(pattern: str) -> np.ndarray:
    paths = sorted(ROOT.glob(pattern))
    if len(paths) != 3:
        raise RuntimeError(f"Expected three matched seeds for {pattern}, found {len(paths)}")
    values = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        summary = payload["summary"]
        if "all_horizons" in summary:
            value = summary["all_horizons"]["mean"]
        else:
            # WeatherBench-2 uses the legacy dense-evaluation schema.
            value = summary["relative_l2_all_horizons_mean"]
        values.append(float(value))
    return np.asarray(values, dtype=float)


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
    "iEEG": {
        "Exact BPTT": "probe_outputs/dense_multistart_rel_l2_1p5k_v1/ieeg/exact_seed*.json",
        "Internal-DW": "probe_outputs/dense_multistart_rel_l2_1p5k_v1/ieeg/dw_seed*.json",
        "Clip": "probe_outputs/internal_dw_AB_dense_1p5k_v1/A/ieeg/clip_K64_seed*.json",
        "JReg": "probe_outputs/internal_dw_AB_dense_1p5k_v1/A/ieeg/jreg_K64_seed*.json",
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


def paired_changes() -> dict[str, dict[str, np.ndarray]]:
    changes: dict[str, dict[str, np.ndarray]] = {}
    for dataset in DATASETS:
        exact = _means(SOURCES[dataset]["Exact BPTT"])
        exact_mean = float(np.mean(exact))
        changes[dataset] = {}
        for method in METHODS:
            if method not in SOURCES[dataset]:
                continue
            method_values = _means(SOURCES[dataset][method])
            # Express each seed relative to the shared Exact-BPTT mean.  The
            # bar height is therefore exactly
            # 100 * (mean(method) - mean(exact)) / mean(exact), while the
            # error bar shows the across-seed variation of that method on the
            # same relative scale.
            changes[dataset][method] = 100.0 * (method_values - exact_mean) / exact_mean
    return changes
