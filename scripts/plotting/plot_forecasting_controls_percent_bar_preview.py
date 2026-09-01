#!/usr/bin/env python3
"""Preview Fig. 6 as paired percent change from Exact BPTT."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from paper_figure_style import (
    COLOR_CLIP,
    COLOR_DW,
    COLOR_EXACT,
    COLOR_GRID,
    COLOR_JREG,
    COLOR_STATIC,
    COLOR_TBPTT,
    FONT_AXIS_LABEL,
    FONT_GROUP,
    FONT_LEGEND,
    FONT_TICK,
    LINE_AXIS,
    LINE_GRID,
    LINE_REFERENCE,
    TEXT_WIDTH_IN,
    apply_paper_style,
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
HATCHES = {
    "Clip": None,
    "JReg": None,
    "TBPTT": None,
    "Static": None,
    "Internal-DW": None,
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


def render(output_name: str = "forecasting_controls_percent_bar_preview") -> None:
    apply_paper_style()
    changes = paired_changes()
    figure, axis = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.18), dpi=220)

    centers = np.arange(len(DATASETS), dtype=float)
    bar_width = 0.13
    for dataset_index, dataset in enumerate(DATASETS):
        available = [method for method in METHODS if method in changes[dataset]]
        offsets = (np.arange(len(available)) - (len(available) - 1) / 2.0) * (bar_width + 0.012)
        for offset, method in zip(offsets, available):
            values = changes[dataset][method]
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1))
            alpha = 1.0 if method == "Internal-DW" else 0.72
            axis.bar(
                dataset_index + offset,
                mean,
                width=bar_width,
                color=COLORS[method],
                edgecolor=COLORS[method],
                linewidth=0.65,
                alpha=alpha,
                hatch=HATCHES[method],
                zorder=3,
            )
            axis.errorbar(
                dataset_index + offset,
                mean,
                yerr=sd,
                fmt="none",
                ecolor=COLORS[method],
                elinewidth=1.0 if method == "Internal-DW" else 0.8,
                capsize=1.8,
                capthick=0.8,
                alpha=alpha,
                zorder=4,
            )

    axis.axhline(
        0.0,
        color=COLOR_EXACT,
        linewidth=LINE_REFERENCE,
        linestyle=(0, (4, 3)),
        zorder=2,
    )
    axis.axvline(3.5, color="#B8BDC3", linewidth=0.75, zorder=1)
    axis.text(
        1.5,
        1.025,
        "history-dominated, weak drive",
        fontsize=FONT_GROUP,
        ha="center",
        va="bottom",
        transform=axis.get_xaxis_transform(),
        clip_on=False,
    )
    axis.text(
        5.5,
        1.025,
        "identified boundaries",
        fontsize=FONT_GROUP,
        ha="center",
        va="bottom",
        transform=axis.get_xaxis_transform(),
        clip_on=False,
    )
    axis.text(
        0.020,
        0.975,
        "better $\downarrow$",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_GROUP + 0.5,
        fontweight="bold",
        color="#555B62",
    )

    axis.set_xlim(-0.62, len(DATASETS) - 0.38)
    axis.set_ylim(-29.0, 50.0)
    axis.set_xticks(centers)
    axis.set_xticklabels(DATASETS)
    axis.set_yticks([-20, 0, 20, 40])
    axis.set_ylabel(
        r"$\Delta_{\rm Exact}$ relative $L_2$ (%)",
        fontsize=FONT_AXIS_LABEL,
        labelpad=3,
    )
    axis.tick_params(axis="both", labelsize=FONT_TICK, width=LINE_AXIS, length=2.4)
    axis.grid(axis="y", color=COLOR_GRID, linewidth=LINE_GRID, zorder=0)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#686E75")

    handles = [
        Line2D(
            [0],
            [0],
            color=COLOR_EXACT,
            linewidth=LINE_REFERENCE,
            linestyle=(0, (4, 3)),
            label="Exact BPTT",
        )
    ] + [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=COLORS[method],
            edgecolor=COLORS[method],
            alpha=1.0 if method == "Internal-DW" else 0.72,
            label=method,
        )
        for method in METHODS
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=6,
        frameon=False,
        fontsize=FONT_LEGEND,
        bbox_to_anchor=(0.53, 1.005),
        columnspacing=0.78,
        handlelength=0.8,
        handletextpad=0.35,
    )
    figure.subplots_adjust(left=0.10, right=0.995, top=0.78, bottom=0.17)

    output = ROOT / "figs" / output_name
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    figure.savefig(
        output.with_suffix(".png"),
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    plt.close(figure)


def main() -> None:
    render()


if __name__ == "__main__":
    main()
