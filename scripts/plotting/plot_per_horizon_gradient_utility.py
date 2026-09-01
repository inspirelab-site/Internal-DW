#!/usr/bin/env python3
"""Plot per-horizon gradient magnitude and held-out utility.

Each curve uses gradients already measured by
``probe_heldout_delayed_gradient_utility.py`` at a frozen Exact-BPTT
checkpoint.  The quantity at x=k is the gradient produced by loss k alone;
it is not the cumulative BPTT gradient.  Exact BPTT later sums these vectors
over k.

Left axis:
    A(k) = ||g_k^A|| / ||g_1^A||

Right axis:
    U(k) = <g_k^A / ||g_k^A||, (1/K) sum_j g_j^B>

U(k) is the existing first-order held-out utility, not an SNR estimate.  Its
scale is dataset dependent, so the overview is intended to compare signs and
within-dataset trends rather than utility magnitudes across datasets.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from paper_figure_style import (
    BAND_ALPHA,
    COLOR_GRID,
    COLOR_MAGNITUDE,
    COLOR_SPINE,
    COLOR_UTILITY,
    FONT_DENSE_LABEL,
    FONT_DENSE_LEGEND,
    FONT_DENSE_TICK,
    FONT_DENSE_TITLE,
    LINE_AXIS,
    LINE_GRID,
    LINE_MAIN,
    LINE_REFERENCE,
    LINE_SECONDARY,
    TEXT_WIDTH_IN,
    apply_paper_style,
)


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title: str
    path: Path
    schema: str = "standard"


DATASETS = (
    DatasetSpec(
        "mg",
        "MG",
        ROOT / "probe_outputs" / "real_heldout_gradient_utility_v1" / "mg_seed0.json",
    ),
    DatasetSpec(
        "narma",
        "NARMA-5",
        ROOT / "probe_outputs" / "real_heldout_gradient_utility_v1" / "narma_seed0.json",
    ),
    DatasetSpec(
        "ettm1",
        "ETTm1",
        ROOT
        / "probe_outputs"
        / "ettm_internal_dw_diagnostics_v1"
        / "ettm1"
        / "delayed_exact_seed0.json",
    ),
    DatasetSpec(
        "ettm2",
        "ETTm2",
        ROOT
        / "probe_outputs"
        / "ettm_internal_dw_diagnostics_v1"
        / "ettm2"
        / "delayed_exact_seed0.json",
    ),
    DatasetSpec(
        "ieeg",
        "iEEG",
        ROOT / "probe_outputs" / "real_heldout_gradient_utility_v1" / "ieeg_seed0.json",
    ),
    DatasetSpec(
        "fmri",
        "fMRI",
        ROOT / "probe_outputs" / "real_heldout_gradient_utility_v1" / "fmri_seed0.json",
    ),
    DatasetSpec(
        "shear",
        "Shear",
        ROOT / "probe_outputs" / "real_heldout_gradient_utility_v1" / "shear_seed0.json",
    ),
    DatasetSpec(
        "wb2",
        "WB2",
        ROOT
        / "probe_outputs"
        / "real_heldout_gradient_utility_wb2_n8_v1"
        / "wb2_seed0.json",
    ),
)


@dataclass(frozen=True)
class Curves:
    horizon: np.ndarray
    amplitude_median: np.ndarray
    amplitude_q10: np.ndarray
    amplitude_q90: np.ndarray
    utility_median: np.ndarray
    utility_q25: np.ndarray
    utility_q75: np.ndarray
    pairs: int
    negative_fraction: float


def _array(summary: dict, metric: str, statistic: str) -> np.ndarray:
    return np.asarray(summary[metric][statistic], dtype=float)


def load_curves(spec: DatasetSpec) -> Curves:
    if not spec.path.is_file():
        raise FileNotFoundError(f"missing probe output: {spec.path}")
    payload = json.loads(spec.path.read_text(encoding="utf-8"))
    horizon = np.asarray(payload["horizons"], dtype=int)
    if spec.schema == "standard":
        summary = payload["summary"]
        amplitude_metric = "full_amplitude_H1"
        utility_metric = "full_window_utility"
        utilities = np.asarray(
            [record["metrics"][utility_metric] for record in payload["pair_records"]],
            dtype=float,
        )
    elif spec.schema == "open_arm":
        summary = payload["arms"]["open"]["summary"]
        amplitude_metric = "full_amplitude_H1"
        utility_metric = "full_window_utility_to_open_target"
        utilities = np.asarray(
            [
                record["metrics_by_arm"]["open"][utility_metric]
                for record in payload["pair_records"]
            ],
            dtype=float,
        )
    else:
        raise ValueError(f"unsupported schema {spec.schema!r} for {spec.key}")
    utility_median = _array(summary, utility_metric, "median")
    return Curves(
        horizon=horizon,
        amplitude_median=_array(summary, amplitude_metric, "median"),
        amplitude_q10=_array(summary, amplitude_metric, "q10"),
        amplitude_q90=_array(summary, amplitude_metric, "q90"),
        utility_median=utility_median,
        utility_q25=np.quantile(utilities, 0.25, axis=0),
        utility_q75=np.quantile(utilities, 0.75, axis=0),
        pairs=int(payload["num_pairs"]),
        negative_fraction=float(np.mean(utility_median < 0.0)),
    )


def _nice_bound(value: float) -> float:
    if not np.isfinite(value) or value <= 1e-12:
        return 1.0
    exponent = 10.0 ** np.floor(np.log10(value))
    scaled = value / exponent
    if scaled <= 1.0:
        nice = 1.0
    elif scaled <= 2.0:
        nice = 2.0
    elif scaled <= 5.0:
        nice = 5.0
    else:
        nice = 10.0
    return float(nice * exponent)


def utility_limits(curves: Curves) -> tuple[float, float, float]:
    values = np.concatenate(
        [curves.utility_q25, curves.utility_q75, curves.utility_median, [0.0]]
    )
    bound = _nice_bound(float(np.nanmax(np.abs(values))))
    return -1.05 * bound, 1.05 * bound, bound


def draw_panel(
    axis: plt.Axes,
    spec: DatasetSpec,
    curves: Curves,
    *,
    show_left_ticks: bool,
    show_right_ticks: bool,
    amplitude_ylim: tuple[float, float],
    show_axis_labels: bool,
    compact: bool,
) -> tuple[Line2D, Line2D]:
    magnitude_color = COLOR_MAGNITUDE
    utility_color = COLOR_UTILITY
    harmful_color = "#D97706"
    grid_color = COLOR_GRID

    k = curves.horizon
    axis.fill_between(
        k,
        np.maximum(curves.amplitude_q10, 1e-12),
        np.maximum(curves.amplitude_q90, 1e-12),
        color=magnitude_color,
        alpha=BAND_ALPHA,
        linewidth=0,
        zorder=1,
    )
    (amplitude_line,) = axis.plot(
        k,
        curves.amplitude_median,
        color=magnitude_color,
        linewidth=LINE_MAIN,
        zorder=3,
    )
    axis.axhline(
        1.0,
        color="#7D848C",
        linewidth=LINE_REFERENCE,
        linestyle=(0, (3.0, 2.0)),
        zorder=0,
    )
    axis.set_yscale("log")
    axis.set_ylim(*amplitude_ylim)
    axis.set_xlim(float(k[0]), float(k[-1]))
    if compact:
        tick_values = (int(k[0]), int(k[-1]))
        axis.set_xticks(tick_values, [str(value) for value in tick_values])
    else:
        middle = int(round((int(k[0]) + int(k[-1])) / 2.0))
        tick_values = (int(k[0]), middle, int(k[-1]))
        axis.set_xticks(tick_values, [str(value) for value in tick_values])
    axis.set_title(
        spec.title
        + "\n"
        + rf"$U<0$: {100.0 * curves.negative_fraction:.0f}%",
        pad=3.0,
        fontweight="normal",
    )
    axis.grid(axis="x", color=grid_color, linewidth=LINE_GRID, alpha=0.75)
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", width=LINE_AXIS, length=2.4, pad=1.5)
    axis.tick_params(
        axis="y",
        colors=magnitude_color,
        width=LINE_AXIS,
        length=2.4 if show_left_ticks else 0,
        labelleft=show_left_ticks,
        pad=1.5,
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(magnitude_color)
    axis.spines["bottom"].set_color(COLOR_SPINE)
    axis.spines["left"].set_linewidth(LINE_AXIS)
    axis.spines["bottom"].set_linewidth(LINE_AXIS)

    utility_axis = axis.twinx()
    utility_low, utility_high, utility_tick = utility_limits(curves)
    utility_axis.set_ylim(utility_low, utility_high)
    utility_axis.axhspan(
        utility_low,
        0.0,
        color=harmful_color,
        alpha=0.045,
        linewidth=0,
        zorder=0,
    )
    utility_axis.fill_between(
        k,
        curves.utility_q25,
        curves.utility_q75,
        color=utility_color,
        alpha=BAND_ALPHA,
        linewidth=0,
        zorder=1,
    )
    (utility_line,) = utility_axis.plot(
        k,
        curves.utility_median,
        color=utility_color,
        linewidth=LINE_MAIN,
        zorder=4,
    )
    utility_axis.axhline(
        0.0,
        color=utility_color,
        linewidth=LINE_REFERENCE,
        linestyle=(0, (2.2, 2.0)),
        alpha=0.8,
        zorder=2,
    )
    utility_axis.tick_params(
        axis="y",
        colors=utility_color,
        width=LINE_AXIS,
        length=2.4 if show_right_ticks else 0,
        labelright=show_right_ticks,
        pad=1.5,
    )
    utility_axis.spines["top"].set_visible(False)
    utility_axis.spines["left"].set_visible(False)
    utility_axis.spines["right"].set_color(utility_color)
    utility_axis.spines["right"].set_linewidth(LINE_AXIS)
    utility_axis.spines["bottom"].set_visible(False)
    utility_axis.set_yticks((0.0,) if compact else (-utility_tick, 0.0, utility_tick))

    if show_axis_labels:
        axis.set_ylabel(r"$A(k)$", color=magnitude_color)
        utility_axis.set_ylabel(
            r"$U(k)$",
            color=utility_color,
            rotation=-90,
            labelpad=13,
        )

    return amplitude_line, utility_line


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-horizon full-gradient magnitude and held-out utility."
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated dataset keys or 'all'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "figs" / "per_horizon_gradient_utility.pdf",
        help="Output PDF path; a PNG preview is written beside it.",
    )
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()

    requested = (
        {spec.key for spec in DATASETS}
        if args.datasets.strip().lower() == "all"
        else {item.strip().lower() for item in args.datasets.split(",") if item.strip()}
    )
    selected = [spec for spec in DATASETS if spec.key in requested]
    unknown = requested - {spec.key for spec in DATASETS}
    if unknown:
        raise ValueError(f"unknown dataset keys: {sorted(unknown)}")
    if not selected:
        raise ValueError("no datasets selected")

    loaded = [(spec, load_curves(spec)) for spec in selected]
    amplitude_high = max(
        float(np.nanmax(curves.amplitude_q90)) for _, curves in loaded
    )
    amplitude_ylim = (0.45, max(2.0, 1.10 * amplitude_high))

    apply_paper_style()
    plt.rcParams.update(
        {
            "font.size": FONT_DENSE_LABEL,
            "axes.titlesize": FONT_DENSE_TITLE,
            "axes.labelsize": FONT_DENSE_LABEL,
            "xtick.labelsize": FONT_DENSE_TICK,
            "ytick.labelsize": FONT_DENSE_TICK,
            "legend.fontsize": FONT_DENSE_LEGEND,
        }
    )

    single_panel = len(loaded) == 1
    if single_panel:
        rows, columns = 1, 1
        figure_size = (3.25, 2.20)
    else:
        columns = len(loaded)
        rows = 1
        figure_size = (TEXT_WIDTH_IN, 1.62)

    figure, axes_raw = plt.subplots(rows, columns, figsize=figure_size, squeeze=False)
    axes = list(axes_raw.flat)
    if single_panel:
        figure.subplots_adjust(
            left=0.205,
            right=0.795,
            bottom=0.225,
            top=0.715,
        )
    else:
        figure.subplots_adjust(
            left=0.070,
            right=0.940,
            bottom=0.215,
            top=0.70,
            wspace=0.31,
        )

    magnitude_line = utility_line = None
    for index, (axis, (spec, curves)) in enumerate(zip(axes, loaded)):
        row = index // columns
        column = index % columns
        magnitude_line, utility_line = draw_panel(
            axis,
            spec,
            curves,
            show_left_ticks=(column == 0),
            show_right_ticks=(
                single_panel or column == columns - 1 or index == len(loaded) - 1
            ),
            amplitude_ylim=amplitude_ylim,
            show_axis_labels=single_panel,
            compact=not single_panel,
        )
        if single_panel and row == rows - 1:
            axis.set_xlabel("forecast step $k$")

    for axis in axes[len(loaded) :]:
        axis.remove()

    if not single_panel:
        figure.text(
            0.012,
            0.46,
            r"$A(k)$",
            rotation=90,
            va="center",
            ha="left",
            color="#30343B",
            fontsize=FONT_DENSE_LABEL,
        )
        figure.text(
            0.988,
            0.46,
            r"$U(k)$",
            rotation=-90,
            va="center",
            ha="right",
            color="#6D4BB3",
            fontsize=FONT_DENSE_LABEL,
        )
        figure.text(
            0.505,
            -0.008,
            "forecast step $k$",
            va="bottom",
            ha="center",
            fontsize=FONT_DENSE_LABEL,
        )
    figure.legend(
        [magnitude_line, utility_line],
        [r"per-horizon magnitude $A(k)$", r"held-out utility $U(k)$"],
        loc="upper center",
        bbox_to_anchor=(0.50, 0.985),
        ncol=2,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.5,
    )

    output_pdf = args.output.with_suffix(".pdf")
    output_png = args.output.with_suffix(".png")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(
        output_png,
        dpi=args.dpi,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    plt.close(figure)
    for spec, curves in loaded:
        print(
            f"[{spec.key}] K={int(curves.horizon[-1])} pairs={curves.pairs} "
            f"A(K)={curves.amplitude_median[-1]:.4g} "
            f"harmful_U_fraction={curves.negative_fraction:.3f}"
        )
    print(f"[out] {output_pdf}")
    print(f"[out] {output_png}")


if __name__ == "__main__":
    main()
