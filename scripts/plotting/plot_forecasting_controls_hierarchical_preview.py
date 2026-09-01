#!/usr/bin/env python3
"""Preview Fig. 6 with the positive regimes visually prioritized."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from paper_figure_style import (
    COLOR_DW,
    COLOR_EXACT,
    COLOR_GRID,
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
from forecasting_control_metrics import (
    COLORS,
    METHODS,
    paired_changes,
)


ROOT = Path(__file__).resolve().parents[2]
PRIMARY = ["MG", "ETTm1", "ETTm2", "Shear"]
BOUNDARIES = ["NARMA-5", "fMRI", "iEEG", "WB2"]


def _draw_bar(axis, x, mean, sd, method, width, emphasize=False):
    # Match the paper's existing bars (e.g. Fig. 4c): simple flat fills from
    # the shared palette, without a colored outline or hollow-box treatment.
    axis.bar(
        x,
        mean,
        width=width,
        color=COLORS[method],
        edgecolor="none",
        linewidth=0.0,
        alpha=0.90,
        zorder=3,
    )
    axis.errorbar(
        x,
        mean,
        yerr=sd,
        fmt="none",
        ecolor=COLORS[method],
        elinewidth=1.0 if emphasize else 0.8,
        capsize=2.0 if emphasize else 1.7,
        capthick=0.8,
        alpha=0.95,
        zorder=4,
    )


def _style_axis(axis):
    axis.grid(axis="y", color=COLOR_GRID, linewidth=LINE_GRID, zorder=0)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#686E75")
    axis.tick_params(
        axis="both", labelsize=FONT_TICK, width=LINE_AXIS, length=2.3
    )


def main() -> None:
    apply_paper_style()
    changes = paired_changes()

    figure = plt.figure(figsize=(TEXT_WIDTH_IN, 2.78), dpi=220)
    outer = figure.add_gridspec(
        1,
        2,
        width_ratios=(2.08, 1.0),
        left=0.09,
        right=0.995,
        top=0.72,
        bottom=0.14,
        wspace=0.22,
    )
    primary_grid = outer[0].subgridspec(2, 2, wspace=0.30, hspace=0.34)
    boundary_grid = outer[1].subgridspec(2, 2, wspace=0.62, hspace=0.48)
    main_axes = [
        figure.add_subplot(primary_grid[0, 0]),
        figure.add_subplot(primary_grid[0, 1]),
        figure.add_subplot(primary_grid[1, 0]),
        figure.add_subplot(primary_grid[1, 1]),
    ]
    small_axes = [
        figure.add_subplot(boundary_grid[0, 0]),
        figure.add_subplot(boundary_grid[0, 1]),
        figure.add_subplot(boundary_grid[1, 0]),
        figure.add_subplot(boundary_grid[1, 1]),
    ]

    # The four regimes supporting the central claim occupy larger independent
    # axes.  Local vertical scales keep their distinct effect sizes legible;
    # showing ticks on every panel makes those scale changes explicit.
    bar_width = 0.61
    for axis, dataset in zip(main_axes, PRIMARY):
        available = [method for method in METHODS if method in changes[dataset]]
        lower = [0.0]
        upper = [0.0]
        for method_index, method in enumerate(available):
            values = changes[dataset][method]
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1))
            lower.append(mean - sd)
            upper.append(mean + sd)
            emphasize = method == "Internal-DW"
            _draw_bar(
                axis, method_index, mean, sd, method, bar_width,
                emphasize=emphasize,
            )
            if emphasize:
                axis.annotate(
                    f"{mean:.1f}",
                    xy=(method_index, mean),
                    xytext=(8, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=FONT_TICK,
                    fontweight="bold",
                    color=COLOR_DW,
                    clip_on=False,
                )

        axis.axhline(
            0.0,
            color=COLOR_EXACT,
            linewidth=LINE_REFERENCE,
            linestyle=(0, (4, 3)),
            zorder=2,
        )
        data_low = min(lower)
        data_high = max(upper)
        span = max(data_high - data_low, 4.0)
        axis.set_ylim(data_low - 0.16 * span, data_high + 0.18 * span)
        axis.set_xlim(-0.68, len(available) - 0.02)
        axis.set_xticks([])
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.set_title(dataset, fontsize=FONT_GROUP + 0.1, pad=2.0)
        _style_axis(axis)

    main_axes[0].text(
        -0.36,
        -5.0,
        "better $\downarrow$",
        transform=main_axes[0].transData,
        ha="left",
        va="bottom",
        fontsize=FONT_GROUP,
        fontweight="bold",
        color="#555B62",
    )

    # Boundary cases are retained as compact, shared-scale small multiples.
    boundary_methods = ["Clip", "JReg", "Internal-DW"]
    boundary_extents = []
    for panel_index, (axis, dataset) in enumerate(zip(small_axes, BOUNDARIES)):
        lower = [0.0]
        upper = [0.0]
        for method_index, method in enumerate(boundary_methods):
            values = changes[dataset][method]
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1))
            lower.append(mean - sd)
            upper.append(mean + sd)
            _draw_bar(
                axis,
                method_index,
                mean,
                sd,
                method,
                0.54,
                emphasize=method == "Internal-DW",
            )
        axis.axhline(
            0.0,
            color=COLOR_EXACT,
            linewidth=LINE_REFERENCE,
            linestyle=(0, (4, 3)),
            zorder=2,
        )
        axis.set_title(dataset, fontsize=FONT_GROUP, pad=2.0)
        axis.set_xlim(-0.65, 2.65)
        axis.set_xticks([])
        if panel_index % 2 == 1:
            axis.tick_params(axis="y", labelleft=False)
        _style_axis(axis)
        boundary_extents.append((min(lower), max(upper)))

    # Each boundary row shares a scale: NARMA-5/fMRI require a broad range,
    # whereas iEEG/WB2 can use a tighter range without exaggerating differences
    # between the two neighboring datasets.
    for row_index in range(2):
        row_slice = slice(2 * row_index, 2 * row_index + 2)
        row_extents = boundary_extents[row_slice]
        data_low = min(extent[0] for extent in row_extents)
        data_high = max(extent[1] for extent in row_extents)
        span = max(data_high - data_low, 1.0)
        limits = (data_low - 0.13 * span, data_high + 0.14 * span)
        for axis in small_axes[row_slice]:
            axis.set_ylim(*limits)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=3))

    # Group labels establish the intended reading order.
    main_left = main_axes[0].get_position().x0
    main_right = main_axes[1].get_position().x1
    boundary_left = small_axes[0].get_position().x0
    boundary_right = small_axes[1].get_position().x1
    figure.text(
        (main_left + main_right) / 2,
        0.795,
        "history-dominated, weak drive",
        ha="center",
        va="bottom",
        fontsize=FONT_GROUP + 0.4,
        fontweight="bold",
    )
    figure.text(
        (boundary_left + boundary_right) / 2,
        0.795,
        "identified boundaries",
        ha="center",
        va="bottom",
        fontsize=FONT_GROUP,
    )
    figure.text(
        0.018,
        0.44,
        r"$\Delta_{\rm Full}$ relative $L_2$ (%)",
        rotation=90,
        ha="center",
        va="center",
        fontsize=FONT_AXIS_LABEL,
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=COLOR_EXACT,
            linewidth=LINE_REFERENCE,
            linestyle=(0, (4, 3)),
            label="Full BPTT",
        )
    ] + [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=COLORS[method],
            edgecolor="none",
            linewidth=0.0,
            alpha=0.90,
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
        bbox_to_anchor=(0.53, 0.94),
        columnspacing=0.76,
        handlelength=0.8,
        handletextpad=0.34,
    )

    output = ROOT / "figs" / "forecasting_controls_hierarchical_preview"
    figure.savefig(
        output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02
    )
    figure.savefig(
        output.with_suffix(".png"),
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
