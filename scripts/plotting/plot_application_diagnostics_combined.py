#!/usr/bin/env python3
"""Combine the two application-data diagnostics into one paper figure.

Panel (a) retains the per-horizon gradient magnitude and held-out utility
curves.  Panel (b) summarizes the controlled-noise gain response as an aligned
heatmap, so the two observable checks read as one application-data story
without implying that one causes the other.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from paper_figure_style import (
    COLOR_IDENTITY,
    COLOR_MAGNITUDE,
    COLOR_SPINE,
    COLOR_UTILITY,
    FONT_DENSE_LABEL,
    FONT_DENSE_LEGEND,
    FONT_DENSE_TICK,
    FONT_DENSE_TITLE,
    FONT_GROUP,
    LINE_AXIS,
    LINE_REFERENCE,
    TEXT_WIDTH_IN,
    apply_paper_style,
)
from added_noise_panel_data import load_panels
from application_gradient_plot_ops import DATASETS, draw_panel, load_curves


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the combined application-data diagnostics."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "figs" / "application_data_diagnostics_combined.pdf",
        help="Output PDF path; a PNG preview is written beside it.",
    )
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument(
        "--added-noise-summary",
        type=Path,
        default=ROOT
        / "probe_outputs/application_diagnostics_v1/added_noise_gain_summary.json",
    )
    args = parser.parse_args()

    panels = load_panels(args.added_noise_summary)

    loaded = [(spec, load_curves(spec)) for spec in DATASETS]
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

    figure = plt.figure(figsize=(TEXT_WIDTH_IN, 2.68))
    top_grid = figure.add_gridspec(
        1,
        len(DATASETS),
        left=0.072,
        right=0.925,
        bottom=0.56,
        top=0.79,
        wspace=0.31,
    )

    top_axes = [
        figure.add_subplot(top_grid[0, index]) for index in range(len(DATASETS))
    ]
    magnitude_line = utility_line = None
    for index, (axis, (spec, curves)) in enumerate(zip(top_axes, loaded)):
        magnitude_line, utility_line = draw_panel(
            axis,
            spec,
            curves,
            show_left_ticks=(index == 0),
            show_right_ticks=(index == len(DATASETS) - 1),
            amplitude_ylim=amplitude_ylim,
            show_axis_labels=False,
            compact=True,
        )

    figure.text(
        0.012,
        0.635,
        r"$A(k)$",
        rotation=90,
        va="center",
        ha="left",
        color=COLOR_MAGNITUDE,
        fontsize=FONT_DENSE_LABEL,
    )
    figure.text(
        0.985,
        0.635,
        r"$U(k)$",
        rotation=-90,
        va="center",
        ha="right",
        color=COLOR_UTILITY,
        fontsize=FONT_DENSE_LABEL,
    )
    figure.text(
        0.50,
        0.495,
        "forecast step $k$",
        va="center",
        ha="center",
        fontsize=FONT_DENSE_LABEL,
    )
    figure.text(
        0.015,
        0.955,
        "(a) Gradient size and held-out utility",
        ha="left",
        va="center",
        fontsize=FONT_GROUP,
    )
    figure.legend(
        [magnitude_line, utility_line],
        [r"magnitude $A(k)$", r"held-out utility $U(k)$"],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.995),
        ncol=2,
        frameon=False,
        handlelength=1.9,
        handletextpad=0.45,
        columnspacing=1.25,
    )

    heatmap_axis = figure.add_axes([0.072, 0.105, 0.853, 0.225])
    gain_rows = []
    for ratio_index in range(3):
        gain_rows.append([panel.alpha[ratio_index] for panel in panels])
        gain_rows.append([panel.nonlinear[ratio_index] for panel in panels])
    gain_matrix = np.asarray(gain_rows, dtype=float)

    gain_cmap = LinearSegmentedColormap.from_list(
        "gain_blue",
        ("#F7FAFC", "#C7DDF0", COLOR_IDENTITY, "#123A63"),
    )
    image = heatmap_axis.imshow(
        gain_matrix,
        vmin=0.0,
        vmax=1.0,
        cmap=gain_cmap,
        aspect="auto",
        interpolation="nearest",
    )
    heatmap_axis.set_xticks(range(len(DATASETS)))
    heatmap_axis.set_xticklabels([spec.title for spec in DATASETS])
    heatmap_axis.set_yticks(range(6))
    heatmap_axis.set_yticklabels(
        (
            r"4  $\alpha$",
            r"4  $m$",
            r"1  $\alpha$",
            r"1  $m$",
            r".25  $\alpha$",
            r".25  $m$",
        )
    )
    heatmap_axis.tick_params(axis="x", length=0, pad=2.0)
    heatmap_axis.tick_params(axis="y", width=LINE_AXIS, length=2.2, pad=2.0)
    heatmap_axis.spines["top"].set_visible(False)
    heatmap_axis.spines["right"].set_visible(False)
    heatmap_axis.spines["left"].set_color(COLOR_SPINE)
    heatmap_axis.spines["bottom"].set_color(COLOR_SPINE)
    heatmap_axis.spines["left"].set_linewidth(LINE_AXIS)
    heatmap_axis.spines["bottom"].set_linewidth(LINE_AXIS)
    heatmap_axis.axvline(3.5, color="white", linewidth=2.0)
    heatmap_axis.axvline(3.5, color="#707780", linewidth=LINE_AXIS)
    for boundary in (1.5, 3.5):
        heatmap_axis.axhline(
            boundary,
            color="white",
            linewidth=1.5,
        )
    heatmap_axis.text(
        1.5,
        -0.78,
        "DW-Generic",
        ha="center",
        va="bottom",
        fontsize=FONT_DENSE_LABEL,
        clip_on=False,
    )
    heatmap_axis.text(
        5.5,
        -0.78,
        "DW-Prior",
        ha="center",
        va="bottom",
        fontsize=FONT_DENSE_LABEL,
        clip_on=False,
    )
    heatmap_axis.set_ylabel(
        "residual / added noise",
        labelpad=4.0,
        fontsize=FONT_DENSE_TICK,
    )
    figure.text(
        0.015,
        0.415,
        "(b) Estimated gains as added noise increases",
        ha="left",
        va="center",
        fontsize=FONT_GROUP,
    )

    heatmap_box = heatmap_axis.get_position()
    colorbar_axis = figure.add_axes(
        [heatmap_box.x1 + 0.011, heatmap_box.y0, 0.012, heatmap_box.height]
    )
    colorbar = figure.colorbar(image, cax=colorbar_axis, ticks=(0.0, 0.5, 1.0))
    colorbar.ax.tick_params(width=LINE_AXIS, length=2.0, pad=1.5)
    colorbar.outline.set_linewidth(LINE_REFERENCE)
    colorbar.set_label("gain", rotation=90, labelpad=2.5)

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
    print(f"[out] {output_pdf}")
    print(f"[out] {output_png}")


if __name__ == "__main__":
    main()
