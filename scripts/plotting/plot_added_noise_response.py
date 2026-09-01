#!/usr/bin/env python3
"""Plot the added-noise gain response as two shared-scale panels.

The left panel shows the identity-route gain alpha and the right panel shows
the nonlinear-route gain m.  Each dataset keeps the same color in both panels.
The horizontal order is chosen so that added noise increases from left to
right; the exact residual-to-added-noise ratios remain 4, 1, and 0.25.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from paper_figure_style import (
    COLOR_GRID,
    COLOR_SPINE,
    FONT_AXIS_LABEL,
    FONT_DENSE_TICK,
    FONT_LEGEND,
    FONT_PANEL_TITLE,
    LINE_AXIS,
    LINE_GRID,
    LINE_MAIN,
    MARKER_EDGE,
    MARKER_SIZE,
    TEXT_WIDTH_IN,
    apply_paper_style,
)


ROOT = Path(__file__).resolve().parents[2]
RATIO_LABELS = ("4", "1", ".25")
NOISE_LABELS = ("low", "medium", "high")

# A colorblind-friendly categorical palette.  Colors identify datasets and
# remain fixed between the alpha and m panels.
DATASET_COLORS = (
    "#0072B2",  # MG
    "#D55E00",  # NARMA-5
    "#009E73",  # ETTm1
    "#CC79A7",  # ETTm2
    "#56B4E9",  # iEEG
    "#E69F00",  # fMRI
    "#7A5195",  # Shear
    "#6B7280",  # WB2
)
DATASET_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h")


@dataclass(frozen=True)
class Panel:
    title: str
    estimator: str
    alpha: tuple[float, float, float]
    nonlinear: tuple[float, float, float]


PANELS = (
    Panel("MG", "DW-Generic", (.662, .318, .047), (.656, .311, .047)),
    Panel("NARMA", "DW-Generic", (.096, .061, .013), (.090, .058, .019)),
    Panel("ETTm1", "DW-Generic", (.638, .617, .430), (.494, .484, .336)),
    Panel("ETTm2", "DW-Generic", (.791, .749, .502), (.716, .663, .406)),
    Panel("iEEG", "DW-Prior", (.748, .463, .258), (.746, .457, .258)),
    Panel("fMRI", "DW-Prior", (.884, .687, .411), (.848, .619, .366)),
    Panel("Shear", "DW-Prior", (.934, .826, .579), (.871, .769, .541)),
    Panel("WB2", "DW-Prior", (.993, .976, .924), (.994, .976, .924)),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Internal-DW gains as added noise increases."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "figs" / "added_noise_response.pdf",
        help="Output PDF path; a PNG preview is written beside it.",
    )
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()

    apply_paper_style()

    figure, axes = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH_IN, 2.02), sharex=True, sharey=True
    )
    figure.subplots_adjust(
        left=0.082,
        right=0.995,
        bottom=0.225,
        top=0.70,
        wspace=0.17,
    )

    x = (0, 1, 2)
    for axis_index, (axis, field, panel_title) in enumerate(
        zip(axes, ("alpha", "nonlinear"), (r"Identity gain $\alpha$", r"Nonlinear gain $m$"))
    ):
        for dataset_index, panel in enumerate(PANELS):
            axis.plot(
                x,
                getattr(panel, field),
                color=DATASET_COLORS[dataset_index],
                linewidth=LINE_MAIN,
                marker=DATASET_MARKERS[dataset_index],
                markersize=MARKER_SIZE - 0.2,
                markerfacecolor="white",
                markeredgecolor=DATASET_COLORS[dataset_index],
                markeredgewidth=MARKER_EDGE,
                zorder=3,
            )
        axis.set_title(panel_title, pad=4.0, fontsize=FONT_PANEL_TITLE)
        axis.set_xlim(-0.18, 2.18)
        axis.set_ylim(0.0, 1.04)
        axis.set_xticks(x, NOISE_LABELS)
        axis.set_yticks((0.0, 0.5, 1.0))
        axis.grid(axis="y", color=COLOR_GRID, linewidth=LINE_GRID, alpha=0.8)
        axis.set_axisbelow(True)
        axis.tick_params(
            axis="both", width=LINE_AXIS, length=2.5, pad=1.5,
            labelsize=FONT_DENSE_TICK,
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(COLOR_SPINE)
        axis.spines["bottom"].set_color(COLOR_SPINE)
        axis.spines["left"].set_linewidth(LINE_AXIS)
        axis.spines["bottom"].set_linewidth(LINE_AXIS)
        if axis_index > 0:
            axis.spines["left"].set_visible(False)
            axis.tick_params(axis="y", length=0)

    axes[0].set_ylabel("gain", labelpad=3.5)
    figure.supxlabel(
        r"added noise  $\longrightarrow$",
        x=0.54, y=0.035, fontsize=FONT_AXIS_LABEL,
    )
    legend_handles = [
        Line2D(
            [], [], color=color, linewidth=LINE_MAIN, marker=marker,
            markersize=MARKER_SIZE - 0.2, markerfacecolor="white",
            markeredgecolor=color, markeredgewidth=MARKER_EDGE,
            label=panel.title,
        )
        for panel, color, marker in zip(PANELS, DATASET_COLORS, DATASET_MARKERS)
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.535, 0.995),
        ncol=4,
        frameon=False,
        handlelength=1.45,
        columnspacing=1.05,
        handletextpad=0.35,
        fontsize=FONT_LEGEND,
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
    print(f"[out] {output_pdf}")
    print(f"[out] {output_png}")


if __name__ == "__main__":
    main()
