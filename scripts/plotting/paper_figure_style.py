"""Shared visual style for figures embedded in the ICLR paper.

Sizes are specified for figures rendered directly at the paper's final
``\textwidth``.  Dense five- or eight-panel layouts may reduce text slightly,
but no essential label should fall below ``FONT_DENSE_TICK``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


TEXT_WIDTH_IN = 5.5

FONT_FAMILY = "Arial"
FONT_BODY = 7.8
FONT_PANEL_TITLE = 8.4
FONT_AXIS_LABEL = 7.8
FONT_TICK = 6.8
FONT_LEGEND = 7.4
FONT_ANNOTATION = 6.7
FONT_GROUP = 8.2
FONT_FIGURE_TITLE = 9.0
FONT_DENSE_TITLE = 7.3
FONT_DENSE_LABEL = 6.9
FONT_DENSE_TICK = 6.2
FONT_DENSE_LEGEND = 6.8

LINE_MAIN = 1.65
LINE_SECONDARY = 1.25
LINE_REFERENCE = 0.85
LINE_AXIS = 0.80
LINE_GRID = 0.52
MARKER_SIZE = 4.6
MARKER_EDGE = 0.85
BAND_ALPHA = 0.16

COLOR_EXACT = "#5B5B5B"
COLOR_DW = "#0072B2"
COLOR_CLIP = "#D55E00"
COLOR_JREG = "#7A5195"
COLOR_TBPTT = "#E69F00"
COLOR_STATIC = "#009E73"
COLOR_MAGNITUDE = "#30343B"
COLOR_UTILITY = "#6D4BB3"
COLOR_IDENTITY = "#2563A6"
COLOR_NONLINEAR = "#D97706"
COLOR_GRID = "#D9DEE4"
COLOR_SPINE = "#626971"


def apply_paper_style() -> None:
    """Apply the common paper-figure typography and vector-font settings."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY, "Liberation Sans", "DejaVu Sans"],
            "font.weight": "normal",
            "font.size": FONT_BODY,
            "mathtext.fontset": "stixsans",
            "axes.titlesize": FONT_PANEL_TITLE,
            "axes.titleweight": "normal",
            "axes.labelsize": FONT_AXIS_LABEL,
            "axes.labelweight": "normal",
            "xtick.labelsize": FONT_TICK,
            "ytick.labelsize": FONT_TICK,
            "legend.fontsize": FONT_LEGEND,
            "axes.linewidth": LINE_AXIS,
            "xtick.major.width": LINE_AXIS,
            "ytick.major.width": LINE_AXIS,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )
