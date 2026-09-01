import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

from paper_figure_style import (
    FONT_ANNOTATION,
    FONT_FIGURE_TITLE,
    FONT_GROUP,
    FONT_LEGEND,
    TEXT_WIDTH_IN,
    apply_paper_style,
)


OUT = Path(os.environ.get(
    "INTRO_GRADIENT_FIG_DIR",
    Path(__file__).resolve().parents[2] / "figs",
))
BLUE = "#1f77b4"
ORANGE = "#d97706"
INK = "#202124"
MUTED = "#686c72"
RULE = "#d7d9dc"

FULL = [(23, 7), (28, 14), (33, 23), (28, 55), (23, 92)]
XOFF = [9, 31, 56, 84, 116]
WIDTHS = [15, 17, 18, 20, 23]


def text(ax, x, y, value, *, size=FONT_ANNOTATION, weight="normal", color=INK,
         ha="center", va="center"):
    ax.text(x, y, value, fontsize=size, fontweight=weight, color=color,
            ha=ha, va=va, family="DejaVu Sans")


def mixed_bar(ax, x, base, width, signal_h, noise_h, *, outline=None,
              crossed=False):
    if outline is not None:
        ax.add_patch(Rectangle(
            (x, base), width, outline, facecolor="none", edgecolor=MUTED,
            linewidth=0.65, linestyle=(0, (3, 2)), alpha=0.85
        ))
    if crossed:
        ax.plot([x, x + width], [base, base + outline], color=MUTED, lw=0.8)
        ax.plot([x, x + width], [base + outline, base], color=MUTED, lw=0.8)
        return
    ax.add_patch(Rectangle(
        (x, base), width, signal_h, facecolor=BLUE, edgecolor="none"
    ))
    ax.add_patch(Rectangle(
        (x, base + signal_h), width, noise_h, facecolor=ORANGE,
        edgecolor=ORANGE, linewidth=0.35, hatch="////", alpha=0.33
    ))


def update_range(ax, x0, center, arrow_len, spread, *, full_outline=False):
    sx = x0 + 177
    ex = x0 + 232
    if full_outline:
        ax.add_patch(Polygon(
            [(sx, center), (ex, center + 37), (ex, center - 37)], closed=True,
            facecolor="none", edgecolor=MUTED, linewidth=0.65,
            linestyle=(0, (3, 2)), alpha=0.75
        ))
    ax.add_patch(Polygon(
        [(sx, center), (ex, center + spread), (ex, center - spread)], closed=True,
        facecolor=ORANGE, edgecolor=ORANGE, linewidth=0.45, alpha=0.16
    ))
    ax.add_patch(FancyArrowPatch(
        (sx + 1, center), (sx + arrow_len, center), arrowstyle="-|>",
        mutation_scale=7, linewidth=1.5, color=BLUE, shrinkA=0, shrinkB=0
    ))


def draw_panel(ax, idx, title, subtitle, scales, *, cutoff_from=None,
               arrow_len=40, spread=20, bottom_lines=(), forward_note=None):
    x0 = idx * 240
    xc = x0 + 120
    if idx:
        ax.plot([x0, x0], [14, 220], color=RULE, lw=0.8)
    text(ax, xc, 214, title, size=6.4)
    text(ax, xc, 189, subtitle, size=4.8, color=MUTED)

    base = 57
    ax.plot([x0 + 7, x0 + 172], [base, base], color=RULE, lw=0.9)
    for j, ((sig, noi), dx, width) in enumerate(zip(FULL, XOFF, WIDTHS)):
        full_h = sig + noi
        if cutoff_from is not None and j >= cutoff_from:
            mixed_bar(ax, x0 + dx, base, width, 0, 0,
                      outline=full_h, crossed=True)
            continue
        scale = scales[j]
        if isinstance(scale, (tuple, list)):
            signal_scale, noise_scale = scale
        else:
            signal_scale = noise_scale = scale
        outline = full_h if min(signal_scale, noise_scale) < 0.999 else None
        mixed_bar(ax, x0 + dx, base, width,
                  sig * signal_scale, noi * noise_scale,
                  outline=outline)

    text(ax, x0 + 154, 126, r"$\Sigma$", size=15)
    update_range(ax, x0, 126, arrow_len, spread,
                 full_outline=idx in (1, 2, 3, 4))
    text(ax, x0 + 76, 40, r"step  $k$",
         size=4.8, color=MUTED)
    if forward_note:
        text(ax, xc, 25, forward_note, size=5.2, color=MUTED)
    for row, line in enumerate(bottom_lines):
        text(ax, xc, 25 - row * 13, line, size=4.8, color=MUTED)


def main():
    apply_paper_style()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 1.22), dpi=220)
    fig.patch.set_alpha(0)
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, 272)
    ax.axis("off")

    ax.add_patch(Rectangle((245, 241), 17, 8, color=BLUE))
    text(ax, 268, 245, "signal", size=4.7, color=MUTED, ha="left")
    ax.add_patch(Rectangle(
        (390, 241), 17, 8, facecolor=ORANGE, edgecolor=ORANGE,
        linewidth=0.35, hatch="////", alpha=0.33
    ))
    text(ax, 413, 245, "noise", size=4.7, color=MUTED, ha="left")
    ax.add_patch(Rectangle(
        (545, 241), 17, 8, facecolor="none", edgecolor=MUTED,
        linewidth=0.65, linestyle=(0, (3, 2))
    ))
    text(ax, 568, 245, "full-BPTT size", size=4.7,
         color=MUTED, ha="left")
    ax.add_patch(Polygon(
        [(750, 245), (785, 252), (785, 238)], closed=True,
        facecolor=ORANGE, edgecolor=ORANGE, linewidth=0.4, alpha=0.16
    ))
    ax.add_patch(FancyArrowPatch(
        (751, 245), (780, 245), arrowstyle="-|>", mutation_scale=7,
        linewidth=1.5, color=BLUE, shrinkA=0, shrinkB=0
    ))
    text(ax, 794, 245, "update and noise range", size=4.7,
         color=MUTED, ha="left")

    draw_panel(
        ax, 0, "Exact BPTT", "all routes",
        [1, 1, 1, 1, 1], arrow_len=42, spread=37,
        bottom_lines=("late noise", "can dominate")
    )
    draw_panel(
        ax, 1, "LR / clipping", "global scaling",
        [0.45] * 5, arrow_len=20, spread=17,
        bottom_lines=("signal and noise", "shrink together")
    )
    draw_panel(
        ax, 2, "Jacobian regularization", "expansion > 1 penalized",
        [0.72, 0.60, 0.50, 0.38, 0.28], arrow_len=24, spread=14,
        bottom_lines=("growth limited;", "forward constrained")
    )
    draw_panel(
        ax, 3, "Step cutoff", "late steps removed",
        [1, 1, 1, 1, 1], cutoff_from=2, arrow_len=22, spread=14,
        bottom_lines=("late signal", "is also lost")
    )
    draw_panel(
        ax, 4, "Internal-DW", "reliability weights",
        [
            (0.90, 0.75),
            (0.65, 0.50),
            (0.88, 0.25),
            (0.72, 0.12),
            (0.85, 0.06),
        ],
        arrow_len=40, spread=12,
        bottom_lines=("signal retained;", "noise reduced")
    )

    fig.subplots_adjust(left=0.004, right=0.996, top=0.995, bottom=0.01)
    fig.savefig(OUT / "intro_gradient_controls.png", dpi=220,
                transparent=True, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(OUT / "intro_gradient_controls.pdf", transparent=True,
                bbox_inches="tight", pad_inches=0.025)


if __name__ == "__main__":
    main()
