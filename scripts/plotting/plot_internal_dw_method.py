import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from paper_figure_style import (
    FONT_ANNOTATION,
    FONT_FIGURE_TITLE,
    FONT_GROUP,
    TEXT_WIDTH_IN,
    apply_paper_style,
)


OUT = Path(os.environ.get(
    "INTERNAL_DW_METHOD_FIG_DIR",
    Path(__file__).resolve().parents[2] / "figs",
))

INK = "#202124"
MUTED = "#686c72"
RULE = "#d7d9dc"
BLUE = "#1f77b4"
ORANGE = "#d97706"
PURPLE = "#6f5cc2"


def text(ax, x, y, value, *, size=FONT_ANNOTATION, weight="normal", color=INK,
         ha="center", va="center"):
    ax.text(x, y, value, fontsize=size, fontweight=weight, color=color,
            ha=ha, va=va, family="DejaVu Sans")


def arrow(ax, start, end, *, color=INK, lw=1.25, scale=8,
          connectionstyle=None):
    kwargs = {}
    if connectionstyle is not None:
        kwargs["connectionstyle"] = connectionstyle
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=scale,
        linewidth=lw, color=color, shrinkA=0, shrinkB=0, **kwargs
    ))


def line(ax, points, *, color=INK, lw=1.25, style="-"):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, lw=lw, linestyle=style,
            solid_capstyle="round", solid_joinstyle="round")


def box(ax, x, y, w, h, label, *, edge=RULE, face="none", size=7.5,
        weight="normal", color=INK, radius=5, lw=0.9):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw
    ))
    text(ax, x + w / 2, y + h / 2, label, size=size, weight=weight,
         color=color)


def matrix_icon(ax, x, y, kind, label):
    cell = 10
    gap = 2.5
    colors = {
        "total": [[BLUE, ORANGE], [ORANGE, BLUE]],
        "noise": [[ORANGE, ORANGE], [ORANGE, ORANGE]],
        "signal": [[BLUE, BLUE], [BLUE, BLUE]],
    }[kind]
    for row in range(2):
        for col in range(2):
            ax.add_patch(Rectangle(
                (x + col * (cell + gap), y + (1 - row) * (cell + gap)),
                cell, cell, facecolor=colors[row][col],
                edgecolor=colors[row][col], linewidth=0.5, alpha=0.35
            ))
    text(ax, x + 11, y - 8, label, size=7.1, color=MUTED)


def main():
    apply_paper_style()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 1.48), dpi=220)
    fig.patch.set_alpha(0)
    ax.set_xlim(0, 1500)
    ax.set_ylim(0, 300)
    ax.axis("off")

    ax.plot([600, 600], [20, 290], color=RULE, lw=0.9)

    # ------------------------------------------------------------------
    # (a) Backward-only residual routing.
    # ------------------------------------------------------------------
    text(ax, 300, 286, "(a) Backward-only route weights", size=7.4)

    text(ax, 28, 225, "forward", size=7.2, color=MUTED, ha="left")
    text(ax, 78, 213, r"$x$", size=10.0)
    arrow(ax, (92, 213), (136, 213), lw=1.2)
    # Split into the identity and nonlinear branches.
    line(ax, [(136, 213), (136, 244), (404, 244), (436, 213)], lw=1.2)
    arrow(ax, (404, 244), (436, 213), lw=1.2)
    text(ax, 270, 255, "identity", size=7.2, color=MUTED)
    line(ax, [(136, 213), (136, 178)], lw=1.2)
    arrow(ax, (136, 178), (215, 178), lw=1.2)
    box(ax, 215, 159, 140, 38, r"$F_\theta(x)$", edge=MUTED,
        face="none", size=9.0, radius=4)
    arrow(ax, (355, 178), (436, 208), lw=1.2)
    ax.add_patch(Circle((448, 213), 12, facecolor="none",
                        edgecolor=INK, linewidth=1.0))
    text(ax, 448, 213, "+", size=10.0)
    arrow(ax, (460, 213), (520, 213), lw=1.2)
    text(ax, 541, 213, r"$y$", size=10.0)
    text(ax, 315, 139, r"forward value unchanged:  $y=x+F_\theta(x)$",
         size=6.2, color=MUTED)

    ax.plot([25, 575], [125, 125], color=RULE, lw=0.7)

    text(ax, 535, 105, r"incoming $v$", size=8.0)
    arrow(ax, (520, 82), (463, 82), color=PURPLE, lw=1.5)
    ax.add_patch(Circle((449, 82), 11, facecolor="none",
                        edgecolor=PURPLE, linewidth=1.0))
    # Upper identity VJP route, moving right to left.
    line(ax, [(438, 82), (410, 105), (182, 105), (149, 82)],
         color=PURPLE, lw=1.45)
    arrow(ax, (183, 105), (149, 82), color=PURPLE, lw=1.45)
    box(ax, 350, 94, 42, 23, r"$\alpha$", edge=BLUE,
        face="#e8f1f7", size=8.4, color=BLUE, radius=5, lw=1.0)
    text(ax, 285, 115, r"identity VJP:  $v$", size=5.9, color=MUTED)
    # Lower nonlinear VJP route.
    line(ax, [(438, 82), (410, 50), (182, 50), (149, 82)],
         color=PURPLE, lw=1.45)
    arrow(ax, (183, 50), (149, 82), color=PURPLE, lw=1.45)
    box(ax, 350, 39, 42, 23, r"$m$", edge=BLUE,
        face="#e8f1f7", size=8.4, color=BLUE, radius=5, lw=1.0)
    text(ax, 250, 38, r"nonlinear VJP:  $J_F^\top v$", size=5.9,
         color=MUTED)
    ax.add_patch(Circle((136, 82), 12, facecolor="none",
                        edgecolor=PURPLE, linewidth=1.0))
    text(ax, 136, 82, "+", size=10.0, color=PURPLE)
    box(ax, 20, 61, 100, 42, r"$\alpha v+mJ_F^\top v$", edge=PURPLE,
        face="none", size=5.9, radius=5, lw=1.0)
    arrow(ax, (124, 82), (120, 82), color=PURPLE, lw=1.5)
    text(ax, 315, 16, r"Exact BPTT is the special case  $\alpha=m=1$",
         size=5.9, color=MUTED)

    # ------------------------------------------------------------------
    # (b) Automatic gain calibration.
    # ------------------------------------------------------------------
    text(ax, 1050, 286, "(b) Automatic gain calibration", size=7.4)

    # Task-probe lane.
    text(ax, 632, 232, "task probe", size=5.9, color=MUTED, ha="left")
    ax.add_patch(Rectangle((635, 188), 24, 18, facecolor=BLUE,
                           edgecolor="none"))
    ax.add_patch(Rectangle((635, 206), 24, 13, facecolor=ORANGE,
                           edgecolor=ORANGE, linewidth=0.4, hatch="////",
                           alpha=0.36))
    arrow(ax, (730, 204), (770, 204), lw=1.0)
    box(ax, 770, 179, 125, 50, "fully open\nroute VJP", edge=MUTED,
        face="none", size=6.2, radius=5)
    arrow(ax, (895, 204), (930, 204), lw=1.0)
    matrix_icon(ax, 935, 193, "total", r"$T$")

    # Noise-probe lane.
    text(ax, 632, 130, "noise probe", size=5.9, color=MUTED, ha="left")
    box(ax, 630, 75, 104, 46, "$\\mathcal{Q}_d$",
        edge=ORANGE, face="#fbf1e6", size=7.4, radius=5, lw=1.0)
    arrow(ax, (734, 98), (770, 98), lw=1.0)
    box(ax, 770, 73, 125, 50, "fully open\nroute VJP", edge=MUTED,
        face="none", size=6.2, radius=5)
    text(ax, 832, 63, r"$\widetilde\epsilon_{1:K}\sim\mathcal{Q}_d$",
         size=6.8, color=MUTED)
    arrow(ax, (895, 98), (930, 98), lw=1.0)
    matrix_icon(ax, 935, 87, "noise", r"$R$")

    # Merge the two estimated moments.
    arrow(ax, (985, 204), (1030, 166), lw=1.0)
    arrow(ax, (985, 98), (1030, 143), lw=1.0)
    box(ax, 1030, 128, 125, 50, r"$P=(T-R)_{+}$",
        edge=BLUE, face="#e8f1f7", size=5.9, color=INK, radius=5, lw=1.0)
    arrow(ax, (1155, 153), (1190, 153), lw=1.0)
    box(ax, 1190, 123, 145, 60, "joint 2 x 2\nWiener solve", edge=INK,
        face="none", size=6.4, radius=5, lw=1.0)
    arrow(ax, (1335, 153), (1370, 153), lw=1.0)
    box(ax, 1370, 123, 110, 60, "$w_{k,\\ell}$\n$=(\\alpha,m)$",
        edge=BLUE, face="#e8f1f7", size=6.8,
        color=BLUE, radius=5, lw=1.1)
    text(ax, 1425, 108, "one pair per horizon and layer",
         size=5.8, color=MUTED)

    text(ax, 1050, 28,
         r"Only $\mathcal{Q}_d$ changes; the VJP probes and Wiener solve are shared.",
         size=5.9, color=MUTED)

    fig.subplots_adjust(left=0.004, right=0.996, top=0.995, bottom=0.01)
    fig.savefig(OUT / "internal_dw_method.png", dpi=220,
                transparent=True, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(OUT / "internal_dw_method.pdf", transparent=True,
                bbox_inches="tight", pad_inches=0.025)


if __name__ == "__main__":
    main()
