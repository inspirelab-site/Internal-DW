#!/usr/bin/env python3
"""Plot the four-panel known-SNR mechanism-to-forecasting closure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

from paper_figure_style import (
    BAND_ALPHA,
    COLOR_CLIP,
    COLOR_DW,
    COLOR_EXACT,
    COLOR_GRID,
    COLOR_JREG,
    COLOR_STATIC,
    COLOR_TBPTT,
    FONT_DENSE_LABEL,
    FONT_DENSE_LEGEND,
    FONT_DENSE_TICK,
    FONT_DENSE_TITLE,
    LINE_AXIS,
    LINE_GRID,
    LINE_MAIN,
    LINE_REFERENCE,
    LINE_SECONDARY,
    MARKER_EDGE,
    MARKER_SIZE,
    TEXT_WIDTH_IN,
    apply_paper_style,
)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _median_range(axis, x, summary, *, color, label=None, logy=False):
    center = np.asarray(summary["median"], dtype=float)
    low = np.asarray(summary["min"], dtype=float)
    high = np.asarray(summary["max"], dtype=float)
    axis.plot(x, center, color=color, lw=LINE_MAIN, label=label)
    axis.fill_between(x, low, high, color=color, alpha=BAND_ALPHA, linewidth=0)
    if logy:
        axis.set_yscale("log")


def _log_tick_with_plain_one(value, _position):
    if value <= 0:
        return ""
    exponent = np.log10(value)
    rounded = int(np.rint(exponent))
    if not np.isclose(exponent, rounded):
        return ""
    return "1" if rounded == 0 else rf"$10^{{{rounded}}}$"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--closure-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    closure_root = Path(args.closure_root)
    panels_1_2 = _load(closure_root / "profiles/panels_1_2.json")
    route_risk = _load(closure_root / "risk/summary.json")
    forecasting = _load(closure_root / "forecast_summary.json")

    horizons = np.asarray(panels_1_2["horizon"], dtype=float)
    colors = {
        "dark": "#30343B",
        "blue": COLOR_DW,
        "orange": "#d95f02",
        "purple": "#6a3d9a",
        "green": "#1b9e77",
        "teal": "#2a9d8f",
        "gray": COLOR_EXACT,
        "light_gray": "#bdbdbd",
    }

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

    # The figure is read as a left-to-right mechanism chain, so keep the four
    # panels wide and deliberately shallow at the paper's final text width.
    fig = plt.figure(figsize=(TEXT_WIDTH_IN, 1.68))
    # Explicit spacer columns allow the SNR axis label to breathe between
    # panels (a) and (b), while reclaiming space on the right of (b) for the
    # wider categorical x axis in panel (d).
    grid = GridSpec(
        1,
        7,
        figure=fig,
        width_ratios=[1.15, 1.00, 1.12, 0.72, 0.86, 0.70, 1.30],
        wspace=0.0,
    )
    ax_mag = fig.add_subplot(grid[0, 0])
    ax_prefix = fig.add_subplot(grid[0, 2])
    route_grid = grid[0, 4].subgridspec(
        2, 1, height_ratios=[0.18, 0.82], hspace=0.05
    )
    ax_route_top = fig.add_subplot(route_grid[0, 0])
    ax_route = fig.add_subplot(route_grid[1, 0], sharex=ax_route_top)
    ax_forecast = fig.add_subplot(grid[0, 6])

    # Panel 1: retain the established failure-profile protocol and statistics.
    _median_range(
        ax_mag,
        horizons,
        panels_1_2["panel_1"]["total_gradient_rms_relative_to_h1"],
        color=colors["dark"],
        logy=True,
    )
    ax_mag.axhline(1.0, color="0.65", ls="--", lw=LINE_REFERENCE)
    ax_mag.set_title("(a) Large gradients need not\nbe reliable", pad=2.0)
    ax_mag.set_ylabel(
        "gradient magnitude / step 1",
        color=colors["dark"],
        fontweight="bold",
    )
    ax_mag.tick_params(axis="y", colors=colors["dark"])
    ax_mag.set_xlabel("step $k$", fontweight="bold")
    ax_mag.set_xlim(0.5, horizons[-1] + 0.5)
    ax_mag.set_xticks([1, 16, int(horizons[-1])])

    gradient_snr_summary = panels_1_2["panel_1"]["gradient_snr"]
    gradient_snr = np.asarray(gradient_snr_summary["median"], dtype=float)
    ax_snr = ax_mag.twinx()
    _median_range(
        ax_snr,
        horizons,
        gradient_snr_summary,
        color=colors["purple"],
        logy=True,
    )
    snr_floor = max(
        float(np.nanmin(gradient_snr_summary["min"])) * 0.65, 1e-4
    )
    snr_ceiling = float(np.nanmax(gradient_snr_summary["max"])) * 1.35
    ax_snr.set_ylim(snr_floor, snr_ceiling)
    ax_snr.yaxis.set_major_formatter(FuncFormatter(_log_tick_with_plain_one))
    ax_snr.axhspan(snr_floor, 1.0, color=colors["orange"], alpha=0.045)
    ax_snr.axhline(1.0, color=colors["purple"], ls=":", lw=LINE_REFERENCE)
    ax_snr.tick_params(axis="y", colors=colors["purple"], pad=0.8)
    ax_snr.set_ylabel(
        "gradient SNR",
        color=colors["purple"],
        labelpad=0.8,
        fontweight="bold",
    )
    ax_snr.text(
        0.40,
        0.78,
        "SNR",
        transform=ax_snr.transAxes,
        ha="center",
        va="top",
        fontsize=FONT_DENSE_LABEL,
        color=colors["purple"],
    )
    ax_snr.text(
        0.72,
        0.045,
        "SNR < 1",
        transform=ax_snr.transAxes,
        ha="right",
        va="bottom",
        fontsize=FONT_DENSE_TICK,
        color="0.38",
    )
    ax_snr.spines["top"].set_visible(False)

    # Panel 2: retain the established prefix bias--innovation decomposition.
    prefix_panel = panels_1_2["panel_2"]
    _median_range(
        ax_prefix,
        horizons,
        prefix_panel["omitted_signal_bias_over_exact"],
        color=colors["blue"],
        label="missing signal",
    )
    _median_range(
        ax_prefix,
        horizons,
        prefix_panel["innovation_risk_over_exact"],
        color=colors["orange"],
        label="innovation",
    )
    _median_range(
        ax_prefix,
        horizons,
        prefix_panel["total_prefix_risk_over_exact"],
        color=colors["dark"],
        label="total",
    )
    k_grad_star = int(prefix_panel["median_curve_best_prefix_horizon"])
    ax_prefix.axvline(k_grad_star, color=colors["teal"], ls=":", lw=1.05)
    ax_prefix.text(
        k_grad_star - 0.5,
        0.16,
        rf"$K^\star={k_grad_star}$",
        ha="right",
        va="center",
        fontsize=FONT_DENSE_TICK,
        color=colors["teal"],
    )
    ax_prefix.axhline(1.0, color="0.55", ls="--", lw=LINE_REFERENCE)
    ax_prefix.set_ylim(0.0, 1.06)
    ax_prefix.set_xlim(0.5, horizons[-1] + 0.5)
    ax_prefix.set_xticks([1, 16, int(horizons[-1])])
    ax_prefix.set_title("(b) Long horizons can\nincrease gradient risk", pad=2.0)
    ax_prefix.set_ylabel("risk / Full", labelpad=0.5, fontweight="bold")
    ax_prefix.set_xlabel("last step $k$", fontweight="bold")
    ax_prefix.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.94),
        handlelength=0.68,
        handletextpad=0.35,
        labelspacing=0.20,
        borderpad=0.05,
    )

    # Panel 3: frozen training-stream local route risk, three matched seeds.
    route_methods = route_risk["methods"]
    route_reference = route_methods["full_bptt"]["mean"]
    route_keys = ["misplaced", "online_dw", "local_oracle"]
    route_labels = ["misplaced", "DW", "local oracle"]
    route_values = [
        route_methods[key]["mean"] for key in route_keys
    ]
    route_positions = np.arange(len(route_values))
    ax_route.bar(
        route_positions,
        route_values,
        yerr=[route_methods[key]["sample_sd"] for key in route_keys],
        capsize=2,
        color=[
            colors["light_gray"],
            colors["blue"],
            colors["green"],
        ],
        alpha=0.90,
        width=0.72,
    )
    route_label_offsets = [0.010, 0.010, 0.010]
    for position, value, offset in zip(
        route_positions, route_values, route_label_offsets
    ):
        ax_route.text(
            position,
            value + offset,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=FONT_DENSE_TICK,
        )
    ax_route.set_xticks(route_positions)
    ax_route.set_xticklabels(
        route_labels, rotation=27, ha="right", rotation_mode="anchor"
    )
    ax_route.set_xlim(-0.55, 2.55)
    ax_route.set_ylim(0.0, 1.2 * max(route_methods[k]['mean'] + route_methods[k]['sample_sd'] for k in route_keys))
    ax_route.set_ylabel("risk / open", labelpad=0.5, fontweight="bold")

    # Show fully open BPTT as a reference line on a small broken-axis strip.
    # This preserves its exact value of one without compressing the three
    # informative bars into the bottom quarter of the panel.
    ax_route_top.axhline(
        route_reference, color="0.4", ls="--", lw=LINE_REFERENCE
    )
    ax_route_top.text(
        -0.30,
        route_reference - 0.005,
        "Full BPTT",
        ha="left",
        va="top",
        fontsize=FONT_DENSE_TICK,
        color="0.35",
    )
    ax_route_top.set_ylim(0.97, 1.03)
    ax_route_top.set_yticks([1.0])
    ax_route_top.set_yticklabels(["1.00"])
    ax_route_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_route_top.set_title("(c) Internal-DW reduces\nroute risk", pad=2.0)

    break_style = dict(
        marker=[(-1, -0.5), (1, 0.5)],
        markersize=4.2,
        linestyle="none",
        color="0.35",
        mec="0.35",
        mew=LINE_AXIS,
        clip_on=False,
    )
    ax_route_top.plot([0], [0], transform=ax_route_top.transAxes, **break_style)
    ax_route.plot([0], [1], transform=ax_route.transAxes, **break_style)

    # Panel 4: matched three-seed forecasting means and sample deviations.
    # Methods are placed on the
    # x axis so the vertical displacement directly communicates lower error.
    forecast_results = forecasting["methods"]
    forecast_keys = [
        "full_bptt",
        "clip",
        "jreg",
        "tbptt",
        "internal_dw",
    ]
    forecast_labels = ["Full", "Clip", "JReg", "TBPTT", "DW"]
    forecast_values = np.asarray(
        [forecast_results[key]["mean"] for key in forecast_keys]
    )
    x_positions = np.arange(len(forecast_keys))
    markers = ["o", "^", "D", "s", "o"]
    marker_colors = [
        COLOR_EXACT,
        COLOR_CLIP,
        COLOR_JREG,
        COLOR_TBPTT,
        COLOR_DW,
    ]
    for x, value, marker, color, key in zip(
        x_positions, forecast_values, markers, marker_colors, forecast_keys
    ):
        kwargs = {}
        if key == "tbptt":
            kwargs = {
                "facecolors": "white",
                "edgecolors": color,
                "linewidths": MARKER_EDGE,
            }
        else:
            kwargs = {"color": color}
        ax_forecast.scatter(
            x, value, s=MARKER_SIZE**2, marker=marker, zorder=3, **kwargs
        )
    forecast_sd = np.asarray([forecast_results[k]['sample_sd'] for k in forecast_keys])
    ax_forecast.errorbar(x_positions, forecast_values, yerr=forecast_sd,
                        fmt='none', ecolor='0.4', capsize=2, lw=0.8)
    exact_value = float(forecast_results["full_bptt"]["mean"])
    ax_forecast.axhline(
        exact_value, color="0.55", ls="--", lw=LINE_REFERENCE
    )
    low = float(np.min(forecast_values - forecast_sd)) - 0.012
    high = float(np.max(forecast_values + forecast_sd)) + 0.012
    ax_forecast.set_xlim(-0.55, len(forecast_keys) - 0.45)
    ax_forecast.set_ylim(low, high)
    ax_forecast.set_xticks(x_positions)
    ax_forecast.set_xticklabels(
        forecast_labels, rotation=28, ha="right", rotation_mode="anchor"
    )
    ax_forecast.set_title("(d) Internal-DW lowers\nforecast error", pad=2.0)
    ax_forecast.set_ylabel("relative $L_2$", labelpad=0.5, fontweight="bold")
    ax_forecast.yaxis.set_major_locator(plt.MaxNLocator(4))

    for axis in (ax_mag, ax_prefix, ax_route, ax_forecast):
        axis.grid(True, which="major", color=COLOR_GRID, lw=LINE_GRID)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(length=2.4, width=LINE_AXIS, pad=1.4)

    ax_route_top.grid(False)
    ax_route_top.spines["top"].set_visible(False)
    ax_route_top.spines["right"].set_visible(False)
    ax_route_top.spines["bottom"].set_visible(False)
    ax_route_top.tick_params(length=2.4, width=LINE_AXIS, pad=1.4)

    fig.subplots_adjust(left=0.060, right=0.995, bottom=0.270, top=0.835)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
    fig.savefig(
        out.with_suffix(".png"), dpi=450, bbox_inches="tight", pad_inches=0.01
    )
    plt.close(fig)

    summary = {
        "panels_1_2_source": str(closure_root / "profiles/panels_1_2.json"),
        "closure_root": str(closure_root),
        "repetitions": int(panels_1_2["protocol"]["repetitions"]),
        "median_curve_k_grad_star": k_grad_star,
        "route_risk_over_open": dict(
            zip(["fully open", *route_labels], [route_reference, *route_values])
        ),
        "forecast_mean_relative_l2": dict(
            zip(forecast_labels, forecast_values.tolist())
        ),
    }
    out.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[out] {out.with_suffix('.pdf')}")
    print(f"[out] {out.with_suffix('.png')}")
    print(f"[out] {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
