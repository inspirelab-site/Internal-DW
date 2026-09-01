#!/usr/bin/env python3
"""Render seed-wise quadratic K sweeps for the main text and appendix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT / "probe_outputs" / "internal_dw_fixed_horizon_positive_v1" / "test"
)
DEFAULT_OUTPUT = (
    ROOT / "probe_outputs" / "internal_dw_fixed_horizon_positive_v1"
    / "summary_complete4" / "fixed_horizon_test_upward_quadratic_compact_large"
)
DEFAULT_APPENDIX_OUTPUT = (
    ROOT / "probe_outputs" / "internal_dw_fixed_horizon_positive_v1"
    / "summary_complete4" / "fixed_horizon_seedwise_fits_appendix"
)
SPECS = {
    "mg": ("MG", 64, [8, 12, 16, 24, 32, 48, 64]),
    "ettm1": ("ETTm1", 128, [16, 24, 32, 48, 64, 96, 128]),
    "ettm2": ("ETTm2", 128, [16, 24, 32, 48, 64, 96, 128]),
    "shear": ("Shear flow", 48, [8, 12, 16, 24, 32, 40, 48]),
}
ARMS = (
    ("exact", "Full BPTT", "#5B5B5B", -0.060),
    ("dw", "Internal-DW", "#0072B2", 0.060),
)


def metric(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    if "all_horizons" in summary:
        return float(summary["all_horizons"]["mean"])
    return float(summary["relative_l2_all_horizons_mean"])


def load(input_root: Path, data: str, arm: str, H: int, ks: list[int]) -> np.ndarray:
    values = np.empty((3, len(ks)), dtype=np.float64)
    for seed in range(3):
        for index, K in enumerate(ks):
            path = input_root / data / f"{arm}_K{K}_seed{seed}_H{H}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            values[seed, index] = metric(path)
    return values


def upward_quadratic(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    """Least-squares quadratic with non-negative curvature on the K grid."""
    center = float(np.mean(x))
    scale = float(np.ptp(x))
    z = (x - center) / scale
    coefficients = np.polyfit(z, y, 2)
    at_curvature_boundary = bool(coefficients[0] < 0.0)
    if at_curvature_boundary:
        slope, intercept = np.polyfit(z, y, 1)
        coefficients = np.asarray([0.0, slope, intercept], dtype=np.float64)

    x_fit = np.linspace(float(x.min()), float(x.max()), 400)
    z_fit = (x_fit - center) / scale
    y_fit = np.polyval(coefficients, z_fit)

    a, b, _ = coefficients
    if a > 1e-12:
        z_opt = float(np.clip(-b / (2.0 * a), z.min(), z.max()))
    else:
        endpoint_z = np.asarray([z.min(), z.max()])
        z_opt = float(endpoint_z[np.argmin(np.polyval(coefficients, endpoint_z))])
    k_opt = float(center + scale * z_opt)
    y_opt = float(np.polyval(coefficients, z_opt))
    return {
        "x_fit": x_fit,
        "y_fit": y_fit,
        "coefficients": coefficients,
        "K_opt": k_opt,
        "y_opt": y_opt,
        "curvature_at_constraint_boundary": at_curvature_boundary,
    }


def collect(input_root: Path) -> tuple[dict[str, object], dict[str, object]]:
    fitted: dict[str, object] = {}
    ledger: dict[str, object] = {}
    for data, (label, H, ks) in SPECS.items():
        x = np.asarray(ks, dtype=np.float64)
        fitted[data] = {"label": label, "H": H, "K": x}
        ledger[data] = {"label": label, "H": H, "K": ks}
        for arm, _, _, _ in ARMS:
            values = load(input_root, data, arm, H, ks)
            seed_fits = [upward_quadratic(x, values[seed]) for seed in range(3)]
            fitted[data][arm] = {"values": values, "fits": seed_fits}
            ledger[data][arm] = {
                "seed_values": [values[seed].tolist() for seed in range(3)],
                "seed_fits": [
                    {
                        "K_opt": fit["K_opt"],
                        "y_opt": fit["y_opt"],
                        "normalized_coefficients_a_b_c": fit["coefficients"].tolist(),
                        "curvature_at_constraint_boundary": fit[
                            "curvature_at_constraint_boundary"
                        ],
                    }
                    for fit in seed_fits
                ],
            }
    return fitted, ledger


def save_figure(figure: plt.Figure, output: Path, ledger: dict[str, object]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output.with_suffix(".png"), dpi=400, bbox_inches="tight",
        pad_inches=0.0, facecolor="white",
    )
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.0)
    output.with_suffix(".json").write_text(
        json.dumps(ledger, indent=2), encoding="utf-8"
    )
    plt.close(figure)
    print(f"[out] {output.with_suffix('.png')}")
    print(f"[out] {output.with_suffix('.pdf')}")


def legend_handles(include_optima: bool = False) -> list[Line2D]:
    handles = [
        Line2D(
            [0], [0], color=color, marker="o", markerfacecolor=color,
            markeredgecolor="white", linewidth=1.6, markersize=4.2, label=label,
        )
        for _, label, color, _ in ARMS
    ]
    if include_optima:
        handles.extend(
            [
                Line2D(
                    [0], [0], color="#5B5B5B", linestyle=(0, (3.0, 2.1)),
                    linewidth=1.1, label=r"$K^\star_{\mathrm{Full}}$",
                ),
                Line2D(
                    [0], [0], color="#0072B2", linestyle=(0, (3.0, 2.1)),
                    linewidth=1.1, label=r"$K^\star_{\mathrm{DW}}$",
                ),
            ]
        )
    return handles


def render_main(
    fitted: dict[str, object], ledger: dict[str, object], output: Path
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 4, figsize=(7.15, 1.72), dpi=220)

    for axis, (data, (label, H, ks)) in zip(axes, SPECS.items()):
        x = np.asarray(ks, dtype=np.float64)
        x_display = np.arange(len(ks), dtype=np.float64)
        visible: list[float] = []
        for arm, _, color, arm_offset in ARMS:
            block = fitted[data][arm]
            values = block["values"]
            curves = np.stack([fit["y_fit"] for fit in block["fits"]], axis=0)
            x_fit = block["fits"][0]["x_fit"]
            x_fit_display = np.interp(x_fit, x, x_display)
            curve_mean = curves.mean(axis=0)
            curve_sd = curves.std(axis=0, ddof=1)
            axis.fill_between(
                x_fit_display, curve_mean - curve_sd, curve_mean + curve_sd,
                color=color, alpha=0.13, linewidth=0.0, zorder=1,
            )
            axis.plot(
                x_fit_display, curve_mean, color=color, linewidth=1.65, zorder=3
            )
            optimum_index = int(np.argmin(curve_mean))
            optimum_k = float(x_fit[optimum_index])
            optimum_x = float(x_fit_display[optimum_index])
            optimum_y = float(curve_mean[optimum_index])
            axis.axvline(
                optimum_x, color=color, linestyle=(0, (3.0, 2.1)),
                linewidth=0.95, alpha=0.82, zorder=2,
            )
            axis.scatter(
                [optimum_x], [optimum_y], marker="v", s=20.0, color=color,
                edgecolor="white", linewidth=0.35, zorder=6,
            )
            ledger[data][arm]["mean_seed_fit_K_opt"] = optimum_k
            for seed in range(3):
                axis.scatter(
                    x_display + arm_offset, values[seed],
                    s=9.0, color=color, edgecolor="none", alpha=0.28, zorder=4,
                )
            visible.extend(values.ravel().tolist())
            visible.extend((curve_mean - curve_sd).tolist())
            visible.extend((curve_mean + curve_sd).tolist())

        span = float(np.ptp(visible))
        pad = max(0.07 * span, 0.004)
        axis.set_ylim(min(visible) - pad, max(visible) + pad)
        axis.set_xlim(-0.19, float(len(ks) - 1) + 0.19)
        axis.set_xticks(x_display, [str(value) for value in ks])
        axis.set_title(f"{label}, $H={H}$", fontsize=9.4, pad=2.7)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.tick_params(labelsize=7.35, width=0.75, length=2.3, pad=1.3)
        axis.grid(axis="y", color="#D9DDE2", linewidth=0.45, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)

    figure.legend(
        handles=legend_handles(include_optima=True), loc="upper center",
        bbox_to_anchor=(0.53, 0.97), ncol=4, frameon=False, fontsize=8.15,
        handlelength=1.65, columnspacing=1.05, handletextpad=0.38,
    )
    figure.supxlabel("training horizon $K$", x=0.53, y=0.055, fontsize=9.6)
    figure.supylabel("relative $L_2$", x=0.004, y=0.47, fontsize=9.6)
    figure.subplots_adjust(
        left=0.070, right=0.996, bottom=0.25, top=0.70, wspace=0.38
    )
    save_figure(figure, output, ledger)


def render_appendix(
    fitted: dict[str, object], ledger: dict[str, object], output: Path
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    # Four dataset columns by three seed rows.
    figure, axes = plt.subplots(3, 4, figsize=(7.15, 4.65), dpi=220)

    y_limits: dict[str, tuple[float, float]] = {}
    for data in SPECS:
        visible: list[float] = []
        for arm, _, _, _ in ARMS:
            block = fitted[data][arm]
            visible.extend(block["values"].ravel().tolist())
            for fit in block["fits"]:
                visible.extend(fit["y_fit"].tolist())
        span = float(np.ptp(visible))
        pad = max(0.07 * span, 0.004)
        y_limits[data] = (min(visible) - pad, max(visible) + pad)

    for row, seed in enumerate(range(3)):
        for col, (data, (label, H, ks)) in enumerate(SPECS.items()):
            axis = axes[row, col]
            x = np.asarray(ks, dtype=np.float64)
            x_display = np.arange(len(ks), dtype=np.float64)
            for arm, _, color, arm_offset in ARMS:
                block = fitted[data][arm]
                fit = block["fits"][seed]
                values = block["values"][seed]
                axis.plot(
                    np.interp(fit["x_fit"], x, x_display), fit["y_fit"], color=color,
                    linewidth=1.45, zorder=2,
                )
                axis.scatter(
                    x_display + arm_offset, values,
                    s=12.0, color=color, edgecolor="white", linewidth=0.25,
                    alpha=0.72, zorder=4,
                )
                axis.axvline(
                    np.interp(fit["K_opt"], x, x_display), color=color,
                    linestyle=(0, (3.0, 2.1)),
                    linewidth=1.0, alpha=0.90, zorder=1,
                )

            axis.set_xlim(-0.19, float(len(ks) - 1) + 0.19)
            axis.set_ylim(*y_limits[data])
            axis.set_xticks(x_display, [str(value) for value in ks])
            axis.tick_params(
                labelsize=7.0, width=0.7, length=2.1, pad=1.1,
                labelbottom=(row == 2),
            )
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
            axis.grid(axis="y", color="#D9DDE2", linewidth=0.42, zorder=0)
            axis.spines[["top", "right"]].set_visible(False)
            if row == 0:
                axis.set_title(f"{label}, $H={H}$", fontsize=9.0, pad=3.0)
            if col == 0:
                axis.set_ylabel(f"Seed {seed}\nrelative $L_2$", fontsize=8.4)

    figure.legend(
        handles=legend_handles(), loc="upper center", bbox_to_anchor=(0.54, 0.995),
        ncol=2, frameon=False, fontsize=8.8, handlelength=1.8,
        columnspacing=1.4, handletextpad=0.42,
    )
    figure.supxlabel("training horizon $K$", x=0.54, y=0.018, fontsize=9.4)
    figure.subplots_adjust(
        left=0.080, right=0.995, bottom=0.10, top=0.90,
        hspace=0.30, wspace=0.34,
    )
    save_figure(figure, output, ledger)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--appendix-output", type=Path, default=DEFAULT_APPENDIX_OUTPUT
    )
    args = parser.parse_args()
    fitted, ledger = collect(args.input_root)
    render_main(fitted, ledger, args.output)
    render_appendix(fitted, ledger, args.appendix_output)


if __name__ == "__main__":
    main()
