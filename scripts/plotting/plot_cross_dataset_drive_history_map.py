#!/usr/bin/env python3
"""Plot the common long-horizon drive--history map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_figure_style import apply_paper_style


# Match the smaller wrapfigure used in the paper.  Keeping the physical font
# sizes unchanged while reducing the canvas makes the map more compact without
# shrinking its labels in the final PDF.
WRAP_WIDTH_IN = 0.38 * 5.5


def _spec(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"expected LABEL=JSON, got {text!r}")
    label, path = text.split("=", 1)
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--fmri", required=True, type=Path)
    parser.add_argument("--long-horizon-min", type=int, default=8)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    points = []
    for label, path in [_spec(value) for value in args.dataset]:
        row = json.loads(path.read_text(encoding="utf-8"))
        for method in ("linear", "nonlinear"):
            summary = row["long_horizon_summary"][method]
            points.append(
                {
                    "dataset": label,
                    "method": method,
                    "drive": summary["mean_long_drive_value"],
                    "history": summary["mean_long_history_value"],
                    "null": summary["mean_long_null_history_value"],
                }
            )

    fmri = json.loads(args.fmri.read_text(encoding="utf-8"))
    if "drive_value" not in fmri:
        raise ValueError("fMRI JSON predates drive_value; rerun the updated probe")
    columns = [
        index
        for index, horizon in enumerate(fmri["horizons"])
        if int(horizon) >= int(args.long_horizon_min)
    ]
    if not columns:
        raise ValueError("fMRI has no requested long horizons")
    drive = float(np.mean(np.asarray(fmri["drive_value"])[columns]))
    fmri_methods = {
        "linear": ("linear_ridge", "circular_shift_linear_null"),
        "nonlinear": ("nonlinear_local_analog", "circular_shift_analog_null"),
    }
    for method, (key, null_key) in fmri_methods.items():
        points.append(
            {
                "dataset": "fMRI",
                "method": method,
                "drive": drive,
                "history": float(
                    np.mean(np.asarray(fmri["methods"][key]["history_value"])[columns])
                ),
                "null": float(
                    np.mean(
                        np.asarray(fmri["methods"][null_key]["history_value"])[columns]
                    )
                ),
            }
        )

    datasets = list(dict.fromkeys(point["dataset"] for point in points))
    # Average the two probes before clipping.  The main-text map displays only
    # detected predictive value; averaging after clipping would spuriously
    # preserve a positive coordinate when the probes disagree around zero.
    averaged_points = []
    for dataset in datasets:
        rows = [point for point in points if point["dataset"] == dataset]
        mean_drive = float(np.mean([row["drive"] for row in rows]))
        mean_history = float(
            np.mean([row["history"] - row["null"] for row in rows])
        )
        averaged_points.append(
            {
                "dataset": dataset,
                "drive_unclipped": mean_drive,
                "history_beyond_null_unclipped": mean_history,
                "drive": max(0.0, mean_drive),
                "history_beyond_null": max(0.0, mean_history),
                "n_probes": len(rows),
            }
        )
    fixed_colors = {
        "MG": "#0072B2",
        "MG+drive": "#4D4D4D",
        "NARMA": "#E66101",
        "iEEG": "#D73027",
        "Shear": "#E69F00",
        "WB2": "#009E73",
        "fMRI": "#8C510A",
        "SEVIR": "#5D6D7E",
        "ETTm1": "#CC79A7",
        "ETTm2": "#5E3C99",
    }
    fallback = plt.get_cmap("tab10")(np.linspace(0.0, 0.82, len(datasets)))
    colors = {
        dataset: fixed_colors.get(dataset, fallback[index])
        for index, dataset in enumerate(datasets)
    }
    apply_paper_style()
    plt.rcParams.update(
        {
            "axes.labelcolor": "#30343B",
            "xtick.color": "#4D535C",
            "ytick.color": "#4D535C",
        }
    )
    figure, axis = plt.subplots(
        figsize=(WRAP_WIDTH_IN, WRAP_WIDTH_IN * 2.25 / 3.55), facecolor="white"
    )
    axis.set_facecolor("#FCFCFD")
    # Two pairs are effectively coincident after averaging and clipping.
    # Split-color markers keep both datasets visible without moving their data.
    overlap_groups = [("MG", "ETTm1"), ("fMRI", "WB2")]
    overlap_coordinates = {}
    for left_dataset, right_dataset in overlap_groups:
        group = {left_dataset, right_dataset}
        overlap_rows = [
            point for point in averaged_points if point["dataset"] in group
        ]
        if len(overlap_rows) != 2:
            continue
        x_pair = [point["drive"] for point in overlap_rows]
        y_pair = [point["history_beyond_null"] for point in overlap_rows]
        overlap_xy = (float(np.mean(x_pair)), float(np.mean(y_pair)))
        overlap_coordinates.update(
            {left_dataset: overlap_xy, right_dataset: overlap_xy}
        )
        axis.plot(
            [overlap_xy[0]],
            [overlap_xy[1]],
            linestyle="none",
            marker="o",
            markersize=7.8,
            fillstyle="left",
            markerfacecolor=colors[left_dataset],
            markerfacecoloralt=colors[right_dataset],
            markeredgecolor="white",
            markeredgewidth=0.9,
            zorder=5,
        )

    for point in averaged_points:
        dataset = point["dataset"]
        x_value = point["drive"]
        y_value = point["history_beyond_null"]
        if dataset not in overlap_coordinates:
            axis.scatter(
                x_value,
                y_value,
                s=58,
                marker="o",
                color=colors[dataset],
                edgecolor="white",
                linewidth=0.9,
                zorder=3,
            )

    axis.axhline(0.0, color="#68707A", linestyle=(0, (4, 3)), linewidth=0.8, zorder=1)
    axis.axvline(0.0, color="#9AA1AA", linestyle=(0, (1.5, 2.5)), linewidth=0.7, zorder=1)
    axis.set_xlim(-0.035, 0.64)
    axis.set_ylim(-0.035, 0.77)
    axis.set_xticks([0.0, 0.2, 0.4, 0.6])
    axis.set_yticks([0.0, 0.2, 0.4, 0.6])
    axis.set_xlabel(
        "drive-only gain over\n" + r"the mean predictor  $\Psi_D$",
        fontsize=7.1,
        labelpad=3,
    )
    axis.set_ylabel(
        "additional\n" + r"history gain  $\Delta\Psi_H$",
        fontsize=7.1,
        labelpad=3,
    )
    axis.tick_params(axis="both", labelsize=6.2, width=0.75, length=2.8)
    axis.grid(color="#DDE2E8", alpha=0.7, linewidth=0.55)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#525860")
    axis.spines["bottom"].set_color("#525860")
    axis.spines["left"].set_linewidth(0.9)
    axis.spines["bottom"].set_linewidth(0.9)
    # Keep a single compact column in the empty far-right region so the legend
    # is easy to scan and never covers the Driven-MG intervention point.
    legend_items = [
        "MG",
        "MG+drive",
        "NARMA",
        "iEEG",
        "Shear",
        "WB2",
        "ETTm1",
        "ETTm2",
        "fMRI",
    ]
    display_names = {"MG+drive": "Driven MG"}
    legend_x = 0.735
    legend_y0 = 0.955
    legend_dy = 0.082
    for row_index, dataset in enumerate(legend_items):
        y_pos = legend_y0 - row_index * legend_dy
        axis.scatter(
            [legend_x],
            [y_pos],
            s=23,
            marker="o",
            color=colors[dataset],
            edgecolor="white",
            linewidth=0.65,
            transform=axis.transAxes,
            clip_on=False,
            zorder=8,
        )
        axis.text(
            legend_x + 0.040,
            y_pos,
            display_names.get(dataset, dataset),
            transform=axis.transAxes,
            ha="left",
            va="center",
            fontsize=5.8,
            color="#20242A",
            zorder=8,
        )
    # Keep the map compact at half-column width: the axes are deliberately
    # shorter relative to the labels and markers, while their scales remain
    # linear so geometric distances retain their meaning.
    figure.subplots_adjust(left=0.24, right=0.925, bottom=0.255, top=0.945)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    png = args.output.with_suffix(".png")
    pdf = args.output.with_suffix(".pdf")
    json_path = args.output.with_suffix(".json")
    figure.savefig(
        png, dpi=320, bbox_inches="tight", pad_inches=0.015, facecolor="white"
    )
    figure.savefig(pdf, bbox_inches="tight", pad_inches=0.015, facecolor="white")
    json_path.write_text(
        json.dumps(
            {
                "long_horizon_min": args.long_horizon_min,
                "aggregation": "mean linear/nonlinear probe values, then clip each coordinate at zero",
                "points": averaged_points,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[out] {png}")
    print(f"[out] {pdf}")
    print(f"[out] {json_path}")


if __name__ == "__main__":
    main()
