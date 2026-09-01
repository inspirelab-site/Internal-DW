#!/usr/bin/env python3
"""Plot the predicted Driven-MG regime path against a scaled simulator path.

At each drive strength the predicted point is the midpoint of the linear and
nonlinear readout estimates.  The crossed-simulator reference is rescaled by
one nonnegative multiplicative factor per axis (no translation or rotation),
so its ordering and trajectory shape remain unchanged while its magnitude is
comparable to the finite-readout coordinate system.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_figure_style import apply_paper_style


DEFAULT_SCALES = tuple(round(value / 100.0, 2) for value in range(0, 9))


def _parse_scales(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values or len(set(values)) != len(values):
        raise ValueError(f"invalid unique scale list: {text!r}")
    if min(values) < 0:
        raise ValueError("drive strengths must be nonnegative")
    return tuple(sorted(values))


def _tag(scale: float) -> str:
    return f"{scale:.2f}".replace(".", "p")


def _drive_digest(path: Path) -> tuple[str, dict]:
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
        for split in ("train", "validation", "test"):
            array = np.ascontiguousarray(archive[f"{split}_drive"])
            digest.update(split.encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(array.view(np.uint8))
    return digest.hexdigest(), metadata


def _load(
    result_root: Path, data_root: Path, expected_scales: tuple[float, ...]
) -> tuple[list[dict], dict]:
    rows = []
    drive_digests = []
    reference_splits = None
    for scale in expected_scales:
        tag = _tag(scale)
        result_path = result_root / f"ds{tag}.json"
        data_path = data_root / f"mg_ds{tag}.npz"
        if not result_path.is_file() or not data_path.is_file():
            raise FileNotFoundError(
                f"missing scale={scale:.2f}: result={result_path.is_file()} "
                f"data={data_path.is_file()}"
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        recorded = float(
            result["input_metadata"]["parameters"]["drive_scale"]
        )
        if not np.isclose(recorded, scale, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"{result_path} records drive_scale={recorded}, expected {scale}"
            )
        if bool(result.get("official_test_touched", True)):
            raise ValueError(f"official test was touched at drive_scale={scale}")

        digest, metadata = _drive_digest(data_path)
        drive_digests.append(digest)
        splits = metadata.get("split_indices")
        if reference_splits is None:
            reference_splits = splits
        elif splits != reference_splits:
            raise ValueError(f"split indices differ at drive_scale={scale}")

        row = {
            "drive_scale": scale,
            "selected_history_window": int(result["selected_history_window"]),
            "result_json": str(result_path),
            "data_npz": str(data_path),
        }
        for method in ("linear", "nonlinear"):
            if "shapley" not in result or method not in result["shapley"]:
                raise ValueError(
                    f"{result_path} lacks H-only/Shapley results; rerun the readout probe"
                )
            summary = result["long_horizon_summary"][method]
            shapley = result["shapley"][method]
            row[method] = {
                "history_value": float(summary["mean_long_history_value"]),
                "history_null": float(summary["mean_long_null_history_value"]),
                "history_beyond_null": float(
                    summary["mean_long_history_beyond_null"]
                ),
                "drive_value": float(summary["mean_long_drive_value"]),
                "history_shapley": float(shapley["mean_long_history_shapley"]),
                "drive_shapley": float(shapley["mean_long_drive_shapley"]),
                "null_history_shapley": float(
                    shapley["mean_long_null_history_shapley"]
                ),
                "delta_history_shapley": float(
                    shapley["mean_long_history_shapley_beyond_null"]
                ),
                "explained_value": float(shapley["mean_long_explained_value"]),
            }
        rows.append(row)

    if len(set(drive_digests)) != 1:
        raise ValueError(
            "drive arrays differ across strengths; the sweep is not common-random-number paired"
        )
    checks = {
        "common_drive_sha256": drive_digests[0],
        "common_drive_realizations": True,
        "common_split_indices": True,
        "official_test_untouched": True,
    }
    return rows, checks


def _limits(values: list[float]) -> tuple[float, float]:
    low = min(min(values), 0.0)
    high = max(max(values), 0.0)
    span = max(high - low, 0.05)
    return low - 0.10 * span, high + 0.12 * span


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("probe_outputs/driven_mg_strength_sweep_v1/points"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("artifacts/driven_mg_strength_sweep_v1/data"),
    )
    parser.add_argument(
        "--scales", default=",".join(f"{value:.2f}" for value in DEFAULT_SCALES)
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "probe_outputs/driven_mg_strength_sweep_v1/drive_history_vs_strength"
        ),
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path(
            "probe_outputs/driven_mg_strength_sweep_v1/simulator_ground_truth.json"
        ),
        help="Optional simulator crossed-Shapley reference.",
    )
    args = parser.parse_args()

    scales = _parse_scales(args.scales)
    rows, checks = _load(args.result_root, args.data_root, scales)
    ground_truth = None
    if args.ground_truth.exists():
        ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
        truth_scales = [float(row["drive_scale"]) for row in ground_truth["points"]]
        if len(truth_scales) != len(scales) or not np.allclose(truth_scales, scales):
            raise ValueError(
                f"ground-truth scales {truth_scales} do not match requested {list(scales)}"
            )

    apply_paper_style()
    figure, axis = plt.subplots(figsize=(3.15, 2.35), facecolor="white")
    scale_values = np.asarray([row["drive_scale"] for row in rows], dtype=np.float64)
    key_indices = {
        index
        for index, scale in enumerate(scale_values)
        if np.isclose(scale, 0.0) or np.isclose(scale, 0.03) or np.isclose(scale, 0.08)
    }
    linear_x = np.asarray(
        [row["linear"]["drive_shapley"] for row in rows], dtype=np.float64
    )
    nonlinear_x = np.asarray(
        [row["nonlinear"]["drive_shapley"] for row in rows], dtype=np.float64
    )
    linear_y = np.asarray(
        [row["linear"]["delta_history_shapley"] for row in rows], dtype=np.float64
    )
    nonlinear_y = np.asarray(
        [row["nonlinear"]["delta_history_shapley"] for row in rows],
        dtype=np.float64,
    )
    predicted_x = 0.5 * (linear_x + nonlinear_x)
    predicted_y = 0.5 * (linear_y + nonlinear_y)
    all_x, all_y = predicted_x.tolist(), predicted_y.tolist()

    axis.plot(
        predicted_x,
        predicted_y,
        label="predicted",
        color="#1976B3",
        marker="o",
        markersize=4.2,
        markeredgecolor="white",
        markeredgewidth=0.55,
        linewidth=1.45,
        zorder=4,
    )
    if len(predicted_x) >= 2:
        axis.annotate(
            "",
            xy=(predicted_x[-1], predicted_y[-1]),
            xytext=(predicted_x[-2], predicted_y[-2]),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#1976B3",
                "linewidth": 1.25,
                "mutation_scale": 8,
                "shrinkA": 4,
                "shrinkB": 4,
            },
            zorder=5,
        )

    truth_axis_scale = None
    if ground_truth is not None:
        raw_truth_x = np.asarray(
            [row["drive_shapley_fraction"] for row in ground_truth["points"]],
            dtype=np.float64,
        )
        raw_truth_y = np.asarray(
            [row["history_shapley_beyond_null"] for row in ground_truth["points"]],
            dtype=np.float64,
        )
        # Positive least-squares scaling changes magnitude only.  It cannot
        # translate, rotate, reverse, or otherwise manufacture tracking.
        drive_scale = max(
            0.0,
            float(np.dot(raw_truth_x, predicted_x) / max(np.dot(raw_truth_x, raw_truth_x), 1e-30)),
        )
        history_scale = max(
            0.0,
            float(np.dot(raw_truth_y, predicted_y) / max(np.dot(raw_truth_y, raw_truth_y), 1e-30)),
        )
        truth_x = drive_scale * raw_truth_x
        truth_y = history_scale * raw_truth_y
        truth_axis_scale = {"drive": drive_scale, "history": history_scale}
        all_x.extend(truth_x.tolist())
        all_y.extend(truth_y.tolist())
        axis.plot(
            truth_x,
            truth_y,
            label="scaled ground truth",
            color="#303030",
            linestyle=(0, (2.2, 1.8)),
            marker="o",
            markersize=3.8,
            markerfacecolor="white",
            markeredgecolor="#303030",
            markeredgewidth=0.8,
            linewidth=1.05,
            zorder=3,
        )
        for index in sorted(key_indices):
            scale = scale_values[index]
            offset = (4, 4) if not np.isclose(scale, 0.08) else (4, -8)
            axis.annotate(
                rf"$\lambda_D={scale:.2f}$",
                (truth_x[index], truth_y[index]),
                xytext=offset,
                textcoords="offset points",
                color="#303030",
                fontsize=5.6,
                zorder=6,
            )

    axis.axhline(0.0, color="#777777", linestyle=(0, (3, 3)), linewidth=0.75)
    axis.axvline(0.0, color="#999999", linestyle=(0, (2, 3)), linewidth=0.70)
    axis.set_xlim(*_limits(all_x))
    axis.set_ylim(*_limits(all_y))
    axis.set_xlabel(r"drive contribution  $\phi_D$", fontsize=7.2, labelpad=2)
    axis.set_ylabel(
        r"history beyond null  $\Delta\phi_H$", fontsize=7.2, labelpad=2
    )
    axis.tick_params(axis="both", labelsize=6.3, length=2.6, width=0.7)
    axis.grid(color="#DDE2E8", linewidth=0.55, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        loc="upper right",
        frameon=False,
        fontsize=6.2,
        handlelength=1.45,
        labelspacing=0.25,
        borderaxespad=0.35,
    )
    figure.subplots_adjust(left=0.20, right=0.985, bottom=0.20, top=0.88)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    png = args.output.with_suffix(".png")
    pdf = args.output.with_suffix(".pdf")
    summary = args.output.with_suffix(".json")
    figure.savefig(png, dpi=320, bbox_inches="tight", pad_inches=0.025)
    figure.savefig(pdf, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)
    summary.write_text(
        json.dumps(
            {
                "experiment": "driven_mg_strength_regime_trajectory",
                "drive_scales": list(scales),
                "long_horizon_definition": "mean over k >= 8",
                "history_quantity": "mean_long_history_beyond_null",
                "drive_quantity": "mean_long_drive_value",
                "checks": checks,
                "simulator_reference": ground_truth,
                "predicted_midpoint": {
                    "drive_shapley": predicted_x.tolist(),
                    "history_shapley_beyond_null": predicted_y.tolist(),
                },
                "simulator_axis_scale": truth_axis_scale,
                "comparability_note": (
                    "The predicted path is the linear/nonlinear midpoint. The "
                    "simulator path receives one nonnegative least-squares "
                    "multiplicative scale per axis, with no translation, rotation, "
                    "or reversal; its ordering and trajectory shape are unchanged."
                    if ground_truth is not None
                    else None
                ),
                "points": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[out] {png}")
    print(f"[out] {pdf}")
    print(f"[out] {summary}")


if __name__ == "__main__":
    main()
