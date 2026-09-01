#!/usr/bin/env python3
"""Add the common-map drive coordinate to an existing completed fMRI JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "probes"))

from fmri_drive_history_ops import (
    collect_subjects as _collect,
    common_time as _common_time,
    future_targets as _targets,
    mse_by_subject as _mse_by_subject,
    valid_starts as _starts,
)

from internal_dw.datasets.hcp import build_hcp_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-path", required=True)
    args = parser.parse_args()

    row = json.loads(args.input.read_text(encoding="utf-8"))
    split_args = SimpleNamespace(
        data_path=str(args.data_path),
        movie=int(row.get("movie", 1)),
        seed=0,
        train_ratio=0.70,
        val_ratio=0.15,
        roi_dim=int(row.get("roi_dim", 400)),
        visual_only=False,
    )
    train_set, val_set, test_set = build_hcp_splits(split_args)
    train, train_subjects = _collect(train_set, "train")
    validation, validation_subjects = _collect(val_set, "validation")
    test, test_subjects = _collect(test_set, "test")
    train, validation, test = _common_time(train, validation, test)
    for key, actual in (
        ("train_subjects", train_subjects),
        ("validation_subjects", validation_subjects),
        ("test_subjects", test_subjects),
    ):
        if list(row[key]) != list(actual):
            raise ValueError(f"subject split mismatch while enriching {key}")

    fit = np.concatenate([train, validation], axis=0)
    shared = fit.mean(axis=0)
    test_residual = test - shared[None]
    horizons = [int(value) for value in row["horizons"]]
    window = int(row["selected_history_window"])
    score_stride = int(row["score_stride"])
    starts = _starts(test.shape[1], window, max(horizons), score_stride)
    residual_targets = _targets(test_residual, starts, horizons)
    state_targets = _targets(test, starts, horizons)
    drive_mse = _mse_by_subject(np.zeros_like(residual_targets), residual_targets)
    mean_mse = _mse_by_subject(np.zeros_like(state_targets), state_targets)
    drive_value = 1.0 - drive_mse.sum(axis=0) / np.maximum(mean_mse.sum(axis=0), 1e-30)

    old_drive = np.asarray(row["drive_only_mse_by_test_subject_and_horizon"])
    np.testing.assert_allclose(drive_mse, old_drive, rtol=2e-5, atol=1e-7)
    row["drive_value_definition"] = (
        "1 - MSE(shared movie response) / MSE(subject-centered zero mean)"
    )
    row["mean_baseline_mse_by_test_subject_and_horizon"] = mean_mse.tolist()
    row["drive_value"] = drive_value.tolist()
    row["enriched_from"] = str(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    print("fMRI shared-response drive value")
    for horizon, value in zip(horizons, drive_value):
        print(f"H{horizon:<2d} {value:+.4f}")
    print(f"[out] {args.output}")


if __name__ == "__main__":
    main()
