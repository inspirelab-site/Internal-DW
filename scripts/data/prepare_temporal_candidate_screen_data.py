#!/usr/bin/env python3
"""Prepare checkpoint-free inputs for temporal-regime screening.

This script performs only deterministic, unsupervised data preparation.  It
does not fit a forecasting model and never uses the official test split to
choose a variable, history window, or condition.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


GAIT_CONDITIONS = ("norm", "fast", "slow", "metnrm", "metfst", "metslw")
ETT_FILES = ("ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv")


def _calendar_features(stamps: Iterable[str]) -> np.ndarray:
    rows = []
    for stamp in stamps:
        dt = datetime.fromisoformat(str(stamp).strip().strip('"'))
        hour = dt.hour + dt.minute / 60.0
        day_of_week = dt.weekday()
        day_of_year = dt.timetuple().tm_yday - 1
        rows.append(
            (
                np.sin(2.0 * np.pi * hour / 24.0),
                np.cos(2.0 * np.pi * hour / 24.0),
                np.sin(2.0 * np.pi * day_of_week / 7.0),
                np.cos(2.0 * np.pi * day_of_week / 7.0),
                np.sin(2.0 * np.pi * day_of_year / 365.25),
                np.cos(2.0 * np.pi * day_of_year / 365.25),
            )
        )
    return np.asarray(rows, dtype=np.float32)


def _chunks(array: np.ndarray, length: int) -> np.ndarray:
    count = int(array.shape[0]) // int(length)
    if count < 1:
        raise ValueError(f"cannot make a length-{length} chunk from {array.shape}")
    return np.ascontiguousarray(
        array[: count * int(length)].reshape(count, int(length), *array.shape[1:]),
        dtype=np.float32,
    )


def _write(
    path: Path,
    states: tuple[np.ndarray, np.ndarray, np.ndarray],
    drives: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    metadata: dict,
    force: bool,
) -> None:
    if path.exists() and not force:
        print(f"[skip] {path}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_state": states[0],
        "validation_state": states[1],
        "test_state": states[2],
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    if drives is not None:
        payload.update(
            train_drive=drives[0],
            validation_drive=drives[1],
            test_drive=drives[2],
        )
    np.savez_compressed(path, **payload)
    print(
        f"[out] {path} train/val/test="
        f"{states[0].shape}/{states[1].shape}/{states[2].shape}",
        flush=True,
    )


def _load_gait_recordings(root: Path, condition: str, detrend: bool) -> list[np.ndarray]:
    files = sorted(root.glob(f"si*.{condition}"))
    if len(files) != 10:
        raise FileNotFoundError(
            f"expected 10 gait recordings for {condition}, found {len(files)} under {root}"
        )
    rows = []
    for path in files:
        values = np.loadtxt(path, dtype=np.float32).reshape(-1)
        if detrend:
            time = np.arange(len(values), dtype=np.float64)
            slope, intercept = np.polyfit(time, values.astype(np.float64), 1)
            values = (values - (slope * time + intercept)).astype(np.float32)
        rows.append(values[:, None])
    return rows


def _balanced_recording_chunks(
    recordings: list[np.ndarray], indices: np.ndarray, chunk: int
) -> np.ndarray:
    counts = [len(recordings[int(index)]) // int(chunk) for index in indices]
    keep = min(counts)
    if keep < 1:
        raise ValueError("a gait subject has no complete chunk")
    chunks = [_chunks(recordings[int(index)], chunk)[:keep] for index in indices]
    return np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.float32)


def _prepare_gait(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(int(args.seed))
    order = rng.permutation(10)
    train_ids, validation_ids, test_ids = order[:6], order[6:8], order[8:]
    for condition in GAIT_CONDITIONS:
        for detrend in (False, True):
            recordings = _load_gait_recordings(args.gait_root, condition, detrend)
            states = tuple(
                _balanced_recording_chunks(recordings, ids, args.chunk)
                for ids in (train_ids, validation_ids, test_ids)
            )
            suffix = "detrended" if detrend else "raw"
            metadata = {
                "dataset": "gait",
                "condition": condition,
                "preprocessing": suffix,
                "subject_split": {
                    "train": train_ids.tolist(),
                    "validation": validation_ids.tolist(),
                    "test": test_ids.tolist(),
                },
                "selection_inputs": "condition and detrending are prespecified controls",
                "official_test_reserved": True,
            }
            _write(
                args.output_root / f"gait_{condition}_{suffix}.npz",
                states,
                None,
                metadata,
                args.force,
            )


def _resolve_ett_small(root: Path) -> Path:
    candidates = (root / "ETT-small", root / "ETDataset-main" / "ETT-small")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"cannot find ETT-small below {root}")


def _read_ett(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if len(header) < 2 or header[0].lower() != "date":
            raise ValueError(f"unexpected ETT header in {path}: {header[:3]}")
        stamps, values = [], []
        for row in reader:
            if not row:
                continue
            stamps.append(row[0])
            values.append([float(value) for value in row[1:]])
    return stamps, np.asarray(values, dtype=np.float32)


def _prepare_ett(args: argparse.Namespace) -> None:
    root = _resolve_ett_small(args.ett_root)
    for filename in ETT_FILES:
        path = root / filename
        stamps, values = _read_ett(path)
        drive = _calendar_features(stamps)
        factor = 4 if filename.startswith("ETTm") else 1
        train_end = 12 * 30 * 24 * factor
        validation_end = train_end + 4 * 30 * 24 * factor
        test_end = validation_end + 4 * 30 * 24 * factor
        if len(values) < test_end:
            raise ValueError(
                f"{path} has {len(values)} rows, fewer than standard ETT split {test_end}"
            )
        slices = (
            slice(0, train_end),
            slice(train_end, validation_end),
            slice(validation_end, test_end),
        )
        states = tuple(_chunks(values[part], args.chunk) for part in slices)
        drives = tuple(_chunks(drive[part], args.chunk) for part in slices)
        label = filename[:-4].lower()
        metadata = {
            "dataset": label,
            "source": str(path),
            "frequency_minutes": 15 if factor == 4 else 60,
            "calendar_drive": "hour/day-of-week/day-of-year sine and cosine",
            "split": "standard ETT chronological 12/4/4 month split",
            "official_test_reserved": True,
        }
        _write(
            args.output_root / f"{label}.npz",
            states,
            drives,
            metadata,
            args.force,
        )


def _read_electricity_selected(path: Path, maximum_clients: int) -> tuple[list[str], np.ndarray, list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle, delimiter=";"))
    client_count = len(header) - 1
    candidate_count = min(client_count, max(int(maximum_clients) * 2, int(maximum_clients)))
    positions = np.unique(
        np.linspace(1, client_count, num=candidate_count, dtype=np.int64)
    ).tolist()
    selected_names = [header[position].strip('"') for position in positions]

    try:
        import pandas as pd  # type: ignore

        frame = pd.read_csv(
            path,
            sep=";",
            decimal=",",
            usecols=[0] + positions,
            dtype={name: "float32" for name in selected_names},
        )
        stamps = frame.iloc[:, 0].astype(str).tolist()
        values = frame.iloc[:, 1:].to_numpy(dtype=np.float32, copy=True)
    except ImportError:
        stamps, rows = [], []
        wanted = [0] + positions
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter=";")
            next(reader)
            for row in reader:
                if not row:
                    continue
                stamps.append(row[0].strip('"'))
                rows.append(
                    [float(row[position].replace(",", ".")) for position in positions]
                )
        values = np.asarray(rows, dtype=np.float32)

    # The archive is sampled every 15 minutes; hourly averaging removes the
    # arbitrary sub-hour phase and matches common Electricity benchmark usage.
    blocks = len(values) // 4
    values = values[: 4 * blocks].reshape(blocks, 4, -1).mean(axis=1)
    stamps = stamps[: 4 * blocks : 4]

    # Client selection uses only the chronological training prefix.  This
    # removes meters that were not installed during the training interval.
    train_end = int(0.60 * len(values))
    prefix = values[:train_end]
    coverage = np.mean(prefix > 0.0, axis=0)
    variation = np.std(prefix, axis=0)
    order = np.lexsort((-variation, -coverage))
    keep = order[: min(int(maximum_clients), len(order))]
    keep = keep[np.argsort(keep)]
    return stamps, np.ascontiguousarray(values[:, keep]), [selected_names[i] for i in keep]


def _prepare_electricity(args: argparse.Namespace) -> None:
    path = args.electricity_root / "LD2011_2014.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    stamps, values, clients = _read_electricity_selected(path, args.electricity_clients)
    drive = _calendar_features(stamps)
    train_end = int(0.60 * len(values))
    validation_end = int(0.80 * len(values))
    slices = (slice(0, train_end), slice(train_end, validation_end), slice(validation_end, None))
    states = tuple(_chunks(values[part], args.chunk) for part in slices)
    drives = tuple(_chunks(drive[part], args.chunk) for part in slices)
    metadata = {
        "dataset": "electricity",
        "source": str(path),
        "resampling": "15-minute load averaged to hourly",
        "clients": clients,
        "client_selection": "train-prefix coverage then variance from evenly spaced candidates",
        "calendar_drive": "hour/day-of-week/day-of-year sine and cosine",
        "split": "chronological 60/20/20",
        "official_test_reserved": True,
    }
    _write(
        args.output_root / "electricity.npz",
        states,
        drives,
        metadata,
        args.force,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("all", "gait", "ett", "electricity"), default="all")
    parser.add_argument("--gait-root", type=Path, required=True)
    parser.add_argument("--ett-root", type=Path, required=True)
    parser.add_argument("--electricity-root", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("probe_inputs/temporal_candidate_regime_v1"),
    )
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--electricity-clients", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.only in ("all", "gait"):
        _prepare_gait(args)
    if args.only in ("all", "ett"):
        _prepare_ett(args)
    if args.only in ("all", "electricity"):
        _prepare_electricity(args)


if __name__ == "__main__":
    main()
