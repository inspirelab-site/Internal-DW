#!/usr/bin/env python3
"""Fast integrity check for the local normalized WeatherBench-2 pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/weatherbench2_1p5_pilot")
    parser.add_argument("--sample-times", type=int, default=64)
    args = parser.parse_args()

    root = Path(args.data)
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    channels = int(meta["channels"])
    height = int(meta["height"])
    width = int(meta["width"])
    if channels != len(meta["channel_names"]):
        raise ValueError("channel count/name mismatch")

    summaries = {}
    for split in ("train", "val", "test"):
        info = meta["splits"][split]
        state = np.load(root / info["state_file"], mmap_mode="r")
        features = np.load(root / info["time_features_file"], mmap_mode="r")
        expected = (int(info["steps"]), channels, height, width)
        if state.dtype != np.float32 or tuple(state.shape) != expected:
            raise ValueError(
                f"{split}: state is {state.dtype} {state.shape}, expected float32 {expected}"
            )
        if features.dtype != np.float32 or features.shape != (expected[0], 4):
            raise ValueError(f"{split}: invalid time features {features.dtype} {features.shape}")

        n_sample = min(max(2, int(args.sample_times)), expected[0])
        times = np.linspace(0, expected[0] - 1, n_sample, dtype=np.int64)
        # Spatial stride keeps the check below a few MiB while covering every
        # channel, season, and part of the globe.
        sample = np.asarray(state[times, :, ::8, ::8], dtype=np.float32)
        if not np.isfinite(sample).all() or not np.isfinite(features).all():
            raise ValueError(f"{split}: non-finite values")
        if np.any(np.max(np.abs(sample), axis=(0, 2, 3)) == 0):
            raise ValueError(f"{split}: at least one channel is identically zero in the audit sample")
        summaries[split] = {
            "shape": list(expected),
            "sample_mean": float(sample.mean(dtype=np.float64)),
            "sample_std": float(sample.std(dtype=np.float64)),
            "sample_abs_max": float(np.abs(sample).max()),
        }

    train_mean = summaries["train"]["sample_mean"]
    train_std = summaries["train"]["sample_std"]
    if abs(train_mean) > 0.25 or not 0.65 <= train_std <= 1.35:
        raise ValueError(
            f"train normalization audit failed: mean={train_mean:.4f}, std={train_std:.4f}"
        )
    print(json.dumps({"status": "ok", "data": str(root), "splits": summaries}, indent=2))


if __name__ == "__main__":
    main()
