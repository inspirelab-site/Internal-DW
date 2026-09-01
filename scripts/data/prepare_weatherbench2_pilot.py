#!/usr/bin/env python
"""Extract a manageable multi-level WeatherBench-2 1.5-degree pilot.

The official remote archive is read in its native eight-time-step chunks.  The
selected fields are written once to local float32 ``.npy`` memory maps and then
normalized in place with train-only per-channel statistics.  Training never
reads from the network.

Default state (29 channels):

  geopotential, temperature, u wind, v wind, specific humidity
      at 1000, 850, 700, 500, 250 hPa                         25
  surface pressure, 2m temperature, 10m u wind, 10m v wind    4

Default pilot years are deliberately small enough for mechanism qualification:
2007--2015 train, 2016 validation, 2017 test.  Expand the train years only after
the exact model shows both forecast skill and expanding tangent dynamics.

Dependencies in the server environment:

  python -m pip install "xarray>=2023.1" "zarr<3" gcsfs

Usage:

  python scripts/data/prepare_weatherbench2_pilot.py \
      --out data/weatherbench2_1p5_pilot
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_SOURCE = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
)
DEFAULT_PRESSURE = (
    "geopotential", "temperature", "u_component_of_wind",
    "v_component_of_wind", "specific_humidity",
)
DEFAULT_SURFACE = (
    "surface_pressure", "2m_temperature",
    "10m_u_component_of_wind", "10m_v_component_of_wind",
)
DEFAULT_LEVELS = (1000, 850, 700, 500, 250)


def _year_range(spec: str):
    lo, hi = str(spec).split("-")
    return int(lo), int(hi)


def _time_features(times) -> np.ndarray:
    times = np.asarray(times).astype("datetime64[h]")
    hour = (times - times.astype("datetime64[D]")).astype("timedelta64[h]").astype(np.float32)
    day = (
        times.astype("datetime64[D]") - times.astype("datetime64[Y]")
    ).astype("timedelta64[D]").astype(np.float32)
    two_pi = 2.0 * np.pi
    return np.stack(
        [
            np.sin(two_pi * hour / 24.0), np.cos(two_pi * hour / 24.0),
            np.sin(two_pi * day / 365.25), np.cos(two_pi * day / 365.25),
        ],
        axis=1,
    ).astype(np.float32)


def _indices_for_years(times, years_spec: str) -> np.ndarray:
    lo, hi = _year_range(years_spec)
    years = np.asarray(times).astype("datetime64[Y]").astype(np.int64) + 1970
    return np.flatnonzero((years >= lo) & (years <= hi))


def _fetch_block(ds, sl, pressure_vars, levels, surface_vars) -> np.ndarray:
    fields = []
    for variable in pressure_vars:
        value = (
            ds[variable]
            .isel(time=sl)
            .sel(level=list(levels))
            .transpose("time", "level", "latitude", "longitude")
            .values
        )
        fields.append(np.asarray(value, dtype=np.float32))
    for variable in surface_vars:
        value = (
            ds[variable]
            .isel(time=sl)
            .transpose("time", "latitude", "longitude")
            .values
        )
        fields.append(np.asarray(value, dtype=np.float32)[:, None])
    block = np.concatenate(fields, axis=1)
    if not np.isfinite(block).all():
        raise ValueError(f"non-finite WeatherBench2 values in time slice {sl}")
    return block


def _filled_prefix(mmap: np.ndarray, block_times: int) -> int:
    """Recover the contiguous written prefix of a preallocated raw mmap.

    ``open_memmap`` leaves unwritten pages as exact zeros.  Atmospheric state
    fields are nonzero at several independent probe coordinates, so four
    sparse scalar traces identify the first unwritten time without scanning
    the full (tens-of-GiB) array.  We round down to a native block boundary and
    safely overwrite a block if the previous process stopped mid-assignment.
    """
    _, channels, height, width = mmap.shape
    probes = []
    coordinates = (
        (0, height // 2, width // 2),
        (min(1, channels - 1), height // 3, width // 3),
        (min(2, channels - 1), (2 * height) // 3, (2 * width) // 3),
        (channels - 1, height // 2, width // 4),
    )
    for channel, row, column in coordinates:
        probes.append(np.asarray(mmap[:, channel, row, column]))
    written = np.logical_or.reduce([np.not_equal(value, 0.0) for value in probes])
    missing = np.flatnonzero(~written)
    prefix = int(missing[0]) if missing.size else int(mmap.shape[0])
    if prefix < mmap.shape[0]:
        prefix = (prefix // block_times) * block_times
    return prefix


def _accumulate_stats(block: np.ndarray, total: np.ndarray,
                      total_sq: np.ndarray) -> int:
    total += block.sum(axis=(0, 2, 3), dtype=np.float64)
    total_sq += np.square(block, dtype=np.float64).sum(
        axis=(0, 2, 3), dtype=np.float64
    )
    return int(block.shape[0] * block.shape[2] * block.shape[3])


def _write_progress(path: Path, progress: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--out", default="data/weatherbench2_1p5_pilot")
    parser.add_argument("--pressure-vars", nargs="+", default=list(DEFAULT_PRESSURE))
    parser.add_argument("--surface-vars", nargs="+", default=list(DEFAULT_SURFACE))
    parser.add_argument("--levels", nargs="+", type=int, default=list(DEFAULT_LEVELS))
    parser.add_argument("--train-years", default="2007-2015")
    parser.add_argument("--val-years", default="2016-2016")
    parser.add_argument("--test-years", default="2017-2017")
    parser.add_argument("--block-times", type=int, default=8)
    parser.add_argument(
        "--download-workers", type=int, default=4,
        help="Concurrent native Zarr-block reads. Four usually saturates one server link.",
    )
    parser.add_argument(
        "--resume-train-at", type=int, default=-1,
        help="Known safe raw-train prefix for a legacy partial without a progress sidecar.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        import xarray as xr
        import gcsfs  # noqa: F401 - registers the gs:// filesystem
        import zarr  # noqa: F401 - registers xarray's Zarr backend
    except ImportError as exc:
        raise SystemExit(
            "Missing WeatherBench-2 preprocessing dependencies in this Python environment. "
            "For the server's Python 3.9 environment run: python -m pip install "
            "'zarr==2.18.2' 'numcodecs==0.12.1' 'gcsfs==2024.6.1'"
        ) from exc
    if "zarr" not in xr.backends.list_engines():
        raise SystemExit(
            "xarray cannot see its Zarr backend. Verify this same Python with: "
            "python -c \"import xarray,zarr,gcsfs; print(xarray.backends.list_engines())\""
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metadata_path = out / "metadata.json"
    progress_path = out / "download_progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    else:
        progress = {}
    if metadata_path.exists() and not args.overwrite:
        print(f"[skip] completed dataset already exists: {metadata_path}")
        return

    print(f"[open] {args.source}", flush=True)
    ds = xr.open_zarr(
        args.source, consolidated=True, chunks=None,
        storage_options={"token": "anon"},
    )
    required = list(args.pressure_vars) + list(args.surface_vars)
    missing = [name for name in required if name not in ds]
    if missing:
        raise KeyError(f"source is missing variables: {missing}")
    available_levels = {int(x) for x in np.asarray(ds.level.values)}
    missing_levels = [x for x in args.levels if x not in available_levels]
    if missing_levels:
        raise KeyError(f"source is missing pressure levels: {missing_levels}")

    times = np.asarray(ds.time.values)
    latitude = np.asarray(ds.latitude.values, dtype=np.float32)
    longitude = np.asarray(ds.longitude.values, dtype=np.float32)
    channel_names = [
        f"{variable}_{level}"
        for variable in args.pressure_vars for level in args.levels
    ] + list(args.surface_vars)
    channels = len(channel_names)
    height, width = int(latitude.size), int(longitude.size)
    block_times = max(1, int(args.block_times))
    download_workers = max(1, int(args.download_workers))
    split_specs = {
        "train": args.train_years,
        "val": args.val_years,
        "test": args.test_years,
    }

    split_meta = {}
    train_sum = np.zeros(channels, dtype=np.float64)
    train_sumsq = np.zeros(channels, dtype=np.float64)
    train_count = 0
    partial_paths = {}

    for split, years_spec in split_specs.items():
        indices = _indices_for_years(times, years_spec)
        if indices.size == 0:
            raise ValueError(f"no source times for {split} years {years_spec}")
        if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            raise ValueError(f"{split} time selection is not contiguous")
        state_name = f"{split}_state.npy"
        partial = out / f"{state_name}.partial"
        final = out / state_name
        if args.overwrite:
            for path in (partial, final, out / f"{split}_time_features.npy"):
                if path.exists():
                    path.unlink()
        elif final.exists():
            raise FileExistsError(
                f"{final} exists but metadata.json does not.  This may be a partially "
                "normalized run and cannot be resumed safely; inspect it, then use --overwrite."
            )

        n_time = int(indices.size)
        expected_shape = (n_time, channels, height, width)
        if partial.exists():
            mmap = np.load(partial, mmap_mode="r+")
            if mmap.dtype != np.float32 or tuple(mmap.shape) != expected_shape:
                raise ValueError(
                    f"cannot resume {partial}: got {mmap.dtype} {mmap.shape}, "
                    f"expected float32 {expected_shape}"
                )
            known_prefix = int(progress.get(split, -1))
            if split == "train" and int(args.resume_train_at) >= 0:
                known_prefix = max(known_prefix, int(args.resume_train_at))
            resume_at = (
                known_prefix if known_prefix >= 0
                else _filled_prefix(mmap, block_times)
            )
            if not 0 <= resume_at <= n_time or resume_at % block_times:
                raise ValueError(
                    f"invalid resume prefix for {split}: {resume_at}; "
                    f"expected a multiple of {block_times} in [0,{n_time}]"
                )
            print(f"[{split}] resuming raw mmap at {resume_at}/{n_time}", flush=True)
        else:
            mmap = np.lib.format.open_memmap(
                partial, mode="w+", dtype=np.float32, shape=expected_shape,
            )
            resume_at = 0
        source_start = int(indices[0])
        print(
            f"[{split}] years={years_spec}, steps={n_time}, "
            f"shape={mmap.shape}, raw={mmap.nbytes / 1024**3:.1f} GiB",
            flush=True,
        )
        # Reconstruct train statistics for the prefix written by an earlier
        # process.  This is local sequential I/O and is much cheaper than
        # redownloading the same remote chunks.
        if split == "train" and resume_at:
            print(f"  {split}: recovering statistics for 0:{resume_at}", flush=True)
            for start in range(0, resume_at, 32):
                stop = min(start + 32, resume_at)
                recovered = np.asarray(mmap[start:stop])
                if (not np.isfinite(recovered).all()
                        or np.any(np.max(np.abs(recovered), axis=(1, 2, 3)) == 0)):
                    raise ValueError(
                        f"resume prefix contains an invalid/unwritten state in {start}:{stop}; "
                        "lower --resume-train-at to the last known safe progress line"
                    )
                train_count += _accumulate_stats(
                    recovered, train_sum, train_sumsq
                )

        def fetch(local_start):
            local_stop = min(local_start + block_times, n_time)
            source_slice = slice(
                source_start + local_start, source_start + local_stop
            )
            value = _fetch_block(
                ds, source_slice, args.pressure_vars, args.levels, args.surface_vars
            )
            return local_start, local_stop, value

        starts = iter(range(resume_at, n_time, block_times))
        written_since_resume = 0
        safe_prefix = resume_at
        completed_starts = set()
        next_report = ((resume_at // (block_times * 100)) + 1) * block_times * 100
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=download_workers
        ) as executor:
            pending = set()
            for _ in range(download_workers):
                try:
                    pending.add(executor.submit(fetch, next(starts)))
                except StopIteration:
                    break
            while pending:
                done, pending = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    local_start, local_stop, block = future.result()
                    mmap[local_start:local_stop] = block
                    completed_starts.add(local_start)
                    written_since_resume += local_stop - local_start
                    if split == "train":
                        train_count += _accumulate_stats(
                            block, train_sum, train_sumsq
                        )
                    try:
                        pending.add(executor.submit(fetch, next(starts)))
                    except StopIteration:
                        pass
                while safe_prefix in completed_starts:
                    completed_starts.remove(safe_prefix)
                    safe_prefix = min(safe_prefix + block_times, n_time)
                completed = min(n_time, resume_at + written_since_resume)
                if completed >= next_report or not pending:
                    mmap.flush()
                    progress[split] = int(safe_prefix)
                    _write_progress(progress_path, progress)
                    print(f"  {split}: {completed}/{n_time}", flush=True)
                    while next_report <= completed:
                        next_report += block_times * 100
        mmap.flush()
        del mmap
        features_name = f"{split}_time_features.npy"
        np.save(out / features_name, _time_features(times[indices]))
        partial_paths[split] = (partial, final)
        split_meta[split] = {
            "years": years_spec,
            "steps": n_time,
            "state_file": state_name,
            "time_features_file": features_name,
        }

    if train_count <= 0:
        raise RuntimeError("empty training statistics")
    mean = train_sum / train_count
    variance = np.maximum(train_sumsq / train_count - mean * mean, 0.0)
    std = np.sqrt(variance)
    if not np.isfinite(std).all() or (std <= 0).any():
        bad = [channel_names[i] for i in np.flatnonzero((std <= 0) | ~np.isfinite(std))]
        raise ValueError(f"invalid train standard deviations for {bad}")

    print("[normalize] applying train-only per-channel mean/std in place", flush=True)
    scale_mean = mean.astype(np.float32)[None, :, None, None]
    scale_std = std.astype(np.float32)[None, :, None, None]
    for split, (partial, final) in partial_paths.items():
        mmap = np.load(partial, mmap_mode="r+")
        for start in range(0, mmap.shape[0], max(block_times, 32)):
            stop = min(start + max(block_times, 32), mmap.shape[0])
            mmap[start:stop] = (mmap[start:stop] - scale_mean) / scale_std
        mmap.flush()
        del mmap
        partial.replace(final)
        print(f"  [done] {split} -> {final}", flush=True)

    np.save(out / "latitude.npy", latitude)
    np.save(out / "longitude.npy", longitude)
    metadata = {
        "format_version": 1,
        "source": args.source,
        "pressure_variables": list(args.pressure_vars),
        "surface_variables": list(args.surface_vars),
        "levels_hpa": [int(x) for x in args.levels],
        "channel_names": channel_names,
        "channels": channels,
        "height": height,
        "width": width,
        "step_hours": 6,
        "normalization": {
            "mean": [float(x) for x in mean],
            "std": [float(x) for x in std],
            "source_split": "train",
        },
        "splits": split_meta,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    total_gib = sum((out / value["state_file"]).stat().st_size for value in split_meta.values()) / 1024**3
    print(f"[out] {metadata_path}; normalized states={total_gib:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
