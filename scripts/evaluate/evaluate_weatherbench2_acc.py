#!/usr/bin/env python
"""WeatherBench-style ACC evaluation for the local WeatherBench-2 pilot.

The training arrays are already standardized channel-wise.  ACC is therefore
computed in standardized coordinates after subtracting a train-only,
calendar-day x UTC-hour pixel climatology.  This is equivalent to computing
ACC in physical units for each channel.  Latitude weights are proportional to
cos(latitude).

By default the script evaluates the three variables most commonly shown in
WeatherBench tables: Z500, T850, and T2m.  Use ``--channels all`` to evaluate
all channels, but note that the corresponding climatology cache is much
larger.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from internal_dw.models.registry import build_model
from internal_dw.utils import unwrap_model


DEFAULT_CHANNELS = ("geopotential_500", "temperature_850", "2m_temperature")
DISPLAY_NAMES = {
    "geopotential_500": "Z500",
    "temperature_850": "T850",
    "2m_temperature": "T2m",
}


def _load_state_dict_once(model, checkpoint: dict) -> None:
    if "model" in checkpoint:
        state = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint
    state = {
        (key[len("module."):] if isinstance(key, str) and key.startswith("module.") else key): value
        for key, value in state.items() if key != "_metadata"
    }
    unwrap_model(model).load_state_dict(state, strict=True)


def _split_timestamps(metadata: dict, split: str) -> np.ndarray:
    split_meta = metadata["splits"][split]
    lo, hi = (int(value) for value in str(split_meta["years"]).split("-"))
    step_hours = int(metadata.get("step_hours", 6))
    start = np.datetime64(f"{lo:04d}-01-01T00", "h")
    stop = np.datetime64(f"{hi + 1:04d}-01-01T00", "h")
    timestamps = np.arange(start, stop, np.timedelta64(step_hours, "h"))
    expected = int(split_meta["steps"])
    if timestamps.size != expected:
        raise ValueError(
            f"metadata time range for {split} gives {timestamps.size} steps, expected {expected}"
        )
    return timestamps


def _calendar_bins(timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    days = timestamps.astype("datetime64[D]")
    months = (
        timestamps.astype("datetime64[M]").astype(np.int64)
        - timestamps.astype("datetime64[Y]").astype("datetime64[M]").astype(np.int64)
    ).astype(np.int64)
    month_starts = timestamps.astype("datetime64[M]").astype("datetime64[D]")
    day = (days - month_starts).astype("timedelta64[D]").astype(np.int64)
    hour = (
        timestamps.astype("datetime64[h]") - days.astype("datetime64[h]")
    ).astype("timedelta64[h]").astype(np.int64)
    if np.any(hour % 6):
        raise ValueError("the WB2 pilot is expected on a 6-hour UTC grid")
    return months, day, hour // 6


def _resolve_channels(metadata: dict, specs: Sequence[str]) -> Tuple[List[str], List[int]]:
    names = list(metadata["channel_names"])
    requested = list(specs)
    if len(requested) == 1 and requested[0].strip().lower() == "all":
        requested = names
    missing = [name for name in requested if name not in names]
    if missing:
        raise ValueError(f"unknown channels {missing}; available channels: {names}")
    return requested, [names.index(name) for name in requested]


def _cache_token(channel_names: Sequence[str]) -> str:
    if tuple(channel_names) == DEFAULT_CHANNELS:
        return "z500_t850_t2m"
    if len(channel_names) > 8:
        return f"all{len(channel_names)}"
    return "_".join(name.replace("_component_of_wind", "wind") for name in channel_names)


def _build_or_load_climatology(
    data_root: Path,
    metadata: dict,
    channel_names: Sequence[str],
    channel_indices: Sequence[int],
    refresh: bool = False,
) -> Tuple[np.ndarray, Path]:
    """Build a train-only calendar-day x UTC-hour pixel climatology.

    The cache shape is [12,31,4,C,H,W].  Invalid calendar cells remain NaN.
    A temporary mmap is atomically renamed, so interrupted construction never
    leaves a cache that a later evaluator mistakes for complete.
    """
    cache = data_root / f"acc_climatology_{_cache_token(channel_names)}.npy"
    meta_cache = cache.with_suffix(".json")
    expected = {
        "format_version": 2,
        "channels": list(channel_names),
        "source_split": "train",
        "binning": "calendar-month x day-of-month x UTC-hour (6-hourly)",
    }
    if cache.is_file() and meta_cache.is_file() and not refresh:
        saved = json.loads(meta_cache.read_text(encoding="utf-8"))
        if all(saved.get(key) == value for key, value in expected.items()):
            return np.load(cache, mmap_mode="r"), cache

    train_meta = metadata["splits"]["train"]
    train = np.load(data_root / train_meta["state_file"], mmap_mode="r")
    timestamps = _split_timestamps(metadata, "train")
    month, day, slot = _calendar_bins(timestamps)
    height, width = int(metadata["height"]), int(metadata["width"])
    shape = (12, 31, 4, len(channel_indices), height, width)
    temporary = cache.with_name(cache.name + f".tmp.{os.getpid()}")
    temporary_meta = meta_cache.with_name(meta_cache.name + f".tmp.{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    climatology = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=shape
    )
    climatology[:] = np.nan
    started = time.time()
    bins_written = 0
    for mon in range(12):
        for dom in range(31):
            for utc_slot in range(4):
                indices = np.flatnonzero(
                    (month == mon) & (day == dom) & (slot == utc_slot)
                )
                if indices.size == 0:
                    continue
                # Only the selected channels are materialized.  Looping over
                # the roughly nine training years avoids an advanced-indexing
                # copy of all 29 channels.
                total = np.zeros(
                    (len(channel_indices), height, width), dtype=np.float64
                )
                for index in indices:
                    total += np.asarray(train[int(index), channel_indices], dtype=np.float64)
                climatology[mon, dom, utc_slot] = (total / indices.size).astype(np.float32)
                bins_written += 1
        print(
            f"[climatology] month {mon + 1:02d}/12; bins={bins_written}; "
            f"elapsed={(time.time() - started) / 60:.1f} min",
            flush=True,
        )
    climatology.flush()
    del climatology, train
    saved_meta = dict(expected)
    saved_meta.update({"shape": list(shape), "bins_written": bins_written})
    temporary_meta.write_text(json.dumps(saved_meta, indent=2), encoding="utf-8")
    os.replace(temporary, cache)
    os.replace(temporary_meta, meta_cache)
    print(f"[climatology] saved {cache}", flush=True)
    return np.load(cache, mmap_mode="r"), cache


def _metric_store(horizons: Iterable[int], channels: int) -> Dict[int, dict]:
    return {
        int(h): {
            "cross": np.zeros(channels, dtype=np.float64),
            "forecast_energy": np.zeros(channels, dtype=np.float64),
            "truth_energy": np.zeros(channels, dtype=np.float64),
            "acc_sum": np.zeros(channels, dtype=np.float64),
            "acc_sumsq": np.zeros(channels, dtype=np.float64),
            "acc_count": np.zeros(channels, dtype=np.int64),
            "mse_sum": np.zeros(channels, dtype=np.float64),
            "samples": 0,
        }
        for h in horizons
    }


def _relative_l2_store(horizons: Iterable[int]) -> Dict[int, dict]:
    """Endpoint relative-L2 over the complete standardized WB2 state.

    This matches ``internal_dw.evaluation.metrics.relative_l2``: first take the
    error/target norm ratio for each initialization, then average starts.
    Keeping this separate from the three-channel ACC store lets the paper use
    one all-channel relative-L2 metric without building an all-channel
    climatology cache.
    """
    return {
        int(h): {"sum": 0.0, "sumsq": 0.0, "count": 0}
        for h in horizons
    }


def _update_relative_l2(store: dict, prediction: np.ndarray, truth: np.ndarray,
                        eps: float = 1e-8) -> None:
    batch = int(prediction.shape[0])
    pred_flat = prediction.reshape(batch, -1).astype(np.float64, copy=False)
    truth_flat = truth.reshape(batch, -1).astype(np.float64, copy=False)
    numerator = np.linalg.norm(pred_flat - truth_flat, axis=1)
    denominator = np.maximum(np.linalg.norm(truth_flat, axis=1), eps)
    values = numerator / denominator
    store["sum"] += float(values.sum())
    store["sumsq"] += float(np.square(values).sum())
    store["count"] += batch


def _finalize_relative_l2(store: dict) -> tuple[float, float]:
    count = int(store["count"])
    mean = float(store["sum"] / max(count, 1))
    second = float(store["sumsq"] / max(count, 1))
    return mean, float(np.sqrt(max(second - mean ** 2, 0.0)))


def _update_metrics(store: dict, prediction: np.ndarray, truth: np.ndarray,
                    climate: np.ndarray, weights: np.ndarray) -> None:
    forecast_anomaly = prediction - climate
    truth_anomaly = truth - climate
    # Arrays are [batch, channel, latitude, longitude].  Reduce only the two
    # spatial axes so that every initialization contributes one score per
    # channel.
    spatial_axes = (-2, -1)
    cross = np.sum(weights * forecast_anomaly * truth_anomaly, axis=spatial_axes)
    forecast_energy = np.sum(weights * forecast_anomaly ** 2, axis=spatial_axes)
    truth_energy = np.sum(weights * truth_anomaly ** 2, axis=spatial_axes)
    denominator = np.sqrt(forecast_energy * truth_energy)
    valid = denominator > 0
    spatial_acc = np.full_like(cross, np.nan)
    spatial_acc[valid] = cross[valid] / denominator[valid]
    store["cross"] += cross.sum(axis=0)
    store["forecast_energy"] += forecast_energy.sum(axis=0)
    store["truth_energy"] += truth_energy.sum(axis=0)
    store["acc_sum"] += np.where(valid, spatial_acc, 0.0).sum(axis=0)
    store["acc_sumsq"] += np.where(valid, spatial_acc ** 2, 0.0).sum(axis=0)
    store["acc_count"] += valid.sum(axis=0)
    store["mse_sum"] += np.sum(
        weights * (prediction - truth) ** 2, axis=spatial_axes
    ).sum(axis=0)
    store["samples"] += prediction.shape[0]


def _finalize_metrics(store: dict, physical_stds: np.ndarray) -> dict:
    pooled_denominator = np.sqrt(store["forecast_energy"] * store["truth_energy"])
    pooled = np.divide(
        store["cross"], pooled_denominator,
        out=np.full_like(store["cross"], np.nan), where=pooled_denominator > 0,
    )
    counts = store["acc_count"]
    mean = np.divide(
        store["acc_sum"], counts,
        out=np.full_like(store["acc_sum"], np.nan), where=counts > 0,
    )
    second = np.divide(
        store["acc_sumsq"], counts,
        out=np.full_like(store["acc_sumsq"], np.nan), where=counts > 0,
    )
    sd = np.sqrt(np.maximum(second - mean ** 2, 0.0))
    rmse_z = np.sqrt(store["mse_sum"] / max(int(store["samples"]), 1))
    return {
        "n": int(store["samples"]),
        "acc_pooled": pooled.tolist(),
        "acc_mean_per_start": mean.tolist(),
        "acc_sd_across_starts": sd.tolist(),
        "rmse_z": rmse_z.tolist(),
        "rmse_physical": (rmse_z * physical_stds).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--data", default="data/weatherbench2_1p5_pilot")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--horizons", type=int, nargs="+", default=[4, 12, 20, 28, 40, 48])
    parser.add_argument(
        "--train-horizon", type=int, default=0,
        help="training/BPTT horizon K; 0 uses the largest requested horizon",
    )
    parser.add_argument("--channels", nargs="+", default=list(DEFAULT_CHANNELS))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--start-stride", type=int, default=1)
    parser.add_argument("--num-starts", type=int, default=0,
                        help="0 evaluates every valid test initialization")
    parser.add_argument("--evenly-spaced-starts", action="store_true",
                        help="when --num-starts is set, cover the full split instead of taking its prefix")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--refresh-climatology", action="store_true")
    parser.add_argument("--build-climatology-only", action="store_true")
    parser.add_argument("--out", default=None)
    cli = parser.parse_args()

    data_root = Path(cli.data)
    metadata = json.loads((data_root / "metadata.json").read_text(encoding="utf-8"))
    channel_names, channel_indices = _resolve_channels(metadata, cli.channels)
    climatology, climatology_path = _build_or_load_climatology(
        data_root, metadata, channel_names, channel_indices,
        refresh=cli.refresh_climatology,
    )
    if cli.build_climatology_only:
        print(f"[done] climatology ready: {climatology_path}")
        return
    if not cli.ckpt:
        parser.error("--ckpt is required unless --build-climatology-only is used")
    if not torch.cuda.is_available():
        raise RuntimeError("WeatherBench-2 ACC rollout evaluation requires CUDA")
    horizons = sorted(set(int(h) for h in cli.horizons))
    if not horizons or horizons[0] <= 0:
        raise ValueError("all horizons must be positive")
    if cli.batch_size <= 0 or cli.start_stride <= 0 or cli.num_starts < 0:
        raise ValueError("invalid batch/start arguments")

    checkpoint = torch.load(cli.ckpt, map_location="cpu", weights_only=False)
    if "args" not in checkpoint:
        raise KeyError(f"checkpoint has no saved args: {cli.ckpt}")
    args = argparse.Namespace(**checkpoint["args"])
    if str(getattr(args, "dataset", "")) != "weatherbench2":
        raise ValueError(f"checkpoint dataset is {getattr(args, 'dataset', None)!r}, not weatherbench2")
    if str(getattr(args, "model_name", "")) != "unet_field":
        raise ValueError("this evaluator currently targets the matched WB2 U-Net runs")
    args.field_channels = int(metadata["channels"])
    args.field_height = int(metadata["height"])
    args.field_width = int(metadata["width"])
    args.roi_dim = args.field_channels * args.field_height * args.field_width
    args.stim_dim = 4
    args.dataset_has_external_input = True

    torch.cuda.set_device(cli.gpu)
    device = torch.device(f"cuda:{cli.gpu}")
    # Frozen evaluation must not inherit controller variants from the shell.
    os.environ.pop("DUAL_WIENER_CONST", None)
    os.environ.pop("DUAL_WIENER_INNOVATION_FILE", None)
    os.environ.pop("DUAL_WIENER_INNOVATION_KEY", None)
    model = build_model(args, rank=cli.gpu)
    _load_state_dict_once(model, checkpoint)
    raw = unwrap_model(model).eval()
    saved_epoch = int(checkpoint.get("epoch", -1))
    del checkpoint

    split_meta = metadata["splits"][cli.split]
    state = np.load(data_root / split_meta["state_file"], mmap_mode="r")
    stim = np.load(data_root / split_meta["time_features_file"], mmap_mode="r")
    timestamps = _split_timestamps(metadata, cli.split)
    months, days, slots = _calendar_bins(timestamps)
    latitude = np.load(data_root / "latitude.npy")
    weights = np.cos(np.deg2rad(latitude.astype(np.float64)))[:, None]
    weights = np.broadcast_to(weights, (metadata["height"], metadata["width"])).copy()
    weights /= weights.sum()

    window = int(getattr(args, "window_size", 0))
    max_horizon = max(horizons)
    train_horizon = int(cli.train_horizon or max_horizon)
    if train_horizon <= 0 or train_horizon > max_horizon:
        raise ValueError("train-horizon must satisfy 0 < train-horizon <= max(horizons)")
    starts = np.arange(window, state.shape[0] - max_horizon + 1, cli.start_stride, dtype=np.int64)
    if cli.num_starts:
        if cli.evenly_spaced_starts and starts.size > cli.num_starts:
            positions = np.rint(
                np.linspace(0, starts.size - 1, int(cli.num_starts))
            ).astype(np.int64)
            starts = starts[np.unique(positions)]
        else:
            starts = starts[:cli.num_starts]
    if starts.size == 0:
        raise RuntimeError("no valid forecast initializations")
    stores = _metric_store(horizons, len(channel_indices))
    relative_l2_stores = _relative_l2_store(horizons)
    selected_relative_l2_stores = _relative_l2_store(horizons)
    horizon_set = set(horizons)
    physical_stds = np.asarray(metadata["normalization"]["std"], dtype=np.float64)[channel_indices]
    step_hours = int(metadata.get("step_hours", 6))
    started_at = time.time()

    print(
        f"[WB2 ACC] checkpoint={cli.ckpt}; epoch={saved_epoch}; split={cli.split}; "
        f"starts={starts.size}; batch={cli.batch_size}; horizons={horizons}; "
        f"channels={channel_names}", flush=True,
    )
    with torch.inference_mode():
        for batch_low in range(0, starts.size, cli.batch_size):
            batch_starts = starts[batch_low:batch_low + cli.batch_size]
            histories = np.stack(
                [np.asarray(state[start - window:start]) for start in batch_starts], axis=0
            )
            history = torch.from_numpy(histories).to(device=device, dtype=torch.float32)
            for offset in range(max_horizon):
                stim_windows = np.stack(
                    [np.asarray(stim[start + offset - window:start + offset]) for start in batch_starts],
                    axis=0,
                )
                stim_window = torch.from_numpy(stim_windows).to(device=device, dtype=torch.float32)
                prediction = raw(
                    stim_window, history, return_aux=False,
                    horizon_index=offset, total_horizon=max_horizon,
                )
                prediction = prediction[:, 0] if prediction.dim() == 5 else prediction
                history = torch.cat([history[:, 1:], prediction.unsqueeze(1)], dim=1)
                horizon = offset + 1
                if horizon not in horizon_set:
                    continue
                prediction_all = prediction.float().cpu().numpy().astype(np.float64)
                prediction_np = prediction_all[:, channel_indices]
                target_indices = batch_starts + offset
                truth_all = np.stack(
                    [np.asarray(state[index]) for index in target_indices], axis=0
                ).astype(np.float64)
                truth_np = truth_all[:, channel_indices]
                climate_np = np.stack(
                    [
                        np.asarray(climatology[months[index], days[index], slots[index]])
                        for index in target_indices
                    ], axis=0,
                ).astype(np.float64)
                if not np.isfinite(climate_np).all():
                    raise ValueError("a requested test calendar bin is absent from train climatology")
                _update_metrics(stores[horizon], prediction_np, truth_np, climate_np, weights)
                _update_relative_l2(
                    relative_l2_stores[horizon], prediction_all, truth_all
                )
                _update_relative_l2(
                    selected_relative_l2_stores[horizon], prediction_np, truth_np
                )

            completed = min(batch_low + len(batch_starts), starts.size)
            if cli.progress_every > 0 and (
                completed == starts.size or completed // cli.progress_every != batch_low // cli.progress_every
            ):
                elapsed = time.time() - started_at
                print(
                    f"[WB2 ACC] {completed}/{starts.size} starts; "
                    f"elapsed={elapsed / 60:.1f} min; {elapsed / completed:.3f} s/start",
                    flush=True,
                )

    result = {
        "format_version": 1,
        "checkpoint": str(cli.ckpt),
        "checkpoint_epoch": saved_epoch,
        "data": str(data_root),
        "split": cli.split,
        "channels": channel_names,
        "display_names": [DISPLAY_NAMES.get(name, name) for name in channel_names],
        "horizons": horizons,
        "step_hours": step_hours,
        "num_starts": int(starts.size),
        "start_stride": int(cli.start_stride),
        "start_selection": (
            "evenly_spaced_over_split"
            if cli.num_starts and cli.evenly_spaced_starts
            else "all_eligible" if not cli.num_starts else "prefix"
        ),
        "climatology_cache": str(climatology_path),
        "protocol": {
            "climatology": "train-only calendar-day x UTC-hour pixel mean",
            "area_weights": "cos(latitude), normalized over the global grid",
            "primary_acc": "pooled anomaly cross-product over starts and space",
            "secondary_acc": "spatial ACC per start, then mean over starts",
        },
        "per_horizon": {},
    }
    labels = result["display_names"]
    print("\n=== WeatherBench-2 anomaly correlation ===")
    for horizon in horizons:
        row = _finalize_metrics(stores[horizon], physical_stds)
        rel_mean, rel_sd = _finalize_relative_l2(relative_l2_stores[horizon])
        selected_rel_mean, selected_rel_sd = _finalize_relative_l2(
            selected_relative_l2_stores[horizon]
        )
        row["relative_l2_all_channels_mean_per_start"] = float(rel_mean)
        row["relative_l2_all_channels_sd_across_starts"] = float(rel_sd)
        row["relative_l2_selected_channels_mean_per_start"] = float(selected_rel_mean)
        row["relative_l2_selected_channels_sd_across_starts"] = float(selected_rel_sd)
        row["days"] = horizon * step_hours / 24.0
        result["per_horizon"][str(horizon)] = row
        print(f"H={horizon:2d} ({row['days']:g} d), n={row['n']}")
        for index, label in enumerate(labels):
            print(
                f"  {label:<5} pooled_ACC={row['acc_pooled'][index]:.5f}  "
                f"mean-start_ACC={row['acc_mean_per_start'][index]:.5f}  "
                f"RMSEz={row['rmse_z'][index]:.5f}"
            )
        print(
            "  all   mean-start relative-L2="
            f"{row['relative_l2_all_channels_mean_per_start']:.5f}"
        )
        print(
            "  selected mean-start relative-L2="
            f"{row['relative_l2_selected_channels_mean_per_start']:.5f}"
        )

    relative_l2_curve = np.asarray([
        result["per_horizon"][str(h)][
            "relative_l2_all_channels_mean_per_start"
        ]
        for h in horizons
    ], dtype=np.float64)
    selected_relative_l2_curve = np.asarray([
        result["per_horizon"][str(h)][
            "relative_l2_selected_channels_mean_per_start"
        ]
        for h in horizons
    ], dtype=np.float64)
    # When the caller requests every lead 1..K, these are the same headline
    # summaries used by the unified vector/field multi-origin evaluator.  Keep
    # the legacy key for old callers, but make the dense-curve semantics
    # explicit and report the long half separately.
    dense_prefix = horizons == list(range(1, max_horizon + 1))
    long_start = int(np.ceil(max_horizon / 2.0))
    long_mask = np.asarray(horizons, dtype=np.int64) >= long_start
    in_mask = np.asarray(horizons, dtype=np.int64) <= train_horizon
    out_mask = np.asarray(horizons, dtype=np.int64) > train_horizon
    result["summary"] = {
        "relative_l2_all_channels_horizon_mean": float(relative_l2_curve.mean()),
        "relative_l2_all_horizons_mean": (
            float(relative_l2_curve.mean()) if dense_prefix else None
        ),
        "relative_l2_in_horizon_mean": (
            float(relative_l2_curve[in_mask].mean()) if dense_prefix else None
        ),
        "relative_l2_out_of_horizon_mean": (
            float(relative_l2_curve[out_mask].mean())
            if dense_prefix and bool(out_mask.any()) else None
        ),
        "relative_l2_long_half_mean": (
            float(relative_l2_curve[long_mask].mean()) if dense_prefix else None
        ),
        "relative_l2_final_horizon": float(relative_l2_curve[-1]),
        "relative_l2_selected_channels_all_horizons_mean": (
            float(selected_relative_l2_curve.mean()) if dense_prefix else None
        ),
        "relative_l2_selected_channels_in_horizon_mean": (
            float(selected_relative_l2_curve[in_mask].mean()) if dense_prefix else None
        ),
        "relative_l2_selected_channels_out_of_horizon_mean": (
            float(selected_relative_l2_curve[out_mask].mean())
            if dense_prefix and bool(out_mask.any()) else None
        ),
        "relative_l2_selected_channels_final_horizon": float(
            selected_relative_l2_curve[-1]
        ),
        "relative_l2_long_half_start": long_start,
        "relative_l2_train_horizon": train_horizon,
        "relative_l2_eval_horizon": max_horizon,
        "relative_l2_horizons": horizons,
        "dense_horizons_1_to_K": bool(dense_prefix),
        "aggregation": "mean over rolling forecast origins at each lead, then mean over leads",
    }

    out = Path(cli.out) if cli.out else Path(cli.ckpt).parent / "weatherbench2_acc.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
