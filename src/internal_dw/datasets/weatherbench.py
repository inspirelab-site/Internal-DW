"""WeatherBench-1 5.625deg (32x64) as a DRIVEN-CHAOTIC autoregressive testbed (regime 3).

The atmosphere is chaotic (leading Lyapunov exponent > 0, ~1-2 week predictability) yet
DRIVEN by a per-step external signal (diurnal + seasonal forcing), so long-horizon
prediction keeps reducible signal. This is exactly the regime our Predictability-Horizon
Principle flags as the boundary of routing (driven chaos: lambda_1>0 AND long-horizon
reducible signal). We flatten the [C,32,64] field to a state vector so it runs through the
same official_mamba_state AR pipeline + ResGrad routing as HCP / Lorenz-96, and (by
default) hand the model the forcing phase as an explicit external input, paralleling HCP's
stimulus.

Data: per-variable yearly NetCDF from scripts/download_weatherbench.sh. We stack the
selected channels (default Z500, T850, T2m), subsample to 6-hourly, split BY YEAR
(train 1979-2015 / val 2016 / test 2017-2018), z-score per channel with train-only stats,
and chunk each split's continuous series into fixed-length segments (the "trajectories").

Interface matches the other datasets: __getitem__ returns
    {"state": [T, D], "external_input": [T, S] or None, "label": int, "metadata": {...}}
and build_weatherbench_splits(args) returns (train, val, test) and sets args.roi_dim = D.
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from .base import SequenceDataset

# Default headline channels. geopotential_500 (Z500) and temperature_850 (T850) are the
# WeatherBench verification variables; 2m_temperature (T2m) carries the strong diurnal
# cycle that makes the system *driven*.
_DEFAULT_VARS = "geopotential_500 temperature_850 2m_temperature"


def _pick_field_var(ds):
    """Return the name of the data variable that has time+lat+lon dims (robust to short
    codes like 'z','t','t2m')."""
    dims = set(ds.dims)
    lat = next((d for d in dims if "lat" in d.lower()), None)
    lon = next((d for d in dims if "lon" in d.lower()), None)
    tim = next((d for d in dims if "time" in d.lower()), None)
    if lat is None or lon is None or tim is None:
        raise ValueError(f"could not find time/lat/lon dims in {dims}")
    for name, da in ds.data_vars.items():
        if lat in da.dims and lon in da.dims and tim in da.dims:
            return name, tim, lat, lon
    raise ValueError(f"no field variable with dims ({tim},{lat},{lon}) in {list(ds.data_vars)}")


def _load_variable(folder: Path, step_hours: int):
    """Open all yearly NetCDFs for one variable (eagerly, per file -- no dask needed),
    concatenate along time, subsample to step_hours. Returns
    (array [T, lat, lon] float32, time index as np.datetime64 [T])."""
    import xarray as xr
    files = sorted(folder.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"no .nc files in {folder} (run scripts/download_weatherbench.sh)")
    arrs, times = [], []
    for f in files:
        ds = xr.open_dataset(f)                      # eager numpy load, no dask
        try:
            name, tim, lat, lon = _pick_field_var(ds)
            da = ds[name]
            for d in list(da.dims):                 # drop singleton pressure-level dim
                if d not in (tim, lat, lon) and da.sizes[d] == 1:
                    da = da.squeeze(d, drop=True)
            da = da.transpose(tim, lat, lon)
            arrs.append(np.asarray(da.values, dtype=np.float32))
            times.append(np.asarray(da[tim].values))
        finally:
            ds.close()
    arr = np.concatenate(arrs, axis=0)
    tim_all = np.concatenate(times, axis=0)
    order = np.argsort(tim_all, kind="stable")      # ensure global time order
    arr, tim_all = arr[order], tim_all[order]
    step = max(1, int(step_hours))                  # hourly -> step_hours (year lengths divisible by 6)
    return arr[::step], tim_all[::step]


def _time_features(times) -> np.ndarray:
    """[T, 4] diurnal+seasonal forcing phase (sin/cos of hour-of-day and day-of-year)."""
    t = times.astype("datetime64[h]")
    hour = (t - t.astype("datetime64[D]")).astype("timedelta64[h]").astype(np.float32)  # 0..23
    doy = ((t.astype("datetime64[D]") - t.astype("datetime64[Y]")).astype("timedelta64[D]")
           .astype(np.float32))  # 0..~365
    two_pi = 2.0 * np.pi
    return np.stack([
        np.sin(two_pi * hour / 24.0), np.cos(two_pi * hour / 24.0),
        np.sin(two_pi * doy / 365.25), np.cos(two_pi * doy / 365.25),
    ], axis=1).astype(np.float32)


class WeatherBenchDataset(SequenceDataset):
    dataset_name = "weatherbench"
    has_external_input = True          # diurnal+seasonal forcing phase (set 0 to disable)
    task_type = "sequence_vector"
    evaluator_name = "generic"

    def __init__(self, states, stims, split="train", field_shape=None):
        self.states = states           # [n, L, D] float32, normalized
        self.stims = stims             # [n, L, S] float32 or None
        self.split = str(split)
        # When set to (C, H, W) the flat state is served as a 2-D field, so the
        # grid baselines (UNet/CNN/FNO) can be trained on the same cached data
        # as the flattened sequence models.  main.py infers field_channels /
        # field_height / field_width from this shape.
        self.field_shape = tuple(field_shape) if field_shape else None

    def __len__(self):
        return int(self.states.shape[0])

    def __getitem__(self, index):
        stim = None
        if self.stims is not None:
            stim = torch.from_numpy(np.ascontiguousarray(self.stims[index]))
        state = torch.from_numpy(np.ascontiguousarray(self.states[index]))
        if self.field_shape is not None:
            state = state.reshape(state.shape[0], *self.field_shape)      # [L, C, H, W]
        return {
            "state": state,
            "external_input": stim,
            "label": int(index),
            "metadata": {"dataset": self.dataset_name, "split": self.split, "index": int(index)},
        }


def _chunk(arr, L):
    """[T, ...] -> [n, L, ...] non-overlapping segments (drop remainder)."""
    n = arr.shape[0] // L
    if n == 0:
        raise ValueError(f"split has {arr.shape[0]} steps < segment length {L}; lower wb_seg_len")
    return arr[: n * L].reshape(n, L, *arr.shape[1:])


def _sqrt_area_weight(H: int, W: int) -> "np.ndarray":
    """sqrt(cos(lat)) per grid cell, normalised to unit mean, shape [1, 1, H, W].

    Folding this into the stored state makes every downstream quadratic --- the
    rollout loss, the Dual-Wiener total/noise probes, the calibrated innovation
    covariance, and T/R --- latitude-weighted at once, which is the WeatherBench
    convention.  Weighting only the loss would leave the probes measuring route
    moments for a different geometry than the one being trained.  Unit mean keeps
    loss magnitudes comparable to the unweighted cache.
    """
    lat = 90.0 - (180.0 / H) * (np.arange(H) + 0.5)
    w = np.clip(np.cos(np.deg2rad(lat)), 0.0, None)
    w = w / w.mean()
    return np.sqrt(w).astype(np.float32)[None, None, :, None] * np.ones((1, 1, 1, W), dtype=np.float32)


def build_weatherbench_splits(args) -> Tuple[WeatherBenchDataset, WeatherBenchDataset, WeatherBenchDataset]:
    root = Path(args.data_path)
    var_names = str(getattr(args, "wb_vars", _DEFAULT_VARS)).split()
    step = int(getattr(args, "wb_step_hours", 6))
    L = int(getattr(args, "wb_seg_len", 256))
    use_stim = int(getattr(args, "wb_time_features", 1))
    yr_tr = str(getattr(args, "wb_train_years", "1979-2015"))
    yr_va = str(getattr(args, "wb_val_years", "2016-2016"))
    yr_te = str(getattr(args, "wb_test_years", "2017-2018"))

    def _yr_range(s):
        a, b = s.split("-"); return int(a), int(b)
    (tr0, tr1), (va0, va1), (te0, te1) = _yr_range(yr_tr), _yr_range(yr_va), _yr_range(yr_te)

    aw = int(getattr(args, "wb_area_weight", 0))
    cache = root / (f"wb_cache_{'_'.join(var_names)}_step{step}_L{L}_stim{use_stim}"
                    f"_{yr_tr}_{yr_va}_{yr_te}" + ("_aw" if aw else "") + ".npz")
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        C, H, W = int(z["C"]), int(z["H"]), int(z["W"])
        splits = {k: z[k] for k in ("s_tr", "s_va", "s_te")}
        stims = None if int(z["use_stim"]) == 0 else {k: z[k] for k in ("x_tr", "x_va", "x_te")}
        print(f"[WB] loaded cache {cache.name}: state_dim={C*H*W} (C={C},{H}x{W})")
    else:
        print(f"[WB] reading {var_names} @ {step}h from {root} ...")
        chans, times0 = [], None
        for v in var_names:
            arr, times = _load_variable(root / v, step)     # [T,H,W]
            if times0 is None:
                times0 = times
            elif len(times) != len(times0):
                raise ValueError(f"{v} has {len(times)} steps != {len(times0)}; variables misaligned")
            chans.append(arr)
        data = np.stack(chans, axis=1)                      # [T, C, H, W]
        T, C, H, W = data.shape
        years = times0.astype("datetime64[Y]").astype(int) + 1970

        def _sel(y0, y1):
            m = (years >= y0) & (years <= y1)
            return data[m], (_time_features(times0[m]) if use_stim else None)

        d_tr, x_tr = _sel(tr0, tr1); d_va, x_va = _sel(va0, va1); d_te, x_te = _sel(te0, te1)
        if d_tr.shape[0] == 0 or d_va.shape[0] == 0 or d_te.shape[0] == 0:
            raise ValueError(f"empty split; years present {years.min()}-{years.max()}")

        # per-channel z-score with TRAIN stats only (channels have wildly different scales)
        mu = d_tr.mean(axis=(0, 2, 3), keepdims=True)
        sd = d_tr.std(axis=(0, 2, 3), keepdims=True); sd[sd == 0] = 1.0
        gw = _sqrt_area_weight(H, W) if aw else None
        if gw is not None:
            print(f"[WB] latitude area weighting ON: sqrt(cos lat), unit mean, "
                  f"range {gw.min():.3f}-{gw.max():.3f}")
        def norm(d):
            z = (d - mu) / sd
            if gw is not None:
                z = z * gw
            return z.astype(np.float32).reshape(d.shape[0], C * H * W)

        splits = {"s_tr": _chunk(norm(d_tr), L), "s_va": _chunk(norm(d_va), L), "s_te": _chunk(norm(d_te), L)}
        stims = None
        if use_stim:
            stims = {"x_tr": _chunk(x_tr, L), "x_va": _chunk(x_va, L), "x_te": _chunk(x_te, L)}
        save = dict(C=C, H=H, W=W, use_stim=use_stim, **splits)
        if stims is not None:
            save.update(stims)
        np.savez_compressed(cache, **save)
        print(f"[WB] cached -> {cache.name}: state_dim={C*H*W} (C={C},{H}x{W})")

    D = C * H * W
    args.roi_dim = D
    if use_stim:
        args.stim_dim = int(next(iter(stims.values())).shape[-1])

    # Field mode: serve the same cached data as [L, C, H, W] so the grid
    # baselines the weather literature uses (UNet/CNN/FNO) can be trained and
    # scored on it.  The evaluator and loss path switch with the task type.
    as_field = bool(int(getattr(args, "wb_field", 0)))
    field_shape = (C, H, W) if as_field else None
    # The grid shape is recorded either way.  In flat mode the loader still serves
    # [D] and the evaluator stays generic, but a model that wants to reshape the
    # state internally (official_mamba_field) needs to know what grid it came from.
    args.field_channels, args.field_height, args.field_width = C, H, W
    if as_field:
        WeatherBenchDataset.task_type = "field2d"
        WeatherBenchDataset.evaluator_name = "field2d"
    else:
        WeatherBenchDataset.task_type = "sequence_vector"
        WeatherBenchDataset.evaluator_name = "generic"

    ntr, nva, nte = splits["s_tr"].shape[0], splits["s_va"].shape[0], splits["s_te"].shape[0]
    print(f"[WB] state_dim={D}  external_input={'on' if use_stim else 'off'}  "
          f"layout={'field [C,H,W]' if as_field else 'flat [D]'}  "
          f"train/val/test = {ntr}/{nva}/{nte} segments of length {L} ({step}h step)")

    def mk(sk, xk, sp):
        return WeatherBenchDataset(splits[sk], (stims[xk] if stims is not None else None), sp,
                                   field_shape=field_shape)

    return (mk("s_tr", "x_tr", "train"), mk("s_va", "x_va", "val"), mk("s_te", "x_te", "test"))
