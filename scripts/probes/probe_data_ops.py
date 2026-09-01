"""Dataset, draw-planning, and nested-state helpers for paper probes."""

from __future__ import annotations

import argparse
import collections
import glob as _glob
import random

import numpy as np
import torch

from internal_dw.data_utils.state_ops import unpack_batch, zero_external_input_like


def _metadata_item(metadata, key: str, index: int):
    if not isinstance(metadata, dict) or key not in metadata:
        return None
    value = metadata[key]
    if torch.is_tensor(value):
        value = value[index]
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def sample_field_candidates(
    loader, args, horizon: int, count: int, seed: int, *, unique_groups=False
):
    """Take at most one auditable forecast start from each loaded segment."""
    candidates = []
    window = int(getattr(args, "window_size", 1))
    generator = random.Random(int(seed))
    seen_groups = set()
    for batch_index, batch in enumerate(loader):
        state, stimulus, _, metadata = unpack_batch(batch)
        for sample_index in range(int(state.shape[0])):
            trajectory = state[sample_index : sample_index + 1].cpu()
            drive = (
                stimulus[sample_index : sample_index + 1].cpu()
                if stimulus is not None
                else None
            )
            low = window
            high = int(trajectory.shape[1]) - int(horizon)
            if high < low:
                continue
            path = _metadata_item(metadata, "path", sample_index)
            trajectory_index = _metadata_item(
                metadata, "trajectory_index", sample_index
            )
            group_id = (
                f"{path}::trajectory={trajectory_index}"
                if path is not None and trajectory_index is not None
                else f"batch={batch_index}::sample={sample_index}"
            )
            if unique_groups and group_id in seen_groups:
                continue
            seen_groups.add(group_id)
            candidates.append(
                {
                    "state": trajectory,
                    "stim": drive,
                    "start": generator.randint(low, high),
                    "group_id": group_id,
                }
            )
            if len(candidates) >= int(count):
                return candidates
    return candidates


def sample_unique_field_trajectories(
    dataset, args, horizon: int, count: int, seed: int
):
    """Sample one window per physical The-Well trajectory, balanced by file."""
    if not hasattr(dataset, "index"):
        raise TypeError("grouped The-Well sampling requires dataset.index")
    grouped = {}
    for dataset_index, entry in enumerate(dataset.index):
        path, trajectory_index, _, _ = entry
        grouped.setdefault((str(path), int(trajectory_index)), []).append(
            dataset_index
        )
    generator = random.Random(int(seed))
    by_path = collections.defaultdict(list)
    for group in grouped:
        by_path[group[0]].append(group)
    paths = sorted(by_path)
    if not paths:
        return []
    base, remainder = divmod(int(count), len(paths))
    shuffled_paths = paths.copy()
    generator.shuffle(shuffled_paths)
    allocations = {path: min(base, len(by_path[path])) for path in paths}
    for path in shuffled_paths[:remainder]:
        allocations[path] = min(allocations[path] + 1, len(by_path[path]))
    while sum(allocations.values()) < int(count):
        available = [
            path for path in paths if allocations[path] < len(by_path[path])
        ]
        if not available:
            break
        minimum = min(allocations[path] for path in available)
        choices = [path for path in available if allocations[path] == minimum]
        allocations[generator.choice(choices)] += 1

    selected = []
    for path in paths:
        choices = by_path[path].copy()
        generator.shuffle(choices)
        selected.extend(choices[: allocations[path]])
    generator.shuffle(selected)
    window = int(getattr(args, "window_size", 1))
    candidates = []
    for path, trajectory_index in selected:
        dataset_index = generator.choice(grouped[(path, trajectory_index)])
        sample = dataset[dataset_index]
        state = sample["state"].unsqueeze(0).cpu()
        stimulus = sample.get("external_input")
        stimulus = stimulus.unsqueeze(0).cpu() if stimulus is not None else None
        low, high = window, int(state.shape[1]) - int(horizon)
        if high < low:
            continue
        candidates.append(
            {
                "state": state,
                "stim": stimulus,
                "start": generator.randint(low, high),
                "group_id": f"{path}::trajectory={trajectory_index}",
            }
        )
        if len(candidates) >= int(count):
            break
    return candidates


def rollout_field(raw, item, args, horizon: int, device, *, graph: bool):
    """Closed-loop rollout used by field covariance and route probes."""
    state = item["state"].to(device=device, dtype=torch.float32)
    stimulus = item["stim"]
    if stimulus is None:
        stimulus = zero_external_input_like(
            state, int(getattr(args, "stim_dim", 1))
        )
    else:
        stimulus = stimulus.to(device=device, dtype=torch.float32)
    start = int(item["start"])
    window = int(getattr(args, "window_size", 1))
    history = state[:, start - window : start]
    predictions, targets = [], []
    context = torch.enable_grad() if graph else torch.no_grad()
    with context:
        for step in range(int(horizon)):
            target_t = start + step
            stim_window = stimulus[:, target_t - window : target_t]
            prediction = raw(
                stim_window,
                history,
                return_aux=False,
                horizon_index=step,
                total_horizon=horizon,
            )
            predictions.append(prediction)
            targets.append(state[:, target_t : target_t + 1])
            history = torch.cat([history[:, 1:], prediction], dim=1)
    return predictions, targets


def _rfft_frequency_weights(width: int, reference: torch.Tensor) -> torch.Tensor:
    frequency_width = width // 2 + 1
    weights = reference.new_full((frequency_width,), 2.0)
    weights[0] = 1.0
    if width % 2 == 0:
        weights[-1] = 1.0
    return weights


def fit_spectral_covariance(
    residuals: torch.Tensor, variance_floor: float = 1e-8
):
    """Fit per-horizon/channel diagonal scale and normalized 2-D spectrum."""
    if residuals.ndim != 5 or residuals.shape[0] < 2:
        raise ValueError("spectral covariance expects [N,K,C,H,W], N>=2")
    residuals = residuals.float()
    centered = residuals - residuals.mean(dim=0, keepdim=True)
    variance = centered.square().mean(dim=0)
    floor = variance.mean().clamp_min(1e-30) * float(variance_floor)
    sigma = torch.sqrt(variance + floor)
    whitened = centered / sigma.unsqueeze(0)
    transform = torch.fft.rfft2(whitened, dim=(-2, -1), norm="ortho")
    power = transform.abs().square().mean(dim=0)
    width = int(residuals.shape[-1])
    weights = _rfft_frequency_weights(width, power)
    spectral_variance = (
        power * weights.view(1, 1, 1, -1)
    ).sum(dim=(-2, -1), keepdim=True) / float(power.shape[-2] * width)
    power = (power / spectral_variance.clamp_min(1e-20)).clamp_min(0.0)
    return {
        "sigma": sigma.contiguous(),
        "filter": power.sqrt().contiguous(),
    }


def sample_spectral_noise(model, epsilon: torch.Tensor) -> torch.Tensor:
    white_spectrum = torch.fft.rfft2(epsilon, dim=(-2, -1), norm="ortho")
    standardized = torch.fft.irfft2(
        white_spectrum * model["filter"],
        s=epsilon.shape[-2:],
        dim=(-2, -1),
        norm="ortho",
    )
    return standardized * model["sigma"]

def _as_btd(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[None]                      # [T,D] -> [1,T,D]
    if x.ndim != 3:
        raise SystemExit(f"{name}: expected [T,D] or [B,T,D], got {x.shape}")
    return x


def _training_view(state: np.ndarray, stim: np.ndarray | None, a):
    """Match the training split and normalization used by memory testbeds.

    The original reach probes intentionally consumed a user-supplied array as
    is.  Estimator calibration is different: its covariance must be measured
    in the checkpoint's actual input coordinates and from training data only.
    """
    mode = str(getattr(a, "data_preprocess", "none") or "none").lower()
    if mode == "none":
        return state, stim

    train_ratio = float(getattr(a, "split_train_ratio", 0.7))
    val_ratio = float(getattr(a, "split_val_ratio", 0.15))
    seed = int(getattr(a, "seed", 0))

    base_mode = mode.removesuffix("_val").removesuffix("_test")
    requested_split = (
        "val" if mode.endswith("_val") else
        "test" if mode.endswith("_test") else "train"
    )

    if base_mode in ("mackey_glass", "narma"):
        n = int(state.shape[0])
        order = np.random.default_rng(seed).permutation(n)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_train + n_val >= n:
            n_train, n_val = max(1, n - 2), 1
        split_indices = {
            "train": order[:n_train],
            "val": order[n_train:n_train + n_val],
            "test": order[n_train + n_val:],
        }
        if split_indices["test"].size == 0:
            split_indices["test"] = split_indices["val"]
        train_state = state[split_indices["train"]]
        mean = float(train_state.mean())
        std = float(train_state.std()) or 1.0
        idx = split_indices[requested_split]
        state = ((state[idx] - mean) / std).astype(np.float32)
        stim = stim[idx].astype(np.float32) if stim is not None else None
        print(
            f"[data] {base_mode} seed={seed} split={requested_split} "
            f"trajectories={len(idx)}/{n}; state normalized with training "
            f"mean={mean:.6g} std={std:.6g}",
            flush=True,
        )
        return state, stim

    if base_mode == "ieeg":
        if state.shape[0] != 1:
            raise SystemExit(
                f"ieeg preprocessing expects one cached continuous series, got {state.shape}"
            )
        x = state[0]
        T = int(x.shape[0])
        n_train = int(round(T * train_ratio))
        n_val = int(round(T * val_ratio))
        if n_train + n_val >= T:
            n_train, n_val = int(0.7 * T), int(0.15 * T)
        chunk = int(getattr(a, "ieeg_chunk", 1024))
        gap = int(getattr(a, "ieeg_split_gap", 256))
        train = x[:max(chunk, n_train - gap)]
        validation = x[n_train:n_train + max(chunk, n_val - gap)]
        test = x[n_train + n_val:]
        mean = train.mean(axis=0, keepdims=True)
        std = train.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        selected = {"train": train, "val": validation, "test": test}[requested_split]
        selected = ((selected - mean) / std).astype(np.float32)
        n_chunks = int(selected.shape[0] // chunk)
        if n_chunks < 1:
            raise SystemExit(
                f"iEEG {requested_split} segment {selected.shape[0]} is shorter "
                f"than chunk={chunk}"
            )
        state = np.ascontiguousarray(
            selected[:n_chunks * chunk].reshape(
                n_chunks, chunk, selected.shape[1]
            )
        )
        print(
            f"[data] ieeg split={requested_split}, training normalization: "
            f"T={T} -> {n_chunks}x{chunk}x{selected.shape[1]}",
            flush=True,
        )
        return state, None

    if base_mode == "prepared_temporal":
        if requested_split != "train":
            raise SystemExit(
                "prepared_temporal preprocessing accepts training arrays only; "
                "pass train_state/train_drive so normalization matches the checkpoint"
            )
        state_mean = state.mean(
            axis=(0, 1), keepdims=True, dtype=np.float64
        ).astype(np.float32)
        state_scale = state.std(
            axis=(0, 1), keepdims=True, dtype=np.float64
        ).astype(np.float32)
        state_scale = np.maximum(state_scale, np.float32(1e-5))
        state = ((state - state_mean) / state_scale).astype(np.float32)
        if stim is not None:
            stim_mean = stim.mean(
                axis=(0, 1), keepdims=True, dtype=np.float64
            ).astype(np.float32)
            stim_scale = stim.std(
                axis=(0, 1), keepdims=True, dtype=np.float64
            ).astype(np.float32)
            stim_scale = np.maximum(stim_scale, np.float32(1e-5))
            stim = ((stim - stim_mean) / stim_scale).astype(np.float32)
        print(
            "[data] prepared_temporal training coordinates: per-channel "
            f"standardization over {state.shape[0]}x{state.shape[1]}",
            flush=True,
        )
        return state, stim

    raise SystemExit(f"unknown --data-preprocess {mode!r}")


def load_arrays(a) -> tuple[np.ndarray, np.ndarray | None]:
    """Return state [B,T,D] and optional stim [B,T,S]."""
    if a.hcp_dir:
        # One HCP .npy holds BOTH the parcellated fMRI and the movie features,
        # so reuse the repo's own reader instead of guessing the layout.
        from internal_dw.datasets.hcp import discover_hcp_files, _load_fmri_and_stim
        files = discover_hcp_files(a.hcp_dir, a.movie)
        # Estimator-calibration probes must only use the same training subjects
        # that were available to the checkpoint.  Keep ``all`` as the default
        # for the older read-only reach probes, which predate this option.
        hcp_split = str(getattr(a, "hcp_split", "all") or "all").lower()
        if hcp_split not in ("all", "train", "val", "test"):
            raise SystemExit(f"unknown --hcp-split {hcp_split!r}")
        if hcp_split != "all":
            rng = random.Random(int(getattr(a, "seed", 0)))
            rng.shuffle(files)
            n = len(files)
            train_ratio = float(getattr(a, "hcp_train_ratio", 0.7))
            val_ratio = float(getattr(a, "hcp_val_ratio", 0.15))
            n_train = max(1, int(round(n * train_ratio)))
            n_val = max(1, int(round(n * val_ratio)))
            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1
            split_files = {
                "train": files[:n_train],
                "val": files[n_train:n_train + n_val],
                "test": files[n_train + n_val:] or files[n_train:n_train + n_val],
            }
            files = split_files[hcp_split]
            print(
                f"[data] HCP seed={int(getattr(a, 'seed', 0))} "
                f"split={hcp_split} subjects={len(files)}",
                flush=True,
            )
        if a.batch > 0:
            files = files[: a.batch]
        pairs = [_load_fmri_and_stim(str(f), a.roi_dim) for f in files]
        T = min(min(p[0].shape[0], p[1].shape[0]) for p in pairs)
        state = np.stack([p[0][:T] for p in pairs]).astype(np.float32)
        stim = np.stack([p[1][:T] for p in pairs]).astype(np.float32)
        print(f"[data] HCP {len(files)} subjects -> state {state.shape} stim {stim.shape}",
              flush=True)
        return state, stim

    if a.glob:
        files = sorted(_glob.glob(a.glob))
        if not files:
            raise SystemExit(f"--glob matched nothing: {a.glob}")
        files = files[: a.batch] if a.batch > 0 else files
        mats = [np.load(f) for f in files]
        T = min(m.shape[0] for m in mats)
        state = np.stack([m[:T] for m in mats]).astype(np.float32)
        print(f"[data] {len(files)} files -> state {state.shape}", flush=True)
        stim = None
        if a.stim_glob:
            sf = sorted(_glob.glob(a.stim_glob))[: len(files)]
            if len(sf) != len(files):
                raise SystemExit("--stim-glob count does not match --glob count")
            sm = [np.load(f) for f in sf]
            stim = np.stack([m[:T] for m in sm]).astype(np.float32)
            print(f"[data] stim {stim.shape}", flush=True)
        return state, stim

    if not a.npz:
        raise SystemExit("pass --npz, --glob or --hcp-dir")
    if a.npz.endswith(".npy"):
        state, stim = _as_btd(np.load(a.npz), "state"), None
        return _training_view(state, stim, a)
    z = np.load(a.npz, allow_pickle=True)
    keys = list(z.files)
    if a.state_key:
        if a.state_key not in keys:
            raise SystemExit(f"--state-key {a.state_key!r} not in {keys}")
        state = _as_btd(z[a.state_key], "state")
    else:
        cands = [(k, z[k]) for k in keys
                 if getattr(z[k], "ndim", 0) in (2, 3) and z[k].dtype.kind == "f"]
        if not cands:
            raise SystemExit(f"no float array in {a.npz}; keys={keys}")
        k, v = max(cands, key=lambda kv: kv[1].size)
        print(f"[data] using state key {k!r}", flush=True)
        state = _as_btd(v, "state")
    stim = _as_btd(z[a.stim_key], "stim") if a.stim_key else None
    print(f"[data] state {state.shape}" + (f" stim {stim.shape}" if stim is not None else ""),
          flush=True)
    return _training_view(state, stim, a)


def plan_draws(state, a):
    """Return (rows, [start arrays], axis description).

    ``rows`` indexes the realization axis of ``state``; a single long series is
    broadcast to B identical rows so the two cases share one code path.

    If the series has a realization axis (B>1) the batch IS that axis and the
    draws are different t0.  A single long series falls back to batching over
    START indices -- a different, weaker axis -- and the draws are then disjoint
    interleaved start sets.  The probe records which one it used.
    """
    B, T, _ = state.shape
    need = a.burnin + a.K + 1
    if T < need:
        raise SystemExit(f"series too short: T={T} < burnin+K+1={need}")
    lo, hi = a.burnin, T - a.K - 2
    if hi < lo:
        raise SystemExit(f"no admissible start: burnin={a.burnin} K={a.K} T={T}")
    n_draw = max(1, int(a.draws))

    if B > 1:
        n = B if a.batch <= 0 else min(B, a.batch)
        rows = np.arange(n, dtype=np.int64)
        if a.t0 >= 0:
            if not (lo <= a.t0 <= hi):
                raise SystemExit(f"--t0 {a.t0} outside admissible [{lo},{hi}]")
            t0s = [int(a.t0)]
        else:
            t0s = sorted(set(np.linspace(lo, hi, n_draw).astype(np.int64).tolist()))
        draws = [np.full(n, int(t), dtype=np.int64) for t in t0s]
        axis = f"realizations (B={n} independent series; {len(draws)} t0 draws)"
        return rows, draws, axis

    n = a.batch if a.batch > 0 else 16
    rows = np.zeros(n, dtype=np.int64)          # broadcast the one series
    grid = np.unique(np.linspace(lo, hi, n * n_draw).astype(np.int64))
    draws = []
    for d in range(n_draw):
        s = grid[d::n_draw]
        if len(s) >= n:
            draws.append(s[:n])
    if not draws:
        draws = [grid[:n]]
    axis = f"starts (B={n} start indices in ONE series; {len(draws)} disjoint draws)"
    return rows, draws, axis


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def load_blob(path: str, device: str) -> dict:
    blob = torch.load(path, map_location=device)
    sd = blob
    for key in ("model", "state_dict", "model_state_dict"):
        if isinstance(blob, dict) and key in blob:
            sd = blob[key]
            break
    if isinstance(sd, dict):
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    return sd


def make_args(a, state_dim: int, stim_dim: int, max_horizon: int) -> argparse.Namespace:
    return argparse.Namespace(
        model_name="official_mamba_state",
        simple_hidden_dim=a.hidden,
        simple_depth=a.depth,
        simple_dropout=0.0,
        simple_residual=True,
        mamba_d_state=a.d_state,
        mamba_d_conv=a.d_conv,
        mamba_expand=a.expand,
        mamba_bptt_horizon=a.K,
        mamba_burnin=a.burnin,
        window_size=a.window,
        roi_dim=state_dim,
        state_dim=state_dim,
        output_dim=state_dim,
        n_parcels=state_dim,
        # registry.build_model reads the drive width from stim_dim and decides
        # whether the token carries it from dataset_has_external_input -- that
        # flag, not use_stim, is what sets in_proj's input width.
        stim_dim=stim_dim,
        input_dim=stim_dim,
        dataset_has_external_input=stim_dim > 0,
        use_stim=stim_dim > 0,
        resgrad_routing=True,
        resgrad_policy="dualwiener",
        resgrad_block_gate=0.0,
        resgrad_ratio_threshold=0.13,
        resgrad_target_open_frac=0.0,
        resgrad_outer=False,
        # buffer shape must match the checkpoint exactly
        dual_wiener_max_horizon=max_horizon,
        dual_wiener_ema=0.95,
        dual_wiener_residual_ema=0.99,
        dual_wiener_warmup_batches=8,
        dual_wiener_probe_every=4,
        dual_wiener_min_probes=1,
        dual_wiener_noise_model=getattr(
            a, "dual_wiener_noise_model", "diagonal_gaussian"
        ),
    )


# --------------------------------------------------------------------------
# nested-state helpers.  MambaStackState = Tuple[(conv_state, ssm_state), ...]
# --------------------------------------------------------------------------
def tree_leaves(obj, out=None):
    out = [] if out is None else out
    if torch.is_tensor(obj):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            tree_leaves(o, out)
    return out


def tree_map_tensors(obj, fn):
    if torch.is_tensor(obj):
        return fn(obj)
    if isinstance(obj, tuple):
        return tuple(tree_map_tensors(o, fn) for o in obj)
    if isinstance(obj, list):
        return [tree_map_tensors(o, fn) for o in obj]
    return obj


def state_labels(h) -> list[str]:
    """Label each recurrent-state leaf.  Falls back to opaque names if the
    stack is not the expected sequence of (conv_state, ssm_state) pairs."""
    labels: list[str] = []
    if isinstance(h, (list, tuple)):
        for li, layer in enumerate(h):
            if (isinstance(layer, (list, tuple)) and len(layer) == 2
                    and all(torch.is_tensor(t) for t in layer)):
                labels += [f"conv{li}", f"ssm{li}"]
            else:
                labels += [f"state{li}_{j}" for j in range(len(tree_leaves(layer)))]
    return labels
