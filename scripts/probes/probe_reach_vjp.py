#!/usr/bin/env python
"""Measure the ACTUAL backward reach of a trained Dual-Wiener gate.

The gate's own diagnostic ``log_identity_reach`` is only the log-sum of the
internal ``alpha``.  That quantity ignores three things that also carry delayed
credit:

  * the ``m * J_F`` branch route,
  * every mixed path in the expansion of ``prod(alpha*I + m*J_F)``,
  * the OUTER autoregressive skip ``x_{t+1} = x_t + Delta_t``, which is not
    routed at all.

So ``prod(alpha)`` is a lower bound on one particular path, not the reach.  This
probe measures the reach directly: it back-propagates each horizon's loss to the
rollout root and compares the resulting root-VJP against the same VJP computed
with the gate fully open.

Because the operator is backward-only (straight-through), the forward pass is
identical across arms -- the probe asserts this and reports the residual.

Arms (all on the same minibatch, same burn-in, same rollout):

  open     alpha=1, m=1                 reference
  learned  alpha, m from the checkpoint
  alpha1   alpha=1, m=m_learned         isolates what alpha<1 alone costs
  m0       alpha=alpha_learned, m=0     internal identity chain only
                                        (the outer AR skip is still open, so
                                         comparing this arm's ratio against
                                         prod(alpha) shows how much credit the
                                         unrouted outer skip carries)

FORMAT 2 adds, over format 1:

  * ``--draws N``    N independent (t0 / start-set) draws; every statistic is
                     reported as mean and sd across draws, so the numbers carry
                     an error bar instead of resting on one minibatch.
  * component split  the root is [input frame] + [(conv_state, ssm_state) per
                     layer].  Ratios and cosines are reported for ``all``,
                     ``frame``, ``conv``, ``ssm`` and ``state`` (=conv+ssm)
                     separately, because a gate can preserve the frame route
                     while destroying the recurrent one and the pooled number
                     hides that.
  * float64          every norm, dot and cosine is REDUCED in float64.  The
                     fMRI root is ~2.6e6 wide and cosines reach 1e-2, where a
                     float32 accumulation is not trustworthy.  (Vectors are
                     stored float32 -- they come out of a float32 model, so the
                     cast adds no information; only the reduction needs the
                     wider accumulator.)
  * both losses      each rollout is differentiated twice, under rel_l2 (the
                     training loss) AND under 0.5*mse (the loss the Dual-Wiener
                     calibration probe q_tot actually uses).  Same graph, same
                     minibatch, so the two are directly comparable.

Usage
-----
  # Mackey-Glass (batch axis = 40 independent trajectories, 8 different t0)
  python scripts/probes/probe_reach_vjp.py \
      --ckpt experiments/dual_wiener_screen/mackey_glass/tau30_K32/dualwiener/seed0/last.pth \
      --npz  data/synthetic/mg_D8_tau30.0_dt1.0_sdt0.1_T2048_traj40_b0.2_g0.1_n10.0_tr1000_s0.npz \
      --state-key trajs --hidden 128 --K 32 --burnin 32 --draws 8 \
      --out reach_mg.npz

  # NARMA (driven: pass the stimulus)
  python scripts/probes/probe_reach_vjp.py \
      --ckpt experiments/dual_wiener_screen/narma/L5_K32/dualwiener/seed0/last.pth \
      --npz  data/synthetic/narma_D8_L5_T2048_traj40_tr200_u0.5_bd1_dr1.5_s0.npz \
      --state-key y --stim-key u --hidden 128 --K 32 --burnin 32 --draws 8 \
      --out reach_narma.npz

  # iEEG (single series -> batch axis is different STARTS, not realizations)
  python scripts/probes/probe_reach_vjp.py \
      --ckpt experiments/dual_wiener_screen/ieeg/theta_K64/dualwiener/seed0/last.pth \
      --npz  data/synthetic/ieeg_P41CS_enc_macro_theta_20ms_ch0.npz \
      --state-key X --hidden 256 --K 64 --burnin 32 --batch 16 --draws 8 \
      --out reach_ieeg.npz

  # HCP fMRI (batch axis = subjects at ONE movie timepoint)
  export HCP_DATA_PATH=/path/to/hcp_movie_features
  python scripts/probes/probe_reach_vjp.py \
      --ckpt experiments/hcp_movie1/dual_wiener_screen/.../seed0/last.pth \
      --hcp-dir "${HCP_DATA_PATH}" \
      --movie 1 --hidden 4096 --K 64 --burnin 32 --batch 4 --draws 3 \
      --out reach_fmri.npz
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from internal_dw.models import build_model  # noqa: E402

FORMAT_VERSION = 2
LOSSES = ("rel_l2", "mse")
COMPONENTS = ("all", "frame", "conv", "ssm", "state")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
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


def component_index(labels: list[str], sizes: list[int]) -> dict[str, torch.Tensor]:
    """Map each component name to the flat indices it occupies (empty if absent)."""
    offs, o = [], 0
    for n in sizes:
        offs.append((o, o + n))
        o += n
    total = o
    buckets: dict[str, list[tuple[int, int]]] = {c: [] for c in COMPONENTS}
    for lab, rng in zip(labels, offs):
        if lab == "frame":
            buckets["frame"].append(rng)
        elif lab.startswith("conv"):
            buckets["conv"].append(rng)
            buckets["state"].append(rng)
        elif lab.startswith("ssm"):
            buckets["ssm"].append(rng)
            buckets["state"].append(rng)
        else:                                   # unexpected leaf: count as state
            buckets["state"].append(rng)
    out: dict[str, torch.Tensor] = {}
    for c, rs in buckets.items():
        if c == "all":
            continue
        out[c] = (torch.cat([torch.arange(s, e, dtype=torch.long) for s, e in rs])
                  if rs else torch.zeros(0, dtype=torch.long))
    out["all"] = torch.arange(total, dtype=torch.long)   # slot 0 == the whole vector
    return out


# --------------------------------------------------------------------------
# the measurement
# --------------------------------------------------------------------------
def per_horizon_loss(pred: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "mse":
        # matches the Dual-Wiener calibration probe q_tot = 0.5*mean||e||^2
        return 0.5 * (pred - target).pow(2).mean()
    num = (pred - target).pow(2).sum(dim=1).clamp_min(1e-30).sqrt()
    den = target.pow(2).sum(dim=1).clamp_min(1e-30).sqrt()
    return (num / den).mean()


def run_arm(model, xt, ut, rows_t, starts, a, device, on_layout, on_grad):
    """Roll out K steps from a fresh root; report layout, then every root-VJP.

    ``on_layout(labels, sizes)`` fires once, before any gradient.
    ``on_grad(loss_kind, k, flat_f32_cpu)`` fires 2*K times.  The vector is the
    root-VJP flattened over [frame, conv0, ssm0, ...] and concatenated over
    batch rows.  Returns preds[K] on the CPU.
    """
    dw = model.dual_wiener
    if dw is not None:
        dw._collecting = False
        dw._mode = "apply"

    st = torch.as_tensor(starts, device=device)

    # --- teacher-forced burn-in, no grad: defines a realistic root -----------
    with torch.no_grad():
        h = None
        for j in range(a.burnin, 0, -1):
            frame = xt[rows_t, st - j]
            stim = ut[rows_t, st - j] if ut is not None else None
            h = model.step(h, frame, stim_t=stim, horizon_index=-1, total_horizon=a.K)[1]
        cur0 = xt[rows_t, st]

    # --- make the root a leaf ----------------------------------------------
    cur = cur0.detach().clone().requires_grad_(True)
    h = tree_map_tensors(h, lambda t: t.detach().clone().requires_grad_(True))
    roots = [cur] + tree_leaves(h)
    labels = ["frame"] + state_labels(h)
    if len(labels) != len(roots):                       # defensive
        labels = ["frame"] + [f"state{i}" for i in range(len(roots) - 1)]
    sizes = [int(r.numel()) for r in roots]
    on_layout(labels, sizes)

    # --- rollout with grad --------------------------------------------------
    preds = []
    losses = {k: [] for k in LOSSES}
    with torch.enable_grad():
        c = cur
        for k in range(a.K):
            stim = ut[rows_t, st + k] if ut is not None else None
            out = model.step(h, c, stim_t=stim, horizon_index=k, total_horizon=a.K)
            pred, h = out[0], out[1]
            tgt = xt[rows_t, st + k + 1]
            preds.append(pred.detach().float().cpu())
            for kind in LOSSES:
                losses[kind].append(per_horizon_loss(pred, tgt, kind))
            c = pred

        # One graph, differentiated 2*K times; the last call frees it.
        pairs = [(kind, k) for kind in LOSSES for k in range(a.K)]
        for i, (kind, k) in enumerate(pairs):
            gs = torch.autograd.grad(
                losses[kind][k], roots,
                retain_graph=(i < len(pairs) - 1), allow_unused=True,
            )
            flat = torch.cat([
                (g if g is not None else torch.zeros_like(r)).reshape(-1).float().cpu()
                for g, r in zip(gs, roots)
            ])
            on_grad(kind, k, flat)
    return torch.stack(preds)


class Acc:
    """Accumulate norm/ratio/cos per (loss, arm, component, k) across draws."""

    def __init__(self, K: int):
        self.K = K
        self.d: dict[tuple, list[float]] = {}

    def add(self, kind, arm, comp, k, key, v):
        self.d.setdefault((kind, arm, comp, k, key), []).append(float(v))

    def stat(self, kind, arm, comp, key):
        m = np.full(self.K, np.nan)
        s = np.full(self.K, np.nan)
        n = np.zeros(self.K, dtype=np.int64)
        for k in range(self.K):
            v = self.d.get((kind, arm, comp, k, key))
            if v:
                m[k] = float(np.mean(v))
                s[k] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
                n[k] = len(v)
        return m, s, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz", default=None)
    ap.add_argument("--glob", default=None, help="one plain .npy per realization")
    ap.add_argument("--stim-glob", default=None)
    ap.add_argument("--hcp-dir", default=None,
                    help="HCP clip_sdxl_features_atlas root; batch = subjects at one timepoint")
    ap.add_argument("--movie", type=int, default=1)
    ap.add_argument("--roi-dim", type=int, default=400)
    ap.add_argument("--stim-dim", type=int, default=0,
                    help="model-side drive width; 0 = read it off the checkpoint's in_proj")
    ap.add_argument("--state-key", default=None)
    ap.add_argument("--stim-key", default=None)
    ap.add_argument("--out", default="reach_vjp.npz")
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--burnin", type=int, default=32)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--d_state", type=int, default=16)
    ap.add_argument("--d_conv", type=int, default=4)
    ap.add_argument("--expand", type=int, default=2)
    ap.add_argument("--batch", type=int, default=0, help="0 = use every realization")
    ap.add_argument("--draws", type=int, default=8, help="independent t0 / start-set draws")
    ap.add_argument("--t0", type=int, default=-1,
                    help="pin one absolute start (realization axis); forces draws=1")
    ap.add_argument("--arms", default="open,learned,alpha1,m0")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    state, stim = load_arrays(a)
    state_dim = state.shape[2]

    sd = load_blob(a.ckpt, a.device)
    if "dual_wiener.coefficients" not in sd:
        raise SystemExit("checkpoint has no dual_wiener.coefficients -- this probe is for the "
                         "dualwiener policy only")
    max_horizon, depth_ck, _ = sd["dual_wiener.coefficients"].shape
    if depth_ck != a.depth:
        print(f"[ckpt] overriding --depth {a.depth} -> {depth_ck}", flush=True)
        a.depth = int(depth_ck)
    print(f"[ckpt] max_horizon={max_horizon} depth={depth_ck} "
          f"seen={int(sd.get('dual_wiener.seen_batches', torch.tensor(-1)))} "
          f"solved={int(sd.get('dual_wiener.solved_batches', torch.tensor(-1)))}", flush=True)

    # The MODEL-side drive width is not the raw feature width: HCP CLIP/SDXL
    # features arrive as 256*1664 flattened tokens and the backbone mean-pools
    # them down to input_dim.  Read that width off in_proj so we never guess it,
    # then apply the SAME pooling here (cheaper than shipping the raw tensor).
    raw_stim_dim = int(stim.shape[2]) if stim is not None else 0
    stim_dim = raw_stim_dim
    if raw_stim_dim:
        w = sd.get("in_proj.1.weight")
        w0 = sd.get("in_proj.0.weight")
        in_width = int(w.shape[1]) if w is not None else (int(w0.shape[0]) if w0 is not None else None)
        if a.stim_dim > 0:
            stim_dim = a.stim_dim
        elif in_width is not None:
            stim_dim = in_width - state_dim
        if stim_dim <= 0:
            raise SystemExit(f"derived stim_dim={stim_dim} from in_proj width {in_width} "
                             f"and state_dim {state_dim}; pass --stim-dim explicitly")
        if stim_dim != raw_stim_dim:
            if raw_stim_dim % stim_dim:
                raise SystemExit(f"raw stim {raw_stim_dim} is not a multiple of model stim_dim "
                                 f"{stim_dim}; cannot reproduce the backbone's mean-pool")
            npool = raw_stim_dim // stim_dim
            print(f"[data] pooling stim {raw_stim_dim} -> {stim_dim} (mean over {npool} tokens, "
                  f"matching SimpleAR._pool_stim)", flush=True)
            stim = stim.reshape(stim.shape[0], stim.shape[1], npool, stim_dim).mean(axis=2)

    model = build_model(make_args(a, state_dim, stim_dim, int(max_horizon)))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[ckpt] {len(missing)} missing keys (first: {missing[:3]})", flush=True)
    if unexpected:
        print(f"[ckpt] {len(unexpected)} unexpected keys (first: {unexpected[:3]})", flush=True)
    model = model.to(a.device).eval()
    dw = model.dual_wiener
    if dw is None:
        raise SystemExit("model was not built with a DualWienerController")

    if a.t0 >= 0:
        a.draws = 1
    rows, draw_starts, axis = plan_draws(state, a)
    if stim_dim and stim is None:
        raise SystemExit("model expects external input but no stimulus was supplied")
    print(f"[batch] axis = {axis}", flush=True)

    xt = torch.as_tensor(state[rows], dtype=torch.float32, device=a.device)
    ut = (torch.as_tensor(stim[rows], dtype=torch.float32, device=a.device)
          if stim is not None else None)
    rows_t = torch.arange(len(rows), device=a.device)

    learned = dw.coefficients.detach().clone()
    alpha_l = learned[: a.K, :, 0]
    log_prod_alpha = (
        torch.log(alpha_l.clamp_min(1e-30)).sum(dim=1).cumsum(dim=0).double().cpu()
    )

    def coeffs_for(arm: str) -> torch.Tensor:
        c = learned.clone()
        if arm == "open":
            c[...] = 1.0
        elif arm == "learned":
            pass
        elif arm == "alpha1":
            c[..., 0] = 1.0
        elif arm == "m0":
            c[..., 1] = 0.0
        else:
            raise SystemExit(f"unknown arm {arm!r}")
        return c

    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    arms = ["open"] + [x for x in arms if x != "open"]     # open must run first

    acc = Acc(a.K)
    fwd_dev = {arm: 0.0 for arm in arms}
    layout = {"labels": None, "sizes": None, "index": None}

    def on_layout(labels, sizes):
        if layout["labels"] is None:
            layout["labels"] = labels
            layout["sizes"] = sizes
            layout["index"] = component_index(labels, sizes)
            print("[root] " + ", ".join(f"{l}:{s}" for l, s in zip(labels, sizes))
                  + f"  (total {sum(sizes)})", flush=True)

    for di, starts in enumerate(draw_starts):
        ref: dict[tuple, torch.Tensor] = {}
        preds_ref = None
        for arm in arms:
            dw.coefficients.copy_(coeffs_for(arm))

            def on_grad(kind, k, flat, _arm=arm):
                idx = layout["index"]
                if _arm == "open":
                    ref[(kind, k)] = flat
                for comp in COMPONENTS:
                    sel = idx[comp]
                    if sel.numel() == 0:
                        continue
                    gv = flat.double() if comp == "all" else flat[sel].double()
                    gn = float(gv.norm())
                    acc.add(kind, _arm, comp, k, "norm", gn)
                    if _arm == "open":
                        acc.add(kind, _arm, comp, k, "ratio", 1.0)
                        acc.add(kind, _arm, comp, k, "cos", 1.0)
                        continue
                    r = ref[(kind, k)]
                    rv = r.double() if comp == "all" else r[sel].double()
                    rn = float(rv.norm())
                    acc.add(kind, _arm, comp, k, "ratio",
                            gn / rn if rn > 0 else float("nan"))
                    acc.add(kind, _arm, comp, k, "cos",
                            float(torch.dot(gv, rv)) / (gn * rn)
                            if gn > 0 and rn > 0 else float("nan"))

            preds = run_arm(model, xt, ut, rows_t, starts, a, a.device, on_layout, on_grad)
            if preds_ref is None:
                preds_ref = preds
            else:
                fwd_dev[arm] = max(fwd_dev[arm], float((preds - preds_ref).abs().max()))
                del preds
        del ref, preds_ref
        print(f"[draw {di + 1}/{len(draw_starts)}] starts {starts[0]}..{starts[-1]} done",
              flush=True)
    dw.coefficients.copy_(learned)

    # ---- pack -------------------------------------------------------------
    table = {"horizons": np.arange(a.K), "log_prod_alpha": log_prod_alpha.numpy()}
    cache = {}
    for kind in LOSSES:
        for arm in arms:
            for comp in COMPONENTS:
                for key in ("norm", "ratio", "cos"):
                    m, s, n = acc.stat(kind, arm, comp, key)
                    cache[(kind, arm, comp, key)] = (m, s, n)
                    base = f"{kind}__{arm}__{comp}__{key}"
                    table[base + "_mean"] = m
                    table[base + "_sd"] = s
                    table[base + "_n"] = n

    # ---- report -----------------------------------------------------------
    sizes = layout["sizes"]
    print()
    print(f"batch axis : {axis}")
    print(f"draws      : {len(draw_starts)}   K={a.K}  burnin={a.burnin}")
    print(f"root       : {sum(sizes)} dims over {len(sizes)} leaves; reductions in float64")
    print(f"forward deviation across arms (should be 0, straight-through): "
          f"{max(fwd_dev.values()):.3e}")
    other = [x for x in arms if x != "open"]
    for kind in LOSSES:
        for comp in ("all", "frame", "state"):
            print()
            print(f"### loss={kind}   component={comp}   (mean +- sd over {len(draw_starts)} draws)")
            hdr = f"{'k':>4s} {'||g_open||':>11s}" + "".join(
                f" {arm[:7] + '_rat':>18s} {arm[:7] + '_cos':>18s}" for arm in other
            ) + f" {'prod(alpha)':>11s}"
            print(hdr)
            print("-" * len(hdr))
            om = cache[(kind, "open", comp, "norm")][0]
            for k in range(a.K):
                if not (k < 4 or k >= a.K - 3 or k % max(1, a.K // 6) == 0):
                    continue
                row = f"{k:4d} {om[k]:11.4e}"
                for arm in other:
                    rm, rs, _ = cache[(kind, arm, comp, "ratio")]
                    cm, cs, _ = cache[(kind, arm, comp, "cos")]
                    row += f" {rm[k]:9.3e}+-{rs[k]:7.1e} {cm[k]:9.4f}+-{cs[k]:7.1e}"
                row += f" {np.exp(np.clip(table['log_prod_alpha'][k], -700, 0)):11.4e}"
                print(row)

    meta = {"format_version": FORMAT_VERSION, "ckpt": a.ckpt, "axis": axis,
            "losses": list(LOSSES), "components": list(COMPONENTS), "K": a.K,
            "burnin": a.burnin, "arms": arms, "state_dim": int(state_dim),
            "stim_dim": int(stim_dim), "batch": int(len(rows)),
            "draws": [d.tolist() for d in draw_starts],
            "root_labels": layout["labels"], "root_sizes": sizes,
            "reduction_dtype": "float64", "storage_dtype": "float32",
            "forward_deviation": fwd_dev}
    np.savez(a.out, meta=json.dumps(meta), **table)
    print(f"\n[out] {a.out}", flush=True)


if __name__ == "__main__":
    main()
