#!/usr/bin/env python
"""Shared checkpoint, data, and model setup for the paper probes.

Data normalization, drive-width inference, stimulus pooling, and draw planning
are centralized in :mod:`probe_data_ops`.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))

from probe_data_ops import (  # noqa: E402
    load_arrays,
    load_blob,
    make_args,
    plan_draws,
    state_labels,
    tree_leaves,
    tree_map_tensors,
)

from internal_dw.models import build_model  # noqa: E402

__all__ = [
    "add_common_args", "setup", "Bundle",
    "tree_leaves", "tree_map_tensors", "state_labels",
]


def add_common_args(ap: argparse.ArgumentParser) -> None:
    """Register the argument names ``probe_data_ops.make_args`` reads."""
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz", default=None)
    ap.add_argument("--glob", default=None, help="one plain .npy per realization")
    ap.add_argument("--stim-glob", default=None)
    ap.add_argument("--hcp-dir", default=None,
                    help="HCP clip_sdxl_features_atlas root; batch = subjects at one timepoint")
    ap.add_argument("--hcp-split", choices=("all", "train", "val", "test"),
                    default="all",
                    help="optional seed-matched HCP subject split; calibration should use train")
    ap.add_argument("--hcp-train-ratio", type=float, default=0.7)
    ap.add_argument("--hcp-val-ratio", type=float, default=0.15)
    ap.add_argument("--movie", type=int, default=1)
    ap.add_argument("--roi-dim", type=int, default=400)
    ap.add_argument("--stim-dim", type=int, default=0,
                    help="model-side drive width; 0 = read it off the checkpoint's in_proj")
    ap.add_argument("--state-key", default=None)
    ap.add_argument("--stim-key", default=None)
    ap.add_argument(
        "--data-preprocess",
        choices=(
            "none", "mackey_glass", "mackey_glass_val", "mackey_glass_test",
            "narma", "narma_val", "narma_test",
            "ieeg", "ieeg_val", "ieeg_test",
            "prepared_temporal",
        ),
        default="none",
        help=(
            "reproduce the named testbed normalization and select train by "
            "default, or an explicit held-out _val/_test split"
        ),
    )
    ap.add_argument("--split-train-ratio", type=float, default=0.7)
    ap.add_argument("--split-val-ratio", type=float, default=0.15)
    ap.add_argument("--ieeg-chunk", type=int, default=1024)
    ap.add_argument("--ieeg-split-gap", type=int, default=256)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--burnin", type=int, default=32)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument(
        "--dual-wiener-noise-model",
        choices=("diagonal_gaussian", "lagged_residual_bootstrap", "spatial_spectrum"),
        default="diagonal_gaussian",
        help="noise estimator used by the controller built for this probe",
    )
    ap.add_argument("--d_state", type=int, default=16)
    ap.add_argument("--d_conv", type=int, default=4)
    ap.add_argument("--expand", type=int, default=2)
    ap.add_argument("--batch", type=int, default=0, help="0 = use every realization")
    ap.add_argument("--draws", type=int, default=4, help="independent t0 / start-set draws")
    ap.add_argument("--t0", type=int, default=-1,
                    help="pin one absolute start (realization axis); forces draws=1")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--exact-checkpoint-adapter",
        action="store_true",
        help=(
            "allow a checkpoint without routing-controller buffers by attaching "
            "a fresh fully-open internal Dual-Wiener controller for frozen VJP "
            "measurement; model parameters still come only from the checkpoint"
        ),
    )


class Bundle:
    """Everything a probe needs after setup."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def setup(a) -> Bundle:
    state, stim = load_arrays(a)
    state_dim = state.shape[2]

    sd = load_blob(a.ckpt, a.device)
    dual_coefficients = sd.get("dual_wiener.coefficients")
    exact_adapter = bool(getattr(a, "exact_checkpoint_adapter", False))
    if dual_coefficients is None and not exact_adapter:
        raise SystemExit(
            "checkpoint lacks dual_wiener.coefficients; pass "
            "--exact-checkpoint-adapter only "
            "for a frozen Exact-BPTT checkpoint"
        )
    if dual_coefficients is not None:
        max_horizon, depth_ck, _ = dual_coefficients.shape
        if depth_ck != a.depth:
            print(f"[ckpt] overriding --depth {a.depth} -> {depth_ck}", flush=True)
            a.depth = int(depth_ck)
        print(f"[ckpt] controller=routewise max_horizon={max_horizon} depth={depth_ck} "
              f"seen={int(sd.get('dual_wiener.seen_batches', torch.tensor(-1)))} "
              f"solved={int(sd.get('dual_wiener.solved_batches', torch.tensor(-1)))}", flush=True)
    else:
        # Exact-BPTT checkpoints contain only the backbone.  A probe still
        # needs the route autograd nodes in order to observe identity and
        # nonlinear covectors.  Attach a fresh controller whose coefficients
        # are initialized fully open; none of its empty moments are used.
        max_horizon = int(a.K)
        depth_ck = int(a.depth)
        print(
            f"[ckpt] controller=exact_adapter max_horizon={max_horizon} "
            f"depth={depth_ck}; temporary internal routes are fully open",
            flush=True,
        )
    if a.K > max_horizon:
        raise SystemExit(f"--K {a.K} exceeds the checkpoint's max_horizon {max_horizon}")

    # Model-side drive width is not the raw feature width (HCP pools 256*1664
    # tokens down to input_dim).  Read it off in_proj and apply the same pool.
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
            print(f"[data] pooling stim {raw_stim_dim} -> {stim_dim} "
                  f"(mean over {npool} tokens, matching SimpleAR._pool_stim)", flush=True)
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

    return Bundle(model=model, dw=dw, xt=xt, ut=ut, rows_t=rows_t,
                  draw_starts=draw_starts, axis=axis, state_dim=int(state_dim),
                  stim_dim=int(stim_dim), max_horizon=int(max_horizon),
                  depth=int(a.depth), batch=int(len(rows)))


def burn_in_root(bundle: Bundle, starts, a):
    """Teacher-forced burn-in under no_grad, then make the root a set of leaves.

    Returns ``(cur, h, roots, labels)`` where ``roots`` is the flat leaf list
    ``[frame] + state leaves`` -- the same root definition the reach probe uses,
    so reach numbers and these numbers refer to the same point in the graph.
    """
    model, xt, ut, rows_t = bundle.model, bundle.xt, bundle.ut, bundle.rows_t
    st = torch.as_tensor(starts, device=a.device)
    with torch.no_grad():
        h = None
        for j in range(a.burnin, 0, -1):
            frame = xt[rows_t, st - j]
            stim = ut[rows_t, st - j] if ut is not None else None
            h = model.step(h, frame, stim_t=stim, horizon_index=-1, total_horizon=a.K)[1]
        cur0 = xt[rows_t, st]
    cur = cur0.detach().clone().requires_grad_(True)
    h = tree_map_tensors(h, lambda t: t.detach().clone().requires_grad_(True))
    roots = [cur] + tree_leaves(h)
    labels = ["frame"] + state_labels(h)
    if len(labels) != len(roots):
        labels = ["frame"] + [f"state{i}" for i in range(len(roots) - 1)]
    return cur, h, roots, labels, st
