#!/usr/bin/env python
"""Horizon-resolved held-out descent diagnostic at a frozen Exact checkpoint.

The probe deliberately does not estimate innovation SNR.  For disjoint
calibration/evaluation examples A and B it measures whether a unit parameter
step in a horizon-specific gradient direction from A is a descent direction on
B.  It also separates the temporal delayed component

    g_delay(k) = g_full(k) - g_local(k),

where g_local(k) is obtained by detaching the state/history entering the final
rollout step.  Detaching changes no forward value and retains the final model
call's parameter gradient.  Thus g_delay contains credit from loss k to model
calls at earlier rollout steps.  This is a path intervention, not a statistical
signal/noise decomposition.

Primary results are first-order and require no step-size choice.  A small set
of horizons additionally receives reversible, equal-Euclidean-norm parameter
steps and is evaluated by a fresh forward pass on B.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from internal_dw.data_utils.state_ops import unpack_batch, zero_external_input_like
from internal_dw.datasets.registry import build_dataloaders
from internal_dw.models.registry import build_model
from internal_dw.utils import load_checkpoint, unwrap_model
from gradient_probe_ops import (
    GradientProjection,
    burn_recurrent as _burn_recurrent,
    detach_state as _detach_state,
    force_fully_open as _force_fully_open,
    infer_thewell_shape as _infer_thewell_shape,
    namespace as _namespace,
    restore_open_state as _restore_open_state,
    snapshot_open_state as _snapshot_open_state,
    step_loss as _step_loss,
)


def _tree_detach(raw, value):
    return _detach_state(raw, value)


def _tree_max_abs(a, b) -> float:
    if torch.is_tensor(a):
        return float((a.detach() - b.detach()).abs().max().cpu())
    if isinstance(a, (tuple, list)):
        return max((_tree_max_abs(x, y) for x, y in zip(a, b)), default=0.0)
    return 0.0


def _prepare_stim(state: torch.Tensor, stim: Optional[torch.Tensor], args) -> torch.Tensor:
    if stim is None:
        return zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))
    return stim


def _strict_candidate_pool(
    loader, args, recurrent: bool, K: int, count: int, seed: int,
    start_fraction: float,
):
    """Return at most one start from each loader batch.

    This prevents two members of a held-out pair from being two starts chosen
    from the same loader item.  Dataset-level metadata are exported so stronger
    grouping assumptions can be audited after the run.
    """
    window = int(getattr(args, "window_size", 1))
    burn = max(0, int(getattr(args, "mamba_burnin", 0))) if recurrent else 0
    rng = random.Random(int(seed))
    fraction = min(max(float(start_fraction), 0.0), 1.0)
    wanted = max(int(count), 4 * int(count))
    candidates = []
    for batch_index, batch in enumerate(loader):
        state, stim, _, metadata = unpack_batch(batch)
        state = state[:1].cpu()
        stim = stim[:1].cpu() if stim is not None else None
        total_time = int(state.shape[1])
        low = max(1, burn) if recurrent else window
        high = total_time - int(K)
        if high < low:
            continue
        # One common relative position per loader item.  This aligns the
        # rollout-start semantics without assuming a universal absolute time
        # origin across subjects, trajectories, dates, or simulations.
        start = int(round(low + fraction * (high - low)))
        candidates.append(
            {
                "state": state,
                "stim": stim,
                "start": int(start),
                "batch_index": int(batch_index),
                "metadata": str(metadata),
            }
        )
        if len(candidates) >= wanted:
            break
    rng.shuffle(candidates)
    return candidates[: int(count)]


def _projected_norm(vector: torch.Tensor, projection: GradientProjection) -> float:
    # Fixed coordinate sampling estimates a full-space squared norm after p/m
    # scaling.  For an unprojected vector this factor is one.
    scale = float(projection.total) / float(projection.size)
    return math.sqrt(max(scale * float(torch.dot(vector, vector)), 0.0))


def _projected_dot(a: torch.Tensor, b: torch.Tensor, projection: GradientProjection) -> float:
    scale = float(projection.total) / float(projection.size)
    return scale * float(torch.dot(a, b))


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(a.norm() * b.norm())
    if not math.isfinite(denom) or denom <= 1e-30:
        return 0.0
    return float(torch.dot(a, b) / denom)


def _summary(array: np.ndarray) -> dict[str, Any]:
    ddof = 1 if int(array.shape[0]) > 1 else 0
    return {
        "mean": np.mean(array, axis=0).tolist(),
        "sd": np.std(array, axis=0, ddof=ddof).tolist(),
        "median": np.median(array, axis=0).tolist(),
        "q10": np.quantile(array, 0.10, axis=0).tolist(),
        "q90": np.quantile(array, 0.90, axis=0).tolist(),
    }


def _recurrent_full_local_projected(raw, params, projection, state, stim, start, K, args, loss_type):
    burn = max(0, int(getattr(args, "mamba_burnin", 0)))
    hidden = _burn_recurrent(raw, state, stim, start, burn)
    x_in = state[:, int(start) - 1]
    losses = []
    predictions = []
    local_inputs = []
    for k in range(int(K)):
        target_t = int(start) + k
        stim_in = stim[:, target_t - 1] if stim is not None else None
        local_inputs.append((_tree_detach(raw, hidden), x_in.detach(), stim_in))
        prediction, hidden = raw.step(
            hidden, x_in, stim_in, return_aux=False,
            horizon_index=k, total_horizon=K,
        )
        predictions.append(prediction.detach())
        losses.append(_step_loss(prediction, state[:, target_t], loss_type))
        x_in = prediction

    full_vectors = []
    full_losses = []
    for k, loss in enumerate(losses):
        grads = torch.autograd.grad(
            loss, params, retain_graph=(k < int(K) - 1), allow_unused=True
        )
        full_vectors.append(projection.apply(grads))
        full_losses.append(float(loss.detach().cpu()))

    local_vectors = []
    local_losses = []
    max_forward_diff = 0.0
    for k, (hidden_in, x_local, stim_in) in enumerate(local_inputs):
        target_t = int(start) + k
        prediction, _ = raw.step(
            hidden_in, x_local, stim_in, return_aux=False,
            horizon_index=k, total_horizon=K,
        )
        max_forward_diff = max(
            max_forward_diff,
            float((prediction.detach() - predictions[k]).abs().max().cpu()),
        )
        loss = _step_loss(prediction, state[:, target_t], loss_type)
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        local_vectors.append(projection.apply(grads))
        local_losses.append(float(loss.detach().cpu()))
    return full_vectors, local_vectors, full_losses, local_losses, max_forward_diff


def _window_full_at_horizon(raw, params, projection, state, stim, start, horizon, K, args, loss_type):
    window = int(getattr(args, "window_size", 1))
    history = state[:, int(start) - window : int(start)]
    prediction = None
    for k in range(int(horizon)):
        target_t = int(start) + k
        stim_window = stim[:, target_t - window : target_t] if stim is not None else None
        if hasattr(raw, "set_resgrad_context"):
            raw.set_resgrad_context(k, K)
        prediction = raw(stim_window, history, return_aux=False)
        history = torch.cat([history[:, 1:], prediction], dim=1)
    target = state[:, int(start) + int(horizon) - 1 : int(start) + int(horizon)]
    loss = _step_loss(prediction, target, loss_type)
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return projection.apply(grads), prediction.detach(), float(loss.detach().cpu())


def _window_local_at_horizon(raw, params, projection, state, stim, start, horizon, K, args, loss_type):
    window = int(getattr(args, "window_size", 1))
    history = state[:, int(start) - window : int(start)]
    with torch.no_grad():
        for k in range(max(0, int(horizon) - 1)):
            target_t = int(start) + k
            stim_window = stim[:, target_t - window : target_t] if stim is not None else None
            if hasattr(raw, "set_resgrad_context"):
                raw.set_resgrad_context(k, K)
            prediction = raw(stim_window, history, return_aux=False)
            history = torch.cat([history[:, 1:], prediction], dim=1)
    history = history.detach()
    k = int(horizon) - 1
    target_t = int(start) + k
    stim_window = stim[:, target_t - window : target_t] if stim is not None else None
    if hasattr(raw, "set_resgrad_context"):
        raw.set_resgrad_context(k, K)
    prediction = raw(stim_window, history, return_aux=False)
    target = state[:, target_t : target_t + 1]
    loss = _step_loss(prediction, target, loss_type)
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return projection.apply(grads), prediction.detach(), float(loss.detach().cpu())


def _full_local_projected(raw, params, projection, candidate, K, args, recurrent, loss_type, device):
    state = candidate["state"].to(device=device, dtype=torch.float32)
    stim = candidate["stim"]
    stim = None if stim is None else stim.to(device=device, dtype=torch.float32)
    stim = _prepare_stim(state, stim, args)
    if recurrent:
        return _recurrent_full_local_projected(
            raw, params, projection, state, stim, candidate["start"], K, args, loss_type
        )

    full, local, full_losses, local_losses = [], [], [], []
    max_forward_diff = 0.0
    for horizon in range(1, int(K) + 1):
        fv, fp, fl = _window_full_at_horizon(
            raw, params, projection, state, stim, candidate["start"], horizon, K, args, loss_type
        )
        lv, lp, ll = _window_local_at_horizon(
            raw, params, projection, state, stim, candidate["start"], horizon, K, args, loss_type
        )
        full.append(fv)
        local.append(lv)
        full_losses.append(fl)
        local_losses.append(ll)
        max_forward_diff = max(
            max_forward_diff, float((fp - lp).abs().max().cpu())
        )
        if horizon == 1 or horizon == K or horizon % 8 == 0:
            print(f"      horizon {horizon}/{K}", flush=True)
    return full, local, full_losses, local_losses, max_forward_diff


def _grads_to_dense(grads, params):
    return [
        (grad.detach() if grad is not None else torch.zeros_like(param))
        for grad, param in zip(grads, params)
    ]


def _full_local_tensors_at_horizon(raw, params, candidate, horizon, K, args, recurrent, loss_type, device):
    state = candidate["state"].to(device=device, dtype=torch.float32)
    stim = candidate["stim"]
    stim = None if stim is None else stim.to(device=device, dtype=torch.float32)
    stim = _prepare_stim(state, stim, args)
    start = int(candidate["start"])

    if recurrent:
        burn = max(0, int(getattr(args, "mamba_burnin", 0)))
        hidden = _burn_recurrent(raw, state, stim, start, burn)
        x_in = state[:, start - 1]
        local_hidden = local_x = local_stim = None
        prediction = None
        for k in range(int(horizon)):
            target_t = start + k
            stim_in = stim[:, target_t - 1] if stim is not None else None
            if k == int(horizon) - 1:
                local_hidden, local_x, local_stim = _tree_detach(raw, hidden), x_in.detach(), stim_in
            prediction, hidden = raw.step(
                hidden, x_in, stim_in, return_aux=False,
                horizon_index=k, total_horizon=K,
            )
            x_in = prediction
        target = state[:, start + int(horizon) - 1]
        full_loss = _step_loss(prediction, target, loss_type)
        full = _grads_to_dense(torch.autograd.grad(full_loss, params, allow_unused=True), params)
        local_prediction, _ = raw.step(
            local_hidden, local_x, local_stim, return_aux=False,
            horizon_index=int(horizon) - 1, total_horizon=K,
        )
        local_loss = _step_loss(local_prediction, target, loss_type)
        local = _grads_to_dense(torch.autograd.grad(local_loss, params, allow_unused=True), params)
        fwd_diff = float((prediction.detach() - local_prediction.detach()).abs().max().cpu())
    else:
        window = int(getattr(args, "window_size", 1))
        history = state[:, start - window : start]
        prediction = None
        local_history = None
        for k in range(int(horizon)):
            target_t = start + k
            stim_window = stim[:, target_t - window : target_t] if stim is not None else None
            if hasattr(raw, "set_resgrad_context"):
                raw.set_resgrad_context(k, K)
            if k == int(horizon) - 1:
                local_history = history.detach()
                local_stim = stim_window
            prediction = raw(stim_window, history, return_aux=False)
            history = torch.cat([history[:, 1:], prediction], dim=1)
        target_t = start + int(horizon) - 1
        target = state[:, target_t : target_t + 1]
        full_loss = _step_loss(prediction, target, loss_type)
        full = _grads_to_dense(torch.autograd.grad(full_loss, params, allow_unused=True), params)
        if hasattr(raw, "set_resgrad_context"):
            raw.set_resgrad_context(int(horizon) - 1, K)
        local_prediction = raw(local_stim, local_history, return_aux=False)
        local_loss = _step_loss(local_prediction, target, loss_type)
        local = _grads_to_dense(torch.autograd.grad(local_loss, params, allow_unused=True), params)
        fwd_diff = float((prediction.detach() - local_prediction.detach()).abs().max().cpu())

    delayed = [f - l for f, l in zip(full, local)]
    return full, local, delayed, fwd_diff


def _rollout_losses(raw, candidate, K, args, recurrent, loss_type, device) -> np.ndarray:
    state = candidate["state"].to(device=device, dtype=torch.float32)
    stim = candidate["stim"]
    stim = None if stim is None else stim.to(device=device, dtype=torch.float32)
    stim = _prepare_stim(state, stim, args)
    start = int(candidate["start"])
    values = []
    with torch.no_grad():
        if recurrent:
            burn = max(0, int(getattr(args, "mamba_burnin", 0)))
            hidden = _burn_recurrent(raw, state, stim, start, burn)
            x_in = state[:, start - 1]
            for k in range(int(K)):
                target_t = start + k
                stim_in = stim[:, target_t - 1] if stim is not None else None
                prediction, hidden = raw.step(
                    hidden, x_in, stim_in, return_aux=False,
                    horizon_index=k, total_horizon=K,
                )
                values.append(float(_step_loss(prediction, state[:, target_t], loss_type).cpu()))
                x_in = prediction
        else:
            window = int(getattr(args, "window_size", 1))
            history = state[:, start - window : start]
            for k in range(int(K)):
                target_t = start + k
                stim_window = stim[:, target_t - window : target_t] if stim is not None else None
                if hasattr(raw, "set_resgrad_context"):
                    raw.set_resgrad_context(k, K)
                prediction = raw(stim_window, history, return_aux=False)
                values.append(float(_step_loss(
                    prediction, state[:, target_t : target_t + 1], loss_type
                ).cpu()))
                history = torch.cat([history[:, 1:], prediction], dim=1)
    return np.asarray(values, dtype=np.float64)


def _tensor_norm(vectors: Iterable[torch.Tensor]) -> float:
    return math.sqrt(sum(float(v.double().pow(2).sum().cpu()) for v in vectors))


def _tensor_dot(a: Iterable[torch.Tensor], b: Iterable[torch.Tensor]) -> float:
    return sum(float((x.double() * y.double()).sum().cpu()) for x, y in zip(a, b))


def _parameter_norm(params) -> float:
    return math.sqrt(sum(float(p.detach().double().pow(2).sum().cpu()) for p in params))


def _apply_step(params, unit_direction, radius: float) -> None:
    with torch.no_grad():
        for parameter, direction in zip(params, unit_direction):
            parameter.add_(direction, alpha=-float(radius))


def _undo_step(params, unit_direction, radius: float) -> None:
    with torch.no_grad():
        for parameter, direction in zip(params, unit_direction):
            parameter.add_(direction, alpha=float(radius))


def _parse_float_list(text: str) -> list[float]:
    return [float(x) for x in str(text).split(",") if str(x).strip()]


def _parse_horizons(text: str, K: int) -> list[int]:
    values = {int(x) for x in str(text).split(",") if str(x).strip()}
    values.add(int(K))
    return sorted(x for x in values if 1 <= x <= int(K))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--num_pairs", type=int, default=8)
    parser.add_argument("--finite_pairs", type=int, default=1)
    parser.add_argument("--finite_horizons", default="1,8,16,32")
    parser.add_argument("--relative_radii", default="2.5e-7,5e-7,1e-6")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--calibration_split", choices=["train", "val"], default="train"
    )
    parser.add_argument(
        "--evaluation_split", choices=["val", "test"], default="test"
    )
    parser.add_argument("--loss_type", choices=["auto", "rel_l2", "mse"], default="auto")
    parser.add_argument("--coord_subsample", type=int, default=250_000)
    parser.add_argument("--start_fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    cli = parser.parse_args()

    checkpoint = torch.load(cli.ckpt, map_location="cpu", weights_only=False)
    args = _namespace(checkpoint["args"])
    args.num_workers = 0
    args.local_batch_size = 1
    train_loader, val_loader, test_loader = build_dataloaders(args, rank=0, world_size=1)
    _infer_thewell_shape(args, train_loader)
    model = build_model(args, rank=0)
    device = torch.device(f"cuda:{int(cli.gpu)}" if torch.cuda.is_available() else "cpu")
    load_checkpoint(model, cli.ckpt, map_location=str(device), strict=True)
    raw = unwrap_model(model).to(device).eval()
    recurrent = bool(getattr(raw, "is_recurrent_state_ar", False))
    params = [p for p in raw.parameters() if p.requires_grad]
    projection = GradientProjection(params, cli.coord_subsample, cli.seed)

    if cli.loss_type == "auto":
        candidate = getattr(args, "mamba_loss_type", None) if recurrent else getattr(args, "bptt_loss_type", None)
        loss_type = "rel_l2" if str(candidate).lower() == "rel_l2" else "mse"
    else:
        loss_type = cli.loss_type

    calibration_loader = train_loader if cli.calibration_split == "train" else val_loader
    evaluation_loader = val_loader if cli.evaluation_split == "val" else test_loader
    calibration_pool = _strict_candidate_pool(
        calibration_loader, args, recurrent, cli.K, int(cli.num_pairs), cli.seed,
        cli.start_fraction,
    )
    evaluation_pool = _strict_candidate_pool(
        evaluation_loader, args, recurrent, cli.K, int(cli.num_pairs), cli.seed + 1,
        cli.start_fraction,
    )
    num_pairs = min(int(cli.num_pairs), len(calibration_pool), len(evaluation_pool))
    if num_pairs < 1:
        raise RuntimeError(
            "need at least one feasible calibration/evaluation pair, got "
            f"{len(calibration_pool)}/{len(evaluation_pool)}"
        )
    pairs = list(zip(calibration_pool[:num_pairs], evaluation_pool[:num_pairs]))
    finite_horizons = _parse_horizons(cli.finite_horizons, cli.K)
    relative_radii = _parse_float_list(cli.relative_radii)

    print(
        f"[probe] dataset={getattr(args, 'dataset', '?')} model={getattr(args, 'model_name', '?')} "
        f"K={cli.K} pairs={num_pairs} splits={cli.calibration_split}->{cli.evaluation_split} "
        f"loss={loss_type} "
        f"projection={projection.size}/{projection.total}", flush=True,
    )

    saved_open = _snapshot_open_state(raw)
    _force_fully_open(raw)
    pair_records = []
    metric_rows: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "full_norm_A", "delayed_norm_A", "delayed_fraction_A",
            "full_amplitude_H1", "delayed_amplitude_H1",
            "full_matched_utility", "full_window_utility",
            "delayed_component_reproducibility_cosine",
            "delayed_matched_utility", "delayed_window_utility",
            "full_cross_batch_cosine",
        )
    }
    global_forward_max = 0.0
    try:
        with torch.enable_grad():
            for pair_index, (a, b) in enumerate(pairs):
                print(
                    f"  pair {pair_index + 1}/{num_pairs}: "
                    f"A=batch{a['batch_index']}@{a['start']} "
                    f"B=batch{b['batch_index']}@{b['start']}", flush=True,
                )
                af, al, afl, all_, afd = _full_local_projected(
                    raw, params, projection, a, cli.K, args, recurrent, loss_type, device
                )
                bf, bl, bfl, bll, bfd = _full_local_projected(
                    raw, params, projection, b, cli.K, args, recurrent, loss_type, device
                )
                global_forward_max = max(global_forward_max, afd, bfd)
                ad = [x - y for x, y in zip(af, al)]
                bd = [x - y for x, y in zip(bf, bl)]
                b_window = sum(bf) / float(cli.K)

                full_norm = np.asarray([_projected_norm(x, projection) for x in af])
                delayed_norm = np.asarray([_projected_norm(x, projection) for x in ad])
                local_norm = np.asarray([_projected_norm(x, projection) for x in al])
                full_utility = np.asarray([
                    _projected_dot(x, y, projection) / max(_projected_norm(x, projection), 1e-30)
                    for x, y in zip(af, bf)
                ])
                full_window_utility = np.asarray([
                    _projected_dot(x, b_window, projection) / max(_projected_norm(x, projection), 1e-30)
                    for x in af
                ])
                delayed_matched = np.asarray([
                    _projected_dot(x, y, projection) / max(_projected_norm(x, projection), 1e-30)
                    for x, y in zip(ad, bf)
                ])
                delayed_window = np.asarray([
                    _projected_dot(x, b_window, projection) / max(_projected_norm(x, projection), 1e-30)
                    for x in ad
                ])
                delayed_repro = np.asarray([_cosine(x, y) for x, y in zip(ad, bd)])
                full_cos = np.asarray([_cosine(x, y) for x, y in zip(af, bf)])
                delayed_fraction = delayed_norm / np.maximum(full_norm, 1e-30)
                full_amp = full_norm / max(float(full_norm[0]), 1e-30)
                delayed_amp = delayed_norm / max(float(delayed_norm[0]), 1e-30)

                row_values = {
                    "full_norm_A": full_norm,
                    "delayed_norm_A": delayed_norm,
                    "delayed_fraction_A": delayed_fraction,
                    "full_amplitude_H1": full_amp,
                    "delayed_amplitude_H1": delayed_amp,
                    "full_matched_utility": full_utility,
                    "full_window_utility": full_window_utility,
                    "delayed_component_reproducibility_cosine": delayed_repro,
                    "delayed_matched_utility": delayed_matched,
                    "delayed_window_utility": delayed_window,
                    "full_cross_batch_cosine": full_cos,
                }
                for key, value in row_values.items():
                    metric_rows[key].append(value)
                pair_records.append(
                    {
                        "pair_index": pair_index,
                        "A": {k: a[k] for k in ("batch_index", "start", "metadata")},
                        "B": {k: b[k] for k in ("batch_index", "start", "metadata")},
                        "full_loss_A": afl,
                        "local_loss_A": all_,
                        "full_loss_B": bfl,
                        "local_loss_B": bll,
                        "forward_max_diff": max(afd, bfd),
                        "local_full_loss_max_diff": max(
                            max(abs(x - y) for x, y in zip(afl, all_)),
                            max(abs(x - y) for x, y in zip(bfl, bll)),
                        ),
                        "metrics": {key: value.tolist() for key, value in row_values.items()},
                    }
                )
                del af, al, bf, bl, ad, bd
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        finite_records = []
        parameter_norm = _parameter_norm(params)
        for pair_index, (a, b) in enumerate(pairs[: int(cli.finite_pairs)]):
            baseline = _rollout_losses(raw, b, cli.K, args, recurrent, loss_type, device)
            for horizon in finite_horizons:
                print(f"  finite pair={pair_index} horizon={horizon}", flush=True)
                with torch.enable_grad():
                    af, _, ad, afd = _full_local_tensors_at_horizon(
                        raw, params, a, horizon, cli.K, args, recurrent, loss_type, device
                    )
                    bf, _, _, bfd = _full_local_tensors_at_horizon(
                        raw, params, b, horizon, cli.K, args, recurrent, loss_type, device
                    )
                global_forward_max = max(global_forward_max, afd, bfd)
                direction_norm = _tensor_norm(ad)
                if direction_norm <= 1e-30:
                    continue
                unit = [x / direction_norm for x in ad]
                predicted_matched = _tensor_dot(unit, bf)
                for relative_radius in relative_radii:
                    radius = float(relative_radius) * parameter_norm
                    _apply_step(params, unit, radius)
                    try:
                        after = _rollout_losses(raw, b, cli.K, args, recurrent, loss_type, device)
                    finally:
                        _undo_step(params, unit, radius)
                    restored = _rollout_losses(raw, b, cli.K, args, recurrent, loss_type, device)
                    finite_records.append(
                        {
                            "pair_index": pair_index,
                            "horizon": int(horizon),
                            "relative_radius": float(relative_radius),
                            "radius": radius,
                            "parameter_norm": parameter_norm,
                            "direction": "temporal_delayed_full_minus_final_step_local",
                            "direction_norm_before_normalization": direction_norm,
                            "predicted_matched_utility": predicted_matched,
                            "actual_matched_utility": float(
                                (baseline[horizon - 1] - after[horizon - 1]) / max(radius, 1e-30)
                            ),
                            "actual_window_utility": float(
                                (baseline.mean() - after.mean()) / max(radius, 1e-30)
                            ),
                            "matched_loss_before": float(baseline[horizon - 1]),
                            "matched_loss_after": float(after[horizon - 1]),
                            "window_loss_before": float(baseline.mean()),
                            "window_loss_after": float(after.mean()),
                            "restore_loss_max_diff": float(np.max(np.abs(restored - baseline))),
                            "forward_detach_max_diff": max(afd, bfd),
                        }
                    )
                del af, ad, bf, unit
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        _restore_open_state(raw, saved_open)

    stacked = {key: np.stack(value) for key, value in metric_rows.items()}
    output = {
        "format_version": 1,
        "semantics": {
            "primary": "heldout first-order descent utility, not innovation SNR",
            "delayed_component": "full horizon gradient minus final-step local gradient after detaching its incoming state/history",
            "matched_utility": "dot(unit calibration direction, heldout full gradient of the same horizon)",
            "window_utility": "dot(unit calibration direction, heldout mean full gradient over horizons)",
            "finite_step": "reversible equal-Euclidean-norm SGD-shaped parameter intervention",
        },
        "checkpoint": cli.ckpt,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset": str(getattr(args, "dataset", "")),
        "thewell_dataset_name": str(getattr(args, "thewell_dataset_name", "")),
        "model_name": str(getattr(args, "model_name", "")),
        "calibration_split": cli.calibration_split,
        "evaluation_split": cli.evaluation_split,
        "K": int(cli.K),
        "horizons": list(range(1, int(cli.K) + 1)),
        "num_pairs": num_pairs,
        "shared_relative_start_fraction": float(cli.start_fraction),
        "loss_type": loss_type,
        "probe_graph": "fully_open_exact_checkpoint",
        "gradient_projection": projection.metadata(),
        "projection_scaling": "p/m Horvitz-style scale for sampled-coordinate dot products and norms",
        "forward_detach_max_diff": global_forward_max,
        "pair_records": pair_records,
        "summary": {key: _summary(value) for key, value in stacked.items()},
        "fraction_positive": {
            key: np.mean(value > 0.0, axis=0).tolist()
            for key, value in stacked.items() if "utility" in key
        },
        "finite_step_records": finite_records,
        "finite_horizons": finite_horizons,
        "relative_radii": relative_radii,
        "checkpoint_data_args": {
            "window_size": int(getattr(args, "window_size", 1)),
            "mamba_burnin": int(getattr(args, "mamba_burnin", 0)),
            "thewell_time_subsample": int(getattr(args, "thewell_time_subsample", 1)),
            "thewell_spatial_subsample": int(getattr(args, "thewell_spatial_subsample", 1)),
        },
    }
    out_path = Path(cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    selected = sorted(set(h for h in (1, 2, 4, 8, 16, 24, 32, 48, cli.K) if h <= cli.K))
    print("\n H   delay amp   delay/full   heldout matched U   heldout window U   positive")
    for horizon in selected:
        k = horizon - 1
        print(
            f"{horizon:>2}   {np.median(stacked['delayed_amplitude_H1'][:, k]):>9.3g}   "
            f"{np.median(stacked['delayed_fraction_A'][:, k]):>10.3f}   "
            f"{np.median(stacked['delayed_matched_utility'][:, k]):>17.4g}   "
            f"{np.median(stacked['delayed_window_utility'][:, k]):>16.4g}   "
            f"{np.mean(stacked['delayed_matched_utility'][:, k] > 0):>8.3f}"
        )
    print(f"forward detach max diff={global_forward_max:.3e}")
    print(f"[out] {out_path}")


if __name__ == "__main__":
    main()
