#!/usr/bin/env python
"""Frozen-checkpoint, per-start delayed-gradient dynamics probe.

For the loss incurred at every closed-loop horizon k, this script computes the
fully-open parameter gradient g[t,k] = d ell[t,k] / d theta.  It exports the
quantities that are hidden by the usual seed/start-averaged table:

  * ||g[t,k]|| / ||g[t,1]||                 amplitude per rollout start;
  * cos(g[t,1], g[t,k])                     direction relative to H1;
  * cos(g[t,k-1], g[t,k])                   local direction change;
  * log(||g[t,k]|| / ||g[t,k-1]||)          local expansion/contraction;
  * ||sum_t g[t,k]|| / sum_t ||g[t,k]||     cross-start coherence.

The checkpoint is always probed with a fully-open backward graph, independent
of how it was trained.  Parameter coordinates may be projected to one fixed
random subset; the same coordinates are used for every start and horizon, so
amplitude ratios and cosines remain comparable without retaining enormous
full-gradient vectors.

Both recurrent-state models (official_mamba_state) and windowed AR models
(unet_field and related models) are supported.  Each horizon is recomputed in
an independent graph.  This is slower than retaining one K-step graph, but it
keeps peak memory low enough for high-resolution PDE pilots.
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

from internal_dw.data_utils.state_ops import unpack_batch, zero_external_input_like
from internal_dw.datasets.registry import build_dataloaders
from internal_dw.models.registry import build_model
from internal_dw.utils import load_checkpoint, unwrap_model


def _namespace(saved: Any) -> argparse.Namespace:
    if isinstance(saved, argparse.Namespace):
        return saved
    if isinstance(saved, dict):
        return argparse.Namespace(**saved)
    raise TypeError(f"checkpoint args must be dict/Namespace, got {type(saved)!r}")


def _snapshot_open_state(raw) -> dict[str, Any]:
    names = (
        "resgrad_routing", "resgrad_policy", "resgrad_block_gate",
        "resgrad_ratio_threshold", "resgrad_current_horizon",
        "resgrad_current_total_horizon",
    )
    return {name: getattr(raw, name) for name in names if hasattr(raw, name)}


def _force_fully_open(raw) -> None:
    if hasattr(raw, "resgrad_routing"):
        raw.resgrad_routing = False
    if hasattr(raw, "resgrad_policy"):
        raw.resgrad_policy = "all"
    if hasattr(raw, "resgrad_block_gate"):
        raw.resgrad_block_gate = 1.0
    if hasattr(raw, "set_resgrad_context"):
        raw.set_resgrad_context(None, None)


def _restore_open_state(raw, saved: dict[str, Any]) -> None:
    for name, value in saved.items():
        setattr(raw, name, value)


def _detach_state(raw, state):
    if hasattr(raw, "detach_state"):
        return raw.detach_state(state)
    if torch.is_tensor(state):
        return state.detach()
    if isinstance(state, tuple):
        return tuple(_detach_state(raw, x) for x in state)
    if isinstance(state, list):
        return [_detach_state(raw, x) for x in state]
    return state


def _burn_recurrent(raw, state, stim, start_t: int, burn: int):
    batch = int(state.shape[0])
    with torch.no_grad():
        hidden = raw.init_state(batch, state.device, state.dtype)
        burn_start = max(0, int(start_t) - int(burn))
        burn_end = max(burn_start, int(start_t) - 1)
        for j in range(burn_start, burn_end):
            stim_j = stim[:, j] if stim is not None else None
            _, hidden = raw.step(hidden, state[:, j], stim_j, return_aux=False)
    return _detach_state(raw, hidden)


def _step_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    batch = int(pred.shape[0])
    delta = (pred - target).reshape(batch, -1)
    if loss_type == "mse":
        return delta.pow(2).mean()
    target_flat = target.reshape(batch, -1)
    return (delta.norm(dim=1) / target_flat.norm(dim=1).clamp_min(1e-8)).mean()


class GradientProjection:
    """Fixed coordinate projection without concatenating the full GPU gradient."""

    def __init__(self, params: list[torch.nn.Parameter], size: int, seed: int):
        self.params = params
        self.total = int(sum(p.numel() for p in params))
        requested = int(size)
        self.projected = requested > 0 and requested < self.total
        if not self.projected:
            self.size = self.total
            self.maps = None
            return

        self.size = requested
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        # Sampling with replacement avoids allocating randperm(total) for the
        # very large fMRI checkpoint.  Duplicate probability is negligible at
        # the default projection sizes and is recorded in the JSON metadata.
        global_indices = torch.randint(
            0, self.total, (self.size,), generator=generator, dtype=torch.int64
        )
        self.unique_coordinates = int(torch.unique(global_indices).numel())
        self.maps: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        offset = 0
        for parameter_index, parameter in enumerate(params):
            end = offset + int(parameter.numel())
            mask = (global_indices >= offset) & (global_indices < end)
            output_positions = torch.nonzero(mask, as_tuple=False).flatten()
            if output_positions.numel():
                local_indices = global_indices[output_positions] - offset
                self.maps.append((parameter_index, output_positions, local_indices))
            offset = end

    def apply(self, grads: Iterable[Optional[torch.Tensor]]) -> torch.Tensor:
        grads = list(grads)
        if not self.projected:
            pieces = []
            for grad, parameter in zip(grads, self.params):
                value = grad if grad is not None else torch.zeros_like(parameter)
                pieces.append(value.detach().float().reshape(-1).cpu())
            return torch.cat(pieces)

        output = torch.zeros(self.size, dtype=torch.float32)
        assert self.maps is not None
        for parameter_index, output_positions, local_indices in self.maps:
            grad = grads[parameter_index]
            if grad is None:
                continue
            selected = grad.detach().reshape(-1)[local_indices.to(grad.device)]
            output[output_positions] = selected.float().cpu()
        return output

    def metadata(self) -> dict[str, Any]:
        out = {
            "total_parameter_coordinates": self.total,
            "projected": bool(self.projected),
            "projection_size": self.size,
        }
        if self.projected:
            out["unique_projection_coordinates"] = self.unique_coordinates
            out["sampling"] = "fixed_uniform_with_replacement"
        else:
            out["sampling"] = "all_coordinates"
        return out


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denominator = float(a.norm() * b.norm())
    if not math.isfinite(denominator) or denominator <= 1e-30:
        return 0.0
    return float(torch.dot(a, b) / denominator)


def _infer_thewell_shape(args, train_loader) -> None:
    if str(getattr(args, "dataset", "")) != "the_well":
        return
    sample = train_loader.dataset[0]
    state = sample["state"] if isinstance(sample, dict) else sample[0]
    args.field_channels = int(state.shape[1])
    args.field_height = int(state.shape[2])
    args.field_width = int(state.shape[3])
    if str(getattr(args, "model_name", "")) in {
        "official_mamba_state", "official_pc_mamba_state",
        "official_atlas_mamba_state", "official_tangent_atlas_mamba_state",
        "official_shadow_perturb_mamba", "shadow_perturb_mamba",
        "official_mamba_fixeda_perturb", "cyclic_graph_ar",
    }:
        args.roi_dim = int(state[0].numel())


def _candidate_pool(loader, args, recurrent: bool, K: int, count: int, seed: int):
    candidates = []
    window = int(getattr(args, "window_size", 1))
    burn = max(0, int(getattr(args, "mamba_burnin", 0))) if recurrent else 0
    wanted = max(int(count) * 4, int(count))
    for batch_index, batch in enumerate(loader):
        state, stim, _, metadata = unpack_batch(batch)
        state = state[:1].cpu()
        stim = stim[:1].cpu() if stim is not None else None
        total_time = int(state.shape[1])
        low = max(1, burn) if recurrent else window
        high = total_time - int(K)
        if high < low:
            continue
        number_here = min(max(1, count), high - low + 1)
        starts = np.linspace(low, high, num=number_here).round().astype(int)
        for start in np.unique(starts):
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
    random.Random(int(seed)).shuffle(candidates)
    return candidates[: int(count)]


def _gradient_at_horizon(
    raw,
    params,
    projection: GradientProjection,
    state,
    stim,
    start_t: int,
    horizon: int,
    total_horizon: int,
    args,
    recurrent: bool,
    loss_type: str,
):
    if recurrent:
        burn = max(0, int(getattr(args, "mamba_burnin", 0)))
        hidden = _burn_recurrent(raw, state, stim, start_t, burn)
        x_in = state[:, start_t - 1]
        prediction = None
        for k in range(int(horizon)):
            target_t = int(start_t) + k
            stim_in = stim[:, target_t - 1] if stim is not None else None
            prediction, hidden = raw.step(
                hidden,
                x_in,
                stim_in,
                return_aux=False,
                horizon_index=k,
                total_horizon=total_horizon,
            )
            x_in = prediction
        assert prediction is not None
        target = state[:, int(start_t) + int(horizon) - 1]
        loss = _step_loss(prediction, target, loss_type)
    else:
        window = int(getattr(args, "window_size", 1))
        history = state[:, int(start_t) - window : int(start_t)]
        prediction = None
        for k in range(int(horizon)):
            target_t = int(start_t) + k
            stim_window = (
                stim[:, target_t - window : target_t]
                if stim is not None else None
            )
            if hasattr(raw, "set_resgrad_context"):
                raw.set_resgrad_context(k, total_horizon)
            prediction = raw(stim_window, history, return_aux=False)
            history = torch.cat([history[:, 1:], prediction], dim=1)
        assert prediction is not None
        target = state[:, int(start_t) + int(horizon) - 1 : int(start_t) + int(horizon)]
        loss = _step_loss(prediction, target, loss_type)

    grads = torch.autograd.grad(loss, params, allow_unused=True)
    vector = projection.apply(grads)
    return vector, float(loss.detach().cpu())


def _all_recurrent_horizon_gradients(
    raw,
    params,
    projection: GradientProjection,
    state,
    stim,
    start_t: int,
    K: int,
    args,
    loss_type: str,
):
    """Reuse one K-step recurrent rollout graph for all K residual demands.

    This matches the established recurrent per-step probe and avoids repeating
    O(K^2) forward steps.  The windowed high-resolution field path deliberately
    keeps the independent-graph implementation above because retaining a full
    U-Net rollout graph can dominate even a 100-GB accelerator.
    """
    burn = max(0, int(getattr(args, "mamba_burnin", 0)))
    hidden = _burn_recurrent(raw, state, stim, start_t, burn)
    x_in = state[:, int(start_t) - 1]
    losses = []
    for k in range(int(K)):
        target_t = int(start_t) + k
        stim_in = stim[:, target_t - 1] if stim is not None else None
        prediction, hidden = raw.step(
            hidden,
            x_in,
            stim_in,
            return_aux=False,
            horizon_index=k,
            total_horizon=K,
        )
        losses.append(_step_loss(prediction, state[:, target_t], loss_type))
        x_in = prediction

    vectors = []
    loss_values = []
    for k, loss in enumerate(losses):
        grads = torch.autograd.grad(
            loss, params, retain_graph=(k < int(K) - 1), allow_unused=True
        )
        vectors.append(projection.apply(grads))
        loss_values.append(float(loss.detach().cpu()))
    return vectors, loss_values


def _summary(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "sd": np.std(values, axis=0, ddof=1 if values.shape[0] > 1 else 0).tolist(),
        "median": np.median(values, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--num_starts", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--loss_type", choices=["auto", "rel_l2", "mse"], default="auto")
    parser.add_argument("--coord_subsample", type=int, default=250_000)
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
    params = [parameter for parameter in raw.parameters() if parameter.requires_grad]
    projection = GradientProjection(params, cli.coord_subsample, cli.seed)

    if cli.loss_type == "auto":
        candidate = (
            getattr(args, "mamba_loss_type", None)
            if recurrent else getattr(args, "bptt_loss_type", None)
        )
        loss_type = "rel_l2" if str(candidate).lower() == "rel_l2" else "mse"
    else:
        loss_type = cli.loss_type

    loader = val_loader if cli.split == "val" else test_loader
    pool = _candidate_pool(loader, args, recurrent, cli.K, cli.num_starts, cli.seed)
    if not pool:
        raise RuntimeError(
            f"no feasible {cli.split} starts for K={cli.K}; reduce K or increase sequence length"
        )

    print(
        f"[probe] dataset={getattr(args, 'dataset', '?')} model={getattr(args, 'model_name', '?')} "
        f"recurrent={recurrent} K={cli.K} starts={len(pool)} loss={loss_type} "
        f"projection={projection.size}/{projection.total}",
        flush=True,
    )

    saved_open = _snapshot_open_state(raw)
    _force_fully_open(raw)
    all_norms = []
    all_amp = []
    all_cos_h1 = []
    all_cos_prev = []
    all_log_growth = []
    all_losses = []
    per_start = []
    batch_sums: list[torch.Tensor] | None = None
    norm_sums = np.zeros(cli.K, dtype=np.float64)
    try:
        with torch.enable_grad():
            for start_index, item in enumerate(pool):
                state = item["state"].to(device=device, dtype=torch.float32)
                stim = item["stim"]
                if stim is None:
                    stim = zero_external_input_like(
                        state, int(getattr(args, "stim_dim", 1))
                    )
                else:
                    stim = stim.to(device=device, dtype=torch.float32)

                if recurrent:
                    vectors, losses = _all_recurrent_horizon_gradients(
                        raw, params, projection, state, stim, item["start"],
                        cli.K, args, loss_type,
                    )
                    print(
                        f"  start {start_index + 1}/{len(pool)} "
                        f"t={item['start']} H=1..{cli.K}",
                        flush=True,
                    )
                else:
                    vectors = []
                    losses = []
                    for horizon in range(1, int(cli.K) + 1):
                        vector, loss = _gradient_at_horizon(
                            raw, params, projection, state, stim, item["start"],
                            horizon, cli.K, args, recurrent, loss_type,
                        )
                        vectors.append(vector)
                        losses.append(loss)
                        if horizon == 1 or horizon == cli.K or horizon % 8 == 0:
                            print(
                                f"  start {start_index + 1}/{len(pool)} "
                                f"t={item['start']} H={horizon}/{cli.K}",
                                flush=True,
                            )

                norms = np.asarray([float(vector.norm()) for vector in vectors])
                reference = max(float(norms[0]), 1e-30)
                amplitude = norms / reference
                cosine_h1 = np.asarray([_cosine(vectors[0], vector) for vector in vectors])
                cosine_previous = np.ones(cli.K, dtype=np.float64)
                log_growth = np.zeros(cli.K, dtype=np.float64)
                for k in range(1, cli.K):
                    cosine_previous[k] = _cosine(vectors[k - 1], vectors[k])
                    log_growth[k] = math.log(max(float(norms[k]), 1e-30) / max(float(norms[k - 1]), 1e-30))

                if batch_sums is None:
                    batch_sums = [vector.clone() for vector in vectors]
                else:
                    for k, vector in enumerate(vectors):
                        batch_sums[k] += vector
                norm_sums += norms

                all_norms.append(norms)
                all_amp.append(amplitude)
                all_cos_h1.append(cosine_h1)
                all_cos_prev.append(cosine_previous)
                all_log_growth.append(log_growth)
                all_losses.append(np.asarray(losses))
                per_start.append(
                    {
                        "start_index": start_index,
                        "sequence_batch_index": item["batch_index"],
                        "start_t": item["start"],
                        "gradient_norm": norms.tolist(),
                        "amplitude_relative_h1": amplitude.tolist(),
                        "cosine_h1": cosine_h1.tolist(),
                        "cosine_previous": cosine_previous.tolist(),
                        "local_log_growth": log_growth.tolist(),
                        "loss": losses,
                    }
                )
                del vectors, state, stim
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        _restore_open_state(raw, saved_open)

    assert batch_sums is not None
    norms_array = np.stack(all_norms)
    amp_array = np.stack(all_amp)
    cos_h1_array = np.stack(all_cos_h1)
    cos_prev_array = np.stack(all_cos_prev)
    log_growth_array = np.stack(all_log_growth)
    losses_array = np.stack(all_losses)
    coherence = np.asarray(
        [float(vector.norm()) / max(float(norm_sums[k]), 1e-30) for k, vector in enumerate(batch_sums)]
    )
    batchmean_cos_h1 = np.asarray([_cosine(batch_sums[0], vector) for vector in batch_sums])

    output = {
        "format_version": 1,
        "checkpoint": cli.ckpt,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset": str(getattr(args, "dataset", "")),
        "thewell_dataset_name": str(getattr(args, "thewell_dataset_name", "")),
        "model_name": str(getattr(args, "model_name", "")),
        "probe_graph": "fully_open",
        "recurrent_state_model": recurrent,
        "split": cli.split,
        "K": int(cli.K),
        "horizons": list(range(1, int(cli.K) + 1)),
        "num_starts": len(pool),
        "loss_type": loss_type,
        "gradient_projection": projection.metadata(),
        "per_start": per_start,
        "summary": {
            "gradient_norm": _summary(norms_array),
            "amplitude_relative_h1": _summary(amp_array),
            "cosine_h1": _summary(cos_h1_array),
            "cosine_previous": _summary(cos_prev_array),
            "local_log_growth": _summary(log_growth_array),
            "loss": _summary(losses_array),
            "cross_start_coherence": coherence.tolist(),
            "batchmean_cosine_h1": batchmean_cos_h1.tolist(),
            "coherence_noise_floor": float(1.0 / math.sqrt(len(pool))),
            "fraction_local_contraction": np.mean(log_growth_array[:, 1:] < 0.0, axis=0).tolist(),
            "fraction_direction_reversal": np.mean(cos_prev_array[:, 1:] < 0.0, axis=0).tolist(),
        },
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
    print("\n H   median_amp   median_cos_H1   median_cos_prev   coherence")
    for horizon in selected:
        k = horizon - 1
        print(
            f"{horizon:>2}   {np.median(amp_array[:, k]):>10.3g}   "
            f"{np.median(cos_h1_array[:, k]):>13.4f}   "
            f"{np.median(cos_prev_array[:, k]):>15.4f}   {coherence[k]:>9.4f}"
        )
    print(f"\n[out] {out_path}")


if __name__ == "__main__":
    main()
