#!/usr/bin/env python3
"""Unified dense-horizon, multi-origin relative-L2 evaluation.

This script is evaluation-only.  It loads the arguments and weights saved in
an existing checkpoint and evaluates a genuine lead-time curve:

    E_h = mean_unit mean_origin ||y_hat(t+h|t)-y(t+h)||_2 / ||y(t+h)||_2.

Unlike ``test/horizon_H_rel_l2`` in the legacy evaluator, ``h`` is a forecast
lead, not the period of a teacher-forcing refresh.  Every origin is rolled out
once to K, so all K lead times are obtained in the same pass.  Origins are
deduplicated within a physical test unit (subject, trajectory, or file) before
the unit is averaged; this prevents overlapping The-Well windows from giving a
long trajectory extra weight.

WeatherBench-2 has a purpose-built continuous-time evaluator and is handled by
``evaluate_weatherbench2_acc.py``.  This script targets the assigned vector
models and The-Well U-Net checkpoints.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from internal_dw.data_utils.state_ops import (  # noqa: E402
    get_batch_time_shape,
    unpack_batch,
    zero_external_input_like,
)
from internal_dw.datasets import build_dataloaders  # noqa: E402
from internal_dw.datasets.registry import (  # noqa: E402
    dataset_evaluator_name,
    dataset_has_external_input,
    dataset_task_type,
)
from internal_dw.models import build_model  # noqa: E402
from internal_dw.utils import seed_everything, unwrap_model  # noqa: E402


def select_origins(
    length: int,
    window: int,
    horizon: int,
    stride: int = 1,
    maximum: int = 64,
) -> np.ndarray:
    """Return deterministic, evenly spread target indices for lead one."""
    length, window, horizon = int(length), int(window), int(horizon)
    if min(window, horizon, stride) <= 0:
        raise ValueError("window, horizon, and stride must be positive")
    candidates = np.arange(window, length - horizon + 1, int(stride), dtype=np.int64)
    if maximum > 0 and candidates.size > int(maximum):
        # Even coverage is preferable to taking only the beginning of a movie
        # or recording.  Rounding can duplicate an index only in degenerate
        # cases; unique preserves chronological order.
        positions = np.rint(np.linspace(0, candidates.size - 1, int(maximum))).astype(np.int64)
        candidates = candidates[np.unique(positions)]
    return candidates


def _state_dict(checkpoint: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    for key in ("model", "state_dict", "model_state_dict"):
        if key in checkpoint:
            state = checkpoint[key]
            break
    else:
        state = checkpoint
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint does not contain a model state dictionary")
    return {
        (name[len("module."):] if str(name).startswith("module.") else str(name)): value
        for name, value in state.items()
        if name != "_metadata"
    }


def _metadata_at(metadata: Mapping[str, Any], key: str, index: int, default=None):
    if key not in metadata:
        return default
    value = metadata[key]
    if torch.is_tensor(value):
        item = value[index]
        return item.item() if item.numel() == 1 else item.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        item = value[index]
        return item.item() if np.asarray(item).ndim == 0 else item.tolist()
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def unit_descriptors(metadata: Mapping[str, Any], labels, batch_size: int, offset: int):
    """Return (unit key, segment base, time subsample) for each batch item."""
    descriptors = []
    for index in range(int(batch_size)):
        subject = _metadata_at(metadata, "subject_id", index)
        path = _metadata_at(metadata, "path", index)
        trajectory = _metadata_at(metadata, "trajectory_index", index)
        if subject is not None:
            key = f"subject:{subject}"
        elif path is not None and trajectory is not None:
            key = f"trajectory:{path}::{trajectory}"
        elif labels is not None:
            if torch.is_tensor(labels):
                label = labels[index].item()
            elif isinstance(labels, (list, tuple, np.ndarray)):
                label = labels[index]
            else:
                label = labels
            key = f"item:{label}"
        else:
            key = f"item:{offset + index}"
        base = int(_metadata_at(metadata, "start", index, 0) or 0)
        subsample = int(_metadata_at(metadata, "time_subsample", index, 1) or 1)
        descriptors.append((key, base, subsample))
    return descriptors


def _repeat_start_major(frames: Sequence[torch.Tensor]) -> torch.Tensor:
    """[S] of [B,...] -> [S*B,...], with each start retaining all B units."""
    return torch.stack(list(frames), dim=0).flatten(0, 1)


def _step_caller(raw):
    parameters = inspect.signature(raw.step).parameters
    accepts_horizon = "horizon_index" in parameters

    def call(recurrent, frame, stimulus, lead):
        kwargs = {"return_aux": False}
        if accepts_horizon:
            kwargs["horizon_index"] = int(lead)
        result = raw.step(recurrent, frame, stimulus, **kwargs)
        if not isinstance(result, (tuple, list)) or len(result) < 2:
            raise TypeError("recurrent step must return (prediction, state)")
        return result[0], result[1]

    return call


def recurrent_origin_chunk(raw, state, stim, origins: Sequence[int], window: int, K: int):
    """Return per-origin/per-unit relative-L2, shape [S,B,K]."""
    origins = [int(value) for value in origins]
    B = int(state.shape[0])
    S = len(origins)
    call_step = _step_caller(raw)
    recurrent = raw.init_state(S * B, state.device, state.dtype)

    # Context is x[t-W],...,x[t-1].  As in the official evaluator, the first
    # W-1 frames initialize the recurrent state and x[t-1] is the first input.
    for offset in range(max(int(window) - 1, 0)):
        frame = _repeat_start_major(
            [state[:, origin - window + offset] for origin in origins]
        )
        stimulus = _repeat_start_major(
            [stim[:, origin - window + offset] for origin in origins]
        )
        _prediction, recurrent = call_step(recurrent, frame, stimulus, 0)

    x_in = _repeat_start_major([state[:, origin - 1] for origin in origins])
    values = torch.empty(S, B, int(K), dtype=torch.float64, device="cpu")
    for offset in range(int(K)):
        stimulus = _repeat_start_major(
            [stim[:, origin + offset - 1] for origin in origins]
        )
        prediction, recurrent = call_step(recurrent, x_in, stimulus, offset + 1)
        target = _repeat_start_major([state[:, origin + offset] for origin in origins])
        error = (prediction - target).reshape(S, B, -1).norm(dim=-1)
        denominator = target.reshape(S, B, -1).norm(dim=-1).clamp_min(1e-8)
        values[:, :, offset] = (error / denominator).double().cpu()
        x_in = prediction
    return values.numpy()


def _forward_caller(raw):
    parameters = inspect.signature(raw.forward).parameters
    accepts_horizon = "horizon_index" in parameters
    accepts_total = "total_horizon" in parameters

    def call(stimulus_window, history, lead, K):
        kwargs = {"return_aux": False}
        if accepts_horizon:
            kwargs["horizon_index"] = int(lead)
        if accepts_total:
            kwargs["total_horizon"] = int(K)
        return raw(stimulus_window, history, **kwargs)

    return call


def standard_origin_chunk(raw, state, stim, origins: Sequence[int], window: int, K: int):
    """Dense multi-origin rollout for windowed vector/field models."""
    origins = [int(value) for value in origins]
    B = int(state.shape[0])
    S = len(origins)
    call_forward = _forward_caller(raw)
    history = _repeat_start_major(
        [state[:, origin - window:origin] for origin in origins]
    )
    values = torch.empty(S, B, int(K), dtype=torch.float64, device="cpu")
    for offset in range(int(K)):
        stimulus_window = _repeat_start_major(
            [stim[:, origin + offset - window:origin + offset] for origin in origins]
        )
        prediction = call_forward(stimulus_window, history, offset, K)
        if prediction.dim() == history.dim():
            prediction = prediction[:, 0]
        target = _repeat_start_major([state[:, origin + offset] for origin in origins])
        error = (prediction - target).reshape(S, B, -1).norm(dim=-1)
        denominator = target.reshape(S, B, -1).norm(dim=-1).clamp_min(1e-8)
        values[:, :, offset] = (error / denominator).double().cpu()
        history = torch.cat([history[:, 1:], prediction.unsqueeze(1)], dim=1)
    return values.numpy()


def _bootstrap_interval(values: np.ndarray, draws: int, seed: int = 0):
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1 or draws <= 0:
        return [None, None]
    rng = np.random.default_rng(int(seed))
    sample = rng.integers(0, values.size, size=(int(draws), values.size))
    estimates = values[sample].mean(axis=1)
    return [float(x) for x in np.quantile(estimates, [0.025, 0.975])]


def summarize_unit_origins(
    unit_origins: Mapping[str, Mapping[int, np.ndarray]],
    eval_horizon: int,
    train_horizon: int,
    bootstrap_draws: int,
) -> Dict[str, Any]:
    if not unit_origins:
        raise RuntimeError("evaluation produced no valid origins")
    unit_keys = sorted(unit_origins)
    unit_curves = []
    for key in unit_keys:
        origins = unit_origins[key]
        curve = np.stack([origins[index] for index in sorted(origins)], axis=0).mean(axis=0)
        unit_curves.append(curve)
    curves = np.stack(unit_curves, axis=0)
    mean_curve = curves.mean(axis=0)
    if not 0 < int(train_horizon) <= int(eval_horizon):
        raise ValueError(
            f"expected 0 < train_horizon <= eval_horizon, got "
            f"{train_horizon} and {eval_horizon}"
        )
    sd_curve = (
        curves.std(axis=0, ddof=1)
        if len(curves) > 1
        else np.full(eval_horizon, np.nan)
    )
    long_start = int(math.ceil(eval_horizon / 2.0))
    unit_all = curves.mean(axis=1)
    unit_in = curves[:, :train_horizon].mean(axis=1)
    unit_out = (
        curves[:, train_horizon:].mean(axis=1)
        if eval_horizon > train_horizon
        else None
    )
    unit_long = curves[:, long_start - 1:].mean(axis=1)
    unit_final = curves[:, -1]

    def metric(values):
        values = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(values.mean()),
            "sd_across_units": float(values.std(ddof=1)) if values.size > 1 else None,
            "cluster_bootstrap_95ci": _bootstrap_interval(values, bootstrap_draws),
        }

    return {
        "num_units": int(len(unit_keys)),
        "num_unique_origins": int(sum(len(value) for value in unit_origins.values())),
        "per_horizon": {
            "horizons": list(range(1, int(eval_horizon) + 1)),
            "mean_rel_l2": [float(x) for x in mean_curve],
            "sd_across_units": [
                float(x) if np.isfinite(x) else None for x in sd_curve
            ],
        },
        "summary": {
            "all_horizons": metric(unit_all),
            "in_horizon": {
                "start_horizon": 1,
                "end_horizon": int(train_horizon),
                **metric(unit_in),
            },
            "out_of_horizon": (
                {
                    "start_horizon": int(train_horizon) + 1,
                    "end_horizon": int(eval_horizon),
                    **metric(unit_out),
                }
                if unit_out is not None
                else None
            ),
            "long_half": {"start_horizon": long_start, **metric(unit_long)},
            "final_horizon": {"horizon": int(eval_horizon), **metric(unit_final)},
        },
    }


def build_public_result(
    aggregate: Mapping[str, Any],
    *,
    checkpoint_path: str,
    checkpoint_epoch: int,
    dataset: str,
    model_name: str,
    method: str,
    seed: int,
    split: str,
    train_horizon: int,
    eval_horizon: int,
) -> Dict[str, Any]:
    """Build the stable, reader-facing test-result schema.

    Execution diagnostics and per-unit curves intentionally stay out of this
    file. The aggregate curve and uncertainty are sufficient to reproduce the
    paper metric and plots.
    """
    primary = aggregate["summary"]["all_horizons"]
    return {
        "format_version": 3,
        "status": "complete",
        "dataset": str(dataset),
        "method": str(method),
        "seed": int(seed),
        "split": str(split),
        "primary_metric": {
            "name": "mean_relative_l2",
            "value": float(primary["mean"]),
            "horizons": f"1:{int(eval_horizon)}",
            "lower_is_better": True,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "epoch": int(checkpoint_epoch),
        },
        "train_horizon": int(train_horizon),
        "eval_horizon": int(eval_horizon),
        "summary": aggregate["summary"],
        "per_horizon": aggregate["per_horizon"],
        "evaluation": {
            "model": str(model_name),
            "num_units": int(aggregate["num_units"]),
            "num_unique_origins": int(aggregate["num_unique_origins"]),
        },
        "protocol": {
            "quantity": "lead-specific free-rollout relative L2",
            "aggregation": (
                "origins within physical unit, then equal-weight mean across units"
            ),
            "same_rollout_supplies_all_horizons": True,
            "in_horizon": f"1:{int(train_horizon)}",
            "out_of_horizon": (
                f"{int(train_horizon) + 1}:{int(eval_horizon)}"
                if int(eval_horizon) > int(train_horizon)
                else None
            ),
        },
    }


def _prepare_args(checkpoint: Mapping[str, Any], cli) -> argparse.Namespace:
    if "args" not in checkpoint:
        raise KeyError(f"checkpoint has no saved args: {cli.ckpt}")
    args = argparse.Namespace(**dict(checkpoint["args"]))
    args.local_batch_size = int(cli.batch_size or getattr(args, "local_batch_size", 1))
    args.num_workers = int(cli.num_workers)
    args.dataset_has_external_input = dataset_has_external_input(args.dataset)
    args.dataset_task_type = dataset_task_type(args.dataset)
    args.dataset_evaluator_name = dataset_evaluator_name(args.dataset)
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--max-horizon", type=int, required=True)
    parser.add_argument(
        "--train-horizon", type=int, default=0,
        help="training/BPTT horizon K; 0 treats max-horizon as K for backward compatibility",
    )
    parser.add_argument("--origin-stride", type=int, default=1)
    parser.add_argument("--max-origins-per-item", type=int, default=64,
                        help="0 uses every legal origin in every dataset item")
    parser.add_argument("--origin-batch", type=int, default=8,
                        help="number of rollout origins packed into one model batch")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="0 reuses the checkpoint's evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument(
        "--method-label",
        default="",
        help="reader-facing method name stored in the result JSON",
    )
    cli = parser.parse_args()
    if min(cli.max_horizon, cli.origin_stride, cli.origin_batch) <= 0:
        parser.error("max-horizon, origin-stride, and origin-batch must be positive")
    train_horizon = int(cli.train_horizon or cli.max_horizon)
    if train_horizon <= 0 or train_horizon > cli.max_horizon:
        parser.error("train-horizon must satisfy 0 < train-horizon <= max-horizon")
    if not torch.cuda.is_available():
        raise RuntimeError("dense multi-origin checkpoint evaluation requires CUDA")

    torch.cuda.set_device(cli.gpu)
    device = torch.device(f"cuda:{cli.gpu}")
    seed_everything(0)
    # Frozen evaluation must not inherit a shell-side routing intervention or
    # require a training-only prior artifact.  Learned coefficients are in the
    # checkpoint and routing changes backward values only.
    for name in (
        "DUAL_WIENER_CONST", "DUAL_WIENER_INNOVATION_FILE",
        "DUAL_WIENER_INNOVATION_KEY", "DUAL_WIENER_SPECTRUM_OAS",
        "RESGRAD_ALPHA_PERIOD", "RESGRAD_ALPHA_VALUE", "RESGRAD_ORACLE",
    ):
        os.environ.pop(name, None)

    checkpoint = torch.load(cli.ckpt, map_location="cpu", weights_only=False)
    args = _prepare_args(checkpoint, cli)
    if str(args.dataset) == "weatherbench2":
        raise ValueError(
            "use scripts/evaluate/evaluate_weatherbench2_acc.py for the continuous WB2 timeline"
        )
    train_loader, val_loader, test_loader = build_dataloaders(args, rank=0, world_size=1)
    loader = test_loader if cli.split == "test" else val_loader

    # Main infers field dimensions from the first training example before model
    # construction.  Reproduce that path for old The-Well checkpoints.
    if str(args.dataset) == "the_well":
        sample = train_loader.dataset[0]
        state0 = sample["state"] if isinstance(sample, dict) else sample[0]
        args.field_channels = int(state0.shape[1])
        args.field_height = int(state0.shape[2])
        args.field_width = int(state0.shape[3])
        if str(getattr(args, "model_name", "")) != "unet_field":
            args.roi_dim = int(state0[0].numel())

    model = build_model(args, rank=cli.gpu)
    missing, unexpected = unwrap_model(model).load_state_dict(_state_dict(checkpoint), strict=True)
    if missing or unexpected:
        raise RuntimeError(f"strict checkpoint load mismatch: missing={missing}, unexpected={unexpected}")
    raw = unwrap_model(model).eval()
    epoch = int(checkpoint.get("epoch", -1))
    del checkpoint, model

    recurrent = bool(getattr(raw, "is_official_state_mamba", False))
    if not recurrent and not bool(getattr(raw, "is_standard_autoregressive", False)):
        raise TypeError(
            f"unsupported assigned model {type(raw).__name__}: expected recurrent state Mamba or standard AR"
        )

    K = int(cli.max_horizon)
    window = int(getattr(args, "window_size", 0))
    if window <= 0:
        raise ValueError(f"invalid checkpoint window_size={window}")
    unit_origins: MutableMapping[str, MutableMapping[int, np.ndarray]] = defaultdict(dict)
    item_offset = 0
    start_time = time.time()

    print(
        f"[dense-eval] dataset={args.dataset}; model={args.model_name}; epoch={epoch}; "
        f"split={cli.split}; train_K={train_horizon}; eval_H={K}; "
        f"window={window}; recurrent={recurrent}; "
        f"max_origins/item={cli.max_origins_per_item}; origin_batch={cli.origin_batch}",
        flush=True,
    )
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            state, stim, labels, metadata = unpack_batch(batch)
            state = state.to(device=device, dtype=torch.float32, non_blocking=True)
            if stim is None:
                stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))
            stim = stim.to(device=device, dtype=torch.float32, non_blocking=True)
            B, T, _shape = get_batch_time_shape(state)
            origins = select_origins(
                T, window, K, stride=cli.origin_stride,
                maximum=cli.max_origins_per_item,
            )
            if origins.size == 0:
                item_offset += B
                continue
            descriptors = unit_descriptors(metadata, labels, B, item_offset)
            evaluator = recurrent_origin_chunk if recurrent else standard_origin_chunk
            for low in range(0, origins.size, cli.origin_batch):
                selected = origins[low:low + cli.origin_batch]
                curves = evaluator(raw, state, stim, selected, window, K)
                for start_index, local_origin in enumerate(selected.tolist()):
                    for item_index, (key, base, subsample) in enumerate(descriptors):
                        absolute_origin = int(base + int(local_origin) * subsample)
                        curve = np.asarray(curves[start_index, item_index], dtype=np.float64)
                        previous = unit_origins[key].get(absolute_origin)
                        if previous is None:
                            unit_origins[key][absolute_origin] = curve
            item_offset += B
            if cli.progress_every > 0 and (
                (batch_index + 1) % cli.progress_every == 0
                or batch_index + 1 == len(loader)
            ):
                elapsed = time.time() - start_time
                print(
                    f"[dense-eval] batches={batch_index + 1}/{len(loader)}; "
                    f"units={len(unit_origins)}; unique_origins="
                    f"{sum(len(value) for value in unit_origins.values())}; "
                    f"elapsed={elapsed / 60:.1f} min",
                    flush=True,
                )

    aggregate = summarize_unit_origins(
        unit_origins, K, train_horizon, cli.bootstrap_draws
    )
    method_label = cli.method_label.strip() or "unspecified"
    result = build_public_result(
        aggregate,
        checkpoint_path=str(cli.ckpt),
        checkpoint_epoch=epoch,
        dataset=str(args.dataset),
        model_name=str(args.model_name),
        method=method_label,
        seed=int(getattr(args, "seed", -1)),
        split=str(cli.split),
        train_horizon=train_horizon,
        eval_horizon=K,
    )
    output = Path(cli.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== dense multi-origin relative L2 ===")
    out_summary = result["summary"]["out_of_horizon"]
    out_text = (
        f"OOD {train_horizon + 1}:{K}={out_summary['mean']:.6f}; "
        if out_summary is not None
        else "OOD=n/a; "
    )
    print(
        f"IN 1:{train_horizon}={result['summary']['in_horizon']['mean']:.6f}; "
        f"{out_text}"
        f"ALL 1:{K}={result['summary']['all_horizons']['mean']:.6f}; "
        f"H{K}={result['summary']['final_horizon']['mean']:.6f}; "
        f"units={result['num_units']}; origins={result['num_unique_origins']}"
    )
    print(f"[out] {output}")


if __name__ == "__main__":
    main()
