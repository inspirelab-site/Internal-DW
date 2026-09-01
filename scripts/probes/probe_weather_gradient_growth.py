#!/usr/bin/env python
"""Paired frozen-checkpoint gradient-growth probe for WeatherBench U-Nets.

The probe answers a narrower question than the Wiener/SNR estimator: does a
loss at a late rollout horizon produce an abnormally amplified backward signal?
It deliberately does *not* interpret cross-start gradient alignment as SNR.

For every requested checkpoint, rollout start, backward arm, and horizon k it
reports

    parameter norm       || d ell_k / d theta ||
    parameter gain       || d ell_k / d theta || / || d ell_k / d xhat_k ||
    history gain         || d ell_k / d H_0 || / || d ell_k / d xhat_k ||

where H_0 is the initial W-frame history and ell_k is the same relative-L2
loss used to train the current WeatherBench U-Net runs.  Dividing by the local
output gradient separates backward amplification from the trivial effect that
late rollout residuals may simply be larger.

Two backward arms are available on the *same* frozen model and forward rollout:

  open    ordinary fully-open BPTT;
  native  the checkpoint's saved routing (c=.90 or learned Dual-Wiener gains).

Routing is backward-only, so their loss curves must agree.  The script checks
this numerically.  Exact-BPTT checkpoints have native == open and are evaluated
only once.

Example (one H200, quick paired screen):

  python scripts/probes/probe_weather_gradient_growth.py \
    --checkpoint exact=experiments/weatherbench/resgrad_unet/unet_c64_D4_W16/ckpt_K48/seed0/best.pth \
    --checkpoint static090=experiments/weatherbench/resgrad_unet/unet_c64_D4_W16/dwc0.90_K48/seed0/best.pth \
    --checkpoint dw=experiments/weatherbench/resgrad_unet/unet_c64_D4_W16/dualwiener_K48/seed0/best.pth \
    --gpu 0 --K 48 --num-starts 2 \
    --out probe_outputs/weather_gradient_growth/three_models.json

Increase --num-starts to 8 only after the quick screen is informative.  The
default selected horizons deliberately include the late DW gain cliff while
keeping the repeated-backward cost manageable.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.utils.checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from internal_dw.data_utils.state_ops import unpack_batch
from internal_dw.datasets.registry import build_dataloaders
from internal_dw.models.registry import build_model
from internal_dw.utils import unwrap_model


FORMAT_VERSION = 1


def _parse_checkpoint(spec: str) -> Tuple[str, str]:
    if "=" not in spec:
        path = str(Path(spec))
        return Path(path).parent.name, path
    label, path = spec.split("=", 1)
    label, path = label.strip(), path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError("--checkpoint must be LABEL=PATH")
    return label, path


def _parse_horizons(spec: str, K: int) -> List[int]:
    if spec.strip().lower() == "auto":
        candidates = [1, 2, 4, 8, 12, 16, 24, 32, 40, 44, 46, 47, 48, K]
    else:
        candidates = [int(x) for x in re.split(r"[\s,]+", spec.strip()) if x]
    out = sorted({h for h in candidates if 1 <= h <= K})
    if not out:
        raise ValueError(f"no requested horizon lies in [1,{K}]")
    return out


def _checkpoint_args(path: str) -> Tuple[argparse.Namespace, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "args" not in checkpoint:
        raise ValueError(f"checkpoint has no saved args: {path}")
    return argparse.Namespace(**checkpoint["args"]), checkpoint


def _load_checkpoint_state(model, checkpoint: dict) -> None:
    """Load an already-open checkpoint without reading the large file twice."""
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


def _compatible_data_args(reference, other, label: str) -> None:
    keys = (
        "dataset", "data_path", "window_size", "field_channels", "field_height",
        "field_width", "wb2_seg_len", "wb2_train_stride", "wb2_eval_stride",
    )
    mismatches = []
    for key in keys:
        a, b = getattr(reference, key, None), getattr(other, key, None)
        if a != b:
            mismatches.append(f"{key}: {a!r} != {b!r}")
    if mismatches:
        raise ValueError(f"checkpoint {label!r} does not share the reference data setup: "
                         + "; ".join(mismatches))


def _collect_paired_starts(loader, K: int, window: int, count: int, seed: int):
    """Collect deterministic CPU-resident (state, stim, start) triples.

    Starts are spread across validation segments rather than taking adjacent
    initializations from one segment.  Every checkpoint reuses this exact pool.
    """
    rng = np.random.default_rng(seed)
    pool = []
    for batch in loader:
        state, stim, _, _ = unpack_batch(batch)
        for bi in range(int(state.shape[0])):
            length = int(state.shape[1])
            lo, hi = int(window), int(length - K)
            if hi < lo:
                continue
            choices = np.arange(lo, hi + 1, dtype=np.int64)
            start = int(rng.choice(choices))
            pool.append((
                state[bi:bi + 1].detach().cpu().contiguous(),
                None if stim is None else stim[bi:bi + 1].detach().cpu().contiguous(),
                start,
            ))
            if len(pool) >= count:
                return pool
    return pool


def _snapshot_routing(raw) -> dict:
    return {
        "resgrad_routing": bool(getattr(raw, "resgrad_routing", False)),
        "resgrad_policy": str(getattr(raw, "resgrad_policy", "all")),
        "resgrad_block_gate": float(getattr(raw, "resgrad_block_gate", 1.0)),
    }


def _configure_arm(raw, snapshot: dict, arm: str) -> None:
    if arm == "open":
        raw.resgrad_routing = False
        raw.resgrad_policy = "all"
        raw.resgrad_block_gate = 1.0
    elif arm == "native":
        raw.resgrad_routing = snapshot["resgrad_routing"]
        raw.resgrad_policy = snapshot["resgrad_policy"]
        raw.resgrad_block_gate = snapshot["resgrad_block_gate"]
    else:
        raise ValueError(f"unknown backward arm {arm!r}")


def _native_is_open(snapshot: dict) -> bool:
    return (not snapshot["resgrad_routing"]
            or snapshot["resgrad_policy"].lower() in ("all", "full", "normal"))


def _window_step(raw, horizon_index, total_horizon, stim_window, history):
    if hasattr(raw, "set_resgrad_context"):
        raw.set_resgrad_context(
            horizon_index=int(horizon_index), total_horizon=int(total_horizon)
        )
    return raw(stim_window, history, return_aux=False)


def _rollout(raw, history, stim, start: int, K: int, window: int,
             use_checkpoint: bool):
    predictions, targets = [], []
    h = history
    for k in range(K):
        target_t = int(start + k)
        stim_window = None if stim is None else stim[:, target_t - window:target_t]
        if use_checkpoint:
            # Match the exact-gradient training option: save activation memory
            # by recomputing each U-Net step during backward.
            pred = torch.utils.checkpoint.checkpoint(
                _window_step,
                raw, int(k), int(K), stim_window, h,
                use_reentrant=False,
            )
        else:
            pred = _window_step(raw, int(k), int(K), stim_window, h)
        if pred.dim() == h.dim() - 1:
            pred = pred.unsqueeze(1)
        x_ar = pred[:, 0]
        predictions.append(x_ar)
        targets.append(_CURRENT_STATE[:, target_t])
        h = torch.cat([h[:, 1:], pred], dim=1)
    return predictions, targets


# Set only while a rollout graph is being built.  Keeping it module-local lets
# _rollout's checkpoint function stay free of a second copy of the large state
# sequence in every recomputation argument list.
_CURRENT_STATE: torch.Tensor


def _step_loss(pred: torch.Tensor, target: torch.Tensor, kind: str,
               channel: Optional[int] = None) -> torch.Tensor:
    if channel is not None:
        pred = pred[:, channel:channel + 1]
        target = target[:, channel:channel + 1]
    batch = int(pred.shape[0])
    delta = (pred - target).reshape(batch, -1)
    if kind == "mse":
        return delta.pow(2).mean()
    target_flat = target.reshape(batch, -1)
    return (delta.norm(dim=1) / target_flat.norm(dim=1).clamp_min(1e-8)).mean()


def _norm_of(grads: Iterable[Optional[torch.Tensor]]) -> float:
    terms = [g.detach().float().pow(2).sum() for g in grads if g is not None]
    if not terms:
        return 0.0
    return float(torch.stack(terms).sum().sqrt().cpu())


def _probe_one_graph(raw, params, state, stim, start: int, K: int,
                     horizons: Sequence[int], loss_kind: str,
                     channel: Optional[int], use_checkpoint: bool):
    global _CURRENT_STATE
    _CURRENT_STATE = state
    window = int(raw.window_size)
    root = state[:, start - window:start].detach().clone().requires_grad_(True)
    predictions, targets = _rollout(
        raw, root, stim, start, K, window, use_checkpoint=use_checkpoint
    )
    losses = [_step_loss(p, t, loss_kind, channel=channel)
              for p, t in zip(predictions, targets)]

    rows = []
    for horizon in horizons:
        index = int(horizon - 1)
        # root + current output + theta in one traversal.  d ell/d xhat is the
        # local output gradient; the other two include the rollout chain.
        grads = torch.autograd.grad(
            losses[index], [root, predictions[index], *params],
            retain_graph=True, create_graph=False, allow_unused=True,
        )
        root_norm = _norm_of(grads[:1])
        local_norm = _norm_of(grads[1:2])
        param_norm = _norm_of(grads[2:])
        denom = max(local_norm, 1e-30)
        rows.append({
            "horizon": int(horizon),
            "loss": float(losses[index].detach().cpu()),
            "local_output_grad_norm": local_norm,
            "parameter_grad_norm": param_norm,
            "parameter_amplification": param_norm / denom,
            "history_grad_norm": root_norm,
            "history_amplification": root_norm / denom,
        })
        del grads

    aggregate = torch.stack(losses).mean()
    aggregate_grads = torch.autograd.grad(
        aggregate, [root, *params], retain_graph=False,
        create_graph=False, allow_unused=True,
    )
    aggregate_row = {
        "loss": float(aggregate.detach().cpu()),
        "history_grad_norm": _norm_of(aggregate_grads[:1]),
        "parameter_grad_norm": _norm_of(aggregate_grads[1:]),
    }
    del aggregate_grads, aggregate, losses, predictions, targets, root
    _CURRENT_STATE = None
    return rows, aggregate_row


def _mean_sd(values: Sequence[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "values": [float(x) for x in arr],
    }


def _persistence_summary(pool, horizons: Sequence[int], loss_kind: str,
                         channel: Optional[int]) -> dict:
    """Paired no-change forecast on the exact starts used by the probe."""
    per_horizon = {}
    for horizon in horizons:
        values = []
        for state, _, start in pool:
            prediction = state[:, start - 1]
            target = state[:, start + int(horizon) - 1]
            values.append(float(_step_loss(
                prediction, target, loss_kind, channel=channel
            )))
        per_horizon[str(horizon)] = {"loss": _mean_sd(values)}
    return {"per_horizon": per_horizon}


def _summarize_arm(samples: list, horizons: Sequence[int]) -> dict:
    metrics = (
        "loss", "local_output_grad_norm", "parameter_grad_norm",
        "parameter_amplification", "history_grad_norm", "history_amplification",
    )
    per_horizon = {}
    for hi, horizon in enumerate(horizons):
        per_horizon[str(horizon)] = {
            key: _mean_sd([sample["per_horizon"][hi][key] for sample in samples])
            for key in metrics
        }
    aggregate = {
        key: _mean_sd([sample["aggregate"][key] for sample in samples])
        for key in ("loss", "history_grad_norm", "parameter_grad_norm")
    }
    return {"per_horizon": per_horizon, "aggregate": aggregate, "samples": samples}


def _profile_summary(arm_summary: dict, horizons: Sequence[int]) -> dict:
    def series(key):
        return np.asarray([
            arm_summary["per_horizon"][str(h)][key]["mean"] for h in horizons
        ], dtype=np.float64)

    pnorm = series("parameter_grad_norm")
    pamp = series("parameter_amplification")
    hamp = series("history_amplification")
    n_edge = min(3, len(horizons))

    def ratio(v):
        early = float(np.median(v[:n_edge]))
        late = float(np.median(v[-n_edge:]))
        return late / max(early, 1e-30)

    return {
        "parameter_norm_late_over_early": ratio(pnorm),
        "parameter_amplification_late_over_early": ratio(pamp),
        "history_amplification_late_over_early": ratio(hamp),
        "parameter_norm_peak_over_h1": float(pnorm.max() / max(pnorm[0], 1e-30)),
        "parameter_amplification_peak_over_h1": float(pamp.max() / max(pamp[0], 1e-30)),
        "history_amplification_peak_over_h1": float(hamp.max() / max(hamp[0], 1e-30)),
        "parameter_norm_peak_horizon": int(horizons[int(np.argmax(pnorm))]),
        "history_amplification_peak_horizon": int(horizons[int(np.argmax(hamp))]),
    }


def _print_model_table(label: str, result: dict, horizons: Sequence[int],
                       persistence: dict) -> None:
    open_ = result["arms"]["open"]
    native = result["arms"]["native"]
    print(f"\n=== {label}: per-horizon relative-L2 gradients ===")
    print("   H    loss/persist   open||g_theta||   open A_theta    open A_hist   "
          "native/open theta  native/open hist")
    for horizon in horizons:
        o = open_["per_horizon"][str(horizon)]
        n = native["per_horizon"][str(horizon)]
        op = o["parameter_grad_norm"]["mean"]
        oh = o["history_grad_norm"]["mean"]
        np_ = n["parameter_grad_norm"]["mean"]
        nh = n["history_grad_norm"]["mean"]
        persistence_loss = persistence["per_horizon"][str(horizon)]["loss"]["mean"]
        print(
            f"{horizon:4d}  {o['loss']['mean'] / max(persistence_loss, 1e-30):12.6e}  "
            f"{op:16.6e}  "
            f"{o['parameter_amplification']['mean']:13.6e}  "
            f"{o['history_amplification']['mean']:12.6e}  "
            f"{np_ / max(op, 1e-30):17.6e}  {nh / max(oh, 1e-30):16.6e}"
        )
    p = result["open_profile"]
    print(
        "open profile: "
        f"late/early ||g_theta||={p['parameter_norm_late_over_early']:.3g}, "
        f"late/early A_theta={p['parameter_amplification_late_over_early']:.3g}, "
        f"late/early A_hist={p['history_amplification_late_over_early']:.3g}; "
        f"peak/H1 A_hist={p['history_amplification_peak_over_h1']:.3g} "
        f"at H{p['history_amplification_peak_horizon']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", action="append", required=True, metavar="LABEL=PATH",
        help="repeat for exact, static, and DW checkpoints",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--K", type=int, default=48)
    parser.add_argument(
        "--horizons", default="auto",
        help="comma/space list of one-based horizons, or 'auto'",
    )
    parser.add_argument("--num-starts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--loss", choices=("rel_l2", "mse"), default="rel_l2")
    parser.add_argument(
        "--channel", type=int, default=-1,
        help="-1 probes the joint training loss; 0/1/2 isolates one Weather channel",
    )
    parser.add_argument(
        "--params", default=".",
        help="regex over parameter names; default is every trainable model parameter",
    )
    parser.add_argument(
        "--no-checkpoint", action="store_true",
        help="retain all rollout activations instead of recomputing during backward",
    )
    parser.add_argument(
        "--out", default="probe_outputs/weather_gradient_growth/three_models.json"
    )
    cli = parser.parse_args()

    if cli.K <= 0 or cli.num_starts <= 0:
        raise ValueError("--K and --num-starts must be positive")
    horizons = _parse_horizons(cli.horizons, cli.K)
    checkpoints = [_parse_checkpoint(spec) for spec in cli.checkpoint]
    labels = [label for label, _ in checkpoints]
    if len(labels) != len(set(labels)):
        raise ValueError(f"duplicate checkpoint labels: {labels}")
    for label, path in checkpoints:
        if not Path(path).is_file():
            raise FileNotFoundError(f"{label}: {path}")

    if not torch.cuda.is_available():
        raise RuntimeError("the Weather U-Net gradient probe requires CUDA")
    torch.cuda.set_device(cli.gpu)
    device = torch.device(f"cuda:{cli.gpu}")
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    reference_args, _ = _checkpoint_args(checkpoints[0][1])
    if str(getattr(reference_args, "dataset", "")) != "weatherbench2":
        raise ValueError("reference checkpoint is not WeatherBench-2")
    if str(getattr(reference_args, "model_name", "")) != "unet_field":
        raise ValueError("this probe currently targets model_name=unet_field")
    window = int(getattr(reference_args, "window_size", 0))
    if window <= 0:
        raise ValueError(f"invalid window_size={window}")
    for label, path in checkpoints[1:]:
        args_i, _ = _checkpoint_args(path)
        _compatible_data_args(reference_args, args_i, label)

    # Data is built once and retained on CPU.  This guarantees paired starts
    # even if a checkpoint saved a different training batch size.
    reference_args.num_workers = 0
    reference_args.local_batch_size = 1
    train_loader, val_loader, test_loader = build_dataloaders(
        reference_args, rank=0, world_size=1
    )
    del train_loader
    loader = val_loader if cli.split == "val" else test_loader
    pool = _collect_paired_starts(
        loader, cli.K, window, cli.num_starts, cli.seed
    )
    del val_loader, test_loader, loader
    if len(pool) < cli.num_starts:
        raise RuntimeError(
            f"only found {len(pool)} feasible starts for K={cli.K}, "
            f"requested {cli.num_starts}"
        )
    print(
        f"paired WeatherBench pool: split={cli.split}, starts={len(pool)}, "
        f"K={cli.K}, horizons={horizons}, window={window}", flush=True
    )
    persistence = _persistence_summary(
        pool, horizons, cli.loss, None if cli.channel < 0 else cli.channel
    )

    output = {
        "format_version": FORMAT_VERSION,
        "K": int(cli.K),
        "horizons": horizons,
        "num_starts": len(pool),
        "seed": int(cli.seed),
        "split": cli.split,
        "loss": cli.loss,
        "channel": int(cli.channel),
        "gradient_checkpointing": not cli.no_checkpoint,
        "persistence": persistence,
        "models": {},
    }

    for label, path in checkpoints:
        started = time.time()
        args_i, checkpoint = _checkpoint_args(path)
        # Do not let a launch-shell constant-gain or corrected-estimator env var
        # silently alter the frozen checkpoint being diagnosed.
        os.environ.pop("DUAL_WIENER_CONST", None)
        os.environ.pop("DUAL_WIENER_INNOVATION_FILE", None)
        os.environ.pop("DUAL_WIENER_INNOVATION_KEY", None)

        saved_epoch = checkpoint.get("epoch", -1)
        model = build_model(args_i, rank=cli.gpu)
        _load_checkpoint_state(model, checkpoint)
        raw = unwrap_model(model).to(device).eval()
        snapshot = _snapshot_routing(raw)
        named = [(name, p) for name, p in raw.named_parameters()
                 if p.requires_grad and re.search(cli.params, name)]
        if not named:
            raise ValueError(f"--params {cli.params!r} matched no parameters")
        params = [p for _, p in named]
        n_params = sum(int(p.numel()) for p in params)
        print(
            f"\n[{label}] epoch={checkpoint.get('epoch', '?')} "
            f"native={snapshot['resgrad_policy']} routing={snapshot['resgrad_routing']} "
            f"params={n_params:,}", flush=True
        )
        del checkpoint

        evaluated_arms = ["open"] if _native_is_open(snapshot) else ["open", "native"]
        arm_samples: Dict[str, list] = {}
        forward_losses: Dict[str, list] = {}
        for arm in evaluated_arms:
            _configure_arm(raw, snapshot, arm)
            samples = []
            for sample_index, (state_cpu, stim_cpu, start) in enumerate(pool):
                state = state_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
                stim = (None if stim_cpu is None else
                        stim_cpu.to(device=device, dtype=torch.float32, non_blocking=True))
                with torch.enable_grad():
                    rows, aggregate = _probe_one_graph(
                        raw, params, state, stim, start, cli.K, horizons,
                        cli.loss, None if cli.channel < 0 else cli.channel,
                        use_checkpoint=not cli.no_checkpoint,
                    )
                samples.append({
                    "sample_index": int(sample_index), "start": int(start),
                    "per_horizon": rows, "aggregate": aggregate,
                })
                del state, stim
                torch.cuda.empty_cache()
                print(
                    f"[{label}/{arm}] {sample_index + 1}/{len(pool)} done "
                    f"({time.time() - started:.1f}s)", flush=True
                )
            arm_samples[arm] = samples
            forward_losses[arm] = [[row["loss"] for row in s["per_horizon"]]
                                   for s in samples]

        native_alias = _native_is_open(snapshot)
        if native_alias:
            arm_samples["native"] = copy.deepcopy(arm_samples["open"])
            forward_losses["native"] = copy.deepcopy(forward_losses["open"])
        max_forward_deviation = float(np.max(np.abs(
            np.asarray(forward_losses["native"], dtype=np.float64)
            - np.asarray(forward_losses["open"], dtype=np.float64)
        )))
        if max_forward_deviation > 1e-6:
            raise RuntimeError(
                f"{label}: native/open loss deviation {max_forward_deviation:.3e}; "
                "routing is not forward-identical"
            )

        arms = {arm: _summarize_arm(arm_samples[arm], horizons)
                for arm in ("open", "native")}
        result = {
            "checkpoint": path,
            "checkpoint_epoch": int(saved_epoch) if saved_epoch is not None else -1,
            "saved_routing": snapshot,
            "native_aliases_open": bool(native_alias),
            "max_native_open_loss_deviation": max_forward_deviation,
            "parameter_pattern": cli.params,
            "parameter_count": int(n_params),
            "arms": arms,
            "open_profile": _profile_summary(arms["open"], horizons),
            "elapsed_seconds": float(time.time() - started),
        }
        output["models"][label] = result
        _print_model_table(label, result, horizons, persistence)

        del raw, model, params, named, arm_samples
        gc.collect()
        torch.cuda.empty_cache()

    out_path = Path(cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
    print(f"\n[out] {out_path}")
    print(
        "Interpretation: growth of open A_hist or A_theta diagnoses backward "
        "amplification. Growth of raw ||g_theta|| alone may only reflect larger "
        "late-horizon residuals. This probe does not estimate gradient SNR."
    )


if __name__ == "__main__":
    main()
