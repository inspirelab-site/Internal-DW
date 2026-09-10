"""Standard autoregressive one-step losses for non-Koopman baselines."""

from __future__ import annotations

import random
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.utils.checkpoint

from internal_dw.data_utils.state_ops import (
    corrcoef_flat,
    elementwise_state_loss,
    get_batch_time_shape,
    time_point,
    time_window,
    append_time_point,
    zero_external_input_like,
)
from internal_dw.evaluation.metrics import relative_l2
from internal_dw.utils import unwrap_model


def _detach_recurrent_state(h):
    """Detach tensor/tuple/list recurrent states recursively.

    Official-state Mamba uses nested tuples of (conv_state, ssm_state), while
    older prototypes used a single tensor.
    """
    if torch.is_tensor(h):
        return h.detach()
    if isinstance(h, tuple):
        return tuple(_detach_recurrent_state(x) for x in h)
    if isinstance(h, list):
        return [_detach_recurrent_state(x) for x in h]
    if isinstance(h, dict):
        return {k: _detach_recurrent_state(v) for k, v in h.items()}
    return h


def _forward_jacobian_fd_penalty(
    x: torch.Tensor,
    forward_fn,
    *,
    eps: float,
    target: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random-direction finite-difference penalty on forward expansion.

    This is a forward-Jacobian baseline, not a backward routing operator. For
    one Rademacher direction per sample it estimates ``J(x)v`` with two
    ordinary forwards and penalizes only expansion above ``target``. The
    perturbation is scaled by each sample's RMS so one epsilon works for both
    normalized vectors and spatial fields. Finite differences avoid
    second-order autograd and retain gradients with respect to model parameters.
    """
    if eps <= 0.0:
        raise ValueError(f"forward_jacobian_eps must be positive, got {eps}")
    if target < 0.0:
        raise ValueError(
            f"forward_jacobian_target must be non-negative, got {target}"
        )

    v = torch.empty_like(x).bernoulli_(0.5).mul_(2.0).sub_(1.0)
    reduce_dims = tuple(range(1, x.dim()))
    x_scale = x.detach().float().pow(2).mean(dim=reduce_dims, keepdim=True).sqrt()
    x_scale = x_scale.clamp_min(1e-3).to(dtype=x.dtype)
    delta = float(eps) * x_scale * v

    y_plus = forward_fn(x + delta)
    y_minus = forward_fn(x - delta)
    if isinstance(y_plus, (tuple, list)):
        y_plus = y_plus[0]
    if isinstance(y_minus, (tuple, list)):
        y_minus = y_minus[0]

    out_scale = x_scale.reshape(x_scale.shape[0], *([1] * (y_plus.dim() - 1)))
    jv = (y_plus - y_minus) / (2.0 * float(eps) * out_scale)
    jv_rms = jv.float().reshape(jv.shape[0], -1).pow(2).mean(dim=1).sqrt()
    penalty = torch.relu(jv_rms - float(target)).pow(2).mean()
    return penalty, jv_rms.detach().mean()


def _reduce_metric_scalars(values, reduction: str = "mean", default: float = 0.0) -> float:
    """Reduce detached diagnostic scalars with at most one device sync.

    Recurrent rollouts may emit thousands of scalar diagnostics per minibatch.
    Calling ``.cpu()`` when each scalar is produced serializes the CUDA stream
    thousands of times.  Keep them on-device and transfer only the final
    reduction used for logging.  This helper never participates in the loss or
    gradient graph.
    """
    if not values:
        return float(default)
    tensor_value = next((value for value in values if torch.is_tensor(value)), None)
    if tensor_value is None:
        if reduction == "min":
            return float(min(values))
        if reduction == "max":
            return float(max(values))
        return float(sum(values) / len(values))
    device = tensor_value.device
    scalars = [
        value.detach().float().mean()
        if torch.is_tensor(value)
        else torch.tensor(float(value), device=device, dtype=torch.float32)
        for value in values
    ]
    stacked = torch.stack(scalars)
    if reduction == "min":
        result = stacked.min()
    elif reduction == "max":
        result = stacked.max()
    elif reduction == "mean":
        result = stacked.mean()
    else:
        raise ValueError(f"Unknown metric reduction {reduction!r}")
    return float(result.cpu())


def _candidate_starts(T: int, W: int, stride: int) -> list[int]:
    # t means the target index; history is [t-W, t), target is t.
    return list(range(W, T, max(int(stride), 1)))


def _shared_rollout_start(
    starts: list[int],
    *,
    randomize: bool,
    epoch: int,
    device: torch.device,
) -> int:
    """Choose one rollout start for a complete matched DDP microbatch.

    A batch-conditioned horizon Gram is defined from the mean gradient over
    every sample participating in the DDP update.  Choosing one start per rank
    would mix different relative horizons before forming that mean.  Rank zero
    therefore chooses ``t0`` and broadcasts it to all ranks.
    """

    if not starts:
        raise ValueError("shared-start training requires at least one rollout start")
    distributed = bool(
        dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    )
    rank = dist.get_rank() if distributed else 0
    if rank == 0:
        if randomize:
            selected = random.choice(starts)
        else:
            selected = starts[max(int(epoch) - 1, 0) % len(starts)]
    else:
        selected = starts[0]
    if distributed:
        encoded = torch.tensor([selected], device=device, dtype=torch.long)
        dist.broadcast(encoded, src=0)
        selected = int(encoded.item())
    if selected not in starts:
        raise RuntimeError(
            f"broadcast rollout start {selected} is invalid for local starts "
            f"[{starts[0]}, {starts[-1]}]"
        )
    return int(selected)


















def _ar_forward_pred(raw, stim_window: torch.Tensor | None, history: torch.Tensor) -> torch.Tensor:
    """Return one predicted next frame [B,...] for any standard AR model."""
    out = raw(stim_window, history, return_aux=False)
    if isinstance(out, tuple):
        out = out[0]
    if out.dim() == history.dim():
        # [B,1,...]
        return out[:, 0]
    return out


def _checkpointed_windowed_step(
    raw,
    horizon_index: int,
    total_horizon: int,
    stim_window: torch.Tensor | None,
    history: torch.Tensor,
):
    """Tensor-only forward used by non-reentrant activation checkpointing."""

    if hasattr(raw, "set_resgrad_context"):
        raw.set_resgrad_context(
            horizon_index=horizon_index,
            total_horizon=total_horizon,
        )
    return raw(stim_window, history, return_aux=False)




































def _windowed_step_maybe_checkpointed(raw, stim_window, history, horizon_index, total_horizon, use_checkpoint):
    """Run one closed-loop windowed-AR rollout step, optionally under
    gradient checkpointing.

    Exact-gradient analog of _recurrent_step_maybe_checkpointed for
    windowed-history standard-AR models (unet_field, resnet_vector,
    transformer_vector, ...) trained through compute_full_bptt_rollout_loss.
    Unlike the recurrent-state case there is no persistent state tuple to
    flatten/unflatten -- (stim_window, history) are the only tensors that
    need to be recomputed on backward. Diagnostic aux fields (resgrad_gate,
    branch_residual_ratio) are not available for checkpointed steps, same
    as the recurrent path.
    """
    if not use_checkpoint:
        if hasattr(raw, "set_resgrad_context"):
            raw.set_resgrad_context(horizon_index=horizon_index, total_horizon=total_horizon)
        return raw(stim_window, history, return_aux=False)
    return torch.utils.checkpoint.checkpoint(
        _checkpointed_windowed_step,
        raw, horizon_index, total_horizon, stim_window, history,
        use_reentrant=False,
    )


def _backward_scale_value(value, scale: float):
    """Keep a forward value fixed while multiplying its incoming gradient.

    ``scale=0`` returns a genuinely detached value, so a sampled ARTBP cut also
    releases the preceding autograd graph.  Tuples/lists are handled
    recursively because recurrent Mamba states contain several tensors per
    layer.
    """
    if torch.is_tensor(value):
        if scale <= 0.0:
            return value.detach()
        if scale == 1.0:
            return value
        return value.detach() + float(scale) * (value - value.detach())
    if isinstance(value, tuple):
        return tuple(_backward_scale_value(v, scale) for v in value)
    if isinstance(value, list):
        return [_backward_scale_value(v, scale) for v in value]
    raise TypeError(f"unsupported ARTBP carry type: {type(value)!r}")


def _sample_artbp_edge_scale(
    expected_segment_length: int,
    device: torch.device,
    *,
    draw: Optional[float] = None,
) -> float:
    """Sample the ARTBP multiplier for one temporal edge.

    With ``c=1/L``, the segment length is geometric with mean ``L``.  A cut has
    multiplier zero; a surviving edge has multiplier ``1/(1-c)``.  Hence the
    expected multiplier is one, which is the local compensation used by ARTBP.
    ``draw`` is exposed only for deterministic unit tests.
    """
    length = int(expected_segment_length)
    if length <= 1:
        return 1.0
    cut_probability = 1.0 / float(length)
    if draw is None:
        draw = float(torch.rand((), device=device).item())
    if float(draw) < cut_probability:
        return 0.0
    return 1.0 / (1.0 - cut_probability)


def compute_full_bptt_rollout_loss(
    raw,
    state: torch.Tensor,
    stim: Optional[torch.Tensor],
    start_t: int,
    args,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Closed-loop rollout objective for the paper's U-Net backbone."""

    _, total_steps, _ = get_batch_time_shape(state)
    window = int(args.window_size)
    requested = int(getattr(args, "bptt_horizon", 0))
    horizon = min(max(requested, 0), total_steps - int(start_t))
    zero = time_point(
        state,
        max(min(int(start_t), total_steps - 1), 0),
        keep_time=True,
    ).new_tensor(0.0)
    empty_logs = {
        "ar/bptt_loss": 0.0,
        "ar/bptt_base_loss": 0.0,
        "ar/bptt_horizon": 0.0,
        "ar/bptt_first_rel_l2": 0.0,
        "ar/bptt_last_rel_l2": 0.0,
        "ar/bptt_detach_period": 0.0,
        "ar/artbp_expected_segment_length": 0.0,
        "ar/artbp_cut_fraction": 0.0,
        "ar/resgrad_routing": 0.0,
        "ar/resgrad_gate_mean": 1.0,
        "ar/resgrad_gate_min": 1.0,
        "ar/resgrad_gate_max": 1.0,
    }
    if horizon <= 0 or int(start_t) - window < 0:
        return zero, empty_logs

    if stim is None:
        stim = zero_external_input_like(
            state, int(getattr(args, "stim_dim", 1))
        )
    history = time_window(
        state, int(start_t) - window, int(start_t)
    )
    loss_name = str(
        getattr(args, "bptt_loss_type", getattr(args, "ar_loss", "mse"))
    )
    use_checkpoint = bool(
        getattr(args, "bptt_grad_checkpoint", False)
    ) and bool(raw.training)
    detach_period = max(0, int(getattr(args, "bptt_detach_period", 0)))
    artbp_length = max(
        0, int(getattr(args, "artbp_expected_segment_length", 0))
    )
    if detach_period > 0 and artbp_length > 1:
        raise ValueError("TBPTT and ARTBP cannot be enabled together")

    losses = []
    rel_errors = []
    edge_scales = []
    total_probes = []
    noise_probes = []

    for step in range(horizon):
        target_t = int(start_t) + step
        stimulus_window = time_window(
            stim, target_t - window, target_t
        )
        if use_checkpoint:
            prediction = _windowed_step_maybe_checkpointed(
                raw,
                stimulus_window,
                history,
                step,
                horizon,
                True,
            )
        else:
            if hasattr(raw, "set_resgrad_context"):
                raw.set_resgrad_context(
                    horizon_index=step, total_horizon=horizon
                )
            prediction = raw(
                stimulus_window, history, return_aux=False
            )

        predicted_state = prediction[:, 0]
        target_window = time_point(
            state, target_t, keep_time=True
        )
        target_state = target_window[:, 0]

        if raw.training and hasattr(raw, "dual_wiener_probe_terms"):
            total_probe, noise_probe = raw.dual_wiener_probe_terms(
                predicted_state, target_state, step
            )
            if total_probe is not None and noise_probe is not None:
                total_probes.append(total_probe)
                noise_probes.append(noise_probe)

        if loss_name == "rel_l2":
            step_loss = relative_l2(
                predicted_state, target_state
            )
        else:
            step_loss = elementwise_state_loss(
                prediction, target_window, loss=loss_name
            )
        losses.append(step_loss)
        with torch.no_grad():
            rel_errors.append(
                relative_l2(predicted_state, target_state)
            )

        history = torch.cat([history[:, 1:], prediction], dim=1)
        has_next = (step + 1) < horizon
        if (
            has_next
            and detach_period > 0
            and ((step + 1) % detach_period) == 0
        ):
            history = history.detach()
        elif has_next and artbp_length > 1 and raw.training:
            edge_scale = _sample_artbp_edge_scale(
                artbp_length, history.device
            )
            history = _backward_scale_value(history, edge_scale)
            edge_scales.append(edge_scale)

    if hasattr(raw, "dual_wiener_set_probe_losses"):
        total_probe = (
            torch.stack(total_probes).mean()
            if total_probes
            else None
        )
        noise_probe = (
            torch.stack(noise_probes).mean()
            if noise_probes
            else None
        )
        raw.dual_wiener_set_probe_losses(total_probe, noise_probe)

    loss = torch.stack(losses).mean()
    logs = dict(empty_logs)
    logs.update(
        {
            "ar/bptt_loss": float(loss.detach().cpu()),
            "ar/bptt_base_loss": float(loss.detach().cpu()),
            "ar/bptt_horizon": float(horizon),
            "ar/bptt_first_rel_l2": float(
                rel_errors[0].detach().cpu()
            ),
            "ar/bptt_last_rel_l2": float(
                rel_errors[-1].detach().cpu()
            ),
            "ar/bptt_detach_period": float(detach_period),
            "ar/artbp_expected_segment_length": float(
                artbp_length
            ),
            "ar/artbp_cut_fraction": (
                float(sum(scale == 0.0 for scale in edge_scales))
                / float(len(edge_scales))
                if edge_scales
                else 0.0
            ),
        }
    )
    if hasattr(raw, "dual_wiener_diagnostics"):
        diagnostics = raw.dual_wiener_diagnostics(horizon)
        logs.update(diagnostics)
        logs["ar/resgrad_routing"] = 1.0
        logs["ar/resgrad_gate_mean"] = float(
            diagnostics.get("ar/dual_wiener_m_mean", 1.0)
        )
        logs["ar/resgrad_gate_min"] = float(
            diagnostics.get("ar/dual_wiener_m_min", 1.0)
        )
        logs["ar/resgrad_gate_max"] = float(
            diagnostics.get("ar/dual_wiener_m_max", 1.0)
        )
    return loss, logs







def _flatten_mamba_state(h):
    """Flatten a MambaStackState (tuple of (conv_state, ssm_state) per block)
    into a flat tuple of tensors, so it can be passed through
    torch.utils.checkpoint.checkpoint, which recomputes forward activations
    on the backward pass instead of storing them -- an exact-gradient
    alternative to ResGrad's approximate-gradient routing for reducing BPTT
    memory. Paired with _unflatten_mamba_state.
    """
    flat = []
    for layer_state in h:
        flat.extend(layer_state)
    return tuple(flat)


def _unflatten_mamba_state(flat, depth):
    """Inverse of _flatten_mamba_state.

    The per-layer arity is inferred rather than hard-coded at 2, so recurrent
    backbones whose layer state is not a (conv_state, ssm_state) pair -- e.g.
    a residual GRU carrying a single hidden tensor per layer -- round-trip
    correctly through checkpointing.  For the Mamba stack this is exactly the
    previous behaviour (per == 2).
    """
    per = max(1, len(flat) // max(1, depth))
    return tuple(tuple(flat[i * per:(i + 1) * per]) for i in range(depth))


def _checkpointed_recurrent_step(
    raw,
    depth: int,
    x_in: torch.Tensor,
    stim_in: torch.Tensor | None,
    horizon_index: int,
    total_horizon: int,
    *h_flat: torch.Tensor,
):
    """Rebuild one recurrent step during activation-checkpoint backward."""

    h = _unflatten_mamba_state(h_flat, depth)
    prediction, h_next, _ = raw.step(
        h,
        x_in,
        stim_in,
        return_aux=True,
        horizon_index=horizon_index,
        total_horizon=total_horizon,
    )
    return (prediction,) + _flatten_mamba_state(h_next)




def _recurrent_step_maybe_checkpointed(raw, h, x_in, stim_in, horizon_index, total_horizon, use_checkpoint):
    """Run one closed-loop rollout step, optionally under gradient
    checkpointing.

    Checkpointing recomputes this step's forward pass during backward
    instead of retaining its activations, giving the *exact* full-BPTT
    gradient at roughly O(1)-in-K memory (per step), at the cost of ~2x
    forward compute. This is the natural baseline for what ResGrad routing
    is being compared against: an existing, gradient-exact way to make long
    BPTT horizons memory-feasible. Diagnostic aux fields (resgrad_gate,
    dt stats, etc.) are not preserved for checkpointed steps since they are
    not needed for the loss and recomputing dict/aux structures through
    checkpoint adds unnecessary complexity; use non-checkpointed runs for
    fine-grained per-step diagnostics.
    """
    if not use_checkpoint:
        return raw.step(h, x_in, stim_in, return_aux=True, horizon_index=horizon_index, total_horizon=total_horizon)
    depth = len(h)
    h_flat = _flatten_mamba_state(h)
    outputs = torch.utils.checkpoint.checkpoint(
        _checkpointed_recurrent_step,
        raw, depth, x_in, stim_in, horizon_index, total_horizon,
        *h_flat,
        use_reentrant=False,
    )
    pred = outputs[0]
    h_next = _unflatten_mamba_state(outputs[1:], depth)
    return pred, h_next, {}


def _gather_recurrent_start_batch(
    sequence: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """Gather time indices and flatten them in start-major, then sample order."""

    gathered = sequence.index_select(1, indices)  # [B, starts, ...]
    order = (1, 0) + tuple(range(2, gathered.dim()))
    gathered = gathered.permute(order).contiguous()
    return gathered.reshape(
        int(indices.numel()) * int(sequence.shape[0]), *sequence.shape[2:]
    )




def _packed_recurrent_validation_loss(
    raw,
    state: torch.Tensor,
    stim: torch.Tensor,
    starts: list[int],
    *,
    burn: int,
    horizon: int,
    loss_name: str,
    decay: float,
    start_batch: int,
):
    """Evaluate identical rollout starts in start-major packed batches.

    This changes only execution batching.  Every start receives its own zero
    recurrent state, the same ground-truth burn-in frames, and the same closed
    rollout as the serial reference.
    """

    B = int(state.shape[0])
    burn_start_values = [max(0, int(s) - burn) for s in starts]
    burn_lengths = [
        max(burn_start, int(s) - 1) - burn_start
        for s, burn_start in zip(starts, burn_start_values)
    ]
    if not burn_lengths or len(set(burn_lengths)) != 1:
        return None
    burn_length = int(burn_lengths[0])

    weighted_loss = state.new_tensor(0.0, dtype=torch.float64)
    weighted_first = state.new_tensor(0.0, dtype=torch.float64)
    weighted_last = state.new_tensor(0.0, dtype=torch.float64)
    start_count = 0

    for chunk_start in range(0, len(starts), int(start_batch)):
        chunk = starts[chunk_start : chunk_start + int(start_batch)]
        groups = len(chunk)
        chunk_starts = torch.as_tensor(chunk, device=state.device, dtype=torch.long)
        burn_starts = torch.as_tensor(
            burn_start_values[chunk_start : chunk_start + groups],
            device=state.device,
            dtype=torch.long,
        )

        h = raw.init_state(groups * B, state.device, state.dtype)
        for offset in range(burn_length):
            indices = burn_starts + int(offset)
            state_in = _gather_recurrent_start_batch(state, indices)
            stim_in = (
                _gather_recurrent_start_batch(stim, indices)
                if stim is not None
                else None
            )
            _, h = raw.step(h, state_in, stim_in, return_aux=False)

        x_in = _gather_recurrent_start_batch(state, chunk_starts - 1)
        chunk_losses = []
        chunk_rels = []
        for k in range(int(horizon)):
            input_indices = chunk_starts + int(k) - 1
            target_indices = chunk_starts + int(k)
            stim_in = (
                _gather_recurrent_start_batch(stim, input_indices)
                if stim is not None
                else None
            )
            pred, h, _ = _recurrent_step_maybe_checkpointed(
                raw,
                h,
                x_in,
                stim_in,
                k,
                int(horizon),
                use_checkpoint=False,
            )
            target = _gather_recurrent_start_batch(state, target_indices)
            rel = relative_l2(pred, target)
            if loss_name == "rel_l2":
                step_loss = rel
            else:
                step_loss = elementwise_state_loss(
                    pred.unsqueeze(1), target.unsqueeze(1), loss=loss_name
                )
            if decay != 1.0:
                step_loss = (decay ** k) * step_loss
            chunk_losses.append(step_loss)
            chunk_rels.append(rel)
            x_in = pred

        chunk_loss = torch.stack(chunk_losses).mean()
        weighted_loss = weighted_loss + chunk_loss.double() * groups
        weighted_first = weighted_first + chunk_rels[0].double() * groups
        weighted_last = weighted_last + chunk_rels[-1].double() * groups
        start_count += groups

    if start_count <= 0:
        return None
    denom = float(start_count)
    return (
        weighted_loss / denom,
        weighted_first / denom,
        weighted_last / denom,
        burn_length,
    )


def compute_recurrent_state_bptt_loss(
    model,
    state: torch.Tensor,
    stim: torch.Tensor,
    args,
    epoch: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Closed-loop recurrent objective used by the paper's Mamba backbone.

    Ground-truth burn-in builds a detached recurrent state. The following K
    predictions are fed back into the model and differentiated with full BPTT,
    periodic TBPTT, or ARTBP. Internal-DW only changes backward routing; the
    forward rollout and scalar forecasting objective remain unchanged.
    """

    raw = unwrap_model(model)
    batch_size, total_steps, _ = get_batch_time_shape(state)
    horizon = max(1, int(getattr(args, "mamba_bptt_horizon", 8)))
    burnin = max(0, int(getattr(args, "mamba_burnin", 64)))
    detach_period = max(0, int(getattr(args, "bptt_detach_period", 0)))
    artbp_length = max(
        0, int(getattr(args, "artbp_expected_segment_length", 0))
    )
    if detach_period > 0 and artbp_length > 1:
        raise ValueError("TBPTT and ARTBP cannot be enabled together")

    loss_name = str(
        getattr(args, "mamba_loss_type", getattr(args, "ar_loss", "rel_l2"))
    )
    decay = float(getattr(args, "mamba_loss_decay", 1.0))
    if stim is None:
        stim = zero_external_input_like(
            state, int(getattr(args, "stim_dim", 1))
        )

    max_start = total_steps - horizon
    if max_start < 1:
        if hasattr(raw, "dual_wiener_set_probe_losses"):
            raw.dual_wiener_set_probe_losses(None, None)
        zero = state.new_tensor(0.0)
        return zero, {
            "loss": 0.0,
            "ar/recurrent_bptt_loss": 0.0,
            "ar/recurrent_bptt_horizon": 0.0,
            "ar/recurrent_burnin": float(burnin),
            "ar/recurrent_first_rel_l2": 0.0,
            "ar/recurrent_last_rel_l2": 0.0,
        }

    min_start = burnin if max_start >= burnin else 1
    stride = max(1, int(getattr(args, "mamba_train_stride", 1)))
    starts = list(range(min_start, max_start + 1, stride))
    if not starts:
        starts = list(range(1, max_start + 1, stride))

    if raw.training:
        count = int(getattr(args, "mamba_train_starts_per_sequence", -1))
        if count <= 0:
            count = int(getattr(args, "ar_train_starts_per_sequence", 1))
        count = max(1, min(count, len(starts)))
        randomize = bool(getattr(args, "ar_train_random_starts", True))
        if bool(getattr(args, "ar_shared_rollout_start", False)):
            chosen = [
                _shared_rollout_start(
                    starts,
                    randomize=randomize,
                    epoch=int(epoch),
                    device=state.device,
                )
            ]
        elif randomize:
            chosen = random.sample(starts, count)
        else:
            offset = max(int(epoch) - 1, 0) % len(starts)
            chosen = [starts[(offset + i) % len(starts)] for i in range(count)]
    else:
        eval_stride = max(1, int(getattr(args, "ar_eval_stride", 4)))
        chosen = list(range(min_start, max_start + 1, eval_stride)) or [
            min_start
        ]

    validation_start_batch = max(
        1, int(getattr(args, "recurrent_val_start_batch", 1))
    )
    if (
        not raw.training
        and validation_start_batch > 1
        and bool(getattr(raw, "is_official_state_mamba", False))
        and getattr(raw, "block_groups", None) is None
    ):
        packed = _packed_recurrent_validation_loss(
            raw,
            state,
            stim,
            chosen,
            burn=burnin,
            horizon=horizon,
            loss_name=loss_name,
            decay=decay,
            start_batch=validation_start_batch,
        )
        if packed is not None:
            base_loss, first_rel, last_rel, used_burnin = packed
            logs = {
                "loss": float(base_loss.detach().cpu()),
                "ar/recurrent_bptt_loss": float(base_loss.detach().cpu()),
                "ar/recurrent_bptt_horizon": float(horizon),
                "ar/recurrent_burnin": float(burnin),
                "ar/recurrent_used_burnin": float(used_burnin),
                "ar/recurrent_num_starts": float(len(chosen)),
                "ar/recurrent_first_rel_l2": float(first_rel.detach().cpu()),
                "ar/recurrent_last_rel_l2": float(last_rel.detach().cpu()),
                "ar/one_step_rel_l2": float(first_rel.detach().cpu()),
                "ar/packed_validation_start_batch": float(
                    validation_start_batch
                ),
            }
            if hasattr(raw, "dual_wiener_diagnostics"):
                logs.update(raw.dual_wiener_diagnostics(horizon))
            if hasattr(raw, "dual_wiener_set_probe_losses"):
                raw.dual_wiener_set_probe_losses(None, None)
            return base_loss, logs

    rollout_losses = []
    first_rels = []
    last_rels = []
    correlations = []
    hidden_norms = []
    alpha_means = []
    alpha_mins = []
    alpha_maxs = []
    route_gates = []
    route_enabled = []
    route_ratios = []
    route_branch_norms = []
    route_residual_norms = []
    prediction_stds = []
    target_stds = []
    used_burnins = []
    artbp_edge_scales = []
    jacobian_terms = []
    jacobian_gains = []
    dw_total_terms = []
    dw_noise_terms = []

    for start_t in chosen:
        burn_start = max(0, int(start_t) - burnin)
        burn_end = max(burn_start, int(start_t) - 1)
        used_burnins.append(float(burn_end - burn_start))

        hidden = raw.init_state(
            batch_size, state.device, state.dtype
        )
        if burn_end > burn_start:
            with torch.no_grad():
                for time_index in range(burn_start, burn_end):
                    stimulus = (
                        stim[:, time_index] if stim is not None else None
                    )
                    _, hidden = raw.step(
                        hidden,
                        state[:, time_index],
                        stimulus,
                        return_aux=False,
                    )
        hidden = _detach_recurrent_state(hidden)
        model_input = state[:, int(start_t) - 1]
        step_losses = []
        step_rels = []

        for step in range(horizon):
            target_t = int(start_t) + step
            stimulus = stim[:, target_t - 1] if stim is not None else None

            if (
                step == 0
                and raw.training
                and float(
                    getattr(args, "forward_jacobian_lambda", 0.0)
                )
                != 0.0
            ):
                fixed_hidden = _detach_recurrent_state(hidden)

                def recurrent_forward(value):
                    output = raw.step(
                        fixed_hidden,
                        value,
                        stimulus,
                        return_aux=False,
                        horizon_index=0,
                        total_horizon=horizon,
                    )
                    return output[0]

                penalty, gain = _forward_jacobian_fd_penalty(
                    model_input,
                    recurrent_forward,
                    eps=float(
                        getattr(args, "forward_jacobian_eps", 1e-3)
                    ),
                    target=float(
                        getattr(args, "forward_jacobian_target", 1.0)
                    ),
                )
                jacobian_terms.append(penalty)
                jacobian_gains.append(gain)

            prediction, hidden, aux = _recurrent_step_maybe_checkpointed(
                raw,
                hidden,
                model_input,
                stimulus,
                step,
                horizon,
                use_checkpoint=bool(
                    getattr(args, "recurrent_grad_checkpoint", False)
                )
                and raw.training,
            )
            target = state[:, target_t]

            if raw.training and hasattr(raw, "dual_wiener_probe_terms"):
                total_probe, noise_probe = raw.dual_wiener_probe_terms(
                    prediction, target, step
                )
                if total_probe is not None and noise_probe is not None:
                    dw_total_terms.append(total_probe)
                    dw_noise_terms.append(noise_probe)

            if loss_name == "rel_l2":
                step_loss = relative_l2(prediction, target)
            else:
                step_loss = elementwise_state_loss(
                    prediction.unsqueeze(1),
                    target.unsqueeze(1),
                    loss=loss_name,
                )
            if decay != 1.0:
                step_loss = (decay ** step) * step_loss
            step_losses.append(step_loss)

            with torch.no_grad():
                rel = relative_l2(prediction, target) if not bool(getattr(args, "compact_recurrent_logging", False)) else step_loss.detach()
                step_rels.append(rel)
                if not bool(getattr(args, "compact_recurrent_logging", False)):
                    correlations.append(corrcoef_flat(prediction, target))
                    prediction_stds.append(prediction.std())
                    target_stds.append(target.std())
                    if isinstance(aux, dict):
                        hidden_norms.append(
                            aux.get(
                                "hidden_norm", prediction.new_tensor(0.0)
                            ).detach()
                        )
                        alpha_means.append(
                            aux.get(
                                "alpha_mean", prediction.new_tensor(0.0)
                            ).detach()
                        )
                        alpha_mins.append(
                            aux.get(
                                "alpha_min", prediction.new_tensor(0.0)
                            ).detach()
                        )
                        alpha_maxs.append(
                            aux.get(
                                "alpha_max", prediction.new_tensor(0.0)
                            ).detach()
                        )
                        for key, destination in (
                            ("resgrad_gate", route_gates),
                            ("resgrad_routing", route_enabled),
                            ("resgrad_branch_residual_ratio", route_ratios),
                            ("resgrad_branch_norm", route_branch_norms),
                            ("resgrad_residual_norm", route_residual_norms),
                        ):
                            if key in aux:
                                destination.append(aux[key].detach())

            has_next = (step + 1) < horizon
            if (
                raw.training
                and has_next
                and detach_period > 0
                and ((step + 1) % detach_period) == 0
            ):
                model_input = prediction.detach()
                hidden = _detach_recurrent_state(hidden)
            elif raw.training and has_next and artbp_length > 1:
                edge_scale = _sample_artbp_edge_scale(
                    artbp_length, prediction.device
                )
                model_input = _backward_scale_value(
                    prediction, edge_scale
                )
                hidden = _backward_scale_value(hidden, edge_scale)
                artbp_edge_scales.append(edge_scale)
            else:
                model_input = prediction

        if step_losses:
            rollout_losses.append(torch.stack(step_losses).mean())
            first_rels.append(step_rels[0].detach())
            last_rels.append(step_rels[-1].detach())

    if not rollout_losses:
        if hasattr(raw, "dual_wiener_set_probe_losses"):
            raw.dual_wiener_set_probe_losses(None, None)
        zero = state.new_tensor(0.0)
        return zero, {
            "loss": 0.0,
            "ar/recurrent_bptt_loss": 0.0,
        }

    if hasattr(raw, "dual_wiener_set_probe_losses"):
        total_probe = (
            torch.stack(dw_total_terms).mean()
            if dw_total_terms
            else None
        )
        noise_probe = (
            torch.stack(dw_noise_terms).mean()
            if dw_noise_terms
            else None
        )
        raw.dual_wiener_set_probe_losses(total_probe, noise_probe)

    base_loss = torch.stack(rollout_losses).mean()
    jacobian_loss = (
        torch.stack(jacobian_terms).mean()
        if jacobian_terms
        else state.new_tensor(0.0)
    )
    jacobian_lambda = float(
        getattr(args, "forward_jacobian_lambda", 0.0)
    )
    loss = base_loss + jacobian_lambda * jacobian_loss

    if bool(getattr(args, "compact_recurrent_logging", False)):
        return loss, {"loss": loss.detach(), "ar/recurrent_bptt_loss": base_loss.detach(),
                      "ar/recurrent_num_starts": float(len(chosen))}

    if raw.training and bool(getattr(args, "fast_train_logging", False)):
        return loss, {
            "loss": loss.detach(),
            "ar/recurrent_bptt_loss": base_loss.detach(),
            "ar/recurrent_bptt_horizon": float(horizon),
        }

    logs = {
        "loss": float(loss.detach().cpu()),
        "ar/recurrent_bptt_loss": float(base_loss.detach().cpu()),
        "ar/recurrent_bptt_horizon": float(horizon),
        "ar/recurrent_burnin": float(burnin),
        "ar/recurrent_used_burnin": (
            sum(used_burnins) / max(len(used_burnins), 1)
        ),
        "ar/recurrent_num_starts": float(len(chosen)),
        "ar/recurrent_first_rel_l2": _reduce_metric_scalars(
            first_rels
        ),
        "ar/recurrent_last_rel_l2": _reduce_metric_scalars(last_rels),
        "ar/one_step_rel_l2": _reduce_metric_scalars(first_rels),
        "ar/one_step_corr": _reduce_metric_scalars(correlations),
        "ar/recurrent_hidden_norm": _reduce_metric_scalars(
            hidden_norms
        ),
        "ar/recurrent_alpha_mean": _reduce_metric_scalars(alpha_means),
        "ar/recurrent_alpha_min": _reduce_metric_scalars(
            alpha_mins, "min"
        ),
        "ar/recurrent_alpha_max": _reduce_metric_scalars(
            alpha_maxs, "max"
        ),
        "ar/recurrent_pred_std": _reduce_metric_scalars(
            prediction_stds
        ),
        "ar/recurrent_gt_std": _reduce_metric_scalars(target_stds),
        "ar/resgrad_routing": _reduce_metric_scalars(route_enabled),
        "ar/resgrad_gate_mean": _reduce_metric_scalars(
            route_gates, default=1.0
        ),
        "ar/resgrad_gate_min": _reduce_metric_scalars(
            route_gates, "min", default=1.0
        ),
        "ar/resgrad_gate_max": _reduce_metric_scalars(
            route_gates, "max", default=1.0
        ),
        "ar/resgrad_branch_residual_ratio": _reduce_metric_scalars(
            route_ratios
        ),
        "ar/resgrad_branch_norm": _reduce_metric_scalars(
            route_branch_norms
        ),
        "ar/resgrad_residual_norm": _reduce_metric_scalars(
            route_residual_norms
        ),
        "ar/bptt_detach_period": float(detach_period),
        "ar/artbp_expected_segment_length": float(artbp_length),
        "ar/artbp_cut_fraction": (
            float(sum(scale == 0.0 for scale in artbp_edge_scales))
            / float(len(artbp_edge_scales))
            if artbp_edge_scales
            else 0.0
        ),
        "ar/forward_jacobian_loss": float(
            jacobian_loss.detach().cpu()
        ),
        "ar/forward_jacobian_lambda": jacobian_lambda,
        "ar/forward_jacobian_weighted_loss": float(
            (jacobian_lambda * jacobian_loss).detach().cpu()
        ),
        "ar/forward_jacobian_gain": _reduce_metric_scalars(
            jacobian_gains
        ),
    }
    if hasattr(raw, "dual_wiener_diagnostics"):
        logs.update(raw.dual_wiener_diagnostics(horizon))
    return loss, logs





def compute_autoregressive_one_step_loss(
    model,
    state,
    stim,
    args,
    epoch: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Paper loss for the recurrent Mamba and windowed U-Net backbones."""

    raw = unwrap_model(model)
    if bool(getattr(raw, "is_recurrent_state_ar", False)):
        return compute_recurrent_state_bptt_loss(
            model, state, stim, args, epoch=epoch
        )

    _batch, total_steps, _ = get_batch_time_shape(state)
    window = int(args.window_size)
    if total_steps <= window:
        raise ValueError(
            "Sequence too short for autoregressive training: "
            f"T={total_steps}, window={window}"
        )
    if stim is None:
        stim = zero_external_input_like(
            state, int(getattr(args, "stim_dim", 1))
        )

    if raw.training:
        starts = _candidate_starts(
            total_steps, window, int(getattr(args, "ar_train_stride", 1))
        )
        count = max(
            1,
            min(
                int(getattr(args, "ar_train_starts_per_sequence", 1)),
                len(starts),
            ),
        )
        randomize = bool(getattr(args, "ar_train_random_starts", True))
        if bool(getattr(args, "ar_shared_rollout_start", False)):
            chosen = [
                _shared_rollout_start(
                    starts,
                    randomize=randomize,
                    epoch=int(epoch),
                    device=state.device,
                )
            ]
        elif randomize:
            chosen = random.sample(starts, count)
        else:
            offset = max(int(epoch) - 1, 0) % len(starts)
            chosen = [starts[(offset + i) % len(starts)] for i in range(count)]
    else:
        chosen = _candidate_starts(
            total_steps, window, int(getattr(args, "ar_eval_stride", 4))
        )

    one_step_losses = []
    bptt_losses = []
    bptt_logs_accum: Dict[str, float] = {}
    corrs = []
    relative_errors = []
    bptt_enabled = bool(getattr(args, "bptt_loss", False)) and (
        raw.training or bool(getattr(args, "bptt_eval", False))
    )
    bptt_lambda = float(getattr(args, "bptt_lambda", 0.0))

    for target_time in chosen:
        history = time_window(state, target_time - window, target_time)
        stim_window = time_window(stim, target_time - window, target_time)
        target = time_point(state, target_time, keep_time=True)
        pred = raw(stim_window, history, return_aux=False)

        loss_name = str(getattr(args, "ar_loss", "rel_l2"))
        if loss_name == "rel_l2":
            one_step = relative_l2(pred[:, 0], target[:, 0])
        else:
            one_step = elementwise_state_loss(pred, target, loss=loss_name)
        one_step_losses.append(one_step)

        if bptt_enabled and bptt_lambda != 0.0:
            bptt_loss, bptt_logs = compute_full_bptt_rollout_loss(
                raw, state, stim, target_time, args
            )
            bptt_losses.append(bptt_loss)
            for key, value in bptt_logs.items():
                bptt_logs_accum[key] = (
                    bptt_logs_accum.get(key, 0.0) + float(value)
                )

        with torch.no_grad():
            corrs.append(corrcoef_flat(pred[:, 0], target[:, 0]))
            relative_errors.append(relative_l2(pred[:, 0], target[:, 0]))

    loss_one_step = torch.stack(one_step_losses).mean()
    if bptt_losses:
        loss_bptt = torch.stack(bptt_losses).mean()
    else:
        loss_bptt = loss_one_step.new_tensor(0.0)
        bptt_lambda = 0.0

    one_step_lambda = float(getattr(args, "ar_one_step_lambda", 1.0))
    loss_one_step_weighted = one_step_lambda * loss_one_step
    loss_bptt_weighted = bptt_lambda * loss_bptt

    jacobian_lambda = float(getattr(args, "forward_jacobian_lambda", 0.0))
    if raw.training and jacobian_lambda != 0.0 and chosen:
        target_time = int(chosen[0])
        history = time_window(state, target_time - window, target_time)
        stim_window = time_window(stim, target_time - window, target_time)

        def _jreg_forward(value):
            return _ar_forward_pred(raw, stim_window, value)

        jacobian_loss, jacobian_gain = _forward_jacobian_fd_penalty(
            history,
            _jreg_forward,
            eps=float(getattr(args, "forward_jacobian_eps", 1e-3)),
            target=float(getattr(args, "forward_jacobian_target", 1.0)),
        )
    else:
        jacobian_loss = loss_one_step.new_tensor(0.0)
        jacobian_gain = loss_one_step.new_tensor(0.0)
    jacobian_weighted = jacobian_lambda * jacobian_loss

    loss = loss_one_step_weighted + loss_bptt_weighted + jacobian_weighted
    if raw.training and bool(getattr(args, "fast_train_logging", False)):
        return loss, {
            "loss": loss.detach(),
            "ar/loss_one_step": loss_one_step.detach(),
            "ar/bptt_loss": loss_bptt.detach(),
        }

    logs = {
        "loss": float(loss.detach().cpu()),
        "ar/loss_one_step": float(loss_one_step.detach().cpu()),
        "ar/one_step_lambda": one_step_lambda,
        "ar/loss_one_step_weighted": float(
            loss_one_step_weighted.detach().cpu()
        ),
        "ar/bptt_loss": float(loss_bptt.detach().cpu()),
        "ar/bptt_lambda": bptt_lambda,
        "ar/bptt_weighted_loss": float(loss_bptt_weighted.detach().cpu()),
        "ar/forward_jacobian_loss": float(jacobian_loss.detach().cpu()),
        "ar/forward_jacobian_lambda": jacobian_lambda,
        "ar/forward_jacobian_weighted_loss": float(
            jacobian_weighted.detach().cpu()
        ),
        "ar/forward_jacobian_gain": float(jacobian_gain.detach().cpu()),
        "ar/corr": float(sum(corrs) / max(len(corrs), 1)),
        "ar/rel_l2": float(
            sum(relative_errors) / max(len(relative_errors), 1)
        ),
        "ar/num_starts": float(len(chosen)),
        "ar/window_size": float(window),
    }
    if bptt_logs_accum:
        for key, value in bptt_logs_accum.items():
            if key not in {"ar/bptt_loss", "ar/bptt_lambda"}:
                logs[key] = value / len(bptt_losses)
    if hasattr(raw, "dual_wiener_diagnostics"):
        logs.update(
            raw.dual_wiener_diagnostics(
                int(getattr(args, "bptt_horizon", 1))
            )
        )
    return loss, logs
