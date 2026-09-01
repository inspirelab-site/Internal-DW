"""Shape helpers for framework-level Koopman-Gram operations.

The project-level convention is:
    single sample state: [T, ...]
    batched state:       [B, T, ...]

These helpers deliberately preserve spatial/field structure before model input.
Flattening is used only at scalar loss / metric reduction time.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F


def get_batch_time_shape(state: torch.Tensor) -> Tuple[int, int, torch.Size]:
    if state.dim() < 3:
        raise ValueError(f"Expected state [B,T,...], got shape={tuple(state.shape)}")
    return int(state.shape[0]), int(state.shape[1]), state.shape[2:]


def time_window(state: torch.Tensor, start: int, end: int) -> torch.Tensor:
    return state[:, int(start):int(end)]


def time_point(state: torch.Tensor, t: int, keep_time: bool = False) -> torch.Tensor:
    t = int(t)
    return state[:, t:t + 1] if keep_time else state[:, t]


def last_time_point(state_window: torch.Tensor, keep_time: bool = False) -> torch.Tensor:
    return state_window[:, -1:] if keep_time else state_window[:, -1]


def append_time_point(history: torch.Tensor, next_frame: torch.Tensor) -> torch.Tensor:
    """Roll a [B,W,...] history forward by appending next_frame.

    next_frame may be [B,...] or [B,1,...].
    """
    if next_frame.dim() == history.dim() - 1:
        next_frame = next_frame.unsqueeze(1)
    if next_frame.dim() != history.dim() or next_frame.shape[1] != 1:
        raise ValueError(
            f"next_frame must be [B,...] or [B,1,...]; got history={tuple(history.shape)}, next={tuple(next_frame.shape)}"
        )
    return torch.cat([history[:, 1:], next_frame], dim=1)


def flatten_for_loss(x: torch.Tensor) -> torch.Tensor:
    """Flatten non-batch dimensions for elementwise scalar reduction only."""
    return x.reshape(x.shape[0], -1)


def elementwise_state_loss(pred: torch.Tensor, target: torch.Tensor, loss: str = "huber", delta: float = 20.0) -> torch.Tensor:
    """Generic pointwise state loss that works for vector and field states.

    This should not be used for domain-specific field losses such as spectral,
    conservation, or PDE-residual losses.
    """
    p = flatten_for_loss(pred)
    y = flatten_for_loss(target)
    if loss == "huber":
        return F.huber_loss(p, y, delta=float(delta))
    if loss == "mse":
        return F.mse_loss(p, y)
    if loss == "l1":
        return F.l1_loss(p, y)
    raise ValueError(f"Unknown elementwise loss={loss}")


def zero_external_input_like(state: torch.Tensor, input_dim: int = 1) -> torch.Tensor:
    """Create a zero external-input sequence [T,input_dim] or [B,T,input_dim]."""
    if state.dim() == 0:
        raise ValueError("state must have a time dimension")
    if state.dim() >= 3:  # [B,T,...]
        return state.new_zeros(state.shape[0], state.shape[1], int(input_dim))
    return state.new_zeros(state.shape[0], int(input_dim))


def unpack_batch(batch: Any):
    """Accept both the new dict protocol and old tuple batches.

    Returns:
        state, external_input, label, metadata
    """
    if isinstance(batch, dict):
        state = batch["state"]
        external_input = batch.get("external_input", None)
        label = batch.get("label", None)
        metadata = batch.get("metadata", {})
        return state, external_input, label, metadata
    if isinstance(batch, (tuple, list)) and len(batch) >= 3:
        return batch[0], batch[1], batch[2], {}
    raise TypeError(f"Unsupported batch type: {type(batch)}")


def corrcoef_flat(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Correlation after flattening all non-batch dimensions.

    For HCP [B,ROI], this is a sample-wise flattened version of the generic metric.
    HCP-specific ROI-wise metrics should live in an HCP evaluator.
    """
    p = flatten_for_loss(pred)
    y = flatten_for_loss(target)
    p = p - p.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    num = (p * y).sum(dim=1)
    den = torch.sqrt((p.square().sum(dim=1) + eps) * (y.square().sum(dim=1) + eps))
    return (num / den.clamp_min(eps)).mean()
