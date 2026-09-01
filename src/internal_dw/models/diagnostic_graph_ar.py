"""Diagnostic AR controls for cyclic_graph_ar.

This file defines two parameter-matched controls:

1) simple_delta_ar
   No graph and no closed-loop graph state:
       x_{t+1} = x_t + f_theta(x_t, u_{t+1})

2) feedforward_graph_ar
   ROI graph message passing with *unshared* graph blocks:
       h^{s+1} = F_{theta_s}(h^s, A, u)
   It keeps the same recurrent-state training interface as cyclic_graph_ar, but
   the internal graph computation is a feed-forward stack rather than repeatedly
   applying one shared update operator.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .simple_ar import _StimPoolMixin

Tensor = torch.Tensor


def _ring_adj(n: int, topk: int, device=None, dtype=None) -> Tensor:
    topk = int(max(1, min(topk, max(1, n // 2))))
    A = torch.zeros(n, n, device=device, dtype=dtype or torch.float32)
    idx = torch.arange(n, device=device)
    for s in range(1, topk + 1):
        A[idx, (idx + s) % n] = 1.0
        A[idx, (idx - s) % n] = 1.0
    return A / A.sum(-1, keepdim=True).clamp_min(1.0)


def _skip_adj(n: int, topk: int, device=None, dtype=None) -> Tensor:
    A = _ring_adj(n, max(1, topk), device=device, dtype=dtype)
    idx = torch.arange(n, device=device)
    for s in [4, 16, 64]:
        if s < n:
            A[idx, (idx + s) % n] = 1.0
            A[idx, (idx - s) % n] = 1.0
    return A / A.sum(-1, keepdim=True).clamp_min(1.0)


def _dense_adj(n: int, device=None, dtype=None) -> Tensor:
    A = torch.ones(n, n, device=device, dtype=dtype or torch.float32)
    A.fill_diagonal_(0.0)
    return A / A.sum(-1, keepdim=True).clamp_min(1.0)


def _make_adj(n: int, topk: int, edge_type: str, device=None, dtype=None) -> Tensor:
    edge_type = str(edge_type).lower()
    if edge_type in {"dense", "full"}:
        return _dense_adj(n, device=device, dtype=dtype)
    if edge_type == "skip":
        return _skip_adj(n, topk, device=device, dtype=dtype)
    return _ring_adj(n, topk, device=device, dtype=dtype)


class _FFGraphBlock(nn.Module):
    def __init__(self, hidden_dim: int, mult: int = 2, dropout: float = 0.0):
        super().__init__()
        inner = int(hidden_dim * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(3 * hidden_dim),
            nn.Linear(3 * hidden_dim, inner),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(inner, hidden_dim),
        )

    def forward(self, h: Tensor, msg: Tensor, context: Tensor, alpha: float) -> Tensor:
        dh = self.net(torch.cat([h, msg, context.expand_as(h)], dim=-1))
        h = h + float(alpha) * dh
        return F.layer_norm(h, h.shape[-1:])


class FeedForwardGraphAR(nn.Module, _StimPoolMixin):
    """Non-cyclic graph baseline with unshared graph blocks.

    This model preserves the ROI graph and roughly the same recurrent-state
    interface as cyclic_graph_ar.  However, the internal graph steps use
    different parameters F_{theta_s}; therefore it is a standard feed-forward
    graph stack, not a closed-loop shared-operator computation.
    """

    is_standard_autoregressive = True
    is_recurrent_state_ar = True
    is_feedforward_graph_ar = True

    def __init__(
        self,
        state_dim: int,
        input_dim: int = 0,
        state_shape: Optional[Sequence[int]] = None,
        hidden_dim: int = 48,
        graph_steps: int = 3,
        topk: int = 2,
        alpha: float = 0.10,
        carry: float = 0.25,
        dropout: float = 0.0,
        has_external_input: bool = True,
        residual: bool = True,
        edge_type: str = "ring",
        message_hidden_mult: int = 2,
        **unused_kwargs,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.input_dim = int(input_dim)
        self.state_shape = tuple(int(x) for x in state_shape) if state_shape is not None else None
        self.task_type = "field2d" if self.state_shape is not None and len(self.state_shape) == 3 else "vector"
        self.hidden_dim = int(hidden_dim)
        self.graph_steps = max(1, int(graph_steps))
        self.topk = max(1, int(topk))
        self.alpha = float(alpha)
        self.carry = float(carry)
        self.has_external_input = bool(has_external_input)
        self.residual = bool(residual)
        self.edge_type = str(edge_type)

        self.roi_embed = nn.Parameter(torch.randn(self.state_dim, self.hidden_dim) * 0.02)
        self.x_encoder = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.stim_encoder = nn.Sequential(
            nn.LayerNorm(max(1, self.input_dim)),
            nn.Linear(max(1, self.input_dim), self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        ) if self.has_external_input and self.input_dim > 0 else None
        self.global_encoder = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.prev_state_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.blocks = nn.ModuleList([
            _FFGraphBlock(self.hidden_dim, message_hidden_mult, dropout)
            for _ in range(self.graph_steps)
        ])
        self.readout = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

        adj = _make_adj(self.state_dim, self.topk, self.edge_type, device=torch.device("cpu"), dtype=torch.float32)
        self.register_buffer("adj_norm", adj, persistent=True)

    def init_state(self, batch_size: int, device=None, dtype=None) -> Tensor:
        device = device if device is not None else next(self.parameters()).device
        dtype = dtype if dtype is not None else next(self.parameters()).dtype
        return torch.zeros(batch_size, self.state_dim, self.hidden_dim, device=device, dtype=dtype)

    def detach_state(self, h: Tensor) -> Tensor:
        return h.detach()

    def _flatten_state(self, x_t: Tensor) -> Tensor:
        return x_t.reshape(x_t.shape[0], -1)

    def _reshape_pred(self, pred_flat: Tensor, ref: Tensor) -> Tensor:
        return pred_flat.reshape(ref.shape)

    def _pool_stim_point(self, stim_t: Optional[Tensor], B: int, device, dtype) -> Tensor:
        if not self.has_external_input or self.input_dim <= 0:
            return torch.zeros(B, 0, device=device, dtype=dtype)
        if stim_t is None:
            return torch.zeros(B, self.input_dim, device=device, dtype=dtype)
        if stim_t.dim() == 1:
            stim_t = stim_t.unsqueeze(0)
        pooled = self._pool_stim(stim_t.unsqueeze(1), B, 1, device, dtype)
        return pooled[:, 0]

    def step(
        self,
        h: Optional[Tensor],
        x_t: Tensor,
        stim_t: Optional[Tensor] = None,
        return_aux: bool = False,
        horizon_index: Optional[int] = None,
    ):
        B = x_t.shape[0]
        x_flat = self._flatten_state(x_t)
        if x_flat.shape[-1] != self.state_dim:
            raise ValueError(f"FeedForwardGraphAR expected state_dim={self.state_dim}, got {x_flat.shape[-1]}")
        if h is None:
            h = self.init_state(B, x_t.device, x_t.dtype)

        roi_input = self.x_encoder(x_flat.unsqueeze(-1)) + self.roi_embed.to(x_t.device, x_t.dtype).unsqueeze(0)
        global_ctx = self.global_encoder(x_flat).unsqueeze(1)
        if self.stim_encoder is not None:
            stim_vec = self._pool_stim_point(stim_t, B, x_t.device, x_t.dtype)
            stim_ctx = self.stim_encoder(stim_vec).unsqueeze(1)
        else:
            stim_ctx = torch.zeros(B, 1, self.hidden_dim, device=x_t.device, dtype=x_t.dtype)
        context = global_ctx + stim_ctx

        # Keep the same temporal-state interface/carry as cyclic_graph_ar, but
        # remove internal closed-loop sharing by using unshared graph blocks.
        h = roi_input + context + self.carry * self.prev_state_proj(h)
        h = F.layer_norm(h, h.shape[-1:])

        adj = self.adj_norm.to(device=x_t.device, dtype=x_t.dtype)
        internal_changes = []
        msg_norms = []
        for block in self.blocks:
            msg = torch.einsum("ij,bjh->bih", adj, h)
            h0 = h
            h = block(h, msg, context, self.alpha)
            internal_changes.append((h - h0).norm(dim=-1).mean())
            msg_norms.append(msg.norm(dim=-1).mean())

        delta = self.readout(h).squeeze(-1)
        pred_flat = x_flat + delta if self.residual else delta
        pred = self._reshape_pred(pred_flat, x_t)

        aux: Dict[str, Tensor] = {}
        if return_aux:
            edge_vals = adj[adj > 0]
            aux = {
                "hidden_norm": h.norm(dim=-1).mean().detach(),
                "alpha_mean": torch.as_tensor(self.alpha, device=x_t.device, dtype=x_t.dtype),
                "alpha_min": torch.as_tensor(self.alpha, device=x_t.device, dtype=x_t.dtype),
                "alpha_max": torch.as_tensor(self.alpha, device=x_t.device, dtype=x_t.dtype),
                "cyclic_delta_norm": delta.norm(dim=-1).mean().detach(),
                "cyclic_delta_x_ratio": (delta.norm(dim=-1) / x_flat.norm(dim=-1).clamp_min(1e-6)).mean().detach(),
                "cyclic_internal_change": torch.stack(internal_changes).mean().detach(),
                "cyclic_msg_norm": torch.stack(msg_norms).mean().detach(),
                "cyclic_edge_weight_mean": edge_vals.mean().detach() if edge_vals.numel() else torch.zeros((), device=x_t.device),
                "cyclic_edge_weight_max": adj.max().detach(),
            }
        if return_aux:
            return pred, h, aux
        return pred, h

    def forward(self, x: Tensor, stim: Optional[Tensor] = None):
        h = self.init_state(x.shape[0], x.device, x.dtype)
        pred, _h, aux = self.step(
            h,
            x[:, -1] if x.dim() >= 3 else x,
            None if stim is None else stim[:, -1],
            return_aux=True,
        )
        return pred, aux


class SimpleDeltaAR(nn.Module, _StimPoolMixin):
    """Non-graph residual baseline."""

    is_standard_autoregressive = True
    is_recurrent_state_ar = True
    is_simple_delta_ar = True

    def __init__(
        self,
        state_dim: int,
        input_dim: int = 0,
        state_shape: Optional[Sequence[int]] = None,
        hidden_dim: int = 128,
        depth: int = 3,
        dropout: float = 0.0,
        has_external_input: bool = True,
        residual: bool = True,
        **unused_kwargs,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.input_dim = int(input_dim)
        self.state_shape = tuple(int(x) for x in state_shape) if state_shape is not None else None
        self.task_type = "field2d" if self.state_shape is not None and len(self.state_shape) == 3 else "vector"
        self.hidden_dim = int(hidden_dim)
        self.depth = max(1, int(depth))
        self.dropout = float(dropout)
        self.has_external_input = bool(has_external_input)
        self.residual = bool(residual)

        in_dim = self.state_dim + (self.input_dim if self.has_external_input else 0)
        layers = []
        d = in_dim
        for _ in range(max(1, self.depth - 1)):
            layers.extend([nn.LayerNorm(d), nn.Linear(d, self.hidden_dim), nn.GELU(), nn.Dropout(self.dropout)])
            d = self.hidden_dim
        layers.extend([nn.LayerNorm(d), nn.Linear(d, self.state_dim)])
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def init_state(self, batch_size: int, device=None, dtype=None) -> Tensor:
        device = device if device is not None else next(self.parameters()).device
        dtype = dtype if dtype is not None else next(self.parameters()).dtype
        return torch.zeros(batch_size, 1, device=device, dtype=dtype)

    def detach_state(self, h: Tensor) -> Tensor:
        return h.detach()

    def _flatten_state(self, x_t: Tensor) -> Tensor:
        return x_t.reshape(x_t.shape[0], -1)

    def _reshape_pred(self, pred_flat: Tensor, ref: Tensor) -> Tensor:
        return pred_flat.reshape(ref.shape)

    def _pool_stim_point(self, stim_t: Optional[Tensor], B: int, device, dtype) -> Tensor:
        if not self.has_external_input or self.input_dim <= 0:
            return torch.zeros(B, 0, device=device, dtype=dtype)
        if stim_t is None:
            return torch.zeros(B, self.input_dim, device=device, dtype=dtype)
        if stim_t.dim() == 1:
            stim_t = stim_t.unsqueeze(0)
        pooled = self._pool_stim(stim_t.unsqueeze(1), B, 1, device, dtype)
        return pooled[:, 0]

    def step(
        self,
        h: Optional[Tensor],
        x_t: Tensor,
        stim_t: Optional[Tensor] = None,
        return_aux: bool = False,
        horizon_index: Optional[int] = None,
    ):
        B = x_t.shape[0]
        x_flat = self._flatten_state(x_t)
        if x_flat.shape[-1] != self.state_dim:
            raise ValueError(f"SimpleDeltaAR expected state_dim={self.state_dim}, got {x_flat.shape[-1]}")
        stim_vec = self._pool_stim_point(stim_t, B, x_t.device, x_t.dtype)
        inp = torch.cat([x_flat, stim_vec], dim=-1) if stim_vec.numel() else x_flat
        delta = self.net(inp)
        pred_flat = x_flat + delta if self.residual else delta
        pred = self._reshape_pred(pred_flat, x_t)
        if h is None:
            h = self.init_state(B, x_t.device, x_t.dtype)

        aux: Dict[str, Tensor] = {}
        if return_aux:
            z = torch.zeros((), device=x_t.device, dtype=x_t.dtype)
            aux = {
                "hidden_norm": z,
                "alpha_mean": z,
                "alpha_min": z,
                "alpha_max": z,
                "cyclic_delta_norm": delta.norm(dim=-1).mean().detach(),
                "cyclic_delta_x_ratio": (delta.norm(dim=-1) / x_flat.norm(dim=-1).clamp_min(1e-6)).mean().detach(),
                "cyclic_internal_change": z,
                "cyclic_msg_norm": z,
                "cyclic_edge_weight_mean": z,
                "cyclic_edge_weight_max": z,
            }
        if return_aux:
            return pred, h, aux
        return pred, h

    def forward(self, x: Tensor, stim: Optional[Tensor] = None):
        h = self.init_state(x.shape[0], x.device, x.dtype)
        pred, _h, aux = self.step(
            h,
            x[:, -1] if x.dim() >= 3 else x,
            None if stim is None else stim[:, -1],
            return_aux=True,
        )
        return pred, aux


DiagnosticFeedForwardGraphAR = FeedForwardGraphAR
DiagnosticSimpleDeltaAR = SimpleDeltaAR
