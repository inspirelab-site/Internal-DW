"""Predictive-coding Mamba autoregressive model.

This model keeps the same dependency-free official-state Mamba recurrent
backbone as ``official_mamba_state`` but changes the readout into a
predictive-coding / filtering form:

    prior      p_t = history-driven prediction from the recurrent token
    evidence   s_t = stimulus-conditioned evidence prediction
    error      eps = s_t - p_t
    gate       g_t = precision / Kalman-gain-like correction gate
    prediction x_{t+1} = p_t + g_t * eps

The external recurrent state is still the Mamba inference state for every
layer.  The class is intentionally API-compatible with OfficialStateMambaARModel
so the existing chunked-BPTT training/evaluation code can be reused.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from .official_state_mamba import (
    MambaLayerState,
    MambaStackState,
    StatefulMambaBlock,
    detach_mamba_stack_state,
)
from .simple_ar import _StimPoolMixin


class OfficialPredictiveCodingMambaARModel(nn.Module, _StimPoolMixin):
    """Predictive-coding readout on top of the official-state Mamba backbone."""

    is_standard_autoregressive = True
    is_recurrent_state_ar = True
    is_official_state_mamba = True
    is_predictive_coding_mamba = True

    def __init__(
        self,
        state_dim: int,
        input_dim: int = 0,
        state_shape: Optional[Sequence[int]] = None,
        hidden_dim: int = 512,
        depth: int = 4,
        dropout: float = 0.0,
        has_external_input: bool = True,
        residual: bool = True,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_dt_rank: int | str = "auto",
        mamba_dt_min: float = 1e-3,
        mamba_dt_max: float = 1e-1,
        mamba_norm_type: str = "layer",
        pc_gate_init: float = 0.35,
        pc_gate_scalar: bool = False,
        pc_gate_min: float = 0.0,
        pc_gate_max: float = 1.0,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.input_dim = int(input_dim)
        self.state_shape = tuple(int(x) for x in state_shape) if state_shape is not None else None
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.has_external_input = bool(has_external_input)
        self.residual = bool(residual)
        self.task_type = "field2d" if self.state_shape is not None and len(self.state_shape) == 3 else "vector"
        self.pc_gate_scalar = bool(pc_gate_scalar)
        self.pc_gate_min = float(pc_gate_min)
        self.pc_gate_max = float(pc_gate_max)
        if self.pc_gate_max < self.pc_gate_min:
            raise ValueError("pc_gate_max must be >= pc_gate_min")

        in_dim = self.state_dim + (self.input_dim if self.has_external_input else 0)
        self.in_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.blocks = nn.ModuleList([
            StatefulMambaBlock(
                d_model=self.hidden_dim,
                d_state=int(mamba_d_state),
                d_conv=int(mamba_d_conv),
                expand=int(mamba_expand),
                dt_rank=mamba_dt_rank,
                dt_min=float(mamba_dt_min),
                dt_max=float(mamba_dt_max),
                dropout=float(dropout),
                norm_type=mamba_norm_type,
            )
            for _ in range(max(1, self.depth))
        ])

        def make_state_head(out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, out_dim),
            )

        # prior_head predicts a history-driven next state.  With residual=True it
        # predicts a delta from x_t, matching the official_mamba_state baseline.
        self.prior_head = make_state_head(self.state_dim)
        # evidence_head predicts stimulus-conditioned sensory evidence.  It uses
        # the same recurrent token, so history can modulate stimulus processing.
        self.evidence_head = make_state_head(self.state_dim)
        gate_dim = 1 if self.pc_gate_scalar else self.state_dim
        self.gate_head = make_state_head(gate_dim)
        self.reset_pc_parameters(float(pc_gate_init))

    def reset_pc_parameters(self, gate_init: float = 0.35):
        gate_init = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
        logit = torch.logit(torch.tensor(gate_init, dtype=torch.float32))
        last = self.gate_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, float(logit))

    def init_state(self, batch_size: int, device=None, dtype=None) -> MambaStackState:
        return tuple(block.init_state(batch_size, device=device, dtype=dtype) for block in self.blocks)

    def detach_state(self, h: MambaStackState) -> MambaStackState:
        return detach_mamba_stack_state(h)

    def _flatten_state(self, x_t: torch.Tensor) -> torch.Tensor:
        return x_t.reshape(x_t.shape[0], -1)

    def _reshape_pred(self, pred_flat: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return pred_flat.reshape(ref.shape)

    def _pool_stim_point(self, stim_t: Optional[torch.Tensor], B: int, device, dtype) -> torch.Tensor:
        if not self.has_external_input:
            return torch.zeros(B, 0, device=device, dtype=dtype)
        if stim_t is None:
            return torch.zeros(B, self.input_dim, device=device, dtype=dtype)
        if stim_t.dim() == 1:
            stim_t = stim_t.unsqueeze(0)
        pooled = self._pool_stim(stim_t.unsqueeze(1), B, 1, device, dtype)
        return pooled[:, 0]

    def make_token(self, x_t: torch.Tensor, stim_t: Optional[torch.Tensor]) -> torch.Tensor:
        B = x_t.shape[0]
        x_flat = self._flatten_state(x_t)
        stim = self._pool_stim_point(stim_t, B, x_t.device, x_t.dtype)
        inp = torch.cat([x_flat, stim], dim=-1) if self.has_external_input else x_flat
        return self.in_proj(inp)

    def step(
        self,
        h: Optional[MambaStackState],
        x_t: torch.Tensor,
        stim_t: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        horizon_index: Optional[int] = None,
    ):
        B = x_t.shape[0]
        if h is None:
            h = self.init_state(B, x_t.device, x_t.dtype)
        token = self.make_token(x_t, stim_t)

        next_states: list[MambaLayerState] = []
        dt_means, dt_mins, dt_maxs = [], [], []
        ssm_norms, conv_norms = [], []
        z = token
        for block, layer_state in zip(self.blocks, h):
            z, next_state, aux_l = block.step(z, layer_state)
            next_states.append(next_state)
            dt_means.append(aux_l["dt_mean"])
            dt_mins.append(aux_l["dt_min"])
            dt_maxs.append(aux_l["dt_max"])
            ssm_norms.append(aux_l["ssm_state_norm"])
            conv_norms.append(aux_l["conv_state_norm"])
        h_next = tuple(next_states)

        x_flat = self._flatten_state(x_t)
        prior_delta_or_frame = self.prior_head(z)
        prior_flat = x_flat + prior_delta_or_frame if self.residual else prior_delta_or_frame

        evidence_delta_or_frame = self.evidence_head(z)
        evidence_flat = x_flat + evidence_delta_or_frame if self.residual else evidence_delta_or_frame

        gate_raw = self.gate_head(z)
        gate = torch.sigmoid(gate_raw)
        if self.pc_gate_scalar:
            gate = gate.expand(-1, self.state_dim)
        if self.pc_gate_min != 0.0 or self.pc_gate_max != 1.0:
            gate = self.pc_gate_min + (self.pc_gate_max - self.pc_gate_min) * gate

        eps_flat = evidence_flat - prior_flat
        corr_flat = gate * eps_flat
        pred_flat = prior_flat + corr_flat
        pred = self._reshape_pred(pred_flat, x_t)

        if return_aux:
            prior = self._reshape_pred(prior_flat, x_t)
            evidence = self._reshape_pred(evidence_flat, x_t)
            correction = self._reshape_pred(corr_flat, x_t)
            eps = self._reshape_pred(eps_flat, x_t)
            aux = {
                "h_next": h_next,
                "hidden_norm": torch.stack(ssm_norms).mean() if ssm_norms else pred_flat.new_tensor(0.0),
                "mamba_ssm_state_norm": torch.stack(ssm_norms).mean() if ssm_norms else pred_flat.new_tensor(0.0),
                "mamba_conv_state_norm": torch.stack(conv_norms).mean() if conv_norms else pred_flat.new_tensor(0.0),
                "mamba_dt_mean": torch.stack(dt_means).mean() if dt_means else pred_flat.new_tensor(0.0),
                "mamba_dt_min": torch.stack(dt_mins).min() if dt_mins else pred_flat.new_tensor(0.0),
                "mamba_dt_max": torch.stack(dt_maxs).max() if dt_maxs else pred_flat.new_tensor(0.0),
                "alpha_mean": torch.stack(dt_means).mean() if dt_means else pred_flat.new_tensor(0.0),
                "alpha_min": torch.stack(dt_mins).min() if dt_mins else pred_flat.new_tensor(0.0),
                "alpha_max": torch.stack(dt_maxs).max() if dt_maxs else pred_flat.new_tensor(0.0),
                "pc_prior_pred": prior,
                "pc_evidence_pred": evidence,
                "pc_correction": correction,
                "pc_error": eps,
                "pc_gate": gate,
                "pc_gate_mean": gate.mean(),
                "pc_gate_min": gate.min(),
                "pc_gate_max": gate.max(),
                "pc_error_norm": eps_flat.norm(dim=-1).mean(),
                "pc_correction_norm": corr_flat.norm(dim=-1).mean(),
            }
            return pred, h_next, aux
        return pred, h_next

    def burn_in(self, x_seq: torch.Tensor, stim_seq: Optional[torch.Tensor] = None, h0: Optional[MambaStackState] = None, detach: bool = False):
        B, T = x_seq.shape[:2]
        h = h0 if h0 is not None else self.init_state(B, x_seq.device, x_seq.dtype)
        ctx = torch.no_grad() if detach else torch.enable_grad()
        with ctx:
            for j in range(T):
                stim_j = stim_seq[:, j] if stim_seq is not None else None
                _, h = self.step(h, x_seq[:, j], stim_j, return_aux=False)
        return self.detach_state(h) if detach else h

    def predict_frame_from_history(self, history: torch.Tensor, stim_window: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, W = history.shape[:2]
        h = self.init_state(B, history.device, history.dtype)
        stim = self._pool_stim(stim_window, B, W, history.device, history.dtype) if self.has_external_input else None
        pred = None
        for j in range(W):
            stim_j = stim[:, j] if stim is not None else None
            pred, h = self.step(h, history[:, j], stim_j, return_aux=False)
        if pred is None:
            raise ValueError("history must contain at least one frame")
        return pred

    def step_history(self, history: torch.Tensor, stim_window: Optional[torch.Tensor] = None) -> torch.Tensor:
        frame = self.predict_frame_from_history(history, stim_window)
        return torch.cat([history[:, 1:], frame.unsqueeze(1)], dim=1)

    def forward(self, stim_window: Optional[torch.Tensor], history: torch.Tensor, return_aux: bool = False):
        frame = self.predict_frame_from_history(history, stim_window)
        pred = frame.unsqueeze(1)
        if return_aux:
            return pred, {"pred_frame": frame}
        return pred
