"""Official Mamba with fixed macro A, local diagonal perturbation, and a
homogenization-derived shared cell corrector.

This module keeps the official-state Mamba step structure: input projection,
causal depthwise convolution, selective B/C/dt projections, dt-projected input
injection, C readout, D skip, gate, output projection, residual connection, and
the external inference state ``(conv_state, ssm_state)`` for every layer.

The internal diagonal SSM transition is changed from the original Mamba

    h_{t+1} = dA_t * h_t + dB_t * x_t

to

    h_{t+1} = (A + DeltaA_t) * h_t + dB_t * x_t.

Here A is a learned time-independent discrete diagonal macro transition and
DeltaA_t is not predicted by an independent head in this hard version.
Instead, the model learns a shared bounded corrector C_t = C_eta(xi_t), and
the local transition is parameterized so that the exact diagonal conjugacy
equation holds by construction:

    A + DeltaA_t = A * (1 + C_{t+1}) / (1 + C_t),
    (A + DeltaA_t) * (1 + C_t) = A * (1 + C_{t+1}),

which is the diagonal version of

    (A + DeltaA(xi_t))(I + C(xi_t)) = (I + C(T xi_t)) A.

The fixed-A KG term in this file is *not* a matrix-power norm.  When used by the
BPTT trainer, it is computed on empirical SSM hidden-error directions e_1:

    mean_k ||A^k e_1||^2 / ||e_1||^2.

Thus KG constrains A on the actual error space sampled by the rollout, while
the hard corrector parameterization constrains DeltaA_t as a bounded
fast-coordinate corrector.  The cell residual is kept only as a diagnostic.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .simple_ar import _StimPoolMixin
from .official_state_mamba import (
    RMSNorm,
    _inverse_softplus,
    MambaLayerState,
    MambaStackState,
    detach_mamba_stack_state,
)


class FixedAPerturbMambaBlock(nn.Module):
    """One official-style Mamba block with fixed-A + diagonal perturbation.

    ``A`` and ``DeltaA_t`` have shape [d_inner, d_state] and
    [B, d_inner, d_state], respectively.  In this hard-conjugacy version,
    only the bounded corrector C_t is learned; DeltaA_t is generated from
    C_t and C_{t+1} by the exact diagonal conjugacy equation.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        conv_bias: bool = True,
        dropout: float = 0.0,
        norm_type: str = "layer",
        fixeda_init: float = 0.98,
        fixeda_max: float = 0.999,
        perturb_eps: float = 0.02,
        perturb_bound: str = "tanh",
        perturb_hidden_mult: int = 1,
        corrector_rho: float = 0.5,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)
        self.fixeda_max = float(fixeda_max)
        self.perturb_eps = float(perturb_eps)
        self.perturb_bound = str(perturb_bound)
        self.corrector_rho = float(corrector_rho)

        if norm_type == "rms":
            self.norm = RMSNorm(self.d_model)
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(self.d_model)
        else:
            raise ValueError(f"Unknown norm_type={norm_type!r}; expected 'layer' or 'rms'.")

        # Official Mamba surrounding modules.
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            bias=conv_bias,
            padding=0,
        )
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(float(dropout))

        # Learned discrete diagonal macro transition A in [0, fixeda_max].
        init_a = float(fixeda_init)
        init_a = max(1e-4, min(init_a, self.fixeda_max - 1e-4))
        init_logit = math.log(init_a / (self.fixeda_max - init_a))
        self.A_logit = nn.Parameter(torch.full((self.d_inner, self.d_state), float(init_logit)))

        # Shared fast-context trunk.
        # Hard-conjugacy version: only learn the corrector C_t.
        # DeltaA_t is not predicted by an independent head.
        ph = max(self.d_inner, int(perturb_hidden_mult) * self.d_inner)
        self.fast_norm = nn.LayerNorm(2 * self.d_inner)
        self.fast_trunk = nn.Sequential(
            nn.Linear(2 * self.d_inner, ph),
            nn.GELU(),
        )
        self.corrector_head = nn.Linear(ph, self.d_inner * self.d_state)

        if not (0.0 <= self.corrector_rho < 1.0):
            raise ValueError(
                f"fixeda_corrector_rho must satisfy 0 <= rho < 1, got {self.corrector_rho}"
            )

        # Start exactly as fixed-A: C_t = 0, hence DeltaA_t = 0.
        nn.init.zeros_(self.corrector_head.weight)
        nn.init.zeros_(self.corrector_head.bias)

        self.reset_parameters(dt_init_floor=dt_init_floor)

    def reset_parameters(self, dt_init_floor: float = 1e-4):
        dt = torch.exp(
            torch.empty(self.d_inner).uniform_(math.log(self.dt_min), math.log(self.dt_max))
        ).clamp(min=float(dt_init_floor))
        with torch.no_grad():
            self.dt_proj.bias.copy_(_inverse_softplus(dt))

    def fixed_A(self, dtype=None, device=None) -> torch.Tensor:
        A = self.fixeda_max * torch.sigmoid(self.A_logit)
        if dtype is not None or device is not None:
            A = A.to(
                dtype=dtype if dtype is not None else A.dtype,
                device=device if device is not None else A.device,
            )
        return A

    def init_state(self, batch_size: int, device=None, dtype=None):
        device = device if device is not None else next(self.parameters()).device
        dtype = dtype if dtype is not None else next(self.parameters()).dtype
        conv_state = torch.zeros(batch_size, self.d_inner, self.d_conv, device=device, dtype=dtype)
        ssm_state = torch.zeros(batch_size, self.d_inner, self.d_state, device=device, dtype=dtype)
        # Hard corrector state C_t. Initialize C_0 = 0.
        corrector_state = torch.zeros(batch_size, self.d_inner, self.d_state, device=device, dtype=dtype)
        return conv_state, ssm_state, corrector_state

    def _next_corrector(self, x_conv: torch.Tensor, ssm_state: torch.Tensor) -> torch.Tensor:
        """Predict bounded C_{t+1} from the current fast context.

        Hard diagonal conjugacy uses stored C_t and newly predicted C_{t+1}:

            A + DeltaA_t = A * (1 + C_{t+1}) / (1 + C_t).

        Thus DeltaA_t is a derived quantity, not an independent learned head.
        """
        h_summary = ssm_state.mean(dim=-1)
        feat = torch.cat([x_conv, h_summary], dim=-1)
        z = self.fast_trunk(self.fast_norm(feat))
        raw_c = self.corrector_head(z).view(x_conv.shape[0], self.d_inner, self.d_state)
        # Since corrector_rho < 1, 1 + C_t is bounded away from zero.
        c_next = self.corrector_rho * torch.tanh(raw_c)
        return c_next

    def step(self, token: torch.Tensor, state):
        # Backward-compatible: old states may contain only (conv_state, ssm_state).
        if len(state) == 2:
            conv_state, ssm_state = state
            c_prev = torch.zeros_like(ssm_state)
        else:
            conv_state, ssm_state, c_prev = state
        residual = token
        token = self.norm(token)

        xz = self.in_proj(token)
        x, z_gate = xz.chunk(2, dim=-1)

        conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
        conv_state = conv_state.clone()
        conv_state[:, :, -1] = x
        weight = self.conv1d.weight.squeeze(1)
        x_conv = torch.sum(conv_state * weight.unsqueeze(0), dim=-1)
        if self.conv1d.bias is not None:
            x_conv = x_conv + self.conv1d.bias
        x_conv = self.act(x_conv)

        x_dbl = self.x_proj(x_conv)
        dt_raw, B_t, C_t = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))
        dB = dt.unsqueeze(-1) * B_t.unsqueeze(1)

        A = self.fixed_A(dtype=x_conv.dtype, device=x_conv.device)
        A_view = A.unsqueeze(0)

        # Hard diagonal conjugacy.
        # (A + DeltaA_t) * (1 + C_t) = A * (1 + C_{t+1})
        c_next = self._next_corrector(x_conv, ssm_state)
        A_eff = A_view * (1.0 + c_next) / (1.0 + c_prev)
        delta_A = A_eff - A_view

        base = ssm_state * A_view
        r_t = ssm_state * delta_A
        input_term = x_conv.unsqueeze(-1) * dB
        ssm_state = ssm_state * A_eff + input_term

        y = torch.einsum("bdn,bn->bd", ssm_state, C_t)
        y = y + self.D.to(dtype=y.dtype, device=y.device).unsqueeze(0) * x_conv
        y = y * self.act(z_gate)
        out = self.out_proj(y)
        out = residual + self.dropout(out)

        eps = 1e-8
        cell_residual = (A_view + delta_A) * (1.0 + c_prev) - A_view * (1.0 + c_next)
        aux = {
            "dt_mean": dt.mean(),
            "dt_min": dt.min(),
            "dt_max": dt.max(),
            "ssm_state_norm": ssm_state.reshape(ssm_state.shape[0], -1).norm(dim=1).mean(),
            "conv_state_norm": conv_state.reshape(conv_state.shape[0], -1).norm(dim=1).mean(),
            "fixeda_r": r_t,
            "fixeda_base": base,
            "fixeda_A": A,
            "fixeda_delta_A": delta_A,
            "fixeda_C_prev": c_prev,
            "fixeda_C_next": c_next,
            "fixeda_cell_residual": cell_residual,
            "fixeda_delta_rel": r_t.reshape(r_t.shape[0], -1).norm(dim=1).mean()
            / (base.reshape(base.shape[0], -1).norm(dim=1).mean() + eps),
            "fixeda_delta_abs": delta_A.abs().mean(),
            "fixeda_C_abs": c_next.abs().mean(),
            "fixeda_A_mean": A.mean(),
            "fixeda_A_max": A.max(),
        }
        return out, (conv_state, ssm_state, c_next), aux


class OfficialMambaFixedAPerturbARModel(nn.Module, _StimPoolMixin):
    """Official-state Mamba AR model with fixed-A perturbation transition."""

    is_standard_autoregressive = True
    is_recurrent_state_ar = True
    is_official_state_mamba = True
    is_mamba_fixeda_perturb = True

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
        fixeda_init: float = 0.98,
        fixeda_max: float = 0.999,
        fixeda_perturb_eps: float = 0.02,
        fixeda_perturb_bound: str = "tanh",
        fixeda_perturb_hidden_mult: int = 1,
        fixeda_kg_weight: float = 0.0,
        fixeda_kg_horizon: int = 0,
        fixeda_cell_weight: float = 0.0,
        fixeda_corrector_rho: float = 0.5,
        # Deprecated aliases kept only so old scripts/checkpoints do not crash.
        fixeda_corr_weight: float = 0.0,
        fixeda_mean_weight: float = 0.0,
        fixeda_delta_weight: float = 0.0,
        fixeda_corr_solve_lambda: float = 1e-2,
        fixeda_corr_detach: bool = True,
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

        self.fixeda_kg_weight = float(fixeda_kg_weight)
        self.fixeda_kg_horizon = int(fixeda_kg_horizon)
        self.fixeda_cell_weight = float(fixeda_cell_weight)
        self.fixeda_corrector_rho = float(fixeda_corrector_rho)

        # Deprecated bookkeeping fields. They are not used in the strict cell implementation.
        self.fixeda_corr_weight = float(fixeda_corr_weight)
        self.fixeda_mean_weight = float(fixeda_mean_weight)
        self.fixeda_delta_weight = float(fixeda_delta_weight)
        self.fixeda_corr_solve_lambda = float(fixeda_corr_solve_lambda)
        self.fixeda_corr_detach = bool(fixeda_corr_detach)

        in_dim = self.state_dim + (self.input_dim if self.has_external_input else 0)
        self.in_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.blocks = nn.ModuleList([
            FixedAPerturbMambaBlock(
                d_model=self.hidden_dim,
                d_state=int(mamba_d_state),
                d_conv=int(mamba_d_conv),
                expand=int(mamba_expand),
                dt_rank=mamba_dt_rank,
                dt_min=float(mamba_dt_min),
                dt_max=float(mamba_dt_max),
                dropout=float(dropout),
                norm_type=mamba_norm_type,
                fixeda_init=float(fixeda_init),
                fixeda_max=float(fixeda_max),
                perturb_eps=float(fixeda_perturb_eps),
                perturb_bound=str(fixeda_perturb_bound),
                perturb_hidden_mult=int(fixeda_perturb_hidden_mult),
                corrector_rho=float(fixeda_corrector_rho),
            )
            for _ in range(max(1, self.depth))
        ])
        self.out = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.state_dim),
        )

    def init_state(self, batch_size: int, device=None, dtype=None) -> MambaStackState:
        return tuple(block.init_state(batch_size, device=device, dtype=dtype) for block in self.blocks)

    def detach_state(self, h):
        if torch.is_tensor(h):
            return h.detach()
        if isinstance(h, tuple):
            return tuple(self.detach_state(x) for x in h)
        if isinstance(h, list):
            return [self.detach_state(x) for x in h]
        if isinstance(h, dict):
            return {k: self.detach_state(v) for k, v in h.items()}
        return h

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

        next_states: List[MambaLayerState] = []
        dt_means, dt_mins, dt_maxs = [], [], []
        ssm_norms, conv_norms = [], []
        fixeda_rs, fixeda_bases, fixeda_As = [], [], []
        fixeda_delta_As = []
        fixeda_C_prevs, fixeda_C_nexts, fixeda_cell_residuals = [], [], []
        fixeda_delta_rels, fixeda_delta_abs, fixeda_C_abs = [], [], []
        fixeda_A_means, fixeda_A_maxs = [], []

        z = token
        for block, layer_state in zip(self.blocks, h):
            z, next_state, aux_l = block.step(z, layer_state)
            next_states.append(next_state)
            dt_means.append(aux_l["dt_mean"])
            dt_mins.append(aux_l["dt_min"])
            dt_maxs.append(aux_l["dt_max"])
            ssm_norms.append(aux_l["ssm_state_norm"])
            conv_norms.append(aux_l["conv_state_norm"])
            fixeda_rs.append(aux_l["fixeda_r"])
            fixeda_bases.append(aux_l["fixeda_base"])
            fixeda_As.append(aux_l["fixeda_A"])
            fixeda_delta_As.append(aux_l["fixeda_delta_A"])
            fixeda_C_prevs.append(aux_l["fixeda_C_prev"])
            fixeda_C_nexts.append(aux_l["fixeda_C_next"])
            fixeda_cell_residuals.append(aux_l["fixeda_cell_residual"])
            fixeda_delta_rels.append(aux_l["fixeda_delta_rel"])
            fixeda_delta_abs.append(aux_l["fixeda_delta_abs"])
            fixeda_C_abs.append(aux_l["fixeda_C_abs"])
            fixeda_A_means.append(aux_l["fixeda_A_mean"])
            fixeda_A_maxs.append(aux_l["fixeda_A_max"])
        h_next = tuple(next_states)

        delta_or_frame = self.out(z)
        x_flat = self._flatten_state(x_t)
        pred_flat = x_flat + delta_or_frame if self.residual else delta_or_frame
        pred = self._reshape_pred(pred_flat, x_t)

        if return_aux:
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
                "fixeda_r": torch.stack(fixeda_rs, dim=1),                # [B,L,D,N]
                "fixeda_base": torch.stack(fixeda_bases, dim=1),          # [B,L,D,N]
                "fixeda_A": torch.stack(fixeda_As, dim=0),               # [L,D,N]
                "fixeda_delta_A": torch.stack(fixeda_delta_As, dim=1),          # [B,L,D,N]
                "fixeda_C_prev": torch.stack(fixeda_C_prevs, dim=1),            # [B,L,D,N]
                "fixeda_C_next": torch.stack(fixeda_C_nexts, dim=1),            # [B,L,D,N]
                "fixeda_cell_residual": torch.stack(fixeda_cell_residuals, dim=1),  # [B,L,D,N]
                "fixeda_delta_rel": torch.stack(fixeda_delta_rels).mean(),
                "fixeda_delta_abs": torch.stack(fixeda_delta_abs).mean(),
                "fixeda_C_abs": torch.stack(fixeda_C_abs).mean(),
                "fixeda_A_mean": torch.stack(fixeda_A_means).mean(),
                "fixeda_A_max": torch.stack(fixeda_A_maxs).max(),
            }
            return pred, h_next, aux
        return pred, h_next

    def burn_in(
        self,
        x_seq: torch.Tensor,
        stim_seq: Optional[torch.Tensor] = None,
        h0: Optional[MambaStackState] = None,
        detach: bool = False,
    ):
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

    def fixeda_A_stack(self) -> torch.Tensor:
        return torch.stack([b.fixed_A() for b in self.blocks], dim=0)  # [L,d_inner,d_state]

    def fixeda_kg_loss_from_error(self, e1: Optional[torch.Tensor]) -> torch.Tensor:
        """Empirical-error KG loss: mean_k ||A^k e1||^2 / ||e1||^2.

        Args:
            e1: [B,L,d_inner,d_state] empirical SSM hidden-error direction.
                This is detached by the caller/inside this method so KG trains A
                on sampled error directions rather than shrinking the error by
                backpropagating into the rollout path.
        """
        A = self.fixeda_A_stack()
        if e1 is None:
            return A.new_tensor(0.0)
        K = int(self.fixeda_kg_horizon)
        if K <= 0:
            return A.new_tensor(0.0)
        e = e1.detach().to(dtype=A.dtype, device=A.device)
        if e.numel() == 0:
            return A.new_tensor(0.0)
        A_view = A.unsqueeze(0)
        denom = e.pow(2).sum(dim=(1, 2, 3)).clamp_min(1e-8)  # [B]
        cur = e
        terms = []
        for _ in range(K):
            cur = cur * A_view
            ratio = cur.pow(2).sum(dim=(1, 2, 3)) / denom
            terms.append(ratio.mean())
        return torch.stack(terms).mean() if terms else A.new_tensor(0.0)

    def fixeda_segment_regularization(
        self,
        aux_steps: List[Dict[str, torch.Tensor]],
        hidden_errors: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute strict cell loss and empirical-error KG for one BPTT segment.

        Cell loss is local in time: it averages adjacent pairs inside the current
        BPTT segment and never uses the long KG horizon as a model unroll length.
        """
        A_stack = self.fixeda_A_stack()
        z = A_stack.new_tensor(0.0)
        if not aux_steps:
            return {
                "kg": z,
                "cell": z,
                "cell_pairs": z,
                "cell_res_abs": z,
                "corrector_abs": z,
                "corrector_max": z,
                "corrector_mean_abs": z,
                "delta_rel": z,
                "delta_abs": z,
                "delta_mean_abs": z,
                "A_mean": A_stack.mean().detach(),
                "A_max": A_stack.max().detach(),
            }

        delta_seq = torch.stack([a["fixeda_delta_A"] for a in aux_steps], dim=1)          # [B,T,L,D,N]
        c_prev_seq = torch.stack([a["fixeda_C_prev"] for a in aux_steps], dim=1)          # [B,T,L,D,N]
        c_next_seq = torch.stack([a["fixeda_C_next"] for a in aux_steps], dim=1)          # [B,T,L,D,N]
        cell_res_seq = torch.stack([a["fixeda_cell_residual"] for a in aux_steps], dim=1) # [B,T,L,D,N]
        A = aux_steps[-1]["fixeda_A"].to(dtype=delta_seq.dtype, device=delta_seq.device)
        eps = delta_seq.new_tensor(1e-8)

        # In the hard parameterization, the exact cell residual should be zero
        # up to floating-point roundoff.  Cell is kept as a diagnostic/loss hook.
        denom = delta_seq.pow(2).sum().clamp_min(float(eps))
        cell = cell_res_seq.pow(2).sum() / denom
        cell_res_abs = cell_res_seq.abs().mean().detach()
        cell_pairs = delta_seq.new_tensor(float(delta_seq.shape[1]))

        c_seq = c_next_seq

        e1 = hidden_errors[0] if hidden_errors else None
        kg = self.fixeda_kg_loss_from_error(e1)

        return {
            "kg": kg,
            "cell": cell,
            "cell_pairs": cell_pairs.detach(),
            "cell_res_abs": cell_res_abs,
            "corrector_abs": c_seq.abs().mean().detach(),
            "corrector_max": c_seq.abs().max().detach(),
            "corrector_mean_abs": c_seq.mean(dim=(0, 1)).abs().mean().detach(),
            "delta_rel": torch.stack([a["fixeda_delta_rel"] for a in aux_steps]).mean().detach(),
            "delta_abs": torch.stack([a["fixeda_delta_abs"] for a in aux_steps]).mean().detach(),
            "delta_mean_abs": delta_seq.mean(dim=(0, 1)).abs().mean().detach(),
            "A_mean": A.mean().detach(),
            "A_max": A.max().detach(),
        }
