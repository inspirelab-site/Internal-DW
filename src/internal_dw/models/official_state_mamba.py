"""Official-state Mamba autoregressive model.

This is the *only* Mamba backbone used by the official-shadow patch.  It does
not maintain a rolling token buffer.  The external recurrent state is the real
Mamba inference state for every layer:

    conv_state_l : [B, d_inner, d_conv]
    ssm_state_l  : [B, d_inner, d_state]

The same class supports HCP vector states [B, D] and The Well fields [B, C, H, W]
by flattening frames internally and reshaping predictions back to the reference
frame shape.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import StimulusPoolingMixin
from .dual_wiener import DualWienerController


MambaLayerState = Tuple[torch.Tensor, torch.Tensor]
MambaStackState = Tuple[MambaLayerState, ...]


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


def detach_mamba_stack_state(h: MambaStackState) -> MambaStackState:
    return tuple((conv.detach(), ssm.detach()) for conv, ssm in h)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class StatefulMambaBlock(nn.Module):
    """One pure-PyTorch Mamba block in step/inference mode.

    Input/output token shape: [B, d_model].  The persistent recurrent state is
    (conv_state, ssm_state).  This follows the official Mamba update equations
    without fused CUDA kernels.
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

        if norm_type == "rms":
            self.norm = RMSNorm(self.d_model)
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(self.d_model)
        else:
            raise ValueError(f"Unknown norm_type={norm_type!r}; expected 'layer' or 'rms'.")

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

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(float(dropout))
        self.reset_parameters(dt_init_floor=dt_init_floor)

    def reset_parameters(self, dt_init_floor: float = 1e-4):
        dt = torch.exp(
            torch.empty(self.d_inner).uniform_(math.log(self.dt_min), math.log(self.dt_max))
        ).clamp(min=float(dt_init_floor))
        with torch.no_grad():
            self.dt_proj.bias.copy_(_inverse_softplus(dt))

    def init_state(self, batch_size: int, device=None, dtype=None) -> MambaLayerState:
        device = device if device is not None else next(self.parameters()).device
        dtype = dtype if dtype is not None else next(self.parameters()).dtype
        conv_state = torch.zeros(batch_size, self.d_inner, self.d_conv, device=device, dtype=dtype)
        ssm_state = torch.zeros(batch_size, self.d_inner, self.d_state, device=device, dtype=dtype)
        return conv_state, ssm_state

    def step(
        self,
        token: torch.Tensor,
        state: MambaLayerState,
        *,
        forward_branch_scale: float = 1.0,
        dual_wiener: Optional[DualWienerController] = None,
        route_horizon: int = -1,
        route_layer: int = -1,
        collect_diagnostics: bool = True,
    ):
        """Advance one Mamba block with either open or DW-routed gradients."""

        conv_state, ssm_state = state
        routed_pair = None
        if dual_wiener is None:
            residual = token
            branch_input = token
        else:
            residual, branch_input = dual_wiener.route_pair(
                token, route_horizon, route_layer
            )
            conv_state = dual_wiener.branch_state_input(
                conv_state, route_horizon, route_layer
            )
            ssm_state = dual_wiener.branch_state_input(
                ssm_state, route_horizon, route_layer
            )
            routed_pair = dual_wiener.current_pair(
                route_horizon, route_layer, token
            )
        token = self.norm(branch_input)

        x, z_gate = self.in_proj(token).chunk(2, dim=-1)
        conv_state = torch.roll(conv_state, shifts=-1, dims=-1).clone()
        conv_state[:, :, -1] = x
        weight = self.conv1d.weight.squeeze(1)
        x_conv = torch.sum(
            conv_state * weight.unsqueeze(0), dim=-1
        )
        if self.conv1d.bias is not None:
            x_conv = x_conv + self.conv1d.bias
        x_conv = self.act(x_conv)

        dt_raw, B_t, C_t = torch.split(
            self.x_proj(x_conv),
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        dt = F.softplus(self.dt_proj(dt_raw))
        A = -torch.exp(self.A_log.float()).to(
            dtype=x_conv.dtype, device=x_conv.device
        )
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
        dB = dt.unsqueeze(-1) * B_t.unsqueeze(1)
        ssm_state = (
            ssm_state * dA + x_conv.unsqueeze(-1) * dB
        )

        y = torch.einsum("bdn,bn->bd", ssm_state, C_t)
        y = y + self.D.to(
            dtype=y.dtype, device=y.device
        ).unsqueeze(0) * x_conv
        y = y * self.act(z_gate)
        branch = self.dropout(self.out_proj(y))
        output = residual + float(forward_branch_scale) * branch

        if not collect_diagnostics:
            return output, (conv_state, ssm_state), {}

        branch_norm = branch.detach().float().reshape(
            branch.shape[0], -1
        ).norm(dim=1).mean()
        residual_norm = residual.detach().float().reshape(
            residual.shape[0], -1
        ).norm(dim=1).mean()
        ratio = branch_norm / (residual_norm + 1e-8)
        gate = (
            routed_pair[1].detach()
            if routed_pair is not None
            else branch.new_tensor(1.0)
        )
        aux = {
            "dt_mean": dt.mean(),
            "dt_min": dt.min(),
            "dt_max": dt.max(),
            "ssm_state_norm": ssm_state.reshape(
                ssm_state.shape[0], -1
            ).norm(dim=1).mean(),
            "conv_state_norm": conv_state.reshape(
                conv_state.shape[0], -1
            ).norm(dim=1).mean(),
            "resgrad_gate": gate,
            "resgrad_branch_norm": branch_norm,
            "resgrad_residual_norm": residual_norm,
            "resgrad_branch_residual_ratio": ratio,
        }
        if routed_pair is not None:
            aux["dual_wiener_alpha"] = routed_pair[0].detach()
            aux["dual_wiener_m"] = routed_pair[1].detach()
        return output, (conv_state, ssm_state), aux


class OfficialStateMambaARModel(nn.Module, StimulusPoolingMixin):
    """Official-state Mamba AR model for vector or 2D-field states."""

    is_standard_autoregressive = True
    is_recurrent_state_ar = True
    is_official_state_mamba = True

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
        resgrad_routing: bool = False,
        resgrad_policy: str = "all",
        untie_groups: int = 1,
        dual_wiener_ema: float = 0.95,
        dual_wiener_residual_ema: float = 0.99,
        dual_wiener_warmup_batches: int = 8,
        dual_wiener_probe_every: int = 4,
        dual_wiener_min_probes: int = 1,
        dual_wiener_noise_model: str = "diagonal_gaussian",
        dual_wiener_max_horizon: int = 1024,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.input_dim = int(input_dim)
        self.state_shape = (
            tuple(int(value) for value in state_shape)
            if state_shape is not None
            else None
        )
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.has_external_input = bool(has_external_input)
        self.residual = bool(residual)
        self.task_type = (
            "field2d"
            if self.state_shape is not None
            and len(self.state_shape) == 3
            else "vector"
        )
        self.resgrad_routing = bool(resgrad_routing)
        self.resgrad_policy = str(resgrad_policy).lower()
        if self.resgrad_routing and self.resgrad_policy != "dualwiener":
            raise ValueError(
                "The released router supports resgrad_policy='dualwiener' "
                "only; disable routing for full BPTT."
            )
        if not self.resgrad_routing:
            self.resgrad_policy = "all"

        self.dual_wiener = None
        if self.resgrad_routing:
            self.dual_wiener = DualWienerController(
                state_dim=self.state_dim,
                depth=self.depth,
                max_horizon=int(dual_wiener_max_horizon),
                ema=float(dual_wiener_ema),
                residual_ema=float(dual_wiener_residual_ema),
                warmup_batches=int(dual_wiener_warmup_batches),
                probe_every=int(dual_wiener_probe_every),
                min_probes=int(dual_wiener_min_probes),
                noise_model=str(dual_wiener_noise_model),
            )

        self.resgrad_forward_branch_scale = 1.0
        self.resgrad_current_horizon = -1
        self.resgrad_current_total_horizon = -1
        self.untie_groups = max(1, int(untie_groups))

        input_size = self.state_dim + (
            self.input_dim if self.has_external_input else 0
        )
        self.in_proj = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

        def make_stack():
            return nn.ModuleList(
                [
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
                ]
            )

        if self.untie_groups == 1:
            self.blocks = make_stack()
            self.block_groups = None
        else:
            self.blocks = None
            self.block_groups = nn.ModuleList(
                [make_stack() for _ in range(self.untie_groups)]
            )
        self.out = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.state_dim),
        )

    def _active_stack(self, horizon_index: Optional[int] = None):
        """The depth-L block stack to use at this rollout step.

        untie_groups=1 -> always self.blocks. Otherwise pick group
        min(k*G//K, G-1) from horizon_index k and total_horizon K; if either is
        unknown (e.g. burn-in / one-step forward) fall back to group 0."""
        if self.block_groups is None:
            return self.blocks
        G = len(self.block_groups)
        k = int(horizon_index) if horizon_index is not None else int(getattr(self, "resgrad_current_horizon", -1))
        K = int(getattr(self, "resgrad_current_total_horizon", -1))
        if k < 0 or K <= 0:
            return self.block_groups[0]
        return self.block_groups[min((k * G) // K, G - 1)]

    def init_state(self, batch_size: int, device=None, dtype=None) -> MambaStackState:
        stack = self.blocks if self.block_groups is None else self.block_groups[0]
        return tuple(block.init_state(batch_size, device=device, dtype=dtype) for block in stack)

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


    def set_resgrad_context(self, horizon_index: Optional[int] = None, total_horizon: Optional[int] = None):
        self.resgrad_current_horizon = -1 if horizon_index is None else int(horizon_index)
        self.resgrad_current_total_horizon = -1 if total_horizon is None else int(total_horizon)

    # ---- dual-Wiener training hooks -------------------------------------
    # The trainer invokes these around the ordinary loss backward.  Probe
    # gradients are obtained with autograd.grad, so they never accumulate in
    # model parameters.
    def dual_wiener_begin_batch(self) -> None:
        if self.dual_wiener is not None:
            self.dual_wiener.begin_batch()

    def dual_wiener_probe_terms(self, prediction, target, horizon_index):
        if self.dual_wiener is None or not self.training:
            return None, None
        return self.dual_wiener.probe_terms(prediction, target, horizon_index)

    def dual_wiener_set_probe_losses(self, total, noise) -> None:
        if self.dual_wiener is not None:
            self.dual_wiener.set_probe_losses(total, noise)

    def dual_wiener_calibrate(self) -> bool:
        return bool(self.dual_wiener is not None and self.dual_wiener.calibrate())

    def dual_wiener_end_batch(self) -> None:
        if self.dual_wiener is not None:
            self.dual_wiener.end_batch()

    def dual_wiener_diagnostics(self, horizon: int):
        if self.dual_wiener is None:
            return {}
        return self.dual_wiener.diagnostics(horizon)

    def dual_wiener_export_state(self, horizon: int):
        if self.dual_wiener is None:
            return None
        return self.dual_wiener.export_state(horizon)

    def step(
        self,
        h: Optional[MambaStackState],
        x_t: torch.Tensor,
        stim_t: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        horizon_index: Optional[int] = None,
        total_horizon: Optional[int] = None,
        ratio_collector: Optional[list] = None,
    ):
        batch_size = x_t.shape[0]
        if h is None:
            h = self.init_state(
                batch_size, x_t.device, x_t.dtype
            )
        if horizon_index is not None or total_horizon is not None:
            self.set_resgrad_context(
                horizon_index=horizon_index,
                total_horizon=total_horizon,
            )

        z = self.make_token(x_t, stim_t)
        next_states: List[MambaLayerState] = []
        dt_means = []
        dt_mins = []
        dt_maxs = []
        ssm_norms = []
        conv_norms = []
        route_gates = []
        route_ratios = []
        branch_norms = []
        residual_norms = []
        dual_alphas = []
        dual_ms = []

        active_blocks = self._active_stack(horizon_index)
        route_horizon = (
            int(horizon_index)
            if horizon_index is not None
            else -1
        )
        for layer_index, (block, layer_state) in enumerate(
            zip(active_blocks, h)
        ):
            z, next_state, block_aux = block.step(
                z,
                layer_state,
                forward_branch_scale=self.resgrad_forward_branch_scale,
                dual_wiener=self.dual_wiener,
                route_horizon=route_horizon,
                route_layer=layer_index,
                collect_diagnostics=not getattr(self, "compact_recurrent_logging", False),
            )
            next_states.append(next_state)
            if not block_aux:
                continue
            dt_means.append(block_aux["dt_mean"])
            dt_mins.append(block_aux["dt_min"])
            dt_maxs.append(block_aux["dt_max"])
            ssm_norms.append(block_aux["ssm_state_norm"])
            conv_norms.append(block_aux["conv_state_norm"])
            route_gates.append(block_aux["resgrad_gate"])
            route_ratios.append(
                block_aux["resgrad_branch_residual_ratio"]
            )
            branch_norms.append(block_aux["resgrad_branch_norm"])
            residual_norms.append(
                block_aux["resgrad_residual_norm"]
            )
            if "dual_wiener_alpha" in block_aux:
                dual_alphas.append(block_aux["dual_wiener_alpha"])
                dual_ms.append(block_aux["dual_wiener_m"])

        h_next = tuple(next_states)
        predicted_flat = self.out(z)
        input_flat = self._flatten_state(x_t)
        if self.residual:
            predicted_flat = input_flat + predicted_flat
        prediction = self._reshape_pred(predicted_flat, x_t)

        if not return_aux:
            return prediction, h_next
        if getattr(self, "compact_recurrent_logging", False):
            return prediction, h_next, {}
        zero = predicted_flat.new_tensor(0.0)
        one = predicted_flat.new_tensor(1.0)
        aux = {
            "innovation_features": z.detach(),
            "h_next": h_next,
            "hidden_norm": (
                torch.stack(ssm_norms).mean()
                if ssm_norms
                else zero
            ),
            "mamba_ssm_state_norm": (
                torch.stack(ssm_norms).mean()
                if ssm_norms
                else zero
            ),
            "mamba_conv_state_norm": (
                torch.stack(conv_norms).mean()
                if conv_norms
                else zero
            ),
            "mamba_dt_mean": (
                torch.stack(dt_means).mean()
                if dt_means
                else zero
            ),
            "mamba_dt_min": (
                torch.stack(dt_mins).min() if dt_mins else zero
            ),
            "mamba_dt_max": (
                torch.stack(dt_maxs).max() if dt_maxs else zero
            ),
            "alpha_mean": (
                torch.stack(dt_means).mean()
                if dt_means
                else zero
            ),
            "alpha_min": (
                torch.stack(dt_mins).min() if dt_mins else zero
            ),
            "alpha_max": (
                torch.stack(dt_maxs).max() if dt_maxs else zero
            ),
            "resgrad_gate": (
                torch.stack(route_gates).mean()
                if route_gates
                else one
            ),
            "resgrad_gate_per_layer": (
                torch.stack(route_gates)
                if route_gates
                else predicted_flat.new_zeros(0)
            ),
            "resgrad_routing": predicted_flat.new_tensor(
                1.0 if self.dual_wiener is not None else 0.0
            ),
            "resgrad_branch_residual_ratio": (
                torch.stack(route_ratios).mean()
                if route_ratios
                else zero
            ),
            "resgrad_branch_norm": (
                torch.stack(branch_norms).mean()
                if branch_norms
                else zero
            ),
            "resgrad_residual_norm": (
                torch.stack(residual_norms).mean()
                if residual_norms
                else zero
            ),
            "dual_wiener_alpha": (
                torch.stack(dual_alphas).mean()
                if dual_alphas
                else one
            ),
            "dual_wiener_m": (
                torch.stack(dual_ms).mean()
                if dual_ms
                else one
            ),
        }
        return prediction, h_next, aux

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
