"""Clean autoregressive baselines for testing compiled backward-graph losses.

These models intentionally avoid AE/latent/token machinery.  They expose the
same standard AR interface as FNO/U-Net models:

    forward(stim_window, history) -> [B,1,...]
    step_history(history, stim_window=None) -> shifted history with prediction

Field models are for The Well. Vector models are for HCP/fMRI.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dual_wiener import DualWienerController


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, groups: int = 8):
        super().__init__()
        pad = kernel_size // 2
        g = max(1, min(int(groups), int(out_ch)))
        while out_ch % g != 0 and g > 1:
            g -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class CNNFieldARModel(nn.Module):
    """Small residual CNN next-frame predictor for 2D fields."""

    is_standard_autoregressive = True
    task_type = "field2d"

    def __init__(
        self,
        field_channels: int,
        window_size: int = 4,
        hidden_channels: int = 96,
        depth: int = 5,
        groups: int = 8,
        use_grid: bool = True,
        normalize: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        self.field_channels = int(field_channels)
        self.window_size = int(window_size)
        self.hidden_channels = int(hidden_channels)
        self.depth = int(depth)
        self.groups = int(groups)
        self.use_grid = bool(use_grid)
        self.normalize = bool(normalize)
        self.residual = bool(residual)
        in_ch = self.window_size * self.field_channels + (2 if self.use_grid else 0)
        layers = [_ConvBlock(in_ch, hidden_channels, groups=groups)]
        for _ in range(max(0, depth - 2)):
            layers.append(_ConvBlock(hidden_channels, hidden_channels, groups=groups))
        layers.append(nn.Conv2d(hidden_channels, self.field_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)
        self.register_buffer("state_mean", torch.zeros(1, 1, self.field_channels, 1, 1))
        self.register_buffer("state_std", torch.ones(1, 1, self.field_channels, 1, 1))
        self.normalizer_fitted = False

    def set_state_normalizer(self, mean: torch.Tensor, std: torch.Tensor):
        self.state_mean.copy_(mean.detach().float().view(1, 1, self.field_channels, 1, 1).to(self.state_mean.device))
        self.state_std.copy_(std.detach().float().clamp_min(1e-6).view(1, 1, self.field_channels, 1, 1).to(self.state_std.device))
        self.normalizer_fitted = True

    def _grid(self, b, h, w, device, dtype):
        x = torch.linspace(0, 1, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        y = torch.linspace(0, 1, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([x, y], dim=1)

    def _normalize_history(self, history):
        return (history - self.state_mean) / self.state_std if self.normalize else history

    def _decode(self, frame):
        return frame * self.state_std[:, 0] + self.state_mean[:, 0] if self.normalize else frame

    def predict_frame_from_history(self, history: torch.Tensor) -> torch.Tensor:
        b, win, c, h, w = history.shape
        if win != self.window_size or c != self.field_channels:
            raise ValueError(f"Expected history [B,{self.window_size},{self.field_channels},H,W], got {tuple(history.shape)}")
        hn = self._normalize_history(history)
        x = hn.reshape(b, win * c, h, w)
        if self.use_grid:
            x = torch.cat([x, self._grid(b, h, w, x.device, x.dtype)], dim=1)
        delta_or_frame = self.net(x)
        if self.residual:
            last_norm = hn[:, -1]
            pred_norm = last_norm + delta_or_frame
        else:
            pred_norm = delta_or_frame
        return self._decode(pred_norm)

    def step_history(self, history: torch.Tensor, stim_window: Optional[torch.Tensor] = None) -> torch.Tensor:
        frame = self.predict_frame_from_history(history)
        return torch.cat([history[:, 1:], frame.unsqueeze(1)], dim=1)

    def forward(self, stim_window: Optional[torch.Tensor], history: torch.Tensor, return_aux: bool = False):
        frame = self.predict_frame_from_history(history)
        pred = frame.unsqueeze(1)
        if return_aux:
            return pred, {"pred_frame": frame}
        return pred


class ConvGRUCell2d(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_channels = int(hidden_channels)
        self.gates = nn.Conv2d(input_channels + hidden_channels, 2 * hidden_channels, kernel_size, padding=pad)
        self.cand = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=pad)

    def forward(self, x, h):
        if h is None:
            h = x.new_zeros(x.shape[0], self.hidden_channels, x.shape[-2], x.shape[-1])
        z, r = self.gates(torch.cat([x, h], dim=1)).chunk(2, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        n = torch.tanh(self.cand(torch.cat([x, r * h], dim=1)))
        return (1.0 - z) * n + z * h


class ConvGRUFieldARModel(nn.Module):
    """ConvGRU baseline for The Well 2D fields."""

    is_standard_autoregressive = True
    task_type = "field2d"

    def __init__(
        self,
        field_channels: int,
        window_size: int = 4,
        hidden_channels: int = 96,
        groups: int = 8,
        use_grid: bool = True,
        normalize: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        self.field_channels = int(field_channels)
        self.window_size = int(window_size)
        self.hidden_channels = int(hidden_channels)
        self.use_grid = bool(use_grid)
        self.normalize = bool(normalize)
        self.residual = bool(residual)
        in_ch = self.field_channels + (2 if self.use_grid else 0)
        self.in_proj = _ConvBlock(in_ch, hidden_channels, groups=groups)
        self.cell = ConvGRUCell2d(hidden_channels, hidden_channels)
        self.out = nn.Sequential(_ConvBlock(hidden_channels, hidden_channels, groups=groups), nn.Conv2d(hidden_channels, self.field_channels, 3, padding=1))
        self.register_buffer("state_mean", torch.zeros(1, 1, self.field_channels, 1, 1))
        self.register_buffer("state_std", torch.ones(1, 1, self.field_channels, 1, 1))
        self.normalizer_fitted = False

    def set_state_normalizer(self, mean: torch.Tensor, std: torch.Tensor):
        self.state_mean.copy_(mean.detach().float().view(1, 1, self.field_channels, 1, 1).to(self.state_mean.device))
        self.state_std.copy_(std.detach().float().clamp_min(1e-6).view(1, 1, self.field_channels, 1, 1).to(self.state_std.device))
        self.normalizer_fitted = True

    def _grid(self, b, h, w, device, dtype):
        x = torch.linspace(0, 1, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        y = torch.linspace(0, 1, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([x, y], dim=1)

    def _normalize_history(self, history):
        return (history - self.state_mean) / self.state_std if self.normalize else history

    def _decode(self, frame):
        return frame * self.state_std[:, 0] + self.state_mean[:, 0] if self.normalize else frame

    def predict_frame_from_history(self, history: torch.Tensor) -> torch.Tensor:
        b, win, c, h, w = history.shape
        hn = self._normalize_history(history)
        grid = self._grid(b, h, w, history.device, history.dtype) if self.use_grid else None
        h_state = None
        for t in range(win):
            x = hn[:, t]
            if grid is not None:
                x = torch.cat([x, grid], dim=1)
            h_state = self.cell(self.in_proj(x), h_state)
        delta_or_frame = self.out(h_state)
        pred_norm = hn[:, -1] + delta_or_frame if self.residual else delta_or_frame
        return self._decode(pred_norm)

    def step_history(self, history: torch.Tensor, stim_window: Optional[torch.Tensor] = None) -> torch.Tensor:
        frame = self.predict_frame_from_history(history)
        return torch.cat([history[:, 1:], frame.unsqueeze(1)], dim=1)

    def forward(self, stim_window: Optional[torch.Tensor], history: torch.Tensor, return_aux: bool = False):
        frame = self.predict_frame_from_history(history)
        pred = frame.unsqueeze(1)
        if return_aux:
            return pred, {"pred_frame": frame}
        return pred


class _StimPoolMixin:
    def _pool_stim(self, stim_window: Optional[torch.Tensor], B: int, T: int, device, dtype) -> torch.Tensor:
        if not self.has_external_input:
            return torch.zeros(B, T, 0, device=device, dtype=dtype)
        if stim_window is None:
            return torch.zeros(B, T, self.input_dim, device=device, dtype=dtype)
        x = stim_window.float()
        if x.dim() == 2:
            x = x.unsqueeze(1).expand(-1, T, -1)
        if x.dim() > 3:
            x = x.reshape(x.shape[0], x.shape[1], -1)
        flat = x.shape[-1]
        if flat != self.input_dim and self.input_dim > 0 and flat % self.input_dim == 0:
            x = x.reshape(x.shape[0], x.shape[1], flat // self.input_dim, self.input_dim).mean(dim=2)
        if x.shape[-1] != self.input_dim:
            # Last-resort adaptive pooling so clean baselines do not explode on tokenized HCP features.
            x = F.adaptive_avg_pool1d(x.reshape(-1, 1, x.shape[-1]), self.input_dim).reshape(x.shape[0], x.shape[1], self.input_dim)
        if x.shape[1] != T:
            if x.shape[1] > T:
                x = x[:, -T:]
            else:
                pad = x[:, -1:].expand(-1, T - x.shape[1], -1)
                x = torch.cat([x, pad], dim=1)
        return x.to(device=device, dtype=dtype)


class GatedTransformerEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer with stopgrad-gated residual branches.

    nn.TransformerEncoderLayer's residual connections are buried inside a
    black-box forward (no hook point for gating), so this is a from-scratch
    equivalent (same pre-norm math as nn.TransformerEncoderLayer(norm_first=
    True)) exposing both residual sub-blocks -- self-attention and the
    feedforward branch -- to the same stopgrad-gate construction used by
    StatefulMambaBlock.step, unet_field.ResidualBlock, and
    ResidualMLPBlock. Each sub-block is gated independently but with the
    same policy-resolved base gate/ratio for a given rollout step.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout_attn = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.dropout_ff = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout_out = nn.Dropout(dropout)

    def _gated_residual(
        self,
        residual: torch.Tensor,
        branch: torch.Tensor,
        residual_grad_gate: float | torch.Tensor,
        resgrad_policy: str,
        resgrad_ratio_threshold: float,
        ratio_collector: Optional[list],
        gate_collector: Optional[list],
        dual_wiener: Optional[DualWienerController] = None,
        route_horizon: int = -1,
        route_layer: int = -1,
    ) -> torch.Tensor:
        policy = str(resgrad_policy).lower()
        branch_norm = branch.detach().float().reshape(branch.shape[0], -1).norm(dim=1).mean()
        residual_norm = residual.detach().float().reshape(residual.shape[0], -1).norm(dim=1).mean()
        branch_residual_ratio = branch_norm / (residual_norm + 1e-8)
        if ratio_collector is not None:
            ratio_collector.append(float(branch_residual_ratio.detach().cpu()))

        if policy in ("ratio", "act_ratio", "dynamic", "dynamic_ratio", "branch_ratio", "norm_ratio"):
            base_gate = (
                float(residual_grad_gate)
                if not torch.is_tensor(residual_grad_gate)
                else float(residual_grad_gate.detach().float().mean().cpu())
            )
            gate_val = 1.0 if float(branch_residual_ratio.detach().cpu()) >= float(resgrad_ratio_threshold) else base_gate
        else:
            gate_val = residual_grad_gate

        if gate_collector is not None:
            gate_collector.append(
                float(gate_val) if not torch.is_tensor(gate_val) else float(gate_val.detach().float().mean().cpu())
            )

        if policy == "dualwiener":
            if dual_wiener is None:
                raise ValueError("resgrad_policy='dualwiener' requires a DualWienerController")
            # The branch has already been evaluated from the value-identical
            # branch reference created in forward().  route_pair applies the
            # two custom backward coefficients without changing this sum.
            return residual + branch
        if torch.is_tensor(gate_val):
            gate_tensor = gate_val.to(device=branch.device, dtype=branch.dtype)
            branch = branch.detach() + gate_tensor * (branch - branch.detach())
        else:
            gate_float = float(gate_val)
            if gate_float <= 0.0:
                branch = branch.detach()
            elif gate_float < 1.0:
                branch = branch.detach() + gate_float * (branch - branch.detach())
            # gate >= 1: ordinary autograd, no change.
        return residual + branch

    def forward(
        self,
        x: torch.Tensor,
        residual_grad_gate: float | torch.Tensor = 1.0,
        resgrad_policy: str = "all",
        resgrad_ratio_threshold: float = 0.05,
        ratio_collector: Optional[list] = None,
        gate_collector: Optional[list] = None,
        dual_wiener: Optional[DualWienerController] = None,
        route_horizon: int = -1,
        route_layer_base: int = 0,
    ) -> torch.Tensor:
        if str(resgrad_policy).lower() == "dualwiener":
            if dual_wiener is None:
                raise ValueError("resgrad_policy='dualwiener' requires a DualWienerController")
            residual, branch_input = dual_wiener.route_pair(
                x, int(route_horizon), int(route_layer_base)
            )
        else:
            residual = x
            branch_input = x
        normed = self.norm1(branch_input)
        attn_out, _ = self.self_attn(normed, normed, normed, need_weights=False)
        branch = self.dropout_attn(attn_out)
        x = self._gated_residual(
            residual, branch, residual_grad_gate, resgrad_policy,
            resgrad_ratio_threshold, ratio_collector, gate_collector,
            dual_wiener, route_horizon, route_layer_base,
        )

        if str(resgrad_policy).lower() == "dualwiener":
            residual, branch_input = dual_wiener.route_pair(
                x, int(route_horizon), int(route_layer_base) + 1
            )
        else:
            residual = x
            branch_input = x
        normed = self.norm2(branch_input)
        branch = self.linear2(self.dropout_ff(self.act(self.linear1(normed))))
        branch = self.dropout_out(branch)
        x = self._gated_residual(
            residual, branch, residual_grad_gate, resgrad_policy,
            resgrad_ratio_threshold, ratio_collector, gate_collector,
            dual_wiener, route_horizon, route_layer_base + 1,
        )
        return x


class TransformerVectorARModel(nn.Module, _StimPoolMixin):
    """Clean Transformer encoder AR baseline for HCP/vector states."""

    is_standard_autoregressive = True
    task_type = "vector"

    def __init__(
        self,
        state_dim: int,
        input_dim: int = 0,
        window_size: int = 4,
        hidden_dim: int = 512,
        depth: int = 4,
        nhead: int = 8,
        dropout: float = 0.1,
        has_external_input: bool = True,
        residual: bool = True,
        ff_mult: int = 2,
        enable_error_transport: bool = False,
        etm_diag_base: float = 0.0,
        etm_diag_scale: float = 0.5,
        etm_transport_param: str = "full",
        etm_lowrank_rank: int = 16,
        etm_metric_atoms: int = 64,
        etm_metric_temperature: float = 1.0,
        resgrad_routing: bool = False,
        resgrad_policy: str = "all",
        resgrad_block_gate: float = 1.0,
        resgrad_ratio_threshold: float = 0.05,
        resgrad_keep_every: int = 8,
        resgrad_keep_tail: int = 0,
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
        self.window_size = int(window_size)
        self.hidden_dim = int(hidden_dim)
        self.has_external_input = bool(has_external_input)
        self.residual = bool(residual)
        self.enable_error_transport = bool(enable_error_transport)
        self.resgrad_routing = bool(resgrad_routing)
        self.resgrad_policy = str(resgrad_policy).lower()
        self.resgrad_block_gate = float(resgrad_block_gate)
        self.resgrad_ratio_threshold = float(resgrad_ratio_threshold)
        self.resgrad_keep_every = max(1, int(resgrad_keep_every))
        self.resgrad_keep_tail = max(0, int(resgrad_keep_tail))
        self.depth = int(depth)
        self.resgrad_current_horizon = -1
        self.resgrad_current_total_horizon = -1
        self.etm_diag_base = float(etm_diag_base)
        self.etm_diag_scale = float(etm_diag_scale)
        self.etm_transport_param = str(etm_transport_param).lower()
        self.etm_lowrank_rank = max(1, int(etm_lowrank_rank))
        self.etm_metric_atoms = max(1, int(etm_metric_atoms))
        self.etm_metric_temperature = max(1e-4, float(etm_metric_temperature))
        in_dim = self.state_dim + (self.input_dim if self.has_external_input else 0)
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        if self.resgrad_routing:
            # nn.TransformerEncoderLayer's residual adds are not hookable, so
            # ResGrad gating requires a from-scratch equivalent stack.
            self.gated_layers = nn.ModuleList([
                GatedTransformerEncoderLayer(hidden_dim, nhead, int(ff_mult) * hidden_dim, dropout=dropout)
                for _ in range(depth)
            ])
            self.encoder = None
        else:
            enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=int(ff_mult) * hidden_dim, dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
            self.gated_layers = None
        self.pos = nn.Parameter(torch.zeros(1, window_size, hidden_dim))
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, self.state_dim))
        self.dual_wiener = None
        if self.resgrad_routing and self.resgrad_policy == "dualwiener":
            self.dual_wiener = DualWienerController(
                state_dim=self.state_dim,
                depth=2 * self.depth,
                max_horizon=int(dual_wiener_max_horizon),
                ema=float(dual_wiener_ema),
                residual_ema=float(dual_wiener_residual_ema),
                warmup_batches=int(dual_wiener_warmup_batches),
                probe_every=int(dual_wiener_probe_every),
                min_probes=int(dual_wiener_min_probes),
                noise_model=str(dual_wiener_noise_model),
            )
        if self.enable_error_transport:
            fut_dim = self.input_dim if self.has_external_input else 0
            etm_hidden = min(512, hidden_dim)
            self.etm_future_proj = nn.Sequential(
                nn.LayerNorm(fut_dim),
                nn.Linear(fut_dim, hidden_dim),
                nn.GELU(),
            ) if fut_dim > 0 else None
            if self.etm_transport_param in {"lowrank", "residual_lowrank", "reslowrank", "lr"}:
                r = self.etm_lowrank_rank
                # Predict a residual low-rank correction G = U diag(s) V^T.
                # The returned matrix is J = base*I + scale*G. With base=1,
                # fitting J e ~= Delta learns only the non-identity part:
                # G e ~= Delta - e.
                self.etm_head = nn.Sequential(
                    nn.LayerNorm(2 * hidden_dim),
                    nn.Linear(2 * hidden_dim, etm_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(etm_hidden, 2 * self.state_dim * r + r),
                )
            else:
                self.etm_head = nn.Sequential(
                    nn.LayerNorm(2 * hidden_dim),
                    nn.Linear(2 * hidden_dim, etm_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(etm_hidden, self.state_dim * self.state_dim),
                )
            # Start from J ~= etm_diag_base * I. For the low-rank product,
            # keep U/V randomly initialized but set singular strengths s=0;
            # zeroing all outputs would kill gradients through U diag(s) V^T.
            if self.etm_transport_param in {"lowrank", "residual_lowrank", "reslowrank", "lr"}:
                r = self.etm_lowrank_rank
                with torch.no_grad():
                    self.etm_head[-1].weight[-r:].zero_()
                    self.etm_head[-1].bias[-r:].zero_()
            else:
                nn.init.zeros_(self.etm_head[-1].weight)
                nn.init.zeros_(self.etm_head[-1].bias)

            # Attention-style trajectory metric.  It is not a transport
            # Jacobian.  It defines a PSD, low-rank error metric
            #   M_t = B diag(a_t) B^T,
            #   e^T M_t e = sum_i a_i (b_i^T e)^2.
            # B is a shared dictionary of reusable error directions, while
            # a_t is a context/horizon-dependent attention over those atoms.
            m = self.etm_metric_atoms
            self.etm_metric_basis = nn.Parameter(torch.randn(self.state_dim, m) / (float(self.state_dim) ** 0.5))
            self.etm_metric_head = nn.Sequential(
                nn.LayerNorm(2 * hidden_dim),
                nn.Linear(2 * hidden_dim, etm_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(etm_hidden, m),
            )
            self.etm_metric_horizon = nn.Sequential(
                nn.Linear(1, etm_hidden),
                nn.GELU(),
                nn.Linear(etm_hidden, m),
            )
            self.etm_metric_log_scale = nn.Parameter(torch.zeros(()))
        else:
            self.etm_future_proj = None
            self.etm_head = None
            self.etm_metric_basis = None
            self.etm_metric_head = None
            self.etm_metric_horizon = None
            self.etm_metric_log_scale = None

    def _resgrad_gate_for_step(self, horizon_index: Optional[int] = None) -> float:
        """Return the nonlinear-branch gradient gate for this rollout step.

        Byte-for-byte the same policy switch as
        OfficialStateMambaARModel._resgrad_gate_for_step,
        UNetFieldModel._resgrad_gate_for_step, and
        ResNetVectorARModel._resgrad_gate_for_step.
        """
        if not self.resgrad_routing:
            return 1.0
        pol = str(self.resgrad_policy).lower()
        k = int(horizon_index) if horizon_index is not None else int(getattr(self, "resgrad_current_horizon", -1))
        K = int(getattr(self, "resgrad_current_total_horizon", -1))
        base_gate = float(self.resgrad_block_gate)
        if pol in ("all", "full", "normal"):
            return 1.0
        if pol in ("none", "identity", "id"):
            return 0.0
        if pol in ("fixed", "scalar"):
            return base_gate
        if pol in ("periodic", "every"):
            if k < 0:
                return base_gate
            return 1.0 if (k % self.resgrad_keep_every == 0) else base_gate
        if pol in ("tail", "last"):
            if k < 0 or K <= 0:
                return base_gate
            return 1.0 if k >= max(0, K - self.resgrad_keep_tail) else base_gate
        if pol in ("head", "first"):
            if k < 0:
                return base_gate
            return 1.0 if k < self.resgrad_keep_tail else base_gate
        return base_gate

    def set_resgrad_context(self, horizon_index: Optional[int] = None, total_horizon: Optional[int] = None):
        self.resgrad_current_horizon = -1 if horizon_index is None else int(horizon_index)
        self.resgrad_current_total_horizon = -1 if total_horizon is None else int(total_horizon)

    # ---- Internal-DW training hooks.  These mirror UNetFieldModel and the
    # shipped Mamba implementation so the generic training loop remains
    # backbone-agnostic.
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
        return {} if self.dual_wiener is None else self.dual_wiener.diagnostics(horizon)

    def dual_wiener_export_state(self, horizon: int):
        return None if self.dual_wiener is None else self.dual_wiener.export_state(horizon)

    def encode_context(
        self,
        history: torch.Tensor,
        stim_window: Optional[torch.Tensor] = None,
        ratio_collector: Optional[list] = None,
        gate_collector: Optional[list] = None,
    ) -> torch.Tensor:
        B, W, D = history.shape
        stim = self._pool_stim(stim_window, B, W, history.device, history.dtype)
        x = torch.cat([history, stim], dim=-1) if self.has_external_input else history
        h = self.in_proj(x) + self.pos[:, :W]
        if self.resgrad_routing:
            gate = self._resgrad_gate_for_step()
            for layer_index, layer in enumerate(self.gated_layers):
                h = layer(
                    h,
                    residual_grad_gate=gate,
                    resgrad_policy=self.resgrad_policy,
                    resgrad_ratio_threshold=self.resgrad_ratio_threshold,
                    ratio_collector=ratio_collector,
                    gate_collector=gate_collector,
                    dual_wiener=self.dual_wiener,
                    route_horizon=int(self.resgrad_current_horizon),
                    route_layer_base=2 * layer_index,
                )
        else:
            h = self.encoder(h)
        return h[:, -1]

    def predict_frame_from_history(
        self,
        history: torch.Tensor,
        stim_window: Optional[torch.Tensor] = None,
        ratio_collector: Optional[list] = None,
        gate_collector: Optional[list] = None,
    ) -> torch.Tensor:
        ctx = self.encode_context(history, stim_window, ratio_collector=ratio_collector, gate_collector=gate_collector)
        delta_or_frame = self.out(ctx)
        return history[:, -1] + delta_or_frame if self.residual else delta_or_frame

    def predict_transport_matrix(
        self,
        history: torch.Tensor,
        stim_window: Optional[torch.Tensor] = None,
        future_stim: Optional[torch.Tensor] = None,
        version: str = "A",
        detach_context: bool = True,
    ) -> torch.Tensor:
        """Predict a full signed error-transport matrix J_{t,K}.

        Returns J with shape [B, state_dim, state_dim].  The transport prediction
        is e_K ~= J e_1, and the suppression metric is W = J^T J.
        Version A uses current history/current stimulus context. Version B also
        summarizes future external inputs u_{t:t+K}.
        """
        if self.etm_head is None:
            raise RuntimeError("This model was built without enable_error_transport=True")
        B = history.shape[0]
        ctx = self.encode_context(history, stim_window)
        if detach_context:
            ctx = ctx.detach()
        version = str(version).upper()
        if version == "B" and self.has_external_input and future_stim is not None and self.etm_future_proj is not None:
            if future_stim.dim() == 2:
                future_stim = future_stim.unsqueeze(1)
            T_fut = int(future_stim.shape[1]) if future_stim.dim() >= 3 else 1
            fut = self._pool_stim(future_stim, B, T_fut, history.device, history.dtype).mean(dim=1)
            fut_ctx = self.etm_future_proj(fut)
            if detach_context:
                fut_ctx = fut_ctx.detach()
        else:
            fut_ctx = torch.zeros_like(ctx)
        raw = self.etm_head(torch.cat([ctx, fut_ctx], dim=-1))
        if self.etm_transport_param in {"lowrank", "residual_lowrank", "reslowrank", "lr"}:
            r = self.etm_lowrank_rank
            d = self.state_dim
            u_raw, v_raw, s_raw = torch.split(raw, [d * r, d * r, r], dim=-1)
            U = F.normalize(u_raw.view(B, d, r), dim=1, eps=1e-6)
            V = F.normalize(v_raw.view(B, d, r), dim=1, eps=1e-6)
            svals = torch.tanh(s_raw).view(B, 1, r)
            G = torch.bmm(U * svals, V.transpose(1, 2))
            J = self.etm_diag_scale * G
        else:
            raw = raw.view(B, self.state_dim, self.state_dim)
            J = self.etm_diag_scale * torch.tanh(raw)
        if self.etm_diag_base != 0.0:
            eye = torch.eye(self.state_dim, device=J.device, dtype=J.dtype).unsqueeze(0)
            J = J + self.etm_diag_base * eye
        return J

    def predict_error_metric_score(
        self,
        history: torch.Tensor,
        error: torch.Tensor,
        stim_window: Optional[torch.Tensor] = None,
        future_stim: Optional[torch.Tensor] = None,
        horizon: int | float | None = None,
        version: str = "A",
        detach_context: bool = True,
        detach_metric: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """Score one-step error under a trajectory-aware attention metric.

        This head is intentionally different from predict_transport_matrix().
        It learns reusable error-direction atoms B and a context/horizon gating
        distribution a_t.  The scalar score is

            s(e) = e^T B diag(a_t) B^T e
                 = sum_i a_i (b_i^T e)^2.

        If detach_metric=True, B and a_t are stopped so gradients flow only to
        the error vector.  This is the mode used by the main AR loss.
        """
        if self.etm_metric_head is None or self.etm_metric_basis is None:
            raise RuntimeError("This model was built without the ETM attention metric head")
        Bsz = history.shape[0]
        ctx = self.encode_context(history, stim_window)
        if detach_context:
            ctx = ctx.detach()
        version = str(version).upper()
        if version == "B" and self.has_external_input and future_stim is not None and self.etm_future_proj is not None:
            if future_stim.dim() == 2:
                future_stim = future_stim.unsqueeze(1)
            T_fut = int(future_stim.shape[1]) if future_stim.dim() >= 3 else 1
            fut = self._pool_stim(future_stim, Bsz, T_fut, history.device, history.dtype).mean(dim=1)
            fut_ctx = self.etm_future_proj(fut)
            if detach_context:
                fut_ctx = fut_ctx.detach()
        else:
            fut_ctx = torch.zeros_like(ctx)

        logits = self.etm_metric_head(torch.cat([ctx, fut_ctx], dim=-1))
        if horizon is not None:
            h = torch.as_tensor(float(horizon), device=history.device, dtype=history.dtype)
            h = torch.log1p(h).view(1, 1).expand(Bsz, 1)
            logits = logits + self.etm_metric_horizon(h)

        basis = F.normalize(self.etm_metric_basis, dim=0, eps=1e-6)
        log_scale = self.etm_metric_log_scale
        if detach_metric:
            logits = logits.detach()
            basis = basis.detach()
            log_scale = log_scale.detach()

        weights = torch.softmax(logits / self.etm_metric_temperature, dim=-1)
        err = error.reshape(Bsz, -1)
        proj = err @ basis
        scale = F.softplus(log_scale) + 1e-6
        score = scale * (weights * proj.pow(2)).sum(dim=-1)
        entropy = -(weights * (weights.clamp_min(1e-12)).log()).sum(dim=-1).mean()
        max_w = weights.max(dim=-1).values.mean()
        return score, {
            "attn_entropy": entropy,
            "attn_max": max_w,
            "score_scale": scale.detach(),
            "basis_fro": basis.detach().norm(),
        }

    def predict_transport_diag(
        self,
        history: torch.Tensor,
        stim_window: Optional[torch.Tensor] = None,
        future_stim: Optional[torch.Tensor] = None,
        version: str = "A",
        detach_context: bool = True,
    ) -> torch.Tensor:
        # Backward-compatible diagnostic: return diag(J).
        J = self.predict_transport_matrix(history, stim_window, future_stim, version, detach_context)
        return torch.diagonal(J, dim1=-2, dim2=-1)

    def step_history(self, history: torch.Tensor, stim_window: Optional[torch.Tensor] = None) -> torch.Tensor:
        frame = self.predict_frame_from_history(history, stim_window)
        return torch.cat([history[:, 1:], frame.unsqueeze(1)], dim=1)

    def forward(
        self,
        stim_window: Optional[torch.Tensor],
        history: torch.Tensor,
        return_aux: bool = False,
        horizon_index: Optional[int] = None,
        total_horizon: Optional[int] = None,
        ratio_collector: Optional[list] = None,
    ):
        if horizon_index is not None or total_horizon is not None:
            self.set_resgrad_context(horizon_index=horizon_index, total_horizon=total_horizon)
        collector = ratio_collector
        gate_collector = [] if return_aux else None
        if return_aux and collector is None:
            collector = []
        frame = self.predict_frame_from_history(history, stim_window, ratio_collector=collector, gate_collector=gate_collector)
        pred = frame.unsqueeze(1)
        if return_aux:
            aux = {"pred_frame": frame}
            aux["resgrad_gate"] = (
                frame.new_tensor(float(sum(gate_collector) / len(gate_collector)))
                if gate_collector else frame.new_tensor(float(self._resgrad_gate_for_step(horizon_index)))
            )
            aux["resgrad_routing"] = frame.new_tensor(1.0 if self.resgrad_routing else 0.0)
            if collector:
                aux["resgrad_branch_residual_ratio"] = frame.new_tensor(float(sum(collector) / len(collector)))
            return pred, aux
        return pred


class BridgeTransformerVectorARModel(nn.Module, _StimPoolMixin):
    """Output-space Transformer AR model matched to the older HCP raw Transformer baseline.

    This is intentionally clean: no AE/latent/token branch.  It follows the
    earlier OutputBridgeTransformerModel shape: concatenate stimulus and fMRI
    tokens, project to a modest hidden size, run a Transformer encoder, and
    predict the next fMRI frame directly.  The prediction can optionally be
    residualized, but the default is absolute prediction because the older raw
    HCP Transformer used an absolute output head.
    """

    is_standard_autoregressive = True
    task_type = "vector"

    def __init__(
        self,
        state_dim: int,
        input_dim: int = 0,
        window_size: int = 8,
        hidden_dim: int = 192,
        depth: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
        has_external_input: bool = True,
        residual: bool = False,
        ff_mult: int = 2,
        enable_error_transport: bool = False,
        etm_diag_base: float = 0.0,
        etm_diag_scale: float = 0.5,
        etm_transport_param: str = "full",
        etm_lowrank_rank: int = 16,
        etm_metric_atoms: int = 64,
        etm_metric_temperature: float = 1.0,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.input_dim = int(input_dim)
        self.window_size = int(window_size)
        self.hidden_dim = int(hidden_dim)
        self.has_external_input = bool(has_external_input)
        self.residual = bool(residual)
        self.enable_error_transport = bool(enable_error_transport)
        self.etm_diag_base = float(etm_diag_base)
        self.etm_diag_scale = float(etm_diag_scale)
        self.etm_transport_param = str(etm_transport_param).lower()
        self.etm_lowrank_rank = max(1, int(etm_lowrank_rank))
        self.etm_metric_atoms = max(1, int(etm_metric_atoms))
        self.etm_metric_temperature = max(1e-4, float(etm_metric_temperature))
        in_dim = self.state_dim + (self.input_dim if self.has_external_input else 0)
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        if depth > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=int(ff_mult) * hidden_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        else:
            self.encoder = nn.Identity()
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.state_dim),
        )
        if self.enable_error_transport:
            fut_dim = self.input_dim if self.has_external_input else 0
            etm_hidden = min(512, hidden_dim)
            self.etm_future_proj = nn.Sequential(
                nn.LayerNorm(fut_dim),
                nn.Linear(fut_dim, hidden_dim),
                nn.GELU(),
            ) if fut_dim > 0 else None
            if self.etm_transport_param in {"lowrank", "residual_lowrank", "reslowrank", "lr"}:
                r = self.etm_lowrank_rank
                self.etm_head = nn.Sequential(
                    nn.LayerNorm(2 * hidden_dim),
                    nn.Linear(2 * hidden_dim, etm_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(etm_hidden, 2 * self.state_dim * r + r),
                )
            else:
                self.etm_head = nn.Sequential(
                    nn.LayerNorm(2 * hidden_dim),
                    nn.Linear(2 * hidden_dim, etm_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(etm_hidden, self.state_dim * self.state_dim),
                )
            if self.etm_transport_param in {"lowrank", "residual_lowrank", "reslowrank", "lr"}:
                r = self.etm_lowrank_rank
                with torch.no_grad():
                    self.etm_head[-1].weight[-r:].zero_()
                    self.etm_head[-1].bias[-r:].zero_()
            else:
                nn.init.zeros_(self.etm_head[-1].weight)
                nn.init.zeros_(self.etm_head[-1].bias)

            # Attention-style trajectory metric.  It is not a transport
            # Jacobian.  It defines a PSD, low-rank error metric
            #   M_t = B diag(a_t) B^T,
            #   e^T M_t e = sum_i a_i (b_i^T e)^2.
            # B is a shared dictionary of reusable error directions, while
            # a_t is a context/horizon-dependent attention over those atoms.
            m = self.etm_metric_atoms
            self.etm_metric_basis = nn.Parameter(torch.randn(self.state_dim, m) / (float(self.state_dim) ** 0.5))
            self.etm_metric_head = nn.Sequential(
                nn.LayerNorm(2 * hidden_dim),
                nn.Linear(2 * hidden_dim, etm_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(etm_hidden, m),
            )
            self.etm_metric_horizon = nn.Sequential(
                nn.Linear(1, etm_hidden),
                nn.GELU(),
                nn.Linear(etm_hidden, m),
            )
            self.etm_metric_log_scale = nn.Parameter(torch.zeros(()))
        else:
            self.etm_future_proj = None
            self.etm_head = None
            self.etm_metric_basis = None
            self.etm_metric_head = None
            self.etm_metric_horizon = None
            self.etm_metric_log_scale = None

    def encode_context(self, history: torch.Tensor, stim_window: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, W, D = history.shape
        stim = self._pool_stim(stim_window, B, W, history.device, history.dtype)
        x = torch.cat([history, stim], dim=-1) if self.has_external_input else history
        h = self.in_proj(x)
        h = self.encoder(h)
        return h[:, -1]

    def predict_frame_from_history(self, history: torch.Tensor, stim_window: Optional[torch.Tensor] = None) -> torch.Tensor:
        ctx = self.encode_context(history, stim_window)
        pred = self.head(ctx)
        return history[:, -1] + pred if self.residual else pred

    def predict_transport_matrix(
        self,
        history: torch.Tensor,
        stim_window: Optional[torch.Tensor] = None,
        future_stim: Optional[torch.Tensor] = None,
        version: str = "A",
        detach_context: bool = True,
    ) -> torch.Tensor:
        if self.etm_head is None:
            raise RuntimeError("This model was built without enable_error_transport=True")
        B = history.shape[0]
        ctx = self.encode_context(history, stim_window)
        if detach_context:
            ctx = ctx.detach()
        version = str(version).upper()
        if version == "B" and self.has_external_input and future_stim is not None and self.etm_future_proj is not None:
            if future_stim.dim() == 2:
                future_stim = future_stim.unsqueeze(1)
            T_fut = int(future_stim.shape[1]) if future_stim.dim() >= 3 else 1
            fut = self._pool_stim(future_stim, B, T_fut, history.device, history.dtype).mean(dim=1)
            fut_ctx = self.etm_future_proj(fut)
            if detach_context:
                fut_ctx = fut_ctx.detach()
        else:
            fut_ctx = torch.zeros_like(ctx)
        raw = self.etm_head(torch.cat([ctx, fut_ctx], dim=-1))
        if self.etm_transport_param in {"lowrank", "residual_lowrank", "reslowrank", "lr"}:
            r = self.etm_lowrank_rank
            d = self.state_dim
            u_raw, v_raw, s_raw = torch.split(raw, [d * r, d * r, r], dim=-1)
            U = F.normalize(u_raw.view(B, d, r), dim=1, eps=1e-6)
            V = F.normalize(v_raw.view(B, d, r), dim=1, eps=1e-6)
            svals = torch.tanh(s_raw).view(B, 1, r)
            G = torch.bmm(U * svals, V.transpose(1, 2))
            J = self.etm_diag_scale * G
        else:
            raw = raw.view(B, self.state_dim, self.state_dim)
            J = self.etm_diag_scale * torch.tanh(raw)
        if self.etm_diag_base != 0.0:
            eye = torch.eye(self.state_dim, device=J.device, dtype=J.dtype).unsqueeze(0)
            J = J + self.etm_diag_base * eye
        return J

    def predict_error_metric_score(
        self,
        history: torch.Tensor,
        error: torch.Tensor,
        stim_window: Optional[torch.Tensor] = None,
        future_stim: Optional[torch.Tensor] = None,
        horizon: int | float | None = None,
        version: str = "A",
        detach_context: bool = True,
        detach_metric: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """Score one-step error under a trajectory-aware attention metric.

        This head is intentionally different from predict_transport_matrix().
        It learns reusable error-direction atoms B and a context/horizon gating
        distribution a_t.  The scalar score is

            s(e) = e^T B diag(a_t) B^T e
                 = sum_i a_i (b_i^T e)^2.

        If detach_metric=True, B and a_t are stopped so gradients flow only to
        the error vector.  This is the mode used by the main AR loss.
        """
        if self.etm_metric_head is None or self.etm_metric_basis is None:
            raise RuntimeError("This model was built without the ETM attention metric head")
        Bsz = history.shape[0]
        ctx = self.encode_context(history, stim_window)
        if detach_context:
            ctx = ctx.detach()
        version = str(version).upper()
        if version == "B" and self.has_external_input and future_stim is not None and self.etm_future_proj is not None:
            if future_stim.dim() == 2:
                future_stim = future_stim.unsqueeze(1)
            T_fut = int(future_stim.shape[1]) if future_stim.dim() >= 3 else 1
            fut = self._pool_stim(future_stim, Bsz, T_fut, history.device, history.dtype).mean(dim=1)
            fut_ctx = self.etm_future_proj(fut)
            if detach_context:
                fut_ctx = fut_ctx.detach()
        else:
            fut_ctx = torch.zeros_like(ctx)

        logits = self.etm_metric_head(torch.cat([ctx, fut_ctx], dim=-1))
        if horizon is not None:
            h = torch.as_tensor(float(horizon), device=history.device, dtype=history.dtype)
            h = torch.log1p(h).view(1, 1).expand(Bsz, 1)
            logits = logits + self.etm_metric_horizon(h)

        basis = F.normalize(self.etm_metric_basis, dim=0, eps=1e-6)
        log_scale = self.etm_metric_log_scale
        if detach_metric:
            logits = logits.detach()
            basis = basis.detach()
            log_scale = log_scale.detach()

        weights = torch.softmax(logits / self.etm_metric_temperature, dim=-1)
        err = error.reshape(Bsz, -1)
        proj = err @ basis
        scale = F.softplus(log_scale) + 1e-6
        score = scale * (weights * proj.pow(2)).sum(dim=-1)
        entropy = -(weights * (weights.clamp_min(1e-12)).log()).sum(dim=-1).mean()
        max_w = weights.max(dim=-1).values.mean()
        return score, {
            "attn_entropy": entropy,
            "attn_max": max_w,
            "score_scale": scale.detach(),
            "basis_fro": basis.detach().norm(),
        }

    def predict_transport_diag(
        self,
        history: torch.Tensor,
        stim_window: Optional[torch.Tensor] = None,
        future_stim: Optional[torch.Tensor] = None,
        version: str = "A",
        detach_context: bool = True,
    ) -> torch.Tensor:
        J = self.predict_transport_matrix(history, stim_window, future_stim, version, detach_context)
        return torch.diagonal(J, dim1=-2, dim2=-1)

    def step_history(self, history: torch.Tensor, stim_window: Optional[torch.Tensor] = None) -> torch.Tensor:
        frame = self.predict_frame_from_history(history, stim_window)
        return torch.cat([history[:, 1:], frame.unsqueeze(1)], dim=1)

    def forward(self, stim_window: Optional[torch.Tensor], history: torch.Tensor, return_aux: bool = False):
        frame = self.predict_frame_from_history(history, stim_window)
        pred = frame.unsqueeze(1)
        if return_aux:
            return pred, {"pred_frame": frame, "state": frame}
        return pred


class TCNVectorARModel(nn.Module, _StimPoolMixin):
    """Clean temporal-convolution AR baseline for HCP/vector states."""

    is_standard_autoregressive = True
    task_type = "vector"

    def __init__(
        self,
        state_dim: int,
        input_dim: int = 0,
        window_size: int = 4,
        hidden_dim: int = 512,
        depth: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        has_external_input: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.input_dim = int(input_dim)
        self.window_size = int(window_size)
        self.hidden_dim = int(hidden_dim)
        self.has_external_input = bool(has_external_input)
        self.residual = bool(residual)
        in_dim = self.state_dim + (self.input_dim if self.has_external_input else 0)
        self.in_proj = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)
        layers = []
        for i in range(depth):
            dilation = 2 ** i
            pad = dilation * (kernel_size - 1)
            layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=pad, dilation=dilation))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, self.state_dim))

    def predict_frame_from_history(self, history: torch.Tensor, stim_window: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, W, D = history.shape
        stim = self._pool_stim(stim_window, B, W, history.device, history.dtype)
        x = torch.cat([history, stim], dim=-1) if self.has_external_input else history
        x = x.transpose(1, 2)
        h = self.in_proj(x)
        h0 = h
        h = self.net(h)
        # Conv1d padding is causal-left plus extra right; crop back to W.
        h = h[..., :W] + h0
        last = h[..., -1]
        delta_or_frame = self.out(last)
        return history[:, -1] + delta_or_frame if self.residual else delta_or_frame

    def step_history(self, history: torch.Tensor, stim_window: Optional[torch.Tensor] = None) -> torch.Tensor:
        frame = self.predict_frame_from_history(history, stim_window)
        return torch.cat([history[:, 1:], frame.unsqueeze(1)], dim=1)

    def forward(self, stim_window: Optional[torch.Tensor], history: torch.Tensor, return_aux: bool = False):
        frame = self.predict_frame_from_history(history, stim_window)
        pred = frame.unsqueeze(1)
        if return_aux:
            return pred, {"pred_frame": frame}
        return pred


class ResidualMLPBlock(nn.Module):
    """Pre-norm residual MLP block: y = x + F(LayerNorm(x)).

    Same shape as one layer of StatefulMambaBlock (official_state_mamba.py):
    identity residual + a gateable nonlinear branch, no activation after the
    add. Used by ResNetVectorARModel as the plain "deep residual network"
    backbone for testing whether ResGrad's stopgrad-gated routing
    generalizes beyond the Mamba SSM stack.
    """

    def __init__(self, dim: int, hidden_mult: int = 2, dropout: float = 0.0):
        super().__init__()
        inner = int(dim * hidden_mult)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, inner)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(inner, dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        x: torch.Tensor,
        residual_grad_gate: float | torch.Tensor = 1.0,
        resgrad_policy: str = "all",
        resgrad_ratio_threshold: float = 0.05,
        ratio_collector: Optional[list] = None,
        gate_collector: Optional[list] = None,
    ) -> torch.Tensor:
        residual = x
        branch = self.fc2(self.dropout(self.act(self.fc1(self.norm(x)))))

        # Residual-gradient routing: same stopgrad construction as
        # StatefulMambaBlock.step / unet_field.ResidualBlock.forward.
        policy = str(resgrad_policy).lower()
        branch_norm = branch.detach().float().reshape(branch.shape[0], -1).norm(dim=1).mean()
        residual_norm = residual.detach().float().reshape(residual.shape[0], -1).norm(dim=1).mean()
        branch_residual_ratio = branch_norm / (residual_norm + 1e-8)
        if ratio_collector is not None:
            ratio_collector.append(float(branch_residual_ratio.detach().cpu()))

        if policy in ("ratio", "act_ratio", "dynamic", "dynamic_ratio", "branch_ratio", "norm_ratio"):
            base_gate = (
                float(residual_grad_gate)
                if not torch.is_tensor(residual_grad_gate)
                else float(residual_grad_gate.detach().float().mean().cpu())
            )
            gate_val = 1.0 if float(branch_residual_ratio.detach().cpu()) >= float(resgrad_ratio_threshold) else base_gate
        else:
            gate_val = residual_grad_gate

        if gate_collector is not None:
            gate_collector.append(
                float(gate_val) if not torch.is_tensor(gate_val) else float(gate_val.detach().float().mean().cpu())
            )

        if torch.is_tensor(gate_val):
            gate_tensor = gate_val.to(device=branch.device, dtype=branch.dtype)
            branch = branch.detach() + gate_tensor * (branch - branch.detach())
        else:
            gate_float = float(gate_val)
            if gate_float <= 0.0:
                branch = branch.detach()
            elif gate_float < 1.0:
                branch = branch.detach() + gate_float * (branch - branch.detach())
            # gate >= 1: ordinary autograd, no change.
        return residual + branch


class ResNetVectorARModel(nn.Module, _StimPoolMixin):
    """Plain deep pre-norm residual MLP AR baseline for HCP/vector states.

    Second (non-recurrent, feedforward) backbone for ResGrad's residual-
    structure generalization test. The whole window is flattened into one
    token (same convention as OfficialStateMambaARModel.make_token, but
    applied to the full window at once rather than per-frame), then passed
    through N stacked ResidualMLPBlock instances, each independently
    stopgrad-gated by the same per-rollout-step policy switch used by
    StatefulMambaBlock and unet_field.UNetFieldModel.
    """

    is_standard_autoregressive = True
    task_type = "vector"

    def __init__(
        self,
        state_dim: int,
        input_dim: int = 0,
        window_size: int = 4,
        hidden_dim: int = 512,
        depth: int = 4,
        hidden_mult: int = 2,
        dropout: float = 0.0,
        has_external_input: bool = True,
        residual: bool = True,
        resgrad_routing: bool = False,
        resgrad_policy: str = "all",
        resgrad_block_gate: float = 1.0,
        resgrad_ratio_threshold: float = 0.05,
        resgrad_keep_every: int = 8,
        resgrad_keep_tail: int = 0,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.input_dim = int(input_dim)
        self.window_size = int(window_size)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.has_external_input = bool(has_external_input)
        self.residual = bool(residual)
        self.resgrad_routing = bool(resgrad_routing)
        self.resgrad_policy = str(resgrad_policy).lower()
        self.resgrad_block_gate = float(resgrad_block_gate)
        self.resgrad_ratio_threshold = float(resgrad_ratio_threshold)
        self.resgrad_keep_every = max(1, int(resgrad_keep_every))
        self.resgrad_keep_tail = max(0, int(resgrad_keep_tail))
        self.resgrad_current_horizon = -1
        self.resgrad_current_total_horizon = -1

        per_step_dim = self.state_dim + (self.input_dim if self.has_external_input else 0)
        in_dim = self.window_size * per_step_dim
        self.in_proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, self.hidden_dim), nn.GELU())
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(self.hidden_dim, hidden_mult=hidden_mult, dropout=dropout)
            for _ in range(max(1, self.depth))
        ])
        self.out = nn.Sequential(nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.state_dim))

    def _resgrad_gate_for_step(self, horizon_index: Optional[int] = None) -> float:
        """Return the nonlinear-branch gradient gate for this rollout step.

        Byte-for-byte the same policy switch as
        OfficialStateMambaARModel._resgrad_gate_for_step and
        UNetFieldModel._resgrad_gate_for_step.
        """
        if not self.resgrad_routing:
            return 1.0
        pol = str(self.resgrad_policy).lower()
        k = int(horizon_index) if horizon_index is not None else int(getattr(self, "resgrad_current_horizon", -1))
        K = int(getattr(self, "resgrad_current_total_horizon", -1))
        base_gate = float(self.resgrad_block_gate)
        if pol in ("all", "full", "normal"):
            return 1.0
        if pol in ("none", "identity", "id"):
            return 0.0
        if pol in ("fixed", "scalar"):
            return base_gate
        if pol in ("periodic", "every"):
            if k < 0:
                return base_gate
            return 1.0 if (k % self.resgrad_keep_every == 0) else base_gate
        if pol in ("tail", "last"):
            if k < 0 or K <= 0:
                return base_gate
            return 1.0 if k >= max(0, K - self.resgrad_keep_tail) else base_gate
        if pol in ("head", "first"):
            if k < 0:
                return base_gate
            return 1.0 if k < self.resgrad_keep_tail else base_gate
        return base_gate

    def set_resgrad_context(self, horizon_index: Optional[int] = None, total_horizon: Optional[int] = None):
        self.resgrad_current_horizon = -1 if horizon_index is None else int(horizon_index)
        self.resgrad_current_total_horizon = -1 if total_horizon is None else int(total_horizon)

    def predict_frame_from_history(
        self,
        history: torch.Tensor,
        stim_window: Optional[torch.Tensor] = None,
        ratio_collector: Optional[list] = None,
        gate_collector: Optional[list] = None,
    ) -> torch.Tensor:
        B, W, D = history.shape
        stim = self._pool_stim(stim_window, B, W, history.device, history.dtype)
        x = torch.cat([history, stim], dim=-1) if self.has_external_input else history
        x = x.reshape(B, -1)
        h = self.in_proj(x)
        gate = self._resgrad_gate_for_step()
        for block in self.blocks:
            h = block(
                h,
                residual_grad_gate=gate,
                resgrad_policy=self.resgrad_policy,
                resgrad_ratio_threshold=self.resgrad_ratio_threshold,
                ratio_collector=ratio_collector,
                gate_collector=gate_collector,
            )
        delta_or_frame = self.out(h)
        return history[:, -1] + delta_or_frame if self.residual else delta_or_frame

    def step_history(self, history: torch.Tensor, stim_window: Optional[torch.Tensor] = None) -> torch.Tensor:
        frame = self.predict_frame_from_history(history, stim_window)
        return torch.cat([history[:, 1:], frame.unsqueeze(1)], dim=1)

    def forward(
        self,
        stim_window: Optional[torch.Tensor],
        history: torch.Tensor,
        return_aux: bool = False,
        horizon_index: Optional[int] = None,
        total_horizon: Optional[int] = None,
        ratio_collector: Optional[list] = None,
    ):
        if horizon_index is not None or total_horizon is not None:
            self.set_resgrad_context(horizon_index=horizon_index, total_horizon=total_horizon)
        collector = ratio_collector
        gate_collector = [] if return_aux else None
        if return_aux and collector is None:
            collector = []
        frame = self.predict_frame_from_history(history, stim_window, ratio_collector=collector, gate_collector=gate_collector)
        pred = frame.unsqueeze(1)
        if return_aux:
            aux = {"pred_frame": frame}
            aux["resgrad_gate"] = (
                frame.new_tensor(float(sum(gate_collector) / len(gate_collector)))
                if gate_collector else frame.new_tensor(float(self._resgrad_gate_for_step(horizon_index)))
            )
            aux["resgrad_routing"] = frame.new_tensor(1.0 if self.resgrad_routing else 0.0)
            if collector:
                aux["resgrad_branch_residual_ratio"] = frame.new_tensor(float(sum(collector) / len(collector)))
            return pred, aux
        return pred
