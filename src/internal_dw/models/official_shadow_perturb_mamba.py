"""Official-state shadow-perturbed Mamba.

This replaces the earlier rolling-token-buffer shadow prototype.  The Mamba
backbone here maintains the true per-layer Mamba inference states
(conv_state, ssm_state).  The additional shadow state follows

    z_{t+1} = (A0 + U diag(beta_t) V^T) z_t + b_theta(x_t,u_t),
    beta_t uses a configurable bound mode. By default tanh keeps old behavior;
    linear removes the hard bound and only scales raw beta by shadow_perturb_eps.

The shadow state conditions the Mamba token and readout, so this is a
transition-level controlled Mamba, not output-level weighted fusion.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .official_state_mamba import RMSNorm, StatefulMambaBlock, MambaStackState, detach_mamba_stack_state
from .simple_ar import _StimPoolMixin


class ChannelWiseStablePerturbTransport(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        channels: int = 64,
        rank: int = 4,
        a_init_scale: float = 0.98,
        a_init_noise: float = 1e-3,
        perturb_eps: float = 0.02,
        perturb_bound: str = "tanh",
        beta_hidden: int = 512,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        channels = int(channels)
        if hidden_dim % channels != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by shadow_channels={channels}")
        self.hidden_dim = hidden_dim
        self.channels = channels
        self.channel_dim = hidden_dim // channels
        self.rank = int(rank)
        self.perturb_eps = float(perturb_eps)
        self.perturb_bound = str(perturb_bound).lower()
        if self.perturb_bound not in {"tanh", "linear", "none"}:
            raise ValueError(f"Unsupported perturb_bound={perturb_bound!r}; choose tanh, linear, or none")

        d = self.channel_dim
        eye = torch.eye(d).unsqueeze(0).repeat(channels, 1, 1)
        A = float(a_init_scale) * eye
        if float(a_init_noise) > 0:
            A = A + float(a_init_noise) * torch.randn_like(A) / math.sqrt(float(d))
        self.A = nn.Parameter(A)

        scale = 1.0 / math.sqrt(float(d))
        self.U = nn.Parameter(scale * torch.randn(channels, d, self.rank))
        self.V = nn.Parameter(scale * torch.randn(channels, d, self.rank))
        self.beta_net = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, int(beta_hidden)),
            nn.GELU(),
            nn.Linear(int(beta_hidden), channels * self.rank),
        )
        self.force = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.force[-1].weight)
        nn.init.zeros_(self.force[-1].bias)

    def effective_A(self) -> torch.Tensor:
        return self.A

    def apply_A(self, z_flat: torch.Tensor) -> torch.Tensor:
        B = z_flat.shape[0]
        z = z_flat.reshape(B, self.channels, self.channel_dim)
        out = torch.einsum("bci,coi->bco", z, self.A)
        return out.reshape(B, self.hidden_dim)

    def forward(self, z_flat: torch.Tensor, token: torch.Tensor):
        B = z_flat.shape[0]
        C, d, r = self.channels, self.channel_dim, self.rank
        z = z_flat.reshape(B, C, d)

        base = torch.einsum("bci,coi->bco", z, self.A)
        beta_in = torch.cat([z_flat, token], dim=-1)
        beta_raw = self.beta_net(beta_in).reshape(B, C, r)
        if self.perturb_bound == "tanh":
            # Old behavior: hard bound |beta| <= perturb_eps.
            beta = torch.tanh(beta_raw) * self.perturb_eps
        elif self.perturb_bound == "linear":
            # No hard clipping/saturation; perturb_eps is only a scale factor.
            # Delta loss can still softly control the realized perturbation size.
            beta = beta_raw * self.perturb_eps
        else:  # "none"
            # Fully unbounded raw coefficients. Use with a nonzero delta loss.
            beta = beta_raw
        vtz = torch.einsum("bci,cir->bcr", z, self.V)
        delta = torch.einsum("bcr,cir->bci", beta * vtz, self.U)
        force = self.force(token).reshape(B, C, d)
        z_next = base + delta + force
        z_next_flat = z_next.reshape(B, self.hidden_dim)

        base_norm_for_loss = base.reshape(B, -1).norm(dim=-1).mean().detach().clamp_min(1e-8)
        delta_norm_for_loss = delta.reshape(B, -1).norm(dim=-1).mean()
        delta_loss = (delta_norm_for_loss / base_norm_for_loss).square()
        with torch.no_grad():
            A_norm = self.A.reshape(C, -1).norm(dim=-1).mean()
            base_norm = base.reshape(B, -1).norm(dim=-1).mean().clamp_min(1e-8)
            delta_norm = delta.reshape(B, -1).norm(dim=-1).mean()
            force_norm = force.reshape(B, -1).norm(dim=-1).mean()
            beta_abs = beta.abs().mean()
            try:
                sigma = torch.linalg.svdvals(self.A.float()).amax(dim=-1).mean().to(z_flat.dtype)
            except Exception:
                sigma = z_flat.new_tensor(0.0)
        aux = {
            "shadow_delta_loss": delta_loss,
            "shadow_A_norm": A_norm.detach(),
            "shadow_A_sigma": sigma.detach(),
            "shadow_delta_rel": (delta_norm / base_norm).detach(),
            "shadow_force_rel": (force_norm / base_norm).detach(),
            "shadow_beta_abs": beta_abs.detach(),
        }
        return z_next_flat, aux


class OfficialShadowPerturbMambaARModel(nn.Module, _StimPoolMixin):
    is_standard_autoregressive = True
    is_recurrent_state_ar = True
    is_official_shadow_perturb_mamba = True

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
        shadow_channels: int = 64,
        perturb_rank: int = 4,
        perturb_eps: float = 0.02,
        perturb_bound: str = "tanh",
        a_init_scale: float = 0.98,
        a_init_noise: float = 1e-3,
        shadow_condition_scale: float = 1.0,
        shadow_output_scale: float = 1.0,
        shadow_kg_horizon: int = 0,
        shadow_kg_weight: float = 0.0,
        shadow_delta_weight: float = 0.0,
        shadow_spec_weight: float = 0.0,
        shadow_spec_max: float = 0.999,
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
        self.shadow_condition_scale = float(shadow_condition_scale)
        self.shadow_output_scale = float(shadow_output_scale)
        self.shadow_kg_horizon = int(shadow_kg_horizon)
        self.shadow_kg_weight = float(shadow_kg_weight)
        self.shadow_delta_weight = float(shadow_delta_weight)
        self.shadow_spec_weight = float(shadow_spec_weight)
        self.shadow_spec_max = float(shadow_spec_max)

        in_dim = self.state_dim + (self.input_dim if self.has_external_input else 0)
        self.in_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.shadow = ChannelWiseStablePerturbTransport(
            hidden_dim=self.hidden_dim,
            channels=int(shadow_channels),
            rank=int(perturb_rank),
            a_init_scale=float(a_init_scale),
            a_init_noise=float(a_init_noise),
            perturb_eps=float(perturb_eps),
            perturb_bound=str(perturb_bound),
            beta_hidden=max(128, self.hidden_dim),
        )
        self.shadow_norm = RMSNorm(self.hidden_dim)
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
        self.out = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.state_dim),
        )
        if self.shadow_kg_weight != 0.0 and self.shadow_kg_horizon > 0:
            self.target_encoder = nn.Sequential(nn.LayerNorm(self.state_dim), nn.Linear(self.state_dim, self.hidden_dim))
        else:
            self.target_encoder = None

    def init_state(self, batch_size: int, device=None, dtype=None):
        device = device if device is not None else next(self.parameters()).device
        dtype = dtype if dtype is not None else next(self.parameters()).dtype
        h_m = tuple(block.init_state(batch_size, device=device, dtype=dtype) for block in self.blocks)
        z = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        return (h_m, z)

    def detach_state(self, state):
        h_m, z = state
        return (detach_mamba_stack_state(h_m), z.detach())

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

    def _tokenize(self, x_t: torch.Tensor, stim_t: Optional[torch.Tensor]) -> torch.Tensor:
        B = x_t.shape[0]
        x_flat = self._flatten_state(x_t)
        stim = self._pool_stim_point(stim_t, B, x_t.device, x_t.dtype)
        inp = torch.cat([x_flat, stim], dim=-1) if self.has_external_input else x_flat
        return self.in_proj(inp)

    def step(self, state, x_t: torch.Tensor, stim_t: Optional[torch.Tensor] = None, return_aux: bool = False, horizon_index: Optional[int] = None):
        B = x_t.shape[0]
        if state is None:
            state = self.init_state(B, x_t.device, x_t.dtype)
        h_m, z_shadow = state

        token = self._tokenize(x_t, stim_t)
        z_next, aux_shadow = self.shadow(z_shadow, token)
        conditioned = token + self.shadow_condition_scale * self.shadow_norm(z_next)

        next_states = []
        y = conditioned
        dt_means, dt_mins, dt_maxs = [], [], []
        ssm_norms, conv_norms = [], []
        for block, layer_state in zip(self.blocks, h_m):
            y, next_state, aux_l = block.step(y, layer_state)
            next_states.append(next_state)
            dt_means.append(aux_l["dt_mean"])
            dt_mins.append(aux_l["dt_min"])
            dt_maxs.append(aux_l["dt_max"])
            ssm_norms.append(aux_l["ssm_state_norm"])
            conv_norms.append(aux_l["conv_state_norm"])
        h_m_next = tuple(next_states)

        readout_state = y + self.shadow_output_scale * self.shadow_norm(z_next)
        delta_or_frame = self.out(readout_state)
        x_flat = self._flatten_state(x_t)
        pred_flat = x_flat + delta_or_frame if self.residual else delta_or_frame
        pred = self._reshape_pred(pred_flat, x_t)
        next_state = (h_m_next, z_next)

        if return_aux:
            aux = {
                "h_next": next_state,
                "z_next": z_next,
                "hidden_norm": (torch.stack(ssm_norms).mean() if ssm_norms else pred_flat.new_tensor(0.0)),
                "mamba_ssm_state_norm": torch.stack(ssm_norms).mean() if ssm_norms else pred_flat.new_tensor(0.0),
                "mamba_conv_state_norm": torch.stack(conv_norms).mean() if conv_norms else pred_flat.new_tensor(0.0),
                "mamba_dt_mean": torch.stack(dt_means).mean() if dt_means else pred_flat.new_tensor(0.0),
                "mamba_dt_min": torch.stack(dt_mins).min() if dt_mins else pred_flat.new_tensor(0.0),
                "mamba_dt_max": torch.stack(dt_maxs).max() if dt_maxs else pred_flat.new_tensor(0.0),
                "alpha_mean": torch.stack(dt_means).mean() if dt_means else pred_flat.new_tensor(0.0),
                "alpha_min": torch.stack(dt_mins).min() if dt_mins else pred_flat.new_tensor(0.0),
                "alpha_max": torch.stack(dt_maxs).max() if dt_maxs else pred_flat.new_tensor(0.0),
            }
            aux.update(aux_shadow)
            return pred, next_state, aux
        return pred, next_state

    def burn_in(self, x_seq: torch.Tensor, stim_seq: Optional[torch.Tensor] = None, h0=None, detach: bool = False):
        B, T = x_seq.shape[:2]
        s = h0 if h0 is not None else self.init_state(B, x_seq.device, x_seq.dtype)
        ctx = torch.no_grad() if detach else torch.enable_grad()
        with ctx:
            for j in range(T):
                stim_j = stim_seq[:, j] if stim_seq is not None else None
                _, s = self.step(s, x_seq[:, j], stim_j, return_aux=False)
        return self.detach_state(s) if detach else s

    def predict_frame_from_history(self, history: torch.Tensor, stim_window: Optional[torch.Tensor] = None):
        B, W = history.shape[:2]
        s = self.init_state(B, history.device, history.dtype)
        stim = self._pool_stim(stim_window, B, W, history.device, history.dtype) if self.has_external_input else None
        pred = None
        for j in range(W):
            stim_j = stim[:, j] if stim is not None else None
            pred, s = self.step(s, history[:, j], stim_j, return_aux=False)
        if pred is None:
            raise ValueError("history must contain at least one frame")
        return pred

    def step_history(self, history: torch.Tensor, stim_window: Optional[torch.Tensor] = None):
        frame = self.predict_frame_from_history(history, stim_window)
        return torch.cat([history[:, 1:], frame.unsqueeze(1)], dim=1)

    def forward(self, stim_window: Optional[torch.Tensor], history: torch.Tensor, return_aux: bool = False):
        frame = self.predict_frame_from_history(history, stim_window)
        pred = frame.unsqueeze(1)
        if return_aux:
            return pred, {"pred_frame": frame}
        return pred

    def shadow_spectral_loss(self) -> torch.Tensor:
        if self.shadow_spec_weight == 0.0:
            return self.shadow.A.new_tensor(0.0)
        sigma = torch.linalg.svdvals(self.shadow.A.float()).amax(dim=-1).to(self.shadow.A.dtype)
        return F.relu(sigma - self.shadow_spec_max).square().mean()

    def shadow_folded_loss(self, z0: torch.Tensor, future_state: torch.Tensor) -> torch.Tensor:
        if self.target_encoder is None or self.shadow_kg_weight == 0.0 or self.shadow_kg_horizon <= 0:
            return z0.new_tensor(0.0)
        B, K = future_state.shape[:2]
        K = min(int(K), int(self.shadow_kg_horizon))
        if K <= 0:
            return z0.new_tensor(0.0)
        flat = future_state[:, :K].reshape(B * K, -1)
        target = self.target_encoder(flat).reshape(B, K, self.hidden_dim).detach()
        z = z0
        losses = []
        for k in range(K):
            losses.append(F.mse_loss(z, target[:, k]))
            if k != K - 1:
                z = self.shadow.apply_A(z)
        return torch.stack(losses).mean()
