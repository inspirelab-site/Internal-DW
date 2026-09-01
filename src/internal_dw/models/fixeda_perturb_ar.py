"""Clean fixed-A perturbation autoregressive model with channel-wise dense A.

This model is intentionally NOT a Mamba wrapper.  The only recurrent state is
``z_t`` and the only recurrent transition is

    z_{t+1} = A z_t + DeltaA_t z_t + b_t,

where ``A`` is a learned stable channel-wise dense macro-transition,
``DeltaA_t z_t`` is an explicitly returned low-rank context-dependent
perturbation, and ``b_t`` is an input/context forcing term.  No extra recurrent
branch, Mamba state, conv cache, or learned corrector state is maintained.

Important: the corrector is NOT produced by this model.  The corrector is a
loss-level object computed from the perturbation sequence ``r_t = DeltaA_t z_t``
and the same channel-wise dense ``A`` by solving a minimum-energy block
tridiagonal discrete corrector problem.  This prevents a free neural ``psi``
head from hiding accumulated drift.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .simple_ar import _StimPoolMixin


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(int(dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class FixedAPerturbTransport(nn.Module):
    """Channel-wise dense stable A plus low-rank context perturbation.

    The latent state is reshaped as [B, C, d].  Each channel has its own dense
    macro-transition A_c in R^{d x d}.  The matrix is stabilized by a differentiable
    spectral normalization guard in ``A_matrix()`` so ||A_c||_2 <= a_hard_max.
    This keeps the formal fixed-A transition intact without weakening it to a
    diagonal transition.
    """

    def __init__(
        self,
        hidden_dim: int,
        channels: int = 64,
        rank: int = 4,
        a_init_scale: float = 0.98,
        a_init_noise: float = 1e-3,
        a_hard_max: float = 0.9999,
        perturb_eps: float = 0.02,
        perturb_bound: str = "tanh",
        context_hidden: int = 512,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        channels = int(channels)
        if hidden_dim % channels != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by fixeda_channels={channels}")
        self.hidden_dim = hidden_dim
        self.channels = channels
        self.channel_dim = hidden_dim // channels
        self.rank = int(rank)
        self.perturb_eps = float(perturb_eps)
        self.perturb_bound = str(perturb_bound).lower()
        if self.perturb_bound not in {"tanh", "linear", "none"}:
            raise ValueError(f"Unsupported perturb_bound={perturb_bound!r}; choose tanh, linear, or none")
        self.a_hard_max = float(a_hard_max)

        C, d, r = self.channels, self.channel_dim, self.rank

        # Channel-wise full A, initialized near a stable identity.  It remains a
        # full dense matrix; A_matrix() applies a spectral-norm guard, not a
        # diagonal simplification.
        A = torch.eye(d).unsqueeze(0).repeat(C, 1, 1) * float(a_init_scale)
        if float(a_init_noise) > 0:
            A = A + float(a_init_noise) * torch.randn_like(A)
        self.A_raw = nn.Parameter(A)

        scale = 1.0 / math.sqrt(float(d))
        self.U = nn.Parameter(scale * torch.randn(C, d, r))
        self.V = nn.Parameter(scale * torch.randn(C, d, r))

        self.beta_net = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, int(context_hidden)),
            nn.GELU(),
            nn.Linear(int(context_hidden), C * r),
        )
        self.force = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, int(context_hidden)),
            nn.GELU(),
            nn.Linear(int(context_hidden), hidden_dim),
        )
        # Start from nearly pure fixed-A dynamics; allow forcing to open during training.
        nn.init.zeros_(self.force[-1].weight)
        nn.init.zeros_(self.force[-1].bias)

    def A_matrix(self) -> torch.Tensor:
        """Return stabilized channel-wise dense macro-transition [C,d,d]."""
        A = self.A_raw
        # SVD is cheap at the intended channel_dim (e.g. 8, 16, 32) and avoids
        # silently turning the method into a diagonal transition.  The clamp is a
        # hard guard; a separate soft spectral loss can use a lower threshold.
        sigma = torch.linalg.svdvals(A.float()).amax(dim=-1).to(dtype=A.dtype, device=A.device)
        scale = (self.a_hard_max / sigma.clamp_min(1e-8)).clamp(max=1.0)
        return A * scale.view(-1, 1, 1)

    def apply_A(self, z_flat: torch.Tensor, A: Optional[torch.Tensor] = None) -> torch.Tensor:
        if A is None:
            A = self.A_matrix()
        B = int(z_flat.shape[0])
        C, d = self.channels, self.channel_dim
        z = z_flat.reshape(B, C, d)
        A = A.to(device=z_flat.device, dtype=z_flat.dtype)
        y = torch.einsum("cij,bcj->bci", A, z)
        return y.reshape(B, self.hidden_dim)

    def spectral_loss(self, spec_max: float = 0.999) -> torch.Tensor:
        A = self.A_matrix()
        sigma = torch.linalg.svdvals(A.float()).amax(dim=-1)
        return F.relu(sigma - float(spec_max)).square().mean().to(dtype=A.dtype)

    def forward(self, z_flat: torch.Tensor, token: torch.Tensor):
        B = int(z_flat.shape[0])
        C, d, r = self.channels, self.channel_dim, self.rank
        z = z_flat.reshape(B, C, d)
        A = self.A_matrix().to(dtype=z_flat.dtype, device=z_flat.device)

        base = torch.einsum("cij,bcj->bci", A, z)
        beta_in = torch.cat([z_flat, token], dim=-1)
        beta_raw = self.beta_net(beta_in).reshape(B, C, r)
        if self.perturb_bound == "tanh":
            beta = torch.tanh(beta_raw) * self.perturb_eps
        elif self.perturb_bound == "linear":
            beta = beta_raw * self.perturb_eps
        else:
            beta = beta_raw

        # DeltaA_t z_t = U diag(beta_t) V^T z_t, independently per channel.
        vtz = torch.einsum("bci,cir->bcr", z, self.V)
        delta = torch.einsum("bcr,cir->bci", beta * vtz, self.U)
        force = self.force(token).reshape(B, C, d)
        z_next = base + delta + force
        z_next_flat = z_next.reshape(B, self.hidden_dim)

        base_flat = base.reshape(B, -1)
        delta_flat = delta.reshape(B, -1)
        force_flat = force.reshape(B, -1)
        base_norm = base_flat.norm(dim=-1).mean().detach().clamp_min(1e-8)
        delta_loss = (delta_flat.norm(dim=-1).mean() / base_norm).square()
        force_loss = (force_flat.norm(dim=-1).mean() / base_norm).square()

        with torch.no_grad():
            sigma = torch.linalg.svdvals(A.float()).amax().to(z_flat.dtype)
            base_norm_ng = base_flat.norm(dim=-1).mean().clamp_min(1e-8)
            aux = {
                "fixeda_A_sigma": sigma.detach(),
                "fixeda_delta_rel": (delta_flat.norm(dim=-1).mean() / base_norm_ng).detach(),
                "fixeda_force_rel": (force_flat.norm(dim=-1).mean() / base_norm_ng).detach(),
                "fixeda_beta_abs": beta.abs().mean().detach(),
            }

        aux.update({
            "fixeda_z_prev": z_flat,
            "fixeda_base": base_flat,
            "fixeda_delta": delta_flat,
            "fixeda_force": force_flat,
            "fixeda_delta_loss": delta_loss,
            "fixeda_force_loss": force_loss,
        })
        return z_next_flat, aux


class FixedAPerturbARModel(nn.Module, _StimPoolMixin):
    """Autoregressive model with a clean fixed-A latent transition."""

    is_standard_autoregressive = True
    is_recurrent_state_ar = True
    is_fixeda_perturb_ar = True

    def __init__(
        self,
        state_dim: int,
        input_dim: int = 0,
        state_shape: Optional[Sequence[int]] = None,
        hidden_dim: int = 512,
        encoder_depth: int = 2,
        decoder_depth: int = 2,
        dropout: float = 0.0,
        has_external_input: bool = True,
        residual: bool = True,
        fixeda_channels: int = 64,
        perturb_rank: int = 4,
        perturb_eps: float = 0.02,
        perturb_bound: str = "tanh",
        a_init_scale: float = 0.98,
        a_init_noise: float = 1e-3,
        corrector_rank: int = 0,  # kept for CLI compatibility; intentionally unused
        decoder_use_context: bool = True,
        fixeda_delta_weight: float = 0.0,
        fixeda_force_weight: float = 0.0,
        fixeda_spec_weight: float = 0.0,
        fixeda_spec_max: float = 0.999,
        fixeda_corrector_weight: float = 0.0,
        fixeda_mean_weight: float = 0.0,
        fixeda_psi_weight: float = 0.0,
        fixeda_smooth_weight: float = 0.0,
    ):
        super().__init__()
        _ = corrector_rank  # corrector is solved in the loss, not learned here
        self.state_dim = int(state_dim)
        self.input_dim = int(input_dim)
        self.state_shape = tuple(int(x) for x in state_shape) if state_shape is not None else None
        self.hidden_dim = int(hidden_dim)
        self.has_external_input = bool(has_external_input)
        self.residual = bool(residual)
        self.decoder_use_context = bool(decoder_use_context)

        self.fixeda_delta_weight = float(fixeda_delta_weight)
        self.fixeda_force_weight = float(fixeda_force_weight)
        self.fixeda_spec_weight = float(fixeda_spec_weight)
        self.fixeda_spec_max = float(fixeda_spec_max)
        self.fixeda_corrector_weight = float(fixeda_corrector_weight)
        self.fixeda_mean_weight = float(fixeda_mean_weight)
        self.fixeda_psi_weight = float(fixeda_psi_weight)
        self.fixeda_smooth_weight = float(fixeda_smooth_weight)

        in_dim = self.state_dim + (self.input_dim if self.has_external_input else 0)
        enc_layers = [nn.LayerNorm(in_dim), nn.Linear(in_dim, self.hidden_dim), nn.GELU(), nn.Dropout(float(dropout))]
        for _ in range(max(0, int(encoder_depth) - 1)):
            enc_layers += [nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.hidden_dim), nn.GELU(), nn.Dropout(float(dropout))]
        self.encoder = nn.Sequential(*enc_layers)

        self.transport = FixedAPerturbTransport(
            hidden_dim=self.hidden_dim,
            channels=int(fixeda_channels),
            rank=int(perturb_rank),
            a_init_scale=float(a_init_scale),
            a_init_noise=float(a_init_noise),
            perturb_eps=float(perturb_eps),
            perturb_bound=str(perturb_bound),
            context_hidden=max(128, self.hidden_dim),
        )
        self.z_norm = RMSNorm(self.hidden_dim)
        dec_in_dim = self.hidden_dim + (self.hidden_dim if self.decoder_use_context else 0)
        dec_layers = [nn.LayerNorm(dec_in_dim), nn.Linear(dec_in_dim, self.hidden_dim), nn.GELU(), nn.Dropout(float(dropout))]
        for _ in range(max(0, int(decoder_depth) - 1)):
            dec_layers += [nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.hidden_dim), nn.GELU(), nn.Dropout(float(dropout))]
        dec_layers += [nn.Linear(self.hidden_dim, self.state_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    @property
    def fixeda_channels(self) -> int:
        return int(self.transport.channels)

    @property
    def fixeda_channel_dim(self) -> int:
        return int(self.transport.channel_dim)

    def init_state(self, batch_size: int, device=None, dtype=None):
        device = device if device is not None else next(self.parameters()).device
        dtype = dtype if dtype is not None else next(self.parameters()).dtype
        return torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)

    def detach_state(self, state):
        return state.detach()

    def fixeda_A_matrix(self) -> torch.Tensor:
        return self.transport.A_matrix()

    def fixeda_apply_A(self, z_flat: torch.Tensor, A: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.transport.apply_A(z_flat, A=A)

    def fixeda_spectral_loss(self) -> torch.Tensor:
        if self.fixeda_spec_weight == 0.0:
            return self.transport.A_raw.new_tensor(0.0)
        return self.transport.spectral_loss(self.fixeda_spec_max)

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
        return self.encoder(inp)

    def step(
        self,
        state,
        x_t: torch.Tensor,
        stim_t: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        horizon_index: Optional[int] = None,
    ):
        del horizon_index
        B = x_t.shape[0]
        if state is None:
            state = self.init_state(B, x_t.device, x_t.dtype)
        z = state
        token = self._tokenize(x_t, stim_t)
        z_next, aux_transport = self.transport(z, token)

        z_dec = self.z_norm(z_next)
        dec_in = torch.cat([z_dec, token], dim=-1) if self.decoder_use_context else z_dec
        delta_or_frame = self.decoder(dec_in)
        x_flat = self._flatten_state(x_t)
        pred_flat = x_flat + delta_or_frame if self.residual else delta_or_frame
        pred = self._reshape_pred(pred_flat, x_t)

        if return_aux:
            aux = {
                "h_next": z_next,
                "z_next": z_next,
                "hidden_norm": z_next.norm(dim=-1).mean().detach(),
                "alpha_mean": aux_transport.get("fixeda_A_sigma", z_next.new_tensor(0.0)).detach(),
                "alpha_min": aux_transport.get("fixeda_A_sigma", z_next.new_tensor(0.0)).detach(),
                "alpha_max": aux_transport.get("fixeda_A_sigma", z_next.new_tensor(0.0)).detach(),
            }
            aux.update(aux_transport)
            return pred, z_next, aux
        return pred, z_next

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
        h = self.init_state(B, history.device, history.dtype)
        for j in range(W):
            stim_j = stim_window[:, j] if stim_window is not None else None
            _, h = self.step(h, history[:, j], stim_j, return_aux=False)
        pred, _ = self.step(h, history[:, -1], stim_window[:, -1] if stim_window is not None else None, return_aux=False)
        if W <= 0:
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
