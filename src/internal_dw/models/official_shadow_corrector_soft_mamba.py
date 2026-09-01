"""Soft corrector-constrained shadow-perturbed official Mamba.

This file intentionally keeps the original official_shadow_perturb_mamba
forward path unchanged:

    z_t = A z_{t-1} + DeltaA_t z_{t-1} + force(x_t)

The only new component is a corrector head psi_t and auxiliary losses that
encourage the realized perturbation DeltaA_t z_{t-1} to be explainable as a
removable telescoping corrector:

    DeltaA_t z_{t-1} \approx psi_t - A psi_{t-1}.

Use model_name="official_shadow_corrector_soft_mamba" to compare against the
unmodified perturb baseline. The base perturb block, Mamba stack, readout, and
all non-corrector hyperparameters are copied from the original model.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .official_shadow_perturb_mamba import (
    ChannelWiseStablePerturbTransport,
    OfficialShadowPerturbMambaARModel,
)
from .official_state_mamba import detach_mamba_stack_state


_EPS = 1e-8


def _parse_horizons(value) -> tuple[int, ...]:
    if value is None:
        return (4, 8, 16)
    if isinstance(value, (list, tuple)):
        out = [int(v) for v in value if int(v) > 0]
        return tuple(out) if out else (4, 8, 16)
    s = str(value).replace(";", ",").replace(" ", "")
    if not s:
        return (4, 8, 16)
    out = []
    for part in s.split(","):
        if part:
            v = int(part)
            if v > 0:
                out.append(v)
    return tuple(out) if out else (4, 8, 16)


class ChannelWiseSoftCorrectorTransport(ChannelWiseStablePerturbTransport):
    """Original A+DeltaA transport plus a soft corrector diagnostic head."""

    def __init__(self, *args, corrector_hidden: int = 512, **kwargs):
        super().__init__(*args, **kwargs)
        corrector_hidden = int(corrector_hidden)
        self.psi_net = nn.Sequential(
            nn.LayerNorm(2 * self.hidden_dim),
            nn.Linear(2 * self.hidden_dim, corrector_hidden),
            nn.GELU(),
            nn.Linear(corrector_hidden, self.hidden_dim),
        )
        # Start from a no-corrector explanation; the task forward is unchanged.
        nn.init.zeros_(self.psi_net[-1].weight)
        nn.init.zeros_(self.psi_net[-1].bias)

    @classmethod
    def from_existing(cls, old: ChannelWiseStablePerturbTransport, corrector_hidden: int = 512):
        new = cls(
            hidden_dim=old.hidden_dim,
            channels=old.channels,
            rank=old.rank,
            a_init_scale=1.0,
            a_init_noise=0.0,
            perturb_eps=old.perturb_eps,
            perturb_bound=old.perturb_bound,
            beta_hidden=max(128, old.hidden_dim),
            corrector_hidden=corrector_hidden,
        )
        with torch.no_grad():
            new.A.copy_(old.A)
            new.U.copy_(old.U)
            new.V.copy_(old.V)
        new.beta_net.load_state_dict(old.beta_net.state_dict())
        new.force.load_state_dict(old.force.state_dict())
        return new

    def forward(self, z_flat: torch.Tensor, token: torch.Tensor, psi_prev: torch.Tensor):
        B = z_flat.shape[0]
        C, d, r = self.channels, self.channel_dim, self.rank
        z = z_flat.reshape(B, C, d)
        psi_prev_c = psi_prev.reshape(B, C, d)

        base = torch.einsum("bci,coi->bco", z, self.A)
        beta_in = torch.cat([z_flat, token], dim=-1)
        beta_raw = self.beta_net(beta_in).reshape(B, C, r)
        if self.perturb_bound == "tanh":
            beta = torch.tanh(beta_raw) * self.perturb_eps
        elif self.perturb_bound == "linear":
            beta = beta_raw * self.perturb_eps
        else:
            beta = beta_raw
        vtz = torch.einsum("bci,cir->bcr", z, self.V)
        delta = torch.einsum("bcr,cir->bci", beta * vtz, self.U)
        force = self.force(token).reshape(B, C, d)

        # Original perturb forward path is preserved exactly.
        z_next = base + delta + force
        z_next_flat = z_next.reshape(B, self.hidden_dim)

        psi = self.psi_net(beta_in).reshape(B, C, d)
        A_psi_prev = torch.einsum("bci,coi->bco", psi_prev_c, self.A)
        corr = psi - A_psi_prev
        corr_err = delta - corr

        base_norm_for_loss = base.reshape(B, -1).norm(dim=-1).mean().detach().clamp_min(_EPS)
        delta_norm_for_loss = delta.reshape(B, -1).norm(dim=-1).mean()
        delta_loss = (delta_norm_for_loss / base_norm_for_loss).square()

        delta_vec = delta.reshape(B, -1)
        corr_vec = corr.reshape(B, -1)
        corr_err_vec = corr_err.reshape(B, -1)
        delta_denom = delta_vec.detach().norm(dim=-1).mean().clamp_min(_EPS)
        corrector_loss = (corr_err_vec.norm(dim=-1).mean() / delta_denom).square()
        psi_loss = (psi.reshape(B, -1).norm(dim=-1).mean() / base_norm_for_loss).square()

        with torch.no_grad():
            A_norm = self.A.reshape(C, -1).norm(dim=-1).mean()
            base_norm = base.reshape(B, -1).norm(dim=-1).mean().clamp_min(_EPS)
            delta_norm = delta.reshape(B, -1).norm(dim=-1).mean()
            force_norm = force.reshape(B, -1).norm(dim=-1).mean()
            beta_abs = beta.abs().mean()
            corr_norm = corr_vec.norm(dim=-1).mean()
            corr_err_norm = corr_err_vec.norm(dim=-1).mean()
            psi_norm = psi.reshape(B, -1).norm(dim=-1).mean()
            try:
                sigma = torch.linalg.svdvals(self.A.float()).amax(dim=-1).mean().to(z_flat.dtype)
            except Exception:
                sigma = z_flat.new_tensor(0.0)

        aux = {
            "shadow_delta_loss": delta_loss,
            "shadow_corrector_loss": corrector_loss,
            "shadow_psi_loss": psi_loss,
            "shadow_A_norm": A_norm.detach(),
            "shadow_A_sigma": sigma.detach(),
            "shadow_delta_rel": (delta_norm / base_norm).detach(),
            "shadow_force_rel": (force_norm / base_norm).detach(),
            "shadow_beta_abs": beta_abs.detach(),
            "shadow_corrector_rel": (corr_err_norm / delta_norm.clamp_min(_EPS)).detach(),
            "shadow_corrector_norm_rel": (corr_norm / delta_norm.clamp_min(_EPS)).detach(),
            "shadow_psi_norm_rel": (psi_norm / base_norm).detach(),
            # Keep graph for optional cumulative-correction loss.
            "shadow_correction_vec": delta_vec,
        }
        return z_next_flat, psi.reshape(B, self.hidden_dim), aux


class OfficialShadowCorrectorSoftMambaARModel(OfficialShadowPerturbMambaARModel):
    is_standard_autoregressive = True
    is_recurrent_state_ar = True
    is_official_shadow_perturb_mamba = True
    is_official_shadow_corrector_soft_mamba = True

    def __init__(
        self,
        *args,
        shadow_corrector_weight: float = 0.0,
        shadow_cum_weight: float = 0.0,
        shadow_psi_weight: float = 0.0,
        shadow_cum_horizons="4,8,16",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.shadow_corrector_weight = float(shadow_corrector_weight)
        self.shadow_cum_weight = float(shadow_cum_weight)
        self.shadow_psi_weight = float(shadow_psi_weight)
        self.shadow_eta_weight = 0.0
        self.shadow_cum_horizons = _parse_horizons(shadow_cum_horizons)
        # Replace the transport with a corrector-aware copy, preserving all old
        # perturb parameters exactly.
        self.shadow = ChannelWiseSoftCorrectorTransport.from_existing(
            self.shadow,
            corrector_hidden=max(128, self.hidden_dim),
        )

    def init_state(self, batch_size: int, device=None, dtype=None):
        h_m, z = super().init_state(batch_size, device=device, dtype=dtype)
        psi = torch.zeros_like(z)
        return (h_m, z, psi)

    def detach_state(self, state):
        h_m, z, psi = state
        return (detach_mamba_stack_state(h_m), z.detach(), psi.detach())

    def step(self, state, x_t: torch.Tensor, stim_t: Optional[torch.Tensor] = None, return_aux: bool = False, horizon_index: Optional[int] = None):
        B = x_t.shape[0]
        if state is None:
            state = self.init_state(B, x_t.device, x_t.dtype)
        h_m, z_shadow, psi_prev = state

        token = self._tokenize(x_t, stim_t)
        z_next, psi_next, aux_shadow = self.shadow(z_shadow, token, psi_prev)
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
        next_state = (h_m_next, z_next, psi_next)

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

    def shadow_cumulative_loss(self, correction_seq: torch.Tensor) -> torch.Tensor:
        """Sublinear-drift penalty for correction vectors [B, L, H]."""
        if self.shadow_cum_weight == 0.0 or correction_seq.numel() == 0:
            return correction_seq.new_tensor(0.0)
        B, L = correction_seq.shape[:2]
        if L <= 1:
            return correction_seq.new_tensor(0.0)
        prefix = torch.cat([correction_seq.new_zeros(B, 1, correction_seq.shape[-1]), torch.cumsum(correction_seq, dim=1)], dim=1)
        step_norm = correction_seq.detach().norm(dim=-1).mean().clamp_min(_EPS)
        losses = []
        for K in self.shadow_cum_horizons:
            K = int(K)
            if K <= 0 or K > L:
                continue
            window_sum = prefix[:, K:] - prefix[:, :-K]
            ratio = window_sum.norm(dim=-1) / (float(K) * step_norm)
            losses.append(ratio.square().mean())
        return torch.stack(losses).mean() if losses else correction_seq.new_tensor(0.0)
