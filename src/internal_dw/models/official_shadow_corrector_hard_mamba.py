"""Hard corrector-reparameterized official Mamba.

This is the stricter version of the homogenization-inspired corrector idea.
The raw DeltaA contribution is not added directly to the shadow state. Instead,
the local correction is forced to enter through a telescoping corrector:

    z_t = A z_{t-1} + force(x_t) + (psi_t - A psi_{t-1}) + eta_t.

The optional eta_t is a small residual derived from the original low-rank
perturb block and scaled by shadow_eta_scale. Set shadow_eta_scale=0 for the
pure hard corrector version.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .official_shadow_corrector_soft_mamba import (
    _EPS,
    _parse_horizons,
    OfficialShadowCorrectorSoftMambaARModel,
)
from .official_shadow_perturb_mamba import ChannelWiseStablePerturbTransport


class ChannelWiseHardCorrectorTransport(ChannelWiseStablePerturbTransport):
    def __init__(self, *args, corrector_hidden: int = 512, eta_scale: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.eta_scale = float(eta_scale)
        corrector_hidden = int(corrector_hidden)
        self.psi_net = nn.Sequential(
            nn.LayerNorm(2 * self.hidden_dim),
            nn.Linear(2 * self.hidden_dim, corrector_hidden),
            nn.GELU(),
            nn.Linear(corrector_hidden, self.hidden_dim),
        )
        nn.init.zeros_(self.psi_net[-1].weight)
        nn.init.zeros_(self.psi_net[-1].bias)

    @classmethod
    def from_existing(cls, old: ChannelWiseStablePerturbTransport, corrector_hidden: int = 512, eta_scale: float = 0.0):
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
            eta_scale=eta_scale,
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
        raw_delta = torch.einsum("bcr,cir->bci", beta * vtz, self.U)
        force = self.force(token).reshape(B, C, d)

        psi = self.psi_net(beta_in).reshape(B, C, d)
        A_psi_prev = torch.einsum("bci,coi->bco", psi_prev_c, self.A)
        corrector = psi - A_psi_prev
        eta = self.eta_scale * raw_delta

        # Hard corrector forward: no free raw DeltaA drift except optional eta.
        z_next = base + force + corrector + eta
        z_next_flat = z_next.reshape(B, self.hidden_dim)

        base_norm_for_loss = base.reshape(B, -1).norm(dim=-1).mean().detach().clamp_min(_EPS)
        raw_delta_norm = raw_delta.reshape(B, -1).norm(dim=-1).mean()
        eta_norm = eta.reshape(B, -1).norm(dim=-1).mean()
        eta_loss = (eta_norm / base_norm_for_loss).square()
        psi_loss = (psi.reshape(B, -1).norm(dim=-1).mean() / base_norm_for_loss).square()

        raw_delta_vec = raw_delta.reshape(B, -1)
        corrector_vec = corrector.reshape(B, -1)
        eta_vec = eta.reshape(B, -1)

        with torch.no_grad():
            A_norm = self.A.reshape(C, -1).norm(dim=-1).mean()
            base_norm = base.reshape(B, -1).norm(dim=-1).mean().clamp_min(_EPS)
            force_norm = force.reshape(B, -1).norm(dim=-1).mean()
            beta_abs = beta.abs().mean()
            corr_norm = corrector_vec.norm(dim=-1).mean()
            psi_norm = psi.reshape(B, -1).norm(dim=-1).mean()
            try:
                sigma = torch.linalg.svdvals(self.A.float()).amax(dim=-1).mean().to(z_flat.dtype)
            except Exception:
                sigma = z_flat.new_tensor(0.0)

        aux = {
            "shadow_delta_loss": eta_loss,
            "shadow_eta_loss": eta_loss,
            "shadow_psi_loss": psi_loss,
            "shadow_A_norm": A_norm.detach(),
            "shadow_A_sigma": sigma.detach(),
            "shadow_delta_rel": (eta_norm / base_norm).detach(),
            "shadow_raw_delta_rel": (raw_delta_norm / base_norm).detach(),
            "shadow_force_rel": (force_norm / base_norm).detach(),
            "shadow_beta_abs": beta_abs.detach(),
            "shadow_corrector_rel": z_flat.new_tensor(0.0),
            "shadow_corrector_norm_rel": (corr_norm / base_norm).detach(),
            "shadow_psi_norm_rel": (psi_norm / base_norm).detach(),
            "shadow_eta_scale": z_flat.new_tensor(self.eta_scale),
            # Penalize cumulative drift of the actual telescoping correction.
            "shadow_correction_vec": corrector_vec,
            "shadow_eta_vec": eta_vec,
        }
        return z_next_flat, psi.reshape(B, self.hidden_dim), aux


class OfficialShadowCorrectorHardMambaARModel(OfficialShadowCorrectorSoftMambaARModel):
    is_official_shadow_corrector_soft_mamba = False
    is_official_shadow_corrector_hard_mamba = True

    def __init__(
        self,
        *args,
        shadow_corrector_weight: float = 0.0,
        shadow_cum_weight: float = 0.0,
        shadow_psi_weight: float = 0.0,
        shadow_eta_weight: float = 0.0,
        shadow_eta_scale: float = 0.0,
        shadow_cum_horizons="4,8,16",
        **kwargs,
    ):
        super().__init__(
            *args,
            shadow_corrector_weight=shadow_corrector_weight,
            shadow_cum_weight=shadow_cum_weight,
            shadow_psi_weight=shadow_psi_weight,
            shadow_cum_horizons=shadow_cum_horizons,
            **kwargs,
        )
        self.shadow_eta_weight = float(shadow_eta_weight)
        self.shadow_eta_scale = float(shadow_eta_scale)
        self.shadow_cum_horizons = _parse_horizons(shadow_cum_horizons)
        self.shadow = ChannelWiseHardCorrectorTransport.from_existing(
            self.shadow,
            corrector_hidden=max(128, self.hidden_dim),
            eta_scale=float(shadow_eta_scale),
        )
