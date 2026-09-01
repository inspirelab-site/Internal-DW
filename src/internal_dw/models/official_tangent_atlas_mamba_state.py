"""Tangent-atlas Mamba autoregressive model.

This is the non-anchor Riemannian version of the atlas model.  It does not use a
separate ridge/direct predictor and it does not decode a full next fMRI state
from a latent code.  Instead, the current state x_t is treated as the base point
of a local chart and the atlas predicts a local tangent displacement:

    z_t^i        = phi_i(x_t, c_t)
    v_t^{i->j}   = T_ij(z_t^i, u_{t+1}, c_t)
    dx_t^{i->j}  = B_j(c_t) v_t^{i->j}
    xhat_{t+1}   = x_t + sum_{i,j} pi_t^i pi_{t+1}^j dx_t^{i->j}

So the Riemannian base point is x_t itself.  This is not an external prediction
anchor: the learned part is the tangent field dx_t, and rollout composes local
retractions x <- x + dx.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .official_state_mamba import (
    MambaLayerState,
    MambaStackState,
    StatefulMambaBlock,
    detach_mamba_stack_state,
)
from .simple_ar import _StimPoolMixin


class OfficialTangentAtlasMambaARModel(nn.Module, _StimPoolMixin):
    """Mamba backbone with local tangent charts and learned retractions."""

    is_standard_autoregressive = True
    is_recurrent_state_ar = True
    is_official_state_mamba = True
    is_atlas_mamba = True
    is_tangent_atlas_mamba = True

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
        atlas_num_charts: int = 4,
        atlas_latent_dim: int = 64,
        atlas_chart_emb_dim: int = 64,
        atlas_hidden_dim: int = 512,
        atlas_temperature: float = 1.0,
        atlas_perturb_std: float = 0.02,
        atlas_perturb_min_ratio: float = 0.15,
        tangent_scale_init: float = 0.05,
        atlas_hard_delta_norm: bool = False,
        atlas_hard_delta_min_x_ratio: float = 0.05,
        atlas_delta_scale_init_ratio: float = 0.01,
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

        self.atlas_num_charts = int(max(1, atlas_num_charts))
        self.atlas_latent_dim = int(atlas_latent_dim)
        self.atlas_chart_emb_dim = int(atlas_chart_emb_dim)
        self.atlas_temperature = float(max(atlas_temperature, 1e-4))
        self.atlas_perturb_std = float(max(atlas_perturb_std, 0.0))
        self.atlas_perturb_min_ratio = float(max(atlas_perturb_min_ratio, 0.0))
        self.atlas_hard_delta_norm = bool(atlas_hard_delta_norm)
        self.atlas_hard_delta_min_x_ratio = float(max(atlas_hard_delta_min_x_ratio, 0.0))
        self.atlas_delta_scale_init_ratio = float(max(atlas_delta_scale_init_ratio, 1e-8))

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

        C = self.atlas_num_charts
        R = self.atlas_latent_dim
        E = self.atlas_chart_emb_dim
        H = int(atlas_hidden_dim)
        self.chart_embed = nn.Embedding(C, E)
        self.chart_selector = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, H),
            nn.GELU(),
            nn.Linear(H, C),
        )
        self.next_chart_selector = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim),
        ) if False else nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, H),
            nn.GELU(),
            nn.Linear(H, C),
        )

        def mlp(in_features: int, out_features: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Linear(in_features, H),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(H, H),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(H, out_features),
            )

        self.encoder = mlp(self.state_dim + self.hidden_dim + E, R)
        self.overlap_transition = mlp(R + self.hidden_dim + 2 * E, R)
        dyn_in = R + self.hidden_dim + (self.input_dim if self.has_external_input else 0) + 2 * E
        self.dynamic_transition = mlp(dyn_in, R)
        # Local tangent decoder / first-order retraction basis B_j(c_t).
        self.tangent_decoder = mlp(R + self.hidden_dim + E, self.state_dim)
        # Optional learned scalar for the raw tangent decoder.  In hard-norm mode
        # this only affects chart auxiliary diagnostics; the final displacement
        # is re-normalized below.
        self.log_tangent_scale = nn.Parameter(torch.tensor(float(tangent_scale_init)).log())

        # Context-conditioned positive scale for structural anti-identity mode.
        # The model predicts a scale ratio relative to ||x_t||, then applies it
        # to a unit tangent direction.  This prevents the exact zero-delta
        # identity solution without using an external predictor.
        self.tangent_scale_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, H),
            nn.GELU(),
            nn.Linear(H, 1),
        )
        # Initialize near atlas_delta_scale_init_ratio above the hard floor.
        with torch.no_grad():
            last = self.tangent_scale_head[-1]
            if hasattr(last, "weight"):
                last.weight.zero_()
            if hasattr(last, "bias"):
                # softplus(-6) ~= 0.0025, so initial ratio is close to the floor.
                last.bias.fill_(-6.0)

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

    def make_token(self, x_t: torch.Tensor, stim_t: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        B = x_t.shape[0]
        x_flat = self._flatten_state(x_t)
        stim = self._pool_stim_point(stim_t, B, x_t.device, x_t.dtype)
        inp = torch.cat([x_flat, stim], dim=-1) if self.has_external_input else x_flat
        return self.in_proj(inp), stim

    def _chart_embeds(self, B: int, device, dtype) -> torch.Tensor:
        ids = torch.arange(self.atlas_num_charts, device=device)
        emb = self.chart_embed(ids).to(dtype=dtype)
        return emb.unsqueeze(0).expand(B, -1, -1)

    def _encode_all(self, x_flat: torch.Tensor, ctx: torch.Tensor, chart_emb: torch.Tensor) -> torch.Tensor:
        B, C, E = chart_emb.shape
        x_rep = x_flat.unsqueeze(1).expand(B, C, self.state_dim)
        ctx_rep = ctx.unsqueeze(1).expand(B, C, self.hidden_dim)
        inp = torch.cat([x_rep, ctx_rep, chart_emb], dim=-1)
        return self.encoder(inp.reshape(B * C, -1)).reshape(B, C, self.atlas_latent_dim)

    def _overlap_all(self, z_i: torch.Tensor, ctx: torch.Tensor, chart_emb: torch.Tensor) -> torch.Tensor:
        B, C, R = z_i.shape
        emb_i = chart_emb.unsqueeze(2).expand(B, C, C, self.atlas_chart_emb_dim)
        emb_j = chart_emb.unsqueeze(1).expand(B, C, C, self.atlas_chart_emb_dim)
        z_rep = z_i.unsqueeze(2).expand(B, C, C, R)
        ctx_rep = ctx[:, None, None, :].expand(B, C, C, self.hidden_dim)
        inp = torch.cat([z_rep, ctx_rep, emb_i, emb_j], dim=-1)
        return self.overlap_transition(inp.reshape(B * C * C, -1)).reshape(B, C, C, R)

    def _dynamic_all(self, z_i: torch.Tensor, ctx: torch.Tensor, stim: torch.Tensor, chart_emb: torch.Tensor) -> torch.Tensor:
        B, C, R = z_i.shape
        emb_i = chart_emb.unsqueeze(2).expand(B, C, C, self.atlas_chart_emb_dim)
        emb_j = chart_emb.unsqueeze(1).expand(B, C, C, self.atlas_chart_emb_dim)
        z_rep = z_i.unsqueeze(2).expand(B, C, C, R)
        ctx_rep = ctx[:, None, None, :].expand(B, C, C, self.hidden_dim)
        pieces = [z_rep, ctx_rep]
        if self.has_external_input:
            pieces.append(stim[:, None, None, :].expand(B, C, C, self.input_dim))
        pieces.extend([emb_i, emb_j])
        inp = torch.cat(pieces, dim=-1)
        return self.dynamic_transition(inp.reshape(B * C * C, -1)).reshape(B, C, C, R)

    def _decode_tangent_by_chart(self, v: torch.Tensor, ctx: torch.Tensor, chart_emb: torch.Tensor) -> torch.Tensor:
        B, C, R = v.shape
        ctx_rep = ctx.unsqueeze(1).expand(B, C, self.hidden_dim)
        inp = torch.cat([v, ctx_rep, chart_emb], dim=-1)
        delta = self.tangent_decoder(inp.reshape(B * C, -1)).reshape(B, C, self.state_dim)
        return delta * self.log_tangent_scale.exp().clamp(1e-4, 10.0)

    def _decode_tangent_pair(self, v_pair: torch.Tensor, ctx: torch.Tensor, chart_emb: torch.Tensor) -> torch.Tensor:
        B, C, _, R = v_pair.shape
        emb_j = chart_emb.unsqueeze(1).expand(B, C, C, self.atlas_chart_emb_dim)
        ctx_rep = ctx[:, None, None, :].expand(B, C, C, self.hidden_dim)
        inp = torch.cat([v_pair, ctx_rep, emb_j], dim=-1)
        delta = self.tangent_decoder(inp.reshape(B * C * C, -1)).reshape(B, C, C, self.state_dim)
        return delta * self.log_tangent_scale.exp().clamp(1e-4, 10.0)

    @staticmethod
    def _weighted_mse_by_chart(value: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        diff = (value - target).pow(2).mean(dim=-1)
        return (diff * weights).sum() / weights.sum().clamp_min(eps)

    def _apply_hard_delta_norm(self, delta_flat: torch.Tensor, x_flat: torch.Tensor, ctx: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Structurally prevent the identity shortcut in tangent-atlas mode.

        Without this, the model can set delta_hat ~= 0 and obtain a strong
        one-step score because x_{t+1} is very close to x_t.  In hard-norm
        mode we separate direction and scale:

            delta_hat = gamma_t * delta_raw / ||delta_raw||,
            gamma_t   = (rho_x + softplus(g(ctx))) * ||x_t||.

        This is not an external anchor: x_t is only the base-point norm used to
        set a nonzero local tangent scale.
        """
        if not self.atlas_hard_delta_norm:
            aux = {
                "hard_delta_scale_ratio": delta_flat.new_zeros(()),
                "hard_delta_scale": delta_flat.new_zeros(()),
                "hard_delta_raw_norm": delta_flat.reshape(delta_flat.shape[0], -1).norm(dim=-1).mean().detach(),
            }
            return delta_flat, aux

        eps = 1e-8
        raw = delta_flat.reshape(delta_flat.shape[0], -1)
        raw_norm = raw.norm(dim=-1, keepdim=True).clamp_min(eps)
        direction = raw / raw_norm

        # Use the current base-point norm as a target-free local scale proxy.
        x_norm = x_flat.reshape(x_flat.shape[0], -1).norm(dim=-1, keepdim=True).clamp_min(eps)
        scale_extra = F.softplus(self.tangent_scale_head(ctx))
        scale_ratio = self.atlas_hard_delta_min_x_ratio + scale_extra
        gamma = scale_ratio * x_norm
        scaled = direction * gamma
        aux = {
            "hard_delta_scale_ratio": scale_ratio.mean().detach(),
            "hard_delta_scale": gamma.mean().detach(),
            "hard_delta_raw_norm": raw_norm.mean().detach(),
        }
        return scaled.reshape_as(delta_flat), aux

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
        token, stim = self.make_token(x_t, stim_t)

        next_states: list[MambaLayerState] = []
        dt_means, dt_mins, dt_maxs = [], [], []
        ssm_norms, conv_norms = [], []
        ctx = token
        for block, layer_state in zip(self.blocks, h):
            ctx, next_state, aux_l = block.step(ctx, layer_state)
            next_states.append(next_state)
            dt_means.append(aux_l["dt_mean"])
            dt_mins.append(aux_l["dt_min"])
            dt_maxs.append(aux_l["dt_max"])
            ssm_norms.append(aux_l["ssm_state_norm"])
            conv_norms.append(aux_l["conv_state_norm"])
        h_next = tuple(next_states)

        x_flat = self._flatten_state(x_t)
        chart_emb = self._chart_embeds(B, x_t.device, x_t.dtype)
        logits = self.chart_selector(ctx)
        next_logits = self.next_chart_selector(ctx)
        pi = F.softmax(logits / self.atlas_temperature, dim=-1)
        pi_next = F.softmax(next_logits / self.atlas_temperature, dim=-1)

        z = self._encode_all(x_flat, ctx, chart_emb)
        v_pair = self._dynamic_all(z, ctx, stim, chart_emb)                 # [B,Ci,Cj,R]
        delta_pair = self._decode_tangent_pair(v_pair, ctx, chart_emb)       # [B,Ci,Cj,D]
        weights_ij = pi[:, :, None] * pi_next[:, None, :]
        delta_raw_flat = (delta_pair * weights_ij[..., None]).sum(dim=(1, 2))
        delta_flat, hard_delta_aux = self._apply_hard_delta_norm(delta_raw_flat, x_flat, ctx)
        pred_flat = x_flat + delta_flat
        pred = self._reshape_pred(pred_flat, x_t)

        if return_aux:
            tau_pair = self._overlap_all(z, ctx, chart_emb)
            # Source-marginal target-chart tangent coordinates/deltas for cheap
            # per-chart diagnostics and auxiliary losses.
            v_by_chart = (v_pair * pi[:, :, None, None]).sum(dim=1)          # [B,Cj,R]
            delta_by_chart = self._decode_tangent_by_chart(v_by_chart, ctx, chart_emb)
            pred_by_chart = x_flat[:, None, :] + delta_by_chart
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
                "atlas_ctx": ctx,
                "atlas_pi": pi,
                "atlas_pi_next": pi_next,
                "atlas_z": z,
                # Keep this key for the existing trainer hooks.  In tangent
                # atlas it means target-chart tangent coordinate, not next-state
                # latent coordinate.
                "atlas_z_next_by_chart": v_by_chart,
                "atlas_v_pair": v_pair,
                "atlas_delta_by_chart": delta_by_chart,
                "atlas_delta_pair": delta_pair,
                "atlas_delta_mix": delta_flat.detach(),
                "atlas_delta_raw_mix": delta_raw_flat.detach(),
                "atlas_hard_delta_scale_ratio": hard_delta_aux["hard_delta_scale_ratio"],
                "atlas_hard_delta_scale": hard_delta_aux["hard_delta_scale"],
                "atlas_hard_delta_raw_norm": hard_delta_aux["hard_delta_raw_norm"],
                "atlas_pred_by_chart": pred_by_chart,
                # No global state reconstruction in tangent atlas.  This key is
                # only for compatibility when old rec losses are zero.
                "atlas_rec_by_chart": x_flat[:, None, :].expand(B, self.atlas_num_charts, self.state_dim),
                "atlas_rec_mix": x_t,
                "atlas_tau_pair": tau_pair,
                "atlas_pair_dyn": v_pair,
                "atlas_chart_emb": chart_emb,
                "atlas_tangent_scale": self.log_tangent_scale.exp().detach(),
                "atlas_pi_entropy": (-(pi * (pi.clamp_min(1e-8)).log()).sum(dim=-1).mean()),
                "atlas_pi_next_entropy": (-(pi_next * (pi_next.clamp_min(1e-8)).log()).sum(dim=-1).mean()),
                "atlas_pi_balance": (pi.mean(dim=0) - (1.0 / self.atlas_num_charts)).pow(2).mean(),
                "atlas_pi_next_balance": (pi_next.mean(dim=0) - (1.0 / self.atlas_num_charts)).pow(2).mean(),
            }
            return pred, h_next, aux
        return pred, h_next

    def atlas_auxiliary_losses(self, aux: dict, x_in: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        x_flat = self._flatten_state(x_in)
        target_flat = self._flatten_state(target)
        target_delta = target_flat - x_flat
        ctx = aux["atlas_ctx"]
        pi = aux["atlas_pi"]
        pi_next = aux["atlas_pi_next"]
        z = aux["atlas_z"]
        v_by_chart = aux["atlas_z_next_by_chart"]
        delta_by_chart = aux["atlas_delta_by_chart"]
        chart_emb = aux["atlas_chart_emb"]
        tau_pair = aux["atlas_tau_pair"]
        B, C, R = z.shape

        losses: dict[str, torch.Tensor] = {}
        zero = target_flat.new_tensor(0.0)
        # No full-state autoencoder objective in tangent atlas.
        losses["rec"] = zero
        losses["chart_inv"] = zero
        losses["dyn"] = self._weighted_mse_by_chart(
            delta_by_chart,
            target_delta[:, None, :].expand_as(delta_by_chart),
            pi_next,
        )
        losses["pred_chart"] = losses["dyn"]

        # Same-state overlap consistency in chart coordinates.
        z_j = z.unsqueeze(1).expand(B, C, C, R)
        weights_ij = pi[:, :, None] * pi[:, None, :]
        overlap_err = (tau_pair - z_j).pow(2).mean(dim=-1)
        losses["overlap"] = (overlap_err * weights_ij).sum() / weights_ij.sum().clamp_min(1e-8)

        if C >= 3:
            emb = chart_emb
            z_ij = tau_pair
            emb_j = emb[:, None, :, None, :].expand(B, C, C, C, self.atlas_chart_emb_dim)
            emb_k = emb[:, None, None, :, :].expand(B, C, C, C, self.atlas_chart_emb_dim)
            ctx_rep = ctx[:, None, None, None, :].expand(B, C, C, C, self.hidden_dim)
            z_ij_rep = z_ij[:, :, :, None, :].expand(B, C, C, C, R)
            inp = torch.cat([z_ij_rep, ctx_rep, emb_j, emb_k], dim=-1)
            tau_jk_of_ij = self.overlap_transition(inp.reshape(B * C * C * C, -1)).reshape(B, C, C, C, R)
            tau_ik = tau_pair[:, :, None, :, :].expand(B, C, C, C, R)
            weights_ijk = pi[:, :, None, None] * pi[:, None, :, None] * pi[:, None, None, :]
            cyc_err = (tau_jk_of_ij - tau_ik).pow(2).mean(dim=-1)
            losses["cocycle"] = (cyc_err * weights_ijk).sum() / weights_ijk.sum().clamp_min(1e-8)
        else:
            losses["cocycle"] = zero

        # Non-collapse now measures sensitivity of the local retraction B_j to
        # perturbations in tangent coordinates v, rather than full-state AE
        # sensitivity.
        if self.atlas_perturb_std > 0.0:
            xi = torch.randn_like(v_by_chart) * self.atlas_perturb_std
            delta_pert = self._decode_tangent_by_chart(v_by_chart + xi, ctx, chart_emb)
            dx = (delta_pert - delta_by_chart).reshape(B, C, -1).norm(dim=-1)
            dz = xi.reshape(B, C, -1).norm(dim=-1).clamp_min(1e-8)
            ratio = dx / dz
            min_ratio = ratio.new_tensor(self.atlas_perturb_min_ratio)
            noncollapse = F.relu(min_ratio - ratio).pow(2)
            losses["noncollapse"] = (noncollapse * pi_next).sum() / pi_next.sum().clamp_min(1e-8)
            losses["perturb_ratio"] = ratio.mean().detach()
        else:
            losses["noncollapse"] = zero
            losses["perturb_ratio"] = zero

        # Diagnostics only.
        with torch.no_grad():
            target_z = self._encode_all(target_flat, ctx, chart_emb)
        losses["pi_balance"] = aux["atlas_pi_balance"] + aux["atlas_pi_next_balance"]
        losses["pi_entropy"] = aux["atlas_pi_entropy"] + aux["atlas_pi_next_entropy"]
        losses["target_z_norm"] = target_z.norm(dim=-1).mean().detach()
        losses["z_next_norm"] = v_by_chart.norm(dim=-1).mean().detach()
        return losses

    def burn_in(self, x_seq: torch.Tensor, stim_seq: Optional[torch.Tensor] = None, h0: Optional[MambaStackState] = None, detach: bool = False):
        B, T = x_seq.shape[:2]
        h = h0 if h0 is not None else self.init_state(B, x_seq.device, x_seq.dtype)
        ctx_mgr = torch.no_grad() if detach else torch.enable_grad()
        with ctx_mgr:
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
