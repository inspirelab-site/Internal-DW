"""Local-atlas Mamba autoregressive model.

This model is the minimal implementation of the Riemannian/atlas idea from the
research discussion:

  * do not predict the next fMRI vector only in one global coordinate system;
  * learn local chart encoders phi_i and inverse charts psi_i;
  * learn chart transition maps tau_{ij} for overlap consistency;
  * learn stimulus-conditioned dynamical transition maps T_{ij};
  * train local maps with reconstruction, local invertibility, overlap/cocycle,
    and dynamic chart-transition losses.

It keeps the same dependency-free official-state Mamba recurrent backbone API as
``official_mamba_state`` so the existing chunked-BPTT trainer/evaluator can be
reused.  The recurrent token supplies the moving context c_t; atlas charts are
conditioned by c_t and a learned chart embedding.
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


class OfficialAtlasMambaARModel(nn.Module, _StimPoolMixin):
    """Mamba backbone with learned local charts and chart transition maps.

    The model maintains C local charts.  For each chart i,

        z_i = phi_i(x_t, c_t)
        rec_i = psi_i(z_i, c_t)

    For every pair i -> j,

        z_next_{i,j} = T_{ij}(z_i, u_{t+1}, c_t)
        z_same_j     = tau_{ij}(z_i, c_t)

    The output uses soft chart weights pi_t and pi_{t+1} predicted from the
    recurrent Mamba token.  Auxiliary losses are returned in ``aux`` and are
    added by the recurrent BPTT loss function when their corresponding weights
    are non-zero.
    """

    is_standard_autoregressive = True
    is_recurrent_state_ar = True
    is_official_state_mamba = True
    is_atlas_mamba = True

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
        atlas_perturb_min_ratio: float = 0.20,
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

        # Shared chart-conditioned phi_i and psi_i.  Chart identity enters only
        # through the learned chart embedding, which keeps the parameter count
        # stable as C grows.
        self.encoder = mlp(self.state_dim + self.hidden_dim + E, R)
        self.decoder = mlp(R + self.hidden_dim + E, self.state_dim)
        # tau_{ij}: same-state coordinate conversion between overlapping charts.
        self.overlap_transition = mlp(R + self.hidden_dim + 2 * E, R)
        # T_{ij}: stimulus-conditioned dynamical transition from chart i at t to
        # chart j at t+1.
        dyn_in = R + self.hidden_dim + (self.input_dim if self.has_external_input else 0) + 2 * E
        self.dynamic_transition = mlp(dyn_in, R)

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
        return emb.unsqueeze(0).expand(B, -1, -1)  # [B,C,E]

    def _encode_all(self, x_flat: torch.Tensor, ctx: torch.Tensor, chart_emb: torch.Tensor) -> torch.Tensor:
        B, C, E = chart_emb.shape
        x_rep = x_flat.unsqueeze(1).expand(B, C, self.state_dim)
        ctx_rep = ctx.unsqueeze(1).expand(B, C, self.hidden_dim)
        inp = torch.cat([x_rep, ctx_rep, chart_emb], dim=-1)
        return self.encoder(inp.reshape(B * C, -1)).reshape(B, C, self.atlas_latent_dim)

    def _decode_all(self, z: torch.Tensor, ctx: torch.Tensor, chart_emb: torch.Tensor) -> torch.Tensor:
        B, C, R = z.shape
        ctx_rep = ctx.unsqueeze(1).expand(B, C, self.hidden_dim)
        inp = torch.cat([z, ctx_rep, chart_emb], dim=-1)
        return self.decoder(inp.reshape(B * C, -1)).reshape(B, C, self.state_dim)

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

    @staticmethod
    def _weighted_mse_by_chart(value: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # value,target: [B,C,D], weights: [B,C]
        diff = (value - target).pow(2).mean(dim=-1)
        return (diff * weights).sum() / weights.sum().clamp_min(eps)

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

        z = self._encode_all(x_flat, ctx, chart_emb)  # [B,C,R]
        rec_by_chart = self._decode_all(z, ctx, chart_emb)
        rec_flat = (rec_by_chart * pi.unsqueeze(-1)).sum(dim=1)

        # Dynamic transition T_{ij}.  First produce pairwise transitions, then
        # marginalize over the source chart i to obtain one next coordinate in
        # each target chart j.
        z_pair_next = self._dynamic_all(z, ctx, stim, chart_emb)  # [B,C_i,C_j,R]
        z_next_by_chart = (z_pair_next * pi[:, :, None, None]).sum(dim=1)  # [B,C_j,R]
        pred_by_chart = self._decode_all(z_next_by_chart, ctx, chart_emb)
        pred_flat = (pred_by_chart * pi_next.unsqueeze(-1)).sum(dim=1)
        pred = self._reshape_pred(pred_flat, x_t)

        if return_aux:
            tau_pair = self._overlap_all(z, ctx, chart_emb)  # [B,C_i,C_j,R]
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
                "atlas_z_next_by_chart": z_next_by_chart,
                "atlas_pred_by_chart": pred_by_chart,
                "atlas_rec_by_chart": rec_by_chart,
                "atlas_rec_mix": self._reshape_pred(rec_flat, x_t),
                "atlas_tau_pair": tau_pair,
                "atlas_pair_dyn": z_pair_next,
                "atlas_chart_emb": chart_emb,
                "atlas_pi_entropy": (-(pi * (pi.clamp_min(1e-8)).log()).sum(dim=-1).mean()),
                "atlas_pi_next_entropy": (-(pi_next * (pi_next.clamp_min(1e-8)).log()).sum(dim=-1).mean()),
                "atlas_pi_balance": (pi.mean(dim=0) - (1.0 / self.atlas_num_charts)).pow(2).mean(),
                "atlas_pi_next_balance": (pi_next.mean(dim=0) - (1.0 / self.atlas_num_charts)).pow(2).mean(),
            }
            return pred, h_next, aux
        return pred, h_next

    # Auxiliary losses are methods so the training loop does not need to know
    # the internal tensor shapes beyond passing target/current frames.
    def atlas_auxiliary_losses(self, aux: dict, x_in: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        x_flat = self._flatten_state(x_in)
        target_flat = self._flatten_state(target)
        ctx = aux["atlas_ctx"]
        pi = aux["atlas_pi"]
        pi_next = aux["atlas_pi_next"]
        z = aux["atlas_z"]
        z_next = aux["atlas_z_next_by_chart"]
        chart_emb = aux["atlas_chart_emb"]
        B, C, R = z.shape

        target_z = self._encode_all(target_flat, ctx, chart_emb).detach()
        rec_by_chart = aux["atlas_rec_by_chart"]
        tau_pair = aux["atlas_tau_pair"]

        losses: dict[str, torch.Tensor] = {}
        losses["rec"] = self._weighted_mse_by_chart(rec_by_chart, x_flat[:, None, :].expand_as(rec_by_chart), pi)
        losses["dyn"] = self._weighted_mse_by_chart(z_next, target_z, pi_next)

        # Decoded per-chart target loss.  This complements main pred loss by
        # making each likely target chart locally predictive, not only the soft
        # mixture output.
        pred_by_chart = aux["atlas_pred_by_chart"]
        losses["pred_chart"] = self._weighted_mse_by_chart(pred_by_chart, target_flat[:, None, :].expand_as(pred_by_chart), pi_next)

        # Same-state overlap consistency: tau_{ij}(phi_i(x)) = phi_j(x).
        z_j = z.unsqueeze(1).expand(B, C, C, R)
        weights_ij = pi[:, :, None] * pi[:, None, :]
        overlap_err = (tau_pair - z_j).pow(2).mean(dim=-1)
        losses["overlap"] = (overlap_err * weights_ij).sum() / weights_ij.sum().clamp_min(1e-8)

        # Cocycle consistency: tau_{ik} = tau_{jk} o tau_{ij}.  This is the
        # atlas gluing relation.  We compute it for all triples when C is small.
        if C >= 3:
            emb = chart_emb
            # tau_ij is [B,i,j,R].  Apply tau_{jk} to tau_ij for all k.
            z_ij = tau_pair  # [B,i,j,R]
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
            losses["cocycle"] = target_flat.new_tensor(0.0)

        # Local chart invertibility: phi_i(psi_i(z_i + xi)) = z_i + xi.  This
        # is the main anti-AE-collapse constraint that makes phi/psi behave like
        # local coordinates rather than only a reconstruction bottleneck.
        if self.atlas_perturb_std > 0.0:
            xi = torch.randn_like(z) * self.atlas_perturb_std
            z_pert = z + xi
            x_pert = self._decode_all(z_pert, ctx, chart_emb)
            # Re-encode each decoded chart sample in the same chart.  Flatten B*C.
            x_pert_flat = x_pert.reshape(B * C, self.state_dim)
            ctx_bc = ctx[:, None, :].expand(B, C, self.hidden_dim).reshape(B * C, self.hidden_dim)
            emb_bc = chart_emb.reshape(B * C, self.atlas_chart_emb_dim)
            z_back = self.encoder(torch.cat([x_pert_flat, ctx_bc, emb_bc], dim=-1)).reshape(B, C, R)
            inv_err = (z_back - z_pert).pow(2).mean(dim=-1)
            losses["chart_inv"] = (inv_err * pi).sum() / pi.sum().clamp_min(1e-8)
            # Anti-collapse: the inverse chart must have non-trivial decoded
            # sensitivity.  The previous diagnostic-only floor (1e-3) allowed a
            # self-consistent but useless atlas whose decoder barely moved in
            # observation space.  This hinge gives the user-visible
            # --atlas_perturb_min_ratio a direct effect.
            dx = (x_pert - rec_by_chart).reshape(B, C, -1).norm(dim=-1)
            dz = xi.reshape(B, C, -1).norm(dim=-1).clamp_min(1e-8)
            ratio = dx / dz
            min_ratio = ratio.new_tensor(self.atlas_perturb_min_ratio)
            noncollapse = F.relu(min_ratio - ratio).pow(2)
            losses["noncollapse"] = (noncollapse * pi).sum() / pi.sum().clamp_min(1e-8)
            losses["perturb_ratio"] = ratio.mean().detach()
        else:
            losses["chart_inv"] = target_flat.new_tensor(0.0)
            losses["noncollapse"] = target_flat.new_tensor(0.0)
            losses["perturb_ratio"] = target_flat.new_tensor(0.0)

        losses["pi_balance"] = aux["atlas_pi_balance"] + aux["atlas_pi_next_balance"]
        losses["pi_entropy"] = aux["atlas_pi_entropy"] + aux["atlas_pi_next_entropy"]
        losses["target_z_norm"] = target_z.norm(dim=-1).mean().detach()
        losses["z_next_norm"] = z_next.norm(dim=-1).mean().detach()
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
