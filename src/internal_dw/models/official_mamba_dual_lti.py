"""Dual-state model with true stateful Mamba A_t branch + channel-wise LTI A branch.

This is the strict version of the proposed model:

    s_t = (s_t^{Mamba}, z_t^{LTI})

where s_t^{Mamba} is the actual persistent Mamba state
(conv_state, ssm_state) for each layer, not a rolling token buffer.  The LTI
branch has a fixed channel-wise transition A, enabling large-K Gram/KG loss.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .official_state_mamba import OfficialStateMambaVectorARModel
from .simple_ar import _StimPoolMixin


class ChannelWiseLinearTransition(nn.Module):
    """Channel-wise full matrix transition z_c <- A_c z_c.

    z: [B, C, D], A: [C, D, D].  This keeps the LTI A computationally small and
    makes long powers A^k cheap channel-by-channel.
    """

    def __init__(self, channels: int, channel_dim: int, init_scale: float = 0.98, init_noise: float = 1e-3):
        super().__init__()
        self.channels = int(channels)
        self.channel_dim = int(channel_dim)
        eye = torch.eye(self.channel_dim).unsqueeze(0).repeat(self.channels, 1, 1)
        A = float(init_scale) * eye
        if init_noise > 0:
            A = A + float(init_noise) * torch.randn_like(A)
        self.A = nn.Parameter(A)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bci,cij->bcj", z, self.A)

    def spectral_stats(self) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            svals = torch.linalg.svdvals(self.A.float())
            sigma = svals[..., 0]
            return {"sigma_max": sigma.max(), "sigma_mean": sigma.mean(), "sigma_min": sigma.min()}

    def spectral_penalty(self, max_sigma: float = 1.0) -> torch.Tensor:
        if max_sigma <= 0:
            return self.A.new_tensor(0.0)
        svals = torch.linalg.svdvals(self.A.float())
        sigma = svals[..., 0].to(self.A.dtype)
        return F.relu(sigma - float(max_sigma)).pow(2).mean()


class OfficialMambaDualLTIKGVectorARModel(nn.Module, _StimPoolMixin):
    """True Mamba-state + channel-wise LTI/KG dual-state AR model.

    Short Kshort training uses BPTT through both branches.  Long Klong KG loss is
    applied only to the fixed-A LTI branch via ||A^q e_1||, avoiding long Mamba
    BPTT while still imposing large-horizon constraints.
    """

    is_standard_autoregressive = True
    is_recurrent_state_ar = True
    is_mamba_dual_lti_kg = True
    is_official_mamba_dual_lti_kg = True
    task_type = "vector"

    def __init__(
        self,
        state_dim: int,
        input_dim: int = 0,
        hidden_dim: int = 512,
        depth: int = 4,
        dropout: float = 0.0,
        has_external_input: bool = True,
        residual: bool = True,
        # True Mamba branch hyperparameters.
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_dt_rank: int | str = "auto",
        mamba_dt_min: float = 1e-3,
        mamba_dt_max: float = 1e-1,
        mamba_norm_type: str = "layer",
        # LTI/KG branch hyperparameters.
        lti_channels: int = 16,
        lti_channel_dim: int = 32,
        lti_input_hidden: int = 512,
        lti_decoder_hidden: int = 512,
        lti_a_init_scale: float = 0.98,
        lti_a_init_noise: float = 1e-3,
        # Fusion.
        fusion_mode: str = "fixed",
        fusion_alpha_short: float = 0.75,
        fusion_switch_k: int = 16,
        fusion_tau: float = 4.0,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.has_external_input = bool(has_external_input)
        self.residual = bool(residual)

        self.mamba = OfficialStateMambaVectorARModel(
            state_dim=state_dim,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            has_external_input=has_external_input,
            residual=residual,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            mamba_dt_rank=mamba_dt_rank,
            mamba_dt_min=mamba_dt_min,
            mamba_dt_max=mamba_dt_max,
            mamba_norm_type=mamba_norm_type,
        )

        self.lti_channels = int(lti_channels)
        self.lti_channel_dim = int(lti_channel_dim)
        self.lti_dim = self.lti_channels * self.lti_channel_dim

        in_dim = self.state_dim + (self.input_dim if self.has_external_input else 0)
        self.lti_input = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(lti_input_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(lti_input_hidden), self.lti_dim),
        )
        self.lti_transition = ChannelWiseLinearTransition(
            self.lti_channels,
            self.lti_channel_dim,
            init_scale=lti_a_init_scale,
            init_noise=lti_a_init_noise,
        )
        self.lti_decoder = nn.Sequential(
            nn.LayerNorm(self.lti_dim),
            nn.Linear(self.lti_dim, int(lti_decoder_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(lti_decoder_hidden), self.state_dim),
        )
        self.lti_target_encoder = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Linear(self.state_dim, int(lti_input_hidden)),
            nn.GELU(),
            nn.Linear(int(lti_input_hidden), self.lti_dim),
        )

        self.fusion_mode = str(fusion_mode)
        self.fusion_alpha_short = float(fusion_alpha_short)
        self.fusion_switch_k = int(fusion_switch_k)
        self.fusion_tau = float(fusion_tau)
        if self.fusion_mode == "learned":
            self.fusion_logit = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter("fusion_logit", None)

        # Eval-only branch override.  None means normal fusion/schedule.
        # "mamba" and "lti" force the returned prediction to come from
        # the corresponding branch, while both internal states are still updated.
        self._eval_branch_mode: Optional[str] = None

    def set_eval_branch_mode(self, mode: Optional[str] = None):
        if mode is None:
            self._eval_branch_mode = None
            return
        mode = str(mode).lower()
        aliases = {
            "mamba": "mamba",
            "mamba_only": "mamba",
            "at": "mamba",
            "lti": "lti",
            "lti_only": "lti",
            "a": "lti",
        }
        if mode not in aliases:
            raise ValueError(f"Unknown eval branch mode: {mode}. Use None, 'mamba', or 'lti'.")
        self._eval_branch_mode = aliases[mode]

    def get_eval_branch_mode(self) -> Optional[str]:
        return self._eval_branch_mode

    def init_state(self, batch_size: int, device=None, dtype=None):
        device = device if device is not None else next(self.parameters()).device
        dtype = dtype if dtype is not None else next(self.parameters()).dtype
        h_m = self.mamba.init_state(batch_size, device=device, dtype=dtype)
        z_lti = torch.zeros(batch_size, self.lti_channels, self.lti_channel_dim, device=device, dtype=dtype)
        return (h_m, z_lti)

    def detach_state(self, state):
        h_m, z_lti = state
        h_m = tuple((conv.detach(), ssm.detach()) for conv, ssm in h_m)
        return (h_m, z_lti.detach())

    def _pool_stim_point(self, stim_t: Optional[torch.Tensor], B: int, device, dtype) -> torch.Tensor:
        return self.mamba._pool_stim_point(stim_t, B, device, dtype)

    def _lti_input_term(self, x_t: torch.Tensor, stim_t: Optional[torch.Tensor]) -> torch.Tensor:
        B = x_t.shape[0]
        stim = self._pool_stim_point(stim_t, B, x_t.device, x_t.dtype)
        inp = torch.cat([x_t.reshape(B, -1), stim], dim=-1) if self.has_external_input else x_t.reshape(B, -1)
        return self.lti_input(inp).view(B, self.lti_channels, self.lti_channel_dim)

    def lti_decode(self, z_lti: torch.Tensor) -> torch.Tensor:
        B = z_lti.shape[0]
        return self.lti_decoder(z_lti.reshape(B, -1))

    def encode_target_lti(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return self.lti_target_encoder(x.reshape(B, -1)).view(B, self.lti_channels, self.lti_channel_dim)

    def fusion_alpha(self, horizon_index: Optional[int] = None, device=None, dtype=None) -> torch.Tensor | float:
        if self.fusion_mode == "learned" and self.fusion_logit is not None:
            return torch.sigmoid(self.fusion_logit)
        if self.fusion_mode == "schedule" and horizon_index is not None:
            k = torch.as_tensor(float(horizon_index), device=device, dtype=dtype)
            return torch.sigmoid((float(self.fusion_switch_k) - k) / max(float(self.fusion_tau), 1e-6))
        return float(self.fusion_alpha_short)

    def step(
        self,
        state,
        x_t: torch.Tensor,
        stim_t: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        horizon_index: Optional[int] = None,
    ):
        h_m, z_lti = state
        pred_m, h_m_next, aux_m = self.mamba.step(h_m, x_t, stim_t, return_aux=True)

        z_lti_next = self.lti_transition(z_lti) + self._lti_input_term(x_t, stim_t)
        pred_lti = self.lti_decode(z_lti_next)

        branch_mode = getattr(self, "_eval_branch_mode", None)
        if branch_mode == "mamba":
            alpha = pred_m.new_tensor(1.0)
            pred = pred_m
            branch_mode_id = pred_m.new_tensor(1.0)
        elif branch_mode == "lti":
            alpha = pred_m.new_tensor(0.0)
            pred = pred_lti
            branch_mode_id = pred_m.new_tensor(2.0)
        else:
            alpha = self.fusion_alpha(horizon_index, device=x_t.device, dtype=x_t.dtype)
            pred = alpha * pred_m + (1.0 - alpha) * pred_lti
            branch_mode_id = pred_m.new_tensor(0.0)
        next_state = (h_m_next, z_lti_next)

        if return_aux:
            aux: Dict[str, Any] = dict(aux_m)
            aux.update({
                "pred_mamba": pred_m,
                "pred_lti": pred_lti,
                "z_lti": z_lti_next,
                "fusion_alpha": alpha if torch.is_tensor(alpha) else pred.new_tensor(float(alpha)),
                "eval_branch_mode_id": branch_mode_id,
                "mamba_lti_pred_l1": (pred_m - pred_lti).abs().mean(),
                "lti_state_norm": z_lti_next.reshape(z_lti_next.shape[0], -1).norm(dim=1).mean(),
            })
            with torch.no_grad():
                stats = self.lti_transition.spectral_stats()
                aux["lti_A_sigma_max"] = stats["sigma_max"].to(pred.device, pred.dtype)
                aux["lti_A_sigma_mean"] = stats["sigma_mean"].to(pred.device, pred.dtype)
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

    def kg_gram_loss_from_z(
        self,
        z_pred: torch.Tensor,
        target_x: torch.Tensor,
        horizon: int,
        decay: float = 1.0,
        detach_target_encoder: bool = True,
        include_k0: bool = True,
        reduce: str = "mean",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Channel-wise KG/Gram loss: mean_q ||A^q (z_pred - E(x_target))||^2."""
        K = max(1, int(horizon))
        target_z = self.encode_target_lti(target_x)
        if detach_target_encoder:
            target_z = target_z.detach()
        e = z_pred - target_z
        losses = []
        rels = []
        start_q = 0 if include_k0 else 1
        cur = e
        denom = target_z.pow(2).mean().sqrt().clamp_min(1e-6)
        for q in range(K):
            if q >= start_q:
                step = cur.pow(2).mean()
                if decay != 1.0:
                    step = (float(decay) ** q) * step
                losses.append(step)
                rels.append(cur.pow(2).mean().sqrt() / denom)
            if q != K - 1:
                cur = self.lti_transition(cur)
        if losses:
            loss = torch.stack(losses).mean() if reduce == "mean" else torch.stack(losses).sum()
            rel = torch.stack(rels).mean()
        else:
            loss = e.new_tensor(0.0)
            rel = e.new_tensor(0.0)
        logs = {"kg_first_latent_mse": e.pow(2).mean().detach(), "kg_mean_rel": rel.detach()}
        return loss, logs



    def kg_supervised_loss_from_z(
        self,
        z_pred: torch.Tensor,
        target_x_seq: torch.Tensor,
        horizon: int,
        decay: float = 1.0,
        detach_target_encoder: bool = True,
        include_k0: bool = True,
        reduce: str = "mean",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Supervised folded KG loss against future GT latent targets.

        z_pred is the LTI latent state at the first predicted target time, e.g.
        z_{t+1}.  target_x_seq contains the corresponding future ground-truth
        states [x_{t+1}, x_{t+2}, ..., x_{t+K}] with shape [B, K, state_dim].

        The loss compares a cheap autonomous folded rollout of the fixed-A LTI
        state against future GT latents:

            mean_q || A^q z_pred - E_A(x_{t+1+q}) ||^2.

        This is different from kg_gram_loss_from_z, which only penalizes the
        propagated first latent error A^q(z_pred - E_A(x_{t+1})).  The supervised
        version directly teaches the fixed-A branch to be a non-trivial long
        predictor instead of only being a stable damping anchor.
        """
        if target_x_seq.dim() == 2:
            target_x_seq = target_x_seq.unsqueeze(1)
        K_avail = int(target_x_seq.shape[1])
        K = max(1, min(int(horizon), K_avail))
        target_x_seq = target_x_seq[:, :K]

        B, K_eff, D = target_x_seq.shape
        flat_targets = target_x_seq.reshape(B * K_eff, D)
        target_z = self.encode_target_lti(flat_targets).reshape(B, K_eff, self.lti_channels, self.lti_channel_dim)
        if detach_target_encoder:
            target_z = target_z.detach()

        cur = z_pred
        losses = []
        rels = []
        start_q = 0 if include_k0 else 1
        first_mse = (cur - target_z[:, 0]).pow(2).mean()
        last_rel = z_pred.new_tensor(0.0)

        for q in range(K_eff):
            if q >= start_q:
                diff = cur - target_z[:, q]
                step = diff.pow(2).mean()
                if decay != 1.0:
                    step = (float(decay) ** q) * step
                denom = target_z[:, q].pow(2).mean().sqrt().clamp_min(1e-6)
                rel = diff.pow(2).mean().sqrt() / denom
                losses.append(step)
                rels.append(rel)
                last_rel = rel
            if q != K_eff - 1:
                cur = self.lti_transition(cur)

        if losses:
            loss = torch.stack(losses).mean() if reduce == "mean" else torch.stack(losses).sum()
            mean_rel = torch.stack(rels).mean()
        else:
            loss = z_pred.new_tensor(0.0)
            mean_rel = z_pred.new_tensor(0.0)
        logs = {
            "kg_sup_first_latent_mse": first_mse.detach(),
            "kg_sup_mean_rel": mean_rel.detach(),
            "kg_sup_last_rel": last_rel.detach(),
            "kg_sup_horizon_eff": z_pred.new_tensor(float(K_eff)),
        }
        return loss, logs

    def lti_spectral_penalty(self, max_sigma: float = 1.0) -> torch.Tensor:
        return self.lti_transition.spectral_penalty(max_sigma=max_sigma)

    def predict_frame_from_history(self, history: torch.Tensor, stim_window: Optional[torch.Tensor] = None) -> torch.Tensor:
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

    def step_history(self, history: torch.Tensor, stim_window: Optional[torch.Tensor] = None) -> torch.Tensor:
        frame = self.predict_frame_from_history(history, stim_window)
        return torch.cat([history[:, 1:], frame.unsqueeze(1)], dim=1)

    def forward(self, stim_window: Optional[torch.Tensor], history: torch.Tensor, return_aux: bool = False):
        frame = self.predict_frame_from_history(history, stim_window)
        pred = frame.unsqueeze(1)
        if return_aux:
            return pred, {"pred_frame": frame}
        return pred
