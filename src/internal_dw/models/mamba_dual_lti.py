"""Mamba + channel-wise LTI/KG dual-state recurrent model.

This model implements the design:

    s_t = (s_t^{A_t}, s_t^{A})

where s_t^{A_t} is the ordinary Mamba-style input-dependent recurrent state,
and s_t^{A} is a time-invariant channel-wise linear state.  The short branch is
trained by ordinary short BPTT.  The LTI branch admits a cheap large-K Gram/KG
loss because its transition A is fixed over time and channel-wise.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_state import MambaStateVectorARModel
from .simple_ar import _StimPoolMixin


class ChannelWiseLinearTransition(nn.Module):
    """Channel-wise full matrix transition z_c <- A_c z_c.

    z has shape [B, C, D].  A has shape [C, D, D].  This is much cheaper than a
    dense [CD, CD] matrix but still more expressive than a diagonal transition.
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
        # z: [B,C,D], A: [C,D,D], output: [B,C,D]
        return torch.einsum("bci,cij->bcj", z, self.A)

    def spectral_stats(self) -> Dict[str, torch.Tensor]:
        # C is normally small enough that this is fine for logging/regularizing.
        with torch.no_grad():
            svals = torch.linalg.svdvals(self.A.float())
            sigma = svals[..., 0]
            return {
                "sigma_max": sigma.max(),
                "sigma_mean": sigma.mean(),
                "sigma_min": sigma.min(),
            }

    def spectral_penalty(self, max_sigma: float = 1.0) -> torch.Tensor:
        if max_sigma <= 0:
            return self.A.new_tensor(0.0)
        svals = torch.linalg.svdvals(self.A.float())
        sigma = svals[..., 0].to(self.A.dtype)
        return F.relu(sigma - float(max_sigma)).pow(2).mean()


class MambaDualLTIKGVectorARModel(nn.Module, _StimPoolMixin):
    """Dual-state AR model: Mamba A_t state + channel-wise LTI A state.

    Short prediction uses both branches.  The LTI branch additionally exposes
    a channel-wise Gram/KG loss on A^k times the first latent error.
    """

    is_standard_autoregressive = True
    is_recurrent_state_ar = True
    is_mamba_dual_lti_kg = True
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
        alpha_min: float = 0.0,
        alpha_max: float = 0.995,
        lti_channels: int = 16,
        lti_channel_dim: int = 32,
        lti_input_hidden: int = 512,
        lti_decoder_hidden: int = 512,
        lti_a_init_scale: float = 0.98,
        lti_a_init_noise: float = 1e-3,
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

        self.mamba = MambaStateVectorARModel(
            state_dim=state_dim,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            has_external_input=has_external_input,
            residual=residual,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )

        self.lti_channels = int(lti_channels)
        self.lti_channel_dim = int(lti_channel_dim)
        self.lti_dim = self.lti_channels * self.lti_channel_dim

        in_dim = self.state_dim + (self.input_dim if self.has_external_input else 0)
        # B phi(x_t,u_t) term.  It injects current observation/stimulus into the
        # time-invariant state; A itself remains fixed and channel-wise.
        self.lti_input = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(lti_input_hidden)),
            nn.GELU(),
            nn.Dropout(dropout),
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
            nn.Dropout(dropout),
            nn.Linear(int(lti_decoder_hidden), self.state_dim),
        )
        # Latent target encoder for KG error.  Kept separate from input injection
        # because source injection and latent coordinates need not be identical.
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


    def load_state_dict(self, state_dict, strict: bool = True):
        """Allow initializing the Mamba submodule from an old mamba_state_vector checkpoint.

        Old checkpoints have keys like ``in_proj.1.weight`` and ``layers.0...``.
        This dual-state model stores the same backbone under ``mamba.*``.  When
        such a checkpoint is loaded with --non_strict_ckpt, we automatically
        prefix matching old keys with ``mamba.`` so the backbone is reused while
        the LTI/KG branch is freshly initialized.
        """
        sd = dict(state_dict)
        own = super().state_dict()
        # If this already looks like a dual checkpoint, use it directly.
        if not any(str(k).startswith("mamba.") for k in sd.keys()):
            mapped = {}
            for k, v in sd.items():
                mk = "mamba." + str(k)
                if mk in own and tuple(own[mk].shape) == tuple(v.shape):
                    mapped[mk] = v
                elif k in own and tuple(own[k].shape) == tuple(v.shape):
                    mapped[k] = v
            # Keep original entries too for any exact matches; mapped prefix wins.
            sd = {**sd, **mapped}
        return super().load_state_dict(sd, strict=strict)

    def init_state(self, batch_size: int, device=None, dtype=None):
        device = device if device is not None else next(self.parameters()).device
        dtype = dtype if dtype is not None else next(self.parameters()).dtype
        h_m = self.mamba.init_state(batch_size, device=device, dtype=dtype)
        z_lti = torch.zeros(batch_size, self.lti_channels, self.lti_channel_dim, device=device, dtype=dtype)
        return (h_m, z_lti)

    def detach_state(self, state):
        h_m, z_lti = state
        return (h_m.detach(), z_lti.detach())

    def _pool_stim_point(self, stim_t: Optional[torch.Tensor], B: int, device, dtype) -> torch.Tensor:
        return self.mamba._pool_stim_point(stim_t, B, device, dtype)

    def _lti_input_term(self, x_t: torch.Tensor, stim_t: Optional[torch.Tensor]) -> torch.Tensor:
        B = x_t.shape[0]
        stim = self._pool_stim_point(stim_t, B, x_t.device, x_t.dtype)
        inp = torch.cat([x_t.reshape(B, -1), stim], dim=-1) if self.has_external_input else x_t.reshape(B, -1)
        b = self.lti_input(inp).view(B, self.lti_channels, self.lti_channel_dim)
        return b

    def lti_decode(self, z_lti: torch.Tensor) -> torch.Tensor:
        B = z_lti.shape[0]
        return self.lti_decoder(z_lti.reshape(B, -1))

    def encode_target_lti(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return self.lti_target_encoder(x.reshape(B, -1)).view(B, self.lti_channels, self.lti_channel_dim)

    def fusion_alpha(self, horizon_index: Optional[int] = None, device=None, dtype=None) -> torch.Tensor | float:
        # horizon_index is 1-based when supplied.  For ordinary step() training
        # inside Kshort, fixed alpha is normally used.
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

        alpha = self.fusion_alpha(horizon_index, device=x_t.device, dtype=x_t.dtype)
        pred = alpha * pred_m + (1.0 - alpha) * pred_lti
        next_state = (h_m_next, z_lti_next)

        if return_aux:
            aux: Dict[str, Any] = dict(aux_m)
            aux.update({
                "pred_mamba": pred_m,
                "pred_lti": pred_lti,
                "z_lti": z_lti_next,
                "fusion_alpha": alpha if torch.is_tensor(alpha) else pred.new_tensor(float(alpha)),
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
        """Penalize ||A^q (z_pred - E(x_target))|| for q up to horizon.

        This is the channel-wise KG/Gram loss: the first latent prediction error
        should not survive under long powers of the time-invariant A.
        """
        K = max(1, int(horizon))
        target_z = self.encode_target_lti(target_x)
        if detach_target_encoder:
            target_z = target_z.detach()
        e = z_pred - target_z
        losses = []
        rels = []
        start_q = 0 if include_k0 else 1
        cur = e
        for q in range(0, K):
            if q >= start_q:
                step = cur.pow(2).mean()
                if decay != 1.0:
                    step = (float(decay) ** q) * step
                losses.append(step)
                denom = target_z.pow(2).mean().sqrt().clamp_min(1e-6)
                rels.append(cur.pow(2).mean().sqrt() / denom)
            if q != K - 1:
                cur = self.lti_transition(cur)
        if losses:
            loss = torch.stack(losses).mean() if reduce == "mean" else torch.stack(losses).sum()
            rel = torch.stack(rels).mean()
        else:
            loss = e.new_tensor(0.0)
            rel = e.new_tensor(0.0)
        logs = {
            "kg_first_latent_mse": e.pow(2).mean().detach(),
            "kg_mean_rel": rel.detach(),
        }
        return loss, logs

    def lti_spectral_penalty(self, max_sigma: float = 1.0) -> torch.Tensor:
        return self.lti_transition.spectral_penalty(max_sigma=max_sigma)

    def predict_frame_from_history(self, history: torch.Tensor, stim_window: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, W = history.shape[:2]
        s = self.init_state(B, history.device, history.dtype)
        stim = self._pool_stim(stim_window, B, W, history.device, history.dtype) if self.has_external_input else None
        for j in range(max(W - 1, 0)):
            stim_j = stim[:, j] if stim is not None else None
            _, s = self.step(s, history[:, j], stim_j, return_aux=False)
        stim_last = stim[:, W - 1] if stim is not None and W > 0 else None
        pred, _ = self.step(s, history[:, -1], stim_last, return_aux=False)
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
