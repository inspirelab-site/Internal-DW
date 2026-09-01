"""Causal-query predictive-subspace attention AR model.

This model implements the non-leaking version of the idea:

    global learned reference dictionary / predictive subspace
    + causal time-point query
    + compact latent transition kernel
    + AR rollout at test time

The reference tokens are trainable parameters learned from training data.  They
are not recomputed from the current test sequence, so they do not contain future
fMRI from the test subject.  Each time point t produces its own causal query
from x_{<=t}, u_{<=t}; the query attends to the global dictionary to obtain a
low-dimensional latent coordinate z_t.  A small latent transition propagates
z_t to z_{t+1}, and the decoder maps the propagated state back to fMRI.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .simple_ar import _StimPoolMixin


def _detach_state(h):
    if torch.is_tensor(h):
        return h.detach()
    if isinstance(h, tuple):
        return tuple(_detach_state(x) for x in h)
    if isinstance(h, list):
        return [_detach_state(x) for x in h]
    return h


class CausalQueryPredictiveSubspaceARModel(nn.Module, _StimPoolMixin):
    """Time-preserving predictive-subspace AR model.

    The model is recurrent-state compatible with the existing HCP Mamba BPTT
    trainer/evaluator.  Its recurrence state is only the causal query encoder
    hidden state, not future information.

    Step semantics match the existing recurrent AR models:
        step(h, x_t, u_t) -> predicted x_{t+1}, h_next

    Internally:
        h_t      = GRUCell([x_t,u_t], h_{t-1})          causal query state
        q_t      = W_q h_t                              time-t query
        z_t      = Attn(q_t, K_ref, V_ref)              coordinate in dictionary
        z_{t+1}  = A z_t + B u_t + b                    compact transition
        xhat_{t+1}= D(z_{t+1}) or x_t + D(z_{t+1})      decoded prediction
    """

    is_standard_autoregressive = True
    is_recurrent_state_ar = True
    is_predictive_subspace_attention = True

    def __init__(
        self,
        state_dim: int,
        input_dim: int = 0,
        state_shape: Optional[Sequence[int]] = None,
        hidden_dim: int = 512,
        latent_dim: int = 128,
        num_refs: int = 128,
        key_dim: int = 128,
        transition_rank: int = 16,
        dropout: float = 0.0,
        has_external_input: bool = True,
        residual: bool = True,
        diag_init: float = 0.98,
        max_diag: float = 0.999,
        temperature: float = 1.0,
        entropy_weight: float = 0.0,
        current_rec_weight: float = 0.0,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.input_dim = int(input_dim)
        self.state_shape = tuple(int(x) for x in state_shape) if state_shape is not None else None
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.num_refs = int(num_refs)
        self.key_dim = int(key_dim)
        self.transition_rank = int(transition_rank)
        self.has_external_input = bool(has_external_input)
        self.residual = bool(residual)
        self.temperature = float(temperature)
        self.entropy_weight = float(entropy_weight)
        self.current_rec_weight = float(current_rec_weight)
        self.max_diag = float(max_diag)
        self.task_type = "field2d" if self.state_shape is not None and len(self.state_shape) == 3 else "vector"

        in_dim = self.state_dim + (self.input_dim if self.has_external_input else 0)
        self.input_norm = nn.LayerNorm(in_dim) if bool(use_layernorm) else nn.Identity()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.hidden_norm = nn.LayerNorm(self.hidden_dim) if bool(use_layernorm) else nn.Identity()
        self.query_proj = nn.Linear(self.hidden_dim, self.key_dim)

        # Global learned reference dictionary.  This is the "attention-learned"
        # predictive subspace; every time point has its own causal query into it.
        self.ref_keys = nn.Parameter(torch.randn(self.num_refs, self.key_dim) / math.sqrt(max(1, self.key_dim)))
        self.ref_values = nn.Parameter(torch.randn(self.num_refs, self.latent_dim) / math.sqrt(max(1, self.latent_dim)))

        if self.has_external_input:
            self.stim_to_latent = nn.Sequential(
                nn.LayerNorm(self.input_dim),
                nn.Linear(self.input_dim, self.latent_dim),
            )
        else:
            self.stim_to_latent = None

        # Compact transition: diagonal + low-rank.  This makes model^k cheap and
        # analyzable in the latent space while keeping the deployed model causal.
        diag_init = max(min(float(diag_init), self.max_diag), -self.max_diag)
        # Store an unconstrained value whose tanh maps near diag_init/max_diag.
        init_ratio = max(min(diag_init / max(self.max_diag, 1e-6), 0.999), -0.999)
        self.raw_diag = nn.Parameter(torch.full((self.latent_dim,), math.atanh(init_ratio)))

        r = max(0, self.transition_rank)
        if r > 0:
            self.A_u = nn.Parameter(torch.randn(self.latent_dim, r) * 1e-3)
            self.A_v = nn.Parameter(torch.randn(self.latent_dim, r) * 1e-3)
        else:
            self.register_parameter("A_u", None)
            self.register_parameter("A_v", None)
        self.trans_bias = nn.Parameter(torch.zeros(self.latent_dim))

        dec_hidden = max(self.hidden_dim, self.latent_dim)
        self.decoder = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, dec_hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(dec_hidden, self.state_dim),
        )
        self.current_decoder = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, dec_hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(dec_hidden, self.state_dim),
        )

    def init_state(self, batch_size: int, device=None, dtype=None):
        device = device if device is not None else next(self.parameters()).device
        dtype = dtype if dtype is not None else next(self.parameters()).dtype
        return torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)

    def detach_state(self, h):
        return _detach_state(h)

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

    def _causal_update(self, h, x_t: torch.Tensor, stim_t: Optional[torch.Tensor]):
        B = x_t.shape[0]
        if h is None:
            h = self.init_state(B, x_t.device, x_t.dtype)
        x_flat = self._flatten_state(x_t)
        stim = self._pool_stim_point(stim_t, B, x_t.device, x_t.dtype)
        inp = torch.cat([x_flat, stim], dim=-1) if self.has_external_input else x_flat
        token = self.input_proj(self.input_norm(inp))
        h_next = self.gru(token, h)
        return h_next, x_flat, stim

    def _attend_subspace(self, h: torch.Tensor):
        q = self.query_proj(self.hidden_norm(h))
        q = F.normalize(q, dim=-1)
        k = F.normalize(self.ref_keys, dim=-1)
        temp = max(float(self.temperature), 1e-4)
        logits = q @ k.t() / temp
        attn = torch.softmax(logits, dim=-1)
        z = attn @ self.ref_values
        return z, attn, logits

    def transition(self, z: torch.Tensor, stim: Optional[torch.Tensor] = None) -> torch.Tensor:
        diag = self.max_diag * torch.tanh(self.raw_diag).to(dtype=z.dtype, device=z.device)
        z_next = z * diag.unsqueeze(0)
        if self.A_u is not None and self.A_v is not None:
            # z @ (U V^T) = (z @ V) @ U^T.  Scale by sqrt(rank) for stable init.
            r = max(1, self.A_u.shape[1])
            z_next = z_next + (z @ self.A_v.to(dtype=z.dtype, device=z.device)) @ self.A_u.to(dtype=z.dtype, device=z.device).t() / math.sqrt(r)
        if self.has_external_input and stim is not None:
            z_next = z_next + self.stim_to_latent(stim)
        z_next = z_next + self.trans_bias.to(dtype=z.dtype, device=z.device).unsqueeze(0)
        return z_next

    def decode(self, z: torch.Tensor, ref_x: torch.Tensor, residual_base: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.decoder(z)
        if self.residual:
            base = self._flatten_state(ref_x) if residual_base is None else residual_base
            out = base + out
        return self._reshape_pred(out, ref_x)

    def step(self, h, x_t: torch.Tensor, stim_t: Optional[torch.Tensor] = None, return_aux: bool = False, horizon_index: Optional[int] = None):
        h_next, x_flat, stim = self._causal_update(h, x_t, stim_t)
        z_t, attn, _logits = self._attend_subspace(h_next)
        z_next = self.transition(z_t, stim)
        pred = self.decode(z_next, x_t, residual_base=x_flat)

        if not return_aux:
            return pred, h_next

        with torch.no_grad():
            ent = (-(attn * attn.clamp_min(1e-8).log()).sum(dim=-1)).mean()
            maxp = attn.max(dim=-1).values.mean()
            diag = self.max_diag * torch.tanh(self.raw_diag)
        aux = {
            "h_next": h_next,
            "hidden_norm": h_next.norm(dim=-1).mean(),
            "alpha_mean": diag.mean(),
            "alpha_min": diag.min(),
            "alpha_max": diag.max(),
            "subspace_attn_entropy": ent,
            "subspace_attn_max": maxp,
            "subspace_z_norm": z_t.norm(dim=-1).mean(),
            "subspace_z_next_norm": z_next.norm(dim=-1).mean(),
        }

        # Optional current-frame reconstruction anchor: z_t should locate the
        # current time point, not only be a hidden code used for next output.
        if self.current_rec_weight != 0.0:
            cur_flat = self.current_decoder(z_t)
            if self.residual:
                # Current latent should reconstruct the current frame itself,
                # so do not add x_t as residual; use absolute decoder here.
                pass
            cur = self._reshape_pred(cur_flat, x_t)
            denom = x_t.reshape(x_t.shape[0], -1).norm(dim=-1).clamp_min(1e-8)
            aux["subspace_current_rec_loss"] = ((cur - x_t).reshape(x_t.shape[0], -1).norm(dim=-1) / denom).mean()
        if self.entropy_weight != 0.0:
            # Encourage non-degenerate dictionary usage.  Positive weight should
            # penalize low entropy: loss term is -entropy.
            aux["subspace_neg_entropy_loss"] = -ent
        return pred, h_next, aux

    def burn_in(self, x_seq: torch.Tensor, stim_seq: Optional[torch.Tensor] = None, h0=None, detach: bool = False):
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
