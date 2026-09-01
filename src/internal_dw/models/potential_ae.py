"""Stimulus-driven coboundary potential autoencoder.

Stage A learns a potential coordinate

    v_t = V_psi(x_t)

and a stimulus-driven external-work increment

    Delta v_t = G_eta(u_t),

with endpoint-pair coboundary structure

    V(x_j) - V(x_i) ~= W(u_{i:j}) = sum_{tau=i}^{j-1} G(u_tau).

The decoder maps potential coordinates back to the potential-explained fMRI
component:

    x_t^Phi = D_omega(v_t).

This is intentionally not a plain AE and not a direct predictor of full future
fMRI.  The potential coordinate is state-dependent, while its endpoint change is
constrained by external input only; path-dependent residuals are left for Stage B.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, depth: int, dropout: float = 0.0):
        super().__init__()
        depth = max(int(depth), 1)
        layers = []
        d = int(in_dim)
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden_dim), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PotentialAEVectorModel(nn.Module):
    """Potential autoencoder with stimulus-driven latent increments.

    Args:
        state_dim: observable state dimension, e.g. 400 HCP parcels.
        input_dim: external stimulus feature dimension.  If input_dim <= 0,
            ``stim_delta`` returns zeros and the model degenerates to a static
            potential AE.
    """

    is_potential_ae = True

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        depth: int = 2,
        dropout: float = 0.0,
        potential_hidden_dim: int | None = None,
        potential_depth: int | None = None,
        input_dim: int = 0,
        stim_hidden_dim: int | None = None,
        stim_depth: int | None = None,
        stim_context_len: int = 1,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.input_dim = int(input_dim)
        self.stim_context_len = max(1, int(stim_context_len))
        delta_hidden = int(potential_hidden_dim if potential_hidden_dim is not None else hidden_dim)
        delta_depth = int(potential_depth if potential_depth is not None else depth)
        stim_hidden = int(stim_hidden_dim if stim_hidden_dim is not None else delta_hidden)
        stim_depth = int(stim_depth if stim_depth is not None else delta_depth)

        self.encoder = MLP(self.state_dim, self.latent_dim, self.hidden_dim, self.depth, dropout)
        self.decoder = MLP(self.latent_dim, self.state_dim, self.hidden_dim, self.depth, dropout)

        # Backward-compatible local delta decoder.  The new stimulus-coboundary
        # Stage A does not need it as the main constraint, but older diagnostics
        # may still call decode_delta().
        self.delta_decoder = MLP(self.latent_dim, self.state_dim, delta_hidden, delta_depth, dropout)

        if self.input_dim > 0:
            # G sees a causal stimulus context, flattened as
            # [u_{t-L+1}, ..., u_t].  L=1 recovers the previous one-step
            # stimulus-only increment.
            self.stim_encoder = MLP(self.input_dim * self.stim_context_len, self.latent_dim, stim_hidden, stim_depth, dropout)
        else:
            self.stim_encoder = None

    def _apply_time_shared(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if x.dim() == 3:
            y = x.reshape(-1, orig_shape[-1])
            z = module(y)
            return z.reshape(orig_shape[0], orig_shape[1], -1)
        return module(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply_time_shared(self.encoder, x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._apply_time_shared(self.decoder, z)

    def decode_delta(self, dz: torch.Tensor) -> torch.Tensor:
        return self._apply_time_shared(self.delta_decoder, dz)

    def _normalize_stim_shape(self, u: torch.Tensor) -> torch.Tensor:
        """Convert raw HCP stimulus tensors to per-time feature vectors.

        The HCP loader can expose SDXL/CLIP features either as [B,T,1664],
        [B,T,256,1664], or already-flattened [B,T,256*1664].  The stimulus
        increment network G expects the old-RR style per-time vector [*,1664],
        so token dimensions must be mean-pooled instead of concatenated.
        """
        if self.input_dim <= 0:
            return u

        # Case: explicit token dimension, e.g. [B,T,N,D] or [B,N,D].
        if u.dim() >= 3 and int(u.shape[-1]) == self.input_dim and u.dim() > 3:
            return u.mean(dim=-2)

        # Case: flattened tokens, e.g. [B,T,N*D] or [B,N*D].
        last = int(u.shape[-1])
        if last != self.input_dim and last % self.input_dim == 0:
            n_tokens = last // self.input_dim
            return u.reshape(*u.shape[:-1], n_tokens, self.input_dim).mean(dim=-2)

        return u

    def _zero_delta_like(self, u: torch.Tensor | None, like_v: torch.Tensor | None = None) -> torch.Tensor:
        if like_v is not None:
            return like_v.new_zeros(*like_v.shape[:-1], self.latent_dim)
        if u is None:
            raise ValueError("stim_delta needs either stimulus or like_v when input_dim <= 0")
        return u.new_zeros(*u.shape[:-1], self.latent_dim)

    def _causal_context_flatten(self, u_seq: torch.Tensor) -> torch.Tensor:
        """Build causal stimulus windows for a sequence.

        Input:  [B,T,S]
        Output: [B,T,L*S], where row t contains
                [u_{t-L+1}, ..., u_t] with left zero padding.
        """
        if int(u_seq.shape[-1]) != self.input_dim:
            raise RuntimeError(
                f"stim_delta expected last dim {self.input_dim}, got shape {tuple(u_seq.shape)}. "
                "For HCP flattened token features, the last dimension should be divisible by stim_dim."
            )
        B, T, S = u_seq.shape
        L = self.stim_context_len
        if L <= 1:
            return u_seq
        pad = u_seq.new_zeros(B, L - 1, S)
        padded = torch.cat([pad, u_seq], dim=1)
        chunks = [padded[:, i:i + T] for i in range(L)]
        ctx = torch.stack(chunks, dim=2)  # [B,T,L,S]
        return ctx.reshape(B, T, L * S)

    def _single_context_flatten(self, u_window: torch.Tensor) -> torch.Tensor:
        """Flatten one causal context window.

        Input after normalization: [B,S] or [B,L,S].
        Output: [B,L*S] with left zero padding/truncation to stim_context_len.
        """
        u_window = self._normalize_stim_shape(u_window)
        if u_window.dim() == 2:
            # Current stimulus only; put it in the most recent slot.
            if int(u_window.shape[-1]) != self.input_dim:
                raise RuntimeError(f"stim_delta window expected feature dim {self.input_dim}, got {tuple(u_window.shape)}")
            if self.stim_context_len <= 1:
                return u_window
            z = u_window.new_zeros(u_window.shape[0], self.stim_context_len, self.input_dim)
            z[:, -1] = u_window
            return z.reshape(u_window.shape[0], self.stim_context_len * self.input_dim)

        if u_window.dim() != 3:
            raise RuntimeError(f"stim_delta_context_window expects [B,S] or [B,L,S] after pooling, got {tuple(u_window.shape)}")
        if int(u_window.shape[-1]) != self.input_dim:
            raise RuntimeError(f"stim_delta window expected feature dim {self.input_dim}, got {tuple(u_window.shape)}")
        B, L_in, S = u_window.shape
        L = self.stim_context_len
        if L_in > L:
            u_window = u_window[:, -L:]
            L_in = L
        if L_in < L:
            pad = u_window.new_zeros(B, L - L_in, S)
            u_window = torch.cat([pad, u_window], dim=1)
        return u_window.reshape(B, L * S)

    def stim_delta_context_window(self, u_window: torch.Tensor | None, like_v: torch.Tensor | None = None) -> torch.Tensor:
        """Return G(u_{t-L+1:t}) for one transition.

        This is used by Stage B rollout/evaluation, where each step needs a
        single stimulus-history-conditioned potential increment.
        """
        if self.stim_encoder is None:
            return self._zero_delta_like(u_window, like_v=like_v)
        if u_window is None:
            raise ValueError("PotentialAEVectorModel with input_dim>0 requires stimulus for stim_delta_context_window")
        flat = self._single_context_flatten(u_window)
        return self.stim_encoder(flat)

    def stim_delta(self, u: torch.Tensor | None, like_v: torch.Tensor | None = None) -> torch.Tensor:
        """Return stimulus-history-driven potential increments.

        For sequence input [B,T,S] (or HCP token variants), output [B,T,Lz],
        where each increment uses the recent ``stim_context_len`` external
        inputs.  For single-step input [B,S], output one increment with zero
        left padding.
        """
        if self.stim_encoder is None:
            return self._zero_delta_like(u, like_v=like_v)
        if u is None:
            raise ValueError("PotentialAEVectorModel with input_dim>0 requires stimulus for stim_delta")
        u = self._normalize_stim_shape(u)
        if u.dim() == 2:
            return self.stim_delta_context_window(u, like_v=like_v)
        if u.dim() != 3:
            raise RuntimeError(f"stim_delta expects [B,S] or [B,T,S] after pooling, got {tuple(u.shape)}")
        flat = self._causal_context_flatten(u)
        return self._apply_time_shared(self.stim_encoder, flat)

    def potential_value(self, z: torch.Tensor) -> torch.Tensor:
        # The potential is vector-valued; return z itself.
        return z

    def forward(self, x: torch.Tensor, stim: torch.Tensor | None = None):
        v = self.encode(x)
        recon = self.decode(v)
        return recon, v, v
