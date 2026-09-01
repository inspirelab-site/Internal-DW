"""State-token path models for data-dependent temporal state abstraction.

The main idea is to replace a dense clock-time future path by a short ordered
sequence of latent state tokens.  Each token has a learned temporal receptive
field, parameterized by a center time and a decay/width.  Future states are
constructed by softly mixing token states according to these receptive fields.

This is intentionally implemented with the existing path-generator interface:
    history -> generate_path(history, horizon=K) -> [B,K,...]
so it can be trained/evaluated by the current finite-horizon path losses for
both HCP/vector data and The Well/field data.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_field import ConvGNAct, ResidualBlock


def _ordered_centers_from_raw(raw: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Map unconstrained [B,L] values to ordered centers in [0,1]."""
    # Positive interval lengths, normalized to sum to one.  The cumulative midpoints
    # are strictly ordered and stay inside (0,1).
    intervals = F.softplus(raw) + eps
    total = intervals.sum(dim=-1, keepdim=True).clamp_min(eps)
    left = torch.cumsum(intervals, dim=-1) - intervals
    centers = (left + 0.5 * intervals) / total
    return centers.clamp(0.0, 1.0)


def _temporal_token_weights(
    center: torch.Tensor,
    log_decay: torch.Tensor,
    gate_logit: torch.Tensor,
    horizon: int,
    decay_min: float = 0.5,
    decay_max: float = 80.0,
) -> torch.Tensor:
    """Return soft token assignment weights [B,K,L].

    center/log_decay/gate_logit: [B,L].  center is normalized event time in [0,1].
    decay controls locality. Larger decay = narrower token receptive field.
    """
    B, L = center.shape
    if horizon <= 1:
        u = center.new_zeros(1)
    else:
        u = torch.linspace(0.0, 1.0, horizon, device=center.device, dtype=center.dtype)
    u = u.view(1, horizon, 1)
    c = center.view(B, 1, L)
    decay = (F.softplus(log_decay) + float(decay_min)).clamp(max=float(decay_max)).view(B, 1, L)
    gate = gate_logit.view(B, 1, L)
    logits = gate - decay * torch.abs(u - c)
    return torch.softmax(logits, dim=-1)


class StateTokenPathVectorModel(nn.Module):
    """Finite-horizon state-token model for vector trajectories such as HCP fMRI.

    The model predicts a short sequence of ordered latent state tokens from the
    history.  Each token has a temporal center and decay.  A future frame is a
    soft mixture of token values, optionally with a small future-stimulus forcing
    term when stimulus features are available.
    """

    is_standard_autoregressive = True
    is_path_generator = True
    is_state_token_path = True
    uses_future_stimulus = True
    task_type = "vector"

    def __init__(
        self,
        state_dim: int,
        input_dim: int = 1,
        window_size: int = 4,
        path_horizon: int = 8,
        num_tokens: int = 8,
        hidden_dim: int = 512,
        token_mlp_layers: int = 2,
        has_external_input: bool = False,
        decay_min: float = 0.5,
        decay_max: float = 80.0,
        residual_from_last: bool = True,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.input_dim = int(input_dim)
        self.window_size = int(window_size)
        self.path_horizon = int(path_horizon)
        self.num_tokens = int(num_tokens)
        self.hidden_dim = int(hidden_dim)
        self.has_external_input = bool(has_external_input)
        self.decay_min = float(decay_min)
        self.decay_max = float(decay_max)
        self.residual_from_last = bool(residual_from_last)

        in_dim = self.window_size * self.state_dim
        if self.has_external_input:
            in_dim += self.window_size * self.input_dim

        layers = []
        d = in_dim
        for _ in range(max(1, int(token_mlp_layers))):
            layers += [nn.Linear(d, self.hidden_dim), nn.GELU(), nn.LayerNorm(self.hidden_dim)]
            d = self.hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.token_head = nn.Linear(self.hidden_dim, self.num_tokens * self.state_dim)
        self.param_head = nn.Linear(self.hidden_dim, self.num_tokens * 3)  # center intervals, log_decay, gate_logit

        if self.has_external_input:
            self.future_stim_head = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.state_dim),
            )
        else:
            self.future_stim_head = None

    def _pool_stim(self, stim: torch.Tensor) -> torch.Tensor:
        if stim is None:
            return None
        if stim.dim() > 3:
            return stim.reshape(stim.shape[0], stim.shape[1], -1)
        return stim

    def _context(self, history: torch.Tensor, stim_window: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = history.shape[0]
        x = history.reshape(B, self.window_size, -1)
        if x.shape[-1] != self.state_dim:
            raise ValueError(f"Expected vector state_dim={self.state_dim}, got history {tuple(history.shape)}")
        feat = [x.reshape(B, -1)]
        if self.has_external_input:
            if stim_window is None:
                stim_window = history.new_zeros(B, self.window_size, self.input_dim)
            stim_window = self._pool_stim(stim_window)
            # If upstream HCP features are flattened token grids, recover the old mean-pooled dim when possible.
            if stim_window.shape[-1] != self.input_dim and stim_window.shape[-1] % self.input_dim == 0:
                stim_window = stim_window.view(B, stim_window.shape[1], -1, self.input_dim).mean(dim=2)
            if stim_window.shape[-1] != self.input_dim:
                raise ValueError(f"Expected stim_dim={self.input_dim}, got {tuple(stim_window.shape)}")
            feat.append(stim_window.reshape(B, -1))
        return torch.cat(feat, dim=-1)

    def _tokenize(self, history: torch.Tensor, stim_window: Optional[torch.Tensor] = None):
        B = history.shape[0]
        h = self.encoder(self._context(history, stim_window))
        token = self.token_head(h).view(B, self.num_tokens, self.state_dim)
        params = self.param_head(h).view(B, self.num_tokens, 3)
        centers = _ordered_centers_from_raw(params[..., 0])
        log_decay = params[..., 1]
        gate_logit = params[..., 2]
        return token, centers, log_decay, gate_logit, h

    def encode_path_observable(self, frame: torch.Tensor) -> torch.Tensor:
        return frame.reshape(frame.shape[0], -1)

    def generate_path(
        self,
        history: torch.Tensor,
        horizon: Optional[int] = None,
        return_aux: bool = False,
        stim_window: Optional[torch.Tensor] = None,
        stim_future: Optional[torch.Tensor] = None,
    ):
        K = int(horizon or self.path_horizon)
        if K < 1:
            raise ValueError("horizon must be >= 1")
        if K > self.path_horizon:
            raise ValueError(f"Requested horizon={K}, model path_horizon={self.path_horizon}")
        B = history.shape[0]
        token, centers, log_decay, gate_logit, code = self._tokenize(history, stim_window=stim_window)
        weights = _temporal_token_weights(centers, log_decay, gate_logit, K, self.decay_min, self.decay_max)
        path = torch.einsum("bkl,bld->bkd", weights, token)
        if self.residual_from_last:
            path = history.reshape(B, self.window_size, -1)[:, -1:].expand(B, K, self.state_dim) + path

        if self.has_external_input and self.future_stim_head is not None and stim_future is not None:
            stim_future = self._pool_stim(stim_future)
            if stim_future.shape[-1] != self.input_dim and stim_future.shape[-1] % self.input_dim == 0:
                stim_future = stim_future.view(B, stim_future.shape[1], -1, self.input_dim).mean(dim=2)
            if stim_future.shape[1] >= K and stim_future.shape[-1] == self.input_dim:
                force = self.future_stim_head(stim_future[:, :K].reshape(B * K, self.input_dim)).view(B, K, self.state_dim)
                path = path + force

        if return_aux:
            first = path[:, :1] - history.reshape(B, self.window_size, -1)[:, -1:].expand(B, 1, self.state_dim)
            rest = path[:, 1:] - path[:, :-1] if K > 1 else path[:, :0]
            deltas = torch.cat([first, rest], dim=1)
            aux = {
                "path_generator_deltas": deltas,
                "path_generator_code": code,
                "path_pred": path,
                "state_token_centers": centers,
                "state_token_log_decay": log_decay,
                "state_token_gates": torch.sigmoid(gate_logit),
                "state_token_weights": weights,
            }
            return path, aux
        return path

    def predict_frame_from_history(self, history: torch.Tensor, stim_window: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.generate_path(history, horizon=1, return_aux=False, stim_window=stim_window)[:, 0]

    def step_history(self, history: torch.Tensor) -> torch.Tensor:
        frame = self.predict_frame_from_history(history)
        return torch.cat([history[:, 1:], frame.unsqueeze(1)], dim=1)

    def forward(self, stim_window: Optional[torch.Tensor], history: torch.Tensor, return_aux: bool = False):
        out = self.generate_path(history, horizon=1, return_aux=return_aux, stim_window=stim_window)
        if return_aux:
            path, aux = out
            aux["pred_frame"] = path[:, 0]
            return path[:, :1], aux
        return out[:, :1]


class StateTokenPathFieldModel(nn.Module):
    """State-token finite-horizon path model for 2D fields.

    History is encoded by a small CNN.  The model emits L coarse state tokens,
    each with a learned temporal center and decay.  Future coarse states are
    soft mixtures of the L tokens and are then bilinearly lifted to full fields.
    """

    is_standard_autoregressive = True
    is_path_generator = True
    is_state_token_path = True
    uses_future_stimulus = False
    task_type = "field2d"

    def __init__(
        self,
        field_channels: int,
        window_size: int = 4,
        path_horizon: int = 8,
        num_tokens: int = 8,
        generator_size: int = 8,
        base_channels: int = 64,
        code_channels: int = 64,
        groups: int = 8,
        use_grid: bool = True,
        normalize: bool = True,
        decay_min: float = 0.5,
        decay_max: float = 80.0,
        residual_from_last: bool = True,
    ):
        super().__init__()
        self.field_channels = int(field_channels)
        self.window_size = int(window_size)
        self.path_horizon = int(path_horizon)
        self.num_tokens = int(num_tokens)
        self.generator_size = int(generator_size)
        self.base_channels = int(base_channels)
        self.code_channels = int(code_channels)
        self.groups = int(groups)
        self.use_grid = bool(use_grid)
        self.normalize = bool(normalize)
        self.decay_min = float(decay_min)
        self.decay_max = float(decay_max)
        self.residual_from_last = bool(residual_from_last)

        in_ch = self.window_size * self.field_channels + (2 if self.use_grid else 0)
        mid_ch = max(self.base_channels, self.code_channels)
        self.encoder = nn.Sequential(
            ResidualBlock(in_ch, self.base_channels, groups=self.groups),
            nn.Conv2d(self.base_channels, mid_ch, kernel_size=3, stride=2, padding=1),
            ResidualBlock(mid_ch, mid_ch, groups=self.groups),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=2, padding=1),
            ResidualBlock(mid_ch, self.code_channels, groups=self.groups),
        )
        self.token_head = nn.Sequential(
            ConvGNAct(self.code_channels, self.code_channels, groups=self.groups),
            nn.Conv2d(self.code_channels, self.num_tokens * self.field_channels, kernel_size=1),
        )
        self.param_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.code_channels, self.code_channels),
            nn.GELU(),
            nn.Linear(self.code_channels, self.num_tokens * 3),
        )

        self.register_buffer("state_mean", torch.zeros(1, 1, self.field_channels, 1, 1))
        self.register_buffer("state_std", torch.ones(1, 1, self.field_channels, 1, 1))
        self.normalizer_fitted = False

    def set_state_normalizer(self, mean: torch.Tensor, std: torch.Tensor):
        mean = mean.detach().float().view(1, 1, self.field_channels, 1, 1)
        std = std.detach().float().clamp_min(1e-6).view(1, 1, self.field_channels, 1, 1)
        self.state_mean.copy_(mean.to(self.state_mean.device))
        self.state_std.copy_(std.to(self.state_std.device))
        self.normalizer_fitted = True

    def _grid(self, b: int, h: int, w: int, device, dtype) -> torch.Tensor:
        gridx = torch.linspace(0, 1, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        gridy = torch.linspace(0, 1, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([gridx, gridy], dim=1)

    def _normalize_history(self, history: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return history
        return (history - self.state_mean) / self.state_std

    def _normalize_frame(self, frame: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return frame
        return (frame.unsqueeze(1) - self.state_mean)[:, 0] / self.state_std[:, 0]

    def _decode_frame(self, frame_norm: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return frame_norm
        return frame_norm * self.state_std[:, 0] + self.state_mean[:, 0]

    def forward_features(self, history: torch.Tensor) -> torch.Tensor:
        if history.dim() != 5:
            raise ValueError(f"StateTokenPathField expects history [B,W,C,H,W], got {tuple(history.shape)}")
        b, win, c, h, w = history.shape
        if win != self.window_size or c != self.field_channels:
            raise ValueError(f"Expected history [B,{self.window_size},{self.field_channels},H,W], got {tuple(history.shape)}")
        x = self._normalize_history(history).reshape(b, win * c, h, w)
        if self.use_grid:
            x = torch.cat([x, self._grid(b, h, w, x.device, x.dtype)], dim=1)
        return x

    def _tokenize(self, history: torch.Tensor):
        b = history.shape[0]
        feat = self.forward_features(history)
        code = self.encoder(feat)
        code = F.adaptive_avg_pool2d(code, (self.generator_size, self.generator_size))
        token = self.token_head(code).view(b, self.num_tokens, self.field_channels, self.generator_size, self.generator_size)
        params = self.param_head(code).view(b, self.num_tokens, 3)
        centers = _ordered_centers_from_raw(params[..., 0])
        log_decay = params[..., 1]
        gate_logit = params[..., 2]
        return token, centers, log_decay, gate_logit, code

    def encode_path_observable(self, frame: torch.Tensor) -> torch.Tensor:
        if frame.dim() != 4:
            raise ValueError(f"Expected frame [B,C,H,W], got {tuple(frame.shape)}")
        frame = self._normalize_frame(frame)
        pooled = F.adaptive_avg_pool2d(frame, (self.generator_size, self.generator_size))
        return pooled.reshape(pooled.shape[0], -1)

    def generate_path(self, history: torch.Tensor, horizon: Optional[int] = None, return_aux: bool = False, **_kwargs):
        K = int(horizon or self.path_horizon)
        if K < 1:
            raise ValueError("horizon must be >= 1")
        if K > self.path_horizon:
            raise ValueError(f"Requested horizon={K}, model path_horizon={self.path_horizon}")
        b, _, _, h, w = history.shape
        token, centers, log_decay, gate_logit, code = self._tokenize(history)
        weights = _temporal_token_weights(centers, log_decay, gate_logit, K, self.decay_min, self.decay_max)
        coarse = torch.einsum("bkl,blcgh->bkcgh", weights, token)
        if self.residual_from_last:
            last_norm = self._normalize_frame(history[:, -1])
            z0 = F.adaptive_avg_pool2d(last_norm, (self.generator_size, self.generator_size))
            coarse = z0.unsqueeze(1) + coarse
        flat = coarse.reshape(b * K, self.field_channels, self.generator_size, self.generator_size)
        full_norm = F.interpolate(flat, size=(h, w), mode="bilinear", align_corners=False)
        full = self._decode_frame(full_norm).view(b, K, self.field_channels, h, w)
        if return_aux:
            z0 = F.adaptive_avg_pool2d(self._normalize_frame(history[:, -1]), (self.generator_size, self.generator_size))
            first = coarse[:, :1] - z0.unsqueeze(1)
            rest = coarse[:, 1:] - coarse[:, :-1] if K > 1 else coarse[:, :0]
            deltas = torch.cat([first, rest], dim=1)
            aux = {
                "path_generator_deltas": deltas,
                "path_generator_code": code,
                "path_pred": full,
                "state_token_centers": centers,
                "state_token_log_decay": log_decay,
                "state_token_gates": torch.sigmoid(gate_logit),
                "state_token_weights": weights,
                "state_token_coarse_path": coarse,
            }
            return full, aux
        return full

    def predict_frame_from_history(self, history: torch.Tensor) -> torch.Tensor:
        return self.generate_path(history, horizon=1, return_aux=False)[:, 0]

    def step_history(self, history: torch.Tensor) -> torch.Tensor:
        frame = self.predict_frame_from_history(history)
        return torch.cat([history[:, 1:], frame.unsqueeze(1)], dim=1)

    def forward(self, stim_window: Optional[torch.Tensor], history: torch.Tensor, return_aux: bool = False):
        path, aux = self.generate_path(history, horizon=1, return_aux=True)
        pred = path[:, :1]
        if return_aux:
            aux["pred_frame"] = pred[:, 0]
            return pred, aux
        return pred
