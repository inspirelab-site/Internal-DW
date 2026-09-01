"""Finite-horizon low-dimensional path-generator model for 2D fields.

The model does not learn a one-step map and then unroll it during training.
Instead it maps the initial/history state to a *generator*: a sequence of
coarse velocity increments.  A deterministic integration operator converts
those increments into a predicted future path.

Training can supervise the entire induced path without saving a K-step graph of
an expensive recurrent predictor.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_field import ConvGNAct, ResidualBlock


class LowDimPathGeneratorFieldModel(nn.Module):
    """Low-dimensional finite-horizon generator for field trajectories.

    Interface compatibility:
      - forward(stim_window, history) returns the first generated frame as
        [B, 1, C, H, W], so the existing AR evaluator can roll it out.
      - generate_path(history, horizon=K) returns [B, K, C, H, W].

    Internally the model predicts coarse normalized velocity increments
    [B, K, C, G, G].  These increments are integrated in the coarse latent grid
    and then bilinearly lifted to the full field.  This keeps the generator
    low-dimensional and prevents the method from degenerating into direct
    full-resolution sequence prediction.
    """

    is_standard_autoregressive = True
    is_path_generator = True
    task_type = "field2d"

    def __init__(
        self,
        field_channels: int,
        window_size: int = 4,
        path_horizon: int = 8,
        generator_size: int = 8,
        base_channels: int = 64,
        code_channels: int = 64,
        groups: int = 8,
        use_grid: bool = True,
        normalize: bool = True,
        velocity_scale: float = 1.0,
    ):
        super().__init__()
        self.field_channels = int(field_channels)
        self.window_size = int(window_size)
        self.path_horizon = int(path_horizon)
        self.generator_size = int(generator_size)
        self.base_channels = int(base_channels)
        self.code_channels = int(code_channels)
        self.groups = int(groups)
        self.use_grid = bool(use_grid)
        self.normalize = bool(normalize)
        self.velocity_scale = float(velocity_scale)

        in_ch = self.window_size * self.field_channels + (2 if self.use_grid else 0)
        mid_ch = max(self.base_channels, self.code_channels)
        self.encoder = nn.Sequential(
            ResidualBlock(in_ch, self.base_channels, groups=self.groups),
            nn.Conv2d(self.base_channels, mid_ch, kernel_size=3, stride=2, padding=1),
            ResidualBlock(mid_ch, mid_ch, groups=self.groups),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=2, padding=1),
            ResidualBlock(mid_ch, self.code_channels, groups=self.groups),
        )
        self.generator_head = nn.Sequential(
            ConvGNAct(self.code_channels, self.code_channels, groups=self.groups),
            nn.Conv2d(self.code_channels, self.path_horizon * self.field_channels, kernel_size=1),
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
            raise ValueError(f"PathGenerator expects history [B,W,C,H,W], got {tuple(history.shape)}")
        b, win, c, h, w = history.shape
        if win != self.window_size or c != self.field_channels:
            raise ValueError(
                f"Expected history [B,{self.window_size},{self.field_channels},H,W], got {tuple(history.shape)}"
            )
        x = self._normalize_history(history).reshape(b, win * c, h, w)
        if self.use_grid:
            x = torch.cat([x, self._grid(b, h, w, x.device, x.dtype)], dim=1)
        return x

    def encode_path_observable(self, frame: torch.Tensor) -> torch.Tensor:
        """Low-dimensional normalized observable used by the MMSBM-style loss."""
        if frame.dim() != 4:
            raise ValueError(f"Expected frame [B,C,H,W], got {tuple(frame.shape)}")
        frame = self._normalize_frame(frame)
        pooled = F.adaptive_avg_pool2d(frame, (self.generator_size, self.generator_size))
        return pooled.reshape(pooled.shape[0], -1)

    def predict_generator(self, history: torch.Tensor, horizon: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return coarse velocity increments and encoder code.

        Returns:
            deltas: [B, K, C, G, G] in normalized coarse-state coordinates.
            code:   [B, code_channels, G, G]
        """
        b, _, _, h, w = history.shape
        K = int(horizon or self.path_horizon)
        if K < 1:
            raise ValueError("horizon must be >= 1")
        if K > self.path_horizon:
            raise ValueError(
                f"Requested horizon={K}, but model was built with path_horizon={self.path_horizon}."
            )
        feat = self.forward_features(history)
        code = self.encoder(feat)
        code = F.adaptive_avg_pool2d(code, (self.generator_size, self.generator_size))
        deltas = self.generator_head(code)
        deltas = deltas.view(b, self.path_horizon, self.field_channels, self.generator_size, self.generator_size)
        deltas = deltas[:, :K] * self.velocity_scale
        return deltas, code

    def integrate_generator(self, history: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """Integrate coarse velocity increments into full-resolution future frames."""
        b, K, c, g1, g2 = deltas.shape
        _, _, _, h, w = history.shape
        last_norm = self._normalize_frame(history[:, -1])
        z0 = F.adaptive_avg_pool2d(last_norm, (g1, g2))
        z_path = z0.unsqueeze(1) + torch.cumsum(deltas, dim=1)
        z_flat = z_path.reshape(b * K, c, g1, g2)
        full_norm = F.interpolate(z_flat, size=(h, w), mode="bilinear", align_corners=False)
        full_norm = full_norm.view(b, K, c, h, w)
        full = self._decode_frame(full_norm.reshape(b * K, c, h, w)).view(b, K, c, h, w)
        return full

    def generate_path(self, history: torch.Tensor, horizon: Optional[int] = None, return_aux: bool = False):
        deltas, code = self.predict_generator(history, horizon=horizon)
        path = self.integrate_generator(history, deltas)
        if return_aux:
            aux = {
                "path_generator_deltas": deltas,
                "path_generator_code": code,
                "path_pred": path,
            }
            return path, aux
        return path

    def predict_frame_from_history(self, history: torch.Tensor) -> torch.Tensor:
        return self.generate_path(history, horizon=1, return_aux=False)[:, 0]

    def step_history(self, history: torch.Tensor) -> torch.Tensor:
        frame = self.predict_frame_from_history(history)
        return torch.cat([history[:, 1:], frame.unsqueeze(1)], dim=1)

    def forward(self, stim_window: Optional[torch.Tensor], history: torch.Tensor, return_aux: bool = False):
        # The current implementation uses the history/initial state.  The
        # external-input sequence is kept in the signature for compatibility and
        # for future conditioning extensions.
        path, aux = self.generate_path(history, horizon=1, return_aux=True)
        pred = path[:, :1]
        if return_aux:
            aux["pred_frame"] = pred[:, 0]
            return pred, aux
        return pred
