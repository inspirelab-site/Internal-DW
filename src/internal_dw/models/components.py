"""Small model components shared by the paper backbones.

This module intentionally contains only backbone-agnostic utilities used by
the released Mamba and U-Net implementations.  Keeping them here avoids
retaining the old cross-backbone prototype modules solely for two helpers.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


class StimulusPoolingMixin:
    """Normalize an optional external-input window to ``[B, T, input_dim]``."""

    def _pool_stim(
        self,
        stim_window: Optional[torch.Tensor],
        batch_size: int,
        time_steps: int,
        device,
        dtype,
    ) -> torch.Tensor:
        if not self.has_external_input:
            return torch.zeros(batch_size, time_steps, 0, device=device, dtype=dtype)
        if stim_window is None:
            return torch.zeros(
                batch_size, time_steps, self.input_dim, device=device, dtype=dtype
            )

        x = stim_window.float()
        if x.dim() == 2:
            x = x.unsqueeze(1).expand(-1, time_steps, -1)
        if x.dim() > 3:
            x = x.reshape(x.shape[0], x.shape[1], -1)

        flat_dim = x.shape[-1]
        if (
            flat_dim != self.input_dim
            and self.input_dim > 0
            and flat_dim % self.input_dim == 0
        ):
            x = x.reshape(
                x.shape[0], x.shape[1], flat_dim // self.input_dim, self.input_dim
            ).mean(dim=2)
        if x.shape[-1] != self.input_dim:
            x = F.adaptive_avg_pool1d(
                x.reshape(-1, 1, x.shape[-1]), self.input_dim
            ).reshape(x.shape[0], x.shape[1], self.input_dim)

        if x.shape[1] != time_steps:
            if x.shape[1] > time_steps:
                x = x[:, -time_steps:]
            else:
                pad = x[:, -1:].expand(-1, time_steps - x.shape[1], -1)
                x = torch.cat([x, pad], dim=1)
        return x.to(device=device, dtype=dtype)
