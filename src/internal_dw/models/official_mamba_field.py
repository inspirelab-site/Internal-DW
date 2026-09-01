"""Field-input variant of the state Mamba autoregressive model.

Purpose
-------
On a gridded field the flattened-vector model has to learn a dense
``Linear(C*H*W -> hidden)``, which carries no locality prior; on WeatherBench that
run reached only ACC ~0.44 on Z500 at day 3.  This variant keeps the temporal core
untouched and replaces only the two projections that touch the state:

    in_proj : [B, C*H*W (+ stim)] -> conv encoder -> [B, hidden]
    out     : [B, hidden]         -> conv decoder -> [B, C*H*W]

Everything between them --- the Mamba block stack, the residual merges, the route
VJP hooks, the Dual-Wiener controller and the box solve --- is inherited unchanged
from :class:`OfficialStateMambaARModel`.  That file is not modified, so the vector
experiments (fMRI, iEEG, Mackey-Glass, NARMA) are unaffected by anything here.

Design constraints that matter for the routing experiment
---------------------------------------------------------
1. **The encoder and decoder are strictly feed-forward: no residual merges.**
   A U-Net-style encoder would introduce *spatial* ``y = x + F(x)`` merges, and the
   routing hooks would then weight spatial refinement rather than temporal credit.
   Keeping them plain guarantees that every routed merge lives in the Mamba stack
   and is therefore temporal.
2. ``state_dim`` is unchanged, so the output space, the innovation covariance and
   the latitude-weighted ACC evaluation all operate on exactly the same quantity as
   before.  The field shape is used only inside these two modules.
3. Longitude is periodic and latitude is not, so the convolutions pad circularly
   along W and by replication along H.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .official_state_mamba import OfficialStateMambaARModel


class _GeoPad(nn.Module):
    """Pad circularly in longitude (W) and by replication in latitude (H)."""

    def __init__(self, pad: int = 1) -> None:
        super().__init__()
        self.pad = int(pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.pad
        if p <= 0:
            return x
        x = F.pad(x, (p, p, 0, 0), mode="circular")
        return F.pad(x, (0, 0, p, p), mode="replicate")


def _conv_block(cin: int, cout: int, stride: int, groups: int) -> nn.Sequential:
    return nn.Sequential(
        _GeoPad(1),
        nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=0),
        nn.GroupNorm(min(groups, cout), cout),
        nn.GELU(),
    )


class _FieldEncoder(nn.Module):
    """[B, C*H*W (+ stim)] -> [B, hidden].

    The state part is reshaped to the grid and passed through two stride-2
    convolutions (H,W -> H/4, W/4); the stimulus, if any, is appended after the
    spatial reduction so it is never convolved with the field.
    """

    def __init__(self, state_shape: Sequence[int], stim_dim: int, hidden_dim: int,
                 base_ch: int = 64, groups: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        C, H, W = (int(v) for v in state_shape)
        self.state_shape = (C, H, W)
        self.state_dim = C * H * W
        self.stim_dim = int(stim_dim)
        self.stem = _conv_block(C, base_ch, stride=1, groups=groups)
        self.down1 = _conv_block(base_ch, base_ch, stride=2, groups=groups)
        self.down2 = _conv_block(base_ch, 2 * base_ch, stride=2, groups=groups)
        self.out_h, self.out_w = H // 4, W // 4
        flat = 2 * base_ch * self.out_h * self.out_w
        self.proj = nn.Sequential(
            nn.LayerNorm(flat + self.stim_dim),
            nn.Linear(flat + self.stim_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        B = inp.shape[0]
        field = inp[:, : self.state_dim].reshape(B, *self.state_shape)
        z = self.down2(self.down1(self.stem(field))).reshape(B, -1)
        if self.stim_dim > 0:
            z = torch.cat([z, inp[:, self.state_dim:]], dim=-1)
        return self.proj(z)


class _FieldDecoder(nn.Module):
    """[B, hidden] -> [B, C*H*W], mirroring the encoder."""

    def __init__(self, state_shape: Sequence[int], hidden_dim: int,
                 base_ch: int = 64, groups: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        C, H, W = (int(v) for v in state_shape)
        self.state_shape = (C, H, W)
        self.ch = 2 * base_ch
        self.h, self.w = H // 4, W // 4
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.ch * self.h * self.w),
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            _conv_block(self.ch, base_ch, stride=1, groups=groups),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            _conv_block(base_ch, base_ch, stride=1, groups=groups),
        )
        self.head = nn.Sequential(_GeoPad(1), nn.Conv2d(base_ch, C, kernel_size=3, padding=0))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        y = self.proj(z).reshape(B, self.ch, self.h, self.w)
        y = self.head(self.up2(self.up1(y)))
        return y.reshape(B, -1)


class OfficialFieldMambaARModel(OfficialStateMambaARModel):
    """State Mamba with convolutional field projections.

    Requires ``state_shape=(C, H, W)``.  All routing arguments are inherited and
    behave identically; only ``in_proj`` and ``out`` differ from the parent.
    """

    def __init__(self, *args, field_base_ch: int = 64, field_groups: int = 8, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.state_shape is None or len(self.state_shape) != 3:
            raise ValueError(
                "official_mamba_field requires a 3-D state_shape (C, H, W); got "
                f"{self.state_shape!r}. Use official_mamba_state for vector states."
            )
        C, H, W = (int(v) for v in self.state_shape)
        if H % 4 or W % 4:
            raise ValueError(f"field height and width must be divisible by 4; got {H}x{W}")
        if C * H * W != self.state_dim:
            raise ValueError(f"state_shape {self.state_shape} does not match state_dim {self.state_dim}")

        stim_dim = self.input_dim if self.has_external_input else 0
        dropout = 0.0
        for m in self.in_proj.modules():
            if isinstance(m, nn.Dropout):
                dropout = float(m.p)
                break

        # Replace only the two state-facing projections.  The parent's step() calls
        # self.in_proj(concat) and self.out(z) with exactly these shapes.
        self.in_proj = _FieldEncoder(self.state_shape, stim_dim, self.hidden_dim,
                                     base_ch=int(field_base_ch), groups=int(field_groups),
                                     dropout=dropout)
        self.out = _FieldDecoder(self.state_shape, self.hidden_dim,
                                 base_ch=int(field_base_ch), groups=int(field_groups),
                                 dropout=dropout)
        if self.predict_sigma:
            self.sigma_out = _FieldDecoder(self.state_shape, self.hidden_dim,
                                           base_ch=int(field_base_ch), groups=int(field_groups),
                                           dropout=dropout)
