"""Field-compatible Koopman-Gramian model with a small U-Net-style encoder/decoder.

This model is intended for 2D field datasets such as The Well. It preserves
spatial structure before the encoder:

    state window: [B, W, C, H, W]
    latent:       [B, latent_dim] or [B, latent_channels, latent_channel_dim]
    decoded:      [B, C, H, W]

The Koopman / KG interface matches RawKoopmanGramianModel:
    encode_state, decode_state, stimulus_force, transition_latent, effective_A.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .koopman_gram import KoopmanLatentAdapter


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv = ConvBlock(in_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return self.conv(x)


class FieldUNetEncoder(nn.Module):
    """Encode a time window [B,W,C,H,W] into a vector latent."""

    def __init__(
        self,
        field_channels: int,
        window_size: int,
        latent_dim: int,
        base_channels: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.field_channels = int(field_channels)
        self.window_size = int(window_size)
        self.latent_dim = int(latent_dim)
        in_ch = self.window_size * self.field_channels
        c0 = int(base_channels)
        c1, c2, c3 = c0 * 2, c0 * 4, c0 * 8
        self.inc = ConvBlock(in_ch, c0, dropout=dropout)
        self.down1 = DownBlock(c0, c1, dropout=dropout)
        self.down2 = DownBlock(c1, c2, dropout=dropout)
        self.down3 = DownBlock(c2, c3, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, max(c3, min(2048, latent_dim))),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(max(c3, min(2048, latent_dim)), latent_dim),
        )

    def forward(self, state_window: torch.Tensor) -> torch.Tensor:
        if state_window.dim() != 5:
            raise ValueError(f"FieldUNetEncoder expects [B,W,C,H,W], got {tuple(state_window.shape)}")
        b, w, c, h, ww = state_window.shape
        if w != self.window_size or c != self.field_channels:
            raise ValueError(
                f"Expected state window [B,{self.window_size},{self.field_channels},H,W], "
                f"got {tuple(state_window.shape)}"
            )
        x = state_window.reshape(b, w * c, h, ww)
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return self.to_latent(self.pool(x))


class FieldUNetDecoder(nn.Module):
    """Decode latent vector into one field frame [B,C,H,W]."""

    def __init__(
        self,
        field_channels: int,
        latent_dim: int,
        base_channels: int = 32,
        seed_size: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.field_channels = int(field_channels)
        self.latent_dim = int(latent_dim)
        self.seed_size = int(seed_size)
        c0 = int(base_channels)
        c1, c2, c3 = c0 * 2, c0 * 4, c0 * 8
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, c3 * self.seed_size * self.seed_size),
            nn.GELU(),
        )
        self.up1 = UpBlock(c3, c2, dropout=dropout)
        self.up2 = UpBlock(c2, c1, dropout=dropout)
        self.up3 = UpBlock(c1, c0, dropout=dropout)
        self.out = nn.Conv2d(c0, self.field_channels, kernel_size=1)

    def forward(self, z: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        b = z.shape[0]
        c3 = self.up1.conv.net[0].in_channels
        x = self.from_latent(z).reshape(b, c3, self.seed_size, self.seed_size)
        h, w = int(output_size[0]), int(output_size[1])
        # Coarse-to-fine sizes. This keeps the code independent of exact H/W.
        s1 = (max(self.seed_size * 2, h // 4), max(self.seed_size * 2, w // 4))
        s2 = (max(self.seed_size * 4, h // 2), max(self.seed_size * 4, w // 2))
        x = self.up1(x, s1)
        x = self.up2(x, s2)
        x = self.up3(x, (h, w))
        return self.out(x)


class KoopmanFieldGramianModel(nn.Module):
    """2D field Koopman-Gramian model.

    This is a first field-compatible baseline. It uses a convolutional
    U-Net-style autoencoder and the same channelized Koopman transition as the
    vector model.
    """

    def __init__(
        self,
        stim_dim: int,
        fmri_dim: int = 0,  # kept for registry compatibility; not used as field dim
        window_size: int = 8,
        hidden_dim: int = 512,
        stim_depth: int = 0,
        stim_nhead: int = 4,
        dropout: float = 0.0,
        stable_linear: bool = True,
        spectral_bound: float = 1.02,
        residual_transition: bool = True,
        dt: float = 0.1,
        damping: float = 0.0,
        force_scale: float = 1.0,
        use_koopman_encoder: bool = True,
        koopman_latent_dim: Optional[int] = None,
        koopman_encoder_mid_dim: Optional[int] = None,
        koopman_encoder_residual: bool = True,  # kept for API compatibility
        koopman_latent_channels: Optional[int] = None,
        koopman_latent_channel_dim: Optional[int] = None,
        koopman_channel_shared_A: bool = True,
        koopman_use_latent_adapter: bool = False,
        koopman_adapter_dim: int = 256,
        koopman_adapter_alpha: float = 0.1,
        field_channels: int = 1,
        field_base_channels: int = 32,
        field_seed_size: int = 8,
    ):
        super().__init__()
        if not use_koopman_encoder:
            raise ValueError("KoopmanFieldGramianModel requires --use_koopman_encoder.")
        self.stim_dim = int(stim_dim)
        self.window_size = int(window_size)
        self.hidden_dim = int(hidden_dim)
        self.use_koopman_encoder = True
        self.stable_linear = bool(stable_linear)
        self.spectral_bound = float(spectral_bound)
        self.residual_transition = bool(residual_transition)
        self.dt = float(dt)
        self.damping = float(damping)
        self.koopman_channel_shared_A = bool(koopman_channel_shared_A)
        self.field_channels = int(field_channels)
        self.last_spatial_size: Optional[Tuple[int, int]] = None

        if koopman_latent_channels is not None or koopman_latent_channel_dim is not None:
            if koopman_latent_channels is None or koopman_latent_channel_dim is None:
                raise ValueError("Set both koopman_latent_channels and koopman_latent_channel_dim.")
            self.latent_channels = int(koopman_latent_channels)
            self.latent_channel_dim = int(koopman_latent_channel_dim)
            implied = self.latent_channels * self.latent_channel_dim
            if koopman_latent_dim is not None and int(koopman_latent_dim) != implied:
                raise ValueError(f"koopman_latent_dim={koopman_latent_dim} conflicts with channels*dim={implied}")
            self.latent_dim = implied
            self.channelized_koopman = True
        else:
            self.latent_dim = int(koopman_latent_dim or 1024)
            self.latent_channels = 1
            self.latent_channel_dim = self.latent_dim
            self.channelized_koopman = False
        self.state_dim = self.latent_channel_dim if self.channelized_koopman else self.latent_dim

        base_ch = int(koopman_encoder_mid_dim or field_base_channels)
        self.koopman_encoder = FieldUNetEncoder(
            field_channels=self.field_channels,
            window_size=self.window_size,
            latent_dim=self.latent_dim,
            base_channels=base_ch,
            dropout=dropout,
        )
        self.koopman_decoder = FieldUNetDecoder(
            field_channels=self.field_channels,
            latent_dim=self.latent_dim,
            base_channels=base_ch,
            seed_size=field_seed_size,
            dropout=dropout,
        )

        self.koopman_latent_adapter = None
        if koopman_use_latent_adapter:
            self.koopman_latent_adapter = KoopmanLatentAdapter(self.latent_dim, koopman_adapter_dim, koopman_adapter_alpha)

        if self.channelized_koopman and not self.koopman_channel_shared_A:
            self.A_delta = nn.Parameter(torch.zeros(self.latent_channels, self.latent_channel_dim, self.latent_channel_dim))
            nn.init.normal_(self.A_delta, mean=0.0, std=1e-4)
            self.register_buffer("I", torch.eye(self.latent_channel_dim).unsqueeze(0).repeat(self.latent_channels, 1, 1))
        else:
            self.A_delta = nn.Parameter(torch.zeros(self.state_dim, self.state_dim))
            nn.init.normal_(self.A_delta, mean=0.0, std=1e-4)
            self.register_buffer("I", torch.eye(self.state_dim))

        self.stim_in = nn.Linear(self.window_size * self.stim_dim, hidden_dim)
        self.stim_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if stim_depth > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=stim_nhead,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.stim_seq_in = nn.Linear(stim_dim, hidden_dim)
            self.stim_encoder = nn.TransformerEncoder(layer, num_layers=stim_depth)
            self.use_stim_transformer = True
        else:
            self.stim_seq_in = None
            self.stim_encoder = None
            self.use_stim_transformer = False

        self.force_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, self.latent_dim),
        )
        self.log_force_scale = nn.Parameter(torch.tensor(float(force_scale)).log())

    def _to_channel(self, z: torch.Tensor) -> torch.Tensor:
        if not self.channelized_koopman or z.dim() == 3:
            return z
        return z.reshape(z.shape[0], self.latent_channels, self.latent_channel_dim)

    def _to_flat(self, z: torch.Tensor) -> torch.Tensor:
        return z if z.dim() == 2 else z.reshape(z.shape[0], self.latent_dim)

    def effective_A(self) -> torch.Tensor:
        W = (1.0 - self.damping * self.dt) * self.I + self.dt * self.A_delta if self.residual_transition else self.A_delta
        if not self.stable_linear:
            return W
        sigma = torch.linalg.matrix_norm(W, ord=2).clamp_min(1e-6)
        scale = torch.clamp(self.spectral_bound / sigma, max=1.0)
        return W * scale.view(-1, 1, 1) if W.dim() == 3 else W * scale

    def encode_stim(self, stim_window: torch.Tensor) -> torch.Tensor:
        b, w, e = stim_window.shape
        if w != self.window_size:
            raise ValueError(f"Expected stim window length={self.window_size}, got {tuple(stim_window.shape)}")
        if self.use_stim_transformer:
            h = self.stim_seq_in(stim_window)
            h = self.stim_encoder(h)
            return h[:, -1]
        return self.stim_dropout(F.gelu(self.stim_in(stim_window.reshape(b, w * e))))

    def encode_state(self, state_window: torch.Tensor) -> torch.Tensor:
        if state_window.dim() != 5:
            raise ValueError(f"KoopmanFieldGramianModel expects state window [B,W,C,H,W], got {tuple(state_window.shape)}")
        self.last_spatial_size = (int(state_window.shape[-2]), int(state_window.shape[-1]))
        z = self.koopman_encoder(state_window)
        if self.koopman_latent_adapter is not None:
            z = self.koopman_latent_adapter(z)
        return self._to_channel(z) if self.channelized_koopman else z

    def decode_state(self, z: torch.Tensor) -> torch.Tensor:
        if self.last_spatial_size is None:
            raise RuntimeError("decode_state called before encode_state established the field spatial size.")
        return self.koopman_decoder(self._to_flat(z), self.last_spatial_size)

    def stimulus_force(self, stim_window: torch.Tensor) -> torch.Tensor:
        u = self.encode_stim(stim_window)
        force = torch.exp(self.log_force_scale) * self.force_head(u)
        if self.channelized_koopman:
            return force.reshape(force.shape[0], self.latent_channels, self.latent_channel_dim)
        return force

    def transition_latent(self, z_t: torch.Tensor, stim_window: torch.Tensor) -> torch.Tensor:
        A = self.effective_A()
        force = self.stimulus_force(stim_window)
        if self.channelized_koopman:
            zc = self._to_channel(z_t)
            if A.dim() == 3:
                return torch.einsum("bci,coi->bco", zc, A) + force
            return F.linear(zc, A, None) + force
        return F.linear(z_t, A, None) + force

    @torch.no_grad()
    def compute_latent_gramian(self, horizon: int = 100, normalize: bool = True) -> torch.Tensor:
        horizon = int(max(horizon, 1))
        A = self.effective_A().detach()
        if A.dim() == 3:
            c, d, _ = A.shape
            G = torch.zeros(c, d, d, device=A.device, dtype=A.dtype)
            Ak = torch.eye(d, device=A.device, dtype=A.dtype).unsqueeze(0).repeat(c, 1, 1)
            for _ in range(horizon):
                G.add_(Ak.transpose(-2, -1) @ Ak)
                Ak = A @ Ak
            if normalize:
                scale = torch.diagonal(G, dim1=-2, dim2=-1).sum(-1) / float(d)
                G = G / scale.clamp_min(1e-8).view(c, 1, 1)
            return G
        d = A.shape[0]
        G = torch.zeros(d, d, device=A.device, dtype=A.dtype)
        Ak = torch.eye(d, device=A.device, dtype=A.dtype)
        for _ in range(horizon):
            G.add_(Ak.transpose(0, 1) @ Ak)
            Ak = A @ Ak
        return G / (torch.trace(G) / float(d) + 1e-8) if normalize else G

    @torch.no_grad()
    def compute_gramian_powers(self, horizon: int = 100):
        horizon = int(max(horizon, 1))
        A = self.effective_A().detach()
        if A.dim() == 3:
            c, d, _ = A.shape
            powers = []
            Ak = torch.eye(d, device=A.device, dtype=A.dtype).unsqueeze(0).repeat(c, 1, 1)
            for _ in range(horizon):
                powers.append(Ak.clone())
                Ak = A @ Ak
            return powers
        d = A.shape[0]
        powers = []
        Ak = torch.eye(d, device=A.device, dtype=A.dtype)
        for _ in range(horizon):
            powers.append(Ak.clone())
            Ak = A @ Ak
        return powers

    def forward(self, stim_window: torch.Tensor, state_window: torch.Tensor, step_idx=None, return_aux: bool = False):
        z_t = self.encode_state(state_window)
        z_next = self.transition_latent(z_t, stim_window)
        pred_frame = self.decode_state(z_next)
        pred = pred_frame.unsqueeze(1)
        if not return_aux:
            return pred
        recon = self.decode_state(z_t).unsqueeze(1)
        A = self.effective_A()
        return pred, {
            "z": z_t,
            "z_next": z_next,
            "recon": recon,
            "A_sigma": torch.linalg.matrix_norm(A.detach(), ord=2).mean(),
            "force_scale": torch.exp(self.log_force_scale.detach()),
            "koopman_encoder": 1.0,
            "koopman_channelized": float(self.channelized_koopman),
            "koopman_channels": float(self.latent_channels),
            "koopman_channel_dim": float(self.latent_channel_dim),
            "field_channels": float(self.field_channels),
        }
