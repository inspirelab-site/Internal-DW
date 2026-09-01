"""Autoregressive U-Net baseline for 2D field sequences.

Interface matches FNOFieldModel:
    history [B, W, C, H, W] -> pred [B, 1, C, H, W]

This model is intentionally simple and deterministic: no BatchNorm, no dropout.
It is useful as a lower-variance AR baseline for The Well experiments.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dual_wiener import DualWienerController


class ConvGNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        groups = max(1, min(int(groups), int(out_channels)))
        while out_channels % groups != 0 and groups > 1:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        self.conv1 = ConvGNAct(in_channels, out_channels, groups=groups)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(max(1, min(int(groups), int(out_channels))), out_channels),
        )
        self.skip = (
            nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        *,
        resgrad_policy: str = "all",
        dual_wiener: Optional[DualWienerController] = None,
        route_horizon: int = -1,
        route_layer: int = -1,
        gain_collector: Optional[list] = None,
    ) -> torch.Tensor:
        if dual_wiener is not None and str(resgrad_policy).lower() not in (
            "all",
            "dualwiener",
        ):
            raise ValueError("Internal-DW blocks support all or dualwiener")
        if dual_wiener is None:
            skip_input = x
            branch_input = x
        else:
            skip_input, branch_input = dual_wiener.route_pair(
                x, int(route_horizon), int(route_layer)
            )
            if gain_collector is not None:
                gain_collector.append(
                    dual_wiener.current_pair(
                        int(route_horizon), int(route_layer), x
                    )
                )
        residual = self.skip(skip_input)
        branch = self.conv2(self.conv1(branch_input))
        return self.act(branch + residual)


class UNet2DCore(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        depth: int = 4,
        channel_mult: int = 2,
        groups: int = 8,
    ):
        super().__init__()
        self.depth = int(depth)
        base_channels = int(base_channels)
        channel_mult = int(channel_mult)

        chs = [base_channels * (channel_mult ** i) for i in range(self.depth)]
        self.in_block = ResidualBlock(in_channels, chs[0], groups=groups)

        self.down_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for i in range(self.depth - 1):
            self.downsample.append(nn.Conv2d(chs[i], chs[i], kernel_size=3, stride=2, padding=1))
            self.down_blocks.append(ResidualBlock(chs[i], chs[i + 1], groups=groups))

        self.mid = nn.ModuleList([
            ResidualBlock(chs[-1], chs[-1], groups=groups),
            ResidualBlock(chs[-1], chs[-1], groups=groups),
        ])

        self.up_blocks = nn.ModuleList()
        for i in reversed(range(self.depth - 1)):
            self.up_blocks.append(ResidualBlock(chs[i + 1] + chs[i], chs[i], groups=groups))

        self.out = nn.Sequential(
            ConvGNAct(chs[0], chs[0], groups=groups),
            nn.Conv2d(chs[0], out_channels, kernel_size=1),
        )
        self.num_residual_blocks = 2 * self.depth + 1

    def forward(
        self,
        x: torch.Tensor,
        *,
        dual_wiener: Optional[DualWienerController] = None,
        route_horizon: int = -1,
        gain_collector: Optional[list] = None,
    ) -> torch.Tensor:
        route_layer = 0

        def apply_block(
            block: ResidualBlock, value: torch.Tensor
        ) -> torch.Tensor:
            nonlocal route_layer
            output = block(
                value,
                dual_wiener=dual_wiener,
                route_horizon=route_horizon,
                route_layer=route_layer,
                gain_collector=gain_collector,
            )
            route_layer += 1
            return output

        skips = []
        x = apply_block(self.in_block, x)
        skips.append(x)
        for downsample, block in zip(
            self.downsample, self.down_blocks
        ):
            x = downsample(x)
            x = apply_block(block, x)
            skips.append(x)

        for block in self.mid:
            x = apply_block(block, x)

        for block, skip in zip(
            self.up_blocks, reversed(skips[:-1])
        ):
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            x = torch.cat([x, skip], dim=1)
            x = apply_block(block, x)
        return self.out(x)


class UNetFieldModel(nn.Module):
    """Autoregressive U-Net baseline for 2D field sequences."""

    is_standard_autoregressive = True
    task_type = "field2d"

    def __init__(
        self,
        field_channels: int,
        window_size: int = 4,
        base_channels: int = 64,
        depth: int = 4,
        channel_mult: int = 2,
        groups: int = 8,
        use_grid: bool = True,
        normalize: bool = True,
        resgrad_routing: bool = False,
        resgrad_policy: str = "all",
        field_height: int = 0,
        field_width: int = 0,
        dual_wiener_ema: float = 0.95,
        dual_wiener_residual_ema: float = 0.99,
        dual_wiener_warmup_batches: int = 8,
        dual_wiener_probe_every: int = 4,
        dual_wiener_min_probes: int = 1,
        dual_wiener_noise_model: str = "diagonal_gaussian",
        dual_wiener_max_horizon: int = 1024,
    ):
        super().__init__()
        self.field_channels = int(field_channels)
        self.window_size = int(window_size)
        self.base_channels = int(base_channels)
        self.depth = int(depth)
        self.channel_mult = int(channel_mult)
        self.groups = int(groups)
        self.use_grid = bool(use_grid)
        self.normalize = bool(normalize)
        self.resgrad_routing = bool(resgrad_routing)
        self.resgrad_policy = str(resgrad_policy).lower()
        if self.resgrad_routing and self.resgrad_policy != "dualwiener":
            raise ValueError(
                "The released router supports resgrad_policy='dualwiener' "
                "only; disable routing for full BPTT."
            )
        if not self.resgrad_routing:
            self.resgrad_policy = "all"
        self.resgrad_current_horizon = -1
        self.resgrad_current_total_horizon = -1
        self.field_height = int(field_height)
        self.field_width = int(field_width)

        in_ch = self.window_size * self.field_channels + (2 if self.use_grid else 0)
        self.core = UNet2DCore(
            in_channels=in_ch,
            out_channels=self.field_channels,
            base_channels=self.base_channels,
            depth=self.depth,
            channel_mult=self.channel_mult,
            groups=self.groups,
        )

        self.dual_wiener = None
        if self.resgrad_routing:
            if self.field_height <= 0 or self.field_width <= 0:
                raise ValueError(
                    "U-Net dualwiener routing requires positive field_height and field_width"
                )
            self.dual_wiener = DualWienerController(
                state_dim=self.field_channels * self.field_height * self.field_width,
                depth=self.core.num_residual_blocks,
                max_horizon=int(dual_wiener_max_horizon),
                ema=float(dual_wiener_ema),
                residual_ema=float(dual_wiener_residual_ema),
                warmup_batches=int(dual_wiener_warmup_batches),
                probe_every=int(dual_wiener_probe_every),
                min_probes=int(dual_wiener_min_probes),
                noise_model=str(dual_wiener_noise_model),
                spatial_shape=(
                    self.field_channels,
                    self.field_height,
                    self.field_width,
                ),
            )

        self.register_buffer("state_mean", torch.zeros(1, 1, self.field_channels, 1, 1))
        self.register_buffer("state_std", torch.ones(1, 1, self.field_channels, 1, 1))
        # This is an ephemeral derived tensor, not model state.  Keeping the
        # zero-length sentinel as a registered buffer makes DDP include it in
        # the initial coalesced module-state broadcast even though it is
        # non-persistent.  Some Blackwell/PyTorch/NCCL combinations treat that
        # empty CUDA buffer as an illegal-access launch.  A plain attribute is
        # sufficient: _grid() already rebuilds it whenever device, dtype, or
        # spatial shape changes.
        self._grid_cache = torch.empty(0)
        self.normalizer_fitted = False

    def set_state_normalizer(self, mean: torch.Tensor, std: torch.Tensor):
        mean = mean.detach().float().view(1, 1, self.field_channels, 1, 1)
        std = std.detach().float().clamp_min(1e-6).view(1, 1, self.field_channels, 1, 1)
        self.state_mean.copy_(mean.to(self.state_mean.device))
        self.state_std.copy_(std.to(self.state_std.device))
        self.normalizer_fitted = True

    def _grid(self, b: int, h: int, w: int, device, dtype) -> torch.Tensor:
        cache = self._grid_cache
        if (
            cache.numel() == 0
            or tuple(cache.shape) != (1, 2, h, w)
            or cache.device != device
            or cache.dtype != dtype
        ):
            gridx = (
                torch.linspace(0, 1, h, device=device, dtype=dtype)
                .view(1, 1, h, 1)
                .expand(1, 1, h, w)
            )
            gridy = (
                torch.linspace(0, 1, w, device=device, dtype=dtype)
                .view(1, 1, 1, w)
                .expand(1, 1, h, w)
            )
            self._grid_cache = torch.cat([gridx, gridy], dim=1)
            cache = self._grid_cache
        return cache.expand(b, -1, -1, -1)

    def _normalize_history(self, history: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return history
        return (history - self.state_mean) / self.state_std

    def _decode_output(self, frame: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return frame
        return frame * self.state_std[:, 0] + self.state_mean[:, 0]

    def forward_features(self, history: torch.Tensor) -> torch.Tensor:
        if history.dim() != 5:
            raise ValueError(f"UNetFieldModel expects history [B,W,C,H,W], got {tuple(history.shape)}")
        b, win, c, h, w = history.shape
        if win != self.window_size or c != self.field_channels:
            raise ValueError(
                f"Expected history [B,{self.window_size},{self.field_channels},H,W], got {tuple(history.shape)}"
            )
        history = self._normalize_history(history)
        x = history.reshape(b, win * c, h, w)
        if self.use_grid:
            x = torch.cat([x, self._grid(b, h, w, x.device, x.dtype)], dim=1)
        return x


    def set_resgrad_context(self, horizon_index: Optional[int] = None, total_horizon: Optional[int] = None):
        self.resgrad_current_horizon = -1 if horizon_index is None else int(horizon_index)
        self.resgrad_current_total_horizon = -1 if total_horizon is None else int(total_horizon)

    # ---- dual-Wiener training hooks -------------------------------------
    def dual_wiener_begin_batch(self) -> None:
        if self.dual_wiener is not None:
            self.dual_wiener.begin_batch()

    def dual_wiener_probe_terms(self, prediction, target, horizon_index):
        if self.dual_wiener is None or not self.training:
            return None, None
        return self.dual_wiener.probe_terms(prediction, target, horizon_index)

    def dual_wiener_set_probe_losses(self, total, noise) -> None:
        if self.dual_wiener is not None:
            self.dual_wiener.set_probe_losses(total, noise)

    def dual_wiener_calibrate(self) -> bool:
        return bool(self.dual_wiener is not None and self.dual_wiener.calibrate())

    def dual_wiener_end_batch(self) -> None:
        if self.dual_wiener is not None:
            self.dual_wiener.end_batch()

    def dual_wiener_diagnostics(self, horizon: int):
        if self.dual_wiener is None:
            return {}
        return self.dual_wiener.diagnostics(horizon)

    def dual_wiener_export_state(self, horizon: int):
        if self.dual_wiener is None:
            return None
        return self.dual_wiener.export_state(horizon)

    def predict_frame_from_history(
        self,
        history: torch.Tensor,
        gain_collector: Optional[list] = None,
    ) -> torch.Tensor:
        output = self.core(
            self.forward_features(history),
            dual_wiener=self.dual_wiener,
            route_horizon=int(self.resgrad_current_horizon),
            gain_collector=gain_collector,
        )
        return self._decode_output(output)

    def step_history(self, history: torch.Tensor) -> torch.Tensor:
        frame = self.predict_frame_from_history(history)
        return torch.cat([history[:, 1:], frame.unsqueeze(1)], dim=1)

    def forward(
        self,
        stim_window: Optional[torch.Tensor],
        history: torch.Tensor,
        return_aux: bool = False,
        horizon_index: Optional[int] = None,
        total_horizon: Optional[int] = None,
        ratio_collector: Optional[list] = None,
    ):
        if horizon_index is not None or total_horizon is not None:
            self.set_resgrad_context(
                horizon_index=horizon_index,
                total_horizon=total_horizon,
            )
        gains = [] if return_aux else None
        frame = self.predict_frame_from_history(
            history, gain_collector=gains
        )
        prediction = frame.unsqueeze(1)
        if not return_aux:
            return prediction

        one = frame.new_tensor(1.0)
        if gains:
            alphas = torch.stack([pair[0] for pair in gains])
            ms = torch.stack([pair[1] for pair in gains])
            alpha = alphas.mean()
            m = ms.mean()
        else:
            alpha = one
            m = one
        return prediction, {
            "pred_frame": frame,
            "resgrad_gate": m,
            "resgrad_routing": frame.new_tensor(
                1.0 if self.dual_wiener is not None else 0.0
            ),
            "dual_wiener_alpha": alpha,
            "dual_wiener_m": m,
        }
