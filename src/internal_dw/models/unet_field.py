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

from .fno_field import FoldedLatentPropagator
from .dual_wiener import DualWienerController
from .global_horizon_wiener import GlobalHorizonWienerController


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
        residual_grad_gate: float | torch.Tensor = 1.0,
        resgrad_policy: str = "all",
        resgrad_ratio_threshold: float = 0.05,
        ratio_collector: Optional[list] = None,
        gate_collector: Optional[list] = None,
        dual_wiener: Optional[DualWienerController] = None,
        route_horizon: int = -1,
        route_layer: int = -1,
        dual_alpha_collector: Optional[list] = None,
        dual_m_collector: Optional[list] = None,
    ) -> torch.Tensor:
        policy = str(resgrad_policy).lower()
        dual_pair = None
        if policy == "dualwiener":
            if dual_wiener is None:
                raise ValueError("resgrad_policy='dualwiener' requires a DualWienerController")
            skip_input, branch_input = dual_wiener.route_pair(
                x, int(route_horizon), int(route_layer)
            )
            # Coefficients are already applied by route_pair. Fetching them
            # again is needed only for optional diagnostics; doing so on every
            # block/horizon would force scalar GPU->CPU synchronizations below.
            if (
                gate_collector is not None
                or dual_alpha_collector is not None
                or dual_m_collector is not None
            ):
                dual_pair = dual_wiener.current_pair(
                    int(route_horizon), int(route_layer), x
                )
        else:
            skip_input = x
            branch_input = x

        residual = self.skip(skip_input)
        branch = self.conv2(self.conv1(branch_input))

        # Forward value is always skip(x) + branch(x).  Legacy policies scale
        # only the nonlinear input VJP.  Dual-Wiener independently scales the
        # skip and nonlinear input VJPs at the merge by (alpha, m).
        ratio_policies = (
            "ratio", "act_ratio", "dynamic", "dynamic_ratio",
            "branch_ratio", "norm_ratio",
        )
        branch_residual_ratio = None
        if ratio_collector is not None or policy in ratio_policies:
            branch_norm = (
                branch.detach().float().reshape(branch.shape[0], -1).norm(dim=1).mean()
            )
            residual_norm = (
                residual.detach().float().reshape(residual.shape[0], -1).norm(dim=1).mean()
            )
            branch_residual_ratio = branch_norm / (residual_norm + 1e-8)
            if ratio_collector is not None:
                ratio_collector.append(float(branch_residual_ratio.detach().cpu()))

        if policy in ratio_policies:
            base_gate = (
                float(residual_grad_gate)
                if not torch.is_tensor(residual_grad_gate)
                else float(residual_grad_gate.detach().float().mean().cpu())
            )
            assert branch_residual_ratio is not None
            gate_val = 1.0 if float(branch_residual_ratio.detach().cpu()) >= float(resgrad_ratio_threshold) else base_gate
        else:
            gate_val = residual_grad_gate

        if dual_pair is not None:
            alpha_value = float(dual_pair[0].detach().float().cpu())
            m_value = float(dual_pair[1].detach().float().cpu())
            if dual_alpha_collector is not None:
                dual_alpha_collector.append(alpha_value)
            if dual_m_collector is not None:
                dual_m_collector.append(m_value)
            # Preserve the legacy one-number gate diagnostic as the nonlinear
            # route coefficient.  Dedicated alpha/m diagnostics are returned
            # separately by UNetFieldModel.
            gate_val = m_value

        if gate_collector is not None:
            gate_collector.append(
                float(gate_val) if not torch.is_tensor(gate_val) else float(gate_val.detach().float().mean().cpu())
            )

        if policy == "dualwiener":
            # Both route coefficients were applied at the block input by the
            # controller.  Applying the legacy branch stop-gradient here would
            # multiply m twice.
            pass
        elif torch.is_tensor(gate_val):
            gate_tensor = gate_val.to(device=branch.device, dtype=branch.dtype)
            branch = branch.detach() + gate_tensor * (branch - branch.detach())
        else:
            gate_float = float(gate_val)
            if gate_float <= 0.0:
                branch = branch.detach()
            elif gate_float < 1.0:
                branch = branch.detach() + gate_float * (branch - branch.detach())
            # gate >= 1: ordinary autograd, no change.
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
        residual_grad_gate: float | torch.Tensor = 1.0,
        resgrad_policy: str = "all",
        resgrad_ratio_threshold: float = 0.05,
        ratio_collector: Optional[list] = None,
        gate_collector: Optional[list] = None,
        dual_wiener: Optional[DualWienerController] = None,
        route_horizon: int = -1,
        dual_alpha_collector: Optional[list] = None,
        dual_m_collector: Optional[list] = None,
    ) -> torch.Tensor:
        route_layer = 0

        def blk(block: ResidualBlock, inp: torch.Tensor) -> torch.Tensor:
            nonlocal route_layer
            out = block(
                inp,
                residual_grad_gate=residual_grad_gate,
                resgrad_policy=resgrad_policy,
                resgrad_ratio_threshold=resgrad_ratio_threshold,
                ratio_collector=ratio_collector,
                gate_collector=gate_collector,
                dual_wiener=dual_wiener,
                route_horizon=route_horizon,
                route_layer=route_layer,
                dual_alpha_collector=dual_alpha_collector,
                dual_m_collector=dual_m_collector,
            )
            route_layer += 1
            return out

        skips = []
        x = blk(self.in_block, x)
        skips.append(x)
        for down, block in zip(self.downsample, self.down_blocks):
            x = down(x)
            x = blk(block, x)
            skips.append(x)

        for block in self.mid:
            x = blk(block, x)

        # Drop the bottleneck skip; use the remaining encoder skips from deep to shallow.
        for block, skip in zip(self.up_blocks, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = blk(block, x)
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
        folded_enabled: bool = False,
        folded_pool_size: int = 8,
        folded_init_scale: float = 0.98,
        resgrad_routing: bool = False,
        resgrad_policy: str = "all",
        resgrad_block_gate: float = 1.0,
        resgrad_ratio_threshold: float = 0.05,
        resgrad_keep_every: int = 8,
        resgrad_keep_tail: int = 0,
        field_height: int = 0,
        field_width: int = 0,
        dual_wiener_ema: float = 0.95,
        dual_wiener_residual_ema: float = 0.99,
        dual_wiener_warmup_batches: int = 8,
        dual_wiener_probe_every: int = 4,
        dual_wiener_min_probes: int = 1,
        dual_wiener_noise_model: str = "diagonal_gaussian",
        dual_wiener_max_horizon: int = 1024,
        global_horizon_wiener: bool = False,
        global_wiener_ridge: float = 1e-8,
        global_wiener_anchor: float = 0.0,
        global_wiener_local_fidelity: float = 0.0,
        global_wiener_solver_iters: int = 256,
        global_wiener_sketch_dim: int = 8192,
        global_wiener_sketch_seed: int = 1729,
        global_wiener_noise_draws: int = 4,
        global_wiener_batch_conditioned: bool = True,
        global_wiener_superbatch_groups: int = 1,
        global_wiener_static_gain: float = -1.0,
        global_wiener_static_mode: str = "delayed_tied",
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
        self.folded_enabled = bool(folded_enabled)
        self.folded_pool_size = int(folded_pool_size)
        self.resgrad_routing = bool(resgrad_routing)
        self.resgrad_policy = str(resgrad_policy).lower()
        self.resgrad_block_gate = float(resgrad_block_gate)
        self.resgrad_ratio_threshold = float(resgrad_ratio_threshold)
        self.resgrad_keep_every = max(1, int(resgrad_keep_every))
        self.resgrad_keep_tail = max(0, int(resgrad_keep_tail))
        self.resgrad_current_horizon = -1
        self.resgrad_current_total_horizon = -1
        self.field_height = int(field_height)
        self.field_width = int(field_width)
        self.folded_latent_dim = self.field_channels * self.folded_pool_size * self.folded_pool_size
        self.folded_propagator = (
            FoldedLatentPropagator(self.folded_latent_dim, init_scale=folded_init_scale)
            if self.folded_enabled else None
        )

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
        if self.resgrad_routing and self.resgrad_policy == "dualwiener":
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
        self.global_horizon_wiener = None
        if bool(global_horizon_wiener):
            if self.dual_wiener is not None:
                raise ValueError(
                    "global-horizon Wiener and internal dualwiener routing are "
                    "alternative backward operators; enable only one"
                )
            if self.field_height <= 0 or self.field_width <= 0:
                raise ValueError(
                    "U-Net global-horizon Wiener requires positive field dimensions"
                )
            self.global_horizon_wiener = GlobalHorizonWienerController(
                state_dim=self.field_channels * self.field_height * self.field_width,
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
                ridge=float(global_wiener_ridge),
                anchor=float(global_wiener_anchor),
                local_fidelity=float(global_wiener_local_fidelity),
                solver_iterations=int(global_wiener_solver_iters),
                sketch_dim=int(global_wiener_sketch_dim),
                sketch_seed=int(global_wiener_sketch_seed),
                noise_draws=int(global_wiener_noise_draws),
                batch_conditioned=bool(global_wiener_batch_conditioned),
                superbatch_groups=int(global_wiener_superbatch_groups),
                static_gain=float(global_wiener_static_gain),
                static_mode=str(global_wiener_static_mode),
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

    def _resgrad_gate_for_step(self, horizon_index: Optional[int] = None) -> float:
        """Return the nonlinear-branch gradient gate for this rollout step.

        Mirrors OfficialStateMambaARModel._resgrad_gate_for_step: the same
        gate value is applied to every ResidualBlock in the U-Net stack for
        a given closed-loop rollout step k. Forward values are unchanged
        regardless of policy; only the backward graph through each block's
        nonlinear branch is affected.
        """
        if not self.resgrad_routing:
            return 1.0
        pol = str(self.resgrad_policy).lower()
        k = int(horizon_index) if horizon_index is not None else int(getattr(self, "resgrad_current_horizon", -1))
        K = int(getattr(self, "resgrad_current_total_horizon", -1))
        base_gate = float(self.resgrad_block_gate)
        if pol in ("all", "full", "normal"):
            return 1.0
        if pol in ("none", "identity", "id"):
            return 0.0
        if pol in ("fixed", "scalar"):
            return base_gate
        if pol in ("periodic", "every"):
            if k < 0:
                return base_gate
            return 1.0 if (k % self.resgrad_keep_every == 0) else base_gate
        if pol in ("tail", "last"):
            if k < 0 or K <= 0:
                return base_gate
            return 1.0 if k >= max(0, K - self.resgrad_keep_tail) else base_gate
        if pol in ("head", "first"):
            if k < 0:
                return base_gate
            return 1.0 if k < self.resgrad_keep_tail else base_gate
        return base_gate

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

    def global_wiener_begin_batch(self) -> None:
        if self.global_horizon_wiener is not None:
            self.global_horizon_wiener.begin_batch()

    def global_wiener_observe_and_sample(self, prediction, target, horizon_index):
        if self.global_horizon_wiener is None or not self.training:
            return None
        return self.global_horizon_wiener.observe_and_sample(
            horizon_index, prediction, target
        )

    def global_wiener_add_probe_terms(self, horizon_index, total, noise) -> None:
        if self.global_horizon_wiener is not None:
            self.global_horizon_wiener.add_probe_terms(
                horizon_index, total, noise
            )

    def global_wiener_weight_loss(self, loss, horizon_index):
        if self.global_horizon_wiener is None:
            return loss
        return self.global_horizon_wiener.weight_loss(loss, horizon_index)

    def global_wiener_calibrate(self, parameters) -> bool:
        return bool(
            self.global_horizon_wiener is not None
            and self.global_horizon_wiener.calibrate(parameters)
        )

    def global_wiener_end_batch(self) -> None:
        if self.global_horizon_wiener is not None:
            self.global_horizon_wiener.end_batch()

    def global_wiener_diagnostics(self, horizon: int):
        if self.global_horizon_wiener is None:
            return {}
        return self.global_horizon_wiener.diagnostics(horizon)

    def global_wiener_export_state(self, horizon: int):
        if self.global_horizon_wiener is None:
            return None
        return self.global_horizon_wiener.export_state(horizon)

    def predict_frame_from_history(
        self,
        history: torch.Tensor,
        ratio_collector: Optional[list] = None,
        gate_collector: Optional[list] = None,
        dual_alpha_collector: Optional[list] = None,
        dual_m_collector: Optional[list] = None,
    ) -> torch.Tensor:
        gate = self._resgrad_gate_for_step()
        # dynamic_ratio decides its own per-block gate from branch/residual
        # norms; other policies use this pre-computed scalar/tensor directly.
        out = self.core(
            self.forward_features(history),
            residual_grad_gate=gate,
            resgrad_policy=self.resgrad_policy,
            resgrad_ratio_threshold=self.resgrad_ratio_threshold,
            ratio_collector=ratio_collector,
            gate_collector=gate_collector,
            dual_wiener=self.dual_wiener,
            route_horizon=int(self.resgrad_current_horizon),
            dual_alpha_collector=dual_alpha_collector,
            dual_m_collector=dual_m_collector,
        )
        return self._decode_output(out)

    def step_history(self, history: torch.Tensor) -> torch.Tensor:
        frame = self.predict_frame_from_history(history)
        return torch.cat([history[:, 1:], frame.unsqueeze(1)], dim=1)

    def encode_folded_observable(self, frame: torch.Tensor) -> torch.Tensor:
        if frame.dim() != 4:
            raise ValueError(f"Expected frame [B,C,H,W], got {tuple(frame.shape)}")
        if self.normalize:
            frame = (frame.unsqueeze(1) - self.state_mean) / self.state_std
            frame = frame[:, 0]
        pooled = F.adaptive_avg_pool2d(frame, (self.folded_pool_size, self.folded_pool_size))
        return pooled.reshape(pooled.shape[0], -1)

    def folded_step(self, h: torch.Tensor) -> torch.Tensor:
        if self.folded_propagator is None:
            raise RuntimeError("U-Net folded propagator is not enabled. Pass --fno_folded_loss.")
        return self.folded_propagator.step(h)

    def folded_A_sigma(self) -> float:
        if self.folded_propagator is None:
            return 0.0
        return float(self.folded_propagator.spectral_sigma().detach().cpu())

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
            self.set_resgrad_context(horizon_index=horizon_index, total_horizon=total_horizon)
        collector = ratio_collector
        gate_collector = [] if return_aux else None
        dual_alpha_collector = [] if return_aux else None
        dual_m_collector = [] if return_aux else None
        if return_aux and collector is None:
            collector = []
        frame = self.predict_frame_from_history(
            history,
            ratio_collector=collector,
            gate_collector=gate_collector,
            dual_alpha_collector=dual_alpha_collector,
            dual_m_collector=dual_m_collector,
        )
        pred = frame.unsqueeze(1)
        if return_aux:
            aux = {"pred_frame": frame}
            if self.folded_enabled:
                aux["folded_h_pred"] = self.encode_folded_observable(frame)
            # Aggregate the per-block resolved gate (correct even for
            # dynamic_ratio, where each block independently decides its own
            # gate from its branch/residual ratio) rather than the
            # policy-level scalar from _resgrad_gate_for_step, which for
            # dynamic_ratio is only the fallback base_gate.
            aux["resgrad_gate"] = (
                frame.new_tensor(float(sum(gate_collector) / len(gate_collector)))
                if gate_collector else frame.new_tensor(float(self._resgrad_gate_for_step(horizon_index)))
            )
            aux["resgrad_routing"] = frame.new_tensor(1.0 if self.resgrad_routing else 0.0)
            if collector:
                aux["resgrad_branch_residual_ratio"] = frame.new_tensor(float(sum(collector) / len(collector)))
            if dual_alpha_collector:
                aux["dual_wiener_alpha"] = frame.new_tensor(
                    float(sum(dual_alpha_collector) / len(dual_alpha_collector))
                )
            if dual_m_collector:
                aux["dual_wiener_m"] = frame.new_tensor(
                    float(sum(dual_m_collector) / len(dual_m_collector))
                )
            return pred, aux
        return pred
