"""FNO field baseline following the original FNO / neuraloperator implementation.

The model is still wrapped in the repository's autoregressive interface:
    history [B, W, C, H, W] -> pred [B, 1, C, H, W]

Backends:
  - backend="neuralop": use neuraloperator.models.FNO directly when installed.
  - backend="original": local implementation matching the original Li et al. FNO2d
    style: channel-last lifting/projection, rfft2 spectral convolutions, 1x1 skip
    convolutions, GELU, and spatial padding.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dual_wiener import DualWienerController


class SpectralConv2d(nn.Module):
    """Original FNO-style 2D spectral convolution.

    This follows the public FNO implementation pattern: rfft2, two complex
    weight tensors for positive/negative height frequencies, and irfft2 without
    orthonormal FFT normalization.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        scale = 1.0 / max(1, self.in_channels * self.out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def compl_mul2d(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # x: [B, in_ch, m1, m2], weights: [in_ch, out_ch, m1, m2]
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x)
        m1 = min(self.modes1, h)
        m2 = min(self.modes2, x_ft.shape[-1])
        out_ft = torch.zeros(
            b, self.out_channels, h, x_ft.shape[-1], device=x.device, dtype=torch.cfloat
        )
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(h, w))


class OriginalFNO2dCore(nn.Module):
    """Original Li et al. FNO2d-style core for fixed-resolution 2D fields."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 64,
        modes1: int = 16,
        modes2: int = 16,
        n_layers: int = 4,
        padding: int = 9,
        projection_channels: int = 128,
        residual_blocks: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.width = int(width)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.n_layers = int(n_layers)
        self.padding = int(padding)
        self.residual_blocks = bool(residual_blocks)
        self.capture_route_activations = False
        self.route_activations = []

        # Original FNO uses Linear layers on the channel-last grid representation.
        self.fc0 = nn.Linear(self.in_channels, self.width)
        self.convs = nn.ModuleList(
            [SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(self.n_layers)]
        )
        self.ws = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(self.n_layers)])
        self.fc1 = nn.Linear(self.width, int(projection_channels))
        self.fc2 = nn.Linear(int(projection_channels), self.out_channels)

    def forward(
        self,
        x: torch.Tensor,
        *,
        dual_wiener: Optional[DualWienerController] = None,
        route_horizon: int = -1,
    ) -> torch.Tensor:
        # x: [B, in_channels, H, W]
        h0, w0 = int(x.shape[-2]), int(x.shape[-1])
        x = x.permute(0, 2, 3, 1)  # [B,H,W,C]
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # [B,width,H,W]

        if self.padding > 0:
            x = F.pad(x, (0, self.padding, 0, self.padding))

        self.route_activations = []
        for i, (conv, w) in enumerate(zip(self.convs, self.ws)):
            if self.residual_blocks:
                if dual_wiener is not None:
                    skip_input, branch_input = dual_wiener.route_pair(
                        x, int(route_horizon), int(i)
                    )
                else:
                    skip_input = branch_input = x
                branch = conv(branch_input) + w(branch_input)
                if self.capture_route_activations:
                    self.route_activations.append(branch)
                x = skip_input + branch
            else:
                x = conv(x) + w(x)
            if i != len(self.convs) - 1:
                x = F.gelu(x)

        if self.padding > 0:
            x = x[..., :h0, :w0]

        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.permute(0, 3, 1, 2).contiguous()


class NeuralOperatorFNO2dCore(nn.Module):
    """Thin wrapper around neuraloperator.models.FNO."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 64,
        modes1: int = 16,
        modes2: int = 16,
        n_layers: int = 4,
        padding: int = 0,
        projection_channels: int = 128,
    ):
        super().__init__()
        try:
            from neuralop.models import FNO  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "--fno_backend neuralop requires the official neuraloperator package. "
                "Install it with: pip install neuraloperator"
            ) from exc

        # neuraloperator has changed the FNO constructor several times.
        # Build a superset of the official-style arguments, then filter it
        # against the installed package signature so older versions do not
        # crash on keywords such as `lifting_channels`.
        # Current neuraloperator API uses ratio arguments rather than
        # absolute lifting/projection channel counts:
        #   FNO(n_modes, in_channels, out_channels, hidden_channels,
        #       lifting_channel_ratio=2, projection_channel_ratio=2, ...)
        # Older releases may use different names. We provide a superset and
        # filter by the installed signature below.
        lift_proj_ratio = max(float(projection_channels) / max(float(width), 1.0), 1.0)
        candidate_kwargs = dict(
            n_modes=(int(modes1), int(modes2)),
            hidden_channels=int(width),
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            n_layers=int(n_layers),
            lifting_channel_ratio=lift_proj_ratio,
            projection_channel_ratio=lift_proj_ratio,
            # Older/unofficial variants may accept absolute channel counts.
            lifting_channels=int(projection_channels),
            projection_channels=int(projection_channels),
        )
        if int(padding) > 0:
            # Some neuraloperator versions use domain_padding; others do not.
            candidate_kwargs["domain_padding"] = float(padding) / 100.0 if int(padding) > 1 else float(padding)

        import inspect

        sig = inspect.signature(FNO.__init__)
        accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if accepts_var_kwargs:
            kwargs = dict(candidate_kwargs)
        else:
            allowed = set(sig.parameters.keys())
            kwargs = {k: v for k, v in candidate_kwargs.items() if k in allowed}

        # Fallbacks for older neuraloperator releases.
        # Some versions use `modes` instead of `n_modes`.
        if "n_modes" not in sig.parameters and "modes" in sig.parameters:
            kwargs.pop("n_modes", None)
            kwargs["modes"] = (int(modes1), int(modes2))

        # Some versions use `width` instead of `hidden_channels`.
        if "hidden_channels" not in sig.parameters and "width" in sig.parameters:
            kwargs.pop("hidden_channels", None)
            kwargs["width"] = int(width)

        try:
            self.fno = FNO(**kwargs)
        except TypeError as exc:
            # Last-resort compatibility: remove optional non-essential args.
            for optional_key in ("lifting_channels", "projection_channels", "domain_padding"):
                kwargs.pop(optional_key, None)
            try:
                self.fno = FNO(**kwargs)
            except TypeError as exc2:
                raise TypeError(
                    "Failed to construct neuraloperator.models.FNO with the installed API. "
                    f"Tried kwargs={sorted(kwargs.keys())}. Original error: {exc}"
                ) from exc2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fno(x)


class FoldedLatentPropagator(nn.Module):
    """Small latent propagation operator for training-only folded rollout loss.

    It uses a fixed downsampled field observable as the rollout state.  This is
    the model-agnostic abstraction of the linear Gramian idea: in the linear
    case, future errors propagate by A^k; here we learn a small latent A on a
    common observable space while keeping the expensive FNO graph one-step.
    """

    def __init__(self, latent_dim: int, init_scale: float = 0.98):
        super().__init__()
        self.latent_dim = int(latent_dim)
        eye = torch.eye(self.latent_dim)
        self.A = nn.Parameter(float(init_scale) * eye)
        self.bias = nn.Parameter(torch.zeros(self.latent_dim))

    def step(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.A.T + self.bias

    def spectral_sigma(self) -> torch.Tensor:
        with torch.no_grad():
            return torch.linalg.matrix_norm(self.A.detach(), ord=2)


class FNOFieldModel(nn.Module):
    """Autoregressive FNO baseline for 2D field sequences."""

    is_standard_autoregressive = True
    task_type = "field2d"

    def __init__(
        self,
        field_channels: int,
        window_size: int = 4,
        width: int = 64,
        modes1: int = 16,
        modes2: int = 16,
        n_layers: int = 4,
        padding: int = 9,
        hidden_channels: int = 128,
        use_grid: bool = True,
        backend: str = "neuralop",
        normalize: bool = True,
        folded_enabled: bool = False,
        folded_pool_size: int = 8,
        folded_init_scale: float = 0.98,
        residual_blocks: bool = False,
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
        self.width = int(width)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.n_layers = int(n_layers)
        self.padding = int(padding)
        self.use_grid = bool(use_grid)
        self.backend = str(backend)
        self.normalize = bool(normalize)
        self.folded_enabled = bool(folded_enabled)
        self.folded_pool_size = int(folded_pool_size)
        self.residual_blocks = bool(residual_blocks)
        self.resgrad_routing = bool(resgrad_routing)
        self.resgrad_policy = str(resgrad_policy).lower()
        self.resgrad_current_horizon = -1
        self.resgrad_current_total_horizon = -1
        self.field_height = int(field_height)
        self.field_width = int(field_width)
        if self.resgrad_policy == "dualwiener" and not self.residual_blocks:
            raise ValueError("FNO Internal-DW requires residual_blocks=True")
        self.folded_latent_dim = self.field_channels * self.folded_pool_size * self.folded_pool_size
        self.folded_propagator = (
            FoldedLatentPropagator(self.folded_latent_dim, init_scale=folded_init_scale)
            if self.folded_enabled else None
        )

        in_ch = self.window_size * self.field_channels + (2 if self.use_grid else 0)
        if self.backend == "neuralop":
            self.core = NeuralOperatorFNO2dCore(
                in_channels=in_ch,
                out_channels=self.field_channels,
                width=width,
                modes1=modes1,
                modes2=modes2,
                n_layers=n_layers,
                padding=padding,
                projection_channels=hidden_channels,
            )
        elif self.backend == "original":
            self.core = OriginalFNO2dCore(
                in_channels=in_ch,
                out_channels=self.field_channels,
                width=width,
                modes1=modes1,
                modes2=modes2,
                n_layers=n_layers,
                padding=padding,
                projection_channels=hidden_channels,
                residual_blocks=self.residual_blocks,
            )
        else:
            raise ValueError(f"Unknown fno backend: {backend}. Use 'neuralop' or 'original'.")

        self.register_buffer("state_mean", torch.zeros(1, 1, self.field_channels, 1, 1))
        self.register_buffer("state_std", torch.ones(1, 1, self.field_channels, 1, 1))
        self.normalizer_fitted = False
        self.dual_wiener = None
        if self.resgrad_routing and self.resgrad_policy == "dualwiener":
            if self.backend != "original":
                raise ValueError(
                    "Residual-FNO Internal-DW requires --fno_backend original; "
                    "the neuraloperator backend does not expose residual merges"
                )
            if self.field_height <= 0 or self.field_width <= 0:
                raise ValueError("Residual-FNO Internal-DW requires field_height/field_width")
            self.dual_wiener = DualWienerController(
                state_dim=self.field_channels * self.field_height * self.field_width,
                depth=self.n_layers,
                max_horizon=int(dual_wiener_max_horizon),
                ema=float(dual_wiener_ema),
                residual_ema=float(dual_wiener_residual_ema),
                warmup_batches=int(dual_wiener_warmup_batches),
                probe_every=int(dual_wiener_probe_every),
                min_probes=int(dual_wiener_min_probes),
                noise_model=str(dual_wiener_noise_model),
                spatial_shape=(self.field_channels, self.field_height, self.field_width),
            )

    def set_state_normalizer(self, mean: torch.Tensor, std: torch.Tensor):
        mean = mean.detach().float().view(1, 1, self.field_channels, 1, 1)
        std = std.detach().float().clamp_min(1e-6).view(1, 1, self.field_channels, 1, 1)
        self.state_mean.copy_(mean.to(self.state_mean.device))
        self.state_std.copy_(std.to(self.state_std.device))
        self.normalizer_fitted = True

    def _grid(self, b: int, h: int, w: int, device, dtype) -> torch.Tensor:
        # Same coordinate convention as the original FNO examples: x/y in [0,1].
        gridx = torch.linspace(0, 1, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        gridy = torch.linspace(0, 1, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([gridx, gridy], dim=1)

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
            raise ValueError(f"FNOFieldModel expects history [B,W,C,H,W], got {tuple(history.shape)}")
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

    def decode_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.backend == "original":
            frame = self.core(
                features,
                dual_wiener=self.dual_wiener,
                route_horizon=int(self.resgrad_current_horizon),
            )
        else:
            frame = self.core(features)
        return self._decode_output(frame)

    def set_resgrad_context(self, horizon_index=None, total_horizon=None):
        self.resgrad_current_horizon = -1 if horizon_index is None else int(horizon_index)
        self.resgrad_current_total_horizon = -1 if total_horizon is None else int(total_horizon)

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
        return {} if self.dual_wiener is None else self.dual_wiener.diagnostics(horizon)

    def dual_wiener_export_state(self, horizon: int):
        return None if self.dual_wiener is None else self.dual_wiener.export_state(horizon)

    def predict_frame_from_history(self, history: torch.Tensor) -> torch.Tensor:
        """Predict one next frame from a history tensor.

        This is the inference transition used by the AR rollout.  It is also
        used by the finite-time sensitivity loss, which differentiates the
        model output with respect to the input history instead of introducing
        a separate training-only transition adapter.
        """
        return self.decode_features(self.forward_features(history))

    def step_history(self, history: torch.Tensor) -> torch.Tensor:
        """One autoregressive transition on the full history state.

        Input and output both have shape [B, W, C, H, W].  The output shifts the
        history window left and appends the model-predicted next frame.  This
        map is the object whose state-to-state Jacobian is probed by the FTG
        loss; no explicit Jacobian matrix is formed.
        """
        frame = self.predict_frame_from_history(history)
        return torch.cat([history[:, 1:], frame.unsqueeze(1)], dim=1)

    def encode_folded_observable(self, frame: torch.Tensor) -> torch.Tensor:
        """Map a field frame [B,C,H,W] to a compact fixed observable vector.

        This intentionally avoids a trainable encoder in the first version, so
        the folded loss cannot be minimized by collapsing the latent space.
        """
        if frame.dim() != 4:
            raise ValueError(f"Expected frame [B,C,H,W], got {tuple(frame.shape)}")
        if self.normalize:
            frame = (frame.unsqueeze(1) - self.state_mean) / self.state_std
            frame = frame[:, 0]
        pooled = F.adaptive_avg_pool2d(frame, (self.folded_pool_size, self.folded_pool_size))
        return pooled.reshape(pooled.shape[0], -1)

    def folded_step(self, h: torch.Tensor) -> torch.Tensor:
        if self.folded_propagator is None:
            raise RuntimeError("FNO folded propagator is not enabled. Pass --fno_folded_loss.")
        return self.folded_propagator.step(h)

    def folded_A_sigma(self) -> float:
        if self.folded_propagator is None:
            return 0.0
        return float(self.folded_propagator.spectral_sigma().detach().cpu())

    def forward(self, stim_window: Optional[torch.Tensor], history: torch.Tensor, return_aux: bool = False):
        frame = self.predict_frame_from_history(history)
        pred = frame.unsqueeze(1)
        if return_aux:
            aux = {"pred_frame": frame}
            if self.folded_enabled:
                aux["folded_h_pred"] = self.encode_folded_observable(frame)
            return pred, aux
        return pred
