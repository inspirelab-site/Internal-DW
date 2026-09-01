"""Plug-in PyTorch API for backward-only Internal-DW routing."""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from torch import nn

from internal_dw.models.dual_wiener import DualWienerController


@dataclass(frozen=True)
class InternalDWConfig:
    """Configuration for :class:`InternalDW`.

    Parameters
    ----------
    state_dim:
        Size of the model prediction/state vector.  For a field model this is
        the flattened output size used by the innovation sampler.
    num_layers:
        Number of routed residual blocks per forecast step.
    max_horizon:
        Largest number of forecast steps that can be routed during training.
    ema:
        Exponential moving-average factor for route moments.
    residual_ema:
        EMA factor for residual statistics used by the default innovation
        sampler.
    warmup_batches:
        Number of batches used to initialize residual statistics before route
        probes begin.
    probe_every:
        Run the two read-only route VJP probes once every this many batches.
    min_probes:
        Minimum number of matched moment observations before a route is used.
    noise_model:
        Innovation sampler. ``"diagonal_gaussian"`` is the portable generic
        default. The research implementation additionally supports its
        structured prior samplers.
    spatial_shape:
        Optional ``(channels, height, width)`` shape for spatial samplers.
    """

    state_dim: int
    num_layers: int
    max_horizon: int
    ema: float = 0.95
    residual_ema: float = 0.99
    warmup_batches: int = 8
    probe_every: int = 4
    min_probes: int = 1
    noise_model: str = "diagonal_gaussian"
    spatial_shape: Tuple[int, int, int] | None = None


class InternalDW(nn.Module):
    """Backward-only dual Wiener router for arbitrary residual networks.

    The main operation is::

        skip_x, branch_x = dw.route(x, horizon=k, layer=l)
        y = skip_x + nonlinear_block(branch_x)

    ``skip_x`` and ``branch_x`` are *exactly equal to* ``x`` in the forward
    pass.  In the backward pass their vector-Jacobian products are multiplied
    by the current identity and nonlinear gains, respectively.  Consequently,
    the model architecture, predictions, and local parameter gradients inside
    ``nonlinear_block`` are unchanged; only credit sent farther upstream is
    routed.

    Input
    -----
    ``x`` is the tensor entering a residual merge, ``horizon`` is a zero-based
    forecast-step index, and ``layer`` is a zero-based residual-block index.

    Output
    ------
    A pair ``(skip_x, branch_x)`` with the same shape, dtype, device, and
    forward values as ``x``.
    """

    def __init__(
        self,
        state_dim: int,
        num_layers: int,
        max_horizon: int,
        *,
        ema: float = 0.95,
        residual_ema: float = 0.99,
        warmup_batches: int = 8,
        probe_every: int = 4,
        min_probes: int = 1,
        noise_model: str = "diagonal_gaussian",
        spatial_shape: Tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        config = InternalDWConfig(
            state_dim=int(state_dim),
            num_layers=int(num_layers),
            max_horizon=int(max_horizon),
            ema=float(ema),
            residual_ema=float(residual_ema),
            warmup_batches=int(warmup_batches),
            probe_every=int(probe_every),
            min_probes=int(min_probes),
            noise_model=str(noise_model),
            spatial_shape=spatial_shape,
        )
        if config.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if config.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if config.max_horizon <= 0:
            raise ValueError("max_horizon must be positive")
        self.config = config
        self.controller = DualWienerController(
            state_dim=config.state_dim,
            depth=config.num_layers,
            max_horizon=config.max_horizon,
            ema=config.ema,
            residual_ema=config.residual_ema,
            warmup_batches=config.warmup_batches,
            probe_every=config.probe_every,
            min_probes=config.min_probes,
            noise_model=config.noise_model,
            spatial_shape=config.spatial_shape,
        )

    @classmethod
    def from_config(cls, config: InternalDWConfig) -> "InternalDW":
        """Build a router from an immutable, serializable configuration."""

        return cls(
            state_dim=config.state_dim,
            num_layers=config.num_layers,
            max_horizon=config.max_horizon,
            ema=config.ema,
            residual_ema=config.residual_ema,
            warmup_batches=config.warmup_batches,
            probe_every=config.probe_every,
            min_probes=config.min_probes,
            noise_model=config.noise_model,
            spatial_shape=config.spatial_shape,
        )

    def forward(
        self, x: torch.Tensor, horizon: int, layer: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias for :meth:`route`, allowing the router to be called directly."""

        return self.route(x, horizon=horizon, layer=layer)

    def route(
        self, x: torch.Tensor, *, horizon: int, layer: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return identity- and nonlinear-route views of one residual stream."""

        return self.controller.route_pair(x, int(horizon), int(layer))

    def route_branch_state(
        self, state: torch.Tensor, *, horizon: int, layer: int
    ) -> torch.Tensor:
        """Route an optional recurrent state entering the nonlinear branch.

        This is needed only when a block carries hidden state outside the main
        residual tensor.  Feed the returned tensor to that branch in place of
        the original state.
        """

        return self.controller.branch_state_input(state, int(horizon), int(layer))

    def begin_batch(self) -> None:
        """Start one training batch and reset transient calibration state."""

        self.controller.begin_batch()

    def observe(
        self, prediction: torch.Tensor, target: torch.Tensor, *, horizon: int
    ) -> bool:
        """Record one horizon's residual and, on probe batches, its objectives.

        Parameters are prediction and target tensors with identical shape plus
        a zero-based horizon.  The return value is ``True`` when this batch is
        collecting a matched total/noise probe and ``False`` otherwise.
        """

        total, noise = self.controller.probe_terms(
            prediction, target, int(horizon)
        )
        self.controller.set_probe_losses(total, noise)
        return total is not None and noise is not None

    def calibrate(self) -> bool:
        """Run read-only VJP probes before the optimizer loss backward pass.

        Returns ``True`` when new route moments were measured and gains were
        staged.  This method does not write to model parameter ``.grad`` fields.
        """

        return self.controller.calibrate()

    def end_batch(self) -> None:
        """Commit staged gains after the real loss backward pass."""

        self.controller.end_batch()

    @property
    def gains(self) -> torch.Tensor:
        """Live ``[horizon, layer, 2]`` gain tensor (identity, nonlinear)."""

        return self.controller.coefficients

    def gain(self, *, horizon: int, layer: int) -> torch.Tensor:
        """Return a detached two-vector ``[alpha, m]`` for one route."""

        h = int(horizon)
        l = int(layer)
        if h < 0 or h >= self.config.max_horizon:
            raise IndexError("horizon is outside the configured gain table")
        if l < 0 or l >= self.config.num_layers:
            raise IndexError("layer is outside the configured gain table")
        return self.gains[h, l].detach().clone()

    def diagnostics(self, horizon: int | None = None) -> Dict[str, float]:
        """Return scalar, logger-ready diagnostics for routed horizons."""

        used = self.config.max_horizon if horizon is None else int(horizon)
        return self.controller.diagnostics(used)

    def export_state(self, horizon: int | None = None) -> Dict[str, Any]:
        """Return a JSON-safe audit record with gains and fitted moments."""

        used = self.config.max_horizon if horizon is None else int(horizon)
        return self.controller.export_state(used)


class InternalDWResidual(nn.Module):
    """Convenience wrapper for a standard ``x + branch(x)`` block.

    ``branch`` may accept additional positional or keyword inputs.  Its output
    must have the same shape as ``x`` so that the residual addition is valid.
    """

    def __init__(self, branch: nn.Module, router: InternalDW, layer: int) -> None:
        super().__init__()
        self.branch = branch
        # The parent model should own the one shared router.  Holding only a
        # weak reference here avoids registering that same module once per
        # residual block and duplicating its buffers in state_dict().
        self._router_ref = weakref.ref(router)
        self.layer = int(layer)

    @property
    def router(self) -> InternalDW:
        router = self._router_ref()
        if router is None:
            raise RuntimeError("the shared InternalDW router no longer exists")
        return router

    def forward(
        self, x: torch.Tensor, horizon: int, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        skip_x, branch_x = self.router.route(
            x, horizon=int(horizon), layer=self.layer
        )
        return skip_x + self.branch(branch_x, *args, **kwargs)
