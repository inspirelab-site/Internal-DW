"""Gradient primitives used by the known-SNR failure-profile experiment.

The functions here are deliberately limited to the quantities consumed by
Figure 4 panels 1--2: simulator futures, frozen-model rollouts, per-horizon
parameter gradients, and temporary control of the residual backward routes.
They do not fit alternative routing policies or optimizer-space oracles.
"""

from __future__ import annotations

import contextlib
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from probe_setup import burn_in_root


TensorList = List[torch.Tensor]


def load_process(oracle_file: str, normalization_std: float, device: str):
    """Load diagonal-AR coefficients and normalized innovation scales."""

    with np.load(oracle_file, allow_pickle=True) as archive:
        coefficients = np.asarray(
            archive["oracle_ar_coefficients"], dtype=np.float64
        )
        innovation_std = np.asarray(
            archive["oracle_one_step_innovation_std"], dtype=np.float64
        )
    return (
        torch.as_tensor(coefficients, device=device, dtype=torch.float32),
        torch.as_tensor(
            innovation_std / float(normalization_std),
            device=device,
            dtype=torch.float32,
        ),
    )


def conditional_and_noisy_targets(
    bundle,
    starts: Sequence[int],
    coefficients: torch.Tensor,
    innovation_std_normalized: torch.Tensor,
    normalization_mean: float,
    normalization_std: float,
    horizon: int,
    noise_draws: int,
    seed: int,
) -> Tuple[TensorList, List[TensorList]]:
    """Generate a conditional mean and independent futures from each state."""

    starts_t = torch.as_tensor(starts, device=bundle.xt.device, dtype=torch.long)
    normalized_initial = bundle.xt[bundle.rows_t, starts_t]
    raw_initial = normalized_initial * float(normalization_std) + float(
        normalization_mean
    )
    raw_coefficients = coefficients.to(raw_initial)

    conditional: TensorList = []
    raw_mean = raw_initial
    for _ in range(int(horizon)):
        raw_mean = raw_mean * raw_coefficients
        conditional.append(
            ((raw_mean - float(normalization_mean)) / float(normalization_std)).detach()
        )

    generator = torch.Generator(device=bundle.xt.device.type).manual_seed(int(seed))
    normalized_noise_scale = innovation_std_normalized.to(normalized_initial)
    offset = (
        raw_coefficients * float(normalization_mean) - float(normalization_mean)
    ) / float(normalization_std)
    noisy: List[TensorList] = []
    for _ in range(int(noise_draws)):
        state = normalized_initial
        draw: TensorList = []
        for _horizon in range(int(horizon)):
            innovation = torch.randn(
                state.shape,
                generator=generator,
                device=state.device,
                dtype=state.dtype,
            ) * normalized_noise_scale
            state = state * raw_coefficients + offset + innovation
            draw.append(state.detach())
        noisy.append(draw)
    return conditional, noisy


def rollout_predictions(bundle, starts: Sequence[int], args) -> TensorList:
    """Roll out the frozen forecasting model from the requested starts."""

    model, ut, rows_t = bundle.model, bundle.ut, bundle.rows_t
    current, hidden, _roots, _labels, starts_t = burn_in_root(bundle, starts, args)
    predictions: TensorList = []
    value = current
    with torch.enable_grad():
        for horizon in range(int(args.K)):
            stimulus = ut[rows_t, starts_t + horizon] if ut is not None else None
            output = model.step(
                hidden,
                value,
                stim_t=stimulus,
                horizon_index=horizon,
                total_horizon=int(args.K),
            )
            prediction, hidden = output[0], output[1]
            predictions.append(prediction)
            value = prediction
    return predictions


def horizon_losses(
    predictions: Sequence[torch.Tensor], targets: Sequence[torch.Tensor]
) -> List[torch.Tensor]:
    """Return the equally weighted contribution from each forecast step."""

    scale = 1.0 / max(len(predictions), 1)
    return [
        scale * (prediction - target).square().mean()
        for prediction, target in zip(predictions, targets)
    ]


def trainable_parameters(model) -> Tuple[List[str], List[torch.Tensor]]:
    pairs = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not pairs:
        raise RuntimeError("model has no trainable parameters")
    return [name for name, _ in pairs], [parameter for _, parameter in pairs]


def _gradient_vector(
    loss: torch.Tensor,
    parameters: Sequence[torch.Tensor],
    retain_graph: bool,
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    pieces = []
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            pieces.append(torch.zeros_like(parameter).reshape(-1))
        else:
            pieces.append(gradient.detach().reshape(-1))
    return torch.cat(pieces).float()


def gradient_matrix(
    losses: Sequence[torch.Tensor],
    parameters: Sequence[torch.Tensor],
    retain_after: bool,
) -> torch.Tensor:
    """Stack one flattened parameter-gradient vector per horizon."""

    use_batched = os.environ.get("GLOBAL_ORACLE_BATCHED_VJP", "1").strip().lower()
    use_batched = use_batched not in ("0", "false", "no", "off")
    if use_batched:
        loss_vector = torch.stack(list(losses))
        basis = torch.eye(
            loss_vector.numel(), device=loss_vector.device, dtype=loss_vector.dtype
        )
        try:
            gradients = torch.autograd.grad(
                loss_vector,
                parameters,
                grad_outputs=basis,
                retain_graph=retain_after,
                allow_unused=True,
                is_grads_batched=True,
            )
            pieces = []
            for parameter, gradient in zip(parameters, gradients):
                if gradient is None:
                    pieces.append(
                        torch.zeros(
                            (loss_vector.numel(), parameter.numel()),
                            device=parameter.device,
                            dtype=parameter.dtype,
                        )
                    )
                else:
                    pieces.append(gradient.detach().reshape(loss_vector.numel(), -1))
            return torch.cat(pieces, dim=1).float()
        except (RuntimeError, NotImplementedError) as error:
            if not getattr(gradient_matrix, "_fallback_announced", False):
                print(
                    "[known-SNR] batched VJP unavailable; falling back to "
                    f"sequential horizon VJPs: {type(error).__name__}: {error}",
                    flush=True,
                )
                gradient_matrix._fallback_announced = True

    rows = []
    for index, loss in enumerate(losses):
        keep = retain_after or index + 1 < len(losses)
        rows.append(_gradient_vector(loss, parameters, retain_graph=keep))
    return torch.stack(rows, dim=0)


@contextlib.contextmanager
def routing_operator(controller, coefficients: torch.Tensor):
    """Temporarily install fixed route gains and restore all controller state."""

    saved_coefficients = controller.coefficients.detach().clone()
    saved_const_gain = controller.const_gain
    saved_mode = controller._mode
    saved_collecting = controller._collecting
    saved_period = os.environ.get("RESGRAD_ALPHA_PERIOD")
    saved_value = os.environ.get("RESGRAD_ALPHA_VALUE")
    try:
        with torch.no_grad():
            controller.coefficients.copy_(coefficients.to(controller.coefficients))
        controller.const_gain = None
        controller._mode = "idle"
        controller._collecting = False
        os.environ.pop("RESGRAD_ALPHA_PERIOD", None)
        os.environ.pop("RESGRAD_ALPHA_VALUE", None)
        yield
    finally:
        with torch.no_grad():
            controller.coefficients.copy_(saved_coefficients)
        controller.const_gain = saved_const_gain
        controller._mode = saved_mode
        controller._collecting = saved_collecting
        if saved_period is None:
            os.environ.pop("RESGRAD_ALPHA_PERIOD", None)
        else:
            os.environ["RESGRAD_ALPHA_PERIOD"] = saved_period
        if saved_value is None:
            os.environ.pop("RESGRAD_ALPHA_VALUE", None)
        else:
            os.environ["RESGRAD_ALPHA_VALUE"] = saved_value
