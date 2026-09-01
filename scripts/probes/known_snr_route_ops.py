"""Shared route-moment operations for the known-SNR closure."""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np
import torch

from internal_dw.models.dual_wiener import solve_box_wiener_2x2
from probe_wiener_oracle import rollout


PairMap = Dict[int, Tuple[torch.Tensor, torch.Tensor]]


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return 0.5 * (matrix + matrix.T)


def _moment(pair: Tuple[torch.Tensor, torch.Tensor], rows=None) -> np.ndarray:
    identity, branch = pair
    if rows is not None:
        index = torch.as_tensor(rows, device=identity.device, dtype=torch.long)
        identity = identity.index_select(0, index)
        branch = branch.index_select(0, index)
    vector = torch.stack((identity, branch)).reshape(2, -1)
    return (
        (vector @ vector.t()) / max(int(vector.shape[1]), 1)
    ).detach().cpu().numpy()


def route_moments(pairs: PairMap, rows=None) -> Dict[int, np.ndarray]:
    return {int(route): _moment(pair, rows=rows) for route, pair in pairs.items()}


def normalization(
    npz_path: str,
    state_key: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
):
    with np.load(npz_path, allow_pickle=True) as archive:
        raw = np.asarray(archive[state_key], dtype=np.float32)
        coefficients = np.asarray(archive["coefficients"], dtype=np.float64)
    if raw.ndim != 3:
        raise SystemExit(f"known-SNR state must be [B,T,D], got {raw.shape}")
    order = np.random.default_rng(seed).permutation(raw.shape[0])
    n_train = max(1, int(round(raw.shape[0] * train_ratio)))
    n_val = max(1, int(round(raw.shape[0] * val_ratio)))
    if n_train + n_val >= raw.shape[0]:
        n_train, n_val = max(1, raw.shape[0] - 2), 1
    train = raw[order[:n_train]]
    mean = float(train.mean())
    std = float(train.std()) or 1.0
    return mean, std, coefficients


def process_tensors(
    transition: np.ndarray,
    covariance: np.ndarray,
    device: str,
    dtype: torch.dtype,
):
    covariance = 0.5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(covariance)
    factor = (vectors * np.sqrt(np.maximum(values, 0.0))[None, :]) @ vectors.T
    return (
        torch.as_tensor(transition, device=device, dtype=dtype),
        torch.as_tensor(factor, device=device, dtype=dtype),
    )


def sample_process(
    z: torch.Tensor,
    transition: torch.Tensor,
    factor: torch.Tensor,
) -> list[torch.Tensor]:
    state = torch.zeros_like(z[0])
    output = []
    for horizon in range(z.shape[0]):
        state = state @ transition.t() + z[horizon] @ factor.t()
        output.append(state)
    return output


def conditional_decomposition(
    bundle,
    starts,
    predictions,
    targets,
    coefficients,
    mean,
    std,
    args,
):
    start_tensor = torch.as_tensor(starts, device=args.device)
    raw_mean = bundle.xt[bundle.rows_t, start_tensor] * std + mean
    coefficient_tensor = torch.as_tensor(
        coefficients, device=args.device, dtype=raw_mean.dtype
    )
    signal, noise, total = [], [], []
    for prediction, target in zip(predictions, targets):
        raw_mean = raw_mean * coefficient_tensor
        conditional_mean = (raw_mean - mean) / std
        signal.append((prediction - conditional_mean).detach())
        noise.append((conditional_mean - target).detach())
        total.append((prediction - target).detach())
    max_error = max(
        float((error - bias - innovation).abs().max().detach().cpu())
        for error, bias, innovation in zip(total, signal, noise)
    )
    return signal, noise, total, max_error


def solve_gain(total: np.ndarray, noise: np.ndarray) -> np.ndarray:
    return solve_box_wiener_2x2(
        torch.as_tensor(total, dtype=torch.float64),
        torch.as_tensor(noise, dtype=torch.float64),
    ).detach().cpu().numpy().astype(np.float64)


def _prediction_values(bundle, starts, args, coefficients, collecting: bool):
    controller = bundle.dw
    with torch.no_grad():
        controller.coefficients.copy_(
            torch.as_tensor(
                coefficients,
                device=controller.coefficients.device,
                dtype=controller.coefficients.dtype,
            )
        )
    controller._mode = "apply"
    controller._collecting = bool(collecting)
    controller._slot = 0
    controller._root_refs = []
    predictions, roots, targets = rollout(bundle, starts, args)
    values = [prediction.detach().cpu().clone() for prediction in predictions]
    del predictions, roots, targets
    controller._root_refs = []
    return values


def forward_invariance(
    bundle,
    starts,
    args,
    arms: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Verify that changing backward gains leaves every prediction unchanged."""

    controller = bundle.dw
    saved_coefficients = controller.coefficients.detach().clone()
    saved_mode = controller._mode
    saved_collecting = controller._collecting
    saved_slot = controller._slot
    output = {}
    try:
        for collecting, label in (
            (False, "ordinary_straight_through"),
            (True, "collecting_custom_autograd"),
        ):
            reference = _prediction_values(
                bundle, starts, args, arms["open"], collecting
            )
            comparisons = {}
            for name, coefficients in arms.items():
                values = _prediction_values(
                    bundle, starts, args, coefficients, collecting
                )
                max_abs = max(
                    float((value - base).abs().max())
                    for value, base in zip(values, reference)
                )
                numerator = sum(
                    float((value - base).double().square().sum())
                    for value, base in zip(values, reference)
                )
                denominator = sum(
                    float(base.double().square().sum()) for base in reference
                )
                comparisons[name] = {
                    "bitwise_equal_to_open": all(
                        torch.equal(value, base)
                        for value, base in zip(values, reference)
                    ),
                    "max_absolute_difference": max_abs,
                    "relative_l2_difference": math.sqrt(
                        numerator / max(denominator, 1e-300)
                    ),
                }
            output[label] = comparisons
    finally:
        with torch.no_grad():
            controller.coefficients.copy_(saved_coefficients)
        controller._mode = saved_mode
        controller._collecting = saved_collecting
        controller._slot = saved_slot
        controller._root_refs = []
    output["all_arms_bitwise_equal"] = all(
        item["bitwise_equal_to_open"]
        for mode in ("ordinary_straight_through", "collecting_custom_autograd")
        for item in output[mode].values()
    )
    output["scope"] = "forward predictions only; backward coefficients are changed"
    return output
