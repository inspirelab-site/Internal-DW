#!/usr/bin/env python3
"""Held-out global-gradient oracle audit for the known-SNR AR testbed.

The route probes in ``probe_known_snr_matched_routes.py`` test the local 2x2
Wiener estimand.  This script tests the quantity that the optimizer actually
receives: the complete parameter-gradient vector after every routed residual
merge, recurrent carry, and rollout loss has been composed.

For one frozen checkpoint and one observed state x_t, the diagonal Gaussian AR
process makes both of the following available:

* E[x_{t+k} | x_t], hence the exact conditional-mean full-BPTT gradient;
* independent future innovation draws from the same x_t.

On disjoint fit starts, the script estimates (i) the routewise oracle 2x2 gains
used by the current implementation and (ii) a joint horizon-weight oracle that
keeps all cross-horizon gradient covariances.  On held-out starts it compares
Exact, a static-gain grid, a TBPTT-period grid, checkpoint plug-in DW, fitted
routewise Oracle DW, and the fitted joint horizon oracle.

The same realized gradients are also passed through the next AdamW data-update
map, using the optimizer moments stored in the checkpoint.  Comparing raw and
Adam-space risks directly tests whether adaptive normalization erases or
reorders a routing advantage.  No parameter is updated and no checkpoint is
modified.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from probe_setup import add_common_args, burn_in_root, setup  # noqa: E402
from probe_wiener_oracle import RouteCapture, covector_loss, push  # noqa: E402
from probe_known_snr_matched_routes import _normalization  # noqa: E402
from internal_dw.models.dual_wiener import solve_box_wiener_2x2  # noqa: E402


TensorList = List[torch.Tensor]
PairMap = Dict[int, Tuple[torch.Tensor, torch.Tensor]]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _parse_floats(raw: str) -> List[float]:
    return [float(value) for value in raw.split(",") if value.strip()]


def _parse_ints(raw: str) -> List[int]:
    return [int(value) for value in raw.split(",") if value.strip()]


def _load_process(oracle_file: str, normalization_std: float, device: str):
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


def _conditional_and_noisy_targets(
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
    """Generate independent futures from exactly the same observed state."""

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

    generator_device = bundle.xt.device.type
    generator = torch.Generator(device=generator_device).manual_seed(int(seed))
    normalized_noise_scale = innovation_std_normalized.to(normalized_initial)
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
            # Converting the raw AR recursion to normalized coordinates gives
            # a*x_norm + (a*mean-mean)/std + eps/std.
            offset = (
                raw_coefficients * float(normalization_mean)
                - float(normalization_mean)
            ) / float(normalization_std)
            state = state * raw_coefficients + offset + innovation
            draw.append(state.detach())
        noisy.append(draw)
    return conditional, noisy


def _rollout_predictions(bundle, starts: Sequence[int], args) -> TensorList:
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


def _total_mse(predictions: Sequence[torch.Tensor], targets: Sequence[torch.Tensor]):
    return torch.stack(
        [(prediction - target).square().mean() for prediction, target in zip(predictions, targets)]
    ).mean()


def _horizon_losses(
    predictions: Sequence[torch.Tensor], targets: Sequence[torch.Tensor]
) -> List[torch.Tensor]:
    scale = 1.0 / max(len(predictions), 1)
    return [
        scale * (prediction - target).square().mean()
        for prediction, target in zip(predictions, targets)
    ]


def _trainable_parameters(model) -> Tuple[List[str], List[torch.Tensor]]:
    pairs = [(name, parameter) for name, parameter in model.named_parameters()
             if parameter.requires_grad]
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


def _gradient_matrix(
    losses: Sequence[torch.Tensor],
    parameters: Sequence[torch.Tensor],
    retain_after: bool,
) -> torch.Tensor:
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
            if not getattr(_gradient_matrix, "_fallback_announced", False):
                print(
                    "[global oracle] batched VJP unavailable; falling back to "
                    f"sequential horizon VJPs: {type(error).__name__}: {error}",
                    flush=True,
                )
                _gradient_matrix._fallback_announced = True
    rows = []
    for index, loss in enumerate(losses):
        keep = retain_after or index + 1 < len(losses)
        rows.append(_gradient_vector(loss, parameters, retain_graph=keep))
    return torch.stack(rows, dim=0)


@contextlib.contextmanager
def _routing_operator(
    controller,
    coefficients: torch.Tensor,
    *,
    const_gain: Optional[float] = None,
    tbptt_period: int = 0,
):
    saved_coefficients = controller.coefficients.detach().clone()
    saved_const_gain = controller.const_gain
    saved_mode = controller._mode
    saved_collecting = controller._collecting
    saved_period = os.environ.get("RESGRAD_ALPHA_PERIOD")
    saved_value = os.environ.get("RESGRAD_ALPHA_VALUE")
    try:
        with torch.no_grad():
            controller.coefficients.copy_(coefficients.to(controller.coefficients))
        controller.const_gain = const_gain
        controller._mode = "idle"
        controller._collecting = False
        if int(tbptt_period) > 0:
            os.environ["RESGRAD_ALPHA_PERIOD"] = str(int(tbptt_period))
            os.environ["RESGRAD_ALPHA_VALUE"] = "0.0"
        else:
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


def _route_moment(pair: Tuple[torch.Tensor, torch.Tensor]) -> np.ndarray:
    vector = torch.stack(pair).reshape(2, -1).double()
    return ((vector @ vector.t()) / max(int(vector.shape[1]), 1)).cpu().numpy()


def _mean_matrix_maps(maps: Iterable[Mapping[int, np.ndarray]]) -> Dict[int, np.ndarray]:
    sums: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = defaultdict(int)
    for matrix_map in maps:
        for route, matrix in matrix_map.items():
            sums.setdefault(int(route), np.zeros((2, 2), dtype=np.float64))
            sums[int(route)] += np.asarray(matrix, dtype=np.float64)
            counts[int(route)] += 1
    return {route: value / counts[route] for route, value in sums.items()}


def _fit_routewise_oracle(
    bundle,
    args,
    fit_starts: Sequence[Sequence[int]],
    coefficients: torch.Tensor,
    innovation_std_normalized: torch.Tensor,
    normalization_mean: float,
    normalization_std: float,
) -> Tuple[torch.Tensor, dict]:
    controller = bundle.dw
    capture = RouteCapture(controller)
    signal_maps: List[Dict[int, np.ndarray]] = []
    noise_maps: List[Dict[int, np.ndarray]] = []
    open_coefficients = torch.ones_like(controller.coefficients)
    saved_mode = controller._mode
    saved_collecting = controller._collecting
    try:
        with _routing_operator(controller, open_coefficients):
            for draw_index, starts in enumerate(fit_starts):
                conditional, noisy = _conditional_and_noisy_targets(
                    bundle,
                    starts,
                    coefficients,
                    innovation_std_normalized,
                    normalization_mean,
                    normalization_std,
                    args.K,
                    args.noise_draws,
                    args.seed + 100003 * (draw_index + 1),
                )
                controller.begin_batch()
                controller._mode = "total"
                controller._collecting = True
                controller._slot = 0
                current, hidden, roots, _labels, starts_t = burn_in_root(
                    bundle, starts, args
                )
                predictions: TensorList = []
                value = current
                for horizon in range(int(args.K)):
                    stimulus = (
                        bundle.ut[bundle.rows_t, starts_t + horizon]
                        if bundle.ut is not None else None
                    )
                    output = bundle.model.step(
                        hidden,
                        value,
                        stim_t=stimulus,
                        horizon_index=horizon,
                        total_horizon=int(args.K),
                    )
                    prediction, hidden = output[0], output[1]
                    predictions.append(prediction)
                    value = prediction

                signal_vectors = [
                    (prediction - target).detach()
                    for prediction, target in zip(predictions, conditional)
                ]
                signal_pairs = push(
                    covector_loss(predictions, signal_vectors),
                    roots,
                    capture,
                    retain=True,
                )
                signal_maps.append({
                    route: _route_moment(pair) for route, pair in signal_pairs.items()
                })
                for noise_index, targets in enumerate(noisy):
                    noise_vectors = [
                        (mean_target - noisy_target).detach()
                        for mean_target, noisy_target in zip(conditional, targets)
                    ]
                    is_last = noise_index + 1 == len(noisy)
                    noise_pairs = push(
                        covector_loss(predictions, noise_vectors),
                        roots,
                        capture,
                        retain=not is_last,
                    )
                    noise_maps.append({
                        route: _route_moment(pair) for route, pair in noise_pairs.items()
                    })
                print(
                    f"[route oracle fit] draw {draw_index + 1}/{len(fit_starts)} "
                    f"routes={len(signal_pairs)} noise_draws={len(noisy)}",
                    flush=True,
                )
    finally:
        capture.close()
        controller._mode = saved_mode
        controller._collecting = saved_collecting

    signal = _mean_matrix_maps(signal_maps)
    noise = _mean_matrix_maps(noise_maps)
    routes = sorted(set(signal).intersection(noise))
    fitted = torch.ones_like(controller.coefficients)
    gains = []
    for route in routes:
        total = 0.5 * (signal[route] + signal[route].T)
        covariance = 0.5 * (noise[route] + noise[route].T)
        gain = solve_box_wiener_2x2(
            torch.as_tensor(total + covariance, dtype=torch.float64),
            torch.as_tensor(covariance, dtype=torch.float64),
        ).to(fitted)
        horizon, layer = divmod(int(route), int(bundle.depth))
        if horizon < fitted.shape[0] and layer < fitted.shape[1]:
            fitted[horizon, layer].copy_(gain)
            gains.append(gain.detach().cpu().numpy())
    if not gains:
        raise RuntimeError("routewise oracle fit captured no common routes")
    gain_array = np.stack(gains)
    summary = {
        "routes": len(gains),
        "alpha_mean": float(gain_array[:, 0].mean()),
        "alpha_sd": float(gain_array[:, 0].std()),
        "m_mean": float(gain_array[:, 1].mean()),
        "m_sd": float(gain_array[:, 1].std()),
    }
    print(
        "[route oracle fit] "
        f"alpha={summary['alpha_mean']:.4f}+/-{summary['alpha_sd']:.4f} "
        f"m={summary['m_mean']:.4f}+/-{summary['m_sd']:.4f}",
        flush=True,
    )
    return fitted, summary


def _solve_box_quadratic(covariance: np.ndarray, linear: np.ndarray) -> Tuple[np.ndarray, dict]:
    covariance = 0.5 * (
        np.asarray(covariance, dtype=np.float64)
        + np.asarray(covariance, dtype=np.float64).T
    )
    linear = np.asarray(linear, dtype=np.float64)
    dimension = int(linear.size)
    ridge = 1e-10 * max(float(np.trace(covariance)) / max(dimension, 1), 1.0)
    regularized = covariance + ridge * np.eye(dimension)
    try:
        initial = np.clip(np.linalg.solve(regularized, linear), 0.0, 1.0)
    except np.linalg.LinAlgError:
        initial = np.full(dimension, 0.5, dtype=np.float64)

    method = "projected_coordinate_descent"
    converged = False
    solution = initial.copy()
    try:
        from scipy.optimize import minimize  # type: ignore

        result = minimize(
            lambda value: 0.5 * float(value @ covariance @ value)
            - float(linear @ value),
            initial,
            jac=lambda value: covariance @ value - linear,
            bounds=[(0.0, 1.0)] * dimension,
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-10},
        )
        solution = np.clip(result.x, 0.0, 1.0)
        converged = bool(result.success)
        method = "scipy_L-BFGS-B"
    except Exception:
        for _iteration in range(20000):
            previous = solution.copy()
            for coordinate in range(dimension):
                diagonal = covariance[coordinate, coordinate] + ridge
                if diagonal <= 0:
                    solution[coordinate] = 0.0
                    continue
                remainder = covariance[coordinate] @ solution
                remainder -= covariance[coordinate, coordinate] * solution[coordinate]
                solution[coordinate] = np.clip(
                    (linear[coordinate] - remainder) / diagonal, 0.0, 1.0
                )
            if np.max(np.abs(solution - previous)) < 1e-10:
                converged = True
                break
    gradient = covariance @ solution - linear
    projected_residual = np.where(
        solution <= 1e-9,
        np.minimum(gradient, 0.0),
        np.where(solution >= 1.0 - 1e-9, np.maximum(gradient, 0.0), gradient),
    )
    return solution, {
        "solver": method,
        "converged": converged,
        "projected_gradient_max": float(np.abs(projected_residual).max()),
        "ridge": ridge,
    }


def _fit_global_horizon_oracle(
    bundle,
    args,
    parameters: Sequence[torch.Tensor],
    fit_starts: Sequence[Sequence[int]],
    coefficients: torch.Tensor,
    innovation_std_normalized: torch.Tensor,
    normalization_mean: float,
    normalization_std: float,
) -> Tuple[np.ndarray, dict]:
    controller = bundle.dw
    open_coefficients = torch.ones_like(controller.coefficients)
    covariance = np.zeros((args.K, args.K), dtype=np.float64)
    linear = np.zeros(args.K, dtype=np.float64)
    observations = 0
    parameter_count = sum(int(parameter.numel()) for parameter in parameters)
    with _routing_operator(controller, open_coefficients):
        for draw_index, starts in enumerate(fit_starts):
            conditional, noisy = _conditional_and_noisy_targets(
                bundle,
                starts,
                coefficients,
                innovation_std_normalized,
                normalization_mean,
                normalization_std,
                args.K,
                args.noise_draws,
                args.seed + 200003 * (draw_index + 1),
            )
            predictions = _rollout_predictions(bundle, starts, args)
            clean_matrix = _gradient_matrix(
                _horizon_losses(predictions, conditional),
                parameters,
                retain_after=True,
            )
            clean_target = clean_matrix.sum(dim=0)
            del clean_matrix
            for noise_index, targets in enumerate(noisy):
                keep_after = noise_index + 1 < len(noisy)
                noisy_matrix = _gradient_matrix(
                    _horizon_losses(predictions, targets),
                    parameters,
                    retain_after=keep_after,
                )
                scale = max(parameter_count, 1)
                covariance += (
                    noisy_matrix @ noisy_matrix.t()
                ).double().cpu().numpy() / scale
                linear += (
                    noisy_matrix @ clean_target
                ).double().cpu().numpy() / scale
                observations += 1
                del noisy_matrix
            del predictions, clean_target
            print(
                f"[global oracle fit] draw {draw_index + 1}/{len(fit_starts)} "
                f"noise_draws={len(noisy)}",
                flush=True,
            )
    covariance /= max(observations, 1)
    linear /= max(observations, 1)
    weights, solver = _solve_box_quadratic(covariance, linear)
    summary = {
        "observations": observations,
        "weight_mean": float(weights.mean()),
        "weight_sd": float(weights.std()),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weights": weights.tolist(),
        **solver,
    }
    print(
        f"[global oracle fit] w={weights.mean():.4f}+/-{weights.std():.4f} "
        f"range=[{weights.min():.4f},{weights.max():.4f}] "
        f"solver={solver['solver']} pg={solver['projected_gradient_max']:.2e}",
        flush=True,
    )
    return weights, summary


def _solve_box_quadratic_2x2(
    quadratic: np.ndarray,
    linear: np.ndarray,
) -> Tuple[np.ndarray, dict]:
    """Exactly minimize ``0.5*w^T H*w - b^T*w`` over ``[0,1]^2``.

    Componentwise clipping of the unconstrained solution is not exact when the
    two route contributions have nonzero cross-covariance.  In two dimensions
    the exact box solve is still finite: test the feasible interior stationary
    point and the stationary point (plus endpoints) on each of the four edges.
    """

    matrix = np.asarray(quadratic, dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    vector = np.asarray(linear, dtype=np.float64).reshape(2)
    if matrix.shape != (2, 2):
        raise ValueError(f"expected a 2x2 quadratic, got {matrix.shape}")
    scale = max(float(np.trace(matrix)) / 2.0, 1.0)
    numerical_ridge = 1e-12 * scale
    regularized = matrix + numerical_ridge * np.eye(2, dtype=np.float64)
    candidates: List[np.ndarray] = []

    try:
        interior = np.linalg.solve(regularized, vector)
    except np.linalg.LinAlgError:
        interior = np.linalg.pinv(regularized) @ vector
    if np.all(interior >= 0.0) and np.all(interior <= 1.0):
        candidates.append(interior)

    # alpha is fixed, optimize m on the edge.
    for alpha in (0.0, 1.0):
        if regularized[1, 1] > 0.0:
            m_value = (
                vector[1] - regularized[1, 0] * alpha
            ) / regularized[1, 1]
            candidates.append(np.asarray([alpha, np.clip(m_value, 0.0, 1.0)]))
        candidates.extend([
            np.asarray([alpha, 0.0]),
            np.asarray([alpha, 1.0]),
        ])

    # m is fixed, optimize alpha on the edge.
    for m_value in (0.0, 1.0):
        if regularized[0, 0] > 0.0:
            alpha = (
                vector[0] - regularized[0, 1] * m_value
            ) / regularized[0, 0]
            candidates.append(np.asarray([np.clip(alpha, 0.0, 1.0), m_value]))
        candidates.extend([
            np.asarray([0.0, m_value]),
            np.asarray([1.0, m_value]),
        ])

    def objective(value: np.ndarray) -> float:
        return 0.5 * float(value @ regularized @ value) - float(vector @ value)

    best = min(candidates, key=objective)
    gradient = regularized @ best - vector
    projected = np.where(
        best <= 1e-10,
        np.minimum(gradient, 0.0),
        np.where(best >= 1.0 - 1e-10, np.maximum(gradient, 0.0), gradient),
    )
    return best, {
        "objective": objective(best),
        "projected_gradient_max": float(np.max(np.abs(projected))),
        "numerical_ridge": numerical_ridge,
        "interior_feasible": bool(
            np.all(interior >= 0.0) and np.all(interior <= 1.0)
        ),
    }


@dataclass
class GlobalRouteFitCase:
    starts: Sequence[int]
    conditional_targets: TensorList
    noisy_targets: List[TensorList]
    clean_parameter_gradient: torch.Tensor


def _build_global_route_fit_cases(
    bundle,
    args,
    parameters: Sequence[torch.Tensor],
    fit_starts: Sequence[Sequence[int]],
    coefficients: torch.Tensor,
    innovation_std_normalized: torch.Tensor,
    normalization_mean: float,
    normalization_std: float,
) -> List[GlobalRouteFitCase]:
    """Freeze matched conditional targets/noise draws for route BCD.

    Every coordinate sees the same Monte Carlo sample.  This is important:
    resampling innovations for each route would inject optimizer-order noise
    into what is intended to be an exact empirical coordinate descent step.
    """

    controller = bundle.dw
    open_coefficients = torch.ones_like(controller.coefficients)
    requested_noise = int(args.global_route_noise_draws)
    if requested_noise <= 0:
        requested_noise = int(args.noise_draws)
    cases: List[GlobalRouteFitCase] = []
    with _routing_operator(controller, open_coefficients):
        for draw_index, starts in enumerate(fit_starts):
            conditional, noisy = _conditional_and_noisy_targets(
                bundle,
                starts,
                coefficients,
                innovation_std_normalized,
                normalization_mean,
                normalization_std,
                args.K,
                requested_noise,
                args.seed + 400009 * (draw_index + 1),
            )
            predictions = _rollout_predictions(bundle, starts, args)
            clean_gradient = _gradient_vector(
                _total_mse(predictions, conditional),
                parameters,
                retain_graph=False,
            )
            del predictions
            cases.append(
                GlobalRouteFitCase(
                    starts=list(starts),
                    conditional_targets=conditional,
                    noisy_targets=noisy,
                    clean_parameter_gradient=clean_gradient,
                )
            )
    return cases


def _parameter_gradient_ensemble(
    bundle,
    args,
    parameters: Sequence[torch.Tensor],
    fit_case: GlobalRouteFitCase,
    route_coefficients: torch.Tensor,
) -> torch.Tensor:
    """Return one complete parameter-gradient row per innovation realization."""

    with _routing_operator(bundle.dw, route_coefficients):
        predictions = _rollout_predictions(bundle, fit_case.starts, args)
        losses = [
            _total_mse(predictions, targets)
            for targets in fit_case.noisy_targets
        ]
        gradients = _gradient_matrix(losses, parameters, retain_after=False)
        del predictions, losses
    return gradients


def _global_route_empirical_risk(
    bundle,
    args,
    parameters: Sequence[torch.Tensor],
    fit_cases: Sequence[GlobalRouteFitCase],
    route_coefficients: torch.Tensor,
) -> float:
    squared_error = 0.0
    coordinates = 0
    for fit_case in fit_cases:
        gradients = _parameter_gradient_ensemble(
            bundle, args, parameters, fit_case, route_coefficients
        )
        difference = gradients - fit_case.clean_parameter_gradient.unsqueeze(0)
        squared_error += float(difference.double().square().sum().item())
        coordinates += int(difference.numel())
        del gradients, difference
    return squared_error / max(coordinates, 1)


def _fit_global_conditioned_routes(
    bundle,
    args,
    parameters: Sequence[torch.Tensor],
    fit_starts: Sequence[Sequence[int]],
    coefficients: torch.Tensor,
    innovation_std_normalized: torch.Tensor,
    normalization_mean: float,
    normalization_std: float,
    plugin_coefficients: torch.Tensor,
    route_oracle_coefficients: torch.Tensor,
) -> Tuple[torch.Tensor, dict]:
    """Block-coordinate descent on complete parameter-gradient risk.

    Holding every other route fixed makes the complete parameter gradient
    affine in one route pair,

        G(theta; alpha_r, m_r) = C_r + alpha_r A_r + m_r B_r.

    The resulting coordinate subproblem is the exact global 2x2 box-constrained
    Wiener solve.  It differs from the shipped local route oracle because
    ``A_r`` and ``B_r`` are complete parameter-gradient contributions and the
    right-hand side contains the gradient ``C_r`` supplied by every other path.
    """

    controller = bundle.dw
    initialization = str(args.global_route_init).lower()
    if initialization == "route_oracle":
        working = route_oracle_coefficients.detach().clone()
    elif initialization == "plugin":
        working = plugin_coefficients.detach().clone()
    elif initialization == "open":
        working = torch.ones_like(controller.coefficients)
    elif initialization == "half":
        working = torch.full_like(controller.coefficients, 0.5)
    else:
        raise ValueError(f"unknown global-route initialization {initialization!r}")

    fit_cases = _build_global_route_fit_cases(
        bundle,
        args,
        parameters,
        fit_starts,
        coefficients,
        innovation_std_normalized,
        normalization_mean,
        normalization_std,
    )
    open_coefficients = torch.ones_like(controller.coefficients)
    open_risk = _global_route_empirical_risk(
        bundle, args, parameters, fit_cases, open_coefficients
    )
    initial_risk = _global_route_empirical_risk(
        bundle, args, parameters, fit_cases, working
    )
    risk_trace = [{
        "stage": "initial",
        "risk": initial_risk,
        "risk_over_open": initial_risk / max(open_risk, 1e-300),
    }]

    route_count = min(int(args.K), int(working.shape[0])) * min(
        int(bundle.depth), int(working.shape[1])
    )
    maximum_routes = int(args.global_route_max_routes)
    if maximum_routes > 0:
        route_count = min(route_count, maximum_routes)
    parameter_count = max(sum(int(parameter.numel()) for parameter in parameters), 1)
    updates: List[dict] = []
    affine_checks: List[float] = []

    for sweep in range(int(args.global_route_sweeps)):
        route_indices = list(range(route_count))
        order = str(args.global_route_order).lower()
        if order == "reverse" or (order == "alternating" and sweep % 2 == 0):
            route_indices.reverse()
        elif order not in ("forward", "alternating"):
            raise ValueError(f"unknown global-route order {order!r}")

        sweep_improvement = 0.0
        for position, route_index in enumerate(route_indices):
            horizon, layer = divmod(int(route_index), int(bundle.depth))
            if horizon >= working.shape[0] or layer >= working.shape[1]:
                continue
            old_pair = working[horizon, layer].detach().double().cpu().numpy()
            quadratic = np.zeros((2, 2), dtype=np.float64)
            linear = np.zeros(2, dtype=np.float64)
            observations = 0
            route_affine_checks: List[float] = []
            check_every = int(args.global_route_affine_check_every)
            check_affine = check_every > 0 and position % check_every == 0

            zero_coefficients = working.detach().clone()
            identity_coefficients = working.detach().clone()
            branch_coefficients = working.detach().clone()
            zero_coefficients[horizon, layer].zero_()
            identity_coefficients[horizon, layer].copy_(
                identity_coefficients.new_tensor([1.0, 0.0])
            )
            branch_coefficients[horizon, layer].copy_(
                branch_coefficients.new_tensor([0.0, 1.0])
            )

            for fit_case in fit_cases:
                base = _parameter_gradient_ensemble(
                    bundle, args, parameters, fit_case, zero_coefficients
                )
                identity = _parameter_gradient_ensemble(
                    bundle, args, parameters, fit_case, identity_coefficients
                )
                branch = _parameter_gradient_ensemble(
                    bundle, args, parameters, fit_case, branch_coefficients
                )
                contribution_i = identity - base
                contribution_b = branch - base
                desired = fit_case.clean_parameter_gradient.unsqueeze(0) - base
                if check_affine:
                    actual = _parameter_gradient_ensemble(
                        bundle, args, parameters, fit_case, working
                    )
                    predicted = (
                        base
                        + float(old_pair[0]) * contribution_i
                        + float(old_pair[1]) * contribution_b
                    )
                    closure = actual - predicted
                    relative_closure = float(
                        closure.double().square().mean().item()
                        / max(actual.double().square().mean().item(), 1e-300)
                    )
                    route_affine_checks.append(relative_closure)
                    affine_checks.append(relative_closure)
                    del actual, predicted, closure
                scale = float(parameter_count)
                quadratic[0, 0] += float(
                    (contribution_i.double() * contribution_i.double()).sum().item()
                    / scale
                )
                quadratic[0, 1] += float(
                    (contribution_i.double() * contribution_b.double()).sum().item()
                    / scale
                )
                quadratic[1, 0] = quadratic[0, 1]
                quadratic[1, 1] += float(
                    (contribution_b.double() * contribution_b.double()).sum().item()
                    / scale
                )
                linear[0] += float(
                    (contribution_i.double() * desired.double()).sum().item() / scale
                )
                linear[1] += float(
                    (contribution_b.double() * desired.double()).sum().item() / scale
                )
                observations += int(base.shape[0])
                del base, identity, branch, contribution_i, contribution_b, desired

            quadratic /= max(observations, 1)
            linear /= max(observations, 1)
            covariance_scale = max(float(np.trace(quadratic)) / 2.0, 1e-30)
            ridge = float(args.global_route_ridge) * covariance_scale
            regularized_quadratic = quadratic + ridge * np.eye(2)
            regularized_linear = linear + ridge * old_pair
            new_pair, solver = _solve_box_quadratic_2x2(
                regularized_quadratic, regularized_linear
            )

            def local_objective(value: np.ndarray) -> float:
                delta = value - old_pair
                return (
                    0.5 * float(value @ quadratic @ value)
                    - float(linear @ value)
                    + 0.5 * ridge * float(delta @ delta)
                )

            before = local_objective(old_pair)
            after = local_objective(new_pair)
            improvement = before - after
            sweep_improvement += improvement
            with torch.no_grad():
                working[horizon, layer].copy_(
                    torch.as_tensor(new_pair, device=working.device, dtype=working.dtype)
                )
            updates.append({
                "sweep": sweep + 1,
                "position": position,
                "route_index": route_index,
                "horizon": horizon,
                "layer": layer,
                "old_pair": old_pair.tolist(),
                "new_pair": new_pair.tolist(),
                "observations": observations,
                "quadratic": quadratic.tolist(),
                "linear": linear.tolist(),
                "ridge": ridge,
                "local_objective_before": before,
                "local_objective_after": after,
                "local_objective_improvement": improvement,
                "affine_closure_relative_mse": (
                    float(max(route_affine_checks))
                    if route_affine_checks else None
                ),
                **solver,
            })
            if route_affine_checks and max(route_affine_checks) > float(
                args.global_route_affine_tolerance
            ):
                raise RuntimeError(
                    "complete parameter gradient is not affine in route "
                    f"{route_index}: relative closure MSE "
                    f"{max(route_affine_checks):.3e} exceeds "
                    f"{float(args.global_route_affine_tolerance):.3e}"
                )
            print(
                f"[global route BCD] sweep={sweep + 1}/"
                f"{int(args.global_route_sweeps)} route={route_index} "
                f"({position + 1}/{len(route_indices)}) "
                f"[{old_pair[0]:.3f},{old_pair[1]:.3f}] -> "
                f"[{new_pair[0]:.3f},{new_pair[1]:.3f}] "
                f"dobj={improvement:.3e}",
                flush=True,
            )

        fitted_risk = _global_route_empirical_risk(
            bundle, args, parameters, fit_cases, working
        )
        risk_trace.append({
            "stage": f"sweep_{sweep + 1}",
            "risk": fitted_risk,
            "risk_over_open": fitted_risk / max(open_risk, 1e-300),
            "summed_coordinate_objective_improvement": sweep_improvement,
        })
        print(
            f"[global route BCD] completed sweep {sweep + 1}: "
            f"fit risk/open={fitted_risk / max(open_risk, 1e-300):.6f}",
            flush=True,
        )

    fitted_pairs = working[: int(args.K)].detach().float().cpu().reshape(-1, 2).numpy()
    summary = {
        "enabled": True,
        "initialization": initialization,
        "route_order": str(args.global_route_order),
        "sweeps": int(args.global_route_sweeps),
        "routes_updated_per_sweep": route_count,
        "fit_cases": len(fit_cases),
        "noise_draws_per_case": len(fit_cases[0].noisy_targets) if fit_cases else 0,
        "open_fit_risk": open_risk,
        "risk_trace": risk_trace,
        "alpha_mean": float(fitted_pairs[:, 0].mean()),
        "alpha_sd": float(fitted_pairs[:, 0].std()),
        "m_mean": float(fitted_pairs[:, 1].mean()),
        "m_sd": float(fitted_pairs[:, 1].std()),
        "affine_closure_relative_mse": affine_checks,
        "affine_closure_relative_mse_max": (
            float(max(affine_checks)) if affine_checks else None
        ),
        "updates": updates,
    }
    return working, summary


@dataclass
class AdamUpdateMap:
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor
    parameter: torch.Tensor
    beta1: float
    beta2: float
    eps: float
    learning_rate: float
    weight_decay: float
    next_step: int
    optimizer_kind: str
    grad_clip: float

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        parameters: Sequence[torch.Tensor],
        device: torch.device,
    ) -> Optional["AdamUpdateMap"]:
        blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(blob, dict) or "optimizer" not in blob:
            return None
        optimizer = blob["optimizer"]
        groups = optimizer.get("param_groups", [])
        if len(groups) != 1:
            raise RuntimeError("global-gradient probe currently expects one optimizer group")
        group = groups[0]
        identifiers = list(group.get("params", []))
        if len(identifiers) != len(parameters):
            raise RuntimeError(
                f"optimizer/model parameter count mismatch: {len(identifiers)} vs "
                f"{len(parameters)}"
            )
        first_moments, second_moments, values, steps = [], [], [], []
        states = optimizer.get("state", {})
        for identifier, parameter in zip(identifiers, parameters):
            state = states.get(identifier, states.get(str(identifier), {}))
            first = state.get("exp_avg", torch.zeros_like(parameter, device="cpu"))
            second = state.get("exp_avg_sq", torch.zeros_like(parameter, device="cpu"))
            if tuple(first.shape) != tuple(parameter.shape):
                raise RuntimeError("optimizer state shape does not match model parameter order")
            first_moments.append(first.reshape(-1).float())
            second_moments.append(second.reshape(-1).float())
            values.append(parameter.detach().cpu().reshape(-1).float())
            step = state.get("step", 0)
            steps.append(int(step.item() if isinstance(step, torch.Tensor) else step))
        if len(set(steps)) != 1:
            raise RuntimeError(f"optimizer parameters have unequal Adam steps: {sorted(set(steps))}")
        beta1, beta2 = group.get("betas", (0.9, 0.999))
        checkpoint_args = blob.get("args", {}) if isinstance(blob, dict) else {}
        optimizer_kind = str(checkpoint_args.get("ar_optimizer", "adam")).lower()
        if optimizer_kind not in ("adam", "adamw"):
            raise RuntimeError(
                f"unsupported checkpoint optimizer {optimizer_kind!r}; expected adam/adamw"
            )
        return cls(
            exp_avg=torch.cat(first_moments).to(device),
            exp_avg_sq=torch.cat(second_moments).to(device),
            parameter=torch.cat(values).to(device),
            beta1=float(beta1),
            beta2=float(beta2),
            eps=float(group.get("eps", 1e-8)),
            learning_rate=float(group.get("lr", 1e-4)),
            weight_decay=float(group.get("weight_decay", 0.0)),
            next_step=int(steps[0]) + 1,
            optimizer_kind=optimizer_kind,
            grad_clip=float(checkpoint_args.get("grad_clip", 0.0) or 0.0),
        )

    def clipped(self, gradient: torch.Tensor) -> torch.Tensor:
        if self.grad_clip <= 0.0:
            return gradient
        norm = gradient.double().norm().to(dtype=gradient.dtype)
        scale = torch.clamp(
            gradient.new_tensor(self.grad_clip) / (norm + 1e-6), max=1.0
        )
        return gradient * scale

    def regularized(self, gradient: torch.Tensor) -> torch.Tensor:
        if self.optimizer_kind == "adam" and self.weight_decay != 0.0:
            return gradient + self.weight_decay * self.parameter
        return gradient

    def momentum_only(self, gradient: torch.Tensor) -> torch.Tensor:
        prepared = self.regularized(self.clipped(gradient))
        first = self.beta1 * self.exp_avg + (1.0 - self.beta1) * prepared
        first_hat = first / (1.0 - self.beta1 ** self.next_step)
        return self.learning_rate * first_hat

    def second_moment_only(self, gradient: torch.Tensor) -> torch.Tensor:
        prepared = self.regularized(self.clipped(gradient))
        second = self.beta2 * self.exp_avg_sq + (1.0 - self.beta2) * prepared.square()
        second_hat = second / (1.0 - self.beta2 ** self.next_step)
        return self.learning_rate * prepared / (second_hat.sqrt() + self.eps)

    def __call__(
        self,
        gradient: torch.Tensor,
        fresh: bool = False,
        *,
        full_pipeline: bool = True,
    ) -> torch.Tensor:
        prepared = gradient
        if full_pipeline:
            prepared = self.regularized(self.clipped(prepared))
        if fresh:
            first = (1.0 - self.beta1) * prepared
            second = (1.0 - self.beta2) * prepared.square()
            step = 1
        else:
            first = self.beta1 * self.exp_avg + (1.0 - self.beta1) * prepared
            second = self.beta2 * self.exp_avg_sq + (1.0 - self.beta2) * prepared.square()
            step = self.next_step
        first_hat = first / (1.0 - self.beta1 ** step)
        second_hat = second / (1.0 - self.beta2 ** step)
        data_update = self.learning_rate * first_hat / (second_hat.sqrt() + self.eps)
        if self.optimizer_kind == "adamw" and self.weight_decay != 0.0:
            # PyTorch AdamW applies this common parameter update outside both
            # moment buffers. Keep it so the target-update normalization is the
            # exact optimizer step, not merely its data-gradient component.
            data_update = data_update + (
                self.learning_rate * self.weight_decay * self.parameter
            )
        return data_update


class MetricAccumulator:
    def __init__(self):
        self.rows: Dict[str, List[dict]] = defaultdict(list)

    def add(
        self,
        name: str,
        gradient: torch.Tensor,
        target: torch.Tensor,
        adam_map: Optional[AdamUpdateMap],
    ) -> None:
        eps = 1e-30
        difference = gradient - target
        target_energy = float(target.double().square().mean().item())
        gradient_energy = float(gradient.double().square().mean().item())
        inner = float((gradient.double() * target.double()).mean().item())
        raw = {
            "mse": float(difference.double().square().mean().item()),
            "relative_mse": float(
                difference.double().square().mean().item() / max(target_energy, eps)
            ),
            "cosine": float(inner / max(math.sqrt(gradient_energy * target_energy), eps)),
            "norm_ratio": float(math.sqrt(gradient_energy / max(target_energy, eps))),
        }
        if adam_map is not None:
            clipped_target = adam_map.clipped(target)
            clipped_gradient = adam_map.clipped(gradient)
            prepared_target = adam_map.regularized(clipped_target)
            prepared_gradient = adam_map.regularized(clipped_gradient)
            momentum_target = adam_map.momentum_only(target)
            momentum_gradient = adam_map.momentum_only(gradient)
            second_target = adam_map.second_moment_only(target)
            second_gradient = adam_map.second_moment_only(gradient)
            adam_target = adam_map(target)
            adam_gradient = adam_map(gradient)
            legacy_target = adam_map(target, full_pipeline=False)
            legacy_gradient = adam_map(gradient, full_pipeline=False)
            fresh_target = adam_map(target, fresh=True)
            fresh_gradient = adam_map(gradient, fresh=True)
            clip_target_energy = float(clipped_target.double().square().mean().item())
            prepared_target_energy = float(prepared_target.double().square().mean().item())
            momentum_target_energy = float(momentum_target.double().square().mean().item())
            second_target_energy = float(second_target.double().square().mean().item())
            adam_target_energy = float(adam_target.double().square().mean().item())
            legacy_target_energy = float(legacy_target.double().square().mean().item())
            fresh_target_energy = float(fresh_target.double().square().mean().item())
            raw.update({
                "clip_mse": float(
                    (clipped_gradient - clipped_target).double().square().mean().item()
                ),
                "clip_relative_mse": float(
                    (clipped_gradient - clipped_target).double().square().mean().item()
                    / max(clip_target_energy, eps)
                ),
                "prepared_mse": float(
                    (prepared_gradient - prepared_target).double().square().mean().item()
                ),
                "prepared_relative_mse": float(
                    (prepared_gradient - prepared_target).double().square().mean().item()
                    / max(prepared_target_energy, eps)
                ),
                "momentum_mse": float(
                    (momentum_gradient - momentum_target).double().square().mean().item()
                ),
                "momentum_relative_mse": float(
                    (momentum_gradient - momentum_target).double().square().mean().item()
                    / max(momentum_target_energy, eps)
                ),
                "second_moment_mse": float(
                    (second_gradient - second_target).double().square().mean().item()
                ),
                "second_moment_relative_mse": float(
                    (second_gradient - second_target).double().square().mean().item()
                    / max(second_target_energy, eps)
                ),
                "adam_mse": float(
                    (adam_gradient - adam_target).double().square().mean().item()
                ),
                "adam_relative_mse": float(
                    (adam_gradient - adam_target).double().square().mean().item()
                    / max(adam_target_energy, eps)
                ),
                "legacy_adam_mse": float(
                    (legacy_gradient - legacy_target).double().square().mean().item()
                ),
                "legacy_adam_relative_mse": float(
                    (legacy_gradient - legacy_target).double().square().mean().item()
                    / max(legacy_target_energy, eps)
                ),
                "fresh_adam_relative_mse": float(
                    (fresh_gradient - fresh_target).double().square().mean().item()
                    / max(fresh_target_energy, eps)
                ),
            })
        self.rows[name].append(raw)

    def summarize(self) -> Dict[str, dict]:
        output: Dict[str, dict] = {}
        for name, rows in self.rows.items():
            fields = rows[0].keys()
            output[name] = {}
            for field in fields:
                values = np.asarray([row[field] for row in rows], dtype=np.float64)
                output[name][field] = float(values.mean())
                output[name][f"{field}_sd"] = float(
                    values.std(ddof=1) if values.size > 1 else 0.0
                )
                output[name][f"{field}_se"] = float(
                    output[name][f"{field}_sd"] / math.sqrt(max(values.size, 1))
                )
            output[name]["samples"] = len(rows)
        exact_raw = max(output.get("exact", {}).get("mse", float("nan")), 1e-300)
        geometry_fields = (
            "clip_mse",
            "prepared_mse",
            "momentum_mse",
            "second_moment_mse",
            "legacy_adam_mse",
            "adam_mse",
        )
        exact_geometry = {
            field: max(output.get("exact", {}).get(field, float("nan")), 1e-300)
            for field in geometry_fields
        }
        exact_rows = self.rows.get("exact", [])
        for name, item in output.items():
            item["risk_over_exact"] = float(item["mse"] / exact_raw)
            for field in geometry_fields:
                if field in item:
                    prefix = field[: -len("_mse")]
                    item[f"{prefix}_risk_over_exact"] = float(
                        item[field] / exact_geometry[field]
                    )
            rows = self.rows[name]
            if len(rows) == len(exact_rows) and rows:
                for field in ("mse",) + geometry_fields:
                    if field not in rows[0] or field not in exact_rows[0]:
                        continue
                    differences = np.asarray(
                        [row[field] - ref[field] for row, ref in zip(rows, exact_rows)],
                        dtype=np.float64,
                    )
                    item[f"paired_{field}_difference"] = float(differences.mean())
                    item[f"paired_{field}_difference_se"] = float(
                        differences.std(ddof=1) / math.sqrt(differences.size)
                        if differences.size > 1 else 0.0
                    )
                    item[f"paired_{field}_win_fraction"] = float(
                        np.mean(differences < 0.0)
                    )
        return output


def _evaluate(
    bundle,
    args,
    parameters: Sequence[torch.Tensor],
    eval_starts: Sequence[Sequence[int]],
    coefficients: torch.Tensor,
    innovation_std_normalized: torch.Tensor,
    normalization_mean: float,
    normalization_std: float,
    plugin_coefficients: torch.Tensor,
    route_oracle_coefficients: torch.Tensor,
    global_conditioned_coefficients: Optional[torch.Tensor],
    global_weights: np.ndarray,
    adam_map: Optional[AdamUpdateMap],
    static_values: Sequence[float],
    tbptt_periods: Sequence[int],
) -> Tuple[Dict[str, dict], float, Dict[str, List[dict]]]:
    controller = bundle.dw
    open_coefficients = torch.ones_like(controller.coefficients)
    global_weights_t = torch.as_tensor(
        global_weights, device=bundle.xt.device, dtype=torch.float32
    )
    accumulator = MetricAccumulator()
    maximum_forward_difference = 0.0

    candidate_specs = []
    for value in static_values:
        candidate_specs.append((f"static_{value:g}", "static", float(value)))
    for period in tbptt_periods:
        candidate_specs.append((f"tbptt_{period}", "tbptt", int(period)))
    candidate_specs.extend([
        ("plugin_dw", "coefficients", plugin_coefficients),
        ("route_oracle_dw", "coefficients", route_oracle_coefficients),
    ])
    if global_conditioned_coefficients is not None:
        candidate_specs.append(
            (
                "global_conditioned_route_dw",
                "coefficients",
                global_conditioned_coefficients,
            )
        )

    for draw_index, starts in enumerate(eval_starts):
        conditional, noisy = _conditional_and_noisy_targets(
            bundle,
            starts,
            coefficients,
            innovation_std_normalized,
            normalization_mean,
            normalization_std,
            args.K,
            args.noise_draws,
            args.seed + 300007 * (draw_index + 1),
        )
        with _routing_operator(controller, open_coefficients):
            predictions = _rollout_predictions(bundle, starts, args)
            reference_predictions = [prediction.detach().clone() for prediction in predictions]
            clean_target = _gradient_vector(
                _total_mse(predictions, conditional), parameters, retain_graph=True
            )
            for noise_index, targets in enumerate(noisy):
                keep_after = noise_index + 1 < len(noisy)
                matrix = _gradient_matrix(
                    _horizon_losses(predictions, targets),
                    parameters,
                    retain_after=keep_after,
                )
                exact_gradient = matrix.sum(dim=0)
                global_gradient = global_weights_t @ matrix
                accumulator.add("exact", exact_gradient, clean_target, adam_map)
                accumulator.add(
                    "global_horizon_oracle", global_gradient, clean_target, adam_map
                )
                del matrix, exact_gradient, global_gradient
            del predictions

        for name, kind, value in candidate_specs:
            if kind == "static":
                candidate_coefficients = torch.full_like(
                    controller.coefficients, float(value)
                )
                const_gain = float(value)
                period = 0
            elif kind == "tbptt":
                candidate_coefficients = open_coefficients
                const_gain = None
                period = int(value)
            else:
                candidate_coefficients = value
                const_gain = None
                period = 0
            with _routing_operator(
                controller,
                candidate_coefficients,
                const_gain=const_gain,
                tbptt_period=period,
            ):
                predictions = _rollout_predictions(bundle, starts, args)
                maximum_forward_difference = max(
                    maximum_forward_difference,
                    max(float((prediction.detach() - reference).abs().max().item())
                        for prediction, reference in zip(predictions, reference_predictions)),
                )
                for noise_index, targets in enumerate(noisy):
                    gradient = _gradient_vector(
                        _total_mse(predictions, targets),
                        parameters,
                        retain_graph=noise_index + 1 < len(noisy),
                    )
                    accumulator.add(name, gradient, clean_target, adam_map)
                    del gradient
                del predictions
        del clean_target, reference_predictions
        print(
            f"[heldout evaluate] draw {draw_index + 1}/{len(eval_starts)} "
            f"noise_draws={len(noisy)} arms={2 + len(candidate_specs)}",
            flush=True,
        )
    return accumulator.summarize(), maximum_forward_difference, dict(accumulator.rows)


def _print_results(results: Mapping[str, Mapping[str, float]]) -> None:
    print("\n=== held-out complete parameter-gradient risk ===")
    header = (
        f"{'method':>24s} {'raw relMSE':>11s} {'risk/exact':>11s} "
        f"{'cos':>8s} {'norm':>8s} {'Adam relMSE':>12s} {'Adam/exact':>11s} "
        f"{'freshAdam':>10s}"
    )
    print(header)
    print("-" * len(header))
    ordered = sorted(results, key=lambda name: results[name]["mse"])
    for name in ordered:
        item = results[name]
        print(
            f"{name:>24s} {item['relative_mse']:11.5f} "
            f"{item['risk_over_exact']:11.5f} {item['cosine']:8.4f} "
            f"{item['norm_ratio']:8.4f} "
            f"{item.get('adam_relative_mse', float('nan')):12.5f} "
            f"{item.get('adam_risk_over_exact', float('nan')):11.5f} "
            f"{item.get('fresh_adam_relative_mse', float('nan')):10.5f}"
        )


def _print_optimizer_decomposition(
    results: Mapping[str, Mapping[str, float]]
) -> None:
    if not results or "adam_risk_over_exact" not in next(iter(results.values())):
        return
    print("\n=== optimizer decomposition: held-out risk / Exact ===")
    header = (
        f"{'method':>24s} {'raw':>8s} {'clip':>8s} {'+L2':>8s} "
        f"{'momentum':>9s} {'second':>8s} {'old-map':>8s} {'full':>8s}"
    )
    print(header)
    print("-" * len(header))
    for name in sorted(results, key=lambda key: results[key]["mse"]):
        item = results[name]
        print(
            f"{name:>24s} {item['risk_over_exact']:8.4f} "
            f"{item.get('clip_risk_over_exact', float('nan')):8.4f} "
            f"{item.get('prepared_risk_over_exact', float('nan')):8.4f} "
            f"{item.get('momentum_risk_over_exact', float('nan')):9.4f} "
            f"{item.get('second_moment_risk_over_exact', float('nan')):8.4f} "
            f"{item.get('legacy_adam_risk_over_exact', float('nan')):8.4f} "
            f"{item.get('adam_risk_over_exact', float('nan')):8.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--oracle-file", required=True)
    parser.add_argument("--fit-draws", type=int, default=2)
    parser.add_argument("--eval-draws", type=int, default=2)
    parser.add_argument("--noise-draws", type=int, default=8)
    parser.add_argument("--static-values", default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--tbptt-periods", default="2,4,8,16,32")
    parser.add_argument(
        "--same-state-oracle",
        action="store_true",
        help=(
            "fit route/global oracle weights on independent noise draws from "
            "the same observed state-set used for held-out evaluation"
        ),
    )
    parser.add_argument(
        "--global-conditioned-routes",
        action="store_true",
        help=(
            "fit route pairs by block coordinate descent on complete "
            "parameter-gradient risk; this is substantially more expensive "
            "than the local route oracle"
        ),
    )
    parser.add_argument(
        "--global-route-sweeps",
        type=int,
        default=1,
        help="number of global route coordinate sweeps",
    )
    parser.add_argument(
        "--global-route-noise-draws",
        type=int,
        default=0,
        help="innovation draws used by route BCD; 0 reuses --noise-draws",
    )
    parser.add_argument(
        "--global-route-max-routes",
        type=int,
        default=0,
        help="debug cap on updated routes per sweep; 0 updates all K*depth routes",
    )
    parser.add_argument(
        "--global-route-ridge",
        type=float,
        default=1e-6,
        help=(
            "ridge relative to each coordinate Gram trace, centered on the "
            "current route pair"
        ),
    )
    parser.add_argument(
        "--global-route-init",
        choices=("route_oracle", "plugin", "open", "half"),
        default="route_oracle",
    )
    parser.add_argument(
        "--global-route-order",
        choices=("forward", "reverse", "alternating"),
        default="alternating",
    )
    parser.add_argument(
        "--global-route-affine-check-every",
        type=int,
        default=16,
        help=(
            "verify G=C+alpha*A+m*B every N coordinate updates; 0 disables "
            "the model-specific closure check"
        ),
    )
    parser.add_argument(
        "--global-route-affine-tolerance",
        type=float,
        default=1e-5,
        help="maximum relative-MSE tolerated by the affine route closure check",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if not args.npz or not args.state_key:
        raise SystemExit("--npz and --state-key are required")
    if args.fit_draws < 1 or args.eval_draws < 1 or args.noise_draws < 2:
        raise SystemExit("fit/eval draws must be positive and noise-draws must be >=2")
    if args.global_route_sweeps < 1:
        raise SystemExit("--global-route-sweeps must be >=1")
    if args.global_route_noise_draws == 1 or args.global_route_noise_draws < 0:
        raise SystemExit("--global-route-noise-draws must be 0 or >=2")
    if args.global_route_max_routes < 0:
        raise SystemExit("--global-route-max-routes must be >=0")
    if args.global_route_ridge < 0:
        raise SystemExit("--global-route-ridge must be nonnegative")
    if args.global_route_affine_check_every < 0:
        raise SystemExit("--global-route-affine-check-every must be >=0")
    if args.global_route_affine_tolerance <= 0:
        raise SystemExit("--global-route-affine-tolerance must be positive")
    args.draws = 1 if args.same_state_oracle else int(
        args.fit_draws + args.eval_draws
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    normalization_mean, normalization_std, dataset_coefficients = _normalization(
        args.npz,
        args.state_key,
        args.seed,
        args.split_train_ratio,
        args.split_val_ratio,
    )
    bundle = setup(args)
    process_coefficients, innovation_std_normalized = _load_process(
        args.oracle_file, normalization_std, args.device
    )
    if not np.allclose(
        dataset_coefficients,
        process_coefficients.detach().cpu().numpy(),
        atol=1e-6,
    ):
        raise SystemExit("dataset coefficients disagree with --oracle-file")
    names, parameters = _trainable_parameters(bundle.model)
    parameter_count = sum(int(parameter.numel()) for parameter in parameters)
    print(
        f"[model] trainable tensors={len(parameters)} coordinates={parameter_count:,}",
        flush=True,
    )
    adam_map = AdamUpdateMap.from_checkpoint(
        args.ckpt, parameters, torch.device(args.device)
    )
    if adam_map is None:
        print("[Adam] checkpoint has no optimizer state; Adam columns will be absent")
    else:
        print(
            f"[Adam] kind={adam_map.optimizer_kind} "
            f"checkpoint step={adam_map.next_step - 1} "
            f"next={adam_map.next_step} lr={adam_map.learning_rate:g} "
            f"betas=({adam_map.beta1:g},{adam_map.beta2:g}) "
            f"weight_decay={adam_map.weight_decay:g} "
            f"grad_clip={adam_map.grad_clip:g}",
            flush=True,
        )

    if args.same_state_oracle:
        fit_starts = bundle.draw_starts[:1]
        eval_starts = bundle.draw_starts[:1]
        print(
            "[oracle scope] same observed state-set; fit/eval use independent "
            "Monte Carlo innovation draws",
            flush=True,
        )
    else:
        fit_starts = bundle.draw_starts[: args.fit_draws]
        eval_starts = bundle.draw_starts[args.fit_draws :]
        print("[oracle scope] fit/eval use disjoint observed state-sets", flush=True)
    plugin_coefficients = bundle.dw.coefficients.detach().clone()

    route_oracle_coefficients, route_summary = _fit_routewise_oracle(
        bundle,
        args,
        fit_starts,
        process_coefficients,
        innovation_std_normalized,
        normalization_mean,
        normalization_std,
    )
    global_weights, global_summary = _fit_global_horizon_oracle(
        bundle,
        args,
        parameters,
        fit_starts,
        process_coefficients,
        innovation_std_normalized,
        normalization_mean,
        normalization_std,
    )
    global_conditioned_coefficients = None
    global_conditioned_summary = None
    if args.global_conditioned_routes:
        global_conditioned_coefficients, global_conditioned_summary = (
            _fit_global_conditioned_routes(
                bundle,
                args,
                parameters,
                fit_starts,
                process_coefficients,
                innovation_std_normalized,
                normalization_mean,
                normalization_std,
                plugin_coefficients,
                route_oracle_coefficients,
            )
        )

    static_values = _parse_floats(args.static_values)
    tbptt_periods = _parse_ints(args.tbptt_periods)
    results, maximum_forward_difference, per_realization_metrics = _evaluate(
        bundle,
        args,
        parameters,
        eval_starts,
        process_coefficients,
        innovation_std_normalized,
        normalization_mean,
        normalization_std,
        plugin_coefficients,
        route_oracle_coefficients,
        global_conditioned_coefficients,
        global_weights,
        adam_map,
        static_values,
        tbptt_periods,
    )
    _print_results(results)
    _print_optimizer_decomposition(results)

    static_names = [f"static_{value:g}" for value in static_values]
    tbptt_names = [f"tbptt_{period}" for period in tbptt_periods]
    best_static = min(static_names, key=lambda name: results[name]["mse"])
    best_tbptt = min(tbptt_names, key=lambda name: results[name]["mse"])
    raw_order = sorted(results, key=lambda name: results[name]["mse"])
    adam_order = sorted(
        results,
        key=lambda name: results[name].get("adam_mse", float("inf")),
    )
    print(
        f"\n[oracle envelopes] best static={best_static} "
        f"risk/exact={results[best_static]['risk_over_exact']:.4f}; "
        f"best TBPTT={best_tbptt} "
        f"risk/exact={results[best_tbptt]['risk_over_exact']:.4f}"
    )
    print("[raw ranking]  " + " < ".join(raw_order))
    if adam_map is not None:
        print("[Adam ranking] " + " < ".join(adam_order))
        rank_changed = raw_order != adam_order
        print(f"[Adam effect] ranking_changed={rank_changed}")

    output = {
        "format_version": 3,
        "checkpoint": os.path.abspath(args.ckpt),
        "data": os.path.abspath(args.npz),
        "oracle_file": os.path.abspath(args.oracle_file),
        "K": int(args.K),
        "batch": int(bundle.batch),
        "fit_draws": int(args.fit_draws),
        "eval_draws": int(args.eval_draws),
        "oracle_scope": (
            "same_state_independent_noise_split"
            if args.same_state_oracle else "disjoint_state_sets"
        ),
        "noise_draws": int(args.noise_draws),
        "parameter_tensors": len(parameters),
        "parameter_coordinates": parameter_count,
        "parameter_names": names,
        "normalization_mean": normalization_mean,
        "normalization_std": normalization_std,
        "routewise_oracle": route_summary,
        "global_conditioned_route_oracle": global_conditioned_summary,
        "global_horizon_oracle": global_summary,
        "static_values": static_values,
        "tbptt_periods": tbptt_periods,
        "best_static_oracle_envelope": best_static,
        "best_tbptt_oracle_envelope": best_tbptt,
        "maximum_forward_difference_across_backward_operators": maximum_forward_difference,
        "results": results,
        "per_realization_metrics": per_realization_metrics,
        "raw_ranking": raw_order,
        "adam_ranking": adam_order if adam_map is not None else None,
        "adam_ranking_changed": raw_order != adam_order if adam_map is not None else None,
        "adam": None if adam_map is None else {
            "checkpoint_step": adam_map.next_step - 1,
            "next_step": adam_map.next_step,
            "learning_rate": adam_map.learning_rate,
            "betas": [adam_map.beta1, adam_map.beta2],
            "eps": adam_map.eps,
            "weight_decay": adam_map.weight_decay,
            "optimizer_kind": adam_map.optimizer_kind,
            "grad_clip": adam_map.grad_clip,
            "full_pipeline_order": "global_norm_clip_then_coupled_L2_for_Adam_then_moments",
            "legacy_map_excludes_clip_and_weight_decay": True,
        },
        "interpretation_contract": {
            "target": "exact full-BPTT gradient of MSE to E[x_{t+k}|x_t]",
            "route_oracle": "local 2x2 gains fit on disjoint starts and applied through the complete shipped backward graph",
            "global_conditioned_route_oracle": (
                "exact 2x2 coordinate updates fitted against complete "
                "parameter-gradient risk while all other routes are fixed"
            ),
            "global_oracle": "joint box-constrained horizon-loss weights with all cross-horizon parameter-gradient covariance retained",
            "selection_note": "best static/TBPTT labels are oracle probe envelopes over the displayed grids, not validation-selected paper results",
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(output), handle, indent=2, sort_keys=True)
    print(f"\n[out] {os.path.abspath(args.out)}", flush=True)


if __name__ == "__main__":
    main()
