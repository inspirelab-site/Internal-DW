"""Shared frozen-checkpoint gradient helpers for application diagnostics."""

from __future__ import annotations

import argparse
from typing import Any, Iterable, Optional

import torch


def namespace(saved: Any) -> argparse.Namespace:
    if isinstance(saved, argparse.Namespace):
        return saved
    if isinstance(saved, dict):
        return argparse.Namespace(**saved)
    raise TypeError(f"checkpoint args must be dict/Namespace, got {type(saved)!r}")


def snapshot_open_state(raw) -> dict[str, Any]:
    names = (
        "resgrad_routing",
        "resgrad_policy",
        "resgrad_block_gate",
        "resgrad_ratio_threshold",
        "resgrad_current_horizon",
        "resgrad_current_total_horizon",
    )
    return {name: getattr(raw, name) for name in names if hasattr(raw, name)}


def force_fully_open(raw) -> None:
    if hasattr(raw, "resgrad_routing"):
        raw.resgrad_routing = False
    if hasattr(raw, "resgrad_policy"):
        raw.resgrad_policy = "all"
    if hasattr(raw, "resgrad_block_gate"):
        raw.resgrad_block_gate = 1.0
    if hasattr(raw, "set_resgrad_context"):
        raw.set_resgrad_context(None, None)


def restore_open_state(raw, saved: dict[str, Any]) -> None:
    for name, value in saved.items():
        setattr(raw, name, value)


def add_route_moment(
    store: dict[int, list[Any]], route_index: int, pair
) -> None:
    """Accumulate the normalized 2x2 Gram matrix of one route-message pair."""
    identity, nonlinear = pair
    stacked = torch.stack([identity, nonlinear]).reshape(2, -1)
    gram = (stacked @ stacked.t()).detach().cpu().double()
    count = int(stacked.shape[1])
    route_index = int(route_index)
    if route_index not in store:
        store[route_index] = [gram, count]
    else:
        store[route_index][0] += gram
        store[route_index][1] += count


def mean_route_moments(
    store: dict[int, list[Any]],
) -> dict[int, torch.Tensor]:
    """Convert accumulated route Grams to per-coordinate second moments."""
    return {
        route: value[0] / max(int(value[1]), 1)
        for route, value in store.items()
    }


def configure_open_route_graph(raw):
    """Open Internal-DW while retaining its route-capture autograd nodes."""
    dw = getattr(raw, "dual_wiener", None)
    if dw is None:
        raise RuntimeError("checkpoint/model has no Dual-Wiener controller")
    saved = {
        "routing": getattr(raw, "resgrad_routing", None),
        "policy": getattr(raw, "resgrad_policy", None),
        "collecting": dw._collecting,
        "mode": dw._mode,
        "slot": dw._slot,
        "roots": dw._root_refs,
        "pairs": dw._pair_store,
    }
    raw.resgrad_routing = True
    raw.resgrad_policy = "dualwiener"
    dw._collecting = True
    dw._mode = "total"
    dw._slot = 0
    dw._root_refs = []
    dw._pair_store = {}
    return dw, saved


def reset_route_graph(dw) -> None:
    dw._slot = 0
    dw._root_refs = []
    dw._pair_store = {}


def restore_route_graph(raw, dw, saved: dict[str, Any]) -> None:
    if saved["routing"] is not None:
        raw.resgrad_routing = saved["routing"]
    if saved["policy"] is not None:
        raw.resgrad_policy = saved["policy"]
    dw._collecting = saved["collecting"]
    dw._mode = saved["mode"]
    dw._slot = saved["slot"]
    dw._root_refs = saved["roots"]
    dw._pair_store = saved["pairs"]


def detach_state(raw, state):
    if hasattr(raw, "detach_state"):
        return raw.detach_state(state)
    if torch.is_tensor(state):
        return state.detach()
    if isinstance(state, tuple):
        return tuple(detach_state(raw, item) for item in state)
    if isinstance(state, list):
        return [detach_state(raw, item) for item in state]
    return state


def burn_recurrent(raw, state, stimulus, start_t: int, burn: int):
    batch = int(state.shape[0])
    with torch.no_grad():
        hidden = raw.init_state(batch, state.device, state.dtype)
        burn_start = max(0, int(start_t) - int(burn))
        burn_end = max(burn_start, int(start_t) - 1)
        for index in range(burn_start, burn_end):
            stimulus_t = stimulus[:, index] if stimulus is not None else None
            _, hidden = raw.step(
                hidden, state[:, index], stimulus_t, return_aux=False
            )
    return detach_state(raw, hidden)


def step_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    batch = int(prediction.shape[0])
    delta = (prediction - target).reshape(batch, -1)
    if loss_type == "mse":
        return delta.pow(2).mean()
    target_flat = target.reshape(batch, -1)
    return (
        delta.norm(dim=1) / target_flat.norm(dim=1).clamp_min(1e-8)
    ).mean()


class GradientProjection:
    """Fixed coordinate projection without concatenating the full GPU gradient."""

    def __init__(self, parameters: list[torch.nn.Parameter], size: int, seed: int):
        self.params = parameters
        self.total = int(sum(parameter.numel() for parameter in parameters))
        requested = int(size)
        self.projected = requested > 0 and requested < self.total
        if not self.projected:
            self.size = self.total
            self.maps = None
            return

        self.size = requested
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        global_indices = torch.randint(
            0, self.total, (self.size,), generator=generator, dtype=torch.int64
        )
        self.unique_coordinates = int(torch.unique(global_indices).numel())
        self.maps: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        offset = 0
        for parameter_index, parameter in enumerate(parameters):
            end = offset + int(parameter.numel())
            mask = (global_indices >= offset) & (global_indices < end)
            output_positions = torch.nonzero(mask, as_tuple=False).flatten()
            if output_positions.numel():
                local_indices = global_indices[output_positions] - offset
                self.maps.append(
                    (parameter_index, output_positions, local_indices)
                )
            offset = end

    def apply(self, gradients: Iterable[Optional[torch.Tensor]]) -> torch.Tensor:
        gradients = list(gradients)
        if not self.projected:
            pieces = []
            for gradient, parameter in zip(gradients, self.params):
                value = gradient if gradient is not None else torch.zeros_like(parameter)
                pieces.append(value.detach().float().reshape(-1).cpu())
            return torch.cat(pieces)

        output = torch.zeros(self.size, dtype=torch.float32)
        assert self.maps is not None
        for parameter_index, output_positions, local_indices in self.maps:
            gradient = gradients[parameter_index]
            if gradient is None:
                continue
            selected = gradient.detach().reshape(-1)[local_indices.to(gradient.device)]
            output[output_positions] = selected.float().cpu()
        return output

    def metadata(self) -> dict[str, Any]:
        output = {
            "total_parameter_coordinates": self.total,
            "projected": bool(self.projected),
            "projection_size": self.size,
        }
        if self.projected:
            output["unique_projection_coordinates"] = self.unique_coordinates
            output["sampling"] = "fixed_uniform_with_replacement"
        else:
            output["sampling"] = "all_coordinates"
        return output


def infer_thewell_shape(args, train_loader) -> None:
    if str(getattr(args, "dataset", "")) != "the_well":
        return
    sample = train_loader.dataset[0]
    state = sample["state"] if isinstance(sample, dict) else sample[0]
    args.field_channels = int(state.shape[1])
    args.field_height = int(state.shape[2])
    args.field_width = int(state.shape[3])
    if str(getattr(args, "model_name", "")) in {
        "official_mamba_state",
        "official_pc_mamba_state",
        "official_atlas_mamba_state",
        "official_tangent_atlas_mamba_state",
        "official_shadow_perturb_mamba",
        "shadow_perturb_mamba",
        "official_mamba_fixeda_perturb",
        "cyclic_graph_ar",
    }:
        args.roi_dim = int(state[0].numel())
