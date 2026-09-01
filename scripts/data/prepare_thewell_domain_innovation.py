#!/usr/bin/env python3
"""Cross-fit a physics-matched innovation bank for two The Well systems.

The conditional mean is deliberately domain specific:

* ``shear_flow`` is periodic in both spatial directions.  A complex transfer
  function is fit independently at every 2-D Fourier mode; its phase represents
  translation/advection and its magnitude represents scale-dependent damping.
* ``turbulent_radiative_layer_2D`` is periodic only in x and open in y.  We
  therefore Fourier-transform x only and retain the vertical coordinate.  This
  avoids the false stationarity assumption made by a 2-D random-phase model.

Held-out residual trajectories, not marginal variances, are saved.  Resampling
one complete trajectory preserves channel, spatial, and cross-horizon
covariance in the subsequent route-VJP noise probe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from internal_dw.datasets.thewell import build_thewell_splits  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("shear_flow", "turbulent_radiative_layer_2D"),
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-horizon", type=int, default=32)
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--spatial-subsample", type=int, default=0)
    parser.add_argument("--max-sequences", type=int, default=64)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--ridge-relative", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _dataset(a: argparse.Namespace):
    spatial_subsample = int(a.spatial_subsample)
    if spatial_subsample <= 0:
        spatial_subsample = 4 if a.dataset == "shear_flow" else 2
    args = SimpleNamespace(
        data_path=str(a.data_path),
        thewell_dataset_name=str(a.dataset),
        thewell_sequence_length=int(a.max_horizon) + int(a.window),
        thewell_sequence_stride=int(a.max_horizon) + 1,
        thewell_time_subsample=1,
        thewell_spatial_subsample=spatial_subsample,
        thewell_field_groups="t0_fields,t1_fields,t2_fields",
        thewell_max_trajectories_per_file=0,
        stim_dim=1,
    )
    train, _, _ = build_thewell_splits(args)
    return train, spatial_subsample


def _select_sequences(dataset, maximum: int, seed: int) -> torch.Tensor:
    count = min(int(maximum), len(dataset))
    if count < 8:
        raise ValueError(f"need at least 8 training sequences, found {len(dataset)}")
    rng = np.random.default_rng(int(seed))
    indices = np.sort(rng.choice(len(dataset), size=count, replace=False))
    rows = []
    for position, index in enumerate(indices, start=1):
        state = dataset[int(index)]["state"].float()
        rows.append(state)
        if position == 1 or position % 8 == 0 or position == count:
            print(
                f"[collect] {position}/{count} index={int(index)} "
                f"shape={tuple(state.shape)}",
                flush=True,
            )
    shapes = {tuple(row.shape) for row in rows}
    if len(shapes) != 1:
        raise ValueError(f"training sequences do not share one shape: {sorted(shapes)}")
    return torch.stack(rows)


def _transform(value: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "rfft2":
        return torch.fft.rfft2(value, dim=(-2, -1), norm="ortho")
    if kind == "rfft_x":
        return torch.fft.rfft(value, dim=-2, norm="ortho")
    raise ValueError(kind)


def _inverse(value: torch.Tensor, kind: str, height: int, width: int) -> torch.Tensor:
    if kind == "rfft2":
        return torch.fft.irfft2(
            value, s=(height, width), dim=(-2, -1), norm="ortho"
        )
    if kind == "rfft_x":
        return torch.fft.irfft(value, n=height, dim=-2, norm="ortho")
    raise ValueError(kind)


def crossfit_templates(
    sequences: torch.Tensor,
    *,
    transform: str,
    window: int,
    folds: int,
    ridge_relative: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    n, time, channels, height, width = sequences.shape
    window = int(window)
    if window <= 0 or time <= window:
        raise ValueError(f"invalid window={window} for sequence length {time}")
    horizon = time - window
    folds = max(2, min(int(folds), n))
    assignment = torch.arange(n) % folds
    output = torch.empty(n, horizon, channels, height, width)
    target_energy = torch.zeros(horizon, dtype=torch.float64)
    residual_energy = torch.zeros(horizon, dtype=torch.float64)

    for fold in range(folds):
        fit_idx = torch.where(assignment != fold)[0]
        held_idx = torch.where(assignment == fold)[0]
        fit = sequences.index_select(0, fit_idx).to(device)
        held = sequences.index_select(0, held_idx).to(device)
        fit_x = torch.stack(
            [_transform(fit[:, offset], transform) for offset in range(window)],
            dim=-1,
        )
        held_x = torch.stack(
            [_transform(held[:, offset], transform) for offset in range(window)],
            dim=-1,
        )
        # Batched complex normal equations, independently for every channel and
        # admissible spatial frequency.  With window=2 this is a tiny 2x2 solve
        # but, unlike a one-frame transfer function, matches the U-Net history.
        gram = (
            fit_x.conj().unsqueeze(-1) * fit_x.unsqueeze(-2)
        ).sum(dim=0)
        ridge = gram.diagonal(dim1=-2, dim2=-1).real.mean().clamp_min(1e-20)
        eye = torch.eye(window, device=device, dtype=gram.dtype)
        gram = gram + float(ridge_relative) * ridge * eye

        for step in range(horizon):
            fit_y = _transform(fit[:, window + step], transform)
            cross = (fit_x.conj() * fit_y.unsqueeze(-1)).sum(dim=0)
            transfer = torch.linalg.solve(gram, cross.unsqueeze(-1)).squeeze(-1)
            prediction = _inverse(
                (held_x * transfer.unsqueeze(0)).sum(dim=-1),
                transform,
                height,
                width,
            )
            residual = held[:, window + step] - prediction
            output[held_idx, step] = residual.cpu()
            target_energy[step] += held[:, window + step].double().square().sum().cpu()
            residual_energy[step] += residual.double().square().sum().cpu()
        print(
            f"[crossfit] fold={fold + 1}/{folds} "
            f"fit={len(fit_idx)} held={len(held_idx)}",
            flush=True,
        )
        del fit, held, fit_x, held_x

    ratios = residual_energy / target_energy.clamp_min(1e-30)
    diagnostics = {
        "conditional_residual_energy_fraction": ratios.tolist(),
        "mean_conditional_residual_energy_fraction": float(ratios.mean()),
        "last_conditional_residual_energy_fraction": float(ratios[-1]),
    }
    return output, diagnostics


def main() -> None:
    a = parse_args()
    if a.max_horizon <= 0 or a.max_sequences < 8:
        raise ValueError("max-horizon must be positive and max-sequences at least 8")
    dataset, spatial_subsample = _dataset(a)
    sequences = _select_sequences(dataset, int(a.max_sequences), int(a.seed))
    transform = "rfft2" if a.dataset == "shear_flow" else "rfft_x"
    templates, diagnostics = crossfit_templates(
        sequences,
        transform=transform,
        window=int(a.window),
        folds=int(a.folds),
        ridge_relative=float(a.ridge_relative),
        device=torch.device(a.device),
    )
    estimator = (
        "shear_periodic_transport_crossfit_bootstrap"
        if a.dataset == "shear_flow"
        else "radiative_layer_periodic_x_crossfit_bootstrap"
    )
    metadata = {
        "format_version": 1,
        "dataset": str(a.dataset),
        "fit_split": "train_only_cross_fitted",
        "conditional_model": (
            "per_mode_2d_complex_wiener_transport"
            if transform == "rfft2"
            else "per_x_mode_y_local_complex_wiener_transport"
        ),
        "boundary_assumption": (
            "periodic_x_and_y" if transform == "rfft2" else "periodic_x_open_y"
        ),
        "innovation_sampler": "whole_trajectory_empirical_bootstrap",
        "spatial_transform": transform,
        "spatial_subsample": int(spatial_subsample),
        "max_horizon": int(a.max_horizon),
        "conditioning_window": int(a.window),
        "sequences": int(templates.shape[0]),
        "folds": int(a.folds),
        "ridge_relative": float(a.ridge_relative),
        "seed": int(a.seed),
        **diagnostics,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    # float16 is sufficient for a covariance bootstrap and halves the artifact;
    # the controller promotes to float32 when loading.
    np.savez_compressed(
        a.output,
        innovation_templates=templates.numpy().astype(np.float16),
        innovation_estimator_name=np.asarray(estimator),
        spatial_transform=np.asarray(transform),
    )
    a.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[out] {a.output}")
    print(f"[out] {a.output.with_suffix('.json')}")
    print(
        "[domain innovation] dataset=%s transform=%s N=%d K=%d "
        "mean_residual_fraction=%.5f last=%.5f"
        % (
            a.dataset,
            transform,
            templates.shape[0],
            templates.shape[1],
            diagnostics["mean_conditional_residual_energy_fraction"],
            diagnostics["last_conditional_residual_energy_fraction"],
        )
    )


if __name__ == "__main__":
    main()
