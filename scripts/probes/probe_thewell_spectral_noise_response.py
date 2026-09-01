#!/usr/bin/env python
"""Controlled noise-response audit for Internal Spectral-DW on The Well.

The checkpoint and forward model are frozen.  A controlled local residual
world is constructed from training trajectories:

* the clean route signal is the fitted per-horizon residual mean;
* the true innovation is a random-phase Gaussian with the fitted centered
  spatial spectrum;
* its amplitude is rescaled to the requested trace SNR;
* diagonal and spatial-spectrum plug-ins are refit from the contaminated
  calibration residuals and pushed through the same open route graph.

This is deliberately a mechanism diagnostic, not a forecasting benchmark.
It tests whether the field estimator closes the internal Jacobian gates as
innovation strength increases and whether its oracle-evaluated route risk
falls below the fully-open route.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from internal_dw.datasets.registry import build_dataloaders
from internal_dw.models.dual_wiener import solve_box_wiener_2x2
from internal_dw.models.registry import build_model
from internal_dw.utils import load_checkpoint, unwrap_model
from gradient_probe_ops import (
    add_route_moment,
    configure_open_route_graph,
    infer_thewell_shape,
    mean_route_moments,
    namespace,
    reset_route_graph,
    restore_route_graph,
)
from probe_data_ops import (
    fit_spectral_covariance,
    rollout_field,
    sample_field_candidates,
    sample_spectral_noise,
    sample_unique_field_trajectories,
)
from probe_wiener_oracle import RouteCapture, covector_loss, push, risk_analytic


def _vecs(tensor: torch.Tensor, K: int):
    return [tensor[h : h + 1].unsqueeze(0) for h in range(K)]


def _route_summary(total, signal, true_noise, plugin_noise, depth: int):
    common = set(total) & set(signal) & set(true_noise)
    for store in plugin_noise.values():
        common &= set(store)
    routes = sorted(common)
    if not routes:
        raise RuntimeError("no common internal routes were captured")
    variants = ("spatial_spectrum",)
    accum = {
        name: {"alpha": [], "m": [], "risk": 0.0, "oracle_error": []}
        for name in ("open", "oracle") + variants
    }
    rows = []
    for route in routes:
        P = signal[route].cpu().double().numpy()
        Rt = true_noise[route].cpu().double().numpy()
        T = total[route]
        oracle = solve_box_wiener_2x2(
            torch.as_tensor(P + Rt), torch.as_tensor(Rt)
        ).cpu().double().numpy()
        candidates = {
            "open": np.ones(2, dtype=np.float64),
            "oracle": oracle,
        }
        for name in variants:
            candidates[name] = solve_box_wiener_2x2(
                T, plugin_noise[name][route]
            ).cpu().double().numpy()
        route_row = {
            "route_index": int(route),
            "horizon_index": int(route) // int(depth),
            "layer_index": int(route) % int(depth),
        }
        for name, gain in candidates.items():
            risk = float(risk_analytic(P, Rt, gain))
            accum[name]["alpha"].append(float(gain[0]))
            accum[name]["m"].append(float(gain[1]))
            accum[name]["risk"] += risk
            if name not in ("open", "oracle"):
                accum[name]["oracle_error"].append(float(np.abs(gain - oracle).mean()))
            route_row[name] = {
                "alpha": float(gain[0]), "m": float(gain[1]), "risk": risk,
            }
        rows.append(route_row)
    open_risk = max(float(accum["open"]["risk"]), 1e-300)
    summary = {}
    for name, values in accum.items():
        summary[name] = {
            "routes": len(routes),
            "alpha_mean": float(np.mean(values["alpha"])),
            "m_mean": float(np.mean(values["m"])),
            "risk_over_open": float(values["risk"] / open_risk),
        }
        if values["oracle_error"]:
            summary[name]["mean_abs_gain_error_to_oracle"] = float(
                np.mean(values["oracle_error"])
            )
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--snr", type=float, required=True)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--fit-trajectories", type=int, default=48)
    parser.add_argument("--probe-trajectories", type=int, default=8)
    parser.add_argument("--noise-draws", type=int, default=8)
    parser.add_argument("--variance-floor", type=float, default=1e-8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    cli = parser.parse_args()
    if not np.isfinite(cli.snr) or cli.snr <= 0:
        raise SystemExit("--snr must be finite and positive")

    torch.manual_seed(int(cli.seed))
    np.random.seed(int(cli.seed))
    checkpoint = torch.load(cli.ckpt, map_location="cpu", weights_only=False)
    args = namespace(checkpoint["args"])
    args.num_workers = 0
    args.local_batch_size = 1
    train_loader, val_loader, _ = build_dataloaders(args, rank=0, world_size=1)
    infer_thewell_shape(args, train_loader)
    if str(getattr(args, "model_name", "")) != "unet_field":
        raise ValueError("controlled spectral audit currently requires unet_field")
    device = torch.device(f"cuda:{int(cli.gpu)}" if torch.cuda.is_available() else "cpu")
    model = build_model(args, rank=0)
    load_checkpoint(model, cli.ckpt, map_location=str(device), strict=True)
    raw = unwrap_model(model).to(device).eval()
    for parameter in raw.parameters():
        parameter.requires_grad_(False)
    dw = getattr(raw, "dual_wiener", None)
    if dw is None:
        raise RuntimeError("checkpoint/model has no Internal Dual-Wiener controller")
    K = min(int(cli.K), int(dw.max_horizon))

    if str(getattr(args, "dataset", "")) == "the_well":
        fit_pool = sample_unique_field_trajectories(
            train_loader.dataset, args, K, int(cli.fit_trajectories), int(cli.seed) + 607
        )
        probe_pool = sample_unique_field_trajectories(
            val_loader.dataset, args, K, int(cli.probe_trajectories), int(cli.seed) + 701
        )
        sampling_axis = "unique physical trajectories"
    else:
        # WeatherBench2 exposes independent fixed-length forecast segments via
        # its loader rather than The-Well trajectory objects.  Take one start
        # per sampled segment; train and validation loaders remain disjoint.
        fit_pool = sample_field_candidates(
            train_loader, args, K, int(cli.fit_trajectories), int(cli.seed) + 607,
            unique_groups=False,
        )
        probe_pool = sample_field_candidates(
            val_loader, args, K, int(cli.probe_trajectories), int(cli.seed) + 701,
            unique_groups=False,
        )
        sampling_axis = "disjoint loader segments"
    if len(fit_pool) < 2 or not probe_pool:
        raise RuntimeError(f"insufficient trajectories: fit={len(fit_pool)} probe={len(probe_pool)}")

    residual_batches = []
    for index, item in enumerate(fit_pool):
        predictions, targets = rollout_field(
            raw, item, args, K, device, graph=False
        )
        residual_batches.append(torch.stack(
            [(prediction - target)[0, 0] for prediction, target in zip(predictions, targets)]
        ).cpu())
        print(f"[fit residual] {index + 1}/{len(fit_pool)}", flush=True)
    residuals = torch.stack(residual_batches).to(device)
    clean_signal = residuals.mean(dim=0)
    centered = residuals - clean_signal.unsqueeze(0)
    signal_energy = float(clean_signal.square().mean())
    base_noise_energy = float(centered.square().mean())
    if signal_energy <= 0 or base_noise_energy <= 0:
        raise RuntimeError(
            f"degenerate controlled moments: signal={signal_energy} noise={base_noise_energy}"
        )
    noise_scale = float(np.sqrt(signal_energy / (float(cli.snr) * base_noise_energy)))
    injected_fit = clean_signal.unsqueeze(0) + noise_scale * centered
    base_model = fit_spectral_covariance(centered, float(cli.variance_floor))
    plugin_model = fit_spectral_covariance(injected_fit, float(cli.variance_floor))
    achieved_snr = signal_energy / max(noise_scale * noise_scale * base_noise_energy, 1e-300)
    print(
        f"[controlled] requested_snr={cli.snr:g} achieved_snr={achieved_snr:.6g} "
        f"noise_scale={noise_scale:.6g}", flush=True,
    )

    # WB2 calibration tensors are [N,K,C,H,W] and occupy about 5 GiB each at
    # N=32.  The fitted spectrum dictionaries contain independent compact
    # tensors, so keeping residuals, centered residuals, and injected samples
    # alive while constructing the K-step autograd graph only wastes memory.
    del residuals, centered, injected_fit
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total_store, signal_store, true_noise_store = {}, {}, {}
    plugin_stores = {"spatial_spectrum": {}}
    dw, saved = configure_open_route_graph(raw)
    capture = RouteCapture(dw)
    try:
        for start_index, item in enumerate(probe_pool):
            reset_route_graph(dw)
            predictions, _targets = rollout_field(
                raw, item, args, K, device, graph=True
            )
            roots = list(dw._root_refs)
            if not roots:
                raise RuntimeError("open route graph exposed no autograd root")
            signal_pairs = push(
                covector_loss(predictions, _vecs(clean_signal, K)), roots, capture, retain=True
            )
            for route, pair in signal_pairs.items():
                add_route_moment(signal_store, route, pair)

            generator = torch.Generator(device=device).manual_seed(
                int(cli.seed) * 100003 + start_index * 1013 + 1907
            )
            shape = tuple(int(x) for x in clean_signal.shape)
            for draw in range(int(cli.noise_draws)):
                epsilon = torch.randn(shape, device=device, generator=generator)
                true_noise = noise_scale * sample_spectral_noise(base_model, epsilon)
                total = clean_signal + true_noise
                total_pairs = push(
                    covector_loss(predictions, _vecs(total, K)), roots, capture, retain=True
                )
                true_pairs = push(
                    covector_loss(predictions, _vecs(true_noise, K)), roots, capture, retain=True
                )
                for route, pair in total_pairs.items():
                    add_route_moment(total_store, route, pair)
                for route, pair in true_pairs.items():
                    add_route_moment(true_noise_store, route, pair)
                plugin_noise = sample_spectral_noise(plugin_model, epsilon)
                last = (
                    start_index == len(probe_pool) - 1
                    and draw == int(cli.noise_draws) - 1
                )
                pairs = push(
                    covector_loss(predictions, _vecs(plugin_noise, K)),
                    roots, capture, retain=not last,
                )
                for route, pair in pairs.items():
                    add_route_moment(
                        plugin_stores["spatial_spectrum"], route, pair
                    )
            print(f"[route] {start_index + 1}/{len(probe_pool)}", flush=True)
            del predictions, roots
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        capture.close()
        restore_route_graph(raw, dw, saved)

    total = mean_route_moments(total_store)
    signal = mean_route_moments(signal_store)
    true_noise = mean_route_moments(true_noise_store)
    plugin_noise = {
        name: mean_route_moments(store)
        for name, store in plugin_stores.items()
    }
    summary, rows = _route_summary(total, signal, true_noise, plugin_noise, int(dw.depth))
    output = {
        "format_version": 1,
        "probe": "thewell_internal_spectral_dw_controlled_noise_response",
        "checkpoint": str(cli.ckpt),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset": str(getattr(args, "dataset", "")),
        "thewell_dataset_name": str(getattr(args, "thewell_dataset_name", "")),
        "K": K,
        "requested_snr": float(cli.snr),
        "achieved_snr": float(achieved_snr),
        "noise_scale": noise_scale,
        "fit_unique_trajectories": len(fit_pool),
        "probe_unique_trajectories": len(probe_pool),
        "sampling_axis": sampling_axis,
        "noise_draws_per_probe": int(cli.noise_draws),
        "variant_summary": summary,
        "route_rows": rows,
        "semantics": (
            "Controlled local route experiment. The fitted train residual mean is clean signal; "
            "a centered random-phase spatial Gaussian is injected innovation. risk/open is "
            "computed against the known controlled signal/noise moments."
        ),
    }
    out = Path(cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("\n estimator          alpha/m       risk/open  |gain-oracle|", flush=True)
    for name in ("oracle", "spatial_spectrum"):
        row = summary[name]
        error = row.get("mean_abs_gain_error_to_oracle", 0.0)
        print(
            f" {name:17s} {row['alpha_mean']:.3f}/{row['m_mean']:.3f}  "
            f"{row['risk_over_open']:9.4f}  {error:.4f}", flush=True,
        )
    print(f"[out] {out}", flush=True)


if __name__ == "__main__":
    main()
