#!/usr/bin/env python
"""Controlled artificial-noise response for fMRI Internal DW-Prior.

The forecasting checkpoint is frozen.  The subject-crossfit innovation-template
bank is split into disjoint calibration and evaluation halves.  Evaluation
templates define the true noise process; calibration templates are the only
noise samples available to the DW-Prior plug-in.  The noise law and its scale
remain fixed, while the clean route signal is rescaled to requested trace SNR.

This prevents the artificial SNR from being handed to the estimator.  The
probe reports open, oracle, and Prior-gain risks on the true held-out process.
It performs no training and no optimizer update.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "probes"))

from internal_dw.models.dual_wiener import solve_box_wiener_2x2  # noqa: E402
from gradient_probe_ops import (  # noqa: E402
    add_route_moment,
    configure_open_route_graph,
    mean_route_moments,
    reset_route_graph,
    restore_route_graph,
)
from probe_setup import add_common_args, setup  # noqa: E402
from probe_wiener_oracle import (  # noqa: E402
    RouteCapture,
    covector_loss,
    push,
    risk_analytic,
    rollout,
)


def _load_banks(path: str, seed: int, K: int, D: int):
    with np.load(path, allow_pickle=False) as archive:
        if "innovation_templates" not in archive.files:
            raise ValueError(f"{path} lacks innovation_templates")
        raw = np.asarray(archive["innovation_templates"], dtype=np.float32)
        name = (
            str(np.asarray(archive["innovation_estimator_name"]).reshape(()).item())
            if "innovation_estimator_name" in archive.files
            else "subject_crossfit_template_bootstrap"
        )
    if raw.ndim < 3 or raw.shape[1] < K:
        raise ValueError(f"templates must be [N,K,...] with K>={K}, got {raw.shape}")
    raw = raw[:, :K].reshape(raw.shape[0], K, -1)
    if raw.shape[2] != D:
        raise ValueError(f"template D={raw.shape[2]} does not match model D={D}")
    if raw.shape[0] < 8 or not np.isfinite(raw).all():
        raise ValueError(f"invalid template bank {raw.shape}")
    rng = np.random.default_rng(int(seed) + 29009)
    order = rng.permutation(raw.shape[0])
    cut = raw.shape[0] // 2
    fit = raw[order[:cut]].copy()
    evaluation = raw[order[cut:]].copy()
    # Use the calibration mean only.  Random signs below make both empirical
    # bootstrap processes exactly zero-mean without consulting evaluation data.
    center = fit.mean(axis=0, keepdims=True)
    fit -= center
    evaluation -= center
    return fit, evaluation, name


def _draw_templates(bank: np.ndarray, batch: int, rng: np.random.Generator):
    index = rng.integers(0, bank.shape[0], size=int(batch))
    sign = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=(batch, 1, 1))
    return np.ascontiguousarray(bank[index] * sign, dtype=np.float32)


def _vecs(matrix: torch.Tensor, predictions):
    return [matrix[k].reshape(1, -1).expand_as(predictions[k]) for k in range(len(predictions))]


def _signal_from_residual_mean(bundle, args) -> torch.Tensor:
    rows = []
    for index, starts in enumerate(bundle.draw_starts):
        predictions, _roots, targets = rollout(bundle, starts, args)
        residual = torch.stack(
            [(prediction - target).detach() for prediction, target in zip(predictions, targets)],
            dim=1,
        )
        rows.append(residual.cpu())
        print(f"[clean signal] draw {index + 1}/{len(bundle.draw_starts)}", flush=True)
    stacked = torch.cat(rows, dim=0)  # [draw*B,K,D]
    signal = stacked.mean(dim=0)
    if not torch.isfinite(signal).all() or float(signal.square().mean()) <= 0.0:
        raise RuntimeError("clean residual-mean signal is non-finite or degenerate")
    return signal


def _summary(total, signal, true_noise, prior_noise, depth: int):
    routes = sorted(set(total) & set(signal) & set(true_noise) & set(prior_noise))
    if not routes:
        raise RuntimeError("no common internal routes captured")
    accum = {
        name: {"alpha": [], "m": [], "risk": 0.0, "error": []}
        for name in ("open", "oracle", "prior")
    }
    rows = []
    for route in routes:
        P = signal[route].cpu().double().numpy()
        Rtrue = true_noise[route].cpu().double().numpy()
        T = total[route]
        Rprior = prior_noise[route]
        oracle = solve_box_wiener_2x2(
            torch.as_tensor(P + Rtrue), torch.as_tensor(Rtrue)
        ).cpu().double().numpy()
        prior = solve_box_wiener_2x2(T, Rprior).cpu().double().numpy()
        candidates = {
            "open": np.ones(2, dtype=np.float64),
            "oracle": oracle,
            "prior": prior,
        }
        item = {
            "route_index": int(route),
            "horizon_index": int(route) // int(depth),
            "layer_index": int(route) % int(depth),
        }
        for name, gain in candidates.items():
            risk = float(risk_analytic(P, Rtrue, gain))
            accum[name]["alpha"].append(float(gain[0]))
            accum[name]["m"].append(float(gain[1]))
            accum[name]["risk"] += risk
            if name == "prior":
                accum[name]["error"].append(float(np.abs(gain - oracle).mean()))
            item[name] = {"alpha": float(gain[0]), "m": float(gain[1]), "risk": risk}
        rows.append(item)
    open_risk = max(accum["open"]["risk"], 1e-300)
    report = {}
    for name, values in accum.items():
        report[name] = {
            "routes": len(routes),
            "alpha_mean": float(np.mean(values["alpha"])),
            "m_mean": float(np.mean(values["m"])),
            "risk_over_open": float(values["risk"] / open_risk),
        }
        if values["error"]:
            report[name]["mean_abs_gain_error_to_oracle"] = float(np.mean(values["error"]))
    return report, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--snr", type=float, required=True)
    parser.add_argument("--noise-draws", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not np.isfinite(args.snr) or args.snr <= 0:
        raise SystemExit("--snr must be finite and positive")
    if args.noise_draws < 2:
        raise SystemExit("--noise-draws must be >=2")

    # Build the exact Prior controller used by the checkpoint.  The probe also
    # samples the same artifact manually so fit/evaluation banks can be split.
    os.environ["DUAL_WIENER_INNOVATION_FILE"] = os.path.abspath(args.artifact)
    os.environ["DUAL_WIENER_INNOVATION_KEY"] = "innovation_templates"
    args.dual_wiener_noise_model = "lagged_residual_bootstrap"
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    bundle = setup(args)
    model = bundle.model
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    fit_bank, evaluation_bank, estimator_name = _load_banks(
        args.artifact, int(args.seed), int(args.K), int(bundle.state_dim)
    )
    clean_base = _signal_from_residual_mean(bundle, args)
    base_signal_energy = float(clean_base.double().square().mean())
    true_noise_energy = float(np.mean(evaluation_bank.astype(np.float64) ** 2))
    if base_signal_energy <= 0.0 or true_noise_energy <= 0.0:
        raise RuntimeError(
            f"degenerate energies signal={base_signal_energy} noise={true_noise_energy}"
        )
    signal_scale = float(np.sqrt(float(args.snr) * true_noise_energy / base_signal_energy))
    clean_signal = clean_base * signal_scale
    achieved_snr = float(clean_signal.double().square().mean()) / true_noise_energy
    print(
        f"[controlled] requested_snr={args.snr:g} achieved_snr={achieved_snr:.6g} "
        f"signal_scale={signal_scale:.6g} fitN={len(fit_bank)} evalN={len(evaluation_bank)}",
        flush=True,
    )

    total_store, signal_store, true_store, prior_store = {}, {}, {}, {}
    dw, saved = configure_open_route_graph(model)
    capture = RouteCapture(dw)
    try:
        for draw_index, starts in enumerate(bundle.draw_starts):
            dw.begin_batch()
            reset_route_graph(dw)
            predictions, roots, _targets = rollout(bundle, starts, args)
            if not roots:
                raise RuntimeError("rollout exposed no autograd roots")
            signal_device = clean_signal.to(device=predictions[0].device, dtype=predictions[0].dtype)
            signal_pairs = push(
                covector_loss(predictions, _vecs(signal_device, predictions)),
                roots,
                capture,
                retain=True,
            )
            for route, pair in signal_pairs.items():
                add_route_moment(signal_store, route, pair)

            batch = int(predictions[0].shape[0])
            true_rng = np.random.default_rng(int(args.seed) + 100003 * (draw_index + 1))
            fit_rng = np.random.default_rng(int(args.seed) + 170003 * (draw_index + 1))
            for noise_index in range(int(args.noise_draws)):
                true_np = _draw_templates(evaluation_bank, batch, true_rng)
                prior_np = _draw_templates(fit_bank, batch, fit_rng)
                true_noise = torch.as_tensor(
                    true_np, device=predictions[0].device, dtype=predictions[0].dtype
                )
                prior_noise = torch.as_tensor(
                    prior_np, device=predictions[0].device, dtype=predictions[0].dtype
                )
                total = signal_device.unsqueeze(0) + true_noise
                total_pairs = push(
                    covector_loss(predictions, [total[:, k] for k in range(args.K)]),
                    roots,
                    capture,
                    retain=True,
                )
                true_pairs = push(
                    covector_loss(predictions, [true_noise[:, k] for k in range(args.K)]),
                    roots,
                    capture,
                    retain=True,
                )
                last = (
                    draw_index == len(bundle.draw_starts) - 1
                    and noise_index == int(args.noise_draws) - 1
                )
                prior_pairs = push(
                    covector_loss(predictions, [prior_noise[:, k] for k in range(args.K)]),
                    roots,
                    capture,
                    retain=not last,
                )
                for route, pair in total_pairs.items():
                    add_route_moment(total_store, route, pair)
                for route, pair in true_pairs.items():
                    add_route_moment(true_store, route, pair)
                for route, pair in prior_pairs.items():
                    add_route_moment(prior_store, route, pair)
            print(f"[route] draw {draw_index + 1}/{len(bundle.draw_starts)}", flush=True)
            del predictions, roots
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        capture.close()
        restore_route_graph(model, dw, saved)

    summary, route_rows = _summary(
        mean_route_moments(total_store), mean_route_moments(signal_store),
        mean_route_moments(true_store), mean_route_moments(prior_store),
        int(bundle.depth),
    )
    output = {
        "format_version": 1,
        "probe": "fmri_internal_dw_prior_controlled_noise_response",
        "semantics": {
            "training": "none; frozen checkpoint and backward-only routes",
            "snr_control": "fixed Prior noise law; rescale clean signal only",
            "prior_fit": "first disjoint half of randomly permuted template bank",
            "true_evaluation_noise": "second disjoint half of template bank",
            "primary_metric": "analytic held-out local route-message risk/open",
        },
        "checkpoint": os.path.abspath(args.ckpt),
        "artifact": os.path.abspath(args.artifact),
        "innovation_estimator_name": estimator_name,
        "requested_snr": float(args.snr),
        "achieved_snr": achieved_snr,
        "signal_scale": signal_scale,
        "fit_templates": int(len(fit_bank)),
        "evaluation_templates": int(len(evaluation_bank)),
        "K": int(args.K),
        "draws": int(len(bundle.draw_starts)),
        "noise_draws": int(args.noise_draws),
        "summary": summary,
        "routes": route_rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    for name in ("oracle", "prior"):
        row = summary[name]
        extra = (
            f" error={row['mean_abs_gain_error_to_oracle']:.4f}"
            if "mean_abs_gain_error_to_oracle" in row else ""
        )
        print(
            f"{name:>8}: a/m={row['alpha_mean']:.4f}/{row['m_mean']:.4f} "
            f"risk/open={row['risk_over_open']:.4f}{extra}"
        )
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
