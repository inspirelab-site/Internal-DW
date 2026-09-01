#!/usr/bin/env python
"""Controlled noise-response audit for Internal iEEG DW-Prior.

The prior's cross-horizon innovation-template bank defines a fixed noise law.
For each requested SNR we vary only the clean route-signal strength.  The
template law itself is not rescaled or refit, so this does not leak the
requested SNR into the estimator.  Disjoint halves of the bank provide the
plug-in and evaluation innovations.

The forecasting checkpoint is frozen and every forward pass is identical;
only backward route covectors are measured.
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

from internal_dw.models.dual_wiener import solve_box_wiener_2x2
from gradient_probe_ops import add_route_moment, mean_route_moments
from probe_setup import add_common_args, setup
from probe_wiener_oracle import (
    RouteCapture,
    covector_loss,
    push,
    risk_analytic,
    rollout,
)


def _load_templates(path: str, K: int, state_dim: int, seed: int):
    with np.load(path, allow_pickle=False) as archive:
        if "innovation_templates" not in archive.files:
            raise ValueError(f"{path} has no innovation_templates")
        templates = np.asarray(archive["innovation_templates"], dtype=np.float32)
        estimator = (
            str(np.asarray(archive["innovation_estimator_name"]).reshape(()).item())
            if "innovation_estimator_name" in archive.files
            else "domain_conditional_innovation_bootstrap"
        )
    if templates.ndim != 3 or templates.shape[1] < K or templates.shape[2] != state_dim:
        raise ValueError(
            f"expected innovation_templates [N,K,{state_dim}] with K>={K}, "
            f"got {templates.shape}"
        )
    if templates.shape[0] < 8 or not np.isfinite(templates).all():
        raise ValueError("innovation-template bank is too small or non-finite")
    rng = np.random.default_rng(int(seed) + 9187)
    order = rng.permutation(templates.shape[0])
    split = len(order) // 2
    fit = templates[order[:split], :K].astype(np.float64)
    evaluation = templates[order[split:], :K].astype(np.float64)
    center = fit.mean(axis=0, keepdims=True)
    fit = (fit - center).astype(np.float32)
    evaluation = (evaluation - center).astype(np.float32)
    return fit, evaluation, estimator


def _sample_bank(
    bank: torch.Tensor,
    batch: int,
    generator: torch.Generator,
) -> torch.Tensor:
    indices = torch.randint(
        int(bank.shape[0]), (int(batch),), generator=generator, device=bank.device
    )
    signs = torch.empty(
        (int(batch), 1, 1), device=bank.device, dtype=bank.dtype
    ).bernoulli_(0.5, generator=generator).mul_(2.0).sub_(1.0)
    return bank.index_select(0, indices) * signs


def _add_pairs(store, pairs) -> None:
    for route, pair in pairs.items():
        add_route_moment(store, route, pair)


def _route_summary(total, signal, true_noise, prior_noise, depth: int):
    routes = sorted(set(total) & set(signal) & set(true_noise) & set(prior_noise))
    if not routes:
        raise RuntimeError("no common internal routes were captured")
    accum = {
        name: {"alpha": [], "m": [], "risk": 0.0, "error": []}
        for name in ("open", "oracle", "prior")
    }
    rows = []
    for route in routes:
        P = signal[route].cpu().double().numpy()
        Rt = true_noise[route].cpu().double().numpy()
        T = total[route]
        Rp = prior_noise[route]
        oracle = solve_box_wiener_2x2(
            torch.as_tensor(P + Rt), torch.as_tensor(Rt)
        ).cpu().double().numpy()
        prior = solve_box_wiener_2x2(T, Rp).cpu().double().numpy()
        candidates = {
            "open": np.ones(2, dtype=np.float64),
            "oracle": oracle,
            "prior": prior,
        }
        row = {
            "route_index": int(route),
            "horizon_index": int(route) // int(depth),
            "layer_index": int(route) % int(depth),
        }
        for name, gain in candidates.items():
            risk = float(risk_analytic(P, Rt, gain))
            values = accum[name]
            values["alpha"].append(float(gain[0]))
            values["m"].append(float(gain[1]))
            values["risk"] += risk
            if name == "prior":
                values["error"].append(float(np.abs(gain - oracle).mean()))
            row[name] = {"alpha": float(gain[0]), "m": float(gain[1]), "risk": risk}
        rows.append(row)
    open_risk = max(float(accum["open"]["risk"]), 1e-300)
    summary = {}
    for name, values in accum.items():
        item = {
            "routes": len(routes),
            "alpha_mean": float(np.mean(values["alpha"])),
            "m_mean": float(np.mean(values["m"])),
            "risk_over_open": float(values["risk"] / open_risk),
        }
        if values["error"]:
            item["mean_abs_gain_error_to_oracle"] = float(np.mean(values["error"]))
        summary[name] = item
    return summary, rows


def _base_clean_signal(bundle, args) -> torch.Tensor:
    residual_sum = None
    count = 0
    saved_collecting, saved_mode = bundle.dw._collecting, bundle.dw._mode
    try:
        bundle.dw._collecting = False
        bundle.dw._mode = "apply"
        for index, starts in enumerate(bundle.draw_starts):
            predictions, _roots, targets = rollout(bundle, starts, args)
            residual = torch.stack(
                [(prediction - target).detach().mean(dim=0) for prediction, target in zip(predictions, targets)]
            )
            residual_sum = residual if residual_sum is None else residual_sum + residual
            count += 1
            print(f"[signal] draw {index + 1}/{len(bundle.draw_starts)}", flush=True)
            del predictions, targets, _roots
        if residual_sum is None:
            raise RuntimeError("no rollout draw for clean signal")
        return residual_sum / float(count)
    finally:
        bundle.dw._collecting, bundle.dw._mode = saved_collecting, saved_mode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--snr", type=float, required=True)
    parser.add_argument("--noise-draws", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not np.isfinite(args.snr) or args.snr <= 0:
        raise SystemExit("--snr must be positive")
    if args.noise_draws < 2:
        raise SystemExit("--noise-draws must be at least 2")

    # The prior bank is handled explicitly below; prevent inherited variables
    # from silently selecting a different artifact while the model is built.
    os.environ.pop("DUAL_WIENER_INNOVATION_FILE", None)
    os.environ.pop("DUAL_WIENER_INNOVATION_KEY", None)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    bundle = setup(args)
    fit_np, eval_np, estimator = _load_templates(
        args.artifact, int(args.K), int(bundle.state_dim), int(args.seed)
    )
    fit_bank = torch.as_tensor(fit_np, device=args.device)
    eval_bank = torch.as_tensor(eval_np, device=args.device)
    base_signal = _base_clean_signal(bundle, args).to(args.device)
    base_energy = float(base_signal.square().mean())
    noise_energy = float(eval_bank.square().mean())
    if base_energy <= 0 or noise_energy <= 0:
        raise RuntimeError(
            f"degenerate signal/noise energy: signal={base_energy} noise={noise_energy}"
        )
    signal_scale = float(np.sqrt(float(args.snr) * noise_energy / base_energy))
    clean_signal = signal_scale * base_signal
    achieved_snr = float(clean_signal.square().mean() / max(noise_energy, 1e-300))
    print(
        f"[controlled] requested_snr={args.snr:g} achieved_snr={achieved_snr:.6g} "
        f"signal_scale={signal_scale:.6g} prior={estimator}", flush=True,
    )

    total_store, signal_store, true_store, prior_store = {}, {}, {}, {}
    dw = bundle.dw
    saved = (dw._collecting, dw._mode, dw._slot)
    capture = RouteCapture(dw)
    try:
        for draw_index, starts in enumerate(bundle.draw_starts):
            dw.begin_batch()
            dw._collecting = True
            dw._mode = "total"
            dw._slot = 0
            predictions, roots, _targets = rollout(bundle, starts, args)
            batch = int(predictions[0].shape[0])
            signal_vecs = [clean_signal[k].reshape(1, -1).expand_as(predictions[k]) for k in range(args.K)]
            _add_pairs(
                signal_store,
                push(covector_loss(predictions, signal_vecs), roots, capture, retain=True),
            )
            true_gen = torch.Generator(device=args.device).manual_seed(
                int(args.seed) * 100003 + draw_index * 1009 + 17
            )
            prior_gen = torch.Generator(device=args.device).manual_seed(
                int(args.seed) * 100003 + draw_index * 1009 + 53
            )
            for noise_index in range(int(args.noise_draws)):
                true_noise = _sample_bank(eval_bank, batch, true_gen)
                prior_noise = _sample_bank(fit_bank, batch, prior_gen)
                total_vecs = [
                    signal_vecs[k] + true_noise[:, k].reshape_as(predictions[k])
                    for k in range(args.K)
                ]
                true_vecs = [true_noise[:, k].reshape_as(predictions[k]) for k in range(args.K)]
                prior_vecs = [prior_noise[:, k].reshape_as(predictions[k]) for k in range(args.K)]
                _add_pairs(total_store, push(
                    covector_loss(predictions, total_vecs), roots, capture, retain=True
                ))
                _add_pairs(true_store, push(
                    covector_loss(predictions, true_vecs), roots, capture, retain=True
                ))
                last = (
                    draw_index == len(bundle.draw_starts) - 1
                    and noise_index == int(args.noise_draws) - 1
                )
                _add_pairs(prior_store, push(
                    covector_loss(predictions, prior_vecs), roots, capture, retain=not last
                ))
            print(
                f"[route] draw {draw_index + 1}/{len(bundle.draw_starts)} "
                f"noise={args.noise_draws}", flush=True,
            )
            del predictions, roots, _targets
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        capture.close()
        dw._collecting, dw._mode, dw._slot = saved

    summary, rows = _route_summary(
        mean_route_moments(total_store), mean_route_moments(signal_store),
        mean_route_moments(true_store), mean_route_moments(prior_store),
        int(bundle.depth),
    )
    output = {
        "format_version": 1,
        "probe": "ieeg_internal_dw_prior_controlled_noise_response",
        "semantics": {
            "noise_law": "fixed iEEG prior template law; SNR varies by clean-signal scale only",
            "crossfit": "disjoint template-bank halves for plug-in and evaluation noise",
            "scope": "frozen checkpoint; backward-only route mechanism probe; no training",
        },
        "checkpoint": str(args.ckpt),
        "artifact": str(args.artifact),
        "prior_estimator": estimator,
        "K": int(args.K),
        "requested_snr": float(args.snr),
        "achieved_snr": achieved_snr,
        "signal_scale": signal_scale,
        "fit_templates": int(fit_bank.shape[0]),
        "evaluation_templates": int(eval_bank.shape[0]),
        "noise_draws": int(args.noise_draws),
        "summary": summary,
        "routes": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("\nmethod        alpha       m    risk/open  |gate-oracle|")
    for name in ("oracle", "prior"):
        row = summary[name]
        error = row.get("mean_abs_gain_error_to_oracle", 0.0)
        print(
            f"{name:>8s}  {row['alpha_mean']:>10.4f} {row['m_mean']:>7.4f} "
            f"{row['risk_over_open']:>12.4f} {error:>14.4f}"
        )
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
