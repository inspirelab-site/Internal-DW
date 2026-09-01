#!/usr/bin/env python
"""Frozen-checkpoint low-rank residual-covariance probe for Dual-Wiener.

The current diagonal estimator and a structured residual bootstrap are two
extreme covariance models.  This probe interpolates between them without
training a model:

    C_r = U_r diag(lambda_r) U_r^T
          + diag(diag(C) - diag(U_r diag(lambda_r) U_r^T)).

The residual covariance ``C`` is estimated from fully autoregressive rollout
residuals on the training split.  A separate validation split supplies the
fully-open total route VJPs.  For every requested rank, Gaussian covectors with
covariance ``C_r`` are pushed through exactly the same retained graph.  The
result therefore diagnoses covariance misspecification only; it is not a
training result and does not select a rank from validation performance.

PCA is computed through the sample-space Gram matrix, so the script never
forms the enormous output-space covariance matrix.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from internal_dw.data_utils.state_ops import unpack_batch, zero_external_input_like
from internal_dw.datasets.registry import build_dataloaders
from internal_dw.models.dual_wiener import solve_box_wiener_2x2
from internal_dw.models.registry import build_model
from internal_dw.utils import load_checkpoint, unwrap_model
from probe_gradient_dynamics import _infer_thewell_shape, _namespace
from probe_wiener_oracle import RouteCapture, covector_loss, push


def _metadata_item(metadata, key: str, index: int):
    if not isinstance(metadata, dict) or key not in metadata:
        return None
    value = metadata[key]
    if torch.is_tensor(value):
        value = value[index]
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def _sample_candidates(
    loader, args, K: int, count: int, seed: int, *, unique_groups: bool = False
):
    """Take at most one start per loaded trajectory to limit overlap."""
    candidates = []
    window = int(getattr(args, "window_size", 1))
    rng = random.Random(int(seed))
    seen_groups = set()
    for batch_index, batch in enumerate(loader):
        state, stim, _, metadata = unpack_batch(batch)
        for sample_index in range(int(state.shape[0])):
            trajectory = state[sample_index : sample_index + 1].cpu()
            drive = (
                stim[sample_index : sample_index + 1].cpu()
                if stim is not None else None
            )
            low = window
            high = int(trajectory.shape[1]) - int(K)
            if high < low:
                continue
            path = _metadata_item(metadata, "path", sample_index)
            trajectory_index = _metadata_item(
                metadata, "trajectory_index", sample_index
            )
            group_id = (
                f"{path}::trajectory={trajectory_index}"
                if path is not None and trajectory_index is not None
                else f"batch={batch_index}::sample={sample_index}"
            )
            if unique_groups and group_id in seen_groups:
                continue
            seen_groups.add(group_id)
            candidates.append(
                {
                    "state": trajectory,
                    "stim": drive,
                    "start": rng.randint(low, high),
                    "batch_index": int(batch_index),
                    "sample_index": int(sample_index),
                    "group_id": group_id,
                    "metadata": str(metadata),
                }
            )
            if len(candidates) >= int(count):
                return candidates
    return candidates


def _rollout(raw, item, args, K: int, device: torch.device, graph: bool):
    state = item["state"].to(device=device, dtype=torch.float32)
    stim = item["stim"]
    if stim is None:
        stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))
    else:
        stim = stim.to(device=device, dtype=torch.float32)

    start = int(item["start"])
    window = int(getattr(args, "window_size", 1))
    history = state[:, start - window : start]
    predictions, targets = [], []
    context = torch.enable_grad() if graph else torch.no_grad()
    with context:
        for horizon in range(int(K)):
            target_t = start + horizon
            stim_window = stim[:, target_t - window : target_t]
            prediction = raw(
                stim_window,
                history,
                return_aux=False,
                horizon_index=horizon,
                total_horizon=K,
            )
            predictions.append(prediction)
            targets.append(state[:, target_t : target_t + 1])
            history = torch.cat([history[:, 1:], prediction], dim=1)
    return predictions, targets


def _fit_sample_space_pca(residuals: torch.Tensor, tolerance: float = 1e-6):
    """Fit joint horizon/space PCA to ``residuals[N,K,...]`` on CPU."""
    if residuals.ndim < 3 or residuals.shape[0] < 2:
        raise ValueError("PCA requires residuals shaped [N,K,...] with N >= 2")
    matrix = residuals.reshape(residuals.shape[0], -1).float()
    mean = matrix.mean(dim=0)
    centered = matrix - mean
    n = int(centered.shape[0])
    diagonal = centered.square().mean(dim=0)
    gram = centered @ centered.t() / float(n)
    values, vectors = torch.linalg.eigh(0.5 * (gram + gram.t()))
    order = torch.argsort(values, descending=True)
    # Centering limits the empirical rank to N-1.  Enforcing that algebraic
    # limit also removes the tiny positive eigenvalue left by float32 roundoff.
    values = values[order][: max(n - 1, 0)].clamp_min(0.0)
    vectors = vectors[:, order][:, : max(n - 1, 0)]
    threshold = max(float(values[0]) * float(tolerance), 1e-20)
    keep = values > threshold
    values = values[keep]
    vectors = vectors[:, keep]
    if values.numel() == 0:
        components = centered.new_zeros((0, centered.shape[1]))
    else:
        components = (vectors.t() @ centered) / torch.sqrt(
            float(n) * values[:, None]
        )
        components = torch.nn.functional.normalize(components, dim=1)
    return {
        "mean": mean.contiguous(),
        "diagonal": diagonal.contiguous(),
        "eigenvalues": values.contiguous(),
        "components": components.contiguous(),
        "output_shape": tuple(int(x) for x in residuals.shape[1:]),
    }


def _diagonal_remainder(pca, rank: int) -> torch.Tensor:
    rank = min(max(int(rank), 0), int(pca["eigenvalues"].numel()))
    if rank == 0:
        explained_diagonal = torch.zeros_like(pca["diagonal"])
    else:
        explained_diagonal = (
            pca["eigenvalues"][:rank, None]
            * pca["components"][:rank].square()
        ).sum(dim=0)
    # Roundoff can make a handful of entries slightly negative at full rank.
    return (pca["diagonal"] - explained_diagonal).clamp_min(0.0)


def _noise_sample(pca, rank: int, epsilon: torch.Tensor, z: torch.Tensor):
    rank = min(max(int(rank), 0), int(pca["eigenvalues"].numel()))
    sample = epsilon * torch.sqrt(_diagonal_remainder(pca, rank))
    if rank:
        weights = z[:rank] * torch.sqrt(pca["eigenvalues"][:rank])
        sample = sample + weights @ pca["components"][:rank]
    return sample.reshape(pca["output_shape"])


def _add_moment(store: dict[int, list[Any]], route_index: int, pair) -> None:
    identity, branch = pair
    stacked = torch.stack([identity, branch]).reshape(2, -1)
    gram = (stacked @ stacked.t()).detach().cpu().double()
    count = int(stacked.shape[1])
    route_index = int(route_index)
    if route_index not in store:
        store[route_index] = [gram, count]
    else:
        store[route_index][0] += gram
        store[route_index][1] += count


def _means(store: dict[int, list[Any]]) -> dict[int, torch.Tensor]:
    return {route: value[0] / max(int(value[1]), 1) for route, value in store.items()}


def _configure_open_route_graph(raw):
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


def _reset_graph_state(dw) -> None:
    dw._slot = 0
    dw._root_refs = []
    dw._pair_store = {}


def _restore_route_graph(raw, dw, saved) -> None:
    if saved["routing"] is not None:
        raw.resgrad_routing = saved["routing"]
    if saved["policy"] is not None:
        raw.resgrad_policy = saved["policy"]
    dw._collecting = saved["collecting"]
    dw._mode = saved["mode"]
    dw._slot = saved["slot"]
    dw._root_refs = saved["roots"]
    dw._pair_store = saved["pairs"]


def _route_summary(total, noises, ranks, depth, checkpoint_gains):
    common = set(total)
    for rank in ranks:
        common &= set(noises[rank])
    routes = sorted(common)
    if not routes:
        raise RuntimeError("no route VJPs were captured")

    route_rows = []
    rank_summary = {}
    for rank in ranks:
        alpha, branch, fractions = [], [], []
        for route in routes:
            T = total[route]
            R = noises[rank][route]
            gain = solve_box_wiener_2x2(T, R).cpu().double()
            fraction = float(torch.trace(R) / torch.trace(T).clamp_min(1e-30))
            horizon, layer = divmod(int(route), int(depth))
            row = {
                "rank": int(rank),
                "route_index": int(route),
                "horizon_index": int(horizon),
                "layer_index": int(layer),
                "alpha": float(gain[0]),
                "m": float(gain[1]),
                "noise_fraction": fraction,
                "total_moment": T.tolist(),
                "noise_moment": R.tolist(),
            }
            route_rows.append(row)
            alpha.append(row["alpha"])
            branch.append(row["m"])
            fractions.append(fraction)
        rank_summary[str(rank)] = {
            "routes": len(routes),
            "alpha_mean": float(np.mean(alpha)),
            "alpha_sd": float(np.std(alpha)),
            "m_mean": float(np.mean(branch)),
            "m_sd": float(np.std(branch)),
            "noise_fraction_mean": float(np.mean(fractions)),
            "noise_fraction_median": float(np.median(fractions)),
        }

    checkpoint_values = []
    for route in routes:
        horizon, layer = divmod(int(route), int(depth))
        if horizon < checkpoint_gains.shape[0] and layer < checkpoint_gains.shape[1]:
            checkpoint_values.append(checkpoint_gains[horizon, layer].tolist())
    checkpoint_summary = None
    if checkpoint_values:
        values = np.asarray(checkpoint_values, dtype=np.float64)
        checkpoint_summary = {
            "routes": int(values.shape[0]),
            "alpha_mean": float(values[:, 0].mean()),
            "m_mean": float(values[:, 1].mean()),
        }
    return route_rows, rank_summary, checkpoint_summary


def _make_plot(path: Path, ranks, explained, summaries, checkpoint_summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(ranks))
    alpha = [summaries[str(rank)]["alpha_mean"] for rank in ranks]
    branch = [summaries[str(rank)]["m_mean"] for rank in ranks]
    fraction = [summaries[str(rank)]["noise_fraction_mean"] for rank in ranks]
    labels = [f"{rank}\n({100 * explained[str(rank)]:.0f}%)" for rank in ranks]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    axes[0].plot(x, alpha, "o-", label=r"$\alpha$")
    axes[0].plot(x, branch, "s-", label=r"$m$")
    if checkpoint_summary is not None:
        axes[0].axhline(checkpoint_summary["alpha_mean"], color="C0", ls=":", lw=1)
        axes[0].axhline(checkpoint_summary["m_mean"], color="C1", ls=":", lw=1)
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("mean route gain")
    axes[0].legend(frameon=False)
    axes[1].plot(x, fraction, "o-", color="C3")
    axes[1].axhline(1.0, color="0.4", ls="--", lw=1)
    axes[1].set_ylabel(r"mean $\mathrm{tr}(R)/\mathrm{tr}(T)$")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.set_xlabel("PCA rank (explained residual variance)")
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--pca_samples", type=int, default=32)
    parser.add_argument("--probe_starts", type=int, default=4)
    parser.add_argument("--noise_draws", type=int, default=2)
    parser.add_argument("--ranks", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 31])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    cli = parser.parse_args()

    torch.manual_seed(int(cli.seed))
    np.random.seed(int(cli.seed))
    checkpoint = torch.load(cli.ckpt, map_location="cpu", weights_only=False)
    args = _namespace(checkpoint["args"])
    args.num_workers = 0
    args.local_batch_size = 1
    train_loader, val_loader, _ = build_dataloaders(args, rank=0, world_size=1)
    _infer_thewell_shape(args, train_loader)
    if str(getattr(args, "model_name", "")) != "unet_field":
        raise ValueError("this first probe intentionally supports The-Well unet_field checkpoints only")

    device = torch.device(f"cuda:{int(cli.gpu)}" if torch.cuda.is_available() else "cpu")
    model = build_model(args, rank=0)
    load_checkpoint(model, cli.ckpt, map_location=str(device), strict=True)
    raw = unwrap_model(model).to(device).eval()
    for parameter in raw.parameters():
        parameter.requires_grad_(False)
    dw = getattr(raw, "dual_wiener", None)
    if dw is None:
        raise RuntimeError("the checkpoint has no Dual-Wiener controller")

    K = min(int(cli.K), int(dw.max_horizon))
    pca_pool = _sample_candidates(train_loader, args, K, cli.pca_samples, cli.seed + 11)
    probe_pool = _sample_candidates(val_loader, args, K, cli.probe_starts, cli.seed + 29)
    if len(pca_pool) < 2 or not probe_pool:
        raise RuntimeError(
            f"insufficient feasible trajectories: pca={len(pca_pool)}, probe={len(probe_pool)}"
        )

    print(
        f"[setup] checkpoint_epoch={checkpoint.get('epoch')} K={K} "
        f"pca_samples={len(pca_pool)} probe_starts={len(probe_pool)} device={device}",
        flush=True,
    )
    residual_batches = []
    for index, item in enumerate(pca_pool):
        predictions, targets = _rollout(raw, item, args, K, device, graph=False)
        residual = torch.stack(
            [(prediction - target).squeeze(0) for prediction, target in zip(predictions, targets)]
        ).cpu()
        residual_batches.append(residual)
        print(f"[residual library] {index + 1}/{len(pca_pool)}", flush=True)
    residuals = torch.stack(residual_batches)
    pca = _fit_sample_space_pca(residuals)
    empirical_rank = int(pca["eigenvalues"].numel())
    ranks = sorted({min(max(int(rank), 0), empirical_rank) for rank in cli.ranks} | {0, empirical_rank})
    total_variance = float(pca["diagonal"].sum())
    explained = {
        str(rank): (
            float(pca["eigenvalues"][:rank].sum()) / max(total_variance, 1e-30)
            if rank else 0.0
        )
        for rank in ranks
    }
    print(
        f"[PCA] output_coordinates={pca['diagonal'].numel()} empirical_rank={empirical_rank} "
        + " ".join(f"r{rank}={100*explained[str(rank)]:.1f}%" for rank in ranks),
        flush=True,
    )

    total_store: dict[int, list[Any]] = {}
    noise_stores: dict[int, dict[int, list[Any]]] = {rank: {} for rank in ranks}
    checkpoint_gains = dw.coefficients[:K].detach().cpu().numpy()
    capture = None
    dw, saved = _configure_open_route_graph(raw)
    capture = RouteCapture(dw)
    try:
        for start_index, item in enumerate(probe_pool):
            _reset_graph_state(dw)
            predictions, targets = _rollout(raw, item, args, K, device, graph=True)
            roots = list(dw._root_refs)
            if not roots:
                raise RuntimeError("the fully-open graph exposed no autograd probe root")
            residual_vecs = [(prediction - target).detach() for prediction, target in zip(predictions, targets)]
            actual_pairs = push(covector_loss(predictions, residual_vecs), roots, capture, retain=True)
            for route, pair in actual_pairs.items():
                _add_moment(total_store, route, pair)

            generator = torch.Generator(device="cpu").manual_seed(
                int(cli.seed) * 100003 + start_index * 997 + 101
            )
            for draw in range(int(cli.noise_draws)):
                epsilon = torch.randn(pca["diagonal"].shape, generator=generator)
                z = torch.randn(empirical_rank, generator=generator)
                for rank_index, rank in enumerate(ranks):
                    sample = _noise_sample(pca, rank, epsilon, z)
                    vecs = [
                        sample[horizon : horizon + 1].to(device=device, dtype=predictions[horizon].dtype)
                        for horizon in range(K)
                    ]
                    last = (
                        start_index == len(probe_pool) - 1
                        and draw == int(cli.noise_draws) - 1
                        and rank_index == len(ranks) - 1
                    )
                    pairs = push(
                        covector_loss(predictions, vecs), roots, capture, retain=not last
                    )
                    for route, pair in pairs.items():
                        _add_moment(noise_stores[rank], route, pair)
            print(
                f"[route probe] {start_index + 1}/{len(probe_pool)} "
                f"captured_routes={len(actual_pairs)}",
                flush=True,
            )
            del predictions, targets, roots, residual_vecs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if capture is not None:
            capture.close()
        _restore_route_graph(raw, dw, saved)

    total = _means(total_store)
    noises = {rank: _means(store) for rank, store in noise_stores.items()}
    route_rows, summaries, checkpoint_summary = _route_summary(
        total, noises, ranks, int(dw.depth), checkpoint_gains
    )

    output = {
        "format_version": 1,
        "probe": "frozen_checkpoint_joint_residual_pca_wiener",
        "checkpoint": str(cli.ckpt),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset": str(getattr(args, "dataset", "")),
        "thewell_dataset_name": str(getattr(args, "thewell_dataset_name", "")),
        "model_name": str(getattr(args, "model_name", "")),
        "K": K,
        "pca_split": "train",
        "gain_probe_split": "validation",
        "pca_samples": len(pca_pool),
        "probe_starts": len(probe_pool),
        "noise_draws_per_start": int(cli.noise_draws),
        "residual_coordinates": int(pca["diagonal"].numel()),
        "empirical_rank": empirical_rank,
        "ranks": ranks,
        "explained_variance_fraction": explained,
        "rank_summary": summaries,
        "checkpoint_gain_summary": checkpoint_summary,
        "route_rows": route_rows,
        "interpretation_scope": (
            "Estimator diagnostic only: ranks are not selected by forecast performance, "
            "and residual covariance is not identified with irreducible noise."
        ),
    }
    out = Path(cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    np.savez_compressed(
        out.with_suffix(".npz"),
        eigenvalues=pca["eigenvalues"].numpy(),
        ranks=np.asarray(ranks),
        residual_diagonal=pca["diagonal"].numpy(),
    )
    _make_plot(out, ranks, explained, summaries, checkpoint_summary)

    print("\n=== low-rank covariance route gains ===", flush=True)
    print(" rank  explained     alpha        m   noise_frac", flush=True)
    for rank in ranks:
        row = summaries[str(rank)]
        print(
            f"{rank:5d}  {100*explained[str(rank)]:8.2f}%  "
            f"{row['alpha_mean']:8.4f}  {row['m_mean']:8.4f}  "
            f"{row['noise_fraction_mean']:10.4f}",
            flush=True,
        )
    if checkpoint_summary is not None:
        print(
            f" ckpt              {checkpoint_summary['alpha_mean']:8.4f}  "
            f"{checkpoint_summary['m_mean']:8.4f}",
            flush=True,
        )
    print(f"[out] {out.with_suffix('.json')}", flush=True)
    print(f"[out] {out.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    main()
