"""Lorenz-96 chaotic ODE as a second, cross-domain autoregressive testbed.

    dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F,    i = 1..D  (cyclic)

At F=8 with D=40 this is the canonical spatiotemporally-chaotic toy system of the
data-assimilation / data-driven-forecasting literature: bounded attractor, leading
Lyapunov exponent ~1.67 (Lyapunov time ~0.6 MTU). A closed-loop rollout therefore
accumulates error exponentially -- exactly the regime ResGrad targets, and unlike
the diffusion-dominated PDEs (gray-scott, turbulent radiative layer) where
long-horizon error accumulation is weak and routing has no lever to pull.

It is deliberately a *different silo* from HCP fMRI: same weight-tied rollout
structure, unrelated domain, so reproducing the redundancy result here says the
finding is about autoregressive training, not about brain signals.

Integrated with classical RK4 at a small solver step and subsampled to the
autoregressive step ``l96_dt`` (this decouples solver accuracy from the AR
sampling rate). Generation is cheap, so trajectories are synthesized once and
cached to ``--data_path`` as an .npz.

Interface matches the other datasets: __getitem__ returns
    {"state": [T, D] float32, "external_input": None, "label": int, "metadata": {...}}
and ``build_lorenz96_splits(args)`` returns (train, val, test) and sets
``args.roi_dim = D`` so the model picks up the right state dimension.
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from .base import SequenceDataset


def _l96_rhs(x: np.ndarray, F: float) -> np.ndarray:
    """(x_{i+1} - x_{i-2}) x_{i-1} - x_i + F, cyclic in i."""
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F


def _rk4_step(x: np.ndarray, h: float, F: float) -> np.ndarray:
    k1 = _l96_rhs(x, F)
    k2 = _l96_rhs(x + 0.5 * h * k1, F)
    k3 = _l96_rhs(x + 0.5 * h * k2, F)
    k4 = _l96_rhs(x + h * k3, F)
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_l96(x0: np.ndarray, F: float, h: float, sample_every: int,
                  n_samples: int, n_transient: int) -> np.ndarray:
    """RK4 with step h, keep every ``sample_every``-th state. Returns [n_samples, D] float32."""
    x = x0.astype(np.float64).copy()
    for _ in range(n_transient):
        x = _rk4_step(x, h, F)
    out = np.empty((n_samples, x.size), dtype=np.float32)
    for t in range(n_samples):
        for _ in range(sample_every):
            x = _rk4_step(x, h, F)
        out[t] = x.astype(np.float32)
    if not np.isfinite(out).all():
        raise FloatingPointError("Lorenz-96 integration diverged; reduce l96_solver_dt.")
    return out


def generate_lorenz96(n_traj: int, D: int, F: float, dt: float, solver_dt: float,
                      n_samples: int, transient_mtu: float, seed: int) -> np.ndarray:
    """[n_traj, n_samples, D] float32; each trajectory from an independent perturbed IC."""
    sample_every = max(1, int(round(dt / solver_dt)))
    h = dt / sample_every                       # exact: sample_every*h == dt
    n_transient = int(round(transient_mtu / h))
    trajs = np.empty((n_traj, n_samples, D), dtype=np.float32)
    for i in range(n_traj):
        rng = np.random.default_rng(seed * 100003 + i)
        x0 = F * np.ones(D) + 0.01 * rng.standard_normal(D)   # standard L96 IC: fixed point + noise
        trajs[i] = integrate_l96(x0, F, h, sample_every, n_samples, n_transient)
    return trajs


class Lorenz96Dataset(SequenceDataset):
    dataset_name = "lorenz96"
    has_external_input = False
    task_type = "sequence_vector"
    evaluator_name = "generic"

    def __init__(self, trajs: np.ndarray, mean: float, std: float, split: str = "train"):
        self.trajs = trajs                      # [n, T, D] float32, already normalized
        self.mean = float(mean)
        self.std = float(std)
        self.split = str(split)

    def __len__(self):
        return int(self.trajs.shape[0])

    def __getitem__(self, index):
        return {
            "state": torch.from_numpy(np.ascontiguousarray(self.trajs[index])),
            "external_input": None,
            "label": int(index),
            "metadata": {"dataset": self.dataset_name, "split": self.split, "index": int(index)},
        }


def build_lorenz96_splits(args) -> Tuple[Lorenz96Dataset, Lorenz96Dataset, Lorenz96Dataset]:
    D = int(getattr(args, "l96_dim", 40))
    F = float(getattr(args, "l96_forcing", 8.0))
    # dt sets how far a K-step rollout reaches in Lyapunov times (lambda ~ 1.71,
    # Lyapunov time ~0.58 MTU). At dt=0.025 a K=64 rollout spans ~2.7 Lyapunov
    # times: long enough that error genuinely accumulates, short enough that the
    # methods still separate (at dt=0.05 it is ~5.5, where every method's
    # correlation has already collapsed to zero and nothing is distinguishable).
    dt = float(getattr(args, "l96_dt", 0.025))
    solver_dt = float(getattr(args, "l96_solver_dt", 0.005))
    T = int(getattr(args, "l96_len", 1024))
    n_traj = int(getattr(args, "l96_traj", 40))
    transient = float(getattr(args, "l96_transient", 20.0))
    gen_seed = int(getattr(args, "l96_seed", 0))

    root = Path(args.data_path)
    root.mkdir(parents=True, exist_ok=True)
    cache = root / f"l96_D{D}_F{F}_dt{dt}_sdt{solver_dt}_T{T}_traj{n_traj}_tr{transient}_s{gen_seed}.npz"

    if cache.exists():
        trajs = np.load(cache)["trajs"]
        print(f"[L96] loaded cached trajectories {trajs.shape} from {cache}")
    else:
        print(f"[L96] generating {n_traj} trajectories (D={D}, F={F}, dt={dt}, T={T}) ...")
        trajs = generate_lorenz96(n_traj, D, F, dt, solver_dt, T, transient, gen_seed)
        np.savez_compressed(cache, trajs=trajs)
        print(f"[L96] cached -> {cache}")

    # Split by trajectory (never split a trajectory across sets).
    rng = np.random.default_rng(int(args.seed))
    order = rng.permutation(n_traj)
    n_train = max(1, int(round(n_traj * float(args.train_ratio))))
    n_val = max(1, int(round(n_traj * float(args.val_ratio))))
    if n_train + n_val >= n_traj:
        n_train, n_val = max(1, n_traj - 2), 1
    idx_tr, idx_va, idx_te = order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]
    if idx_te.size == 0:
        idx_te = idx_va

    # Train-split statistics only (no leakage). rel_l2 is scale-free but the
    # one-step MSE term is not, and raw L96 has mean ~2.3, std ~3.6.
    if int(getattr(args, "l96_normalize", 1)):
        mean = float(trajs[idx_tr].mean())
        std = float(trajs[idx_tr].std()) or 1.0
        trajs = ((trajs - mean) / std).astype(np.float32)
    else:
        mean, std = 0.0, 1.0

    args.roi_dim = D                            # the model reads state_dim from here
    print(f"[L96] state_dim={D}  train/val/test = {idx_tr.size}/{idx_va.size}/{idx_te.size} trajectories of length {T}")

    return (
        Lorenz96Dataset(trajs[idx_tr], mean, std, "train"),
        Lorenz96Dataset(trajs[idx_va], mean, std, "val"),
        Lorenz96Dataset(trajs[idx_te], mean, std, "test"),
    )
