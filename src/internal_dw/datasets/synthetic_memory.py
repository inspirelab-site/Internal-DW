"""Two synthetic testbeds whose memory length is a KNOB, not an estimate.

Why these exist.  Every claim in this project is conditional on the training
window K sitting inside or outside the data's predictability horizon K*, and on
both datasets used so far K* had to be *estimated* with a probe we designed
ourselves.  That makes the central comparison rest on a single estimated number.
Mackey--Glass and NARMA remove that dependency: in both, the ground-truth memory
requirement is a generation parameter we set.

    Mackey--Glass   dx/dt = beta*x(t-tau)/(1 + x(t-tau)^n) - gamma*x(t)
        The delay tau IS the memory length, in samples once we fix dt.  It also
        controls the dynamical regime, which is the second reason to prefer it
        over a chaotic PDE here: at beta=0.2, gamma=0.1, n=10 the system is a
        stable limit cycle for tau < ~16.8, weakly chaotic near tau=17, and
        clearly chaotic by tau=30.  So chaoticity and memory length can be
        varied along the same axis and the non-chaotic corner -- absent from
        Lorenz-96 -- is reachable.
        Autonomous: no external input, one information channel.

    NARMA-L         y(t+1) = 0.3 y(t) + 0.05 y(t) * sum_{i<L} y(t-i)
                             + 1.5 u(t-L+1) u(t) + 0.1,   u ~ U[0, 0.5]
        The order L is the memory length for BOTH channels: the state term needs
        y back to t-L+1 and the input term needs u back to t-L+1.  This is the
        controlled analogue of the driven case -- a system with a state channel
        and an input channel whose lag supports are known exactly rather than
        probed.

Both are cheap: a few seconds to generate, cached to --data_path as .npz, so a
full K-sweep is minutes rather than GPU-days.

Interface matches lorenz96.py: __getitem__ returns
    {"state": [T, D] float32, "external_input": [T, S] float32 or None, ...}
and build_*_splits(args) returns (train, val, test) and sets args.roi_dim.
"""
import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from .base import SequenceDataset


# ----------------------------------------------------------------------------
# Mackey--Glass
# ----------------------------------------------------------------------------
def _mg_rhs(x_now: float, x_del: float, beta: float, gamma: float, n: float) -> float:
    return beta * x_del / (1.0 + x_del ** n) - gamma * x_now


def _stationary_ar1(length: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Draw a zero-mean, unit-variance stationary AR(1) forcing path."""
    if length <= 0:
        raise ValueError(f"AR(1) forcing length must be positive, got {length}")
    if not -1.0 < float(rho) < 1.0:
        raise ValueError(f"AR(1) forcing requires |rho| < 1, got {rho}")
    drive = np.empty(int(length), dtype=np.float64)
    drive[0] = rng.standard_normal()
    innovation_scale = np.sqrt(1.0 - float(rho) ** 2)
    innovation = rng.standard_normal(int(length) - 1)
    for index in range(1, int(length)):
        drive[index] = float(rho) * drive[index - 1] + innovation_scale * innovation[index - 1]
    return drive


def integrate_mackey_glass(tau: float, dt: float, n_samples: int, solver_dt: float,
                           beta: float, gamma: float, n_exp: float,
                           n_transient: int, rng: np.random.Generator) -> np.ndarray:
    """RK4 on a ring history buffer; returns [n_samples] float32 sampled every dt.

    The delayed value is read from a circular buffer holding the last tau/solver_dt
    solver states, with linear interpolation so tau need not be a multiple of the
    solver step.
    """
    sub = max(1, int(round(dt / solver_dt)))
    h = dt / sub                                   # exact: sub*h == dt
    n_hist = int(np.ceil(tau / h)) + 2
    # Constant history on [-tau, 0] with a small perturbation is the standard IC.
    hist = 1.2 + 0.02 * rng.standard_normal(n_hist)
    pos = n_hist - 1                               # index of the current value
    delay_steps = tau / h

    def delayed() -> float:
        # linear interpolation between the two buffer slots straddling t - tau
        f = np.floor(delay_steps)
        w = delay_steps - f
        i0 = int((pos - f) % n_hist)
        i1 = int((pos - f - 1) % n_hist)
        return (1.0 - w) * hist[i0] + w * hist[i1]

    def step():
        nonlocal pos
        x = hist[pos]
        xd = delayed()
        # RK4 with the delayed term held at its value over the step.  The delay
        # changes on a timescale of tau >> h, so freezing it costs far less than
        # the O(h^4) local error we keep from the non-delayed part.
        k1 = _mg_rhs(x, xd, beta, gamma, n_exp)
        k2 = _mg_rhs(x + 0.5 * h * k1, xd, beta, gamma, n_exp)
        k3 = _mg_rhs(x + 0.5 * h * k2, xd, beta, gamma, n_exp)
        k4 = _mg_rhs(x + h * k3, xd, beta, gamma, n_exp)
        nxt = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        pos = (pos + 1) % n_hist
        hist[pos] = nxt

    for _ in range(n_transient * sub):
        step()
    out = np.empty(n_samples, dtype=np.float32)
    for t in range(n_samples):
        for _ in range(sub):
            step()
        out[t] = hist[pos]
    if not np.isfinite(out).all():
        raise FloatingPointError("Mackey-Glass integration diverged; reduce mg_solver_dt.")
    return out


def generate_mackey_glass(n_traj: int, D: int, tau: float, dt: float, solver_dt: float,
                          n_samples: int, beta: float, gamma: float, n_exp: float,
                          transient: int, seed: int) -> np.ndarray:
    """[n_traj, n_samples, D] float32.

    D independent series with the same parameters but different initial histories
    are stacked as channels.  They do not interact, so the memory structure per
    channel is exactly tau; the extra channels only give the read-out something
    wider than a scalar to work with, without ever leaking the delayed value into
    the observed state (which a delay embedding would).
    """
    out = np.empty((n_traj, n_samples, D), dtype=np.float32)
    for i in range(n_traj):
        for d in range(D):
            rng = np.random.default_rng(seed * 100003 + i * 997 + d)
            out[i, :, d] = integrate_mackey_glass(tau, dt, n_samples, solver_dt,
                                                  beta, gamma, n_exp, transient, rng)
    return out


def integrate_driven_mackey_glass(
    tau: float,
    dt: float,
    n_samples: int,
    solver_dt: float,
    beta: float,
    gamma: float,
    n_exp: float,
    n_transient: int,
    drive: np.ndarray,
    drive_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Integrate Mackey--Glass with an observed piecewise-constant forcing.

    ``drive[t]`` is held fixed across the RK4 substeps of one autoregressive
    interval.  The first ``n_transient`` values burn in the driven system; the
    remaining values are returned alongside the recorded states by
    :func:`generate_driven_mackey_glass`.
    """
    total_samples = int(n_transient) + int(n_samples)
    drive = np.asarray(drive, dtype=np.float64).reshape(-1)
    if drive.shape[0] != total_samples:
        raise ValueError(
            f"driven MG requires {total_samples} forcing values, got {drive.shape[0]}"
        )

    sub = max(1, int(round(dt / solver_dt)))
    h = dt / sub
    n_hist = int(np.ceil(tau / h)) + 2
    hist = 1.2 + 0.02 * rng.standard_normal(n_hist)
    pos = n_hist - 1
    delay_steps = tau / h

    def delayed() -> float:
        floor_steps = np.floor(delay_steps)
        weight = delay_steps - floor_steps
        index0 = int((pos - floor_steps) % n_hist)
        index1 = int((pos - floor_steps - 1) % n_hist)
        return (1.0 - weight) * hist[index0] + weight * hist[index1]

    def step(force: float) -> None:
        nonlocal pos
        x_now = hist[pos]
        x_delayed = delayed()

        def rhs(value: float) -> float:
            return _mg_rhs(value, x_delayed, beta, gamma, n_exp) + float(drive_scale) * force

        k1 = rhs(x_now)
        k2 = rhs(x_now + 0.5 * h * k1)
        k3 = rhs(x_now + 0.5 * h * k2)
        k4 = rhs(x_now + h * k3)
        next_value = x_now + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        pos = (pos + 1) % n_hist
        hist[pos] = next_value

    output = np.empty(int(n_samples), dtype=np.float32)
    output_index = 0
    for sample_index, force in enumerate(drive):
        for _ in range(sub):
            step(float(force))
        if sample_index >= int(n_transient):
            output[output_index] = hist[pos]
            output_index += 1
    if not np.isfinite(output).all():
        raise FloatingPointError(
            "Driven Mackey-Glass integration diverged; reduce mg_drive_scale or mg_solver_dt."
        )
    return output


def generate_driven_mackey_glass(
    n_traj: int,
    D: int,
    tau: float,
    dt: float,
    solver_dt: float,
    n_samples: int,
    beta: float,
    gamma: float,
    n_exp: float,
    transient: int,
    seed: int,
    drive_scale: float,
    drive_rho: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return driven MG states and the observed forcing, both ``[N,T,D]``.

    Every channel receives an independent forcing realization with the same
    stationary AR(1) law.  The future realization is exposed as
    ``external_input``; it is therefore observed drive, not innovation noise.
    Setting ``drive_scale=0`` exactly recovers the autonomous state generator
    while retaining the forcing array for controlled comparisons.
    """
    states = np.empty((n_traj, n_samples, D), dtype=np.float32)
    drives = np.empty_like(states)
    total_samples = int(transient) + int(n_samples)
    for trajectory in range(int(n_traj)):
        for channel in range(int(D)):
            stream = int(seed) * 100003 + trajectory * 997 + channel
            state_rng = np.random.default_rng(stream)
            drive_rng = np.random.default_rng(stream + 2_147_483_647)
            full_drive = _stationary_ar1(total_samples, drive_rho, drive_rng)
            states[trajectory, :, channel] = integrate_driven_mackey_glass(
                tau,
                dt,
                n_samples,
                solver_dt,
                beta,
                gamma,
                n_exp,
                transient,
                full_drive,
                drive_scale,
                state_rng,
            )
            drives[trajectory, :, channel] = full_drive[int(transient):]
    return states, drives


# ----------------------------------------------------------------------------
# NARMA-L
# ----------------------------------------------------------------------------
def generate_narma(n_traj: int, D: int, order: int, n_samples: int, transient: int,
                   u_scale: float, seed: int, bounded: bool,
                   drive: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (y [n_traj, n_samples, D], u [n_traj, n_samples, D]).

    The classical recursion is only marginally stable and reliably blows up for
    order >~ 15, which would silently destroy any sweep over the memory length.
    ``bounded=True`` wraps the update in a tanh, the standard fix in the reservoir
    literature; it preserves the lag structure (still exactly ``order`` taps on
    both y and u) while guaranteeing a bounded trajectory, so order can be swept
    freely.  ``bounded=False`` reproduces the textbook equation.

    ``drive`` is the coefficient on the input product term (1.5 in the textbook
    equation).  It is the one knob that changes how much of the target comes from
    the exogenous input rather than the state's own history, holding the lag
    structure -- and therefore the memory length -- fixed.  Sweeping it is the
    controlled way to ask whether drive strength is what governs the effect,
    since every other property of the system is untouched.
    """
    L = int(order)
    T = int(n_samples) + int(transient) + L + 1
    y = np.zeros((n_traj, T, D), dtype=np.float64)
    rngs = [np.random.default_rng(seed * 100003 + i) for i in range(n_traj)]
    u = np.stack([r.uniform(0.0, u_scale, size=(T, D)) for r in rngs], axis=0)

    for t in range(L, T - 1):
        ysum = y[:, t - L + 1:t + 1, :].sum(axis=1)          # sum_{i<L} y(t-i)
        nxt = (0.3 * y[:, t, :]
               + 0.05 * y[:, t, :] * ysum
               + float(drive) * u[:, t - L + 1, :] * u[:, t, :]
               + 0.1)
        y[:, t + 1, :] = np.tanh(nxt) if bounded else nxt
        if not bounded and not np.isfinite(y[:, t + 1, :]).all():
            raise FloatingPointError(
                f"NARMA-{L} diverged at t={t}; use --narma_bounded (tanh) for order>10.")

    keep = slice(transient + L + 1, transient + L + 1 + n_samples)
    return (np.ascontiguousarray(y[:, keep, :], dtype=np.float32),
            np.ascontiguousarray(u[:, keep, :], dtype=np.float32))


# ----------------------------------------------------------------------------
# Known-SNR linear Gaussian AR
# ----------------------------------------------------------------------------
_KNOWN_SNR_DEFAULT_COEFFICIENTS = (
    0.995, 0.98, 0.95, 0.90, -0.995, -0.98, -0.95, -0.90,
)


def resolve_known_snr_ar_coefficients(
    dim: int, coefficients: Optional[Sequence[float]] = None
) -> np.ndarray:
    """Return the diagonal AR coefficients used by the identifiable testbed.

    A single supplied coefficient is repeated.  With no explicit list, a
    mixture of slow/fast and positive/negative modes is tiled to ``dim``.  The
    negative modes make the conditional mean oscillate while preserving the
    same closed-form variance and SNR as their positive counterparts.
    """

    dim = int(dim)
    if dim <= 0:
        raise ValueError(f"known-SNR AR dimension must be positive, got {dim}")
    raw = list(coefficients or ())
    if not raw:
        repeats = (dim + len(_KNOWN_SNR_DEFAULT_COEFFICIENTS) - 1) // len(
            _KNOWN_SNR_DEFAULT_COEFFICIENTS
        )
        raw = list((_KNOWN_SNR_DEFAULT_COEFFICIENTS * repeats)[:dim])
    elif len(raw) == 1:
        raw = raw * dim
    elif len(raw) != dim:
        raise ValueError(
            "--snr_ar_coefficients must contain either one value or exactly "
            f"snr_ar_dim={dim} values; got {len(raw)}"
        )
    values = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(np.abs(values) >= 1.0):
        raise ValueError("known-SNR AR coefficients must be finite with |a_j| < 1")
    return values


def known_snr_ar_moments(
    coefficients: Sequence[float], horizons: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytic signal/noise variances and SNR for a stationary diagonal AR.

    The process is ``x[t+1,j] = a[j] x[t,j] + eps[t+1,j]`` with
    ``Var(eps_j)=1-a[j]^2`` and ``x[0]~N(0,I)``.  Consequently every state
    coordinate has unit marginal variance and, conditioned on ``x[t]``,

        Var(E[x[t+k,j] | x[t]]) = a[j]^(2k),
        Var(x[t+k,j] | x[t])    = 1 - a[j]^(2k).

    The returned aggregate SNR is the trace ratio at each horizon.
    """

    coeff = np.asarray(coefficients, dtype=np.float64).reshape(1, -1)
    steps = np.asarray(horizons, dtype=np.int64).reshape(-1, 1)
    if np.any(steps <= 0):
        raise ValueError("known-SNR horizons are one-based positive integers")
    signal = np.power(coeff * coeff, steps)
    innovation = np.maximum(1.0 - signal, 0.0)
    snr = signal.sum(axis=1) / np.maximum(innovation.sum(axis=1), 1e-30)
    return signal, innovation, snr


def generate_known_snr_ar(
    n_traj: int,
    n_samples: int,
    coefficients: Sequence[float],
    seed: int,
) -> np.ndarray:
    """Generate exactly stationary diagonal linear-Gaussian AR trajectories."""

    coeff = np.asarray(coefficients, dtype=np.float64)
    n_traj, n_samples = int(n_traj), int(n_samples)
    if n_traj <= 0 or n_samples < 2:
        raise ValueError("known-SNR AR requires n_traj>0 and n_samples>=2")
    innovation_std = np.sqrt(np.maximum(1.0 - coeff * coeff, 0.0))
    out = np.empty((n_traj, n_samples, coeff.size), dtype=np.float64)
    for trajectory in range(n_traj):
        rng = np.random.default_rng(int(seed) * 100003 + trajectory * 997)
        out[trajectory, 0] = rng.standard_normal(coeff.size)
        noise = rng.standard_normal((n_samples - 1, coeff.size)) * innovation_std
        for time_index in range(n_samples - 1):
            out[trajectory, time_index + 1] = (
                coeff * out[trajectory, time_index] + noise[time_index]
            )
    return np.ascontiguousarray(out, dtype=np.float32)


# ----------------------------------------------------------------------------
# dataset objects
# ----------------------------------------------------------------------------
class _StackedSeriesDataset(SequenceDataset):
    task_type = "sequence_vector"
    evaluator_name = "generic"

    def __init__(self, states: np.ndarray, inputs: Optional[np.ndarray],
                 mean: float, std: float, split: str = "train"):
        self.states = states                     # [n, T, D] float32, normalized
        self.inputs = inputs                     # [n, T, S] float32 or None
        self.mean = float(mean)
        self.std = float(std)
        self.split = str(split)

    def __len__(self):
        return int(self.states.shape[0])

    def __getitem__(self, index):
        ext = None
        if self.inputs is not None:
            ext = torch.from_numpy(np.ascontiguousarray(self.inputs[index]))
        return {
            "state": torch.from_numpy(np.ascontiguousarray(self.states[index])),
            "external_input": ext,
            "label": int(index),
            "metadata": {"dataset": self.dataset_name, "split": self.split, "index": int(index)},
        }


class MackeyGlassDataset(_StackedSeriesDataset):
    dataset_name = "mackey_glass"
    has_external_input = False


class DrivenMackeyGlassDataset(_StackedSeriesDataset):
    dataset_name = "mackey_glass_driven"
    has_external_input = True


class NARMADataset(_StackedSeriesDataset):
    dataset_name = "narma"
    has_external_input = True


class KnownSNRARDataset(_StackedSeriesDataset):
    """Linear-Gaussian AR data with an analytic multi-step innovation SNR."""

    dataset_name = "known_snr_ar"
    has_external_input = False


# ----------------------------------------------------------------------------
# splits
# ----------------------------------------------------------------------------
def _split_indices(n_traj: int, args):
    rng = np.random.default_rng(int(args.seed))
    order = rng.permutation(n_traj)
    n_train = max(1, int(round(n_traj * float(args.train_ratio))))
    n_val = max(1, int(round(n_traj * float(args.val_ratio))))
    if n_train + n_val >= n_traj:
        n_train, n_val = max(1, n_traj - 2), 1
    tr, va, te = order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]
    return tr, va, (va if te.size == 0 else te)


def build_mackey_glass_splits(args):
    D = int(getattr(args, "mg_dim", 8))
    tau = float(getattr(args, "mg_tau", 17.0))
    dt = float(getattr(args, "mg_dt", 1.0))
    solver_dt = float(getattr(args, "mg_solver_dt", 0.1))
    T = int(getattr(args, "mg_len", 2048))
    n_traj = int(getattr(args, "mg_traj", 40))
    beta = float(getattr(args, "mg_beta", 0.2))
    gamma = float(getattr(args, "mg_gamma", 0.1))
    n_exp = float(getattr(args, "mg_n", 10.0))
    transient = int(getattr(args, "mg_transient", 1000))
    gen_seed = int(getattr(args, "mg_seed", 0))

    root = Path(args.data_path)
    root.mkdir(parents=True, exist_ok=True)
    cache = root / (f"mg_D{D}_tau{tau}_dt{dt}_sdt{solver_dt}_T{T}_traj{n_traj}"
                    f"_b{beta}_g{gamma}_n{n_exp}_tr{transient}_s{gen_seed}.npz")

    if cache.exists():
        trajs = np.load(cache)["trajs"]
        print(f"[MG] loaded cached trajectories {trajs.shape} from {cache}")
    else:
        print(f"[MG] generating {n_traj}x{D} series (tau={tau}, dt={dt}, T={T}) ...")
        trajs = generate_mackey_glass(n_traj, D, tau, dt, solver_dt, T,
                                      beta, gamma, n_exp, transient, gen_seed)
        np.savez_compressed(cache, trajs=trajs)
        print(f"[MG] cached -> {cache}")

    idx_tr, idx_va, idx_te = _split_indices(n_traj, args)
    mean = float(trajs[idx_tr].mean())
    std = float(trajs[idx_tr].std()) or 1.0
    trajs = ((trajs - mean) / std).astype(np.float32)

    args.roi_dim = D
    # tau/dt is the ground-truth memory length in AR steps -- the quantity K
    # should be compared against, and the reason this dataset exists.
    print(f"[MG] state_dim={D}  delay={tau/dt:.1f} AR steps  "
          f"train/val/test = {idx_tr.size}/{idx_va.size}/{idx_te.size} trajectories of length {T}")

    return (MackeyGlassDataset(trajs[idx_tr], None, mean, std, "train"),
            MackeyGlassDataset(trajs[idx_va], None, mean, std, "val"),
            MackeyGlassDataset(trajs[idx_te], None, mean, std, "test"))


def build_driven_mackey_glass_splits(args):
    D = int(getattr(args, "mg_dim", 8))
    tau = float(getattr(args, "mg_tau", 30.0))
    dt = float(getattr(args, "mg_dt", 1.0))
    solver_dt = float(getattr(args, "mg_solver_dt", 0.1))
    T = int(getattr(args, "mg_len", 2048))
    n_traj = int(getattr(args, "mg_traj", 40))
    beta = float(getattr(args, "mg_beta", 0.2))
    gamma = float(getattr(args, "mg_gamma", 0.1))
    n_exp = float(getattr(args, "mg_n", 10.0))
    transient = int(getattr(args, "mg_transient", 1000))
    gen_seed = int(getattr(args, "mg_seed", 0))
    drive_scale = float(getattr(args, "mg_drive_scale", 0.08))
    drive_rho = float(getattr(args, "mg_drive_rho", 0.9))

    # Formal experiments use the already regime-validated, pre-split archive.
    # This prevents the optimizer seed from silently changing the data split
    # and guarantees that no test trajectory was used by the validation-only
    # drive--history screen.
    archive_arg = str(getattr(args, "mg_driven_npz", "") or "").strip()
    if archive_arg:
        archive_path = Path(archive_arg)
        if not archive_path.is_file():
            raise FileNotFoundError(f"Driven-MG archive not found: {archive_path}")
        with np.load(archive_path, allow_pickle=False) as archive:
            required = {
                f"{split}_{kind}"
                for split in ("train", "validation", "test")
                for kind in ("state", "drive")
            }
            missing = sorted(required.difference(archive.files))
            if missing:
                raise ValueError(
                    f"Driven-MG archive {archive_path} is missing arrays: {missing}"
                )
            split_arrays = {
                split: (
                    np.asarray(archive[f"{split}_state"], dtype=np.float32),
                    np.asarray(archive[f"{split}_drive"], dtype=np.float32),
                )
                for split in ("train", "validation", "test")
            }
            metadata = {}
            if "metadata_json" in archive.files:
                metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))

        reference_shape = None
        for split, (states, drives) in split_arrays.items():
            if states.ndim != 3 or states.shape != drives.shape or states.shape[0] == 0:
                raise ValueError(
                    f"Driven-MG {split} state/drive must be matching nonempty [N,T,D] "
                    f"arrays, got {states.shape} and {drives.shape}"
                )
            shape = states.shape[1:]
            if reference_shape is None:
                reference_shape = shape
            elif shape != reference_shape:
                raise ValueError(
                    f"Driven-MG split shape mismatch: expected (*,{reference_shape}), "
                    f"got {states.shape} for {split}"
                )
            if not np.isfinite(states).all() or not np.isfinite(drives).all():
                raise ValueError(f"Driven-MG {split} contains non-finite values")

        parameters = metadata.get("parameters", {}) if isinstance(metadata, dict) else {}
        expected = {
            "dim": D,
            "tau": tau,
            "length": T,
            "trajectories": n_traj,
            "drive_scale": drive_scale,
            "drive_rho": drive_rho,
        }
        for key, value in expected.items():
            if key not in parameters:
                continue
            observed = parameters[key]
            if isinstance(value, float):
                matched = bool(np.isclose(float(observed), value, rtol=0.0, atol=1e-12))
            else:
                matched = int(observed) == int(value)
            if not matched:
                raise ValueError(
                    f"Driven-MG archive parameter {key}={observed} does not match "
                    f"requested {value}"
                )

        train_states = split_arrays["train"][0]
        mean = float(train_states.mean())
        std = float(train_states.std()) or 1.0
        normalized = {
            split: (
                np.ascontiguousarray((states - mean) / std, dtype=np.float32),
                np.ascontiguousarray(drives, dtype=np.float32),
            )
            for split, (states, drives) in split_arrays.items()
        }
        D_loaded = int(train_states.shape[-1])
        args.roi_dim = D_loaded
        args.stim_dim = D_loaded
        counts = {split: int(values[0].shape[0]) for split, values in normalized.items()}
        print(
            f"[Driven MG] loaded fixed pre-split archive {archive_path}; "
            f"state_dim={D_loaded} stim_dim={D_loaded} "
            f"train/val/test={counts['train']}/{counts['validation']}/{counts['test']}"
        )
        return (
            DrivenMackeyGlassDataset(*normalized["train"], mean, std, "train"),
            DrivenMackeyGlassDataset(
                *normalized["validation"], mean, std, "val"
            ),
            DrivenMackeyGlassDataset(*normalized["test"], mean, std, "test"),
        )

    root = Path(args.data_path)
    root.mkdir(parents=True, exist_ok=True)
    cache = root / (
        f"mg_driven_D{D}_tau{tau}_dt{dt}_sdt{solver_dt}_T{T}_traj{n_traj}"
        f"_b{beta}_g{gamma}_n{n_exp}_tr{transient}_ds{drive_scale}"
        f"_rho{drive_rho}_s{gen_seed}.npz"
    )
    if cache.exists():
        archive = np.load(cache)
        trajectories, drives = archive["trajs"], archive["drives"]
        print(f"[Driven MG] loaded cached trajectories {trajectories.shape} from {cache}")
    else:
        print(
            f"[Driven MG] generating {n_traj}x{D} series "
            f"(tau={tau}, drive_scale={drive_scale}, drive_rho={drive_rho}) ..."
        )
        trajectories, drives = generate_driven_mackey_glass(
            n_traj,
            D,
            tau,
            dt,
            solver_dt,
            T,
            beta,
            gamma,
            n_exp,
            transient,
            gen_seed,
            drive_scale,
            drive_rho,
        )
        np.savez_compressed(cache, trajs=trajectories, drives=drives)
        print(f"[Driven MG] cached -> {cache}")

    idx_tr, idx_va, idx_te = _split_indices(n_traj, args)
    mean = float(trajectories[idx_tr].mean())
    std = float(trajectories[idx_tr].std()) or 1.0
    trajectories = ((trajectories - mean) / std).astype(np.float32)
    args.roi_dim = D
    args.stim_dim = D
    print(
        f"[Driven MG] state_dim={D} stim_dim={D} delay={tau / dt:.1f} AR steps "
        f"drive_scale={drive_scale} drive_rho={drive_rho} train/val/test="
        f"{idx_tr.size}/{idx_va.size}/{idx_te.size} trajectories of length {T}"
    )
    return (
        DrivenMackeyGlassDataset(
            trajectories[idx_tr], drives[idx_tr], mean, std, "train"
        ),
        DrivenMackeyGlassDataset(
            trajectories[idx_va], drives[idx_va], mean, std, "val"
        ),
        DrivenMackeyGlassDataset(
            trajectories[idx_te], drives[idx_te], mean, std, "test"
        ),
    )


def build_narma_splits(args):
    D = int(getattr(args, "narma_dim", 8))
    L = int(getattr(args, "narma_order", 10))
    T = int(getattr(args, "narma_len", 2048))
    n_traj = int(getattr(args, "narma_traj", 40))
    transient = int(getattr(args, "narma_transient", 200))
    u_scale = float(getattr(args, "narma_u_scale", 0.5))
    bounded = bool(int(getattr(args, "narma_bounded", 1)))
    gen_seed = int(getattr(args, "narma_seed", 0))
    drive = float(getattr(args, "narma_drive", 1.5))

    root = Path(args.data_path)
    root.mkdir(parents=True, exist_ok=True)
    # drive is part of the cache key: sweeping it must not silently reuse the
    # series generated at another drive strength.
    cache = root / (f"narma_D{D}_L{L}_T{T}_traj{n_traj}_tr{transient}"
                    f"_u{u_scale}_bd{int(bounded)}_dr{drive}_s{gen_seed}.npz")

    if cache.exists():
        z = np.load(cache)
        y, u = z["y"], z["u"]
        print(f"[NARMA] loaded cached {y.shape} from {cache}")
    else:
        print(f"[NARMA] generating {n_traj}x{D} series "
              f"(order={L}, T={T}, bounded={bounded}, drive={drive}) ...")
        y, u = generate_narma(n_traj, D, L, T, transient, u_scale, gen_seed, bounded,
                              drive=drive)
        np.savez_compressed(cache, y=y, u=u)
        print(f"[NARMA] cached -> {cache}")

    idx_tr, idx_va, idx_te = _split_indices(n_traj, args)
    mean = float(y[idx_tr].mean())
    std = float(y[idx_tr].std()) or 1.0
    y = ((y - mean) / std).astype(np.float32)

    args.roi_dim = D
    args.stim_dim = D                            # the drive is one channel per series
    # Both the state term and the input term reach back exactly L steps, so L is
    # the ground-truth memory length for BOTH channels.
    print(f"[NARMA] state_dim={D} stim_dim={D}  order L={L} (memory length, both channels)  "
          f"train/val/test = {idx_tr.size}/{idx_va.size}/{idx_te.size} trajectories of length {T}")

    return (NARMADataset(y[idx_tr], u[idx_tr], mean, std, "train"),
            NARMADataset(y[idx_va], u[idx_va], mean, std, "val"),
            NARMADataset(y[idx_te], u[idx_te], mean, std, "test"))


def build_known_snr_ar_splits(args):
    """Build the identifiable oracle-Wiener testbed.

    No empirical normalization is applied: the generator is stationary with
    exact mean zero and per-coordinate variance one.  This keeps the analytic
    innovation covariance byte-for-byte compatible with the covariance file
    consumed by ``DualWienerController``.
    """

    dim = int(getattr(args, "snr_ar_dim", 8))
    length = int(getattr(args, "snr_ar_len", 1024))
    n_traj = int(getattr(args, "snr_ar_traj", 96))
    gen_seed = int(getattr(args, "snr_ar_seed", 0))
    coefficients = resolve_known_snr_ar_coefficients(
        dim, getattr(args, "snr_ar_coefficients", None)
    )

    root = Path(args.data_path)
    root.mkdir(parents=True, exist_ok=True)
    coefficient_key = ",".join(f"{value:.12g}" for value in coefficients)
    digest = hashlib.sha1(coefficient_key.encode("utf-8")).hexdigest()[:12]
    cache = root / (
        f"known_snr_ar_D{dim}_T{length}_traj{n_traj}_a{digest}_s{gen_seed}.npz"
    )
    if cache.exists():
        archive = np.load(cache)
        trajectories = archive["trajs"]
        cached_coefficients = np.asarray(archive["coefficients"], dtype=np.float64)
        if not np.array_equal(cached_coefficients, coefficients):
            raise RuntimeError(f"known-SNR AR cache coefficient mismatch in {cache}")
        print(f"[known-SNR AR] loaded cached trajectories {trajectories.shape} from {cache}")
    else:
        trajectories = generate_known_snr_ar(
            n_traj=n_traj,
            n_samples=length,
            coefficients=coefficients,
            seed=gen_seed,
        )
        np.savez_compressed(
            cache,
            trajs=trajectories,
            coefficients=coefficients,
        )
        print(f"[known-SNR AR] cached -> {cache}")

    idx_tr, idx_va, idx_te = _split_indices(n_traj, args)
    args.roi_dim = dim
    selected_horizons = [1, 2, 4, 8, 16, 32, 64]
    _, _, aggregate_snr = known_snr_ar_moments(coefficients, selected_horizons)
    profile = ", ".join(
        f"H{h}:{value:.3g}" for h, value in zip(selected_horizons, aggregate_snr)
    )
    print(
        "[known-SNR AR] state_dim=%d coefficients=%s train/val/test=%d/%d/%d "
        "trajectories of length %d" % (
            dim,
            np.array2string(coefficients, precision=4, separator=","),
            idx_tr.size,
            idx_va.size,
            idx_te.size,
            length,
        )
    )
    print(f"[known-SNR AR] analytic trace-SNR profile: {profile}")

    # mean=0 and std=1 are analytic, not sample estimates.
    return (
        KnownSNRARDataset(trajectories[idx_tr], None, 0.0, 1.0, "train"),
        KnownSNRARDataset(trajectories[idx_va], None, 0.0, 1.0, "val"),
        KnownSNRARDataset(trajectories[idx_te], None, 0.0, 1.0, "test"),
    )
