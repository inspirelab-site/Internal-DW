"""EXPERIMENT 2 -- the ORACLE gate: an upper bound on what routing can buy.

The gate is set from the *data*, not from the model: scripts/compute_ek_map.py
measures, model-free, a per-(start, horizon) predictability e_k(t), and keeps the
most-predictable fraction p of starts **within each horizon k**.  Training K=64
with that mask and comparing against K=16 answers the question a learned
criterion cannot: does a state-dependent gate have any headroom at all here, or
is K=64 simply worse than K=16 no matter which pathways you keep?

Two properties make this the right ceiling:

  * The quantile is taken WITHIN each k.  e_k(t) grows with k for every start,
    so a global threshold would keep small k and drop large k -- that is
    truncation, and it would delete exactly the fluctuating far-horizon credit
    the method exists to keep.  Selecting within k means the oracle is pure
    state selection at every horizon, open at large k whenever that particular
    start is still predictable there.
  * The map is produced by a local-analog probe on the raw series, so it is not
    circular with respect to the model being trained.

Run-time identification.  The trainer does not tell the model which absolute
start a rollout came from, and threading that through is invasive.  Instead the
oracle identifies the start from the data: at rollout step 0 it matches the
current input frame against the stored reference frames (exact match up to
float error, since both come from the same series).  Starts outside the
reference set fall back to their nearest reference; the reported match distance
tells you whether that ever happens.

Enable with, e.g.

    RESGRAD_ORACLE=ek_map_ieeg_theta.npz RESGRAD_ORACLE_FRAC=0.5 \
      DATASET=ieeg COND=theta K=64 METHOD=oracle SEED=0 GPU=0 \
      bash scripts/train/run_mem_one.sh
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch


class OracleGate:
    """Per-(start, horizon) mask, looked up by matching the rollout's first frame."""

    _inst: Optional["OracleGate"] = None

    def __init__(self, path: str, frac: float):
        z = np.load(path)
        key = f"oracle_p{frac}"
        if key not in z.files:
            avail = [k for k in z.files if k.startswith("oracle_p")]
            raise SystemExit(
                f"{path} has no '{key}'. Available: {avail}. "
                f"Re-run compute_ek_map.py with --oracle-fracs {frac}")
        if "ref_frame" not in z.files:
            raise SystemExit(
                f"{path} predates the oracle support: no 'ref_frame'. "
                f"Re-run scripts/compute_ek_map.py to regenerate it.")
        self.mask = torch.from_numpy(z[key].astype(np.float32))       # [S, K]
        self.ref = torch.from_numpy(np.asarray(z["ref_frame"], np.float32))  # [S, D]
        self.alpha_tab = (torch.from_numpy(np.asarray(z["alpha"], np.float32))
                          if "alpha" in z.files else None)            # [S, K]
        self.alpha_floor = float(os.environ.get("RESGRAD_ALPHA_FLOOR", "0.0"))
        self.horizons = np.asarray(z["horizons"], np.int64)
        self.frac = frac
        self._ref_sq = (self.ref * self.ref).sum(dim=1)               # [S]
        self._dev: Optional[torch.device] = None
        # per-rollout state
        self.cur_idx: Optional[torch.Tensor] = None                   # [B]
        self.match_dist: float = float("nan")
        self.n_lookups = 0
        self.worst_dist = 0.0
        print(f"[oracle] {path} p={frac}: mask {tuple(self.mask.shape)}, "
              f"open fraction {float(self.mask.mean()):.3f}, "
              f"per-k open {np.round(self.mask.mean(dim=0).numpy()[:6], 3)} ...",
              flush=True)

    # ------------------------------------------------------------------
    @classmethod
    def maybe(cls) -> Optional["OracleGate"]:
        """Singleton built from the environment, or None if not requested."""
        if cls._inst is not None:
            return cls._inst
        path = os.environ.get("RESGRAD_ORACLE", "")
        if not path:
            return None
        frac = float(os.environ.get("RESGRAD_ORACLE_FRAC", "0.5"))
        cls._inst = cls(path, frac)
        return cls._inst

    # ------------------------------------------------------------------
    def _to(self, device: torch.device) -> None:
        if self._dev != device:
            self.mask = self.mask.to(device)
            self.ref = self.ref.to(device)
            self._ref_sq = self._ref_sq.to(device)
            self._dev = device

    def begin_rollout(self, x_t: torch.Tensor) -> None:
        """Identify which reference start each batch element is, from frame 0."""
        self._to(x_t.device)
        q = x_t.reshape(x_t.shape[0], -1).float()
        if q.shape[1] != self.ref.shape[1]:
            raise SystemExit(
                f"oracle frame dim {self.ref.shape[1]} != model state dim {q.shape[1]}; "
                f"the ek map was built from a different series")
        d2 = (q * q).sum(dim=1, keepdim=True) + self._ref_sq[None, :] - 2.0 * (q @ self.ref.T)
        best = torch.argmin(d2, dim=1)
        self.cur_idx = best
        md = float(torch.sqrt(torch.clamp(d2.gather(1, best[:, None]), min=0)).mean().cpu())
        self.match_dist = md
        self.worst_dist = max(self.worst_dist, md)
        self.n_lookups += 1
        if self.n_lookups <= 3 or self.n_lookups % 2000 == 0:
            print(f"[oracle] lookup {self.n_lookups}: mean match distance {md:.3e} "
                  f"(worst so far {self.worst_dist:.3e})", flush=True)

    def gate(self, k: int, batch: int, device: torch.device,
             dtype: torch.dtype) -> torch.Tensor:
        """Gate for rollout step k, shaped [B] (caller reshapes to broadcast)."""
        self._to(device)
        if self.cur_idx is None or self.cur_idx.shape[0] != batch:
            # no rollout boundary was seen (e.g. eval path) -- leave fully open
            return torch.ones(batch, device=device, dtype=dtype)
        kk = min(max(int(k), 0), self.mask.shape[1] - 1)
        return self.mask[self.cur_idx, kk].to(dtype)

    # ------------------------------------------------------------------
    def alpha(self, k: int, batch: int, device: torch.device,
              dtype: torch.dtype) -> torch.Tensor:
        """DERIVED reach gate for rollout step k, shaped [B].

        alpha_k = R2_k / R2_{k-1} with R2_k the locally reducible fraction of the
        horizon-k target, so the accumulated product equals R2_k by construction.
        Nothing here is tuned: the table comes straight from the model-free
        local-analog errors. A floor keeps a long rollout from going numerically
        silent; set RESGRAD_ALPHA_FLOOR to change it.
        """
        self._to(device)
        if self.alpha_tab is None:
            raise SystemExit(
                "this ek map has no 'alpha' table -- regenerate it with the "
                "current scripts/compute_ek_map.py")
        if self.cur_idx is None or self.cur_idx.shape[0] != batch:
            return torch.ones(batch, device=device, dtype=dtype)
        kk = min(max(int(k), 0), self.alpha_tab.shape[1] - 1)
        a = self.alpha_tab[self.cur_idx, kk].to(dtype)
        if self.alpha_floor > 0.0:
            a = a.clamp_min(self.alpha_floor)
        return a
