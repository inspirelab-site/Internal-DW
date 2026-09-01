"""PhysioNet long-term gait dynamics: stride intervals with a built-in memory switch.

Why this dataset.  Every other testbed here needs its memory length either set by
a generation parameter (Mackey-Glass tau, NARMA order) or estimated from data.
Gait gives the same contrast in a *real* recording, under experimental control:
free walking shows slowly decaying stride-to-stride correlation, while walking in
time to a metronome does not.  Same subjects, same movement, one manipulation.

Measured here (10 subjects per condition, mean autocorrelation of the stride
interval series):

    lag              1      2      4      8     16     32     64    100    200
    free  norm    +0.488 +0.505 +0.428 +0.368 +0.294 +0.243 +0.187 +0.169 +0.124
    free  fast    +0.599 +0.614 +0.540 +0.459 +0.392 +0.322 +0.248 +0.200 +0.142
    free  slow    +0.668 +0.639 +0.572 +0.499 +0.405 +0.309 +0.245 +0.190 +0.108
    metro nrm     +0.195 +0.115 -0.019 -0.033 -0.030 -0.007 -0.001 +0.004 +0.005
    metro fst     +0.187 +0.066 -0.043 -0.011 -0.013 -0.004 +0.008 +0.007 +0.008
    metro slw     +0.155 +0.013 -0.038 -0.008 -0.002 -0.001 +0.000 +0.013 -0.009

So at a fixed training window K, the free conditions sit well inside their
memory range and the metronome conditions sit far outside it -- the K/K* ratio is
varied by changing K*, not K, which is cheaper and cleanly controlled.

Two caveats worth carrying into any comparison.  The series is SCALAR (one stride
interval per step), and its one-step autocorrelation is 0.49-0.67, far below the
0.99 of Lorenz-96 or the iEEG envelopes -- the one-step task is genuinely harder
here, so absolute correlations are not comparable across datasets.

Files are flat ASCII, one interval (seconds) per line, named si<NN>.<condition>.
Interface matches lorenz96.py.
"""
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from .base import SequenceDataset

_CONDITIONS = ("norm", "fast", "slow", "metnrm", "metfst", "metslw")
_FREE = ("norm", "fast", "slow")


class GaitDataset(SequenceDataset):
    dataset_name = "gait"
    has_external_input = False
    task_type = "sequence_vector"
    evaluator_name = "generic"

    def __init__(self, chunks: np.ndarray, mean: float, std: float, split: str = "train"):
        self.chunks = chunks                     # [n, T, 1] float32, normalized
        self.mean = float(mean)
        self.std = float(std)
        self.split = str(split)

    def __len__(self):
        return int(self.chunks.shape[0])

    def __getitem__(self, index):
        return {
            "state": torch.from_numpy(np.ascontiguousarray(self.chunks[index])),
            "external_input": None,
            "label": int(index),
            "metadata": {"dataset": self.dataset_name, "split": self.split, "index": int(index)},
        }


def _load_condition(root: Path, condition: str) -> List[np.ndarray]:
    files = sorted(root.glob(f"si*.{condition}"))
    if not files:
        raise FileNotFoundError(
            f"No files matching si*.{condition} under {root}. Known conditions: {_CONDITIONS}")
    out = []
    for f in files:
        x = np.loadtxt(f, dtype=np.float32)
        if x.ndim != 1:
            x = x.reshape(-1)
        out.append(x[:, None])                   # [T, 1]
    return out


def _chunk_series(series: List[np.ndarray], chunk_len: int) -> np.ndarray:
    """Cut each recording into non-overlapping chunks; drop remainders."""
    chunks = []
    for x in series:
        n = x.shape[0] // chunk_len
        if n < 1:
            continue
        chunks.append(x[:n * chunk_len].reshape(n, chunk_len, x.shape[1]))
    if not chunks:
        raise ValueError(f"no recording is as long as gait_chunk={chunk_len}")
    return np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.float32)


def build_gait_splits(args):
    root = Path(getattr(args, "gait_root",
                        "data/gait/long-term-recordings-of-gait-dynamics-1.0.0"))
    condition = str(getattr(args, "gait_condition", "norm"))
    chunk_len = int(getattr(args, "gait_chunk", 512))
    detrend = bool(int(getattr(args, "gait_detrend", 0)))

    series = _load_condition(root, condition)
    if detrend:
        # Optional control: removing a per-recording linear trend distinguishes a
        # genuine slowly-decaying correlation from a drift artefact.  Off by
        # default so the headline numbers are on the raw series.
        for i, x in enumerate(series):
            t = np.arange(x.shape[0], dtype=np.float64)
            c = np.polyfit(t, x[:, 0].astype(np.float64), 1)
            series[i] = (x[:, 0] - np.polyval(c, t)).astype(np.float32)[:, None]

    # Split by RECORDING (i.e. by subject): never split one person's walk across
    # train and test, and never let the same subject appear in two splits.
    n = len(series)
    rng = np.random.default_rng(int(args.seed))
    order = rng.permutation(n)
    n_train = max(1, int(round(n * float(args.train_ratio))))
    n_val = max(1, int(round(n * float(args.val_ratio))))
    if n_train + n_val >= n:
        n_train, n_val = max(1, n - 2), 1
    idx_tr = order[:n_train]
    idx_va = order[n_train:n_train + n_val]
    idx_te = order[n_train + n_val:]
    if idx_te.size == 0:
        idx_te = idx_va

    tr = _chunk_series([series[i] for i in idx_tr], chunk_len)
    va = _chunk_series([series[i] for i in idx_va], chunk_len)
    te = _chunk_series([series[i] for i in idx_te], chunk_len)

    mean = float(tr.mean())
    std = float(tr.std()) or 1.0
    tr = ((tr - mean) / std).astype(np.float32)
    va = ((va - mean) / std).astype(np.float32)
    te = ((te - mean) / std).astype(np.float32)

    args.roi_dim = 1
    kind = "free" if condition in _FREE else "metronome"
    print(f"[gait] condition={condition} ({kind})  {n} recordings  state_dim=1  "
          f"train/val/test = {tr.shape[0]}/{va.shape[0]}/{te.shape[0]} chunks of {chunk_len} "
          f"(subjects {idx_tr.size}/{idx_va.size}/{idx_te.size}, detrend={int(detrend)})")

    return (GaitDataset(tr, mean, std, "train"),
            GaitDataset(va, mean, std, "val"),
            GaitDataset(te, mean, std, "test"))
