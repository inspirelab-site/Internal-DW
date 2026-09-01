"""Movie-watching iEEG band envelopes as a long-memory autoregressive testbed.

Why this dataset.  The two real datasets used so far sit at opposite useless
extremes for a memory question: Lorenz-96 decorrelates in ~16 AR steps, and HCP
fMRI in ~3 steps while giving only ~285 timepoints per subject, which is too few
to measure anything at horizon 64.  Intracranial EEG band envelopes are the
standard signal for which long-range temporal correlations are actually
documented, and the recordings are ~480 s at 500 Hz -- 23,943 steps per run at a
20 ms model step, roughly 840x more data per subject than the fMRI.

Measured on P41CS R1 enc macro (16 channels, envelope autocorrelation):

    band     0.02s   0.10s   0.20s   0.50s   1.0s    2.0s     ->0.1
    delta    0.998   0.950   0.824   0.415   0.161   0.041    1.29 s
    theta    0.993   0.855   0.601   0.247   0.137   0.072    1.51 s
    alpha    0.970   0.516   0.220   0.094   0.056   0.036    0.45 s
    beta     0.881   0.153   0.071   0.029   0.014   0.006    0.13 s
    hfb      0.135   0.067   0.053   0.028   0.016   0.008    0.03 s

At a 20 ms step theta therefore has autocorrelation 0.993 one step ahead and
still ~0.14 at 50 steps -- one-step difficulty matched to Lorenz-96 (0.992) but
a decorrelation range 4-5x longer in step units.  That is the regime this
project needs and neither existing real dataset provides.

The files in ``preprocessed_length_matched`` are already band envelopes (their
autocorrelation decays monotonically rather than oscillating at the band
frequency) and are z-scored, hence the ~70% negative values.

Splitting.  Electrode counts differ between subjects, so a single model cannot
span them; the fMRI-style convention of one model per subject applies here too.
We therefore fit one subject and split that subject's recording along TIME into
contiguous train / val / test blocks, with a gap between blocks so no training
window can overlap an evaluation window.

Interface matches lorenz96.py.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from .base import SequenceDataset

_BANDS = ("delta", "theta", "alpha", "beta", "gamma", "hfb", "hfb_ext",
          "broadband_full", "broadband_high", "broadband_low")


def _find_files(root: Path, subject: str, task: str, contact: str, band: str) -> List[Path]:
    """Files are flat and named {subject}_{run}_{task}_{macro|micro}_{band}.fif."""
    pat = f"{subject}_*_{task}_{contact}_{band}.fif"
    hits = sorted(root.glob(pat))
    if not hits:
        raise FileNotFoundError(
            f"No iEEG files matching {pat} under {root}. "
            f"Known bands: {_BANDS}. Check --ieeg_subject/--ieeg_task/--ieeg_contact/--ieeg_band.")
    return hits


def _load_and_decimate(path: Path, step_ms: float, max_channels: int) -> Tuple[np.ndarray, float]:
    """Return [T, C] float32 at the requested step, plus the source sampling rate.

    The envelopes are already smooth on the timescale we decimate to, but we still
    low-pass before striding: scipy's decimate applies an anti-alias FIR, which
    matters because the raw rate (500 Hz) is 25x the target and any residual
    high-frequency content would fold back into exactly the slow band we are
    trying to measure.
    """
    import mne
    from scipy.signal import decimate

    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_fif(str(path), preload=True)
    sfreq = float(raw.info["sfreq"])
    X = raw.get_data()                                   # [C, T]
    if max_channels > 0:
        X = X[:max_channels]
    q = int(round(sfreq * step_ms / 1000.0))
    if q > 1:
        # decimate in stages when q is large; a single long FIR is ill-conditioned
        while q > 12:
            for f in (2, 3, 5):
                if q % f == 0:
                    X = decimate(X, f, ftype="fir", axis=-1, zero_phase=True)
                    q //= f
                    break
            else:
                break
        if q > 1:
            X = decimate(X, q, ftype="fir", axis=-1, zero_phase=True)
    return np.ascontiguousarray(X.T, dtype=np.float32), sfreq


def _chunk(x: np.ndarray, chunk_len: int) -> np.ndarray:
    """[T, C] -> [n, chunk_len, C], dropping the remainder."""
    n = x.shape[0] // chunk_len
    if n < 1:
        raise ValueError(f"segment of {x.shape[0]} steps is shorter than ieeg_chunk={chunk_len}")
    return np.ascontiguousarray(x[:n * chunk_len].reshape(n, chunk_len, x.shape[1]))


class IEEGDataset(SequenceDataset):
    dataset_name = "ieeg"
    has_external_input = False
    task_type = "sequence_vector"
    evaluator_name = "generic"

    def __init__(self, chunks: np.ndarray, mean: np.ndarray, std: np.ndarray, split: str = "train"):
        self.chunks = chunks                     # [n, T, C] float32, normalized
        self.mean = mean
        self.std = std
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


def build_ieeg_splits(args):
    root = Path(getattr(args, "ieeg_root",
                        "data/ieeg/preprocessed_length_matched"))
    subject = str(getattr(args, "ieeg_subject", "P41CS"))
    task = str(getattr(args, "ieeg_task", "enc"))
    contact = str(getattr(args, "ieeg_contact", "macro"))
    band = str(getattr(args, "ieeg_band", "theta"))
    step_ms = float(getattr(args, "ieeg_step_ms", 20.0))
    chunk_len = int(getattr(args, "ieeg_chunk", 1024))
    max_ch = int(getattr(args, "ieeg_max_channels", 0))
    gap = int(getattr(args, "ieeg_split_gap", 256))

    cache_dir = Path(args.data_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"ieeg_{subject}_{task}_{contact}_{band}_{step_ms:g}ms_ch{max_ch}.npz"

    if cache.exists():
        z = np.load(cache)
        X = z["X"]
        print(f"[iEEG] loaded cached {X.shape} from {cache}")
    else:
        files = _find_files(root, subject, task, contact, band)
        segs = []
        for f in files:
            seg, sf = _load_and_decimate(f, step_ms, max_ch)
            print(f"[iEEG] {f.name}: {sf:g} Hz -> {seg.shape[0]} steps x {seg.shape[1]} ch")
            segs.append(seg)
        # Runs are concatenated along time only after chunking boundaries are set,
        # so a chunk never straddles two recordings.
        X = np.concatenate(segs, axis=0) if len(segs) > 1 else segs[0]
        np.savez_compressed(cache, X=X)
        print(f"[iEEG] cached -> {cache}")

    T, C = X.shape
    n_tr = int(round(T * float(args.train_ratio)))
    n_va = int(round(T * float(args.val_ratio)))
    if n_tr + n_va >= T:
        n_tr, n_va = int(0.7 * T), int(0.15 * T)
    # contiguous blocks with a gap, so no training window overlaps an eval window
    tr = X[:max(chunk_len, n_tr - gap)]
    va = X[n_tr:n_tr + max(chunk_len, n_va - gap)]
    te = X[n_tr + n_va:]

    mean = tr.mean(axis=0, keepdims=True)
    std = tr.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    tr = (tr - mean) / std
    va = (va - mean) / std
    te = (te - mean) / std

    c_tr, c_va, c_te = _chunk(tr, chunk_len), _chunk(va, chunk_len), _chunk(te, chunk_len)
    args.roi_dim = C
    print(f"[iEEG] {subject}/{task}/{contact}/{band}  step={step_ms:g}ms  state_dim={C}  "
          f"total {T} steps -> train/val/test = {c_tr.shape[0]}/{c_va.shape[0]}/{c_te.shape[0]} "
          f"chunks of {chunk_len}")

    return (IEEGDataset(c_tr, mean, std, "train"),
            IEEGDataset(c_va, mean, std, "val"),
            IEEGDataset(c_te, mean, std, "test"))
