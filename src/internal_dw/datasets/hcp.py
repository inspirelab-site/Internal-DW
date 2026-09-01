import os
import random
from pathlib import Path
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import torch

from .base import SequenceDataset


def _as_dict(path: str):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == () and obj.dtype == object:
        obj = obj.item()
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a dict-like npy file at {path}, got {type(obj)}")
    return obj


def _load_fmri_and_stim(path: str, roi_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    data = _as_dict(path)

    # New preferred format.
    if "fmri" in data:
        fmri = np.asarray(data["fmri"], dtype=np.float32)
    # Backward-compatible old HCP rest/movie/rest format.
    elif all(k in data for k in ["fmri_rest1", "fmri_movie1", "fmri_rest2"]):
        fmri = np.concatenate([data["fmri_rest1"], data["fmri_movie1"], data["fmri_rest2"]], axis=1).astype(np.float32)
        fmri = fmri.T
    else:
        raise KeyError(f"No supported fMRI keys found in {path}")

    if "z" in data:
        stim = np.asarray(data["z"], dtype=np.float32)
    elif "clip" in data:
        stim = np.asarray(data["clip"], dtype=np.float32)
    elif all(k in data for k in ["clip_rest1", "clip_movie1", "clip_rest2"]):
        stim = np.concatenate([data["clip_rest1"], data["clip_movie1"], data["clip_rest2"]], axis=0).astype(np.float32)
    else:
        raise KeyError(f"No supported stimulus keys found in {path}")

    if fmri.ndim != 2:
        raise ValueError(f"Expected fMRI [T, ROI] or [ROI, T], got {fmri.shape} in {path}")
    if fmri.shape[0] == roi_dim and fmri.shape[1] != roi_dim:
        fmri = fmri.T

    if stim.ndim > 2:
        stim = stim.reshape(stim.shape[0], -1)

    T = min(fmri.shape[0], stim.shape[0])
    if T <= 1:
        raise ValueError(f"Sequence too short in {path}: fmri={fmri.shape}, stim={stim.shape}")
    return np.ascontiguousarray(fmri[:T]), np.ascontiguousarray(stim[:T])


class HCPMovieDataset(SequenceDataset):
    """HCP 7T movie fMRI dataset instance.

    Expected directory:
        data_path/
            subject_id/
                movie1.npy
                movie2.npy
                ...
    """

    dataset_name = "hcp_movie"
    has_external_input = True
    task_type = "sequence_vector"
    evaluator_name = "hcp"

    def __init__(self, files: List[str], subject_to_label: Dict[str, int], roi_dim: int = 400, visual_only: bool = False):
        self.files = list(files)
        self.subject_to_label = dict(subject_to_label)
        self.roi_dim = int(roi_dim)
        self.visual_only = bool(visual_only)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        subject_id = Path(path).parent.name
        fmri, stim = _load_fmri_and_stim(path, self.roi_dim)
        if self.visual_only:
            roi_idx = np.r_[0:30, 200:230, 91:113, 293:318, 331:361, 126:148]
            roi_idx = roi_idx[roi_idx < fmri.shape[1]]
            fmri = fmri[:, roi_idx]
        label = self.subject_to_label.get(subject_id, index)
        return {
            "state": torch.from_numpy(fmri),
            "external_input": torch.from_numpy(stim),
            "label": int(label),
            "metadata": {"path": str(path), "subject_id": str(subject_id), "dataset": self.dataset_name},
        }


def discover_hcp_files(data_path, movie):
    root = Path(data_path)

    files = []

    for subject_dir in sorted(root.iterdir()):
        if not subject_dir.is_dir():
            continue

        # Your actual files look like:
        # sub_100610_MOVIE1_100610.h5_lag6.npy
        pattern = f"*MOVIE{movie}*.npy"
        matches = sorted(subject_dir.glob(pattern))

        if len(matches) == 0:
            continue

        if len(matches) > 1:
            print(f"[WARN] Multiple files found for subject {subject_dir.name}, movie {movie}:")
            for m in matches:
                print(f"    {m}")
            print(f"[WARN] Using first one: {matches[0]}")

        files.append(matches[0])

    if len(files) == 0:
        raise FileNotFoundError(
            f"No files found under {root} with pattern SUBJECT/*MOVIE{movie}*.npy"
        )

    print(f"[HCP] Found {len(files)} files for movie {movie}")
    print(f"[HCP] Example file: {files[0]}")

    return files


def build_hcp_splits(args):
    files = discover_hcp_files(args.data_path, args.movie)
    seed = int(args.seed)
    rng = random.Random(seed)
    rng.shuffle(files)

    subjects = sorted({Path(f).parent.name for f in files})
    subject_to_label = {sid: i for i, sid in enumerate(subjects)}

    n = len(files)
    n_train = max(1, int(round(n * args.train_ratio)))
    n_val = max(1, int(round(n * args.val_ratio)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]
    if not test_files:
        test_files = val_files

    common = dict(subject_to_label=subject_to_label, roi_dim=args.roi_dim, visual_only=args.visual_only)
    return (
        HCPMovieDataset(train_files, **common),
        HCPMovieDataset(val_files, **common),
        HCPMovieDataset(test_files, **common),
    )
