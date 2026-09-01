"""Local The Well dataset wrapper for Koopman-Gram experiments.

Supported Well dataset names / aliases:
    - gray_scott, gray_scott_reaction_diffusion
    - turbulent_flow, turbulent_radiative_layer_2D
    - rayleigh_benard
    - shear_flow
    - viscoelastic_instability

This file does not download data. It assumes the corresponding HDF5 files are
already present locally. It preserves field structure and returns state as
[T, C, H, W] for 2D systems.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .base import SequenceDataset

try:  # optional dependency; only required when using this dataset
    import h5py
except Exception:  # pragma: no cover
    h5py = None


THEWELL_NAME_ALIASES: Dict[str, str] = {
    "gray_scott": "gray_scott_reaction_diffusion",
    "gray-scott": "gray_scott_reaction_diffusion",
    "gray_scott_reaction_diffusion": "gray_scott_reaction_diffusion",
    "turbulent_flow": "turbulent_radiative_layer_2D",
    "turbulent-flow": "turbulent_radiative_layer_2D",
    "turbulent_radiative_layer_2D": "turbulent_radiative_layer_2D",
    "rayleigh_benard": "rayleigh_benard",
    "rayleigh-benard": "rayleigh_benard",
    "shear_flow": "shear_flow",
    "shear-flow": "shear_flow",
    "viscoelastic_instability": "viscoelastic_instability",
    "viscoelastic-instability": "viscoelastic_instability",
}

SUPPORTED_THEWELL_DATASETS = sorted(set(THEWELL_NAME_ALIASES.values()))


def canonical_thewell_name(name: str) -> str:
    key = str(name).strip()
    if key not in THEWELL_NAME_ALIASES:
        raise ValueError(
            f"Unsupported The Well dataset '{name}'. Supported aliases: {sorted(THEWELL_NAME_ALIASES.keys())}"
        )
    return THEWELL_NAME_ALIASES[key]


def _decode_attr_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (str, bytes)):
        xs = [x]
    else:
        xs = list(x)
    out = []
    for item in xs:
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return out


def _find_hdf5_files(data_path: str, dataset_name: str, split: str) -> List[str]:
    """Search common local layouts for The Well HDF5 files."""
    root = Path(data_path).expanduser()
    candidates = [
        root,
        root / split,
        root / dataset_name,
        root / dataset_name / split,
        root / split / dataset_name,
    ]
    files: List[str] = []
    for c in candidates:
        if c.exists():
            files.extend(str(p) for p in sorted(c.glob("*.h5")))
            files.extend(str(p) for p in sorted(c.glob("*.hdf5")))
    # de-duplicate while preserving order
    seen = set()
    unique = []
    for f in files:
        if f not in seen:
            unique.append(f)
            seen.add(f)
    if not unique:
        raise FileNotFoundError(
            "No The Well HDF5 files found. Tried layouts like: "
            f"{root}/, {root}/{split}/, {root}/{dataset_name}/{split}/"
        )
    return unique


def _infer_n_steps_and_traj(h5) -> Tuple[int, int]:
    n_traj = int(h5.attrs.get("n_trajectories", 0) or 0)
    n_steps = 0
    for group_name in ("t0_fields", "t1_fields", "t2_fields"):
        if group_name not in h5:
            continue
        group = h5[group_name]
        for field_name in _decode_attr_list(group.attrs.get("field_names", [])):
            if field_name not in group:
                continue
            ds = group[field_name]
            sample_varying = bool(ds.attrs.get("sample_varying", True))
            time_varying = bool(ds.attrs.get("time_varying", True))
            shape = ds.shape
            if sample_varying and len(shape) > 0 and n_traj <= 0:
                n_traj = int(shape[0])
            if time_varying:
                t_axis = 1 if sample_varying else 0
                if len(shape) > t_axis:
                    n_steps = max(n_steps, int(shape[t_axis]))
    if n_traj <= 0:
        n_traj = 1
    if n_steps <= 0:
        if "dimensions" in h5 and "time" in h5["dimensions"]:
            n_steps = int(len(h5["dimensions"]["time"]))
        else:
            raise ValueError("Could not infer number of time steps from The Well HDF5 file.")
    return n_traj, n_steps


def _read_field(ds, sample_idx: int, start: int, length: int, n_spatial_dims: int, time_subsample: int = 1) -> np.ndarray:
    """Read one Well field as [T, *spatial, C]."""
    sample_varying = bool(ds.attrs.get("sample_varying", True))
    time_varying = bool(ds.attrs.get("time_varying", True))

    idx = []
    if sample_varying:
        idx.append(int(sample_idx))
    if time_varying:
        step = max(int(time_subsample), 1)
        idx.append(slice(int(start), int(start) + (int(length) - 1) * step + 1, step))
    arr = np.asarray(ds[tuple(idx)] if idx else ds[...], dtype=np.float32)

    if not time_varying:
        arr = np.expand_dims(arr, axis=0)
        arr = np.repeat(arr, int(length), axis=0)

    # Expected after indexing: [T, spatial..., component_dims...] for time-varying,
    # or [T, spatial...] for scalar constant fields.
    if arr.ndim < 1 + n_spatial_dims:
        raise ValueError(f"Field shape too small after indexing: shape={arr.shape}, n_spatial_dims={n_spatial_dims}")

    # For scalar fields, add channel. For vector/tensor fields, flatten component dims.
    if arr.ndim == 1 + n_spatial_dims:
        arr = arr[..., None]
    else:
        spatial = arr.shape[1:1 + n_spatial_dims]
        comp = int(np.prod(arr.shape[1 + n_spatial_dims:]))
        arr = arr.reshape(arr.shape[0], *spatial, comp)
    return arr


class TheWell2DDataset(SequenceDataset):
    """The Well 2D field dataset wrapper.

    Returns:
        state: [T, C, H, W]
        external_input: zero dummy tensor [T, input_dim]

    Notes:
        The Well's package-level WellDataset returns B x T x H x W x C format;
        the raw HDF5 spec stores fields in groups such as t0_fields/t1_fields.
        This wrapper directly reads local HDF5 files and converts to channel-first
        PyTorch format for image/field models.
    """

    dataset_name = "the_well"
    has_external_input = False
    task_type = "field2d"
    evaluator_name = "field2d"

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        well_dataset_name: str = "gray_scott_reaction_diffusion",
        sequence_length: int = 0,
        sequence_stride: int = 1,
        input_dim: int = 1,
        field_groups: Optional[Sequence[str]] = None,
        max_trajectories_per_file: int = 0,
        time_subsample: int = 1,
        spatial_subsample: int = 1,
    ):
        if h5py is None:
            raise ImportError("Using dataset=the_well requires h5py. Install it with `pip install h5py`.")
        self.well_dataset_name = canonical_thewell_name(well_dataset_name)
        self.split = str(split)
        self.data_path = str(data_path)
        self.sequence_length = int(sequence_length)
        self.sequence_stride = max(int(sequence_stride), 1)
        self.input_dim = int(input_dim)
        self.field_groups = tuple(field_groups or ("t0_fields", "t1_fields"))
        self.max_trajectories_per_file = int(max_trajectories_per_file)
        self.time_subsample = max(int(time_subsample), 1)
        self.spatial_subsample = max(int(spatial_subsample), 1)
        self.files = _find_hdf5_files(self.data_path, self.well_dataset_name, self.split)

        self.index: List[Tuple[str, int, int, int]] = []  # file, traj, start, length
        for path in self.files:
            with h5py.File(path, "r") as h5:
                n_traj, n_steps = _infer_n_steps_and_traj(h5)
            if self.max_trajectories_per_file > 0:
                n_traj = min(n_traj, self.max_trajectories_per_file)
            length = self.sequence_length if self.sequence_length > 0 else n_steps
            raw_span = (length - 1) * self.time_subsample + 1
            if raw_span > n_steps:
                raise ValueError(f"sequence_length={length} with time_subsample={self.time_subsample} requires raw_span={raw_span}, exceeds n_steps={n_steps} in {path}")
            starts = range(0, n_steps - raw_span + 1, self.sequence_stride)
            for traj in range(n_traj):
                for start in starts:
                    self.index.append((path, traj, int(start), int(length)))
        if not self.index:
            raise RuntimeError(f"No samples indexed for The Well dataset at {self.data_path}")

    def __len__(self):
        return len(self.index)

    def _read_state(self, path: str, traj_idx: int, start: int, length: int) -> torch.Tensor:
        fields = []
        with h5py.File(path, "r") as h5:
            n_spatial_dims = int(h5.attrs.get("n_spatial_dims", 2))
            if n_spatial_dims != 2:
                raise ValueError(f"TheWell2DDataset only supports 2D data; got n_spatial_dims={n_spatial_dims} in {path}")
            for group_name in self.field_groups:
                if group_name not in h5:
                    continue
                group = h5[group_name]
                for field_name in _decode_attr_list(group.attrs.get("field_names", [])):
                    if field_name not in group:
                        continue
                    field = _read_field(group[field_name], traj_idx, start, length, n_spatial_dims, self.time_subsample)
                    fields.append(field)
        if not fields:
            raise ValueError(f"No fields found in groups={self.field_groups} for {path}")
        # [T,H,W,C_total] -> [T,C,H,W]
        state = np.concatenate(fields, axis=-1)
        state = np.moveaxis(state, -1, 1)
        if self.spatial_subsample > 1:
            # Deterministic strided pilot grid.  This is deliberately applied
            # after all scalar/vector/tensor fields have been aligned, so every
            # channel observes exactly the same spatial coordinates.  It makes
            # high-resolution Well systems practical for mechanism screening;
            # the factor is recorded in sample metadata and checkpoint args.
            state = state[:, :, :: self.spatial_subsample, :: self.spatial_subsample]
        return torch.from_numpy(np.ascontiguousarray(state.astype(np.float32)))

    def __getitem__(self, index):
        path, traj_idx, start, length = self.index[index]
        state = self._read_state(path, traj_idx, start, length)
        external_input = torch.zeros(state.shape[0], self.input_dim, dtype=state.dtype)
        return {
            "state": state,
            "external_input": external_input,
            "label": int(traj_idx),
            "metadata": {
                "dataset": self.dataset_name,
                "well_dataset_name": self.well_dataset_name,
                "path": path,
                "trajectory_index": int(traj_idx),
                "start": int(start),
                "length": int(length),
                "time_subsample": int(self.time_subsample),
                "spatial_subsample": int(self.spatial_subsample),
                "state_format": "TCHW",
            },
        }


def build_thewell_splits(args):
    name = getattr(args, "thewell_dataset_name", "gray_scott_reaction_diffusion")
    field_groups = tuple(str(getattr(args, "thewell_field_groups", "t0_fields,t1_fields")).split(","))
    common = dict(
        data_path=args.data_path,
        well_dataset_name=name,
        sequence_length=int(getattr(args, "thewell_sequence_length", 0)),
        sequence_stride=int(getattr(args, "thewell_sequence_stride", 1)),
        input_dim=int(getattr(args, "stim_dim", 1)),
        field_groups=[g.strip() for g in field_groups if g.strip()],
        max_trajectories_per_file=int(getattr(args, "thewell_max_trajectories_per_file", 0)),
        time_subsample=int(getattr(args, "thewell_time_subsample", 1)),
        spatial_subsample=int(getattr(args, "thewell_spatial_subsample", 1)),
    )
    return (
        TheWell2DDataset(split="train", **common),
        TheWell2DDataset(split="valid", **common),
        TheWell2DDataset(split="test", **common),
    )
