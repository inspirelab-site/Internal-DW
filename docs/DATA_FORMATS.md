# Dataset layout and input contracts

This document describes the files consumed by the released loaders. Paths may
be repository-relative or absolute; pass a non-default path through the
environment variable shown for each dataset. Raw datasets and generated caches
are intentionally excluded from Git.

## Common in-memory sample contract

Every dataset adapter returns one dictionary per sequence:

```python
{
    "state": Tensor,                    # [T, D] or [T, C, H, W]
    "external_input": Tensor | None,    # [T, S] when present
    "label": int | Tensor,
    "metadata": dict,
}
```

Time is always the first dimension of a single sample. The collated batch is
therefore `[B, T, ...]`. Vector datasets use `state.shape == [T, D]`; spatial
datasets use channel-first `state.shape == [T, C, H, W]`. An external input
must have the same `T` as its state. Loaders convert numerical arrays to
`float32`; supplying finite `float32` arrays avoids an unnecessary copy.

A new adapter should subclass `SequenceDataset` in
`src/internal_dw/datasets/base.py`, return the dictionary above, and be
registered in `src/internal_dw/datasets/registry.py`.

## Recommended top-level layout

```text
data/
  synthetic/                         # generated MG, NARMA, known-SNR caches
  hcp_movie_features/
    <subject_id>/
      <anything>MOVIE1<anything>.npy
  ieeg/
    preprocessed_length_matched/
      <subject>_<run>_<task>_<contact>_<band>.fif
  weatherbench2_1p5_pilot/
    metadata.json
    latitude.npy
    longitude.npy
    train_state.npy
    train_time_features.npy
    val_state.npy
    val_time_features.npy
    test_state.npy
    test_time_features.npy

probe_inputs/
  temporal_candidate_regime_v1/
    ettm1.npz
    ettm2.npz

external/
  the_well/
    gradient_pilots/
      datasets/
        shear_flow/
          data/
            train/*.h5
            valid/*.h5
            test/*.h5
```

Only the datasets required for a selected experiment need to be present.

## Synthetic MG, NARMA, and known-SNR AR

No download or manual file construction is required. The dataset loaders
generate trajectories on first use and reuse a parameter-specific compressed
cache on later runs. `DATA_DIR` selects the cache directory and defaults to
`data/synthetic` in `scripts/train/run_mem_one.sh`.

The generated caches use the following internal arrays:

| Dataset | NPZ arrays | Shape |
|---|---|---|
| Mackey--Glass | `trajs` | `[trajectory, time, state_dim]` |
| Driven Mackey--Glass | `trajs`, `drives` | both `[trajectory, time, state_dim]` |
| NARMA | `y`, `u` | both `[trajectory, time, state_dim]` |
| known-SNR AR | `trajs`, `coefficients` | `[trajectory, time, state_dim]`, `[state_dim]` |

Do not rename a generated cache: its filename records all data-generation
parameters and prevents incompatible runs from silently sharing data.

## ETTm1 and ETTm2

The preparation script accepts either of these upstream layouts:

```text
<ETT_ROOT>/ETT-small/ETTm1.csv
<ETT_ROOT>/ETT-small/ETTm2.csv
```

or

```text
<ETT_ROOT>/ETDataset-main/ETT-small/ETTm1.csv
<ETT_ROOT>/ETDataset-main/ETT-small/ETTm2.csv
```

Each CSV must have `date` as its first column followed by numerical state
columns. Prepare the standard chronological splits with:

```bash
python scripts/data/prepare_temporal_candidate_screen_data.py \
  --only ett \
  --ett-root /path/to/ETDataset \
  --output-root probe_inputs/temporal_candidate_regime_v1
```

The resulting `ettm1.npz` and `ettm2.npz` contain:

```text
train_state, validation_state, test_state    float32 [S, T, D]
train_drive, validation_drive, test_drive    float32 [S, T, 6]
metadata_json                                scalar JSON string
```

`S` is the number of non-overlapping chunks, `T` is the preparation
`--chunk` value (256 by default), and `D` is the number of ETT variables. The
six drive channels are sine/cosine encodings of hour, day of week, and day of
year. Set `PREPARED_NPZ=/absolute/path/ettm1.npz` for a runner, or
`PREPARED_INPUT_ROOT` for the directory containing all prepared archives.
Only training-split statistics are used for standardization.

The same archive contract is accepted for another vector time series. An
autonomous archive omits all three `*_drive` arrays; a driven archive must
provide all three and align their first two dimensions with `*_state`.

## Movie iEEG

The raw input directory is flat:

```text
data/ieeg/preprocessed_length_matched/
  P41CS_R1_enc_macro_theta.fif
  P41CS_R2_enc_macro_theta.fif
  ...
```

Files are discovered with
`<subject>_*_<task>_<contact>_<band>.fif`. The paper default is subject
`P41CS`, task `enc`, contact `macro`, band `theta`, and a 20 ms model step.
Pass a different raw directory with:

```bash
EXTRA_ARGS="--ieeg_root /path/to/preprocessed_length_matched" \
  DATASET=ieeg COND=theta K=64 METHOD=dwstructured SEED=0 GPU=0 \
  bash scripts/train/run_mem_one.sh
```

Reading raw FIF files requires MNE and SciPy. The loader anti-alias filters and
decimates each run, concatenates the time axis, and writes a reusable cache
named `ieeg_<subject>_<task>_<contact>_<band>_<step>ms_ch<max>.npz` under
`DATA_DIR`. That cache contains one `float32` array `X` with shape `[T, C]`.
Train/validation/test blocks and normalization are constructed by the loader;
the raw recording must not be manually pre-split.

## HCP movie fMRI

Set `DATA_PATH` (training) or `HCP_DATA_PATH` (frozen-checkpoint probes) to a
directory containing one subdirectory per subject:

```text
data/hcp_movie_features/
  100610/
    sub_100610_MOVIE1_100610.h5_lag6.npy
  102311/
    sub_102311_MOVIE1_102311.h5_lag6.npy
  ...
```

The filename may vary, but it must match `*MOVIE<movie>*.npy`. There should be
one match per subject and movie; if several match, the loader warns and uses
the first lexicographic path.

Each preferred-format `.npy` is a pickled Python dictionary created with
`np.save`, not a bare matrix:

```python
payload = {
    "fmri": fmri.astype("float32"),  # [T, ROI], or [ROI, T]
    "z": stimulus.astype("float32"), # [T, S]
}
np.save(output_path, payload, allow_pickle=True)
```

`clip` is accepted as an alias for `z`. Extra stimulus dimensions are flattened
to `[T, S]`; fMRI and stimulus are truncated to their common time length. The
paper runner uses `ROI=400`. The loader also supports the legacy
`fmri_rest1/fmri_movie1/fmri_rest2` and
`clip_rest1/clip_movie1/clip_rest2` keys, but new data should use `fmri` and
`z`.

## The Well shear flow

The public runner requires all three upstream splits:

```text
external/the_well/gradient_pilots/datasets/shear_flow/data/
  train/*.h5
  valid/*.h5
  test/*.h5
```

Set `WELL_REPO` to the The Well checkout root, `PILOT_BASE` to the directory
containing `datasets/`, or `DATA_PATH` directly to the directory containing
`train/valid/test`. The helper `scripts/data/download_thewell_registry.py`
downloads registry entries without modifying their HDF5 contents; it does not
replace the upstream license.

Files must follow The Well HDF5 specification. In particular, the root records
`n_spatial_dims=2` and the field groups (`t0_fields`, `t1_fields`, optionally
`t2_fields`) list their datasets in the `field_names` attribute. Field datasets
must carry the standard `sample_varying` and `time_varying` attributes. The
loader combines all selected scalar/vector/tensor fields and returns
`float32 [T, C, H, W]`. It does not accept an arbitrary HDF5 tensor with an
unrelated key.

## WeatherBench-2

The training loader does not access cloud Zarr. First materialize a local,
train-normalized pilot:

```bash
python scripts/data/prepare_weatherbench2_pilot.py \
  --out data/weatherbench2_1p5_pilot

python scripts/data/validate_weatherbench2_pilot.py \
  --data data/weatherbench2_1p5_pilot
```

Cloud preprocessing additionally requires `xarray`, Zarr v2, `gcsfs`, and
their storage dependencies. The completed directory contains `metadata.json`,
`latitude.npy`, `longitude.npy`, and two files per split. The exact filenames
are read from `metadata.json`; the defaults are illustrated in the recommended
layout above.

The arrays must satisfy:

```text
<split>_state.npy          float32 [T, C, H, W]
<split>_time_features.npy  float32 [T, 4]
latitude.npy               float32 [H]
longitude.npy              float32 [W]
```

The four time features are sine/cosine encodings of hour and day of year.
`metadata.json` must provide `channel_names`, `channels`, `height`, `width`, and
`splits.{train,val,test}.{steps,state_file,time_features_file}`. Set
`DATA_PATH=/path/to/weatherbench2_1p5_pilot` when invoking
`scripts/train/run_wb2_arm.sh`.

## Checkpoint and test-output layout

Training creates an experiment directory rather than a single user-authored
checkpoint file:

```text
<SAVE_BASE>/<dataset-and-arm>/seed0/
  best.pth          # lowest validation loss; used for final testing
  last.pth          # latest resumable optimizer/training state
  train_logs.jsonl
  eval_results.json # produced by the low-level train-and-test runner
```

The released reproduction wrappers print the selected `best.pth` and the
separate compact test JSON path. Keep the directory hierarchy intact when
unpacking a result bundle because the paper assemblers use stable relative
paths. `SAVE_BASE` changes checkpoint placement; `OUT_ROOT` changes compact
test-result placement for wrappers that expose it.
