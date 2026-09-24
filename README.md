# Internal-DW

Official research code for **Large Distant Gradients Need Not Be Reliable:
Reliability-Weighted Credit Assignment for Long-Horizon Autoregressive
Forecasting**.

Internal-DW is a backward-only operator for residual autoregressive models. It
keeps the forward computation unchanged and applies automatically estimated
Wiener gains to the identity and nonlinear route messages during backpropagation.

## Install

Python 3.10+ is required. Create and activate the Conda environment, then
install the PyTorch version used for the paper:

```bash
conda create -n internal-dw python=3.11 pip -y
conda activate internal-dw
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python scripts/utils/check_install.py --require-conda --require-cuda && \
python -m pip install -e ".[paper,test]" && \
python -m pip check && \
bash scripts/reproduce/00_smoke_test.sh
```

On a multi-GPU host, verify that PyTorch sees every CUDA device and that NCCL
collectives work before launching a paper run:

```bash
python scripts/utils/check_ddp_cuda.py
```

The reusable operator itself depends only on NumPy and PyTorch. The `paper`
extra installs the scientific-data, plotting, and tabular dependencies used by
the released experiments.

## Reproduce the known-SNR closure

This experiment generates its own data. Run the complete numerical experiment
on one GPU (no DDP):

```bash
GPU=0 bash scripts/reproduce/train_test_known_snr.sh
```

Settings are in `configs/reproduce/known_snr.json`. Clip, JReg, and TBPTT
are selected using seed-0 training `val/loss`; the selected settings, Full
BPTT, and Internal-DW are then trained and tested over seeds 0, 1, and 2.
Testing uses the common dense evaluator: a continuous 48-step rollout,
with equal weighting over horizons, origins, and trajectories.

Stages can also be run separately: `prepare`, `screen`, `train`, `test`,
`probes` (gradient profiles and local risk), or `risk` (local risk only).
The frozen probes retain their recorded workers=4 reference training;
all five forecasting methods use workers=0, batch=4, accumulation=8.
The risk probe keeps the reference network weights fixed and updates only
the lagged DW controller over two training-stream passes.

Outputs under `experiments/known_snr/`:

- `training/<method>/<value>/seed<seed>/best.pth`: selected checkpoint.
- `selection.json`: candidate validation losses and selected parameters.
- `test/<method>/seed<seed>.json`: test metric in `primary_metric.value`.
- `forecast_summary.json`: three-seed means and sample standard deviations.
- `profiles/panels_1_2.json` and `risk/summary.json`: mechanism measurements.

Use `KNOWN_SNR_ROOT` and `KNOWN_SNR_DATA_DIR` to change output and data
locations. Plotting is separate:

```bash
bash scripts/reproduce/03_figure_4.sh
```

## Train and test a model

The MG example is the shortest complete training-to-test workflow. Run the two
matched arms separately:

```bash
ARM=full_bptt SEED=0 GPU=0 \
  bash scripts/reproduce/train_test_mg.sh

ARM=internal_dw SEED=0 GPU=0 \
  bash scripts/reproduce/train_test_mg.sh
```

The canonical paper settings are stored in `configs/reproduce/`. The launcher
loads the corresponding Full-BPTT or Internal-DW configuration and prints the
resolved world size, local batch, accumulation count, and effective batch
before training. With four GPUs, use `GPUS=0,1,2,3` instead of `GPU=0`; the
launcher preserves the recorded global microbatch and accumulation schedule.
Explicit `BATCH` or `GRAD_ACCUM` values override the paper defaults.

Each command first trains with validation-based checkpoint selection and then
evaluates the selected checkpoint once on the test split. Training and testing
can also be invoked independently:

```bash
ARM=internal_dw SEED=0 GPU=0 \
  bash scripts/reproduce/train_test_mg.sh train

ARM=internal_dw SEED=0 GPU=0 \
  bash scripts/reproduce/train_test_mg.sh test
```

By default, the files are written directly to the locations consumed by the
paper result assembler. Set `SAVE_BASE` and `OUT_ROOT` to place checkpoints and
test records elsewhere. A completed Internal-DW run prints the two exact paths:

```text
[checkpoint] experiments/internal_dw_assigned_v1/mackey_glass/tau30_K32/dualwiener_structured/seed0/best.pth
[test-json] probe_outputs/dense_multistart_rel_l2_1p5k_v1/mg/dw_seed0.json
[result] mean_relative_l2=... horizons=1:48 lower_is_better=True
```

The test JSON is intentionally compact. Its first fields identify the dataset,
method, seed, split, primary metric, and selected checkpoint. `summary` contains
the aggregate in-horizon, out-of-horizon, and full-range metrics;
`per_horizon` contains the curve used by plotting code. It does not include
training diagnostics or per-example prediction arrays. For example:

```bash
python -c 'import json; r=json.load(open("probe_outputs/dense_multistart_rel_l2_1p5k_v1/mg/dw_seed0.json")); print(r["primary_metric"]); print(r["checkpoint"])'
```

Training/testing and plotting are separate. Paper plots only read completed
test/probe JSON files; they never launch training. After the required result
files have been generated, render one paper item or all available items with:

```bash
bash scripts/reproduce/05_figure_6.sh
bash scripts/reproduce/render_all.sh
```

### Optional validation-selected Clip/JReg controls

The Figure 6 one-command launcher uses the final validation-selected settings
in `configs/reproduce/figure6.sh` and `configs/reproduce/ieeg.json`.
To repeat parameter selection instead, run:

```bash
GPUS=0,1,2,3 bash scripts/reproduce/sweep_clip_jreg.sh
```

This runs four independent single-GPU jobs, not DDP. Each method receives
three seed-0 candidates: Clip thresholds `0.1, 0.3, 1.0` and JReg coefficients
`0.01, 0.1, 1.0`. Candidates are trained without test evaluation; the minimum
saved training `val/loss` selects one value per dataset and method. Dense
validation relative-L2 is not used for parameter selection. The
screening stops after writing the selected value; it does not run test or
start seeds 1/2 by default. For iEEG, selection uses
the equal-weight mean validation score across all 16 participants, with one
shared coefficient. Other training settings retain the Figure 6 defaults.

Results are written under `experiments/clip_jreg_val_sweep_v1/`. Each method
has a `selection.json`; `sweep_summary.json` records choices, test means/sample
standard deviations across seed-level averages, and failures. Rerun the same
command to resume; failed jobs do not stop the remaining jobs, and selection
requires all candidates (and all iEEG participants). Do not run two copies
against the same `SWEEP_ROOT` simultaneously.

Use `--datasets mg ettm1 ettm2 shear` for a subset, `--stage selected` to
explicitly start selected three-seed train/test runs after reviewing the screening,
or `--dry-run` to print the plan. `JOBS_PER_GPU=2` enables two independent jobs
per GPU without changing per-run batches. Dataset locations
are supplied with `PREPARED_INPUT_ROOT` (ETTm archives), `IEEG_PREPARED_ROOT`
(containing `prepared/sub-CS*.npz`), `SHEAR_DATA_PATH` (containing train/valid/test),
`HCP_DATA_PATH`, and `WB2_DATA_PATH`. Set `WELL_REPO` when its data utilities
are kept outside the checkout.

### Movie iEEG: prepare, train, and test

The iEEG benchmark uses **16 participants with subject-specific models and visual stimulus**, not
the earlier single-subject/no-stimulus experiment. Prepare only iEEG with:

```bash
FIF_ROOT=/path/to/preprocessed_length_matched \
CLIP_ROOT=/path/to/clip_features \
  bash scripts/reproduce/train_test_ieeg.sh prepare
```

This command reads the existing preprocessed theta FIF files and aligned 25-fps
CLIP features; it does not download data or prepare any other dataset. See
[`docs/DATA_FORMATS.md#movie-ieeg`](docs/DATA_FORMATS.md#movie-ieeg) for the required
files and preprocessing contract. Append `--subject sub-CS41` to prepare just
one participant. Training never automatically repeats preparation.

```bash
# All 16 participants, Full BPTT / Internal-DW / Clip / JReg, seeds 0/1/2.
# Runs sequentially on ONE GPU; no DDP or server-specific scheduler.
GPU=0 bash scripts/reproduce/train_test_ieeg.sh

# One participant/method/seed; train and test can also be called separately.
SUBJECTS=sub-CS41 ARM=internal_dw SEED=0 GPU=0 \
  bash scripts/reproduce/train_test_ieeg.sh train
SUBJECTS=sub-CS41 ARM=internal_dw SEED=0 GPU=0 \
  bash scripts/reproduce/train_test_ieeg.sh test
```

Defaults are recorded in `configs/reproduce/ieeg.json`: K=64, test H=96,
batch=4, accumulation=8, up to 100 epochs, validation patience=20, and no
activation checkpointing. The shared model already accepts stimulus;
Internal-DW builds one history-only noise-template bank from each participant's
own training chunks and shares it across training seeds. Compact recurrent
logging removes optional diagnostics, not DW calibration or training losses.

Inputs default to `probe_inputs/ieeg_cohort_v1/prepared/sub-CS*.npz`.
Each run writes `best.pth`, `last.pth`, `train.log`, `train_logs.jsonl`, and
`test.json` in `experiments/ieeg_cohort_v1/<subject>/<method>/seedN/` and prints
the checkpoint/result paths. Read `test.json -> primary_metric.value` for mean
relative L2 over horizons 1:96. Rerunning the same command resumes checkpoints,
skips validated completed tests, and continues past failed tasks. Do not launch
overlapping queues on different servers. Use `--dry-run` to inspect commands.

After all runs finish, `bash scripts/reproduce/train_test_ieeg.sh summary`
writes `cohort_summary.json`. Each seed first averages participants equally;
the table reports the mean and sample SD across the three seed averages.
Percentage changes are paired within participant and seed before averaging.
Figure 6 now reads these cohort results exclusively; incomplete cohorts do not
fall back to old single-participant results.

The same prepared subjects feed the existing probes and timing measurement:

```bash
GPU=0 bash scripts/reproduce/train_test_ieeg.sh regime
GPU=0 bash scripts/reproduce/train_test_ieeg.sh utility
GPU=0 bash scripts/reproduce/train_test_ieeg.sh noise
# Use an idle, unshared GPU after the Full/DW runs and their templates exist.
GPU=0 bash scripts/reproduce/train_test_ieeg.sh timing
```

Probes use seed 0 and preserve the existing per-subject definitions. Regime
scores are averaged before display clipping. Utility bands describe variation
across participant median curves, not training seeds; noise gains weight
participants equally. Timing warms both arms up for 5 or 10 epochs (depending
on chunk count), verifies DW calibration, and measures 3 training-loop epochs
per run, excluding validation, checkpoint I/O and template construction. It
reports mean and sample SD across three paired repeat-level participant means
in `experiments/ieeg_cohort_timing_v1/timing_summary.json`.

These public commands use fresh output directories and do not resume or
overwrite the private `ieeg_fif_visual_v3` training queues.

### One-command Figure 6 reproduction

After preparing the eight datasets described below, the entire Figure 6
benchmark can be trained and tested sequentially on one GPU. In particular,
ETTm1, ETTm2, and The Well shear flow are external datasets and are not bundled
with this repository. Download and prepare them before launching the full
queue:

```bash
# ETTm1 and ETTm2
git clone --depth 1 https://github.com/zhouhaoyi/ETDataset.git external/ETDataset
python scripts/data/prepare_temporal_candidate_screen_data.py \
  --only ett \
  --ett-root external/ETDataset \
  --output-root probe_inputs/temporal_candidate_regime_v1

# The Well shear flow (all three upstream splits)
git clone --depth 1 https://github.com/PolymathicAI/the_well.git external/the_well
for split in train valid test; do
  python scripts/data/download_thewell_registry.py \
    --registry external/the_well/the_well/utils/registry.yaml \
    --base-path external/the_well/gradient_pilots \
    --dataset shear_flow \
    --split "${split}" \
    --parallel
done
```

These commands must produce the following inputs:

```text
probe_inputs/temporal_candidate_regime_v1/ettm1.npz
probe_inputs/temporal_candidate_regime_v1/ettm2.npz
external/the_well/gradient_pilots/datasets/shear_flow/data/{train,valid,test}/*.h5
```

See [`docs/DATA_FORMATS.md`](docs/DATA_FORMATS.md) for accepted alternate paths,
environment-variable overrides, array keys, and shapes. Once the required
inputs are present, run:

```bash
GPU=0 bash scripts/reproduce/train_test_figure6_all.sh
```

The queue runs the four favorable cases first (MG, ETTm1, ETTm2, and shear
flow), followed by NARMA-5, iEEG, movie fMRI, and WeatherBench-2. Within each
dataset it completes every reported arm and seeds 0, 1, and 2 before moving to
the next dataset. Clip, JReg, Static gain, and TBPTT use the validation-selected
values recorded in `configs/reproduce/figure6.sh` (iEEG uses
`configs/reproduce/ieeg.json`). The queue is resumable: nonempty
test JSON files are skipped and interrupted checkpoints resume automatically.
Failure is isolated by dataset: a missing or malformed input is recorded and
the queue continues with every later dataset. After all datasets have been
attempted, the command returns a nonzero status and writes
`probe_outputs/figure6_reproduction_failures.txt` if anything failed. Fixing
the listed inputs and rerunning the same command resumes only missing work.
When all inputs are complete, the queue also renders Figure 6.

For a short queue check restricted to MG, use:

```bash
GPU=0 SEEDS=0 REPRO_DATASETS=mg RENDER_FIGURE6=0 \
  bash scripts/reproduce/train_test_figure6_all.sh
```

External datasets are not downloaded by the training queue. A missing prepared
input fails only its own dataset rather than discarding the rest of the queue.

## Add Internal-DW to a model

The only architectural edit is made **before** each residual branch:

```python
from internal_dw import InternalDW

dw = InternalDW(
    state_dim=hidden_size,
    num_layers=num_residual_blocks,
    max_horizon=training_horizon,
)

def residual_block(x, horizon, layer, nonlinear):
    skip_x, branch_x = dw.route(x, horizon=horizon, layer=layer)
    return skip_x + nonlinear(branch_x)
```

Inputs:

- `x`: tensor entering the residual merge;
- `horizon`: zero-based autoregressive forecast step;
- `layer`: zero-based residual-block index.

Output: `(skip_x, branch_x)`, two tensors with the same shape, dtype, device,
and **exact forward values** as `x`. Their backward vector-Jacobian products
are multiplied by the current identity gain `alpha` and nonlinear gain `m`.
Local parameter gradients inside `nonlinear` remain open.

One training batch uses this lifecycle:

```python
optimizer.zero_grad(set_to_none=True)
dw.begin_batch()

losses = []
state = initial_state
for k, target in enumerate(targets.unbind(dim=1)):
    state = model_step(state, horizon=k)  # model_step calls dw.route(...)
    losses.append(criterion(state, target))
    dw.observe(state, target, horizon=k)

loss = torch.stack(losses).mean()
dw.calibrate()   # read-only VJP probes; does not fill parameter .grad
loss.backward()  # ordinary optimizer gradient, routed with lagged gains
dw.end_batch()   # commits newly estimated gains for the next batch
optimizer.step()
```

Use `dw.gains` for the `[horizon, layer, 2]` tensor, `dw.diagnostics()` for
logger-ready scalars, and `dw.export_state()` for a JSON-safe audit record.
The complete runnable example is
[`examples/quickstart_internal_dw.py`](examples/quickstart_internal_dw.py), and
the precise API contract is in
[`docs/INTERNAL_DW_API.md`](docs/INTERNAL_DW_API.md).

## Repository layout

```text
src/internal_dw/router.py        public plug-in API
src/internal_dw/models/          paper backbones (Mamba, U-Net) and DW controller
src/internal_dw/datasets/        benchmark dataset adapters
src/internal_dw/training/        losses, routing, and training loop
src/internal_dw/evaluation/      rollout evaluation utilities
examples/                        minimal third-party integration
scripts/reproduce/               stable Figure/Table entry points
scripts/data/                    data and train-only prior preparation
scripts/train/                   matched training arms
scripts/evaluate/                rollout evaluation and metric export
scripts/probes/                  frozen-checkpoint mechanism experiments
scripts/plotting/                final paper plotting only
scripts/results/                 validation selection and result assembly
tests/                           numerical and integration tests
docs/PAPER_CODE_INDEX.md         paper-item to implementation map
docs/DATA_FORMATS.md             dataset trees, file keys, and tensor contracts
```

Cluster launchers, monitoring scripts, checkpoints, generated figures, raw
datasets, and exploratory ledgers are intentionally local-only and excluded by
`.gitignore`. They are not part of the public API.
The lower-level script counts and functional breakdown are documented in
[`scripts/README.md`](scripts/README.md).

## Reproduce the paper

Paper reproduction has three explicit stages: training writes a validation-
selected checkpoint, testing writes a compact test JSON, and rendering reads
only completed JSON/NPZ ledgers. The commands below perform the final rendering
stage and never retrain a model.

After placing the released result bundle at the repository root (so its
`probe_outputs/`, `experiments/`, and `artifacts/` paths are preserved), run:

```bash
bash scripts/reproduce/01_figures_1_2.sh
bash scripts/reproduce/02_figure_3.sh
bash scripts/reproduce/03_figure_4.sh
bash scripts/reproduce/04_figure_5.sh
bash scripts/reproduce/05_figure_6.sh
bash scripts/reproduce/06_figures_7_8.sh
bash scripts/reproduce/07_timing_table.sh
```

Or render everything with:

```bash
bash scripts/reproduce/render_all.sh
```

The exact source JSON/NPZ path is recorded beside every generated paper figure.
See [`scripts/reproduce/README.md`](scripts/reproduce/README.md) and
[`docs/PAPER_CODE_INDEX.md`](docs/PAPER_CODE_INDEX.md) for the full mapping.

### Full eight-dataset benchmark (Figure 6)

Figure 6 includes datasets with different licenses and storage layouts, so its
wrapper never guesses private mount paths. The matched per-dataset runners are:

| Data | Training entry point | Test entry point |
|---|---|---|
| MG, NARMA-5, ETTm1, ETTm2 | `scripts/train/run_mem_one.sh` | `scripts/evaluate/run_assigned_dense_eval_1p5k_one.sh` |
| iEEG (16 participants + stimulus) | `scripts/reproduce/train_test_ieeg.sh train` | `scripts/reproduce/train_test_ieeg.sh test` |
| Movie fMRI | `scripts/train/train_hcp_resgrad_mamba_v2.sh` | `scripts/evaluate/run_assigned_dense_eval_1p5k_one.sh` |
| Shear flow | `scripts/train/run_thewell_arm.sh` | `scripts/evaluate/run_assigned_dense_eval_1p5k_one.sh` |
| WeatherBench-2 | `scripts/train/run_wb2_arm.sh` | `scripts/evaluate/run_assigned_dense_eval_1p5k_one.sh` |

For example, train and test one matched MG seed as separate stages:

```bash
ARM=full_bptt SEED=0 GPU=0 bash scripts/reproduce/train_test_mg.sh train
ARM=internal_dw SEED=0 GPU=0 bash scripts/reproduce/train_test_mg.sh train

ARM=full_bptt SEED=0 GPU=0 bash scripts/reproduce/train_test_mg.sh test
ARM=internal_dw SEED=0 GPU=0 bash scripts/reproduce/train_test_mg.sh test
```

Training is resumable and writes `best.pth`; testing is idempotent and writes
one compact JSON per checkpoint. Once all required seed-level test JSON files
exist, `scripts/reproduce/05_figure_6.sh` exports the unnormalized ledger and
renders the figure without touching any checkpoint.

The benchmark controls use the same validation-then-test separation. Static
gain uses `scripts/train/run_positive_static_gain_one.sh`,
`scripts/results/select_positive_static_gain.py`, and
`scripts/evaluate/evaluate_positive_static_gain_one.sh`. TBPTT uses
`scripts/train/run_tbptt_positive_sweep_one.sh`,
`scripts/results/build_tbptt_results.py`, and
`scripts/evaluate/evaluate_tbptt_positive_selected_one.sh`. Standard dense
rollout ledgers are produced by
`scripts/evaluate/run_assigned_dense_eval_1p5k_one.sh`,
`scripts/evaluate/run_assigned_dense_eval_one.sh`,
`scripts/evaluate/run_internal_dw_AB_dense_eval_1p5k_one.sh`, and
`scripts/evaluate/run_tbptt_dense_eval_1p5k_one.sh`. These evaluators never
select a checkpoint or hyperparameter from the test split.

## Data and checkpoints

Synthetic MG, NARMA, and known-SNR data can be generated by the retained data
scripts. ETTm1 and ETTm2 must first be downloaded from the upstream
[`zhouhaoyi/ETDataset`](https://github.com/zhouhaoyi/ETDataset) repository and
then converted with `scripts/data/prepare_temporal_candidate_screen_data.py`.
Movie iEEG, HCP movie fMRI, The Well, and WeatherBench-2 must be obtained under their
upstream licenses; their paths are passed through `DATA_PATH`, `WELL_REPO`, or
the dataset-specific environment variables documented in each runner.
The Well registry helper is `scripts/data/download_thewell_registry.py`; it
does not bypass or replace the upstream data license.

The loaders expect the following top-level organization by default:

```text
data/
  synthetic/                       # generated MG/NARMA caches
  known_snr/                       # generated Known-SNR archive
  hcp_movie_features/<subject>/*MOVIE1*.npy
  ieeg/preprocessed_length_matched/*.fif
  ieeg/clip_features/{clip_projected.npy,clip_frames.csv}
  weatherbench2_1p5_pilot/{metadata.json,*.npy}
probe_inputs/temporal_candidate_regime_v1/{ettm1,ettm2}.npz
external/the_well/gradient_pilots/datasets/shear_flow/data/
  {train,valid,test}/*.h5
```

The exact file keys, array shapes, accepted alternate paths, preparation
commands, and the common in-memory sample contract are specified in
[`docs/DATA_FORMATS.md`](docs/DATA_FORMATS.md). In particular, HCP subject
files are dictionary-valued `.npy` files containing `fmri` and `z`, prepared
ETT archives contain fixed `train/validation/test` state and drive arrays, and
WeatherBench-2 filenames are resolved through `metadata.json`.

### Path configuration (no script editing required)

All public launchers infer the repository root from their own location and use
repository-relative defaults. No author-specific mount path is required. To
keep a dataset, checkpoint bundle, or output directory elsewhere, set the
corresponding environment variable on the command line; values may be absolute
or relative paths.

| Variable | Purpose | Default |
|---|---|---|
| `PROJECT_ROOT` | Repository checkout | inferred automatically |
| `DATA_PATH` | Dataset path for the HCP, The Well, or WB2 training runner | dataset-specific path under `data/` or `external/` |
| `DATA_DIR` | Generated synthetic cache directory | `data/synthetic` |
| `FIF_ROOT`, `CLIP_ROOT` | iEEG preprocessed FIF and existing CLIP inputs | required for iEEG preparation |
| `IEEG_PREPARED_ROOT` | Prepared subject archives | `probe_inputs/ieeg_cohort_v1` |
| `IEEG_RUN_ROOT` | iEEG checkpoints and per-run test records | `experiments/ieeg_cohort_v1` |
| `IEEG_PROBE_ROOT` | iEEG regime/utility/noise probe outputs | `probe_outputs/ieeg_cohort_v1` |
| `IEEG_TIMING_ROOT` | Paired iEEG timing outputs | `experiments/ieeg_cohort_timing_v1` |
| `HCP_DATA_PATH` | HCP path used by frozen-checkpoint probes | `data/hcp_movie_features` |
| `WELL_REPO` | Checkout containing The Well data utilities | `external/the_well` |
| `PREPARED_NPZ` | One prepared ETT or other temporal archive | derived under `probe_inputs/` |
| `SAVE_BASE` | Training/checkpoint output root | dataset-specific path under `experiments/` |
| `ROOT`, `OUT`, `OUT_ROOT` | Figure-specific input/output override | documented in each reproduction wrapper |

For example, an HCP run with data and outputs outside the checkout is:

```bash
DATA_PATH=/datasets/hcp_movie_features \
SAVE_BASE=/scratch/internal_dw/hcp_runs \
GPUS=0,1,2,3 SEEDS=0 \
  bash scripts/train/train_hcp_resgrad_mamba_v2.sh
```

The HCP component of the frozen-checkpoint diagnostic can be redirected in the
same way:

```bash
HCP_DATA_PATH=/datasets/hcp_movie_features GPU=0 DATASETS=fmri \
  bash scripts/probes/run_added_noise_response_all.sh
```

If a required path is absent, the runner exits with the missing variable and
path instead of silently falling back to a private server mount.

Large checkpoints and result ledgers are not stored in Git. Public releases
should attach them as a versioned archive with checksums and preserve the
relative paths expected by the reproduction scripts.

## Tests

```bash
python -m pytest -q
```

The tests cover forward invariance, independent identity/nonlinear VJP gains,
the constrained 2x2 Wiener solve, recurrent evaluation, dataset preparation,
and the paper probes.

Maintainers preparing the first public repository should follow
[`docs/GITHUB_RELEASE.md`](docs/GITHUB_RELEASE.md) for the staged-tree audit,
initial push, and separate release of large result artifacts.

## Citation and license

This project is released under the MIT License; see [`LICENSE`](LICENSE).
Citation metadata will be added when the anonymous manuscript is
de-anonymized.
