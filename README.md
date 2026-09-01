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

## Train and test a model

The MG example is the shortest complete training-to-test workflow. Run the two
matched arms separately:

```bash
ARM=full_bptt SEED=0 GPU=0 \
  bash scripts/reproduce/train_test_mg.sh

ARM=internal_dw SEED=0 GPU=0 \
  bash scripts/reproduce/train_test_mg.sh
```

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
| MG, NARMA-5, iEEG, ETTm1, ETTm2 | `scripts/train/run_mem_one.sh` | `scripts/evaluate/run_assigned_dense_eval_1p5k_one.sh` |
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
scripts. ETT archives are prepared by
`scripts/data/prepare_temporal_candidate_screen_data.py`. HCP movie fMRI, The Well,
and WeatherBench-2 must be obtained under their upstream licenses; their paths
are passed through `DATA_PATH`, `WELL_REPO`, or the dataset-specific environment
variables documented in each runner.
The Well registry helper is `scripts/data/download_thewell_registry.py`; it
does not bypass or replace the upstream data license.

The loaders expect the following top-level organization by default:

```text
data/
  synthetic/                       # generated MG/NARMA/known-SNR caches
  hcp_movie_features/<subject>/*MOVIE1*.npy
  ieeg/preprocessed_length_matched/*.fif
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
| `DATA_DIR` | Generated synthetic or iEEG cache directory | `data/synthetic` |
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
