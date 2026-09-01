# Internal-DW

Official research code for **Large Distant Gradients Need Not Be Reliable:
Reliability-Weighted Credit Assignment for Long-Horizon Autoregressive
Forecasting**.

Internal-DW is a backward-only operator for residual autoregressive models. It
keeps the forward computation unchanged and applies automatically estimated
Wiener gains to the identity and nonlinear route messages during backpropagation.

## Install

Python 3.10+ and PyTorch 2.1+ are required. Install the PyTorch build matching
your CUDA driver first, then install this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[paper,test]"
bash scripts/reproduce/00_smoke_test.sh
```

On a multi-GPU host, verify that the requested PyTorch build sees every CUDA
device before launching a paper run:

```bash
python scripts/utils/check_ddp_cuda.py
```

The reusable operator itself depends only on NumPy and PyTorch. The `paper`
extra installs plotting, neuroimaging, and tabular dependencies used by the
experiments.

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
src/internal_dw/models/          forecasting backbones and DW controller
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
```

Cluster launchers, monitoring scripts, checkpoints, generated figures, raw
datasets, and exploratory ledgers are intentionally local-only and excluded by
`.gitignore`. They are not part of the public API.
The lower-level script counts and functional breakdown are documented in
[`scripts/README.md`](scripts/README.md).

## Reproduce the paper

There are two levels of reproduction:

1. `render` rebuilds a paper figure/table from the released compact result
   ledgers. It is fast and is the default.
2. `run` recomputes the underlying experiment before rendering. It requires the
   corresponding datasets/checkpoints and can take multiple GPU-days.

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

Append `run` to an individual command to recompute it, for example:

```bash
GPUS=0,1,2,3 bash scripts/reproduce/03_figure_4.sh run
GPU=0 bash scripts/reproduce/06_figures_7_8.sh run
```

The exact source JSON/NPZ path is recorded beside every generated paper figure.
See [`scripts/reproduce/README.md`](scripts/reproduce/README.md) and
[`docs/PAPER_CODE_INDEX.md`](docs/PAPER_CODE_INDEX.md) for the full mapping.

### Full eight-dataset benchmark (Figure 6)

Figure 6 includes datasets with different licenses and storage layouts, so its
wrapper never guesses private mount paths. The matched per-dataset runners are:

| Data | Training entry point |
|---|---|
| MG, NARMA-5, iEEG, ETTm1, ETTm2 | `scripts/train/run_mem_one.sh` |
| Movie fMRI | `scripts/train/train_hcp_resgrad_mamba_v2.sh` |
| Shear flow | `scripts/train/run_thewell_arm.sh` |
| WeatherBench-2 | `scripts/train/run_wb2_arm.sh` |

For example, one matched MG seed is launched as:

```bash
DATASET=mackey_glass COND=tau30 K=32 METHOD=ckpt SEED=0 GPU=0 \
  bash scripts/train/run_mem_one.sh
DATASET=mackey_glass COND=tau30 K=32 METHOD=dwstructured SEED=0 GPU=0 \
  bash scripts/train/run_mem_one.sh
```

Each low-level runner is idempotent and supports resume. Once all required
seed-level evaluations exist, `scripts/reproduce/05_figure_6.sh` exports the
unnormalized ledger and renders the figure.

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
