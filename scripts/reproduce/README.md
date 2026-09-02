# Paper reproduction entry points

The numbered scripts are rendering entry points: they rebuild figures and
tables from completed JSON/NPZ ledgers. Model training and checkpoint testing
use the separate `scripts/train/` and `scripts/evaluate/` entry points described
in the root README.

Do not edit these scripts to insert machine-specific paths. Set `DATA_PATH`,
`HCP_DATA_PATH`, `WELL_REPO`, `PREPARED_NPZ`, or `SAVE_BASE` when invoking a
runner, as described in the root README's **Path configuration** section.

Run the public API smoke test first:

```bash
bash scripts/reproduce/00_smoke_test.sh
```

The known-SNR mechanism closure is self-contained and precedes the benchmark
workflow. It generates its synthetic archive and analytic reference, trains
the matched seed-0 forecasting arms, runs all frozen-checkpoint probes, and
renders Figure 4:

```bash
GPU=0 GPUS=0,1,2,3 bash scripts/reproduce/train_test_known_snr.sh
```

Omit `GPUS` to execute the same resumable stages on one GPU; the launcher
adjusts accumulation to retain effective batch 32.

The reader-facing MG workflow keeps training and testing explicit while using
the same output contract as the benchmark evaluators:

```bash
ARM=full_bptt SEED=0 GPU=0 bash scripts/reproduce/train_test_mg.sh train
ARM=full_bptt SEED=0 GPU=0 bash scripts/reproduce/train_test_mg.sh test
```

The launcher reads the arm-specific canonical settings from
`configs/reproduce/`. It also accepts `GPUS=0,1,2,3` and automatically divides
the recorded global microbatch over DDP ranks without changing the recorded
gradient-accumulation schedule.

The `train` phase writes `best.pth`; the `test` phase reads that checkpoint and
writes one compact test JSON. Omitting the phase runs both in order. Plotting
entry points consume JSON files only.

To train and test every Figure 6 dataset, reported arm, and matched seed
sequentially on one GPU, run:

```bash
GPU=0 bash scripts/reproduce/train_test_figure6_all.sh
```

The four favorable datasets run first. The queue skips completed test JSONs,
resumes interrupted checkpoints, and renders Figure 6 after all runs finish.
Failures are isolated by dataset, so a bad download is recorded while later
datasets continue; the final process status is nonzero until every requested
dataset succeeds. Licensed datasets must already be present in the layouts
documented in `docs/DATA_FORMATS.md`.

Then reproduce paper items in order:

```bash
bash scripts/reproduce/01_figures_1_2.sh
bash scripts/reproduce/02_figure_3.sh
bash scripts/reproduce/03_figure_4.sh
bash scripts/reproduce/04_figure_5.sh
bash scripts/reproduce/05_figure_6.sh
bash scripts/reproduce/06_figures_7_8.sh
bash scripts/reproduce/07_timing_table.sh
```

Use `bash scripts/reproduce/render_all.sh` after installing the released result
bundle. The exact train, test, and ledger-builder responsible for each plotted
input is listed in `docs/PAPER_CODE_INDEX.md`.
