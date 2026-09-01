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

The reader-facing MG workflow keeps training and testing explicit while using
the same output contract as the benchmark evaluators:

```bash
ARM=full_bptt SEED=0 GPU=0 bash scripts/reproduce/train_test_mg.sh train
ARM=full_bptt SEED=0 GPU=0 bash scripts/reproduce/train_test_mg.sh test
```

The `train` phase writes `best.pth`; the `test` phase reads that checkpoint and
writes one compact test JSON. Omitting the phase runs both in order. Plotting
entry points consume JSON files only.

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
