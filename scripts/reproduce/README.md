# Paper reproduction entry points

Every script accepts one of two modes:

- `render` (default): rebuild a figure/table from released JSON/NPZ ledgers.
- `run`: recompute the underlying experiment, then render it. This may require
  GPUs, licensed datasets, and the checkpoints described in the root README.

Run the public API smoke test first:

```bash
bash scripts/reproduce/00_smoke_test.sh
```

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

Figure 5 has two independent frozen-checkpoint diagnostics. Its `run` mode
first measures held-out gradient utility, then runs the controlled-noise
response for all eight datasets and assembles
`probe_outputs/application_diagnostics_v1/added_noise_gain_summary.json`.
The plotting code reads that ledger; no paper gain values are embedded in the
plotting module.

Use `bash scripts/reproduce/render_all.sh` after installing the released result
bundle.  `GPUS=0,1,2,3` selects devices for scripts whose `run` mode supports
parallel execution; the fixed-horizon and timing wrappers use `GPU=0` by
default and are safely resumable at their lower-level entry points.
