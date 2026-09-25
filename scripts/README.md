# Script organization

Start with the commands in the root README and `reproduce/`.
Lower-level implementations are grouped by function:

| Directory | Purpose |
|---|---|
| `reproduce/` | Reader-facing training, selection, testing, and rendering commands |
| `data/` | Dataset preparation and train-only prior construction |
| `train/` | Matched training arms |
| `evaluate/` | Checkpoint rollout and forecasting metrics |
| `probes/` | Frozen-checkpoint and data-level mechanism diagnostics |
| `results/` | Validation selection and aggregation of completed result records |
| `plotting/` | Paper figures from completed records |
| `utils/` | Installation and device checks |

Hyperparameters are selected using seed-0 validation loss, then held fixed
across the reported training seeds. Test results never select a checkpoint
or hyperparameter. Plotting does not launch training.

See `docs/PAPER_CODE_INDEX.md` for the entry points for each paper item and
`docs/DATA_FORMATS.md` for input formats.
