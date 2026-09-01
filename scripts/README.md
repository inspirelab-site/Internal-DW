# Script organization

Reader-facing commands are organized by task rather than by server:

| Directory | Files | Share of 78 lower-level public scripts | Purpose |
|---|---:|---:|---|
| `probes/` | 27 | 34.62% | Six paper diagnostic families plus their runners and shared machinery |
| `plotting/` | 12 | 15.38% | Only the final paper figures and their shared style/data helpers |
| `train/` | 11 | 14.10% | Matched training arms and timing/fixed-horizon cells |
| `data/` | 11 | 14.10% | Dataset generation, preparation, and domain-prior construction |
| `evaluate/` | 9 | 11.54% | Standardized checkpoint evaluation; no plotting or selection |
| `results/` | 7 | 8.97% | Validation-only selection and JSON/table result assembly |
| `utils/` | 1 | 1.28% | Environment validation |
| **Total** | **78** | **100%** | |

`reproduce/` is a separate ten-script facade organized in paper order. Most
readers should start there and never call the lower-level files directly.

A path-level import audit initially produced 78 files.  A subsequent
content-level audit separated the small helper surfaces used by the paper from
standalone exploratory experiments that happened to live in the same modules.
The resulting self-contained closure has 78 files and no import from a
local-only module.  The superseded source files remain in the authors'
checkout but are excluded from the release allowlist. Four files were added
back after this audit because Figure 5(b)'s eight-dataset controlled-noise
panel previously retained final numbers without its full public result path.

The boundaries are intentional: `evaluate/` runs a trained checkpoint and
computes forecasting metrics; `probes/` computes mechanism diagnostics that
are not ordinary benchmark metrics; `results/` performs no training or model
inference and only selects from validation results or assembles existing JSON
records; `plotting/` renders finalized records. The two selection scripts stay
separate from aggregation so that the validation-only, no-test-leakage rule is
visible in code.

The seven files in `results/` are the complete post-processing surface:

| File | Role |
|---|---|
| `build_known_snr_results.py` | Four subcommands assemble all five known-SNR panels and their ledger |
| `build_added_noise_results.py` | Validate the eight heterogeneous controlled-noise probes and assemble Figure 5(b) |
| `build_fixed_horizon_results.py` | Aggregate matched seeds and select training horizons from validation |
| `build_training_time_table.py` | Aggregate paired timing repeats and emit the appendix table |
| `build_tbptt_results.py` | Select TBPTT segment length on validation, then aggregate its three test seeds |
| `select_positive_static_gain.py` | Select the static-gain candidate before launching the remaining seeds |
| `export_figure6_raw_metrics.py` | Export the unnormalized values already underlying Figure 6 |

Here “selection” never means choosing a favorable random seed. Seed 0 is the
predeclared validation sweep used to choose a hyperparameter; after that choice
is frozen, every requested seed is evaluated and retained.

## What counts as a probe

The 27 release files under `probes/` are not 27 independent scientific probes.
They contain thirteen Python experiment/data-probe entry points, nine shell
runners, and five shared route/VJP or dataset-support modules. Superseded diagnostics
may remain in the authors' checkout but are ignored by Git. The release entry points
implement six probe families appearing in the paper:

1. predictive-regime scores (including the driven-MG reference trajectory);
2. known-SNR gradient magnitude/SNR and prefix-risk decomposition;
3. known-SNR gain identification;
4. held-out route-gradient risk;
5. application-data held-out gradient utility; and
6. added-noise gain response for vector and spatial testbeds.

The iEEG and fMRI cross-fitted builders are retained because they construct the
paper's Prior sampler artifacts, not because they introduce additional
mechanism claims. Likewise, shared VJP and route-moment modules are
implementation dependencies, not separate experiments.

Machine-specific launch, queue, status, resume, and superseded exploratory
scripts remain in the authors' local checkout and are excluded by `.gitignore`.
