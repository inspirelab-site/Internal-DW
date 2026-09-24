# Paper code index

This index maps the benchmark PDF to the retained code. Paths are relative to the repository root.

## Shared implementation

- Public plug-in API: `src/internal_dw/router.py`
- Minimal third-party example: `examples/quickstart_internal_dw.py`
- Training entry point: `src/main.py`
- Internal-DW controller: `src/internal_dw/models/dual_wiener.py`
- Sequence backbone: `src/internal_dw/models/official_state_mamba.py`
- Field backbone: `src/internal_dw/models/unet_field.py`
- Rollout losses and backward routing: `src/internal_dw/training/ar_losses.py`
- Training/calibration loop: `src/internal_dw/training/trainer.py`
- Dense fixed-origin evaluation: `scripts/evaluate/evaluate_dense_multistart_rel_l2.py`

The active wrappers for the eight benchmark datasets are `scripts/train/run_mem_one.sh`,
`scripts/train/train_hcp_resgrad_mamba_v2.sh`, `scripts/train/run_thewell_arm.sh`, and
`scripts/train/run_wb2_arm.sh`.

Stable reader-facing reproduction commands are grouped in
`scripts/reproduce/`.  The lower-level files listed below remain the auditable
implementation behind those commands; machine-specific launch, queue, status,
and recovery scripts are local-only and excluded by `.gitignore`.

## Paper figures and tables

| Paper item | Primary retained entry points |
|---|---|
| Fig. 1, gradient controls | `scripts/plotting/plot_intro_gradient_controls.py` |
| Fig. 2, Internal-DW method | `scripts/plotting/plot_internal_dw_method.py` |
| Fig. 3, predictive regimes | `scripts/probes/run_cross_dataset_drive_history_map_4gpu.sh`, `scripts/plotting/plot_cross_dataset_drive_history_map.py`, `scripts/probes/run_driven_mg_strength_sweep.sh` |
| Fig. 4, known-SNR closure | End-to-end: `scripts/reproduce/train_test_known_snr.sh`; settings: `configs/reproduce/known_snr.json`; analytic reference: `scripts/data/prepare_known_snr_oracle.py`; profiles: `scripts/probes/run_known_snr_exact_panels12.sh`; frozen training-stream risk: `scripts/probes/probe_known_snr_online_risk.py`; test: `scripts/evaluate/evaluate_dense_multistart_rel_l2.py`; profile aggregation/rendering: `scripts/results/build_known_snr_results.py`, `scripts/plotting/plot_known_snr_closure.py` |
| Fig. 5, application diagnostics | Panel (a): `scripts/probes/run_real_heldout_gradient_utility_single_gpu.sh`, `scripts/probes/run_ettm_internal_dw_diagnostic_one.sh`, `scripts/probes/run_wb2_heldout_gradient_utility_n8.sh`. Panel (b): `scripts/probes/run_added_noise_response_all.sh` and `scripts/results/build_added_noise_results.py`. Rendering: `scripts/plotting/plot_application_diagnostics_combined.py` |
| Fig. 6, forecasting controls | One-GPU train/test queue: `scripts/reproduce/train_test_figure6_all.sh`; rendering: `scripts/reproduce/05_figure_6.sh`, `scripts/plotting/plot_forecasting_controls_hierarchical_preview.py`, `scripts/results/export_figure6_raw_metrics.py` |
| Fig. 7 and App. Fig. 8, fixed-H K sweep | `scripts/reproduce/06_figures_7_8.sh`, `scripts/train/run_internal_dw_fixed_horizon_single_cell.sh`, `scripts/results/build_fixed_horizon_results.py`, `scripts/plotting/plot_complete4_fixed_horizon_quadratic.py` |
| App. timing table | `scripts/reproduce/07_timing_table.sh`, `scripts/train/run_internal_dw_compute_overhead_one.sh`, `scripts/results/build_training_time_table.py` |

## Dataset and prior preparation

- Driven MG: `scripts/data/generate_driven_mackey_glass.py`
- ETTm1/ETTm2 prepared archives: `scripts/data/prepare_temporal_candidate_screen_data.py`
- iEEG prior diagnostics: `scripts/probes/probe_ieeg_blocked_longmemory_templates.py`
- Movie-fMRI subject-crossfit prior: `scripts/data/prepare_hcp_shared_response_innovation.py`, `scripts/probes/probe_fmri_subject_crossfit_templates.py`
- Shear spectral prior: `scripts/data/prepare_thewell_domain_innovation.py`
- WeatherBench-2 preparation: `scripts/data/prepare_weatherbench2_pilot.py`, `scripts/data/validate_weatherbench2_pilot.py`

## Scope rule

Code for cross-backbone studies, RL, rejected candidate datasets, global-horizon
Wiener routing, and superseded mechanism probes is intentionally absent because
none of those experiments appears in the benchmark PDF.

The GitHub allowlist in `.gitignore` further excludes local cluster operations
and one-off development probes without deleting them from the research
checkout.  This keeps the released code surface smaller than the full local
experiment history while preserving every dependency of the documented
reproduction entry points.
