#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh" "${1:-render}"

if [[ "${MODE}" == run ]]; then
  GPU=${GPU:-0} DATASETS=mg,narma,ieeg,shear,wb2 \
    bash scripts/probes/run_real_heldout_gradient_utility_single_gpu.sh
  GPU=${GPU:-0} bash scripts/probes/run_wb2_heldout_gradient_utility_n8.sh
  for data in ettm1 ettm2; do
    DATA="${data}" TASK=delayed GPU=${GPU:-0} \
      bash scripts/probes/run_ettm_internal_dw_diagnostic_one.sh
  done
  GPU=${GPU:-0} bash scripts/probes/run_added_noise_response_all.sh
fi

ADDED_NOISE_SUMMARY="probe_outputs/application_diagnostics_cohort_v1/added_noise_gain_summary.json"
"${PYTHON}" scripts/results/build_added_noise_results.py \
  --output "${ADDED_NOISE_SUMMARY}"

# The plotting modules validate the per-dataset ledgers they consume.
"${PYTHON}" scripts/plotting/plot_application_diagnostics_combined.py \
  --added-noise-summary "${ADDED_NOISE_SUMMARY}" \
  --output figs/application_data_diagnostics_combined.pdf
echo "[done] figs/application_data_diagnostics_combined.pdf"
