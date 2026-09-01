#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh" "${1:-render}"

MAP_ROOT=${MAP_ROOT:-probe_outputs/cross_dataset_drive_history_map_v1}
MG_ROOT=${MG_ROOT:-probe_outputs/driven_mg_strength_sweep_v1}

if [[ "${MODE}" == run ]]; then
  GPUS=${GPUS:-0,1,2,3} bash scripts/probes/run_cross_dataset_drive_history_map_4gpu.sh
  GPUS=${GPUS:-0,1,2,3} bash scripts/probes/run_driven_mg_strength_sweep.sh
else
  for name in mg narma ieeg shear wb2 fmri; do require_file "${MAP_ROOT}/${name}.json"; done
  "${PYTHON}" scripts/plotting/plot_cross_dataset_drive_history_map.py \
    --dataset "MG=${MAP_ROOT}/mg.json" \
    --dataset "NARMA=${MAP_ROOT}/narma.json" \
    --dataset "iEEG=${MAP_ROOT}/ieeg.json" \
    --dataset "Shear=${MAP_ROOT}/shear.json" \
    --dataset "WB2=${MAP_ROOT}/wb2.json" \
    --dataset "ETTm1=probe_outputs/temporal_candidate_regime_v1/ettm1.json" \
    --dataset "ETTm2=probe_outputs/temporal_candidate_regime_v1/ettm2.json" \
    --fmri "${MAP_ROOT}/fmri.json" --long-horizon-min 8 \
    --output "${MAP_ROOT}/long_horizon_map"

  require_file "${MG_ROOT}/simulator_ground_truth.json"
  "${PYTHON}" scripts/plotting/plot_driven_mg_strength_sweep.py \
    --data-root artifacts/driven_mg_strength_sweep_v1/data \
    --result-root "${MG_ROOT}/points" \
    --scales 0.00,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08 \
    --ground-truth "${MG_ROOT}/simulator_ground_truth.json" \
    --output "${MG_ROOT}/drive_history_vs_strength"
fi

echo "[done] Figure 3 ledgers under ${MAP_ROOT} and ${MG_ROOT}"

