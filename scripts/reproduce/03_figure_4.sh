#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh" "${1:-render}"

SEED=${SEED:-0}
ROOT=${ROOT:-probe_outputs/known_snr_diagonal_ar_closure/seed${SEED}}
OUT=${OUT:-figs/known_snr_closure}

if [[ "${MODE}" == run ]]; then
  GPUS=${GPUS:-0,1,2,3} SEED=${SEED} bash scripts/probes/run_known_snr_exact_panels12.sh
  GPU=${GPU:-0} SEED=${SEED} bash scripts/probes/run_known_snr_diagonal_ar_gain.sh
  GPU=${GPU:-0} SEED=${SEED} bash scripts/probes/run_known_snr_diagonal_ar_route_risk.sh
  GPUS=${GPUS:-0,1,2,3} SEED=${SEED} bash scripts/train/run_known_snr_diagonal_ar_forecasting.sh
  GPUS=${GPUS:-0,1,2,3} SEED=${SEED} bash scripts/train/run_known_snr_diagonal_ar_controls_ddp4.sh
fi

for file in panels_1_2.json gain_identification.json route_risk.json forecasting_controls.json; do
  require_file "${ROOT}/${file}"
done
"${PYTHON}" scripts/results/build_known_snr_results.py closure --root "${ROOT}"
"${PYTHON}" scripts/plotting/plot_known_snr_closure.py --closure-root "${ROOT}" --out "${OUT}"
echo "[done] ${OUT}.pdf"
