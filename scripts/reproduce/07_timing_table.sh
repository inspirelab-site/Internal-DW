#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh" "${1:-render}"

ROOT=${ROOT:-experiments/internal_dw_compute_overhead_v2_nockpt}
GPU=${GPU:-0}

if [[ "${MODE}" == run ]]; then
  for repeat in 0 1 2; do
    for data in mg ettm1 ettm2 shear narma ieeg fmri wb2; do
      for arm in exact dw; do
        DATA=${data} ARM=${arm} REPEAT=${repeat} GPU=${GPU} ROOT=${ROOT} \
          bash scripts/train/run_internal_dw_compute_overhead_one.sh
      done
    done
  done
fi

"${PYTHON}" scripts/results/build_training_time_table.py \
  --root "${ROOT}" --warmup-epochs 1 \
  --appendix-out docs/appendix_training_time.tex
echo "[done] ${ROOT}/summary.csv"
echo "[done] docs/appendix_training_time.tex"
