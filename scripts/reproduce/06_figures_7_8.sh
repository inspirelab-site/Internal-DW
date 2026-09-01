#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh" "${1:-render}"

OUT_ROOT=${OUT_ROOT:-probe_outputs/internal_dw_fixed_horizon_positive_v1}
GPU=${GPU:-0}

if [[ "${MODE}" == run ]]; then
  for spec in "mg:64:8,12,16,24,32,48,64" \
              "ettm1:128:16,24,32,48,64,96,128" \
              "ettm2:128:16,24,32,48,64,96,128" \
              "shear:48:8,12,16,24,32,40,48"; do
    IFS=: read -r data fixed_h grid <<< "${spec}"
    IFS=, read -r -a ks <<< "${grid}"
    for k in "${ks[@]}"; do
      for seed in 0 1 2; do
        for arm in exact dw; do
          DATA=${data} ARM=${arm} K=${k} SEED=${seed} FIXED_H=${fixed_h} \
            GPU=${GPU} OUT_ROOT=${OUT_ROOT} \
            bash scripts/train/run_internal_dw_fixed_horizon_single_cell.sh
        done
      done
    done
  done
fi

"${PYTHON}" scripts/results/build_fixed_horizon_results.py --root "${OUT_ROOT}"
"${PYTHON}" scripts/plotting/plot_complete4_fixed_horizon_quadratic.py \
  --input-root "${OUT_ROOT}/test" \
  --output "${OUT_ROOT}/summary_complete4/fixed_horizon_main" \
  --appendix-output "${OUT_ROOT}/summary_complete4/fixed_horizon_seedwise_appendix"
echo "[done] Figure 7 and Appendix Figure 8 under ${OUT_ROOT}/summary_complete4/"
