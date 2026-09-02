#!/usr/bin/env bash
# Complete the known-SNR chain with matched seed-0 forecasting performance.
set -euo pipefail
cd "$(dirname "$0")/../.."

SEED=${SEED:-0}
GPUS=${GPUS:-${GPU:-0}}
FIRST_GPU=${GPUS%%,*}
BATCH=${BATCH:-4}
IFS=',' read -r -a CLOSURE_GPU_IDS <<< "${GPUS}"
KNOWN_SNR_FORECAST_WORLD_SIZE=${#CLOSURE_GPU_IDS[@]}
TARGET_EFFECTIVE_BATCH=32
if [[ -z "${GRAD_ACCUM:-}" ]]; then
  denominator=$(( BATCH * KNOWN_SNR_FORECAST_WORLD_SIZE ))
  if (( TARGET_EFFECTIVE_BATCH % denominator != 0 )); then
    echo "BATCH=${BATCH} x world=${KNOWN_SNR_FORECAST_WORLD_SIZE} does not divide target effective batch 32" >&2
    exit 2
  fi
  GRAD_ACCUM=$(( TARGET_EFFECTIVE_BATCH / denominator ))
fi
if (( BATCH * KNOWN_SNR_FORECAST_WORLD_SIZE * GRAD_ACCUM != TARGET_EFFECTIVE_BATCH )); then
  echo "effective batch mismatch: BATCH=${BATCH} world=${KNOWN_SNR_FORECAST_WORLD_SIZE} accum=${GRAD_ACCUM}, expected 32" >&2
  exit 2
fi
DATA=${DATA:-data/synthetic/known_snr_ar_D8_T1024_traj96_a5fd79afa1e5f_s0.npz}
ARTIFACT=${ARTIFACT:-artifacts/known_snr_diagonal_ar_closure/diagonal_ar1_seed${SEED}_K32.npz}
EXACT_RUN=${EXACT_RUN:-experiments/known_snr_oracle/known_snr_ar/mixed_K32/ckpt/seed${SEED}}
SAVE_BASE=${SAVE_BASE:-experiments/known_snr_diagonal_ar_closure}
DW_RUN=${DW_RUN:-${SAVE_BASE}/known_snr_ar/mixed_K32/dualwiener_diag_ar/seed${SEED}}
ROOT=${ROOT:-probe_outputs/known_snr_diagonal_ar_closure/seed${SEED}}
FORECAST_OUT=${FORECAST_OUT:-${ROOT}/forecasting.json}

[[ "${SEED}" == "0" ]] || {
  echo "This closure reuses the existing seed-0 Exact checkpoint; set SEED=0." >&2
  exit 2
}
[[ -f "${DATA}" ]] || { echo "known-SNR archive not found: ${DATA}" >&2; exit 2; }
[[ -f "${EXACT_RUN}/best.pth" ]] || { echo "Exact checkpoint missing: ${EXACT_RUN}/best.pth" >&2; exit 2; }
[[ -f "${EXACT_RUN}/eval_results.json" ]] || { echo "Exact evaluation missing: ${EXACT_RUN}/eval_results.json" >&2; exit 2; }

mkdir -p "$(dirname "${ARTIFACT}")" "${ROOT}" logs/known_snr_diagonal_ar_closure
python scripts/data/prepare_known_snr_diagonal_ar1.py \
  --data "${DATA}" \
  --output "${ARTIFACT}" \
  --max-horizon 32 \
  --split-seed "${SEED}" \
  --train-ratio 0.7 \
  --val-ratio 0.15

# The controller resolves the sampler when the model is constructed; use an
# absolute path because a resumed server job may have a different shell cwd.
ARTIFACT_ABS=$(python - "${ARTIFACT}" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)

echo "[closure] training/resuming diagonal-AR(1) Internal-DW on GPUs ${GPUS}"
echo "[closure] effective batch=${BATCH} x ${KNOWN_SNR_FORECAST_WORLD_SIZE} x ${GRAD_ACCUM} = ${TARGET_EFFECTIVE_BATCH}"
env \
  GPU="${FIRST_GPU}" \
  GPUS="${GPUS}" \
  DATASET=known_snr_ar \
  COND=mixed \
  K=32 \
  METHOD=dualwiener_diag_ar \
  SEED="${SEED}" \
  DUAL_WIENER_INNOVATION_FILE="${ARTIFACT_ABS}" \
  SAVE_BASE="${SAVE_BASE}" \
  DATA_DIR=data/synthetic \
  BATCH="${BATCH}" \
  GRAD_ACCUM="${GRAD_ACCUM}" \
  EPOCHS=100 \
  ES=20 \
  LR=1e-4 \
  WEIGHT_DECAY=1e-4 \
  GRAD_CLIP=1.0 \
  AR_OPTIMIZER=adam \
  AR_SCHEDULER=step \
  AR_MIN_LR=1e-6 \
  MAMBA_LOSS_TYPE=mse \
  MAMBA_TRAIN_STARTS=1 \
  HIDDEN=128 \
  SIMPLE_DEPTH=4 \
  SIMPLE_NHEAD=8 \
  SKIP_EXISTING=1 \
  RESUME=auto \
  bash scripts/train/run_mem_one.sh

python scripts/results/build_known_snr_results.py forecasting \
  --exact-run "${EXACT_RUN}" \
  --dw-run "${DW_RUN}" \
  --artifact "${ARTIFACT}" \
  --output "${FORECAST_OUT}" \
  --dw-world-size "${KNOWN_SNR_FORECAST_WORLD_SIZE}" \
  --horizons 1 2 4 8 16 32
