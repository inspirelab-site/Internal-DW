#!/usr/bin/env bash
# Idempotent one-GPU train/resume + fixed-H evaluation cell.
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd -P)
cd "${PROJECT_ROOT}"

DATA=${DATA:?set DATA}
ARM=${ARM:?set ARM}
K=${K:?set K}
SEED=${SEED:?set SEED}
FIXED_H=${FIXED_H:?set FIXED_H}
GPU=${GPU:?set one physical GPU id}
TRAIN_ROOT=${TRAIN_ROOT:-experiments/internal_dw_baselines_k_v1}
SYNC_ROOT=${SYNC_ROOT:-artifacts/internal_dw_baselines_k_v1_sync}
OUT_ROOT=${OUT_ROOT:-probe_outputs/internal_dw_fixed_horizon_positive_v1}

val_out="${OUT_ROOT}/val/${DATA}/${ARM}_K${K}_seed${SEED}_H${FIXED_H}.json"
test_out="${OUT_ROOT}/test/${DATA}/${ARM}_K${K}_seed${SEED}_H${FIXED_H}.json"
if [[ -s "${val_out}" && -s "${test_out}" ]]; then
  echo "[skip complete cell] ${DATA} ${ARM} K=${K} seed=${SEED} H=${FIXED_H}"
  exit 0
fi

assigned=0
case "${DATA}:${K}" in
  mg:32|ettm1:64|ettm2:64|shear:32) assigned=1 ;;
esac
if (( assigned == 0 )); then
  PHASE=B DATA="${DATA}" ARM="${ARM}" K="${K}" SEED="${SEED}" \
    GPUS="${GPU}" ROOT="${TRAIN_ROOT}" SYNC_ROOT="${SYNC_ROOT}" \
    bash scripts/train/run_internal_dw_baseline_or_k_one.sh
else
  echo "[reuse assigned checkpoint] ${DATA} ${ARM} K=${K} seed=${SEED}"
fi

for split in val test; do
  (
    DATA="${DATA}"
    ARM="${ARM}"
    K="${K}"
    SEED="${SEED}"
    SPLIT="${split}"
    FIXED_H="${FIXED_H}"
    GPU="${GPU}"
    OUT_ROOT="${OUT_ROOT}"
    MAX_ORIGINS="${MAX_ORIGINS:-64}"
    BOOTSTRAP_DRAWS="${BOOTSTRAP_DRAWS:-10000}"
    source scripts/evaluate/run_internal_dw_fixed_horizon_one.sh
  )
done
echo "[single-cell-complete] ${DATA} ${ARM} K=${K} seed=${SEED}"
