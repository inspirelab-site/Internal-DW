#!/usr/bin/env bash
# Train one validation-selected static Internal-DW control on 1, 2, or 4 GPUs.
# The frozen control uses alpha=m=c at every internal residual route.
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

DATA=${DATA:?set DATA=mg|ettm1|ettm2|shear}
GAIN=${GAIN:?set GAIN=0.3|0.6|0.9}
SEED=${SEED:?set SEED=0|1|2}
GPUS=${GPUS:-${GPU:-0}}
TRAIN_ROOT=${TRAIN_ROOT:-experiments/internal_dw_static_positive_v1}
PREPARED_INPUT_ROOT=${PREPARED_INPUT_ROOT:-probe_inputs/temporal_candidate_regime_v1}

case "${GAIN}" in 0.3|0.6|0.9) ;; *) echo "[refuse] GAIN must be 0.3, 0.6, or 0.9" >&2; exit 2 ;; esac
case "${SEED}" in 0|1|2) ;; *) echo "[refuse] SEED must be 0, 1, or 2" >&2; exit 2 ;; esac
gpu_text=${GPUS//,/ }
read -r -a STATIC_GPU_IDS <<< "${gpu_text}"
STATIC_WORLD_SIZE=${#STATIC_GPU_IDS[@]}
case "${STATIC_WORLD_SIZE}" in
  1|2|4) ;;
  *) echo "[refuse] supported world sizes are 1, 2, and 4; got GPUS=${GPUS}" >&2; exit 2 ;;
esac

configure_static_batch() {
  local target=$1 preferred_local=$2
  local local_batch=${STATIC_LOCAL_BATCH_OVERRIDE:-${preferred_local}}
  local denom=$((STATIC_WORLD_SIZE * local_batch))
  if (( denom <= 0 || target % denom != 0 )); then
    local_batch=$((target / STATIC_WORLD_SIZE))
    denom=$((STATIC_WORLD_SIZE * local_batch))
  fi
  (( denom > 0 && target % denom == 0 )) || {
    echo "[refuse] cannot preserve effective batch ${target} on ${STATIC_WORLD_SIZE} GPUs" >&2
    exit 2
  }
  STATIC_LOCAL_BATCH=${STATIC_LOCAL_BATCH_OVERRIDE:-${local_batch}}
  STATIC_GRAD_ACCUM=${STATIC_GRAD_ACCUM_OVERRIDE:-$((target / (STATIC_WORLD_SIZE * STATIC_LOCAL_BATCH)))}
  (( STATIC_WORLD_SIZE * STATIC_LOCAL_BATCH * STATIC_GRAD_ACCUM == target )) || {
    echo "[refuse] static effective-batch mismatch" >&2
    exit 2
  }
  echo "[batch] world=${STATIC_WORLD_SIZE} local=${STATIC_LOCAL_BATCH} accum=${STATIC_GRAD_ACCUM} effective=${target}"
}

# A static control must never inherit an estimator/prior from the shell.
unset DUAL_WIENER_INNOVATION_FILE DUAL_WIENER_INNOVATION_KEY
unset DUAL_WIENER_DOMAIN_NOISE_MODEL DUAL_WIENER_DOMAIN_TAG
unset DUAL_WIENER_SPECTRUM_OAS GLOBAL_WIENER_STATIC_GAIN

echo "[static] data=${DATA} c=${GAIN} seed=${SEED} GPUS=${GPUS}"
case "${DATA}" in
  mg)
    configure_static_batch 32 4
    env DATASET=mackey_glass COND=tau30 K=32 METHOD="dwc${GAIN}" \
      SEED="${SEED}" GPUS="${GPUS}" HIDDEN=128 \
      BATCH="${STATIC_LOCAL_BATCH}" GRAD_ACCUM="${STATIC_GRAD_ACCUM}" EPOCHS=100 ES=20 LR=1e-4 \
      WEIGHT_DECAY=1e-4 GRAD_CLIP=1.0 AR_OPTIMIZER=adam \
      AR_SCHEDULER=step NUM_WORKERS=4 MAMBA_TRAIN_STARTS=16 \
      RECURRENT_EVAL_HORIZON_BATCH=9 SAVE_BASE="${TRAIN_ROOT}" \
      SKIP_EXISTING=1 RESUME=auto bash scripts/train/run_mem_one.sh
    ;;
  ettm1|ettm2)
    prepared_npz="${PREPARED_INPUT_ROOT}/${DATA}.npz"
    [[ -s "${prepared_npz}" ]] || { echo "[missing] ${prepared_npz}" >&2; exit 3; }
    configure_static_batch 32 8
    env DATASET=prepared_temporal_driven COND="${DATA}" K=64 \
      METHOD="dwc${GAIN}" SEED="${SEED}" GPUS="${GPUS}" HIDDEN=128 \
      BATCH="${STATIC_LOCAL_BATCH}" GRAD_ACCUM="${STATIC_GRAD_ACCUM}" EPOCHS=100 ES=20 LR=1e-4 \
      WEIGHT_DECAY=1e-4 GRAD_CLIP=1.0 AR_OPTIMIZER=adam \
      AR_SCHEDULER=step NUM_WORKERS=0 MAMBA_TRAIN_STARTS=16 \
      PREPARED_NPZ="${prepared_npz}" RECURRENT_EVAL_HORIZON_BATCH=9 \
      SAVE_BASE="${TRAIN_ROOT}" SKIP_EXISTING=1 RESUME=auto \
      bash scripts/train/run_mem_one.sh
    ;;
  shear)
    configure_static_batch 4 1
    env DATASET=shear_flow METHOD="dwc${GAIN}" SEED="${SEED}" K=32 \
      GPUS="${GPUS}" BATCH="${STATIC_LOCAL_BATCH}" GRAD_ACCUM="${STATIC_GRAD_ACCUM}" EPOCHS=100 \
      EARLY_STOP_PATIENCE=20 LR=3e-4 GRAD_CLIP=1.0 \
      AR_SCHEDULER=cosine AR_MIN_LR=1e-6 NUM_WORKERS=0 \
      SAVE_BASE="${TRAIN_ROOT}" SKIP_EXISTING=1 RESUME=auto \
      bash scripts/train/run_thewell_arm.sh
    ;;
  *) echo "[refuse] unknown DATA=${DATA}" >&2; exit 2 ;;
esac
