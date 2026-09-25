#!/usr/bin/env bash
# One TBPTT segment-length candidate under the primary positive-regime
# training protocol.  DATA in {mg,ettm1,ettm2,shear}; S is the backward
# segment length and K remains the unchanged forward/loss horizon.
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
source configs/reproduce/figure6.sh

DATA=${DATA:?set DATA=mg|ettm1|ettm2|shear}
S=${S:?set the TBPTT segment length}
SEED=${SEED:?set SEED}
GPU=${GPU:-}
GPUS=${GPUS:-${GPU}}
[[ -n "${GPUS}" ]] || { echo "[refuse] set GPUS=0,1,2,3 (DDP) or GPU=0" >&2; exit 2; }
gpu_text=${GPUS//,/ }
read -r -a TBPTT_GPU_IDS <<< "${gpu_text}"
TBPTT_WORLD_SIZE=${#TBPTT_GPU_IDS[@]}
case "${TBPTT_WORLD_SIZE}" in
  1|2|4) ;;
  *) echo "[refuse] supported world sizes are 1, 2, and 4; got GPUS=${GPUS}" >&2; exit 2 ;;
esac
SWEEP_ROOT=${SWEEP_ROOT:-experiments/tbptt_positive_sweep_v1}

# Keep the optimizer-level effective batch identical when switching an
# interrupted single-GPU run to DDP.  This is essential for a fair resume:
#   MG/ETT: global effective batch 32; Shear: global effective batch 4.
configure_batch() {
  local target=$1 preferred_local=$2
  if (( target % (TBPTT_WORLD_SIZE * preferred_local) == 0 )); then
    LOCAL_BATCH=${LOCAL_BATCH_OVERRIDE:-${preferred_local}}
  else
    LOCAL_BATCH=${LOCAL_BATCH_OVERRIDE:-$(( target / TBPTT_WORLD_SIZE ))}
  fi
  local denom=$(( TBPTT_WORLD_SIZE * LOCAL_BATCH ))
  (( denom > 0 && target % denom == 0 )) || {
    echo "[refuse] cannot preserve effective batch ${target}: world=${TBPTT_WORLD_SIZE}, local_batch=${LOCAL_BATCH}" >&2
    exit 2
  }
  ACCUM=${GRAD_ACCUM_OVERRIDE:-$(( target / denom ))}
  (( TBPTT_WORLD_SIZE * LOCAL_BATCH * ACCUM == target )) || {
    echo "[refuse] effective batch mismatch: world*local*accum=$((TBPTT_WORLD_SIZE * LOCAL_BATCH * ACCUM)), expected ${target}" >&2
    exit 2
  }
  echo "[batch] world=${TBPTT_WORLD_SIZE} local=${LOCAL_BATCH} accum=${ACCUM} effective=${target}"
}

case "${DATA}" in
  mg)
    K=32
    configure_batch 32 4
    [[ "${S}" == 8 || "${S}" == 16 ]] || {
      echo "[refuse] MG candidates are S=8,16" >&2; exit 2;
    }
    DATASET=mackey_glass COND=tau30 K="${K}" METHOD="tbptt${S}" \
      SEED="${SEED}" GPUS="${GPUS}" HIDDEN=128 \
      BATCH="${LOCAL_BATCH}" GRAD_ACCUM="${ACCUM}" \
      EPOCHS=100 ES=20 LR=1e-4 WEIGHT_DECAY=1e-4 GRAD_CLIP=1.0 \
      AR_OPTIMIZER=adam AR_SCHEDULER=step NUM_WORKERS=4 \
      RECURRENT_EVAL_HORIZON_BATCH=9 SAVE_BASE="${SWEEP_ROOT}/mg" \
      SKIP_EXISTING=1 RESUME=auto bash scripts/train/run_mem_one.sh
    ;;

  ettm1|ettm2)
    K=64
    configure_batch 32 $((32 / TBPTT_WORLD_SIZE))
    [[ "${S}" == 16 || "${S}" == 32 ]] || {
      echo "[refuse] ETTm candidates are S=16,32" >&2; exit 2;
    }
    prepared_npz="${PREPARED_INPUT_ROOT:-probe_inputs/temporal_candidate_regime_v1}/${DATA}.npz"
    [[ -s "${prepared_npz}" ]] || {
      echo "[missing] ${prepared_npz}" >&2; exit 3;
    }
    DATASET=prepared_temporal_driven COND="${DATA}" K="${K}" \
      METHOD="tbptt${S}" SEED="${SEED}" GPUS="${GPUS}" HIDDEN=128 \
      BATCH="${LOCAL_BATCH}" GRAD_ACCUM="${ACCUM}" \
      EPOCHS=100 ES=20 LR=1e-4 \
      WEIGHT_DECAY=1e-4 GRAD_CLIP="${FIGURE6_COMMON_GRAD_CLIP}" AR_OPTIMIZER=adam \
      AR_SCHEDULER=step NUM_WORKERS=0 MAMBA_TRAIN_STARTS=16 \
      PREPARED_NPZ="${prepared_npz}" RECURRENT_EVAL_HORIZON_BATCH=9 \
      SAVE_BASE="${SWEEP_ROOT}/${DATA}" SKIP_EXISTING=1 RESUME=auto \
      bash scripts/train/run_mem_one.sh
    ;;

  shear)
    K=32
    configure_batch 4 1
    [[ "${S}" == 8 || "${S}" == 16 ]] || {
      echo "[refuse] shear candidates are S=8,16" >&2; exit 2;
    }
    DATASET=shear_flow METHOD="tbptt${S}" SEED="${SEED}" K="${K}" \
      GPUS="${GPUS}" BATCH="${LOCAL_BATCH}" GRAD_ACCUM="${ACCUM}" EPOCHS=100 \
      EARLY_STOP_PATIENCE=20 LR=3e-4 GRAD_CLIP=1.0 \
      AR_SCHEDULER=cosine NUM_WORKERS=0 \
      SAVE_BASE="${SWEEP_ROOT}/shear" SKIP_EXISTING=1 RESUME=auto \
      bash scripts/train/run_thewell_arm.sh
    ;;

  *) echo "[refuse] unknown DATA=${DATA}" >&2; exit 2 ;;
esac
