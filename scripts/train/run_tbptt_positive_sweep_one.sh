#!/usr/bin/env bash
# One TBPTT segment-length candidate under the primary positive-regime
# training protocol.  DATA in {mg,ettm1,ettm2,shear}; S is the backward
# segment length and K remains the unchanged forward/loss horizon.
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

DATA=${DATA:?set DATA=mg|ettm1|ettm2|shear}
S=${S:?set the TBPTT segment length}
SEED=${SEED:?set SEED}
GPU=${GPU:-}
GPUS=${GPUS:-${GPU}}
[[ -n "${GPUS}" ]] || { echo "[refuse] set GPUS=0,1,2,3 (DDP) or GPU=0" >&2; exit 2; }
gpu_text=${GPUS//,/ }
read -r -a GPU_IDS <<< "${gpu_text}"
WORLD_SIZE=${#GPU_IDS[@]}
case "${WORLD_SIZE}" in
  1|2|4) ;;
  *) echo "[refuse] supported world sizes are 1, 2, and 4; got GPUS=${GPUS}" >&2; exit 2 ;;
esac
SWEEP_ROOT=${SWEEP_ROOT:-experiments/tbptt_positive_sweep_v1}

# Keep the optimizer-level effective batch identical when switching an
# interrupted single-GPU run to DDP.  This is essential for a fair resume:
#   MG/ETT: global effective batch 32; Shear: global effective batch 4.
configure_batch() {
  local target=$1 preferred_local=$2
  if (( target % (WORLD_SIZE * preferred_local) == 0 )); then
    LOCAL_BATCH=${LOCAL_BATCH_OVERRIDE:-${preferred_local}}
  else
    LOCAL_BATCH=${LOCAL_BATCH_OVERRIDE:-$(( target / WORLD_SIZE ))}
  fi
  local denom=$(( WORLD_SIZE * LOCAL_BATCH ))
  (( denom > 0 && target % denom == 0 )) || {
    echo "[refuse] cannot preserve effective batch ${target}: world=${WORLD_SIZE}, local_batch=${LOCAL_BATCH}" >&2
    exit 2
  }
  ACCUM=${GRAD_ACCUM_OVERRIDE:-$(( target / denom ))}
  (( WORLD_SIZE * LOCAL_BATCH * ACCUM == target )) || {
    echo "[refuse] effective batch mismatch: world*local*accum=$((WORLD_SIZE * LOCAL_BATCH * ACCUM)), expected ${target}" >&2
    exit 2
  }
  echo "[batch] world=${WORLD_SIZE} local=${LOCAL_BATCH} accum=${ACCUM} effective=${target}"
}

case "${DATA}" in
  mg)
    K=32
    configure_batch 32 4
    [[ "${S}" == 8 || "${S}" == 16 ]] || {
      echo "[refuse] MG candidates are S=8,16" >&2; exit 2;
    }
    # Reuse the completed S=8 tree exactly; do not copy or retrain it.
    if [[ "${S}" == 8 ]]; then
      out="experiments/tbptt_mg_a8/mackey_glass/tau30_K32/ckpt/seed${SEED}"
      if [[ -s "${out}/best.pth" ]]; then
        echo "[reuse] MG S=8 seed${SEED}: ${out}/best.pth"
        exit 0
      fi
      echo "[missing] expected completed MG S=8 checkpoint: ${out}/best.pth" >&2
      exit 3
    fi
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
    configure_batch 32 $((32 / WORLD_SIZE))
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
      WEIGHT_DECAY=1e-4 GRAD_CLIP=0.1 AR_OPTIMIZER=adam \
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
    # Reuse the completed S=8 tree exactly.
    if [[ "${S}" == 8 ]]; then
      out="experiments/thewell_shear_final_lr3e4/shear_flow/unet_b32_D4_W2_K32_ds4/tbptt8/seed${SEED}"
      if [[ -s "${out}/best.pth" ]]; then
        echo "[reuse] shear S=8 seed${SEED}: ${out}/best.pth"
        exit 0
      fi
      echo "[missing] expected completed shear S=8 checkpoint: ${out}/best.pth" >&2
      exit 3
    fi
    DATASET=shear_flow METHOD="tbptt${S}" SEED="${SEED}" K="${K}" \
      GPUS="${GPUS}" BATCH="${LOCAL_BATCH}" GRAD_ACCUM="${ACCUM}" EPOCHS=100 \
      EARLY_STOP_PATIENCE=20 LR=3e-4 GRAD_CLIP=1.0 \
      AR_SCHEDULER=cosine NUM_WORKERS=0 \
      SAVE_BASE="${SWEEP_ROOT}/shear" SKIP_EXISTING=1 RESUME=auto \
      bash scripts/train/run_thewell_arm.sh
    ;;

  *) echo "[refuse] unknown DATA=${DATA}" >&2; exit 2 ;;
esac
