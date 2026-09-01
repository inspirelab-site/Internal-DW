#!/usr/bin/env bash
# Train one validation-selected static Internal-DW control on all four GPUs.
# The frozen control uses alpha=m=c at every internal residual route.
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

DATA=${DATA:?set DATA=mg|ettm1|ettm2|shear}
GAIN=${GAIN:?set GAIN=0.3|0.6|0.9}
SEED=${SEED:?set SEED=0|1|2}
GPUS=${GPUS:-0,1,2,3}
TRAIN_ROOT=${TRAIN_ROOT:-experiments/internal_dw_static_positive_v1}
PREPARED_INPUT_ROOT=${PREPARED_INPUT_ROOT:-probe_inputs/temporal_candidate_regime_v1}

case "${GAIN}" in 0.3|0.6|0.9) ;; *) echo "[refuse] GAIN must be 0.3, 0.6, or 0.9" >&2; exit 2 ;; esac
case "${SEED}" in 0|1|2) ;; *) echo "[refuse] SEED must be 0, 1, or 2" >&2; exit 2 ;; esac
gpu_text=${GPUS//,/ }
read -r -a GPU_IDS <<< "${gpu_text}"
[[ ${#GPU_IDS[@]} -eq 4 ]] || {
  echo "[refuse] this formal runner requires four GPUs; got GPUS=${GPUS}" >&2
  exit 2
}

# A static control must never inherit an estimator/prior from the shell.
unset DUAL_WIENER_INNOVATION_FILE DUAL_WIENER_INNOVATION_KEY
unset DUAL_WIENER_DOMAIN_NOISE_MODEL DUAL_WIENER_DOMAIN_TAG
unset DUAL_WIENER_SPECTRUM_OAS GLOBAL_WIENER_STATIC_GAIN

echo "[static] data=${DATA} c=${GAIN} seed=${SEED} GPUS=${GPUS}"
case "${DATA}" in
  mg)
    # Original optimizer-level batch is 32.  DDP4: 4 * B4 * accum2 = 32.
    env DATASET=mackey_glass COND=tau30 K=32 METHOD="dwc${GAIN}" \
      SEED="${SEED}" GPUS="${GPUS}" HIDDEN=128 \
      BATCH=4 GRAD_ACCUM=2 EPOCHS=100 ES=20 LR=1e-4 \
      WEIGHT_DECAY=1e-4 GRAD_CLIP=1.0 AR_OPTIMIZER=adam \
      AR_SCHEDULER=step NUM_WORKERS=4 MAMBA_TRAIN_STARTS=16 \
      RECURRENT_EVAL_HORIZON_BATCH=9 SAVE_BASE="${TRAIN_ROOT}" \
      SKIP_EXISTING=1 RESUME=auto bash scripts/train/run_mem_one.sh
    ;;
  ettm1|ettm2)
    prepared_npz="${PREPARED_INPUT_ROOT}/${DATA}.npz"
    [[ -s "${prepared_npz}" ]] || { echo "[missing] ${prepared_npz}" >&2; exit 3; }
    # Original optimizer-level batch is 32.  DDP4: 4 * B8 = 32.
    env DATASET=prepared_temporal_driven COND="${DATA}" K=64 \
      METHOD="dwc${GAIN}" SEED="${SEED}" GPUS="${GPUS}" HIDDEN=128 \
      BATCH=8 GRAD_ACCUM=1 EPOCHS=100 ES=20 LR=1e-4 \
      WEIGHT_DECAY=1e-4 GRAD_CLIP=1.0 AR_OPTIMIZER=adam \
      AR_SCHEDULER=step NUM_WORKERS=0 MAMBA_TRAIN_STARTS=16 \
      PREPARED_NPZ="${prepared_npz}" RECURRENT_EVAL_HORIZON_BATCH=9 \
      SAVE_BASE="${TRAIN_ROOT}" SKIP_EXISTING=1 RESUME=auto \
      bash scripts/train/run_mem_one.sh
    ;;
  shear)
    # Original optimizer-level batch is 4.  DDP4: 4 * B1 = 4.
    env DATASET=shear_flow METHOD="dwc${GAIN}" SEED="${SEED}" K=32 \
      GPUS="${GPUS}" BATCH=1 GRAD_ACCUM=1 EPOCHS=100 \
      EARLY_STOP_PATIENCE=20 LR=3e-4 GRAD_CLIP=1.0 \
      AR_SCHEDULER=cosine AR_MIN_LR=1e-6 NUM_WORKERS=0 \
      SAVE_BASE="${TRAIN_ROOT}" SKIP_EXISTING=1 RESUME=auto \
      bash scripts/train/run_thewell_arm.sh
    ;;
  *) echo "[refuse] unknown DATA=${DATA}" >&2; exit 2 ;;
esac

