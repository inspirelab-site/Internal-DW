#!/usr/bin/env bash
# Matched seed-0 forecasting controls for the known-SNR diagonal-AR closure.
# Trains only the missing Clip, TBPTT-8, and Forward-JReg arms.  Exact-BPTT
# and diagonal-AR Internal-DW are reused by the summarizer.
set -euo pipefail
trap '' HUP

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

SEED=${SEED:-0}
GPUS=${GPUS:-0,1,2,3}
FIRST_GPU=${GPUS%%,*}
BATCH=${BATCH:-4}
TARGET_EFFECTIVE_BATCH=32
IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
WORLD_SIZE=${#GPU_IDS[@]}
if (( WORLD_SIZE != 4 )); then
  echo "This launcher is the strict DDP-4 protocol; GPUS must contain four ids." >&2
  exit 2
fi
denominator=$(( BATCH * WORLD_SIZE ))
if (( TARGET_EFFECTIVE_BATCH % denominator != 0 )); then
  echo "BATCH=${BATCH} x world=${WORLD_SIZE} does not divide 32" >&2
  exit 2
fi
GRAD_ACCUM=${GRAD_ACCUM:-$(( TARGET_EFFECTIVE_BATCH / denominator ))}
if (( BATCH * WORLD_SIZE * GRAD_ACCUM != TARGET_EFFECTIVE_BATCH )); then
  echo "effective batch must be 32, got $((BATCH * WORLD_SIZE * GRAD_ACCUM))" >&2
  exit 2
fi

SAVE_ROOT=${SAVE_ROOT:-experiments/known_snr_diagonal_ar_controls_v1}
LOG_ROOT=${LOG_ROOT:-logs/known_snr_diagonal_ar_controls_v1}
OUT_ROOT=${OUT_ROOT:-probe_outputs/known_snr_diagonal_ar_closure/seed${SEED}}
ARMS=${ARMS:-"clip tbptt8 jreg"}
EXACT_RUN=${EXACT_RUN:-experiments/known_snr_oracle/known_snr_ar/mixed_K32/ckpt/seed${SEED}}
DW_RUN=${DW_RUN:-experiments/known_snr_diagonal_ar_closure/known_snr_ar/mixed_K32/dualwiener_diag_ar/seed${SEED}}
mkdir -p "${SAVE_ROOT}" "${LOG_ROOT}" "${OUT_ROOT}"

[[ "${SEED}" == 0 ]] || { echo "This closure is pre-registered at seed 0." >&2; exit 2; }
[[ -s "${EXACT_RUN}/eval_results.json" ]] || { echo "missing Exact result" >&2; exit 2; }
[[ -s "${DW_RUN}/eval_results.json" ]] || { echo "missing diagonal-AR DW result" >&2; exit 2; }

cat > "${OUT_ROOT}/forecasting_controls_protocol.json" <<EOF
{
  "seed": 0,
  "K": 32,
  "world_size": 4,
  "local_batch_size": ${BATCH},
  "grad_accum_steps": ${GRAD_ACCUM},
  "effective_batch_size": 32,
  "optimizer": "adam",
  "learning_rate": 0.0001,
  "weight_decay": 0.0001,
  "scheduler": "step",
  "min_learning_rate": 0.000001,
  "epochs": 100,
  "early_stop_patience": 20,
  "loss": "mse",
  "train_starts_per_sequence": 1,
  "arms": {
    "clip": {"method": "ckpt", "grad_clip": 0.1},
    "tbptt8": {"method": "tbptt8", "period": 8, "grad_clip": 1.0},
    "jreg": {"method": "ckpt", "grad_clip": 1.0, "lambda": 0.1, "target": 1.0, "epsilon": 0.001}
  }
}
EOF

run_arm() {
  local arm="$1" method="$2" grad_clip="$3" extra_args="$4" port="$5"
  local save_base="${SAVE_ROOT}/${arm}"
  local log="${LOG_ROOT}/${arm}_seed${SEED}.log"
  echo "[launch] ${arm}: DDP4 GPUs=${GPUS}, effective batch=${BATCH}x4x${GRAD_ACCUM}=32"
  env -u DUAL_WIENER_CONST -u DUAL_WIENER_INNOVATION_FILE \
    -u DUAL_WIENER_INNOVATION_KEY -u DUAL_WIENER_DOMAIN_NOISE_MODEL \
    -u GLOBAL_WIENER_BATCH_CONDITIONED -u GLOBAL_WIENER_SUPERBATCH_GROUPS \
    -u RESGRAD_ALPHA_PERIOD -u RESGRAD_ALPHA_VALUE \
    GPU="${FIRST_GPU}" GPUS="${GPUS}" \
    DATASET=known_snr_ar COND=mixed K=32 METHOD="${method}" SEED="${SEED}" \
    BATCH="${BATCH}" GRAD_ACCUM="${GRAD_ACCUM}" \
    EPOCHS=100 ES=20 LR=1e-4 WEIGHT_DECAY=1e-4 GRAD_CLIP="${grad_clip}" \
    AR_OPTIMIZER=adam AR_SCHEDULER=step AR_MIN_LR=1e-6 \
    MAMBA_LOSS_TYPE=mse MAMBA_TRAIN_STARTS=1 \
    HIDDEN=128 SIMPLE_DEPTH=4 SIMPLE_NHEAD=8 NUM_WORKERS=0 \
    RECURRENT_EVAL_HORIZON_BATCH=9 EXTRA_ARGS="${extra_args}" \
    SAVE_BASE="${save_base}" SKIP_EXISTING=1 RESUME=auto MASTER_PORT="${port}" \
    bash scripts/train/run_mem_one.sh 2>&1 | tee -a "${log}"
}

# ARMS allows different four-GPU servers to claim disjoint controls.  Run
# directories are arm-specific, so the jobs never overwrite one another.
for arm in ${ARMS}; do
  case "${arm}" in
    clip)
      run_arm clip ckpt 0.1 "" "${CLIP_PORT:-61201}"
      ;;
    tbptt8)
      run_arm tbptt8 tbptt8 1.0 "" "${TBPTT_PORT:-61202}"
      ;;
    jreg)
      run_arm jreg ckpt 1.0 \
        "--forward_jacobian_lambda 0.1 --forward_jacobian_target 1.0 --forward_jacobian_eps 0.001" \
        "${JREG_PORT:-61203}"
      ;;
    *)
      echo "unknown arm '${arm}'; choose from: clip tbptt8 jreg" >&2
      exit 2
      ;;
  esac
done

CLIP_RUN="${SAVE_ROOT}/clip/known_snr_ar/mixed_K32/ckpt/seed${SEED}"
TBPTT_RUN="${SAVE_ROOT}/tbptt8/known_snr_ar/mixed_K32/tbptt8/seed${SEED}"
JREG_RUN="${SAVE_ROOT}/jreg/known_snr_ar/mixed_K32/ckpt/seed${SEED}"
if [[ -s "${CLIP_RUN}/eval_results.json" \
   && -s "${TBPTT_RUN}/eval_results.json" \
   && -s "${JREG_RUN}/eval_results.json" ]]; then
  python scripts/results/build_known_snr_results.py controls \
    --exact-run "${EXACT_RUN}" \
    --dw-run "${DW_RUN}" \
    --clip-run "${CLIP_RUN}" \
    --tbptt-run "${TBPTT_RUN}" \
    --jreg-run "${JREG_RUN}" \
    --protocol "${OUT_ROOT}/forecasting_controls_protocol.json" \
    --output "${OUT_ROOT}/forecasting_controls.json"
  python scripts/results/build_known_snr_results.py closure \
    --root "${OUT_ROOT}" --quiet
else
  echo "[partial] requested arms finished; another server is completing the remaining controls"
fi

echo "[done] matched known-SNR forecasting controls"
