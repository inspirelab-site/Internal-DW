#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# Evaluation-only wrapper for one paper checkpoint.
# Example:
#   DATA=mg ARM=dw SEED=0 GPU=0 bash scripts/evaluate/run_assigned_dense_eval_one.sh

DATA=${DATA:?set DATA=mg|narma|ieeg|fmri|shear|wb2}
ARM=${ARM:?set ARM=exact|dw}
SEED=${SEED:?set SEED=0|1|2}
GPU=${GPU:?set one physical GPU id}
EVAL_NUM=${EVAL_NUM:-1}
EVAL_DEN=${EVAL_DEN:-1}
if (( EVAL_NUM <= 0 || EVAL_DEN <= 0 )); then
  echo "EVAL_NUM and EVAL_DEN must be positive" >&2
  exit 2
fi
OUT_ROOT=${OUT_ROOT:-probe_outputs/dense_multistart_rel_l2_v1}
MAX_ORIGINS=${MAX_ORIGINS:-64}
ORIGIN_STRIDE=${ORIGIN_STRIDE:-1}
BOOTSTRAP_DRAWS=${BOOTSTRAP_DRAWS:-10000}
NUM_WORKERS=${NUM_WORKERS:-0}
WB2_DATA=${WB2_DATA:-data/weatherbench2_1p5_pilot}
WB2_BATCH=${WB2_BATCH:-2}
WB2_NUM_STARTS=${WB2_NUM_STARTS:-${MAX_ORIGINS}}

case "${DATA}:${ARM}" in
  mg:exact)
    CKPT="experiments/memtest/mackey_glass/tau30_K32/ckpt/seed${SEED}/best.pth"; K=32; ORIGIN_BATCH=${ORIGIN_BATCH:-16} ;;
  mg:dw)
    CKPT="experiments/internal_dw_assigned_v1/mackey_glass/tau30_K32/dualwiener_structured/seed${SEED}/best.pth"; K=32; ORIGIN_BATCH=${ORIGIN_BATCH:-16} ;;
  narma:exact)
    CKPT="experiments/memtest/narma/L5_K32/ckpt/seed${SEED}/best.pth"; K=32; ORIGIN_BATCH=${ORIGIN_BATCH:-16} ;;
  narma:dw)
    CKPT="experiments/internal_dw_assigned_v1/narma/L5_K32/dualwiener_structured/seed${SEED}/best.pth"; K=32; ORIGIN_BATCH=${ORIGIN_BATCH:-16} ;;
  ieeg:exact)
    CKPT="experiments/memtest_rerun/ieeg/theta_K64/ckpt/seed${SEED}/best.pth"; K=64; ORIGIN_BATCH=${ORIGIN_BATCH:-8} ;;
  ieeg:dw)
    CKPT="experiments/internal_dw_prior_vector_v1/ieeg/theta_K64/dualwiener_domain_ieeg_longmemory/seed${SEED}/best.pth"; K=64; ORIGIN_BATCH=${ORIGIN_BATCH:-8} ;;
  fmri:exact)
    CKPT="experiments/hcp_movie1/resgrad_mamba_converge/official_mamba_state_hid4096_D4_residual_gradCheckpoint_BPTT64_S16/seed${SEED}/best.pth"; K=64; ORIGIN_BATCH=${ORIGIN_BATCH:-2} ;;
  fmri:dw)
    CKPT="experiments/hcp_movie1/internal_dw_prior_fmri_ddp4_v1/official_mamba_state_hid4096_D4_residual_resgradDualWienerDomainSubjectCrossfit_BPTT64_S16/seed${SEED}/best.pth"; K=64; ORIGIN_BATCH=${ORIGIN_BATCH:-2} ;;
  shear:exact)
    CKPT="experiments/thewell_shear_final_lr3e4/shear_flow/unet_b32_D4_W2_K32_ds4/exact/seed${SEED}/best.pth"; K=32; ORIGIN_BATCH=${ORIGIN_BATCH:-2} ;;
  shear:dw)
    CKPT="experiments/thewell_shear_final_lr3e4/shear_flow/unet_b32_D4_W2_K32_ds4/dualwiener_spectral/seed${SEED}/best.pth"; K=32; ORIGIN_BATCH=${ORIGIN_BATCH:-2} ;;
  wb2:exact)
    CKPT="experiments/weatherbench2/fullbudget_ddp4_v1/unet_c32_D4_W2_K48/exact/seed${SEED}/best.pth"; K=48 ;;
  wb2:dw)
    CKPT="experiments/weatherbench2/assigned_spectral_ddp4_v1/unet_c32_D4_W2_K48/dualwiener_spectral/seed${SEED}/best.pth"; K=48 ;;
  *)
    echo "unsupported DATA=${DATA} ARM=${ARM}" >&2
    exit 2
    ;;
esac

TRAIN_K=${K}
EVAL_HORIZON=${EVAL_HORIZON:-$(( (TRAIN_K * EVAL_NUM + EVAL_DEN - 1) / EVAL_DEN ))}
if (( EVAL_HORIZON < TRAIN_K )); then
  echo "EVAL_HORIZON=${EVAL_HORIZON} must be >= TRAIN_K=${TRAIN_K}" >&2
  exit 2
fi

if [[ ! -s "${CKPT}" ]]; then
  echo "missing checkpoint: ${CKPT}" >&2
  exit 3
fi

OUT="${OUT_ROOT}/${DATA}/${ARM}_seed${SEED}.json"
mkdir -p "$(dirname "${OUT}")"
if [[ -s "${OUT}" && "${FORCE:-0}" != "1" ]]; then
  echo "[skip] ${OUT} already exists"
  exit 0
fi

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "[start] data=${DATA} arm=${ARM} seed=${SEED} train_K=${TRAIN_K} eval_H=${EVAL_HORIZON} physical_gpu=${GPU}"
echo "[ckpt] ${CKPT}"
if [[ "${DATA}" == "wb2" ]]; then
  mapfile -t HORIZONS < <(seq 1 "${EVAL_HORIZON}")
  python -u scripts/evaluate/evaluate_weatherbench2_acc.py \
    --ckpt "${CKPT}" \
    --data "${WB2_DATA}" \
    --split test \
    --gpu 0 \
    --horizons "${HORIZONS[@]}" \
    --train-horizon "${TRAIN_K}" \
    --batch-size "${WB2_BATCH}" \
    --start-stride "${ORIGIN_STRIDE}" \
    --num-starts "${WB2_NUM_STARTS}" \
    --evenly-spaced-starts \
    --out "${OUT}"
else
  python -u scripts/evaluate/evaluate_dense_multistart_rel_l2.py \
    --ckpt "${CKPT}" \
    --out "${OUT}" \
    --gpu 0 \
    --split test \
    --max-horizon "${EVAL_HORIZON}" \
    --train-horizon "${TRAIN_K}" \
    --origin-stride "${ORIGIN_STRIDE}" \
    --max-origins-per-item "${MAX_ORIGINS}" \
    --origin-batch "${ORIGIN_BATCH}" \
    --num-workers "${NUM_WORKERS}" \
    --bootstrap-draws "${BOOTSTRAP_DRAWS}"
fi
echo "[done] ${OUT}"
