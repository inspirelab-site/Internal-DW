#!/usr/bin/env bash
# Evaluation-only wrapper for one existing TBPTT-8 checkpoint under the
# paper's dense multi-origin 1:1.5K relative-L2 protocol.
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

DATA=${DATA:?set DATA=mg|narma|ieeg|fmri|shear}
SEED=${SEED:?set SEED=0|1|2}
GPU=${GPU:?set one physical GPU id}
OUT_ROOT=${OUT_ROOT:-probe_outputs/tbptt_dense_multistart_rel_l2_1p5k_v1}
MAX_ORIGINS=${MAX_ORIGINS:-64}
ORIGIN_STRIDE=${ORIGIN_STRIDE:-1}
BOOTSTRAP_DRAWS=${BOOTSTRAP_DRAWS:-10000}
NUM_WORKERS=${NUM_WORKERS:-0}
SWEEP_ROOT=${SWEEP_ROOT:-experiments/tbptt_positive_sweep_v1}

case "${DATA}" in
  mg)
    CKPT=${CKPT:-${SWEEP_ROOT}/mg/mackey_glass/tau30_K32/tbptt8/seed${SEED}/best.pth}
    K=32; H=48; ORIGIN_BATCH=${ORIGIN_BATCH:-16}
    ;;
  narma)
    CKPT=${CKPT:?set CKPT to the TBPTT-8 checkpoint}
    K=32; H=48; ORIGIN_BATCH=${ORIGIN_BATCH:-16}
    ;;
  ieeg)
    CKPT=${CKPT:?set CKPT to the subject-specific TBPTT-8 checkpoint}
    K=64; H=96; ORIGIN_BATCH=${ORIGIN_BATCH:-8}
    ;;
  fmri)
    CKPT=${CKPT:?set CKPT to the TBPTT-8 checkpoint}
    K=64; H=96; ORIGIN_BATCH=${ORIGIN_BATCH:-2}
    ;;
  shear)
    CKPT=${CKPT:-${SWEEP_ROOT}/shear/shear_flow/unet_b32_D4_W2_K32_ds4/tbptt8/seed${SEED}/best.pth}
    K=32; H=48; ORIGIN_BATCH=${ORIGIN_BATCH:-2}
    ;;
  *)
    echo "[refuse] unsupported DATA=${DATA}" >&2
    exit 2
    ;;
esac

[[ -s "${CKPT}" ]] || { echo "[missing] ${CKPT}" >&2; exit 3; }
OUT="${OUT_ROOT}/${DATA}/tbptt8_seed${SEED}.json"
mkdir -p "$(dirname "${OUT}")"
if [[ -s "${OUT}" && "${FORCE:-0}" != "1" ]]; then
  echo "[skip] ${OUT} already exists"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "[start] data=${DATA} TBPTT-8 seed=${SEED} train_K=${K} eval_H=${H} physical_gpu=${GPU}"
echo "[ckpt] ${CKPT}"
python -u scripts/evaluate/evaluate_dense_multistart_rel_l2.py \
  --ckpt "${CKPT}" \
  --out "${OUT}" \
  --gpu 0 \
  --split test \
  --max-horizon "${H}" \
  --train-horizon "${K}" \
  --origin-stride "${ORIGIN_STRIDE}" \
  --max-origins-per-item "${MAX_ORIGINS}" \
  --origin-batch "${ORIGIN_BATCH}" \
  --num-workers "${NUM_WORKERS}" \
  --bootstrap-draws "${BOOTSTRAP_DRAWS}" \
  --method-label tbptt

echo "[done] ${OUT}"
