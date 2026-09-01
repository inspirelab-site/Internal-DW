#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT=$(cd "$(dirname "$0")/../.." && pwd -P)
cd "${PROJECT_ROOT:-${SCRIPT_ROOT}}"

# Evaluation-only runner for one checkpoint in the fixed-horizon K sweep.
# Every training horizon for a dataset is evaluated at the same forecast
# horizon, so K changes the training/backward path without changing the metric.
#
# Example:
#   DATA=mg ARM=dw K=32 SEED=1 SPLIT=val FIXED_H=64 GPU=0 \
#     bash scripts/evaluate/run_internal_dw_fixed_horizon_one.sh

DATA=${DATA:?set DATA=mg|narma|ieeg|fmri|ettm1|ettm2|shear|wb2}
ARM=${ARM:?set ARM=exact|dw}
K=${K:?set the training horizon}
SEED=${SEED:-1}
SPLIT=${SPLIT:-val}
FIXED_H=${FIXED_H:?set the common evaluation horizon for this dataset}
GPU=${GPU:?set one physical GPU id}

OUT_ROOT=${OUT_ROOT:-probe_outputs/internal_dw_fixed_horizon_positive_v1}
MAX_ORIGINS=${MAX_ORIGINS:-64}
ORIGIN_STRIDE=${ORIGIN_STRIDE:-1}
BOOTSTRAP_DRAWS=${BOOTSTRAP_DRAWS:-10000}
NUM_WORKERS=${NUM_WORKERS:-0}
WB2_DATA=${WB2_DATA:-data/weatherbench2_1p5_pilot}

case "${ARM}" in exact|dw) ;; *) echo "unsupported ARM=${ARM}" >&2; exit 2 ;; esac
case "${SPLIT}" in val|test) ;; *) echo "unsupported SPLIT=${SPLIT}" >&2; exit 2 ;; esac
if (( K <= 0 || FIXED_H <= 0 || K > FIXED_H )); then
  echo "require 0 < K <= FIXED_H; got K=${K}, FIXED_H=${FIXED_H}" >&2
  exit 2
fi

BASE=experiments/internal_dw_baselines_k_v1/B
case "${DATA}:${K}:${ARM}" in
  # The assigned-K checkpoints were trained in the primary experiment roots
  # and reused by the sweep.  All other K values live below BASE.
  mg:32:exact)
    CKPT="experiments/memtest/mackey_glass/tau30_K32/ckpt/seed${SEED}/best.pth" ;;
  mg:32:dw)
    CKPT="experiments/internal_dw_assigned_v1/mackey_glass/tau30_K32/dualwiener_structured/seed${SEED}/best.pth" ;;
  narma:32:exact)
    CKPT="experiments/memtest/narma/L5_K32/ckpt/seed${SEED}/best.pth" ;;
  narma:32:dw)
    CKPT="experiments/internal_dw_assigned_v1/narma/L5_K32/dualwiener_structured/seed${SEED}/best.pth" ;;
  ieeg:64:exact)
    CKPT="experiments/memtest_rerun/ieeg/theta_K64/ckpt/seed${SEED}/best.pth" ;;
  ieeg:64:dw)
    CKPT="experiments/internal_dw_prior_vector_v1/ieeg/theta_K64/dualwiener_domain_ieeg_longmemory/seed${SEED}/best.pth" ;;
  fmri:64:exact)
    CKPT="experiments/hcp_movie1/resgrad_mamba_converge/official_mamba_state_hid4096_D4_residual_gradCheckpoint_BPTT64_S16/seed${SEED}/best.pth" ;;
  fmri:64:dw)
    CKPT="experiments/hcp_movie1/internal_dw_prior_fmri_ddp4_v1/official_mamba_state_hid4096_D4_residual_resgradDualWienerDomainSubjectCrossfit_BPTT64_S16/seed${SEED}/best.pth" ;;
  wb2:48:exact)
    CKPT="experiments/weatherbench2/fullbudget_ddp4_v1/unet_c32_D4_W2_K48/exact/seed${SEED}/best.pth" ;;
  wb2:48:dw)
    CKPT="experiments/weatherbench2/assigned_spectral_ddp4_v1/unet_c32_D4_W2_K48/dualwiener_spectral/seed${SEED}/best.pth" ;;
  ettm1:64:exact|ettm2:64:exact)
    CKPT="experiments/temporal_candidate_end2end_single_v1/prepared_temporal_driven/${DATA}_K64/ckpt/seed${SEED}/best.pth" ;;
  ettm1:64:dw|ettm2:64:dw)
    CKPT="experiments/temporal_candidate_end2end_single_v1/prepared_temporal_driven/${DATA}_K64/dualwiener_structured/seed${SEED}/best.pth" ;;
  shear:32:exact)
    CKPT="experiments/thewell_shear_final_lr3e4/shear_flow/unet_b32_D4_W2_K32_ds4/exact/seed${SEED}/best.pth" ;;
  shear:32:dw)
    CKPT="experiments/thewell_shear_final_lr3e4/shear_flow/unet_b32_D4_W2_K32_ds4/dualwiener_spectral/seed${SEED}/best.pth" ;;
  mg:*:exact)
    CKPT="${BASE}/exact/mackey_glass/tau30_K${K}/ckpt/seed${SEED}/best.pth" ;;
  mg:*:dw)
    CKPT="${BASE}/dw/mackey_glass/tau30_K${K}/dualwiener_structured/seed${SEED}/best.pth" ;;
  narma:*:exact)
    CKPT="${BASE}/exact/narma/L5_K${K}/ckpt/seed${SEED}/best.pth" ;;
  narma:*:dw)
    CKPT="${BASE}/dw/narma/L5_K${K}/dualwiener_structured/seed${SEED}/best.pth" ;;
  ieeg:*:exact)
    CKPT="${BASE}/exact/ieeg/theta_K${K}/ckpt/seed${SEED}/best.pth" ;;
  ieeg:*:dw)
    CKPT="${BASE}/dw/ieeg/theta_K${K}/dualwiener_domain_ieeg_longmemory/seed${SEED}/best.pth" ;;
  fmri:*:exact)
    CKPT="${BASE}/exact/fmri_K${K}/official_mamba_state_hid4096_D4_residual_baseline_BPTT${K}_S16/seed${SEED}/best.pth" ;;
  fmri:*:dw)
    CKPT="${BASE}/dw/fmri_K${K}/official_mamba_state_hid4096_D4_residual_resgradDualWienerDomainSubjectCrossfit_BPTT64_S16/seed${SEED}/best.pth" ;;
  ettm1:*:exact|ettm2:*:exact)
    CKPT="${BASE}/exact/prepared_temporal_driven/${DATA}_K${K}/ckpt/seed${SEED}/best.pth" ;;
  ettm1:*:dw|ettm2:*:dw)
    CKPT="${BASE}/dw/prepared_temporal_driven/${DATA}_K${K}/dualwiener_structured/seed${SEED}/best.pth" ;;
  shear:*:exact)
    CKPT="${BASE}/exact/shear_flow/unet_b32_D4_W2_K${K}_ds4/exact/seed${SEED}/best.pth" ;;
  shear:*:dw)
    CKPT="${BASE}/dw/shear_flow/unet_b32_D4_W2_K${K}_ds4/dualwiener_spectral/seed${SEED}/best.pth" ;;
  wb2:*:exact)
    CKPT="${BASE}/exact/weatherbench2/unet_c32_D4_W2_K${K}/exact/seed${SEED}/best.pth" ;;
  wb2:*:dw)
    CKPT="${BASE}/dw/weatherbench2/unet_c32_D4_W2_K${K}/dualwiener_spectral/seed${SEED}/best.pth" ;;
  *)
    echo "unsupported DATA:K:ARM=${DATA}:${K}:${ARM}" >&2
    exit 2 ;;
esac

case "${DATA}" in
  mg) ORIGIN_BATCH=${ORIGIN_BATCH:-16} ;;
  narma) ORIGIN_BATCH=${ORIGIN_BATCH:-16} ;;
  ieeg) ORIGIN_BATCH=${ORIGIN_BATCH:-8} ;;
  fmri) ORIGIN_BATCH=${ORIGIN_BATCH:-2} ;;
  ettm1|ettm2) ORIGIN_BATCH=${ORIGIN_BATCH:-8} ;;
  shear) ORIGIN_BATCH=${ORIGIN_BATCH:-2} ;;
esac

if [[ ! -s "${CKPT}" ]]; then
  echo "missing checkpoint: ${CKPT}" >&2
  exit 3
fi

OUT="${OUT_ROOT}/${SPLIT}/${DATA}/${ARM}_K${K}_seed${SEED}_H${FIXED_H}.json"
mkdir -p "$(dirname "${OUT}")"
if [[ -s "${OUT}" && "${FORCE:-0}" != "1" ]]; then
  echo "[skip] ${OUT} already exists"
  exit 0
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[dry-run] data=${DATA} arm=${ARM} seed=${SEED} train_K=${K} fixed_H=${FIXED_H} split=${SPLIT}"
  echo "[ckpt] ${CKPT}"
  echo "[out] ${OUT}"
  exit 0
fi

if [[ "${DISABLE_EVAL_FLOCK:-0}" != "1" ]] && command -v flock >/dev/null 2>&1; then
  exec {EVAL_LOCK_FD}>"${OUT}.lock"
  if ! flock -n "${EVAL_LOCK_FD}"; then
    echo "[skip] evaluation already claimed: ${OUT}"
    exit 0
  fi
fi

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "[start] data=${DATA} arm=${ARM} seed=${SEED} train_K=${K} fixed_H=${FIXED_H} split=${SPLIT} physical_gpu=${GPU}"
echo "[ckpt] ${CKPT}"
if [[ "${ARM}" == "exact" ]]; then
  METHOD_LABEL=full_bptt
else
  METHOD_LABEL=internal_dw
fi
if [[ "${DATA}" == "wb2" ]]; then
  mapfile -t HORIZONS < <(seq 1 "${FIXED_H}")
  python -u scripts/evaluate/evaluate_weatherbench2_acc.py \
    --ckpt "${CKPT}" --data "${WB2_DATA}" --split "${SPLIT}" --gpu 0 \
    --horizons "${HORIZONS[@]}" --train-horizon "${K}" \
    --batch-size "${WB2_BATCH:-2}" --start-stride "${ORIGIN_STRIDE}" \
    --num-starts "${WB2_NUM_STARTS:-${MAX_ORIGINS}}" --evenly-spaced-starts \
    --method-label "${METHOD_LABEL}" \
    --out "${OUT}"
else
  python -u scripts/evaluate/evaluate_dense_multistart_rel_l2.py \
    --ckpt "${CKPT}" \
    --out "${OUT}" \
    --gpu 0 \
    --split "${SPLIT}" \
    --max-horizon "${FIXED_H}" \
    --train-horizon "${K}" \
    --origin-stride "${ORIGIN_STRIDE}" \
    --max-origins-per-item "${MAX_ORIGINS}" \
    --origin-batch "${ORIGIN_BATCH}" \
    --num-workers "${NUM_WORKERS}" \
    --bootstrap-draws "${BOOTSTRAP_DRAWS}" \
    --method-label "${METHOD_LABEL}"
fi
echo "[done] ${OUT}"
