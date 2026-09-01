#!/usr/bin/env bash
set -euo pipefail

# Single-GPU, serial real-data delayed-gradient utility audit.
# No training is launched.  Every dataset is probed at a frozen Exact-BPTT
# checkpoint, and completed JSON files are skipped unless FORCE=1.

GPU="${GPU:-0}"
SEED="${SEED:-0}"
CALIBRATION_SPLIT="${CALIBRATION_SPLIT:-train}"
EVALUATION_SPLIT="${EVALUATION_SPLIT:-test}"
FORCE="${FORCE:-0}"
DATASETS="${DATASETS:-mg,narma,ieeg,shear,wb2}"
OUT_ROOT="${OUT_ROOT:-probe_outputs/real_heldout_gradient_utility_v1}"
LOG_ROOT="${LOG_ROOT:-logs/real_heldout_gradient_utility_v1}"
SEQ_PAIRS="${SEQ_PAIRS:-8}"
FIELD_PAIRS="${FIELD_PAIRS:-4}"
WB2_PAIRS="${WB2_PAIRS:-2}"
FINITE_PAIRS="${FINITE_PAIRS:-1}"
COORDS_SEQ="${COORDS_SEQ:-250000}"
COORDS_FIELD="${COORDS_FIELD:-100000}"
RADII="${RADII:-2.5e-7,5e-7,1e-6}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

contains_dataset() {
  case ",${DATASETS}," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

run_one() {
  local label="$1"
  local ckpt="$2"
  local K="$3"
  local pairs="$4"
  local coords="$5"
  local finite_horizons="$6"
  local out="${OUT_ROOT}/${label}_seed${SEED}.json"
  local log="${LOG_ROOT}/${label}_seed${SEED}.log"

  if [[ ! -f "${ckpt}" ]]; then
    echo "[fatal] missing checkpoint: ${ckpt}" >&2
    exit 2
  fi
  if [[ -s "${out}" && "${FORCE}" != "1" ]]; then
    echo "[skip] ${label}: ${out} already exists"
    return
  fi

  echo "[start] ${label} K=${K} pairs=${pairs} GPU=${GPU} $(date -u)"
  CUDA_VISIBLE_DEVICES="${GPU}" python -u scripts/probes/probe_heldout_delayed_gradient_utility.py \
    --ckpt "${ckpt}" \
    --K "${K}" \
    --num_pairs "${pairs}" \
    --finite_pairs "${FINITE_PAIRS}" \
    --finite_horizons "${finite_horizons}" \
    --relative_radii "${RADII}" \
    --coord_subsample "${coords}" \
    --calibration_split "${CALIBRATION_SPLIT}" \
    --evaluation_split "${EVALUATION_SPLIT}" \
    --seed "${SEED}" \
    --gpu 0 \
    --out "${out}" \
    > "${log}" 2>&1
  echo "[done] ${label}: ${out} $(date -u)"
  tail -n 14 "${log}"
}

if contains_dataset mg; then
  run_one \
    mg \
    "experiments/memtest/mackey_glass/tau30_K32/ckpt/seed${SEED}/best.pth" \
    32 "${SEQ_PAIRS}" "${COORDS_SEQ}" "1,4,8,16,32"
fi

if contains_dataset narma; then
  run_one \
    narma \
    "experiments/memtest/narma/L5_K32/ckpt/seed${SEED}/best.pth" \
    32 "${SEQ_PAIRS}" "${COORDS_SEQ}" "1,4,8,16,32"
fi

if contains_dataset ieeg; then
  run_one \
    ieeg \
    "experiments/memtest_rerun/ieeg/theta_K64/ckpt/seed${SEED}/best.pth" \
    64 "${SEQ_PAIRS}" "${COORDS_SEQ}" "1,8,16,32,64"
fi

if contains_dataset shear; then
  run_one \
    shear \
    "experiments/thewell_shear_final_lr3e4/shear_flow/unet_b32_D4_W2_K32_ds4/exact/seed${SEED}/best.pth" \
    32 "${FIELD_PAIRS}" "${COORDS_FIELD}" "1,4,8,16,32"
fi

if contains_dataset wb2; then
  run_one \
    wb2 \
    "experiments/weatherbench2/fullbudget_ddp4_v1/unet_c32_D4_W2_K48/exact/seed${SEED}/best.pth" \
    48 "${WB2_PAIRS}" "${COORDS_FIELD}" "1,8,16,32,48"
fi

echo "[all done] $(date -u)"
