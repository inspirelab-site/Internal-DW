#!/usr/bin/env bash
set -euo pipefail

# One frozen-checkpoint ETTm mechanism diagnostic.  This script never trains.

cd "$(dirname "$0")/../.."
DATA=${DATA:?set DATA=ettm1 or ettm2}
TASK=${TASK:?set TASK=delayed, learned, or noise}
GPU=${GPU:?set one physical GPU id}
SNR=${SNR:-}
SEED=${SEED:-0}
FORCE=${FORCE:-0}
OUT_ROOT=${OUT_ROOT:-probe_outputs/ettm_internal_dw_diagnostics_v1}
LOG_ROOT=${LOG_ROOT:-logs/ettm_internal_dw_diagnostics_v1}

case "${DATA}" in ettm1|ettm2) ;; *) echo "[refuse] DATA=${DATA}" >&2; exit 2 ;; esac
case "${TASK}" in delayed|learned|noise) ;; *) echo "[refuse] TASK=${TASK}" >&2; exit 2 ;; esac
case "${SEED}" in 0|1|2) ;; *) echo "[refuse] SEED=${SEED}" >&2; exit 2 ;; esac
if [[ "${GPU}" == *,* || "${GPU}" == *" "* ]]; then
  echo "[refuse] GPU must be one physical id" >&2; exit 2
fi

EXACT="experiments/temporal_candidate_end2end_single_v1/prepared_temporal_driven/${DATA}_K64/ckpt/seed${SEED}/best.pth"
DW="experiments/temporal_candidate_end2end_single_v1/prepared_temporal_driven/${DATA}_K64/dualwiener_structured/seed${SEED}/best.pth"
NPZ="probe_inputs/temporal_candidate_regime_v1/${DATA}.npz"
mkdir -p "${OUT_ROOT}/${DATA}" "${LOG_ROOT}/${DATA}"

case "${TASK}" in
  delayed)
    CKPT="${EXACT}"
    OUT="${OUT_ROOT}/${DATA}/delayed_exact_seed${SEED}.json"
    ;;
  learned)
    CKPT="${DW}"
    OUT="${OUT_ROOT}/${DATA}/learned_utility_seed${SEED}.json"
    ;;
  noise)
    [[ -n "${SNR}" ]] || { echo '[refuse] noise task requires SNR' >&2; exit 2; }
    CKPT="${DW}"
    token=${SNR//./p}
    OUT="${OUT_ROOT}/${DATA}/noise_snr${token}.npz"
    ;;
esac
LOG="${LOG_ROOT}/${DATA}/$(basename "${OUT}").log"
LOCK="${OUT}.lock"
ACTIVE="${OUT}.active"

[[ -s "${CKPT}" ]] || { echo "[missing] ${CKPT}" >&2; exit 3; }
if [[ "${TASK}" == noise ]]; then
  [[ -s "${NPZ}" ]] || { echo "[missing] ${NPZ}" >&2; exit 3; }
fi
if [[ -s "${OUT}" && "${FORCE}" != 1 ]]; then echo "[skip] ${OUT}"; exit 0; fi
if command -v flock >/dev/null 2>&1; then
  exec {RUN_LOCK_FD}>"${LOCK}"
  flock -n "${RUN_LOCK_FD}" || { echo "[skip] already claimed: ${OUT}"; exit 0; }
fi
if [[ -s "${OUT}" && "${FORCE}" != 1 ]]; then exit 0; fi

cleanup() { rm -f "${ACTIVE}"; }
trap cleanup EXIT INT TERM
printf 'pid=%s\nhost=%s\ngpu=%s\nstarted=%s\n' \
  "${BASHPID}" "$(hostname)" "${GPU}" "$(date -Is)" > "${ACTIVE}"

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
unset DUAL_WIENER_CONST DUAL_WIENER_INNOVATION_FILE DUAL_WIENER_INNOVATION_KEY
unset DUAL_WIENER_FREEZE_GAINS DUAL_WIENER_RESET_NOISE_MOMENTS
unset RESGRAD_ALPHA_PERIOD RESGRAD_ALPHA_VALUE

echo "[start] ${DATA} ${TASK} seed${SEED} SNR=${SNR:-NA} GPU=${GPU} $(date -Is)"
case "${TASK}" in
  delayed)
    python -u scripts/probes/probe_heldout_delayed_gradient_utility.py \
      --ckpt "${CKPT}" --K 64 --num_pairs "${NUM_PAIRS:-8}" \
      --finite_pairs "${FINITE_PAIRS:-1}" \
      --finite_horizons "1,8,16,32,64" \
      --relative_radii "${RADII:-2.5e-7,5e-7,1e-6}" \
      --coord_subsample "${COORDS:-250000}" \
      --calibration_split train --evaluation_split test \
      --seed "${SEED}" --gpu 0 --out "${OUT}" > "${LOG}" 2>&1
    ;;
  learned)
    python -u scripts/probes/probe_real_internal_dw_utility.py \
      --ckpt "${CKPT}" --K 64 --num_pairs "${NUM_PAIRS:-8}" \
      --finite_pairs "${FINITE_PAIRS:-1}" \
      --finite_horizons "2,8,16,32,64" \
      --relative_radii "${RADII:-2.5e-7,5e-7,1e-6}" \
      --coord_subsample "${COORDS:-250000}" \
      --calibration_split train --evaluation_split test \
      --seed "${SEED}" --gpu 0 --out "${OUT}" > "${LOG}" 2>&1
    ;;
  noise)
    python -u scripts/probes/probe_wiener_oracle.py \
      --ckpt "${CKPT}" --npz "${NPZ}" \
      --state-key train_state --stim-key train_drive \
      --data-preprocess prepared_temporal \
      --hidden 128 --depth 4 --K 64 --burnin 32 --batch 32 \
      --dual-wiener-noise-model lagged_residual_bootstrap \
      --draws 4 --noise-draws "${NOISE_DRAWS:-64}" \
      --signal residual --sigma-mode iso --snr "${SNR}" --grid 101 \
      --pipeline "${PIPELINE_BATCHES:-480}" --sigma-source plugin \
      --pipeline-reset-buffers --seed "${SEED}" --device cuda --out "${OUT}" \
      > "${LOG}" 2>&1
    ;;
esac

[[ -s "${OUT}" ]] || { echo "[failed] output missing: ${OUT}" >&2; exit 4; }
echo "[done] ${OUT} $(date -Is)"
tail -n 16 "${LOG}"
