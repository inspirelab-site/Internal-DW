#!/usr/bin/env bash
# Prepare ONLY iEEG, or run its subject-specific models on one GPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
MODE=${1:-run}
if (( $# )); then shift; fi
PYTHON=${PYTHON:-python}
if [[ "${MODE}" == prepare ]]; then
  : "${FIF_ROOT:?Set FIF_ROOT to the preprocessed theta FIF directory}"
  : "${CLIP_ROOT:?Set CLIP_ROOT to the directory containing clip_projected.npy and clip_frames.csv}"
  exec "${PYTHON}" scripts/data/prepare_ieeg.py \
    --fif-root "${FIF_ROOT}" --clip-root "${CLIP_ROOT}" \
    --out "${IEEG_PREPARED_ROOT:-probe_inputs/ieeg_cohort_v1}" "$@"
fi
if [[ "${MODE}" == regime || "${MODE}" == utility || "${MODE}" == noise ]]; then
  exec "${PYTHON}" -u scripts/reproduce/probe_ieeg.py "${MODE}" --gpu "${GPU:-0}" "$@"
fi
OPTIONS=()
if [[ "${MODE}" == timing ]]; then
  exec "${PYTHON}" -u scripts/reproduce/time_ieeg.py --gpu "${GPU:-0}" "$@"
fi
[[ -z "${SUBJECTS:-}" ]] || { read -r -a ITEMS <<< "${SUBJECTS}"; OPTIONS+=(--subjects "${ITEMS[@]}"); }
[[ -z "${ARM:-}" ]] || OPTIONS+=(--arms "${ARM}")
[[ -z "${SEEDS:-${SEED:-}}" ]] || { read -r -a ITEMS <<< "${SEEDS:-${SEED}}"; OPTIONS+=(--seeds "${ITEMS[@]}"); }
exec "${PYTHON}" -u scripts/reproduce/train_test_ieeg.py "${MODE}" \
  --gpu "${GPU:-0}" "${OPTIONS[@]}" "$@"
