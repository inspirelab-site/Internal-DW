#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# Evaluation only: one completed A (Clip/JReg) or B (K-sweep) checkpoint.
PHASE=${PHASE:?set A or B}
DATA=${DATA:?set mg|narma|ieeg|fmri|shear|wb2|ettm1|ettm2}
ARM=${ARM:?set clip|jreg|exact|dw}
K=${K:?set training horizon}
SEED=${SEED:?set seed}
GPU=${GPU:?set one physical GPU}

ROOT=${ROOT:-experiments/internal_dw_baselines_k_v1}
OUT_ROOT=${OUT_ROOT:-probe_outputs/internal_dw_AB_dense_1p5k_v1}
SYNC_ROOT=${SYNC_ROOT:-artifacts/internal_dw_baselines_k_v1_sync}
MAX_ORIGINS=${MAX_ORIGINS:-64}
ORIGIN_STRIDE=${ORIGIN_STRIDE:-1}
BOOTSTRAP_DRAWS=${BOOTSTRAP_DRAWS:-10000}
WB2_DATA=${WB2_DATA:-data/weatherbench2_1p5_pilot}

case "${PHASE}:${ARM}" in
  A:clip|A:jreg|B:exact|B:dw) ;;
  *) echo "invalid PHASE:ARM=${PHASE}:${ARM}" >&2; exit 2 ;;
esac

TASK_ID="${PHASE}_${DATA}_${ARM}_K${K}_s${SEED}"
SOURCE_DONE=${SOURCE_DONE:-${SYNC_ROOT}/${TASK_ID}.done}
if [[ ! -s "${SOURCE_DONE}" ]]; then
  echo "[wait] missing nonempty completion certificate: ${SOURCE_DONE}"
  exit 0
fi

base="${ROOT}/${PHASE}/${ARM}"
case "${DATA}" in
  mg) search="${base}/mackey_glass/tau30_K${K}"; origin_batch=16 ;;
  narma) search="${base}/narma/L5_K${K}"; origin_batch=16 ;;
  ieeg) search="${base}/ieeg/theta_K${K}"; origin_batch=8 ;;
  ettm1|ettm2) search="${base}/prepared_temporal_driven/${DATA}_K${K}"; origin_batch=8 ;;
  fmri) search="${base}/fmri_K${K}"; origin_batch=2 ;;
  shear) search="${base}/shear_flow/unet_b32_D4_W2_K${K}_ds4"; origin_batch=2 ;;
  wb2) search="${base}/weatherbench2/unet_c32_D4_W2_K${K}" ;;
  *) echo "unknown DATA=${DATA}" >&2; exit 2 ;;
esac

mapfile -t matches < <(find "${search}" -type f -path "*/seed${SEED}/best.pth" 2>/dev/null | sort)
if (( ${#matches[@]} == 0 )); then
  echo "[wait] no completed checkpoint below ${search} for seed${SEED}"
  exit 0
fi
if (( ${#matches[@]} != 1 )); then
  printf '[ambiguous] expected one checkpoint, found %d:\n' "${#matches[@]}" >&2
  printf '  %s\n' "${matches[@]}" >&2
  exit 3
fi
CKPT=${matches[0]}
RUN_DIR=$(dirname "${CKPT}")
EVAL_HORIZON=$(( (3 * K + 1) / 2 ))
OUT="${OUT_ROOT}/${PHASE}/${DATA}/${ARM}_K${K}_seed${SEED}.json"
COMPLETE_MARKER="${OUT}.source_complete"
NOT_EVALUABLE_MARKER="${OUT}.not_evaluable"
mkdir -p "$(dirname "${OUT}")"
# A nonempty .done file is the cross-server completion certificate.  Some
# runners intentionally do not write eval_results.json, so requiring that file
# incorrectly excludes completed WB2 and legacy fMRI checkpoints.
if [[ -s "${OUT}" && ! -s "${COMPLETE_MARKER}" && \
      "${SOURCE_DONE}" -ot "${OUT}" ]]; then
  printf 'checkpoint=%s\nsource_done=%s\ncompleted=%s\n' \
    "${CKPT}" "${SOURCE_DONE}" "legacy-output-newer-than-source-done" \
    > "${COMPLETE_MARKER}"
fi
if [[ -s "${OUT}" && -s "${COMPLETE_MARKER}" && "${FORCE:-0}" != "1" ]]; then
  echo "[skip] ${OUT}"
  exit 0
fi
if command -v flock >/dev/null 2>&1; then
  exec {EVAL_LOCK_FD}>"${OUT}.lock"
  if ! flock -n "${EVAL_LOCK_FD}"; then
    echo "[skip] evaluation already claimed: ${OUT}"
    exit 0
  fi
fi
if [[ -s "${OUT}" && -s "${COMPLETE_MARKER}" && "${FORCE:-0}" != "1" ]]; then exit 0; fi

# Shear test trajectories do not contain a legal origin at K=48,H=72.  Keep
# the predeclared 1.5K protocol fixed and record N/E instead of silently using
# a shorter evaluation horizon.
if [[ "${DATA}" == shear && "${K}" == 48 && "${EVAL_HORIZON}" == 72 ]]; then
  printf 'task=%s\ncheckpoint=%s\nreason=no valid test origin at fixed H=72\ncompleted=%s\n' \
    "${TASK_ID}" "${CKPT}" "$(date -Is)" > "${NOT_EVALUABLE_MARKER}"
  echo "[n/e] ${TASK_ID}: fixed H=72 leaves no valid test origin"
  exit 0
fi

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
echo "[start] ${PHASE} ${DATA} ${ARM} seed${SEED} K=${K} eval_H=${EVAL_HORIZON} GPU=${GPU}"
echo "[ckpt] ${CKPT}"

if [[ "${DATA}" == wb2 ]]; then
  mapfile -t horizons < <(seq 1 "${EVAL_HORIZON}")
  python -u scripts/evaluate/evaluate_weatherbench2_acc.py \
    --ckpt "${CKPT}" --data "${WB2_DATA}" --split test --gpu 0 \
    --horizons "${horizons[@]}" --train-horizon "${K}" \
    --batch-size "${WB2_BATCH:-2}" --start-stride "${ORIGIN_STRIDE}" \
    --num-starts "${WB2_NUM_STARTS:-${MAX_ORIGINS}}" --evenly-spaced-starts \
    --out "${OUT}"
else
  python -u scripts/evaluate/evaluate_dense_multistart_rel_l2.py \
    --ckpt "${CKPT}" --out "${OUT}" --gpu 0 --split test \
    --max-horizon "${EVAL_HORIZON}" --train-horizon "${K}" \
    --origin-stride "${ORIGIN_STRIDE}" --max-origins-per-item "${MAX_ORIGINS}" \
    --origin-batch "${origin_batch}" --num-workers 0 \
    --bootstrap-draws "${BOOTSTRAP_DRAWS}"
fi
rm -f "${NOT_EVALUABLE_MARKER}"
printf 'checkpoint=%s\nsource_done=%s\ncompleted=%s\n' \
  "${CKPT}" "${SOURCE_DONE}" "$(date -Is)" > "${COMPLETE_MARKER}"
echo "[done] ${OUT}"
