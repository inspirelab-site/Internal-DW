#!/usr/bin/env bash
# Dense 1:1.5K test evaluation of one validation-selected static checkpoint.
set -euo pipefail

cd "$(dirname "$0")/../.."
DATA=${DATA:?set DATA=mg|ettm1|ettm2|shear}
SEED=${SEED:?set SEED=0|1|2}
GPU=${GPU:?set one physical GPU id}
SELECTION=${SELECTION:-probe_outputs/internal_dw_static_positive_v1/selection.json}
TRAIN_ROOT=${TRAIN_ROOT:-experiments/internal_dw_static_positive_v1}
OUT_ROOT=${OUT_ROOT:-probe_outputs/internal_dw_static_positive_v1/dense}
[[ -s "${SELECTION}" ]] || { echo "[missing] ${SELECTION}" >&2; exit 2; }

# Avoid unhealthy shared-cluster MPS initialization by default.
if [[ "${USE_MPS:-0}" != 1 ]]; then
  export CUDA_MPS_PIPE_DIRECTORY="$(pwd)/.mps_bypass"
  mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}"
fi

GAIN=$(python - "${SELECTION}" "${DATA}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["datasets"][sys.argv[2]]["selected_c"])
PY
)
case "${GAIN}" in 0.3|0.6|0.9) ;; *) echo "[bad selection] ${DATA}: ${GAIN}" >&2; exit 2 ;; esac

case "${DATA}" in
  mg)
    K=32; H=48; ORIGIN_BATCH=${ORIGIN_BATCH:-16}
    CKPT="${TRAIN_ROOT}/mackey_glass/tau30_K32/dwc${GAIN}/seed${SEED}/best.pth"
    ;;
  ettm1|ettm2)
    K=64; H=96; ORIGIN_BATCH=${ORIGIN_BATCH:-8}
    CKPT="${TRAIN_ROOT}/prepared_temporal_driven/${DATA}_K64/dwc${GAIN}/seed${SEED}/best.pth"
    ;;
  shear)
    K=32; H=48; ORIGIN_BATCH=${ORIGIN_BATCH:-2}
    CKPT="${TRAIN_ROOT}/shear_flow/unet_b32_D4_W2_K32_ds4/dwc${GAIN}/seed${SEED}/best.pth"
    ;;
  *) echo "[refuse] unknown DATA=${DATA}" >&2; exit 2 ;;
esac
[[ -s "${CKPT}" ]] || { echo "[missing] ${CKPT}" >&2; exit 3; }

OUT="${OUT_ROOT}/${DATA}/static_c${GAIN}_seed${SEED}.json"
mkdir -p "$(dirname "${OUT}")"
if [[ -s "${OUT}" && "${FORCE_EVAL:-0}" != 1 ]]; then
  echo "[skip] ${OUT}"
  exit 0
fi
if command -v flock >/dev/null 2>&1; then
  exec {EVAL_LOCK_FD}>"${OUT}.lock"
  flock -n "${EVAL_LOCK_FD}" || { echo "[skip] already claimed: ${OUT}"; exit 0; }
fi

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Gains only affect backward.  Test prediction uses the trained checkpoint's
# parameters and must not depend on a shell-side routing intervention.
unset DUAL_WIENER_CONST DUAL_WIENER_INNOVATION_FILE DUAL_WIENER_INNOVATION_KEY

python -u scripts/evaluate/evaluate_dense_multistart_rel_l2.py \
  --ckpt "${CKPT}" --out "${OUT}" --gpu 0 --split test \
  --max-horizon "${H}" --train-horizon "${K}" \
  --origin-stride 1 --max-origins-per-item 64 \
  --origin-batch "${ORIGIN_BATCH}" --num-workers 0 --bootstrap-draws 10000 \
  --method-label static_gain

python - "${OUT}" "${K}" "${H}" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
k, h = map(int, sys.argv[2:])
assert data["split"] == "test"
assert data["train_horizon"] == k and data["eval_horizon"] == h
assert data["primary_metric"]["horizons"] == f"1:{h}"
assert data["per_horizon"]["horizons"] == list(range(1, h + 1))
print(f"[certified] ALL 1:{h}={data['primary_metric']['value']:.6f}")
PY
echo "[done] ${DATA} c=${GAIN} seed${SEED}: ${OUT}"
