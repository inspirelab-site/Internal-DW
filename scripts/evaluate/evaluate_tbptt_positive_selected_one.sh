#!/usr/bin/env bash
# Dense fixed-protocol test of one already-selected TBPTT checkpoint.
set -euo pipefail
cd "$(dirname "$0")/../.."

DATA=${DATA:?set DATA=mg|ettm1|ettm2|shear}
SEED=${SEED:?set SEED}
GPU=${GPU:?set one physical GPU id}
SELECTION=${SELECTION:-probe_outputs/tbptt_positive_sweep_v1/selection.json}
SWEEP_ROOT=${SWEEP_ROOT:-experiments/tbptt_positive_sweep_v1}
OUT_ROOT=${OUT_ROOT:-probe_outputs/tbptt_positive_sweep_v1/dense}
[[ -s "${SELECTION}" ]] || { echo "[missing] ${SELECTION}" >&2; exit 2; }

S=$(python - "${SELECTION}" "${DATA}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["datasets"][sys.argv[2]]["selected_S"])
PY
)

case "${DATA}" in
  mg)
    K=32; H=48; origin_batch=16
    if [[ "${S}" == 8 ]]; then
      legacy="probe_outputs/tbptt_dense_multistart_rel_l2_1p5k_v1/mg/tbptt8_seed${SEED}.json"
      [[ -s "${legacy}" ]] || { echo "[missing] ${legacy}" >&2; exit 3; }
      echo "[reuse] ${legacy}"
      exit 0
    fi
    ckpt="${SWEEP_ROOT}/mg/mackey_glass/tau30_K32/tbptt${S}/seed${SEED}/best.pth"
    ;;
  ettm1|ettm2)
    K=64; H=96; origin_batch=16
    ckpt="${SWEEP_ROOT}/${DATA}/prepared_temporal_driven/${DATA}_K64/tbptt${S}/seed${SEED}/best.pth"
    ;;
  shear)
    K=32; H=48; origin_batch=2
    if [[ "${S}" == 8 ]]; then
      legacy="probe_outputs/tbptt_dense_multistart_rel_l2_1p5k_v1/shear/tbptt8_seed${SEED}.json"
      [[ -s "${legacy}" ]] || { echo "[missing] ${legacy}" >&2; exit 3; }
      echo "[reuse] ${legacy}"
      exit 0
    fi
    ckpt="${SWEEP_ROOT}/shear/shear_flow/unet_b32_D4_W2_K32_ds4/tbptt${S}/seed${SEED}/best.pth"
    ;;
  *) echo "[refuse] unknown DATA=${DATA}" >&2; exit 2 ;;
esac

[[ -s "${ckpt}" ]] || { echo "[missing] ${ckpt}" >&2; exit 4; }
out="${OUT_ROOT}/${DATA}/tbptt${S}_seed${SEED}.json"
mkdir -p "$(dirname "${out}")"
if [[ -s "${out}" && "${FORCE_EVAL:-0}" != 1 ]]; then
  echo "[skip] ${out}"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
python -u scripts/evaluate/evaluate_dense_multistart_rel_l2.py \
  --ckpt "${ckpt}" --out "${out}" --gpu 0 --split test \
  --max-horizon "${H}" --train-horizon "${K}" \
  --origin-stride 1 --max-origins-per-item 64 \
  --origin-batch "${origin_batch}" --num-workers 0 --bootstrap-draws 10000 \
  --method-label tbptt
echo "[done] ${DATA} selected S=${S} seed${SEED}: ${out}"
