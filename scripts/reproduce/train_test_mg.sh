#!/usr/bin/env bash
# Reader-facing train/test example for one MG seed and one training arm.
set -euo pipefail

REPO_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd -P)}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

ACTION=${1:-all}
ARM=${ARM:?set ARM=full_bptt or internal_dw}
SEED=${SEED:-0}
GPU=${GPU:-0}
OUT_ROOT=${OUT_ROOT:-probe_outputs/dense_multistart_rel_l2_1p5k_v1}
MAX_ORIGINS=${MAX_ORIGINS:-64}
BOOTSTRAP_DRAWS=${BOOTSTRAP_DRAWS:-10000}

case "${ACTION}" in
  train|test|all) ;;
  *) echo "usage: ARM=<full_bptt|internal_dw> $0 [train|test|all]" >&2; exit 2 ;;
esac

case "${ARM}" in
  full_bptt)
    METHOD=ckpt
    TAG=ckpt
    METHOD_LABEL=full_bptt
    DEFAULT_SAVE_BASE=experiments/memtest
    RESULT_ARM=exact
    ;;
  internal_dw)
    METHOD=dwstructured
    TAG=dualwiener_structured
    METHOD_LABEL=internal_dw
    DEFAULT_SAVE_BASE=experiments/internal_dw_assigned_v1
    RESULT_ARM=dw
    ;;
  *) echo "ARM must be full_bptt or internal_dw; got ${ARM}" >&2; exit 2 ;;
esac
SAVE_BASE=${SAVE_BASE:-${DEFAULT_SAVE_BASE}}

RUN_DIR="${SAVE_BASE}/mackey_glass/tau30_K32/${TAG}/seed${SEED}"
CHECKPOINT="${RUN_DIR}/best.pth"
TEST_JSON="${OUT_ROOT}/mg/${RESULT_ARM}_seed${SEED}.json"

if [[ "${ACTION}" == train || "${ACTION}" == all ]]; then
  RUN_MODE=train \
  DATASET=mackey_glass COND=tau30 K=32 METHOD="${METHOD}" \
  SEED="${SEED}" GPU="${GPU}" SAVE_BASE="${SAVE_BASE}" \
    bash scripts/train/run_mem_one.sh
fi

if [[ ! -s "${CHECKPOINT}" ]]; then
  echo "[missing checkpoint] ${CHECKPOINT}" >&2
  exit 3
fi

if [[ "${ACTION}" == test || "${ACTION}" == all ]]; then
  mkdir -p "$(dirname "${TEST_JSON}")"
  RESULT_IS_CURRENT=0
  if [[ -s "${TEST_JSON}" ]] && python -c \
    'import json,sys; r=json.load(open(sys.argv[1])); raise SystemExit(0 if r.get("format_version") == 3 and "primary_metric" in r else 1)' \
    "${TEST_JSON}"; then
    RESULT_IS_CURRENT=1
  fi
  if [[ "${RESULT_IS_CURRENT}" == 1 && "${FORCE:-0}" != 1 ]]; then
    echo "[skip existing test] ${TEST_JSON}"
  else
    export CUDA_VISIBLE_DEVICES="${GPU}"
    python -u scripts/evaluate/evaluate_dense_multistart_rel_l2.py \
      --ckpt "${CHECKPOINT}" \
      --out "${TEST_JSON}" \
      --gpu 0 \
      --split test \
      --max-horizon 48 \
      --train-horizon 32 \
      --origin-stride 1 \
      --max-origins-per-item "${MAX_ORIGINS}" \
      --origin-batch 16 \
      --num-workers 0 \
      --bootstrap-draws "${BOOTSTRAP_DRAWS}" \
      --method-label "${METHOD_LABEL}"
  fi
fi

echo "[checkpoint] ${CHECKPOINT}"
if [[ -s "${TEST_JSON}" ]]; then
  python - "${TEST_JSON}" <<'PY'
import json
import sys

path = sys.argv[1]
result = json.load(open(path, encoding="utf-8"))
metric = result["primary_metric"]
print(f"[test-json] {path}")
print(
    f"[result] {metric['name']}={metric['value']:.6f} "
    f"horizons={metric['horizons']} lower_is_better={metric['lower_is_better']}"
)
PY
fi
