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
OUT_ROOT=${OUT_ROOT:-probe_outputs/dense_multistart_rel_l2_1p5k_v1}

case "${ACTION}" in
  train|test|all) ;;
  *) echo "usage: ARM=<full_bptt|internal_dw> $0 [train|test|all]" >&2; exit 2 ;;
esac

case "${ARM}" in
  full_bptt)
    DEFAULT_CONFIG_FILE="${REPO_ROOT}/configs/reproduce/mg_full_bptt.sh"
    METHOD=ckpt
    TAG=ckpt
    METHOD_LABEL=full_bptt
    DEFAULT_SAVE_BASE=experiments/memtest
    RESULT_ARM=exact
    ;;
  internal_dw)
    DEFAULT_CONFIG_FILE="${REPO_ROOT}/configs/reproduce/mg_internal_dw.sh"
    METHOD=dwstructured
    TAG=dualwiener_structured
    METHOD_LABEL=internal_dw
    DEFAULT_SAVE_BASE=experiments/internal_dw_assigned_v1
    RESULT_ARM=dw
    ;;
  *) echo "ARM must be full_bptt or internal_dw; got ${ARM}" >&2; exit 2 ;;
esac

CONFIG_FILE=${CONFIG_FILE:-${DEFAULT_CONFIG_FILE}}
[[ -r "${CONFIG_FILE}" ]] || {
  echo "reproduction config not found: ${CONFIG_FILE}" >&2
  exit 2
}
# shellcheck source=/dev/null
source "${CONFIG_FILE}"

GPUS=${GPUS:-${GPU:-0}}
GPU=${GPU:-${GPUS%%,*}}
IFS=',' read -r -a REPRO_GPU_IDS <<< "${GPUS}"
REPRO_WORLD_SIZE=${#REPRO_GPU_IDS[@]}

# Preserve the paper's global microbatch and number of DW calibrations per
# optimizer update when moving between one GPU and DDP. Explicit BATCH or
# GRAD_ACCUM values remain available as intentional reader overrides.
if [[ -z "${BATCH:-}" ]]; then
  if (( PAPER_GLOBAL_MICROBATCH % REPRO_WORLD_SIZE != 0 )); then
    echo "paper global microbatch ${PAPER_GLOBAL_MICROBATCH} is not divisible by ${REPRO_WORLD_SIZE} GPUs; set BATCH explicitly" >&2
    exit 2
  fi
  BATCH=$((PAPER_GLOBAL_MICROBATCH / REPRO_WORLD_SIZE))
fi
GRAD_ACCUM=${GRAD_ACCUM:-${PAPER_GRAD_ACCUM_STEPS}}
NUM_WORKERS=${NUM_WORKERS:-${PAPER_NUM_WORKERS}}
EPOCHS=${EPOCHS:-${PAPER_EPOCHS}}
ES=${ES:-${PAPER_EARLY_STOP_PATIENCE}}
LR=${LR:-${PAPER_LEARNING_RATE}}
AR_OPTIMIZER=${AR_OPTIMIZER:-${PAPER_OPTIMIZER}}
AR_SGD_MOMENTUM=${AR_SGD_MOMENTUM:-${PAPER_SGD_MOMENTUM}}
WEIGHT_DECAY=${WEIGHT_DECAY:-${PAPER_WEIGHT_DECAY}}
GRAD_CLIP=${GRAD_CLIP:-${PAPER_GRAD_CLIP}}
AR_SCHEDULER=${AR_SCHEDULER:-${PAPER_SCHEDULER}}
MAX_ORIGINS=${MAX_ORIGINS:-${PAPER_MAX_ORIGINS_PER_ITEM}}
ORIGIN_BATCH=${ORIGIN_BATCH:-${PAPER_ORIGIN_BATCH}}
BOOTSTRAP_DRAWS=${BOOTSTRAP_DRAWS:-${PAPER_BOOTSTRAP_DRAWS}}

REPRO_GLOBAL_MICROBATCH=$((REPRO_WORLD_SIZE * BATCH))
REPRO_EFFECTIVE_BATCH=$((REPRO_GLOBAL_MICROBATCH * GRAD_ACCUM))
echo "[paper-config] ${CONFIG_FILE#${REPO_ROOT}/}"
echo "[paper-config] world=${REPRO_WORLD_SIZE} local_batch=${BATCH} accum=${GRAD_ACCUM} effective_batch=${REPRO_EFFECTIVE_BATCH}"
if (( REPRO_GLOBAL_MICROBATCH != PAPER_GLOBAL_MICROBATCH || GRAD_ACCUM != PAPER_GRAD_ACCUM_STEPS )); then
  echo "[config override] this run differs from the canonical paper microbatch schedule"
fi

SAVE_BASE=${SAVE_BASE:-${DEFAULT_SAVE_BASE}}

RUN_DIR="${SAVE_BASE}/${PAPER_DATASET}/${PAPER_CONDITION}_K${PAPER_TRAIN_HORIZON}/${TAG}/seed${SEED}"
CHECKPOINT="${RUN_DIR}/best.pth"
TEST_JSON="${OUT_ROOT}/mg/${RESULT_ARM}_seed${SEED}.json"

if [[ "${ACTION}" == train || "${ACTION}" == all ]]; then
  RUN_MODE=train \
  DATASET="${PAPER_DATASET}" COND="${PAPER_CONDITION}" \
  K="${PAPER_TRAIN_HORIZON}" METHOD="${METHOD}" \
  SEED="${SEED}" GPU="${GPU}" GPUS="${GPUS}" SAVE_BASE="${SAVE_BASE}" \
  BATCH="${BATCH}" GRAD_ACCUM="${GRAD_ACCUM}" NUM_WORKERS="${NUM_WORKERS}" \
  EPOCHS="${EPOCHS}" ES="${ES}" LR="${LR}" \
  AR_OPTIMIZER="${AR_OPTIMIZER}" AR_SGD_MOMENTUM="${AR_SGD_MOMENTUM}" \
  WEIGHT_DECAY="${WEIGHT_DECAY}" GRAD_CLIP="${GRAD_CLIP}" \
  AR_SCHEDULER="${AR_SCHEDULER}" \
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
      --max-horizon "${PAPER_EVAL_HORIZON}" \
      --train-horizon "${PAPER_TRAIN_HORIZON}" \
      --origin-stride 1 \
      --max-origins-per-item "${MAX_ORIGINS}" \
      --origin-batch "${ORIGIN_BATCH}" \
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
