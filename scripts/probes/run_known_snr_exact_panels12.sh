#!/usr/bin/env bash
# Strict rerun of known-SNR Figure panels 1--2 on the frozen Exact checkpoint.
set -euo pipefail
trap '' HUP

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)/src:$(pwd)/scripts:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

IFS=',' read -r -a GPU_IDS <<< "${GPUS:-0}"
if (( ${#GPU_IDS[@]} < 1 || ${#GPU_IDS[@]} > 4 )); then
  echo "[error] GPUS must contain one to four physical GPU ids" >&2
  exit 2
fi

SEED=${SEED:-0}
REPETITIONS=${REPETITIONS:-4}
EVAL_NOISE_DRAWS=${EVAL_NOISE_DRAWS:-64}
BATCH=${BATCH:-0}
FORCE=${FORCE:-0}
CKPT=${CKPT:?set CKPT to the frozen Full-BPTT checkpoint}
DATA=${DATA:?set DATA to the Known-SNR NPZ archive}
ORACLE_FILE=${ORACLE_FILE:?set ORACLE_FILE to the prepared analytic reference}
ROOT=${ROOT:-experiments/known_snr/profiles}
REPLICATE_DIR=${REPLICATE_DIR:-${ROOT}/panels_1_2_replicates}
OUT=${OUT:-${ROOT}/panels_1_2.json}
LOG_DIR=${LOG_DIR:-${ROOT}/logs}

if ! [[ "${REPETITIONS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] REPETITIONS must be positive" >&2
  exit 2
fi
for path in "${CKPT}" "${DATA}" "${ORACLE_FILE}"; do
  if [[ ! -f "${path}" ]]; then
    echo "[error] missing required file: ${path}" >&2
    exit 2
  fi
done
mkdir -p "${REPLICATE_DIR}" "${LOG_DIR}"

run_one() {
  local replicate=$1 gpu=$2
  local result="${REPLICATE_DIR}/replicate${replicate}.json"
  local log="${LOG_DIR}/replicate${replicate}.log"
  if [[ "${FORCE}" != "1" && -s "${result}" ]]; then
    echo "[reuse] replicate ${replicate}: ${result}"
    return 0
  fi
  echo "[launch] panel12 replicate=${replicate} GPU=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u scripts/probes/probe_known_snr_failure_profile.py \
    --ckpt "${CKPT}" --exact-checkpoint-adapter \
    --npz "${DATA}" --state-key trajs --data-preprocess none \
    --oracle-file "${ORACLE_FILE}" \
    --trajectory-split test --split-seed "${SEED}" \
    --checkpoint-seed "${SEED}" --strict-exact-closure \
    --hidden 128 --depth 4 --K 32 --burnin 32 --window 16 \
    --batch "${BATCH}" \
    --eval-noise-draws "${EVAL_NOISE_DRAWS}" \
    --seed "${replicate}" --device cuda --out "${result}" \
    >"${log}" 2>&1
}

status=0
for ((base=0; base<REPETITIONS; base+=${#GPU_IDS[@]})); do
  pids=()
  for ((lane=0; lane<${#GPU_IDS[@]}; lane++)); do
    replicate=$((base + lane))
    if (( replicate >= REPETITIONS )); then
      break
    fi
    run_one "${replicate}" "${GPU_IDS[$lane]}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done
  if (( status != 0 )); then
    echo "[failed] inspect ${LOG_DIR}/replicate*.log" >&2
    exit 1
  fi
done

inputs=()
for ((replicate=0; replicate<REPETITIONS; replicate++)); do
  inputs+=(--input "${REPLICATE_DIR}/replicate${replicate}.json")
done
python scripts/results/build_known_snr_results.py panels12 \
  "${inputs[@]}" --out "${OUT}"

echo "[done] strict Exact-checkpoint panels 1--2"
echo "[out] ${OUT}"
