#!/usr/bin/env bash
# Controlled Driven-MG sweep.  All strengths share initial conditions, forcing
# realizations, and split indices; only the multiplier lambda_D changes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

GPUS_CSV=${GPUS:-0,1,2,3}
IFS=',' read -r -a GPU_LIST <<< "${GPUS_CSV}"
if [[ ${#GPU_LIST[@]} -lt 1 ]]; then
  echo "GPUS must contain at least one physical GPU id" >&2
  exit 2
fi

SCALES_TEXT=${SCALES:-"0.00 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08"}
read -r -a SCALE_LIST <<< "${SCALES_TEXT}"
DATA_ROOT=${DATA_ROOT:-artifacts/driven_mg_strength_sweep_v1/data}
OUT_ROOT=${OUT_ROOT:-probe_outputs/driven_mg_strength_sweep_v1}
LOG_ROOT=${LOG_ROOT:-logs/driven_mg_strength_sweep_v1}
FORCE=${FORCE:-0}
mkdir -p "${DATA_ROOT}" "${OUT_ROOT}/points" "${LOG_ROOT}"

json_ok() {
  python - "$1" <<'PY'
import json, sys
try:
    payload = json.load(open(sys.argv[1]))
    if not all(
        "mean_long_history_shapley_beyond_null" in payload.get("shapley", {}).get(name, {})
        for name in ("linear", "nonlinear")
    ):
        raise ValueError("missing common-estimand Shapley coalitions")
except Exception:
    raise SystemExit(1)
PY
}

npz_ok() {
  python - "$1" <<'PY'
import numpy as np, sys
try:
    with np.load(sys.argv[1], allow_pickle=False) as z:
        required = {f"{s}_{q}" for s in ("train", "validation", "test") for q in ("state", "drive")}
        if not required.issubset(z.files): raise ValueError("missing arrays")
except Exception:
    raise SystemExit(1)
PY
}

run_point() {
  local gpu=$1 scale=$2 tag data result log
  tag=${scale/./p}
  data="${DATA_ROOT}/mg_ds${tag}.npz"
  result="${OUT_ROOT}/points/ds${tag}.json"
  log="${LOG_ROOT}/ds${tag}.log"
  {
    echo "[point] lambda_D=${scale} GPU=${gpu} $(date -Is)"
    if [[ "${FORCE}" == "1" ]] || ! npz_ok "${data}"; then
      python -u scripts/data/generate_driven_mackey_glass.py \
        --output "${data}" \
        --dim 8 --tau 30 --dt 1 --solver-dt 0.1 \
        --length 2048 --trajectories 40 --transient 1000 \
        --beta 0.2 --gamma 0.1 --n-exp 10 \
        --drive-scale "${scale}" --drive-rho 0.9 \
        --generator-seed 2027 --split-seed 0
    else
      echo "[reuse data] ${data}"
    fi
    if [[ "${FORCE}" == "1" ]] || ! json_ok "${result}"; then
      CUDA_VISIBLE_DEVICES="${gpu}" python -u scripts/probes/probe_cross_dataset_drive_history.py \
        --input-npz "${data}" \
        --input-dataset-name mackey_glass_driven \
        --output "${result}" \
        --label "Driven-MG-${scale}" \
        --horizons 1,2,4,8,16,32 \
        --history-windows 1,2,4,8,16 \
        --long-horizon-min 8 \
        --max-coordinates 64 --pca-rank 8 --pca-frames 2048 \
        --max-train-sequences 64 --max-val-sequences 32 --max-test-sequences 32 \
        --fit-samples 2048 --val-samples 1024 --test-samples 1024 \
        --null-repeats 8 --bootstraps 1000 --seed 0 \
        --screen-on-validation --device cuda
    else
      echo "[reuse result] ${result}"
    fi
    echo "[done point] lambda_D=${scale} $(date -Is)"
  } 2>&1 | tee "${log}"
}

lane() {
  local lane_index=$1 gpu=$2 index
  for ((index=lane_index; index<${#SCALE_LIST[@]}; index+=${#GPU_LIST[@]})); do
    run_point "${gpu}" "${SCALE_LIST[index]}"
  done
}

pids=()
for ((lane_index=0; lane_index<${#GPU_LIST[@]}; lane_index++)); do
  lane "${lane_index}" "${GPU_LIST[lane_index]}" &
  pids+=("$!")
done
echo "[launched] lanes=${pids[*]} GPUS=${GPUS_CSV} scales=${SCALES_TEXT}"

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if [[ ${status} -ne 0 ]]; then
  echo "[failed] at least one Driven-MG strength point failed" >&2
  exit ${status}
fi

scales_csv=${SCALES_TEXT// /,}
python -u scripts/probes/probe_driven_mg_simulator_ground_truth.py \
  --scales "${scales_csv}" \
  --output "${OUT_ROOT}/simulator_ground_truth.json" \
  2>&1 | tee "${LOG_ROOT}/simulator_ground_truth.log"

python -u scripts/plotting/plot_driven_mg_strength_sweep.py \
  --data-root "${DATA_ROOT}" \
  --result-root "${OUT_ROOT}/points" \
  --scales "${scales_csv}" \
  --ground-truth "${OUT_ROOT}/simulator_ground_truth.json" \
  --output "${OUT_ROOT}/drive_history_vs_strength" \
  2>&1 | tee "${LOG_ROOT}/plot.log"

echo "[complete] ${OUT_ROOT} $(date -Is)"
