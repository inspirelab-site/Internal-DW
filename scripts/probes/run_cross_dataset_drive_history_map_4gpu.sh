#!/usr/bin/env bash
set -euo pipefail

# Four independent single-GPU lanes.  These are model-free diagnostics; the
# checkpoint paths are read only for their dataset/split configuration.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

GPUS_CSV=${GPUS:-0,1,2,3}
IFS=',' read -r -a GPU_LIST <<< "${GPUS_CSV}"
if [[ ${#GPU_LIST[@]} -ne 4 ]]; then
  echo "GPUS must contain four comma-separated physical GPU ids" >&2
  exit 2
fi

OUT_ROOT=${OUT_ROOT:-probe_outputs/cross_dataset_drive_history_map_v1}
LOG_ROOT=${LOG_ROOT:-logs/cross_dataset_drive_history_map_v1}
HCP_DATA=${HCP_DATA:-data/hcp_movie_features}
FMRI_SOURCE=${FMRI_SOURCE:-probe_outputs/drive_history_decomposition_v1/fmri_drive_history.json}
mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

run_generic() {
  local gpu=$1 label=$2 checkpoint=$3 horizons=$4 windows=$5
  shift 5
  local output="${OUT_ROOT}/${label}.json"
  if [[ -s "${output}" && "${FORCE:-0}" != "1" ]]; then
    echo "[reuse] ${output}"
    return 0
  fi
  echo "[start] ${label} GPU=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u scripts/probes/probe_cross_dataset_drive_history.py \
    --checkpoint "${checkpoint}" \
    --output "${output}" \
    --label "${label}" \
    --horizons "${horizons}" \
    --history-windows "${windows}" \
    --long-horizon-min 8 \
    --device cuda \
    "$@" \
    2>&1 | tee "${LOG_ROOT}/${label}.log"
}

lane0() {
  run_generic "${GPU_LIST[0]}" mg \
    experiments/dual_wiener_screen/mackey_glass/tau30_K32/dualwiener/seed0/best.pth \
    1,2,4,8,16,24,32 1,2,4,8,16 \
    --max-coordinates 64 --pca-rank 8 --pca-frames 2048
  run_generic "${GPU_LIST[0]}" narma \
    experiments/dual_wiener_screen/narma/L5_K32/dualwiener/seed0/best.pth \
    1,2,4,8,16,24,32 1,2,4,8 \
    --max-coordinates 64 --pca-rank 8 --pca-frames 2048

  # Preserve the already-completed fMRI history experiment.  Adding the map's
  # drive coordinate only reloads the subject data and performs no predictor fit.
  local fmri_out="${OUT_ROOT}/fmri.json"
  local need_fmri=1
  if [[ -s "${fmri_out}" && "${FORCE:-0}" != "1" ]]; then
    if python - "${fmri_out}" <<'PY'
import json, sys
raise SystemExit(0 if "drive_value" in json.load(open(sys.argv[1])) else 1)
PY
    then need_fmri=0; fi
  fi
  if [[ ${need_fmri} -eq 1 ]]; then
    if [[ ! -s "${FMRI_SOURCE}" ]]; then
      echo "missing completed fMRI source JSON: ${FMRI_SOURCE}" >&2
      exit 3
    fi
    echo "[enrich] fmri shared-response drive coordinate"
    python -u scripts/data/enrich_fmri_drive_history_map.py \
      --input "${FMRI_SOURCE}" --output "${fmri_out}" --data-path "${HCP_DATA}" \
      2>&1 | tee "${LOG_ROOT}/fmri.log"
  else
    echo "[reuse] ${fmri_out}"
  fi
}

lane1() {
  GPU="${GPU_LIST[1]}" bash scripts/reproduce/train_test_ieeg.sh regime
}

lane2() {
  run_generic "${GPU_LIST[2]}" shear \
    experiments/thewell_shear_final_lr3e4/shear_flow/unet_b32_D4_W2_K32_ds4/dualwiener_spectral/seed0/best.pth \
    1,2,4,8,16,24,32 1,2,4,8 \
    --max-coordinates 4096 --pca-rank 32 --pca-frames 2048 \
    --max-train-sequences 64 --max-val-sequences 32 --max-test-sequences 32 \
    --fit-samples 2048 --val-samples 1024 --test-samples 1024 --knn-chunk 16
}

lane3() {
  run_generic "${GPU_LIST[3]}" wb2 \
    experiments/weatherbench2/assigned_spectral_ddp4_v1/unet_c32_D4_W2_K48/dualwiener_spectral/seed0/best.pth \
    1,2,4,8,16,24,32,48 1,2,4,8 \
    --max-coordinates 2048 --pca-rank 32 --pca-frames 1024 \
    --max-train-sequences 32 --max-val-sequences 16 --max-test-sequences 16 \
    --fit-samples 1024 --val-samples 512 --test-samples 512 \
    --neighbors 24 --knn-chunk 8
}

lane0 > "${LOG_ROOT}/lane0.log" 2>&1 & p0=$!
lane1 > "${LOG_ROOT}/lane1.log" 2>&1 & p1=$!
lane2 > "${LOG_ROOT}/lane2.log" 2>&1 & p2=$!
lane3 > "${LOG_ROOT}/lane3.log" 2>&1 & p3=$!
echo "[launched] lane0=${p0} lane1=${p1} lane2=${p2} lane3=${p3} GPUS=${GPUS_CSV}"

status=0
for pid in "${p0}" "${p1}" "${p2}" "${p3}"; do
  if ! wait "${pid}"; then status=1; fi
done
if [[ ${status} -ne 0 ]]; then
  echo "[failed] at least one drive--history lane failed" >&2
  exit ${status}
fi

python -u scripts/plotting/plot_cross_dataset_drive_history_map.py \
  --dataset "MG=${OUT_ROOT}/mg.json" \
  --dataset "NARMA=${OUT_ROOT}/narma.json" \
  --dataset "iEEG=${IEEG_PROBE_ROOT:-probe_outputs/ieeg_cohort_v1}/regime/cohort.json" \
  --dataset "Shear=${OUT_ROOT}/shear.json" \
  --dataset "WB2=${OUT_ROOT}/wb2.json" \
  --dataset "ETTm1=probe_outputs/temporal_candidate_regime_v1/ettm1.json" \
  --dataset "ETTm2=probe_outputs/temporal_candidate_regime_v1/ettm2.json" \
  --fmri "${OUT_ROOT}/fmri.json" \
  --long-horizon-min 8 \
  --output "${OUT_ROOT}/long_horizon_map" \
  2>&1 | tee "${LOG_ROOT}/plot.log"

echo "[done] ${OUT_ROOT}"
