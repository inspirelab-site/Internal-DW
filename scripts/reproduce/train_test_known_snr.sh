#!/usr/bin/env bash
# End-to-end, resumable one-GPU reproduction of the known-SNR closure.
set -euo pipefail

REPO_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd -P)}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/scripts:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# shellcheck source=/dev/null
source "${REPO_ROOT}/configs/reproduce/known_snr.sh"

GPU=${GPU:-0}
if [[ "${GPU}" == *,* || "${GPU}" == *" "* ]]; then
  echo "GPU must name the one physical device used by Full BPTT and frozen probes." >&2
  exit 2
fi
GPUS=${GPUS:-${GPU}}
IFS=',' read -r -a KNOWN_SNR_ROUTING_GPU_IDS <<< "${GPUS}"
KNOWN_SNR_ROUTING_WORLD_SIZE=${#KNOWN_SNR_ROUTING_GPU_IDS[@]}
if (( KNOWN_SNR_ROUTING_WORLD_SIZE != 1 \
   && KNOWN_SNR_ROUTING_WORLD_SIZE != 2 \
   && KNOWN_SNR_ROUTING_WORLD_SIZE != 4 )); then
  echo "GPUS must contain one, two, or four physical device ids." >&2
  exit 2
fi
KNOWN_SNR_ROUTING_DENOMINATOR=$((
  KNOWN_SNR_ROUTING_LOCAL_BATCH * KNOWN_SNR_ROUTING_WORLD_SIZE
))
if (( KNOWN_SNR_EFFECTIVE_BATCH % KNOWN_SNR_ROUTING_DENOMINATOR != 0 )); then
  echo "cannot preserve known-SNR effective batch ${KNOWN_SNR_EFFECTIVE_BATCH}" >&2
  exit 2
fi
KNOWN_SNR_ROUTING_GRAD_ACCUM=$((
  KNOWN_SNR_EFFECTIVE_BATCH / KNOWN_SNR_ROUTING_DENOMINATOR
))
echo "[protocol] Full BPTT: world=1 local_batch=${KNOWN_SNR_EXACT_LOCAL_BATCH} accum=${KNOWN_SNR_EXACT_GRAD_ACCUM_STEPS}"
echo "[protocol] routed arms: world=${KNOWN_SNR_ROUTING_WORLD_SIZE} local_batch=${KNOWN_SNR_ROUTING_LOCAL_BATCH} accum=${KNOWN_SNR_ROUTING_GRAD_ACCUM}"
if (( KNOWN_SNR_ROUTING_WORLD_SIZE != KNOWN_SNR_CANONICAL_ROUTING_WORLD_SIZE )); then
  echo "[protocol] hardware-adapted routing run; optimizer effective batch remains ${KNOWN_SNR_EFFECTIVE_BATCH}"
fi

KNOWN_SNR_DATA=${KNOWN_SNR_DATA:-data/synthetic/known_snr_ar_D8_T1024_traj96_a5fd79afa1e5f_s0.npz}
KNOWN_SNR_DATA_DIR=${KNOWN_SNR_DATA_DIR:-$(dirname "${KNOWN_SNR_DATA}")}
KNOWN_SNR_EXACT_BASE=${KNOWN_SNR_EXACT_BASE:-experiments/known_snr_oracle}
KNOWN_SNR_EXACT_RUN=${KNOWN_SNR_EXACT_RUN:-${KNOWN_SNR_EXACT_BASE}/known_snr_ar/mixed_K32/ckpt/seed0}
KNOWN_SNR_ORACLE_FILE=${KNOWN_SNR_ORACLE_FILE:-artifacts/known_snr_ar/oracle_K32.npz}
KNOWN_SNR_DW_BASE=${KNOWN_SNR_DW_BASE:-experiments/known_snr_diagonal_ar_closure}
KNOWN_SNR_DW_RUN=${KNOWN_SNR_DW_RUN:-${KNOWN_SNR_DW_BASE}/known_snr_ar/mixed_K32/dualwiener_diag_ar/seed0}
KNOWN_SNR_CONTROL_BASE=${KNOWN_SNR_CONTROL_BASE:-experiments/known_snr_diagonal_ar_controls_v1}
KNOWN_SNR_ROOT=${KNOWN_SNR_ROOT:-probe_outputs/known_snr_diagonal_ar_closure/seed0}
KNOWN_SNR_FIGURE=${KNOWN_SNR_FIGURE:-figs/known_snr_closure}

echo "[known-SNR] stage 1/7: train/test the frozen Full-BPTT checkpoint"
GPU="${GPU}" GPUS="${GPU}" \
  DATASET=known_snr_ar COND=mixed K="${KNOWN_SNR_TRAIN_HORIZON}" \
  METHOD=ckpt SEED="${KNOWN_SNR_SEED}" \
  BATCH="${KNOWN_SNR_EXACT_LOCAL_BATCH}" \
  GRAD_ACCUM="${KNOWN_SNR_EXACT_GRAD_ACCUM_STEPS}" \
  EPOCHS="${KNOWN_SNR_EPOCHS}" ES="${KNOWN_SNR_EARLY_STOP_PATIENCE}" \
  LR="${KNOWN_SNR_LEARNING_RATE}" WEIGHT_DECAY="${KNOWN_SNR_WEIGHT_DECAY}" \
  GRAD_CLIP="${KNOWN_SNR_GRAD_CLIP}" NUM_WORKERS="${KNOWN_SNR_NUM_WORKERS}" \
  MAMBA_LOSS_TYPE=mse MAMBA_TRAIN_STARTS=1 \
  DATA_DIR="${KNOWN_SNR_DATA_DIR}" \
  SAVE_BASE="${KNOWN_SNR_EXACT_BASE}" SKIP_EXISTING=1 RESUME=auto \
    bash scripts/train/run_mem_one.sh

for required in "${KNOWN_SNR_DATA}" \
                "${KNOWN_SNR_EXACT_RUN}/best.pth" \
                "${KNOWN_SNR_EXACT_RUN}/eval_results.json"; do
  [[ -s "${required}" ]] || { echo "[missing] ${required}" >&2; exit 3; }
done

echo "[known-SNR] stage 2/7: construct the analytic A,Q oracle"
python scripts/data/prepare_known_snr_oracle.py \
  --data "${KNOWN_SNR_DATA}" --output "${KNOWN_SNR_ORACLE_FILE}" \
  --max-horizon "${KNOWN_SNR_TRAIN_HORIZON}"

echo "[known-SNR] stage 3/7: gradient magnitude/SNR and prefix trade-off"
GPUS="${GPUS}" SEED="${KNOWN_SNR_SEED}" \
  REPETITIONS="${KNOWN_SNR_PANEL_REPETITIONS}" \
  EVAL_NOISE_DRAWS="${KNOWN_SNR_FUTURE_NOISE_DRAWS}" \
  CKPT="${KNOWN_SNR_EXACT_RUN}/best.pth" DATA="${KNOWN_SNR_DATA}" \
  ORACLE_FILE="${KNOWN_SNR_ORACLE_FILE}" ROOT="${KNOWN_SNR_ROOT}" \
    bash scripts/probes/run_known_snr_exact_panels12.sh

echo "[known-SNR] stage 4/7: train-only gain identification"
GPU="${GPU}" SEED="${KNOWN_SNR_SEED}" \
  DRAWS="${KNOWN_SNR_PROBE_STARTS}" NOISE_DRAWS="${KNOWN_SNR_FUTURE_NOISE_DRAWS}" \
  CKPT="${KNOWN_SNR_EXACT_RUN}/best.pth" DATA="${KNOWN_SNR_DATA}" \
  OUT_DIR="${KNOWN_SNR_ROOT}" \
    bash scripts/probes/run_known_snr_diagonal_ar_gain.sh

echo "[known-SNR] stage 5/7: held-out route-gradient risk"
GPU="${GPU}" SEED="${KNOWN_SNR_SEED}" \
  DRAWS="${KNOWN_SNR_PROBE_STARTS}" NOISE_DRAWS="${KNOWN_SNR_FUTURE_NOISE_DRAWS}" \
  CKPT="${KNOWN_SNR_EXACT_RUN}/best.pth" DATA="${KNOWN_SNR_DATA}" \
  OUT_DIR="${KNOWN_SNR_ROOT}" \
    bash scripts/probes/run_known_snr_diagonal_ar_route_risk.sh

echo "[known-SNR] stage 6/7: matched Internal-DW forecasting"
GPU="${GPU}" GPUS="${GPUS}" SEED="${KNOWN_SNR_SEED}" \
  BATCH="${KNOWN_SNR_ROUTING_LOCAL_BATCH}" \
  GRAD_ACCUM="${KNOWN_SNR_ROUTING_GRAD_ACCUM}" \
  DATA="${KNOWN_SNR_DATA}" EXACT_RUN="${KNOWN_SNR_EXACT_RUN}" \
  SAVE_BASE="${KNOWN_SNR_DW_BASE}" DW_RUN="${KNOWN_SNR_DW_RUN}" \
  ROOT="${KNOWN_SNR_ROOT}" \
    bash scripts/train/run_known_snr_diagonal_ar_forecasting.sh

echo "[known-SNR] stage 7/7: forecasting controls and final figure"
GPU="${GPU}" GPUS="${GPUS}" SEED="${KNOWN_SNR_SEED}" \
  BATCH="${KNOWN_SNR_ROUTING_LOCAL_BATCH}" \
  GRAD_ACCUM="${KNOWN_SNR_ROUTING_GRAD_ACCUM}" \
  EXACT_RUN="${KNOWN_SNR_EXACT_RUN}" DW_RUN="${KNOWN_SNR_DW_RUN}" \
  SAVE_ROOT="${KNOWN_SNR_CONTROL_BASE}" OUT_ROOT="${KNOWN_SNR_ROOT}" \
    bash scripts/train/run_known_snr_diagonal_ar_controls_ddp4.sh

ROOT="${KNOWN_SNR_ROOT}" OUT="${KNOWN_SNR_FIGURE}" \
  bash scripts/reproduce/03_figure_4.sh render

echo "[checkpoint:full-bptt] ${KNOWN_SNR_EXACT_RUN}/best.pth"
echo "[checkpoint:internal-dw] ${KNOWN_SNR_DW_RUN}/best.pth"
echo "[result-ledger] ${KNOWN_SNR_ROOT}/closure_summary.json"
echo "[figure] ${KNOWN_SNR_FIGURE}.pdf"
