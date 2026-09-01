#!/usr/bin/env bash
# Stage 1 of the known-SNR closure: gain identification only.
# One frozen Exact-BPTT checkpoint, one physical GPU, no training and no DDP.
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)/src:$(pwd)/scripts:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

GPU=${GPU:-0}
SEED=${SEED:-0}
K=${K:-32}
DRAWS=${DRAWS:-8}
NOISE_DRAWS=${NOISE_DRAWS:-64}
BATCH=${BATCH:-0}
FORCE=${FORCE:-0}

CKPT=${CKPT:-experiments/known_snr_oracle/known_snr_ar/mixed_K32/ckpt/seed0/best.pth}
DATA=${DATA:-data/synthetic/known_snr_ar_D8_T1024_traj96_a5fd79afa1e5f_s0.npz}
OUT_DIR=${OUT_DIR:-probe_outputs/known_snr_diagonal_ar_closure/seed${SEED}}
OUT=${OUT:-${OUT_DIR}/gain_identification.json}

if [[ "${WORLD_SIZE:-1}" != "1" ]]; then
  echo "[error] this frozen route probe is single-process; do not use torchrun/DDP" >&2
  exit 2
fi
if [[ "${GPU}" == *,* || "${GPU}" == *" "* ]]; then
  echo "[error] GPU must be one physical id, got '${GPU}'" >&2
  exit 2
fi
if [[ "${K}" != "32" ]]; then
  echo "[error] the pre-registered closure uses K=32" >&2
  exit 2
fi
for value in "${DRAWS}" "${NOISE_DRAWS}"; do
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[error] draw counts must be positive integers, got '${value}'" >&2
    exit 2
  fi
done
for path in "${CKPT}" "${DATA}"; do
  if [[ ! -f "${path}" ]]; then
    echo "[error] missing required file: ${path}" >&2
    exit 2
  fi
done

mkdir -p "${OUT_DIR}"
if [[ "${FORCE}" != "1" && -s "${OUT}" ]]; then
  echo "[reuse] ${OUT}; set FORCE=1 to recompute"
  exit 0
fi

echo "=== known-SNR diagonal-AR(1) gain identification ==="
echo "[scope] frozen Exact-BPTT checkpoint; train-only AR fit; validation routes; test untouched"
echo "[config] GPU=${GPU} K=${K} starts=${DRAWS} common-white/process=${NOISE_DRAWS}"
echo "[checkpoint] ${CKPT}"

CUDA_VISIBLE_DEVICES="${GPU}" python -u scripts/probes/probe_known_snr_diagonal_ar_gain.py \
  --ckpt "${CKPT}" \
  --exact-checkpoint-adapter \
  --npz "${DATA}" --state-key trajs --data-preprocess none \
  --hidden 128 --depth 4 --K "${K}" --burnin 32 --window 16 \
  --batch "${BATCH}" --draws "${DRAWS}" --noise-draws "${NOISE_DRAWS}" \
  --seed "${SEED}" --device cuda --out "${OUT}"

echo "=== DONE ==="
echo "[out] ${OUT}"

