#!/usr/bin/env bash
# Stage 2 of the known-SNR closure: held-out route risk for frozen Stage-1 gains.
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
GAIN_JSON=${GAIN_JSON:-${OUT_DIR}/gain_identification.json}
GAIN_NPZ=${GAIN_NPZ:-${OUT_DIR}/gain_identification.npz}
OUT=${OUT:-${OUT_DIR}/route_risk.json}

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
for path in "${CKPT}" "${DATA}" "${GAIN_JSON}" "${GAIN_NPZ}"; do
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

echo "=== known-SNR diagonal-AR(1) held-out route risk ==="
echo "[scope] fixed Stage-1 gains; untouched test trajectories; no training"
echo "[config] GPU=${GPU} K=${K} test-starts=${DRAWS} true-process-draws=${NOISE_DRAWS}"

CUDA_VISIBLE_DEVICES="${GPU}" python -u scripts/probes/probe_known_snr_diagonal_ar_route_risk.py \
  --ckpt "${CKPT}" \
  --exact-checkpoint-adapter \
  --npz "${DATA}" --state-key trajs --data-preprocess none \
  --gain-json "${GAIN_JSON}" --gain-npz "${GAIN_NPZ}" \
  --hidden 128 --depth 4 --K "${K}" --burnin 32 --window 16 \
  --batch "${BATCH}" --draws "${DRAWS}" --noise-draws "${NOISE_DRAWS}" \
  --seed "${SEED}" --device cuda --out "${OUT}"

python scripts/results/build_known_snr_results.py closure \
  --root "${OUT_DIR}" --quiet

echo "=== DONE ==="
echo "[out] ${OUT}"
