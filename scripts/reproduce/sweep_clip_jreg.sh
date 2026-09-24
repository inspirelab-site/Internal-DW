#!/usr/bin/env bash
# Independent single-GPU jobs; unchanged per-run batch/accumulation.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
exec "${PYTHON:-python}" -u scripts/reproduce/sweep_clip_jreg.py \
  --gpus "${GPUS:-0,1,2,3}" \
  --jobs-per-gpu "${JOBS_PER_GPU:-1}" \
  --root "${SWEEP_ROOT:-experiments/clip_jreg_val_sweep_v1}" "$@"
