#!/usr/bin/env bash
# Four independent single-GPU workers; unchanged per-run batch/accumulation.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
exec "${PYTHON:-python}" -u scripts/reproduce/sweep_clip_jreg.py \
  --gpus "${GPUS:-0,1,2,3}" \
  --root "${SWEEP_ROOT:-experiments/clip_jreg_val_sweep_v1}" "$@"
