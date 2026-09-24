#!/usr/bin/env bash
# Sequential, single-GPU Known-SNR reproduction.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH:-}"
mode=${1:-all}
if (( $# )); then shift; fi
exec "${PYTHON:-python}" -u scripts/reproduce/train_test_known_snr.py "${mode}" \
  --gpu "${GPU:-0}" --root "${KNOWN_SNR_ROOT:-experiments/known_snr}" \
  --data "${KNOWN_SNR_DATA_DIR:-data/known_snr}" "$@"
