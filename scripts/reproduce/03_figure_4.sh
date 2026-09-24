#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh" "${1:-render}"
ROOT=${KNOWN_SNR_ROOT:-experiments/known_snr}
OUT=${OUT:-figs/known_snr_closure}
if [[ "${MODE}" == run ]]; then
  bash scripts/reproduce/train_test_known_snr.sh
fi
"${PYTHON}" scripts/plotting/plot_known_snr_closure.py --closure-root "${ROOT}" --out "${OUT}"
