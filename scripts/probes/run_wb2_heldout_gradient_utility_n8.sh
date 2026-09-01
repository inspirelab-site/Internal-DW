#!/usr/bin/env bash
set -euo pipefail

# Re-run only the frozen-checkpoint WB2 delayed-gradient audit with eight
# disjoint train-to-test pairs.  Training is never launched and the original
# n=2 artifact is preserved under its old output root.
cd "$(dirname "$0")/../.."

GPU=${GPU:-0}
OUT_ROOT=${OUT_ROOT:-probe_outputs/real_heldout_gradient_utility_wb2_n8_v1}
LOG_ROOT=${LOG_ROOT:-logs/real_heldout_gradient_utility_wb2_n8_v1}
mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

env GPU="${GPU}" DATASETS=wb2 WB2_PAIRS=8 FINITE_PAIRS=0 \
  COORDS_FIELD=100000 FORCE=1 OUT_ROOT="${OUT_ROOT}" LOG_ROOT="${LOG_ROOT}" \
  bash scripts/probes/run_real_heldout_gradient_utility_single_gpu.sh

