#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

# Evaluate one existing checkpoint through 1.5 times its training horizon.
# Required: DATA, ARM, SEED, GPU. Checkpoint mappings remain centralized in
# run_assigned_dense_eval_one.sh.
export EVAL_NUM=3
export EVAL_DEN=2
export OUT_ROOT=${OUT_ROOT:-probe_outputs/dense_multistart_rel_l2_1p5k_v1}
exec bash scripts/evaluate/run_assigned_dense_eval_one.sh
