#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh" "${1:-render}"

# Figures 1 and 2 are explanatory/vector figures and require no checkpoint.
"${PYTHON}" scripts/plotting/plot_intro_gradient_controls.py
"${PYTHON}" scripts/plotting/plot_internal_dw_method.py
echo "[done] figs/intro_gradient_controls.pdf"
echo "[done] figs/internal_dw_method.pdf"

