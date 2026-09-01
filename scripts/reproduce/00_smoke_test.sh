#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh" render

echo "[1/4] validate PyTorch installation"
"${PYTHON}" scripts/utils/check_install.py

echo "[2/4] import public API"
"${PYTHON}" -c 'from internal_dw import InternalDW, InternalDWResidual; print("Internal-DW API: OK")'

echo "[3/4] run minimal training example"
"${PYTHON}" examples/quickstart_internal_dw.py

echo "[4/4] run public API tests"
"${PYTHON}" -m pytest -q tests/test_public_internal_dw_api.py tests/test_dual_wiener.py
