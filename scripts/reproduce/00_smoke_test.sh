#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh" render

echo "[1/3] import public API"
"${PYTHON}" -c 'from internal_dw import InternalDW, InternalDWResidual; print("Internal-DW API: OK")'

echo "[2/3] run minimal training example"
"${PYTHON}" examples/quickstart_internal_dw.py

echo "[3/3] run public API tests"
"${PYTHON}" -m pytest -q tests/test_public_internal_dw_api.py tests/test_dual_wiener.py

