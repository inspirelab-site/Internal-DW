#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "${REPO_ROOT}"

PYTHON=${PYTHON:-python}
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/scripts/probes:${REPO_ROOT}/scripts/plotting:${REPO_ROOT}/scripts/data:${REPO_ROOT}/scripts/evaluate:${REPO_ROOT}/scripts/results:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

MODE=${1:-render}
case "${MODE}" in
  render|run) ;;
  *) echo "usage: $0 [render|run]" >&2; exit 2 ;;
esac

require_file() {
  [[ -s "$1" ]] || {
    echo "[missing] $1" >&2
    echo "Run '$0 run' or download the released result bundle." >&2
    exit 3
  }
}

run_if_requested() {
  if [[ "${MODE}" == run ]]; then
    "$@"
  fi
}
