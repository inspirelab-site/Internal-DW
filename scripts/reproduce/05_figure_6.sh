#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh" "${1:-render}"

if [[ "${MODE}" == run ]]; then
  cat >&2 <<'EOF'
Figure 6 is the full eight-dataset benchmark.  Train the requested matched
seeds with the dataset runners documented in README.md, then rerun this script
in render mode.  The runner deliberately does not guess cluster topology or
download licensed HCP/WeatherBench data.
EOF
  exit 2
fi

"${PYTHON}" scripts/results/export_figure6_raw_metrics.py
"${PYTHON}" scripts/plotting/plot_forecasting_controls_hierarchical_preview.py
echo "[done] probe_outputs/figure6_raw_metrics_v1/"
echo "[done] figs/forecasting_controls_hierarchical_preview.pdf"

