#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd -P)"
for script in 01_figures_1_2.sh 02_figure_3.sh 03_figure_4.sh \
              04_figure_5.sh 05_figure_6.sh 06_figures_7_8.sh \
              07_timing_table.sh; do
  echo "===== ${script} ====="
  bash "${HERE}/${script}" render
done

