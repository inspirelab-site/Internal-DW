#!/usr/bin/env bash
# Canonical Figure 6 training horizons and validation-selected controls.

FIGURE6_POSITIVE_DATASETS=(mg ettm1 ettm2 shear)
FIGURE6_BOUNDARY_DATASETS=(narma ieeg fmri wb2)
FIGURE6_POSITIVE_ARMS=(full_bptt internal_dw clip jreg tbptt static)
FIGURE6_BOUNDARY_ARMS=(full_bptt internal_dw clip jreg)

# Training workers recorded by the reference runs (not evaluation workers).
figure6_num_workers() {
  local data=$1 arm=$2 seed=$3
  case "${data}" in
    mg|narma|fmri) echo 4 ;;
    ettm1|ettm2|ieeg) echo 0 ;;
    shear)
      case "${arm}" in clip|jreg) echo 2 ;; *) echo 0 ;; esac ;;
    wb2)
      case "${arm}" in
        clip|jreg) echo 2 ;;
        exact|full_bptt) if [[ "${seed}" == 0 ]]; then echo 2; else echo 0; fi ;;
        dw|internal_dw) echo 0 ;;
        *) echo "unsupported WB2 arm: ${arm}" >&2; return 2 ;;
      esac ;;
    *) echo "unknown dataset: ${data}" >&2; return 2 ;;
  esac
}

declare -A FIGURE6_TRAIN_HORIZON=(
  [mg]=32 [ettm1]=64 [ettm2]=64 [shear]=32
  [narma]=32 [ieeg]=64 [fmri]=64 [wb2]=48
)

# Memory-safe one-GPU geometry. The product local_batch x accumulation retains
# the optimizer-level effective batch used by the corresponding paper run.
declare -A FIGURE6_SINGLE_LOCAL_BATCH=(
  [mg]=4 [ettm1]=32 [ettm2]=32 [shear]=1
  [narma]=4 [ieeg]=4 [fmri]=2 [wb2]=1
)
declare -A FIGURE6_SINGLE_GRAD_ACCUM=(
  [mg]=8 [ettm1]=1 [ettm2]=1 [shear]=4
  [narma]=8 [ieeg]=8 [fmri]=16 [wb2]=8
)

# Selected once from seed-0 validation; test results were not read by either
# selector. These values are fixed when reproducing the reported benchmark.
declare -A FIGURE6_CLIP_NORM=(
  [mg]=1.0 [ettm1]=0.3 [ettm2]=0.3 [shear]=0.3
  [narma]=0.1 [ieeg]=0.3 [fmri]=1.0 [wb2]=1.0
)
declare -A FIGURE6_JREG_LAMBDA=(
  [mg]=1.0 [ettm1]=1.0 [ettm2]=1.0 [shear]=0.01
  [narma]=1.0 [ieeg]=0.01 [fmri]=1.0 [wb2]=0.01
)

declare -A FIGURE6_STATIC_GAIN=(
  [mg]=0.6 [ettm1]=0.3 [ettm2]=0.6 [shear]=0.3
)
declare -A FIGURE6_TBPTT_SEGMENT=(
  [mg]=8 [ettm1]=16 [ettm2]=32 [shear]=8
)
