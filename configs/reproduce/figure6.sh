#!/usr/bin/env bash
# Canonical Figure 6 training horizons and validation-selected controls.

FIGURE6_POSITIVE_DATASETS=(mg ettm1 ettm2 shear)
FIGURE6_BOUNDARY_DATASETS=(narma ieeg fmri wb2)
FIGURE6_POSITIVE_ARMS=(full_bptt internal_dw clip jreg tbptt static)
FIGURE6_BOUNDARY_ARMS=(full_bptt internal_dw clip jreg)

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
declare -A FIGURE6_STATIC_GAIN=(
  [mg]=0.6 [ettm1]=0.3 [ettm2]=0.6 [shear]=0.3
)
declare -A FIGURE6_TBPTT_SEGMENT=(
  [mg]=8 [ettm1]=16 [ettm2]=32 [shear]=8
)
