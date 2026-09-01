#!/usr/bin/env bash
set -euo pipefail

# HCP official_mamba_state with Residual-Gradient Routing.
#
# This script is now a full ablation runner. You can still run the same command:
#
#   bash scripts/train/train_hcp_resgrad_mamba_v2.sh
#
# By default it runs the core HCP ablation set:
#   baseline16 none64 dyn013 dyn014 dyn015
#
# Useful examples:
#   # Core ablation, seed 2, 4 GPUs
#   GPUS=0,1,2,3 BATCH=8 bash scripts/train/train_hcp_resgrad_mamba_v2.sh
#
#   # Best method multi-seed
#   ABLATIONS="dyn013" SEEDS="3 4 5 6" GPUS=0,1,2,3 BATCH=8 bash scripts/train/train_hcp_resgrad_mamba_v2.sh
#
#   # Threshold sweep
#   ABLATIONS="threshold" SEEDS="2" GPUS=0,1,2,3 BATCH=8 bash scripts/train/train_hcp_resgrad_mamba_v2.sh
#
#   # Policy sweep
#   ABLATIONS="policy" SEEDS="2" GPUS=0,1,2,3 BATCH=8 bash scripts/train/train_hcp_resgrad_mamba_v2.sh
#
#   # Memory stress / OOM check, use EPOCHS=1 first
#   ABLATIONS="memory" EPOCHS=1 SEEDS="2" GPUS=0,1,2,3 BATCH=8 bash scripts/train/train_hcp_resgrad_mamba_v2.sh
#
#   # Auto-calibrated dynamic-ratio threshold (targets a fixed branch-open
#   # fraction instead of a fixed threshold, recalibrated every epoch).
#   # Sanity-check with EPOCHS=1 first and watch resgrad_gate_mean converge.
#   ABLATIONS="dyn013" TARGET_OPEN_FRAC=0.25 SEEDS="0 2 3" EPOCHS=1 GPUS=0,1,2,3 BATCH=8 bash scripts/train/train_hcp_resgrad_mamba_v2.sh
#
#   # Gradient-checkpointing baseline: exact full-BPTT gradient at ~O(1)-in-K
#   # memory via activation recomputation, instead of ResGrad's approximate
#   # gradient. Compare its peak memory / wall-clock / final metrics against
#   # baseline64 (same K, no checkpoint) and dyn013 (ResGrad).
#   ABLATIONS="ckpt64" SEEDS="2" GPUS=0,1,2,3 BATCH=8 bash scripts/train/train_hcp_resgrad_mamba_v2.sh
#
# Available individual ablations:
#   baseline16    ordinary BPTT, no routing, K=16
#   baseline32    ordinary BPTT, no routing, K=32, may be heavy
#   baseline64    ordinary BPTT, no routing, K=64, likely OOM/heavy
#   ckpt64        ordinary BPTT, no routing, K=64, gradient-checkpointed (exact gradient, ~O(1)-in-K memory, ~2x forward compute)
#   none64        K=64, branch gradient closed, identity residual gradient only
#   dyn012        K=64, dynamic ratio threshold=0.12, may be heavy
#   dyn013        K=64, dynamic ratio threshold=0.13, current best
#   dyn014        K=64, dynamic ratio threshold=0.14
#   dyn015        K=64, dynamic ratio threshold=0.15, under-opened ablation
#   periodic4     K=64, open branch gradient every 4 steps
#   periodic8     K=64, open branch gradient every 8 steps
#   periodic16    K=64, open branch gradient every 16 steps
#   tail4         K=64, open branch gradient for last 4 rollout steps
#   tail8         K=64, open branch gradient for last 8 rollout steps
#   tail16        K=64, open branch gradient for last 16 rollout steps
#
# Group aliases:
#   core       baseline16 none64 dyn013 dyn014 dyn015
#   threshold  dyn012 dyn013 dyn014 dyn015
#   policy     none64 periodic8 tail8 dyn013
#   memory     baseline16 baseline32 baseline64 none64 dyn015 dyn013
#   all        baseline16 none64 dyn012 dyn013 dyn014 dyn015 periodic8 tail8

REPO_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Some shared clusters export a system MPS pipe even when its daemon is not
# healthy.  A private empty pipe directory selects ordinary CUDA contexts.
# Set USE_MPS=1 only when the host MPS service is known to be healthy.
if [[ "${USE_MPS:-0}" != "1" ]]; then
  export CUDA_MPS_PIPE_DIRECTORY="${REPO_ROOT}/.mps_bypass"
fi
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}"

# -------------------------
# Data / hardware
# -------------------------
DATA_PATH=${DATA_PATH:-"data/hcp_movie_features"}
MOVIE=${MOVIE:-1}
GPUS=${GPUS:-${GPU:-0,1,2,3}}
SEEDS=${SEEDS:-"2"}
RUN_MODE=${MODE:-train_and_test}
if [[ "${AR_SHARED_ROLLOUT_START:-0}" == "1" ]]; then
  AR_SHARED_START_FLAG=(--ar_shared_rollout_start)
else
  AR_SHARED_START_FLAG=(--no-ar_shared_rollout_start)
fi
RESUME=${RESUME:-auto}

export CUDA_VISIBLE_DEVICES=${GPUS}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
BASE_MASTER_PORT=${MASTER_PORT:-37410}
JOB_IDX=0

# -------------------------
# Ablation selection
# -------------------------
# Default is the paper-core ablation. Override with:
#   ABLATIONS="dyn013" or ABLATIONS="threshold" etc.
ABLATIONS=${ABLATIONS:-"core"}

# -------------------------
# Shared training config
# -------------------------
WINDOW=${WINDOW:-16}
BATCH=${BATCH:-8}
# Gradient accumulation: effective batch = GRAD_ACCUM x BATCH. Use to match a
# multi-GPU effective batch on a single card, e.g. GRAD_ACCUM=4 BATCH=8 on 1
# GPU == 4 GPUs x BATCH=8. Default 1 (no accumulation).
GRAD_ACCUM=${GRAD_ACCUM:-1}
LR=${LR:-1e-4}
EPOCHS=${EPOCHS:-50}
# Early stopping on the (long-horizon) val loss. 0 = off (original behavior:
# train the full EPOCHS). >0 = stop after this many consecutive non-improving
# evals. For a fair converged-vs-converged comparison, raise EPOCHS (e.g. 100)
# and set EARLY_STOP >= 20 -- the patience must exceed the ~10-15 epoch
# recoverable-dip window at intermediate open fractions, or it will kill a
# seed that would have recovered. best.pth (what we eval) is always the
# lowest-val-loss epoch regardless of when we stop.
EARLY_STOP=${EARLY_STOP:-0}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
GRAD_CLIP=${GRAD_CLIP:-1.0}
TRAIN_STRIDE=${TRAIN_STRIDE:-1}
EVAL_STRIDE=${EVAL_STRIDE:-4}

# -------------------------
# Model config
# -------------------------
HIDDEN=${HIDDEN:-4096}
DEPTH=${DEPTH:-4}
DROPOUT=${DROPOUT:-0.0}
SIMPLE_RESIDUAL=${SIMPLE_RESIDUAL:-1}

MAMBA_D_STATE=${MAMBA_D_STATE:-16}
MAMBA_D_CONV=${MAMBA_D_CONV:-4}
MAMBA_EXPAND=${MAMBA_EXPAND:-2}

# Default rollout config for ResGrad K64 experiments.
DEFAULT_BPTT_K=${DEFAULT_BPTT_K:-64}
DEFAULT_BURNIN=${DEFAULT_BURNIN:-32}
DEFAULT_TRAIN_STARTS=${DEFAULT_TRAIN_STARTS:-16}
DEFAULT_LOSS_TYPE=${DEFAULT_LOSS_TYPE:-rel_l2}
DEFAULT_LOSS_DECAY=${DEFAULT_LOSS_DECAY:-1.0}

# Baseline rollout starts. Set lower if baseline32/64 is too heavy.
BASELINE16_TRAIN_STARTS=${BASELINE16_TRAIN_STARTS:-16}
BASELINE32_TRAIN_STARTS=${BASELINE32_TRAIN_STARTS:-16}
BASELINE64_TRAIN_STARTS=${BASELINE64_TRAIN_STARTS:-16}

# Routing defaults.
DEFAULT_RES_BLOCK_GATE=${DEFAULT_RES_BLOCK_GATE:-0.0}
DEFAULT_KEEP_EVERY=${DEFAULT_KEEP_EVERY:-8}
DEFAULT_KEEP_TAIL=${DEFAULT_KEEP_TAIL:-8}

# Auto-calibration of --resgrad_ratio_threshold (dynamic_ratio policy only).
# Set TARGET_OPEN_FRAC (e.g. 0.25) to recalibrate the threshold every
# CALIB_EVERY_BATCHES training batches (using that batch's own data) via a
# no-grad rollout, so the branch-open fraction stays near TARGET_OPEN_FRAC
# even as the ratio distribution drifts during training. Leave at 0 to use
# the fixed --resgrad_ratio_threshold (THRESHOLD) as before.
#   TARGET_OPEN_FRAC="0.25" SEEDS="0 2 3" ABLATIONS="dyn013" bash scripts/train/train_hcp_resgrad_mamba_v2.sh
TARGET_OPEN_FRAC=${TARGET_OPEN_FRAC:-0}
CALIB_NUM_STARTS=${CALIB_NUM_STARTS:-4}
CALIB_EVERY_BATCHES=${CALIB_EVERY_BATCHES:-1}
# EMA smoothing of the calibrated threshold (0 = off, original behavior).
# Nonzero (e.g. 0.9) damps per-batch threshold noise -- the "selection churn"
# suspected of destabilizing intermediate targets like 0.5 -- and gets its own
# output dir suffix (emaX.Y) so it never collides with non-EMA runs.
CALIB_EMA=${CALIB_EMA:-0}

# ResGrad open-fraction SCHEDULE (cheap routing early, near-dense + checkpointing
# late). Off by default (SCHED_OPEN_START<0). When on: hold SCHED_OPEN_START for
# most of training, ramp to SCHED_OPEN_END (1.0=dense) between ramp epochs, and
# from SCHED_CKPT_FROM_EP force gradient checkpointing so the high-g tail stays
# at the checkpointing memory floor instead of paying dense stored-activation cost.
# Example: SCHED_OPEN_START=0.1 SCHED_OPEN_END=1.0 SCHED_RAMP_START_EP=40
#          SCHED_RAMP_END_EP=45 SCHED_CKPT_FROM_EP=40 EPOCHS=50
SCHED_OPEN_START=${SCHED_OPEN_START:--1}
SCHED_OPEN_END=${SCHED_OPEN_END:-1.0}
SCHED_RAMP_START_EP=${SCHED_RAMP_START_EP:-0}
SCHED_RAMP_END_EP=${SCHED_RAMP_END_EP:-0}
SCHED_CKPT_FROM_EP=${SCHED_CKPT_FROM_EP:--1}

SAVE_BASE=${SAVE_BASE:-"experiments/hcp_movie${MOVIE}/resgrad_mamba"}

# Logging / safety.
DRY_RUN=${DRY_RUN:-0}
SKIP_EXISTING=${SKIP_EXISTING:-0}

expand_ablation_list() {
  local item
  for item in ${ABLATIONS}; do
    case "${item}" in
      core)
        echo "baseline16 none64 dyn013 dyn014 dyn015"
        ;;
      threshold)
        echo "dyn012 dyn013 dyn014 dyn015"
        ;;
      policy)
        echo "none64 periodic8 tail8 dyn013"
        ;;
      memory)
        echo "baseline16 baseline32 baseline64 none64 dyn015 dyn013"
        ;;
      all)
        echo "baseline16 none64 dyn012 dyn013 dyn014 dyn015 periodic8 tail8"
        ;;
      *)
        echo "${item}"
        ;;
    esac
  done | tr ' ' '\n' | awk 'NF && !seen[$0]++'
}

set_ablation_config() {
  local ablation="$1"

  # Global variables filled by this function.
  EXP_MODE="dynamic_ratio"
  K="${DEFAULT_BPTT_K}"
  BURNIN="${DEFAULT_BURNIN}"
  STARTS="${DEFAULT_TRAIN_STARTS}"
  LOSS_TYPE="${DEFAULT_LOSS_TYPE}"
  LOSS_DECAY="${DEFAULT_LOSS_DECAY}"

  THRESHOLD="0.13"
  BLOCK_GATE="${DEFAULT_RES_BLOCK_GATE}"
  KEEP_EVERY="${DEFAULT_KEEP_EVERY}"
  KEEP_TAIL="${DEFAULT_KEEP_TAIL}"
  SNR_FRAC=""     # calibrated target open fraction for the snr policy (empty => none)
  CURRENT_GLOBAL_WIENER_STATIC_GAIN=""

  ABLATION_TAG=""

  case "${ablation}" in
    baseline|exact)
      EXP_MODE="baseline"
      K="${DEFAULT_BPTT_K}"
      STARTS="${DEFAULT_TRAIN_STARTS}"
      THRESHOLD="NA"
      BLOCK_GATE="1.0"
      ABLATION_TAG="baseline_BPTT${K}_S${STARTS}"
      ;;

    baseline16)
      EXP_MODE="baseline"
      K=16
      STARTS="${BASELINE16_TRAIN_STARTS}"
      THRESHOLD="NA"
      BLOCK_GATE="1.0"
      ABLATION_TAG="baseline_BPTT16_S${STARTS}"
      ;;

    baseline32)
      EXP_MODE="baseline"
      K=32
      STARTS="${BASELINE32_TRAIN_STARTS}"
      THRESHOLD="NA"
      BLOCK_GATE="1.0"
      ABLATION_TAG="baseline_BPTT32_S${STARTS}"
      ;;

    baseline64)
      EXP_MODE="baseline"
      K=64
      STARTS="${BASELINE64_TRAIN_STARTS}"
      THRESHOLD="NA"
      BLOCK_GATE="1.0"
      ABLATION_TAG="baseline_BPTT64_S${STARTS}"
      ;;

    ckpt64)
      EXP_MODE="checkpoint"
      K=64
      STARTS="${BASELINE64_TRAIN_STARTS}"
      THRESHOLD="NA"
      BLOCK_GATE="1.0"
      ABLATION_TAG="gradCheckpoint_BPTT64_S${STARTS}"
      ;;

    none64)
      EXP_MODE="none"
      K=64
      STARTS="${DEFAULT_TRAIN_STARTS}"
      THRESHOLD="0.13"
      BLOCK_GATE="0.0"
      ABLATION_TAG="resgradNone_BPTT64_S${STARTS}"
      ;;

    dyn012)
      EXP_MODE="dynamic_ratio"
      K=64
      STARTS="${DEFAULT_TRAIN_STARTS}"
      THRESHOLD="0.12"
      BLOCK_GATE="0.0"
      ABLATION_TAG="resgradDynamicRatio0.12_BPTT64_S${STARTS}_base0.0"
      ;;

    dyn013)
      EXP_MODE="dynamic_ratio"
      K=64
      STARTS="${DEFAULT_TRAIN_STARTS}"
      THRESHOLD="0.13"
      BLOCK_GATE="0.0"
      ABLATION_TAG="resgradDynamicRatio0.13_BPTT64_S${STARTS}_base0.0"
      ;;

    dyn014)
      EXP_MODE="dynamic_ratio"
      K=64
      STARTS="${DEFAULT_TRAIN_STARTS}"
      THRESHOLD="0.14"
      BLOCK_GATE="0.0"
      ABLATION_TAG="resgradDynamicRatio0.14_BPTT64_S${STARTS}_base0.0"
      ;;

    dyn015)
      EXP_MODE="dynamic_ratio"
      K=64
      STARTS="${DEFAULT_TRAIN_STARTS}"
      THRESHOLD="0.15"
      BLOCK_GATE="0.0"
      ABLATION_TAG="resgradDynamicRatio0.15_BPTT64_S${STARTS}_base0.0"
      ;;

    periodic4)
      EXP_MODE="periodic"
      K=64
      STARTS="${DEFAULT_TRAIN_STARTS}"
      THRESHOLD="0.13"
      BLOCK_GATE="0.0"
      KEEP_EVERY=4
      ABLATION_TAG="resgradPeriodic4_BPTT64_S${STARTS}_base0.0"
      ;;

    periodic8)
      EXP_MODE="periodic"
      K=64
      STARTS="${DEFAULT_TRAIN_STARTS}"
      THRESHOLD="0.13"
      BLOCK_GATE="0.0"
      KEEP_EVERY=8
      ABLATION_TAG="resgradPeriodic8_BPTT64_S${STARTS}_base0.0"
      ;;

    periodic16)
      EXP_MODE="periodic"
      K=64
      STARTS="${DEFAULT_TRAIN_STARTS}"
      THRESHOLD="0.13"
      BLOCK_GATE="0.0"
      KEEP_EVERY=16
      ABLATION_TAG="resgradPeriodic16_BPTT64_S${STARTS}_base0.0"
      ;;

    tail4)
      EXP_MODE="tail"
      K=64
      STARTS="${DEFAULT_TRAIN_STARTS}"
      THRESHOLD="0.13"
      BLOCK_GATE="0.0"
      KEEP_TAIL=4
      ABLATION_TAG="resgradTail4_BPTT64_S${STARTS}_base0.0"
      ;;

    tail8)
      EXP_MODE="tail"
      K=64
      STARTS="${DEFAULT_TRAIN_STARTS}"
      THRESHOLD="0.13"
      BLOCK_GATE="0.0"
      KEEP_TAIL=8
      ABLATION_TAG="resgradTail8_BPTT64_S${STARTS}_base0.0"
      ;;

    tail16)
      EXP_MODE="tail"
      K=64
      STARTS="${DEFAULT_TRAIN_STARTS}"
      THRESHOLD="0.13"
      BLOCK_GATE="0.0"
      KEEP_TAIL=16
      ABLATION_TAG="resgradTail16_BPTT64_S${STARTS}_base0.0"
      ;;

    # --- SNR / soft-Kalman sigma gate (needs the sigma head: --mamba_crps) ---
    # snrk: parameter-free soft Kalman gain  m = SNR^2/(1+SNR^2) per step. No
    # threshold, no target fraction; g emerges from the model's own sigma head.
    snrk)
      EXP_MODE="snrk"
      K=64
      STARTS="${DEFAULT_TRAIN_STARTS}"
      THRESHOLD="NA"
      BLOCK_GATE="0.0"
      ABLATION_TAG="resgradSNRK_BPTT64_S${STARTS}"
      ;;

    # snr<frac>: SNR gate calibrated to open fraction <frac>, e.g. snr0.5 / snr0.25.
    # Cuts high-amplitude steps whose amplitude is only large noise (||Delta||/||sigma||).
    snr[0-9]*)
      EXP_MODE="snr"
      K=64
      STARTS="${DEFAULT_TRAIN_STARTS}"
      THRESHOLD="1.0"
      BLOCK_GATE="0.0"
      SNR_FRAC="${ablation#snr}"
      ABLATION_TAG="resgradSNR${SNR_FRAC}_BPTT64_S${STARTS}"
      ;;

    # snrt<val>: parameter-free FIXED SNR threshold, e.g. snrt1.0. g emerges from
    # the data's SNR under one universal threshold (no per-dataset tuning).
    snrt*)
      EXP_MODE="snr"
      K=64
      STARTS="${DEFAULT_TRAIN_STARTS}"
      THRESHOLD="${ablation#snrt}"
      BLOCK_GATE="0.0"
      SNR_FRAC="0"
      ABLATION_TAG="resgradSNRt${THRESHOLD}_BPTT64_S${STARTS}"
      ;;

    # dual/dualcoh: legacy cross-example coherence proxy. HCP's aligned-subject
    # batches make its shared-mean assumption less implausible, but it still does
    # not independently identify SNR and is not the new Kalman/Wiener method.
    dual|dualcoh)
      EXP_MODE="dualcoh"
      K=64
      STARTS="${DEFAULT_TRAIN_STARTS}"
      THRESHOLD="NA"
      BLOCK_GATE="0.0"
      ABLATION_TAG="resgradDual_BPTT64_S${STARTS}"
      ;;

    # Noise-identified two-route soft Wiener gate.  Unlike legacy dualcoh,
    # this estimates total/noise route covariances with quadratic VJP probes
    # and a lagged diagonal residual covariance; no sigma head is built.
    dw|dualwiener)
      EXP_MODE="dualwiener"
      K=64
      STARTS="${DEFAULT_TRAIN_STARTS}"
      THRESHOLD="NA"
      BLOCK_GATE="0.0"
      ABLATION_TAG="resgradDualWiener_BPTT64_S${STARTS}"
      ;;

    # Training-only VARX innovation covariance with OAS shrinkage.  The
    # external file is prepared without a forecasting checkpoint; the first
    # DUAL_WIENER_MIN_PROBES route probes remain fully open.
    dwoas|dualwiener_oas)
      EXP_MODE="dualwiener"
      K=64
      STARTS="${DEFAULT_TRAIN_STARTS}"
      THRESHOLD="NA"
      BLOCK_GATE="0.0"
      [[ -n "${DUAL_WIENER_INNOVATION_FILE:-}" ]] || {
        echo "dualwiener_oas needs DUAL_WIENER_INNOVATION_FILE=<estimate.npz>" >&2
        exit 2
      }
      [[ -f "${DUAL_WIENER_INNOVATION_FILE}" ]] || {
        echo "OAS innovation file not found: ${DUAL_WIENER_INNOVATION_FILE}" >&2
        exit 2
      }
      export DUAL_WIENER_INNOVATION_KEY=innovation_variance
      ABLATION_TAG="resgradDualWienerOASInit${DUAL_WIENER_MIN_PROBES:-8}_BPTT64_S${STARTS}"
      ;;

    # Domain-conditional fMRI covariance, e.g. the repeated-subject shared
    # response estimator.  It uses the same route probes/solver as DW but is
    # deliberately not labeled as the universal VARX+OAS estimator.
    dwdomain|dualwiener_domain)
      EXP_MODE="dualwiener"
      K=64
      STARTS="${DEFAULT_TRAIN_STARTS}"
      THRESHOLD="NA"
      BLOCK_GATE="0.0"
      [[ -n "${DUAL_WIENER_INNOVATION_FILE:-}" ]] || {
        echo "dualwiener_domain needs DUAL_WIENER_INNOVATION_FILE=<domain-estimate.npz>" >&2
        exit 2
      }
      [[ -f "${DUAL_WIENER_INNOVATION_FILE}" ]] || {
        echo "domain innovation file not found: ${DUAL_WIENER_INNOVATION_FILE}" >&2
        exit 2
      }
      export DUAL_WIENER_INNOVATION_KEY="${DUAL_WIENER_INNOVATION_KEY:-innovation_variance}"
      if [[ -z "${DUAL_WIENER_NOISE_MODEL:-}" && "${DUAL_WIENER_INNOVATION_KEY}" == "innovation_templates" ]]; then
        DUAL_WIENER_NOISE_MODEL=lagged_residual_bootstrap
      fi
      ABLATION_TAG="resgradDualWienerDomain${HCP_DOMAIN_TAG:-SharedResponse}_BPTT64_S${STARTS}"
      ;;

    # Joint horizon-loss Wiener solve in complete parameter-gradient space.
    # It reuses the domain innovation artifact but leaves all internal Mamba
    # residual routes fully open.
    ghw|globalwiener|global_horizon_wiener)
      EXP_MODE="globalwiener"
      K=64
      STARTS="${DEFAULT_TRAIN_STARTS}"
      THRESHOLD="NA"
      BLOCK_GATE="1.0"
      ABLATION_TAG="globalHorizonWienerGeneric_BPTT64_S${STARTS}"
      if [[ "${GLOBAL_WIENER_LOCAL_FIDELITY:-0}" != "0" && "${GLOBAL_WIENER_LOCAL_FIDELITY:-0}" != "0.0" ]]; then
        ABLATION_TAG="${ABLATION_TAG}_Fidelity${GLOBAL_WIENER_LOCAL_FIDELITY//./p}"
      fi
      ;;

    ghwdomain|globalwiener_domain|global_horizon_wiener_domain)
      EXP_MODE="globalwiener"
      K=64
      STARTS="${DEFAULT_TRAIN_STARTS}"
      THRESHOLD="NA"
      BLOCK_GATE="1.0"
      [[ -n "${DUAL_WIENER_INNOVATION_FILE:-}" ]] || {
        echo "globalwiener_domain needs DUAL_WIENER_INNOVATION_FILE=<domain-estimate.npz>" >&2
        exit 2
      }
      [[ -f "${DUAL_WIENER_INNOVATION_FILE}" ]] || {
        echo "domain innovation file not found: ${DUAL_WIENER_INNOVATION_FILE}" >&2
        exit 2
      }
      export DUAL_WIENER_INNOVATION_KEY="${DUAL_WIENER_INNOVATION_KEY:-innovation_variance}"
      ABLATION_TAG="globalHorizonWienerDomain${HCP_DOMAIN_TAG:-SharedResponse}_BPTT64_S${STARTS}"
      if [[ "${GLOBAL_WIENER_LOCAL_FIDELITY:-0}" != "0" && "${GLOBAL_WIENER_LOCAL_FIDELITY:-0}" != "0.0" ]]; then
        ABLATION_TAG="${ABLATION_TAG}_Fidelity${GLOBAL_WIENER_LOCAL_FIDELITY//./p}"
      fi
      ;;

    ghwstatic*|globalwiener_static*|global_horizon_wiener_static*)
      EXP_MODE="globalwiener"
      K=64
      STARTS="${DEFAULT_TRAIN_STARTS}"
      THRESHOLD="NA"
      BLOCK_GATE="1.0"
      CURRENT_GLOBAL_WIENER_STATIC_GAIN="${GLOBAL_WIENER_STATIC_GAIN:-${ablation##*static}}"
      if [[ ! "${CURRENT_GLOBAL_WIENER_STATIC_GAIN}" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]]; then
        echo "globalwiener_static needs a gain in [0,1], e.g. globalwiener_static0.7" >&2
        exit 2
      fi
      static_tag="${CURRENT_GLOBAL_WIENER_STATIC_GAIN//./p}"
      ABLATION_TAG="globalHorizonWienerStatic${static_tag}_BPTT64_S${STARTS}"
      ;;

    # Structured residual bootstrap: retains cross-coordinate and
    # cross-horizon residual covariance without requiring a spatial grid.
    dwstructured|dualwiener_structured)
      EXP_MODE="dualwiener"
      K=64
      STARTS="${DEFAULT_TRAIN_STARTS}"
      THRESHOLD="NA"
      BLOCK_GATE="0.0"
      DUAL_WIENER_NOISE_MODEL="lagged_residual_bootstrap"
      ABLATION_TAG="resgradDualWienerStructured_BPTT64_S${STARTS}"
      ;;

    *)
      echo "Unknown ablation '${ablation}'." >&2
      echo "Use one of: baseline16 baseline32 baseline64 ckpt64 none64 dyn012 dyn013 dyn014 dyn015 periodic4 periodic8 periodic16 tail4 tail8 tail16 snrk snr0.5 snr0.25 snrt1.0 dualcoh dualwiener dualwiener_domain globalwiener_domain dualwiener_structured" >&2
      echo "Or group aliases: core threshold policy memory all" >&2
      exit 2
      ;;
  esac

  if [[ "${TARGET_OPEN_FRAC}" != "0" && "${EXP_MODE}" == "dynamic_ratio" ]]; then
    ABLATION_TAG="${ABLATION_TAG}_calibP${TARGET_OPEN_FRAC}"
    if [[ "${CALIB_EMA}" != "0" ]]; then
      ABLATION_TAG="${ABLATION_TAG}ema${CALIB_EMA}"
    fi
  fi
  # Scheduled open fraction: distinct dir suffix so a scheduled run never
  # collides with a fixed-g run of the same target. Schedule is ON whenever
  # SCHED_OPEN_START is non-negative (the -1 default starts with '-').
  if [[ "${EXP_MODE}" == "dynamic_ratio" && "${SCHED_OPEN_START}" != -* ]]; then
    ABLATION_TAG="${ABLATION_TAG}_sched${SCHED_OPEN_START}to${SCHED_OPEN_END}_ramp${SCHED_RAMP_START_EP}-${SCHED_RAMP_END_EP}_ckpt${SCHED_CKPT_FROM_EP}"
  fi
}

build_resgrad_args() {
  RESGRAD_ARGS=()

  if [[ "${EXP_MODE}" == "baseline" ]]; then
    RESGRAD_ARGS=(
      --no-resgrad_routing
      --resgrad_policy all
      --resgrad_block_gate 1.0
      --no-recurrent_grad_checkpoint
    )
  elif [[ "${EXP_MODE}" == "checkpoint" ]]; then
    # Ordinary (unrouted) BPTT, but with each rollout step wrapped in
    # torch.utils.checkpoint: exact full-BPTT gradient at ~O(1)-in-K memory
    # per step, ~2x forward compute. The baseline this whole ablation set is
    # implicitly asking "why not just do this instead of ResGrad" -- so run
    # it and compare peak memory / wall-clock / final metrics directly
    # against baseline64 (same K, no checkpoint) and dyn013 (ResGrad).
    RESGRAD_ARGS=(
      --no-resgrad_routing
      --resgrad_policy all
      --resgrad_block_gate 1.0
      --recurrent_grad_checkpoint
    )
  elif [[ "${EXP_MODE}" == "none" ]]; then
    RESGRAD_ARGS=(
      --resgrad_routing
      --resgrad_policy none
      --resgrad_block_gate 0.0
      --resgrad_ratio_threshold "${THRESHOLD}"
      --resgrad_keep_every "${KEEP_EVERY}"
      --resgrad_keep_tail "${KEEP_TAIL}"
    )
  elif [[ "${EXP_MODE}" == "dualcoh" ]]; then
    RESGRAD_ARGS=(
      --resgrad_routing
      --resgrad_policy dualcoh
      --resgrad_block_gate 0.0
      --no-recurrent_grad_checkpoint
    )
  elif [[ "${EXP_MODE}" == "dualwiener" ]]; then
    RESGRAD_ARGS=(
      --resgrad_routing
      --resgrad_policy dualwiener
      --resgrad_block_gate 0.0
      --dual_wiener_ema "${DUAL_WIENER_EMA:-0.95}"
      --dual_wiener_residual_ema "${DUAL_WIENER_RESIDUAL_EMA:-0.99}"
      --dual_wiener_warmup_batches "${DUAL_WIENER_WARMUP_BATCHES:-8}"
      --dual_wiener_probe_every "${DUAL_WIENER_PROBE_EVERY:-4}"
      --dual_wiener_min_probes "${DUAL_WIENER_MIN_PROBES:-1}"
      --dual_wiener_noise_model "${DUAL_WIENER_NOISE_MODEL:-diagonal_gaussian}"
      --no-recurrent_grad_checkpoint
    )
  elif [[ "${EXP_MODE}" == "globalwiener" ]]; then
    RESGRAD_ARGS=(
      --no-resgrad_routing
      --resgrad_policy all
      --resgrad_block_gate 1.0
      --global_horizon_wiener
      --dual_wiener_ema "${GLOBAL_WIENER_EMA:-0.95}"
      --dual_wiener_residual_ema "${GLOBAL_WIENER_RESIDUAL_EMA:-0.99}"
      --dual_wiener_warmup_batches "${GLOBAL_WIENER_WARMUP_BATCHES:-8}"
      --dual_wiener_probe_every "${GLOBAL_WIENER_PROBE_EVERY:-32}"
      --dual_wiener_min_probes "${GLOBAL_WIENER_MIN_PROBES:-4}"
      --dual_wiener_noise_model "${GLOBAL_WIENER_NOISE_MODEL:-diagonal_gaussian}"
      --global_wiener_ridge "${GLOBAL_WIENER_RIDGE:-1e-8}"
      --global_wiener_anchor "${GLOBAL_WIENER_ANCHOR:-0}"
      --global_wiener_local_fidelity "${GLOBAL_WIENER_LOCAL_FIDELITY:-0}"
      --global_wiener_solver_iters "${GLOBAL_WIENER_SOLVER_ITERS:-256}"
      --global_wiener_sketch_dim "${GLOBAL_WIENER_SKETCH_DIM:-8192}"
      --global_wiener_sketch_seed "${GLOBAL_WIENER_SKETCH_SEED:-1729}"
      --global_wiener_noise_draws "${GLOBAL_WIENER_NOISE_DRAWS:-4}"
      --no-recurrent_grad_checkpoint
    )
    if [[ -n "${CURRENT_GLOBAL_WIENER_STATIC_GAIN:-}" ]]; then
      RESGRAD_ARGS+=(
        --global_wiener_static_gain "${CURRENT_GLOBAL_WIENER_STATIC_GAIN}"
        --global_wiener_static_mode "${GLOBAL_WIENER_STATIC_MODE:-delayed_tied}"
        --no-global_wiener_batch_conditioned
      )
    elif [[ "${GLOBAL_WIENER_BATCH_CONDITIONED:-1}" == "1" ]]; then
      RESGRAD_ARGS+=(--global_wiener_batch_conditioned)
    else
      RESGRAD_ARGS+=(--no-global_wiener_batch_conditioned)
    fi
  elif [[ "${EXP_MODE}" == "dynamic_ratio" ]]; then
    RESGRAD_ARGS=(
      --resgrad_routing
      --resgrad_policy dynamic_ratio
      --resgrad_block_gate "${BLOCK_GATE}"
      --resgrad_ratio_threshold "${THRESHOLD}"
      --resgrad_keep_every "${KEEP_EVERY}"
      --resgrad_keep_tail "${KEEP_TAIL}"
    )
    if [[ "${TARGET_OPEN_FRAC}" != "0" ]]; then
      RESGRAD_ARGS+=(
        --resgrad_target_open_frac "${TARGET_OPEN_FRAC}"
        --resgrad_calib_num_starts "${CALIB_NUM_STARTS}"
        --resgrad_calib_every_batches "${CALIB_EVERY_BATCHES}"
        --resgrad_calib_ema "${CALIB_EMA}"
      )
    fi
  elif [[ "${EXP_MODE}" == "periodic" ]]; then
    RESGRAD_ARGS=(
      --resgrad_routing
      --resgrad_policy periodic
      --resgrad_block_gate "${BLOCK_GATE}"
      --resgrad_ratio_threshold "${THRESHOLD}"
      --resgrad_keep_every "${KEEP_EVERY}"
      --resgrad_keep_tail "${KEEP_TAIL}"
    )
  elif [[ "${EXP_MODE}" == "tail" ]]; then
    RESGRAD_ARGS=(
      --resgrad_routing
      --resgrad_policy tail
      --resgrad_block_gate "${BLOCK_GATE}"
      --resgrad_ratio_threshold "${THRESHOLD}"
      --resgrad_keep_every "${KEEP_EVERY}"
      --resgrad_keep_tail "${KEEP_TAIL}"
    )
  elif [[ "${EXP_MODE}" == "snr" ]]; then
    # SNR gate: route internal branches by ||Delta||/||sigma||. sigma is the model's
    # learned noise head (built by --mamba_crps, decoupled MSE-mean + detached-mean CRPS).
    RESGRAD_ARGS=(
      --resgrad_routing
      --resgrad_policy snr
      --resgrad_block_gate 0.0
      --resgrad_ratio_threshold "${THRESHOLD}"
      --mamba_crps 1.0
    )
    if [[ -n "${SNR_FRAC}" && "${SNR_FRAC}" != "0" ]]; then
      RESGRAD_ARGS+=(
        --resgrad_target_open_frac "${SNR_FRAC}"
        --resgrad_calib_num_starts "${CALIB_NUM_STARTS}"
        --resgrad_calib_every_batches "${CALIB_EVERY_BATCHES}"
        --resgrad_calib_ema "${CALIB_EMA}"
      )
    else
      RESGRAD_ARGS+=(--resgrad_target_open_frac 0)
    fi
  elif [[ "${EXP_MODE}" == "snrk" ]]; then
    # Soft Kalman-gain gate: m = SNR^2/(1+SNR^2) per step. Parameter-free; g emerges.
    RESGRAD_ARGS=(
      --resgrad_routing
      --resgrad_policy snrk
      --resgrad_block_gate 0.0
      --resgrad_target_open_frac 0
      --mamba_crps 1.0
    )
  else
    echo "Unknown EXP_MODE '${EXP_MODE}'" >&2
    exit 2
  fi
}

run_one() {
  local seed="$1"
  local ablation="$2"

  set_ablation_config "${ablation}"
  build_resgrad_args

  JOB_IDX=$((JOB_IDX + 1))
  export MASTER_PORT=$((BASE_MASTER_PORT + 1000 * seed + JOB_IDX))

  local residual_arg="--simple_residual"
  local residual_tag="residual"
  if [[ "${SIMPLE_RESIDUAL}" != "1" ]]; then
    residual_arg="--no-simple_residual"
    residual_tag="noresidual"
  fi

  local out_dir="${SAVE_BASE}/official_mamba_state_hid${HIDDEN}_D${DEPTH}_${residual_tag}_${ABLATION_TAG}/seed${seed}"

  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    if [[ -f "${out_dir}/best.pth" || -f "${out_dir}/test_metrics.json" || -f "${out_dir}/metrics.json" ]]; then
      echo "[SKIP] Existing result found: ${out_dir}"
      return 0
    fi
  fi

  echo
  echo "======================================================================"
  echo "ablation=${ablation} | seed=${seed}"
  echo "exp_mode=${EXP_MODE}"
  echo "hidden=${HIDDEN} depth=${DEPTH} K=${K} starts=${STARTS} burnin=${BURNIN}"
  echo "threshold=${THRESHOLD} block_gate=${BLOCK_GATE} keep_every=${KEEP_EVERY} keep_tail=${KEEP_TAIL}"
  echo "save_root=${out_dir}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} MASTER_PORT=${MASTER_PORT}"
  echo "======================================================================"
  echo

  local cmd=(
    python src/main.py
    --mode "${RUN_MODE}"
    --resume "${RESUME}"
    --dataset hcp_movie
    --data_path "${DATA_PATH}"
    --movie "${MOVIE}"
    --model_name official_mamba_state
    --window_size "${WINDOW}"
    --local_batch_size "${BATCH}"
    --base_lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --num_epochs "${EPOCHS}"
    --early_stop_patience "${EARLY_STOP}"
    --ar_loss mse
    --ar_one_step_lambda 1.0
    --ar_train_starts_per_sequence 1
    --ar_train_stride "${TRAIN_STRIDE}"
    --ar_train_random_starts
    "${AR_SHARED_START_FLAG[@]}"
    --ar_eval_stride "${EVAL_STRIDE}"
    --eval_every 1
    --grad_clip "${GRAD_CLIP}"
    --grad_accum_steps "${GRAD_ACCUM}"
    --resgrad_sched_open_start "${SCHED_OPEN_START}"
    --resgrad_sched_open_end "${SCHED_OPEN_END}"
    --resgrad_sched_ramp_start_epoch "${SCHED_RAMP_START_EP}"
    --resgrad_sched_ramp_end_epoch "${SCHED_RAMP_END_EP}"
    --resgrad_sched_ckpt_from_epoch "${SCHED_CKPT_FROM_EP}"
    --simple_hidden_dim "${HIDDEN}"
    --simple_depth "${DEPTH}"
    --simple_dropout "${DROPOUT}"
    ${residual_arg}
    --mamba_bptt_horizon "${K}"
    --mamba_burnin "${BURNIN}"
    --mamba_train_starts_per_sequence "${STARTS}"
    --mamba_train_stride "${TRAIN_STRIDE}"
    --mamba_loss_type "${LOSS_TYPE}"
    --mamba_loss_decay "${LOSS_DECAY}"
    --mamba_d_state "${MAMBA_D_STATE}"
    --mamba_d_conv "${MAMBA_D_CONV}"
    --mamba_expand "${MAMBA_EXPAND}"
  )

  cmd+=("${RESGRAD_ARGS[@]}")

  cmd+=(
    --no-bridge_control_loss
    --bridge_control_lambda 0
    --bridge_control_next_lambda 0
    --test_horizons 1 2 4 8 16 32 64 96 128
    --koopman_long_loss none
    --koopman_lambda_gram 0
    --koopman_lambda_one 1
    --koopman_lambda_point 1
    --koopman_lambda_delta 0
    --koopman_lambda_rec 0
    --koopman_lambda_latent 0
    --no-bptt_loss
    --bptt_lambda 0
    --bptt_horizon 0
    --no-comp_graph_loss
    --comp_lambda 0
    --comp_horizon 0
    --no-frontier_graph_loss
    --frontier_lambda 0
    --frontier_horizon 0
    --no-long_error_cloud_loss
    --no-transport_coh_loss
    --no-source_color_loss
    --no-error_proj_pseudo_loss
    --no-error_proj_cob_loss
    --no-error_proj_v2_loss
    --no-error_align_loss
    --seed "${seed}"
    --save_root "${out_dir}"
  )

  if [[ -n "${EXTRA_ARGS:-}" ]]; then
    # Experiment-only opt-in flags (for example Forward-JReg). Existing runs
    # are byte-for-byte unchanged when EXTRA_ARGS is empty.
    read -r -a _extra_argv <<< "${EXTRA_ARGS}"
    cmd+=("${_extra_argv[@]}")
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY_RUN] '
    printf '%q ' "${cmd[@]}"
    echo
  else
    "${cmd[@]}"
  fi
}

echo "Requested ABLATIONS=${ABLATIONS}"
echo "Expanded ablations:"
mapfile -t EXPANDED_ABLATIONS < <(expand_ablation_list)
printf '  %s\n' "${EXPANDED_ABLATIONS[@]}"

for SEED in ${SEEDS}; do
  for ABLATION in "${EXPANDED_ABLATIONS[@]}"; do
    run_one "${SEED}" "${ABLATION}"
  done
done
