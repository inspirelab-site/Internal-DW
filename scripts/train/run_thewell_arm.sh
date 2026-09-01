#!/usr/bin/env bash
set -euo pipefail

# One matched The Well routing arm on one or more physical GPUs.  Use this same
# runner for Exact, TBPTT, and Internal-DW so optimization, data, model, and
# evaluation settings remain paired.
#
# Example:
#   DATASET=shear_flow METHOD=dualwiener GPU=1 \
#     bash scripts/train/run_thewell_arm.sh

export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1

# The shared system MPS daemon on the H200 nodes has previously wedged DDP
# jobs.  An empty private pipe directory makes CUDA use ordinary contexts.
# Set USE_MPS=1 only when the node's MPS service is known to be healthy.
if [[ "${USE_MPS:-0}" != "1" ]]; then
  export CUDA_MPS_PIPE_DIRECTORY="$(pwd)/.mps_bypass"
  mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}"
fi

REPO_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

DATASET=${DATASET:?Set one of turbulent_radiative_layer_2D, shear_flow, viscoelastic_instability}
METHOD=${METHOD:?Set METHOD=exact, tbptt<S>, artbp8, dualwiener, dualwiener_spectral, dualwiener_structured, or dwc<gain>}
GPU=${GPU:-}
GPUS=${GPUS:-${GPU}}
if [[ -z "${GPUS}" ]]; then
  echo "[error] set GPU=<id> for one GPU or GPUS=<comma-separated ids> for DDP" >&2
  exit 2
fi
FIRST_GPU=${GPUS%%,*}
SEED=${SEED:-0}
EPOCHS=${EPOCHS:-20}
K=${K:-32}
BASE=${BASE:-32}
DEPTH=${DEPTH:-4}
WINDOW=${WINDOW:-2}
BATCH=${BATCH:-1}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LR=${LR:-1e-4}
GRAD_CLIP=${GRAD_CLIP:-1.0}
AR_SCHEDULER=${AR_SCHEDULER:-step}
AR_STEP_SIZE=${AR_STEP_SIZE:-100}
AR_GAMMA=${AR_GAMMA:-0.5}
AR_MIN_LR=${AR_MIN_LR:-1e-6}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-0}
NUM_WORKERS=${NUM_WORKERS:-2}
RESUME=${RESUME:-auto}
SKIP_EXISTING=${SKIP_EXISTING:-1}
RUN_MODE=${RUN_MODE:-train_and_test}
if [[ "${AR_SHARED_ROLLOUT_START:-0}" == "1" ]]; then
  AR_SHARED_START_FLAG=(--ar_shared_rollout_start)
else
  AR_SHARED_START_FLAG=(--no-ar_shared_rollout_start)
fi

WELL_REPO=${WELL_REPO:-external/the_well}
PILOT_BASE=${PILOT_BASE:-${WELL_REPO}/gradient_pilots}
DATA_PATH=${DATA_PATH:-${PILOT_BASE}/datasets/${DATASET}/data}
SAVE_BASE=${SAVE_BASE:-experiments/thewell_four_arm}

case "${DATASET}" in
  turbulent_radiative_layer_2D)
    SPATIAL_SUBSAMPLE=${SPATIAL_SUBSAMPLE:-2}
    SEQ_LEN=${SEQ_LEN:-64}
    SEQ_STRIDE=${SEQ_STRIDE:-16}
    ;;
  shear_flow)
    SPATIAL_SUBSAMPLE=${SPATIAL_SUBSAMPLE:-4}
    SEQ_LEN=${SEQ_LEN:-64}
    SEQ_STRIDE=${SEQ_STRIDE:-16}
    ;;
  viscoelastic_instability)
    SPATIAL_SUBSAMPLE=${SPATIAL_SUBSAMPLE:-4}
    SEQ_LEN=${SEQ_LEN:-48}
    SEQ_STRIDE=${SEQ_STRIDE:-4}
    ;;
  *)
    echo "[error] unsupported DATASET=${DATASET}" >&2
    exit 2
    ;;
esac

if [[ ! -d "${DATA_PATH}/train" || ! -d "${DATA_PATH}/valid" || ! -d "${DATA_PATH}/test" ]]; then
  echo "[error] missing train/valid/test under ${DATA_PATH}" >&2
  exit 3
fi

# Never inherit a constant or corrected-estimator pilot from another run.  A
# domain innovation path is captured first and restored only by the explicit
# dualwiener_domain method below.
DOMAIN_INNOVATION_FILE=${DUAL_WIENER_INNOVATION_FILE:-}
DOMAIN_INNOVATION_KEY=${DUAL_WIENER_INNOVATION_KEY:-}
unset DUAL_WIENER_CONST DUAL_WIENER_INNOVATION_FILE DUAL_WIENER_INNOVATION_KEY
unset DUAL_WIENER_SPECTRUM_OAS

case "${METHOD}" in
  exact)
    TAG=exact
    ROUTE_ARGS=(--no-resgrad_routing --resgrad_policy all --resgrad_block_gate 1.0)
    if [[ "${MATCHED_NO_ACTIVATION_CKPT:-0}" == "1" ]]; then
      GRAPH_ARGS=(--no-bptt_grad_checkpoint --bptt_detach_period 0)
    else
      GRAPH_ARGS=(--bptt_grad_checkpoint --bptt_detach_period 0)
    fi
    ;;
  tbptt[0-9]*)
    period="${METHOD#tbptt}"
    if (( period < 1 )); then
      echo "[error] TBPTT period must be positive, got ${period}" >&2
      exit 4
    fi
    TAG="tbptt${period}"
    ROUTE_ARGS=(--no-resgrad_routing --resgrad_policy all --resgrad_block_gate 1.0)
    GRAPH_ARGS=(--bptt_grad_checkpoint --bptt_detach_period "${period}")
    ;;
  artbp8)
    TAG=artbp8
    ROUTE_ARGS=(--no-resgrad_routing --resgrad_policy all --resgrad_block_gate 1.0)
    GRAPH_ARGS=(
      --bptt_grad_checkpoint --bptt_detach_period 0
      --artbp_expected_segment_length 8
    )
    ;;
  dualwiener)
    TAG=dualwiener
    ROUTE_ARGS=(
      --resgrad_routing --resgrad_policy dualwiener --resgrad_block_gate 1.0
      --dual_wiener_ema "${DW_EMA:-0.95}"
      --dual_wiener_residual_ema "${DW_RESIDUAL_EMA:-0.99}"
      --dual_wiener_warmup_batches "${DW_WARMUP:-8}"
      --dual_wiener_probe_every "${DW_PROBE_EVERY:-4}"
      --dual_wiener_max_horizon "${K}"
    )
    GRAPH_ARGS=(--no-bptt_grad_checkpoint --bptt_detach_period 0)
    ;;
  dualwiener_spectral|dualwiener_spatial)
    TAG=dualwiener_spectral
    ROUTE_ARGS=(
      --resgrad_routing --resgrad_policy dualwiener --resgrad_block_gate 1.0
      --dual_wiener_ema "${DW_EMA:-0.95}"
      --dual_wiener_residual_ema "${DW_RESIDUAL_EMA:-0.99}"
      --dual_wiener_warmup_batches "${DW_WARMUP:-8}"
      --dual_wiener_probe_every "${DW_PROBE_EVERY:-4}"
      --dual_wiener_noise_model spatial_spectrum
      --dual_wiener_max_horizon "${K}"
    )
    GRAPH_ARGS=(--no-bptt_grad_checkpoint --bptt_detach_period 0)
    ;;
  dualwiener_spectral_oas|dwspectraloas)
    TAG=dualwiener_spectral_oas_init8
    export DUAL_WIENER_SPECTRUM_OAS=1
    ROUTE_ARGS=(
      --resgrad_routing --resgrad_policy dualwiener --resgrad_block_gate 1.0
      --dual_wiener_ema "${DW_EMA:-0.95}"
      --dual_wiener_residual_ema "${DW_RESIDUAL_EMA:-0.99}"
      --dual_wiener_warmup_batches "${DW_WARMUP:-8}"
      --dual_wiener_probe_every "${DW_PROBE_EVERY:-4}"
      --dual_wiener_min_probes "${DW_MIN_PROBES:-8}"
      --dual_wiener_noise_model spatial_spectrum
      --dual_wiener_max_horizon "${K}"
    )
    GRAPH_ARGS=(--no-bptt_grad_checkpoint --bptt_detach_period 0)
    ;;
  dualwiener_domain|dwdomain)
    [[ -n "${DOMAIN_INNOVATION_FILE}" ]] || {
      echo "[error] METHOD=${METHOD} needs DUAL_WIENER_INNOVATION_FILE" >&2
      exit 4
    }
    [[ -f "${DOMAIN_INNOVATION_FILE}" ]] || {
      echo "[error] missing domain innovation file: ${DOMAIN_INNOVATION_FILE}" >&2
      exit 4
    }
    export DUAL_WIENER_INNOVATION_FILE="${DOMAIN_INNOVATION_FILE}"
    export DUAL_WIENER_INNOVATION_KEY=innovation_variance
    TAG="dualwiener_domain_${DOMAIN_TAG:-conditional}"
    ROUTE_ARGS=(
      --resgrad_routing --resgrad_policy dualwiener --resgrad_block_gate 1.0
      --dual_wiener_ema "${DW_EMA:-0.95}"
      --dual_wiener_residual_ema "${DW_RESIDUAL_EMA:-0.99}"
      --dual_wiener_warmup_batches "${DW_WARMUP:-8}"
      --dual_wiener_probe_every "${DW_PROBE_EVERY:-4}"
      --dual_wiener_min_probes "${DW_MIN_PROBES:-8}"
      --dual_wiener_noise_model lagged_residual_bootstrap
      --dual_wiener_max_horizon "${K}"
    )
    GRAPH_ARGS=(--no-bptt_grad_checkpoint --bptt_detach_period 0)
    ;;
  dualwiener_structured|dualwiener_bootstrap)
    TAG=dualwiener_structured
    ROUTE_ARGS=(
      --resgrad_routing --resgrad_policy dualwiener --resgrad_block_gate 1.0
      --dual_wiener_ema "${DW_EMA:-0.95}"
      --dual_wiener_residual_ema "${DW_RESIDUAL_EMA:-0.99}"
      --dual_wiener_warmup_batches "${DW_WARMUP:-8}"
      --dual_wiener_probe_every "${DW_PROBE_EVERY:-4}"
      --dual_wiener_noise_model lagged_residual_bootstrap
      --dual_wiener_max_horizon "${K}"
    )
    GRAPH_ARGS=(--no-bptt_grad_checkpoint --bptt_detach_period 0)
    ;;
  globalwiener|global_horizon_wiener)
    TAG=global_horizon_wiener_generic
    if [[ "${GLOBAL_WIENER_LOCAL_FIDELITY:-0}" != "0" && "${GLOBAL_WIENER_LOCAL_FIDELITY:-0}" != "0.0" ]]; then
      TAG="${TAG}_fidelity${GLOBAL_WIENER_LOCAL_FIDELITY//./p}"
    fi
    ROUTE_ARGS=(
      --no-resgrad_routing --resgrad_policy all --resgrad_block_gate 1.0
      --global_horizon_wiener
      --dual_wiener_ema "${GLOBAL_WIENER_EMA:-0.95}"
      --dual_wiener_residual_ema "${GLOBAL_WIENER_RESIDUAL_EMA:-0.99}"
      --dual_wiener_warmup_batches "${GLOBAL_WIENER_WARMUP:-8}"
      --dual_wiener_probe_every "${GLOBAL_WIENER_PROBE_EVERY:-16}"
      --dual_wiener_min_probes "${GLOBAL_WIENER_MIN_PROBES:-4}"
      --dual_wiener_noise_model diagonal_gaussian
      --dual_wiener_max_horizon "${K}"
      --global_wiener_ridge "${GLOBAL_WIENER_RIDGE:-1e-8}"
      --global_wiener_local_fidelity "${GLOBAL_WIENER_LOCAL_FIDELITY:-0}"
      --global_wiener_sketch_dim "${GLOBAL_WIENER_SKETCH_DIM:-8192}"
      --global_wiener_noise_draws "${GLOBAL_WIENER_NOISE_DRAWS:-4}"
      --global_wiener_batch_conditioned
    )
    GRAPH_ARGS=(--no-bptt_grad_checkpoint --bptt_detach_period 0)
    ;;
  globalwiener_domain|global_horizon_wiener_domain)
    if [[ -n "${DOMAIN_INNOVATION_FILE}" ]]; then
      [[ -f "${DOMAIN_INNOVATION_FILE}" ]] || {
        echo "[error] missing DUAL_WIENER_INNOVATION_FILE=${DOMAIN_INNOVATION_FILE}" >&2
        exit 4
      }
      export DUAL_WIENER_INNOVATION_FILE="${DOMAIN_INNOVATION_FILE}"
      # Field-domain priors are stored as structured innovation templates.
      # Keep an explicit caller override, but choose the field-native key by
      # default so already-running queue drivers remain valid after upgrades.
      export DUAL_WIENER_INNOVATION_KEY="${DOMAIN_INNOVATION_KEY:-innovation_templates}"
    fi
    TAG="global_horizon_wiener_prior_${DOMAIN_TAG:-conditional}"
    if [[ "${GLOBAL_WIENER_LOCAL_FIDELITY:-0}" != "0" && "${GLOBAL_WIENER_LOCAL_FIDELITY:-0}" != "0.0" ]]; then
      TAG="${TAG}_fidelity${GLOBAL_WIENER_LOCAL_FIDELITY//./p}"
    fi
    ROUTE_ARGS=(
      --no-resgrad_routing --resgrad_policy all --resgrad_block_gate 1.0
      --global_horizon_wiener
      --dual_wiener_ema "${GLOBAL_WIENER_EMA:-0.95}"
      --dual_wiener_residual_ema "${GLOBAL_WIENER_RESIDUAL_EMA:-0.99}"
      --dual_wiener_warmup_batches "${GLOBAL_WIENER_WARMUP:-8}"
      --dual_wiener_probe_every "${GLOBAL_WIENER_PROBE_EVERY:-16}"
      --dual_wiener_min_probes "${GLOBAL_WIENER_MIN_PROBES:-4}"
      --dual_wiener_noise_model "${GLOBAL_WIENER_NOISE_MODEL:-spatial_spectrum}"
      --dual_wiener_max_horizon "${K}"
      --global_wiener_ridge "${GLOBAL_WIENER_RIDGE:-1e-8}"
      --global_wiener_local_fidelity "${GLOBAL_WIENER_LOCAL_FIDELITY:-0}"
      --global_wiener_sketch_dim "${GLOBAL_WIENER_SKETCH_DIM:-8192}"
      --global_wiener_noise_draws "${GLOBAL_WIENER_NOISE_DRAWS:-4}"
      --global_wiener_batch_conditioned
    )
    GRAPH_ARGS=(--no-bptt_grad_checkpoint --bptt_detach_period 0)
    ;;
  globalwiener_static*|global_horizon_wiener_static*)
    STATIC_GAIN="${GLOBAL_WIENER_STATIC_GAIN:-${METHOD##*static}}"
    [[ "${STATIC_GAIN}" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]] || {
      echo "[error] global static gain must lie in [0,1]; got ${STATIC_GAIN}" >&2
      exit 4
    }
    static_tag="${STATIC_GAIN//./p}"
    TAG="global_horizon_wiener_static${static_tag}"
    ROUTE_ARGS=(
      --no-resgrad_routing --resgrad_policy all --resgrad_block_gate 1.0
      --global_horizon_wiener
      --global_wiener_static_gain "${STATIC_GAIN}"
      --global_wiener_static_mode "${GLOBAL_WIENER_STATIC_MODE:-delayed_tied}"
      --dual_wiener_max_horizon "${K}"
      --no-global_wiener_batch_conditioned
    )
    GRAPH_ARGS=(--no-bptt_grad_checkpoint --bptt_detach_period 0)
    ;;
  dwc[0-9]*|static[0-9]*)
    if [[ "${METHOD}" == dwc* ]]; then
      STATIC_GAIN=${METHOD#dwc}
    else
      STATIC_GAIN=${METHOD#static}
    fi
    if [[ ! "${STATIC_GAIN}" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]]; then
      echo "[error] static gain must lie in [0,1]; got ${STATIC_GAIN}" >&2
      exit 4
    fi
    TAG="dwc${STATIC_GAIN}"
    export DUAL_WIENER_CONST="${STATIC_GAIN}"
    ROUTE_ARGS=(
      --resgrad_routing --resgrad_policy dualwiener --resgrad_block_gate 1.0
      --dual_wiener_max_horizon "${K}"
    )
    GRAPH_ARGS=(--no-bptt_grad_checkpoint --bptt_detach_period 0)
    ;;
  *)
    echo "[error] unsupported METHOD=${METHOD}; use exact, tbptt<S>, artbp8, dualwiener, dualwiener_spectral, dualwiener_domain, dualwiener_structured, or dwc<gain>" >&2
    exit 4
    ;;
esac

MODEL_NAME=${MODEL_NAME:-unet_field}
if [[ "${MODEL_NAME}" == "residual_fno_field" ]]; then
  MODEL_TAG="resfno_w${FNO_WIDTH:-64}_D${FNO_LAYERS:-4}_W${WINDOW}_K${K}_ds${SPATIAL_SUBSAMPLE}"
else
  MODEL_TAG="unet_b${BASE}_D${DEPTH}_W${WINDOW}_K${K}_ds${SPATIAL_SUBSAMPLE}"
fi
OUT=${OUT:-${SAVE_BASE}/${DATASET}/${MODEL_TAG}/${TAG}/seed${SEED}}
if [[ "${SKIP_EXISTING}" == "1" && -f "${OUT}/eval_results.json" ]]; then
  echo "[skip] completed: ${OUT}"
  exit 0
fi
if [[ "${SKIP_EXISTING}" == "1" && "${RUN_MODE}" == "train" && -f "${OUT}/.train_complete" && -s "${OUT}/best.pth" ]]; then
  echo "[skip] completed training: ${OUT}"
  exit 0
fi
mkdir -p "${OUT}" logs/thewell_four_arm

export CUDA_VISIBLE_DEVICES="${GPUS}"
export MASTER_ADDR=127.0.0.1
IFS=',' read -r -a WELL_GPU_IDS <<< "${GPUS}"
WELL_WORLD_SIZE=${#WELL_GPU_IDS[@]}
export MASTER_PORT=${MASTER_PORT:-$((51000 + FIRST_GPU * 101 + SEED * 17 + RANDOM % 500))}

echo "=== The Well ${DATASET} ${TAG} seed${SEED} ==="
echo "physical_gpus=${GPUS}; world_size=${WELL_WORLD_SIZE}; K=${K}; epochs=${EPOCHS}; local_batch=${BATCH}; accum=${GRAD_ACCUM}; data=${DATA_PATH}"
echo "output=${OUT}"

RUNTIME_ARGS=()
if [[ "${FAST_TRAIN_RUNTIME:-0}" == "1" ]]; then
  # Performance-only path; full validation/test diagnostics remain enabled.
  RUNTIME_ARGS+=(--fast_train_logging --no-log_gpu_memory)
fi

python -u src/main.py \
  --mode "${RUN_MODE}" \
  --resume "${RESUME}" \
  --seed "${SEED}" \
  --save_root "${OUT}" \
  --dataset the_well \
  --data_path "${DATA_PATH}" \
  --thewell_dataset_name "${DATASET}" \
  --thewell_sequence_length "${SEQ_LEN}" \
  --thewell_sequence_stride "${SEQ_STRIDE}" \
  --thewell_time_subsample 1 \
  --thewell_spatial_subsample "${SPATIAL_SUBSAMPLE}" \
  --thewell_field_groups t0_fields,t1_fields,t2_fields \
  --thewell_max_trajectories_per_file 0 \
  --model_name "${MODEL_NAME}" \
  --window_size "${WINDOW}" \
  --unet_base_channels "${BASE}" \
  --unet_depth "${DEPTH}" \
  --unet_channel_mult 2 \
  --unet_groups 8 \
  --unet_use_grid \
  --unet_normalize \
  --fno_width "${FNO_WIDTH:-64}" --fno_layers "${FNO_LAYERS:-4}" \
  --fno_modes1 "${FNO_MODES1:-16}" --fno_modes2 "${FNO_MODES2:-16}" \
  --fno_hidden_channels "${FNO_HIDDEN_CHANNELS:-128}" --fno_backend original \
  --fno_use_grid --fno_normalize \
  --local_batch_size "${BATCH}" \
  --grad_accum_steps "${GRAD_ACCUM}" \
  --num_workers "${NUM_WORKERS}" \
  --num_epochs "${EPOCHS}" \
  --base_lr "${LR}" \
  --weight_decay 0 \
  --ar_optimizer adam \
  --ar_scheduler "${AR_SCHEDULER}" \
  --ar_step_size "${AR_STEP_SIZE}" \
  --ar_gamma "${AR_GAMMA}" \
  --ar_min_lr "${AR_MIN_LR}" \
  --eval_every 1 \
  --early_stop_patience "${EARLY_STOP_PATIENCE}" \
  --grad_clip "${GRAD_CLIP}" \
  --ar_loss rel_l2 \
  --ar_one_step_lambda 1.0 \
  --ar_train_starts_per_sequence 1 \
  --ar_train_stride 1 \
  --ar_train_random_starts \
  "${AR_SHARED_START_FLAG[@]}" \
  --ar_eval_stride 4 \
  --bptt_loss \
  --bptt_eval \
  --bptt_horizon "${K}" \
  --bptt_lambda 1.0 \
  --bptt_loss_type rel_l2 \
  --test_horizons 1 2 4 8 16 24 32 40 \
  "${RUNTIME_ARGS[@]}" \
  "${ROUTE_ARGS[@]}" \
  "${GRAPH_ARGS[@]}" \
  --no-field_peh_loss \
  --field_peh_lambda 0 \
  ${EXTRA_ARGS:-}

if [[ "${RUN_MODE}" == "train" ]]; then
  if [[ ! -s "${OUT}/best.pth" ]]; then
    echo "[failed] train mode exited without ${OUT}/best.pth" >&2
    exit 4
  fi
  touch "${OUT}/.train_complete"
fi
echo "[done] ${DATASET} ${TAG} seed${SEED}; output=${OUT}"
