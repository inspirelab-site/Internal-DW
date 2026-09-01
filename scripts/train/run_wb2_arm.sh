#!/usr/bin/env bash
# One matched WeatherBench-2 arm on one or more physical GPUs.
set -euo pipefail
REPO_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd -P)}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export MASTER_ADDR=127.0.0.1
export PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
if [[ "${USE_MPS:-0}" != "1" ]]; then
  export CUDA_MPS_PIPE_DIRECTORY="$(pwd)/.mps_bypass"
  mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}"
fi

METHOD=${METHOD:?set METHOD (exact | tbptt8 | dualwiener | dualwiener_spectral | dualwiener_spectral_oas | dwc0.90)}
# Backward compatible: existing queues pass GPU=2, while one DDP job may pass
# GPUS=0,1,2,3.  src/main.py spawns one rank per visible device.
GPU=${GPU:-}
GPUS=${GPUS:-${GPU}}
if [[ -z "${GPUS}" ]]; then
  echo "set GPU=<id> for one GPU or GPUS=<comma-separated ids> for DDP" >&2
  exit 2
fi
SEED=${SEED:-0}
EPOCHS=${EPOCHS:-20}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-0}
DATA_PATH=${DATA_PATH:-data/weatherbench2_1p5_pilot}
K=${K:-48}
WINDOW=${WINDOW:-2}
SEG_LEN=${SEG_LEN:-64}
BASE_CH=${BASE_CH:-32}
DEPTH=${DEPTH:-4}
BATCH=${BATCH:-1}
GRAD_ACCUM=${GRAD_ACCUM:-8}
STARTS=${STARTS:-1}
LR=${LR:-1e-4}
GRAD_CLIP=${GRAD_CLIP:-1.0}
NUM_WORKERS=${NUM_WORKERS:-2}
RESUME=${RESUME:-auto}
SAVE_BASE=${SAVE_BASE:-experiments/weatherbench2/four_arm/unet_c32_D4_W2_K48}
MODE=${MODE:-train}
if [[ "${AR_SHARED_ROLLOUT_START:-0}" == "1" ]]; then
  AR_SHARED_START_FLAG=(--ar_shared_rollout_start)
else
  AR_SHARED_START_FLAG=(--no-ar_shared_rollout_start)
fi

# Do not inherit a controller variant from the launch shell.
DOMAIN_INNOVATION_FILE=${DUAL_WIENER_INNOVATION_FILE:-}
DOMAIN_INNOVATION_KEY=${DUAL_WIENER_INNOVATION_KEY:-}
unset DUAL_WIENER_CONST DUAL_WIENER_INNOVATION_FILE DUAL_WIENER_INNOVATION_KEY
unset DUAL_WIENER_SPECTRUM_OAS

case "${METHOD}" in
  exact)
    TAG=exact
    # Reuse the already-trained seed-0 exact pilot and restore all optimizer
    # state.  This path is intentionally the same as the qualification run.
    SAVE_ROOT=${SAVE_ROOT:-experiments/weatherbench2/exact_pilot/unet_c32_D4_W2_K48/seed${SEED}}
    ROUTE_ARGS=(--no-resgrad_routing --resgrad_policy all --resgrad_block_gate 1.0)
    if [[ "${MATCHED_NO_ACTIVATION_CKPT:-0}" == "1" ]]; then
      GRAPH_ARGS=(--no-bptt_grad_checkpoint --bptt_detach_period 0)
    else
      GRAPH_ARGS=(--bptt_grad_checkpoint --bptt_detach_period 0)
    fi
    ;;
  tbptt8)
    TAG=tbptt8
    SAVE_ROOT=${SAVE_ROOT:-${SAVE_BASE}/${TAG}/seed${SEED}}
    ROUTE_ARGS=(--no-resgrad_routing --resgrad_policy all --resgrad_block_gate 1.0)
    GRAPH_ARGS=(--bptt_grad_checkpoint --bptt_detach_period 8)
    ;;
  dualwiener)
    TAG=dualwiener
    SAVE_ROOT=${SAVE_ROOT:-${SAVE_BASE}/${TAG}/seed${SEED}}
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
  dualwiener_spectral|dwspectral)
    TAG=dualwiener_spectral
    SAVE_ROOT=${SAVE_ROOT:-${SAVE_BASE}/${TAG}/seed${SEED}}
    ROUTE_ARGS=(
      --resgrad_routing --resgrad_policy dualwiener --resgrad_block_gate 1.0
      --dual_wiener_ema "${DW_EMA:-0.95}"
      --dual_wiener_residual_ema "${DW_RESIDUAL_EMA:-0.99}"
      --dual_wiener_warmup_batches "${DW_WARMUP:-8}"
      --dual_wiener_probe_every "${DW_PROBE_EVERY:-4}"
      --dual_wiener_max_horizon "${K}"
      --dual_wiener_noise_model spatial_spectrum
    )
    GRAPH_ARGS=(--no-bptt_grad_checkpoint --bptt_detach_period 0)
    ;;
  dualwiener_spectral_oas|dwspectraloas)
    TAG=dualwiener_spectral_oas_init8
    SAVE_ROOT=${SAVE_ROOT:-${SAVE_BASE}/${TAG}/seed${SEED}}
    export DUAL_WIENER_SPECTRUM_OAS=1
    ROUTE_ARGS=(
      --resgrad_routing --resgrad_policy dualwiener --resgrad_block_gate 1.0
      --dual_wiener_ema "${DW_EMA:-0.95}"
      --dual_wiener_residual_ema "${DW_RESIDUAL_EMA:-0.99}"
      --dual_wiener_warmup_batches "${DW_WARMUP:-8}"
      --dual_wiener_probe_every "${DW_PROBE_EVERY:-4}"
      --dual_wiener_min_probes "${DW_MIN_PROBES:-8}"
      --dual_wiener_max_horizon "${K}"
      --dual_wiener_noise_model spatial_spectrum
    )
    GRAPH_ARGS=(--no-bptt_grad_checkpoint --bptt_detach_period 0)
    ;;
  globalwiener|global_horizon_wiener)
    TAG=global_horizon_wiener_generic
    if [[ "${GLOBAL_WIENER_LOCAL_FIDELITY:-0}" != "0" && "${GLOBAL_WIENER_LOCAL_FIDELITY:-0}" != "0.0" ]]; then
      TAG="${TAG}_fidelity${GLOBAL_WIENER_LOCAL_FIDELITY//./p}"
    fi
    SAVE_ROOT=${SAVE_ROOT:-${SAVE_BASE}/${TAG}/seed${SEED}}
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
        exit 2
      }
      export DUAL_WIENER_INNOVATION_FILE="${DOMAIN_INNOVATION_FILE}"
    fi
    export DUAL_WIENER_INNOVATION_KEY="${DOMAIN_INNOVATION_KEY:-innovation_variance}"
    TAG="global_horizon_wiener_prior_${DOMAIN_TAG:-conditional}"
    if [[ "${GLOBAL_WIENER_LOCAL_FIDELITY:-0}" != "0" && "${GLOBAL_WIENER_LOCAL_FIDELITY:-0}" != "0.0" ]]; then
      TAG="${TAG}_fidelity${GLOBAL_WIENER_LOCAL_FIDELITY//./p}"
    fi
    SAVE_ROOT=${SAVE_ROOT:-${SAVE_BASE}/${TAG}/seed${SEED}}
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
      exit 2
    }
    static_tag="${STATIC_GAIN//./p}"
    TAG="global_horizon_wiener_static${static_tag}"
    SAVE_ROOT=${SAVE_ROOT:-${SAVE_BASE}/${TAG}/seed${SEED}}
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
    TAG="dwc${STATIC_GAIN}"
    SAVE_ROOT=${SAVE_ROOT:-${SAVE_BASE}/${TAG}/seed${SEED}}
    export DUAL_WIENER_CONST="${STATIC_GAIN}"
    ROUTE_ARGS=(
      --resgrad_routing --resgrad_policy dualwiener --resgrad_block_gate 1.0
      --dual_wiener_max_horizon "${K}"
    )
    GRAPH_ARGS=(--no-bptt_grad_checkpoint --bptt_detach_period 0)
    ;;
  *)
    echo "unknown METHOD=${METHOD}; expected exact, tbptt8, dualwiener, dualwiener_spectral, dualwiener_spectral_oas, or dwc<gain>" >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES=${GPUS}
FIRST_GPU=${GPUS%%,*}
IFS=',' read -r -a WB2_GPU_IDS <<< "${GPUS}"
WB2_WORLD_SIZE=${#WB2_GPU_IDS[@]}
export MASTER_PORT=${MASTER_PORT:-$((49600 + FIRST_GPU * 37 + SEED * 11 + RANDOM % 200))}
mkdir -p "${SAVE_ROOT}" logs/weatherbench2/four_arm
MODEL_NAME=${MODEL_NAME:-unet_field}

echo "=== WB2 ${TAG} seed${SEED}; physical_gpus=${GPUS}; world_size=${WB2_WORLD_SIZE}; K=${K}; epochs=${EPOCHS}; resume=${RESUME}; out=${SAVE_ROOT} === $(date)"

python -u src/main.py \
  --mode "${MODE}" --dataset weatherbench2 --data_path "${DATA_PATH}" --seed "${SEED}" \
  --resume "${RESUME}" \
  --wb2_seg_len "${SEG_LEN}" --wb2_train_stride "${SEG_LEN}" --wb2_eval_stride "${SEG_LEN}" \
  --model_name "${MODEL_NAME}" \
  --unet_base_channels "${BASE_CH}" --unet_depth "${DEPTH}" --unet_channel_mult 2 \
  --unet_groups 8 --unet_use_grid --no-unet_normalize \
  --fno_width "${FNO_WIDTH:-64}" --fno_layers "${FNO_LAYERS:-4}" \
  --fno_modes1 "${FNO_MODES1:-16}" --fno_modes2 "${FNO_MODES2:-16}" \
  --fno_hidden_channels "${FNO_HIDDEN_CHANNELS:-128}" --fno_backend original \
  --fno_use_grid --no-fno_normalize \
  --window_size "${WINDOW}" \
  --local_batch_size "${BATCH}" --grad_accum_steps "${GRAD_ACCUM}" --num_workers "${NUM_WORKERS}" \
  --base_lr "${LR}" --weight_decay 1e-4 --grad_clip "${GRAD_CLIP}" \
  --num_epochs "${EPOCHS}" --early_stop_patience "${EARLY_STOP_PATIENCE}" --eval_every 1 \
  --probe_checkpoint_epochs 0 1 2 3 5 10 20 \
  --ar_loss rel_l2 --ar_one_step_lambda 1.0 \
  --ar_train_starts_per_sequence "${STARTS}" --ar_train_stride 1 --ar_train_random_starts \
  "${AR_SHARED_START_FLAG[@]}" --ar_eval_stride 4 \
  --bptt_loss --bptt_eval --bptt_horizon "${K}" --bptt_lambda 1.0 --bptt_loss_type rel_l2 \
  --test_horizons 4 8 12 20 28 40 48 \
  "${ROUTE_ARGS[@]}" "${GRAPH_ARGS[@]}" \
  --no-bridge_control_loss --bridge_control_lambda 0 \
  --koopman_long_loss none --no-comp_graph_loss --no-frontier_graph_loss \
  --save_root "${SAVE_ROOT}" ${EXTRA_ARGS:-} \
  2>&1 | tee -a "logs/weatherbench2/four_arm/${TAG}_s${SEED}.log"

echo "=== done WB2 ${TAG} seed${SEED} === $(date)"
