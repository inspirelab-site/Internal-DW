#!/usr/bin/env bash
set -euo pipefail

# Paper-aligned Movie-fMRI runner.
#
# Examples:
#   ABLATIONS="baseline dualwiener_domain" SEEDS="0 1 2" GPUS=0,1,2,3 \
#     DUAL_WIENER_INNOVATION_FILE=artifacts/hcp_subject_crossfit.npz \
#     bash scripts/train/train_hcp_resgrad_mamba_v2.sh
#
# Supported arms:
#   baseline / exact          Full BPTT at DEFAULT_BPTT_K
#   baseline16/32/64          Full BPTT horizon controls
#   ckpt64                    Full BPTT with activation checkpointing
#   dw / dualwiener           online Internal-DW estimator
#   dwoas / dualwiener_oas    train-only OAS innovation prior
#   dwdomain / dualwiener_domain
#                              subject-crossfit fMRI innovation prior
#   dwstructured / dualwiener_structured
#                              structured residual-bootstrap estimator

REPO_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

if [[ "${USE_MPS:-0}" != "1" ]]; then
  export CUDA_MPS_PIPE_DIRECTORY="${REPO_ROOT}/.mps_bypass"
fi
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}"

DATA_PATH=${DATA_PATH:-data/hcp_movie_features}
MOVIE=${MOVIE:-1}
GPUS=${GPUS:-${GPU:-0}}
SEEDS=${SEEDS:-0}
ABLATIONS=${ABLATIONS:-baseline}
RUN_MODE=${MODE:-train_and_test}
RESUME=${RESUME:-auto}
SAVE_BASE=${SAVE_BASE:-experiments/hcp_movie${MOVIE}/resgrad_mamba}
SKIP_EXISTING=${SKIP_EXISTING:-1}
DRY_RUN=${DRY_RUN:-0}

WINDOW=${WINDOW:-16}
BATCH=${BATCH:-8}
GRAD_ACCUM=${GRAD_ACCUM:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
LR=${LR:-1e-4}
EPOCHS=${EPOCHS:-100}
EARLY_STOP=${EARLY_STOP:-20}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
GRAD_CLIP=${GRAD_CLIP:-1.0}
TRAIN_STRIDE=${TRAIN_STRIDE:-1}
EVAL_STRIDE=${EVAL_STRIDE:-4}

HIDDEN=${HIDDEN:-4096}
DEPTH=${DEPTH:-4}
DROPOUT=${DROPOUT:-0.0}
MAMBA_D_STATE=${MAMBA_D_STATE:-16}
MAMBA_D_CONV=${MAMBA_D_CONV:-4}
MAMBA_EXPAND=${MAMBA_EXPAND:-2}
DEFAULT_BPTT_K=${DEFAULT_BPTT_K:-64}
BURNIN=${BURNIN:-32}
STARTS=${STARTS:-16}
LOSS_TYPE=${LOSS_TYPE:-rel_l2}
LOSS_DECAY=${LOSS_DECAY:-1.0}

if [[ "${AR_SHARED_ROLLOUT_START:-0}" == "1" ]]; then
  AR_SHARED_START_FLAG=(--ar_shared_rollout_start)
else
  AR_SHARED_START_FLAG=(--no-ar_shared_rollout_start)
fi

export CUDA_VISIBLE_DEVICES="${GPUS}"
BASE_MASTER_PORT=${MASTER_PORT:-37410}
JOB_IDX=0

run_one() {
  local seed="$1"
  local arm="$2"
  local k="${DEFAULT_BPTT_K}"
  local tag=""
  local noise_model="${DUAL_WIENER_NOISE_MODEL:-diagonal_gaussian}"
  local route_args=()

  unset DUAL_WIENER_CONST

  case "${arm}" in
    baseline|exact)
      tag="baseline_BPTT${k}_S${STARTS}"
      route_args=(--no-resgrad_routing --resgrad_policy all --resgrad_block_gate 1.0 --no-recurrent_grad_checkpoint)
      ;;
    baseline16|baseline32|baseline64)
      k="${arm#baseline}"
      tag="baseline_BPTT${k}_S${STARTS}"
      route_args=(--no-resgrad_routing --resgrad_policy all --resgrad_block_gate 1.0 --no-recurrent_grad_checkpoint)
      ;;
    ckpt64)
      k=64
      tag="gradCheckpoint_BPTT64_S${STARTS}"
      route_args=(--no-resgrad_routing --resgrad_policy all --resgrad_block_gate 1.0 --recurrent_grad_checkpoint)
      ;;
    dw|dualwiener)
      tag="resgradDualWiener_BPTT${k}_S${STARTS}"
      route_args=(--resgrad_routing --resgrad_policy dualwiener --resgrad_block_gate 0.0 --no-recurrent_grad_checkpoint)
      ;;
    dwoas|dualwiener_oas)
      [[ -n "${DUAL_WIENER_INNOVATION_FILE:-}" && -f "${DUAL_WIENER_INNOVATION_FILE}" ]] || {
        echo "arm ${arm} requires DUAL_WIENER_INNOVATION_FILE=<train-only-oas.npz>" >&2
        return 2
      }
      export DUAL_WIENER_INNOVATION_KEY="${DUAL_WIENER_INNOVATION_KEY:-innovation_variance}"
      tag="resgradDualWienerOASInit${DUAL_WIENER_MIN_PROBES:-8}_BPTT${k}_S${STARTS}"
      route_args=(--resgrad_routing --resgrad_policy dualwiener --resgrad_block_gate 0.0 --no-recurrent_grad_checkpoint)
      ;;
    dwdomain|dualwiener_domain)
      [[ -n "${DUAL_WIENER_INNOVATION_FILE:-}" && -f "${DUAL_WIENER_INNOVATION_FILE}" ]] || {
        echo "arm ${arm} requires DUAL_WIENER_INNOVATION_FILE=<subject-crossfit.npz>" >&2
        return 2
      }
      export DUAL_WIENER_INNOVATION_KEY="${DUAL_WIENER_INNOVATION_KEY:-innovation_variance}"
      if [[ "${DUAL_WIENER_INNOVATION_KEY}" == "innovation_templates" ]]; then
        noise_model="${DUAL_WIENER_NOISE_MODEL:-lagged_residual_bootstrap}"
      fi
      tag="resgradDualWienerDomain${HCP_DOMAIN_TAG:-SharedResponse}_BPTT${k}_S${STARTS}"
      route_args=(--resgrad_routing --resgrad_policy dualwiener --resgrad_block_gate 0.0 --no-recurrent_grad_checkpoint)
      ;;
    dwstructured|dualwiener_structured)
      noise_model=lagged_residual_bootstrap
      tag="resgradDualWienerStructured_BPTT${k}_S${STARTS}"
      route_args=(--resgrad_routing --resgrad_policy dualwiener --resgrad_block_gate 0.0 --no-recurrent_grad_checkpoint)
      ;;
    *)
      echo "unsupported ABLATION=${arm}" >&2
      return 2
      ;;
  esac

  if [[ "${arm}" == dw* || "${arm}" == dualwiener* ]]; then
    route_args+=(
      --dual_wiener_ema "${DUAL_WIENER_EMA:-0.95}"
      --dual_wiener_residual_ema "${DUAL_WIENER_RESIDUAL_EMA:-0.99}"
      --dual_wiener_warmup_batches "${DUAL_WIENER_WARMUP_BATCHES:-8}"
      --dual_wiener_probe_every "${DUAL_WIENER_PROBE_EVERY:-4}"
      --dual_wiener_min_probes "${DUAL_WIENER_MIN_PROBES:-1}"
      --dual_wiener_noise_model "${noise_model}"
    )
  fi

  local out_dir="${SAVE_BASE}/official_mamba_state_hid${HIDDEN}_D${DEPTH}_residual_${tag}/seed${seed}"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${out_dir}/eval_results.json" ]]; then
    echo "[skip] ${out_dir}"
    return 0
  fi

  JOB_IDX=$((JOB_IDX + 1))
  export MASTER_PORT=$((BASE_MASTER_PORT + seed * 1000 + JOB_IDX))
  mkdir -p "${out_dir}"

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
    --grad_accum_steps "${GRAD_ACCUM}"
    --num_workers "${NUM_WORKERS}"
    --base_lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --num_epochs "${EPOCHS}"
    --early_stop_patience "${EARLY_STOP}"
    --eval_every 1
    --grad_clip "${GRAD_CLIP}"
    --ar_loss mse
    --ar_one_step_lambda 1.0
    --ar_train_starts_per_sequence 1
    --ar_train_stride "${TRAIN_STRIDE}"
    --ar_train_random_starts
    "${AR_SHARED_START_FLAG[@]}"
    --ar_eval_stride "${EVAL_STRIDE}"
    --simple_hidden_dim "${HIDDEN}"
    --simple_depth "${DEPTH}"
    --simple_dropout "${DROPOUT}"
    --simple_residual
    --mamba_bptt_horizon "${k}"
    --mamba_burnin "${BURNIN}"
    --mamba_train_starts_per_sequence "${STARTS}"
    --mamba_train_stride "${TRAIN_STRIDE}"
    --mamba_loss_type "${LOSS_TYPE}"
    --mamba_loss_decay "${LOSS_DECAY}"
    --mamba_d_state "${MAMBA_D_STATE}"
    --mamba_d_conv "${MAMBA_D_CONV}"
    --mamba_expand "${MAMBA_EXPAND}"
    --no-bptt_loss
    --test_horizons 1 2 4 8 16 32 64 96 128
    --seed "${seed}"
    --save_root "${out_dir}"
  )
  cmd+=("${route_args[@]}")

  if [[ -n "${EXTRA_ARGS:-}" ]]; then
    read -r -a extra_argv <<< "${EXTRA_ARGS}"
    cmd+=("${extra_argv[@]}")
  fi

  echo "=== HCP arm=${arm} seed=${seed} K=${k} GPUs=${GPUS} -> ${out_dir} ==="
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY_RUN] '; printf '%q ' "${cmd[@]}"; echo
  else
    "${cmd[@]}"
  fi
}

for seed in ${SEEDS}; do
  for arm in ${ABLATIONS}; do
    run_one "${seed}" "${arm}"
  done
done
