#!/usr/bin/env bash
# One idempotent run for the competing-baseline (A) or horizon-interaction (B)
# study.  The server queues below are the intended entry points.
set -euo pipefail

SCRIPT_ROOT=$(cd "$(dirname "$0")/../.." && pwd -P)
cd "${PROJECT_ROOT:-${SCRIPT_ROOT}}"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

PHASE=${PHASE:?set PHASE=A or B}
DATA=${DATA:?set DATA=mg|narma|ieeg|fmri|shear|wb2|ettm1|ettm2}
ARM=${ARM:?set ARM=clip|jreg|exact|dw}
K=${K:?set K}
SEED=${SEED:?set SEED}
GPUS=${GPUS:?set GPUS}

source configs/reproduce/figure6.sh
export NUM_WORKERS=${NUM_WORKERS:-$(figure6_num_workers "${DATA}" "${ARM}" "${SEED}")}
echo "[paper-config] data=${DATA} arm=${ARM} seed=${SEED} num_workers=${NUM_WORKERS}"

case "${PHASE}" in
  A) [[ "${ARM}" == clip || "${ARM}" == jreg ]] || {
       echo "[refuse] phase A only launches the two missing arms: clip or jreg" >&2; exit 2; } ;;
  B) [[ "${ARM}" == exact || "${ARM}" == dw ]] || {
       echo "[refuse] phase B only compares exact or assigned dw" >&2; exit 2; } ;;
  *) echo "[refuse] PHASE must be A or B" >&2; exit 2 ;;
esac

ROOT=${ROOT:-experiments/internal_dw_baselines_k_v1}
SYNC_ROOT=${SYNC_ROOT:-artifacts/internal_dw_baselines_k_v1_sync}
EVAL_ROOT=${EVAL_ROOT:-probe_outputs/internal_dw_baselines_k_v1}
# Hyperparameter screening must not evaluate candidates on the test split.
TRAIN_ONLY=${TRAIN_ONLY:-0}
if [[ "${TRAIN_ONLY}" == 1 ]]; then
  export RUN_MODE=train MODE=train
fi
mkdir -p "${SYNC_ROOT}" "${EVAL_ROOT}"

run_id="${PHASE}_${DATA}_${ARM}_K${K}_s${SEED}"
done_file="${SYNC_ROOT}/${run_id}.done"
lock_file="${SYNC_ROOT}/${run_id}.lock"
if [[ -s "${done_file}" ]]; then
  echo "[skip] ${run_id}: done"
  exit 0
fi
if command -v flock >/dev/null 2>&1; then
  exec {RUN_LOCK_FD}>"${lock_file}"
  if ! flock -n "${RUN_LOCK_FD}"; then
    echo "[skip] ${run_id}: claimed by another server"
    exit 0
  fi
fi
if [[ -s "${done_file}" ]]; then exit 0; fi

GRAD_CLIP=1.0
EXTRA_ARGS=""
if [[ "${ARM}" == clip ]]; then
  # Exact already uses the standard 1.0 safeguard.  This is the deliberately
  # stronger amplitude-only baseline used in the paper comparison.
  GRAD_CLIP=${CLIP_NORM:-0.1}
elif [[ "${ARM}" == jreg ]]; then
  EXTRA_ARGS="--forward_jacobian_lambda ${JREG_LAMBDA:-0.1} --forward_jacobian_target ${JREG_TARGET:-1.0} --forward_jacobian_eps ${JREG_EPS:-0.001}"
fi

out=""
case "${DATA}" in
  mg|narma|ieeg|ettm1|ettm2)
    case "${DATA}" in
      mg) dataset=mackey_glass; cond=tau30; hidden=128 ;;
      narma) dataset=narma; cond=L5; hidden=128 ;;
      ieeg) dataset=ieeg; cond=theta; hidden=256 ;;
      ettm1|ettm2)
        dataset=prepared_temporal_driven
        cond="${DATA}"
        hidden=128
        prepared_npz="${PREPARED_INPUT_ROOT:-probe_inputs/temporal_candidate_regime_v1}/${cond}.npz"
        [[ -s "${prepared_npz}" ]] || { echo "[missing] ${prepared_npz}" >&2; exit 3; }
        export PREPARED_NPZ="${prepared_npz}"
        export MAMBA_TRAIN_STARTS=16
        ;;
    esac
    method=ckpt
    tag=ckpt
    if [[ "${ARM}" == dw ]]; then
      if [[ "${DATA}" == ieeg ]]; then
        method=dualwiener_domain
        tag=dualwiener_domain_ieeg_longmemory
        prior=${IEEG_PRIOR:-artifacts/domain_innovation/ieeg_theta_longmemory_factor_templates_K64.npz}
        [[ -s "${prior}" ]] || { echo "[missing] ${prior}" >&2; exit 3; }
        export DUAL_WIENER_INNOVATION_FILE="${prior}"
        export DUAL_WIENER_INNOVATION_KEY=innovation_templates
        export DUAL_WIENER_DOMAIN_NOISE_MODEL=lagged_residual_bootstrap
        export DUAL_WIENER_DOMAIN_TAG=ieeg_longmemory
        export DUAL_WIENER_MIN_PROBES=8
      else
        method=dualwiener_structured
        tag=dualwiener_structured
        unset DUAL_WIENER_INNOVATION_FILE DUAL_WIENER_INNOVATION_KEY || true
      fi
    fi
    save_base="${ROOT}/${PHASE}/${ARM}"
    out="${save_base}/${dataset}/${cond}_K${K}/${tag}/seed${SEED}"
    # These defaults describe the original one-GPU optimizer geometry.  The
    # explicit overrides are used by the DDP completion queues to preserve the
    # same *global* effective batch instead of multiplying it by world size.
    batch=${MEM_BATCH:-4}; accum=${MEM_GRAD_ACCUM:-8}
    if [[ "${DATA}" == ettm1 || "${DATA}" == ettm2 ]]; then
      # Match the completed ETTm K=64 runs exactly.  K=128 is attempted with
      # the same optimizer geometry; callers may override only after an OOM.
      batch=${ETT_BATCH:-32}; accum=${ETT_GRAD_ACCUM:-1}
    elif [[ "${DATA}" == ieeg && "${K}" -ge 128 ]]; then
      batch=2; accum=16
    fi
    env DATASET="${dataset}" COND="${cond}" K="${K}" METHOD="${method}" \
      SEED="${SEED}" GPUS="${GPUS}" HIDDEN="${hidden}" \
      BATCH="${batch}" GRAD_ACCUM="${accum}" EPOCHS=100 ES=20 LR=1e-4 \
      NUM_WORKERS="${NUM_WORKERS}" RECURRENT_EVAL_HORIZON_BATCH=3 \
      GRAD_CLIP="${GRAD_CLIP}" EXTRA_ARGS="${EXTRA_ARGS}" \
      SAVE_BASE="${save_base}" SKIP_EXISTING=1 RESUME=auto \
      bash scripts/train/run_mem_one.sh
    ;;

  shear)
    method=exact; tag=exact
    if [[ "${ARM}" == dw ]]; then method=dualwiener_spectral; tag=dualwiener_spectral; fi
    save_base="${ROOT}/${PHASE}/${ARM}"
    out="${save_base}/shear_flow/unet_b32_D4_W2_K${K}_ds4/${tag}/seed${SEED}"
    env DATASET=shear_flow METHOD="${method}" SEED="${SEED}" K="${K}" \
      GPUS="${GPUS}" BATCH="${SHEAR_BATCH:-1}" \
      GRAD_ACCUM="${SHEAR_GRAD_ACCUM:-4}" EPOCHS=100 \
      EARLY_STOP_PATIENCE=20 LR=3e-4 GRAD_CLIP="${GRAD_CLIP}" \
      EXTRA_ARGS="${EXTRA_ARGS}" SAVE_BASE="${save_base}" \
      SKIP_EXISTING=1 RESUME=auto bash scripts/train/run_thewell_arm.sh
    ;;

  wb2)
    method=exact; tag=exact
    if [[ "${ARM}" == dw ]]; then method=dualwiener_spectral; tag=dualwiener_spectral; fi
    out="${ROOT}/${PHASE}/${ARM}/weatherbench2/unet_c32_D4_W2_K${K}/${tag}/seed${SEED}"
    env METHOD="${method}" SEED="${SEED}" K="${K}" GPUS="${GPUS}" \
      BATCH="${WB2_TRAIN_BATCH:-1}" GRAD_ACCUM="${WB2_TRAIN_GRAD_ACCUM:-2}" \
      EPOCHS=100 EARLY_STOP_PATIENCE=20 LR=1e-4 \
      GRAD_CLIP="${GRAD_CLIP}" EXTRA_ARGS="${EXTRA_ARGS}" \
      SAVE_ROOT="${out}" SKIP_EXISTING=1 RESUME=auto \
      bash scripts/train/run_wb2_arm.sh
    eval_out="${EVAL_ROOT}/${PHASE}/${DATA}/${ARM}_K${K}_seed${SEED}.json"
    mkdir -p "$(dirname "${eval_out}")"
    if [[ "${TRAIN_ONLY}" != 1 && ! -s "${eval_out}" ]]; then
      first_gpu=${GPUS%%,*}
      CUDA_VISIBLE_DEVICES="${first_gpu}" python -u scripts/evaluate/evaluate_weatherbench2_acc.py \
        --ckpt "${out}/best.pth" --data "${WB2_DATA_PATH:-data/weatherbench2_1p5_pilot}" \
        --gpu 0 --split test --horizons 4 8 12 20 28 40 48 \
        --channels geopotential_500 temperature_850 2m_temperature \
        --batch-size "${WB2_EVAL_BATCH:-8}" --num-starts 0 --start-stride 1 \
        --out "${eval_out}"
    fi
    ;;

  fmri)
    save_base="${ROOT}/${PHASE}/${ARM}/fmri_K${K}"
    ablation=baseline
    domain_args=()
    if [[ "${ARM}" == dw ]]; then
      ablation=dualwiener_domain
      artifact_dir=${FMRI_ARTIFACT_DIR:-artifacts/internal_dw_prior_fmri_k_sweep_v1}
      artifact="${artifact_dir}/hcp_movie1_seed${SEED}_subject_crossfit_templates_K${K}.npz"
      mkdir -p "${artifact_dir}"
      if [[ ! -s "${artifact}" ]]; then
        first_gpu=${GPUS%%,*}
        CUDA_VISIBLE_DEVICES="${first_gpu}" python -u scripts/probes/probe_fmri_subject_crossfit_templates.py \
          --data-path "${HCP_DATA_PATH:-data/hcp_movie_features}" \
          --movie 1 --seed "${SEED}" --roi-dim 400 --max-horizon "${K}" \
          --device cuda --output "${artifact}"
      fi
      domain_args=(
        DUAL_WIENER_INNOVATION_FILE="${artifact}"
        DUAL_WIENER_INNOVATION_KEY=innovation_templates
        DUAL_WIENER_NOISE_MODEL=lagged_residual_bootstrap
        DUAL_WIENER_WARMUP_BATCHES=8
        DUAL_WIENER_PROBE_EVERY=4
        DUAL_WIENER_MIN_PROBES=8
        HCP_DOMAIN_TAG=SubjectCrossfit
      )
    fi
    batch=${FMRI_BATCH:-2}; accum=${FMRI_GRAD_ACCUM:-4}
    if [[ "${K}" -ge 128 && -z "${FMRI_BATCH:-}" && -z "${FMRI_GRAD_ACCUM:-}" ]]; then
      batch=1; accum=8
    fi
    for assignment in "${domain_args[@]}"; do
      export "${assignment}"
    done
    env GPUS="${GPUS}" ABLATIONS="${ablation}" SEEDS="${SEED}" \
      DEFAULT_BPTT_K="${K}" DEFAULT_TRAIN_STARTS=16 EPOCHS=100 EARLY_STOP=20 \
      BATCH="${batch}" GRAD_ACCUM="${accum}" GRAD_CLIP="${GRAD_CLIP}" \
      EXTRA_ARGS="${EXTRA_ARGS}" DATA_PATH="${HCP_DATA_PATH:-data/hcp_movie_features}" \
      SAVE_BASE="${save_base}" SKIP_EXISTING=1 RESUME=auto \
      bash scripts/train/train_hcp_resgrad_mamba_v2.sh
    out="${save_base}"
    ;;
  *) echo "[refuse] unknown DATA=${DATA}" >&2; exit 2 ;;
esac

printf 'run_id=%s\nout=%s\ncompleted=%s\n' "${run_id}" "${out}" "$(date -Is)" > "${done_file}"
echo "[done] ${run_id}; out=${out}"
