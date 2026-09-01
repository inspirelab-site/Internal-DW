#!/usr/bin/env bash
set -euo pipefail

# One matched timing arm on one physical GPU.  This is deliberately a
# single-GPU benchmark: Exact and assigned Internal-DW run consecutively on
# the same card, with identical data/model/optimizer geometry.  DDP launch
# and communication overhead therefore cannot hide the operator overhead.

REPO_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

DATA=${DATA:?set DATA=mg|narma|ieeg|fmri|ettm1|ettm2|shear|wb2}
ARM=${ARM:?set ARM=exact|dw}
REPEAT=${REPEAT:?set REPEAT}
GPU=${GPU:?set one physical GPU id}
ROOT=${ROOT:-experiments/internal_dw_compute_overhead_v2_nockpt}
EPOCHS=${EPOCHS:-4}
SEED_BASE=${SEED_BASE:-9100}
SEED=$((SEED_BASE + REPEAT))

[[ "${ARM}" == exact || "${ARM}" == dw ]] || {
  echo "[error] ARM must be exact or dw" >&2; exit 2;
}
(( EPOCHS >= 3 )) || {
  echo "[error] use EPOCHS>=3: epoch 1 is discarded as CUDA warm-up" >&2; exit 2;
}

save_base="${ROOT}/repeat${REPEAT}/${ARM}"
# Retain peak-memory logging, but avoid per-minibatch scalar .item() syncs that
# would otherwise measure diagnostics rather than the training operator.
extra_args="${EXTRA_ARGS:-} --synchronize_epoch_timing --fast_train_logging --no-log_gpu_memory"
out=""
assigned_method=""

case "${DATA}" in
  mg|narma|ettm1|ettm2|ieeg)
    case "${DATA}" in
      mg)
        dataset=mackey_glass; cond=tau30; K=32; hidden=128
        batch=${MEM_BATCH:-4}; accum=${MEM_GRAD_ACCUM:-8}
        dw_method=dualwiener_structured; dw_tag=dualwiener_structured
        ;;
      narma)
        dataset=narma; cond=L5; K=32; hidden=128
        batch=${MEM_BATCH:-4}; accum=${MEM_GRAD_ACCUM:-8}
        dw_method=dualwiener_structured; dw_tag=dualwiener_structured
        ;;
      ettm1|ettm2)
        dataset=prepared_temporal_driven; cond=${DATA}; K=64; hidden=128
        batch=${ETT_BATCH:-32}; accum=${ETT_GRAD_ACCUM:-1}
        dw_method=dualwiener_structured; dw_tag=dualwiener_structured
        export PREPARED_NPZ="${PREPARED_INPUT_ROOT:-probe_inputs/temporal_candidate_regime_v1}/${cond}.npz"
        [[ -s "${PREPARED_NPZ}" ]] || {
          echo "[missing] ${PREPARED_NPZ}" >&2; exit 3;
        }
        export MAMBA_TRAIN_STARTS=16
        ;;
      ieeg)
        dataset=ieeg; cond=theta; K=64; hidden=256
        batch=${MEM_BATCH:-4}; accum=${MEM_GRAD_ACCUM:-8}
        dw_method=dualwiener_domain; dw_tag=dualwiener_domain_ieeg_longmemory
        prior=${IEEG_PRIOR:-artifacts/domain_innovation/ieeg_theta_longmemory_factor_templates_K64.npz}
        [[ -s "${prior}" ]] || { echo "[missing] ${prior}" >&2; exit 3; }
        export DUAL_WIENER_INNOVATION_FILE="${prior}"
        export DUAL_WIENER_INNOVATION_KEY=innovation_templates
        export DUAL_WIENER_DOMAIN_NOISE_MODEL=lagged_residual_bootstrap
        export DUAL_WIENER_DOMAIN_TAG=ieeg_longmemory
        export DUAL_WIENER_MIN_PROBES=8
        ;;
    esac
    if [[ "${ARM}" == exact ]]; then
      # Runtime isolation: Exact and DW both retain the full forward graph and
      # both disable activation recomputation/checkpointing.  Only the backward
      # routing/calibration operator differs.
      method=dense; tag=dense; assigned_method="Exact BPTT"
    else
      method=${dw_method}; tag=${dw_tag}; assigned_method="assigned Internal-DW"
    fi
    out="${save_base}/${dataset}/${cond}_K${K}/${tag}/seed${SEED}"
    ;;

  shear)
    K=32
    if [[ "${ARM}" == exact ]]; then
      method=exact; tag=exact; assigned_method="Exact BPTT"
    else
      method=dualwiener_spectral; tag=dualwiener_spectral
      assigned_method="assigned Internal-DW (spectral)"
    fi
    out="${save_base}/shear_flow/unet_b32_D4_W2_K32_ds4/${tag}/seed${SEED}"
    ;;

  wb2)
    K=48
    if [[ "${ARM}" == exact ]]; then
      method=exact; tag=exact; assigned_method="Exact BPTT"
    else
      method=dualwiener_spectral; tag=dualwiener_spectral
      assigned_method="assigned Internal-DW (spectral)"
    fi
    out="${save_base}/weatherbench2/unet_c32_D4_W2_K48/${tag}/seed${SEED}"
    ;;

  fmri)
    K=64
    if [[ "${ARM}" == exact ]]; then
      ablation=baseline; tag="baseline_BPTT64_S16"; assigned_method="Exact BPTT"
    else
      ablation=dualwiener_domain; tag="resgradDualWienerDomainSubjectCrossfit_BPTT64_S16"
      assigned_method="assigned Internal-DW (subject cross-fit prior)"
      prior=${FMRI_PRIOR:-artifacts/internal_dw_prior_fmri_v1/hcp_movie1_seed0_subject_crossfit_templates_K64.npz}
      [[ -s "${prior}" ]] || { echo "[missing] ${prior}" >&2; exit 3; }
      export DUAL_WIENER_INNOVATION_FILE="${prior}"
      export DUAL_WIENER_INNOVATION_KEY=innovation_templates
      export DUAL_WIENER_NOISE_MODEL=lagged_residual_bootstrap
      export DUAL_WIENER_MIN_PROBES=8
      export HCP_DOMAIN_TAG=SubjectCrossfit
    fi
    out="${save_base}/official_mamba_state_hid4096_D4_residual_${tag}/seed${SEED}"
    ;;

  *) echo "[error] unsupported DATA=${DATA}" >&2; exit 2 ;;
esac

done_marker="${out}/.timing_complete"
if [[ -f "${done_marker}" ]]; then
  echo "[skip] completed timing arm: ${out}"
  exit 0
fi
if [[ -s "${out}/train_logs.jsonl" ]]; then
  echo "[refuse] incomplete timing output already exists: ${out}" >&2
  echo "         use a new ROOT=... or move this directory aside before rerunning" >&2
  exit 4
fi
mkdir -p "${out}"

gpu_name=$(nvidia-smi --id="${GPU}" --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)
export TIMING_META_OUT="${out}/timing_meta.json"
export TIMING_META_DATA="${DATA}" TIMING_META_ARM="${ARM}"
export TIMING_META_METHOD="${assigned_method}" TIMING_META_REPEAT="${REPEAT}"
export TIMING_META_SEED="${SEED}" TIMING_META_GPU="${GPU}"
export TIMING_META_GPU_NAME="${gpu_name}" TIMING_META_EPOCHS="${EPOCHS}"
export TIMING_META_K="${K}"
python - <<'PY'
import json, os, platform
from pathlib import Path
try:
    import torch
    torch_version = torch.__version__
    cuda_version = torch.version.cuda
except Exception:
    torch_version = cuda_version = "unknown"
out = Path(os.environ["TIMING_META_OUT"])
payload = {
    "dataset": os.environ["TIMING_META_DATA"],
    "arm": os.environ["TIMING_META_ARM"],
    "method": os.environ["TIMING_META_METHOD"],
    "repeat": int(os.environ["TIMING_META_REPEAT"]),
    "seed": int(os.environ["TIMING_META_SEED"]),
    "gpu_id": os.environ["TIMING_META_GPU"],
    "gpu_name": os.environ["TIMING_META_GPU_NAME"].strip(),
    "epochs": int(os.environ["TIMING_META_EPOCHS"]),
    "training_horizon": int(os.environ["TIMING_META_K"]),
    "torch": torch_version,
    "cuda": cuda_version,
    "host": platform.node(),
    "timing_protocol": "single GPU; no activation checkpointing; CUDA synchronized; epoch 1 discarded",
    "activation_checkpointing": False,
}
out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

echo "[timing] data=${DATA} arm=${ARM} repeat=${REPEAT} gpu=${GPU} out=${out}"

# Bash occasionally receives EBADF while opening a script directly on the
# CIFS research mount.  Copy the selected high-level runner to node-local
# storage with retries; PROJECT_ROOT keeps all relative paths anchored here.
localize_runner() {
  local src=$1 dst attempt
  dst="/tmp/internal_dw_timing_$(basename "${src}")_${USER:-user}_$$_${RANDOM}.sh"
  for attempt in 1 2 3 4 5 6 7 8; do
    if cp "${src}" "${dst}" 2>/dev/null; then
      chmod 700 "${dst}"
      printf '%s\n' "${dst}"
      return 0
    fi
    echo "[retry] could not read ${src} (attempt ${attempt}/8)" >&2
    sleep 1
  done
  echo "[error] failed to localize ${src}" >&2
  return 1
}

case "${DATA}" in
  mg|narma|ettm1|ettm2|ieeg) child_runner=$(localize_runner scripts/train/run_mem_one.sh) ;;
  shear) child_runner=$(localize_runner scripts/train/run_thewell_arm.sh) ;;
  wb2) child_runner=$(localize_runner scripts/train/run_wb2_arm.sh) ;;
  fmri) child_runner=$(localize_runner scripts/train/train_hcp_resgrad_mamba_v2.sh) ;;
esac
cleanup_child_runner() { rm -f -- "${child_runner}"; }
trap cleanup_child_runner EXIT

case "${DATA}" in
  mg|narma|ettm1|ettm2|ieeg)
    env DATASET="${dataset}" COND="${cond}" K="${K}" METHOD="${method}" \
      SEED="${SEED}" GPU="${GPU}" GPUS="${GPU}" HIDDEN="${hidden}" \
      BATCH="${batch}" GRAD_ACCUM="${accum}" NUM_WORKERS="${NUM_WORKERS:-0}" \
      EPOCHS="${EPOCHS}" ES=0 LR=1e-4 RUN_MODE=train RESUME="" \
      SAVE_BASE="${save_base}" SKIP_EXISTING=0 FAST_TRAIN_RUNTIME=0 \
      EXTRA_ARGS="${extra_args}" PROJECT_ROOT="${REPO_ROOT}" bash "${child_runner}"
    ;;
  shear)
    env DATASET=shear_flow METHOD="${method}" SEED="${SEED}" K=32 \
      GPU="${GPU}" GPUS="${GPU}" BATCH="${SHEAR_BATCH:-1}" \
      GRAD_ACCUM="${SHEAR_GRAD_ACCUM:-4}" NUM_WORKERS="${NUM_WORKERS:-2}" \
      EPOCHS="${EPOCHS}" EARLY_STOP_PATIENCE=0 LR=3e-4 RUN_MODE=train \
      RESUME="" SAVE_BASE="${save_base}" SKIP_EXISTING=0 FAST_TRAIN_RUNTIME=0 \
      MATCHED_NO_ACTIVATION_CKPT=1 \
      EXTRA_ARGS="${extra_args}" PROJECT_ROOT="${REPO_ROOT}" bash "${child_runner}"
    ;;
  wb2)
    env METHOD="${method}" SEED="${SEED}" K=48 GPU="${GPU}" GPUS="${GPU}" \
      BATCH="${WB2_BATCH:-1}" GRAD_ACCUM="${WB2_GRAD_ACCUM:-2}" \
      NUM_WORKERS="${NUM_WORKERS:-2}" EPOCHS="${EPOCHS}" \
      EARLY_STOP_PATIENCE=0 MODE=train RESUME="" SAVE_ROOT="${out}" \
      MATCHED_NO_ACTIVATION_CKPT=1 \
      EXTRA_ARGS="${extra_args}" PROJECT_ROOT="${REPO_ROOT}" bash "${child_runner}"
    ;;
  fmri)
    env GPUS="${GPU}" ABLATIONS="${ablation}" SEEDS="${SEED}" \
      DEFAULT_BPTT_K=64 DEFAULT_TRAIN_STARTS=16 EPOCHS="${EPOCHS}" \
      EARLY_STOP=0 BATCH="${FMRI_BATCH:-2}" GRAD_ACCUM="${FMRI_GRAD_ACCUM:-4}" \
      MODE=train RESUME="" SAVE_BASE="${save_base}" SKIP_EXISTING=0 \
      EXTRA_ARGS="${extra_args}" \
      DATA_PATH="${HCP_DATA_PATH:-data/hcp_movie_features}" \
      PROJECT_ROOT="${REPO_ROOT}" bash "${child_runner}"
    ;;
esac

touch "${done_marker}"
echo "[done] timing ${DATA}/${ARM}/repeat${REPEAT}"
