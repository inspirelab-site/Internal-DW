#!/usr/bin/env bash
set -euo pipefail

# Controlled backward-noise response of the shipped Internal-DW estimator.
# Frozen real-data checkpoints; one GPU; no model training or parameter update.

GPU="${GPU:-0}"
DATASETS="${DATASETS:-mg,ieeg}"
SNRS="${SNRS:-4,2,1,0.5,0.25}"
NOISE_DRAWS="${NOISE_DRAWS:-64}"
# The controller's residual EMA needs roughly 459 batches to exceed 99%
# effective convergence at the shipped decay.  A shorter run is a smoke test
# only and must not be used to judge the deployable plug-in estimator.
PIPELINE_BATCHES="${PIPELINE_BATCHES:-480}"
FORCE="${FORCE:-0}"
OUT_ROOT="${OUT_ROOT:-probe_outputs/real_internal_dw_injected_noise_v1}"
LOG_ROOT="${LOG_ROOT:-logs/real_internal_dw_injected_noise_v1}"

if [[ "${GPU}" == *,* || "${GPU}" == *" "* ]]; then
  echo "[fatal] GPU must be one physical id, got '${GPU}'" >&2
  exit 2
fi
if [[ "${WORLD_SIZE:-1}" != "1" ]]; then
  echo "[fatal] this route-level probe is single-process, not DDP" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"
unset DUAL_WIENER_CONST DUAL_WIENER_INNOVATION_FILE DUAL_WIENER_INNOVATION_KEY
unset RESGRAD_ALPHA_PERIOD RESGRAD_ALPHA_VALUE

contains_dataset() {
  case ",${DATASETS}," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

run_level() {
  local label="$1" ckpt="$2" data="$3" state_key="$4" stim_key="$5"
  local preprocess="$6" hidden="$7" horizon="$8" snr="$9"
  local token="${snr//./p}"
  local out="${OUT_ROOT}/${label}_snr${token}.npz"
  local log="${LOG_ROOT}/${label}_snr${token}.log"

  if [[ ! -f "${ckpt}" || ! -f "${data}" ]]; then
    echo "[fatal] missing ${label} checkpoint or data: ${ckpt} ${data}" >&2
    exit 2
  fi
  if [[ -s "${out}" && "${FORCE}" != "1" ]]; then
    echo "[skip] ${out}"
    return
  fi

  local stim_args=()
  if [[ -n "${stim_key}" ]]; then
    stim_args=(--stim-key "${stim_key}")
  fi
  echo "[start] ${label} SNR=${snr} K=${horizon} GPU=${GPU} $(date -u)"
  CUDA_VISIBLE_DEVICES="${GPU}" python -u scripts/probes/probe_wiener_oracle.py \
    --ckpt "${ckpt}" --npz "${data}" --state-key "${state_key}" \
    "${stim_args[@]}" --data-preprocess "${preprocess}" \
    --hidden "${hidden}" --depth 4 --K "${horizon}" --burnin 32 --batch 0 \
    --draws 4 --noise-draws "${NOISE_DRAWS}" \
    --signal residual --sigma-mode iso --snr "${snr}" --grid 101 \
    --pipeline "${PIPELINE_BATCHES}" --sigma-source both \
    --pipeline-reset-buffers --seed 0 --device cuda --out "${out}" \
    > "${log}" 2>&1
  echo "[done] ${out} $(date -u)"
  tail -n 12 "${log}"
}

IFS=',' read -r -a snr_values <<< "${SNRS}"

if contains_dataset mg; then
  for snr in "${snr_values[@]}"; do
    run_level mg \
      experiments/dual_wiener_screen/mackey_glass/tau30_K32/dualwiener/seed0/best.pth \
      data/synthetic/mg_D8_tau30.0_dt1.0_sdt0.1_T2048_traj40_b0.2_g0.1_n10.0_tr1000_s0.npz \
      trajs "" mackey_glass_test 128 32 "${snr}"
  done
fi

if contains_dataset ieeg; then
  for snr in "${snr_values[@]}"; do
    run_level ieeg \
      experiments/dual_wiener_screen/ieeg/theta_K64/dualwiener/seed0/best.pth \
      data/synthetic/ieeg_P41CS_enc_macro_theta_20ms_ch0.npz \
      X "" ieeg_test 256 64 "${snr}"
  done
fi

echo "[all done] $(date -u)"
