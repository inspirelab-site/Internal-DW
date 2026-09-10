#!/usr/bin/env bash
set -euo pipefail

# Recompute every frozen-checkpoint source consumed by Figure 5(b).
# The probe is single-process because it measures route-level VJPs.  Set
# DATASETS to a comma-separated subset to distribute cases across servers.

cd "$(dirname "$0")/../.."
GPU="${GPU:-0}"
DATASETS="${DATASETS:-mg,narma,ettm1,ettm2,ieeg,fmri,shear,wb2}"
SNRS="${SNRS:-4,1,0.25}"
FORCE="${FORCE:-0}"
PIPELINE_BATCHES="${PIPELINE_BATCHES:-480}"
NOISE_DRAWS="${NOISE_DRAWS:-64}"
PRIOR_NOISE_DRAWS="${PRIOR_NOISE_DRAWS:-8}"
HCP_DATA_PATH="${HCP_DATA_PATH:-data/hcp_movie_features}"

if [[ "${GPU}" == *,* || "${GPU}" == *" "* || "${WORLD_SIZE:-1}" != 1 ]]; then
  echo "[fatal] Figure 5(b) probes require one physical GPU id" >&2
  exit 2
fi

contains_dataset() {
  case ",${DATASETS}," in *",$1,"*) return 0 ;; *) return 1 ;; esac
}

require_file() {
  [[ -s "$1" ]] || { echo "[fatal] missing $1" >&2; exit 3; }
}

run_vector() {
  local dataset="$1" label ckpt data state_key stim_key preprocess hidden horizon out_root
  case "${dataset}" in
    mg)
      label=mg
      ckpt=experiments/dual_wiener_screen/mackey_glass/tau30_K32/dualwiener/seed0/best.pth
      data=data/synthetic/mg_D8_tau30.0_dt1.0_sdt0.1_T2048_traj40_b0.2_g0.1_n10.0_tr1000_s0.npz
      state_key=trajs; stim_key=""; preprocess=mackey_glass_test; hidden=128; horizon=32
      out_root=probe_outputs/real_internal_dw_estimator_sweep_v1/mg_structured
      ;;
    narma)
      label=narma
      ckpt=experiments/dual_wiener_screen/narma/L5_K32/dualwiener/seed0/best.pth
      data=data/synthetic/narma_D8_L5_T2048_traj40_tr200_u0.5_bd1_dr1.5_s0.npz
      state_key=y; stim_key=u; preprocess=narma_test; hidden=128; horizon=32
      out_root=probe_outputs/real_internal_dw_estimator_sweep_v1/structured_40gb
      ;;
  esac
  require_file "${ckpt}"; require_file "${data}"
  mkdir -p "${out_root}" "logs/application_added_noise_v1/${label}"
  IFS=',' read -r -a levels <<< "${SNRS}"
  for snr in "${levels[@]}"; do
    local token="${snr//./p}"
    local out="${out_root}/${label}_structured_snr${token}.npz"
    local log="logs/application_added_noise_v1/${label}/snr${token}.log"
    if [[ -s "${out}" && "${FORCE}" != 1 ]]; then echo "[skip] ${out}"; continue; fi
    local stim_args=()
    [[ -n "${stim_key}" ]] && stim_args=(--stim-key "${stim_key}")
    CUDA_VISIBLE_DEVICES="${GPU}" python -u scripts/probes/probe_wiener_oracle.py \
      --ckpt "${ckpt}" --npz "${data}" --state-key "${state_key}" \
      "${stim_args[@]}" --data-preprocess "${preprocess}" \
      --hidden "${hidden}" --depth 4 --K "${horizon}" --burnin 32 --batch 0 \
      --dual-wiener-noise-model lagged_residual_bootstrap \
      --draws 4 --noise-draws "${NOISE_DRAWS}" --signal residual \
      --sigma-mode iso --snr "${snr}" --grid 101 \
      --pipeline "${PIPELINE_BATCHES}" --sigma-source plugin \
      --pipeline-reset-buffers --seed 0 --device cuda --out "${out}" \
      > "${log}" 2>&1
    echo "[done] ${out}"
  done
}

run_prior() {
  if [[ "$1" == ieeg ]]; then
    GPU="${GPU}" bash scripts/reproduce/train_test_ieeg.sh noise
    return
  fi
  local dataset="$1" ckpt artifact out_root probe data_args=()
  case "${dataset}" in
    fmri)
      ckpt=experiments/hcp_movie1/internal_dw_prior_fmri_ddp4_v1/official_mamba_state_hid4096_D4_residual_resgradDualWienerDomainSubjectCrossfit_BPTT64_S16/seed0/best.pth
      artifact=artifacts/internal_dw_prior_fmri_v1/hcp_movie1_seed0_subject_crossfit_templates_K64.npz
      out_root=probe_outputs/real_internal_dw_prior_noise_response_v1/fmri
      probe=scripts/probes/probe_fmri_prior_noise_response.py
      [[ -d "${HCP_DATA_PATH}" ]] || {
        echo "[fatal] missing HCP data directory: ${HCP_DATA_PATH}" >&2
        echo "Set HCP_DATA_PATH=/path/to/hcp_movie_features and rerun." >&2
        exit 3
      }
      data_args=(--hcp-dir "${HCP_DATA_PATH}" --hcp-split val --movie 1 --roi-dim 400 --data-preprocess none --hidden 4096 --depth 4 --K 64 --burnin 32 --batch 8 --draws 4)
      ;;
  esac
  require_file "${ckpt}"; require_file "${artifact}"
  mkdir -p "${out_root}" "logs/application_added_noise_v1/${dataset}"
  IFS=',' read -r -a levels <<< "${SNRS}"
  for snr in "${levels[@]}"; do
    local token="${snr//./p}" suffix=""
    [[ "${dataset}" == fmri ]] && suffix=_seed0
    local out="${out_root}/${dataset}_prior_snr${token}${suffix}.json"
    local log="logs/application_added_noise_v1/${dataset}/snr${token}.log"
    if [[ -s "${out}" && "${FORCE}" != 1 ]]; then echo "[skip] ${out}"; continue; fi
    CUDA_VISIBLE_DEVICES="${GPU}" python -u "${probe}" \
      --ckpt "${ckpt}" --artifact "${artifact}" "${data_args[@]}" \
      --dual-wiener-noise-model lagged_residual_bootstrap \
      --snr "${snr}" --noise-draws "${PRIOR_NOISE_DRAWS}" \
      --seed 0 --device cuda --out "${out}" > "${log}" 2>&1
    echo "[done] ${out}"
  done
}

run_spectral() {
  local dataset="$1" ckpt horizon fit probe noise out_root prefix
  case "${dataset}" in
    shear)
      ckpt=experiments/thewell_shear_final_lr3e4/shear_flow/unet_b32_D4_W2_K32_ds4/dualwiener_spectral/seed0/best.pth
      horizon=32; fit=48; probe=8; noise=8
      out_root=probe_outputs/real_internal_dw_estimator_sweep_v2/shear
      prefix=shear
      ;;
    wb2)
      ckpt=experiments/weatherbench2/spectral_fullbudget_seed0/unet_c32_D4_W2_K48/dualwiener_spectral/seed0/best.pth
      horizon=48; fit=32; probe=4; noise=4
      out_root=probe_outputs/real_internal_dw_estimator_sweep_v2/wb2
      prefix=wb2_spectral_seed0
      ;;
  esac
  require_file "${ckpt}"
  mkdir -p "${out_root}" "logs/application_added_noise_v1/${dataset}"
  IFS=',' read -r -a levels <<< "${SNRS}"
  for snr in "${levels[@]}"; do
    local token="${snr//./p}"
    local out="${out_root}/${prefix}_snr${token}.json"
    local log="logs/application_added_noise_v1/${dataset}/snr${token}.log"
    if [[ -s "${out}" && "${FORCE}" != 1 ]]; then echo "[skip] ${out}"; continue; fi
    CUDA_VISIBLE_DEVICES="${GPU}" python -u scripts/probes/probe_thewell_spectral_noise_response.py \
      --ckpt "${ckpt}" --K "${horizon}" --snr "${snr}" \
      --fit-trajectories "${fit}" --probe-trajectories "${probe}" \
      --noise-draws "${noise}" --seed 0 --gpu 0 --out "${out}" \
      > "${log}" 2>&1
    echo "[done] ${out}"
  done
}

unset DUAL_WIENER_CONST DUAL_WIENER_INNOVATION_FILE DUAL_WIENER_INNOVATION_KEY
unset RESGRAD_ALPHA_PERIOD RESGRAD_ALPHA_VALUE

for dataset in mg narma; do
  contains_dataset "${dataset}" && run_vector "${dataset}"
done
for dataset in ettm1 ettm2; do
  if contains_dataset "${dataset}"; then
    IFS=',' read -r -a levels <<< "${SNRS}"
    for snr in "${levels[@]}"; do
      DATA="${dataset}" TASK=noise SNR="${snr}" GPU="${GPU}" FORCE="${FORCE}" \
        PIPELINE_BATCHES="${PIPELINE_BATCHES}" NOISE_DRAWS="${NOISE_DRAWS}" \
        bash scripts/probes/run_ettm_internal_dw_diagnostic_one.sh
    done
  fi
done
for dataset in ieeg fmri; do
  contains_dataset "${dataset}" && run_prior "${dataset}"
done
for dataset in shear wb2; do
  contains_dataset "${dataset}" && run_spectral "${dataset}"
done

# A subset is useful for distributing cases across machines; assemble only
# when this invocation requested the full panel. Partial workers leave their
# idempotent raw outputs for the final builder invocation.
complete=1
for dataset in mg narma ettm1 ettm2 ieeg fmri shear wb2; do
  contains_dataset "${dataset}" || complete=0
done
if [[ "${complete}" == 1 ]]; then
  python scripts/results/build_added_noise_results.py
else
  echo "[partial] run python scripts/results/build_added_noise_results.py after all datasets finish"
fi
