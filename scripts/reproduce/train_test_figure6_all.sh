#!/usr/bin/env bash
# Sequential, resumable one-GPU reproduction of every Figure 6 training arm.
set -euo pipefail

REPO_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd -P)}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# shellcheck source=/dev/null
source "${REPO_ROOT}/configs/reproduce/figure6.sh"

GPU=${GPU:-0}
SEEDS_TEXT=${SEEDS:-"0 1 2"}
read -r -a FIGURE6_SEEDS <<< "${SEEDS_TEXT}"
for seed in "${FIGURE6_SEEDS[@]}"; do
  case "${seed}" in 0|1|2) ;; *) echo "SEEDS must contain only 0, 1, 2" >&2; exit 2 ;; esac
done

FIGURE6_TRAIN_ROOT=${FIGURE6_TRAIN_ROOT:-experiments/figure6_reproduction_v1}
FIGURE6_SYNC_ROOT=${FIGURE6_SYNC_ROOT:-artifacts/figure6_reproduction_v1_sync}
FIGURE6_AB_OUT_ROOT=${FIGURE6_AB_OUT_ROOT:-probe_outputs/internal_dw_AB_dense_1p5k_v1}
FIGURE6_STATIC_ROOT=${FIGURE6_STATIC_ROOT:-experiments/internal_dw_static_positive_v1}
FIGURE6_TBPTT_ROOT=${FIGURE6_TBPTT_ROOT:-experiments/tbptt_positive_sweep_v1}
REPRO_DATASETS=${REPRO_DATASETS:-}
FIGURE6_FAILURE_REPORT=${FIGURE6_FAILURE_REPORT:-probe_outputs/figure6_reproduction_failures.txt}

wants_dataset() {
  local candidate=$1
  [[ -z "${REPRO_DATASETS}" ]] && return 0
  local item
  for item in ${REPRO_DATASETS}; do
    [[ "${item}" == "${candidate}" ]] && return 0
  done
  return 1
}

primary_result_path() {
  local data=$1 arm=$2 seed=$3
  case "${data}:${arm}" in
    mg:full_bptt) echo "probe_outputs/dense_multistart_rel_l2_1p5k_v1/mg/exact_seed${seed}.json" ;;
    mg:internal_dw) echo "probe_outputs/dense_multistart_rel_l2_1p5k_v1/mg/dw_seed${seed}.json" ;;
    ettm1:full_bptt|ettm2:full_bptt) echo "probe_outputs/temporal_candidate_dense_1p5k_v1/${data}/exact_seed${seed}.json" ;;
    ettm1:internal_dw|ettm2:internal_dw) echo "probe_outputs/temporal_candidate_dense_1p5k_v1/${data}/generic_seed${seed}.json" ;;
    shear:full_bptt|narma:full_bptt|ieeg:full_bptt|fmri:full_bptt|wb2:full_bptt)
      echo "probe_outputs/dense_multistart_rel_l2_1p5k_v1/${data}/exact_seed${seed}.json" ;;
    shear:internal_dw|narma:internal_dw|ieeg:internal_dw|fmri:internal_dw|wb2:internal_dw)
      echo "probe_outputs/dense_multistart_rel_l2_1p5k_v1/${data}/dw_seed${seed}.json" ;;
    *) echo "unsupported primary result ${data}:${arm}" >&2; return 2 ;;
  esac
}

control_result_path() {
  local data=$1 arm=$2 seed=$3
  local k=${FIGURE6_TRAIN_HORIZON[$data]}
  case "${arm}" in
    clip|jreg) echo "${FIGURE6_AB_OUT_ROOT}/A/${data}/${arm}_K${k}_seed${seed}.json" ;;
    static) echo "probe_outputs/internal_dw_static_positive_v1/dense/${data}/static_c${FIGURE6_STATIC_GAIN[$data]}_seed${seed}.json" ;;
    tbptt)
      case "${data}" in
        mg|shear) echo "probe_outputs/tbptt_dense_multistart_rel_l2_1p5k_v1/${data}/tbptt${FIGURE6_TBPTT_SEGMENT[$data]}_seed${seed}.json" ;;
        ettm1|ettm2) echo "probe_outputs/tbptt_positive_sweep_v1/dense_candidates/${data}/tbptt${FIGURE6_TBPTT_SEGMENT[$data]}_seed${seed}.json" ;;
      esac
      ;;
    *) echo "unsupported control arm ${arm}" >&2; return 2 ;;
  esac
}

run_phase_arm() {
  local phase=$1 data=$2 arm=$3 seed=$4 target=$5
  local k=${FIGURE6_TRAIN_HORIZON[$data]}
  local batch=${FIGURE6_SINGLE_LOCAL_BATCH[$data]}
  local accum=${FIGURE6_SINGLE_GRAD_ACCUM[$data]}
  if [[ -s "${target}" ]]; then
    echo "[skip result] ${target}"
    return 0
  fi
  mkdir -p "$(dirname "${target}")"
  env PHASE="${phase}" DATA="${data}" ARM="${arm}" K="${k}" SEED="${seed}" \
    GPUS="${GPU}" ROOT="${FIGURE6_TRAIN_ROOT}" SYNC_ROOT="${FIGURE6_SYNC_ROOT}" \
    EVAL_ROOT="${FIGURE6_AB_OUT_ROOT}" \
    MEM_BATCH="${batch}" MEM_GRAD_ACCUM="${accum}" \
    ETT_BATCH="${batch}" ETT_GRAD_ACCUM="${accum}" \
    SHEAR_BATCH="${batch}" SHEAR_GRAD_ACCUM="${accum}" \
    FMRI_BATCH="${batch}" FMRI_GRAD_ACCUM="${accum}" \
    WB2_TRAIN_BATCH="${batch}" WB2_TRAIN_GRAD_ACCUM="${accum}" \
      bash scripts/train/run_internal_dw_baseline_or_k_one.sh || return $?
  env PHASE="${phase}" DATA="${data}" ARM="${arm}" K="${k}" SEED="${seed}" \
    GPU="${GPU}" ROOT="${FIGURE6_TRAIN_ROOT}" SYNC_ROOT="${FIGURE6_SYNC_ROOT}" \
    OUT_ROOT="${FIGURE6_AB_OUT_ROOT}" OUT="${target}" \
      bash scripts/evaluate/run_internal_dw_AB_dense_eval_1p5k_one.sh || return $?
  [[ -s "${target}" ]] || { echo "[failed] missing test JSON ${target}" >&2; return 3; }
}

run_primary() {
  local data=$1 arm=$2 seed=$3
  local target
  target=$(primary_result_path "${data}" "${arm}" "${seed}")
  if [[ -s "${target}" ]]; then
    echo "[skip result] ${target}"
    return 0
  fi
  if [[ "${data}" == mg ]]; then
    ARM="${arm}" SEED="${seed}" GPU="${GPU}" \
      bash scripts/reproduce/train_test_mg.sh || return $?
  else
    local phase_arm=exact
    [[ "${arm}" == internal_dw ]] && phase_arm=dw
    run_phase_arm B "${data}" "${phase_arm}" "${seed}" "${target}"
  fi
}

evaluate_dense_control() {
  local ckpt=$1 target=$2 data=$3 method_label=$4
  local k=${FIGURE6_TRAIN_HORIZON[$data]}
  local h=$(((3 * k + 1) / 2))
  local origin_batch=16
  case "${data}" in
    ettm1|ettm2) origin_batch=8 ;;
    shear) origin_batch=2 ;;
  esac
  [[ -s "${target}" ]] && { echo "[skip result] ${target}"; return 0; }
  [[ -s "${ckpt}" ]] || { echo "[missing checkpoint] ${ckpt}" >&2; return 3; }
  mkdir -p "$(dirname "${target}")"
  CUDA_VISIBLE_DEVICES="${GPU}" python -u scripts/evaluate/evaluate_dense_multistart_rel_l2.py \
    --ckpt "${ckpt}" --out "${target}" --gpu 0 --split test \
    --max-horizon "${h}" --train-horizon "${k}" \
    --origin-stride 1 --max-origins-per-item 64 \
    --origin-batch "${origin_batch}" --num-workers 0 --bootstrap-draws 10000 \
    --method-label "${method_label}" || return $?
}

run_static() {
  local data=$1 seed=$2
  local gain=${FIGURE6_STATIC_GAIN[$data]}
  local target ckpt
  target=$(control_result_path "${data}" static "${seed}")
  [[ -s "${target}" ]] && { echo "[skip result] ${target}"; return 0; }
  DATA="${data}" GAIN="${gain}" SEED="${seed}" GPUS="${GPU}" \
    TRAIN_ROOT="${FIGURE6_STATIC_ROOT}" \
      bash scripts/train/run_positive_static_gain_one.sh || return $?
  case "${data}" in
    mg) ckpt="${FIGURE6_STATIC_ROOT}/mackey_glass/tau30_K32/dwc${gain}/seed${seed}/best.pth" ;;
    ettm1|ettm2) ckpt="${FIGURE6_STATIC_ROOT}/prepared_temporal_driven/${data}_K64/dwc${gain}/seed${seed}/best.pth" ;;
    shear) ckpt="${FIGURE6_STATIC_ROOT}/shear_flow/unet_b32_D4_W2_K32_ds4/dwc${gain}/seed${seed}/best.pth" ;;
  esac
  evaluate_dense_control "${ckpt}" "${target}" "${data}" static_gain || return $?
}

run_tbptt() {
  local data=$1 seed=$2
  local segment=${FIGURE6_TBPTT_SEGMENT[$data]}
  local target ckpt legacy
  target=$(control_result_path "${data}" tbptt "${seed}")
  [[ -s "${target}" ]] && { echo "[skip result] ${target}"; return 0; }
  DATA="${data}" S="${segment}" SEED="${seed}" GPU="${GPU}" \
    SWEEP_ROOT="${FIGURE6_TBPTT_ROOT}" \
      bash scripts/train/run_tbptt_positive_sweep_one.sh || return $?
  case "${data}" in
    mg)
      legacy="experiments/tbptt_mg_a8/mackey_glass/tau30_K32/ckpt/seed${seed}/best.pth"
      ckpt="${FIGURE6_TBPTT_ROOT}/mg/mackey_glass/tau30_K32/tbptt${segment}/seed${seed}/best.pth"
      [[ -s "${legacy}" ]] && ckpt="${legacy}"
      ;;
    ettm1|ettm2) ckpt="${FIGURE6_TBPTT_ROOT}/${data}/prepared_temporal_driven/${data}_K64/tbptt${segment}/seed${seed}/best.pth" ;;
    shear)
      legacy="experiments/thewell_shear_final_lr3e4/shear_flow/unet_b32_D4_W2_K32_ds4/tbptt8/seed${seed}/best.pth"
      ckpt="${FIGURE6_TBPTT_ROOT}/shear/shear_flow/unet_b32_D4_W2_K32_ds4/tbptt${segment}/seed${seed}/best.pth"
      [[ -s "${legacy}" ]] && ckpt="${legacy}"
      ;;
  esac
  evaluate_dense_control "${ckpt}" "${target}" "${data}" tbptt || return $?
}

run_dataset() {
  local data=$1 kind=$2
  if [[ "${data}" == ieeg ]]; then
    env GPU="${GPU}" SEEDS="${FIGURE6_SEEDS[*]}" ARM="" SUBJECTS="" \
      bash scripts/reproduce/train_test_ieeg.sh
    return $?
  fi
  local -a arms
  if [[ "${kind}" == positive ]]; then
    arms=("${FIGURE6_POSITIVE_ARMS[@]}")
  else
    arms=("${FIGURE6_BOUNDARY_ARMS[@]}")
  fi
  local arm seed target phase_arm
  for arm in "${arms[@]}"; do
    for seed in "${FIGURE6_SEEDS[@]}"; do
      echo "===== Figure 6: ${data} / ${arm} / seed ${seed} ====="
      case "${arm}" in
        full_bptt|internal_dw)
          run_primary "${data}" "${arm}" "${seed}" || return $?
          ;;
        clip|jreg)
          target=$(control_result_path "${data}" "${arm}" "${seed}") || return $?
          run_phase_arm A "${data}" "${arm}" "${seed}" "${target}" || return $?
          ;;
        static) run_static "${data}" "${seed}" || return $? ;;
        tbptt) run_tbptt "${data}" "${seed}" || return $? ;;
      esac
    done
  done
}

echo "[queue] GPU=${GPU}; seeds=${FIGURE6_SEEDS[*]}"
echo "[queue] positive datasets first: ${FIGURE6_POSITIVE_DATASETS[*]}"
FIGURE6_FAILURES=()
run_dataset_isolated() {
  local data=$1 kind=$2 rc
  if run_dataset "${data}" "${kind}"; then
    echo "[dataset done] ${data}"
    return 0
  else
    rc=$?
    FIGURE6_FAILURES+=("${data}:${rc}")
    echo "[dataset failed] ${data} (rc=${rc}); continuing with the next dataset" >&2
    return 0
  fi
}

for data in "${FIGURE6_POSITIVE_DATASETS[@]}"; do
  wants_dataset "${data}" && run_dataset_isolated "${data}" positive
done
for data in "${FIGURE6_BOUNDARY_DATASETS[@]}"; do
  wants_dataset "${data}" && run_dataset_isolated "${data}" boundary
done

mkdir -p "$(dirname "${FIGURE6_FAILURE_REPORT}")"
{
  echo "updated_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if (( ${#FIGURE6_FAILURES[@]} == 0 )); then
    echo "state=DONE"
  else
    echo "state=PARTIAL"
    printf 'failed_dataset=%s\n' "${FIGURE6_FAILURES[@]}"
  fi
} > "${FIGURE6_FAILURE_REPORT}"

if (( ${#FIGURE6_FAILURES[@]} == 0 )) \
   && [[ -z "${REPRO_DATASETS}" && "${RENDER_FIGURE6:-1}" == 1 ]]; then
  bash scripts/reproduce/05_figure_6.sh
fi
if (( ${#FIGURE6_FAILURES[@]} > 0 )); then
  echo "[partial] all datasets were attempted; failures: ${FIGURE6_FAILURES[*]}" >&2
  echo "[failure report] ${FIGURE6_FAILURE_REPORT}" >&2
  echo "[resume] fix the reported dataset inputs and rerun the same command" >&2
  exit 1
fi
echo "[done] Figure 6 training and testing queue"
