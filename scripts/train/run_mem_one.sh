#!/usr/bin/env bash
# =============================================================================
# ONE arm of the memory-length testbed sweep, on ONE card.
#
#   DATASET=narma COND=L20 K=32 METHOD=ckpt SEED=0 GPU=0 \
#     nohup bash scripts/train/run_mem_one.sh > /dev/null 2>&1 &
#
# DATASET: mackey_glass | mackey_glass_driven | narma | ieeg | gait | known_snr_ar |
#          prepared_temporal_autonomous | prepared_temporal_driven
# COND:    the memory-length knob for that dataset (see the case block below).
#          Each dataset has a LONG-memory and a SHORT-memory setting; the point
#          of the sweep is to compare arms at a FIXED K while K* changes.
# METHOD:  ckpt (exact gradient) | tbptt<S> | artbp<L> | none (g=0) |
#          p<frac> (calibrated, e.g. p0.5)
#
# Idempotent: exits if eval_results.json exists (SKIP_EXISTING=0 to force).
# --resume auto picks up last.pth if interrupted, then auto-tests.
# Log: mem_<DATASET>_<COND>_<TAG>_s<SEED>.log in the repo root.
# =============================================================================
set -uo pipefail
REPO_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd -P)}"
cd "${REPO_ROOT}"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export MASTER_ADDR=127.0.0.1
# Some shared clusters export a system MPS pipe even when its daemon is not
# healthy. Point CUDA at a private empty pipe directory by default so jobs use
# ordinary CUDA contexts. Set USE_MPS=1 only when the host MPS service is known
# to be healthy.
if [[ "${USE_MPS:-0}" != "1" ]]; then
  export CUDA_MPS_PIPE_DIRECTORY="$(pwd)/.mps_bypass"
fi
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY"

DATASET=${DATASET:?set DATASET (mackey_glass|mackey_glass_driven|narma|ieeg|gait|known_snr_ar|prepared_temporal_autonomous|prepared_temporal_driven)}
COND=${COND:?set COND (see script header)}
K=${K:?set K}
METHOD=${METHOD:?set METHOD (ckpt|none|p<frac>)}
SEED=${SEED:?set SEED}
GPU=${GPU:-}
GPUS=${GPUS:-${GPU}}
if [[ -z "${GPUS}" ]]; then
  echo "set GPU=<id> for one GPU or GPUS=<comma-separated ids> for DDP" >&2
  exit 2
fi
FIRST_GPU=${GPUS%%,*}
BATCH=${BATCH:-4}
GRAD_ACCUM=${GRAD_ACCUM:-8}
NUM_WORKERS=${NUM_WORKERS:-4}
EPOCHS=${EPOCHS:-100}
ES=${ES:-20}
LR=${LR:-1e-4}
AR_OPTIMIZER=${AR_OPTIMIZER:-adam}
AR_SGD_MOMENTUM=${AR_SGD_MOMENTUM:-0.0}
AR_SGD_NESTEROV=${AR_SGD_NESTEROV:-0}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
GRAD_CLIP=${GRAD_CLIP:-1.0}
AR_SCHEDULER=${AR_SCHEDULER:-step}
AR_MIN_LR=${AR_MIN_LR:-1e-6}
MAMBA_LOSS_TYPE=${MAMBA_LOSS_TYPE:-rel_l2}
MAMBA_TRAIN_STARTS=${MAMBA_TRAIN_STARTS:-16}
DATA_DIR=${DATA_DIR:-"data/synthetic"}
SAVE_BASE=${SAVE_BASE:-"experiments/memtest"}
SKIP_EXISTING=${SKIP_EXISTING:-1}
RESUME=${RESUME:-auto}
RUN_MODE=${RUN_MODE:-train_and_test}
if [[ "${AR_SGD_NESTEROV}" == "1" ]]; then
  SGD_NESTEROV_FLAG=(--ar_sgd_nesterov)
else
  SGD_NESTEROV_FLAG=(--no-ar_sgd_nesterov)
fi
if [[ "${GLOBAL_WIENER_BATCH_CONDITIONED:-1}" == "1" ]]; then
  GLOBAL_WIENER_BATCH_FLAG=(--global_wiener_batch_conditioned)
else
  GLOBAL_WIENER_BATCH_FLAG=(--no-global_wiener_batch_conditioned)
fi
if [[ "${AR_SHARED_ROLLOUT_START:-0}" == "1" ]]; then
  AR_SHARED_START_FLAG=(--ar_shared_rollout_start)
else
  AR_SHARED_START_FLAG=(--no-ar_shared_rollout_start)
fi

# Self-contained fast-evaluation default.  All datasets handled by this
# launcher are vector sequences.  Use the whole nine-horizon sweep on >=80-GB
# accelerators and a conservative group of three otherwise.  An explicit
# RECURRENT_EVAL_HORIZON_BATCH from the caller always wins.
if [[ -z "${RECURRENT_EVAL_HORIZON_BATCH:-}" ]]; then
  GPU_TOTAL_MB=$(nvidia-smi --id="${FIRST_GPU}" --query-gpu=memory.total \
    --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]' || true)
  if [[ "${GPU_TOTAL_MB}" =~ ^[0-9]+$ ]] && (( GPU_TOTAL_MB >= 80000 )); then
    RECURRENT_EVAL_HORIZON_BATCH=9
  else
    RECURRENT_EVAL_HORIZON_BATCH=3
  fi
fi
export RECURRENT_EVAL_HORIZON_BATCH

# ---- per-dataset knobs -------------------------------------------------------
# HIDDEN is set per dataset because the state dimensions differ by ~80x
# (gait D=1, MG/NARMA D=8, iEEG D=80) and Lorenz's hidden 512 would be absurd on
# a scalar series.  It MUST stay fixed within a dataset: the long/short-memory
# comparison is only controlled if capacity is identical across the two arms.
case "${DATASET}" in
  mackey_glass_driven)
    HIDDEN=${HIDDEN:-128}
    case "${COND}" in
      tau30_ds0p08_rho0p9)
        DRIVEN_MG_NPZ=${DRIVEN_MG_NPZ:-artifacts/driven_mg_regime/driven_mg_tau30_ds0.08_rho0.9_seed2027.npz}
        [[ -f "${DRIVEN_MG_NPZ}" ]] || {
          echo "formal Driven-MG archive not found: ${DRIVEN_MG_NPZ}" >&2; exit 2;
        }
        DS=(--mg_tau 30.0 --mg_dim 8 --mg_len 2048 --mg_traj 40
            --mg_drive_scale 0.08 --mg_drive_rho 0.9 --mg_seed 2027
            --mg_driven_npz "${DRIVEN_MG_NPZ}")
        ;;
      *)
        echo "unknown COND ${COND} for ${DATASET} (tau30_ds0p08_rho0p9)" >&2
        exit 2
        ;;
    esac
    ;;
  mackey_glass)
    HIDDEN=${HIDDEN:-128}
    case "${COND}" in
      tau30) DS=(--mg_tau 30.0) ;;   # chaotic,  memory ~16 steps
      tau17) DS=(--mg_tau 17.0) ;;   # weakly chaotic
      tau12) DS=(--mg_tau 12.0) ;;   # stable limit cycle, K* effectively infinite
      *) echo "unknown COND ${COND} for ${DATASET} (tau30|tau17|tau12)"; exit 2 ;;
    esac
    DS+=(--mg_dim 8 --mg_len 2048 --mg_traj 40)
    ;;
  narma)
    HIDDEN=${HIDDEN:-128}
    case "${COND}" in
      L20) DS=(--narma_order 20) ;;  # memory ~32 steps
      L10) DS=(--narma_order 10) ;;
      L5)  DS=(--narma_order 5)  ;;  # memory ~6 steps
      *) echo "unknown COND ${COND} for ${DATASET} (L20|L10|L5)"; exit 2 ;;
    esac
    DS+=(--narma_dim 8 --narma_len 2048 --narma_traj 40 --narma_bounded 1)
    ;;
  known_snr_ar)
    HIDDEN=${HIDDEN:-128}
    SNR_AR_DIM=${SNR_AR_DIM:-8}
    SNR_AR_LEN=${SNR_AR_LEN:-1024}
    SNR_AR_TRAJ=${SNR_AR_TRAJ:-96}
    case "${COND}" in
      mixed)
        # Default: slow/fast positive and negative modes.  The signs make the
        # conditional mean oscillatory; |a| alone controls the analytic SNR.
        SNR_AR_COEFFICIENTS=${SNR_AR_COEFFICIENTS:-"0.995 0.98 0.95 0.90 -0.995 -0.98 -0.95 -0.90"}
        ;;
      slow)
        SNR_AR_COEFFICIENTS=${SNR_AR_COEFFICIENTS:-"0.995 -0.995 0.99 -0.99 0.98 -0.98 0.97 -0.97"}
        ;;
      fast)
        SNR_AR_COEFFICIENTS=${SNR_AR_COEFFICIENTS:-"0.95 -0.95 0.925 -0.925 0.90 -0.90 0.85 -0.85"}
        ;;
      *) echo "unknown COND ${COND} for ${DATASET} (mixed|slow|fast)"; exit 2 ;;
    esac
    read -r -a SNR_AR_COEFFICIENT_ARRAY <<< "${SNR_AR_COEFFICIENTS}"
    DS=(--snr_ar_dim "${SNR_AR_DIM}" --snr_ar_len "${SNR_AR_LEN}"
        --snr_ar_traj "${SNR_AR_TRAJ}" --snr_ar_seed "${SNR_AR_DATA_SEED:-0}"
        --snr_ar_coefficients "${SNR_AR_COEFFICIENT_ARRAY[@]}")
    ;;
  ieeg)
    HIDDEN=${HIDDEN:-256}
    case "${COND}" in
      theta) DS=(--ieeg_band theta) ;;   # autocorr 0.116 at 50 steps
      delta) DS=(--ieeg_band delta) ;;
      alpha) DS=(--ieeg_band alpha) ;;
      hfb)   DS=(--ieeg_band hfb)   ;;   # decorrelated within 1-2 steps
      *) echo "unknown COND ${COND} for ${DATASET} (theta|delta|alpha|hfb)"; exit 2 ;;
    esac
    DS+=(--ieeg_subject "${IEEG_SUBJECT:-P41CS}" --ieeg_task enc --ieeg_contact macro
         --ieeg_step_ms 20 --ieeg_chunk 1024)
    ;;
  gait)
    HIDDEN=${HIDDEN:-128}
    case "${COND}" in
      norm|fast|slow)          DS=(--gait_condition "${COND}") ;;  # free, autocorr >0.1 at lag 200
      metnrm|metfst|metslw)    DS=(--gait_condition "${COND}") ;;  # metronome, zero by lag 4
      *) echo "unknown COND ${COND} for ${DATASET}"; exit 2 ;;
    esac
    DS+=(--gait_chunk 512)
    ;;
  prepared_temporal_autonomous)
    HIDDEN=${HIDDEN:-128}
    PREPARED_INPUT_ROOT=${PREPARED_INPUT_ROOT:-probe_inputs/temporal_candidate_regime_v1}
    PREPARED_NPZ=${PREPARED_NPZ:-${PREPARED_INPUT_ROOT}/${COND}.npz}
    [[ -f "${PREPARED_NPZ}" ]] || {
      echo "prepared temporal archive not found: ${PREPARED_NPZ}" >&2; exit 2;
    }
    DS=(--prepared_temporal_npz "${PREPARED_NPZ}" --prepared_temporal_standardize 1)
    ;;
  prepared_temporal_driven)
    case "${COND}" in
      electricity) HIDDEN=${HIDDEN:-256} ;;
      etth1|etth2|ettm1|ettm2) HIDDEN=${HIDDEN:-128} ;;
      *) echo "unknown driven temporal COND=${COND}" >&2; exit 2 ;;
    esac
    PREPARED_INPUT_ROOT=${PREPARED_INPUT_ROOT:-probe_inputs/temporal_candidate_regime_v1}
    PREPARED_NPZ=${PREPARED_NPZ:-${PREPARED_INPUT_ROOT}/${COND}.npz}
    [[ -f "${PREPARED_NPZ}" ]] || {
      echo "prepared temporal archive not found: ${PREPARED_NPZ}" >&2; exit 2;
    }
    DS=(--prepared_temporal_npz "${PREPARED_NPZ}" --prepared_temporal_standardize 1)
    ;;
  *) echo "unknown DATASET ${DATASET}"; exit 2 ;;
esac

# ---- routing arm -------------------------------------------------------------
# Shared routing-arm definitions used by the paper experiments.
unset RESGRAD_ALPHA_PERIOD RESGRAD_ALPHA_VALUE
case "${METHOD}" in
  dense) RG=(--no-resgrad_routing --resgrad_policy all --resgrad_block_gate 1.0 --no-recurrent_grad_checkpoint); TAG="dense" ;;
  ckpt)  RG=(--no-resgrad_routing --resgrad_policy all --resgrad_block_gate 1.0 --recurrent_grad_checkpoint);    TAG="ckpt"  ;;
  tbptt[0-9]*) period="${METHOD#tbptt}"
           if (( period < 1 )); then
             echo "TBPTT period must be positive, got ${period}" >&2; exit 2
           fi
           # Keep the recurrent-state TBPTT implementation used by the
           # completed TBPTT-8 checkpoints: the same periodic hard cut is
           # applied to the autoregressive input and every Mamba state carry.
           # This makes an S sweep change only S, not the backward graph
           # implementation.
           export RESGRAD_ALPHA_PERIOD="${period}" RESGRAD_ALPHA_VALUE=0.0
           RG=(--no-resgrad_routing --resgrad_policy all --resgrad_block_gate 1.0 \
               --recurrent_grad_checkpoint --bptt_detach_period 0)
           TAG="tbptt${period}" ;;
  artbp[0-9]*) length="${METHOD#artbp}"
           if (( length <= 1 )); then
             echo "ARTBP expected segment length must exceed one, got ${length}" >&2; exit 2
           fi
           RG=(--no-resgrad_routing --resgrad_policy all --resgrad_block_gate 1.0 \
               --recurrent_grad_checkpoint \
               --artbp_expected_segment_length "${length}")
           TAG="artbp${length}" ;;
  none)  RG=(--resgrad_routing --resgrad_policy none --resgrad_block_gate 0.0 --resgrad_ratio_threshold 0.13 --no-recurrent_grad_checkpoint); TAG="none" ;;
  p[0-9]*)    frac="${METHOD#p}"
         RG=(--resgrad_routing --resgrad_policy dynamic_ratio --resgrad_block_gate 0.0 \
             --resgrad_ratio_threshold 0.13 \
             --resgrad_target_open_frac "${frac}" \
             --resgrad_calib_num_starts 4 --resgrad_calib_every_batches 1 --resgrad_calib_ema 0 \
             --no-recurrent_grad_checkpoint); TAG="p${frac}" ;;
  # snr<frac>: THE principled gate -- route internal branches by the signal-to-noise
  # ||Delta||/||sigma|| (sigma = the model's learned noise head, built by --mamba_crps),
  # calibrated to open fraction <frac>. Unlike dynamic_ratio it CUTS high-amplitude
  # steps whose amplitude is only large noise (the over-expansion driver). Needs the
  # decoupled sigma training (MSE-mean + detached-mean CRPS), which --mamba_crps triggers.
  snr[0-9]*)  frac="${METHOD#snr}"
         RG=(--resgrad_routing --resgrad_policy snr --resgrad_block_gate 0.0 \
             --resgrad_ratio_threshold 1.0 \
             --resgrad_target_open_frac "${frac}" \
             --resgrad_calib_num_starts 4 --resgrad_calib_every_batches 1 --resgrad_calib_ema 0 \
             --mamba_crps 1.0 \
             --no-recurrent_grad_checkpoint); TAG="snr${frac}" ;;
  # snrt<val>: PARAMETER-FREE SNR gate -- FIXED threshold <val> (no target open
  # fraction, no calibration). Opens a step iff ||Delta||/||sigma|| >= <val>, so the
  # open fraction g EMERGES from the data's own SNR (small g on noisy systems, larger
  # g on signal-rich ones) with the SAME universal threshold. snrt1.0 = "signal
  # exceeds noise". This is what defeats "just sweep K": the threshold is not tuned
  # per dataset. (If it opens ~nothing, the ||Delta||/||sigma|| scale is off -- report g.)
  snrt*) thr="${METHOD#snrt}"
         RG=(--resgrad_routing --resgrad_policy snr --resgrad_block_gate 0.0 \
             --resgrad_ratio_threshold "${thr}" \
             --resgrad_target_open_frac 0 \
             --mamba_crps 1.0 \
             --no-recurrent_grad_checkpoint); TAG="snrt${thr}" ;;
  # snrk: SOFT Kalman-gain SNR gate -- per step m = SNR^2/(1+SNR^2), SNR=||Delta||/||sigma||.
  # No threshold, no target fraction: the (soft, per-step) gate IS the optimal
  # Wiener/Kalman weighting, estimated from the model's own sigma head. Parameter-free
  # and not brittle (noisy steps get a small nonzero gate, not a hard cut). This is the
  # principled deployable gate; g emerges as its average.
  snrk) RG=(--resgrad_routing --resgrad_policy snrk --resgrad_block_gate 0.0 \
             --resgrad_target_open_frac 0 \
             --mamba_crps 1.0 \
             --no-recurrent_grad_checkpoint); TAG="snrk" ;;
  # coherence: THE derived optimal gate. Each pathway's backward gradient is scaled by
  # its per-batch coherence c_k = ||E_i g_i||^2 / E_i||g_i||^2 (= the Wiener gain m_k*),
  # computed in the backward pass. No sigma head, no threshold, no target fraction, no
  # CRPS -- g emerges from the gradient's cross-start alignment. Needs batch > 1.
  coh|coherence) RG=(--resgrad_routing --resgrad_policy coherence --resgrad_block_gate 0.0 \
             --no-recurrent_grad_checkpoint); TAG="coherence" ;;
  # dual/dualcoh: legacy cross-example coherence proxy.  It is retained only
  # for reproducing old runs; heterogeneous batch rows do not identify SNR.
  dual|dualcoh) RG=(--resgrad_routing --resgrad_policy dualcoh --resgrad_block_gate 0.0 \
             --no-recurrent_grad_checkpoint); TAG="dual" ;;
  # dwc<c>: EVALUATION BASELINE ONLY -- the routing operator runs with
  # alpha = m = <c> frozen at every route, no probe and no covariance estimate
  # (DUAL_WIENER_CONST gates both in DualWienerController).  This is the static
  # soft-gain control: it answers whether the gain is just "multiply the delayed
  # gradient by something below one", or whether the routewise covariance
  # estimate is doing work.  A validation-selected best-c over a few values is
  # the fair comparator.  It is NOT the method and must not be reported as one;
  # dual_wiener_gains.json records estimator="frozen_constant_gain_baseline".
  dwc[0-9]*) c="${METHOD#dwc}"
         RG=(--resgrad_routing --resgrad_policy dualwiener --resgrad_block_gate 0.0 \
             --no-recurrent_grad_checkpoint); TAG="dwc${c}"
         export DUAL_WIENER_CONST="${c}" ;;
  # dualwiener/dw: noise-identified two-route SOFT gate.  Periodic quadratic
  # probes estimate the 2x2 covariance of (identity, nonlinear) credit, while a
  # lagged diagonal residual covariance supplies observation noise.  No sigma
  # head and no target open fraction; alpha and m both emerge from the data.
  dw|dualwiener|dwpooled|dualwiener_pooled)
         noise_model="${DUAL_WIENER_NOISE_MODEL:-diagonal_gaussian}"
         case "${METHOD}" in
           dwpooled|dualwiener_pooled)
             min_probes="${DUAL_WIENER_MIN_PROBES:-8}"; TAG="dualwiener_pooled" ;;
           *) min_probes="${DUAL_WIENER_MIN_PROBES:-1}"; TAG="dualwiener" ;;
         esac
         RG=(--resgrad_routing --resgrad_policy dualwiener --resgrad_block_gate 0.0 \
             --dual_wiener_ema "${DUAL_WIENER_EMA:-0.95}" \
             --dual_wiener_residual_ema "${DUAL_WIENER_RESIDUAL_EMA:-0.99}" \
             --dual_wiener_warmup_batches "${DUAL_WIENER_WARMUP_BATCHES:-8}" \
             --dual_wiener_probe_every "${DUAL_WIENER_PROBE_EVERY:-4}" \
             --dual_wiener_min_probes "${min_probes}" \
             --dual_wiener_noise_model "${noise_model}" \
             --no-recurrent_grad_checkpoint) ;;
  # Oracle-Wiener uses the same probes, route placement, EMA and 2x2 box
  # solver as the deployable plug-in method.  The sole intervention is that R
  # is generated from the known conditional innovation covariance of the
  # synthetic process instead of centered prediction residuals.
  dworacle|dualwiener_oracle|dworaclepooled|dualwiener_oracle_pooled)
         [[ -n "${DUAL_WIENER_INNOVATION_FILE:-}" ]] || {
           echo "METHOD=${METHOD} needs DUAL_WIENER_INNOVATION_FILE=<oracle.npz>"; exit 2;
         }
         [[ -f "${DUAL_WIENER_INNOVATION_FILE}" ]] || {
           echo "oracle innovation file not found: ${DUAL_WIENER_INNOVATION_FILE}"; exit 2;
         }
         export DUAL_WIENER_INNOVATION_KEY=innovation_variance
         case "${METHOD}" in
           dworaclepooled|dualwiener_oracle_pooled)
             min_probes="${DUAL_WIENER_MIN_PROBES:-8}"; TAG="dualwiener_oracle_pooled" ;;
           *) min_probes="${DUAL_WIENER_MIN_PROBES:-1}"; TAG="dualwiener_oracle" ;;
         esac
         RG=(--resgrad_routing --resgrad_policy dualwiener --resgrad_block_gate 0.0 \
             --dual_wiener_ema "${DUAL_WIENER_EMA:-0.95}" \
             --dual_wiener_residual_ema "${DUAL_WIENER_RESIDUAL_EMA:-0.99}" \
             --dual_wiener_warmup_batches "${DUAL_WIENER_WARMUP_BATCHES:-8}" \
             --dual_wiener_probe_every "${DUAL_WIENER_PROBE_EVERY:-4}" \
             --dual_wiener_min_probes "${min_probes}" \
              --dual_wiener_noise_model diagonal_gaussian \
              --no-recurrent_grad_checkpoint) ;;
  # Strict train-only diagonal AR(1) sampler used by the known-SNR closure.
  # Keep a distinct tag from the older full-VAR/OAS autocovariance arms so a
  # completed checkpoint can never be silently reused as this estimator.
  dwdiagar|dualwiener_diag_ar)
         [[ -n "${DUAL_WIENER_INNOVATION_FILE:-}" ]] || {
           echo "METHOD=${METHOD} needs DUAL_WIENER_INNOVATION_FILE=<diagonal-ar1.npz>"; exit 2;
         }
         [[ -f "${DUAL_WIENER_INNOVATION_FILE}" ]] || {
           echo "diagonal AR(1) innovation file not found: ${DUAL_WIENER_INNOVATION_FILE}"; exit 2;
         }
         export DUAL_WIENER_INNOVATION_KEY=innovation_variance
         RG=(--resgrad_routing --resgrad_policy dualwiener --resgrad_block_gate 0.0 \
             --dual_wiener_ema "${DUAL_WIENER_EMA:-0.95}" \
             --dual_wiener_residual_ema "${DUAL_WIENER_RESIDUAL_EMA:-0.99}" \
             --dual_wiener_warmup_batches "${DUAL_WIENER_WARMUP_BATCHES:-8}" \
             --dual_wiener_probe_every "${DUAL_WIENER_PROBE_EVERY:-4}" \
             --dual_wiener_min_probes "${DUAL_WIENER_MIN_PROBES:-1}" \
             --dual_wiener_noise_model diagonal_gaussian \
             --no-recurrent_grad_checkpoint)
         TAG="dualwiener_diag_ar" ;;
  # Domain-conditional Wiener: the route solver is unchanged, while R comes
  # from an explicit dataset prior recorded in the external artifact.  This is
  # intentionally separate from the universal VAR/VARX+OAS arm below.
  dwdomain|dualwiener_domain)
         [[ -n "${DUAL_WIENER_INNOVATION_FILE:-}" ]] || {
           echo "METHOD=${METHOD} needs DUAL_WIENER_INNOVATION_FILE=<domain-estimate.npz>"; exit 2;
         }
         [[ -f "${DUAL_WIENER_INNOVATION_FILE}" ]] || {
           echo "domain innovation estimate not found: ${DUAL_WIENER_INNOVATION_FILE}"; exit 2;
         }
         # Domain artifacts may expose either a diagonal variance [K,D] or
         # whole innovation trajectories under ``innovation_templates``.
         # Respect an explicit key instead of silently forcing every domain
         # estimator through the diagonal loader (which, for example, reads
         # the iEEG [N,K,D] template bank with the wrong shape).
         export DUAL_WIENER_INNOVATION_KEY="${DUAL_WIENER_INNOVATION_KEY:-innovation_variance}"
         domain_noise_model="${DUAL_WIENER_DOMAIN_NOISE_MODEL:-}"
         if [[ -z "${domain_noise_model}" ]]; then
           if [[ "${DUAL_WIENER_INNOVATION_KEY}" == "innovation_templates" ]]; then
             domain_noise_model="lagged_residual_bootstrap"
           else
             domain_noise_model="diagonal_gaussian"
           fi
         fi
         domain_tag="${DUAL_WIENER_DOMAIN_TAG:-conditional}"
         RG=(--resgrad_routing --resgrad_policy dualwiener --resgrad_block_gate 0.0 \
             --dual_wiener_ema "${DUAL_WIENER_EMA:-0.95}" \
             --dual_wiener_residual_ema "${DUAL_WIENER_RESIDUAL_EMA:-0.99}" \
             --dual_wiener_warmup_batches "${DUAL_WIENER_WARMUP_BATCHES:-8}" \
             --dual_wiener_probe_every "${DUAL_WIENER_PROBE_EVERY:-4}" \
             --dual_wiener_min_probes "${DUAL_WIENER_MIN_PROBES:-8}" \
             --dual_wiener_noise_model "${domain_noise_model}" \
             --no-recurrent_grad_checkpoint)
         TAG="dualwiener_domain_${domain_tag}" ;;
  # Training-only autocovariance-matched linear-Gaussian innovation process.
  # The raw and OAS-shrunk files share the same sampled transitions; only Q is
  # shrunk in the latter.  Distinct tags prevent either arm from being confused
  # with the analytic oracle or the online centered-residual plug-in estimator.
  dwacm|dualwiener_acm|dwacmshrink|dualwiener_acm_shrink|dwacmpooled|dualwiener_acm_pooled|dwacmshrinkpooled|dualwiener_acm_shrink_pooled)
         [[ -n "${DUAL_WIENER_INNOVATION_FILE:-}" ]] || {
           echo "METHOD=${METHOD} needs DUAL_WIENER_INNOVATION_FILE=<estimate.npz>"; exit 2;
         }
         [[ -f "${DUAL_WIENER_INNOVATION_FILE}" ]] || {
           echo "innovation estimate not found: ${DUAL_WIENER_INNOVATION_FILE}"; exit 2;
         }
         export DUAL_WIENER_INNOVATION_KEY=innovation_variance
         case "${METHOD}" in
           dwacmpooled|dualwiener_acm_pooled|dwacmshrinkpooled|dualwiener_acm_shrink_pooled)
             min_probes="${DUAL_WIENER_MIN_PROBES:-8}" ;;
           *) min_probes="${DUAL_WIENER_MIN_PROBES:-1}" ;;
         esac
         RG=(--resgrad_routing --resgrad_policy dualwiener --resgrad_block_gate 0.0 \
             --dual_wiener_ema "${DUAL_WIENER_EMA:-0.95}" \
             --dual_wiener_residual_ema "${DUAL_WIENER_RESIDUAL_EMA:-0.99}" \
             --dual_wiener_warmup_batches "${DUAL_WIENER_WARMUP_BATCHES:-8}" \
             --dual_wiener_probe_every "${DUAL_WIENER_PROBE_EVERY:-4}" \
             --dual_wiener_min_probes "${min_probes}" \
             --dual_wiener_noise_model diagonal_gaussian \
             --no-recurrent_grad_checkpoint)
         case "${METHOD}" in
           dwacm|dualwiener_acm) TAG="dualwiener_acm" ;;
           dwacmshrink|dualwiener_acm_shrink) TAG="dualwiener_acm_shrink" ;;
           dwacmpooled|dualwiener_acm_pooled) TAG="dualwiener_acm_pooled" ;;
           *) TAG="dualwiener_acm_shrink_pooled" ;;
         esac ;;
  # Geometry-agnostic structured covariance estimator for vector-valued
  # sequences.  One lagged, centered residual field is bootstrapped jointly
  # across output coordinates and rollout horizons, so this retains covariance
  # that the coordinatewise diagonal-Gaussian estimator discards.  This is the
  # non-spatial analogue of the Fourier/spectral estimator used for 2-D fields.
  dwstructured|dualwiener_structured)
         RG=(--resgrad_routing --resgrad_policy dualwiener --resgrad_block_gate 0.0 \
             --dual_wiener_ema "${DUAL_WIENER_EMA:-0.95}" \
             --dual_wiener_residual_ema "${DUAL_WIENER_RESIDUAL_EMA:-0.99}" \
             --dual_wiener_warmup_batches "${DUAL_WIENER_WARMUP_BATCHES:-8}" \
             --dual_wiener_probe_every "${DUAL_WIENER_PROBE_EVERY:-4}" \
             --dual_wiener_min_probes "${DUAL_WIENER_MIN_PROBES:-1}" \
             --dual_wiener_noise_model lagged_residual_bootstrap \
             --no-recurrent_grad_checkpoint); TAG="dualwiener_structured" ;;
  # Global-horizon Wiener: jointly solve one coefficient per rollout loss in
  # complete parameter-gradient space.  This is the deployable counterpart of
  # the global oracle probe; it leaves every internal residual route fully open.
  ghw|globalwiener|global_horizon_wiener)
         RG=(--no-resgrad_routing --resgrad_policy all --resgrad_block_gate 1.0 \
             --global_horizon_wiener \
             --dual_wiener_ema "${GLOBAL_WIENER_EMA:-0.95}" \
             --dual_wiener_residual_ema "${GLOBAL_WIENER_RESIDUAL_EMA:-0.99}" \
             --dual_wiener_warmup_batches "${GLOBAL_WIENER_WARMUP_BATCHES:-8}" \
             --dual_wiener_probe_every "${GLOBAL_WIENER_PROBE_EVERY:-16}" \
             --dual_wiener_min_probes "${GLOBAL_WIENER_MIN_PROBES:-4}" \
             --dual_wiener_noise_model "${GLOBAL_WIENER_NOISE_MODEL:-diagonal_gaussian}" \
             --global_wiener_ridge "${GLOBAL_WIENER_RIDGE:-1e-8}" \
             --global_wiener_anchor "${GLOBAL_WIENER_ANCHOR:-0}" \
             --global_wiener_local_fidelity "${GLOBAL_WIENER_LOCAL_FIDELITY:-0}" \
             --global_wiener_solver_iters "${GLOBAL_WIENER_SOLVER_ITERS:-256}" \
             --global_wiener_sketch_dim "${GLOBAL_WIENER_SKETCH_DIM:-8192}" \
             --global_wiener_sketch_seed "${GLOBAL_WIENER_SKETCH_SEED:-1729}" \
             --global_wiener_noise_draws "${GLOBAL_WIENER_NOISE_DRAWS:-4}" \
             --global_wiener_superbatch_groups "${GLOBAL_WIENER_SUPERBATCH_GROUPS:-1}" \
             "${GLOBAL_WIENER_BATCH_FLAG[@]}" \
              --no-recurrent_grad_checkpoint)
         fidelity_tag=${GLOBAL_WIENER_LOCAL_FIDELITY:-0}
         if [[ "${fidelity_tag}" == "0" || "${fidelity_tag}" == "0.0" || "${fidelity_tag}" == "0.00" ]]; then
           TAG="global_horizon_wiener"
         else
           TAG="global_horizon_wiener_fidelity${fidelity_tag//./p}"
         fi ;;
  ghwstatic*|globalwiener_static*|global_horizon_wiener_static*)
         static_gain="${GLOBAL_WIENER_STATIC_GAIN:-${METHOD##*static}}"
         [[ "${static_gain}" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]] || {
           echo "METHOD=${METHOD} needs GLOBAL_WIENER_STATIC_GAIN in [0,1] "
                "or a suffix such as globalwiener_static0.7" >&2
           exit 2
         }
         static_tag="${static_gain//./p}"
         RG=(--no-resgrad_routing --resgrad_policy all --resgrad_block_gate 1.0 \
             --global_horizon_wiener \
             --global_wiener_static_gain "${static_gain}" \
             --global_wiener_static_mode "${GLOBAL_WIENER_STATIC_MODE:-delayed_tied}" \
             --no-global_wiener_batch_conditioned \
             --no-recurrent_grad_checkpoint)
         TAG="global_horizon_wiener_static${static_tag}" ;;
  # Same joint horizon solve, with a training-only domain innovation artifact.
  # The artifact may contain diagonal variances, a temporally coupled process,
  # or whole innovation templates; DualWiener's loader selects the appropriate
  # sampler from its metadata.
  ghwdomain|globalwiener_domain|global_horizon_wiener_domain)
         [[ -n "${DUAL_WIENER_INNOVATION_FILE:-}" ]] || {
           echo "METHOD=${METHOD} needs DUAL_WIENER_INNOVATION_FILE=<domain-estimate.npz>"; exit 2;
         }
         [[ -f "${DUAL_WIENER_INNOVATION_FILE}" ]] || {
           echo "domain innovation estimate not found: ${DUAL_WIENER_INNOVATION_FILE}"; exit 2;
         }
         export DUAL_WIENER_INNOVATION_KEY="${DUAL_WIENER_INNOVATION_KEY:-innovation_variance}"
         domain_noise_model="${GLOBAL_WIENER_NOISE_MODEL:-}"
         if [[ -z "${domain_noise_model}" ]]; then
           if [[ "${DUAL_WIENER_INNOVATION_KEY}" == "innovation_templates" ]]; then
             domain_noise_model="lagged_residual_bootstrap"
           else
             domain_noise_model="diagonal_gaussian"
           fi
         fi
         domain_tag="${GLOBAL_WIENER_DOMAIN_TAG:-conditional}"
         RG=(--no-resgrad_routing --resgrad_policy all --resgrad_block_gate 1.0 \
             --global_horizon_wiener \
             --dual_wiener_ema "${GLOBAL_WIENER_EMA:-0.95}" \
             --dual_wiener_residual_ema "${GLOBAL_WIENER_RESIDUAL_EMA:-0.99}" \
             --dual_wiener_warmup_batches "${GLOBAL_WIENER_WARMUP_BATCHES:-8}" \
             --dual_wiener_probe_every "${GLOBAL_WIENER_PROBE_EVERY:-16}" \
             --dual_wiener_min_probes "${GLOBAL_WIENER_MIN_PROBES:-4}" \
             --dual_wiener_noise_model "${domain_noise_model}" \
             --global_wiener_ridge "${GLOBAL_WIENER_RIDGE:-1e-8}" \
             --global_wiener_anchor "${GLOBAL_WIENER_ANCHOR:-0}" \
             --global_wiener_local_fidelity "${GLOBAL_WIENER_LOCAL_FIDELITY:-0}" \
             --global_wiener_solver_iters "${GLOBAL_WIENER_SOLVER_ITERS:-256}" \
             --global_wiener_sketch_dim "${GLOBAL_WIENER_SKETCH_DIM:-8192}" \
             --global_wiener_sketch_seed "${GLOBAL_WIENER_SKETCH_SEED:-1729}" \
             --global_wiener_noise_draws "${GLOBAL_WIENER_NOISE_DRAWS:-4}" \
             --global_wiener_superbatch_groups "${GLOBAL_WIENER_SUPERBATCH_GROUPS:-1}" \
             "${GLOBAL_WIENER_BATCH_FLAG[@]}" \
             --no-recurrent_grad_checkpoint)
         fidelity_tag=${GLOBAL_WIENER_LOCAL_FIDELITY:-0}
         if [[ "${fidelity_tag}" == "0" || "${fidelity_tag}" == "0.0" || "${fidelity_tag}" == "0.00" ]]; then
           TAG="global_horizon_wiener_domain_${domain_tag}"
         else
           TAG="global_horizon_wiener_domain_${domain_tag}_fidelity${fidelity_tag//./p}"
         fi ;;
  # Hybrid backward operator: first truncate temporal state/carry paths every
  # S rollout steps (the same semantics as METHOD=tbpttS), then jointly weight
  # the resulting per-horizon parameter gradients with Global-Horizon Wiener.
  # The controller's total and innovation probes traverse this already-cut
  # graph, so it solves for sum_k w_k * g_tilde_k^(S), not for full-BPTT g_k.
  ghwdomain_tbptt[0-9]*|globalwiener_domain_tbptt[0-9]*|global_horizon_wiener_domain_tbptt[0-9]*)
         period="${METHOD##*tbptt}"
         if [[ ! "${period}" =~ ^[0-9]+$ ]] || (( period < 1 )); then
           echo "Global-Wiener+TBPTT period must be positive, got ${period}" >&2
           exit 2
         fi
         [[ -n "${DUAL_WIENER_INNOVATION_FILE:-}" ]] || {
           echo "METHOD=${METHOD} needs DUAL_WIENER_INNOVATION_FILE=<domain-estimate.npz>"; exit 2;
         }
         [[ -f "${DUAL_WIENER_INNOVATION_FILE}" ]] || {
           echo "domain innovation estimate not found: ${DUAL_WIENER_INNOVATION_FILE}"; exit 2;
         }
         export DUAL_WIENER_INNOVATION_KEY="${DUAL_WIENER_INNOVATION_KEY:-innovation_variance}"
         export RESGRAD_ALPHA_PERIOD="${period}" RESGRAD_ALPHA_VALUE=0.0
         domain_tag="${GLOBAL_WIENER_DOMAIN_TAG:-conditional}"
         RG=(--no-resgrad_routing --resgrad_policy all --resgrad_block_gate 1.0 \
             --global_horizon_wiener \
             --dual_wiener_ema "${GLOBAL_WIENER_EMA:-0.95}" \
             --dual_wiener_residual_ema "${GLOBAL_WIENER_RESIDUAL_EMA:-0.99}" \
             --dual_wiener_warmup_batches "${GLOBAL_WIENER_WARMUP_BATCHES:-8}" \
             --dual_wiener_probe_every "${GLOBAL_WIENER_PROBE_EVERY:-16}" \
             --dual_wiener_min_probes "${GLOBAL_WIENER_MIN_PROBES:-4}" \
             --dual_wiener_noise_model "${GLOBAL_WIENER_NOISE_MODEL:-diagonal_gaussian}" \
             --global_wiener_ridge "${GLOBAL_WIENER_RIDGE:-1e-8}" \
             --global_wiener_anchor "${GLOBAL_WIENER_ANCHOR:-0}" \
             --global_wiener_local_fidelity "${GLOBAL_WIENER_LOCAL_FIDELITY:-0}" \
             --global_wiener_solver_iters "${GLOBAL_WIENER_SOLVER_ITERS:-256}" \
             --global_wiener_sketch_dim "${GLOBAL_WIENER_SKETCH_DIM:-8192}" \
             --global_wiener_sketch_seed "${GLOBAL_WIENER_SKETCH_SEED:-1729}" \
             --global_wiener_noise_draws "${GLOBAL_WIENER_NOISE_DRAWS:-4}" \
             --global_wiener_superbatch_groups "${GLOBAL_WIENER_SUPERBATCH_GROUPS:-1}" \
             "${GLOBAL_WIENER_BATCH_FLAG[@]}" \
             --no-recurrent_grad_checkpoint)
         TAG="global_horizon_wiener_domain_${domain_tag}_tbptt${period}" ;;
  # oracle: THE CEILING. The mask is not learned -- it comes from the model-free
  # per-(start,horizon) predictability map e_k(t),
  # keeping the most-predictable fraction of starts WITHIN each horizon k. This
  # answers "is there any gate at K=64 that reaches what a per-dataset K-sweep
  # finds?", which no learned statistic can answer. Needs:
  #   RESGRAD_ORACLE=<ek_map.npz> RESGRAD_ORACLE_FRAC=<p>
  oracle) RG=(--resgrad_routing --resgrad_policy oracle --resgrad_block_gate 0.0 \
             --no-recurrent_grad_checkpoint); TAG="oracle${RESGRAD_ORACLE_FRAC:-0.5}"
         [[ -n "${RESGRAD_ORACLE:-}" ]] || { echo "METHOD=oracle needs RESGRAD_ORACLE=<ek_map.npz>"; exit 2; }
         export RESGRAD_ORACLE RESGRAD_ORACLE_FRAC ;;
  # Placement ablation at a matched open budget: keep the SAME fraction of steps
  # open but concentrate it at the window's end (tail) vs. spread it evenly
  # (periodic). tail<n>: open the last n of K steps. periodic<n>: open every n-th
  # step. Same budget, different position -> tests "which pathways", not "how many".
  tail*) keep="${METHOD#tail}"
         RG=(--resgrad_routing --resgrad_policy tail --resgrad_block_gate 0.0 \
             --resgrad_ratio_threshold 0.13 --resgrad_keep_tail "${keep}" \
             --no-recurrent_grad_checkpoint); TAG="tail${keep}" ;;
  periodic*) every="${METHOD#periodic}"
         RG=(--resgrad_routing --resgrad_policy periodic --resgrad_block_gate 0.0 \
             --resgrad_ratio_threshold 0.13 --resgrad_keep_every "${every}" \
             --no-recurrent_grad_checkpoint); TAG="periodic${every}" ;;
  # ---- TEMPORAL-RESIDUAL (outer) variants -----------------------------------
  # Same keep/cut budget as tail/head/periodic above, but the gate is applied to
  # the OUTER per-step residual delta (pred=x_t+delta) with every INTERNAL block
  # left fully open (--resgrad_outer). This is the theory's per-step Jacobian
  # I+m*J_F realized one-to-one; the internal-vs-outer pair isolates whether the
  # routing OBJECT (muddy internal branches) is what costs accuracy at large K.
  # op<frac>: THE temporal-residual method -- outer Delta gated by the dynamic
  # ||Delta||/||x_t|| ratio, calibrated to open fraction <frac> (the mechanism's
  # ratio criterion on the clean per-step object). This is what we compare to K16.
  op[0-9]*)   frac="${METHOD#op}"
         RG=(--resgrad_routing --resgrad_outer --resgrad_policy dynamic_ratio --resgrad_block_gate 0.0 \
             --resgrad_ratio_threshold 0.13 \
             --resgrad_target_open_frac "${frac}" \
             --resgrad_calib_num_starts 4 --resgrad_calib_every_batches 1 --resgrad_calib_ema 0 \
             --no-recurrent_grad_checkpoint); TAG="op${frac}" ;;
  otail*) keep="${METHOD#otail}"
         RG=(--resgrad_routing --resgrad_outer --resgrad_policy tail --resgrad_block_gate 0.0 \
             --resgrad_keep_tail "${keep}" --no-recurrent_grad_checkpoint); TAG="otail${keep}" ;;
  ohead*) keep="${METHOD#ohead}"
         RG=(--resgrad_routing --resgrad_outer --resgrad_policy head --resgrad_block_gate 0.0 \
             --resgrad_keep_tail "${keep}" --no-recurrent_grad_checkpoint); TAG="ohead${keep}" ;;
  operiodic*) every="${METHOD#operiodic}"
         RG=(--resgrad_routing --resgrad_outer --resgrad_policy periodic --resgrad_block_gate 0.0 \
             --resgrad_keep_every "${every}" --no-recurrent_grad_checkpoint); TAG="operiodic${every}" ;;
  *) echo "unknown METHOD ${METHOD}"; exit 2 ;;
esac

# Cross-backbone controls use the same explicit residual implementation in the
# open and DW arms.  This avoids comparing PyTorch's opaque TransformerEncoder
# against the hookable layer solely because one arm needs route access.
if [[ "${FORCE_HOOKABLE_RESIDUALS:-0}" == "1" && "${METHOD}" == "ckpt" ]]; then
  RG=(--resgrad_routing --resgrad_policy all --resgrad_block_gate 1.0 --no-recurrent_grad_checkpoint)
fi

out="${SAVE_BASE}/${DATASET}/${COND}_K${K}/${TAG}/seed${SEED}"
if [[ "${SKIP_EXISTING}" == "1" && -f "${out}/eval_results.json" ]]; then
  echo "[SKIP] ${out} (eval_results.json exists)"; exit 0
fi
if [[ "${SKIP_EXISTING}" == "1" && "${RUN_MODE}" == "train" && -f "${out}/.train_complete" && -s "${out}/best.pth" ]]; then
  echo "[SKIP] ${out} (.train_complete exists)"; exit 0
fi
mkdir -p "${out}"
# Prevent two queue drivers from training/resuming the same run concurrently.
# The lock file may remain on disk, but the advisory lock is released when the
# launcher and all of its child processes exit.
if command -v flock >/dev/null 2>&1; then
  exec {MEM_RUN_LOCK_FD}>"${out}/.run.lock"
  if ! flock -n "${MEM_RUN_LOCK_FD}"; then
    echo "[SKIP] ${out} (another process holds the run lock)"
    exit 0
  fi
fi
export CUDA_VISIBLE_DEVICES=${GPUS}
IFS=',' read -r -a MEM_GPU_IDS <<< "${GPUS}"
MEM_WORLD_SIZE=${#MEM_GPU_IDS[@]}
export MASTER_PORT=${MASTER_PORT:-$((45000 + K + SEED * 17 + FIRST_GPU * 7 + RANDOM % 300))}
# Unique per run: include SAVE_BASE basename and K so two runs that share
# (DATASET,COND,METHOD,SEED) but differ in save tree or window do not tee into
# the same file. Otherwise e.g. memtest and memtest_sched interleave one log.
LOGTAG="$(basename "${SAVE_BASE}")_${DATASET}_${COND}_K${K}_${TAG}_s${SEED}"
echo "=== ${DATASET}/${COND} K=${K} ${TAG} seed${SEED} on GPUs ${GPUS} -> ${out} === $(date)"
echo "[config] DATASET=${DATASET} COND=${COND} HIDDEN=${HIDDEN} K=${K} METHOD=${METHOD} SEED=${SEED} LR=${LR} optimizer=${AR_OPTIMIZER} momentum=${AR_SGD_MOMENTUM} weight_decay=${WEIGHT_DECAY} grad_clip=${GRAD_CLIP} scheduler=${AR_SCHEDULER} EPOCHS=${EPOCHS} ES=${ES} world=${MEM_WORLD_SIZE} local_batch=${BATCH} accum=${GRAD_ACCUM} EXTRA_ARGS='${EXTRA_ARGS:-}'"
echo "[config] RECURRENT_EVAL_HORIZON_BATCH=${RECURRENT_EVAL_HORIZON_BATCH} AR_SHARED_ROLLOUT_START=${AR_SHARED_ROLLOUT_START:-0}"
echo "[config] dataset args: ${DS[*]}"

MODEL_NAME=${MODEL_NAME:-official_mamba_state}
SIMPLE_DEPTH=${SIMPLE_DEPTH:-4}
SIMPLE_NHEAD=${SIMPLE_NHEAD:-8}
if [[ "${MODEL_NAME}" == "official_mamba_state" ]]; then
  AR_TRAIN_STARTS=1
  ROLLOUT_GRAPH_ARGS=(--no-bptt_loss)
else
  # Standard autoregressive backbones use the generic exact-BPTT rollout.
  # The loss, horizon, starts, split, and optimizer geometry remain matched to
  # the Mamba experiment; only the backbone changes.
  AR_TRAIN_STARTS=${MAMBA_TRAIN_STARTS}
  ROLLOUT_GRAPH_ARGS=(
    --bptt_loss --bptt_eval --bptt_horizon "${K}" --bptt_lambda 1.0
    --bptt_loss_type "${MAMBA_LOSS_TYPE}" --no-bptt_grad_checkpoint
  )
fi

RUNTIME_ARGS=()
if [[ "${FAST_TRAIN_RUNTIME:-0}" == "1" ]]; then
  # Performance-only path: objectives, gradients, checkpoints, validation, and
  # test evaluation are unchanged.  It removes training-time scalar syncs and
  # the explicit per-epoch CUDA memory synchronize.
  RUNTIME_ARGS+=(--fast_train_logging --no-log_gpu_memory)
fi

python src/main.py \
  --mode "${RUN_MODE}" --dataset "${DATASET}" --data_path "${DATA_DIR}" --seed "${SEED}" \
  --num_workers "${NUM_WORKERS}" \
  --resume "${RESUME}" \
  --model_name "${MODEL_NAME}" \
  --simple_hidden_dim "${HIDDEN}" --simple_depth "${SIMPLE_DEPTH}" --simple_nhead "${SIMPLE_NHEAD}" --simple_dropout 0.0 --simple_residual \
  --mamba_d_state 16 --mamba_d_conv 4 --mamba_expand 2 \
  --mamba_bptt_horizon "${K}" --mamba_burnin 32 --mamba_train_starts_per_sequence "${MAMBA_TRAIN_STARTS}" \
  --mamba_loss_type "${MAMBA_LOSS_TYPE}" --mamba_loss_decay 1.0 \
  --window_size 16 --local_batch_size "${BATCH}" --grad_accum_steps "${GRAD_ACCUM}" \
  --base_lr "${LR}" --weight_decay "${WEIGHT_DECAY}" --grad_clip "${GRAD_CLIP}" \
  --num_epochs "${EPOCHS}" --early_stop_patience "${ES}" --eval_every 1 \
  --ar_optimizer "${AR_OPTIMIZER}" --ar_sgd_momentum "${AR_SGD_MOMENTUM}" \
  "${SGD_NESTEROV_FLAG[@]}" \
  --ar_scheduler "${AR_SCHEDULER}" --ar_min_lr "${AR_MIN_LR}" \
  --ar_loss mse --ar_one_step_lambda 1.0 --ar_train_starts_per_sequence "${AR_TRAIN_STARTS}" \
  --ar_train_random_starts "${AR_SHARED_START_FLAG[@]}" --ar_train_stride 1 --ar_eval_stride 4 \
  --test_horizons 1 2 4 8 16 32 64 96 128 \
  "${DS[@]}" \
  "${RG[@]}" \
  "${RUNTIME_ARGS[@]}" \
  --no-bridge_control_loss --bridge_control_lambda 0 \
  --koopman_long_loss none "${ROLLOUT_GRAPH_ARGS[@]}" --no-comp_graph_loss --no-frontier_graph_loss \
  --save_root "${out}" ${EXTRA_ARGS:-} 2>&1 | tee -a "mem_${LOGTAG}.log"
train_rc=${PIPESTATUS[0]}
if (( train_rc != 0 )); then
  echo "[failed] ${DATASET}/${COND} ${TAG} seed${SEED}; python rc=${train_rc}" >&2
  exit "${train_rc}"
fi
if [[ "${RUN_MODE}" == "train" ]]; then
  if [[ ! -s "${out}/best.pth" ]]; then
    echo "[failed] train mode exited without ${out}/best.pth" >&2
    exit 4
  fi
  touch "${out}/.train_complete"
fi
echo "=== done ${DATASET}/${COND} ${TAG} seed${SEED} === $(date)"
