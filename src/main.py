import argparse
from html import parser
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from internal_dw.datasets import build_dataloaders, list_datasets
from internal_dw.datasets.registry import dataset_has_external_input, dataset_task_type, dataset_evaluator_name
from internal_dw.evaluation.eval import evaluate_loss, evaluate_horizon_sweep, save_eval_json
from internal_dw.models import build_model, list_models
from internal_dw.training.trainer import configure_trainable, train_model
from internal_dw.utils import build_exp_dir, cleanup_ddp, load_checkpoint, seed_everything, setup_ddp, unwrap_model


def maybe_wrap(model, rank, world_size):
    if world_size > 1:
        return DDP(
            model,
            device_ids=[rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
            # Reuse parameter-gradient storage as DDP bucket views.  This only
            # changes gradient-buffer allocation/copies; reduction semantics are
            # identical to the ordinary DDP path.
            gradient_as_bucket_view=True,
        )
    return model


def _ddp_stage_sync(label, rank, world_size):
    """Surface asynchronous CUDA faults at the stage that launched them.

    Enabled only for short diagnostics through DDP_STAGE_DEBUG=1, so normal
    training does not pay for extra global device synchronizations.
    """
    if world_size <= 1 or os.environ.get("DDP_STAGE_DEBUG", "0") != "1":
        return
    torch.cuda.synchronize(rank)
    print(f"[ddp-stage] rank={rank} {label}: CUDA sync ok", flush=True)






@torch.no_grad()
def maybe_fit_field_normalizer(model, train_loader, args, rank=0):
    """Fit per-channel Gaussian normalizer for 2D AR field models.

    U-Net keeps train-set normalization statistics as model buffers, so
    checkpoints preserve the exact preprocessing used at train time.
    """
    raw = unwrap_model(model)
    if not getattr(raw, "is_standard_autoregressive", False):
        return
    if not hasattr(raw, "set_state_normalizer"):
        return
    model_name = str(getattr(args, "model_name", ""))
    normalize = bool(getattr(args, "unet_normalize", True))
    if not normalize:
        return
    if getattr(raw, "normalizer_fitted", False):
        return

    device = torch.device(f"cuda:{rank}")
    sum_c = None
    sumsq_c = None
    count = torch.zeros((), device=device, dtype=torch.float64)
    for batch in train_loader:
        state, _, _, _ = __import__('internal_dw.data_utils.state_ops', fromlist=['unpack_batch']).unpack_batch(batch)
        # state: [B,T,C,H,W]. Accumulate on GPU so NCCL all_reduce works.
        state = state.to(device, non_blocking=True).float()
        dims = (0, 1, 3, 4)
        cur_sum = state.sum(dim=dims).double()
        cur_sumsq = state.square().sum(dim=dims).double()
        cur_count = state.shape[0] * state.shape[1] * state.shape[3] * state.shape[4]
        sum_c = cur_sum if sum_c is None else sum_c + cur_sum
        sumsq_c = cur_sumsq if sumsq_c is None else sumsq_c + cur_sumsq
        count += float(cur_count)

    if sum_c is None or float(count.detach().cpu()) <= 0:
        return

    # In DDP, train_loader is sharded. Aggregate statistics across ranks so the
    # The normalizer is fitted on the full training split, not a rank-local shard.
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(sum_c, op=dist.ReduceOp.SUM)
        dist.all_reduce(sumsq_c, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

    mean = sum_c / count.clamp_min(1.0)
    var = (sumsq_c / count.clamp_min(1.0) - mean.square()).clamp_min(1e-12)
    std = torch.sqrt(var)
    raw.set_state_normalizer(mean.float(), std.float())
    if rank == 0:
        print(f"[FieldAR] fitted per-channel normalizer: mean={mean.detach().cpu().tolist()} std={std.detach().cpu().tolist()}")


def worker(rank, args, world_size):
    setup_ddp(rank, world_size)
    seed_everything(args.seed + rank)
    exp_dir = build_exp_dir(args.save_root)

    train_loader, val_loader, test_loader = build_dataloaders(args, rank=rank, world_size=world_size)
    if args.dataset == "the_well":
        sample0 = train_loader.dataset[0]
        state0 = sample0["state"] if isinstance(sample0, dict) else sample0[0]
        if int(getattr(args, "field_channels", 0)) <= 0:
            args.field_channels = int(state0.shape[1])
        if state0.dim() >= 4:
            args.field_height = int(state0.shape[2])
            args.field_width = int(state0.shape[3])
            # Vector-style recurrent models flatten the field internally.
            # Use roi_dim as the per-frame flattened state dimension for all
            # Mamba models that internally flatten The Well fields.
            if str(getattr(args, "model_name", "")) == "official_mamba_state":
                # These Mamba-style vector models flatten each The Well field
                # frame before the token projection.  Therefore roi_dim must
                # be the per-frame flattened field dimension C*H*W, not the
                # HCP default parcel count (400).
                args.roi_dim = int(state0[0].numel())
        if rank == 0:
            print(
                f"[Data] inferred field_shape=({args.field_channels},"
                f"{getattr(args, 'field_height', 0)},{getattr(args, 'field_width', 0)}), "
                f"roi_dim={args.roi_dim} from first training sample"
            )
    args.dataset_has_external_input = dataset_has_external_input(args.dataset)
    args.dataset_task_type = dataset_task_type(args.dataset)
    args.dataset_evaluator_name = dataset_evaluator_name(args.dataset)
    if rank == 0:
        print("[Dataset] "
              f"dataset={args.dataset}; task_type={args.dataset_task_type}; "
              f"evaluator={args.dataset_evaluator_name}; has_external_input={args.dataset_has_external_input}")
    model = build_model(args, rank=rank)
    _ddp_stage_sync("after_build_model", rank, world_size)

    if args.model_ckpt_path:
        load_checkpoint(model, args.model_ckpt_path, map_location=f"cuda:{rank}", strict=not args.non_strict_ckpt)
        if rank == 0:
            print(f"Loaded checkpoint: {args.model_ckpt_path}")
    else:
        maybe_fit_field_normalizer(model, train_loader, args, rank=rank)

    configure_trainable(model, args, rank=rank)
    _ddp_stage_sync("after_configure_trainable", rank, world_size)
    _ddp_stage_sync("before_DDP_wrap", rank, world_size)
    model = maybe_wrap(model, rank, world_size)
    _ddp_stage_sync("after_DDP_wrap", rank, world_size)

    do_train = args.mode in ["train", "train_and_test"]
    trained_this_run = False
    if do_train and int(args.num_epochs) > 0:
        train_model(model, train_loader, val_loader, args, rank=rank, exp_dir=exp_dir)
        trained_this_run = True

    if args.mode in ["test", "train_and_test"]:
        if (not args.model_ckpt_path) and trained_this_run:
            best = os.path.join(exp_dir, "best.pth")
            if os.path.exists(best):
                load_checkpoint(model, best, map_location=f"cuda:{rank}", strict=True)
                if rank == 0:
                    print(f"Loaded best checkpoint: {best}")
        # Evaluation should be done on the full validation/test splits.
        # In DDP, val_loader/test_loader are sharded, so evaluating on rank 0's
        # loader would report metrics on only one shard.  Build unsharded loaders
        # on rank 0 and let other ranks wait at barriers.
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        if rank == 0:
            if world_size > 1:
                _, eval_val_loader, eval_test_loader = build_dataloaders(args, rank=0, world_size=1)
            else:
                eval_val_loader, eval_test_loader = val_loader, test_loader

            val_logs = evaluate_loss(model, eval_val_loader, args, rank=rank, prefix="val")
            test_logs = evaluate_horizon_sweep(model, eval_test_loader, args, rank=rank, prefix="test")
            logs = {**val_logs, **test_logs}
            print(logs)
            save_eval_json(os.path.join(exp_dir, "eval_results.json"), logs)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    cleanup_ddp()


def build_parser():
    p = argparse.ArgumentParser("Internal-DW long-horizon prediction")
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--dataset", type=str, default="hcp_movie", choices=list_datasets())
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--movie", type=int, default=1)
    p.add_argument("--save_root", type=str, default="experiments/internal_dw")
    p.add_argument(
        "--model_name",
        type=str,
        default="official_mamba_state",
        choices=list_models(),
    )
    p.add_argument("--mode", type=str, default="train_and_test", choices=["train", "test", "train_and_test"])
    p.add_argument("--model_ckpt_path", type=str, default="")
    # Resume an interrupted run: "auto" picks up <save_root>/last.pth, or give an explicit
    # path; empty disables. Restores model, optimizer, scheduler, epoch counter, the running
    # best val loss and the early-stop counter, so training continues as if uninterrupted.
    # Checkpoints written before this flag existed lack the last two; they are recovered by
    # replaying train_logs.jsonl. Raise --num_epochs to extend a run past its old budget.
    p.add_argument("--resume", type=str, default="", help='"auto" | path to a .pth | "" (off)')
    # Temporal-untying experiment (official_mamba_state only). 1 = fully-shared
    # rollout (default). G>1 gives G independent block-stack copies; rollout step k
    # (of K) uses group k*G//K, so G=K fully unties. Tests whether routing's
    # near-losslessness comes from parameter reuse across rollout steps.
    p.add_argument("--untie_groups", type=int, default=1)
    p.add_argument("--non_strict_ckpt", action="store_true", default=False)

    # data
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--num_workers", type=int, default=int(os.environ.get("NUM_WORKERS", "4")))
    p.add_argument("--stim_dim", type=int, default=1664)
    p.add_argument("--roi_dim", type=int, default=400)
    p.add_argument("--visual_only", action="store_true", default=False)

    # Mackey-Glass (dataset "mackey_glass"): delay ODE whose delay tau IS the memory
    # length in AR steps (tau/mg_dt), so K* is set rather than estimated. It also
    # spans the regimes: at beta=0.2, gamma=0.1, n=10 it is a stable limit cycle for
    # tau < ~16.8, weakly chaotic near tau=17, and clearly chaotic by tau=30.
    p.add_argument("--mg_dim", type=int, default=8, help="independent series stacked as channels")
    p.add_argument("--mg_tau", type=float, default=17.0, help="delay; = memory length in AR steps when mg_dt=1")
    p.add_argument("--mg_dt", type=float, default=1.0, help="autoregressive step (model time step)")
    p.add_argument("--mg_solver_dt", type=float, default=0.1, help="internal RK4 step")
    p.add_argument("--mg_len", type=int, default=2048, help="samples kept per trajectory")
    p.add_argument("--mg_traj", type=int, default=40, help="number of independent trajectories")
    p.add_argument("--mg_beta", type=float, default=0.2)
    p.add_argument("--mg_gamma", type=float, default=0.1)
    p.add_argument("--mg_n", type=float, default=10.0)
    p.add_argument("--mg_transient", type=int, default=1000, help="AR steps discarded before recording")
    p.add_argument("--mg_seed", type=int, default=0, help="seed for trajectory generation (not the train seed)")
    p.add_argument("--mg_drive_scale", type=float, default=0.08,
                   help="coefficient of the observed forcing in dataset=mackey_glass_driven")
    p.add_argument("--mg_drive_rho", type=float, default=0.9,
                   help="AR(1) correlation of the observed forcing in dataset=mackey_glass_driven")
    p.add_argument(
        "--mg_driven_npz",
        type=str,
        default="",
        help=(
            "optional pre-split Driven-MG archive produced by "
            "scripts/data/generate_driven_mackey_glass.py; using it fixes the data "
            "and split across training seeds"
        ),
    )

    # NARMA-L (dataset "narma"): the driven counterpart. The order L is the exact lag
    # support of BOTH the state term and the input term, so it is the controlled
    # analogue of a system with a fast state channel and a slow input channel.
    # narma_bounded wraps the update in tanh (standard in the reservoir literature)
    # so L can be swept past 10 without the classical recursion diverging.
    p.add_argument("--narma_dim", type=int, default=8, help="independent series stacked as channels")
    p.add_argument("--narma_order", type=int, default=10, help="order L = memory length, both channels")
    p.add_argument("--narma_len", type=int, default=2048, help="samples kept per trajectory")
    p.add_argument("--narma_traj", type=int, default=40, help="number of independent trajectories")
    p.add_argument("--narma_transient", type=int, default=200, help="steps discarded before recording")
    p.add_argument("--narma_u_scale", type=float, default=0.5, help="drive u ~ Uniform[0, u_scale]")
    p.add_argument("--narma_bounded", type=int, default=1, help="1 = tanh-bounded update (needed for L>10)")
    p.add_argument("--narma_seed", type=int, default=0, help="seed for trajectory generation (not the train seed)")
    # Coefficient on the input product term (1.5 in the textbook NARMA equation).
    # This is the only knob that changes how much of the target is supplied by the
    # exogenous input rather than the state's own history, while leaving the lag
    # structure -- and therefore the memory length -- untouched. Sweeping it is the
    # controlled test of whether drive strength governs the routing effect.
    p.add_argument("--narma_drive", type=float, default=1.5,
                   help="coefficient on the u(t-L+1)u(t) drive term (textbook value 1.5)")

    # Identifiable oracle-Wiener testbed. Each channel follows
    # x[t+1,j]=a[j]x[t,j]+eps[t+1,j], Var(eps_j)=1-a[j]^2, so the process is
    # exactly stationary with Var(x_j)=1. At horizon k the conditional signal
    # and innovation variances are a[j]^(2k) and 1-a[j]^(2k), respectively.
    p.add_argument("--snr_ar_dim", type=int, default=8,
                   help="state dimension of dataset=known_snr_ar")
    p.add_argument("--snr_ar_len", type=int, default=1024,
                   help="samples per known-SNR AR trajectory")
    p.add_argument("--snr_ar_traj", type=int, default=96,
                   help="number of independent known-SNR AR trajectories")
    p.add_argument("--snr_ar_seed", type=int, default=0,
                   help="known-SNR AR generation seed (separate from train seed)")
    p.add_argument(
        "--snr_ar_coefficients",
        type=float,
        nargs="*",
        default=None,
        help=(
            "one coefficient (repeated) or snr_ar_dim diagonal AR coefficients; "
            "all must satisfy |a|<1. The default mixes slow/fast positive and "
            "negative modes so the conditional mean contains oscillatory modes."
        ),
    )

    # iEEG band envelopes (dataset "ieeg"). Electrode counts differ between subjects,
    # so one model per subject; the recording is split along TIME into contiguous
    # train/val/test blocks with a gap. At ieeg_step_ms=20 the theta envelope has
    # one-step autocorrelation 0.993 and remains around 0.14 at 50 steps.
    p.add_argument("--ieeg_root", type=str,
                   default="data/ieeg/preprocessed_length_matched",
                   help="directory of flat {subj}_{run}_{task}_{contact}_{band}.fif files")
    p.add_argument("--ieeg_subject", type=str, default="P41CS")
    p.add_argument("--ieeg_task", type=str, default="enc", help="enc | recog")
    p.add_argument("--ieeg_contact", type=str, default="macro", help="macro | micro")
    p.add_argument("--ieeg_band", type=str, default="theta",
                   help="delta|theta|alpha|beta|gamma|hfb|hfb_ext|broadband_full|...")
    p.add_argument("--ieeg_step_ms", type=float, default=20.0, help="AR step in milliseconds")
    p.add_argument("--ieeg_chunk", type=int, default=1024, help="steps per training sequence")
    p.add_argument("--ieeg_max_channels", type=int, default=0, help="0 = keep all channels")
    p.add_argument("--ieeg_split_gap", type=int, default=256,
                   help="steps dropped between splits so no window straddles a boundary")

    # Checkpoint-free temporal archives created by
    # scripts/data/prepare_temporal_candidate_screen_data.py.  Autonomous and driven
    # archives use separate dataset names so the model-side external-input
    # contract remains static and auditable.
    p.add_argument("--prepared_temporal_npz", type=str, default="")
    p.add_argument("--prepared_temporal_standardize", type=int, default=1)

    # WeatherBench-2 memory-mapped field trajectories.
    p.add_argument("--wb2_seg_len", type=int, default=64,
                   help="WeatherBench-2 memory-map trajectory length in 6-hour steps.")
    p.add_argument("--wb2_train_stride", type=int, default=64,
                   help="Start stride between WeatherBench-2 training trajectories.")
    p.add_argument("--wb2_eval_stride", type=int, default=64,
                   help="Start stride between WeatherBench-2 validation/test trajectories.")

    p.add_argument("--window_size", type=int, default=4)
    p.add_argument("--test_horizons", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])

    # The Well / local HDF5 field datasets
    p.add_argument("--thewell_dataset_name", type=str, default="gray_scott_reaction_diffusion",
                   help="The Well subset (for example gray_scott_reaction_diffusion, turbulent_radiative_layer_2D, rayleigh_benard, shear_flow, or viscoelastic_instability).")
    p.add_argument("--thewell_sequence_length", type=int, default=0,
                   help="Number of time steps per sample for The Well. 0 means use the full trajectory in each indexed sample.")
    p.add_argument("--thewell_sequence_stride", type=int, default=1,
                   help="Temporal start stride when slicing The Well trajectories into sequence_length chunks.")
    p.add_argument("--thewell_time_subsample", type=int, default=1,
                   help="Temporal subsampling inside each returned The Well sequence. 10 returns every 10th raw frame.")
    p.add_argument("--thewell_spatial_subsample", type=int, default=1,
                   help="Deterministic stride along both spatial axes for The Well pilot runs. 1 preserves native resolution.")
    p.add_argument("--thewell_field_groups", type=str, default="t0_fields,t1_fields",
                   help="Comma-separated HDF5 field groups to read for The Well, usually t0_fields,t1_fields.")
    p.add_argument("--thewell_max_trajectories_per_file", type=int, default=0,
                   help="Optional debug cap. 0 uses all trajectories per HDF5 file.")

    # 2D field model options
    p.add_argument("--field_channels", type=int, default=0,
                   help="Number of channels in 2D field states. For dataset=the_well this can be inferred from the first training sample.")
    p.add_argument("--field_height", type=int, default=0,
                   help="Spatial height for flattened recurrent field models; inferred for dataset=the_well.")
    p.add_argument("--field_width", type=int, default=0,
                   help="Spatial width for flattened recurrent field models; inferred for dataset=the_well.")

    # U-Net autoregressive baseline options
    p.add_argument("--unet_base_channels", type=int, default=64, help="Base channel width for unet_field.")
    p.add_argument("--unet_depth", type=int, default=4, help="Number of U-Net resolution levels for unet_field.")
    p.add_argument("--unet_channel_mult", type=int, default=2, help="Channel multiplier between U-Net levels.")
    p.add_argument("--unet_groups", type=int, default=8, help="GroupNorm group count for unet_field.")
    p.add_argument("--unet_use_grid", action=argparse.BooleanOptionalAction, default=True, help="Concatenate normalized coordinate channels to U-Net input.")
    p.add_argument("--unet_normalize", action=argparse.BooleanOptionalAction, default=True, help="Use per-channel train-set Gaussian normalization inside the U-Net model.")

    # Recurrent Mamba-style predictive-state baseline.  This is trained with
    # GT burn-in + K-step closed-loop chunked BPTT.  It keeps the recurrent
    # state value across a burn-in window but truncates the gradient graph to
    # the rollout chunk.
    p.add_argument("--simple_hidden_dim", type=int, default=512,
                   help="Hidden width of official_mamba_state.")
    p.add_argument("--simple_depth", type=int, default=4,
                   help="Number of residual Mamba blocks.")
    p.add_argument("--simple_dropout", type=float, default=0.0,
                   help="Dropout probability in official_mamba_state.")
    p.add_argument("--simple_residual", action=argparse.BooleanOptionalAction, default=True,
                   help="Keep the residual skip in each Mamba block.")
    p.add_argument("--mamba_d_state", type=int, default=16,
                   help="SSM state size N for the dependency-free Mamba block.")
    p.add_argument("--mamba_d_conv", type=int, default=4,
                   help="Causal depthwise convolution kernel size for the dependency-free Mamba block.")
    p.add_argument("--mamba_expand", type=int, default=2,
                   help="Inner-channel expansion factor for the dependency-free Mamba block.")
    p.add_argument("--mamba_burnin", type=int, default=64,
                   help="GT burn-in length used before chunked-BPTT closed-loop training.")
    p.add_argument("--mamba_bptt_horizon", type=int, default=8,
                   help="Closed-loop chunk length K for mamba_state_vector training.")
    p.add_argument("--mamba_train_starts_per_sequence", type=int, default=-1,
                   help="Number of random chunk starts per sequence. If <=0, use --ar_train_starts_per_sequence.")
    p.add_argument("--mamba_train_stride", type=int, default=1,
                   help="Candidate start stride for recurrent chunked-BPTT training.")
    p.add_argument("--mamba_loss_type", type=str, default="rel_l2", choices=["mse", "huber", "l1", "rel_l2"],
                   help="Per-step closed-loop loss for recurrent chunked-BPTT training.")
    p.add_argument("--mamba_loss_decay", type=float, default=1.0,
                   help="Optional geometric per-step loss weight. 1.0 means uniform over the K-step chunk.")

    # Residual-gradient routing for long-horizon recurrent BPTT.  This keeps
    # the residual identity gradient path open while optionally detaching the
    # nonlinear residual branch gradient at selected rollout steps.  Forward
    # values are unchanged; only the delayed-credit graph is changed.
    p.add_argument("--resgrad_routing", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable backward-only routing at internal residual merges (supported by official_mamba_state and unet_field).")
    p.add_argument("--resgrad_policy", type=str, default="all",
                   choices=["all", "dualwiener"],
                   help="Use all for full BPTT or dualwiener for Internal-DW.")
    # Historical paper runners include this value in saved command lines. The
    # cleaned models derive routing entirely from resgrad_policy, so the value
    # is accepted for checkpoint/CLI compatibility but is otherwise ignored.
    p.add_argument(
        "--resgrad_block_gate",
        type=float,
        default=1.0,
        help=argparse.SUPPRESS,
    )
    p.add_argument("--dual_wiener_ema", type=float, default=0.95,
                   help="EMA for per-(horizon,layer) 2x2 total/noise route moments used by dualwiener.")
    p.add_argument("--dual_wiener_residual_ema", type=float, default=0.99,
                   help="EMA for the lagged per-horizon diagonal output-residual covariance used by dualwiener (no sigma head).")
    p.add_argument("--dual_wiener_warmup_batches", type=int, default=8,
                   help="Number of fully dense training batches used to initialize residual covariance before dualwiener probes begin.")
    p.add_argument("--dual_wiener_probe_every", type=int, default=4,
                   help="Run the two extra quadratic VJP probes every N training batches for dualwiener.")
    p.add_argument("--dual_wiener_min_probes", type=int, default=1,
                   help="Keep gains fully open while the first N fully-open route probes are averaged arithmetically; after N probes, solve gains and track moments with --dual_wiener_ema. N=1 reproduces the original controller.")
    p.add_argument(
        "--dual_wiener_noise_model",
        choices=(
            "diagonal_gaussian",
            "spatial_spectrum",
            "lagged_residual_bootstrap",
        ),
        default="diagonal_gaussian",
        help=(
            "Output-noise covariance probe for dualwiener. The default keeps "
            "only per-coordinate residual variance; spatial_spectrum uses a "
            "lagged per-horizon/channel random-phase 2-D spectrum for field "
            "models; lagged_residual_bootstrap uses a centered residual from "
            "the previous batch and retains still more covariance without "
            "forming the full covariance matrix."
        ),
    )
    p.add_argument("--dual_wiener_max_horizon", type=int, default=1024,
                   help="Maximum rollout horizon allocated in the dualwiener statistics buffers.")
    p.add_argument("--recurrent_grad_checkpoint", action=argparse.BooleanOptionalAction, default=False,
                   help="Wrap each closed-loop rollout step in official_mamba_state training in torch.utils.checkpoint, recomputing forward activations during backward instead of storing them. This is a gradient-exact alternative to --resgrad_routing for reducing long-BPTT memory (roughly O(1)-in-K memory per step, ~2x forward compute), used as a baseline to compare against ResGrad's approximate-gradient routing.")
    p.add_argument("--bptt_grad_checkpoint", action=argparse.BooleanOptionalAction, default=False,
                   help="Same as --recurrent_grad_checkpoint for the U-Net closed-loop BPTT path.")
    p.add_argument("--bptt_detach_period", type=int, default=0,
                   help="For standard-AR rollout training, detach the closed-loop history every S steps while retaining the full forward rollout and every per-step loss. S<=0 gives exact BPTT; S=8 is TBPTT-8.")
    p.add_argument(
        "--artbp_expected_segment_length",
        type=int,
        default=0,
        help=(
            "Enable geometric ARTBP on temporal rollout edges.  A value L>1 "
            "cuts each edge with probability 1/L and multiplies every surviving "
            "backward edge by L/(L-1), giving an unbiased full-BPTT gradient in "
            "expectation while retaining the complete forward rollout and losses. "
            "L<=1 disables ARTBP."
        ),
    )

    # Exact full-BPTT rollout loss for comparison/diagnostics.
    p.add_argument("--bptt_loss", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable exact full-BPTT multi-step rollout loss for standard AR models.")
    p.add_argument("--bptt_lambda", type=float, default=0.0,
                   help="Weight of exact full-BPTT rollout loss.")
    p.add_argument("--bptt_horizon", type=int, default=0,
                   help="Full-BPTT rollout horizon. If 0, use --mamba_bptt_horizon.")
    p.add_argument("--bptt_loss_type", type=str, default="mse", choices=["mse", "huber", "l1", "rel_l2"],
                   help="Per-step loss used by exact full-BPTT rollout loss.")
    p.add_argument("--bptt_eval", action=argparse.BooleanOptionalAction, default=False,
                   help="Also compute exact BPTT loss in validation. Disabled by default because it is expensive.")
    p.add_argument("--forward_jacobian_lambda", type=float, default=0.0,
                   help="Weight of a randomized finite-difference one-step forward-Jacobian expansion penalty (0 disables it).")
    p.add_argument("--forward_jacobian_target", type=float, default=1.0,
                   help="Penalize random-direction forward Jacobian RMS gain above this value.")
    p.add_argument("--forward_jacobian_eps", type=float, default=1e-3,
                   help="Relative finite-difference radius for --forward_jacobian_lambda.")
    # Standard autoregressive training/evaluation options for U-Net.
    p.add_argument("--ar_loss", type=str, default="rel_l2", choices=["mse", "huber", "l1", "rel_l2"], help="One-step supervised loss for the U-Net backbone.")
    p.add_argument("--ar_one_step_lambda", type=float, default=1.0,
                   help="Weight of the ordinary one-step supervised AR loss. Keep 1 for normal training; set 0 for pure auxiliary fine-tuning such as TF+alignment.")
    p.add_argument("--ar_train_starts_per_sequence", type=int, default=1, help="How many random target times to train from each full trajectory batch.")
    p.add_argument("--ar_train_stride", type=int, default=1, help="Candidate target-time stride for random one-step training windows.")
    p.add_argument("--ar_train_random_starts", action=argparse.BooleanOptionalAction, default=True, help="Randomly sample one-step windows during AR training.")
    p.add_argument(
        "--ar_shared_rollout_start",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use one rollout start shared by every sample and DDP rank in an "
            "optimizer microbatch. This matches Exact/Static baselines to "
            "matched Full-BPTT and Internal-DW training."
        ),
    )
    p.add_argument("--ar_eval_stride", type=int, default=4, help="Start-time stride for horizon sweep evaluation of standard AR models.")
    p.add_argument(
        "--recurrent_eval_horizon_batch",
        type=int,
        default=int(os.environ.get("RECURRENT_EVAL_HORIZON_BATCH", "1")),
        help=(
            "Pack this many refresh horizons into the batch dimension during "
            "tied OfficialStateMamba evaluation. 1 preserves the serial reference "
            "path; try 3 on 40-GB GPUs or up to 9 on H200 for vector data. "
            "Large spatial fields should start with 2-3."
        ),
    )
    p.add_argument(
        "--recurrent_val_start_batch",
        type=int,
        default=int(os.environ.get("RECURRENT_VAL_START_BATCH", "1")),
        help=(
            "Pack this many rollout origins into the batch dimension for the "
            "ordinary OfficialStateMamba validation objective. This preserves "
            "the complete origin grid; 1 uses the serial reference."
        ),
    )
    p.add_argument(
        "--ar_optimizer",
        type=str,
        default="adam",
        choices=["sgd", "adam", "adamw"],
        help=(
            "Optimizer for standard autoregressive models. sgd is plain SGD "
            "unless --ar_sgd_momentum is set; it is useful when the backward "
            "gradient itself, rather than an adaptive optimizer map, is the "
            "experimental object."
        ),
    )
    p.add_argument(
        "--ar_sgd_momentum",
        type=float,
        default=0.0,
        help="Momentum for --ar_optimizer sgd. Zero gives the plain SGD update.",
    )
    p.add_argument(
        "--ar_sgd_nesterov",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Nesterov momentum for SGD; requires positive momentum.",
    )
    p.add_argument(
        "--ar_scheduler",
        type=str,
        default="step",
        choices=["none", "step", "cosine"],
        help=(
            "LR scheduler for the forecasting backbones. cosine uses "
            "CosineAnnealingLR over --num_epochs."
        ),
    )
    p.add_argument("--ar_step_size", type=int, default=100, help="StepLR step size.")
    p.add_argument("--ar_gamma", type=float, default=0.5, help="StepLR decay factor.")
    p.add_argument(
        "--ar_min_lr",
        type=float,
        default=1e-6,
        help="Final learning rate for --ar_scheduler cosine.",
    )

    # optimization
    p.add_argument("--num_epochs", "--epochs", dest="num_epochs", type=int, default=80)
    p.add_argument(
        "--probe_checkpoint_epochs", type=int, nargs="*", default=[],
        help=(
            "Additionally preserve frozen checkpoints at these epochs (0 saves "
            "the initialized model). Intended for training-dynamics probes; "
            "best.pth/last.pth behavior is unchanged."
        ),
    )
    p.add_argument("--base_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--local_batch_size", type=int, default=8)
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--early_stop_patience", type=int, default=0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--grad_accum_steps", type=int, default=1,
                   help="Accumulate gradients over this many micro-batches before each optimizer step, so effective batch = grad_accum_steps x local_batch_size. Lets a single-GPU run match the effective batch of a multi-GPU DDP run (e.g. 1 GPU with local_batch_size=8 and grad_accum_steps=4 matches 4 GPUs x 8). Default 1 = no accumulation.")
    p.add_argument("--log_gpu_memory", action=argparse.BooleanOptionalAction, default=True,
                   help="Log per-epoch CUDA peak/current memory to train_logs.jsonl. Use --no-log_gpu_memory to disable.")
    p.add_argument(
        "--synchronize_epoch_timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Synchronize CUDA immediately before and after each measured training "
            "epoch. This makes train/epoch_wall_seconds a strict end-to-end GPU "
            "wall-clock measurement for compute-overhead benchmarks."
        ),
    )
    p.add_argument(
        "--fast_train_logging",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "During training only, retain a minimal set of scalar diagnostics "
            "on device and materialize them once per epoch. Validation/test "
            "logging is unchanged. This avoids per-batch CUDA synchronizations."
        ),
    )
    p.add_argument("--compact_train_logs", action=argparse.BooleanOptionalAction, default=True,
                   help="Write compact train_logs.jsonl by dropping disabled/zero diagnostic keys.")
    p.add_argument("--compact_recurrent_logging", action="store_true",
                   help="Skip optional recurrent diagnostics and epoch plots; retain loss, checkpoints and DW calibration.")
    p.add_argument("--eval_free_rollout_curves", action=argparse.BooleanOptionalAction, default=True,
                   help="During test evaluation, save free-rollout per-step corr/error curves for standard AR models.")

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.mode == "test":
        world_size = 1
    else:
        world_size = max(torch.cuda.device_count(), 1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the current training entry point.")
    # mp.spawn(worker, args=(args, world_size), nprocs=world_size, join=True)
    if world_size > 1:
        mp.spawn(worker, args=(args, world_size), nprocs=world_size, join=True)
    else:
        worker(0, args, world_size)
