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
from internal_dw.training.losses import resolve_koopman_long_loss
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


def _infer_old_rr_stim_dim_from_batch(tensor, preferred_dim=0):
    """Infer the *post-pooling* per-time stimulus dim used by old RR.

    Old ``RidgeSequenceModel`` used:
        z_window.mean(axis=2) if z_window.ndim == 4 else z_window

    In the current loader, tokenized features may already be flattened as
    [B,T,N*D] instead of [B,T,N,D].  If the flattened dimension is divisible by
    the preferred model-side feature dim (normally 1664), we recover the old
    behavior by interpreting it as [N,D] and mean-pooling over N.
    """
    if tensor is None:
        return 0
    if tensor.dim() < 3:
        return 1
    if tensor.dim() == 4:
        return int(tensor.shape[3])

    flat_dim = 1
    for v in tensor.shape[2:]:
        flat_dim *= int(v)

    preferred_dim = int(preferred_dim or 0)
    if preferred_dim > 0 and flat_dim != preferred_dim and flat_dim % preferred_dim == 0:
        return preferred_dim
    return int(flat_dim)


def maybe_infer_ridge_window_dims(args, train_loader, rank=0):
    """Infer old-RR post-pooling stimulus feature dimension.

    Important: do NOT blindly set stim_dim to the raw flattened tensor dim.
    For HCP CLIP/SDXL features the loader can expose [B,T,425984], which is
    256*1664 flattened tokens.  Old RR would receive [B,T,256,1664] and apply
    mean(axis=2), so the correct per-time dim is 1664.
    """
    if str(getattr(args, "model_name", "")) != "ridge_window":
        return
    if not bool(getattr(args, "dataset_has_external_input", False)):
        return
    from internal_dw.data_utils.state_ops import unpack_batch
    try:
        batch = next(iter(train_loader))
    except StopIteration:
        return
    _state, stim, _a, _b = unpack_batch(batch)
    if stim is None:
        return
    old = int(getattr(args, "stim_dim", 0))
    inferred = _infer_old_rr_stim_dim_from_batch(stim, preferred_dim=old)
    if inferred <= 0:
        return
    args.stim_dim = inferred
    if rank == 0:
        raw_per_time = int(stim[0, 0].numel()) if stim.dim() >= 3 else int(stim[0].numel())
        if old != inferred:
            print(
                f"[RidgeWindow] inferred old-RR pooled stim_dim={inferred} "
                f"from raw per-time dim={raw_per_time}; overriding previous stim_dim={old}",
                flush=True,
            )
        else:
            print(
                f"[RidgeWindow] using old-RR pooled stim_dim={inferred} "
                f"from raw per-time dim={raw_per_time}",
                flush=True,
            )


@torch.no_grad()
def maybe_fit_fno_normalizer(model, train_loader, args, rank=0):
    """Fit per-channel Gaussian normalizer for 2D AR field models.

    FNO and U-Net both keep train-set normalization statistics as model
    buffers, so checkpoints preserve the exact preprocessing used at train time.
    """
    raw = unwrap_model(model)
    if not getattr(raw, "is_standard_autoregressive", False):
        return
    if not hasattr(raw, "set_state_normalizer"):
        return
    model_name = str(getattr(args, "model_name", ""))
    normalize = bool(getattr(args, "fno_normalize", True))
    if model_name == "unet_field":
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
    # FNO normalizer is fitted on the full training split, not a rank-local shard.
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


@torch.no_grad()
def maybe_fit_ridge_window_standardizer(model, train_loader, args, rank=0):
    """Fit old-RR-style StandardScaler statistics for RidgeWindowModel.

    The old RR dynamic predictor standardizes flattened input features before
    applying Ridge.  RidgeWindowModel keeps rollout states in raw fMRI
    coordinates, but its learned weights live in standardized feature
    coordinates; these stats make the trainable model equivalent to that
    feature convention.
    """
    raw = unwrap_model(model)
    if not bool(getattr(raw, "is_ridge_window", False)):
        return
    if not bool(getattr(args, "ridge_window_standardize", True)):
        return
    if bool(getattr(raw, "standardizer_fitted", torch.tensor(False)).item()):
        return

    from internal_dw.data_utils.state_ops import get_batch_time_shape, time_window, unpack_batch, zero_external_input_like

    W = int(args.window_size)
    stride = max(int(getattr(args, "koopman_train_stride", 1)), 1)
    device = torch.device(f"cuda:{rank}")

    state_sum = torch.zeros(raw.latent_dim, device=device)
    state_sumsq = torch.zeros(raw.latent_dim, device=device)
    state_count = torch.zeros((), device=device)

    if raw.has_external_input:
        stim_sum = torch.zeros(raw.stimulus_window_dim, device=device)
        stim_sumsq = torch.zeros(raw.stimulus_window_dim, device=device)
        stim_count = torch.zeros((), device=device)
    else:
        stim_sum = stim_sumsq = stim_count = None

    for batch in train_loader:
        state, stim, _, _ = unpack_batch(batch)
        state = state.to(device, non_blocking=True).float()
        if stim is None:
            stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))
        stim = stim.to(device, non_blocking=True).float()
        _B, T, _ = get_batch_time_shape(state)
        if T <= W:
            continue
        for t in range(W, T, stride):
            s_win = time_window(state, t - W, t).reshape(state.shape[0], -1)
            state_sum += s_win.sum(dim=0)
            state_sumsq += s_win.square().sum(dim=0)
            state_count += s_win.shape[0]

            if raw.has_external_input:
                u_win = raw._pool_external_window(time_window(stim, t - W, t))
                stim_sum += u_win.sum(dim=0)
                stim_sumsq += u_win.square().sum(dim=0)
                stim_count += u_win.shape[0]

    if dist.is_available() and dist.is_initialized():
        for x in [state_sum, state_sumsq, state_count]:
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        if raw.has_external_input:
            for x in [stim_sum, stim_sumsq, stim_count]:
                dist.all_reduce(x, op=dist.ReduceOp.SUM)

    if float(state_count.item()) <= 0:
        raise RuntimeError("No samples found while fitting RidgeWindow standardizer")

    state_mean = state_sum / state_count.clamp_min(1.0)
    state_var = (state_sumsq / state_count.clamp_min(1.0) - state_mean.square()).clamp_min(1e-12)
    state_std = torch.sqrt(state_var)

    if raw.has_external_input:
        stim_mean = stim_sum / stim_count.clamp_min(1.0)
        stim_var = (stim_sumsq / stim_count.clamp_min(1.0) - stim_mean.square()).clamp_min(1e-12)
        stim_std = torch.sqrt(stim_var)
    else:
        stim_mean = stim_std = None

    raw.set_feature_standardizer(state_mean, state_std, stim_mean, stim_std)
    if rank == 0:
        msg = (
            f"[RidgeWindow] fitted StandardScaler stats: n={int(state_count.item())}, "
            f"state_dim={raw.latent_dim}"
        )
        if raw.has_external_input:
            msg += f", stim_dim={raw.stimulus_window_dim}"
        print(msg, flush=True)


def _broadcast_model_state(model, src=0):
    """Broadcast parameters and buffers after rank-0-only initialization."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    raw = unwrap_model(model)
    for tensor in raw.state_dict().values():
        if torch.is_tensor(tensor):
            dist.broadcast(tensor, src=src)


def maybe_fit_ridge_window_closed_form(model, args, rank=0, world_size=1):
    """Fit old-style StandardScaler+Ridge and load it into RidgeWindowModel.

    This reproduces the old RR dynamic predictor exactly: features are
    [stim_window, fmri_history_window] after the old stimulus pooling rule,
    a StandardScaler is fit on X, and sklearn Ridge is fit on raw fMRI targets.

    In distributed runs, rank 0 builds an unsharded train loader, fits the
    sklearn model on the full training split, copies the solution into the
    PyTorch RidgeWindowModel, and broadcasts all parameters/buffers.
    """
    raw = unwrap_model(model)
    if not bool(getattr(raw, "is_ridge_window", False)):
        return
    if not bool(getattr(args, "ridge_window_fit_closed_form", False)):
        return
    if args.model_ckpt_path:
        return

    if rank == 0:
        import numpy as np
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        from internal_dw.data_utils.state_ops import get_batch_time_shape, time_window, unpack_batch, zero_external_input_like

        # Use an unsharded loader so the closed-form RR baseline is fit on the
        # complete training split, not only rank-0's distributed shard.
        full_train_loader, _, _ = build_dataloaders(args, rank=0, world_size=1)
        W = int(args.window_size)
        stride = max(int(getattr(args, "ridge_window_fit_stride", 1)), 1)
        max_samples = int(getattr(args, "ridge_window_max_fit_samples", 0))

        X_list, Y_list = [], []
        total = 0
        for batch in full_train_loader:
            state, stim, _, _ = unpack_batch(batch)
            state = state.float()
            if stim is None:
                stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))
            stim = stim.float()
            B, T, _ = get_batch_time_shape(state)
            if T <= W:
                continue
            for t in range(W, T, stride):
                s_win = time_window(state, t - W, t).reshape(B, -1)
                u_win = raw._pool_external_window(time_window(stim, t - W, t)) if raw.has_external_input else None
                if u_win is not None:
                    X = torch.cat([u_win, s_win], dim=1)
                else:
                    X = s_win
                Y = state[:, t].reshape(B, -1)
                X_list.append(X.cpu().numpy().astype(np.float32, copy=False))
                Y_list.append(Y.cpu().numpy().astype(np.float32, copy=False))
                total += B
                if max_samples > 0 and total >= max_samples:
                    break
            if max_samples > 0 and total >= max_samples:
                break

        if not X_list:
            raise RuntimeError("No samples found for RidgeWindow closed-form fit")
        X_train = np.concatenate(X_list, axis=0)
        Y_train = np.concatenate(Y_list, axis=0)
        if max_samples > 0 and X_train.shape[0] > max_samples:
            X_train = X_train[:max_samples]
            Y_train = Y_train[:max_samples]

        pipe = Pipeline([
            ("x_scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=float(args.ridge_alpha), fit_intercept=True)),
        ])
        print(
            f"[RidgeWindow] closed-form old-style RR fit: X={X_train.shape}, Y={Y_train.shape}, "
            f"alpha={float(args.ridge_alpha)}",
            flush=True,
        )
        pipe.fit(X_train, Y_train)

        scaler = pipe.named_steps["x_scaler"]
        ridge = pipe.named_steps["ridge"]
        mean = torch.as_tensor(scaler.mean_, device=raw.bias.device, dtype=raw.bias.dtype)
        scale_np = scaler.scale_.copy()
        scale_np[scale_np < 1e-12] = 1.0
        scale = torch.as_tensor(scale_np, device=raw.bias.device, dtype=raw.bias.dtype)
        coef = torch.as_tensor(ridge.coef_, device=raw.bias.device, dtype=raw.bias.dtype)
        intercept = torch.as_tensor(ridge.intercept_, device=raw.bias.device, dtype=raw.bias.dtype)

        stim_dim = raw.stimulus_window_dim if raw.has_external_input else 0
        state_dim = raw.latent_dim
        expected = stim_dim + state_dim
        if mean.numel() != expected or coef.shape != (raw.state_dim, expected):
            raise RuntimeError(
                f"Closed-form RR shape mismatch: mean={tuple(mean.shape)}, coef={tuple(coef.shape)}, "
                f"expected features={expected}, output={raw.state_dim}"
            )

        # All writes to Parameters/Buffers must be done under no_grad.
        with torch.no_grad():
            if raw.has_external_input:
                stim_mean = mean[:stim_dim]
                stim_std = scale[:stim_dim]
                state_mean = mean[stim_dim:]
                state_std = scale[stim_dim:]
                raw.set_feature_standardizer(state_mean, state_std, stim_mean, stim_std)
                # Feature order is [stim_feat, state_feat].
                raw.stim_weight.copy_(coef[:, :stim_dim])
                raw.state_weight.copy_(coef[:, stim_dim:])
            else:
                raw.set_feature_standardizer(mean, scale, None, None)
                raw.state_weight.copy_(coef)
            raw.bias.copy_(intercept)
            raw.log_force_scale.zero_()
        print(
            f"[RidgeWindow] loaded closed-form RR into PyTorch model: "
            f"state_weight={tuple(raw.state_weight.shape)}, "
            f"stim_weight={None if raw.stim_weight is None else tuple(raw.stim_weight.shape)}",
            flush=True,
        )

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        _broadcast_model_state(model, src=0)
        dist.barrier()


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
            if str(getattr(args, "model_name", "")) in [
                "official_mamba_state",
                "official_pc_mamba_state",
                "official_atlas_mamba_state",
                "official_tangent_atlas_mamba_state",
                "official_shadow_perturb_mamba",
                "shadow_perturb_mamba",
                "official_mamba_fixeda_perturb",
                "cyclic_graph_ar",
            ]:
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
    maybe_infer_ridge_window_dims(args, train_loader, rank=rank)
    if bool(getattr(args, "stageb_joint_potential_ar", False)):
        args.stageb_original_roi_dim = int(args.roi_dim)
        args.roi_dim = int(args.roi_dim) + int(getattr(args, "potential_latent_dim", 64))
        if rank == 0:
            print(f"[StageB] joint potential AR model_dim={args.roi_dim} = original_dim={args.stageb_original_roi_dim} + latent_dim={getattr(args, 'potential_latent_dim', 64)}", flush=True)
    loss_cfg = resolve_koopman_long_loss(args)
    if rank == 0:
        print("[Dataset] "
              f"dataset={args.dataset}; task_type={args.dataset_task_type}; "
              f"evaluator={args.dataset_evaluator_name}; has_external_input={args.dataset_has_external_input}")
        print("[KG Loss] "
              f"requested={loss_cfg['requested']}; resolved={loss_cfg['resolved']}; "
              f"folded_component={loss_cfg['folded_component']}; note={loss_cfg['note']}")
    model = build_model(args, rank=rank)
    _ddp_stage_sync("after_build_model", rank, world_size)

    if args.model_ckpt_path:
        load_checkpoint(model, args.model_ckpt_path, map_location=f"cuda:{rank}", strict=not args.non_strict_ckpt)
        if rank == 0:
            print(f"Loaded checkpoint: {args.model_ckpt_path}")
    elif getattr(args, "statetok_ae_ckpt", "") and str(getattr(args, "model_name", "")).startswith("state_sequence_"):
        load_checkpoint(model, args.statetok_ae_ckpt, map_location=f"cuda:{rank}", strict=False)
        if rank == 0:
            print(f"Loaded state-sequence AE init checkpoint: {args.statetok_ae_ckpt}")
    elif getattr(args, "clocklat_ae_ckpt", "") and str(getattr(args, "model_name", "")).startswith("clock_latent_ar_"):
        load_checkpoint(model, args.clocklat_ae_ckpt, map_location=f"cuda:{rank}", strict=False)
        if rank == 0:
            print(f"Loaded clock-latent AE init checkpoint: {args.clocklat_ae_ckpt}")
    else:
        maybe_fit_fno_normalizer(model, train_loader, args, rank=rank)
        # Closed-form old-style RR fit also sets the StandardScaler buffers.
        # If it is disabled, fit only the StandardScaler for trainable RR.
        maybe_fit_ridge_window_closed_form(model, args, rank=rank, world_size=world_size)
        maybe_fit_ridge_window_standardizer(model, train_loader, args, rank=rank)

    configure_trainable(model, args, rank=rank)
    _ddp_stage_sync("after_configure_trainable", rank, world_size)
    _ddp_stage_sync("before_DDP_wrap", rank, world_size)
    model = maybe_wrap(model, rank, world_size)
    _ddp_stage_sync("after_DDP_wrap", rank, world_size)

    do_train = args.mode in ["train", "train_and_test"]
    if (
        str(getattr(args, "model_name", "")) == "ridge_window"
        and bool(getattr(args, "ridge_window_fit_closed_form", False))
        and not bool(getattr(args, "ridge_window_finetune", False))
    ):
        do_train = False
        if rank == 0:
            print("[RidgeWindow] closed-form-only mode: skipping gradient training", flush=True)

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
    p.add_argument("--save_root", type=str, default="experiments/koopman_gram")
    p.add_argument("--model_name", type=str, default="koopman_raw_gramian", choices=list_models())
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

    p.add_argument("--resgrad_cut_state", action=argparse.BooleanOptionalAction, default=False,
                   help="residual_gru only: also detach the carried state at gate 0, mirroring "
                        "official_state_mamba. Default False = the paper's stated I+mJ_F intervention, "
                        "which leaves the identity path (and hence cross-time gradient) open.")

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
    p.add_argument("--field_base_channels", type=int, default=32,
                   help="Base convolution width for koopman_field_gramian.")
    p.add_argument("--field_seed_size", type=int, default=8,
                   help="Latent decoder seed grid size for koopman_field_gramian.")

    # FNO autoregressive baseline options
    p.add_argument("--fno_width", type=int, default=64, help="FNO hidden channel width.")
    p.add_argument("--fno_modes1", type=int, default=16, help="Number of Fourier modes along height.")
    p.add_argument("--fno_modes2", type=int, default=16, help="Number of Fourier modes along width/rFFT axis.")
    p.add_argument("--fno_layers", type=int, default=4, help="Number of FNO spectral blocks.")
    p.add_argument("--fno_padding", type=int, default=9, help="Spatial padding used by FNO. Original FNO examples commonly use padding=9 for non-periodic domains.")
    p.add_argument("--fno_hidden_channels", type=int, default=128, help="Projection/lifting channel width used by official FNO implementations.")
    p.add_argument("--fno_use_grid", action=argparse.BooleanOptionalAction, default=True, help="Concatenate normalized coordinate channels to FNO input.")
    p.add_argument("--fno_backend", type=str, default="neuralop", choices=["neuralop", "original"], help="FNO implementation backend: official neuraloperator package or local original-style FNO2d.")
    p.add_argument("--fno_normalize", action=argparse.BooleanOptionalAction, default=True, help="Use original-FNO-style per-channel train-set Gaussian normalization inside the FNO model.")

    # U-Net autoregressive baseline options
    p.add_argument("--unet_base_channels", type=int, default=64, help="Base channel width for unet_field.")
    p.add_argument("--unet_depth", type=int, default=4, help="Number of U-Net resolution levels for unet_field.")
    p.add_argument("--unet_channel_mult", type=int, default=2, help="Channel multiplier between U-Net levels.")
    p.add_argument("--unet_groups", type=int, default=8, help="GroupNorm group count for unet_field.")
    p.add_argument("--unet_use_grid", action=argparse.BooleanOptionalAction, default=True, help="Concatenate normalized coordinate channels to U-Net input.")
    p.add_argument("--unet_normalize", action=argparse.BooleanOptionalAction, default=True, help="Use per-channel train-set Gaussian normalization inside the U-Net model.")

    # Field periodic error homogenization (PEH) for The Well / PDE local-loss tests.
    p.add_argument("--field_peh_loss", action=argparse.BooleanOptionalAction, default=False,
                   help="Add field PEH free-rollout closure regularization on top of local one-step AR loss.")
    p.add_argument("--field_peh_eval", action=argparse.BooleanOptionalAction, default=True,
                   help="Log PEH rollout diagnostics during validation even when the PEH training lambda is zero.")
    p.add_argument("--field_peh_lambda", type=float, default=0.0,
                   help="Weight for field PEH closure loss. Main local one-step loss is still controlled by --ar_one_step_lambda.")
    p.add_argument("--field_peh_horizon", type=int, default=16,
                   help="Free-rollout horizon K used by field PEH.")
    p.add_argument("--field_peh_type", type=str, default="raw", choices=["raw", "wpeh", "ipeh"],
                   help="PEH variant: raw closure, per-channel whitened closure, or innovation closure.")
    p.add_argument("--field_peh_denom", type=str, default="error_sum_detach", choices=["error_sum_detach", "target", "sqrtk"],
                   help="Normalization denominator for the PEH closure norm.")
    p.add_argument("--field_peh_innovation_rho", type=float, default=0.8,
                   help="rho for iPEH innovations e_k - rho e_{k-1}.")
    p.add_argument("--field_peh_start_epoch", type=int, default=1,
                   help="Epoch at which PEH lambda starts.")
    p.add_argument("--field_peh_ramp_epochs", type=int, default=0,
                   help="Linear ramp length for the PEH lambda.")
    p.add_argument("--field_peh_eps", type=float, default=1e-8,
                   help="Numerical epsilon for PEH norms.")
    # Low-dimensional finite-horizon path-generator model.
    p.add_argument("--pathgen_horizon", type=int, default=8, help="Number of future steps generated by path_generator_field during training.")
    p.add_argument("--pathgen_generator_size", type=int, default=8, help="Coarse grid size G for low-dimensional generator velocities [K,C,G,G].")
    p.add_argument("--pathgen_base_channels", type=int, default=64, help="Base channel width for path_generator_field encoder.")
    p.add_argument("--pathgen_code_channels", type=int, default=64, help="Bottleneck/code channel width for path_generator_field.")
    p.add_argument("--pathgen_groups", type=int, default=8, help="GroupNorm group count for path_generator_field.")
    p.add_argument("--pathgen_use_grid", action=argparse.BooleanOptionalAction, default=True, help="Concatenate normalized coordinate channels to the path generator input.")
    p.add_argument("--pathgen_normalize", action=argparse.BooleanOptionalAction, default=True, help="Use per-channel train-set Gaussian normalization inside the path generator.")
    p.add_argument("--pathgen_velocity_scale", type=float, default=1.0, help="Scale applied to generated normalized coarse velocity increments.")
    p.add_argument("--directpath_residual_from_last", action=argparse.BooleanOptionalAction, default=False, help="For direct_path_decoder_field only: decode future frames as residual offsets from the last history frame instead of absolute normalized states.")
    p.add_argument("--pathgen_lambda_path", type=float, default=1.0, help="Weight for paired finite-horizon path supervision.")
    p.add_argument("--pathgen_lambda_mmsbm", type=float, default=0.0, help="Weight for low-dimensional diagonal-Gaussian multi-marginal path error.")
    p.add_argument("--pathgen_lambda_energy", type=float, default=0.0, help="Weight for generator velocity energy penalty.")
    p.add_argument("--pathgen_lambda_code", type=float, default=0.0, help="Weight for path-code bottleneck energy penalty.")
    # State-token path model options.  These implement learned temporal receptive-field
    # state tokens m_i with metadata (center time, decay/locality, activation gate).
    p.add_argument("--statetok_num_tokens", type=int, default=8,
                   help="Maximum number of learned state tokens L for state_token_path_* models.")
    p.add_argument("--statetok_generator_size", type=int, default=8,
                   help="Coarse grid size G for state_token_path_field token fields.")
    p.add_argument("--statetok_base_channels", type=int, default=64,
                   help="Base CNN width for state_token_path_field.")
    p.add_argument("--statetok_code_channels", type=int, default=64,
                   help="Code/token CNN width for state_token_path_field.")
    p.add_argument("--statetok_groups", type=int, default=8,
                   help="GroupNorm group count for state_token_path_field.")
    p.add_argument("--statetok_use_grid", action=argparse.BooleanOptionalAction, default=True,
                   help="Concatenate normalized coordinate grid to state_token_path_field input.")
    p.add_argument("--statetok_normalize", action=argparse.BooleanOptionalAction, default=True,
                   help="Use per-channel train-set Gaussian normalization for state_token_path_field.")
    p.add_argument("--statetok_hidden_dim", type=int, default=512,
                   help="MLP hidden width for state_token_path_vector.")
    p.add_argument("--statetok_mlp_layers", type=int, default=2,
                   help="Number of MLP blocks for state_token_path_vector encoder.")
    p.add_argument("--statetok_decay_min", type=float, default=0.5,
                   help="Minimum temporal decay/locality of each state token. Larger decay means narrower receptive field.")
    p.add_argument("--statetok_decay_max", type=float, default=80.0,
                   help="Maximum temporal decay/locality of each state token.")
    p.add_argument("--statetok_residual_from_last", action=argparse.BooleanOptionalAction, default=True,
                   help="Predict token states as residuals from the last history state.")
    p.add_argument("--statetok_stage", type=str, default="ae", choices=["ae", "transition", "finetune"],
                   help="For state_sequence_ae_*: train trajectory AE, frozen-prior transition, or light finetune.")
    p.add_argument("--statetok_ae_ckpt", type=str, default="",
                   help="Checkpoint from statetok_stage=ae to initialize transition/finetune stages.")
    p.add_argument("--statetok_freeze_ae", action=argparse.BooleanOptionalAction, default=True,
                   help="Freeze posterior encoder/main decoder during transition stage.")
    p.add_argument("--statetok_tune_adapter", action=argparse.BooleanOptionalAction, default=True,
                   help="When AE is frozen, allow small LoRA-like decoder adapters to train.")
    p.add_argument("--statetok_adapter_rank", type=int, default=0,
                   help="LoRA-like decoder adapter rank for state_sequence_ae_* models. 0 disables adapters.")
    p.add_argument("--statetok_adapter_alpha", type=float, default=0.1,
                   help="Scale for LoRA-like decoder adapter residual.")
    p.add_argument("--statetok_code_dim", type=int, default=256,
                   help="Vector code dimension for state_sequence_ae_vector.")
    p.add_argument("--statetok_lambda_rec", type=float, default=1.0,
                   help="AE-stage reconstruction loss weight for state_sequence_ae_*.")
    p.add_argument("--statetok_lambda_forecast", type=float, default=1.0,
                   help="Transition-stage decoded forecast loss weight for state_sequence_ae_*.")
    p.add_argument("--statetok_lambda_latent", type=float, default=0.1,
                   help="Transition-stage prior-vs-posterior state-token content/center/decay matching weight.")
    p.add_argument("--statetok_c_tau", type=float, default=0.1,
                   help="Coefficient for center/tau matching inside the state-token latent loss.")
    p.add_argument("--statetok_c_lambda", type=float, default=0.01,
                   help="Coefficient for log-decay/lambda matching inside the state-token latent loss.")
    p.add_argument("--statetok_lambda_weight", type=float, default=0.0,
                   help="Transition-stage posterior-vs-prior decoder temporal weight-map matching weight for state_sequence_ae_*. This directly aligns W[k,i].")
    p.add_argument("--statetok_lambda_gate", type=float, default=0.0,
                   help="Transition-stage posterior-vs-prior gate matching weight for state_sequence_ae_*.")
    p.add_argument("--statetok_lambda_meta", type=float, default=0.0,
                   help="Weak regularization weight for ordered/covered token metadata.")
    p.add_argument("--statetok_min_separation", type=float, default=0.02,
                   help="Minimum normalized separation between ordered state-token centers.")
    p.add_argument("--statetok_detach_posterior", action=argparse.BooleanOptionalAction, default=True,
                   help="Detach posterior tokens while training prior transition.")
    p.add_argument("--statetok_use_masked_init", action=argparse.BooleanOptionalAction, default=False,
                   help="For state_sequence_ae_* transition stage: initialize m_1 with the Stage-1 encoder applied to [history, zero-masked future], then autoregressively generate m_2..m_L. This is not a direct full-token predictor.")
    p.add_argument("--statetok_ae_mask_prob", type=float, default=0.0,
                   help="AE-stage ignore-mask probability. When >0, random time slots are zeroed in the AE input and target so the Stage-1 encoder/decoder learns that zero-masked slots are invalid rather than content to encode.")
    p.add_argument("--statetok_ae_mask_min_keep", type=int, default=1,
                   help="Minimum number of unmasked time slots per sample for --statetok_ae_mask_prob.")
    p.add_argument("--statetok_ae_mask_zero_weight", type=float, default=0.1,
                   help="Weight for forcing masked AE-stage time slots to reconstruct zero. Visible reconstruction remains unweighted.")

    # Matched clock-time latent AR baseline options.  This is the main AR-class
    # comparison for state_sequence_ae_*: it uses AE + latent rollout, but the
    # latent transition still occurs at every fixed clock step.
    p.add_argument("--clocklat_stage", type=str, default="ae", choices=["ae", "transition", "finetune"],
                   help="For clock_latent_ar_*: train frame AE, frozen latent clock-step transition, or finetune.")
    p.add_argument("--clocklat_ae_ckpt", type=str, default="",
                   help="Checkpoint from clocklat_stage=ae to initialize transition/finetune stages.")
    p.add_argument("--clocklat_freeze_ae", action=argparse.BooleanOptionalAction, default=True,
                   help="Freeze clock-latent frame encoder/main decoder during transition stage.")
    p.add_argument("--clocklat_tune_adapter", action=argparse.BooleanOptionalAction, default=True,
                   help="When clock-latent AE is frozen, allow small LoRA-like decoder adapters to train.")
    p.add_argument("--clocklat_generator_size", type=int, default=8,
                   help="Coarse latent grid size G for clock_latent_ar_field.")
    p.add_argument("--clocklat_base_channels", type=int, default=64,
                   help="Base CNN width for clock_latent_ar_field.")
    p.add_argument("--clocklat_code_channels", type=int, default=64,
                   help="Code/channel width for clock_latent_ar_field.")
    p.add_argument("--clocklat_code_dim", type=int, default=256,
                   help="Vector latent code dimension for clock_latent_ar_vector.")
    p.add_argument("--clocklat_hidden_dim", type=int, default=512,
                   help="GRU/MLP hidden width for clock_latent_ar_*.")
    p.add_argument("--clocklat_adapter_rank", type=int, default=0,
                   help="LoRA-like decoder adapter rank for clock_latent_ar_* models. 0 disables adapters.")
    p.add_argument("--clocklat_adapter_alpha", type=float, default=0.1,
                   help="Scale for clock-latent LoRA-like decoder adapter residual.")
    p.add_argument("--clocklat_lambda_rec", type=float, default=1.0,
                   help="AE-stage reconstruction loss weight for clock_latent_ar_*.")
    p.add_argument("--clocklat_lambda_forecast", type=float, default=1.0,
                   help="Transition-stage decoded forecast loss weight for clock_latent_ar_*.")
    p.add_argument("--clocklat_lambda_latent", type=float, default=0.1,
                   help="Transition-stage predicted-vs-posterior clock-latent matching weight.")
    p.add_argument("--clocklat_detach_posterior", action=argparse.BooleanOptionalAction, default=True,
                   help="Detach posterior clock-time latent codes while training latent AR prior.")
    p.add_argument("--fno_folded_loss", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable training-only folded latent rollout supervision for FNO. Inference remains unchanged.")
    p.add_argument("--fno_fold_lambda", type=float, default=-1.0,
                   help="Weight for FNO folded loss. If negative, use --koopman_lambda_gram.")
    p.add_argument("--fno_fold_horizon", type=int, default=0,
                   help="Latent folded horizon for FNO. If 0, use --koopman_gramian_horizon.")
    p.add_argument("--fno_fold_pool_size", type=int, default=8,
                   help="Adaptive-average-pool size for fixed FNO folded observable.")
    p.add_argument("--fno_fold_init_scale", type=float, default=0.98,
                   help="Initial scale for the folded latent propagation matrix A.")

    # FNO finite-time sensitivity / no-adapter training losses
    p.add_argument("--fno_ftg_loss", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable finite-time sensitivity losses for FNO without adding an inference-time or training-time transition adapter.")
    p.add_argument("--fno_ftg_lambda_amp", type=float, default=0.0,
                   help="Weight for the detached finite-time amplification weighting term. This approximates the first FTG differential term.")
    p.add_argument("--fno_ftg_lambda_bound", type=float, default=0.0,
                   help="Weight for the finite-difference Jacobian amplification bound term. This is a cheap surrogate for controlling the operator-gradient term.")
    p.add_argument("--fno_ftg_horizon", type=int, default=0,
                   help="Finite-time sensitivity horizon. If 0, use --fno_fold_horizon, then --koopman_gramian_horizon.")
    p.add_argument("--fno_ftg_rho", type=float, default=1.05,
                   help="Target local amplification bound for the finite-difference FTG bound loss.")
    p.add_argument("--fno_ftg_bound_eps", type=float, default=1e-3,
                   help="Finite-difference epsilon for estimating local history-transition amplification in the FTG bound loss.")
    p.add_argument("--fno_ftg_eps", type=float, default=1e-8,
                   help="Numerical epsilon for FTG norm ratios.")
    p.add_argument("--fno_ftg_eval", action=argparse.BooleanOptionalAction, default=False,
                   help="Also compute FTG training losses during validation/evaluation loss computation. Disabled by default because JVP/FD probes are expensive.")

    # Clean CNN/GRU/Transformer/TCN baseline options used by the compiled-backward-graph experiments.
    p.add_argument("--simple_hidden_dim", type=int, default=512,
                   help="Hidden width for clean vector baselines; for field CNN/ConvGRU this is used as channel width unless overridden by choosing smaller values.")
    p.add_argument("--simple_depth", type=int, default=4,
                   help="Depth/layer count for clean CNN/Transformer/TCN baselines.")
    p.add_argument("--simple_nhead", type=int, default=8,
                   help="Attention heads for transformer_vector.")
    p.add_argument("--simple_ff_mult", type=int, default=2,
                   help="Feed-forward multiplier for clean Transformer baselines. The older HCP raw Transformer used 2 rather than 4.")
    p.add_argument("--simple_kernel_size", type=int, default=3,
                   help="Temporal convolution kernel size for tcn_vector.")
    p.add_argument("--resnet_hidden_mult", type=int, default=2,
                   help="Inner hidden-layer width multiplier for each ResidualMLPBlock in resnet_vector (inner_dim = simple_hidden_dim * resnet_hidden_mult).")
    p.add_argument("--simple_dropout", type=float, default=0.1,
                   help="Dropout for clean vector baselines.")
    p.add_argument("--simple_groups", type=int, default=8,
                   help="GroupNorm groups for clean field baselines.")
    p.add_argument("--simple_use_grid", action=argparse.BooleanOptionalAction, default=True,
                   help="Concatenate coordinate grid for clean field baselines.")
    p.add_argument("--simple_normalize", action=argparse.BooleanOptionalAction, default=True,
                   help="Use train-set per-channel normalization for clean field baselines.")
    p.add_argument("--simple_residual", action=argparse.BooleanOptionalAction, default=True,
                   help="Predict residual from last state/frame for clean AR baselines.")

    # Causal-query predictive subspace attention.
    p.add_argument("--subspace_hidden_dim", type=int, default=512,
                   help="Causal query GRU hidden width for predictive_subspace_attention.")
    p.add_argument("--subspace_latent_dim", type=int, default=128,
                   help="Predictive latent state dimension z_t.")
    p.add_argument("--subspace_num_refs", type=int, default=128,
                   help="Number of learned reference tokens in the global predictive dictionary.")
    p.add_argument("--subspace_key_dim", type=int, default=128,
                   help="Attention key/query dimension for dictionary lookup.")
    p.add_argument("--subspace_transition_rank", type=int, default=16,
                   help="Low-rank correction rank for the compact latent transition A = diag + UV^T.")
    p.add_argument("--subspace_diag_init", type=float, default=0.98,
                   help="Initial diagonal value of the latent transition.")
    p.add_argument("--subspace_max_diag", type=float, default=0.999,
                   help="Maximum absolute diagonal transition value via tanh parameterization.")
    p.add_argument("--subspace_temperature", type=float, default=1.0,
                   help="Attention temperature for causal query over reference dictionary.")
    p.add_argument("--subspace_entropy_weight", type=float, default=0.0,
                   help="Optional weight for -entropy of dictionary attention; positive encourages broader reference usage.")
    p.add_argument("--subspace_current_rec_weight", type=float, default=0.0,
                   help="Optional auxiliary current-frame reconstruction weight from z_t, to force z_t to locate the current time point.")
    p.add_argument("--subspace_use_layernorm", action=argparse.BooleanOptionalAction, default=True,
                   help="Use LayerNorm in predictive_subspace_attention.")

    # CyclicGraphAR: fixed-step bidirectional ROI graph computation.
    p.add_argument("--cyclic_graph_steps", type=int, default=5,
                   help="Number of internal shared message-passing iterations for cyclic_graph_ar.")
    p.add_argument("--cyclic_graph_topk", type=int, default=8,
                   help="Number of ring/skip neighbors on each side used by cyclic_graph_ar.")
    p.add_argument("--cyclic_graph_alpha", type=float, default=0.10,
                   help="Residual update step size inside each cyclic graph iteration.")
    p.add_argument("--cyclic_graph_carry", type=float, default=0.25,
                   help="How strongly the previous recurrent graph state is carried into the current step.")
    p.add_argument("--cyclic_graph_edge_type", type=str, default="ring", choices=["ring", "skip", "dense"],
                   help="Fixed ROI graph used by cyclic_graph_ar. Use dense only for small state_dim.")
    p.add_argument("--cyclic_graph_learned_edge_gate", action=argparse.BooleanOptionalAction, default=True,
                   help="Learn a positive scalar gate on each fixed edge in cyclic_graph_ar.")
    p.add_argument("--cyclic_graph_message_hidden_mult", type=int, default=2,
                   help="Hidden multiplier inside cyclic_graph_ar edge-message MLP.")

    # Stage-A reusable potential-state autoencoder.
    p.add_argument("--potential_latent_dim", type=int, default=64,
                   help="Latent dimension for potential_ae_vector; this bottleneck is phi_t.")
    p.add_argument("--potential_hidden_dim", type=int, default=256,
                   help="MLP hidden width for potential_ae_vector.")
    p.add_argument("--potential_depth", type=int, default=2,
                   help="Encoder/decoder MLP depth for potential_ae_vector.")
    p.add_argument("--potential_dropout", type=float, default=0.0,
                   help="Dropout for potential_ae_vector encoder/decoder.")
    p.add_argument("--potential_value_hidden_dim", type=int, default=None,
                   help="Hidden width for scalar potential head V(phi). Defaults to --potential_hidden_dim.")
    p.add_argument("--potential_value_depth", type=int, default=None,
                   help="Depth for scalar potential head V(phi). Defaults to --potential_depth.")
    p.add_argument("--potential_stim_hidden_dim", type=int, default=None,
                   help="Hidden width for stimulus increment network G(u). Defaults to --potential_hidden_dim.")
    p.add_argument("--potential_stim_depth", type=int, default=None,
                   help="Depth for stimulus increment network G(u). Defaults to --potential_depth.")
    p.add_argument("--potential_stim_context_len", type=int, default=1,
                   help="Causal stimulus history length L for G(u_{t-L+1:t}) in stimulus-driven potential Stage A/B.")
    p.add_argument("--potential_stimulus_coboundary", action=argparse.BooleanOptionalAction, default=False,
                   help="Stage-A endpoint-pair stimulus coboundary: V(x_j)-V(x_i) is constrained to equal external work W(u_{i:j}).")
    p.add_argument("--potential_rollout_horizon", type=int, default=8,
                   help="Maximum endpoint gap K used by Stage-A pairwise potential sampling. Kept for backward-compatible scripts.")
    p.add_argument("--potential_pair_samples", type=int, default=0,
                   help="Number of endpoint pairs (i,j) sampled per Stage-A batch. <=0 means use all valid pairs in the window.")
    p.add_argument("--potential_pair_min_gap", type=int, default=1,
                   help="Minimum endpoint gap j-i for Stage-A pairwise potential loss.")
    p.add_argument("--potential_pair_max_gap", type=int, default=0,
                   help="Maximum endpoint gap j-i for Stage-A pairwise potential loss. <=0 falls back to --potential_rollout_horizon.")
    p.add_argument("--potential_lambda_roll", type=float, default=0.1,
                   help="Backward-compatible alias for --potential_lambda_cons if the latter is not set.")
    p.add_argument("--potential_lambda_cons", type=float, default=0.1,
                   help="Weight for potential-component consistency D(V(x_i)+W(u_{i:j})) ~= D(V(x_j)); not a full x_j rollout loss.")
    p.add_argument("--potential_cons_loss_type", type=str, default="rel_mse", choices=["mse", "rel_mse", "corr", "rel_l2"],
                   help="Loss for potential-component consistency in Stage A.")
    p.add_argument("--potential_detach_cons_target", action=argparse.BooleanOptionalAction, default=True,
                   help="Detach D(V(x_j)) target in Stage-A component consistency, so gradients mainly move the interval prediction path.")
    p.add_argument("--potential_roll_loss_type", type=str, default="rel_mse", choices=["mse", "rel_mse", "corr", "rel_l2"],
                   help="Diagnostic full-x loss type kept for old logs; endpoint-pair Stage A does not optimize full x rollout.")
    p.add_argument("--potential_rec_loss", type=str, default="rel_l2", choices=["rel_l2", "mse"],
                   help="Reconstruction objective for Stage-A potential AE.")
    p.add_argument("--potential_lambda_rec", type=float, default=1.0,
                   help="Weight for potential AE reconstruction loss.")
    p.add_argument("--potential_lambda_l2", type=float, default=0.0,
                   help="Optional L2 penalty on phi_t.")
    p.add_argument("--potential_lambda_var", type=float, default=0.0,
                   help="Optional negative variance coefficient to discourage latent collapse.")
    p.add_argument("--potential_coboundary_loss", action=argparse.BooleanOptionalAction, default=False,
                   help="Enforce vector coboundary: decoded latent difference D_delta(V(x_{t+1})-V(x_t)) matches a trajectory increment.")
    p.add_argument("--potential_lambda_cob", type=float, default=1.0,
                   help="Weight for the coboundary constraint in Stage-A potential AE.")
    p.add_argument("--potential_cob_target", type=str, default="state_delta",
                   choices=["state_delta", "normalized_state_delta", "state_energy_delta_scalar"],
                   help="Vector coboundary target. Default state_delta enforces D_delta(V(x_{t+1})-V(x_t)) ~= x_{t+1}-x_t.")
    p.add_argument("--potential_cob_loss_type", type=str, default="rel_mse", choices=["mse", "rel_mse", "corr"],
                   help="Loss for coboundary scalar fitting.")
    p.add_argument("--potential_cob_standardize", action=argparse.BooleanOptionalAction, default=True,
                   help="Standardize coboundary target and predicted delta inside each batch/window for stable fitting.")

    # Stage-B residual AR: keep Stage-A potential AE frozen and train an AR
    # model on x - D_A(E_A(x)).  This does not modify Stage A.
    p.add_argument("--stageb_potential_residual", action=argparse.BooleanOptionalAction, default=False,
                   help="Train/evaluate the AR backbone on residual signal after subtracting a frozen Stage-A potential AE reconstruction.")
    p.add_argument("--stageb_potential_ckpt_path", type=str, default="",
                   help="Checkpoint path for the frozen Stage-A potential_ae_vector model.")
    p.add_argument("--stageb_potential_non_strict_ckpt", action=argparse.BooleanOptionalAction, default=False,
                   help="Load Stage-A potential AE checkpoint with strict=False.")
    p.add_argument("--stageb_stim_potential_residual_ar", action=argparse.BooleanOptionalAction, default=False,
                   help="Freeze stimulus-driven Stage A; feed full x history to AR; predict only residual r_{t+1}=x_{t+1}-D(v_t+G(u_t)).")

    p.add_argument("--stageb_joint_potential_ar", action=argparse.BooleanOptionalAction, default=False,
                   help="Stage B joint AR over [v_t, r_t] with frozen Stage-A: v_{t+1}=v_t+G, r_{t+1}=F.")
    p.add_argument("--stageb_lambda_x", type=float, default=0.0,
                   help="Stage-B decoded full-state loss weight. Default 0: decoded x is diagnostic only, so residual cannot bypass potential by relearning the full signal.")
    p.add_argument("--stageb_predict_delta_v", action=argparse.BooleanOptionalAction, default=True,
                   help="Interpret the first latent_dim Stage-B outputs as delta_v, so v_{t+1}=v_t+delta_v_hat. This is the intended coboundary AR parameterization.")
    p.add_argument("--stageb_lambda_v", type=float, default=1.0,
                   help="Stage-B potential-state loss weight.")
    p.add_argument("--stageb_lambda_r", type=float, default=1.0,
                   help="Stage-B residual-state loss weight.")

    # Recurrent Mamba-style predictive-state baseline.  This is trained with
    # GT burn-in + K-step closed-loop chunked BPTT.  It keeps the recurrent
    # state value across a burn-in window but truncates the gradient graph to
    # the rollout chunk.
    p.add_argument("--mamba_alpha_min", type=float, default=0.0,
                   help="Minimum recurrent retention alpha for mamba_state_vector.")
    p.add_argument("--mamba_alpha_max", type=float, default=0.995,
                   help="Maximum recurrent retention alpha for mamba_state_vector; keep <1 for stable memory.")
    p.add_argument("--mamba_d_state", type=int, default=16,
                   help="SSM state size N for the dependency-free Mamba block.")
    p.add_argument("--mamba_d_conv", type=int, default=4,
                   help="Causal depthwise convolution kernel size for the dependency-free Mamba block.")
    p.add_argument("--mamba_expand", type=int, default=2,
                   help="Inner-channel expansion factor for the dependency-free Mamba block.")
    p.add_argument("--mamba_memory_len", type=int, default=-1,
                   help="Rolling token memory length for recurrent Mamba evaluation/training. If <=0, registry uses --window_size.")
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
    p.add_argument("--mamba_var_match_lambda", type=float, default=0.0,
                   help="Weight of the across-start anomaly-Wasserstein distributional term added to the recurrent rollout loss (0=off, default). Faithful training-time version of the w1_anom diagnostic: per-coordinate, mean-removed, distribution-free 1-Wasserstein between the predicted and target across-start marginals at each horizon. Penalizes the mean-collapse that MSE ignores; used for the keystone loss-vs-g experiment.")
    p.add_argument("--mamba_crps", type=float, default=0.0,
                   help="Proper-scoring objective: train the recurrent rollout with closed-form Gaussian CRPS on (mu=prediction, sigma=model spread head) instead of MSE+W1 (0=off, default; typically 1.0). Requires the sigma head (auto-built when this is nonzero). Unlike the anomaly-W1 surrogate, CRPS is a proper scoring rule -- sigma is trained against the same target and cannot be gamed by decorrelated variance; it is the loss family used in probabilistic weather forecasting.")
    p.add_argument("--mamba_loss_weight_mode", type=str, default="uniform", choices=["uniform", "decay"],
                   help="Compatibility option for old scripts. Current recurrent loss uses --mamba_loss_decay; uniform keeps all steps equally weighted when decay=1.")
    p.add_argument("--mamba_loss_decay", type=float, default=1.0,
                   help="Optional geometric per-step loss weight. 1.0 means uniform over the K-step chunk.")

    # Residual-gradient routing for long-horizon recurrent BPTT.  This keeps
    # the residual identity gradient path open while optionally detaching the
    # nonlinear residual branch gradient at selected rollout steps.  Forward
    # values are unchanged; only the delayed-credit graph is changed.
    p.add_argument("--resgrad_routing", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable backward-only routing at internal residual merges (supported by official_mamba_state and unet_field).")
    p.add_argument("--resgrad_policy", type=str, default="all",
                   choices=["all", "none", "identity", "fixed", "periodic", "tail", "head", "ratio", "act_ratio", "dynamic_ratio", "branch_ratio", "norm_ratio", "snr", "snrk", "coherence", "cohall", "dualcoh", "dualwiener", "oracle"],
                   help="Policy for delayed-credit routing during BPTT. all=ordinary BPTT; none keeps only the skip route; ratio/dynamic_ratio is magnitude-based; coherence/dualcoh are legacy cross-example heuristics, not SNR estimators; dualwiener estimates internal soft (alpha,m) from explicit total/noise VJP probes and needs no sigma head.")
    p.add_argument("--resgrad_block_gate", type=float, default=0.0,
                   help="Base nonlinear-branch gradient gate used by fixed/periodic/tail/head/ratio when a branch is not explicitly kept. 0 detaches branch gradient; 1 is ordinary BPTT.")
    p.add_argument("--resgrad_ratio_threshold", type=float, default=0.05,
                   help="For ratio/dynamic_ratio routing, keep nonlinear branch gradient when mean ||branch||/||residual|| at the residual-add point is at least this value.")
    p.add_argument("--resgrad_keep_every", type=int, default=8,
                   help="For --resgrad_policy periodic, keep ordinary branch gradient every N rollout steps.")
    p.add_argument("--resgrad_keep_tail", type=int, default=0,
                   help="For --resgrad_policy tail/head, number of rollout steps whose branch gradients are kept.")
    p.add_argument("--resgrad_outer", action=argparse.BooleanOptionalAction, default=False,
                   help="Temporal-residual routing (official_mamba_state): keep all INTERNAL blocks fully open and instead gate the backward of the OUTER per-step residual delta in pred=x_t+delta. This realizes the mechanism's per-step Jacobian I+m*J_F one-to-one (no branch<->step assumption). The keep/cut policy reuses --resgrad_policy (head/tail/periodic/fixed) or dynamic_ratio (outer ||delta||/||x_t||).")
    p.add_argument("--resgrad_target_open_frac", type=float, default=0.0,
                   help="If >0, auto-calibrate --resgrad_ratio_threshold every --resgrad_calib_every_epochs epochs via a cheap no-grad rollout so dynamic_ratio routing opens approximately this fraction of nonlinear branches, instead of using a fixed hand-picked threshold. 0 (default) disables calibration and uses --resgrad_ratio_threshold as-is.")
    p.add_argument("--resgrad_calib_num_starts", type=int, default=4,
                   help="Number of rollout starts sampled from the current training batch when calibrating --resgrad_target_open_frac.")
    p.add_argument("--resgrad_calib_every_batches", type=int, default=1,
                   help="Recalibrate --resgrad_ratio_threshold every N training batches when --resgrad_target_open_frac > 0, using that batch's own data. 1 = every batch (safest; the ratio distribution can drift quickly early in training, so once-per-epoch calibration can go stale and open far more branches than intended).")
    p.add_argument("--resgrad_calib_ema", type=float, default=0.0,
                   help="EMA coefficient for smoothing the auto-calibrated ratio threshold across recalibrations (applied = ema*prev + (1-ema)*measured). 0.0 (default) = off, original per-batch behavior. At intermediate open fractions (e.g. 0.5) the calibration quantile sits at the median of the ratio distribution, where per-batch threshold noise flips the most gates batch-to-batch (selection churn); ~0.9 damps this while still tracking slow drift of the ratio distribution.")
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
    p.add_argument(
        "--global_horizon_wiener",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Jointly weight the K complete horizon-loss parameter gradients with "
            "a box-constrained Wiener solve. The forward rollout and scalar loss "
            "values are unchanged. Domain innovation sampling reuses the "
            "--dual_wiener_* covariance configuration."
        ),
    )
    p.add_argument(
        "--global_wiener_ridge",
        type=float,
        default=1e-8,
        help="Numerical ridge relative to mean diag(T) in the global horizon solve.",
    )
    p.add_argument(
        "--global_wiener_anchor",
        type=float,
        default=0.0,
        help=(
            "Statistical regularization toward the fully-open horizon weights, "
            "relative to mean diag(T). Zero recovers the unregularized Wiener "
            "solve; positive values add 0.5*lambda*||w-1||^2."
        ),
    )
    p.add_argument(
        "--global_wiener_local_fidelity",
        type=float,
        default=0.0,
        help=(
            "Shrink the joint global-horizon Wiener objective toward separate "
            "per-horizon scalar Wiener risks. 0 keeps the original aggregate "
            "gradient objective; 1 discards cross-horizon cancellation and "
            "solves K independent scalar gains. Values must lie in [0,1]."
        ),
    )
    p.add_argument(
        "--global_wiener_solver_iters",
        type=int,
        default=256,
        help="Projected-FISTA iterations for the K-dimensional box Wiener solve.",
    )
    p.add_argument(
        "--global_wiener_sketch_dim",
        type=int,
        default=8192,
        help=(
            "CountSketch dimension for complete parameter gradients. Positive "
            "values bound calibration memory at O(K*dim); 0 stores exact "
            "gradients and is intended only for small diagnostic models."
        ),
    )
    p.add_argument(
        "--global_wiener_sketch_seed",
        type=int,
        default=1729,
        help="Base seed for the per-probe complete-gradient CountSketch.",
    )
    p.add_argument(
        "--global_wiener_noise_draws",
        type=int,
        default=4,
        help=(
            "Independent coherent innovation trajectories per global-horizon "
            "calibration batch. Their gradient Grams are averaged; the noise "
            "objectives themselves are never averaged before forming a Gram."
        ),
    )
    p.add_argument(
        "--global_wiener_static_gain",
        type=float,
        default=-1.0,
        help=(
            "Disable estimation and use a fixed global-horizon coefficient. "
            "Negative values disable static mode; values in [0,1] enable it."
        ),
    )
    p.add_argument(
        "--global_wiener_static_mode",
        choices=("delayed_tied", "exponential"),
        default="delayed_tied",
        help=(
            "Shape of fixed global-horizon weights. delayed_tied keeps horizon "
            "one open and ties every delayed horizon to c; exponential uses c^k."
        ),
    )
    p.add_argument(
        "--global_wiener_batch_conditioned",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use one rollout start shared by every sample in the optimizer "
            "micro-batch, reduce DDP gradient features before forming T/R, and "
            "apply the resulting horizon weights to that same backward pass."
        ),
    )
    p.add_argument(
        "--global_wiener_superbatch_groups",
        type=int,
        default=1,
        help=(
            "Calibration-only number M of equal disjoint sample subgroups "
            "inside one shared-t0 optimizer batch. T averages M subgroup-mean "
            "gradient Grams and R averages the identical subgroup x innovation-"
            "draw geometry. M=1 preserves the complete-batch controller."
        ),
    )
    # --- ResGrad open-fraction SCHEDULE (cheap routing early, near-dense + checkpointing late) ---
    p.add_argument("--resgrad_sched_open_start", type=float, default=-1.0,
                   help="If >=0, enable the epoch-dependent ResGrad open-fraction schedule and use this as the EARLY-phase target open fraction (e.g. 0.1 or 0.25). <0 (default) disables the schedule and uses the static --resgrad_target_open_frac. Motivated by short-path redundancy: train cheap (low g) for most epochs, then ramp to near-dense to polish.")
    p.add_argument("--resgrad_sched_open_end", type=float, default=1.0,
                   help="Late-phase target open fraction for the schedule (1.0 = fully dense gradient). Only used when --resgrad_sched_open_start >= 0.")
    p.add_argument("--resgrad_sched_ramp_start_epoch", type=int, default=0,
                   help="Epoch at which the open fraction begins ramping from start to end; held at the start value before this. 0 -> epoch 1.")
    p.add_argument("--resgrad_sched_ramp_end_epoch", type=int, default=0,
                   help="Epoch at which the open fraction reaches the end value; held at end after this. 0 -> num_epochs.")
    p.add_argument("--resgrad_sched_ckpt_from_epoch", type=int, default=-1,
                   help="From this epoch on, force gradient checkpointing on (overriding --recurrent_grad_checkpoint) so the high-g / near-dense tail of the schedule keeps peak memory at the checkpointing floor instead of paying dense stored-activation memory. -1 (default) = never auto-toggle.")
    p.add_argument("--recurrent_grad_checkpoint", action=argparse.BooleanOptionalAction, default=False,
                   help="Wrap each closed-loop rollout step in official_mamba_state training in torch.utils.checkpoint, recomputing forward activations during backward instead of storing them. This is a gradient-exact alternative to --resgrad_routing for reducing long-BPTT memory (roughly O(1)-in-K memory per step, ~2x forward compute), used as a baseline to compare against ResGrad's approximate-gradient routing.")
    p.add_argument("--bptt_grad_checkpoint", action=argparse.BooleanOptionalAction, default=False,
                   help="Same as --recurrent_grad_checkpoint but for the generic windowed-history standard-AR BPTT path (compute_full_bptt_rollout_loss), used by unet_field/resnet_vector/transformer_vector/etc. Wraps each closed-loop rollout step in torch.utils.checkpoint.")
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

    # Bridge-Control Distillation for recurrent-state AR models.  This is a
    # training-only privileged-target objective: future anchors construct a
    # globally informed local control target, while the student input remains
    # causal.  With --bridge_control_replace_base it avoids K-step BPTT and
    # trains one recurrent step per sampled time.
    p.add_argument("--bridge_control_loss", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable bridge-control distillation for recurrent Mamba-style AR models.")
    p.add_argument("--bridge_control_replace_base", action=argparse.BooleanOptionalAction, default=True,
                   help="If true, replace recurrent K-step BPTT with local bridge-control distillation. If false, add it as an auxiliary loss.")
    p.add_argument("--bridge_control_lambda", type=float, default=1.0,
                   help="Weight for bridge-control target loss.")
    p.add_argument("--bridge_control_next_lambda", type=float, default=0.0,
                   help="Optional ordinary next-frame anchor used with bridge-control distillation.")
    p.add_argument("--bridge_control_horizons", type=str, default="4,8",
                   help="Comma-separated future anchor horizons m used to build v*_t=sum_m alpha_m (x_{t+m}-x_t)/m.")
    p.add_argument("--bridge_control_target", type=str, default="multiscale_avg", choices=["multiscale_avg", "longest", "endpoint", "max"],
                   help="How to combine future anchors into the bridge-control target.")
    p.add_argument("--bridge_control_horizon_weight", type=str, default="uniform", choices=["uniform", "inverse", "inv", "linear", "horizon"],
                   help="Horizon weighting for multi-scale bridge-control targets.")
    p.add_argument("--bridge_control_loss_type", type=str, default="rel_l2", choices=["mse", "rel_mse", "relative_mse", "rel_l2", "cos", "cos_mse", "cos_rel"],
                   help="Loss between predicted local delta and bridge-induced control target.")
    p.add_argument("--bridge_control_norm_weight", type=float, default=0.25,
                   help="Norm/radius term weight for bridge_control_loss_type=cos_mse/cos_rel.")
    p.add_argument("--bridge_control_next_loss_type", type=str, default="rel_l2", choices=["mse", "huber", "l1", "rel_l2"],
                   help="Loss type for the optional ordinary next-frame anchor.")
    p.add_argument("--bridge_control_target_scale", type=float, default=1.0,
                   help="Global multiplier on bridge-control targets; useful if future average velocity is too small/large.")
    p.add_argument("--bridge_control_detach_target", action=argparse.BooleanOptionalAction, default=True,
                   help="Detach bridge-control targets. Keep true unless doing unusual differentiable target construction.")
    p.add_argument("--bridge_control_starts_per_sequence", type=int, default=-1,
                   help="Number of sampled local bridge-control starts per sequence. If <=0, reuse Mamba/AR start count.")
    p.add_argument("--bridge_control_train_stride", type=int, default=-1,
                   help="Candidate start stride. If <=0, reuse --mamba_train_stride.")
    p.add_argument("--bridge_control_random_starts", action=argparse.BooleanOptionalAction, default=True,
                   help="Randomly sample bridge-control starts during training.")
    p.add_argument("--bridge_control_start_epoch", type=int, default=1,
                   help="First epoch where bridge-control lambda is nonzero.")
    p.add_argument("--bridge_control_ramp_epochs", type=int, default=0,
                   help="Linearly ramp bridge-control lambda over this many epochs after start.")
    p.add_argument("--bridge_control_eps", type=float, default=1e-8,
                   help="Numerical epsilon for bridge-control losses.")

    # Predictive-coding Mamba readout: prior + precision-gated stimulus evidence correction.
    p.add_argument("--pc_gate_init", type=float, default=0.35,
                   help="Initial correction gate for official_pc_mamba_state; 0 means pure prior, 1 means pure evidence.")
    p.add_argument("--pc_gate_scalar", action=argparse.BooleanOptionalAction, default=False,
                   help="Use one scalar correction gate per sample instead of one gate per state dimension.")
    p.add_argument("--pc_gate_min", type=float, default=0.0,
                   help="Minimum correction gate after sigmoid rescaling for official_pc_mamba_state.")
    p.add_argument("--pc_gate_max", type=float, default=1.0,
                   help="Maximum correction gate after sigmoid rescaling for official_pc_mamba_state.")
    p.add_argument("--pc_delta_weight", type=float, default=0.0,
                   help="Auxiliary normalized update/tangent loss weight: (pred-prev_pred) should match (gt_t-gt_{t-1}).")
    p.add_argument("--pc_correction_weight", type=float, default=0.0,
                   help="Auxiliary correction-direction loss weight: correction from prior should point toward target-prior.")
    p.add_argument("--pc_evidence_weight", type=float, default=0.0,
                   help="Auxiliary evidence-head loss weight, forcing the stimulus-conditioned evidence prediction toward the target.")
    p.add_argument("--pc_prior_weight", type=float, default=0.0,
                   help="Auxiliary prior-head loss weight. Usually keep small or zero to avoid overemphasizing identity prior.")
    p.add_argument("--pc_prior_smooth_weight", type=float, default=0.0,
                   help="Auxiliary prior smoothness/persistence loss weight: prior branch is kept close to the previous input state.")
    p.add_argument("--pc_correction_mag_weight", type=float, default=0.0,
                   help="Auxiliary correction magnitude loss weight: ||applied correction|| should match ||target-prior||.")
    p.add_argument("--pc_correction_mag_target", type=float, default=1.0,
                   help="Target ratio for ||applied correction|| / ||target-prior|| when --pc_correction_mag_weight is active.")
    p.add_argument("--pc_gate_target_weight", type=float, default=0.0,
                   help="Auxiliary gate target penalty weight; use with --pc_gate_target to discourage gate saturation.")
    p.add_argument("--pc_gate_target", type=float, default=0.5,
                   help="Target gate value used only by --pc_gate_target_weight.")
    p.add_argument("--pc_gate_floor_weight", type=float, default=0.0,
                   help="Penalty weight for gates below --pc_gate_floor; helps prevent collapse to pure prior.")
    p.add_argument("--pc_gate_floor", type=float, default=0.05,
                   help="Gate floor used only by --pc_gate_floor_weight.")

    # Local-atlas Mamba: learns local chart encoders/inverse charts, chart
    # overlap transitions, and stimulus-conditioned dynamical transition maps.
    p.add_argument("--atlas_num_charts", type=int, default=4,
                   help="Number of learned local charts for atlas/tangent-atlas Mamba models.")
    p.add_argument("--atlas_latent_dim", type=int, default=64,
                   help="Latent coordinate dimension inside each local chart.")
    p.add_argument("--atlas_chart_emb_dim", type=int, default=64,
                   help="Learned chart-identity embedding dimension.")
    p.add_argument("--atlas_hidden_dim", type=int, default=512,
                   help="Hidden width of chart encoder/decoder/transition MLPs.")
    p.add_argument("--atlas_temperature", type=float, default=1.0,
                   help="Softmax temperature for current/next chart selectors.")
    p.add_argument("--atlas_perturb_std", type=float, default=0.02,
                   help="Coordinate perturbation std used by local chart invertibility loss.")
    p.add_argument("--atlas_perturb_min_ratio", type=float, default=0.20,
                   help="Minimum decoded perturbation ratio enforced by atlas noncollapse hinge.")
    p.add_argument("--atlas_rec_weight", type=float, default=0.0,
                   help="Weight for chart reconstruction loss psi_i(phi_i(x))≈x.")
    p.add_argument("--atlas_chart_inv_weight", type=float, default=0.0,
                   help="Weight for local chart invertibility phi_i(psi_i(z+xi))≈z+xi.")
    p.add_argument("--atlas_dyn_weight", type=float, default=0.0,
                   help="Weight for local dynamic transition loss T_ij(phi_i(x_t),u)≈phi_j(x_{t+1}).")
    p.add_argument("--atlas_pred_chart_weight", type=float, default=0.0,
                   help="Weight for per-target-chart decoded prediction loss.")
    p.add_argument("--atlas_overlap_weight", type=float, default=0.0,
                   help="Weight for overlap transition loss tau_ij(phi_i(x))≈phi_j(x).")
    p.add_argument("--atlas_cocycle_weight", type=float, default=0.0,
                   help="Weight for atlas cocycle loss tau_ik≈tau_jk∘tau_ij.")
    p.add_argument("--atlas_compose_weight", type=float, default=0.0,
                   help="Weight for K-step composed latent transition endpoint loss in chart space.")
    p.add_argument("--atlas_balance_weight", type=float, default=0.0,
                   help="Weight for batch-level chart balance loss to prevent early chart collapse.")
    p.add_argument("--atlas_noncollapse_weight", type=float, default=0.0,
                   help="Penalty for decoded chart perturbation ratio below --atlas_perturb_min_ratio.")
    p.add_argument("--atlas_std_weight", type=float, default=0.0,
                   help="Observation-space amplitude floor weight for atlas predictions.")
    p.add_argument("--atlas_std_min_ratio", type=float, default=0.20,
                   help="Minimum pred_std/target_std ratio enforced by --atlas_std_weight.")
    p.add_argument("--atlas_entropy_floor_weight", type=float, default=0.0,
                   help="Entropy-floor penalty weight for current and next chart selectors.")
    p.add_argument("--atlas_entropy_min", type=float, default=1.0,
                   help="Minimum chart-selector entropy used by --atlas_entropy_floor_weight.")
    p.add_argument("--atlas_delta_cos_weight", type=float, default=0.0,
                   help="Weight for atlas decoded transition tangent cosine loss.")
    p.add_argument("--atlas_delta_mse_weight", type=float, default=0.0,
                   help="Weight for normalized atlas decoded transition tangent MSE loss.")
    p.add_argument("--atlas_delta_norm_weight", type=float, default=0.0,
                   help="Anti-identity hinge weight forcing nontrivial predicted tangent norm.")
    p.add_argument("--atlas_delta_norm_min_ratio", type=float, default=0.25,
                   help="Minimum ||delta_hat||/||delta_gt|| ratio enforced by --atlas_delta_norm_weight.")
    p.add_argument("--atlas_hard_delta_norm", action=argparse.BooleanOptionalAction, default=False,
                   help="Tangent-atlas only: structurally normalize delta direction and give it a nonzero scale to prevent identity.")
    p.add_argument("--atlas_hard_delta_min_x_ratio", type=float, default=0.05,
                   help="Tangent-atlas hard-norm floor: minimum ||delta_hat|| as a fraction of ||x_t||.")
    p.add_argument("--atlas_delta_scale_init_ratio", type=float, default=0.01,
                   help="Initial extra positive scale ratio for tangent hard-norm mode, added above the hard floor via softplus head.")
    p.add_argument("--atlas_delta_scale_loss_weight", type=float, default=0.0,
                   help="Optional log-scale matching loss for tangent delta norm: (log||delta_hat|| - log||delta_gt||)^2.")

    # Shadow-perturbed Mamba: A_t = A0 + low-rank DeltaA_t.
    p.add_argument("--shadow_channels", type=int, default=64,
                   help="Number of independent channel blocks for stable base transport A0.")
    p.add_argument("--shadow_perturb_rank", type=int, default=4,
                   help="Low rank R for DeltaA_t = U diag(beta_t) V^T inside each channel.")
    p.add_argument("--shadow_perturb_eps", type=float, default=0.02,
                   help="Scale for perturbation coefficients beta_t. With tanh mode, it is a hard bound; with linear mode, it is only a scale.")
    p.add_argument("--shadow_perturb_bound", type=str, default="tanh", choices=["tanh", "linear", "none"],
                   help="How to map raw beta to perturbation coefficient: tanh=eps*tanh(raw), linear=eps*raw, none=raw.")
    p.add_argument("--shadow_a_init_scale", type=float, default=0.98,
                   help="Initial scale for base A0; should be <1 for stable transport.")
    p.add_argument("--shadow_a_init_noise", type=float, default=1e-3,
                   help="Small random noise added to initial A0.")
    p.add_argument("--shadow_condition_scale", type=float, default=1.0,
                   help="Scale of shadow state injected into Mamba token embedding.")
    p.add_argument("--shadow_output_scale", type=float, default=1.0,
                   help="Scale of shadow state added to final Mamba readout state.")
    p.add_argument("--shadow_kg_horizon", type=int, default=0,
                   help="Folded shadow-supervision horizon on base A0. 0 disables it.")
    p.add_argument("--shadow_kg_weight", type=float, default=0.0,
                   help="Weight for supervised folded loss on the base shadow trace A0^k z.")
    p.add_argument("--shadow_delta_weight", type=float, default=0.0,
                   help="Weight for penalizing realized DeltaA_t z relative to A0 z.")
    p.add_argument("--shadow_spec_weight", type=float, default=0.0,
                   help="Weight for spectral penalty max(0, sigma_max(A0)-shadow_spec_max)^2.")
    p.add_argument("--shadow_spec_max", type=float, default=0.999,
                   help="Maximum allowed per-channel sigma_max(A0) before spectral penalty.")

    # Faithful Mamba fixed-A perturbation: replace only the internal SSM
    # transition dA_t * h by (A + DeltaA_t) * h. The rest of Mamba is kept.
    p.add_argument("--mamba_fixeda_init", type=float, default=0.98,
                   help="Initial discrete diagonal fixed-A retention for official_mamba_fixeda_perturb.")
    p.add_argument("--mamba_fixeda_max", type=float, default=0.999,
                   help="Maximum discrete diagonal fixed-A retention; enforced by sigmoid parameterization.")
    p.add_argument("--mamba_fixeda_perturb_eps", type=float, default=0.02,
                   help="Scale/bound for diagonal DeltaA_t inside the Mamba SSM state update.")
    p.add_argument("--mamba_fixeda_perturb_bound", type=str, default="tanh", choices=["tanh", "sigmoid", "linear"],
                   help="Map raw perturbation to DeltaA_t: tanh=eps*tanh(raw), sigmoid=eps*(2sigmoid(raw)-1), linear=eps*raw.")
    p.add_argument("--mamba_fixeda_perturb_hidden_mult", type=int, default=1,
                   help="Hidden width multiplier for the DeltaA_t generator MLP.")
    p.add_argument("--mamba_fixeda_kg_weight", type=float, default=0.0,
                   help="Weight for finite-horizon Gramian loss on the fixed diagonal A.")
    p.add_argument("--mamba_fixeda_kg_horizon", type=int, default=0,
                   help="Horizon K for finite-horizon Gramian loss on fixed A. 0 disables KG.")
    p.add_argument("--mamba_fixeda_corr_weight", type=float, default=0.0,
                   help="Weight for exact discrete corrector matching loss on r_t=DeltaA_t h_t.")
    p.add_argument("--mamba_fixeda_mean_weight", type=float, default=0.0,
                   help="Weight for mean-drift penalty on r_t=DeltaA_t h_t.")
    p.add_argument("--mamba_fixeda_delta_weight", type=float, default=0.0,
                   help="Optional weight for perturbation magnitude ||r||^2/||A h||^2.")
    p.add_argument("--mamba_fixeda_corr_solve_lambda", type=float, default=1e-2,
                   help="Ridge term in the minimum-energy exact corrector solve.")
    p.add_argument("--mamba_fixeda_corr_detach", action=argparse.BooleanOptionalAction, default=True,
                   help="If true, detach solved psi* and use it as a projection target. If false, backprop through the tridiagonal solve.")

    # CTO-cut: dataset-cut temporal order contrast for recurrent-state AR models.
    # This is NOT predicted-block embedding contrast. It trains two independent
    # local K-step blocks and contrasts model error on the true boundary order
    # tail(A)->head(B) against a swapped boundary head(B)->tail(A).
    p.add_argument("--cto_cut_loss", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable dataset-cut CTO objective for mamba_state_vector.")
    p.add_argument("--cto_cut_lambda", type=float, default=0.0,
                   help="Weight of the CTO-cut order error contrast term.")
    p.add_argument("--cto_cut_block_size", type=int, default=0,
                   help="Local block size K for CTO-cut. If <=0, use --mamba_bptt_horizon.")
    p.add_argument("--cto_cut_order_span", type=int, default=0,
                   help="Number of frames taken from tail(A) and head(B) for the order test. If <=0, use K//2.")
    p.add_argument("--cto_cut_order_horizon", type=int, default=0,
                   help="How many future frames from the second crop to score in the order test. If <=0, use order_span.")
    p.add_argument("--cto_cut_margin", type=float, default=0.1,
                   help="Margin m in softplus(m + E_pos - E_neg).")
    p.add_argument("--cto_cut_starts_per_sequence", type=int, default=-1,
                   help="Number of CTO-cut startpoints per sequence. If <=0, use --mamba_train_starts_per_sequence.")
    p.add_argument("--cto_cut_stride", type=int, default=1,
                   help="Candidate start stride for CTO-cut training.")
    p.add_argument("--cto_cut_replace_base", action=argparse.BooleanOptionalAction, default=True,
                   help="If enabled, train with CTO-cut objective instead of extra base BPTT16. This is the clean version.")
    p.add_argument("--cto_cut_eval", action=argparse.BooleanOptionalAction, default=False,
                   help="Also compute CTO-cut loss during validation. Disabled by default because validation should usually report BPTT loss.")
    p.add_argument("--cto_cut_sequential_backward", action=argparse.BooleanOptionalAction, default=True,
                   help="During training with --cto_cut_replace_base, backward A, B, and order losses sequentially and step once. This avoids retaining all CTO-cut graphs at once.")

    # Error-Transport Metric (ETM) loss.  This implements the current idea:
    # one-step loss controls local bias, while a learned full transport
    # matrix identifies which one-step error directions become dangerous at
    # long horizon.  Version A conditions the metric on current history/context;
    # Version B additionally conditions it on future external inputs.
    p.add_argument("--etm_loss", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable Error-Transport Metric loss for vector Transformer AR models.")
    p.add_argument("--etm_version", type=str, default="A", choices=["A", "B", "a", "b"],
                   help="A: state/current-context transport; B: input-conditioned transport using future stimulus context.")
    p.add_argument("--etm_horizon", type=int, default=0,
                   help="Error-transport horizon. If 0, fall back to --koopman_gramian_horizon.")
    p.add_argument("--etm_lambda_fit", type=float, default=0.0,
                   help="Weight for fitting J e_{t+1} to the observed rollout error e_{t+K}.")
    p.add_argument("--etm_lambda_prop", type=float, default=0.0,
                   help="Weight for suppressing the transport-weighted one-step error ||J e_{t+1}||^2.")
    p.add_argument("--etm_start_epoch", type=int, default=0,
                   help="Do not apply ETM losses before this epoch; use as one-step warmup.")
    p.add_argument("--etm_ramp_epochs", type=int, default=0,
                   help="Linearly ramp joint-stage ETM lambdas over this many epochs after --etm_start_epoch + --etm_j_only_epochs.")
    p.add_argument("--etm_j_only_epochs", type=int, default=0,
                   help="Number of epochs after --etm_start_epoch used as a J-only stage: freeze the AR backbone, train only ETM parameters with fit loss, and force lambda_prop=0.")
    p.add_argument("--etm_j_only_lambda_fit", type=float, default=-1.0,
                   help="Fit-loss weight during the J-only stage. If negative, use --etm_lambda_fit.")
    p.add_argument("--etm_freeze_backbone_during_j_only", action=argparse.BooleanOptionalAction, default=True,
                   help="Freeze all non-ETM parameters during the J-only stage. ETM parameters are identified by names containing 'etm_'.")
    p.add_argument("--etm_freeze_etm_before_start", action=argparse.BooleanOptionalAction, default=True,
                   help="Optionally freeze ETM parameters before --etm_start_epoch. Default false keeps old warmup behavior.")
    p.add_argument("--etm_rollout_sensitivity_eval_mode", action=argparse.BooleanOptionalAction, default=True,
                   help="Temporarily put the raw model in eval mode when constructing no-grad rollout-sensitivity targets, so dropout does not add target noise.")
    p.add_argument("--etm_detach_context", action=argparse.BooleanOptionalAction, default=True,
                   help="Detach shared Transformer context when training the transport head with the fit loss.")
    p.add_argument("--etm_detach_e1_fit", action=argparse.BooleanOptionalAction, default=True,
                   help="Detach e_{t+1} in the transport fitting loss so the fit term trains the transport head, not the predictor.")
    p.add_argument("--etm_normalize_metric", action=argparse.BooleanOptionalAction, default=True,
                   help="Normalize trace(J^T J) to state_dim per sample before using it as a suppression metric.")
    p.add_argument("--etm_diag_base", type=float, default=0.0,
                   help="Identity base for full transport J = base*I + scale*tanh(head). Use 0 for HCP long-horizon forgetting, 1 for identity initialization.")
    p.add_argument("--etm_diag_scale", type=float, default=0.5,
                   help="Scale for full transport J = base*I + scale*tanh(head), and for Version-A low-rank singular strengths.")
    p.add_argument("--etm_lowrank_rank", type=int, default=16,
                   help="Rank R for Version-A low-rank transport J = base*I + scale*U diag(s) V^T. Version B keeps the full matrix head.")
    p.add_argument("--etm_transport_param", type=str, default="full",
                   choices=["full", "lowrank", "residual_lowrank", "reslowrank", "lr", "attn_metric", "attention_metric", "traj_metric", "path_attn_metric", "path_metric", "rollout_path_metric"],
                   help="Parameterization for the ETM auxiliary head. full/residual_lowrank predict transport J; attn_metric uses a PSD metric over the first residual; path_attn_metric applies the same metric to residual sources along a detached rollout path.")
    p.add_argument("--etm_metric_atoms", type=int, default=64,
                   help="Number of shared direction atoms B used by --etm_transport_param attn_metric.")
    p.add_argument("--etm_metric_temperature", type=float, default=1.0,
                   help="Softmax temperature for attention weights in the trajectory-aware metric.")
    p.add_argument("--etm_metric_rank_margin", type=float, default=0.2,
                   help="Pairwise ranking margin for attention-metric fit against detached long-rollout losses.")
    p.add_argument("--etm_metric_rank_weight", type=float, default=0.5,
                   help="Weight on pairwise ranking loss inside the attention-metric fit objective.")
    p.add_argument("--etm_metric_log_fit_weight", type=float, default=1.0,
                   help="Weight on log-score/log-long-loss regression inside the attention-metric fit objective.")
    p.add_argument("--etm_path_source_stride", type=int, default=4,
                   help="For --etm_transport_param path_attn_metric, recompute local residual sources every this many rollout steps, plus the endpoint source for each horizon.")
    p.add_argument("--etm_path_max_sources", type=int, default=0,
                   help="Optional cap on the number of source residuals per horizon for path_attn_metric. 0 means no cap.")
    p.add_argument("--etm_path_source_reduce", type=str, default="mean", choices=["mean", "sum"],
                   help="Reduction over source residuals inside each horizon for path_attn_metric before applying --etm_prop_reduce across horizons.")
    p.add_argument("--etm_prop_reduce", type=str, default="mean", choices=["mean", "sum"],
                   help="Reduction over ETM prop horizons. 'sum' makes the long-horizon metric an accumulated cost instead of averaging back to one-step scale.")
    p.add_argument("--etm_prop_mode", type=str, default="full",
                   choices=["full", "init", "delta", "excess", "guide", "cfg", "cfg_excess", "future_guidance", "geom", "geom_excess", "geometric", "geometric_excess", "signed_guide", "score_correction", "accum", "accumulate", "accumulated", "source", "source_only", "current", "current_transport", "clean_current", "current_source"],
                   help="ETM prop objective: full/init=||J e1||^2; current=fit J to two-rollout current residual transport and penalize ||J e1||^2; guide/cfg_excess=max(0,||J e1||^2-||e1||^2), adding only the CFG-like future score correction; geom_excess=max(0,sum_k ||J_k e1||^2 - H||e1||^2); signed_guide=0.5*(||J e1||^2-||e1||^2); accum/source are residual-source accumulation diagnostics; delta=||(J-I)e1||^2.")
    p.add_argument("--etm_fit_target", type=str, default="sensitivity", choices=["sensitivity", "rollout_error", "long_loss_metric"],
                   help="Target used by the ETM auxiliary head. sensitivity fits J e1 to Delta=Phi(x_{t+1}+e1)-Phi(x_{t+1}); rollout_error fits observed rollout error; long_loss_metric trains an attention metric e^T M e to predict/rank detached long-rollout loss.")
    p.add_argument("--etm_fit_all_steps", action=argparse.BooleanOptionalAction, default=False,
                   help="Fit ETM on multiple horizons 1,1+stride,...,K instead of only endpoint K.")
    p.add_argument("--etm_fit_step_stride", type=int, default=1,
                   help="Stride for multi-horizon ETM fit when --etm_fit_all_steps is enabled.")
    p.add_argument("--etm_strict_horizon", action=argparse.BooleanOptionalAction, default=True,
                   help="If true, skip ETM when available future length is shorter than --etm_horizon; if false, truncate to available length.")
    p.add_argument("--etm_include_tf_residuals", action=argparse.BooleanOptionalAction, default=False,
                   help="Include detached teacher-forced residual source terms in ETM fit: J_{t,k} e1 + sum_r J_{t+r,k-r} b_tf_r ~= e_k.")
    p.add_argument("--etm_residual_step_stride", type=int, default=1,
                   help="Stride over intermediate detached teacher-forced residual sources when --etm_include_tf_residuals is enabled.")
    p.add_argument("--etm_diag_rollout_sensitivity", action=argparse.BooleanOptionalAction, default=False,
                   help="Compute two-rollout test-time sensitivity diagnostics Delta=Phi(x_{t+1}+e1)-Phi(x_{t+1}) while keeping the main ETM loss unchanged.")

    # Compiled surrogate backward-graph loss.  This is the clean implementation
    # of the Loop-SSA/backward-graph-compiler idea: no full BPTT graph is
    # materialized; a no-grad rollout is compiled into pseudo-adjoints that are
    # injected through local one-step graphs.
    p.add_argument("--comp_graph_loss", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable compiled surrogate backward-graph loss for standard AR models.")
    p.add_argument("--comp_lambda", type=float, default=0.0,
                   help="Weight of compiled surrogate backward-graph loss.")
    p.add_argument("--comp_horizon", type=int, default=0,
                   help="Compiled rollout horizon. If 0, fall back to --koopman_gramian_horizon.")
    p.add_argument("--comp_block_size", type=int, default=4,
                   help="Temporal frontier/block size for compiled reverse scan.")
    p.add_argument("--comp_rho", type=float, default=0.95,
                   help="Decay/transport factor used by compiled adjoint scan.")
    p.add_argument("--comp_future_weight", type=float, default=1.0,
                   help="Weight on future compiled adjoints when constructing pseudo-gradients.")
    p.add_argument("--comp_transport", type=str, default="residual_scalar",
                   choices=["identity", "scalar_decay", "residual_scalar", "diagonal", "residual_diagonal", "delete"],
                   help="Strength-reduced block transport rule for compiled reverse scan.")
    p.add_argument("--comp_residual_alpha_max", type=float, default=2.0,
                   help="Clip for residual scalar gate used by --comp_transport residual_scalar/residual_diagonal.")
    p.add_argument("--comp_diag_source", type=str, default="mixed", choices=["residual", "error", "mixed"],
                   help="Feature-wise signal used to build diagonal compiled transport gates.")
    p.add_argument("--comp_diag_identity_mix", type=float, default=0.5,
                   help="Mix diagonal gate with identity: 1.0 is pure identity, 0.0 is full diagonal.")
    p.add_argument("--comp_diag_min", type=float, default=0.25,
                   help="Minimum value after per-sample normalization of diagonal gates.")
    p.add_argument("--comp_diag_max", type=float, default=4.0,
                   help="Maximum value after per-sample normalization of diagonal gates.")
    p.add_argument("--comp_zero_value_loss", action=argparse.BooleanOptionalAction, default=True,
                   help="Use zero-valued pseudo-gradient injection: the compiled term has zero forward value but unchanged gradient, so train/val loss curves stay interpretable.")
    p.add_argument("--comp_start_epoch", type=int, default=0,
                   help="Do not apply compiled-gradient loss before this epoch.")
    p.add_argument("--comp_ramp_epochs", type=int, default=0,
                   help="Linearly ramp --comp_lambda over this many epochs after --comp_start_epoch.")
    p.add_argument("--comp_normalize_adjoints", action=argparse.BooleanOptionalAction, default=True,
                   help="Normalize compiled pseudo-adjoints to the local error RMS scale.")
    p.add_argument("--comp_max_adj_ratio", type=float, default=5.0,
                   help="Maximum pseudo-adjoint RMS scale relative to local error RMS.")
    p.add_argument("--comp_sparsify_keep_frac", type=float, default=1.0,
                   help="Optional path sparsification: keep only this fraction of largest step pseudo-adjoints per sample.")
    p.add_argument("--comp_injection_stride", type=int, default=1,
                   help="Inject pseudo-gradients every N rollout steps to control compute.")
    p.add_argument("--comp_max_injection_steps", type=int, default=0,
                   help="Optional cap on number of local injection steps per sampled rollout; 0 means no cap.")
    p.add_argument("--comp_eval", action=argparse.BooleanOptionalAction, default=False,
                   help="Also compute compiled loss in validation evaluate_loss. Disabled by default because it is expensive and changes val/loss semantics.")

    # Source-colored local repair.  This is a no-gradient long-rollout diagnosis
    # plus one-step on-policy local repair: estimate the future footprints of
    # each local innovation source, orthogonalize them, project the final rollout
    # error onto those source directions, and use the resulting detached source
    # weights to reweight one-step losses at rollout states.
    p.add_argument("--source_color_loss", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable source-colored no-grad rollout diagnosis + source-weighted on-policy local repair.")
    p.add_argument("--source_color_lambda", type=float, default=0.0,
                   help="Weight for source-colored on-policy local repair loss.")
    p.add_argument("--source_color_horizon", type=int, default=0,
                   help="Diagnosis horizon K. If 0, fall back to --koopman_gramian_horizon.")
    p.add_argument("--source_color_perturb_eps", type=float, default=1e-3,
                   help="Small perturbation scale used to estimate each source's future footprint.")
    p.add_argument("--source_color_temperature", type=float, default=1.0,
                   help="Softmax temperature for converting source coefficients into local repair weights.")
    p.add_argument("--source_color_weight_mode", type=str, default="softmax", choices=["softmax", "normalize", "top1"],
                   help="How to convert absolute source coefficients into repair weights.")
    p.add_argument("--source_color_min_norm", type=float, default=1e-8,
                   help="Minimum Gram-Schmidt direction norm; smaller source directions are treated as invalid.")
    p.add_argument("--source_color_local_loss", type=str, default="rel_l2", choices=["mse", "huber", "l1", "rel_l2"],
                   help="Per-sample one-step loss used for the source-weighted local repair branch.")
    p.add_argument("--source_color_repair_topk", type=int, default=8,
                   help="Number of source-colored positions to backprop through. Use 0 or negative to keep all K sources.")
    p.add_argument("--source_color_repair_frac", type=float, default=0.0,
                   help="Alternative sparse repair budget as a fraction of K. Ignored when repair_topk > 0.")
    p.add_argument("--source_color_renorm_selected", action=argparse.BooleanOptionalAction, default=True,
                   help="Renormalize source weights over the selected top-k repair positions.")
    p.add_argument("--source_color_strict_horizon", action=argparse.BooleanOptionalAction, default=False,
                   help="If true, skip source-color diagnosis when fewer than K future steps are available. If false, use the available shorter horizon.")
    p.add_argument("--source_color_eval_mode_diagnosis", action=argparse.BooleanOptionalAction, default=True,
                   help="Temporarily put the raw model in eval mode while constructing no-grad source footprints.")
    p.add_argument("--source_color_eval", action=argparse.BooleanOptionalAction, default=False,
                   help="Also compute source-colored loss during validation. Disabled by default because it is expensive and changes val/loss semantics.")

    # Error-alignment objective with fixed notation:
    #   delta_t^AR = x_AR_t - x_t = p_t + b_t,
    #   p_t = x_AR_t - x_TF_t is detached,
    #   b_t = x_TF_t - x_t has gradient.
    p.add_argument("--error_align_loss", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable long no-grad AR / TF positive alignment loss using p_t=x_AR-x_TF and b_t=x_TF-x.")
    p.add_argument("--error_align_lambda", type=float, default=0.0,
                   help="Weight for positive alignment loss [cos(p_t,b_t)]_+^2.")
    p.add_argument("--error_align_tf_lambda", type=float, default=1.0,
                   help="Weight for the auxiliary teacher-forced loss ||b_t|| at alignment times. Set 0 for alignment-only.")
    p.add_argument("--error_align_horizon", type=int, default=64,
                   help="Long no-grad AR horizon used to expose transported error p_t.")
    p.add_argument("--error_align_num_times", type=int, default=8,
                   help="Number of time indices sampled inside the long rollout for TF alignment.")
    p.add_argument("--error_align_time_stride", type=int, default=1,
                   help="Candidate stride for sampled alignment time indices.")
    p.add_argument("--error_align_random_times", action=argparse.BooleanOptionalAction, default=True,
                   help="Randomly sample alignment time indices inside the long AR horizon.")
    p.add_argument("--error_align_tf_loss", type=str, default="rel_l2", choices=["mse", "huber", "l1", "rel_l2"],
                   help="Teacher-forced loss used for b_t in the alignment stage.")
    p.add_argument("--error_align_eps", type=float, default=1e-8,
                   help="Numerical epsilon for alignment normalization and diagnostics.")
    p.add_argument("--error_align_min_norm", type=float, default=1e-8,
                   help="Ignore alignment samples with ||p_t|| or ||b_t|| below this threshold.")
    p.add_argument("--error_align_strict_horizon", action=argparse.BooleanOptionalAction, default=False,
                   help="Skip alignment if the full requested long horizon is unavailable.")
    p.add_argument("--error_align_eval_mode_rollout", action=argparse.BooleanOptionalAction, default=True,
                   help="Temporarily use eval() for the no-grad long AR rollout used to construct p_t.")
    p.add_argument("--error_align_eval", action=argparse.BooleanOptionalAction, default=False,
                   help="Also compute error alignment loss during validation. Disabled by default.")


    # Live-gradient-frontier / sensitivity-pruned BPTT loss.  This keeps the
    # ordinary one-step loss and adds a sparse long-gradient graph whose branches
    # are progressively pruned by a normalized frontier threshold tau in [0, 1].
    p.add_argument("--frontier_graph_loss", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable sensitivity-pruned live-gradient-frontier loss for standard AR models.")
    p.add_argument("--frontier_lambda", type=float, default=0.0,
                   help="Weight of live-gradient-frontier loss.")
    p.add_argument("--frontier_horizon", type=int, default=0,
                   help="Live-frontier rollout horizon. If 0, fall back to --comp_horizon, then --koopman_gramian_horizon.")
    p.add_argument("--frontier_tau", type=float, default=0.05,
                   help="Normalized pruning threshold in [0,1]. A direction survives if |adjoint_i| / max_j |adjoint_j| >= tau.")
    p.add_argument("--frontier_min_keep_frac", type=float, default=0.0,
                   help="Minimum fraction of adjoint coordinates kept per sample, used as a safety floor.")
    p.add_argument("--frontier_include_current_error", action=argparse.BooleanOptionalAction, default=True,
                   help="Add the current rollout error to the live frontier before pruning and propagation.")
    p.add_argument("--frontier_skip_first_injection", action=argparse.BooleanOptionalAction, default=True,
                   help="Skip local injection at rollout step 0 to avoid duplicating the ordinary one-step loss.")
    p.add_argument("--frontier_zero_value_loss", action=argparse.BooleanOptionalAction, default=True,
                   help="Use zero-valued pseudo-gradient injection for live-frontier loss.")
    p.add_argument("--frontier_start_epoch", type=int, default=0,
                   help="Do not apply live-frontier loss before this epoch.")
    p.add_argument("--frontier_ramp_epochs", type=int, default=0,
                   help="Linearly ramp --frontier_lambda over this many epochs after --frontier_start_epoch.")
    p.add_argument("--frontier_normalize_adjoints", action=argparse.BooleanOptionalAction, default=True,
                   help="Normalize live-frontier adjoints to the local error RMS scale before gradient injection.")
    p.add_argument("--frontier_max_adj_ratio", type=float, default=5.0,
                   help="Maximum live-frontier pseudo-adjoint RMS scale relative to local error RMS.")
    p.add_argument("--frontier_injection_stride", type=int, default=1,
                   help="Inject live-frontier pseudo-gradients every N rollout steps.")
    p.add_argument("--frontier_max_injection_steps", type=int, default=0,
                   help="Optional cap on number of live-frontier local injection steps per sampled rollout; 0 means no cap.")
    p.add_argument("--frontier_eval", action=argparse.BooleanOptionalAction, default=False,
                   help="Also compute live-frontier loss during validation. Disabled by default because it is expensive.")
    p.add_argument("--frontier_grad_scale", type=float, default=1.0,
                   help="Extra scalar multiplier for strict manual frontier gradients before optimizer step.")
    p.add_argument("--frontier_error_mode", type=str, default="raw_error",
                   choices=["raw_error", "mse_grad", "mean_mse_grad"],
                   help="Coordinate error used as the frontier source term e_s. raw_error uses pred-target; mse_grad scales by 2/numel.")
    p.add_argument("--frontier_train_starts_per_sequence", type=int, default=-1,
                   help="Number of startpoints for strict frontier pass. If <=0, use --ar_train_starts_per_sequence.")
    p.add_argument("--frontier_random_starts", action=argparse.BooleanOptionalAction, default=True,
                   help="Randomly sample frontier startpoints during training.")


    # Exact full-BPTT rollout loss for comparison/diagnostics.
    p.add_argument("--bptt_loss", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable exact full-BPTT multi-step rollout loss for standard AR models.")
    p.add_argument("--bptt_lambda", type=float, default=0.0,
                   help="Weight of exact full-BPTT rollout loss.")
    p.add_argument("--bptt_horizon", type=int, default=0,
                   help="Full-BPTT rollout horizon. If 0, fall back to --koopman_gramian_horizon.")
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
    p.add_argument("--bptt_reinforce_lambda", type=float, default=0.0,
                   help="Inside the BPTT window, penalize positive reinforcement [cos(p_k,b_k)]_+^2, where p_k=stopgrad(x_AR-x_TF) and b_k=x_TF-x.")
    p.add_argument("--bptt_cloud_orth_lambda", type=float, default=0.0,
                   help="Inside the BPTT window, penalize off-diagonal cosine^2 among local TF error directions b_k.")
    p.add_argument("--bptt_cloud_mean_lambda", type=float, default=0.0,
                   help="Inside the BPTT window, penalize squared norm of the mean normalized local TF error direction.")
    p.add_argument("--bptt_cloud_radius_lambda", type=float, default=0.0,
                   help="Optional hinge penalty preventing local TF error energy from exceeding --bptt_cloud_radius_target times a detached reference energy.")
    p.add_argument("--bptt_cloud_radius_target", type=float, default=1.0,
                   help="Radius target for the optional BPTT error-cloud radius hinge. Usually leave the lambda at 0 for first experiments.")
    p.add_argument("--bptt_error_cloud_eps", type=float, default=1e-8,
                   help="Numerical epsilon for BPTT error-cloud cosine normalization and diagnostics.")
    p.add_argument("--bptt_error_cloud_min_norm", type=float, default=1e-8,
                   help="Minimum vector norm for valid BPTT error-cloud cosine samples.")
    p.add_argument("--bptt_error_cloud_eval", action=argparse.BooleanOptionalAction, default=False,
                   help="Compute BPTT error-cloud diagnostics during validation when --bptt_eval is also enabled.")

    # Detached long-horizon error-cloud loss.  This keeps short BPTT as the
    # local accuracy anchor, but probes a much longer no-grad AR rollout and
    # regularizes local TF residuals at long offsets.
    p.add_argument("--long_error_cloud_loss", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable detached long-horizon error-cloud regularization. The long AR rollout is no-grad; gradients flow only through local TF predictions at sampled long offsets.")
    p.add_argument("--long_error_cloud_eval", action=argparse.BooleanOptionalAction, default=False,
                   help="Also compute detached long-horizon error-cloud diagnostics during validation.")
    p.add_argument("--long_error_cloud_horizon", type=int, default=64,
                   help="No-grad AR rollout horizon used by the long error-cloud probe.")
    p.add_argument("--long_error_cloud_offsets", type=str, default="16,24,32,48,64",
                   help="Comma-separated one-based horizon steps used for long error-cloud TF residuals. Example: 16,24,32,48,64. Empty string uses stride/random sampling.")
    p.add_argument("--long_error_cloud_num_times", type=int, default=5,
                   help="Number of long offsets to sample when --long_error_cloud_offsets is empty.")
    p.add_argument("--long_error_cloud_time_stride", type=int, default=8,
                   help="Stride for candidate one-based horizon steps when --long_error_cloud_offsets is empty.")
    p.add_argument("--long_error_cloud_random_times", action=argparse.BooleanOptionalAction, default=False,
                   help="Randomly sample long offsets when --long_error_cloud_offsets is empty. Defaults to deterministic for easier comparison.")
    p.add_argument("--long_error_cloud_strict_horizon", action=argparse.BooleanOptionalAction, default=False,
                   help="Skip samples shorter than --long_error_cloud_horizon instead of clipping the horizon.")
    p.add_argument("--long_error_cloud_eval_mode_rollout", action=argparse.BooleanOptionalAction, default=True,
                   help="Run the no-grad long AR probe in eval mode, then restore train mode for local TF branches.")
    p.add_argument("--long_error_cloud_rein_lambda", type=float, default=0.0,
                   help="Long-cloud cosine reinforcement weight for [cos(p_k,b_k)]_+^2 at long offsets. This is kept for ablations/diagnostics; the energy loss below is the preferred closed-loop objective.")
    p.add_argument("--long_error_cloud_energy_lambda", type=float, default=0.0,
                   help="Long-cloud closed-loop energy anti-reinforcement weight. Penalizes local residuals b_k that increase ||p_k+b_k||^2 beyond ||p_k||^2 + rho ||b_k||^2.")
    p.add_argument("--long_error_cloud_energy_rho", type=float, default=1.0,
                   help="Energy anti-reinforcement tolerance rho. rho=1 only penalizes positive cross-energy; rho<1 requires the local residual to partially repair accumulated error.")
    p.add_argument("--long_error_cloud_orth_lambda", type=float, default=0.0,
                   help="Long-cloud off-diagonal cosine^2 weight among sampled long-offset TF residual directions.")
    p.add_argument("--long_error_cloud_mean_lambda", type=float, default=0.0,
                   help="Long-cloud squared mean normalized TF residual direction weight.")
    p.add_argument("--long_error_cloud_eps", type=float, default=1e-8,
                   help="Numerical epsilon for long error-cloud cosine normalization and diagnostics.")
    p.add_argument("--long_error_cloud_min_norm", type=float, default=1e-8,
                   help="Minimum vector norm for valid long error-cloud cosine samples.")

    # Forced Damped Error Dynamics (FDED): a lightweight Newtonian-style
    # regularizer on AR rollout error dynamics. Default mode is endpoint-window
    # Net Error-energy Drift (NED): a detached AR state initializes a short graph
    # rollout, and the loss penalizes input-unexplained endpoint energy growth.
    p.add_argument("--forced_damped_error_loss", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable Forced Damped Error Dynamics / endpoint Net Error-energy Drift regularization.")
    p.add_argument("--forced_damped_eval", action=argparse.BooleanOptionalAction, default=False,
                   help="Also log FDED diagnostics during validation.")
    p.add_argument("--forced_damped_horizon", type=int, default=8,
                   help="No-grad AR rollout horizon / maximum endpoint horizon used by FDED/NED.")
    p.add_argument("--forced_damped_offsets", type=str, default="8",
                   help="Comma-separated one-based endpoint horizons for FDED/NED. Empty string uses stride/num_times.")
    p.add_argument("--forced_damped_endpoint_window", action=argparse.BooleanOptionalAction, default=True,
                   help="Use endpoint-window Net Error-energy Drift. Disable for the older local-edge FDED ablation.")
    p.add_argument("--forced_damped_bptt_span", type=int, default=8,
                   help="Maximum short graph-rollout span for endpoint-window NED from a detached AR start state.")
    p.add_argument("--forced_damped_num_times", type=int, default=5,
                   help="Number of FDED offsets when --forced_damped_offsets is empty.")
    p.add_argument("--forced_damped_time_stride", type=int, default=8,
                   help="Candidate stride for FDED offsets when --forced_damped_offsets is empty.")
    p.add_argument("--forced_damped_random_times", action=argparse.BooleanOptionalAction, default=False,
                   help="Randomly sample FDED offsets when --forced_damped_offsets is empty.")
    p.add_argument("--forced_damped_strict_horizon", action=argparse.BooleanOptionalAction, default=False,
                   help="Skip samples shorter than --forced_damped_horizon instead of clipping the horizon.")
    p.add_argument("--forced_damped_eval_mode_rollout", action=argparse.BooleanOptionalAction, default=True,
                   help="Run the detached FDED/NED rollout in eval mode, then restore train mode for trainable endpoint rollout.")
    p.add_argument("--forced_damped_energy_lambda", type=float, default=0.0,
                   help="FDED input-conditioned positive net error-energy growth penalty weight.")
    p.add_argument("--forced_damped_damping_lambda", type=float, default=0.0,
                   help="Older local-edge FDED damping penalty weight. Ignored by endpoint-window NED unless --no-forced_damped_endpoint_window.")
    p.add_argument("--forced_damped_accel_lambda", type=float, default=0.0,
                   help="Older local-edge FDED acceleration penalty weight. Ignored by endpoint-window NED unless --no-forced_damped_endpoint_window.")
    p.add_argument("--forced_damped_input_beta", type=float, default=0.0,
                   help="External-input work budget coefficient. Positive values allow more error-energy growth when stimulus changes are large.")
    p.add_argument("--forced_damped_damping_margin", type=float, default=0.0,
                   help="Margin for damping: penalize cos(e,v)+margin > 0.")
    p.add_argument("--forced_damped_accel_margin", type=float, default=0.0,
                   help="Margin for acceleration: penalize cos(e,a)+margin > 0.")
    p.add_argument("--forced_damped_eps", type=float, default=1e-8,
                   help="Numerical epsilon for FDED normalization.")
    p.add_argument("--forced_damped_min_norm", type=float, default=1e-8,
                   help="Minimum error/velocity/acceleration norm for valid FDED cosine samples.")

    p.add_argument("--transport_coh_loss", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable transported error-source coherence regularization.")
    p.add_argument("--transport_coh_eval", action=argparse.BooleanOptionalAction, default=False,
                   help="Log transported coherence diagnostics during validation without requiring a nonzero training weight.")
    p.add_argument("--transport_coh_horizon", type=int, default=64,
                   help="Final horizon K for transported source coherence.")
    p.add_argument("--transport_coh_sources", type=str, default="8,16,32,48",
                   help="Comma-separated one-based source horizons h. Empty string uses stride/num_sources.")
    p.add_argument("--transport_coh_num_sources", type=int, default=4,
                   help="Number of source horizons to use when --transport_coh_sources is empty.")
    p.add_argument("--transport_coh_source_stride", type=int, default=8,
                   help="Candidate source stride when --transport_coh_sources is empty.")
    p.add_argument("--transport_coh_random_sources", action=argparse.BooleanOptionalAction, default=False,
                   help="Randomly sample source horizons instead of deterministic spread.")
    p.add_argument("--transport_coh_max_sources", type=int, default=0,
                   help="Optional cap on source count after parsing. 0 means no cap.")
    p.add_argument("--transport_coh_adj_lambda", type=float, default=0.0,
                   help="Version 1: adjacent one-step transported coherence weight.")
    p.add_argument("--transport_coh_multi_lambda", type=float, default=0.0,
                   help="Version 2: multi-step transported pairwise coherence weight.")
    p.add_argument("--transport_coh_proj_lambda", type=float, default=0.0,
                   help="Version 3: projected transported pairwise coherence weight.")
    p.add_argument("--transport_coh_proj_mean_lambda", type=float, default=0.0,
                   help="Version 3: projected spherical mean-direction penalty weight.")
    p.add_argument("--transport_coh_proj_dim", type=int, default=16,
                   help="Random projection dimension for projected spherical version.")
    p.add_argument("--transport_coh_proj_seed", type=int, default=12345,
                   help="Fixed random projection seed.")
    p.add_argument("--transport_coh_top_m", type=int, default=0,
                   help="Use only top-m transported source norms for pairwise loss. 0 means all valid sources.")
    p.add_argument("--transport_coh_weight_power", type=float, default=0.0,
                   help="Detached norm importance power for source-pair weighting. 0 means unweighted.")
    p.add_argument("--transport_coh_positive_only", action=argparse.BooleanOptionalAction, default=True,
                   help="Penalize only positive cosine; negative cosine is treated as cancellation.")
    p.add_argument("--transport_coh_detach_norm", action=argparse.BooleanOptionalAction, default=True,
                   help="Detach cosine normalization norms so the loss mainly shapes direction.")
    p.add_argument("--transport_coh_strict_jvp", action=argparse.BooleanOptionalAction, default=False,
                   help="Use strict=True for autograd.functional.jvp; default false is safer for complex wrappers.")
    p.add_argument("--transport_coh_lowmem", action=argparse.BooleanOptionalAction, default=True,
                   help="Memory-safe mode: long transported diagnostics are detached; V2/V3 use one-step trainable proxy directions to avoid storing K-step JVP graphs.")
    p.add_argument("--transport_coh_eps", type=float, default=1e-8,
                   help="Numerical epsilon for transported coherence losses.")
    p.add_argument("--transport_coh_min_norm", type=float, default=1e-8,
                   help="Minimum norm mask for transported coherence vectors.")

    # Error-projection pseudo-target distillation.  A detached AR rollout builds
    # an oracle anti-reinforcement output correction, while gradients flow only
    # through fresh one-step predictions at selected rollout states.
    p.add_argument("--error_proj_pseudo_loss", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable BPTT-free projection-target distillation from detached rollout error geometry.")
    p.add_argument("--error_proj_pseudo_eval", action=argparse.BooleanOptionalAction, default=False,
                   help="Log projection-target distillation diagnostics during validation.")
    p.add_argument("--error_proj_pseudo_lambda", type=float, default=0.0,
                   help="Weight for projection-target pseudo loss.")
    p.add_argument("--error_proj_pseudo_horizon", type=int, default=64,
                   help="Detached AR rollout horizon used to build projection pseudo-targets.")
    p.add_argument("--error_proj_pseudo_sources", type=str, default="8,16,32,48",
                   help="Comma-separated one-based rollout steps used for pseudo-target fitting. Empty string uses stride.")
    p.add_argument("--error_proj_pseudo_source_stride", type=int, default=8,
                   help="Candidate source stride when --error_proj_pseudo_sources is empty.")
    p.add_argument("--error_proj_pseudo_max_sources", type=int, default=0,
                   help="Optional cap on pseudo source count after parsing. 0 means no cap.")
    p.add_argument("--error_proj_pseudo_gamma", type=float, default=1.0,
                   help="Oracle projection strength gamma used to form pseudo-targets.")
    p.add_argument("--error_proj_pseudo_loss_type", type=str, default="rel_l2", choices=["mse", "huber", "l1", "rel_l2"],
                   help="Local loss used to fit projection pseudo-targets.")
    p.add_argument("--error_proj_pseudo_eps", type=float, default=1e-8,
                   help="Numerical epsilon for projection pseudo-targets.")
    p.add_argument("--error_proj_pseudo_min_error_norm", type=float, default=1e-8,
                   help="Minimum accumulated rollout error norm required to apply projection correction.")
    p.add_argument("--error_proj_pseudo_detach_norm", action=argparse.BooleanOptionalAction, default=True,
                   help="Detach projection normalization norms when building pseudo-targets.")

    # Error-projection V3: coboundary-filtered Poseido.  This keeps the V1
    # accumulated-error correction direction, but subtracts the component of the
    # local reinforcement scalar that can be explained as a potential difference.
    p.add_argument("--error_proj_cob_loss", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable coboundary-filtered Poseido pseudo-target loss.")
    p.add_argument("--error_proj_cob_eval", action=argparse.BooleanOptionalAction, default=False,
                   help="Log coboundary-filtered Poseido diagnostics during validation.")
    p.add_argument("--error_proj_cob_lambda", type=float, default=0.0,
                   help="Weight for coboundary-filtered Poseido pseudo loss.")
    p.add_argument("--error_proj_cob_horizon", type=int, default=64,
                   help="Detached AR rollout horizon used to fit coboundary potential and build pseudo-targets.")
    p.add_argument("--error_proj_cob_sources", type=str, default="8,16,32,48",
                   help="Comma-separated one-based rollout steps used for coboundary pseudo-target fitting. Empty string uses stride.")
    p.add_argument("--error_proj_cob_source_stride", type=int, default=8,
                   help="Candidate coboundary source stride when --error_proj_cob_sources is empty.")
    p.add_argument("--error_proj_cob_max_sources", type=int, default=0,
                   help="Optional cap on coboundary pseudo source count after parsing. 0 means no cap.")
    p.add_argument("--error_proj_cob_gamma", type=float, default=1.0,
                   help="Projection strength gamma used after subtracting the fitted coboundary component.")
    p.add_argument("--error_proj_cob_ridge", type=float, default=1e-3,
                   help="Ridge penalty for the coboundary scale fit.")
    p.add_argument("--error_proj_cob_potential", type=str, default="energy", choices=["energy", "fit_full", "fit_full_no_time"],
                   help="Coboundary potential mode. fit_full now matches the offline diagnostic feature bank and pooled ridge fit; energy uses Phi(e)=c||e||^2.")
    p.add_argument("--error_proj_cob_energy_scale", type=float, default=0.5,
                   help="Scale c for known energy potential Phi(e)=c||e||^2.")
    p.add_argument("--error_proj_cob_fit_scale", action=argparse.BooleanOptionalAction, default=True,
                   help="Fit one scalar beta per detached rollout for the fixed energy potential; disable for fully fixed beta=1.")
    p.add_argument("--error_proj_cob_loss_type", type=str, default="rel_l2", choices=["mse", "huber", "l1", "rel_l2"],
                   help="Local loss used to fit coboundary-filtered pseudo-targets.")
    p.add_argument("--error_proj_cob_eps", type=float, default=1e-8,
                   help="Numerical epsilon for coboundary-filtered projection pseudo-targets.")
    p.add_argument("--error_proj_cob_min_error_norm", type=float, default=1e-8,
                   help="Minimum accumulated rollout error norm required to apply coboundary correction.")
    p.add_argument("--error_proj_cob_detach_norm", action=argparse.BooleanOptionalAction, default=True,
                   help="Detach projection normalization norms when building coboundary pseudo-targets.")

    # Error-projection V2: adjacent transported local innovation.  This is a
    # separate loss from V1 and does not change/remove any V1 options above.
    p.add_argument("--error_proj_v2_loss", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable V2 transported-defect pseudo-target loss. Public V2 flags are preserved, but V2 now uses a detached pseudo target rather than a cosine penalty.")
    p.add_argument("--error_proj_v2_eval", action=argparse.BooleanOptionalAction, default=False,
                   help="Log V2 transported-defect pseudo-target diagnostics during validation.")
    p.add_argument("--error_proj_v2_lambda", type=float, default=0.0,
                   help="Weight for V2 transported-defect pseudo-target loss.")
    p.add_argument("--error_proj_v2_horizon", type=int, default=64,
                   help="Detached AR rollout horizon used to collect adjacent source pairs for V2.")
    p.add_argument("--error_proj_v2_sources", type=str, default="8,16,32,48",
                   help="Comma-separated one-based rollout source steps s used for V2 pairs (s,s+1). Empty string uses stride.")
    p.add_argument("--error_proj_v2_source_stride", type=int, default=8,
                   help="Candidate V2 source stride when --error_proj_v2_sources is empty.")
    p.add_argument("--error_proj_v2_max_pairs", type=int, default=0,
                   help="Optional cap on V2 adjacent pair count after parsing. 0 means no cap.")
    p.add_argument("--error_proj_v2_fd_eps", type=float, default=1e-3,
                   help="Finite-difference epsilon for V2 JVP approximation.")
    p.add_argument("--error_proj_v2_min_norm", type=float, default=1e-8,
                   help="Minimum norm mask for V2 transported/current defect cosine.")
    p.add_argument("--error_proj_v2_normalize_direction", action=argparse.BooleanOptionalAction, default=True,
                   help="Normalize b_s before finite difference and rescale back to approximate J b_s.")
    p.add_argument("--error_proj_v2_gamma", type=float, default=1.0,
                   help="Strength of the V2 pseudo-target correction along -J_s b_s.")
    p.add_argument("--error_proj_v2_eps", type=float, default=1e-8,
                   help="Numerical epsilon for V2 projection denominators.")
    p.add_argument("--error_proj_v2_loss_type", type=str, default="rel_l2", choices=["mse", "huber", "l1", "rel_l2"],
                   help="Local fit loss used by V2 pseudo-target training.")
    p.add_argument("--error_proj_v2_detach_norm", action=argparse.BooleanOptionalAction, default=True,
                   help="Detach V2 projection denominators when building pseudo-targets.")

    # standard autoregressive training/evaluation options for FNO-style models
    p.add_argument("--ar_loss", type=str, default="rel_l2", choices=["mse", "huber", "l1", "rel_l2"], help="One-step supervised loss for standard AR baselines. Original FNO commonly uses relative Lp/L2 loss.")
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
            "batch-conditioned global-horizon Wiener calibration."
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
            "LR scheduler for standard AR/FNO models. cosine uses "
            "CosineAnnealingLR over --num_epochs."
        ),
    )
    p.add_argument("--ar_step_size", type=int, default=100, help="StepLR step_size for standard AR/FNO models.")
    p.add_argument("--ar_gamma", type=float, default=0.5, help="StepLR decay factor for standard AR/FNO models.")
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
    p.add_argument("--eval_free_rollout_curves", action=argparse.BooleanOptionalAction, default=True,
                   help="During test evaluation, save free-rollout per-step corr/error curves for standard AR models.")

    # Koopman-Gram model/loss
    p.add_argument("--hidden_dim", type=int, default=4096)
    p.add_argument("--koopman_horizons", type=int, nargs="+", default=[1])
    p.add_argument("--koopman_train_stride", type=int, default=4)
    p.add_argument("--koopman_eval_stride", type=int, default=4)
    p.add_argument("--koopman_stim_depth", type=int, default=0)
    p.add_argument("--koopman_stim_nhead", type=int, default=4)
    p.add_argument("--koopman_dropout", type=float, default=0.0)
    p.add_argument("--koopman_lambda_one", type=float, default=1.0)
    p.add_argument("--koopman_lambda_point", type=float, default=1.0)
    p.add_argument("--koopman_lambda_delta", type=float, default=0.0)
    p.add_argument("--koopman_lambda_gram", type=float, default=1.0)
    p.add_argument("--koopman_lambda_rec", type=float, default=0.0)
    p.add_argument("--koopman_lambda_latent", type=float, default=0.0)
    p.add_argument(
        "--koopman_long_loss",
        type=str,
        default="gram_error",
        choices=["gram_error", "supervised_folded", "stimulus_folded", "folded_auto", "both", "none"],
        help=(
            "KG long loss mode. folded_auto chooses stimulus_folded when the dataset has external input, "
            "otherwise supervised_folded. both means gram_error + the same auto-selected folded supervised loss."
        ),
    )
    p.add_argument("--koopman_gramian_horizon", type=int, default=64)
    p.add_argument(
        "--koopman_folded_future_stimulus",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Deprecated compatibility flag. Folded stimulus offsets are now selected automatically: "
            "stimulus_folded uses them only when the dataset has external input; supervised_folded never uses them."
        ),
    )
    p.add_argument(
        "--koopman_adaptive_gramian_horizon",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use K_eff=min(koopman_gramian_horizon, remaining future length) for folded samples.",
    )
    p.add_argument("--koopman_normalize_gram", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--koopman_stable_linear", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--koopman_spectral_bound", type=float, default=1.02)
    p.add_argument("--koopman_residual_transition", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--koopman_dt", type=float, default=0.1)
    p.add_argument("--koopman_damping", type=float, default=0.0)
    p.add_argument("--koopman_force_scale", type=float, default=1.0)
    p.add_argument("--koopman_detach_rollout_history", action=argparse.BooleanOptionalAction, default=False)

    # optional learned Koopman encoder / latent channelization
    p.add_argument("--use_koopman_encoder", action="store_true", default=False)
    p.add_argument("--koopman_latent_dim", type=int, default=None)
    p.add_argument("--koopman_encoder_mid_dim", type=int, default=None)
    p.add_argument("--no_koopman_encoder_residual", action="store_true", default=False)
    p.add_argument("--koopman_latent_channels", type=int, default=None)
    p.add_argument("--koopman_latent_channel_dim", type=int, default=None)
    p.add_argument("--koopman_channel_shared_A", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--koopman_train_stage", type=str, default="joint", choices=["joint", "ae", "transition", "adapter", "finetune"])
    p.add_argument("--koopman_use_latent_adapter", action="store_true", default=False)
    p.add_argument("--koopman_adapter_dim", type=int, default=256)
    p.add_argument("--koopman_adapter_alpha", type=float, default=0.1)

    p.add_argument("--ridge_alpha", type=float, default=1e-2)
    p.add_argument("--ridge_fit_closed_form", action="store_true")
    p.add_argument("--ridge_finetune_with_kg", action="store_true")
    p.add_argument("--ridge_use_history", action="store_true")
    p.add_argument("--ridge_history_mode", type=str, default="current", choices=["current", "window"])
    p.add_argument("--ridge_window_standardize", action=argparse.BooleanOptionalAction, default=True,
                   help="Fit train-set StandardScaler statistics for ridge_window features, matching old RR.")
    p.add_argument("--ridge_window_fit_closed_form", action="store_true", default=False,
                   help="For model_name=ridge_window, fit old-style StandardScaler+sklearn Ridge and load the solution into the PyTorch model before optional fine-tuning.")
    p.add_argument("--ridge_window_finetune", action="store_true", default=False,
                   help="Continue gradient training after --ridge_window_fit_closed_form. Without this flag, train_and_test evaluates the closed-form RR solution directly.")
    p.add_argument("--ridge_window_max_fit_samples", type=int, default=0,
                   help="Optional cap for old-style closed-form RR fit. 0 uses all training windows.")
    p.add_argument("--ridge_window_fit_stride", type=int, default=1,
                   help="Temporal stride for old-style closed-form RR fit. Use 1 to match the old RR code.")
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
