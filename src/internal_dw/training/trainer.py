from __future__ import annotations

import errno
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist

from internal_dw.evaluation.eval import evaluate_loss, evaluate_horizon_sweep, save_eval_json
from internal_dw.training.losses import compute_koopman_gramian_loss
from internal_dw.training.ar_losses import (
    compute_autoregressive_one_step_loss,
    compute_potential_ae_loss,
    strict_live_frontier_backward,
    backward_recurrent_state_cto_cut_loss,
)
from internal_dw.data_utils.state_ops import unpack_batch, zero_external_input_like, time_window
from internal_dw.utils import is_rank0, save_checkpoint, unwrap_model


def configure_trainable(model, args, rank=0):
    raw = unwrap_model(model)
    stage = str(args.koopman_train_stage)
    if getattr(raw, "is_clock_latent_ae", False):
        cl_stage = str(getattr(args, "clocklat_stage", "ae"))
        if hasattr(raw, "configure_trainable"):
            raw.configure_trainable(
                stage=cl_stage,
                freeze_ae=bool(getattr(args, "clocklat_freeze_ae", True)),
                tune_adapter=bool(getattr(args, "clocklat_tune_adapter", True)),
            )
        if is_rank0(rank):
            total = sum(p.numel() for p in raw.parameters())
            trainable = sum(p.numel() for p in raw.parameters() if p.requires_grad)
            print(f"[Trainable] clock-latent AR stage={cl_stage}; trainable={trainable:,}/{total:,}")
        return

    if getattr(raw, "is_state_sequence_ae", False):
        st_stage = str(getattr(args, "statetok_stage", "ae"))
        if hasattr(raw, "configure_trainable"):
            raw.configure_trainable(
                stage=st_stage,
                freeze_ae=bool(getattr(args, "statetok_freeze_ae", True)),
                tune_adapter=bool(getattr(args, "statetok_tune_adapter", True)),
            )
        if is_rank0(rank):
            total = sum(p.numel() for p in raw.parameters())
            trainable = sum(p.numel() for p in raw.parameters() if p.requires_grad)
            print(f"[Trainable] state-sequence AE stage={st_stage}; trainable={trainable:,}/{total:,}")
        return

    if getattr(raw, "is_standard_autoregressive", False):
        if is_rank0(rank):
            total = sum(p.numel() for p in raw.parameters())
            trainable = sum(p.numel() for p in raw.parameters() if p.requires_grad)
            print(f"[Trainable] standard autoregressive model; trainable={trainable:,}/{total:,}")
        return
    if not getattr(raw, "use_koopman_encoder", False):
        if is_rank0(rank):
            print(f"[Trainable] raw Koopman mode; ignoring stage={stage}")
        return

    for p in raw.parameters():
        p.requires_grad = False

    def enable(module):
        if module is not None:
            for p in module.parameters():
                p.requires_grad = True

    stim_modules = [raw.stim_in, raw.stim_seq_in, raw.stim_encoder, raw.force_head]
    if stage == "joint" or stage == "finetune":
        for p in raw.parameters():
            p.requires_grad = True
    elif stage == "ae":
        enable(raw.koopman_encoder)
        enable(raw.koopman_decoder)
    elif stage == "transition":
        raw.A_delta.requires_grad = True
        raw.log_force_scale.requires_grad = True
        for m in stim_modules:
            enable(m)
    elif stage == "adapter": #LoRA-like adapter training
        if raw.koopman_latent_adapter is None:
            raise ValueError("koopman_train_stage=adapter requires --koopman_use_latent_adapter")
        enable(raw.koopman_latent_adapter)
        raw.A_delta.requires_grad = True
        raw.log_force_scale.requires_grad = True
        for m in stim_modules:
            enable(m)
    else:
        raise ValueError(f"Unknown koopman_train_stage={stage}")

    if is_rank0(rank):
        total = sum(p.numel() for p in raw.parameters())
        trainable = sum(p.numel() for p in raw.parameters() if p.requires_grad)
        print(f"[Trainable] stage={stage}; trainable={trainable:,}/{total:,}")



def _json_safe_float(value):
    """Convert numbers/tensors to JSON-safe Python scalars."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    if isinstance(value, (int, float)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def _compact_epoch_logs(logs: dict) -> dict:
    """Drop disabled/zero diagnostic keys from epoch logs.

    The training code registers many optional losses and therefore emits many
    zero-valued diagnostics when those losses are disabled.  This keeps the
    JSONL readable while preserving active losses, core losses, GPU memory, and
    evaluation metrics.
    """
    keep_prefixes = (
        "epoch", "train/loss", "val/loss", "train/lr", "gpu/",
        "train/ar/loss_one_step", "val/ar/loss_one_step",
        "train/ar/one_step", "val/ar/one_step",
        "train/ar/bptt", "val/ar/bptt",
        "train/ar/error_proj_pseudo", "val/ar/error_proj_pseudo",
        "train/ar/error_proj_v2", "val/ar/error_proj_v2",
        "train/ar/bridge", "val/ar/bridge",
        "train/ar/resgrad", "val/ar/resgrad",
        "train/potential", "val/potential",
    )
    out = {}
    for k, v in logs.items():
        if k.startswith(keep_prefixes):
            out[k] = v
            continue
        try:
            fv = float(v)
        except Exception:
            out[k] = v
            continue
        if abs(fv) > 1e-12:
            out[k] = v
    return out


def _append_epoch_log(exp_dir: str, epoch_logs: dict, args=None) -> None:
    """Append one epoch of train/val logs, tolerating transient NFS EAGAIN.

    Some shared filesystems briefly return ``EAGAIN`` while metadata from the
    preceding validation JSON write is being committed.  Losing a multi-GPU
    training job for that recoverable condition is unnecessary, so retry only
    the explicitly transient errno values and propagate every other failure.
    """
    path = Path(exp_dir) / "train_logs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    if args is not None and bool(getattr(args, "compact_train_logs", True)):
        epoch_logs = _compact_epoch_logs(epoch_logs)
    safe = {k: _json_safe_float(v) for k, v in epoch_logs.items()}
    payload = json.dumps(safe, sort_keys=True) + "\n"
    transient = {errno.EAGAIN, errno.EWOULDBLOCK, errno.EBUSY}
    attempts = 10
    for attempt in range(attempts):
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(payload)
            return
        except OSError as exc:
            if exc.errno not in transient or attempt + 1 >= attempts:
                raise
            delay = min(0.1 * (2**attempt), 3.0)
            print(
                f"[train-log] transient errno={exc.errno}; retry "
                f"{attempt + 2}/{attempts} in {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)


def _truncate_epoch_log(exp_dir: str, last_epoch: int) -> None:
    """Drop log rows for epochs > last_epoch so a resumed run does not duplicate them."""
    path = Path(exp_dir) / "train_logs.jsonl"
    if not path.exists():
        return
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            if int(float(json.loads(line).get("epoch", 0))) <= last_epoch:
                kept.append(line)
        except Exception:
            kept.append(line)
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def _early_stop_state_from_log(exp_dir: str, last_epoch: int):
    """Recover (best_val, bad_epochs) from train_logs.jsonl.

    Checkpoints written before resume support existed do not carry these, and they
    cannot be inferred from the weights. Since ``best.pth`` is always the lowest
    val/loss epoch, replaying the log reproduces exactly the state the run had.
    Returns (inf, 0) if the log has no usable validation entries.
    """
    path = Path(exp_dir) / "train_logs.jsonl"
    if not path.exists():
        return float("inf"), 0
    best, best_epoch, n_evals_after_best = float("inf"), 0, 0
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        ep = int(float(r.get("epoch", 0)))
        if ep <= last_epoch and "val/loss" in r:
            rows.append((ep, float(r["val/loss"])))
    rows.sort()
    for ep, vl in rows:
        if vl < best:
            best, best_epoch, n_evals_after_best = vl, ep, 0
        else:
            n_evals_after_best += 1
    return best, n_evals_after_best


def _maybe_resume(model, optimizer, scheduler, args, exp_dir, rank):
    """Restore model/optimizer/scheduler/epoch/early-stop state. Returns (start_epoch, best, bad_epochs).

    ``--resume auto`` picks up ``exp_dir/last.pth`` if it exists; a path resumes from
    that file; anything falsy starts fresh. Every rank loads from disk, so this works
    under DDP without broadcasting.
    """
    spec = str(getattr(args, "resume", "") or "")
    if not spec:
        return 1, float("inf"), 0
    ckpt_path = os.path.join(exp_dir, "last.pth") if spec == "auto" else spec
    if not os.path.exists(ckpt_path):
        if is_rank0(rank):
            print(f"[resume] no checkpoint at {ckpt_path}; starting from scratch", flush=True)
        return 1, float("inf"), 0

    map_loc = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt_path, map_location=map_loc, weights_only=False)
    unwrap_model(model).load_state_dict(ck["model"], strict=True)
    if "optimizer" in ck:
        optimizer.load_state_dict(ck["optimizer"])
    elif is_rank0(rank):
        print("[resume] WARNING: checkpoint has no optimizer state; Adam moments restart", flush=True)
    if scheduler is not None and ck.get("scheduler") is not None:
        scheduler.load_state_dict(ck["scheduler"])

    last_epoch = int(ck.get("epoch", 0))
    if "best_val" in ck and "bad_epochs" in ck:
        best, bad_epochs = float(ck["best_val"]), int(ck["bad_epochs"])
        src = "checkpoint"
    else:
        best, bad_epochs = _early_stop_state_from_log(exp_dir, last_epoch)
        src = "train_logs.jsonl (legacy checkpoint)"

    if is_rank0(rank):
        _truncate_epoch_log(exp_dir, last_epoch)
        # A staged continuation commonly resumes an external best.pth into a
        # new output directory.  Keep that validated starting checkpoint as a
        # candidate for final selection.  Without this copy, a continuation
        # that never beats the source best writes no local best.pth and the
        # caller silently evaluates last.pth instead.  Restrict the fallback
        # to an explicitly supplied best.pth; copying an arbitrary last.pth
        # would not preserve the validation-best semantics.
        local_best = os.path.join(exp_dir, "best.pth")
        external_best = (
            spec != "auto"
            and Path(ckpt_path).name == "best.pth"
            and os.path.abspath(ckpt_path) != os.path.abspath(local_best)
        )
        if external_best and not os.path.exists(local_best):
            os.makedirs(exp_dir, exist_ok=True)
            shutil.copy2(ckpt_path, local_best)
            print(
                f"[resume] seeded destination best.pth from validated source {ckpt_path}",
                flush=True,
            )
        print(f"[resume] {ckpt_path} @ epoch {last_epoch} -> continuing at {last_epoch + 1}; "
              f"best_val={best:.6f}, bad_epochs={bad_epochs} (from {src})", flush=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return last_epoch + 1, best, bad_epochs




def _reset_cuda_peak_memory(rank: int, args) -> None:
    """Reset per-device peak CUDA memory counters before a measured region."""
    if not torch.cuda.is_available():
        return
    if not bool(getattr(args, "log_gpu_memory", True)):
        return
    try:
        torch.cuda.set_device(rank)
        torch.cuda.reset_peak_memory_stats(rank)
        torch.cuda.reset_accumulated_memory_stats(rank)
    except Exception:
        # Memory logging must never break training.
        pass


def _cuda_memory_logs(rank: int, prefix: str = "gpu") -> dict:
    """Return CUDA memory diagnostics in GB for the current rank/device.

    allocated/reserved are PyTorch caching-allocator quantities for this process.
    device_used is total device memory minus free memory and therefore includes other
    processes and the CUDA context.  On DDP, each rank logs its own process; rank 0
    writes the values to train_logs.jsonl.
    """
    if not torch.cuda.is_available():
        return {}
    try:
        torch.cuda.set_device(rank)
        torch.cuda.synchronize(rank)
        denom = 1024.0 ** 3
        logs = {
            f"{prefix}/rank": float(rank),
            f"{prefix}/peak_allocated_gb": float(torch.cuda.max_memory_allocated(rank) / denom),
            f"{prefix}/peak_reserved_gb": float(torch.cuda.max_memory_reserved(rank) / denom),
            f"{prefix}/current_allocated_gb": float(torch.cuda.memory_allocated(rank) / denom),
            f"{prefix}/current_reserved_gb": float(torch.cuda.memory_reserved(rank) / denom),
        }
        try:
            free_b, total_b = torch.cuda.mem_get_info(rank)
            logs[f"{prefix}/device_total_gb"] = float(total_b / denom)
            logs[f"{prefix}/device_free_gb"] = float(free_b / denom)
            logs[f"{prefix}/device_used_gb"] = float((total_b - free_b) / denom)
        except Exception:
            pass
        return logs
    except Exception:
        return {}

def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _plot_training_curves(exp_dir: str) -> None:
    """Write simple PNG plots for loss curves and FTG diagnostics.

    This function is intentionally best-effort.  Training should not fail just
    because matplotlib is unavailable on a server.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Logging] Skip plotting because matplotlib is unavailable: {exc}", flush=True)
        return

    log_path = Path(exp_dir) / "train_logs.jsonl"
    rows = _load_jsonl(log_path)
    if not rows:
        return

    def xy(key: str):
        xs, ys = [], []
        for r in rows:
            if key in r and r[key] is not None:
                xs.append(r.get("epoch", len(xs) + 1))
                ys.append(r[key])
        return xs, ys

    def has_key(key: str) -> bool:
        return any((key in r and r[key] is not None) for r in rows)

    # 1) Main train/validation loss curves.
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)
    plotted = False
    for key, label in [
        ("train/loss", "train total"),
        ("train/ar/loss_one_step", "train one-step"),
        ("val/loss", "val one-step"),
    ]:
        xs, ys = xy(key)
        if xs:
            ax.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.4, label=label)
            plotted = True
    if plotted:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training and validation loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(Path(exp_dir) / "loss_curves.png")
    plt.close(fig)

    # 2) CUDA memory diagnostics.
    if has_key("gpu/train/peak_allocated_gb") or has_key("gpu/train/peak_reserved_gb"):
        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)
        plotted = False
        for key, label in [
            ("gpu/train/peak_allocated_gb", "train peak allocated"),
            ("gpu/train/peak_reserved_gb", "train peak reserved"),
            ("gpu/val/peak_allocated_gb", "val peak allocated"),
            ("gpu/val/peak_reserved_gb", "val peak reserved"),
        ]:
            xs, ys = xy(key)
            if xs:
                ax.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.4, label=label)
                plotted = True
        if plotted:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("GPU memory (GB)")
            ax.set_title("CUDA memory diagnostics")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(Path(exp_dir) / "gpu_memory_curves.png")
        plt.close(fig)

    # 3) FTG weighted loss and percentage of one-step loss.
    if has_key("train/ar/ftg_amp_weighted_loss") or has_key("train/ar/ftg_amp_pct_of_one_step"):
        fig, ax1 = plt.subplots(figsize=(7.5, 4.5), dpi=160)
        xs, ys = xy("train/ar/ftg_amp_weighted_loss")
        if xs:
            ax1.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.4, label="FTG weighted loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Weighted FTG loss")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        xs, ys = xy("train/ar/ftg_amp_pct_of_one_step")
        if xs:
            ax2.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.4, linestyle="--", label="FTG % of one-step")
        ax2.set_ylabel("FTG percentage of one-step loss (%)")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        if lines1 or lines2:
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
        ax1.set_title("FTG loss scale")
        fig.tight_layout()
        fig.savefig(Path(exp_dir) / "ftg_loss_scale.png")
        plt.close(fig)

    # 3) FTG tangent-ratio diagnostics.
    ratio_keys = [
        ("train/ar/ftg_amp_ratio", "mean ratio"),
        ("train/ar/ftg_ratio_k1", "k=1"),
        ("train/ar/ftg_ratio_k2", "k=2"),
        ("train/ar/ftg_ratio_klast", "last"),
    ]
    if any(has_key(k) for k, _ in ratio_keys):
        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)
        plotted = False
        for key, label in ratio_keys:
            xs, ys = xy(key)
            if xs:
                ax.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.4, label=label)
                plotted = True
        if plotted:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Amplification ratio")
            ax.set_title("FTG tangent amplification diagnostics")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(Path(exp_dir) / "ftg_ratio_curves.png")
        plt.close(fig)

def _etm_stage_name_for_trainer(args, epoch: int | float = 0) -> str:
    start = int(getattr(args, "etm_start_epoch", 0))
    j_epochs = max(0, int(getattr(args, "etm_j_only_epochs", 0)))
    ep = float(epoch or 0)
    if ep < start:
        return "warmup"
    if j_epochs > 0 and ep < start + j_epochs:
        return "j_only"
    return "joint"


def _is_etm_parameter_name(name: str) -> bool:
    # Current AR transport modules are named etm_head / etm_future_proj and
    # related buffers/parameters contain the substring "etm_".
    return "etm_" in str(name)


def _apply_etm_stage_freezing(model, args, epoch: int | float = 0) -> tuple[str, int, int]:
    """Freeze/unfreeze parameters for staged ETM training.

    warmup: ordinary AR training; optionally freeze ETM head.
    j_only: freeze non-ETM parameters, train only ETM head with fit loss.
    joint:  unfreeze everything for joint fine-tuning.

    The optimizer is created once over all parameters.  Toggling requires_grad
    per epoch is enough: frozen parameters receive no gradients, and when they
    are unfrozen they are still present in the optimizer.
    """
    stage = _etm_stage_name_for_trainer(args, epoch)
    freeze_j_only = bool(getattr(args, "etm_freeze_backbone_during_j_only", True))
    freeze_etm_before_start = bool(getattr(args, "etm_freeze_etm_before_start", True))

    trainable = 0
    frozen = 0
    for name, p in model.named_parameters():
        is_etm = _is_etm_parameter_name(name)
        if stage == "j_only" and freeze_j_only:
            req = is_etm
        elif stage == "warmup" and freeze_etm_before_start:
            req = not is_etm
        else:
            req = True
        p.requires_grad_(req)
        if req:
            trainable += int(p.numel())
        else:
            frozen += int(p.numel())
    return stage, trainable, frozen


def _sample_resgrad_ratios(raw, state, stim, burn, K, num_starts):
    """No-grad closed-loop rollout that only samples per-block dynamic-ratio values.

    Mirrors the burn-in + K-step closed-loop structure used by
    compute_recurrent_state_bptt_loss, but skips losses and target lookups
    entirely. The forward value of a residual block is identical regardless
    of its gradient-routing gate, so sampling under no_grad does not bias
    the ratio distribution relative to what the real (grad-tracked) rollout
    would see.
    """
    B, T = state.shape[0], state.shape[1]
    max_start = T - K
    if max_start < 1:
        return []
    min_start = burn if max_start >= burn else 1
    starts = list(range(min_start, max_start + 1))
    if not starts:
        return []
    chosen = random.sample(starts, min(num_starts, len(starts)))

    ratios: list = []
    with torch.no_grad():
        for start_t in chosen:
            burn_start = max(0, start_t - burn)
            burn_end = max(burn_start, start_t - 1)
            h = raw.init_state(B, state.device, state.dtype)
            for j in range(burn_start, burn_end):
                stim_j = stim[:, j] if stim is not None else None
                _, h = raw.step(h, state[:, j], stim_j, return_aux=False)
            x_in = state[:, start_t - 1]
            for k in range(K):
                target_t = start_t + k
                stim_in = stim[:, target_t - 1] if stim is not None else None
                pred, h, _aux = raw.step(
                    h, x_in, stim_in, return_aux=True,
                    horizon_index=k, total_horizon=K,
                    ratio_collector=ratios,
                )
                x_in = pred
    return ratios


def _sample_resgrad_ratios_windowed(raw, state, stim, window, K, num_starts):
    """No-grad closed-loop rollout for windowed-history AR models (e.g. unet_field).

    Mirrors the K-step closed-loop structure used by
    compute_full_bptt_rollout_loss (ar_losses.py), but skips losses/targets
    entirely and only samples per-block dynamic-ratio values via
    ratio_collector. Forward values do not depend on the gradient-routing
    gate, so sampling under no_grad does not bias the ratio distribution
    relative to what the real (grad-tracked) rollout would see.
    """
    B, T = state.shape[0], state.shape[1]
    max_start = T - K
    if max_start < window:
        return []
    starts = list(range(window, max_start + 1))
    if not starts:
        return []
    chosen = random.sample(starts, min(num_starts, len(starts)))

    ratios: list = []
    with torch.no_grad():
        for start_t in chosen:
            history = time_window(state, start_t - window, start_t)
            for k in range(K):
                target_t = start_t + k
                stim_window = time_window(stim, target_t - window, target_t) if stim is not None else None
                pred = raw(
                    stim_window, history, return_aux=False,
                    horizon_index=k, total_horizon=K,
                    ratio_collector=ratios,
                )
                history = torch.cat([history[:, 1:], pred], dim=1)
    return ratios


def calibrate_resgrad_ratio_threshold(model, state, stim, args, rank=0):
    """Auto-calibrate --resgrad_ratio_threshold to a target branch-open fraction.

    Dynamic-ratio routing gates each residual branch by comparing its
    branch/residual norm ratio against a fixed threshold. That ratio's scale
    is set by the model's *current* weights, so it drifts across seeds and
    can drift quickly within a single epoch -- early training (especially
    epoch 1, right after random init) can shift the ratio distribution
    batch to batch, so calibrating once per epoch can go stale before the
    epoch is over and open far more branches than intended, risking OOM.
    This is meant to be called every batch (or every few batches) using
    that batch's own (state, stim), so the threshold tracks the model's
    current ratio distribution closely instead of a possibly-stale
    once-per-epoch snapshot.
    """
    raw = unwrap_model(model)
    target = float(getattr(args, "resgrad_target_open_frac", 0.0))
    if target <= 0.0 or not bool(getattr(raw, "resgrad_routing", False)):
        return None
    num_starts = max(1, int(getattr(args, "resgrad_calib_num_starts", 4)))

    if bool(getattr(raw, "is_recurrent_state_ar", False)):
        burn = max(0, int(getattr(args, "mamba_burnin", 64)))
        K = max(1, int(getattr(args, "mamba_bptt_horizon", 8)))
        ratios = _sample_resgrad_ratios(raw, state, stim, burn, K, num_starts)
    else:
        window = max(1, int(getattr(args, "window_size", 4)))
        K = max(1, int(getattr(args, "bptt_horizon", 8)))
        ratios = _sample_resgrad_ratios_windowed(raw, state, stim, window, K, num_starts)
    if not ratios:
        torch.cuda.empty_cache()
        return None
    q = max(0.0, min(1.0, 1.0 - target))
    threshold = float(torch.quantile(torch.tensor(ratios), q))
    # Optional EMA smoothing of the calibrated threshold (--resgrad_calib_ema,
    # default 0.0 = off, preserving the original per-batch behavior). At
    # intermediate targets (e.g. 0.5) the quantile sits at the *median* of the
    # ratio distribution, where per-batch sampling noise in the threshold flips
    # the largest number of gates batch-to-batch ("selection churn"); smoothing
    # the threshold removes the calibration-noise component of that churn while
    # still tracking slow drift of the ratio distribution.
    ema = float(getattr(args, "resgrad_calib_ema", 0.0))
    raw_threshold = threshold
    if ema > 0.0:
        prev = getattr(raw, "_resgrad_calib_ema_prev", None)
        if prev is not None:
            threshold = ema * float(prev) + (1.0 - ema) * threshold
        raw._resgrad_calib_ema_prev = threshold
    raw.resgrad_ratio_threshold = threshold
    if is_rank0(rank):
        ema_note = f" (raw={raw_threshold:.4f}, ema={ema:.2f})" if ema > 0.0 else ""
        print(
            f"[ResGrad calib] target_open_frac={target:.3f} -> threshold={threshold:.4f} (n={len(ratios)}){ema_note}",
            flush=True,
        )
    # The no-grad calibration rollout and the grad-tracked training rollout
    # that immediately follows have very different memory shapes (no
    # activations to retain vs. a full backward graph). Without an explicit
    # empty_cache(), PyTorch's caching allocator can leave calibration's
    # blocks reserved in a layout that doesn't get reused cleanly by the
    # training pass, inflating this batch's reported (and real) peak memory.
    torch.cuda.empty_cache()
    return threshold


def _apply_resgrad_schedule(args, epoch, rank=0):
    """Schedule the ResGrad open fraction (and, optionally, gradient
    checkpointing) as a function of epoch, by mutating ``args`` in place.

    Recipe motivated by the short-path/redundancy finding: hold a CHEAP low
    open fraction for most of training (the routed gradient is direction-
    preserving, so it reaches the right neighborhood cheaply), then ramp up to
    (near-)dense for the final epochs to polish under an (almost-)exact
    gradient. Crucially, gradient checkpointing is turned on for that high-g
    tail so peak memory never pays dense BPTT's stored-activation cost -- the
    whole run stays at the checkpointing memory floor while still recovering
    dense-level accuracy.

    Disabled (returns without touching args) unless
    ``--resgrad_sched_open_start >= 0``. When active it overrides the static
    ``--resgrad_target_open_frac`` / ``--recurrent_grad_checkpoint`` per epoch.
    """
    start = float(getattr(args, "resgrad_sched_open_start", -1.0))
    if start < 0.0:
        return  # no schedule; leave static args untouched
    end = float(getattr(args, "resgrad_sched_open_end", 1.0))
    e0 = int(getattr(args, "resgrad_sched_ramp_start_epoch", 0)) or 1
    e1 = int(getattr(args, "resgrad_sched_ramp_end_epoch", 0)) or int(getattr(args, "num_epochs", 50))
    if epoch <= e0:
        g = start
    elif epoch >= e1:
        g = end
    else:
        frac = (epoch - e0) / max(1, (e1 - e0))
        g = start + frac * (end - start)
    args.resgrad_target_open_frac = float(g)
    ckpt_from = int(getattr(args, "resgrad_sched_ckpt_from_epoch", -1))
    if ckpt_from >= 0:
        args.recurrent_grad_checkpoint = bool(epoch >= ckpt_from)
    if is_rank0(rank):
        print(
            f"[ResGrad schedule] epoch {epoch}: target_open_frac={g:.3f} "
            f"checkpoint={bool(getattr(args, 'recurrent_grad_checkpoint', False))}",
            flush=True,
        )


def train_model(model, train_loader, val_loader, args, rank=0, exp_dir="experiments"):
    raw0 = unwrap_model(model)
    params = [p for p in model.parameters() if p.requires_grad]
    if getattr(raw0, "is_potential_ae", False):
        optimizer = torch.optim.AdamW(params, lr=args.base_lr, weight_decay=args.weight_decay)
    elif getattr(raw0, "is_standard_autoregressive", False):
        optimizer_name = str(getattr(args, "ar_optimizer", "adam")).lower()
        if optimizer_name == "sgd":
            momentum = float(getattr(args, "ar_sgd_momentum", 0.0))
            nesterov = bool(getattr(args, "ar_sgd_nesterov", False))
            if nesterov and momentum <= 0.0:
                raise ValueError("--ar_sgd_nesterov requires --ar_sgd_momentum > 0")
            optimizer = torch.optim.SGD(
                params,
                lr=args.base_lr,
                momentum=momentum,
                weight_decay=args.weight_decay,
                nesterov=nesterov,
            )
        elif optimizer_name == "adam":
            # Original FNO training uses Adam rather than AdamW.
            optimizer = torch.optim.Adam(
                params, lr=args.base_lr, weight_decay=args.weight_decay
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                params, lr=args.base_lr, weight_decay=args.weight_decay
            )
        else:
            raise ValueError(f"unknown autoregressive optimizer {optimizer_name!r}")
    else:
        optimizer = torch.optim.AdamW(params, lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = None
    scheduler_name = str(getattr(args, "ar_scheduler", "step"))
    if getattr(raw0, "is_standard_autoregressive", False):
        if scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(getattr(args, "ar_step_size", 100)),
                gamma=float(getattr(args, "ar_gamma", 0.5)),
            )
        elif scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, int(getattr(args, "num_epochs", 1))),
                eta_min=float(getattr(args, "ar_min_lr", 1e-6)),
            )
    start_epoch, best, bad_epochs = _maybe_resume(model, optimizer, scheduler, args, exp_dir, rank)
    probe_epochs = {
        int(value) for value in getattr(args, "probe_checkpoint_epochs", [])
        if int(value) >= 0
    }
    if is_rank0(rank) and start_epoch == 1 and 0 in probe_epochs:
        # Reference the initialized tangent dynamics before any optimization.
        # This diagnostic snapshot never participates in best selection.
        save_checkpoint(
            os.path.join(exp_dir, "probe_epoch_0000.pth"),
            model, optimizer, 0, {}, args,
            extra={
                "best_val": float(best),
                "bad_epochs": int(bad_epochs),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "probe_checkpoint": True,
            },
        )
        print("[probe checkpoint] saved initialized model at epoch 0", flush=True)
    if start_epoch > args.num_epochs:
        if is_rank0(rank):
            print(f"[resume] checkpoint is already at epoch {start_epoch - 1} >= num_epochs="
                  f"{args.num_epochs}; nothing to train. Raise --num_epochs to continue.", flush=True)
        return

    for epoch in range(start_epoch, args.num_epochs + 1):
        _reset_cuda_peak_memory(rank, args)
        etm_stage, etm_trainable_params, etm_frozen_params = _apply_etm_stage_freezing(model, args, epoch)
        model.train()
        # Epoch-dependent ResGrad open-fraction schedule (cheap routing early,
        # near-dense + checkpointing late). Mutates args in place; no-op unless
        # --resgrad_sched_open_start >= 0.
        _apply_resgrad_schedule(args, epoch, rank=rank)
        calibrated_resgrad_threshold = None
        resgrad_target_open_frac = float(getattr(args, "resgrad_target_open_frac", 0.0))
        calib_every_batches = max(1, int(getattr(args, "resgrad_calib_every_batches", 1)))
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        logs_sum = {}
        num_batches = 0
        num_examples_local = 0
        num_optimizer_steps = 0
        # Gradient accumulation: step the optimizer once every grad_accum
        # micro-batches, so the effective batch = grad_accum x local_batch_size.
        # This lets a single-GPU run match the effective batch of a multi-GPU
        # (DDP) run without changing per-GPU memory. grad_accum=1 (default) is
        # identical to the previous per-batch behavior. Not applied to the
        # manual-backward CTO path (which does its own scaled backward).
        grad_accum = max(1, int(getattr(args, "grad_accum_steps", 1)))
        batch_global_controller = getattr(
            unwrap_model(model), "global_horizon_wiener", None
        )
        if (
            batch_global_controller is not None
            and bool(getattr(batch_global_controller, "batch_conditioned", False))
            and grad_accum != 1
        ):
            raise ValueError(
                "batch-conditioned global-horizon Wiener defines one joint "
                "solve over the optimizer batch and therefore requires "
                "--grad_accum_steps 1; increase local/DDP batch size instead"
            )
        n_batches_total = len(train_loader)
        fast_train_logging = bool(getattr(args, "fast_train_logging", False))
        synchronize_epoch_timing = bool(
            getattr(args, "synchronize_epoch_timing", False)
        )
        if synchronize_epoch_timing and torch.cuda.is_available():
            torch.cuda.synchronize(rank)
        epoch_start_time = time.perf_counter()
        for batch_idx, batch in enumerate(train_loader):
            # DDP normally all-reduces every backward call.  During gradient
            # accumulation that communication is redundant: suppress it for
            # intermediate micro-batches and reduce only the completed update.
            # Set the flag before forward, matching DistributedDataParallel's
            # no_sync() contract without indenting the full loss dispatch below.
            sync_gradients = (
                ((batch_idx + 1) % grad_accum == 0)
                or (batch_idx + 1 == n_batches_total)
            )
            defer_ddp_sync = (
                grad_accum > 1
                and hasattr(model, "require_backward_grad_sync")
                and not (
                    bool(getattr(args, "cto_cut_loss", False))
                    and bool(getattr(args, "cto_cut_sequential_backward", True))
                )
            )
            if defer_ddp_sync:
                model.require_backward_grad_sync = sync_gradients
            state, stim, _, _ = unpack_batch(batch)
            state = state.cuda(rank, non_blocking=True).float()
            num_examples_local += int(state.shape[0])
            if stim is None:
                stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))
            stim = stim.cuda(rank, non_blocking=True).float()
            if resgrad_target_open_frac > 0.0 and batch_idx % calib_every_batches == 0:
                calibrated_resgrad_threshold = calibrate_resgrad_ratio_threshold(model, state, stim, args, rank=rank) or calibrated_resgrad_threshold
            if batch_idx % grad_accum == 0:
                optimizer.zero_grad(set_to_none=True)
            raw = unwrap_model(model)
            dual_wiener_active = bool(
                getattr(raw, "dual_wiener", None) is not None
                and str(getattr(raw, "resgrad_policy", "")).lower() == "dualwiener"
            )
            global_wiener_active = bool(
                getattr(raw, "global_horizon_wiener", None) is not None
            )
            if dual_wiener_active:
                raw.dual_wiener_begin_batch()
            if global_wiener_active:
                raw.global_wiener_begin_batch()
            used_manual_backward = False
            if getattr(raw, "is_potential_ae", False):
                loss, logs = compute_potential_ae_loss(model, state, stim, args, epoch=epoch)
            elif getattr(raw, "is_standard_autoregressive", False):
                if (
                    bool(getattr(raw, "is_recurrent_state_ar", False))
                    and bool(getattr(args, "cto_cut_loss", False))
                    and bool(getattr(args, "cto_cut_replace_base", True))
                    and bool(getattr(args, "cto_cut_sequential_backward", True))
                    and raw.training
                ):
                    # Memory-correct dataset-cut CTO: accumulate gradients from
                    # A, B, and order-cut losses one graph at a time.  This
                    # function already calls backward(), so do not call
                    # loss.backward() again here.
                    loss, logs = backward_recurrent_state_cto_cut_loss(
                        model, state, stim, args, epoch=epoch
                    )
                    used_manual_backward = True
                else:
                    loss, logs = compute_autoregressive_one_step_loss(model, state, stim, args, epoch=epoch)
            else:
                loss, logs = compute_koopman_gramian_loss(model, state, stim, args, epoch=epoch)
            if not used_manual_backward:
                if dual_wiener_active:
                    # Two read-only VJP probes update the lagged 2x2 route
                    # covariance. autograd.grad returns only root-input VJPs and
                    # does not add the random probe to parameter .grad fields.
                    raw.dual_wiener_calibrate()
                if global_wiener_active:
                    # K complete horizon gradients and M independent matched
                    # innovation-score gradient sets define the joint global
                    # Wiener problem.  DDP reduces gradient features before the
                    # Gram is formed, and the custom backward-only loss node
                    # applies this same batch's solved coefficients below.
                    # DDP's reducer must not treat these read-only autograd.grad
                    # traversals as optimizer backpropagations.  The Gram is
                    # synchronized explicitly inside the controller.
                    if hasattr(model, "no_sync"):
                        with model.no_sync():
                            raw.global_wiener_calibrate(params)
                    else:
                        raw.global_wiener_calibrate(params)
                # Scale so the accumulated gradient equals the MEAN over the
                # effective (grad_accum x local) batch, matching a single large batch.
                (loss / grad_accum if grad_accum > 1 else loss).backward()
                if defer_ddp_sync:
                    # Keep DDP in its normal state outside this micro-batch.
                    model.require_backward_grad_sync = True
                if dual_wiener_active:
                    # Commit covariance and coefficients only after the real
                    # backward, so each batch uses gains estimated previously.
                    raw.dual_wiener_end_batch()
                if global_wiener_active:
                    raw.global_wiener_end_batch()
            elif dual_wiener_active:
                # Manual-backward objectives do not currently build the matched
                # calibration probes, but the controller still needs its batch
                # lifecycle closed so lagged residual state cannot leak forward.
                raw.dual_wiener_end_batch()
            elif global_wiener_active:
                raw.global_wiener_end_batch()
            # Strict live-gradient-frontier BPTT is a manual pruned reverse pass.
            # It is not part of the returned scalar loss; it directly accumulates
            # extra gradients into model.parameters() using exact local VJPs.
            if getattr(raw, "is_standard_autoregressive", False) and bool(getattr(args, "frontier_graph_loss", False)):
                frontier_logs = strict_live_frontier_backward(model, state, stim, args, epoch=epoch)
                logs.update(frontier_logs)
            # Step once per accumulation window (and always on the last batch,
            # so a trailing partial window is not dropped).
            if ((batch_idx + 1) % grad_accum == 0) or (batch_idx + 1 == n_batches_total):
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                num_optimizer_steps += 1
            for k, v in logs.items():
                if fast_train_logging and torch.is_tensor(v):
                    value = v.detach()
                    logs_sum[k] = value if k not in logs_sum else logs_sum[k] + value
                else:
                    logs_sum[k] = logs_sum.get(k, 0.0) + float(v)
            num_batches += 1

        train_logs = {}
        for k, v in logs_sum.items():
            mean_value = v / max(num_batches, 1)
            if torch.is_tensor(mean_value):
                # A single synchronization per retained epoch metric replaces
                # one synchronization per metric per minibatch.
                mean_value = float(mean_value.detach().cpu())
            train_logs[f"train/{k}"] = mean_value
        if synchronize_epoch_timing and torch.cuda.is_available():
            torch.cuda.synchronize(rank)
        epoch_wall_seconds = time.perf_counter() - epoch_start_time
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        num_examples_global = num_examples_local * world_size
        train_logs["train/epoch_wall_seconds"] = epoch_wall_seconds
        train_logs["train/num_batches"] = float(num_batches)
        train_logs["train/optimizer_steps"] = float(num_optimizer_steps)
        train_logs["train/examples"] = float(num_examples_global)
        train_logs["train/seconds_per_optimizer_step"] = (
            epoch_wall_seconds / max(num_optimizer_steps, 1)
        )
        train_logs["train/examples_per_second"] = (
            num_examples_global / max(epoch_wall_seconds, 1e-12)
        )
        train_logs["train/ar/etm_stage_is_warmup"] = 1.0 if etm_stage == "warmup" else 0.0
        train_logs["train/ar/etm_stage_is_j_only"] = 1.0 if etm_stage == "j_only" else 0.0
        train_logs["train/ar/etm_stage_is_joint"] = 1.0 if etm_stage == "joint" else 0.0
        train_logs["train/ar/etm_trainable_params"] = float(etm_trainable_params)
        train_logs["train/ar/etm_frozen_params"] = float(etm_frozen_params)
        if calibrated_resgrad_threshold is not None:
            train_logs["train/ar/resgrad_calibrated_threshold"] = float(calibrated_resgrad_threshold)
        if bool(getattr(args, "log_gpu_memory", True)):
            train_logs.update(_cuda_memory_logs(rank, prefix="gpu/train"))
        if scheduler is not None:
            scheduler.step()
            train_logs["train/lr"] = float(optimizer.param_groups[0]["lr"])
        if is_rank0(rank):
            msg = f"Epoch {epoch:03d} | train/loss={train_logs.get('train/loss', 0.0):.6f}"
            print(msg, flush=True)

        epoch_logs = {"epoch": float(epoch), **train_logs}

        if epoch % args.eval_every == 0:
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            _reset_cuda_peak_memory(rank, args)
            val_logs = evaluate_loss(model, val_loader, args, rank=rank, prefix="val")
            if bool(getattr(args, "log_gpu_memory", True)):
                val_logs.update(_cuda_memory_logs(rank, prefix="gpu/val"))
            val_loss = val_logs["val/loss"]
            epoch_logs.update(val_logs)
            if is_rank0(rank):
                print(f"Epoch {epoch:03d} | val/loss={val_loss:.6f}", flush=True)
                save_eval_json(os.path.join(exp_dir, "last_val_logs.json"), {**train_logs, **val_logs})
                _append_epoch_log(exp_dir, epoch_logs, args=args)
                _plot_training_curves(exp_dir)
                # Compute the post-epoch early-stop state first, so last.pth carries
                # exactly the state a resume needs to continue as if uninterrupted.
                if val_loss < best:
                    best, bad_epochs = val_loss, 0
                    new_best = True
                else:
                    bad_epochs += 1
                    new_best = False
                resume_state = {
                    "best_val": float(best),
                    "bad_epochs": int(bad_epochs),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                }
                save_checkpoint(os.path.join(exp_dir, "last.pth"), model, optimizer, epoch, val_logs, args,
                                extra=resume_state)
                if epoch in probe_epochs:
                    save_checkpoint(
                        os.path.join(exp_dir, f"probe_epoch_{epoch:04d}.pth"),
                        model, optimizer, epoch, val_logs, args,
                        extra={**resume_state, "probe_checkpoint": True},
                    )
                    print(f"[probe checkpoint] saved epoch {epoch}", flush=True)
                gate_state = None
                global_wiener_state = None
                if bool(getattr(raw, "is_recurrent_state_ar", False)):
                    export_horizon = int(getattr(args, "mamba_bptt_horizon", 1))
                else:
                    export_horizon = int(getattr(args, "bptt_horizon", 1))
                if hasattr(raw, "dual_wiener_export_state"):
                    gate_state = raw.dual_wiener_export_state(
                        export_horizon
                    )
                    if gate_state is not None:
                        save_eval_json(
                            os.path.join(exp_dir, "dual_wiener_gains_last.json"),
                            gate_state,
                        )
                if hasattr(raw, "global_wiener_export_state"):
                    global_wiener_state = raw.global_wiener_export_state(
                        export_horizon
                    )
                    if global_wiener_state is not None:
                        save_eval_json(
                            os.path.join(
                                exp_dir, "global_horizon_wiener_last.json"
                            ),
                            global_wiener_state,
                        )
                if new_best:
                    save_checkpoint(os.path.join(exp_dir, "best.pth"), model, optimizer, epoch, val_logs, args,
                                    extra=resume_state)
                    if gate_state is not None:
                        save_eval_json(
                            os.path.join(exp_dir, "dual_wiener_gains.json"),
                            gate_state,
                        )
                    if global_wiener_state is not None:
                        save_eval_json(
                            os.path.join(exp_dir, "global_horizon_wiener.json"),
                            global_wiener_state,
                        )
            # Make sure rank 0 has finished writing checkpoints before other ranks continue.
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
                # Only rank 0 updates best/bad_epochs (val_loader is sharded, so each
                # rank sees a different val_loss). Without broadcasting, the other ranks
                # keep bad_epochs=0, never take the early-stop branch, and hang at the
                # next barrier once rank 0 breaks out of the loop.
                _bad = torch.tensor([bad_epochs], dtype=torch.long,
                                    device=(f"cuda:{rank}" if torch.cuda.is_available() else "cpu"))
                dist.broadcast(_bad, src=0)
                bad_epochs = int(_bad.item())
            if args.early_stop_patience > 0 and bad_epochs >= args.early_stop_patience:
                if is_rank0(rank):
                    print(f"Early stopping after {bad_epochs} bad evals.")
                break
        else:
            if is_rank0(rank):
                _append_epoch_log(exp_dir, epoch_logs, args=args)

    if is_rank0(rank):
        _plot_training_curves(exp_dir)
        print(f"Best val/loss={best:.6f}")
