"""Standard autoregressive one-step losses for non-Koopman baselines."""

from __future__ import annotations

import random
import math
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint

from internal_dw.data_utils.state_ops import (
    corrcoef_flat,
    elementwise_state_loss,
    get_batch_time_shape,
    time_point,
    time_window,
    append_time_point,
    zero_external_input_like,
)
from internal_dw.evaluation.metrics import relative_l2, trajectory_relative_l2
from internal_dw.utils import unwrap_model
from internal_dw.training.potential_stageb import stageb_residualize_state


def _detach_recurrent_state(h):
    """Detach tensor/tuple/list recurrent states recursively.

    Official-state Mamba uses nested tuples of (conv_state, ssm_state), while
    older prototypes used a single tensor.
    """
    if torch.is_tensor(h):
        return h.detach()
    if isinstance(h, tuple):
        return tuple(_detach_recurrent_state(x) for x in h)
    if isinstance(h, list):
        return [_detach_recurrent_state(x) for x in h]
    if isinstance(h, dict):
        return {k: _detach_recurrent_state(v) for k, v in h.items()}
    return h


def _forward_jacobian_fd_penalty(
    x: torch.Tensor,
    forward_fn,
    *,
    eps: float,
    target: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random-direction finite-difference penalty on forward expansion.

    This is a forward-Jacobian baseline, not a backward routing operator. For
    one Rademacher direction per sample it estimates ``J(x)v`` with two
    ordinary forwards and penalizes only expansion above ``target``. The
    perturbation is scaled by each sample's RMS so one epsilon works for both
    normalized vectors and spatial fields. Finite differences avoid
    second-order autograd and retain gradients with respect to model parameters.
    """
    if eps <= 0.0:
        raise ValueError(f"forward_jacobian_eps must be positive, got {eps}")
    if target < 0.0:
        raise ValueError(
            f"forward_jacobian_target must be non-negative, got {target}"
        )

    v = torch.empty_like(x).bernoulli_(0.5).mul_(2.0).sub_(1.0)
    reduce_dims = tuple(range(1, x.dim()))
    x_scale = x.detach().float().pow(2).mean(dim=reduce_dims, keepdim=True).sqrt()
    x_scale = x_scale.clamp_min(1e-3).to(dtype=x.dtype)
    delta = float(eps) * x_scale * v

    y_plus = forward_fn(x + delta)
    y_minus = forward_fn(x - delta)
    if isinstance(y_plus, (tuple, list)):
        y_plus = y_plus[0]
    if isinstance(y_minus, (tuple, list)):
        y_minus = y_minus[0]

    out_scale = x_scale.reshape(x_scale.shape[0], *([1] * (y_plus.dim() - 1)))
    jv = (y_plus - y_minus) / (2.0 * float(eps) * out_scale)
    jv_rms = jv.float().reshape(jv.shape[0], -1).pow(2).mean(dim=1).sqrt()
    penalty = torch.relu(jv_rms - float(target)).pow(2).mean()
    return penalty, jv_rms.detach().mean()


def _reduce_metric_scalars(values, reduction: str = "mean", default: float = 0.0) -> float:
    """Reduce detached diagnostic scalars with at most one device sync.

    Recurrent rollouts may emit thousands of scalar diagnostics per minibatch.
    Calling ``.cpu()`` when each scalar is produced serializes the CUDA stream
    thousands of times.  Keep them on-device and transfer only the final
    reduction used for logging.  This helper never participates in the loss or
    gradient graph.
    """
    if not values:
        return float(default)
    tensor_value = next((value for value in values if torch.is_tensor(value)), None)
    if tensor_value is None:
        if reduction == "min":
            return float(min(values))
        if reduction == "max":
            return float(max(values))
        return float(sum(values) / len(values))
    device = tensor_value.device
    scalars = [
        value.detach().float().mean()
        if torch.is_tensor(value)
        else torch.tensor(float(value), device=device, dtype=torch.float32)
        for value in values
    ]
    stacked = torch.stack(scalars)
    if reduction == "min":
        result = stacked.min()
    elif reduction == "max":
        result = stacked.max()
    elif reduction == "mean":
        result = stacked.mean()
    else:
        raise ValueError(f"Unknown metric reduction {reduction!r}")
    return float(result.cpu())


def _candidate_starts(T: int, W: int, stride: int) -> list[int]:
    # t means the target index; history is [t-W, t), target is t.
    return list(range(W, T, max(int(stride), 1)))


def _shared_global_wiener_start(
    starts: list[int],
    *,
    randomize: bool,
    epoch: int,
    device: torch.device,
) -> int:
    """Choose one rollout start for a complete matched DDP microbatch.

    A batch-conditioned horizon Gram is defined from the mean gradient over
    every sample participating in the DDP update.  Choosing one start per rank
    would mix different relative horizons before forming that mean.  Rank zero
    therefore chooses ``t0`` and broadcasts it to all ranks.
    """

    if not starts:
        raise ValueError("shared-start training requires at least one rollout start")
    distributed = bool(
        dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    )
    rank = dist.get_rank() if distributed else 0
    if rank == 0:
        if randomize:
            selected = random.choice(starts)
        else:
            selected = starts[max(int(epoch) - 1, 0) % len(starts)]
    else:
        selected = starts[0]
    if distributed:
        encoded = torch.tensor([selected], device=device, dtype=torch.long)
        dist.broadcast(encoded, src=0)
        selected = int(encoded.item())
    if selected not in starts:
        raise RuntimeError(
            f"broadcast rollout start {selected} is invalid for local starts "
            f"[{starts[0]}, {starts[-1]}]"
        )
    return int(selected)




def _apply_state_sequence_ignore_mask(target: torch.Tensor, args) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero out random time slots for AE-stage mask-awareness training.

    This is an *ignore-mask* pretraining objective, not inpainting.  The encoder
    sees zeroed time slots, while the reconstruction loss is split into:

      1) visible-frame reconstruction: reconstruct the unmasked physical frames;
      2) masked-frame zero penalty: keep masked slots near zero.

    Keeping the original target separate is important.  If we simply train on
    target * mask with a single trajectory relative-L2, the trivial near-zero
    reconstruction can dominate and the AE collapses toward the all-zero
    baseline.
    """
    p = float(getattr(args, "statetok_ae_mask_prob", 0.0))
    B, T = target.shape[:2]
    if p <= 0.0:
        time_mask = target.new_ones(B, T)
        return target, time_mask

    keep = (torch.rand(B, T, device=target.device) > p).float()
    min_keep = max(1, min(int(getattr(args, "statetok_ae_mask_min_keep", 1)), T))
    # Guarantee at least min_keep visible slots per sample.
    for b in range(B):
        if int(keep[b].sum().item()) < min_keep:
            idx = torch.randperm(T, device=target.device)[:min_keep]
            keep[b].zero_()
            keep[b, idx] = 1.0

    view_shape = [B, T] + [1] * (target.dim() - 2)
    masked_input = target * keep.view(*view_shape)
    return masked_input, keep


def _masked_ignore_ae_rel_l2(
    recon: torch.Tensor,
    target: torch.Tensor,
    keep: torch.Tensor,
    args,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Relative-L2 ignore-mask AE loss.

    Visible frames are reconstructed against the true target.  Masked frames are
    only softly encouraged to be zero, with weight
    --statetok_ae_mask_zero_weight.  This teaches the zero-mask meaning without
    letting zero targets dominate the AE objective.
    """
    B, T = target.shape[:2]
    view_shape = [B, T] + [1] * (target.dim() - 2)
    visible = keep.view(*view_shape)
    hidden = 1.0 - visible
    eps = float(getattr(args, "rel_l2_eps", 1e-8))
    beta = float(getattr(args, "statetok_ae_mask_zero_weight", 0.1))

    denom = (visible * target).reshape(B, -1).norm(dim=1).clamp_min(eps)
    visible_loss = ((visible * (recon - target)).reshape(B, -1).norm(dim=1) / denom).mean()
    zero_loss = ((hidden * recon).reshape(B, -1).norm(dim=1) / denom).mean()
    loss = visible_loss + beta * zero_loss
    return loss, visible_loss.detach(), zero_loss.detach()



def _mean_sq_per_sample(x: torch.Tensor) -> torch.Tensor:
    """Per-sample squared Euclidean norm.

    Returns a tensor with shape [B].  Keeping this per-sample avoids hidden
    broadcasting bugs when computing finite-time amplification ratios.
    Call .mean() only at the final scalar logging/loss stage.
    """
    return x.reshape(x.shape[0], -1).pow(2).sum(dim=1)


def _history_jvp_step(raw, history: torch.Tensor, tangent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one AR history transition and its JVP wrt the history input.

    This computes J(history) @ tangent without explicitly materializing the
    enormous state-to-state Jacobian.  create_graph=False makes this a detached
    finite-time sensitivity estimate; it is used only to weight the one-step
    error, not to backpropagate through the Jacobian product.

    strict=True is intentional: if step_history() accidentally detaches its
    input or does not depend on the history, we want an explicit error rather
    than a silent zero JVP.
    """
    h = history.detach().requires_grad_(True)
    v = tangent.detach()

    def fn(inp):
        return raw.step_history(inp)

    with torch.enable_grad():
        y, jv = torch.autograd.functional.jvp(
            fn,
            (h,),
            (v,),
            create_graph=False,
            strict=True,
        )

    # fn returns a Tensor, but keep this guard in case a model wrapper returns
    # a single-element tuple/list.
    if isinstance(y, (tuple, list)):
        y = y[0]
    if isinstance(jv, (tuple, list)):
        jv = jv[0]

    return y.detach(), jv.detach()




def _history_jvp_step_with_stim_graph(
    raw,
    history: torch.Tensor,
    tangent: torch.Tensor,
    stim_window: torch.Tensor | None,
    *,
    create_graph: bool = True,
    strict: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one AR history transition and its JVP with optional stimulus.

    This is the transport primitive for propagated-error-source regularization.
    The base history is detached, so the Jacobian is evaluated at a fixed
    rollout/teacher-forced state.  The tangent is *not* detached when
    create_graph=True, so gradients can flow back to the local source b_i.

        next_history, J(history) @ tangent

    The returned next_history is detached because it is used only as the point
    at which the next Jacobian is evaluated.  The returned JVP keeps its graph
    when create_graph=True.
    """
    h = history.detach().requires_grad_(True)
    v = tangent

    def fn(inp):
        nxt, _pred = _ar_step_history_with_stim(raw, inp, stim_window)
        return nxt

    with torch.enable_grad():
        y, jv = torch.autograd.functional.jvp(
            fn,
            (h,),
            (v,),
            create_graph=bool(create_graph),
            strict=bool(strict),
        )
    if isinstance(y, (tuple, list)):
        y = y[0]
    if isinstance(jv, (tuple, list)):
        jv = jv[0]
    return y.detach(), jv if create_graph else jv.detach()

def _compute_fno_ftg_losses(
    raw,
    history: torch.Tensor,
    pred_frame: torch.Tensor,
    target_frame: torch.Tensor,
    one_loss: torch.Tensor,
    args,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Finite-time sensitivity losses for FNO without a training adapter.

    We approximate the first FTG differential term by a detached scalar
    finite-time amplification weight.  Starting from the one-step error e, we
    propagate e through the model's own history-to-history Jacobian using JVPs:

        v_{k+1} = J_k v_k.

    The amplification ratio is detached and used to weight the one-step loss.
    This implements the cheap first-term approximation: local errors that would
    be amplified by the model's future dynamics receive larger weight.

    The second, operator-gradient term is not differentiated exactly.  Instead
    we add a first-order finite-difference bound on the local history-transition
    amplification, which discourages overly expanding local Jacobians without
    requiring Hessian-style differentiation through J.
    """
    device = pred_frame.device
    zero = pred_frame.new_tensor(0.0)

    enabled = bool(getattr(args, "fno_ftg_loss", False))
    if not enabled:
        return zero, zero, {
            "ar/ftg_amp_ratio": 0.0,
            "ar/ftg_amp_loss": 0.0,
            "ar/ftg_bound_loss": 0.0,
            "ar/ftg_lambda_amp": 0.0,
            "ar/ftg_lambda_bound": 0.0,
            "ar/ftg_horizon": 0.0,
            "ar/ftg_sigma_hat": 0.0,
            "ar/ftg_ratio_k0": 0.0,
            "ar/ftg_ratio_k1": 0.0,
            "ar/ftg_ratio_k2": 0.0,
            "ar/ftg_ratio_klast": 0.0,
        }

    lam_amp = float(getattr(args, "fno_ftg_lambda_amp", 0.0))
    lam_bound = float(getattr(args, "fno_ftg_lambda_bound", 0.0))
    do_eval = bool(getattr(args, "fno_ftg_eval", False))
    if (not raw.training) and (not do_eval):
        return zero, zero, {
            "ar/ftg_amp_ratio": 0.0,
            "ar/ftg_amp_loss": 0.0,
            "ar/ftg_bound_loss": 0.0,
            "ar/ftg_lambda_amp": float(lam_amp),
            "ar/ftg_lambda_bound": float(lam_bound),
            "ar/ftg_horizon": 0.0,
            "ar/ftg_sigma_hat": 0.0,
            "ar/ftg_ratio_k0": 0.0,
            "ar/ftg_ratio_k1": 0.0,
            "ar/ftg_ratio_k2": 0.0,
            "ar/ftg_ratio_klast": 0.0,
        }

    K = int(getattr(args, "fno_ftg_horizon", 0))
    if K <= 0:
        K = int(getattr(args, "fno_fold_horizon", 0))
    if K <= 0:
        K = int(getattr(args, "koopman_gramian_horizon", 8))
    K = max(1, int(K))

    eps = float(getattr(args, "fno_ftg_eps", 1e-8))

    # Start from the AR history after appending the current one-step prediction.
    # The initial tangent is zero on old history frames and equals the current
    # one-step error on the newly appended frame.
    with torch.no_grad():
        hist = torch.cat([history.detach()[:, 1:], pred_frame.detach().unsqueeze(1)], dim=1)
        e = (pred_frame - target_frame).detach()
        v_hist = torch.zeros_like(hist)
        v_hist[:, -1] = e

        e_norm = _mean_sq_per_sample(e).clamp_min(eps)  # [B]
        amp_terms = [_mean_sq_per_sample(e)]             # list of [B]

    # Detached finite-time tangent rollout for the first FTG term.
    if lam_amp != 0.0:
        ratio_debug = []

        # k = 0 identity contribution.  This should be close to 1.0.
        r0 = (amp_terms[0] / e_norm).mean().detach()
        ratio_debug.append(r0)

        for _ in range(1, K):
            hist, v_hist = _history_jvp_step(raw, hist, v_hist)
            vk_norm = _mean_sq_per_sample(v_hist[:, -1])
            amp_terms.append(vk_norm)
            ratio_debug.append((vk_norm / e_norm).mean().detach())

        amp_terms_tensor = torch.stack(amp_terms, dim=0)  # [K, B]
        amp_energy = amp_terms_tensor.mean(dim=0)         # [B]
        amp_ratio_per_sample = amp_energy / e_norm        # [B]
        amp_ratio = amp_ratio_per_sample.mean().detach()  # scalar
        amp_loss = amp_ratio * one_loss

        ftg_ratio_k0 = float(ratio_debug[0].detach().cpu()) if len(ratio_debug) > 0 else 0.0
        ftg_ratio_k1 = float(ratio_debug[1].detach().cpu()) if len(ratio_debug) > 1 else 0.0
        ftg_ratio_k2 = float(ratio_debug[2].detach().cpu()) if len(ratio_debug) > 2 else 0.0
        ftg_ratio_klast = float(ratio_debug[-1].detach().cpu()) if len(ratio_debug) > 0 else 0.0
    else:
        amp_ratio = zero
        amp_loss = zero
        ftg_ratio_k0 = 0.0
        ftg_ratio_k1 = 0.0
        ftg_ratio_k2 = 0.0
        ftg_ratio_klast = 0.0

    # First-order finite-difference bound for the second/operator-gradient term.
    # This uses two normal forward passes per step and backpropagates through
    # model parameters, but avoids differentiating through a Jacobian object.
    if lam_bound != 0.0:
        rho = float(getattr(args, "fno_ftg_rho", 1.05))
        fd_eps = float(getattr(args, "fno_ftg_bound_eps", 1e-3))
        hist_fd = torch.cat([history.detach()[:, 1:], pred_frame.detach().unsqueeze(1)], dim=1)
        bound_terms = []
        sigma_logs = []
        for _ in range(K):
            noise = torch.randn_like(hist_fd)
            # Normalize noise per sample so fd_eps has comparable scale.
            noise_norm = noise.reshape(noise.shape[0], -1).norm(dim=1).view(-1, 1, 1, 1, 1).clamp_min(eps)
            noise = noise / noise_norm
            base_next = raw.step_history(hist_fd)
            pert_next = raw.step_history(hist_fd + fd_eps * noise)
            diff = (pert_next - base_next) / max(fd_eps, eps)
            sigma = diff.reshape(diff.shape[0], -1).norm(dim=1).mean()
            bound_terms.append(torch.relu(sigma - rho).pow(2))
            sigma_logs.append(float(sigma.detach().cpu()))
            hist_fd = base_next.detach()
        bound_loss = torch.stack(bound_terms).mean()
        sigma_hat = sum(sigma_logs) / max(len(sigma_logs), 1)
    else:
        bound_loss = zero
        sigma_hat = 0.0

    logs = {
        "ar/ftg_amp_ratio": float(amp_ratio.detach().cpu()),
        "ar/ftg_amp_loss": float(amp_loss.detach().cpu()),
        "ar/ftg_bound_loss": float(bound_loss.detach().cpu()),
        "ar/ftg_lambda_amp": float(lam_amp),
        "ar/ftg_lambda_bound": float(lam_bound),
        "ar/ftg_horizon": float(K),
        "ar/ftg_sigma_hat": float(sigma_hat),
        "ar/ftg_ratio_k0": float(ftg_ratio_k0),
        "ar/ftg_ratio_k1": float(ftg_ratio_k1),
        "ar/ftg_ratio_k2": float(ftg_ratio_k2),
        "ar/ftg_ratio_klast": float(ftg_ratio_klast),
    }
    return amp_loss, bound_loss, logs

def _compute_fno_folded_loss(
    raw,
    h_pred: torch.Tensor,
    state: torch.Tensor,
    target_t: int,
    args,
) -> tuple[torch.Tensor, dict]:
    """Folded latent supervision for FNO-style one-step AR models.

    target_t is the index of the one-step target x_t.
    h_pred is the observable of predicted x_t.

    We learn a small latent propagator A and compare:
        h_pred, A h_pred, A^2 h_pred, ...
    with observables of future GT fields:
        phi(x_t), phi(x_{t+1}), phi(x_{t+2}), ...

    This preserves a one-FNO-forward training graph while injecting
    long-horizon supervision.

    IMPORTANT:
      This function returns the RAW folded loss.
      Lambda weighting is applied only in compute_autoregressive_one_step_loss().
    """
    zero = h_pred.new_tensor(0.0)

    if not bool(getattr(args, "fno_folded_loss", False)):
        return zero, {
            "ar/fold_loss": 0.0,
            "ar/fold_used_horizon": 0.0,
            "ar/fold_A_sigma": 0.0,
            "ar/fold_lambda": 0.0,
            "ar/fold_weighted_loss": 0.0,
        }

    lam = float(getattr(args, "fno_fold_lambda", -1.0))
    if lam < 0:
        lam = float(getattr(args, "koopman_lambda_gram", 0.0))

    if lam == 0.0:
        return zero, {
            "ar/fold_loss": 0.0,
            "ar/fold_used_horizon": 0.0,
            "ar/fold_A_sigma": raw.folded_A_sigma(),
            "ar/fold_lambda": 0.0,
            "ar/fold_weighted_loss": 0.0,
        }

    K = int(getattr(args, "fno_fold_horizon", 0))
    if K <= 0:
        K = int(getattr(args, "koopman_gramian_horizon", 16))

    T = state.shape[1]

    # k=0 compares the one-step prediction with x_t.
    # k=K-1 compares after K-1 latent propagations.
    K_eff = max(1, min(K, T - int(target_t)))

    h = h_pred
    losses = []

    for k in range(K_eff):
        target_frame = state[:, target_t + k]

        with torch.no_grad():
            h_target = raw.encode_folded_observable(target_frame)

        losses.append(torch.mean((h - h_target) ** 2))

        if k != K_eff - 1:
            h = raw.folded_step(h)

    fold_loss = torch.stack(losses).mean()

    logs = {
        "ar/fold_loss": float(fold_loss.detach().cpu()),
        "ar/fold_used_horizon": float(K_eff),
        "ar/fold_A_sigma": raw.folded_A_sigma(),
        "ar/fold_lambda": float(lam),
        "ar/fold_weighted_loss": float((lam * fold_loss).detach().cpu()),
    }

    return fold_loss, logs




def _diag_gaussian_mmsbm_error(pred_obs: torch.Tensor, target_obs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Cheap multi-marginal distribution error in a low-dimensional observable.

    pred_obs/target_obs: [B, D].  This is a diagonal-Gaussian proxy for a
    Wasserstein/SB marginal matching term:
        ||m_pred-m_gt||^2 + ||std_pred-std_gt||^2.
    It is intentionally lightweight and stable for small batches.
    """
    mp = pred_obs.mean(dim=0)
    mt = target_obs.mean(dim=0)
    if pred_obs.shape[0] > 1:
        sp = pred_obs.var(dim=0, unbiased=False).clamp_min(eps).sqrt()
        st = target_obs.var(dim=0, unbiased=False).clamp_min(eps).sqrt()
        std_term = (sp - st).pow(2).mean()
    else:
        std_term = pred_obs.new_tensor(0.0)
    mean_term = (mp - mt).pow(2).mean()
    return mean_term + std_term


def _across_start_anomaly_w1(pred_by_k, tgt_by_k):
    """Faithful training-time version of the ``w1_anom`` diagnostic.

    For each rollout horizon ``k`` we pool predictions and targets over the
    starts+batch into ``[N, D]``, subtract the per-coordinate across-sample mean
    (the \"anomaly\", so the mean --- already handled by MSE --- is not double
    counted), and sum the 1-D sort-based 1-Wasserstein between the predicted and
    target anomaly marginals \emph{per coordinate}, matching the per-coordinate
    structure of ``var_ret``. Distribution-free (no Gaussian/second-moment
    assumption); differentiable in ``pred`` through the sort. Returns None if
    there are too few samples to form a marginal.
    """
    terms = []
    for k in sorted(pred_by_k.keys()):
        P = torch.cat([p.reshape(p.shape[0], -1) for p in pred_by_k[k]], dim=0)  # [N, D]
        T = torch.cat([t.reshape(t.shape[0], -1) for t in tgt_by_k[k]], dim=0)   # [N, D]
        if P.shape[0] < 2:
            continue
        Pa = P - P.mean(dim=0, keepdim=True)             # per-coord across-sample anomaly
        Ta = T - T.mean(dim=0, keepdim=True)
        Ps, _ = torch.sort(Pa, dim=0)                    # sort along samples, per coord (grad through pred)
        Ts, _ = torch.sort(Ta.detach(), dim=0)           # target sorted, no grad
        scale = Ta.detach().std(dim=0) + 1e-6            # [D], per-coord target spread
        w1_c = (Ps - Ts).abs().mean(dim=0) / scale       # [D] relative per-coord 1-Wasserstein
        terms.append(w1_c.mean())                        # mean over coordinates
    if not terms:
        return None
    return torch.stack(terms).mean()                     # mean over horizons


def gaussian_crps(mu, sigma, y, eps=1e-6):
    """Closed-form CRPS of a Gaussian forecast N(mu, sigma^2) against observation y.

    A *proper* scoring rule: minimized only by the true predictive distribution,
    so sigma is trained (against the same y, no ground-truth sigma) toward the
    calibrated conditional spread and cannot be gamed by decorrelated variance
    (unlike the across-start anomaly-W1 surrogate). Differentiable in mu and sigma.
    Returns the mean over all elements.
    """
    _INV_SQRT_PI = 0.5641895835477563
    _INV_SQRT_2 = 0.7071067811865476
    _INV_SQRT_2PI = 0.3989422804014327
    sigma = sigma.clamp_min(eps)
    z = (y - mu) / sigma
    Phi = 0.5 * (1.0 + torch.erf(z * _INV_SQRT_2))
    phi = torch.exp(-0.5 * z * z) * _INV_SQRT_2PI
    crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - _INV_SQRT_PI)
    return crps.mean()


def _gaussian_crps_per_sample(mu, sigma, y, eps=1e-6):
    """The same Gaussian CRPS as :func:`gaussian_crps`, reduced only within B."""

    _INV_SQRT_PI = 0.5641895835477563
    _INV_SQRT_2 = 0.7071067811865476
    _INV_SQRT_2PI = 0.3989422804014327
    sigma = sigma.clamp_min(eps)
    z = (y - mu) / sigma
    phi = torch.exp(-0.5 * z.square()) * _INV_SQRT_2PI
    Phi = 0.5 * (1.0 + torch.erf(z * _INV_SQRT_2))
    crps = sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - _INV_SQRT_PI)
    return crps.reshape(crps.shape[0], -1).mean(dim=1)


def compute_path_generator_loss(
    model,
    state: torch.Tensor,
    stim: torch.Tensor,
    args,
    epoch: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Train a low-dimensional path generator on full finite-horizon paths.

    The model outputs a coarse velocity generator g_{0:K-1}.  The deterministic
    integration operator produces a future path.  We supervise the whole path
    and optionally add a low-dimensional multi-marginal distributional error and
    generator energy/bottleneck penalties.
    """
    raw = unwrap_model(model)
    B, T, _ = get_batch_time_shape(state)
    W = int(args.window_size)
    K_cfg = int(getattr(args, "pathgen_horizon", getattr(raw, "path_horizon", 8)))
    K_cfg = max(1, min(K_cfg, int(getattr(raw, "path_horizon", K_cfg))))

    if T <= W:
        raise ValueError(f"Sequence too short for path-generator training: T={T}, window={W}")

    max_start = T - 1  # target start t must have at least x_t
    starts = list(range(W, max_start + 1, max(int(getattr(args, "ar_train_stride", 1)), 1)))
    starts = [t for t in starts if T - t >= 1]
    if raw.training:
        n_starts = int(getattr(args, "ar_train_starts_per_sequence", 1))
        n_starts = max(1, min(n_starts, len(starts)))
        use_random = bool(getattr(args, "ar_train_random_starts", True))
        if use_random:
            chosen = random.sample(starts, n_starts)
        else:
            offset = max(int(epoch) - 1, 0) % max(1, len(starts))
            chosen = [starts[(offset + i) % len(starts)] for i in range(n_starts)]
    else:
        eval_stride = int(getattr(args, "ar_eval_stride", 4))
        chosen = list(range(W, max_start + 1, max(eval_stride, 1)))

    loss_name = str(getattr(args, "ar_loss", "rel_l2"))
    lambda_path = float(getattr(args, "pathgen_lambda_path", 1.0))
    lambda_mmsbm = float(getattr(args, "pathgen_lambda_mmsbm", 0.0))
    lambda_energy = float(getattr(args, "pathgen_lambda_energy", 0.0))
    lambda_code = float(getattr(args, "pathgen_lambda_code", 0.0))

    path_losses = []
    one_losses = []
    mmsbm_losses = []
    energy_losses = []
    code_losses = []
    used_horizons = []
    corrs = []
    rels = []

    for t in chosen:
        K_eff = max(1, min(K_cfg, T - t))
        history = time_window(state, t - W, t)
        target_path = state[:, t:t + K_eff]
        # Future external input is reserved for later conditioning.  The current
        # model ignores it, but keeping this slice makes the training interface
        # explicit and easy to extend.
        _stim_future = time_window(stim, t, t + K_eff) if stim is not None else None

        if bool(getattr(raw, "uses_future_stimulus", False)):
            stim_hist = time_window(stim, t - W, t) if stim is not None else None
            pred_path, aux = raw.generate_path(
                history, horizon=K_eff, return_aux=True,
                stim_window=stim_hist, stim_future=_stim_future,
            )
        else:
            pred_path, aux = raw.generate_path(history, horizon=K_eff, return_aux=True)

        step_losses = []
        for k in range(K_eff):
            pred_k = pred_path[:, k]
            target_k = target_path[:, k]
            if loss_name == "rel_l2":
                step_losses.append(relative_l2(pred_k, target_k))
            else:
                step_losses.append(elementwise_state_loss(pred_k, target_k, loss=loss_name))
        path_loss = torch.stack(step_losses).mean()
        path_losses.append(path_loss)
        one_losses.append(step_losses[0])

        if lambda_mmsbm != 0.0:
            mm_terms = []
            for k in range(K_eff):
                pred_obs = raw.encode_path_observable(pred_path[:, k])
                tgt_obs = raw.encode_path_observable(target_path[:, k])
                mm_terms.append(_diag_gaussian_mmsbm_error(pred_obs, tgt_obs))
            mmsbm_losses.append(torch.stack(mm_terms).mean())
        else:
            mmsbm_losses.append(path_loss.new_tensor(0.0))

        deltas = aux["path_generator_deltas"]
        energy_losses.append(deltas.pow(2).mean())
        code_losses.append(aux["path_generator_code"].pow(2).mean())
        used_horizons.append(float(K_eff))

        with torch.no_grad():
            corrs.append(corrcoef_flat(pred_path[:, 0], target_path[:, 0]))
            rels.append(relative_l2(pred_path[:, 0], target_path[:, 0]))

    loss_path = torch.stack(path_losses).mean()
    loss_one = torch.stack(one_losses).mean()
    loss_mmsbm = torch.stack(mmsbm_losses).mean()
    loss_energy = torch.stack(energy_losses).mean()
    loss_code = torch.stack(code_losses).mean()

    loss = (
        lambda_path * loss_path
        + lambda_mmsbm * loss_mmsbm
        + lambda_energy * loss_energy
        + lambda_code * loss_code
    )

    logs = {
        "loss": float(loss.detach().cpu()),
        "ar/loss_one_step": float(loss_one.detach().cpu()),
        "pathgen/path_loss": float(loss_path.detach().cpu()),
        "pathgen/mmsbm_loss": float(loss_mmsbm.detach().cpu()),
        "pathgen/energy_loss": float(loss_energy.detach().cpu()),
        "pathgen/code_loss": float(loss_code.detach().cpu()),
        "pathgen/lambda_path": float(lambda_path),
        "pathgen/lambda_mmsbm": float(lambda_mmsbm),
        "pathgen/lambda_energy": float(lambda_energy),
        "pathgen/lambda_code": float(lambda_code),
        "pathgen/used_horizon": float(sum(used_horizons) / max(len(used_horizons), 1)),
        "ar/one_step_corr": float(torch.stack(corrs).mean().detach().cpu()) if corrs else 0.0,
        "ar/one_step_rel_l2": float(torch.stack(rels).mean().detach().cpu()) if rels else 0.0,
        "ar/num_train_starts": float(len(chosen)),
        "ar/window_size": float(W),
        # Keep old keys present so plotting/eval JSONs remain comparable.
        "ar/fold_loss": 0.0,
        "ar/fold_lambda": 0.0,
        "ar/fold_weighted_loss": 0.0,
        "ar/fold_used_horizon": 0.0,
        "ar/fold_A_sigma": 0.0,
        "ar/ftg_amp_loss": 0.0,
        "ar/ftg_bound_loss": 0.0,
        "ar/ftg_lambda_amp": 0.0,
        "ar/ftg_lambda_bound": 0.0,
        "ar/ftg_amp_weighted_loss": 0.0,
        "ar/ftg_bound_weighted_loss": 0.0,
        "ar/ftg_total_weighted_loss": 0.0,
        "ar/ftg_amp_pct_of_one_step": 0.0,
        "ar/ftg_amp_pct_of_total": 0.0,
        "ar/ftg_total_pct_of_one_step": 0.0,
        "ar/ftg_total_pct_of_total": 0.0,
        "ar/ftg_amp_ratio": 0.0,
        "ar/ftg_horizon": 0.0,
        "ar/ftg_sigma_hat": 0.0,
        "ar/ftg_ratio_k0": 0.0,
        "ar/ftg_ratio_k1": 0.0,
        "ar/ftg_ratio_k2": 0.0,
        "ar/ftg_ratio_klast": 0.0,
    }
    return loss, logs



def _state_token_decode_weights_from_meta(centers: torch.Tensor, decay: torch.Tensor, gates: torch.Tensor, length: int) -> torch.Tensor:
    """Compute tied decoder token-time weights W[k,i] from token metadata.

    The model now uses a responsibility-tied assignment:

        alpha_i(k) = softmax_k(-lambda_i |s_k - tau_i|)
        w_i(k)     = alpha_i(k) / sum_j alpha_j(k).

    Gates are accepted for API/checkpoint compatibility but are intentionally
    not used to create an independent decoder-side temporal assignment.
    Returns [B,T,L], matching aux["state_token_weights"].
    """
    T = int(length)
    if T <= 1:
        u = centers.new_zeros(1)
    else:
        u = torch.linspace(0.0, 1.0, T, device=centers.device, dtype=centers.dtype)
    u = u.view(1, 1, T)
    logits = -decay.unsqueeze(-1) * (u - centers.unsqueeze(-1)).abs()
    alpha = torch.softmax(logits, dim=-1)  # [B,L,T]
    w = alpha.transpose(1, 2)              # [B,T,L]
    return w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _state_sequence_meta_losses(aux, args):
    """Regularize ordered state-token metadata."""
    zero = None
    centers = aux.get("posterior_centers", aux.get("prior_centers", None))
    weights = aux.get("state_token_weights", None)
    gates = aux.get("posterior_gates", aux.get("prior_gates", None))
    if centers is not None:
        zero = centers.new_tensor(0.0)
    elif weights is not None:
        zero = weights.new_tensor(0.0)
    else:
        return torch.tensor(0.0), {"statetok/order_loss":0.0,"statetok/coverage_loss":0.0,"statetok/gate_loss":0.0}
    order_loss = zero
    if centers is not None and centers.shape[1] > 1:
        sep = float(getattr(args, "statetok_min_separation", 0.02))
        order_loss = torch.relu(centers[:, :-1] + sep - centers[:, 1:]).pow(2).mean()
    coverage_loss = zero
    if weights is not None:
        # weights can be [B,T,L]. Encourage no hard holes/duplicates but keep weak by default.
        cov = weights.sum(dim=-1)
        coverage_loss = (cov - 1.0).pow(2).mean()
    gate_loss = zero
    if gates is not None:
        # Small sparsity/activation penalty; useful when gates are enabled.
        gate_loss = gates.mean()
    return order_loss + coverage_loss + gate_loss * 0.0, {
        "statetok/order_loss": float(order_loss.detach().cpu()),
        "statetok/coverage_loss": float(coverage_loss.detach().cpu()),
        "statetok/gate_loss": float(gate_loss.detach().cpu()),
    }


def compute_state_sequence_ae_loss(model, state, stim, args, epoch: int = 0):
    """Two-stage state-sequence AE losses.

    Stage ``ae`` trains posterior trajectory encoder/decoder:
        future path X -> M -> X.
    Stage ``transition``/``finetune`` trains causal prior/transition from history:
        history -> M_hat -> future path, matching posterior M from the GT future.
    """
    raw = unwrap_model(model)
    B, T, _ = get_batch_time_shape(state)
    W = int(args.window_size)
    K_cfg = int(getattr(args, "pathgen_horizon", getattr(raw, "path_horizon", 32)))
    K_cfg = max(1, min(K_cfg, int(getattr(raw, "path_horizon", K_cfg))))
    if T <= W:
        raise ValueError(f"Sequence too short for state-sequence AE training: T={T}, window={W}")

    max_start = T - 1
    starts = list(range(W, max_start + 1, max(int(getattr(args, "ar_train_stride", 1)), 1)))
    starts = [t for t in starts if T - t >= 1]
    if raw.training:
        n_starts = max(1, min(int(getattr(args, "ar_train_starts_per_sequence", 1)), len(starts)))
        if bool(getattr(args, "ar_train_random_starts", True)):
            chosen = random.sample(starts, n_starts)
        else:
            offset = max(int(epoch) - 1, 0) % max(1, len(starts))
            chosen = [starts[(offset + i) % len(starts)] for i in range(n_starts)]
    else:
        chosen = list(range(W, max_start + 1, max(int(getattr(args, "ar_eval_stride", 4)), 1)))

    stage = str(getattr(args, "statetok_stage", getattr(raw, "stage", "ae")))
    loss_name = str(getattr(args, "ar_loss", "rel_l2"))
    lam_rec = float(getattr(args, "statetok_lambda_rec", 1.0))
    lam_forecast = float(getattr(args, "statetok_lambda_forecast", 1.0))
    lam_latent = float(getattr(args, "statetok_lambda_latent", 0.1))
    lam_weight = float(getattr(args, "statetok_lambda_weight", 0.0))
    lam_gate = float(getattr(args, "statetok_lambda_gate", 0.0))
    lam_meta = float(getattr(args, "statetok_lambda_meta", 0.0))
    lam_energy = float(getattr(args, "pathgen_lambda_energy", 0.0))

    rec_losses=[]; forecast_losses=[]; latent_losses=[]; weight_losses=[]; gate_match_losses=[]; meta_losses=[]; energy_losses=[]; one_losses=[]; corrs=[]; rels=[]; used=[]
    meta_log_accum={}

    for t in chosen:
        K_eff=max(1,min(K_cfg,T-t))
        history=time_window(state,t-W,t)
        target=state[:,t:t+K_eff]
        stim_hist = time_window(stim, t-W, t) if stim is not None else None
        stim_future = time_window(stim, t, t+K_eff) if stim is not None else None

        if stage == "ae":
            ae_input, ae_keep = _apply_state_sequence_ignore_mask(target, args)
            recon, aux = raw.reconstruct_sequence(ae_input, return_aux=True)
            if float(getattr(args, "statetok_ae_mask_prob", 0.0)) > 0.0:
                # Ignore-mask AE: reconstruct visible frames and softly force
                # masked slots to zero.  This branch is used for both rel_l2 and
                # mse-like losses; otherwise MSE training would become an
                # unintended inpainting objective on masked positions.
                if loss_name == "rel_l2":
                    rec, vis_rec, zero_rec = _masked_ignore_ae_rel_l2(recon, target, ae_keep, args)
                else:
                    Bm, Tm = target.shape[:2]
                    view_shape = [Bm, Tm] + [1] * (target.dim() - 2)
                    visible = ae_keep.view(*view_shape)
                    hidden = 1.0 - visible
                    beta = float(getattr(args, "statetok_ae_mask_zero_weight", 0.1))
                    denom = (visible * target).reshape(Bm, -1).pow(2).mean(dim=1).clamp_min(1e-8)
                    vis_per = (visible * (recon - target)).reshape(Bm, -1).pow(2).mean(dim=1) / denom
                    zero_per = (hidden * recon).reshape(Bm, -1).pow(2).mean(dim=1) / denom
                    vis_rec = vis_per.mean()
                    zero_rec = zero_per.mean()
                    rec = vis_rec + beta * zero_rec
                step = [rec for _ in range(K_eff)]
            else:
                step=[]
                for k in range(K_eff):
                    if loss_name == "rel_l2": step.append(relative_l2(recon[:,k], target[:,k]))
                    else: step.append(elementwise_state_loss(recon[:,k], target[:,k], loss=loss_name))
                rec=torch.stack(step).mean()
                vis_rec = rec.detach()
                zero_rec = rec.new_tensor(0.0)
            rec_losses.append(rec); one_losses.append(step[0])
            mreg, mlogs = _state_sequence_meta_losses(aux,args)
            meta_losses.append(mreg)
            energy_losses.append(recon.new_tensor(0.0))
            forecast_losses.append(rec.new_tensor(0.0)); latent_losses.append(rec.new_tensor(0.0))
            weight_losses.append(rec.new_tensor(0.0)); gate_match_losses.append(rec.new_tensor(0.0))
            pred_for_log=recon
        else:
            with torch.set_grad_enabled(stage == "finetune" and not bool(getattr(args,"statetok_detach_posterior", True))):
                post_tok, post_c, post_d, post_g = raw.encode_sequence(target, return_aux=False)
            if bool(getattr(args,"statetok_detach_posterior", True)):
                post_tok=post_tok.detach(); post_c=post_c.detach(); post_d=post_d.detach(); post_g=post_g.detach()
            pred, aux = raw.generate_path(history, horizon=K_eff, return_aux=True, stim_window=stim_hist, stim_future=stim_future)
            step=[]
            for k in range(K_eff):
                if loss_name == "rel_l2": step.append(relative_l2(pred[:,k], target[:,k]))
                else: step.append(elementwise_state_loss(pred[:,k], target[:,k], loss=loss_name))
            fl=torch.stack(step).mean(); forecast_losses.append(fl); one_losses.append(step[0])
            # Match posterior state tokens and lightly match token metadata.
            # This keeps the content-space alignment separate from the temporal
            # assignment loss below, so --statetok_lambda_weight can be tuned
            # without being hidden under --statetok_lambda_latent.
            ptok=aux["prior_tokens"]
            token_match = (ptok - post_tok).pow(2).mean()
            center_match = (aux["prior_centers"]-post_c).pow(2).mean()
            decay_match = (torch.log(aux["prior_decay"].clamp_min(1e-6))-torch.log(post_d.clamp_min(1e-6))).pow(2).mean()
            c_tau = float(getattr(args, "statetok_c_tau", 0.1))
            c_lambda = float(getattr(args, "statetok_c_lambda", 0.01))
            latent = token_match + c_tau * center_match + c_lambda * decay_match
            latent_losses.append(latent)

            # Directly align the decoder temporal assignment W[k,i].
            # Prior W is returned by generate_path/decode_tokens.  Posterior W
            # is recomputed from posterior centers/decays/raw gates, without
            # running the decoder.
            prior_w = aux["state_token_weights"]
            post_w = _state_token_decode_weights_from_meta(post_c, post_d, post_g, K_eff)
            weight_match = (prior_w - post_w).pow(2).mean()
            weight_losses.append(weight_match)

            # Optional gate matching.  The model stores/logs sigmoid gates in
            # aux, so compare sigmoid posterior gates to sigmoid prior gates.
            post_gate_sig = torch.sigmoid(post_g)
            prior_gate_sig = aux.get("prior_gates", None)
            if prior_gate_sig is None:
                gate_match = weight_match.new_tensor(0.0)
            else:
                gate_match = (prior_gate_sig - post_gate_sig).pow(2).mean()
            gate_match_losses.append(gate_match)
            mreg,mlogs=_state_sequence_meta_losses(aux,args)
            meta_losses.append(mreg)
            deltas=aux.get("path_generator_deltas", None)
            energy_losses.append(deltas.pow(2).mean() if deltas is not None else pred.new_tensor(0.0))
            rec_losses.append(fl.new_tensor(0.0))
            pred_for_log=pred
        for k,v in mlogs.items(): meta_log_accum[k]=meta_log_accum.get(k,0.0)+float(v)
        used.append(float(K_eff))
        with torch.no_grad():
            corrs.append(corrcoef_flat(pred_for_log[:,0], target[:,0]))
            rels.append(relative_l2(pred_for_log[:,0], target[:,0]))

    rec_loss=torch.stack(rec_losses).mean(); forecast_loss=torch.stack(forecast_losses).mean()
    latent_loss=torch.stack(latent_losses).mean(); weight_loss=torch.stack(weight_losses).mean(); gate_match_loss=torch.stack(gate_match_losses).mean()
    meta_loss=torch.stack(meta_losses).mean(); energy_loss=torch.stack(energy_losses).mean()
    if stage == "ae":
        loss = lam_rec*rec_loss + lam_meta*meta_loss
    else:
        loss = (
            lam_forecast*forecast_loss
            + lam_latent*latent_loss
            + lam_weight*weight_loss
            + lam_gate*gate_match_loss
            + lam_meta*meta_loss
            + lam_energy*energy_loss
        )
    n=max(len(chosen),1)
    logs={
        "loss": float(loss.detach().cpu()),
        "ar/loss_one_step": float(torch.stack(one_losses).mean().detach().cpu()) if one_losses else 0.0,
        "statetok/stage_is_ae": 1.0 if stage == "ae" else 0.0,
        "statetok/use_masked_init": 1.0 if bool(getattr(raw, "use_masked_init", False)) else 0.0,
        "statetok/ae_mask_prob": float(getattr(args, "statetok_ae_mask_prob", 0.0)),
        "statetok/rec_loss": float(rec_loss.detach().cpu()),
        "statetok/ae_mask_prob": float(getattr(args, "statetok_ae_mask_prob", 0.0)) if stage == "ae" else 0.0,
        "statetok/ae_mask_zero_weight": float(getattr(args, "statetok_ae_mask_zero_weight", 0.1)) if stage == "ae" else 0.0,
        "statetok/forecast_loss": float(forecast_loss.detach().cpu()),
        "statetok/latent_loss": float(latent_loss.detach().cpu()),
        "statetok/weight_loss": float(weight_loss.detach().cpu()),
        "statetok/gate_match_loss": float(gate_match_loss.detach().cpu()),
        "statetok/lambda_weight": float(lam_weight),
        "statetok/lambda_gate": float(lam_gate),
        "statetok/c_tau": float(getattr(args, "statetok_c_tau", 0.1)),
        "statetok/c_lambda": float(getattr(args, "statetok_c_lambda", 0.01)),
        "statetok/weight_weighted_loss": float((lam_weight * weight_loss).detach().cpu()),
        "statetok/gate_weighted_loss": float((lam_gate * gate_match_loss).detach().cpu()),
        "statetok/meta_loss": float(meta_loss.detach().cpu()),
        "pathgen/path_loss": float((rec_loss if stage=="ae" else forecast_loss).detach().cpu()),
        "pathgen/energy_loss": float(energy_loss.detach().cpu()),
        "pathgen/code_loss": float(latent_loss.detach().cpu()),
        "pathgen/lambda_path": float(lam_forecast if stage!="ae" else lam_rec),
        "pathgen/lambda_energy": float(lam_energy),
        "pathgen/lambda_code": float(lam_latent),
        "pathgen/used_horizon": float(sum(used)/max(len(used),1)),
        "ar/one_step_corr": float(torch.stack(corrs).mean().detach().cpu()) if corrs else 0.0,
        "ar/one_step_rel_l2": float(torch.stack(rels).mean().detach().cpu()) if rels else 0.0,
        "ar/num_train_starts": float(len(chosen)),
        "ar/window_size": float(W),
        "ar/fold_loss":0.0,"ar/fold_lambda":0.0,"ar/fold_weighted_loss":0.0,"ar/fold_used_horizon":0.0,"ar/fold_A_sigma":0.0,
        "ar/ftg_amp_loss":0.0,"ar/ftg_bound_loss":0.0,"ar/ftg_lambda_amp":0.0,"ar/ftg_lambda_bound":0.0,
        "ar/ftg_amp_weighted_loss":0.0,"ar/ftg_bound_weighted_loss":0.0,"ar/ftg_total_weighted_loss":0.0,
        "ar/ftg_amp_pct_of_one_step":0.0,"ar/ftg_amp_pct_of_total":0.0,"ar/ftg_total_pct_of_one_step":0.0,"ar/ftg_total_pct_of_total":0.0,
        "ar/ftg_amp_ratio":0.0,"ar/ftg_horizon":0.0,"ar/ftg_sigma_hat":0.0,"ar/ftg_ratio_k0":0.0,"ar/ftg_ratio_k1":0.0,"ar/ftg_ratio_k2":0.0,"ar/ftg_ratio_klast":0.0,
    }
    for k,v in meta_log_accum.items(): logs[k]=v/n
    return loss, logs



def compute_clock_latent_ae_loss(model, state, stim, args, epoch: int = 0):
    """Two-stage matched clock-time latent AR baseline loss.

    Stage ``ae`` trains a frame-wise AE in the original clock-time grid.
    Stage ``transition`` trains a causal latent AR prior that rolls one fixed
    clock step at a time.  This is the matched baseline for state_sequence_ae_*:
    same idea of AE + frozen/lightly tuned decoder, but without learned
    state/event-time tokens.
    """
    raw = unwrap_model(model)
    B, T, _ = get_batch_time_shape(state)
    W = int(args.window_size)
    K_cfg = int(getattr(args, "pathgen_horizon", getattr(raw, "path_horizon", 32)))
    K_cfg = max(1, min(K_cfg, int(getattr(raw, "path_horizon", K_cfg))))
    if T <= W:
        raise ValueError(f"Sequence too short for clock-latent AE training: T={T}, window={W}")

    starts = list(range(W, T, max(int(getattr(args, "ar_train_stride", 1)), 1)))
    starts = [t for t in starts if T - t >= 1]
    if raw.training:
        n_starts = max(1, min(int(getattr(args, "ar_train_starts_per_sequence", 1)), len(starts)))
        if bool(getattr(args, "ar_train_random_starts", True)):
            chosen = random.sample(starts, n_starts)
        else:
            offset = max(int(epoch) - 1, 0) % max(1, len(starts))
            chosen = [starts[(offset + i) % len(starts)] for i in range(n_starts)]
    else:
        chosen = list(range(W, T, max(int(getattr(args, "ar_eval_stride", 4)), 1)))

    stage = str(getattr(args, "clocklat_stage", getattr(raw, "stage", "ae")))
    loss_name = str(getattr(args, "ar_loss", "rel_l2"))
    lam_rec = float(getattr(args, "clocklat_lambda_rec", 1.0))
    lam_forecast = float(getattr(args, "clocklat_lambda_forecast", 1.0))
    lam_latent = float(getattr(args, "clocklat_lambda_latent", 0.1))
    lam_energy = float(getattr(args, "pathgen_lambda_energy", 0.0))

    rec_losses=[]; forecast_losses=[]; latent_losses=[]; energy_losses=[]; one_losses=[]; corrs=[]; rels=[]; used=[]
    for t in chosen:
        K_eff=max(1,min(K_cfg,T-t))
        history=time_window(state,t-W,t)
        target=state[:,t:t+K_eff]
        stim_hist = time_window(stim, t-W, t) if stim is not None else None
        stim_future = time_window(stim, t, t+K_eff) if stim is not None else None

        if stage == "ae":
            recon, aux = raw.reconstruct_frames(target, return_aux=True)
            step=[]
            for k in range(K_eff):
                if loss_name == "rel_l2": step.append(relative_l2(recon[:,k], target[:,k]))
                else: step.append(elementwise_state_loss(recon[:,k], target[:,k], loss=loss_name))
            rec=torch.stack(step).mean()
            rec_losses.append(rec); one_losses.append(step[0])
            forecast_losses.append(rec.new_tensor(0.0)); latent_losses.append(rec.new_tensor(0.0)); energy_losses.append(rec.new_tensor(0.0))
            pred_for_log=recon
        else:
            with torch.no_grad() if bool(getattr(args,"clocklat_detach_posterior", True)) else torch.enable_grad():
                _, post_aux = raw.reconstruct_frames(target, return_aux=True)
                post_z = post_aux["clock_latent_codes"]
            if bool(getattr(args,"clocklat_detach_posterior", True)):
                post_z = post_z.detach()
            pred, aux = raw.generate_path(history, horizon=K_eff, return_aux=True, stim_window=stim_hist, stim_future=stim_future)
            step=[]
            for k in range(K_eff):
                if loss_name == "rel_l2": step.append(relative_l2(pred[:,k], target[:,k]))
                else: step.append(elementwise_state_loss(pred[:,k], target[:,k], loss=loss_name))
            fl=torch.stack(step).mean()
            forecast_losses.append(fl); one_losses.append(step[0]); rec_losses.append(fl.new_tensor(0.0))
            pred_z = aux["clock_latent_pred_codes"]
            latent_losses.append((pred_z - post_z).pow(2).mean())
            deltas=aux.get("path_generator_deltas", None)
            energy_losses.append(deltas.pow(2).mean() if deltas is not None else pred.new_tensor(0.0))
            pred_for_log=pred
        used.append(float(K_eff))
        with torch.no_grad():
            corrs.append(corrcoef_flat(pred_for_log[:,0], target[:,0]))
            rels.append(relative_l2(pred_for_log[:,0], target[:,0]))

    rec_loss=torch.stack(rec_losses).mean(); forecast_loss=torch.stack(forecast_losses).mean()
    latent_loss=torch.stack(latent_losses).mean(); energy_loss=torch.stack(energy_losses).mean()
    if stage == "ae":
        loss = lam_rec * rec_loss
    else:
        loss = lam_forecast * forecast_loss + lam_latent * latent_loss + lam_energy * energy_loss
    logs={
        "loss": float(loss.detach().cpu()),
        "ar/loss_one_step": float(torch.stack(one_losses).mean().detach().cpu()) if one_losses else 0.0,
        "clocklat/stage_is_ae": 1.0 if stage == "ae" else 0.0,
        "clocklat/rec_loss": float(rec_loss.detach().cpu()),
        "clocklat/forecast_loss": float(forecast_loss.detach().cpu()),
        "clocklat/latent_loss": float(latent_loss.detach().cpu()),
        "pathgen/path_loss": float((rec_loss if stage=="ae" else forecast_loss).detach().cpu()),
        "pathgen/energy_loss": float(energy_loss.detach().cpu()),
        "pathgen/code_loss": float(latent_loss.detach().cpu()),
        "pathgen/lambda_path": float(lam_forecast if stage!="ae" else lam_rec),
        "pathgen/lambda_energy": float(lam_energy),
        "pathgen/lambda_code": float(lam_latent),
        "pathgen/used_horizon": float(sum(used)/max(len(used),1)),
        "ar/one_step_corr": float(torch.stack(corrs).mean().detach().cpu()) if corrs else 0.0,
        "ar/one_step_rel_l2": float(torch.stack(rels).mean().detach().cpu()) if rels else 0.0,
        "ar/num_train_starts": float(len(chosen)),
        "ar/window_size": float(W),
        "ar/fold_loss":0.0,"ar/fold_lambda":0.0,"ar/fold_weighted_loss":0.0,"ar/fold_used_horizon":0.0,"ar/fold_A_sigma":0.0,
        "ar/ftg_amp_loss":0.0,"ar/ftg_bound_loss":0.0,"ar/ftg_lambda_amp":0.0,"ar/ftg_lambda_bound":0.0,
        "ar/ftg_amp_weighted_loss":0.0,"ar/ftg_bound_weighted_loss":0.0,"ar/ftg_total_weighted_loss":0.0,
        "ar/ftg_amp_pct_of_one_step":0.0,"ar/ftg_amp_pct_of_total":0.0,"ar/ftg_total_pct_of_one_step":0.0,"ar/ftg_total_pct_of_total":0.0,
        "ar/ftg_amp_ratio":0.0,"ar/ftg_horizon":0.0,"ar/ftg_sigma_hat":0.0,"ar/ftg_ratio_k0":0.0,"ar/ftg_ratio_k1":0.0,"ar/ftg_ratio_k2":0.0,"ar/ftg_ratio_klast":0.0,
    }
    return loss, logs



def _ar_forward_pred(raw, stim_window: torch.Tensor | None, history: torch.Tensor) -> torch.Tensor:
    """Return one predicted next frame [B,...] for any standard AR model."""
    out = raw(stim_window, history, return_aux=False)
    if isinstance(out, tuple):
        out = out[0]
    if out.dim() == history.dim():
        # [B,1,...]
        return out[:, 0]
    return out


def _ar_step_history_with_stim(raw, history: torch.Tensor, stim_window: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """One rollout step returning (next_history, predicted_frame).

    Prefer model.step_history(history, stim_window) when supported; otherwise fall
    back to forward() and shift the history manually.
    """
    try:
        next_hist = raw.step_history(history, stim_window)
        pred = next_hist[:, -1]
        return next_hist, pred
    except TypeError:
        if hasattr(raw, "step_history"):
            next_hist = raw.step_history(history)
            pred = next_hist[:, -1]
            return next_hist, pred
    pred = _ar_forward_pred(raw, stim_window, history)
    next_hist = torch.cat([history[:, 1:], pred.unsqueeze(1)], dim=1)
    return next_hist, pred



def _field_peh_zero_logs() -> dict:
    """Zero logs for field periodic error homogenization (PEH)."""
    return {
        "ar/field_peh_enabled": 0.0,
        "ar/field_peh_skipped_short": 0.0,
        "ar/field_peh_loss": 0.0,
        "ar/field_peh_lambda": 0.0,
        "ar/field_peh_weighted_loss": 0.0,
        "ar/field_peh_horizon": 0.0,
        "ar/field_peh_available_horizon": 0.0,
        "ar/field_peh_type": 0.0,
        "ar/field_peh_closure_ratio": 0.0,
        "ar/field_peh_mean_error_norm": 0.0,
        "ar/field_peh_sum_error_norm": 0.0,
        "ar/field_peh_error_temporal_cos": 0.0,
        "ar/field_peh_first_rel_l2": 0.0,
        "ar/field_peh_last_rel_l2": 0.0,
    }


def _mean_pairwise_temporal_cos(errors: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean off-diagonal cosine of rollout errors.

    errors: [B,K,...]. Returns scalar. Positive values indicate coherent
    same-direction error drift across rollout time, which PEH is designed to
    reduce.
    """
    if errors.shape[1] <= 1:
        return errors.new_tensor(0.0)
    e = errors.reshape(errors.shape[0], errors.shape[1], -1)
    e = e / e.norm(dim=-1, keepdim=True).clamp_min(float(eps))
    gram = torch.matmul(e, e.transpose(1, 2))
    k = int(errors.shape[1])
    mask = ~torch.eye(k, device=errors.device, dtype=torch.bool).unsqueeze(0)
    return gram.masked_select(mask).mean()


def _field_peh_effective_lambda(args, epoch: int) -> float:
    lam = float(getattr(args, "field_peh_lambda", 0.0))
    start = int(getattr(args, "field_peh_start_epoch", 1))
    ramp = int(getattr(args, "field_peh_ramp_epochs", 0))
    if epoch < start:
        return 0.0
    if ramp > 0:
        return lam * min(1.0, max(0.0, float(epoch - start + 1) / float(ramp)))
    return lam


def compute_field_peh_rollout_loss(
    raw,
    state: torch.Tensor,
    stim: torch.Tensor | None,
    start_t: int,
    args,
    epoch: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Short-BPTT-compatible PEH regularizer for 2D field AR models.

    This is meant for The Well / Gray-Scott experiments where the main training
    signal is a cheap short rollout-BPTT objective, e.g. K=16.  PEH adds an
    additional trajectory-level error-geometry term over a free rollout:

        e_k = x_hat_{t+k}^{AR} - x_{t+k}
        L_raw  = || sum_k e_k || / (sum_k ||e_k||).detach()
        L_wpeh = same closure after per-channel whitening
        L_ipeh = same closure on innovations e_k - rho e_{k-1}

    The denominator is detached by default so the regularizer cannot improve by
    increasing the total error radius.  Gradients flow through the free rollout
    predictions, but the scalar being optimized is not another per-step rollout
    MSE; it shapes the coherent bias component left by the short-BPTT solution.
    """
    B, T, _ = get_batch_time_shape(state)
    W = int(args.window_size)
    K_req = int(getattr(args, "field_peh_horizon", 0))
    K = min(max(K_req, 0), T - int(start_t))
    zero_ref = time_point(state, max(min(int(start_t), T - 1), 0), keep_time=True).new_tensor(0.0)
    zero_logs = _field_peh_zero_logs()
    zero_logs["ar/field_peh_lambda"] = float(getattr(args, "field_peh_lambda", 0.0))
    if K <= 0 or int(start_t) - W < 0:
        zero_logs["ar/field_peh_skipped_short"] = 1.0
        return zero_ref, zero_logs

    if stim is None:
        stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))

    history = time_window(state, int(start_t) - W, int(start_t))
    preds = []
    targets = []
    for s in range(K):
        cur = int(start_t) + s
        stim_window = time_window(stim, cur - W, cur)
        history, pred_frame = _ar_step_history_with_stim(raw, history, stim_window)
        preds.append(pred_frame)
        targets.append(time_point(state, cur, keep_time=False))

    pred_seq = torch.stack(preds, dim=1)       # [B,K,...]
    target_seq = torch.stack(targets, dim=1)   # [B,K,...]
    errors = pred_seq - target_seq

    eps = float(getattr(args, "field_peh_eps", 1e-8))
    peh_type = str(getattr(args, "field_peh_type", "raw")).lower()
    transformed = errors
    type_id = 0.0
    if peh_type in ("wpeh", "whiten", "whitened"):
        type_id = 1.0
        # Prefer the model's fitted channel normalizer if available.  This is
        # cheap and important for multi-channel PDE fields with different scales.
        std = getattr(raw, "state_std", None)
        if torch.is_tensor(std):
            # state_std is [1,1,C,1,1]; errors is [B,K,C,H,W].
            transformed = errors / std.to(errors.device, errors.dtype).clamp_min(eps)
        else:
            dims = (0, 1) + tuple(range(3, errors.dim()))
            ch_std = target_seq.std(dim=dims, keepdim=True).clamp_min(eps)
            transformed = errors / ch_std
    elif peh_type in ("ipeh", "innovation", "innov"):
        type_id = 2.0
        rho = float(getattr(args, "field_peh_innovation_rho", 0.8))
        if errors.shape[1] > 1:
            first = errors[:, :1]
            rest = errors[:, 1:] - rho * errors[:, :-1]
            transformed = torch.cat([first, rest], dim=1)
        else:
            transformed = errors
    elif peh_type not in ("raw", "peh"):
        raise ValueError(f"Unknown --field_peh_type={peh_type}; use raw, wpeh, or ipeh.")

    flat = transformed.reshape(transformed.shape[0], transformed.shape[1], -1)
    step_norm = flat.norm(dim=-1)  # [B,K]
    sum_error = transformed.sum(dim=1)
    sum_norm = sum_error.reshape(sum_error.shape[0], -1).norm(dim=-1)  # [B]
    denom_kind = str(getattr(args, "field_peh_denom", "error_sum_detach")).lower()
    if denom_kind == "target":
        denom = target_seq.reshape(target_seq.shape[0], -1).norm(dim=-1).clamp_min(eps)
    elif denom_kind == "sqrtk":
        denom = (step_norm.square().sum(dim=1).sqrt() * math.sqrt(float(K))).clamp_min(eps).detach()
    else:
        denom = step_norm.sum(dim=1).clamp_min(eps).detach()
    closure_ratio_per = sum_norm / denom
    loss = closure_ratio_per.mean()

    raw_flat = errors.reshape(errors.shape[0], errors.shape[1], -1)
    raw_step_norm = raw_flat.norm(dim=-1)
    raw_sum_norm = errors.sum(dim=1).reshape(errors.shape[0], -1).norm(dim=-1)
    raw_closure_ratio = raw_sum_norm / raw_step_norm.sum(dim=1).clamp_min(eps)
    first_rel = relative_l2(pred_seq[:, 0], target_seq[:, 0])
    last_rel = relative_l2(pred_seq[:, -1], target_seq[:, -1])
    logs = {
        "ar/field_peh_enabled": 1.0,
        "ar/field_peh_skipped_short": 0.0,
        "ar/field_peh_loss": float(loss.detach().cpu()),
        "ar/field_peh_lambda": float(getattr(args, "field_peh_lambda", 0.0)),
        "ar/field_peh_weighted_loss": float((_field_peh_effective_lambda(args, epoch) * loss).detach().cpu()),
        "ar/field_peh_horizon": float(K),
        "ar/field_peh_available_horizon": float(T - int(start_t)),
        "ar/field_peh_type": float(type_id),
        "ar/field_peh_closure_ratio": float(raw_closure_ratio.mean().detach().cpu()),
        "ar/field_peh_mean_error_norm": float(raw_step_norm.mean().detach().cpu()),
        "ar/field_peh_sum_error_norm": float(raw_sum_norm.mean().detach().cpu()),
        "ar/field_peh_error_temporal_cos": float(_mean_pairwise_temporal_cos(errors, eps=eps).detach().cpu()),
        "ar/field_peh_first_rel_l2": float(first_rel.detach().cpu()),
        "ar/field_peh_last_rel_l2": float(last_rel.detach().cpu()),
    }
    return loss, logs

def _source_color_zero_logs() -> dict:
    return {
        "ar/source_color_loss": 0.0,
        "ar/source_color_lambda": 0.0,
        "ar/source_color_weighted_loss": 0.0,
        "ar/source_color_horizon": 0.0,
        "ar/source_color_available_horizon": 0.0,
        "ar/source_color_num_sources": 0.0,
        "ar/source_color_repair_num_sources": 0.0,
        "ar/source_color_repair_frac": 0.0,
        "ar/source_color_valid_frac": 0.0,
        "ar/source_color_max_weight": 0.0,
        "ar/source_color_entropy": 0.0,
        "ar/source_color_dominant_index": 0.0,
        "ar/source_color_raw_offdiag_cos2": 0.0,
        "ar/source_color_deltaK_norm": 0.0,
        "ar/source_color_local_rel_l2": 0.0,
        "ar/source_color_enabled": 0.0,
        "ar/source_color_skipped_short": 0.0,
    }


def _error_align_zero_logs() -> dict:
    return {
        "ar/error_align_enabled": 0.0,
        "ar/error_align_skipped_short": 0.0,
        "ar/error_align_horizon": 0.0,
        "ar/error_align_available_horizon": 0.0,
        "ar/error_align_num_times": 0.0,
        "ar/error_align_tf_loss": 0.0,
        "ar/error_align_loss": 0.0,
        "ar/error_align_lambda": 0.0,
        "ar/error_align_weighted_loss": 0.0,
        "ar/error_align_cos_mean": 0.0,
        "ar/error_align_cos_pos_frac": 0.0,
        "ar/error_align_pos_cos_mean": 0.0,
        "ar/error_align_p_norm": 0.0,
        "ar/error_align_b_norm": 0.0,
        "ar/error_align_delta_norm": 0.0,
        "ar/error_align_E_p": 0.0,
        "ar/error_align_E_b": 0.0,
        "ar/error_align_E_cross": 0.0,
        "ar/error_align_E_cross_pos": 0.0,
        "ar/error_align_decomp_residual_frac": 0.0,
    }


def _positive_alignment_loss_from_p_b(
    p: torch.Tensor,
    b: torch.Tensor,
    eps: float = 1e-8,
    min_norm: float = 1e-8,
) -> tuple[torch.Tensor, dict]:
    """Positive alignment loss for fixed notation delta^AR = p + b.

    p = stopgrad(x_AR - x_TF) is the transported previous error direction.
    b = x_TF - x is the trainable teacher-forced one-step error.

    The returned loss penalizes only positive cosine alignment between p and b:
        [ <p,b> / (||p|| ||b|| + eps) ]_+^2.
    """
    p = p.detach()
    pf = p.reshape(p.shape[0], -1)
    bf = b.reshape(b.shape[0], -1)
    dot = (pf * bf).sum(dim=1)
    pn = pf.norm(dim=1)
    bn = bf.norm(dim=1)
    cos = dot / (pn.clamp_min(eps) * bn.clamp_min(eps))
    valid = (pn > min_norm) & (bn > min_norm)
    pos = torch.relu(cos)
    if valid.any():
        loss = pos.pow(2).masked_select(valid).mean()
    else:
        loss = b.new_tensor(0.0)
    logs = {
        "cos_mean": float(cos.detach().mean().cpu()),
        "cos_pos_frac": float(((cos > 0) & valid).float().mean().detach().cpu()),
        "pos_cos_mean": float(pos.detach().mean().cpu()),
        "p_norm": float(pn.detach().mean().cpu()),
        "b_norm": float(bn.detach().mean().cpu()),
    }
    return loss, logs


def compute_error_alignment_tf_loss(raw, state, stim, t0: int, args) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Long no-grad AR context + teacher-forced positive alignment loss.

    Fixed notation used throughout:
        delta_t^AR = x_AR_t - x_t
        p_t        = x_AR_t - x_TF_t       (detached transported previous error)
        b_t        = x_TF_t - x_t          (trainable one-step TF error)
        delta_t^AR = p_t + b_t

    This function does NOT build a long BPTT graph. The long AR rollout is under
    no_grad. Gradients flow only through teacher-forced one-step predictions
    x_TF_t used to compute b_t.
    """
    zero = state.new_tensor(0.0)
    logs0 = _error_align_zero_logs()
    if not bool(getattr(args, "error_align_loss", False)):
        return zero, zero, logs0
    if not hasattr(raw, "step_history"):
        logs0["ar/error_align_enabled"] = 1.0
        return zero, zero, logs0

    B, T, _shape = get_batch_time_shape(state)
    W = int(args.window_size)
    K_req = int(getattr(args, "error_align_horizon", 64))
    K_req = max(1, int(K_req))
    max_available = int(T) - int(t0)
    strict = bool(getattr(args, "error_align_strict_horizon", False))
    if int(t0) - W < 0 or max_available <= 0 or (strict and max_available < K_req):
        logs0["ar/error_align_enabled"] = 1.0
        logs0["ar/error_align_skipped_short"] = 1.0
        logs0["ar/error_align_horizon"] = float(K_req)
        logs0["ar/error_align_available_horizon"] = float(max_available)
        return zero, zero, logs0
    K = max(1, min(K_req, max_available))

    n_times = int(getattr(args, "error_align_num_times", 8))
    n_times = max(1, min(n_times, K))
    stride = max(1, int(getattr(args, "error_align_time_stride", 1)))
    # Offset 0 has p_t approximately zero because both AR and TF branches start
    # from the same clean history. Prefer offsets >= 1 so p_t reflects transported
    # accumulated AR error.
    candidate_offsets = list(range(1, K, stride))
    if not candidate_offsets:
        candidate_offsets = list(range(K))
    n_times = min(n_times, len(candidate_offsets))
    if bool(getattr(args, "error_align_random_times", True)):
        offsets = random.sample(candidate_offsets, n_times)
        offsets.sort()
    else:
        offsets = candidate_offsets[:n_times]

    eps = float(getattr(args, "error_align_eps", 1e-8))
    min_norm = float(getattr(args, "error_align_min_norm", 1e-8))
    tf_loss_name = str(getattr(args, "error_align_tf_loss", getattr(args, "ar_loss", "mse")))

    was_training = bool(getattr(raw, "training", False))
    use_eval = bool(getattr(args, "error_align_eval_mode_rollout", True))
    if use_eval and was_training:
        raw.eval()

    try:
        # Long AR rollout: no gradient, used only to expose x_AR_t and p_t.
        with torch.no_grad():
            hist = time_window(state, int(t0) - W, int(t0)).detach()
            ar_preds = []
            for k in range(K):
                cur = int(t0) + k
                sw = _safe_stim_window(stim, cur - W, cur)
                hist, pred = _ar_step_history_with_stim(raw, hist, sw)
                ar_preds.append(pred.detach())
                hist = hist.detach()
    finally:
        if use_eval and was_training:
            raw.train()

    tf_losses = []
    align_losses = []
    cos_vals = []
    pos_fracs = []
    pos_means = []
    p_norms = []
    b_norms = []
    delta_norms = []
    Eps = []
    Ebs = []
    Ecross = []
    Ecross_pos = []
    residual_fracs = []

    for k in offsets:
        cur = int(t0) + int(k)
        hist_tf = time_window(state, cur - W, cur)
        sw_tf = _safe_stim_window(stim, cur - W, cur)
        x_tf = _ar_forward_pred(raw, sw_tf, hist_tf)       # trainable x_TF_t
        x_gt = time_point(state, cur, keep_time=False)
        x_ar = ar_preds[int(k)].detach()                   # detached x_AR_t

        # Fixed notation.
        b_t = x_tf - x_gt                                  # trainable local/TF error
        p_t = (x_ar - x_tf.detach()).detach()              # detached transported previous error
        delta_ar = (x_ar - x_gt).detach()

        tf_losses.append(_per_sample_state_loss(x_tf, x_gt, tf_loss_name).mean())
        loss_align, alog = _positive_alignment_loss_from_p_b(p_t, b_t, eps=eps, min_norm=min_norm)
        align_losses.append(loss_align)
        cos_vals.append(torch.as_tensor(alog["cos_mean"], device=state.device, dtype=state.dtype))
        pos_fracs.append(torch.as_tensor(alog["cos_pos_frac"], device=state.device, dtype=state.dtype))
        pos_means.append(torch.as_tensor(alog["pos_cos_mean"], device=state.device, dtype=state.dtype))
        p_norms.append(torch.as_tensor(alog["p_norm"], device=state.device, dtype=state.dtype))
        b_norms.append(torch.as_tensor(alog["b_norm"], device=state.device, dtype=state.dtype))

        with torch.no_grad():
            pf = p_t.reshape(B, -1)
            bf = b_t.detach().reshape(B, -1)
            df = delta_ar.reshape(B, -1)
            dot = (pf * bf).sum(dim=1)
            p2 = pf.pow(2).sum(dim=1)
            b2 = bf.pow(2).sum(dim=1)
            d2 = df.pow(2).sum(dim=1).clamp_min(eps)
            delta_norms.append(df.norm(dim=1).mean())
            Eps.append((p2 / d2).mean())
            Ebs.append((b2 / d2).mean())
            Ecross.append((2.0 * dot / d2).mean())
            Ecross_pos.append(torch.relu(2.0 * dot) .div(d2).mean())
            rec = (p_t + b_t.detach() - delta_ar).reshape(B, -1).norm(dim=1)
            residual_fracs.append((rec / df.norm(dim=1).clamp_min(eps)).mean())

    loss_tf = torch.stack(tf_losses).mean() if tf_losses else zero
    loss_align = torch.stack(align_losses).mean() if align_losses else zero

    logs = _error_align_zero_logs()
    logs.update({
        "ar/error_align_enabled": 1.0,
        "ar/error_align_skipped_short": 0.0,
        "ar/error_align_horizon": float(K),
        "ar/error_align_available_horizon": float(max_available),
        "ar/error_align_num_times": float(len(offsets)),
        "ar/error_align_tf_loss": float(loss_tf.detach().cpu()),
        "ar/error_align_loss": float(loss_align.detach().cpu()),
        "ar/error_align_lambda": float(getattr(args, "error_align_lambda", 0.0)),
        "ar/error_align_weighted_loss": float((float(getattr(args, "error_align_lambda", 0.0)) * loss_align).detach().cpu()),
        "ar/error_align_cos_mean": float(torch.stack(cos_vals).mean().detach().cpu()) if cos_vals else 0.0,
        "ar/error_align_cos_pos_frac": float(torch.stack(pos_fracs).mean().detach().cpu()) if pos_fracs else 0.0,
        "ar/error_align_pos_cos_mean": float(torch.stack(pos_means).mean().detach().cpu()) if pos_means else 0.0,
        "ar/error_align_p_norm": float(torch.stack(p_norms).mean().detach().cpu()) if p_norms else 0.0,
        "ar/error_align_b_norm": float(torch.stack(b_norms).mean().detach().cpu()) if b_norms else 0.0,
        "ar/error_align_delta_norm": float(torch.stack(delta_norms).mean().detach().cpu()) if delta_norms else 0.0,
        "ar/error_align_E_p": float(torch.stack(Eps).mean().detach().cpu()) if Eps else 0.0,
        "ar/error_align_E_b": float(torch.stack(Ebs).mean().detach().cpu()) if Ebs else 0.0,
        "ar/error_align_E_cross": float(torch.stack(Ecross).mean().detach().cpu()) if Ecross else 0.0,
        "ar/error_align_E_cross_pos": float(torch.stack(Ecross_pos).mean().detach().cpu()) if Ecross_pos else 0.0,
        "ar/error_align_decomp_residual_frac": float(torch.stack(residual_fracs).mean().detach().cpu()) if residual_fracs else 0.0,
    })
    return loss_tf, loss_align, logs



def _long_error_cloud_zero_logs() -> dict:
    """Zero-valued logs for detached long-horizon error-cloud regularization."""
    return {
        "ar/long_cloud_enabled": 0.0,
        "ar/long_cloud_skipped_short": 0.0,
        "ar/long_cloud_horizon": 0.0,
        "ar/long_cloud_available_horizon": 0.0,
        "ar/long_cloud_num_times": 0.0,
        "ar/long_cloud_offsets_mean": 0.0,
        "ar/long_cloud_loss": 0.0,
        "ar/long_cloud_rein_loss": 0.0,
        "ar/long_cloud_rein_lambda": 0.0,
        "ar/long_cloud_rein_weighted_loss": 0.0,
        "ar/long_cloud_energy_loss": 0.0,
        "ar/long_cloud_energy_lambda": 0.0,
        "ar/long_cloud_energy_rho": 1.0,
        "ar/long_cloud_energy_weighted_loss": 0.0,
        "ar/long_cloud_energy_violation_frac": 0.0,
        "ar/long_cloud_energy_excess_mean": 0.0,
        "ar/long_cloud_rein_cos_mean": 0.0,
        "ar/long_cloud_rein_cos_pos_frac": 0.0,
        "ar/long_cloud_rein_pos_cos_mean": 0.0,
        "ar/long_cloud_orth_loss": 0.0,
        "ar/long_cloud_orth_lambda": 0.0,
        "ar/long_cloud_orth_weighted_loss": 0.0,
        "ar/long_cloud_mean_loss": 0.0,
        "ar/long_cloud_mean_lambda": 0.0,
        "ar/long_cloud_mean_weighted_loss": 0.0,
        "ar/long_cloud_p_norm": 0.0,
        "ar/long_cloud_b_norm": 0.0,
        "ar/long_cloud_delta_norm": 0.0,
        "ar/long_cloud_E_p": 0.0,
        "ar/long_cloud_E_b": 0.0,
        "ar/long_cloud_E_cross": 0.0,
        "ar/long_cloud_E_cross_pos": 0.0,
        "ar/long_cloud_decomp_residual_frac": 0.0,
        "ar/long_cloud_first_rel_l2": 0.0,
        "ar/long_cloud_last_rel_l2": 0.0,
    }


def _parse_long_error_cloud_steps(raw_steps: str, K: int) -> list[int]:
    """Parse one-based horizon steps and return valid zero-based rollout indices.

    User-facing steps are horizon numbers: 1 means the first predicted frame,
    16 means the 16th predicted frame.  Internally ar_preds[0] is step 1, so
    we convert h -> h-1.
    """
    steps: list[int] = []
    raw_steps = str(raw_steps or "").strip()
    if raw_steps:
        for item in raw_steps.replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            try:
                h = int(item)
            except ValueError:
                continue
            if 1 <= h <= int(K):
                idx = h - 1
                if idx not in steps:
                    steps.append(idx)
    return steps



def _forced_damped_zero_logs() -> dict:
    """Zero logs for Forced Damped Error Dynamics (FDED)."""
    return {
        "ar/fded_enabled": 0.0,
        "ar/fded_skipped_short": 0.0,
        "ar/fded_horizon": 0.0,
        "ar/fded_available_horizon": 0.0,
        "ar/fded_num_times": 0.0,
        "ar/fded_offsets_mean": 0.0,
        "ar/fded_loss": 0.0,
        "ar/fded_energy_loss": 0.0,
        "ar/fded_energy_lambda": 0.0,
        "ar/fded_energy_weighted_loss": 0.0,
        "ar/fded_damping_loss": 0.0,
        "ar/fded_damping_lambda": 0.0,
        "ar/fded_damping_weighted_loss": 0.0,
        "ar/fded_accel_loss": 0.0,
        "ar/fded_accel_lambda": 0.0,
        "ar/fded_accel_weighted_loss": 0.0,
        "ar/fded_input_beta": 0.0,
        "ar/fded_input_work_mean": 0.0,
        "ar/fded_energy_growth_mean": 0.0,
        "ar/fded_energy_excess_mean": 0.0,
        "ar/fded_energy_violation_frac": 0.0,
        "ar/fded_damping_cos_mean": 0.0,
        "ar/fded_damping_pos_frac": 0.0,
        "ar/fded_accel_cos_mean": 0.0,
        "ar/fded_accel_pos_frac": 0.0,
        "ar/fded_e_prev_norm": 0.0,
        "ar/fded_e_next_norm": 0.0,
        "ar/fded_velocity_norm": 0.0,
        "ar/fded_accel_norm": 0.0,
        "ar/fded_first_rel_l2": 0.0,
        "ar/fded_last_rel_l2": 0.0,
    }


def _stimulus_step_work(stim: torch.Tensor | None, cur: int, eps: float) -> torch.Tensor | None:
    """Return per-sample squared stimulus change ||u_cur-u_{cur-1}||^2.

    This is only a lightweight external-forcing budget.  It is normalized inside
    compute_forced_damped_error_dynamics_loss, so its absolute feature scale is
    not used directly.
    """
    if stim is None or cur <= 0 or cur >= int(stim.shape[1]):
        return None
    u0 = time_point(stim, cur - 1, keep_time=False)
    u1 = time_point(stim, cur, keep_time=False)
    df = (u1 - u0).reshape(u1.shape[0], -1)
    return df.pow(2).mean(dim=1).detach().clamp_min(eps)


def _compute_forced_damped_local_edge_loss(raw, state, stim, t0: int, args) -> tuple[torch.Tensor, dict]:
    """BPTT-free local-edge Forced Damped Error Dynamics regularizer.

    The long rollout is used only to expose the AR state distribution and is
    detached.  Gradients flow through fresh one-step predictions from detached
    AR histories at sampled offsets.  This keeps the objective close to rollout
    error dynamics without retaining a long BPTT graph.

    Error-state analogy on an edge ending at time cur:
        e_prev = x_AR(cur-1) - x_GT(cur-1)          # error position before edge
        e_next = F_theta(sg[AR history], u) - x_GT(cur)
        v      = e_next - e_prev                   # error velocity
        a      = v - (e_prev - e_prevprev)          # error acceleration

    The losses penalize input-unexplained positive error-energy growth,
    insufficient damping, and outward error acceleration.
    """
    zero = state.new_tensor(0.0)
    logs0 = _forced_damped_zero_logs()
    if not bool(getattr(args, "forced_damped_error_loss", False)):
        return zero, logs0
    if not hasattr(raw, "step_history"):
        logs0["ar/fded_enabled"] = 1.0
        return zero, logs0

    B, T, _shape = get_batch_time_shape(state)
    W = int(args.window_size)
    K_req = max(1, int(getattr(args, "forced_damped_horizon", 64)))
    # Need target cur and previous target cur-1.  Candidate cur=t0+k, k in [0,K-1].
    max_available = int(T) - int(t0)
    strict = bool(getattr(args, "forced_damped_strict_horizon", False))
    if int(t0) - W < 0 or max_available <= 1 or (strict and max_available < K_req):
        logs0.update({
            "ar/fded_enabled": 1.0,
            "ar/fded_skipped_short": 1.0,
            "ar/fded_horizon": float(K_req),
            "ar/fded_available_horizon": float(max_available),
        })
        return zero, logs0
    K = max(1, min(K_req, max_available))

    explicit = _parse_long_error_cloud_steps(str(getattr(args, "forced_damped_offsets", "8,16,32,48,64")), K)
    # Offset 0 is usually nearly clean because the rollout starts from GT history;
    # prefer offset >=1 so the loss sees real AR-distribution error.
    explicit = [k for k in explicit if 1 <= int(k) < K]
    if explicit:
        offsets = explicit
    else:
        stride = max(1, int(getattr(args, "forced_damped_time_stride", 8)))
        candidate = list(range(1, K, stride))
        if not candidate:
            candidate = list(range(1, K))
        n_times = max(1, min(int(getattr(args, "forced_damped_num_times", 5)), len(candidate)))
        if bool(getattr(args, "forced_damped_random_times", False)) and len(candidate) > n_times:
            offsets = random.sample(candidate, n_times)
            offsets.sort()
        else:
            offsets = candidate[:n_times]
    offsets = [int(k) for k in offsets if 1 <= int(k) < K]
    if not offsets:
        logs0.update({
            "ar/fded_enabled": 1.0,
            "ar/fded_skipped_short": 1.0,
            "ar/fded_horizon": float(K),
            "ar/fded_available_horizon": float(max_available),
        })
        return zero, logs0

    eps = float(getattr(args, "forced_damped_eps", 1e-8))
    min_norm = float(getattr(args, "forced_damped_min_norm", 1e-8))
    lam_energy = float(getattr(args, "forced_damped_energy_lambda", 0.0))
    lam_damping = float(getattr(args, "forced_damped_damping_lambda", 0.0))
    lam_accel = float(getattr(args, "forced_damped_accel_lambda", 0.0))
    input_beta = float(getattr(args, "forced_damped_input_beta", 0.0))
    damping_margin = float(getattr(args, "forced_damped_damping_margin", 0.0))
    accel_margin = float(getattr(args, "forced_damped_accel_margin", 0.0))

    was_training = bool(getattr(raw, "training", False))
    use_eval = bool(getattr(args, "forced_damped_eval_mode_rollout", True))
    hist_by_offset: dict[int, torch.Tensor] = {}
    first_rel = None
    last_rel = None
    try:
        if use_eval and was_training:
            raw.eval()
        with torch.no_grad():
            hist = time_window(state, int(t0) - W, int(t0)).detach()
            offset_set = set(offsets)
            for k in range(K):
                cur = int(t0) + int(k)
                if k in offset_set:
                    hist_by_offset[k] = hist.detach()
                sw = _safe_stim_window(stim, cur - W, cur)
                hist, pred = _ar_step_history_with_stim(raw, hist, sw)
                pred = pred.detach()
                target = time_point(state, cur, keep_time=False).detach()
                rel = relative_l2(pred, target).detach()
                if first_rel is None:
                    first_rel = rel
                last_rel = rel
                hist = hist.detach()
    finally:
        if use_eval and was_training:
            raw.train()

    rel_growth_terms: list[torch.Tensor] = []
    input_work_terms: list[torch.Tensor] = []
    damping_losses: list[torch.Tensor] = []
    accel_losses: list[torch.Tensor] = []
    damping_cos_vals: list[torch.Tensor] = []
    damping_pos_fracs: list[torch.Tensor] = []
    accel_cos_vals: list[torch.Tensor] = []
    accel_pos_fracs: list[torch.Tensor] = []
    e_prev_norms: list[torch.Tensor] = []
    e_next_norms: list[torch.Tensor] = []
    vel_norms: list[torch.Tensor] = []
    accel_norms: list[torch.Tensor] = []

    for k in offsets:
        hist_det = hist_by_offset.get(int(k))
        if hist_det is None:
            continue
        cur = int(t0) + int(k)
        if cur <= 0 or cur >= T:
            continue
        sw = _safe_stim_window(stim, cur - W, cur)
        x_pred = _ar_forward_pred(raw, sw, hist_det.detach())
        x_gt = time_point(state, cur, keep_time=False)
        x_prev_gt = time_point(state, cur - 1, keep_time=False)

        e_prev = hist_det[:, -1].detach() - x_prev_gt.detach()
        e_next = x_pred - x_gt
        v = e_next - e_prev

        epf = e_prev.reshape(B, -1)
        enf = e_next.reshape(B, -1)
        vf = v.reshape(B, -1)
        E_prev = 0.5 * epf.pow(2).sum(dim=1).detach()
        E_next = 0.5 * enf.pow(2).sum(dim=1)
        rel_growth = (E_next - E_prev) / E_prev.clamp_min(eps)
        rel_growth_terms.append(rel_growth)

        work = _stimulus_step_work(stim, cur, eps)
        if work is None:
            work = torch.zeros(B, device=state.device, dtype=state.dtype)
        else:
            work = work.to(device=state.device, dtype=state.dtype)
        input_work_terms.append(work)

        en = epf.norm(dim=1)
        vn = vf.norm(dim=1)
        valid = (en > min_norm) & (vn > min_norm)
        cos_d = (epf * vf).sum(dim=1) / (en.clamp_min(eps) * vn.clamp_min(eps))
        damp_hinge = torch.relu(cos_d + damping_margin)
        damping_losses.append(damp_hinge.pow(2).masked_select(valid).mean() if valid.any() else zero)
        damping_cos_vals.append(cos_d.detach().mean())
        damping_pos_fracs.append(((cos_d + damping_margin > 0) & valid).to(state.dtype).mean().detach())

        if W >= 2 and cur >= 2:
            x_prevprev_gt = time_point(state, cur - 2, keep_time=False)
            e_prevprev = hist_det[:, -2].detach() - x_prevprev_gt.detach()
            v_prev = e_prev - e_prevprev
            acc = v - v_prev
            af = acc.reshape(B, -1)
            an = af.norm(dim=1)
            valid_a = (en > min_norm) & (an > min_norm)
            cos_a = (epf * af).sum(dim=1) / (en.clamp_min(eps) * an.clamp_min(eps))
            acc_hinge = torch.relu(cos_a + accel_margin)
            accel_losses.append(acc_hinge.pow(2).masked_select(valid_a).mean() if valid_a.any() else zero)
            accel_cos_vals.append(cos_a.detach().mean())
            accel_pos_fracs.append(((cos_a + accel_margin > 0) & valid_a).to(state.dtype).mean().detach())
            accel_norms.append(an.detach().mean())
        else:
            accel_losses.append(zero)
            accel_cos_vals.append(zero.detach())
            accel_pos_fracs.append(zero.detach())
            accel_norms.append(zero.detach())

        e_prev_norms.append(en.detach().mean())
        e_next_norms.append(enf.detach().norm(dim=1).mean())
        vel_norms.append(vn.detach().mean())

    if not rel_growth_terms:
        logs0.update({
            "ar/fded_enabled": 1.0,
            "ar/fded_skipped_short": 1.0,
            "ar/fded_horizon": float(K),
            "ar/fded_available_horizon": float(max_available),
        })
        return zero, logs0

    rel_growth = torch.stack(rel_growth_terms, dim=0)  # [S,B]
    input_work = torch.stack(input_work_terms, dim=0).detach()
    input_work_norm = input_work / input_work.mean().clamp_min(eps)
    energy_excess = rel_growth - input_beta * input_work_norm
    energy_loss = torch.relu(energy_excess).pow(2).mean()
    damping_loss = torch.stack(damping_losses).mean() if damping_losses else zero
    accel_loss = torch.stack(accel_losses).mean() if accel_losses else zero
    loss = lam_energy * energy_loss + lam_damping * damping_loss + lam_accel * accel_loss

    with torch.no_grad():
        logs = _forced_damped_zero_logs()
        logs.update({
            "ar/fded_enabled": 1.0,
            "ar/fded_skipped_short": 0.0,
            "ar/fded_horizon": float(K),
            "ar/fded_available_horizon": float(max_available),
            "ar/fded_num_times": float(len(rel_growth_terms)),
            "ar/fded_offsets_mean": float(sum(k + 1 for k in offsets) / max(1, len(offsets))),
            "ar/fded_loss": float(loss.detach().cpu()),
            "ar/fded_energy_loss": float(energy_loss.detach().cpu()),
            "ar/fded_energy_lambda": float(lam_energy),
            "ar/fded_energy_weighted_loss": float((lam_energy * energy_loss).detach().cpu()),
            "ar/fded_damping_loss": float(damping_loss.detach().cpu()),
            "ar/fded_damping_lambda": float(lam_damping),
            "ar/fded_damping_weighted_loss": float((lam_damping * damping_loss).detach().cpu()),
            "ar/fded_accel_loss": float(accel_loss.detach().cpu()),
            "ar/fded_accel_lambda": float(lam_accel),
            "ar/fded_accel_weighted_loss": float((lam_accel * accel_loss).detach().cpu()),
            "ar/fded_input_beta": float(input_beta),
            "ar/fded_input_work_mean": float(input_work.mean().detach().cpu()),
            "ar/fded_energy_growth_mean": float(rel_growth.mean().detach().cpu()),
            "ar/fded_energy_excess_mean": float(energy_excess.mean().detach().cpu()),
            "ar/fded_energy_violation_frac": float((energy_excess > 0).to(state.dtype).mean().detach().cpu()),
            "ar/fded_damping_cos_mean": float(torch.stack(damping_cos_vals).mean().detach().cpu()) if damping_cos_vals else 0.0,
            "ar/fded_damping_pos_frac": float(torch.stack(damping_pos_fracs).mean().detach().cpu()) if damping_pos_fracs else 0.0,
            "ar/fded_accel_cos_mean": float(torch.stack(accel_cos_vals).mean().detach().cpu()) if accel_cos_vals else 0.0,
            "ar/fded_accel_pos_frac": float(torch.stack(accel_pos_fracs).mean().detach().cpu()) if accel_pos_fracs else 0.0,
            "ar/fded_e_prev_norm": float(torch.stack(e_prev_norms).mean().detach().cpu()) if e_prev_norms else 0.0,
            "ar/fded_e_next_norm": float(torch.stack(e_next_norms).mean().detach().cpu()) if e_next_norms else 0.0,
            "ar/fded_velocity_norm": float(torch.stack(vel_norms).mean().detach().cpu()) if vel_norms else 0.0,
            "ar/fded_accel_norm": float(torch.stack(accel_norms).mean().detach().cpu()) if accel_norms else 0.0,
            "ar/fded_first_rel_l2": float(first_rel.detach().cpu()) if first_rel is not None else 0.0,
            "ar/fded_last_rel_l2": float(last_rel.detach().cpu()) if last_rel is not None else 0.0,
        })
    return loss, logs


def _stimulus_cumulative_work(stim: torch.Tensor | None, start_cur: int, end_cur: int, B: int, device, dtype, eps: float) -> torch.Tensor:
    """Cumulative external-forcing budget over [start_cur, end_cur].

    The returned quantity is a per-sample scalar.  Its absolute scale is not
    important because the caller normalizes it across sampled windows before
    applying forced_damped_input_beta.
    """
    work = torch.zeros(B, device=device, dtype=dtype)
    if stim is None:
        return work
    for cur in range(int(start_cur), int(end_cur) + 1):
        step_work = _stimulus_step_work(stim, cur, eps)
        if step_work is not None:
            work = work + step_work.to(device=device, dtype=dtype)
    return work.detach().clamp_min(0.0)


def compute_forced_damped_error_dynamics_loss(raw, state, stim, t0: int, args) -> tuple[torch.Tensor, dict]:
    """Endpoint-window Net Error-energy Drift (NED) regularizer.

    This is the cleaner version of the forced-damped error-dynamics idea.
    Instead of penalizing every local edge E_{t+1}-E_t, it penalizes the net
    window drift

        [ E_{t+K} - E_t - beta * W^u_{t:t+K} ]_+,

    where E_t = 0.5 ||x_hat_t - x_t||^2 and W^u is a lightweight external
    stimulus-work budget.  This matches the coboundary/potential motivation:
    reversible local exchanges may cancel inside the window, while persistent
    net error-energy injection remains.

    To make the endpoint objective trainable, we first run a no-grad rollout to
    get a detached AR-distribution start state, then run a short graph rollout
    from that detached state to the endpoint.  Thus gradients flow through the
    endpoint prediction without retaining a full long rollout graph.

    Set --no-forced_damped_endpoint_window to fall back to the older local-edge
    FDED loss for ablations.
    """
    if not bool(getattr(args, "forced_damped_endpoint_window", True)):
        return _compute_forced_damped_local_edge_loss(raw, state, stim, t0, args)

    zero = state.new_tensor(0.0)
    logs0 = _forced_damped_zero_logs()
    if not bool(getattr(args, "forced_damped_error_loss", False)):
        return zero, logs0
    if not hasattr(raw, "step_history"):
        logs0["ar/fded_enabled"] = 1.0
        return zero, logs0

    B, T, _shape = get_batch_time_shape(state)
    W = int(args.window_size)
    K_req = max(2, int(getattr(args, "forced_damped_horizon", 8)))
    max_available = int(T) - int(t0)
    strict = bool(getattr(args, "forced_damped_strict_horizon", False))
    if int(t0) - W < 0 or max_available <= 2 or (strict and max_available < K_req):
        logs0.update({
            "ar/fded_enabled": 1.0,
            "ar/fded_skipped_short": 1.0,
            "ar/fded_horizon": float(K_req),
            "ar/fded_available_horizon": float(max_available),
        })
        logs0["ar/fded_endpoint_window"] = 1.0
        return zero, logs0
    K = max(2, min(K_req, max_available))

    explicit = _parse_long_error_cloud_steps(str(getattr(args, "forced_damped_offsets", "8")), K)
    # In endpoint mode, each offset is an endpoint horizon h.  h=1 is too short
    # for net-drift, so require zero-based endpoint k>=1.
    explicit = [k for k in explicit if 1 <= int(k) < K]
    if explicit:
        end_offsets = explicit
    else:
        stride = max(1, int(getattr(args, "forced_damped_time_stride", K)))
        candidate = list(range(1, K, stride))
        if (K - 1) not in candidate:
            candidate.append(K - 1)
        candidate = sorted(set(candidate))
        n_times = max(1, min(int(getattr(args, "forced_damped_num_times", 1)), len(candidate)))
        if bool(getattr(args, "forced_damped_random_times", False)) and len(candidate) > n_times:
            end_offsets = random.sample(candidate, n_times)
            end_offsets.sort()
        else:
            end_offsets = candidate[:n_times]
    end_offsets = [int(k) for k in end_offsets if 1 <= int(k) < K]
    if not end_offsets:
        logs0.update({
            "ar/fded_enabled": 1.0,
            "ar/fded_skipped_short": 1.0,
            "ar/fded_horizon": float(K),
            "ar/fded_available_horizon": float(max_available),
        })
        logs0["ar/fded_endpoint_window"] = 1.0
        return zero, logs0

    eps = float(getattr(args, "forced_damped_eps", 1e-8))
    lam_energy = float(getattr(args, "forced_damped_energy_lambda", 0.0))
    input_beta = float(getattr(args, "forced_damped_input_beta", 0.0))
    bptt_span = max(1, int(getattr(args, "forced_damped_bptt_span", K)))

    # For each endpoint k, start from a detached AR state at start_k and run a
    # short graph rollout to k.  start_k>=1 ensures the start error is an actual
    # rollout error rather than the zero GT-history error at the beginning.
    window_pairs: list[tuple[int, int]] = []
    start_set: set[int] = set()
    for end_k in end_offsets:
        start_k = max(1, int(end_k) - int(bptt_span) + 1)
        if start_k >= int(end_k):
            start_k = max(1, int(end_k) - 1)
        window_pairs.append((start_k, int(end_k)))
        start_set.add(start_k)

    hist_by_start: dict[int, torch.Tensor] = {}
    first_rel = None
    last_rel = None
    was_training = bool(getattr(raw, "training", False))
    use_eval = bool(getattr(args, "forced_damped_eval_mode_rollout", True))
    try:
        if use_eval and was_training:
            raw.eval()
        with torch.no_grad():
            hist = time_window(state, int(t0) - W, int(t0)).detach()
            for k in range(K):
                cur = int(t0) + int(k)
                if k in start_set:
                    hist_by_start[k] = hist.detach()
                sw = _safe_stim_window(stim, cur - W, cur)
                hist, pred = _ar_step_history_with_stim(raw, hist, sw)
                pred = pred.detach()
                target = time_point(state, cur, keep_time=False).detach()
                rel = relative_l2(pred, target).detach()
                if first_rel is None:
                    first_rel = rel
                last_rel = rel
                hist = hist.detach()
    finally:
        if use_eval and was_training:
            raw.train()

    rel_growth_terms: list[torch.Tensor] = []
    input_work_terms: list[torch.Tensor] = []
    e_start_norms: list[torch.Tensor] = []
    e_end_norms: list[torch.Tensor] = []
    span_vals: list[int] = []

    for start_k, end_k in window_pairs:
        hist_det = hist_by_start.get(int(start_k))
        if hist_det is None:
            continue
        start_err_time = int(t0) + int(start_k) - 1
        end_cur = int(t0) + int(end_k)
        if start_err_time < 0 or end_cur >= T:
            continue

        x_start_gt = time_point(state, start_err_time, keep_time=False).detach()
        e_start = hist_det[:, -1].detach() - x_start_gt
        esf = e_start.reshape(B, -1)
        E_start = 0.5 * esf.pow(2).sum(dim=1).detach()

        graph_hist = hist_det.detach()
        pred_end = None
        for kk in range(int(start_k), int(end_k) + 1):
            cur = int(t0) + int(kk)
            sw = _safe_stim_window(stim, cur - W, cur)
            graph_hist, pred_end = _ar_step_history_with_stim(raw, graph_hist, sw)
        if pred_end is None:
            continue
        x_end_gt = time_point(state, end_cur, keep_time=False)
        e_end = pred_end - x_end_gt
        eef = e_end.reshape(B, -1)
        E_end = 0.5 * eef.pow(2).sum(dim=1)

        rel_growth = (E_end - E_start) / E_start.clamp_min(eps)
        rel_growth_terms.append(rel_growth)
        input_work_terms.append(_stimulus_cumulative_work(stim, int(t0) + int(start_k), end_cur, B, state.device, state.dtype, eps))
        e_start_norms.append(esf.norm(dim=1).detach().mean())
        e_end_norms.append(eef.detach().norm(dim=1).mean())
        span_vals.append(int(end_k) - int(start_k) + 1)

    if not rel_growth_terms:
        logs0.update({
            "ar/fded_enabled": 1.0,
            "ar/fded_skipped_short": 1.0,
            "ar/fded_horizon": float(K),
            "ar/fded_available_horizon": float(max_available),
        })
        logs0["ar/fded_endpoint_window"] = 1.0
        return zero, logs0

    rel_growth = torch.stack(rel_growth_terms, dim=0)  # [S,B]
    input_work = torch.stack(input_work_terms, dim=0).detach()
    input_work_norm = input_work / input_work.mean().clamp_min(eps)
    energy_excess = rel_growth - input_beta * input_work_norm
    energy_loss = torch.relu(energy_excess).pow(2).mean()
    loss = lam_energy * energy_loss

    with torch.no_grad():
        logs = _forced_damped_zero_logs()
        logs.update({
            "ar/fded_enabled": 1.0,
            "ar/fded_endpoint_window": 1.0,
            "ar/fded_skipped_short": 0.0,
            "ar/fded_horizon": float(K),
            "ar/fded_available_horizon": float(max_available),
            "ar/fded_bptt_span": float(sum(span_vals) / max(1, len(span_vals))),
            "ar/fded_num_times": float(len(rel_growth_terms)),
            "ar/fded_offsets_mean": float(sum(k + 1 for k in end_offsets) / max(1, len(end_offsets))),
            "ar/fded_loss": float(loss.detach().cpu()),
            "ar/fded_energy_loss": float(energy_loss.detach().cpu()),
            "ar/fded_energy_lambda": float(lam_energy),
            "ar/fded_energy_weighted_loss": float((lam_energy * energy_loss).detach().cpu()),
            "ar/fded_damping_loss": 0.0,
            "ar/fded_damping_lambda": float(getattr(args, "forced_damped_damping_lambda", 0.0)),
            "ar/fded_damping_weighted_loss": 0.0,
            "ar/fded_accel_loss": 0.0,
            "ar/fded_accel_lambda": float(getattr(args, "forced_damped_accel_lambda", 0.0)),
            "ar/fded_accel_weighted_loss": 0.0,
            "ar/fded_input_beta": float(input_beta),
            "ar/fded_input_work_mean": float(input_work.mean().detach().cpu()),
            "ar/fded_energy_growth_mean": float(rel_growth.mean().detach().cpu()),
            "ar/fded_energy_excess_mean": float(energy_excess.mean().detach().cpu()),
            "ar/fded_energy_violation_frac": float((energy_excess > 0).to(state.dtype).mean().detach().cpu()),
            "ar/fded_e_prev_norm": float(torch.stack(e_start_norms).mean().detach().cpu()) if e_start_norms else 0.0,
            "ar/fded_e_next_norm": float(torch.stack(e_end_norms).mean().detach().cpu()) if e_end_norms else 0.0,
            "ar/fded_first_rel_l2": float(first_rel.detach().cpu()) if first_rel is not None else 0.0,
            "ar/fded_last_rel_l2": float(last_rel.detach().cpu()) if last_rel is not None else 0.0,
        })
    return loss, logs

def compute_long_error_cloud_loss(raw, state, stim, t0: int, args) -> tuple[torch.Tensor, dict]:
    """Detached long-rollout error-cloud loss.

    This separates two roles:

      * Short BPTT, handled elsewhere, trains local/short-horizon accuracy.
      * This function probes a long AR rollout under no_grad and regularizes
        the local teacher-forced residuals at long offsets.

    For a sampled long horizon step k (one-based h, zero-based index k=h-1):

        delta_k^AR = x_k^AR - x_k
        p_k        = stopgrad(x_k^AR - x_k^TF)
        b_k        = x_k^TF - x_k
        delta_k^AR = p_k + b_k

    The preferred closed-loop energy term penalizes

        [ ||p_k+b_k||^2 - ||p_k||^2 - rho ||b_k||^2 ]_+^2,

    after normalization by ||p_k||^2+||b_k||^2.  rho=1 only suppresses
    positive cross-energy p_k^T b_k; rho<1 asks the local residual to have a
    partial repair effect.

    x_k^AR is produced by a no-gradient K_long rollout.  x_k^TF is a local
    teacher-forced prediction with gradient.  Therefore this loss sees long
    accumulated error directions without storing a K_long-step autograd graph.
    """
    zero = state.new_tensor(0.0)
    logs0 = _long_error_cloud_zero_logs()
    if not bool(getattr(args, "long_error_cloud_loss", False)):
        return zero, logs0
    if not hasattr(raw, "step_history"):
        logs0["ar/long_cloud_enabled"] = 1.0
        return zero, logs0

    B, T, _shape = get_batch_time_shape(state)
    W = int(args.window_size)
    K_req = max(1, int(getattr(args, "long_error_cloud_horizon", 64)))
    max_available = int(T) - int(t0)
    strict = bool(getattr(args, "long_error_cloud_strict_horizon", False))
    if int(t0) - W < 0 or max_available <= 0 or (strict and max_available < K_req):
        logs0["ar/long_cloud_enabled"] = 1.0
        logs0["ar/long_cloud_skipped_short"] = 1.0
        logs0["ar/long_cloud_horizon"] = float(K_req)
        logs0["ar/long_cloud_available_horizon"] = float(max_available)
        return zero, logs0

    K = max(1, min(K_req, max_available))
    stride = max(1, int(getattr(args, "long_error_cloud_time_stride", 8)))
    explicit_steps = _parse_long_error_cloud_steps(getattr(args, "long_error_cloud_offsets", ""), K)
    if explicit_steps:
        offsets = explicit_steps
    else:
        # Prefer steps beyond the short-BPTT horizon so this objective is not a
        # duplicate of the short rollout loss.
        min_h = max(1, int(getattr(args, "bptt_horizon", 8)) + 1)
        candidate_steps = list(range(min_h, K + 1, stride))
        if not candidate_steps:
            candidate_steps = list(range(1, K + 1, stride))
        n_times = max(1, min(int(getattr(args, "long_error_cloud_num_times", 5)), len(candidate_steps)))
        if bool(getattr(args, "long_error_cloud_random_times", False)) and len(candidate_steps) > n_times:
            chosen_steps = random.sample(candidate_steps, n_times)
            chosen_steps.sort()
        else:
            chosen_steps = candidate_steps[:n_times]
        offsets = [h - 1 for h in chosen_steps]

    offsets = [int(k) for k in offsets if 0 <= int(k) < K]
    if not offsets:
        logs0["ar/long_cloud_enabled"] = 1.0
        logs0["ar/long_cloud_skipped_short"] = 1.0
        logs0["ar/long_cloud_horizon"] = float(K)
        logs0["ar/long_cloud_available_horizon"] = float(max_available)
        return zero, logs0

    rein_lam = float(getattr(args, "long_error_cloud_rein_lambda", 0.0))
    energy_lam = float(getattr(args, "long_error_cloud_energy_lambda", 0.0))
    energy_rho = float(getattr(args, "long_error_cloud_energy_rho", 1.0))
    orth_lam = float(getattr(args, "long_error_cloud_orth_lambda", 0.0))
    mean_lam = float(getattr(args, "long_error_cloud_mean_lambda", 0.0))
    eps = float(getattr(args, "long_error_cloud_eps", 1e-8))
    min_norm = float(getattr(args, "long_error_cloud_min_norm", 1e-8))

    was_training = bool(getattr(raw, "training", False))
    use_eval = bool(getattr(args, "long_error_cloud_eval_mode_rollout", True))
    if use_eval and was_training:
        raw.eval()

    try:
        with torch.no_grad():
            hist = time_window(state, int(t0) - W, int(t0)).detach()
            ar_preds = []
            first_rel = None
            last_rel = None
            for k in range(K):
                cur = int(t0) + k
                sw = _safe_stim_window(stim, cur - W, cur)
                hist, pred = _ar_step_history_with_stim(raw, hist, sw)
                pred = pred.detach()
                ar_preds.append(pred)
                target = time_point(state, cur, keep_time=False).detach()
                rel = relative_l2(pred, target).detach()
                if first_rel is None:
                    first_rel = rel
                last_rel = rel
                hist = hist.detach()
    finally:
        if use_eval and was_training:
            raw.train()

    b_list: list[torch.Tensor] = []
    p_list: list[torch.Tensor] = []
    delta_list: list[torch.Tensor] = []
    rein_losses: list[torch.Tensor] = []
    energy_losses: list[torch.Tensor] = []
    energy_violation_fracs: list[torch.Tensor] = []
    energy_excess_means: list[torch.Tensor] = []
    cos_vals: list[torch.Tensor] = []
    pos_fracs: list[torch.Tensor] = []
    pos_means: list[torch.Tensor] = []

    for k in offsets:
        cur = int(t0) + int(k)
        hist_tf = time_window(state, cur - W, cur)
        sw_tf = _safe_stim_window(stim, cur - W, cur)
        x_tf = _ar_forward_pred(raw, sw_tf, hist_tf)       # trainable local branch
        x_gt = time_point(state, cur, keep_time=False)
        x_ar = ar_preds[int(k)].detach()                   # detached long AR probe

        b_t = x_tf - x_gt
        p_t = (x_ar - x_tf.detach()).detach()
        delta_t = (x_ar - x_gt).detach()

        b_list.append(b_t)
        p_list.append(p_t)
        delta_list.append(delta_t)

        rein_loss, rlog = _positive_alignment_loss_from_p_b(p_t, b_t, eps=eps, min_norm=min_norm)
        rein_losses.append(rein_loss)
        cos_vals.append(torch.as_tensor(rlog["cos_mean"], device=state.device, dtype=state.dtype))
        pos_fracs.append(torch.as_tensor(rlog["cos_pos_frac"], device=state.device, dtype=state.dtype))
        pos_means.append(torch.as_tensor(rlog["pos_cos_mean"], device=state.device, dtype=state.dtype))

        # Closed-loop energy anti-reinforcement.  Unlike cosine-only rein, this
        # term is tied directly to the exact decomposition
        #   delta^AR = p + b
        # and penalizes cases where the trainable local residual b increases the
        # current closed-loop error energy beyond ||p||^2 + rho ||b||^2.
        pf = p_t.reshape(B, -1)
        bf = b_t.reshape(B, -1)
        p2_s = pf.pow(2).sum(dim=1)
        b2_s = bf.pow(2).sum(dim=1)
        delta2_s = (pf + bf).pow(2).sum(dim=1)
        denom_s = (p2_s + b2_s).clamp_min(eps)
        energy_excess = (delta2_s - p2_s - energy_rho * b2_s) / denom_s
        energy_hinge = torch.relu(energy_excess)
        energy_losses.append(energy_hinge.pow(2).mean())
        energy_violation_fracs.append((energy_excess > 0).to(state.dtype).mean())
        energy_excess_means.append(energy_excess.mean().detach())

    if not b_list:
        return zero, logs0

    b_flat = torch.stack([b.reshape(B, -1) for b in b_list], dim=1)       # [B,S,D]
    p_flat = torch.stack([p.reshape(B, -1) for p in p_list], dim=1)       # [B,S,D]
    d_flat = torch.stack([d.reshape(B, -1) for d in delta_list], dim=1)   # [B,S,D]
    bn = b_flat.norm(dim=-1)
    pn = p_flat.norm(dim=-1)
    dn = d_flat.norm(dim=-1).clamp_min(eps)
    valid = bn > min_norm

    reinforce_loss = torch.stack(rein_losses).mean() if rein_losses else zero
    energy_loss = torch.stack(energy_losses).mean() if energy_losses else zero
    energy_violation_frac = torch.stack(energy_violation_fracs).mean() if energy_violation_fracs else zero
    energy_excess_mean = torch.stack(energy_excess_means).mean() if energy_excess_means else zero

    u = b_flat / bn.clamp_min(eps).unsqueeze(-1)
    S_eff = int(b_flat.shape[1])
    G = torch.bmm(u, u.transpose(1, 2))
    eye = torch.eye(S_eff, device=state.device, dtype=torch.bool).unsqueeze(0)
    valid_pair = valid.unsqueeze(1) & valid.unsqueeze(2) & (~eye)
    if S_eff > 1 and valid_pair.any():
        orth_loss = G.pow(2).masked_select(valid_pair).mean()
    else:
        orth_loss = zero

    valid_f = valid.to(u.dtype)
    denom = valid_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean_vec = (u * valid_f.unsqueeze(-1)).sum(dim=1) / denom
    mean_loss = mean_vec.pow(2).sum(dim=-1).mean()

    loss = energy_lam * energy_loss + rein_lam * reinforce_loss + orth_lam * orth_loss + mean_lam * mean_loss

    with torch.no_grad():
        bf_det = b_flat.detach()
        dot_pb = (p_flat * bf_det).sum(dim=-1)
        p2 = p_flat.pow(2).sum(dim=-1)
        b2 = bf_det.pow(2).sum(dim=-1)
        d2 = d_flat.pow(2).sum(dim=-1).clamp_min(eps)
        rec = (p_flat + bf_det - d_flat).norm(dim=-1)
        logs = _long_error_cloud_zero_logs()
        logs.update({
            "ar/long_cloud_enabled": 1.0,
            "ar/long_cloud_skipped_short": 0.0,
            "ar/long_cloud_horizon": float(K),
            "ar/long_cloud_available_horizon": float(max_available),
            "ar/long_cloud_num_times": float(len(offsets)),
            "ar/long_cloud_offsets_mean": float(sum(k + 1 for k in offsets) / max(1, len(offsets))),
            "ar/long_cloud_loss": float(loss.detach().cpu()),
            "ar/long_cloud_rein_loss": float(reinforce_loss.detach().cpu()),
            "ar/long_cloud_rein_lambda": float(rein_lam),
            "ar/long_cloud_rein_weighted_loss": float((rein_lam * reinforce_loss).detach().cpu()),
            "ar/long_cloud_energy_loss": float(energy_loss.detach().cpu()),
            "ar/long_cloud_energy_lambda": float(energy_lam),
            "ar/long_cloud_energy_rho": float(energy_rho),
            "ar/long_cloud_energy_weighted_loss": float((energy_lam * energy_loss).detach().cpu()),
            "ar/long_cloud_energy_violation_frac": float(energy_violation_frac.detach().cpu()),
            "ar/long_cloud_energy_excess_mean": float(energy_excess_mean.detach().cpu()),
            "ar/long_cloud_rein_cos_mean": float(torch.stack(cos_vals).mean().detach().cpu()) if cos_vals else 0.0,
            "ar/long_cloud_rein_cos_pos_frac": float(torch.stack(pos_fracs).mean().detach().cpu()) if pos_fracs else 0.0,
            "ar/long_cloud_rein_pos_cos_mean": float(torch.stack(pos_means).mean().detach().cpu()) if pos_means else 0.0,
            "ar/long_cloud_orth_loss": float(orth_loss.detach().cpu()),
            "ar/long_cloud_orth_lambda": float(orth_lam),
            "ar/long_cloud_orth_weighted_loss": float((orth_lam * orth_loss).detach().cpu()),
            "ar/long_cloud_mean_loss": float(mean_loss.detach().cpu()),
            "ar/long_cloud_mean_lambda": float(mean_lam),
            "ar/long_cloud_mean_weighted_loss": float((mean_lam * mean_loss).detach().cpu()),
            "ar/long_cloud_p_norm": float(pn.mean().detach().cpu()),
            "ar/long_cloud_b_norm": float(bn.mean().detach().cpu()),
            "ar/long_cloud_delta_norm": float(dn.mean().detach().cpu()),
            "ar/long_cloud_E_p": float((p2 / d2).mean().detach().cpu()),
            "ar/long_cloud_E_b": float((b2 / d2).mean().detach().cpu()),
            "ar/long_cloud_E_cross": float((2.0 * dot_pb / d2).mean().detach().cpu()),
            "ar/long_cloud_E_cross_pos": float((torch.relu(2.0 * dot_pb) / d2).mean().detach().cpu()),
            "ar/long_cloud_decomp_residual_frac": float((rec / dn).mean().detach().cpu()),
            "ar/long_cloud_first_rel_l2": float(first_rel.detach().cpu()) if first_rel is not None else 0.0,
            "ar/long_cloud_last_rel_l2": float(last_rel.detach().cpu()) if last_rel is not None else 0.0,
        })
    if bridge_extra_logs:
        logs.update(bridge_extra_logs)

    return loss, logs



def _transport_coh_zero_logs() -> dict:
    return {
        "ar/transport_coh_enabled": 0.0,
        "ar/transport_coh_skipped_short": 0.0,
        "ar/transport_coh_horizon": 0.0,
        "ar/transport_coh_available_horizon": 0.0,
        "ar/transport_coh_num_sources": 0.0,
        "ar/transport_coh_sources_mean": 0.0,
        "ar/transport_coh_loss": 0.0,
        "ar/transport_coh_weighted_loss": 0.0,
        "ar/transport_coh_adj_loss": 0.0,
        "ar/transport_coh_adj_lambda": 0.0,
        "ar/transport_coh_adj_weighted_loss": 0.0,
        "ar/transport_coh_multi_loss": 0.0,
        "ar/transport_coh_multi_lambda": 0.0,
        "ar/transport_coh_multi_weighted_loss": 0.0,
        "ar/transport_coh_proj_loss": 0.0,
        "ar/transport_coh_proj_lambda": 0.0,
        "ar/transport_coh_proj_weighted_loss": 0.0,
        "ar/transport_coh_proj_mean_loss": 0.0,
        "ar/transport_coh_proj_mean_lambda": 0.0,
        "ar/transport_coh_proj_mean_weighted_loss": 0.0,
        "ar/transport_coh_pair_pos_frac": 0.0,
        "ar/transport_coh_pair_pos_cos_mean": 0.0,
        "ar/transport_coh_pair_cos_mean": 0.0,
        "ar/transport_coh_energy_ratio_R": 0.0,
        "ar/transport_coh_diffusive_ratio_sqrtR": 0.0,
        "ar/transport_coh_offdiag_frame_potential": 0.0,
        "ar/transport_coh_source_norm": 0.0,
        "ar/transport_coh_contrib_norm": 0.0,
        "ar/transport_coh_top_m": 0.0,
        "ar/transport_coh_proj_dim": 0.0,
    }


def _parse_transport_coh_sources(raw_steps: str, K: int) -> list[int]:
    """Parse one-based source horizon steps.

    h=1 means the first predicted target at t0.  A source at h contributes to
    the final horizon K after K-h future transitions.  We allow h=K for the
    multi/proj losses, where the transported contribution is just the local
    source itself.  Adjacent one-step loss will automatically skip h=K.
    """
    out: list[int] = []
    raw_steps = str(raw_steps or "").strip()
    if raw_steps:
        for item in raw_steps.replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            try:
                h = int(item)
            except ValueError:
                continue
            if 1 <= h <= int(K) and h not in out:
                out.append(h)
    return out


def _transport_coh_local_source(raw, state, stim, W: int, cur: int) -> torch.Tensor:
    """Trainable clean-history local residual b_cur = F(x_{<cur}) - x_cur."""
    hist = time_window(state, int(cur) - int(W), int(cur))
    sw = _safe_stim_window(stim, int(cur) - int(W), int(cur))
    pred = _ar_forward_pred(raw, sw, hist)
    target = time_point(state, int(cur), keep_time=False)
    return pred - target


def _transport_coh_to_final(
    raw,
    state,
    stim,
    W: int,
    t0: int,
    K: int,
    h: int,
    b_h: torch.Tensor,
    *,
    create_graph: bool,
    strict: bool,
) -> torch.Tensor:
    """Propagate a local output error source b_h to the shared final horizon K.

    The source h is one-based relative to t0.  Its target time is
    cur=t0+h-1.  The augmented history immediately after that target is
    [cur-W+1, ..., cur].  The tangent has b_h in the newest frame and zeros in
    older frames.  Repeated JVPs through the teacher-forced transition compute

        c_{h->K} = Phi_{K,h} b_h

    without materializing Jacobians.
    """
    cur = int(t0) + int(h) - 1
    final_cur = int(t0) + int(K) - 1
    if final_cur <= cur:
        return b_h
    hist_start = cur - int(W) + 1
    hist_end = cur + 1
    hist = time_window(state, hist_start, hist_end).detach()
    v_hist = torch.zeros_like(hist)
    v_hist[:, -1] = b_h
    for next_cur in range(cur + 1, final_cur + 1):
        sw = _safe_stim_window(stim, int(next_cur) - int(W), int(next_cur))
        hist, v_hist = _history_jvp_step_with_stim_graph(
            raw,
            hist,
            v_hist,
            sw,
            create_graph=bool(create_graph),
            strict=bool(strict),
        )
    return v_hist[:, -1]


def _transport_coh_pair_loss_from_vectors(
    C: torch.Tensor,
    *,
    top_m: int,
    weight_power: float,
    positive_only: bool,
    detach_norm: bool,
    eps: float,
    min_norm: float,
) -> tuple[torch.Tensor, dict]:
    """Pairwise positive-coherence loss on transported vectors C [B,S,D]."""
    B, S, D = C.shape
    zero = C.new_tensor(0.0)
    if S <= 1:
        return zero, {
            "pair_pos_frac": 0.0,
            "pair_pos_cos_mean": 0.0,
            "pair_cos_mean": 0.0,
            "energy_ratio_R": 0.0,
            "sqrtR": 0.0,
            "frame_potential": 0.0,
            "contrib_norm": 0.0,
        }
    n = C.norm(dim=-1)
    valid = n > float(min_norm)
    denom_n = n.detach() if detach_norm else n
    U = C / denom_n.clamp_min(float(eps)).unsqueeze(-1)
    G = torch.bmm(U, U.transpose(1, 2))
    eye = torch.eye(S, device=C.device, dtype=torch.bool).unsqueeze(0)
    valid_pair = valid.unsqueeze(1) & valid.unsqueeze(2) & (~eye)

    keep_src = valid.clone()
    m = int(top_m)
    if m > 0 and m < S:
        score = n.detach().masked_fill(~valid, -1.0)
        top_idx = score.topk(k=m, dim=1).indices
        keep_src = torch.zeros_like(valid)
        keep_src.scatter_(1, top_idx, True)
        keep_src = keep_src & valid
        valid_pair = keep_src.unsqueeze(1) & keep_src.unsqueeze(2) & (~eye)

    if not valid_pair.any():
        return zero, {
            "pair_pos_frac": 0.0,
            "pair_pos_cos_mean": 0.0,
            "pair_cos_mean": 0.0,
            "energy_ratio_R": 0.0,
            "sqrtR": 0.0,
            "frame_potential": 0.0,
            "contrib_norm": float(n.detach().mean().cpu()),
        }

    pair_cos = G.masked_select(valid_pair)
    if positive_only:
        pair_pen = torch.relu(G).pow(2)
    else:
        pair_pen = G.pow(2)

    if float(weight_power) != 0.0:
        w = n.detach().clamp_min(float(eps)).pow(float(weight_power))
        w = w / w.sum(dim=1, keepdim=True).clamp_min(float(eps))
        Wpair = w.unsqueeze(1) * w.unsqueeze(2)
        denom = Wpair.masked_select(valid_pair).sum().clamp_min(float(eps))
        loss = (pair_pen * Wpair).masked_select(valid_pair).sum() / denom
    else:
        loss = pair_pen.masked_select(valid_pair).mean()

    with torch.no_grad():
        pos = pair_cos > 0
        pos_frac = pos.to(C.dtype).mean() if pair_cos.numel() else C.new_tensor(0.0)
        pos_mean = pair_cos.masked_select(pos).mean() if pos.any() else C.new_tensor(0.0)
        # Coherent-vs-diffusive energy ratio for unnormalized transported vectors.
        sum_c = (C * valid.to(C.dtype).unsqueeze(-1)).sum(dim=1)
        sum_norm2 = sum_c.reshape(B, -1).pow(2).sum(dim=1)
        denom = C.reshape(B, S, -1).pow(2).sum(dim=-1).masked_fill(~valid, 0.0).sum(dim=1).clamp_min(float(eps))
        R = sum_norm2 / denom
        fp = G.pow(2).masked_select(valid_pair).mean()
        logs = {
            "pair_pos_frac": float(pos_frac.detach().cpu()),
            "pair_pos_cos_mean": float(pos_mean.detach().cpu()),
            "pair_cos_mean": float(pair_cos.mean().detach().cpu()) if pair_cos.numel() else 0.0,
            "energy_ratio_R": float(R.mean().detach().cpu()),
            "sqrtR": float(torch.sqrt(R.clamp_min(0.0)).mean().detach().cpu()),
            "frame_potential": float(fp.detach().cpu()),
            "contrib_norm": float(n.detach().mean().cpu()),
        }
    return loss, logs


def compute_transport_coherence_loss(raw, state, stim, t0: int, args) -> tuple[torch.Tensor, dict]:
    """Transported error-source coherence regularization.

    This implements three increasingly expensive variants:

      V1 adjacent one-step transport:
          penalize positive cos(J_{i+1} b_i, b_{i+1}).

      V2 sampled multi-step transport:
          compute c_i = Phi_{K,i+1} b_i by JVP propagation and penalize
          positive pairwise coherence [cos(c_i,c_j)]_+^2.

      V3 projected spherical transport:
          same c_i, but compare random-projected directions z_i=R c_i and add
          a mean-direction penalty.  This is a low-rank spherical-cloud proxy.

    The loss is direction-first: norms are detached in the cosine denominator by
    default, while detached norms can still be used for top-m source selection.
    """
    zero = state.new_tensor(0.0)
    logs0 = _transport_coh_zero_logs()
    if not bool(getattr(args, "transport_coh_loss", False)):
        return zero, logs0
    if not hasattr(raw, "step_history"):
        logs0["ar/transport_coh_enabled"] = 1.0
        return zero, logs0

    B, T, _shape = get_batch_time_shape(state)
    W = int(args.window_size)
    K_req = max(2, int(getattr(args, "transport_coh_horizon", 64)))
    max_available = int(T) - int(t0)
    if int(t0) - W < 0 or max_available < 2:
        logs0["ar/transport_coh_enabled"] = 1.0
        logs0["ar/transport_coh_skipped_short"] = 1.0
        logs0["ar/transport_coh_horizon"] = float(K_req)
        logs0["ar/transport_coh_available_horizon"] = float(max_available)
        return zero, logs0
    K = max(2, min(K_req, max_available))

    explicit = _parse_transport_coh_sources(getattr(args, "transport_coh_sources", ""), K)
    if explicit:
        sources = explicit
    else:
        stride = max(1, int(getattr(args, "transport_coh_source_stride", 8)))
        # Avoid h=K by default so V1 adjacent can use the same source list.
        cand = list(range(1, K, stride))
        if not cand:
            cand = [1]
        nsrc = max(1, min(int(getattr(args, "transport_coh_num_sources", 4)), len(cand)))
        if bool(getattr(args, "transport_coh_random_sources", False)) and len(cand) > nsrc:
            sources = random.sample(cand, nsrc)
            sources.sort()
        else:
            # Spread sources across the horizon rather than taking only early ones.
            if len(cand) <= nsrc:
                sources = cand
            else:
                idx = torch.linspace(0, len(cand) - 1, steps=nsrc).round().long().tolist()
                sources = [cand[i] for i in idx]
    max_sources = int(getattr(args, "transport_coh_max_sources", 0))
    if max_sources > 0:
        sources = sources[:max_sources]
    sources = [int(h) for h in sources if 1 <= int(h) <= K]
    if not sources:
        logs0["ar/transport_coh_enabled"] = 1.0
        logs0["ar/transport_coh_skipped_short"] = 1.0
        logs0["ar/transport_coh_horizon"] = float(K)
        logs0["ar/transport_coh_available_horizon"] = float(max_available)
        return zero, logs0

    adj_lam = float(getattr(args, "transport_coh_adj_lambda", 0.0))
    multi_lam = float(getattr(args, "transport_coh_multi_lambda", 0.0))
    proj_lam = float(getattr(args, "transport_coh_proj_lambda", 0.0))
    proj_mean_lam = float(getattr(args, "transport_coh_proj_mean_lambda", 0.0))
    eps = float(getattr(args, "transport_coh_eps", 1e-8))
    min_norm = float(getattr(args, "transport_coh_min_norm", 1e-8))
    top_m = int(getattr(args, "transport_coh_top_m", 0))
    weight_power = float(getattr(args, "transport_coh_weight_power", 0.0))
    positive_only = bool(getattr(args, "transport_coh_positive_only", True))
    detach_norm = bool(getattr(args, "transport_coh_detach_norm", True))
    strict = bool(getattr(args, "transport_coh_strict_jvp", False))
    # Important OOM guard:
    # The first patch accidentally used create_graph=True even for --transport_coh_eval,
    # so V1 still built long K-step JVP graphs for diagnostics.  In low-memory mode
    # long transported quantities are detached diagnostics only.  Trainable losses use
    # one-step proxy directions, which keeps the graph size comparable to a few extra
    # one-step U-Net calls instead of sum(K-h) JVP calls.
    lowmem = bool(getattr(args, "transport_coh_lowmem", True))
    is_training = bool(getattr(raw, "training", False))
    create_graph_adj = bool(is_training and adj_lam != 0.0 and not lowmem)
    create_graph_proxy = bool(is_training and not lowmem)

    adj_losses: list[torch.Tensor] = []
    adj_cos_logs: list[torch.Tensor] = []
    adj_pos_logs: list[torch.Tensor] = []
    source_norms: list[torch.Tensor] = []

    if adj_lam != 0.0 or bool(getattr(args, "transport_coh_eval", False)):
        for h in sources:
            if h >= K:
                continue
            cur = int(t0) + int(h) - 1
            nxt = cur + 1
            if cur - W < 0 or nxt - W < 0 or nxt >= int(T):
                continue
            if create_graph_adj:
                b_i = _transport_coh_local_source(raw, state, stim, W, cur)
            else:
                with torch.no_grad():
                    b_i = _transport_coh_local_source(raw, state, stim, W, cur).detach()
            b_j = _transport_coh_local_source(raw, state, stim, W, nxt)
            hist_after = time_window(state, cur - W + 1, cur + 1).detach()
            v_hist = torch.zeros_like(hist_after)
            v_hist[:, -1] = b_i
            sw_next = _safe_stim_window(stim, nxt - W, nxt)
            _y, jv_hist = _history_jvp_step_with_stim_graph(
                raw,
                hist_after,
                v_hist,
                sw_next,
                create_graph=create_graph_adj,
                strict=strict,
            )
            a_i = jv_hist[:, -1]
            af = a_i.reshape(B, -1)
            bf = b_j.reshape(B, -1)
            an = af.norm(dim=1)
            bn = bf.norm(dim=1)
            denom = ((an.detach() if detach_norm else an) * (bn.detach() if detach_norm else bn)).clamp_min(eps)
            cos = (af * bf).sum(dim=1) / denom
            valid = (an > min_norm) & (bn > min_norm)
            pen = torch.relu(cos).pow(2) if positive_only else cos.pow(2)
            if valid.any():
                adj_losses.append(pen.masked_select(valid).mean())
                adj_cos_logs.append(cos.detach().masked_select(valid).mean())
                adj_pos_logs.append((cos.detach().masked_select(valid) > 0).to(state.dtype).mean())
                source_norms.append(b_i.detach().reshape(B, -1).norm(dim=1).mean())

    adj_loss = torch.stack(adj_losses).mean() if adj_losses else zero

    # V2/V3.  In exact mode, C contains full transported contributions
    #     c_i = Phi_{K,i+1} b_i
    # with create_graph=True, which is expensive.  In low-memory mode, the
    # trainable loss uses one-step transported proxy directions
    #     a_i = J_{i+1} b_i
    # while full K-step C is computed only as detached diagnostics when requested.
    C_list: list[torch.Tensor] = []          # trainable vectors for loss
    C_diag_list: list[torch.Tensor] = []     # detached full transported diagnostics
    b_norm_list: list[torch.Tensor] = []
    need_loss_vectors = (multi_lam != 0.0 or proj_lam != 0.0 or proj_mean_lam != 0.0)
    need_diag_vectors = bool(getattr(args, "transport_coh_eval", False))

    if need_loss_vectors:
        for h in sources:
            cur = int(t0) + int(h) - 1
            nxt = cur + 1
            if cur - W < 0 or cur >= int(T):
                continue
            b_i = _transport_coh_local_source(raw, state, stim, W, cur)
            if lowmem:
                # One-step trainable proxy: much cheaper than K-step graph.
                if nxt < int(T):
                    hist_after = time_window(state, cur - W + 1, cur + 1).detach()
                    v_hist = torch.zeros_like(hist_after)
                    v_hist[:, -1] = b_i
                    sw_next = _safe_stim_window(stim, nxt - W, nxt)
                    _y, jv_hist = _history_jvp_step_with_stim_graph(
                        raw,
                        hist_after,
                        v_hist,
                        sw_next,
                        create_graph=True,
                        strict=strict,
                    )
                    c_i = jv_hist[:, -1]
                else:
                    c_i = b_i
            else:
                c_i = _transport_coh_to_final(
                    raw,
                    state,
                    stim,
                    W,
                    int(t0),
                    int(K),
                    int(h),
                    b_i,
                    create_graph=create_graph_proxy,
                    strict=strict,
                )
            C_list.append(c_i.reshape(B, -1))
            b_norm_list.append(b_i.detach().reshape(B, -1).norm(dim=1).mean())

    if need_diag_vectors:
        for h in sources:
            cur = int(t0) + int(h) - 1
            if cur - W < 0 or cur >= int(T):
                continue
            with torch.no_grad():
                b_i_det = _transport_coh_local_source(raw, state, stim, W, cur).detach()
            c_det = _transport_coh_to_final(
                raw,
                state,
                stim,
                W,
                int(t0),
                int(K),
                int(h),
                b_i_det,
                create_graph=False,
                strict=strict,
            )
            C_diag_list.append(c_det.reshape(B, -1))

    if C_list:
        C = torch.stack(C_list, dim=1)  # [B,S,D], trainable loss vectors
        multi_loss, plogs_loss = _transport_coh_pair_loss_from_vectors(
            C,
            top_m=top_m,
            weight_power=weight_power,
            positive_only=positive_only,
            detach_norm=detach_norm,
            eps=eps,
            min_norm=min_norm,
        )
        proj_loss = zero
        proj_mean_loss = zero
        proj_dim = int(getattr(args, "transport_coh_proj_dim", 16))
        if proj_lam != 0.0 or proj_mean_lam != 0.0 or bool(getattr(args, "transport_coh_eval", False)):
            D = int(C.shape[-1])
            r = max(1, min(proj_dim, D))
            gen = torch.Generator(device=C.device)
            gen.manual_seed(int(getattr(args, "transport_coh_proj_seed", 12345)))
            Rmat = torch.randn(D, r, device=C.device, dtype=C.dtype, generator=gen) / (float(r) ** 0.5)
            Z = C @ Rmat
            proj_loss, proj_logs = _transport_coh_pair_loss_from_vectors(
                Z,
                top_m=top_m,
                weight_power=weight_power,
                positive_only=positive_only,
                detach_norm=detach_norm,
                eps=eps,
                min_norm=min_norm,
            )
            zn = Z.norm(dim=-1)
            valid = zn > min_norm
            denom_z = zn.detach() if detach_norm else zn
            U = Z / denom_z.clamp_min(eps).unsqueeze(-1)
            if top_m > 0 and top_m < Z.shape[1]:
                score = zn.detach().masked_fill(~valid, -1.0)
                top_idx = score.topk(k=top_m, dim=1).indices
                keep = torch.zeros_like(valid)
                keep.scatter_(1, top_idx, True)
                valid = valid & keep
            vf = valid.to(U.dtype)
            denom = vf.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean_vec = (U * vf.unsqueeze(-1)).sum(dim=1) / denom
            proj_mean_loss = mean_vec.pow(2).sum(dim=-1).mean()
        else:
            proj_logs = {"frame_potential": 0.0}
    else:
        multi_loss = zero
        proj_loss = zero
        proj_mean_loss = zero
        plogs_loss = {
            "pair_pos_frac": 0.0,
            "pair_pos_cos_mean": 0.0,
            "pair_cos_mean": 0.0,
            "energy_ratio_R": 0.0,
            "sqrtR": 0.0,
            "frame_potential": 0.0,
            "contrib_norm": 0.0,
        }
        proj_logs = {"frame_potential": 0.0}
        proj_dim = int(getattr(args, "transport_coh_proj_dim", 16))

    # Prefer detached full-K diagnostics when requested; otherwise report the
    # trainable proxy vectors used by V2/V3.  This avoids building long graphs
    # just for logging in V1.
    if C_diag_list:
        C_log = torch.stack(C_diag_list, dim=1)
        _diag_loss, plogs = _transport_coh_pair_loss_from_vectors(
            C_log,
            top_m=top_m,
            weight_power=weight_power,
            positive_only=positive_only,
            detach_norm=True,
            eps=eps,
            min_norm=min_norm,
        )
    else:
        plogs = plogs_loss

    loss = adj_lam * adj_loss + multi_lam * multi_loss + proj_lam * proj_loss + proj_mean_lam * proj_mean_loss

    with torch.no_grad():
        logs = _transport_coh_zero_logs()
        logs.update({
            "ar/transport_coh_enabled": 1.0,
            "ar/transport_coh_skipped_short": 0.0,
            "ar/transport_coh_horizon": float(K),
            "ar/transport_coh_available_horizon": float(max_available),
            "ar/transport_coh_num_sources": float(len(sources)),
            "ar/transport_coh_sources_mean": float(sum(sources) / max(1, len(sources))),
            "ar/transport_coh_loss": float(loss.detach().cpu()),
            "ar/transport_coh_weighted_loss": float(loss.detach().cpu()),
            "ar/transport_coh_adj_loss": float(adj_loss.detach().cpu()),
            "ar/transport_coh_adj_lambda": float(adj_lam),
            "ar/transport_coh_adj_weighted_loss": float((adj_lam * adj_loss).detach().cpu()),
            "ar/transport_coh_multi_loss": float(multi_loss.detach().cpu()),
            "ar/transport_coh_multi_lambda": float(multi_lam),
            "ar/transport_coh_multi_weighted_loss": float((multi_lam * multi_loss).detach().cpu()),
            "ar/transport_coh_proj_loss": float(proj_loss.detach().cpu()),
            "ar/transport_coh_proj_lambda": float(proj_lam),
            "ar/transport_coh_proj_weighted_loss": float((proj_lam * proj_loss).detach().cpu()),
            "ar/transport_coh_proj_mean_loss": float(proj_mean_loss.detach().cpu()),
            "ar/transport_coh_proj_mean_lambda": float(proj_mean_lam),
            "ar/transport_coh_proj_mean_weighted_loss": float((proj_mean_lam * proj_mean_loss).detach().cpu()),
            "ar/transport_coh_pair_pos_frac": float(plogs.get("pair_pos_frac", 0.0)),
            "ar/transport_coh_pair_pos_cos_mean": float(plogs.get("pair_pos_cos_mean", 0.0)),
            "ar/transport_coh_pair_cos_mean": float(plogs.get("pair_cos_mean", 0.0)),
            "ar/transport_coh_energy_ratio_R": float(plogs.get("energy_ratio_R", 0.0)),
            "ar/transport_coh_diffusive_ratio_sqrtR": float(plogs.get("sqrtR", 0.0)),
            "ar/transport_coh_offdiag_frame_potential": float(plogs.get("frame_potential", 0.0)),
            "ar/transport_coh_source_norm": float(torch.stack(b_norm_list).mean().detach().cpu()) if b_norm_list else (float(torch.stack(source_norms).mean().detach().cpu()) if source_norms else 0.0),
            "ar/transport_coh_contrib_norm": float(plogs.get("contrib_norm", 0.0)),
            "ar/transport_coh_top_m": float(top_m),
            "ar/transport_coh_proj_dim": float(proj_dim),
        })
    return loss, logs

def _per_sample_state_loss(pred: torch.Tensor, target: torch.Tensor, loss_name: str = "rel_l2") -> torch.Tensor:
    """Return one scalar loss per sample for a state/frame tensor."""
    p = pred.reshape(pred.shape[0], -1)
    y = target.reshape(target.shape[0], -1)
    if loss_name == "rel_l2":
        return (p - y).norm(dim=1) / y.norm(dim=1).clamp_min(1e-8)
    if loss_name == "mse":
        return (p - y).pow(2).mean(dim=1)
    if loss_name == "l1":
        return (p - y).abs().mean(dim=1)
    if loss_name == "huber":
        # torch.nn.functional.huber_loss does not expose per-sample reduction.
        delta = 20.0
        diff = (p - y).abs()
        quad = diff.clamp(max=delta)
        lin = diff - quad
        return (0.5 * quad.pow(2) + delta * lin).mean(dim=1)
    raise ValueError(f"Unknown source-color loss_name={loss_name}")


def _global_wiener_per_sample_state_loss(
    pred: torch.Tensor, target: torch.Tensor, loss_name: str
) -> torch.Tensor:
    """Per-sample form of the production scalar rollout objective.

    In particular, :func:`relative_l2` evaluates its norm in float64.  The
    older diagnostic helper above intentionally follows input dtype, so using
    it here would not reproduce the score whose complete-batch mean is trained.
    """

    if loss_name == "rel_l2":
        p = pred.reshape(pred.shape[0], -1).double()
        y = target.reshape(target.shape[0], -1).double()
        return torch.linalg.vector_norm(p - y, dim=1) / torch.linalg.vector_norm(
            y, dim=1
        ).clamp_min(1e-8)
    return _per_sample_state_loss(pred, target, loss_name)


def _safe_stim_window(stim: torch.Tensor | None, start: int, end: int):
    if stim is None:
        return None
    if start < 0 or end > stim.shape[1] or start >= end:
        return None
    return time_window(stim, start, end)


def _batched_modified_gram_schmidt(A: torch.Tensor, min_norm: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched modified Gram-Schmidt over source footprints.

    Args:
        A: [B,K,D] raw source footprints.
    Returns:
        U: [B,K,D] approximately orthonormal source directions.
        valid: [B,K] mask for non-degenerate directions.
    """
    B, K, D = A.shape
    U_cols = []
    valid_cols = []
    for i in range(K):
        v = A[:, i]
        for u in U_cols:
            proj = (v * u).sum(dim=1, keepdim=True)
            v = v - proj * u
        n = v.norm(dim=1, keepdim=True)
        valid = (n[:, 0] > float(min_norm))
        u = v / n.clamp_min(float(min_norm))
        u = torch.where(valid.view(B, 1), u, torch.zeros_like(u))
        U_cols.append(u)
        valid_cols.append(valid)
    U = torch.stack(U_cols, dim=1)
    valid = torch.stack(valid_cols, dim=1)
    return U, valid


def _raw_source_offdiag_cos2(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean squared off-diagonal cosine of raw source footprints."""
    if A.shape[1] <= 1:
        return A.new_tensor(0.0)
    An = A / A.norm(dim=-1, keepdim=True).clamp_min(eps)
    G = torch.bmm(An, An.transpose(1, 2))
    K = A.shape[1]
    mask = ~torch.eye(K, device=A.device, dtype=torch.bool).unsqueeze(0)
    return G.pow(2).masked_select(mask).mean()


def compute_source_colored_local_repair_loss(raw, state, stim, t0: int, args) -> tuple[torch.Tensor, dict]:
    """Source-colored no-grad diagnosis + on-policy one-step local repair.

    The long rollout is used only to construct detached source weights.  The
    returned scalar backpropagates only through one-step predictions at detached
    rollout states, avoiding a full BPTT graph.
    """
    zero = state.new_tensor(0.0)
    logs0 = _source_color_zero_logs()
    if not bool(getattr(args, "source_color_loss", False)):
        return zero, logs0
    if not hasattr(raw, "step_history"):
        logs0["ar/source_color_enabled"] = 1.0
        return zero, logs0

    B, T, _shape = get_batch_time_shape(state)
    W = int(args.window_size)
    K_req = int(getattr(args, "source_color_horizon", 0))
    if K_req <= 0:
        K_req = int(getattr(args, "koopman_gramian_horizon", 64))
    K_req = max(1, int(K_req))
    max_available = int(T) - int(t0)
    strict = bool(getattr(args, "source_color_strict_horizon", False))
    if max_available <= 0 or (strict and max_available < K_req):
        logs0["ar/source_color_enabled"] = 1.0
        logs0["ar/source_color_skipped_short"] = 1.0
        logs0["ar/source_color_available_horizon"] = float(max_available)
        logs0["ar/source_color_horizon"] = float(K_req)
        return zero, logs0
    K = max(1, min(K_req, max_available))
    if K <= 1 or int(t0) - W < 0:
        logs0["ar/source_color_enabled"] = 1.0
        logs0["ar/source_color_skipped_short"] = 1.0
        logs0["ar/source_color_available_horizon"] = float(max_available)
        logs0["ar/source_color_horizon"] = float(K)
        return zero, logs0

    sigma = float(getattr(args, "source_color_perturb_eps", 1e-3))
    sigma = max(sigma, 1e-8)
    min_norm = float(getattr(args, "source_color_min_norm", 1e-8))
    tau = float(getattr(args, "source_color_temperature", 1.0))
    tau = max(tau, 1e-6)
    loss_name = str(getattr(args, "source_color_local_loss", getattr(args, "ar_loss", "rel_l2")))

    was_training = bool(getattr(raw, "training", False))
    use_eval = bool(getattr(args, "source_color_eval_mode_diagnosis", True))
    if use_eval and was_training:
        raw.eval()

    try:
        with torch.no_grad():
            hist = time_window(state, int(t0) - W, int(t0)).detach()
            histories = []
            next_histories = []
            preds = []
            local_residuals = []

            for s in range(K):
                cur = int(t0) + s
                sw = _safe_stim_window(stim, cur - W, cur)
                histories.append(hist.detach())
                next_hist, pred = _ar_step_history_with_stim(raw, hist, sw)
                tgt = time_point(state, cur, keep_time=False).detach()
                preds.append(pred.detach())
                local_residuals.append((pred.detach() - tgt).detach())
                next_histories.append(next_hist.detach())
                hist = next_hist.detach()

            final_pred = preds[-1].detach()
            final_target = time_point(state, int(t0) + K - 1, keep_time=False).detach()
            delta_K = final_pred - final_target

            footprints = []
            for i in range(K):
                r = local_residuals[i]
                r_norm = r.reshape(B, -1).norm(dim=1).view(B, *([1] * (r.dim() - 1)))
                direction = r / r_norm.clamp_min(1e-8)
                pert_hist = next_histories[i].clone()
                pert_hist[:, -1] = pert_hist[:, -1] + sigma * direction

                # Reroll the perturbed trajectory from just after source i to K.
                hpert = pert_hist
                for s in range(i + 1, K):
                    cur = int(t0) + s
                    sw = _safe_stim_window(stim, cur - W, cur)
                    hpert, _ = _ar_step_history_with_stim(raw, hpert, sw)
                pert_final = hpert[:, -1].detach()
                footprints.append(((pert_final - final_pred) / sigma).detach())

            A = torch.stack([a.reshape(B, -1) for a in footprints], dim=1)  # [B,K,D]
            U, valid = _batched_modified_gram_schmidt(A, min_norm=min_norm)
            delta_flat = delta_K.reshape(B, -1)
            coeff = (U * delta_flat[:, None, :]).sum(dim=-1)  # U has unit norm where valid
            coeff = torch.where(valid, coeff, torch.zeros_like(coeff))

            mode = str(getattr(args, "source_color_weight_mode", "softmax")).lower()
            score = coeff.abs()
            if mode == "normalize":
                denom = score.sum(dim=1, keepdim=True).clamp_min(1e-8)
                w = score / denom
                empty = (valid.sum(dim=1, keepdim=True) <= 0)
                w = torch.where(empty, torch.full_like(w, 1.0 / float(K)), w)
            elif mode == "top1":
                idx = score.argmax(dim=1)
                w = torch.zeros_like(score).scatter_(1, idx[:, None], 1.0)
            else:
                logits = score / tau
                logits = logits.masked_fill(~valid, -1e9)
                w = torch.softmax(logits, dim=1)
                empty = (valid.sum(dim=1, keepdim=True) <= 0)
                w = torch.where(empty, torch.full_like(w, 1.0 / float(K)), w)

            w = w.detach()
            dom = score.argmax(dim=1).float().mean()
            maxw = w.max(dim=1).values.mean()
            entropy = (-(w * (w.clamp_min(1e-8)).log()).sum(dim=1) / torch.log(torch.tensor(float(K), device=w.device))).mean()
            raw_offdiag = _raw_source_offdiag_cos2(A)
            local_rel = torch.stack([_per_sample_state_loss(preds[i], time_point(state, int(t0) + i, keep_time=False), "rel_l2") for i in range(K)], dim=1).mean()
            delta_norm = delta_flat.norm(dim=1).mean()

    finally:
        if use_eval and was_training:
            raw.train()

    # One-step on-policy local repair with graph through model parameters only.
    # IMPORTANT: diagnosis may inspect all K sources, but backward only builds
    # one-step graphs for the selected sources. This is the memory-light part.
    topk = int(getattr(args, "source_color_repair_topk", 8))
    frac = float(getattr(args, "source_color_repair_frac", 0.0))
    if topk > 0:
        keep_n = min(K, max(1, topk))
    elif frac > 0.0:
        keep_n = min(K, max(1, int(round(float(K) * frac))))
    else:
        keep_n = K

    # Select source indices per batch by detached weights. To keep the Python
    # graph simple, build one-step graphs for the union of selected indices
    # across the mini-batch. The actual per-sample weights remain w[:, i].
    if keep_n < K:
        selected = torch.topk(w, k=keep_n, dim=1).indices  # [B, keep_n]
        unique_idx = torch.unique(selected.reshape(-1)).detach().cpu().tolist()
        selected_mask = torch.zeros_like(w, dtype=torch.bool)
        selected_mask.scatter_(1, selected, True)
        w_repair = torch.where(selected_mask, w, torch.zeros_like(w))
        if bool(getattr(args, "source_color_renorm_selected", True)):
            w_repair = w_repair / w_repair.sum(dim=1, keepdim=True).clamp_min(1e-8)
    else:
        unique_idx = list(range(K))
        w_repair = w

    repair_loss = zero
    for i in unique_idx:
        i = int(i)
        cur = int(t0) + i
        hist_i = histories[i].detach()
        sw = _safe_stim_window(stim, cur - W, cur)
        pred_i = _ar_forward_pred(raw, sw, hist_i)
        tgt_i = time_point(state, cur, keep_time=False)
        per = _per_sample_state_loss(pred_i, tgt_i, loss_name)
        wi = w_repair[:, i].to(per.device, dtype=per.dtype)
        repair_loss = repair_loss + (wi * per).mean()

    repair_num = float(len(unique_idx))
    repair_frac = float(len(unique_idx)) / float(max(1, K))

    logs = {
        "ar/source_color_loss": float(repair_loss.detach().cpu()),
        "ar/source_color_lambda": float(getattr(args, "source_color_lambda", 0.0)),
        "ar/source_color_weighted_loss": float((float(getattr(args, "source_color_lambda", 0.0)) * repair_loss).detach().cpu()),
        "ar/source_color_horizon": float(K),
        "ar/source_color_available_horizon": float(max_available),
        "ar/source_color_num_sources": float(K),
        "ar/source_color_repair_num_sources": repair_num,
        "ar/source_color_repair_frac": repair_frac,
        "ar/source_color_valid_frac": float(valid.float().mean().detach().cpu()),
        "ar/source_color_max_weight": float(maxw.detach().cpu()),
        "ar/source_color_entropy": float(entropy.detach().cpu()),
        "ar/source_color_dominant_index": float(dom.detach().cpu()),
        "ar/source_color_raw_offdiag_cos2": float(raw_offdiag.detach().cpu()),
        "ar/source_color_deltaK_norm": float(delta_norm.detach().cpu()),
        "ar/source_color_local_rel_l2": float(local_rel.detach().cpu()),
        "ar/source_color_enabled": 1.0,
        "ar/source_color_skipped_short": 0.0,
    }
    return repair_loss, logs



def _pseudo_dot_loss(pred: torch.Tensor, pseudo: torch.Tensor, zero_value: bool = True) -> torch.Tensor:
    """Inner-product pseudo loss whose gradient wrt pred is pseudo / numel.

    If zero_value=True, use (pred - pred.detach()) so the forward scalar is
    exactly zero while the gradient is unchanged.  This is useful because this
    term is not a meaningful supervised loss; it is only a gradient injection
    device.  Without this trick, training curves and early-stopping criteria can
    be dominated by an arbitrary inner-product value even when the injected
    gradient scale is small.
    """
    x = pred - pred.detach() if zero_value else pred
    p = x.reshape(x.shape[0], -1)
    q = pseudo.reshape(p.shape[0], -1)
    return (p * q).sum(dim=1).mean() / max(1, p.shape[1])


def _pseudo_dot_proxy(pred: torch.Tensor, pseudo: torch.Tensor) -> torch.Tensor:
    """Detached diagnostic for the raw pseudo-dot magnitude."""
    p = pred.detach().reshape(pred.shape[0], -1)
    q = pseudo.detach().reshape(p.shape[0], -1)
    return (p * q).sum(dim=1).mean() / max(1, p.shape[1])


def _effective_comp_lambda(args, epoch: int | float = 0) -> float:
    """Schedule compiled-gradient strength separately from the graph structure."""
    base = float(getattr(args, "comp_lambda", 0.0))
    if base == 0.0:
        return 0.0
    start = int(getattr(args, "comp_start_epoch", 0))
    ramp = int(getattr(args, "comp_ramp_epochs", 0))
    ep = float(epoch or 0)
    if ep < start:
        return 0.0
    if ramp > 0:
        scale = min(1.0, max(0.0, (ep - start + 1.0) / float(ramp)))
        return base * scale
    return base


def _etm_stage_name(args, epoch: int | float = 0) -> str:
    """Return the ETM training stage for the current epoch.

    Stages:
      warmup: ETM disabled; ordinary AR training only.
      j_only: freeze AR backbone in trainer.py, train only ETM head with fit loss.
      joint:  train AR backbone and ETM head jointly with fit + prop losses.

    The J-only stage starts at --etm_start_epoch and lasts
    --etm_j_only_epochs epochs.  The joint ramp starts after that stage, not at
    --etm_start_epoch.  This prevents the prop loss from becoming strong before
    the transport head has learned a stable rollout-sensitivity map.
    """
    ep = float(epoch or 0)
    start = int(getattr(args, "etm_start_epoch", 0))
    j_epochs = max(0, int(getattr(args, "etm_j_only_epochs", 0)))
    if ep < start:
        return "warmup"
    if j_epochs > 0 and ep < start + j_epochs:
        return "j_only"
    return "joint"


def _effective_etm_lambdas(args, epoch: int | float = 0) -> tuple[float, float]:
    """Warm-up/J-only/joint schedule for Error-Transport Metric losses."""
    lam_fit = float(getattr(args, "etm_lambda_fit", 0.0))
    lam_prop = float(getattr(args, "etm_lambda_prop", 0.0))
    if lam_fit == 0.0 and lam_prop == 0.0:
        return 0.0, 0.0

    stage = _etm_stage_name(args, epoch)
    if stage == "warmup":
        return 0.0, 0.0

    if stage == "j_only":
        lam_fit_j = float(getattr(args, "etm_j_only_lambda_fit", -1.0))
        if lam_fit_j < 0.0:
            lam_fit_j = lam_fit
        # During J-only, never apply prop: J is only fitted to the detached
        # rollout-sensitivity target while the backbone is frozen by trainer.py.
        return lam_fit_j, 0.0

    # Joint stage: ramp starts after the J-only stage.
    start = int(getattr(args, "etm_start_epoch", 0))
    j_epochs = max(0, int(getattr(args, "etm_j_only_epochs", 0)))
    joint_start = start + j_epochs
    ramp = int(getattr(args, "etm_ramp_epochs", 0))
    ep = float(epoch or 0)
    scale = 1.0
    if ramp > 0:
        scale = min(1.0, max(0.0, (ep - joint_start + 1.0) / float(ramp)))
    return lam_fit * scale, lam_prop * scale


def _future_stim_window_for_etm(stim: torch.Tensor | None, start: int, end: int) -> torch.Tensor | None:
    if stim is None:
        return None
    if end <= start:
        return None
    return time_window(stim, int(start), int(end))



def _etm_batch_r2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Batch-averaged energy R^2 for ETM diagnostics.

    This uses zero as the reference baseline, matching the existing ETM R^2
    diagnostic in this file: 1 - ||pred-target||^2 / ||target||^2.
    """
    B = int(target.shape[0])
    sse = (pred.detach() - target).reshape(B, -1).pow(2).sum(dim=1)
    denom = target.detach().reshape(B, -1).pow(2).sum(dim=1).clamp_min(float(eps))
    return (1.0 - sse / denom).mean()


def _etm_batch_cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Batch-averaged cosine similarity for flattened ETM vectors."""
    B = int(a.shape[0])
    aa = a.detach().reshape(B, -1)
    bb = b.detach().reshape(B, -1)
    num = (aa * bb).sum(dim=1)
    den = aa.norm(dim=1).clamp_min(float(eps)) * bb.norm(dim=1).clamp_min(float(eps))
    return (num / den).mean()


def _etm_normalize_transport_for_prop(J: torch.Tensor, args) -> torch.Tensor:
    """Return the stopped transport operator used by ETM prop losses.

    The fit branch learns an unconstrained transport matrix.  The prop branch
    uses a stopped copy, optionally Frobenius-normalized so that ||J||_F^2 is
    comparable to the state dimension.  This keeps the prop scale stable while
    still allowing the learned anisotropic directions to affect the loss.
    """
    J_prop = J.detach()
    if bool(getattr(args, "etm_normalize_metric", True)):
        d_state = J_prop.shape[-1]
        fro2 = J_prop.pow(2).sum(dim=(1, 2), keepdim=True)
        return J_prop * (float(d_state) / fro2.clamp_min(1e-8)).sqrt()
    return J_prop

def _etm_selected_horizons(K: int, args) -> list[int]:
    """Return 1-indexed horizons used by the ETM multi-step fit.

    If --etm_fit_all_steps is enabled, use horizons 1, 1+stride, ... K.
    Otherwise use only the final endpoint K, reproducing the old behavior.
    """
    K = int(K)
    if K <= 0:
        return []
    if bool(getattr(args, "etm_fit_all_steps", False)):
        stride = max(1, int(getattr(args, "etm_fit_step_stride", 1)))
        hs = list(range(1, K + 1, stride))
        if hs[-1] != K:
            hs.append(K)
        return hs
    return [K]



def _parse_positive_int_list(spec: str, *, max_value: int | None = None) -> list[int]:
    """Parse comma/space separated positive integer offsets."""
    vals: list[int] = []
    if spec is None:
        return vals
    for tok in str(spec).replace(';', ',').replace(' ', ',').split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = int(tok)
        except ValueError:
            continue
        if v <= 0:
            continue
        if max_value is not None:
            v = min(v, int(max_value))
        vals.append(v)
    # keep order but remove duplicates
    out: list[int] = []
    seen: set[int] = set()
    for v in vals:
        if v not in seen:
            out.append(v); seen.add(v)
    return out


def _error_proj_pseudo_zero_logs() -> dict:
    return {
        "ar/error_proj_pseudo_loss": 0.0,
        "ar/error_proj_pseudo_lambda": 0.0,
        "ar/error_proj_pseudo_weighted_loss": 0.0,
        "ar/error_proj_pseudo_horizon": 0.0,
        "ar/error_proj_pseudo_available_horizon": 0.0,
        "ar/error_proj_pseudo_num_sources": 0.0,
        "ar/error_proj_pseudo_gamma": 0.0,
        "ar/error_proj_pseudo_apply_frac": 0.0,
        "ar/error_proj_pseudo_align_cos_mean": 0.0,
        "ar/error_proj_pseudo_pos_cos_mean": 0.0,
        "ar/error_proj_pseudo_defect_norm": 0.0,
        "ar/error_proj_pseudo_prev_error_norm": 0.0,
        "ar/error_proj_pseudo_correction_norm": 0.0,
        "ar/error_proj_pseudo_correction_frac_of_defect": 0.0,
        "ar/error_proj_pseudo_target_rel_l2": 0.0,
        "ar/error_proj_pseudo_skipped_short": 0.0,
    }


def compute_error_projection_pseudo_loss(raw, state, stim, t0: int, args) -> tuple[torch.Tensor, dict]:
    """Detached rollout projection target + local pseudo-target fitting.

    This implements the trainable counterpart of the oracle diagnostic:

        e_s = xhat_s - x_s
        b_s = F(xhat_s, u_s) - x_{s+1}
        delta_y_s = - gamma * [<e_s,b_s>]_+ / (||e_s||^2 + eps) * e_s
        y_target_s = sg(F(xhat_s,u_s) + delta_y_s)

    The rollout used to define e_s and delta_y_s is no-grad.  Gradients flow
    only through a fresh one-step call at selected rollout states.  This avoids
    the long BPTT graph while explicitly training the model to realize the
    output-space anti-reinforcement projection.
    """
    logs0 = _error_proj_pseudo_zero_logs()
    lam = float(getattr(args, "error_proj_pseudo_lambda", 0.0))
    if lam == 0.0 and raw.training:
        return state.new_tensor(0.0), logs0
    if not hasattr(raw, "step_history") and not hasattr(raw, "forward"):
        return state.new_tensor(0.0), logs0

    Bsz, T, _ = get_batch_time_shape(state)
    W = int(args.window_size)
    K_req = int(getattr(args, "error_proj_pseudo_horizon", 64))
    K = max(1, min(K_req, T - int(t0)))
    if K <= 1 or int(t0) - W < 0:
        logs = dict(logs0)
        logs["ar/error_proj_pseudo_skipped_short"] = 1.0
        return state.new_tensor(0.0), logs
    if stim is None:
        stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))

    gamma = float(getattr(args, "error_proj_pseudo_gamma", 1.0))
    eps = float(getattr(args, "error_proj_pseudo_eps", 1e-8))
    min_err = float(getattr(args, "error_proj_pseudo_min_error_norm", 1e-8))
    detach_norm = bool(getattr(args, "error_proj_pseudo_detach_norm", True))
    loss_name = str(getattr(args, "error_proj_pseudo_loss_type", getattr(args, "ar_loss", "rel_l2"))).lower()

    # Source offsets are 1-indexed rollout steps.  If empty, use all steps after
    # the first one; step 1 starts from ground-truth history so e_t is usually 0.
    offsets = _parse_positive_int_list(str(getattr(args, "error_proj_pseudo_sources", "")), max_value=K)
    if not offsets:
        stride = max(1, int(getattr(args, "error_proj_pseudo_source_stride", 1)))
        offsets = list(range(2, K + 1, stride))
    max_sources = int(getattr(args, "error_proj_pseudo_max_sources", 0))
    if max_sources > 0 and len(offsets) > max_sources:
        offsets = offsets[:max_sources]
    source_set = set(offsets)

    items: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]] = []
    cos_vals: list[torch.Tensor] = []
    pos_cos_vals: list[torch.Tensor] = []
    apply_fracs: list[torch.Tensor] = []
    defect_norms: list[torch.Tensor] = []
    err_norms: list[torch.Tensor] = []
    corr_norms: list[torch.Tensor] = []
    corr_fracs: list[torch.Tensor] = []
    target_rels: list[torch.Tensor] = []

    with torch.no_grad():
        hist = time_window(state, int(t0) - W, int(t0)).detach()
        for sidx in range(1, K + 1):
            cur = int(t0) + (sidx - 1)
            sw = time_window(stim, cur - W, cur) if stim is not None else None
            prev_true = time_point(state, cur - 1, keep_time=False).detach()
            e_prev = (hist[:, -1] - prev_true).detach()
            next_hist, pred = _ar_step_history_with_stim(raw, hist, sw)
            target = time_point(state, cur, keep_time=False).detach()
            b = (pred - target).detach()

            e_flat = e_prev.reshape(Bsz, -1)
            b_flat = b.reshape(Bsz, -1)
            dot = (e_flat * b_flat).sum(dim=1)
            e_norm = e_flat.norm(dim=1)
            b_norm = b_flat.norm(dim=1)
            denom = (e_norm * b_norm).clamp_min(eps)
            cos = dot / denom
            valid = (e_norm > min_err) & (b_norm > min_err)
            alpha_denom = e_flat.pow(2).sum(dim=1).clamp_min(eps)
            if detach_norm:
                alpha_denom = alpha_denom.detach()
            alpha = gamma * torch.clamp(dot, min=0.0) / alpha_denom
            alpha = torch.where(valid, alpha, torch.zeros_like(alpha))
            view = [Bsz] + [1] * (b.dim() - 1)
            delta_y = -alpha.view(*view) * e_prev
            pseudo_target = (pred + delta_y).detach()

            if sidx in source_set:
                items.append((hist.detach(), sw.detach() if sw is not None else None, pseudo_target))
                cos_valid = cos[valid]
                if cos_valid.numel() > 0:
                    cos_vals.append(cos_valid.mean())
                    pos = cos_valid[cos_valid > 0]
                    pos_cos_vals.append(pos.mean() if pos.numel() > 0 else cos_valid.new_tensor(0.0))
                    apply_fracs.append((cos_valid > 0).float().mean())
                else:
                    cos_vals.append(cos.new_tensor(0.0))
                    pos_cos_vals.append(cos.new_tensor(0.0))
                    apply_fracs.append(cos.new_tensor(0.0))
                defect_norms.append(b_norm.mean())
                err_norms.append(e_norm.mean())
                cn = delta_y.reshape(Bsz, -1).norm(dim=1)
                corr_norms.append(cn.mean())
                corr_fracs.append((cn / b_norm.clamp_min(eps)).mean())
                target_rels.append(relative_l2(pseudo_target, target).detach())

            hist = next_hist.detach()

    if not items:
        logs = dict(logs0)
        logs["ar/error_proj_pseudo_skipped_short"] = 1.0
        logs["ar/error_proj_pseudo_available_horizon"] = float(K)
        logs["ar/error_proj_pseudo_horizon"] = float(K)
        logs["ar/error_proj_pseudo_lambda"] = float(lam)
        logs["ar/error_proj_pseudo_gamma"] = float(gamma)
        return state.new_tensor(0.0), logs

    losses: list[torch.Tensor] = []
    for hist_i, sw_i, pseudo_target_i in items:
        pred_i = _ar_forward_pred(raw, sw_i, hist_i.detach())
        if loss_name == "rel_l2":
            li = relative_l2(pred_i, pseudo_target_i)
        else:
            li = elementwise_state_loss(pred_i.unsqueeze(1), pseudo_target_i.unsqueeze(1), loss=loss_name)
        losses.append(li)
    loss = torch.stack(losses).mean()

    def mean_log(xs: list[torch.Tensor]) -> float:
        return float(torch.stack(xs).mean().detach().cpu()) if xs else 0.0

    logs = {
        "ar/error_proj_pseudo_loss": float(loss.detach().cpu()),
        "ar/error_proj_pseudo_lambda": float(lam),
        "ar/error_proj_pseudo_weighted_loss": float((lam * loss).detach().cpu()),
        "ar/error_proj_pseudo_horizon": float(K),
        "ar/error_proj_pseudo_available_horizon": float(K),
        "ar/error_proj_pseudo_num_sources": float(len(items)),
        "ar/error_proj_pseudo_gamma": float(gamma),
        "ar/error_proj_pseudo_apply_frac": mean_log(apply_fracs),
        "ar/error_proj_pseudo_align_cos_mean": mean_log(cos_vals),
        "ar/error_proj_pseudo_pos_cos_mean": mean_log(pos_cos_vals),
        "ar/error_proj_pseudo_defect_norm": mean_log(defect_norms),
        "ar/error_proj_pseudo_prev_error_norm": mean_log(err_norms),
        "ar/error_proj_pseudo_correction_norm": mean_log(corr_norms),
        "ar/error_proj_pseudo_correction_frac_of_defect": mean_log(corr_fracs),
        "ar/error_proj_pseudo_target_rel_l2": mean_log(target_rels),
        "ar/error_proj_pseudo_skipped_short": 0.0,
    }

    return loss, logs


def _error_proj_coboundary_zero_logs() -> dict:
    return {
        "ar/error_proj_cob_loss": 0.0,
        "ar/error_proj_cob_lambda": 0.0,
        "ar/error_proj_cob_weighted_loss": 0.0,
        "ar/error_proj_cob_horizon": 0.0,
        "ar/error_proj_cob_available_horizon": 0.0,
        "ar/error_proj_cob_num_sources": 0.0,
        "ar/error_proj_cob_num_fit_steps": 0.0,
        "ar/error_proj_cob_gamma": 0.0,
        "ar/error_proj_cob_ridge": 0.0,
        "ar/error_proj_cob_potential_mode": 0.0,
        "ar/error_proj_cob_energy_scale": 0.0,
        "ar/error_proj_cob_fit_scale": 0.0,
        "ar/error_proj_cob_fit_beta_mean": 0.0,
        "ar/error_proj_cob_apply_frac": 0.0,
        "ar/error_proj_cob_raw_apply_frac": 0.0,
        "ar/error_proj_cob_align_cos_mean": 0.0,
        "ar/error_proj_cob_pos_cos_mean": 0.0,
        "ar/error_proj_cob_a_mean": 0.0,
        "ar/error_proj_cob_a_pos_mean": 0.0,
        "ar/error_proj_cob_phi_delta_mean": 0.0,
        "ar/error_proj_cob_residual_mean": 0.0,
        "ar/error_proj_cob_residual_pos_mean": 0.0,
        "ar/error_proj_cob_residual_frac_of_raw": 0.0,
        "ar/error_proj_cob_explained_r2": 0.0,
        "ar/error_proj_cob_defect_norm": 0.0,
        "ar/error_proj_cob_prev_error_norm": 0.0,
        "ar/error_proj_cob_correction_norm": 0.0,
        "ar/error_proj_cob_correction_frac_of_defect": 0.0,
        "ar/error_proj_cob_target_rel_l2": 0.0,
        "ar/error_proj_cob_skipped_short": 0.0,
    }


def _coboundary_feature_bank(
    e: torch.Tensor,
    xhat: torch.Tensor,
    xtrue: torch.Tensor,
    tau: torch.Tensor,
    eps: float,
) -> dict[str, torch.Tensor]:
    """Diagnostic-matched scalar potential feature bank.

    IMPORTANT: this intentionally mirrors scripts/diagnose_hcp_coboundary.py.
    The earlier training implementation used a different small feature set and
    a per-sequence no-intercept ridge fit; that is why training logs reported a
    residual_frac_of_raw near 0.95 while the offline diagnostic reported ~0.48.

    Shapes:
        e, xhat, xtrue: [B, ...]
        tau: [B]
    Returns:
        feature families q_t, each [B, F].
    """
    Bsz = e.shape[0]
    e_f = e.reshape(Bsz, -1)
    xh_f = xhat.reshape(Bsz, -1)
    xt_f = xtrue.reshape(Bsz, -1)
    e_norm = e_f.norm(dim=1).clamp_min(eps)
    xh_norm = xh_f.norm(dim=1).clamp_min(eps)
    xt_norm = xt_f.norm(dim=1).clamp_min(eps)
    tau = tau.reshape(-1).to(e.device, e.dtype)
    sin1 = torch.sin(2.0 * math.pi * tau)
    cos1 = torch.cos(2.0 * math.pi * tau)

    energy = torch.stack([e_norm.square()], dim=1)
    norm = torch.stack([e_norm, e_norm.square()], dim=1)
    state_feat = torch.stack([
        xh_norm,
        xh_norm.square(),
        xt_norm,
        xt_norm.square(),
        (xh_f * xt_f).sum(dim=1) / (xh_norm * xt_norm).clamp_min(eps),
    ], dim=1)
    interaction = torch.stack([
        (e_f * xh_f).sum(dim=1) / (e_norm * xh_norm).clamp_min(eps),
        (e_f * xt_f).sum(dim=1) / (e_norm * xt_norm).clamp_min(eps),
        (xh_f - xt_f).norm(dim=1).square(),
    ], dim=1)
    time_feat = torch.stack([tau, tau.square(), sin1, cos1], dim=1)
    full_no_time = torch.cat([norm, state_feat, interaction], dim=1)
    full = torch.cat([norm, state_feat, interaction, time_feat], dim=1)
    return {
        "energy": energy,
        "norm": norm,
        "state": state_feat,
        "interaction": interaction,
        "time": time_feat,
        "full_no_time": full_no_time,
        "full": full,
    }


def _coboundary_potential_features(
    e: torch.Tensor,
    xhat: torch.Tensor,
    xtrue: torch.Tensor,
    tau: torch.Tensor,
    eps: float,
    family: str = "full",
) -> torch.Tensor:
    bank = _coboundary_feature_bank(e, xhat, xtrue, tau, eps)
    family = family.lower()
    if family == "fit_full":
        family = "full"
    if family == "fit_full_no_time":
        family = "full_no_time"
    if family not in bank:
        family = "full"
    return bank[family]


def _fit_coboundary_delta(a: torch.Tensor, d_q: torch.Tensor, ridge: float, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Diagnostic-matched pooled ridge fit for a_t ~= Delta q_t^T w.

    This mirrors scripts/diagnose_hcp_coboundary.py::ridge_fit_metrics:
      * flatten all [batch, step] samples into one design matrix;
      * standardize each feature column across the pooled samples;
      * include an unregularized intercept;
      * fit one shared w for the current minibatch/rollout collection.

    This is deliberately NOT a per-sequence fit.  The offline diagnostic that
    gave residual_frac≈0.48 uses one pooled/global fit, so training must use the
    same geometry if we want the logged residual to mean the same thing.
    """
    if a.numel() == 0 or d_q.numel() == 0:
        z = a.new_zeros(a.shape)
        return z, a, a.new_zeros(a.shape[0])
    Bsz, N, F = d_q.shape
    y = a.reshape(-1)                      # [M]
    D = d_q.reshape(Bsz * N, F)            # [M,F]

    finite = torch.isfinite(y) & torch.isfinite(D).all(dim=1)
    if finite.sum() < max(5, F + 1):
        z = a.new_zeros(a.shape)
        return z, a, a.new_zeros(a.shape[0])

    y_fit = y[finite]
    D_fit = D[finite]
    mu = D_fit.mean(dim=0, keepdim=True)
    sd = D_fit.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    X_fit = (D_fit - mu) / sd
    ones = torch.ones((X_fit.shape[0], 1), device=X_fit.device, dtype=X_fit.dtype)
    X_aug = torch.cat([X_fit, ones], dim=1)

    P = X_aug.shape[1]
    A = X_aug.transpose(0, 1) @ X_aug
    reg = float(ridge) * torch.eye(P, device=a.device, dtype=a.dtype)
    reg[-1, -1] = 0.0  # unregularized intercept, as in diagnostic script
    b = X_aug.transpose(0, 1) @ y_fit.unsqueeze(-1)
    try:
        w = torch.linalg.solve(A + reg, b).squeeze(-1)
    except RuntimeError:
        w = torch.linalg.lstsq(A + reg, b).solution.squeeze(-1)

    X_all = (D - mu) / sd
    X_all_aug = torch.cat([X_all, torch.ones((X_all.shape[0], 1), device=a.device, dtype=a.dtype)], dim=1)
    pred_all = (X_all_aug @ w).reshape(Bsz, N)
    residual = a - pred_all

    ss_res = (y_fit - (X_aug @ w)).pow(2).mean()
    ss_tot = (y_fit - y_fit.mean()).pow(2).mean().clamp_min(eps)
    r2_scalar = 1.0 - ss_res / ss_tot
    r2 = r2_scalar.expand(Bsz)
    return pred_all, residual, r2


def _energy_coboundary_delta(
    a: torch.Tensor,
    d_energy: torch.Tensor,
    ridge: float,
    eps: float,
    fit_scale: bool,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Known-potential coboundary using Phi(e)=scale*||e||^2.

    This is the cheap/known-potential version of the coboundary filter.  The
    potential family is fixed to the error energy, so the loss no longer tries
    to explain a_t with a flexible low-dimensional feature set.  Optionally it
    fits only a single scalar beta per rollout sequence:

        a_t ~= beta * scale * (||e_{t+1}||^2 - ||e_t||^2).

    Setting fit_scale=False uses beta=1 and therefore a fully fixed potential.
    """
    if a.numel() == 0 or d_energy.numel() == 0:
        z = a.new_zeros(a.shape)
        return z, a, a.new_zeros(a.shape[0]), a.new_zeros(a.shape[0])
    x = float(scale) * d_energy
    if fit_scale:
        num = (x * a).sum(dim=1)
        den = x.pow(2).sum(dim=1).clamp_min(eps) + float(ridge)
        beta = num / den
    else:
        beta = torch.ones(a.shape[0], device=a.device, dtype=a.dtype)
    phi_delta = beta.unsqueeze(1) * x
    residual = a - phi_delta
    ss_res = residual.pow(2).mean(dim=1)
    ss_tot = (a - a.mean(dim=1, keepdim=True)).pow(2).mean(dim=1).clamp_min(eps)
    r2 = 1.0 - ss_res / ss_tot
    return phi_delta, residual, r2, beta.detach()


def compute_error_projection_coboundary_loss(raw, state, stim, t0: int, args) -> tuple[torch.Tensor, dict]:
    """Coboundary-filtered Poseido pseudo-target loss.

    V1 Poseido uses the raw local reinforcement scalar

        a_s = <e_s, b_s>.

    This coboundary variant first subtracts a boundary/potential-difference
    term and only uses the non-telescoping residual

        r_s = a_s - Delta Phi_s

    for the pseudo-target correction.  In the default mode, the potential is
    fixed to error energy, Phi(e)=c||e||^2, so this replaces the earlier flexible
    low-dimensional coboundary fit with a known-potential filter.
    """
    logs0 = _error_proj_coboundary_zero_logs()
    lam = float(getattr(args, "error_proj_cob_lambda", 0.0))
    if lam == 0.0 and raw.training:
        return state.new_tensor(0.0), logs0
    if not hasattr(raw, "step_history") and not hasattr(raw, "forward"):
        return state.new_tensor(0.0), logs0

    Bsz, T, _ = get_batch_time_shape(state)
    W = int(args.window_size)
    K_req = int(getattr(args, "error_proj_cob_horizon", 64))
    K = max(1, min(K_req, T - int(t0)))
    if K <= 1 or int(t0) - W < 0:
        logs = dict(logs0)
        logs["ar/error_proj_cob_skipped_short"] = 1.0
        return state.new_tensor(0.0), logs
    if stim is None:
        stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))

    gamma = float(getattr(args, "error_proj_cob_gamma", 1.0))
    eps = float(getattr(args, "error_proj_cob_eps", 1e-8))
    ridge = float(getattr(args, "error_proj_cob_ridge", 1e-3))
    potential_mode = str(getattr(args, "error_proj_cob_potential", "energy")).lower()
    energy_scale = float(getattr(args, "error_proj_cob_energy_scale", 0.5))
    fit_scale = bool(getattr(args, "error_proj_cob_fit_scale", True))
    min_err = float(getattr(args, "error_proj_cob_min_error_norm", 1e-8))
    detach_norm = bool(getattr(args, "error_proj_cob_detach_norm", True))
    loss_name = str(getattr(args, "error_proj_cob_loss_type", getattr(args, "ar_loss", "rel_l2"))).lower()

    offsets = _parse_positive_int_list(str(getattr(args, "error_proj_cob_sources", "")), max_value=K)
    if not offsets:
        stride = max(1, int(getattr(args, "error_proj_cob_source_stride", 1)))
        offsets = list(range(2, K + 1, stride))
    max_sources = int(getattr(args, "error_proj_cob_max_sources", 0))
    if max_sources > 0 and len(offsets) > max_sources:
        offsets = offsets[:max_sources]
    source_set = set(offsets)

    rollout_records: list[dict] = []
    a_terms: list[torch.Tensor] = []
    dq_terms: list[torch.Tensor] = []
    d_energy_terms: list[torch.Tensor] = []

    with torch.no_grad():
        hist = time_window(state, int(t0) - W, int(t0)).detach()
        for sidx in range(1, K + 1):
            cur = int(t0) + (sidx - 1)
            sw = time_window(stim, cur - W, cur) if stim is not None else None
            prev_true = time_point(state, cur - 1, keep_time=False).detach()
            xhat_prev = hist[:, -1].detach()
            e_prev = (xhat_prev - prev_true).detach()
            tau0 = torch.full((Bsz,), float(sidx - 1) / float(max(K, 1)), device=state.device, dtype=state.dtype)
            q_prev = _coboundary_potential_features(e_prev, xhat_prev, prev_true, tau0, eps, family=potential_mode) if potential_mode.startswith("fit_") else None
            energy_prev = e_prev.reshape(Bsz, -1).pow(2).sum(dim=1)

            next_hist, pred = _ar_step_history_with_stim(raw, hist, sw)
            target = time_point(state, cur, keep_time=False).detach()
            b = (pred - target).detach()
            e_next = (pred - target).detach()
            tau1 = torch.full((Bsz,), float(sidx) / float(max(K, 1)), device=state.device, dtype=state.dtype)
            q_next = _coboundary_potential_features(e_next, pred.detach(), target, tau1, eps, family=potential_mode) if potential_mode.startswith("fit_") else None
            energy_next = e_next.reshape(Bsz, -1).pow(2).sum(dim=1)

            e_flat = e_prev.reshape(Bsz, -1)
            b_flat = b.reshape(Bsz, -1)
            a = (e_flat * b_flat).sum(dim=1)
            if potential_mode.startswith("fit_"):
                d_q = (q_next - q_prev).detach()
                dq_terms.append(d_q)
            d_energy_terms.append((energy_next - energy_prev).detach())

            a_terms.append(a.detach())
            rollout_records.append({
                "sidx": sidx,
                "hist": hist.detach(),
                "sw": sw.detach() if sw is not None else None,
                "pred": pred.detach(),
                "target": target,
                "b": b,
                "e_prev": e_prev,
                "a": a.detach(),
            })
            hist = next_hist.detach()

    if not rollout_records:
        logs = dict(logs0)
        logs["ar/error_proj_cob_skipped_short"] = 1.0
        return state.new_tensor(0.0), logs

    a_mat = torch.stack(a_terms, dim=1)  # [B,K]
    beta = a_mat.new_zeros(a_mat.shape[0])
    if potential_mode.startswith("fit_"):
        dq_mat = torch.stack(dq_terms, dim=1)  # [B,K,F]
        phi_delta, residual, r2 = _fit_coboundary_delta(a_mat, dq_mat, ridge=ridge, eps=eps)
    else:
        d_energy_mat = torch.stack(d_energy_terms, dim=1)  # [B,K]
        phi_delta, residual, r2, beta = _energy_coboundary_delta(
            a_mat, d_energy_mat, ridge=ridge, eps=eps, fit_scale=fit_scale, scale=energy_scale
        )
    phi_delta = phi_delta.detach()
    residual = residual.detach()
    beta = beta.detach()

    items: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]] = []
    cos_vals: list[torch.Tensor] = []
    pos_cos_vals: list[torch.Tensor] = []
    apply_fracs: list[torch.Tensor] = []
    raw_apply_fracs: list[torch.Tensor] = []
    defect_norms: list[torch.Tensor] = []
    err_norms: list[torch.Tensor] = []
    corr_norms: list[torch.Tensor] = []
    corr_fracs: list[torch.Tensor] = []
    target_rels: list[torch.Tensor] = []
    a_logs: list[torch.Tensor] = []
    a_pos_logs: list[torch.Tensor] = []
    phi_logs: list[torch.Tensor] = []
    res_logs: list[torch.Tensor] = []
    res_pos_logs: list[torch.Tensor] = []
    res_frac_logs: list[torch.Tensor] = []

    with torch.no_grad():
        for j, rec in enumerate(rollout_records):
            if int(rec["sidx"]) not in source_set:
                continue
            e_prev = rec["e_prev"]
            b = rec["b"]
            target = rec["target"]
            pred = rec["pred"]
            e_flat = e_prev.reshape(Bsz, -1)
            b_flat = b.reshape(Bsz, -1)
            e_norm = e_flat.norm(dim=1)
            b_norm = b_flat.norm(dim=1)
            dot = rec["a"]
            cos = dot / (e_norm * b_norm).clamp_min(eps)
            valid = (e_norm > min_err) & (b_norm > min_err)

            r = residual[:, j]
            alpha_denom = e_flat.pow(2).sum(dim=1).clamp_min(eps)
            if detach_norm:
                alpha_denom = alpha_denom.detach()
            alpha = gamma * torch.clamp(r, min=0.0) / alpha_denom
            alpha = torch.where(valid, alpha, torch.zeros_like(alpha))
            view = [Bsz] + [1] * (b.dim() - 1)
            delta_y = -alpha.view(*view) * e_prev
            pseudo_target = (pred + delta_y).detach()
            items.append((rec["hist"], rec["sw"], pseudo_target))

            cos_valid = cos[valid]
            r_valid = r[valid]
            a_valid = dot[valid]
            phi_valid = phi_delta[:, j][valid]
            if cos_valid.numel() > 0:
                cos_vals.append(cos_valid.mean())
                pos = cos_valid[cos_valid > 0]
                pos_cos_vals.append(pos.mean() if pos.numel() > 0 else cos_valid.new_tensor(0.0))
                apply_fracs.append((r_valid > 0).float().mean())
                raw_apply_fracs.append((a_valid > 0).float().mean())
                a_logs.append(a_valid.mean())
                apos = a_valid[a_valid > 0]
                a_pos_logs.append(apos.mean() if apos.numel() > 0 else a_valid.new_tensor(0.0))
                phi_logs.append(phi_valid.mean())
                res_logs.append(r_valid.mean())
                rpos = r_valid[r_valid > 0]
                res_pos_logs.append(rpos.mean() if rpos.numel() > 0 else r_valid.new_tensor(0.0))
                res_frac_logs.append((r_valid.abs().mean() / a_valid.abs().mean().clamp_min(eps)).detach())
            else:
                z = dot.new_tensor(0.0)
                cos_vals.append(z); pos_cos_vals.append(z); apply_fracs.append(z); raw_apply_fracs.append(z)
                a_logs.append(z); a_pos_logs.append(z); phi_logs.append(z); res_logs.append(z); res_pos_logs.append(z); res_frac_logs.append(z)
            defect_norms.append(b_norm.mean())
            err_norms.append(e_norm.mean())
            cn = delta_y.reshape(Bsz, -1).norm(dim=1)
            corr_norms.append(cn.mean())
            corr_fracs.append((cn / b_norm.clamp_min(eps)).mean())
            target_rels.append(relative_l2(pseudo_target, target).detach())

    if not items:
        logs = dict(logs0)
        logs["ar/error_proj_cob_skipped_short"] = 1.0
        logs["ar/error_proj_cob_available_horizon"] = float(K)
        logs["ar/error_proj_cob_horizon"] = float(K)
        logs["ar/error_proj_cob_lambda"] = float(lam)
        logs["ar/error_proj_cob_gamma"] = float(gamma)
        logs["ar/error_proj_cob_ridge"] = float(ridge)
        logs["ar/error_proj_cob_potential_mode"] = 1.0 if potential_mode == "energy" else 2.0
        logs["ar/error_proj_cob_energy_scale"] = float(energy_scale)
        logs["ar/error_proj_cob_fit_scale"] = 1.0 if fit_scale else 0.0
        return state.new_tensor(0.0), logs

    losses: list[torch.Tensor] = []
    for hist_i, sw_i, pseudo_target_i in items:
        pred_i = _ar_forward_pred(raw, sw_i, hist_i.detach())
        if loss_name == "rel_l2":
            li = relative_l2(pred_i, pseudo_target_i)
        else:
            li = elementwise_state_loss(pred_i.unsqueeze(1), pseudo_target_i.unsqueeze(1), loss=loss_name)
        losses.append(li)
    loss = torch.stack(losses).mean()

    def mean_log(xs: list[torch.Tensor]) -> float:
        return float(torch.stack(xs).mean().detach().cpu()) if xs else 0.0

    logs = {
        "ar/error_proj_cob_loss": float(loss.detach().cpu()),
        "ar/error_proj_cob_lambda": float(lam),
        "ar/error_proj_cob_weighted_loss": float((lam * loss).detach().cpu()),
        "ar/error_proj_cob_horizon": float(K),
        "ar/error_proj_cob_available_horizon": float(K),
        "ar/error_proj_cob_num_sources": float(len(items)),
        "ar/error_proj_cob_num_fit_steps": float(len(rollout_records)),
        "ar/error_proj_cob_gamma": float(gamma),
        "ar/error_proj_cob_ridge": float(ridge),
        "ar/error_proj_cob_potential_mode": 1.0 if potential_mode == "energy" else 2.0,
        "ar/error_proj_cob_energy_scale": float(energy_scale),
        "ar/error_proj_cob_fit_scale": 1.0 if fit_scale else 0.0,
        "ar/error_proj_cob_fit_beta_mean": float(beta.mean().detach().cpu()) if beta.numel() > 0 else 0.0,
        "ar/error_proj_cob_apply_frac": mean_log(apply_fracs),
        "ar/error_proj_cob_raw_apply_frac": mean_log(raw_apply_fracs),
        "ar/error_proj_cob_align_cos_mean": mean_log(cos_vals),
        "ar/error_proj_cob_pos_cos_mean": mean_log(pos_cos_vals),
        "ar/error_proj_cob_a_mean": mean_log(a_logs),
        "ar/error_proj_cob_a_pos_mean": mean_log(a_pos_logs),
        "ar/error_proj_cob_phi_delta_mean": mean_log(phi_logs),
        "ar/error_proj_cob_residual_mean": mean_log(res_logs),
        "ar/error_proj_cob_residual_pos_mean": mean_log(res_pos_logs),
        "ar/error_proj_cob_residual_frac_of_raw": mean_log(res_frac_logs),
        "ar/error_proj_cob_explained_r2": float(r2.mean().detach().cpu()),
        "ar/error_proj_cob_defect_norm": mean_log(defect_norms),
        "ar/error_proj_cob_prev_error_norm": mean_log(err_norms),
        "ar/error_proj_cob_correction_norm": mean_log(corr_norms),
        "ar/error_proj_cob_correction_frac_of_defect": mean_log(corr_fracs),
        "ar/error_proj_cob_target_rel_l2": mean_log(target_rels),
        "ar/error_proj_cob_skipped_short": 0.0,
    }
    return loss, logs


def _error_proj_v2_zero_logs() -> dict:
    return {
        "ar/error_proj_v2_loss": 0.0,
        "ar/error_proj_v2_lambda": 0.0,
        "ar/error_proj_v2_weighted_loss": 0.0,
        "ar/error_proj_v2_horizon": 0.0,
        "ar/error_proj_v2_available_horizon": 0.0,
        "ar/error_proj_v2_num_pairs": 0.0,
        "ar/error_proj_v2_fd_eps": 0.0,
        "ar/error_proj_v2_gamma": 0.0,
        "ar/error_proj_v2_apply_frac": 0.0,
        "ar/error_proj_v2_cos_mean": 0.0,
        "ar/error_proj_v2_pos_cos_mean": 0.0,
        "ar/error_proj_v2_jb_norm": 0.0,
        "ar/error_proj_v2_b_next_norm": 0.0,
        "ar/error_proj_v2_b_cur_norm": 0.0,
        "ar/error_proj_v2_correction_norm": 0.0,
        "ar/error_proj_v2_correction_frac_of_defect": 0.0,
        "ar/error_proj_v2_target_rel_l2": 0.0,
        "ar/error_proj_v2_dir_scale": 0.0,
        "ar/error_proj_v2_skipped_short": 0.0,
    }


def compute_error_projection_v2_loss(raw, state, stim, t0: int, args) -> tuple[torch.Tensor, dict]:
    """V2 transported-defect pseudo-target loss.

    This replaces the previous V2 cosine-penalty implementation while keeping
    the same public flags (``--error_proj_v2_*``).  V1 Poseido uses the current
    rollout error ``e_s`` as the anti-reinforcement direction.  This V2 variant
    uses the transported local defect direction instead:

        b_s       = F(xhat_s, u_s) - x_{s+1},
        d_s       = J_s b_s,
        b_{s+1}   = F(xhat_{s+1}, u_{s+1}) - x_{s+2},
        delta     = - gamma * [<d_s,b_{s+1}>]_+ / (||d_s||^2 + eps) * d_s.

    The rollout and pseudo target are detached.  Gradients flow only through a
    fresh one-step prediction at the adjacent next state, which makes this much
    closer to the successful V1 pseudo-target mechanism than to a weak cosine
    regularizer.
    """
    logs0 = _error_proj_v2_zero_logs()
    lam = float(getattr(args, "error_proj_v2_lambda", 0.0))
    if lam == 0.0 and raw.training:
        return state.new_tensor(0.0), logs0
    if not hasattr(raw, "step_history") and not hasattr(raw, "forward"):
        return state.new_tensor(0.0), logs0

    Bsz, T, _ = get_batch_time_shape(state)
    W = int(args.window_size)
    K_req = int(getattr(args, "error_proj_v2_horizon", 64))
    # Need one extra target for b_{s+1}; cur_next <= T-1.
    K = max(1, min(K_req, T - int(t0) - 1))
    if K <= 1 or int(t0) - W < 0:
        logs = dict(logs0)
        logs["ar/error_proj_v2_skipped_short"] = 1.0
        return state.new_tensor(0.0), logs
    if stim is None:
        stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))

    gamma = float(getattr(args, "error_proj_v2_gamma", 1.0))
    eps = float(getattr(args, "error_proj_v2_eps", 1e-8))
    fd_eps = float(getattr(args, "error_proj_v2_fd_eps", 1e-3))
    min_norm = float(getattr(args, "error_proj_v2_min_norm", 1e-8))
    normalize_dir = bool(getattr(args, "error_proj_v2_normalize_direction", True))
    detach_norm = bool(getattr(args, "error_proj_v2_detach_norm", True))
    loss_name = str(getattr(args, "error_proj_v2_loss_type", getattr(args, "ar_loss", "rel_l2"))).lower()

    # Source offsets are 1-indexed rollout steps for b_s.  For source s we need
    # the adjacent next defect b_{s+1}, so s must be <= K.
    offsets = _parse_positive_int_list(str(getattr(args, "error_proj_v2_sources", "")), max_value=K)
    if not offsets:
        stride = max(1, int(getattr(args, "error_proj_v2_source_stride", 1)))
        offsets = list(range(1, K + 1, stride))
    max_pairs = int(getattr(args, "error_proj_v2_max_pairs", 0))
    if max_pairs > 0 and len(offsets) > max_pairs:
        offsets = offsets[:max_pairs]
    source_set = set(offsets)

    # Each item stores the adjacent-next history and pseudo target.  Training is
    # only a fresh one-step fit at hist_next -> pseudo_target_next.
    items: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]] = []
    cos_vals: list[torch.Tensor] = []
    pos_cos_vals: list[torch.Tensor] = []
    apply_fracs: list[torch.Tensor] = []
    jb_norm_logs: list[torch.Tensor] = []
    b_next_norm_logs: list[torch.Tensor] = []
    b_cur_norm_logs: list[torch.Tensor] = []
    corr_norms: list[torch.Tensor] = []
    corr_fracs: list[torch.Tensor] = []
    target_rels: list[torch.Tensor] = []
    dir_scale_logs: list[torch.Tensor] = []

    with torch.no_grad():
        hist = time_window(state, int(t0) - W, int(t0)).detach()
        for sidx in range(1, K + 2):
            cur = int(t0) + (sidx - 1)
            sw = time_window(stim, cur - W, cur) if stim is not None else None
            next_hist, pred = _ar_step_history_with_stim(raw, hist, sw)
            target = time_point(state, cur, keep_time=False).detach()
            b_s = (pred - target).detach()

            if sidx in source_set:
                # Finite-difference J_s b_s around the detached rollout history.
                b_dir = b_s.detach()
                if normalize_dir:
                    flat = b_dir.reshape(Bsz, -1)
                    dir_scale = flat.norm(dim=1).clamp_min(min_norm) / (flat.shape[1] ** 0.5)
                    view = [Bsz] + [1] * (b_dir.dim() - 1)
                    direction = b_dir / dir_scale.view(*view)
                    multiplier = dir_scale.view(*view)
                else:
                    dir_scale = b_dir.reshape(Bsz, -1).norm(dim=1).clamp_min(min_norm)
                    direction = b_dir
                    multiplier = 1.0

                hist_pert = hist.detach().clone()
                hist_pert[:, -1] = hist_pert[:, -1] + fd_eps * direction
                pred_pert = _ar_forward_pred(raw, sw.detach() if sw is not None else None, hist_pert)
                j_b = (pred_pert - pred) / max(fd_eps, 1e-12)
                if normalize_dir:
                    j_b = j_b * multiplier
                j_b = j_b.detach()

                # Adjacent next defect b_{s+1} from the detached rollout state.
                cur_next = cur + 1
                sw_next = time_window(stim, cur_next - W, cur_next) if stim is not None else None
                next_hist_det = next_hist.detach()
                pred_next = _ar_forward_pred(raw, sw_next.detach() if sw_next is not None else None, next_hist_det)
                target_next = time_point(state, cur_next, keep_time=False).detach()
                b_next = (pred_next - target_next).detach()

                jb_flat = j_b.reshape(Bsz, -1)
                bn_flat = b_next.reshape(Bsz, -1)
                dot = (jb_flat * bn_flat).sum(dim=1)
                jb_norm = jb_flat.norm(dim=1)
                bn_norm = bn_flat.norm(dim=1)
                valid = (jb_norm > min_norm) & (bn_norm > min_norm)
                cos = dot / (jb_norm * bn_norm).clamp_min(eps)

                denom = jb_flat.pow(2).sum(dim=1).clamp_min(eps)
                if detach_norm:
                    denom = denom.detach()
                alpha = gamma * torch.clamp(dot, min=0.0) / denom
                alpha = torch.where(valid, alpha, torch.zeros_like(alpha))
                view = [Bsz] + [1] * (j_b.dim() - 1)
                delta_y = -alpha.view(*view) * j_b
                pseudo_target = (pred_next + delta_y).detach()

                items.append((next_hist_det, sw_next.detach() if sw_next is not None else None, pseudo_target))

                cos_valid = cos[valid]
                if cos_valid.numel() > 0:
                    cos_vals.append(cos_valid.mean())
                    pos = cos_valid[cos_valid > 0]
                    pos_cos_vals.append(pos.mean() if pos.numel() > 0 else cos_valid.new_tensor(0.0))
                    apply_fracs.append((cos_valid > 0).float().mean())
                else:
                    cos_vals.append(cos.new_tensor(0.0))
                    pos_cos_vals.append(cos.new_tensor(0.0))
                    apply_fracs.append(cos.new_tensor(0.0))
                jb_norm_logs.append(jb_norm.mean())
                b_next_norm_logs.append(bn_norm.mean())
                b_cur_norm_logs.append(b_s.reshape(Bsz, -1).norm(dim=1).mean())
                cn = delta_y.reshape(Bsz, -1).norm(dim=1)
                corr_norms.append(cn.mean())
                corr_fracs.append((cn / bn_norm.clamp_min(eps)).mean())
                target_rels.append(relative_l2(pseudo_target, target_next).detach())
                dir_scale_logs.append(dir_scale.mean() if torch.is_tensor(dir_scale) else cos.new_tensor(float(dir_scale)))

            hist = next_hist.detach()

    if not items:
        logs = dict(logs0)
        logs["ar/error_proj_v2_skipped_short"] = 1.0
        logs["ar/error_proj_v2_available_horizon"] = float(K)
        logs["ar/error_proj_v2_horizon"] = float(K)
        logs["ar/error_proj_v2_lambda"] = float(lam)
        logs["ar/error_proj_v2_fd_eps"] = float(fd_eps)
        logs["ar/error_proj_v2_gamma"] = float(gamma)
        return state.new_tensor(0.0), logs

    losses: list[torch.Tensor] = []
    for hist_i, sw_i, pseudo_target_i in items:
        pred_i = _ar_forward_pred(raw, sw_i, hist_i.detach())
        if loss_name == "rel_l2":
            li = relative_l2(pred_i, pseudo_target_i)
        else:
            li = elementwise_state_loss(pred_i.unsqueeze(1), pseudo_target_i.unsqueeze(1), loss=loss_name)
        losses.append(li)
    loss = torch.stack(losses).mean()

    def mean_log(xs: list[torch.Tensor]) -> float:
        return float(torch.stack(xs).mean().detach().cpu()) if xs else 0.0

    logs = {
        "ar/error_proj_v2_loss": float(loss.detach().cpu()),
        "ar/error_proj_v2_lambda": float(lam),
        "ar/error_proj_v2_weighted_loss": float((lam * loss).detach().cpu()),
        "ar/error_proj_v2_horizon": float(K),
        "ar/error_proj_v2_available_horizon": float(K),
        "ar/error_proj_v2_num_pairs": float(len(items)),
        "ar/error_proj_v2_fd_eps": float(fd_eps),
        "ar/error_proj_v2_gamma": float(gamma),
        "ar/error_proj_v2_apply_frac": mean_log(apply_fracs),
        "ar/error_proj_v2_cos_mean": mean_log(cos_vals),
        "ar/error_proj_v2_pos_cos_mean": mean_log(pos_cos_vals),
        "ar/error_proj_v2_jb_norm": mean_log(jb_norm_logs),
        "ar/error_proj_v2_b_next_norm": mean_log(b_next_norm_logs),
        "ar/error_proj_v2_b_cur_norm": mean_log(b_cur_norm_logs),
        "ar/error_proj_v2_correction_norm": mean_log(corr_norms),
        "ar/error_proj_v2_correction_frac_of_defect": mean_log(corr_fracs),
        "ar/error_proj_v2_target_rel_l2": mean_log(target_rels),
        "ar/error_proj_v2_dir_scale": mean_log(dir_scale_logs),
        "ar/error_proj_v2_skipped_short": 0.0,
    }
    return loss, logs

def compute_error_transport_metric_loss(
    raw,
    state: torch.Tensor,
    stim: torch.Tensor | None,
    t0: int,
    args,
    pred_first: torch.Tensor | None = None,
    target_first: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Learned full-matrix Error-Transport Metric loss for vector Transformer AR.

    Old endpoint-only behavior:
        fit:  J_{t,K} stop(e_{t+1}) ~= stop(e_{t+K})

    New multi-step behavior, enabled by --etm_fit_all_steps:
        fit:  mean_{k in H} || J_{t,k} stop(e_{t+1}) - stop(e_{t+k}) ||^2
        prop: mean/sum_{k in H} || stop(J_{t,k}) e_{t+1} ||^2

    Optional current-source transport mode, enabled by --etm_prop_mode current:
        Let b_t = e_{t+1} be the current teacher-forced one-step residual.
        Build two no-grad model rollouts from x_{t+1}+b_t and x_{t+1}; use
        their difference Delta_{t,k} = Phi(x_{t+1}+b_t)-Phi(x_{t+1}) as the
        clean target for the future impact of the current residual:

            fit-current: || J_{t,k} stop(b_t) - stop(Delta_{t,k}) ||^2
            prop-current: || stop(J_{t,k}) b_t ||^2

        Future teacher-forced residual sources are not included in this mode;
        they are handled when those future time points are sampled as the
        current one-step residual.

    Optional detached teacher-forced residual source terms, enabled by
    --etm_include_tf_residuals:
        fit: mean_{k in H} || J_{t,k} stop(e_{t+1})
                          + sum_{r=1}^{k-1} J_{t+r,k-r} stop(b^{TF}_{t+r})
                          - stop(e_{t+k}) ||^2

    where b^{TF}_{t+r} is computed using GT history only and detached for the
    fit branch.  In --etm_prop_mode accum/source, the prop branch recomputes
    these teacher-forced residual sources with gradients and optimizes

        prop-accum: || stop(J_{t,k}) e_{t+1}
                    + sum_r stop(J_{t+r,k-r}) b^{TF}_{t+r} ||^2

    so ETM becomes a transport-reweighted residual-source sequence diagnostic.

    Guidance/excess modes for avoiding one-step degeneration:
        Let M_k = J_k^T J_k.  Plain ||J_k b||^2 contains the ordinary
        one-step identity baseline ||b||^2, so it can collapse into a rescaled
        one-step loss when M_k ~= I.  The guide/excess modes subtract this
        baseline and optimize only the future-specific amplification:

            guide:       mean_k [ b^T (M_k-I) b ]_+
            geom_excess: [ sum_k b^T (M_k-I) b ]_+

        This is analogous to classifier-free guidance: use the difference
        between a future-aware score M_k b and the one-step score b, rather
        than replacing the one-step score by M_k b.

    This gives the transport head dense supervision from many rollout errors for
    the same sampled start time, instead of a single final endpoint.  Use
    --etm_fit_step_stride to reduce cost, e.g. stride=4 uses k=1,5,9,...,K.
    """
    zero = state.new_tensor(0.0)
    if not bool(getattr(args, "etm_loss", False)):
        return zero, zero, _etm_zero_logs()
    if not hasattr(raw, "predict_transport_matrix"):
        logs = _etm_zero_logs()
        logs["ar/etm_supported"] = 0.0
        return zero, zero, logs

    Bsz, T, _shape = get_batch_time_shape(state)
    W = int(args.window_size)
    K_req = int(getattr(args, "etm_horizon", 0))
    if K_req <= 0:
        K_req = int(getattr(args, "koopman_gramian_horizon", 8))
    K_req = max(1, int(K_req))

    # Critical: do not silently shorten K near the end of a sequence unless the
    # user explicitly disables strict horizon.  Mixing J_{t,64}, J_{t,47}, ...
    # was the main source of inconsistent ETM supervision.
    strict_horizon = bool(getattr(args, "etm_strict_horizon", True))
    max_available = T - int(t0)
    if strict_horizon and max_available < K_req:
        logs = _etm_zero_logs()
        logs["ar/etm_supported"] = 0.0
        logs["ar/etm_skipped_short_horizon"] = 1.0
        logs["ar/etm_requested_horizon"] = float(K_req)
        logs["ar/etm_available_horizon"] = float(max_available)
        return zero, zero, logs
    K = max(1, min(K_req, max_available))
    if K <= 0 or int(t0) - W < 0:
        return zero, zero, _etm_zero_logs()

    horizons = _etm_selected_horizons(K, args)
    if len(horizons) == 0:
        return zero, zero, _etm_zero_logs()

    history = time_window(state, int(t0) - W, int(t0))
    stim_window = time_window(stim, int(t0) - W, int(t0)) if stim is not None else None
    if target_first is None:
        target_first = time_point(state, int(t0), keep_time=False)
    else:
        target_first = target_first[:, 0] if target_first.dim() == state.dim() else target_first
    if pred_first is None:
        pred_first = _ar_forward_pred(raw, stim_window, history)
    e1 = pred_first - target_first

    # No-gradient rollout once; collect every step's observed rollout error.
    # The ETM fit can then supervise multiple horizons without rerolling for each k.
    # For a stable target, optionally disable dropout while constructing these
    # detached rollout-sensitivity targets.  This does not change the main
    # one-step training forward; it only denoises the auxiliary target for J.
    errors: list[torch.Tensor] = []
    rels: list[torch.Tensor] = []
    # Detached rollout histories before each predicted step.  These are used by
    # the path-residual metric prototype: for source step r, recompute
    # f_theta(stopgrad(hat{s}_{t+r}), u_{t+r}) with gradients and penalize its
    # local residual using a long-horizon metric.
    rollout_histories: list[torch.Tensor] = []
    _use_eval_for_etm_targets = bool(getattr(args, "etm_rollout_sensitivity_eval_mode", True))
    _was_training_for_etm_targets = bool(getattr(raw, "training", False))
    if _use_eval_for_etm_targets and _was_training_for_etm_targets:
        raw.eval()
    try:
        with torch.no_grad():
            hist = history.detach()
            for s in range(K):
                cur = int(t0) + s
                sw = time_window(stim, cur - W, cur) if stim is not None else None
                rollout_histories.append(hist.detach())
                hist, p = _ar_step_history_with_stim(raw, hist, sw)
                tgt = time_point(state, cur, keep_time=False).detach()
                errors.append((p.detach() - tgt).detach())
                rels.append(relative_l2(p, tgt).detach())
    finally:
        if _use_eval_for_etm_targets and _was_training_for_etm_targets:
            raw.train()

    # Attention-style trajectory-aware metric branch.  This deliberately does
    # not fit a transport Jacobian J e ~= Delta.  Instead it trains a PSD
    # low-rank metric M_t = B diag(a_t) B^T so that the scalar score
    #   s_t = e_{t+1}^T M_t e_{t+1}
    # predicts/ranks detached long-rollout loss.  During the prop branch the
    # metric is detached, so gradients flow only through e_{t+1}; the metric
    # cannot collapse to zero to reduce the main AR loss.
    metric_param = str(getattr(args, "etm_transport_param", "full")).lower()
    metric_fit_target = str(getattr(args, "etm_fit_target", "sensitivity")).lower()
    use_path_metric = metric_param in {"path_attn_metric", "path_metric", "rollout_path_metric"}
    use_attention_metric = (
        metric_param in {"attn_metric", "attention_metric", "traj_metric", "path_attn_metric", "path_metric", "rollout_path_metric"}
        or metric_fit_target == "long_loss_metric"
    )

    # Prototype-1: rollout-path residual metric.
    # Instead of scoring only the first residual e_{t+1}, score every local
    # residual source along a detached rollout path:
    #
    #   rho_r = f_theta(stopgrad(\hat{s}_{t+r}), u_{t+r}) - x_{t+r+1}
    #   L_prop = sum_{h in H} mean_{r<h} rho_r^T M_{r,h} rho_r.
    #
    # The rollout histories are no-grad, so this is not full BPTT.  However,
    # each local residual is recomputed with gradients, so the predictor is
    # trained at model-induced states instead of only at the first true state.
    # The metric head is still fitted to detached long-rollout loss and is
    # detached in the prop branch, preventing metric collapse.
    if use_path_metric:
        if not hasattr(raw, "predict_error_metric_score"):
            logs = _etm_zero_logs()
            logs["ar/etm_supported"] = 0.0
            logs["ar/etm_attn_metric"] = 1.0
            logs["ar/etm_path_metric"] = 1.0
            return zero, zero, logs

        version = str(getattr(args, "etm_version", "A")).upper()
        detach_context = bool(getattr(args, "etm_detach_context", True))
        detach_residual_fit = bool(getattr(args, "etm_detach_e1_fit", True))
        eps = 1e-8
        rank_margin = float(getattr(args, "etm_metric_rank_margin", 0.2))
        rank_weight = float(getattr(args, "etm_metric_rank_weight", 0.5))
        log_fit_weight = float(getattr(args, "etm_metric_log_fit_weight", 1.0))
        source_stride = max(1, int(getattr(args, "etm_path_source_stride", getattr(args, "etm_fit_step_stride", 1))))
        max_sources = int(getattr(args, "etm_path_max_sources", 0))

        fit_terms = []
        prop_terms = []
        log_r2_terms = []
        rank_loss_terms = []
        rank_acc_terms = []
        score_mean_terms = []
        target_mean_terms = []
        score_to_res_terms = []
        attn_entropy_terms = []
        attn_max_terms = []
        score_scale_terms = []
        num_source_terms = 0
        first_rel = rels[0]
        last_rel = rels[-1]

        for h in horizons:
            h_int = int(h)
            long_err = errors[h_int - 1].detach()
            long_target = long_err.reshape(Bsz, -1).pow(2).mean(dim=1).detach()
            log_target = torch.log(long_target.clamp_min(eps))

            source_indices = list(range(0, h_int, source_stride))
            if (h_int - 1) not in source_indices:
                source_indices.append(h_int - 1)
            source_indices = sorted(set(source_indices))
            if max_sources > 0 and len(source_indices) > max_sources:
                # Keep a deterministic, approximately uniform subset including endpoints.
                if max_sources == 1:
                    source_indices = [source_indices[-1]]
                else:
                    pos = torch.linspace(0, len(source_indices) - 1, steps=max_sources).round().long().tolist()
                    source_indices = [source_indices[i] for i in pos]
                    source_indices = sorted(set(source_indices))

            h_fit_terms = []
            h_prop_terms = []
            h_log_scores_for_r2 = []

            for r in source_indices:
                cur = int(t0) + int(r)
                hist_r = rollout_histories[int(r)].detach()
                sw_r = time_window(stim, cur - W, cur) if stim is not None else None
                tgt_r = time_point(state, cur, keep_time=False)

                # Recompute the local source residual with gradients through model
                # parameters, but no gradients through the rollout history.
                pred_r = _ar_forward_pred(raw, sw_r, hist_r)
                residual_r = pred_r - tgt_r
                residual_fit = residual_r.detach() if detach_residual_fit else residual_r

                future_stim = None
                rem_horizon = max(1, h_int - int(r))
                if version == "B":
                    future_stim = _future_stim_window_for_etm(stim, cur, int(t0) + h_int)

                score_fit, aux = raw.predict_error_metric_score(
                    hist_r,
                    residual_fit,
                    stim_window=sw_r,
                    future_stim=future_stim,
                    horizon=rem_horizon,
                    version=version,
                    detach_context=detach_context,
                    detach_metric=False,
                )
                log_score = torch.log(score_fit.clamp_min(eps))
                log_fit = (log_score - log_target).pow(2).mean()

                if Bsz > 1 and rank_weight != 0.0:
                    target_diff = log_target[:, None] - log_target[None, :]
                    score_diff = log_score[:, None] - log_score[None, :]
                    sign = target_diff.sign()
                    mask = target_diff.abs() > 1e-3
                    if mask.any():
                        rank_loss = torch.relu(rank_margin - sign[mask] * score_diff[mask]).mean()
                        rank_acc = ((sign[mask] * score_diff[mask]) > 0).float().mean()
                    else:
                        rank_loss = score_fit.new_tensor(0.0)
                        rank_acc = score_fit.new_tensor(0.0)
                else:
                    rank_loss = score_fit.new_tensor(0.0)
                    rank_acc = score_fit.new_tensor(0.0)

                h_fit_terms.append(log_fit_weight * log_fit + rank_weight * rank_loss)
                rank_loss_terms.append(rank_loss.detach())
                rank_acc_terms.append(rank_acc.detach())

                score_prop, aux_prop = raw.predict_error_metric_score(
                    hist_r,
                    residual_r,
                    stim_window=sw_r,
                    future_stim=future_stim,
                    horizon=rem_horizon,
                    version=version,
                    detach_context=True,
                    detach_metric=True,
                )
                h_prop_terms.append(score_prop.mean())

                with torch.no_grad():
                    h_log_scores_for_r2.append(log_score.detach().unsqueeze(-1))
                    score_mean_terms.append(score_fit.detach().mean())
                    target_mean_terms.append(long_target.detach().mean())
                    res_mse = residual_r.detach().reshape(Bsz, -1).pow(2).mean(dim=1).mean().clamp_min(eps)
                    score_to_res_terms.append((score_fit.detach().mean() / res_mse).detach())
                    attn_entropy_terms.append(aux["attn_entropy"].detach())
                    attn_max_terms.append(aux["attn_max"].detach())
                    score_scale_terms.append(aux["score_scale"].detach())
                num_source_terms += 1

            fit_terms.append(torch.stack(h_fit_terms).mean())
            source_reduce = str(getattr(args, "etm_path_source_reduce", "mean")).lower()
            h_prop = torch.stack(h_prop_terms).sum() if source_reduce == "sum" else torch.stack(h_prop_terms).mean()
            prop_terms.append(h_prop)

            with torch.no_grad():
                # Mean source score for this horizon versus the same long target.
                mean_log_score = torch.stack(h_log_scores_for_r2, dim=0).mean(dim=0)
                log_r2_terms.append(_etm_batch_r2(mean_log_score, log_target.detach().unsqueeze(-1)))

        fit_loss = torch.stack(fit_terms).mean()
        prop_reduce = str(getattr(args, "etm_prop_reduce", "mean")).lower()
        prop_loss = torch.stack(prop_terms).sum() if prop_reduce == "sum" else torch.stack(prop_terms).mean()

        with torch.no_grad():
            e1_norm = e1.detach().reshape(Bsz, -1).norm(dim=1).mean()
            eK = errors[horizons[-1] - 1].detach()
            eK_norm = eK.reshape(Bsz, -1).norm(dim=1).mean()
            logs = {
                "ar/etm_supported": 1.0,
                "ar/etm_attn_metric": 1.0,
                "ar/etm_path_metric": 1.0,
                "ar/etm_horizon": float(K),
                "ar/etm_requested_horizon": float(K_req),
                "ar/etm_available_horizon": float(max_available),
                "ar/etm_num_fit_steps": float(len(horizons)),
                "ar/etm_path_num_sources": float(num_source_terms),
                "ar/etm_path_source_stride": float(source_stride),
                "ar/etm_fit_step_stride": float(max(1, int(getattr(args, "etm_fit_step_stride", 1)))),
                "ar/etm_fit_all_steps": 1.0 if bool(getattr(args, "etm_fit_all_steps", False)) else 0.0,
                "ar/etm_skipped_short_horizon": 0.0,
                "ar/etm_version_is_B": 1.0 if version == "B" else 0.0,
                "ar/etm_fit_loss": float(fit_loss.detach().cpu()),
                "ar/etm_prop_loss": float(prop_loss.detach().cpu()),
                "ar/etm_fit_r2": float(torch.stack(log_r2_terms).mean().detach().cpu()),
                "ar/etm_metric_log_r2_long_loss": float(torch.stack(log_r2_terms).mean().detach().cpu()),
                "ar/etm_metric_rank_loss": float(torch.stack(rank_loss_terms).mean().detach().cpu()),
                "ar/etm_metric_rank_acc": float(torch.stack(rank_acc_terms).mean().detach().cpu()),
                "ar/etm_metric_score_mean": float(torch.stack(score_mean_terms).mean().detach().cpu()),
                "ar/etm_metric_target_long_mse": float(torch.stack(target_mean_terms).mean().detach().cpu()),
                "ar/etm_metric_score_to_e1_mse": float(torch.stack(score_to_res_terms).mean().detach().cpu()),
                "ar/etm_metric_attn_entropy": float(torch.stack(attn_entropy_terms).mean().detach().cpu()),
                "ar/etm_metric_attn_max": float(torch.stack(attn_max_terms).mean().detach().cpu()),
                "ar/etm_metric_score_scale": float(torch.stack(score_scale_terms).mean().detach().cpu()),
                "ar/etm_e1_norm": float(e1_norm.detach().cpu()),
                "ar/etm_eK_norm": float(eK_norm.detach().cpu()),
                "ar/etm_first_rel_l2": float(first_rel.detach().cpu()),
                "ar/etm_last_rel_l2": float(last_rel.detach().cpu()),
                "ar/etm_prop_reduce_is_sum": 1.0 if prop_reduce == "sum" else 0.0,
                "ar/etm_stage_is_warmup": 1.0 if _etm_stage_name(args, getattr(args, "_current_epoch_for_logs", 0)) == "warmup" else 0.0,
                "ar/etm_stage_is_j_only": 1.0 if _etm_stage_name(args, getattr(args, "_current_epoch_for_logs", 0)) == "j_only" else 0.0,
                "ar/etm_stage_is_joint": 1.0 if _etm_stage_name(args, getattr(args, "_current_epoch_for_logs", 0)) == "joint" else 0.0,
                "ar/etm_fit_target_is_sensitivity": 0.0,
                "ar/etm_diag_rollout_sensitivity": 0.0,
                "ar/etm_include_tf_residuals": 0.0,
                "ar/etm_current_transport_target": 0.0,
                "ar/etm_fit_r2_initial_only": 0.0,
                "ar/etm_fit_r2_residual_only": 0.0,
                "ar/etm_fit_r2_delta_initial_only": 0.0,
                "ar/etm_initial_contrib_fraction": 0.0,
                "ar/etm_residual_contrib_fraction": 0.0,
                "ar/etm_eK_hat_norm": 0.0,
                "ar/etm_eK_hat_initial_norm": 0.0,
                "ar/etm_residual_contrib_norm": 0.0,
                "ar/etm_tf_residual_norm": 0.0,
                "ar/etm_tf_residual_over_rollout_error": 0.0,
                "ar/etm_diag_mean": 0.0,
                "ar/etm_diag_std": 0.0,
                "ar/etm_matrix_fro_norm": 0.0,
                "ar/etm_matrix_offdiag_fro_norm": 0.0,
                "ar/etm_metric_mean": 0.0,
                "ar/etm_metric_std": 0.0,
                "ar/etm_metric_min": 0.0,
                "ar/etm_metric_max": 0.0,
                "ar/etm_metric_delta_fro_norm": 0.0,
                "ar/etm_metric_delta_over_identity": 0.0,
                "ar/etm_prop_to_e1_mse": 0.0,
                "ar/etm_prop_delta_to_e1_mse": 0.0,
                "ar/etm_prop_vec_cos_e1": 0.0,
                "ar/etm_prop_grad_cos_one_step": 0.0,
                "ar/etm_prop_amplification": 0.0,
                "ar/etm_prop_mode_is_delta": 0.0,
                "ar/etm_prop_mode_is_excess": 0.0,
                "ar/etm_prop_mode_is_accum": 0.0,
                "ar/etm_prop_mode_is_source": 1.0,
                "ar/etm_prop_mode_is_current": 0.0,
                "ar/etm_prop_mode_is_guide": 0.0,
                "ar/etm_prop_mode_is_geom_excess": 0.0,
                "ar/etm_prop_mode_is_signed_guide": 0.0,
                "ar/etm_sens_target_cos_e1": 0.0,
                "ar/etm_sens_target_norm_ratio": 0.0,
                "ar/etm_sens_target_delta_to_e1_mse": 0.0,
                "ar/etm_sens_j_pred_cos_target": 0.0,
                "ar/etm_sens_j_pred_rel_error": 0.0,
                "ar/etm_sens_j_pred_r2_target": 0.0,
                "ar/etm_sens_target_cos_rollout": 0.0,
                "ar/etm_sens_target_r2_rollout": 0.0,
            }
        return fit_loss, prop_loss, logs

    if use_attention_metric:
        if not hasattr(raw, "predict_error_metric_score"):
            logs = _etm_zero_logs()
            logs["ar/etm_supported"] = 0.0
            logs["ar/etm_attn_metric"] = 1.0
            return zero, zero, logs

        version = str(getattr(args, "etm_version", "A")).upper()
        detach_context = bool(getattr(args, "etm_detach_context", True))
        e1_for_fit = e1.detach() if bool(getattr(args, "etm_detach_e1_fit", True)) else e1
        eps = 1e-8
        rank_margin = float(getattr(args, "etm_metric_rank_margin", 0.2))
        rank_weight = float(getattr(args, "etm_metric_rank_weight", 0.5))
        log_fit_weight = float(getattr(args, "etm_metric_log_fit_weight", 1.0))

        fit_terms = []
        prop_terms = []
        log_r2_terms = []
        rank_loss_terms = []
        rank_acc_terms = []
        score_mean_terms = []
        target_mean_terms = []
        score_to_e1_terms = []
        attn_entropy_terms = []
        attn_max_terms = []
        score_scale_terms = []
        first_rel = rels[0]
        last_rel = rels[-1]

        for h in horizons:
            future_stim = None
            if version == "B":
                future_stim = _future_stim_window_for_etm(stim, int(t0), int(t0) + int(h))

            # Detached per-sample long-horizon target.  It is the actual AR
            # rollout error against ground truth, not the two-rollout
            # sensitivity Delta.  This makes the metric learn long-horizon
            # relevance rather than identity-like first-error transport.
            long_err = errors[int(h) - 1].detach()
            long_target = long_err.reshape(Bsz, -1).pow(2).mean(dim=1).detach()

            score_fit, aux = raw.predict_error_metric_score(
                history,
                e1_for_fit,
                stim_window=stim_window,
                future_stim=future_stim,
                horizon=int(h),
                version=version,
                detach_context=detach_context,
                detach_metric=False,
            )
            log_score = torch.log(score_fit.clamp_min(eps))
            log_target = torch.log(long_target.clamp_min(eps))
            log_fit = (log_score - log_target).pow(2).mean()

            # Pairwise ranking: if sample i has larger long rollout loss than
            # sample j, its metric score should also be larger.  We use log
            # scores to reduce scale sensitivity.
            if Bsz > 1 and rank_weight != 0.0:
                target_diff = log_target[:, None] - log_target[None, :]
                score_diff = log_score[:, None] - log_score[None, :]
                sign = target_diff.sign()
                mask = target_diff.abs() > 1e-3
                if mask.any():
                    rank_loss = torch.relu(rank_margin - sign[mask] * score_diff[mask]).mean()
                    rank_acc = ((sign[mask] * score_diff[mask]) > 0).float().mean()
                else:
                    rank_loss = score_fit.new_tensor(0.0)
                    rank_acc = score_fit.new_tensor(0.0)
            else:
                rank_loss = score_fit.new_tensor(0.0)
                rank_acc = score_fit.new_tensor(0.0)

            fit_terms.append(log_fit_weight * log_fit + rank_weight * rank_loss)
            rank_loss_terms.append(rank_loss.detach())
            rank_acc_terms.append(rank_acc.detach())

            # Main AR loss: stop the metric parameters/context, keep gradient
            # through e1.  This gives a directional weighted one-step loss
            # without letting the metric head shrink itself to reduce loss.
            score_prop, aux_prop = raw.predict_error_metric_score(
                history,
                e1,
                stim_window=stim_window,
                future_stim=future_stim,
                horizon=int(h),
                version=version,
                detach_context=True,
                detach_metric=True,
            )
            prop_terms.append(score_prop.mean())

            with torch.no_grad():
                log_r2_terms.append(_etm_batch_r2(log_score.detach().unsqueeze(-1), log_target.detach().unsqueeze(-1)))
                score_mean_terms.append(score_fit.detach().mean())
                target_mean_terms.append(long_target.detach().mean())
                e1_mse = e1.detach().reshape(Bsz, -1).pow(2).mean(dim=1).mean().clamp_min(eps)
                score_to_e1_terms.append((score_fit.detach().mean() / e1_mse).detach())
                attn_entropy_terms.append(aux["attn_entropy"].detach())
                attn_max_terms.append(aux["attn_max"].detach())
                score_scale_terms.append(aux["score_scale"].detach())

        fit_loss = torch.stack(fit_terms).mean()
        prop_reduce = str(getattr(args, "etm_prop_reduce", "mean")).lower()
        prop_loss = torch.stack(prop_terms).sum() if prop_reduce == "sum" else torch.stack(prop_terms).mean()

        with torch.no_grad():
            e1_norm = e1.detach().reshape(Bsz, -1).norm(dim=1).mean()
            eK = errors[horizons[-1] - 1].detach()
            eK_norm = eK.reshape(Bsz, -1).norm(dim=1).mean()
            logs = {
                "ar/etm_supported": 1.0,
                "ar/etm_attn_metric": 1.0,
                "ar/etm_horizon": float(K),
                "ar/etm_requested_horizon": float(K_req),
                "ar/etm_available_horizon": float(max_available),
                "ar/etm_num_fit_steps": float(len(horizons)),
                "ar/etm_fit_step_stride": float(max(1, int(getattr(args, "etm_fit_step_stride", 1)))),
                "ar/etm_fit_all_steps": 1.0 if bool(getattr(args, "etm_fit_all_steps", False)) else 0.0,
                "ar/etm_skipped_short_horizon": 0.0,
                "ar/etm_version_is_B": 1.0 if version == "B" else 0.0,
                "ar/etm_fit_loss": float(fit_loss.detach().cpu()),
                "ar/etm_prop_loss": float(prop_loss.detach().cpu()),
                "ar/etm_fit_r2": float(torch.stack(log_r2_terms).mean().detach().cpu()),
                "ar/etm_metric_log_r2_long_loss": float(torch.stack(log_r2_terms).mean().detach().cpu()),
                "ar/etm_metric_rank_loss": float(torch.stack(rank_loss_terms).mean().detach().cpu()),
                "ar/etm_metric_rank_acc": float(torch.stack(rank_acc_terms).mean().detach().cpu()),
                "ar/etm_metric_score_mean": float(torch.stack(score_mean_terms).mean().detach().cpu()),
                "ar/etm_metric_target_long_mse": float(torch.stack(target_mean_terms).mean().detach().cpu()),
                "ar/etm_metric_score_to_e1_mse": float(torch.stack(score_to_e1_terms).mean().detach().cpu()),
                "ar/etm_metric_attn_entropy": float(torch.stack(attn_entropy_terms).mean().detach().cpu()),
                "ar/etm_metric_attn_max": float(torch.stack(attn_max_terms).mean().detach().cpu()),
                "ar/etm_metric_score_scale": float(torch.stack(score_scale_terms).mean().detach().cpu()),
                "ar/etm_e1_norm": float(e1_norm.detach().cpu()),
                "ar/etm_eK_norm": float(eK_norm.detach().cpu()),
                "ar/etm_first_rel_l2": float(first_rel.detach().cpu()),
                "ar/etm_last_rel_l2": float(last_rel.detach().cpu()),
                "ar/etm_prop_reduce_is_sum": 1.0 if prop_reduce == "sum" else 0.0,
                "ar/etm_stage_is_warmup": 1.0 if _etm_stage_name(args, getattr(args, "_current_epoch_for_logs", 0)) == "warmup" else 0.0,
                "ar/etm_stage_is_j_only": 1.0 if _etm_stage_name(args, getattr(args, "_current_epoch_for_logs", 0)) == "j_only" else 0.0,
                "ar/etm_stage_is_joint": 1.0 if _etm_stage_name(args, getattr(args, "_current_epoch_for_logs", 0)) == "joint" else 0.0,
                "ar/etm_fit_target_is_sensitivity": 0.0,
                "ar/etm_diag_rollout_sensitivity": 0.0,
                "ar/etm_include_tf_residuals": 0.0,
                "ar/etm_current_transport_target": 0.0,
                # Fill old transport diagnostics with zeros so downstream log
                # aggregation keeps a stable schema.
                "ar/etm_fit_r2_initial_only": 0.0,
                "ar/etm_fit_r2_residual_only": 0.0,
                "ar/etm_fit_r2_delta_initial_only": 0.0,
                "ar/etm_initial_contrib_fraction": 0.0,
                "ar/etm_residual_contrib_fraction": 0.0,
                "ar/etm_eK_hat_norm": 0.0,
                "ar/etm_eK_hat_initial_norm": 0.0,
                "ar/etm_residual_contrib_norm": 0.0,
                "ar/etm_tf_residual_norm": 0.0,
                "ar/etm_tf_residual_over_rollout_error": 0.0,
                "ar/etm_diag_mean": 0.0,
                "ar/etm_diag_std": 0.0,
                "ar/etm_matrix_fro_norm": 0.0,
                "ar/etm_matrix_offdiag_fro_norm": 0.0,
                "ar/etm_metric_mean": 0.0,
                "ar/etm_metric_std": 0.0,
                "ar/etm_metric_min": 0.0,
                "ar/etm_metric_max": 0.0,
                "ar/etm_metric_delta_fro_norm": 0.0,
                "ar/etm_metric_delta_over_identity": 0.0,
                "ar/etm_prop_to_e1_mse": 0.0,
                "ar/etm_prop_delta_to_e1_mse": 0.0,
                "ar/etm_prop_vec_cos_e1": 0.0,
                "ar/etm_prop_grad_cos_one_step": 0.0,
                "ar/etm_prop_amplification": 0.0,
                "ar/etm_prop_mode_is_delta": 0.0,
                "ar/etm_prop_mode_is_excess": 0.0,
                "ar/etm_prop_mode_is_accum": 0.0,
                "ar/etm_prop_mode_is_source": 0.0,
                "ar/etm_prop_mode_is_current": 0.0,
                "ar/etm_prop_mode_is_guide": 0.0,
                "ar/etm_prop_mode_is_geom_excess": 0.0,
                "ar/etm_prop_mode_is_signed_guide": 0.0,
                "ar/etm_sens_target_cos_e1": 0.0,
                "ar/etm_sens_target_norm_ratio": 0.0,
                "ar/etm_sens_target_delta_to_e1_mse": 0.0,
                "ar/etm_sens_j_pred_cos_target": 0.0,
                "ar/etm_sens_j_pred_rel_error": 0.0,
                "ar/etm_sens_j_pred_r2_target": 0.0,
                "ar/etm_sens_target_cos_rollout": 0.0,
                "ar/etm_sens_target_r2_rollout": 0.0,
            }
        return fit_loss, prop_loss, logs

    prop_mode = str(getattr(args, "etm_prop_mode", "full")).lower()
    current_transport_mode = prop_mode in {"current", "current_transport", "clean_current", "current_source"}
    fit_target_mode = str(getattr(args, "etm_fit_target", "sensitivity")).lower()
    fit_target_is_sensitivity = fit_target_mode in {"sensitivity", "delta", "rollout_sensitivity", "current"}
    diagnose_rollout_sensitivity = current_transport_mode or fit_target_is_sensitivity or bool(getattr(args, "etm_diag_rollout_sensitivity", False))

    # Test-time rollout-sensitivity diagnostic.  These are differences between
    # two model rollouts with the same future model and stimulus path:
    #   plus: start from x_{t+1}+e_{t+1} = first prediction
    #   zero: start from x_{t+1}
    # Their difference Delta_{t,k} = Phi(x_{t+1}+e_{t+1}) - Phi(x_{t+1})
    # is the clean test-time future effect of the first-step error.  In
    # current_transport_mode this is also used as the fit target; otherwise it
    # is diagnostics only and the main loss remains the original init-only ETM.
    current_deltas: list[torch.Tensor] = []
    if diagnose_rollout_sensitivity:
        _was_training_for_delta = bool(getattr(raw, "training", False))
        if _use_eval_for_etm_targets and _was_training_for_delta:
            raw.eval()
        try:
            with torch.no_grad():
                hist_plus = torch.cat([history.detach()[:, 1:], pred_first.detach().unsqueeze(1)], dim=1)
                hist_zero = torch.cat([history.detach()[:, 1:], target_first.detach().unsqueeze(1)], dim=1)
                current_deltas.append((pred_first.detach() - target_first.detach()).detach())
                for s in range(1, K):
                    cur = int(t0) + s
                    sw = time_window(stim, cur - W, cur) if stim is not None else None
                    hist_plus, p_plus = _ar_step_history_with_stim(raw, hist_plus, sw)
                    hist_zero, p_zero = _ar_step_history_with_stim(raw, hist_zero, sw)
                    current_deltas.append((p_plus.detach() - p_zero.detach()).detach())
        finally:
            if _use_eval_for_etm_targets and _was_training_for_delta:
                raw.train()

    # Detached teacher-forced one-step residual source terms b^{TF}_{t+s}.
    # These are NOT an optimization objective for the predictor.  They are
    # computed under no_grad and used only as fixed source vectors in the ETM
    # transport decomposition.  In current_transport_mode they are intentionally
    # disabled so future source errors cannot shortcut the current-residual
    # transport target.
    include_tf_residuals = bool(getattr(args, "etm_include_tf_residuals", False)) and (not current_transport_mode)
    tf_errors: list[torch.Tensor] = []
    if include_tf_residuals:
        with torch.no_grad():
            for s in range(K):
                cur = int(t0) + s
                hist_gt = time_window(state, cur - W, cur).detach()
                sw_gt = time_window(stim, cur - W, cur).detach() if stim is not None else None
                p_tf = _ar_forward_pred(raw, sw_gt, hist_gt).detach()
                tgt_tf = time_point(state, cur, keep_time=False).detach()
                tf_errors.append((p_tf - tgt_tf).detach())

    # For accumulated/source prop, the teacher-forced residual sources should
    # carry gradient to the predictor.  The J operators are still stopped in the
    # prop branch, but b^{TF}_{t+r}=f_theta(GT history)-GT must remain live so
    # the loss can suppress continuously injected one-step source errors.
    needs_live_tf_sources = include_tf_residuals and prop_mode in {"accum", "accumulate", "accumulated", "source", "source_only"}
    tf_errors_live: list[torch.Tensor] = []
    if needs_live_tf_sources:
        for s in range(K):
            cur = int(t0) + s
            hist_gt = time_window(state, cur - W, cur).detach()
            sw_gt = time_window(stim, cur - W, cur).detach() if stim is not None else None
            p_tf = _ar_forward_pred(raw, sw_gt, hist_gt)
            tgt_tf = time_point(state, cur, keep_time=False).detach()
            tf_errors_live.append(p_tf - tgt_tf)

    version = str(getattr(args, "etm_version", "A")).upper()
    detach_context = bool(getattr(args, "etm_detach_context", True))
    e1_for_fit = e1.detach() if bool(getattr(args, "etm_detach_e1_fit", True)) else e1

    fit_terms = []
    prop_terms = []
    r2_terms = []
    r2_initial_terms = []
    r2_residual_terms = []
    r2_delta_initial_terms = []
    eK_hat_norm_terms = []
    eK_hat_initial_norm_terms = []
    residual_contrib_norm_terms = []
    tf_error_norm_terms = []
    tf_error_ratio_terms = []
    J_fro_terms = []
    offdiag_fro_terms = []
    diag_mean_terms = []
    diag_std_terms = []
    metric_mean_terms = []
    metric_std_terms = []
    metric_min_terms = []
    metric_max_terms = []
    metric_delta_fro_terms = []
    metric_delta_over_I_terms = []
    prop_to_e1_mse_terms = []
    prop_delta_to_e1_mse_terms = []
    prop_vec_cos_e1_terms = []
    prop_grad_cos_one_step_terms = []
    prop_amplification_terms = []
    initial_contrib_fraction_terms = []
    residual_contrib_fraction_terms = []
    sens_target_cos_e1_terms = []
    sens_target_norm_ratio_terms = []
    sens_target_delta_to_e1_mse_terms = []
    sens_j_pred_cos_target_terms = []
    sens_j_pred_rel_error_terms = []
    sens_j_pred_r2_terms = []
    sens_target_cos_rollout_terms = []
    sens_target_r2_rollout_terms = []

    # Accumulators for guidance-style prop modes.  These modes explicitly
    # remove the identity/ordinary-MSE baseline so the auxiliary loss cannot
    # simply become another one-step MSE term.
    guide_future_mse_terms: list[torch.Tensor] = []
    guide_baseline_mse_terms: list[torch.Tensor] = []

    for h in horizons:
        # Fit target for J.  The clean test-time first-error transport target is
        # Delta_{t,k}=Phi(x_{t+1}+e1)-Phi(x_{t+1}).  The older observed-rollout
        # target e_{t+k}=Phi(x_{t+1}+e1)-x_{t+k} is still available for ablation
        # via --etm_fit_target rollout_error.
        if (current_transport_mode or fit_target_is_sensitivity) and len(current_deltas) >= int(h):
            eK = current_deltas[h - 1]
        else:
            eK = errors[h - 1]
        future_stim = None
        if version == "B":
            future_stim = _future_stim_window_for_etm(stim, int(t0), int(t0) + int(h))

        J_fit = raw.predict_transport_matrix(
            history,
            stim_window,
            future_stim=future_stim,
            version=version,
            detach_context=detach_context,
        )

        # Initial-error transport term: J_{t,h} e_{t+1}.
        eK_hat_initial = torch.bmm(J_fit, e1_for_fit.unsqueeze(-1)).squeeze(-1)

        # Optional detached teacher-forced residual source terms:
        #   sum_{r=1}^{h-1} J_{t+r,h-r} stop(b^{TF}_{t+r}).
        # This is intentionally detached in b^{TF}; gradients only train the
        # transport heads J, not the predictor through the teacher-forced residual.
        residual_contrib = torch.zeros_like(eK_hat_initial)
        tf_norm_for_h = eK_hat_initial.new_tensor(0.0)
        tf_ratio_for_h = eK_hat_initial.new_tensor(0.0)
        if include_tf_residuals and int(h) > 1 and len(tf_errors) >= int(h):
            residual_stride = max(1, int(getattr(args, "etm_residual_step_stride", 1)))
            used_residuals = 0
            tf_norm_acc = []
            tf_ratio_acc = []
            for r in range(1, int(h), residual_stride):
                b_r = tf_errors[r].detach()
                src = int(t0) + int(r)
                src_history = time_window(state, src - W, src)
                src_stim_window = time_window(stim, src - W, src) if stim is not None else None

                src_future_stim = None
                if version == "B":
                    src_future_stim = _future_stim_window_for_etm(stim, src, int(t0) + int(h))

                J_src = raw.predict_transport_matrix(
                    src_history,
                    src_stim_window,
                    future_stim=src_future_stim,
                    version=version,
                    detach_context=detach_context,
                )
                residual_contrib = residual_contrib + torch.bmm(
                    J_src, b_r.unsqueeze(-1)
                ).squeeze(-1)

                with torch.no_grad():
                    b_norm = b_r.reshape(Bsz, -1).norm(dim=1).mean()
                    e_norm = errors[r].reshape(Bsz, -1).norm(dim=1).mean().clamp_min(1e-12)
                    tf_norm_acc.append(b_norm.detach())
                    tf_ratio_acc.append((b_norm / e_norm).detach())
                used_residuals += 1

            if used_residuals > 0:
                tf_norm_for_h = torch.stack(tf_norm_acc).mean()
                tf_ratio_for_h = torch.stack(tf_ratio_acc).mean()

        eK_hat = eK_hat_initial + residual_contrib
        fit_terms.append((eK_hat - eK).pow(2).mean())

        # Prop branch: stop-gradient through J, but keep gradients through the
        # one-step residual sources.  The original mode ("full") penalizes only
        # the transported initial residual ||J_{t,h} e_{t+1}||^2.  The new
        # accumulated mode penalizes the ETM-estimated future rollout error,
        #
        #   || J_{t,h} e_{t+1} + sum_r J_{t+r,h-r} b^{TF}_{t+r} ||^2,
        #
        # so the whole teacher-forced residual-source sequence is reweighted by
        # its transported contribution to the h-step error.
        J_metric = _etm_normalize_transport_for_prop(J_fit, args)
        prop_vec_init = torch.bmm(J_metric, e1.unsqueeze(-1)).squeeze(-1)

        prop_vec_source = torch.zeros_like(prop_vec_init)
        if needs_live_tf_sources and int(h) > 1 and len(tf_errors_live) >= int(h):
            residual_stride = max(1, int(getattr(args, "etm_residual_step_stride", 1)))
            for r in range(1, int(h), residual_stride):
                b_r_live = tf_errors_live[r]
                src = int(t0) + int(r)
                src_history = time_window(state, src - W, src)
                src_stim_window = time_window(stim, src - W, src) if stim is not None else None

                src_future_stim = None
                if version == "B":
                    src_future_stim = _future_stim_window_for_etm(stim, src, int(t0) + int(h))

                J_src_prop = raw.predict_transport_matrix(
                    src_history,
                    src_stim_window,
                    future_stim=src_future_stim,
                    version=version,
                    detach_context=detach_context,
                )
                J_src_metric = _etm_normalize_transport_for_prop(J_src_prop, args)
                prop_vec_source = prop_vec_source + torch.bmm(
                    J_src_metric, b_r_live.unsqueeze(-1)
                ).squeeze(-1)

        prop_vec_accum = prop_vec_init + prop_vec_source

        # Optional prop variants.
        #   current:   clean current-source penalty ||J b_t||^2, with J fitted
        #              to two-rollout difference Phi(x_t+b_t)-Phi(x_t).
        #   full/init: original initial-only transport penalty ||J e1||^2.
        #   accum:     accumulated ETM error ||J e1 + sum J b||^2.
        #   source:    source-only accumulated term ||sum J b||^2.
        #   delta:     non-identity part of initial transport ||(J-I)e1||^2.
        #   excess/guide/cfg_excess:
        #              positive future-specific amplification
        #              [||J e1||^2 - ||e1||^2]_+.  Ordinary one-step MSE
        #              already supplies the identity score e1, so this mode
        #              adds only the CFG-like guidance correction
        #              (J^T J e1 - e1) on amplified directions.
        #   geom_excess:
        #              multi-horizon version [sum_k ||J_k e1||^2
        #              - H ||e1||^2]_+, computed after the loop.
        if current_transport_mode:
            prop_vec = prop_vec_init
            prop_loss_h = prop_vec.pow(2).mean()
        elif prop_mode in {"accum", "accumulate", "accumulated"}:
            prop_vec = prop_vec_accum
            prop_loss_h = prop_vec.pow(2).mean()
        elif prop_mode in {"source", "source_only"}:
            prop_vec = prop_vec_source
            prop_loss_h = prop_vec.pow(2).mean()
        elif prop_mode == "delta":
            prop_vec = prop_vec_init
            prop_loss_h = (prop_vec - e1).pow(2).mean()
        elif prop_mode in {"excess", "guide", "cfg", "cfg_excess", "future_guidance"}:
            prop_vec = prop_vec_init
            future_mse_per_sample = prop_vec.pow(2).mean(dim=1)
            base_mse_per_sample = e1.pow(2).mean(dim=1)
            prop_loss_h = (future_mse_per_sample - base_mse_per_sample).clamp_min(0.0).mean()
        elif prop_mode in {"geom", "geom_excess", "geometric", "geometric_excess"}:
            # Store raw future and identity-baseline energies.  After the loop
            # we subtract the multi-horizon identity baseline H||e1||^2, i.e.
            # the finite geometric-series baseline for rho=1.
            prop_vec = prop_vec_init
            guide_future_mse_terms.append(prop_vec.pow(2).mean(dim=1))
            guide_baseline_mse_terms.append(e1.pow(2).mean(dim=1))
            prop_loss_h = prop_vec.new_tensor(0.0)
        elif prop_mode in {"signed_guide", "score_correction"}:
            # Signed correction 0.5*(||J e||^2-||e||^2) whose gradient is
            # (J^T J-I)e.  This is closer to a score/guidance correction but
            # can be negative; use mainly for diagnostics.
            prop_vec = prop_vec_init
            prop_loss_h = 0.5 * (prop_vec.pow(2).mean() - e1.pow(2).mean())
        else:
            prop_vec = prop_vec_init
            prop_loss_h = prop_vec.pow(2).mean()
        prop_terms.append(prop_loss_h)

        with torch.no_grad():
            r2_terms.append(_etm_batch_r2(eK_hat, eK))
            r2_initial_terms.append(_etm_batch_r2(eK_hat_initial, eK))
            r2_residual_terms.append(_etm_batch_r2(residual_contrib, eK))
            r2_delta_initial_terms.append(_etm_batch_r2(eK_hat_initial - e1_for_fit.detach(), eK))

            # Test-time rollout sensitivity diagnostics.  These compare the
            # clean two-rollout target Delta_{t,k}=Phi(x_{t+1}+e1)-Phi(x_{t+1})
            # against the one-step error and against the learned J e1.  They tell
            # us whether the true test-time future effect is already identity-like
            # or whether the learned J failed to fit a non-trivial target.
            if diagnose_rollout_sensitivity and len(current_deltas) >= int(h):
                sens_delta = current_deltas[int(h) - 1].detach()
                pred_delta = eK_hat_initial.detach()
                e1_mse_diag = e1.detach().pow(2).mean().clamp_min(1e-12)
                sens_target_cos_e1_terms.append(_etm_batch_cos(sens_delta, e1.detach()))
                sens_target_norm_ratio_terms.append(
                    sens_delta.reshape(Bsz, -1).norm(dim=1).mean()
                    / e1.detach().reshape(Bsz, -1).norm(dim=1).mean().clamp_min(1e-12)
                )
                sens_target_delta_to_e1_mse_terms.append(((sens_delta - e1.detach()).pow(2).mean() / e1_mse_diag).detach())
                sens_j_pred_cos_target_terms.append(_etm_batch_cos(pred_delta, sens_delta))
                sens_j_pred_rel_error_terms.append(
                    (pred_delta - sens_delta).reshape(Bsz, -1).norm(dim=1).mean()
                    / sens_delta.reshape(Bsz, -1).norm(dim=1).mean().clamp_min(1e-12)
                )
                sens_j_pred_r2_terms.append(_etm_batch_r2(pred_delta, sens_delta))
                # How much of the observed rollout error is the test-time
                # first-error sensitivity target? This is diagnostic only.
                rollout_error_for_h = errors[int(h) - 1].detach()
                sens_target_cos_rollout_terms.append(_etm_batch_cos(sens_delta, rollout_error_for_h))
                sens_target_r2_rollout_terms.append(_etm_batch_r2(sens_delta, rollout_error_for_h))

            eK_hat_norm_terms.append(eK_hat.detach().reshape(Bsz, -1).norm(dim=1).mean())
            eK_hat_initial_norm_terms.append(eK_hat_initial.detach().reshape(Bsz, -1).norm(dim=1).mean())
            residual_contrib_norm_terms.append(residual_contrib.detach().reshape(Bsz, -1).norm(dim=1).mean())
            tf_error_norm_terms.append(tf_norm_for_h.detach())
            tf_error_ratio_terms.append(tf_ratio_for_h.detach())

            # Does the transport metric really differ from ordinary one-step MSE?
            # Because J is stopped in the prop loss, the gradient wrt e1 is
            # proportional to J^T J e1.  If this is nearly collinear with e1,
            # the prop loss is mostly duplicating one-step MSE.
            e1_mse = e1.detach().pow(2).mean().clamp_min(1e-12)
            prop_mse = prop_vec.detach().pow(2).mean()
            prop_delta_mse = (prop_vec.detach() - e1.detach()).pow(2).mean()
            prop_to_e1_mse_terms.append((prop_mse / e1_mse).detach())
            prop_delta_to_e1_mse_terms.append((prop_delta_mse / e1_mse).detach())
            prop_amplification_terms.append(((prop_mse - e1_mse) / e1_mse).detach())
            prop_vec_cos_e1_terms.append(_etm_batch_cos(prop_vec, e1))
            metric_grad_proxy = torch.bmm(J_metric.transpose(1, 2), prop_vec.detach().unsqueeze(-1)).squeeze(-1)
            prop_grad_cos_one_step_terms.append(_etm_batch_cos(metric_grad_proxy, e1))

            ehat_norm = eK_hat.detach().reshape(Bsz, -1).norm(dim=1).mean().clamp_min(1e-12)
            initial_contrib_fraction_terms.append(eK_hat_initial.detach().reshape(Bsz, -1).norm(dim=1).mean() / ehat_norm)
            residual_contrib_fraction_terms.append(residual_contrib.detach().reshape(Bsz, -1).norm(dim=1).mean() / ehat_norm)

            diag = torch.diagonal(J_fit.detach(), dim1=-2, dim2=-1)
            diag_mean_terms.append(diag.mean())
            diag_std_terms.append(diag.std(unbiased=False))
            J_fro_terms.append(J_fit.detach().reshape(J_fit.shape[0], -1).norm(dim=1).mean())
            off = J_fit.detach().clone()
            idx = torch.arange(off.shape[-1], device=off.device)
            off[:, idx, idx] = 0
            offdiag_fro_terms.append(off.reshape(off.shape[0], -1).norm(dim=1).mean())

            ww = J_metric.pow(2)
            metric_mean_terms.append(ww.mean().detach())
            metric_std_terms.append(ww.std(unbiased=False).detach())
            metric_min_terms.append(ww.min().detach())
            metric_max_terms.append(ww.max().detach())

            eye_metric = torch.eye(J_metric.shape[-1], device=J_metric.device, dtype=J_metric.dtype).unsqueeze(0)
            metric_delta = J_metric.detach() - eye_metric
            metric_delta_fro_terms.append(metric_delta.reshape(metric_delta.shape[0], -1).norm(dim=1).mean())
            metric_delta_over_I_terms.append(
                metric_delta.reshape(metric_delta.shape[0], -1).norm(dim=1).mean()
                / eye_metric.reshape(1, -1).norm(dim=1).mean().clamp_min(1e-12)
            )

    fit_loss = torch.stack(fit_terms).mean()
    prop_reduce = str(getattr(args, "etm_prop_reduce", "mean")).lower()
    if prop_mode in {"geom", "geom_excess", "geometric", "geometric_excess"} and len(guide_future_mse_terms) > 0:
        # Multi-horizon positive excess over the identity/geometric baseline.
        # If J_k ~= I for all horizons, this is exactly zero rather than a
        # rescaled one-step MSE.
        future_sum = torch.stack(guide_future_mse_terms, dim=0).sum(dim=0)
        baseline_sum = torch.stack(guide_baseline_mse_terms, dim=0).sum(dim=0)
        prop_loss = (future_sum - baseline_sum).clamp_min(0.0).mean()
    else:
        prop_loss = torch.stack(prop_terms).sum() if prop_reduce == "sum" else torch.stack(prop_terms).mean()

    with torch.no_grad():
        first_rel = rels[0]
        last_rel = rels[-1]
        e_last = current_deltas[horizons[-1] - 1] if current_transport_mode else errors[horizons[-1] - 1]
        e1_norm = e1.detach().reshape(Bsz, -1).norm(dim=1).mean()
        eK_norm = e_last.reshape(Bsz, -1).norm(dim=1).mean()
        eK_hat_norm = torch.stack(eK_hat_norm_terms).mean()
        eK_hat_initial_norm = torch.stack(eK_hat_initial_norm_terms).mean()
        residual_contrib_norm = torch.stack(residual_contrib_norm_terms).mean()
        tf_error_norm = torch.stack(tf_error_norm_terms).mean()
        tf_error_ratio = torch.stack(tf_error_ratio_terms).mean()
        r2 = torch.stack(r2_terms).mean()
        r2_initial = torch.stack(r2_initial_terms).mean()
        r2_residual = torch.stack(r2_residual_terms).mean()
        r2_delta_initial = torch.stack(r2_delta_initial_terms).mean()
        initial_contrib_fraction = torch.stack(initial_contrib_fraction_terms).mean()
        residual_contrib_fraction = torch.stack(residual_contrib_fraction_terms).mean()
        prop_to_e1_mse = torch.stack(prop_to_e1_mse_terms).mean()
        prop_delta_to_e1_mse = torch.stack(prop_delta_to_e1_mse_terms).mean()
        prop_vec_cos_e1 = torch.stack(prop_vec_cos_e1_terms).mean()
        prop_grad_cos_one_step = torch.stack(prop_grad_cos_one_step_terms).mean()
        prop_amplification = torch.stack(prop_amplification_terms).mean()
        metric_delta_fro = torch.stack(metric_delta_fro_terms).mean()
        metric_delta_over_I = torch.stack(metric_delta_over_I_terms).mean()
        a_mean = torch.stack(diag_mean_terms).mean()
        a_std = torch.stack(diag_std_terms).mean()
        J_fro = torch.stack(J_fro_terms).mean()
        offdiag_fro = torch.stack(offdiag_fro_terms).mean()
        metric_mean = torch.stack(metric_mean_terms).mean()
        metric_std = torch.stack(metric_std_terms).mean()
        metric_min = torch.stack(metric_min_terms).min()
        metric_max = torch.stack(metric_max_terms).max()
        if len(sens_target_cos_e1_terms) > 0:
            sens_target_cos_e1 = torch.stack(sens_target_cos_e1_terms).mean()
            sens_target_norm_ratio = torch.stack(sens_target_norm_ratio_terms).mean()
            sens_target_delta_to_e1_mse = torch.stack(sens_target_delta_to_e1_mse_terms).mean()
            sens_j_pred_cos_target = torch.stack(sens_j_pred_cos_target_terms).mean()
            sens_j_pred_rel_error = torch.stack(sens_j_pred_rel_error_terms).mean()
            sens_j_pred_r2 = torch.stack(sens_j_pred_r2_terms).mean()
            sens_target_cos_rollout = torch.stack(sens_target_cos_rollout_terms).mean()
            sens_target_r2_rollout = torch.stack(sens_target_r2_rollout_terms).mean()
        else:
            sens_target_cos_e1 = state.new_tensor(0.0)
            sens_target_norm_ratio = state.new_tensor(0.0)
            sens_target_delta_to_e1_mse = state.new_tensor(0.0)
            sens_j_pred_cos_target = state.new_tensor(0.0)
            sens_j_pred_rel_error = state.new_tensor(0.0)
            sens_j_pred_r2 = state.new_tensor(0.0)
            sens_target_cos_rollout = state.new_tensor(0.0)
            sens_target_r2_rollout = state.new_tensor(0.0)

    logs = {
        "ar/etm_supported": 1.0,
        "ar/etm_horizon": float(K),
        "ar/etm_requested_horizon": float(K_req),
        "ar/etm_available_horizon": float(max_available),
        "ar/etm_num_fit_steps": float(len(horizons)),
        "ar/etm_fit_step_stride": float(max(1, int(getattr(args, "etm_fit_step_stride", 1)))),
        "ar/etm_fit_all_steps": 1.0 if bool(getattr(args, "etm_fit_all_steps", False)) else 0.0,
        "ar/etm_skipped_short_horizon": 0.0,
        "ar/etm_include_tf_residuals": 1.0 if include_tf_residuals else 0.0,
        "ar/etm_residual_step_stride": float(max(1, int(getattr(args, "etm_residual_step_stride", 1)))),
        "ar/etm_version_is_B": 1.0 if version == "B" else 0.0,
        "ar/etm_fit_loss": float(fit_loss.detach().cpu()),
        "ar/etm_prop_loss": float(prop_loss.detach().cpu()),
        "ar/etm_fit_r2": float(r2.detach().cpu()),
        "ar/etm_fit_r2_initial_only": float(r2_initial.detach().cpu()),
        "ar/etm_fit_r2_residual_only": float(r2_residual.detach().cpu()),
        "ar/etm_fit_r2_delta_initial_only": float(r2_delta_initial.detach().cpu()),
        "ar/etm_initial_contrib_fraction": float(initial_contrib_fraction.detach().cpu()),
        "ar/etm_residual_contrib_fraction": float(residual_contrib_fraction.detach().cpu()),
        "ar/etm_e1_norm": float(e1_norm.detach().cpu()),
        "ar/etm_eK_norm": float(eK_norm.detach().cpu()),
        "ar/etm_eK_hat_norm": float(eK_hat_norm.detach().cpu()),
        "ar/etm_eK_hat_initial_norm": float(eK_hat_initial_norm.detach().cpu()),
        "ar/etm_residual_contrib_norm": float(residual_contrib_norm.detach().cpu()),
        "ar/etm_tf_residual_norm": float(tf_error_norm.detach().cpu()),
        "ar/etm_tf_residual_over_rollout_error": float(tf_error_ratio.detach().cpu()),
        "ar/etm_diag_mean": float(a_mean.detach().cpu()),
        "ar/etm_diag_std": float(a_std.detach().cpu()),
        "ar/etm_matrix_fro_norm": float(J_fro.detach().cpu()),
        "ar/etm_matrix_offdiag_fro_norm": float(offdiag_fro.detach().cpu()),
        "ar/etm_metric_mean": float(metric_mean.detach().cpu()),
        "ar/etm_metric_std": float(metric_std.detach().cpu()),
        "ar/etm_metric_min": float(metric_min.detach().cpu()),
        "ar/etm_metric_max": float(metric_max.detach().cpu()),
        "ar/etm_metric_delta_fro_norm": float(metric_delta_fro.detach().cpu()),
        "ar/etm_metric_delta_over_identity": float(metric_delta_over_I.detach().cpu()),
        "ar/etm_prop_to_e1_mse": float(prop_to_e1_mse.detach().cpu()),
        "ar/etm_prop_delta_to_e1_mse": float(prop_delta_to_e1_mse.detach().cpu()),
        "ar/etm_prop_vec_cos_e1": float(prop_vec_cos_e1.detach().cpu()),
        "ar/etm_prop_grad_cos_one_step": float(prop_grad_cos_one_step.detach().cpu()),
        "ar/etm_prop_amplification": float(prop_amplification.detach().cpu()),
        "ar/etm_prop_reduce_is_sum": 1.0 if str(getattr(args, "etm_prop_reduce", "mean")).lower() == "sum" else 0.0,
        "ar/etm_prop_mode_is_delta": 1.0 if str(getattr(args, "etm_prop_mode", "full")).lower() == "delta" else 0.0,
        "ar/etm_prop_mode_is_excess": 1.0 if str(getattr(args, "etm_prop_mode", "full")).lower() == "excess" else 0.0,
        "ar/etm_prop_mode_is_accum": 1.0 if str(getattr(args, "etm_prop_mode", "full")).lower() in {"accum", "accumulate", "accumulated"} else 0.0,
        "ar/etm_prop_mode_is_source": 1.0 if str(getattr(args, "etm_prop_mode", "full")).lower() in {"source", "source_only"} else 0.0,
        "ar/etm_prop_mode_is_current": 1.0 if str(getattr(args, "etm_prop_mode", "full")).lower() in {"current", "current_transport", "clean_current", "current_source"} else 0.0,
        "ar/etm_prop_mode_is_guide": 1.0 if str(getattr(args, "etm_prop_mode", "full")).lower() in {"guide", "cfg", "cfg_excess", "future_guidance"} else 0.0,
        "ar/etm_prop_mode_is_geom_excess": 1.0 if str(getattr(args, "etm_prop_mode", "full")).lower() in {"geom", "geom_excess", "geometric", "geometric_excess"} else 0.0,
        "ar/etm_prop_mode_is_signed_guide": 1.0 if str(getattr(args, "etm_prop_mode", "full")).lower() in {"signed_guide", "score_correction"} else 0.0,
        "ar/etm_current_transport_target": 1.0 if current_transport_mode else 0.0,
        "ar/etm_fit_target_is_sensitivity": 1.0 if fit_target_is_sensitivity else 0.0,
        "ar/etm_diag_rollout_sensitivity": 1.0 if diagnose_rollout_sensitivity else 0.0,
        "ar/etm_stage_is_warmup": 1.0 if _etm_stage_name(args, getattr(args, "_current_epoch_for_logs", 0)) == "warmup" else 0.0,
        "ar/etm_stage_is_j_only": 1.0 if _etm_stage_name(args, getattr(args, "_current_epoch_for_logs", 0)) == "j_only" else 0.0,
        "ar/etm_stage_is_joint": 1.0 if _etm_stage_name(args, getattr(args, "_current_epoch_for_logs", 0)) == "joint" else 0.0,
        "ar/etm_sens_target_cos_e1": float(sens_target_cos_e1.detach().cpu()),
        "ar/etm_sens_target_norm_ratio": float(sens_target_norm_ratio.detach().cpu()),
        "ar/etm_sens_target_delta_to_e1_mse": float(sens_target_delta_to_e1_mse.detach().cpu()),
        "ar/etm_sens_j_pred_cos_target": float(sens_j_pred_cos_target.detach().cpu()),
        "ar/etm_sens_j_pred_rel_error": float(sens_j_pred_rel_error.detach().cpu()),
        "ar/etm_sens_j_pred_r2_target": float(sens_j_pred_r2.detach().cpu()),
        "ar/etm_sens_target_cos_rollout": float(sens_target_cos_rollout.detach().cpu()),
        "ar/etm_sens_target_r2_rollout": float(sens_target_r2_rollout.detach().cpu()),
        "ar/etm_first_rel_l2": float(first_rel.detach().cpu()),
        "ar/etm_last_rel_l2": float(last_rel.detach().cpu()),
    }
    return fit_loss, prop_loss, logs

def _etm_zero_logs() -> Dict[str, float]:
    return {
        "ar/etm_supported": 0.0,
        "ar/etm_attn_metric": 0.0,
        "ar/etm_path_metric": 0.0,
        "ar/etm_path_num_sources": 0.0,
        "ar/etm_path_source_stride": 0.0,
        "ar/etm_metric_log_r2_long_loss": 0.0,
        "ar/etm_metric_rank_loss": 0.0,
        "ar/etm_metric_rank_acc": 0.0,
        "ar/etm_metric_score_mean": 0.0,
        "ar/etm_metric_target_long_mse": 0.0,
        "ar/etm_metric_score_to_e1_mse": 0.0,
        "ar/etm_metric_attn_entropy": 0.0,
        "ar/etm_metric_attn_max": 0.0,
        "ar/etm_metric_score_scale": 0.0,
        "ar/etm_horizon": 0.0,
        "ar/etm_requested_horizon": 0.0,
        "ar/etm_available_horizon": 0.0,
        "ar/etm_num_fit_steps": 0.0,
        "ar/etm_fit_step_stride": 0.0,
        "ar/etm_fit_all_steps": 0.0,
        "ar/etm_skipped_short_horizon": 0.0,
        "ar/etm_include_tf_residuals": 0.0,
        "ar/etm_residual_step_stride": 0.0,
        "ar/etm_version_is_B": 0.0,
        "ar/etm_fit_loss": 0.0,
        "ar/etm_prop_loss": 0.0,
        "ar/etm_fit_r2": 0.0,
        "ar/etm_fit_r2_initial_only": 0.0,
        "ar/etm_fit_r2_residual_only": 0.0,
        "ar/etm_fit_r2_delta_initial_only": 0.0,
        "ar/etm_initial_contrib_fraction": 0.0,
        "ar/etm_residual_contrib_fraction": 0.0,
        "ar/etm_e1_norm": 0.0,
        "ar/etm_eK_norm": 0.0,
        "ar/etm_eK_hat_norm": 0.0,
        "ar/etm_eK_hat_initial_norm": 0.0,
        "ar/etm_residual_contrib_norm": 0.0,
        "ar/etm_tf_residual_norm": 0.0,
        "ar/etm_tf_residual_over_rollout_error": 0.0,
        "ar/etm_diag_mean": 0.0,
        "ar/etm_diag_std": 0.0,
        "ar/etm_matrix_fro_norm": 0.0,
        "ar/etm_matrix_offdiag_fro_norm": 0.0,
        "ar/etm_metric_mean": 0.0,
        "ar/etm_metric_std": 0.0,
        "ar/etm_metric_min": 0.0,
        "ar/etm_metric_max": 0.0,
        "ar/etm_metric_delta_fro_norm": 0.0,
        "ar/etm_metric_delta_over_identity": 0.0,
        "ar/etm_prop_to_e1_mse": 0.0,
        "ar/etm_prop_delta_to_e1_mse": 0.0,
        "ar/etm_prop_vec_cos_e1": 0.0,
        "ar/etm_prop_grad_cos_one_step": 0.0,
        "ar/etm_prop_amplification": 0.0,
        "ar/etm_prop_reduce_is_sum": 0.0,
        "ar/etm_prop_mode_is_delta": 0.0,
        "ar/etm_prop_mode_is_excess": 0.0,
        "ar/etm_prop_mode_is_accum": 0.0,
        "ar/etm_prop_mode_is_source": 0.0,
        "ar/etm_prop_mode_is_current": 0.0,
        "ar/etm_prop_mode_is_guide": 0.0,
        "ar/etm_prop_mode_is_geom_excess": 0.0,
        "ar/etm_prop_mode_is_signed_guide": 0.0,
        "ar/etm_current_transport_target": 0.0,
        "ar/etm_fit_target_is_sensitivity": 0.0,
        "ar/etm_diag_rollout_sensitivity": 0.0,
        "ar/etm_stage_is_warmup": 0.0,
        "ar/etm_stage_is_j_only": 0.0,
        "ar/etm_stage_is_joint": 0.0,
        "ar/etm_sens_target_cos_e1": 0.0,
        "ar/etm_sens_target_norm_ratio": 0.0,
        "ar/etm_sens_target_delta_to_e1_mse": 0.0,
        "ar/etm_sens_j_pred_cos_target": 0.0,
        "ar/etm_sens_j_pred_rel_error": 0.0,
        "ar/etm_sens_j_pred_r2_target": 0.0,
        "ar/etm_sens_target_cos_rollout": 0.0,
        "ar/etm_sens_target_r2_rollout": 0.0,
        "ar/etm_first_rel_l2": 0.0,
        "ar/etm_last_rel_l2": 0.0,
    }


def _normalize_pseudo(pseudo: torch.Tensor, ref_error: torch.Tensor, enabled: bool, max_ratio: float) -> torch.Tensor:
    if not enabled:
        return pseudo
    dims = tuple(range(1, pseudo.dim()))
    p_rms = pseudo.detach().pow(2).mean(dim=dims, keepdim=True).sqrt().clamp_min(1e-8)
    e_rms = ref_error.detach().pow(2).mean(dim=dims, keepdim=True).sqrt().clamp_min(1e-8)
    scale = (e_rms / p_rms).clamp(max=float(max_ratio))
    return pseudo * scale


def _normalize_diag_gate(gate: torch.Tensor, gate_min: float = 0.25, gate_max: float = 4.0) -> torch.Tensor:
    """Normalize an element-wise diagonal transport gate to mean one per sample.

    The compiled adjoint lives in the prediction/state space.  A scalar transport
    uses cI and therefore preserves the error direction.  A diagonal transport
    uses diag(d) and can reweight dimensions/ROIs/channels before local gradient
    injection.  To keep the global strength controlled by comp_rho and
    comp_lambda, we normalize d to have per-sample mean one and clip extreme
    values.
    """
    g = gate.detach().abs()
    dims = tuple(range(1, g.dim()))
    mean = g.mean(dim=dims, keepdim=True).clamp_min(1e-8)
    g = g / mean
    return g.clamp(float(gate_min), float(gate_max))


def _make_diag_gate(
    residual_gate: torch.Tensor,
    error_gate: torch.Tensor,
    source: str,
    identity_mix: float,
    gate_min: float,
    gate_max: float,
) -> torch.Tensor:
    """Create a bounded diagonal gate for block-level adjoint transport.

    source='residual' uses rollout state changes |z_{t+1}-z_t|.
    source='error' uses prediction errors |z_t-z_t^*|.
    source='mixed' averages both.
    identity_mix interpolates the normalized gate with all-ones to reduce noise.
    """
    src = str(source).lower()
    if src in ("residual", "state", "delta"):
        gate = residual_gate
    elif src in ("error", "err"):
        gate = error_gate
    elif src in ("mixed", "mix"):
        gate = 0.5 * (residual_gate + error_gate)
    else:
        raise ValueError(f"Unknown --comp_diag_source={source}")
    gate = _normalize_diag_gate(gate, gate_min=gate_min, gate_max=gate_max)
    m = float(identity_mix)
    if m > 0.0:
        m = max(0.0, min(1.0, m))
        gate = m * torch.ones_like(gate) + (1.0 - m) * gate
    return gate.detach()


def compute_compiled_backward_graph_loss(
    raw,
    state: torch.Tensor,
    stim: torch.Tensor | None,
    t0: int,
    args,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Loop-SSA compiled surrogate backward-graph loss for clean AR models.

    This is a Python-level backward graph planner, not a low-level compiler.  It
    implements the first testable version of the idea:

      1. no-grad K-step rollout to obtain a state tape and horizon errors;
      2. fixed temporal frontiers / blocks;
      3. block-level compiled reverse scan with cheap strength-reduced
         transport, including scalar or diagonal gates; and
      4. local pseudo-gradient injection through one-step graphs.

    The exact long BPTT graph is never materialized.  Autograd only sees the
    local one-step calls in the injection stage.
    """
    W = int(args.window_size)
    Bsz, T, _shape = get_batch_time_shape(state)
    K_req = int(getattr(args, "comp_horizon", 0))
    if K_req <= 0:
        K_req = int(getattr(args, "koopman_gramian_horizon", 8))
    K = max(1, min(K_req, T - int(t0)))
    block = max(1, int(getattr(args, "comp_block_size", 4)))
    rho = float(getattr(args, "comp_rho", 0.95))
    future_weight = float(getattr(args, "comp_future_weight", 1.0))
    transport = str(getattr(args, "comp_transport", "residual_scalar"))

    histories = []
    stim_windows = []
    preds = []
    targets = []
    errors = []
    residual_scales = []
    residual_diag_gates = []
    error_diag_gates = []

    # ------------------------------------------------------------------
    # 1) No-grad rollout: values only, no long autograd graph.
    # ------------------------------------------------------------------
    with torch.no_grad():
        hist = time_window(state, int(t0) - W, int(t0)).detach()
        for s in range(K):
            cur = int(t0) + s
            sw = time_window(stim, cur - W, cur) if stim is not None else None
            histories.append(hist.detach())
            stim_windows.append(sw.detach() if sw is not None else None)
            hist_next, pred = _ar_step_history_with_stim(raw, hist, sw)
            target = time_point(state, cur, keep_time=False).detach()
            err = (pred - target).detach()
            preds.append(pred.detach())
            targets.append(target)
            errors.append(err)
            # Cheap loop-variant gate for residual/invariant strength reduction.
            last = hist[:, -1].detach()
            r_num = (pred.detach() - last).reshape(Bsz, -1).norm(dim=1)
            r_den = last.reshape(Bsz, -1).norm(dim=1).clamp_min(1e-8)
            alpha = (r_num / r_den).clamp(0.0, float(getattr(args, "comp_residual_alpha_max", 2.0)))
            residual_scales.append(alpha)

            gate_min = float(getattr(args, "comp_diag_min", 0.25))
            gate_max = float(getattr(args, "comp_diag_max", 4.0))
            residual_diag_gates.append(_normalize_diag_gate(pred.detach() - last, gate_min, gate_max))
            error_diag_gates.append(_normalize_diag_gate(err, gate_min, gate_max))
            hist = hist_next.detach()

    if not errors:
        z = state.new_tensor(0.0)
        return z, {"ar/comp_loss": 0.0, "ar/comp_horizon": 0.0, "ar/comp_blocks": 0.0}

    # ------------------------------------------------------------------
    # 2) Block-frontier construction and block error aggregation.
    # ------------------------------------------------------------------
    num_blocks = (K + block - 1) // block
    block_errors = []
    block_scales = []
    block_diag_gates = []
    diag_source = str(getattr(args, "comp_diag_source", "mixed"))
    diag_mix = float(getattr(args, "comp_diag_identity_mix", 0.5))
    diag_min = float(getattr(args, "comp_diag_min", 0.25))
    diag_max = float(getattr(args, "comp_diag_max", 4.0))
    for b in range(num_blocks):
        a = b * block
        c = min(K, (b + 1) * block)
        # Later errors within the block get slightly larger weight.  This is a
        # gradient-phi merge node at the block frontier.
        acc = torch.zeros_like(errors[0])
        denom = 0.0
        for idx, s in enumerate(range(a, c)):
            w = rho ** float(c - 1 - s)
            acc = acc + w * errors[s]
            denom += w
        block_errors.append((acc / max(denom, 1e-8)).detach())
        block_scales.append(torch.stack(residual_scales[a:c]).mean(dim=0).detach())
        res_gate = torch.stack(residual_diag_gates[a:c]).mean(dim=0).detach()
        err_gate = torch.stack(error_diag_gates[a:c]).mean(dim=0).detach()
        block_diag_gates.append(_make_diag_gate(res_gate, err_gate, diag_source, diag_mix, diag_min, diag_max))

    # ------------------------------------------------------------------
    # 3) Compiled reverse scan over block frontiers.
    #    This is where exact block Jacobian chains are strength-reduced.
    # ------------------------------------------------------------------
    block_adj = [None for _ in range(num_blocks + 1)]
    block_adj[num_blocks] = torch.zeros_like(errors[0])
    for b in reversed(range(num_blocks)):
        lam_next = block_adj[b + 1]
        if transport == "identity":
            transported = lam_next
            sigma = 1.0
        elif transport == "scalar_decay":
            sigma = rho ** float(min(block, K - b * block))
            transported = sigma * lam_next
        elif transport == "residual_scalar":
            # Invariant identity path plus cheap loop-variant residual gate:
            # J ~= I + alpha I.  We damp it by rho to avoid uncontrolled growth.
            alpha = block_scales[b].view(Bsz, *([1] * (errors[0].dim() - 1)))
            sigma = rho ** float(min(block, K - b * block))
            transported = sigma * (1.0 + alpha) * lam_next
        elif transport in ("diagonal", "diag"):
            # Element-wise diagonal strength reduction:
            # Phi_b ~= rho^h diag(d_b).  The gate d_b is normalized to mean
            # one per sample, clipped, and optionally mixed with identity.
            sigma = rho ** float(min(block, K - b * block))
            transported = sigma * block_diag_gates[b] * lam_next
        elif transport in ("residual_diagonal", "diag_residual"):
            # Residual scalar magnitude + element-wise direction gate.
            alpha = block_scales[b].view(Bsz, *([1] * (errors[0].dim() - 1)))
            sigma = rho ** float(min(block, K - b * block))
            transported = sigma * (1.0 + alpha) * block_diag_gates[b] * lam_next
        elif transport == "delete":
            sigma = 0.0
            transported = torch.zeros_like(lam_next)
        else:
            raise ValueError(f"Unknown --comp_transport={transport}")
        block_adj[b] = (block_errors[b] + future_weight * transported).detach()

    # ------------------------------------------------------------------
    # 4) Prolongation from block adjoints to fine transition outputs.
    # ------------------------------------------------------------------
    pseudo = []
    for s in range(K):
        b = min(num_blocks - 1, s // block)
        c = min(K, (b + 1) * block)
        local = errors[s]
        # Intra-block future merge: keep cheap local horizon structure.
        for q in range(s + 1, c):
            local = local + (rho ** float(q - s)) * errors[q]
        # Cross-block compiled adjoint from next frontier.
        cross = block_adj[b + 1]
        dist_to_frontier = max(1, c - s)
        pe = local + future_weight * (rho ** float(dist_to_frontier)) * cross
        pe = _normalize_pseudo(
            pe.detach(),
            errors[s],
            enabled=bool(getattr(args, "comp_normalize_adjoints", True)),
            max_ratio=float(getattr(args, "comp_max_adj_ratio", 5.0)),
        )
        pseudo.append(pe.detach())

    # Optional path sparsification: keep only largest pseudo-adjoint samples.
    keep_frac = float(getattr(args, "comp_sparsify_keep_frac", 1.0))
    if keep_frac < 1.0:
        norms = torch.stack([p.reshape(Bsz, -1).norm(dim=1) for p in pseudo], dim=1)  # [B,K]
        keep_n = max(1, int(round(K * keep_frac)))
        thresh = torch.topk(norms, k=keep_n, dim=1).values[:, -1].view(Bsz, 1)
        for s in range(K):
            mask = (norms[:, s:s+1] >= thresh).to(pseudo[s].dtype)
            pseudo[s] = pseudo[s] * mask.view(Bsz, *([1] * (pseudo[s].dim() - 1)))

    # ------------------------------------------------------------------
    # 5) Local gradient injection.  Autograd sees only K independent one-step
    #    graphs from detached rollout states to the shared parameters theta.
    # ------------------------------------------------------------------
    loss_terms = []
    max_inject = int(getattr(args, "comp_max_injection_steps", 0))
    stride = max(1, int(getattr(args, "comp_injection_stride", 1)))
    step_ids = list(range(0, K, stride))
    if max_inject > 0 and len(step_ids) > max_inject:
        # Deterministic uniform subsampling to control compute.
        idx = torch.linspace(0, len(step_ids) - 1, max_inject, device=state.device).round().long().tolist()
        step_ids = [step_ids[i] for i in idx]
    for s in step_ids:
        y = _ar_forward_pred(raw, stim_windows[s], histories[s].detach())
        loss_terms.append(_pseudo_dot_loss(y, pseudo[s], zero_value=bool(getattr(args, "comp_zero_value_loss", True))))

    if not loss_terms:
        loss_comp = state.new_tensor(0.0)
    else:
        loss_comp = torch.stack(loss_terms).mean()

    with torch.no_grad():
        mean_err = torch.stack([e.reshape(Bsz, -1).norm(dim=1).mean() for e in errors]).mean()
        mean_pseudo = torch.stack([p.reshape(Bsz, -1).norm(dim=1).mean() for p in pseudo]).mean()
        # The actual pseudo loss may be zero-valued by construction.  This proxy
        # records the raw dot-product scale for debugging only.
        proxy_vals = []
        for s_id in step_ids:
            proxy_vals.append(_pseudo_dot_proxy(preds[s_id], pseudo[s_id]))
        comp_proxy = torch.stack(proxy_vals).mean() if proxy_vals else state.new_tensor(0.0)
        one_rel = relative_l2(preds[0], targets[0])
        last_rel = relative_l2(preds[-1], targets[-1])
        diag_stack = torch.stack(block_diag_gates) if block_diag_gates else torch.ones_like(errors[0]).unsqueeze(0)
        diag_mean = diag_stack.mean()
        diag_std = diag_stack.std(unbiased=False)
        diag_min_val = diag_stack.min()
        diag_max_val = diag_stack.max()

    logs = {
        "ar/comp_loss": float(loss_comp.detach().cpu()),
        "ar/comp_proxy_dot": float(comp_proxy.detach().cpu()),
        "ar/comp_horizon": float(K),
        "ar/comp_block_size": float(block),
        "ar/comp_blocks": float(num_blocks),
        "ar/comp_injection_steps": float(len(step_ids)),
        "ar/comp_mean_error_norm": float(mean_err.detach().cpu()),
        "ar/comp_mean_pseudo_norm": float(mean_pseudo.detach().cpu()),
        "ar/comp_first_rel_l2": float(one_rel.detach().cpu()),
        "ar/comp_last_rel_l2": float(last_rel.detach().cpu()),
        "ar/comp_lambda": float(getattr(args, "comp_lambda", 0.0)),
        "ar/comp_diag_mean": float(diag_mean.detach().cpu()),
        "ar/comp_diag_std": float(diag_std.detach().cpu()),
        "ar/comp_diag_min": float(diag_min_val.detach().cpu()),
        "ar/comp_diag_max": float(diag_max_val.detach().cpu()),
    }
    return loss_comp, logs



def _effective_frontier_lambda(args, epoch: int | float = 0) -> float:
    """Schedule live-gradient-frontier strength."""
    base = float(getattr(args, "frontier_lambda", 0.0))
    if base == 0.0:
        return 0.0
    start = int(getattr(args, "frontier_start_epoch", 0))
    ramp = int(getattr(args, "frontier_ramp_epochs", 0))
    ep = float(epoch or 0)
    if ep < start:
        return 0.0
    if ramp > 0:
        scale = min(1.0, max(0.0, (ep - start + 1.0) / float(ramp)))
        return base * scale
    return base


def _frontier_prune_tensor(x: torch.Tensor, tau: float, min_keep_frac: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prune a live adjoint frontier with a per-sample normalized threshold.

    tau is in [0, 1].  For each sample, element scores are |x_i| / max_i |x_i|.
    Elements with score >= tau survive.  min_keep_frac optionally keeps at least
    a small top fraction per sample, preventing a completely dead frontier when
    all magnitudes are tiny/noisy.
    """
    B = x.shape[0]
    flat = x.detach().abs().reshape(B, -1)
    maxv = flat.amax(dim=1, keepdim=True).clamp_min(1e-12)
    score = flat / maxv
    tau = max(0.0, min(1.0, float(tau)))
    mask = score >= tau

    min_keep_frac = max(0.0, min(1.0, float(min_keep_frac)))
    if min_keep_frac > 0.0 and flat.shape[1] > 0:
        k = max(1, int(round(min_keep_frac * flat.shape[1])))
        kth = torch.topk(score, k=k, dim=1).values[:, -1:].detach()
        mask = mask | (score >= kth)

    mask = mask.to(dtype=x.dtype).view_as(x)
    kept = mask.reshape(B, -1).mean(dim=1).mean().detach()
    score_mean = score.mean(dim=1).mean().detach()
    return x * mask, mask, torch.stack([kept, score_mean])


def _step_history_with_stim_for_vjp(raw, history: torch.Tensor, stim_window: torch.Tensor | None) -> torch.Tensor:
    """Differentiable one-step history transition used for exact short VJP."""
    try:
        return raw.step_history(history, stim_window)
    except TypeError:
        if hasattr(raw, "step_history"):
            return raw.step_history(history)
    pred = _ar_forward_pred(raw, stim_window, history)
    return torch.cat([history[:, 1:], pred.unsqueeze(1)], dim=1)


def _history_vjp_step_with_stim(raw, history: torch.Tensor, stim_window: torch.Tensor | None, adj_next_history: torch.Tensor) -> torch.Tensor:
    """Compute J_history^T @ adj_next_history for one AR history step.

    This is the exact one-step backward edge of the rollout graph, but the graph
    is immediately discarded.  Therefore the full K-step BPTT graph is never
    materialized.
    """
    h = history.detach().requires_grad_(True)
    sw = stim_window.detach() if stim_window is not None else None
    with torch.enable_grad():
        next_h = _step_history_with_stim_for_vjp(raw, h, sw)
        grad_h = torch.autograd.grad(
            outputs=next_h,
            inputs=h,
            grad_outputs=adj_next_history.detach(),
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
    return grad_h.detach()


def compute_live_frontier_backward_graph_loss(
    raw,
    state: torch.Tensor,
    stim: torch.Tensor | None,
    t0: int,
    args,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Sensitivity-pruned live-gradient-frontier loss for clean AR models.

    This replaces the old scalar/diagonal compiled pseudo-adjoint with an
    explicit progressive pruning of the BPTT backward graph.  Starting from the
    last rollout frontier, we scan backward.  At each frontier we:

      1. add the current prediction error to the live adjoint on the newest
         history slot;
      2. normalize each sample's adjoint coordinates by the maximum magnitude;
      3. prune directions whose normalized score is below --frontier_tau;
      4. propagate only the surviving live frontier through the exact one-step
         VJP to the previous history.

    The resulting sparse adjoints are injected through local one-step graphs,
    exactly as a custom gradient on each transition output.  One-step supervised
    loss remains outside this term and should be kept enabled.
    """
    W = int(args.window_size)
    Bsz, T, _shape = get_batch_time_shape(state)
    K_req = int(getattr(args, "frontier_horizon", 0))
    if K_req <= 0:
        K_req = int(getattr(args, "comp_horizon", 0))
    if K_req <= 0:
        K_req = int(getattr(args, "koopman_gramian_horizon", 8))
    K = max(1, min(K_req, T - int(t0)))

    histories: list[torch.Tensor] = []
    stim_windows: list[torch.Tensor | None] = []
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    errors: list[torch.Tensor] = []

    with torch.no_grad():
        hist = time_window(state, int(t0) - W, int(t0)).detach()
        for s in range(K):
            cur = int(t0) + s
            sw = time_window(stim, cur - W, cur) if stim is not None else None
            histories.append(hist.detach())
            stim_windows.append(sw.detach() if sw is not None else None)
            hist_next, pred = _ar_step_history_with_stim(raw, hist, sw)
            target = time_point(state, cur, keep_time=False).detach()
            err = (pred - target).detach()
            preds.append(pred.detach())
            targets.append(target)
            errors.append(err)
            hist = hist_next.detach()

    if not errors:
        z = state.new_tensor(0.0)
        return z, {"ar/frontier_loss": 0.0, "ar/frontier_horizon": 0.0}

    tau = float(getattr(args, "frontier_tau", 0.05))
    min_keep = float(getattr(args, "frontier_min_keep_frac", 0.0))
    include_current = bool(getattr(args, "frontier_include_current_error", True))
    normalize = bool(getattr(args, "frontier_normalize_adjoints", True))
    max_ratio = float(getattr(args, "frontier_max_adj_ratio", 5.0))

    # Reverse live-frontier scan.  live_next is an adjoint wrt the history state
    # after the current transition, i.e. H_{s+1}.
    live_next = torch.zeros_like(histories[-1])
    pseudo = [torch.zeros_like(errors[0]) for _ in range(K)]
    keep_fracs = []
    score_means = []
    live_norms = []
    raw_live_norms = []

    for s in reversed(range(K)):
        lam_next = live_next.clone()
        if include_current:
            # Current loss attaches to the newest slot of H_{s+1}.
            lam_next[:, -1] = lam_next[:, -1] + errors[s]

        raw_norm = lam_next.reshape(Bsz, -1).norm(dim=1).mean().detach()
        lam_pruned, _mask, stats = _frontier_prune_tensor(lam_next, tau=tau, min_keep_frac=min_keep)
        keep_fracs.append(stats[0])
        score_means.append(stats[1])
        live_norms.append(lam_pruned.reshape(Bsz, -1).norm(dim=1).mean().detach())
        raw_live_norms.append(raw_norm)

        py = lam_pruned[:, -1].detach()
        py = _normalize_pseudo(py, errors[s], enabled=normalize, max_ratio=max_ratio)
        pseudo[s] = py.detach()

        # Propagate only the surviving frontier to the previous history.
        # This is the exact local BPTT edge, followed by pruning at the next
        # frontier in the following loop iteration.
        if s > 0:
            live_next = _history_vjp_step_with_stim(raw, histories[s], stim_windows[s], lam_pruned)

    # Local gradient injection.  Skip s=0 by default to avoid duplicating the
    # ordinary one-step loss at the same start point.  Future steps still provide
    # long-horizon sparse credit through predicted rollout states.
    stride = max(1, int(getattr(args, "frontier_injection_stride", getattr(args, "comp_injection_stride", 1))))
    max_inject = int(getattr(args, "frontier_max_injection_steps", getattr(args, "comp_max_injection_steps", 0)))
    skip_first = bool(getattr(args, "frontier_skip_first_injection", True))
    start_s = 1 if skip_first and K > 1 else 0
    step_ids = list(range(start_s, K, stride))
    if max_inject > 0 and len(step_ids) > max_inject:
        idx = torch.linspace(0, len(step_ids) - 1, max_inject, device=state.device).round().long().tolist()
        step_ids = [step_ids[i] for i in idx]

    loss_terms = []
    for s in step_ids:
        y = _ar_forward_pred(raw, stim_windows[s], histories[s].detach())
        loss_terms.append(_pseudo_dot_loss(y, pseudo[s], zero_value=bool(getattr(args, "frontier_zero_value_loss", True))))

    loss_frontier = torch.stack(loss_terms).mean() if loss_terms else state.new_tensor(0.0)

    with torch.no_grad():
        mean_err = torch.stack([e.reshape(Bsz, -1).norm(dim=1).mean() for e in errors]).mean()
        mean_pseudo = torch.stack([p.reshape(Bsz, -1).norm(dim=1).mean() for p in pseudo]).mean()
        proxy_vals = [_pseudo_dot_proxy(preds[s], pseudo[s]) for s in step_ids]
        proxy = torch.stack(proxy_vals).mean() if proxy_vals else state.new_tensor(0.0)
        one_rel = relative_l2(preds[0], targets[0])
        last_rel = relative_l2(preds[-1], targets[-1])
        keep_mean = torch.stack(keep_fracs).mean() if keep_fracs else state.new_tensor(0.0)
        keep_min = torch.stack(keep_fracs).min() if keep_fracs else state.new_tensor(0.0)
        keep_max = torch.stack(keep_fracs).max() if keep_fracs else state.new_tensor(0.0)
        score_mean = torch.stack(score_means).mean() if score_means else state.new_tensor(0.0)
        live_norm = torch.stack(live_norms).mean() if live_norms else state.new_tensor(0.0)
        raw_live_norm = torch.stack(raw_live_norms).mean() if raw_live_norms else state.new_tensor(0.0)
        survival = (live_norm / raw_live_norm.clamp_min(1e-12)).detach()

    logs = {
        "ar/frontier_loss": float(loss_frontier.detach().cpu()),
        "ar/frontier_proxy_dot": float(proxy.detach().cpu()),
        "ar/frontier_horizon": float(K),
        "ar/frontier_tau": float(max(0.0, min(1.0, tau))),
        "ar/frontier_injection_steps": float(len(step_ids)),
        "ar/frontier_keep_frac": float(keep_mean.detach().cpu()),
        "ar/frontier_keep_frac_min": float(keep_min.detach().cpu()),
        "ar/frontier_keep_frac_max": float(keep_max.detach().cpu()),
        "ar/frontier_score_mean": float(score_mean.detach().cpu()),
        "ar/frontier_survival_norm_ratio": float(survival.detach().cpu()),
        "ar/frontier_mean_error_norm": float(mean_err.detach().cpu()),
        "ar/frontier_mean_pseudo_norm": float(mean_pseudo.detach().cpu()),
        "ar/frontier_first_rel_l2": float(one_rel.detach().cpu()),
        "ar/frontier_last_rel_l2": float(last_rel.detach().cpu()),
        "ar/frontier_lambda": float(getattr(args, "frontier_lambda", 0.0)),
    }
    return loss_frontier, logs



def _frontier_error_from_prediction(pred: torch.Tensor, target: torch.Tensor, args) -> torch.Tensor:
    """Detached coordinate gradient attached to the newest history slot.

    For the first strict implementation we use the MSE/coordinate error direction,
    e = pred - target.  This is the natural coordinate-wise frontier signal and
    keeps the pruning rule interpretable.  It is intentionally independent of the
    scalar one-step loss choice (which may be rel-L2 in the rest of the trainer).
    """
    # Optional scale keeps the magnitude close to the gradient of mean squared
    # error.  It does not affect tau-based pruning because tau normalizes per
    # sample, but it affects the absolute injected gradient scale before
    # --frontier_lambda.
    mode = str(getattr(args, "frontier_error_mode", "raw_error")).lower()
    err = (pred - target).detach()
    if mode in ("mse_grad", "mean_mse_grad"):
        return err * (2.0 / max(1, err[0].numel()))
    return err


def strict_live_frontier_backward(
    model,
    state: torch.Tensor,
    stim: torch.Tensor | None,
    args,
    epoch: int = 0,
) -> Dict[str, float]:
    """Strict coordinate-wise live-gradient-frontier BPTT backward pass.

    This function does NOT return a scalar loss.  It directly accumulates extra
    gradients into model parameters, after the ordinary one-step loss has already
    called backward().

    It implements the pruned reverse recursion

        lambda_{s+1} = P_{s+1}(e_{s+1} + lambda_{s+1}^{future})
        grad_theta += frontier_lambda * B_s^T lambda_{s+1}
        lambda_s^{future} = J_s^T lambda_{s+1}

    with exact local VJPs.  The K-step autograd graph is never materialized:
    the rollout is first computed under no_grad to save state values, and each
    local edge H_s -> H_{s+1} is rematerialized, used for one VJP/backward, and
    then discarded.
    """
    raw = unwrap_model(model)
    if (not raw.training) or (not bool(getattr(args, "frontier_graph_loss", False))):
        return _strict_frontier_zero_logs()

    lam_eff = _effective_frontier_lambda(args, epoch=epoch)
    if lam_eff == 0.0:
        logs = _strict_frontier_zero_logs()
        logs["ar/frontier_lambda"] = float(getattr(args, "frontier_lambda", 0.0))
        logs["ar/frontier_effective_lambda"] = 0.0
        return logs

    if bool(getattr(raw, "is_clock_latent_ae", False)) or bool(getattr(raw, "is_state_sequence_ae", False)) or bool(getattr(raw, "is_path_generator", False)):
        return _strict_frontier_zero_logs()

    Bsz, T, _ = get_batch_time_shape(state)
    W = int(args.window_size)
    if T <= W:
        return _strict_frontier_zero_logs()
    if stim is None:
        stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))

    starts = _candidate_starts(T, W, int(getattr(args, "ar_train_stride", 1)))
    if not starts:
        return _strict_frontier_zero_logs()
    n_starts = int(getattr(args, "frontier_train_starts_per_sequence", -1))
    if n_starts <= 0:
        n_starts = int(getattr(args, "ar_train_starts_per_sequence", 1))
    n_starts = max(1, min(n_starts, len(starts)))
    if bool(getattr(args, "frontier_random_starts", getattr(args, "ar_train_random_starts", True))):
        chosen = random.sample(starts, n_starts)
    else:
        offset = max(int(epoch) - 1, 0) % max(1, len(starts))
        chosen = [starts[(offset + i) % len(starts)] for i in range(n_starts)]

    K_req = int(getattr(args, "frontier_horizon", 0))
    if K_req <= 0:
        K_req = int(getattr(args, "comp_horizon", 0))
    if K_req <= 0:
        K_req = int(getattr(args, "koopman_gramian_horizon", 8))
    K_req = max(1, K_req)

    tau = float(getattr(args, "frontier_tau", 0.05))
    min_keep = float(getattr(args, "frontier_min_keep_frac", 0.0))
    include_current = bool(getattr(args, "frontier_include_current_error", True))
    # This global multiplier is separate from lambda warmup.  It is useful if
    # the direct VJP gradient is too large even after lambda scheduling.
    grad_scale = float(getattr(args, "frontier_grad_scale", 1.0))
    inject_scale = float(lam_eff) * float(grad_scale) / float(max(1, len(chosen)))

    keep_fracs: list[torch.Tensor] = []
    keep_mins: list[torch.Tensor] = []
    keep_maxs: list[torch.Tensor] = []
    score_means: list[torch.Tensor] = []
    raw_norms: list[torch.Tensor] = []
    pruned_norms: list[torch.Tensor] = []
    err_norms: list[torch.Tensor] = []
    first_rels: list[torch.Tensor] = []
    last_rels: list[torch.Tensor] = []
    horizons: list[int] = []
    steps_done = 0

    for t0 in chosen:
        K = max(1, min(K_req, T - int(t0)))
        if K <= 0:
            continue
        horizons.append(K)
        histories: list[torch.Tensor] = []
        stim_windows: list[torch.Tensor | None] = []
        preds: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        errors: list[torch.Tensor] = []

        # 1) Store values only.  No long autograd graph is kept.
        with torch.no_grad():
            hist = time_window(state, int(t0) - W, int(t0)).detach()
            for s in range(K):
                cur = int(t0) + s
                sw = time_window(stim, cur - W, cur) if stim is not None else None
                histories.append(hist.detach())
                stim_windows.append(sw.detach() if sw is not None else None)
                hist_next, pred = _ar_step_history_with_stim(raw, hist, sw)
                target = time_point(state, cur, keep_time=False).detach()
                preds.append(pred.detach())
                targets.append(target)
                errors.append(_frontier_error_from_prediction(pred.detach(), target, args))
                hist = hist_next.detach()

        if not histories:
            continue
        with torch.no_grad():
            first_rels.append(relative_l2(preds[0], targets[0]).detach())
            last_rels.append(relative_l2(preds[-1], targets[-1]).detach())

        # lambda_future is an adjoint wrt H_{s+1} propagated from later steps.
        lambda_future = torch.zeros_like(histories[-1])

        for s in reversed(range(K)):
            # 2) Current live frontier on H_{s+1}: future branch + current error.
            lam_next = lambda_future.detach().clone()
            if include_current:
                lam_next[:, -1] = lam_next[:, -1] + errors[s]

            raw_norm = lam_next.reshape(Bsz, -1).norm(dim=1).mean().detach()
            lam_pruned, mask, stats = _frontier_prune_tensor(lam_next, tau=tau, min_keep_frac=min_keep)
            pruned_norm = lam_pruned.reshape(Bsz, -1).norm(dim=1).mean().detach()

            keep_fracs.append(mask.detach().reshape(Bsz, -1).float().mean(dim=1).mean())
            keep_mins.append(mask.detach().reshape(Bsz, -1).float().mean(dim=1).min())
            keep_maxs.append(mask.detach().reshape(Bsz, -1).float().mean(dim=1).max())
            score_means.append(stats[1].detach())
            raw_norms.append(raw_norm)
            pruned_norms.append(pruned_norm)
            err_norms.append(errors[s].reshape(Bsz, -1).norm(dim=1).mean().detach())

            # 3) Rematerialize exactly one local edge H_s -> H_{s+1}.
            h = histories[s].detach().requires_grad_(True)
            sw = stim_windows[s].detach() if stim_windows[s] is not None else None
            with torch.enable_grad():
                next_h = _step_history_with_stim_for_vjp(raw, h, sw)

                # 3a) Propagate the SAME pruned frontier backward to H_s.
                # This is exact local VJP: J_s^T lambda_{s+1}.
                grad_h = torch.autograd.grad(
                    outputs=next_h,
                    inputs=h,
                    grad_outputs=lam_pruned.detach(),
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=False,
                )[0]

                # 3b) Accumulate parameter gradient B_s^T lambda_{s+1} using
                # the SAME lam_pruned.  Only a global scalar lambda is applied.
                if inject_scale != 0.0:
                    torch.autograd.backward(
                        tensors=next_h,
                        grad_tensors=(inject_scale * lam_pruned).detach(),
                        retain_graph=False,
                    )

            # 4) Discard this local graph; only the numeric frontier survives.
            lambda_future = grad_h.detach()
            steps_done += 1
            del h, next_h, grad_h, lam_next, lam_pruned, mask

    if steps_done == 0:
        return _strict_frontier_zero_logs()

    def mean_or_zero(xs):
        return torch.stack(xs).mean() if xs else state.new_tensor(0.0)
    def min_or_zero(xs):
        return torch.stack(xs).min() if xs else state.new_tensor(0.0)
    def max_or_zero(xs):
        return torch.stack(xs).max() if xs else state.new_tensor(0.0)

    raw_mean = mean_or_zero(raw_norms)
    pruned_mean = mean_or_zero(pruned_norms)
    survival = pruned_mean / raw_mean.clamp_min(1e-12)
    horizon_mean = float(sum(horizons) / max(1, len(horizons)))

    return {
        "ar/frontier_loss": 0.0,  # no scalar loss is constructed in the strict implementation
        "ar/frontier_horizon": float(horizon_mean),
        "ar/frontier_tau": float(max(0.0, min(1.0, tau))),
        "ar/frontier_injection_steps": float(steps_done),
        "ar/frontier_keep_frac": float(mean_or_zero(keep_fracs).detach().cpu()),
        "ar/frontier_keep_frac_min": float(min_or_zero(keep_mins).detach().cpu()),
        "ar/frontier_keep_frac_max": float(max_or_zero(keep_maxs).detach().cpu()),
        "ar/frontier_score_mean": float(mean_or_zero(score_means).detach().cpu()),
        "ar/frontier_survival_norm_ratio": float(survival.detach().cpu()),
        "ar/frontier_mean_error_norm": float(mean_or_zero(err_norms).detach().cpu()),
        "ar/frontier_mean_pseudo_norm": float(pruned_mean.detach().cpu()),
        "ar/frontier_proxy_dot": 0.0,
        "ar/frontier_first_rel_l2": float(mean_or_zero(first_rels).detach().cpu()),
        "ar/frontier_last_rel_l2": float(mean_or_zero(last_rels).detach().cpu()),
        "ar/frontier_lambda": float(getattr(args, "frontier_lambda", 0.0)),
        "ar/frontier_effective_lambda": float(lam_eff),
        "ar/frontier_weighted_loss": 0.0,
        "ar/frontier_proxy_weighted": 0.0,
        "ar/frontier_pct_of_one_step": 0.0,
        "ar/frontier_proxy_pct_of_one_step": 0.0,
        "ar/frontier_pct_of_total": 0.0,
        "ar/frontier_strict_manual_bptt": 1.0,
    }


def _strict_frontier_zero_logs() -> Dict[str, float]:
    return {
        "ar/frontier_loss": 0.0,
        "ar/frontier_horizon": 0.0,
        "ar/frontier_tau": 0.0,
        "ar/frontier_injection_steps": 0.0,
        "ar/frontier_keep_frac": 0.0,
        "ar/frontier_keep_frac_min": 0.0,
        "ar/frontier_keep_frac_max": 0.0,
        "ar/frontier_score_mean": 0.0,
        "ar/frontier_survival_norm_ratio": 0.0,
        "ar/frontier_mean_error_norm": 0.0,
        "ar/frontier_mean_pseudo_norm": 0.0,
        "ar/frontier_proxy_dot": 0.0,
        "ar/frontier_first_rel_l2": 0.0,
        "ar/frontier_last_rel_l2": 0.0,
        "ar/frontier_lambda": 0.0,
        "ar/frontier_effective_lambda": 0.0,
        "ar/frontier_weighted_loss": 0.0,
        "ar/frontier_proxy_weighted": 0.0,
        "ar/frontier_pct_of_one_step": 0.0,
        "ar/frontier_proxy_pct_of_one_step": 0.0,
        "ar/frontier_pct_of_total": 0.0,
        "ar/frontier_strict_manual_bptt": 0.0,
    }

def _checkpointed_windowed_step(raw, horizon_index, total_horizon, stim_window, history):
    if hasattr(raw, "set_resgrad_context"):
        raw.set_resgrad_context(horizon_index=horizon_index, total_horizon=total_horizon)
    return raw(stim_window, history, return_aux=False)


def _global_wiener_step_loss(
    raw,
    horizon: int,
    prediction: torch.Tensor,
    target: torch.Tensor,
    step_loss: torch.Tensor,
    loss_at_target,
    per_sample_loss_at_target=None,
) -> torch.Tensor:
    """Attach a global-horizon probe and backward-only horizon coefficient.

    ``loss_at_target`` must rebuild the same scalar step objective with a
    replacement target.  When subgroup pooling is enabled,
    ``per_sample_loss_at_target`` rebuilds the same objective as a ``[B]``
    vector; the controller forms disjoint group sums for calibration while the
    returned real loss remains the unchanged complete-batch scalar.  The
    antithetic difference estimates the component of the *actual optimized
    score* induced by a symmetric innovation draw.  For squared error it
    reduces exactly to the usual linear noise VJP.
    """

    controller = getattr(raw, "global_horizon_wiener", None)
    if controller is None:
        return step_loss
    innovations = raw.global_wiener_observe_and_sample(
        prediction, target, int(horizon)
    )
    if innovations is not None:
        if innovations.ndim != target.ndim + 1:
            raise RuntimeError(
                "global Wiener innovations must have shape [draw,B,...]; got "
                f"{tuple(innovations.shape)} for target {tuple(target.shape)}"
            )
        superbatch_groups = int(
            getattr(controller, "superbatch_groups", 1)
        )
        if superbatch_groups > 1:
            if per_sample_loss_at_target is None:
                raise RuntimeError(
                    "global Wiener superbatch calibration requires a per-sample "
                    "version of the optimized horizon loss"
                )
            total_probe = per_sample_loss_at_target(target)
            if total_probe.ndim != 1 or int(total_probe.shape[0]) != int(
                target.shape[0]
            ):
                raise RuntimeError(
                    "per-sample global Wiener total probe must be [B], got "
                    f"{tuple(total_probe.shape)} for target {tuple(target.shape)}"
                )
            noise_scores = []
            for innovation in innovations.unbind(0):
                plus = per_sample_loss_at_target(target + innovation)
                minus = per_sample_loss_at_target(target - innovation)
                noise_scores.append(0.5 * (plus - minus))
        else:
            total_probe = step_loss
            noise_scores = []
            for innovation in innovations.unbind(0):
                plus = loss_at_target(target + innovation)
                minus = loss_at_target(target - innovation)
                noise_scores.append(0.5 * (plus - minus))
        raw.global_wiener_add_probe_terms(
            int(horizon), total_probe, noise_scores
        )
    return raw.global_wiener_weight_loss(step_loss, int(horizon))


def _windowed_step_maybe_checkpointed(raw, stim_window, history, horizon_index, total_horizon, use_checkpoint):
    """Run one closed-loop windowed-AR rollout step, optionally under
    gradient checkpointing.

    Exact-gradient analog of _recurrent_step_maybe_checkpointed for
    windowed-history standard-AR models (unet_field, resnet_vector,
    transformer_vector, ...) trained through compute_full_bptt_rollout_loss.
    Unlike the recurrent-state case there is no persistent state tuple to
    flatten/unflatten -- (stim_window, history) are the only tensors that
    need to be recomputed on backward. Diagnostic aux fields (resgrad_gate,
    branch_residual_ratio) are not available for checkpointed steps, same
    as the recurrent path.
    """
    if not use_checkpoint:
        if hasattr(raw, "set_resgrad_context"):
            raw.set_resgrad_context(horizon_index=horizon_index, total_horizon=total_horizon)
        return raw(stim_window, history, return_aux=False)
    return torch.utils.checkpoint.checkpoint(
        _checkpointed_windowed_step,
        raw, horizon_index, total_horizon, stim_window, history,
        use_reentrant=False,
    )


def _backward_scale_value(value, scale: float):
    """Keep a forward value fixed while multiplying its incoming gradient.

    ``scale=0`` returns a genuinely detached value, so a sampled ARTBP cut also
    releases the preceding autograd graph.  Tuples/lists are handled
    recursively because recurrent Mamba states contain several tensors per
    layer.
    """
    if torch.is_tensor(value):
        if scale <= 0.0:
            return value.detach()
        if scale == 1.0:
            return value
        return value.detach() + float(scale) * (value - value.detach())
    if isinstance(value, tuple):
        return tuple(_backward_scale_value(v, scale) for v in value)
    if isinstance(value, list):
        return [_backward_scale_value(v, scale) for v in value]
    raise TypeError(f"unsupported ARTBP carry type: {type(value)!r}")


def _sample_artbp_edge_scale(
    expected_segment_length: int,
    device: torch.device,
    *,
    draw: Optional[float] = None,
) -> float:
    """Sample the ARTBP multiplier for one temporal edge.

    With ``c=1/L``, the segment length is geometric with mean ``L``.  A cut has
    multiplier zero; a surviving edge has multiplier ``1/(1-c)``.  Hence the
    expected multiplier is one, which is the local compensation used by ARTBP.
    ``draw`` is exposed only for deterministic unit tests.
    """
    length = int(expected_segment_length)
    if length <= 1:
        return 1.0
    cut_probability = 1.0 / float(length)
    if draw is None:
        draw = float(torch.rand((), device=device).item())
    if float(draw) < cut_probability:
        return 0.0
    return 1.0 / (1.0 - cut_probability)


def compute_full_bptt_rollout_loss(
    raw,
    state: torch.Tensor,
    stim: Optional[torch.Tensor],
    start_t: int,
    args,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Exact closed-loop rollout loss with optional error-cloud regularization.

    Base BPTT term:
        e_k^AR = x_k^AR - x_k,
        L_BPTT = mean_k ||e_k^AR||.

    Optional error-cloud terms use the fixed decomposition at each rollout step:
        e_k^AR = p_k + b_k,
        p_k = stopgrad(x_k^AR - x_k^TF),
        b_k = x_k^TF - x_k.

    Here x_k^AR is produced by the closed-loop BPTT rollout, while x_k^TF is a
    one-step teacher-forced prediction from the ground-truth history at the same
    target time.  The extra cloud losses only backpropagate through b_k; the
    ordinary BPTT loss still backpropagates through the closed-loop AR rollout.

    The intent is not to remove local errors.  It is to discourage coherent
    reinforcement of accumulated rollout error and to make local error directions
    within a BPTT window closer to a bounded, zero-mean, isotropic cloud.
    """
    B, T, _ = get_batch_time_shape(state)
    W = int(args.window_size)
    K_req = int(getattr(args, "bptt_horizon", 0))
    K = min(max(K_req, 0), T - int(start_t))

    zero_ref = time_point(state, max(min(int(start_t), T - 1), 0), keep_time=True).new_tensor(0.0)
    zero_logs = {
        "ar/bptt_loss": 0.0,
        "ar/bptt_base_loss": 0.0,
        "ar/bptt_horizon": 0.0,
        "ar/bptt_first_rel_l2": 0.0,
        "ar/bptt_last_rel_l2": 0.0,
        "ar/bptt_detach_period": 0.0,
        "ar/artbp_expected_segment_length": 0.0,
        "ar/artbp_cut_fraction": 0.0,
        "ar/bptt_reinforce_loss": 0.0,
        "ar/bptt_reinforce_lambda": 0.0,
        "ar/bptt_reinforce_weighted_loss": 0.0,
        "ar/bptt_reinforce_cos_mean": 0.0,
        "ar/bptt_reinforce_cos_pos_frac": 0.0,
        "ar/bptt_reinforce_pos_cos_mean": 0.0,
        "ar/bptt_cloud_orth_loss": 0.0,
        "ar/bptt_cloud_orth_lambda": 0.0,
        "ar/bptt_cloud_orth_weighted_loss": 0.0,
        "ar/bptt_cloud_mean_loss": 0.0,
        "ar/bptt_cloud_mean_lambda": 0.0,
        "ar/bptt_cloud_mean_weighted_loss": 0.0,
        "ar/bptt_cloud_radius_loss": 0.0,
        "ar/bptt_cloud_radius_lambda": 0.0,
        "ar/bptt_cloud_radius_weighted_loss": 0.0,
        "ar/bptt_cloud_radius_ratio": 0.0,
        "ar/bptt_error_p_norm": 0.0,
        "ar/bptt_error_b_norm": 0.0,
        "ar/bptt_error_delta_norm": 0.0,
        "ar/bptt_error_E_p": 0.0,
        "ar/bptt_error_E_b": 0.0,
        "ar/bptt_error_E_cross": 0.0,
        "ar/bptt_error_E_cross_pos": 0.0,
        "ar/bptt_error_decomp_residual_frac": 0.0,
        "ar/resgrad_routing": 0.0,
        "ar/resgrad_gate_mean": 1.0,
        "ar/resgrad_gate_min": 1.0,
        "ar/resgrad_gate_max": 1.0,
        "ar/resgrad_branch_residual_ratio": 0.0,
    }
    if K <= 0 or int(start_t) - W < 0:
        return zero_ref, zero_logs

    if stim is None:
        stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))

    history = time_window(state, int(start_t) - W, int(start_t))
    loss_name = str(getattr(args, "bptt_loss_type", getattr(args, "ar_loss", "mse")))
    losses = []
    first_rel = None
    last_rel = None

    # Optional error-cloud settings.  These lambdas are relative to the BPTT
    # scalar returned here; the outer training loop still multiplies the total
    # by --bptt_lambda.
    rein_lam = float(getattr(args, "bptt_reinforce_lambda", 0.0))
    orth_lam = float(getattr(args, "bptt_cloud_orth_lambda", 0.0))
    mean_lam = float(getattr(args, "bptt_cloud_mean_lambda", 0.0))
    radius_lam = float(getattr(args, "bptt_cloud_radius_lambda", 0.0))
    cloud_enabled = (rein_lam != 0.0) or (orth_lam != 0.0) or (mean_lam != 0.0) or (radius_lam != 0.0) or bool(getattr(args, "bptt_error_cloud_eval", False))
    eps = float(getattr(args, "bptt_error_cloud_eps", 1e-8))
    min_norm = float(getattr(args, "bptt_error_cloud_min_norm", 1e-8))
    radius_target = float(getattr(args, "bptt_cloud_radius_target", 1.0))

    b_list = []
    p_list = []
    delta_list = []
    reinforce_losses = []
    reinforce_cos = []
    reinforce_pos_frac = []
    reinforce_pos_mean = []
    resgrad_gates = []
    resgrad_routing_vals = []
    resgrad_branch_ratios = []
    dual_wiener_total_terms = []
    dual_wiener_noise_terms = []

    use_checkpoint = bool(getattr(args, "bptt_grad_checkpoint", False)) and bool(raw.training)
    detach_period = max(0, int(getattr(args, "bptt_detach_period", 0)))
    artbp_length = max(
        0, int(getattr(args, "artbp_expected_segment_length", 0))
    )
    if detach_period > 0 and artbp_length > 1:
        raise ValueError("TBPTT and ARTBP cannot be enabled in the same run")
    artbp_edge_scales = []
    route_policy = str(getattr(raw, "resgrad_policy", "all")).lower()
    needs_step_route_aux = route_policy in (
        "ratio", "act_ratio", "dynamic", "dynamic_ratio",
        "branch_ratio", "norm_ratio",
    )

    for k in range(K):
        target_t = int(start_t) + k
        stim_window = time_window(stim, target_t - W, target_t)
        has_resgrad_context = hasattr(raw, "set_resgrad_context")
        if use_checkpoint:
            # Exact-gradient alternative to ResGrad routing: recompute this
            # step's activations during backward instead of retaining them,
            # at ~O(1)-in-K memory per step / ~2x forward compute. Mirrors
            # _recurrent_step_maybe_checkpointed for the recurrent-state
            # path. Diagnostic aux fields are unavailable under checkpoint.
            pred = _windowed_step_maybe_checkpointed(raw, stim_window, history, k, K, True)
        elif has_resgrad_context:
            raw.set_resgrad_context(horizon_index=k, total_horizon=K)
            # Dynamic-ratio routing genuinely needs per-block activation
            # diagnostics to resolve its gate. Other policies do not: asking
            # U-Net for return_aux=True at every horizon copied dozens of
            # block-level scalars from GPU to CPU without affecting the loss.
            if needs_step_route_aux:
                pred, step_aux = raw(stim_window, history, return_aux=True)
                if "resgrad_gate" in step_aux:
                    resgrad_gates.append(float(step_aux["resgrad_gate"].detach().cpu()))
                if "resgrad_routing" in step_aux:
                    resgrad_routing_vals.append(float(step_aux["resgrad_routing"].detach().cpu()))
                if "resgrad_branch_residual_ratio" in step_aux:
                    resgrad_branch_ratios.append(float(step_aux["resgrad_branch_residual_ratio"].detach().cpu()))
            else:
                pred = raw(stim_window, history, return_aux=False)
                resgrad_routing_vals.append(
                    1.0 if bool(getattr(raw, "resgrad_routing", False)) else 0.0
                )
                if getattr(raw, "dual_wiener", None) is None and hasattr(
                    raw, "_resgrad_gate_for_step"
                ):
                    resgrad_gates.append(float(raw._resgrad_gate_for_step(k)))
        else:
            pred = raw(stim_window, history, return_aux=False)  # [B,1,...]
        x_ar = pred[:, 0]
        target = time_point(state, target_t, keep_time=True)
        x_gt = target[:, 0]

        # Use the same fully open, matched quadratic probes as the recurrent
        # path.  The controller only returns terms on scheduled probe batches;
        # the optimized rollout loss remains the user-selected objective.
        if raw.training and hasattr(raw, "dual_wiener_probe_terms"):
            dw_total, dw_noise = raw.dual_wiener_probe_terms(x_ar, x_gt, k)
            if dw_total is not None and dw_noise is not None:
                dual_wiener_total_terms.append(dw_total)
                dual_wiener_noise_terms.append(dw_noise)

        if loss_name == "rel_l2":
            step_loss = relative_l2(x_ar, x_gt)
            loss_at_target = lambda replacement: relative_l2(x_ar, replacement)
            per_sample_loss_at_target = lambda replacement: _global_wiener_per_sample_state_loss(
                x_ar, replacement, "rel_l2"
            )
        else:
            step_loss = elementwise_state_loss(pred, target, loss=loss_name)
            loss_at_target = lambda replacement: elementwise_state_loss(
                pred, replacement.unsqueeze(1), loss=loss_name
            )
            per_sample_loss_at_target = lambda replacement: _global_wiener_per_sample_state_loss(
                pred, replacement.unsqueeze(1), loss_name
            )
        step_loss = _global_wiener_step_loss(
            raw,
            k,
            x_ar,
            x_gt,
            step_loss,
            loss_at_target,
            per_sample_loss_at_target,
        )
        losses.append(step_loss)

        with torch.no_grad():
            rel = relative_l2(x_ar, x_gt)
            if first_rel is None:
                first_rel = rel
            last_rel = rel

        if cloud_enabled:
            # Teacher-forced local branch at the same time index.
            hist_tf = time_window(state, target_t - W, target_t)
            x_tf = _ar_forward_pred(raw, stim_window, hist_tf)
            b_t = x_tf - x_gt                                  # trainable local error
            p_t = (x_ar.detach() - x_tf.detach()).detach()     # detached transported previous error
            delta_t = (x_ar.detach() - x_gt).detach()          # detached total AR error

            b_list.append(b_t)
            p_list.append(p_t)
            delta_list.append(delta_t)

            # Reinforcement term: only positive cosine alignment between the
            # existing transported error p_t and the new local TF error b_t.
            rein_loss, rein_logs = _positive_alignment_loss_from_p_b(
                p_t, b_t, eps=eps, min_norm=min_norm
            )
            reinforce_losses.append(rein_loss)
            reinforce_cos.append(torch.as_tensor(rein_logs["cos_mean"], device=state.device, dtype=state.dtype))
            reinforce_pos_frac.append(torch.as_tensor(rein_logs["cos_pos_frac"], device=state.device, dtype=state.dtype))
            reinforce_pos_mean.append(torch.as_tensor(rein_logs["pos_cos_mean"], device=state.device, dtype=state.dtype))

        # Closed-loop feedback.  The default keeps the exact graph.  A positive
        # detach period retains the identical forward rollout and all K losses,
        # but cuts temporal credit at fixed segment boundaries (TBPTT-S).
        history = torch.cat([history[:, 1:], pred], dim=1)
        if detach_period > 0 and (k + 1) < K and ((k + 1) % detach_period) == 0:
            history = history.detach()
        elif artbp_length > 1 and (k + 1) < K and raw.training:
            edge_scale = _sample_artbp_edge_scale(artbp_length, history.device)
            history = _backward_scale_value(history, edge_scale)
            artbp_edge_scales.append(edge_scale)

    if hasattr(raw, "dual_wiener_set_probe_losses"):
        dw_total = (
            torch.stack(dual_wiener_total_terms).mean()
            if dual_wiener_total_terms else None
        )
        dw_noise = (
            torch.stack(dual_wiener_noise_terms).mean()
            if dual_wiener_noise_terms else None
        )
        raw.dual_wiener_set_probe_losses(dw_total, dw_noise)

    base_loss = torch.stack(losses).mean()

    if cloud_enabled and b_list:
        b_flat = torch.stack([b.reshape(B, -1) for b in b_list], dim=1)       # [B,K,D]
        p_flat = torch.stack([p.reshape(B, -1) for p in p_list], dim=1)       # [B,K,D]
        d_flat = torch.stack([d.reshape(B, -1) for d in delta_list], dim=1)   # [B,K,D]
        bn = b_flat.norm(dim=-1)
        pn = p_flat.norm(dim=-1)
        dn = d_flat.norm(dim=-1).clamp_min(eps)
        valid = bn > min_norm

        u = b_flat / bn.clamp_min(eps).unsqueeze(-1)
        G = torch.bmm(u, u.transpose(1, 2))  # [B,K,K]
        K_eff = int(b_flat.shape[1])
        eye = torch.eye(K_eff, device=state.device, dtype=torch.bool).unsqueeze(0)
        valid_pair = valid.unsqueeze(1) & valid.unsqueeze(2) & (~eye)
        if K_eff > 1 and valid_pair.any():
            orth_loss = G.pow(2).masked_select(valid_pair).mean()
        else:
            orth_loss = base_loss.new_tensor(0.0)

        valid_f = valid.to(u.dtype)
        denom = valid_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_vec = (u * valid_f.unsqueeze(-1)).sum(dim=1) / denom
        mean_loss = mean_vec.pow(2).sum(dim=-1).mean()

        # Optional radius hinge.  It only activates if the current local-error
        # energy exceeds radius_target times a detached reference.  If no custom
        # target is supplied, the reference is the current detached energy, so
        # this term is inactive by default; it is mainly for future ablations.
        b_energy = b_flat.pow(2).sum(dim=-1)
        ref_energy = b_energy.detach().mean().clamp_min(eps)
        radius_ratio = b_energy.mean() / ref_energy
        radius_loss = torch.relu(radius_ratio / max(radius_target, eps) - 1.0).pow(2)

        reinforce_loss = torch.stack(reinforce_losses).mean() if reinforce_losses else base_loss.new_tensor(0.0)
        cloud_extra = rein_lam * reinforce_loss + orth_lam * orth_loss + mean_lam * mean_loss + radius_lam * radius_loss

        with torch.no_grad():
            dot_pb = (p_flat * b_flat.detach()).sum(dim=-1)
            p2 = p_flat.pow(2).sum(dim=-1)
            b2 = b_flat.detach().pow(2).sum(dim=-1)
            d2 = d_flat.pow(2).sum(dim=-1).clamp_min(eps)
            rec = (p_flat + b_flat.detach() - d_flat).norm(dim=-1)
            decomp_residual_frac = (rec / dn).mean()
            E_p = (p2 / d2).mean()
            E_b = (b2 / d2).mean()
            E_cross = (2.0 * dot_pb / d2).mean()
            E_cross_pos = (torch.relu(2.0 * dot_pb) / d2).mean()
            b_norm_mean = bn.mean()
            p_norm_mean = pn.mean()
            d_norm_mean = dn.mean()
    else:
        reinforce_loss = base_loss.new_tensor(0.0)
        orth_loss = base_loss.new_tensor(0.0)
        mean_loss = base_loss.new_tensor(0.0)
        radius_loss = base_loss.new_tensor(0.0)
        radius_ratio = base_loss.new_tensor(0.0)
        cloud_extra = base_loss.new_tensor(0.0)
        E_p = base_loss.new_tensor(0.0)
        E_b = base_loss.new_tensor(0.0)
        E_cross = base_loss.new_tensor(0.0)
        E_cross_pos = base_loss.new_tensor(0.0)
        decomp_residual_frac = base_loss.new_tensor(0.0)
        b_norm_mean = base_loss.new_tensor(0.0)
        p_norm_mean = base_loss.new_tensor(0.0)
        d_norm_mean = base_loss.new_tensor(0.0)

    total_loss = base_loss + cloud_extra
    logs = dict(zero_logs)
    logs.update({
        "ar/bptt_loss": float(total_loss.detach().cpu()),
        "ar/bptt_base_loss": float(base_loss.detach().cpu()),
        "ar/bptt_horizon": float(K),
        "ar/bptt_first_rel_l2": float(first_rel.detach().cpu()) if first_rel is not None else 0.0,
        "ar/bptt_last_rel_l2": float(last_rel.detach().cpu()) if last_rel is not None else 0.0,
        "ar/bptt_detach_period": float(detach_period),
        "ar/artbp_expected_segment_length": float(artbp_length),
        "ar/artbp_cut_fraction": (
            float(sum(scale == 0.0 for scale in artbp_edge_scales))
            / float(len(artbp_edge_scales))
            if artbp_edge_scales else 0.0
        ),
        "ar/bptt_reinforce_loss": float(reinforce_loss.detach().cpu()),
        "ar/bptt_reinforce_lambda": float(rein_lam),
        "ar/bptt_reinforce_weighted_loss": float((rein_lam * reinforce_loss).detach().cpu()),
        "ar/bptt_reinforce_cos_mean": float(torch.stack(reinforce_cos).mean().detach().cpu()) if reinforce_cos else 0.0,
        "ar/bptt_reinforce_cos_pos_frac": float(torch.stack(reinforce_pos_frac).mean().detach().cpu()) if reinforce_pos_frac else 0.0,
        "ar/bptt_reinforce_pos_cos_mean": float(torch.stack(reinforce_pos_mean).mean().detach().cpu()) if reinforce_pos_mean else 0.0,
        "ar/bptt_cloud_orth_loss": float(orth_loss.detach().cpu()),
        "ar/bptt_cloud_orth_lambda": float(orth_lam),
        "ar/bptt_cloud_orth_weighted_loss": float((orth_lam * orth_loss).detach().cpu()),
        "ar/bptt_cloud_mean_loss": float(mean_loss.detach().cpu()),
        "ar/bptt_cloud_mean_lambda": float(mean_lam),
        "ar/bptt_cloud_mean_weighted_loss": float((mean_lam * mean_loss).detach().cpu()),
        "ar/bptt_cloud_radius_loss": float(radius_loss.detach().cpu()),
        "ar/bptt_cloud_radius_lambda": float(radius_lam),
        "ar/bptt_cloud_radius_weighted_loss": float((radius_lam * radius_loss).detach().cpu()),
        "ar/bptt_cloud_radius_ratio": float(radius_ratio.detach().cpu()),
        "ar/bptt_error_p_norm": float(p_norm_mean.detach().cpu()),
        "ar/bptt_error_b_norm": float(b_norm_mean.detach().cpu()),
        "ar/bptt_error_delta_norm": float(d_norm_mean.detach().cpu()),
        "ar/bptt_error_E_p": float(E_p.detach().cpu()),
        "ar/bptt_error_E_b": float(E_b.detach().cpu()),
        "ar/bptt_error_E_cross": float(E_cross.detach().cpu()),
        "ar/bptt_error_E_cross_pos": float(E_cross_pos.detach().cpu()),
        "ar/bptt_error_decomp_residual_frac": float(decomp_residual_frac.detach().cpu()),
        "ar/resgrad_routing": float(sum(resgrad_routing_vals) / max(len(resgrad_routing_vals), 1)) if resgrad_routing_vals else 0.0,
        "ar/resgrad_gate_mean": float(sum(resgrad_gates) / max(len(resgrad_gates), 1)) if resgrad_gates else 1.0,
        "ar/resgrad_gate_min": float(min(resgrad_gates)) if resgrad_gates else 1.0,
        "ar/resgrad_gate_max": float(max(resgrad_gates)) if resgrad_gates else 1.0,
        "ar/resgrad_branch_residual_ratio": float(sum(resgrad_branch_ratios) / max(len(resgrad_branch_ratios), 1)) if resgrad_branch_ratios else 0.0,
    })
    # Dual-Wiener already maintains the exact route-table diagnostics. Reuse
    # them once per rollout instead of synchronizing alpha/m at every block and
    # horizon solely to populate the legacy one-number gate fields.
    if getattr(raw, "dual_wiener", None) is not None and hasattr(
        raw, "dual_wiener_diagnostics"
    ):
        dual_diag = raw.dual_wiener_diagnostics(K)
        logs["ar/resgrad_routing"] = 1.0
        logs["ar/resgrad_gate_mean"] = float(
            dual_diag.get("ar/dual_wiener_m_mean", 1.0)
        )
        logs["ar/resgrad_gate_min"] = float(
            dual_diag.get("ar/dual_wiener_m_min", 1.0)
        )
        logs["ar/resgrad_gate_max"] = float(
            dual_diag.get("ar/dual_wiener_m_max", 1.0)
        )
    if getattr(raw, "global_horizon_wiener", None) is not None and hasattr(
        raw, "global_wiener_diagnostics"
    ):
        logs.update(raw.global_wiener_diagnostics(K))
    return total_loss, logs




def _recurrent_rollout_block_loss(
    raw,
    state: torch.Tensor,
    stim: torch.Tensor,
    block_start: int,
    block_size: int,
    burn: int,
    loss_name: str,
    decay: float = 1.0,
):
    """Independent K-step closed-loop local loss for one dataset block.

    ``block_start`` is the first target index.  The first differentiable step
    consumes x_{block_start-1} and predicts x_{block_start}.  The recurrent
    hidden state is constructed from GT frames before the block and then
    detached, so different blocks do not form a long BPTT graph.
    """
    B = state.shape[0]
    K = max(1, int(block_size))

    burn_start = max(0, int(block_start) - int(burn))
    burn_end = max(burn_start, int(block_start) - 1)

    h = raw.init_state(B, state.device, state.dtype)
    if burn_end > burn_start:
        with torch.no_grad():
            for j in range(burn_start, burn_end):
                stim_j = stim[:, j] if stim is not None else None
                _, h = raw.step(h, state[:, j], stim_j, return_aux=False)
    h = h.detach()

    x_in = state[:, int(block_start) - 1]
    losses = []
    first_rel = None
    last_rel = None
    for k in range(K):
        target_t = int(block_start) + k
        stim_in = stim[:, target_t - 1] if stim is not None else None
        pred, h, _aux = raw.step(h, x_in, stim_in, return_aux=True, horizon_index=k, total_horizon=K)
        target = state[:, target_t]
        if loss_name == "rel_l2":
            step_loss = relative_l2(pred, target)
        else:
            step_loss = elementwise_state_loss(pred.unsqueeze(1), target.unsqueeze(1), loss=loss_name)
        if decay != 1.0:
            step_loss = (decay ** k) * step_loss
        losses.append(step_loss)
        with torch.no_grad():
            rel = relative_l2(pred, target)
            if first_rel is None:
                first_rel = rel
            last_rel = rel
        # Closed-loop inside this block only.  No detach inside the block.
        x_in = pred

    loss = torch.stack(losses).mean() if losses else state.new_tensor(0.0)
    return loss, first_rel, last_rel


def _cto_cut_cross_error(
    raw,
    ctx_state: torch.Tensor,
    ctx_stim: torch.Tensor,
    fut_state: torch.Tensor,
    fut_stim: torch.Tensor,
    loss_name: str,
    order_horizon: int,
):
    """Model error for one cropped order test: context crop -> future crop.

    This implements the dataset-cut idea.  The context crop is used as GT
    burn-in for the recurrent state.  Then the model is tested on the boundary
    transition and the following frames of the future crop.  This is a
    teacher-forced error test on the constructed sequence, not predicted-block
    embedding contrast and not a 32-step BPTT rollout.

    ctx_state: [B, r, D]  e.g. tail(A) or head(B)
    fut_state: [B, r, D]  e.g. head(B) or tail(A)
    """
    B = ctx_state.shape[0]
    r_ctx = int(ctx_state.shape[1])
    r_fut = int(fut_state.shape[1])
    H = max(1, min(int(order_horizon), r_fut))

    h = raw.init_state(B, ctx_state.device, ctx_state.dtype)
    # Build the recurrent state from all but the last context frame.  Detaching
    # this burn-in keeps the order test focused on the cut boundary rather than
    # creating another long graph through the context crop.
    if r_ctx > 1:
        with torch.no_grad():
            for j in range(r_ctx - 1):
                stim_j = ctx_stim[:, j] if ctx_stim is not None else None
                _, h = raw.step(h, ctx_state[:, j], stim_j, return_aux=False)
    h = h.detach()

    losses = []
    for q in range(H):
        if q == 0:
            x_in = ctx_state[:, -1]
            stim_in = ctx_stim[:, -1] if ctx_stim is not None else None
        else:
            # Teacher-forced evaluation on the constructed sequence.  This is
            # intentionally not free rollout from the previous prediction.
            x_in = fut_state[:, q - 1]
            stim_in = fut_stim[:, q - 1] if fut_stim is not None else None

        pred, h, _aux = raw.step(h, x_in, stim_in, return_aux=True)
        target = fut_state[:, q]
        if loss_name == "rel_l2":
            step_loss = relative_l2(pred, target)
        else:
            step_loss = elementwise_state_loss(pred.unsqueeze(1), target.unsqueeze(1), loss=loss_name)
        losses.append(step_loss)

    return torch.stack(losses).mean() if losses else ctx_state.new_tensor(0.0)


def compute_recurrent_state_cto_cut_loss(
    model,
    state: torch.Tensor,
    stim: torch.Tensor,
    args,
    epoch: int = 0,
):
    """Dataset-cut CTO objective for recurrent-state AR models.

    For a 2K window [A,B], this computes

        L = 0.5 * (L_K(A) + L_K(B))
            + lambda * softplus(margin + E_pos - E_neg),

    where E_pos is the model error on the cropped true order

        tail(A) -> head(B),

    and E_neg is the model error on the swapped order

        head(B) -> tail(A).

    This is the user's dataset-cut order test.  It does not construct
    predicted block embeddings and it does not concatenate A and B into a
    continuous 32-step BPTT graph.
    """
    raw = unwrap_model(model)
    B, T, _ = get_batch_time_shape(state)
    K = int(getattr(args, "cto_cut_block_size", 0))
    if K <= 0:
        K = int(getattr(args, "mamba_bptt_horizon", 8))
    K = max(1, int(K))

    r = int(getattr(args, "cto_cut_order_span", 0))
    if r <= 0:
        r = max(1, K // 2)
    r = max(1, min(int(r), K))

    order_horizon = int(getattr(args, "cto_cut_order_horizon", 0))
    if order_horizon <= 0:
        order_horizon = r
    order_horizon = max(1, min(int(order_horizon), r))

    burn = max(0, int(getattr(args, "mamba_burnin", 64)))
    loss_name = str(getattr(args, "mamba_loss_type", getattr(args, "ar_loss", "rel_l2")))
    decay = float(getattr(args, "mamba_loss_decay", 1.0))
    lam = float(getattr(args, "cto_cut_lambda", 0.0))
    margin = float(getattr(args, "cto_cut_margin", 0.1))

    if stim is None:
        stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))

    # Need A=[s:s+K-1], B=[s+K:s+2K-1], and the boundary crops.
    max_start = T - 2 * K
    if max_start < 1:
        z = state.new_tensor(0.0)
        return z, {
            "loss": 0.0,
            "ar/cto_cut_loss": 0.0,
            "ar/cto_cut_local_loss": 0.0,
            "ar/cto_cut_order_loss": 0.0,
            "ar/cto_cut_pos_error": 0.0,
            "ar/cto_cut_neg_error": 0.0,
            "ar/cto_cut_margin_gap": 0.0,
            "ar/cto_cut_lambda": float(lam),
            "ar/cto_cut_block_size": float(K),
            "ar/cto_cut_order_span": float(r),
            "ar/cto_cut_order_horizon": float(order_horizon),
            "ar/cto_cut_num_starts": 0.0,
        }

    min_start = burn if max_start >= burn else 1
    stride = max(1, int(getattr(args, "cto_cut_stride", 1)))
    starts = list(range(min_start, max_start + 1, stride))
    if not starts:
        starts = list(range(1, max_start + 1, stride))

    if raw.training:
        n_starts = int(getattr(args, "cto_cut_starts_per_sequence", -1))
        if n_starts <= 0:
            n_starts = int(getattr(args, "mamba_train_starts_per_sequence", -1))
        if n_starts <= 0:
            n_starts = int(getattr(args, "ar_train_starts_per_sequence", 1))
        n_starts = max(1, min(n_starts, len(starts)))
        use_random = bool(getattr(args, "ar_train_random_starts", True))
        if use_random:
            chosen = random.sample(starts, n_starts)
        else:
            offset = max(int(epoch) - 1, 0) % max(1, len(starts))
            chosen = [starts[(offset + i) % len(starts)] for i in range(n_starts)]
    else:
        eval_stride = int(getattr(args, "ar_eval_stride", 4))
        chosen = list(range(min_start, max_start + 1, max(1, eval_stride)))
        if not chosen:
            chosen = [min_start]

    total_losses = []
    local_losses = []
    order_losses = []
    pos_errors = []
    neg_errors = []
    first_rels = []
    last_rels = []

    for s in chosen:
        s = int(s)
        # Local training on A and B as two independent K-step dataset blocks.
        loss_A, first_A, last_A = _recurrent_rollout_block_loss(
            raw, state, stim, s, K, burn, loss_name, decay=decay
        )
        loss_B, first_B, last_B = _recurrent_rollout_block_loss(
            raw, state, stim, s + K, K, burn, loss_name, decay=decay
        )
        local_loss = 0.5 * (loss_A + loss_B)

        # Boundary crops: a=tail(A), b=head(B).
        a_start = s + K - r
        b_start = s + K
        a_state = state[:, a_start:a_start + r]
        b_state = state[:, b_start:b_start + r]
        a_stim = stim[:, a_start:a_start + r] if stim is not None else None
        b_stim = stim[:, b_start:b_start + r] if stim is not None else None

        # Correct order: tail(A) -> head(B).  Swapped order: head(B) -> tail(A).
        err_pos = _cto_cut_cross_error(
            raw, a_state, a_stim, b_state, b_stim, loss_name, order_horizon
        )
        err_neg = _cto_cut_cross_error(
            raw, b_state, b_stim, a_state, a_stim, loss_name, order_horizon
        )
        order_loss = torch.nn.functional.softplus(state.new_tensor(margin) + err_pos - err_neg)

        total = local_loss + lam * order_loss
        total_losses.append(total)
        local_losses.append(local_loss.detach())
        order_losses.append(order_loss.detach())
        pos_errors.append(err_pos.detach())
        neg_errors.append(err_neg.detach())
        if first_A is not None:
            first_rels.append(first_A.detach())
        if last_B is not None:
            last_rels.append(last_B.detach())

    loss = torch.stack(total_losses).mean() if total_losses else state.new_tensor(0.0)
    local_mean = torch.stack(local_losses).mean() if local_losses else state.new_tensor(0.0)
    order_mean = torch.stack(order_losses).mean() if order_losses else state.new_tensor(0.0)
    pos_mean = torch.stack(pos_errors).mean() if pos_errors else state.new_tensor(0.0)
    neg_mean = torch.stack(neg_errors).mean() if neg_errors else state.new_tensor(0.0)

    logs = {
        "loss": float(loss.detach().cpu()),
        "ar/cto_cut_loss": float(loss.detach().cpu()),
        "ar/cto_cut_local_loss": float(local_mean.detach().cpu()),
        "ar/cto_cut_order_loss": float(order_mean.detach().cpu()),
        "ar/cto_cut_pos_error": float(pos_mean.detach().cpu()),
        "ar/cto_cut_neg_error": float(neg_mean.detach().cpu()),
        "ar/cto_cut_margin_gap": float((neg_mean - pos_mean).detach().cpu()),
        "ar/cto_cut_lambda": float(lam),
        "ar/cto_cut_margin": float(margin),
        "ar/cto_cut_block_size": float(K),
        "ar/cto_cut_order_span": float(r),
        "ar/cto_cut_order_horizon": float(order_horizon),
        "ar/cto_cut_num_starts": float(len(chosen)),
        "ar/one_step_rel_l2": float(torch.stack(first_rels).mean().detach().cpu()) if first_rels else 0.0,
        "ar/recurrent_first_rel_l2": float(torch.stack(first_rels).mean().detach().cpu()) if first_rels else 0.0,
        "ar/recurrent_last_rel_l2": float(torch.stack(last_rels).mean().detach().cpu()) if last_rels else 0.0,
        "ar/recurrent_bptt_horizon": float(K),
        "ar/recurrent_burnin": float(burn),
        "ar/recurrent_num_starts": float(len(chosen)),
    }
    return loss, logs



def backward_recurrent_state_cto_cut_loss(
    model,
    state: torch.Tensor,
    stim: torch.Tensor,
    args,
    epoch: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Sequential-backward dataset-cut CTO objective.

    This is the memory-correct implementation of the CTO-cut objective.  It
    accumulates gradients from A-local, B-local, and order-cut losses one graph
    at a time, then returns logs.  The caller must NOT call loss.backward() on
    the returned scalar; optimizer.step() should be called after this function.

    Objective over sampled 2K windows [A,B]:

        mean_s 0.5 * L_K(A_s)
      + mean_s 0.5 * L_K(B_s)
      + lambda * mean_s softplus(margin + E_pos_s - E_neg_s)

    with A_s and B_s sampled as adjacent true dataset blocks.  The sequential
    backward schedule is:

        backward(0.5/n * L_K(A_s))
        backward(0.5/n * L_K(B_s))
        backward(lambda/n * L_order_s)

    for each sampled start s.  This avoids keeping A, B, and order graphs alive
    until one final backward call.
    """
    raw = unwrap_model(model)
    B, T, _ = get_batch_time_shape(state)
    K = int(getattr(args, "cto_cut_block_size", 0))
    if K <= 0:
        K = int(getattr(args, "mamba_bptt_horizon", 8))
    K = max(1, int(K))

    r = int(getattr(args, "cto_cut_order_span", 0))
    if r <= 0:
        r = max(1, K // 2)
    r = max(1, min(int(r), K))

    order_horizon = int(getattr(args, "cto_cut_order_horizon", 0))
    if order_horizon <= 0:
        order_horizon = r
    order_horizon = max(1, min(int(order_horizon), r))

    burn = max(0, int(getattr(args, "mamba_burnin", 64)))
    loss_name = str(getattr(args, "mamba_loss_type", getattr(args, "ar_loss", "rel_l2")))
    decay = float(getattr(args, "mamba_loss_decay", 1.0))
    lam = float(getattr(args, "cto_cut_lambda", 0.0))
    margin = float(getattr(args, "cto_cut_margin", 0.1))

    if stim is None:
        stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))

    max_start = T - 2 * K
    if max_start < 1:
        z = state.new_tensor(0.0)
        return z, {
            "loss": 0.0,
            "ar/cto_cut_loss": 0.0,
            "ar/cto_cut_local_loss": 0.0,
            "ar/cto_cut_order_loss": 0.0,
            "ar/cto_cut_pos_error": 0.0,
            "ar/cto_cut_neg_error": 0.0,
            "ar/cto_cut_margin_gap": 0.0,
            "ar/cto_cut_lambda": float(lam),
            "ar/cto_cut_margin": float(margin),
            "ar/cto_cut_block_size": float(K),
            "ar/cto_cut_order_span": float(r),
            "ar/cto_cut_order_horizon": float(order_horizon),
            "ar/cto_cut_num_starts": 0.0,
            "ar/cto_cut_sequential_backward": 1.0,
        }

    min_start = burn if max_start >= burn else 1
    stride = max(1, int(getattr(args, "cto_cut_stride", 1)))
    starts = list(range(min_start, max_start + 1, stride))
    if not starts:
        starts = list(range(1, max_start + 1, stride))

    n_starts = int(getattr(args, "cto_cut_starts_per_sequence", -1))
    if n_starts <= 0:
        n_starts = int(getattr(args, "mamba_train_starts_per_sequence", -1))
    if n_starts <= 0:
        n_starts = int(getattr(args, "ar_train_starts_per_sequence", 1))
    n_starts = max(1, min(n_starts, len(starts)))
    use_random = bool(getattr(args, "ar_train_random_starts", True))
    if use_random:
        chosen = random.sample(starts, n_starts)
    else:
        offset = max(int(epoch) - 1, 0) % max(1, len(starts))
        chosen = [starts[(offset + i) % len(starts)] for i in range(n_starts)]

    denom = float(max(len(chosen), 1))
    local_vals = []
    order_vals = []
    pos_vals = []
    neg_vals = []
    first_rels = []
    last_rels = []

    for s in chosen:
        s = int(s)

        # A local block: build only A's graph, backward it, then release it.
        loss_A, first_A, _last_A = _recurrent_rollout_block_loss(
            raw, state, stim, s, K, burn, loss_name, decay=decay
        )
        (0.5 / denom * loss_A).backward()
        local_vals.append((0.5 * loss_A.detach()).detach())
        if first_A is not None:
            first_rels.append(first_A.detach())
        del loss_A

        # B local block: independent K-step graph, backward and release.
        loss_B, _first_B, last_B = _recurrent_rollout_block_loss(
            raw, state, stim, s + K, K, burn, loss_name, decay=decay
        )
        (0.5 / denom * loss_B).backward()
        local_vals.append((0.5 * loss_B.detach()).detach())
        if last_B is not None:
            last_rels.append(last_B.detach())
        del loss_B

        # Boundary dataset-cut order test.
        a_start = s + K - r
        b_start = s + K
        a_state = state[:, a_start:a_start + r]
        b_state = state[:, b_start:b_start + r]
        a_stim = stim[:, a_start:a_start + r] if stim is not None else None
        b_stim = stim[:, b_start:b_start + r] if stim is not None else None

        err_pos = _cto_cut_cross_error(
            raw, a_state, a_stim, b_state, b_stim, loss_name, order_horizon
        )
        err_neg = _cto_cut_cross_error(
            raw, b_state, b_stim, a_state, a_stim, loss_name, order_horizon
        )
        order_loss = torch.nn.functional.softplus(state.new_tensor(margin) + err_pos - err_neg)
        if lam != 0.0:
            (lam / denom * order_loss).backward()
        order_vals.append(order_loss.detach())
        pos_vals.append(err_pos.detach())
        neg_vals.append(err_neg.detach())
        del err_pos, err_neg, order_loss

    # For reporting, reconstruct the scalar objective value from detached terms.
    local_mean = torch.stack(local_vals).sum() / denom if local_vals else state.new_tensor(0.0)
    order_mean = torch.stack(order_vals).mean() if order_vals else state.new_tensor(0.0)
    pos_mean = torch.stack(pos_vals).mean() if pos_vals else state.new_tensor(0.0)
    neg_mean = torch.stack(neg_vals).mean() if neg_vals else state.new_tensor(0.0)
    total_value = local_mean + state.new_tensor(lam) * order_mean

    logs = {
        "loss": float(total_value.detach().cpu()),
        "ar/cto_cut_loss": float(total_value.detach().cpu()),
        "ar/cto_cut_local_loss": float(local_mean.detach().cpu()),
        "ar/cto_cut_order_loss": float(order_mean.detach().cpu()),
        "ar/cto_cut_pos_error": float(pos_mean.detach().cpu()),
        "ar/cto_cut_neg_error": float(neg_mean.detach().cpu()),
        "ar/cto_cut_margin_gap": float((neg_mean - pos_mean).detach().cpu()),
        "ar/cto_cut_lambda": float(lam),
        "ar/cto_cut_margin": float(margin),
        "ar/cto_cut_block_size": float(K),
        "ar/cto_cut_order_span": float(r),
        "ar/cto_cut_order_horizon": float(order_horizon),
        "ar/cto_cut_num_starts": float(len(chosen)),
        "ar/cto_cut_sequential_backward": 1.0,
        "ar/one_step_rel_l2": float(torch.stack(first_rels).mean().detach().cpu()) if first_rels else 0.0,
        "ar/recurrent_first_rel_l2": float(torch.stack(first_rels).mean().detach().cpu()) if first_rels else 0.0,
        "ar/recurrent_last_rel_l2": float(torch.stack(last_rels).mean().detach().cpu()) if last_rels else 0.0,
        "ar/recurrent_bptt_horizon": float(K),
        "ar/recurrent_burnin": float(burn),
        "ar/recurrent_num_starts": float(len(chosen)),
    }

    # The trainer branch that calls this function must skip loss.backward().
    # Return a detached scalar only for logging / API consistency.
    return total_value.detach(), logs


def _parse_int_csv(value, default=None) -> list[int]:
    """Parse comma/space separated positive integers from an argparse value."""
    if value is None:
        return list(default or [])
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            try:
                iv = int(v)
            except Exception:
                continue
            if iv > 0:
                out.append(iv)
        return out or list(default or [])
    text = str(value).replace(";", ",").replace(" ", ",")
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            iv = int(part)
        except ValueError:
            continue
        if iv > 0:
            out.append(iv)
    return out or list(default or [])


def _bridge_control_zero_logs() -> dict:
    return {
        "ar/bridge_enabled": 0.0,
        "ar/bridge_replace_base": 0.0,
        "ar/bridge_loss": 0.0,
        "ar/bridge_weighted_loss": 0.0,
        "ar/bridge_lambda": 0.0,
        "ar/bridge_next_loss": 0.0,
        "ar/bridge_next_lambda": 0.0,
        "ar/bridge_total_loss": 0.0,
        "ar/bridge_num_starts": 0.0,
        "ar/bridge_num_valid_horizons": 0.0,
        "ar/bridge_mean_horizon": 0.0,
        "ar/bridge_pred_delta_norm": 0.0,
        "ar/bridge_target_delta_norm": 0.0,
        "ar/bridge_delta_cos": 0.0,
        "ar/bridge_delta_rel_l2": 0.0,
        "ar/bridge_next_rel_l2": 0.0,
        "ar/bridge_next_corr": 0.0,
        "ar/bridge_used_burnin": 0.0,
        "ar/bridge_skipped_short": 0.0,
    }


def _bridge_effective_lambda(args, epoch: int | float = 0) -> float:
    lam = float(getattr(args, "bridge_control_lambda", 0.0))
    start = int(getattr(args, "bridge_control_start_epoch", 1))
    ramp = int(getattr(args, "bridge_control_ramp_epochs", 0))
    ep = int(epoch or 0)
    if ep < start:
        return 0.0
    if ramp > 0:
        return lam * min(1.0, max(0.0, float(ep - start + 1) / float(ramp)))
    return lam


def _bridge_delta_loss(pred_delta: torch.Tensor, target_delta: torch.Tensor, args) -> tuple[torch.Tensor, dict]:
    """Loss between predicted local control and bridge-induced control target."""
    eps = float(getattr(args, "bridge_control_eps", 1e-8))
    loss_type = str(getattr(args, "bridge_control_loss_type", "rel_l2")).lower()
    pf = pred_delta.reshape(pred_delta.shape[0], -1)
    tf = target_delta.reshape(target_delta.shape[0], -1)
    diff = pf - tf
    pn = pf.norm(dim=1)
    tn = tf.norm(dim=1)
    rel = diff.norm(dim=1) / tn.clamp_min(eps)
    cos = (pf * tf).sum(dim=1) / (pn.clamp_min(eps) * tn.clamp_min(eps))

    if loss_type == "mse":
        loss = (pred_delta - target_delta).pow(2).mean()
    elif loss_type in {"rel_mse", "relative_mse"}:
        loss = diff.pow(2).mean() / tf.pow(2).mean().clamp_min(eps)
    elif loss_type == "cos":
        loss = (1.0 - cos).mean()
    elif loss_type in {"cos_mse", "cos_rel"}:
        norm_weight = float(getattr(args, "bridge_control_norm_weight", 0.25))
        loss = (1.0 - cos).mean() + norm_weight * rel.mean()
    else:  # rel_l2
        loss = rel.mean()

    logs = {
        "pred_delta_norm": float(pn.detach().mean().cpu()),
        "target_delta_norm": float(tn.detach().mean().cpu()),
        "delta_cos": float(cos.detach().mean().cpu()),
        "delta_rel_l2": float(rel.detach().mean().cpu()),
    }
    return loss, logs


def _bridge_control_target_from_future(
    state: torch.Tensor,
    cur_t: int,
    horizons: list[int],
    args,
) -> tuple[torch.Tensor | None, dict]:
    """Build a future-informed local control target from true future anchors.

    The target is a multi-scale bridge velocity
        sum_m w_m (x_{t+m} - x_t) / m.
    This is a training-only privileged target.  It does not enter the model
    input, so the deployed AR student remains causal.
    """
    T = int(state.shape[1])
    valid = [int(m) for m in horizons if int(m) > 0 and int(cur_t) + int(m) < T]
    if not valid:
        return None, {"num_valid": 0.0, "mean_horizon": 0.0}

    x0 = state[:, int(cur_t)]
    mode = str(getattr(args, "bridge_control_target", "multiscale_avg")).lower()
    if mode in {"longest", "endpoint", "max"}:
        valid = [max(valid)]

    # Optional horizon weighting.  Uniform is the safest default; inverse
    # reduces domination by long endpoints; linear emphasizes long drift.
    weight_mode = str(getattr(args, "bridge_control_horizon_weight", "uniform")).lower()
    terms = []
    weights = []
    for m in valid:
        delta = (state[:, int(cur_t) + m] - x0) / float(m)
        terms.append(delta)
        if weight_mode in {"inverse", "inv"}:
            weights.append(1.0 / float(m))
        elif weight_mode in {"linear", "horizon"}:
            weights.append(float(m))
        else:
            weights.append(1.0)
    w = torch.as_tensor(weights, device=state.device, dtype=state.dtype)
    w = w / w.sum().clamp_min(1e-8)
    stacked = torch.stack(terms, dim=0)  # [M,B,...]
    view = [len(valid)] + [1] * (stacked.dim() - 1)
    target = (stacked * w.view(*view)).sum(dim=0)

    scale = float(getattr(args, "bridge_control_target_scale", 1.0))
    if scale != 1.0:
        target = target * scale
    if bool(getattr(args, "bridge_control_detach_target", True)):
        target = target.detach()
    return target, {
        "num_valid": float(len(valid)),
        "mean_horizon": float(sum(valid) / max(1, len(valid))),
    }


def compute_recurrent_bridge_control_distill_loss(
    model,
    state: torch.Tensor,
    stim: torch.Tensor | None,
    args,
    epoch: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Local bridge-control distillation for recurrent-state AR models.

    This is the minimal Bridge-to-AR stage-2 objective.  A training-only
    future-informed target is constructed from true future anchors, but the
    student input is strictly causal: a GT burn-in hidden state plus current
    x_t/stim_t.  There is no long BPTT graph; gradients flow through one
    recurrent step only.

        target control:   v*_t = sum_m alpha_m (x_{t+m} - x_t) / m
        student control:  v_theta = pred_{t+1} - x_t
    """
    raw = unwrap_model(model)
    zero_logs = _bridge_control_zero_logs()
    zero_logs["ar/bridge_enabled"] = 1.0 if bool(getattr(args, "bridge_control_loss", False)) else 0.0
    zero_logs["ar/bridge_replace_base"] = 1.0 if bool(getattr(args, "bridge_control_replace_base", False)) else 0.0
    zero_logs["ar/bridge_lambda"] = float(getattr(args, "bridge_control_lambda", 0.0))
    zero_logs["ar/bridge_next_lambda"] = float(getattr(args, "bridge_control_next_lambda", 0.0))

    if not bool(getattr(args, "bridge_control_loss", False)):
        return state.new_tensor(0.0), zero_logs
    lam = _bridge_effective_lambda(args, epoch=epoch) if raw.training else float(getattr(args, "bridge_control_lambda", 0.0))
    next_lam = float(getattr(args, "bridge_control_next_lambda", 0.0))
    if lam == 0.0 and next_lam == 0.0:
        zero_logs["ar/bridge_lambda"] = float(lam)
        return state.new_tensor(0.0), zero_logs

    if stim is None:
        stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))

    B, T, _ = get_batch_time_shape(state)
    burn = max(0, int(getattr(args, "mamba_burnin", 64)))
    horizons = _parse_int_csv(getattr(args, "bridge_control_horizons", "4,8"), default=[4, 8])
    min_h = max(1, min(horizons))
    # cur_t is the observed current frame used as input; x_{cur_t+1} exists for
    # optional next-frame anchoring, and at least one bridge horizon must exist.
    max_cur = int(T) - min_h - 1
    if max_cur < 0:
        logs = dict(zero_logs)
        logs["ar/bridge_skipped_short"] = 1.0
        return state.new_tensor(0.0), logs

    min_cur = burn if max_cur >= burn else 0
    stride_cfg = int(getattr(args, "bridge_control_train_stride", -1))
    if stride_cfg <= 0:
        stride_cfg = int(getattr(args, "mamba_train_stride", 1))
    stride = max(1, stride_cfg)
    starts = list(range(min_cur, max_cur + 1, stride))
    if not starts:
        starts = list(range(0, max_cur + 1, stride))
    if not starts:
        logs = dict(zero_logs)
        logs["ar/bridge_skipped_short"] = 1.0
        return state.new_tensor(0.0), logs

    if raw.training:
        n_starts = int(getattr(args, "bridge_control_starts_per_sequence", -1))
        if n_starts <= 0:
            n_starts = int(getattr(args, "mamba_train_starts_per_sequence", -1))
        if n_starts <= 0:
            n_starts = int(getattr(args, "ar_train_starts_per_sequence", 1))
        n_starts = max(1, min(n_starts, len(starts)))
        if bool(getattr(args, "bridge_control_random_starts", getattr(args, "ar_train_random_starts", True))):
            chosen = random.sample(starts, n_starts)
        else:
            offset = max(int(epoch) - 1, 0) % max(1, len(starts))
            chosen = [starts[(offset + i) % len(starts)] for i in range(n_starts)]
    else:
        eval_stride = int(getattr(args, "ar_eval_stride", 4))
        chosen = list(range(min_cur, max_cur + 1, max(1, eval_stride))) or [min_cur]

    bridge_losses = []
    next_losses = []
    pred_delta_norms = []
    target_delta_norms = []
    delta_cos_vals = []
    delta_rel_vals = []
    next_rels = []
    next_corrs = []
    valid_counts = []
    mean_horizons = []
    used_burns = []

    next_loss_name = str(getattr(args, "bridge_control_next_loss_type", getattr(args, "mamba_loss_type", "rel_l2")))

    for cur_t in chosen:
        target_delta, target_info = _bridge_control_target_from_future(state, int(cur_t), horizons, args)
        if target_delta is None:
            continue

        burn_start = max(0, int(cur_t) - burn)
        burn_end = int(cur_t)
        used_burns.append(float(burn_end - burn_start))

        h = raw.init_state(B, state.device, state.dtype)
        if burn_end > burn_start:
            with torch.no_grad():
                for j in range(burn_start, burn_end):
                    stim_j = stim[:, j] if stim is not None else None
                    _, h = raw.step(h, state[:, j], stim_j, return_aux=False)
        h = _detach_recurrent_state(h)

        x_in = state[:, int(cur_t)]
        stim_in = stim[:, int(cur_t)] if stim is not None else None
        pred, _h_next, _aux = raw.step(h, x_in, stim_in, return_aux=True)
        pred_delta = pred - x_in

        b_loss, b_logs = _bridge_delta_loss(pred_delta, target_delta, args)
        bridge_losses.append(b_loss)
        pred_delta_norms.append(torch.as_tensor(b_logs["pred_delta_norm"], device=state.device, dtype=state.dtype))
        target_delta_norms.append(torch.as_tensor(b_logs["target_delta_norm"], device=state.device, dtype=state.dtype))
        delta_cos_vals.append(torch.as_tensor(b_logs["delta_cos"], device=state.device, dtype=state.dtype))
        delta_rel_vals.append(torch.as_tensor(b_logs["delta_rel_l2"], device=state.device, dtype=state.dtype))
        valid_counts.append(float(target_info["num_valid"]))
        mean_horizons.append(float(target_info["mean_horizon"]))

        if next_lam != 0.0 and int(cur_t) + 1 < T:
            next_target = state[:, int(cur_t) + 1]
            if next_loss_name == "rel_l2":
                n_loss = relative_l2(pred, next_target)
            else:
                n_loss = elementwise_state_loss(pred.unsqueeze(1), next_target.unsqueeze(1), loss=next_loss_name)
            next_losses.append(n_loss)
        with torch.no_grad():
            if int(cur_t) + 1 < T:
                next_target = state[:, int(cur_t) + 1]
                next_rels.append(relative_l2(pred, next_target).detach())
                next_corrs.append(corrcoef_flat(pred, next_target).detach())

    if not bridge_losses:
        logs = dict(zero_logs)
        logs["ar/bridge_skipped_short"] = 1.0
        return state.new_tensor(0.0), logs

    bridge_loss = torch.stack(bridge_losses).mean()
    next_loss = torch.stack(next_losses).mean() if next_losses else bridge_loss.new_tensor(0.0)
    total = float(lam) * bridge_loss + float(next_lam) * next_loss

    logs = dict(zero_logs)
    logs.update({
        "ar/bridge_enabled": 1.0,
        "ar/bridge_replace_base": 1.0 if bool(getattr(args, "bridge_control_replace_base", False)) else 0.0,
        "ar/bridge_loss": float(bridge_loss.detach().cpu()),
        "ar/bridge_weighted_loss": float((float(lam) * bridge_loss).detach().cpu()),
        "ar/bridge_lambda": float(lam),
        "ar/bridge_next_loss": float(next_loss.detach().cpu()),
        "ar/bridge_next_lambda": float(next_lam),
        "ar/bridge_total_loss": float(total.detach().cpu()),
        "ar/bridge_num_starts": float(len(bridge_losses)),
        "ar/bridge_num_valid_horizons": float(sum(valid_counts) / max(len(valid_counts), 1)),
        "ar/bridge_mean_horizon": float(sum(mean_horizons) / max(len(mean_horizons), 1)),
        "ar/bridge_pred_delta_norm": float(torch.stack(pred_delta_norms).mean().detach().cpu()),
        "ar/bridge_target_delta_norm": float(torch.stack(target_delta_norms).mean().detach().cpu()),
        "ar/bridge_delta_cos": float(torch.stack(delta_cos_vals).mean().detach().cpu()),
        "ar/bridge_delta_rel_l2": float(torch.stack(delta_rel_vals).mean().detach().cpu()),
        "ar/bridge_next_rel_l2": float(torch.stack(next_rels).mean().detach().cpu()) if next_rels else 0.0,
        "ar/bridge_next_corr": float(torch.stack(next_corrs).mean().detach().cpu()) if next_corrs else 0.0,
        "ar/bridge_used_burnin": float(sum(used_burns) / max(len(used_burns), 1)),
        "ar/bridge_skipped_short": 0.0,
    })
    return total, logs

def _flatten_mamba_state(h):
    """Flatten a MambaStackState (tuple of (conv_state, ssm_state) per block)
    into a flat tuple of tensors, so it can be passed through
    torch.utils.checkpoint.checkpoint, which recomputes forward activations
    on the backward pass instead of storing them -- an exact-gradient
    alternative to ResGrad's approximate-gradient routing for reducing BPTT
    memory. Paired with _unflatten_mamba_state.
    """
    flat = []
    for layer_state in h:
        flat.extend(layer_state)
    return tuple(flat)


def _unflatten_mamba_state(flat, depth):
    """Inverse of _flatten_mamba_state.

    The per-layer arity is inferred rather than hard-coded at 2, so recurrent
    backbones whose layer state is not a (conv_state, ssm_state) pair -- e.g.
    a residual GRU carrying a single hidden tensor per layer -- round-trip
    correctly through checkpointing.  For the Mamba stack this is exactly the
    previous behaviour (per == 2).
    """
    per = max(1, len(flat) // max(1, depth))
    return tuple(tuple(flat[i * per:(i + 1) * per]) for i in range(depth))


def _checkpointed_recurrent_step(raw, depth, x_in, stim_in, horizon_index, total_horizon, *h_flat):
    h = _unflatten_mamba_state(h_flat, depth)
    pred, h_next, _aux = raw.step(
        h, x_in, stim_in, return_aux=True, horizon_index=horizon_index, total_horizon=total_horizon
    )
    return (pred,) + _flatten_mamba_state(h_next)


def _recurrent_step_maybe_checkpointed(raw, h, x_in, stim_in, horizon_index, total_horizon, use_checkpoint):
    """Run one closed-loop rollout step, optionally under gradient
    checkpointing.

    Checkpointing recomputes this step's forward pass during backward
    instead of retaining its activations, giving the *exact* full-BPTT
    gradient at roughly O(1)-in-K memory (per step), at the cost of ~2x
    forward compute. This is the natural baseline for what ResGrad routing
    is being compared against: an existing, gradient-exact way to make long
    BPTT horizons memory-feasible. Diagnostic aux fields (resgrad_gate,
    dt stats, etc.) are not preserved for checkpointed steps since they are
    not needed for the loss and recomputing dict/aux structures through
    checkpoint adds unnecessary complexity; use non-checkpointed runs for
    fine-grained per-step diagnostics.
    """
    if not use_checkpoint:
        return raw.step(h, x_in, stim_in, return_aux=True, horizon_index=horizon_index, total_horizon=total_horizon)
    depth = len(h)
    h_flat = _flatten_mamba_state(h)
    outputs = torch.utils.checkpoint.checkpoint(
        _checkpointed_recurrent_step,
        raw, depth, x_in, stim_in, horizon_index, total_horizon,
        *h_flat,
        use_reentrant=False,
    )
    pred = outputs[0]
    h_next = _unflatten_mamba_state(outputs[1:], depth)
    return pred, h_next, {}


def _gather_recurrent_start_batch(
    sequence: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    """Gather time indices and flatten them in start-major, then sample order."""

    gathered = sequence.index_select(1, indices)  # [B, starts, ...]
    order = (1, 0) + tuple(range(2, gathered.dim()))
    gathered = gathered.permute(order).contiguous()
    return gathered.reshape(
        int(indices.numel()) * int(sequence.shape[0]), *sequence.shape[2:]
    )


def _plain_recurrent_validation_objective(args, raw) -> bool:
    """Whether packed validation can return the ordinary rollout objective.

    The optimized H200 queue uses this ordinary configuration.  Fall back to
    the serial reference whenever an auxiliary validation loss is enabled.
    """

    if bool(getattr(args, "cto_cut_loss", False)):
        return False
    if bool(getattr(args, "bridge_control_loss", False)):
        return False
    if float(getattr(args, "mamba_var_match_lambda", 0.0)) != 0.0:
        return False
    if float(getattr(args, "mamba_crps", 0.0)) != 0.0:
        return False

    weighted_args = (
        "pc_delta_weight",
        "pc_correction_weight",
        "pc_evidence_weight",
        "pc_prior_weight",
        "pc_prior_smooth_weight",
        "pc_correction_mag_weight",
        "pc_gate_target_weight",
        "pc_gate_floor_weight",
        "atlas_rec_weight",
        "atlas_chart_inv_weight",
        "atlas_dyn_weight",
        "atlas_pred_chart_weight",
        "atlas_overlap_weight",
        "atlas_cocycle_weight",
        "atlas_compose_weight",
        "atlas_balance_weight",
        "atlas_noncollapse_weight",
        "atlas_std_weight",
        "atlas_entropy_floor_weight",
        "atlas_delta_cos_weight",
        "atlas_delta_mse_weight",
        "atlas_delta_norm_weight",
        "atlas_delta_scale_loss_weight",
    )
    if any(float(getattr(args, name, 0.0)) != 0.0 for name in weighted_args):
        return False

    weighted_model_attrs = (
        "shadow_kg_weight",
        "shadow_delta_weight",
        "shadow_spec_weight",
        "fixeda_kg_weight",
        "fixeda_corr_weight",
        "fixeda_mean_weight",
        "fixeda_delta_weight",
        "current_rec_weight",
        "entropy_weight",
    )
    return not any(
        float(getattr(raw, name, 0.0)) != 0.0 for name in weighted_model_attrs
    )


def _packed_recurrent_validation_loss(
    raw,
    state: torch.Tensor,
    stim: torch.Tensor,
    starts: list[int],
    *,
    burn: int,
    horizon: int,
    loss_name: str,
    decay: float,
    start_batch: int,
):
    """Evaluate identical rollout starts in start-major packed batches.

    This changes only execution batching.  Every start receives its own zero
    recurrent state, the same ground-truth burn-in frames, and the same closed
    rollout as the serial reference.
    """

    B = int(state.shape[0])
    burn_start_values = [max(0, int(s) - burn) for s in starts]
    burn_lengths = [
        max(burn_start, int(s) - 1) - burn_start
        for s, burn_start in zip(starts, burn_start_values)
    ]
    if not burn_lengths or len(set(burn_lengths)) != 1:
        return None
    burn_length = int(burn_lengths[0])

    weighted_loss = state.new_tensor(0.0, dtype=torch.float64)
    weighted_first = state.new_tensor(0.0, dtype=torch.float64)
    weighted_last = state.new_tensor(0.0, dtype=torch.float64)
    start_count = 0

    for chunk_start in range(0, len(starts), int(start_batch)):
        chunk = starts[chunk_start : chunk_start + int(start_batch)]
        groups = len(chunk)
        chunk_starts = torch.as_tensor(chunk, device=state.device, dtype=torch.long)
        burn_starts = torch.as_tensor(
            burn_start_values[chunk_start : chunk_start + groups],
            device=state.device,
            dtype=torch.long,
        )

        h = raw.init_state(groups * B, state.device, state.dtype)
        for offset in range(burn_length):
            indices = burn_starts + int(offset)
            state_in = _gather_recurrent_start_batch(state, indices)
            stim_in = (
                _gather_recurrent_start_batch(stim, indices)
                if stim is not None
                else None
            )
            _, h = raw.step(h, state_in, stim_in, return_aux=False)

        x_in = _gather_recurrent_start_batch(state, chunk_starts - 1)
        chunk_losses = []
        chunk_rels = []
        for k in range(int(horizon)):
            input_indices = chunk_starts + int(k) - 1
            target_indices = chunk_starts + int(k)
            stim_in = (
                _gather_recurrent_start_batch(stim, input_indices)
                if stim is not None
                else None
            )
            pred, h, _ = _recurrent_step_maybe_checkpointed(
                raw,
                h,
                x_in,
                stim_in,
                k,
                int(horizon),
                use_checkpoint=False,
            )
            target = _gather_recurrent_start_batch(state, target_indices)
            rel = relative_l2(pred, target)
            if loss_name == "rel_l2":
                step_loss = rel
            else:
                step_loss = elementwise_state_loss(
                    pred.unsqueeze(1), target.unsqueeze(1), loss=loss_name
                )
            if decay != 1.0:
                step_loss = (decay ** k) * step_loss
            chunk_losses.append(step_loss)
            chunk_rels.append(rel)
            x_in = pred

        chunk_loss = torch.stack(chunk_losses).mean()
        weighted_loss = weighted_loss + chunk_loss.double() * groups
        weighted_first = weighted_first + chunk_rels[0].double() * groups
        weighted_last = weighted_last + chunk_rels[-1].double() * groups
        start_count += groups

    if start_count <= 0:
        return None
    denom = float(start_count)
    return (
        weighted_loss / denom,
        weighted_first / denom,
        weighted_last / denom,
        burn_length,
    )


def compute_recurrent_state_bptt_loss(
    model,
    state: torch.Tensor,
    stim: torch.Tensor,
    args,
    epoch: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Chunked-BPTT closed-loop training for recurrent predictive-state models.

    This is the intended objective for ``mamba_state_vector``.  It uses:

      1. a no-grad GT burn-in to construct a long-context recurrent state value;
      2. K-step closed-loop rollout where predictions are fed back as inputs;
      3. ordinary backprop only through the K-step chunk.

    This is Version A in the discussion: inside the chunk, the feedback
    prediction is NOT detached, so future losses can backprop through
    ``xhat_t -> h_t -> xhat_{t+1}``.  The graph is still bounded by K because
    the burn-in state is detached before the rollout chunk.
    """
    raw = unwrap_model(model)
    # Supports both vector states [B,T,D] and field states [B,T,C,H,W].
    # Recurrent models are responsible for returning predictions with the same
    # per-frame shape as their input x_t.
    B, T, _ = get_batch_time_shape(state)
    K_req = int(getattr(args, "mamba_bptt_horizon", 8))
    K_req = max(1, K_req)
    detach_period = max(0, int(getattr(args, "bptt_detach_period", 0)))
    artbp_length = max(
        0, int(getattr(args, "artbp_expected_segment_length", 0))
    )
    if detach_period > 0 and artbp_length > 1:
        raise ValueError("TBPTT and ARTBP cannot be enabled in the same run")
    burn = max(0, int(getattr(args, "mamba_burnin", 64)))
    loss_name = str(getattr(args, "mamba_loss_type", getattr(args, "ar_loss", "rel_l2")))
    decay = float(getattr(args, "mamba_loss_decay", 1.0))
    var_match_lambda = float(getattr(args, "mamba_var_match_lambda", 0.0))  # across-start anomaly-W1 distributional term; 0 = off (default, byte-identical)
    crps_lambda = float(getattr(args, "mamba_crps", 0.0))  # proper-scoring Gaussian-CRPS objective (needs the model's sigma head); 0 = off. Replaces MSE+W1 as the recurrent step loss when set.

    if stim is None:
        stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))

    # Bridge-Control Distillation can replace the base BPTT objective.  This
    # is the intended first implementation of bridge-supervised AR training:
    # a future-informed control target is built offline from true future anchors,
    # but the student receives only causal history/current state.
    if bool(getattr(args, "bridge_control_loss", False)) and bool(getattr(args, "bridge_control_replace_base", False)):
        return compute_recurrent_bridge_control_distill_loss(model, state, stim, args, epoch=epoch)

    # Dataset-cut CTO replaces the base BPTT objective by default.  During
    # validation it is skipped unless --cto_cut_eval is set, so val/loss can
    # remain the ordinary recurrent BPTT diagnostic.
    if bool(getattr(args, "cto_cut_loss", False)):
        do_cto_eval = bool(getattr(args, "cto_cut_eval", False))
        replace_base = bool(getattr(args, "cto_cut_replace_base", True))
        if raw.training or do_cto_eval:
            cto_loss, cto_logs = compute_recurrent_state_cto_cut_loss(
                model, state, stim, args, epoch=epoch
            )
            if replace_base:
                return cto_loss, cto_logs
            # Optional additive mode, for diagnostics only.
            # The clean dataset-cut experiment should keep replace_base=True.

    # target index t: first prediction is x_t from input x_{t-1} and memory.
    # Need t >= 1 and t+K-1 < T.
    max_start = T - K_req
    if max_start < 1:
        z = state.new_tensor(0.0)
        return z, {
            "loss": 0.0,
            "ar/recurrent_bptt_loss": 0.0,
            "ar/recurrent_bptt_horizon": 0.0,
            "ar/recurrent_burnin": float(burn),
            "ar/recurrent_first_rel_l2": 0.0,
            "ar/recurrent_last_rel_l2": 0.0,
        }

    # Prefer starts with full burn-in when possible, but fall back gracefully on
    # short sequences.
    min_start = burn if max_start >= burn else 1
    stride = max(1, int(getattr(args, "mamba_train_stride", 1)))
    starts = list(range(min_start, max_start + 1, stride))
    if not starts:
        starts = list(range(1, max_start + 1, stride))

    if raw.training:
        n_starts = int(getattr(args, "mamba_train_starts_per_sequence", -1))
        if n_starts <= 0:
            n_starts = int(getattr(args, "ar_train_starts_per_sequence", 1))
        n_starts = max(1, min(n_starts, len(starts)))
        controller = getattr(raw, "global_horizon_wiener", None)
        batch_conditioned_global = bool(
            controller is not None
            and getattr(controller, "batch_conditioned", False)
        )
        shared_rollout_start = bool(
            batch_conditioned_global
            or getattr(args, "ar_shared_rollout_start", False)
        )
        if shared_rollout_start:
            # All B trajectories share this one start_t; their mean gradient is
            # the pool used by matched Exact, Static, and global-Wiener arms.
            # In DDP, rank zero broadcasts the same start to every rank.
            n_starts = 1
        use_random = bool(getattr(args, "ar_train_random_starts", True))
        if shared_rollout_start:
            chosen = [
                _shared_global_wiener_start(
                    starts,
                    randomize=use_random,
                    epoch=int(epoch),
                    device=state.device,
                )
            ]
        elif use_random:
            chosen = random.sample(starts, n_starts)
        else:
            offset = max(int(epoch) - 1, 0) % max(1, len(starts))
            chosen = [starts[(offset + i) % len(starts)] for i in range(n_starts)]
    else:
        eval_stride = int(getattr(args, "ar_eval_stride", 4))
        chosen = list(range(min_start, max_start + 1, max(1, eval_stride)))
        if not chosen:
            chosen = [min_start]

    val_start_batch = max(
        1, int(getattr(args, "recurrent_val_start_batch", 1))
    )
    if (
        not raw.training
        and val_start_batch > 1
        and bool(getattr(raw, "is_official_state_mamba", False))
        and getattr(raw, "block_groups", None) is None
        and _plain_recurrent_validation_objective(args, raw)
    ):
        packed = _packed_recurrent_validation_loss(
            raw,
            state,
            stim,
            chosen,
            burn=burn,
            horizon=K_req,
            loss_name=loss_name,
            decay=decay,
            start_batch=val_start_batch,
        )
        if packed is not None:
            base_loss, first_rel, last_rel, used_burn = packed
            logs = {
                "loss": float(base_loss.detach().cpu()),
                "ar/recurrent_bptt_loss": float(base_loss.detach().cpu()),
                "ar/recurrent_bptt_horizon": float(K_req),
                "ar/recurrent_burnin": float(burn),
                "ar/recurrent_used_burnin": float(used_burn),
                "ar/recurrent_num_starts": float(len(chosen)),
                "ar/recurrent_first_rel_l2": float(first_rel.detach().cpu()),
                "ar/recurrent_last_rel_l2": float(last_rel.detach().cpu()),
                "ar/one_step_rel_l2": float(first_rel.detach().cpu()),
                "ar/packed_validation_start_batch": float(val_start_batch),
            }
            if hasattr(raw, "dual_wiener_diagnostics"):
                logs.update(raw.dual_wiener_diagnostics(K_req))
            if hasattr(raw, "dual_wiener_set_probe_losses"):
                raw.dual_wiener_set_probe_losses(None, None)
            return base_loss, logs

    all_losses = []
    first_rels = []
    last_rels = []
    corrs = []
    hidden_norms = []
    alpha_means = []
    alpha_mins = []
    alpha_maxs = []
    resgrad_gates = []
    resgrad_routing_vals = []
    resgrad_branch_ratios = []
    resgrad_branch_norms = []
    resgrad_residual_norms = []
    dual_wiener_total_terms = []
    dual_wiener_noise_terms = []
    pred_stds = []
    gt_stds = []
    dist_pred_by_k = {}  # horizon k -> list of [B,D] pred tensors (grad kept) for the across-start anomaly-W1 term
    dist_tgt_by_k = {}
    used_burns = []
    artbp_edge_scales = []
    cyclic_delta_norms = []
    cyclic_delta_x_ratios = []
    cyclic_internal_changes = []
    cyclic_msg_norms = []
    cyclic_edge_weight_means = []
    cyclic_edge_weight_maxs = []
    shadow_kg_terms = []
    shadow_delta_terms = []
    shadow_spec_terms = []
    shadow_A_sigmas = []
    shadow_delta_rels = []
    shadow_force_rels = []
    shadow_beta_abs_vals = []
    forward_jacobian_terms = []
    forward_jacobian_gains = []

    fixeda_kg_terms = []
    fixeda_corr_terms = []
    fixeda_mean_terms = []
    fixeda_delta_terms = []
    fixeda_cdelta_vals = []
    fixeda_corr_explain_vals = []
    fixeda_delta_rels = []
    fixeda_delta_abs_vals = []
    fixeda_A_means = []
    fixeda_A_maxs = []

    pc_delta_terms = []
    pc_correction_terms = []
    pc_evidence_terms = []
    pc_prior_terms = []
    pc_prior_smooth_terms = []
    pc_corr_mag_terms = []
    pc_gate_target_terms = []
    pc_gate_floor_terms = []
    pc_corr_mag_ratios = []
    pc_target_corr_norms = []
    pc_gate_means = []
    pc_gate_mins = []
    pc_gate_maxs = []
    pc_error_norms = []
    pc_correction_norms = []
    pc_correction_cos_vals = []

    atlas_rec_terms = []
    atlas_chart_inv_terms = []
    atlas_dyn_terms = []
    atlas_pred_chart_terms = []
    atlas_overlap_terms = []
    atlas_cocycle_terms = []
    atlas_compose_terms = []
    atlas_balance_terms = []
    atlas_noncollapse_terms = []
    atlas_std_terms = []
    atlas_entropy_floor_terms = []
    atlas_delta_cos_terms = []
    atlas_delta_mse_terms = []
    atlas_delta_norm_terms = []
    atlas_delta_scale_terms = []
    atlas_delta_cos_vals = []
    atlas_delta_norm_ratios = []
    atlas_delta_pred_stds = []
    atlas_delta_gt_stds = []
    atlas_delta_pred_norms = []
    atlas_delta_gt_norms = []
    atlas_hard_scale_ratios = []
    atlas_hard_scales = []
    atlas_hard_raw_norms = []
    atlas_std_ratios = []
    atlas_pi_entropy_vals = []
    atlas_pi_next_entropy_vals = []
    atlas_perturb_ratios = []
    atlas_target_z_norms = []
    atlas_z_next_norms = []

    subspace_current_rec_terms = []
    subspace_neg_entropy_terms = []
    subspace_attn_entropies = []
    subspace_attn_max_vals = []
    subspace_z_norms = []
    subspace_z_next_norms = []

    bridge_extra_loss = state.new_tensor(0.0)
    bridge_extra_logs = _bridge_control_zero_logs()

    for start_t in chosen:
        # Build h_{start_t-1} from GT frames ending at x_{start_t-2}.  The first
        # differentiable step consumes x_{start_t-1} and predicts x_{start_t}.
        burn_start = max(0, int(start_t) - burn)
        burn_end = max(burn_start, int(start_t) - 1)
        used_burns.append(float(burn_end - burn_start))

        h = raw.init_state(B, state.device, state.dtype)
        if burn_end > burn_start:
            with torch.no_grad():
                for j in range(burn_start, burn_end):
                    stim_j = stim[:, j] if stim is not None else None
                    _, h = raw.step(h, state[:, j], stim_j, return_aux=False)
        h = _detach_recurrent_state(h)

        x_in = state[:, int(start_t) - 1]
        step_losses = []
        step_rels = []
        fixeda_aux_steps = []

        for k in range(K_req):
            target_t = int(start_t) + k
            stim_in = stim[:, target_t - 1] if stim is not None else None
            if (
                k == 0
                and raw.training
                and float(getattr(args, "forward_jacobian_lambda", 0.0)) != 0.0
            ):
                h_for_jreg = _detach_recurrent_state(h)

                def _jreg_recurrent_forward(inp):
                    out = raw.step(
                        h_for_jreg,
                        inp,
                        stim_in,
                        return_aux=False,
                        horizon_index=0,
                        total_horizon=K_req,
                    )
                    return out[0]

                jreg_term, jreg_gain = _forward_jacobian_fd_penalty(
                    x_in,
                    _jreg_recurrent_forward,
                    eps=float(getattr(args, "forward_jacobian_eps", 1e-3)),
                    target=float(getattr(args, "forward_jacobian_target", 1.0)),
                )
                forward_jacobian_terms.append(jreg_term)
                forward_jacobian_gains.append(jreg_gain)
            pred, h, aux = _recurrent_step_maybe_checkpointed(
                raw, h, x_in, stim_in, k, K_req,
                use_checkpoint=bool(getattr(args, "recurrent_grad_checkpoint", False)) and raw.training,
            )
            target = state[:, target_t]

            # The dual-Wiener controller uses a matched half-squared-error
            # probe even when the optimized/reporting loss is relative L2.
            # Its Gaussian derivation is exact for this quadratic probe.  The
            # random noise VJP is measured with autograd.grad in the trainer and
            # therefore never contaminates parameter gradients.
            if raw.training and hasattr(raw, "dual_wiener_probe_terms"):
                dw_total, dw_noise = raw.dual_wiener_probe_terms(pred, target, k)
                if dw_total is not None and dw_noise is not None:
                    dual_wiener_total_terms.append(dw_total)
                    dual_wiener_noise_terms.append(dw_noise)

            if var_match_lambda != 0.0:  # collect for the across-start anomaly-W1 term (grad flows through pred)
                dist_pred_by_k.setdefault(k, []).append(pred)
                dist_tgt_by_k.setdefault(k, []).append(target)

            if crps_lambda != 0.0:  # heteroscedastic objective for the spread head
                sigma = aux.get("sigma") if isinstance(aux, dict) else None
                if sigma is None:
                    raise ValueError("mamba_crps is set but the model produced no sigma; build the model with predict_sigma=True (set --mamba_crps at train time)")
                # DECOUPLED training (avoids the CRPS-only mean-underfit pitfall):
                # the mean (pred) is fit by the ordinary rollout loss with FULL
                # gradient, and sigma is fit by CRPS with the mean DETACHED, so sigma
                # estimates the residual spread -> the irreducible noise as the mean
                # converges, without the saturating CRPS mean-gradient underfitting
                # hard points. A clean noise estimate is what the SNR gate consumes.
                step_loss = relative_l2(pred, target) + crps_lambda * gaussian_crps(pred.detach(), sigma, target)
                loss_at_target = lambda replacement: (
                    relative_l2(pred, replacement)
                    + crps_lambda
                    * gaussian_crps(pred.detach(), sigma, replacement)
                )
                per_sample_loss_at_target = lambda replacement: (
                    _global_wiener_per_sample_state_loss(
                        pred, replacement, "rel_l2"
                    )
                    + crps_lambda
                    * _gaussian_crps_per_sample(
                        pred.detach(), sigma, replacement
                    )
                )
            elif loss_name == "rel_l2":
                step_loss = relative_l2(pred, target)
                loss_at_target = lambda replacement: relative_l2(pred, replacement)
                per_sample_loss_at_target = lambda replacement: _global_wiener_per_sample_state_loss(
                    pred, replacement, "rel_l2"
                )
            else:
                step_loss = elementwise_state_loss(pred.unsqueeze(1), target.unsqueeze(1), loss=loss_name)
                loss_at_target = lambda replacement: elementwise_state_loss(
                    pred.unsqueeze(1), replacement.unsqueeze(1), loss=loss_name
                )
                per_sample_loss_at_target = lambda replacement: _global_wiener_per_sample_state_loss(
                    pred.unsqueeze(1), replacement.unsqueeze(1), loss_name
                )
            if decay != 1.0:
                step_loss = (decay ** k) * step_loss
                unscaled_loss_at_target = loss_at_target
                loss_at_target = lambda replacement: (
                    (decay ** k) * unscaled_loss_at_target(replacement)
                )
                unscaled_per_sample_loss_at_target = per_sample_loss_at_target
                per_sample_loss_at_target = lambda replacement: (
                    (decay ** k)
                    * unscaled_per_sample_loss_at_target(replacement)
                )
            step_loss = _global_wiener_step_loss(
                raw,
                k,
                pred,
                target,
                step_loss,
                loss_at_target,
                per_sample_loss_at_target,
            )
            step_losses.append(step_loss)

            with torch.no_grad():
                rel = relative_l2(pred, target)
                step_rels.append(rel)
                corrs.append(corrcoef_flat(pred, target))
                pred_stds.append(pred.std().detach())
                gt_stds.append(target.std().detach())
                if isinstance(aux, dict):
                    hidden_norms.append(aux.get("hidden_norm", pred.new_tensor(0.0)).detach())
                    alpha_means.append(aux.get("alpha_mean", pred.new_tensor(0.0)).detach())
                    alpha_mins.append(aux.get("alpha_min", pred.new_tensor(0.0)).detach())
                    alpha_maxs.append(aux.get("alpha_max", pred.new_tensor(0.0)).detach())
                    if "resgrad_gate" in aux:
                        resgrad_gates.append(aux["resgrad_gate"].detach())
                    if "resgrad_routing" in aux:
                        resgrad_routing_vals.append(aux["resgrad_routing"].detach())
                    if "resgrad_branch_residual_ratio" in aux:
                        resgrad_branch_ratios.append(aux["resgrad_branch_residual_ratio"].detach())
                    if "resgrad_branch_norm" in aux:
                        resgrad_branch_norms.append(aux["resgrad_branch_norm"].detach())
                    if "resgrad_residual_norm" in aux:
                        resgrad_residual_norms.append(aux["resgrad_residual_norm"].detach())
                    if "cyclic_delta_norm" in aux:
                        cyclic_delta_norms.append(float(aux["cyclic_delta_norm"].detach().cpu()))
                    if "cyclic_delta_x_ratio" in aux:
                        cyclic_delta_x_ratios.append(float(aux["cyclic_delta_x_ratio"].detach().cpu()))
                    if "cyclic_internal_change" in aux:
                        cyclic_internal_changes.append(float(aux["cyclic_internal_change"].detach().cpu()))
                    if "cyclic_msg_norm" in aux:
                        cyclic_msg_norms.append(float(aux["cyclic_msg_norm"].detach().cpu()))
                    if "cyclic_edge_weight_mean" in aux:
                        cyclic_edge_weight_means.append(float(aux["cyclic_edge_weight_mean"].detach().cpu()))
                    if "cyclic_edge_weight_max" in aux:
                        cyclic_edge_weight_maxs.append(float(aux["cyclic_edge_weight_max"].detach().cpu()))
                    if "shadow_A_sigma" in aux:
                        shadow_A_sigmas.append(float(aux["shadow_A_sigma"].detach().cpu()))
                    if "shadow_delta_rel" in aux:
                        shadow_delta_rels.append(float(aux["shadow_delta_rel"].detach().cpu()))
                    if "shadow_force_rel" in aux:
                        shadow_force_rels.append(float(aux["shadow_force_rel"].detach().cpu()))
                    if "shadow_beta_abs" in aux:
                        shadow_beta_abs_vals.append(float(aux["shadow_beta_abs"].detach().cpu()))
                    if "fixeda_delta_rel" in aux:
                        fixeda_delta_rels.append(float(aux["fixeda_delta_rel"].detach().cpu()))
                    if "fixeda_delta_abs" in aux:
                        fixeda_delta_abs_vals.append(float(aux["fixeda_delta_abs"].detach().cpu()))
                    if "fixeda_A_mean" in aux:
                        fixeda_A_means.append(float(aux["fixeda_A_mean"].detach().cpu()))
                    if "fixeda_A_max" in aux:
                        fixeda_A_maxs.append(float(aux["fixeda_A_max"].detach().cpu()))
                    if "subspace_attn_entropy" in aux:
                        subspace_attn_entropies.append(float(aux["subspace_attn_entropy"].detach().cpu()))
                    if "subspace_attn_max" in aux:
                        subspace_attn_max_vals.append(float(aux["subspace_attn_max"].detach().cpu()))
                    if "subspace_z_norm" in aux:
                        subspace_z_norms.append(float(aux["subspace_z_norm"].detach().cpu()))
                    if "subspace_z_next_norm" in aux:
                        subspace_z_next_norms.append(float(aux["subspace_z_next_norm"].detach().cpu()))

                if isinstance(aux, dict):
                    if "subspace_current_rec_loss" in aux:
                        subspace_current_rec_terms.append(aux["subspace_current_rec_loss"])
                    if "subspace_neg_entropy_loss" in aux:
                        subspace_neg_entropy_terms.append(aux["subspace_neg_entropy_loss"])

            # Observation anchoring for atlas models: prevent the atlas
            # predictor from satisfying internal chart consistency with an
            # almost-zero decoded output.  This is intentionally an amplitude
            # floor, not a variance-matching objective, so it does not force the
            # model to overfit noisy variance once the minimum ratio is reached.
            if isinstance(aux, dict) and "atlas_z_next_by_chart" in aux:
                atlas_std_w_tmp = float(getattr(args, "atlas_std_weight", 0.0))
                pred_std_step = pred.reshape(pred.shape[0], -1).std(dim=-1).mean()
                target_std_step = target.reshape(target.shape[0], -1).std(dim=-1).mean().clamp_min(1e-8)
                std_ratio_step = pred_std_step / target_std_step
                atlas_std_ratios.append(float(std_ratio_step.detach().cpu()))
                if atlas_std_w_tmp != 0.0:
                    min_ratio = pred.new_tensor(float(getattr(args, "atlas_std_min_ratio", 0.20)))
                    atlas_std_terms.append(F.relu(min_ratio - std_ratio_step).pow(2))

                entropy_w_tmp = float(getattr(args, "atlas_entropy_floor_weight", 0.0))
                if entropy_w_tmp != 0.0 and "atlas_pi" in aux:
                    h_pi = (-(aux["atlas_pi"] * aux["atlas_pi"].clamp_min(1e-8).log()).sum(dim=-1)).mean()
                    h_next = (-(aux.get("atlas_pi_next", aux["atlas_pi"]) * aux.get("atlas_pi_next", aux["atlas_pi"]).clamp_min(1e-8).log()).sum(dim=-1)).mean()
                    h_min = pred.new_tensor(float(getattr(args, "atlas_entropy_min", 1.0)))
                    atlas_entropy_floor_terms.append(0.5 * (F.relu(h_min - h_pi).pow(2) + F.relu(h_min - h_next).pow(2)))

            # Directional observation anchoring for atlas models.
            # The std/noncollapse anchors can force nonzero output amplitude,
            # but they do not make the decoded transition point in the true
            # next-state direction.  This term anchors the learned chart
            # transition to the local observation-space tangent
            #     x_{t+1} - x_t,
            # so increased variance must carry predictive signal rather than
            # arbitrary amplitude.
            if isinstance(aux, dict) and "atlas_z_next_by_chart" in aux:
                delta_cos_w_tmp = float(getattr(args, "atlas_delta_cos_weight", 0.0))
                delta_mse_w_tmp = float(getattr(args, "atlas_delta_mse_weight", 0.0))
                delta_norm_w_tmp = float(getattr(args, "atlas_delta_norm_weight", 0.0))
                if delta_cos_w_tmp != 0.0 or delta_mse_w_tmp != 0.0 or delta_norm_w_tmp != 0.0:
                    gt_prev = x_in.detach()
                    pred_delta = pred - gt_prev
                    target_delta = target - gt_prev
                    pred_delta_flat = pred_delta.reshape(pred_delta.shape[0], -1)
                    target_delta_flat = target_delta.reshape(target_delta.shape[0], -1)
                    pred_delta_norm = pred_delta_flat.norm(dim=-1).clamp_min(1e-8)
                    target_delta_norm = target_delta_flat.norm(dim=-1).clamp_min(1e-8)
                    delta_cos = (pred_delta_flat * target_delta_flat).sum(dim=-1) / (pred_delta_norm * target_delta_norm)
                    norm_ratio = pred_delta_norm / target_delta_norm
                    pred_delta_std = pred_delta_flat.std(dim=-1)
                    target_delta_std = target_delta_flat.std(dim=-1).clamp_min(1e-8)
                    atlas_delta_cos_vals.append(float(delta_cos.mean().detach().cpu()))
                    atlas_delta_norm_ratios.append(float(norm_ratio.mean().detach().cpu()))
                    atlas_delta_pred_stds.append(float(pred_delta_std.mean().detach().cpu()))
                    atlas_delta_gt_stds.append(float(target_delta_std.mean().detach().cpu()))
                    atlas_delta_pred_norms.append(float(pred_delta_norm.mean().detach().cpu()))
                    atlas_delta_gt_norms.append(float(target_delta_norm.mean().detach().cpu()))
                    if "atlas_hard_delta_scale_ratio" in aux:
                        atlas_hard_scale_ratios.append(float(aux["atlas_hard_delta_scale_ratio"].detach().cpu()))
                    if "atlas_hard_delta_scale" in aux:
                        atlas_hard_scales.append(float(aux["atlas_hard_delta_scale"].detach().cpu()))
                    if "atlas_hard_delta_raw_norm" in aux:
                        atlas_hard_raw_norms.append(float(aux["atlas_hard_delta_raw_norm"].detach().cpu()))
                    if delta_cos_w_tmp != 0.0:
                        atlas_delta_cos_terms.append((1.0 - delta_cos).mean())
                    if delta_mse_w_tmp != 0.0:
                        # Normalize by the target tangent energy so this loss
                        # is comparable across subjects/batches and does not
                        # simply duplicate absolute x-space MSE.
                        denom = target_delta_flat.pow(2).mean(dim=-1).clamp_min(1e-8)
                        delta_mse = (pred_delta_flat - target_delta_flat).pow(2).mean(dim=-1) / denom
                        atlas_delta_mse_terms.append(delta_mse.mean())
                    if delta_norm_w_tmp != 0.0:
                        # Anti-identity constraint for tangent atlas. In the
                        # tangent formulation x_hat = x_t + delta_hat, output
                        # variance is inherited from x_t, so the old std floor
                        # cannot prevent delta_hat ~= 0. This hinge directly
                        # forbids the near-zero tangent shortcut.
                        min_ratio = pred.new_tensor(float(getattr(args, "atlas_delta_norm_min_ratio", 0.25)))
                        atlas_delta_norm_terms.append(F.relu(min_ratio - norm_ratio).pow(2).mean())
                    scale_w_tmp = float(getattr(args, "atlas_delta_scale_loss_weight", 0.0))
                    if scale_w_tmp != 0.0:
                        # Direct scale supervision for hard-norm tangent mode.
                        # This avoids relying only on the ratio hinge, whose
                        # gradient is weak when target_delta_norm is huge.
                        scale_loss = (pred_delta_norm.log() - target_delta_norm.detach().log()).pow(2).mean()
                        atlas_delta_scale_terms.append(scale_loss)

            if isinstance(aux, dict) and "pc_gate" in aux:
                pc_gate = aux["pc_gate"]
                pc_gate_means.append(float(pc_gate.mean().detach().cpu()))
                pc_gate_mins.append(float(pc_gate.min().detach().cpu()))
                pc_gate_maxs.append(float(pc_gate.max().detach().cpu()))
                if "pc_error_norm" in aux:
                    pc_error_norms.append(float(aux["pc_error_norm"].detach().cpu()))
                if "pc_correction_norm" in aux:
                    pc_correction_norms.append(float(aux["pc_correction_norm"].detach().cpu()))

                delta_w_pc = float(getattr(args, "pc_delta_weight", 0.0))
                if delta_w_pc != 0.0:
                    gt_prev = state[:, target_t - 1]
                    pred_delta = pred - x_in
                    gt_delta = target - gt_prev
                    denom = gt_delta.reshape(gt_delta.shape[0], -1).pow(2).mean().clamp_min(1e-8)
                    pc_delta_terms.append((pred_delta - gt_delta).pow(2).mean() / denom)

                corr_w_pc = float(getattr(args, "pc_correction_weight", 0.0))
                corr_mag_w_pc = float(getattr(args, "pc_correction_mag_weight", 0.0))
                if (corr_w_pc != 0.0 or corr_mag_w_pc != 0.0) and "pc_correction" in aux and "pc_prior_pred" in aux:
                    corr = aux["pc_correction"].reshape(pred.shape[0], -1)
                    # Detach the prior in the correction target so this auxiliary
                    # term trains the correction/gate to move from the current
                    # prior toward the target, without turning the prior itself
                    # into a shortcut for this term.
                    target_corr = (target - aux["pc_prior_pred"].detach()).reshape(pred.shape[0], -1)
                    cos = F.cosine_similarity(corr, target_corr, dim=-1, eps=1e-8)
                    if corr_w_pc != 0.0:
                        pc_correction_terms.append((1.0 - cos).mean())
                    pc_correction_cos_vals.append(float(cos.mean().detach().cpu()))

                    # Direction-only cosine allowed the correction magnitude to
                    # explode.  This term makes the *actual applied correction*
                    # have the right norm relative to the required target-prior
                    # update.  It is scale-normalized per sample.
                    corr_norm = corr.norm(dim=-1)
                    target_corr_norm = target_corr.norm(dim=-1).clamp_min(1e-6)
                    ratio = corr_norm / target_corr_norm
                    pc_corr_mag_ratios.append(float(ratio.mean().detach().cpu()))
                    pc_target_corr_norms.append(float(target_corr_norm.mean().detach().cpu()))
                    if corr_mag_w_pc != 0.0:
                        target_ratio = float(getattr(args, "pc_correction_mag_target", 1.0))
                        pc_corr_mag_terms.append((ratio - target_ratio).pow(2).mean())

                evidence_w_pc = float(getattr(args, "pc_evidence_weight", 0.0))
                if evidence_w_pc != 0.0 and "pc_evidence_pred" in aux:
                    evidence = aux["pc_evidence_pred"]
                    if loss_name == "rel_l2":
                        pc_evidence_terms.append(relative_l2(evidence, target))
                    else:
                        pc_evidence_terms.append(elementwise_state_loss(evidence.unsqueeze(1), target.unsqueeze(1), loss=loss_name))

                prior_w_pc = float(getattr(args, "pc_prior_weight", 0.0))
                if prior_w_pc != 0.0 and "pc_prior_pred" in aux:
                    prior = aux["pc_prior_pred"]
                    if loss_name == "rel_l2":
                        pc_prior_terms.append(relative_l2(prior, target))
                    else:
                        pc_prior_terms.append(elementwise_state_loss(prior.unsqueeze(1), target.unsqueeze(1), loss=loss_name))

                # Optional prior-smooth loss pins the prior branch to the
                # history-based persistence prediction x_{t-1}.  This is not
                # meant to improve one-step accuracy directly; it prevents the
                # prior/evidence decomposition from becoming arbitrary.
                prior_smooth_w_pc = float(getattr(args, "pc_prior_smooth_weight", 0.0))
                if prior_smooth_w_pc != 0.0 and "pc_prior_pred" in aux:
                    prior = aux["pc_prior_pred"]
                    prior_target = x_in.detach()
                    denom = prior_target.reshape(prior_target.shape[0], -1).pow(2).mean().clamp_min(1e-8)
                    pc_prior_smooth_terms.append((prior - prior_target).pow(2).mean() / denom)

                gate_floor_w_pc = float(getattr(args, "pc_gate_floor_weight", 0.0))
                if gate_floor_w_pc != 0.0:
                    floor = float(getattr(args, "pc_gate_floor", 0.05))
                    pc_gate_floor_terms.append(F.relu(floor - pc_gate).pow(2).mean())

                gate_target_w_pc = float(getattr(args, "pc_gate_target_weight", 0.0))
                if gate_target_w_pc != 0.0:
                    gate_target = float(getattr(args, "pc_gate_target", 0.5))
                    pc_gate_target_terms.append((pc_gate - gate_target).pow(2).mean())

            if isinstance(aux, dict) and hasattr(raw, "atlas_auxiliary_losses") and "atlas_z_next_by_chart" in aux:
                atlas_losses = raw.atlas_auxiliary_losses(aux, x_in, target)
                if float(getattr(args, "atlas_rec_weight", 0.0)) != 0.0:
                    atlas_rec_terms.append(atlas_losses.get("rec", pred.new_tensor(0.0)))
                if float(getattr(args, "atlas_chart_inv_weight", 0.0)) != 0.0:
                    atlas_chart_inv_terms.append(atlas_losses.get("chart_inv", pred.new_tensor(0.0)))
                if float(getattr(args, "atlas_dyn_weight", 0.0)) != 0.0:
                    atlas_dyn_terms.append(atlas_losses.get("dyn", pred.new_tensor(0.0)))
                if float(getattr(args, "atlas_pred_chart_weight", 0.0)) != 0.0:
                    atlas_pred_chart_terms.append(atlas_losses.get("pred_chart", pred.new_tensor(0.0)))
                if float(getattr(args, "atlas_overlap_weight", 0.0)) != 0.0:
                    atlas_overlap_terms.append(atlas_losses.get("overlap", pred.new_tensor(0.0)))
                if float(getattr(args, "atlas_cocycle_weight", 0.0)) != 0.0:
                    atlas_cocycle_terms.append(atlas_losses.get("cocycle", pred.new_tensor(0.0)))
                if float(getattr(args, "atlas_balance_weight", 0.0)) != 0.0:
                    atlas_balance_terms.append(atlas_losses.get("pi_balance", pred.new_tensor(0.0)))
                if float(getattr(args, "atlas_noncollapse_weight", 0.0)) != 0.0:
                    atlas_noncollapse_terms.append(atlas_losses.get("noncollapse", pred.new_tensor(0.0)))
                # K-step composition endpoint in chart space: the final latent
                # reached by composing local transitions through closed-loop BPTT
                # should agree with the final GT state's coordinate in the final
                # predicted chart.  This is not an x-space endpoint loss.
                if k == K_req - 1 and float(getattr(args, "atlas_compose_weight", 0.0)) != 0.0:
                    atlas_compose_terms.append(atlas_losses.get("dyn", pred.new_tensor(0.0)))
                with torch.no_grad():
                    if "atlas_pi_entropy" in aux:
                        atlas_pi_entropy_vals.append(float(aux["atlas_pi_entropy"].detach().cpu()))
                    if "atlas_pi_next_entropy" in aux:
                        atlas_pi_next_entropy_vals.append(float(aux["atlas_pi_next_entropy"].detach().cpu()))
                    if "perturb_ratio" in atlas_losses:
                        atlas_perturb_ratios.append(float(atlas_losses["perturb_ratio"].detach().cpu()))
                    if "target_z_norm" in atlas_losses:
                        atlas_target_z_norms.append(float(atlas_losses["target_z_norm"].detach().cpu()))
                    if "z_next_norm" in atlas_losses:
                        atlas_z_next_norms.append(float(atlas_losses["z_next_norm"].detach().cpu()))

            if isinstance(aux, dict) and "fixeda_r" in aux:
                fixeda_aux_steps.append(aux)

            if isinstance(aux, dict) and hasattr(raw, "shadow_folded_loss") and "z_next" in aux:
                kg_w = float(getattr(raw, "shadow_kg_weight", 0.0))
                kg_h = int(getattr(raw, "shadow_kg_horizon", 0))
                if kg_w != 0.0 and kg_h > 0 and target_t < T:
                    end_t = min(T, target_t + kg_h)
                    if end_t > target_t:
                        future = state[:, target_t:end_t]
                        shadow_kg_terms.append(raw.shadow_folded_loss(aux["z_next"], future))
                delta_w = float(getattr(raw, "shadow_delta_weight", 0.0))
                if delta_w != 0.0 and "shadow_delta_loss" in aux:
                    shadow_delta_terms.append(aux["shadow_delta_loss"])
                spec_w = float(getattr(raw, "shadow_spec_weight", 0.0))
                if spec_w != 0.0 and hasattr(raw, "shadow_spectral_loss"):
                    shadow_spec_terms.append(raw.shadow_spectral_loss())

            # Version A keeps both temporal carries open: the prediction fed to
            # the next step and the recurrent state.  ARTBP samples one shared
            # edge decision for both carries, so no temporal path bypasses the
            # randomized cut.  A surviving edge is compensated by the inverse
            # survival probability; the forward prediction and state values are
            # unchanged.
            if (
                detach_period > 0
                and (k + 1) < K_req
                and ((k + 1) % detach_period) == 0
                and raw.training
            ):
                x_in = pred.detach()
                h = _detach_recurrent_state(h)
            elif artbp_length > 1 and (k + 1) < K_req and raw.training:
                edge_scale = _sample_artbp_edge_scale(
                    artbp_length, pred.device
                )
                x_in = _backward_scale_value(pred, edge_scale)
                h = _backward_scale_value(h, edge_scale)
                artbp_edge_scales.append(edge_scale)
            else:
                x_in = pred

        if fixeda_aux_steps and hasattr(raw, "fixeda_segment_regularization"):
            reg = raw.fixeda_segment_regularization(fixeda_aux_steps)
            fixeda_kg_terms.append(reg.get("kg", state.new_tensor(0.0)))
            fixeda_corr_terms.append(reg.get("corr", state.new_tensor(0.0)))
            fixeda_mean_terms.append(reg.get("mean", state.new_tensor(0.0)))
            fixeda_delta_terms.append(reg.get("delta", state.new_tensor(0.0)))
            fixeda_cdelta_vals.append(float(reg.get("cdelta", state.new_tensor(0.0)).detach().cpu()))
            fixeda_corr_explain_vals.append(float(reg.get("corr_explain", state.new_tensor(0.0)).detach().cpu()))

        if step_losses:
            all_losses.append(torch.stack(step_losses).mean())
            first_rels.append(step_rels[0].detach())
            last_rels.append(step_rels[-1].detach())

    if not all_losses:
        if hasattr(raw, "dual_wiener_set_probe_losses"):
            raw.dual_wiener_set_probe_losses(None, None)
        z = state.new_tensor(0.0)
        return z, {"loss": 0.0, "ar/recurrent_bptt_loss": 0.0}

    if hasattr(raw, "dual_wiener_set_probe_losses"):
        dw_total = torch.stack(dual_wiener_total_terms).mean() if dual_wiener_total_terms else None
        dw_noise = torch.stack(dual_wiener_noise_terms).mean() if dual_wiener_noise_terms else None
        raw.dual_wiener_set_probe_losses(dw_total, dw_noise)

    base_loss = torch.stack(all_losses).mean()
    loss = base_loss
    forward_jacobian_loss = (
        torch.stack(forward_jacobian_terms).mean()
        if forward_jacobian_terms
        else state.new_tensor(0.0)
    )
    forward_jacobian_lambda = float(
        getattr(args, "forward_jacobian_lambda", 0.0)
    )
    loss = loss + forward_jacobian_lambda * forward_jacobian_loss
    var_match_w1 = None
    if var_match_lambda != 0.0:  # add the distributional (across-start anomaly-W1) term to the OPTIMIZED loss; base_loss (pure MSE) stays the logged diagnostic
        var_match_w1 = _across_start_anomaly_w1(dist_pred_by_k, dist_tgt_by_k)
        if var_match_w1 is not None:
            loss = loss + var_match_lambda * var_match_w1
    if bool(getattr(args, "bridge_control_loss", False)) and not bool(getattr(args, "bridge_control_replace_base", False)):
        bridge_extra_loss, bridge_extra_logs = compute_recurrent_bridge_control_distill_loss(
            model, state, stim, args, epoch=epoch
        )
        loss = loss + bridge_extra_loss
    shadow_kg_loss = torch.stack(shadow_kg_terms).mean() if shadow_kg_terms else state.new_tensor(0.0)
    shadow_delta_loss = torch.stack(shadow_delta_terms).mean() if shadow_delta_terms else state.new_tensor(0.0)
    shadow_spec_loss = torch.stack(shadow_spec_terms).mean() if shadow_spec_terms else state.new_tensor(0.0)
    raw_kg_w = float(getattr(raw, "shadow_kg_weight", 0.0))
    raw_delta_w = float(getattr(raw, "shadow_delta_weight", 0.0))
    raw_spec_w = float(getattr(raw, "shadow_spec_weight", 0.0))
    if raw_kg_w != 0.0:
        loss = loss + raw_kg_w * shadow_kg_loss
    if raw_delta_w != 0.0:
        loss = loss + raw_delta_w * shadow_delta_loss
    if raw_spec_w != 0.0:
        loss = loss + raw_spec_w * shadow_spec_loss

    fixeda_kg_loss = torch.stack(fixeda_kg_terms).mean() if fixeda_kg_terms else state.new_tensor(0.0)
    fixeda_corr_loss = torch.stack(fixeda_corr_terms).mean() if fixeda_corr_terms else state.new_tensor(0.0)
    fixeda_mean_loss = torch.stack(fixeda_mean_terms).mean() if fixeda_mean_terms else state.new_tensor(0.0)
    fixeda_delta_loss = torch.stack(fixeda_delta_terms).mean() if fixeda_delta_terms else state.new_tensor(0.0)
    fixeda_kg_w = float(getattr(raw, "fixeda_kg_weight", 0.0))
    fixeda_corr_w = float(getattr(raw, "fixeda_corr_weight", 0.0))
    fixeda_mean_w = float(getattr(raw, "fixeda_mean_weight", 0.0))
    fixeda_delta_w = float(getattr(raw, "fixeda_delta_weight", 0.0))
    if fixeda_kg_w != 0.0:
        loss = loss + fixeda_kg_w * fixeda_kg_loss
    if fixeda_corr_w != 0.0:
        loss = loss + fixeda_corr_w * fixeda_corr_loss
    if fixeda_mean_w != 0.0:
        loss = loss + fixeda_mean_w * fixeda_mean_loss
    if fixeda_delta_w != 0.0:
        loss = loss + fixeda_delta_w * fixeda_delta_loss

    subspace_current_rec_loss = torch.stack(subspace_current_rec_terms).mean() if subspace_current_rec_terms else state.new_tensor(0.0)
    subspace_neg_entropy_loss = torch.stack(subspace_neg_entropy_terms).mean() if subspace_neg_entropy_terms else state.new_tensor(0.0)
    subspace_current_rec_w = float(getattr(raw, "current_rec_weight", 0.0))
    subspace_entropy_w = float(getattr(raw, "entropy_weight", 0.0))
    if subspace_current_rec_w != 0.0:
        loss = loss + subspace_current_rec_w * subspace_current_rec_loss
    if subspace_entropy_w != 0.0:
        loss = loss + subspace_entropy_w * subspace_neg_entropy_loss

    pc_delta_loss = torch.stack(pc_delta_terms).mean() if pc_delta_terms else state.new_tensor(0.0)
    pc_correction_loss = torch.stack(pc_correction_terms).mean() if pc_correction_terms else state.new_tensor(0.0)
    pc_evidence_loss = torch.stack(pc_evidence_terms).mean() if pc_evidence_terms else state.new_tensor(0.0)
    pc_prior_loss = torch.stack(pc_prior_terms).mean() if pc_prior_terms else state.new_tensor(0.0)
    pc_prior_smooth_loss = torch.stack(pc_prior_smooth_terms).mean() if pc_prior_smooth_terms else state.new_tensor(0.0)
    pc_corr_mag_loss = torch.stack(pc_corr_mag_terms).mean() if pc_corr_mag_terms else state.new_tensor(0.0)
    pc_gate_target_loss = torch.stack(pc_gate_target_terms).mean() if pc_gate_target_terms else state.new_tensor(0.0)
    pc_gate_floor_loss = torch.stack(pc_gate_floor_terms).mean() if pc_gate_floor_terms else state.new_tensor(0.0)
    pc_delta_w = float(getattr(args, "pc_delta_weight", 0.0))
    pc_correction_w = float(getattr(args, "pc_correction_weight", 0.0))
    pc_evidence_w = float(getattr(args, "pc_evidence_weight", 0.0))
    pc_prior_w = float(getattr(args, "pc_prior_weight", 0.0))
    pc_prior_smooth_w = float(getattr(args, "pc_prior_smooth_weight", 0.0))
    pc_corr_mag_w = float(getattr(args, "pc_correction_mag_weight", 0.0))
    pc_gate_target_w = float(getattr(args, "pc_gate_target_weight", 0.0))
    pc_gate_floor_w = float(getattr(args, "pc_gate_floor_weight", 0.0))
    if pc_delta_w != 0.0:
        loss = loss + pc_delta_w * pc_delta_loss
    if pc_correction_w != 0.0:
        loss = loss + pc_correction_w * pc_correction_loss
    if pc_evidence_w != 0.0:
        loss = loss + pc_evidence_w * pc_evidence_loss
    if pc_prior_w != 0.0:
        loss = loss + pc_prior_w * pc_prior_loss
    if pc_prior_smooth_w != 0.0:
        loss = loss + pc_prior_smooth_w * pc_prior_smooth_loss
    if pc_corr_mag_w != 0.0:
        loss = loss + pc_corr_mag_w * pc_corr_mag_loss
    if pc_gate_target_w != 0.0:
        loss = loss + pc_gate_target_w * pc_gate_target_loss
    if pc_gate_floor_w != 0.0:
        loss = loss + pc_gate_floor_w * pc_gate_floor_loss
    atlas_rec_loss = torch.stack(atlas_rec_terms).mean() if atlas_rec_terms else state.new_tensor(0.0)
    atlas_chart_inv_loss = torch.stack(atlas_chart_inv_terms).mean() if atlas_chart_inv_terms else state.new_tensor(0.0)
    atlas_dyn_loss = torch.stack(atlas_dyn_terms).mean() if atlas_dyn_terms else state.new_tensor(0.0)
    atlas_pred_chart_loss = torch.stack(atlas_pred_chart_terms).mean() if atlas_pred_chart_terms else state.new_tensor(0.0)
    atlas_overlap_loss = torch.stack(atlas_overlap_terms).mean() if atlas_overlap_terms else state.new_tensor(0.0)
    atlas_cocycle_loss = torch.stack(atlas_cocycle_terms).mean() if atlas_cocycle_terms else state.new_tensor(0.0)
    atlas_compose_loss = torch.stack(atlas_compose_terms).mean() if atlas_compose_terms else state.new_tensor(0.0)
    atlas_balance_loss = torch.stack(atlas_balance_terms).mean() if atlas_balance_terms else state.new_tensor(0.0)
    atlas_noncollapse_loss = torch.stack(atlas_noncollapse_terms).mean() if atlas_noncollapse_terms else state.new_tensor(0.0)
    atlas_std_loss = torch.stack(atlas_std_terms).mean() if atlas_std_terms else state.new_tensor(0.0)
    atlas_entropy_floor_loss = torch.stack(atlas_entropy_floor_terms).mean() if atlas_entropy_floor_terms else state.new_tensor(0.0)
    atlas_delta_cos_loss = torch.stack(atlas_delta_cos_terms).mean() if atlas_delta_cos_terms else state.new_tensor(0.0)
    atlas_delta_mse_loss = torch.stack(atlas_delta_mse_terms).mean() if atlas_delta_mse_terms else state.new_tensor(0.0)
    atlas_delta_norm_loss = torch.stack(atlas_delta_norm_terms).mean() if atlas_delta_norm_terms else state.new_tensor(0.0)
    atlas_delta_scale_loss = torch.stack(atlas_delta_scale_terms).mean() if atlas_delta_scale_terms else state.new_tensor(0.0)
    atlas_rec_w = float(getattr(args, "atlas_rec_weight", 0.0))
    atlas_chart_inv_w = float(getattr(args, "atlas_chart_inv_weight", 0.0))
    atlas_dyn_w = float(getattr(args, "atlas_dyn_weight", 0.0))
    atlas_pred_chart_w = float(getattr(args, "atlas_pred_chart_weight", 0.0))
    atlas_overlap_w = float(getattr(args, "atlas_overlap_weight", 0.0))
    atlas_cocycle_w = float(getattr(args, "atlas_cocycle_weight", 0.0))
    atlas_compose_w = float(getattr(args, "atlas_compose_weight", 0.0))
    atlas_balance_w = float(getattr(args, "atlas_balance_weight", 0.0))
    atlas_noncollapse_w = float(getattr(args, "atlas_noncollapse_weight", 0.0))
    atlas_std_w = float(getattr(args, "atlas_std_weight", 0.0))
    atlas_entropy_floor_w = float(getattr(args, "atlas_entropy_floor_weight", 0.0))
    atlas_delta_cos_w = float(getattr(args, "atlas_delta_cos_weight", 0.0))
    atlas_delta_mse_w = float(getattr(args, "atlas_delta_mse_weight", 0.0))
    atlas_delta_norm_w = float(getattr(args, "atlas_delta_norm_weight", 0.0))
    atlas_delta_scale_w = float(getattr(args, "atlas_delta_scale_loss_weight", 0.0))
    if atlas_rec_w != 0.0:
        loss = loss + atlas_rec_w * atlas_rec_loss
    if atlas_chart_inv_w != 0.0:
        loss = loss + atlas_chart_inv_w * atlas_chart_inv_loss
    if atlas_dyn_w != 0.0:
        loss = loss + atlas_dyn_w * atlas_dyn_loss
    if atlas_pred_chart_w != 0.0:
        loss = loss + atlas_pred_chart_w * atlas_pred_chart_loss
    if atlas_overlap_w != 0.0:
        loss = loss + atlas_overlap_w * atlas_overlap_loss
    if atlas_cocycle_w != 0.0:
        loss = loss + atlas_cocycle_w * atlas_cocycle_loss
    if atlas_compose_w != 0.0:
        loss = loss + atlas_compose_w * atlas_compose_loss
    if atlas_balance_w != 0.0:
        loss = loss + atlas_balance_w * atlas_balance_loss
    if atlas_noncollapse_w != 0.0:
        loss = loss + atlas_noncollapse_w * atlas_noncollapse_loss
    if atlas_std_w != 0.0:
        loss = loss + atlas_std_w * atlas_std_loss
    if atlas_entropy_floor_w != 0.0:
        loss = loss + atlas_entropy_floor_w * atlas_entropy_floor_loss
    if atlas_delta_cos_w != 0.0:
        loss = loss + atlas_delta_cos_w * atlas_delta_cos_loss
    if atlas_delta_mse_w != 0.0:
        loss = loss + atlas_delta_mse_w * atlas_delta_mse_loss
    if atlas_delta_norm_w != 0.0:
        loss = loss + atlas_delta_norm_w * atlas_delta_norm_loss
    if atlas_delta_scale_w != 0.0:
        loss = loss + atlas_delta_scale_w * atlas_delta_scale_loss

    # Optional additive CTO-cut mode.  This is not the clean default, but is kept
    # for controlled diagnostics against the old additive experiments.
    if bool(getattr(args, "cto_cut_loss", False)) and raw.training and not bool(getattr(args, "cto_cut_replace_base", True)):
        cto_loss, cto_logs = compute_recurrent_state_cto_cut_loss(model, state, stim, args, epoch=epoch)
        loss = base_loss + cto_loss
    else:
        cto_logs = {}

    if raw.training and bool(getattr(args, "fast_train_logging", False)):
        # The full dictionary below contains many useful validation diagnostics,
        # but converting each tensor to a Python float synchronizes CUDA once per
        # field and per minibatch.  Training only needs these core loss scalars;
        # trainer.py accumulates the detached tensors on device until epoch end.
        return loss, {
            "loss": loss.detach(),
            "ar/recurrent_bptt_loss": base_loss.detach(),
            "ar/recurrent_bptt_horizon": float(K_req),
        }

    logs = {
        "loss": float(loss.detach().cpu()),
        "ar/recurrent_bptt_loss": float(base_loss.detach().cpu()),
        "ar/forward_jacobian_loss": float(forward_jacobian_loss.detach().cpu()),
        "ar/forward_jacobian_lambda": float(forward_jacobian_lambda),
        "ar/forward_jacobian_weighted_loss": float(
            (forward_jacobian_lambda * forward_jacobian_loss).detach().cpu()
        ),
        "ar/forward_jacobian_gain": _reduce_metric_scalars(
            forward_jacobian_gains
        ),
        "ar/bridge_extra_loss": float(bridge_extra_loss.detach().cpu()),
        "ar/recurrent_bptt_horizon": float(K_req),
        "ar/bptt_detach_period": float(detach_period),
        "ar/artbp_expected_segment_length": float(artbp_length),
        "ar/artbp_cut_fraction": (
            float(sum(scale == 0.0 for scale in artbp_edge_scales))
            / float(len(artbp_edge_scales))
            if artbp_edge_scales else 0.0
        ),
        "ar/recurrent_burnin": float(burn),
        "ar/recurrent_used_burnin": float(sum(used_burns) / max(len(used_burns), 1)),
        "ar/recurrent_num_starts": float(len(chosen)),
        "ar/recurrent_first_rel_l2": float(torch.stack(first_rels).mean().detach().cpu()) if first_rels else 0.0,
        "ar/recurrent_last_rel_l2": float(torch.stack(last_rels).mean().detach().cpu()) if last_rels else 0.0,
        "ar/one_step_rel_l2": float(torch.stack(first_rels).mean().detach().cpu()) if first_rels else 0.0,
        "ar/one_step_corr": float(torch.stack(corrs).mean().detach().cpu()) if corrs else 0.0,
        "ar/recurrent_hidden_norm": _reduce_metric_scalars(hidden_norms),
        "ar/recurrent_alpha_mean": _reduce_metric_scalars(alpha_means),
        "ar/recurrent_alpha_min": _reduce_metric_scalars(alpha_mins, "min"),
        "ar/recurrent_alpha_max": _reduce_metric_scalars(alpha_maxs, "max"),
        "ar/resgrad_routing": _reduce_metric_scalars(resgrad_routing_vals),
        "ar/resgrad_gate_mean": _reduce_metric_scalars(resgrad_gates, default=1.0),
        "ar/resgrad_gate_min": _reduce_metric_scalars(resgrad_gates, "min", default=1.0),
        "ar/resgrad_gate_max": _reduce_metric_scalars(resgrad_gates, "max", default=1.0),
        "ar/resgrad_branch_residual_ratio": _reduce_metric_scalars(resgrad_branch_ratios),
        "ar/resgrad_branch_norm": _reduce_metric_scalars(resgrad_branch_norms),
        "ar/resgrad_residual_norm": _reduce_metric_scalars(resgrad_residual_norms),
        "ar/recurrent_pred_std": _reduce_metric_scalars(pred_stds),
        "ar/recurrent_gt_std": _reduce_metric_scalars(gt_stds),
        "ar/recurrent_var_match_w1": float(var_match_w1.detach().cpu()) if var_match_w1 is not None else 0.0,
        "ar/cyclic_delta_norm": float(sum(cyclic_delta_norms) / max(len(cyclic_delta_norms), 1)) if cyclic_delta_norms else 0.0,
        "ar/cyclic_delta_x_ratio": float(sum(cyclic_delta_x_ratios) / max(len(cyclic_delta_x_ratios), 1)) if cyclic_delta_x_ratios else 0.0,
        "ar/cyclic_internal_change": float(sum(cyclic_internal_changes) / max(len(cyclic_internal_changes), 1)) if cyclic_internal_changes else 0.0,
        "ar/cyclic_msg_norm": float(sum(cyclic_msg_norms) / max(len(cyclic_msg_norms), 1)) if cyclic_msg_norms else 0.0,
        "ar/cyclic_edge_weight_mean": float(sum(cyclic_edge_weight_means) / max(len(cyclic_edge_weight_means), 1)) if cyclic_edge_weight_means else 0.0,
        "ar/cyclic_edge_weight_max": float(max(cyclic_edge_weight_maxs)) if cyclic_edge_weight_maxs else 0.0,
        "ar/shadow_kg_loss": float(shadow_kg_loss.detach().cpu()),
        "ar/shadow_delta_loss": float(shadow_delta_loss.detach().cpu()),
        "ar/shadow_spec_loss": float(shadow_spec_loss.detach().cpu()),
        "ar/shadow_kg_weight": float(raw_kg_w),
        "ar/shadow_delta_weight": float(raw_delta_w),
        "ar/shadow_spec_weight": float(raw_spec_w),
        "ar/shadow_A_sigma": float(sum(shadow_A_sigmas) / max(len(shadow_A_sigmas), 1)) if shadow_A_sigmas else 0.0,
        "ar/shadow_delta_rel": float(sum(shadow_delta_rels) / max(len(shadow_delta_rels), 1)) if shadow_delta_rels else 0.0,
        "ar/shadow_force_rel": float(sum(shadow_force_rels) / max(len(shadow_force_rels), 1)) if shadow_force_rels else 0.0,
        "ar/shadow_beta_abs": float(sum(shadow_beta_abs_vals) / max(len(shadow_beta_abs_vals), 1)) if shadow_beta_abs_vals else 0.0,
        "ar/fixeda_kg_loss": float(fixeda_kg_loss.detach().cpu()),
        "ar/fixeda_corr_loss": float(fixeda_corr_loss.detach().cpu()),
        "ar/fixeda_mean_loss": float(fixeda_mean_loss.detach().cpu()),
        "ar/fixeda_delta_loss": float(fixeda_delta_loss.detach().cpu()),
        "ar/fixeda_kg_weight": float(fixeda_kg_w),
        "ar/fixeda_corr_weight": float(fixeda_corr_w),
        "ar/fixeda_mean_weight": float(fixeda_mean_w),
        "ar/fixeda_delta_weight": float(fixeda_delta_w),
        "ar/fixeda_kg_horizon": float(getattr(raw, "fixeda_kg_horizon", 0)),
        "ar/fixeda_cdeltaK": float(sum(fixeda_cdelta_vals) / max(len(fixeda_cdelta_vals), 1)) if fixeda_cdelta_vals else 0.0,
        "ar/fixeda_corr_explain": float(sum(fixeda_corr_explain_vals) / max(len(fixeda_corr_explain_vals), 1)) if fixeda_corr_explain_vals else 0.0,
        "ar/fixeda_delta_rel": float(sum(fixeda_delta_rels) / max(len(fixeda_delta_rels), 1)) if fixeda_delta_rels else 0.0,
        "ar/fixeda_delta_abs": float(sum(fixeda_delta_abs_vals) / max(len(fixeda_delta_abs_vals), 1)) if fixeda_delta_abs_vals else 0.0,
        "ar/fixeda_A_mean": float(sum(fixeda_A_means) / max(len(fixeda_A_means), 1)) if fixeda_A_means else 0.0,
        "ar/fixeda_A_max": float(max(fixeda_A_maxs)) if fixeda_A_maxs else 0.0,
        "ar/subspace_current_rec_loss": float(subspace_current_rec_loss.detach().cpu()),
        "ar/subspace_current_rec_weight": float(subspace_current_rec_w),
        "ar/subspace_neg_entropy_loss": float(subspace_neg_entropy_loss.detach().cpu()),
        "ar/subspace_entropy_weight": float(subspace_entropy_w),
        "ar/subspace_attn_entropy": float(sum(subspace_attn_entropies) / max(len(subspace_attn_entropies), 1)) if subspace_attn_entropies else 0.0,
        "ar/subspace_attn_max": float(sum(subspace_attn_max_vals) / max(len(subspace_attn_max_vals), 1)) if subspace_attn_max_vals else 0.0,
        "ar/subspace_z_norm": float(sum(subspace_z_norms) / max(len(subspace_z_norms), 1)) if subspace_z_norms else 0.0,
        "ar/subspace_z_next_norm": float(sum(subspace_z_next_norms) / max(len(subspace_z_next_norms), 1)) if subspace_z_next_norms else 0.0,
        "ar/pc_delta_loss": float(pc_delta_loss.detach().cpu()),
        "ar/pc_delta_weight": float(pc_delta_w),
        "ar/pc_correction_loss": float(pc_correction_loss.detach().cpu()),
        "ar/pc_correction_weight": float(pc_correction_w),
        "ar/pc_evidence_loss": float(pc_evidence_loss.detach().cpu()),
        "ar/pc_evidence_weight": float(pc_evidence_w),
        "ar/pc_prior_loss": float(pc_prior_loss.detach().cpu()),
        "ar/pc_prior_weight": float(pc_prior_w),
        "ar/pc_prior_smooth_loss": float(pc_prior_smooth_loss.detach().cpu()),
        "ar/pc_prior_smooth_weight": float(pc_prior_smooth_w),
        "ar/pc_correction_mag_loss": float(pc_corr_mag_loss.detach().cpu()),
        "ar/pc_correction_mag_weight": float(pc_corr_mag_w),
        "ar/pc_gate_target_loss": float(pc_gate_target_loss.detach().cpu()),
        "ar/pc_gate_target_weight": float(pc_gate_target_w),
        "ar/pc_gate_floor_loss": float(pc_gate_floor_loss.detach().cpu()),
        "ar/pc_gate_floor_weight": float(pc_gate_floor_w),
        "ar/pc_gate_mean": float(sum(pc_gate_means) / max(len(pc_gate_means), 1)) if pc_gate_means else 0.0,
        "ar/pc_gate_min": float(min(pc_gate_mins)) if pc_gate_mins else 0.0,
        "ar/pc_gate_max": float(max(pc_gate_maxs)) if pc_gate_maxs else 0.0,
        "ar/pc_error_norm": float(sum(pc_error_norms) / max(len(pc_error_norms), 1)) if pc_error_norms else 0.0,
        "ar/pc_correction_norm": float(sum(pc_correction_norms) / max(len(pc_correction_norms), 1)) if pc_correction_norms else 0.0,
        "ar/pc_target_correction_norm": float(sum(pc_target_corr_norms) / max(len(pc_target_corr_norms), 1)) if pc_target_corr_norms else 0.0,
        "ar/pc_correction_mag_ratio": float(sum(pc_corr_mag_ratios) / max(len(pc_corr_mag_ratios), 1)) if pc_corr_mag_ratios else 0.0,
        "ar/pc_correction_cos": float(sum(pc_correction_cos_vals) / max(len(pc_correction_cos_vals), 1)) if pc_correction_cos_vals else 0.0,
        "ar/atlas_rec_loss": float(atlas_rec_loss.detach().cpu()),
        "ar/atlas_rec_weight": float(atlas_rec_w),
        "ar/atlas_chart_inv_loss": float(atlas_chart_inv_loss.detach().cpu()),
        "ar/atlas_chart_inv_weight": float(atlas_chart_inv_w),
        "ar/atlas_dyn_loss": float(atlas_dyn_loss.detach().cpu()),
        "ar/atlas_dyn_weight": float(atlas_dyn_w),
        "ar/atlas_pred_chart_loss": float(atlas_pred_chart_loss.detach().cpu()),
        "ar/atlas_pred_chart_weight": float(atlas_pred_chart_w),
        "ar/atlas_overlap_loss": float(atlas_overlap_loss.detach().cpu()),
        "ar/atlas_overlap_weight": float(atlas_overlap_w),
        "ar/atlas_cocycle_loss": float(atlas_cocycle_loss.detach().cpu()),
        "ar/atlas_cocycle_weight": float(atlas_cocycle_w),
        "ar/atlas_compose_loss": float(atlas_compose_loss.detach().cpu()),
        "ar/atlas_compose_weight": float(atlas_compose_w),
        "ar/atlas_balance_loss": float(atlas_balance_loss.detach().cpu()),
        "ar/atlas_balance_weight": float(atlas_balance_w),
        "ar/atlas_noncollapse_loss": float(atlas_noncollapse_loss.detach().cpu()),
        "ar/atlas_noncollapse_weight": float(atlas_noncollapse_w),
        "ar/atlas_std_loss": float(atlas_std_loss.detach().cpu()),
        "ar/atlas_std_weight": float(atlas_std_w),
        "ar/atlas_std_min_ratio": float(getattr(args, "atlas_std_min_ratio", 0.20)),
        "ar/atlas_std_ratio": float(sum(atlas_std_ratios) / max(len(atlas_std_ratios), 1)) if atlas_std_ratios else 0.0,
        "ar/atlas_entropy_floor_loss": float(atlas_entropy_floor_loss.detach().cpu()),
        "ar/atlas_entropy_floor_weight": float(atlas_entropy_floor_w),
        "ar/atlas_entropy_min": float(getattr(args, "atlas_entropy_min", 1.0)),
        "ar/atlas_delta_cos_loss": float(atlas_delta_cos_loss.detach().cpu()),
        "ar/atlas_delta_cos_weight": float(atlas_delta_cos_w),
        "ar/atlas_delta_mse_loss": float(atlas_delta_mse_loss.detach().cpu()),
        "ar/atlas_delta_mse_weight": float(atlas_delta_mse_w),
        "ar/atlas_delta_norm_loss": float(atlas_delta_norm_loss.detach().cpu()),
        "ar/atlas_delta_norm_weight": float(atlas_delta_norm_w),
        "ar/atlas_delta_norm_min_ratio": float(getattr(args, "atlas_delta_norm_min_ratio", 0.25)),
        "ar/atlas_delta_scale_loss": float(atlas_delta_scale_loss.detach().cpu()),
        "ar/atlas_delta_scale_loss_weight": float(atlas_delta_scale_w),
        "ar/atlas_delta_cos": float(sum(atlas_delta_cos_vals) / max(len(atlas_delta_cos_vals), 1)) if atlas_delta_cos_vals else 0.0,
        "ar/atlas_delta_norm_ratio": float(sum(atlas_delta_norm_ratios) / max(len(atlas_delta_norm_ratios), 1)) if atlas_delta_norm_ratios else 0.0,
        "ar/atlas_delta_pred_std": float(sum(atlas_delta_pred_stds) / max(len(atlas_delta_pred_stds), 1)) if atlas_delta_pred_stds else 0.0,
        "ar/atlas_delta_gt_std": float(sum(atlas_delta_gt_stds) / max(len(atlas_delta_gt_stds), 1)) if atlas_delta_gt_stds else 0.0,
        "ar/atlas_delta_pred_norm": float(sum(atlas_delta_pred_norms) / max(len(atlas_delta_pred_norms), 1)) if atlas_delta_pred_norms else 0.0,
        "ar/atlas_delta_gt_norm": float(sum(atlas_delta_gt_norms) / max(len(atlas_delta_gt_norms), 1)) if atlas_delta_gt_norms else 0.0,
        "ar/atlas_hard_delta_scale_ratio": float(sum(atlas_hard_scale_ratios) / max(len(atlas_hard_scale_ratios), 1)) if atlas_hard_scale_ratios else 0.0,
        "ar/atlas_hard_delta_scale": float(sum(atlas_hard_scales) / max(len(atlas_hard_scales), 1)) if atlas_hard_scales else 0.0,
        "ar/atlas_hard_delta_raw_norm": float(sum(atlas_hard_raw_norms) / max(len(atlas_hard_raw_norms), 1)) if atlas_hard_raw_norms else 0.0,
        "ar/atlas_perturb_min_ratio": float(getattr(args, "atlas_perturb_min_ratio", 0.20)),
        "ar/atlas_pi_entropy": float(sum(atlas_pi_entropy_vals) / max(len(atlas_pi_entropy_vals), 1)) if atlas_pi_entropy_vals else 0.0,
        "ar/atlas_pi_next_entropy": float(sum(atlas_pi_next_entropy_vals) / max(len(atlas_pi_next_entropy_vals), 1)) if atlas_pi_next_entropy_vals else 0.0,
        "ar/atlas_perturb_ratio": float(sum(atlas_perturb_ratios) / max(len(atlas_perturb_ratios), 1)) if atlas_perturb_ratios else 0.0,
        "ar/atlas_target_z_norm": float(sum(atlas_target_z_norms) / max(len(atlas_target_z_norms), 1)) if atlas_target_z_norms else 0.0,
        "ar/atlas_z_next_norm": float(sum(atlas_z_next_norms) / max(len(atlas_z_next_norms), 1)) if atlas_z_next_norms else 0.0,
    }
    if bridge_extra_logs:
        logs.update(bridge_extra_logs)
    logs.update(cto_logs)
    if hasattr(raw, "dual_wiener_diagnostics"):
        logs.update(raw.dual_wiener_diagnostics(K_req))
    if getattr(raw, "global_horizon_wiener", None) is not None and hasattr(
        raw, "global_wiener_diagnostics"
    ):
        logs.update(raw.global_wiener_diagnostics(K_req))
    return loss, logs



def compute_stageb_joint_potential_ar_loss(model, state: torch.Tensor, stim: torch.Tensor | None, args, epoch: int | float = 0):
    """Stage B loss with a frozen coboundary potential model.

    Frozen Stage A defines
        v_t = V_A(x_t),  r_t = x_t - D_A(v_t),  s_t=[v_t,r_t].

    The Stage-B backbone does *not* predict the full x and is not allowed to
    use the residual branch to relearn the whole signal.  Its output is

        [delta_v_hat, r_hat_{t+1}],

    and the next recurrent state is

        v_hat_{t+1} = v_t + delta_v_hat,
        s_hat_{t+1} = [v_hat_{t+1}, r_hat_{t+1}],
        x_hat_{t+1} = D_A(v_hat_{t+1}) + r_hat_{t+1}.

    Training losses are primarily on the potential increment delta_v and the
    frozen residual target r.  The decoded x loss is optional and should usually
    be a diagnostic or very small auxiliary weight, because a large x-loss lets
    the residual branch compensate for potential errors and bypass Stage A.
    """
    from internal_dw.training.potential_stageb import (
        make_joint_state,
        split_joint,
        decode_joint,
        stageb_prediction_to_next_joint,
    )

    raw = unwrap_model(model)
    B, T, _ = get_batch_time_shape(state)
    W = int(args.window_size)
    if T <= W:
        raise ValueError(f"Sequence too short for Stage-B joint AR: T={T}, window={W}")
    if stim is None:
        stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))

    joint, stageb_logs = make_joint_state(state, args)  # [B,T,L+D]
    v_true, r_true = split_joint(joint, args)

    if raw.training:
        starts = _candidate_starts(T, W, int(getattr(args, "ar_train_stride", 1)))
        n_starts = max(1, min(int(getattr(args, "ar_train_starts_per_sequence", 1)), len(starts)))
        chosen = random.sample(starts, n_starts) if bool(getattr(args, "ar_train_random_starts", True)) else starts[:n_starts]
    else:
        chosen = _candidate_starts(T, W, int(getattr(args, "ar_eval_stride", 4)))

    loss_xs, loss_vs, loss_rs, loss_v_abs, corrs, rels = [], [], [], [], [], []
    for t in chosen:
        hist = time_window(joint, t-W, t)
        prev_joint = hist[:, -1]
        prev_v, _prev_r = split_joint(prev_joint, args)
        stim_window = time_window(stim, t-W, t)
        pred_raw = raw(stim_window, hist, return_aux=False)
        pred_frame = pred_raw[:, 0] if pred_raw.dim() == joint.dim() else pred_raw

        pred_next_joint, pred_delta_v, pred_v, pred_r = stageb_prediction_to_next_joint(pred_frame, prev_joint, args)
        target_v = v_true[:, t]
        target_delta_v = target_v - prev_v
        target_r = r_true[:, t]
        x_pred = decode_joint(pred_next_joint, args)
        x_tgt = state[:, t]

        lv = elementwise_state_loss(pred_delta_v, target_delta_v, loss="mse")
        lva = elementwise_state_loss(pred_v, target_v, loss="mse")
        lr = elementwise_state_loss(pred_r, target_r, loss=str(getattr(args, "ar_loss", "mse")))
        lx = elementwise_state_loss(x_pred.unsqueeze(1), x_tgt.unsqueeze(1), loss=str(getattr(args, "ar_loss", "mse")))
        loss_vs.append(lv); loss_v_abs.append(lva); loss_rs.append(lr); loss_xs.append(lx)
        corrs.append(corrcoef_flat(x_pred, x_tgt))
        rels.append(relative_l2(x_pred, x_tgt))

    loss_x = torch.stack(loss_xs).mean() if loss_xs else state.new_tensor(0.0)
    loss_v = torch.stack(loss_vs).mean() if loss_vs else state.new_tensor(0.0)
    loss_v_state = torch.stack(loss_v_abs).mean() if loss_v_abs else state.new_tensor(0.0)
    loss_r = torch.stack(loss_rs).mean() if loss_rs else state.new_tensor(0.0)
    # Diagnostic decoded-x one-step loss.  Do not confuse this with the actual
    # training objective when stageb_lambda_x=0.
    loss_one = loss_x

    bptt_loss = state.new_tensor(0.0)
    bptt_first_rel = state.new_tensor(0.0)
    bptt_last_rel = state.new_tensor(0.0)
    bptt_enabled = bool(getattr(args, "bptt_loss", False)) and (raw.training or bool(getattr(args, "bptt_eval", False)))
    bptt_lambda = float(getattr(args, "bptt_lambda", 0.0))
    if bptt_enabled and bptt_lambda != 0.0:
        K = max(1, int(getattr(args, "bptt_horizon", 8)))
        bls, first_rels, last_rels = [], [], []
        for t in chosen:
            max_k = min(K, T - t)
            if max_k <= 0:
                continue
            hist = time_window(joint, t-W, t).clone()
            step_losses, step_rels = [], []
            for k in range(max_k):
                target_t = t + k
                stim_window = time_window(stim, target_t-W, target_t) if target_t-W >= 0 else time_window(stim, 0, W)
                prev_joint = hist[:, -1]
                pred_raw = raw(stim_window, hist, return_aux=False)
                pred_frame = pred_raw[:, 0] if pred_raw.dim() == joint.dim() else pred_raw
                pred_next_joint, _pred_delta_v, _pred_v, _pred_r = stageb_prediction_to_next_joint(pred_frame, prev_joint, args)
                x_pred = decode_joint(pred_next_joint, args)
                x_tgt = state[:, target_t]
                step_losses.append(elementwise_state_loss(x_pred.unsqueeze(1), x_tgt.unsqueeze(1), loss=str(getattr(args, "bptt_loss_type", getattr(args, "ar_loss", "mse")))))
                step_rels.append(relative_l2(x_pred, x_tgt))
                hist = append_time_point(hist, pred_next_joint)
            if step_losses:
                bls.append(torch.stack(step_losses).mean())
                first_rels.append(step_rels[0])
                last_rels.append(step_rels[-1])
        if bls:
            bptt_loss = torch.stack(bls).mean()
            bptt_first_rel = torch.stack(first_rels).mean()
            bptt_last_rel = torch.stack(last_rels).mean()

    lam_x = float(getattr(args, "stageb_lambda_x", 0.0))
    lam_v = float(getattr(args, "stageb_lambda_v", 1.0))
    lam_r = float(getattr(args, "stageb_lambda_r", 1.0))
    one_step_lambda = float(getattr(args, "ar_one_step_lambda", 1.0))
    loss_stageb = lam_x * loss_x + lam_v * loss_v + lam_r * loss_r
    loss = one_step_lambda * loss_stageb + bptt_lambda * bptt_loss

    eps_log = state.new_tensor(1e-8)
    logs = {
        "loss": float(loss.detach()),
        "ar/loss_one_step": float(loss_one.detach()),
        "ar/one_step_lambda": float(one_step_lambda),
        "ar/loss_one_step_weighted": float((one_step_lambda * loss_stageb).detach()),
        "ar/one_step_corr": float(torch.stack(corrs).mean().detach()) if corrs else 0.0,
        "ar/one_step_rel_l2": float(torch.stack(rels).mean().detach()) if rels else 0.0,
        "ar/stageb_joint_enabled": 1.0,
        "ar/stageb_predict_delta_v": 1.0 if bool(getattr(args, "stageb_predict_delta_v", True)) else 0.0,
        "ar/stageb_loss_x": float(loss_x.detach()),
        "ar/stageb_loss_v_delta": float(loss_v.detach()),
        "ar/stageb_loss_v_state": float(loss_v_state.detach()),
        "ar/stageb_loss_r": float(loss_r.detach()),
        "ar/stageb_lambda_x": lam_x,
        "ar/stageb_lambda_v": lam_v,
        "ar/stageb_lambda_r": lam_r,
        "ar/bptt_loss": float(bptt_loss.detach()),
        "ar/bptt_weighted_loss": float((bptt_lambda * bptt_loss).detach()),
        "ar/bptt_lambda": bptt_lambda,
        "ar/bptt_horizon": float(getattr(args, "bptt_horizon", 0)),
        "ar/bptt_first_rel_l2": float(bptt_first_rel.detach()),
        "ar/bptt_last_rel_l2": float(bptt_last_rel.detach()),
        "ar/bptt_pct_of_one_step": float((100.0 * bptt_lambda * bptt_loss / loss_stageb.detach().clamp_min(eps_log)).detach()),
        "ar/num_train_starts": float(len(chosen)),
        "ar/window_size": float(W),
    }
    logs.update(stageb_logs)
    return loss, logs

def _stageb_stim_context(stim: torch.Tensor, target_t: int, args) -> torch.Tensor:
    from internal_dw.training.potential_stageb import stageb_stim_context_window
    return stageb_stim_context_window(stim, target_t, int(getattr(args, "potential_stim_context_len", 1)))


def compute_stageb_stim_potential_residual_ar_loss(model, state: torch.Tensor, stim: torch.Tensor | None, args, epoch: int | float = 0):
    """Stage B residual AR on top of frozen stimulus-driven potential Stage A.

    Frozen Stage A defines
        v_t = V(x_t),       v_{t+1} = v_t + G(u_t),       x^Phi_t = D(v_t).

    Stage B model input is the full reconstructed/rolled signal history x, not
    separated v/r.  Its output is only the residual r_{t+1}.  The next rollout
    state is
        xhat_{t+1} = D(v_t + G(u_t)) + rhat_{t+1}.
    """
    from internal_dw.training.potential_stageb import (
        get_stageb_potential_model,
        stimulus_potential_decomposition_logs,
    )

    raw = unwrap_model(model)
    B, T, _ = get_batch_time_shape(state)
    W = int(args.window_size)
    if T <= W:
        raise ValueError(f"Sequence too short for Stage-B stimulus-potential residual AR: T={T}, window={W}")
    if stim is None:
        stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))

    pot = get_stageb_potential_model(args, state.device)
    with torch.no_grad():
        v_all = pot.encode(state).detach()
        stageb_logs = stimulus_potential_decomposition_logs(state, args)

    if raw.training:
        starts = _candidate_starts(T, W, int(getattr(args, "ar_train_stride", 1)))
        n_starts = max(1, min(int(getattr(args, "ar_train_starts_per_sequence", 1)), len(starts)))
        chosen = random.sample(starts, n_starts) if bool(getattr(args, "ar_train_random_starts", True)) else starts[:n_starts]
    else:
        chosen = _candidate_starts(T, W, int(getattr(args, "ar_eval_stride", 4)))

    loss_rs, loss_xs, corrs, rels = [], [], [], []
    for t in chosen:
        hist_x = time_window(state, t-W, t)
        stim_window = time_window(stim, t-W, t)
        pred_raw = raw(stim_window, hist_x, return_aux=False)
        pred_r = pred_raw[:, 0] if pred_raw.dim() == state.dim() else pred_raw
        with torch.no_grad():
            v_prev = v_all[:, t-1]
            dv = pot.stim_delta_context_window(_stageb_stim_context(stim, t, args), like_v=v_prev)
            v_next = v_prev + dv
            x_phi_next = pot.decode(v_next).detach()
            target_r = state[:, t] - x_phi_next
        x_pred = x_phi_next + pred_r
        loss_r = elementwise_state_loss(pred_r.unsqueeze(1), target_r.unsqueeze(1), loss=str(getattr(args, "ar_loss", "mse")))
        loss_x = elementwise_state_loss(x_pred.unsqueeze(1), state[:, t].unsqueeze(1), loss=str(getattr(args, "ar_loss", "mse")))
        loss_rs.append(loss_r); loss_xs.append(loss_x)
        corrs.append(corrcoef_flat(x_pred, state[:, t]))
        rels.append(relative_l2(x_pred, state[:, t]))

    loss_r = torch.stack(loss_rs).mean() if loss_rs else state.new_tensor(0.0)
    loss_x = torch.stack(loss_xs).mean() if loss_xs else state.new_tensor(0.0)
    loss_one = loss_x

    bptt_loss = state.new_tensor(0.0)
    bptt_first_rel = state.new_tensor(0.0)
    bptt_last_rel = state.new_tensor(0.0)
    bptt_enabled = bool(getattr(args, "bptt_loss", False)) and (raw.training or bool(getattr(args, "bptt_eval", False)))
    bptt_lambda = float(getattr(args, "bptt_lambda", 0.0))
    if bptt_enabled and bptt_lambda != 0.0:
        K = max(1, int(getattr(args, "bptt_horizon", 8)))
        bls, first_rels, last_rels = [], [], []
        for t in chosen:
            max_k = min(K, T - t)
            if max_k <= 0:
                continue
            hist_x = time_window(state, t-W, t).clone()
            with torch.no_grad():
                v_prev = v_all[:, t-1]
            step_losses, step_rels = [], []
            for k in range(max_k):
                target_t = t + k
                stim_window = time_window(stim, target_t-W, target_t)
                pred_raw = raw(stim_window, hist_x, return_aux=False)
                pred_r = pred_raw[:, 0] if pred_raw.dim() == state.dim() else pred_raw
                with torch.no_grad():
                    dv = pot.stim_delta_context_window(_stageb_stim_context(stim, target_t, args), like_v=v_prev)
                    v_next = v_prev + dv
                    x_phi_next = pot.decode(v_next).detach()
                    target_r = state[:, target_t] - x_phi_next
                x_pred = x_phi_next + pred_r
                step_losses.append(elementwise_state_loss(pred_r.unsqueeze(1), target_r.unsqueeze(1), loss=str(getattr(args, "bptt_loss_type", getattr(args, "ar_loss", "mse")))))
                step_rels.append(relative_l2(x_pred, state[:, target_t]))
                hist_x = append_time_point(hist_x, x_pred)
                v_prev = v_next.detach()
            if step_losses:
                bls.append(torch.stack(step_losses).mean())
                first_rels.append(step_rels[0])
                last_rels.append(step_rels[-1])
        if bls:
            bptt_loss = torch.stack(bls).mean()
            bptt_first_rel = torch.stack(first_rels).mean()
            bptt_last_rel = torch.stack(last_rels).mean()

    lam_r = float(getattr(args, "stageb_lambda_r", 1.0))
    lam_x = float(getattr(args, "stageb_lambda_x", 0.0))
    one_step_lambda = float(getattr(args, "ar_one_step_lambda", 1.0))
    loss_stageb = lam_r * loss_r + lam_x * loss_x
    loss = one_step_lambda * loss_stageb + bptt_lambda * bptt_loss
    eps_log = state.new_tensor(1e-8)
    logs = {
        "loss": float(loss.detach()),
        "ar/loss_one_step": float(loss_one.detach()),
        "ar/one_step_lambda": float(one_step_lambda),
        "ar/loss_one_step_weighted": float((one_step_lambda * loss_stageb).detach()),
        "ar/one_step_corr": float(torch.stack(corrs).mean().detach()) if corrs else 0.0,
        "ar/one_step_rel_l2": float(torch.stack(rels).mean().detach()) if rels else 0.0,
        "ar/stageb_stim_potential_residual_enabled": 1.0,
        "ar/stageb_stim_context_len": float(getattr(args, "potential_stim_context_len", 1)),
        "ar/stageb_loss_x": float(loss_x.detach()),
        "ar/stageb_loss_r": float(loss_r.detach()),
        "ar/stageb_lambda_x": lam_x,
        "ar/stageb_lambda_r": lam_r,
        "ar/bptt_loss": float(bptt_loss.detach()),
        "ar/bptt_weighted_loss": float((bptt_lambda * bptt_loss).detach()),
        "ar/bptt_lambda": bptt_lambda,
        "ar/bptt_horizon": float(getattr(args, "bptt_horizon", 0)),
        "ar/bptt_first_rel_l2": float(bptt_first_rel.detach()),
        "ar/bptt_last_rel_l2": float(bptt_last_rel.detach()),
        "ar/bptt_pct_of_one_step": float((100.0 * bptt_lambda * bptt_loss / loss_stageb.detach().clamp_min(eps_log)).detach()),
        "ar/num_train_starts": float(len(chosen)),
        "ar/window_size": float(W),
    }
    logs.update(stageb_logs)
    return loss, logs


def compute_autoregressive_one_step_loss(
    model,
    state,
    stim,
    args,
    epoch: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Teacher-forced next-step training for autoregressive field models.

    This is the standard FNO-style training objective. It samples only a small
    number of target times from each full trajectory, avoiding the old KG loop
    that retained many FFT graphs before backward.

    If fno_folded_loss is enabled, the returned loss is:

        loss = loss_one_step + fno_fold_lambda * loss_fold

    Logs report one-step, raw folded, weighted folded, and total losses
    separately.
    """
    raw = unwrap_model(model)
    if bool(getattr(args, "stageb_stim_potential_residual_ar", False)):
        return compute_stageb_stim_potential_residual_ar_loss(model, state, stim, args, epoch=epoch)
    if bool(getattr(args, "stageb_joint_potential_ar", False)):
        return compute_stageb_joint_potential_ar_loss(model, state, stim, args, epoch=epoch)
    if bool(getattr(raw, "is_recurrent_state_ar", False)):
        return compute_recurrent_state_bptt_loss(model, state, stim, args, epoch=epoch)
    if bool(getattr(raw, "is_clock_latent_ae", False)):
        return compute_clock_latent_ae_loss(model, state, stim, args, epoch=epoch)
    if bool(getattr(raw, "is_state_sequence_ae", False)):
        return compute_state_sequence_ae_loss(model, state, stim, args, epoch=epoch)
    if bool(getattr(raw, "is_path_generator", False)):
        return compute_path_generator_loss(model, state, stim, args, epoch=epoch)

    B, T, _ = get_batch_time_shape(state)
    W = int(args.window_size)

    if T <= W:
        raise ValueError(
            f"Sequence too short for one-step AR training: T={T}, window={W}"
        )

    stageb_logs = {}
    if bool(getattr(args, "stageb_potential_residual", False)):
        # Freeze Stage-A coboundary potential AE and train this AR model only
        # on residual dynamics r_t = x_t - D_A(E_A(x_t)).
        state, stageb_logs = stageb_residualize_state(state, args)

    if stim is None:
        stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))

    if raw.training:
        starts = _candidate_starts(T, W, int(getattr(args, "ar_train_stride", 1)))
        n_starts = int(getattr(args, "ar_train_starts_per_sequence", 1))
        n_starts = max(1, min(n_starts, len(starts)))
        controller = getattr(raw, "global_horizon_wiener", None)
        batch_conditioned_global = bool(
            controller is not None
            and getattr(controller, "batch_conditioned", False)
        )
        shared_rollout_start = bool(
            batch_conditioned_global
            or getattr(args, "ar_shared_rollout_start", False)
        )
        if shared_rollout_start:
            n_starts = 1

        # Random windows are standard during training unless explicitly disabled.
        use_random = bool(getattr(args, "ar_train_random_starts", True))
        if shared_rollout_start:
            chosen = [
                _shared_global_wiener_start(
                    starts,
                    randomize=use_random,
                    epoch=int(epoch),
                    device=state.device,
                )
            ]
        elif use_random:
            chosen = random.sample(starts, n_starts)
        else:
            offset = max(int(epoch) - 1, 0) % max(1, len(starts))
            chosen = [starts[(offset + i) % len(starts)] for i in range(n_starts)]
    else:
        # Validation/test loss should cover the trajectory deterministically.
        # The previous implementation used only the first target time because
        # evaluate_loss() calls this function with epoch=0 and n_starts=1.
        # This made best.pth selection depend on a single one-step target.
        eval_stride = int(getattr(args, "ar_eval_stride", 4))
        chosen = _candidate_starts(T, W, eval_stride)

    one_losses = []
    fold_losses = []
    ftg_amp_losses = []
    ftg_bound_losses = []
    comp_losses = []
    etm_fit_losses = []
    etm_prop_losses = []
    frontier_losses = []
    bptt_losses = []
    long_cloud_losses = []
    forced_damped_losses = []
    source_color_losses = []
    error_proj_pseudo_losses = []
    error_proj_cob_losses = []
    error_proj_v2_losses = []
    error_align_tf_losses = []
    error_align_losses = []
    field_peh_losses = []
    corrs = []
    rels = []

    fold_log_accum = {
        "ar/fold_loss": 0.0,
        "ar/fold_used_horizon": 0.0,
        "ar/fold_A_sigma": 0.0,
        "ar/fold_lambda": 0.0,
        "ar/fold_weighted_loss": 0.0,
    }
    n_fold_logs = 0

    ftg_log_accum = {
        "ar/ftg_amp_ratio": 0.0,
        "ar/ftg_amp_loss": 0.0,
        "ar/ftg_bound_loss": 0.0,
        "ar/ftg_lambda_amp": 0.0,
        "ar/ftg_lambda_bound": 0.0,
        "ar/ftg_horizon": 0.0,
        "ar/ftg_sigma_hat": 0.0,
        "ar/ftg_ratio_k0": 0.0,
        "ar/ftg_ratio_k1": 0.0,
        "ar/ftg_ratio_k2": 0.0,
        "ar/ftg_ratio_klast": 0.0,
    }
    n_ftg_logs = 0

    comp_log_accum = {
        "ar/comp_loss": 0.0,
        "ar/comp_horizon": 0.0,
        "ar/comp_block_size": 0.0,
        "ar/comp_blocks": 0.0,
        "ar/comp_injection_steps": 0.0,
        "ar/comp_mean_error_norm": 0.0,
        "ar/comp_mean_pseudo_norm": 0.0,
        "ar/comp_proxy_dot": 0.0,
        "ar/comp_first_rel_l2": 0.0,
        "ar/comp_last_rel_l2": 0.0,
        "ar/comp_lambda": 0.0,
    }
    n_comp_logs = 0

    etm_log_accum = _etm_zero_logs()
    n_etm_logs = 0

    frontier_log_accum = {
        "ar/frontier_loss": 0.0,
        "ar/frontier_horizon": 0.0,
        "ar/frontier_tau": 0.0,
        "ar/frontier_injection_steps": 0.0,
        "ar/frontier_keep_frac": 0.0,
        "ar/frontier_keep_frac_min": 0.0,
        "ar/frontier_keep_frac_max": 0.0,
        "ar/frontier_score_mean": 0.0,
        "ar/frontier_survival_norm_ratio": 0.0,
        "ar/frontier_mean_error_norm": 0.0,
        "ar/frontier_mean_pseudo_norm": 0.0,
        "ar/frontier_proxy_dot": 0.0,
        "ar/frontier_first_rel_l2": 0.0,
        "ar/frontier_last_rel_l2": 0.0,
        "ar/frontier_lambda": 0.0,
    }
    n_frontier_logs = 0

    bptt_log_accum = {
        "ar/bptt_loss": 0.0,
        "ar/bptt_horizon": 0.0,
        "ar/bptt_first_rel_l2": 0.0,
        "ar/bptt_last_rel_l2": 0.0,
        "ar/bptt_lambda": 0.0,
    }
    n_bptt_logs = 0

    long_cloud_log_accum = _long_error_cloud_zero_logs()
    n_long_cloud_logs = 0

    forced_damped_log_accum = _forced_damped_zero_logs()
    n_forced_damped_logs = 0

    transport_coh_losses = []
    transport_coh_log_accum = _transport_coh_zero_logs()
    n_transport_coh_logs = 0

    source_color_log_accum = _source_color_zero_logs()
    n_source_color_logs = 0

    error_proj_pseudo_log_accum = _error_proj_pseudo_zero_logs()
    n_error_proj_pseudo_logs = 0

    error_proj_cob_log_accum = _error_proj_coboundary_zero_logs()
    n_error_proj_cob_logs = 0

    error_proj_v2_log_accum = _error_proj_v2_zero_logs()
    n_error_proj_v2_logs = 0

    error_align_log_accum = _error_align_zero_logs()
    n_error_align_logs = 0

    field_peh_log_accum = _field_peh_zero_logs()
    n_field_peh_logs = 0

    return_aux = bool(getattr(args, "fno_folded_loss", False))

    for t in chosen:
        history = time_window(state, t - W, t)
        stim_window = time_window(stim, t - W, t)
        target = time_point(state, t, keep_time=True)

        if return_aux:
            pred, aux = raw(stim_window, history, return_aux=True)
        else:
            pred = raw(stim_window, history, return_aux=False)
            aux = {}

        loss_name = str(getattr(args, "ar_loss", "rel_l2"))

        if loss_name == "rel_l2":
            one_loss = relative_l2(pred[:, 0], target[:, 0])
        else:
            one_loss = elementwise_state_loss(pred, target, loss=loss_name)

        one_losses.append(one_loss)

        if return_aux:
            fold_loss, fold_logs = _compute_fno_folded_loss(
                raw,
                aux["folded_h_pred"],
                state,
                t,
                args,
            )
            fold_losses.append(fold_loss)

            for k, v in fold_logs.items():
                fold_log_accum[k] = fold_log_accum.get(k, 0.0) + float(v)
            n_fold_logs += 1

        if bool(getattr(args, "fno_ftg_loss", False)) and hasattr(raw, "step_history"):
            ftg_amp_loss, ftg_bound_loss, ftg_logs = _compute_fno_ftg_losses(
                raw=raw,
                history=history,
                pred_frame=pred[:, 0],
                target_frame=target[:, 0],
                one_loss=one_loss,
                args=args,
            )
            ftg_amp_losses.append(ftg_amp_loss)
            ftg_bound_losses.append(ftg_bound_loss)
            for k, v in ftg_logs.items():
                ftg_log_accum[k] = ftg_log_accum.get(k, 0.0) + float(v)
            n_ftg_logs += 1

        comp_enabled = bool(getattr(args, "comp_graph_loss", False)) and (raw.training or bool(getattr(args, "comp_eval", False)))
        comp_lambda = _effective_comp_lambda(args, epoch=epoch) if raw.training else float(getattr(args, "comp_lambda", 0.0))
        if comp_enabled and comp_lambda != 0.0:
            comp_loss, comp_logs = compute_compiled_backward_graph_loss(raw, state, stim, t, args)
            comp_losses.append(comp_loss)
            for k, v in comp_logs.items():
                comp_log_accum[k] = comp_log_accum.get(k, 0.0) + float(v)
            n_comp_logs += 1

        etm_lam_fit, etm_lam_prop = _effective_etm_lambdas(args, epoch=epoch) if raw.training else (float(getattr(args, "etm_lambda_fit", 0.0)), float(getattr(args, "etm_lambda_prop", 0.0)))
        etm_enabled = bool(getattr(args, "etm_loss", False)) and (raw.training or bool(getattr(args, "etm_eval", False)))
        if etm_enabled and (etm_lam_fit != 0.0 or etm_lam_prop != 0.0):
            # Let compute_error_transport_metric_loss log the current ETM stage
            # without changing its public call signature.
            setattr(args, "_current_epoch_for_logs", float(epoch or 0))
            etm_fit, etm_prop, etm_logs = compute_error_transport_metric_loss(
                raw, state, stim, t, args, pred_first=pred[:, 0], target_first=target
            )
            etm_fit_losses.append(etm_fit)
            etm_prop_losses.append(etm_prop)
            for k, v in etm_logs.items():
                etm_log_accum[k] = etm_log_accum.get(k, 0.0) + float(v)
            n_etm_logs += 1

        # Strict live-gradient-frontier BPTT is implemented as a manual backward
        # pass in trainer.py after loss_one.backward().  Do not build an old
        # scalar/pseudo frontier loss here; that would keep multiple local graphs
        # alive until the final backward and is not the method we want.
        frontier_enabled = False
        frontier_lambda_now = 0.0

        bptt_enabled = bool(getattr(args, "bptt_loss", False)) and (raw.training or bool(getattr(args, "bptt_eval", False)))
        bptt_lambda = float(getattr(args, "bptt_lambda", 0.0))
        if bptt_enabled and bptt_lambda != 0.0:
            bptt_loss, bptt_logs = compute_full_bptt_rollout_loss(raw, state, stim, t, args)
            bptt_losses.append(bptt_loss)
            for k, v in bptt_logs.items():
                bptt_log_accum[k] = bptt_log_accum.get(k, 0.0) + float(v)
            bptt_log_accum["ar/bptt_lambda"] += float(bptt_lambda)
            n_bptt_logs += 1

        # Field PEH: short-BPTT + free-rollout error homogenization.
        # This is the The Well / Gray-Scott transfer of the PEH idea.
        field_peh_lambda_now = _field_peh_effective_lambda(args, epoch) if raw.training else float(getattr(args, "field_peh_lambda", 0.0))
        field_peh_enabled = bool(getattr(args, "field_peh_loss", False)) and (raw.training or bool(getattr(args, "field_peh_eval", False)))
        if field_peh_enabled and (field_peh_lambda_now != 0.0 or bool(getattr(args, "field_peh_eval", False))):
            fp_loss, fp_logs = compute_field_peh_rollout_loss(raw, state, stim, t, args, epoch=epoch)
            field_peh_losses.append(fp_loss)
            for k, v in fp_logs.items():
                field_peh_log_accum[k] = field_peh_log_accum.get(k, 0.0) + float(v)
            n_field_peh_logs += 1

        # Detached long-horizon error-cloud regularizer.  This is deliberately
        # separated from the short BPTT loss: the long AR rollout is no-grad,
        # while gradients flow only through local TF residuals at long offsets.
        long_cloud_enabled = bool(getattr(args, "long_error_cloud_loss", False)) and (raw.training or bool(getattr(args, "long_error_cloud_eval", False)))
        long_cloud_nonzero = (
            float(getattr(args, "long_error_cloud_rein_lambda", 0.0)) != 0.0
            or float(getattr(args, "long_error_cloud_energy_lambda", 0.0)) != 0.0
            or float(getattr(args, "long_error_cloud_orth_lambda", 0.0)) != 0.0
            or float(getattr(args, "long_error_cloud_mean_lambda", 0.0)) != 0.0
            or bool(getattr(args, "long_error_cloud_eval", False))
        )
        if long_cloud_enabled and long_cloud_nonzero:
            long_cloud_loss, long_cloud_logs = compute_long_error_cloud_loss(raw, state, stim, t, args)
            long_cloud_losses.append(long_cloud_loss)
            for k, v in long_cloud_logs.items():
                long_cloud_log_accum[k] = long_cloud_log_accum.get(k, 0.0) + float(v)
            n_long_cloud_logs += 1

        # Forced Damped Error Dynamics (FDED).  The long AR probe is no-grad,
        # while fresh one-step predictions from detached AR histories provide
        # local gradients on the rollout-state distribution.
        forced_damped_enabled = bool(getattr(args, "forced_damped_error_loss", False)) and (raw.training or bool(getattr(args, "forced_damped_eval", False)))
        forced_damped_nonzero = (
            float(getattr(args, "forced_damped_energy_lambda", 0.0)) != 0.0
            or float(getattr(args, "forced_damped_damping_lambda", 0.0)) != 0.0
            or float(getattr(args, "forced_damped_accel_lambda", 0.0)) != 0.0
            or bool(getattr(args, "forced_damped_eval", False))
        )
        if forced_damped_enabled and forced_damped_nonzero:
            fded_loss, fded_logs = compute_forced_damped_error_dynamics_loss(raw, state, stim, t, args)
            forced_damped_losses.append(fded_loss)
            for k, v in fded_logs.items():
                forced_damped_log_accum[k] = forced_damped_log_accum.get(k, 0.0) + float(v)
            n_forced_damped_logs += 1

        # Transported error-source coherence regularizer.  This uses JVPs to
        # propagate clean-history local residuals b_i to a shared horizon, then
        # softly penalizes positive pairwise coherence of important transported
        # directions.
        transport_coh_enabled = bool(getattr(args, "transport_coh_loss", False)) and (raw.training or bool(getattr(args, "transport_coh_eval", False)))
        transport_coh_nonzero = (
            float(getattr(args, "transport_coh_adj_lambda", 0.0)) != 0.0
            or float(getattr(args, "transport_coh_multi_lambda", 0.0)) != 0.0
            or float(getattr(args, "transport_coh_proj_lambda", 0.0)) != 0.0
            or float(getattr(args, "transport_coh_proj_mean_lambda", 0.0)) != 0.0
            or bool(getattr(args, "transport_coh_eval", False))
        )
        if transport_coh_enabled and transport_coh_nonzero:
            tc_loss, tc_logs = compute_transport_coherence_loss(raw, state, stim, t, args)
            transport_coh_losses.append(tc_loss)
            for k, v in tc_logs.items():
                transport_coh_log_accum[k] = transport_coh_log_accum.get(k, 0.0) + float(v)
            n_transport_coh_logs += 1

        source_color_enabled = bool(getattr(args, "source_color_loss", False)) and (raw.training or bool(getattr(args, "source_color_eval", False)))
        source_color_lambda_now = float(getattr(args, "source_color_lambda", 0.0))
        if source_color_enabled and source_color_lambda_now != 0.0:
            sc_loss, sc_logs = compute_source_colored_local_repair_loss(raw, state, stim, t, args)
            source_color_losses.append(sc_loss)
            for k, v in sc_logs.items():
                source_color_log_accum[k] = source_color_log_accum.get(k, 0.0) + float(v)
            n_source_color_logs += 1

        # Projection-target distillation from detached rollout error geometry.
        # This turns the oracle anti-reinforcement projection into an explicit
        # local pseudo-target.  The long rollout that defines the target is
        # no-grad; gradients flow only through fresh one-step predictions.
        error_proj_pseudo_enabled = bool(getattr(args, "error_proj_pseudo_loss", False)) and (raw.training or bool(getattr(args, "error_proj_pseudo_eval", False)))
        error_proj_pseudo_lambda_now = float(getattr(args, "error_proj_pseudo_lambda", 0.0))
        if error_proj_pseudo_enabled and (error_proj_pseudo_lambda_now != 0.0 or bool(getattr(args, "error_proj_pseudo_eval", False))):
            ep_loss, ep_logs = compute_error_projection_pseudo_loss(raw, state, stim, t, args)
            error_proj_pseudo_losses.append(ep_loss)
            for k, v in ep_logs.items():
                error_proj_pseudo_log_accum[k] = error_proj_pseudo_log_accum.get(k, 0.0) + float(v)
            n_error_proj_pseudo_logs += 1

        # V3 coboundary-filtered Poseido.  This keeps the V1 accumulated-error
        # correction direction but subtracts a fitted potential-difference term
        # from the local reinforcement scalar before constructing the pseudo target.
        error_proj_cob_enabled = bool(getattr(args, "error_proj_cob_loss", False)) and (raw.training or bool(getattr(args, "error_proj_cob_eval", False)))
        error_proj_cob_lambda_now = float(getattr(args, "error_proj_cob_lambda", 0.0))
        if error_proj_cob_enabled and (error_proj_cob_lambda_now != 0.0 or bool(getattr(args, "error_proj_cob_eval", False))):
            epc_loss, epc_logs = compute_error_projection_coboundary_loss(raw, state, stim, t, args)
            error_proj_cob_losses.append(epc_loss)
            for k, v in epc_logs.items():
                error_proj_cob_log_accum[k] = error_proj_cob_log_accum.get(k, 0.0) + float(v)
            n_error_proj_cob_logs += 1

        # V2 adjacent transported local innovation loss.  This is separate from
        # V1 projection pseudo-target loss and can be enabled independently.
        error_proj_v2_enabled = bool(getattr(args, "error_proj_v2_loss", False)) and (raw.training or bool(getattr(args, "error_proj_v2_eval", False)))
        error_proj_v2_lambda_now = float(getattr(args, "error_proj_v2_lambda", 0.0))
        if error_proj_v2_enabled and (error_proj_v2_lambda_now != 0.0 or bool(getattr(args, "error_proj_v2_eval", False))):
            ep2_loss, ep2_logs = compute_error_projection_v2_loss(raw, state, stim, t, args)
            error_proj_v2_losses.append(ep2_loss)
            for k, v in ep2_logs.items():
                error_proj_v2_log_accum[k] = error_proj_v2_log_accum.get(k, 0.0) + float(v)
            n_error_proj_v2_logs += 1

        # Long no-grad AR context + one-step TF alignment.
        # This is intentionally NOT short-BPTT and NOT long-BPTT: p_t is
        # detached from a long AR rollout, while b_t is a trainable TF error.
        error_align_enabled = bool(getattr(args, "error_align_loss", False)) and (raw.training or bool(getattr(args, "error_align_eval", False)))
        error_align_lambda_now = float(getattr(args, "error_align_lambda", 0.0))
        error_align_tf_lambda_now = float(getattr(args, "error_align_tf_lambda", 1.0))
        if error_align_enabled and (error_align_lambda_now != 0.0 or error_align_tf_lambda_now != 0.0):
            ea_tf_loss, ea_align_loss, ea_logs = compute_error_alignment_tf_loss(raw, state, stim, t, args)
            error_align_tf_losses.append(ea_tf_loss)
            error_align_losses.append(ea_align_loss)
            for k, v in ea_logs.items():
                error_align_log_accum[k] = error_align_log_accum.get(k, 0.0) + float(v)
            n_error_align_logs += 1

        with torch.no_grad():
            corrs.append(corrcoef_flat(pred[:, 0], target[:, 0]))
            rels.append(relative_l2(pred[:, 0], target[:, 0]))

    loss_one = torch.stack(one_losses).mean()

    if fold_losses:
        loss_fold = torch.stack(fold_losses).mean()
        fold_lambda = float(getattr(args, "fno_fold_lambda", -1.0))
        if fold_lambda < 0:
            fold_lambda = float(getattr(args, "koopman_lambda_gram", 0.0))
    else:
        loss_fold = loss_one.new_tensor(0.0)
        fold_lambda = 0.0

    loss_fold_weighted = fold_lambda * loss_fold

    if ftg_amp_losses:
        loss_ftg_amp = torch.stack(ftg_amp_losses).mean()
        loss_ftg_bound = torch.stack(ftg_bound_losses).mean()
        ftg_lambda_amp = float(getattr(args, "fno_ftg_lambda_amp", 0.0))
        ftg_lambda_bound = float(getattr(args, "fno_ftg_lambda_bound", 0.0))
    else:
        loss_ftg_amp = loss_one.new_tensor(0.0)
        loss_ftg_bound = loss_one.new_tensor(0.0)
        ftg_lambda_amp = 0.0
        ftg_lambda_bound = 0.0

    if comp_losses:
        loss_comp = torch.stack(comp_losses).mean()
        # Use the scheduled/ramped lambda for the actual gradient weight.
        # The previous v2_fix used the base lambda here, so --comp_start_epoch
        # and --comp_ramp_epochs only controlled whether the compiled loss was
        # computed, but did not ramp its strength once enabled.
        comp_lambda = _effective_comp_lambda(args, epoch=epoch) if raw.training else float(getattr(args, "comp_lambda", 0.0))
    else:
        loss_comp = loss_one.new_tensor(0.0)
        comp_lambda = 0.0

    if etm_fit_losses:
        loss_etm_fit = torch.stack(etm_fit_losses).mean()
        loss_etm_prop = torch.stack(etm_prop_losses).mean()
        etm_lambda_fit, etm_lambda_prop = _effective_etm_lambdas(args, epoch=epoch) if raw.training else (float(getattr(args, "etm_lambda_fit", 0.0)), float(getattr(args, "etm_lambda_prop", 0.0)))
    else:
        loss_etm_fit = loss_one.new_tensor(0.0)
        loss_etm_prop = loss_one.new_tensor(0.0)
        etm_lambda_fit = 0.0
        etm_lambda_prop = 0.0

    if frontier_losses:
        loss_frontier = torch.stack(frontier_losses).mean()
        frontier_lambda = _effective_frontier_lambda(args, epoch=epoch) if raw.training else float(getattr(args, "frontier_lambda", 0.0))
    else:
        loss_frontier = loss_one.new_tensor(0.0)
        frontier_lambda = 0.0

    if bptt_losses:
        loss_bptt = torch.stack(bptt_losses).mean()
        bptt_lambda = float(getattr(args, "bptt_lambda", 0.0))
    else:
        loss_bptt = loss_one.new_tensor(0.0)
        bptt_lambda = 0.0

    if long_cloud_losses:
        loss_long_cloud = torch.stack(long_cloud_losses).mean()
    else:
        loss_long_cloud = loss_one.new_tensor(0.0)

    if forced_damped_losses:
        loss_forced_damped = torch.stack(forced_damped_losses).mean()
    else:
        loss_forced_damped = loss_one.new_tensor(0.0)

    if transport_coh_losses:
        loss_transport_coh = torch.stack(transport_coh_losses).mean()
    else:
        loss_transport_coh = loss_one.new_tensor(0.0)

    if source_color_losses:
        loss_source_color = torch.stack(source_color_losses).mean()
        source_color_lambda = float(getattr(args, "source_color_lambda", 0.0))
    else:
        loss_source_color = loss_one.new_tensor(0.0)
        source_color_lambda = 0.0

    if error_proj_pseudo_losses:
        loss_error_proj_pseudo = torch.stack(error_proj_pseudo_losses).mean()
        error_proj_pseudo_lambda = float(getattr(args, "error_proj_pseudo_lambda", 0.0))
    else:
        loss_error_proj_pseudo = loss_one.new_tensor(0.0)
        error_proj_pseudo_lambda = 0.0

    if error_proj_cob_losses:
        loss_error_proj_cob = torch.stack(error_proj_cob_losses).mean()
        error_proj_cob_lambda = float(getattr(args, "error_proj_cob_lambda", 0.0))
    else:
        loss_error_proj_cob = loss_one.new_tensor(0.0)
        error_proj_cob_lambda = 0.0

    if error_proj_v2_losses:
        loss_error_proj_v2 = torch.stack(error_proj_v2_losses).mean()
        error_proj_v2_lambda = float(getattr(args, "error_proj_v2_lambda", 0.0))
    else:
        loss_error_proj_v2 = loss_one.new_tensor(0.0)
        error_proj_v2_lambda = 0.0

    if error_align_losses:
        loss_error_align_tf = torch.stack(error_align_tf_losses).mean()
        loss_error_align = torch.stack(error_align_losses).mean()
        error_align_tf_lambda = float(getattr(args, "error_align_tf_lambda", 1.0))
        error_align_lambda = float(getattr(args, "error_align_lambda", 0.0))
    else:
        loss_error_align_tf = loss_one.new_tensor(0.0)
        loss_error_align = loss_one.new_tensor(0.0)
        error_align_tf_lambda = 0.0
        error_align_lambda = 0.0

    if field_peh_losses:
        loss_field_peh = torch.stack(field_peh_losses).mean()
        field_peh_lambda = _field_peh_effective_lambda(args, epoch) if raw.training else float(getattr(args, "field_peh_lambda", 0.0))
    else:
        loss_field_peh = loss_one.new_tensor(0.0)
        field_peh_lambda = 0.0

    loss_ftg_amp_weighted = ftg_lambda_amp * loss_ftg_amp
    loss_ftg_bound_weighted = ftg_lambda_bound * loss_ftg_bound
    loss_comp_weighted = comp_lambda * loss_comp
    loss_etm_fit_weighted = etm_lambda_fit * loss_etm_fit
    loss_etm_prop_weighted = etm_lambda_prop * loss_etm_prop
    loss_etm_weighted = loss_etm_fit_weighted + loss_etm_prop_weighted
    loss_frontier_weighted = frontier_lambda * loss_frontier
    loss_bptt_weighted = bptt_lambda * loss_bptt
    # compute_long_error_cloud_loss already applies its internal lambdas.
    loss_long_cloud_weighted = loss_long_cloud
    # compute_forced_damped_error_dynamics_loss already applies its internal lambdas.
    loss_forced_damped_weighted = loss_forced_damped
    # compute_transport_coherence_loss already applies its internal lambdas.
    loss_transport_coh_weighted = loss_transport_coh
    loss_source_color_weighted = source_color_lambda * loss_source_color
    loss_error_proj_pseudo_weighted = error_proj_pseudo_lambda * loss_error_proj_pseudo
    loss_error_proj_cob_weighted = error_proj_cob_lambda * loss_error_proj_cob
    loss_error_proj_v2_weighted = error_proj_v2_lambda * loss_error_proj_v2
    loss_error_align_tf_weighted = error_align_tf_lambda * loss_error_align_tf
    loss_error_align_weighted = error_align_lambda * loss_error_align
    loss_error_align_total_weighted = loss_error_align_tf_weighted + loss_error_align_weighted
    loss_field_peh_weighted = field_peh_lambda * loss_field_peh
    # When --comp_zero_value_loss is enabled, loss_comp is numerically zero by
    # design although its gradient is nonzero.  For diagnostics, report a
    # proxy weighted magnitude based on the raw detached pseudo dot.
    comp_proxy_dot_for_log = (
        torch.as_tensor(
            comp_log_accum.get("ar/comp_proxy_dot", 0.0) / max(float(n_comp_logs), 1.0),
            device=loss_one.device,
            dtype=loss_one.dtype,
        )
        if n_comp_logs > 0
        else loss_one.new_tensor(0.0)
    )
    loss_comp_proxy_weighted = comp_lambda * comp_proxy_dot_for_log
    frontier_proxy_dot_for_log = (
        torch.as_tensor(
            frontier_log_accum.get("ar/frontier_proxy_dot", 0.0) / max(float(n_frontier_logs), 1.0),
            device=loss_one.device,
            dtype=loss_one.dtype,
        )
        if n_frontier_logs > 0
        else loss_one.new_tensor(0.0)
    )
    loss_frontier_proxy_weighted = frontier_lambda * frontier_proxy_dot_for_log

    one_step_lambda = float(getattr(args, "ar_one_step_lambda", 1.0))
    loss_one_weighted = one_step_lambda * loss_one
    forward_jacobian_lambda = float(
        getattr(args, "forward_jacobian_lambda", 0.0)
    )
    if raw.training and forward_jacobian_lambda != 0.0 and chosen:
        jreg_t = int(chosen[0])
        jreg_history = time_window(state, jreg_t - W, jreg_t)
        jreg_stim = time_window(stim, jreg_t - W, jreg_t)

        def _jreg_windowed_forward(inp):
            return _ar_forward_pred(raw, jreg_stim, inp)

        forward_jacobian_loss, forward_jacobian_gain = (
            _forward_jacobian_fd_penalty(
                jreg_history,
                _jreg_windowed_forward,
                eps=float(getattr(args, "forward_jacobian_eps", 1e-3)),
                target=float(getattr(args, "forward_jacobian_target", 1.0)),
            )
        )
    else:
        forward_jacobian_loss = loss_one.new_tensor(0.0)
        forward_jacobian_gain = loss_one.new_tensor(0.0)
    forward_jacobian_weighted = (
        forward_jacobian_lambda * forward_jacobian_loss
    )
    loss = loss_one_weighted + loss_fold_weighted + loss_ftg_amp_weighted + loss_ftg_bound_weighted + loss_comp_weighted + loss_etm_weighted + loss_frontier_weighted + loss_bptt_weighted + loss_long_cloud_weighted + loss_forced_damped_weighted + loss_transport_coh_weighted + loss_source_color_weighted + loss_error_proj_pseudo_weighted + loss_error_proj_cob_weighted + loss_error_proj_v2_weighted + loss_error_align_total_weighted + loss_field_peh_weighted + forward_jacobian_weighted

    ftg_total_weighted = loss_ftg_amp_weighted + loss_ftg_bound_weighted
    eps_log = 1e-12

    if raw.training and bool(getattr(args, "fast_train_logging", False)):
        # Preserve the exact objective while avoiding the large diagnostic
        # dictionary's per-field device-to-host synchronization on every batch.
        return loss, {
            "loss": loss.detach(),
            "ar/loss_one_step": loss_one.detach(),
            "ar/one_step_lambda": float(one_step_lambda),
        }

    logs = {
        "loss": float(loss.detach().cpu()),
        "ar/loss_one_step": float(loss_one.detach().cpu()),
        "ar/one_step_lambda": float(one_step_lambda),
        "ar/loss_one_step_weighted": float(loss_one_weighted.detach().cpu()),
        "ar/forward_jacobian_loss": float(forward_jacobian_loss.detach().cpu()),
        "ar/forward_jacobian_lambda": float(forward_jacobian_lambda),
        "ar/forward_jacobian_weighted_loss": float(
            forward_jacobian_weighted.detach().cpu()
        ),
        "ar/forward_jacobian_gain": float(
            forward_jacobian_gain.detach().cpu()
        ),
        "ar/fold_loss": float(loss_fold.detach().cpu()),
        "ar/fold_lambda": float(fold_lambda),
        "ar/fold_weighted_loss": float(loss_fold_weighted.detach().cpu()),
        "ar/ftg_amp_loss": float(loss_ftg_amp.detach().cpu()),
        "ar/ftg_bound_loss": float(loss_ftg_bound.detach().cpu()),
        "ar/ftg_lambda_amp": float(ftg_lambda_amp),
        "ar/ftg_lambda_bound": float(ftg_lambda_bound),
        "ar/ftg_amp_weighted_loss": float(loss_ftg_amp_weighted.detach().cpu()),
        "ar/ftg_bound_weighted_loss": float(loss_ftg_bound_weighted.detach().cpu()),
        "ar/ftg_total_weighted_loss": float(ftg_total_weighted.detach().cpu()),
        "ar/comp_loss": float(loss_comp.detach().cpu()),
        "ar/comp_lambda": float(comp_lambda),
        "ar/comp_weighted_loss": float(loss_comp_weighted.detach().cpu()),
        "ar/comp_proxy_weighted": float(loss_comp_proxy_weighted.detach().cpu()),
        "ar/comp_effective_lambda": float(comp_lambda),
        "ar/etm_fit_loss": float(loss_etm_fit.detach().cpu()),
        "ar/etm_prop_loss": float(loss_etm_prop.detach().cpu()),
        "ar/etm_lambda_fit": float(etm_lambda_fit),
        "ar/etm_lambda_prop": float(etm_lambda_prop),
        "ar/etm_fit_weighted_loss": float(loss_etm_fit_weighted.detach().cpu()),
        "ar/etm_prop_weighted_loss": float(loss_etm_prop_weighted.detach().cpu()),
        "ar/etm_weighted_loss": float(loss_etm_weighted.detach().cpu()),
        "ar/etm_pct_of_one_step": float((100.0 * loss_etm_weighted / loss_one.detach().clamp_min(eps_log)).detach().cpu()),
        "ar/etm_pct_of_total": float((100.0 * loss_etm_weighted / loss.detach().clamp_min(eps_log)).detach().cpu()),
        "ar/frontier_loss": float(loss_frontier.detach().cpu()),
        "ar/frontier_lambda": float(frontier_lambda),
        "ar/frontier_weighted_loss": float(loss_frontier_weighted.detach().cpu()),
        "ar/frontier_proxy_weighted": float(loss_frontier_proxy_weighted.detach().cpu()),
        "ar/frontier_effective_lambda": float(frontier_lambda),
        "ar/bptt_loss": float(loss_bptt.detach().cpu()),
        "ar/bptt_lambda": float(bptt_lambda),
        "ar/bptt_weighted_loss": float(loss_bptt_weighted.detach().cpu()),
        "ar/field_peh_loss": float(loss_field_peh.detach().cpu()),
        "ar/field_peh_lambda": float(field_peh_lambda),
        "ar/field_peh_weighted_loss": float(loss_field_peh_weighted.detach().cpu()),
        "ar/field_peh_pct_of_one_step": float((100.0 * loss_field_peh_weighted / loss_one.detach().clamp_min(eps_log)).detach().cpu()),
        "ar/field_peh_pct_of_total": float((100.0 * loss_field_peh_weighted / loss.detach().clamp_min(eps_log)).detach().cpu()),
        "ar/long_cloud_loss": float(loss_long_cloud.detach().cpu()),
        "ar/long_cloud_weighted_loss": float(loss_long_cloud_weighted.detach().cpu()),
        "ar/long_cloud_pct_of_one_step": float((100.0 * loss_long_cloud_weighted / loss_one.detach().clamp_min(eps_log)).detach().cpu()),
        "ar/long_cloud_pct_of_total": float((100.0 * loss_long_cloud_weighted / loss.detach().clamp_min(eps_log)).detach().cpu()),
        "ar/fded_loss": float(loss_forced_damped.detach().cpu()),
        "ar/fded_weighted_loss": float(loss_forced_damped_weighted.detach().cpu()),
        "ar/fded_pct_of_one_step": float((100.0 * loss_forced_damped_weighted / loss_one.detach().clamp_min(eps_log)).detach().cpu()),
        "ar/fded_pct_of_total": float((100.0 * loss_forced_damped_weighted / loss.detach().clamp_min(eps_log)).detach().cpu()),
        "ar/transport_coh_loss": float(loss_transport_coh.detach().cpu()),
        "ar/transport_coh_weighted_loss": float(loss_transport_coh_weighted.detach().cpu()),
        "ar/transport_coh_pct_of_one_step": float((100.0 * loss_transport_coh_weighted / loss_one.detach().clamp_min(eps_log)).detach().cpu()),
        "ar/transport_coh_pct_of_total": float((100.0 * loss_transport_coh_weighted / loss.detach().clamp_min(eps_log)).detach().cpu()),
        "ar/source_color_loss": float(loss_source_color.detach().cpu()),
        "ar/source_color_lambda": float(source_color_lambda),
        "ar/source_color_weighted_loss": float(loss_source_color_weighted.detach().cpu()),
        "ar/source_color_pct_of_one_step": float((100.0 * loss_source_color_weighted / loss_one.detach().clamp_min(eps_log)).detach().cpu()),
        "ar/source_color_pct_of_total": float((100.0 * loss_source_color_weighted / loss.detach().clamp_min(eps_log)).detach().cpu()),
        "ar/error_proj_pseudo_loss": float(loss_error_proj_pseudo.detach().cpu()),
        "ar/error_proj_pseudo_lambda": float(error_proj_pseudo_lambda),
        "ar/error_proj_pseudo_weighted_loss": float(loss_error_proj_pseudo_weighted.detach().cpu()),
        "ar/error_proj_pseudo_pct_of_one_step": float((100.0 * loss_error_proj_pseudo_weighted / loss_one.detach().clamp_min(eps_log)).detach().cpu()),
        "ar/error_proj_pseudo_pct_of_total": float((100.0 * loss_error_proj_pseudo_weighted / loss.detach().clamp_min(eps_log)).detach().cpu()),
        "ar/error_proj_cob_loss": float(loss_error_proj_cob.detach().cpu()),
        "ar/error_proj_cob_lambda": float(error_proj_cob_lambda),
        "ar/error_proj_cob_weighted_loss": float(loss_error_proj_cob_weighted.detach().cpu()),
        "ar/error_proj_cob_pct_of_one_step": float((100.0 * loss_error_proj_cob_weighted / loss_one.detach().clamp_min(eps_log)).detach().cpu()),
        "ar/error_proj_cob_pct_of_total": float((100.0 * loss_error_proj_cob_weighted / loss.detach().clamp_min(eps_log)).detach().cpu()),
        "ar/error_proj_v2_loss": float(loss_error_proj_v2.detach().cpu()),
        "ar/error_proj_v2_lambda": float(error_proj_v2_lambda),
        "ar/error_proj_v2_weighted_loss": float(loss_error_proj_v2_weighted.detach().cpu()),
        "ar/error_proj_v2_pct_of_one_step": float((100.0 * loss_error_proj_v2_weighted / loss_one.detach().clamp_min(eps_log)).detach().cpu()),
        "ar/error_proj_v2_pct_of_total": float((100.0 * loss_error_proj_v2_weighted / loss.detach().clamp_min(eps_log)).detach().cpu()),
        "ar/error_align_tf_loss": float(loss_error_align_tf.detach().cpu()),
        "ar/error_align_tf_lambda": float(error_align_tf_lambda),
        "ar/error_align_tf_weighted_loss": float(loss_error_align_tf_weighted.detach().cpu()),
        "ar/error_align_loss": float(loss_error_align.detach().cpu()),
        "ar/error_align_lambda": float(error_align_lambda),
        "ar/error_align_weighted_loss": float(loss_error_align_weighted.detach().cpu()),
        "ar/error_align_total_weighted_loss": float(loss_error_align_total_weighted.detach().cpu()),
        "ar/error_align_pct_of_one_step": float((100.0 * loss_error_align_total_weighted / loss_one.detach().clamp_min(eps_log)).detach().cpu()),
        "ar/error_align_pct_of_total": float((100.0 * loss_error_align_total_weighted / loss.detach().clamp_min(eps_log)).detach().cpu()),
        "ar/bptt_pct_of_one_step": float((100.0 * loss_bptt_weighted / loss_one.detach().clamp_min(eps_log)).detach().cpu()),
        "ar/comp_pct_of_one_step": float((100.0 * loss_comp_weighted / loss_one.detach().clamp_min(eps_log)).detach().cpu()),
        "ar/comp_proxy_pct_of_one_step": float((100.0 * loss_comp_proxy_weighted / loss_one.detach().clamp_min(eps_log)).detach().cpu()),
        "ar/comp_pct_of_total": float((100.0 * loss_comp_weighted / loss.detach().clamp_min(eps_log)).detach().cpu()),
        "ar/frontier_pct_of_one_step": float((100.0 * loss_frontier_weighted / loss_one.detach().clamp_min(eps_log)).detach().cpu()),
        "ar/frontier_proxy_pct_of_one_step": float((100.0 * loss_frontier_proxy_weighted / loss_one.detach().clamp_min(eps_log)).detach().cpu()),
        "ar/frontier_pct_of_total": float((100.0 * loss_frontier_weighted / loss.detach().clamp_min(eps_log)).detach().cpu()),
        "ar/ftg_amp_pct_of_one_step": float((100.0 * loss_ftg_amp_weighted / loss_one.detach().clamp_min(eps_log)).detach().cpu()),
        "ar/ftg_amp_pct_of_total": float((100.0 * loss_ftg_amp_weighted / loss.detach().clamp_min(eps_log)).detach().cpu()),
        "ar/ftg_total_pct_of_one_step": float((100.0 * ftg_total_weighted / loss_one.detach().clamp_min(eps_log)).detach().cpu()),
        "ar/ftg_total_pct_of_total": float((100.0 * ftg_total_weighted / loss.detach().clamp_min(eps_log)).detach().cpu()),
        "ar/one_step_corr": (
            float(torch.stack(corrs).mean().detach().cpu()) if corrs else 0.0
        ),
        "ar/one_step_rel_l2": (
            float(torch.stack(rels).mean().detach().cpu()) if rels else 0.0
        ),
        "ar/num_train_starts": float(len(chosen)),
        "ar/window_size": float(W),
    }

    if n_fold_logs > 0:
        for k, v in fold_log_accum.items():
            if k not in {
                "ar/fold_loss",
                "ar/fold_lambda",
                "ar/fold_weighted_loss",
            }:
                logs[k] = v / float(n_fold_logs)

        # Make these exactly consistent with the actual loss construction.
        logs["ar/fold_loss"] = float(loss_fold.detach().cpu())
        logs["ar/fold_lambda"] = float(fold_lambda)
        logs["ar/fold_weighted_loss"] = float(loss_fold_weighted.detach().cpu())
    else:
        logs["ar/fold_used_horizon"] = 0.0
        logs["ar/fold_A_sigma"] = 0.0

    if n_ftg_logs > 0:
        for k, v in ftg_log_accum.items():
            if k not in {
                "ar/ftg_amp_loss",
                "ar/ftg_bound_loss",
                "ar/ftg_lambda_amp",
                "ar/ftg_lambda_bound",
            }:
                logs[k] = v / float(n_ftg_logs)
        logs["ar/ftg_amp_loss"] = float(loss_ftg_amp.detach().cpu())
        logs["ar/ftg_bound_loss"] = float(loss_ftg_bound.detach().cpu())
        logs["ar/ftg_lambda_amp"] = float(ftg_lambda_amp)
        logs["ar/ftg_lambda_bound"] = float(ftg_lambda_bound)
    else:
        logs.setdefault("ar/ftg_amp_ratio", 0.0)
        logs.setdefault("ar/ftg_horizon", 0.0)
        logs.setdefault("ar/ftg_sigma_hat", 0.0)
        logs.setdefault("ar/ftg_ratio_k0", 0.0)
        logs.setdefault("ar/ftg_ratio_k1", 0.0)
        logs.setdefault("ar/ftg_ratio_k2", 0.0)
        logs.setdefault("ar/ftg_ratio_klast", 0.0)

    if n_comp_logs > 0:
        for k, v in comp_log_accum.items():
            if k not in {"ar/comp_loss", "ar/comp_lambda"}:
                logs[k] = v / float(n_comp_logs)
        logs["ar/comp_loss"] = float(loss_comp.detach().cpu())
        logs["ar/comp_lambda"] = float(comp_lambda)
    else:
        logs.setdefault("ar/comp_horizon", 0.0)
        logs.setdefault("ar/comp_block_size", 0.0)
        logs.setdefault("ar/comp_blocks", 0.0)
        logs.setdefault("ar/comp_injection_steps", 0.0)
        logs.setdefault("ar/comp_mean_error_norm", 0.0)
        logs.setdefault("ar/comp_mean_pseudo_norm", 0.0)
        logs.setdefault("ar/comp_first_rel_l2", 0.0)
        logs.setdefault("ar/comp_last_rel_l2", 0.0)

    if n_etm_logs > 0:
        for k, v in etm_log_accum.items():
            if k not in {"ar/etm_fit_loss", "ar/etm_prop_loss"}:
                logs[k] = v / float(n_etm_logs)
        logs["ar/etm_fit_loss"] = float(loss_etm_fit.detach().cpu())
        logs["ar/etm_prop_loss"] = float(loss_etm_prop.detach().cpu())
    else:
        for k, v in _etm_zero_logs().items():
            logs.setdefault(k, v)

    if n_frontier_logs > 0:
        for k, v in frontier_log_accum.items():
            if k not in {"ar/frontier_loss", "ar/frontier_lambda"}:
                logs[k] = v / float(n_frontier_logs)
        logs["ar/frontier_loss"] = float(loss_frontier.detach().cpu())
        logs["ar/frontier_lambda"] = float(frontier_lambda)
    else:
        logs.setdefault("ar/frontier_horizon", 0.0)
        logs.setdefault("ar/frontier_tau", 0.0)
        logs.setdefault("ar/frontier_injection_steps", 0.0)
        logs.setdefault("ar/frontier_keep_frac", 0.0)
        logs.setdefault("ar/frontier_survival_norm_ratio", 0.0)
        logs.setdefault("ar/frontier_first_rel_l2", 0.0)
        logs.setdefault("ar/frontier_last_rel_l2", 0.0)

    if n_bptt_logs > 0:
        for k, v in bptt_log_accum.items():
            if k not in {"ar/bptt_loss", "ar/bptt_lambda"}:
                logs[k] = v / float(n_bptt_logs)
        logs["ar/bptt_loss"] = float(loss_bptt.detach().cpu())
        logs["ar/bptt_lambda"] = float(bptt_lambda)
    else:
        logs.setdefault("ar/bptt_horizon", 0.0)
        logs.setdefault("ar/bptt_first_rel_l2", 0.0)
        logs.setdefault("ar/bptt_last_rel_l2", 0.0)

    if hasattr(raw, "dual_wiener_diagnostics"):
        logs.update(raw.dual_wiener_diagnostics(int(getattr(args, "bptt_horizon", 1))))

    if n_long_cloud_logs > 0:
        for k, v in long_cloud_log_accum.items():
            if k not in {"ar/long_cloud_loss"}:
                logs[k] = v / float(n_long_cloud_logs)
        logs["ar/long_cloud_loss"] = float(loss_long_cloud.detach().cpu())
        logs["ar/long_cloud_weighted_loss"] = float(loss_long_cloud_weighted.detach().cpu())
    else:
        for k, v in _long_error_cloud_zero_logs().items():
            logs.setdefault(k, v)
        logs.setdefault("ar/long_cloud_weighted_loss", 0.0)
        logs.setdefault("ar/long_cloud_pct_of_one_step", 0.0)
        logs.setdefault("ar/long_cloud_pct_of_total", 0.0)

    if n_forced_damped_logs > 0:
        for k, v in forced_damped_log_accum.items():
            if k not in {"ar/fded_loss"}:
                logs[k] = v / float(n_forced_damped_logs)
        logs["ar/fded_loss"] = float(loss_forced_damped.detach().cpu())
        logs["ar/fded_weighted_loss"] = float(loss_forced_damped_weighted.detach().cpu())
    else:
        for k, v in _forced_damped_zero_logs().items():
            logs.setdefault(k, v)
        logs.setdefault("ar/fded_weighted_loss", 0.0)
        logs.setdefault("ar/fded_pct_of_one_step", 0.0)
        logs.setdefault("ar/fded_pct_of_total", 0.0)

    if n_transport_coh_logs > 0:
        for k, v in transport_coh_log_accum.items():
            if k not in {"ar/transport_coh_loss", "ar/transport_coh_weighted_loss"}:
                logs[k] = v / float(n_transport_coh_logs)
        logs["ar/transport_coh_loss"] = float(loss_transport_coh.detach().cpu())
        logs["ar/transport_coh_weighted_loss"] = float(loss_transport_coh_weighted.detach().cpu())
    else:
        for k, v in _transport_coh_zero_logs().items():
            logs.setdefault(k, v)
        logs.setdefault("ar/transport_coh_weighted_loss", 0.0)
        logs.setdefault("ar/transport_coh_pct_of_one_step", 0.0)
        logs.setdefault("ar/transport_coh_pct_of_total", 0.0)

    if n_source_color_logs > 0:
        for k, v in source_color_log_accum.items():
            if k not in {"ar/source_color_loss", "ar/source_color_lambda", "ar/source_color_weighted_loss"}:
                logs[k] = v / float(n_source_color_logs)
        logs["ar/source_color_loss"] = float(loss_source_color.detach().cpu())
        logs["ar/source_color_lambda"] = float(source_color_lambda)
        logs["ar/source_color_weighted_loss"] = float(loss_source_color_weighted.detach().cpu())
    else:
        for k, v in _source_color_zero_logs().items():
            logs.setdefault(k, v)

    if n_error_proj_pseudo_logs > 0:
        for k, v in error_proj_pseudo_log_accum.items():
            if k not in {"ar/error_proj_pseudo_loss", "ar/error_proj_pseudo_lambda", "ar/error_proj_pseudo_weighted_loss"}:
                logs[k] = v / float(n_error_proj_pseudo_logs)
        logs["ar/error_proj_pseudo_loss"] = float(loss_error_proj_pseudo.detach().cpu())
        logs["ar/error_proj_pseudo_lambda"] = float(error_proj_pseudo_lambda)
        logs["ar/error_proj_pseudo_weighted_loss"] = float(loss_error_proj_pseudo_weighted.detach().cpu())
    else:
        for k, v in _error_proj_pseudo_zero_logs().items():
            logs.setdefault(k, v)
        logs.setdefault("ar/error_proj_pseudo_pct_of_one_step", 0.0)
        logs.setdefault("ar/error_proj_pseudo_pct_of_total", 0.0)

    if n_error_proj_cob_logs > 0:
        for k, v in error_proj_cob_log_accum.items():
            if k not in {"ar/error_proj_cob_loss", "ar/error_proj_cob_lambda", "ar/error_proj_cob_weighted_loss"}:
                logs[k] = v / float(n_error_proj_cob_logs)
        logs["ar/error_proj_cob_loss"] = float(loss_error_proj_cob.detach().cpu())
        logs["ar/error_proj_cob_lambda"] = float(error_proj_cob_lambda)
        logs["ar/error_proj_cob_weighted_loss"] = float(loss_error_proj_cob_weighted.detach().cpu())
    else:
        for k, v in _error_proj_coboundary_zero_logs().items():
            logs.setdefault(k, v)
        logs.setdefault("ar/error_proj_cob_pct_of_one_step", 0.0)
        logs.setdefault("ar/error_proj_cob_pct_of_total", 0.0)

    if n_error_proj_v2_logs > 0:
        for k, v in error_proj_v2_log_accum.items():
            if k not in {"ar/error_proj_v2_loss", "ar/error_proj_v2_lambda", "ar/error_proj_v2_weighted_loss"}:
                logs[k] = v / float(n_error_proj_v2_logs)
        logs["ar/error_proj_v2_loss"] = float(loss_error_proj_v2.detach().cpu())
        logs["ar/error_proj_v2_lambda"] = float(error_proj_v2_lambda)
        logs["ar/error_proj_v2_weighted_loss"] = float(loss_error_proj_v2_weighted.detach().cpu())
    else:
        for k, v in _error_proj_v2_zero_logs().items():
            logs.setdefault(k, v)
        logs.setdefault("ar/error_proj_v2_pct_of_one_step", 0.0)
        logs.setdefault("ar/error_proj_v2_pct_of_total", 0.0)

    if n_error_align_logs > 0:
        for k, v in error_align_log_accum.items():
            if k not in {"ar/error_align_tf_loss", "ar/error_align_loss", "ar/error_align_lambda", "ar/error_align_weighted_loss"}:
                logs[k] = v / float(n_error_align_logs)
        logs["ar/error_align_tf_loss"] = float(loss_error_align_tf.detach().cpu())
        logs["ar/error_align_tf_lambda"] = float(error_align_tf_lambda)
        logs["ar/error_align_tf_weighted_loss"] = float(loss_error_align_tf_weighted.detach().cpu())
        logs["ar/error_align_loss"] = float(loss_error_align.detach().cpu())
        logs["ar/error_align_lambda"] = float(error_align_lambda)
        logs["ar/error_align_weighted_loss"] = float(loss_error_align_weighted.detach().cpu())
        logs["ar/error_align_total_weighted_loss"] = float(loss_error_align_total_weighted.detach().cpu())
    else:
        for k, v in _error_align_zero_logs().items():
            logs.setdefault(k, v)

    if n_field_peh_logs > 0:
        for k, v in field_peh_log_accum.items():
            if k not in {"ar/field_peh_loss", "ar/field_peh_lambda", "ar/field_peh_weighted_loss"}:
                logs[k] = v / float(n_field_peh_logs)
        logs["ar/field_peh_loss"] = float(loss_field_peh.detach().cpu())
        logs["ar/field_peh_lambda"] = float(field_peh_lambda)
        logs["ar/field_peh_weighted_loss"] = float(loss_field_peh_weighted.detach().cpu())
    else:
        for k, v in _field_peh_zero_logs().items():
            logs.setdefault(k, v)
        logs.setdefault("ar/field_peh_pct_of_one_step", 0.0)
        logs.setdefault("ar/field_peh_pct_of_total", 0.0)

    if stageb_logs:
        logs.update(stageb_logs)

    return loss, logs


def _potential_coboundary_target(state: torch.Tensor, mode: str) -> torch.Tensor:
    """Build a data-space target for vector-potential coboundary fitting.

    Stage A now treats the encoder output itself as the vector potential

        v_t = V(x_t).

    The meaningful coboundary is Delta v_t = v_{t+1}-v_t.  To prevent v_t from
    being an arbitrary static AE code, the model decodes Delta v_t back to a
    trajectory quantity.  The default target is the state increment

        y_t = x_{t+1} - x_t,

    so D_delta(v_{t+1}-v_t) must explain the local change in data space.
    """
    mode = str(mode).lower()
    if mode in {"state_delta", "dx", "increment"}:
        return state[:, 1:] - state[:, :-1]
    if mode == "normalized_state_delta":
        dx = state[:, 1:] - state[:, :-1]
        denom = state[:, :-1].reshape(state.shape[0], state.shape[1]-1, -1).norm(dim=-1).view(state.shape[0], state.shape[1]-1, *([1]*(state.dim()-2))).clamp_min(1e-8)
        return dx / denom
    if mode == "state_energy_delta_scalar":
        flat = state.reshape(state.shape[0], state.shape[1], -1)
        energy = 0.5 * flat.pow(2).sum(dim=-1)
        return energy[:, 1:] - energy[:, :-1]
    raise ValueError(f"Unknown potential_cob_target={mode}")

def _standardize_pair(pred: torch.Tensor, target: torch.Tensor, eps: float):
    # Standardize on pooled batch/time entries.  This prevents the scalar head
    # from wasting optimization on target scale before it learns the coboundary
    # geometry.
    t_mean = target.mean()
    t_std = target.std(unbiased=False).clamp_min(eps)
    p_mean = pred.mean()
    p_std = pred.std(unbiased=False).clamp_min(eps)
    return (pred - p_mean) / p_std, (target - t_mean) / t_std


def _rel_mse(pred: torch.Tensor, target: torch.Tensor, eps: float) -> torch.Tensor:
    return (pred - target).pow(2).mean() / target.pow(2).mean().clamp_min(eps)


def _loss_by_type(pred: torch.Tensor, target: torch.Tensor, loss_type: str, eps: float) -> torch.Tensor:
    loss_type = str(loss_type).lower()
    if loss_type == "mse":
        return (pred - target).pow(2).mean()
    if loss_type in {"rel_mse", "relative_mse"}:
        return _rel_mse(pred, target, eps)
    if loss_type == "rel_l2":
        pf = pred.reshape(pred.shape[0], -1)
        tf = target.reshape(target.shape[0], -1)
        return (pf - tf).norm(dim=1).mean() / tf.norm(dim=1).mean().clamp_min(eps)
    if loss_type == "corr":
        p0 = pred - pred.mean()
        t0 = target - target.mean()
        corr = (p0 * t0).mean() / (p0.pow(2).mean().sqrt() * t0.pow(2).mean().sqrt()).clamp_min(eps)
        return 1.0 - corr
    raise ValueError(f"Unknown loss_type={loss_type}")


def _sample_potential_pairs(T: int, num_pairs: int, min_gap: int, max_gap: int, device: torch.device):
    """Sample endpoint pairs (i,j) with 0 <= i < j < T."""
    min_gap = max(1, int(min_gap))
    max_gap = int(max_gap) if int(max_gap) > 0 else T - 1
    max_gap = min(max_gap, T - 1)
    if max_gap < min_gap:
        max_gap = min_gap
    # For small windows the all-pairs set is cheap and more stable.
    all_pairs = [(i, j) for i in range(T) for j in range(i + min_gap, min(T, i + max_gap + 1))]
    if not all_pairs:
        raise ValueError(f"No valid endpoint pairs for T={T}, min_gap={min_gap}, max_gap={max_gap}")
    if num_pairs <= 0 or num_pairs >= len(all_pairs):
        idx = torch.tensor([i for i, _j in all_pairs], device=device, dtype=torch.long)
        jdx = torch.tensor([j for _i, j in all_pairs], device=device, dtype=torch.long)
        return idx, jdx
    choice = torch.randint(0, len(all_pairs), (int(num_pairs),), device=device)
    pair_tensor = torch.tensor(all_pairs, device=device, dtype=torch.long)
    picked = pair_tensor[choice]
    return picked[:, 0], picked[:, 1]


def _gather_time(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather time indices shared by all batch items from [B,T,...] -> [B,P,...]."""
    return x.index_select(dim=1, index=idx)


def compute_stimulus_potential_ae_loss(model, state: torch.Tensor, stim: torch.Tensor | None, args, epoch: int | float = 0):
    """Stage A endpoint-pair loss for an external-input-driven potential.

    This is deliberately *not* an AR rollout objective.  Stage A learns a state
    function and an external-work model:

        v_t = V(x_t),                 x_t^Phi = D(v_t)
        W(u_{i:j}) = sum_{tau=i}^{j-1} G(u_{tau-L+1:tau})

    and constrains arbitrary endpoint pairs in the same trajectory by

        V(x_j) - V(x_i) ~= W(u_{i:j}).

    The decoder is trained mainly by static reconstruction so that V(x) carries
    useful fMRI information.  A weaker component-consistency loss only requires
    the potential component to be self-consistent,

        D(V(x_i) + W(u_{i:j})) ~= D(V(x_j)),

    not equal to the full x_j.  The full x_j contains path-dependent residual
    information that Stage B is supposed to model.
    """
    raw = unwrap_model(model)
    if stim is None:
        stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))
    out = raw(state)
    recon, v = out[0], out[1]
    eps = float(getattr(args, "rel_l2_eps", 1e-8))
    B, T, _ = get_batch_time_shape(state)
    if T < 2:
        raise ValueError("stimulus potential endpoint loss requires sequence state with T>=2")
    if stim.shape[1] < T - 1:
        raise ValueError(f"stim sequence too short for Stage-A endpoint potential: stim T={stim.shape[1]}, state T={T}")

    # Static reconstruction: this is the supervised anchor preventing collapse.
    rec_mse = (recon - state).pow(2).mean()
    rec_rel = ((recon - state).reshape(B, -1).norm(dim=1) / state.reshape(B, -1).norm(dim=1).clamp_min(eps)).mean()
    rec_loss = rec_rel if str(getattr(args, "potential_rec_loss", "rel_l2")) == "rel_l2" else rec_mse

    centered = state - state.mean(dim=(0, 1), keepdim=True)
    rec_r2 = 1.0 - (recon - state).pow(2).sum() / centered.pow(2).sum().clamp_min(eps)
    # Direct static reconstruction corr, useful because user's stimulus-only
    # baseline is usually reported as x-space correlation.
    r0 = recon - recon.mean()
    x0 = state - state.mean()
    rec_corr = (r0 * x0).mean() / (r0.pow(2).mean().sqrt() * x0.pow(2).mean().sqrt()).clamp_min(eps)

    phi_l2 = v.pow(2).mean()
    reduce_dims = tuple(range(v.dim() - 1))
    phi_centered = v - v.mean(dim=reduce_dims, keepdim=True)
    phi_var = phi_centered.pow(2).mean()

    # External work increments G(u_{tau-L+1:tau}) for tau=0..T-2.
    g = raw.stim_delta(stim[:, :T-1])
    csum = torch.cat([g.new_zeros(g.shape[0], 1, g.shape[-1]), g.cumsum(dim=1)], dim=1)  # [B,T,L]

    pair_samples = int(getattr(args, "potential_pair_samples", 0))
    min_gap = int(getattr(args, "potential_pair_min_gap", 1))
    max_gap = int(getattr(args, "potential_pair_max_gap", getattr(args, "potential_rollout_horizon", T - 1)))
    i_idx, j_idx = _sample_potential_pairs(T, pair_samples, min_gap, max_gap, state.device)
    gaps = (j_idx - i_idx).float()

    v_i = _gather_time(v, i_idx)
    v_j = _gather_time(v, j_idx)
    w_ij = _gather_time(csum, j_idx) - _gather_time(csum, i_idx)
    target_delta = v_j - v_i
    v_pred_j = v_i + w_ij

    pot_loss_type = str(getattr(args, "potential_cob_loss_type", "rel_mse"))
    pair_pot_loss = _loss_by_type(w_ij, target_delta, pot_loss_type, eps)
    pair_pot_mse = (w_ij - target_delta).pow(2).mean()
    pair_resid = w_ij - target_delta
    pair_pot_r2 = 1.0 - pair_resid.pow(2).sum() / (target_delta - target_delta.mean()).pow(2).sum().clamp_min(eps)
    p0 = w_ij - w_ij.mean()
    t0 = target_delta - target_delta.mean()
    pair_pot_corr = (p0 * t0).mean() / (p0.pow(2).mean().sqrt() * t0.pow(2).mean().sqrt()).clamp_min(eps)
    pair_residual_frac = pair_resid.reshape(pair_resid.shape[0], -1).norm(dim=1).mean() / target_delta.reshape(target_delta.shape[0], -1).norm(dim=1).mean().clamp_min(eps)
    pair_pred_norm = w_ij.reshape(w_ij.shape[0], -1).norm(dim=1).mean()
    pair_target_norm = target_delta.reshape(target_delta.shape[0], -1).norm(dim=1).mean()

    # Potential-component consistency, not full x_j prediction.
    x_phi_pred_j = raw.decode(v_pred_j)
    x_phi_j = _gather_time(recon, j_idx).detach() if bool(getattr(args, "potential_detach_cons_target", True)) else _gather_time(recon, j_idx)
    cons_loss_type = str(getattr(args, "potential_cons_loss_type", getattr(args, "potential_roll_loss_type", "rel_mse")))
    cons_loss = _loss_by_type(x_phi_pred_j, x_phi_j, cons_loss_type, eps)
    cons_mse = (x_phi_pred_j - x_phi_j).pow(2).mean()
    xp0 = x_phi_pred_j - x_phi_pred_j.mean()
    xt0 = x_phi_j - x_phi_j.mean()
    cons_corr = (xp0 * xt0).mean() / (xp0.pow(2).mean().sqrt() * xt0.pow(2).mean().sqrt()).clamp_min(eps)

    # Diagnostic only: how much the potential component resembles full x_j.
    x_j = _gather_time(state, j_idx)
    xfull_loss = _loss_by_type(x_phi_pred_j, x_j, str(getattr(args, "potential_roll_loss_type", "rel_mse")), eps)
    xf0 = x_phi_pred_j - x_phi_pred_j.mean()
    xj0 = x_j - x_j.mean()
    xfull_corr = (xf0 * xj0).mean() / (xf0.pow(2).mean().sqrt() * xj0.pow(2).mean().sqrt()).clamp_min(eps)

    # Keep old names as aliases for continuity, but now they mean endpoint-pair
    # potential loss and component consistency, not AR rollout-to-x loss.
    loss = float(getattr(args, "potential_lambda_rec", 0.5)) * rec_loss
    loss = loss + float(getattr(args, "potential_lambda_cob", 0.4)) * pair_pot_loss
    loss = loss + float(getattr(args, "potential_lambda_cons", getattr(args, "potential_lambda_roll", 0.1))) * cons_loss
    loss = loss + float(getattr(args, "potential_lambda_l2", 0.0)) * phi_l2
    loss = loss - float(getattr(args, "potential_lambda_var", 0.0)) * phi_var

    logs = {
        "loss": float(loss.detach()),
        "potential_rec_loss": float(rec_loss.detach()),
        "potential_rec_mse": float(rec_mse.detach()),
        "potential_rec_rel_l2": float(rec_rel.detach()),
        "potential_rec_r2": float(rec_r2.detach()),
        "potential_rec_corr": float(rec_corr.detach()),
        "potential_phi_l2": float(phi_l2.detach()),
        "potential_phi_var": float(phi_var.detach()),
        "potential_latent_dim": float(getattr(raw, "latent_dim", 0)),
        "potential_stim_cob_enabled": 1.0,
        "potential_endpoint_pair_enabled": 1.0,
        "potential_pair_samples": float(i_idx.numel()),
        "potential_pair_min_gap": float(gaps.min().detach()),
        "potential_pair_max_gap": float(gaps.max().detach()),
        "potential_pair_mean_gap": float(gaps.mean().detach()),
        "potential_stim_context_len": float(getattr(raw, "stim_context_len", getattr(args, "potential_stim_context_len", 1))),
        "potential_cob_loss": float(pair_pot_loss.detach()),
        "potential_pair_pot_loss": float(pair_pot_loss.detach()),
        "potential_cob_mse": float(pair_pot_mse.detach()),
        "potential_pair_pot_mse": float(pair_pot_mse.detach()),
        "potential_cob_r2": float(pair_pot_r2.detach()),
        "potential_pair_pot_r2": float(pair_pot_r2.detach()),
        "potential_cob_corr": float(pair_pot_corr.detach()),
        "potential_pair_pot_corr": float(pair_pot_corr.detach()),
        "potential_cob_residual_frac": float(pair_residual_frac.detach()),
        "potential_pair_residual_frac": float(pair_residual_frac.detach()),
        "potential_cob_pred_norm": float(pair_pred_norm.detach()),
        "potential_cob_target_norm": float(pair_target_norm.detach()),
        "potential_component_cons_loss": float(cons_loss.detach()),
        "potential_component_cons_mse": float(cons_mse.detach()),
        "potential_component_cons_corr": float(cons_corr.detach()),
        "potential_xroll_loss": float(cons_loss.detach()),
        "potential_xroll_mse": float(cons_mse.detach()),
        "potential_x_to_gt_diag_loss": float(xfull_loss.detach()),
        "potential_x_to_gt_diag_corr": float(xfull_corr.detach()),
        "potential_lambda_rec": float(getattr(args, "potential_lambda_rec", 0.5)),
        "potential_lambda_cob": float(getattr(args, "potential_lambda_cob", 0.4)),
        "potential_lambda_cons": float(getattr(args, "potential_lambda_cons", getattr(args, "potential_lambda_roll", 0.1))),
        "potential_lambda_roll": float(getattr(args, "potential_lambda_roll", 0.1)),
    }
    return loss, logs


def compute_potential_ae_loss(model, state: torch.Tensor, stim: torch.Tensor | None, args, epoch: int | float = 0):
    """Stage-A coboundary-constrained potential AE loss.

    The model still learns a time-independent reusable state function

        phi_t = E(x_t),  x_t ~= D(phi_t),

    but it is no longer a plain AE.  A scalar potential head V(phi) is trained
    so that a chosen trajectory scalar y_t is represented as an exact
    coboundary in phi-space:

        y_t ~= V(phi_{t+1}) - V(phi_t).

    With the default target y_t = 0.5(||x_{t+1}||^2 - ||x_t||^2), the target is
    itself a boundary term.  Therefore phi must carry information that supports
    a reusable potential function, rather than being an arbitrary compression.
    """
    if bool(getattr(args, "potential_stimulus_coboundary", False)):
        return compute_stimulus_potential_ae_loss(model, state, stim, args, epoch=epoch)

    raw = unwrap_model(model)
    out = raw(state)
    if len(out) == 2:
        recon, phi = out
        value = phi.new_zeros(phi.shape[:2]) if phi.dim() == 3 else phi.new_zeros(phi.shape[0])
    else:
        recon, phi, value = out
    eps = float(getattr(args, "rel_l2_eps", 1e-8))
    B = state.shape[0]

    rec_mse = (recon - state).pow(2).mean()
    rec_rel = ((recon - state).reshape(B, -1).norm(dim=1) / state.reshape(B, -1).norm(dim=1).clamp_min(eps)).mean()
    loss_type = str(getattr(args, "potential_rec_loss", "rel_l2"))
    rec_loss = rec_rel if loss_type == "rel_l2" else rec_mse

    phi_l2 = phi.pow(2).mean()
    reduce_dims = tuple(range(phi.dim() - 1))
    phi_centered = phi - phi.mean(dim=reduce_dims, keepdim=True)
    phi_var = phi_centered.pow(2).mean()

    if phi.dim() == 3 and phi.shape[1] > 1:
        dphi = phi[:, 1:] - phi[:, :-1]
        dx = state[:, 1:] - state[:, :-1]
        dphi_norm = dphi.reshape(dphi.shape[0], dphi.shape[1], -1).norm(dim=-1).mean()
        dx_norm = dx.reshape(dx.shape[0], dx.shape[1], -1).norm(dim=-1).mean()
    else:
        dphi_norm = phi.new_tensor(0.0)
        dx_norm = state.new_tensor(0.0)

    target_centered = state - state.mean(dim=(0, 1), keepdim=True)
    sse = (recon - state).pow(2).sum()
    sst = target_centered.pow(2).sum().clamp_min(eps)
    rec_r2 = 1.0 - sse / sst

    cob_enabled = bool(getattr(args, "potential_coboundary_loss", False))
    cob_loss = state.new_tensor(0.0)
    cob_mse = state.new_tensor(0.0)
    cob_rel_mse = state.new_tensor(0.0)
    cob_r2 = state.new_tensor(0.0)
    cob_corr = state.new_tensor(0.0)
    cob_target_norm = state.new_tensor(0.0)
    cob_pred_norm = state.new_tensor(0.0)
    cob_residual_frac = state.new_tensor(0.0)

    if cob_enabled:
        if state.dim() != 3 or state.shape[1] < 2:
            raise ValueError("potential_coboundary_loss requires sequence state with T>=2")
        if phi.dim() != 3:
            raise ValueError(f"potential vector must have shape [B,T,L], got {tuple(phi.shape)}")
        dv = phi[:, 1:] - phi[:, :-1]
        target_delta = _potential_coboundary_target(state, getattr(args, "potential_cob_target", "state_delta")).detach()

        # Vector-potential coboundary.  If target is data-space, decode Delta v.
        # If target is scalar, fall back to the squared norm of Delta v as a
        # scalar diagnostic target.
        raw_model = unwrap_model(model)
        if target_delta.dim() == state.dim():
            if not hasattr(raw_model, "decode_delta"):
                raise ValueError("Vector coboundary target requires PotentialAE.decode_delta")
            pred_delta = raw_model.decode_delta(dv)
        else:
            pred_delta = 0.5 * dv.pow(2).sum(dim=-1)

        pred_for_loss, target_for_loss = pred_delta, target_delta
        if bool(getattr(args, "potential_cob_standardize", False)):
            pred_for_loss, target_for_loss = _standardize_pair(pred_delta, target_delta, eps)

        resid = pred_for_loss - target_for_loss
        cob_mse = resid.pow(2).mean()
        denom = target_for_loss.pow(2).mean().clamp_min(eps)
        cob_rel_mse = cob_mse / denom
        cob_r2 = 1.0 - resid.pow(2).sum() / (target_for_loss - target_for_loss.mean()).pow(2).sum().clamp_min(eps)
        p0 = pred_for_loss - pred_for_loss.mean()
        t0 = target_for_loss - target_for_loss.mean()
        cob_corr = (p0 * t0).mean() / (p0.pow(2).mean().sqrt() * t0.pow(2).mean().sqrt()).clamp_min(eps)
        cob_target_norm = target_delta.reshape(target_delta.shape[0], -1).norm(dim=1).mean()
        cob_pred_norm = pred_delta.reshape(pred_delta.shape[0], -1).norm(dim=1).mean()
        cob_residual_frac = resid.reshape(resid.shape[0], -1).norm(dim=1).mean() / target_for_loss.reshape(target_for_loss.shape[0], -1).norm(dim=1).mean().clamp_min(eps)

        cob_loss_type = str(getattr(args, "potential_cob_loss_type", "rel_mse"))
        if cob_loss_type == "mse":
            cob_loss = cob_mse
        elif cob_loss_type == "corr":
            cob_loss = 1.0 - cob_corr
        else:
            cob_loss = cob_rel_mse

    loss = float(getattr(args, "potential_lambda_rec", 1.0)) * rec_loss
    loss = loss + float(getattr(args, "potential_lambda_cob", 1.0)) * cob_loss
    loss = loss + float(getattr(args, "potential_lambda_l2", 0.0)) * phi_l2
    loss = loss - float(getattr(args, "potential_lambda_var", 0.0)) * phi_var

    logs = {
        "loss": float(loss.detach()),
        "potential_rec_loss": float(rec_loss.detach()),
        "potential_rec_mse": float(rec_mse.detach()),
        "potential_rec_rel_l2": float(rec_rel.detach()),
        "potential_rec_r2": float(rec_r2.detach()),
        "potential_phi_l2": float(phi_l2.detach()),
        "potential_phi_var": float(phi_var.detach()),
        "potential_dphi_norm": float(dphi_norm.detach()),
        "potential_dx_norm": float(dx_norm.detach()),
        "potential_latent_dim": float(getattr(raw, "latent_dim", 0)),
        "potential_cob_enabled": float(cob_enabled),
        "potential_cob_loss": float(cob_loss.detach()),
        "potential_cob_mse": float(cob_mse.detach()),
        "potential_cob_rel_mse": float(cob_rel_mse.detach()),
        "potential_cob_r2": float(cob_r2.detach()),
        "potential_cob_corr": float(cob_corr.detach()),
        "potential_cob_target_norm": float(cob_target_norm.detach()),
        "potential_cob_pred_norm": float(cob_pred_norm.detach()),
        "potential_cob_residual_frac": float(cob_residual_frac.detach()),
        "potential_cob_lambda": float(getattr(args, "potential_lambda_cob", 0.0)) if cob_enabled else 0.0,
    }
    return loss, logs
