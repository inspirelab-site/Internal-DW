"""Rollout Jacobian Consistency (RJC) loss for standard AR models.

This module implements a *local* rollout regularizer:

    sum_{k=0}^{K-1} || J_theta(s_roll_k, u_k) v
                    - sg[J_theta(s_gt_k, u_k) v] ||^2

where J_theta is the one-step predictor Jacobian w.r.t. the history state.
The rollout histories are detached before computing the local JVP terms, so this
is NOT a K-step Jacobian product and does NOT backpropagate through the rollout
path like BPTT.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from internal_dw.data_utils.state_ops import append_time_point, time_window


def _as_next_frame(pred: torch.Tensor) -> torch.Tensor:
    """Convert model output [B,1,...] or [B,...] to [B,...]."""
    if pred.dim() >= 3 and pred.shape[1] == 1:
        return pred[:, 0]
    return pred


def _rms_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    flat = v.reshape(v.shape[0], -1)
    rms = torch.sqrt(flat.square().mean(dim=1, keepdim=True) + eps)
    while rms.dim() < v.dim():
        rms = rms.unsqueeze(-1)
    return v / rms


def _predict_next(raw, stim_window: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
    """One-step predictor output as a single next frame [B,...]."""
    out = raw(stim_window, history, return_aux=False)
    return _as_next_frame(out)


def _finite_difference_jvp(
    raw,
    stim_window: torch.Tensor,
    history: torch.Tensor,
    direction: torch.Tensor,
    eps_fd: float,
    target_branch: bool,
) -> torch.Tensor:
    """Approximate J(history) v with finite difference.

    target_branch=True uses no_grad and returns a detached target.  The rollout
    branch keeps parameter gradients through the two local forward passes.
    """
    direction = _rms_normalize(direction)
    if target_branch:
        with torch.no_grad():
            y0 = _predict_next(raw, stim_window, history)
            y1 = _predict_next(raw, stim_window, history + float(eps_fd) * direction)
            return ((y1 - y0) / float(eps_fd)).detach()
    y0 = _predict_next(raw, stim_window, history)
    y1 = _predict_next(raw, stim_window, history + float(eps_fd) * direction)
    return (y1 - y0) / float(eps_fd)


def _cosine_mean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    af = a.reshape(a.shape[0], -1)
    bf = b.reshape(b.shape[0], -1)
    return F.cosine_similarity(af, bf, dim=1).mean()


def compute_rollout_jacobian_consistency_loss(
    raw,
    state: torch.Tensor,
    stim: torch.Tensor,
    start_t: int,
    args,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute path-wise-sum RJC loss for one rollout start.

    Args:
        raw: unwrapped standard AR model with forward(stim_window, history).
        state: [B,T,...] ground-truth sequence.
        stim: [B,T,...] external input sequence, possibly zero dummy input.
        start_t: target time t.  Initial history is state[:, t-W:t].
        args: training args with rjc_* fields.

    Returns:
        loss: scalar path-wise sum over rollout steps, averaged over random dirs.
        logs: detached scalar diagnostics.
    """
    device = state.device
    W = int(getattr(args, "window_size", 1))
    T = int(state.shape[1])
    start_t = int(start_t)
    K_req = int(getattr(args, "rjc_horizon", 0))
    K = max(0, min(K_req, T - start_t))
    if K <= 0:
        z = state.new_tensor(0.0)
        return z, {
            "ar/rjc_loss": 0.0,
            "ar/rjc_step_mean": 0.0,
            "ar/rjc_mse_mean": 0.0,
            "ar/rjc_cos_mean": 0.0,
            "ar/rjc_steps": 0.0,
            "ar/rjc_dirs": float(getattr(args, "rjc_num_directions", 1)),
        }

    eps_fd = float(getattr(args, "rjc_eps_fd", 1e-2))
    num_dirs = max(1, int(getattr(args, "rjc_num_directions", 1)))
    loss_type = str(getattr(args, "rjc_loss_type", "mse"))

    # Rollout construction is explicitly no-grad.  This samples self-generated
    # states without retaining the graph from step 0 to step k.
    roll_history = time_window(state, start_t - W, start_t).detach().clone()

    step_losses = []
    step_mses = []
    step_coss = []

    for kk in range(K):
        target_t = start_t + kk
        if target_t - W < 0 or target_t >= T:
            break
        h_gt = time_window(state, target_t - W, target_t).detach()
        h_roll = roll_history.detach()
        stim_window = time_window(stim, target_t - W, target_t).detach()

        dir_losses = []
        dir_mses = []
        dir_coss = []
        for _ in range(num_dirs):
            v = _rms_normalize(torch.randn_like(h_gt))
            jv_gt = _finite_difference_jvp(
                raw=raw,
                stim_window=stim_window,
                history=h_gt,
                direction=v,
                eps_fd=eps_fd,
                target_branch=True,
            )
            jv_roll = _finite_difference_jvp(
                raw=raw,
                stim_window=stim_window,
                history=h_roll,
                direction=v,
                eps_fd=eps_fd,
                target_branch=False,
            )
            mse = F.mse_loss(jv_roll, jv_gt)
            cos = _cosine_mean(jv_roll, jv_gt)
            if loss_type == "mse":
                local_loss = mse
            elif loss_type == "cosine_mse":
                local_loss = mse + (1.0 - cos)
            else:
                raise ValueError(f"Unknown rjc_loss_type={loss_type}; expected mse or cosine_mse")
            dir_losses.append(local_loss)
            dir_mses.append(mse.detach())
            dir_coss.append(cos.detach())

        step_loss = torch.stack(dir_losses).mean()
        step_losses.append(step_loss)
        step_mses.append(torch.stack(dir_mses).mean())
        step_coss.append(torch.stack(dir_coss).mean())

        # Advance rollout history with detached prediction.
        with torch.no_grad():
            pred_next = _predict_next(raw, stim_window, h_roll).detach()
            roll_history = append_time_point(h_roll, pred_next).detach()

    if not step_losses:
        z = state.new_tensor(0.0)
        return z, {
            "ar/rjc_loss": 0.0,
            "ar/rjc_step_mean": 0.0,
            "ar/rjc_mse_mean": 0.0,
            "ar/rjc_cos_mean": 0.0,
            "ar/rjc_steps": 0.0,
            "ar/rjc_dirs": float(num_dirs),
        }

    # Important: path-wise SUM over k=1..K, not terminal-only and not a mean over k.
    loss_sum = torch.stack(step_losses).sum()
    steps = float(len(step_losses))
    logs = {
        "ar/rjc_loss": float(loss_sum.detach().cpu()),
        "ar/rjc_step_mean": float((loss_sum.detach() / max(steps, 1.0)).cpu()),
        "ar/rjc_mse_mean": float(torch.stack(step_mses).mean().detach().cpu()),
        "ar/rjc_cos_mean": float(torch.stack(step_coss).mean().detach().cpu()),
        "ar/rjc_steps": steps,
        "ar/rjc_dirs": float(num_dirs),
    }
    return loss_sum, logs
