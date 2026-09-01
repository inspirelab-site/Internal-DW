"""Utilities for Stage-B joint potential/residual AR with frozen Stage A.

Stage A learns a vector potential

    v_t = V_A(x_t),    x_t^Phi = D_A(v_t),    r_t = x_t - x_t^Phi.

Stage B freezes Stage A and trains an AR backbone on the joint state

    s_t = [v_t, r_t].

At rollout time, the backbone predicts s_{t+1}=[v_{t+1}, r_{t+1}], and the
observable prediction is reconstructed as

    x_{t+1} = D_A(v_{t+1}) + r_{t+1}.

This uses the potential state as a recurrent AR state; it is not a static
subtraction of an AE reconstruction.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from internal_dw.models.potential_ae import PotentialAEVectorModel
from internal_dw.utils import load_checkpoint

_STAGEB_CACHE: Dict[Tuple[str, int, int, int, int, float, int, int], PotentialAEVectorModel] = {}


def _orig_state_dim_from_args(args) -> int:
    d = int(getattr(args, "stageb_original_roi_dim", 0))
    if d <= 0:
        # During standalone utilities/tests, roi_dim may still be the original dim.
        if bool(getattr(args, "stageb_joint_potential_ar", False)):
            d = int(getattr(args, "roi_dim", 0)) - int(getattr(args, "potential_latent_dim", 64))
        else:
            d = int(getattr(args, "roi_dim", 0))
    if d <= 0:
        raise ValueError("Stage-B requires a positive original state dimension")
    return d


def get_stageb_potential_model(args, device: torch.device | int) -> PotentialAEVectorModel:
    path = str(getattr(args, "stageb_potential_ckpt_path", ""))
    if not path:
        raise ValueError("Stage-B requires --stageb_potential_ckpt_path")

    if isinstance(device, int):
        device_key = int(device)
        torch_device = torch.device(f"cuda:{device}")
    else:
        torch_device = torch.device(device)
        device_key = torch_device.index if torch_device.index is not None else -1

    latent = int(getattr(args, "potential_latent_dim", 64))
    hidden = int(getattr(args, "potential_hidden_dim", 256))
    depth = int(getattr(args, "potential_depth", 2))
    dropout = float(getattr(args, "potential_dropout", 0.0))
    state_dim = _orig_state_dim_from_args(args)
    input_dim = int(getattr(args, "stim_dim", 0))
    stim_context_len = int(getattr(args, "potential_stim_context_len", 1))
    key = (path, device_key, latent, hidden, depth, dropout, state_dim, input_dim, stim_context_len)
    if key in _STAGEB_CACHE:
        return _STAGEB_CACHE[key]

    model = PotentialAEVectorModel(
        state_dim=state_dim,
        latent_dim=latent,
        hidden_dim=hidden,
        depth=depth,
        dropout=dropout,
        potential_hidden_dim=getattr(args, "potential_value_hidden_dim", None),
        potential_depth=getattr(args, "potential_value_depth", None),
        input_dim=input_dim,
        stim_hidden_dim=getattr(args, "potential_stim_hidden_dim", None),
        stim_depth=getattr(args, "potential_stim_depth", None),
        stim_context_len=stim_context_len,
    ).to(torch_device)
    load_checkpoint(model, path, map_location=torch_device, strict=not bool(getattr(args, "stageb_potential_non_strict_ckpt", False)))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _STAGEB_CACHE[key] = model
    return model


@torch.no_grad()
def potential_decompose(state: torch.Tensor, args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return potential reconstruction x^Phi, residual r, vector potential v."""
    pot = get_stageb_potential_model(args, state.device)
    recon, v, _ = pot(state)
    residual = state - recon
    return recon.detach(), residual.detach(), v.detach()


@torch.no_grad()
def make_joint_state(state: torch.Tensor, args) -> tuple[torch.Tensor, dict[str, float]]:
    recon, residual, v = potential_decompose(state, args)
    joint = torch.cat([v, residual], dim=-1)
    logs = stageb_decomposition_logs(state, recon, residual, v)
    return joint.detach(), logs


def split_joint(joint: torch.Tensor, args) -> tuple[torch.Tensor, torch.Tensor]:
    latent = int(getattr(args, "potential_latent_dim", 64))
    v = joint[..., :latent]
    r = joint[..., latent:]
    return v, r


def decode_joint(joint: torch.Tensor, args) -> torch.Tensor:
    pot = get_stageb_potential_model(args, joint.device)
    v, r = split_joint(joint, args)
    return pot.decode(v) + r


def stageb_prediction_to_next_joint(pred_frame: torch.Tensor, prev_joint: torch.Tensor, args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert raw Stage-B network output to the next joint state.

    The clean Stage-B design treats the first latent_dim outputs as a potential
    increment, not an absolute potential state:

        delta_v_hat = output[..., :latent_dim]
        v_hat_next = v_prev + delta_v_hat
        r_hat_next = output[..., latent_dim:]
        joint_hat_next = [v_hat_next, r_hat_next]

    This prevents the residual branch from silently redefining the whole state: it
    is trained against the frozen Stage-A residual target only, while the
    potential branch explicitly models the coboundary/increment coordinate.

    Returns:
        joint_hat_next, delta_v_hat, v_hat_next, r_hat_next
    """
    prev_v, _prev_r = split_joint(prev_joint, args)
    out_v, out_r = split_joint(pred_frame, args)
    if bool(getattr(args, "stageb_predict_delta_v", True)):
        delta_v_hat = out_v
        v_hat_next = prev_v + delta_v_hat
    else:
        v_hat_next = out_v
        delta_v_hat = v_hat_next - prev_v
    joint_hat_next = torch.cat([v_hat_next, out_r], dim=-1)
    return joint_hat_next, delta_v_hat, v_hat_next, out_r


def stageb_decomposition_logs(state: torch.Tensor, recon: torch.Tensor, residual: torch.Tensor, v: torch.Tensor) -> dict[str, float]:
    B = state.shape[0]
    flat_x = state.reshape(B, -1)
    flat_res = residual.reshape(B, -1)
    denom = flat_x.pow(2).sum(dim=1).clamp_min(1e-8)
    rec_rel = torch.sqrt(flat_res.pow(2).sum(dim=1) / denom).mean()
    rec_mse = (recon - state).pow(2).mean()
    var = (state - state.mean()).pow(2).mean().clamp_min(1e-8)
    rec_r2 = 1.0 - rec_mse / var
    res_norm = flat_res.norm(dim=1).mean()
    x_norm = flat_x.norm(dim=1).mean().clamp_min(1e-8)
    return {
        "ar/stageb_potential_rec_r2": float(rec_r2.detach()),
        "ar/stageb_potential_rec_rel_l2": float(rec_rel.detach()),
        "ar/stageb_residual_norm_ratio": float((res_norm / x_norm).detach()),
        "ar/stageb_phi_norm": float(v.reshape(v.shape[0], -1).norm(dim=1).mean().detach()),
    }


# Backward-compatible static residualization used by old scripts.
@torch.no_grad()
def potential_reconstruction_and_residual(state: torch.Tensor, args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return potential_decompose(state, args)


@torch.no_grad()
def stageb_residualize_state(state: torch.Tensor, args) -> tuple[torch.Tensor, Dict[str, float]]:
    recon, residual, v = potential_decompose(state, args)
    return residual.detach(), stageb_decomposition_logs(state, recon, residual, v)


# -----------------------------------------------------------------------------
# New Stage-B mode for stimulus-driven potential + residual AR
# -----------------------------------------------------------------------------

def potential_encode_state(state: torch.Tensor, args) -> torch.Tensor:
    """Encode x -> v using frozen Stage A. Output is detached."""
    pot = get_stageb_potential_model(args, state.device)
    with torch.no_grad():
        return pot.encode(state).detach()


def potential_decode_v(v: torch.Tensor, args) -> torch.Tensor:
    """Decode v -> x^Phi using frozen Stage A. Output is detached."""
    pot = get_stageb_potential_model(args, v.device)
    with torch.no_grad():
        return pot.decode(v).detach()


def potential_stim_delta(stim: torch.Tensor, args, like_v: torch.Tensor | None = None) -> torch.Tensor:
    """Compute G(u) using frozen Stage A. Output is detached.

    If ``stim`` is a full causal window [B,L,S], prefer
    ``potential_stim_delta_context`` to avoid ambiguity with a length-L sequence.
    """
    pot = get_stageb_potential_model(args, stim.device if stim is not None else like_v.device)
    with torch.no_grad():
        return pot.stim_delta(stim, like_v=like_v).detach()


def potential_stim_delta_context(stim_window: torch.Tensor, args, like_v: torch.Tensor | None = None) -> torch.Tensor:
    """Compute one G(u_{t-L+1:t}) from a causal stimulus window."""
    pot = get_stageb_potential_model(args, stim_window.device if stim_window is not None else like_v.device)
    with torch.no_grad():
        return pot.stim_delta_context_window(stim_window, like_v=like_v).detach()


def stageb_stim_context_window(stim: torch.Tensor, target_t: int, context_len: int) -> torch.Tensor:
    """Return stimulus context for predicting x[target_t] from x[target_t-1].

    Uses transition stimuli ending at index target_t-1:
    [u_{target_t-L}, ..., u_{target_t-1}], with left zero padding.
    Supports tokenized stimulus tensors because padding preserves trailing dims.
    """
    L = max(1, int(context_len))
    end = max(0, min(int(target_t), int(stim.shape[1])))
    start = max(0, end - L)
    ctx = stim[:, start:end]
    missing = L - int(ctx.shape[1])
    if missing > 0:
        pad_shape = (stim.shape[0], missing, *stim.shape[2:])
        pad = stim.new_zeros(pad_shape)
        ctx = torch.cat([pad, ctx], dim=1)
    return ctx


@torch.no_grad()
def stimulus_potential_decomposition_logs(state: torch.Tensor, args) -> dict[str, float]:
    """Static V/D reconstruction logs for a frozen stimulus-potential model."""
    pot = get_stageb_potential_model(args, state.device)
    v = pot.encode(state)
    recon = pot.decode(v)
    residual = state - recon
    return stageb_decomposition_logs(state, recon, residual, v)


@torch.no_grad()
def stimulus_potential_rollout_from_context(
    state: torch.Tensor,
    stim: torch.Tensor,
    args,
    start_t: int,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll frozen potential from x_{start_t} for ``horizon`` steps.

    Returns:
        v_seq: [B,horizon,L] with v_{start_t+1 : start_t+horizon}
        x_phi_seq: [B,horizon,D] decoded potential component.
    """
    pot = get_stageb_potential_model(args, state.device)
    v = pot.encode(state[:, start_t])
    vs = []
    xs = []
    max_t = stim.shape[1]
    for i in range(horizon):
        target_t = start_t + i + 1
        ctx = stageb_stim_context_window(stim, target_t, int(getattr(args, "potential_stim_context_len", 1)))
        dv = pot.stim_delta_context_window(ctx)
        v = v + dv
        vs.append(v)
        xs.append(pot.decode(v))
    return torch.stack(vs, dim=1).detach(), torch.stack(xs, dim=1).detach()
