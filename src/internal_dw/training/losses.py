import numpy as np
import torch
import torch.nn.functional as F

from internal_dw.utils import unwrap_model
from internal_dw.data_utils.state_ops import (
    append_time_point,
    corrcoef_flat,
    elementwise_state_loss,
    get_batch_time_shape,
    last_time_point,
    time_point,
    time_window,
    zero_external_input_like,
)


def safe_float(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu())
    return float(x)


def _horizons(args):
    hs = getattr(args, "koopman_horizons", [1])
    return sorted({int(h) for h in hs if int(h) >= 1})


def resolve_koopman_long_loss(args):
    """Resolve user-facing KG loss mode into the actual folded objective.

    Semantics:
      - supervised_folded: autonomous folded supervised KG loss.
      - stimulus_folded: controlled folded supervised KG loss. If the dataset has
        no external input, this automatically falls back to supervised_folded.
      - folded_auto: choose stimulus_folded if external input exists, otherwise supervised_folded.
      - both: gram_error + automatically selected folded supervised KG loss.
    """
    requested = str(getattr(args, "koopman_long_loss", "gram_error"))
    has_external_input = bool(getattr(args, "dataset_has_external_input", False))

    note = ""
    folded_component = None
    resolved = requested

    if requested == "folded_auto":
        resolved = "stimulus_folded" if has_external_input else "supervised_folded"
        folded_component = resolved
        note = f"folded_auto resolved to {resolved} because dataset_has_external_input={has_external_input}"
    elif requested == "both":
        resolved = "both"
        folded_component = "stimulus_folded" if has_external_input else "supervised_folded"
        note = f"both means gram_error + {folded_component} because dataset_has_external_input={has_external_input}"
    elif requested == "stimulus_folded" and not has_external_input:
        resolved = "supervised_folded"
        folded_component = "supervised_folded"
        note = "stimulus_folded requested, but dataset_has_external_input=False; falling back to supervised_folded"
    elif requested in ["supervised_folded", "stimulus_folded"]:
        folded_component = requested
        note = f"using explicit folded component: {requested}"
    elif requested in ["gram_error", "none"]:
        note = f"using {requested}"
    else:
        raise ValueError(f"Unknown koopman_long_loss={requested}")

    return {
        "requested": requested,
        "resolved": resolved,
        "folded_component": folded_component,
        "has_external_input": has_external_input,
        "note": note,
    }


def _quad_form(e, G):
    """e^T G e for dense or channelized latent/state tensors."""
    if G is None:
        return e.new_tensor(0.0)
    d = e.shape[-1]
    if G.dim() == 3:
        Ge = torch.einsum("bci,cij->bcj", e, G)
        return (Ge * e).sum(dim=-1).mean() / float(d)
    Ge = torch.matmul(e, G)
    return (Ge * e).sum(dim=-1).mean() / float(d)


@torch.no_grad()
def _koopman_powers_and_grams(raw, horizon, normalize=True):
    """Prefix powers A^k and Gramians G_K=sum_{j=0}^{K-1}(A^j)^T A^j.

    The returned powers follow the same orientation as F.linear / model.transition_latent.
    Works for dense A [d,d] and independent channel A [C,d,d].
    """
    horizon = int(max(horizon, 1))
    A = raw.effective_A().detach()
    if A.dim() == 3:
        C, d, _ = A.shape
        powers, grams, scales = [], [], []
        Ak = torch.eye(d, device=A.device, dtype=A.dtype).unsqueeze(0).repeat(C, 1, 1)
        G = torch.zeros(C, d, d, device=A.device, dtype=A.dtype)
        for _ in range(horizon):
            powers.append(Ak.clone())
            G = G + Ak.transpose(-2, -1) @ Ak
            scale = torch.diagonal(G, dim1=-2, dim2=-1).sum(-1) / float(d) if normalize else G.new_ones(C)
            scale = scale.clamp_min(1e-8)
            grams.append(G / scale.view(C, 1, 1))
            scales.append(scale)
            Ak = A @ Ak
        return powers, grams, scales

    d = A.shape[0]
    powers, grams, scales = [], [], []
    Ak = torch.eye(d, device=A.device, dtype=A.dtype)
    G = torch.zeros(d, d, device=A.device, dtype=A.dtype)
    for _ in range(horizon):
        powers.append(Ak.clone())
        G = G + Ak.transpose(0, 1) @ Ak
        scale = torch.trace(G) / float(d) if normalize else G.new_tensor(1.0)
        scale = scale.clamp_min(1e-8)
        grams.append(G / scale)
        scales.append(scale)
        Ak = A @ Ak
    return powers, grams, scales


@torch.no_grad()
def _future_stimulus_offsets(raw, stim, cur_t, window, K_eff):
    """Compute forced-rollout offsets c_k where A^k z + c_k predicts future states.

    c_0=0, c_k=A c_{k-1}+b_theta(s_{t+k-W:t+k}).
    Returned shape is [B,K,...], aligned with folded targets y_{cur_t+k}.
    """
    K_eff = int(max(K_eff, 1))
    A = raw.effective_A().detach()
    if getattr(raw, "channelized_koopman", False):
        state_shape = (stim.shape[0], raw.latent_channels, raw.latent_channel_dim)
    else:
        state_shape = (stim.shape[0], raw.latent_dim)
    c = stim.new_zeros(state_shape)
    offsets = [c.clone()]
    for kk in range(1, K_eff):
        tt = cur_t + kk
        stim_window = stim[:, tt - int(window):tt]
        force = raw.stimulus_force(stim_window)
        if A.dim() == 3:
            c = torch.einsum("bci,coi->bco", c, A) + force
        else:
            c = torch.matmul(c, A.t()) + force
        offsets.append(c.clone())
    return torch.stack(offsets, dim=1)


@torch.no_grad()
def _supervised_folded_targets(seq, cur_t, powers, grams, scales, K_eff, offsets=None):
    """Build G, q, const for folded supervised KG loss.

    It represents mean_k ||A^k yhat - y_{t+k}||_G-like target after folding:
        yhat^T G_K yhat - 2 yhat^T q_{t,K} + const.
    seq supports [B,T,D] or [B,T,C,d].
    """
    K_eff = int(max(K_eff, 1))
    G = grams[K_eff - 1]
    scale = scales[K_eff - 1]
    D = seq.shape[-1]
    q = seq[:, cur_t].new_zeros(seq[:, cur_t].shape)

    if G.dim() == 3:
        const = seq.new_zeros(())
        for kk in range(K_eff):
            y_future = seq[:, cur_t + kk]
            if offsets is not None:
                y_future = y_future - offsets[:, kk]
            q = q + torch.einsum("bci,cij->bcj", y_future, powers[kk])
            const = const + (y_future.pow(2).sum(dim=-1) / scale.view(1, -1)).mean() / float(D)
        q = q / scale.view(1, -1, 1)
        return G, q, const

    const = seq.new_zeros(())
    for kk in range(K_eff):
        y_future = seq[:, cur_t + kk]
        if offsets is not None:
            y_future = y_future - offsets[:, kk]
        q = q + torch.matmul(y_future, powers[kk])
        const = const + y_future.pow(2).sum(dim=-1).mean()
    q = q / scale
    const = const / (scale * float(D))
    return G, q, const


def _supervised_folded_loss_from_terms(yhat, G, q, const):
    D = yhat.shape[-1]
    if G.dim() == 3:
        Gy = torch.einsum("bci,cij->bcj", yhat, G)
        quad = (Gy * yhat).sum(dim=-1).mean() / float(D)
    else:
        Gy = torch.matmul(yhat, G)
        quad = (Gy * yhat).sum(dim=-1).mean() / float(D)
    linear = 2.0 * (yhat * q).sum(dim=-1).mean() / float(D)
    return quad - linear + const


def compute_koopman_gramian_loss(model, state, stim, args, epoch=0):
    """Koopman-Gram training loss.

    Supported long losses:
      - gram_error: one-step error weighted by G_K.
      - supervised_folded: folded long-horizon supervision using future fMRI/latent targets.
      - stimulus_folded: controlled supervised_folded with cumulative future stimulus forcing offsets.
      - folded_auto: auto-select supervised_folded or stimulus_folded based on dataset_has_external_input.
      - both: gram_error + auto-selected folded supervised KG loss.
      - none: disables the long KG term.
    """
    raw = unwrap_model(model)
    B, T, state_shape = get_batch_time_shape(state)
    if stim is None:
        stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))
    W = int(args.window_size)
    horizons = _horizons(args)
    Kmax = max(horizons) if horizons else 1
    stride = max(int(args.koopman_train_stride), 1)
    loss_cfg = resolve_koopman_long_loss(args)
    long_loss = loss_cfg["resolved"]
    requested_long_loss = loss_cfg["requested"]
    folded_component = loss_cfg["folded_component"]
    has_external_input = bool(loss_cfg["has_external_input"])
    gram_horizon = int(args.koopman_gramian_horizon)
    adaptive_horizon = bool(getattr(args, "koopman_adaptive_gramian_horizon", True))

    if T <= W:
        raise ValueError(f"Sequence too short: T={T}, window={W}")

    lam_one = float(args.koopman_lambda_one)
    lam_point = float(args.koopman_lambda_point)
    lam_delta = float(args.koopman_lambda_delta)
    lam_gram = float(args.koopman_lambda_gram)
    lam_rec = float(args.koopman_lambda_rec)
    lam_latent = float(args.koopman_lambda_latent)
    lam_ridge = float(getattr(args, "ridge_alpha", 0.0)) if bool(getattr(raw, "is_ridge_window", False)) else 0.0
    normalize_gram = bool(args.koopman_normalize_gram)
    detach_history = bool(args.koopman_detach_rollout_history)
    point_loss_name = "mse" if bool(getattr(raw, "is_ridge_window", False)) else "huber"

    need_gram = long_loss in ["gram_error", "supervised_folded", "stimulus_folded", "both"] and lam_gram != 0.0
    if need_gram:
        powers, grams, scales = _koopman_powers_and_grams(raw, gram_horizon, normalize=normalize_gram)
        G_full = grams[-1]
        if G_full.dim() == 3:
            gram_trace_mean = torch.diagonal(G_full, dim1=-2, dim2=-1).sum(-1).mean() / float(G_full.shape[-1])
            gram_diag_mean = torch.diagonal(G_full, dim1=-2, dim2=-1).mean()
            gram_diag_max = torch.diagonal(G_full, dim1=-2, dim2=-1).max()
        else:
            gram_trace_mean = torch.trace(G_full) / float(G_full.shape[0])
            gram_diag_mean = torch.diagonal(G_full).mean()
            gram_diag_max = torch.diagonal(G_full).max()
    else:
        powers = grams = scales = G_full = None
        gram_trace_mean = gram_diag_mean = gram_diag_max = state.new_tensor(0.0)

    last_start = T - Kmax + 1
    if last_start <= W:
        last_start = T
    starts = list(range(W, last_start, stride)) or [W]

    # Do not backprop through every possible start by default.
    # Full-start Koopman rollout keeps one computation graph per start and per
    # rollout step, which is extremely memory-heavy for field AE models.  This
    # mirrors the AR/Mamba trainer behavior: sample a bounded number of starts
    # during training, and use a deterministic subset during evaluation.
    max_starts = int(getattr(args, "koopman_train_starts_per_sequence", -1))
    if max_starts <= 0:
        max_starts = int(getattr(args, "ar_train_starts_per_sequence", -1))
    if max_starts > 0 and len(starts) > max_starts:
        if raw.training:
            perm = torch.randperm(len(starts), device=state.device)[:max_starts]
            chosen = sorted(int(i) for i in perm.detach().cpu().tolist())
            starts = [starts[i] for i in chosen]
        else:
            # Stable validation logs: evenly cover the sequence instead of random starts.
            chosen = torch.linspace(0, len(starts) - 1, steps=max_starts)
            chosen = torch.unique(chosen.round().long()).tolist()
            starts = [starts[int(i)] for i in chosen]

    losses = []
    logs = {}

    for t in starts:
        history = time_window(state, t - W, t)
        point_losses, delta_losses, rec_losses, latent_losses = [], [], [], []
        gram_error_losses, supfold_losses, long_losses = [], [], []
        endpoint_corrs, used_future_horizons = [], []
        corrector_abs_vals, corrector_next_abs_vals = [], []
        corrector_diff_abs_vals, corrector_delta_rel_vals = [], []

        for k in range(1, Kmax + 1):
            cur_t = t + k - 1
            if cur_t >= T or cur_t - W < 0:
                break

            stim_window = time_window(stim, cur_t - W, cur_t)
            target = time_point(state, cur_t, keep_time=True)
            pred, aux = raw(stim_window, history, return_aux=True)
            if isinstance(aux, dict) and "corrector_abs" in aux:
                corrector_abs_vals.append(aux["corrector_abs"].detach())
                corrector_next_abs_vals.append(aux.get("corrector_next_abs", aux["corrector_abs"]).detach())
                corrector_diff_abs_vals.append(aux.get("corrector_diff_abs", aux["corrector_abs"].new_tensor(0.0)).detach())
                corrector_delta_rel_vals.append(aux.get("corrector_delta_rel", aux["corrector_abs"].new_tensor(0.0)).detach())

            if k in horizons:
                point_losses.append(elementwise_state_loss(pred, target, loss=point_loss_name, delta=20.0))
                prev_frame = last_time_point(history, keep_time=True)
                delta_losses.append(elementwise_state_loss(pred - prev_frame, target - prev_frame, loss="huber", delta=20.0))
                endpoint_corrs.append(corrcoef_flat(pred.detach(), target.detach()))

            if bool(getattr(raw, "use_koopman_encoder", False)):
                rec_losses.append(elementwise_state_loss(aux["recon"], last_time_point(history, keep_time=True), loss="huber", delta=20.0))
                with torch.no_grad():
                    target_window = time_window(state, cur_t - W + 1, cur_t + 1)
                    z_target = raw.encode_state(target_window)
                yhat = aux["z_next"]
                e = yhat - z_target
                latent_losses.append(F.mse_loss(yhat, z_target))
            elif bool(getattr(raw, "kg_latent_is_window", False)):
                # Window-RR uses the flattened history window as the latent state.
                # The point loss is still computed on the decoded next frame, but
                # KG/folded losses propagate errors in the companion latent state.
                with torch.no_grad():
                    target_window = time_window(state, cur_t - W + 1, cur_t + 1)
                    z_target = raw.encode_state(target_window)
                yhat = aux["z_next"]
                e = yhat - z_target
                latent_losses.append(F.mse_loss(yhat, z_target))
            else:
                yhat = pred[:, 0]
                e = yhat - target[:, 0]

            if need_gram and long_loss in ["gram_error", "both"]:
                gram_error_losses.append(_quad_form(e, G_full))

            if need_gram and long_loss in ["supervised_folded", "stimulus_folded", "both"]:
                remaining = T - cur_t
                K_eff = min(gram_horizon, remaining) if adaptive_horizon else gram_horizon
                K_eff = int(max(K_eff, 1))
                if cur_t + K_eff <= T:
                    if bool(getattr(raw, "use_koopman_encoder", False)) or bool(getattr(raw, "kg_latent_is_window", False)):
                        latent_future = []
                        with torch.no_grad():
                            for jj in range(cur_t, cur_t + K_eff):
                                win_j = time_window(state, jj - W + 1, jj + 1)
                                latent_future.append(raw.encode_state(win_j))
                            folded_seq = torch.stack(latent_future, dim=1)
                        folded_cur_t = 0
                    else:
                        folded_seq = state
                        folded_cur_t = cur_t

                    offsets = None
                    if folded_component == "stimulus_folded":
                        if not has_external_input:
                            raise ValueError("Internal error: stimulus folded component selected without external input.")
                        offsets = _future_stimulus_offsets(raw, stim, cur_t, W, K_eff)

                    G_eff, q_eff, const_eff = _supervised_folded_targets(
                        seq=folded_seq,
                        cur_t=folded_cur_t,
                        powers=powers,
                        grams=grams,
                        scales=scales,
                        K_eff=K_eff,
                        offsets=offsets,
                    )
                    supfold = _supervised_folded_loss_from_terms(yhat, G_eff, q_eff, const_eff)
                    supfold = supfold / max(float(K_eff), 1.0)
                    supfold_losses.append(supfold)
                    used_future_horizons.append(float(K_eff))

            if long_loss == "gram_error" and gram_error_losses:
                long_losses.append(gram_error_losses[-1])
            elif long_loss in ["supervised_folded", "stimulus_folded"] and supfold_losses:
                long_losses.append(supfold_losses[-1])
            elif long_loss == "both":
                parts = []
                if gram_error_losses:
                    parts.append(gram_error_losses[-1])
                if supfold_losses:
                    parts.append(supfold_losses[-1])
                if parts:
                    long_losses.append(torch.stack(parts).sum())

            next_frame = pred.detach() if detach_history else pred
            history = append_time_point(history, next_frame)

        if not point_losses:
            continue

        loss_point = torch.stack(point_losses).mean()
        loss_one = point_losses[0]
        loss_delta = torch.stack(delta_losses).mean() if delta_losses else loss_point.new_tensor(0.0)
        loss_rec = torch.stack(rec_losses).mean() if rec_losses else loss_point.new_tensor(0.0)
        loss_latent = torch.stack(latent_losses).mean() if latent_losses else loss_point.new_tensor(0.0)
        loss_ridge = raw.ridge_l2_penalty() if bool(getattr(raw, "is_ridge_window", False)) else loss_point.new_tensor(0.0)
        loss_gram_error = torch.stack(gram_error_losses).mean() if gram_error_losses else loss_point.new_tensor(0.0)
        loss_supfold = torch.stack(supfold_losses).mean() if supfold_losses else loss_point.new_tensor(0.0)
        loss_long = torch.stack(long_losses).mean() if long_losses else loss_point.new_tensor(0.0)
        corrector_abs_mean = torch.stack(corrector_abs_vals).mean() if corrector_abs_vals else loss_point.new_tensor(0.0)
        corrector_next_abs_mean = torch.stack(corrector_next_abs_vals).mean() if corrector_next_abs_vals else loss_point.new_tensor(0.0)
        corrector_diff_abs_mean = torch.stack(corrector_diff_abs_vals).mean() if corrector_diff_abs_vals else loss_point.new_tensor(0.0)
        corrector_delta_rel_mean = torch.stack(corrector_delta_rel_vals).mean() if corrector_delta_rel_vals else loss_point.new_tensor(0.0)

        loss = (
            lam_one * loss_one
            + lam_point * loss_point
            + lam_delta * loss_delta
            + lam_rec * loss_rec
            + lam_latent * loss_latent
            + lam_ridge * loss_ridge
            + lam_gram * loss_long
        )
        losses.append(loss)

        step_logs = {
            "loss": safe_float(loss),
            "koop/loss_one": safe_float(loss_one),
            "koop/loss_point": safe_float(loss_point),
            "koop/loss_delta": safe_float(loss_delta),
            "koop/loss_rec": safe_float(loss_rec),
            "koop/loss_latent": safe_float(loss_latent),
            "koop/loss_gram_error": safe_float(loss_gram_error),
            "koop/loss_supfold": safe_float(loss_supfold),
            "koop/loss_long": safe_float(loss_long),
            "koop/loss_ridge_l2": safe_float(loss_ridge),
            "koop/one_step_corr": safe_float(torch.stack(endpoint_corrs).mean()) if endpoint_corrs else 0.0,
            "koop/used_future_horizon": float(np.mean(used_future_horizons)) if used_future_horizons else 0.0,
            "koop/corrector_abs": safe_float(corrector_abs_mean),
            "koop/corrector_next_abs": safe_float(corrector_next_abs_mean),
            "koop/corrector_diff_abs": safe_float(corrector_diff_abs_mean),
            "koop/corrector_delta_rel": safe_float(corrector_delta_rel_mean),
        }
        for key, val in step_logs.items():
            logs.setdefault(key, []).append(val)

    if not losses:
        return state.new_tensor(0.0, requires_grad=True), {}

    total = torch.stack(losses).mean()
    log_dict = {k: float(np.mean(v)) if v else 0.0 for k, v in logs.items()}
    log_dict.update({
        "loss": safe_float(total),
        "koop/Kmax": float(Kmax),
        "koop/gramian_horizon": float(gram_horizon),
        "koop/long_loss_code": float({"none": 0, "gram_error": 1, "supervised_folded": 2, "stimulus_folded": 3, "both": 4}.get(long_loss, -1)),
        "koop/folded_component_code": float({None: 0, "supervised_folded": 1, "stimulus_folded": 2}.get(folded_component, -1)),
        "koop/dataset_has_external_input": float(has_external_input),
        "koop/folded_future_stimulus": float(folded_component == "stimulus_folded"),
        "koop/lambda_delta": float(lam_delta),
        "koop/lambda_rec": float(lam_rec),
        "koop/lambda_latent": float(lam_latent),
        "koop/lambda_ridge_l2": float(lam_ridge),
        "koop/lambda_gram": float(lam_gram),
        "koop/gram_trace_mean": safe_float(gram_trace_mean),
        "koop/gram_diag_mean": safe_float(gram_diag_mean),
        "koop/gram_diag_max": safe_float(gram_diag_max),
        "koop/A_sigma": safe_float(torch.linalg.matrix_norm(raw.effective_A().detach(), ord=2).mean()),
        "koop/force_scale": safe_float(torch.exp(raw.log_force_scale.detach())),
    })
    return total, log_dict
