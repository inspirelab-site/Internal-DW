import json
import os

import numpy as np
import torch

from internal_dw.training.losses import compute_koopman_gramian_loss
from internal_dw.training.ar_losses import compute_autoregressive_one_step_loss, compute_potential_ae_loss
from internal_dw.data_utils.state_ops import (
    corrcoef_flat,
    get_batch_time_shape,
    time_window,
    append_time_point,
    unpack_batch,
    zero_external_input_like,
)
from internal_dw.evaluation.metrics import relative_l2, trajectory_relative_l2, trajectory_corrcoef_flat
from internal_dw.utils import unwrap_model
from internal_dw.training.potential_stageb import stageb_residualize_state, make_joint_state, decode_joint, stageb_prediction_to_next_joint


@torch.no_grad()
def evaluate_loss(model, dataloader, args, rank=0, prefix="val"):
    model.eval()
    vals = []
    logs_accum = {}
    n = 0
    for batch in dataloader:
        state, stim, _, _ = unpack_batch(batch)
        state = state.cuda(rank, non_blocking=True).float()
        if stim is None:
            stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))
        stim = stim.cuda(rank, non_blocking=True).float()
        raw = unwrap_model(model)
        if getattr(raw, "is_potential_ae", False):
            loss, logs = compute_potential_ae_loss(model, state, stim, args)
        elif getattr(raw, "is_standard_autoregressive", False):
            loss, logs = compute_autoregressive_one_step_loss(model, state, stim, args)
        else:
            loss, logs = compute_koopman_gramian_loss(model, state, stim, args)
        vals.append(float(loss.detach().cpu()))
        for k, v in logs.items():
            logs_accum[k] = logs_accum.get(k, 0.0) + float(v)
        n += 1

    # Prefer the loss reported by the loss function itself when available.
    # This matters for objectives whose validation/checkpoint metric should be
    # the full training objective, e.g. recurrent PEH cell closure:
    #   logs["loss"] = L_BPTT + lambda_cell L_cell.
    # Some older paths returned the base scalar while still logging the full
    # objective, which made val/loss and best.pth selection silently optimize
    # only the local BPTT term.  The returned scalar is still logged for sanity.
    returned_loss = sum(vals) / max(len(vals), 1)
    if n > 0 and "loss" in logs_accum:
        objective_loss = logs_accum["loss"] / max(n, 1)
    else:
        objective_loss = returned_loss

    out = {f"{prefix}/loss": objective_loss}
    if abs(float(objective_loss) - float(returned_loss)) > 1e-12:
        out[f"{prefix}/loss_returned_scalar"] = returned_loss
    for k, v in logs_accum.items():
        out[f"{prefix}/{k}"] = v / max(n, 1)
    return out


@torch.no_grad()
def evaluate_horizon_sweep(model, dataloader, args, rank=0, prefix="test"):
    """
    Evaluate autoregressive rollout with *refresh horizons*.

    IMPORTANT SEMANTICS:
      horizon H means the ground-truth state is injected into the rollout
      history every H prediction steps to clip accumulated error.

    Therefore:
      H=1  -> teacher-forced one-step evaluation at every step
      H=8  -> free rollout for 8 steps, then replace the last prediction with GT
      H=64 -> free rollout for 64 steps, then replace with GT

    This is different from a lead-time metric where horizon h means only the
    h-th future prediction is scored.  The old evaluator used the lead-time
    meaning; this function uses the refresh-interval meaning used in the HCP
    experiments.
    """
    model.eval()
    raw = unwrap_model(model)

    if getattr(raw, "is_potential_ae", False):
        return evaluate_potential_ae(model, dataloader, args, rank=rank, prefix=prefix)

    if getattr(raw, "is_standard_autoregressive", False):
        if getattr(raw, "is_recurrent_state_ar", False):
            out = evaluate_recurrent_state_horizon_sweep(model, dataloader, args, rank=rank, prefix=prefix)
            out.update(evaluate_recurrent_state_free_rollout(model, dataloader, args, rank=rank, prefix=prefix))
            return out
        # Existing metrics: one-step autoregressive evaluator.  For
        # path_generator_field this intentionally calls forward(), which returns
        # only the first generated frame, so these metrics measure the old
        # one-step AR-style rollout behavior.
        out = evaluate_standard_ar_horizon_sweep(model, dataloader, args, rank=rank, prefix=prefix)
        if bool(getattr(args, "eval_free_rollout_curves", True)):
            out.update(evaluate_standard_ar_free_rollout_curves(model, dataloader, args, rank=rank, prefix=prefix))
        if _should_eval_field_long_rollout(args):
            out.update(evaluate_field_long_rollout(model, dataloader, args, rank=rank, prefix=prefix))

        # New metrics for finite-horizon path generators: use generate_path() to
        # produce K frames at a time, then roll out chunk by chunk.  These are
        # the metrics that correspond to the actual segment-generator idea.
        if getattr(raw, "is_path_generator", False):
            out.update(evaluate_path_generator_chunked_horizon_sweep(model, dataloader, args, rank=rank, prefix=prefix))
            if _should_eval_field_long_rollout(args):
                out.update(evaluate_path_generator_chunked_long_rollout(model, dataloader, args, rank=rank, prefix=prefix))
        return out

    return evaluate_koopman_refresh_horizon_sweep(model, dataloader, args, rank=rank, prefix=prefix)


@torch.no_grad()
def evaluate_potential_ae(model, dataloader, args, rank=0, prefix="test"):
    """Report reconstruction metrics for Stage-A potential AE."""
    model.eval()
    vals = []
    r2s = []
    rels = []
    mses = []
    cob_r2s = []
    cob_corrs = []
    cob_losses = []
    cob_fracs = []
    for batch in dataloader:
        state, stim, _, _ = unpack_batch(batch)
        state = state.cuda(rank, non_blocking=True).float()
        if stim is None:
            stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))
        stim = stim.cuda(rank, non_blocking=True).float()
        loss, logs = compute_potential_ae_loss(model, state, stim, args)
        vals.append(float(loss.detach().cpu()))
        r2s.append(float(logs.get("potential_rec_r2", 0.0)))
        rels.append(float(logs.get("potential_rec_rel_l2", 0.0)))
        mses.append(float(logs.get("potential_rec_mse", 0.0)))
        cob_r2s.append(float(logs.get("potential_cob_r2", 0.0)))
        cob_corrs.append(float(logs.get("potential_cob_corr", 0.0)))
        cob_losses.append(float(logs.get("potential_cob_loss", 0.0)))
        cob_fracs.append(float(logs.get("potential_cob_residual_frac", 0.0)))
    return {
        f"{prefix}/potential_loss": float(np.mean(vals)) if vals else 0.0,
        f"{prefix}/potential_rec_r2": float(np.mean(r2s)) if r2s else 0.0,
        f"{prefix}/potential_rec_rel_l2": float(np.mean(rels)) if rels else 0.0,
        f"{prefix}/potential_rec_mse": float(np.mean(mses)) if mses else 0.0,
        f"{prefix}/potential_cob_r2": float(np.mean(cob_r2s)) if cob_r2s else 0.0,
        f"{prefix}/potential_cob_corr": float(np.mean(cob_corrs)) if cob_corrs else 0.0,
        f"{prefix}/potential_cob_loss": float(np.mean(cob_losses)) if cob_losses else 0.0,
        f"{prefix}/potential_cob_residual_frac": float(np.mean(cob_fracs)) if cob_fracs else 0.0,
    }


@torch.no_grad()
def evaluate_koopman_refresh_horizon_sweep(model, dataloader, args, rank=0, prefix="test"):
    """Refresh-horizon evaluation for Koopman/Ridge-style models.

    For each H in args.test_horizons, run through the sequence once.  The model
    is always asked to predict the next frame.  Its prediction is scored at each
    step.  The rollout history is updated with the prediction except every H
    steps, where the ground-truth target frame is inserted instead.
    """
    model.eval()
    raw = unwrap_model(model)
    W = int(args.window_size)
    horizons = sorted({int(h) for h in args.test_horizons if int(h) >= 1})
    corr_by_h = {h: [] for h in horizons}
    rel_l2_by_h = {h: [] for h in horizons}

    for batch in dataloader:
        state, stim, _, _ = unpack_batch(batch)
        state = state.cuda(rank, non_blocking=True).float()
        if stim is None:
            stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))
        stim = stim.cuda(rank, non_blocking=True).float()
        if bool(getattr(args, "stageb_potential_residual", False)):
            state, _stageb_logs = stageb_residualize_state(state, args)
        _B, T, _state_shape = get_batch_time_shape(state)
        if T <= W:
            continue

        for H in horizons:
            history = time_window(state, 0, W).clone()
            steps_since_refresh = 0
            preds = []
            targets = []

            for target_t in range(W, T):
                # Keep the same feature convention as training: predict x[target_t]
                # from the previous W external inputs [target_t-W, target_t).
                stim_win = time_window(stim, target_t - W, target_t)
                z = raw.encode_state(history)
                z_next = raw.transition_latent(z, stim_win)
                pred_frame = raw.decode_state(z_next)
                target_frame = state[:, target_t]

                # Save the model prediction for scoring.  The GT refresh below is
                # used only to update the future history; it must not replace the
                # saved prediction trajectory.
                preds.append(pred_frame.unsqueeze(1))
                targets.append(target_frame.unsqueeze(1))

                steps_since_refresh += 1
                if steps_since_refresh >= H:
                    next_frame = target_frame
                    steps_since_refresh = 0
                else:
                    next_frame = pred_frame
                history = append_time_point(history, next_frame)

            if preds:
                pred_seq = torch.cat(preds, dim=1)
                target_seq = torch.cat(targets, dim=1)
                corr_by_h[H].append(trajectory_corrcoef_flat(pred_seq, target_seq).detach().cpu())
                rel_l2_by_h[H].append(trajectory_relative_l2(pred_seq, target_seq).detach().cpu())

    out = {}
    for h, vals in corr_by_h.items():
        out[f"{prefix}/horizon_{h}_corr"] = float(torch.stack(vals).mean()) if vals else 0.0
    for h, vals in rel_l2_by_h.items():
        out[f"{prefix}/horizon_{h}_rel_l2"] = float(torch.stack(vals).mean()) if vals else 0.0
    return out



@torch.no_grad()
def evaluate_stageb_joint_horizon_sweep(model, dataloader, args, rank=0, prefix="test"):
    model.eval()
    W = int(args.window_size)
    horizons = sorted({int(h) for h in args.test_horizons if int(h) >= 1})
    corr_by_h = {h: [] for h in horizons}
    rel_l2_by_h = {h: [] for h in horizons}

    for batch in dataloader:
        state, stim, _, _ = unpack_batch(batch)
        state = state.cuda(rank, non_blocking=True).float()
        if stim is None:
            stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))
        stim = stim.cuda(rank, non_blocking=True).float()
        joint, _ = make_joint_state(state, args)
        _B, T, _ = get_batch_time_shape(state)
        if T <= W:
            continue
        for H in horizons:
            history = time_window(joint, 0, W).clone()
            steps_since_refresh = 0
            preds, targets = [], []
            for target_t in range(W, T):
                stim_win = time_window(stim, target_t - W, target_t)
                pred = model(stim_win, history, return_aux=False)
                pred_frame = pred[:, 0] if pred.dim() == joint.dim() else pred
                pred_joint, _delta_v, _v_next, _r_next = stageb_prediction_to_next_joint(pred_frame, history[:, -1], args)
                x_pred = decode_joint(pred_joint, args)
                x_target = state[:, target_t]
                preds.append(x_pred.unsqueeze(1)); targets.append(x_target.unsqueeze(1))
                steps_since_refresh += 1
                if steps_since_refresh >= H:
                    next_joint = joint[:, target_t]
                    steps_since_refresh = 0
                else:
                    next_joint = pred_joint
                history = append_time_point(history, next_joint)
            if preds:
                pred_seq = torch.cat(preds, dim=1)
                target_seq = torch.cat(targets, dim=1)
                corr_by_h[H].append(trajectory_corrcoef_flat(pred_seq, target_seq).detach().cpu())
                rel_l2_by_h[H].append(trajectory_relative_l2(pred_seq, target_seq).detach().cpu())
    out = {}
    for h, vals in corr_by_h.items():
        out[f"{prefix}/horizon_{h}_corr"] = float(torch.stack(vals).mean()) if vals else 0.0
    for h, vals in rel_l2_by_h.items():
        out[f"{prefix}/horizon_{h}_rel_l2"] = float(torch.stack(vals).mean()) if vals else 0.0
    return out


@torch.no_grad()
def evaluate_stageb_joint_free_rollout_curves(model, dataloader, args, rank=0, prefix="test"):
    model.eval()
    W = int(args.window_size)
    corr_curves=[]; rel_l2_curves=[]; err_norm_curves=[]; err_sq_curves=[]; target_norm_curves=[]; pred_lens=[]
    for batch in dataloader:
        state, stim, _, _ = unpack_batch(batch)
        state = state.cuda(rank, non_blocking=True).float()
        if stim is None:
            stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))
        stim = stim.cuda(rank, non_blocking=True).float()
        joint, _ = make_joint_state(state, args)
        _B,T,_ = get_batch_time_shape(state)
        if T <= W: continue
        history = time_window(joint,0,W).clone(); preds=[]; targets=[]
        for target_t in range(W,T):
            stim_win=time_window(stim,target_t-W,target_t)
            pred=model(stim_win,history,return_aux=False)
            pred_frame=pred[:,0] if pred.dim()==joint.dim() else pred
            pred_joint,_delta_v,_v_next,_r_next=stageb_prediction_to_next_joint(pred_frame, history[:,-1], args)
            x_pred=decode_joint(pred_joint,args); x_target=state[:,target_t]
            preds.append(x_pred.unsqueeze(1)); targets.append(x_target.unsqueeze(1))
            history=append_time_point(history,pred_joint)
        if not preds: continue
        pred_seq=torch.cat(preds,dim=1); target_seq=torch.cat(targets,dim=1)
        err=pred_seq-target_seq; B,L=err.shape[:2]
        err_flat=err.reshape(B,L,-1); tgt_flat=target_seq.reshape(B,L,-1)
        err_norm=err_flat.norm(dim=-1); target_norm=tgt_flat.norm(dim=-1).clamp_min(1e-8)
        corr_curves.append(timestep_flattened_corr(pred_seq,target_seq).detach().cpu())
        rel_l2_curves.append((err_norm/target_norm).mean(dim=0).detach().cpu())
        err_norm_curves.append(err_norm.mean(dim=0).detach().cpu())
        err_sq_curves.append(err_flat.pow(2).sum(dim=-1).mean(dim=0).detach().cpu())
        target_norm_curves.append(target_norm.mean(dim=0).detach().cpu())
        pred_lens.append(float(L))
    out={f"{prefix}/free_rollout_context_len":float(W), f"{prefix}/free_rollout_pred_len":float(np.mean(pred_lens)) if pred_lens else float('nan')}
    if not corr_curves: return out
    min_len=min(int(c.numel()) for c in corr_curves)
    corr=torch.stack([c[:min_len] for c in corr_curves]).mean(0)
    rel=torch.stack([c[:min_len] for c in rel_l2_curves]).mean(0)
    errn=torch.stack([c[:min_len] for c in err_norm_curves]).mean(0)
    errs=torch.stack([c[:min_len] for c in err_sq_curves]).mean(0)
    tgtn=torch.stack([c[:min_len] for c in target_norm_curves]).mean(0)
    out[f"{prefix}/free_rollout_corr_mean"]=float(corr.mean()); out[f"{prefix}/free_rollout_rel_l2_mean"]=float(rel.mean())
    out[f"{prefix}/free_rollout_error_norm_final"]=float(errn[-1]); out[f"{prefix}/free_rollout_error_sq_final"]=float(errs[-1])
    out[f"{prefix}/free_rollout_corr_curve"]=[float(x) for x in corr.tolist()]
    out[f"{prefix}/free_rollout_rel_l2_curve"]=[float(x) for x in rel.tolist()]
    out[f"{prefix}/free_rollout_error_norm_curve"]=[float(x) for x in errn.tolist()]
    out[f"{prefix}/free_rollout_error_sq_curve"]=[float(x) for x in errs.tolist()]
    out[f"{prefix}/free_rollout_target_norm_curve"]=[float(x) for x in tgtn.tolist()]
    return out


@torch.no_grad()
def evaluate_stageb_stim_potential_residual_horizon_sweep(model, dataloader, args, rank=0, prefix="test"):
    """Refresh-horizon evaluation for B: full-x input, residual output.

    Frozen Stage A rolls v by v <- v + G(u).  The AR model predicts only the
    residual r, and the scored prediction is x = D(v) + r.
    """
    from internal_dw.training.potential_stageb import get_stageb_potential_model, stageb_stim_context_window

    model.eval()
    W = int(args.window_size)
    horizons = sorted({int(h) for h in args.test_horizons if int(h) >= 1})
    corr_by_h = {h: [] for h in horizons}
    rel_l2_by_h = {h: [] for h in horizons}
    pot = get_stageb_potential_model(args, rank)

    for batch in dataloader:
        state, stim, _, _ = unpack_batch(batch)
        state = state.cuda(rank, non_blocking=True).float()
        if stim is None:
            stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))
        stim = stim.cuda(rank, non_blocking=True).float()
        _B, T, _ = get_batch_time_shape(state)
        if T <= W:
            continue

        for H in horizons:
            history = time_window(state, 0, W).clone()
            v_prev = pot.encode(state[:, W-1]).detach()
            steps_since_refresh = 0
            preds, targets = [], []
            for target_t in range(W, T):
                stim_win = time_window(stim, target_t-W, target_t)
                pred = model(stim_win, history, return_aux=False)
                pred_r = pred[:, 0] if pred.dim() == state.dim() else pred
                stim_ctx = stageb_stim_context_window(stim, target_t, int(getattr(args, "potential_stim_context_len", 1)))
                v_next = v_prev + pot.stim_delta_context_window(stim_ctx, like_v=v_prev).detach()
                x_phi = pot.decode(v_next).detach()
                x_pred = x_phi + pred_r
                target_frame = state[:, target_t]
                preds.append(x_pred.unsqueeze(1)); targets.append(target_frame.unsqueeze(1))

                steps_since_refresh += 1
                if steps_since_refresh >= H:
                    next_frame = target_frame
                    v_prev = pot.encode(target_frame).detach()
                    steps_since_refresh = 0
                else:
                    next_frame = x_pred
                    v_prev = v_next.detach()
                history = append_time_point(history, next_frame)

            if preds:
                pred_seq = torch.cat(preds, dim=1)
                target_seq = torch.cat(targets, dim=1)
                corr_by_h[H].append(trajectory_corrcoef_flat(pred_seq, target_seq).detach().cpu())
                rel_l2_by_h[H].append(trajectory_relative_l2(pred_seq, target_seq).detach().cpu())
    out = {}
    for h, vals in corr_by_h.items():
        out[f"{prefix}/horizon_{h}_corr"] = float(torch.stack(vals).mean()) if vals else 0.0
    for h, vals in rel_l2_by_h.items():
        out[f"{prefix}/horizon_{h}_rel_l2"] = float(torch.stack(vals).mean()) if vals else 0.0
    return out


@torch.no_grad()
def evaluate_stageb_stim_potential_residual_free_rollout_curves(model, dataloader, args, rank=0, prefix="test"):
    from internal_dw.training.potential_stageb import get_stageb_potential_model, stageb_stim_context_window

    model.eval()
    W = int(args.window_size)
    corr_curves=[]; rel_l2_curves=[]; err_norm_curves=[]; err_sq_curves=[]; target_norm_curves=[]; pred_lens=[]
    pot = get_stageb_potential_model(args, rank)
    for batch in dataloader:
        state, stim, _, _ = unpack_batch(batch)
        state = state.cuda(rank, non_blocking=True).float()
        if stim is None:
            stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))
        stim = stim.cuda(rank, non_blocking=True).float()
        _B, T, _ = get_batch_time_shape(state)
        if T <= W:
            continue
        history = time_window(state, 0, W).clone()
        v_prev = pot.encode(state[:, W-1]).detach()
        preds=[]; targets=[]
        for target_t in range(W, T):
            stim_win = time_window(stim, target_t-W, target_t)
            pred = model(stim_win, history, return_aux=False)
            pred_r = pred[:, 0] if pred.dim() == state.dim() else pred
            stim_ctx = stageb_stim_context_window(stim, target_t, int(getattr(args, "potential_stim_context_len", 1)))
            v_next = v_prev + pot.stim_delta_context_window(stim_ctx, like_v=v_prev).detach()
            x_phi = pot.decode(v_next).detach()
            x_pred = x_phi + pred_r
            x_target = state[:, target_t]
            preds.append(x_pred.unsqueeze(1)); targets.append(x_target.unsqueeze(1))
            history = append_time_point(history, x_pred)
            v_prev = v_next.detach()
        if not preds:
            continue
        pred_seq=torch.cat(preds,dim=1); target_seq=torch.cat(targets,dim=1)
        err=pred_seq-target_seq; B,L=err.shape[:2]
        err_flat=err.reshape(B,L,-1); tgt_flat=target_seq.reshape(B,L,-1)
        err_norm=err_flat.norm(dim=-1); target_norm=tgt_flat.norm(dim=-1).clamp_min(1e-8)
        corr_curves.append(timestep_flattened_corr(pred_seq,target_seq).detach().cpu())
        rel_l2_curves.append((err_norm/target_norm).mean(dim=0).detach().cpu())
        err_norm_curves.append(err_norm.mean(dim=0).detach().cpu())
        err_sq_curves.append(err_flat.pow(2).sum(dim=-1).mean(dim=0).detach().cpu())
        target_norm_curves.append(target_norm.mean(dim=0).detach().cpu())
        pred_lens.append(float(L))
    out={f"{prefix}/free_rollout_context_len":float(W), f"{prefix}/free_rollout_pred_len":float(np.mean(pred_lens)) if pred_lens else float('nan')}
    if not corr_curves:
        return out
    min_len=min(int(c.numel()) for c in corr_curves)
    corr=torch.stack([c[:min_len] for c in corr_curves]).mean(0)
    rel=torch.stack([c[:min_len] for c in rel_l2_curves]).mean(0)
    errn=torch.stack([c[:min_len] for c in err_norm_curves]).mean(0)
    errs=torch.stack([c[:min_len] for c in err_sq_curves]).mean(0)
    tgtn=torch.stack([c[:min_len] for c in target_norm_curves]).mean(0)
    out[f"{prefix}/free_rollout_corr_mean"]=float(corr.mean()); out[f"{prefix}/free_rollout_rel_l2_mean"]=float(rel.mean())
    out[f"{prefix}/free_rollout_error_norm_final"]=float(errn[-1]); out[f"{prefix}/free_rollout_error_sq_final"]=float(errs[-1])
    out[f"{prefix}/free_rollout_corr_curve"]=[float(x) for x in corr.tolist()]
    out[f"{prefix}/free_rollout_rel_l2_curve"]=[float(x) for x in rel.tolist()]
    out[f"{prefix}/free_rollout_error_norm_curve"]=[float(x) for x in errn.tolist()]
    out[f"{prefix}/free_rollout_error_sq_curve"]=[float(x) for x in errs.tolist()]
    out[f"{prefix}/free_rollout_target_norm_curve"]=[float(x) for x in tgtn.tolist()]
    return out


@torch.no_grad()
def evaluate_standard_ar_horizon_sweep(model, dataloader, args, rank=0, prefix="test"):
    if bool(getattr(args, "stageb_stim_potential_residual_ar", False)):
        return evaluate_stageb_stim_potential_residual_horizon_sweep(model, dataloader, args, rank=rank, prefix=prefix)
    if bool(getattr(args, "stageb_joint_potential_ar", False)):
        return evaluate_stageb_joint_horizon_sweep(model, dataloader, args, rank=rank, prefix=prefix)
    """Refresh-horizon metrics for ordinary one-step AR models.

    horizon H means: free-roll for H predicted steps, then refresh the history
    with the ground-truth frame.  H=1 is teacher-forced one-step evaluation.
    """
    model.eval()
    W = int(args.window_size)
    horizons = sorted({int(h) for h in args.test_horizons if int(h) >= 1})
    corr_by_h = {h: [] for h in horizons}
    rel_l2_by_h = {h: [] for h in horizons}

    for batch in dataloader:
        state, stim, _, _ = unpack_batch(batch)
        state = state.cuda(rank, non_blocking=True).float()
        if stim is None:
            stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))
        stim = stim.cuda(rank, non_blocking=True).float()
        if bool(getattr(args, "stageb_potential_residual", False)):
            state, _stageb_logs = stageb_residualize_state(state, args)
        _B, T, _state_shape = get_batch_time_shape(state)
        if T <= W:
            continue

        for H in horizons:
            history = time_window(state, 0, W).clone()
            steps_since_refresh = 0
            preds = []
            targets = []

            for target_t in range(W, T):
                stim_win = time_window(stim, target_t - W, target_t)
                pred = model(stim_win, history, return_aux=False)
                pred_frame = pred[:, 0] if pred.dim() == state.dim() else pred
                target_frame = state[:, target_t]

                # Save the model prediction for scoring.  The GT refresh below is
                # used only to update the future history; it must not replace the
                # saved prediction trajectory.
                preds.append(pred_frame.unsqueeze(1))
                targets.append(target_frame.unsqueeze(1))

                steps_since_refresh += 1
                if steps_since_refresh >= H:
                    next_frame = target_frame
                    steps_since_refresh = 0
                else:
                    next_frame = pred_frame
                history = append_time_point(history, next_frame)

            if preds:
                pred_seq = torch.cat(preds, dim=1)
                target_seq = torch.cat(targets, dim=1)
                corr_by_h[H].append(trajectory_corrcoef_flat(pred_seq, target_seq).detach().cpu())
                rel_l2_by_h[H].append(trajectory_relative_l2(pred_seq, target_seq).detach().cpu())

    out = {}
    for h, vals in corr_by_h.items():
        out[f"{prefix}/horizon_{h}_corr"] = float(torch.stack(vals).mean()) if vals else 0.0
    for h, vals in rel_l2_by_h.items():
        out[f"{prefix}/horizon_{h}_rel_l2"] = float(torch.stack(vals).mean()) if vals else 0.0
    return out




@torch.no_grad()
def evaluate_standard_ar_free_rollout_curves(model, dataloader, args, rank=0, prefix="test"):
    if bool(getattr(args, "stageb_stim_potential_residual_ar", False)):
        return evaluate_stageb_stim_potential_residual_free_rollout_curves(model, dataloader, args, rank=rank, prefix=prefix)
    if bool(getattr(args, "stageb_joint_potential_ar", False)):
        return evaluate_stageb_joint_free_rollout_curves(model, dataloader, args, rank=rank, prefix=prefix)
    """Uninterrupted free-rollout curves for ordinary one-step AR models.

    These per-step curves are intended to diagnose whether error growth behaves
    more like coherent accumulation (error norm ~ K) or diffusive accumulation
    (error norm ~ sqrt(K)).  They are computed on the full test split after the
    initial context window and are independent of refresh-horizon metrics.
    """
    model.eval()
    W = int(args.window_size)

    corr_curves = []
    rel_l2_curves = []
    err_norm_curves = []
    err_sq_curves = []
    target_norm_curves = []
    pred_lens = []

    for batch in dataloader:
        state, stim, _, _ = unpack_batch(batch)
        state = state.cuda(rank, non_blocking=True).float()
        if stim is None:
            stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))
        stim = stim.cuda(rank, non_blocking=True).float()
        _B, T, _state_shape = get_batch_time_shape(state)
        if T <= W:
            continue

        history = time_window(state, 0, W).clone()
        preds = []
        targets = []
        for target_t in range(W, T):
            stim_win = time_window(stim, target_t - W, target_t)
            pred = model(stim_win, history, return_aux=False)
            pred_frame = pred[:, 0] if pred.dim() == state.dim() else pred
            target_frame = state[:, target_t]
            preds.append(pred_frame.unsqueeze(1))
            targets.append(target_frame.unsqueeze(1))
            history = append_time_point(history, pred_frame)

        if not preds:
            continue

        pred_seq = torch.cat(preds, dim=1)
        target_seq = torch.cat(targets, dim=1)
        err = pred_seq - target_seq
        B, L = err.shape[:2]
        err_flat = err.reshape(B, L, -1)
        tgt_flat = target_seq.reshape(B, L, -1)
        err_norm = err_flat.norm(dim=-1)                  # [B,L]
        target_norm = tgt_flat.norm(dim=-1).clamp_min(1e-8)
        rel_l2_t = (err_norm / target_norm).mean(dim=0)   # [L]
        err_norm_t = err_norm.mean(dim=0)
        err_sq_t = err_flat.pow(2).sum(dim=-1).mean(dim=0)
        target_norm_t = target_norm.mean(dim=0)

        corr_curves.append(timestep_flattened_corr(pred_seq, target_seq).detach().cpu())
        rel_l2_curves.append(rel_l2_t.detach().cpu())
        err_norm_curves.append(err_norm_t.detach().cpu())
        err_sq_curves.append(err_sq_t.detach().cpu())
        target_norm_curves.append(target_norm_t.detach().cpu())
        pred_lens.append(float(L))

    out = {
        f"{prefix}/free_rollout_context_len": float(W),
        f"{prefix}/free_rollout_pred_len": float(np.mean(pred_lens)) if pred_lens else float("nan"),
    }
    if not corr_curves:
        return out

    min_len = min(int(c.numel()) for c in corr_curves)
    corr = torch.stack([c[:min_len] for c in corr_curves], dim=0).mean(dim=0)
    rel = torch.stack([c[:min_len] for c in rel_l2_curves], dim=0).mean(dim=0)
    errn = torch.stack([c[:min_len] for c in err_norm_curves], dim=0).mean(dim=0)
    errs = torch.stack([c[:min_len] for c in err_sq_curves], dim=0).mean(dim=0)
    tgtn = torch.stack([c[:min_len] for c in target_norm_curves], dim=0).mean(dim=0)

    out[f"{prefix}/free_rollout_corr_mean"] = float(corr.mean())
    out[f"{prefix}/free_rollout_rel_l2_mean"] = float(rel.mean())
    out[f"{prefix}/free_rollout_error_norm_final"] = float(errn[-1])
    out[f"{prefix}/free_rollout_error_sq_final"] = float(errs[-1])
    out[f"{prefix}/free_rollout_corr_curve"] = [float(x) for x in corr.tolist()]
    out[f"{prefix}/free_rollout_rel_l2_curve"] = [float(x) for x in rel.tolist()]
    out[f"{prefix}/free_rollout_error_norm_curve"] = [float(x) for x in errn.tolist()]
    out[f"{prefix}/free_rollout_error_sq_curve"] = [float(x) for x in errs.tolist()]
    out[f"{prefix}/free_rollout_target_norm_curve"] = [float(x) for x in tgtn.tolist()]

    def loglog_slope(y: torch.Tensor, start: int, end: int) -> float:
        # start/end are one-based rollout steps, inclusive-ish for readability.
        if y.numel() < 3:
            return float("nan")
        a = max(1, int(start))
        b = min(int(end), int(y.numel()))
        if b <= a + 1:
            return float("nan")
        k = torch.arange(a, b + 1, dtype=torch.float32)
        yy = y[a - 1:b].float().clamp_min(1e-12)
        x = torch.log(k)
        z = torch.log(yy)
        x = x - x.mean()
        z = z - z.mean()
        return float((x * z).sum() / x.pow(2).sum().clamp_min(1e-12))

    out[f"{prefix}/free_rollout_error_norm_loglog_slope_8_64"] = loglog_slope(errn, 8, 64)
    out[f"{prefix}/free_rollout_error_norm_loglog_slope_16_96"] = loglog_slope(errn, 16, 96)
    out[f"{prefix}/free_rollout_error_sq_loglog_slope_8_64"] = loglog_slope(errs, 8, 64)
    out[f"{prefix}/free_rollout_error_sq_loglog_slope_16_96"] = loglog_slope(errs, 16, 96)
    return out



def _repeat_recurrent_batch(value, repeats: int):
    """Repeat the batch axis of a nested recurrent state, horizon-major."""
    if torch.is_tensor(value):
        return value.repeat((int(repeats),) + (1,) * (value.dim() - 1))
    if isinstance(value, tuple):
        return tuple(_repeat_recurrent_batch(item, repeats) for item in value)
    if isinstance(value, list):
        return [_repeat_recurrent_batch(item, repeats) for item in value]
    if isinstance(value, dict):
        return {key: _repeat_recurrent_batch(item, repeats) for key, item in value.items()}
    raise TypeError(f"Unsupported recurrent-state leaf {type(value)!r}")


@torch.inference_mode()
def _evaluate_recurrent_state_horizon_sweep_packed(
    raw,
    dataloader,
    args,
    rank,
    prefix,
    horizons,
    horizon_batch,
):
    """Evaluate several refresh horizons in one larger model batch.

    For the ordinary tied-stack OfficialStateMamba model, ``horizon_index``
    changes only backward routing; its no-grad forward map is identical for all
    refresh horizons.  We can therefore keep one recurrent trajectory per
    horizon in the batch dimension and replace nine serial scans by a few
    larger scans.  Temporally untied and dual-branch models are deliberately
    excluded by the caller because their forward maps may depend on the lead.
    """
    W = int(args.window_size)
    corr_by_h = {h: [] for h in horizons}
    rel_l2_by_h = {h: [] for h in horizons}
    mse_by_h = {h: [] for h in horizons}
    out = {f"{prefix}/recurrent_context_len": float(W)}

    for batch in dataloader:
        state, stim, _, _ = unpack_batch(batch)
        state = state.cuda(rank, non_blocking=True).float()
        if stim is None:
            stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))
        stim = stim.cuda(rank, non_blocking=True).float()
        B, T, _ = get_batch_time_shape(state)
        if T <= W:
            continue

        # Every refresh horizon starts from the same teacher-forced context.
        # Compute it once per data batch, then replicate its value for each
        # horizon group.  The step implementation is functional and does not
        # mutate the input state tensors.
        base_state = raw.init_state(B, state.device, state.dtype)
        for j in range(0, max(W - 1, 0)):
            base_stim = stim[:, j]
            base_pred, base_state = raw.step(
                base_state,
                state[:, j],
                base_stim,
                return_aux=False,
                horizon_index=0,
            )
            del base_pred

        target_seq = state[:, W:T]
        frame_shape = tuple(state.shape[2:])
        for chunk_start in range(0, len(horizons), int(horizon_batch)):
            chunk = horizons[chunk_start : chunk_start + int(horizon_batch)]
            groups = len(chunk)
            recurrent = _repeat_recurrent_batch(base_state, groups)
            x_in = state[:, W - 1].repeat((groups,) + (1,) * (state.dim() - 2))
            pred_steps = []

            for offset, target_t in enumerate(range(W, T), start=1):
                stim_in = stim[:, target_t - 1].repeat(
                    (groups,) + (1,) * (stim.dim() - 2)
                )
                pred_flat_batch, recurrent = raw.step(
                    recurrent,
                    x_in,
                    stim_in,
                    return_aux=False,
                    # For a tied stack this index affects backward routing only;
                    # evaluation is no-grad and all forward values stay exact.
                    horizon_index=0,
                )
                pred_group = pred_flat_batch.reshape(groups, B, *frame_shape)
                pred_steps.append(pred_group.unsqueeze(2))

                target_group = state[:, target_t].unsqueeze(0).expand(
                    groups, B, *frame_shape
                )
                refresh = torch.tensor(
                    [offset % horizon == 0 for horizon in chunk],
                    device=state.device,
                    dtype=torch.bool,
                )
                refresh = refresh.reshape(groups, *([1] * (pred_group.dim() - 1)))
                next_group = torch.where(refresh, target_group, pred_group)
                x_in = next_group.reshape(groups * B, *frame_shape)

            pred_chunk = torch.cat(pred_steps, dim=2)
            for index, horizon in enumerate(chunk):
                pred_seq = pred_chunk[index]
                corr_by_h[horizon].append(
                    trajectory_corrcoef_flat(pred_seq, target_seq).detach().cpu()
                )
                rel_l2_by_h[horizon].append(
                    trajectory_relative_l2(pred_seq, target_seq).detach().cpu()
                )
                mse_by_h[horizon].append(
                    (pred_seq - target_seq).square().mean().detach().cpu()
                )

    for horizon, values in corr_by_h.items():
        out[f"{prefix}/horizon_{horizon}_corr"] = (
            float(torch.stack(values).mean()) if values else 0.0
        )
    for horizon, values in rel_l2_by_h.items():
        out[f"{prefix}/horizon_{horizon}_rel_l2"] = (
            float(torch.stack(values).mean()) if values else 0.0
        )
    for horizon, values in mse_by_h.items():
        out[f"{prefix}/horizon_{horizon}_mse"] = (
            float(torch.stack(values).mean()) if values else 0.0
        )
    return out


@torch.inference_mode()
def evaluate_recurrent_state_horizon_sweep(model, dataloader, args, rank=0, prefix="test"):
    """Refresh-horizon evaluation for recurrent predictive-state models.

    For dual Mamba+LTI models this also evaluates three branch modes:
      1. normal/schedule fusion: existing keys, e.g. test/horizon_64_corr
      2. Mamba-only:           test/mamba_only_horizon_64_corr
      3. LTI-only:             test/lti_only_horizon_64_corr

    IMPORTANT: this evaluator now passes ``horizon_index`` into raw.step().
    Without this, a schedule fusion gate cannot switch from Mamba to LTI at
    long horizons and the long branch can be silently ignored during eval.
    """
    model.eval()
    raw = unwrap_model(model)
    W = int(args.window_size)
    horizons = sorted({int(h) for h in args.test_horizons if int(h) >= 1})

    is_dual = bool(getattr(raw, "is_official_mamba_dual_lti_kg", False)) or hasattr(raw, "set_eval_branch_mode")
    horizon_batch = max(1, int(getattr(args, "recurrent_eval_horizon_batch", 1)))
    can_pack_horizons = bool(
        horizon_batch > 1
        and getattr(raw, "is_official_state_mamba", False)
        and getattr(raw, "block_groups", None) is None
        and not is_dual
    )
    if can_pack_horizons:
        return _evaluate_recurrent_state_horizon_sweep_packed(
            raw,
            dataloader,
            args,
            rank=rank,
            prefix=prefix,
            horizons=horizons,
            horizon_batch=horizon_batch,
        )
    branch_modes = [(None, "")]
    if is_dual:
        branch_modes += [("mamba", "mamba_only_"), ("lti", "lti_only_")]

    out = {f"{prefix}/recurrent_context_len": float(W)}

    for branch_mode, key_prefix in branch_modes:
        if hasattr(raw, "set_eval_branch_mode"):
            raw.set_eval_branch_mode(branch_mode)

        corr_by_h = {h: [] for h in horizons}
        rel_l2_by_h = {h: [] for h in horizons}
        mse_by_h = {h: [] for h in horizons}

        for batch in dataloader:
            state, stim, _, _ = unpack_batch(batch)
            state = state.cuda(rank, non_blocking=True).float()
            if stim is None:
                stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))
            stim = stim.cuda(rank, non_blocking=True).float()
            B, T, _ = get_batch_time_shape(state)
            if T <= W:
                continue

            for H in horizons:
                # Burn in through x_0,...,x_{W-2}; first recurrent prediction consumes
                # x_{W-1} and predicts x_W.
                h = raw.init_state(B, state.device, state.dtype)
                for j in range(0, max(W - 1, 0)):
                    try:
                        _, h = raw.step(h, state[:, j], stim[:, j], return_aux=False, horizon_index=0)
                    except TypeError:
                        _, h = raw.step(h, state[:, j], stim[:, j], return_aux=False)

                x_in = state[:, W - 1]
                steps_since_refresh = 0
                preds = []
                targets = []

                for target_t in range(W, T):
                    # Lead time since the most recent GT refresh.  This is the correct
                    # horizon index for schedule fusion: short lead times use Mamba,
                    # long lead times transition toward the LTI branch.
                    lead_idx = steps_since_refresh + 1
                    try:
                        pred_frame, h = raw.step(
                            h, x_in, stim[:, target_t - 1], return_aux=False, horizon_index=lead_idx
                        )
                    except TypeError:
                        pred_frame, h = raw.step(h, x_in, stim[:, target_t - 1], return_aux=False)
                    target_frame = state[:, target_t]
                    preds.append(pred_frame.unsqueeze(1))
                    targets.append(target_frame.unsqueeze(1))

                    steps_since_refresh += 1
                    if steps_since_refresh >= H:
                        x_in = target_frame
                        steps_since_refresh = 0
                    else:
                        x_in = pred_frame

                if preds:
                    pred_seq = torch.cat(preds, dim=1)
                    target_seq = torch.cat(targets, dim=1)
                    corr_by_h[H].append(trajectory_corrcoef_flat(pred_seq, target_seq).detach().cpu())
                    rel_l2_by_h[H].append(trajectory_relative_l2(pred_seq, target_seq).detach().cpu())
                    mse_by_h[H].append((pred_seq - target_seq).square().mean().detach().cpu())

        for hval, vals in corr_by_h.items():
            out[f"{prefix}/{key_prefix}horizon_{hval}_corr"] = float(torch.stack(vals).mean()) if vals else 0.0
        for hval, vals in rel_l2_by_h.items():
            out[f"{prefix}/{key_prefix}horizon_{hval}_rel_l2"] = float(torch.stack(vals).mean()) if vals else 0.0
        for hval, vals in mse_by_h.items():
            out[f"{prefix}/{key_prefix}horizon_{hval}_mse"] = float(torch.stack(vals).mean()) if vals else 0.0

    if hasattr(raw, "set_eval_branch_mode"):
        raw.set_eval_branch_mode(None)
    return out



@torch.inference_mode()
def evaluate_recurrent_state_free_rollout(model, dataloader, args, rank=0, prefix="test"):
    """Uninterrupted recurrent free rollout after the initial context.

    For dual Mamba+LTI models this evaluates normal/schedule, Mamba-only, and
    LTI-only free rollout curves.  It also passes horizon_index to raw.step(),
    so schedule fusion actually changes with lead time.
    """
    model.eval()
    raw = unwrap_model(model)
    W = int(args.window_size)
    is_dual = bool(getattr(raw, "is_official_mamba_dual_lti_kg", False)) or hasattr(raw, "set_eval_branch_mode")
    branch_modes = [(None, "")]
    if is_dual:
        branch_modes += [("mamba", "mamba_only_"), ("lti", "lti_only_")]

    out = {}
    for branch_mode, key_prefix in branch_modes:
        if hasattr(raw, "set_eval_branch_mode"):
            raw.set_eval_branch_mode(branch_mode)

        rel_l2_vals = []
        corr_curves = []
        pred_lens = []

        for batch in dataloader:
            state, stim, _, _ = unpack_batch(batch)
            state = state.cuda(rank, non_blocking=True).float()
            if stim is None:
                stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))
            stim = stim.cuda(rank, non_blocking=True).float()
            B, T, _ = get_batch_time_shape(state)
            if T <= W:
                continue

            h = raw.init_state(B, state.device, state.dtype)
            for j in range(0, max(W - 1, 0)):
                try:
                    _, h = raw.step(h, state[:, j], stim[:, j], return_aux=False, horizon_index=0)
                except TypeError:
                    _, h = raw.step(h, state[:, j], stim[:, j], return_aux=False)
            x_in = state[:, W - 1]

            preds = []
            for target_t in range(W, T):
                lead_idx = target_t - W + 1
                try:
                    pred_frame, h = raw.step(
                        h, x_in, stim[:, target_t - 1], return_aux=False, horizon_index=lead_idx
                    )
                except TypeError:
                    pred_frame, h = raw.step(h, x_in, stim[:, target_t - 1], return_aux=False)
                preds.append(pred_frame.unsqueeze(1))
                x_in = pred_frame

            if not preds:
                continue
            pred_seq = torch.cat(preds, dim=1)
            target_seq = state[:, W:T]
            rel_l2_vals.append(trajectory_relative_l2(pred_seq, target_seq).detach().cpu())
            corr_curves.append(timestep_flattened_corr(pred_seq, target_seq).detach().cpu())
            pred_lens.append(float(pred_seq.shape[1]))

        out[f"{prefix}/{key_prefix}recurrent_rollout_rel_l2"] = (
            float(torch.stack(rel_l2_vals).mean()) if rel_l2_vals else float("nan")
        )
        out[f"{prefix}/{key_prefix}recurrent_rollout_context_len"] = float(W)
        out[f"{prefix}/{key_prefix}recurrent_rollout_pred_len"] = float(np.mean(pred_lens)) if pred_lens else float("nan")

        if corr_curves:
            min_len = min(int(c.numel()) for c in corr_curves)
            curve = torch.stack([c[:min_len] for c in corr_curves], dim=0).mean(dim=0)
            out[f"{prefix}/{key_prefix}recurrent_rollout_corr_mean"] = float(curve.mean())
            out[f"{prefix}/{key_prefix}recurrent_rollout_corr_mean_0_32"] = float(curve[: min(32, min_len)].mean())
            out[f"{prefix}/{key_prefix}recurrent_rollout_corr_mean_0_64"] = float(curve[: min(64, min_len)].mean())
            out[f"{prefix}/{key_prefix}recurrent_rollout_corr_curve"] = [float(x) for x in curve.tolist()]

    if hasattr(raw, "set_eval_branch_mode"):
        raw.set_eval_branch_mode(None)
    return out



def save_eval_json(path, logs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(logs, f, indent=2, sort_keys=True)


def _should_eval_field_long_rollout(args):
    """Run DiffusionRollout-style metrics for The Well / field models."""
    return (
        str(getattr(args, "dataset", "")) == "the_well"
        or str(getattr(args, "task_type", "")) == "field2d"
        or str(getattr(args, "evaluator", "")) == "field2d"
        or str(getattr(args, "model_name", "")) == "koopman_field_gramian"
    )


@torch.no_grad()
def evaluate_field_long_rollout(model, dataloader, args, rank=0, prefix="test"):
    """
    DiffusionRollout-style long free-rollout evaluation.

    Protocol:
      1. Use the first W frames as context.
      2. Autoregressively predict every remaining frame.
      3. Compute trajectory-level relative L2 over the whole predicted future.

    For turbulent_radiative_layer_2D, to align with DiffusionRollout as closely
    as possible, use:
        --window_size 7
        --thewell_sequence_length 101
    so the model predicts the remaining 94 frames.
    """
    model.eval()
    W = int(args.window_size)

    rel_l2_vals = []
    pred_lens = []
    corr_curves = []

    for batch in dataloader:
        state, stim, _, _ = unpack_batch(batch)
        state = state.cuda(rank, non_blocking=True).float()
        if stim is None:
            stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))
        stim = stim.cuda(rank, non_blocking=True).float()

        B, T, _state_shape = get_batch_time_shape(state)
        if T <= W:
            continue

        history = state[:, :W].clone()
        preds = []

        for t in range(W, T):
            stim_window = time_window(stim, t - W, t)
            pred = model(stim_window, history, return_aux=False)

            # Accept either [B, 1, ...] or [B, ...].
            if pred.dim() == state.dim():
                pred_frame = pred[:, 0]
            elif pred.dim() == state.dim() - 1:
                pred_frame = pred
            else:
                raise ValueError(
                    "Unexpected prediction shape in long rollout: "
                    f"pred={tuple(pred.shape)}, state={tuple(state.shape)}"
                )

            preds.append(pred_frame.unsqueeze(1))
            history = torch.cat([history[:, 1:], pred_frame.unsqueeze(1)], dim=1)

        if not preds:
            continue

        pred_seq = torch.cat(preds, dim=1)
        target_seq = state[:, W:T]

        rel_l2_vals.append(trajectory_relative_l2(pred_seq, target_seq).detach().cpu())
        corr_t = timestep_flattened_corr(pred_seq, target_seq)
        pred_lens.append(float(pred_seq.shape[1]))
        corr_curves.append(corr_t.detach().cpu())

    out = {
        f"{prefix}/rollout_rel_l2": (
            float(torch.stack(rel_l2_vals).mean()) if rel_l2_vals else float("nan")
        ),
        f"{prefix}/rollout_context_len": float(W),
        f"{prefix}/rollout_pred_len": (
            float(np.mean(pred_lens)) if pred_lens else float("nan")
        ),
    }

    if corr_curves:
        min_len = min(int(c.numel()) for c in corr_curves)
        curve = torch.stack([c[:min_len] for c in corr_curves], dim=0).mean(dim=0)
        # Smooth alternatives to the old threshold-based T>0.9 metric.
        # For uniformly spaced rollout frames, the curve mean is also the
        # normalized AUC of the correlation curve.
        out[f"{prefix}/rollout_corr_mean"] = float(curve.mean())
        out[f"{prefix}/rollout_corr_auc"] = float(curve.mean())
        out[f"{prefix}/rollout_corr_mean_0_32"] = float(curve[: min(32, min_len)].mean())
        out[f"{prefix}/rollout_corr_mean_0_64"] = float(curve[: min(64, min_len)].mean())
        # JSON-safe list. This may be long, but for T=101 it is only 94 values.
        out[f"{prefix}/rollout_corr_curve"] = [float(x) for x in curve.tolist()]

    return out


@torch.no_grad()
def evaluate_path_generator_chunked_horizon_sweep(model, dataloader, args, rank=0, prefix="test"):
    """Refresh-horizon metrics for path_generator_field using chunked inference.

    This is different from evaluate_standard_ar_horizon_sweep for path generators.
    The standard evaluator calls model(...), and path_generator_field.forward()
    intentionally returns only the first generated frame for one-step AR
    compatibility.  This function calls raw.generate_path(history, horizon=L),
    so a trained K-step generator actually emits a segment of up to K frames at
    once before the history is updated.

    For each refresh horizon H, the model free-runs for H scored frames, then
    the last frame in the history is refreshed with the ground-truth target.
    If H is larger than the model path_horizon, multiple generated chunks are
    used before refresh.
    """
    model.eval()
    raw = unwrap_model(model)
    if not getattr(raw, "is_path_generator", False):
        return {}

    W = int(args.window_size)
    K_model = int(getattr(raw, "path_horizon", getattr(args, "pathgen_horizon", 1)))
    K_model = max(K_model, 1)
    horizons = sorted({int(h) for h in args.test_horizons if int(h) >= 1})
    corr_by_h = {h: [] for h in horizons}
    rel_l2_by_h = {h: [] for h in horizons}
    used_chunks_by_h = {h: [] for h in horizons}

    for batch in dataloader:
        state, stim, _, _ = unpack_batch(batch)
        state = state.cuda(rank, non_blocking=True).float()
        if stim is None:
            stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))
        stim = stim.cuda(rank, non_blocking=True).float()
        _B, T, _state_shape = get_batch_time_shape(state)
        if T <= W:
            continue

        for H in horizons:
            history = time_window(state, 0, W).clone()
            target_t = W
            steps_since_refresh = 0
            preds = []
            targets = []
            used_chunks = []

            while target_t < T:
                # Do not generate across a refresh boundary.  If H < K_model,
                # this produces H frames and then refreshes.  If H > K_model,
                # this produces multiple K_model chunks before refresh.
                remaining_seq = T - target_t
                remaining_until_refresh = H - steps_since_refresh
                chunk_len = min(K_model, remaining_until_refresh, remaining_seq)
                if chunk_len <= 0:
                    # Defensive fallback; should not happen unless H is invalid.
                    chunk_len = min(K_model, remaining_seq)
                    steps_since_refresh = 0

                if bool(getattr(raw, "uses_future_stimulus", False)):
                    stim_hist = time_window(stim, max(0, target_t - W), target_t)
                    stim_future = time_window(stim, target_t, target_t + chunk_len)
                    path = raw.generate_path(
                        history, horizon=chunk_len, return_aux=False,
                        stim_window=stim_hist, stim_future=stim_future,
                    )
                else:
                    path = raw.generate_path(history, horizon=chunk_len, return_aux=False)
                if path.dim() != state.dim():
                    raise ValueError(
                        "Path generator chunked horizon sweep expected path "
                        f"[B,L,...], got {tuple(path.shape)} with state {tuple(state.shape)}"
                    )

                target_chunk = state[:, target_t : target_t + chunk_len]
                preds.append(path)
                targets.append(target_chunk)
                used_chunks.append(float(chunk_len))

                # Update history with generated frames.  If this chunk ends at a
                # refresh boundary, replace only the final history frame by the
                # corresponding ground-truth target; predictions remain scored.
                hist_chunk = path
                steps_since_refresh += chunk_len
                if steps_since_refresh >= H:
                    hist_chunk = hist_chunk.clone()
                    hist_chunk[:, -1] = target_chunk[:, -1]
                    steps_since_refresh = 0

                history = torch.cat([history, hist_chunk], dim=1)[:, -W:].contiguous()
                target_t += chunk_len

            if preds:
                pred_seq = torch.cat(preds, dim=1)
                target_seq = torch.cat(targets, dim=1)
                corr_by_h[H].append(trajectory_corrcoef_flat(pred_seq, target_seq).detach().cpu())
                rel_l2_by_h[H].append(trajectory_relative_l2(pred_seq, target_seq).detach().cpu())
                used_chunks_by_h[H].extend(used_chunks)

    out = {f"{prefix}/chunked_segment_len": float(K_model)}
    for h, vals in corr_by_h.items():
        out[f"{prefix}/chunked_horizon_{h}_corr"] = float(torch.stack(vals).mean()) if vals else 0.0
    for h, vals in rel_l2_by_h.items():
        out[f"{prefix}/chunked_horizon_{h}_rel_l2"] = float(torch.stack(vals).mean()) if vals else 0.0
    for h, vals in used_chunks_by_h.items():
        out[f"{prefix}/chunked_horizon_{h}_mean_chunk_len"] = float(np.mean(vals)) if vals else 0.0
    return out


@torch.no_grad()
def evaluate_path_generator_chunked_long_rollout(model, dataloader, args, rank=0, prefix="test"):
    """DiffusionRollout-style long rollout for path_generator_field using chunks.

    The old evaluate_field_long_rollout calls model(...), so a path generator
    emits only one frame at a time through forward().  This function calls
    raw.generate_path() and therefore evaluates the intended segment-level
    inference: context -> K generated frames -> next context -> K generated
    frames, until the full future is predicted.
    """
    model.eval()
    raw = unwrap_model(model)
    if not getattr(raw, "is_path_generator", False):
        return {}

    W = int(args.window_size)
    K_model = int(getattr(raw, "path_horizon", getattr(args, "pathgen_horizon", 1)))
    K_model = max(K_model, 1)

    rel_l2_vals = []
    pred_lens = []
    corr_curves = []
    chunk_lens = []

    for batch in dataloader:
        state, stim, _, _ = unpack_batch(batch)
        state = state.cuda(rank, non_blocking=True).float()
        if stim is None:
            stim = zero_external_input_like(state, int(getattr(args, "stim_dim", 1)))
        stim = stim.cuda(rank, non_blocking=True).float()

        B, T, _state_shape = get_batch_time_shape(state)
        if T <= W:
            continue

        history = state[:, :W].clone()
        preds = []
        t = W
        while t < T:
            chunk_len = min(K_model, T - t)
            path = raw.generate_path(history, horizon=chunk_len, return_aux=False)
            if path.dim() != state.dim():
                raise ValueError(
                    "Path generator chunked long rollout expected path "
                    f"[B,L,...], got {tuple(path.shape)} with state {tuple(state.shape)}"
                )
            preds.append(path)
            chunk_lens.append(float(chunk_len))
            history = torch.cat([history, path], dim=1)[:, -W:].contiguous()
            t += chunk_len

        if not preds:
            continue

        pred_seq = torch.cat(preds, dim=1)
        target_seq = state[:, W:T]

        rel_l2_vals.append(trajectory_relative_l2(pred_seq, target_seq).detach().cpu())
        corr_t = timestep_flattened_corr(pred_seq, target_seq)
        pred_lens.append(float(pred_seq.shape[1]))
        corr_curves.append(corr_t.detach().cpu())

    out = {
        f"{prefix}/chunked_rollout_rel_l2": (
            float(torch.stack(rel_l2_vals).mean()) if rel_l2_vals else float("nan")
        ),
        f"{prefix}/chunked_rollout_context_len": float(W),
        f"{prefix}/chunked_rollout_segment_len": float(K_model),
        f"{prefix}/chunked_rollout_mean_chunk_len": (
            float(np.mean(chunk_lens)) if chunk_lens else float("nan")
        ),
        f"{prefix}/chunked_rollout_pred_len": (
            float(np.mean(pred_lens)) if pred_lens else float("nan")
        ),
    }

    if corr_curves:
        min_len = min(int(c.numel()) for c in corr_curves)
        curve = torch.stack([c[:min_len] for c in corr_curves], dim=0).mean(dim=0)
        out[f"{prefix}/chunked_rollout_corr_mean"] = float(curve.mean())
        out[f"{prefix}/chunked_rollout_corr_auc"] = float(curve.mean())
        out[f"{prefix}/chunked_rollout_corr_mean_0_32"] = float(curve[: min(32, min_len)].mean())
        out[f"{prefix}/chunked_rollout_corr_mean_0_64"] = float(curve[: min(64, min_len)].mean())
        out[f"{prefix}/chunked_rollout_corr_curve"] = [float(x) for x in curve.tolist()]

    return out


@torch.no_grad()
def timestep_flattened_corr(pred_seq, target_seq, eps=1e-8):
    """
    Per-timestep Pearson correlation after flattening all non-batch/time dims.

    pred_seq, target_seq:
        [B, T_pred, ...]

    Returns:
        corr_t: [T_pred], averaged over batch.
    """
    B, T_pred = pred_seq.shape[:2]
    pred = pred_seq.reshape(B, T_pred, -1)
    target = target_seq.reshape(B, T_pred, -1)

    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    num = (pred * target).sum(dim=-1)
    den = torch.sqrt(
        pred.pow(2).sum(dim=-1) * target.pow(2).sum(dim=-1)
    ).clamp_min(eps)

    corr = num / den
    return corr.mean(dim=0)


@torch.no_grad()
def t_above_threshold(corr_t, threshold=0.9):
    """
    Normalized prefix duration for which correlation remains above threshold.

    Returns a value in [0, 1]. This is the conservative interpretation of
    T^{>0.9}: stop counting after the first time the correlation drops below
    the threshold.
    """
    T = int(corr_t.numel())
    if T == 0:
        return float("nan")

    count = 0
    for ok in corr_t >= threshold:
        if bool(ok):
            count += 1
        else:
            break

    return float(count) / float(T)
