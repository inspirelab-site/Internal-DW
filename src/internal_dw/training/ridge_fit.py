import torch


def _ridge_feature_current(model, flat_state, external, t):
    """Build exactly the same current-state feature used by RidgeKoopman.forward.

    flat_state: [B,T,state_dim]
    external: None or [B,T,...]
    t: index of latest observed time. Target is t+1.
    """
    x_t = flat_state[:, t]
    features = [x_t]

    if model.B is not None and external is not None:
        # Forward receives an external_window and internally takes the latest entry.
        # Use a length-1 window here so closed-form fit and rollout forward share
        # the exact same pooling/flattening logic.
        u_t = model._pool_latest_external(external[:, t:t + 1])
        features.append(u_t)

    ones = torch.ones(x_t.shape[0], 1, device=x_t.device, dtype=x_t.dtype)
    features.append(ones)
    return torch.cat(features, dim=-1)


@torch.no_grad()
def fit_closed_form_ridge(model, loader, args, device=None):
    """Memory-safe closed-form ridge fit for RidgeKoopmanModel.

    The old implementation concatenated the full design matrix X and target Y on
    GPU, then solved (X^T X + alpha I)W = X^T Y. For HCP this can easily occupy
    tens of GB. This implementation streams over batches/time points and only
    accumulates X^T X and X^T Y on CPU, which are small for ridge_koopman.

    Important: this is not rollout/BPTT training and it does not keep any graph.
    It is pure teacher-forced one-step ridge:
        [x_t, u_t, 1] -> x_{t+1}.
    """
    raw = model.module if hasattr(model, "module") else model
    model_device = raw.A.device

    xtx = None
    xty = None
    n_rows = 0

    # Accumulate in float64 on CPU for numerical stability and low GPU memory.
    accum_device = torch.device("cpu")
    accum_dtype = torch.float64

    for batch in loader:
        state = batch["state"].to(accum_device, dtype=accum_dtype)
        external = batch.get("external_input", None)
        if external is not None:
            external = external.to(accum_device, dtype=accum_dtype)

        B, T = state.shape[:2]
        flat_state = state.reshape(B, T, -1)

        # Same alignment as the training/eval code:
        # history = state[:, t-W+1:t+1], target = state[:, t+1].
        # With target index cur_t in loss.py, latest observed index is cur_t-1.
        for t in range(int(args.window_size) - 1, T - 1):
            X = _ridge_feature_current(raw, flat_state, external, t)
            Y = flat_state[:, t + 1]

            if xtx is None:
                d_in = X.shape[1]
                d_out = Y.shape[1]
                xtx = torch.zeros(d_in, d_in, device=accum_device, dtype=accum_dtype)
                xty = torch.zeros(d_in, d_out, device=accum_device, dtype=accum_dtype)

            xtx.add_(X.T @ X)
            xty.add_(X.T @ Y)
            n_rows += X.shape[0]

    if xtx is None or xty is None or n_rows == 0:
        raise RuntimeError("No rows were collected for closed-form ridge fit.")

    alpha = float(args.ridge_alpha)
    eye = torch.eye(xtx.shape[0], device=accum_device, dtype=accum_dtype)
    eye[-1, -1] = 0.0  # do not regularize bias

    W = torch.linalg.solve(xtx + alpha * eye, xty)  # [D_in, state_dim]
    W = W.to(device=model_device, dtype=raw.A.dtype)

    state_dim = int(raw.state_dim)
    input_dim = int(raw.input_dim) if raw.B is not None else 0

    raw.A.copy_(W[:state_dim].T)
    offset = state_dim

    if raw.B is not None:
        raw.B.copy_(W[offset:offset + input_dim].T)
        offset += input_dim

    raw.bias.copy_(W[offset])

    # Return a tiny diagnostic dict for optional printing/debugging.
    return {
        "num_rows": int(n_rows),
        "feature_dim": int(W.shape[0]),
        "target_dim": int(W.shape[1]),
        "ridge_alpha": alpha,
    }
