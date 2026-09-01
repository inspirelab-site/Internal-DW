import torch

from internal_dw.data_utils.state_ops import flatten_for_loss


@torch.no_grad()
def trajectory_relative_l2(pred_seq, target_seq, eps=1e-8):
    """
    DiffusionRollout-style trajectory-level relative L2.

    pred_seq, target_seq:
        [B, T_pred, ...]

    Returns:
        mean_b ||pred_b - target_b||_2 / ||target_b||_2
    """
    B = pred_seq.shape[0]
    pred_flat = pred_seq.reshape(B, -1).double()
    target_flat = target_seq.reshape(B, -1).double()

    err = torch.linalg.vector_norm(pred_flat - target_flat, dim=1)
    den = torch.linalg.vector_norm(target_flat, dim=1).clamp_min(eps)

    return (err / den).mean()


def trajectory_corrcoef_flat(pred_seq, target_seq, eps=1e-8):
    """
    Trajectory-level Pearson correlation.

    pred_seq, target_seq:
        [B, T_pred, ...]

    Returns:
        mean_b corr(vec(pred_seq_b), vec(target_seq_b))

    This matches the refresh-horizon protocol: first build the whole
    rollout trajectory under a given refresh interval H, then compute
    correlation against the whole ground-truth trajectory.
    """
    B = pred_seq.shape[0]
    pred = pred_seq.reshape(B, -1).double()
    target = target_seq.reshape(B, -1).double()

    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)

    num = (pred * target).sum(dim=1)
    den = torch.sqrt(
        pred.square().sum(dim=1) * target.square().sum(dim=1)
    ).clamp_min(eps)

    return (num / den).mean()


def corrcoef_torch(pred, target, eps=1e-8):
    """Feature-wise correlation for vector sequence data."""
    pred = pred.reshape(-1, pred.shape[-1])
    target = target.reshape(-1, target.shape[-1])
    pred = pred - pred.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    num = (pred * target).sum(dim=0)
    den = torch.sqrt((pred.square().sum(dim=0) + eps) * (target.square().sum(dim=0) + eps))
    return (num / den.clamp_min(eps)).mean()


def corrcoef_flat(pred, target, eps=1e-8):
    """Sample-wise correlation after flattening non-batch dimensions."""
    pred = flatten_for_loss(pred)
    target = flatten_for_loss(target)
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    num = (pred * target).sum(dim=1)
    den = torch.sqrt((pred.square().sum(dim=1) + eps) * (target.square().sum(dim=1) + eps))
    return (num / den.clamp_min(eps)).mean()


def relative_l2(pred, target, eps=1e-8):
    """
    Endpoint/sample-level relative L2.

    This now uses mean of per-sample ratios:
        mean_b ||pred_b-target_b||_2 / ||target_b||_2

    This is more consistent with trajectory_relative_l2 than the previous
    ratio-of-means implementation.
    """
    p = flatten_for_loss(pred).double()
    y = flatten_for_loss(target).double()
    err = torch.linalg.vector_norm(p - y, dim=1)
    den = torch.linalg.vector_norm(y, dim=1).clamp_min(eps)
    return (err / den).mean()
