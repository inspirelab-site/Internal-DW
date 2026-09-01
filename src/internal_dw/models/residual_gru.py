"""Residual GRU autoregressive model -- a second recurrent-state backbone.

Why this exists.  ResGrad's intervention only means "control temporal credit"
when the residual path it gates is the path that carries state across time.
That holds for the stateful Mamba stack; it does NOT hold for the windowed
Transformer/ResNet/UNet backbones, whose residual connections are within a
timestep (spatial / across layers), so gating them degrades the one-step map
instead (measured: gating costs H1 0.013 on Mamba vs 0.035--0.063 on the
windowed backbones).  To test whether the phenomenon is specific to Mamba or
holds for recurrent-state models generally, we need a *different family* of
recurrent-state model.  A GRU is the natural choice: gated RNN, not an SSM.

The key structural fact is that a standard GRU update is already residual,

    h_{t+1} = h_t + z_t * (n_t - h_t)  =  h_t + F(h_t, u_t),

so the identity path is the temporal carrier by construction -- no modification
of the cell is needed to expose it.  We gate the branch F exactly as in
Eq. (I + m J_F): the forward value is unchanged, only the branch's contribution
to the backward graph is scaled.

NOTE on a deliberate difference from ``official_state_mamba``: that
implementation additionally detaches ``conv_state``/``ssm_state`` at gate 0, so
its g=0 corner cuts the cross-time gradient entirely.  Here we implement the
paper's stated intervention faithfully -- identity path open, branch scaled --
so the cross-time gradient still flows along the identity chain at g=0.  The
comparison between the two is informative in itself and is reported as such.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn

GRUStackState = Tuple[torch.Tensor, ...]


class ResidualGRUCell(nn.Module):
    """One gated-recurrent layer written so the identity path is explicit."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.x2h = nn.Linear(input_dim, 3 * hidden_dim)
        self.h2h = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._init_for_long_memory(hidden_dim)

    def _init_for_long_memory(self, H: int):
        """Standard long-memory initialisation for a gated recurrent cell.

        With default init the update gate sits near z=0.5, so the state is
        half-rewritten every step and the model starts with very short memory --
        the wrong prior for a long-horizon rollout task.  We (i) bias the update
        gate negative so z starts small (h changes slowly), and (ii) use an
        orthogonal recurrent map so repeated application neither explodes nor
        collapses.  Gate order is [r, z, n].
        """
        with torch.no_grad():
            for lin in (self.x2h, self.h2h):
                if lin.bias is not None:
                    lin.bias.zero_()
            # orthogonal init on the recurrent (h -> gates) map, per gate block
            for blk in range(3):
                nn.init.orthogonal_(self.h2h.weight[blk * H:(blk + 1) * H], gain=1.0)
            # z is the middle block: negative bias => small z => long memory
            self.h2h.bias[H:2 * H].fill_(-2.0)

    def branch(self, h: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """F(h,u) in h_new = h + F(h,u); the standard GRU delta z*(n-h)."""
        gx = self.x2h(u)
        gh = self.h2h(self.norm(h))
        xr, xz, xn = gx.chunk(3, dim=-1)
        hr, hz, hn = gh.chunk(3, dim=-1)
        r = torch.sigmoid(xr + hr)
        z = torch.sigmoid(xz + hz)
        n = torch.tanh(xn + r * hn)
        return self.dropout(z * (n - h))


class ResidualGRUARModel(nn.Module):
    """Stacked residual GRU with the same step()/state interface as the Mamba stack."""

    # Both flags are required: trainer.py dispatches on is_standard_autoregressive
    # first, and only then checks is_recurrent_state_ar to pick the recurrent
    # BPTT path.  Without the first flag the model falls through to the Koopman
    # loss, which is the wrong objective entirely.
    is_standard_autoregressive = True
    is_recurrent_state_ar = True

    def __init__(
        self,
        roi_dim: int,
        stim_dim: int = 0,
        hidden_dim: int = 512,
        depth: int = 4,
        dropout: float = 0.0,
        has_external_input: bool = True,
        residual: bool = True,
        resgrad_routing: bool = False,
        resgrad_policy: str = "all",
        resgrad_block_gate: float = 1.0,
        resgrad_ratio_threshold: float = 0.05,
        resgrad_cut_state: bool = False,
    ):
        super().__init__()
        self.roi_dim = int(roi_dim)
        self.stim_dim = int(stim_dim) if has_external_input else 0
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.has_external_input = bool(has_external_input)
        # Predict the increment, not the frame: pred = x_t + Delta.  The Mamba
        # stack does the same (`pred_flat = x_flat + delta` under --simple_residual),
        # and on smoothly-evolving states it is the difference between learning a
        # small correction and reconstructing the whole state from scratch --
        # Lorenz-96 has autocorr(k=1) = 0.992, so without this even the one-step
        # map is far harder than it should be.
        self.residual = bool(residual)
        self.resgrad_routing = bool(resgrad_routing)
        self.resgrad_policy = str(resgrad_policy).lower()
        self.resgrad_block_gate = float(resgrad_block_gate)
        self.resgrad_ratio_threshold = float(resgrad_ratio_threshold)
        # False = the paper's stated intervention (I + m J_F): identity path stays
        # differentiable, so cross-time gradient survives at g=0.
        # True  = mirror official_state_mamba, which ALSO detaches the carried
        # state at gate 0 and therefore cuts cross-time gradient entirely.
        # Running both isolates whether the Mamba result depends on that extra cut.
        self.resgrad_cut_state = bool(resgrad_cut_state)

        in_dim = self.roi_dim + self.stim_dim
        self.in_proj = nn.Linear(in_dim, self.hidden_dim)
        self.cells = nn.ModuleList(
            [ResidualGRUCell(self.hidden_dim, self.hidden_dim, dropout) for _ in range(self.depth)]
        )
        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.roi_dim)

    # ---- state -----------------------------------------------------------
    def init_state(self, batch_size: int, device=None, dtype=None) -> GRUStackState:
        # Each layer's state is a 1-tuple, not a bare tensor: the checkpointing
        # helpers in ar_losses.py flatten the stack by iterating each layer's
        # state, so a bare tensor would be split along the batch dimension.
        return tuple(
            (torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype),)
            for _ in range(self.depth)
        )

    def detach_state(self, h: GRUStackState) -> GRUStackState:
        return tuple((s[0].detach(),) for s in h)

    # ---- routing ---------------------------------------------------------
    def _gate_for(self, branch: torch.Tensor, residual: torch.Tensor, ratio_collector=None):
        """Return the scalar gate for this (layer, step), mirroring the Mamba policy."""
        if not self.resgrad_routing:
            return 1.0
        b = branch.detach().float().reshape(branch.shape[0], -1).norm(dim=1).mean()
        r = residual.detach().float().reshape(residual.shape[0], -1).norm(dim=1).mean()
        ratio = float((b / (r + 1e-8)).detach().cpu())
        if ratio_collector is not None:
            ratio_collector.append(ratio)
        if self.resgrad_policy in ("ratio", "act_ratio", "dynamic", "dynamic_ratio",
                                   "branch_ratio", "norm_ratio"):
            return 1.0 if ratio >= self.resgrad_ratio_threshold else self.resgrad_block_gate
        if self.resgrad_policy == "none":
            return 0.0
        return self.resgrad_block_gate if self.resgrad_policy != "all" else 1.0

    @staticmethod
    def _apply_gate(branch: torch.Tensor, gate: float) -> torch.Tensor:
        """Forward value unchanged; backward contribution scaled by `gate`."""
        if gate >= 1.0:
            return branch
        if gate <= 0.0:
            return branch.detach()
        return branch.detach() + gate * (branch - branch.detach())

    # ---- one step --------------------------------------------------------
    def step(
        self,
        h: Optional[GRUStackState],
        x_t: torch.Tensor,
        stim_t: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        horizon_index: Optional[int] = None,
        total_horizon: Optional[int] = None,
        ratio_collector: Optional[list] = None,
    ):
        B = x_t.shape[0]
        x_flat = x_t.reshape(B, -1)
        if self.has_external_input and self.stim_dim > 0:
            s = torch.zeros(B, self.stim_dim, device=x_flat.device, dtype=x_flat.dtype) \
                if stim_t is None else stim_t.reshape(B, -1)[:, : self.stim_dim]
            u = torch.cat([x_flat, s], dim=-1)
        else:
            u = x_flat
        if h is None:
            h = self.init_state(B, x_flat.device, x_flat.dtype)

        tok = self.in_proj(u)
        new_h, gates = [], []
        for i, cell in enumerate(self.cells):
            residual = h[i][0] if isinstance(h[i], (tuple, list)) else h[i]   # temporal identity path
            branch = cell.branch(residual, tok)               # F(h,u)
            g = self._gate_for(branch, residual, ratio_collector)
            gates.append(float(g))
            branch = self._apply_gate(branch, g)
            hi = residual + branch                            # h_{t+1} = h_t + m*F  (forward: m=1)
            if self.resgrad_cut_state and g <= 0.0:
                hi = hi.detach()                              # Mamba-style: cut the carry too
            elif self.resgrad_cut_state and g < 1.0:
                hi = hi.detach() + g * (hi - hi.detach())
            new_h.append((hi,))
            tok = hi
        delta = self.out_proj(self.out_norm(tok)).reshape(x_t.shape)
        pred = x_t + delta if self.residual else delta
        h_new = tuple(new_h)
        if not return_aux:
            return pred, h_new
        aux = {"resgrad_gate": pred.new_tensor(sum(gates) / max(1, len(gates)))}
        return pred, h_new, aux

    def forward(self, stim, history, return_aux: bool = False):
        """Windowed convenience path: burn in over the history window, predict next.

        The recurrent BPTT path uses step() directly; this exists so the model can
        also be called by the windowed evaluators.  It must return (pred, aux) when
        return_aux is set -- callers unpack a 2-tuple.
        """
        B, W = history.shape[0], history.shape[1]
        h = self.init_state(B, history.device, history.dtype)
        pred = None
        for j in range(W):
            s = stim[:, j] if stim is not None else None
            pred, h = self.step(h, history[:, j], s)
        if return_aux:
            return pred, {}
        return pred
