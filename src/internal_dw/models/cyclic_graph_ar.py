"""Cyclic graph autoregressive model for vector fMRI states.

This is intentionally small and explicit.  The model treats each ROI as a node
and executes the same bidirectional message-passing operator for a fixed number
of internal steps before reading out the next state.  Unlike a feed-forward GNN,
the graph update is shared across internal steps, so bidirectional edges create a
cyclic computation system rather than a DAG-style layer stack.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .simple_ar import _StimPoolMixin


GraphState = torch.Tensor


def detach_graph_state(h: Optional[GraphState]) -> Optional[GraphState]:
    return None if h is None else h.detach()


class CyclicGraphARModel(nn.Module, _StimPoolMixin):
    """Closed-loop / cyclic ROI graph predictor.

    Per time step, the model builds ROI-node states from the current fMRI state,
    stimulus context, and the previous graph state.  It then performs S rounds of
    shared bidirectional message passing over a fixed ROI graph and reads out a
    residual delta.

    The external recurrent state is the final ROI-node state ``h`` with shape
    ``[B, state_dim, node_dim]``.  It is used by the existing recurrent BPTT
    training path in ``ar_losses.py``.
    """

    is_standard_autoregressive = True
    is_recurrent_state_ar = True
    is_cyclic_graph_ar = True

    def __init__(
        self,
        state_dim: int,
        input_dim: int = 0,
        state_shape: Optional[Sequence[int]] = None,
        hidden_dim: int = 256,
        graph_steps: int = 5,
        topk: int = 8,
        alpha: float = 0.10,
        carry: float = 0.25,
        dropout: float = 0.0,
        has_external_input: bool = True,
        residual: bool = True,
        edge_type: str = "ring",
        learned_edge_gate: bool = True,
        message_hidden_mult: int = 2,
        **unused_kwargs,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.input_dim = int(input_dim)
        self.state_shape = tuple(int(x) for x in state_shape) if state_shape is not None else None
        self.task_type = "field2d" if self.state_shape is not None and len(self.state_shape) == 3 else "vector"
        self.hidden_dim = int(hidden_dim)
        self.graph_steps = max(1, int(graph_steps))
        self.topk = max(1, int(topk))
        self.alpha = float(alpha)
        self.carry = float(carry)
        self.has_external_input = bool(has_external_input)
        self.residual = bool(residual)
        self.edge_type = str(edge_type)
        self.learned_edge_gate = bool(learned_edge_gate)

        self.roi_embed = nn.Parameter(torch.randn(self.state_dim, self.hidden_dim) * 0.02)
        self.x_encoder = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.stim_encoder = nn.Sequential(
            nn.LayerNorm(max(1, self.input_dim)),
            nn.Linear(max(1, self.input_dim), self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        ) if self.has_external_input and self.input_dim > 0 else None
        self.global_encoder = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.prev_state_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        msg_hid = int(message_hidden_mult) * self.hidden_dim
        self.msg_mlp = nn.Sequential(
            nn.LayerNorm(2 * self.hidden_dim),
            nn.Linear(2 * self.hidden_dim, msg_hid),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(msg_hid, self.hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.LayerNorm(3 * self.hidden_dim),
            nn.Linear(3 * self.hidden_dim, 4 * self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(4 * self.hidden_dim, self.hidden_dim),
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )
        # Start close to identity, but not by hard-coding a separate anchor.  The
        # cyclic graph must learn a correction field; residual=True only sets the
        # current state as the base point of the correction.
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

        edge_index, edge_weight = self._make_edges(self.state_dim, self.topk, self.edge_type)
        self.register_buffer("edge_index", edge_index, persistent=True)
        self.register_buffer("base_edge_weight", edge_weight, persistent=True)
        deg = torch.zeros(self.state_dim, dtype=torch.float32)
        deg.index_add_(0, edge_index[1].cpu(), edge_weight.cpu().float())
        self.register_buffer("in_degree", deg.clamp_min(1.0), persistent=True)
        # OOM-safe normalized dense adjacency for all-ROI runs.  For N=400 this
        # is tiny (400x400) and avoids materializing [B, E, H] edge tensors in
        # every internal graph step.  This keeps the cyclic closed-loop update
        # but changes pairwise edge MLP messages into neighbor-state aggregation.
        adj = torch.zeros(self.state_dim, self.state_dim, dtype=torch.float32)
        adj.index_put_((edge_index[1].cpu(), edge_index[0].cpu()), edge_weight.cpu().float(), accumulate=True)
        adj = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        self.register_buffer("adj_norm", adj, persistent=True)
        if self.learned_edge_gate:
            # Kept for compatibility/logging, but the OOM-safe path defaults to
            # fixed adjacency.  Per-edge gates require rebuilding sparse messages
            # and are disabled in the script by default.
            self.edge_logit = nn.Parameter(torch.zeros(edge_index.shape[1]))
        else:
            self.register_parameter("edge_logit", None)

    @staticmethod
    def _make_edges(n: int, topk: int, edge_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct a deterministic bidirectional ROI graph.

        ``ring`` connects each ROI to +/- 1..topk neighbors by index.  This is a
        safe no-metadata default for all 400 ROIs.  ``dense`` is available for
        small n only; it is usually too expensive for 400 ROI with large hidden
        widths.  ``skip`` adds multi-scale offsets to the ring.
        """
        n = int(n)
        topk = max(1, int(topk))
        edge_type = str(edge_type).lower()
        src, dst = [], []
        if edge_type == "dense":
            for i in range(n):
                for j in range(n):
                    if i != j:
                        src.append(j); dst.append(i)
        else:
            offsets = list(range(1, topk + 1))
            if edge_type == "skip":
                # Multi-scale offsets, still O(N topk).  Useful if Schaefer ROI
                # ordering separates related regions by nonlocal jumps.
                offsets = []
                v = 1
                while len(offsets) < topk:
                    offsets.append(v)
                    v *= 2
            for i in range(n):
                for off in offsets[:topk]:
                    j1 = (i + off) % n
                    j2 = (i - off) % n
                    src.extend([j1, j2])
                    dst.extend([i, i])
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
        return edge_index, edge_weight

    def init_state(self, batch_size: int, device=None, dtype=None) -> GraphState:
        device = device if device is not None else next(self.parameters()).device
        dtype = dtype if dtype is not None else next(self.parameters()).dtype
        return torch.zeros(batch_size, self.state_dim, self.hidden_dim, device=device, dtype=dtype)

    def detach_state(self, h: GraphState) -> GraphState:
        return h.detach()

    def _flatten_state(self, x_t: torch.Tensor) -> torch.Tensor:
        return x_t.reshape(x_t.shape[0], -1)

    def _reshape_pred(self, pred_flat: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return pred_flat.reshape(ref.shape)

    def _pool_stim_point(self, stim_t: Optional[torch.Tensor], B: int, device, dtype) -> torch.Tensor:
        if not self.has_external_input or self.input_dim <= 0:
            return torch.zeros(B, 0, device=device, dtype=dtype)
        if stim_t is None:
            return torch.zeros(B, self.input_dim, device=device, dtype=dtype)
        if stim_t.dim() == 1:
            stim_t = stim_t.unsqueeze(0)
        pooled = self._pool_stim(stim_t.unsqueeze(1), B, 1, device, dtype)
        return pooled[:, 0]

    def _edge_weight(self, dtype, device) -> torch.Tensor:
        w = self.base_edge_weight.to(device=device, dtype=dtype)
        if self.edge_logit is not None:
            # Around 1.0 at init, positive and bounded enough for stability.
            w = w * (2.0 * torch.sigmoid(self.edge_logit.to(device=device, dtype=dtype)))
        return w

    def step(
        self,
        h: Optional[GraphState],
        x_t: torch.Tensor,
        stim_t: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        horizon_index: Optional[int] = None,
    ):
        B = x_t.shape[0]
        x_flat = self._flatten_state(x_t)
        if x_flat.shape[-1] != self.state_dim:
            raise ValueError(f"CyclicGraphAR expected flattened state_dim={self.state_dim}, got {x_flat.shape[-1]}")
        if h is None:
            h = self.init_state(B, x_t.device, x_t.dtype)

        roi_input = self.x_encoder(x_flat.unsqueeze(-1)) + self.roi_embed.to(device=x_t.device, dtype=x_t.dtype).unsqueeze(0)
        global_ctx = self.global_encoder(x_flat).unsqueeze(1)
        if self.stim_encoder is not None:
            stim_vec = self._pool_stim_point(stim_t, B, x_t.device, x_t.dtype)
            stim_ctx = self.stim_encoder(stim_vec).unsqueeze(1)
        else:
            stim_ctx = torch.zeros(B, 1, self.hidden_dim, device=x_t.device, dtype=x_t.dtype)
        context = global_ctx + stim_ctx

        # Inject current observation every AR step, while carrying a bounded
        # recurrent graph state across time.
        h = roi_input + context + self.carry * self.prev_state_proj(h)
        h = F.layer_norm(h, h.shape[-1:])

        # OOM-safe closed-loop graph update.  Instead of explicit pairwise
        # messages with tensors [B, E, H], aggregate neighbor states as A @ H.
        # This still forms a cyclic bidirectional graph system because the same
        # update operator is recurrently applied over the ROI graph for S steps.
        adj = self.adj_norm.to(device=x_t.device, dtype=x_t.dtype)
        edge_w = self._edge_weight(x_t.dtype, x_t.device)

        internal_changes = []
        msg_norms = []
        for _ in range(self.graph_steps):
            # agg[b, i] = sum_j A[i, j] h[b, j]
            agg = torch.einsum("ij,bjh->bih", adj, h)
            upd = self.update_mlp(torch.cat([h, agg, context.expand(-1, self.state_dim, -1)], dim=-1))
            h_next = F.layer_norm(h + self.alpha * upd, h.shape[-1:])
            internal_changes.append((h_next - h).reshape(B, -1).norm(dim=-1).mean())
            msg_norms.append(agg.reshape(B, -1).norm(dim=-1).mean())
            h = h_next

        delta_flat = self.readout(h).squeeze(-1)
        pred_flat = x_flat + delta_flat if self.residual else delta_flat
        pred = self._reshape_pred(pred_flat, x_t)

        if return_aux:
            delta_norm = delta_flat.reshape(B, -1).norm(dim=-1).mean()
            x_norm = x_flat.reshape(B, -1).norm(dim=-1).mean().clamp_min(1e-8)
            aux = {
                "h_next": h,
                "hidden_norm": h.reshape(B, -1).norm(dim=-1).mean(),
                "alpha_mean": pred.new_tensor(float(self.alpha)),
                "alpha_min": pred.new_tensor(float(self.alpha)),
                "alpha_max": pred.new_tensor(float(self.alpha)),
                "cyclic_delta_norm": delta_norm,
                "cyclic_delta_x_ratio": delta_norm / x_norm,
                "cyclic_internal_change": torch.stack(internal_changes).mean() if internal_changes else pred.new_tensor(0.0),
                "cyclic_msg_norm": torch.stack(msg_norms).mean() if msg_norms else pred.new_tensor(0.0),
                "cyclic_edge_weight_mean": edge_w.mean(),
                "cyclic_edge_weight_max": edge_w.max(),
            }
            return pred, h, aux
        return pred, h

    def burn_in(self, x_seq: torch.Tensor, stim_seq: Optional[torch.Tensor] = None, h0: Optional[GraphState] = None, detach: bool = False):
        B, T = x_seq.shape[:2]
        h = h0 if h0 is not None else self.init_state(B, x_seq.device, x_seq.dtype)
        ctx = torch.no_grad() if detach else torch.enable_grad()
        with ctx:
            for j in range(T):
                stim_j = stim_seq[:, j] if stim_seq is not None else None
                _, h = self.step(h, x_seq[:, j], stim_j, return_aux=False)
        return self.detach_state(h) if detach else h

    def predict_frame_from_history(self, history: torch.Tensor, stim_window: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, W = history.shape[:2]
        h = self.init_state(B, history.device, history.dtype)
        stim = self._pool_stim(stim_window, B, W, history.device, history.dtype) if self.has_external_input else None
        pred = None
        for j in range(W):
            stim_j = stim[:, j] if stim is not None else None
            pred, h = self.step(h, history[:, j], stim_j, return_aux=False)
        if pred is None:
            raise ValueError("history must contain at least one frame")
        return pred

    def step_history(self, history: torch.Tensor, stim_window: Optional[torch.Tensor] = None) -> torch.Tensor:
        frame = self.predict_frame_from_history(history, stim_window)
        return torch.cat([history[:, 1:], frame.unsqueeze(1)], dim=1)

    def forward(self, stim_window: Optional[torch.Tensor], history: torch.Tensor, return_aux: bool = False):
        frame = self.predict_frame_from_history(history, stim_window)
        pred = frame.unsqueeze(1)
        if return_aux:
            return pred, {"pred_frame": frame}
        return pred
