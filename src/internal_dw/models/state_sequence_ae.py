"""Two-stage state-sequence autoencoders for event/state-time dynamics.

Stage 1 (``--statetok_stage ae``):
    Dense clock-time trajectory X[1:T] -> short ordered state-token sequence M[1:L]
    -> reconstruct X[1:T].  Each token has a temporal center, decay and gate.

Stage 2/finetune (``--statetok_stage transition`` or ``finetune``):
    Freeze or lightly tune the AE, extract posterior tokens from the ground-truth
    future trajectory, and train a causal prior/transition model from the input
    history to predicted tokens.  The predicted tokens are decoded back to the
    future path.

The implementation intentionally exposes the same ``generate_path(history,K)``
interface as the other path models so the existing horizon evaluator works.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ordered_centers(raw: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Map raw [B,L] values to ordered centers in [0,1]."""
    intervals = F.softplus(raw) + eps
    total = intervals.sum(dim=-1, keepdim=True).clamp_min(eps)
    left = torch.cumsum(intervals, dim=-1) - intervals
    return ((left + 0.5 * intervals) / total).clamp(0.0, 1.0)


def _decay_from_raw(raw: torch.Tensor, decay_min: float, decay_max: float) -> torch.Tensor:
    return (F.softplus(raw) + float(decay_min)).clamp(max=float(decay_max))


def _time_grid(T: int, device, dtype):
    if T <= 1:
        return torch.zeros(1, device=device, dtype=dtype)
    return torch.linspace(0.0, 1.0, T, device=device, dtype=dtype)


def _encode_weights(centers, decay, T: int):
    """Local time receptive fields normalized over clock time: [B,L,T]."""
    u = _time_grid(T, centers.device, centers.dtype).view(1, 1, T)
    logits = -decay.unsqueeze(-1) * (u - centers.unsqueeze(-1)).abs()
    return torch.softmax(logits, dim=-1)


def _decode_weights_from_encoder_alpha(encoder_weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Tie decoder assignment to encoder receptive fields.

    encoder_weights is alpha with shape [B,L,T] and is normalized over time
    for each token: sum_k alpha_i(k)=1.  The decoder needs weights with
    shape [B,T,L] normalized over tokens for each clock-time step.  We
    therefore use the column-normalized transpose of alpha:

        w_i(k) = alpha_i(k) / sum_j alpha_j(k).

    This enforces responsibility consistency: a token that pools information
    from a time region is also the token used to reconstruct that region.
    """
    w = encoder_weights.transpose(1, 2)  # [B,T,L]
    return w / w.sum(dim=-1, keepdim=True).clamp_min(float(eps))


def _decode_weights(centers, decay, gates, T: int):
    """Tied token-to-time decoder assignment [B,T,L].

    The old implementation defined an independent decoder-side softmax over
    tokens using gates.  That allowed a token to observe one time region through
    alpha_i(k) but reconstruct a different region through w_i(k).  The tied
    version derives w directly from the encoder receptive fields alpha, so
    observation responsibility and reconstruction responsibility are aligned.

    Gates are kept in the signature for checkpoint/API compatibility, but they
    do not change the assignment.  If a gate is needed later, it should be used
    as a token-amplitude modulation rather than as an independent temporal
    reassignment.
    """
    alpha = _encode_weights(centers, decay, T)
    return _decode_weights_from_encoder_alpha(alpha)


def _meta_from_raw(raw, decay_min, decay_max):
    centers = _ordered_centers(raw[..., 0])
    decay = _decay_from_raw(raw[..., 1], decay_min, decay_max)
    gates = raw[..., 2]
    return centers, decay, gates


class LowRankAdapter(nn.Module):
    """Small LoRA-like residual adapter for vector decoder output."""
    def __init__(self, dim: int, rank: int = 0, alpha: float = 0.1):
        super().__init__()
        self.rank = int(rank)
        self.alpha = float(alpha)
        if self.rank > 0:
            self.down = nn.Linear(dim, self.rank, bias=False)
            self.up = nn.Linear(self.rank, dim, bias=False)
            nn.init.zeros_(self.up.weight)
        else:
            self.down = self.up = None

    def forward(self, x):
        if self.rank <= 0:
            return x
        return x + self.alpha * self.up(self.down(x))


class TinyConvEncoder(nn.Module):
    def __init__(self, in_ch, code_ch=64, grid=8, base=64):
        super().__init__()
        self.grid = int(grid)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.GELU(),
            nn.Conv2d(base, base, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((self.grid, self.grid)),
            nn.Conv2d(base, code_ch, 3, padding=1), nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)


class TinyConvDecoder(nn.Module):
    def __init__(self, out_ch, code_ch=64, base=64, adapter_rank=0, adapter_alpha=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(code_ch, base, 3, padding=1), nn.GELU(),
            nn.Conv2d(base, base, 3, padding=1), nn.GELU(),
            nn.Conv2d(base, out_ch, 3, padding=1),
        )
        self.adapter_rank = int(adapter_rank)
        if self.adapter_rank > 0:
            self.adapt_down = nn.Conv2d(out_ch, self.adapter_rank, 1, bias=False)
            self.adapt_up = nn.Conv2d(self.adapter_rank, out_ch, 1, bias=False)
            nn.init.zeros_(self.adapt_up.weight)
            self.adapter_alpha = float(adapter_alpha)
        else:
            self.adapt_down = self.adapt_up = None
            self.adapter_alpha = 0.0
    def forward(self, z, output_size):
        y = F.interpolate(z, size=output_size, mode="bilinear", align_corners=False)
        y = self.net(y)
        if self.adapter_rank > 0:
            y = y + self.adapter_alpha * self.adapt_up(self.adapt_down(y))
        return y


class StateSequenceAEVectorModel(nn.Module):
    is_standard_autoregressive = True
    is_path_generator = True
    is_state_sequence_ae = True
    task_type = "vector"
    uses_future_stimulus = True

    def __init__(self, state_dim:int, input_dim:int=1, window_size:int=4, path_horizon:int=32,
                 num_tokens:int=8, code_dim:int=256, hidden_dim:int=512, has_external_input:bool=False,
                 decay_min:float=0.5, decay_max:float=80.0, adapter_rank:int=0, adapter_alpha:float=0.1,
                 stage:str="ae", freeze_ae_in_transition:bool=True, use_masked_init:bool=False):
        super().__init__()
        self.state_dim=int(state_dim); self.input_dim=int(input_dim); self.window_size=int(window_size)
        self.path_horizon=int(path_horizon); self.num_tokens=int(num_tokens); self.code_dim=int(code_dim)
        self.hidden_dim=int(hidden_dim); self.has_external_input=bool(has_external_input)
        self.decay_min=float(decay_min); self.decay_max=float(decay_max)
        self.stage=str(stage); self.freeze_ae_in_transition=bool(freeze_ae_in_transition)
        self.use_masked_init=bool(use_masked_init)

        self.frame_encoder = nn.Sequential(nn.Linear(self.state_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, code_dim))
        self.seq_summary = nn.Sequential(nn.Linear(code_dim*2, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.posterior_param = nn.Linear(hidden_dim, num_tokens*3)
        self.frame_decoder = nn.Sequential(nn.Linear(code_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, state_dim))
        self.output_adapter = LowRankAdapter(state_dim, adapter_rank, adapter_alpha)

        prior_in = window_size*state_dim
        if self.has_external_input:
            prior_in += window_size*input_dim
        self.prior_context = nn.Sequential(nn.Linear(prior_in, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        # Used by --statetok_use_masked_init: m_1 is extracted by the frozen
        # Stage-1 encoder from [history, masked future], then mapped to the
        # recurrent hidden state.  This preserves an AR transition m_i -> m_{i+1}.
        self.masked_init_context = nn.Sequential(nn.Linear(code_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.prior_init_token = nn.Linear(hidden_dim, code_dim)
        self.prior_cell = nn.GRUCell(code_dim, hidden_dim)
        self.prior_token = nn.Linear(hidden_dim, code_dim)
        self.prior_param = nn.Linear(hidden_dim, 3)
        self.prior_start = nn.Parameter(torch.zeros(code_dim))
        self.future_stim_head = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, state_dim)) if self.has_external_input else None

    def set_stage(self, stage:str, freeze_ae:Optional[bool]=None):
        self.stage=str(stage)
        if freeze_ae is not None:
            self.freeze_ae_in_transition=bool(freeze_ae)

    def configure_trainable(self, stage=None, freeze_ae=None, tune_adapter=True):
        if stage is not None: self.stage=str(stage)
        if freeze_ae is not None: self.freeze_ae_in_transition=bool(freeze_ae)
        for p in self.parameters(): p.requires_grad=True
        if self.stage in ("transition", "finetune") and self.freeze_ae_in_transition:
            for m in [self.frame_encoder, self.seq_summary, self.posterior_param, self.frame_decoder]:
                for p in m.parameters(): p.requires_grad=False
            # Keep only output adapters trainable when requested.
            if tune_adapter:
                for p in self.output_adapter.parameters(): p.requires_grad=True
            else:
                for p in self.output_adapter.parameters(): p.requires_grad=False

    def _pool_stim(self, stim):
        if stim is None: return None
        if stim.dim()>3: return stim.reshape(stim.shape[0], stim.shape[1], -1)
        return stim

    def _context(self, history, stim_window=None):
        B=history.shape[0]
        x=history.reshape(B,self.window_size,-1)
        feat=[x.reshape(B,-1)]
        if self.has_external_input:
            if stim_window is None: stim_window=history.new_zeros(B,self.window_size,self.input_dim)
            stim_window=self._pool_stim(stim_window)
            if stim_window.shape[-1]!=self.input_dim and stim_window.shape[-1] % self.input_dim == 0:
                stim_window=stim_window.view(B, stim_window.shape[1], -1, self.input_dim).mean(dim=2)
            feat.append(stim_window.reshape(B,-1))
        return torch.cat(feat,dim=-1)

    def encode_sequence(self, X, return_aux=False):
        B,T=X.shape[:2]
        x=X.reshape(B,T,-1)
        h=self.frame_encoder(x.reshape(B*T,-1)).view(B,T,self.code_dim)
        summary=torch.cat([h.mean(dim=1), h.std(dim=1, unbiased=False)], dim=-1)
        s=self.seq_summary(summary)
        raw=self.posterior_param(s).view(B,self.num_tokens,3)
        centers,decay,gates=_meta_from_raw(raw,self.decay_min,self.decay_max)
        ew=_encode_weights(centers,decay,T) # B,L,T
        tokens=torch.einsum('blt,btd->bld', ew, h)
        if return_aux:
            return tokens, centers, decay, gates, {"frame_code":h, "enc_weights":ew}
        return tokens, centers, decay, gates

    def decode_tokens(self, tokens, centers, decay, gates, length:int, stim_future=None):
        B,L,D=tokens.shape
        enc_w = _encode_weights(centers, decay, int(length))
        dw = _decode_weights_from_encoder_alpha(enc_w)
        z=torch.einsum('btl,bld->btd', dw, tokens)
        y=self.frame_decoder(z.reshape(B*int(length),D)).view(B,int(length),self.state_dim)
        y=self.output_adapter(y)
        if self.has_external_input and self.future_stim_head is not None and stim_future is not None:
            sf=self._pool_stim(stim_future)
            if sf.shape[-1]!=self.input_dim and sf.shape[-1] % self.input_dim == 0:
                sf=sf.view(B,sf.shape[1],-1,self.input_dim).mean(dim=2)
            if sf.shape[1] >= length and sf.shape[-1] == self.input_dim:
                y = y + self.future_stim_head(sf[:,:length].reshape(B*int(length), self.input_dim)).view(B,int(length),self.state_dim)
        return y, {"state_token_weights": dw, "decode_from_encoder_weights": enc_w, "clock_time_latents": z}

    def reconstruct_sequence(self, X, return_aux=False):
        tok,c,d,g,aux=self.encode_sequence(X, return_aux=True)
        y,daux=self.decode_tokens(tok,c,d,g,X.shape[1])
        allaux={**aux, **daux, "posterior_tokens":tok, "posterior_centers":c, "posterior_decay":d, "posterior_gates":torch.sigmoid(g)}
        return (y, allaux) if return_aux else y

    def _masked_init_token_from_history(self, history, length=None, stim_window=None):
        """Extract m_1 with the Stage-1 sequence encoder from [history, masked future].

        This is intentionally only an initialization mechanism.  It does NOT
        directly predict the full token sequence.  The remaining tokens are
        still generated by the recurrent transition m_i -> m_{i+1}.
        """
        B = history.shape[0]
        future_len = int(length or self.path_horizon)
        T_init = int(self.window_size) + max(1, future_len)
        hist = history.reshape(B, self.window_size, self.state_dim)
        init_seq = hist.new_zeros(B, T_init, self.state_dim)
        init_seq[:, :self.window_size] = hist[:, -self.window_size:]
        tokens, centers, decay, gates = self.encode_sequence(init_seq, return_aux=False)
        return tokens[:, 0], {
            "masked_init_tokens": tokens,
            "masked_init_centers": centers,
            "masked_init_decay": decay,
            "masked_init_gates": torch.sigmoid(gates),
        }

    def prior_tokens(self, history, length=None, stim_window=None):
        B=history.shape[0]; L=self.num_tokens
        extra_aux = {}
        if self.use_masked_init:
            m1, extra_aux = self._masked_init_token_from_history(history, length=length, stim_window=stim_window)
            h = self.masked_init_context(m1)
            inp = m1
            toks=[]; raws=[]
            for i in range(L):
                if i == 0:
                    tok = m1
                    raw = self.prior_param(h)
                else:
                    h=self.prior_cell(inp,h)
                    tok=self.prior_token(h)
                    raw=self.prior_param(h)
                toks.append(tok); raws.append(raw)
                inp=tok
        else:
            ctx=self.prior_context(self._context(history, stim_window))
            h=ctx
            inp=self.prior_start.view(1,-1).expand(B,-1) + self.prior_init_token(ctx)
            toks=[]; raws=[]
            for _ in range(L):
                h=self.prior_cell(inp,h)
                tok=self.prior_token(h)
                raw=self.prior_param(h)
                toks.append(tok); raws.append(raw)
                inp=tok
        tokens=torch.stack(toks,dim=1)
        raw=torch.stack(raws,dim=1)
        centers=_ordered_centers(raw[...,0])
        decay=_decay_from_raw(raw[...,1], self.decay_min, self.decay_max)
        gates=raw[...,2]
        return tokens, centers, decay, gates, extra_aux

    def generate_path(self, history, horizon=None, return_aux=False, stim_window=None, stim_future=None):
        K=int(horizon or self.path_horizon)
        tokens,c,d,g,paux=self.prior_tokens(history, length=K, stim_window=stim_window)
        path,aux=self.decode_tokens(tokens,c,d,g,K,stim_future=stim_future)
        if return_aux:
            aux.update(paux)
            aux.update({"path_pred":path,"prior_tokens":tokens,"prior_centers":c,"prior_decay":d,"prior_gates":torch.sigmoid(g),"path_generator_code":tokens})
            first=path[:,:1]-history.reshape(history.shape[0], self.window_size, -1)[:,-1:].expand(path.shape[0],1,self.state_dim)
            rest=path[:,1:]-path[:,:-1] if K>1 else path[:,:0]
            aux["path_generator_deltas"]=torch.cat([first,rest],dim=1)
            return path, aux
        return path

    def predict_frame_from_history(self, history, stim_window=None):
        return self.generate_path(history,1,False,stim_window=stim_window)[:,0]
    def step_history(self, history):
        frame=self.predict_frame_from_history(history)
        return torch.cat([history[:,1:], frame.unsqueeze(1)], dim=1)
    def encode_path_observable(self, frame): return frame.reshape(frame.shape[0],-1)
    def forward(self, stim_window, history, return_aux=False):
        out=self.generate_path(history,1,return_aux,stim_window=stim_window)
        if return_aux:
            p,a=out; a["pred_frame"]=p[:,0]; return p[:,:1],a
        return out[:,:1]


class StateSequenceAEFieldModel(nn.Module):
    is_standard_autoregressive=True
    is_path_generator=True
    is_state_sequence_ae=True
    task_type="field"

    def __init__(self, field_channels:int, window_size:int=7, path_horizon:int=32, num_tokens:int=8,
                 generator_size:int=8, code_channels:int=64, base_channels:int=64,
                 decay_min:float=0.5, decay_max:float=80.0, adapter_rank:int=0, adapter_alpha:float=0.1,
                 stage:str="ae", freeze_ae_in_transition:bool=True, use_masked_init:bool=False):
        super().__init__()
        self.field_channels=int(field_channels); self.window_size=int(window_size); self.path_horizon=int(path_horizon)
        self.num_tokens=int(num_tokens); self.generator_size=int(generator_size); self.code_channels=int(code_channels)
        self.decay_min=float(decay_min); self.decay_max=float(decay_max); self.stage=str(stage)
        self.freeze_ae_in_transition=bool(freeze_ae_in_transition)
        self.use_masked_init=bool(use_masked_init)
        self.frame_encoder=TinyConvEncoder(field_channels, code_channels, generator_size, base_channels)
        flat=code_channels*generator_size*generator_size
        self.seq_summary=nn.Sequential(nn.Linear(flat*2, base_channels*4), nn.GELU(), nn.LayerNorm(base_channels*4))
        self.posterior_param=nn.Linear(base_channels*4, num_tokens*3)
        self.frame_decoder=TinyConvDecoder(field_channels, code_channels, base_channels, adapter_rank, adapter_alpha)
        # Prior recurrent model operates on flattened tokens.
        hist_flat=window_size*field_channels*generator_size*generator_size
        self.hist_encoder=TinyConvEncoder(field_channels*window_size, code_channels, generator_size, base_channels)
        self.prior_context=nn.Sequential(nn.Linear(flat, base_channels*4), nn.GELU(), nn.LayerNorm(base_channels*4))
        # Used by --statetok_use_masked_init: m_1 comes from the frozen
        # Stage-1 encoder applied to [history, masked future], while m_2..m_L
        # are still produced autoregressively by prior_cell.
        self.masked_init_context=nn.Sequential(nn.Linear(flat, base_channels*4), nn.GELU(), nn.LayerNorm(base_channels*4))
        self.prior_cell=nn.GRUCell(flat, base_channels*4)
        self.prior_token=nn.Linear(base_channels*4, flat)
        self.prior_param=nn.Linear(base_channels*4, 3)
        self.prior_start=nn.Parameter(torch.zeros(flat))

    def set_stage(self, stage, freeze_ae=None):
        self.stage=str(stage)
        if freeze_ae is not None: self.freeze_ae_in_transition=bool(freeze_ae)

    def configure_trainable(self, stage=None, freeze_ae=None, tune_adapter=True):
        if stage is not None: self.stage=str(stage)
        if freeze_ae is not None: self.freeze_ae_in_transition=bool(freeze_ae)
        for p in self.parameters(): p.requires_grad=True
        if self.stage in ("transition","finetune") and self.freeze_ae_in_transition:
            for m in [self.frame_encoder, self.seq_summary, self.posterior_param, self.frame_decoder.net]:
                for p in m.parameters(): p.requires_grad=False
            if self.frame_decoder.adapter_rank > 0 and tune_adapter:
                for p in self.frame_decoder.adapt_down.parameters(): p.requires_grad=True
                for p in self.frame_decoder.adapt_up.parameters(): p.requires_grad=True
            elif self.frame_decoder.adapter_rank > 0:
                for p in self.frame_decoder.adapt_down.parameters(): p.requires_grad=False
                for p in self.frame_decoder.adapt_up.parameters(): p.requires_grad=False

    def encode_sequence(self, X, return_aux=False):
        B,T,C,H,W=X.shape
        h=self.frame_encoder(X.reshape(B*T,C,H,W)).view(B,T,self.code_channels,self.generator_size,self.generator_size)
        hf=h.reshape(B,T,-1)
        summary=torch.cat([hf.mean(dim=1), hf.std(dim=1,unbiased=False)], dim=-1)
        s=self.seq_summary(summary)
        raw=self.posterior_param(s).view(B,self.num_tokens,3)
        centers,decay,gates=_meta_from_raw(raw,self.decay_min,self.decay_max)
        ew=_encode_weights(centers,decay,T)
        toks_flat=torch.einsum('blt,btd->bld', ew, hf)
        tokens=toks_flat.view(B,self.num_tokens,self.code_channels,self.generator_size,self.generator_size)
        if return_aux:
            return tokens,centers,decay,gates,{"frame_code":h,"enc_weights":ew}
        return tokens,centers,decay,gates

    def decode_tokens(self,tokens,centers,decay,gates,length:int,spatial_size:Tuple[int,int]):
        B,L,C,G,_=tokens.shape
        enc_w = _encode_weights(centers, decay, int(length))
        dw = _decode_weights_from_encoder_alpha(enc_w)
        z=torch.einsum('btl,blcgh->btcgh', dw, tokens)
        y=self.frame_decoder(z.reshape(B*int(length),C,G,G), spatial_size).view(B,int(length),self.field_channels,*spatial_size)
        return y,{"state_token_weights": dw, "decode_from_encoder_weights": enc_w, "clock_time_latents": z}

    def reconstruct_sequence(self,X,return_aux=False):
        tok,c,d,g,aux=self.encode_sequence(X,True)
        y,daux=self.decode_tokens(tok,c,d,g,X.shape[1],(X.shape[-2],X.shape[-1]))
        allaux={**aux,**daux,"posterior_tokens":tok,"posterior_centers":c,"posterior_decay":d,"posterior_gates":torch.sigmoid(g)}
        return (y,allaux) if return_aux else y

    def _masked_init_token_from_history(self, history, length=None, stim_window=None):
        """Extract m_1 with the Stage-1 sequence encoder from [history, masked future].

        The future slots are zero-filled.  This is NOT a direct full-token
        prediction: only the first token initializes the AR transition.
        """
        B,W,C,H,Wd=history.shape
        future_len = int(length or self.path_horizon)
        T_init = int(self.window_size) + max(1, future_len)
        init_seq=history.new_zeros(B,T_init,C,H,Wd)
        init_seq[:,:self.window_size]=history[:,-self.window_size:]
        tokens,centers,decay,gates=self.encode_sequence(init_seq,return_aux=False)
        return tokens[:,0], {
            "masked_init_tokens": tokens,
            "masked_init_centers": centers,
            "masked_init_decay": decay,
            "masked_init_gates": torch.sigmoid(gates),
        }

    def prior_tokens(self,history,length=None,stim_window=None):
        B,W,C,H,Wd=history.shape
        extra_aux={}
        if self.use_masked_init:
            m1,extra_aux=self._masked_init_token_from_history(history,length=length,stim_window=stim_window)
            inp=m1.reshape(B,-1)
            h=self.masked_init_context(inp)
            toks=[]; raws=[]
            for i in range(self.num_tokens):
                if i == 0:
                    tok=inp
                    raw=self.prior_param(h)
                else:
                    h=self.prior_cell(inp,h)
                    tok=self.prior_token(h)
                    raw=self.prior_param(h)
                toks.append(tok); raws.append(raw); inp=tok
        else:
            ctx_code=self.hist_encoder(history.reshape(B,W*C,H,Wd)).reshape(B,-1)
            h=self.prior_context(ctx_code)
            inp=self.prior_start.view(1,-1).expand(B,-1)+ctx_code
            toks=[]; raws=[]
            for _ in range(self.num_tokens):
                h=self.prior_cell(inp,h)
                tok=self.prior_token(h)
                raw=self.prior_param(h)
                toks.append(tok); raws.append(raw); inp=tok
        tok=torch.stack(toks,dim=1).view(B,self.num_tokens,self.code_channels,self.generator_size,self.generator_size)
        raw=torch.stack(raws,dim=1)
        centers=_ordered_centers(raw[...,0]); decay=_decay_from_raw(raw[...,1],self.decay_min,self.decay_max); gates=raw[...,2]
        return tok,centers,decay,gates,extra_aux

    def generate_path(self,history,horizon=None,return_aux=False,stim_window=None,stim_future=None):
        K=int(horizon or self.path_horizon)
        tok,c,d,g,paux=self.prior_tokens(history,length=K)
        path,aux=self.decode_tokens(tok,c,d,g,K,(history.shape[-2],history.shape[-1]))
        if return_aux:
            aux.update(paux)
            aux.update({"path_pred":path,"prior_tokens":tok,"prior_centers":c,"prior_decay":d,"prior_gates":torch.sigmoid(g),"path_generator_code":tok})
            first=path[:,:1]-history[:,-1:].expand(path.shape[0],1,*path.shape[2:])
            rest=path[:,1:]-path[:,:-1] if K>1 else path[:,:0]
            aux["path_generator_deltas"]=torch.cat([first,rest],dim=1)
            return path,aux
        return path

    def predict_frame_from_history(self,history,stim_window=None): return self.generate_path(history,1,False)[:,0]
    def step_history(self,history):
        frame=self.predict_frame_from_history(history)
        return torch.cat([history[:,1:],frame.unsqueeze(1)],dim=1)
    def encode_path_observable(self,frame): return frame.reshape(frame.shape[0],-1)
    def forward(self,stim_window,history,return_aux=False):
        out=self.generate_path(history,1,return_aux)
        if return_aux:
            p,a=out; a["pred_frame"]=p[:,0]; return p[:,:1],a
        return out[:,:1]


class StateSequenceDirectFieldModel(StateSequenceAEFieldModel):
    """No-token-transition ablation for state_sequence_ae_field.

    It keeps the exact same posterior encoder/decoder as StateSequenceAEFieldModel,
    so a stage-1 AE checkpoint can be loaded.  The only change is the causal prior:
    all state tokens and metadata are predicted in parallel from the history,
    instead of recurrently rolling m_i -> m_{i+1}.  This tests whether the gain
    comes from state-token iteration or merely direct prediction in token space.
    """
    is_state_sequence_direct = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        flat = self.code_channels * self.generator_size * self.generator_size
        hid = self.prior_context[0].out_features if hasattr(self.prior_context[0], "out_features") else flat
        # Reuse hist_encoder + prior_context from the base class.
        self.direct_tokens = nn.Linear(hid, self.num_tokens * flat)
        self.direct_params = nn.Linear(hid, self.num_tokens * 3)

    def prior_tokens(self, history, length=None, stim_window=None):
        B, W, C, H, Wd = history.shape
        ctx_code = self.hist_encoder(history.reshape(B, W * C, H, Wd)).reshape(B, -1)
        h = self.prior_context(ctx_code)
        tok = self.direct_tokens(h).view(B, self.num_tokens, self.code_channels, self.generator_size, self.generator_size)
        raw = self.direct_params(h).view(B, self.num_tokens, 3)
        centers, decay, gates = _meta_from_raw(raw, self.decay_min, self.decay_max)
        return tok, centers, decay, gates, {}


class StateSequenceDirectVectorModel(StateSequenceAEVectorModel):
    """No-token-transition ablation for state_sequence_ae_vector."""
    is_state_sequence_direct = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Reuse prior_context, but replace recurrent token rollout by parallel heads.
        self.direct_tokens = nn.Linear(self.hidden_dim, self.num_tokens * self.code_dim)
        self.direct_params = nn.Linear(self.hidden_dim, self.num_tokens * 3)

    def prior_tokens(self, history, length=None, stim_window=None):
        B = history.shape[0]
        ctx = self.prior_context(self._context(history, stim_window=stim_window))
        tok = self.direct_tokens(ctx).view(B, self.num_tokens, self.code_dim)
        raw = self.direct_params(ctx).view(B, self.num_tokens, 3)
        centers, decay, gates = _meta_from_raw(raw, self.decay_min, self.decay_max)
        return tok, centers, decay, gates, {}
