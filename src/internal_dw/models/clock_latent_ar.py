"""Matched clock-time latent autoregressive baselines.

These models are intended as fair AR baselines for state_sequence_ae_*.
They use a two-stage pipeline with a frame-wise/window-wise AE first and a
clock-time latent transition second:

Stage 1 (``--clocklat_stage ae``):
    x_t -> z_t -> x_t for every clock-time frame.

Stage 2 (``--clocklat_stage transition``):
    freeze/lightly tune the AE, encode the history in clock-time latent space,
    iterate a latent GRU one clock step at a time, and decode each predicted
    latent state back to the original observation space.

The important contrast with state_sequence_ae_* is that the iteration unit here
is still the fixed clock-time step, not a learned state-token/event-time step.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .state_sequence_ae import TinyConvEncoder, TinyConvDecoder, LowRankAdapter


class ClockLatentARFieldModel(nn.Module):
    is_standard_autoregressive = True
    is_path_generator = True
    is_clock_latent_ae = True
    task_type = "field"

    def __init__(
        self,
        field_channels: int,
        window_size: int = 7,
        path_horizon: int = 32,
        generator_size: int = 8,
        code_channels: int = 64,
        base_channels: int = 64,
        hidden_dim: int = 512,
        adapter_rank: int = 0,
        adapter_alpha: float = 0.1,
        stage: str = "ae",
        freeze_ae_in_transition: bool = True,
    ):
        super().__init__()
        self.field_channels = int(field_channels)
        self.window_size = int(window_size)
        self.path_horizon = int(path_horizon)
        self.generator_size = int(generator_size)
        self.code_channels = int(code_channels)
        self.hidden_dim = int(hidden_dim)
        self.stage = str(stage)
        self.freeze_ae_in_transition = bool(freeze_ae_in_transition)

        self.frame_encoder = TinyConvEncoder(field_channels, code_channels, generator_size, base_channels)
        self.frame_decoder = TinyConvDecoder(field_channels, code_channels, base_channels, adapter_rank, adapter_alpha)
        flat = code_channels * generator_size * generator_size
        self.flat_dim = flat
        self.hist_context = nn.GRU(input_size=flat, hidden_size=hidden_dim, batch_first=True)
        self.latent_in = nn.Linear(flat, hidden_dim)
        self.cell = nn.GRUCell(flat, hidden_dim)
        self.to_code = nn.Linear(hidden_dim, flat)

    def set_stage(self, stage: str, freeze_ae: Optional[bool] = None):
        self.stage = str(stage)
        if freeze_ae is not None:
            self.freeze_ae_in_transition = bool(freeze_ae)

    def configure_trainable(self, stage=None, freeze_ae=None, tune_adapter=True):
        if stage is not None:
            self.stage = str(stage)
        if freeze_ae is not None:
            self.freeze_ae_in_transition = bool(freeze_ae)
        for p in self.parameters():
            p.requires_grad = True
        if self.stage in ("transition", "finetune") and self.freeze_ae_in_transition:
            for m in [self.frame_encoder, self.frame_decoder.net]:
                for p in m.parameters():
                    p.requires_grad = False
            if self.frame_decoder.adapter_rank > 0:
                for p in self.frame_decoder.adapt_down.parameters():
                    p.requires_grad = bool(tune_adapter)
                for p in self.frame_decoder.adapt_up.parameters():
                    p.requires_grad = bool(tune_adapter)

    def encode_frame(self, x):
        return self.frame_encoder(x)

    def decode_code(self, z, spatial_size: Tuple[int, int]):
        return self.frame_decoder(z, spatial_size)

    def reconstruct_frames(self, X, return_aux=False):
        # X: [B,T,C,H,W]
        B, T, C, H, W = X.shape
        z = self.encode_frame(X.reshape(B * T, C, H, W))
        y = self.decode_code(z, (H, W)).view(B, T, C, H, W)
        aux = {"clock_latent_codes": z.view(B, T, self.code_channels, self.generator_size, self.generator_size)}
        return (y, aux) if return_aux else y

    def _encode_history_codes(self, history):
        B, W, C, H, Wd = history.shape
        z = self.encode_frame(history.reshape(B * W, C, H, Wd)).view(B, W, -1)
        return z

    def generate_path(self, history, horizon=None, return_aux=False, stim_window=None, stim_future=None):
        K = int(horizon or self.path_horizon)
        B, W, C, H, Wd = history.shape
        hist_z = self._encode_history_codes(history)
        _, h_n = self.hist_context(hist_z)
        h = h_n[-1]
        inp = hist_z[:, -1]
        preds = []
        codes = []
        for _ in range(K):
            h = self.cell(inp, h)
            dz = self.to_code(h)
            # residual latent step keeps the baseline close to clock-time AR
            next_flat = inp + dz
            code = next_flat.view(B, self.code_channels, self.generator_size, self.generator_size)
            frame = self.decode_code(code, (H, Wd))
            preds.append(frame)
            codes.append(code)
            inp = next_flat
        path = torch.stack(preds, dim=1)
        if return_aux:
            latent_codes = torch.stack(codes, dim=1)
            first = path[:, :1] - history[:, -1:].expand(path.shape[0], 1, *path.shape[2:])
            rest = path[:, 1:] - path[:, :-1] if K > 1 else path[:, :0]
            aux = {
                "path_pred": path,
                "clock_latent_pred_codes": latent_codes,
                "path_generator_code": latent_codes,
                "path_generator_deltas": torch.cat([first, rest], dim=1),
            }
            return path, aux
        return path

    def predict_frame_from_history(self, history, stim_window=None):
        return self.generate_path(history, 1, False)[:, 0]

    def step_history(self, history):
        frame = self.predict_frame_from_history(history)
        return torch.cat([history[:, 1:], frame.unsqueeze(1)], dim=1)

    def encode_path_observable(self, frame):
        return frame.reshape(frame.shape[0], -1)

    def forward(self, stim_window, history, return_aux=False):
        out = self.generate_path(history, 1, return_aux)
        if return_aux:
            p, a = out
            a["pred_frame"] = p[:, 0]
            return p[:, :1], a
        return out[:, :1]


class ClockLatentARVectorModel(nn.Module):
    is_standard_autoregressive = True
    is_path_generator = True
    is_clock_latent_ae = True
    task_type = "vector"
    uses_future_stimulus = True

    def __init__(
        self,
        state_dim: int,
        input_dim: int = 1,
        window_size: int = 4,
        path_horizon: int = 32,
        code_dim: int = 256,
        hidden_dim: int = 512,
        has_external_input: bool = False,
        adapter_rank: int = 0,
        adapter_alpha: float = 0.1,
        stage: str = "ae",
        freeze_ae_in_transition: bool = True,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.input_dim = int(input_dim)
        self.window_size = int(window_size)
        self.path_horizon = int(path_horizon)
        self.code_dim = int(code_dim)
        self.hidden_dim = int(hidden_dim)
        self.has_external_input = bool(has_external_input)
        self.stage = str(stage)
        self.freeze_ae_in_transition = bool(freeze_ae_in_transition)

        self.frame_encoder = nn.Sequential(nn.Linear(self.state_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, code_dim))
        self.frame_decoder = nn.Sequential(nn.Linear(code_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, state_dim))
        self.output_adapter = LowRankAdapter(state_dim, adapter_rank, adapter_alpha)
        gru_in = code_dim + (self.input_dim if self.has_external_input else 0)
        self.hist_context = nn.GRU(input_size=gru_in, hidden_size=hidden_dim, batch_first=True)
        self.cell = nn.GRUCell(gru_in, hidden_dim)
        self.to_code = nn.Linear(hidden_dim, code_dim)
        self.future_stim_head = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, state_dim)) if self.has_external_input else None

    def _pool_stim(self, stim):
        if stim is None:
            return None
        if stim.dim() > 3:
            return stim.reshape(stim.shape[0], stim.shape[1], -1)
        return stim

    def set_stage(self, stage: str, freeze_ae: Optional[bool] = None):
        self.stage = str(stage)
        if freeze_ae is not None:
            self.freeze_ae_in_transition = bool(freeze_ae)

    def configure_trainable(self, stage=None, freeze_ae=None, tune_adapter=True):
        if stage is not None:
            self.stage = str(stage)
        if freeze_ae is not None:
            self.freeze_ae_in_transition = bool(freeze_ae)
        for p in self.parameters():
            p.requires_grad = True
        if self.stage in ("transition", "finetune") and self.freeze_ae_in_transition:
            for m in [self.frame_encoder, self.frame_decoder]:
                for p in m.parameters():
                    p.requires_grad = False
            for p in self.output_adapter.parameters():
                p.requires_grad = bool(tune_adapter)

    def encode_frame(self, x):
        return self.frame_encoder(x.reshape(x.shape[0], -1))

    def decode_code(self, z):
        return self.output_adapter(self.frame_decoder(z))

    def reconstruct_frames(self, X, return_aux=False):
        B, T = X.shape[:2]
        x = X.reshape(B * T, -1)
        z = self.frame_encoder(x).view(B, T, self.code_dim)
        y = self.output_adapter(self.frame_decoder(z.reshape(B * T, self.code_dim))).view(B, T, self.state_dim)
        aux = {"clock_latent_codes": z}
        return (y, aux) if return_aux else y

    def _stim_features(self, stim, B, T, device, dtype):
        if not self.has_external_input:
            return None
        if stim is None:
            return torch.zeros(B, T, self.input_dim, device=device, dtype=dtype)
        sf = self._pool_stim(stim)
        if sf.shape[-1] != self.input_dim and sf.shape[-1] % self.input_dim == 0:
            sf = sf.view(B, sf.shape[1], -1, self.input_dim).mean(dim=2)
        return sf

    def generate_path(self, history, horizon=None, return_aux=False, stim_window=None, stim_future=None):
        K = int(horizon or self.path_horizon)
        B, W = history.shape[:2]
        hist_x = history.reshape(B * W, -1)
        hist_z = self.frame_encoder(hist_x).view(B, W, self.code_dim)
        stim_h = self._stim_features(stim_window, B, W, history.device, history.dtype)
        if self.has_external_input:
            hist_in = torch.cat([hist_z, stim_h], dim=-1)
        else:
            hist_in = hist_z
        _, h_n = self.hist_context(hist_in)
        h = h_n[-1]
        inp_z = hist_z[:, -1]
        sf = self._stim_features(stim_future, B, K, history.device, history.dtype)
        preds = []
        codes = []
        for k in range(K):
            cell_in = torch.cat([inp_z, sf[:, k]], dim=-1) if self.has_external_input else inp_z
            h = self.cell(cell_in, h)
            dz = self.to_code(h)
            inp_z = inp_z + dz
            y = self.decode_code(inp_z)
            if self.has_external_input and self.future_stim_head is not None and sf is not None:
                y = y + self.future_stim_head(sf[:, k])
            preds.append(y)
            codes.append(inp_z)
        path = torch.stack(preds, dim=1)
        if return_aux:
            latent_codes = torch.stack(codes, dim=1)
            first = path[:, :1] - history[:, -1:].expand(path.shape[0], 1, *path.shape[2:])
            rest = path[:, 1:] - path[:, :-1] if K > 1 else path[:, :0]
            aux = {
                "path_pred": path,
                "clock_latent_pred_codes": latent_codes,
                "path_generator_code": latent_codes,
                "path_generator_deltas": torch.cat([first, rest], dim=1),
            }
            return path, aux
        return path

    def predict_frame_from_history(self, history, stim_window=None):
        return self.generate_path(history, 1, False, stim_window=stim_window)[:, 0]

    def step_history(self, history):
        frame = self.predict_frame_from_history(history)
        return torch.cat([history[:, 1:], frame.unsqueeze(1)], dim=1)

    def encode_path_observable(self, frame):
        return frame.reshape(frame.shape[0], -1)

    def forward(self, stim_window, history, return_aux=False):
        out = self.generate_path(history, 1, return_aux, stim_window=stim_window)
        if return_aux:
            p, a = out
            a["pred_frame"] = p[:, 0]
            return p[:, :1], a
        return out[:, :1]
