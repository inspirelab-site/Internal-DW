import torch
import torch.nn as nn


class RidgeWindowModel(nn.Module):
    """
    Trainable old-style dynamic RR predictor with KG-compatible companion state.

    Dynamic predictor, without anchor/fusion:
        x_t = W_x std(vec(x_{t-W:t-1})) + W_u std(vec(u_{t-W:t-1})) + b.

    The rollout/KG latent is the RAW flattened fMRI history window
        s_t = [x_{t-W}, ..., x_{t-1}] in R^{W*D},
    so refresh/free-rollout history remains in the original signal coordinates.

    Feature standardization is absorbed into the effective companion operator:
        W_x std(s) = (W_x / sigma_x) s - W_x (mu_x / sigma_x).
    This makes effective_A() and stimulus_force() exactly consistent with
    forward() and with the old StandardScaler + Ridge feature convention.
    """

    is_ridge_window = True

    def __init__(
        self,
        state_dim: int,
        input_dim: int = 0,
        window_size: int = 1,
        has_external_input: bool = False,
        ridge_alpha: float = 1e-2,
        stimulus_pool: str = "none",
        standardize_features: bool = True,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.input_dim = int(input_dim)
        self.window_size = int(window_size)
        self.has_external_input = bool(has_external_input and input_dim > 0)
        self.ridge_alpha = float(ridge_alpha)
        self.stimulus_pool = str(stimulus_pool)
        self.standardize_features = bool(standardize_features)

        self.latent_dim = self.window_size * self.state_dim
        self.channelized_koopman = False
        self.latent_channels = 1
        self.latent_channel_dim = self.latent_dim
        self.use_koopman_encoder = False
        self.kg_latent_is_window = True

        self.stimulus_window_dim = self.window_size * self.input_dim if self.has_external_input else 0

        # Weights live in standardized feature coordinates, matching old
        # StandardScaler+Ridge.  Bias is in raw output coordinates.
        self.state_weight = nn.Parameter(torch.zeros(self.state_dim, self.latent_dim))
        if self.has_external_input:
            self.stim_weight = nn.Parameter(torch.zeros(self.state_dim, self.stimulus_window_dim))
        else:
            self.stim_weight = None
        self.bias = nn.Parameter(torch.zeros(self.state_dim))
        self.log_force_scale = nn.Parameter(torch.tensor(0.0))

        # Train-set feature statistics.  Defaults make standardization identity
        # until main.py fits them from train_loader.
        self.register_buffer("state_mean", torch.zeros(self.latent_dim))
        self.register_buffer("state_std", torch.ones(self.latent_dim))
        if self.has_external_input:
            self.register_buffer("stim_mean", torch.zeros(self.stimulus_window_dim))
            self.register_buffer("stim_std", torch.ones(self.stimulus_window_dim))
        else:
            self.register_buffer("stim_mean", torch.zeros(0))
            self.register_buffer("stim_std", torch.ones(0))
        self.register_buffer("standardizer_fitted", torch.tensor(False))

        self.reset_parameters()

    def reset_parameters(self):
        # Persistence-like initialization in standardized coordinates.  If
        # features are standardized later, this is only an initialization; the
        # fitted stats will be absorbed consistently by effective_A().
        with torch.no_grad():
            self.state_weight.zero_()
            start = (self.window_size - 1) * self.state_dim
            self.state_weight[:, start:start + self.state_dim].copy_(torch.eye(self.state_dim))
            if self.stim_weight is not None:
                self.stim_weight.zero_()
            self.bias.zero_()
            self.log_force_scale.zero_()

    @torch.no_grad()
    def set_feature_standardizer(self, state_mean, state_std, stim_mean=None, stim_std=None):
        state_mean = state_mean.detach().to(device=self.state_mean.device, dtype=self.state_mean.dtype).flatten()
        state_std = state_std.detach().to(device=self.state_std.device, dtype=self.state_std.dtype).flatten().clamp_min(1e-6)
        if state_mean.numel() != self.latent_dim or state_std.numel() != self.latent_dim:
            raise ValueError(
                f"state standardizer shape mismatch: got mean={tuple(state_mean.shape)}, std={tuple(state_std.shape)}, "
                f"expected {self.latent_dim}"
            )
        self.state_mean.copy_(state_mean)
        self.state_std.copy_(state_std)

        if self.has_external_input:
            if stim_mean is None or stim_std is None:
                raise ValueError("RidgeWindow has external input and requires stim_mean/stim_std")
            stim_mean = stim_mean.detach().to(device=self.stim_mean.device, dtype=self.stim_mean.dtype).flatten()
            stim_std = stim_std.detach().to(device=self.stim_std.device, dtype=self.stim_std.dtype).flatten().clamp_min(1e-6)
            if stim_mean.numel() != self.stimulus_window_dim or stim_std.numel() != self.stimulus_window_dim:
                raise ValueError(
                    f"stim standardizer shape mismatch: got mean={tuple(stim_mean.shape)}, std={tuple(stim_std.shape)}, "
                    f"expected {self.stimulus_window_dim}"
                )
            self.stim_mean.copy_(stim_mean)
            self.stim_std.copy_(stim_std)
        self.standardizer_fitted.fill_(True)

    def _pool_external_window(self, external_window):
        """Return old-RR stimulus features for a full stimulus history window.

        Exact old code path:
            if z_window.ndim == 4: z_window = z_window.mean(axis=2)
            stim_feat = z_window.reshape(B, -1)

        Compatibility addition: in the current loader, [B,W,N,D] may already be
        flattened as [B,W,N*D].  If the per-time flattened dim is a multiple of
        ``self.input_dim`` (normally 1664), we reshape to [B,W,N,D] and mean over
        N.  This recovers the old RR feature convention and avoids using the raw
        425984-d token-flattened vector.
        """
        if external_window is None:
            return None
        u = external_window
        if u.dim() < 3:
            raise RuntimeError(
                f"RidgeWindow expected external_window [B,W,...], got {tuple(external_window.shape)}"
            )
        if u.shape[1] != self.window_size:
            raise RuntimeError(f"RidgeWindow expected external window_size={self.window_size}, got {tuple(u.shape)}")

        if u.dim() == 4:
            # Old RR: [B,W,N,D] -> [B,W,D].
            u = u.mean(dim=2)
        elif u.dim() == 3:
            # Current HCP loader may expose [B,W,N*D].  Recover old pooling if
            # D=self.input_dim divides the flattened dimension.
            per_time_flat = int(u.shape[2])
            if self.input_dim > 0 and per_time_flat != self.input_dim and per_time_flat % self.input_dim == 0:
                n_tokens = per_time_flat // self.input_dim
                u = u.reshape(u.shape[0], u.shape[1], n_tokens, self.input_dim).mean(dim=2)
        else:
            # More exotic [B,W,...] tensors: only old 4D token format is pooled;
            # otherwise flatten remaining dimensions as old _make_input did.
            pass

        per_time_dim = int(u[0, 0].numel())
        if per_time_dim != self.input_dim:
            raise RuntimeError(
                f"RidgeWindow stimulus per-time dim mismatch after old-RR pooling: got {per_time_dim}, "
                f"expected {self.input_dim}; external_window shape={tuple(external_window.shape)}. "
                "Set --stim_dim to the post-pooling per-time feature dimension, e.g. 1664 for 256x1664 flattened tokens."
            )
        return u.reshape(u.shape[0], -1)

    def _standardize_state_flat(self, z):
        if not self.standardize_features:
            return z
        return (z - self.state_mean.view(1, -1)) / self.state_std.view(1, -1).clamp_min(1e-6)

    def _standardize_stim_flat(self, u):
        if not self.standardize_features:
            return u
        return (u - self.stim_mean.view(1, -1)) / self.stim_std.view(1, -1).clamp_min(1e-6)

    def encode_state(self, state_window: torch.Tensor) -> torch.Tensor:
        # KG/rollout latent is RAW history window, not standardized feature.
        if state_window.shape[1] != self.window_size:
            raise RuntimeError(
                f"RidgeWindow expected window_size={self.window_size}, got state_window shape={tuple(state_window.shape)}"
            )
        return state_window.reshape(state_window.shape[0], -1)

    def decode_state(self, z: torch.Tensor) -> torch.Tensor:
        return z[..., -self.state_dim:]

    def effective_A(self) -> torch.Tensor:
        """Raw-coordinate companion transition over flattened history state [W*D]."""
        D = self.state_dim
        W = self.window_size
        A = self.state_weight.new_zeros(self.latent_dim, self.latent_dim)
        if W > 1:
            A[: (W - 1) * D, D: W * D] = torch.eye((W - 1) * D, device=A.device, dtype=A.dtype)

        if self.standardize_features:
            bottom = self.state_weight / self.state_std.view(1, -1).clamp_min(1e-6)
        else:
            bottom = self.state_weight
        A[(W - 1) * D: W * D, :] = bottom
        return A

    def _state_standardization_offset(self):
        if not self.standardize_features:
            return self.bias.new_zeros(self.state_dim)
        state_offset = self.state_mean / self.state_std.clamp_min(1e-6)
        return self.state_weight @ state_offset

    def stimulus_force(self, external_window):
        """Return raw-coordinate latent forcing [B,W*D], nonzero only in final frame block."""
        if external_window is None:
            raise ValueError("RidgeWindow.stimulus_force requires external_window for batch size/alignment")
        Bsz = external_window.shape[0]
        force = external_window.new_zeros(Bsz, self.latent_dim)

        # Bias plus the constant term induced by state feature standardization.
        last = self.bias.unsqueeze(0).expand(Bsz, -1) - self._state_standardization_offset().view(1, -1)

        if self.stim_weight is not None:
            u_raw = self._pool_external_window(external_window)
            u_std = self._standardize_stim_flat(u_raw)
            last = last + torch.exp(self.log_force_scale) * (u_std @ self.stim_weight.T)
        force[:, -self.state_dim:] = last
        return force

    # Compatibility aliases used by existing KG-loss code.
    def force_from_stimulus(self, external_window):
        return self.stimulus_force(external_window)

    def stimulus_force_from_window(self, external_window):
        return self.stimulus_force(external_window)

    def force_from_window(self, external_window):
        return self.stimulus_force(external_window)

    def get_A(self):
        return self.effective_A()

    def transition_latent(self, z: torch.Tensor, external_window=None) -> torch.Tensor:
        return z @ self.effective_A().T + self.stimulus_force(external_window)

    def transition_raw(self, y_t: torch.Tensor, external_window=None) -> torch.Tensor:
        return self.transition_latent(y_t, external_window)

    def ridge_l2_penalty(self):
        # Mean squared scaled-coordinate coefficients, matching the parameter
        # space of StandardScaler+Ridge and keeping scale independent of feature count.
        terms = [self.state_weight.square().mean()]
        if self.stim_weight is not None:
            terms.append(self.stim_weight.square().mean())
        return torch.stack(terms).mean()

    def forward(self, external_window, state_window, return_aux: bool = False):
        z = self.encode_state(state_window)
        z_next = self.transition_latent(z, external_window)
        pred = self.decode_state(z_next).unsqueeze(1)
        if not return_aux:
            return pred
        aux = {
            "z": z,
            "z_next": z_next,
            "A": self.effective_A(),
        }
        with torch.no_grad():
            aux["A_sigma"] = torch.linalg.matrix_norm(self.effective_A().detach(), ord=2)
            if self.stim_weight is not None:
                aux["force_scale"] = self.stim_weight.detach().norm() / max(float(self.stim_weight.numel()) ** 0.5, 1.0)
            else:
                aux["force_scale"] = torch.tensor(0.0, device=z.device)
        return pred, aux
