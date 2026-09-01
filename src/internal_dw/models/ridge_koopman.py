import torch
import torch.nn as nn


class RidgeKoopmanModel(nn.Module):
    """
    Raw-signal linear autoregressive Koopman/Ridge model.

    The recurrent state is the latest frame x_t, not the flattened history:
        z_t = x_t
        z_{t+1} = A z_t + B u_t + b

    During rollout, the trainer/evaluator appends predictions to the history,
    so encode_state(history) = history[:, -1] makes this a true AR model.

    A is the only matrix used for KG/Gramian powers. B is optional forcing and
    is never exponentiated. For token stimulus [B,W,N_tokens,D], the latest
    token set is mean-pooled to [B,D] by default.
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int = 0,
        has_external_input: bool = False,
        ridge_alpha: float = 1e-2,
        stimulus_pool: str = "mean",
    ):
        super().__init__()

        self.state_dim = int(state_dim)
        # Compatibility with generic Koopman-Gram loss code.
        # RidgeKoopman uses the raw state itself as the latent state, so
        # latent_dim == state_dim and no channelized/encoder path is used.
        self.latent_dim = self.state_dim
        self.channelized_koopman = False
        self.latent_channels = 1
        self.latent_channel_dim = self.state_dim
        self.use_koopman_encoder = False

        self.input_dim = int(input_dim)
        self.has_external_input = bool(has_external_input and input_dim > 0)
        self.ridge_alpha = float(ridge_alpha)
        self.stimulus_pool = str(stimulus_pool)

        self.A = nn.Parameter(torch.eye(self.state_dim))
        self.bias = nn.Parameter(torch.zeros(self.state_dim))
        self.log_force_scale = nn.Parameter(torch.tensor(0.0))

        if self.has_external_input:
            self.B = nn.Parameter(torch.zeros(self.state_dim, self.input_dim))
        else:
            self.B = None

    def encode_state(self, state_window: torch.Tensor) -> torch.Tensor:
        """Use only the latest frame as the AR state."""
        x_t = state_window[:, -1]
        z = x_t.reshape(x_t.shape[0], -1)

        if z.shape[-1] != self.state_dim:
            raise RuntimeError(
                f"RidgeKoopman state dim mismatch: got {z.shape[-1]}, "
                f"expected state_dim={self.state_dim}; "
                f"state_window shape={tuple(state_window.shape)}"
            )
        return z

    def decode_state(self, z: torch.Tensor) -> torch.Tensor:
        """Identity decoder: latent is raw/direct signal."""
        return z

    def effective_A(self) -> torch.Tensor:
        return self.A

    def _pool_latest_external(self, external_window):
        """
        Convert external_window to [B, input_dim].

        Supported input shapes:
        [B, W, input_dim]
        [B, W, N_token, input_dim]
        [B, W, N_token * input_dim]
        """
        if external_window is None:
            return None

        # Take latest external input.
        u_t = external_window[:, -1]

        # Case 1: [B, N_token, input_dim]
        if u_t.dim() == 3:
            if u_t.shape[-1] != self.input_dim:
                raise RuntimeError(
                    f"RidgeKoopman stimulus dim mismatch: got token dim={u_t.shape[-1]}, "
                    f"expected input_dim={self.input_dim}. u_t shape={tuple(u_t.shape)}"
                )
            return u_t.mean(dim=1)

        # Case 2: [B, D]
        if u_t.dim() == 2:
            D = u_t.shape[-1]

            # Already pooled: [B, input_dim]
            if D == self.input_dim:
                return u_t

            # Flattened token features: [B, N_token * input_dim]
            if D % self.input_dim == 0:
                n_token = D // self.input_dim
                u_t = u_t.reshape(u_t.shape[0], n_token, self.input_dim)
                return u_t.mean(dim=1)

            raise RuntimeError(
                f"RidgeKoopman stimulus dim mismatch: got u_t dim={D}, "
                f"expected input_dim={self.input_dim}, or a multiple of it. "
                f"external_window shape={tuple(external_window.shape)}"
            )

        raise RuntimeError(
            f"Unsupported external_window shape={tuple(external_window.shape)}"
        )

    def _pool_latest_external(self, external_window):
        """
        Convert external_window to [B, input_dim].

        Supported:
        [B, W, input_dim]
        [B, W, N_token, input_dim]
        [B, W, N_token * input_dim]
        """
        if external_window is None:
            return None

        u_t = external_window[:, -1]

        # [B, N_token, input_dim] -> [B, input_dim]
        if u_t.dim() == 3:
            if u_t.shape[-1] != self.input_dim:
                raise RuntimeError(
                    f"RidgeKoopman stimulus dim mismatch: got token dim={u_t.shape[-1]}, "
                    f"expected input_dim={self.input_dim}; u_t shape={tuple(u_t.shape)}"
                )
            return u_t.mean(dim=1)

        # [B, D]
        if u_t.dim() == 2:
            D = u_t.shape[-1]

            # already pooled
            if D == self.input_dim:
                return u_t

            # flattened token features: [B, N_token * input_dim]
            if self.input_dim > 0 and D % self.input_dim == 0:
                n_token = D // self.input_dim
                u_t = u_t.reshape(u_t.shape[0], n_token, self.input_dim)
                return u_t.mean(dim=1)

            raise RuntimeError(
                f"RidgeKoopman stimulus dim mismatch: got u_t dim={D}, "
                f"expected input_dim={self.input_dim} or multiple of it; "
                f"external_window shape={tuple(external_window.shape)}"
            )

        raise RuntimeError(f"Unsupported external_window shape={tuple(external_window.shape)}")


    def stimulus_force(self, external_window):
        """
        Return scaled B u_t with shape [B, state_dim].
        """
        if self.B is None or external_window is None:
            return 0.0

        u_t = self._pool_latest_external(external_window)
        force = u_t @ self.B.T
        return torch.exp(self.log_force_scale) * force


    # Compatibility aliases for different KG-loss code paths.
    def force_from_stimulus(self, external_window):
        return self.stimulus_force(external_window)


    def stimulus_force_from_window(self, external_window):
        return self.stimulus_force(external_window)


    def force_from_window(self, external_window):
        return self.stimulus_force(external_window)


    def get_A(self):
        return self.effective_A()


    def rollout_latent(self, z0, external_windows=None, steps=1, return_all=False):
        """
        Simple autoregressive latent rollout.

        z0: [B, state_dim]
        external_windows:
            None, or list/tuple length steps, or tensor with time dimension.
        """
        z = z0
        zs = []

        for k in range(int(steps)):
            ext_k = None
            if external_windows is not None:
                if isinstance(external_windows, (list, tuple)):
                    ext_k = external_windows[k]
                else:
                    # expected [B, steps, ...] or [B, steps, W, ...]
                    ext_k = external_windows[:, k]

            z = self.transition_latent(z, ext_k)
            zs.append(z)

        if return_all:
            return torch.stack(zs, dim=1)
        return z

    def transition_latent(self, z: torch.Tensor, external_window=None) -> torch.Tensor:
        out = z @ self.A.T + self.bias
        if self.B is not None and external_window is not None:
            out = out + self.stimulus_force(external_window)
        return out

    def transition_raw(self, y_t: torch.Tensor, external_window=None) -> torch.Tensor:
        return self.transition_latent(y_t, external_window)

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
            aux["A_sigma"] = torch.linalg.matrix_norm(self.A.detach(), ord=2)
            if self.B is not None:
                aux["force_scale"] = self.B.detach().norm() / max(float(self.B.numel()) ** 0.5, 1.0)

        return pred, aux
