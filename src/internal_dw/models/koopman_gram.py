import torch
import torch.nn as nn
import torch.nn.functional as F


class KoopmanWindowEncoder(nn.Module):
    def __init__(self, fmri_dim, latent_dim, window_size, hidden_dim, mid_dim=None, use_residual=True, dropout=0.0):
        super().__init__()
        self.fmri_dim = int(fmri_dim)
        self.latent_dim = int(latent_dim)
        self.window_size = int(window_size)
        self.use_residual = bool(use_residual)
        mid_dim = int(mid_dim or hidden_dim)
        self.frame_encoder = nn.Sequential(
            nn.Linear(self.fmri_dim, mid_dim), nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(mid_dim, mid_dim), nn.GELU(),
        )
        self.temporal_mixer = nn.Sequential(
            nn.Linear(self.window_size * mid_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, self.latent_dim),
        )
        self.latest_shortcut = nn.Linear(self.fmri_dim, self.latent_dim, bias=False)
        self.shortcut_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, fmri_window):
        b, w, c = fmri_window.shape
        if w != self.window_size or c != self.fmri_dim:
            raise ValueError(f"Expected fMRI window [B,{self.window_size},{self.fmri_dim}], got {tuple(fmri_window.shape)}")
        h = self.frame_encoder(fmri_window)
        z = self.temporal_mixer(h.reshape(b, -1))
        if self.use_residual:
            z = z + torch.tanh(self.shortcut_scale) * self.latest_shortcut(fmri_window[:, -1])
        return z


class KoopmanLatentAdapter(nn.Module):
    def __init__(self, latent_dim, adapter_dim=256, alpha=0.1):
        super().__init__()
        self.alpha = float(alpha)
        self.down = nn.Linear(int(latent_dim), int(adapter_dim))
        self.up = nn.Linear(int(adapter_dim), int(latent_dim))
        nn.init.normal_(self.down.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, z):
        return z + self.alpha * self.up(F.gelu(self.down(z)))


class RawKoopmanGramianModel(nn.Module):
    """Koopman-Gramian predictor with optional learned channelized latent state.

    Raw mode:
        z_t = y_t, z_{t+1} = A z_t + b_theta(s_{t-W:t}).

    Encoder mode:
        z_t = E_phi(y_{t-W:t}), yhat_{t+1}=D_psi(A z_t + b_theta(s)).
    """

    def __init__(
        self, stim_dim, fmri_dim, window_size=8, hidden_dim=512, stim_depth=0, stim_nhead=4,
        dropout=0.0, stable_linear=True, spectral_bound=1.02, residual_transition=True,
        dt=0.1, damping=0.0, force_scale=1.0, use_koopman_encoder=False,
        koopman_latent_dim=None, koopman_encoder_mid_dim=None, koopman_encoder_residual=True,
        koopman_latent_channels=None, koopman_latent_channel_dim=None, koopman_channel_shared_A=True,
        koopman_use_latent_adapter=False, koopman_adapter_dim=256, koopman_adapter_alpha=0.1,
    ):
        super().__init__()
        self.stim_dim = int(stim_dim)
        self.fmri_dim = int(fmri_dim)
        self.window_size = int(window_size)
        self.hidden_dim = int(hidden_dim)
        self.use_koopman_encoder = bool(use_koopman_encoder)
        self.stable_linear = bool(stable_linear)
        self.spectral_bound = float(spectral_bound)
        self.residual_transition = bool(residual_transition)
        self.dt = float(dt)
        self.damping = float(damping)
        self.koopman_channel_shared_A = bool(koopman_channel_shared_A)

        if self.use_koopman_encoder:
            if koopman_latent_channels is not None or koopman_latent_channel_dim is not None:
                if koopman_latent_channels is None or koopman_latent_channel_dim is None:
                    raise ValueError("Set both koopman_latent_channels and koopman_latent_channel_dim.")
                self.latent_channels = int(koopman_latent_channels)
                self.latent_channel_dim = int(koopman_latent_channel_dim)
                implied = self.latent_channels * self.latent_channel_dim
                if koopman_latent_dim is not None and int(koopman_latent_dim) != implied:
                    raise ValueError(f"koopman_latent_dim={koopman_latent_dim} conflicts with channels*dim={implied}")
                self.latent_dim = implied
                self.channelized_koopman = True
            else:
                self.latent_dim = int(koopman_latent_dim or fmri_dim)
                self.latent_channels = 1
                self.latent_channel_dim = self.latent_dim
                self.channelized_koopman = False
        else:
            self.latent_dim = self.fmri_dim
            self.latent_channels = 1
            self.latent_channel_dim = self.fmri_dim
            self.channelized_koopman = False

        self.state_dim = self.latent_channel_dim if self.channelized_koopman else self.latent_dim

        if self.use_koopman_encoder:
            self.koopman_encoder = KoopmanWindowEncoder(
                self.fmri_dim, self.latent_dim, self.window_size, hidden_dim,
                mid_dim=koopman_encoder_mid_dim, use_residual=koopman_encoder_residual, dropout=dropout,
            )
            self.koopman_decoder = nn.Sequential(
                nn.Linear(self.latent_dim, hidden_dim), nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, self.fmri_dim),
            )
        else:
            self.koopman_encoder = None
            self.koopman_decoder = None

        self.koopman_latent_adapter = None
        if self.use_koopman_encoder and koopman_use_latent_adapter:
            self.koopman_latent_adapter = KoopmanLatentAdapter(self.latent_dim, koopman_adapter_dim, koopman_adapter_alpha)

        if self.channelized_koopman and not self.koopman_channel_shared_A:
            self.A_delta = nn.Parameter(torch.zeros(self.latent_channels, self.latent_channel_dim, self.latent_channel_dim))
            nn.init.normal_(self.A_delta, mean=0.0, std=1e-4)
            self.register_buffer("I", torch.eye(self.latent_channel_dim).unsqueeze(0).repeat(self.latent_channels, 1, 1))
        else:
            self.A_delta = nn.Parameter(torch.zeros(self.state_dim, self.state_dim))
            nn.init.normal_(self.A_delta, mean=0.0, std=1e-4)
            self.register_buffer("I", torch.eye(self.state_dim))

        self.stim_in = nn.Linear(self.window_size * self.stim_dim, hidden_dim)
        self.stim_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if stim_depth > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=stim_nhead, dim_feedforward=hidden_dim * 2,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            self.stim_seq_in = nn.Linear(stim_dim, hidden_dim)
            self.stim_encoder = nn.TransformerEncoder(layer, num_layers=stim_depth)
            self.use_stim_transformer = True
        else:
            self.stim_seq_in = None
            self.stim_encoder = None
            self.use_stim_transformer = False

        self.force_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, self.latent_dim),
        )
        self.log_force_scale = nn.Parameter(torch.tensor(float(force_scale)).log())

    def _to_channel(self, z):
        if not self.channelized_koopman or z.dim() == 3:
            return z
        return z.reshape(z.shape[0], self.latent_channels, self.latent_channel_dim)

    def _to_flat(self, z):
        return z if z.dim() == 2 else z.reshape(z.shape[0], self.latent_dim)

    def effective_A(self):
        W = (1.0 - self.damping * self.dt) * self.I + self.dt * self.A_delta if self.residual_transition else self.A_delta
        if not self.stable_linear:
            return W
        sigma = torch.linalg.matrix_norm(W, ord=2).clamp_min(1e-6)
        scale = torch.clamp(self.spectral_bound / sigma, max=1.0)
        return W * scale.view(-1, 1, 1) if W.dim() == 3 else W * scale

    def encode_stim(self, stim_window):
        b, w, e = stim_window.shape
        if self.use_stim_transformer:
            h = self.stim_seq_in(stim_window)
            h = self.stim_encoder(h)
            return h[:, -1]
        return self.stim_dropout(F.gelu(self.stim_in(stim_window.reshape(b, w * e))))

    def encode_state(self, fmri_window):
        if self.use_koopman_encoder:
            z = self.koopman_encoder(fmri_window)
            if self.koopman_latent_adapter is not None:
                z = self.koopman_latent_adapter(z)
            return self._to_channel(z) if self.channelized_koopman else z
        return fmri_window[:, -1]

    def decode_state(self, z):
        return self.koopman_decoder(self._to_flat(z)) if self.use_koopman_encoder else z

    def stimulus_force(self, stim_window):
        u = self.encode_stim(stim_window)
        force = torch.exp(self.log_force_scale) * self.force_head(u)
        if self.channelized_koopman:
            return force.reshape(force.shape[0], self.latent_channels, self.latent_channel_dim)
        return force

    def transition_latent(self, z_t, stim_window):
        A = self.effective_A()
        force = self.stimulus_force(stim_window)
        if self.channelized_koopman:
            zc = self._to_channel(z_t)
            if A.dim() == 3:
                return torch.einsum("bci,coi->bco", zc, A) + force
            return F.linear(zc, A, None) + force
        return F.linear(z_t, A, None) + force

    @torch.no_grad()
    def compute_latent_gramian(self, horizon=100, normalize=True):
        horizon = int(max(horizon, 1))
        A = self.effective_A().detach()
        if A.dim() == 3:
            c, d, _ = A.shape
            G = torch.zeros(c, d, d, device=A.device, dtype=A.dtype)
            Ak = torch.eye(d, device=A.device, dtype=A.dtype).unsqueeze(0).repeat(c, 1, 1)
            for _ in range(horizon):
                G.add_(Ak.transpose(-2, -1) @ Ak)
                Ak = A @ Ak
            if normalize:
                scale = torch.diagonal(G, dim1=-2, dim2=-1).sum(-1) / float(d)
                G = G / scale.clamp_min(1e-8).view(c, 1, 1)
            return G
        d = A.shape[0]
        G = torch.zeros(d, d, device=A.device, dtype=A.dtype)
        Ak = torch.eye(d, device=A.device, dtype=A.dtype)
        for _ in range(horizon):
            G.add_(Ak.transpose(0, 1) @ Ak)
            Ak = A @ Ak
        return G / (torch.trace(G) / float(d) + 1e-8) if normalize else G

    @torch.no_grad()
    def compute_gramian_powers(self, horizon=100):
        horizon = int(max(horizon, 1))
        A = self.effective_A().detach()
        if A.dim() == 3:
            c, d, _ = A.shape
            powers = []
            Ak = torch.eye(d, device=A.device, dtype=A.dtype).unsqueeze(0).repeat(c, 1, 1)
            for _ in range(horizon):
                powers.append(Ak.clone())
                Ak = A @ Ak
            return powers
        d = A.shape[0]
        powers = []
        Ak = torch.eye(d, device=A.device, dtype=A.dtype)
        for _ in range(horizon):
            powers.append(Ak.clone())
            Ak = A @ Ak
        return powers

    def forward(self, stim_window, fmri_window, step_idx=None, return_aux=False):
        z_t = self.encode_state(fmri_window)
        z_next = self.transition_latent(z_t, stim_window)
        pred_frame = self.decode_state(z_next)
        pred = pred_frame.unsqueeze(1)
        if not return_aux:
            return pred
        A = self.effective_A()
        recon = self.decode_state(z_t).unsqueeze(1)
        return pred, {
            "z": z_t,
            "z_next": z_next,
            "recon": recon,
            "A_sigma": torch.linalg.matrix_norm(A.detach(), ord=2).mean(),
            "force_scale": torch.exp(self.log_force_scale.detach()),
            "koopman_encoder": float(self.use_koopman_encoder),
            "koopman_channelized": float(self.channelized_koopman),
            "koopman_channels": float(self.latent_channels),
            "koopman_channel_dim": float(self.latent_channel_dim),
        }
