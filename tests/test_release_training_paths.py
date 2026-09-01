from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

import main as release_main
from internal_dw.models.official_state_mamba import OfficialStateMambaARModel
from internal_dw.training.ar_losses import (
    _recurrent_step_maybe_checkpointed,
    _windowed_step_maybe_checkpointed,
    compute_recurrent_state_bptt_loss,
)


class _TinyWindowedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.75))
        self.context = None

    def set_resgrad_context(self, horizon_index, total_horizon):
        self.context = (int(horizon_index), int(total_horizon))

    def forward(self, stim_window, history, return_aux=False):
        del stim_window, return_aux
        return history[:, -1] * self.scale


class _TinyRecurrentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.75))

    def step(
        self,
        hidden,
        x_in,
        stim_in,
        *,
        return_aux,
        horizon_index,
        total_horizon,
    ):
        del stim_in, horizon_index, total_horizon
        previous = hidden[0][0]
        prediction = self.scale * x_in + 0.25 * previous
        next_hidden = ((prediction,),)
        if return_aux:
            return prediction, next_hidden, {}
        return prediction, next_hidden


def test_windowed_activation_checkpoint_retains_parameter_gradient():
    model = _TinyWindowedModel()
    history = torch.randn(2, 4, 3)
    prediction = _windowed_step_maybe_checkpointed(
        model,
        None,
        history,
        horizon_index=1,
        total_horizon=4,
        use_checkpoint=True,
    )
    prediction.square().mean().backward()
    assert model.context == (1, 4)
    assert model.scale.grad is not None
    assert torch.isfinite(model.scale.grad)


def test_recurrent_activation_checkpoint_retains_parameter_gradient():
    model = _TinyRecurrentModel()
    x_in = torch.randn(2, 3)
    hidden = ((torch.zeros_like(x_in),),)
    prediction, next_hidden, aux = _recurrent_step_maybe_checkpointed(
        model,
        hidden,
        x_in,
        None,
        horizon_index=1,
        total_horizon=4,
        use_checkpoint=True,
    )
    prediction.square().mean().backward()
    assert next_hidden[0][0].shape == x_in.shape
    assert aux == {}
    assert model.scale.grad is not None
    assert torch.isfinite(model.scale.grad)


def test_release_worker_enters_trainer_with_autograd_enabled():
    """Catch a stray no_grad decorator on the public training entry point."""

    observed = []
    model = nn.Linear(1, 1)
    args = SimpleNamespace(
        seed=0,
        save_root="unused",
        dataset="mackey_glass",
        model_ckpt_path="",
        mode="train",
        num_epochs=1,
    )

    def fake_train_model(*unused_args, **unused_kwargs):
        observed.append(torch.is_grad_enabled())

    with (
        mock.patch.object(release_main, "setup_ddp"),
        mock.patch.object(release_main, "seed_everything"),
        mock.patch.object(release_main, "build_exp_dir", return_value="unused"),
        mock.patch.object(
            release_main,
            "build_dataloaders",
            return_value=([], [], []),
        ),
        mock.patch.object(release_main, "build_model", return_value=model),
        mock.patch.object(release_main, "maybe_fit_field_normalizer"),
        mock.patch.object(release_main, "configure_trainable"),
        mock.patch.object(
            release_main,
            "maybe_wrap",
            side_effect=lambda value, rank, world_size: value,
        ),
        mock.patch.object(release_main, "_ddp_stage_sync"),
        mock.patch.object(
            release_main,
            "train_model",
            side_effect=fake_train_model,
        ),
        mock.patch.object(release_main, "cleanup_ddp"),
    ):
        release_main.worker(0, args, 1)

    assert observed == [True]


def test_paper_mamba_full_bptt_and_internal_dw_losses_backpropagate():
    """Exercise the two public MG arms that the release README launches."""

    for use_internal_dw, use_checkpoint in ((False, True), (True, False)):
        torch.manual_seed(3)
        model = OfficialStateMambaARModel(
            state_dim=3,
            input_dim=1,
            hidden_dim=8,
            depth=1,
            dropout=0.0,
            has_external_input=False,
            residual=True,
            mamba_d_state=2,
            mamba_d_conv=2,
            mamba_expand=1,
            resgrad_routing=use_internal_dw,
            resgrad_policy="dualwiener" if use_internal_dw else "all",
            dual_wiener_max_horizon=3,
        )
        model.train()
        state = torch.randn(2, 9, 3)
        stimulus = torch.zeros(2, 9, 1)
        args = SimpleNamespace(
            mamba_bptt_horizon=3,
            mamba_burnin=2,
            bptt_detach_period=0,
            artbp_expected_segment_length=0,
            mamba_loss_type="rel_l2",
            mamba_loss_decay=1.0,
            stim_dim=1,
            mamba_train_stride=1,
            mamba_train_starts_per_sequence=1,
            ar_train_random_starts=False,
            ar_shared_rollout_start=False,
            recurrent_grad_checkpoint=use_checkpoint,
            forward_jacobian_lambda=0.0,
            fast_train_logging=True,
        )

        if use_internal_dw:
            model.dual_wiener_begin_batch()
        loss, _ = compute_recurrent_state_bptt_loss(
            model, state, stimulus, args, epoch=1
        )
        assert loss.requires_grad
        if use_internal_dw:
            model.dual_wiener_calibrate()
        loss.backward()
        if use_internal_dw:
            model.dual_wiener_end_batch()

        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
