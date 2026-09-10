"""Paper-facing model registry.

The public release intentionally exposes only the two backbones used in the
paper. Historical architecture searches and abandoned model families are not
part of the reproducibility surface.
"""

from __future__ import annotations

from .official_state_mamba import OfficialStateMambaARModel
from .unet_field import UNetFieldModel


_MODELS = {
    "official_mamba_state": OfficialStateMambaARModel,
    "unet_field": UNetFieldModel,
}


def list_models() -> list[str]:
    """Return the model names supported by the paper release."""

    return sorted(_MODELS)


def _field_shape(args):
    is_field = (
        str(getattr(args, "dataset_task_type", "")) == "field2d"
        or str(getattr(args, "dataset", "")) == "the_well"
    )
    if not is_field:
        return None

    channels = int(getattr(args, "field_channels", 0))
    height = int(getattr(args, "field_height", 0))
    width = int(getattr(args, "field_width", 0))
    if min(channels, height, width) <= 0:
        raise ValueError(
            "Field data require inferred field_channels, field_height, and "
            "field_width before model construction."
        )
    return channels, height, width


def _dual_wiener_kwargs(args) -> dict:
    return {
        "dual_wiener_ema": getattr(args, "dual_wiener_ema", 0.95),
        "dual_wiener_residual_ema": getattr(
            args, "dual_wiener_residual_ema", 0.99
        ),
        "dual_wiener_warmup_batches": getattr(
            args, "dual_wiener_warmup_batches", 8
        ),
        "dual_wiener_probe_every": getattr(args, "dual_wiener_probe_every", 4),
        "dual_wiener_min_probes": getattr(args, "dual_wiener_min_probes", 1),
        "dual_wiener_noise_model": getattr(
            args, "dual_wiener_noise_model", "diagonal_gaussian"
        ),
        "dual_wiener_max_horizon": getattr(
            args, "dual_wiener_max_horizon", 1024
        ),
    }


def build_model(args, rank: int = 0):
    """Build one paper backbone and move it to the selected CUDA device."""

    model_name = str(args.model_name)
    if model_name not in _MODELS:
        raise ValueError(
            f"Unknown model_name={model_name}. Available: {list_models()}"
        )

    routing = {
        "resgrad_routing": getattr(args, "resgrad_routing", False),
        "resgrad_policy": getattr(args, "resgrad_policy", "all"),
    }

    if model_name == "official_mamba_state":
        state_shape = _field_shape(args)
        state_dim = int(args.roi_dim)
        if state_shape is not None:
            state_dim = state_shape[0] * state_shape[1] * state_shape[2]
            args.roi_dim = state_dim

        model = OfficialStateMambaARModel(
            state_dim=state_dim,
            input_dim=int(getattr(args, "stim_dim", 0)),
            state_shape=state_shape,
            hidden_dim=getattr(args, "simple_hidden_dim", 512),
            depth=getattr(args, "simple_depth", 4),
            dropout=getattr(args, "simple_dropout", 0.0),
            has_external_input=getattr(
                args, "dataset_has_external_input", False
            ),
            residual=getattr(args, "simple_residual", True),
            mamba_d_state=getattr(args, "mamba_d_state", 16),
            mamba_d_conv=getattr(args, "mamba_d_conv", 4),
            mamba_expand=getattr(args, "mamba_expand", 2),
            untie_groups=getattr(args, "untie_groups", 1),
            **routing,
            **_dual_wiener_kwargs(args),
        )
        model.compact_recurrent_logging = bool(getattr(args, "compact_recurrent_logging", False))
        return model.cuda(rank)

    if int(getattr(args, "field_channels", 0)) <= 0:
        raise ValueError(
            "unet_field requires positive field_channels; the field dataset "
            "loader normally infers this value."
        )
    model = UNetFieldModel(
        field_channels=int(args.field_channels),
        window_size=int(args.window_size),
        base_channels=getattr(args, "unet_base_channels", 64),
        depth=getattr(args, "unet_depth", 4),
        channel_mult=getattr(args, "unet_channel_mult", 2),
        groups=getattr(args, "unet_groups", 8),
        use_grid=getattr(args, "unet_use_grid", True),
        normalize=getattr(args, "unet_normalize", True),
        field_height=getattr(args, "field_height", 0),
        field_width=getattr(args, "field_width", 0),
        **routing,
        **_dual_wiener_kwargs(args),
    )
    return model.cuda(rank)
