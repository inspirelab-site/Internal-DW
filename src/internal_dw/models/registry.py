from .koopman_gram import RawKoopmanGramianModel
from .koopman_field_gram import KoopmanFieldGramianModel
from .koopman_field_corrector import KoopmanFieldHardCorrectorModel
from .ridge_koopman import RidgeKoopmanModel
from .fno_field import FNOFieldModel
from .unet_field import UNetFieldModel
from .path_generator_field import LowDimPathGeneratorFieldModel
from .direct_path_decoder_field import DirectPathDecoderFieldModel
from .ridge_window import RidgeWindowModel
from .state_token_path import StateTokenPathFieldModel, StateTokenPathVectorModel
from .state_sequence_ae import (
    StateSequenceAEFieldModel, StateSequenceAEVectorModel,
    StateSequenceDirectFieldModel, StateSequenceDirectVectorModel,
)
from .clock_latent_ar import ClockLatentARFieldModel, ClockLatentARVectorModel
from .simple_ar import CNNFieldARModel, ConvGRUFieldARModel, TransformerVectorARModel, BridgeTransformerVectorARModel, TCNVectorARModel, ResNetVectorARModel
from .potential_ae import PotentialAEVectorModel
from .official_state_mamba import OfficialStateMambaARModel
from .official_mamba_field import OfficialFieldMambaARModel
from .residual_gru import ResidualGRUARModel
from .predictive_subspace_attention import CausalQueryPredictiveSubspaceARModel
from .official_pc_mamba_state import OfficialPredictiveCodingMambaARModel
from .official_atlas_mamba_state import OfficialAtlasMambaARModel
from .official_tangent_atlas_mamba_state import OfficialTangentAtlasMambaARModel
from .official_shadow_perturb_mamba import OfficialShadowPerturbMambaARModel
from .official_mamba_fixeda_perturb import OfficialMambaFixedAPerturbARModel
from .cyclic_graph_ar import CyclicGraphARModel
from .diagnostic_graph_ar import FeedForwardGraphAR, SimpleDeltaAR

_MODELS = {
    "koopman_raw_gramian": RawKoopmanGramianModel,
    "koopman_field_gramian": KoopmanFieldGramianModel,
    "koopman_field_hard_corrector": KoopmanFieldHardCorrectorModel,
    "ridge_koopman": RidgeKoopmanModel,
    "fno_field": FNOFieldModel,
    "residual_fno_field": FNOFieldModel,
    "unet_field": UNetFieldModel,
    "residual_gru": ResidualGRUARModel,
    "path_generator_field": LowDimPathGeneratorFieldModel,
    "direct_path_decoder_field": DirectPathDecoderFieldModel,
    "ridge_window": RidgeWindowModel,
    "state_token_path_field": StateTokenPathFieldModel,
    "state_token_path_vector": StateTokenPathVectorModel,
    "state_sequence_ae_field": StateSequenceAEFieldModel,
    "state_sequence_ae_vector": StateSequenceAEVectorModel,
    "state_sequence_direct_field": StateSequenceDirectFieldModel,
    "state_sequence_direct_vector": StateSequenceDirectVectorModel,
    "clock_latent_ar_field": ClockLatentARFieldModel,
    "clock_latent_ar_vector": ClockLatentARVectorModel,
    "cnn_field": CNNFieldARModel,
    "convgru_field": ConvGRUFieldARModel,
    "transformer_vector": TransformerVectorARModel,
    "bridge_transformer_vector": BridgeTransformerVectorARModel,
    "tcn_vector": TCNVectorARModel,
    "resnet_vector": ResNetVectorARModel,
    "official_mamba_state": OfficialStateMambaARModel,
    "official_mamba_field": OfficialFieldMambaARModel,
    "predictive_subspace_attention": CausalQueryPredictiveSubspaceARModel,
    "official_pc_mamba_state": OfficialPredictiveCodingMambaARModel,
    "official_atlas_mamba_state": OfficialAtlasMambaARModel,
    "official_tangent_atlas_mamba_state": OfficialTangentAtlasMambaARModel,
    "official_shadow_perturb_mamba": OfficialShadowPerturbMambaARModel,
    "official_mamba_fixeda_perturb": OfficialMambaFixedAPerturbARModel,
    "cyclic_graph_ar": CyclicGraphARModel,
    "simple_delta_ar": SimpleDeltaAR,
    "feedforward_graph_ar": FeedForwardGraphAR,
    "potential_ae_vector": PotentialAEVectorModel,
}


def list_models():
    return sorted(_MODELS.keys())


def build_model(args, rank=0):
    if args.model_name not in _MODELS:
        raise ValueError(
            f"Unknown model_name={args.model_name}. Available: {list_models()}"
        )

    cls = _MODELS[args.model_name]

    # ------------------------------------------------------------
    # Ridge Koopman model: raw/direct signal linear AR baseline
    # ------------------------------------------------------------
    if args.model_name == "ridge_koopman":
        kwargs = dict(
            state_dim=args.roi_dim,
            input_dim=args.stim_dim,
            has_external_input=getattr(args, "dataset_has_external_input", False),
            ridge_alpha=getattr(args, "ridge_alpha", 1e-2),
        )
        model = cls(**kwargs)
        return model.cuda(rank)


    # ------------------------------------------------------------
    # Old-style trainable window RR dynamic predictor
    # ------------------------------------------------------------
    if args.model_name == "ridge_window":
        kwargs = dict(
            state_dim=args.roi_dim,
            input_dim=args.stim_dim,
            window_size=args.window_size,
            has_external_input=getattr(args, "dataset_has_external_input", False),
            ridge_alpha=getattr(args, "ridge_alpha", 1e-2),
            standardize_features=getattr(args, "ridge_window_standardize", True),
        )
        model = cls(**kwargs)
        return model.cuda(rank)







    # ------------------------------------------------------------
    # Stage-A potential-state autoencoder
    # ------------------------------------------------------------
    if args.model_name == "potential_ae_vector":
        model = cls(
            state_dim=args.roi_dim,
            latent_dim=getattr(args, "potential_latent_dim", 64),
            hidden_dim=getattr(args, "potential_hidden_dim", 256),
            depth=getattr(args, "potential_depth", 2),
            dropout=getattr(args, "potential_dropout", 0.0),
            potential_hidden_dim=getattr(args, "potential_value_hidden_dim", None),
            potential_depth=getattr(args, "potential_value_depth", None),
            input_dim=getattr(args, "stim_dim", 0) if bool(getattr(args, "dataset_has_external_input", False)) else getattr(args, "stim_dim", 0),
            stim_hidden_dim=getattr(args, "potential_stim_hidden_dim", None),
            stim_depth=getattr(args, "potential_stim_depth", None),
            stim_context_len=getattr(args, "potential_stim_context_len", 1),
        )
        return model.cuda(rank)

    # ------------------------------------------------------------
    # Clean standard AR baselines for compiler/surrogate-gradient experiments
    # ------------------------------------------------------------
    if args.model_name == "cnn_field":
        if getattr(args, "field_channels", 0) <= 0:
            raise ValueError("cnn_field requires --field_channels > 0; for dataset=the_well this is normally inferred automatically.")
        model = cls(
            field_channels=args.field_channels,
            window_size=args.window_size,
            hidden_channels=getattr(args, "simple_hidden_dim", 96),
            depth=getattr(args, "simple_depth", 5),
            groups=getattr(args, "simple_groups", 8),
            use_grid=getattr(args, "simple_use_grid", True),
            normalize=getattr(args, "simple_normalize", True),
            residual=getattr(args, "simple_residual", True),
        )
        return model.cuda(rank)

    if args.model_name == "convgru_field":
        if getattr(args, "field_channels", 0) <= 0:
            raise ValueError("convgru_field requires --field_channels > 0; for dataset=the_well this is normally inferred automatically.")
        model = cls(
            field_channels=args.field_channels,
            window_size=args.window_size,
            hidden_channels=getattr(args, "simple_hidden_dim", 96),
            groups=getattr(args, "simple_groups", 8),
            use_grid=getattr(args, "simple_use_grid", True),
            normalize=getattr(args, "simple_normalize", True),
            residual=getattr(args, "simple_residual", True),
        )
        return model.cuda(rank)

    if args.model_name == "transformer_vector":
        model = cls(
            state_dim=args.roi_dim,
            input_dim=args.stim_dim,
            window_size=args.window_size,
            hidden_dim=getattr(args, "simple_hidden_dim", 512),
            depth=getattr(args, "simple_depth", 4),
            nhead=getattr(args, "simple_nhead", 8),
            dropout=getattr(args, "simple_dropout", 0.1),
            has_external_input=getattr(args, "dataset_has_external_input", False),
            residual=getattr(args, "simple_residual", True),
            ff_mult=getattr(args, "simple_ff_mult", 2),
            enable_error_transport=getattr(args, "etm_loss", False),
            etm_diag_base=getattr(args, "etm_diag_base", 0.0),
            etm_diag_scale=getattr(args, "etm_diag_scale", 0.5),
            etm_transport_param=getattr(args, "etm_transport_param", "full"),
            etm_lowrank_rank=getattr(args, "etm_lowrank_rank", 16),
            etm_metric_atoms=getattr(args, "etm_metric_atoms", 64),
            etm_metric_temperature=getattr(args, "etm_metric_temperature", 1.0),
            resgrad_routing=getattr(args, "resgrad_routing", False),
            resgrad_policy=getattr(args, "resgrad_policy", "all"),
            resgrad_block_gate=getattr(args, "resgrad_block_gate", 1.0),
            resgrad_ratio_threshold=getattr(args, "resgrad_ratio_threshold", 0.05),
            resgrad_keep_every=getattr(args, "resgrad_keep_every", 8),
            resgrad_keep_tail=getattr(args, "resgrad_keep_tail", 0),
            dual_wiener_ema=getattr(args, "dual_wiener_ema", 0.95),
            dual_wiener_residual_ema=getattr(args, "dual_wiener_residual_ema", 0.99),
            dual_wiener_warmup_batches=getattr(args, "dual_wiener_warmup_batches", 8),
            dual_wiener_probe_every=getattr(args, "dual_wiener_probe_every", 4),
            dual_wiener_min_probes=getattr(args, "dual_wiener_min_probes", 1),
            dual_wiener_noise_model=getattr(args, "dual_wiener_noise_model", "diagonal_gaussian"),
            dual_wiener_max_horizon=getattr(args, "dual_wiener_max_horizon", 1024),
        )
        return model.cuda(rank)

    if args.model_name == "resnet_vector":
        model = cls(
            state_dim=args.roi_dim,
            input_dim=args.stim_dim,
            window_size=args.window_size,
            hidden_dim=getattr(args, "simple_hidden_dim", 512),
            depth=getattr(args, "simple_depth", 4),
            hidden_mult=getattr(args, "resnet_hidden_mult", 2),
            dropout=getattr(args, "simple_dropout", 0.0),
            has_external_input=getattr(args, "dataset_has_external_input", False),
            residual=getattr(args, "simple_residual", True),
            resgrad_routing=getattr(args, "resgrad_routing", False),
            resgrad_policy=getattr(args, "resgrad_policy", "all"),
            resgrad_block_gate=getattr(args, "resgrad_block_gate", 1.0),
            resgrad_ratio_threshold=getattr(args, "resgrad_ratio_threshold", 0.05),
            resgrad_keep_every=getattr(args, "resgrad_keep_every", 8),
            resgrad_keep_tail=getattr(args, "resgrad_keep_tail", 0),
        )
        return model.cuda(rank)

    if args.model_name == "bridge_transformer_vector":
        model = cls(
            state_dim=args.roi_dim,
            input_dim=args.stim_dim,
            window_size=args.window_size,
            hidden_dim=getattr(args, "simple_hidden_dim", 192),
            depth=getattr(args, "simple_depth", 2),
            nhead=getattr(args, "simple_nhead", 4),
            dropout=getattr(args, "simple_dropout", 0.1),
            has_external_input=getattr(args, "dataset_has_external_input", False),
            residual=getattr(args, "simple_residual", False),
            ff_mult=getattr(args, "simple_ff_mult", 2),
            enable_error_transport=getattr(args, "etm_loss", False),
            etm_diag_base=getattr(args, "etm_diag_base", 0.0),
            etm_diag_scale=getattr(args, "etm_diag_scale", 0.5),
            etm_transport_param=getattr(args, "etm_transport_param", "full"),
            etm_lowrank_rank=getattr(args, "etm_lowrank_rank", 16),
            etm_metric_atoms=getattr(args, "etm_metric_atoms", 64),
            etm_metric_temperature=getattr(args, "etm_metric_temperature", 1.0),
        )
        return model.cuda(rank)



    # ------------------------------------------------------------
    # Diagnostic controls for cyclic graph AR
    # ------------------------------------------------------------
    if args.model_name in {"simple_delta_ar", "feedforward_graph_ar"}:
        state_dim = int(args.roi_dim)
        model = cls(
            state_dim=state_dim,
            input_dim=int(getattr(args, "stim_dim", 0)),
            hidden_dim=getattr(args, "simple_hidden_dim", 128),
            depth=getattr(args, "simple_depth", 3),
            dropout=getattr(args, "simple_dropout", 0.0),
            has_external_input=getattr(args, "dataset_has_external_input", False),
            residual=getattr(args, "simple_residual", True),
            graph_steps=getattr(args, "cyclic_graph_steps", 3),
            topk=getattr(args, "cyclic_graph_topk", 2),
            alpha=getattr(args, "cyclic_graph_alpha", 0.10),
            carry=getattr(args, "cyclic_graph_carry", 0.25),
            edge_type=getattr(args, "cyclic_graph_edge_type", "ring"),
            message_hidden_mult=getattr(args, "cyclic_graph_message_hidden_mult", 2),
        )
        return model.cuda(rank)


    # Second recurrent-state family (gated RNN rather than SSM).  Its residual
    # path IS the temporal carrier, h_{t+1} = h_t + F, which is the condition
    # under which routing means "control temporal credit" at all.
    if args.model_name == "residual_gru":
        model = cls(
            roi_dim=int(args.roi_dim),
            stim_dim=int(getattr(args, "stim_dim", 0)),
            hidden_dim=int(getattr(args, "simple_hidden_dim", 512)),
            depth=int(getattr(args, "simple_depth", 4)),
            dropout=float(getattr(args, "simple_dropout", 0.0)),
            has_external_input=bool(getattr(args, "dataset_has_external_input", True)),
            residual=bool(getattr(args, "simple_residual", True)),
            resgrad_routing=bool(getattr(args, "resgrad_routing", False)),
            resgrad_policy=str(getattr(args, "resgrad_policy", "all")),
            resgrad_block_gate=float(getattr(args, "resgrad_block_gate", 1.0)),
            resgrad_ratio_threshold=float(getattr(args, "resgrad_ratio_threshold", 0.05)),
            resgrad_cut_state=bool(getattr(args, "resgrad_cut_state", False)),
        )
        return model.cuda(rank)

    if args.model_name in {"official_mamba_state", "official_mamba_field", "official_pc_mamba_state", "official_atlas_mamba_state", "official_tangent_atlas_mamba_state", "official_shadow_perturb_mamba", "official_mamba_fixeda_perturb", "cyclic_graph_ar", "predictive_subspace_attention"}:
        state_shape = None
        if args.model_name == "official_mamba_field":
            # This backbone reshapes the flat state internally, so it needs the grid
            # even when the loader serves a flat vector (e.g. WeatherBench flat mode).
            C = int(getattr(args, "field_channels", 0))
            H = int(getattr(args, "field_height", 0))
            Wd = int(getattr(args, "field_width", 0))
            if C <= 0 or H <= 0 or Wd <= 0:
                raise ValueError(
                    "official_mamba_field requires field_channels/height/width; the "
                    "dataset loader must record the grid shape."
                )
            state_shape = (C, H, Wd)
        elif str(getattr(args, "dataset_task_type", "")) == "field2d" or str(getattr(args, "dataset", "")) == "the_well":
            C = int(getattr(args, "field_channels", 0))
            H = int(getattr(args, "field_height", 0))
            Wd = int(getattr(args, "field_width", 0))
            if C <= 0 or H <= 0 or Wd <= 0:
                raise ValueError(f"{args.model_name} on the_well requires inferred field_channels/height/width")
            state_shape = (C, H, Wd)

        state_dim = int(args.roi_dim)
        if state_shape is not None:
            # Field datasets are flattened inside the Mamba-vector models.
            # Keep this robust even if args.roi_dim was not inferred upstream.
            state_dim = int(state_shape[0]) * int(state_shape[1]) * int(state_shape[2])
            args.roi_dim = state_dim

        common = dict(
            state_dim=state_dim,
            input_dim=args.stim_dim,
            state_shape=state_shape,
            hidden_dim=getattr(args, "simple_hidden_dim", 512),
            depth=getattr(args, "simple_depth", 4),
            dropout=getattr(args, "simple_dropout", 0.0),
            has_external_input=getattr(args, "dataset_has_external_input", False),
            residual=getattr(args, "simple_residual", True),
            mamba_d_state=getattr(args, "mamba_d_state", 16),
            mamba_d_conv=getattr(args, "mamba_d_conv", 4),
            mamba_expand=getattr(args, "mamba_expand", 2),
        )
        if args.model_name == "cyclic_graph_ar":
            model = cls(
                **common,
                graph_steps=getattr(args, "cyclic_graph_steps", 5),
                topk=getattr(args, "cyclic_graph_topk", 8),
                alpha=getattr(args, "cyclic_graph_alpha", 0.10),
                carry=getattr(args, "cyclic_graph_carry", 0.25),
                edge_type=getattr(args, "cyclic_graph_edge_type", "ring"),
                learned_edge_gate=getattr(args, "cyclic_graph_learned_edge_gate", True),
                message_hidden_mult=getattr(args, "cyclic_graph_message_hidden_mult", 2),
            )
        elif args.model_name in ("official_mamba_state", "official_mamba_field"):
            extra = {}
            if args.model_name == "official_mamba_field":
                extra = dict(
                    field_base_ch=getattr(args, "field_base_channels", 64),
                    field_groups=getattr(args, "field_groups", 8),
                )
            model = cls(
                **common,
                **extra,
                resgrad_routing=getattr(args, "resgrad_routing", False),
                resgrad_policy=getattr(args, "resgrad_policy", "all"),
                resgrad_block_gate=getattr(args, "resgrad_block_gate", 1.0),
                resgrad_ratio_threshold=getattr(args, "resgrad_ratio_threshold", 0.05),
                resgrad_keep_every=getattr(args, "resgrad_keep_every", 8),
                resgrad_keep_tail=getattr(args, "resgrad_keep_tail", 0),
                resgrad_outer=getattr(args, "resgrad_outer", False),
                untie_groups=getattr(args, "untie_groups", 1),
                predict_sigma=(float(getattr(args, "mamba_crps", 0.0)) != 0.0),
                dual_wiener_ema=getattr(args, "dual_wiener_ema", 0.95),
                dual_wiener_residual_ema=getattr(args, "dual_wiener_residual_ema", 0.99),
                dual_wiener_warmup_batches=getattr(args, "dual_wiener_warmup_batches", 8),
                dual_wiener_probe_every=getattr(args, "dual_wiener_probe_every", 4),
                dual_wiener_min_probes=getattr(args, "dual_wiener_min_probes", 1),
                dual_wiener_noise_model=getattr(
                    args, "dual_wiener_noise_model", "diagonal_gaussian"
                ),
                dual_wiener_max_horizon=getattr(args, "dual_wiener_max_horizon", 1024),
                global_horizon_wiener=getattr(args, "global_horizon_wiener", False),
                global_wiener_ridge=getattr(args, "global_wiener_ridge", 1e-8),
                global_wiener_anchor=getattr(args, "global_wiener_anchor", 0.0),
                global_wiener_local_fidelity=getattr(
                    args, "global_wiener_local_fidelity", 0.0
                ),
                global_wiener_solver_iters=getattr(
                    args, "global_wiener_solver_iters", 256
                ),
                global_wiener_sketch_dim=getattr(
                    args, "global_wiener_sketch_dim", 8192
                ),
                global_wiener_sketch_seed=getattr(
                    args, "global_wiener_sketch_seed", 1729
                ),
                global_wiener_noise_draws=getattr(
                    args, "global_wiener_noise_draws", 4
                ),
                global_wiener_batch_conditioned=getattr(
                    args, "global_wiener_batch_conditioned", True
                ),
                global_wiener_superbatch_groups=getattr(
                    args, "global_wiener_superbatch_groups", 1
                ),
                global_wiener_static_gain=getattr(
                    args, "global_wiener_static_gain", -1.0
                ),
                global_wiener_static_mode=getattr(
                    args, "global_wiener_static_mode", "delayed_tied"
                ),
            )
        elif args.model_name == "predictive_subspace_attention":
            model = cls(
                state_dim=state_dim,
                input_dim=args.stim_dim,
                state_shape=state_shape,
                hidden_dim=getattr(args, "subspace_hidden_dim", getattr(args, "simple_hidden_dim", 512)),
                latent_dim=getattr(args, "subspace_latent_dim", 128),
                num_refs=getattr(args, "subspace_num_refs", 128),
                key_dim=getattr(args, "subspace_key_dim", 128),
                transition_rank=getattr(args, "subspace_transition_rank", 16),
                dropout=getattr(args, "simple_dropout", 0.0),
                has_external_input=getattr(args, "dataset_has_external_input", False),
                residual=getattr(args, "simple_residual", True),
                diag_init=getattr(args, "subspace_diag_init", 0.98),
                max_diag=getattr(args, "subspace_max_diag", 0.999),
                temperature=getattr(args, "subspace_temperature", 1.0),
                entropy_weight=getattr(args, "subspace_entropy_weight", 0.0),
                current_rec_weight=getattr(args, "subspace_current_rec_weight", 0.0),
                use_layernorm=getattr(args, "subspace_use_layernorm", True),
            )
        elif args.model_name == "official_pc_mamba_state":
            model = cls(
                **common,
                pc_gate_init=getattr(args, "pc_gate_init", 0.35),
                pc_gate_scalar=getattr(args, "pc_gate_scalar", False),
                pc_gate_min=getattr(args, "pc_gate_min", 0.0),
                pc_gate_max=getattr(args, "pc_gate_max", 1.0),
            )

        elif args.model_name in {"official_atlas_mamba_state", "official_tangent_atlas_mamba_state"}:
            model = cls(
                **common,
                atlas_num_charts=getattr(args, "atlas_num_charts", 4),
                atlas_latent_dim=getattr(args, "atlas_latent_dim", 64),
                atlas_chart_emb_dim=getattr(args, "atlas_chart_emb_dim", 64),
                atlas_hidden_dim=getattr(args, "atlas_hidden_dim", 512),
                atlas_temperature=getattr(args, "atlas_temperature", 1.0),
                atlas_perturb_std=getattr(args, "atlas_perturb_std", 0.02),
                atlas_perturb_min_ratio=getattr(args, "atlas_perturb_min_ratio", 0.20),
                atlas_hard_delta_norm=getattr(args, "atlas_hard_delta_norm", False),
                atlas_hard_delta_min_x_ratio=getattr(args, "atlas_hard_delta_min_x_ratio", 0.05),
                atlas_delta_scale_init_ratio=getattr(args, "atlas_delta_scale_init_ratio", 0.01),
            )
        elif args.model_name == "official_mamba_fixeda_perturb":
            model = cls(
                **common,
                fixeda_init=getattr(args, "mamba_fixeda_init", 0.98),
                fixeda_max=getattr(args, "mamba_fixeda_max", 0.999),
                fixeda_perturb_eps=getattr(args, "mamba_fixeda_perturb_eps", 0.02),
                fixeda_perturb_bound=getattr(args, "mamba_fixeda_perturb_bound", "tanh"),
                fixeda_perturb_hidden_mult=getattr(args, "mamba_fixeda_perturb_hidden_mult", 1),
                fixeda_kg_weight=getattr(args, "mamba_fixeda_kg_weight", 0.0),
                fixeda_kg_horizon=getattr(args, "mamba_fixeda_kg_horizon", 0),
                fixeda_corr_weight=getattr(args, "mamba_fixeda_corr_weight", 0.0),
                fixeda_mean_weight=getattr(args, "mamba_fixeda_mean_weight", 0.0),
                fixeda_delta_weight=getattr(args, "mamba_fixeda_delta_weight", 0.0),
                fixeda_corr_solve_lambda=getattr(args, "mamba_fixeda_corr_solve_lambda", 1e-2),
                fixeda_corr_detach=getattr(args, "mamba_fixeda_corr_detach", True),
            )
        else:
            model = cls(
                **common,
                shadow_channels=getattr(args, "shadow_channels", 64),
                perturb_rank=getattr(args, "shadow_perturb_rank", 4),
                perturb_eps=getattr(args, "shadow_perturb_eps", 0.02),
                perturb_bound=getattr(args, "shadow_perturb_bound", "tanh"),
                a_init_scale=getattr(args, "shadow_a_init_scale", 0.98),
                a_init_noise=getattr(args, "shadow_a_init_noise", 1e-3),
                shadow_condition_scale=getattr(args, "shadow_condition_scale", 1.0),
                shadow_output_scale=getattr(args, "shadow_output_scale", 1.0),
                shadow_kg_horizon=getattr(args, "shadow_kg_horizon", 0),
                shadow_kg_weight=getattr(args, "shadow_kg_weight", 0.0),
                shadow_delta_weight=getattr(args, "shadow_delta_weight", 0.0),
                shadow_spec_weight=getattr(args, "shadow_spec_weight", 0.0),
                shadow_spec_max=getattr(args, "shadow_spec_max", 0.999),
            )
        return model.cuda(rank)



    # ------------------------------------------------------------
    # Standard autoregressive U-Net field baseline
    # ------------------------------------------------------------
    if args.model_name == "unet_field":
        if getattr(args, "field_channels", 0) <= 0:
            raise ValueError(
                "unet_field requires --field_channels > 0. It is normally inferred automatically "
                "for dataset=the_well; otherwise pass it explicitly."
            )
        model = cls(
            field_channels=args.field_channels,
            window_size=args.window_size,
            base_channels=getattr(args, "unet_base_channels", 64),
            depth=getattr(args, "unet_depth", 4),
            channel_mult=getattr(args, "unet_channel_mult", 2),
            groups=getattr(args, "unet_groups", 8),
            use_grid=getattr(args, "unet_use_grid", True),
            normalize=getattr(args, "unet_normalize", True),
            folded_enabled=getattr(args, "fno_folded_loss", False),
            folded_pool_size=getattr(args, "fno_fold_pool_size", 8),
            folded_init_scale=getattr(args, "fno_fold_init_scale", 0.98),
            resgrad_routing=getattr(args, "resgrad_routing", False),
            resgrad_policy=getattr(args, "resgrad_policy", "all"),
            resgrad_block_gate=getattr(args, "resgrad_block_gate", 1.0),
            resgrad_ratio_threshold=getattr(args, "resgrad_ratio_threshold", 0.05),
            resgrad_keep_every=getattr(args, "resgrad_keep_every", 8),
            resgrad_keep_tail=getattr(args, "resgrad_keep_tail", 0),
            field_height=getattr(args, "field_height", 0),
            field_width=getattr(args, "field_width", 0),
            dual_wiener_ema=getattr(args, "dual_wiener_ema", 0.95),
            dual_wiener_residual_ema=getattr(args, "dual_wiener_residual_ema", 0.99),
            dual_wiener_warmup_batches=getattr(args, "dual_wiener_warmup_batches", 8),
            dual_wiener_probe_every=getattr(args, "dual_wiener_probe_every", 4),
            dual_wiener_min_probes=getattr(args, "dual_wiener_min_probes", 1),
            dual_wiener_noise_model=getattr(
                args, "dual_wiener_noise_model", "diagonal_gaussian"
            ),
            dual_wiener_max_horizon=getattr(args, "dual_wiener_max_horizon", 1024),
            global_horizon_wiener=getattr(args, "global_horizon_wiener", False),
            global_wiener_ridge=getattr(args, "global_wiener_ridge", 1e-8),
            global_wiener_anchor=getattr(args, "global_wiener_anchor", 0.0),
            global_wiener_local_fidelity=getattr(
                args, "global_wiener_local_fidelity", 0.0
            ),
            global_wiener_solver_iters=getattr(
                args, "global_wiener_solver_iters", 256
            ),
            global_wiener_sketch_dim=getattr(
                args, "global_wiener_sketch_dim", 8192
            ),
            global_wiener_sketch_seed=getattr(
                args, "global_wiener_sketch_seed", 1729
            ),
            global_wiener_noise_draws=getattr(
                args, "global_wiener_noise_draws", 4
            ),
            global_wiener_batch_conditioned=getattr(
                args, "global_wiener_batch_conditioned", True
            ),
            global_wiener_superbatch_groups=getattr(
                args, "global_wiener_superbatch_groups", 1
            ),
            global_wiener_static_gain=getattr(
                args, "global_wiener_static_gain", -1.0
            ),
            global_wiener_static_mode=getattr(
                args, "global_wiener_static_mode", "delayed_tied"
            ),
        )
        return model.cuda(rank)

    # ------------------------------------------------------------
    # Standard autoregressive FNO field baseline
    # ------------------------------------------------------------
    if args.model_name in ("fno_field", "residual_fno_field"):
        if getattr(args, "field_channels", 0) <= 0:
            raise ValueError(
                "fno_field requires --field_channels > 0. It is normally inferred automatically "
                "for dataset=the_well; otherwise pass it explicitly."
            )
        kwargs = dict(
            field_channels=args.field_channels,
            window_size=args.window_size,
            width=getattr(args, "fno_width", 64),
            modes1=getattr(args, "fno_modes1", 16),
            modes2=getattr(args, "fno_modes2", 16),
            n_layers=getattr(args, "fno_layers", 4),
            padding=getattr(args, "fno_padding", 8),
            hidden_channels=getattr(args, "fno_hidden_channels", 128),
            use_grid=getattr(args, "fno_use_grid", True),
            backend=getattr(args, "fno_backend", "neuralop"),
            normalize=getattr(args, "fno_normalize", True),
            folded_enabled=getattr(args, "fno_folded_loss", False),
            folded_pool_size=getattr(args, "fno_fold_pool_size", 8),
            folded_init_scale=getattr(args, "fno_fold_init_scale", 0.98),
            residual_blocks=(args.model_name == "residual_fno_field"),
            resgrad_routing=getattr(args, "resgrad_routing", False),
            resgrad_policy=getattr(args, "resgrad_policy", "all"),
            field_height=getattr(args, "field_height", 0),
            field_width=getattr(args, "field_width", 0),
            dual_wiener_ema=getattr(args, "dual_wiener_ema", 0.95),
            dual_wiener_residual_ema=getattr(args, "dual_wiener_residual_ema", 0.99),
            dual_wiener_warmup_batches=getattr(args, "dual_wiener_warmup_batches", 8),
            dual_wiener_probe_every=getattr(args, "dual_wiener_probe_every", 4),
            dual_wiener_min_probes=getattr(args, "dual_wiener_min_probes", 1),
            dual_wiener_noise_model=getattr(args, "dual_wiener_noise_model", "diagonal_gaussian"),
            dual_wiener_max_horizon=getattr(args, "dual_wiener_max_horizon", 1024),
        )
        model = cls(**kwargs)
        return model.cuda(rank)

    # ------------------------------------------------------------
    # Common kwargs for neural Koopman models
    # ------------------------------------------------------------
    common_koopman_kwargs = dict(
        stim_dim=args.stim_dim,
        window_size=args.window_size,
        hidden_dim=args.hidden_dim,
        stim_depth=args.koopman_stim_depth,
        stim_nhead=args.koopman_stim_nhead,
        dropout=args.koopman_dropout,
        stable_linear=args.koopman_stable_linear,
        spectral_bound=args.koopman_spectral_bound,
        residual_transition=args.koopman_residual_transition,
        dt=args.koopman_dt,
        damping=args.koopman_damping,
        force_scale=args.koopman_force_scale,
        use_koopman_encoder=args.use_koopman_encoder,
        koopman_latent_dim=args.koopman_latent_dim,
        koopman_encoder_mid_dim=args.koopman_encoder_mid_dim,
        koopman_encoder_residual=not args.no_koopman_encoder_residual,
        koopman_latent_channels=args.koopman_latent_channels,
        koopman_latent_channel_dim=args.koopman_latent_channel_dim,
        koopman_channel_shared_A=args.koopman_channel_shared_A,
        koopman_use_latent_adapter=args.koopman_use_latent_adapter,
        koopman_adapter_dim=args.koopman_adapter_dim,
        koopman_adapter_alpha=args.koopman_adapter_alpha,
    )

    # ------------------------------------------------------------
    # HCP/vector neural Koopman
    # ------------------------------------------------------------
    if args.model_name == "koopman_raw_gramian":
        kwargs = dict(
            **common_koopman_kwargs,
            fmri_dim=args.roi_dim,
        )

    # ------------------------------------------------------------
    # Field/The-Well neural Koopman
    # ------------------------------------------------------------
    elif args.model_name in {"koopman_field_gramian", "koopman_field_hard_corrector"}:
        if getattr(args, "field_channels", 0) <= 0:
            raise ValueError(
                "koopman_field_gramian requires --field_channels > 0. "
                "For turbulent_radiative_layer_2D, try --field_channels 4."
            )

        kwargs = dict(
            **common_koopman_kwargs,
            field_channels=args.field_channels,
            field_base_channels=getattr(args, "field_base_channels", 32),
            field_seed_size=getattr(args, "field_seed_size", 8),
        )
        if args.model_name == "koopman_field_hard_corrector":
            kwargs.update(
                corrector_rho=getattr(args, "koopman_corrector_rho", 0.1),
                corrector_hidden_mult=getattr(args, "koopman_corrector_hidden_mult", 2),
                corrector_init_std=getattr(args, "koopman_corrector_init_std", 1e-2),
                corrector_use_stim=getattr(args, "koopman_corrector_use_stim", True),
                corrector_detach_next_context=getattr(args, "koopman_corrector_detach_next_context", False),
                corrector_min_denominator=getattr(args, "koopman_corrector_min_denominator", 1e-4),
            )

    else:
        raise ValueError(f"Unsupported model_name={args.model_name}")

    model = cls(**kwargs)
    return model.cuda(rank)
