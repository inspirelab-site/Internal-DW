import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from internal_dw.models.dual_wiener import (  # noqa: E402
    DualWienerController,
    solve_box_wiener_2x2,
)
from internal_dw.models.official_state_mamba import OfficialStateMambaARModel  # noqa: E402
from internal_dw.models.unet_field import ResidualBlock, UNetFieldModel  # noqa: E402


class DualWienerSolverTest(unittest.TestCase):
    def test_diagonal_solution_matches_scalar_wiener_gains(self):
        total = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
        noise = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        result = solve_box_wiener_2x2(total, noise)
        self.assertTrue(torch.allclose(result, torch.tensor([0.5, 0.75]), atol=2e-5))

    def test_box_solution_is_no_worse_than_componentwise_clamp(self):
        total = torch.tensor([[2.0, -1.4], [-1.4, 2.0]])
        noise = torch.tensor([[0.9, -0.2], [-0.2, 0.4]])
        result = solve_box_wiener_2x2(total, noise)

        # Reconstruct the same PSD-adjusted objective by comparing against a
        # dense grid.  The analytic edge enumeration should attain its minimum.
        from internal_dw.models.dual_wiener import _project_psd_2x2

        t = _project_psd_2x2(total)
        r = _project_psd_2x2(noise)
        p = _project_psd_2x2(t - r)
        c = p + r
        c = c + (c.diagonal().sum() * 1e-6 + 1e-20) * torch.eye(2)
        b = p @ torch.ones(2)

        def objective(w):
            return w @ c @ w - 2.0 * (w @ b)

        grid = torch.linspace(0.0, 1.0, 401)
        brute = min(objective(torch.tensor([a, m])) for a in grid for m in grid)
        self.assertLessEqual(float(objective(result)), float(brute) + 2e-4)


class DualWienerAutogradTest(unittest.TestCase):
    def test_ddp_probe_moments_are_pooled_before_running_update(self):
        controller = DualWienerController(
            state_dim=1,
            depth=1,
            max_horizon=1,
            min_probes=1,
        )
        controller._probe_acc = {
            0: (torch.tensor([1.0, 0.0, 1.0]), 1)
        }

        def fake_all_reduce(tensor, op=None):
            del op
            if tensor.ndim == 2:
                tensor.add_(torch.tensor([[3.0, 0.0, 3.0]]))
            else:
                tensor.add_(torch.tensor([1], dtype=tensor.dtype))

        with (
            mock.patch(
                "internal_dw.models.dual_wiener.dist.is_available", return_value=True
            ),
            mock.patch(
                "internal_dw.models.dual_wiener.dist.is_initialized", return_value=True
            ),
            mock.patch(
                "internal_dw.models.dual_wiener.dist.get_world_size", return_value=2
            ),
            mock.patch(
                "internal_dw.models.dual_wiener.dist.all_reduce",
                side_effect=fake_all_reduce,
            ),
        ):
            controller._update_probe_moments("total")

        self.assertTrue(
            torch.equal(
                controller.total_moments[0, 0],
                torch.tensor([2.0, 0.0, 2.0]),
            )
        )
        self.assertEqual(int(controller.total_updates[0, 0]), 1)

    def test_min_probe_initialization_uses_arithmetic_mean_then_ema(self):
        controller = DualWienerController(
            state_dim=1,
            depth=1,
            max_horizon=1,
            ema=0.95,
            min_probes=3,
        )
        for value, expected in ((1.0, 1.0), (3.0, 2.0), (5.0, 3.0)):
            controller._probe_acc = {
                0: (torch.tensor([value, 0.0, value]), 1)
            }
            controller._update_probe_moments("total")
            self.assertTrue(
                torch.allclose(
                    controller.total_moments[0, 0],
                    torch.tensor([expected, 0.0, expected]),
                )
            )
        controller._probe_acc = {0: (torch.tensor([9.0, 0.0, 9.0]), 1)}
        controller._update_probe_moments("total")
        self.assertTrue(
            torch.allclose(
                controller.total_moments[0, 0],
                torch.tensor([3.3, 0.0, 3.3]),
            )
        )

    def test_min_probe_initialization_keeps_routes_open_until_ready(self):
        torch.manual_seed(29)
        model = OfficialStateMambaARModel(
            state_dim=2,
            input_dim=0,
            hidden_dim=4,
            depth=1,
            dropout=0.0,
            has_external_input=False,
            residual=True,
            mamba_d_state=2,
            mamba_d_conv=2,
            mamba_expand=1,
            resgrad_routing=True,
            resgrad_policy="dualwiener",
            dual_wiener_warmup_batches=1,
            dual_wiener_probe_every=1,
            dual_wiener_min_probes=3,
        )
        model.train()

        def run_batch():
            model.zero_grad(set_to_none=True)
            model.dual_wiener_begin_batch()
            state = model.init_state(3, torch.device("cpu"), torch.float32)
            frame = torch.randn(3, 2)
            prediction, state, _ = model.step(
                state,
                frame,
                None,
                return_aux=True,
                horizon_index=0,
                total_horizon=1,
            )
            target = torch.randn_like(prediction)
            total, noise = model.dual_wiener_probe_terms(prediction, target, 0)
            model.dual_wiener_set_probe_losses(total, noise)
            calibrated = model.dual_wiener_calibrate()
            (prediction - target).square().mean().backward()
            model.dual_wiener_end_batch()
            return calibrated

        self.assertFalse(run_batch())  # dense residual warm-up
        self.assertFalse(run_batch())  # route probe 1/3
        self.assertFalse(run_batch())  # route probe 2/3
        self.assertTrue(
            torch.equal(model.dual_wiener.coefficients[0, 0], torch.ones(2))
        )
        self.assertTrue(run_batch())   # route probe 3/3; first solve
        self.assertEqual(int(model.dual_wiener.total_updates[0, 0]), 3)
        self.assertEqual(int(model.dual_wiener.noise_updates[0, 0]), 3)
        self.assertEqual(int(model.dual_wiener.solved_batches), 1)

    def test_route_scales_input_credit_but_not_local_parameter_gradient(self):
        controller = DualWienerController(
            state_dim=1, depth=1, max_horizon=2, warmup_batches=1
        )
        controller.coefficients[0, 0] = torch.tensor([0.3, 0.7])
        token = torch.tensor([[2.0]], requires_grad=True)
        theta = torch.tensor(4.0, requires_grad=True)

        identity, branch_input = controller.route_pair(token, 0, 0)
        output = identity + theta * branch_input
        output.sum().backward()

        # dL/dtoken = alpha + m*theta, while dL/dtheta remains the ordinary
        # local derivative branch_input.  The gate changes credit, not learning
        # of parameters inside the current nonlinear branch.
        self.assertTrue(torch.allclose(token.grad, torch.tensor([[3.1]])))
        self.assertTrue(torch.allclose(theta.grad, torch.tensor(2.0)))

    def test_calibration_mode_is_fully_open(self):
        controller = DualWienerController(
            state_dim=1, depth=1, max_horizon=2, warmup_batches=1
        )
        controller.coefficients[0, 0] = torch.tensor([0.1, 0.2])
        reference = torch.ones(1)
        self.assertAlmostEqual(float(controller.route_coefficient(0, 0, reference)), 0.1)
        controller._mode = "total"
        self.assertAlmostEqual(float(controller.route_coefficient(0, 0, reference)), 1.0)
        self.assertAlmostEqual(float(controller.route_coefficient(0, 1, reference)), 1.0)

    def test_structured_noise_probe_preserves_a_lagged_residual_field(self):
        controller = DualWienerController(
            state_dim=4,
            depth=1,
            max_horizon=2,
            warmup_batches=1,
            probe_every=1,
            noise_model="lagged_residual_bootstrap",
        )
        controller.seen_batches.fill_(1)
        controller.residual_updates.fill_(1)
        controller.residual_mean.zero_()
        controller.residual_template[0] = torch.tensor([1.0, -2.0, 3.0, -4.0])
        controller.residual_template[1] = torch.tensor([-4.0, 3.0, -2.0, 1.0])
        controller.residual_template_valid.fill_(True)
        controller.begin_batch()

        reference = torch.zeros(2, 1, 2, 2)
        covector = controller.noise_covector(0, reference)
        self.assertIsNotNone(covector)
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0])
        self.assertTrue(torch.equal(covector[0].reshape(-1).abs(), expected))
        self.assertTrue(torch.equal(covector[1].reshape(-1).abs(), expected))
        later = controller.noise_covector(1, reference)
        # All horizons share the same random sign for each trajectory, so the
        # lagged residual draw retains temporal cross-covariance.
        self.assertTrue(
            torch.equal(
                torch.sign(covector[:, 0, 0, 0]),
                -torch.sign(later[:, 0, 0, 0]),
            )
        )
        # The covariance sketch is runtime state, not a checkpoint burden.
        self.assertNotIn("residual_template", controller.state_dict())

    def test_spatial_spectrum_probe_generates_unit_variance_correlated_fields(self):
        torch.manual_seed(19)
        controller = DualWienerController(
            state_dim=8 * 8,
            depth=1,
            max_horizon=1,
            warmup_batches=1,
            probe_every=1,
            noise_model="spatial_spectrum",
            spatial_shape=(1, 8, 8),
        )
        controller.seen_batches.fill_(1)
        controller.residual_updates.fill_(1)
        controller.residual_mean.zero_()
        controller.residual_second.fill_(1.0)
        # Unit-variance spectrum concentrated at DC: every draw is spatially
        # constant, but its value varies as N(0,1) across batch items.
        controller.spatial_spectrum_power.zero_()
        controller.spatial_spectrum_power[0, 0, 0, 0] = 64.0
        controller.spatial_spectrum_updates.fill_(1)
        controller.begin_batch()

        reference = torch.zeros(512, 1, 1, 8, 8)
        covector = controller.noise_covector(0, reference)
        self.assertIsNotNone(covector)
        self.assertTrue(
            torch.equal(covector[..., :-1], covector[..., 1:])
        )
        self.assertAlmostEqual(float(covector.square().mean()), 1.0, delta=0.15)
        self.assertIn("spatial_spectrum_power", controller.state_dict())
        self.assertIn("spatial_spectrum_updates", controller.state_dict())

    def test_spatial_spectrum_oas_uses_rfft_weighted_shrinkage(self):
        with mock.patch.dict(
            os.environ, {"DUAL_WIENER_SPECTRUM_OAS": "1"}, clear=False
        ):
            controller = DualWienerController(
                state_dim=4 * 4,
                depth=1,
                max_horizon=1,
                noise_model="spatial_spectrum",
                spatial_shape=(1, 4, 4),
            )
        # Flush one deliberately colored empirical spectrum.  The residual
        # moments are included because _flush_residuals commits both stores in
        # one transaction at the end of a training batch.
        controller._pending_residual[0] = (
            torch.zeros(16),
            torch.ones(16) * 8.0,
            8,
        )
        colored = torch.ones(1, 4, 3)
        colored[..., 0, 0] = 20.0
        controller._pending_spatial_power[0] = (colored * 8.0, 8)
        controller._flush_residuals()

        shrinkage = float(controller.spatial_spectrum_oas_shrinkage[0])
        self.assertGreater(shrinkage, 0.0)
        self.assertLessEqual(shrinkage, 1.0)
        self.assertEqual(int(controller.spatial_spectrum_samples[0]), 8)
        power = controller.spatial_spectrum_power[0]
        weights = torch.tensor([1.0, 2.0, 1.0])
        variance = (power * weights).sum() / 16.0
        self.assertAlmostEqual(float(variance), 1.0, places=5)
        self.assertTrue(controller.export_state(1)["spatial_spectrum_oas"])

    def test_spatial_spectrum_requires_an_explicit_field_shape(self):
        with self.assertRaises(ValueError):
            DualWienerController(
                state_dim=64, depth=1, noise_model="spatial_spectrum"
            )
        with self.assertRaises(ValueError):
            DualWienerController(
                state_dim=63,
                depth=1,
                noise_model="spatial_spectrum",
                spatial_shape=(1, 8, 8),
            )

    def test_field_controller_does_not_allocate_dense_external_process(self):
        controller = DualWienerController(
            state_dim=32 * 32,
            depth=1,
            max_horizon=1,
            noise_model="spatial_spectrum",
            spatial_shape=(1, 32, 32),
        )
        self.assertEqual(controller.external_ar_transition_matrix.numel(), 0)
        self.assertEqual(controller.external_ar_innovation_factor.numel(), 0)
        self.assertEqual(controller.external_ar_observation_matrix.numel(), 0)

    def test_noise_model_rejects_unknown_choice(self):
        with self.assertRaises(ValueError):
            DualWienerController(state_dim=1, depth=1, noise_model="mystery")

    def test_oracle_ar_probe_preserves_cross_horizon_innovation_covariance(self):
        torch.manual_seed(23)
        coefficients = np.asarray([0.8, -0.6], dtype=np.float32)
        one_step_std = np.sqrt(1.0 - coefficients**2).astype(np.float32)
        horizons = np.arange(1, 4, dtype=np.int64)[:, None]
        variance = (1.0 - np.power(coefficients[None, :] ** 2, horizons)).astype(
            np.float32
        )
        with tempfile.TemporaryDirectory() as directory:
            calibration = Path(directory) / "oracle.npz"
            np.savez_compressed(
                calibration,
                innovation_variance=variance,
                oracle_ar_coefficients=coefficients,
                oracle_one_step_innovation_std=one_step_std,
            )
            with mock.patch.dict(
                os.environ,
                {
                    "DUAL_WIENER_INNOVATION_FILE": str(calibration),
                    "DUAL_WIENER_INNOVATION_KEY": "innovation_variance",
                },
                clear=False,
            ):
                controller = DualWienerController(
                    state_dim=2,
                    depth=1,
                    max_horizon=3,
                    warmup_batches=1,
                    probe_every=1,
                )
            controller.seen_batches.fill_(1)
            controller.begin_batch()
            reference = torch.zeros(30000, 2)
            first = controller.noise_covector(0, reference)
            second = controller.noise_covector(1, reference)

        expected_cross = torch.as_tensor(coefficients * (1.0 - coefficients**2))
        empirical_cross = (first * second).mean(dim=0)
        self.assertTrue(torch.allclose(empirical_cross, expected_cross, atol=0.02))
        expected_second_variance = torch.as_tensor(1.0 - coefficients**4)
        self.assertTrue(
            torch.allclose(
                second.var(dim=0, unbiased=False),
                expected_second_variance,
                atol=0.02,
            )
        )
        exported = controller.export_state(3)
        self.assertEqual(
            exported["estimator"], "oracle_linear_gaussian_process_wiener"
        )
        self.assertTrue(exported["external_innovation_joint_process"])

    def test_lagged_external_process_maps_latent_noise_to_output(self):
        torch.manual_seed(31)
        # Scalar AR(2): x[t+1] = .5 x[t] + .2 x[t-1] + eps.
        transition = np.asarray([[0.5, 0.2], [1.0, 0.0]], dtype=np.float32)
        observation = np.asarray([[1.0, 0.0]], dtype=np.float32)
        innovation = np.asarray([[0.3]], dtype=np.float32)
        impulse = np.asarray([[np.sqrt(0.3)], [0.0]], dtype=np.float64)
        accumulated = 0.0
        variance = []
        for _ in range(3):
            projected = observation.astype(np.float64) @ impulse
            accumulated += float((projected @ projected.T).item())
            variance.append([accumulated])
            impulse = transition.astype(np.float64) @ impulse
        with tempfile.TemporaryDirectory() as directory:
            calibration = Path(directory) / "lagged.npz"
            np.savez_compressed(
                calibration,
                innovation_variance=np.asarray(variance, dtype=np.float32),
                ar_transition_matrix=transition,
                ar_observation_matrix=observation,
                one_step_innovation_covariance=innovation,
            )
            with mock.patch.dict(
                os.environ,
                {"DUAL_WIENER_INNOVATION_FILE": str(calibration)},
                clear=False,
            ):
                controller = DualWienerController(
                    state_dim=1,
                    depth=1,
                    max_horizon=3,
                    warmup_batches=1,
                    probe_every=1,
                )
            controller.seen_batches.fill_(1)
            controller.begin_batch()
            reference = torch.zeros(40000, 1)
            draws = [controller.noise_covector(h, reference) for h in range(3)]

        self.assertEqual(tuple(controller.external_ar_transition_matrix.shape), (2, 2))
        for horizon, draw in enumerate(draws):
            self.assertAlmostEqual(
                float(draw.var(unbiased=False)), variance[horizon][0], delta=0.015
            )

    def test_no_grad_diagnostic_allows_rollout_beyond_gain_table(self):
        controller = DualWienerController(
            state_dim=1, depth=1, max_horizon=2, warmup_batches=1
        )
        with torch.no_grad():
            pair = controller.current_pair(1024, 0, torch.ones(1))
        self.assertTrue(torch.equal(pair, torch.ones(2)))

    def test_probe_does_not_accumulate_parameter_gradients(self):
        torch.manual_seed(3)
        model = OfficialStateMambaARModel(
            state_dim=4,
            input_dim=2,
            hidden_dim=8,
            depth=2,
            dropout=0.0,
            has_external_input=True,
            residual=True,
            mamba_d_state=3,
            mamba_d_conv=2,
            mamba_expand=1,
            resgrad_routing=True,
            resgrad_policy="dualwiener",
            dual_wiener_warmup_batches=1,
            dual_wiener_probe_every=1,
        )
        model.train()

        def run_batch():
            model.dual_wiener_begin_batch()
            h = model.init_state(3, torch.device("cpu"), torch.float32)
            x = torch.randn(3, 4)
            total_terms = []
            noise_terms = []
            losses = []
            for k in range(3):
                prediction, h, _ = model.step(
                    h,
                    x,
                    torch.randn(3, 2),
                    return_aux=True,
                    horizon_index=k,
                    total_horizon=3,
                )
                target = torch.randn_like(prediction)
                total, noise = model.dual_wiener_probe_terms(prediction, target, k)
                if total is not None:
                    total_terms.append(total)
                    noise_terms.append(noise)
                losses.append((prediction - target).square().mean())
                x = prediction
            model.dual_wiener_set_probe_losses(
                torch.stack(total_terms).mean() if total_terms else None,
                torch.stack(noise_terms).mean() if noise_terms else None,
            )
            loss = torch.stack(losses).mean()
            calibrated = model.dual_wiener_calibrate()
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
            loss.backward()
            self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
            model.dual_wiener_end_batch()
            model.zero_grad(set_to_none=True)
            return calibrated

        self.assertFalse(run_batch())  # dense warm-up; also initializes R_y
        self.assertTrue(run_batch())
        controller: DualWienerController = model.dual_wiener
        self.assertGreater(int(controller.solved_batches), 0)
        self.assertTrue(bool(((controller.coefficients[:3] >= 0.0) &
                              (controller.coefficients[:3] <= 1.0)).all()))


class DualWienerUNetTest(unittest.TestCase):
    def test_two_route_block_gradient_matches_skip_branch_decomposition(self):
        torch.manual_seed(7)
        block = ResidualBlock(2, 3, groups=1)
        controller = DualWienerController(
            state_dim=2 * 4 * 4, depth=1, max_horizon=1, warmup_batches=1
        )
        controller.coefficients[0, 0] = torch.tensor([0.25, 0.6])

        x = torch.randn(2, 2, 4, 4, requires_grad=True)
        block(
            x,
            resgrad_policy="dualwiener",
            dual_wiener=controller,
            route_horizon=0,
            route_layer=0,
        ).sum().backward()
        routed_gradient = x.grad.detach().clone()

        reference = x.detach().clone().requires_grad_(True)
        skip = block.skip(reference)
        branch = block.conv2(block.conv1(reference))
        merged = skip + branch
        # Use the functional form so the block's in-place SiLU does not
        # overwrite the tensor with respect to which the merge VJP is taken.
        output = torch.nn.functional.silu(merged)
        merge_vjp = torch.autograd.grad(output.sum(), merged, retain_graph=True)[0]
        skip_vjp = torch.autograd.grad(skip, reference, merge_vjp, retain_graph=True)[0]
        branch_vjp = torch.autograd.grad(branch, reference, merge_vjp)[0]
        expected = 0.25 * skip_vjp + 0.6 * branch_vjp
        self.assertTrue(torch.allclose(routed_gradient, expected, atol=2e-6, rtol=2e-5))

    def test_unet_forward_value_is_unchanged(self):
        torch.manual_seed(11)
        dense = UNetFieldModel(
            field_channels=2,
            window_size=2,
            base_channels=4,
            depth=2,
            groups=2,
            use_grid=True,
            normalize=False,
        )
        routed = UNetFieldModel(
            field_channels=2,
            window_size=2,
            base_channels=4,
            depth=2,
            groups=2,
            use_grid=True,
            normalize=False,
            resgrad_routing=True,
            resgrad_policy="dualwiener",
            field_height=8,
            field_width=8,
            dual_wiener_max_horizon=3,
        )
        routed.load_state_dict(dense.state_dict(), strict=False)
        routed.dual_wiener.coefficients.uniform_(0.05, 0.95)
        history = torch.randn(2, 2, 2, 8, 8)
        dense.set_resgrad_context(1, 3)
        routed.set_resgrad_context(1, 3)
        self.assertTrue(torch.equal(dense(None, history), routed(None, history)))

    def test_windowed_unet_probe_calibrates_from_observed_history_root(self):
        torch.manual_seed(13)
        model = UNetFieldModel(
            field_channels=2,
            window_size=2,
            base_channels=4,
            depth=2,
            groups=2,
            use_grid=False,
            normalize=False,
            resgrad_routing=True,
            resgrad_policy="dualwiener",
            field_height=8,
            field_width=8,
            dual_wiener_warmup_batches=1,
            dual_wiener_probe_every=1,
            dual_wiener_max_horizon=2,
        )
        model.train()

        def run_batch():
            model.zero_grad(set_to_none=True)
            model.dual_wiener_begin_batch()
            history = torch.randn(2, 2, 2, 8, 8)
            losses, total_terms, noise_terms = [], [], []
            for k in range(2):
                prediction = model(
                    None, history, horizon_index=k, total_horizon=2
                )[:, 0]
                target = torch.randn_like(prediction)
                total, noise = model.dual_wiener_probe_terms(prediction, target, k)
                if total is not None:
                    total_terms.append(total)
                    noise_terms.append(noise)
                losses.append((prediction - target).square().mean())
                history = torch.cat([history[:, 1:], prediction.unsqueeze(1)], dim=1)
            model.dual_wiener_set_probe_losses(
                torch.stack(total_terms).mean() if total_terms else None,
                torch.stack(noise_terms).mean() if noise_terms else None,
            )
            calibrated = model.dual_wiener_calibrate()
            torch.stack(losses).mean().backward()
            model.dual_wiener_end_batch()
            return calibrated

        self.assertFalse(run_batch())
        self.assertTrue(run_batch())
        self.assertEqual(model.dual_wiener.depth, model.core.num_residual_blocks)
        self.assertGreater(int(model.dual_wiener.solved_batches), 0)


if __name__ == "__main__":
    unittest.main()
