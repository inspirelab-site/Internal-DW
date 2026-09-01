import unittest

import torch

from internal_dw.training.ar_losses import _forward_jacobian_fd_penalty


class ForwardJacobianFiniteDifferenceTest(unittest.TestCase):
    def test_identity_at_unit_target_has_negligible_penalty(self):
        torch.manual_seed(0)
        x = torch.randn(5, 7)
        penalty, gain = _forward_jacobian_fd_penalty(
            x, lambda z: z, eps=1e-3, target=1.0
        )
        self.assertLess(float(penalty), 1e-5)
        self.assertAlmostEqual(float(gain), 1.0, places=3)

    def test_expanding_map_is_penalized_and_differentiable(self):
        torch.manual_seed(1)
        layer = torch.nn.Linear(6, 6, bias=False)
        with torch.no_grad():
            layer.weight.copy_(2.0 * torch.eye(6))
        x = torch.randn(4, 6)
        penalty, gain = _forward_jacobian_fd_penalty(
            x, layer, eps=1e-3, target=1.0
        )
        penalty.backward()
        self.assertAlmostEqual(float(gain), 2.0, places=3)
        self.assertAlmostEqual(float(penalty), 1.0, places=2)
        self.assertIsNotNone(layer.weight.grad)
        self.assertTrue(torch.isfinite(layer.weight.grad).all())

    def test_field_shape_uses_per_sample_scaling(self):
        torch.manual_seed(2)
        x = torch.randn(2, 3, 2, 5, 5)
        penalty, gain = _forward_jacobian_fd_penalty(
            x, lambda z: z[:, -1], eps=1e-3, target=2.0
        )
        self.assertEqual(penalty.ndim, 0)
        self.assertEqual(gain.ndim, 0)
        self.assertTrue(torch.isfinite(penalty))


if __name__ == "__main__":
    unittest.main()
