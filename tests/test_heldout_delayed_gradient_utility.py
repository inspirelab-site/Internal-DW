from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "probes"))

from gradient_probe_ops import GradientProjection
from probe_heldout_delayed_gradient_utility import (
    _apply_step,
    _projected_dot,
    _projected_norm,
    _recurrent_full_local_projected,
    _undo_step,
)


class _ToyRecurrent(torch.nn.Module):
    is_recurrent_state_ar = True

    def __init__(self):
        super().__init__()
        self.w_x = torch.nn.Parameter(torch.tensor(0.7))
        self.w_h = torch.nn.Parameter(torch.tensor(0.3))

    def init_state(self, batch, device, dtype):
        return torch.zeros(batch, 1, device=device, dtype=dtype)

    def detach_state(self, state):
        return state.detach()

    def step(
        self, hidden, x_in, stim_in, return_aux=False,
        horizon_index=None, total_horizon=None,
    ):
        prediction = self.w_x * x_in + self.w_h * hidden
        return prediction, prediction


class _Args:
    mamba_burnin = 0


class HeldoutUtilityTests(unittest.TestCase):
    def test_projected_scaling_unprojected(self):
        p = torch.nn.Parameter(torch.zeros(2))
        projection = GradientProjection([p], 0, 0)
        a = torch.tensor([3.0, 4.0])
        b = torch.tensor([1.0, 2.0])
        self.assertAlmostEqual(_projected_norm(a, projection), 5.0)
        self.assertAlmostEqual(_projected_dot(a, b, projection), 11.0)

    def test_reversible_equal_norm_step(self):
        p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        before = p.detach().clone()
        direction = [torch.tensor([0.6, 0.8])]
        _apply_step([p], direction, 0.25)
        self.assertAlmostEqual(float(torch.norm(before - p)), 0.25, places=6)
        _undo_step([p], direction, 0.25)
        torch.testing.assert_close(p, before)

    def test_temporal_decomposition_has_zero_h1_and_nonzero_h2(self):
        model = _ToyRecurrent()
        params = list(model.parameters())
        projection = GradientProjection(params, 0, 0)
        state = torch.tensor([[[1.0], [0.2], [-0.1], [0.4]]])
        stim = torch.zeros(1, 4, 1)
        full, local, full_loss, local_loss, fwd = _recurrent_full_local_projected(
            model, params, projection, state, stim,
            start=1, K=2, args=_Args(), loss_type="mse",
        )
        self.assertEqual(fwd, 0.0)
        self.assertAlmostEqual(full_loss[0], local_loss[0], places=7)
        self.assertAlmostEqual(full_loss[1], local_loss[1], places=7)
        torch.testing.assert_close(full[0], local[0])
        self.assertGreater(float(torch.norm(full[1] - local[1])), 1e-6)


if __name__ == "__main__":
    unittest.main()
