import unittest

import torch

from internal_dw.training.ar_losses import (
    _backward_scale_value,
    _sample_artbp_edge_scale,
)


class ArtbpEdgeTest(unittest.TestCase):
    def test_forward_value_is_unchanged(self):
        value = torch.tensor([1.5, -2.0], requires_grad=True)
        routed = _backward_scale_value(value, 8.0 / 7.0)
        torch.testing.assert_close(routed, value)

    def test_cut_detaches_the_preceding_graph(self):
        value = torch.tensor(2.0, requires_grad=True) * 3.0
        routed = _backward_scale_value(value, 0.0)
        self.assertFalse(routed.requires_grad)
        self.assertEqual(float(routed), float(value))

    def test_geometric_artbp_edge_is_unbiased(self):
        length = 8
        cut_probability = 1.0 / length
        cut = _sample_artbp_edge_scale(
            length, torch.device("cpu"), draw=0.0
        )
        survive = _sample_artbp_edge_scale(
            length, torch.device("cpu"), draw=cut_probability
        )
        self.assertEqual(cut, 0.0)
        self.assertAlmostEqual(survive, length / (length - 1))
        self.assertAlmostEqual(
            cut_probability * cut + (1.0 - cut_probability) * survive,
            1.0,
        )

    def test_same_scale_reaches_every_recurrent_carry(self):
        first = torch.tensor(1.0, requires_grad=True)
        second = torch.tensor(2.0, requires_grad=True)
        state = ((first, second),)
        routed = _backward_scale_value(state, 8.0 / 7.0)
        loss = routed[0][0] + routed[0][1]
        loss.backward()
        self.assertAlmostEqual(float(first.grad), 8.0 / 7.0, places=6)
        self.assertAlmostEqual(float(second.grad), 8.0 / 7.0, places=6)

    def test_multistep_gradient_matches_full_bptt_in_expectation(self):
        length = 4
        cut_probability = 1.0 / length
        survive = length / (length - 1)

        exact_theta = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
        exact_x = torch.tensor(1.0, dtype=torch.float64)
        exact_loss = exact_theta.new_tensor(0.0)
        for _ in range(3):
            exact_x = exact_theta * exact_x
            exact_loss = exact_loss + exact_x.square()
        exact_loss.backward()

        expected_gradient = 0.0
        for first_scale, first_prob in (
            (0.0, cut_probability),
            (survive, 1.0 - cut_probability),
        ):
            for second_scale, second_prob in (
                (0.0, cut_probability),
                (survive, 1.0 - cut_probability),
            ):
                theta = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
                x = torch.tensor(1.0, dtype=torch.float64)
                loss = theta.new_tensor(0.0)
                for step, scale in enumerate((first_scale, second_scale, 1.0)):
                    x = theta * x
                    loss = loss + x.square()
                    if step < 2:
                        x = _backward_scale_value(x, scale)
                loss.backward()
                expected_gradient += (
                    first_prob * second_prob * float(theta.grad)
                )

        self.assertAlmostEqual(
            expected_gradient, float(exact_theta.grad), places=10
        )


if __name__ == "__main__":
    unittest.main()
