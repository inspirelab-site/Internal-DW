from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "probes"))

from probe_real_internal_dw_utility import _make_arms


class RealInternalDWUtilityHelpersTest(unittest.TestCase):
    def test_three_arms_preserve_pair_multiset_for_permutation(self):
        weights = torch.arange(6 * 3 * 2, dtype=torch.float32).reshape(6, 3, 2) / 40.0
        arms = _make_arms(weights, K=6, seed=7)
        self.assertEqual(set(arms), {"open", "learned", "permuted"})
        self.assertTrue(torch.equal(arms["open"][:6], torch.ones_like(weights)))
        self.assertTrue(torch.equal(arms["learned"], weights))
        self.assertFalse(torch.equal(arms["permuted"][:6], weights))
        for layer in range(weights.shape[1]):
            original = sorted(map(tuple, weights[:, layer].tolist()))
            permuted = sorted(map(tuple, arms["permuted"][:6, layer].tolist()))
            self.assertEqual(original, permuted)

    def test_unused_horizons_are_unchanged(self):
        weights = torch.linspace(0.1, 0.9, 8 * 2 * 2).reshape(8, 2, 2)
        arms = _make_arms(weights, K=4, seed=5)
        for value in arms.values():
            torch.testing.assert_close(value[4:], weights[4:])

    def test_invalid_gain_is_rejected(self):
        weights = torch.ones(4, 2, 2)
        weights[0, 0, 0] = 1.1
        with self.assertRaises(RuntimeError):
            _make_arms(weights, K=4, seed=0)


if __name__ == "__main__":
    unittest.main()
