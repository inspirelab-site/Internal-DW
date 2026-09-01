import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FMRI = _load("fmri_drive_history", "scripts/data/fmri_drive_history_ops.py")
GENERIC = _load("generic_drive_history", "scripts/probes/probe_cross_dataset_drive_history.py")


class FmriDecompositionTests(unittest.TestCase):
    def test_future_targets_preserve_subject_start_horizon_axes(self):
        residual = np.arange(2 * 20 * 3, dtype=np.float32).reshape(2, 20, 3)
        starts = FMRI.valid_starts(20, window=4, maximum_horizon=8, stride=3)
        targets = FMRI.future_targets(residual, starts, [1, 2, 4, 8])
        self.assertEqual(targets.shape, (2, len(starts), 4, 3))
        np.testing.assert_array_equal(targets[0, 0, 0], residual[0, starts[0]])
        np.testing.assert_array_equal(targets[1, -1, -1], residual[1, starts[-1] + 7])

    def test_mse_is_reported_by_subject_and_horizon(self):
        target = np.zeros((2, 3, 4, 5), dtype=np.float32)
        prediction = np.ones_like(target)
        np.testing.assert_allclose(FMRI.mse_by_subject(prediction, target), 1.0)


class GenericMapTests(unittest.TestCase):
    def test_future_drive_uses_the_complete_known_path(self):
        drive = np.arange(2 * 10 * 3, dtype=np.float32).reshape(2, 10, 3)
        pairs = np.asarray([[0, 2], [1, 4]], dtype=np.int64)
        feature = GENERIC._future_drive(drive, pairs, horizon=4)
        self.assertEqual(feature.shape, (2, 12))
        np.testing.assert_array_equal(feature[0], drive[0, 2:6].reshape(-1))
        np.testing.assert_array_equal(feature[1], drive[1, 4:8].reshape(-1))

    def test_no_external_drive_is_exactly_an_intercept_model(self):
        pairs = np.asarray([[0, 2], [1, 4]], dtype=np.int64)
        feature = GENERIC._future_drive(None, pairs, horizon=4)
        self.assertEqual(feature.shape, (2, 0))
        target = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        prediction = GENERIC._ridge(
            feature,
            target,
            np.empty((3, 0), dtype=np.float32),
            ridge=1e-2,
            device="cpu",
        )
        np.testing.assert_allclose(prediction, [[2.0, 3.0]] * 3)

    def test_field_coordinate_sample_is_channel_stratified(self):
        indices = GENERIC._select_coordinates((4, 8, 8), maximum=32, seed=2)
        channels = indices // 64
        self.assertEqual(len(indices), 32)
        np.testing.assert_array_equal(np.bincount(channels, minlength=4), [8] * 4)


if __name__ == "__main__":
    unittest.main()
