import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from internal_dw.datasets.prepared_temporal import build_prepared_temporal_splits


class PreparedTemporalDatasetTest(unittest.TestCase):
    def _archive(self, root: Path, driven: bool) -> Path:
        rng = np.random.default_rng(7)
        payload = {
            "train_state": rng.normal(size=(4, 16, 3)).astype(np.float32),
            "validation_state": rng.normal(size=(2, 16, 3)).astype(np.float32),
            "test_state": rng.normal(size=(2, 16, 3)).astype(np.float32),
            "metadata_json": np.asarray(json.dumps({"dataset": "toy"})),
        }
        if driven:
            for split, count in (("train", 4), ("validation", 2), ("test", 2)):
                payload[f"{split}_drive"] = rng.normal(size=(count, 16, 2)).astype(np.float32)
        path = root / ("driven.npz" if driven else "autonomous.npz")
        np.savez(path, **payload)
        return path

    def test_autonomous_contract_and_train_standardization(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                prepared_temporal_npz=str(self._archive(Path(directory), False)),
                prepared_temporal_standardize=1,
                roi_dim=99,
                stim_dim=99,
            )
            train, validation, test = build_prepared_temporal_splits(args, False)
            self.assertEqual(args.roi_dim, 3)
            self.assertFalse(train.has_external_input)
            self.assertIsNone(train[0]["external_input"])
            self.assertEqual(len(validation), 2)
            self.assertEqual(len(test), 2)
            stacked = np.stack([train[i]["state"].numpy() for i in range(len(train))])
            np.testing.assert_allclose(stacked.mean(axis=(0, 1)), 0.0, atol=2e-6)
            np.testing.assert_allclose(stacked.std(axis=(0, 1)), 1.0, atol=2e-6)

    def test_driven_contract_sets_stimulus_dimension(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                prepared_temporal_npz=str(self._archive(Path(directory), True)),
                prepared_temporal_standardize=1,
                roi_dim=99,
                stim_dim=99,
            )
            train, _, _ = build_prepared_temporal_splits(args, True)
            self.assertEqual(args.roi_dim, 3)
            self.assertEqual(args.stim_dim, 2)
            self.assertTrue(train.has_external_input)
            self.assertEqual(tuple(train[0]["external_input"].shape), (16, 2))

    def test_external_contract_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                prepared_temporal_npz=str(self._archive(Path(directory), False)),
                prepared_temporal_standardize=1,
            )
            with self.assertRaises(ValueError):
                build_prepared_temporal_splits(args, True)


if __name__ == "__main__":
    unittest.main()
