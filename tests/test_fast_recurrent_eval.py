import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from internal_dw.evaluation.eval import evaluate_recurrent_state_horizon_sweep  # noqa: E402
from internal_dw.models.official_state_mamba import OfficialStateMambaARModel  # noqa: E402
from internal_dw.training.ar_losses import compute_recurrent_state_bptt_loss  # noqa: E402


class PackedRecurrentEvaluationTest(unittest.TestCase):
    def test_packed_validation_origins_match_serial_reference(self):
        torch.manual_seed(7)
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
            resgrad_routing=False,
            resgrad_policy="all",
        )
        model.eval()
        state = torch.randn(2, 37, 4)
        stim = torch.randn(2, 37, 2)
        base_args = dict(
            mamba_bptt_horizon=7,
            mamba_burnin=8,
            mamba_loss_type="rel_l2",
            mamba_loss_decay=1.0,
            ar_eval_stride=3,
            stim_dim=2,
        )

        with torch.no_grad():
            serial_loss, serial_logs = compute_recurrent_state_bptt_loss(
                model,
                state,
                stim,
                SimpleNamespace(**base_args, recurrent_val_start_batch=1),
            )
            packed_loss, packed_logs = compute_recurrent_state_bptt_loss(
                model,
                state,
                stim,
                SimpleNamespace(**base_args, recurrent_val_start_batch=4),
            )

        self.assertAlmostEqual(
            float(serial_loss), float(packed_loss), places=6
        )
        self.assertAlmostEqual(
            float(serial_logs["loss"]), float(packed_logs["loss"]), places=6
        )
        self.assertEqual(
            serial_logs["ar/recurrent_num_starts"],
            packed_logs["ar/recurrent_num_starts"],
        )

    def test_packed_refresh_horizons_match_serial_reference(self):
        torch.manual_seed(11)
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
        )
        model.eval()
        batch = {
            "state": torch.randn(2, 21, 4),
            "external_input": torch.randn(2, 21, 2),
        }
        base_args = dict(
            window_size=4,
            test_horizons=[1, 2, 4, 7],
            stim_dim=2,
        )

        # The production evaluator moves batches to CUDA.  Replacing only that
        # transport call lets this equivalence test exercise both algorithms on
        # CPU in CI without changing either implementation.
        with mock.patch.object(torch.Tensor, "cuda", lambda tensor, *a, **k: tensor):
            serial = evaluate_recurrent_state_horizon_sweep(
                model,
                [batch],
                SimpleNamespace(**base_args, recurrent_eval_horizon_batch=1),
            )
            packed = evaluate_recurrent_state_horizon_sweep(
                model,
                [batch],
                SimpleNamespace(**base_args, recurrent_eval_horizon_batch=4),
            )

        self.assertEqual(serial.keys(), packed.keys())
        for key in serial:
            self.assertAlmostEqual(float(serial[key]), float(packed[key]), places=6, msg=key)


if __name__ == "__main__":
    unittest.main()
