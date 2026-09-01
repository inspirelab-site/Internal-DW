import json

import numpy as np

from scripts.evaluate.evaluate_dense_multistart_rel_l2 import (
    build_public_result,
    summarize_unit_origins,
)


def test_public_result_puts_primary_information_first_and_omits_diagnostics():
    unit_origins = {
        "unit:0": {
            0: np.asarray([0.2, 0.4, 0.6]),
            1: np.asarray([0.4, 0.6, 0.8]),
        },
        "unit:1": {
            0: np.asarray([0.3, 0.5, 0.7]),
            1: np.asarray([0.5, 0.7, 0.9]),
        },
    }
    aggregate = summarize_unit_origins(
        unit_origins, eval_horizon=3, train_horizon=2, bootstrap_draws=32
    )
    result = build_public_result(
        aggregate,
        checkpoint_path="outputs/example/best.pth",
        checkpoint_epoch=7,
        dataset="mackey_glass",
        model_name="official_mamba_state",
        method="internal_dw",
        seed=0,
        split="test",
        train_horizon=2,
        eval_horizon=3,
    )

    assert list(result)[:7] == [
        "format_version",
        "status",
        "dataset",
        "method",
        "seed",
        "split",
        "primary_metric",
    ]
    assert result["primary_metric"]["value"] == result["summary"]["all_horizons"]["mean"]
    assert result["checkpoint"] == {"path": "outputs/example/best.pth", "epoch": 7}
    assert "per_unit" not in result
    assert "duplicate_origins_discarded" not in result
    json.dumps(result, allow_nan=False)
