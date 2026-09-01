import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "plotting"))
SPEC = importlib.util.spec_from_file_location(
    "driven_mg_strength_plot", ROOT / "scripts/plotting/plot_driven_mg_strength_sweep.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_point(data_root: Path, result_root: Path, scale: float) -> None:
    tag = MODULE._tag(scale)
    metadata = {
        "parameters": {"drive_scale": scale},
        "split_indices": {"train": [0, 1], "validation": [2], "test": [3]},
    }
    drive = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    arrays = {"metadata_json": np.asarray(json.dumps(metadata))}
    for split in ("train", "validation", "test"):
        arrays[f"{split}_drive"] = drive
        arrays[f"{split}_state"] = drive + scale
    np.savez_compressed(data_root / f"mg_ds{tag}.npz", **arrays)
    summary = {
        method: {
            "mean_long_history_value": 0.6,
            "mean_long_null_history_value": 0.1,
            "mean_long_history_beyond_null": 0.5,
            "mean_long_drive_value": 0.2,
        }
        for method in ("linear", "nonlinear")
    }
    (result_root / f"ds{tag}.json").write_text(
        json.dumps(
            {
                "input_metadata": metadata,
                "official_test_touched": False,
                "selected_history_window": 4,
                "long_horizon_summary": summary,
                "shapley": {
                    method: {
                        "mean_long_history_shapley": 0.55,
                        "mean_long_drive_shapley": 0.25,
                        "mean_long_null_history_shapley": 0.05,
                        "mean_long_history_shapley_beyond_null": 0.50,
                        "mean_long_explained_value": 0.80,
                    }
                    for method in ("linear", "nonlinear")
                },
            }
        )
    )


def test_load_requires_and_verifies_common_random_numbers(tmp_path):
    data_root = tmp_path / "data"
    result_root = tmp_path / "results"
    data_root.mkdir()
    result_root.mkdir()
    for scale in (0.0, 0.04, 0.08):
        _write_point(data_root, result_root, scale)
    rows, checks = MODULE._load(result_root, data_root, (0.0, 0.04, 0.08))
    assert [row["drive_scale"] for row in rows] == [0.0, 0.04, 0.08]
    assert checks["common_drive_realizations"] is True
    assert checks["common_split_indices"] is True
    assert rows[0]["linear"]["history_beyond_null"] == 0.5
    assert rows[0]["linear"]["history_shapley"] == 0.55
    assert rows[0]["linear"]["delta_history_shapley"] == 0.50


def test_scale_tag_is_stable():
    assert MODULE._tag(0.0) == "0p00"
    assert MODULE._tag(0.08) == "0p08"
