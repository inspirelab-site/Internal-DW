import json
from types import SimpleNamespace

import numpy as np

from internal_dw.datasets.registry import dataset_has_external_input
from internal_dw.datasets.synthetic_memory import (
    build_driven_mackey_glass_splits,
    generate_driven_mackey_glass,
    generate_mackey_glass,
)


def _arguments():
    return dict(
        n_traj=2,
        D=2,
        tau=8.0,
        dt=1.0,
        solver_dt=0.25,
        n_samples=32,
        beta=0.2,
        gamma=0.1,
        n_exp=10.0,
        transient=16,
        seed=7,
    )


def test_driven_mg_shapes_reproducibility_and_metadata():
    arguments = _arguments()
    first_state, first_drive = generate_driven_mackey_glass(
        **arguments, drive_scale=0.02, drive_rho=0.8
    )
    second_state, second_drive = generate_driven_mackey_glass(
        **arguments, drive_scale=0.02, drive_rho=0.8
    )
    assert first_state.shape == (2, 32, 2)
    assert first_drive.shape == first_state.shape
    assert np.isfinite(first_state).all()
    assert np.isfinite(first_drive).all()
    np.testing.assert_array_equal(first_state, second_state)
    np.testing.assert_array_equal(first_drive, second_drive)
    assert dataset_has_external_input("mackey_glass_driven")


def test_zero_drive_recovers_autonomous_mg_states():
    arguments = _arguments()
    autonomous = generate_mackey_glass(**arguments)
    driven, _ = generate_driven_mackey_glass(
        **arguments, drive_scale=0.0, drive_rho=0.8
    )
    np.testing.assert_array_equal(driven, autonomous)


def test_presplit_archive_is_fixed_and_train_normalized(tmp_path):
    arguments = _arguments()
    states, drives = generate_driven_mackey_glass(
        **arguments, drive_scale=0.02, drive_rho=0.8
    )
    # Four disjoint synthetic trajectories are sufficient to exercise the
    # archive loader without invoking the expensive formal generator.
    states = np.concatenate([states, states + 0.1], axis=0)
    drives = np.concatenate([drives, drives], axis=0)
    archive = tmp_path / "driven_mg.npz"
    metadata = {
        "parameters": {
            "dim": 2,
            "tau": 8.0,
            "length": 32,
            "trajectories": 4,
            "drive_scale": 0.02,
            "drive_rho": 0.8,
        }
    }
    np.savez_compressed(
        archive,
        metadata_json=np.asarray(json.dumps(metadata)),
        train_state=states[:2],
        train_drive=drives[:2],
        validation_state=states[2:3],
        validation_drive=drives[2:3],
        test_state=states[3:],
        test_drive=drives[3:],
    )
    args = SimpleNamespace(
        mg_dim=2,
        mg_tau=8.0,
        mg_dt=1.0,
        mg_solver_dt=0.25,
        mg_len=32,
        mg_traj=4,
        mg_beta=0.2,
        mg_gamma=0.1,
        mg_n=10.0,
        mg_transient=16,
        mg_seed=7,
        mg_drive_scale=0.02,
        mg_drive_rho=0.8,
        mg_driven_npz=str(archive),
        data_path=str(tmp_path),
        seed=999,
        train_ratio=0.7,
        val_ratio=0.15,
    )
    train, validation, test = build_driven_mackey_glass_splits(args)
    assert (len(train), len(validation), len(test)) == (2, 1, 1)
    assert args.roi_dim == args.stim_dim == 2
    stacked_train = train.states
    assert abs(float(stacked_train.mean())) < 1e-6
    assert abs(float(stacked_train.std()) - 1.0) < 1e-6
    np.testing.assert_array_equal(train.inputs, drives[:2])
