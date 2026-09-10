"""Small, synthetic checks of the public iEEG cohort interface (no FIF/GPU needed)."""
import copy
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.data import prepare_ieeg as prep
from scripts.reproduce import train_test_ieeg as run
from internal_dw.models.official_state_mamba import OfficialStateMambaARModel
from internal_dw.training.ar_losses import compute_recurrent_state_bptt_loss


def test_alignment_and_train_only_scaling():
    values = np.arange(12).reshape(3, 4)
    actual = prep.align_features(np.array([0., .02, .04, .06, .08]), np.array([0., .04, .08]), values)
    np.testing.assert_array_equal(actual, values[[0, 0, 1, 1, 2]])
    arrays = {'train_state': np.arange(8, dtype=float).reshape(1, 4, 2),
              'validation_state': np.full((1, 4, 2), 100.), 'test_state': np.full((1, 4, 2), 200.)}
    prep.normalize_splits(arrays, 'state')
    np.testing.assert_allclose(arrays['train_state'].mean(axis=(0, 1)), 0., atol=1e-6)
    assert arrays['test_state'].mean() > arrays['validation_state'].mean() > 10


def test_clip_input_contract(tmp_path):
    np.save(tmp_path / 'clip_projected.npy', np.zeros((2, 512), dtype=np.float32))
    (tmp_path / 'clip_frames.csv').write_text('frame\nframe_000000.png\nframe_000001.png\n')
    values, times = prep.load_clip(tmp_path, 25.)
    assert values.shape == (2, 512)
    np.testing.assert_allclose(times, [0., .04])


def test_training_reuses_existing_data_and_valid_results(tmp_path, monkeypatch):
    from contextlib import contextmanager
    args = SimpleNamespace(prepared=tmp_path / 'prepared', out=tmp_path / 'runs',
                           gpu='0', mode='run', dry_run=False)
    archive = args.prepared / 'prepared/sub-CS41.npz'
    prep.save_npz(archive, train_state=np.ones((1, 4, 2)))
    calls = []
    @contextmanager
    def no_lock(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        yield 0
    def fake_execute(command, log, env, fds):
        calls.append(command)
        if '--save_root' in command:
            output = Path(command[command.index('--save_root')+1])
            (output / 'best.pth').write_text('test checkpoint')
        else:
            target = Path(command[command.index('--out')+1])
            prep.write_json(target, dict(status='complete', split='test', method='full_bptt',
                train_horizon=64, eval_horizon=96,
                primary_metric=dict(name='mean_relative_l2', horizons='1:96', value=1.)))
    monkeypatch.setattr(run, 'lock', no_lock)
    monkeypatch.setattr(run, 'execute', fake_execute)
    run.run_one(args, 'sub-CS41', 'full_bptt', 0)
    assert len(calls) == 2
    run.run_one(args, 'sub-CS41', 'full_bptt', 0)
    assert len(calls) == 2
    assert all('prepare_ieeg.py' not in ' '.join(cmd) for cmd in calls)


def test_prepare_chunk_boundaries_and_stimulus(tmp_path, monkeypatch):
    paths = [tmp_path / 'P41CS_R1_enc_macro_theta.fif', tmp_path / 'P41CS_R2_enc_macro_theta.fif']
    n = 24000
    class Raw:
        ch_names = ['x', 'y']; n_times = n; info = {'sfreq': 50.}
        def close(self): pass
    monkeypatch.setitem(sys.modules, 'mne', SimpleNamespace(io=SimpleNamespace(read_raw_fif=lambda *a, **kw: Raw())))
    state = np.column_stack([np.arange(n), np.arange(n)**2]).astype(np.float32)
    monkeypatch.setattr(prep, '_load_and_decimate', lambda *a, **kw: (state.copy(), 50.))
    times = np.arange(12000) / 25
    # Alignment/chunking is dimension-independent; keep this integration fixture small.
    drive = np.repeat(times[:, None], 3, axis=1).astype(np.float32)
    arrays, meta = prep.adapt_subject('P41CS', paths, drive, times, 25.)
    assert arrays['train_state'].shape == (32, 1024, 2)
    assert arrays['validation_state'].shape == (6, 1024, 2)
    assert arrays['test_drive'].shape == (6, 1024, 3)
    np.testing.assert_array_equal(arrays['train_state'][:16], arrays['train_state'][16:])
    assert meta['splits']['train']['movie_stop_seconds'] == (round(.7*n)-256)*.02


def test_training_settings_and_methods(tmp_path):
    for arm in run.ARMS:
        cmd = run.train_command(tmp_path / 's.npz', tmp_path / arm, arm, 2)
        def option(k): return cmd[cmd.index('--' + k) + 1]
        assert option('local_batch_size') == '4' and option('grad_accum_steps') == '8'
        assert option('prepared_temporal_standardize') == '0'
        assert option('early_stop_patience') == '20' and option('num_epochs') == '100'
        assert '--no-recurrent_grad_checkpoint' in cmd and '--compact_recurrent_logging' in cmd
        assert ('--resgrad_routing' in cmd) == (arm == 'internal_dw')
        if arm == 'internal_dw': assert option('dual_wiener_min_probes') == '8'
        if arm == 'clip': assert option('grad_clip') == '0.1'
        if arm == 'jreg': assert option('forward_jacobian_lambda') == '0.1'


def test_timing_preserves_training_and_checks_readiness(tmp_path):
    from scripts.reproduce import time_ieeg as timing
    spec = dict(archive=tmp_path / 'subject.npz', warmup_epochs=5, measured_epochs=3)
    cmd = timing.timing_command(spec, 'internal_dw', 0, tmp_path / 'time')
    assert timing.option(cmd, 'num_epochs') == '8'
    assert timing.option(cmd, 'local_batch_size') == '4'
    assert timing.option(cmd, 'grad_accum_steps') == '8'
    assert timing.option(cmd, 'dual_wiener_min_probes') == '8'
    assert '--no-recurrent_grad_checkpoint' in cmd
    assert '--compact_recurrent_logging' in cmd
    assert '--synchronize_epoch_timing' in cmd
    row = {'epoch': 5, 'train/epoch_wall_seconds': 1., 'train/loss': 1.}
    with pytest.raises(RuntimeError, match='not fully calibrated'):
        timing.validate_epoch(row, 5, True)
    row.update({'timing/dw_ready_fraction': 1., 'timing/dw_solved_batches': 1})
    timing.validate_epoch(row, 5, True)


def test_template_uses_only_training_state(tmp_path):
    from scripts.probes import probe_ieeg_blocked_longmemory_templates as templates
    path = tmp_path / 'subject.npz'
    x = np.arange(48, dtype=np.float32).reshape(2, 6, 4)
    prep.save_npz(path, train_state=x, test_state=np.full_like(x, np.nan))
    actual = templates._load_training_series(SimpleNamespace(prepared_npz=path))
    np.testing.assert_array_equal(actual, x.reshape(-1, 4))


def test_complete_cohort_and_paired_aggregation(tmp_path):
    for i, subject in enumerate(run.SUBJECTS):
        for seed in range(3):
            for arm in run.ARMS:
                full = i + 1 + seed * .01
                multiplier = 1 if arm == 'full_bptt' else (0.8 if i % 2 else 1.1)
                p = tmp_path / subject / arm / f'seed{seed}' / 'test.json'
                prep.write_json(p, dict(status='complete', split='test', method=arm,
                    train_horizon=64, eval_horizon=96,
                    primary_metric=dict(name='mean_relative_l2', horizons='1:96', value=full*multiplier)))
    rows = run.collect_cohort(tmp_path)
    dw = next(r for r in rows if r['arm'] == 'internal_dw')
    assert dw['mean_percent'] == pytest.approx(-5.)
    assert dw['sd_percent'] == pytest.approx(0., abs=1e-12)
    missing = tmp_path / run.SUBJECTS[-1] / 'clip/seed2/test.json'
    missing.unlink()
    with pytest.raises(FileNotFoundError): run.collect_cohort(tmp_path)


@pytest.mark.parametrize('routed,jreg', [(False, 0.), (True, 0.), (False, .1)])
def test_compact_runtime_preserves_loss_and_gradients(routed, jreg):
    torch.set_num_threads(1); torch.manual_seed(19)
    model = OfficialStateMambaARModel(state_dim=4, input_dim=2, hidden_dim=8, depth=2,
        dropout=0., has_external_input=True, residual=True, mamba_d_state=3,
        mamba_d_conv=2, mamba_expand=1, resgrad_routing=routed,
        resgrad_policy='dualwiener' if routed else 'all')
    if routed:
        with torch.no_grad():
            model.dual_wiener.coefficients[..., 0].fill_(.6)
            model.dual_wiener.coefficients[..., 1].fill_(.3)
            model.dual_wiener.seen_batches.fill_(model.dual_wiener.warmup_batches)
            model.dual_wiener.residual_updates.fill_(1)
    compact = copy.deepcopy(model); compact.compact_recurrent_logging = True
    args = SimpleNamespace(mamba_bptt_horizon=4, mamba_burnin=4, mamba_loss_type='rel_l2',
        mamba_loss_decay=1., mamba_train_starts_per_sequence=2, ar_eval_stride=2, stim_dim=2,
        recurrent_val_start_batch=1, forward_jacobian_lambda=jreg, forward_jacobian_eps=.001,
        forward_jacobian_target=1.)
    state = torch.randn(2, 24, 4); drive = torch.randn(2, 24, 2)
    losses = []; gradients = []
    for net, flag in [(model, False), (compact, True)]:
        args.compact_recurrent_logging = flag
        net.train(); random.seed(91); torch.manual_seed(73)
        net.dual_wiener_begin_batch()
        loss, _ = compute_recurrent_state_bptt_loss(net, state, drive, args)
        net.dual_wiener_calibrate(); loss.backward(); net.dual_wiener_end_batch()
        losses.append(loss.detach()); gradients.append([p.grad for p in net.parameters()])
    torch.testing.assert_close(*losses, rtol=0, atol=0)
    for a, b in zip(*gradients):
        if a is None: assert b is None
        else: torch.testing.assert_close(a, b, rtol=0, atol=0)
    model.eval(); compact.eval(); args.forward_jacobian_lambda = 0.
    with torch.no_grad():
        args.compact_recurrent_logging = False
        expected, _ = compute_recurrent_state_bptt_loss(model, state, drive, args)
        args.compact_recurrent_logging = True; args.recurrent_val_start_batch = 16
        actual, _ = compute_recurrent_state_bptt_loss(compact, state, drive, args)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-6, atol=2e-6)
