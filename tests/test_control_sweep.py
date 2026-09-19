import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.reproduce import sweep_clip_jreg as sweep


CONFIG = sweep.read(sweep.ROOT / 'configs/reproduce/control_sweep.json')


def test_three_candidates_each_and_canonical_geometry():
    assert all(len(g) == 3 for g in CONFIG['grids'].values())
    assert CONFIG['selection_seed'] == 0
    assert CONFIG['report_seeds'] == [0, 1, 2]


def test_selection_is_minimum_and_stable_on_ties():
    assert sweep.choose([0.1, 0.3, 1.0], [2, 1, 3]) == 0.3
    assert sweep.choose([0.1, 0.3, 1.0], [1, 1, 3]) == 0.1
    with pytest.raises(ValueError):
        sweep.choose([0.1, 0.3, 1.0], [1, float('nan'), 3])


def test_metric_rejects_test_during_selection(tmp_path):
    path = tmp_path / 'record.json'
    path.write_text(json.dumps(dict(status='complete', split='test',
                                   primary_metric=dict(name='mean_relative_l2', value=1.0))))
    with pytest.raises(ValueError):
        sweep.metric(path, 'val')


@pytest.mark.parametrize('dataset', ['mg', 'ettm1', 'ettm2', 'shear', 'narma', 'fmri', 'wb2'])
def test_training_only_and_geometry(tmp_path, dataset):
    task = (dataset, 'jreg', 0.01, 0, 'all')
    cmd, env = sweep.train_spec(task, tmp_path, '2', CONFIG)
    assert env['TRAIN_ONLY'] == '1'
    assert env['GPUS'] == '2'
    assert int(env['NUM_WORKERS']) == CONFIG['datasets'][dataset]['num_workers']
    assert env['JREG_LAMBDA'] == '0.01'
    assert env['ROOT'].startswith(str(tmp_path))
    assert int(env['MEM_BATCH']) == CONFIG['datasets'][dataset]['batch']
    assert int(env['MEM_GRAD_ACCUM']) == CONFIG['datasets'][dataset]['accum']


@pytest.mark.parametrize('arm,value,key', [('clip', 0.3, '--grad_clip'),
                                         ('jreg', 0.01, '--forward_jacobian_lambda')])
def test_ieeg_uses_public_model_and_only_changes_coefficient(tmp_path, arm, value, key):
    cmd, env = sweep.train_spec(('ieeg', arm, value, 0, 'sub-CS41'), tmp_path, '0', CONFIG)
    assert cmd[cmd.index(key) + 1] == str(value)
    assert cmd[cmd.index('--mode') + 1] == 'train'
    assert cmd[cmd.index('--local_batch_size') + 1] == '4'
    assert cmd[cmd.index('--grad_accum_steps') + 1] == '8'
    assert cmd[cmd.index('--num_workers') + 1] == '0'


def test_candidates_and_seeds_have_distinct_paths(tmp_path):
    paths = {sweep.run_dir(tmp_path, ('mg', a, v, s, 'all'))
             for a, grid in CONFIG['grids'].items() for v in grid for s in [0, 1, 2]}
    assert len(paths) == 18


@pytest.mark.parametrize('dataset', list(CONFIG['datasets']))
def test_validation_evaluator_explicit_split(tmp_path, dataset):
    cmd = sweep.eval_command(tmp_path / 'best.pth', tmp_path / 'val.json', dataset, 'clip', 'val', CONFIG)
    assert cmd[cmd.index('--split') + 1] == 'val'
    assert 'test' not in cmd


def test_failed_job_does_not_stop_queue(monkeypatch):
    completed = []
    def fake_one(args, config, task, gpu, split):
        if task == 'bad':
            raise RuntimeError('missing input')
        completed.append(task)
    monkeypatch.setattr(sweep, 'one', fake_one)
    failures = sweep.phase(SimpleNamespace(gpus=['0', '1', '2', '3']), CONFIG,
                           ['bad', 'good1', 'good2', 'good3'], 'val')
    assert sorted(completed) == ['good1', 'good2', 'good3']
    assert len(failures) == 1


def test_resume_reuses_only_matching_completed_run(tmp_path, monkeypatch):
    task = ('mg', 'clip', 0.3, 0, 'all')
    args = SimpleNamespace(root=tmp_path, dry_run=False)
    output = sweep.run_dir(tmp_path, task)
    calls = []
    def fake_execute(cmd, log, env):
        calls.append(cmd)
        if '--out' in cmd:
            result = Path(cmd[cmd.index('--out') + 1])
            result.write_text(json.dumps(dict(status='complete', split='val',
                                             primary_metric=dict(name='mean_relative_l2', value=0.5))))
        else:
            (output / 'best.pth').touch()
    monkeypatch.setattr(sweep, 'execute', fake_execute)
    sweep.one(args, CONFIG, task, '0', 'val')
    sweep.one(args, CONFIG, task, '1', 'val')
    assert len(calls) == 2
    assert sweep.metric(output / 'val.json', 'val') == 0.5
