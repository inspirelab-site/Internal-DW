"""TBPTT selection and evaluation use only the requested experiment trees."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts.results import build_tbptt_results as results

ROOT = Path(__file__).resolve().parents[1]
BASH = os.environ.get('BASH_EXECUTABLE') or shutil.which('bash')


def shell_path(path):
    value = str(path).replace('\\', '/')
    return '/' + value[0].lower() + value[2:] if len(value) > 1 and value[1] == ':' else value


def test_select_and_aggregate_in_fresh_tree(tmp_path):
    root = tmp_path / 'runs'
    selection = tmp_path / 'selection.json'
    for candidates in results._checkpoint_map(root).values():
        for index, path in enumerate(candidates.values()):
            assert path.is_relative_to(root)
            path.parent.mkdir(parents=True)
            torch.save({'best_val': 0.2 + index, 'epoch': 10}, path)
            (path.parent / '.train_complete').touch()
    results.select(SimpleNamespace(root=root, out=selection))
    picked = json.loads(selection.read_text())
    assert picked['datasets']['mg']['selected_S'] == 8
    assert picked['datasets']['shear']['selected_S'] == 8
    dense = tmp_path / 'dense'
    for data, item in picked['datasets'].items():
        for seed in range(3):
            path = results._result_path(dense, data, item['selected_S'], seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({'summary': {'all_horizons': {'mean': 1 + seed}}}))
    out = tmp_path / 'summary.json'
    results.aggregate(SimpleNamespace(selection=selection, dense_root=dense, out=out))
    for row in json.loads(out.read_text())['results']:
        assert row['mean'] == 2 and row['sample_sd'] == 1
        assert all(Path(path).is_relative_to(dense) for path in row['sources'])
    results._result_path(dense, 'mg', 8, 0).unlink()
    with pytest.raises(SystemExit):
        results.aggregate(SimpleNamespace(selection=selection, dense_root=dense, out=out))


@pytest.mark.parametrize('data,segment', [('mg',8),('mg',16),('shear',8),('shear',16),
                                        ('ettm1',16),('ettm1',32),('ettm2',16),('ettm2',32)])
def test_selected_evaluator_reads_current_checkpoint(tmp_path, data, segment):
    if not BASH:
        pytest.skip('Bash required')
    script = tmp_path / 'scripts/evaluate/evaluate_tbptt_positive_selected_one.sh'
    script.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / script.relative_to(tmp_path), script)
    checkpoint = results._checkpoint_map(tmp_path / 'runs')[data][segment]
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'checkpoint is not loaded by this test')
    selection = tmp_path / 'selection.json'
    selection.write_text(json.dumps({'datasets': {data: {'selected_S': segment}}}))
    hook = tmp_path / 'hook.sh'
    hook.write_text('python() { if [[ "$1" == "-" ]]; then "$REAL_PYTHON" "$@"; '
                    'else printf "ARGV_BEGIN\\n"; printf "%s\\n" "$@"; return 73; fi; }\n')
    env = dict(os.environ, DATA=data, SEED='0', GPU='0', SELECTION='selection.json',
               SWEEP_ROOT='runs', OUT_ROOT='dense', FORCE_EVAL='1',
               BASH_ENV=shell_path(hook), REAL_PYTHON=shell_path(sys.executable))
    proc = subprocess.run([BASH, str(script.relative_to(tmp_path))], cwd=tmp_path,
                          env=env, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 73, proc.stdout + proc.stderr
    argv = proc.stdout.split('ARGV_BEGIN\n')[1].splitlines()
    assert argv[argv.index('--ckpt') + 1] == checkpoint.relative_to(tmp_path).as_posix()
    assert argv[argv.index('--out') + 1] == f'dense/{data}/tbptt{segment}_seed0.json'
    assert argv[argv.index('--max-horizon') + 1] == ('48' if data in ('mg','shear') else '96')
