"""Execute the real shell chain, intercepting only the final Python process."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts.reproduce import sweep_clip_jreg as sweep

ROOT = Path(__file__).resolve().parents[1]
BASH = os.environ.get('BASH_EXECUTABLE') or shutil.which('bash')


def bash_path(path):
    path = str(path).replace('\\', '/')
    return '/' + path[0].lower() + path[2:] if len(path) > 1 and path[1] == ':' else path


def capture(tmp_path, script, env):
    if not BASH:
        pytest.skip('Bash required')
    for name in ('run_mem_one.sh', 'run_tbptt_positive_sweep_one.sh',
                 'run_internal_dw_baseline_or_k_one.sh', 'run_thewell_arm.sh'):
        dest = tmp_path / 'scripts/train' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / 'scripts/train' / name, dest)
    dest = tmp_path / 'configs/reproduce/figure6.sh'
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / 'configs/reproduce/figure6.sh', dest)
    hook = tmp_path / 'intercept.sh'
    hook.write_text("python() { printf '\\nARGV_BEGIN\\n'; printf '%s\\n' \"$@\"; "
                    "printf 'ARGV_END\\n'; return 73; }\n", encoding='utf-8')
    resolved = dict(os.environ, **env)
    resolved.update(BASH_ENV=bash_path(hook), PROJECT_ROOT=bash_path(tmp_path))
    result = subprocess.run([BASH, script], cwd=tmp_path, env=resolved,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 73, result.stdout + result.stderr
    return result.stdout.split('ARGV_BEGIN\n', 1)[1].split('ARGV_END', 1)[0].splitlines()


def value(argv, key):
    return argv[argv.index(key) + 1]


def test_ett_selected_segments():
    if not BASH:
        pytest.skip('Bash required')
    result = subprocess.check_output(
        [BASH, '-c', 'source configs/reproduce/figure6.sh; '
         'echo "${FIGURE6_TBPTT_SEGMENT[ettm1]} ${FIGURE6_TBPTT_SEGMENT[ettm2]}"'],
        cwd=ROOT, text=True)
    assert result.strip() == '32 32'


def test_tbptt_uses_only_configured_output_tree():
    for script in ('scripts/train/run_tbptt_positive_sweep_one.sh',
                   'scripts/reproduce/train_test_figure6_all.sh'):
        source = (ROOT / script).read_text()
        assert 'experiments/tbptt_mg_a8/' not in source
        assert 'experiments/thewell_shear_final_lr3e4/' not in source


@pytest.mark.parametrize('data', ['ettm1', 'ettm2'])
@pytest.mark.parametrize('segment', [16, 32])
def test_ett_tbptt_final_training_arguments(tmp_path, data, segment):
    prepared = tmp_path / 'inputs'
    prepared.mkdir()
    (prepared / (data + '.npz')).write_bytes(b'not loaded: training intercepted')
    argv = capture(tmp_path, 'scripts/train/run_tbptt_positive_sweep_one.sh', dict(
        DATA=data, S=str(segment), SEED='0', GPUS='0', RUN_MODE='train',
        PREPARED_INPUT_ROOT=bash_path(prepared), SWEEP_ROOT='fresh'))
    expected = {'--grad_clip': '1.0', '--num_workers': '0', '--ar_scheduler': 'step',
                '--base_lr': '1e-4', '--weight_decay': '1e-4', '--local_batch_size': '32',
                '--grad_accum_steps': '1', '--mamba_bptt_horizon': '64',
                '--mamba_train_starts_per_sequence': '16', '--mamba_loss_type': 'rel_l2',
                '--num_epochs': '100', '--early_stop_patience': '20', '--mode': 'train',
                '--bptt_detach_period': str(segment)}
    for key, expected_value in expected.items():
        assert value(argv, key) == expected_value, key


@pytest.mark.parametrize('arm,coefficient', [('clip', 0.3), ('jreg', 0.01)])
@pytest.mark.parametrize('from_sweep', [False, True])
def test_shear_final_training_arguments(tmp_path, arm, coefficient, from_sweep):
    for split in ('train', 'valid', 'test'):
        (tmp_path / 'data' / split).mkdir(parents=True)
    env = dict(PHASE='A', DATA='shear', ARM=arm, K='32', SEED='0', GPUS='0',
               ROOT='fresh', SYNC_ROOT='sync', EVAL_ROOT='eval', TRAIN_ONLY='1',
               DATA_PATH='data', CLIP_NORM=str(coefficient), JREG_LAMBDA=str(coefficient))
    if from_sweep:
        config = sweep.read(ROOT / 'configs/reproduce/control_sweep.json')
        _, sweep_env = sweep.train_spec(('shear', arm, coefficient, 0, 'all'), tmp_path, '0', config)
        for key in ('NUM_WORKERS', 'AR_SCHEDULER', 'SHEAR_BATCH', 'SHEAR_GRAD_ACCUM'):
            env[key] = sweep_env[key]
    argv = capture(tmp_path, 'scripts/train/run_internal_dw_baseline_or_k_one.sh', env)
    expected = {'--ar_scheduler': 'cosine', '--ar_min_lr': '1e-6', '--num_workers': '0',
                '--base_lr': '3e-4', '--weight_decay': '0', '--ar_optimizer': 'adam',
                '--local_batch_size': '1', '--grad_accum_steps': '4',
                '--num_epochs': '100', '--early_stop_patience': '20',
                '--bptt_horizon': '32', '--bptt_loss_type': 'rel_l2', '--mode': 'train',
                '--grad_clip': str(coefficient) if arm == 'clip' else '1.0'}
    for key, expected_value in expected.items():
        assert value(argv, key) == expected_value, key
    if arm == 'jreg':
        assert float(value(argv, '--forward_jacobian_lambda')) == coefficient
