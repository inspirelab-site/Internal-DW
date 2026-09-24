"""Check reference worker counts without launching training or reading datasets."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which('bash')
if Path('D:/Git/bin/bash.exe').exists():
    BASH = 'D:/Git/bin/bash.exe'


def shell(command):
    if not BASH:
        pytest.skip('Bash is required for launcher tests')
    return subprocess.check_output([BASH, '-c', command], cwd=ROOT, text=True).strip()


def test_reference_workers_all_methods_and_seeds():
    output = shell('source configs/reproduce/figure6.sh; '
                   'for d in mg narma ettm1 ettm2 shear fmri ieeg wb2; do '
                   'for a in exact dw clip jreg; do for s in 0 1 2; do '
                   'echo "$d $a $s $(figure6_num_workers "$d" "$a" "$s")"; '
                   'done; done; done')
    rows = output.splitlines()
    assert len(rows) == 96
    sweep = json.loads((ROOT / 'configs/reproduce/control_sweep.json').read_text())
    for row in rows:
        data, arm, seed, workers = row.split()
        if data in ('mg', 'narma', 'fmri'):
            expected = 4
        elif data in ('ettm1', 'ettm2', 'ieeg'):
            expected = 0
        elif data == 'shear':
            expected = 2 if arm in ('clip', 'jreg') else 0
        else:
            expected = 2 if arm in ('clip', 'jreg') or (arm == 'exact' and seed == '0') else 0
        assert int(workers) == expected, row
        if arm in ('clip', 'jreg'):
            assert sweep['datasets'][data]['num_workers'] == expected


def test_known_snr_full_reference_workers():
    config = json.loads((ROOT / 'configs/reproduce/known_snr.json').read_text())
    assert config['probes']['reference_workers'] == 4
    assert config['training']['num_workers'] == 0


def test_baseline_resolves_and_exports_workers_before_launch():
    source = (ROOT / 'scripts/train/run_internal_dw_baseline_or_k_one.sh').read_text()
    # Execute only the configuration preamble: no directories, locks or training.
    preamble = source.split('case "${PHASE}" in', 1)[0]
    prefix = 'PROJECT_ROOT=$(pwd); PHASE=A DATA=mg ARM=clip K=32 SEED=0 GPUS=0; '
    assert shell(prefix + 'unset NUM_WORKERS;\n' + preamble + '\necho "$NUM_WORKERS"').splitlines()[-1] == '4'
    assert shell(prefix + 'NUM_WORKERS=2;\n' + preamble + '\necho "$NUM_WORKERS"').splitlines()[-1] == '2'
    assert 'NUM_WORKERS=0 RECURRENT_EVAL_HORIZON_BATCH' not in source


SELECTED = {
    'mg': (1.0, 1.0), 'ettm1': (0.3, 1.0), 'ettm2': (0.3, 1.0),
    'shear': (0.3, 0.01), 'narma': (0.1, 1.0), 'ieeg': (0.3, 0.01),
    'fmri': (1.0, 1.0), 'wb2': (1.0, 0.01),
}


@pytest.mark.parametrize('dataset', SELECTED)
def test_selected_control_defaults_and_sweep_overrides(dataset):
    source = (ROOT / 'scripts/train/run_internal_dw_baseline_or_k_one.sh').read_text()
    block = 'GRAD_CLIP=1.0' + source.split('GRAD_CLIP=1.0', 1)[1].split('out=""', 1)[0]
    prefix = ('source configs/reproduce/figure6.sh; '
              f'DATA={dataset}; unset CLIP_NORM JREG_LAMBDA JREG_TARGET JREG_EPS; ')
    clip, jreg = SELECTED[dataset]
    assert float(shell(prefix + 'ARM=clip; ' + block + '\necho "$GRAD_CLIP"')) == clip
    args = shell(prefix + 'ARM=jreg; ' + block + '\necho "$EXTRA_ARGS"').split()
    assert float(args[args.index('--forward_jacobian_lambda') + 1]) == jreg
    assert float(shell(prefix + 'ARM=clip; CLIP_NORM=0.1; ' + block + '\necho "$GRAD_CLIP"')) == 0.1
    args = shell(prefix + 'ARM=jreg; JREG_LAMBDA=0.1; ' + block + '\necho "$EXTRA_ARGS"').split()
    assert float(args[args.index('--forward_jacobian_lambda') + 1]) == 0.1


def test_ieeg_selected_defaults_match_figure6():
    from scripts.reproduce.train_test_ieeg import train_command
    for arm, key, value in [('clip', '--grad_clip', SELECTED['ieeg'][0]),
                            ('jreg', '--forward_jacobian_lambda', SELECTED['ieeg'][1])]:
        cmd = train_command(Path('prepared/sub-CS41.npz'), Path('out'), arm, 0)
        assert float(cmd[cmd.index(key) + 1]) == value
    output = shell('source configs/reproduce/figure6.sh; '
                   'echo "${FIGURE6_CLIP_NORM[ieeg]} ${FIGURE6_JREG_LAMBDA[ieeg]}"')
    assert tuple(map(float, output.split())) == SELECTED['ieeg']


def test_one_command_propagates_selected_controls():
    source = (ROOT / 'scripts/reproduce/train_test_figure6_all.sh').read_text()
    function = source.split('run_phase_arm() {', 1)[1].split('run_primary()', 1)[0]
    assert 'CLIP_NORM="${CLIP_NORM:-${FIGURE6_CLIP_NORM[$data]}}"' in function
    assert 'JREG_LAMBDA="${JREG_LAMBDA:-${FIGURE6_JREG_LAMBDA[$data]}}"' in function
    assert 'bash scripts/reproduce/train_test_ieeg.sh' in source
