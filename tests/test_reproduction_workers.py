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
    assert shell('source configs/reproduce/known_snr.sh; echo "$KNOWN_SNR_NUM_WORKERS"') == '4'


def test_baseline_resolves_and_exports_workers_before_launch():
    source = (ROOT / 'scripts/train/run_internal_dw_baseline_or_k_one.sh').read_text()
    # Execute only the configuration preamble: no directories, locks or training.
    preamble = source.split('case "${PHASE}" in', 1)[0]
    prefix = 'PROJECT_ROOT=$(pwd); PHASE=A DATA=mg ARM=clip K=32 SEED=0 GPUS=0; '
    assert shell(prefix + 'unset NUM_WORKERS;\n' + preamble + '\necho "$NUM_WORKERS"').splitlines()[-1] == '4'
    assert shell(prefix + 'NUM_WORKERS=2;\n' + preamble + '\necho "$NUM_WORKERS"').splitlines()[-1] == '2'
    assert 'NUM_WORKERS=0 RECURRENT_EVAL_HORIZON_BATCH' not in source
