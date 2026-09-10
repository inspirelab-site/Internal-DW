"""Subject-specific stimulus-present iEEG, using the shared trainer and evaluator."""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data.prepare_ieeg import write_json

CONFIG_PATH = ROOT / 'configs/reproduce/ieeg.json'
CONFIG = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
SUBJECTS = CONFIG['subjects']
ARMS = CONFIG['methods']
DISPLAY = {'full_bptt': 'Exact BPTT', 'internal_dw': 'Internal-DW', 'clip': 'Clip', 'jreg': 'JReg'}


def result_root():
    return Path(os.environ.get('IEEG_RUN_ROOT', ROOT / 'experiments/ieeg_cohort_v1')).resolve()


def train_command(archive, output, arm, seed):
    options = dict(CONFIG['train_options'], seed=seed, data_path=str(archive.parent),
                   prepared_temporal_npz=str(archive), save_root=str(output))
    flags = list(CONFIG['flags'])
    if arm == 'internal_dw':
        options.update(CONFIG['dw_options']); flags.append('resgrad_routing')
    elif arm in ('clip', 'jreg'):
        options.update(CONFIG[arm + '_options'])
    elif arm != 'full_bptt':
        raise ValueError('Unknown method: ' + arm)
    command = [sys.executable, '-u', str(ROOT / 'src/main.py')]
    for key, value in options.items():
        command += ['--' + key, str(value)]
    return command + ['--' + flag for flag in flags] + [
        '--test_horizons', '1', '2', '4', '8', '16', '32', '64', '96']


def test_value(path, arm):
    data = json.loads(path.read_text(encoding='utf-8'))
    metric = data['primary_metric']
    value = float(metric['value'])
    if (data.get('status') != 'complete' or data.get('split') != 'test'
            or data.get('method') != arm or data.get('train_horizon') != 64
            or data.get('eval_horizon') != 96 or metric.get('name') != 'mean_relative_l2'
            or metric.get('horizons') != '1:96' or not math.isfinite(value)):
        raise ValueError('Invalid cohort test record: ' + str(path))
    return value


def collect_cohort(root=None):
    """Require the complete cohort; never fall back to single-subject results."""
    root = result_root() if root is None else Path(root)
    values = {}; paths = {}
    for arm in ARMS:
        paths[arm] = [root / s / arm / f'seed{seed}/test.json'
                      for seed in CONFIG['seeds'] for s in SUBJECTS]
        values[arm] = [test_value(p, arm) for p in paths[arm]]
    full = values['full_bptt']
    rows = []
    count = len(SUBJECTS)
    for arm in ARMS:
        per_seed = [statistics.mean(values[arm][s*count:(s+1)*count]) for s in range(3)]
        paired = [100 * (v / f - 1) for v, f in zip(values[arm], full)]
        changes = [statistics.mean(paired[s*count:(s+1)*count]) for s in range(3)]
        rows.append(dict(method=DISPLAY[arm], arm=arm, per_seed=per_seed,
                         mean=statistics.mean(per_seed), sample_sd=statistics.stdev(per_seed),
                         paired_seed_changes=changes, mean_percent=statistics.mean(changes),
                         sd_percent=statistics.stdev(changes),
                         source_jsons=[str(p) for p in paths[arm]]))
    return rows


@contextmanager
def lock(path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield stream.fileno()


def clean_env(gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED='1')
    for key in list(env):
        if key.startswith(('DUAL_WIENER_', 'TORCHELASTIC_', 'IEEG_FAST_')) or key in (
                'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
            env.pop(key)
    env['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    return env


def execute(command, log, env, fds=()):
    print('[log]', log, flush=True)
    with log.open('a', encoding='utf-8') as stream:
        stream.write('\n[command] ' + json.dumps(command) + '\n'); stream.flush()
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                       pass_fds=tuple(fds), check=True)


def source_signature(archive):
    return dict(path=str(archive.resolve()), bytes=archive.stat().st_size,
                mtime_ns=archive.stat().st_mtime_ns)


def ensure_prior(archive, subject_dir, env, fds=()):
    prior = subject_dir / 'noise_templates_K64.npz'
    signature = source_signature(archive)
    marker = subject_dir / 'noise_templates_source.json'
    with lock(subject_dir / 'prior.lock') as fd:
        if prior.exists():
            if not marker.is_file() or json.loads(marker.read_text()) != signature:
                raise ValueError('Template/data mismatch: use a fresh IEEG_RUN_ROOT')
        else:
            partial = prior.with_suffix('.partial.npz')
            execute([sys.executable, '-u', str(ROOT / 'scripts/probes/probe_ieeg_blocked_longmemory_templates.py'),
                     '--prepared-npz', str(archive), '--output', str(partial),
                     '--subject', 'P' + archive.stem.removeprefix('sub-CS') + 'CS',
                     '--max-horizon', '64', '--device', 'cuda', '--seed', '0'],
                    subject_dir / 'prior.log', env, (*fds, fd))
            partial.with_suffix('.json').replace(prior.with_suffix('.json'))
            write_json(marker, signature)
            partial.replace(prior)
    return prior


def run_one(args, subject, arm, seed):
    archive = args.prepared / 'prepared' / (subject + '.npz')
    if not archive.is_file():
        raise FileNotFoundError(f'{archive}: run train_test_ieeg.sh prepare first')
    output = args.out / subject / arm / f'seed{seed}'
    command = train_command(archive, output, arm, seed)
    if args.dry_run:
        print(json.dumps(dict(subject=subject, arm=arm, seed=seed, mode=args.mode,
                              command=command, test_json=str(output / 'test.json')))); return
    with lock(output / 'run.lock') as fd:
        fingerprint = dict(protocol=CONFIG, archive=source_signature(archive), arm=arm, seed=seed)
        config = output / 'run_config.json'
        if config.is_file():
            if json.loads(config.read_text()) != fingerprint:
                raise ValueError('Configuration/data changed: use a fresh IEEG_RUN_ROOT')
        elif any((output / name).exists() for name in ('last.pth', 'best.pth', 'test.json', 'train.complete')):
            raise ValueError('Existing artifacts have no matching public run_config.json')
        else:
            write_json(config, fingerprint)
        env = clean_env(args.gpu)
        result = output / 'test.json'
        if args.mode in ('run', 'train') and not (output / 'train.complete').exists():
            if arm == 'internal_dw':
                prior = ensure_prior(archive, args.out / subject, env, (fd,))
                env.update(DUAL_WIENER_INNOVATION_FILE=str(prior),
                           DUAL_WIENER_INNOVATION_KEY='innovation_templates')
            execute(command, output / 'train.log', env, (fd,))
            if not (output / 'best.pth').is_file():
                raise RuntimeError('Training did not save best.pth')
            (output / 'train.complete').touch()
        if args.mode in ('run', 'test'):
            if not (output / 'best.pth').is_file():
                raise FileNotFoundError('Missing best.pth: train this subject/method/seed first')
            if not result.is_file():
                partial = result.with_suffix('.partial.json')
                execute([sys.executable, '-u', str(ROOT / 'scripts/evaluate/evaluate_dense_multistart_rel_l2.py'),
                         '--ckpt', str(output / 'best.pth'), '--out', str(partial), '--gpu', '0',
                         '--split', 'test', '--max-horizon', '96', '--train-horizon', '64',
                         '--origin-stride', '1', '--max-origins-per-item', '64', '--origin-batch', '8',
                         '--batch-size', '4', '--num-workers', '0', '--method-label', arm],
                        output / 'test.log', env, (fd,))
                test_value(partial, arm); partial.replace(result)
            print(f'[result] {subject} {arm} seed{seed}: {test_value(result, arm):.6f}')
            print('[test-json]', result)
        print('[checkpoint]', output / 'best.pth', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['run', 'train', 'test', 'summary'], nargs='?', default='run')
    p.add_argument('--prepared', type=Path, default=os.environ.get('IEEG_PREPARED_ROOT', ROOT / 'probe_inputs/ieeg_cohort_v1'))
    p.add_argument('--out', type=Path, default=result_root())
    p.add_argument('--subjects', nargs='+', choices=SUBJECTS, default=SUBJECTS)
    p.add_argument('--arms', nargs='+', choices=ARMS, default=ARMS)
    p.add_argument('--seeds', nargs='+', type=int, choices=[0, 1, 2], default=[0, 1, 2])
    p.add_argument('--gpu', default='0')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(); args.prepared = args.prepared.resolve(); args.out = args.out.resolve()
    if args.mode == 'summary':
        rows = collect_cohort(args.out)
        write_json(args.out / 'cohort_summary.json', dict(subjects=SUBJECTS, seeds=[0, 1, 2], rows=rows))
        print(json.dumps(rows, indent=2)); return
    if not args.gpu.isdigit():
        p.error('Choose one GPU; runs are sequential and are not DDP')
    failures = []
    for subject in args.subjects:
        for arm in args.arms:
            for seed in args.seeds:
                try:
                    run_one(args, subject, arm, seed)
                except Exception as exc:
                    failures.append(dict(subject=subject, arm=arm, seed=seed, error=str(exc)))
                    print('[failed]', failures[-1], flush=True)
    if failures:
        raise SystemExit(f'{len(failures)} runs failed; fix the inputs and rerun the same command')


if __name__ == '__main__':
    main()
