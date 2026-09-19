"""Seed-0 validation-only selection, followed by three-seed selected controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import queue
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.reproduce.train_test_ieeg import CONFIG as IEEG_CONFIG, train_command


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def tag(value):
    return format(float(value), '.8g').replace('.', 'p')


def metric(path, split):
    data = read(path)
    value = float(data['primary_metric']['value'])
    if (data.get('status') != 'complete' or data.get('split') != split
            or data['primary_metric']['name'] != 'mean_relative_l2'
            or not math.isfinite(value)):
        raise ValueError('Invalid ' + split + ' result: ' + str(path))
    return value


def choose(grid, scores):
    if len(grid) != len(scores) or not all(math.isfinite(v) for v in scores):
        raise ValueError('Every candidate must have a finite validation score')
    # Predeclared ties favor the first value in the configured grid.
    return min(zip(grid, scores), key=lambda pair: pair[1])[0]


def run_dir(root, task):
    data, arm, value, seed, subject = task
    return root / data / arm / tag(value) / subject / ('seed' + str(seed))


def env_for(gpu):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(('DUAL_WIENER_', 'TORCHELASTIC_')) or key in (
                'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
            env.pop(key, None)
    env.update(PYTHONUNBUFFERED='1', PROJECT_ROOT=str(ROOT),
               PYTHONPATH=str(ROOT / 'src') + os.pathsep + str(ROOT),
               CUDA_VISIBLE_DEVICES=gpu)
    return env


def train_spec(task, output, gpu, config):
    data, arm, value, seed, subject = task
    env = env_for(gpu)
    if data == 'ieeg':
        prepared = Path(env.get('IEEG_PREPARED_ROOT', ROOT / 'probe_inputs/ieeg_cohort_v1'))
        archive = prepared / 'prepared' / (subject + '.npz')
        cmd = train_command(archive, output, arm, seed)
        cmd[cmd.index('--num_workers') + 1] = str(config['datasets'][data]['num_workers'])
        key = '--grad_clip' if arm == 'clip' else '--forward_jacobian_lambda'
        cmd[cmd.index(key) + 1] = str(value)
        return cmd, env
    spec = config['datasets'][data]
    env['NUM_WORKERS'] = str(spec['num_workers'])
    env.update(PHASE='A', DATA=data, ARM=arm, SEED=str(seed), K=str(spec['K']),
               GPUS=gpu, ROOT=str(output / 'train'), SYNC_ROOT=str(output / 'sync'),
               EVAL_ROOT=str(output / 'unused_test'), TRAIN_ONLY='1', SKIP_EXISTING='1',
               CLIP_NORM=str(value if arm == 'clip' else 1.0),
               JREG_LAMBDA=str(value if arm == 'jreg' else 0.0),
               JREG_TARGET='1.0', JREG_EPS='0.001')
    for prefix in ('MEM', 'ETT', 'SHEAR', 'FMRI', 'WB2_TRAIN'):
        env[prefix + '_BATCH'] = str(spec['batch'])
        env[prefix + '_GRAD_ACCUM'] = str(spec['accum'])
    if data == 'shear' and env.get('SHEAR_DATA_PATH'):
        env['DATA_PATH'] = env['SHEAR_DATA_PATH']
    if data == 'wb2':
        env['DATA_PATH'] = env.get('WB2_DATA_PATH', 'data/weatherbench2_1p5_pilot')
    return ['bash', 'scripts/train/run_internal_dw_baseline_or_k_one.sh'], env


def eval_command(ckpt, output, data, arm, split, config):
    k = config['datasets'][data]['K']
    h = (3 * k + 1) // 2
    common = ['--ckpt', str(ckpt), '--out', str(output), '--gpu', '0',
              '--split', split, '--train-horizon', str(k), '--method-label', arm]
    if data == 'wb2':
        return [sys.executable, '-u', 'scripts/evaluate/evaluate_weatherbench2_acc.py',
                *common, '--data', os.environ.get('WB2_DATA_PATH', 'data/weatherbench2_1p5_pilot'),
                '--horizons', *map(str, range(1, h + 1)), '--batch-size', '2',
                '--start-stride', '1', '--num-starts', '64', '--evenly-spaced-starts']
    origin_batch = {'ettm1': 8, 'ettm2': 8, 'ieeg': 8, 'fmri': 2, 'shear': 2}.get(data, 16)
    return [sys.executable, '-u', 'scripts/evaluate/evaluate_dense_multistart_rel_l2.py',
            *common, '--max-horizon', str(h), '--origin-stride', '1',
            '--max-origins-per-item', '64', '--origin-batch', str(origin_batch),
            '--num-workers', '0', '--bootstrap-draws', '10000']


def execute(command, path, env):
    with path.open('a', encoding='utf-8') as stream:
        stream.write('\n[command] ' + json.dumps(command) + '\n'); stream.flush()
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def one(args, config, task, gpu, split):
    output = run_dir(args.root, task)
    cmd, env = train_spec(task, output, gpu, config)
    if args.dry_run:
        print(json.dumps(dict(task=task, gpu=gpu, split=split, command=cmd,
                              output=str(output))), flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    # Store actual command/env so candidate directories cannot silently mix settings.
    keys = ('DATA_PATH', 'HCP_DATA_PATH', 'SHEAR_DATA_PATH', 'WB2_DATA_PATH',
            'PREPARED_INPUT_ROOT', 'IEEG_PREPARED_ROOT', 'EXTRA_ARGS', 'WELL_REPO')
    protocol = dict(task=task, config=config, ieeg=IEEG_CONFIG if task[0] == 'ieeg' else None,
                    paths={k: env.get(k) for k in keys}, command=cmd)
    protocol = json.loads(json.dumps(protocol))
    manifest = output / 'sweep_config.json'
    if manifest.exists() and read(manifest) != protocol:
        raise ValueError('Settings changed; choose a fresh SWEEP_ROOT: ' + str(output))
    write(manifest, protocol)
    complete = output / 'train.complete'
    print('[start]', gpu, task, split, 'log=' + str(output / 'train.log'), flush=True)
    if not complete.exists():
        execute(cmd, output / 'train.log', env)
    checkpoints = list(output.rglob('best.pth'))
    if len(checkpoints) != 1:
        raise ValueError('Expected exactly one best.pth under ' + str(output))
    complete.touch()
    result = output / (split + '.json')
    if not result.exists():
        partial = result.with_suffix('.partial.json')
        execute(eval_command(checkpoints[0], partial, task[0], task[1], split, config),
                output / (split + '.log'), env_for(gpu))
        metric(partial, split)
        partial.replace(result)
    print('[done]', task, split, metric(result, split), flush=True)


def phase(args, config, tasks, split):
    pending = queue.Queue()
    for task in tasks:
        pending.put(task)
    failures = []
    def worker(gpu):
        while True:
            try:
                task = pending.get_nowait()
            except queue.Empty:
                return
            try:
                one(args, config, task, gpu, split)
            except Exception as exc:
                failure = dict(task=task, split=split, error=str(exc))
                failures.append(failure)
                print('[failed]', failure, flush=True)
            finally:
                pending.task_done()
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        list(pool.map(worker, args.gpus))
    return failures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', default='0,1,2,3')
    p.add_argument('--root', type=Path, default=ROOT / 'experiments/clip_jreg_val_sweep_v1')
    p.add_argument('--config', type=Path, default=ROOT / 'configs/reproduce/control_sweep.json')
    p.add_argument('--datasets', nargs='+')
    p.add_argument('--stage', choices=['all', 'screen', 'selected'], default='all')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    args.root = args.root.resolve(); config = read(args.config)
    args.gpus = args.gpus.split(',')
    if len(set(args.gpus)) != len(args.gpus) or not all(g.isdigit() for g in args.gpus):
        p.error('Use unique physical GPU IDs separated by commas')
    datasets = args.datasets or list(config['datasets'])
    if not set(datasets) <= set(config['datasets']) or len(set(datasets)) != len(datasets):
        p.error('Use known, unique dataset names')
    seed = config['selection_seed']
    subjects = lambda d: IEEG_CONFIG['subjects'] if d == 'ieeg' else ['all']
    tasks = [(d, a, v, seed, s) for d in datasets for a, grid in config['grids'].items()
             for v in grid for s in subjects(d)]
    failures = []
    if args.stage != 'selected':
        failures.extend(phase(args, config, tasks, 'val'))
    if args.dry_run:
        print('[dry-run] No training/evaluation started. Selected runs require actual validation results.')
        return
    selected = {}; report_tasks = []
    for d in datasets:
        for a, grid in config['grids'].items():
            try:
                scores = [statistics.mean(metric(run_dir(args.root, (d, a, v, seed, s)) / 'val.json', 'val')
                                          for s in subjects(d)) for v in grid]
                best = choose(grid, scores)
                record = dict(dataset=d, method=a, selection_seed=seed, grid=grid,
                              validation_scores=scores, selected=best, subjects=subjects(d),
                              aggregation='equal subject mean', metric=config['selection_metric'],
                              test_used_for_selection=False)
                selection_path = args.root / d / a / 'selection.json'
                if selection_path.exists() and read(selection_path) != record:
                    raise ValueError('Selection changed; use a fresh SWEEP_ROOT')
                write(selection_path, record)
                selected[d + '/' + a] = record
                report_tasks += [(d, a, best, r, s) for r in config['report_seeds'] for s in subjects(d)]
                print('[selected]', d, a, best, scores, flush=True)
            except Exception as exc:
                failures.append(dict(dataset=d, arm=a, stage='selection', error=str(exc)))
                print('[selection blocked]', d, a, str(exc), flush=True)
    if args.stage != 'screen':
        failures.extend(phase(args, config, report_tasks, 'test'))
    results = {}
    if args.stage != 'screen':
        for name, selection in selected.items():
            d, a = name.split('/')
            try:
                per_seed = [statistics.mean(metric(run_dir(args.root, (d, a, selection['selected'], r, s))
                                                   / 'test.json', 'test') for s in subjects(d))
                            for r in config['report_seeds']]
                results[name] = dict(per_seed=per_seed, mean=statistics.mean(per_seed),
                                     sample_sd=statistics.stdev(per_seed))
            except (OSError, ValueError, KeyError):
                results[name] = None
    write(args.root / 'sweep_summary.json', dict(selected=selected, results=results, failures=failures,
                                               stage=args.stage, status='partial' if failures else 'complete'))
    print('[summary]', args.root / 'sweep_summary.json', flush=True)
    if failures:
        raise SystemExit('Some tasks failed; fix inputs and rerun the same command.')


if __name__ == '__main__':
    main()
