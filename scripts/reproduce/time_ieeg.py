"""Single-GPU paired timing of the stimulus-present iEEG cohort."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import runpy
import statistics as stats
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.reproduce.train_test_ieeg import (
    SUBJECTS, CONFIG, ROOT as REPO, clean_env, lock as exclusive_lock,
    execute, train_command, result_root, source_signature)
from scripts.data.prepare_ieeg import write_json

def set_option(command, key, value):
    command[command.index('--' + key) + 1] = str(value)

def latest_epoch(path):
    if not path.is_file(): return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows[-1] if rows else None

def bind_host(root, name):
    import socket
    path = root / (name + '.owner.json')
    host = socket.gethostname()
    if path.exists():
        if json.loads(path.read_text()) != host: raise RuntimeError('Timing root belongs to another host')
    else:
        with path.open('x') as stream: json.dump(host, stream)


ARMS = ('full_bptt', 'internal_dw')
REPEATS = (0, 1, 2)
DEFAULT_OUT = Path(os.environ.get('IEEG_TIMING_ROOT', REPO / 'experiments/ieeg_cohort_timing_v1'))
SOURCE_RUNS = result_root()
PREPARED = Path(os.environ.get('IEEG_PREPARED_ROOT', REPO / 'probe_inputs/ieeg_cohort_v1'))


def option(command, name):
    return command[command.index('--' + name) + 1]


def subject_spec(subject):
    archive = (PREPARED / 'prepared' / (subject + '.npz')).resolve()
    prior = (SOURCE_RUNS / subject / 'noise_templates_K64.npz').resolve()
    if not archive.is_file() or not prior.is_file():
        raise ValueError('Prepare/train this subject before timing: ' + subject)
    import numpy as np
    with np.load(archive, allow_pickle=False) as data:
        shape = data['train_state'].shape
    for arm in ARMS:
        reference = json.loads((SOURCE_RUNS / subject / arm / 'seed0/run_config.json').read_text())
        if reference['protocol'] != CONFIG or reference['archive'] != source_signature(archive):
            raise ValueError('Forecast settings/data differ: ' + subject)
    if json.loads((SOURCE_RUNS / subject / 'noise_templates_source.json').read_text()) != source_signature(archive):
        raise ValueError('Template/data mismatch')
    batches = math.ceil(shape[0] / CONFIG['train_options']['local_batch_size'])
    dw = CONFIG['dw_options']
    warmup = math.ceil((dw['dual_wiener_warmup_batches'] + dw['dual_wiener_min_probes']
                       * dw['dual_wiener_probe_every']) / batches)
    return dict(subject=subject, archive=str(archive), prior=str(prior), train_chunks=shape[0],
                batches_per_epoch=batches, warmup_epochs=max(1, warmup), measured_epochs=3,
                state_dim=shape[-1], archive_bytes=archive.stat().st_size,
                prior_bytes=prior.stat().st_size, archive_mtime_ns=archive.stat().st_mtime_ns,
                prior_mtime_ns=prior.stat().st_mtime_ns)

def timing_command(spec, arm, repeat, out):
    cmd = train_command(Path(spec['archive']), out, arm, 9100 + repeat)
    epochs = spec['warmup_epochs'] + spec['measured_epochs']
    for key, value in dict(num_epochs=epochs, early_stop_patience=0,
                           eval_every=epochs + 1, resume='').items():
        set_option(cmd, key, value)
    cmd += ['--synchronize_epoch_timing', '--fast_train_logging', '--no-log_gpu_memory']
    return cmd


def timing_env(gpu, arm, spec, uuid):
    env = clean_env(gpu)
    env.update(IEEG_TIMING_WARMUP=str(spec['warmup_epochs']), IEEG_TIMING_GPU_UUID=uuid)
    if arm == 'internal_dw':
        env.update(DUAL_WIENER_INNOVATION_FILE=spec['prior'],
                   DUAL_WIENER_INNOVATION_KEY='innovation_templates')
    return env


def gpu_info(gpu):
    output = subprocess.check_output(['nvidia-smi', '--id=' + gpu,
        '--query-gpu=uuid,name,driver_version,power.limit', '--format=csv,noheader,nounits'], text=True)
    rows = list(csv.reader(output.strip().splitlines()))
    if len(rows) != 1 or len(rows[0]) != 4:
        raise RuntimeError('Expected exactly one physical GPU')
    return dict(zip(('uuid', 'name', 'driver', 'power_limit_watts'), [s.strip() for s in rows[0]]))


def check_exclusive(uuid, own_pid=None):
    output = subprocess.check_output(['nvidia-smi', '--id=' + uuid,
        '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    others = []
    found_self = False
    for row in csv.reader(output.strip().splitlines()):
        if not row:
            continue
        if len(row) != 2 or not row[1].strip().isdigit():
            raise RuntimeError('Cannot verify GPU process ownership: ' + repr(row))
        if row[0].strip() == uuid:
            if int(row[1]) == own_pid:
                found_self = True
            else:
                others.append(int(row[1]))
    if others:
        raise RuntimeError(f'GPU {uuid} is occupied by other compute PIDs {others}; '
                           'do not time concurrently with baseline training')
    if own_pid is not None and not found_self:
        raise RuntimeError('Training PID is not visible on the requested GPU; '
                           'cannot verify physical GPU identity/exclusive ownership')


def validate_epoch(row, warmup, is_dw):
    epoch = int(row['epoch'])
    elapsed = float(row['train/epoch_wall_seconds'])
    if not math.isfinite(elapsed) or elapsed <= 0 or not math.isfinite(float(row['train/loss'])):
        raise ValueError('Invalid timing/loss measurement')
    # This is checked at the END of the final warm-up epoch, before any
    # measured epoch begins, and again after every measured epoch.
    if is_dw and epoch >= warmup:
        if row.get('timing/dw_ready_fraction', 0.) < 1. or row.get('timing/dw_solved_batches', 0) < 1:
            raise RuntimeError('DW is not fully calibrated; refusing a misleading timing result')


def train_entry():
    """Use the shared trainer; add audits AFTER its synchronized timer."""
    sys.path.insert(0, str(REPO / 'src'))
    import torch
    import internal_dw.training.trainer as trainer
    warmup = int(os.environ['IEEG_TIMING_WARMUP'])
    uuid = os.environ['IEEG_TIMING_GPU_UUID']
    shared_train = trainer.train_model

    def audited_train(model, train_loader, val_loader, args, rank=0, exp_dir=None):
        check_exclusive(uuid, os.getpid())
        write_json(Path(exp_dir) / 'runtime.json', dict(torch=torch.__version__, cuda=torch.version.cuda,
                   python=platform.python_version(), gpu_name=torch.cuda.get_device_name(0), gpu_uuid=uuid))
        shared_append = trainer._append_epoch_log
        controller = getattr(model, 'dual_wiener', None)

        def append(exp_dir, epoch_logs, args=None):
            # Neither nvidia-smi nor these CPU copies are inside epoch timing.
            check_exclusive(uuid, os.getpid())
            row = dict(epoch_logs)
            if controller is not None:
                diag = controller.diagnostics(64)
                row['timing/dw_ready_fraction'] = diag.get('ar/dual_wiener_probe_ready_fraction', 0.)
                row['timing/dw_solved_batches'] = int(controller.solved_batches.detach().cpu())
            shared_append(exp_dir, row, args=args)
            validate_epoch(row, warmup, controller is not None)
            phase = 'warmup' if int(row['epoch']) <= warmup else 'measured'
            print(f"[timing] epoch={int(row['epoch'])} {phase} seconds={row['train/epoch_wall_seconds']:.3f}", flush=True)

        trainer._append_epoch_log = append
        try:
            return shared_train(model, train_loader, val_loader, args, rank=rank, exp_dir=exp_dir)
        finally:
            trainer._append_epoch_log = shared_append

    trainer.train_model = audited_train
    sys.argv = [str(REPO / 'src/main.py')] + sys.argv[2:]
    sys.argv += ['--recurrent_val_start_batch', '16']
    runpy.run_path(sys.argv[0], run_name='__main__')


def reduce_run(out, spec, arm):
    rows = [json.loads(line) for line in (out / 'train_logs.jsonl').read_text().splitlines() if line.strip()]
    total = spec['warmup_epochs'] + spec['measured_epochs']
    if [int(r['epoch']) for r in rows] != list(range(1, total + 1)):
        raise ValueError('Missing or repeated epochs: ' + str(out))
    for row in rows:
        validate_epoch(row, spec['warmup_epochs'], arm == 'internal_dw')
    measured = rows[spec['warmup_epochs']:]
    times = [float(r['train/epoch_wall_seconds']) for r in measured]
    return dict(seconds_per_epoch=stats.median(times), epoch_seconds=times,
                measured_epoch_indices=[int(r['epoch']) for r in measured],
                runtime=json.loads((out / 'runtime.json').read_text()), log=str(out / 'train_logs.jsonl'))


def aggregate(pairs, subjects):
    by = {(p['subject'], p['repeat']): p for p in pairs}
    expected = {(s, r) for s in subjects for r in REPEATS}
    if len(by) != len(pairs) or set(by) != expected:
        raise ValueError('Require all subject/repeat pairs, without duplicates')
    hardware = {p['gpu_uuid'] for p in pairs}
    versions = {json.dumps(v['runtime'], sort_keys=True) for p in pairs for v in p['arms'].values()}
    if len(hardware) != 1 or len(versions) != 1:
        raise ValueError('Cannot aggregate different GPUs or software environments')
    repeats = []
    for repeat in REPEATS:
        selected = [by[(s, repeat)] for s in subjects]
        repeats.append(dict(repeat=repeat,
            full_bptt=stats.mean(p['arms']['full_bptt']['seconds_per_epoch'] for p in selected),
            internal_dw=stats.mean(p['arms']['internal_dw']['seconds_per_epoch'] for p in selected),
            increase_percent=stats.mean(100 * (p['arms']['internal_dw']['seconds_per_epoch'] /
                                                p['arms']['full_bptt']['seconds_per_epoch'] - 1) for p in selected)))
    return dict(status='complete', cohort_complete=set(subjects) == set(SUBJECTS), participants=len(subjects),
                paired_repeats=3, units='seconds per participant training epoch',
                aggregation='Per run: median of 3 measured epochs. Equal-participant mean within each repeat; '
                            'mean and sample SD over 3 repeat means. Percentage increases are paired first.',
                gpu_uuid=next(iter(hardware)), per_repeat=repeats,
                **{k: dict(mean=stats.mean(r[k] for r in repeats), sd=stats.stdev(r[k] for r in repeats))
                   for k in ('full_bptt', 'internal_dw', 'increase_percent')})


def collect(root, subjects):
    pairs = []
    for subject in subjects:
        for repeat in REPEATS:
            path = root / subject / f'repeat{repeat}' / 'result.json'
            if path.exists():
                record = json.loads(path.read_text())
                if record['status'] != 'complete' or (record['subject'], record['repeat']) != (subject, repeat):
                    raise ValueError('Invalid completed timing pair: ' + str(path))
                if set(record['arms']) != set(ARMS) or any(
                        not math.isfinite(float(v['seconds_per_epoch'])) or float(v['seconds_per_epoch']) <= 0
                        for v in record['arms'].values()):
                    raise ValueError('Invalid paired timing values: ' + str(path))
                pairs.append(record)
    return pairs


def report(root, subjects, write=False):
    pairs = collect(root, subjects)
    print(f'[timing progress] {len(pairs)}/{len(subjects) * 3} complete pairs; root={root}')
    if (root / 'active.json').exists():
        active = json.loads((root / 'active.json').read_text())
        active_out = root / active['out_relative']
        row = latest_epoch(active_out / 'train_logs.jsonl') or {}
        print(f"[latest run] {active['subject']} repeat{active['repeat']} {active['arm']} "
              f"epoch={int(row.get('epoch', 0))}/{active['epochs']} "
              f"warmup={active['warmup']} s/epoch={row.get('train/epoch_wall_seconds', '-')}")
        print('[log] ' + str(active_out / 'train.log'))
        print('(Latest recorded progress, not a process-liveness check.)')
    if len(pairs) != len(subjects) * 3:
        for subject in subjects:
            done = [p['repeat'] for p in pairs if p['subject'] == subject]
            print(subject, 'completed repeats:', done)
        return
    result = aggregate(pairs, subjects)
    print(json.dumps(result, indent=2))
    if write:
        write_json(root / 'timing_summary.json', result)
        print('[out] ' + str(root / 'timing_summary.json'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('run', 'status', 'summarize'), default='run')
    p.add_argument('--gpu', default='0', help='One idle physical GPU; no concurrent jobs')
    p.add_argument('--subjects', nargs='+', choices=SUBJECTS, default=list(SUBJECTS))
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(); args.out = args.out.resolve()
    if len(set(args.subjects)) != len(args.subjects) or not args.gpu.isdigit():
        p.error('Use distinct subjects and one GPU ID')
    if args.out == SOURCE_RUNS.resolve() or SOURCE_RUNS.resolve() in args.out.parents:
        p.error('Timing outputs must not be inside the forecasting runs')
    if args.mode != 'run':
        report(args.out, args.subjects, write=args.mode == 'summarize'); return
    specs = {s: subject_spec(s) for s in args.subjects}
    for spec in specs.values():
        print(f"[plan] {spec['subject']} warmup={spec['warmup_epochs']} measured=3 epochs; 3 paired repeats")
    if args.dry_run:
        print('[preflight OK] Cohort settings checked; no training launched.'); return
    if os.name != 'posix':
        p.error('Run on a Linux GPU server')
    with exclusive_lock(args.out / 'timing.lock') as fd:
        info = gpu_info(args.gpu)
        check_exclusive(info['uuid'])
        bind_host(args.out, 'timing')
        sources = ('src/main.py', 'src/internal_dw/training/trainer.py',
                   'src/internal_dw/training/ar_losses.py', 'src/internal_dw/models/dual_wiener.py',
                   'src/internal_dw/models/official_state_mamba.py', 'src/internal_dw/models/registry.py',
                   'scripts/reproduce/train_test_ieeg.py', 'scripts/reproduce/time_ieeg.py', 'configs/reproduce/ieeg.json')
        protocol = dict(version=1, hardware=info, subjects=args.subjects, specs=specs,
                        sources={f: hashlib.sha256((REPO / f).read_bytes()).hexdigest() for f in sources},
                        batch=4, accumulation=8, K=64, activation_checkpointing=False, repeats=3,
                        excluded='validation, testing, checkpoint I/O, prior construction, GPU ownership checks')
        config = args.out / 'timing_protocol.json'
        if config.exists() and json.loads(config.read_text()) != protocol:
            raise ValueError('Timing protocol/GPU changed; choose a fresh --out')
        write_json(config, protocol)
        for index, subject in enumerate(args.subjects):
            spec = specs[subject]
            for repeat in REPEATS:
                pair_dir = args.out / subject / f'repeat{repeat}'
                if (pair_dir / 'result.json').exists():
                    collect(args.out, [subject]); print(f'[skip] {subject} repeat{repeat}'); continue
                # An interrupted pair starts fresh (both arms), in a new directory.
                # Never splice pre- and post-restart epochs into one timing sample.
                attempt = 1
                while (pair_dir / f'attempt{attempt:03d}').exists():
                    attempt += 1
                work = pair_dir / f'attempt{attempt:03d}'; work.mkdir(parents=True)
                order = ARMS if (index + repeat) % 2 == 0 else ARMS[::-1]
                values = {}
                for arm in order:
                    check_exclusive(info['uuid'])
                    out = work / arm; out.mkdir()
                    cmd = timing_command(spec, arm, repeat, out)
                    write_json(out / 'command.json', cmd)
                    # Timing entry adds readiness checks, then runs the
                    # exact same trainer/compact logging as the cohort experiment.
                    child = [sys.executable, '-u', str(Path(__file__).resolve()), '--train-entry'] + cmd[3:]
                    write_json(args.out / 'active.json', dict(subject=subject, repeat=repeat, arm=arm,
                               out_relative=out.relative_to(args.out).as_posix(),
                               epochs=spec['warmup_epochs'] + 3, warmup=spec['warmup_epochs']))
                    print(f'[start] {subject} repeat{repeat} {arm}; log={out / "train.log"}', flush=True)
                    execute(child, out / 'train.log', timing_env(args.gpu, arm, spec, info['uuid']), (fd,))
                    check_exclusive(info['uuid'])
                    values[arm] = reduce_run(out, spec, arm)
                    print(f"[done] {subject} repeat{repeat} {arm} {values[arm]['seconds_per_epoch']:.3f} s/epoch", flush=True)
                write_json(pair_dir / 'result.json', dict(status='complete', subject=subject, repeat=repeat,
                    seed=9100 + repeat, gpu_uuid=info['uuid'], order=order, spec=spec, arms=values))
        report(args.out, args.subjects, write=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--train-entry':
        train_entry()
    else:
        main()
