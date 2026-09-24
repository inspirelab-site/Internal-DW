"""Known-SNR reproduction: seed-0 selection, three-seed testing, and frozen probes."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
CONFIG = ROOT/'configs/reproduce/known_snr.json'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run_dir(root, arm, value, seed):
    return root/'training'/arm/format(value, 'g').replace('.', 'p')/f'seed{seed}'


def settings(config, arm, value, seed):
    args = dict(config['training'], seed=seed)
    if arm == 'clip':
        args['grad_clip'] = value
    elif arm == 'jreg':
        args['forward_jacobian_lambda'] = value
    elif arm == 'risk_reference':
        args['num_workers'] = config['probes']['reference_workers']
    elif arm == 'internal_dw':
        args.update(config['dw'])
        args.update(resgrad_routing=True, resgrad_policy='dualwiener',
                    resgrad_block_gate=0.0, recurrent_grad_checkpoint=False)
    elif arm not in ('full_bptt', 'tbptt'):
        raise ValueError('Unknown method: ' + arm)
    return args


def saved_loss(checkpoint):
    import torch
    blob = torch.load(checkpoint, map_location='cpu', weights_only=False)
    value = float(blob['metrics']['val/loss'])
    if not math.isfinite(value):
        raise ValueError('Nonfinite training val/loss')
    return value


def select(config, root):
    if config['selection_metric'] != 'val/loss' or config['selection_seed'] != 0:
        raise ValueError('Select on saved seed-0 training val/loss only')
    selected, scores = {}, {}
    for arm, grid in config['grids'].items():
        losses = []
        for value in grid:
            folder = run_dir(root, arm, value, 0)
            if not (folder/'train.complete').exists():
                raise ValueError('Candidate training incomplete: ' + str(folder))
            losses.append(saved_loss(folder/'best.pth'))
        selected[arm] = grid[min(range(len(grid)), key=lambda i: losses[i])]
        scores[arm] = dict(grid=grid, val_losses=losses)
    write(root/'selection.json', dict(metric='val/loss', seed=0, selected=selected,
                                     candidates=scores))
    return selected


def environment(gpu):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(('DUAL_WIENER_', 'TORCHELASTIC_', 'RESGRAD_', 'GLOBAL_WIENER_')) or key in (
                'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
            env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1',
               PYTHONPATH=str(ROOT/'src') + os.pathsep + str(ROOT))
    return env


def call(command, opt, env=None):
    print('[run]', ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=ROOT,
                   env=env or environment(opt.gpu), check=True)


def prepare(opt, config):
    from types import SimpleNamespace
    from internal_dw.datasets.synthetic_memory import build_known_snr_ar_splits
    args = SimpleNamespace(**config['training'], data_path=str(opt.data), seed=0)
    build_known_snr_ar_splits(args)
    archives = list(opt.data.glob('known_snr_ar*.npz'))
    if len(archives) != 1:
        raise ValueError('Use a data directory containing one Known-SNR archive')
    return archives[0]


def train_one(opt, config):
    from main import build_parser, worker
    import torch
    folder = run_dir(opt.root, opt.arm, opt.value, opt.seed)
    if (folder/'train.complete').exists():
        return
    args = build_parser().parse_args(['--data_path', str(opt.data)])
    vars(args).update(settings(config, opt.arm, opt.value, opt.seed))
    args.mode, args.save_root, args.resume = 'train', str(folder), 'auto'
    if opt.arm == 'tbptt':
        from scripts.train.known_snr_tbptt import install, validate_args, IMPLEMENTATION
        args.known_snr_tbptt_period = int(opt.value)
        args.tbptt_implementation = IMPLEMENTATION
        validate_args(args)
        install(args.known_snr_tbptt_period)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('One visible CUDA GPU is required per run')
    worker(0, args, 1)
    if not (folder/'best.pth').exists():
        raise RuntimeError('Training did not produce best.pth')
    (folder/'train.complete').touch()


def train(opt, config, arm, value, seed, archive):
    folder = run_dir(opt.root, arm, value, seed)
    if (folder/'train.complete').exists():
        print('[completed]', folder, flush=True)
        return folder/'best.pth'
    env = environment(opt.gpu)
    if arm == 'internal_dw':
        prior = opt.root/'priors'/f'seed{seed}.npz'
        call([sys.executable, 'scripts/data/prepare_known_snr_diagonal_ar1.py',
              '--data', archive, '--output', prior, '--max-horizon', 32,
              '--split-seed', seed, '--train-ratio', .7, '--val-ratio', .15], opt)
        env.update(DUAL_WIENER_INNOVATION_FILE=str(prior),
                   DUAL_WIENER_INNOVATION_KEY='innovation_variance')
    call([sys.executable, __file__, 'train-one', '--root', opt.root, '--data', opt.data,
          '--config', opt.config, '--arm', arm, '--value', value, '--seed', seed], opt, env)
    return folder/'best.pth'


def forecast(opt, config, archive):
    selected = select(config, opt.root)
    results = {}
    for arm, value in dict(full_bptt=0, internal_dw=0, **selected).items():
        scores = []
        for seed in config['seeds']:
            checkpoint = run_dir(opt.root, arm, value, seed)/'best.pth'
            if opt.mode in ('all', 'train'):
                train(opt, config, arm, value, seed, archive)
            if opt.mode == 'train':
                continue
            out = opt.root/'test'/arm/f'seed{seed}.json'
            # Same evaluator and weighting as the other forecasting datasets.
            call([sys.executable, 'scripts/evaluate/evaluate_dense_multistart_rel_l2.py',
                  '--ckpt', checkpoint, '--data-path', opt.data, '--out', out,
                  '--split', 'test', '--gpu', 0, '--train-horizon', 32,
                  '--max-horizon', config['evaluation']['horizon'],
                  '--origin-stride', config['evaluation']['origin_stride'],
                  '--max-origins-per-item', config['evaluation']['max_origins'],
                  '--origin-batch', 16, '--num-workers', 0, '--method-label', arm], opt)
            scores.append(read(out)['primary_metric']['value'])
            print('[test-json]', out, flush=True)
        if scores:
            results[arm] = dict(per_seed=scores, mean=statistics.mean(scores),
                                sample_sd=statistics.stdev(scores))
    if results:
        write(opt.root/'forecast_summary.json',
              dict(metric='mean_relative_l2', horizons=[1, 48], methods=results,
                   selected=selected, seeds=config['seeds']))
        print('[forecast-summary]', opt.root/'forecast_summary.json')


def probes(opt, config, archive):
    # The published frozen probes use workers=4 reference training;
    # matched forecasting comparisons above all use workers=0.
    for seed in config['seeds']:
        train(opt, config, 'risk_reference', 0, seed, archive)
    if opt.mode in ('all', 'probes'):
        oracle = opt.root/'oracle.npz'
        call([sys.executable, 'scripts/data/prepare_known_snr_oracle.py',
              '--data', archive, '--output', oracle, '--max-horizon', 32], opt)
        env = environment(opt.gpu)
        env.update(GPUS=str(opt.gpu), SEED='0', REPETITIONS=str(config['probes']['repetitions']),
                   EVAL_NOISE_DRAWS=str(config['probes']['draws']),
                   CKPT=str(run_dir(opt.root, 'risk_reference', 0, 0)/'best.pth'),
                   DATA=str(archive), ORACLE_FILE=str(oracle), ROOT=str(opt.root/'profiles'),
                   LOG_DIR=str(opt.root/'profiles/logs'))
        call(['bash', 'scripts/probes/run_known_snr_exact_panels12.sh'], opt, env)
    risk_root = opt.root/'risk'
    entry = 'scripts/probes/probe_known_snr_online_risk.py'
    for seed in config['seeds']:
        checkpoint = run_dir(opt.root, 'risk_reference', 0, seed)/'best.pth'
        prior = opt.root/'risk_priors'/f'seed{seed}.npz'
        call([sys.executable, 'scripts/data/prepare_known_snr_diagonal_ar1.py',
              '--data', archive, '--output', prior, '--max-horizon', 32,
              '--split-seed', seed, '--train-ratio', .7, '--val-ratio', .15], opt)
        env = environment(opt.gpu)
        env.update(DUAL_WIENER_INNOVATION_FILE=str(prior),
                   DUAL_WIENER_INNOVATION_KEY='innovation_variance')
        call([sys.executable, entry, 'collect', '--ckpt', checkpoint,
              '--data', opt.data, '--root', risk_root, '--seed', seed,
              '--epochs', config['probes']['epochs']], opt, env)
        for snapshot in sorted((risk_root/f'seed{seed}/snapshots').glob('*.pt')):
            if snapshot.with_suffix('.risk.json').exists():
                continue
            call([sys.executable, entry, 'evaluate', '--snapshot', snapshot,
                  '--root', risk_root, '--draws', config['probes']['draws']], opt, env)
    call([sys.executable, entry, 'summary', '--root', risk_root], opt)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['all', 'prepare', 'screen', 'train', 'test', 'probes', 'risk', 'train-one'])
    p.add_argument('--root', type=Path, default=ROOT/'experiments/known_snr')
    p.add_argument('--data', type=Path, default=ROOT/'data/known_snr')
    p.add_argument('--config', type=Path, default=CONFIG)
    p.add_argument('--gpu', default='0')
    p.add_argument('--arm', choices=['full_bptt', 'internal_dw', 'clip', 'jreg', 'tbptt', 'risk_reference'])
    p.add_argument('--value', type=float, default=0)
    p.add_argument('--seed', type=int, default=0)
    opt = p.parse_args()
    opt.root, opt.data, opt.config = opt.root.resolve(), opt.data.resolve(), opt.config.resolve()
    config = read(opt.config)
    if opt.mode == 'train-one':
        train_one(opt, config)
        return
    archive = prepare(opt, config)
    if opt.mode in ('all', 'screen'):
        for arm, grid in config['grids'].items():
            for value in grid:
                train(opt, config, arm, value, 0, archive)
        select(config, opt.root)
    if opt.mode in ('all', 'train', 'test'):
        forecast(opt, config, archive)
    if opt.mode in ('all', 'probes', 'risk'):
        probes(opt, config, archive)


if __name__ == '__main__':
    main()
