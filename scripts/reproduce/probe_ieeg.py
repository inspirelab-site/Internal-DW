"""Run the existing regime/utility/noise probes on each prepared iEEG subject."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.reproduce.train_test_ieeg import SUBJECTS, clean_env, execute, lock, result_root
from scripts.data.prepare_ieeg import write_json


def probe_root():
    return Path(os.environ.get('IEEG_PROBE_ROOT', ROOT / 'probe_outputs/ieeg_cohort_v1')).resolve()


def aggregate(kind, out):
    if kind == 'noise':
        for snr in ('4', '1', '0.25'):
            paths = [out / f'{s}_snr{snr.replace(".", "p")}.json' for s in SUBJECTS]
            records = [json.loads(p.read_text()) for p in paths]
            if any(r['requested_snr'] != float(snr) for r in records):
                raise ValueError('Mismatched noise levels')
            gain = np.asarray([[r['summary']['prior'][k] for k in ('alpha_mean', 'm_mean')] for r in records])
            if not np.isfinite(gain).all() or np.any((gain < 0) | (gain > 1)):
                raise ValueError('Invalid noise gains')
            write_json(out / f'snr{snr.replace(".", "p")}.json', dict(subjects=SUBJECTS,
                requested_snr=float(snr), summary=dict(prior=dict(zip(('alpha_mean', 'm_mean'), gain.mean(0).tolist()))),
                aggregation='Equal-participant means of within-participant route gains', sources=list(map(str, paths))))
        return
    paths = [out / f'{s}.json' for s in SUBJECTS]
    records = [json.loads(p.read_text()) for p in paths]
    payload = dict(subjects=SUBJECTS, sources=list(map(str, paths)))
    if kind == 'regime':
        fields = ('mean_long_drive_value', 'mean_long_history_value', 'mean_long_null_history_value')
        summary = {}
        for readout in ('linear', 'nonlinear'):
            values = np.asarray([[r['long_horizon_summary'][readout][f] for f in fields] for r in records])
            if not np.isfinite(values).all(): raise ValueError('Nonfinite regime estimates')
            summary[readout] = dict(zip(fields, values.mean(0).tolist()))
        payload.update(long_horizon_summary=summary, aggregation='Equal-participant raw means; clipping is display-only')
    else:
        summary = {}
        for key in ('full_amplitude_H1', 'full_window_utility'):
            values = np.asarray([r['summary'][key]['median'] for r in records])
            if values.shape != (16, 64) or not np.isfinite(values).all():
                raise ValueError('Require all 16 participant curves and 64 horizons')
            summary[key] = {name: np.quantile(values, q, axis=0).tolist()
                            for name, q in [('q10', .1), ('q25', .25), ('median', .5), ('q75', .75), ('q90', .9)]}
        payload.update(horizons=list(range(1, 65)), summary=summary,
                       aggregation='Quantiles across 16 participant median curves; not training-seed SD')
    write_json(out / 'cohort.json', payload)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind', choices=['regime', 'utility', 'noise'])
    p.add_argument('--subjects', nargs='+', choices=SUBJECTS, default=SUBJECTS)
    p.add_argument('--gpu', default='0')
    p.add_argument('--prepared', type=Path, default=os.environ.get('IEEG_PREPARED_ROOT', ROOT / 'probe_inputs/ieeg_cohort_v1'))
    p.add_argument('--runs', type=Path, default=result_root())
    p.add_argument('--out', type=Path, default=probe_root())
    p.add_argument('--summarize', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(); out = args.out.resolve() / args.kind
    if args.summarize: aggregate(args.kind, out); return
    env = clean_env(args.gpu)
    for subject in args.subjects:
        archive = (args.prepared / 'prepared' / (subject + '.npz')).resolve()
        base = (args.runs / subject).resolve()
        common = [sys.executable, '-u']
        if args.kind == 'regime':
            command = common + [str(ROOT / 'scripts/probes/probe_cross_dataset_drive_history.py'),
                '--input-npz', str(archive), '--input-dataset-name', 'ieeg_fif_visual_' + subject,
                '--label', subject, '--horizons', '1,2,4,8,16,24,32,48,64',
                '--history-windows', '1,2,4,8,16', '--long-horizon-min', '8',
                '--max-coordinates', '256', '--pca-rank', '32', '--pca-frames', '2048', '--seed', '0', '--device', 'cuda']
            jobs = [(command, out / (subject + '.json'), '--output')]
        elif args.kind == 'utility':
            command = common + [str(ROOT / 'scripts/probes/probe_heldout_delayed_gradient_utility.py'),
                '--ckpt', str(base / 'full_bptt/seed0/best.pth'), '--K', '64', '--num_pairs', '8',
                '--finite_pairs', '1', '--finite_horizons', '1,8,16,32,64',
                '--relative_radii', '2.5e-7,5e-7,1e-6', '--coord_subsample', '250000',
                '--calibration_split', 'train', '--evaluation_split', 'test', '--seed', '0', '--gpu', '0']
            jobs = [(command, out / (subject + '.json'), '--out')]
        else:
            command = common + [str(ROOT / 'scripts/probes/probe_ieeg_prior_noise_response.py'),
                '--prepared-checkpoint', '--ckpt', str(base / 'internal_dw/seed0/best.pth'),
                '--npz', str(archive), '--artifact', str(base / 'noise_templates_K64.npz'),
                '--K', '64', '--burnin', '32', '--batch', '16', '--draws', '4',
                '--dual-wiener-noise-model', 'lagged_residual_bootstrap',
                '--noise-draws', '8', '--seed', '0', '--device', 'cuda']
            jobs = [(command + ['--snr', snr], out / f'{subject}_snr{snr.replace(".", "p")}.json', '--out')
                    for snr in ('4', '1', '0.25')]
        for command, target, flag in jobs:
            if args.dry_run: print(json.dumps(command + [flag, str(target)])); continue
            with lock(target.with_suffix('.lock')) as fd:
                if not target.exists():
                    partial = target.with_suffix('.partial.json')
                    execute(command + [flag, str(partial)], target.with_suffix('.log'), env, (fd,))
                    json.loads(partial.read_text()); partial.replace(target)
                print('[probe]', target)
    if not args.dry_run and set(args.subjects) == set(SUBJECTS): aggregate(args.kind, out)


if __name__ == '__main__': main()
