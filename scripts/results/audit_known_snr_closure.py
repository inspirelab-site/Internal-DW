"""Read-only audit of a reproduced closure; never changes curves or checkpoints."""
import argparse
import hashlib
import io
import json
from pathlib import Path


EXPECTED = dict(seed=0, dataset='known_snr_ar', model_name='official_mamba_state',
                simple_hidden_dim=128, simple_depth=4, mamba_bptt_horizon=32,
                mamba_burnin=32, mamba_train_starts_per_sequence=1, mamba_loss_type='mse',
                window_size=16, local_batch_size=4, grad_accum_steps=8, num_workers=4,
                base_lr=0.0001, weight_decay=0.0001, grad_clip=1.0, num_epochs=100,
                early_stop_patience=20, ar_optimizer='adam', ar_scheduler='step',
                snr_ar_dim=8, snr_ar_len=1024, snr_ar_traj=96, snr_ar_seed=0,
                snr_ar_coefficients=[0.995, 0.98, 0.95, 0.9, -0.995, -0.98, -0.95, -0.9])


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def panel_audit(root):
    panel = read(root / 'panels_1_2.json')
    curves = panel['panel_2']
    total = curves['total_prefix_risk_over_exact']['median']
    result = dict(protocol=panel['protocol'], first_total_risk=total[0],
                  full_total_risk=total[-1],
                  best_prefix=curves['median_curve_best_prefix_horizon'],
                  first_missing_signal=curves['omitted_signal_bias_over_exact']['median'][0],
                  first_innovation=curves['innovation_risk_over_exact']['median'][0],
                  original_paper_reference=dict(first_total_risk=0.515726459583593,
                                                best_prefix=24), replicates=[])
    for p in sorted((root / 'panels_1_2_replicates').glob('replicate*.json')):
        data = read(p); profile = data['profile']
        total_raw = profile['prefix_total_risk_to_full_clean_target']
        result['replicates'].append(dict(file=str(p), seed=data.get('seed'),
            batch=data.get('batch'), protocol=data.get('protocol'),
            noise_draws=profile.get('eval_noise_draws'),
            first_risk_raw=total_raw[0], full_risk_denominator=total_raw[-1],
            first_risk_ratio=total_raw[0] / total_raw[-1]))
    route = root / 'route_risk.json'
    if route.is_file():
        result['route_risk_ratios'] = {k: v['sum_mse_over_sum_open']
                                     for k, v in read(route)['methods'].items()}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('probe_outputs/known_snr_diagonal_ar_closure/seed0'))
    p.add_argument('--ckpt', type=Path, default=Path('experiments/known_snr_oracle/known_snr_ar/mixed_K32/ckpt/seed0/best.pth'))
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    result = panel_audit(args.root)
    import torch
    raw = args.ckpt.read_bytes()
    ckpt = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
    actual = ckpt.get('args', {})
    if not isinstance(actual, dict):
        actual = vars(actual)
    result['checkpoint'] = dict(path=str(args.ckpt.resolve()),
        sha256=hashlib.sha256(raw).hexdigest(), epoch=ckpt.get('epoch'),
        best_val=ckpt.get('best_val'), original_paper_epoch=47,
        original_paper_best_val=1.0703514218330383,
        arguments={k: actual.get(k) for k in EXPECTED},
        differing_arguments={k: dict(expected=v, actual=actual.get(k))
                             for k, v in EXPECTED.items() if actual.get(k) != v})
    result['note'] = ('Panel (b) divides every prefix risk by its own full-horizon risk, '
                      'then takes the median over repetitions. The first value is not fixed to 0.5. '
                      'Audit uses no test scores for hyperparameter selection.')
    output = args.out or args.root / 'reproduction_audit.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    print('[out]', output)


if __name__ == '__main__':
    main()
