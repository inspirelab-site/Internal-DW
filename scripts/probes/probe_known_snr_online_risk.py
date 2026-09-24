"""Measure local route risk on a frozen Full-BPTT training stream."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT), str(Path(__file__).resolve().parent)]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def frozen_pass(opt):

    import torch
    from main import build_parser
    from internal_dw.datasets import build_dataloaders
    from internal_dw.models import build_model
    from internal_dw.utils import seed_everything
    from internal_dw.training.trainer import unpack_batch
    from known_snr_online_core import frozen_step,cpu_state,rng_state,assert_equal
    folder=opt.output or opt.root/f'seed{opt.seed}'
    if (folder/'complete.json').exists():
        print('[skip complete frozen pass]',folder);return
    if (folder/'snapshots').exists():
        raise ValueError('Partial pass exists; use a fresh --root to avoid mixing gain histories')
    blob=torch.load(opt.ckpt,map_location='cpu',weights_only=False)
    previous=blob['args'] if isinstance(blob['args'],dict) else vars(blob['args'])
    if previous.get('resgrad_routing') or previous['seed']!=opt.seed:
        raise ValueError('Expected seed-matched Full-BPTT checkpoint')
    args=build_parser().parse_args(['--data_path',str(opt.data)])
    vars(args).update(previous)
    args.data_path=str(opt.data)
    args.dataset='known_snr_ar';args.seed=opt.seed;args.num_workers=4;args.local_batch_size=4
    args.grad_accum_steps=8;args.snr_ar_coefficients=previous['snr_ar_coefficients']
    args.mamba_train_starts_per_sequence=1;args.mamba_loss_type='mse';args.mamba_loss_decay=1.0
    args.resgrad_routing=True;args.resgrad_policy='dualwiener';args.recurrent_grad_checkpoint=False
    args.dual_wiener_ema=.95;args.dual_wiener_residual_ema=.99
    args.dual_wiener_warmup_batches=8;args.dual_wiener_probe_every=4;args.dual_wiener_min_probes=1
    args.dual_wiener_noise_model='diagonal_gaussian';args.dual_wiener_max_horizon=32
    args.dataset_has_external_input=False;args.dataset_task_type='sequence_vector'
    args.forward_jacobian_lambda=0;args.bptt_detach_period=0;args.bptt_loss=False
    seed_everything(opt.seed)
    loader,_,_=build_dataloaders(args,rank=0,world_size=1)
    model=build_model(args,rank=0)
    missing,unexpected=model.load_state_dict(blob['model'],strict=False)
    if unexpected or any(not k.startswith('dual_wiener.') for k in missing):
        raise ValueError('Checkpoint adapter mismatch: '+str((missing,unexpected)))
    initial={k:v.detach().cpu().clone() for k,v in model.named_parameters()}
    original_buffers={k:v.detach().cpu().clone() for k,v in model.named_buffers()
                      if not k.startswith('dual_wiener.')}
    model.train();folder.mkdir(parents=True,exist_ok=True)
    losses=[];first=12
    for epoch in range(1,opt.epochs+1):
        for batch in loader:
            state,stim,_,_=unpack_batch(batch)
            state=state.cuda().float()
            if stim is None: stim=torch.zeros((*state.shape[:2],1),device=state.device)
            else: stim=stim.cuda().float()
            seen=int(model.dual_wiener.seen_batches.item())
            # One warm-up example retained separately; EVERY batch after seen=12 measured.
            path=folder/f'snapshots/batch_{seen:05d}.pt' if not opt.no_capture and (seen==0 or seen>=first) else None
            loss=frozen_step(model,state,stim,args,epoch,path)
            losses.append(loss)
            print(f'[frozen pass] seed={opt.seed} epoch={epoch}/{opt.epochs} batch={seen} loss={loss:.6g} snapshot={path is not None}',flush=True)
    assert_equal(initial,{k:v.detach().cpu() for k,v in model.named_parameters()},'frozen parameters')
    assert_equal(original_buffers,{k:v.detach().cpu() for k,v in model.named_buffers()
                                  if not k.startswith('dual_wiener.')},'non-controller buffers')
    torch.save(dict(model=cpu_state(model),rng=rng_state(),losses=losses),folder/'final_state.pt')
    write(folder/'complete.json',dict(status='complete',seed=opt.seed,epochs=opt.epochs,
        batches=len(losses),parameters_unchanged=True,checkpoint=str(opt.ckpt),
        first_reported_seen_batch=first,optimizer_steps=0))



def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['collect', 'evaluate', 'summary'])
    p.add_argument('--ckpt', type=Path)
    p.add_argument('--data', type=Path)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--draws', type=int, default=64)
    p.add_argument('--snapshot', type=Path)
    opt = p.parse_args()
    opt.root = opt.root.resolve()
    opt.output = None
    opt.no_capture = False
    if opt.epochs < 1 or opt.draws < 2:
        p.error('epochs >= 1 and draws >= 2 required')
    if opt.mode == 'collect':
        if opt.ckpt is None or opt.data is None:
            p.error('collect requires --ckpt and --data')
        frozen_pass(opt)
    elif opt.mode == 'evaluate':
        from known_snr_online_core import replay
        from internal_dw.models import build_model
        if opt.snapshot is None:
            p.error('evaluate requires --snapshot')
        result = replay(opt.snapshot, opt.draws, lambda args: build_model(args, rank=0))
        write(opt.snapshot.with_suffix('.risk.json'), result)
    else:
        import statistics
        per_seed = []
        for seed in (0, 1, 2):
            folder = opt.root/f'seed{seed}'
            if not (folder/'complete.json').exists():
                raise ValueError('Incomplete collection: ' + str(folder))
            rows = [json.loads(p.with_suffix('.risk.json').read_text())
                    for p in sorted((folder/'snapshots').glob('*.pt'))]
            rows = [r for r in rows if r['post_warmup']]
            if not rows:
                raise ValueError('No post-warmup measurements')
            totals = {m: sum(r['methods'][m]['sum_risk'] for r in rows)
                      for m in rows[0]['methods']}
            per_seed.append(dict(seed=seed, batches=len(rows),
                risk_over_full={m: v/totals['full_bptt'] for m,v in totals.items()}))
        methods = {m: dict(mean=statistics.mean(r['risk_over_full'][m] for r in per_seed),
                    sample_sd=statistics.stdev(r['risk_over_full'][m] for r in per_seed))
                   for m in per_seed[0]['risk_over_full']}
        write(opt.root/'summary.json', dict(status='complete', methods=methods,
            per_seed=per_seed,
            protocol='Frozen BPTT weights; lagged gains within the training stream',
            aggregation='Within seed: ratio of summed local risks; across seeds: mean and sample SD'))
        print('[risk-summary]', opt.root/'summary.json')


if __name__ == '__main__':
    main()

