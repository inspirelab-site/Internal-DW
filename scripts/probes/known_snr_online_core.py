"""Frozen training-stream collection and fully-open local route-risk measurement."""
import copy
import random
from pathlib import Path
import sys
from contextlib import contextmanager

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts/probes')]
from internal_dw.training.ar_losses import compute_autoregressive_one_step_loss as compute
from probe_wiener_oracle import RouteCapture, push, covector_loss
from known_snr_route_ops import process_tensors, sample_process, route_moments, solve_gain


def cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(value):
    random.setstate(value['python']);np.random.set_state(value['numpy']);torch.set_rng_state(value['torch'])
    if value['cuda']: torch.cuda.set_rng_state_all(value['cuda'])


def assert_equal(a,b,path=''):
    if isinstance(a,torch.Tensor): equal=isinstance(b,torch.Tensor) and torch.equal(a,b)
    elif isinstance(a,np.ndarray): equal=np.array_equal(a,b)
    elif isinstance(a,dict):
        if a.keys()!=b.keys(): raise AssertionError('Keys differ: '+path)
        for k in a: assert_equal(a[k],b[k],path+'/'+str(k))
        return
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b): raise AssertionError(path)
        for k,(x,y) in enumerate(zip(a,b)): assert_equal(x,y,path+'/'+str(k))
        return
    else: equal=a==b
    if not equal: raise AssertionError('Values differ: '+path)


@contextmanager
def watch(raw, diagnostic=False):
    """Recording never changes training calculations; diagnostic replay is isolated."""
    values=dict(predictions=[],initial=[])
    step,probe=raw.step,raw.dual_wiener_probe_terms
    owned=('step' in raw.__dict__,'dual_wiener_probe_terms' in raw.__dict__)
    def step_hook(hidden,x,*args,**kwargs):
        if kwargs.get('horizon_index',-1)==0: values['initial'].append(x.detach().clone())
        return step(hidden,x,*args,**kwargs)
    def probe_hook(prediction,target,horizon):
        values['predictions'].append(prediction)
        # The replay forces route capture even on non-calibration batches.
        # Do not generate extra calibration noise or update replay statistics.
        return (None,None) if diagnostic else probe(prediction,target,horizon)
    raw.step,raw.dual_wiener_probe_terms=step_hook,probe_hook
    try: yield values
    finally:
        if owned[0]: raw.step=step
        else: del raw.step
        if owned[1]: raw.dual_wiener_probe_terms=probe
        else: del raw.dual_wiener_probe_terms


def frozen_step(model,state,stim,args,epoch,snapshot=None):
    model.zero_grad(set_to_none=True)
    model.dual_wiener_begin_batch()
    dw=model.dual_wiener
    packet=None
    if snapshot:
        packet=dict(model=cpu_state(model),args=copy.deepcopy(vars(args)),
            state=state.detach().cpu().clone(),stim=stim.detach().cpu().clone(),rng=rng_state(),
            epoch=epoch,seen_batches=int(dw.seen_batches.item()),
            gains=dw.coefficients.detach().cpu().clone())
        with watch(model) as values: loss,logs=compute(model,state,stim,args,epoch=epoch)
        if len(values['initial'])!=1 or len(values['predictions'])!=args.mamba_bptt_horizon:
            raise ValueError('Expected exactly one closed-loop rollout')
        packet.update(loss=loss.detach().cpu(),initial=values['initial'][0].cpu(),
                      predictions=torch.stack(values['predictions']).detach().cpu())
    else:
        loss,logs=compute(model,state,stim,args,epoch=epoch)
    # EXACT same ordering as the public trainer. No optimizer exists here.
    model.dual_wiener_calibrate()
    (loss/args.grad_accum_steps).backward()
    model.dual_wiener_end_batch()
    if packet:
        snapshot=Path(snapshot);snapshot.parent.mkdir(parents=True,exist_ok=True)
        if snapshot.exists(): raise ValueError('Snapshot already exists: '+str(snapshot))
        temporary=snapshot.with_suffix('.partial');torch.save(packet,temporary);temporary.replace(snapshot)
    model.zero_grad(set_to_none=True)
    return float(loss.detach().cpu())


def _risk(P, R, weight):
    weight = np.asarray(weight, dtype=np.float64)
    delta = weight - 1.0
    return float(delta @ P @ delta + weight @ R @ weight)


def measure(model, values, coefficients, draws, noise_seed):
    dw = model.dual_wiener
    predictions = values['predictions']
    K, depth = len(predictions), dw.depth
    applied = dw.coefficients[:K].detach().cpu().numpy().copy()
    initial = values['initial'][0]
    mean = initial
    a = torch.as_tensor(coefficients, device=initial.device, dtype=initial.dtype)
    signal = []
    for prediction in predictions:
        mean = mean * a
        signal.append((prediction - mean).detach())
    transition, factor = process_tensors(np.diag(coefficients),
        np.diag(1-np.asarray(coefficients)**2), str(initial.device), initial.dtype)
    capture = RouteCapture(dw)
    saved_mode = dw._mode

    def moments(vectors):
        pairs = push(covector_loss(predictions, vectors), dw._root_refs, capture, retain=True)
        if set(pairs) != set(range(K*depth)):
            raise ValueError('Incomplete fully-open route capture')
        return route_moments(pairs)

    def noise_moments(seed):
        generator = torch.Generator(device='cpu').manual_seed(seed)
        total = {}
        for _ in range(draws):
            white = torch.randn((K,*predictions[0].shape), generator=generator).to(predictions[0])
            for route, value in moments(sample_process(white, transition, factor)).items():
                total[route] = total.get(route, 0) + value/draws
        return total

    try:
        # Identical open-path convention to the original route-risk probe.
        # This mode opens BOTH token routes and recurrent-state routes.
        # Actual applied coefficients are saved above and never overwritten.
        dw._mode = 'total'
        P = moments(signal)
        R_fit = noise_moments(noise_seed)
        R_eval = noise_moments(noise_seed+1000003)
        oracle = np.empty_like(applied)
        for route in P:
            h,l = divmod(route,depth)
            oracle[h,l] = solve_gain(P[route]+R_fit[route], R_fit[route])
        arms = dict(full_bptt=np.ones_like(applied), online_dw=applied,
                    misplaced=np.roll(applied,K//2,axis=0), local_oracle=oracle)
        results = {}
        for name, gains in arms.items():
            rows = []
            for route in sorted(P):
                h,l = divmod(route,depth)
                w = gains[h,l]; delta = w-1
                rows.append(dict(horizon=h+1,layer=l+1,
                    bias_mse=float(delta@P[route]@delta), noise_mse=float(w@R_eval[route]@w),
                    risk=_risk(P[route], R_eval[route], w)))
            results[name] = dict(sum_risk=sum(r['risk'] for r in rows), routes=rows)
        denominator = results['full_bptt']['sum_risk']
        if denominator <= 0 or not np.isfinite(denominator):
            raise ValueError('Invalid fully-open reference risk')
        for value in results.values():
            value['risk_over_full'] = value['sum_risk']/denominator
            if not np.isfinite(value['risk_over_full']) or value['sum_risk'] < -1e-12:
                raise ValueError('Invalid local risk')
        return dict(methods=results, applied_gains=applied.tolist(),
            local_oracle_gains=oracle.tolist(),
            route_moments=[dict(horizon=r//depth+1,layer=r%depth+1,
                P=P[r].tolist(),R_fit=R_fit[r].tolist(),R_eval=R_eval[r].tolist()) for r in sorted(P)])
    finally:
        capture.close()
        dw._mode = saved_mode


def replay(path,draws,build_model):
    from types import SimpleNamespace
    packet=torch.load(path,map_location='cpu',weights_only=False);args=SimpleNamespace(**packet['args'])
    model=build_model(args);model.load_state_dict(packet['model'],strict=True);model.train()
    device=next(model.parameters()).device
    if float(getattr(args,'simple_dropout',0))!=0: raise ValueError('Replay currently requires dropout=0')
    model.dual_wiener._external_needs_noise_reset=False
    model.dual_wiener_begin_batch()
    model.dual_wiener._collecting=True  # only in the isolated diagnostic process
    restore_rng(packet['rng'])
    with watch(model,diagnostic=True) as values:
        loss,_=compute(model,packet['state'].to(device),packet['stim'].to(device),args,epoch=packet['epoch'])
    assert_equal(torch.stack(values['predictions']).detach().cpu(),packet['predictions'],'predictions')
    assert_equal(loss.detach().cpu(),packet['loss'],'loss')
    assert_equal(values['initial'][0].cpu(),packet['initial'],'initial')
    assert_equal(model.dual_wiener.coefficients.cpu(),packet['gains'],'actual applied gains')
    before=cpu_state(model)
    result=measure(model,values,args.snr_ar_coefficients,draws,900001+args.seed*10000019+packet['seen_batches']*17)
    assert_equal(before,cpu_state(model),'diagnostic state')
    result.update(status='complete',seed=args.seed,epoch=packet['epoch'],seen_batches=packet['seen_batches'],
        post_warmup=packet['seen_batches']>=args.dual_wiener_warmup_batches+args.dual_wiener_probe_every,
        forward_replay_bitwise_equal=True,draws_fit=draws,draws_eval=draws,
        scope='Local token-route risk on the fully-open graph with lagged training-stream gains',
        oracle_caveat='Current-batch local oracle; independent fitting and evaluation Monte Carlo draws')
    return result
