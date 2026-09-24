"""Small CPU checks for the public Known-SNR workflow."""
import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.reproduce import train_test_known_snr as run
from scripts.probes import known_snr_online_core as risk
from scripts.train import known_snr_tbptt as tbptt
from internal_dw.models.official_state_mamba import OfficialStateMambaARModel

CONFIG = run.read(run.CONFIG)


def setup():
    torch.set_num_threads(1)
    model = OfficialStateMambaARModel(state_dim=2, input_dim=0, hidden_dim=8, depth=2,
        has_external_input=False, mamba_d_state=2, mamba_d_conv=2, mamba_expand=1,
        resgrad_routing=True, resgrad_policy='dualwiener', dual_wiener_warmup_batches=1,
        dual_wiener_probe_every=2, dual_wiener_max_horizon=4)
    args = SimpleNamespace(mamba_bptt_horizon=4, mamba_burnin=3, mamba_train_starts_per_sequence=1,
        mamba_loss_type='mse', mamba_loss_decay=1, ar_train_random_starts=True,
        recurrent_grad_checkpoint=False, ar_loss='mse', ar_one_step_lambda=1,
        bptt_loss=False, compact_recurrent_logging=True, forward_jacobian_lambda=0,
        seed=0, snr_ar_coefficients=[.8,-.7], grad_accum_steps=2,
        dual_wiener_warmup_batches=1, dual_wiener_probe_every=2)
    return model, args


def test_selection_uses_saved_training_loss_not_test(tmp_path):
    for arm, grid in CONFIG['grids'].items():
        for i, value in enumerate(grid):
            folder = run.run_dir(tmp_path, arm, value, 0)
            folder.mkdir(parents=True)
            torch.save({'metrics': {'val/loss': [3., 1., 2.][i]}}, folder/'best.pth')
            (folder/'train.complete').touch()
            (folder/'test.json').write_text('not read for selection')
    result = run.select(CONFIG, tmp_path)
    assert result == {arm: grid[1] for arm, grid in CONFIG['grids'].items()}
    run.run_dir(tmp_path, 'tbptt', 16, 0).joinpath('train.complete').unlink()
    with pytest.raises(ValueError, match='incomplete'):
        run.select(CONFIG, tmp_path)


def test_methods_share_training_geometry():
    for arm, value in [('full_bptt',0),('internal_dw',0),('clip',.3),('jreg',.01),('tbptt',4)]:
        args = run.settings(CONFIG, arm, value, 2)
        assert (args['seed'], args['num_workers'], args['local_batch_size'], args['grad_accum_steps']) == (2,0,4,8)
        assert args['mamba_bptt_horizon'] == 32
    assert run.settings(CONFIG,'risk_reference',0,0)['num_workers'] == 4


def test_prepare_uses_public_stationary_generator(tmp_path):
    archive = run.prepare(SimpleNamespace(data=tmp_path), CONFIG)
    with np.load(archive) as data:
        assert data['trajs'].shape == (96, 1024, 8)
        np.testing.assert_array_equal(data['coefficients'], CONFIG['training']['snr_ar_coefficients'])
    assert run.prepare(SimpleNamespace(data=tmp_path), CONFIG) == archive


@pytest.mark.parametrize('period', [4, 8, 16])
@pytest.mark.parametrize('checkpointed', [False, True])
def test_tbptt_forward_and_gradients_match_explicit_cuts(period, checkpointed):
    from torch.utils.checkpoint import checkpoint
    torch.set_num_threads(1)
    torch.manual_seed(47)
    kwargs = dict(state_dim=2,input_dim=0,hidden_dim=8,depth=2,has_external_input=False,
        mamba_d_state=2,mamba_d_conv=2,mamba_expand=1,resgrad_routing=False,resgrad_policy='all')
    model = OfficialStateMambaARModel(**kwargs)
    initial = torch.randn(2,2)
    original = OfficialStateMambaARModel.step
    def rollout(explicit):
        model.zero_grad(set_to_none=True)
        origin = initial.clone().requires_grad_(True)
        x = origin
        h = model.init_state(2,x.device,x.dtype)
        outputs = []
        for k in range(33):
            def step(inp,*flat,index=k):
                state = tuple((flat[i],flat[i+1]) for i in range(0,len(flat),2))
                cut = explicit and index > 0 and index % period == 0
                if cut:
                    inp = inp.detach()
                pred,state = model.step(state,inp,horizon_index=index,total_horizon=33)
                if cut:
                    state = tuple((a.detach(),b.detach()) for a,b in state)
                return (pred,) + tuple(v for pair in state for v in pair)
            flat = tuple(v for pair in h for v in pair)
            result = checkpoint(step,x,*flat,use_reentrant=False) if checkpointed else step(x,*flat)
            x = result[0]
            h = tuple((result[i],result[i+1]) for i in range(1,len(result),2))
            outputs.append(x)
        values = torch.stack(outputs)
        (values.square()*torch.linspace(.2,1.7,33).view(33,1,1)).mean().backward()
        return (values.detach(), origin.grad.clone(),
                [p.grad.clone() if p.grad is not None else None for p in model.parameters()])
    try:
        reference = rollout(True)
        tbptt.install(period)
        actual = rollout(False)
    finally:
        tbptt.install(0)
        OfficialStateMambaARModel.step = original
    torch.testing.assert_close(actual[0], reference[0])
    torch.testing.assert_close(actual[1], reference[1])
    for a,b in zip(actual[2],reference[2]):
        if a is None or b is None:
            assert a is b
        else:
            torch.testing.assert_close(a,b)


def test_forecast_calls_shared_evaluator(tmp_path, monkeypatch):
    opt = SimpleNamespace(root=tmp_path, data=tmp_path/'data', mode='test')
    monkeypatch.setattr(run, 'select', lambda *a: dict(clip=.1,jreg=.1,tbptt=4))
    commands = []
    def call(command, *args):
        commands.append(command)
        assert command[1] == 'scripts/evaluate/evaluate_dense_multistart_rel_l2.py'
        assert command[command.index('--max-horizon')+1] == 48
        assert command[command.index('--split')+1] == 'test'
        run.write(command[command.index('--out')+1], dict(primary_metric=dict(value=1.)))
    monkeypatch.setattr(run, 'call', call)
    run.forecast(opt, CONFIG, tmp_path/'data.npz')
    assert len(commands) == 15
    assert run.read(tmp_path/'forecast_summary.json')['methods']['full_bptt']['sample_sd'] == 0


@pytest.mark.parametrize('period', [4,8,16])
def test_tbptt_cut_location_and_reset(period, monkeypatch):
    original_step = OfficialStateMambaARModel.step
    def step(self, h, x, stim=None, **kw):
        return x*2, tuple((a*2,b*2) for a,b in h)
    monkeypatch.setattr(tbptt, '_original_step', step)
    x = torch.ones(1, requires_grad=True)
    h = ((x, x),)
    tbptt.install(period)
    try:
        pred, state = tbptt.truncated_step(None,h,x,horizon_index=period)
        assert not pred.requires_grad
        assert not state[0][0].requires_grad
        pred, state = tbptt.truncated_step(None,h,x,horizon_index=period-1)
        assert pred.requires_grad and state[0][0].requires_grad
    finally:
        tbptt.install(0)
        OfficialStateMambaARModel.step = original_step


def test_capture_keeps_weights_and_rng_unchanged(tmp_path):
    torch.manual_seed(5); random.seed(5); np.random.seed(5)
    model, args = setup()
    state, stim = torch.randn(2,16,2), torch.zeros(2,16,1)
    initial, rng = copy.deepcopy(model), risk.rng_state()
    ends = []
    for capture in (False, True):
        m = copy.deepcopy(initial); risk.restore_rng(rng)
        before = {k:v.detach().clone() for k,v in m.named_parameters()}
        losses = [risk.frozen_step(m,state,stim,args,1,tmp_path/f'batch{i}.pt' if capture else None)
                  for i in range(3)]
        risk.assert_equal(before,{k:v.detach() for k,v in m.named_parameters()})
        ends.append((risk.cpu_state(m),losses,risk.rng_state()))
    risk.assert_equal(*ends)


@pytest.mark.parametrize('nontrivial', [False, True])
def test_local_risk_exact_replay_and_formula(tmp_path, nontrivial):
    torch.manual_seed(42); random.seed(42)
    model,args = setup()
    if nontrivial:
        model.dual_wiener.coefficients.uniform_(.2,.8)
    path = tmp_path/'batch.pt'
    risk.frozen_step(model,torch.randn(2,16,2),torch.zeros(2,16,1),args,1,path)
    row = risk.replay(path,2,lambda args:setup()[0])
    assert row == risk.replay(path,2,lambda args:setup()[0])
    assert row['forward_replay_bitwise_equal']
    assert row['methods']['full_bptt']['risk_over_full'] == 1.
    if not nontrivial:
        assert row['methods']['online_dw'] == row['methods']['full_bptt']
    applied = np.array(row['applied_gains'])
    for moment, measured in zip(row['route_moments'],row['methods']['online_dw']['routes']):
        w = applied[measured['horizon']-1, measured['layer']-1]
        assert measured['risk'] == pytest.approx(risk._risk(np.array(moment['P']),np.array(moment['R_eval']),w), rel=1e-6)


def test_optional_plot_reads_public_summaries(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    monkeypatch.syspath_prepend(str(run.ROOT/'scripts/plotting'))
    from scripts.plotting import plot_known_snr_closure as plot
    def band(values):
        return {key: list(values) for key in ('median','min','max')}
    run.write(tmp_path/'profiles/panels_1_2.json', dict(
        horizon=list(range(1,33)), protocol=dict(repetitions=4),
        panel_1=dict(total_gradient_rms_relative_to_h1=band(np.geomspace(1,100,32)),
                     gradient_snr=band(np.geomspace(20,.2,32))),
        panel_2=dict(omitted_signal_bias_over_exact=band(np.linspace(.5,0,32)),
                     innovation_risk_over_exact=band(np.linspace(0,1,32)),
                     total_prefix_risk_over_exact=band(np.linspace(.5,1,32)),
                     median_curve_best_prefix_horizon=16)))
    run.write(tmp_path/'risk/summary.json', dict(methods={
        key: dict(mean=value,sample_sd=.02) for key,value in
        [('full_bptt',1),('misplaced',.3),('online_dw',.2),('local_oracle',.25)]}))
    run.write(tmp_path/'forecast_summary.json', dict(methods={
        key: dict(mean=value,sample_sd=.01) for key,value in
        [('full_bptt',1.05),('clip',1.04),('jreg',1.04),('tbptt',.99),('internal_dw',.98)]}))
    monkeypatch.setattr(plot.sys if hasattr(plot,'sys') else __import__('sys'), 'argv',
        ['plot', '--closure-root', str(tmp_path), '--out', str(tmp_path/'figure')])
    plot.main()
    assert (tmp_path/'figure.pdf').stat().st_size > 0
