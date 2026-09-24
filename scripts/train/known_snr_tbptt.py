"""Known-SNR TBPTT: cut incoming predictions and outgoing recurrent states."""
from internal_dw.models.official_state_mamba import OfficialStateMambaARModel

IMPLEMENTATION = 'input_prediction_output_state'
_original_step = OfficialStateMambaARModel.step
_period = 0


def truncated_step(self, h, x_t, stim_t=None, return_aux=False,
                horizon_index=None, total_horizon=None, ratio_collector=None):
    cut = (_period > 0 and horizon_index is not None
           and int(horizon_index) > 0 and int(horizon_index) % _period == 0)
    if cut:
        x_t = x_t.detach()
    result = _original_step(self,h,x_t,stim_t,return_aux=return_aux,
        horizon_index=horizon_index,total_horizon=total_horizon,ratio_collector=ratio_collector)
    if not cut:
        return result
    prediction, next_state, *aux = result
    next_state = tuple((conv.detach(),ssm.detach()) for conv,ssm in next_state)
    if aux:
        metadata = dict(aux[0])
        if 'h_next' in metadata: metadata['h_next'] = next_state
        return prediction,next_state,metadata
    return prediction,next_state


def install(period):
    global _period
    if int(period) < 0: raise ValueError('Negative truncation period')
    _period = int(period)
    OfficialStateMambaARModel.step = truncated_step if _period else _original_step


def validate_args(args):
    if args.known_snr_tbptt_period not in (4,8,16):
        raise ValueError('Use the declared TBPTT grid 4/8/16')
    if args.bptt_detach_period != 0 or args.resgrad_routing:
        raise ValueError('Do not combine TBPTT cuts with another routing/truncation operator')
    if args.tbptt_implementation != IMPLEMENTATION:
        raise ValueError('Unexpected TBPTT implementation')
