# Internal-DW API contract

## Residual routing

```python
skip_x, branch_x = router.route(x, horizon=k, layer=l)
y = skip_x + branch(branch_x)
```

| Name | Type | Meaning |
|---|---|---|
| `x` | `torch.Tensor` | Residual stream immediately before the split. |
| `horizon` | `int` | Zero-based autoregressive forecast step in `[0, max_horizon)`. |
| `layer` | `int` | Zero-based routed block in `[0, num_layers)`. |
| `skip_x` | `torch.Tensor` | Forward-identical view whose upstream VJP is scaled by `alpha`. |
| `branch_x` | `torch.Tensor` | Forward-identical view whose upstream VJP is scaled by `m`. |

The split must happen before evaluating the nonlinear branch. Applying a gain
to `branch(x)` afterwards would also scale the branch's local parameter
gradient and is not the operator evaluated in the paper.

For a recurrent block with a separate hidden state entering only the nonlinear
route, use:

```python
routed_hidden = router.route_branch_state(hidden, horizon=k, layer=l)
```

## Calibration lifecycle

1. `begin_batch()` resets transient state and decides whether the batch is a
   probe batch.
2. Call `observe(prediction, target, horizon=k)` for every forecast step. This
   always updates lagged residual statistics and, on a probe batch, accumulates
   matched total/noise objectives.
3. Call `calibrate()` after constructing the scalar training loss but before
   `loss.backward()`. It uses `torch.autograd.grad` to measure route messages
   without adding anything to parameter `.grad` fields.
4. Call `loss.backward()` normally.
5. Call `end_batch()` immediately after backward. Estimated gains are lagged:
   they are committed now and used starting with the next batch.
6. Call `optimizer.step()` according to the usual accumulation schedule.

With gradient accumulation, repeat the router lifecycle for each micro-batch;
the optimizer zero/step schedule remains unchanged. With DDP, construct the
router before wrapping the model so its buffers are in the module state dict.

## State and outputs

- `gains`: live tensor of shape `[max_horizon, num_layers, 2]`; the last axis is
  `[identity alpha, nonlinear m]` and every value lies in `[0, 1]`.
- `gain(horizon=..., layer=...)`: detached copy of one two-vector.
- `diagnostics(horizon=None)`: scalar dictionary suitable for TensorBoard or
  Weights & Biases.
- `export_state(horizon=None)`: JSON-safe dictionary containing gains, route
  moments, readiness, residual statistics, and summary diagnostics.
- `state_dict()`: standard PyTorch checkpoint state including all fitted
  moments and coefficients.

## Generic and structured noise models

The portable default is `noise_model="diagonal_gaussian"`, which estimates
coordinatewise residual variance online. The paper also evaluates structured
train-only priors for some domains. Those advanced samplers are implemented by
the underlying `DualWienerController` and configured by the paper runners; a
third-party integration should begin with the generic model unless it has a
pre-specified domain innovation sampler.

