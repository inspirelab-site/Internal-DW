# Internal dual-Wiener routing

This implementation applies a backward-only soft gate at every internal Mamba
residual merge and rollout horizon:

\[
  \nabla_x L \leftarrow \alpha_{k,l} d_I + m_{k,l} d_J,
\]

where `d_I` and `d_J` are the identity- and nonlinear-route VJPs. The forward
model and parameter gradients local to the current nonlinear branch are
unchanged. Recurrent state inputs use the same branch coefficient `m` so a
future loss cannot bypass the routed nonlinear temporal path through Mamba's
state carry.

## Estimator

For route credit `d = s + n`, the clean target is `s_I + s_J`. With signal
covariance `P` and noise covariance `R`, the two coefficients solve

\[
  \min_{0\leq w\leq1}
  w^\top(P+R)w - 2 w^\top P\mathbf 1,
  \qquad w=(\alpha,m).
\]

The solver includes the cross-route covariance and enumerates the interior,
edges, and corners of the two-dimensional box. It does not clamp two scalar
gains independently.

Every calibration batch performs two read-only VJP probes on the fully open
graph:

- a half-squared-error probe estimates `P + R`;
- a matched Gaussian output-noise probe estimates `R`.

The current screening launchers retain the repository's `rel_l2` training loss
for direct comparability with existing baselines. Consequently, the gain is
optimal for the quadratic calibration surrogate under the assumptions above,
not exactly for the `rel_l2` update itself. A fully matched MSE experiment is a
useful later ablation, but changing only the proposed method's training loss in
the first screen would confound the baseline comparison.

The Gaussian noise is generated from a lagged, centered, per-horizon diagonal
EMA of prediction residuals from earlier batches. Gains are committed only
after the real backward and are therefore used with a one-batch lag. The
Gaussian probes use `autograd.grad` only with respect to rollout roots, so their
random gradients never enter model-parameter `.grad` fields.

## What may be claimed

This is an approximate, routewise Wiener/Kalman-style gain under an explicit
diagonal-Gaussian plug-in noise assumption. It is not an exact Kalman gain and
gradient coherence is not used as a substitute for SNR. A learned sigma head is
not required.

The main approximation is that centered training residual variation is used as
observation noise. It also contains changing model error, so it can overestimate
`R` early in training. The internal recurrent-state route shares the `m`
estimated at the corresponding token branch. Cross-fitted held-out residuals,
structured output covariance, and a separately identified recurrent-state
noise model are natural follow-up variants.

The full coefficient and covariance audit record for the best checkpoint is
written to `dual_wiener_gains.json`; the last epoch is written to
`dual_wiener_gains_last.json`.
