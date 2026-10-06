# Visuals used in the project overview

## Known-SNR motivation animation

`known-snr-motivation.gif` opens the project README; `known-snr-motivation.png`
provides the static version. The animation uses archived measurements from the
paper's known-SNR experiment, with the same per-horizon gradient definitions as
[`probe_known_snr_failure_profile.py`](../../scripts/probes/probe_known_snr_failure_profile.py).

The frozen Full-BPTT checkpoint uses training seed 0. The measurements cover
32 forecast steps and 15 test trajectories, with four Monte Carlo repetitions
and 64 future-noise draws per repetition. These are Monte Carlo repetitions,
not four training seeds. Curves show medians; bands show the min–max range.

| Panel | Quantity |
|---|---|
| Gradient size | Per-step gradient RMS relative to step 1, on a linear axis |
| Gradient SNR | Predictable gradient energy divided by innovation energy, on a linear axis |
| Gradient composition | Innovation energy / (predictable energy + innovation energy) at the current step; blue is the complementary predictable fraction |

The composition bar is a per-step decomposition, not the noise fraction of
the summed update. The finite-sample signal–innovation cross term is excluded
from this ratio. The approximately 1.24× growth per step is a descriptive
log-linear fit to the median RMS curve over all 32 steps; the plotted
observations retain their measured fluctuations.

The 25 fps animation interpolates between measured steps for playback. Dot
markers remain at measured integer horizons. The exported
[`known-snr-motivation.json`](known-snr-motivation.json) preserves the measured
curves, uncertainty ranges, protocol, and SHA-256 hashes of the four source
`replicate*.json` files. Machine-specific source paths are omitted.

The source measurements are the archived paper `panels_1_2_replicates` outputs.
The current reproduction workflow writes corresponding profiles under
`experiments/known_snr/profiles/`; see the
[known-SNR reproduction guide](../REPRODUCING.md#reproduce-the-known-snr-closure).

## Paper figures

These images are display copies of figures from
[the paper, arXiv v2](https://arxiv.org/abs/2609.12890v2).
They are bundled for the README and do not replace the experiment result ledgers.

| Asset | Paper figure | Source in the supplied manuscript |
|---|---|---|
| `gradient-controls.png` | Figure 1 | `figs/intro_gradient_controls.pdf` |
| `internal-dw-method.png` | Figure 2 | `figure/fig02_internal_dw_method.pdf` |
| `forecasting-results.png` | Figure 6 | `figure/unified_compact/fig06_forecasting.pdf` |

All three were rendered directly from the PDFs referenced by
`iclr2027_conference.tex`, using Poppler at 2200 pixels on the longest edge:

```bash
pdftoppm -singlefile -scale-to 2200 -png INPUT.pdf OUTPUT_PREFIX
```

The supplied manuscript also contains older PNG exports. In particular, the
Figure 6 PNG differs from the PDF used in the current manuscript; regenerate
from the referenced PDF when refreshing the README.
