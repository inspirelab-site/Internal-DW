# Paper figures used in the project overview

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
