"""Public, model-independent interface for Internal-DW.

The research implementation lives in :mod:`internal_dw.models.dual_wiener`.  This
package exposes the small surface needed to integrate it into another PyTorch
model without depending on the experiment-specific model classes.
"""

from .router import InternalDW, InternalDWConfig, InternalDWResidual
from internal_dw.models.dual_wiener import solve_box_wiener_2x2

__all__ = [
    "InternalDW",
    "InternalDWConfig",
    "InternalDWResidual",
    "solve_box_wiener_2x2",
]

