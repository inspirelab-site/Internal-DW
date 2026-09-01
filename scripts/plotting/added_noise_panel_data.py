"""Validated loader for the assembled Figure 5(b) result ledger."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Panel:
    title: str
    estimator: str
    alpha: tuple[float, float, float]
    nonlinear: tuple[float, float, float]


EXPECTED_DATASETS = ("MG", "NARMA", "ETTm1", "ETTm2", "iEEG", "fMRI", "Shear", "WB2")


def load_panels(path: Path) -> tuple[Panel, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise ValueError(f"unsupported added-noise ledger: {path}")
    snrs = tuple(float(value) for value in payload.get("snr_order", ()))
    if snrs != (4.0, 1.0, 0.25):
        raise ValueError(f"expected SNR order (4,1,.25), got {snrs}")
    panels = tuple(
        Panel(
            title=str(item["dataset"]),
            estimator=str(item["estimator"]),
            alpha=tuple(float(value) for value in item["identity_gain"]),
            nonlinear=tuple(float(value) for value in item["nonlinear_gain"]),
        )
        for item in payload.get("panels", ())
    )
    if tuple(panel.title for panel in panels) != EXPECTED_DATASETS:
        raise ValueError("added-noise ledger has the wrong dataset order")
    values = np.asarray(
        [[*panel.alpha, *panel.nonlinear] for panel in panels], dtype=float
    )
    if values.shape != (8, 6) or not np.isfinite(values).all():
        raise ValueError(f"invalid added-noise gain matrix {values.shape}")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("added-noise gains must lie in [0,1]")
    return panels
