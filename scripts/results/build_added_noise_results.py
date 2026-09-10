#!/usr/bin/env python3
"""Assemble Figure 5(b) gains from the eight frozen-checkpoint probes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SNRS = (4.0, 1.0, 0.25)


def _token(snr: float) -> str:
    return f"{snr:g}".replace(".", "p")


def _pipeline_gain(path: Path) -> tuple[float, float]:
    with np.load(path, allow_pickle=False) as archive:
        if "pipeline_plugin_final" not in archive.files:
            raise ValueError(f"{path} lacks pipeline_plugin_final")
        values = np.asarray(archive["pipeline_plugin_final"], dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] != 2 or not np.isfinite(values).all():
        raise ValueError(f"invalid pipeline gains in {path}: {values.shape}")
    gain = values.mean(axis=tuple(range(values.ndim - 1)))
    return float(gain[0]), float(gain[1])


def _json_gain(path: Path, family: str) -> tuple[float, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if family == "prior":
        values = payload["summary"]["prior"]
    elif family == "spatial_spectrum":
        values = payload["variant_summary"]["spatial_spectrum"]
    else:
        raise ValueError(f"unknown JSON gain family {family!r}")
    gain = np.asarray([values["alpha_mean"], values["m_mean"]], dtype=float)
    if not np.isfinite(gain).all() or np.any((gain < 0.0) | (gain > 1.0)):
        raise ValueError(f"invalid {family} gains in {path}: {gain}")
    return float(gain[0]), float(gain[1])


def _source_specs(root: Path):
    return (
        (
            "MG",
            "DW-Generic",
            "npz",
            root / "probe_outputs/real_internal_dw_estimator_sweep_v1/mg_structured",
            "mg_structured_snr{token}.npz",
            "pipeline",
        ),
        (
            "NARMA",
            "DW-Generic",
            "npz",
            root / "probe_outputs/real_internal_dw_estimator_sweep_v1/structured_40gb",
            "narma_structured_snr{token}.npz",
            "pipeline",
        ),
        (
            "ETTm1",
            "DW-Generic",
            "npz",
            root / "probe_outputs/ettm_internal_dw_diagnostics_v1/ettm1",
            "noise_snr{token}.npz",
            "pipeline",
        ),
        (
            "ETTm2",
            "DW-Generic",
            "npz",
            root / "probe_outputs/ettm_internal_dw_diagnostics_v1/ettm2",
            "noise_snr{token}.npz",
            "pipeline",
        ),
        (
            "iEEG",
            "DW-Prior",
            "json",
            Path(os.environ.get('IEEG_PROBE_ROOT', root / 'probe_outputs/ieeg_cohort_v1')) / 'noise',
            "snr{token}.json",
            "prior",
        ),
        (
            "fMRI",
            "DW-Prior",
            "json",
            root / "probe_outputs/real_internal_dw_prior_noise_response_v1/fmri",
            "fmri_prior_snr{token}_seed0.json",
            "prior",
        ),
        (
            "Shear",
            "DW-Prior",
            "json",
            root / "probe_outputs/real_internal_dw_estimator_sweep_v2/shear",
            "shear_snr{token}.json",
            "spatial_spectrum",
        ),
        (
            "WB2",
            "DW-Prior",
            "json",
            root / "probe_outputs/real_internal_dw_estimator_sweep_v2/wb2",
            "wb2_spectral_seed0_snr{token}.json",
            "spatial_spectrum",
        ),
    )


def build(root: Path) -> dict:
    panels = []
    for dataset, estimator, kind, directory, template, family in _source_specs(root):
        identity, nonlinear, sources = [], [], []
        for snr in SNRS:
            path = directory / template.format(token=_token(snr))
            if not path.is_file():
                raise FileNotFoundError(f"missing {dataset} SNR={snr:g}: {path}")
            gain = _pipeline_gain(path) if kind == "npz" else _json_gain(path, family)
            identity.append(gain[0])
            nonlinear.append(gain[1])
            sources.append(str(path.resolve()))
        panels.append(
            {
                "dataset": dataset,
                "estimator": estimator,
                "identity_gain": identity,
                "nonlinear_gain": nonlinear,
                "sources": sources,
            }
        )
    return {
        "format_version": 1,
        "result": "figure_5b_added_noise_gain_response",
        "snr_order": list(SNRS),
        "aggregation": "mean over all reported horizon-layer routes",
        "panels": panels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "probe_outputs/application_diagnostics_v1/added_noise_gain_summary.json",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    payload = build(root)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("dataset  estimator       alpha (SNR 4/1/.25)       m (SNR 4/1/.25)")
    for panel in payload["panels"]:
        alpha = "/".join(f"{value:.3f}" for value in panel["identity_gain"])
        nonlinear = "/".join(
            f"{value:.3f}" for value in panel["nonlinear_gain"]
        )
        print(
            f"{panel['dataset']:7s}  {panel['estimator']:10s}  "
            f"{alpha:>19s}  {nonlinear:>19s}"
        )
    print(f"[out] {output}")


if __name__ == "__main__":
    main()
