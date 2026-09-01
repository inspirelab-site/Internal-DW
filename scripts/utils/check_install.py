#!/usr/bin/env python3
"""Validate Conda and PyTorch before installing Internal-DW."""

from __future__ import annotations

import argparse
import os
import re
import sys


def _version_tuple(version: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        raise ValueError(f"cannot parse PyTorch version {version!r}")
    return int(match.group(1)), int(match.group(2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check Conda, PyTorch, and optional CUDA availability."
    )
    parser.add_argument(
        "--require-conda",
        action="store_true",
        help="fail unless a Conda environment is currently activated",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail unless PyTorch can use at least one CUDA device",
    )
    args = parser.parse_args()

    if args.require_conda and not os.environ.get("CONDA_PREFIX"):
        raise SystemExit(
            "[failed] no active Conda environment; run `conda activate internal-dw`"
        )

    try:
        import torch
    except (ImportError, OSError) as exc:
        raise SystemExit(
            "[failed] PyTorch is not usable. Install PyTorch before Internal-DW. "
            f"Original error: {exc}"
        ) from exc

    try:
        torch_version = _version_tuple(torch.__version__)
    except ValueError as exc:
        raise SystemExit(f"[failed] {exc}") from exc
    if torch_version < (2, 1):
        raise SystemExit(
            f"[failed] PyTorch >=2.1 is required; found {torch.__version__}"
        )

    cuda_ready = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_ready else 0
    if args.require_cuda and not cuda_ready:
        raise SystemExit("[failed] this PyTorch installation cannot use CUDA")

    env_name = os.environ.get("CONDA_DEFAULT_ENV", "not-active")
    print(f"[ok] conda={env_name} python={sys.version.split()[0]}")
    print(
        f"[ok] torch={torch.__version__} torch_cuda={torch.version.cuda} "
        f"cuda_available={cuda_ready} devices={device_count}"
    )


if __name__ == "__main__":
    main()
