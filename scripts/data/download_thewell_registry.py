#!/usr/bin/env python
"""Python-3.9-compatible downloader for a checked-out The Well registry.

This intentionally does not import ``the_well``.  Recent package versions use
Python-3.10-only type syntax in modules imported by ``the_well.data.__init__``,
even though downloading only needs registry.yaml and curl.
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import yaml


def _basename(url: str) -> str:
    return Path(urlparse(url).path).name


def _download_pairs(pairs, parallel: bool) -> None:
    pending = []
    for url, destination in pairs:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size > 0:
            print(f"[reuse] {destination}", flush=True)
            continue
        partial = destination.with_name(destination.name + ".part")
        pending.append((url, destination, partial))

    if not pending:
        return

    command = ["curl", "--fail", "--location", "--create-dirs", "--continue-at", "-"]
    if parallel and len(pending) > 1:
        command.extend(["--parallel", "--parallel-max", "4"])
    for url, _, partial in pending:
        command.extend(["--output", str(partial), url])
    subprocess.run(command, check=True)
    for _, destination, partial in pending:
        if not partial.exists() or partial.stat().st_size <= 0:
            raise RuntimeError(f"curl returned without a nonempty file: {partial}")
        os.replace(str(partial), str(destination))
        print(f"[saved] {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True, choices=["train", "valid", "test"])
    parser.add_argument("--first-only", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    args = parser.parse_args()

    registry_path = Path(args.registry)
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if args.dataset not in registry:
        raise KeyError(f"unknown dataset {args.dataset!r} in {registry_path}")
    entry = registry[args.dataset]
    urls = list(entry[args.split])
    if args.first_only:
        urls = urls[:1]
    if not urls:
        raise RuntimeError(f"empty registry entry: {args.dataset}/{args.split}")

    base = Path(args.base_path).expanduser().resolve()
    stats_destination = base / "datasets" / args.dataset / "stats.yaml"
    _download_pairs([(entry["stats"], stats_destination)], parallel=False)
    split_root = base / "datasets" / args.dataset / "data" / args.split
    pairs = [(url, split_root / _basename(url)) for url in urls]
    print(
        f"[registry] dataset={args.dataset} split={args.split} files={len(pairs)} "
        f"destination={split_root}",
        flush=True,
    )
    _download_pairs(pairs, parallel=bool(args.parallel))


if __name__ == "__main__":
    main()

