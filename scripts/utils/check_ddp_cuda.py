#!/usr/bin/env python3
"""Small single-node CUDA/NCCL/DDP health check.

It exercises the same mp.spawn + per-rank device binding + DDP parameter
broadcast used by src/main.py, without touching datasets or experiments.
"""

import argparse
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank: int, world_size: int, master_port: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    device = torch.device("cuda", rank)
    value = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(value)
    expected = world_size * (world_size + 1) / 2
    if value.item() != expected:
        raise RuntimeError(f"rank {rank}: all_reduce={value.item()} expected={expected}")

    model = torch.nn.Sequential(
        torch.nn.Conv2d(2, 8, kernel_size=3, padding=1),
        torch.nn.GELU(),
        torch.nn.Conv2d(8, 2, kernel_size=3, padding=1),
    ).to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], broadcast_buffers=False
    )
    x = torch.randn(1, 2, 32, 32, device=device)
    loss = model(x).square().mean()
    loss.backward()
    torch.cuda.synchronize(device)
    print(
        f"[ok] host={socket.gethostname()} rank={rank} "
        f"device={torch.cuda.get_device_name(rank)} loss={loss.item():.6g}",
        flush=True,
    )
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--master-port", type=int, default=58991)
    args = parser.parse_args()
    world_size = args.world_size or torch.cuda.device_count()
    if world_size < 2:
        raise SystemExit(f"need at least two visible GPUs; found {world_size}")
    mp.spawn(
        _worker,
        args=(world_size, args.master_port),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
