import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def seed_everything(seed: int = 1024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_ddp(rank: int, world_size: int):
    if world_size <= 1:
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29410")
    # Bind each spawned process before NCCL creates communicators.  Initializing
    # the process group while every child still has CUDA device 0 selected can
    # make NCCL's first parameter broadcast use the wrong device context on
    # some driver/PyTorch combinations (observed as an asynchronous illegal
    # memory access rather than a useful device-mismatch error).
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_rank0(rank: int) -> bool:
    return int(rank) == 0


def build_exp_dir(save_root: str) -> str:
    path = Path(save_root)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


# def save_checkpoint(path, model, optimizer=None, epoch=None, metrics=None, args=None):
#     payload = {"model": unwrap_model(model).state_dict()}
#     if optimizer is not None:
#         payload["optimizer"] = optimizer.state_dict()
#     if epoch is not None:
#         payload["epoch"] = epoch
#     if metrics is not None:
#         payload["metrics"] = metrics
#     if args is not None:
#         payload["args"] = vars(args) if hasattr(args, "__dict__") else args
#     torch.save(payload, path)

def save_checkpoint(path, model, optimizer=None, epoch=None, metrics=None, args=None, extra=None):
    """Atomically write a checkpoint.

    ``extra`` carries whatever else is needed to resume exactly where we stopped
    (scheduler state, the running best validation loss, the early-stop counter).
    Older checkpoints simply lack these keys; the resume path reconstructs them.
    """
    payload = {"model": unwrap_model(model).state_dict()}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    if metrics is not None:
        payload["metrics"] = metrics
    if args is not None:
        payload["args"] = vars(args) if hasattr(args, "__dict__") else args
    if extra:
        payload.update(extra)

    # Atomic save:
    # 1. write to a temporary file first;
    # 2. flush and fsync it;
    # 3. atomically replace the target file.
    # This prevents other ranks from reading a half-written .pth file.
    path = str(path)
    tmp_path = path + ".tmp"

    torch.save(payload, tmp_path)

    # Force the temporary checkpoint to be flushed to disk before rename.
    try:
        with open(tmp_path, "rb") as f:
            os.fsync(f.fileno())
    except Exception:
        pass

    os.replace(tmp_path, path)

def load_checkpoint(model, path: str, map_location="cpu", strict: bool = True):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint state must be a dict, got {type(state)} from {path}")

    state = dict(state)
    state.pop("_metadata", None)

    state = {
        (k[len("module."):] if isinstance(k, str) and k.startswith("module.") else k): v
        for k, v in state.items()
    }
    unwrap_model(model).load_state_dict(state, strict=strict)
    return ckpt
