import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist


def setup_dist():
    """Single machine distributed setup"""
    if dist.is_initialized():
        return


    free_port = _find_free_port()

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(free_port)
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    dist.init_process_group(
        backend="nccl" if th.cuda.is_available() else "gloo",
        world_size=1,
        rank=0
    )
    return 0, 1


def dev():
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    # Simple file loading without MPI
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    # No-op for single machine
    pass


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()


def get_rank():
    return 0


def get_world_size():
    return 1


def local_rank():
    return 0