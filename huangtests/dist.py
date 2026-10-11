"""Process-group setup that works under torchrun, srun, Slurm job arrays and plain `python`.

Backend choice
--------------
* GPUs present  -> NCCL (NVLink / NVLink-Switch inside a Vera Rubin NVL72 rack).
* No GPU        -> gloo, so the same scripts run on a laptop or in CI (x86_64 and arm64).

Rank discovery
--------------
torchrun sets RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT.
`srun --ntasks=N` sets SLURM_PROCID / SLURM_NTASKS / SLURM_LOCALID instead; we map those
onto the torchrun names so `torch.distributed` can read them.
A Slurm job array is *not* a process group: each array task is an independent process
(see `shard_indices`), so no init happens unless WORLD_SIZE > 1.
"""

from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    backend: str  # "nccl", "gloo" or "none"

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _map_slurm_env() -> None:
    """Translate srun's environment into torchrun's, if torchrun was not used."""
    if "RANK" in os.environ or "SLURM_PROCID" not in os.environ:
        return
    os.environ.setdefault("RANK", os.environ["SLURM_PROCID"])
    os.environ.setdefault("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1"))
    os.environ.setdefault("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))
    os.environ.setdefault("MASTER_PORT", "29500")
    if "MASTER_ADDR" not in os.environ and "SLURM_JOB_NODELIST" in os.environ:
        # First hostname of the allocation; `scontrol` is available on every Slurm node.
        import subprocess

        nodes = subprocess.run(
            ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        os.environ["MASTER_ADDR"] = nodes[0]


def init_distributed(timeout_minutes: int = 30) -> DistInfo:
    """Initialise torch.distributed (if launched with >1 process) and pick this rank's device."""
    _map_slurm_env()
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, timeout=_dt.timedelta(minutes=timeout_minutes))
    elif world_size == 1:
        backend = "none"

    return DistInfo(rank, world_size, local_rank, device, backend)


def is_main() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def all_reduce_mean(value: torch.Tensor | float, device: torch.device) -> float:
    """Average a scalar across ranks (no-op when running single-process)."""
    t = torch.as_tensor(value, dtype=torch.float32, device=device)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return float(t.item())


def cleanup() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def shard_indices(n_items: int) -> tuple[list[int], int, int]:
    """Indices this process owns for an embarrassingly parallel sweep.

    Works for either launcher:
      * Slurm job array:  SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT (one GPU per task)
      * torchrun:         RANK / WORLD_SIZE                             (one GPU per rank)
    Returns (indices, shard_id, n_shards). Items are dealt round-robin so shards are balanced.
    """
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        shard = int(os.environ["SLURM_ARRAY_TASK_ID"]) - int(os.environ.get("SLURM_ARRAY_TASK_MIN", 0))
        n_shards = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    else:
        shard = int(os.environ.get("RANK", 0))
        n_shards = int(os.environ.get("WORLD_SIZE", 1))
    return list(range(shard, n_items, n_shards)), shard, n_shards
