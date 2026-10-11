"""Collective-communication sanity check: every rank contributes (rank + 1), the all-reduced
sum must equal W(W+1)/2 on every rank. On GPUs it also times a 256 MB all-reduce, which on
a Vera Rubin NVL72 node goes over NVLink / NVLink Switch (NCCL); on CPUs it uses gloo.

    torchrun --standalone --nproc_per_node=4 tests/allreduce_check.py        # one node
    srun --ntasks=8 --gpus-per-task=1 python tests/allreduce_check.py        # srun launch
"""

from __future__ import annotations

import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from huangtests import cleanup, init_distributed  # noqa: E402


def main() -> None:
    info = init_distributed()
    dev_name = torch.cuda.get_device_name(info.device) if info.device.type == "cuda" else "cpu"
    print(f"[allreduce] rank {info.rank}/{info.world_size} local_rank {info.local_rank} "
          f"backend {info.backend} device {dev_name} host {os.uname().nodename}", flush=True)
    if info.world_size == 1:
        print("[allreduce] single process: nothing to reduce (launch with torchrun --nproc_per_node=N)")
        return

    t = torch.tensor([float(info.rank + 1)], device=info.device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = info.world_size * (info.world_size + 1) / 2
    assert t.item() == expected, f"rank {info.rank}: got {t.item()}, expected {expected}"

    n_elem = (256 if info.device.type == "cuda" else 4) * 1024 * 1024 // 4   # 256 MB on GPU, 4 MB on CPU
    buf = torch.ones(n_elem, device=info.device)
    for _ in range(3):                                                          # warm-up
        dist.all_reduce(buf)
    if info.device.type == "cuda":
        torch.cuda.synchronize()
    dist.barrier()
    t0 = time.time()
    iters = 10
    for _ in range(iters):
        dist.all_reduce(buf)
    if info.device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    gb = buf.numel() * 4 / 1e9
    # ring all-reduce moves 2(W-1)/W of the buffer per rank; report "bus bandwidth" like nccl-tests
    bus_bw = gb * 2 * (info.world_size - 1) / info.world_size / dt
    if info.is_main:
        print(f"[allreduce] OK: sum={t.item():.0f}  {gb*1e3:.0f} MB all-reduce {dt*1e3:.2f} ms  "
              f"bus bandwidth {bus_bw:.1f} GB/s  ({info.backend}, {info.world_size} ranks)", flush=True)
    cleanup()


if __name__ == "__main__":
    main()
