"""Neural posterior estimation with PyTorch DistributedDataParallel (Sec. 4.3b, row
"Neural posterior inference").

A conditional normalising flow q(theta | x) is trained by maximum likelihood on
(theta, x) pairs from the simulator. Each GPU holds a full replica of the flow and a
disjoint shard of the simulation set (DistributedSampler); gradients are averaged with
NCCL all-reduce every step, so the effective batch is `batch_size * world_size`.

Launch
  single node   torchrun --standalone --nproc_per_node=4 npe/train_npe_ddp.py --n-sims 200000
  multi node    see slurm/ddp_multi_node.sbatch
  CPU / CI      torchrun --standalone --nproc_per_node=2 npe/train_npe_ddp.py --n-sims 2000 --epochs 2

Why not `sbi` directly? sbi's trainer is single-device. For the per-round training in the
sequential SBI loop that is fine (see sbi_job_array.py — one round / seed per GPU). This
script is the data-parallel path for the large final-round fits where one GPU is too slow.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]  # run from anywhere without installing the package
from huangtests import all_reduce_mean, cleanup, init_distributed  # noqa: E402
from simulate import PRIOR_HIGH, PRIOR_LOW, THETA_DIM, X_DIM, load_or_simulate  # noqa: E402


# --------------------------------------------------------------------------- flow model
class CouplingLayer(nn.Module):
    """Affine coupling (RealNVP-style) conditioned on a context vector."""

    def __init__(self, dim: int, ctx_dim: int, hidden: int, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.net = nn.Sequential(
            nn.Linear(dim + ctx_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _shift_scale(self, inp: torch.Tensor, ctx: torch.Tensor):
        h = self.net(torch.cat([inp * self.mask, ctx], dim=-1))
        log_s, t = h.chunk(2, dim=-1)
        log_s = torch.tanh(log_s) * (1.0 - self.mask)
        t = t * (1.0 - self.mask)
        return log_s, t

    def forward(self, theta: torch.Tensor, ctx: torch.Tensor):
        log_s, t = self._shift_scale(theta, ctx)
        z = theta * self.mask + (1.0 - self.mask) * (theta * torch.exp(log_s) + t)
        return z, log_s.sum(dim=-1)

    def inverse(self, z: torch.Tensor, ctx: torch.Tensor):
        log_s, t = self._shift_scale(z, ctx)
        return z * self.mask + (1.0 - self.mask) * ((z - t) * torch.exp(-log_s))


class ConditionalFlow(nn.Module):
    """q(theta | x): embed x, then push theta through alternating coupling layers."""

    def __init__(self, theta_dim: int, x_dim: int, n_layers: int = 6, hidden: int = 128, ctx_dim: int = 64):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(x_dim, hidden), nn.SiLU(), nn.Linear(hidden, ctx_dim), nn.SiLU())
        masks = [((torch.arange(theta_dim) + i) % 2 == 0) for i in range(n_layers)]
        self.layers = nn.ModuleList(CouplingLayer(theta_dim, ctx_dim, hidden, m) for m in masks)
        self.theta_dim = theta_dim

    def log_prob(self, theta_u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ctx = self.embed(x)
        z, logdet = theta_u, torch.zeros(theta_u.shape[0], device=theta_u.device)
        for layer in self.layers:
            z, ld = layer(z, ctx)
            logdet = logdet + ld
        log_base = -0.5 * (z**2).sum(-1) - 0.5 * self.theta_dim * math.log(2 * math.pi)
        return log_base + logdet

    # DDP only synchronises gradients for calls that go through the wrapper's forward(),
    # so training must call model(theta_u, x), never model.module.log_prob(...).
    forward = log_prob

    @torch.no_grad()
    def sample(self, x: torch.Tensor, n: int) -> torch.Tensor:
        ctx = self.embed(x).repeat_interleave(n, dim=0)
        z = torch.randn(ctx.shape[0], self.theta_dim, device=x.device)
        for layer in reversed(self.layers):
            z = layer.inverse(z, ctx)
        return z.view(x.shape[0], n, self.theta_dim)


def to_unconstrained(theta: torch.Tensor) -> torch.Tensor:
    u = ((theta - PRIOR_LOW.to(theta)) / (PRIOR_HIGH - PRIOR_LOW).to(theta)).clamp(1e-6, 1 - 1e-6)
    return torch.logit(u)


def to_constrained(theta_u: torch.Tensor) -> torch.Tensor:
    return PRIOR_LOW.to(theta_u) + (PRIOR_HIGH - PRIOR_LOW).to(theta_u) * torch.sigmoid(theta_u)


# ----------------------------------------------------------------------------- training
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None, help=".pt with theta/x from the PhysiCell ensemble")
    p.add_argument("--n-sims", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256, help="per-GPU batch size")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--out", default="runs/npe_ddp")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    info = init_distributed()
    torch.manual_seed(args.seed + info.rank)
    dev = info.device

    theta, x = load_or_simulate(args.data, args.n_sims, args.seed)
    n_val = max(1, len(theta) // 10)
    x_mean, x_std = x[n_val:].mean(0), x[n_val:].std(0) + 1e-6
    x_n = (x - x_mean) / x_std
    theta_u = to_unconstrained(theta)
    train_ds = TensorDataset(theta_u[n_val:], x_n[n_val:])
    val_ds = TensorDataset(theta_u[:n_val], x_n[:n_val])

    train_sampler = DistributedSampler(train_ds, num_replicas=info.world_size, rank=info.rank, shuffle=True, seed=args.seed)
    val_sampler = DistributedSampler(val_ds, num_replicas=info.world_size, rank=info.rank, shuffle=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=1024, sampler=val_sampler)

    model = ConditionalFlow(THETA_DIM, X_DIM, args.n_layers, args.hidden).to(dev)
    if info.world_size > 1:
        model = DDP(model, device_ids=[info.local_rank] if dev.type == "cuda" else None)
    core = model.module if isinstance(model, DDP) else model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs * len(train_dl)))

    if info.is_main:
        dev_name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"
        print(f"[npe] world_size={info.world_size} backend={info.backend} device={dev_name} "
              f"train={len(train_ds)} val={len(val_ds)} per-GPU batch={args.batch_size}", flush=True)

    t0 = time.time()
    history = []
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        tot, cnt = 0.0, 0
        for th, xx in train_dl:
            th, xx = th.to(dev, non_blocking=True), xx.to(dev, non_blocking=True)
            loss = -model(th, xx).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()          # DDP all-reduces gradients here (NCCL on GPU, gloo on CPU)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += loss.item() * th.shape[0]
            cnt += th.shape[0]
        train_nll = all_reduce_mean(tot / max(cnt, 1), dev)

        model.eval()
        vtot, vcnt = 0.0, 0
        with torch.no_grad():
            for th, xx in val_dl:
                th, xx = th.to(dev), xx.to(dev)
                vtot += -core.log_prob(th, xx).sum().item()
                vcnt += th.shape[0]
        val_nll = all_reduce_mean(vtot / max(vcnt, 1), dev)
        history.append({"epoch": epoch, "train_nll": train_nll, "val_nll": val_nll})
        if info.is_main:
            print(f"[npe] epoch {epoch:3d}  train NLL {train_nll:8.4f}  val NLL {val_nll:8.4f}", flush=True)
    train_time = time.time() - t0

    # ---- quick posterior check on rank 0: posterior mean error and 90% interval coverage
    if info.is_main:
        os.makedirs(args.out, exist_ok=True)
        n_check = min(200, n_val)
        xs, ths = x_n[:n_check].to(dev), theta[:n_check].to(dev)
        samples = to_constrained(core.sample(xs, 500))             # (n_check, 500, 3)
        post_mean = samples.mean(1)
        rmse = ((post_mean - ths) ** 2).mean(0).sqrt()
        lo, hi = samples.quantile(0.05, dim=1), samples.quantile(0.95, dim=1)
        coverage = ((ths >= lo) & (ths <= hi)).float().mean(0)
        metrics = {
            "world_size": info.world_size,
            "backend": info.backend,
            "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
            "n_train": len(train_ds),
            "epochs": args.epochs,
            "train_time_s": round(train_time, 2),
            "samples_per_s_total": round(len(train_ds) * args.epochs / max(train_time, 1e-9), 1),
            "final_train_nll": history[-1]["train_nll"],
            "final_val_nll": history[-1]["val_nll"],
            "posterior_mean_rmse": [round(v, 4) for v in rmse.tolist()],
            "coverage_90pct": [round(v, 3) for v in coverage.tolist()],
            "history": history,
        }
        with open(os.path.join(args.out, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        torch.save({"state_dict": core.state_dict(), "x_mean": x_mean, "x_std": x_std,
                    "prior_low": PRIOR_LOW, "prior_high": PRIOR_HIGH, "args": vars(args)},
                   os.path.join(args.out, "flow.pt"))
        print(f"[npe] done in {train_time:.1f}s  posterior-mean RMSE {metrics['posterior_mean_rmse']}  "
              f"90% coverage {metrics['coverage_90pct']}", flush=True)

    cleanup()


if __name__ == "__main__":
    main()
