"""Toy stochastic tumour–immune simulator used as a stand-in for the PhysiCell ensemble.

In the project the simulator is the PhysiCell agent-based model run on Vera CPUs
(Sec. 4.3b, row "ABM simulation ensemble"); its summary statistics are written to disk
and loaded by the neural posterior scripts with `--data`. This file exists only so the
GPU scripts in this repo can be exercised end to end without PhysiCell.

Parameters (theta, 3-dim, uniform box prior)
  rho    tumour proliferation probability per step      [0, 0.5]
  kappa  immune kill efficiency                          [0, 1]
  lam    immune recruitment rate per 100 tumour cells    [0, 5]
Summary statistics (x, 12-dim)
  log1p(tumour count) and log1p(immune count) at six observation times.
"""

from __future__ import annotations

import torch

PRIOR_LOW = torch.tensor([0.0, 0.0, 0.0])
PRIOR_HIGH = torch.tensor([0.5, 1.0, 5.0])
THETA_DIM = 3
X_DIM = 12

_T_STEPS = 30
_OBS_TIMES = (5, 10, 15, 20, 25, 30)
_CARRYING_CAPACITY = 2000.0
_IMMUNE_DEATH = 0.05
_N_TUMOUR0, _N_IMMUNE0 = 100.0, 20.0


def sample_prior(n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    u = torch.rand(n, THETA_DIM, generator=generator)
    return PRIOR_LOW + (PRIOR_HIGH - PRIOR_LOW) * u


@torch.no_grad()
def simulate(theta: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Vectorised over the batch: theta (N, 3) -> x (N, 12)."""
    theta = theta.float()
    rho, kappa, lam = theta.unbind(-1)
    n = theta.shape[0]
    n_t = torch.full((n,), _N_TUMOUR0)
    n_i = torch.full((n,), _N_IMMUNE0)
    out = []
    for t in range(1, _T_STEPS + 1):
        p_birth = (rho * (1.0 - n_t / _CARRYING_CAPACITY)).clamp(0.0, 0.999)
        births = torch.binomial(n_t, p_birth, generator=generator)
        p_kill = (kappa * n_i / (n_i + n_t + 1.0)).clamp(0.0, 0.999)
        kills = torch.binomial(n_t, p_kill, generator=generator)
        recruits = torch.poisson(lam * n_t / 100.0, generator=generator)
        deaths = torch.binomial(n_i, torch.full_like(n_i, _IMMUNE_DEATH), generator=generator)
        n_t = (n_t + births - kills).clamp(min=0.0)
        n_i = (n_i + recruits - deaths).clamp(min=0.0)
        if t in _OBS_TIMES:
            out.append(torch.log1p(n_t))
            out.append(torch.log1p(n_i))
    return torch.stack(out, dim=-1)


def make_dataset(n_sims: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic (theta, x) pairs; every rank calls this with the same seed."""
    g = torch.Generator().manual_seed(seed)
    theta = sample_prior(n_sims, g)
    x = simulate(theta, g)
    return theta, x


def load_or_simulate(path: str | None, n_sims: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`--data file.pt` holds {"theta": (N,3), "x": (N,D)} written by the PhysiCell→SBI bridge."""
    if path:
        d = torch.load(path, map_location="cpu")
        return d["theta"].float(), d["x"].float()
    return make_dataset(n_sims, seed)


if __name__ == "__main__":
    th, xx = make_dataset(5, seed=0)
    print("theta\n", th, "\nx\n", xx)
