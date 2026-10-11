"""Embarrassingly parallel NPE with the `sbi` package: one independent fit per GPU
(Sec. 4.3b, row "Neural posterior inference"; Sec. 4.5b, "PhysiCell–SBI bridge").

`sbi` trains on a single device. The parallelism in the project comes from running many
independent fits at once — different random seeds (deep ensemble), different rounds of the
sequential loop, different observed datasets — as a Slurm job array with one GPU per task:

    sbatch --array=0-7 slurm/job_array_single_gpu.sbatch npe/sbi_job_array.py

Each task trains a neural spline flow on the same simulation set, draws posterior samples
for the observation `x_o`, and writes them to `<out>/samples_task<k>.pt`. Afterwards

    python npe/sbi_job_array.py --combine --out runs/sbi_array

pools the shards into an equal-weight ensemble posterior and reports its mean and sd.
Outside Slurm, `--task-id k --n-tasks n` emulates one array task (used by the tests).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
from simulate import PRIOR_HIGH, PRIOR_LOW, load_or_simulate, simulate  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None)
    p.add_argument("--n-sims", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0, help="simulation-set seed (shared by all tasks)")
    p.add_argument("--epochs", type=int, default=50, help="sbi max_num_epochs")
    p.add_argument("--n-samples", type=int, default=2000, help="posterior draws per task")
    p.add_argument("--task-id", type=int, default=None, help="override SLURM_ARRAY_TASK_ID")
    p.add_argument("--n-tasks", type=int, default=None, help="override SLURM_ARRAY_TASK_COUNT")
    p.add_argument("--out", default="runs/sbi_array")
    p.add_argument("--combine", action="store_true", help="pool samples_task*.pt into an ensemble")
    return p.parse_args()


def task_info(args: argparse.Namespace) -> tuple[int, int]:
    tid = args.task_id if args.task_id is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    n = args.n_tasks if args.n_tasks is not None else int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    return tid, n


def observation(seed: int = 12345) -> tuple[torch.Tensor, torch.Tensor]:
    """A synthetic 'observed' dataset with known parameters, for checking the posterior."""
    theta_true = torch.tensor([[0.25, 0.6, 2.0]])
    g = torch.Generator().manual_seed(seed)
    return theta_true, simulate(theta_true, g)


def run_task(args: argparse.Namespace) -> None:
    from sbi.inference import NPE
    from sbi.utils import BoxUniform

    tid, n_tasks = task_info(args)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1000 + tid)  # ensemble member k differs only in its init / minibatch order

    theta, x = load_or_simulate(args.data, args.n_sims, args.seed)
    prior = BoxUniform(PRIOR_LOW.to(device), PRIOR_HIGH.to(device), device=device)
    inference = NPE(prior=prior, density_estimator="nsf", device=device)
    inference.append_simulations(theta.to(device), x.to(device))
    inference.train(max_num_epochs=args.epochs, training_batch_size=256, show_train_summary=False)
    posterior = inference.build_posterior()

    theta_true, x_o = observation()
    samples = posterior.sample((args.n_samples,), x=x_o.to(device), show_progress_bars=False).cpu()

    os.makedirs(args.out, exist_ok=True)
    torch.save({"samples": samples, "task_id": tid, "n_tasks": n_tasks, "theta_true": theta_true,
                "device": torch.cuda.get_device_name(0) if device != "cpu" else "cpu"},
               os.path.join(args.out, f"samples_task{tid}.pt"))
    print(f"[sbi] task {tid}/{n_tasks} on {device}: posterior mean {samples.mean(0).tolist()} "
          f"(true {theta_true[0].tolist()})", flush=True)


def combine(args: argparse.Namespace) -> None:
    files = sorted(glob.glob(os.path.join(args.out, "samples_task*.pt")))
    if not files:
        raise SystemExit(f"no samples_task*.pt in {args.out}")
    shards = [torch.load(f, map_location="cpu") for f in files]
    pooled = torch.cat([s["samples"] for s in shards])
    theta_true = shards[0]["theta_true"][0]
    summary = {
        "n_members": len(shards),
        "n_samples": int(pooled.shape[0]),
        "ensemble_mean": [round(v, 4) for v in pooled.mean(0).tolist()],
        "ensemble_sd": [round(v, 4) for v in pooled.std(0).tolist()],
        "theta_true": theta_true.tolist(),
        "members": [{"task_id": s["task_id"], "mean": [round(v, 4) for v in s["samples"].mean(0).tolist()]} for s in shards],
    }
    with open(os.path.join(args.out, "ensemble.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    a = parse_args()
    combine(a) if a.combine else run_task(a)
