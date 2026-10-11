"""Helpers that launch the scripts the way Slurm would, but on CPU with 2 processes (gloo)."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _env(extra: dict | None = None) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""          # force the gloo/CPU path even on a GPU box
    env["OMP_NUM_THREADS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    env.update(extra or {})
    return env


def run_torchrun(script: str, *args: str, nproc: int = 2, timeout: int = 900) -> subprocess.CompletedProcess:
    """Launch `script` under torchrun with `nproc` CPU processes."""
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}",
           os.path.join(REPO, script), *args]
    return _run(cmd, _env(), timeout)


def run_array_task(script: str, *args: str, task_id: int, n_tasks: int, timeout: int = 900) -> subprocess.CompletedProcess:
    """Launch `script` as one Slurm array task (SLURM_ARRAY_TASK_ID / _COUNT set)."""
    env = _env({"SLURM_ARRAY_TASK_ID": str(task_id), "SLURM_ARRAY_TASK_MIN": "0",
                "SLURM_ARRAY_TASK_COUNT": str(n_tasks)})
    return _run([sys.executable, os.path.join(REPO, script), *args], env, timeout)


def _run(cmd: list[str], env: dict, timeout: int) -> subprocess.CompletedProcess:
    res = subprocess.run(cmd, env=env, cwd=REPO, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        print("STDOUT:\n", res.stdout[-4000:], "\nSTDERR:\n", res.stderr[-4000:])
    return res


@pytest.fixture
def tmp_out(tmp_path):
    return str(tmp_path / "out")
