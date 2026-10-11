"""Embarrassingly parallel scripts run as emulated Slurm array tasks (one GPU per task in
production; here each task is a CPU process)."""

import json
import os

import pytest

from conftest import run_array_task


def test_niche_sections_dealt_across_tasks(tmp_out):
    pytest.importorskip("torch_geometric")
    for k in range(2):
        res = run_array_task("omics/niche_job_array.py", "--n-sections", "3", "--n-cells", "300",
                             "--epochs", "20", "--out", tmp_out, task_id=k, n_tasks=2)
        assert res.returncode == 0
    outs = [json.load(open(os.path.join(tmp_out, f"niche_task{k}.json"))) for k in range(2)]
    sections = sorted(s["section"] for o in outs for s in o["sections"])
    assert sections == [0, 1, 2], "sections dealt round-robin with no overlap"
    assert all(0.0 <= s["test_accuracy"] <= 1.0 for o in outs for s in o["sections"])


def test_sbi_array_members_and_ensemble(tmp_out):
    pytest.importorskip("sbi")
    for k in range(2):
        res = run_array_task("npe/sbi_job_array.py", "--n-sims", "400", "--epochs", "2",
                             "--n-samples", "50", "--out", tmp_out, task_id=k, n_tasks=2, timeout=1200)
        assert res.returncode == 0
    res = run_array_task("npe/sbi_job_array.py", "--combine", "--out", tmp_out, task_id=0, n_tasks=1)
    assert res.returncode == 0
    ens = json.load(open(os.path.join(tmp_out, "ensemble.json")))
    assert ens["n_members"] == 2 and ens["n_samples"] == 100


def test_extract_array_launcher(tmp_out):
    """shard_indices must give the same split for SLURM_ARRAY_* as for RANK/WORLD_SIZE."""
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from huangtests import shard_indices

    os.environ.update({"SLURM_ARRAY_TASK_ID": "1", "SLURM_ARRAY_TASK_MIN": "0", "SLURM_ARRAY_TASK_COUNT": "3"})
    try:
        idx, shard, n = shard_indices(10)
        assert (idx, shard, n) == ([1, 4, 7], 1, 3)
    finally:
        for k in ("SLURM_ARRAY_TASK_ID", "SLURM_ARRAY_TASK_MIN", "SLURM_ARRAY_TASK_COUNT"):
            os.environ.pop(k, None)
    os.environ.update({"RANK": "2", "WORLD_SIZE": "3"})
    try:
        assert shard_indices(10) == ([2, 5, 8], 2, 3)
    finally:
        os.environ.pop("RANK"), os.environ.pop("WORLD_SIZE")
