"""Data-parallel scripts run under torchrun with 2 CPU processes (gloo). On a GPU node the
same scripts use NCCL; nothing in the scripts changes."""

import json
import os

import pytest

from conftest import run_torchrun


def test_allreduce_two_ranks():
    res = run_torchrun("tests/allreduce_check.py")
    assert res.returncode == 0
    assert "[allreduce] OK: sum=3" in res.stdout


def test_npe_ddp_trains_and_saves(tmp_out):
    res = run_torchrun("npe/train_npe_ddp.py", "--n-sims", "3000", "--epochs", "3",
                       "--batch-size", "64", "--n-layers", "4", "--hidden", "32", "--out", tmp_out)
    assert res.returncode == 0
    m = json.load(open(os.path.join(tmp_out, "metrics.json")))
    assert m["world_size"] == 2 and m["backend"] == "gloo"
    assert m["history"][-1]["val_nll"] < m["history"][0]["val_nll"], "validation NLL should fall"
    assert os.path.exists(os.path.join(tmp_out, "flow.pt"))
    assert len(m["coverage_90pct"]) == 3


def test_omics_finetune_ddp(tmp_out):
    pytest.importorskip("transformers")
    res = run_torchrun("omics/finetune_ddp.py", "--model-name", "tiny", "--n-cells", "600",
                       "--epochs", "1", "--batch-size", "16", "--out", tmp_out)
    assert res.returncode == 0
    m = json.load(open(os.path.join(tmp_out, "metrics.json")))
    assert m["world_size"] == 2
    assert 0.0 <= m["eval_accuracy"] <= 1.0


def test_llm_lora_ddp_then_sharded_extraction(tmp_out):
    pytest.importorskip("peft")
    res = run_torchrun("llm/lora_finetune.py", "--model-name", "tiny", "--epochs", "120",
                       "--batch-size", "4", "--out", tmp_out)
    assert res.returncode == 0
    m = json.load(open(os.path.join(tmp_out, "metrics.json")))
    assert m["world_size"] == 2 and m["n_pairs"] == 24
    assert os.path.exists(os.path.join(tmp_out, "adapter"))
    assert os.path.exists(os.path.join(tmp_out, "model", "vocab.json"))

    # extraction as 2 torchrun ranks (one GPU per rank in production)
    ext = os.path.join(tmp_out, "extract")
    res = run_torchrun("llm/extract_sharded.py", "--model-name", "tiny",
                       "--model-dir", os.path.join(tmp_out, "model"), "--out", ext)
    assert res.returncode == 0
    manifests = [json.load(open(os.path.join(ext, f"manifest_shard{k}.json"))) for k in range(2)]
    covered = sorted(a for m in manifests for a in m["article_ids"])
    assert covered == sorted(set(covered)) and len(covered) == 8, "every article exactly once"
    assert sum(m["n_sentences"] for m in manifests) > 0
    assert sum(m["n_parsed"] for m in manifests) > 0, "the toy model should emit some 4-field rules"
    for k in range(2):
        assert os.path.exists(os.path.join(ext, f"rules_shard{k}.jsonl"))
