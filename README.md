# Multi-GPU test code for virtual tissue

[![CPU smoke tests (x86_64 + arm64)](https://github.com/xyz111131/VirtualTissue/actions/workflows/ci.yml/badge.svg)](https://github.com/xyz111131/VirtualTissue/actions/workflows/ci.yml)

Small, runnable scripts showing how each GPU workload in our proposal
(*Sec. 4.3b, compute table*) will use several Rubin GPUs in parallel on the Huang Complex.
Every script runs in minutes on toy data, so it can be launched as-is on the ARM test node
or an NVL72 rack to check the software stack before the real data and models go in.
The same code runs on CPUs (gloo backend) on x86_64 **and arm64**.

## Steps with parallelization

| Proposal row (Sec. 4.3b) | Script | Parallel pattern | Launcher |
|---|---|---|---|
| Neural posterior inference — large final-round fits | `npe/train_npe_ddp.py` | **Data-parallel DDP**: one flow replica per GPU, simulation set sharded by `DistributedSampler`, gradients averaged by NCCL all-reduce every step | `torchrun` (`slurm/ddp_single_node.sbatch`, `slurm/ddp_multi_node.sbatch`) |
| Neural posterior inference — sequential rounds, seed ensemble | `npe/sbi_job_array.py` | **Job array**: one independent `sbi` fit per GPU (sbi trains on one device); shards pooled into an ensemble posterior | `sbatch --array` (`slurm/job_array_single_gpu.sbatch`) |
| Rules from omics data — Geneformer / Nicheformer on dissociated atlases | `omics/finetune_ddp.py` | **Data-parallel DDP** via Hugging Face `Trainer` (automatic under torchrun) | `torchrun` |
| Rules from omics data — NicheCompass / CellNEST per spatial section | `omics/niche_job_array.py` | **Job array**: one PyTorch Geometric model per section per GPU | `sbatch --array` |
| Rules from the literature — LoRA fine-tuning | `llm/lora_finetune.py` | **Data-parallel DDP** with PEFT/LoRA adapters; a 30B-class model fits one 288 GB Rubin GPU in bf16, so no model parallelism is needed | `torchrun` or `accelerate launch --multi_gpu` |
| Rules from the literature — extraction over 10⁴–10⁵ articles | `llm/extract_sharded.py` | **Sharded inference**: article *i* goes to GPU *i mod W*; no communication, linear scaling; optional vLLM engine | `sbatch --array` (`slurm/llm_extract_array.sbatch`) or `torchrun` |
| (sanity) | `tests/allreduce_check.py` | NCCL all-reduce correctness + bus bandwidth over NVLink | `torchrun` |

The ABM ensemble itself (PhysiCell, 2–4 × 10⁵ runs) is CPU-only on the Vera cores and is
not part of this repository; `npe/simulate.py` is a toy stand-in whose output format
(parameters `theta`, summary statistics `x` tensors) is what the PhysiCell→SBI bridge writes.

Two launch patterns cover everything:

* **`torchrun`** for anything with gradient synchronisation. One launcher per node spawns one
  process per GPU; `huangtests/dist.py` picks NCCL on GPUs and gloo on CPUs and also
  understands `srun`'s environment.
* **Slurm job arrays** for independent fits or inference shards. Scripts read
  `SLURM_ARRAY_TASK_ID`/`SLURM_ARRAY_TASK_COUNT` (or `RANK`/`WORLD_SIZE`) through
  `huangtests.shard_indices`, so the same script also works under torchrun on one node.

## Quick start (any machine, no GPU needed)

```bash
pip install torch            # CPU wheel is enough locally; see "ARM / Huang Complex" below
pip install -r requirements-ci.txt && pip install -e .
pytest -v                    # ~1–2 min: 2-process gloo DDP + emulated job arrays
```

Run a script by hand with two CPU processes:

```bash
torchrun --standalone --nproc_per_node=2 npe/train_npe_ddp.py --n-sims 20000 --epochs 8
# [npe] world_size=2 backend=gloo ... epoch 7 train NLL 2.13 val NLL 2.16
# [npe] posterior-mean RMSE [0.039, 0.216, 0.352]  90% coverage [0.935, 0.935, 0.915]
```

## On the Huang Complex

```bash
# 1. environment: build the Apptainer image once (aarch64 host), or fill in Option B of slurm/env.sh
apptainer build env/huang-arm64.sif env/huang-arm64.def

# 2. collective sanity check on 4 GPUs of one node, then 2 nodes
sbatch slurm/ddp_single_node.sbatch tests/allreduce_check.py
sbatch slurm/ddp_multi_node.sbatch  tests/allreduce_check.py

# 3. the workloads
sbatch slurm/ddp_single_node.sbatch      npe/train_npe_ddp.py --data runs/physicell/sims.pt --epochs 30
sbatch slurm/job_array_single_gpu.sbatch npe/sbi_job_array.py --data runs/physicell/sims.pt
sbatch slurm/ddp_single_node.sbatch      omics/finetune_ddp.py --model-name ctheodoris/Geneformer --data cells.pt
sbatch slurm/job_array_single_gpu.sbatch omics/niche_job_array.py --n-sections 10
sbatch slurm/ddp_single_node.sbatch      llm/lora_finetune.py --model-name <HF id> --data rules_curated.jsonl
sbatch slurm/llm_extract_array.sbatch    --articles corpus.jsonl --model-name <HF id> --adapter runs/llm_lora/adapter --engine vllm
```

`--partition`, `--account` and module names in `slurm/` are placeholders until the
Supercomputer Center publishes the scheduler layout. Every script writes a `metrics.json`
(world size, device, throughput, loss/accuracy) so scaling from 1 → 4 → 8 GPUs can be
tabulated directly.

## ARM (Vera) readiness

* The CI matrix includes GitHub's `ubuntu-24.04-arm` runner, so every push checks that
  PyTorch, PyTorch Geometric, transformers, peft and sbi import and run the distributed
  code paths on aarch64. Nothing in this repo needs a compiled extension: PyG is used
  without `torch-scatter`/`torch-cluster`, and the kNN graph is built with `torch.cdist`.
* `env/huang-arm64.def` wraps NVIDIA's multi-arch PyTorch container (26.09: CUDA 13.4,
  initial Rubin support) so the test node and the NVL72 racks run identical software.
* Known porting item from the proposal (Sec. 4.5b, Q1): CellNEST pins PyTorch 1.13 /
  CUDA 11.7 and older PyG extensions; `omics/niche_job_array.py` is the pattern it will be
  moved onto (plain PyG 2.x, one section per GPU).

## Layout

```
huangtests/dist.py        process-group init (torchrun / srun / job array), NCCL-or-gloo, shard_indices
npe/                      simulate.py (toy ABM stand-in), train_npe_ddp.py, sbi_job_array.py
omics/                    finetune_ddp.py (HF Trainer, Geneformer-style), niche_job_array.py (PyG GAT)
llm/                      lora_finetune.py, extract_sharded.py, common.py, data/ (24 toy sentence→rule pairs, 8 toy articles)
slurm/                    env.sh, ddp_single_node.sbatch, ddp_multi_node.sbatch, job_array_single_gpu.sbatch, llm_extract_array.sbatch
env/huang-arm64.def       Apptainer definition (NVIDIA PyTorch container + Python deps)
tests/                    allreduce_check.py + pytest suite (2-process gloo DDP, emulated Slurm arrays)
.github/workflows/ci.yml  x86_64 + arm64 CI
```

## License

BSD 3-Clause (see `LICENSE`). All project software will be released under open-source
licenses, as stated in the proposal.
