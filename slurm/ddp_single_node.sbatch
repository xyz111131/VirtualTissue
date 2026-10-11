#!/bin/bash
#SBATCH --job-name=ddp-1node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # one torchrun launcher per node; it spawns one process per GPU
#SBATCH --gpus-per-node=4            # Rubin GPUs on this node
#SBATCH --cpus-per-task=16           # Vera cores for data loading
#SBATCH --time=01:00:00
#SBATCH --partition=rubin            # TODO: partition / account names from the Supercomputer Center
#SBATCH --output=slurm-%x-%j.out
#
# Data-parallel training on all GPUs of one node. Usage:
#   sbatch slurm/ddp_single_node.sbatch npe/train_npe_ddp.py --n-sims 200000 --epochs 30
#   sbatch slurm/ddp_single_node.sbatch omics/finetune_ddp.py --model-name ctheodoris/Geneformer --data cells.pt
#   sbatch slurm/ddp_single_node.sbatch llm/lora_finetune.py --model-name <HF id> --data rules_curated.jsonl
#   sbatch slurm/ddp_single_node.sbatch tests/allreduce_check.py

set -euo pipefail
source "$(dirname "$0")/env.sh"      # container / module setup (aarch64 PyTorch + NCCL)

export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / SLURM_GPUS_ON_NODE))
export NCCL_DEBUG=INFO               # prints the NVLink/NVSwitch topology NCCL picked; drop once verified

run_py torchrun --standalone --nnodes=1 --nproc_per_node="$SLURM_GPUS_ON_NODE" "$@"
