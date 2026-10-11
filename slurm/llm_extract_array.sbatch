#!/bin/bash
#SBATCH --job-name=llm-extract
#SBATCH --array=0-15                 # 16 shards of the article corpus -> 16 GPUs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=rubin            # TODO
#SBATCH --output=slurm-%x-%A_%a.out
#
# Rule extraction over the PMC corpus: each task loads the fine-tuned model on its own
# GPU and processes every 16th article. Throughput scales linearly with the array size
# because there is no communication between shards.
#   sbatch slurm/llm_extract_array.sbatch --articles corpus.jsonl --model-name <HF id> \
#          --adapter runs/llm_lora/adapter --engine vllm --out runs/llm_extract
# Afterwards:  cat runs/llm_extract/rules_shard*.jsonl > runs/llm_extract/rules_all.jsonl

set -euo pipefail
source "$(dirname "$0")/env.sh"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

run_py python llm/extract_sharded.py "$@"
