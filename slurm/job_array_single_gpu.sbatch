#!/bin/bash
#SBATCH --job-name=gpu-array
#SBATCH --array=0-7                  # 8 independent tasks -> 8 GPUs (any number of nodes)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=rubin            # TODO
#SBATCH --output=slurm-%x-%A_%a.out
#
# Embarrassingly parallel sweeps: one GPU per array task, no inter-GPU communication.
#   sbatch slurm/job_array_single_gpu.sbatch npe/sbi_job_array.py --n-sims 200000       # NPE ensemble / rounds
#   sbatch slurm/job_array_single_gpu.sbatch omics/niche_job_array.py --n-sections 10   # one section per GPU
# The scripts read SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT to pick their share of the work.

set -euo pipefail
source "$(dirname "$0")/env.sh"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

run_py python "$@"
