#!/bin/bash
#SBATCH --job-name=ddp-2node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # one torchrun per node
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --partition=rubin            # TODO
#SBATCH --output=slurm-%x-%j.out
#
# Same scripts across several nodes. Inside one NVL72 rack NCCL rides NVLink Switch between
# nodes; across racks it falls back to the inter-rack fabric. Usage identical to the
# single-node script, e.g.
#   sbatch slurm/ddp_multi_node.sbatch npe/train_npe_ddp.py --n-sims 2000000 --epochs 30

set -euo pipefail
source "$(dirname "$0")/env.sh"

export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / SLURM_GPUS_ON_NODE))
export NCCL_DEBUG=INFO

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
export MASTER_ADDR MASTER_PORT

# srun starts one torchrun per node; the rendezvous on MASTER_ADDR joins them into one job of
# nnodes * nproc_per_node ranks.
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -c "
  run_py torchrun --nnodes=$SLURM_NNODES --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT $*"
