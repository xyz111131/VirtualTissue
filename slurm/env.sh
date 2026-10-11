#!/bin/bash
# Environment for the Huang Complex (aarch64 Vera CPUs + Rubin GPUs). Sourced by every sbatch
# script. Pick ONE of the two options below once the Supercomputer Center publishes its
# recommended stack; both define a `run_py` function that the sbatch scripts call (it is
# exported so `srun bash -c` on other nodes sees it too).

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_DIR
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_DIR"

# Option A — Apptainer image built from env/huang-arm64.def (NVIDIA's multi-arch PyTorch
# container with CUDA, NCCL and an aarch64 PyTorch build). Preferred: identical software
# on the ARM test node and the NVL72 racks.
export HUANG_IMAGE="${HUANG_IMAGE:-$REPO_DIR/env/huang-arm64.sif}"
if [[ -f "$HUANG_IMAGE" ]]; then
  run_py() { apptainer exec --nv --bind "$REPO_DIR" "$HUANG_IMAGE" "$@"; }
else
  # Option B — site modules + a venv with aarch64 wheels (torch publishes linux_aarch64 CUDA
  # wheels; PyG, transformers, peft, sbi are pure Python).
  # module load cuda nccl python          # TODO: module names on the Huang Complex
  # source "$REPO_DIR/.venv/bin/activate"
  run_py() { "$@"; }
fi
export -f run_py
