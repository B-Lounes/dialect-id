#!/usr/bin/env bash
#SBATCH --job-name=dialect-id-train
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -euo pipefail

: "${DATASET_ROOT:?export DATASET_ROOT to the clean Parquet dataset}"
: "${RUN_DIR:?export RUN_DIR to an external run directory}"
: "${MODEL_TYPE:?export MODEL_TYPE, for example wavlm}"

export REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
export NNODES="${SLURM_NNODES:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}"
export MASTER_PORT="${MASTER_PORT:-29500}"
if [[ -z "${MASTER_ADDR:-}" ]]; then
  export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
fi

srun --ntasks="$NNODES" --ntasks-per-node=1 \
  bash "$REPO_ROOT/recipes/slurm/run_node_torchrun.sh"
