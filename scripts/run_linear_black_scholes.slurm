#!/usr/bin/env bash
# Adapt resources and environment activation to the target cluster.
# Dimensions and independent runs execute sequentially on one visible GPU.
#SBATCH --job-name=ineus-linear-bs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

CONFIG="${CONFIG:-configs/linear_black_scholes.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-results/linear_black_scholes_${SLURM_JOB_ID}}"

cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the repository root}"

# Cluster-specific example:
# source /path/to/miniconda3/etc/profile.d/conda.sh
# conda activate ineus

args=(--config "${CONFIG}" --output-dir "${OUTPUT_DIR}")
if [[ -n "${SEED:-}" ]]; then
  args+=(--seed "${SEED}")
fi
if [[ -n "${RUNS:-}" ]]; then
  args+=(--runs "${RUNS}")
fi
if [[ -n "${EPOCHS:-}" ]]; then
  args+=(--epochs "${EPOCHS}")
fi

export MPLBACKEND=Agg
srun python -u -m ineus "${args[@]}"
