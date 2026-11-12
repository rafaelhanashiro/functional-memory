#!/bin/bash
# Submit from the worktree; choose partition and GPU allocation explicitly.
# Usage: sbatch [resources] runtime_estimate.sbatch COMMIT CONFIG OUTPUT_JSON
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --open-mode=append
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
LAUNCH_COMMIT="${1:?Expected committed source}"
RECIPE="${2:?Expected input recipe}"
REPORT="${3:?Expected qualification JSON path}"
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$PWD/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS=1
TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/runtime-estimate-${SLURM_JOB_ID}.XXXXXX")"
trap 'rm -rf "$TMPDIR"' EXIT
export TMPDIR
export JAX_COMPILATION_CACHE_DIR="$PWD/scaffolds/transformer/validation/logs/runtime-estimate-${SLURM_JOB_ID}/jax-cache"
"$PYTHON" -m scaffolds.transformer.validation.runtime_estimate \
  --source-commit "$LAUNCH_COMMIT" --config "$RECIPE" --output "$REPORT"
