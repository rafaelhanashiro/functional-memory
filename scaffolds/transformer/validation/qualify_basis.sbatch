#!/bin/bash
# Synthetic RBF correctness and attention timing; no dataset or training run is opened.
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:b300:1
#SBATCH --time=01:00:00
#SBATCH --open-mode=append
set -euo pipefail
SOURCE="${1:?Expected committed source}"
OUTPUT="${2:?Expected validation record path}"
shift 2
cd "${SLURM_SUBMIT_DIR:?Submit from the experiment worktree}"
python3 scaffolds/launch/frozen.py check --launch-commit "$SOURCE" --experiment attention_scaling --scaffold transformer
git diff --exit-code "$SOURCE" -- scaffolds/transformer/validation/qualify_basis.py scaffolds/transformer/validation/qualify_basis.sbatch
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$PWD/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1 TRANSFORMER_CLUSTER=kennel
export SHARED_TRANSFORMER_SOURCE_COMMIT="$SOURCE"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
BASIS_JOB_TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/rbf-qualification-${SLURM_JOB_ID}.XXXXXX")"
export TMPDIR="$BASIS_JOB_TMPDIR"
trap 'rm -rf -- "$BASIS_JOB_TMPDIR"' EXIT
CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
export JAX_COMPILATION_CACHE_DIR="$PWD/.cache/rbf-qualification/jax"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$PYTHON" -m scaffolds.transformer.validation.qualify_basis --configs "$@" --output "$OUTPUT"
