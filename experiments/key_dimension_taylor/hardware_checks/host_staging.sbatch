#!/usr/bin/env bash
#SBATCH --job-name=dkm-a6000-host-check
#SBATCH --partition=sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
set -euo pipefail
# Explicit CLI carries immutable policy, commit, selected names and isolated
# output root. This launcher contains no production scheduling path.
HOST_STAGING_ROOT="${HOST_STAGING_RUNTIME_ROOT:?Set the frozen runtime checkout}"
HOST_STAGING_PYTHON="${HOST_STAGING_PYTHON:?Set the pinned Python executable}"
export NCCL_P2P_DISABLE=1
export POLYNOMIAL_ATTENTION_HOST_STAGING=1
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
export PYTHONPATH="$HOST_STAGING_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
cd "$HOST_STAGING_ROOT"
exec "$HOST_STAGING_PYTHON" experiments/key_dimension_taylor/hardware_checks/host_staging.py "$@"
