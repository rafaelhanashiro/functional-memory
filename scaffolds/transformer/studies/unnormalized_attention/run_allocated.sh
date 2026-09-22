#!/bin/bash
# Run inside one Slurm GPU allocation; scheduler resource flags belong to the
# cluster's submission script, not this portable execution entry point.
set -euo pipefail
CONFIG="$(realpath "${1:?Expected absolute config YAML path}")"
STATE_ROOT="$(realpath -m "${2:?Expected cache and preflight directory outside the checkout}")"
GPU_COUNT="${3:-4}"
GPU_TYPE="${4:-H200}"
PYTHON="${5:-python}"
: "${SLURM_JOB_ID:?Run inside a Slurm GPU allocation}"
EXECUTION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$EXECUTION_ROOT"
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo 'Commit configuration/code changes before launching the frozen checkout.' >&2
  exit 1
fi
SOURCE_COMMIT="$(git rev-parse HEAD)"
CONFIG_SHA="$(sha256sum "$CONFIG" | cut -d ' ' -f 1)"
export SHARED_TRANSFORMER_SOURCE_COMMIT="$SOURCE_COMMIT"
export PYTHONPATH="$EXECUTION_ROOT"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export JAX_COMPILATION_CACHE_DIR="${STATE_ROOT}/jax"
export XDG_CACHE_HOME="${STATE_ROOT}/xdg"
export TMPDIR="${STATE_ROOT}/tmp/${SLURM_JOB_ID}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"
CHECKS="${STATE_ROOT}/preflight/${SOURCE_COMMIT}-${CONFIG_SHA}-${GPU_TYPE}-${GPU_COUNT}"
echo "job=$SLURM_JOB_ID source=$SOURCE_COMMIT expected_gpus=$GPU_COUNT expected_type=$GPU_TYPE"
echo "slurm_job_gpus=${SLURM_JOB_GPUS:-unset} cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
# Acceptance and production share the same allocation. Matching passed stages
# are reused on requeue, and training automatically resumes its own workdir.
"$PYTHON" -m scaffolds.transformer.studies.unnormalized_attention.preflight "$CONFIG" \
  --directory "$CHECKS" --gpu-count "$GPU_COUNT" --gpu-type "$GPU_TYPE"
exec "$PYTHON" -m scaffolds.transformer.train "$CONFIG"
