#!/bin/bash
#SBATCH --job-name=exp-attn-250m
#SBATCH --partition=mit_preemptable
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gpus=h200:2
set -euo pipefail
MODE="${1:?Expected cpu-tests, preflight, or train}"
EXECUTION_ROOT="${2:?Expected absolute experiment checkout}"
PROJECT_ROOT="${3:?Expected absolute storage project root}"
EXPECTED_COMMIT="${4:?Expected frozen source commit}"
GPU_COUNT="${5:-2}"
GPU_TYPE="${6:-H200}"
cd "$EXECUTION_ROOT"
if [[ "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" ]] || [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "Frozen execution checkout changed: $EXECUTION_ROOT" >&2
  exit 1
fi
export UNNORMALIZED_ATTENTION_SOURCE_COMMIT="$EXPECTED_COMMIT"
export UV_PROJECT_ENVIRONMENT="${PROJECT_ROOT}/.envs/unnormalized_attention"
export UV_CACHE_DIR="${PROJECT_ROOT}/.cache/uv"
export UV_LINK_MODE=copy
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export PYTHONPATH="$EXECUTION_ROOT"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
CACHE_ROOT="${PROJECT_ROOT}/.cache/unnormalized_attention"
export JAX_COMPILATION_CACHE_DIR="${CACHE_ROOT}/jax"
export TMPDIR="${CACHE_ROOT}/tmp/${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
mkdir -p "$TMPDIR" "$JAX_COMPILATION_CACHE_DIR" "$XDG_CACHE_HOME"
CONFIG="${EXECUTION_ROOT}/experiments/unnormalized_attention/configs/base.yaml"
CHECKS="${CACHE_ROOT}/preflight/${EXPECTED_COMMIT}-${GPU_TYPE}-${GPU_COUNT}"
echo "mode=$MODE source=$EXPECTED_COMMIT job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0}"
RUN=(uv run --project experiments/unnormalized_attention --locked --extra cuda --no-sync python)
if [[ "$MODE" == cpu-tests ]]; then
  export JAX_PLATFORMS=cpu
  "${RUN[@]}" -m pytest -c experiments/unnormalized_attention/pyproject.toml experiments/unnormalized_attention/tests -m 'not gpu' -q
elif [[ "$MODE" == preflight ]]; then
  "${RUN[@]}" -m experiments.unnormalized_attention.preflight "$CONFIG" --directory "$CHECKS" --gpu-count "$GPU_COUNT" --gpu-type "$GPU_TYPE"
elif [[ "$MODE" == train ]]; then
  "${RUN[@]}" -m experiments.unnormalized_attention.preflight "$CONFIG" --directory "$CHECKS" --gpu-count "$GPU_COUNT" --gpu-type "$GPU_TYPE" --check-only
  "${RUN[@]}" -m experiments.unnormalized_attention.train "$CONFIG"
else
  echo "Unknown mode: $MODE" >&2
  exit 1
fi
