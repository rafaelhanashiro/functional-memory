#!/usr/bin/env bash
# Submit from a dedicated worktree; resource choices are supplied to sbatch.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the worktree root}"
LAUNCH_COMMIT="${1:?Pass the committed benchmark source}"
shift
git diff --exit-code "$LAUNCH_COMMIT" -- scaffolds/transformer ':!scaffolds/transformer/validation'
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$PWD/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
TASK_TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/query-key-tuning-${SLURM_JOB_ID}.XXXXXX")"
trap 'rm -rf -- "$TASK_TMPDIR"' EXIT
export TMPDIR="$TASK_TMPDIR"
export JAX_COMPILATION_CACHE_DIR="$PWD/.cache/query-key-tuning/jax"
export XDG_CACHE_HOME="$PWD/.cache/query-key-tuning/xdg"
CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export CUDA_HOME
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas"
export TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
export PATH="$CUDA_HOME/bin:$PATH"
printf 'job=%s source=%s environment=%s\n' "$SLURM_JOB_ID" "$LAUNCH_COMMIT" "$TRANSFORMER_CUDA_ENVIRONMENT"
nvidia-smi --query-gpu=name,uuid,temperature.gpu,clocks.current.sm,power.limit --format=csv
"$PYTHON" -u -m scaffolds.transformer.benchmarks.tune_query_key "$@"
if [[ "${QUERY_KEY_QUALIFY:-0}" == 1 ]]; then
  "$PYTHON" -m pytest -q scaffolds/transformer/tests/test_query_key_tiling.py
fi
