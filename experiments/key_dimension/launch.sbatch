#!/bin/bash
#SBATCH --job-name=key_dimension
#SBATCH --partition=mit_preemptable
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gpus=h200:2
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
set -euo pipefail

MODE="${1:?Expected preflight or train}"
MANIFEST="${2:?Expected sweep manifest}"
RUN_NAME="${3:-preflight}"
# Use the standard-library interpreter only to locate the isolated uv project.
mapfile -t LOCATIONS < <(python3 - "$MANIFEST" "$RUN_NAME" <<'PY'
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
print(manifest['project_root'])
print(manifest['execution_root'])
print(manifest['source_commit'])
run = next((run for run in manifest['runs'] if run['name'] == sys.argv[2]), None)
print(run['config_path'] if run else '')
PY
)
PROJECT_ROOT="${LOCATIONS[0]}"
EXECUTION_ROOT="${LOCATIONS[1]}"
EXPECTED_COMMIT="${LOCATIONS[2]}"
CONFIG_PATH="${LOCATIONS[3]:-}"
LOG_DIR="${PROJECT_ROOT}/results/logs/key_dimension/${RUN_NAME}"
CACHE_ROOT="${PROJECT_ROOT}/.cache/key_dimension"
mkdir -p "$LOG_DIR" "$CACHE_ROOT/uv" "$CACHE_ROOT/jax" "$CACHE_ROOT/tmp" "${PROJECT_ROOT}/.envs"
exec >>"${LOG_DIR}/slurm-${SLURM_JOB_ID}.out" 2>&1
cd "$EXECUTION_ROOT"
if [[ "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" ]] || [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "Frozen execution checkout changed; refusing to run: $EXECUTION_ROOT" >&2
  exit 1
fi
export KEY_DIMENSION_SOURCE_COMMIT="$EXPECTED_COMMIT"
export UV_PROJECT_ENVIRONMENT="${PROJECT_ROOT}/.envs/key_dimension"
export UV_CACHE_DIR="${CACHE_ROOT}/uv"
export UV_LINK_MODE=copy
export JAX_COMPILATION_CACHE_DIR="${CACHE_ROOT}/jax"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TMPDIR="${CACHE_ROOT}/tmp"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="${EXECUTION_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg
UV=(uv run --project "${EXECUTION_ROOT}/experiments/key_dimension" --locked --extra cuda)
echo "mode=$MODE run=$RUN_NAME slurm_job_id=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} source_commit=$EXPECTED_COMMIT"
if [[ "$MODE" == preflight ]]; then
  "${UV[@]}" python -m experiments.key_dimension.preflight "$MANIFEST"
elif [[ "$MODE" == train ]]; then
  if [[ -z "$CONFIG_PATH" ]]; then echo "Unknown run: $RUN_NAME" >&2; exit 1; fi
  "${UV[@]}" python -m experiments.key_dimension.preflight "$MANIFEST" --check-only
  "${UV[@]}" python -m experiments.key_dimension.train "$CONFIG_PATH"
  "${UV[@]}" python -m experiments.key_dimension.plot --manifest "$MANIFEST"
else
  echo "Unknown mode: $MODE" >&2
  exit 1
fi
