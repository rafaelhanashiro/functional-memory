#!/bin/bash
#SBATCH --job-name=key_dimension
#SBATCH --partition=mit_preemptable
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:2
set -euo pipefail

MODE="${1:?Expected preflight or train}"
MANIFEST="${2:?Expected sweep manifest}"
RUN_NAME="${3:-preflight}"
# sweep.py precreates the retained log directory and gives Slurm both output
# paths, so errors here are captured before Python or the environment starts.
case "$MANIFEST" in
  /*/results/key_dimension/*.json) PROJECT_ROOT="${MANIFEST%/results/key_dimension/*}" ;;
  *) echo "Expected an absolute manifest JSON path under results/key_dimension: $MANIFEST" >&2; exit 1 ;;
esac
BOOTSTRAP_PYTHON="${PROJECT_ROOT}/.envs/shared_transformer/bin/python"
if [[ ! -x "$BOOTSTRAP_PYTHON" ]]; then
  echo "Missing shared Transformer interpreter: $BOOTSTRAP_PYTHON" >&2
  exit 1
fi
BOOTSTRAP_OUTPUT="$("$BOOTSTRAP_PYTHON" - "$MANIFEST" "$RUN_NAME" "$PROJECT_ROOT" <<'PY'
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
if Path(manifest['project_root']).resolve() != Path(sys.argv[3]).resolve():
    raise ValueError('Manifest project_root does not match its canonical location')
print(manifest['project_root'])
print(manifest['execution_root'])
print(manifest['source_commit'])
run = next((run for run in manifest['runs'] if run['name'] == sys.argv[2]), None)
print(run['config_path'] if run else '')
PY
)"
mapfile -t LOCATIONS <<< "$BOOTSTRAP_OUTPUT"
PROJECT_ROOT="${LOCATIONS[0]}"
EXECUTION_ROOT="${LOCATIONS[1]}"
EXPECTED_COMMIT="${LOCATIONS[2]}"
CONFIG_PATH="${LOCATIONS[3]:-}"
LOG_DIR="${PROJECT_ROOT}/logs/key_dimension/${RUN_NAME}"
CACHE_ROOT="${PROJECT_ROOT}/.cache/key_dimension"
mkdir -p "$LOG_DIR" "$CACHE_ROOT/uv" "$CACHE_ROOT/jax" "$CACHE_ROOT/tmp" "${PROJECT_ROOT}/.envs"
exec >>"${LOG_DIR}/slurm-${SLURM_JOB_ID}.out" 2>&1
cd "$EXECUTION_ROOT"
if [[ "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" ]] || [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "Frozen execution checkout changed; refusing to run: $EXECUTION_ROOT" >&2
  exit 1
fi
export KEY_DIMENSION_SOURCE_COMMIT="$EXPECTED_COMMIT"
export UV_PROJECT_ENVIRONMENT="${PROJECT_ROOT}/.envs/shared_transformer"
export UV_CACHE_DIR="${CACHE_ROOT}/uv"
export UV_LINK_MODE=copy
export JAX_COMPILATION_CACHE_DIR="${CACHE_ROOT}/jax"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TMPDIR="${CACHE_ROOT}/tmp"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export PYTHONPATH="${EXECUTION_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg
UV=(uv run --project "${EXECUTION_ROOT}/scaffolds/transformer" --locked --extra cuda)
echo "mode=$MODE run=$RUN_NAME slurm_job_id=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} source_commit=$EXPECTED_COMMIT"
if [[ "$MODE" == preflight ]]; then
  "${UV[@]}" python -m scaffolds.transformer.studies.key_dimension.preflight "$MANIFEST"
elif [[ "$MODE" == train ]]; then
  if [[ -z "$CONFIG_PATH" ]]; then echo "Unknown run: $RUN_NAME" >&2; exit 1; fi
  "${UV[@]}" python -m scaffolds.transformer.studies.key_dimension.preflight "$MANIFEST" --check-only
  "${UV[@]}" python -m scaffolds.transformer.studies.key_dimension.train "$CONFIG_PATH"
  "${UV[@]}" python -m scaffolds.transformer.studies.key_dimension.plot --manifest "$MANIFEST"
else
  echo "Unknown mode: $MODE" >&2
  exit 1
fi
