#!/bin/bash
#SBATCH --job-name=learned_quadratic_attention
#SBATCH --partition=standard
#SBATCH --nodelist=dogo
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:h100:4
set -euo pipefail

MODE="${1:?Expected preflight, preflight-and-submit, or train}"
MANIFEST="${2:?Expected sweep manifest}"
RUN_NAME="${3:-preflight}"
case "$MANIFEST" in
  /*/results/learned_quadratic_attention/sweep.json) PROJECT_ROOT="${MANIFEST%/results/learned_quadratic_attention/sweep.json}" ;;
  *) echo "Expected absolute canonical results/learned_quadratic_attention/sweep.json path: $MANIFEST" >&2; exit 1 ;;
esac
PYTHON="${PROJECT_ROOT}/.envs/polynomial_attention/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing locked polynomial-attention interpreter: $PYTHON" >&2
  exit 1
fi
BOOTSTRAP_OUTPUT="$("$PYTHON" - "$MANIFEST" "$RUN_NAME" "$PROJECT_ROOT" <<'PY'
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
if Path(manifest['project_root']).resolve() != Path(sys.argv[3]).resolve():
    raise ValueError('Manifest project_root does not match its canonical location')
print(manifest['execution_root'])
print(manifest['source_commit'])
run = next((run for run in manifest['runs'] if run['name'] == sys.argv[2]), None)
print(run['config_path'] if run else '')
PY
)"
mapfile -t LOCATIONS <<< "$BOOTSTRAP_OUTPUT"
EXECUTION_ROOT="${LOCATIONS[0]}"
EXPECTED_COMMIT="${LOCATIONS[1]}"
CONFIG_PATH="${LOCATIONS[2]:-}"
LOG_DIR="${PROJECT_ROOT}/results/logs/learned_quadratic_attention/${RUN_NAME}"
CACHE_ROOT="${PROJECT_ROOT}/.cache/learned_quadratic_attention"
mkdir -p "$LOG_DIR" "$CACHE_ROOT/jax" "$CACHE_ROOT/tmp"
exec >>"${LOG_DIR}/slurm-${SLURM_JOB_ID}.out" 2>&1
cd "$EXECUTION_ROOT"
if [[ "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" ]] || [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "Frozen execution checkout changed; refusing to run: $EXECUTION_ROOT" >&2
  exit 1
fi
export POLYNOMIAL_ATTENTION_SOURCE_COMMIT="$EXPECTED_COMMIT"
export JAX_COMPILATION_CACHE_DIR="${CACHE_ROOT}/jax"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TMPDIR="${CACHE_ROOT}/tmp"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export PYTHONPATH="${EXECUTION_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg
echo "mode=$MODE run=$RUN_NAME slurm_job_id=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} source_commit=$EXPECTED_COMMIT"
if [[ "$MODE" == preflight || "$MODE" == preflight-and-submit ]]; then
  "$PYTHON" -m experiments.learned_quadratic_attention.preflight "$MANIFEST"
  "$PYTHON" -m experiments.learned_quadratic_attention.plot --manifest "$MANIFEST"
  if [[ "$MODE" == preflight-and-submit ]]; then
    "$PYTHON" -m experiments.learned_quadratic_attention.sweep --submit \
      --project-root "$PROJECT_ROOT" --execution-root "$EXECUTION_ROOT"
  fi
elif [[ "$MODE" == train ]]; then
  if [[ -z "$CONFIG_PATH" ]]; then echo "Unknown run: $RUN_NAME" >&2; exit 1; fi
  "$PYTHON" -m experiments.learned_quadratic_attention.preflight "$MANIFEST" \
    --check-only --check-hardware --run-name "$RUN_NAME"
  TRAIN_STATUS=0
  "$PYTHON" -m experiments.polynomial_attention.train "$CONFIG_PATH" || TRAIN_STATUS=$?
  "$PYTHON" -m experiments.learned_quadratic_attention.plot --manifest "$MANIFEST"
  if [[ "$TRAIN_STATUS" == 0 ]]; then
    "$PYTHON" -m experiments.learned_quadratic_attention.retention --manifest "$MANIFEST" --run-name "$RUN_NAME"
  fi
  exit "$TRAIN_STATUS"
else
  echo "Unknown mode: $MODE" >&2
  exit 1
fi
