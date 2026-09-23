#!/bin/bash
set -euo pipefail
MANIFEST="${1:?Expected canonical manifest}"
MODE="${2:?Expected preflight or train}"
RUN_NAME="${3:?Expected run name}"
if [[ "$MODE" != preflight && "$MODE" != train ]]; then
  echo "Expected preflight or train" >&2
  exit 1
fi
# The caller supplies the dedicated locked interpreter, never a running
# experiment's shared environment. The scheduler inherits this export.
PYTHON="${GLA_PYTHON:?Export the absolute dedicated locked Python path}"
mapfile -t LOCATIONS < <("$PYTHON" - "$MANIFEST" <<'PY'
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
print(manifest['execution_root'])
print(manifest['artifact_root'])
PY
)
EXECUTION_ROOT="${LOCATIONS[0]}"
ARTIFACT_ROOT="${LOCATIONS[1]}"
cd "$EXECUTION_ROOT"
export PYTHONPATH="${EXECUTION_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export JAX_COMPILATION_CACHE_DIR="${ARTIFACT_ROOT}/.cache/gla_key_dimension/jax"
export EJKERNEL_TRITON_CACHE_DIR="${ARTIFACT_ROOT}/.cache/gla_key_dimension/triton"
export TMPDIR="${ARTIFACT_ROOT}/.cache/gla_key_dimension/tmp/${SLURM_JOB_ID:-manual}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR" "$EJKERNEL_TRITON_CACHE_DIR" "$TMPDIR"
source "${EXECUTION_ROOT}/scaffolds/transformer/launchers/kennel_runtime.sh"
exec "$PYTHON" -m scaffolds.transformer.studies.gla_key_dimension.sweep "$MODE" \
  --manifest "$MANIFEST" --run-name "$RUN_NAME"
