#!/bin/bash
# Static SoS qualification; submit once on each intended architecture.
# Outputs belong to ignored scaffold validation logs, never scientific runs.
#SBATCH --job-name=qualify-static-sos
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the worktree}"
LAUNCH_COMMIT="${1:?Supply the frozen source commit}"
python3 scaffolds/launch/frozen.py check --launch-commit "$LAUNCH_COMMIT" --experiment transformer_pp --scaffold transformer
GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
case "$GPU_NAMES" in
  *H100*) GPU_TYPE=H100 ;;
  *B200*) GPU_TYPE=B200 ;;
  *) echo "Static SoS qualification requires H100 or B200" >&2; exit 1 ;;
esac
export GPU_TYPE
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$PWD/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda,cpu PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
TASK_TMP="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/static-sos-${SLURM_JOB_ID}.XXXXXX")"
trap 'rm -rf -- "$TASK_TMP"' EXIT
export TMPDIR="$TASK_TMP"
export JAX_COMPILATION_CACHE_DIR="$PWD/.cache/transformer_pp/jax"
OUTPUT="scaffolds/transformer/validation/logs/static-sos-${SLURM_JOB_ID}"
mkdir -p "$OUTPUT" "$JAX_COMPILATION_CACHE_DIR"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
"$PYTHON" -m pytest -c scaffolds/transformer/pyproject.toml --import-mode=importlib -m gpu -q \
  scaffolds/transformer/tests/test_mosaic_static_sos.py --junitxml="$OUTPUT/gpu-tests.xml"
"$PYTHON" - "$OUTPUT/gpu-tests.xml" <<'PY'
import sys
from scaffolds.transformer.acceptance import require_passing_tests
print('All GPU qualification cases passed:', require_passing_tests(sys.argv[1]))
PY
