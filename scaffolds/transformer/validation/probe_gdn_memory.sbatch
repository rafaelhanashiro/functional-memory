#!/bin/bash
# Diagnose GDN allocator settings with one sensitive kernel test and a full-recipe
# two-GPU pilot. This is not full acceptance and never launches production training.
#
#   sbatch ... probe_gdn_memory.sbatch experiments/attention_scaling/runs/<run>/config.yaml <source> [H100] [all|pilot]
# The pilot-only mode skips the kernel test and is diagnostic-only.
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:h100:2
#SBATCH --time=00:40:00
set -euo pipefail
CONFIG="${1:?Expected experiments/attention_scaling/runs/<run>/config.yaml}"
LAUNCH_COMMIT="${2:?Expected committed source}"
GPU_TYPE="${3:-H100}"
PROBE_STAGE="${4:-all}"
case "$PROBE_STAGE" in
  all|pilot) ;;
  *) echo "Expected probe stage all or pilot, got: $PROBE_STAGE" >&2; exit 1 ;;
esac
EXECUTION_ROOT="${SLURM_SUBMIT_DIR:?Submit from the experiment worktree root}"
ADAPTER=experiments.attention_scaling.acceptance
cd "$EXECUTION_ROOT"
if [[ ! "$CONFIG" =~ ^experiments/attention_scaling/runs/(r[0-9a-f]{12})/config\.yaml$ ]]; then
  echo "Expected a repository-relative attention recipe: $CONFIG" >&2
  exit 1
fi
RUN="${BASH_REMATCH[1]}"
RUN_ROOT="experiments/attention_scaling/runs/$RUN"
python3 scaffolds/launch/frozen.py check --launch-commit "$LAUNCH_COMMIT" --experiment attention_scaling --scaffold transformer
git cat-file -e "$LAUNCH_COMMIT:$CONFIG"
PYTHON="$EXECUTION_ROOT/scaffolds/transformer/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing the worktree environment; sync its locked CUDA environment first" >&2
  exit 1
fi

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
source scaffolds/transformer/launchers/kennel_runtime.sh
export TRANSFORMER_HOST_STAGING="${TRANSFORMER_HOST_STAGING:-0}"
export SHARED_TRANSFORMER_SOURCE_COMMIT="$LAUNCH_COMMIT"
export TRANSFORMER_CLUSTER=kennel TRANSFORMER_GPU_TYPE="$GPU_TYPE"
export PYTHONPATH="$EXECUTION_ROOT" JAX_PLATFORMS=cuda
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
CACHE="$EXECUTION_ROOT/.cache/attention_scaling"
export JAX_COMPILATION_CACHE_DIR="$CACHE/jax" XDG_CACHE_HOME="$CACHE/xdg"
GDN_PROBE_TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/gdn-memory-${SLURM_JOB_ID}.XXXXXX")"
export TMPDIR="$GDN_PROBE_TMPDIR"
trap 'rm -rf -- "$GDN_PROBE_TMPDIR"' EXIT
CUDA_HOME="$("$PYTHON" -c 'import importlib.util; print(next(iter(importlib.util.find_spec("nvidia.cuda_nvcc").submodule_search_locations)))')"
export CUDA_HOME
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
eval "$("$PYTHON" -m "$ADAPTER" "$CONFIG" --runtime-env)"
PROBE_DIRECTORY="$RUN_ROOT/.runtime/memory-probe/$GPU_TYPE-2-$SLURM_JOB_ID"
VALIDATION_DIRECTORY="scaffolds/transformer/validation/runs/gdn-independent-precision-${GPU_TYPE,,}"
JUNIT="scaffolds/transformer/validation/logs/gdn-memory-$SLURM_JOB_ID.xml"
OUTPUT="$VALIDATION_DIRECTORY/memory-probe-$SLURM_JOB_ID.json"
mkdir -p "$PROBE_DIRECTORY" "$VALIDATION_DIRECTORY" scaffolds/transformer/validation/logs \
  "$JAX_COMPILATION_CACHE_DIR" "$XDG_CACHE_HOME"
"$PYTHON" - "$LAUNCH_COMMIT" "${BASH_SOURCE[0]}" "$PROBE_STAGE" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from scaffolds.transformer.acceptance import runtime_identity

print(json.dumps({
    "diagnostic": "gdn_memory_probe", "launch_source_commit": sys.argv[1],
    "probe_stage": sys.argv[3], "diagnostic_only": True,
    "checkout_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "executed_script_sha256": hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest(),
    "slurm_job_id": os.environ["SLURM_JOB_ID"], "node": os.environ.get("SLURMD_NODENAME"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "runtime": runtime_identity(),
}, sort_keys=True), flush=True)
PY

if [[ "$PROBE_STAGE" == all ]]; then
  GDN_PROBE_FIRST_GPU="${CUDA_VISIBLE_DEVICES:-0}"
  GDN_PROBE_FIRST_GPU="${GDN_PROBE_FIRST_GPU%%,*}"
  echo "memory_probe_stage=kernel_test timeout=8m first_visible_gpu=$GDN_PROBE_FIRST_GPU"
  CUDA_VISIBLE_DEVICES="$GDN_PROBE_FIRST_GPU" timeout --signal=INT --kill-after=60s 8m \
    "$PYTHON" -m pytest -c scaffolds/transformer/pyproject.toml -q -s -x --import-mode=importlib \
    'scaffolds/transformer/tests/test_gated_deltanet_kernel.py::test_fused_output_and_all_gradients_match_the_recurrence[float32-0.005-64]' \
    --junitxml="$JUNIT"
  "$PYTHON" - "$JUNIT" <<'PY'
from pathlib import Path
import sys
from scaffolds.transformer.acceptance import require_passing_tests

if require_passing_tests(Path(sys.argv[1])) != 1:
    raise SystemExit("Expected exactly one passing kernel test")
PY
else
  echo "memory_probe_stage=kernel_test skipped=true diagnostic_only=true reason=pilot_only_mode"
fi

echo "memory_probe_stage=full_recipe_pilot timeout=30m directory=$PROBE_DIRECTORY"
timeout --signal=INT --kill-after=60s 30m \
  "$PYTHON" -m "$ADAPTER" "$CONFIG" --worker pilot --directory "$PROBE_DIRECTORY" \
  --gpu-count 2 --gpu-type "$GPU_TYPE" --output "$OUTPUT" --stop-after-step 20
# Keep the pilot's metrics and checkpoint for inspection after either outcome.
echo "memory_probe_pilot_output=$OUTPUT"
