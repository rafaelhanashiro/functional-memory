#!/bin/bash
# Qualify the full GDN recipe without starting production training.
# Submit from the experiment worktree; production must wait for this job to finish
# before using the same acceptance directory.
#
#   sbatch ... qualify_gdn_runtime.sbatch experiments/attention_scaling/runs/<run>/config.yaml <source> [H100] [production-hours=24] [gpu-count=2]
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:h100:2
#SBATCH --time=01:00:00
set -euo pipefail
CONFIG="${1:?Expected experiments/attention_scaling/runs/<run>/config.yaml}"
LAUNCH_COMMIT="${2:?Expected committed source}"
GPU_TYPE="${3:-H100}"
PRODUCTION_HOURS="${4:-24}"
GPU_COUNT="${5:-2}"
if [[ ! "$GPU_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "Expected a positive GPU count: $GPU_COUNT" >&2
  exit 1
fi
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

# Match the production launcher's default; preserve explicit allocator settings.
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
LOGS="$RUN_ROOT/.runtime"
export JAX_COMPILATION_CACHE_DIR="$CACHE/jax" XDG_CACHE_HOME="$CACHE/xdg"
GDN_RUNTIME_TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/gdn-runtime-${SLURM_JOB_ID}.XXXXXX")"
export TMPDIR="$GDN_RUNTIME_TMPDIR"
trap 'rm -rf -- "$GDN_RUNTIME_TMPDIR"' EXIT
CUDA_HOME="$("$PYTHON" -c 'import importlib.util; print(next(iter(importlib.util.find_spec("nvidia.cuda_nvcc").submodule_search_locations)))')"
export CUDA_HOME
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
eval "$("$PYTHON" -m "$ADAPTER" "$CONFIG" --runtime-env)"
mkdir -p "$LOGS" "$JAX_COMPILATION_CACHE_DIR" "$XDG_CACHE_HOME"
echo "runtime_qualification run=$RUN job=$SLURM_JOB_ID launch_commit=$LAUNCH_COMMIT gpu_type=$GPU_TYPE gpu_count=$GPU_COUNT preallocate=$XLA_PYTHON_CLIENT_PREALLOCATE memory_fraction=${XLA_PYTHON_CLIENT_MEM_FRACTION:-default}"
# The acceptance hardware worker verifies the allocated GPU count and type.
# It also checks initialization, full-recipe training/evaluation, and exact resume.
"$PYTHON" -m "$ADAPTER" "$CONFIG" --directory "$LOGS/acceptance/$GPU_TYPE-$GPU_COUNT" \
  --gpu-count "$GPU_COUNT" --gpu-type "$GPU_TYPE" --test-cache "$CACHE/acceptance-tests"
# Acceptance's generic limit is 48 hours; enforce the actual planned allocation.
"$PYTHON" - "$LOGS/acceptance/$GPU_TYPE-$GPU_COUNT/pilot.json" "$PRODUCTION_HOURS" <<'PY'
import json
import math
from pathlib import Path
import sys

pilot = json.loads(Path(sys.argv[1]).read_text())
allocation_hours = float(sys.argv[2])
projected_hours = float(pilot["projected_hours"])
if not math.isfinite(allocation_hours) or allocation_hours <= 2:
    raise SystemExit("Production allocation must exceed the two-hour reserve")
if pilot.get("status") != "passed" or not math.isfinite(projected_hours) or projected_hours <= 0:
    raise SystemExit("Missing valid passed-pilot runtime projection")
if projected_hours > allocation_hours - 2:
    raise SystemExit(
        f"Pilot projects {projected_hours:.2f}h, exceeding the {allocation_hours - 2:.2f}h "
        f"budget within the {allocation_hours:.2f}h production allocation"
    )
print(f"runtime_projection_hours={projected_hours:.3f} production_allocation_hours={allocation_hours:g}", flush=True)
PY
