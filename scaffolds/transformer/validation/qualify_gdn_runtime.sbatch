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
source scaffolds/launch/watchdog.sh
finish_qualification() {
  local result=$?
  trap - EXIT
  stop_watchdog
  rm -rf -- "$GDN_RUNTIME_TMPDIR"
  exit "$result"
}
trap finish_qualification EXIT
# Resolve Slurm's actual output path, including any caller-selected log name.
GDN_QUALIFICATION_LOG="$(scontrol show job "$SLURM_JOB_ID" -o | tr ' ' '\n' | sed -n 's/^StdOut=//p')"
if [[ -z "$GDN_QUALIFICATION_LOG" ]]; then
  echo "Cannot resolve qualification log for the stall watchdog" >&2
  exit 1
fi
start_watchdog "$GDN_QUALIFICATION_LOG"
CUDA_HOME="$("$PYTHON" -c 'import importlib.util; print(next(iter(importlib.util.find_spec("nvidia.cuda_nvcc").submodule_search_locations)))')"
export CUDA_HOME
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
eval "$("$PYTHON" -m "$ADAPTER" "$CONFIG" --runtime-env)"
mkdir -p "$LOGS" "$JAX_COMPILATION_CACHE_DIR" "$XDG_CACHE_HOME"
echo "runtime_qualification run=$RUN job=$SLURM_JOB_ID launch_commit=$LAUNCH_COMMIT gpu_type=$GPU_TYPE gpu_count=$GPU_COUNT preallocate=$XLA_PYTHON_CLIENT_PREALLOCATE memory_fraction=${XLA_PYTHON_CLIENT_MEM_FRACTION:-default}"
# The acceptance hardware worker verifies the allocated GPU count and type.
# It also checks initialization, full-recipe training/evaluation, and exact resume.
run_step "$PYTHON" -c '
from dataclasses import replace
import math
import sys
from scaffolds.transformer.acceptance import attention_spec, main
hours = float(sys.argv[1])
if not math.isfinite(hours) or hours <= 2:
    raise SystemExit("Production allocation must exceed the two-hour reserve")
main(lambda config: replace(attention_spec(config, entry_module="experiments.attention_scaling.acceptance"),
                            allocation_hours=hours), sys.argv[2:])
' "$PRODUCTION_HOURS" "$CONFIG" --directory "$LOGS/acceptance/$GPU_TYPE-$GPU_COUNT" \
  --gpu-count "$GPU_COUNT" --gpu-type "$GPU_TYPE" --test-cache "$CACHE/acceptance-tests"
# Enforce and record the planned production allocation even when matching passed
# acceptance is reused. This budget does not change model/runtime test identities.
"$PYTHON" - "$LOGS/acceptance/$GPU_TYPE-$GPU_COUNT/pilot.json" "$PRODUCTION_HOURS" <<'PY'
import json
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
from scaffolds.transformer.acceptance import write_json

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
success_path = Path(sys.argv[1]).with_name("success.json")
success = json.loads(success_path.read_text())
if success.get("status") != "passed" or success["identity"] != pilot["identity"]:
    raise SystemExit("Production budget must refer to the matching passed acceptance")
success["runtime_budget"] = {
    "production_allocation_hours": allocation_hours,
    "maximum_projected_hours": allocation_hours - 2,
    "projected_hours": projected_hours,
    "within_budget": True,
}
launcher = Path("scaffolds/transformer/validation/qualify_gdn_runtime.sbatch")
success["qualification"] = {
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "launcher": {"path": str(launcher), "sha256": hashlib.sha256(launcher.read_bytes()).hexdigest()},
    "allocation": subprocess.check_output(["scontrol", "show", "job", "-o", os.environ["SLURM_JOB_ID"]], text=True).strip(),
}
write_json(success_path, success)
PY
