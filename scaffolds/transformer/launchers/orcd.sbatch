#!/bin/bash
# ORCD launcher for any Transformer experiment. Submit from the experiment's
# worktree root; resource policy lives here, outside model and training code.
#
#   sbatch --job-name=<run> --output=experiments/<study>/runs/<run>/training.log \
#     scaffolds/transformer/launchers/orcd.sbatch \
#     experiments.<study>.acceptance \
#     experiments/<study>/runs/<run>/config.yaml "$(git rev-parse HEAD)" [GPU_COUNT] [GPU_TYPES]
# One GPU is the default. For an explicit multi-GPU run, override both --gres and GPU_COUNT.
#
# A job starts in whichever GPU partition frees first. It asks for 12 hours
# with a 6-hour minimum: Slurm's backfill trims the limit to
# mit_normal_gpu's 6-hour maximum when it starts the job there. Ten minutes before
# the limit the job requeues itself and resumes from its latest checkpoint.
# Preemption in mit_preemptable also requeues. Per-user caps are 2 GPUs in
# mit_normal_gpu and 4 in mit_preemptable; independent jobs use one GPU each.
#SBATCH --job-name=transformer
#SBATCH --partition=mit_preemptable,mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=12:00:00
#SBATCH --time-min=06:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
ADAPTER="${1:?Expected the experiment acceptance module}"
CONFIG="${2:?Expected experiments/<study>/runs/<run>/config.yaml}"
LAUNCH_COMMIT="${3:?Expected the launch commit}"
GPU_COUNT="${4:-1}"
GPU_TYPES="${5:-H200}"
EXECUTION_ROOT="${SLURM_SUBMIT_DIR:?Submit from the experiment worktree root}"
cd "$EXECUTION_ROOT"
if [[ ! "$CONFIG" =~ ^experiments/([A-Za-z0-9_]+)/runs/(r[0-9a-f]{12})/config\.yaml$ ]]; then
  echo "Expected a repository-relative recipe experiments/<study>/runs/<run>/config.yaml: $CONFIG" >&2
  exit 1
fi
EXPERIMENT="${BASH_REMATCH[1]}"
RUN="${BASH_REMATCH[2]}"
RUN_ROOT="experiments/$EXPERIMENT/runs/$RUN"
PYTHON="$EXECUTION_ROOT/scaffolds/transformer/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing the worktree environment; run uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink" >&2
  exit 1
fi
# Only the frozen execution set must match the launch commit; outputs and analysis may change.
# ORCD's system python3 is 3.6, older than frozen.py supports, so use the locked interpreter.
"$PYTHON" scaffolds/launch/frozen.py check --launch-commit "$LAUNCH_COMMIT" --experiment "$EXPERIMENT" --scaffold transformer
git cat-file -e "$LAUNCH_COMMIT:$CONFIG"
exec >>"$RUN_ROOT/training.log" 2>&1

GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u)"
if [[ "$(wc -l <<<"$GPU_NAMES")" != 1 ]]; then
  echo "Mixed GPU types in one allocation: $GPU_NAMES" >&2
  exit 1
fi
GPU_TYPE=""
for candidate in ${GPU_TYPES//,/ }; do
  if [[ "$GPU_NAMES" == *"$candidate"* ]]; then GPU_TYPE="$candidate"; break; fi
done
if [[ -z "$GPU_TYPE" ]]; then
  echo "Allocated $GPU_NAMES; this experiment accepts only $GPU_TYPES" >&2
  exit 1
fi
echo "run=$RUN job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} launch_commit=$LAUNCH_COMMIT gpu_type=$GPU_TYPE"

LOGS="$RUN_ROOT/.runtime"
CACHE="$EXECUTION_ROOT/.cache/$EXPERIMENT"
export SHARED_TRANSFORMER_SOURCE_COMMIT="$LAUNCH_COMMIT"
# Execution facts for the W&B mirror; the mode may be overridden per submission.
export TRANSFORMER_CLUSTER=orcd TRANSFORMER_GPU_TYPE="$GPU_TYPE"
WANDB_MODE_SETTING="${TRANSFORMER_WANDB_MODE:-online}"
export PYTHONPATH="$EXECUTION_ROOT"
TRANSFORMER_JOB_TMPDIR=""
source scaffolds/launch/watchdog.sh
finish_record() {
  local result=$?
  local finalizer_result=0
  trap - EXIT
  stop_watchdog
  # Before the finalizer, which never reads it: a requeue the finalizer requests can end this shell before it returns.
  if [[ -n "${TRANSFORMER_JOB_TMPDIR:-}" ]]; then
    rm -rf -- "$TRANSFORMER_JOB_TMPDIR"
  fi
  "$PYTHON" -m scaffolds.launch.finalize --run "$RUN_ROOT" --exit-code "$result" || finalizer_result=$?
  # Preserve the worker's failure even if recording it also fails.
  if (( result == 0 )); then result=$finalizer_result; fi
  exit "$result"
}
trap finish_record EXIT
requeue_before_limit() {
  echo "Requeueing job $SLURM_JOB_ID before its time limit; it resumes from the latest checkpoint"
  scontrol requeue "$SLURM_JOB_ID"
}
trap requeue_before_limit USR1
start_watchdog "$RUN_ROOT/training.log"
export JAX_PLATFORMS=cuda
# ORCD H100/H200 allocations use direct device transfer; an explicit setting overrides it.
export TRANSFORMER_HOST_STAGING="${TRANSFORMER_HOST_STAGING:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export JAX_COMPILATION_CACHE_DIR="$CACHE/jax" XDG_CACHE_HOME="$CACHE/xdg"
# Compiler and pytest temporaries are disposable and stay node-local, outside the checkout, as on Kennel. Inside the
# worktree, a test's scratch study would be recorded with paths relative to the checkout (scaffolds.records), which
# the records cannot resolve again (tests/test_gradient_events.py).
TRANSFORMER_JOB_TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/functional-memory-${SLURM_JOB_ID}.XXXXXX")"
export TMPDIR="$TRANSFORMER_JOB_TMPDIR"
# jaxlib assembles Triton custom-call PTX with the first ptxas among its CUDA roots,
# which is the node's /usr/local/cuda unless CUDA_HOME names another; pin the locked toolkit.
CUDA_HOME="$("$PYTHON" -c 'import importlib.util; print(next(iter(importlib.util.find_spec("nvidia.cuda_nvcc").submodule_search_locations)))')"
export CUDA_HOME
eval "$("$PYTHON" -m "$ADAPTER" "$CONFIG" --runtime-env)"
mkdir -p "$LOGS" "$JAX_COMPILATION_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"

"$PYTHON" - "$RUN_ROOT" "$CONFIG" "$LAUNCH_COMMIT" "$GPU_TYPE" "$GPU_COUNT" "$EXPERIMENT" <<'PY'
import hashlib, json, os, subprocess, sys
from datetime import UTC, datetime
from pathlib import Path
from scaffolds.launch.frozen import frozen_hash
from scaffolds.records import load_run, update_record, validate_record
from scaffolds.transformer.configs import load_config
root, config, launch, gpu_type, gpu_count, experiment = sys.argv[1:]
run_root = Path(root)
recipe = load_config(config)
if recipe.train.workdir.resolve() != run_root.resolve():
    raise ValueError("The allocated config must write into its own run directory")
if recipe.train.checkpoint_dir is not None and recipe.train.checkpoint_dir.resolve() != (run_root / "checkpoints").resolve():
    raise ValueError("Canonical run checkpoints must be colocated under runs/<run>/checkpoints")
record = load_run(run_root)
validate_record(run_root, verify_hashes=False)
attempt = {
    "launch_commit": launch, "config_path": config,
    "config_sha256": hashlib.sha256(Path(config).read_bytes()).hexdigest(),
    "home_cluster": "orcd", "gpu_type": gpu_type, "gpu_count": int(gpu_count),
    "slurm_job_id": os.environ["SLURM_JOB_ID"], "slurm_restart_count": os.environ.get("SLURM_RESTART_COUNT", "0"),
    "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"), "started_utc": datetime.now(UTC).isoformat(),
    "acceptance_directory": str((run_root / ".runtime" / "acceptance" / f"{gpu_type}-{gpu_count}").resolve()),
    "allocation": subprocess.check_output(["scontrol", "show", "job", "-o", os.environ["SLURM_JOB_ID"]], text=True).strip(),
}
update_record(run_root, {
    "source_commit": launch, "home_cluster": "orcd", "status": "acceptance",
    "frozen_execution_sha256": frozen_hash(Path.cwd(), launch, experiment, "transformer"),
    "attempts": [*record.get("attempts", []), attempt],
    "log_path": str(run_root / "training.log"),
})
PY
# Acceptance and production share the allocation; matching passed stages are reused on a requeue
# to the same GPU model, and passed test stages are shared with the worktree's other runs.
run_step "$PYTHON" -m "$ADAPTER" "$CONFIG" --directory "$LOGS/acceptance/$GPU_TYPE-$GPU_COUNT" \
  --gpu-count "$GPU_COUNT" --gpu-type "$GPU_TYPE" --test-cache "$CACHE/acceptance-tests"
"$PYTHON" - "$RUN_ROOT" "$LOGS/acceptance/$GPU_TYPE-$GPU_COUNT/success.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
from scaffolds.records import update_record
root, path = map(Path, sys.argv[1:])
summary = json.loads(path.read_text())
# Replace, not merge into, an earlier attempt's failed acceptance.
update_record(root, {"acceptance": None})
update_record(root, {"acceptance": {"summary": summary, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}})
PY
run_step "$PYTHON" -m scaffolds.transformer.train "$CONFIG" --wandb-mode "$WANDB_MODE_SETTING"
run_step "$PYTHON" -m scaffolds.transformer.retention "$RUN_ROOT/config.json"
