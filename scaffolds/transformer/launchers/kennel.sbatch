#!/bin/bash
# Kennel launcher for any Transformer experiment. Submit from the experiment's
# worktree root; resource policy lives here, outside model and training code.
#
#   sbatch --job-name=<run> --output=experiments/<experiment>/logs/<run>/slurm-%j.out \
#     scaffolds/transformer/launchers/kennel.sbatch \
#     experiments.<experiment>.adapter.execution.acceptance \
#     experiments/<experiment>/configs/<run>.yaml "$(git rev-parse HEAD)" [GPU_COUNT] [GPU_TYPES]
#SBATCH --job-name=transformer
#SBATCH --partition=sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --exclude=collie,samoyed,doodle,terrier
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
ADAPTER="${1:?Expected the experiment acceptance module}"
CONFIG="${2:?Expected experiments/<experiment>/configs/<run>.yaml}"
LAUNCH_COMMIT="${3:?Expected the launch commit}"
GPU_COUNT="${4:-4}"
GPU_TYPES="${5:-B300,H100}"
EXECUTION_ROOT="${SLURM_SUBMIT_DIR:?Submit from the experiment worktree root}"
cd "$EXECUTION_ROOT"
if [[ ! "$CONFIG" =~ ^experiments/([A-Za-z0-9_]+)/configs/([^/]+)\.yaml$ ]]; then
  echo "Expected a repository-relative recipe experiments/<experiment>/configs/<run>.yaml: $CONFIG" >&2
  exit 1
fi
EXPERIMENT="${BASH_REMATCH[1]}"
RUN="${BASH_REMATCH[2]}"
# Only the frozen execution set must match the launch commit; outputs and analysis may change.
python3 scaffolds/launch/frozen.py check --launch-commit "$LAUNCH_COMMIT" --experiment "$EXPERIMENT" --scaffold transformer

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

source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$EXECUTION_ROOT/scaffolds/transformer/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing the worktree environment; run uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink" >&2
  exit 1
fi
LOGS="experiments/$EXPERIMENT/logs/$RUN"
CACHE="$EXECUTION_ROOT/.cache/$EXPERIMENT"
export SHARED_TRANSFORMER_SOURCE_COMMIT="$LAUNCH_COMMIT"
# Execution facts for the W&B mirror; the mode may be overridden per submission.
export TRANSFORMER_CLUSTER=kennel TRANSFORMER_GPU_TYPE="$GPU_TYPE"
WANDB_MODE_SETTING="${TRANSFORMER_WANDB_MODE:-online}"
export PYTHONPATH="$EXECUTION_ROOT"
export JAX_PLATFORMS=cuda
export TRANSFORMER_HOST_STAGING="${TRANSFORMER_HOST_STAGING:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export JAX_COMPILATION_CACHE_DIR="$CACHE/jax" XDG_CACHE_HOME="$CACHE/xdg" TMPDIR="$CACHE/tmp/$SLURM_JOB_ID"
# jaxlib assembles Triton custom-call PTX with the first ptxas among its CUDA roots,
# which is the node's /usr/local/cuda unless CUDA_HOME names another; pin the locked toolkit.
CUDA_HOME="$("$PYTHON" -c 'import importlib.util; print(next(iter(importlib.util.find_spec("nvidia.cuda_nvcc").submodule_search_locations)))')"
export CUDA_HOME
eval "$("$PYTHON" -m "$ADAPTER" --runtime-env)"
mkdir -p "$LOGS/submissions" "$JAX_COMPILATION_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"

"$PYTHON" - "$LOGS/submissions/$SLURM_JOB_ID-${SLURM_RESTART_COUNT:-0}.json" "$CONFIG" "$LAUNCH_COMMIT" "$GPU_TYPE" "$GPU_COUNT" <<'PY'
import hashlib, json, os, subprocess, sys
from datetime import UTC, datetime
from pathlib import Path
path, config, launch, gpu_type, gpu_count = sys.argv[1:]
Path(path).write_text(json.dumps({
    "launch_commit": launch, "config_path": config,
    "config_sha256": hashlib.sha256(Path(config).read_bytes()).hexdigest(),
    "home_cluster": "kennel", "gpu_type": gpu_type, "gpu_count": int(gpu_count),
    "slurm_job_id": os.environ["SLURM_JOB_ID"], "slurm_restart_count": os.environ.get("SLURM_RESTART_COUNT", "0"),
    "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"), "started_utc": datetime.now(UTC).isoformat(),
    "allocation": subprocess.check_output(["scontrol", "show", "job", "-o", os.environ["SLURM_JOB_ID"]], text=True).strip(),
}, indent=2) + "\n")
PY
# Acceptance and production share the allocation; matching passed stages are reused on requeue.
"$PYTHON" -m "$ADAPTER" "$CONFIG" --directory "$LOGS/acceptance/$GPU_TYPE-$GPU_COUNT" \
  --gpu-count "$GPU_COUNT" --gpu-type "$GPU_TYPE"
exec "$PYTHON" -m scaffolds.transformer.train "$CONFIG" --wandb-mode "$WANDB_MODE_SETTING"
