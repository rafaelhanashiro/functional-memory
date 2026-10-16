#!/bin/bash
# Qualify the B300 CUDA 13 environment (cuBLAS >= 13.2): replay a checkpoint that failed
# twice on the CUDA 12 environment, then run every GPU test on one B300.
#
#   sbatch -w akita --gres=gpu:b300:1 scaffolds/transformer/validation/qualify_b300_cuda13.sbatch \
#     "$(git rev-parse HEAD)" RUN_DIR CHECKPOINT STOP REPLAYS
#
# Submit from the worktree root; RUN_DIR holds the retained checkpoint.
#SBATCH --job-name=qualify-b300-cuda13
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
LAUNCH_COMMIT="${1:?Expected committed source}"
RUN_DIR=$(cd "${2:?Expected the run directory}" && pwd)
CHECKPOINT="${3:?Expected the checkpoint update}"
STOP="${4:?Expected the last update to replay}"
REPLAYS="${5:-2}"
python3 scaffolds/launch/frozen.py check --launch-commit "$LAUNCH_COMMIT" --experiment attention_scaling --scaffold transformer
source scaffolds/transformer/launchers/kennel_runtime.sh
if [[ "$TRANSFORMER_CUDA_ENVIRONMENT" != cuda13 ]]; then
  echo "Expected a B300 allocation with the CUDA 13 environment; got $TRANSFORMER_CUDA_ENVIRONMENT" >&2
  exit 1
fi
PYTHON="$PWD/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
DIRECTORY="scaffolds/transformer/validation/runs/b300-cuda13"
CACHE="$PWD/.cache/b300-cuda13-qualification"
mkdir -p "$DIRECTORY" scaffolds/transformer/validation/logs "$CACHE/tmp" "$CACHE/replays"
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 TRANSFORMER_HOST_STAGING=0
export TMPDIR="$CACHE/tmp" JAX_COMPILATION_CACHE_DIR="$CACHE/jax" WANDB_MODE=disabled
export SHARED_TRANSFORMER_SOURCE_COMMIT="$LAUNCH_COMMIT"
CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "job=$SLURM_JOB_ID node=$(hostname) driver=$DRIVER environment=$TRANSFORMER_CUDA_ENVIRONMENT commit=$LAUNCH_COMMIT"

REPLAY_LOG="scaffolds/transformer/validation/logs/b300-cuda13-replays-$SLURM_JOB_ID.jsonl"
for index in $(seq 1 "$REPLAYS"); do
  echo "=== replay $index"
  "$PYTHON" scaffolds/transformer/validation/replay_checkpoint.py --run "$RUN_DIR" --checkpoint "$CHECKPOINT" \
      --stop "$STOP" --scratch "$CACHE/replays/$SLURM_JOB_ID-$index" \
    | tee /dev/stderr | sed -n 's/^REPLAY_RESULT //p' >> "$REPLAY_LOG" \
    || echo '{"status": "error"}' >> "$REPLAY_LOG"
done

JUNIT="scaffolds/transformer/validation/logs/b300-cuda13-gpu-tests-$SLURM_JOB_ID.xml"
result=0
"$PYTHON" -m pytest -q -m gpu scaffolds/transformer/tests --junitxml="$JUNIT" || result=$?
"$PYTHON" - "$DIRECTORY" "$JUNIT" "$REPLAY_LOG" "$LAUNCH_COMMIT" "$DRIVER" "$RUN_DIR" "$CHECKPOINT" "$STOP" "$result" <<'PY'
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
import hashlib, json, os, sys
import jax
from scaffolds.transformer.acceptance import require_passing_tests
directory, junit, replay_log, commit, driver, run, checkpoint, stop, result = sys.argv[1:]
root = Path.cwd()
devices = [device.device_kind for device in jax.local_devices()]
status = int(result)
try:
    count = require_passing_tests(Path(junit))
    assert len(devices) == 1 and "B300" in devices[0].upper(), devices
except Exception:
    status = status or 1
    count = None
replays = [json.loads(line) for line in Path(replay_log).read_text().splitlines() if line.strip()]
digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
value = {
    "source_commit": commit, "completed_utc": datetime.now(UTC).isoformat(),
    "home_cluster": "kennel", "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "node": os.environ["SLURMD_NODENAME"], "gpu_type": "B300", "device_kinds": devices,
    "driver_version": driver, "cuda_environment": os.environ["TRANSFORMER_CUDA_ENVIRONMENT"],
    "packages": {name: metadata.version(name) for name in (
        "jax", "jaxlib", "jax-cuda13-plugin", "nvidia-cublas", "nvidia-cuda-runtime", "nvidia-cuda-nvcc",
        "nvidia-cudnn-cu13", "nvidia-nccl-cu13", "triton")},
    "dependency_lock_sha256": digest("scaffolds/transformer/uv.lock"),
    "runtime_environment": {key: os.environ.get(key) for key in (
        "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE", "CUDA_HOME", "TRITON_PTXAS_PATH")},
    "gpu_tests": {"status": "passed" if status == 0 else "failed", "tests_passed": count, "exit_code": status,
                  "junit": {"path": junit, "sha256": digest(junit)}},
    "replay": {"run": str(Path(run).relative_to(root)) if Path(run).is_relative_to(root) else run,
               "checkpoint": int(checkpoint), "stop": int(stop), "results": replays,
               "log": {"path": replay_log, "sha256": digest(replay_log)}},
}
Path(directory, "tests.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
print(json.dumps(value), flush=True)
raise SystemExit(status)
PY
