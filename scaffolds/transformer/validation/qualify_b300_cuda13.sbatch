#!/bin/bash
# Qualify the B300 CUDA 13 environment (cuBLAS >= 13.2): optionally replay a checkpoint
# that failed on the CUDA 12 environment, then run every GPU test file on one B300, each in
# its own process so a crashed kernel cannot fail the files after it. COMPARE names tests
# that also run, each in a fresh process, under both environments.
#
#   sbatch -w akita --gres=gpu:b300:1 scaffolds/transformer/validation/qualify_b300_cuda13.sbatch \
#     "$(git rev-parse HEAD)" RUN_DIR CHECKPOINT STOP REPLAYS [COMPARE...]
#
# Submit from the worktree root; RUN_DIR holds the retained checkpoint ("-" skips replays).
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
RUN_DIR="${2:?Expected the run directory or -}"
CHECKPOINT="${3:?Expected the checkpoint update}"
STOP="${4:?Expected the last update to replay}"
REPLAYS="${5:-2}"
shift 5 || shift $#
COMPARE=("$@")
if [[ "$RUN_DIR" == - ]]; then REPLAYS=0; else RUN_DIR=$(cd "$RUN_DIR" && pwd); fi
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
: > "$REPLAY_LOG"
for index in $(seq 1 "$REPLAYS"); do
  echo "=== replay $index"
  "$PYTHON" scaffolds/transformer/validation/replay_checkpoint.py --run "$RUN_DIR" --checkpoint "$CHECKPOINT" \
      --stop "$STOP" --scratch "$CACHE/replays/$SLURM_JOB_ID-$index" \
    | tee /dev/stderr | sed -n 's/^REPLAY_RESULT //p' >> "$REPLAY_LOG" \
    || echo '{"status": "error"}' >> "$REPLAY_LOG"
done

LOGS="scaffolds/transformer/validation/logs/b300-cuda13-gpu-$SLURM_JOB_ID"
mkdir -p "$LOGS"
for file in scaffolds/transformer/tests/test_*.py; do
  name=$(basename "$file" .py)
  echo "=== $name"
  code=0
  "$PYTHON" -m pytest -q -m gpu "$file" --junitxml="$LOGS/$name.xml" || code=$?
  echo "$name $code" >> "$LOGS/exit-codes.txt"
done
# The same tests in fresh processes under both environments.
CUDA12_PYTHON="$PWD/scaffolds/transformer/.venv/bin/python"
for test in "${COMPARE[@]}"; do
  for environment in cuda13 cuda12; do
    if [[ "$environment" == cuda12 ]]; then python="$CUDA12_PYTHON"; else python="$PYTHON"; fi
    home="$(transformer_cuda_home "$python")"
    code=0
    CUDA_HOME="$home" TRITON_PTXAS_PATH="$home/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$home/bin/ptxas" \
      "$python" -m pytest -q -p no:cacheprovider "$test" || code=$?
    printf '%s\t%s\t%s\n' "$environment" "$code" "$test" >> "$LOGS/compare.tsv"
  done
done
"$PYTHON" - "$DIRECTORY" "$LOGS" "$REPLAY_LOG" "$LAUNCH_COMMIT" "$DRIVER" "$RUN_DIR" "$CHECKPOINT" "$STOP" <<'PY'
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
import hashlib, json, os, sys
import xml.etree.ElementTree as ET
import jax
directory, logs, replay_log, commit, driver, run, checkpoint, stop = sys.argv[1:]
root = Path.cwd()
devices = [device.device_kind for device in jax.local_devices()]
files, failed = {}, []
for line in Path(logs, "exit-codes.txt").read_text().split("\n"):
    if not line.strip():
        continue
    name, code = line.split()
    junit = Path(logs, f"{name}.xml")
    counts = {"exit_code": int(code)}
    if junit.exists():
        suite = ET.parse(junit).getroot()
        suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
        counts.update({key: int(suite.get(key, 0)) for key in ("tests", "failures", "errors", "skipped")})
        counts["failed_tests"] = [case.get("name") for case in suite.iter("testcase")
                                  if case.find("failure") is not None or case.find("error") is not None]
    if counts["exit_code"] == 5:  # no GPU tests in this file
        continue
    files[name] = counts
    if counts["exit_code"] != 0:
        failed.append(name)
status = 1 if failed or not (len(devices) == 1 and "B300" in devices[0].upper()) else 0
compare = [dict(zip(("environment", "exit_code", "test"), line.split("\t")))
           for line in (Path(logs, "compare.tsv").read_text().splitlines() if Path(logs, "compare.tsv").exists() else [])]
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
    "gpu_tests": {"status": "passed" if status == 0 else "failed", "isolation": "one process per test file",
                  "files": files, "failed_files": failed,
                  "tests_passed": sum(c.get("tests", 0) - c.get("failures", 0) - c.get("errors", 0) - c.get("skipped", 0)
                                      for c in files.values())},
    "compare_environments": compare,
    "replay": {"run": run, "checkpoint": int(checkpoint), "stop": int(stop), "results": replays,
               "log": {"path": replay_log, "sha256": digest(replay_log)}},
}
Path(directory, "tests.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
print(json.dumps(value), flush=True)
raise SystemExit(status)
PY
