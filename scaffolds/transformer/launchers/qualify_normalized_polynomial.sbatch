#!/bin/bash
# Shared-kernel qualification. Submit from an isolated, committed worktree.
# Usage: sbatch [resources] this-script LAUNCH_COMMIT RUN [kernel|updates]
#SBATCH --job-name=normalized-polynomial
#SBATCH --partition=sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=collie,samoyed,doodle,terrier
#SBATCH --time=01:00:00
#SBATCH --open-mode=append
set -euo pipefail
LAUNCH_COMMIT="${1:?Expected source commit}"
RUN="${2:?Expected validation run name}"
PHASE="${3:-kernel}"
[[ "$RUN" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
[[ "$PHASE" == kernel || "$PHASE" == updates ]] || exit 2
cd "${SLURM_SUBMIT_DIR:?Submit from the kernel worktree}"
python3 scaffolds/launch/frozen.py check --launch-commit "$LAUNCH_COMMIT" --experiment shared_kernel_validation --scaffold transformer
PYTHON="$PWD/scaffolds/transformer/.venv/bin/python"
RECORD="scaffolds/transformer/validation/runs/$RUN"
LOGS="scaffolds/transformer/validation/logs/$RUN"
mkdir -p "$RECORD" "$LOGS"
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export JAX_COMPILATION_CACHE_DIR="$PWD/.cache/normalized-polynomial/jax"
export XDG_CACHE_HOME="$PWD/.cache/normalized-polynomial/xdg"
export TMPDIR="$PWD/.cache/normalized-polynomial/tmp/$SLURM_JOB_ID"
mkdir -p "$JAX_COMPILATION_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"
source scaffolds/transformer/launchers/kennel_runtime.sh
CUDA_HOME="$("$PYTHON" -c 'import importlib.util; print(next(iter(importlib.util.find_spec("nvidia.cuda_nvcc").submodule_search_locations)))')"
export CUDA_HOME
"$PYTHON" - "$RECORD/metadata.json" "$LAUNCH_COMMIT" "$PHASE" <<'PY'
import json, os, pathlib, socket, subprocess, sys
from datetime import datetime, UTC
from scaffolds.launch.frozen import frozen_hash
root = pathlib.Path.cwd()
metadata = dict(source_commit=sys.argv[2], phase=sys.argv[3], home_cluster="kennel",
    artifact_root=str(root), hostname=socket.gethostname(), started_utc=datetime.now(UTC).isoformat(),
    frozen_execution_set_sha256=frozen_hash(root, sys.argv[2], "shared_kernel_validation", "transformer"),
    slurm_job_id=os.environ["SLURM_JOB_ID"], partition=os.environ["SLURM_JOB_PARTITION"],
    hardware=subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True).splitlines(),
    runtime={key: os.environ.get(key) for key in ("CUDA_HOME", "JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES",
        "TRANSFORMER_HOST_STAGING", "NCCL_P2P_DISABLE", "XLA_PYTHON_CLIENT_PREALLOCATE")})
pathlib.Path(sys.argv[1]).write_text(json.dumps(metadata, indent=2) + "\n")
PY
if [[ "$PHASE" == kernel ]]; then
  "$PYTHON" -m pytest -c scaffolds/transformer/pyproject.toml --import-mode=importlib -q -m gpu \
    scaffolds/transformer/tests/test_normalized_polynomial_fused_attention.py \
    --junitxml="$LOGS/tests.xml"
  "$PYTHON" - "$LOGS/tests.xml" "$RECORD/tests.json" <<'PY'
import json, pathlib, sys, xml.etree.ElementTree as ET
from scaffolds.transformer.acceptance import require_passing_tests
require_passing_tests(pathlib.Path(sys.argv[1]))
root = ET.parse(sys.argv[1]).getroot()
suites = root.findall("testsuite") if root.tag == "testsuites" else [root]
result = {key: sum(int(suite.get(key, 0)) for suite in suites) for key in ("tests", "failures", "errors", "skipped")}
result["seconds"] = sum(float(suite.get("time", 0)) for suite in suites)
pathlib.Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + "\n")
PY
  "$PYTHON" -m scaffolds.transformer.benchmarks.benchmark_normalized_polynomial --output "$RECORD/benchmark.json"
else
  for parameterization in direct sum_of_squares; do
    for degree in 2 4; do
      "$PYTHON" -m scaffolds.transformer.benchmarks.benchmark_train_step \
        --config "scaffolds/transformer/validation/configs/normalized-polynomial-$parameterization-m$degree.yaml" \
        --degrees "$degree" --device-count 2 --global-micro-batch-size 32 --accumulation-steps 4 \
        --output "$RECORD/$parameterization-m$degree.json"
    done
  done
fi
