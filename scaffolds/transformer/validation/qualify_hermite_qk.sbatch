#!/bin/bash
# Compile the fixed Hermite and matched Taylor kernels with canonical QK norm.
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
LAUNCH_COMMIT="${1:?Expected committed source}"
GPU_TYPE="${2:?Expected H100 or B300}"
PYTHON="$PWD/scaffolds/transformer/.venv/bin/python"
DIRECTORY="scaffolds/transformer/validation/runs/hermite-qk-${GPU_TYPE,,}"
python3 scaffolds/launch/frozen.py check --launch-commit "$LAUNCH_COMMIT" --experiment hermite_attention --scaffold transformer
mkdir -p "$DIRECTORY" .cache/hermite-qk-qualification/tmp
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export TRANSFORMER_HOST_STAGING=0
export TMPDIR="$PWD/.cache/hermite-qk-qualification/tmp"
export JAX_COMPILATION_CACHE_DIR="$PWD/.cache/hermite-qk-qualification/jax"
CUDA_HOME="$("$PYTHON" -c 'import importlib.util; print(next(iter(importlib.util.find_spec("nvidia.cuda_nvcc").submodule_search_locations)))')"
export CUDA_HOME
JUNIT="scaffolds/transformer/validation/logs/hermite-qk-${GPU_TYPE,,}-$SLURM_JOB_ID.xml"
result=0
"$PYTHON" -m pytest -q -c scaffolds/transformer/pyproject.toml --import-mode=importlib -m gpu \
  scaffolds/transformer/tests/test_fixed_polynomial_fused.py \
  scaffolds/transformer/tests/test_fixed_square_factors_fused.py \
  scaffolds/transformer/tests/test_normalized_polynomial_fused_attention.py \
  scaffolds/transformer/tests/test_qk_norm.py \
  experiments/hermite_attention/tests/test_qk_norm.py --junitxml="$JUNIT" || result=$?
"$PYTHON" - "$DIRECTORY" "$JUNIT" "$LAUNCH_COMMIT" "$GPU_TYPE" "$result" <<'PY'
from datetime import UTC, datetime
from pathlib import Path
import hashlib, json, os, sys
import jax
from scaffolds.transformer.acceptance import require_passing_tests
directory, junit, commit, gpu, result = sys.argv[1:]
devices = [device.device_kind for device in jax.local_devices()]
status = int(result)
try:
    count = require_passing_tests(Path(junit))
    assert len(devices) == 1 and gpu in devices[0].upper(), devices
except Exception:
    status = status or 1
    count = None
value = {
    "source_commit": commit, "completed_utc": datetime.now(UTC).isoformat(),
    "home_cluster": "kennel", "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "node": os.environ["SLURMD_NODENAME"], "gpu_type": gpu, "device_kinds": devices,
    "status": "passed" if status == 0 else "failed", "tests_passed": count, "exit_code": status,
    "dependency_lock_sha256": hashlib.sha256(Path("scaffolds/transformer/uv.lock").read_bytes()).hexdigest(),
    "junit": {"path": junit, "sha256": hashlib.sha256(Path(junit).read_bytes()).hexdigest()},
}
Path(directory, "tests.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
print(json.dumps(value), flush=True)
raise SystemExit(status)
PY
