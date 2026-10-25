#!/bin/bash
# Qualify degree 12: Mosaic query and static kernels and the one-pass Triton query-key kernel,
# then time one layer's forward and backward at degrees 8 and 12 (one GPU, micro-batch 32).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
LAUNCH_COMMIT="${1:?Expected committed source}"
GPU_TYPE="${2:?Expected H100 or B300}"
DIRECTORY="scaffolds/transformer/validation/runs/degree12-${GPU_TYPE,,}"
python3 scaffolds/launch/frozen.py check --launch-commit "$LAUNCH_COMMIT" --experiment attention_scaling --scaffold transformer
mkdir -p "$DIRECTORY" scaffolds/transformer/validation/logs
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$PWD/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/degree12-${SLURM_JOB_ID}.XXXXXX")"; trap 'rm -rf "$TMPDIR"' EXIT
export TMPDIR JAX_COMPILATION_CACHE_DIR="$TMPDIR/jax" XDG_CACHE_HOME="$TMPDIR/xdg"
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
JUNIT="scaffolds/transformer/validation/logs/degree12-${GPU_TYPE,,}-$SLURM_JOB_ID.xml"
result=0
"$PYTHON" -m pytest -q -m gpu -c scaffolds/transformer/pyproject.toml --import-mode=importlib \
  scaffolds/transformer/tests/test_mosaic_query.py scaffolds/transformer/tests/test_mosaic_unnormalized.py \
  scaffolds/transformer/tests/test_mosaic_taylor.py scaffolds/transformer/tests/test_query_key_polynomial.py \
  --junitxml="$JUNIT" || result=$?
"$PYTHON" - "$DIRECTORY" "$JUNIT" "$LAUNCH_COMMIT" "$GPU_TYPE" "$result" <<'PY'
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
import hashlib, json, math, os, sys, time
import jax
import jax.numpy as jnp
from scaffolds.transformer.acceptance import require_passing_tests
from scaffolds.transformer.attention.backends.fused_query_key_polynomial import fused_query_key_polynomial_attention
from scaffolds.transformer.attention.backends.mosaic.api import query_polynomial_attention
directory, junit, commit, gpu, result = sys.argv[1:]
devices = [device.device_kind for device in jax.local_devices()]
status = int(result)
try:
    count = require_passing_tests(Path(junit))
    assert len(devices) == 1 and gpu in devices[0].upper(), devices
except Exception:
    status = status or 1
    count = None


def milliseconds(function, *operands, repeats=10):
    jax.block_until_ready(function(*operands))
    start = time.perf_counter()
    for _ in range(repeats):
        output = function(*operands)
    jax.block_until_ready(output)
    return (time.perf_counter() - start) / repeats * 1e3


keys = jax.random.split(jax.random.key(0), 6)
shape = (32, 2048, 16, 64)  # one production micro-batch on one GPU
q, k = (jax.random.normal(key, shape, jnp.float32) for key in keys[:2])
q = (q / jnp.linalg.norm(q, axis=-1, keepdims=True) * 7.0 * math.sqrt(64)).astype(jnp.bfloat16)
k = (k / jnp.linalg.norm(k, axis=-1, keepdims=True)).astype(jnp.bfloat16)
v, do = (jax.random.normal(key, shape, jnp.bfloat16) for key in keys[2:4])
timing = {}
for degree in (8, 12):
    taylor = jnp.asarray([1 / math.factorial(r) for r in range(degree + 1)], jnp.float32)
    c = jnp.broadcast_to(taylor, (*shape[:3], degree + 1))
    f = jnp.ones((*shape[:3], degree + 1), jnp.float32)
    for name, attend, operands in (
            ("mosaic_query", lambda q, k, v, c: query_polynomial_attention(q, k, v, c), (q, k, v, c)),
            ("triton_query_key", fused_query_key_polynomial_attention, (q, k, v, c, f))):
        def step(*args, attend=attend):
            out, pullback = jax.vjp(attend, *args)
            return pullback(do.astype(out.dtype))
        timing[f"{name}_degree{degree}_forward_backward_ms"] = milliseconds(jax.jit(step), *operands)
value = {
    "source_commit": commit, "completed_utc": datetime.now(UTC).isoformat(),
    "home_cluster": "kennel", "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "node": os.environ["SLURMD_NODENAME"], "gpu_type": gpu, "device_kinds": devices,
    "status": "passed" if status == 0 else "failed", "tests_passed": count, "exit_code": status,
    "timing_shape": {"batch": 32, "time": 2048, "heads": 16, "width": 64, "score_scale": 7.0},
    "timing": timing,
    "dependency_lock_sha256": hashlib.sha256(Path("scaffolds/transformer/uv.lock").read_bytes()).hexdigest(),
    "runtime_environment": {key: os.environ.get(key) for key in (
        "TRANSFORMER_CUDA_ENVIRONMENT", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE")},
    "junit": {"path": junit, "sha256": hashlib.sha256(Path(junit).read_bytes()).hexdigest()},
}
Path(directory, "tests.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
print(json.dumps(value), flush=True)
raise SystemExit(status)
PY
