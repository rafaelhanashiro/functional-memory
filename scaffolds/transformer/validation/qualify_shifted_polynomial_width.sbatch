#!/bin/bash
# Qualify the row-power shifted kernel (attention/backends/fused_shifted_polynomial.py) at Q/K width 64 on an ORCD
# Hopper GPU: the width-64 cases (tests/test_shifted_polynomial_key_width.py: the fused programs against FP64 at degrees
# 8 and 16, and the fused pure tanh and tanh query models against the blocked reference's) and the module's own GPU
# cases at Q/K width 4 (tests/test_shifted_polynomial_attention.py), then time the kernel at the reference shape's
# micro-batch (benchmarks/benchmark_shifted_polynomial_width.py). Submit from the worktree root; the log stays in the
# ignored scaffold validation logs:
# sbatch --gres=gpu:h200:1 --exclude=node3009,node3209,node2900,node1708,node1709,node4508,node2801,node3409 \
#   --output=scaffolds/transformer/validation/logs/shifted-width-%j.log \
#   scaffolds/transformer/validation/qualify_shifted_polynomial_width.sbatch
# (H100: --gres=gpu:h100:1.)
#SBATCH --job-name=qualify-shifted-width
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the worktree root}"
PYTHON="$PWD/scaffolds/transformer/.venv/bin/python"
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda,cpu PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
CUDA_HOME="$("$PYTHON" -c 'import importlib.util; print(next(iter(importlib.util.find_spec("nvidia.cuda_nvcc").submodule_search_locations)))')"
export CUDA_HOME
echo "job=$SLURM_JOB_ID node=$(hostname) commit=$(git rev-parse HEAD) dirty=$(git status --porcelain -- scaffolds | wc -l)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
status=0
"$PYTHON" -m pytest -c scaffolds/transformer/pyproject.toml -m gpu -q -p no:cacheprovider \
  scaffolds/transformer/tests/test_shifted_polynomial_key_width.py \
  scaffolds/transformer/tests/test_shifted_polynomial_attention.py || status=$?
# The width-64 cases' largest errors against FP64, each relative to the largest entry of its reference.
"$PYTHON" - <<'PY' || status=$?
import jax, jax.numpy as jnp
from scaffolds.transformer.attention.output_scaling import max_abs_head_rms
from scaffolds.transformer.attention.shifted_polynomial import shifted_polynomial_attention
from scaffolds.transformer.tests.test_shifted_polynomial_attention import _oracle
from scaffolds.transformer.tests.test_shifted_polynomial_key_width import _inputs
for degree in (8, 16):
    args, cot = _inputs(257, degree)
    with jax.enable_x64(True):
        oracle, pull = jax.vjp(lambda *a: _oracle(*a, "identity"), *(jnp.asarray(x, jnp.float64) for x in args))
        expected = (oracle, *pull(jnp.asarray(cot, jnp.float64)))
    def normalized(*a):
        result, shift = shifted_polynomial_attention(*a, transform="identity", backend="fused")
        return max_abs_head_rms(result, 1e-6, shift[..., None])
    y, pull = jax.vjp(jax.jit(normalized), *map(jnp.asarray, args))
    actual = (y, *pull(jnp.asarray(cot)))
    errors = {name: float(jnp.max(jnp.abs(a - e)) / jnp.max(jnp.abs(e)))
              for name, a, e in zip(("output", "dq", "dk", "dv", "dc"), actual, expected, strict=True)}
    print(f"degree {degree}: largest error / largest entry", {k: f"{v:.2e}" for k, v in errors.items()})
PY
"$PYTHON" scaffolds/transformer/benchmarks/benchmark_shifted_polynomial_width.py --degrees 8 16 || status=$?
exit "$status"
