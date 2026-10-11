#!/bin/bash
# One-GPU kernel benchmark; full-model fit/timing remains part of run acceptance.
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export GDN_BENCHMARK_SOURCE="${1:?Expected committed source}"
shift
python3 scaffolds/launch/frozen.py check --launch-commit "$GDN_BENCHMARK_SOURCE" --experiment attention_scaling --scaffold transformer
PYTHON="$PWD/scaffolds/transformer/.venv/bin/python"
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 TRANSFORMER_HOST_STAGING=0
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export EJKERNEL_TRITON_CACHE_DIR="$PWD/.cache/gdn-independent-qualification/triton"
export JAX_COMPILATION_CACHE_DIR="$PWD/.cache/gdn-independent-qualification/jax"
GDN_JOB_TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/gdn-benchmark-${SLURM_JOB_ID}.XXXXXX")"
export TMPDIR="$GDN_JOB_TMPDIR"
trap 'rm -rf -- "$GDN_JOB_TMPDIR"' EXIT
mkdir -p scaffolds/transformer/validation/logs
source scaffolds/transformer/launchers/kennel_runtime.sh
CUDA_HOME="$("$PYTHON" -c 'import importlib.util; print(next(iter(importlib.util.find_spec("nvidia.cuda_nvcc").submodule_search_locations)))')"
export CUDA_HOME
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
"$PYTHON" -m scaffolds.transformer.validation.benchmark_gdn_precision \
  --output "scaffolds/transformer/validation/runs/gdn-precision-benchmark-$SLURM_JOB_ID/benchmark.json" "$@"
