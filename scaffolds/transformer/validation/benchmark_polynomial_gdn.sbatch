#!/bin/bash
# Exact full-shape quadratic GDN kernel timing; one GPU is sufficient.
# sbatch ... benchmark_polynomial_gdn.sbatch <committed-source> <output-json> [benchmark options]
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:45:00
set -euo pipefail
SOURCE="${1:?Expected committed source}"
OUTPUT="${2:?Expected benchmark output JSON}"
shift 2
cd "${SLURM_SUBMIT_DIR:?Submit from the experiment worktree}"
python3 scaffolds/launch/frozen.py check --launch-commit "$SOURCE" --experiment attention_scaling --scaffold transformer
PYTHON="$PWD/scaffolds/transformer/.venv/bin/python"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
source scaffolds/transformer/launchers/kennel_runtime.sh
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
export JAX_COMPILATION_CACHE_DIR="$PWD/.cache/polynomial_gdn_benchmark/jax"
export XDG_CACHE_HOME="$PWD/.cache/polynomial_gdn_benchmark/xdg"
BENCHMARK_TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/polynomial-gdn-${SLURM_JOB_ID}.XXXXXX")"
export TMPDIR="$BENCHMARK_TMPDIR"
trap 'rm -rf -- "$BENCHMARK_TMPDIR"' EXIT
CUDA_HOME="$("$PYTHON" -c 'import importlib.util; print(next(iter(importlib.util.find_spec("nvidia.cuda_nvcc").submodule_search_locations)))')"
export CUDA_HOME
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas"
export TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
"$PYTHON" -m scaffolds.transformer.validation.benchmark_polynomial_gdn --output "$OUTPUT" "$@"
