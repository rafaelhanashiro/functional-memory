#!/bin/bash
#SBATCH --job-name=gdn-occupancy
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=samoyed
#SBATCH --time=00:30:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the worktree}"
OUTPUT="${1:?Expected benchmark JSON output path}"
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$PWD/scaffolds/transformer/.venv/bin/python"
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_enable_command_buffer="
export XLA_PYTHON_CLIENT_PREALLOCATE=false
CUDA_HOME="$("$PYTHON" -c 'import importlib.util; print(next(iter(importlib.util.find_spec("nvidia.cuda_nvcc").submodule_search_locations)))')"
export CUDA_HOME
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
BENCHMARK_TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/gdn-occupancy-${SLURM_JOB_ID}.XXXXXX")"
export TMPDIR="$BENCHMARK_TMPDIR"
trap 'rm -rf -- "$BENCHMARK_TMPDIR"' EXIT
"$PYTHON" -m scaffolds.transformer.validation.benchmark_polynomial_gdn_occupancy --output "$OUTPUT" --batch 8 --length 2048 --heads 16 --width 64 --degree 3 --repeats 5
