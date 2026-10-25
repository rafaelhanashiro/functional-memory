#!/usr/bin/env bash
#SBATCH --job-name=qk-head-precision
#SBATCH --partition=debug
#SBATCH --nodelist=mastiff
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=scaffolds/transformer/validation/runs/qk-head-controls/precision-%j.log
set -euo pipefail
export PYTHONPATH="$PWD"
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$PWD/scaffolds/transformer/.venv/bin/python"
export JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
PROBE_TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/qk-head-precision-${SLURM_JOB_ID}.XXXXXX")"
trap 'rm -rf -- "$PROBE_TMPDIR"' EXIT
export TMPDIR="$PROBE_TMPDIR"
export JAX_COMPILATION_CACHE_DIR="$PWD/.cache/qk-head-precision/jax"
export XDG_CACHE_HOME="$PWD/.cache/qk-head-precision/xdg"
mkdir -p "$JAX_COMPILATION_CACHE_DIR" "$XDG_CACHE_HOME"
"$PYTHON" -u scaffolds/transformer/validation/runs/qk-head-controls/precision_probe.py \
  scaffolds/transformer/validation/runs/qk-head-controls/precision-report.json
