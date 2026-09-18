#!/bin/bash
#SBATCH --job-name=poly-kernel-check
#SBATCH --partition=debug
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/data/cl/u/rafah/home/projects/functional-memory/results/logs/polynomial_attention_kernel/slurm-%j.out
#SBATCH --error=/data/cl/u/rafah/home/projects/functional-memory/results/logs/polynomial_attention_kernel/slurm-%j.out
set -euo pipefail
cd /data/cl/u/rafah/home/projects/functional-memory/worktrees/polynomial_attention_fused_final
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
PYTHON_BIN=/data/cl/u/rafah/home/projects/functional-memory/.envs/polynomial_attention/bin/python
REPORT_ROOT=/data/cl/u/rafah/home/projects/functional-memory/results/polynomial_attention_kernel
EXECUTION_COMMIT="$(git rev-parse HEAD)"
EXECUTION_STATUS="$(git status --porcelain)"
printf "source_commit=%s\nsource_status=%s\n" "$EXECUTION_COMMIT" "$EXECUTION_STATUS"
[[ "$EXECUTION_COMMIT" == 0b7a37b* ]]
[[ -z "$EXECUTION_STATUS" ]]
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
JAX_PLATFORMS=cuda "$PYTHON_BIN" -m pytest -c experiments/polynomial_attention/pyproject.toml experiments/polynomial_attention/tests/test_attention.py experiments/polynomial_attention/tests/test_fused_attention.py -m gpu -q --junitxml="$REPORT_ROOT/gpu-tests.xml"
JAX_PLATFORMS=cuda "$PYTHON_BIN" -m experiments.polynomial_attention.benchmark_attention --degrees 0 2 4 8 16 --output "$REPORT_ROOT/benchmark.json"
