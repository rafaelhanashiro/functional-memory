#!/bin/bash
#SBATCH --job-name=poly-b300-check
#SBATCH --partition=sweep
#SBATCH --nodelist=akita
#SBATCH --gres=gpu:b300:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:20:00
#SBATCH --requeue
#SBATCH --output=/data/cl/u/rafah/home/projects/functional-memory/results/logs/polynomial_attention_b300/slurm-%j.out
#SBATCH --error=/data/cl/u/rafah/home/projects/functional-memory/results/logs/polynomial_attention_b300/slurm-%j.out
set -euo pipefail
cd /data/cl/u/rafah/home/projects/functional-memory/worktrees/polynomial_queue
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
PYTHON_BIN=/data/cl/u/rafah/home/projects/functional-memory/.envs/polynomial_attention/bin/python
REPORT_ROOT=/data/cl/u/rafah/home/projects/functional-memory/results/polynomial_attention_b300
EXPECTED_COMMIT=1a8836a75836671cdf83a3ab7763425a6a88cac7
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain)" ]]
printf 'source_commit=%s\n' "$EXPECTED_COMMIT"
nvidia-smi --query-gpu=name,uuid,memory.total,power.limit,driver_version --format=csv
nvidia-smi topo -m
"$PYTHON_BIN" -m pytest -c experiments/polynomial_attention/pyproject.toml experiments/polynomial_attention/tests/test_attention.py experiments/polynomial_attention/tests/test_fused_attention.py -m gpu -q --junitxml="$REPORT_ROOT/gpu-tests.xml"
"$PYTHON_BIN" -m experiments.polynomial_attention.benchmark_train_step --degrees 2 16 softmax --implementations fused --output "$REPORT_ROOT/train-step.json"
printf 'b300_validation_complete=true\n'
