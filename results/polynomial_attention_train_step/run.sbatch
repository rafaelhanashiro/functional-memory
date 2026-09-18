#!/bin/bash
#SBATCH --job-name=poly-train-bench
#SBATCH --partition=standard
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=/data/cl/u/rafah/home/projects/functional-memory/results/logs/polynomial_attention_train_step/slurm-%j.out
#SBATCH --error=/data/cl/u/rafah/home/projects/functional-memory/results/logs/polynomial_attention_train_step/slurm-%j.out
set -euo pipefail
cd /data/cl/u/rafah/home/projects/functional-memory/worktrees/polynomial_attention_fused_benchmark
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda
[[ "$(git rev-parse HEAD)" == 3537262* ]]
[[ -z "$(git status --porcelain)" ]]
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
/data/cl/u/rafah/home/projects/functional-memory/.envs/polynomial_attention/bin/python -m experiments.polynomial_attention.benchmark_train_step --output /data/cl/u/rafah/home/projects/functional-memory/results/polynomial_attention_train_step/benchmark.json
