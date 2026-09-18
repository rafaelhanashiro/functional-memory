#!/bin/bash
#SBATCH --job-name=poly-throughput-check
#SBATCH --partition=standard
#SBATCH --gres=gpu:h100:4
#SBATCH --nodelist=dogo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --output=/data/cl/u/rafah/home/projects/functional-memory/results/logs/polynomial_attention_throughput/slurm-%j.out
#SBATCH --error=/data/cl/u/rafah/home/projects/functional-memory/results/logs/polynomial_attention_throughput/slurm-%j.out
set -euo pipefail
cd /data/cl/u/rafah/home/projects/functional-memory/worktrees/polynomial_throughput
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda
PYTHON_BIN=/data/cl/u/rafah/home/projects/functional-memory/.envs/polynomial_attention/bin/python
REPORT_ROOT=/data/cl/u/rafah/home/projects/functional-memory/results/polynomial_attention_throughput
TRACE_ROOT=/data/cl/u/rafah/home/projects/functional-memory/.cache/polynomial_attention/throughput/final-traces
PILOT_ROOT=/data/cl/u/rafah/home/projects/functional-memory/.cache/polynomial_attention/throughput/packed-pilot-${SLURM_JOB_ID}
EXECUTION_COMMIT="$(git rev-parse HEAD)"
EXECUTION_STATUS="$(git status --porcelain)"
printf 'source_commit=%s\nsource_status=%s\n' "$EXECUTION_COMMIT" "$EXECUTION_STATUS"
[[ "$EXECUTION_COMMIT" == 0f655fb7098ae73e5923fc295fe0b5c3012184ee ]]
[[ -z "$EXECUTION_STATUS" ]]
nvidia-smi --query-gpu=name,uuid,memory.total,power.limit --format=csv
nvidia-smi topo -m
"$PYTHON_BIN" -m pytest -c experiments/polynomial_attention/pyproject.toml experiments/polynomial_attention/tests/test_attention.py experiments/polynomial_attention/tests/test_fused_attention.py -m gpu -q --junitxml="$REPORT_ROOT/gpu-tests.xml"
"$PYTHON_BIN" -m experiments.polynomial_attention.benchmark_train_step --degrees 2 4 8 16 softmax --implementations fused --trace-dir "$TRACE_ROOT" --output "$REPORT_ROOT/train-step.json"
"$PYTHON_BIN" -m experiments.polynomial_attention.analyze_profile "$TRACE_ROOT" --output "$REPORT_ROOT/profile-summary.json"
"$PYTHON_BIN" -m experiments.polynomial_attention.benchmark_packed_run --output "$REPORT_ROOT/packed-pilot.json" --workdir "$PILOT_ROOT"
printf 'throughput_validation_complete=true\n'
