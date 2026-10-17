#!/usr/bin/env bash
# One-GPU signed-write qualification and isolated checkpoint throughput replay.
# Arguments: launch commit, original run directory, output directory.
#SBATCH --job-name=gdn-tanh-b300-bench
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b300:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
LAUNCH_COMMIT=$1
ORIGINAL_RUN=$2
OUT=$3
python3 -m scaffolds.launch.frozen check --launch-commit "$LAUNCH_COMMIT" \
  --experiment attention_scaling --scaffold transformer
git diff --exit-code "$LAUNCH_COMMIT" -- scaffolds/transformer/validation/replay_checkpoint.py
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$PWD/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TRANSFORMER_HOST_STAGING=0 SHARED_TRANSFORMER_SOURCE_COMMIT="$LAUNCH_COMMIT"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/gdn-tanh-bench-${SLURM_JOB_ID}.XXXXXX")"
export TMPDIR XDG_CACHE_HOME="$TMPDIR/xdg"
export JAX_COMPILATION_CACHE_DIR="$PWD/.cache/gdn-tanh-b300/jax"
trap 'rm -rf -- "$TMPDIR"' EXIT
mkdir -p "$OUT" "$JAX_COMPILATION_CACHE_DIR" "$XDG_CACHE_HOME"
LOG_DIR="$PWD/scaffolds/transformer/validation/logs/gdn-tanh-b300/$SLURM_JOB_ID"
mkdir -p "$LOG_DIR"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$PYTHON" -m pytest -q scaffolds/transformer/tests/test_gdn_tanh_write_kernel.py \
  --junitxml="$LOG_DIR/kernel-tests.xml"
"$PYTHON" -m scaffolds.transformer.benchmarks.benchmark_checkpoint_replay \
  --run "$ORIGINAL_RUN" --checkpoint 11500 --warmup 5 --iterations 30 \
  --baseline-steps 9000 10000 11000 --scratch "$LOG_DIR/replay" \
  --output "$OUT/benchmark.json"
