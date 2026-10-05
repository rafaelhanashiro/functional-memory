#!/usr/bin/env bash
# Replay a retained checkpoint several times in one allocation, each variant with its own
# XLA flags: "default" uses the launcher's runtime; "sync" also runs every collective on
# the compute stream (--xla_gpu_disable_async_collectives=ALLCOLLECTIVES), so no kernel
# overlaps a cuBLAS kernel.
#
#   sbatch -w akita replay_checkpoint.sbatch EXECUTION_ROOT RUN_DIR CHECKPOINT STOP OUT_DIR default sync ...
#
# EXECUTION_ROOT is a checkout at the run's source commit with its own synced environment.
# Submit from the primary checkout's root, which provides replay_checkpoint.py.
#SBATCH --job-name=replay-checkpoint
#SBATCH --partition=long
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
set -euo pipefail
REPLAY="$SLURM_SUBMIT_DIR/scaffolds/transformer/validation/replay_checkpoint.py"
EXECUTION_ROOT=$(cd "$1" && pwd); RUN_DIR=$(cd "$2" && pwd); CHECKPOINT=$3; STOP=$4; OUT_DIR=$5; shift 5
cd "$EXECUTION_ROOT"
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$EXECUTION_ROOT/scaffolds/transformer/.venv/bin/python"
CACHE="$EXECUTION_ROOT/.cache/replay"
mkdir -p "$CACHE/jax" "$CACHE/xdg" "$CACHE/tmp/$SLURM_JOB_ID" "$OUT_DIR"
export PYTHONPATH="$EXECUTION_ROOT" JAX_PLATFORMS=cuda TRANSFORMER_HOST_STAGING="${TRANSFORMER_HOST_STAGING:-0}"
export SHARED_TRANSFORMER_SOURCE_COMMIT="$(git rev-parse HEAD)" WANDB_MODE=disabled
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export JAX_COMPILATION_CACHE_DIR="$CACHE/jax" XDG_CACHE_HOME="$CACHE/xdg" TMPDIR="$CACHE/tmp/$SLURM_JOB_ID"
CUDA_HOME="$("$PYTHON" -c 'import importlib.util; print(next(iter(importlib.util.find_spec("nvidia.cuda_nvcc").submodule_search_locations)))')"
export CUDA_HOME
BASE_XLA_FLAGS="${XLA_FLAGS:-}"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "job=$SLURM_JOB_ID node=$(hostname) gpu=$GPU commit=$SHARED_TRANSFORMER_SOURCE_COMMIT run=$RUN_DIR checkpoint=$CHECKPOINT stop=$STOP"
index=0
for variant in "$@"; do
  index=$((index + 1))
  case "$variant" in
    default) export XLA_FLAGS="$BASE_XLA_FLAGS" ;;
    sync) export XLA_FLAGS="$BASE_XLA_FLAGS --xla_gpu_disable_async_collectives=ALLCOLLECTIVES" ;;
    *) echo "unknown variant $variant" >&2; exit 2 ;;
  esac
  echo "=== replay $index variant=$variant XLA_FLAGS=$XLA_FLAGS"
  "$PYTHON" "$REPLAY" --run "$RUN_DIR" --checkpoint "$CHECKPOINT" --stop "$STOP" \
      --scratch "$OUT_DIR/$SLURM_JOB_ID-$index-$variant" || echo "REPLAY_RESULT {\"status\": \"error\", \"exit\": $?}"
done
