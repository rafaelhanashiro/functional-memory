#!/usr/bin/env bash
# Matched single-B300 full-training-loop timings; arguments: launch commit.
#SBATCH --job-name=gdn-mosaic-b300-bench
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
python3 -m scaffolds.launch.frozen check --launch-commit "$LAUNCH_COMMIT" \
  --experiment attention_scaling --scaffold transformer
# Validation records are mutable, but these benchmark input recipes are frozen.
git diff --exit-code "$LAUNCH_COMMIT" -- \
  scaffolds/transformer/validation/configs/gdn-mosaic-b300-gdn.yaml \
  scaffolds/transformer/validation/configs/gdn-mosaic-b300-mosaic.yaml
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$PWD/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TRANSFORMER_HOST_STAGING=0 SHARED_TRANSFORMER_SOURCE_COMMIT="$LAUNCH_COMMIT"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/gdn-mosaic-bench-${SLURM_JOB_ID}.XXXXXX")"
export TMPDIR XDG_CACHE_HOME="$TMPDIR/xdg"
export JAX_COMPILATION_CACHE_DIR="$PWD/.cache/gdn-mosaic-b300/jax"
trap 'rm -rf -- "$TMPDIR"' EXIT
OUT="$PWD/scaffolds/transformer/validation/runs/gdn-mosaic-b300"
LOG_DIR="$PWD/scaffolds/transformer/validation/logs/gdn-mosaic-b300/$SLURM_JOB_ID"
mkdir -p "$OUT" "$LOG_DIR" "$JAX_COMPILATION_CACHE_DIR" "$XDG_CACHE_HOME"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
"$PYTHON" -m pytest -q \
  scaffolds/transformer/tests/test_gdn_tanh_write_kernel.py \
  scaffolds/transformer/tests/test_mosaic_query.py \
  -k 'production_length_dense_k64_v64 or production_context_matches_triton' \
  --junitxml="$LOG_DIR/kernel-tests.xml"
# Reverse order in round two to expose drift; each invocation starts a fresh process.
for trial in gdn-1 mosaic-1 mosaic-2 gdn-2; do
  mechanism="${trial%-*}"
  "$PYTHON" -m scaffolds.transformer.benchmarks.benchmark_training_throughput \
    --config "scaffolds/transformer/validation/configs/gdn-mosaic-b300-$mechanism.yaml" \
    --warmup 5 --iterations 30 --scratch "$LOG_DIR/$trial" --output "$OUT/$trial.json"
done
