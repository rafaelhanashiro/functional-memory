#!/usr/bin/env bash
# Direct upstream FLA/cuDNN and matched full-model baselines; argument: launch commit.
#SBATCH --job-name=upstream-b300-bench
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b300:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
LAUNCH_COMMIT=$1
python3 -m scaffolds.launch.frozen check --launch-commit "$LAUNCH_COMMIT" \
  --experiment attention_scaling --scaffold transformer
git diff --exit-code "$LAUNCH_COMMIT" -- \
  scaffolds/transformer/validation/configs/upstream-b300-cudnn.yaml \
  scaffolds/transformer/validation/configs/upstream-b300-mosaic.yaml
source scaffolds/transformer/launchers/kennel_runtime.sh
JPY="$PWD/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
TPY="$PWD/scaffolds/transformer/benchmarks/upstream_runtime/.venv/bin/python"
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TRANSFORMER_HOST_STAGING=0 SHARED_TRANSFORMER_SOURCE_COMMIT="$LAUNCH_COMMIT"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/upstream-bench-${SLURM_JOB_ID}.XXXXXX")"
export TMPDIR XDG_CACHE_HOME="$TMPDIR/xdg"
export JAX_COMPILATION_CACHE_DIR="$PWD/.cache/upstream-b300/jax"
trap 'rm -rf -- "$TMPDIR"' EXIT
OUT="$PWD/scaffolds/transformer/validation/runs/upstream-b300"
LOG_DIR="$PWD/scaffolds/transformer/validation/logs/upstream-b300/$SLURM_JOB_ID"
mkdir -p "$OUT" "$LOG_DIR" "$JAX_COMPILATION_CACHE_DIR" "$XDG_CACHE_HOME"
select_toolkit() {
  CUDA_HOME="$(transformer_cuda_home "$1")"
  export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
}
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
select_toolkit "$JPY"
"$JPY" -m pytest -q \
  'scaffolds/transformer/tests/test_softmax_backends.py::test_cudnn_matches_manual_outputs_input_and_parameter_gradients[64]' \
  'scaffolds/transformer/tests/test_softmax_backends.py::test_cudnn_production_context_forward_and_backward_are_finite[64]' \
  --junitxml="$LOG_DIR/cudnn-tests.xml"
"$JPY" -m scaffolds.transformer.benchmarks.benchmark_upstream_jax \
  --output "$OUT/jax-kernels.json" --batch 64 --warmup 5 --iterations 30 --rounds 2
select_toolkit "$TPY"
"$TPY" -m scaffolds.transformer.benchmarks.benchmark_upstream_torch \
  --mode kernel --output "$OUT/torch-kernels.json" --batch 64 \
  --warmup 5 --iterations 30 --rounds 2 --cuda-graph
# Reverse the mechanism order for the second set; every full-model trial is a fresh process.
for trial in cudnn-1 mosaic-1 fla-1 fla-2 mosaic-2 cudnn-2; do
  mechanism="${trial%-*}"
  if [[ "$mechanism" == fla ]]; then
    select_toolkit "$TPY"
    "$TPY" -m scaffolds.transformer.benchmarks.benchmark_upstream_torch \
      --mode model --output "$OUT/$trial.json" --seed 0 --batch 64 --global-batch 256 \
      --warmup 5 --iterations 30 --manifest /data/cl/u/rafah/home/datasets/fineweb_12b/train/manifest.json
  else
    select_toolkit "$JPY"
    "$JPY" -m scaffolds.transformer.benchmarks.benchmark_training_throughput \
      --config "scaffolds/transformer/validation/configs/upstream-b300-$mechanism.yaml" \
      --warmup 5 --iterations 30 --scratch "$LOG_DIR/$trial" --output "$OUT/$trial.json"
  fi
done
