#!/usr/bin/env bash
# Triton polynomial-GDN profiling and precision experiments on one B300.
# Usage: sbatch ... benchmark_poly_gdn_tuning.sbatch COMMIT [profile|confirm|diagnose] [PRECISION] [WARPS]
#SBATCH --job-name=poly-gdn-tuning
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b300:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:50:00
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
LAUNCH_COMMIT=${1:?Expected committed source}
PHASE=${2:-profile}
python3 -m scaffolds.launch.frozen check --launch-commit "$LAUNCH_COMMIT" \
  --experiment attention_scaling --scaffold transformer
git diff --exit-code "$LAUNCH_COMMIT" -- \
  scaffolds/transformer/validation/configs/poly-gdn-tuning-dk16.yaml \
  scaffolds/transformer/validation/configs/poly-gdn-tuning-dk64.yaml
source scaffolds/transformer/launchers/kennel_runtime.sh
JPY="$PWD/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
TPY="$PWD/scaffolds/transformer/benchmarks/upstream_runtime/.venv/bin/python"
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TRANSFORMER_HOST_STAGING=0 SHARED_TRANSFORMER_SOURCE_COMMIT="$LAUNCH_COMMIT"
export POLY_GDN_VALUE_PRECISION=tf32x3 POLY_GDN_NUM_WARPS=2
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
POLY_JOB_TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/poly-gdn-${SLURM_JOB_ID}.XXXXXX")"
export TMPDIR="$POLY_JOB_TMPDIR" XDG_CACHE_HOME="$POLY_JOB_TMPDIR/xdg"
export JAX_COMPILATION_CACHE_DIR="$PWD/.cache/poly-gdn-tuning/jax"
export EJKERNEL_TRITON_CACHE_COMPILES=1
# The bridge key does not include helper-function source; isolate by frozen commit.
export EJKERNEL_TRITON_CACHE_DIR="$PWD/.cache/poly-gdn-tuning/bridge/$LAUNCH_COMMIT"
export TRITON_CACHE_DIR="$PWD/.cache/poly-gdn-tuning/triton"
trap 'rm -rf -- "$POLY_JOB_TMPDIR"' EXIT
OUT="$PWD/scaffolds/transformer/validation/runs/poly-gdn-tuning/$SLURM_JOB_ID"
LOG_DIR="$PWD/scaffolds/transformer/validation/logs/poly-gdn-tuning/$SLURM_JOB_ID"
mkdir -p "$OUT" "$LOG_DIR" "$XDG_CACHE_HOME" "$JAX_COMPILATION_CACHE_DIR"
select_toolkit() {
  CUDA_HOME="$(transformer_cuda_home "$1")"
  export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
}
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
case "$PHASE" in
  diagnose)
    select_toolkit "$JPY"
    export CUDA_LAUNCH_BLOCKING=1
    set +e
    for trial in matmul:4 forward:4 gradient:4 forward:2 gradient:2; do
      test_case=${trial%:*}
      warps=${trial#*:}
      "$JPY" -m scaffolds.transformer.benchmarks.diagnose_poly_gdn \
        --case "$test_case" --warps "$warps" > "$LOG_DIR/$test_case-w$warps.log" 2>&1
      echo "${test_case}_w${warps}_exit=$?"
    done
    set -e
    ;;
  profile)
    select_toolkit "$JPY"
    "$JPY" -m pytest -q scaffolds/transformer/tests/test_polynomial_gdn_kernel.py \
      scaffolds/transformer/tests/test_cubic_gdn_kernel.py \
      --maxfail=1 --junitxml="$LOG_DIR/exact-tests.xml"
    # Candidate numerical rejection is evidence; retain it and still measure the alternatives.
    set +e
    "$JPY" -m pytest -q scaffolds/transformer/tests/test_polynomial_gdn_precision.py \
      --junitxml="$LOG_DIR/precision-tests.xml"
    PRECISION_TEST_EXIT=$?
    for precision in bf16_read bf16_values; do
      POLY_GDN_VALUE_PRECISION="$precision" "$JPY" -m pytest -q \
        scaffolds/transformer/tests/test_polynomial_gdn_kernel.py \
        scaffolds/transformer/tests/test_cubic_gdn_kernel.py \
        -k 'bfloat16 or bf16' --junitxml="$LOG_DIR/$precision-stress-tests.xml"
      echo "${precision}_stress_test_exit=$?"
    done
    set -e
    echo "precision_test_exit=$PRECISION_TEST_EXIT"
    select_toolkit "$TPY"
    for width in 16 64; do
      "$TPY" -m scaffolds.transformer.benchmarks.benchmark_upstream_torch \
        --mode kernel --implementations fla --output "$OUT/fla-k${width}-b16.json" \
        --batch 16 --key-dim "$width" --value-dim 64 --warmup 5 --iterations 30 --rounds 2 --cuda-graph
    done
    select_toolkit "$JPY"
    set +e
    "$JPY" -m scaffolds.transformer.benchmarks.benchmark_poly_gdn_tuning \
      --output "$OUT/poly-kernels.json" --batches 16 --key-dims 16 64 --value-dim 64 \
      --precisions tf32x3 bf16_read bf16_values --warps 1 2 --reference-warps 2 \
      --warmup 5 --iterations 20 --rounds 2 --profile-stages
    KERNEL_EXIT=$?
    set -e
    echo "kernel_matrix_exit=$KERNEL_EXIT; inspect per-variant numerical outcomes before confirmation"
    ;;
  confirm)
    PRECISION=${3:?Expected selected precision}
    WARPS=${4:?Expected selected warp count}
    select_toolkit "$JPY"
    "$JPY" -m scaffolds.transformer.benchmarks.benchmark_poly_gdn_tuning \
      --output "$OUT/poly-kernels.json" --batches 64 --key-dims 16 64 --value-dim 64 \
      --precisions tf32x3 "$PRECISION" --warps 2 "$WARPS" --reference-warps 2 \
      --warmup 5 --iterations 30 --rounds 2
    select_toolkit "$TPY"
    for width in 16 64; do
      "$TPY" -m scaffolds.transformer.benchmarks.benchmark_upstream_torch \
        --mode kernel --implementations fla --output "$OUT/fla-k${width}-b64.json" \
        --batch 64 --key-dim "$width" --value-dim 64 --warmup 5 --iterations 30 --rounds 2
    done
    select_toolkit "$JPY"
    for trial in 1 2; do
      if [[ "$trial" == 1 ]]; then settings=(exact candidate); else settings=(candidate exact); fi
      for width in 16 64; do
      for setting in "${settings[@]}"; do
        if [[ "$setting" == exact ]]; then
          export POLY_GDN_VALUE_PRECISION=tf32x3 POLY_GDN_NUM_WARPS=2
        else
          export POLY_GDN_VALUE_PRECISION="$PRECISION" POLY_GDN_NUM_WARPS="$WARPS"
        fi
        "$JPY" -m scaffolds.transformer.benchmarks.benchmark_training_throughput \
          --config "scaffolds/transformer/validation/configs/poly-gdn-tuning-dk${width}.yaml" \
          --warmup 5 --iterations 30 --scratch "$LOG_DIR/k${width}-$setting-$trial" \
          --output "$OUT/training-k${width}-$setting-$trial.json"
      done
      done
    done
    ;;
  *) echo "Unknown phase: $PHASE" >&2; exit 2 ;;
esac
