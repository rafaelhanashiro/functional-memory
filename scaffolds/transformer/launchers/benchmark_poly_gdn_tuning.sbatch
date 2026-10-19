#!/usr/bin/env bash
# Triton polynomial-GDN profiling and precision experiments on one B300.
# Usage: sbatch ... benchmark_poly_gdn_tuning.sbatch COMMIT PHASE [PRECISION] [WARPS] [MIN_CTAS]
# PHASE: profile, tune, qualify, confirm, or diagnose.
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
export POLY_GDN_VALUE_PRECISION=tf32x3 POLY_GDN_NUM_WARPS=2 POLY_GDN_MIN_CTAS_PER_SM=0
# Keep compiler flags controlled across the occupancy/precision comparison.
export JAX_TRITON_PTXAS_EXTRA_FLAGS=""
unset JAX_TRITON_PTXAS_DEVICE_DEBUG
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
  tune)
    select_toolkit "$JPY"
    "$JPY" -m scaffolds.transformer.benchmarks.benchmark_poly_gdn_tuning \
      --output "$OUT/poly-kernels.json" --batches 16 --key-dims 16 64 --value-dim 64 \
      --precisions tf32x3 bf16_values --warps 2 --min-ctas 0 1 \
      --reference-warps 2 --reference-min-ctas 0 \
      --warmup 5 --iterations 20 --rounds 2 --profile-stages
    select_toolkit "$TPY"
    for width in 16 64; do
      "$TPY" -m scaffolds.transformer.benchmarks.benchmark_upstream_torch \
        --mode kernel --implementations fla --output "$OUT/fla-k${width}-b16.json" \
        --batch 16 --key-dim "$width" --value-dim 64 --warmup 5 --iterations 30 --rounds 2
    done
    ;;
  qualify)
    PRECISION=${3:?Expected selected precision}
    export POLY_GDN_NUM_WARPS=${4:?Expected selected warp count}
    export POLY_GDN_MIN_CTAS_PER_SM=${5:?Expected selected minimum CTA guidance}
    select_toolkit "$JPY"
    "$JPY" -m pytest -q scaffolds/transformer/tests/test_polynomial_gdn_kernel.py \
      scaffolds/transformer/tests/test_cubic_gdn_kernel.py \
      --maxfail=1 --junitxml="$LOG_DIR/exact-tests.xml"
    POLY_GDN_VALUE_PRECISION="$PRECISION" "$JPY" -m pytest -q \
      scaffolds/transformer/tests/test_polynomial_gdn_kernel.py \
      scaffolds/transformer/tests/test_cubic_gdn_kernel.py \
      -k 'bfloat16 or bf16' --junitxml="$LOG_DIR/selected-stress-tests.xml"
    ;;
  confirm)
    PRECISION=${3:?Expected selected precision}
    WARPS=${4:?Expected selected warp count}
    MIN_CTAS=${5:?Expected selected minimum CTA guidance}
    select_toolkit "$JPY"
    "$JPY" -m scaffolds.transformer.benchmarks.benchmark_poly_gdn_tuning \
      --output "$OUT/poly-kernels.json" --batches 64 --key-dims 16 64 --value-dim 64 \
      --precisions tf32x3 "$PRECISION" --warps 2 "$WARPS" --reference-warps 2 \
      --min-ctas 0 "$MIN_CTAS" --reference-min-ctas 0 \
      --warmup 5 --iterations 30 --rounds 2
    select_toolkit "$TPY"
    for width in 16 64; do
      "$TPY" -m scaffolds.transformer.benchmarks.benchmark_upstream_torch \
        --mode kernel --implementations fla --output "$OUT/fla-k${width}-b64.json" \
        --batch 64 --key-dim "$width" --value-dim 64 --warmup 5 --iterations 30 --rounds 2
    done
    for trial in 1 2; do
      if [[ "$trial" == 1 ]]; then
        widths=(16 64); settings=(exact candidate native)
      else
        widths=(64 16); settings=(native candidate exact)
      fi
      for width in "${widths[@]}"; do
      for setting in "${settings[@]}"; do
        if [[ "$setting" == native ]]; then
          select_toolkit "$TPY"
          export POLY_GDN_VALUE_PRECISION=tf32x3 POLY_GDN_NUM_WARPS=2 POLY_GDN_MIN_CTAS_PER_SM=0
          "$TPY" -m scaffolds.transformer.benchmarks.benchmark_upstream_torch \
            --mode model --output "$OUT/training-k${width}-native-${trial}.json" \
            --batch 64 --global-batch 256 --length 2048 --heads 16 --key-dim "$width" --value-dim 64 \
            --hidden-size 1024 --layers 10 --intermediate-size 3008 --vocab-size 32000 \
            --warmup 5 --iterations 30 --rounds 1 --seed 0 \
            --learning-rate 3e-4 --min-learning-rate 3e-5 --lr-warmup-steps 400 --total-steps 19074 \
            --manifest /data/cl/u/rafah/home/datasets/fineweb_12b/train/manifest.json
          continue
        fi
        select_toolkit "$JPY"
        if [[ "$setting" == exact ]]; then
          export POLY_GDN_VALUE_PRECISION=tf32x3 POLY_GDN_NUM_WARPS=2 POLY_GDN_MIN_CTAS_PER_SM=0
        else
          export POLY_GDN_VALUE_PRECISION="$PRECISION" POLY_GDN_NUM_WARPS="$WARPS" POLY_GDN_MIN_CTAS_PER_SM="$MIN_CTAS"
        fi
        # Keep slow controls bounded. Each report records its actual sample count.
        # Estimate only selects the window; all reported times measure real updates.
        read -r WARMUP ITERATIONS < <(python3 - "$OUT/poly-kernels.json" "$width" <<'PY'
import json, math, os, sys
report = json.load(open(sys.argv[1]))
shape = report["shapes"][f"b64_t2048_h16_k{sys.argv[2]}_v64"]
matches = [v for v in shape["variants"].values()
           if v["precision"] == os.environ["POLY_GDN_VALUE_PRECISION"]
           and v["num_warps"] == int(os.environ["POLY_GDN_NUM_WARPS"])
           and v["min_ctas_per_sm"] == int(os.environ["POLY_GDN_MIN_CTAS_PER_SM"])]
assert len(matches) == 1 and matches[0]["accuracy_passed"]
seconds = matches[0]["operations"]["forward_backward"]["summary"]["mean_seconds"]
estimated_update = 40 * seconds + 1  # Ten layers, four microbatches, plus other work.
assert math.isfinite(estimated_update) and estimated_update > 0
print(2 if estimated_update > 20 else 5, max(3, min(30, int(120 / estimated_update))))
PY
        )
        echo "training_k${width}_${setting}_trial${trial}: warmup=$WARMUP measured=$ITERATIONS"
        "$JPY" -m scaffolds.transformer.benchmarks.benchmark_training_throughput \
          --config "scaffolds/transformer/validation/configs/poly-gdn-tuning-dk${width}.yaml" \
          --warmup "$WARMUP" --iterations "$ITERATIONS" --scratch "$LOG_DIR/k${width}-$setting-$trial" \
          --output "$OUT/training-k${width}-$setting-$trial.json"
      done
      done
    done
    ;;
  *) echo "Unknown phase: $PHASE" >&2; exit 2 ;;
esac
