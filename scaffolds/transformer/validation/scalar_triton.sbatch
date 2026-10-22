#!/bin/bash
# One physical H100 for every comparison in this invocation.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
python3 -m scaffolds.launch.frozen check --launch-commit "${SHARED_TRANSFORMER_SOURCE_COMMIT:?}" \
  --experiment attention_scaling --scaffold transformer
unset LD_LIBRARY_PATH
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$PWD/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
export PYTHONPATH="$PWD"
export EJKERNEL_TRITON_CACHE_COMPILES=1 EJKERNEL_TRITON_CACHE_DIR="$PWD/.cache/scalar-triton"
unset JAX_PLATFORMS
TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/scalar-triton-${SLURM_JOB_ID}.XXXXXX")"
export TMPDIR
trap 'rm -rf -- "$TMPDIR"' EXIT
OUT="scaffolds/transformer/validation/runs/scalar-triton/$SLURM_JOB_ID"
mkdir -p "$OUT"
nvidia-smi --query-gpu=index,uuid,name,driver_version --format=csv > "$OUT/gpus.csv"
scontrol show job "$SLURM_JOB_ID" > "$OUT/allocation.txt"
if [[ "${PHASE:-all}" == tests || "${PHASE:-all}" == all ]]; then
  "$PYTHON" -m pytest -x -q -m gpu scaffolds/transformer/tests/test_triton_scalar.py \
    --junitxml="$OUT/tests.xml" > "$OUT/tests.log" 2>&1
fi
if [[ "${PHASE:-all}" == bench || "${PHASE:-all}" == all ]]; then
  for batch in ${BATCHES:-1 4 16 32}; do
    for kernel in ${KERNELS:-rbf3 rbf5 rbf9 rbf20 sine poly3 poly8 exp}; do
      extra=()
      if [[ "$batch" == 16 ]]; then extra+=(--precise); fi
      if [[ "$batch" == 16 && ( "$kernel" == rbf3 || "$kernel" == rbf20 ) ]]; then
        extra+=(--tune)
      fi
      if [[ "$batch" == 16 && ( "$kernel" == rbf20 || "$kernel" == poly3 ) ]]; then
        extra+=(--profile "$OUT/profiles/$kernel")
      fi
      status=0
      timeout --signal=TERM --kill-after=30 1200 "$PYTHON" \
        -m scaffolds.transformer.benchmarks.benchmark_scalar --kernel "$kernel" --batch "$batch" \
        --mode rms --output "$OUT/$kernel-rms-b$batch.json" "${extra[@]}" \
        > "$OUT/$kernel-rms-b$batch.log" 2>&1 || status=$?
      printf '%s %s\n' "$kernel-rms-b$batch" "$status" >> "$OUT/exit-status.txt"
    done
  done
  for kernel in exp poly3 poly8; do
    status=0
    timeout --signal=TERM --kill-after=30 1200 "$PYTHON" \
      -m scaffolds.transformer.benchmarks.benchmark_scalar --kernel "$kernel" --batch 16 \
      --mode sum --output "$OUT/$kernel-sum-b16.json" \
      > "$OUT/$kernel-sum-b16.log" 2>&1 || status=$?
    printf '%s %s\n' "$kernel-sum-b16" "$status" >> "$OUT/exit-status.txt"
  done
fi
if [[ "${PHASE:-all}" == train || "${PHASE:-all}" == all ]]; then
  for run in ${RUNS:-r10b2cb22687c r873f76c0a542}; do
    for backend in blocked triton_scalar; do
      status=0
      timeout --signal=TERM --kill-after=30 1500 "$PYTHON" \
        -m scaffolds.transformer.benchmarks.benchmark_accumulation \
        --config "experiments/attention_scaling/runs/$run/config.yaml" \
        --micro-batch-size "${MICROBATCH:-16}" --backend "$backend" \
        --warmup 2 --iterations 5 --output "$OUT/$run-$backend.json" \
        --trace-dir "$OUT/profiles/$run-$backend" > "$OUT/$run-$backend.log" 2>&1 || status=$?
      printf '%s %s\n' "$run-$backend" "$status" >> "$OUT/exit-status.txt"
    done
  done
fi
