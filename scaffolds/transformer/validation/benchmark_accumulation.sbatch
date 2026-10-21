#!/bin/bash
# Submit from the worktree, selecting a partition explicitly with sbatch.
# Each case runs in a fresh process; failed/OOM cases remain in the report.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
python3 -m scaffolds.launch.frozen check \
  --launch-commit "${SHARED_TRANSFORMER_SOURCE_COMMIT:?}" \
  --experiment attention_scaling --scaffold transformer
unset LD_LIBRARY_PATH
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$PWD/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
export PYTHONPATH="$PWD"
# Basis restoration may inspect the CPU platform; allow both backends.
unset JAX_PLATFORMS
TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/rbf-accumulation-${SLURM_JOB_ID}.XXXXXX")"
export TMPDIR
trap 'rm -rf -- "$TMPDIR"' EXIT
OUT="scaffolds/transformer/validation/runs/rbf-accumulation/$SLURM_JOB_ID"
mkdir -p "$OUT"
for run in ${RUNS:-r10b2cb22687c r873f76c0a542}; do
  for implementation in python compiled; do
    sizes="1"
    if [[ "$implementation" == compiled ]]; then
      sizes="${MICROBATCHES:-1 2 4 8 16 32 64}"
    fi
    for microbatch in $sizes; do
      name="$run-$implementation-mb$microbatch"
      status=0
      timeout --signal=TERM --kill-after=30 600 "$PYTHON" \
        -m scaffolds.transformer.benchmarks.benchmark_accumulation \
        --config "experiments/attention_scaling/runs/$run/config.yaml" \
        --implementation "$implementation" --micro-batch-size "$microbatch" \
        --output "$OUT/$name.json" --warmup 2 --iterations 3 \
        > "$OUT/$name.log" 2>&1 || status=$?
      printf '%s %s\n' "$name" "$status" >> "$OUT/exit-status.txt"
    done
  done
done
