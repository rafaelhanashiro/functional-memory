#!/bin/bash
# Stress every Mosaic kernel in its own process at both weight precisions; a process that
# prints nothing for 120 s is reported as hung. KERNELS defaults to the GPU's kernels.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$PWD/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export CUDA_HOME PYTHONPATH="$PWD" JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "gpu=$GPU commit=$(git rev-parse HEAD) job=$SLURM_JOB_ID"
case "$GPU" in *B300*) DEFAULT="forward dq dk dv" ;; *) DEFAULT="forward dq dkv" ;; esac
for precision in bf16 bf16x3; do
  for family in query taylor; do
    for kernel in ${KERNELS:-$DEFAULT}; do
      log="$(mktemp)"
      "$PYTHON" scaffolds/transformer/validation/stress_mosaic.py "$family" "$kernel" "$precision" \
        "${CALLS:-100000}" > "$log" 2>&1 &
      pid=$!; last=""; stall=0
      while kill -0 "$pid" 2>/dev/null; do
        sleep 10
        now="$(tail -1 "$log")"
        if [[ "$now" == "$last" ]]; then stall=$((stall + 10)); else stall=0; last="$now"; fi
        if (( stall >= 120 )); then echo "HUNG $family $kernel $precision after: $now"; kill -9 "$pid"; break; fi
      done
      wait "$pid"; status=$?
      echo "$family $kernel $precision exit=$status $(grep -v hwloc "$log" | tail -1)"
      rm -f "$log"
    done
  done
done
