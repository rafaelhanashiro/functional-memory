#!/bin/bash
# Actual 4B-study attention shape, isolated from production runs.
#SBATCH --job-name=profile-query-key-4b
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b300:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
LAUNCH_COMMIT="${1:?Expected committed source}"
MODE="${2:-all}"
SAMPLES="${3:-10}"
if [[ "$MODE" != all && "$MODE" != kernel && "$MODE" != training ]]; then exit 2; fi
python3 scaffolds/launch/frozen.py check --launch-commit "$LAUNCH_COMMIT" --experiment transformer_200m_4b --scaffold transformer
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$PWD/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 WANDB_MODE=disabled
TASK_TMP=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/query-key-profile-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf "$TASK_TMP"' EXIT
export TMPDIR="$TASK_TMP" JAX_COMPILATION_CACHE_DIR="$PWD/.cache/query-key-profile/jax"
CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
OUTPUT="scaffolds/transformer/validation/runs/query-key-4b-b300"
mkdir -p "$OUTPUT"
nvidia-smi --query-gpu=index,uuid,name,temperature.gpu,clocks.sm,power.draw --format=csv
if [[ "$MODE" != training ]]; then
  "$PYTHON" -m scaffolds.transformer.benchmarks.benchmark_query_key_profile \
    --output "$OUTPUT/kernel.json" --profile-dir "$PWD/.cache/query-key-profile/profiles" \
    --include-query-mosaic
fi
if [[ "$MODE" != kernel ]]; then
 for RUN_ID in ra2b851b0dd7e r833fbb6eedcb rfd48f7bbe9f4; do
  "$PYTHON" -m scaffolds.transformer.benchmarks.benchmark_training_throughput \
    --config "experiments/transformer_200m_4b/runs/$RUN_ID/config.yaml" \
    --experiment transformer_200m_4b --warmup 3 --iterations "$SAMPLES" \
    --scratch "$PWD/.cache/query-key-profile/pilot-$SLURM_JOB_ID-$RUN_ID" \
    --output "$OUTPUT/training-$RUN_ID.json"
 done
fi
