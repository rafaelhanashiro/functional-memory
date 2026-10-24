#!/bin/bash
# Replay actual full-model attention gradients against an independent oracle.
#SBATCH --job-name=diagnose-query-key-b300
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b300:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
LAUNCH_COMMIT="${1:?Expected committed source}"
CONFIG="${2:?Expected frozen config}"
OUTPUT="${3:?Expected fresh diagnostic report path}"
shift 3
python3 scaffolds/launch/frozen.py check --launch-commit "$LAUNCH_COMMIT" --experiment transformer_200m_4b --scaffold transformer
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$PWD/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
# Host callbacks need a CPU backend while CUDA remains the default device.
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda,cpu OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export SHARED_TRANSFORMER_SOURCE_COMMIT="$LAUNCH_COMMIT"
TASK_TMP=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/query-key-roundoff-${SLURM_JOB_ID}-XXXXXX")
trap 'rm -rf "$TASK_TMP"' EXIT
export TMPDIR="$TASK_TMP" JAX_COMPILATION_CACHE_DIR="$PWD/.cache/transformer_200m_4b/jax"
CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
nvidia-smi --query-gpu=index,uuid,name,temperature.gpu,clocks.sm,power.draw --format=csv
"$PYTHON" -m scaffolds.transformer.benchmarks.diagnose_query_key_model_roundoff \
  --config "$CONFIG" --output "$OUTPUT" "$@"
