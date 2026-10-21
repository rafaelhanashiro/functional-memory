#!/usr/bin/env bash
# Bounded numerical diagnostic; no training or throughput claims.
#SBATCH --job-name=poly-gdn-split-check
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b300:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
LAUNCH_COMMIT=${1:?Expected committed source}
python3 -m scaffolds.launch.frozen check --launch-commit "$LAUNCH_COMMIT" \
  --experiment attention_scaling --scaffold transformer
git diff --exit-code "$LAUNCH_COMMIT" -- \
  scaffolds/transformer/validation/diagnose_poly_gdn_batch_split.py \
  scaffolds/transformer/validation/diagnose_poly_gdn_batch_split.sbatch
source scaffolds/transformer/launchers/kennel_runtime.sh
JPY="$PWD/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" TRANSFORMER_HOST_STAGING=0
export SHARED_TRANSFORMER_SOURCE_COMMIT="$LAUNCH_COMMIT"
export POLY_GDN_VALUE_PRECISION=tf32x3 POLY_GDN_NUM_WARPS=2 POLY_GDN_MIN_CTAS_PER_SM=0
export JAX_TRITON_PTXAS_EXTRA_FLAGS=""
unset JAX_TRITON_PTXAS_DEVICE_DEBUG
CUDA_HOME="$(transformer_cuda_home "$JPY")"
export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="
POLY_DIAGNOSTIC_TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/poly-gdn-split-${SLURM_JOB_ID}.XXXXXX")"
export TMPDIR="$POLY_DIAGNOSTIC_TMPDIR" XDG_CACHE_HOME="$POLY_DIAGNOSTIC_TMPDIR/xdg"
export JAX_COMPILATION_CACHE_DIR="$PWD/.cache/poly-gdn-tuning/jax"
export EJKERNEL_TRITON_CACHE_COMPILES=1
export EJKERNEL_TRITON_CACHE_DIR="$PWD/.cache/poly-gdn-tuning/bridge/$LAUNCH_COMMIT"
export TRITON_CACHE_DIR="$PWD/.cache/poly-gdn-tuning/triton"
trap 'rm -rf -- "$POLY_DIAGNOSTIC_TMPDIR"' EXIT
OUT="$PWD/scaffolds/transformer/validation/runs/poly-gdn-tuning/$SLURM_JOB_ID"
mkdir -p "$OUT" "$XDG_CACHE_HOME" "$JAX_COMPILATION_CACHE_DIR"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
"$JPY" -m scaffolds.transformer.validation.diagnose_poly_gdn_batch_split \
  --output "$OUT/batch-split-diagnostic.json"
