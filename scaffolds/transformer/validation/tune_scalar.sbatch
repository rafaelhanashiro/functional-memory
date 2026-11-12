#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
python3 -m scaffolds.launch.frozen check --launch-commit "${SHARED_TRANSFORMER_SOURCE_COMMIT:?}" \
  --experiment attention_scaling --scaffold transformer
unset LD_LIBRARY_PATH
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$PWD/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
export PYTHONPATH="$PWD" EJKERNEL_TRITON_CACHE_COMPILES=1 EJKERNEL_TRITON_CACHE_DIR="$PWD/.cache/scalar-triton"
unset JAX_PLATFORMS
TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/scalar-tuning-${SLURM_JOB_ID}.XXXXXX")"
export TMPDIR
trap 'rm -rf -- "$TMPDIR"' EXIT
OUT="scaffolds/transformer/validation/runs/scalar-triton/$SLURM_JOB_ID"
mkdir -p "$OUT"
nvidia-smi --query-gpu=index,uuid,name,driver_version --format=csv > "$OUT/gpus.csv"
scontrol show job "$SLURM_JOB_ID" > "$OUT/allocation.txt"
for kernel in rbf3 rbf20; do
  "$PYTHON" -m scaffolds.transformer.validation.tune_scalar --kernel "$kernel" \
    --output "$OUT/$kernel-tuning.json" > "$OUT/$kernel-tuning.log" 2>&1
done
