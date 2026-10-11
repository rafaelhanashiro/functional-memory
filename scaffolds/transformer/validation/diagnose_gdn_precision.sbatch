#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export GDN_DIAGNOSTIC_SOURCE="${1:?Expected source commit}"
python3 scaffolds/launch/frozen.py check --launch-commit "$GDN_DIAGNOSTIC_SOURCE" --experiment attention_scaling --scaffold transformer
PYTHON="$PWD/scaffolds/transformer/.venv/bin/python"
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 TRANSFORMER_HOST_STAGING=0
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
source scaffolds/transformer/launchers/kennel_runtime.sh
CUDA_HOME="$("$PYTHON" -c 'import importlib.util; print(next(iter(importlib.util.find_spec("nvidia.cuda_nvcc").submodule_search_locations)))')"
export CUDA_HOME
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
for precision in tf32 ieee; do
  export TRITON_F32_DEFAULT="$precision"
  export EJKERNEL_TRITON_CACHE_DIR="$PWD/.cache/gdn-precision-$precision/triton"
  export JAX_COMPILATION_CACHE_DIR="$PWD/.cache/gdn-precision-$precision/jax"
  export TMPDIR="$PWD/.cache/gdn-precision-$precision/tmp"
  mkdir -p "$TMPDIR"
  "$PYTHON" -m scaffolds.transformer.validation.diagnose_gdn_precision \
    --output "scaffolds/transformer/validation/runs/gdn-precision-$SLURM_JOB_ID/$precision.json"
done
