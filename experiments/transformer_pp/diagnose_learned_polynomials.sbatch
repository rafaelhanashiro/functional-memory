#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --open-mode=append
set -euo pipefail

cd "${1:?Expected worktree root}"
launch_commit="${2:?Expected launch commit}"
run="experiments/transformer_pp/runs/${3:?Expected diagnostic run ID}"
python3 scaffolds/launch/frozen.py check --launch-commit "$launch_commit" --experiment transformer_pp --scaffold transformer
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
source scaffolds/transformer/launchers/kennel_runtime.sh
python_bin="$PWD/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
CUDA_HOME="$(transformer_cuda_home "$python_bin")"
export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
status=0
"$python_bin" -m experiments.transformer_pp.diagnose_learned_polynomials --run "$run" || status=$?
"$python_bin" -m scaffolds.launch.finalize --run "$run" --exit-code "$status"
exit "$status"
