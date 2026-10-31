#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --open-mode=append
set -euo pipefail
EXECUTION_ROOT="${1:?Expected checkout}"
EXPECTED_COMMIT="${2:?Expected commit}"
RUN_ID="${3:?Expected run ID}"
cd "$EXECUTION_ROOT"
python3 scaffolds/launch/frozen.py check --launch-commit "$EXPECTED_COMMIT" --experiment transformer_pp --scaffold transformer
export PYTHONPATH="$EXECUTION_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$EXECUTION_ROOT/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
EVALUATION_TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/generator-transfer-${SLURM_JOB_ID}.XXXXXX")"
export TMPDIR="$EVALUATION_TMPDIR"
trap 'rm -rf -- "$EVALUATION_TMPDIR"' EXIT
RUN_ROOT="$EXECUTION_ROOT/experiments/transformer_pp/runs/$RUN_ID"
RESULT=0
"$PYTHON" -m experiments.transformer_pp.evaluate_generator_transfer --run "$RUN_ROOT" || RESULT=$?
"$PYTHON" -m scaffolds.launch.finalize --run "$RUN_ROOT" --exit-code "$RESULT"
exit "$RESULT"
