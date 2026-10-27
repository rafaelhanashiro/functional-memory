#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@180
set -euo pipefail
EXECUTION_ROOT="${1:?Expected execution checkout}"
EXPECTED_COMMIT="${2:?Expected source commit}"
RUN_ID="${3:?Expected run ID}"
cd "$EXECUTION_ROOT"
python3 scaffolds/launch/frozen.py check --launch-commit "$EXPECTED_COMMIT" --experiment transformer_200m_4b --scaffold transformer
export PYTHONPATH="$EXECUTION_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DISTILLATION_SOURCE_COMMIT="$EXPECTED_COMMIT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}" MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 MPLBACKEND=Agg
# Submission/record preparation runs on CPU; the allocated job must use CUDA.
unset JAX_PLATFORM_NAME
export JAX_PLATFORMS=cuda
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$EXECUTION_ROOT/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
DISTILL_TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/coefficient-distill-${SLURM_JOB_ID}.XXXXXX")"
export TMPDIR="$DISTILL_TMPDIR"
export JAX_COMPILATION_CACHE_DIR="$EXECUTION_ROOT/.cache/coefficient-distillation/jax"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
RUN_ROOT="$EXECUTION_ROOT/experiments/transformer_200m_4b/runs/$RUN_ID"
CHILD=""
cleanup() { rm -rf -- "$DISTILL_TMPDIR"; }
trap cleanup EXIT
request_requeue() {
  if [[ -n "$CHILD" ]]; then
    kill -USR1 "$CHILD" || true
    RESULT=0
    wait "$CHILD" || RESULT=$?
    "$PYTHON" -m scaffolds.launch.finalize --run "$RUN_ROOT" --exit-code "$RESULT"
  fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap request_requeue USR1
"$PYTHON" -m experiments.transformer_200m_4b.coefficient_distillation --run "$RUN_ROOT" &
CHILD=$!
RESULT=0
wait "$CHILD" || RESULT=$?
CHILD=""
"$PYTHON" -m scaffolds.launch.finalize --run "$RUN_ROOT" --exit-code "$RESULT"
exit "$RESULT"
