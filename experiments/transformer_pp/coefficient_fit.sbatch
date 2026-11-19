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
shift 3
cd "$EXECUTION_ROOT"
python3 scaffolds/launch/frozen.py check --launch-commit "$EXPECTED_COMMIT" --experiment transformer_pp --scaffold transformer
export PYTHONPATH="$EXECUTION_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export COEFFICIENT_ORACLE_SOURCE_COMMIT="$EXPECTED_COMMIT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 MPLBACKEND=Agg
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$EXECUTION_ROOT/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
ORACLE_TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/coefficient-fit-${SLURM_JOB_ID}.XXXXXX")"
export TMPDIR="$ORACLE_TMPDIR"
export JAX_COMPILATION_CACHE_DIR="$EXECUTION_ROOT/.cache/coefficient-fit/jax"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
RUN_ROOT="$EXECUTION_ROOT/experiments/transformer_pp/runs/$RUN_ID"
CHILD=""
cleanup() { rm -rf -- "$ORACLE_TMPDIR"; }
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
"$PYTHON" -m scaffolds.transformer.training.coefficient_fit --run "$RUN_ROOT" "$@" &
CHILD=$!
RESULT=0
wait "$CHILD" || RESULT=$?
CHILD=""
if (( RESULT == 0 )); then
  JAX_PLATFORMS=cpu "$PYTHON" experiments/transformer_pp/results/diagnostics/overfitting/coefficient_oracle/analyze_batch_coefficients.py \
    --run "$RUN_ROOT" --if-applicable || RESULT=$?
fi
"$PYTHON" -m scaffolds.launch.finalize --run "$RUN_ROOT" --exit-code "$RESULT"
exit "$RESULT"
