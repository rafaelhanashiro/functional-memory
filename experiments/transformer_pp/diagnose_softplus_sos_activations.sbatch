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
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$EXECUTION_ROOT/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
PROBE_TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/query-probe-${SLURM_JOB_ID}.XXXXXX")"
export TMPDIR="$PROBE_TMPDIR"
trap 'rm -rf -- "$PROBE_TMPDIR"' EXIT
RUN_ROOT="$EXECUTION_ROOT/experiments/transformer_pp/runs/$RUN_ID"
RESULT=0
# Qualify the transformed SoS kernels used by these two saved checkpoints.
"$PYTHON" -m pytest -c scaffolds/transformer/pyproject.toml --import-mode=importlib -q \
  'scaffolds/transformer/tests/test_mosaic_softplus_sos.py::test_composed_forward_and_generator_vjp_match_direct_squares[bf16x3-1-8]' \
  'scaffolds/transformer/tests/test_mosaic_softplus_sos.py::test_composed_forward_and_generator_vjp_match_direct_squares[bf16x3-1-12]' \
  'scaffolds/transformer/tests/test_mosaic_softplus_sos.py::test_production_context_dispatch_and_vjp[3.0-1-8]' \
  'scaffolds/transformer/tests/test_mosaic_softplus_sos.py::test_production_context_dispatch_and_vjp[3.0-1-12]' \
  --junitxml="$RUN_ROOT/.runtime/acceptance/probe_gpu.xml" || RESULT=$?
if [[ "$RESULT" == 0 ]]; then
  "$PYTHON" -m experiments.transformer_pp.diagnose_query_activations --run "$RUN_ROOT" || RESULT=$?
fi
"$PYTHON" -m scaffolds.launch.finalize --run "$RUN_ROOT" --exit-code "$RESULT"
exit "$RESULT"
