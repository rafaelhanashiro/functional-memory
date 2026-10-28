#!/bin/bash
# ORCD launcher for one fixed-input coefficient fit; submit it through
# `launch_coefficient_fit.py submit <run> --cluster orcd`, which requests one typed
# Hopper GPU in mit_preemptable or mit_normal_gpu for 12 hours with a 6-hour minimum.
# Ten minutes before the limit the fit saves its exact state and the job requeues;
# preemption also requeues, resuming from the latest checkpoint (every 100 updates).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@600
set -euo pipefail
EXECUTION_ROOT="${1:?Expected execution checkout}"
EXPECTED_COMMIT="${2:?Expected source commit}"
RUN_ID="${3:?Expected run ID}"
GPU_TYPES="${4:?Expected the accepted GPU type, for example H100}"
shift 4
cd "$EXECUTION_ROOT"
PYTHON="$EXECUTION_ROOT/scaffolds/transformer/.venv/bin/python"
# ORCD's system python3 is 3.6, older than frozen.py supports, so use the locked interpreter.
"$PYTHON" scaffolds/launch/frozen.py check --launch-commit "$EXPECTED_COMMIT" --experiment transformer_200m_4b --scaffold transformer
RUN_ROOT="$EXECUTION_ROOT/experiments/transformer_200m_4b/runs/$RUN_ID"

GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u)"
GPU_TYPE=""
for candidate in ${GPU_TYPES//,/ }; do
  if [[ "$GPU_NAMES" == *"$candidate"* ]]; then GPU_TYPE="$candidate"; break; fi
done
if [[ -z "$GPU_TYPE" || "$(wc -l <<<"$GPU_NAMES")" != 1 ]]; then
  echo "Allocated $GPU_NAMES; this fit accepts only $GPU_TYPES" >&2
  exit 1
fi
echo "run=$RUN_ID job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} launch_commit=$EXPECTED_COMMIT gpu_type=$GPU_TYPE"

CACHE="$EXECUTION_ROOT/.cache/coefficient-fit"
export PYTHONPATH="$EXECUTION_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export COEFFICIENT_ORACLE_SOURCE_COMMIT="$EXPECTED_COMMIT"
export TRANSFORMER_CLUSTER=orcd TRANSFORMER_GPU_TYPE="$GPU_TYPE"
export JAX_PLATFORMS=cuda TRANSFORMER_HOST_STAGING="${TRANSFORMER_HOST_STAGING:-0}" XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 MPLBACKEND=Agg
export JAX_COMPILATION_CACHE_DIR="$CACHE/jax" XDG_CACHE_HOME="$CACHE/xdg" TMPDIR="$CACHE/tmp/$SLURM_JOB_ID"
# jaxlib assembles Triton custom-call PTX with the first ptxas among its CUDA roots; pin the locked toolkit.
CUDA_HOME="$("$PYTHON" -c 'import importlib.util; print(next(iter(importlib.util.find_spec("nvidia.cuda_nvcc").submodule_search_locations)))')"
export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas"
mkdir -p "$JAX_COMPILATION_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"

source scaffolds/launch/watchdog.sh
finish_record() {
  local result=$?
  trap - EXIT
  stop_watchdog
  rm -rf -- "$TMPDIR"
  "$PYTHON" -m scaffolds.launch.finalize --run "$RUN_ROOT" --exit-code "$result" || true
  exit "$result"
}
trap finish_record EXIT
request_requeue() {
  echo "Saving the fit's exact state and requeueing job $SLURM_JOB_ID before its time limit"
  if [[ -n "$STEP_PID" ]]; then
    kill -USR1 "$STEP_PID" || true
    wait "$STEP_PID" || true
    STEP_PID=""
  fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap request_requeue USR1
start_watchdog "$RUN_ROOT/training.log"

RESULT=0
run_step "$PYTHON" -m scaffolds.transformer.training.coefficient_fit --run "$RUN_ROOT" "$@" || RESULT=$?
STEP_PID=""
if (( RESULT == 0 )); then
  run_step env JAX_PLATFORMS=cpu "$PYTHON" experiments/transformer_200m_4b/results/exploratory/coefficient_oracle/analyze_batch_coefficients.py \
    --run "$RUN_ROOT" --if-applicable || RESULT=$?
fi
exit "$RESULT"
