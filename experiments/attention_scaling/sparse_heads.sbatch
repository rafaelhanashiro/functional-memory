#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@180
set -euo pipefail
EXECUTION_ROOT="${1:?Expected worktree}"
EXPECTED_COMMIT="${2:?Expected source commit}"
shift 2
cd "$EXECUTION_ROOT"
python3 scaffolds/launch/frozen.py check --launch-commit "$EXPECTED_COMMIT" --experiment attention_scaling --scaffold state_rank
export PYTHONPATH="$EXECUTION_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SPARSE_HEADS_SOURCE_COMMIT="$EXPECTED_COMMIT"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
PYTHON="$EXECUTION_ROOT/scaffolds/state_rank/.venv/bin/python"
SPARSE_TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/sparse-heads-${SLURM_JOB_ID}.XXXXXX")"
export TMPDIR="$SPARSE_TMPDIR"
trap 'rm -rf -- "$SPARSE_TMPDIR"' EXIT
CHILD=""
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
for RUN_ID in "$@"; do
  RUN_ROOT="$EXECUTION_ROOT/experiments/attention_scaling/runs/$RUN_ID"
  "$PYTHON" -m experiments.attention_scaling.sparse_heads --run "$RUN_ROOT" >> "$RUN_ROOT/training.log" 2>&1 &
  CHILD=$!
  RESULT=0
  wait "$CHILD" || RESULT=$?
  CHILD=""
  "$PYTHON" -m scaffolds.launch.finalize --run "$RUN_ROOT" --exit-code "$RESULT"
  if (( RESULT != 0 )); then
    "$PYTHON" -c 'import sys; from scaffolds.records import load_run; sys.exit(load_run(sys.argv[1])["status"] != "numerical_failure")' "$RUN_ROOT" || exit "$RESULT"
  fi
done
