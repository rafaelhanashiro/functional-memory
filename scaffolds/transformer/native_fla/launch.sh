#!/bin/bash
# Submit with python -m scaffolds.transformer.native_fla.launch, from its worktree.
set -euo pipefail
RUN_ROOT="${1:?Expected the canonical run directory}"
LAUNCH_COMMIT="${2:?Expected the frozen launch commit}"
PHASE="${3:-production}"
EXECUTION_ROOT="${SLURM_SUBMIT_DIR:?Expected a Slurm allocation}"
cd "$EXECUTION_ROOT"
PYTHON="$EXECUTION_ROOT/scaffolds/transformer/native_fla/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo 'Missing native FLA environment; run uv sync --project scaffolds/transformer/native_fla --locked --link-mode hardlink' >&2
  exit 1
fi
if [[ ! "$RUN_ROOT" =~ ^experiments/([A-Za-z0-9_]+)/runs/r[0-9a-f]{12}$ ]]; then
  echo "Expected a canonical experiment run directory: $RUN_ROOT" >&2
  exit 1
fi
EXPERIMENT="${BASH_REMATCH[1]}"
if [[ "$PHASE" != acceptance && "$PHASE" != production ]]; then
  echo "Unknown launch phase: $PHASE" >&2
  exit 1
fi
export PYTHONPATH="$EXECUTION_ROOT" PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export NATIVE_FLA_SOURCE_COMMIT="$LAUNCH_COMMIT" SHARED_TRANSFORMER_SOURCE_COMMIT="$LAUNCH_COMMIT"
export TRANSFORMER_CLUSTER=kennel TRANSFORMER_GPU_TYPE=B300 TRANSFORMER_CUDA_ENVIRONMENT=native_fla_cuda13
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
JOB_TMPDIR=""
QUALIFYING=1
ALREADY_FINISHED=0
TRAINER_PID=""
STOP_REQUESTED=0
SIGNAL_GENERATION=0
request_checkpoint() {
  STOP_REQUESTED=1
  SIGNAL_GENERATION=$((SIGNAL_GENERATION + 1))
  if [[ -n "$TRAINER_PID" ]] && kill -0 "$TRAINER_PID" 2>/dev/null; then
    kill -USR1 "$TRAINER_PID" 2>/dev/null || true
  fi
}
trap request_checkpoint USR1
run_trainer() {
  # If the advance signal arrived during qualification, requeue before starting
  # another worker. Never send a signal to a not-yet-initialized child.
  if (( STOP_REQUESTED )); then return 75; fi
  # Ignore USR1 through interpreter startup; the trainer installs its checkpoint
  # handler before updates and also checks Slurm's absolute end-time deadline.
  (trap '' USR1; exec "$@") &
  TRAINER_PID=$!
  local result=0 before_wait
  while true; do
    # bash wait returns immediately when the shell handles USR1. Keep waiting
    # for the trainer to finish its update and durable checkpoint.
    before_wait=$SIGNAL_GENERATION
    if wait "$TRAINER_PID"; then result=0; else result=$?; fi
    if (( SIGNAL_GENERATION == before_wait )); then break; fi
  done
  TRAINER_PID=""
  return "$result"
}
finish() {
  local result=$? finalizer_result=0
  trap - EXIT
  local finish_args=()
  if (( QUALIFYING )); then finish_args+=(--acceptance); fi
  if (( ! ALREADY_FINISHED )); then
    "$PYTHON" -m scaffolds.transformer.native_fla.launch --run "$RUN_ROOT" --event finish \
      --exit-code "$result" "${finish_args[@]}" || finalizer_result=$?
  fi
  if [[ -n "$JOB_TMPDIR" ]]; then rm -rf -- "$JOB_TMPDIR"; fi
  if (( finalizer_result != 0 )); then exit "$finalizer_result"; fi
  if (( result == 75 && ! QUALIFYING )); then
    if scontrol requeue "$SLURM_JOB_ID"; then exit 0; fi
    "$PYTHON" -m scaffolds.transformer.native_fla.launch --run "$RUN_ROOT" --event finish --exit-code 1
    exit 1
  fi
  exit "$result"
}
trap finish EXIT
python3 scaffolds/launch/frozen.py check --launch-commit "$LAUNCH_COMMIT" --experiment "$EXPERIMENT" --scaffold transformer
GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u)"
if [[ "$GPU_NAMES" != *B300* ]] || [[ "$(wc -l <<<"$GPU_NAMES")" != 1 ]]; then
  echo "The qualified native FLA runtime requires B300; allocated: $GPU_NAMES" >&2
  exit 1
fi
TRANSFORMER_ALLOCATION_SECONDS="$("$PYTHON" -m scaffolds.launch.allocation "$SLURM_JOB_ID")"
export TRANSFORMER_ALLOCATION_SECONDS
JOB_TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/native-fla-${SLURM_JOB_ID}.XXXXXX")"
export TMPDIR="$JOB_TMPDIR"
export TRITON_CACHE_DIR="$EXECUTION_ROOT/.cache/native-fla/triton"
mkdir -p "$TRITON_CACHE_DIR"
CUDA_HOME="$("$PYTHON" - <<'PY'
from pathlib import Path
import sysconfig
toolkit = Path(sysconfig.get_path("purelib")) / "nvidia" / "cu13"
if not (toolkit / "bin" / "ptxas").is_file():
    raise RuntimeError("Missing the locked CUDA 13 toolkit")
print(toolkit)
PY
)"
export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
echo "run=$RUN_ROOT phase=$PHASE job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} launch_commit=$LAUNCH_COMMIT gpu_type=B300 cuda_environment=native_fla_cuda13 allocation_seconds=$TRANSFORMER_ALLOCATION_SECONDS"
"$PYTHON" -m scaffolds.transformer.native_fla.launch --run "$RUN_ROOT" --event start \
  --launch-commit "$LAUNCH_COMMIT" --acceptance
# This entrypoint reuses success only when the source, recipe, data and runtime
# identity match. Every production allocation passes this guard before training.
if [[ "$PHASE" == production ]]; then
  "$PYTHON" -m scaffolds.transformer.native_fla.acceptance --config "$RUN_ROOT/config.yaml" \
    --output "$RUN_ROOT/.runtime/acceptance" >>"$RUN_ROOT/.runtime/acceptance.log" 2>&1
else
  "$PYTHON" -m scaffolds.transformer.native_fla.acceptance --config "$RUN_ROOT/config.yaml" \
    --output "$RUN_ROOT/.runtime/acceptance"
fi
"$PYTHON" -m scaffolds.transformer.native_fla.launch --run "$RUN_ROOT" --event finish --acceptance --exit-code 0
if [[ "$PHASE" == acceptance ]]; then ALREADY_FINISHED=1; exit 0; fi
QUALIFYING=0
"$PYTHON" -m scaffolds.transformer.native_fla.launch --run "$RUN_ROOT" --event start --launch-commit "$LAUNCH_COMMIT"
run_trainer "$PYTHON" -m scaffolds.transformer.native_fla.train --config "$RUN_ROOT/config.yaml"
