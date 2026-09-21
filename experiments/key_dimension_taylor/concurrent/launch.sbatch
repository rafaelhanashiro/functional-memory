#!/bin/bash
set -euo pipefail
MODE="${1:?Expected qualify or train}"
POLICY="${2:?Expected execution amendment}"
RUN_NAME="${3:?Expected run name}"
SEGMENT="${4:-0}"
# The policy is generated only after both checkouts are committed and frozen.
mapfile -t LOCATION < <(python3 - "$POLICY" <<'PY'
import json, sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text())
m = json.loads(Path(p['training_manifest']).read_text())
print(m['project_root'] + '/.envs/polynomial_attention/bin/python')
print(p['orchestrator_root'] + '/experiments/key_dimension_taylor/concurrent')
print(p['training_root'])
PY
)
PYTHON="${LOCATION[0]}"
SCRIPTS="${LOCATION[1]}"
cd "${LOCATION[2]}"
export PYTHONPATH="${LOCATION[2]}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg
CHILD=""
stop_children() {
  if [[ -n "$CHILD" ]]; then
    kill -TERM -- "-$CHILD" 2>/dev/null || true
    for attempt in {1..15}; do
      if ! kill -0 -- "-$CHILD" 2>/dev/null; then break; fi
      sleep 1
    done
    kill -KILL -- "-$CHILD" 2>/dev/null || true
    wait "$CHILD" 2>/dev/null || true
  fi
}
failed_exit() {
  local exit_status=$?
  trap - EXIT TERM INT
  stop_children
  if [[ "$MODE" == train && "$exit_status" != 0 ]]; then
    timeout 30s "$PYTHON" "$SCRIPTS/production.py" --policy "$POLICY" --run-name "$RUN_NAME" \
      --segment "$SEGMENT" --record-interruption "$exit_status" || true
  fi
  exit "$exit_status"
}
trap failed_exit EXIT
trap 'exit 143' TERM INT
echo "mode=$MODE run=$RUN_NAME job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} node=$SLURMD_NODENAME partition=$SLURM_JOB_PARTITION"
if [[ "$MODE" == qualify ]]; then
  setsid "$PYTHON" "$SCRIPTS/qualification.py" --policy "$POLICY" --profile b300 &
  CHILD=$!
  wait "$CHILD"
  CHILD=""
  "$PYTHON" "$SCRIPTS/schedule.py" --policy "$POLICY" --submit b300
elif [[ "$MODE" == train ]]; then
  setsid "$PYTHON" "$SCRIPTS/production.py" --policy "$POLICY" --run-name "$RUN_NAME" --segment "$SEGMENT" &
  CHILD=$!
  wait "$CHILD"
  CHILD=""
else
  echo "Unknown mode: $MODE" >&2
  exit 1
fi
