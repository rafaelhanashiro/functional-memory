#!/bin/bash
set -euo pipefail
POLICY="${1:?Expected validation policy}"
PROFILE="${2:?Expected GPU profile}"
mapfile -t LOCATION < <(python3 - "$POLICY" <<'PY'
import json, sys
from pathlib import Path
p=json.loads(Path(sys.argv[1]).read_text())
m=json.loads(Path(p['training_manifest']).read_text())
print(m['project_root']+'/.envs/polynomial_attention/bin/python')
print(p['orchestrator_root']+'/experiments/key_dimension_taylor/hardware_checks')
print(p['training_root'])
print(p['output_root'])
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
STAGE="kernels"
finish() {
  local exit_status=$?
  trap - EXIT TERM INT
  if [[ -n "$CHILD" ]]; then
    kill -TERM -- "-$CHILD" 2>/dev/null || true
    for attempt in {1..15}; do
      if ! kill -0 -- "-$CHILD" 2>/dev/null; then break; fi
      sleep 1
    done
    kill -KILL -- "-$CHILD" 2>/dev/null || true
    wait "$CHILD" 2>/dev/null || true
  fi
  if [[ "$exit_status" != 0 ]]; then
    "$PYTHON" - "$SCRIPTS" "$POLICY" "$PROFILE" "$STAGE" "$exit_status" <<'PY'
import json,os,sys
from datetime import UTC,datetime
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import common
p=json.loads(Path(sys.argv[2]).read_text())
root=Path(p['output_root'])/sys.argv[3]
record={'identity':p['identity'],'profile':sys.argv[3],'stage':sys.argv[4],
        'status':'launcher_exit','failure_kind':'infrastructure','exit_code':int(sys.argv[5]),
        'slurm_job_id':os.environ.get('SLURM_JOB_ID'), 'slurm_restart_count':os.environ.get('SLURM_RESTART_COUNT','0'),
        'recorded_utc':datetime.now(UTC).isoformat()}
if record['exit_code'] in (130,137,143):
    record['status']='interrupted'
else:
    evidence=root/('kernels.json' if record['stage']=='kernels' else 'pilots-shard0.json')
    if evidence.exists():
        result=json.loads(evidence.read_text())
        record['stage_status']=result.get('status')
        if result.get('status') in ('numerical_failure','memory_failure','runtime_ineligible'):
            record['failure_kind']=result['status']
common.write_json(root/'interruptions'/f"{record['slurm_job_id']}-{record['slurm_restart_count']}.json",record)
PY
  fi
  timeout 60s "$PYTHON" "$SCRIPTS/manage.py" --policy "$POLICY" || true
  exit "$exit_status"
}
trap finish EXIT
trap 'exit 143' TERM INT
echo "validation_only profile=$PROFILE job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} node=$SLURMD_NODENAME partition=$SLURM_JOB_PARTITION"
setsid "$PYTHON" "$SCRIPTS/runner.py" --policy "$POLICY" --profile "$PROFILE" --stage kernels &
CHILD=$!
wait "$CHILD"
CHILD=""
STAGE="pilots"
setsid "$PYTHON" "$SCRIPTS/runner.py" --policy "$POLICY" --profile "$PROFILE" --stage pilots --shard 0 &
CHILD=$!
wait "$CHILD"
CHILD=""
STAGE="complete"
