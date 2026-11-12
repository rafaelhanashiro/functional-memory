#!/usr/bin/env bash
# A bounded replay of an existing checkpoint; never writes into the parent run.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
RUN_DIR="$PWD/experiments/transformer_pp/runs/${1:?run ID}"
LAUNCH_COMMIT="${2:?launch commit}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python3 -m scaffolds.launch.frozen check --launch-commit "$LAUNCH_COMMIT" --experiment transformer_pp --scaffold transformer
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$PWD/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
export JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
export JAX_COMPILATION_CACHE_DIR="$RUN_DIR/.runtime/jax-cache"
TASK_TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/gate-spike-${SLURM_JOB_ID}.XXXXXX")"
export TMPDIR="$TASK_TMPDIR"
trap 'rm -rf "$TASK_TMPDIR"' EXIT

"$PYTHON" - "$RUN_DIR" "$LAUNCH_COMMIT" <<'PY'
from pathlib import Path
from datetime import datetime, UTC
import hashlib, json, os, subprocess, sys, yaml
from scaffolds.records import load_run, update_record
from scaffolds.launch.frozen import frozen_hash
run, commit = Path(sys.argv[1]), sys.argv[2]
recipe = yaml.safe_load((run/'config.yaml').read_text())
parent = Path(recipe['diagnostic']['parent_run_path'])
assert hashlib.sha256((parent/'config.yaml').read_bytes()).hexdigest() == recipe['diagnostic']['parent_config_sha256']
identity = json.loads((parent/'data_identity.json').read_text())
for split in ('train', 'eval'):
    manifest = Path(recipe['data'][f'{split}_manifest'])
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == identity[f'{split}_manifest_sha256']
attempt = {'launch_commit':commit, 'home_cluster':'kennel', 'slurm_job_id':os.environ['SLURM_JOB_ID'],
           'slurm_partition':os.environ.get('SLURM_JOB_PARTITION'), 'gpu_count':1,
           'gpu_type':subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip(),
           'started_utc':datetime.now(UTC).isoformat(), 'slurm_restart_count':os.environ.get('SLURM_RESTART_COUNT','0')}
update_record(run, {'status':'running', 'source_commit':commit,
    'frozen_execution_sha256':frozen_hash(Path.cwd(),commit,'transformer_pp','transformer'),
    'attempts':[*load_run(run).get('attempts',[]),attempt], 'log_path':str(run/'training.log'),
    'execution':{'environment':{key:os.environ.get(key) for key in ('TRANSFORMER_CUDA_ENVIRONMENT','CUDA_HOME','XLA_FLAGS','XLA_PYTHON_CLIENT_PREALLOCATE')},
                 'native_scaffold_source_commit':recipe['diagnostic']['parent_source_commit']}})
(run/'data_identity.json').write_text(json.dumps(identity,indent=2,sort_keys=True)+'\n')
PY

mapfile -t REPLAY_SETTINGS < <("$PYTHON" - "$RUN_DIR/config.yaml" <<'PY'
import sys,yaml
diagnostic=yaml.safe_load(open(sys.argv[1]))['diagnostic']
print(diagnostic['parent_run_path'])
print('1' if diagnostic.get('retain_spike_state',False) else '0')
print(diagnostic.get('mode','replay'))
print(diagnostic.get('start_step',7500))
print(diagnostic.get('first_batch_update',7501))
print(diagnostic.get('max_batches',256))
print(diagnostic.get('threshold',1.0))
PY
)
PARENT_RUN="${REPLAY_SETTINGS[0]}"
REPLAY_OPTIONS=()
if [[ "${REPLAY_SETTINGS[1]}" == 1 ]]; then REPLAY_OPTIONS+=(--retain-spike-state); fi
set +e
if [[ "${REPLAY_SETTINGS[2]}" == frozen_batches ]]; then
  "$PYTHON" -m experiments.transformer_pp.diagnose_frozen_gate_batches --parent-run "$PARENT_RUN" --output "$RUN_DIR" --checkpoint "${REPLAY_SETTINGS[3]}" --first-batch-update "${REPLAY_SETTINGS[4]}" --max-batches "${REPLAY_SETTINGS[5]}" --threshold "${REPLAY_SETTINGS[6]}"
  RESULT=$?
  if [[ "$RESULT" == 0 ]]; then
    "$PYTHON" -m experiments.transformer_pp.diagnose_gate_vjp --run-dir "$RUN_DIR"
    RESULT=$?
  fi
else
  "$PYTHON" -m experiments.transformer_pp.diagnose_gate_spikes --parent-run "$PARENT_RUN" --output "$RUN_DIR" --checkpoint 7500 --end-step 7570 "${REPLAY_OPTIONS[@]}"
  RESULT=$?
fi
set -e
"$PYTHON" - "$RUN_DIR" "$RESULT" <<'PY'
from pathlib import Path
from datetime import datetime, UTC
import json,sys
from scaffolds.records import capture_artifacts,load_run,update_record
run, code = Path(sys.argv[1]), int(sys.argv[2])
attempts=load_run(run)['attempts']
attempts[-1].update(exit_code=code,finished_utc=datetime.now(UTC).isoformat())
summary_path=run/'summary.json'
summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
checkpoint=summary.get('diagnostic_checkpoint')
checkpoint_path=run/checkpoint['relative_directory'] if checkpoint else None
update_record(run,{'status':'complete' if code == 0 else 'failed', 'attempts':attempts,
                   'error':None if code == 0 else f'Diagnostic process exited {code}',
                   **capture_artifacts(run,checkpoint_directory=checkpoint_path)})
PY
exit "$RESULT"
