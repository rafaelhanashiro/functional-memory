#!/bin/bash
#SBATCH --job-name=pa-softmax-b300
#SBATCH --partition=standard
#SBATCH --nodelist=akita
#SBATCH --gres=gpu:b300:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/data/cl/u/rafah/home/projects/functional-memory/results/logs/polynomial_attention/200m-softmax-seed0/slurm-%j.out
#SBATCH --error=/data/cl/u/rafah/home/projects/functional-memory/results/logs/polynomial_attention/200m-softmax-seed0/slurm-%j.out
set -euo pipefail
PROJECT_ROOT=/data/cl/u/rafah/home/projects/functional-memory
EXECUTION_ROOT="$PROJECT_ROOT/worktrees/polynomial_queue"
RUN_ROOT="$PROJECT_ROOT/results/polynomial_attention/200m-softmax-seed0"
EXPECTED_COMMIT=1a8836a75836671cdf83a3ab7763425a6a88cac7
PYTHON_BIN="$PROJECT_ROOT/.envs/polynomial_attention/bin/python"
export PROJECT_ROOT RUN_ROOT
cd "$EXECUTION_ROOT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]
export POLYNOMIAL_ATTENTION_SOURCE_COMMIT="$EXPECTED_COMMIT"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$EXECUTION_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export JAX_COMPILATION_CACHE_DIR="$PROJECT_ROOT/.cache/polynomial_attention/jax"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache/polynomial_attention/xdg"
export TMPDIR="$PROJECT_ROOT/.cache/polynomial_attention/tmp"
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export MPLBACKEND=Agg
"$PYTHON_BIN" -m experiments.polynomial_attention.preflight "$PROJECT_ROOT/results/polynomial_attention/sweep.json" --check-only
"$PYTHON_BIN" - <<'PY'
import hashlib, json, os, subprocess
from pathlib import Path
import jax
root = Path(os.environ['PROJECT_ROOT'])
run = Path(os.environ['RUN_ROOT'])
migration = json.loads((run / 'hardware_migration.json').read_text())
assert migration['status'] in ('ready', 'running', 'complete'), migration['status']
if migration['status'] == 'complete':
    assert (run / 'completion.json').is_file()
assert migration['new_job_id'] == os.environ['SLURM_JOB_ID']
active_jobs = subprocess.check_output(['squeue', '-h', '-u', 'rafah', '-o', '%A'], text=True).split()
assert migration['previous_job_id'] not in active_jobs, 'Previous writer is still active'
for name, expected in migration['protected_sha256'].items():
    assert hashlib.sha256((run / name).read_bytes()).hexdigest() == expected, name
checkpoints = [int(p.name) for p in (run / 'checkpoints').iterdir()
               if p.name.isdigit() and (p / '_CHECKPOINT_METADATA').is_file()]
assert checkpoints and max(checkpoints) >= migration['resume_checkpoint_step']
benchmark = json.loads((root / 'results/polynomial_attention_b300/train-step.json').read_text())
assert benchmark['status'] == 'complete'
assert benchmark['provenance']['source_commit'] == migration['source_commit']
assert not benchmark['provenance']['source_dirty']
devices = jax.devices()
assert len(devices) == 4 and all('B300' in d.device_kind for d in devices), devices
launch = json.loads((run / 'launch.json').read_text())
assert launch['slurm_job_id'] == os.environ['SLURM_JOB_ID']
print('migration_hardware_passed=true devices=' + str([d.device_kind for d in devices]), flush=True)
print('migration_minimum_restored_step=' + str(migration['resume_checkpoint_step']), flush=True)
PY
nvidia-smi --query-gpu=name,uuid,memory.total,power.limit,driver_version --format=csv
# Retain the original attempt within the final job's log before removing the
# superseded standalone log on successful completion.
if [[ -f "$PROJECT_ROOT/results/logs/polynomial_attention/200m-softmax-seed0/slurm-199652.out" ]]; then
  echo 'BEGIN prior H100 attempt 199652, retained for provenance'
  cat "$PROJECT_ROOT/results/logs/polynomial_attention/200m-softmax-seed0/slurm-199652.out"
  echo 'END prior H100 attempt; beginning B300 continuation'
fi
"$PYTHON_BIN" -m experiments.polynomial_attention.train "$RUN_ROOT/config.yaml"
"$PYTHON_BIN" - <<'PY'
import datetime, json, os, subprocess
from pathlib import Path
run = Path(os.environ['RUN_ROOT'])
def write_atomic(path, data):
    temp = path.with_suffix(path.suffix + '.migration-tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)
migration_path = run / 'hardware_migration.json'
migration = json.loads(migration_path.read_text())
completion = json.loads((run / 'completion.json').read_text())
assert completion['slurm_job_id'] == os.environ['SLURM_JOB_ID']
assert completion['final_step'] == 19074 and completion['restored']
assert completion['restored_step'] >= migration['resume_checkpoint_step']
assert completion['source_commit'] == migration['source_commit']
migration.update(status='complete', completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 restored_step=completion['restored_step'], final_step=completion['final_step'],
                 current_scheduler_record=subprocess.check_output(['scontrol', 'show', 'job', os.environ['SLURM_JOB_ID'], '-o'], text=True).strip(),
                 previous_attempt_log_included_in= str(run.parent.parent / 'logs/polynomial_attention/200m-softmax-seed0' / ('slurm-' + os.environ['SLURM_JOB_ID'] + '.out')))
accounting = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', migration['previous_job_id'] + ',' + os.environ['SLURM_JOB_ID'], '--format=JobID,State,ElapsedRaw'], text=True).splitlines()
migration['allocation_accounting_at_training_completion'] = accounting
migration['total_allocation_seconds_at_training_completion'] = sum(int(row.split('|')[2]) for row in accounting if row.strip())
write_atomic(migration_path, migration)
metadata_path = run / 'metadata.json'
metadata = json.loads(metadata_path.read_text())
metadata['hardware_migration_path'] = str(migration_path)
metadata['execution_job_ids'] = [migration['previous_job_id'], os.environ['SLURM_JOB_ID']]
metadata['hardware_history'] = ['4 NVIDIA H100 SXM on dogo', '4 NVIDIA B300 SXM6 AC on akita']
write_atomic(metadata_path, metadata)
Path(migration['previous_launch']['log_path']).unlink(missing_ok=True)
print('b300_continuation_complete=true', flush=True)
PY
"$PYTHON_BIN" -m experiments.polynomial_attention.plot --manifest "$PROJECT_ROOT/results/polynomial_attention/sweep.json"
