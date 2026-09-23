#!/bin/bash
#SBATCH --job-name=unnormalized-attention
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:b300:4
#SBATCH --time=2-00:00:00
#SBATCH --requeue
set -euo pipefail
EXECUTION_ROOT="${1:?Expected frozen experiment checkout}"
PROJECT_ROOT="${2:?Expected durable artifact root}"
EXPECTED_COMMIT="${3:?Expected frozen source commit}"
CONFIG="${4:?Expected absolute resolved recipe}"
GPU_COUNT="${5:-4}"
GPU_TYPE="${6:-B300}"
cd "$EXECUTION_ROOT"
if [[ "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
  echo "Frozen execution checkout changed: $EXECUTION_ROOT" >&2
  exit 1
fi
PYTHON="$PROJECT_ROOT/.envs/shared_transformer/bin/python"
"$PYTHON" - "$PROJECT_ROOT" "$EXPECTED_COMMIT" "$CONFIG" "$GPU_COUNT" "$GPU_TYPE" <<'PY'
import hashlib, json, os, subprocess, sys
from datetime import UTC, datetime
from pathlib import Path
root, source, config, gpu_count, gpu_type = sys.argv[1:]
directory = Path(root) / '.cache/unnormalized_attention/submissions'
directory.mkdir(parents=True, exist_ok=True)
job = os.environ['SLURM_JOB_ID']
restart = os.environ.get('SLURM_RESTART_COUNT', '0')
value = {
    'source_commit': source, 'config_path': config,
    'config_sha256': hashlib.sha256(Path(config).read_bytes()).hexdigest(),
    'home_cluster': 'kennel', 'artifact_root': root,
    'slurm_job_id': job, 'slurm_restart_count': restart,
    'slurm_partition': os.environ.get('SLURM_JOB_PARTITION'),
    'gpu_count': int(gpu_count), 'gpu_type': gpu_type,
    'started_utc': datetime.now(UTC).isoformat(),
    'allocation': subprocess.check_output(['scontrol', 'show', 'job', '-o', job], text=True).strip(),
}
(directory / f'{job}-{restart}.json').write_text(json.dumps(value, indent=2) + '\n')
PY
exec bash "$EXECUTION_ROOT/scaffolds/transformer/studies/unnormalized_attention/run_kennel.sh" \
  "$CONFIG" "$PROJECT_ROOT/.cache/unnormalized_attention" "$GPU_COUNT" "$GPU_TYPE" "$PYTHON"
