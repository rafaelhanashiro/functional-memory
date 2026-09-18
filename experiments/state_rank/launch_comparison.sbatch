#!/bin/bash
#SBATCH --job-name=random-rank
#SBATCH --partition=mit_preemptable
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
set -euo pipefail

EXECUTION_ROOT="${1:?Expected frozen execution checkout}"
PROJECT_ROOT="${2:?Expected retained project root}"
MANIFEST="${3:?Expected comparison manifest}"
CONDITION="${4:?Expected comparison condition}"
EXPECTED_COMMIT="${5:?Expected frozen source commit}"
cd "$EXECUTION_ROOT"
if [[ "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" ]] || [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "Frozen execution checkout changed; refusing to run: $EXECUTION_ROOT" >&2
  exit 1
fi
export PYTHONPATH="$EXECUTION_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
echo "condition=$CONDITION source=$EXPECTED_COMMIT job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0}"
"$PYTHON" -m experiments.state_rank.compare_random --manifest "$MANIFEST" --condition "$CONDITION" --require-cuda
"$PYTHON" -m experiments.state_rank.plot_comparison --results "$(dirname "$MANIFEST")"
