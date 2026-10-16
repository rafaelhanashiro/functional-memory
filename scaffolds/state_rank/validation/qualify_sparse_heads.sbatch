#!/bin/bash
# Submit from the isolated worktree after committing all inputs.
# Arguments: LAUNCH_COMMIT CORPUS_DIRECTORY MANIFEST_SHA256 TOKENS_SHA256
#SBATCH --job-name=sparse-head-qualification
#SBATCH --partition=debug
#SBATCH --nodelist=dogo
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=scaffolds/state_rank/validation/logs/sparse-head-qualification-%j.log
set -euo pipefail

LAUNCH_COMMIT="${1:?Expected the launch commit}"
CORPUS_DIRECTORY="${2:?Expected the token corpus directory}"
MANIFEST_SHA256="${3:?Expected the frozen manifest checksum}"
TOKENS_SHA256="${4:?Expected the frozen token archive checksum}"
EXECUTION_ROOT="${SLURM_SUBMIT_DIR:?Submit from the isolated worktree}"
cd "$EXECUTION_ROOT"
python3 scaffolds/launch/frozen.py check --launch-commit "$LAUNCH_COMMIT" \
  --experiment attention_scaling --scaffold state_rank
# Scaffold validation records are intentionally excluded from the generic frozen
# set. The executable qualification itself must still match its launch commit.
git diff --exit-code "$LAUNCH_COMMIT" -- \
  scaffolds/state_rank/validation/qualify_sparse_heads.py \
  scaffolds/state_rank/validation/qualify_sparse_heads.sbatch
git cat-file -e "$LAUNCH_COMMIT:scaffolds/state_rank/validation/qualify_sparse_heads.py"
git cat-file -e "$LAUNCH_COMMIT:scaffolds/state_rank/validation/qualify_sparse_heads.sbatch"

PYTHON="$EXECUTION_ROOT/scaffolds/state_rank/.venv/bin/python"
export PYTHONPATH="$EXECUTION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 MPLBACKEND=Agg
QUALIFICATION_TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/sparse-head-qualification-${SLURM_JOB_ID}.XXXXXX")"
export TMPDIR="$QUALIFICATION_TMPDIR"
trap 'rm -rf -- "$QUALIFICATION_TMPDIR"' EXIT
OUTPUT_DIRECTORY="$EXECUTION_ROOT/scaffolds/state_rank/validation/logs/sparse-head-${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}"
mkdir -p "$OUTPUT_DIRECTORY"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$PYTHON" -m scaffolds.state_rank.validation.qualify_sparse_heads \
  --source-commit "$LAUNCH_COMMIT" --corpus "$CORPUS_DIRECTORY" \
  --manifest-sha256 "$MANIFEST_SHA256" --tokens-sha256 "$TOKENS_SHA256" \
  --output "$OUTPUT_DIRECTORY"
