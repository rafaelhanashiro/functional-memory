#!/bin/bash
#SBATCH --job-name=random-rank
#SBATCH --partition=standard
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
set -euo pipefail

EXECUTION_ROOT="${1:?Expected frozen execution checkout}"
PROJECT_ROOT="${2:?Expected retained project root}"
MANIFEST="${3:?Expected comparison manifest}"
CONDITION="${4:?Expected comparison condition}"
EXPECTED_COMMIT="${5:?Expected frozen source commit}"
DRIVER="${6:-compare_random}"
case "$DRIVER" in
  compare_random|compare_deltanet) ;;
  *) echo "Unsupported comparison driver: $DRIVER" >&2; exit 1 ;;
esac
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
# Kennel jobs use models pre-downloaded at their pinned revisions; never fetch mid-run.
export HF_HUB_OFFLINE=1
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
echo "condition=$CONDITION source=$EXPECTED_COMMIT job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0}"
if [[ "$DRIVER" == compare_deltanet && "$CONDITION" == pretrained ]]; then
  "$PYTHON" -m pytest -q -p no:cacheprovider \
    scaffolds/state_rank/tests/test_spectrum.py::test_native_gated_delta_rule_matches_explicit_recurrence \
    scaffolds/state_rank/tests/test_spectrum.py::test_tiny_gdn_cached_states_match_independent_prefixes \
    scaffolds/state_rank/tests/test_deltanet_states.py::test_native_all_layer_states_match_existing_spectra_and_prefix_checks
fi
"$PYTHON" -m "scaffolds.state_rank.$DRIVER" --manifest "$MANIFEST" --condition "$CONDITION" --require-cuda
"$PYTHON" -m scaffolds.state_rank.plot_comparison --results "$(dirname "$MANIFEST")"
