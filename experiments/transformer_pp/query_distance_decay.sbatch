#!/bin/bash
# Full shared acceptance/training, followed by the requested completed-run comparison.
# Arguments are identical to scaffolds/transformer/launchers/kennel.sbatch.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the isolated experiment worktree}"
CONFIG="${2:?Expected the run config as the second argument}"
RUN_ROOT="$(dirname "$CONFIG")"
PYTHON="$PWD/scaffolds/transformer/.venv/bin/python"
finish_record() {
  local result=$?
  trap - EXIT
  JAX_PLATFORMS=cpu "$PYTHON" -m scaffolds.launch.finalize --run "$RUN_ROOT" --exit-code "$result"
  exit "$result"
}
bash scaffolds/transformer/launchers/kennel.sbatch "$@"
trap finish_record EXIT
export JAX_PLATFORMS=cpu
"$PYTHON" -m scaffolds.records index experiments/transformer_pp
"$PYTHON" -m scaffolds.records refresh experiments/transformer_pp/results/exploratory/query_distance_decay
"$PYTHON" -m experiments.transformer_pp.results.exploratory.query_distance_decay.compare
