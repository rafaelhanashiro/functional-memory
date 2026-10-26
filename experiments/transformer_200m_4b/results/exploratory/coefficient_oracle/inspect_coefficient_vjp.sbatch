#!/bin/bash
# One GPU, no training. Submit after the selected fitting run completes.
# sbatch --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=01:00:00 ... \
#   <this-script> <worktree> <committed-source> <query-run-id>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --open-mode=append
set -euo pipefail
EXECUTION_ROOT="${1:?Expected execution checkout}"
EXPECTED_COMMIT="${2:?Expected committed source}"
RUN_ID="${3:?Expected completed query fitting run ID}"
shift 3
for OPTION in "$@"; do
  case "$OPTION" in
    --record-parity-discrepancies|--overwrite) ;;
    *) echo "Unsupported diagnostic option: $OPTION" >&2; exit 1 ;;
  esac
done
if [[ ! "$RUN_ID" =~ ^r[0-9a-f]{12}$ ]]; then echo 'Invalid run ID' >&2; exit 1; fi
cd "$EXECUTION_ROOT"
TOPIC=experiments/transformer_200m_4b/results/exploratory/coefficient_oracle
python3 scaffolds/launch/frozen.py check --launch-commit "$EXPECTED_COMMIT" --experiment transformer_200m_4b --scaffold transformer
for NAME in inspect_coefficient_vjp.py inspect_coefficient_vjp.sbatch inspect_batch_fit.py analyze_batch_coefficients.py; do
  git cat-file -e "$EXPECTED_COMMIT:$TOPIC/$NAME"
  git diff --exit-code "$EXPECTED_COMMIT" -- "$TOPIC/$NAME"
done
GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
if [[ "$GPU_NAMES" != *B300* && "$GPU_NAMES" != *H100* ]]; then
  echo "Use a qualified B300 or H100 GPU, allocated: $GPU_NAMES" >&2
  exit 1
fi
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$EXECUTION_ROOT/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
export PYTHONPATH="$EXECUTION_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
PROBE_TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/coefficient-vjp-${SLURM_JOB_ID}.XXXXXX")"
export TMPDIR="$PROBE_TMPDIR"
export JAX_COMPILATION_CACHE_DIR="$EXECUTION_ROOT/.cache/coefficient-fit/jax"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
cleanup() { rm -rf -- "$PROBE_TMPDIR"; }
trap cleanup EXIT
RUN_ROOT="$EXECUTION_ROOT/experiments/transformer_200m_4b/runs/$RUN_ID"
mkdir -p "$RUN_ROOT/.runtime"
"$PYTHON" -m experiments.transformer_200m_4b.results.exploratory.coefficient_oracle.inspect_coefficient_vjp \
  --run "$RUN_ROOT" --sequence-index 0 --uniform-queries 16 "$@" \
  2>&1 | tee -a "$RUN_ROOT/.runtime/coefficient-vjp-probe.log"
