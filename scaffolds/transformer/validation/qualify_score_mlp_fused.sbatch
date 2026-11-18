#!/bin/bash
# Qualify the fused score-MLP kernel on one GPU. Step "tests": the GPU tests of both score-MLP
# test files, then their error record, runs/score-mlp/fused-gpu-qualification.json. Step
# "benchmark": per-layer times and the tile sweep, runs/score-mlp/fused-throughput.json.
# Arguments: the committed source, then the step. Submit from the worktree root, for example:
#   sbatch --partition=debug --gres=gpu:h100:1 --exclude=collie,samoyed,doodle,terrier,shiba \
#     --output=scaffolds/transformer/validation/logs/score-mlp/%j.log \
#     scaffolds/transformer/validation/qualify_score_mlp_fused.sbatch "$(git rev-parse HEAD)" tests
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the worktree root}"
LAUNCH_COMMIT="${1:?Pass the committed source}"
STEP="${2:?Expected tests or benchmark}"
python3 scaffolds/launch/frozen.py check --launch-commit "$LAUNCH_COMMIT" --experiment transformer_pp \
  --scaffold transformer
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$PWD/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" PYTHONUNBUFFERED=1
TASK_TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/score-mlp-fused-${SLURM_JOB_ID}.XXXXXX")"
trap 'rm -rf -- "$TASK_TMPDIR"' EXIT
export TMPDIR="$TASK_TMPDIR" XDG_CACHE_HOME="$TASK_TMPDIR/xdg"
RECORDS=scaffolds/transformer/validation/runs/score-mlp
LOGS="scaffolds/transformer/validation/logs/score-mlp/$SLURM_JOB_ID"
mkdir -p "$RECORDS" "$LOGS"
printf 'job=%s node=%s source=%s step=%s environment=%s\n' "$SLURM_JOB_ID" "$SLURMD_NODENAME" "$LAUNCH_COMMIT" \
  "$STEP" "$TRANSFORMER_CUDA_ENVIRONMENT"
nvidia-smi --query-gpu=name,driver_version,pci.bus_id --format=csv,noheader
status=0
case "$STEP" in
  tests)
    "$PYTHON" -m pytest -c scaffolds/transformer/pyproject.toml -q -rfE --import-mode=importlib \
      -p no:cacheprovider -m gpu scaffolds/transformer/tests/test_score_mlp_attention.py \
      scaffolds/transformer/tests/test_score_mlp_fused.py --junitxml="$LOGS/gpu-tests.xml" || status=$?
    "$PYTHON" -m scaffolds.transformer.validation.qualify_score_mlp_fused --launch-commit "$LAUNCH_COMMIT" \
      --junit "$LOGS/gpu-tests.xml" --exit-code "$status" --output "$RECORDS/fused-gpu-qualification.json" \
      || status=$?
    ;;
  benchmark)
    "$PYTHON" -m scaffolds.transformer.benchmarks.benchmark_score_mlp --output "$RECORDS/fused-throughput.json" \
      || status=$?
    ;;
  *)
    echo "Unknown step: $STEP" >&2
    status=2
    ;;
esac
exit "$status"
