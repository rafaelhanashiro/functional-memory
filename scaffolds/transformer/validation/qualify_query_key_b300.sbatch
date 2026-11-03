#!/usr/bin/env bash
# B300 qualification of the fused query-key polynomial kernel at the Transformer++ shape: BF16,
# context 2048, Q/K and value width 64, degrees 2-12, with the tuned backward tiles. Runs every
# GPU-marked query-key check: FP64 gradients, full-context comparisons with the blocked and
# original geometries, output RMSNorm, causality, nonfinite isolation, and QK RMSNorm composition.
# Submit from the root of a clean checkout or detached worktree of the committed source:
#   sbatch --partition=debug --gres=gpu:b300:1 --cpus-per-task=8 --mem=96G --time=00:30:00 \
#     --job-name=qk-b300-qualify \
#     --output=scaffolds/transformer/validation/logs/query-key-b300/%x-%j.log \
#     scaffolds/transformer/validation/qualify_query_key_b300.sbatch "$(git rev-parse HEAD)"
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the checkout root}"
LAUNCH_COMMIT="${1:?Pass the committed source}"
git diff --exit-code "$LAUNCH_COMMIT" -- scaffolds/transformer ':!scaffolds/transformer/validation'
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$PWD/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1
TASK_TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/query-key-b300-${SLURM_JOB_ID}.XXXXXX")"
trap 'rm -rf -- "$TASK_TMPDIR"' EXIT
export TMPDIR="$TASK_TMPDIR" XDG_CACHE_HOME="$PWD/.cache/query-key-b300/xdg"
CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
export PATH="$CUDA_HOME/bin:$PATH"
LOGS=scaffolds/transformer/validation/logs/query-key-b300
mkdir -p "$LOGS"
printf 'job=%s node=%s source=%s environment=%s\n' "$SLURM_JOB_ID" "$(hostname -s)" "$LAUNCH_COMMIT" \
  "$TRANSFORMER_CUDA_ENVIRONMENT"
nvidia-smi --query-gpu=name,uuid,driver_version,temperature.gpu,clocks.current.sm --format=csv
"$PYTHON" -m pytest -q -rA -m gpu -p no:cacheprovider -k "gpu_tuned or query_key" \
  --junitxml="$LOGS/gpu-tests-$SLURM_JOB_ID.xml" \
  scaffolds/transformer/tests/test_query_key_tiling.py \
  scaffolds/transformer/tests/test_qk_rms_attention_backends.py::test_full_context_qk_rms_backend
