#!/usr/bin/env bash
# GPU qualification of pure sigmoid query coefficients (diagonal and dense, with and without
# order scaling) on the 200M / 4B production composition. Submit from the worktree root:
#   sbatch --partition=debug --gres=gpu:b300:1 --cpus-per-task=8 --mem=96G --time=01:00:00 \
#     --output=scaffolds/transformer/validation/logs/sigmoid-coefficients/%x-%j.log \
#     scaffolds/transformer/validation/qualify_sigmoid_coefficients.sbatch "$(git rev-parse HEAD)"
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the worktree root}"
LAUNCH_COMMIT="${1:?Pass the committed source}"
git diff --exit-code "$LAUNCH_COMMIT" -- scaffolds/transformer ':!scaffolds/transformer/validation' experiments/transformer_pp/acceptance.py
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$PWD/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1
TASK_TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/sigmoid-coefficients-${SLURM_JOB_ID}.XXXXXX")"
trap 'rm -rf -- "$TASK_TMPDIR"' EXIT
export TMPDIR="$TASK_TMPDIR" XDG_CACHE_HOME="$PWD/.cache/sigmoid-coefficients/xdg"
CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
export PATH="$CUDA_HOME/bin:$PATH"
printf 'job=%s source=%s environment=%s\n' "$SLURM_JOB_ID" "$LAUNCH_COMMIT" "$TRANSFORMER_CUDA_ENVIRONMENT"
nvidia-smi --query-gpu=name,uuid,temperature.gpu,clocks.current.sm --format=csv
BACKEND=scaffolds/transformer/tests/test_qk_rms_attention_backends.py::test_full_context_qk_rms_backend
"$PYTHON" -m pytest -q -rA scaffolds/transformer/tests/test_sigmoid_coefficients.py "$BACKEND[query-4-mosaic]"
