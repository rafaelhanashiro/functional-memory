#!/usr/bin/env bash
# GPU qualification of the blocked backend's causal strips: kernel accuracy against FP64 and
# full-model micro-steps, scan versus strips. Submit from the worktree root, once per GPU type:
#   sbatch --partition=debug --gres=gpu:h100:1 --cpus-per-task=8 --mem=96G --time=01:00:00 \
#     --output=scaffolds/transformer/validation/logs/blocked-strips/%x-%j.log \
#     scaffolds/transformer/validation/qualify_blocked_strips.sbatch "$(git rev-parse HEAD)"
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the worktree root}"
LAUNCH_COMMIT="${1:?Pass the committed source}"
git diff --exit-code "$LAUNCH_COMMIT" -- scaffolds/transformer ':!scaffolds/transformer/validation'
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$PWD/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
TASK_TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/blocked-strips-${SLURM_JOB_ID}.XXXXXX")"
trap 'rm -rf -- "$TASK_TMPDIR"' EXIT
export TMPDIR="$TASK_TMPDIR" XDG_CACHE_HOME="$PWD/.cache/blocked-strips/xdg"
CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
export PATH="$CUDA_HOME/bin:$PATH"
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')"
OUT=scaffolds/transformer/validation/runs/blocked-strips
mkdir -p "$OUT"
printf 'job=%s source=%s environment=%s gpu=%s\n' "$SLURM_JOB_ID" "$LAUNCH_COMMIT" "$TRANSFORMER_CUDA_ENVIRONMENT" "$GPU"
nvidia-smi --query-gpu=name,uuid,temperature.gpu,clocks.current.sm --format=csv
# STAGES (default "tests kernel model") and RECIPES select what runs.
STAGES="${STAGES:-tests kernel model}"
RECIPES="${RECIPES:-experiments/transformer_pp/runs/r5adc29a91e3b/config.yaml experiments/transformer_pp/runs/r82a80d3091e2/config.yaml}"
if [[ " $STAGES " == *" tests "* ]]; then
  "$PYTHON" -m pytest -q -rA scaffolds/transformer/tests/test_blocked_strips_h100.py
  "$PYTHON" -m pytest -q -rA -m gpu scaffolds/transformer/tests/test_spline_attention.py
fi
if [[ " $STAGES " == *" kernel "* ]]; then
  "$PYTHON" -m scaffolds.transformer.validation.blocked_strips_qualification kernel \
    --output "$OUT/kernel-$GPU-$SLURM_JOB_ID.json"
fi
if [[ " $STAGES " == *" model "* ]]; then
  # shellcheck disable=SC2086
  "$PYTHON" -m scaffolds.transformer.validation.blocked_strips_qualification model --micro-batch 16 \
    --recipes $RECIPES --implementations ${IMPLEMENTATIONS:-scan strips} --output "$OUT/model-$GPU-$SLURM_JOB_ID.json"
fi
