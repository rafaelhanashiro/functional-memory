#!/bin/bash
#SBATCH --job-name=exp-attn-env
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
EXECUTION_ROOT="${1:?Pass the experiment checkout}"
PROJECT_ROOT="${2:?Pass the storage project root}"
export UV_PROJECT_ENVIRONMENT="${PROJECT_ROOT}/.envs/unnormalized_attention"
export UV_CACHE_DIR="${PROJECT_ROOT}/.cache/uv"
export UV_LINK_MODE=copy
cd "$EXECUTION_ROOT"
uv sync --project experiments/unnormalized_attention --locked --extra cuda
