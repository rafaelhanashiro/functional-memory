#!/bin/bash

#SBATCH --job-name=ofm_materialize
#SBATCH --partition=mit_preemptable
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

PROJECT_DIR="$(realpath "${SLURM_SUBMIT_DIR:-$PWD}")"
PROJECT_ROOT="/home/rafah/projects/ofm"
SCRIPT_DIR="${PROJECT_DIR}/scripts/materialize_fineweb"
CONFIG_PATH="${1:?Usage: sbatch scripts/materialize_fineweb/materialize_fineweb.sbatch <config.yaml>}"

cd "${PROJECT_DIR}"

if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${PROJECT_DIR}/${CONFIG_PATH}"
fi

CACHE_DIR="${PROJECT_ROOT}/.caches"
ENV_DIR="${PROJECT_ROOT}/.envs"
JOB_TMPROOT="${CACHE_DIR}/tmp/materialize-${SLURM_JOB_ID}"
JOB_TMPDIR="${JOB_TMPROOT}/scratch"
mkdir -p \
  "${CACHE_DIR}/uv" \
  "${CACHE_DIR}/hf/datasets" \
  "${CACHE_DIR}/hf/transformers" \
  "${JOB_TMPROOT}" \
  "${JOB_TMPDIR}" \
  "${ENV_DIR}/cuda"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export TOKENIZERS_PARALLELISM=true
export UV_CACHE_DIR="${CACHE_DIR}/uv"
export UV_PROJECT_ENVIRONMENT="${ENV_DIR}/cuda"
export UV_LINK_MODE=copy
export HF_HOME="${CACHE_DIR}/hf"
export HF_DATASETS_CACHE="${CACHE_DIR}/hf/datasets"
export TRANSFORMERS_CACHE="${CACHE_DIR}/hf/transformers"
export XDG_CACHE_HOME="${JOB_TMPROOT}/xdg"
export TMPDIR="${JOB_TMPDIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

set +e
uv run --python 3.11 --extra cuda python \
  "${SCRIPT_DIR}/materialize_fineweb.py" "${CONFIG_PATH}" \
  >"${JOB_TMPROOT}/materialize.out" \
  2>"${JOB_TMPROOT}/materialize.err"
status=$?
set -e
if ((status == 0)) && [[ -d "${JOB_TMPROOT}" ]]; then
  find "${JOB_TMPROOT}" -depth -delete || true
fi
exit "${status}"
