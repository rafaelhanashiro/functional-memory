#!/bin/bash

#SBATCH --job-name=ofm_fineweb_12b
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
OUTPUT_DIR="${PROJECT_DIR}/runs/fineweb_12b/data"
CACHE_DIR="${PROJECT_ROOT}/.caches"
ENV_DIR="${PROJECT_ROOT}/.envs"
JOB_TMPDIR="${CACHE_DIR}/tmp/fineweb-12b-${SLURM_JOB_ID}"

cd "${PROJECT_DIR}"
mkdir -p \
  "${OUTPUT_DIR}/logs" \
  "${CACHE_DIR}/uv" \
  "${CACHE_DIR}/hf/datasets" \
  "${CACHE_DIR}/hf/transformers" \
  "${CACHE_DIR}/python" \
  "${JOB_TMPDIR}" \
  "${ENV_DIR}/cuda"

exec >"${OUTPUT_DIR}/logs/materialize-${SLURM_JOB_ID}.out" \
  2>"${OUTPUT_DIR}/logs/materialize-${SLURM_JOB_ID}.err"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export TOKENIZERS_PARALLELISM=true
export UV_CACHE_DIR="${CACHE_DIR}/uv"
export UV_PROJECT_ENVIRONMENT="${ENV_DIR}/cuda"
export UV_LINK_MODE=copy
export HF_HOME="${CACHE_DIR}/hf"
export HF_DATASETS_CACHE="${CACHE_DIR}/hf/datasets"
export TRANSFORMERS_CACHE="${CACHE_DIR}/hf/transformers"
export XDG_CACHE_HOME="${CACHE_DIR}/python"
export TMPDIR="${JOB_TMPDIR}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

uv run --python 3.11 --extra cuda python \
  scripts/materialize_fineweb/materialize_fineweb.py \
  "${OUTPUT_DIR}" \
  --dataset-name HuggingFaceFW/fineweb-edu \
  --dataset-config sample-100BT \
  --dataset-split train \
  --dataset-revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
  --tokenizer-name NousResearch/Llama-2-7b-hf \
  --tokenizer-revision 8efe6c9b93655b934e27bd9981e3ec13e55aee9d \
  --sequence-length 2049 \
  --train-sequences 5859375 \
  --eval-sequences 4888 \
  --shard-sequences 65536 \
  --tokenizer-batch-size 64 \
  --validation-fraction 0.01 \
  --shuffle-seed 0 \
  --shuffle-buffer-size 10000 \
  --max-epochs 1
