#!/bin/bash
# Qualify a run's kernels on an ORCD GPU before production: its adapter's acceptance (CPU and GPU tests,
# initialization, pilot and resume) on the allocated GPU, in orcd.sbatch's environment, without training. The
# acceptance output stays in the ignored scaffold validation logs; passed test stages go to the experiment's test
# cache, which production acceptance at the same commit, environment, runtime and GPU model reuses. Submit from the
# experiment worktree root once the run is committed (the frozen execution set must match the launch commit):
# run=<run>; sbatch --job-name="qualify-$run" --gres=gpu:h100:1 \
#   --exclude=node3009,node3209,node2900,node1708,node1709,node4508,node2801,node3409 \
#   --output="scaffolds/transformer/validation/logs/acceptance-$run-%j.log" \
#   scaffolds/transformer/validation/qualify_acceptance_orcd.sbatch experiments.<study>.<adapter> \
#   "experiments/<study>/runs/$run/config.yaml" "$(git rev-parse HEAD)" H100
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=03:00:00
set -euo pipefail
ADAPTER="${1:?Expected the experiment acceptance module}"
CONFIG="${2:?Expected experiments/<study>/runs/<run>/config.yaml}"
LAUNCH_COMMIT="${3:?Expected the launch commit}"
GPU_TYPE="${4:-H100}"
cd "${SLURM_SUBMIT_DIR:?Submit from the experiment worktree root}"
if [[ ! "$CONFIG" =~ ^experiments/([A-Za-z0-9_]+)/runs/(r[0-9a-f]{12})/config\.yaml$ ]]; then
  echo "Expected a repository-relative recipe experiments/<study>/runs/<run>/config.yaml: $CONFIG" >&2
  exit 1
fi
EXPERIMENT="${BASH_REMATCH[1]}"
RUN="${BASH_REMATCH[2]}"
PYTHON="$PWD/scaffolds/transformer/.venv/bin/python"
"$PYTHON" scaffolds/launch/frozen.py check --launch-commit "$LAUNCH_COMMIT" --experiment "$EXPERIMENT" --scaffold transformer
GPU_NAMES="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sort -u)"
echo "run=$RUN job=$SLURM_JOB_ID node=$(hostname) launch_commit=$LAUNCH_COMMIT gpu=$GPU_NAMES"
if [[ "$GPU_NAMES" != *"$GPU_TYPE"* ]]; then
  echo "Allocated $GPU_NAMES; this qualification is for $GPU_TYPE" >&2
  exit 1
fi
CACHE="$PWD/.cache/$EXPERIMENT"
export SHARED_TRANSFORMER_SOURCE_COMMIT="$LAUNCH_COMMIT" TRANSFORMER_CLUSTER=orcd TRANSFORMER_GPU_TYPE="$GPU_TYPE"
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda TRANSFORMER_HOST_STAGING="${TRANSFORMER_HOST_STAGING:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export JAX_COMPILATION_CACHE_DIR="$CACHE/jax" XDG_CACHE_HOME="$CACHE/xdg" TMPDIR="$CACHE/tmp/$SLURM_JOB_ID"
# orcd.sbatch keeps TMPDIR inside the worktree, which puts pytest's tmp_path studies inside the checkout: the records
# store such a study's paths relative to the checkout (.cache/...) and cannot resolve them again, so
# test_gradient_events.py fails there. Kennel's launcher keeps TMPDIR node-local; give pytest a node-local base too.
# Neither location is part of a test stage's identity, so production at the same commit reuses these reports.
PYTEST_BASETEMP="${SLURM_TMPDIR:-/tmp}/functional-memory-pytest-$SLURM_JOB_ID"
export PYTEST_ADDOPTS="--basetemp=$PYTEST_BASETEMP"
trap 'rm -rf -- "$PYTEST_BASETEMP"' EXIT
CUDA_HOME="$("$PYTHON" -c 'import importlib.util; print(next(iter(importlib.util.find_spec("nvidia.cuda_nvcc").submodule_search_locations)))')"
export CUDA_HOME
eval "$("$PYTHON" -m "$ADAPTER" "$CONFIG" --runtime-env)"
mkdir -p "$JAX_COMPILATION_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"
"$PYTHON" -m "$ADAPTER" "$CONFIG" --directory "scaffolds/transformer/validation/logs/acceptance-$RUN-$SLURM_JOB_ID" \
  --gpu-count 1 --gpu-type "$GPU_TYPE" --test-cache "$CACHE/acceptance-tests"
