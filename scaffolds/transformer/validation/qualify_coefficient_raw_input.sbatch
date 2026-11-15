#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
EXECUTION_ROOT="${1:?Expected checkout}"
LAUNCH_COMMIT="${2:?Expected source commit}"
cd "$EXECUTION_ROOT"
python3 scaffolds/launch/frozen.py check --launch-commit "$LAUNCH_COMMIT" --experiment transformer_pp --scaffold transformer
export PYTHONPATH="$EXECUTION_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$EXECUTION_ROOT/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
QUALIFICATION_TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/coefficient-raw-input-${SLURM_JOB_ID}.XXXXXX")"
export TMPDIR="$QUALIFICATION_TMPDIR"
trap 'rm -rf -- "$QUALIFICATION_TMPDIR"' EXIT
mkdir -p scaffolds/transformer/validation/logs
"$PYTHON" -m pytest -c scaffolds/transformer/pyproject.toml --import-mode=importlib -q -m gpu \
  scaffolds/transformer/tests/test_coefficient_raw_input.py \
  --junitxml=scaffolds/transformer/validation/logs/coefficient_raw_input_gpu.xml
