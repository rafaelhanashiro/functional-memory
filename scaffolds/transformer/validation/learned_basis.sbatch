#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
python3 -m scaffolds.launch.frozen check --launch-commit "${SHARED_TRANSFORMER_SOURCE_COMMIT:?}" \
  --experiment attention_scaling --scaffold transformer
unset LD_LIBRARY_PATH
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$PWD/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
export PATH="$CUDA_HOME/bin:$PATH" PYTHONPATH="$PWD"
export EJKERNEL_TRITON_CACHE_COMPILES=1 EJKERNEL_TRITON_CACHE_DIR="$PWD/.cache/learned-basis"
unset JAX_PLATFORMS
TMPDIR="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/learned-basis-${SLURM_JOB_ID}.XXXXXX")"
export TMPDIR
trap 'rm -rf -- "$TMPDIR"' EXIT
OUT="scaffolds/transformer/validation/runs/learned-basis/$SLURM_JOB_ID"
mkdir -p "$OUT"
nvidia-smi --query-gpu=index,uuid,name,driver_version --format=csv > "$OUT/gpus.csv"
scontrol show job "$SLURM_JOB_ID" > "$OUT/allocation.txt"
"$PYTHON" -m pytest -x -q -m gpu scaffolds/transformer/tests/test_learned_basis.py \
  scaffolds/transformer/tests/test_triton_scalar.py --junitxml="$OUT/tests.xml" > "$OUT/tests.log" 2>&1
"$PYTHON" - "$OUT/tests.xml" <<'PY'
import sys, xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
assert root.findall('.//testcase'), 'No GPU tests executed'
assert not root.findall('.//skipped'), 'GPU qualification must not skip tests'
PY
