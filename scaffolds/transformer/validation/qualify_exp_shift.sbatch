#!/bin/bash
# Qualify the fused unnormalized exponential shifted by its causal row maximum (model.attention_exp_shift: row_max)
# on the granted GPU: the whole fused unnormalized module (test_unnormalized_fused_attention.py), whose test_shifted_*
# cases take Q/K widths 4, 8, 16 and 64 against the blocked backend and dense FP64, at ordinary scores and at row
# maxima beyond FP32's e^88.7, with head RMSNorm's offset and through a two-layer model; its unshifted cases check
# the historical kernels beside them. The GPU case of test_exp_row_max_shift.py qualifies the blocked backend's shifted
# strips and scan at production precision. After the tests, benchmarks/benchmark_exp_shift.py times both kernels.
# Submit from this checkout; outputs stay in ignored scaffold validation logs:
# sbatch --partition=standard,long --nodelist=akita --output=scaffolds/transformer/validation/logs/exp-shift-%j.log \
#   scaffolds/transformer/validation/qualify_exp_shift.sbatch <source-commit>
# With --gres=gpu:2 the job runs on the first granted GPU not listed in FAULTY_GPUS.
#SBATCH --job-name=qualify-exp-shift
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the worktree}"
LAUNCH_COMMIT="${1:?Supply the frozen source commit}"
python3 scaffolds/launch/frozen.py check --launch-commit "$LAUNCH_COMMIT" --experiment transformer_pp --scaffold transformer
# GPUs that fail CUDA initialization (CLUSTERS.md): dogo's H100 at Slurm index 4 and mastiff's at index 1.
FAULTY_GPUS="GPU-355f34a2-be0c-ba44-022b-1a056f643b80 GPU-86f64c73-c368-bcfb-49e5-6f24ab5eb7a5"
mapfile -t GRANTED < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
SELECTED=""
for uuid in "${GRANTED[@]}"; do
  case " $FAULTY_GPUS " in *" $uuid "*) echo "Skipping faulty GPU $uuid" ;; *) SELECTED="$uuid"; break ;; esac
done
[[ -n "$SELECTED" ]] || { echo "Every granted GPU is a known faulty one: ${GRANTED[*]}" >&2; exit 3; }
export CUDA_VISIBLE_DEVICES="$SELECTED"
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$PWD/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda,cpu PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
TASK_TMP="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/exp-shift-${SLURM_JOB_ID}.XXXXXX")"
trap 'rm -rf -- "$TASK_TMP"' EXIT
export TMPDIR="$TASK_TMP"
export JAX_COMPILATION_CACHE_DIR="$PWD/.cache/transformer_pp/jax"
OUTPUT="scaffolds/transformer/validation/logs/exp-shift-${SLURM_JOB_ID}"
mkdir -p "$OUTPUT" "$JAX_COMPILATION_CACHE_DIR"
nvidia-smi --id="$SELECTED" --query-gpu=name,uuid,pci.bus_id,driver_version,compute_cap --format=csv
echo "Environment $TRANSFORMER_ENVIRONMENT_DIR, source $(git rev-parse HEAD), launch commit $LAUNCH_COMMIT"
"$PYTHON" -m pytest -c scaffolds/transformer/pyproject.toml --import-mode=importlib -m gpu -q -rs --durations=15 \
  scaffolds/transformer/tests/test_unnormalized_fused_attention.py scaffolds/transformer/tests/test_exp_row_max_shift.py \
  --junitxml="$OUTPUT/gpu-tests.xml"
"$PYTHON" - "$OUTPUT/gpu-tests.xml" <<'PY'
import sys
from scaffolds.transformer.acceptance import require_passing_tests
print('All GPU qualification cases passed:', require_passing_tests(sys.argv[1]))
PY
# Kernel time, unshifted and shifted, at the reference shape's attention (micro-batch 16, 14 heads, 2,048 tokens).
"$PYTHON" -m scaffolds.transformer.benchmarks.benchmark_exp_shift --output "$OUTPUT/throughput.json"
