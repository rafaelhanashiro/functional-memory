#!/bin/bash
# Qualify the Mosaic score-function families (prefactor and swish), Q/K width 4, the static learnable-constant
# broadcast, and the fused exponential at narrow keys, on the granted GPU, one suite per job.
# Submit from this checkout; outputs stay in ignored scaffold validation logs. Choose the node with
# --nodelist or --exclude, for example:
# sbatch --nodelist=shiba --output=scaffolds/transformer/validation/logs/score-prefactor-%j.log \
#   scaffolds/transformer/validation/qualify_score_prefactor.sbatch <source-commit> <suite>
# Suites, each well inside 30 minutes on one GPU: prefactor1 and prefactor2 (test_mosaic_score_prefactor.py: the
# FP64 grid and narrow keys; then the production context, input power, prefix, overflow, extremes and
# repeatability), transforms1 and transforms2 (test_mosaic_score_transform.py: its FP64 grid; then the rest), sos1
# (test_mosaic_softplus_sos.py), sos2 (test_mosaic_softplus_zero_constant.py), pointwise (the pointwise GPU suites,
# the strips' composition, and the query and Taylor kernels' width-4 cases), fused (the fused exponential at Q/K
# widths 4, 8 and 16, and at 64); several suites join with commas, e.g. pointwise,fused. With --gres=gpu:2 the job
# runs on the first granted GPU not listed in FAULTY_GPUS.
#SBATCH --job-name=qualify-score-prefactor
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
SUITES="${2:?Supply the suites (prefactor1, prefactor2, transforms1, transforms2, sos1, sos2, pointwise, fused), joined with commas}"
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
TASK_TMP="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/score-prefactor-${SLURM_JOB_ID}.XXXXXX")"
trap 'rm -rf -- "$TASK_TMP"' EXIT
export TMPDIR="$TASK_TMP"
export JAX_COMPILATION_CACHE_DIR="$PWD/.cache/transformer_pp/jax"
OUTPUT="scaffolds/transformer/validation/logs/score-prefactor-${SUITES//,/-}-${SLURM_JOB_ID}"
mkdir -p "$OUTPUT" "$JAX_COMPILATION_CACHE_DIR"
nvidia-smi --id="$SELECTED" --query-gpu=name,uuid,pci.bus_id,driver_version,compute_cap --format=csv
echo "Environment $TRANSFORMER_ENVIRONMENT_DIR, source $(git rev-parse HEAD), launch commit $LAUNCH_COMMIT"
TESTS=scaffolds/transformer/tests
SELECTION=()
IFS=, read -ra SUITE_LIST <<< "$SUITES"
for SUITE in "${SUITE_LIST[@]}"; do
  case "$SUITE" in
    prefactor1) SELECTION+=("$TESTS/test_mosaic_score_prefactor.py::test_outputs_and_vjp_match_fp64"
                            "$TESTS/test_mosaic_score_prefactor.py::test_narrow_key_widths_match_fp64") ;;
    prefactor2) for test in test_production_context_matches_fp64_and_the_strips test_softplus_prefactor_matches_the_input_power_kernel \
                            test_outputs_and_vjp_ignore_future_tokens test_an_overflowing_score_poisons_its_row_as_in_the_strips \
                            test_extreme_and_zero_scores test_repeated_calls_are_bitwise_identical; do
                  SELECTION+=("$TESTS/test_mosaic_score_prefactor.py::$test")
                done ;;
    transforms1) SELECTION+=("$TESTS/test_mosaic_score_transform.py::test_outputs_and_vjp_match_fp64") ;;
    transforms2) for test in test_every_key_width_matches_fp64 test_production_context_matches_fp64_and_the_strips \
                             test_largest_degree_fits_the_production_context test_outputs_and_vjp_ignore_future_tokens \
                             test_future_nonfinite_keys_and_coefficients_do_not_poison_the_prefix \
                             test_nonfinite_weight_poisons_its_row_even_with_zero_values \
                             test_an_overflowing_score_poisons_its_row_as_in_the_strips test_extreme_and_zero_scores \
                             test_repeated_calls_are_bitwise_identical \
                             test_model_dispatch_uses_the_kernel_and_reports_transform_diagnostics \
                             test_softplus_degree_nine_at_narrow_key_widths; do
                   SELECTION+=("$TESTS/test_mosaic_score_transform.py::$test")
                 done ;;
    sos1) SELECTION+=("$TESTS/test_mosaic_softplus_sos.py") ;;
    sos2) SELECTION+=("$TESTS/test_mosaic_softplus_zero_constant.py") ;;
    pointwise)
      # Every width-4 case of the query and Taylor kernels' width grids, then the pointwise and composition suites.
      mapfile -t WIDTH4 < <("$PYTHON" -m pytest -c scaffolds/transformer/pyproject.toml --import-mode=importlib -m gpu \
        -q --collect-only "$TESTS/test_mosaic_query.py" "$TESTS/test_mosaic_taylor.py" \
        | grep -E '::test_outputs_and_vjp_match_fp64_and_triton\[[0-9]+-4-bf16(x3)?\]$' \
        | sed -E 's#^(scaffolds/transformer/)?tests/#scaffolds/transformer/tests/#')
      (( ${#WIDTH4[@]} > 0 )) || { echo "No width-4 cases collected" >&2; exit 1; }
      SELECTION+=("${WIDTH4[@]}" "$TESTS/test_pointwise_score_transform.py" "$TESTS/test_pointwise_attention.py"
                  "$TESTS/test_blocked_composition.py") ;;
    fused) SELECTION+=("$TESTS/test_unnormalized_fused_attention.py::test_bf16_narrow_keys_at_the_production_context_match_blocked_and_fp64"
                       "$TESTS/test_unnormalized_fused_attention.py::test_bf16_production_context_outputs_and_vjp_match_blocked") ;;
    *) echo "Unknown suite: $SUITE" >&2; exit 2 ;;
  esac
done
printf 'Selection (%d):\n' "${#SELECTION[@]}"; printf '  %s\n' "${SELECTION[@]}"
"$PYTHON" -m pytest -c scaffolds/transformer/pyproject.toml --import-mode=importlib -m gpu -q -rs --durations=15 \
  "${SELECTION[@]}" --junitxml="$OUTPUT/gpu-tests.xml"
"$PYTHON" - "$OUTPUT/gpu-tests.xml" <<'PY'
import sys
from scaffolds.transformer.acceptance import require_passing_tests
print('All GPU qualification cases passed:', require_passing_tests(sys.argv[1]))
PY
