#!/bin/bash
# Qualify head RMSNorm's max-abs prescale (model.attention_head_rms_prescale: max_abs) on the granted GPU: the GPU
# cases of tests/test_head_rms_prescale.py. They hold the device's own FP32 arithmetic to host FP64 at every output
# scale of the measured probe (1e-25 to 1e30, the plain evaluation's failing band of 9e14 to 1e18 included) and
# through the exponential's row shift; run the Mosaic kernels (fixed Taylor, static, query, and the softplus sum of
# squares with its input power) at the production context into the prescaled normalization against dense FP64, with
# the values at their own scale and scaled by 1e16; and take one training gradient of two-layer BF16 models on those
# kernels at Q/K widths 4 and 64, every head output above 1e15, against the same models in FP64. The existing Mosaic
# case with the unprescaled head RMSNorm (test_mosaic_unnormalized.py) runs beside them as the control.
# B300 does not admit the transformed kernels and its query kernel is not qualified at Q/K width 4: those cases skip
# there, which the final check allows, and no other skip.
# Submit from this checkout to the long partition, where the job waits for a free GPU and preempts nobody; outputs
# stay in ignored scaffold validation logs. Choose the GPU by node (CLUSTERS.md lists the failing ones):
# sbatch --partition=long --nodelist=dogo --output=scaffolds/transformer/validation/logs/head-rms-prescale-%j.log \
#   scaffolds/transformer/validation/qualify_head_rms_prescale.sbatch <source-commit>
# (dogo: H100; shiba: B200; akita: B300.) Then record it:
# python3 scaffolds/transformer/validation/record_head_rms_prescale_qualification.py <job-id> <record-name>
# With --gres=gpu:2 the job runs on the first granted GPU not listed in FAULTY_GPUS.
#SBATCH --job-name=qualify-head-rms-prescale
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the worktree}"
LAUNCH_COMMIT="${1:?Supply the frozen source commit}"
# The code under test is scaffold code: the check covers scaffolds/transformer outside validation/, scaffolds/records
# and scaffolds/launch, and no study ("none"), so runs allocated meanwhile do not stop a pending job.
python3 scaffolds/launch/frozen.py check --launch-commit "$LAUNCH_COMMIT" --experiment none --scaffold transformer
# GPUs that fail CUDA initialization (CLUSTERS.md): dogo's H100 at Slurm index 4 and mastiff's at indices 1 and 7.
FAULTY_GPUS="GPU-355f34a2-be0c-ba44-022b-1a056f643b80 GPU-86f64c73-c368-bcfb-49e5-6f24ab5eb7a5"
FAULTY_GPUS+=" GPU-cea82b77-b090-2e61-f425-c4b1608e8739"
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
TASK_TMP="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/head-rms-prescale-${SLURM_JOB_ID}.XXXXXX")"
trap 'rm -rf -- "$TASK_TMP"' EXIT
export TMPDIR="$TASK_TMP"
export JAX_COMPILATION_CACHE_DIR="$PWD/.cache/transformer_pp/jax"
OUTPUT="scaffolds/transformer/validation/logs/head-rms-prescale-${SLURM_JOB_ID}"
mkdir -p "$OUTPUT" "$JAX_COMPILATION_CACHE_DIR"
nvidia-smi --id="$SELECTED" --query-gpu=name,uuid,pci.bus_id,driver_version,compute_cap --format=csv
echo "Environment $TRANSFORMER_ENVIRONMENT_DIR, source $(git rev-parse HEAD), launch commit $LAUNCH_COMMIT"
TESTS=scaffolds/transformer/tests
"$PYTHON" -m pytest -c scaffolds/transformer/pyproject.toml --import-mode=importlib -m gpu -q -rs --durations=20 \
  -o junit_family=xunit1 "$TESTS/test_head_rms_prescale.py" \
  "$TESTS/test_mosaic_unnormalized.py::test_fixed_linear_full_context_and_head_rmsnorm_match_independent_fp64" \
  --junitxml="$OUTPUT/gpu-tests.xml"
"$PYTHON" - "$OUTPUT/gpu-tests.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET

ALLOWED_SKIP = "B300 does not admit the transformed kernels"
cases = list(ET.parse(sys.argv[1]).iter("testcase"))
failed = [case.get("name") for case in cases if case.find("failure") is not None or case.find("error") is not None]
skipped = [(case.get("name"), case.find("skipped").get("message", "")) for case in cases
           if case.find("skipped") is not None]
unexpected = [name for name, message in skipped if ALLOWED_SKIP not in message]
if not cases or failed or unexpected:
    raise SystemExit(f"Qualification failed: {len(cases)} cases, failed {failed}, unexpected skips {unexpected}")
print(f"All GPU qualification cases passed: {len(cases) - len(skipped)} passed, {len(skipped)} skipped as not "
      f"admitted on this GPU: {[name for name, _ in skipped]}")
PY
