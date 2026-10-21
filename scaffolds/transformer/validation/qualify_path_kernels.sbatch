#!/bin/bash
# Qualify the fused PaTH kernels and the GDN bridge port on one GPU: the GPU tests, the
# per-layer benchmark, and the GDN launch comparison; on B300 also FLA's own PyTorch kernels.
# Arguments: committed source, GPU type (H100 or B300), and the step: tests or benchmark.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
LAUNCH_COMMIT="${1:?Expected committed source}"
GPU_TYPE="${2:?Expected H100 or B300}"
STEP="${3:?Expected tests or benchmark}"
GDN_REFERENCE_COMMIT=7fb44dc
python3 scaffolds/launch/frozen.py check --launch-commit "$LAUNCH_COMMIT" --experiment attention_scaling --scaffold transformer
source scaffolds/transformer/launchers/kennel_runtime.sh
PYTHON="$PWD/scaffolds/transformer/$TRANSFORMER_ENVIRONMENT_DIR/bin/python"
CUDA_HOME="$(transformer_cuda_home "$PYTHON")"
export CUDA_HOME TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
# The custom-call bridge executes outside XLA command buffers.
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_enable_command_buffer="
CACHE="$PWD/.cache/path-kernels-qualification"
export EJKERNEL_TRITON_CACHE_DIR="$CACHE/triton" JAX_COMPILATION_CACHE_DIR="$CACHE/jax"
export TMPDIR="${SLURM_TMPDIR:-/tmp}/path-kernels-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
DIRECTORY="scaffolds/transformer/validation/runs/path-kernels/${GPU_TYPE,,}"
LOGS="scaffolds/transformer/validation/logs/path-kernels/$SLURM_JOB_ID"
mkdir -p "$DIRECTORY" "$LOGS"
nvidia-smi --query-gpu=name,driver_version,pci.bus_id --format=csv,noheader
status=0
if [[ "$STEP" == tests ]]; then
  DESELECT=()
  if [[ "$GPU_TYPE" == B300 ]]; then
    # FLA's GDN kernel crashes at width 128 on B300 and poisons the process (CLUSTERS.md);
    # GDN at width 128 or more is not qualified there.
    GDN=scaffolds/transformer/tests/test_gated_deltanet_kernel.py
    for precision in float32-0.005 bfloat16-0.02; do
      for width in 128 256; do
        DESELECT+=(--deselect "$GDN::test_fused_output_and_all_gradients_match_the_recurrence[$precision-$width]"
                   --deselect "$GDN::test_independent_fused_output_and_all_gradients_match_recurrence[$precision-$width-256]")
      done
    done
  fi
  for suite in path:test_path_attention_fused.py gdn:"test_gated_deltanet_kernel.py test_gdn_tanh_write_kernel.py"; do
    name="${suite%%:*}"
    files=()
    for file in ${suite#*:}; do files+=("scaffolds/transformer/tests/$file"); done
    extra=()
    if [[ "$name" == gdn ]]; then extra=("${DESELECT[@]}"); fi
    "$PYTHON" -m pytest -c scaffolds/transformer/pyproject.toml -q --import-mode=importlib -m gpu "${files[@]}" \
      "${extra[@]}" --junitxml="$LOGS/$name-tests.xml" || status=$?
  done
  "$PYTHON" -m scaffolds.transformer.validation.compare_gdn_port --commit "$GDN_REFERENCE_COMMIT" \
    --output "$DIRECTORY/gdn-port.json" || status=$?
else
  UPSTREAM=()
  if [[ "$GPU_TYPE" == B300 ]]; then
    UPSTREAM_PYTHON="$PWD/scaffolds/transformer/benchmarks/upstream_runtime/.venv/bin/python"
    "$UPSTREAM_PYTHON" scaffolds/transformer/benchmarks/benchmark_upstream_path.py \
      --output "$DIRECTORY/upstream-fla.json" --parity "$TMPDIR/parity.npz" || status=$?
    UPSTREAM=(--upstream "$TMPDIR/parity.npz")
  fi
  "$PYTHON" -m scaffolds.transformer.benchmarks.benchmark_path --output "$DIRECTORY/benchmark.json" \
    "${UPSTREAM[@]}" || status=$?
fi
"$PYTHON" - "$DIRECTORY" "$LOGS" "$LAUNCH_COMMIT" "$GPU_TYPE" "$STEP" "$status" <<'PY'
from datetime import UTC, datetime
from pathlib import Path
import hashlib, json, os, sys
import xml.etree.ElementTree as ET
import jax
directory, logs, commit, gpu, step, status = sys.argv[1:]
devices = [device.device_kind for device in jax.local_devices()]
suites = {}
for junit in sorted(Path(logs).glob("*-tests.xml")):
    cases = list(ET.parse(junit).iter("testcase"))
    suites[junit.stem] = {
        "cases": len(cases),
        "failed": sum(any(case.find(tag) is not None for tag in ("failure", "error")) for case in cases),
        "skipped": sum(case.find("skipped") is not None for case in cases),
        "seconds": round(sum(float(case.get("time", 0)) for case in cases), 1),
        "junit": {"path": str(junit), "sha256": hashlib.sha256(junit.read_bytes()).hexdigest()}}
passed = int(status) == 0 and all(s["failed"] == 0 and s["skipped"] == 0 for s in suites.values())
value = {
    "source_commit": commit, "completed_utc": datetime.now(UTC).isoformat(), "home_cluster": "kennel",
    "slurm_job_id": os.environ["SLURM_JOB_ID"], "node": os.environ["SLURMD_NODENAME"], "gpu_type": gpu,
    "device_kinds": devices, "step": step, "status": "passed" if passed else "failed", "exit_code": int(status),
    "cuda_environment": os.environ.get("TRANSFORMER_CUDA_ENVIRONMENT"),
    "dependency_lock_sha256": hashlib.sha256(Path("scaffolds/transformer/uv.lock").read_bytes()).hexdigest(),
    "runtime_environment": {key: os.environ.get(key) for key in (
        "XLA_FLAGS", "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE", "CUDA_HOME")},
    **({"suites": suites} if suites else {}),
}
assert len(devices) == 1 and gpu in devices[0].upper(), devices
Path(directory, f"{step}-job.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
print(json.dumps(value), flush=True)
raise SystemExit(0 if passed else 1)
PY
