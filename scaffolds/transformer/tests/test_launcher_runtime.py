"""Check cluster policy without reserving hardware or importing JAX."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "launchers" / "kennel_runtime.sh"


@pytest.mark.parametrize("gpu,override,staging,nccl,fraction", [
    ("NVIDIA A100-PCIE-40GB", None, "1", "1", None),
    ("NVIDIA RTX A6000", None, "1", "1", "0.90"),
    ("NVIDIA H100 PCIe", None, "0", None, None),
    ("NVIDIA B300", None, "0", None, None),
    ("NVIDIA A100-PCIE-40GB", "0", "0", None, None),
    ("NVIDIA H100 PCIe", "1", "1", "1", None),
])
def test_kennel_transport_policy(tmp_path, gpu, override, staging, nccl, fraction):
    executable = tmp_path / "nvidia-smi"
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$TEST_GPU_NAME"\n')
    executable.chmod(0o755)
    env = {key: value for key, value in os.environ.items()
           if key not in ("TRANSFORMER_HOST_STAGING", "NCCL_P2P_DISABLE", "XLA_PYTHON_CLIENT_MEM_FRACTION",
                          "XLA_FLAGS")}
    env.update(PATH=str(tmp_path) + os.pathsep + env["PATH"], TEST_GPU_NAME=gpu)
    if override is not None:
        env["TRANSFORMER_HOST_STAGING"] = override
    result = subprocess.run([
        'bash', '-euc', 'source "$1"; "$2" -c "$3"', 'test', str(SCRIPT), sys.executable,
        'import json,os; print(json.dumps(dict(os.environ)))',
    ], env=env, check=True, text=True, capture_output=True)
    resolved = json.loads(result.stdout)
    assert resolved["TRANSFORMER_HOST_STAGING"] == staging
    assert resolved.get("NCCL_P2P_DISABLE") == nccl
    assert resolved.get("XLA_PYTHON_CLIENT_MEM_FRACTION") == fraction
    assert resolved["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


@pytest.mark.parametrize("gpu,existing,expected", [
    ("NVIDIA B300 SXM6 AC", None, "--xla_gpu_disable_async_collectives=ALLCOLLECTIVES"),
    ("NVIDIA B300 SXM6 AC", "--xla_gpu_enable_command_buffer=",
     "--xla_gpu_enable_command_buffer= --xla_gpu_disable_async_collectives=ALLCOLLECTIVES"),
    ("NVIDIA B300 SXM6 AC", "--xla_gpu_disable_async_collectives=ALLREDUCE",
     "--xla_gpu_disable_async_collectives=ALLREDUCE"),
    ("NVIDIA H100 80GB HBM3", None, None),
])
def test_b300_collectives_stay_on_the_compute_stream(tmp_path, gpu, existing, expected):
    executable = tmp_path / "nvidia-smi"
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$TEST_GPU_NAME"\n')
    executable.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if key != "XLA_FLAGS"}
    env.update(PATH=str(tmp_path) + os.pathsep + env["PATH"], TEST_GPU_NAME=gpu)
    if existing is not None:
        env["XLA_FLAGS"] = existing
    result = subprocess.run([
        'bash', '-euc', 'source "$1"; "$2" -c "$3"', 'test', str(SCRIPT), sys.executable,
        'import json,os; print(json.dumps(os.environ.get("XLA_FLAGS")))',
    ], env=env, check=True, text=True, capture_output=True)
    assert json.loads(result.stdout) == expected
