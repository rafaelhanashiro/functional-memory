"""Qualify the full native FLA recipe and an exact fresh-process training resume."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch

from .checkpointing import checkpoint_metadata
from .config import load_config, parameter_count


ROOT = Path(__file__).resolve().parents[3]
RUNTIME_KEYS = (
    "TRANSFORMER_CUDA_ENVIRONMENT", "CUDA_HOME", "NCCL_P2P_DISABLE",
    "CUBLAS_WORKSPACE_CONFIG", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    "PYTORCH_CUDA_ALLOC_CONF", "FLA_CONV_BACKEND",
    "TRITON_CACHE_DIR", "FLA_CACHE_RESULTS",
)
PILOT_STEPS = 3
SPLIT_STEP = 2


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def acceptance_identity(config_path: Path) -> dict:
    from scaffolds.launch.frozen import frozen_hash

    config = load_config(config_path)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Native FLA acceptance requires the production allocation of one CUDA GPU")
    source = (os.environ.get("NATIVE_FLA_SOURCE_COMMIT") or os.environ.get("SHARED_TRANSFORMER_SOURCE_COMMIT")
              or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    return {
        "launch_commit": source,
        "config_sha256": _sha256(config_path),
        "dependency_lock_sha256": _sha256(Path(__file__).with_name("uv.lock")),
        "frozen_execution_sha256": frozen_hash(ROOT, source, config_path.parents[2].name, "transformer"),
        "data_manifests": {name: {"path": config["data"][name + "_manifest"],
                                   "sha256": _sha256(Path(config["data"][name + "_manifest"]))}
                           for name in ("train", "eval")},
        "reference": config["reference"],
        "gpu_count": torch.cuda.device_count(), "gpu_type": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "runtime": {name: os.environ.get(name) for name in RUNTIME_KEYS},
        "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
        "pilot_steps": PILOT_STEPS, "split_step": SPLIT_STEP,
        "resume_comparison": "exact",
    }


def compare_checkpoint_states(reference: Path, resumed: Path) -> dict:
    """Compare actual states, not serialized-file bytes or validation loss alone.

    A mismatch is a failed qualification. There is deliberately no automatic
    tolerance fallback for upstream kernels that turn out to be nondeterministic.
    """
    for directory in (reference, resumed):
        checkpoint_metadata(directory)
    left = torch.load(reference / "state.pt", map_location="cpu", weights_only=False)
    right = torch.load(resumed / "state.pt", map_location="cpu", weights_only=False)
    differences = []
    tensor_count = 0
    scalar_count = 0

    def compare(a, b, path):
        nonlocal tensor_count, scalar_count
        if isinstance(a, torch.Tensor):
            tensor_count += 1
            if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
                differences.append({"path": path, "reason": "tensor type/shape/dtype differs"})
            elif not torch.equal(a, b):
                details = {"path": path, "reason": "tensor values differ"}
                if a.numel() and a.is_floating_point():
                    delta = (a - b).float()
                    details["max_absolute_error"] = float(delta.abs().max())
                    denominator = float(torch.linalg.vector_norm(a.float()))
                    details["relative_l2_error"] = float(torch.linalg.vector_norm(delta)) / max(denominator, 1e-30)
                differences.append(details)
        elif isinstance(a, np.ndarray):
            if not isinstance(b, np.ndarray) or a.dtype != b.dtype or not np.array_equal(a, b):
                differences.append({"path": path, "reason": "NumPy RNG state differs"})
        elif isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                differences.append({"path": path, "reason": "mapping keys differ"})
            else:
                for key in a:
                    compare(a[key], b[key], f"{path}.{key}")
        elif isinstance(a, (tuple, list)):
            if type(a) is not type(b) or len(a) != len(b):
                differences.append({"path": path, "reason": "sequence type/length differs"})
            else:
                for index, (first, second) in enumerate(zip(a, b, strict=True)):
                    compare(first, second, f"{path}.{index}")
        else:
            scalar_count += 1
            if type(a) is not type(b) or a != b:
                differences.append({"path": path, "reason": "scalar differs"})

    # The split pilot additionally evaluates at its interruption step, so its
    # best-validation selection can legitimately differ. It must not change the
    # model, optimizer, RNG, input cursor or training configuration.
    for key in ("model", "optimizer", "rng", "step", "data_cursor", "config_sha256"):
        compare(left[key], right[key], key)
    return {"exact": not differences, "tensor_count": tensor_count, "scalar_count": scalar_count,
            "step": left["step"], "data_cursor": left["data_cursor"],
            "difference_count": len(differences), "differences": differences[:20]}


def compare_pilot_summaries(reference: dict, resumed: dict, config: dict) -> dict:
    differences = []
    expected = {
        "final_step": PILOT_STEPS,
        "data_cursor": PILOT_STEPS * config["data"]["batch_size"],
        "parameter_count": parameter_count(config),
        "training_tokens": PILOT_STEPS * config["data"]["batch_size"] * (config["data"]["sequence_length"] - 1),
    }
    for label, summary in (("reference", reference), ("resumed", resumed)):
        for key, value in expected.items():
            if summary.get(key) != value:
                differences.append(f"{label}.{key}")
        for key in ("validation_nll", "validation_perplexity"):
            if not isinstance(summary.get(key), (float, int)) or not math.isfinite(summary[key]):
                differences.append(f"{label}.{key}.nonfinite")
        last = summary.get("last_train", {})
        for key in ("train_loss", "grad_norm", "learning_rate"):
            if not isinstance(last.get(key), (float, int)) or not math.isfinite(last[key]):
                differences.append(f"{label}.last_train.{key}.nonfinite")
    for key in ("validation_nll", "validation_perplexity"):
        if reference.get(key) != resumed.get(key):
            differences.append(key)
    for key in ("train_loss", "grad_norm", "learning_rate"):
        if reference.get("last_train", {}).get(key) != resumed.get("last_train", {}).get(key):
            differences.append(f"last_train.{key}")
    return {"exact": not differences, "differences": differences, **expected}


def _run_logged(command: list[str], logfile: Path, *, env=None) -> None:
    with logfile.open("w") as stream:
        completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=env)
    if completed.returncode:
        tail = logfile.read_text(errors="replace")[-5000:]
        raise RuntimeError(f"Acceptance subprocess failed ({completed.returncode}): {logfile}\n{tail}")


def run_acceptance(config_path: str | Path, output: str | Path) -> dict:
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = acceptance_identity(config_path)
    identity_sha256 = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    success_path = output / "success.json"
    if success_path.exists():
        previous = json.loads(success_path.read_text())
        if previous.get("status") == "passed" and previous.get("identity_sha256") == identity_sha256:
            print(json.dumps({"event": "acceptance_reused", "identity_sha256": identity_sha256}), flush=True)
            return previous
        success_path.unlink()
    # These are only disposable pilot directories owned by this acceptance run.
    # Start the uninterrupted control fresh after an interrupted qualification.
    for name in ("reference", "resumed"):
        if (output / name).exists():
            shutil.rmtree(output / name)
    config = load_config(config_path)
    print(json.dumps({"event": "acceptance_start", **identity}), flush=True)
    test_xml = output / "cpu-tests.xml"
    test_command = [sys.executable, "-m", "pytest", "-q", "scaffolds/transformer/native_fla/tests",
                    "-m", "not gpu", f"--junitxml={test_xml}"]
    _run_logged(test_command, output / "cpu-tests.log", env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    cases = list(ET.parse(test_xml).iter("testcase"))
    if not cases or any(any(case.find(tag) is not None for tag in ("failure", "error", "skipped")) for case in cases):
        raise RuntimeError("Native FLA CPU acceptance requires passing tests with no skips")
    pilots = {}
    for name in ("reference", "split", "resumed"):
        print(json.dumps({"event": "acceptance_worker", "worker": name}), flush=True)
        command = [sys.executable, "-m", "scaffolds.transformer.native_fla.acceptance",
                   "--config", str(config_path), "--output", str(output),
                   "--worker", name]
        _run_logged(command, output / f"{name}.log")
        pilots[name] = json.loads((output / f"{name}.json").read_text())
    resume = compare_checkpoint_states(output / "reference/checkpoints/3", output / "resumed/checkpoints/3")
    metrics = compare_pilot_summaries(pilots["reference"], pilots["resumed"], config)
    optimizer_groups = json.loads((output / "reference/metadata.json").read_text())["optimizer_parameter_groups"]
    resumed_groups = json.loads((output / "resumed/metadata.json").read_text())["optimizer_parameter_groups"]
    if optimizer_groups != resumed_groups or optimizer_groups["parameter_count"] != parameter_count(config):
        raise RuntimeError("Native FLA optimizer policy changed across fresh-process resume")
    report = {"status": "passed" if resume["exact"] and metrics["exact"] else "failed",
              "identity": identity, "identity_sha256": identity_sha256,
              **{key: identity[key] for key in ("launch_commit", "config_sha256", "dependency_lock_sha256",
                                               "gpu_count", "gpu_type")},
              "cpu_tests": {"cases": len(cases), "command": test_command},
              "resume": resume, "metrics": metrics, "pilots": pilots,
              "optimizer_parameter_groups": optimizer_groups}
    _write_json(output / "audit.json", report)
    if report["status"] != "passed":
        raise RuntimeError(f"Native FLA resume differs from uninterrupted training; see {output / 'audit.json'}")
    _write_json(success_path, report)
    for name in ("reference", "resumed"):
        shutil.rmtree(output / name)
    print(json.dumps({"event": "acceptance_passed", "identity_sha256": identity_sha256,
                      "resume": resume, "metrics": metrics}), flush=True)
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker", choices=("reference", "split", "resumed"))
    args = parser.parse_args(argv)
    if args.worker:
        from .train import run_training

        directory = args.output / ("reference" if args.worker == "reference" else "resumed")
        summary = run_training(args.config, output_dir=directory,
                               stop_after_step=SPLIT_STEP if args.worker == "split" else PILOT_STEPS,
                               eval_batches=1, qualification=True)
        if summary.get("status") != "complete":
            raise RuntimeError(f"Qualification worker did not complete: {summary}")
        _write_json(args.output / f"{args.worker}.json", summary)
    else:
        run_acceptance(args.config, args.output)


if __name__ == "__main__":
    main()
