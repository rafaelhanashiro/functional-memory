"""Qualify B300 query-key backward tiles without launching scientific training.

Run inside a one-B300 allocation, from a committed checkout. Each GPU-heavy
stage runs in a fresh process. Original geometry is a process-local diagnostic
override; frozen recipes, production run artifacts, and the learning-rate horizon
are unchanged. The actual production launcher repeats its own acceptance.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
MODULE = "scaffolds.transformer.validation.qualify_query_key_b300"
MODEL_MODULE = "scaffolds.transformer.benchmarks.qualify_query_key_model"
ADAPTER = "experiments.transformer_200m_4b.acceptance"
STUDY = "transformer_200m_4b"
PILOT_STEPS = 20


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def artifact(path):
    return {"path": str(Path(path).resolve()), "sha256": digest(path)}


def recipes(paths):
    from scaffolds.transformer.configs import load_config
    result = []
    for path in paths:
        path = path.resolve()
        config = load_config(path)
        model = config.model
        if (model.attention_coefficient_conditioning != "query_key"
                or model.attention_degree not in (8, 12)
                or model.attention_backend != "fused"
                or config.train.dtype != "bfloat16"
                or config.data.sequence_length != 2049
                or model.qk_head_dim != 64 or model.value_head_dim != 64):
            raise ValueError(f"Expected a fused BF16 degree-8/12 width-64 query-key recipe: {path}")
        result.append({"path": path, "run_id": path.parent.name,
                       "degree": model.attention_degree, "sha256": digest(path)})
    if len({row["run_id"] for row in result}) != len(result):
        raise ValueError("Duplicate run IDs")
    return result


def hardware(config_paths):
    import importlib.metadata
    import jax
    import jax.numpy as jnp
    from scaffolds.transformer.attention.backends import fused_query_key_polynomial as kernel
    from scaffolds.transformer.benchmarks.benchmark_attention import provenance

    devices = jax.local_devices()
    if (len(devices) != 1 or devices[0].platform != "gpu"
            or "B300" not in devices[0].device_kind
            or getattr(devices[0], "compute_capability", None) != "10.3"):
        raise RuntimeError(f"Require exactly one B300: {devices}")
    selected = {}
    for row in recipes(config_paths):
        tiles = kernel._backward_tile_sizes(64, 64, degree=row["degree"], time=2048,
                                            dtype=jnp.bfloat16, interpret=False)
        if tiles != (32, 32) or kernel._tile_sizes(64, 64) != (64, 64):
            raise RuntimeError(f"Production geometry is not forward64/backward32: {tiles}")
        selected[row["run_id"]] = {"degree": row["degree"], "forward": [64, 64], "backward": list(tiles)}
    source = provenance(devices[0])
    for package in ("jax-cuda13-plugin", "nvidia-cublas", "nvidia-cudnn-cu13", "triton"):
        source["package_versions"][package] = importlib.metadata.version(package)
    return {"status": "complete", "provenance": source, "production_geometry": selected}


def throughput_worker(args):
    import jax
    from scaffolds.transformer.attention.backends import fused_query_key_polynomial as kernel
    from scaffolds.transformer.benchmarks.benchmark_training_throughput import benchmark

    if len(args.config) != 1 or args.scratch is None:
        raise ValueError("Throughput worker requires exactly one --config and --scratch")
    observed = hardware(args.config)
    selector = kernel._backward_tile_sizes
    selected = [32, 32] if args.geometry == "production" else [64, 64]
    audit = {"geometry": args.geometry, "backward_tiles": selected, "forward_tiles": [64, 64],
             "override": None if args.geometry == "production" else "process-local original 64x64 backward selector",
             "production_geometry_before_override": observed["production_geometry"],
             "source_commit": args.source_commit, "diagnostic_source_sha256": digest(__file__),
             "unchanged": ["recipe", "initialization", "data", "precision", "learning-rate horizon", "global batch"]}
    if args.geometry == "original":
        kernel._backward_tile_sizes = lambda *a, **kw: (64, 64)
    jax.clear_caches()
    try:
        outcome = benchmark(SimpleNamespace(config=args.config[0], scratch=args.scratch, output=args.output,
                                             experiment=STUDY, warmup=args.warmup, iterations=args.iterations))
        if outcome["status"] != "complete" or not outcome["all_finite"] or not outcome["scratch_removed"]:
            raise RuntimeError("Incomplete throughput qualification")
    finally:
        kernel._backward_tile_sizes = selector
        if args.output.exists():
            outcome = read(args.output)
            outcome["geometry_audit"] = audit
            write(args.output, outcome)


def qualify(args):
    from scaffolds.transformer.acceptance import require_passing_tests, validate_resume_audit

    rows = recipes(args.config)
    output = args.output_dir.resolve()
    summary_path = output / "summary.json"
    if summary_path.exists():
        raise ValueError(f"Choose a fresh output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    scratch = ROOT / ".cache/query-key-b300-qualification" / (
        f"{os.environ.get('SLURM_JOB_ID', 'no-job')}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}")
    if scratch.exists() or output.is_relative_to(scratch):
        raise ValueError("Diagnostic scratch must be fresh and separate from reports")
    scratch.mkdir(parents=True)
    environment = dict(os.environ, PYTHONUNBUFFERED="1", WANDB_MODE="disabled",
                       SHARED_TRANSFORMER_SOURCE_COMMIT=args.source_commit)
    report = {"format": "b300-query-key-production-qualification", "schema_version": 1,
              "status": "running", "source_commit": args.source_commit,
              "started_utc": datetime.now(UTC).isoformat(), "home_cluster": "kennel",
              "node": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "dependency_lock_sha256": digest(ROOT / "scaffolds/transformer/uv.lock"),
              "orchestrator_source_sha256": digest(__file__),
              "recipes": [{**row, "path": str(row["path"])} for row in rows],
              "scratch": str(scratch), "scratch_removed": False,
              "notes": ["Focused GPU tests; CPU checks are recorded separately before launch.",
                        "Fresh-process throughput workers compare only backward geometry.",
                        "Pilot/resume workers use the exact recipe and audit checkpoint state, data, and RNG.",
                        "Diagnostic workers do not apply the full-training allocation-budget gate; production acceptance repeats it."],
              "stages": [], "runs": {}}
    started = time.monotonic()
    write(summary_path, report)

    def run(name, command):
        entry = {"name": name, "command": list(map(str, command)), "status": "running",
                 "started_utc": datetime.now(UTC).isoformat()}
        report["stages"].append(entry)
        report["active_stage"] = name
        write(summary_path, report)
        print(f"QUALIFICATION_STAGE {name}", flush=True)
        stage_started = time.monotonic()
        result = subprocess.run(entry["command"], cwd=ROOT, env=environment, check=False)
        entry.update(exit_code=result.returncode, elapsed_seconds=time.monotonic() - stage_started,
                     status="complete" if result.returncode == 0 else "failed")
        write(summary_path, report)
        if result.returncode:
            raise RuntimeError(f"{name} exited {result.returncode}")
        return entry

    common = [sys.executable, "-m", MODULE, "--source-commit", args.source_commit]
    try:
        hardware_path = output / "hardware.json"
        run("hardware", [*common, "--worker", "hardware", "--config", *args.config, "--output", hardware_path])
        report["hardware"] = artifact(hardware_path)
        tests = ["scaffolds/transformer/tests/test_query_key_tiling.py"] + [
            "scaffolds/transformer/tests/test_qk_rms_attention_backends.py::"
            f"test_full_context_qk_rms_backend[query_key-{degree}-fused]"
            for degree in sorted({row["degree"] for row in rows})]
        junit = output / "gpu-tests.xml"
        entry = run("focused-gpu-tests", [sys.executable, "-m", "pytest", "-c", "scaffolds/transformer/pyproject.toml",
                    "-q", "--import-mode=importlib", "-m", "gpu", *tests, f"--junitxml={junit}"])
        entry.update(tests_passed=require_passing_tests(junit), junit=artifact(junit))
        write(summary_path, report)
        for row in rows:
            run_id, config = row["run_id"], row["path"]
            directory = output / run_id
            directory.mkdir()
            current = report["runs"][run_id] = {"degree": row["degree"]}
            parity = directory / "full-model-parity.json"
            run(f"{run_id}:full-model-parity", [sys.executable, "-m", MODEL_MODULE,
                                               "--config", config, "--output", parity])
            if read(parity)["status"] != "complete":
                raise RuntimeError(f"Full-model parity failed for {run_id}")
            current["full_model_parity"] = artifact(parity)
            timing = {}
            for geometry in ("production", "original"):
                path = directory / f"throughput-{geometry}.json"
                run(f"{run_id}:throughput-{geometry}", [*common, "--worker", "throughput", "--config", config,
                    "--geometry", geometry, "--scratch", scratch / run_id / f"throughput-{geometry}",
                    "--output", path, "--warmup", args.warmup, "--iterations", args.iterations])
                data = read(path)
                if data["status"] != "complete" or not data["all_finite"] or not data["scratch_removed"]:
                    raise RuntimeError(f"Incomplete {geometry} throughput for {run_id}")
                timing[geometry] = {**artifact(path), "steady_training": data["steady_training"],
                                    "steady_compute": data["steady_compute"]}
            timing["mean_update_speedup"] = (timing["original"]["steady_training"]["mean_seconds"]
                                             / timing["production"]["steady_training"]["mean_seconds"])
            current["throughput"] = timing
            pilot_dir = scratch / run_id / "resume"
            pilot_path, resume_path = directory / "pilot-result.json", directory / "resume-result.json"
            worker = [sys.executable, "-m", ADAPTER, str(config), "--directory", str(pilot_dir),
                      "--gpu-count", "1", "--gpu-type", "B300", "--worker", "pilot"]
            run(f"{run_id}:pilot", [*worker, "--output", pilot_path, "--stop-after-step", PILOT_STEPS])
            before = read(pilot_path)
            if (before["final_step"] != PILOT_STEPS or before["data_parallel_devices"] != 1
                    or before["attention_backend"] != "fused"):
                raise RuntimeError(f"Wrong pilot step, allocation, or backend for {run_id}")
            measurements = [json.loads(line) for line in (pilot_dir / "pilot/metrics.jsonl").read_text().splitlines() if line.strip()]
            validation = next(item for item in measurements
                              if item["step"] == PILOT_STEPS and item.get("validation_nll") is not None)
            if not math.isfinite(validation["validation_nll"]) or not validation["attention_diagnostics"]["valid"]:
                raise RuntimeError(f"Invalid pilot validation or attention diagnostics for {run_id}")
            if any((pilot_dir / "pilot").glob("numerical_failure_step_*.json")):
                raise RuntimeError(f"Pilot numerical failure for {run_id}")
            run(f"{run_id}:resume", [*worker, "--output", resume_path, "--stop-after-step", PILOT_STEPS + 1])
            after = read(resume_path)
            if (after["restored_step"] != PILOT_STEPS or after["final_step"] != PILOT_STEPS + 1
                    or after["data_parallel_devices"] != 1 or after["attention_backend"] != "fused"
                    or after["data_start_sequence"] != PILOT_STEPS * 256):
                raise RuntimeError(f"Wrong resume step, allocation, or data cursor for {run_id}")
            validate_resume_audit(before, after)
            current["pilot_resume"] = {"status": "complete", "pilot": artifact(pilot_path), "resume": artifact(resume_path),
                "pilot_step": PILOT_STEPS, "resumed_step": PILOT_STEPS + 1,
                "validation_nll": validation["validation_nll"], "attention_diagnostics_valid": True,
                "checkpoint_state_equal": before["checkpoint_audit"]["checkpoint_state_sha256"] == after["checkpoint_audit"]["start_state_sha256"],
                "next_batch_and_rng_equal": before["checkpoint_audit"]["next_update"] == after["checkpoint_audit"]["first_update"],
                "scratch_removed": False}
            write(summary_path, report)
            shutil.rmtree(pilot_dir)
            current["pilot_resume"]["scratch_removed"] = True
            if digest(config) != row["sha256"]:
                raise RuntimeError(f"Input recipe changed for {run_id}")
            write(summary_path, report)
        report["source_recipes_unchanged"] = all(digest(row["path"]) == row["sha256"] for row in rows)
        if not report["source_recipes_unchanged"]:
            raise RuntimeError("Input recipes changed")
        report["status"] = "complete"
        report.pop("active_stage", None)
        write(summary_path, report)
        shutil.rmtree(scratch)
        report["scratch_removed"] = True
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        if report["stages"]:
            report["stages"][-1].update(status="failed", validation_error=report["error"])
        raise
    finally:
        report.update(finished_utc=datetime.now(UTC).isoformat(), elapsed_seconds=time.monotonic() - started)
        write(summary_path, report)
    print("QUALIFICATION_COMPLETE " + str(summary_path), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-commit", default=os.environ.get("SHARED_TRANSFORMER_SOURCE_COMMIT"))
    parser.add_argument("--worker", choices=("hardware", "throughput"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--geometry", choices=("production", "original"), default="production")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    args.config = [path.resolve() for path in args.config]
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    args.source_commit = args.source_commit or source
    if args.source_commit != source:
        parser.error("Qualification must run from the specified source commit")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run inside a Slurm allocation")
    if min(args.warmup, args.iterations) < 3:
        parser.error("Require at least three warmup and timed updates")
    if args.worker:
        if args.output is None:
            parser.error("Workers require --output")
        args.output = args.output.resolve()
        if args.worker == "hardware":
            write(args.output, hardware(args.config))
        else:
            if args.scratch is None or not args.scratch.resolve().is_relative_to(ROOT / ".cache"):
                parser.error("Throughput scratch must be under this worktree's .cache")
            throughput_worker(args)
    else:
        if args.output_dir is None:
            parser.error("Qualification requires --output-dir")
        qualify(args)


if __name__ == "__main__":
    main()
