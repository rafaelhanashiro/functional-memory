"""Qualify pilot runtime projection against one warm production validation.

Run as a module in a single-GPU allocation. Pilot, full validation, and resume
use separate processes; only the disposable pilot's cadence/artifact paths differ
from the input recipe. A failed phase retains its checkpoint and logs for review.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[3]
MODULE = "scaffolds.transformer.validation.runtime_estimate"


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record_path(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _check_source(commit):
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    expected = subprocess.check_output(["git", "rev-parse", commit], cwd=ROOT, text=True).strip()
    if actual != expected:
        raise RuntimeError(f"Expected source {expected}, found {actual}")
    # Mutable validation records are not execution code. Check this harness
    # separately, alongside the ordinary frozen scaffold and dependency lock.
    subprocess.run(["git", "diff", "--exit-code", expected, "--", "scaffolds/transformer",
                    ":!scaffolds/transformer/validation", str(Path(__file__).relative_to(ROOT))],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    for script in (Path(__file__), Path(__file__).with_suffix(".sbatch")):
        tracked_script = subprocess.check_output(
            ["git", "show", f"{expected}:{script.relative_to(ROOT)}"], cwd=ROOT)
        if tracked_script != script.read_bytes():
            raise RuntimeError("Qualification harness differs from the launch commit")
    return actual


def _hardware():
    import jax

    devices = jax.local_devices()
    if jax.default_backend() != "gpu" or len(devices) != 1:
        raise RuntimeError(f"Qualification requires exactly one GPU, got {devices}")
    return {
        "devices": [{"kind": d.device_kind, "compute_capability": d.compute_capability} for d in devices],
        "packages": {name: version(name) for name in ("jax", "jaxlib", "flax", "optax", "orbax-checkpoint")},
        "node": os.environ.get("SLURMD_NODENAME"), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "environment": {name: os.environ.get(name) for name in (
            "TRANSFORMER_CUDA_ENVIRONMENT", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE",
            "XLA_PYTHON_CLIENT_MEM_FRACTION", "TRANSFORMER_HOST_STAGING", "NCCL_P2P_DISABLE")},
    }


def _summarize(values):
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise RuntimeError("Timing samples must be positive and finite")
    return {"count": len(values), "min": min(values), "median": statistics.median(values),
            "mean": statistics.mean(values), "max": max(values), "samples": values}


def _measure_validation(directory, production):
    import jax

    from scaffolds.transformer.configs import load_config
    from scaffolds.transformer.data.loaders import make_eval_iterator
    from scaffolds.transformer.data.packed import load_packed_manifest, manifest_fingerprint
    from scaffolds.transformer.training.checkpointing import (
        make_checkpoint_manager, restore_latest_checkpoint, state_sha256,
    )
    from scaffolds.transformer.training.diagnostics import attention_diagnostics, fixed_diagnostic_sample
    from scaffolds.transformer.training.distributed import resolve_batch_layout
    from scaffolds.transformer.training.loop import evaluate
    from scaffolds.transformer.training.state import create_state

    config = load_config(directory / "pilot" / "config.json")
    layout = resolve_batch_layout(config)
    state = create_state(config, jax.random.PRNGKey(config.train.seed))
    manager = make_checkpoint_manager(config)
    try:
        state, restored_step = restore_latest_checkpoint(manager, state)
        if restored_step is None:
            raise RuntimeError("Full validation requires the pilot checkpoint")
    finally:
        manager.close()
    checkpoint_sha = state_sha256(state)
    pilot = json.loads((directory / "pilot.json").read_text())["outcome"]
    if checkpoint_sha != pilot["checkpoint_audit"]["final_state_sha256"]:
        raise RuntimeError("Full validation restored a different pilot state")
    manifest = load_packed_manifest(config.data.eval_manifest) if config.data.source == "packed" else None
    if manifest is not None:
        identity = json.loads((directory / "pilot" / "data_identity.json").read_text())
        if manifest_fingerprint(config.data.eval_manifest) != identity["eval_manifest_sha256"]:
            raise RuntimeError("Held-out manifest changed after the pilot")
    sample, provenance = fixed_diagnostic_sample(config, validated_manifest=manifest)

    def diagnostics():
        report = attention_diagnostics(state, config, sample, provenance)
        if not report["valid"]:
            raise FloatingPointError(f"Invalid attention diagnostics: {report.get('issues')}")
        return report

    def evaluation(batches, samples=None):
        value = evaluate(state, make_eval_iterator(config.data, validated_manifest=manifest), batches,
                         eval_seed=config.train.seed, micro_batch_size=layout.global_micro_batch_size,
                         batch_seconds=samples)
        if not math.isfinite(value):
            raise FloatingPointError("Full validation produced a nonfinite NLL")
        return value

    warmup_start = time.perf_counter()
    diagnostics()
    diagnostic_warmup = time.perf_counter() - warmup_start
    warmup_start = time.perf_counter()
    evaluation(1)
    evaluation_warmup = time.perf_counter() - warmup_start
    batch_seconds = []
    start = time.perf_counter()
    report = diagnostics()
    diagnostic_seconds = time.perf_counter() - start
    evaluation_start = time.perf_counter()
    loss = evaluation(production.train.eval_batches, batch_seconds)
    evaluation_seconds = time.perf_counter() - evaluation_start
    combined = time.perf_counter() - start
    return {
        "restored_step": restored_step, "checkpoint_state_sha256": checkpoint_sha,
        "validation_nll": loss, "validation_perplexity": math.exp(loss),
        "validation_batches": production.train.eval_batches, "global_batch_size": config.data.batch_size,
        "micro_batch_size": layout.global_micro_batch_size,
        "prediction_tokens": production.train.eval_batches * config.data.batch_size * (config.data.sequence_length - 1),
        "diagnostics_valid": report["valid"], "diagnostic_sample": provenance,
        "diagnostic_warmup_seconds": diagnostic_warmup, "evaluation_warmup_seconds": evaluation_warmup,
        "diagnostic_seconds": diagnostic_seconds, "evaluation_seconds": evaluation_seconds,
        "combined_seconds": combined, "evaluation_batch_seconds": _summarize(batch_seconds),
        "scope": "One warm full held-out evaluation plus one warm diagnostic call; excludes checkpoint restore and compilation",
    }


def _phase(args):
    from scaffolds.transformer.configs import load_config
    from scaffolds.transformer.training.loop import run_training

    hardware = _hardware()
    if args.phase == "validation":
        result = _measure_validation(args.directory, load_config(args.config))
    else:
        config = load_config(args.directory / "pilot.yaml")
        result = run_training(config, resume=args.phase == "resume",
                              stop_after_step=args.pilot_steps + (args.phase == "resume"),
                              audit_checkpoint=True, profile_timing=args.phase == "pilot")
    _write(args.directory / f"{args.phase}.json", {"hardware": hardware, "outcome": result})


def _run(args):
    from scaffolds.transformer.acceptance import project_runtime, validate_resume_audit
    from scaffolds.transformer.configs import config_to_dict, load_config
    from scaffolds.transformer.configs.loader import save_config

    if args.directory.exists() or args.output.exists():
        raise FileExistsError("Use fresh qualification directory and output paths")
    production = load_config(args.config)
    if production.continuation.parent_checkpoint or not 2 <= args.pilot_steps < production.optim.total_steps - 1:
        raise ValueError("Use a fresh-training recipe with room for pilot and resume updates")
    original_sha = _sha(args.config)
    args.directory.mkdir(parents=True)
    config = load_config(args.config)
    config.train.source_commit = args.source_commit
    config.train.workdir = args.directory / "pilot"
    config.train.checkpoint_dir = None
    config.train.wandb_mode = "disabled"
    config.train.eval_every = args.pilot_steps
    config.train.eval_early_every = None
    config.train.eval_early_until = 0
    config.train.eval_batches = 1
    save_config(config, args.directory / "pilot.yaml")
    record = {
        "format": "shared-transformer-runtime-estimate-qualification", "version": 1,
        "started_utc": datetime.now(UTC).isoformat(), "source_commit": args.source_commit,
        "input_recipe": _record_path(args.config), "input_recipe_sha256": original_sha,
        "recipe": config_to_dict(production), "pilot_steps": args.pilot_steps,
        "evidence_directory": _record_path(args.directory), "status": "running",
    }
    try:
        for phase in ("pilot", "validation", "resume"):
            print(f"runtime_estimate_phase={phase}", flush=True)
            command = [sys.executable, "-m", MODULE, "--phase", phase, "--config", str(args.config),
                       "--directory", str(args.directory), "--source-commit", args.source_commit,
                       "--pilot-steps", str(args.pilot_steps)]
            with (args.directory / f"{phase}.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        pilot = json.loads((args.directory / "pilot.json").read_text())
        resumed = json.loads((args.directory / "resume.json").read_text())
        full = json.loads((args.directory / "validation.json").read_text())["outcome"]
        validate_resume_audit(pilot["outcome"], resumed["outcome"])
        if _sha(args.config) != original_sha:
            raise RuntimeError("Input recipe changed during qualification")
        _check_source(args.source_commit)
        timing = pilot["outcome"]["timing"]
        projection = project_runtime(timing, production)
        rates = projection["seconds_per_unit"]
        predicted = rates["diagnostics"] + rates["evaluation_batch"] * production.train.eval_batches
        record.update(status="passed", hardware=pilot["hardware"], timing=timing, projection=projection,
                      full_validation=full, validation_comparison={
                          "predicted_seconds": predicted, "observed_seconds": full["combined_seconds"],
                          "observed_over_predicted": full["combined_seconds"] / predicted,
                          "relative_error": (predicted - full["combined_seconds"]) / full["combined_seconds"],
                          "timing_threshold_applied": False},
                      warm_sample_summaries={name: _summarize(timing[name]) for name in (
                          "diagnostic_seconds", "evaluation_batch_seconds", "checkpoint_seconds")},
                      exact_resume_valid=True, checkpoint_audits={
                          "pilot": pilot["outcome"]["checkpoint_audit"],
                          "resume": resumed["outcome"]["checkpoint_audit"]})
        record["warm_sample_summaries"]["training_update_seconds"] = _summarize(timing["train_step_seconds"][1:])
        record["finished_utc"] = datetime.now(UTC).isoformat()
        _write(args.output, record)
        shutil.rmtree(args.directory / "pilot" / "checkpoints")
        record["scratch_checkpoints_removed"] = True
        _write(args.output, record)
        print(json.dumps({"output": str(args.output), "status": record["status"],
                          **record["validation_comparison"]}), flush=True)
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}",
                      finished_utc=datetime.now(UTC).isoformat(), scratch_checkpoints_removed=False)
        _write(args.output, record)
        (args.directory / "failure.txt").write_text(traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-commit", default=os.environ.get("LAUNCH_COMMIT"))
    parser.add_argument("--directory", type=Path, default=ROOT / "scaffolds/transformer/validation/logs" /
                        f"runtime-estimate-{os.environ.get('SLURM_JOB_ID', 'local')}")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pilot-steps", type=int, default=20)
    parser.add_argument("--phase", choices=("pilot", "validation", "resume"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.source_commit or (not args.phase and args.output is None):
        parser.error("--source-commit (or LAUNCH_COMMIT) and --output are required")
    args.config, args.directory = args.config.resolve(), args.directory.resolve()
    if args.output is not None:
        args.output = args.output.resolve()
    args.source_commit = _check_source(args.source_commit)
    (_phase if args.phase else _run)(args)


if __name__ == "__main__":
    main()
