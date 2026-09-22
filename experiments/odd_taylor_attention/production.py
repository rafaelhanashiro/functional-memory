"""Run a 5,000-update stability stage, verify its restart, then finish the budget."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from experiments.odd_taylor_attention.sweep import STABILITY_STEPS, manifest_path, write_json


def validate_stability_stage(manifest: dict, run: dict, outcome: dict, records: list[dict]) -> dict:
    """Finite committed evaluation and exact durable state are mandatory to proceed."""
    audit = outcome.get("checkpoint_audit", {})
    if (outcome.get("final_step") != STABILITY_STEPS
            or outcome.get("latest_checkpoint_after") != STABILITY_STEPS
            or outcome.get("attention_degree") != run["attention_degree"]
            or outcome.get("data_parallel_devices") != manifest["scheduler"]["gpu_count"]
            or run["config"]["optim"]["total_steps"] != 19074):
        raise RuntimeError("Stability stage changed the degree, full schedule, checkpoint, or allocation")
    hashes = [audit.get(key) for key in ("final_state_sha256", "checkpoint_state_sha256")]
    update = audit.get("next_update", {})
    hashes += [update.get(key) for key in ("batch_sha256", "rng_sha256")]
    if any(not isinstance(value, str) or len(value) != 64
           or any(char not in "0123456789abcdef" for char in value) for value in hashes):
        raise RuntimeError("Stability stage has no valid exact checkpoint/input/RNG audit")
    if hashes[0] != hashes[1] or update.get("step") != STABILITY_STEPS + 1:
        raise RuntimeError("Stability stage durable state or next-update cursor changed")
    validation = next((row for row in records if row["step"] == STABILITY_STEPS), None)
    if (validation is None or validation.get("source_commit") != manifest["source_commit"]
            or validation.get("attention_degree") != run["attention_degree"]
            or not all(math.isfinite(validation.get(key, math.nan))
                       for key in ("validation_nll", "validation_perplexity"))):
        raise RuntimeError("Stability stage requires its matching finite committed validation")
    return validation


def validate_restart(stage: dict, restored: dict) -> None:
    audit = stage["checkpoint_audit"]
    if (restored.get("restored_step") != STABILITY_STEPS
            or restored.get("state_sha256") != audit["final_state_sha256"]
            or restored.get("next_update") != audit["next_update"]):
        raise RuntimeError("Stability restart changed checkpoint state, next batch, or RNG")


def worker_result(run: dict, worker: str) -> dict:
    from experiments.shared_transformer.configs import load_config
    config = load_config(run["config_path"])
    if worker in ("stability", "continuation"):
        from experiments.shared_transformer.training.loop import run_training
        return run_training(config, resume=True,
                            stop_after_step=STABILITY_STEPS if worker == "stability" else None,
                            audit_checkpoint=True)
    if worker != "restore":
        raise ValueError(f"Unknown production worker: {worker}")
    import jax
    import numpy as np
    from dataclasses import replace
    from experiments.shared_transformer.data.loaders import make_train_iterator
    from experiments.shared_transformer.data.packed import load_packed_manifest
    from experiments.shared_transformer.training.checkpointing import (
        make_checkpoint_manager, restore_latest_checkpoint, state_sha256,
    )
    from experiments.shared_transformer.training.distributed import resolve_batch_layout
    from experiments.shared_transformer.training.loop import _update_audit, resolve_packed_data_cursor
    from experiments.shared_transformer.training.state import create_state
    rng = jax.random.PRNGKey(config.train.seed)
    with make_checkpoint_manager(config) as manager:
        state, restored_step = restore_latest_checkpoint(manager, create_state(config, rng))
        if restored_step != STABILITY_STEPS or int(state.step) != STABILITY_STEPS:
            raise RuntimeError("Stability checkpoint is not the latest durable state")
        digest = state_sha256(state)
    train_manifest = load_packed_manifest(config.data.train_manifest)
    eval_manifest = load_packed_manifest(config.data.eval_manifest)
    start_sequence = resolve_packed_data_cursor(
        config, restored_step, validated_train_manifest=train_manifest,
        validated_eval_manifest=eval_manifest,
    )
    layout = resolve_batch_layout(config)
    iterator = make_train_iterator(replace(config.data, batch_size=layout.global_micro_batch_size),
                                   start_sequence=start_sequence, validated_manifest=train_manifest)
    batches = np.stack([next(iterator) for _ in range(layout.grad_accum_steps)])
    return {"restored_step": restored_step, "state_sha256": digest,
            "next_update": _update_audit(restored_step + 1, batches,
                                          jax.random.fold_in(rng, restored_step + 1))}


def launch_worker(manifest: dict, run: dict, worker: str, output: Path) -> dict:
    command = [sys.executable, "-m", "experiments.odd_taylor_attention.production",
               "--manifest", str(manifest_path(Path(manifest["project_root"]))),
               "--run-name", run["name"], "--worker", worker, "--output", str(output)]
    subprocess.run(command, cwd=manifest["execution_root"], check=True)
    return json.loads(output.read_text())


def run_production(manifest: dict, run: dict) -> dict:
    from experiments.odd_taylor_attention.preflight import run_preflight
    from experiments.polynomial_attention.plot import committed_records
    run_preflight(manifest, check_only=True, run_name=run["name"])
    root = Path(run["workdir"])
    gate_path = root / "stability_gate.json"
    stage_path = root / "stability_stage.json"
    stage = "stability"
    try:
        if not gate_path.exists():
            checkpoints = [int(path.name) for path in (root / "checkpoints").iterdir()
                           if path.is_dir() and path.name.isdigit()]
            if checkpoints and max(checkpoints) > STABILITY_STEPS:
                raise RuntimeError("Run advanced beyond the stability stage without its durable gate")
            outcome = (json.loads(stage_path.read_text()) if stage_path.exists()
                       else launch_worker(manifest, run, "stability", stage_path))
            validation = validate_stability_stage(manifest, run, outcome, committed_records(root))
            stage = "stability-restart"
            restored = launch_worker(manifest, run, "restore", root / "stability_restart.json")
            validate_restart(outcome, restored)
            write_json(gate_path, {
                "format": "odd-taylor-stability-gate", "version": 1, "status": "passed",
                "identity": manifest["identity"], "source_commit": manifest["source_commit"],
                "attention_degree": run["attention_degree"], "step": STABILITY_STEPS,
                "full_schedule_steps": run["config"]["optim"]["total_steps"],
                "checkpoint_audit": outcome["checkpoint_audit"], "validation": validation,
                "restart": restored, "completed_utc": datetime.now(UTC).isoformat(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            })
        gate = json.loads(gate_path.read_text())
        if (gate.get("identity") != manifest["identity"] or gate.get("status") != "passed"
                or gate.get("attention_degree") != run["attention_degree"] or gate.get("step") != STABILITY_STEPS
                or gate.get("full_schedule_steps") != run["config"]["optim"]["total_steps"]):
            raise RuntimeError("Stability gate no longer matches the frozen run")
        stage = "continuation"
        outcome = launch_worker(manifest, run, "continuation", root / "production_outcome.json")
        if outcome.get("start_step") == STABILITY_STEPS:
            audit = outcome.get("checkpoint_audit", {})
            validate_restart({"checkpoint_audit": gate["checkpoint_audit"]}, {
                "restored_step": outcome.get("restored_step"),
                "state_sha256": audit.get("start_state_sha256"),
                "next_update": audit.get("first_update"),
            })
        # Keep the attempt history, but remove the active failure marker after
        # an infrastructure interruption has been successfully resumed.
        (root / "production_failure.json").unlink(missing_ok=True)
        return outcome
    except BaseException as error:
        failures = [json.loads(path.read_text()) for path in sorted(root.glob("numerical_failure_step_*.json"))]
        record = {
            "identity": manifest["identity"], "source_commit": manifest["source_commit"],
            "attention_degree": run["attention_degree"], "stage": stage, "status": "failed",
            "error": f"{type(error).__name__}: {error}", "numerical_failures": failures,
            "failed_utc": datetime.now(UTC).isoformat(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        write_json(root / "production_failure.json", record)
        attempt = f"{os.environ.get('SLURM_JOB_ID', 'local')}-{os.environ.get('SLURM_RESTART_COUNT', '0')}"
        write_json(root / "failure_history" / f"production-{attempt}.json", record)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--worker", choices=("stability", "restore", "continuation"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    run = next(run for run in manifest["runs"] if run["name"] == args.run_name)
    if args.worker:
        if args.output is None:
            parser.error("Workers require --output")
        from experiments.odd_taylor_attention.preflight import verify_manifest
        verify_manifest(manifest)
        write_json(args.output, worker_result(run, args.worker))
    else:
        run_production(manifest, run)


if __name__ == "__main__":
    main()
