"""Execute one checkpoint-aligned segment without changing the full LR schedule."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from scaffolds.transformer.studies.key_dimension_taylor.sweep import active_runs, manifest_path, write_json


def validate_segment_restart(previous: dict, restored: dict, expected_start: int) -> None:
    audit = previous.get("checkpoint_audit", {})
    if (previous.get("final_step") != expected_start or restored.get("restored_step") != expected_start
            or restored.get("state_sha256") != audit.get("final_state_sha256")
            or audit.get("final_state_sha256") != audit.get("checkpoint_state_sha256")
            or restored.get("next_update") != audit.get("next_update")):
        raise RuntimeError("Segment restart changed durable state, next batch, or RNG")


def worker_result(run: dict, worker: str, end_step: int) -> dict:
    from scaffolds.transformer.configs import load_config
    config = load_config(run["config_path"])
    if worker == "train":
        from scaffolds.transformer.training.loop import run_training
        return run_training(config, resume=True, stop_after_step=end_step, audit_checkpoint=True)
    if worker != "restore":
        raise ValueError(f"Unknown worker: {worker}")
    import jax
    import numpy as np
    from dataclasses import replace
    from scaffolds.transformer.data.loaders import make_train_iterator
    from scaffolds.transformer.data.packed import load_packed_manifest
    from scaffolds.transformer.training.checkpointing import make_checkpoint_manager, restore_latest_checkpoint, state_sha256
    from scaffolds.transformer.training.distributed import resolve_batch_layout
    from scaffolds.transformer.training.loop import _update_audit, resolve_packed_data_cursor
    from scaffolds.transformer.training.state import create_state
    rng = jax.random.PRNGKey(config.train.seed)
    with make_checkpoint_manager(config) as manager:
        state, restored_step = restore_latest_checkpoint(manager, create_state(config, rng))
        if restored_step is None or int(state.step) != restored_step:
            raise RuntimeError("Continuation lacks its durable checkpoint")
        digest = state_sha256(state)
    train_manifest = load_packed_manifest(config.data.train_manifest)
    eval_manifest = load_packed_manifest(config.data.eval_manifest)
    cursor = resolve_packed_data_cursor(config, restored_step, validated_train_manifest=train_manifest,
                                        validated_eval_manifest=eval_manifest)
    layout = resolve_batch_layout(config)
    iterator = make_train_iterator(replace(config.data, batch_size=layout.global_micro_batch_size),
                                   start_sequence=cursor, validated_manifest=train_manifest)
    batches = np.stack([next(iterator) for _ in range(layout.grad_accum_steps)])
    return {"restored_step": restored_step, "state_sha256": digest,
            "next_update": _update_audit(restored_step + 1, batches, jax.random.fold_in(rng, restored_step + 1))}


def launch_worker(manifest: dict, run: dict, worker: str, end_step: int, output: Path) -> dict:
    command = [sys.executable, "-m", "scaffolds.transformer.studies.key_dimension_taylor.production",
                    "--manifest", str(manifest_path(Path(manifest["project_root"]))), "--run-name", run["name"],
                    "--worker", worker, "--end-step", str(end_step), "--output", str(output)]
    process = subprocess.run(command, cwd=manifest["execution_root"], check=False)
    if process.returncode:
        failure = json.loads(output.read_text()) if output.exists() else {}
        if failure.get("numerical_failure"):
            raise FloatingPointError(failure["error"])
        raise subprocess.CalledProcessError(process.returncode, command)
    return json.loads(output.read_text())


def run_segment(manifest: dict, run: dict, index: int) -> dict:
    from scaffolds.transformer.studies.key_dimension_taylor.preflight import run_preflight
    run_preflight(manifest, check_only=True, run_name=run["name"])
    pilot = json.loads((Path(manifest["preflight_dir"]) / "checks" / f"{run['name']}.json").read_text())
    segments = pilot["runtime"]["segments"]
    if index < 0 or index >= len(segments):
        raise ValueError("Unknown production segment")
    segment = segments[index]
    start, end = segment["start_step"], segment["end_step"]
    root = Path(run["workdir"])
    segment_root = root / "segments"
    segment_root.mkdir(exist_ok=True)
    output = segment_root / f"segment-{index:03d}.json"
    if (root / "completion.json").exists():
        # The shared trainer validates completion and its retained-best receipt.
        return launch_worker(manifest, run, "train", run["config"]["optim"]["total_steps"], output)
    stage = "restore" if index else "train"
    try:
        if any(root.glob("numerical_failure_step_*.json")):
            raise FloatingPointError("This coordinate already has a numerical failure; later segments are suppressed")
        restored = None
        if index:
            previous_path = segment_root / f"segment-{index - 1:03d}.json"
            previous = json.loads(previous_path.read_text())
            restored = launch_worker(manifest, run, "restore", end, segment_root / f"restore-{index:03d}.json")
            if restored["restored_step"] == start:
                validate_segment_restart(previous, restored, start)
            elif not start < restored["restored_step"] <= end:
                raise RuntimeError("Requeued segment restored outside its assigned update interval")
        stage = "train"
        result = launch_worker(manifest, run, "train", end, output)
        if result.get("final_step") != end or not start <= result.get("start_step", -1) <= end:
            raise RuntimeError("Segment did not preserve its assigned update interval")
        audit = result.get("checkpoint_audit", {})
        if audit.get("final_state_sha256") != audit.get("checkpoint_state_sha256") or not audit.get("final_state_sha256"):
            raise RuntimeError("Segment lacks an exact live/durable state audit")
        if restored is not None and result.get("start_step") < end:
            if (audit.get("start_state_sha256") != restored["state_sha256"]
                    or audit.get("first_update") != restored["next_update"]):
                raise RuntimeError("Training process differs from the independently audited restart")
        receipt = {"format": "key-dimension-taylor-segment", "identity": manifest["identity"],
                   "source_commit": manifest["source_commit"], "index": index, **segment,
                   "qk_head_dim": run["qk_head_dim"], "attention_degree": run["attention_degree"],
                   "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "status": "complete",
                   "completed_utc": datetime.now(UTC).isoformat()}
        write_json(segment_root / f"receipt-{index:03d}.json", receipt)
        (root / "production_failure.json").unlink(missing_ok=True)
        return result
    except BaseException as error:
        failures = [json.loads(path.read_text()) for path in sorted(root.glob("numerical_failure_step_*.json"))]
        record = {"identity": manifest["identity"], "source_commit": manifest["source_commit"],
                  "qk_head_dim": run["qk_head_dim"], "attention_degree": run["attention_degree"],
                  "segment_index": index, **segment, "stage": stage, "status": "failed", "numerical_failures": failures,
                  "failure_kind": "numerical" if failures or isinstance(error, FloatingPointError) else "infrastructure",
                  "error": f"{type(error).__name__}: {error}", "failed_utc": datetime.now(UTC).isoformat(),
                  "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
        write_json(root / "production_failure.json", record)
        attempt = f"{os.environ.get('SLURM_JOB_ID', 'local')}-{os.environ.get('SLURM_RESTART_COUNT', '0')}"
        write_json(root / "failure_history" / f"segment-{index:03d}-{attempt}.json", record)
        raise


def record_interruption(manifest: dict, run: dict, index: int, exit_code: int) -> dict:
    """Bounded launcher fallback for signals and failures outside Python guards."""
    root = Path(run["workdir"])
    path = root / "production_failure.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("slurm_job_id") == os.environ.get("SLURM_JOB_ID"):
            return existing
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    record = {"identity": manifest["identity"], "source_commit": manifest["source_commit"],
              "qk_head_dim": run["qk_head_dim"], "attention_degree": run["attention_degree"],
              "segment_index": index, "step": metadata.get("step"), "status": "failed",
              "failure_kind": "infrastructure", "stage": "launcher_exit_or_signal", "exit_code": exit_code,
              "error": f"Allocation launcher exited with status {exit_code}",
              "failed_utc": datetime.now(UTC).isoformat(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    write_json(path, record)
    attempt = f"{os.environ.get('SLURM_JOB_ID', 'local')}-{os.environ.get('SLURM_RESTART_COUNT', '0')}"
    write_json(root / "failure_history" / f"launcher-{index:03d}-{attempt}.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--segment", type=int, default=0)
    parser.add_argument("--worker", choices=("restore", "train"))
    parser.add_argument("--end-step", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--record-interruption", type=int)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    run = next(run for run in active_runs(manifest) if run["name"] == args.run_name)
    if args.record_interruption is not None:
        record_interruption(manifest, run, args.segment, args.record_interruption)
    elif args.worker:
        if args.end_step is None or args.output is None:
            parser.error("Workers require --end-step and --output")
        from scaffolds.transformer.studies.key_dimension_taylor.preflight import verify_manifest
        verify_manifest(manifest)
        try:
            write_json(args.output, worker_result(run, args.worker, args.end_step))
        except BaseException as error:
            write_json(args.output, {"status": "failed", "error": f"{type(error).__name__}: {error}",
                                    "numerical_failure": isinstance(error, FloatingPointError)})
            raise
    else:
        run_segment(manifest, run, args.segment)


if __name__ == "__main__":
    main()
