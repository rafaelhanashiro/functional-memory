"""Qualify one recipe's full-shape training on one GPU through acceptance's own pilot worker.

Production acceptance (``scaffolds/transformer/acceptance.py``) trains a disposable pilot of the
frozen recipe, projects the full run from its timing, and resumes it in a fresh process. This tool
runs those workers on a qualification copy of a recipe (its workdir and checkpoints under
``--directory``, W&B disabled, every other field unchanged and checked by the adapter) and records
what acceptance does not keep:

``pilot``   acceptance's pilot worker (the recipe's model, batch, accumulation, data, seed and
            schedule; validation and a checkpoint at its stop), with every update's exact loss and
            update numerics (loss, pre-clip gradient norm, updated parameter norm) and the allocator's
            peak; acceptance's checks of the pilot, its runtime projection, and its two-hour-reserve
            gate for each ``--limit-hours``. The same process then trains the uninterrupted reference,
            a fresh run through the pilot's stop and ``--continue-updates`` further updates, and
            continues the pilot's checkpoint for those updates (a restored state compiles its own
            update: its step is strongly typed and its arrays carry shardings). The pilot's directory
            is restored to its stop afterwards.
``resume``  acceptance's resume worker in a fresh process from the pilot's checkpoint, through the
            same updates: acceptance's audit (restored state, next batch and RNG), then every update's
            loss and numerics and the final state compared bitwise with the uninterrupted run and the
            in-process continuation. ``--tag`` keeps resumes apart (for example one with an empty
            compilation cache); the pilot's directory is restored to its stop after each.
``parity``  instrumentation arms from initialization in one process, so that every arm runs the
            same compiled update: gradient-event capture off; capture as the recipe sets it; capture
            forced to fire on every update from the second (min_norm 0 and min_history 1 in a copy
            whose other fields are the recipe's, with the capture rule's warmup gate and factor set to
            zero in the capture code alone, so the optimizer's warmup is untouched); and validation
            with its attention diagnostics every ``--diagnostic-interval`` updates. Every update's loss,
            numerics and resulting state hash must equal the capture-off arm's. Each event's recorded
            pre-update state hash must equal the capture-off state before that update, and each kept
            capture must restore to exactly that state.

Modes may be chained in one process (``pilot,parity``), reusing the pilot's compiled update.

    python -m scaffolds.transformer.benchmarks.qualify_training pilot,parity RECIPE \\
        --adapter experiments.transformer_pp.score_bias_acceptance --directory DIR
    python -m scaffolds.transformer.benchmarks.qualify_training resume RECIPE \\
        --adapter experiments.transformer_pp.score_bias_acceptance --directory DIR
"""
from __future__ import annotations

import argparse
from argparse import Namespace
import contextlib
import copy
import dataclasses
from datetime import UTC, datetime
import importlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import statistics
import time

PILOT = "pilot"
AT_STOP = "pilot.at-stop"
ARMS = ("capture_off", "capture_recipe", "capture_forced", "diagnostics")
EXECUTION_FIELDS = {"train.workdir", "train.checkpoint_dir", "train.wandb_mode"}


def _bits(value) -> str:
    import numpy as np

    return np.asarray(value).tobytes().hex()


def _memory() -> dict | None:
    import jax

    stats = jax.local_devices()[0].memory_stats()
    if stats is None:
        return None
    return {key: stats.get(key) for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit", "largest_alloc_size")}


def _write(path: Path, value) -> None:
    from scaffolds.transformer.benchmarks.benchmark_attention import write_report

    write_report(path, value)


class CompileCacheLog(logging.Handler):
    """Persistent compilation cache hits and misses by module. Every ``run_training`` call compiles its
    update again (the optimizer is a static field of the state), so separate calls share one executable
    only through the persistent cache, which production launchers also enable."""

    PREFIXES = {"Persistent compilation cache hit for": "hit", "PERSISTENT COMPILATION CACHE MISS for": "miss"}

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.events = []
        self.phase = None

    def emit(self, record):
        message = record.getMessage()
        for prefix, kind in self.PREFIXES.items():
            if message.startswith(prefix):
                module = message.split("'")[1]
                self.events.append({"phase": self.phase, "module": module, "cache": kind})
                if "train_step" in module:
                    print(f"compile_cache {kind} module={module} phase={self.phase}", flush=True)

    def install(self):
        logger = logging.getLogger("jax._src.compiler")
        # Record the cache's DEBUG lines without printing every other compiler message.
        logger.propagate = False
        console = logging.StreamHandler()
        console.setLevel(logging.WARNING)
        logger.addHandler(console)
        logger.addHandler(self)
        logger.setLevel(logging.DEBUG)
        return self

    def summary(self, phase=None) -> dict:
        events = [event for event in self.events if phase is None or event["phase"] == phase]
        steps = [event for event in events if "train_step" in event["module"]]
        return {"hits": sum(event["cache"] == "hit" for event in events),
                "misses": sum(event["cache"] == "miss" for event in events), "train_step": steps}


CACHE_LOG = CompileCacheLog()


class StepRecorder:
    """Wrap the trainer's jitted updates. Each row holds the update's loss and update numerics as exact
    bits, the allocator's statistics before it was dispatched (the previous update has completed: the
    trainer reads every loss), and, with ``hash_states``, the SHA256 of the state it returned."""

    NAMES = ("train_step", "accumulated_train_step")

    def __init__(self, *, hash_states: bool = False):
        self.hash_states = hash_states
        self.rows = []

    def __enter__(self):
        from scaffolds.transformer.training import loop

        self._loop = loop
        self._originals = {name: getattr(loop, name) for name in self.NAMES}
        for name, function in self._originals.items():
            setattr(loop, name, self._wrap(function))
        return self

    def __exit__(self, *exc):
        for name, function in self._originals.items():
            setattr(self._loop, name, function)

    def _wrap(self, function):
        from scaffolds.transformer.training.checkpointing import state_sha256

        def wrapped(state, *args, **kwargs):
            row = {"memory_before": _memory()}
            outputs = function(state, *args, **kwargs)
            row["_step"], row["_loss"], row["_numerics"] = outputs[0].step, outputs[1], outputs[2]
            if self.hash_states:
                row["state_sha256"] = state_sha256(outputs[0])
            self.rows.append(row)
            return outputs
        return wrapped

    def finish(self) -> list[dict]:
        import numpy as np

        rows = []
        for row in self.rows:
            loss, numerics = np.asarray(row.pop("_loss")), np.asarray(row.pop("_numerics"))
            rows.append({"update": int(row.pop("_step")), "loss": float(loss), "loss_bits": _bits(loss),
                         "numerics": [float(value) for value in numerics.astype(np.float64)],
                         "numerics_dtype": str(numerics.dtype), "numerics_bits": _bits(numerics), **row})
        self.rows = []
        return rows


def compare_updates(reference: list[dict], other: list[dict], *, states: bool = False) -> dict:
    """Bitwise agreement of matching updates' losses and numerics (and resulting state hashes)."""
    keys = ("loss_bits", "numerics_bits", *(("state_sha256",) if states else ()))
    by_update = {row["update"]: row for row in reference}
    differences = []
    for row in other:
        expected = by_update.get(row["update"])
        if expected is None:
            differences.append({"update": row["update"], "missing_in_reference": True})
            continue
        for key in keys:
            if row[key] != expected[key]:
                differences.append({"update": row["update"], "field": key, "reference": expected.get(key),
                                    "value": row.get(key),
                                    **({"loss_difference": row["loss"] - expected["loss"],
                                        "numerics_difference": [a - b for a, b in zip(row["numerics"],
                                                                                      expected["numerics"])]}
                                       if key != "state_sha256" else {})})
    return {"updates": [row["update"] for row in other], "fields": list(keys), "identical": not differences,
            "differences": differences}


def provenance(experiment: str | None, allow_dirty: bool) -> dict:
    import jax
    from scaffolds.transformer.benchmarks import benchmark_training_throughput as throughput
    from scaffolds.transformer.benchmarks.benchmark_attention import provenance as source_provenance

    device = jax.local_devices()[0]
    if allow_dirty or experiment is None:
        source = source_provenance(device)
        source["frozen_execution_checked"] = False
    else:
        # Refuses unless every frozen execution file (scaffolds, the study's recipes and modules) is committed.
        source = throughput._execution_provenance(device, experiment)
        source["frozen_execution_checked"] = True
    source.update(node=os.environ.get("SLURMD_NODENAME"), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                  environment={name: os.environ.get(name) for name in (
                      "JAX_PLATFORMS", "JAX_COMPILATION_CACHE_DIR", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_FLAGS",
                      "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "TRANSFORMER_HOST_STAGING", "OMP_NUM_THREADS",
                      "SLURM_CPUS_PER_TASK", "TMPDIR")})
    return source


def qualification_copy(recipe: Path, directory: Path) -> tuple[Path, dict]:
    """The recipe with its workdir and checkpoints under ``directory`` and W&B disabled; nothing else differs."""
    from scaffolds.transformer.configs import config_to_dict, load_config, save_config

    config = load_config(recipe)
    original = config_to_dict(config)
    config.train.workdir = directory / "run"
    config.train.checkpoint_dir = directory / "run" / "checkpoints"
    config.train.wandb_mode = "disabled"
    path = directory / "recipe.yaml"
    if path.exists():
        if config_to_dict(load_config(path)) != config_to_dict(config):
            raise ValueError(f"{path} is not this recipe's qualification copy")
    else:
        save_config(config, path)
    copied = config_to_dict(load_config(path))
    changed = sorted(f"{section}.{key}" for section in original for key in original[section]
                     if original[section][key] != copied[section].get(key))
    if not set(changed) <= EXECUTION_FIELDS:
        raise ValueError(f"The qualification copy changes more than its execution fields: {changed}")
    return path, {"source": str(recipe), "source_sha256": _sha256(recipe), "copy": str(path),
                  "copy_sha256": _sha256(path), "changed_fields": changed}


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def acceptance_context(adapter: str | None, copy_path: Path, gpu_type: str):
    """The adapter's acceptance spec and sealed experiment identity, keyed to this process's devices."""
    import jax
    from scaffolds.transformer import acceptance
    from scaffolds.transformer.configs import load_config

    devices = jax.local_devices()
    if len(devices) != 1:
        raise RuntimeError(f"Qualify on exactly one device; got {devices}")
    if devices[0].platform == "gpu" and gpu_type.upper() not in devices[0].device_kind.upper():
        raise RuntimeError(f"Expected one {gpu_type} GPU; got {devices[0].device_kind}")
    if adapter:
        module = importlib.import_module(adapter)
        spec = module.experiment_spec(load_config(copy_path), entry_module=module.__name__)
    else:
        spec = acceptance.attention_spec(load_config(copy_path), entry_module=__name__)
    experiment = acceptance.identity(spec, copy_path, 1, gpu_type)  # runs the adapter's recipe check
    return spec, acceptance.on_devices(experiment, [device.device_kind for device in devices])


def _timing_summary(timing: dict) -> dict:
    steps = timing.get("train_step_seconds") or []
    warm = steps[1:]
    return {"first_update_seconds": steps[0] if steps else None, "warm_updates": len(warm),
            "median_seconds_per_update": statistics.median(warm) if warm else None,
            "min_seconds_per_update": min(warm) if warm else None,
            "max_seconds_per_update": max(warm) if warm else None,
            "runtime_seconds": timing.get("runtime_seconds")}


def _worker_args(args, directory: Path, stop: int) -> Namespace:
    return Namespace(worker="pilot", config=args.copy, directory=directory, stop_after_step=stop,
                     gpu_count=1, gpu_type=args.gpu_type)


def _replace_directory(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    os.rename(source, destination)


def run_pilot(args, spec, experiment) -> dict:
    """Acceptance's pilot stage with its projection and budget gate, the uninterrupted run, and the
    in-process continuation of the pilot's checkpoint."""
    from scaffolds.transformer import acceptance
    from scaffolds.transformer.configs import load_config

    root = args.directory / "acceptance"
    if (root / PILOT).exists():
        raise ValueError(f"{root / PILOT} exists; qualify in a fresh directory")
    start = acceptance.continuation_start_step(experiment["config"])
    stop = start + spec.pilot_steps
    batch_size = experiment["config"]["data"]["batch_size"]
    report = {"pilot_start": start, "pilot_stop": stop, "memory_at_start": _memory()}
    print(f"qualify pilot updates {start + 1}-{stop}", flush=True)
    CACHE_LOG.phase = "pilot"
    with StepRecorder() as recorder:
        started = time.perf_counter()
        outcome = acceptance._worker(spec, _worker_args(args, root, stop), experiment)
        report["pilot_wall_seconds"] = time.perf_counter() - started
        rows = recorder.finish()
    report["memory_after_pilot"] = _memory()
    acceptance.write_json(root / "pilot-result.json", outcome)
    # Acceptance's own checks of the pilot (acceptance._run).
    if (outcome["final_step"] != stop or outcome["data_parallel_devices"] != 1
            or outcome.get("start_step", 0) != start):
        raise RuntimeError(f"Incomplete full-recipe pilot: {outcome}")
    if spec.required_backend is not None and outcome["attention_backend"] != spec.required_backend:
        raise RuntimeError(f"Production requires backend {spec.required_backend!r}; "
                           f"resolved {outcome['attention_backend']!r}")
    metrics = [json.loads(line) for line in (root / PILOT / "metrics.jsonl").read_text().splitlines()]
    validation = next(row for row in metrics if row["step"] == stop)
    if not math.isfinite(validation["validation_nll"]) or not validation["attention_diagnostics"]["valid"]:
        raise RuntimeError("Pilot validation or attention diagnostics failed")
    projection = acceptance.project_runtime(outcome["timing"], load_config(args.recipe))
    acceptance.write_json(root / "runtime-projection.json", projection)
    budgets = {}
    for hours in args.limit_hours:
        try:
            acceptance.require_allocation_budget(projection["projected_seconds"], int(round(hours * 3600)))
            budgets[f"{hours:g}h"] = {"passes": True}
        except RuntimeError as error:
            budgets[f"{hours:g}h"] = {"passes": False, "error": str(error)}
    finite = all(math.isfinite(value) for row in rows for value in (row["loss"], *row["numerics"]))
    peaks = [row["memory_before"]["peak_bytes_in_use"] for row in rows if row["memory_before"]]
    report.update(
        outcome=outcome, updates=rows, all_losses_and_numerics_finite=finite,
        validation={key: validation[key] for key in ("step", "validation_nll", "validation_perplexity")},
        attention_diagnostics_valid=validation["attention_diagnostics"]["valid"],
        attention_backend=outcome["attention_backend"],
        attention_implementation=outcome.get("attention_implementation"),
        timing_summary=_timing_summary(outcome["timing"]), runtime_projection=projection,
        projected_hours=projection["projected_seconds"] / 3600, budget_gate=budgets,
        # Before update k the allocator's peak covers updates 1..k-1; the pilot's peak adds validation,
        # its checkpoint, and the audit's restored copy of the state.
        training_peak_bytes=max(peaks) if peaks else None,
        training_peak_note="allocator peak_bytes_in_use read before each update, so through the pilot's "
                           "last update but one",
    )
    if not finite:
        raise FloatingPointError("Nonfinite pilot loss or update numerics")
    end = stop + args.continue_updates
    # The uninterrupted reference: a fresh run through the pilot's validation and checkpoint and on, in
    # this process, so its update is the pilot's program for a freshly initialized state.
    print(f"qualify uninterrupted run updates {start + 1}-{end}", flush=True)
    CACHE_LOG.phase = "uninterrupted"
    with StepRecorder() as recorder:
        straight = acceptance._worker(spec, _worker_args(args, args.directory / "uninterrupted", end), experiment)
        uninterrupted = recorder.finish()
    if straight["final_step"] != end or straight["restored_step"] is not None:
        raise RuntimeError(f"Incomplete uninterrupted run: {straight}")
    # The in-process continuation: the pilot's checkpoint restored in this process. A restored state's
    # update compiles separately from a fresh state's (its step is strongly typed and its arrays carry
    # shardings), which this separates from a fresh process. The pilot's directory returns to its stop.
    shutil.copytree(root / PILOT, root / AT_STOP, symlinks=True)
    print(f"qualify continuation updates {stop + 1}-{end} from the pilot's checkpoint in its process", flush=True)
    CACHE_LOG.phase = "continuation"
    with StepRecorder() as recorder:
        continued = acceptance._worker(spec, _worker_args(args, root, end), experiment)
        continuation = recorder.finish()
    if (continued["restored_step"] != stop or continued["final_step"] != end
            or continued["data_start_sequence"] != stop * batch_size):
        raise RuntimeError(f"Continuation did not start from the pilot's checkpoint: {continued}")
    acceptance.validate_resume_audit(outcome, continued)
    _replace_directory(root / PILOT, root / "pilot.continued")
    os.rename(root / AT_STOP, root / PILOT)
    report.update(
        continuation_end=end, uninterrupted_updates=uninterrupted, uninterrupted_audit=straight["checkpoint_audit"],
        uninterrupted_validation_nll=_validation_nll(args.directory / "uninterrupted" / PILOT, stop),
        continuation_outcome=continued, continuation_updates=continuation,
        continuation_audit=continued["checkpoint_audit"], memory_after_continuation=_memory(),
        compared={
            # The same program on the same inputs twice in one process.
            "pilot_vs_uninterrupted": compare_updates(uninterrupted, rows),
            "pilot_validation_nll_equal": report["validation"]["validation_nll"] == _validation_nll(
                args.directory / "uninterrupted" / PILOT, stop),
            # The restored state's program against the fresh state's, after the checkpoint.
            "continuation_vs_uninterrupted": {
                **compare_updates(uninterrupted, continuation),
                "final_state_equal": continued["checkpoint_audit"]["final_state_sha256"]
                                     == straight["checkpoint_audit"]["final_state_sha256"]},
        },
        compile_cache={phase: CACHE_LOG.summary(phase) for phase in ("pilot", "uninterrupted", "continuation")})
    return report


def _validation_nll(workdir: Path, step: int) -> float | None:
    rows = [json.loads(line) for line in (workdir / "metrics.jsonl").read_text().splitlines()]
    return next((row["validation_nll"] for row in rows if row["step"] == step), None)


def run_resume(args, spec, experiment) -> dict:
    """Acceptance's resume stage in this fresh process, compared with both references."""
    from scaffolds.transformer import acceptance

    root = args.directory / "acceptance"
    pilot = json.loads((args.directory / "pilot.json").read_text())["pilot"]
    stop, end = pilot["pilot_stop"], pilot["continuation_end"]
    batch_size = experiment["config"]["data"]["batch_size"]
    pilot_outcome = json.loads((root / "pilot-result.json").read_text())
    shutil.copytree(root / PILOT, root / AT_STOP, symlinks=True)
    print(f"qualify resume ({args.tag}) updates {stop + 1}-{end} in a fresh process", flush=True)
    CACHE_LOG.phase = f"resume-{args.tag}"
    with StepRecorder() as recorder:
        resumed = acceptance._worker(spec, _worker_args(args, root, end), experiment)
        rows = recorder.finish()
    report = {"tag": args.tag, "outcome": resumed, "updates": rows, "memory_after": _memory(),
              "timing_summary": _timing_summary(resumed["timing"]),
              "compile_cache": CACHE_LOG.summary(f"resume-{args.tag}")}
    # Acceptance's resume checks (acceptance._run): restored at the pilot's stop, data position, audit.
    if (resumed["restored_step"] != stop or resumed["final_step"] != end
            or resumed["data_start_sequence"] != stop * batch_size):
        raise RuntimeError(f"Pilot resume did not preserve optimizer and data position: {resumed}")
    acceptance.validate_resume_audit(pilot_outcome, resumed)
    report["acceptance_resume_audit"] = "passed"
    audit = resumed["checkpoint_audit"]
    report["compared"] = {}
    for name in ("uninterrupted", "continuation"):
        reference = pilot[f"{name}_audit"]
        report["compared"][f"resume_vs_{name}"] = {
            **compare_updates(pilot[f"{name}_updates"], rows),
            "final_state_equal": audit["final_state_sha256"] == reference["final_state_sha256"],
            "final_state_sha256": {name: reference["final_state_sha256"], "resumed": audit["final_state_sha256"]},
        }
    reference = pilot["continuation_audit"]
    report["compared"]["start_state_equal"] = audit["start_state_sha256"] == reference["start_state_sha256"]
    report["compared"]["first_update_batch_and_rng_equal"] = audit["first_update"] == reference["first_update"]
    _replace_directory(root / PILOT, root / f"pilot.resumed-{args.tag}")
    os.rename(root / AT_STOP, root / PILOT)
    return report


def arm_config(base, arm: str, workdir: Path, source_commit: str, interval: int):
    """An arm's copy of the qualification recipe; only execution, validation cadence and capture differ."""
    from scaffolds.transformer.configs.base import validate_config

    config = copy.deepcopy(base)
    config.train.workdir, config.train.checkpoint_dir, config.train.wandb_mode = workdir, None, "disabled"
    config.train.source_commit = source_commit
    # As acceptance's pilot worker sets them: no early cadence, one validation batch.
    config.train.eval_early_every, config.train.eval_early_until, config.train.eval_batches = None, 0, 1
    config.train.eval_every = interval if arm == "diagnostics" else config.optim.total_steps
    if arm in ("capture_off", "diagnostics"):
        config.train = dataclasses.replace(config.train, gradient_event_capture=False)
    elif arm == "capture_forced":
        config.train = dataclasses.replace(config.train, gradient_event_capture=True, gradient_event_factor=1.0001,
                                           gradient_event_min_norm=0.0, gradient_event_min_history=1)
    elif arm != "capture_recipe":
        raise ValueError(arm)
    validate_config(config)
    return config


@contextlib.contextmanager
def forced_rule(enabled: bool):
    """Fire the capture rule on every update with a finite norm and one earlier norm: its warmup gate and
    factor are zero in the capture code alone (the recipe's optimizer warmup is unchanged)."""
    from scaffolds.transformer.training.gradient_events import GradientEventRule

    if not enabled:
        yield
        return
    original = GradientEventRule.__dict__["from_config"]

    def from_config(cls, config):
        return dataclasses.replace(original.__func__(cls, config), warmup_steps=0, factor=0.0)

    GradientEventRule.from_config = classmethod(from_config)
    try:
        yield
    finally:
        GradientEventRule.from_config = original


def run_parity(args, spec, experiment, pilot_updates=None) -> dict:
    """Instrumentation arms in one process; every arm's updates must equal the capture-off arm's."""
    import jax
    from scaffolds.transformer.configs import load_config
    from scaffolds.transformer.training import loop
    from scaffolds.transformer.training.checkpointing import gradient_event_directory, restore_state, state_sha256
    from scaffolds.transformer.training.gradient_events import read_state
    from scaffolds.transformer.training.state import create_state

    base = load_config(args.copy)
    updates = args.parity_updates
    root = args.directory / "parity"
    arms, configs = {}, {}
    for arm in ARMS:
        workdir = root / arm
        if workdir.exists():
            raise ValueError(f"{workdir} exists; qualify in a fresh directory")
        config = configs[arm] = arm_config(base, arm, workdir, experiment["source_commit"], args.diagnostic_interval)
        print(f"qualify parity arm {arm}: updates 1-{updates}", flush=True)
        CACHE_LOG.phase = f"parity-{arm}"
        with StepRecorder(hash_states=True) as recorder, forced_rule(arm == "capture_forced"):
            started = time.perf_counter()
            outcome = loop.run_training(config, resume=True, stop_after_step=updates, audit_checkpoint=True)
            wall = time.perf_counter() - started
            rows = recorder.finish()
        result = {"workdir": str(workdir), "wall_seconds": wall, "updates": rows,
                  "attention_backend": outcome["attention_backend"],
                  "final_state_sha256": outcome["checkpoint_audit"]["final_state_sha256"],
                  "checkpoint_state_sha256": outcome["checkpoint_audit"]["checkpoint_state_sha256"],
                  "compile_cache": CACHE_LOG.summary(f"parity-{arm}")}
        metrics_path = workdir / "metrics.jsonl"
        if metrics_path.exists():
            result["validations"] = [
                {"step": row["step"], "validation_nll": row["validation_nll"],
                 "attention_diagnostics_valid": row["attention_diagnostics"]["valid"],
                 "score_biases_reported": "score_biases" in (row["attention_diagnostics"].get("coefficients") or {})}
                for row in map(json.loads, metrics_path.read_text().splitlines())]
        events_path = workdir / "gradient_events.jsonl"
        if events_path.exists():
            result["events"] = [json.loads(line) for line in events_path.read_text().splitlines()]
        arms[arm] = result
    reference = arms["capture_off"]
    after = {row["update"]: row["state_sha256"] for row in reference["updates"]}
    checks = {}
    for arm in ARMS[1:]:
        check = compare_updates(reference["updates"], arms[arm]["updates"], states=True)
        check["final_state_equal"] = arms[arm]["final_state_sha256"] == reference["final_state_sha256"]
        checks[arm] = check
    spikes = [row for row in arms["capture_forced"].get("events", []) if row["event"] == "gradient_spike"]
    checks["capture_forced"]["events"] = [
        {"update": row["update"], "retained": row["retained"], "gradient_norm_pre_clip": row["gradient_norm_pre_clip"],
         "pre_update_state_matches_capture_off": row["pre_update_state_sha256"] == after.get(row["update"] - 1)}
        for row in spikes]
    if not spikes:
        raise RuntimeError("The forced capture arm logged no gradient events")
    if any(row["event"] == "gradient_spike" for row in arms["capture_recipe"].get("events", [])):
        raise RuntimeError("The recipe's capture rule fired during warmup")
    # Restore every kept capture into the training state's structure and hash it.
    template = create_state(configs["capture_forced"], jax.random.PRNGKey(base.train.seed))
    capture_root = gradient_event_directory(configs["capture_forced"])
    restored = []
    for entry in read_state(capture_root)["current"]["retained"]:
        state = restore_state(capture_root / entry["directory"] / str(entry["pre_update_step"]), template)
        digest = state_sha256(state)
        del state
        restored.append({"update": entry["update"], "pre_update_step": entry["pre_update_step"],
                         "restored_state_sha256": digest,
                         "equals_recorded": digest == entry["pre_update_state_sha256"],
                         "equals_capture_off_state_before_update": digest == after.get(entry["update"] - 1)})
    del template
    checks["capture_forced"]["restored_captures"] = restored
    if pilot_updates is not None:
        # The pilot ran the recipe as acceptance does (capture as the recipe sets it, validation at its stop).
        checks["pilot"] = compare_updates(reference["updates"],
                                          [row for row in pilot_updates if row["update"] <= updates])
    passed = (all(check["identical"] and check["final_state_equal"] for arm, check in checks.items() if arm != "pilot")
              and all(event["pre_update_state_matches_capture_off"] for event in checks["capture_forced"]["events"])
              and bool(restored) and all(item["equals_recorded"] and item["equals_capture_off_state_before_update"]
                                         for item in restored)
              and all(row["attention_diagnostics_valid"] for row in arms["diagnostics"].get("validations", []))
              and len(arms["diagnostics"].get("validations", [])) == updates // args.diagnostic_interval
              and checks.get("pilot", {"identical": True})["identical"])
    return {"updates": updates, "diagnostic_interval": args.diagnostic_interval,
            "forced_rule": "warmup_steps 0 and factor 0 in GradientEventRule (capture code only); copy sets "
                           "gradient_event_factor 1.0001, min_norm 0, min_history 1",
            "arms": arms, "checks": checks, "passed": passed}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("modes", help="comma-separated: pilot, resume, parity")
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--adapter", help="acceptance adapter module (its experiment_spec); default: attention_spec")
    parser.add_argument("--experiment", help="study name for the frozen-execution check (e.g. transformer_pp)")
    parser.add_argument("--allow-dirty", action="store_true", help="skip the frozen-execution check (CPU smoke tests)")
    parser.add_argument("--gpu-type", default="H100")
    parser.add_argument("--continue-updates", type=int, default=2)
    parser.add_argument("--parity-updates", type=int, default=10)
    parser.add_argument("--diagnostic-interval", type=int, default=2)
    parser.add_argument("--limit-hours", type=float, nargs="+", default=(24, 48, 72))
    parser.add_argument("--tag", default="warm-cache", help="resume: names this resume's kept directory and report")
    args = parser.parse_args(argv)
    modes = args.modes.split(",")
    if not set(modes) <= {"pilot", "resume", "parity"} or ("resume" in modes and len(modes) > 1):
        parser.error("modes: pilot and/or parity in one process, or resume alone")
    CACHE_LOG.install()
    args.recipe, args.directory = args.recipe.resolve(), args.directory.resolve()
    args.directory.mkdir(parents=True, exist_ok=True)
    args.copy, copy_facts = qualification_copy(args.recipe, args.directory)
    spec, experiment = acceptance_context(args.adapter, args.copy, args.gpu_type)
    common = {"format": "training-qualification", "version": 1, "recipe": copy_facts,
              "adapter": args.adapter, "provenance": provenance(args.experiment, args.allow_dirty),
              "experiment_identity": experiment["identity"], "device_kinds": experiment["device_kinds"],
              "started_utc": datetime.now(UTC).isoformat()}
    if "resume" in modes:
        report = {**common, "resume": run_resume(args, spec, experiment)}
        report["finished_utc"] = datetime.now(UTC).isoformat()
        _write(args.directory / f"resume-{args.tag}.json", report)
        print(json.dumps(report["resume"]["compared"], indent=1), flush=True)
        return
    pilot = None
    if "pilot" in modes:
        pilot = run_pilot(args, spec, experiment)
        report = {**common, "pilot": pilot, "finished_utc": datetime.now(UTC).isoformat()}
        _write(args.directory / "pilot.json", report)
        print(json.dumps({key: pilot[key] for key in ("timing_summary", "projected_hours", "budget_gate",
                                                      "training_peak_bytes", "attention_backend", "compared")},
                         indent=1), flush=True)
    if "parity" in modes:
        parity = run_parity(args, spec, experiment, pilot_updates=pilot["updates"] if pilot else None)
        report = {**common, "parity": parity, "finished_utc": datetime.now(UTC).isoformat()}
        _write(args.directory / "parity.json", report)
        print(json.dumps({"passed": parity["passed"], "checks": parity["checks"]}, indent=1), flush=True)
        if not parity["passed"]:
            raise SystemExit("Instrumentation parity failed")


if __name__ == "__main__":
    main()
