"""Event-triggered capture of the exact pre-update state of post-warmup gradient spikes.

``train.gradient_event_capture`` enables it; the threshold and retention are recipe fields,
fixed before launch. After every update the trainer applies one rule to the pre-clip global
gradient norm it already fetches: update ``k`` is a *gradient event* when

- ``k > optim.warmup_steps`` (the update's learning rate is past warmup),
- its norm ``g`` is finite,
- at least ``gradient_event_min_history`` finite norms of earlier updates are known, and
- ``g > max(gradient_event_min_norm, gradient_event_factor * median(history))``,

where the history is the last ``gradient_event_history`` finite norms of earlier updates,
warmup updates and events included. It persists across requeues.

An event saves the state the update started from (parameters, Adam moments, and
``state.step == k - 1``) synchronously, in the checkpoint manager's ``{"state": state}``
item layout, at ``<checkpoint storage>/gradient_events/<k>/<k - 1>/default``, with the
event's record as ``<k>/event.json``. Each ``<k>`` is thus a one-step checkpoint-manager root
whose step directory names the state's own step, like every checkpoint of the run. At most
``gradient_event_max_states`` event states are kept: the run's first event and the largest
later events by norm. A larger later event replaces the smallest kept later one, whose
directory is deleted. Every event, kept or not, is appended to the run's
``gradient_events.jsonl``. An update whose loss, gradient norm, or updated parameters are
nonfinite also saves its pre-update state, outside the cap, before the run stops with its
numerical failure. Saving never changes the update: it reads the state the step left intact
(one-device steps do not donate their input; the data-parallel step does, so it is given a
device copy).

``state.json`` in the capture root holds, for every checkpoint a resume may restore, the
history and kept list as of that checkpoint (written before the checkpoint's save is
issued), and the current kept list for the run record. A resume restores its checkpoint's
snapshot and deletes the captures that snapshot does not reference: those made after that
checkpoint, whose updates the resumed run repeats, the capture of a superseded numerical
failure, and a capture replaced at or before that checkpoint that an earlier snapshot still
referenced. A replaced capture that a restorable snapshot still references is deleted once a
later checkpoint is durable. A resume also appends a ``resume`` row with its restored step
to the log: rows that earlier attempts logged for later updates are superseded, because the
resumed run repeats those updates.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import shutil
import time

import jax
import numpy as np

from scaffolds.transformer.training.checkpointing import (
    GRADIENT_EVENT_DIRECTORY,
    gradient_event_directory,
    save_state_now,
    state_sha256,
)
from scaffolds.transformer.training.diagnostics import attention_identity, json_number
from scaffolds.transformer.training.metrics import write_json_atomic


FORMAT = "shared-transformer-gradient-events"
VERSION = 1
LOG_NAME = "gradient_events.jsonl"
STATE_NAME = "state.json"
EVENT_NAME = "event.json"
SPIKE = "gradient_spike"
NONFINITE = "nonfinite_update"


def event_log_path(config) -> Path:
    """The run's append-only event log, a root JSONL hashed with the run's other raw outputs."""
    return Path(config.train.workdir) / LOG_NAME


@dataclass(frozen=True)
class Decision:
    update: int
    gradient_norm: float
    triggered: bool
    running_median: float | None
    threshold: float | None
    history_length: int


@dataclass(frozen=True)
class GradientEventRule:
    warmup_steps: int
    factor: float
    min_norm: float
    history: int
    min_history: int
    max_states: int

    @classmethod
    def from_config(cls, config) -> "GradientEventRule":
        train = config.train
        return cls(warmup_steps=config.optim.warmup_steps, factor=train.gradient_event_factor,
                   min_norm=train.gradient_event_min_norm, history=train.gradient_event_history,
                   min_history=train.gradient_event_min_history, max_states=train.gradient_event_max_states)

    def decide(self, update: int, gradient_norm: float, history: Sequence[float]) -> Decision:
        """The rule for update ``update`` given the finite norms of earlier updates, oldest first."""
        recent = list(history)[-self.history:]
        median = float(np.median(recent)) if recent else None
        threshold = (max(self.min_norm, self.factor * median)
                     if median is not None and len(recent) >= self.min_history else None)
        triggered = (update > self.warmup_steps and math.isfinite(gradient_norm)
                     and threshold is not None and gradient_norm > threshold)
        return Decision(update, gradient_norm, triggered, median, threshold, len(recent))

    def describe(self) -> dict:
        return asdict(self)


def select_for_retention(retained: Sequence[Mapping], gradient_norm: float, max_states: int):
    """Whether a new event is kept, and the kept event it replaces (None when a slot is free).

    ``retained`` lists the kept events in the order they were kept, so its first entry is
    the run's first event, which is never replaced. A full list keeps the largest later
    events: a new event replaces the smallest kept later event when its norm is larger
    (equal norms keep the earlier event).
    """
    if len(retained) < max_states:
        return True, None
    later = list(retained[1:])
    if not later:
        return False, None
    smallest = min(later, key=lambda entry: (entry["gradient_norm_pre_clip"], -entry["update"]))
    if gradient_norm > smallest["gradient_norm_pre_clip"]:
        return True, smallest
    return False, None


def _discard_reason(update: int, restored_step: int | None, failures: Collection[int]) -> str:
    """Why a resume deletes a capture that no snapshot it can restore references."""
    if restored_step is None:
        return "made by an earlier execution that left no checkpoint of this run"
    if update in failures:
        return "kept for a numerical failure that this retraining supersedes"
    if update > restored_step:
        return f"made after checkpoint step {restored_step}, from which the run repeats the update"
    return f"replaced at or before checkpoint step {restored_step}, whose state no longer keeps it"


def device_copy(state):
    """Every array of ``state`` copied on its own devices, for a step that donates its input."""
    return jax.tree.map(lambda leaf: leaf.copy() if isinstance(leaf, jax.Array) else leaf, state)


def justification(entry: Mapping) -> str:
    """Why the run record keeps a capture."""
    if entry["event"] == NONFINITE:
        return f"pre-update state of nonfinite update {entry['update']}, kept for its numerical failure"
    return (f"pre-update state of gradient event at update {entry['update']}, "
            f"norm {entry['gradient_norm_pre_clip']:.6g}")


def read_state(root: Path) -> dict | None:
    path = Path(root) / STATE_NAME
    if not path.is_file():
        return None
    state = json.loads(path.read_text())
    if state.get("format") != FORMAT or state.get("version") != VERSION:
        raise ValueError(f"Unsupported gradient-event state: {path}")
    return state


def _current(state: Mapping) -> list[dict]:
    return [*state["current"]["retained"], *state["current"]["failures"]]


def recorded_captures(config) -> list[dict] | None:
    """The kept captures as ``capture_artifacts`` diagnostic checkpoints; None when capture is off."""
    if not config.train.gradient_event_capture:
        return None
    root = gradient_event_directory(config)
    state = read_state(root)
    if state is None:
        raise ValueError(f"Gradient-event state is unavailable: {root / STATE_NAME}")
    return [{"path": root / entry["directory"], "justification": justification(entry),
             **{key: entry[key] for key in ("update", "pre_update_step", "event", "gradient_norm_pre_clip",
                                             "pre_update_state_sha256")}}
            for entry in _current(state)]


def prune_unrecorded_captures(config) -> None:
    """Delete everything in a completed run's capture root except its kept captures and state."""
    root = gradient_event_directory(config)
    if not root.is_dir():
        return
    state = read_state(root)
    keep = {entry["directory"] for entry in _current(state)} if state is not None else set()
    for path in root.iterdir():
        if path.name != STATE_NAME and path.name not in keep:
            _remove(path)


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _numbers(values: Mapping[str, float] | None) -> dict | None:
    return None if values is None else {key: json_number(value) for key, value in values.items()}


def _rng_derivation(layout) -> str:
    if layout.device_count > 1:
        microbatch = "microbatch j on device d uses fold_in(fold_in(step_rng, j), d)"
    elif layout.grad_accum_steps > 1:
        microbatch = "microbatch j uses jax.random.split(step_rng, grad_accum_steps)[j]"
    else:
        microbatch = "the update's one microbatch uses step_rng"
    return (f"step_rng = jax.random.fold_in(jax.random.PRNGKey(seed), update); {microbatch}; "
            "each forward's dropout stream is fold_in(that key, 0)")


def _execution_context(config, layout, device_kind: str) -> dict:
    """Facts shared by every event of one execution."""
    workdir = Path(config.train.workdir)
    record_path = workdir / "record.json"
    record = json.loads(record_path.read_text()) if record_path.is_file() else {}
    identity_path = workdir / "data_identity.json"
    return {
        "seed": config.train.seed,
        "rng_derivation": _rng_derivation(layout),
        "model_dropout": config.model.dropout,
        "microbatch_layout": {
            "global_batch_size": layout.global_batch_size, "grad_accum_steps": layout.grad_accum_steps,
            "global_micro_batch_size": layout.global_micro_batch_size, "device_count": layout.device_count,
            "local_micro_batch_size": layout.local_micro_batch_size,
            "order": ("microbatch j holds sequences [first + j * global_micro_batch_size, "
                      "first + (j + 1) * global_micro_batch_size); device d takes the d-th "
                      "local_micro_batch_size of them"),
        },
        "data_source": config.data.source,
        "data_identity": json.loads(identity_path.read_text()) if identity_path.is_file() else None,
        "device_kind": device_kind,
        "device_count": layout.device_count,
        "attention_backend": config.model.attention_backend,
        "attention_implementation": attention_identity(config).get("attention_implementation"),
        "source_commit": config.train.source_commit or None,
        "frozen_execution_sha256": record.get("frozen_execution_sha256"),
    }


class GradientEventCapture:
    """One execution's trigger history, kept captures, and event log (see the module docstring)."""

    def __init__(self, config, layout, *, attempt_id: str, device_kind: str):
        self.rule = GradientEventRule.from_config(config)
        self.root = gradient_event_directory(config)
        self.log_path = event_log_path(config)
        self.batch_size = layout.global_batch_size
        self.clip_norm = config.optim.grad_clip_norm
        self.attempt_id = attempt_id
        self.context = {"rule": self.rule.describe(), **_execution_context(config, layout, device_kind)}
        self.history: deque[float] = deque(maxlen=self.rule.history)
        self.retained: list[dict] = []
        self.failures: list[dict] = []
        # Checkpoint step -> history and kept list as of that checkpoint, for each checkpoint a
        # resume may restore: the latest durable one and one whose save is in flight.
        self.snapshots: dict[int, dict] = {}

    @classmethod
    def resume(cls, config, layout, restored_step: int | None, *, attempt_id: str,
               device_kind: str) -> "GradientEventCapture":
        """The trigger state as of the restored checkpoint; empty for a run without one."""
        capture = cls(config, layout, attempt_id=attempt_id, device_kind=device_kind)
        saved = read_state(capture.root)
        if restored_step is not None:
            snapshot = (saved or {}).get("snapshots", {}).get(str(restored_step))
            if snapshot is None:
                raise ValueError(f"No gradient-event state for checkpoint step {restored_step} in "
                                 f"{capture.root / STATE_NAME}; restore the capture root with the checkpoints")
            if saved["rule"] != capture.rule.describe():
                raise ValueError(f"Gradient-event rule differs from the run's: {capture.root / STATE_NAME}")
            missing = [entry["update"] for entry in snapshot["retained"]
                       if not (capture.root / entry["directory"] / str(entry["pre_update_step"])
                               / "default").is_dir()]
            if missing:
                raise ValueError(f"Kept gradient-event captures {missing} are missing from {capture.root}")
            capture.history.extend(snapshot["history"])
            capture.retained = [dict(entry) for entry in snapshot["retained"]]
            capture.snapshots = {restored_step: snapshot}
        if restored_step is not None or capture.log_path.exists():
            # The boundary for the log's readers: this execution repeats every update after restored_step.
            capture._log({"schema_version": VERSION, "event": "resume", "restored_step": restored_step,
                          "timestamp_utc": datetime.now(UTC).isoformat(), "attempt_id": attempt_id})
        capture._write_state()
        failures = {entry["update"] for entry in saved["current"]["failures"]} if saved is not None else set()
        capture._remove_unreferenced(reason=lambda update: _discard_reason(update, restored_step, failures))
        print(f"gradient_event_capture restored_step={restored_step} history={len(capture.history)} "
              f"retained={','.join(str(entry['update']) for entry in capture.retained) or 'none'}", flush=True)
        return capture

    def observe(self, update: int, gradient_norm: float | None) -> Decision | None:
        """Apply the rule to an update, then add its finite norm to the history; the decision if it triggers."""
        if gradient_norm is None:
            raise RuntimeError("Gradient-event capture requires every update's pre-clip gradient norm")
        decision = self.rule.decide(update, gradient_norm, self.history)
        if math.isfinite(gradient_norm):
            self.history.append(gradient_norm)
        return decision if decision.triggered else None

    def capture(self, decision: Decision, pre_update, *, loss: float, update_numerics, clip_scale: float,
                learning_rate: float, audit: Callable[[], Mapping], telemetry: Mapping | None = None,
                preceding_telemetry: Mapping | None = None) -> dict:
        """Log a gradient event and keep its pre-update state when the retention rule allows."""
        keep, replaced = select_for_retention(self.retained, decision.gradient_norm, self.rule.max_states)
        record = self._record(SPIKE, decision, pre_update, loss=loss, update_numerics=update_numerics,
                              clip_scale=clip_scale, learning_rate=learning_rate, audit=audit(),
                              telemetry=telemetry, preceding_telemetry=preceding_telemetry)
        record.update(retained=keep, replaces=replaced["update"] if replaced is not None else None)
        if keep:
            entry = self._save(decision.update, pre_update, record)
            if entry is None:
                record.update(retained=False, replaces=None)
            else:
                if replaced is not None:
                    self.retained.remove(replaced)
                self.retained.append(entry)
                # Persist the new list before deleting what only the old one referenced.
                self._write_state()
                self._remove_unreferenced()
        self._log(record)
        print(f"gradient_event update={decision.update} gradient_norm={decision.gradient_norm:.6f} "
              f"running_median={decision.running_median:.6f} threshold={decision.threshold:.6f} "
              f"retained={'yes' if record['retained'] else 'no'}"
              + (f" replaces={record['replaces']}" if record["replaces"] is not None else "")
              + (f" capture_seconds={record['capture_seconds']:.1f}" if "capture_seconds" in record else ""),
              flush=True)
        return record

    def capture_failure(self, update: int, pre_update, *, loss: float, update_numerics, learning_rate: float,
                        audit: Callable[[], Mapping], telemetry: Mapping | None = None,
                        preceding_telemetry: Mapping | None = None) -> None:
        """Keep a nonfinite update's pre-update state, outside the cap. Never raises: the
        numerical failure that follows must still be recorded."""
        try:
            numerics = self._numerics(loss, update_numerics)
            decision = self.rule.decide(update, numerics[1], self.history)
            record = self._record(NONFINITE, decision, pre_update, loss=loss, update_numerics=update_numerics,
                                  clip_scale=None, learning_rate=learning_rate, audit=audit(),
                                  telemetry=telemetry, preceding_telemetry=preceding_telemetry)
            record.update(retained=True, replaces=None)
            entry = self._save(update, pre_update, record)
            if entry is None:
                record["retained"] = False
            else:
                self.failures.append(entry)
                self._write_state()
            self._log(record)
            print(f"gradient_event_nonfinite update={update} retained={'yes' if record['retained'] else 'no'}",
                  flush=True)
        except Exception as error:
            print(f"gradient_event_capture_failed update={update} error={type(error).__name__}: {error}", flush=True)

    def before_checkpoint(self, step: int, manager) -> None:
        """Snapshot the trigger state for the checkpoint about to be saved at ``step``.

        Waiting for any save in flight first (the manager's own save would wait for it next)
        makes the latest checkpoint durable, so only it and ``step`` can be restored.
        """
        manager.wait_until_finished()
        durable = manager.latest_step()
        self.snapshots = {saved: snapshot for saved, snapshot in self.snapshots.items() if saved == durable}
        self.snapshots[step] = {"step": step, "history": list(self.history),
                                "retained": [dict(entry) for entry in self.retained]}
        self._write_state()
        self._remove_unreferenced()

    def checkpoint_durable(self, step: int) -> None:
        """``step``'s checkpoint is durable: a resume can restore no earlier snapshot."""
        if step in self.snapshots:
            self.snapshots = {step: self.snapshots[step]}
        self._write_state()
        self._remove_unreferenced()

    def _numerics(self, loss: float, update_numerics) -> list[float]:
        if update_numerics is None:
            return [float(loss), math.nan, math.nan]
        return np.asarray(jax.device_get(update_numerics), dtype=np.float64).tolist()

    def _record(self, event: str, decision: Decision, pre_update, *, loss, update_numerics, clip_scale,
                learning_rate, audit: Mapping, telemetry, preceding_telemetry) -> dict:
        update = decision.update
        if audit["step"] != update:
            raise ValueError(f"Batch audit of update {audit['step']} recorded for update {update}")
        numerics = self._numerics(loss, update_numerics)
        try:
            state_hash, hash_error = state_sha256(pre_update), None
        except FloatingPointError as error:
            state_hash, hash_error = None, str(error)
        first = (update - 1) * self.batch_size
        record = {
            "schema_version": VERSION, "event": event, "update": update, "pre_update_step": update - 1,
            "timestamp_utc": datetime.now(UTC).isoformat(), "attempt_id": self.attempt_id,
            "loss": json_number(loss), "gradient_norm_pre_clip": json_number(numerics[1]),
            "updated_parameter_norm": json_number(numerics[2]),
            "gradient_clip_norm": self.clip_norm, "gradient_clip_scale": clip_scale,
            "running_median": decision.running_median, "threshold": decision.threshold,
            "history_length": decision.history_length, "learning_rate": learning_rate,
            "sequences": {"first": first, "end": first + self.batch_size},
            "batch_sha256": audit["batch_sha256"], "rng_sha256": audit["rng_sha256"],
            "pre_update_state_sha256": state_hash,
            # Relative to the run's checkpoint storage; the state is in its step directory, pre_update_step.
            "capture_directory": f"{GRADIENT_EVENT_DIRECTORY}/{update}",
            "telemetry": _numbers(telemetry), "preceding_telemetry": _numbers(preceding_telemetry),
            **self.context,
        }
        if hash_error is not None:
            record["pre_update_state_sha256_error"] = hash_error
        return record

    def _save(self, update: int, state, record: dict) -> dict | None:
        """Write the state and its record; the kept-list entry, or None when saving failed."""
        directory = self.root / str(update)
        try:
            if directory.exists():
                _remove(directory)
            start = time.perf_counter()
            save_state_now(directory / str(update - 1), state)
            record["capture_seconds"] = time.perf_counter() - start
            write_json_atomic(directory / EVENT_NAME, record)
        except Exception as error:
            # A capture is a diagnostic: its failure is recorded, and training continues.
            record.pop("capture_seconds", None)
            record["capture_error"] = f"{type(error).__name__}: {error}"
            print(f"gradient_event_capture_failed update={update} error={record['capture_error']}", flush=True)
            shutil.rmtree(directory, ignore_errors=True)
            return None
        return {"update": update, "pre_update_step": update - 1, "event": record["event"],
                "gradient_norm_pre_clip": record["gradient_norm_pre_clip"],
                "pre_update_state_sha256": record["pre_update_state_sha256"], "directory": str(update)}

    def _referenced(self) -> set[str]:
        names = {entry["directory"] for entry in (*self.retained, *self.failures)}
        for snapshot in self.snapshots.values():
            names.update(entry["directory"] for entry in snapshot["retained"])
        return names

    def _remove_unreferenced(self, reason: Callable[[int], str] | None = None) -> None:
        if not self.root.is_dir():
            return
        referenced = self._referenced()
        for path in sorted(self.root.iterdir()):
            if path.name == STATE_NAME or path.name in referenced:
                continue
            if reason is not None and path.name.isdigit():
                self._log({"schema_version": VERSION, "event": "capture_discarded", "update": int(path.name),
                           "timestamp_utc": datetime.now(UTC).isoformat(), "attempt_id": self.attempt_id,
                           "reason": reason(int(path.name))})
                print(f"gradient_event_capture_discarded update={path.name}", flush=True)
            _remove(path)

    def _write_state(self) -> None:
        write_json_atomic(self.root / STATE_NAME, {
            "format": FORMAT, "version": VERSION, "rule": self.rule.describe(),
            "snapshots": {str(step): snapshot for step, snapshot in sorted(self.snapshots.items())},
            "current": {"retained": self.retained, "failures": self.failures},
        })

    def _log(self, record: Mapping) -> None:
        line = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
