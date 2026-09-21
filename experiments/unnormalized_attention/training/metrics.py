from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable


METRICS_SCHEMA_VERSION = 1


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _unlink_durable(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _fsync_directory(path.parent)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Replace a JSON file atomically and durably on the local filesystem."""
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _write_text_atomic(path, text)


class ValidationMetricsStore:
    """Append-only validation metrics with checkpoint-aware crash recovery."""

    def __init__(self, workdir: Path):
        self.metrics_path = workdir / "metrics.jsonl"
        self.pending_path = workdir / "pending_validation.json"
        self._event_ids = self._load_event_ids()

    def _load_event_ids(self) -> set[str]:
        if not self.metrics_path.exists():
            return set()
        event_ids = set()
        with self.metrics_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Malformed metrics JSONL at {self.metrics_path}:{line_number}"
                    ) from error
                event_id = record.get("event_id")
                if not isinstance(event_id, str):
                    raise ValueError(
                        f"Missing event_id at {self.metrics_path}:{line_number}"
                    )
                event_ids.add(event_id)
        return event_ids

    def stage(self, record: dict[str, Any]) -> None:
        if record.get("schema_version") != METRICS_SCHEMA_VERSION:
            raise ValueError("Validation metric has an unsupported schema version")
        if record.get("event") != "validation":
            raise ValueError("Only validation events may be staged")
        if not isinstance(record.get("event_id"), str):
            raise ValueError("Validation metric must have an event_id")
        if not isinstance(record.get("step"), int):
            raise ValueError("Validation metric must have an integer step")
        write_json_atomic(self.pending_path, record)

    def pending(self, latest_checkpoint_step: int | None) -> dict[str, Any] | None:
        if not self.pending_path.exists():
            return None
        record = json.loads(self.pending_path.read_text(encoding="utf-8"))
        step = int(record["step"])
        if latest_checkpoint_step is None or step > latest_checkpoint_step:
            _unlink_durable(self.pending_path)
            print(
                f"discarded_uncommitted_validation_step={step} "
                f"latest_checkpoint={latest_checkpoint_step}",
                flush=True,
            )
            return None
        return record

    def commit(
        self,
        record: dict[str, Any],
        *,
        latest_checkpoint_step: int | None,
        mirror: Callable[[dict[str, Any]], None] | None = None,
    ) -> bool:
        step = int(record["step"])
        if latest_checkpoint_step is None or step > latest_checkpoint_step:
            raise ValueError(
                f"Cannot commit validation step {step} without a durable checkpoint"
            )
        event_id = str(record["event_id"])
        appended = event_id not in self._event_ids
        if appended:
            existing = (
                self.metrics_path.read_text(encoding="utf-8")
                if self.metrics_path.exists()
                else ""
            )
            line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            _write_text_atomic(self.metrics_path, existing + line)
            self._event_ids.add(event_id)
        if mirror is not None:
            mirror(record)
        _unlink_durable(self.pending_path)
        return appended
