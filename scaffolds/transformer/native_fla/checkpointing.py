"""Atomic native PyTorch training snapshots, including the next input and RNG state."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, Iterable

import numpy as np
import torch


FORMAT = "native-fla-training-state"
VERSION = 1


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    cuda_states = state["cuda"]
    if cuda_states and (not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count()):
        raise ValueError("Checkpoint CUDA RNG allocation differs from the current allocation")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if cuda_states:
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])


def checkpoint_metadata(directory: str | Path) -> dict[str, Any]:
    directory = Path(directory)
    try:
        metadata = json.loads((directory / "metadata.json").read_text())
        valid = (
            metadata.get("format") == FORMAT
            and metadata.get("version") == VERSION
            and type(metadata.get("step")) is int
            and metadata["step"] >= 0
            and directory.name.isdecimal()
            and int(directory.name) == metadata["step"]
            and (directory / "state.pt").stat().st_size == metadata.get("state_size_bytes")
        )
    except (OSError, ValueError, TypeError, AttributeError) as error:
        raise ValueError(f"Invalid native FLA checkpoint: {directory}") from error
    if not valid:
        raise ValueError(f"Invalid native FLA checkpoint: {directory}")
    return metadata


def save_checkpoint(
    directory: str | Path, *, model, optimizer, step: int, data_cursor,
    best_validation, config_sha256: str, extra: dict[str, Any] | None = None,
) -> Path:
    """Publish one immutable step directory after both files are durable.

    The caller saves at most once per completed update. A failed write never
    publishes a numeric directory, and an existing checkpoint is never replaced.
    The learning-rate schedule is a function of the completed ``step``; stateful
    scheduler state, if used by a future runtime, belongs in ``extra``.
    """
    directory = Path(directory)
    if (type(step) is not int or step < 0 or not directory.name.isdecimal()
            or int(directory.name) != step):
        raise ValueError("Checkpoint directory must be named for its nonnegative completed step")
    if not isinstance(config_sha256, str) or not config_sha256:
        raise ValueError("Checkpoint requires its immutable configuration hash")
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        raise FileExistsError(f"Checkpoint step is already published: {directory}")
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    try:
        state = {
            "format": FORMAT, "version": VERSION, "step": step,
            "config_sha256": config_sha256,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "rng": capture_rng_state(), "data_cursor": data_cursor,
            "best_validation": best_validation, "extra": extra or {},
        }
        with (staging / "state.pt").open("wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        metadata = {
            "format": FORMAT, "version": VERSION, "step": step,
            "config_sha256": config_sha256, "data_cursor": data_cursor,
            "best_validation": best_validation,
            "state_size_bytes": (staging / "state.pt").stat().st_size,
        }
        with (staging / "metadata.json").open("w") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _sync_directory(staging)
        os.rename(staging, directory)
        _sync_directory(directory.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return directory


def load_checkpoint(
    directory: str | Path, *, model, optimizer=None,
    expected_config_sha256: str | None = None, restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore a trusted local training snapshot and return its progress fields.

    Loading into CPU first also restores a CUDA checkpoint onto the model's
    current device, without relying on saved device indices. PyTorch's optimizer
    loader moves moment tensors to their corresponding parameter devices.
    """
    directory = Path(directory)
    metadata = checkpoint_metadata(directory)
    if expected_config_sha256 is not None and metadata["config_sha256"] != expected_config_sha256:
        raise ValueError("Checkpoint configuration hash differs from the frozen recipe")
    state = torch.load(directory / "state.pt", map_location="cpu", weights_only=False)
    for key in ("format", "version", "step", "config_sha256", "data_cursor", "best_validation"):
        if state.get(key) != metadata[key]:
            raise ValueError(f"Checkpoint payload disagrees with metadata: {key}")
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if restore_rng:
        restore_rng_state(state["rng"])
    return {key: state[key] for key in ("step", "data_cursor", "best_validation", "config_sha256", "extra")}


def checkpoint_directories(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    directories = sorted((path for path in root.iterdir() if path.is_dir() and path.name.isdecimal()),
                         key=lambda path: int(path.name))
    for directory in directories:
        checkpoint_metadata(directory)
    return directories


def latest_checkpoint(root: str | Path) -> Path | None:
    directories = checkpoint_directories(root)
    return directories[-1] if directories else None


def prune_checkpoints(root: str | Path, keep_steps: Iterable[int]) -> list[Path]:
    """Remove published checkpoints only after every requested retained step exists."""
    keep = set(keep_steps)
    if not keep or any(type(step) is not int or step < 0 for step in keep):
        raise ValueError("Retention requires at least one nonnegative checkpoint step")
    directories = checkpoint_directories(root)
    available = {int(path.name) for path in directories}
    if missing := keep - available:
        raise ValueError(f"Cannot prune checkpoints: retained steps are absent: {sorted(missing)}")
    removed = []
    for directory in directories:
        if int(directory.name) not in keep:
            shutil.rmtree(directory)
            removed.append(directory)
    if removed:
        _sync_directory(Path(root))
    return removed
