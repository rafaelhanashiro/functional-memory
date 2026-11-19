from __future__ import annotations

import bisect
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


FORMAT_NAME = "ofm-packed-sequences"
FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
PROGRESS_NAME = "progress.json"

LoadedPackedManifest = tuple[Path, dict[str, Any]]


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp_path, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_fingerprint(path: str | Path) -> str:
    manifest_path = Path(path).expanduser().resolve()
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def load_packed_manifest(path: str | Path) -> LoadedPackedManifest:
    manifest_path = Path(path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())

    if manifest.get("format") != FORMAT_NAME:
        raise ValueError(f"Unsupported packed-data format in {manifest_path}")
    if manifest.get("version") != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported packed-data version {manifest.get('version')} in {manifest_path}"
        )

    sequence_length = manifest.get("sequence_length")
    num_sequences = manifest.get("num_sequences")
    dtype = manifest.get("dtype")
    shards = manifest.get("shards")
    if not isinstance(sequence_length, int) or sequence_length < 2:
        raise ValueError(f"Invalid sequence_length in {manifest_path}")
    if not isinstance(num_sequences, int) or num_sequences < 1:
        raise ValueError(f"Invalid num_sequences in {manifest_path}")
    if dtype not in {"uint16", "int32"}:
        raise ValueError(f"Unsupported dtype {dtype!r} in {manifest_path}")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"No shards listed in {manifest_path}")

    expected_start = 0
    root = manifest_path.parent
    for index, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise ValueError(f"Invalid shard {index} in {manifest_path}")
        relative_path = shard.get("path")
        sequence_start = shard.get("sequence_start")
        shard_sequences = shard.get("num_sequences")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"Invalid path for shard {index} in {manifest_path}")
        if sequence_start != expected_start:
            raise ValueError(f"Non-contiguous shard {index} in {manifest_path}")
        if not isinstance(shard_sequences, int) or shard_sequences < 1:
            raise ValueError(f"Invalid size for shard {index} in {manifest_path}")

        shard_path = (root / relative_path).resolve()
        if not shard_path.is_relative_to(root):
            raise ValueError(f"Shard {index} escapes the manifest directory")
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing packed-data shard: {shard_path}")
        checksum = shard.get("sha256")
        if checksum is not None:
            if not isinstance(checksum, str) or len(checksum) != 64:
                raise ValueError(f"Invalid checksum for shard {index} in {manifest_path}")
            actual_checksum = _sha256_file(shard_path)
            if actual_checksum != checksum:
                raise ValueError(
                    f"Checksum mismatch for packed-data shard: {shard_path}"
                )
        expected_start += shard_sequences

    if expected_start != num_sequences:
        raise ValueError(
            f"Shard sequence count {expected_start} does not match manifest count "
            f"{num_sequences} in {manifest_path}"
        )
    return manifest_path, manifest


def resolve_packed_manifest(
    path: str | Path,
    validated_manifest: LoadedPackedManifest | None = None,
) -> LoadedPackedManifest:
    """Load and validate a manifest, or reuse one validated earlier in this process."""
    manifest_path = Path(path).expanduser().resolve()
    if validated_manifest is None:
        return load_packed_manifest(manifest_path)
    validated_path, manifest = validated_manifest
    if validated_path != manifest_path:
        raise ValueError(
            f"Validated packed-data manifest {validated_path} does not match {manifest_path}"
        )
    return validated_path, manifest


class PackedSequenceWriter:
    """Write a fixed number of token sequences to atomic NumPy shards."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        sequence_length: int,
        num_sequences: int,
        shard_sequences: int,
        dtype: str = "uint16",
        metadata: dict[str, Any] | None = None,
    ):
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if num_sequences < 1:
            raise ValueError("num_sequences must be positive")
        if shard_sequences < 1:
            raise ValueError("shard_sequences must be positive")
        if dtype not in {"uint16", "int32"}:
            raise ValueError(f"Unsupported packed token dtype: {dtype}")

        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        existing = list(self.output_dir.glob("shard-*.npy"))
        existing.extend(self.output_dir.glob("shard-*.npy.tmp"))
        if (self.output_dir / MANIFEST_NAME).exists() or existing:
            raise FileExistsError(f"Packed-data output is not empty: {self.output_dir}")

        self.sequence_length = sequence_length
        self.num_sequences = num_sequences
        self.shard_sequences = shard_sequences
        self.dtype = np.dtype(dtype)
        self.metadata = metadata or {}
        self.num_written = 0
        self.shards: list[dict[str, Any]] = []
        self._shard: np.memmap | None = None
        self._shard_path: Path | None = None
        self._shard_rows = 0
        self._shard_capacity = 0

    @property
    def remaining(self) -> int:
        return self.num_sequences - self.num_written

    def _open_shard(self) -> None:
        if self._shard is not None or self.remaining < 1:
            return
        shard_index = len(self.shards)
        self._shard_capacity = min(self.shard_sequences, self.remaining)
        self._shard_rows = 0
        self._shard_path = self.output_dir / f"shard-{shard_index:05d}.npy"
        temp_path = self._shard_path.with_suffix(".npy.tmp")
        self._shard = np.lib.format.open_memmap(
            temp_path,
            mode="w+",
            dtype=self.dtype,
            shape=(self._shard_capacity, self.sequence_length),
        )

    def _finish_shard(self) -> None:
        if self._shard is None or self._shard_path is None:
            return
        if self._shard_rows != self._shard_capacity:
            raise RuntimeError("Cannot finalize a partially written packed-data shard")

        sequence_start = self.num_written - self._shard_rows
        temp_path = self._shard_path.with_suffix(".npy.tmp")
        self._shard.flush()
        del self._shard
        self._shard = None
        os.replace(temp_path, self._shard_path)
        self.shards.append(
            {
                "path": self._shard_path.name,
                "sequence_start": sequence_start,
                "num_sequences": self._shard_rows,
            }
        )
        self._shard_path = None
        self._shard_rows = 0
        self._shard_capacity = 0

    def write(self, sequences: np.ndarray) -> int:
        values = np.asarray(sequences)
        if values.ndim != 2 or values.shape[1] != self.sequence_length:
            raise ValueError(
                f"Expected sequences shaped [N, {self.sequence_length}], got {values.shape}"
            )
        if values.size and (int(values.min()) < 0 or int(values.max()) > np.iinfo(self.dtype).max):
            raise ValueError(f"Token ids do not fit in {self.dtype.name}")

        requested = len(values)
        if requested > self.remaining:
            raise ValueError(
                f"Attempted to write {requested} sequences with only {self.remaining} remaining"
            )

        source_start = 0
        while source_start < requested:
            self._open_shard()
            if self._shard is None:
                raise RuntimeError("Packed-data writer has no available shard")
            count = min(requested - source_start, self._shard_capacity - self._shard_rows)
            source_end = source_start + count
            shard_end = self._shard_rows + count
            self._shard[self._shard_rows : shard_end] = values[source_start:source_end]
            self._shard_rows = shard_end
            self.num_written += count
            source_start = source_end
            if self._shard_rows == self._shard_capacity:
                self._finish_shard()
        return requested

    def finish(self) -> Path:
        if self.num_written != self.num_sequences:
            raise RuntimeError(
                f"Packed-data writer expected {self.num_sequences} sequences but received "
                f"{self.num_written}"
            )
        self._finish_shard()
        manifest = {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "dtype": self.dtype.name,
            "sequence_length": self.sequence_length,
            "num_sequences": self.num_sequences,
            "num_tokens": self.num_sequences * self.sequence_length,
            "metadata": self.metadata,
            "shards": self.shards,
        }
        manifest_path = self.output_dir / MANIFEST_NAME
        _atomic_write_json(manifest_path, manifest)
        return manifest_path


class ResumablePackedSequenceWriter(PackedSequenceWriter):
    """Append to finalized shards after deterministically replaying their prefix."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        sequence_length: int,
        num_sequences: int,
        shard_sequences: int,
        dtype: str = "uint16",
        metadata: dict[str, Any] | None = None,
    ):
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if num_sequences < 1:
            raise ValueError("num_sequences must be positive")
        if shard_sequences < 1:
            raise ValueError("shard_sequences must be positive")
        if dtype not in {"uint16", "int32"}:
            raise ValueError(f"Unsupported packed token dtype: {dtype}")

        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sequence_length = sequence_length
        self.num_sequences = num_sequences
        self.shard_sequences = shard_sequences
        self.dtype = np.dtype(dtype)
        self.metadata = metadata or {}
        self.num_written = 0
        self.shards: list[dict[str, Any]] = []
        self._shard: np.memmap | None = None
        self._shard_path: Path | None = None
        self._shard_rows = 0
        self._shard_capacity = 0
        self.progress_path = self.output_dir / PROGRESS_NAME

        for temp_path in self.output_dir.glob("shard-*.npy.tmp"):
            temp_path.unlink()
        progress_temp = self.progress_path.with_suffix(self.progress_path.suffix + ".tmp")
        if progress_temp.exists():
            progress_temp.unlink()

        if self.progress_path.exists():
            progress = json.loads(self.progress_path.read_text())
            expected = {
                "format": "ofm-packed-build-progress",
                "version": 1,
                "dtype": self.dtype.name,
                "sequence_length": self.sequence_length,
                "num_sequences": self.num_sequences,
                "shard_sequences": self.shard_sequences,
                "metadata": self.metadata,
            }
            for key, value in expected.items():
                if progress.get(key) != value:
                    raise ValueError(
                        f"Packed-data progress {key} does not match this build: "
                        f"{self.progress_path}"
                    )
            shards = progress.get("shards")
            if not isinstance(shards, list):
                raise ValueError(f"Invalid shard progress in {self.progress_path}")
            self.shards = shards
            self._validate_existing_shards()
            self.num_written = sum(shard["num_sequences"] for shard in self.shards)
        else:
            existing = list(self.output_dir.glob("shard-*.npy"))
            if (self.output_dir / MANIFEST_NAME).exists() or existing:
                raise FileExistsError(
                    f"Packed-data output has no resumable progress: {self.output_dir}"
                )
            self._write_progress()

        self._source_sequences = 0
        self._replay_remaining = self.num_written
        self._replay_shard_index = 0
        self._replay_shard_offset = 0
        self._replay_array: np.ndarray | None = None

    @property
    def remaining(self) -> int:
        return self.num_sequences - self._source_sequences

    def _progress_value(self) -> dict[str, Any]:
        return {
            "format": "ofm-packed-build-progress",
            "version": 1,
            "dtype": self.dtype.name,
            "sequence_length": self.sequence_length,
            "num_sequences": self.num_sequences,
            "shard_sequences": self.shard_sequences,
            "metadata": self.metadata,
            "shards": self.shards,
        }

    def _write_progress(self) -> None:
        _atomic_write_json(self.progress_path, self._progress_value())

    def _validate_existing_shards(self) -> None:
        expected_start = 0
        listed_paths: set[str] = set()
        for index, shard in enumerate(self.shards):
            expected_path = f"shard-{index:05d}.npy"
            if shard.get("path") != expected_path:
                raise ValueError(f"Invalid resumed shard path in {self.progress_path}")
            if shard.get("sequence_start") != expected_start:
                raise ValueError(f"Non-contiguous resumed shard in {self.progress_path}")
            shard_rows = shard.get("num_sequences")
            expected_rows = min(self.shard_sequences, self.num_sequences - expected_start)
            if not isinstance(shard_rows, int) or shard_rows != expected_rows:
                raise ValueError(f"Invalid resumed shard size in {self.progress_path}")
            checksum = shard.get("sha256")
            if not isinstance(checksum, str) or len(checksum) != 64:
                raise ValueError(f"Invalid resumed shard checksum in {self.progress_path}")

            shard_path = self.output_dir / expected_path
            if not shard_path.is_file():
                raise FileNotFoundError(f"Missing resumed packed-data shard: {shard_path}")
            array = np.load(shard_path, mmap_mode="r", allow_pickle=False)
            if array.shape != (shard_rows, self.sequence_length):
                raise ValueError(f"Invalid resumed shard shape: {shard_path}")
            if array.dtype != self.dtype:
                raise ValueError(f"Invalid resumed shard dtype: {shard_path}")
            del array
            if _sha256_file(shard_path) != checksum:
                raise ValueError(f"Checksum mismatch for resumed shard: {shard_path}")

            listed_paths.add(expected_path)
            expected_start += shard_rows

        actual_paths = {path.name for path in self.output_dir.glob("shard-*.npy")}
        untracked_paths = actual_paths - listed_paths
        next_shard = f"shard-{len(self.shards):05d}.npy"
        if untracked_paths - {next_shard}:
            raise ValueError(f"Untracked packed-data shards in {self.output_dir}")
        for relative_path in untracked_paths:
            (self.output_dir / relative_path).unlink()
        if expected_start > self.num_sequences:
            raise ValueError(f"Resumed shards exceed target size in {self.progress_path}")

    def _open_shard(self) -> None:
        if self._shard is not None or self.num_written >= self.num_sequences:
            return
        shard_index = len(self.shards)
        persisted_remaining = self.num_sequences - self.num_written
        self._shard_capacity = min(self.shard_sequences, persisted_remaining)
        self._shard_rows = 0
        self._shard_path = self.output_dir / f"shard-{shard_index:05d}.npy"
        temp_path = self._shard_path.with_suffix(".npy.tmp")
        self._shard = np.lib.format.open_memmap(
            temp_path,
            mode="w+",
            dtype=self.dtype,
            shape=(self._shard_capacity, self.sequence_length),
        )

    def _finish_shard(self) -> None:
        previous_count = len(self.shards)
        super()._finish_shard()
        if len(self.shards) == previous_count:
            return
        shard = self.shards[-1]
        shard_path = self.output_dir / shard["path"]
        shard["sha256"] = _sha256_file(shard_path)
        self._write_progress()

    def _verify_replay(self, values: np.ndarray) -> int:
        replay_count = min(len(values), self._replay_remaining)
        value_offset = 0
        while value_offset < replay_count:
            shard = self.shards[self._replay_shard_index]
            if self._replay_array is None:
                shard_path = self.output_dir / shard["path"]
                self._replay_array = np.load(shard_path, mmap_mode="r", allow_pickle=False)

            available = shard["num_sequences"] - self._replay_shard_offset
            count = min(replay_count - value_offset, available)
            expected = self._replay_array[
                self._replay_shard_offset : self._replay_shard_offset + count
            ]
            actual = values[value_offset : value_offset + count]
            if not np.array_equal(expected, actual):
                sequence = self._source_sequences + value_offset
                raise ValueError(
                    f"Replayed packed data differs at sequence {sequence} in {self.output_dir}"
                )

            value_offset += count
            self._replay_shard_offset += count
            if self._replay_shard_offset == shard["num_sequences"]:
                self._replay_shard_index += 1
                self._replay_shard_offset = 0
                self._replay_array = None

        self._replay_remaining -= replay_count
        return replay_count

    def write(self, sequences: np.ndarray) -> int:
        values = np.asarray(sequences)
        if values.ndim != 2 or values.shape[1] != self.sequence_length:
            raise ValueError(
                f"Expected sequences shaped [N, {self.sequence_length}], got {values.shape}"
            )
        if values.size and (int(values.min()) < 0 or int(values.max()) > np.iinfo(self.dtype).max):
            raise ValueError(f"Token ids do not fit in {self.dtype.name}")
        if len(values) > self.remaining:
            raise ValueError(
                f"Attempted to consume {len(values)} sequences with only "
                f"{self.remaining} remaining"
            )

        replayed = self._verify_replay(values)
        if replayed < len(values):
            super().write(values[replayed:])
        self._source_sequences += len(values)
        return len(values)

    def finish(self) -> Path:
        if self._source_sequences != self.num_sequences:
            raise RuntimeError(
                f"Packed-data source expected {self.num_sequences} sequences but produced "
                f"{self._source_sequences}"
            )
        if self._replay_remaining:
            raise RuntimeError("Packed-data replay did not reach the persisted prefix")
        return super().finish()


class PackedTokenPacker:
    """Concatenate tokenized documents and feed fixed sequences to a writer."""

    def __init__(self, writer: PackedSequenceWriter):
        self.writer = writer
        self.buffer: list[int] = []
        self.offset = 0

    @property
    def done(self) -> bool:
        return self.writer.remaining == 0

    def add_document(self, token_ids: list[int], eos_token_id: int | None) -> None:
        if self.done or not token_ids:
            return
        self.buffer.extend(token_ids)
        if eos_token_id is not None:
            self.buffer.append(eos_token_id)
        self._drain()

    def _drain(self) -> None:
        available = len(self.buffer) - self.offset
        count = min(available // self.writer.sequence_length, self.writer.remaining)
        if count:
            end = self.offset + count * self.writer.sequence_length
            values = np.asarray(self.buffer[self.offset : end], dtype=np.int32)
            self.writer.write(values.reshape(count, self.writer.sequence_length))
            self.offset = end

        if self.offset and (self.offset >= 1_000_000 or self.done):
            del self.buffer[: self.offset]
            self.offset = 0


class PackedBatchIterator:
    """Read fixed batches from immutable packed sequence shards."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        batch_size: int,
        sequence_length: int,
        start_sequence: int = 0,
        validated_manifest: LoadedPackedManifest | None = None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.manifest_path, self.manifest = resolve_packed_manifest(
            manifest_path,
            validated_manifest,
        )
        if self.manifest["sequence_length"] != sequence_length:
            raise ValueError(
                f"Configured sequence length {sequence_length} does not match packed-data "
                f"length {self.manifest['sequence_length']}"
            )
        if start_sequence < 0 or start_sequence > self.manifest["num_sequences"]:
            raise ValueError(f"Invalid packed-data start sequence: {start_sequence}")

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.cursor = start_sequence
        self.shards = self.manifest["shards"]
        self.shard_ends = [
            shard["sequence_start"] + shard["num_sequences"] for shard in self.shards
        ]
        self._open_index: int | None = None
        self._open_array: np.ndarray | None = None

    def __iter__(self) -> PackedBatchIterator:
        return self

    def _array_for_shard(self, index: int) -> np.ndarray:
        if self._open_index != index:
            shard_path = self.manifest_path.parent / self.shards[index]["path"]
            array = np.load(shard_path, mmap_mode="r", allow_pickle=False)
            expected_shape = (self.shards[index]["num_sequences"], self.sequence_length)
            if array.shape != expected_shape:
                raise ValueError(
                    f"Packed-data shard {shard_path} has shape {array.shape}, expected "
                    f"{expected_shape}"
                )
            if array.dtype.name != self.manifest["dtype"]:
                raise ValueError(
                    f"Packed-data shard {shard_path} has dtype {array.dtype}, expected "
                    f"{self.manifest['dtype']}"
                )
            self._open_index = index
            self._open_array = array
        if self._open_array is None:
            raise RuntimeError("Packed-data shard failed to open")
        return self._open_array

    def __next__(self) -> np.ndarray:
        end_cursor = self.cursor + self.batch_size
        if end_cursor > self.manifest["num_sequences"]:
            raise StopIteration

        batch = np.empty((self.batch_size, self.sequence_length), dtype=np.int32)
        batch_offset = 0
        while self.cursor < end_cursor:
            shard_index = bisect.bisect_right(self.shard_ends, self.cursor)
            shard = self.shards[shard_index]
            local_start = self.cursor - shard["sequence_start"]
            count = min(end_cursor - self.cursor, shard["num_sequences"] - local_start)
            local_end = local_start + count
            array = self._array_for_shard(shard_index)
            batch[batch_offset : batch_offset + count] = array[local_start:local_end]
            self.cursor += count
            batch_offset += count
        return batch
