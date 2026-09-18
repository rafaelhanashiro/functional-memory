from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


FORMAT_NAME = "ofm-packed-sequences"
FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"

LoadedPackedManifest = tuple[Path, dict[str, Any]]


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
