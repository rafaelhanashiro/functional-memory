"""Tiny deterministic packed-corpus fixture; production code is read-only."""
from pathlib import Path
from typing import Any
import os
import numpy as np
from experiments.key_dimension.data.packed import FORMAT_NAME, FORMAT_VERSION, MANIFEST_NAME
from experiments.key_dimension.training.metrics import write_json_atomic as _atomic_write_json


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
