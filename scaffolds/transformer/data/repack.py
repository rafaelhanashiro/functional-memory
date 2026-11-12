"""Re-cut a packed corpus into rows of another length, preserving its token stream.

Packed shards hold consecutive, disjoint slices of one token stream: documents joined by
EOS and cut into fixed rows. Concatenating the rows in manifest order recovers the stream,
and cutting it into rows of a new length gives the corpus the packer would have written at
that length, with the same tokens in the same order. The trailing partial row is dropped.
Each source shard is checked against its manifest checksum as it is read, and the new
manifest records the source manifest's path and checksum.

    python -m scaffolds.transformer.data.repack SOURCE/manifest.json DESTINATION --sequence-length 4097
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np

from scaffolds.transformer.data.packed import FORMAT_NAME, FORMAT_VERSION, MANIFEST_NAME


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_shard(root: Path, shard: dict, sequence_length: int, dtype: str) -> np.ndarray:
    path = (root / shard["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Shard escapes the manifest directory: {path}")
    data = path.read_bytes()
    if shard.get("sha256") is not None and _sha256(data) != shard["sha256"]:
        raise ValueError(f"Checksum mismatch for packed-data shard: {path}")
    array = np.load(io.BytesIO(data), allow_pickle=False)
    if array.shape != (shard["num_sequences"], sequence_length) or array.dtype.name != dtype:
        raise ValueError(f"Packed-data shard {path} has shape {array.shape} and dtype {array.dtype}")
    return array


def repack(source_manifest: str | Path, destination: str | Path, sequence_length: int,
           rows_per_shard: int | None = None) -> dict:
    """Write the source corpus's token stream as rows of ``sequence_length`` under ``destination``."""
    source_path = Path(source_manifest).expanduser().resolve()
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    if source.get("format") != FORMAT_NAME or source.get("version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported packed-data manifest: {source_path}")
    if not isinstance(sequence_length, int) or sequence_length < 2:
        raise ValueError("sequence_length must be an integer of at least 2")
    source_length, dtype = source["sequence_length"], source["dtype"]
    if rows_per_shard is None:
        rows_per_shard = max(1, source["shards"][0]["num_sequences"] * source_length // sequence_length)
    destination = Path(destination).expanduser().resolve()
    if (destination / MANIFEST_NAME).exists():
        raise FileExistsError(f"A packed corpus already exists at {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    shards, pending, carry = [], [], np.empty(0, dtype=dtype)
    written = 0

    def flush(rows: np.ndarray) -> None:
        nonlocal written
        buffer = io.BytesIO()
        np.save(buffer, np.ascontiguousarray(rows), allow_pickle=False)
        data = buffer.getvalue()
        name = f"shard-{len(shards):05d}.npy"
        (destination / name).write_bytes(data)
        shards.append({"num_sequences": int(rows.shape[0]), "path": name,
                       "sequence_start": written, "sha256": _sha256(data)})
        written += int(rows.shape[0])

    for shard in source["shards"]:
        array = _read_shard(source_path.parent, shard, source_length, dtype)
        stream = np.concatenate([carry, array.reshape(-1)])
        count = stream.size // sequence_length
        carry = stream[count * sequence_length:].copy()
        pending.append(stream[:count * sequence_length].reshape(count, sequence_length))
        rows = np.concatenate(pending) if len(pending) > 1 else pending[0]
        while rows.shape[0] >= rows_per_shard:
            flush(rows[:rows_per_shard])
            rows = rows[rows_per_shard:]
        pending = [rows] if rows.shape[0] else []
    if pending:
        flush(np.concatenate(pending))
    if not shards:
        raise ValueError("The source corpus holds fewer tokens than one row of the new length")

    manifest = {
        "dtype": dtype, "format": FORMAT_NAME,
        "metadata": {**source.get("metadata", {}), "repacked_from": {
            "manifest": str(source_path), "manifest_sha256": _sha256(source_bytes),
            "sequence_length": source_length, "num_sequences": source["num_sequences"],
            "dropped_tail_tokens": int(carry.size)}},
        "num_sequences": written, "num_tokens": written * sequence_length,
        "sequence_length": sequence_length, "shards": shards, "version": FORMAT_VERSION,
    }
    temporary = destination / f".{MANIFEST_NAME}.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination / MANIFEST_NAME)
    return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--rows-per-shard", type=int)
    args = parser.parse_args(argv)
    manifest = repack(args.source_manifest, args.destination, args.sequence_length, args.rows_per_shard)
    print(json.dumps({key: manifest[key] for key in ("num_sequences", "num_tokens", "sequence_length")}
                     | {"shards": len(manifest["shards"]),
                        "dropped_tail_tokens": manifest["metadata"]["repacked_from"]["dropped_tail_tokens"]}))


if __name__ == "__main__":
    main()
