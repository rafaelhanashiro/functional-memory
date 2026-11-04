"""Write a packed corpus's long documents as single-document rows, beside an index of every document.

A packed corpus is one token stream of documents, each followed by an end-of-document token,
cut into fixed rows. This tool reads the stream in manifest order, checking each shard against
its manifest checksum, and indexes every complete document by the stream offset of its first
token, its length (without the end-of-document token), and the offset where its opening first
appears; the document the stream cuts off at its end is not indexed. Web crawls repeat pages,
sometimes revised, so documents that begin with the same ``opening_tokens`` tokens count as
one document, first seen at the earliest of them. Each document of at least
``sequence_length - 1`` tokens whose opening appears here first becomes one row: an
end-of-document token, as precedes every document in the stream, then the document's first
``sequence_length - 1`` tokens, so position 0 is a document start as in training. Rows keep
stream order. Beside the manifest, ``documents.npy`` holds every document's
``[start, length, first_start]`` (``first_start`` is ``start`` for a document shorter than
the opening) and ``rows.npy`` each row's document number. Since every row is a first
appearance, a run that trained on the first T tokens of the stream never saw the rows whose
document starts at or after T, a suffix of the rows that begins at
``first_unseen_row(manifest, T)``.

    python -m scaffolds.transformer.data.long_documents SOURCE/manifest.json DESTINATION --sequence-length 16385
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
from scaffolds.transformer.data.repack import _read_shard, _sha256

DOCUMENTS_NAME, ROWS_NAME = "documents.npy", "rows.npy"


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def write_long_documents(source_manifest: str | Path, destination: str | Path, sequence_length: int,
                         rows_per_shard: int = 1024, opening_tokens: int = 1024) -> dict:
    """Write each first appearance of a document of at least ``sequence_length - 1`` tokens as one row."""
    source_path = Path(source_manifest).expanduser().resolve()
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    if source.get("format") != FORMAT_NAME or source.get("version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported packed-data manifest: {source_path}")
    eos = source.get("metadata", {}).get("eos_token_id")
    if not isinstance(eos, int):
        raise ValueError(f"The manifest records no eos_token_id: {source_path}")
    if not isinstance(sequence_length, int) or sequence_length < 2:
        raise ValueError("sequence_length must be an integer of at least 2")
    if not isinstance(rows_per_shard, int) or rows_per_shard < 1:
        raise ValueError("rows_per_shard must be a positive integer")
    if not isinstance(opening_tokens, int) or not 1 <= opening_tokens < sequence_length:
        raise ValueError("opening_tokens must be a positive integer of at most sequence_length - 1")
    destination = Path(destination).expanduser().resolve()
    if (destination / MANIFEST_NAME).exists():
        raise FileExistsError(f"A packed corpus already exists at {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    need, dtype = sequence_length - 1, source["dtype"]
    shards, pending, starts, lengths, firsts, row_documents = [], [], [], [], [], []
    written = documents = repeated = 0
    first_seen: dict[bytes, int] = {}  # Digest of a document's opening -> where that opening first appears.

    def flush() -> None:
        nonlocal written
        data = _npy_bytes(np.stack(pending))
        name = f"shard-{len(shards):05d}.npy"
        (destination / name).write_bytes(data)
        shards.append({"num_sequences": len(pending), "path": name, "sequence_start": written, "sha256": _sha256(data)})
        written += len(pending)
        pending.clear()

    def emit(tokens: np.ndarray, document: int) -> None:
        pending.append(np.concatenate([np.array([eos], dtype=dtype), tokens]))
        row_documents.append(document)
        if len(pending) == rows_per_shard:
            flush()

    def first_start(start: int, tokens: np.ndarray) -> int:
        """Where the opening of a document at ``start`` (``tokens``: its first tokens) first appears."""
        return first_seen.setdefault(hashlib.blake2b(tokens[:opening_tokens].tobytes(), digest_size=16).digest(), start)

    def keep(document: int, start: int, length: int, first: int, tokens: np.ndarray) -> None:
        nonlocal repeated
        if length >= need:
            if first == start:
                emit(tokens[:need], document)
            else:
                repeated += 1

    # The open document starts at stream offset `open_start`; `head` holds its first tokens, at most `need`.
    base, open_start, head = 0, 0, np.empty(0, dtype=dtype)
    for shard in source["shards"]:
        array = _read_shard(source_path.parent, shard, source["sequence_length"], dtype).reshape(-1)
        ends = np.flatnonzero(array == eos)
        if ends.size == 0:
            head = np.concatenate([head, array[:need - head.size]])
            base += array.size
            continue
        # The open document closes at the first end-of-document token.
        end = int(ends[0])
        length = base + end - open_start
        tokens = np.concatenate([head, array[:min(end, need - head.size)]])
        first = first_start(open_start, tokens) if length >= opening_tokens else open_start
        starts.append(np.array([open_start], dtype=np.int64))
        lengths.append(np.array([length], dtype=np.int64))
        firsts.append(np.array([first], dtype=np.int64))
        keep(documents, open_start, length, first, tokens)
        # Documents that start and end inside this shard.
        local = ends[:-1] + 1
        inside = ends[1:] - local
        inside_starts = base + local.astype(np.int64)
        inside_firsts = inside_starts.copy()
        for offset in np.flatnonzero(inside >= opening_tokens):
            start, size = int(inside_starts[offset]), int(inside[offset])
            tokens = array[local[offset]:local[offset] + min(size, need)]
            inside_firsts[offset] = first_start(start, tokens)
            keep(documents + 1 + int(offset), start, size, int(inside_firsts[offset]), tokens)
        starts.append(inside_starts)
        lengths.append(inside.astype(np.int64))
        firsts.append(inside_firsts)
        documents += 1 + inside.size
        open_start = base + int(ends[-1]) + 1
        head = array[ends[-1] + 1:ends[-1] + 1 + need].copy()
        base += array.size
    if pending:
        flush()
    if not shards:
        raise ValueError(f"The corpus holds no complete document of at least {need} tokens")

    table = np.stack([np.concatenate(starts), np.concatenate(lengths), np.concatenate(firsts)], axis=1)
    documents_bytes, rows_bytes = _npy_bytes(table), _npy_bytes(np.asarray(row_documents, dtype=np.int64))
    (destination / DOCUMENTS_NAME).write_bytes(documents_bytes)
    (destination / ROWS_NAME).write_bytes(rows_bytes)
    manifest = {
        "dtype": dtype, "format": FORMAT_NAME,
        "metadata": {**source.get("metadata", {}), "long_documents_from": {
            "manifest": str(source_path), "manifest_sha256": _sha256(source_bytes),
            "sequence_length": source["sequence_length"], "num_sequences": source["num_sequences"],
            "stream_tokens": base, "unindexed_tail_tokens": base - open_start,
            "documents": int(table.shape[0]), "minimum_document_tokens": need,
            "opening_tokens": opening_tokens, "repeated_long_documents": repeated,
            "row": (f"end-of-document token, then the first {need} tokens of a document whose first "
                    f"{opening_tokens} tokens no earlier document begins with"),
            "documents_file": DOCUMENTS_NAME, "documents_sha256": _sha256(documents_bytes),
            "rows_file": ROWS_NAME, "rows_sha256": _sha256(rows_bytes)}},
        "num_sequences": written, "num_tokens": written * sequence_length,
        "sequence_length": sequence_length, "shards": shards, "version": FORMAT_VERSION,
    }
    temporary = destination / f".{MANIFEST_NAME}.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination / MANIFEST_NAME)
    return manifest


def row_documents(manifest_path: str | Path) -> np.ndarray:
    """Each row's document as ``[start, length, first_start]`` in the source stream, checked against the manifest."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    provenance = json.loads(manifest_path.read_text())["metadata"]["long_documents_from"]
    arrays = []
    for name in ("documents", "rows"):
        data = (manifest_path.parent / provenance[f"{name}_file"]).read_bytes()
        if _sha256(data) != provenance[f"{name}_sha256"]:
            raise ValueError(f"Checksum mismatch for {provenance[f'{name}_file']} beside {manifest_path}")
        arrays.append(np.load(io.BytesIO(data), allow_pickle=False))
    documents, rows = arrays
    return documents[rows]


def first_unseen_row(manifest_path: str | Path, consumed_tokens: int) -> int:
    """The first row whose document starts at or after the first ``consumed_tokens`` tokens of the stream."""
    if consumed_tokens < 0:
        raise ValueError("consumed_tokens must be non-negative")
    return int(np.searchsorted(row_documents(manifest_path)[:, 0], consumed_tokens, side="left"))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--rows-per-shard", type=int, default=1024)
    parser.add_argument("--opening-tokens", type=int, default=1024)
    args = parser.parse_args(argv)
    manifest = write_long_documents(args.source_manifest, args.destination, args.sequence_length, args.rows_per_shard,
                                    args.opening_tokens)
    provenance = manifest["metadata"]["long_documents_from"]
    print(json.dumps({key: manifest[key] for key in ("num_sequences", "num_tokens", "sequence_length")}
                     | {key: provenance[key] for key in ("documents", "repeated_long_documents", "stream_tokens",
                                                         "unindexed_tail_tokens")}
                     | {"shards": len(manifest["shards"])}))


if __name__ == "__main__":
    main()
