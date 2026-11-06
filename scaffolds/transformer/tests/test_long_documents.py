"""Long documents of a packed corpus become single-document rows that map back to the stream."""
import hashlib
import json

import numpy as np
import pytest

from scaffolds.transformer.data.long_documents import first_unseen_row, row_documents, write_long_documents
from scaffolds.transformer.data.packed import PackedBatchIterator, load_packed_manifest

EOS = 2
# Lengths of the documents in stream order; 0 is an empty document, 30 spans several shards.
LENGTHS = (5, 0, 12, 3, 30, 7, 11, 2, 25, 9)


def _write_corpus(root, sequence_length=7, rows_per_shard=(2, 3, 1, 4, 3), documents=None):
    """Pack documents (LENGTHS random ones by default), each followed by EOS, and cut the stream into rows."""
    rng = np.random.default_rng(0)
    if documents is None:
        documents = [rng.integers(3, 32000, size=length, dtype=np.uint16) for length in LENGTHS]
    lengths = [len(document) for document in documents]
    stream = np.concatenate([np.append(document, np.uint16(EOS)) for document in documents])
    rows = stream[:sum(rows_per_shard) * sequence_length].reshape(-1, sequence_length)
    root.mkdir(parents=True)
    shards, start = [], 0
    for index, count in enumerate(rows_per_shard):
        path = root / f"shard-{index:05d}.npy"
        np.save(path, rows[start:start + count])
        shards.append({"num_sequences": count, "path": path.name, "sequence_start": start,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        start += count
    manifest = {"dtype": "uint16", "format": "ofm-packed-sequences", "metadata": {"eos_token_id": EOS},
                "num_sequences": start, "num_tokens": start * sequence_length,
                "sequence_length": sequence_length, "shards": shards, "version": 1}
    (root / "manifest.json").write_text(json.dumps(manifest))
    offsets = np.cumsum([0] + [length + 1 for length in lengths[:-1]])
    complete = [i for i, (offset, length) in enumerate(zip(offsets, lengths)) if offset + length < rows.size]
    return root / "manifest.json", documents, offsets, complete


@pytest.mark.parametrize("rows_per_shard", [1, 2, 1024])
def test_rows_are_long_documents_in_stream_order(tmp_path, rows_per_shard):
    source, documents, offsets, complete = _write_corpus(tmp_path / "source")
    length = 11
    manifest = write_long_documents(source, tmp_path / "target", length, rows_per_shard, opening_tokens=10)
    long = [i for i in complete if LENGTHS[i] >= length - 1]
    assert long == [2, 4, 6] and complete == list(range(8))  # the 25- and 9-token documents are cut off
    path, loaded = load_packed_manifest(tmp_path / "target" / "manifest.json")
    assert loaded["num_sequences"] == manifest["num_sequences"] == len(long)
    rows = next(iter(PackedBatchIterator(path, batch_size=len(long), sequence_length=length)))
    for row, index in zip(rows, long):
        np.testing.assert_array_equal(row, np.concatenate([[EOS], documents[index][:length - 1]]))
    index = np.load(tmp_path / "target" / "documents.npy")
    starts = offsets[complete]
    np.testing.assert_array_equal(index, np.stack([starts, np.asarray(LENGTHS)[complete], starts], axis=1))
    np.testing.assert_array_equal(row_documents(path), index[long])
    provenance = loaded["metadata"]["long_documents_from"]
    assert provenance["documents"] == len(complete) and provenance["minimum_document_tokens"] == length - 1
    assert provenance["stream_tokens"] == provenance["num_sequences"] * provenance["sequence_length"]
    assert provenance["unindexed_tail_tokens"] == provenance["stream_tokens"] - offsets[8]
    assert provenance["manifest_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_first_unseen_row_follows_consumed_tokens(tmp_path):
    source, _, offsets, _ = _write_corpus(tmp_path / "source")
    write_long_documents(source, tmp_path / "target", 11, opening_tokens=10)
    path = tmp_path / "target" / "manifest.json"
    # Rows hold documents 2, 4 and 6, starting at stream offsets 7, 24 and 63.
    assert [offsets[2], offsets[4], offsets[6]] == [7, 24, 63]
    assert [first_unseen_row(path, consumed) for consumed in (0, 7, 8, 24, 25, 63, 64, 10**9)] == [0, 0, 1, 1, 2, 2, 3, 3]


def test_rejects_corruption_an_existing_target_and_a_corpus_without_long_documents(tmp_path):
    source, _, _, _ = _write_corpus(tmp_path / "source")
    write_long_documents(source, tmp_path / "target", 11, opening_tokens=10)
    with pytest.raises(FileExistsError):
        write_long_documents(source, tmp_path / "target", 11, opening_tokens=10)
    rows = tmp_path / "target" / "rows.npy"
    rows.write_bytes(rows.read_bytes()[:-1] + b"\x01")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        first_unseen_row(tmp_path / "target" / "manifest.json", 0)
    with pytest.raises(ValueError, match="no complete document"):
        write_long_documents(source, tmp_path / "too-long", 40, opening_tokens=10)
    shard = tmp_path / "source" / "shard-00001.npy"
    data = bytearray(shard.read_bytes()); data[-1] ^= 1; shard.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="Checksum mismatch"):
        write_long_documents(source, tmp_path / "corrupt", 11, opening_tokens=10)


def test_rows_keep_only_the_first_appearance_of_a_documents_opening(tmp_path):
    rng = np.random.default_rng(1)
    fresh = lambda size: rng.integers(3, 32000, size=size, dtype=np.uint16)
    a, short = fresh(12), fresh(5)
    documents = [
        a,                                       # 0: long, first appearance -> row
        fresh(3),                                # 1: shorter than the opening
        a.copy(),                                # 2: exact copy of 0
        np.concatenate([a[:6], fresh(8)]),       # 3: revised 0, same opening
        short,                                   # 4: short, but long enough to have an opening
        np.concatenate([short[:4], fresh(10)]),  # 5: long, opening first seen in 4
        fresh(11),                               # 6: long, first appearance -> row
        fresh(9),                                # 7: cut off by the end of the stream
    ]
    lengths = [len(document) for document in documents]
    stream_rows = (sum(lengths) + len(documents)) // 7
    source, _, offsets, complete = _write_corpus(tmp_path / "source", rows_per_shard=(2, 3, stream_rows - 5),
                                                 documents=documents)
    assert complete == list(range(7))
    manifest = write_long_documents(source, tmp_path / "target", 11, opening_tokens=4)
    path = tmp_path / "target" / "manifest.json"
    rows = next(iter(PackedBatchIterator(path, batch_size=2, sequence_length=11)))
    for row, index in zip(rows, (0, 6)):
        np.testing.assert_array_equal(row, np.concatenate([[EOS], documents[index][:10]]))
    index = np.load(tmp_path / "target" / "documents.npy")
    np.testing.assert_array_equal(index[:, 2], offsets[[0, 1, 0, 0, 4, 4, 6]])
    np.testing.assert_array_equal(row_documents(path), index[[0, 6]])
    provenance = manifest["metadata"]["long_documents_from"]
    assert manifest["num_sequences"] == 2 and provenance["repeated_long_documents"] == 3
    assert provenance["opening_tokens"] == 4
