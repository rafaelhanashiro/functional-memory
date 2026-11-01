"""Re-cutting a packed corpus keeps its token stream and writes a loadable corpus."""
import hashlib
import json

import numpy as np
import pytest

from scaffolds.transformer.data.packed import PackedBatchIterator, load_packed_manifest
from scaffolds.transformer.data.repack import repack


def _write_corpus(root, rows_per_shard=(5, 4), sequence_length=7, checksums=True):
    root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    shards, start, stream = [], 0, []
    for index, rows in enumerate(rows_per_shard):
        array = rng.integers(0, 32000, size=(rows, sequence_length), dtype=np.uint16)
        np.save(root / f"shard-{index:05d}.npy", array)
        shard = {"num_sequences": rows, "path": f"shard-{index:05d}.npy", "sequence_start": start}
        if checksums:
            shard["sha256"] = hashlib.sha256((root / shard["path"]).read_bytes()).hexdigest()
        shards.append(shard)
        start += rows
        stream.append(array.reshape(-1))
    manifest = {"dtype": "uint16", "format": "ofm-packed-sequences", "metadata": {"shuffle_seed": 0},
                "num_sequences": start, "num_tokens": start * sequence_length,
                "sequence_length": sequence_length, "shards": shards, "version": 1}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root / "manifest.json", np.concatenate(stream)


@pytest.mark.parametrize("length,rows_per_shard", [(13, 2), (13, None), (3, 4)])
def test_repack_preserves_the_token_stream(tmp_path, length, rows_per_shard):
    source, stream = _write_corpus(tmp_path / "source")
    manifest = repack(source, tmp_path / "target", length, rows_per_shard)
    path, loaded = load_packed_manifest(tmp_path / "target" / "manifest.json")
    count = stream.size // length
    assert loaded["num_sequences"] == manifest["num_sequences"] == count
    assert loaded["num_tokens"] == count * length and loaded["sequence_length"] == length
    rows = next(iter(PackedBatchIterator(path, batch_size=count, sequence_length=length)))
    np.testing.assert_array_equal(rows.reshape(-1), stream[:count * length])
    provenance = loaded["metadata"]["repacked_from"]
    assert provenance["dropped_tail_tokens"] == stream.size - count * length
    assert provenance["manifest_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert loaded["metadata"]["shuffle_seed"] == 0


def test_repack_rejects_a_corrupted_source_and_an_existing_target(tmp_path):
    source, _ = _write_corpus(tmp_path / "source")
    shard = tmp_path / "source" / "shard-00001.npy"
    data = bytearray(shard.read_bytes()); data[-1] ^= 1; shard.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="Checksum mismatch"):
        repack(source, tmp_path / "target", 13)
    clean, _ = _write_corpus(tmp_path / "clean", checksums=False)
    repack(clean, tmp_path / "done", 13)
    with pytest.raises(FileExistsError):
        repack(clean, tmp_path / "done", 13)
