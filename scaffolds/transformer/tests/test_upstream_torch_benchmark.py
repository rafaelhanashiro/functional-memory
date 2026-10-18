"""CPU checks for packed token alignment and the upstream benchmark's schedule."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scaffolds.transformer.benchmarks.benchmark_upstream_torch import digest, learning_rate, load_tokens


def test_npy_headers_and_shard_boundary_preserve_exact_next_token_targets(tmp_path):
    arrays = [np.arange(8, dtype=np.uint16).reshape(2, 4), np.arange(8, 16, dtype=np.uint16).reshape(2, 4)]
    shards = []
    for index, array in enumerate(arrays):
        path = tmp_path / f"shard-{index}.npy"
        np.save(path, array)
        shards.append({"path": path.name, "num_sequences": 2, "sequence_start": 2 * index, "sha256": digest(path)})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"format": "ofm-packed-sequences", "version": 1, "sequence_length": 4,
                                "dtype": "uint16", "num_sequences": 4, "shards": shards}))
    args = SimpleNamespace(global_batch=3, length=3, manifest=path, vocab_size=32)
    tokens, identity = load_tokens(args)
    np.testing.assert_array_equal(tokens, np.arange(12).reshape(3, 4))
    np.testing.assert_array_equal(tokens[:, :-1], [[0, 1, 2], [4, 5, 6], [8, 9, 10]])
    np.testing.assert_array_equal(tokens[:, 1:], [[1, 2, 3], [5, 6, 7], [9, 10, 11]])
    assert [shard["sequences_read"] for shard in identity["shards"]] == [2, 1]
    (tmp_path / "shard-0.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_tokens(args)


def test_learning_rate_matches_zero_start_warmup_and_cosine_endpoints():
    args = SimpleNamespace(learning_rate=3e-4, min_learning_rate=3e-5, lr_warmup_steps=400, total_steps=19074)
    assert learning_rate(args, 0) == 0
    assert learning_rate(args, 200) == pytest.approx(1.5e-4)
    assert learning_rate(args, 400) == pytest.approx(3e-4)
    assert learning_rate(args, 19074) == pytest.approx(3e-5)
    assert learning_rate(args, 20000) == pytest.approx(3e-5)
