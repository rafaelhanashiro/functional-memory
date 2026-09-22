import json

import numpy as np
import pytest

from experiments.shared_transformer.configs import Config
from experiments.shared_transformer.data.packed import PackedBatchIterator, load_packed_manifest
from experiments.shared_transformer.tests.packed_fixtures import PackedSequenceWriter
from experiments.shared_transformer.data.loaders import make_train_iterator, make_eval_iterator
from experiments.shared_transformer.training.loop import resolve_packed_data_cursor


def test_packed_cross_shard_batches_resume_without_repeat_or_skip(tmp_path):
    values = np.arange(60, dtype=np.int32).reshape(12, 5)
    writer = PackedSequenceWriter(tmp_path / "data", sequence_length=5, num_sequences=12, shard_sequences=3)
    writer.write(values)
    path = writer.finish()
    all_batches = list(PackedBatchIterator(path, batch_size=2, sequence_length=5))
    resumed = list(PackedBatchIterator(path, batch_size=2, sequence_length=5, start_sequence=4))
    np.testing.assert_array_equal(np.concatenate(all_batches), values)
    np.testing.assert_array_equal(np.concatenate(resumed), values[4:])


def test_synthetic_data_is_cursor_and_microbatch_invariant():
    cfg = Config().data
    cfg.batch_size, cfg.sequence_length, cfg.synthetic_vocab_size = 4, 5, 32
    full = next(make_train_iterator(cfg))
    cfg.batch_size = 2
    it = make_train_iterator(cfg)
    np.testing.assert_array_equal(np.concatenate([next(it), next(it)]), full)
    np.testing.assert_array_equal(next(make_train_iterator(cfg, start_sequence=2)), full[2:])
    np.testing.assert_array_equal(next(make_eval_iterator(cfg)), next(make_eval_iterator(cfg)))


def test_packed_identity_change_rejected_on_resume(tmp_path):
    cfg = Config()
    cfg.train.workdir = tmp_path
    cfg.data.source, cfg.data.batch_size, cfg.data.sequence_length = "packed", 2, 5
    cfg.optim.total_steps, cfg.train.eval_batches = 2, 1
    for name in ("train", "eval"):
        writer = PackedSequenceWriter(tmp_path / name, sequence_length=5, num_sequences=4, shard_sequences=3)
        writer.write(np.arange(20).reshape(4, 5))
        setattr(cfg.data, f"{name}_manifest", str(writer.finish()))
    assert resolve_packed_data_cursor(cfg, 0) == 0
    assert resolve_packed_data_cursor(cfg, 1) == 2
    path = tmp_path / "train" / "manifest.json"
    data = json.loads(path.read_text())
    data["metadata"] = {"different": True}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identity"):
        resolve_packed_data_cursor(cfg, 1)


def test_manifest_rejects_shard_traversal(tmp_path):
    writer = PackedSequenceWriter(tmp_path / "data", sequence_length=5, num_sequences=2, shard_sequences=2)
    writer.write(np.zeros((2, 5)))
    path = writer.finish()
    data = json.loads(path.read_text())
    data["shards"][0]["path"] = "../../escape.npy"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="escapes"):
        load_packed_manifest(path)
