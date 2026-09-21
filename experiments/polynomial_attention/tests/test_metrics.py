import json

import pytest

from experiments.polynomial_attention.training.metrics import METRICS_SCHEMA_VERSION, ValidationMetricsStore


def validation_record(step: int) -> dict:
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "event": "validation",
        "event_id": f"validation:{step}",
        "step": step,
    }


def test_validation_metric_requires_a_checkpoint_before_commit(tmp_path):
    store = ValidationMetricsStore(tmp_path)
    record = validation_record(10)
    store.stage(record)

    with pytest.raises(ValueError, match="without a durable checkpoint"):
        store.commit(record, latest_checkpoint_step=9)

    assert store.pending_path.exists()
    assert not store.metrics_path.exists()


def test_uncheckpointed_pending_metric_is_discarded_on_resume(tmp_path):
    store = ValidationMetricsStore(tmp_path)
    store.stage(validation_record(10))

    assert store.pending(latest_checkpoint_step=9) is None
    assert not store.pending_path.exists()
    assert not store.metrics_path.exists()


def test_checkpointed_metric_commit_is_append_only_and_idempotent(tmp_path):
    store = ValidationMetricsStore(tmp_path)
    record = validation_record(10)
    mirrored = []
    store.stage(record)

    assert store.commit(
        record,
        latest_checkpoint_step=10,
        mirror=mirrored.append,
    )
    assert not store.pending_path.exists()
    assert not store.commit(record, latest_checkpoint_step=10)

    lines = store.metrics_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [record]
    assert mirrored == [record]


def test_checkpointed_pending_metric_can_be_recovered(tmp_path):
    first_store = ValidationMetricsStore(tmp_path)
    record = validation_record(10)
    record["attention_diagnostics"] = {
        "step": 10, "valid": True,
        "sample": {"selection": "first_heldout_sequence", "num_sequences": 1},
        "layers": [{"score_min": -3.0, "normalizer_min": 1.0}],
    }
    first_store.stage(record)

    resumed_store = ValidationMetricsStore(tmp_path)
    pending = resumed_store.pending(latest_checkpoint_step=10)
    assert pending == record
    assert resumed_store.commit(pending, latest_checkpoint_step=10)

    recovered = json.loads(
        resumed_store.metrics_path.read_text(encoding="utf-8").strip()
    )
    assert recovered == record
