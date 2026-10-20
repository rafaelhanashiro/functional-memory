"""Explicit source upgrades preserve the scientific recipe and complete training state."""
from dataclasses import replace
import json

import numpy as np
import pytest

from scaffolds.transformer.configs.loader import save_config
from scaffolds.transformer.tests.packed_fixtures import PackedSequenceWriter
from scaffolds.transformer.tests.test_training import tiny_config
from scaffolds.transformer.training import loop


OLD_SOURCE = "1" * 40
NEW_SOURCE = "2" * 40


def initialize(config):
    config.model.attention_backend = "blocked"
    config.model.attention_degree = 2
    config.train.source_commit = OLD_SOURCE
    recipe = replace(config, train=replace(config.train, source_commit="auto"))
    save_config(recipe, config.train.workdir / "config.yaml")
    loop.ensure_run_config(config, resume=True)
    return config


def new_source(config):
    return replace(config, train=replace(config.train, source_commit=NEW_SOURCE))


def test_source_transition_requires_exact_opt_in_without_rewriting_recipe(tmp_path):
    config = initialize(tiny_config(tmp_path))
    paths = [tmp_path / name for name in ("config.yaml", "config.json")]
    before = [path.read_bytes() for path in paths]
    changed = new_source(config)
    with pytest.raises(ValueError, match="does not match"):
        loop.ensure_run_config(changed, resume=True)
    with pytest.raises(ValueError, match="source does not match"):
        loop.ensure_run_config(changed, resume=True, resume_source_commit="3" * 40)
    assert loop.ensure_run_config(changed, resume=True, resume_source_commit=OLD_SOURCE)
    assert [path.read_bytes() for path in paths] == before


@pytest.mark.parametrize("section,name,value", [
    ("model", "qk_head_dim", 4),
    ("optim", "learning_rate", 0.002),
    ("data", "shuffle_seed", 17),
])
def test_source_permission_does_not_allow_scientific_config_changes(tmp_path, section, name, value):
    changed = new_source(initialize(tiny_config(tmp_path)))
    setattr(changed, section, replace(getattr(changed, section), **{name: value}))
    before = (tmp_path / "config.json").read_bytes()
    with pytest.raises(ValueError, match="does not match"):
        loop.ensure_run_config(changed, resume=True, resume_source_commit=OLD_SOURCE)
    assert (tmp_path / "config.json").read_bytes() == before


@pytest.mark.parametrize("prior_source", ["", "1" * 39, "A" * 40, True])
def test_source_permission_requires_full_commit(tmp_path, prior_source):
    changed = new_source(initialize(tiny_config(tmp_path)))
    with pytest.raises(ValueError, match="full prior source"):
        loop.ensure_run_config(changed, resume=True, resume_source_commit=prior_source)


def test_source_permission_cannot_initialize_a_run_or_replace_missing_checkpoint(tmp_path):
    fresh = tiny_config(tmp_path)
    with pytest.raises(ValueError, match="initialized run"):
        loop.ensure_run_config(fresh, resume=True, resume_source_commit=OLD_SOURCE)
    config = initialize(fresh)
    before = (tmp_path / "config.json").read_bytes()
    with pytest.raises(ValueError, match="initialized run"):
        loop.ensure_run_config(new_source(config), resume=False, resume_source_commit=OLD_SOURCE)
    with pytest.raises(ValueError, match="retained resumable checkpoint"):
        loop.run_training(new_source(config), stop_after_step=1, resume_source_commit=OLD_SOURCE)
    assert (tmp_path / "config.json").read_bytes() == before


def test_cli_forwards_explicit_source_resume_environment(tmp_path, monkeypatch):
    from scaffolds.transformer import train

    initialize(tiny_config(tmp_path))
    monkeypatch.setenv("SHARED_TRANSFORMER_SOURCE_COMMIT", NEW_SOURCE)
    monkeypatch.setenv("TRANSFORMER_RESUME_SOURCE_COMMIT", OLD_SOURCE)
    calls = []
    monkeypatch.setattr(train, "run_training", lambda config, **kwargs: calls.append((config, kwargs)))
    train.main([str(tmp_path / "config.yaml")])
    assert calls[0][0].train.source_commit == NEW_SOURCE
    assert calls[0][1]["resume_source_commit"] == OLD_SOURCE
    assert calls[0][1]["resume"] is True


def test_source_transition_preserves_optimizer_rng_data_cursor_and_requeue(tmp_path):
    config = tiny_config(tmp_path / "full")
    config.data.source = "packed"
    for split, count in (("train", 20), ("eval", 4)):
        writer = PackedSequenceWriter(tmp_path / split, sequence_length=5,
                                      num_sequences=count, shard_sequences=3)
        writer.write(np.arange(count * 5).reshape(count, 5) % 16)
        setattr(config.data, f"{split}_manifest", str(writer.finish()))
    config = initialize(config)
    full = loop.run_training(config, audit_checkpoint=True)
    paused_config = initialize(replace(config, train=replace(config.train, workdir=tmp_path / "resume")))
    before = loop.run_training(paused_config, stop_after_step=2, audit_checkpoint=True)
    root = paused_config.train.workdir
    recipe_bytes = (root / "config.yaml").read_bytes()
    changed = new_source(paused_config)
    resumed = loop.run_training(changed, stop_after_step=3, audit_checkpoint=True,
                                resume_source_commit=OLD_SOURCE)
    # Source provenance advances only after the complete checkpoint was restored.
    assert json.loads((root / "config.json").read_text())["train"]["source_commit"] == NEW_SOURCE
    assert (root / "config.yaml").read_bytes() == recipe_bytes
    a, b = before["checkpoint_audit"], resumed["checkpoint_audit"]
    assert a["checkpoint_state_sha256"] == b["start_state_sha256"]
    assert a["next_update"] == b["first_update"]
    assert resumed["restored_step"] == 2
    assert resumed["data_start_sequence"] == 2 * config.data.batch_size
    # Slurm retains the original authorization on subsequent requeues.
    completed = loop.run_training(changed, audit_checkpoint=True, resume_source_commit=OLD_SOURCE)
    assert completed["checkpoint_audit"]["final_state_sha256"] == full["checkpoint_audit"]["final_state_sha256"]
    assert completed["restored_step"] == 3
    assert (root / "config.yaml").read_bytes() == recipe_bytes
