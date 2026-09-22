"""Pilot and CLI overrides must never reuse production checkpoint storage."""
from importlib import import_module

import pytest

from scaffolds.transformer.configs import save_config
from scaffolds.transformer.tests.test_training import tiny_config
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.checkpointing import checkpoint_directory


@pytest.mark.parametrize("study", [
    "polynomial_attention", "learned_quadratic_attention",
    "odd_taylor_attention", "key_dimension_taylor",
])
def test_pilot_overrides_production_checkpoint_directory(tmp_path, monkeypatch, study):
    config = tiny_config(tmp_path / "results/production/runs/run")
    config.train.checkpoint_dir = tmp_path / "checkpoints/production/run"
    config.train.checkpoint_dir.mkdir(parents=True)
    sentinel = config.train.checkpoint_dir / "retained-state"
    sentinel.write_bytes(b"original checkpoint")
    recipe = tmp_path / "recipe.yaml"
    save_config(config, recipe)
    pilot = tmp_path / "pilot"

    def train_pilot(candidate, **kwargs):
        destination = checkpoint_directory(candidate)
        assert destination == pilot / "checkpoints"
        destination.mkdir(parents=True)
        (destination / "pilot-state").write_bytes(b"temporary checkpoint")
        return {"final_step": 2}

    monkeypatch.setattr(loop, "run_training", train_pilot)
    preflight = import_module(f"scaffolds.transformer.studies.{study}.preflight")
    manifest = {"runs": [{"name": "run", "config_path": str(recipe)}]}
    assert preflight.worker_result(manifest, "pilot", "run", pilot, 2)["final_step"] == 2
    assert list(config.train.checkpoint_dir.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"original checkpoint"


def test_cli_new_workdir_requires_explicit_reuse_of_checkpoint_location(tmp_path, monkeypatch):
    from scaffolds.transformer import train

    config = tiny_config(tmp_path / "production")
    config.train.checkpoint_dir = tmp_path / "checkpoints/production"
    recipe = tmp_path / "recipe.yaml"
    save_config(config, recipe)
    captured = []
    monkeypatch.setattr(train, "run_training", lambda cfg, **kwargs: captured.append(checkpoint_directory(cfg)))
    fresh = tmp_path / "fresh"
    train.main([str(recipe), "--workdir", str(fresh)])
    assert captured[-1] == fresh / "checkpoints"
    explicit = tmp_path / "checkpoints/fresh"
    train.main([str(recipe), "--workdir", str(fresh), "--checkpoint-dir", str(explicit)])
    assert captured[-1] == explicit
