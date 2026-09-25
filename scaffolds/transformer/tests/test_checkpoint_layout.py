"""Pilot and CLI overrides must never reuse production checkpoint storage."""
import pytest

from scaffolds.transformer.configs import save_config
from scaffolds.transformer.tests.test_training import tiny_config
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.checkpointing import checkpoint_directory


def test_acceptance_pilot_never_uses_production_checkpoint_directory(tmp_path, monkeypatch):
    from argparse import Namespace

    from scaffolds.transformer import acceptance

    config = tiny_config(tmp_path / "experiments/production/runs/run")
    config.train.checkpoint_dir = tmp_path / "experiments/production/checkpoints/run"
    config.train.checkpoint_dir.mkdir(parents=True)
    sentinel = config.train.checkpoint_dir / "retained-state"
    sentinel.write_bytes(b"original checkpoint")
    recipe = tmp_path / "recipe.yaml"
    save_config(config, recipe)
    directory = tmp_path / "acceptance"

    def train_pilot(candidate, **kwargs):
        destination = checkpoint_directory(candidate)
        assert destination == directory / "pilot" / "checkpoints"
        destination.mkdir(parents=True)
        (destination / "pilot-state").write_bytes(b"temporary checkpoint")
        return {"final_step": 2}

    monkeypatch.setattr(loop, "run_training", train_pilot)
    spec = acceptance.AcceptanceSpec(entry_module="unused", cpu_tests=(), pilot_steps=2)
    args = Namespace(worker="pilot", config=recipe, directory=directory, stop_after_step=2)
    assert acceptance._worker(spec, args, {"source_commit": "a" * 40})["final_step"] == 2
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
