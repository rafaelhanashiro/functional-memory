"""Isolation and timing boundaries of the disposable fresh-training benchmark."""
from types import SimpleNamespace
import json

import numpy as np
import pytest
import yaml

from scaffolds.transformer.benchmarks import benchmark_training_throughput as benchmark
from scaffolds.transformer.configs import Config, config_to_dict


def setup_trial(tmp_path, monkeypatch):
    config = Config()
    config.model.initialization = "transformer_reference"
    config.optim.total_steps = 100
    config.train.micro_batch_size = config.data.batch_size
    config.train.eval_every = config.train.ckpt_every = 1
    source = tmp_path / "input.yaml"
    source.write_text(yaml.safe_dump(config_to_dict(config)))
    args = SimpleNamespace(config=source, scratch=tmp_path / "scratch", output=tmp_path / "report.json",
                           warmup=2, iterations=3)
    monkeypatch.setattr(benchmark.jax, "local_devices", lambda: [SimpleNamespace(
        platform="gpu", memory_stats=lambda: {"peak_bytes_in_use": 123})])
    monkeypatch.setattr(benchmark, "_execution_provenance", lambda *_: {"source_commit": "a" * 40})
    monkeypatch.setattr(benchmark.jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(benchmark.loop, "train_step", lambda step: (
        SimpleNamespace(step=step), 2.0, np.array([2.0, 1.0, 3.0])))
    return args, config


def test_fresh_trial_preserves_recipe_and_excludes_warmup_and_checkpoint(tmp_path, monkeypatch):
    args, original = setup_trial(tmp_path, monkeypatch)
    original_bytes = args.config.read_bytes()
    original_function = benchmark.loop.train_step

    def run(config, *, resume, stop_after_step):
        assert resume is False and stop_after_step == 5
        assert config.model == original.model and config.data == original.data and config.optim == original.optim
        assert config.train.micro_batch_size == original.train.micro_batch_size
        assert config.train.eval_every > 5 and config.train.ckpt_every > 5
        assert config.train.wandb_mode == "disabled"
        assert config.train.checkpoint_dir == args.scratch / "checkpoints"
        config.train.workdir.mkdir()
        (config.train.workdir / "metadata.json").write_text(json.dumps({"parameter_count": 42}))
        (config.train.workdir / "data_identity.json").write_text(json.dumps({"dataset_sha256": "test"}))
        for step in range(1, 6):
            benchmark.loop.train_step(step)
        # A checkpoint/finish cost occurs after the last measured completion.
        return {"final_step": 5, "timing": {"checkpoint_seconds": [100.]}}

    monkeypatch.setattr(benchmark.loop, "run_training", run)
    report = benchmark.benchmark(args)
    assert report["status"] == "complete" and report["all_finite"]
    assert report["parameter_count"] == 42
    assert report["data_identity"] == {"dataset_sha256": "test"}
    assert len(report["samples"]) == 5 and report["steady_training"]["updates"] == 3
    assert report["samples"][0]["interval_seconds"] is None
    assert report["scratch_removed"] and not args.scratch.exists()
    assert args.config.read_bytes() == original_bytes
    assert benchmark.loop.train_step is original_function
    assert json.loads(args.output.read_text())["scratch_removed"]


def test_failed_update_retains_scratch_and_restores_training_functions(tmp_path, monkeypatch):
    args, _ = setup_trial(tmp_path, monkeypatch)
    monkeypatch.setattr(benchmark.loop, "train_step", lambda step: (
        SimpleNamespace(step=step), float("nan"), np.array([float("nan"), float("inf"), 3.])))
    original_function = benchmark.loop.train_step

    def fail(config, **_):
        config.train.workdir.mkdir()
        benchmark.loop.train_step(1)
        raise FloatingPointError("bad update")

    monkeypatch.setattr(benchmark.loop, "run_training", fail)
    with pytest.raises(FloatingPointError, match="bad update"):
        benchmark.benchmark(args)
    report = json.loads(args.output.read_text())
    assert report["status"] == "failed" and args.scratch.exists()
    assert report["samples"][0]["loss"] is None
    assert report["samples"][0]["update_numerics"] == [None, None, 3.]
    assert not report["samples"][0]["finite_loss_and_update_numerics"]
    assert benchmark.loop.train_step is original_function


def test_report_cannot_be_deleted_with_scratch(tmp_path, monkeypatch):
    args, _ = setup_trial(tmp_path, monkeypatch)
    args.output = args.scratch / "report.json"
    with pytest.raises(ValueError, match="Scratch"):
        benchmark.benchmark(args)
    assert not args.scratch.exists()
