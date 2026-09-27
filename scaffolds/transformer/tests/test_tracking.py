"""The W&B mirror: naming, metric payloads, and an offline training run."""
import json
from pathlib import Path

import pytest

from scaffolds.transformer.configs import Config
from scaffolds.transformer.tests.test_training import tiny_config
from scaffolds.transformer.training import loop, tracking


def test_experiment_workdir_names_group_run_and_log_directory(tmp_path):
    config = Config()
    config.train.workdir = tmp_path / "experiments/my_experiment/runs/200m-seed0"
    settings = tracking.run_settings(config)
    assert settings["group"] == "my_experiment"
    assert settings["run_name"] == "200m-seed0"
    assert settings["directory"] == config.train.workdir
    assert (settings["entity"], settings["project"]) == ("rafah-massachusetts-institute-of-technology",
                                                         "functional-memory")


def test_explicit_settings_and_non_experiment_workdirs(tmp_path):
    config = Config()
    config.train.workdir = tmp_path / "scratch"
    assert tracking.run_settings(config)["group"] is None
    assert tracking.run_settings(config)["directory"] == tmp_path / "scratch"
    config.train.wandb_group, config.train.wandb_run_name, config.train.wandb_dir = "g", "n", tmp_path / "w"
    settings = tracking.run_settings(config)
    assert (settings["group"], settings["run_name"], settings["directory"]) == ("g", "n", tmp_path / "w")


def test_attention_metrics_use_kernel_layer_prefixes():
    diagnostics = {
        "attention_kernel": "learned_quadratic",
        "layers": [{"layer": 0, "kernel_max": 2.5, "invalid_kernel_count": 0, "note": "text"},
                   {"layer": 1, "kernel_max": 3.5, "finite": True}],
        "coefficients": {"effective_coefficient_names": ["c0", "c1"], "effective": [1.0, 0.5]},
    }
    assert tracking.attention_metrics(diagnostics) == {
        "attention/learned_quadratic/layer00/kernel_max": 2.5,
        "attention/learned_quadratic/layer00/invalid_kernel_count": 0,
        "attention/learned_quadratic/layer01/kernel_max": 3.5,
        "attention/learned_quadratic/coefficients/c0": 1.0,
        "attention/learned_quadratic/coefficients/c1": 0.5,
    }


def test_tags_name_kernel_width_degree_and_hardware():
    config = Config()
    config.model.attention_degree = 4
    tags = tracking.run_tags(config, {"gpu_type": "B300", "gpu_count": 4, "cluster": "kennel"})
    assert tags == ["taylor", f"dk{config.model.qk_head_dim}", "m4", "qk_norm", "4xB300", "kennel"]


def test_offline_training_mirrors_metrics_and_pins_the_run(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_CONFIG_DIR", str(tmp_path / "wandb-config"))
    monkeypatch.setenv("WANDB_CACHE_DIR", str(tmp_path / "wandb-cache"))
    workdir = tmp_path / "experiments/demo/runs/tiny"
    config = tiny_config(workdir)
    config.train.wandb_mode = "offline"
    result = loop.run_training(config)
    assert result["final_step"] == 4
    identity = json.loads((workdir / "wandb_run.json").read_text())
    assert identity["run_name"] == "tiny" and identity["project"] == "functional-memory"
    records = [json.loads(line) for line in (workdir / "metrics.jsonl").read_text().splitlines()]
    assert {record["wandb_run_id"] for record in records} == {identity["run_id"]}
    runs = list((workdir / "wandb").glob("offline-run-*"))
    assert len(runs) == 1 and runs[0].name.endswith(identity["run_id"])
