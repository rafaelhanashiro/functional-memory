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


@pytest.mark.parametrize("norm,post_clip,scale,clipped", [
    (4.0, 1.0, 0.25, 1), (0.5, 0.5, 1.0, 0), (0.0, 0.0, 1.0, 0),
])
def test_training_mirror_reports_gradient_clipping(monkeypatch, norm, post_clip, scale, clipped):
    logged = []
    monkeypatch.setattr(tracking, "_log", lambda run, payload, step: logged.append((payload, step)))
    tracking.log_train(None, 10, 2.0, 7.0, 0.001, gradient_norm=norm, gradient_clip_norm=1.0)
    payload, step = logged[0]
    assert step == 10
    assert payload["train/gradient_norm_pre_clip"] == norm
    assert payload["train/gradient_norm_post_clip"] == post_clip
    assert payload["train/gradient_clip_scale"] == scale
    assert payload["train/gradient_clipped"] == clipped


def test_gradient_windows_include_updates_between_log_steps(tmp_path, monkeypatch, capsys):
    import jax.numpy as jnp

    config = tiny_config(tmp_path)
    config.train.log_every = 2
    config.optim.grad_clip_norm = 1.0
    actual_step = loop.accumulated_train_step
    # Inject known diagnostic values without changing the optimizer update.
    # The spike at step 3 must be retained even though that step is not logged.
    norms = {1: 0.5, 2: 1.5, 3: 5.0, 4: 0.0}

    def step_with_known_norm(*args, **kwargs):
        state, loss, numerics, *telemetry_statistics = actual_step(*args, **kwargs)
        return state, loss, numerics.at[1].set(jnp.asarray(norms[int(state.step)])), *telemetry_statistics

    monkeypatch.setattr(loop, "accumulated_train_step", step_with_known_norm)
    logged = []
    monkeypatch.setattr(tracking, "_log", lambda run, payload, step: logged.append((payload, step)))
    loop.run_training(config)
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["train_gradient_norm_count"] for r in records] == [2, 2]
    assert [r["train_gradient_norm_mean"] for r in records] == [1.0, 2.5]
    assert [r["train_gradient_norm_max"] for r in records] == [1.5, 5.0]
    assert [r["train_gradient_clipped_fraction"] for r in records] == [0.5, 0.5]
    assert all(r["gradient_clip_norm"] == 1.0 for r in records)
    train = [(payload, step) for payload, step in logged if "train/nll" in payload]
    assert [step for _, step in train] == [1, 2, 4]
    assert [p["train/gradient_norm_pre_clip"] for p, _ in train] == [0.5, 1.5, 0.0]
    windows = [payload for payload, _ in logged if "val/nll" in payload]
    assert [p["train/gradient_norm_pre_clip_window_max"] for p in windows] == [1.5, 5.0]
    assert [p["train/gradient_clipped_fraction"] for p in windows] == [0.5, 0.5]
    assert "gradient_norm_pre_clip=0.500000 gradient_clip_scale=1.000000" in capsys.readouterr().out


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
