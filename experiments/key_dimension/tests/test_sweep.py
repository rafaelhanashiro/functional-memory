from pathlib import Path
import json
import os
import subprocess
import sys

import pytest

from experiments.key_dimension.configs import load_config, config_to_dict
from experiments.key_dimension.preflight import passed_check, record_success
from experiments.key_dimension.sweep import KEY_DIMENSIONS, make_manifest, parameter_count, resolve_runs, submit_sweep


BASE = Path(__file__).resolve().parents[1] / "configs/base.yaml"


def make_test_manifest(tmp_path):
    base = load_config(BASE)
    for name in ("train_manifest", "eval_manifest"):
        path = tmp_path / f"{name}.json"
        path.write_text('{"dataset": "test"}')
        setattr(base.data, name, str(path))
    lock = tmp_path / "execution/experiments/key_dimension/uv.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("test locked dependencies\n")
    return base, make_manifest(base, tmp_path / "project", tmp_path / "execution", "a" * 40)


def test_fixed_backbone_sweep_and_exact_prediction_budget(tmp_path):
    base = load_config(BASE)
    runs = resolve_runs(base, tmp_path, "a" * 40)
    assert [cfg.model.qk_head_dim for _, cfg in runs] == list(KEY_DIMENSIONS)
    assert [parameter_count(cfg.model) for _, cfg in runs] == [
        179590144, 180245504, 181556224, 184177664, 189420544, 199906304, 220877824,
    ]
    reference = config_to_dict(runs[0][1])
    for name, config in runs:
        assert name == f"200m-dk{config.model.qk_head_dim:03d}-seed0"
        assert config.train.workdir == tmp_path / "results/key_dimension" / name
        assert config.model.value_head_dim == 64
        assert config.optim.total_steps * config.data.batch_size * (config.data.sequence_length - 1) == 10000269312
        current = config_to_dict(config)
        for candidate in (reference, current):
            candidate["model"].pop("qk_head_dim", None)
            candidate["train"].pop("workdir", None)
            candidate["train"].pop("wandb_run_name", None)
        assert current == reference
    assert base.model.qk_head_dim == 64
    assert not (tmp_path / "results").exists()


def test_preflight_identity_includes_data_and_dependency_lock(tmp_path):
    base, manifest = make_test_manifest(tmp_path)
    execution_root = Path(manifest["execution_root"])
    lock = execution_root / "experiments/key_dimension/uv.lock"
    lock.write_text("changed locked dependencies\n")
    changed_lock = make_manifest(base, Path(manifest["project_root"]), execution_root, "a" * 40)
    assert changed_lock["identity"] != manifest["identity"]
    Path(base.data.train_manifest).write_text('{"dataset": "changed"}')
    changed_data = make_manifest(base, Path(manifest["project_root"]), execution_root, "a" * 40)
    assert changed_data["identity"] != changed_lock["identity"]


def test_dk256_extension_preserves_recipe_and_submits_only_new_run(tmp_path, monkeypatch):
    base, original = make_test_manifest(tmp_path)
    project = Path(original["project_root"])
    execution = Path(original["execution_root"])
    extension = make_manifest(base, project, execution, "b" * 40, (256,))
    assert len(extension["runs"]) == 1
    run = extension["runs"][0]
    assert run["qk_head_dim"] == 256 and run["value_head_dim"] == 64
    assert run["parameter_count"] == 262820864
    assert run["training_tokens"] == 10000269312
    reference = config_to_dict(resolve_runs(base, project, "b" * 40, (128,))[0][1])
    current = json.loads(json.dumps(run["config"]))
    for config in (reference, current):
        config["model"].pop("qk_head_dim")
        config["train"].pop("workdir")
        config["train"].pop("wandb_run_name")
    assert reference == current
    original_path = project / "results/key_dimension/sweep.json"
    original_path.parent.mkdir(parents=True)
    original_path.write_text(json.dumps(original))
    original_bytes = original_path.read_bytes()
    commands = []

    def fake_submit(command, execution_root, dependency=None):
        commands.append((command, dependency))
        return str(200 + len(commands))

    monkeypatch.setattr("experiments.key_dimension.sweep.submit_job", fake_submit)
    monkeypatch.setattr("experiments.key_dimension.plot.aggregate", lambda manifest: [])
    path = project / "results/key_dimension/sweep-dk256.json"
    submitted = submit_sweep(extension, base, path)
    assert len(commands) == 2
    assert "--job-name=kd-preflight-dk256" in commands[0][0]
    assert "--job-name=kd-256" in commands[1][0]
    assert commands[1][1] == "201" and "--dependency=afterok:201" in commands[1][0]
    assert original_path.read_bytes() == original_bytes
    checkpoint_dir = project / "experiments/key_dimension/checkpoints" / run["name"]
    assert (Path(run["workdir"]) / "checkpoints").resolve() == checkpoint_dir.resolve()
    assert submitted["runs"][0]["checkpoint_dir"] == str(checkpoint_dir.resolve())
    with pytest.raises(ValueError, match="already has submitted jobs"):
        submit_sweep(extension, base, path)


def test_dimensions_must_be_nonempty_and_distinct(tmp_path):
    base = load_config(BASE)
    for dimensions in ((), (256, 256)):
        with pytest.raises(ValueError, match="distinct"):
            resolve_runs(base, tmp_path, "a" * 40, dimensions)


def test_submit_serializes_success_only_dependencies_and_retry_configuration(tmp_path, monkeypatch):
    base, manifest = make_test_manifest(tmp_path)
    commands = []

    def fake_submit(command, execution_root, dependency=None):
        assert execution_root == Path(manifest["execution_root"])
        assert "--requeue" in command and "--gres=gpu:h200:2" in command
        assert not any(argument.startswith("--gpus=") for argument in command)
        output = next(item.removeprefix("--output=") for item in command if item.startswith("--output="))
        assert f"--error={output}" in command
        assert "--open-mode=append" in command
        assert output.endswith("/slurm-%j.out")
        assert Path(output).parent.is_dir(), "Slurm must be able to open logs before launching the script"
        commands.append((command, dependency))
        return str(100 + len(commands))

    monkeypatch.setattr("experiments.key_dimension.sweep.submit_job", fake_submit)
    monkeypatch.setattr("experiments.key_dimension.plot.aggregate", lambda manifest: [])
    submitted = submit_sweep(manifest, base)
    assert len(commands) == 8
    assert commands[0][1] is None
    assert "--time=01:00:00" in commands[0][0]
    for index, (command, dependency) in enumerate(commands[1:], start=1):
        assert dependency == str(100 + index)
        assert f"--dependency=afterok:{dependency}" in command
    for run in submitted["runs"]:
        launch = json.loads((Path(run["workdir"]) / "launch.json").read_text())
        assert launch["slurm_job_id"] == submitted["jobs"][run["name"]]
        assert launch["log_path"].endswith(f"/{run['name']}/slurm-{launch['slurm_job_id']}.out")
        assert (Path(run["workdir"]) / "REPORT.md").exists()
    with pytest.raises(ValueError, match="already has submitted jobs"):
        submit_sweep(manifest, base)


def test_interrupted_preflight_cannot_be_reused_as_success(tmp_path):
    path = tmp_path / "check.json"
    manifest = {"identity": "exact", "source_commit": "a" * 40, "dependency_lock_sha256": "lock"}
    assert not passed_check(path, "exact")
    path.write_text('{"identity":"exact","status":"running"}')
    assert not passed_check(path, "exact")
    record_success(path, manifest)
    assert passed_check(path, "exact")
    assert not passed_check(path, "different")


@pytest.mark.parametrize("manifest_name", ("sweep.json", "sweep-dk256.json"))
def test_launcher_bootstraps_without_python_on_compute_node_path(tmp_path, manifest_name):
    project = tmp_path / "project"
    interpreter = project / ".envs/key_dimension/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    manifest = project / "results/key_dimension" / manifest_name
    manifest.parent.mkdir(parents=True)
    # A deliberately mismatched root causes a controlled bootstrap failure,
    # before any scheduler, GPU, or training command can run.
    manifest.write_text(json.dumps({"project_root": str(tmp_path / "wrong-project")}))
    environment = dict(os.environ, PATH=str(tmp_path / "empty-path"), BASH_ENV="")
    result = subprocess.run(
        ["/bin/bash", str(BASE.parents[1] / "launch.sbatch"), "preflight", str(manifest)],
        env=environment, text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert "project_root does not match its canonical location" in result.stderr
    assert "command not found" not in result.stderr
