from pathlib import Path
import json
import os
import subprocess
import sys

import pytest

from experiments.polynomial_attention.configs import load_config, config_to_dict
from experiments.polynomial_attention.preflight import (
    PREFLIGHT_CHECKS, passed_check, preflight_complete, record_success,
    require_passing_tests, run_preflight,
)
from experiments.polynomial_attention.sweep import (
    ATTENTION_DEGREES, make_manifest, parameter_count, prepare_sweep,
    resolve_runs, run_name, submit_preflight, submit_sweep,
)

BASE = Path(__file__).resolve().parents[1] / "configs/base.yaml"


def make_test_manifest(tmp_path):
    base = load_config(BASE)
    for name in ("train_manifest", "eval_manifest"):
        path = tmp_path / f"{name}.json"
        path.write_text('{"dataset": "test"}')
        setattr(base.data, name, str(path))
    lock = tmp_path / "execution/experiments/polynomial_attention/uv.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("test locked dependencies\n")
    return base, make_manifest(base, tmp_path / "project", tmp_path / "execution", "a" * 40)


def test_fixed_backbone_sweep_and_exact_prediction_budget(tmp_path):
    base = load_config(BASE)
    runs = resolve_runs(base, tmp_path, "a" * 40)
    assert [cfg.model.attention_degree for _, cfg in runs] == list(ATTENTION_DEGREES)
    assert [parameter_count(cfg.model) for _, cfg in runs] == [199906304] * 6
    reference = None
    for name, config in runs:
        assert name == run_name(config.model.attention_degree)
        assert config.train.workdir == tmp_path / "results/polynomial_attention" / name
        assert config.model.qk_head_dim == config.model.value_head_dim == 64
        assert config.model.attention_backend == "blocked"
        assert config.optim.total_steps * config.data.batch_size * (config.data.sequence_length - 1) == 10000269312
        current = config_to_dict(config)
        current["model"].pop("attention_degree")
        current["train"].pop("workdir")
        current["train"].pop("wandb_run_name")
        if reference is None:
            reference = current
        assert current == reference
    assert base.model.attention_degree is None
    assert not (tmp_path / "results").exists()


def test_preflight_identity_includes_data_and_dependency_lock(tmp_path):
    base, manifest = make_test_manifest(tmp_path)
    execution_root = Path(manifest["execution_root"])
    lock = execution_root / "experiments/polynomial_attention/uv.lock"
    lock.write_text("changed locked dependencies\n")
    changed_lock = make_manifest(base, Path(manifest["project_root"]), execution_root, "a" * 40)
    assert changed_lock["identity"] != manifest["identity"]
    Path(base.data.train_manifest).write_text('{"dataset": "changed"}')
    changed_data = make_manifest(base, Path(manifest["project_root"]), execution_root, "a" * 40)
    assert changed_data["identity"] != changed_lock["identity"]


def install_fake_scheduler(monkeypatch, manifest):
    commands = []

    def fake_submit(command, execution_root, dependency=None, dependency_type="afterok"):
        assert execution_root == Path(manifest["execution_root"])
        assert "--requeue" in command and "--gres=gpu:h200:2" in command
        assert not any(argument.startswith("--gpus=") for argument in command)
        output = next(item.removeprefix("--output=") for item in command if item.startswith("--output="))
        assert f"--error={output}" in command and "--open-mode=append" in command
        assert output.endswith("/slurm-%j.out")
        assert Path(output).parent.is_dir()
        commands.append((command, dependency, dependency_type))
        return str(100 + len(commands))

    monkeypatch.setattr("experiments.polynomial_attention.sweep.submit_job", fake_submit)
    monkeypatch.setattr("experiments.polynomial_attention.plot.aggregate", lambda manifest: [])
    monkeypatch.setattr("experiments.polynomial_attention.preflight.verify_manifest", lambda manifest: None)
    return commands


@pytest.mark.parametrize("launch", [False, True])
def test_only_preflight_is_submitted_until_all_checks_pass(tmp_path, monkeypatch, launch):
    base, manifest = make_test_manifest(tmp_path)
    commands = install_fake_scheduler(monkeypatch, manifest)
    submitted = submit_preflight(manifest, base, launch=launch)
    assert len(commands) == 1
    assert commands[0][1] is None
    assert commands[0][0][-2] == ("preflight-and-submit" if launch else "preflight")
    assert submitted["jobs"] == {"preflight": "101"}
    assert submitted["launch_after_preflight"] is launch
    with pytest.raises(RuntimeError, match="no successful matching complete preflight"):
        submit_sweep(manifest, base)
    assert len(commands) == 1
    root = Path(manifest["preflight_dir"])
    record_success(root / "success.json", manifest)
    with pytest.raises(RuntimeError, match="no successful matching complete preflight"):
        submit_sweep(manifest, base)
    for name in PREFLIGHT_CHECKS:
        record_success(root / "checks" / f"{name}.json", manifest)
    submitted = submit_sweep(manifest, base)
    assert len(commands) == 7
    for index, (command, dependency, dependency_type) in enumerate(commands[1:], start=1):
        assert dependency == str(100 + index)
        assert dependency_type == ("afterok" if index == 1 else "afterany")
        assert f"--dependency={dependency_type}:{dependency}" in command
    for run in submitted["runs"]:
        launch_record = json.loads((Path(run["workdir"]) / "launch.json").read_text())
        assert launch_record["slurm_job_id"] == submitted["jobs"][run["name"]]
        assert launch_record["log_path"].endswith(f"/{run['name']}/slurm-{launch_record['slurm_job_id']}.out")
        assert (Path(run["workdir"]) / "REPORT.md").exists()
    # Repeating a production submission resumes its manifest, without duplicates.
    submit_sweep(manifest, base)
    assert len(commands) == 7
    with pytest.raises(ValueError, match="already has a preflight job"):
        submit_preflight(manifest, base, launch=launch)


def test_interrupted_preflight_cannot_be_reused_as_success(tmp_path):
    path = tmp_path / "check.json"
    manifest = {"identity": "exact", "source_commit": "a" * 40, "dependency_lock_sha256": "lock"}
    assert not passed_check(path, "exact")
    path.write_text('{"identity":"exact","status":"running"}')
    assert not passed_check(path, "exact")
    record_success(path, manifest)
    assert passed_check(path, "exact")
    assert not passed_check(path, "different")


@pytest.mark.parametrize("kind", ["skipped", "failure", "error", "too-few"])
def test_parity_gate_rejects_skips_failures_and_missing_checks(tmp_path, kind):
    xml = tmp_path / "checks.xml"
    inner = "" if kind == "too-few" else f"<{kind}/>"
    xml.write_text(f'<testsuite><testcase name="case">{inner}</testcase></testsuite>')
    with pytest.raises(RuntimeError, match="passing checks"):
        require_passing_tests(xml, minimum=6)


def test_preflight_pilots_all_degrees_in_processes_and_resumes_controls(tmp_path, monkeypatch):
    base, manifest = make_test_manifest(tmp_path)
    prepare_sweep(manifest, base)
    monkeypatch.setattr("experiments.polynomial_attention.preflight.verify_manifest", lambda manifest: None)
    calls = []

    def fake_worker(manifest, worker, output, *, name="", workdir=None, stop=2, cpu=False):
        calls.append((worker, name, stop, cpu))
        if worker == "hardware":
            return {"devices": ["H200", "H200"]}
        if worker == "initialization":
            return {"parameter_count": 199906304, "parameter_sha256": "same"}
        assert workdir is not None
        return {"final_step": stop, "restored_step": 2 if stop == 3 else None,
                "data_parallel_devices": 2, "data_start_sequence": 512 if stop == 3 else 0}

    def fake_pytest(command, **kwargs):
        xml = Path(next(argument.removeprefix("--junitxml=") for argument in command if argument.startswith("--junitxml=")))
        xml.write_text('<testsuite>' + '<testcase name="check"/>' * 7 + '</testsuite>')

    monkeypatch.setattr("experiments.polynomial_attention.preflight.launch_worker", fake_worker)
    monkeypatch.setattr("experiments.polynomial_attention.preflight.subprocess.run", fake_pytest)
    monkeypatch.setattr("experiments.polynomial_attention.plot.committed_records", lambda workdir: [{"step": 2, "validation_nll": 3.0, "validation_perplexity": 20.0}])
    run_preflight(manifest)
    assert preflight_complete(manifest)
    initial = [name for worker, name, _, cpu in calls if worker == "initialization" and cpu]
    assert initial == ["original", *[run_name(m) for m in ATTENTION_DEGREES]]
    pilots = [(name, stop) for worker, name, stop, _ in calls if worker == "pilot"]
    assert [name for name, stop in pilots if stop == 2] == [run_name(m) for m in ATTENTION_DEGREES]
    assert [name for name, stop in pilots if stop == 3] == [run_name(16), run_name(None)]
    previous = list(calls)
    run_preflight(manifest, check_only=True)
    assert calls == previous


def test_launcher_bootstraps_without_python_on_compute_node_path(tmp_path):
    project = tmp_path / "project"
    interpreter = project / ".envs/polynomial_attention/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    manifest = project / "results/polynomial_attention/sweep.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"project_root": str(tmp_path / "wrong-project")}))
    environment = dict(os.environ, PATH=str(tmp_path / "empty-path"), BASH_ENV="")
    result = subprocess.run(
        ["/bin/bash", str(BASE.parents[1] / "launch.sbatch"), "preflight", str(manifest)],
        env=environment, text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert "project_root does not match its canonical location" in result.stderr
    assert "command not found" not in result.stderr


def test_failed_launcher_preflight_never_submits_production(tmp_path):
    project = tmp_path / "project"
    interpreter = project / ".envs/polynomial_attention/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    execution = tmp_path / "execution"
    execution.mkdir()
    manifest_path = project / "results/polynomial_attention/sweep.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({"project_root": str(project), "execution_root": str(execution), "source_commit": "abc", "runs": []}))
    commands = tmp_path / "commands"
    commands.mkdir()
    git = commands / "git"
    git.write_text('#!/bin/bash\nif [[ "$1" == rev-parse ]]; then echo abc; fi\n')
    git.chmod(0o755)
    calls = tmp_path / "uv-calls"
    uv = commands / "uv"
    uv.write_text(f'#!/bin/bash\necho "JAX_PLATFORMS=$JAX_PLATFORMS $*" >> "{calls}"\nexit 19\n')
    uv.chmod(0o755)
    result = subprocess.run(
        ["/bin/bash", str(BASE.parents[1] / "launch.sbatch"), "preflight-and-submit", str(manifest_path)],
        env=dict(os.environ, PATH=f"{commands}:/usr/bin:/bin", SLURM_JOB_ID="42", BASH_ENV="", JAX_PLATFORMS="cpu"),
        capture_output=True, text=True,
    )
    assert result.returncode == 19
    lines = calls.read_text().splitlines()
    assert len(lines) == 1 and "polynomial_attention.preflight" in lines[0]
    assert lines[0].startswith("JAX_PLATFORMS=cuda ")
    assert "polynomial_attention.sweep" not in lines[0]
