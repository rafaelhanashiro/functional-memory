from pathlib import Path
import hashlib
import json
import os
import shlex
import subprocess
import sys

import pytest

from experiments.polynomial_attention.configs import load_config, config_to_dict
from experiments.polynomial_attention.preflight import (
    PREFLIGHT_CHECKS, passed_check, preflight_complete, record_success,
    require_passing_tests, required_checks, run_preflight,
)
from experiments.polynomial_attention.sweep import (
    ATTENTION_DEGREES, make_manifest, parameter_count, prepare_sweep,
    resolve_runs, resolve_scheduler_profile, run_name, scheduler_profile,
    submit_preflight, submit_sweep,
)

BASE = Path(__file__).resolve().parents[1] / "configs/base.yaml"


def make_test_manifest(tmp_path, *, gpu_count=2, backend="blocked"):
    base = load_config(BASE)
    base.model.attention_backend = backend
    for name in ("train_manifest", "eval_manifest"):
        path = tmp_path / f"{name}.json"
        path.write_text('{"dataset": "test"}')
        setattr(base.data, name, str(path))
    lock = tmp_path / "execution/experiments/polynomial_attention/uv.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("test locked dependencies\n")
    return base, make_manifest(base, tmp_path / "project", tmp_path / "execution", "a" * 40,
                               scheduler=scheduler_profile(gpu_count))


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


def test_fused_backend_propagates_to_every_run_and_changes_identity(tmp_path):
    base, blocked = make_test_manifest(tmp_path)
    base.model.attention_backend = "fused"
    fused = make_manifest(base, Path(blocked["project_root"]), Path(blocked["execution_root"]), blocked["source_commit"])
    assert all(run["config"]["model"]["attention_backend"] == "fused" for run in fused["runs"])
    assert fused["identity"] != blocked["identity"]
    root = Path(fused["preflight_dir"])
    record_success(root / "success.json", fused)
    for name in PREFLIGHT_CHECKS:
        record_success(root / "checks" / f"{name}.json", fused)
    assert not preflight_complete(fused)
    record_success(root / "checks/attention-fused-gpu.json", fused)
    assert preflight_complete(fused)


def test_allocation_changes_identity_without_changing_recipe(tmp_path):
    base, manifest = make_test_manifest(tmp_path)
    changed = make_manifest(base, Path(manifest["project_root"]), Path(manifest["execution_root"]),
                            manifest["source_commit"], scheduler=scheduler_profile(4))
    assert changed["identity"] != manifest["identity"]
    assert changed["preflight_dir"] != manifest["preflight_dir"]
    assert changed["runs"] == manifest["runs"]
    assert changed["scheduler"]["gpu_count"] == 4


def test_followup_submission_preserves_persisted_allocation(tmp_path):
    base, manifest = make_test_manifest(tmp_path, gpu_count=4)
    project_root = Path(manifest["project_root"])
    assert resolve_scheduler_profile(project_root) == scheduler_profile(2)
    prepare_sweep(manifest, base)
    assert resolve_scheduler_profile(project_root) == scheduler_profile(4)
    assert resolve_scheduler_profile(project_root, 4) == scheduler_profile(4)
    with pytest.raises(ValueError, match="differs from the persisted sweep"):
        resolve_scheduler_profile(project_root, 2)


def test_legacy_manifest_is_not_silently_reused(tmp_path):
    manifest_path = tmp_path / "results/polynomial_attention/sweep.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text('{"jobs":{"preflight":"123"}}')
    with pytest.raises(ValueError, match="predates the Kennel scheduler profile"):
        resolve_scheduler_profile(tmp_path)


@pytest.mark.parametrize("gpu_count", [0, 1, 3, 8, True, "2"])
def test_unsupported_allocations_are_rejected(gpu_count):
    with pytest.raises(ValueError, match="GPU count must be 2 or 4"):
        scheduler_profile(gpu_count)


def install_fake_scheduler(monkeypatch, manifest):
    commands = []

    def fake_submit(command, execution_root, dependency=None, dependency_type="afterok"):
        assert execution_root == Path(manifest["execution_root"])
        assert "--requeue" in command
        assert f"--gres=gpu:h100:{manifest['scheduler']['gpu_count']}" in command
        assert "--partition=long" in command
        assert "--nodes=1" in command and "--ntasks=1" in command
        assert "--cpus-per-task=32" in command and "--mem=256G" in command
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
@pytest.mark.parametrize("gpu_count", [2, 4])
def test_only_preflight_is_submitted_until_all_checks_pass(tmp_path, monkeypatch, launch, gpu_count):
    base, manifest = make_test_manifest(tmp_path, gpu_count=gpu_count)
    commands = install_fake_scheduler(monkeypatch, manifest)
    submitted = submit_preflight(manifest, base, launch=launch)
    assert len(commands) == 1
    assert commands[0][1] is None
    assert "--time=08:00:00" in commands[0][0]
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
        assert "--time=7-00:00:00" in command
        assert dependency == str(100 + index)
        assert dependency_type == ("afterok" if index == 1 else "afterany")
        assert f"--dependency={dependency_type}:{dependency}" in command
    for run in submitted["runs"]:
        launch_record = json.loads((Path(run["workdir"]) / "launch.json").read_text())
        assert launch_record["slurm_job_id"] == submitted["jobs"][run["name"]]
        assert launch_record["scheduler"] == manifest["scheduler"]
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


@pytest.mark.parametrize("gpu_count", [2, 4])
@pytest.mark.parametrize("backend", ["blocked", "fused"])
@pytest.mark.parametrize("retain_zero", [False, True])
def test_preflight_pilots_active_degrees_in_processes_and_resumes_controls(tmp_path, monkeypatch, gpu_count, backend, retain_zero):
    base, manifest = make_test_manifest(tmp_path, gpu_count=gpu_count, backend=backend)
    prepare_sweep(manifest, base)
    if retain_zero:
        manifest["runs"][0]["retained_completed"] = True
    monkeypatch.setattr("experiments.polynomial_attention.preflight.verify_manifest", lambda manifest: None)
    calls = []
    test_commands = []

    def fake_worker(manifest, worker, output, *, name="", workdir=None, stop=2, cpu=False):
        calls.append((worker, name, stop, cpu))
        if worker == "hardware":
            return {"devices": ["H100"] * gpu_count}
        if worker == "initialization":
            return {"parameter_count": 199906304, "parameter_sha256": "same"}
        assert workdir is not None
        return {"final_step": stop, "restored_step": 2 if stop == 3 else None,
                "data_parallel_devices": gpu_count, "data_start_sequence": 512 if stop == 3 else 0,
                "timing": {"runtime_seconds": 20., "train_step_seconds": [10., 1.],
                           "evaluation_seconds": [2.], "checkpoint_seconds": [1.]}}

    def fake_pytest(command, **kwargs):
        test_commands.append(command)
        xml = Path(next(argument.removeprefix("--junitxml=") for argument in command if argument.startswith("--junitxml=")))
        xml.write_text('<testsuite>' + '<testcase name="check"/>' * 7 + '</testsuite>')

    monkeypatch.setattr("experiments.polynomial_attention.preflight.launch_worker", fake_worker)
    monkeypatch.setattr("experiments.polynomial_attention.preflight.subprocess.run", fake_pytest)
    monkeypatch.setattr("experiments.polynomial_attention.plot.committed_records", lambda workdir: [{"step": 2, "validation_nll": 3.0, "validation_perplexity": 20.0}])
    run_preflight(manifest)
    assert preflight_complete(manifest)
    fused_test_path = "experiments/polynomial_attention/tests/test_fused_attention.py"
    assert any(fused_test_path in command for command in test_commands) == (backend == "fused")
    assert len(test_commands) == (3 if backend == "fused" else 2)
    active_names = [run_name(m) for m in ATTENTION_DEGREES if not (retain_zero and m == 0)]
    initial = [name for worker, name, _, cpu in calls if worker == "initialization" and cpu]
    assert initial == ["original", *active_names]
    pilots = [(name, stop) for worker, name, stop, _ in calls if worker == "pilot"]
    assert [name for name, stop in pilots if stop == 2] == active_names
    assert [name for name, stop in pilots if stop == 3] == [run_name(16), run_name(None)]
    previous = list(calls)
    run_preflight(manifest, check_only=True)
    assert calls == previous


def test_retained_control_does_not_replace_active_preflight_gate(tmp_path):
    _, manifest = make_test_manifest(tmp_path, backend="fused")
    manifest["runs"][0]["retained_completed"] = True
    manifest["runs"][0]["config"]["model"]["attention_backend"] = "blocked"
    checks = required_checks(manifest)
    assert run_name(0) not in checks
    assert checks == ("attention-cpu", "attention-gpu", "initialization",
                      *[run_name(m) for m in ATTENTION_DEGREES[1:]], "attention-fused-gpu")
    root = Path(manifest["preflight_dir"])
    record_success(root / "success.json", manifest)
    for name in checks:
        if name != run_name(2):
            record_success(root / "checks" / f"{name}.json", manifest)
    record_success(root / "checks" / f"{run_name(0)}.json", manifest)
    assert not preflight_complete(manifest)
    record_success(root / "checks" / f"{run_name(2)}.json", manifest)
    assert preflight_complete(manifest)


@pytest.mark.parametrize("retained", [False, True])
def test_preflight_accepts_historical_source_only_for_validated_retained_run(tmp_path, monkeypatch, retained):
    from experiments.polynomial_attention.configs import save_config
    from experiments.polynomial_attention.preflight import verify_manifest
    from experiments.polynomial_attention.sweep import manifest_identity_payload

    base, manifest = make_test_manifest(tmp_path)
    prepare_sweep(manifest, base)
    run = manifest["runs"][0]
    config = load_config(run["config_path"])
    config.train.source_commit = "b" * 40
    save_config(config, Path(run["config_path"]))
    run["config"] = config_to_dict(config)
    if retained:
        manifest["schema_version"] = 3
        run["retained_completed"] = True
        run["retained_completion_sha256"] = "c" * 64
    manifest["identity"] = hashlib.sha256(json.dumps(manifest_identity_payload(manifest), sort_keys=True).encode()).hexdigest()
    monkeypatch.setattr("experiments.polynomial_attention.preflight.subprocess.check_output",
                        lambda command, **kwargs: manifest["source_commit"] if "rev-parse" in command else "")
    validated = []
    monkeypatch.setattr("experiments.polynomial_attention.sweep.validate_retained_run", validated.append)
    if retained:
        verify_manifest(manifest)
        assert validated == [run]
    else:
        with pytest.raises(ValueError, match="Active run source"):
            verify_manifest(manifest)
        assert not validated


def test_aggregate_preserves_retained_control_source_and_rejects_wrong_metrics(tmp_path):
    from experiments.polynomial_attention.plot import aggregate
    from experiments.polynomial_attention.tests.test_plot import fixture_run, write_completion

    retained, workdir = fixture_run(tmp_path, 0)
    retained["retained_completed"] = True
    retained["config"]["train"] = {"source_commit": "a" * 40}
    write_completion(retained, workdir)
    manifest = {"source_commit": "b" * 40, "identity": "mixed-sources",
                "project_root": str(tmp_path), "runs": [retained]}
    output = tmp_path / "plots"
    rows = aggregate(manifest, output)
    assert rows[0]["status"] == "complete"
    assert rows[0]["source_commit"] == "a" * 40
    assert "Retained results from earlier source revisions" in (output / "README.md").read_text()
    retained["config"]["train"]["source_commit"] = "b" * 40
    with pytest.raises(ValueError, match="Metric identity"):
        aggregate(manifest, output)


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


@pytest.mark.parametrize("preflight_status", [0, 19])
def test_launcher_preserves_allocation_and_submits_only_after_success(tmp_path, preflight_status):
    project = tmp_path / "project"
    interpreter = project / ".envs/polynomial_attention/bin/python"
    interpreter.parent.mkdir(parents=True)
    calls = tmp_path / "python-calls"
    interpreter.write_text(
        '#!/bin/bash\n'
        f'if [[ "$1" == - ]]; then exec {shlex.quote(sys.executable)} "$@"; fi\n'
        f'echo "JAX_PLATFORMS=$JAX_PLATFORMS $*" >> {shlex.quote(str(calls))}\n'
        f'exit {preflight_status}\n'
    )
    interpreter.chmod(0o755)
    execution = tmp_path / "execution"
    execution.mkdir()
    manifest_path = project / "results/polynomial_attention/sweep.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({"project_root": str(project), "execution_root": str(execution),
                                        "source_commit": "abc", "scheduler": scheduler_profile(4),
                                        "runs": [{"name": "200m-m000-seed0", "config_path": str(project / "config.yaml")}]}))
    commands = tmp_path / "commands"
    commands.mkdir()
    git = commands / "git"
    git.write_text('#!/bin/bash\nif [[ "$1" == rev-parse ]]; then echo abc; fi\n')
    git.chmod(0o755)
    result = subprocess.run(
        ["/bin/bash", str(BASE.parents[1] / "launch.sbatch"), "preflight-and-submit", str(manifest_path)],
        env=dict(os.environ, PATH=f"{commands}:/usr/bin:/bin", SLURM_JOB_ID="42", BASH_ENV="", JAX_PLATFORMS="cpu"),
        capture_output=True, text=True,
    )
    assert result.returncode == preflight_status
    lines = calls.read_text().splitlines()
    assert "polynomial_attention.preflight" in lines[0]
    assert lines[0].startswith("JAX_PLATFORMS=cuda ")
    assert "polynomial_attention.sweep" not in lines[0]
    if preflight_status:
        assert len(lines) == 1
    else:
        assert len(lines) == 2
        assert "polynomial_attention.sweep --submit --gpu-count 4" in lines[1]
