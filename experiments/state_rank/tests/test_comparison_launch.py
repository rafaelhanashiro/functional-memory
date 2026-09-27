"""Offline checks of serial Slurm submission and durable submission bookkeeping."""

import json
from pathlib import Path
import subprocess

import pytest

from experiments.state_rank import launch_comparison
from scaffolds.records import create_run, load_run
from scaffolds.state_rank import compare_random
from scaffolds.state_rank.tests.test_compare_random import comparison_project


@pytest.fixture
def launch_fixture(tmp_path, monkeypatch):
    execution = tmp_path / "execution"
    project = tmp_path / "retained"
    execution.mkdir()
    project.mkdir()
    manifest_path = project / "manifest.json"
    manifest = {
        "source_commit": "a" * 40,
        "execution_root": str(execution),
        "project_root": str(project),
        "conditions": [{"name": "pretrained", "weights": "pretrained", "seed": None}] + [
            {"name": f"random-seed{seed}", "weights": "random", "seed": seed} for seed in range(5)
        ],
        "jobs": {},
        "config": {"seeds": list(range(5)), "output_directory": "allocated-by-create-run"},
        "dependency_lock_sha256": "lock", "inputs_sha256": "inputs",
        "baseline_metadata_sha256": "metadata", "baseline_spectra_sha256": "spectra",
    }
    study = execution / "experiments/state_rank"
    manifest["run_directories"] = {
        condition["name"]: str(create_run(study, {**manifest["config"], "condition": condition}))
        for condition in manifest["conditions"]
    }
    manifest["run_config_sha256"] = {
        name: load_run(path)["config"]["sha256"] for name, path in manifest["run_directories"].items()
    }
    manifest["identity"] = launch_comparison.manifest_identity(manifest)
    manifest_path.write_text(json.dumps(manifest))
    frozen = {"changed": False, "calls": [], "committed": True}
    submissions = []
    handler = {"sbatch": None}

    def fake_run(command, **kwargs):
        if command[:2] == ["python3", "scaffolds/launch/frozen.py"]:
            assert kwargs["cwd"] == execution and command[command.index("--launch-commit") + 1] == manifest["source_commit"]
            frozen["calls"].append(command)
            if frozen["changed"]:
                raise subprocess.CalledProcessError(1, command, stderr="Frozen execution set changed since launch")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["git", "cat-file", "-e"]:
            if not frozen["committed"]:
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if handler["sbatch"] is None:
            raise AssertionError(f"Unexpected real submission attempt: {command}")
        return handler["sbatch"](command, **kwargs)

    monkeypatch.setattr(launch_comparison.subprocess, "run", fake_run)
    monkeypatch.setattr(launch_comparison, "utc_now", lambda: "2026-09-18T12:00:00+00:00")
    monkeypatch.setattr(launch_comparison, "source_commit", lambda: manifest["source_commit"])
    monkeypatch.setattr(launch_comparison, "validate_manifest", lambda _: None)
    monkeypatch.setattr(launch_comparison, "frozen_hash", lambda *args: "f" * 64)
    return manifest_path, manifest, frozen, handler, submissions


def test_submits_serial_success_dependencies_and_persists_every_job(launch_fixture, monkeypatch):
    path, manifest, frozen, handler, submissions = launch_fixture
    expected_jobs = {}

    def fake_sbatch(command, *, check, text, capture_output, cwd):
        assert check is text is capture_output is True
        assert command[:2] == ["sbatch", "--parsable"]
        # A later submission must not happen before all previous IDs reach disk.
        assert json.loads(path.read_text())["jobs"] == expected_jobs
        index = len(submissions)
        name = manifest["conditions"][index]["name"]
        job_id = str(1200 + index)
        expected_log = Path(manifest["run_directories"][name]) / "training.log"
        assert f"--job-name=rank-{name}" in command
        assert f"--output={expected_log}" in command
        assert f"--error={expected_log}" in command
        assert expected_log.parent.is_dir()
        dependencies = [part for part in command if part.startswith("--dependency=")]
        assert dependencies == ([] if index == 0 else [f"--dependency=afterok:{1199 + index}"])
        assert command[-6:] == [
            str(Path(manifest["execution_root"]) / "experiments/state_rank/launch_comparison.sbatch"),
            manifest["execution_root"], manifest["project_root"], str(path.resolve()), name,
            manifest["source_commit"],
        ]
        submissions.append(command)
        expected_jobs[name] = job_id
        # Slurm's parsable output may include a cluster suffix.
        return subprocess.CompletedProcess(command, 0, stdout=f"{job_id};test-cluster\n", stderr="")

    handler["sbatch"] = fake_sbatch
    result = launch_comparison.submit(path)
    saved = json.loads(path.read_text())
    assert len(submissions) == 6
    assert len(frozen["calls"]) == 1
    assert result == saved
    assert saved["jobs"] == expected_jobs
    assert saved["submitted_utc"] == "2026-09-18T12:00:00+00:00"
    for name, directory in manifest["run_directories"].items():
        record = load_run(directory)
        assert record["attempts"][-1]["slurm_job_id"] == expected_jobs[name]
        assert record["status"] == "submitted"


def test_refuses_changed_frozen_execution_set_before_any_submission(launch_fixture):
    path, _, frozen, _, _ = launch_fixture
    frozen["changed"] = True
    before = path.read_bytes()
    with pytest.raises(subprocess.CalledProcessError):
        launch_comparison.submit(path)
    assert path.read_bytes() == before


def test_refuses_existing_jobs_without_resubmitting_or_overwriting(launch_fixture):
    path, manifest, _, handler, _ = launch_fixture
    manifest["jobs"] = {"pretrained": "912"}
    path.write_text(json.dumps(manifest))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="already has submitted jobs"):
        launch_comparison.submit(path)
    assert path.read_bytes() == before


def test_failed_later_submission_preserves_successful_ids(launch_fixture, monkeypatch):
    path, _, _, handler, submissions = launch_fixture

    def fake_sbatch(command, **kwargs):
        submissions.append(command)
        if len(submissions) == 1:
            return subprocess.CompletedProcess(command, 0, stdout="1200\n", stderr="")
        assert json.loads(path.read_text())["jobs"] == {"pretrained": "1200"}
        assert "--dependency=afterok:1200" in command
        raise subprocess.CalledProcessError(1, command, stderr="scheduler temporarily unavailable")

    handler["sbatch"] = fake_sbatch
    with pytest.raises(subprocess.CalledProcessError):
        launch_comparison.submit(path)
    assert len(submissions) == 2
    assert json.loads(path.read_text())["jobs"] == {"pretrained": "1200"}
    # Retrying cannot duplicate the already-submitted pretrained condition.
    with pytest.raises(ValueError, match="already has submitted jobs"):
        launch_comparison.submit(path)
    assert len(submissions) == 2


def test_rejects_unparseable_scheduler_response(launch_fixture, monkeypatch):
    path, _, _, handler, _ = launch_fixture
    handler["sbatch"] = lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="invalid response\n", stderr="")
    with pytest.raises(RuntimeError, match="Unexpected sbatch output"):
        launch_comparison.submit(path)
    assert json.loads(path.read_text())["jobs"] == {}


def test_deltanet_routes_driver_and_keeps_distinct_logs(launch_fixture, monkeypatch):
    path, manifest, _, handler, submissions = launch_fixture
    manifest["experiment"] = "state_rank/random_vs_pretrained_deltanet"
    path.write_text(json.dumps(manifest))

    def fake_sbatch(command, **kwargs):
        assert command[-1] == "compare_deltanet"
        assert "--cpus-per-task=8" in command
        assert "--time=02:00:00" in command
        output = next(part for part in command if part.startswith("--output="))
        assert "/experiments/state_rank/runs/" in output and output.endswith("/training.log")
        submissions.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=f"{1300 + len(submissions)}\n", stderr="")

    handler["sbatch"] = fake_sbatch
    result = launch_comparison.submit(path)
    assert len(result["jobs"]) == 6


def test_unknown_experiment_cannot_redirect_log_directory(launch_fixture):
    path, manifest, _, handler, _ = launch_fixture
    manifest["experiment"] = "../../unexpected"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsupported comparison"):
        launch_comparison.submit(path)


def test_uncommitted_prepared_recipe_cannot_submit(launch_fixture):
    path, _, frozen, _, submissions = launch_fixture
    frozen["committed"] = False
    before = path.read_bytes()
    with pytest.raises(subprocess.CalledProcessError):
        launch_comparison.submit(path)
    assert path.read_bytes() == before
    assert submissions == []


def test_submission_records_new_launch_commit_without_mutating_recipes(launch_fixture, monkeypatch):
    path, manifest, _, handler, submissions = launch_fixture
    before = {name: (Path(root) / "config.yaml").read_bytes() for name, root in manifest["run_directories"].items()}
    old_identity = manifest["identity"]
    # Preparation preceded the commit that introduced the canonical recipes.
    manifest["source_commit"] = "b" * 40
    handler["sbatch"] = lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="1234\n", stderr="")
    result = launch_comparison.submit(path)
    assert result["source_commit"] == "b" * 40
    assert result["identity"] != old_identity
    for name, root in result["run_directories"].items():
        assert (Path(root) / "config.yaml").read_bytes() == before[name]


def test_prepare_allocates_canonical_conditions_and_checks_frozen_recipes(comparison_project, monkeypatch):
    root, _, config, baseline = comparison_project
    study = root / "experiments/state_rank"
    study.mkdir(parents=True)
    (study / "sweep.yaml").write_text(json.dumps({"recipes": {"random_vs_pretrained": config},
                                                 "home_cluster": "kennel", "run_ids": []}))
    monkeypatch.setattr(launch_comparison, "EXECUTION_ROOT", root)
    path = launch_comparison.prepare_runs("llama", root, study)
    manifest = json.loads(path.read_text())
    assert len(manifest["run_directories"]) == 3
    first = Path(manifest["run_directories"]["pretrained"])
    assert path == first / ".runtime/comparison/manifest.json"
    assert (path.parent / "inputs.json").read_bytes() == (baseline / "inputs.json").read_bytes()
    compare_random.validate_manifest(manifest)
    for condition in manifest["conditions"]:
        run = compare_random.condition_directory(manifest, condition["name"])
        assert run.parent == study / "runs"
        record = load_run(run)
        assert record["checkpoint_retention"] == "none"
        assert record["config"]["sha256"] == manifest["run_config_sha256"][condition["name"]]
    recipe = first / "config.yaml"
    recipe.write_text(recipe.read_text() + "\n")
    with pytest.raises(ValueError, match="Canonical run configuration changed"):
        compare_random.validate_manifest(manifest)


def test_invalid_recipe_does_not_allocate_any_runs(comparison_project, monkeypatch):
    root, _, config, _ = comparison_project
    study = root / "experiments/state_rank"
    study.mkdir(parents=True)
    (study / "sweep.yaml").write_text(json.dumps({"run_ids": []}))
    invalid = {**config, "allow_tf32": True}
    with pytest.raises(ValueError, match="TF32 off"):
        launch_comparison.prepare_runs("llama", root, study, config=invalid)
    assert not (study / "runs").exists()


def test_condition_metadata_updates_colocated_canonical_record(tmp_path):
    run = create_run(tmp_path / "study", {"output_directory": "allocated"})
    raw = {"status": "complete", "source_commit": "a" * 40,
           "weights": "random", "seed": 2, "slurm_job_id": "123", "slurm_restart_count": 0,
           "hostname": "node", "gpu": "H100", "device": "cuda", "validation": {"passed": True}}
    compare_random.write_condition_metadata(run, raw)
    record = load_run(run)
    assert record["status"] == "complete" and record["seed"] == 2
    assert record["execution"]["raw_metadata"]["sha256"] == compare_random.sha256_file(run / "metadata.json")
    assert record["attempts"][0]["slurm_job_id"] == "123"
    compare_random.write_condition_metadata(run, raw)
    assert len(load_run(run)["attempts"]) == 1
