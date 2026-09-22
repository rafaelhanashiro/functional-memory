"""Offline checks of serial Slurm submission and durable submission bookkeeping."""

import json
from pathlib import Path
import subprocess

import pytest

from scaffolds.state_rank import compare_random, launch_comparison


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
    }
    manifest_path.write_text(json.dumps(manifest))
    git_state = {"dirty": "", "head": manifest["source_commit"]}
    git_calls = []
    submissions = []

    def fake_git(command, *, cwd, text):
        assert text is True
        git_calls.append((command, Path(cwd)))
        if command == ["git", "status", "--porcelain", "--untracked-files=normal"]:
            assert Path(cwd) == execution
            return git_state["dirty"]
        if command == ["git", "rev-parse", "HEAD"]:
            # source_commit is imported by the launcher but defined in compare_random.
            assert Path(cwd) == Path(compare_random.__file__).resolve().parent
            return git_state["head"] + "\n"
        raise AssertionError(f"Unexpected subprocess command: {command}")

    def reject_real_submission(command, **kwargs):
        raise AssertionError(f"Unexpected real submission attempt: {command}")

    monkeypatch.setattr(launch_comparison.subprocess, "check_output", fake_git)
    monkeypatch.setattr(launch_comparison.subprocess, "run", reject_real_submission)
    monkeypatch.setattr(launch_comparison, "utc_now", lambda: "2026-09-18T12:00:00+00:00")
    return manifest_path, manifest, git_state, git_calls, submissions


def test_submits_serial_success_dependencies_and_persists_every_job(launch_fixture, monkeypatch):
    path, manifest, _, git_calls, submissions = launch_fixture
    expected_jobs = {}

    def fake_sbatch(command, *, check, text, capture_output):
        assert check is text is capture_output is True
        assert command[:2] == ["sbatch", "--parsable"]
        # A later submission must not happen before all previous IDs reach disk.
        assert json.loads(path.read_text())["jobs"] == expected_jobs
        index = len(submissions)
        name = manifest["conditions"][index]["name"]
        job_id = str(1200 + index)
        expected_log = Path(manifest["project_root"]) / "results/logs/state_rank/random_vs_pretrained" / name / "slurm-%j.out"
        assert f"--job-name=rank-{name}" in command
        assert f"--output={expected_log}" in command
        assert f"--error={expected_log}" in command
        assert expected_log.parent.is_dir()
        dependencies = [part for part in command if part.startswith("--dependency=")]
        assert dependencies == ([] if index == 0 else [f"--dependency=afterok:{1199 + index}"])
        assert command[-6:] == [
            str(Path(manifest["execution_root"]) / "scaffolds/state_rank/launch_comparison.sbatch"),
            manifest["execution_root"], manifest["project_root"], str(path.resolve()), name,
            manifest["source_commit"],
        ]
        submissions.append(command)
        expected_jobs[name] = job_id
        # Slurm's parsable output may include a cluster suffix.
        return subprocess.CompletedProcess(command, 0, stdout=f"{job_id};test-cluster\n", stderr="")

    monkeypatch.setattr(launch_comparison.subprocess, "run", fake_sbatch)
    result = launch_comparison.submit(path)
    saved = json.loads(path.read_text())
    assert len(submissions) == 6
    assert len(git_calls) == 2
    assert result == saved
    assert saved["jobs"] == expected_jobs
    assert saved["submitted_utc"] == "2026-09-18T12:00:00+00:00"


@pytest.mark.parametrize("dirty", [" M scaffolds/state_rank/compare_random.py\n", "?? untracked.py\n"])
def test_refuses_dirty_execution_checkout_before_any_submission(launch_fixture, dirty):
    path, _, git_state, _, _ = launch_fixture
    git_state["dirty"] = dirty
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Commit the tested scaffold"):
        launch_comparison.submit(path)
    assert path.read_bytes() == before
    assert not (path.parent / "results/logs").exists()


def test_refuses_execution_commit_drift_before_submission(launch_fixture):
    path, _, git_state, _, _ = launch_fixture
    git_state["head"] = "b" * 40
    before = path.read_bytes()
    with pytest.raises(ValueError, match="frozen checkout"):
        launch_comparison.submit(path)
    assert path.read_bytes() == before


def test_refuses_existing_jobs_without_resubmitting_or_overwriting(launch_fixture):
    path, manifest, _, _, _ = launch_fixture
    manifest["jobs"] = {"pretrained": "912"}
    path.write_text(json.dumps(manifest))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="already has submitted jobs"):
        launch_comparison.submit(path)
    assert path.read_bytes() == before


def test_failed_later_submission_preserves_successful_ids(launch_fixture, monkeypatch):
    path, _, _, _, submissions = launch_fixture

    def fake_sbatch(command, *, check, text, capture_output):
        submissions.append(command)
        if len(submissions) == 1:
            return subprocess.CompletedProcess(command, 0, stdout="1200\n", stderr="")
        assert json.loads(path.read_text())["jobs"] == {"pretrained": "1200"}
        assert "--dependency=afterok:1200" in command
        raise subprocess.CalledProcessError(1, command, stderr="scheduler temporarily unavailable")

    monkeypatch.setattr(launch_comparison.subprocess, "run", fake_sbatch)
    with pytest.raises(subprocess.CalledProcessError):
        launch_comparison.submit(path)
    assert len(submissions) == 2
    assert json.loads(path.read_text())["jobs"] == {"pretrained": "1200"}
    # Retrying cannot duplicate the already-submitted pretrained condition.
    with pytest.raises(ValueError, match="already has submitted jobs"):
        launch_comparison.submit(path)
    assert len(submissions) == 2


def test_rejects_unparseable_scheduler_response(launch_fixture, monkeypatch):
    path, _, _, _, _ = launch_fixture
    monkeypatch.setattr(
        launch_comparison.subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="invalid response\n", stderr=""),
    )
    with pytest.raises(RuntimeError, match="Unexpected sbatch output"):
        launch_comparison.submit(path)
    assert json.loads(path.read_text())["jobs"] == {}


def test_deltanet_routes_driver_and_keeps_distinct_logs(launch_fixture, monkeypatch):
    path, manifest, _, _, submissions = launch_fixture
    manifest["experiment"] = "state_rank/random_vs_pretrained_deltanet"
    path.write_text(json.dumps(manifest))

    def fake_sbatch(command, **kwargs):
        assert command[-1] == "compare_deltanet"
        assert "--cpus-per-task=8" in command
        assert "--time=02:00:00" in command
        output = next(part for part in command if part.startswith("--output="))
        assert "/results/logs/state_rank/random_vs_pretrained_deltanet/" in output
        submissions.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=f"{1300 + len(submissions)}\n", stderr="")

    monkeypatch.setattr(launch_comparison.subprocess, "run", fake_sbatch)
    result = launch_comparison.submit(path)
    assert len(result["jobs"]) == 6


def test_unknown_experiment_cannot_redirect_log_directory(launch_fixture):
    path, manifest, _, _, _ = launch_fixture
    manifest["experiment"] = "../../unexpected"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsupported comparison"):
        launch_comparison.submit(path)
