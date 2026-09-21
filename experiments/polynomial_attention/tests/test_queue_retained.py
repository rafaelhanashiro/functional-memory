import copy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.polynomial_attention.sweep import (
    make_manifest, prepare_sweep, scheduler_options, scheduler_profile,
    submit_sweep, validate_retained_run,
)
from experiments.polynomial_attention.tests.test_sweep import make_test_manifest


def completed_zero(tmp_path):
    base, old = make_test_manifest(tmp_path, gpu_count=4)
    prepare_sweep(old, base)
    run = copy.deepcopy(old["runs"][0])
    workdir = Path(run["workdir"])
    step = run["config"]["optim"]["total_steps"]
    record = {
        "event": "validation", "step": step, "checkpoint_step": step,
        "training_tokens": run["training_tokens"], "attention_degree": 0,
        "source_commit": old["source_commit"], "qk_head_dim": 64,
        "parameter_count": run["parameter_count"],
        "validation_nll": 3.85, "validation_perplexity": 47.0,
    }
    (workdir / "metrics.jsonl").write_text(json.dumps(record) + "\n")
    completion = workdir / "completion.json"
    completion.write_text(json.dumps({**record, "final_step": step}))
    run.update(retained_completed=True, retained_completion_sha256=hashlib.sha256(completion.read_bytes()).hexdigest())
    base.model.attention_backend = "fused"
    return base, old, run


def revised(base, old, retained):
    return make_manifest(base, Path(old["project_root"]), Path(old["execution_root"]), "b" * 40,
                         scheduler=scheduler_profile(4, node="dogo"), retained_runs=[retained])


def test_sxm_jobs_have_explicit_hardware_and_bounded_walltime():
    profile = scheduler_profile(4, node="dogo")
    for preflight in (True, False):
        options = scheduler_options(profile, preflight=preflight)
        assert "--nodelist=dogo" in options and "--gres=gpu:h100:4" in options
        assert "--partition=standard" in options and "--requeue" in options
    assert "--time=08:00:00" in scheduler_options(profile, preflight=False)
    with pytest.raises(ValueError, match="SXM node"):
        scheduler_profile(4, node="mastiff")


def test_retained_completion_keeps_original_config_and_changes_identity(tmp_path):
    base, old, retained = completed_zero(tmp_path)
    before = Path(retained["config_path"]).read_bytes()
    new = revised(base, old, retained)
    assert new["schema_version"] == 3 and new["identity"] != old["identity"]
    assert new["runs"][0]["config"]["train"]["source_commit"] == "a" * 40
    assert all(run["config"]["train"]["source_commit"] == "b" * 40 for run in new["runs"][1:])
    assert all(run["config"]["model"]["attention_backend"] == "fused" for run in new["runs"][1:])
    assert Path(retained["config_path"]).read_bytes() == before
    assert new["jobs"] == {}


def test_retained_control_cannot_mask_a_changed_recipe_or_completion(tmp_path):
    base, old, retained = completed_zero(tmp_path)
    base.optim.learning_rate *= 2
    with pytest.raises(ValueError, match="scientific recipe"):
        revised(base, old, retained)
    completion = Path(retained["workdir"]) / "completion.json"
    completion.write_text(completion.read_text() + "\n")
    with pytest.raises(ValueError, match="Retained completion changed"):
        validate_retained_run(retained)


def test_queued_active_jobs_all_require_successful_preflight_and_skip_zero(tmp_path, monkeypatch):
    base, old, retained = completed_zero(tmp_path)
    manifest = revised(base, old, retained)
    manifest["jobs"]["preflight"] = "900"
    commands = []
    monkeypatch.setattr("experiments.polynomial_attention.sweep.prepare_sweep", lambda manifest, base: manifest)
    monkeypatch.setattr("experiments.polynomial_attention.preflight.verify_manifest", lambda manifest: None)
    monkeypatch.setattr("experiments.polynomial_attention.plot.aggregate", lambda manifest: [])

    def submit(command, execution_root, dependency=None, dependency_type="afterok"):
        commands.append(command)
        return str(900 + len(commands))

    monkeypatch.setattr("experiments.polynomial_attention.sweep.submit_job", submit)
    submitted = submit_sweep(manifest, base, queue_after_preflight=True)
    assert len(commands) == 5
    assert retained["name"] not in submitted["jobs"]
    for index, command in enumerate(commands):
        deps = next(arg.removeprefix("--dependency=") for arg in command if arg.startswith("--dependency="))
        assert "afterok:900" in deps.split(",")
        if index:
            assert f"afterany:{900 + index}" in deps.split(",")
        assert command[-1] == manifest["runs"][index + 1]["name"]
        assert "--nodelist=dogo" in command and "--gres=gpu:h100:4" in command
    submit_sweep(manifest, base, queue_after_preflight=True)
    assert len(commands) == 5
