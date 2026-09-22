"""Orchestration tests use tiny metadata fixtures and never submit real jobs."""
from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.studies.polynomial_attention.preflight import record_success
from scaffolds.transformer.studies.odd_taylor_attention import preflight, sweep

BASE = Path(__file__).resolve().parents[5] / "results/polynomial_attention/configs/base.yaml"


def make_test_manifest(tmp_path):
    base = load_config(BASE)
    base.model.attention_backend = "fused"
    root, execution = tmp_path / "project", tmp_path / "execution"
    for name in ("train_manifest", "eval_manifest"):
        path = tmp_path / f"{name}.json"
        path.write_text('{"dataset":"fixture"}')
        setattr(base.data, name, str(path))
    lock = execution / "scaffolds/transformer/uv.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("locked-dependencies\n")
    controls = []
    for name, degree in zip(sweep.CONTROL_NAMES, (2, None), strict=True):
        config = copy.deepcopy(base)
        config.model.attention_degree = degree
        config.train.source_commit = "b" * 40
        config.train.workdir = root / "results/polynomial_attention" / name
        config.train.wandb_run_name = name
        workdir = config.train.workdir
        workdir.mkdir(parents=True)
        raw = config_to_dict(config)
        raw["model"].pop("quadratic_mode", None)
        raw["model"].pop("quadratic_sharing", None)
        config_path = workdir / "config.yaml"
        config_path.write_text(yaml.safe_dump(raw))
        count = 199906304
        tokens = config.optim.total_steps * config.data.batch_size * (config.data.sequence_length - 1)
        record = {"event": "validation", "step": config.optim.total_steps, "checkpoint_step": config.optim.total_steps,
                  "source_commit": config.train.source_commit, "qk_head_dim": 64, "attention_degree": degree,
                  "parameter_count": count, "training_tokens": tokens, "validation_nll": 2.8,
                  "validation_perplexity": 16.0, "tokens_per_second": 100.0}
        completion = {**record, "final_step": config.optim.total_steps}
        (workdir / "metrics.jsonl").write_text(json.dumps(record) + "\n")
        (workdir / "completion.json").write_text(json.dumps(completion))
        run = {"name": name, "attention_degree": degree, "qk_head_dim": 64, "value_head_dim": 64,
               "parameter_count": count, "workdir": str(workdir), "config_path": str(config_path),
               "config": raw, "training_tokens": tokens, "retained_completed": True,
               "original_sweep_identity": "old-identity"}
        run["provenance_sha256"] = {str(path): sweep.sha256(path) for path in
                                     (config_path, workdir / "metrics.jsonl", workdir / "completion.json")}
        controls.append(run)
    manifest = sweep.make_manifest(base, root, execution, "a" * 40, controls=controls)
    return base, manifest


def mark_passed(manifest, *, rejected=()):
    root = Path(manifest["preflight_dir"])
    record_success(root / "success.json", manifest)
    for name in preflight.SHARED_CHECKS:
        record_success(root / "checks" / f"{name}.json", manifest)
    for run in manifest["runs"]:
        path = root / "checks" / f"{run['name']}.json"
        if run["name"] in rejected:
            sweep.write_json(path, {"identity": manifest["identity"], "status": "ineligible", "reason": "numerical"})
        else:
            record_success(path, manifest)


def test_two_matched_recipes_and_correct_parameter_counts(tmp_path):
    base, manifest = make_test_manifest(tmp_path)
    assert [run["attention_degree"] for run in manifest["runs"]] == list(sweep.VARIANTS)
    assert [run["parameter_count"] for run in manifest["runs"]] == [199906304] * 2
    assert [run["coefficient_count"] for run in manifest["runs"]] == [0, 0]
    for run in manifest["runs"]:
        assert run["training_tokens"] == 10000269312
        assert run["config"]["optim"]["total_steps"] == 19074
        assert run["config"]["train"]["seed"] == 0
        assert run["config"]["train"]["micro_batch_size"] == 64
        assert run["config"]["model"]["attention_backend"] == "fused"
        assert sweep.scientific_recipe(run["config"]) == sweep.scientific_recipe(manifest["controls"][0]["config"])
    assert base.model.quadratic_mode == "fixed"
    assert manifest["scheduler"]["node"] == "dogo"
    assert manifest["scheduler"]["gpu_count"] == 4
    assert manifest["scheduler"]["production_time"] == "08:00:00"
    assert manifest["scheduler"]["preflight_time"] == "02:00:00"
    assert not sweep.manifest_path(Path(manifest["project_root"])).exists()


def test_historical_configs_are_unchanged_and_checksum_protected(tmp_path):
    _, manifest = make_test_manifest(tmp_path)
    control = manifest["controls"][0]
    assert "quadratic_mode" not in control["config"]["model"]
    path = Path(control["config_path"])
    before = path.read_bytes()
    sweep.validate_control(control)
    assert path.read_bytes() == before
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="artifact changed"):
        sweep.validate_control(control)


def test_portable_retained_controls_preserve_provenance_and_validate_hashes(tmp_path):
    _, manifest = make_test_manifest(tmp_path)
    root = Path(manifest["project_root"])
    retained_runs = []
    for control in manifest["controls"]:
        old_path = Path(control["config_path"])
        new_path = root / "results/polynomial_attention/configs" / f"{control['name']}.yaml"
        new_path.parent.mkdir(parents=True, exist_ok=True)
        old_path.rename(new_path)
        record = json.loads((Path(control["workdir"]) / "metrics.jsonl").read_text())
        retained_runs.append({
            **{key: control[key] for key in ("name", "attention_degree", "qk_head_dim", "value_head_dim", "parameter_count")},
            "attention_backend": "fused", "seed": 0, "source_commit": "b" * 40,
            "resolved_config": {"path": str(new_path.relative_to(root)), "sha256": sweep.sha256(new_path)},
            "metrics": {"path": str((Path(control["workdir"]) / "metrics.jsonl").relative_to(root)),
                        "sha256": sweep.sha256(Path(control["workdir"]) / "metrics.jsonl")},
            "completion": str((Path(control["workdir"]) / "completion.json").relative_to(root)),
            "selected_checkpoint": {"path": f"checkpoints/polynomial_attention/{control['name']}/{record['step']}",
                                    **{key: record[key] for key in ("step", "validation_nll", "validation_perplexity")}},
            "terminal": {key: record[key] for key in ("step", "training_tokens", "validation_nll", "validation_perplexity")},
            "execution_attempts": [{"gpu_kind": "B300", "gpu_count": 4, "status": "complete"}],
        })
    provenance = {"format": "functional-memory-retained-experiment", "schema_version": 1,
                  "experiment": "polynomial_attention", "historical_sweep": {"identity": "original", "source_commit": "b" * 40},
                  "runs": retained_runs}
    path = root / "results/polynomial_attention/provenance/manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance))
    controls = sweep.historical_controls(root)
    assert [control["name"] for control in controls] == list(sweep.CONTROL_NAMES)
    assert all(control["original_sweep_identity"] == "original" for control in controls)
    assert controls[1]["retained_provenance"]["execution_attempts"][0]["gpu_kind"] == "B300"
    assert controls[0]["config_path"].endswith("results/polynomial_attention/configs/200m-m002-seed0.yaml")
    assert str(path) in controls[0]["provenance_sha256"]
    metrics = root / retained_runs[0]["metrics"]["path"]
    metrics.write_text(metrics.read_text() + "\n")
    with pytest.raises(ValueError, match="metrics checksum changed"):
        sweep.historical_controls(root)


def test_recipe_changes_are_rejected(tmp_path):
    base, manifest = make_test_manifest(tmp_path)
    base.optim.learning_rate *= 2
    with pytest.raises(ValueError, match="preserve"):
        sweep.make_manifest(base, Path(manifest["project_root"]), Path(manifest["execution_root"]),
                            manifest["source_commit"], controls=manifest["controls"])


def test_data_and_dependency_changes_alter_identity(tmp_path):
    base, manifest = make_test_manifest(tmp_path)
    args = (base, Path(manifest["project_root"]), Path(manifest["execution_root"]), manifest["source_commit"])
    Path(base.data.train_manifest).write_text("changed data")
    data_changed = sweep.make_manifest(*args, controls=manifest["controls"])
    assert data_changed["identity"] != manifest["identity"]
    (Path(manifest["execution_root"]) / "scaffolds/transformer/uv.lock").write_text("new lock")
    lock_changed = sweep.make_manifest(*args, controls=manifest["controls"])
    assert lock_changed["identity"] != data_changed["identity"]


def test_prepare_retains_configs_and_checkpoints_outside_frozen_checkout(tmp_path):
    base, manifest = make_test_manifest(tmp_path)
    before = [Path(run["config_path"]).read_bytes() for run in manifest["controls"]]
    sweep.prepare_sweep(manifest, base)
    sweep.prepare_sweep(manifest, base)
    for run in manifest["runs"]:
        assert Path(run["config_path"]).is_file()
        assert "/results/odd_taylor_attention/configs/" in run["config_path"]
        link = Path(run["workdir"]) / "checkpoints"
        assert link.is_symlink() and link.resolve() == Path(run["checkpoint_dir"])
        assert not str(link.resolve()).startswith(manifest["execution_root"])
    assert before == [Path(run["config_path"]).read_bytes() for run in manifest["controls"]]


def test_eligibility_is_per_variant_after_shared_checks(tmp_path):
    _, manifest = make_test_manifest(tmp_path)
    rejected = manifest["runs"][0]["name"]
    mark_passed(manifest, rejected=(rejected,))
    assert preflight.preflight_complete(manifest)
    assert preflight.eligible_runs(manifest) == {run["name"] for run in manifest["runs"][1:]}
    (Path(manifest["preflight_dir"]) / "checks/attention-fused-gpu.json").unlink()
    assert not preflight.preflight_complete(manifest)
    assert not preflight.eligible_runs(manifest)


def test_identity_and_frozen_checkout_gates(tmp_path, monkeypatch):
    base, manifest = make_test_manifest(tmp_path)
    sweep.prepare_sweep(manifest, base)
    monkeypatch.setattr(preflight.subprocess, "check_output", lambda command, **kw: manifest["source_commit"] if "rev-parse" in command else "")
    preflight.verify_manifest(manifest)
    manifest["scheduler"]["gpu_count"] = 2
    with pytest.raises(ValueError, match="identity does not match"):
        preflight.verify_manifest(manifest)


def test_submission_serializes_only_eligible_variants(tmp_path, monkeypatch):
    base, manifest = make_test_manifest(tmp_path)
    manifest["jobs"]["preflight"] = "99"
    rejected = manifest["runs"][1]["name"]
    mark_passed(manifest, rejected=(rejected,))
    monkeypatch.setattr(preflight, "verify_manifest", lambda manifest: None)
    monkeypatch.setattr("scaffolds.transformer.studies.odd_taylor_attention.plot.aggregate", lambda manifest: [])
    calls = []
    def submit(command, execution, dependency=None, dependency_type="afterok"):
        calls.append((command, dependency, dependency_type))
        assert "--nodelist=dogo" in command
        assert "--gres=gpu:h100:4" in command and "--time=08:00:00" in command
        return str(100 + len(calls))
    monkeypatch.setattr(sweep, "submit_job", submit)
    result = sweep.submit_sweep(manifest, base)
    assert len(calls) == 1 and rejected not in result["jobs"]
    assert [call[1:] for call in calls] == [("99", "afterok")]
    for command, dependency, dependency_type in calls:
        assert f"--dependency={dependency_type}:{dependency}" in command
    sweep.submit_sweep(result, base)
    assert len(calls) == 1


def initializations(manifest):
    fixed = {"backbone_sha256": "same", "backbone_parameter_count": 199906304,
             "coefficient_count": 0, "parameter_count": 199906304,
             "coefficient_shape": None, "initial_coefficient_pair": None}
    return {name: dict(fixed) for name in ["fixed", *[run["name"] for run in manifest["runs"]]]}


@pytest.mark.parametrize("field,value", [("backbone_sha256", "changed"), ("coefficient_count", 7),
                                         ("parameter_count", 199906305)])
def test_initialization_gate_checks_backbone_and_coefficients(tmp_path, field, value):
    _, manifest = make_test_manifest(tmp_path)
    values = initializations(manifest)
    preflight.validate_initializations(values, manifest)
    values[manifest["runs"][0]["name"]][field] = value
    with pytest.raises(RuntimeError):
        preflight.validate_initializations(values, manifest)


def test_runtime_projection_uses_allocation_limit_with_margin():
    run = {"name": "pilot", "config": {"optim": {"total_steps": 100}, "train": {"eval_every": 40, "eval_batches": 7, "ckpt_every": 25}}}
    outcome = {"timing": {"runtime_seconds": 120., "train_step_seconds": [100., 2.], "evaluation_seconds": [3.], "checkpoint_seconds": [4.]}}
    result = preflight.projected_runtime(run, outcome, sweep.scheduler_profile())
    assert result["projected_total_seconds"] == 647.0
    assert result["required_seconds_with_margin"] == pytest.approx(711.7)
    assert result["limit_seconds"] == 28800
    outcome["timing"]["train_step_seconds"][1] = 300.
    with pytest.raises(preflight.RuntimeLimitExceeded):
        preflight.projected_runtime(run, outcome, sweep.scheduler_profile())


def checkpoint_audit(*, start, final):
    def update(step):
        return {"step": step, "batch_sha256": f"{step:064x}", "rng_sha256": f"{step + 100:064x}"}
    return {
        "start_state_sha256": f"{start:064x}", "final_state_sha256": f"{final:064x}",
        "checkpoint_state_sha256": f"{final:064x}",
        "first_update": update(start + 1), "next_update": update(final + 1),
    }


@pytest.mark.parametrize("change,error", [
    ("live_state", "Durable checkpoint"),
    ("durable_state", "Durable checkpoint"),
    ("restart_state", "Restarted state"),
    ("batch", "next batch or RNG"),
    ("rng", "next batch or RNG"),
    ("first_step", "invalid first_update step"),
    ("next_step", "invalid next_update step"),
    ("bad_hash", "invalid SHA256"),
    ("missing_audit", "missing its checkpoint audit"),
])
def test_resume_audit_rejects_state_input_and_rng_changes(change, error):
    initial = {"start_step": 0, "final_step": 2, "checkpoint_audit": checkpoint_audit(start=0, final=2)}
    resumed = {"start_step": 2, "final_step": 3, "checkpoint_audit": checkpoint_audit(start=2, final=3)}
    preflight.validate_resume_audit(initial, resumed)
    audit = resumed["checkpoint_audit"]
    if change in ("live_state", "durable_state", "restart_state"):
        key = {"live_state": "final_state_sha256", "durable_state": "checkpoint_state_sha256",
               "restart_state": "start_state_sha256"}[change]
        audit[key] = "f" * 64
    elif change in ("batch", "rng"):
        audit["first_update"][f"{change}_sha256"] = "f" * 64
    elif change in ("first_step", "next_step"):
        audit["first_update" if change == "first_step" else "next_update"]["step"] += 1
    elif change == "bad_hash":
        audit["first_update"]["batch_sha256"] = "not-a-sha256"
    else:
        resumed.pop("checkpoint_audit")
    with pytest.raises(RuntimeError, match=error):
        preflight.validate_resume_audit(initial, resumed)


def mock_preflight_workers(monkeypatch, manifest, *, numerical_run=None, wrong_resume=False, infrastructure=False):
    calls = []
    monkeypatch.setattr(preflight, "verify_manifest", lambda manifest: None)
    def worker(manifest, worker, output, *, name="", workdir=None, stop=2, cpu=False):
        calls.append((worker, name, stop))
        if worker == "hardware":
            return {"devices": ["H100"] * 4}
        if worker == "initialization":
            return initializations(manifest)[name]
        if infrastructure:
            raise subprocess.CalledProcessError(1, ["pilot"])
        if name == numerical_run:
            raise preflight.NumericalPilotFailure("zero final denominator")
        resumed = stop == 3
        audit = checkpoint_audit(start=2 if resumed else 0, final=stop)
        if resumed and wrong_resume:
            audit["start_state_sha256"] = "f" * 64
        return {"final_step": stop, "start_step": 2 if resumed else 0, "restored_step": 2 if resumed else None,
                "data_parallel_devices": 4, "data_start_sequence": 512 if resumed else 0,
                "checkpoint_audit": audit,
                "timing": {"runtime_seconds": 20., "train_step_seconds": [10., .5],
                           "evaluation_seconds": [2.], "checkpoint_seconds": [1.]}}
    def pytest_command(command, **kwargs):
        junit = Path(next(value.removeprefix("--junitxml=") for value in command if value.startswith("--junitxml=")))
        junit.write_text('<testsuite>' + '<testcase name="check"/>' * 4 + '</testsuite>')
    monkeypatch.setattr(preflight, "launch_worker", worker)
    monkeypatch.setattr(preflight.subprocess, "run", pytest_command)
    monkeypatch.setattr("scaffolds.transformer.studies.polynomial_attention.plot.committed_records", lambda path: [{"step": 2, "validation_nll": 2., "validation_perplexity": 7.4}])
    return calls


def test_numeric_pilot_failure_does_not_suppress_other_variants(tmp_path, monkeypatch):
    base, manifest = make_test_manifest(tmp_path)
    sweep.prepare_sweep(manifest, base)
    rejected = manifest["runs"][0]["name"]
    calls = mock_preflight_workers(monkeypatch, manifest, numerical_run=rejected)
    preflight.run_preflight(manifest)
    assert preflight.preflight_complete(manifest)
    assert rejected not in preflight.eligible_runs(manifest)
    for run in manifest["runs"][1:]:
        assert ("pilot", run["name"], 2) in calls
        assert calls.count(("pilot", run["name"], 3)) == 1
    retained = Path(manifest["project_root"]) / "results/odd_taylor_attention/preflight" / manifest["identity"]
    assert (retained / "success.json").exists()
    assert (retained / "attention-fused-gpu.xml").exists()
    assert json.loads((retained / "provenance.json").read_text())["eligible_runs"] == sorted(preflight.eligible_runs(manifest))
    with pytest.raises(RuntimeError, match="own pilot"):
        preflight.run_preflight(manifest, check_only=True, run_name=rejected)


def test_cpu_and_distributed_preflight_route_tests_without_skipping(tmp_path, monkeypatch):
    base, manifest = make_test_manifest(tmp_path)
    sweep.prepare_sweep(manifest, base)
    mock_preflight_workers(monkeypatch, manifest)
    calls, minimums = {}, {}
    real_require = preflight.require_passing_tests
    monkeypatch.setenv("XLA_FLAGS", "--xla_existing_fixture_flag=true")

    def pytest_command(command, **kwargs):
        junit = Path(next(value.removeprefix("--junitxml=") for value in command if value.startswith("--junitxml=")))
        stage = junit.stem
        calls[stage] = command, kwargs
        count = 2 if stage == "attention-distributed-cpu" else 4
        junit.write_text('<testsuite>' + '<testcase name="check"/>' * count + '</testsuite>')

    def require(junit, minimum):
        minimums[junit.stem] = minimum
        return real_require(junit, minimum)

    monkeypatch.setattr(preflight.subprocess, "run", pytest_command)
    monkeypatch.setattr(preflight, "require_passing_tests", require)
    preflight.run_preflight(manifest)
    assert minimums == {"attention-cpu": 4, "attention-distributed-cpu": 2, "attention-fused-gpu": 4}
    assert set(preflight.SHARED_CHECKS) == {*minimums, "initialization"}
    command, options = calls["attention-cpu"]
    assert command[command.index("-k") + 1] == "not distributed"
    assert command[command.index("-m", 3) + 1] == "not gpu"
    assert options["env"]["JAX_PLATFORMS"] == "cpu"
    assert options["env"]["XLA_FLAGS"] == "--xla_existing_fixture_flag=true"
    assert [Path(arg).name for arg in command if arg.endswith(".py")] == ["test_odd_attention.py", "test_odd_training.py"]
    command, options = calls["attention-distributed-cpu"]
    assert command[command.index("-k") + 1] == "distributed"
    assert options["env"]["JAX_PLATFORMS"] == "cpu"
    assert options["env"]["XLA_FLAGS"] == "--xla_force_host_platform_device_count=2"
    assert [Path(arg).name for arg in command if arg.endswith(".py")] == ["test_odd_training.py"]
    command, options = calls["attention-fused-gpu"]
    assert "-k" not in command
    assert command[command.index("-m", 3) + 1] == "gpu"
    assert options["env"]["JAX_PLATFORMS"] == "cuda"
    assert options["env"]["XLA_FLAGS"] == "--xla_existing_fixture_flag=true"
    assert [Path(arg).name for arg in command if arg.endswith(".py")] == ["test_odd_attention.py"]
    assert all(options["check"] and options["cwd"] == manifest["execution_root"] for _, options in calls.values())


@pytest.mark.parametrize("wrong_resume,infrastructure", [(True, False), (False, True)])
def test_correctness_or_infrastructure_failures_block_production(tmp_path, monkeypatch, wrong_resume, infrastructure):
    base, manifest = make_test_manifest(tmp_path)
    sweep.prepare_sweep(manifest, base)
    mock_preflight_workers(monkeypatch, manifest, wrong_resume=wrong_resume, infrastructure=infrastructure)
    with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
        preflight.run_preflight(manifest)
    assert not preflight.preflight_complete(manifest)
    assert not preflight.eligible_runs(manifest)


def test_odd_summary_best_step_matches_latest_tie_retention(tmp_path, monkeypatch):
    from scaffolds.transformer.studies.odd_taylor_attention import plot

    base, manifest = make_test_manifest(tmp_path)
    run = manifest["runs"][0]
    records = [{"step": step, "validation_nll": 2.0, "validation_perplexity": 7.4,
                "attention_degree": run["attention_degree"]}
               for step in (2, 4)]
    monkeypatch.setattr(plot, "historical_summary", lambda *args: ({"best_step": 2}, records))
    row, _ = plot.summarize_run(run, manifest["source_commit"])
    assert row["best_step"] == 4
    assert row["best_ppl"] == 7.4


def test_report_has_four_labels_and_rejects_mislabeled_degrees(tmp_path):
    from scaffolds.transformer.studies.odd_taylor_attention.plot import aggregate, summarize_run
    base, manifest = make_test_manifest(tmp_path)
    sweep.prepare_sweep(manifest, base)
    report = Path(manifest["project_root"]) / "results/odd_taylor_attention"
    authored = "# Experiment methods\n\nKeep the authored reproduction instructions.\n"
    (report / "README.md").write_text(authored)
    rows = aggregate(manifest)
    assert (report / "README.md").read_text().startswith(authored)
    assert len(rows) == 4 and len({row["kernel"] for row in rows}) == 4
    assert sum(row["status"] == "complete" for row in rows) == 2
    report = Path(manifest["project_root"]) / "results/odd_taylor_attention"
    for name in ("summary.csv", "terminal_metrics.pdf", "learning_curves.pdf", "stability.pdf", "throughput.pdf"):
        assert (report / ("figures" if name.endswith(".pdf") else "") / name).exists()
    assert "One seed per variant does not estimate seed variation" in (report / "README.md").read_text()
    run = manifest["runs"][0]
    record = {"event": "validation", "step": 2, "checkpoint_step": 2, "source_commit": manifest["source_commit"],
              "qk_head_dim": 64, "attention_degree": 2, "parameter_count": run["parameter_count"],
              "training_tokens": 1048576, "validation_nll": 2., "validation_perplexity": 7.4,
              "quadratic_mode": "unrestricted", "quadratic_sharing": "global"}
    (Path(run["workdir"]) / "metrics.jsonl").write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="identity|degree"):
        summarize_run(run, manifest["source_commit"])


def stage_fixture(manifest, run):
    audit = checkpoint_audit(start=0, final=5000)
    outcome = {"final_step": 5000, "latest_checkpoint_after": 5000,
               "attention_degree": run["attention_degree"], "data_parallel_devices": 4,
               "checkpoint_audit": audit}
    validation = {"step": 5000, "source_commit": manifest["source_commit"],
                  "attention_degree": run["attention_degree"], "validation_nll": 3.,
                  "validation_perplexity": 20.0855369232}
    return outcome, validation


@pytest.mark.parametrize("change", ["nonfinite", "missing_validation", "source", "degree", "schedule",
                                    "state", "batch_hash", "next_step", "checkpoint", "allocation"])
def test_stability_gate_rejects_invalid_validation_state_and_recipe(tmp_path, change):
    from scaffolds.transformer.studies.odd_taylor_attention.production import validate_stability_stage
    _, manifest = make_test_manifest(tmp_path)
    run = manifest["runs"][0]
    outcome, validation = stage_fixture(manifest, run)
    assert validate_stability_stage(manifest, run, outcome, [validation]) == validation
    if change == "nonfinite":
        validation["validation_nll"] = float("nan")
    elif change == "missing_validation":
        validation["step"] = 4000
    elif change == "source":
        validation["source_commit"] = "bad"
    elif change == "degree":
        outcome["attention_degree"] = 2
    elif change == "schedule":
        run["config"]["optim"]["total_steps"] = 5000
    elif change == "state":
        outcome["checkpoint_audit"]["checkpoint_state_sha256"] = "f" * 64
    elif change == "batch_hash":
        outcome["checkpoint_audit"]["next_update"]["batch_sha256"] = "bad"
    elif change == "next_step":
        outcome["checkpoint_audit"]["next_update"]["step"] = 5000
    elif change == "checkpoint":
        outcome["latest_checkpoint_after"] = 4000
    else:
        outcome["data_parallel_devices"] = 2
    with pytest.raises(RuntimeError):
        validate_stability_stage(manifest, run, outcome, [validation])


@pytest.mark.parametrize("change", ["state", "batch", "rng", "step"])
def test_stage_restart_requires_exact_state_input_and_rng(tmp_path, change):
    from scaffolds.transformer.studies.odd_taylor_attention.production import validate_restart
    _, manifest = make_test_manifest(tmp_path)
    outcome, _ = stage_fixture(manifest, manifest["runs"][0])
    audit = outcome["checkpoint_audit"]
    restart = {"restored_step": 5000, "state_sha256": audit["final_state_sha256"],
               "next_update": copy.deepcopy(audit["next_update"])}
    validate_restart(outcome, restart)
    if change == "state":
        restart["state_sha256"] = "f" * 64
    elif change == "step":
        restart["restored_step"] -= 1
    else:
        restart["next_update"][f"{change}_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="changed checkpoint state, next batch, or RNG"):
        validate_restart(outcome, restart)


def test_both_production_jobs_use_afterany_to_survive_first_failure(tmp_path, monkeypatch):
    base, manifest = make_test_manifest(tmp_path)
    manifest["jobs"]["preflight"] = "99"
    mark_passed(manifest)
    monkeypatch.setattr(preflight, "verify_manifest", lambda manifest: None)
    monkeypatch.setattr("scaffolds.transformer.studies.odd_taylor_attention.plot.aggregate", lambda manifest: [])
    calls = []
    def submit(command, execution, dependency=None, dependency_type="afterok"):
        calls.append((dependency, dependency_type))
        return str(100 + len(calls))
    monkeypatch.setattr(sweep, "submit_job", submit)
    sweep.submit_sweep(manifest, base)
    assert calls == [("99", "afterok"), ("101", "afterany")]


@pytest.mark.parametrize("bad_restart", [False, True])
def test_production_continues_only_after_validated_stability_restart(tmp_path, monkeypatch, bad_restart):
    from scaffolds.transformer.studies.odd_taylor_attention import production
    base, manifest = make_test_manifest(tmp_path)
    sweep.prepare_sweep(manifest, base)
    run = manifest["runs"][0]
    outcome, validation = stage_fixture(manifest, run)
    monkeypatch.setattr(preflight, "run_preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr("scaffolds.transformer.studies.polynomial_attention.plot.committed_records", lambda root: [validation])
    calls = []
    def worker(manifest, run, worker, output):
        calls.append(worker)
        if worker == "stability":
            return outcome
        if worker == "restore":
            return {"restored_step": 5000,
                    "state_sha256": "f" * 64 if bad_restart else outcome["checkpoint_audit"]["final_state_sha256"],
                    "next_update": outcome["checkpoint_audit"]["next_update"]}
        assert (Path(run["workdir"]) / "stability_gate.json").exists()
        return {"start_step": 5000, "restored_step": 5000, "final_step": 19074,
                "checkpoint_audit": {"start_state_sha256": outcome["checkpoint_audit"]["final_state_sha256"],
                                     "first_update": outcome["checkpoint_audit"]["next_update"]}}
    monkeypatch.setattr(production, "launch_worker", worker)
    if bad_restart:
        with pytest.raises(RuntimeError):
            production.run_production(manifest, run)
        assert calls == ["stability", "restore"]
        assert (Path(run["workdir"]) / "production_failure.json").exists()
        assert not (Path(run["workdir"]) / "stability_gate.json").exists()
    else:
        (Path(run["workdir"]) / "production_failure.json").write_text('{"error":"previous interruption"}')
        assert production.run_production(manifest, run)["final_step"] == 19074
        assert calls == ["stability", "restore", "continuation"]
        assert not (Path(run["workdir"]) / "production_failure.json").exists()


def test_report_exposes_failed_pilot_update(tmp_path):
    from scaffolds.transformer.studies.odd_taylor_attention.plot import summarize_run
    base, manifest = make_test_manifest(tmp_path)
    sweep.prepare_sweep(manifest, base)
    run = manifest["runs"][0]
    sweep.write_json(Path(run["workdir"]) / "preflight_failure.json", {
        "error": "nonfinite gradients", "reason": "numerical",
        "numerical_failures": [{"pilot": run["name"], "record": {"step": 2}}],
    })
    row, records = summarize_run(run, manifest["source_commit"])
    assert not records and row["status"] == "preflight-ineligible"
    assert row["failure_step"] == 2
    assert row["terminal_nll"] == row["terminal_ppl"] == ""
