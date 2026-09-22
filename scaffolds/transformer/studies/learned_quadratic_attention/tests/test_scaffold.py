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
from scaffolds.transformer.studies.learned_quadratic_attention import preflight, sweep

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


def test_four_matched_recipes_and_correct_parameter_counts(tmp_path):
    base, manifest = make_test_manifest(tmp_path)
    assert [(run["quadratic_mode"], run["quadratic_sharing"]) for run in manifest["runs"]] == list(sweep.VARIANTS)
    assert [run["parameter_count"] for run in manifest["runs"]] == [199906306, 199906624] * 2
    assert [run["coefficient_count"] for run in manifest["runs"]] == [2, 320] * 2
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
        assert "/results/learned_quadratic_attention/configs/" in run["config_path"]
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
    monkeypatch.setattr("scaffolds.transformer.studies.learned_quadratic_attention.plot.aggregate", lambda manifest: [])
    calls = []
    def submit(command, execution, dependency=None, dependency_type="afterok"):
        calls.append((command, dependency, dependency_type))
        assert "--nodelist=dogo" in command
        assert "--gres=gpu:h100:4" in command and "--time=08:00:00" in command
        return str(100 + len(calls))
    monkeypatch.setattr(sweep, "submit_job", submit)
    result = sweep.submit_sweep(manifest, base)
    assert len(calls) == 3 and rejected not in result["jobs"]
    assert [call[1:] for call in calls] == [("99", "afterok"), ("101", "afterany"), ("102", "afterany")]
    for command, dependency, dependency_type in calls:
        assert f"--dependency={dependency_type}:{dependency}" in command
    sweep.submit_sweep(result, base)
    assert len(calls) == 3


def initializations(manifest):
    fixed = {"backbone_sha256": "same", "backbone_parameter_count": 199906304,
             "coefficient_count": 0, "parameter_count": 199906304,
             "coefficient_shape": None, "initial_coefficient_pair": None}
    values = {"fixed": fixed}
    for run in manifest["runs"]:
        values[run["name"]] = {**fixed, "parameter_count": run["parameter_count"],
                               "coefficient_count": run["coefficient_count"],
                               "coefficient_shape": [2] if run["quadratic_sharing"] == "global" else [10, 16, 2],
                               "initial_coefficient_pair": [1., 1.] if run["quadratic_mode"] == "positive" else [1., .5]}
    return values


@pytest.mark.parametrize("field,value", [("backbone_sha256", "changed"), ("coefficient_count", 7),
                                         ("initial_coefficient_pair", [1., .6])])
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
    assert result["projected_total_seconds"] == 407.0
    assert result["required_seconds_with_margin"] == pytest.approx(447.7)
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
    retained = Path(manifest["project_root"]) / "results/learned_quadratic_attention/preflight" / manifest["identity"]
    assert (retained / "success.json").exists()
    assert (retained / "attention-fused-gpu.xml").exists()
    assert json.loads((retained / "provenance.json").read_text())["eligible_runs"] == sorted(preflight.eligible_runs(manifest))
    with pytest.raises(RuntimeError, match="own pilot"):
        preflight.run_preflight(manifest, check_only=True, run_name=rejected)


@pytest.mark.parametrize("wrong_resume,infrastructure", [(True, False), (False, True)])
def test_correctness_or_infrastructure_failures_block_production(tmp_path, monkeypatch, wrong_resume, infrastructure):
    base, manifest = make_test_manifest(tmp_path)
    sweep.prepare_sweep(manifest, base)
    mock_preflight_workers(monkeypatch, manifest, wrong_resume=wrong_resume, infrastructure=infrastructure)
    with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
        preflight.run_preflight(manifest)
    assert not preflight.preflight_complete(manifest)
    assert not preflight.eligible_runs(manifest)


def test_learned_summary_best_step_matches_latest_tie_retention(tmp_path, monkeypatch):
    from scaffolds.transformer.studies.learned_quadratic_attention import plot

    base, manifest = make_test_manifest(tmp_path)
    run = manifest["runs"][0]
    records = [{"step": step, "validation_nll": 2.0, "validation_perplexity": 7.4,
                "quadratic_mode": run["quadratic_mode"], "quadratic_sharing": run["quadratic_sharing"]}
               for step in (2, 4)]
    monkeypatch.setattr(plot, "historical_summary", lambda *args: ({"best_step": 2}, records))
    row, _ = plot.summarize_run(run, manifest["source_commit"])
    assert row["best_step"] == 4
    assert row["best_ppl"] == 7.4


def test_report_has_six_labels_and_rejects_mislabeled_coefficients(tmp_path):
    from scaffolds.transformer.studies.learned_quadratic_attention.plot import aggregate, summarize_run
    base, manifest = make_test_manifest(tmp_path)
    sweep.prepare_sweep(manifest, base)
    report = Path(manifest["project_root"]) / "results/learned_quadratic_attention"
    authored = "# Experiment methods\n\nKeep the authored reproduction instructions.\n"
    (report / "README.md").write_text(authored)
    rows = aggregate(manifest)
    assert (report / "README.md").read_text().startswith(authored)
    assert len(rows) == 6 and len({row["kernel"] for row in rows}) == 6
    assert sum(row["status"] == "complete" for row in rows) == 2
    report = Path(manifest["project_root"]) / "results/learned_quadratic_attention"
    for name in ("summary.csv", "terminal_metrics.pdf", "learning_curves.pdf", "coefficient_trajectories.pdf", "stability.pdf", "throughput.pdf"):
        assert (report / ("figures" if name.endswith(".pdf") else "") / name).exists()
    assert "One seed per variant does not estimate seed variation" in (report / "README.md").read_text()
    run = manifest["runs"][0]
    record = {"event": "validation", "step": 2, "checkpoint_step": 2, "source_commit": manifest["source_commit"],
              "qk_head_dim": 64, "attention_degree": 2, "parameter_count": run["parameter_count"],
              "training_tokens": 1048576, "validation_nll": 2., "validation_perplexity": 7.4,
              "quadratic_mode": "unrestricted", "quadratic_sharing": "global"}
    (Path(run["workdir"]) / "metrics.jsonl").write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="family/sharing"):
        summarize_run(run, manifest["source_commit"])
