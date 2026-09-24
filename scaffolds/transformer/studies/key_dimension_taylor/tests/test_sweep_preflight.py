"""Synthetic provenance and scheduler fixtures never submit real jobs."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.studies.polynomial_attention.preflight import record_success
from scaffolds.transformer.studies.key_dimension_taylor import preflight, production, sweep

BASE = Path(__file__).resolve().parents[5] / "results/polynomial_attention/configs/base.yaml"


def make_fixture(tmp_path, widths=sweep.KEY_DIMENSIONS, degrees=sweep.DEGREES):
    root, execution = tmp_path / "project", tmp_path / "execution"
    base = load_config(BASE)
    base.model.attention_backend = "fused"
    for key in ("train_manifest", "eval_manifest"):
        path = tmp_path / f"{key}.json"
        path.write_text('{"fixture": true}')
        setattr(base.data, key, str(path))
    data = {key: {"path": str(getattr(base.data, key)), "sha256": sweep.sha256(Path(getattr(base.data, key)))}
            for key in ("train_manifest", "eval_manifest")}
    lock = execution / "scaffolds/transformer/uv.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("fixture lock")
    for experiment, pairs in (("polynomial_attention", [(64, degree) for degree in sweep.DEGREES] + [(64, None)]),
                              ("key_dimension", [(width, None) for width in sweep.KEY_DIMENSIONS])):
        retained = []
        for width, degree in pairs:
            name = (f"200m-dk{width:03d}-seed0" if experiment == "key_dimension" else
                    f"200m-m{degree:03d}-seed0" if degree is not None else "200m-softmax-seed0")
            config = copy.deepcopy(base)
            config.model.qk_head_dim, config.model.attention_degree = width, degree
            config.train.source_commit = "b" * 40
            config.train.workdir = root / "results" / experiment / name
            config.train.wandb_run_name = name
            raw = config_to_dict(config)
            if experiment == "key_dimension":
                raw["model"].pop("attention_degree")
                raw["model"]["attention_backend"] = "cudnn"
                raw["data"]["train_manifest"] = "/unavailable/original/train.json"
                raw["data"]["eval_manifest"] = "/unavailable/original/eval.json"
            workdir = config.train.workdir
            workdir.mkdir(parents=True)
            config_path = root / "experiments" / experiment / "configs/runs" / f"{name}.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(yaml.safe_dump(raw))
            tokens = config.optim.total_steps * config.data.batch_size * (config.data.sequence_length - 1)
            record = {"event": "validation", "step": 19074, "checkpoint_step": 19074,
                      "source_commit": "b" * 40, "qk_head_dim": width, "attention_degree": degree,
                      "parameter_count": sweep.parameter_count(config.model), "training_tokens": tokens,
                      "validation_nll": 2.8, "validation_perplexity": 16.0, "tokens_per_second": 100.}
            if experiment == "key_dimension":
                record.pop("attention_degree")
            metrics = workdir / "metrics.jsonl"
            metrics.write_text(json.dumps(record) + "\n")
            completion = workdir / "completion.json"
            completion.write_text(json.dumps({**record, "final_step": 19074}))
            checkpoint = root / "experiments" / experiment / "checkpoints" / name / "19074"
            checkpoint.mkdir(parents=True)
            metadata = checkpoint / "_CHECKPOINT_METADATA"
            metadata.write_text('{"fixture": true}')
            log = root / "logs" / experiment / name / "slurm-99.out"
            log.parent.mkdir(parents=True)
            log.write_text("complete\n")
            def artifact(path):
                return {"path": str(path.relative_to(root)), "sha256": sweep.sha256(path)}
            retained.append({"name": name, "qk_head_dim": width, "value_head_dim": 64,
                             "attention_degree": degree, "parameter_count": record["parameter_count"],
                             "seed": 0, "source_commit": "b" * 40, "resolved_config": artifact(config_path),
                             "metrics": artifact(metrics), "completion": str(completion.relative_to(root)),
                             "slurm_log": artifact(log), "selected_checkpoint": {
                                 "path": str(checkpoint.relative_to(root)), "metadata_sha256": sweep.sha256(metadata),
                                 **{key: record[key] for key in ("step", "validation_nll", "validation_perplexity")}},
                             "terminal": {key: record[key] for key in ("step", "training_tokens", "validation_nll", "validation_perplexity")}})
        sweep.write_json(root / "results" / experiment / "provenance" / "manifest.json", {
            "format": "functional-memory-retained-experiment", "version": 1, "experiment": experiment,
            "dependency_lock_sha256": "old-lock", "data_identity": data, "runs": retained})
    manifest = sweep.make_manifest(base, root, execution, "a" * 40, widths, degrees)
    return base, manifest


def mark_passed(manifest, rejected=()):
    root = Path(manifest["preflight_dir"])
    record_success(root / "success.json", manifest)
    for name in preflight.SHARED_CHECKS:
        record_success(root / "checks" / f"{name}.json", manifest)
    for run in sweep.active_runs(manifest):
        path = root / "checks" / f"{run['name']}.json"
        if run["name"] in rejected:
            sweep.write_json(path, {"identity": manifest["identity"], "status": "ineligible", "reason": "numerical"})
        else:
            record_success(path, manifest, runtime={"segments": [
                {"start_step": 0, "end_step": 10000, "projected_seconds": 20000.},
                {"start_step": 10000, "end_step": 19074, "projected_seconds": 19000.}]})


def test_default_grid_reuses_exact_references_and_preserves_recipe(tmp_path):
    base, manifest = make_fixture(tmp_path)
    assert len(manifest["runs"]) == 20 and len(sweep.active_runs(manifest)) == 16
    assert len(manifest["controls"]) == 5
    assert [(row["qk_head_dim"], row["attention_degree"]) for row in manifest["runs"]] == [
        (width, degree) for width in sweep.KEY_DIMENSIONS for degree in sweep.DEGREES]
    for run in manifest["runs"]:
        assert run["config"]["optim"]["total_steps"] == 19074 and run["training_tokens"] == 10000269312
        assert run["parameter_count"] == 199906304 + (run["qk_head_dim"] - 64) * 327680
        assert run["value_head_dim"] == 64
        assert run["retained_completed"] == (run["qk_head_dim"] == 64)
    assert all(row["original_experiment"] == "key_dimension" for row in manifest["controls"])
    assert manifest["alternative_softmax64"]["original_experiment"] == "polynomial_attention"
    assert manifest["scheduler"]["node"] == "dogo" and manifest["scheduler"]["gpu_count"] == 4
    assert manifest["execution_policy"]["allocation_fraction"] == .8


def test_subsets_are_canonical_cartesian_grids_and_identity_changes(tmp_path):
    base, full = make_fixture(tmp_path)
    args = base, Path(full["project_root"]), Path(full["execution_root"]), full["source_commit"]
    subset = sweep.make_manifest(*args, (128, 8), (16, 2))
    assert [(row["qk_head_dim"], row["attention_degree"]) for row in subset["runs"]] == [(8, 2), (8, 16), (128, 2), (128, 16)]
    assert subset["identity"] != full["identity"]
    assert len(subset["controls"]) == 2
    with pytest.raises(ValueError, match="distinct"):
        sweep.make_manifest(*args, (8, 8), (2,))


def test_historical_hashes_and_recipe_are_strict(tmp_path):
    base, manifest = make_fixture(tmp_path)
    retained = manifest["controls"][0]
    path = Path(retained["metrics_path"])
    before = path.read_bytes()
    path.write_bytes(before + b"\n")
    with pytest.raises(ValueError, match="Historical artifact changed"):
        sweep.validate_retained_run(retained)
    path.write_bytes(before)
    base.optim.learning_rate *= 2
    with pytest.raises(ValueError, match="scientific recipe"):
        sweep.make_manifest(base, Path(manifest["project_root"]), Path(manifest["execution_root"]), manifest["source_commit"])


def test_missing_historical_logs_and_checkpoints_are_explicit(tmp_path):
    _, manifest = make_fixture(tmp_path)
    root = Path(manifest["project_root"])
    old = manifest["controls"][0]
    original = old["historical_provenance"]
    (root / original["slurm_log"]["path"]).unlink()
    (root / original["selected_checkpoint"]["path"] / "_CHECKPOINT_METADATA").unlink()
    reference = sweep.retained_runs(root, "key_dimension", [old["name"]])[0]
    assert {item["kind"] for item in reference["unavailable_historical_artifacts"]} == {"slurm_log", "selected_checkpoint"}
    assert all(Path(path).exists() for path in reference["provenance_sha256"])
    assert reference["historical_provenance"] == original


def test_prepare_writes_only_active_run_artifacts_outside_checkout(tmp_path):
    _, manifest = make_fixture(tmp_path)
    historical = {path: Path(path).read_bytes() for run in [*manifest["controls"], *manifest["runs"]]
                  if run.get("retained_completed") for path in run["provenance_sha256"]}
    sweep.prepare_sweep(manifest)
    sweep.prepare_sweep(manifest)
    for run in sweep.active_runs(manifest):
        assert Path(run["config_path"]).is_file()
        assert (Path(run["workdir"]) / "checkpoints").resolve() == Path(run["checkpoint_dir"])
        assert not str(Path(run["checkpoint_dir"])).startswith(manifest["execution_root"])
    assert all(Path(path).read_bytes() == content for path, content in historical.items())


def test_identity_freeze_gate(tmp_path, monkeypatch):
    _, manifest = make_fixture(tmp_path)
    sweep.prepare_sweep(manifest)
    monkeypatch.setattr(preflight.subprocess, "check_output", lambda command, **kwargs: manifest["source_commit"] if "rev-parse" in command else "")
    preflight.verify_manifest(manifest)
    manifest["runs"][0]["qk_head_dim"] = 64
    with pytest.raises(ValueError, match="identity"):
        preflight.verify_manifest(manifest)


def test_eligibility_is_independent_for_every_coordinate(tmp_path):
    _, manifest = make_fixture(tmp_path)
    rejected = sweep.active_runs(manifest)[0]["name"]
    mark_passed(manifest, (rejected,))
    assert preflight.preflight_complete(manifest)
    assert len(preflight.eligible_runs(manifest)) == 15
    assert rejected not in preflight.eligible_runs(manifest)
    (Path(manifest["preflight_dir"]) / "checks/kernels-gpu.json").unlink()
    assert not preflight.preflight_complete(manifest) and not preflight.eligible_runs(manifest)


def timing_fixture(step_seconds=2.):
    return {"timing": {"runtime_seconds": 200., "train_step_seconds": [100., step_seconds],
                       "evaluation_seconds": [3.], "checkpoint_seconds": [10.]}}


def test_runtime_segments_use_eighty_percent_and_checkpoint_boundaries(tmp_path):
    _, manifest = make_fixture(tmp_path)
    run = sweep.active_runs(manifest)[0]
    runtime = preflight.projected_runtime(run, timing_fixture(), timing_fixture(), manifest["scheduler"])
    assert runtime["usable_seconds"] == 23040 and len(runtime["segments"]) > 1
    segments = runtime["segments"]
    assert segments[0]["start_step"] == 0 and segments[-1]["end_step"] == 19074
    for previous, current in zip(segments, segments[1:]):
        assert previous["end_step"] == current["start_step"]
    for segment in segments:
        assert segment["projected_seconds"] <= 23040
        assert segment["end_step"] == 19074 or segment["end_step"] % 500 == 0
    with pytest.raises(preflight.RuntimeLimitExceeded):
        preflight.projected_runtime(run, timing_fixture(100.), timing_fixture(), manifest["scheduler"])


def test_segments_depend_afterok_but_next_coordinate_afterany(tmp_path, monkeypatch):
    _, manifest = make_fixture(tmp_path, (8, 16), (2,))
    manifest["jobs"]["preflight"] = "90"
    mark_passed(manifest)
    monkeypatch.setattr(preflight, "verify_manifest", lambda manifest: None)
    monkeypatch.setattr("scaffolds.transformer.studies.key_dimension_taylor.plot.aggregate", lambda manifest: [])
    calls = []
    def submit(command, execution, dependency=None, dependency_type="afterok"):
        calls.append((command, dependency, dependency_type))
        assert "--kill-on-invalid-dep=yes" in command
        return str(100 + len(calls))
    monkeypatch.setattr(sweep, "submit_job", submit)
    result = sweep.submit_sweep(manifest)
    assert [(dep, kind) for _, dep, kind in calls] == [("90", "afterok"), ("101", "afterok"), ("102", "afterany"), ("103", "afterok")]
    assert all(f"--dependency={kind}:{dep}" in command for command, dep, kind in calls)
    assert result["jobs"][manifest["runs"][0]["name"]] == ["101", "102"]
    # A failed first segment cancels its dependent tail; afterany on the canceled
    # last job still releases the next coordinate instead of suppressing it.
    assert calls[1][2] == "afterok" and calls[2][2] == "afterany"
    sweep.submit_sweep(result)
    assert len(calls) == 4


def audit(start, final):
    def update(step):
        return {"step": step, "batch_sha256": f"{step:064x}", "rng_sha256": f"{step + 1:064x}"}
    return {"start_state_sha256": f"{start:064x}", "final_state_sha256": f"{final:064x}",
            "checkpoint_state_sha256": f"{final:064x}", "first_update": update(start + 1), "next_update": update(final + 1)}


@pytest.mark.parametrize("change", ["state", "batch", "rng", "step"])
def test_segment_restart_requires_exact_state_and_next_input(change):
    previous = {"final_step": 500, "checkpoint_audit": audit(0, 500)}
    restored = {"restored_step": 500, "state_sha256": previous["checkpoint_audit"]["final_state_sha256"],
                "next_update": copy.deepcopy(previous["checkpoint_audit"]["next_update"])}
    production.validate_segment_restart(previous, restored, 500)
    if change == "state":
        restored["state_sha256"] = "f" * 64
    elif change == "step":
        restored["restored_step"] -= 1
    else:
        restored["next_update"][f"{change}_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="restart changed"):
        production.validate_segment_restart(previous, restored, 500)


def test_all_preflight_gates_route_correctly_and_pilot_failure_is_independent(tmp_path, monkeypatch):
    _, manifest = make_fixture(tmp_path, (8, 16), (2,))
    sweep.prepare_sweep(manifest)
    monkeypatch.setattr(preflight, "verify_manifest", lambda manifest: None)
    rejected = manifest["runs"][0]["name"]
    calls, tests = [], {}
    def worker(manifest, kind, output, *, name="", workdir=None, stop=32, cpu=False):
        calls.append((kind, name, stop))
        if kind == "hardware":
            return {"device_kinds": ["H100"] * 4}
        if kind == "initialization":
            width = int(name.removeprefix("reference-dk")) if name.startswith("reference-dk") else next(run["qk_head_dim"] for run in manifest["runs"] if run["name"] == name)
            return {"parameter_count": 199906304 + (width - 64) * 327680, "parameter_sha256": f"dk{width}"}
        if name == rejected:
            workdir.mkdir(parents=True)
            (workdir / "checkpoints").mkdir()
            sweep.write_json(workdir / "numerical_failure_step_00000002.json", {"step": 2, "error": "nonfinite"})
            raise preflight.NumericalPilotFailure("nonfinite")
        return {"final_step": stop, "start_step": 32 if stop == 33 else 0,
                "restored_step": 32 if stop == 33 else None, "data_start_sequence": 8192 if stop == 33 else 0,
                "data_parallel_devices": 4, "checkpoint_audit": audit(32 if stop == 33 else 0, stop),
                **timing_fixture(.5)}
    def command(args, **kwargs):
        xml = Path(next(arg.removeprefix("--junitxml=") for arg in args if arg.startswith("--junitxml=")))
        count = {"kernels-cpu": 20, "training-audit-cpu": 29, "training-distributed-cpu": 1, "kernels-gpu": 105}[xml.stem]
        tests[xml.stem] = args, kwargs
        xml.write_text('<testsuite>' + '<testcase name="check"/>' * count + '</testsuite>')
    monkeypatch.setattr(preflight, "launch_worker", worker)
    monkeypatch.setattr(preflight.subprocess, "run", command)
    monkeypatch.setattr("scaffolds.transformer.studies.polynomial_attention.plot.committed_records", lambda root: [{"step": 32, "validation_nll": 3., "validation_perplexity": 20.}])
    preflight.run_preflight(manifest)
    assert preflight.preflight_complete(manifest)
    assert preflight.eligible_runs(manifest) == {manifest["runs"][1]["name"]}
    assert ("pilot", manifest["runs"][1]["name"], 33) in calls
    for stage, (command, kwargs) in tests.items():
        assert kwargs["env"]["JAX_PLATFORMS"] == ("cuda" if stage == "kernels-gpu" else "cpu")
        if stage == "training-distributed-cpu":
            assert kwargs["env"]["XLA_FLAGS"] == "--xla_force_host_platform_device_count=2"
            assert command[command.index("-k") + 1] == "distributed"
    record = json.loads((Path(manifest["runs"][0]["workdir"]) / "preflight_failure.json").read_text())
    assert record["numerical_failures"][0]["record"]["step"] == 2
    assert not list(Path(manifest["preflight_dir"]).glob("attempts/*/*/checkpoints"))


def test_numerical_worker_failure_survives_subprocess_boundary(tmp_path, monkeypatch):
    _, manifest = make_fixture(tmp_path, (8,), (2,))
    output = tmp_path / "worker.json"
    sweep.write_json(output, {"status": "failed", "error": "nonfinite optimizer state", "numerical_failure": True})
    monkeypatch.setattr(production.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 1))
    with pytest.raises(FloatingPointError, match="nonfinite optimizer state"):
        production.launch_worker(manifest, manifest["runs"][0], "train", 500, output)


def test_launcher_failure_receipt_preserves_existing_numerical_failure(tmp_path, monkeypatch):
    _, manifest = make_fixture(tmp_path, (8,), (2,))
    sweep.prepare_sweep(manifest)
    run = manifest["runs"][0]
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    first = production.record_interruption(manifest, run, 0, 143)
    assert first["failure_kind"] == "infrastructure" and first["exit_code"] == 143
    path = Path(run["workdir"]) / "production_failure.json"
    numerical = {"failure_kind": "numerical", "slurm_job_id": "123", "error": "bad state"}
    sweep.write_json(path, numerical)
    assert production.record_interruption(manifest, run, 0, 1) == numerical
