"""Offline orchestration checks: immutable recipes, terminal selection, and independent submissions."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import yaml

from experiments.transformer_200m_4b import pipeline
from experiments.transformer_200m_4b.acceptance import check_config
from scaffolds.records import load_run, load_study, update_record
from scaffolds.transformer.configs import config_to_dict, load_config


@pytest.fixture
def study(tmp_path, monkeypatch):
    recipe = yaml.safe_load((pipeline.STUDY / "recipe.yaml").read_text())
    directory = tmp_path / "experiments/transformer_200m_4b"
    directory.mkdir(parents=True)
    for split in ("train", "eval"):
        manifest = tmp_path / f"{split}.json"
        manifest.write_text("{}")
        recipe["data"][f"{split}_manifest"] = str(manifest)
    directory.joinpath("recipe.yaml").write_text(yaml.safe_dump(recipe))
    lock = tmp_path / "scaffolds/transformer/uv.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("test lock")
    monkeypatch.setattr(pipeline, "STUDY", directory)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "validate_record", lambda *_args, **_kwargs: {"unavailable": []})
    runs = pipeline.prepare_pilots()
    for index, run in enumerate(runs):
        update_record(run, {"status": "paused", "step": 1500,
                            "acceptance": {"summary": {"status": "passed"}},
                            "submission": {"slurm_job_id": str(100 + index), "launch_commit": "pilot-source",
                                           "stop_after_step": 1500}})
        write_metrics(run, (100, 0.1), (1500, 3.0 + index))
    monkeypatch.setattr(pipeline, "_queued_jobs", lambda _jobs: "")
    return directory, runs


def write_metrics(run, *rows):
    (run / "metrics.jsonl").write_text("".join(json.dumps({
        "event": "validation", "step": step, "validation_nll": nll,
    }) + "\n" for step, nll in rows))


def test_pilots_preserve_full_schedule_and_differ_only_in_learning_rate(study):
    directory, runs = study
    assert len(runs) == 3 and len(set(runs)) == 3
    signatures = []
    for peak, run in zip(pipeline.PEAKS, runs, strict=True):
        config = load_config(run / "config.yaml")
        check_config(config)
        assert config.optim.learning_rate == peak and config.optim.min_learning_rate == peak * .1
        assert (config.optim.total_steps, config.optim.warmup_steps) == (7570, 455)
        assert (config.train.eval_early_every, config.train.eval_early_until, config.train.eval_every) == (100, 1500, 500)
        signatures.append(pipeline._recipe_signature(config, ignore_lr=True))
    assert signatures[0] == signatures[1] == signatures[2]
    before = [(run / "config.yaml").read_bytes() for run in runs]
    assert pipeline.prepare_pilots() == runs
    assert [(run / "config.yaml").read_bytes() for run in runs] == before
    selection = yaml.safe_load((directory / "results/lr_selection.yaml").read_text())
    assert selection["canonical_learning_rate"] is None
    assert all(entry["sources"]["config.yaml"] for entry in selection["runs"])


def test_choose_uses_terminal_pilot_loss_and_lower_peak_on_exact_tie(study):
    _, runs = study
    write_metrics(runs[0], (100, 0.01), (1500, 3.2))
    write_metrics(runs[1], (100, 2), (1500, 3))
    write_metrics(runs[2], (100, 1), (1500, 3))
    peak, outcomes = pipeline.choose_lr(runs)
    assert peak == .001
    assert [row["nll"] for row in outcomes] == [3.2, 3, 3]


def test_concluded_production_numerical_failure_remains_visible_but_ineligible(study):
    _, runs = study
    update_record(runs[0], {"status": "numerical_failure", "step": 100})
    peak, outcomes = pipeline.choose_lr(runs)
    assert peak == .001
    assert outcomes[0]["status"] == "numerical_failure" and outcomes[0]["nll"] is None
    for run in runs:
        update_record(run, {"status": "numerical_failure"})
    with pytest.raises(ValueError, match="All LR pilots failed"):
        pipeline.choose_lr(runs)


@pytest.mark.parametrize("status", ["pending", "running", "failed", "acceptance_failed", "acceptance_numerical_failure", "complete"])
def test_nonterminal_or_operational_outcome_blocks_selection(study, status):
    _, runs = study
    update_record(runs[0], {"status": status})
    with pytest.raises(ValueError, match="all pilots must finish"):
        pipeline.choose_lr(runs)


@pytest.mark.parametrize("rows", [((1400, 2),), ((1500, float("nan")),), ((1500, 2), (1500, 2)), ((1500, 2), (1600, 2))])
def test_missing_nonfinite_duplicate_or_nonterminal_measurement_blocks_selection(study, rows):
    _, runs = study
    write_metrics(runs[0], *rows)
    with pytest.raises(ValueError, match="step.?1500"):
        pipeline.choose_lr(runs)


def test_outcome_step_acceptance_artifacts_and_candidate_membership_are_required(study, monkeypatch):
    _, runs = study
    update_record(runs[0], {"step": 1400})
    with pytest.raises(ValueError, match="predeclared step"):
        pipeline.choose_lr(runs)
    update_record(runs[0], {"step": 1500, "acceptance": {"summary": {"status": "failed"}}})
    with pytest.raises(ValueError, match="acceptance"):
        pipeline.choose_lr(runs)
    update_record(runs[0], {"acceptance": {"summary": {"status": "passed"}}})
    monkeypatch.setattr(pipeline, "validate_record", lambda *_args, **_kwargs: {"unavailable": ["training.log"]})
    with pytest.raises(ValueError, match="unavailable"):
        pipeline.choose_lr(runs)
    monkeypatch.setattr(pipeline, "validate_record", lambda *_args, **_kwargs: {"unavailable": []})
    with pytest.raises(ValueError, match="exactly the three"):
        pipeline.choose_lr(runs[:2])


def test_different_pilot_protocol_is_not_an_lr_comparison(study):
    _, runs = study
    path = runs[1] / "config.yaml"
    config = yaml.safe_load(path.read_text())
    config["optim"]["weight_decay"] = .2
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="settings other than learning rate"):
        pipeline.choose_lr(runs)


def test_existing_label_cannot_silently_reuse_another_recipe_or_phase(study):
    directory, runs = study
    base = config_to_dict(load_config(directory / "recipe.yaml"))
    base["optim"].update(learning_rate=.0005, min_learning_rate=.00005)
    assert pipeline._allocate(base, "softmax-lr-0.0005", "lr_pilot") == runs[0]
    with pytest.raises(ValueError, match="different recipe or phase"):
        pipeline._allocate(base, "softmax-lr-0.0005", "comparison")
    base["optim"]["learning_rate"] = .003
    with pytest.raises(ValueError, match="different recipe or phase"):
        pipeline._allocate(base, "softmax-lr-0.0005", "lr_pilot")


def test_nine_comparison_configs_apply_entire_coefficient_scaling(study, tmp_path):
    directory, _ = study
    base = config_to_dict(load_config(directory / "recipe.yaml"))
    before = copy.deepcopy(base)
    assert len(pipeline.COMPARISONS) == 9
    for kind, degree in pipeline.COMPARISONS:
        config = pipeline.variant_config(base, kind, degree)
        path = tmp_path / "variant.yaml"
        path.write_text(yaml.safe_dump(config))
        check_config(load_config(path))
        model = config["model"]
        assert model["attention_normalization"] is False
        assert model["attention_output_scaling"] == "head_rms_norm"
        assert model["attention_backend"] == ("fused" if kind in ("exp", "query_key") else "mosaic")
        if kind in ("static", "query", "query_key"):
            assert model["attention_coefficient_scale"] == "factorial"
        if kind in ("query", "query_key"):
            assert model["attention_coefficient_amplitude_scale"] == "factorial"
            assert model["attention_coefficient_amplitude_initial"] == 0
    assert base == before


@pytest.mark.parametrize("stop", [None, 1500])
def test_new_submission_requests_one_gpu_and_preserves_scientific_recipe(study, monkeypatch, stop):
    directory, _ = study
    base = config_to_dict(load_config(directory / "recipe.yaml"))
    run = pipeline._allocate(base, "single-gpu", "comparison")
    before = (run / "config.yaml").read_bytes()
    commands = []
    monkeypatch.setattr(pipeline.subprocess, "check_output",
                        lambda command, **_kwargs: commands.append(command) or "600\n")
    assert pipeline._submit(run, "source", stop=stop) == "600"
    command, = commands
    assert {"--gres=gpu:1", "--cpus-per-task=8", "--mem=96G", "--time=2-00:00:00"} <= set(command)
    assert not any(argument.startswith("--dependency=") for argument in command)
    launcher = command.index("scaffolds/transformer/launchers/kennel.sbatch")
    assert command[launcher + 1:] == [
        "experiments.transformer_200m_4b.acceptance",
        f"experiments/transformer_200m_4b/runs/{run.name}/config.yaml", "source", "1", "B300,H100",
        *([str(stop)] if stop is not None else []),
    ]
    record = load_run(run)
    assert record["submission"]["gpu_count"] == 1
    assert record["submission"]["stop_after_step"] == stop
    assert (run / "config.yaml").read_bytes() == before
    frozen = load_config(run / "config.yaml")
    assert frozen.data.batch_size == 256 and frozen.train.micro_batch_size == 16


@pytest.mark.parametrize("degree", [8, 12])
def test_query_key_submission_requests_one_b300_and_seven_days_without_changing_recipe(study, monkeypatch, degree):
    directory, _ = study
    base = config_to_dict(load_config(directory / "recipe.yaml"))
    run = pipeline._allocate(pipeline.variant_config(base, "query_key", degree),
                             f"query-key-{degree}", "comparison")
    before = (run / "config.yaml").read_bytes()
    commands = []
    monkeypatch.setattr(pipeline.subprocess, "check_output",
                        lambda command, **_kwargs: commands.append(command) or "601\n")
    assert pipeline._submit(run, "source") == "601"
    command, = commands
    assert {"--gres=gpu:b300:1", "--cpus-per-task=8", "--mem=96G", "--time=7-00:00:00"} <= set(command)
    assert not any(argument.startswith("--dependency=") for argument in command)
    assert command[-2:] == ["1", "B300"]
    assert load_run(run)["submission"]["gpu_count"] == 1
    assert (run / "config.yaml").read_bytes() == before
    frozen = load_config(run / "config.yaml")
    assert frozen.model.attention_backend == "fused" and frozen.model.attention_degree == degree


def test_pilots_schedule_independently_and_selection_waits_for_all_three(study, monkeypatch):
    _, runs = study
    monkeypatch.setattr(pipeline, "_git", lambda *_args: "source")
    monkeypatch.setattr("scaffolds.launch.frozen.changes", lambda *_args: [])
    submissions, commands = [], []
    def submit(run, source, **kwargs):
        submissions.append((run, source, kwargs))
        return str(200 + len(submissions))
    monkeypatch.setattr(pipeline, "_submit", submit)
    monkeypatch.setattr(pipeline.subprocess, "check_output", lambda command, **_kwargs: commands.append(command) or "500\n")
    pipeline.submit_pilots()
    assert [item[2] for item in submissions] == [{"stop": 1500}] * 3
    assert "--dependency=afterany:201:202:203" in commands[0]
    assert all(load_run(run)["lr_selection_job_id"] == "500" for run in runs)


def test_existing_operational_failure_or_stale_job_is_not_silently_reused(study):
    _, runs = study
    update_record(runs[0], {"status": "failed"})
    with pytest.raises(ValueError, match="diagnosed and explicitly requeued"):
        pipeline._submit(runs[0], "source", stop=1500)
    update_record(runs[0], {"status": "pending"})
    with pytest.raises(ValueError, match="stale submission"):
        pipeline._submit(runs[0], "source", stop=1500)


def _offline_advance(monkeypatch):
    calls = []
    staged = False
    def git(*args):
        nonlocal staged
        calls.append(args)
        if args[0] == "add":
            staged = True
        elif args[0] == "commit":
            staged = False
        if args[0] == "rev-parse":
            return "HEAD"
        if args[:3] == ("diff", "--cached", "--name-only") and staged:
            return "experiments/transformer_200m_4b/recipe.yaml"
        return ""
    monkeypatch.setattr(pipeline, "_git", git)
    monkeypatch.setattr("scaffolds.launch.frozen.changes", lambda *_args: [])
    monkeypatch.setitem(sys.modules, "experiments.transformer_200m_4b.results.plot", SimpleNamespace(regenerate=lambda: None))
    return calls


def test_advance_freezes_nine_recipes_before_any_submission_and_schedules_report(study, monkeypatch):
    directory, pilots = study
    before = [(run / "config.yaml").read_bytes() for run in pilots]
    calls = _offline_advance(monkeypatch)
    submissions, reports = [], []
    def submit(run, source, **kwargs):
        assert len([row for row in load_study(directory)["runs"] if row.get("phase") == "comparison"]) == 9
        assert any(call[0] == "commit" for call in calls)
        submissions.append((run, kwargs))
        return str(300 + len(submissions))
    monkeypatch.setattr(pipeline, "_submit", submit)
    monkeypatch.setattr(pipeline, "_submit_report", lambda runs, jobs: reports.append((runs, jobs)))
    comparisons = pipeline.advance(submit=True)
    assert [(run / "config.yaml").read_bytes() for run in pilots] == before
    assert len(comparisons) == 9
    assert yaml.safe_load((directory / "recipe.yaml").read_text())["optim"]["learning_rate"] == .0005
    assert all(not kwargs.get("dependency") for _, kwargs in submissions)
    assert reports == [(comparisons, [str(i) for i in range(301, 310)])]
    with pytest.raises(ValueError, match="already finalized"):
        pipeline.prepare_pilots()


def test_advance_refuses_live_jobs_and_changes_since_launch(study, monkeypatch):
    directory, _ = study
    _offline_advance(monkeypatch)
    original = (directory / "recipe.yaml").read_bytes()
    monkeypatch.setattr(pipeline, "_queued_jobs", lambda jobs: jobs[0])
    with pytest.raises(ValueError, match="can still execute"):
        pipeline.advance(submit=False)
    monkeypatch.setattr(pipeline, "_queued_jobs", lambda _jobs: "")
    monkeypatch.setattr("scaffolds.launch.frozen.changes",
                        lambda _root, source, *_args: ["changed.py"] if source == "pilot-source" else [])
    with pytest.raises(ValueError, match="changed since the LR pilots"):
        pipeline.advance(submit=False)
    assert (directory / "recipe.yaml").read_bytes() == original


def test_advance_refuses_default_recipe_drift(study, monkeypatch):
    directory, _ = study
    _offline_advance(monkeypatch)
    path = directory / "recipe.yaml"
    recipe = yaml.safe_load(path.read_text())
    recipe["optim"]["weight_decay"] = .2
    path.write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError, match="differs from the frozen LR pilot protocol"):
        pipeline.advance(submit=False)


def test_report_submission_waits_for_every_comparison_and_records_its_job(study, monkeypatch):
    _, runs = study
    commands = []
    monkeypatch.setattr(pipeline.subprocess, "check_output", lambda command, **_kwargs: commands.append(command) or "800\n")
    assert pipeline._submit_report(runs, ["301", "302", "303"]) == "800"
    assert "--dependency=afterany:301:302:303" in commands[0]
    assert commands[0][-1] == "report"
    assert all(load_run(run)["report_job_id"] == "800" for run in runs)


def test_report_refreshes_outputs_without_allocating_or_mutating_recipes(study, monkeypatch):
    directory, runs = study
    calls = []
    before = [(run / "config.yaml").read_bytes() for run in runs]
    monkeypatch.setattr(pipeline, "refresh_selection", lambda path: calls.append(("selection", path)))
    monkeypatch.setattr(pipeline, "regenerate_index", lambda path: calls.append(("index", path)))
    monkeypatch.setitem(sys.modules, "experiments.transformer_200m_4b.results.plot",
                        SimpleNamespace(regenerate=lambda: calls.append(("plot", directory))))
    pipeline.report()
    assert [kind for kind, _ in calls] == ["selection", "index", "plot"]
    assert [(run / "config.yaml").read_bytes() for run in runs] == before
