"""Saved comparisons pin their run identity and scientific inputs."""
import json

import pytest
import yaml

from scaffolds.transformer.analysis import verify_result_selection


def test_result_selection_pins_exactly_the_scientific_inputs(tmp_path):
    from scaffolds.records.selections import run_sources

    run = tmp_path / "experiments/toy/runs/r000000000001"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text("model: {}\n")
    (run / "record.json").write_text("{}\n")
    metrics = run / "metrics.jsonl"
    metrics.write_text(json.dumps({"validation_perplexity": 12.0}) + "\n")
    result = tmp_path / "results/toy"
    result.mkdir(parents=True)
    selection = {"runs": [{"study": "toy", "run_id": run.name, "sources": run_sources(tmp_path, "toy", run.name)}]}
    assert set(selection["runs"][0]["sources"]) == {"config.yaml", "metrics.jsonl"}
    (result / "selection.yaml").write_text(yaml.safe_dump(selection))
    assert verify_result_selection(result) == selection
    # Provenance is not a plotted input.
    (run / "record.json").write_text(json.dumps({"attempts": [{"node": "n1"}]}) + "\n")
    assert verify_result_selection(result) == selection
    metrics.write_text(metrics.read_text() + "\n")
    with pytest.raises(ValueError, match="metrics.jsonl"):
        verify_result_selection(result)
    metrics.write_text(json.dumps({"validation_perplexity": 12.0}) + "\n")
    (run / "numerical_failure_step_00000003.json").write_text("{}\n")
    with pytest.raises(ValueError, match="numerical_failure_step_00000003.json"):
        verify_result_selection(result)


def test_selected_ids_change_inputs_without_changing_plotting_code(tmp_path):
    from scaffolds.records import create_run
    from scaffolds.records.selections import run_sources
    from scaffolds.transformer.analysis import selected_run_directory, selected_views

    def recipe(value):
        return {"model": {"qk_head_dim": 8, "value_head_dim": 8}, "data": {"batch_size": 1, "sequence_length": 2},
                "optim": {"total_steps": 1}, "value": value}

    study = tmp_path / "experiments/toy"
    first = create_run(study, recipe(1), label="old execution")
    second = create_run(study, recipe(2), label="new execution")
    result = tmp_path / "results/toy"
    result.mkdir(parents=True)

    def select(run):
        sources = run_sources(tmp_path, "toy", run.name)
        selection = {"runs": [{"study": "toy", "run_id": run.name, "label": "baseline", "sources": sources}]}
        (result / "selection.yaml").write_text(yaml.safe_dump(selection))

    select(first)
    assert selected_run_directory(result, study, "baseline") == first
    assert [r["run_id"] for r in selected_views(result, study)["runs"]] == [first.name]
    select(second)
    assert selected_run_directory(result, study, "baseline") == second
    selected = selected_views(result, study)["runs"]
    assert [r["run_id"] for r in selected] == [second.name]
    assert selected[0]["name"] == "baseline"
    with pytest.raises(ValueError, match="Expected one selected run"):
        selected_run_directory(result, study, first.name)


def _run(tmp_path, name, *, metadata, completion=None, failure=None, record_extra=None):
    from pathlib import Path

    run = Path(tmp_path) / "experiments/toy/runs" / name
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump({
        "model": {"qk_head_dim": 64, "value_head_dim": 64, "attention_degree": 3},
        "data": {"batch_size": 256, "sequence_length": 2049}, "optim": {"total_steps": 10}}))
    (run / "record.json").write_text(json.dumps({"run_id": name, "label": "toy", "source_commit": "abc",
                                                 "quadratic_sharing": "global", **(record_extra or {})}))
    (run / "metadata.json").write_text(json.dumps(metadata))
    if completion is not None:
        (run / "completion.json").write_text(json.dumps(completion))
    if failure is not None:
        (run / "numerical_failure_step_00000007.json").write_text(json.dumps(failure))
    return run


def test_run_view_reads_identity_from_the_recipe_and_status_from_raw_outputs(tmp_path):
    from scaffolds.transformer.analysis import run_view

    complete = run_view(_run(tmp_path, "r000000000001", metadata={"status": "complete", "parameter_count": 5},
                             completion={"parameter_count": 7}), "label")
    assert (complete["name"], complete["status"], complete["source_commit"]) == ("label", "complete", "abc")
    assert (complete["attention_degree"], complete["qk_head_dim"], complete["parameter_count"]) == (3, 64, 7)
    # Settings an older recipe left implicit come from the record, then from defaults.
    assert (complete["quadratic_sharing"], complete["quadratic_mode"]) == ("global", "fixed")
    assert complete["training_tokens"] == 10 * 256 * 2048
    failed = run_view(_run(tmp_path, "r000000000002", metadata={"status": "failed", "error": "E: nan"},
                           failure={"step": 7, "error": "nan"}))
    assert (failed["status"], failed["error"], failed["failure"]["step"]) == ("numerical-failure", "E: nan", 7)


def test_selected_best_uses_pinned_validation_best_and_ignores_provenance():
    from scaffolds.transformer.analysis import selected_best

    records = [{"step": 1, "validation_nll": 1.0, "validation_perplexity": 2.7},
               {"step": 2, "validation_nll": 1.0, "validation_perplexity": 2.7},
               {"step": 3, "validation_nll": 2.0, "validation_perplexity": 7.4}]
    assert selected_best({}, records)["step"] == 2
    run = {"best_validation": records[0], "selected_checkpoint": {"step": 2}}
    assert selected_best(run, records)["step"] == 1
    run["selected_checkpoint"] = {"step": 3}
    assert selected_best(run, records)["step"] == 1
    for best in (records[2], dict(records[0], validation_perplexity=999), dict(records[0], step=4)):
        with pytest.raises(ValueError, match="validation-best"):
            selected_best({"best_validation": best}, records)
    assert selected_best({}, []) is None


def test_run_view_keeps_numerical_failure_visible_without_status_metadata(tmp_path):
    from scaffolds.transformer.analysis import run_view

    directory = _run(tmp_path, "r000000000001", metadata={}, failure={"step": 7, "error": "nan"})
    (directory / "metadata.json").unlink()
    view = run_view(directory)
    assert (view["status"], view["error"]) == ("numerical-failure", "nan")
    (directory / "completion.json").write_text("{}\n")
    with pytest.raises(ValueError, match="Numerical failure has a completion"):
        run_view(directory)


@pytest.mark.parametrize("changed", [{"validation_nll": 99}, {"validation_perplexity": 99}, {"qk_head_dim": 8}])
def test_summary_rejects_completion_disagreement_with_terminal_validation(tmp_path, changed):
    from scaffolds.transformer.analysis import run_view, summarize_run

    identity = {"source_commit": "abc", "attention_degree": 3, "qk_head_dim": 64, "parameter_count": 7}
    final = {**identity, "event": "validation", "step": 10, "checkpoint_step": 10,
             "training_tokens": 10 * 256 * 2048, "validation_nll": 1.0, "validation_perplexity": 2.7}
    directory = _run(tmp_path, "r000000000001", metadata={**identity, "status": "complete"},
                     completion={**final, "final_step": 10, **changed})
    (directory / "metrics.jsonl").write_text(json.dumps(final) + "\n")
    with pytest.raises(ValueError, match="Completion"):
        summarize_run(run_view(directory), "abc")
