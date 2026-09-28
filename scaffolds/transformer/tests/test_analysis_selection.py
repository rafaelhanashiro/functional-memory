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


def test_selected_best_prefers_the_retained_checkpoint_only_at_a_minimum():
    from scaffolds.transformer.analysis import selected_best

    records = [{"step": 1, "validation_nll": 1.0}, {"step": 2, "validation_nll": 1.0}, {"step": 3, "validation_nll": 2.0}]
    assert selected_best({}, records)["step"] == 2
    assert selected_best({"selected_checkpoint": {"step": 1}}, records)["step"] == 1
    assert selected_best({"selected_checkpoint": {"step": 3}}, records)["step"] == 2
    assert selected_best({}, []) is None
