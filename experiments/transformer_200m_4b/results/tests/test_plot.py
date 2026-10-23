"""Pinned comparisons preserve incomplete outcomes and regenerate byte-identically."""

import copy
import csv
import json
import math
from pathlib import Path

import pytest
import yaml

from experiments.transformer_200m_4b.results import plot
from scaffolds.records import create_run, update_record
from scaffolds.records.selections import run_sources


STUDY = "transformer_200m_4b"
RECIPE = Path(__file__).resolve().parents[2] / "recipe.yaml"


def validation(step, nll):
    return {"event": "validation", "step": step, "validation_nll": nll,
            "validation_perplexity": math.exp(nll)}


def make_run(root, label, status="pending", measurements=(), *, changes=None, failure_step=None):
    recipe = copy.deepcopy(yaml.safe_load(RECIPE.read_text()))
    for section, fields in (changes or {}).items():
        recipe[section].update(fields)
    run = create_run(root / "experiments" / STUDY, recipe, label=label)
    update_record(run, {"status": status})
    if measurements:
        (run / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in measurements))
    if status == "complete":
        terminal = next(row for row in measurements if row["step"] == 7570)
        (run / "completion.json").write_text(json.dumps({"final_step": 7570, **terminal}))
    if failure_step is not None:
        (run / f"numerical_failure_step_{failure_step:08d}.json").write_text(
            json.dumps({"step": failure_step, "error": "nonfinite update"}))
    return run


def selection(root, runs):
    return {"runs": [{"study": STUDY, "run_id": run.name,
                      "label": json.loads((run / "record.json").read_text())["label"],
                      "sources": run_sources(root, STUDY, run.name)} for run in runs]}


def test_collect_rejects_changed_scientific_inputs(tmp_path):
    run = make_run(tmp_path, "pilot", "paused", [validation(1500, 3.0)])
    pinned = selection(tmp_path, [run])
    (run / "metrics.jsonl").write_text(json.dumps(validation(1500, 2.9)) + "\n")
    with pytest.raises(ValueError, match="Scientific inputs.*differ"):
        plot.collect(pinned, root=tmp_path)


@pytest.mark.parametrize("changes", [
    {"model": {"initialization": "transformer_reference"}},
    {"model": {"initialization": "flax_default"}},
    {"model": {"qk_norm_type": "cosine_log_scale"}},
    {"model": {"rms_norm_fp32": False}},
    {"model": {"num_layers": 10, "hidden_size": 1024}},
    {"optim": {"total_steps": 19074}},
])
def test_collect_rejects_other_training_recipes_even_with_valid_pins(tmp_path, changes):
    run = make_run(tmp_path, "legacy", changes=changes)
    with pytest.raises(ValueError, match="not part of the 200M/4B recipe"):
        plot.collect(selection(tmp_path, [run]), root=tmp_path)


def test_terminal_latest_and_numerical_failure_remain_distinct(tmp_path):
    done = make_run(tmp_path, "completed", "complete", [validation(1000, 2.0), validation(7570, 3.0)])
    (done / "best_validation.json").write_text(json.dumps(validation(1000, 2.0)))
    failed = make_run(tmp_path, "numerical", "numerical_failure",
                      [validation(1000, 3.2), validation(1100, 3.1)], failure_step=1140)
    paused = make_run(tmp_path, "pilot", "paused", [validation(1500, 3.05)])
    rows = plot.collect(selection(tmp_path, [done, failed, paused]), root=tmp_path)
    assert rows[0]["terminal"]["validation_nll"] == rows[0]["latest"]["validation_nll"] == 3.0
    assert rows[0]["terminal"]["final_step"] == 7570  # never substitute validation best
    assert rows[1]["latest"]["step"] == 1100 and rows[1]["failure_step"] == 1140
    assert rows[1]["terminal"] is None and rows[1]["status"] == "numerical_failure"
    assert rows[2]["terminal"] is None and rows[2]["latest"]["step"] == 1500


def test_outcome_contradictions_are_rejected_even_when_pins_are_current(tmp_path):
    run = make_run(tmp_path, "bad", "complete", [validation(7570, 3.0)])
    update_record(run, {"status": "failed"})
    with pytest.raises(ValueError, match="completion record but status"):
        plot.collect(selection(tmp_path, [run]), root=tmp_path)
    update_record(run, {"status": "complete"})
    (run / "completion.json").write_text(json.dumps({"final_step": 7570, **validation(7570, 2.0)}))
    with pytest.raises(ValueError, match="disagrees with terminal validation"):
        plot.collect(selection(tmp_path, [run]), root=tmp_path)


def test_regeneration_is_byte_identical_and_keeps_missing_and_failed_rows_visible(tmp_path, monkeypatch):
    pilot = make_run(tmp_path, "pilot", "paused", [validation(100, 4.0), validation(1500, 3.0)])
    done = make_run(tmp_path, "completed", "complete", [validation(7570, 2.9)])
    failed = make_run(tmp_path, "numerical", "numerical_failure", [validation(1000, 3.4)], failure_step=1131)
    pending = make_run(tmp_path, "pending")
    operational = make_run(tmp_path, "operational", "failed")
    directory = tmp_path / "experiments" / STUDY / "results"
    directory.mkdir()
    (directory / "lr_selection.yaml").write_text(yaml.safe_dump({"pilot_steps": 1500, **selection(tmp_path, [pilot])}))
    (directory / "selection.yaml").write_text(yaml.safe_dump(selection(tmp_path, [done, failed, pending, operational])))
    monkeypatch.setattr(plot, "DIRECTORY", directory)
    original_save, captured = plot.save, {}

    def capture(fig, name):
        if name == "attention_comparison":
            captured["labels"] = [label.get_text() for label in fig.axes[0].get_xticklabels()]
            captured["statuses"] = [label.get_text() for label in fig.axes[0].texts]
        original_save(fig, name)

    monkeypatch.setattr(plot, "save", capture)
    plot.regenerate()
    outputs = ("summary.csv", "learning_rates.png", "learning_rates.pdf",
               "attention_comparison.png", "attention_comparison.pdf")
    before = {name: (directory / name).read_bytes() for name in outputs}
    plot.regenerate()
    assert {name: (directory / name).read_bytes() for name in outputs} == before
    assert captured["labels"] == ["completed", "numerical", "pending", "operational"]
    assert captured["statuses"] == ["complete", "numerical_failure", "pending", "failed"]
    rows = {row["label"]: row for row in csv.DictReader((directory / "summary.csv").read_text().splitlines())}
    assert set(rows) == {"pilot", "completed", "numerical", "pending", "operational"}
    assert rows["numerical"]["failure_step"] == "1131"
    assert rows["numerical"]["latest_step"] == "1000"
    assert rows["numerical"]["terminal_validation_nll"] == ""
    assert rows["pending"]["latest_step"] == rows["operational"]["latest_step"] == ""


def test_comparison_with_no_measurements_still_renders_its_coordinates(tmp_path, monkeypatch):
    pending = make_run(tmp_path, "pending")
    directory = tmp_path / "experiments" / STUDY / "results"
    directory.mkdir()
    (directory / "lr_selection.yaml").write_text(yaml.safe_dump({"pilot_steps": 1500, "runs": []}))
    (directory / "selection.yaml").write_text(yaml.safe_dump(selection(tmp_path, [pending])))
    monkeypatch.setattr(plot, "DIRECTORY", directory)
    plot.regenerate()
    assert (directory / "attention_comparison.png").exists()
    assert "pending" in (directory / "summary.csv").read_text()
