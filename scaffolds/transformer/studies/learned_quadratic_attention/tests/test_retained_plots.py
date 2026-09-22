import hashlib
import json
import shutil

import pytest
import yaml

from scaffolds.transformer.studies.learned_quadratic_attention.plot import retained_plot_manifest
from scaffolds.transformer.studies.odd_taylor_attention.plot import aggregate


def retained_fixture(root):
    family = "odd_taylor_attention"
    results = root / "results" / family
    records = [
        {"event": "validation", "step": step, "checkpoint_step": step, "training_tokens": step * 100,
         "source_commit": "a" * 40, "qk_head_dim": 64, "value_head_dim": 64, "attention_degree": 2,
         "parameter_count": 199906304, "validation_nll": nll, "validation_perplexity": ppl}
        for step, nll, ppl in [(5, 2.0, 7.39), (10, 2.1, 8.17)]
    ]

    def store(relative, content):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    runs = []
    for name, degree, complete in [("200m-m002-seed0", 2, True), ("200m-m001-seed0", 1, False)]:
        directory = f"results/{family}/runs/{name}"
        config = {"model": {"attention_degree": degree, "qk_head_dim": 64, "value_head_dim": 64},
                  "train": {"source_commit": "a" * 40, "seed": 0, "workdir": "/removed/source/checkout"},
                  "optim": {"total_steps": 10}, "data": {"batch_size": 10, "sequence_length": 11}}
        metadata = {"source_commit": "a" * 40, "parameter_count": 199906304,
                    "status": "complete" if complete else "failed", "step": 10 if complete else 3}
        run = {"name": name, "attention_degree": degree, "qk_head_dim": 64, "value_head_dim": 64,
               "parameter_count": 199906304, "source_commit": "a" * 40, "seed": 0,
               "status": "complete" if complete else "numerical-failure",
               "quadratic_mode": "fixed", "quadratic_sharing": "global", "coefficient_count": 0,
               "resolved_config": store(f"results/{family}/configs/{name}.yaml", yaml.safe_dump(config)),
               "metadata": store(f"{directory}/metadata.json", json.dumps(metadata)),
               "metrics": None, "completion": None, "terminal": None, "selected_checkpoint": None}
        if complete:
            run.update(retained_completed=True, original_sweep_identity="historical",
                       metrics=store(f"{directory}/metrics.jsonl", "\n".join(json.dumps(r) for r in records)),
                       terminal={key: records[-1][key] for key in ("step", "training_tokens", "validation_nll", "validation_perplexity")},
                       selected_checkpoint={"path": f"checkpoints/polynomial_attention/{name}/5", "step": 5,
                                            "validation_nll": 2.0, "validation_perplexity": 7.39})
            completion = dict(final_step=10, training_tokens=1000, attention_degree=2,
                              parameter_count=199906304, source_commit="a" * 40)
            run["completion"] = store(f"{directory}/completion.json", json.dumps(completion))["path"]
        else:
            failure = {"step": 3, "attention_degree": 1, "error": "nonfinite update"}
            run["failure"] = dict(store(f"{directory}/numerical_failure_step_00000003.json", json.dumps(failure)), step=3)
        runs.append(run)
    manifest = {"format": "functional-memory-retained-experiment", "experiment": family,
                "source_commit": "a" * 40, "identity": "retained", "controls": runs[:1], "runs": runs[1:]}
    path = results / "provenance/manifest.json"
    store(str(path.relative_to(root)), json.dumps(manifest))
    (results / "README.md").write_text("# Authored question\n\nMethod stays here.\n\n<!-- BEGIN GENERATED RESULTS -->\nold\n<!-- END GENERATED RESULTS -->\n\nLimitations stay here.\n")
    return path, manifest


def test_relocated_retained_plot_preserves_failure_and_terminal_best_distinction(tmp_path, monkeypatch):
    original = tmp_path / "original"
    path, manifest = retained_fixture(original)
    copy = tmp_path / "copy"
    shutil.copytree(original, copy)
    shutil.rmtree(original)
    monkeypatch.chdir(tmp_path)
    relocated = copy / path.relative_to(original)
    rows = aggregate(manifest, manifest_path=relocated)
    assert rows[0]["terminal_ppl"] == 8.17 and rows[0]["best_ppl"] == 7.39
    assert rows[1]["terminal_ppl"] == "" and rows[1]["failure_step"] == 3
    report = relocated.parent.parent / "README.md"
    text = report.read_text()
    assert "Method stays here." in text and text.endswith("Limitations stay here.\n")
    assert (report.parent / "figures/terminal_metrics.pdf").exists()
    assert not (copy / "checkpoints").exists()


@pytest.mark.parametrize("corrupt", ["metrics", "failure", "terminal"])
def test_retained_study_rejects_changed_evidence_or_imputed_endpoint(tmp_path, corrupt):
    path, manifest = retained_fixture(tmp_path)
    if corrupt == "terminal":
        manifest["runs"][0]["terminal"] = {"step": 10}
    else:
        run = manifest["controls"][0] if corrupt == "metrics" else manifest["runs"][0]
        artifact = tmp_path / run[corrupt]["path"]
        artifact.write_text(artifact.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum|terminal endpoint"):
        retained_plot_manifest(manifest, path)
