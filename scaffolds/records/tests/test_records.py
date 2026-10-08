from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scaffolds.records import (
    capture_artifacts, create_run, load_run, load_study, regenerate_index, run_directory,
    update_record, validate_record,
)


def recipe():
    return {"model": {"attention_degree": 2, "qk_head_dim": 64, "attention_normalization": False}, "train": {"seed": 0}}


def test_identical_recipes_get_distinct_ids_and_resume_keeps_identity(tmp_path):
    study = tmp_path / "study"
    first = create_run(study, recipe(), "same-label")
    second = create_run(study, recipe(), "same-label")
    assert first != second
    assert len(first.name) == 13
    assert load_run(first)["label"] == "same-label"
    original = (first / "config.yaml").read_bytes()
    update_record(first, {"attempts": [{"job_id": "12"}], "status": "running"})
    record = update_record(first, {"attempts": [{"job_id": "12"}, {"job_id": "13"}], "status": "complete"})
    assert record["run_id"] == first.name
    assert (first / "config.yaml").read_bytes() == original
    assert run_directory(study, first.name) == first
    with pytest.raises(ValueError, match="found 2"):
        run_directory(study, "same-label")
    assert len(load_study(study / "sweep.yaml")["runs"]) == 2


def test_collision_never_overwrites_existing_or_orphaned_run(tmp_path, monkeypatch):
    import scaffolds.records as records
    ids = iter(["a" * 32, "a" * 32, "b" * 32, "c" * 32])
    monkeypatch.setattr(records.uuid, "uuid4", lambda: SimpleNamespace(hex=next(ids)))
    study = tmp_path / "study"
    first = create_run(study, recipe())
    sentinel = first / "sentinel"
    sentinel.write_text("preserve")
    orphan = study / "runs" / ("r" + "b" * 12)
    orphan.mkdir()
    second = create_run(study, recipe())
    assert second.name == "r" + "c" * 12
    assert sentinel.read_text() == "preserve"
    assert orphan.is_dir()


def test_concurrent_registration_does_not_lose_runs(tmp_path):
    study = tmp_path / "study"
    with ThreadPoolExecutor(max_workers=4) as pool:
        runs = list(pool.map(lambda seed: create_run(study, {"train": {"seed": seed}}), range(8)))
    assert len({run.name for run in runs}) == 8
    assert set(load_study(study)["run_ids"]) == {run.name for run in runs}


def test_index_reproduces_bytes_and_preserves_unknown_legacy_fields(tmp_path):
    study = tmp_path / "study"
    run = create_run(study, recipe(), "fixed-degree-two")
    (run / "metrics.jsonl").write_text('\n'.join(json.dumps(x) for x in [
        {"step": 20, "validation_nll": 2.0, "validation_perplexity": 7.0},
        {"step": 10, "validation_nll": 3.0, "validation_perplexity": 20.0},
    ]) + '\n')
    update_record(run, {"status": "complete", "source_commit": "source", "execution": {"attempts": [{"gpu": "H100"}]}})
    first = regenerate_index(study).read_bytes()
    assert regenerate_index(study).read_bytes() == first
    with (study / "runs.csv").open() as stream:
        row, = list(csv.DictReader(stream))
    assert row["attention_normalization"] == "False"
    assert row["qk_norm"] == ""
    assert row["coefficient_mode"] == ""
    assert row["coefficient_conditioning"] == ""
    assert row["attention_family"] == ""
    assert row["home_cluster"] == ""
    assert row["training_steps"] == ""
    assert row["prediction_tokens"] == ""
    assert row["terminal_step"] == "20"
    assert row["terminal_validation_nll"] == "2.0"
    assert row["metrics"] == f"runs/{run.name}/metrics.jsonl"
    assert load_study(study)["runs"][0]["execution"]["attempts"][0]["gpu"] == "H100"


def test_index_exposes_family_cluster_and_full_budget_separately_from_outcome(tmp_path):
    study = tmp_path / "study"
    config = {
        "model": {"attention_mechanism": "dot_product", "attention_degree": 2,
                  "attention_coefficients": "fixed", "attention_fixed_coefficients": [1.0, 2.0, 1.0]},
        "data": {"batch_size": 256, "sequence_length": 2049},
        "optim": {"total_steps": 19074}, "train": {"seed": 0},
    }
    run = create_run(study, config, "arbitrary-label")
    update_record(run, {"polynomial_family": "hermite", "home_cluster": "kennel", "status": "numerical_failure"})
    (run / "metrics.jsonl").write_text('{"step": 10, "validation_nll": 2.0}\n')
    regenerate_index(study)
    with (study / "runs.csv").open() as stream:
        row, = list(csv.DictReader(stream))
    assert row["attention_mechanism"] == "dot_product"
    assert row["attention_family"] == "hermite"
    assert row["coefficient_mode"] == "fixed"
    assert row["home_cluster"] == "kennel"
    assert row["training_steps"] == "19074"
    assert row["prediction_tokens"] == "10000269312"
    assert row["terminal_step"] == "10"


@pytest.mark.parametrize("quadratic_mode", ["positive", "unrestricted"])
def test_index_recognizes_legacy_learned_quadratic_recipe(tmp_path, quadratic_mode):
    study = tmp_path / "study"
    create_run(study, {"model": {"attention_degree": 2, "quadratic_mode": quadratic_mode,
                                 "quadratic_sharing": "per_head"}})
    with (study / "runs.csv").open() as stream:
        row, = list(csv.DictReader(stream))
    assert row["attention_family"] == "learned_polynomial"
    assert row["coefficient_mode"] == "learned"
    assert row["quadratic_mode"] == quadratic_mode
    assert row["coefficient_sharing"] == "per_head"


def test_index_distinguishes_query_conditioning_and_keeps_recipe_identity(tmp_path):
    study = tmp_path / "study"
    runs = {}
    for conditioning in ("static", "query"):
        config = {"model": {"attention_degree": 1, "attention_coefficients": "learned",
                            "attention_coefficient_conditioning": conditioning,
                            "attention_coefficient_sharing": "per_head",
                            "attention_normalization": False,
                            "attention_output_scaling": "head_rms_norm"},
                  "optim": {"warmup_steps": 400}}
        runs[conditioning] = create_run(study, config)
    with (study / "runs.csv").open() as stream:
        rows = {row["run_id"]: row for row in csv.DictReader(stream)}
    for conditioning, run in runs.items():
        assert rows[run.name]["coefficient_conditioning"] == conditioning
        assert rows[run.name]["attention_family"] == "learned_polynomial"
        assert rows[run.name]["attention_output_scaling"] == "head_rms_norm"
    assert load_run(runs["query"])["config"]["sha256"] != load_run(runs["static"])["config"]["sha256"]
    first = (study / "runs.csv").read_bytes()
    assert regenerate_index(study).read_bytes() == first


def test_index_keeps_custom_polynomial_distinct_without_inventing_basis(tmp_path):
    study = tmp_path / "study"
    create_run(study, {"model": {"attention_degree": 2, "attention_coefficients": "fixed",
                                 "attention_fixed_coefficients": [1.0, 2.0, 1.0]}}, "hermite-looking-label")
    with (study / "runs.csv").open() as stream:
        row, = list(csv.DictReader(stream))
    assert row["attention_family"] == "polynomial"


def test_record_updates_preserve_execution_evidence_and_reject_identity_changes(tmp_path):
    run = create_run(tmp_path / "study", recipe(), "label")
    update_record(run, {"execution": {"hardware": "H100", "acceptance": {"passed": 10}}})
    record = update_record(run, {"execution": {"attempts": ["first", "resume"]}})
    assert record["execution"] == {"hardware": "H100", "acceptance": {"passed": 10}, "attempts": ["first", "resume"]}
    for field in ("run_id", "label", "name", "config"):
        with pytest.raises(ValueError, match="immutable"):
            update_record(run, {field: "changed"})


def test_integrity_detects_changed_config_and_scientific_artifact(tmp_path):
    run = create_run(tmp_path / "study", recipe())
    metrics = run / "metrics.jsonl"
    metrics.write_text('{"step": 1}\n')
    update_record(run, {"metrics": {"path": "metrics.jsonl", "sha256": hashlib.sha256(metrics.read_bytes()).hexdigest()}, "log": {"path": "training.log", "sha256": "a" * 64}})
    result = validate_record(run)
    assert result["unavailable"] == ["training.log"]
    metrics.write_text('{"step": 2}\n')
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_record(run)
    metrics.write_text('{"step": 1}\n')
    (run / "config.yaml").write_text("model: {}\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_record(run)


def test_study_rejects_duplicate_or_unregistered_path_ids(tmp_path):
    study = tmp_path / "study"
    run = create_run(study, recipe())
    path = study / "sweep.yaml"
    sweep = yaml.safe_load(path.read_text())
    sweep["run_ids"].append(run.name)
    path.write_text(yaml.safe_dump(sweep))
    with pytest.raises(ValueError, match="Duplicate"):
        load_study(study)
    sweep["run_ids"] = ["../elsewhere"]
    path.write_text(yaml.safe_dump(sweep))
    with pytest.raises(ValueError, match="Invalid"):
        load_study(study)


def test_standalone_trainer_record_creation(tmp_path):
    run = tmp_path / "smoke"
    run.mkdir()
    (run / "config.yaml").write_text(yaml.safe_dump(recipe()))
    record = update_record(run, {"status": "running", "execution": {"runtime": {"host_staging": False}}})
    assert record["run_id"] == "smoke"
    assert record["config"]["sha256"] == hashlib.sha256((run / "config.yaml").read_bytes()).hexdigest()
    assert load_run(run)["execution"]["runtime"]["host_staging"] is False


def test_non_transformer_study_index_uses_flat_settings(tmp_path):
    study = tmp_path / "state-study"
    run = create_run(study, {"model": "organization/model", "condition": "random", "seed": 4})
    update_record(run, {"status": "complete"})
    regenerate_index(study)
    with (study / "runs.csv").open() as stream:
        row, = list(csv.DictReader(stream))
    assert row["model_id"] == "organization/model"
    assert row["condition"] == "random"
    assert row["seed"] == "4"
    assert row["attention_degree"] == ""


def test_state_recipe_records_allocated_output_directory(tmp_path):
    run = create_run(tmp_path / "state-study", {
        "model": "organization/model", "condition": {"name": "random", "seed": 2},
        "output_directory": "placeholder",
    })
    recipe = yaml.safe_load((run / "config.yaml").read_text())
    assert Path(recipe["output_directory"]) == run
    with (run.parent.parent / "runs.csv").open() as stream:
        row, = list(csv.DictReader(stream))
    assert row["condition"] == "random"
    assert row["seed"] == "2"


def test_stray_git_directory_is_not_a_checkout(tmp_path):
    (tmp_path / ".git").mkdir()
    run = create_run(tmp_path / "study", recipe())
    assert load_run(run)["config"]["path"] == str((run / "config.yaml").resolve())


@pytest.mark.parametrize("worktree", [False, True])
def test_paths_are_stored_relative_to_the_checkout(tmp_path, worktree):
    checkout = tmp_path / "checkout"
    if worktree:
        checkout.mkdir()
        (checkout / ".git").write_text("gitdir: /elsewhere/.git/worktrees/checkout\n")
    else:
        (checkout / ".git").mkdir(parents=True)
        (checkout / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    run = create_run(checkout / "experiments" / "study", recipe())
    assert load_run(run)["config"]["path"] == f"experiments/study/runs/{run.name}/config.yaml"
    validate_record(run)


def test_capture_uses_explicit_best_checkpoint_and_detects_shard_changes(tmp_path):
    run = create_run(tmp_path / "study", recipe())
    (run / "metrics.jsonl").write_text('{"step": 10, "validation_nll": 2.0}\n')
    (run / "completion.json").write_text('{"final_step": 20}\n')
    (run / "training.log").write_text("finished\n")
    best = run / "checkpoints" / "10"
    latest = run / "checkpoints" / "20"
    best.mkdir(parents=True)
    latest.mkdir()
    (best / "shard").write_bytes(b"validation best")
    (latest / "shard").write_bytes(b"last update")
    updates = capture_artifacts(run, best, include_log=True)
    assert Path(updates["selected_checkpoint"]["path"]) == best
    assert updates["selected_checkpoint"]["file_count"] == 1
    assert all(not item["path"].endswith("record.json") for item in updates["artifacts"])
    update_record(run, updates)
    validate_record(run)
    (best / "shard").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="tree checksum mismatch"):
        validate_record(run)


def test_checkpoint_size_check_reads_no_contents_and_detects_changed_files(tmp_path):
    run = create_run(tmp_path / "study", recipe())
    best = run / "checkpoints" / "10"
    best.mkdir(parents=True)
    (best / "shard").write_bytes(b"validation best")
    update_record(run, capture_artifacts(run, best))
    stored = load_run(run)["selected_checkpoint"]["path"]
    result = validate_record(run, checkpoint_hashes=False)
    assert result["size_checked"] == [stored] and stored not in result["verified"]
    # Same-size edits need contents, which only the full hash reads.
    (best / "shard").write_bytes(b"validation BEST")
    validate_record(run, checkpoint_hashes=False)
    with pytest.raises(ValueError, match="tree checksum mismatch"):
        validate_record(run)
    (best / "shard").write_bytes(b"validation best")
    (best / "extra").write_bytes(b"")
    with pytest.raises(ValueError, match="names or sizes"):
        validate_record(run, checkpoint_hashes=False)
    (best / "extra").unlink()
    (best / "shard").write_bytes(b"truncated")
    with pytest.raises(ValueError, match="names or sizes"):
        validate_record(run, checkpoint_hashes=False)


@pytest.mark.parametrize("checkpoint_hashes", [False, True])
def test_cli_validates_named_runs_and_hashes_checkpoints_only_on_request(
        tmp_path, monkeypatch, capsys, checkpoint_hashes):
    from scaffolds.records.__main__ import main

    study = tmp_path / "study"
    first, second = create_run(study, recipe(), "first"), create_run(study, recipe(), "second")
    best = first / "checkpoints" / "10"
    best.mkdir(parents=True)
    (best / "shard").write_bytes(b"validation best")
    update_record(first, capture_artifacts(first, best))
    arguments = ["records", "validate", str(study), "first"]
    monkeypatch.setattr("sys.argv", arguments + (["--checkpoint-hashes"] if checkpoint_hashes else []))
    main()
    [result] = json.loads(capsys.readouterr().out)["runs"]
    stored = load_run(first)["selected_checkpoint"]["path"]
    assert result["run_id"] == load_run(first)["run_id"]
    assert (stored in result["verified"]) is checkpoint_hashes
    assert (stored in result["size_checked"]) is not checkpoint_hashes
    monkeypatch.setattr("sys.argv", ["records", "validate", str(study)])
    main()
    assert [run["run_id"] for run in json.loads(capsys.readouterr().out)["runs"]] == [
        load_run(first)["run_id"], load_run(second)["run_id"]]


def test_capture_never_silently_ignores_unavailable_selected_checkpoint(tmp_path):
    run = create_run(tmp_path / "study", recipe())
    with pytest.raises(ValueError, match="unavailable"):
        capture_artifacts(run, run / "checkpoints" / "missing")


def test_cli_allocates_fully_resolved_fresh_transformer_recipe(tmp_path, monkeypatch, capsys):
    from scaffolds.records.__main__ import main
    source = tmp_path / "template.yaml"
    source.write_text("model: {}\ntrain:\n  source_commit: historical-source\n")
    study = tmp_path / "study"
    monkeypatch.setattr("sys.argv", ["records", "create", "--study", str(study), "--config", str(source), "--label", "fresh"])
    main()
    run = Path(capsys.readouterr().out.strip())
    saved = yaml.safe_load((run / "config.yaml").read_text())
    assert saved["train"]["source_commit"] == "auto"
    assert saved["model"]["tie_embeddings"] is False
    assert saved["model"]["qk_norm"] is True
    assert saved["optim"]["weight_decay_policy"] == "exclude_embeddings_norms_biases_v1"
    assert load_run(run)["label"] == "fresh"
    validate_record(run)


@pytest.mark.parametrize("label", [None, "new-run"])
def test_cli_fresh_run_resets_copied_tracking_identity(tmp_path, monkeypatch, capsys, label):
    from scaffolds.records.__main__ import main

    source = tmp_path / "historical.yaml"
    source.write_text(yaml.safe_dump({"model": {}, "train": {
        "source_commit": "historical-source", "wandb_group": "old-study",
        "wandb_run_name": "old-run", "wandb_dir": "/old/worktree/run/wandb",
    }}))
    original = source.read_bytes()
    study = tmp_path / "attention_scaling"
    arguments = ["records", "create", "--study", str(study), "--config", str(source)]
    if label is not None:
        arguments.extend(["--label", label])
    monkeypatch.setattr("sys.argv", arguments)
    main()
    run = Path(capsys.readouterr().out.strip())
    saved = yaml.safe_load((run / "config.yaml").read_text())
    assert saved["train"]["source_commit"] == "auto"
    assert saved["train"]["wandb_group"] == "attention_scaling"
    assert saved["train"]["wandb_run_name"] == label
    assert saved["train"]["wandb_dir"] is None
    assert source.read_bytes() == original


@pytest.mark.parametrize("train_section", ["", "train: {}\n"])
def test_cli_rejects_copied_plain_qk_recipe_before_allocating(tmp_path, monkeypatch, capsys, train_section):
    from scaffolds.records.__main__ import main

    source = tmp_path / "legacy.yaml"
    source.write_text("model:\n  qk_norm: false\n" + train_section)
    original = source.read_bytes()
    study = tmp_path / "study"
    monkeypatch.setattr("sys.argv", ["records", "create", "--study", str(study), "--config", str(source)])
    with pytest.raises(SystemExit) as failure:
        main()
    assert failure.value.code == 2
    assert "--allow-plain-qk" in capsys.readouterr().err
    assert not study.exists()
    assert source.read_bytes() == original


def test_cli_resolves_model_only_recipe_to_canonical_qk_norm(tmp_path, monkeypatch, capsys):
    from scaffolds.records.__main__ import main

    source = tmp_path / "recipe.yaml"
    source.write_text("model:\n  attention_degree: 2\n")
    study = tmp_path / "study"
    monkeypatch.setattr("sys.argv", ["records", "create", "--study", str(study), "--config", str(source)])
    main()
    run = Path(capsys.readouterr().out.strip())
    saved = yaml.safe_load((run / "config.yaml").read_text())
    assert saved["model"]["qk_norm"] is True
    assert saved["model"]["initialization"] == "transformer_reference"
    assert saved["train"]["source_commit"] == "auto"
    validate_record(run)


@pytest.mark.parametrize("mechanism, scheme", [("gla", "gla_reference"), ("gated_deltanet", "gated_deltanet_reference")])
def test_cli_resolves_each_mechanism_to_its_reference_initialization(tmp_path, monkeypatch, capsys, mechanism, scheme):
    from scaffolds.records.__main__ import main

    source = tmp_path / "recipe.yaml"
    source.write_text(f"model:\n  attention_mechanism: {mechanism}\n")
    monkeypatch.setattr("sys.argv", ["records", "create", "--study", str(tmp_path / "study"), "--config", str(source)])
    main()
    run = Path(capsys.readouterr().out.strip())
    assert yaml.safe_load((run / "config.yaml").read_text())["model"]["initialization"] == scheme


def test_cli_requires_explicit_opt_out_for_flax_default_initialization(tmp_path, monkeypatch, capsys):
    from scaffolds.records.__main__ import main

    source = tmp_path / "flax.yaml"
    source.write_text("model:\n  initialization: flax_default\ntrain: {}\n")
    study = tmp_path / "study"
    monkeypatch.setattr("sys.argv", ["records", "create", "--study", str(study), "--config", str(source)])
    with pytest.raises(SystemExit) as failure:
        main()
    assert failure.value.code == 2
    assert "--allow-flax-default-init" in capsys.readouterr().err
    assert not study.exists()
    monkeypatch.setattr("sys.argv", ["records", "create", "--study", str(study), "--config", str(source),
                                    "--allow-flax-default-init"])
    main()
    run = Path(capsys.readouterr().out.strip())
    assert yaml.safe_load((run / "config.yaml").read_text())["model"]["initialization"] == "flax_default"
    validate_record(run)


def test_cli_allows_explicit_plain_qk_opt_out(tmp_path, monkeypatch, capsys):
    from scaffolds.records.__main__ import main

    source = tmp_path / "plain.yaml"
    source.write_text("model:\n  qk_norm: false\ntrain: {}\n")
    study = tmp_path / "study"
    monkeypatch.setattr("sys.argv", ["records", "create", "--study", str(study),
                                    "--config", str(source), "--allow-plain-qk"])
    main()
    run = Path(capsys.readouterr().out.strip())
    assert yaml.safe_load((run / "config.yaml").read_text())["model"]["qk_norm"] is False
    validate_record(run)


@pytest.mark.parametrize("model", [{"attention_mechanism": "gla", "qk_norm": False},
                                  {"attention_degree": 0, "qk_norm": False}])
def test_cli_qk_requirement_exempts_mechanisms_without_qk_normalization(tmp_path, monkeypatch, capsys, model):
    from scaffolds.records.__main__ import main

    source = tmp_path / "recipe.yaml"
    source.write_text(yaml.safe_dump({"model": model, "train": {}}))
    study = tmp_path / "study"
    monkeypatch.setattr("sys.argv", ["records", "create", "--study", str(study), "--config", str(source)])
    main()
    run = Path(capsys.readouterr().out.strip())
    assert yaml.safe_load((run / "config.yaml").read_text())["model"]["qk_norm"] is False
    validate_record(run)
