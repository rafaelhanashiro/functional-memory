"""Checkpoint/requeue and provenance checks without a model download or GPU."""

import json
from pathlib import Path

import pytest
import torch
import yaml

from experiments.attention_scaling import sparse_heads as driver
from scaffolds.records import load_run, update_record


@pytest.fixture(autouse=True)
def clear_stop(monkeypatch):
    monkeypatch.setattr(driver, "STOP_REQUESTED", False)


def make_run(tmp_path, name="r000000000001"):
    run = tmp_path / name
    run.mkdir()
    (run / ".runtime").mkdir()
    config = {"fit": {"steps": 8, "evaluation_every": 4,
                       "checkpoint_selection": "minimum valid validation error"}}
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    update_record(run, {"status": "running"})
    return run, config


def model_optimizer_generator():
    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = torch.nn.Linear(3, 2)
    return [model], torch.optim.Adam(model.parameters(), lr=0.01), torch.Generator().manual_seed(42)


def update_model(models, optimizer, generator):
    x = torch.randn(7, 3, generator=generator)
    y = torch.randn(7, 2, generator=generator)
    optimizer.zero_grad()
    (models[0](x) - y).square().mean().backward()
    optimizer.step()


def valid_rows(error=0.5):
    return [{"output_relative_rms": error, "invalid_fraction": 0.0}]


def test_resume_restores_optimizer_rng_weights_and_discards_replayed_metrics(tmp_path):
    run, config = make_run(tmp_path)
    models, optimizer, generator = model_optimizer_generator()
    best = {"step": 0, "score": 0.25, "rows": valid_rows()}
    driver.save_checkpoint(run, 0, models, optimizer, generator, best, config, evaluated_step=0)
    for _ in range(2):
        update_model(models, optimizer, generator)
    driver.save_checkpoint(run, 2, models, optimizer, generator, best, config, evaluated_step=0)
    for _ in range(2):
        update_model(models, optimizer, generator)
    expected = driver.cpu_states(models)
    expected_rng = generator.get_state().clone()
    rows = [{"step": 0, "validation": valid_rows()}, *[{"step": n, "loss": n} for n in range(1, 5)],
            {"step": 4, "validation": valid_rows(0.1)}]
    (run / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows) + '{"step":')
    # A validation JSON written before checkpoint publication is superseded on resume.
    driver.write_json(run / "best_validation.json", {"step": 4})
    resumed_models, resumed_optimizer, resumed_generator = model_optimizer_generator()
    step, resumed_best, evaluated_step = driver.restore_checkpoint(
        run, resumed_models, resumed_optimizer, resumed_generator, config)
    assert (step, resumed_best, evaluated_step) == (2, best, 0)
    assert not driver.scheduled_validation(step, config["fit"])
    assert driver.scheduled_validation(4, config["fit"])
    assert json.loads((run / "best_validation.json").read_text())["step"] == 0
    assert [json.loads(line)["step"] for line in (run / "metrics.jsonl").read_text().splitlines()] == [0, 1, 2]
    for _ in range(2):
        update_model(resumed_models, resumed_optimizer, resumed_generator)
    for name, value in resumed_models[0].state_dict().items():
        torch.testing.assert_close(value, expected[0][name], rtol=0, atol=0)
    torch.testing.assert_close(resumed_generator.get_state(), expected_rng, rtol=0, atol=0)
    assert sorted(path.name for path in (run / "checkpoints").iterdir()) == ["0", "2"]


def test_interrupted_checkpoint_pointer_publication_recovers_same_step(tmp_path, monkeypatch):
    run, config = make_run(tmp_path)
    models, optimizer, generator = model_optimizer_generator()
    driver.save_checkpoint(run, 0, models, optimizer, generator, None, config, evaluated_step=-1)
    original_write = driver.write_json

    def interrupt_pointer(path, payload):
        if Path(path).name == "latest.json":
            raise OSError("simulated kill after atomic checkpoint replacement")
        original_write(path, payload)

    monkeypatch.setattr(driver, "write_json", interrupt_pointer)
    best = {"step": 0, "score": 0.25, "rows": valid_rows()}
    with pytest.raises(OSError, match="simulated kill"):
        driver.save_checkpoint(run, 0, models, optimizer, generator, best, config, evaluated_step=0)
    monkeypatch.setattr(driver, "write_json", original_write)
    step, restored_best, evaluated = driver.restore_checkpoint(run, models, optimizer, generator, config)
    assert (step, restored_best, evaluated) == (0, best, 0)
    assert not (run / ".runtime/latest_pending.json").exists()


def test_checkpoint_corruption_is_rejected_before_loading(tmp_path):
    run, config = make_run(tmp_path)
    models, optimizer, generator = model_optimizer_generator()
    driver.save_checkpoint(run, 0, models, optimizer, generator, None, config)
    path = run / "checkpoints/0/state.pt"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checkpoint checksum"):
        driver.restore_checkpoint(run, models, optimizer, generator, config)


def completed_run_fixture(tmp_path):
    run, config = make_run(tmp_path)
    models, optimizer, generator = model_optimizer_generator()
    best = {"step": 4, "score": 0.25, "rows": valid_rows()}
    driver.save_checkpoint(run, 4, models, optimizer, generator, best, config, evaluated_step=4)
    driver.save_checkpoint(run, 8, models, optimizer, generator, best, config, evaluated_step=8)
    driver.write_json(run / "best_validation.json", {"step": 4})
    (run / "metrics.jsonl").write_text('{"step":8}\n')
    dependency = tmp_path / "activation_manifest.json"
    dependency.write_text("{}\n")
    files = ["config.yaml", "metrics.jsonl", "best_validation.json", "checkpoints/4/state.pt"]
    completion = {"status": "complete", "step": 8, "best_step": 4,
                  "evidence": {"run_files": {name: driver.sha256_file(run / name) for name in files},
                               "dependencies": [{"path": str(dependency), "sha256": driver.sha256_file(dependency)}]}}
    driver.write_json(run / "completion.json", completion)
    return run, config


def test_completion_recovery_verifies_evidence_before_pruning_and_is_idempotent(tmp_path, monkeypatch):
    run, config = completed_run_fixture(tmp_path)
    original_remove = driver.shutil.rmtree

    def interrupt_cleanup(path):
        assert load_run(run)["status"] == "complete"
        assert (run / ".runtime/latest.json").exists()
        raise OSError("simulated interruption after outcome commit")

    monkeypatch.setattr(driver.shutil, "rmtree", interrupt_cleanup)
    with pytest.raises(OSError, match="outcome commit"):
        driver.recover_completion(run, config)
    assert load_run(run)["status"] == "complete"
    monkeypatch.setattr(driver.shutil, "rmtree", original_remove)
    assert driver.recover_completion(run, config)
    assert driver.recover_completion(run, config)
    assert sorted(path.name for path in (run / "checkpoints").iterdir()) == ["4"]
    assert not (run / ".runtime/latest.json").exists()
    checkpoint = torch.load(run / "checkpoints/4/state.pt", weights_only=False)
    assert "optimizer" in checkpoint and "generator" in checkpoint


def test_completion_recovery_refuses_changed_evidence_and_preserves_resume(tmp_path):
    run, config = completed_run_fixture(tmp_path)
    (run / "metrics.jsonl").write_text('{"step":7}\n')
    with pytest.raises(ValueError, match="Completion evidence changed"):
        driver.recover_completion(run, config)
    assert load_run(run)["status"] == "running"
    assert (run / "checkpoints/8/state.pt").exists()
    assert (run / ".runtime/latest.json").exists()


def test_warm_start_requires_complete_parent_and_verified_best(tmp_path):
    run, config = completed_run_fixture(tmp_path)
    with pytest.raises(ValueError, match="must be complete"):
        driver.verified_warm_start(run)
    driver.recover_completion(run, config)
    path, state = driver.verified_warm_start(run)
    assert state["step"] == 4
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="Warm-start checkpoint checksum"):
        driver.verified_warm_start(run)


def test_teacher_files_are_verified_against_recorded_model_identity(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"weights")
    teacher = {"model_id": "test/model", "revision": "fixed"}
    record = {"teacher_artifacts": {**teacher, "snapshot": str(tmp_path),
               "files": {path.name: {"sha256": driver.sha256_file(path), "size_bytes": path.stat().st_size}}}}
    driver.verify_teacher_files(record, {"teacher": teacher})
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Pretrained teacher file changed"):
        driver.verify_teacher_files(record, {"teacher": teacher})


def test_stop_interrupts_capture_and_evaluation_before_gpu_work(monkeypatch):
    monkeypatch.setattr(driver, "STOP_REQUESTED", True)
    config = {"data": {"tokens_sha256": "unused"}, "teacher": {}, "heads": [{}]}
    with pytest.raises(driver.PauseRequested):
        driver.load_activations(Path("unused"), config, Path("unused"))
    with pytest.raises(driver.PauseRequested):
        driver.evaluate([None], None, None, config)


def test_paused_driver_preserves_checkpoint_status_without_claiming_completion(tmp_path, monkeypatch):
    run, config = make_run(tmp_path)
    driver.write_json(run / ".runtime/latest.json", {"step": 3})

    def paused(*args, **kwargs):
        raise driver.PauseRequested()

    monkeypatch.setattr(driver, "_train", paused)
    driver.train(run, config, benchmark=True)
    assert load_run(run)["status"] == "preempted"
    assert load_run(run)["step"] == 3
    assert not (run / "completion.json").exists()


def test_invalid_validation_never_qualifies_and_preserves_scientific_failure(tmp_path):
    run, config = make_run(tmp_path)
    rows = [{"output_relative_rms": None, "invalid_fraction": 0.25}]
    assert driver.validation_score(rows) == float("inf")
    assert driver.validation_score(valid_rows(float("nan"))) == float("inf")
    driver.record_no_valid_checkpoint(run, config, rows)
    record = load_run(run)
    assert record["status"] == "numerical_failure"
    assert record["terminal"]["validation"] == rows
    assert (run / "numerical_failure_step_8.json").exists()
    assert not (run / "completion.json").exists()
