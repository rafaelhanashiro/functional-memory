"""Acceptance qualifies a continuation's restored state and remaining updates."""
import json
from types import SimpleNamespace

import pytest

from scaffolds.transformer import acceptance
from scaffolds.transformer.configs import Config, config_to_dict, save_config


def continuation_config(tmp_path):
    config = Config()
    config.continuation.parent_config = str(tmp_path / "parent.yaml")
    config.continuation.parent_config_sha256 = "a" * 64
    config.continuation.parent_checkpoint = str(tmp_path / "parent/7500")
    config.continuation.parent_checkpoint_sha256 = "b" * 64
    config.continuation.parent_step = 7500
    config.optim.total_steps = 8000
    config.optim.learning_rate = config.optim.min_learning_rate = 1e-4
    config.optim.warmup_steps = 0
    config.train.eval_every = config.train.ckpt_every = 100
    config.train.eval_batches = 19
    return config


def test_remaining_projection_excludes_parent_training(tmp_path):
    config = continuation_config(tmp_path)
    assert acceptance.continuation_start_step(config) == 7500
    assert acceptance.continuation_start_step(config_to_dict(config)) == 7500
    assert acceptance.remaining_work_counts(config) == (6, 6)
    config.continuation.parent_checkpoint = ""
    assert acceptance.continuation_start_step(config) == 0
    assert acceptance.remaining_work_counts(config) == (80, 80)


def test_scratch_pilot_targets_absolute_stop_without_changing_recipe(tmp_path, monkeypatch):
    from scaffolds.transformer.training import loop
    from scaffolds.transformer.training.cadence import evaluation_due

    config = continuation_config(tmp_path)
    path = tmp_path / "config.yaml"
    save_config(config, path)
    original = path.read_bytes()
    captured = {}

    def train(scratch, **options):
        captured.update(config=scratch, options=options)
        return {"final_step": options["stop_after_step"]}

    monkeypatch.setattr(loop, "run_training", train)
    args = SimpleNamespace(worker="pilot", config=path, directory=tmp_path / "acceptance", stop_after_step=7520)
    spec = acceptance.AcceptanceSpec(entry_module="unused", cpu_tests=(), pilot_steps=20)
    acceptance._worker(spec, args, {"source_commit": "c" * 40})
    scratch = captured["config"]
    assert evaluation_due(scratch, 7520)
    assert scratch.train.eval_every == 7520
    assert scratch.train.eval_batches == 1
    assert scratch.optim.total_steps == 8000
    assert scratch.continuation.parent_step == 7500
    assert captured["options"] == {"resume": True, "stop_after_step": 7520, "audit_checkpoint": True}
    assert path.read_bytes() == original


def test_acceptance_offsets_pilot_resume_and_runtime_projection(tmp_path, monkeypatch):
    config = continuation_config(tmp_path)
    args = SimpleNamespace(config=tmp_path / "recipe.yaml", directory=tmp_path / "acceptance",
                           gpu_count=1, gpu_type="B300")
    experiment = {"identity": "test", "source_commit": "c" * 40, "config": config_to_dict(config),
                  "allocation_seconds": 24 * 3600}
    spec = acceptance.AcceptanceSpec(entry_module="unused", cpu_tests=(), pilot_steps=20)
    calls = []
    audited = []

    def worker(spec, args, name, output, *, stop, **unused):
        if name == "hardware":
            return {"attention_backend": "mosaic", "device_kinds": ["NVIDIA B300"]}
        assert name == "pilot"
        calls.append(stop)
        pilot = args.directory / "pilot"
        pilot.mkdir(exist_ok=True)
        (pilot / "metrics.jsonl").write_text(json.dumps({
            "step": 7520, "validation_nll": 2.6, "attention_diagnostics": {"valid": True},
        }) + "\n")
        return {"start_step": 7500 if stop == 7520 else 7520, "final_step": stop,
                "restored_step": None if stop == 7520 else 7520,
                "data_parallel_devices": 1, "attention_backend": "mosaic",
                "data_start_sequence": 7520 * config.data.batch_size,
                "timing": {"train_step_seconds": [30.0, 2.0], "evaluation_seconds": [3.0],
                           "checkpoint_seconds": [4.0], "runtime_seconds": 50.0}}

    monkeypatch.setattr(acceptance, "_run_worker", worker)
    monkeypatch.setattr(acceptance, "load_config", lambda _: config)
    monkeypatch.setattr(acceptance, "passed", lambda path, _: path.stem in {
        "cpu-tests", "cpu-distributed-tests", "gpu-tests", "initialization"})
    monkeypatch.setattr(acceptance, "validate_resume_audit", lambda before, after: audited.append((before, after)))
    # Final provenance reads test-stage records even when their checks were cached.
    args.directory.mkdir()
    for stage in ("cpu-tests", "cpu-distributed-tests", "gpu-tests"):
        (args.directory / f"{stage}.json").write_text("{}\n")
    acceptance._run(spec, args, experiment)
    assert calls == [7520, 7521]
    assert len(audited) == 1
    result = json.loads((args.directory / "pilot.json").read_text())
    expected_seconds = 50 + 2 * 500 + 3 * 19 * 6 + 4 * 6
    assert result["projected_hours"] == pytest.approx(expected_seconds / 3600)
    assert (args.directory / "success.json").exists()


@pytest.mark.parametrize("broken", (None, "parameters", "optimizer", "coefficient_clock"))
def test_initialization_checks_restored_backbone_and_new_optimizer(tmp_path, monkeypatch, broken):
    import jax.numpy as jnp
    from scaffolds.transformer import models
    from scaffolds.transformer.training import continuation

    config = continuation_config(tmp_path)
    config.model.attention_coefficients = "learned"
    audit = {"parent_parameters_sha256": "p", "backbone_parameters_sha256": "p",
             "parent_optimizer_sha256": "o", "backbone_optimizer_sha256": "o",
             "new_coefficient_optimizer_count": 0, "parent_step": 7500}
    if broken in ("parameters", "optimizer"):
        audit[f"backbone_{broken}_sha256"] = "wrong"
    elif broken == "coefficient_clock":
        audit["new_coefficient_optimizer_count"] = 7500
    state = SimpleNamespace(params={"weight": jnp.array([2., 3.])}, step=7500)
    calls = []

    def initialize(current):
        assert current.model.attention_backend == "blocked"
        calls.append(current)
        return state, audit

    monkeypatch.setattr(continuation, "initialize_continuation_state", initialize)
    monkeypatch.setattr(models, "model_parameter_count", lambda _: 2)
    monkeypatch.setattr(models, "build_model", lambda *a, **kw: pytest.fail("random initialization used"))
    if broken is None:
        result = acceptance.initialization_facts(config)
        assert result["parameter_count"] == 2
        assert result["continuation"] == audit
        assert len(result["parameter_sha256"]) == 64
    else:
        with pytest.raises(RuntimeError, match="backbone|optimizer clock"):
            acceptance.initialization_facts(config)
    assert len(calls) == 1


def test_continuations_require_restore_tests_and_skip_random_backbone_comparison(tmp_path):
    config = continuation_config(tmp_path)
    spec = acceptance.attention_spec(config, entry_module="unused")
    assert spec.check_initialization is None
    paths = acceptance.test_paths(spec, config_to_dict(config), resolved_backend="blocked", cpu=True)
    assert "scaffolds/transformer/tests/test_continuation.py" in paths
    assert "scaffolds/transformer/tests/test_continuation_acceptance.py" in paths
