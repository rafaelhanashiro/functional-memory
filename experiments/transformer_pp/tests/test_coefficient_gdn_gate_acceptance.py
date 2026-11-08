"""The GDN coefficient-gate run changes only the gate and starts at the control's parameters."""
import hashlib
import json

import pytest

from experiments.transformer_pp import coefficient_gdn_gate_acceptance as adapter
from scaffolds.transformer.acceptance import (
    COEFFICIENT_GATE_TESTS, COEFFICIENT_GDN_GATE_TESTS, test_paths as selected_test_paths,
)
from scaffolds.transformer.configs import config_to_dict
from scaffolds.transformer.models import model_parameter_count


def variant():
    config = adapter.control_config()
    config.model.attention_coefficient_gate = "gdn"
    return config


def test_control_recipe_is_pinned():
    assert hashlib.sha256(adapter.CONTROL.read_bytes()).hexdigest() == adapter.CONTROL_SHA256


def test_the_gated_run_preserves_the_control_budget_and_recipe():
    config = variant()
    adapter.check_recipe(config)
    assert config.optim.total_steps == 7570 and config.optim.warmup_steps == 455 and config.train.seed == 42
    assert config.data.batch_size == 256 and config.data.sequence_length == 2049
    assert config.model.attention_degree == 4 and config.model.qk_head_dim == 64
    assert config.model.positional_encoding == "rope" and config.model.attention_backend == "mosaic"
    assert config.model.attention_output_scaling == "head_rms_norm" and config.model.qk_norm_head_scale
    assert config.model.attention_coefficient_kernel_weight_decay and not config.model.attention_forget_gate
    assert config.model.attention_coefficient_amplitude_initial == 0


def test_the_pinned_control_builds_the_model_it_trained_and_the_gate_adds_its_parameters():
    recorded = json.loads((adapter.CONTROL.parent / "metadata.json").read_text())["parameter_count"]
    assert model_parameter_count(adapter.control_config().model) == recorded == 199396486
    model = variant().model
    gate = model.num_layers * model.num_heads * (model.hidden_size + 2) * (model.attention_degree + 1)
    assert model_parameter_count(model) == recorded + gate


@pytest.mark.parametrize("gate", ("none", "sigmoid"))
def test_other_gates_are_not_this_run(gate):
    config = variant()
    config.model.attention_coefficient_gate = gate
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


@pytest.mark.parametrize(("section", "field", "value"), (
    ("model", "attention_degree", 8),
    ("optim", "total_steps", 18925),
    ("optim", "learning_rate", .002),
    ("optim", "weight_decay", .0),
    ("data", "batch_size", 128),
    ("train", "seed", 43),
    ("model", "qk_norm_head_scale", False),
    ("model", "attention_output_scaling", "none"),
    ("model", "attention_coefficient_kernel_weight_decay", False),
    ("model", "attention_coefficient_projection_initial_std", .02),
    ("model", "initialization", "transformer_reference"),
))
def test_unrelated_scientific_changes_are_rejected(section, field, value):
    config = variant()
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


def test_execution_names_may_change():
    config = variant()
    config.train.workdir = "/new/run"
    config.train.checkpoint_dir = "/new/run/checkpoints"
    config.train.wandb_run_name = "query-m4-w-decay-order-gdn-gates"
    config.train.wandb_group = "transformer_pp"
    adapter.check_recipe(config)


def test_a_copied_manifest_must_hash_like_the_control(tmp_path, monkeypatch):
    copy = tmp_path / "manifest.json"
    copy.write_text("{}")
    config = variant()
    config.data.train_manifest = str(copy)
    with pytest.raises(ValueError, match="dataset manifest"):
        adapter.check_recipe(config)
    monkeypatch.setitem(adapter.MANIFEST_SHA256, "train_manifest", hashlib.sha256(b"{}").hexdigest())
    adapter.check_recipe(config)


def test_acceptance_adds_the_gdn_gate_and_adapter_checks():
    config = variant()
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    cpu = selected_test_paths(spec, config_to_dict(config), resolved_backend="mosaic", cpu=True)
    gpu = selected_test_paths(spec, config_to_dict(config), resolved_backend="mosaic", cpu=False)
    assert COEFFICIENT_GDN_GATE_TESTS in cpu and COEFFICIENT_GATE_TESTS not in cpu
    assert "experiments/transformer_pp/tests/test_coefficient_gdn_gate_acceptance.py" in cpu
    assert {path for path in gpu if path.startswith(f"{COEFFICIENT_GDN_GATE_TESTS}::")} == {
        f"{COEFFICIENT_GDN_GATE_TESTS}::{test}[4]" for test in (
            "test_gdn_gated_coefficients_through_mosaic_match_fp64",
            "test_gdn_gated_mosaic_model_matches_float32_blocked_model")}
    assert spec.check_initialization is adapter.check_initialization and spec.check_config is adapter.check_recipe


def test_initialization_check_finds_the_shared_start_and_epsilon_level_logit_parity(monkeypatch):
    import jax
    from flax import nnx
    from scaffolds.transformer.models import build_model
    from scaffolds.transformer.tests.test_coefficient_gdn_gates import _config, _ungated
    from scaffolds.transformer.training.checkpointing import state_sha256

    config = _config(4)  # a small backbone, not the control; pin its own ungated start instead
    config.train.seed = 42
    monkeypatch.setattr(adapter, "check_recipe", lambda config: None)
    monkeypatch.setattr(adapter, "LOGIT_TOKENS", 16)
    rng = jax.random.PRNGKey(42)
    ungated = build_model(_ungated(config).model, nnx.Rngs(params=rng, dropout=jax.random.fold_in(rng, 0)))
    monkeypatch.setattr(adapter, "CONTROL_INITIALIZATION_SHA256", state_sha256(nnx.state(ungated, nnx.Param)))
    parity = adapter.check_initialization(config)["ungated_parity"]
    assert parity["gate_parameters_repeat_one_draw_over_orders"] and parity["shared_parameters_match_ungated_recipe"]
    assert parity["gate_parameter_count"] == 2 * 2 * (16 + 2) * 5
    logits = parity["initial_logits"]
    assert logits["logit_difference_relative_to_largest"] <= logits["tolerance_relative_to_largest"]
    assert logits["epsilon_free_logit_difference_relative_to_largest"] <= logits["rounding_tolerance_relative_to_largest"]
    monkeypatch.setattr(adapter, "CONTROL_INITIALIZATION_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="rca97cc9db8f2's start"):
        adapter.check_initialization(config)
