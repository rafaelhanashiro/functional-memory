"""Pointwise baselines preserve their exp control's recipe, initialization and decay."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from experiments.transformer_pp import pointwise_attention_acceptance as adapter
from scaffolds.transformer import acceptance
from scaffolds.transformer.configs import config_to_dict


def variant(transform="softplus"):
    config = adapter.control_config()
    for key, value in adapter.MODEL_CHANGES.items():
        setattr(config.model, key, value)
    config.model.attention_score_transform = transform
    return config


def test_control_pin_and_model_shape():
    record = json.loads((adapter.RUNS / adapter.CONTROL_RUN / "record.json").read_text())
    assert record["config"]["sha256"] == adapter.CONTROL_CONFIG_SHA256
    config = adapter.control_config()
    assert (config.model.num_layers, config.model.hidden_size, config.model.num_heads) == (15, 896, 14)
    assert (config.data.batch_size, config.data.sequence_length, config.optim.total_steps) == (256, 2049, 7570)
    assert config.train.seed == 42


@pytest.mark.parametrize("transform", adapter.TRANSFORMS)
def test_only_the_weight_function_and_backend_change(transform):
    config = variant(transform)
    adapter.check_recipe(config)
    assert config.model.attention_coefficients == "fixed"
    assert config.model.attention_coefficient_conditioning == "static"
    assert config.model.positional_encoding == "rope"
    assert not config.model.attention_forget_gate
    assert config.model.attention_output_scaling == "head_rms_norm"
    assert config.model.qk_norm_head_scale


@pytest.mark.parametrize("section,key,value", (
    ("model", "attention_fixed_coefficients", (1.0, 1.0)),
    ("model", "attention_score_transform", "identity"),
    ("model", "positional_encoding", "none"),
    ("model", "attention_output_scaling", "none"),
    ("model", "qk_norm_head_scale", False),
    ("model", "initialization", "transformer_reference"),
    ("optim", "learning_rate", 0.002),
    ("optim", "weight_decay", 0.0),
    ("optim", "total_steps", 7571),
    ("data", "batch_size", 128),
    ("train", "seed", 43),
))
def test_unrequested_recipe_changes_are_rejected(section, key, value):
    config = variant()
    setattr(getattr(config, section), key, value)
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


def test_execution_paths_and_labels_may_change():
    config = variant()
    config.train.workdir = "/new/run"
    config.train.checkpoint_dir = "/new/run/checkpoints"
    config.train.source_commit = "new-source"
    config.train.wandb_run_name = "pointwise-softplus"
    config.train.wandb_group = "transformer_pp"
    adapter.check_recipe(config)


@pytest.mark.parametrize("transform", adapter.TRANSFORMS)
def test_actual_initial_arrays_and_decay_match_on_small_backbone(transform):
    control = adapter.control_config()
    config = variant(transform)
    for current in (control, config):
        current.model = replace(current.model, vocab_size=64, num_layers=1, hidden_size=64,
                                num_heads=1, num_kv_heads=1, intermediate_size=128, max_seq_len=16)
    assert adapter.parameter_facts(config) == adapter.parameter_facts(control)


@pytest.mark.parametrize("field", ("parameter_count", "parameter_sha256", "weight_decay_mask_sha256"))
def test_initialization_gate_rejects_parameter_or_decay_changes(monkeypatch, field):
    facts = deepcopy(adapter.CONTROL_INITIALIZATION)
    facts[field] = "changed"
    monkeypatch.setattr(adapter, "parameter_facts", lambda config: facts)
    with pytest.raises(RuntimeError, match="Initial parameters or decay mask differ"):
        adapter.check_initialization(variant())


@pytest.mark.parametrize("transform", adapter.TRANSFORMS)
def test_acceptance_keeps_kernel_tests_pilot_resume_and_initialization(transform):
    config = variant(transform)
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    assert spec.check_config is adapter.check_recipe
    assert spec.check_initialization is adapter.check_initialization
    assert spec.required_backend == "mosaic"
    assert spec.pilot_steps == 20
    for cpu in (True, False):
        paths = acceptance.test_paths(spec, config_to_dict(config), resolved_backend="mosaic", cpu=cpu)
        assert {adapter.KERNEL_TESTS, adapter.INTEGRATION_TESTS, adapter.EMBEDDING_TESTS} <= set(paths)
        if cpu:
            assert adapter.OWN_TESTS in paths
    assert "resume" in acceptance.CHECKS


@pytest.mark.parametrize("gpu", adapter.GPU_TYPES)
def test_qualified_gpu_candidates_are_admitted(gpu):
    adapter.require_gpu_type(["--gpu-type", gpu])


def test_swish_is_a_pointwise_baseline():
    """The swish run (W = s sigmoid(s)) is the softplus baseline's recipe with its transform alone changed."""
    assert adapter.TRANSFORMS == ("sigmoid", "softplus", "swish")
    softplus, swish = (config_to_dict(variant(transform)) for transform in ("softplus", "swish"))
    assert {key for key in softplus["model"] if softplus["model"][key] != swish["model"][key]} == {
        "attention_score_transform"}
    for transform in ("abs", "identity"):
        with pytest.raises(ValueError):
            adapter.check_recipe(variant(transform))


def test_other_gpu_types_are_rejected_before_acceptance():
    with pytest.raises(ValueError, match="only on H100, B200"):
        adapter.require_gpu_type(["--gpu-type", "B300"])
    adapter.require_gpu_type(["--runtime-env"])
