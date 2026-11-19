"""Frozen recipes, initialization comparisons and qualification for zero constants."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.transformer_pp import softplus_zero_constant_acceptance as adapter
from scaffolds.transformer import acceptance
from scaffolds.transformer.configs import config_to_dict


KEYS = tuple(adapter.CONTROLS)


def variant(key, transform="softplus"):
    config = adapter.control_config(key)
    config.model.attention_coefficient_zero_constant = True
    config.model.attention_score_transform = transform
    for name, value in adapter.CAPTURE_POLICY.items():
        setattr(config.train, name, value)
    return config


@pytest.mark.parametrize("key", KEYS)
def test_controls_are_exact_complete_parents_and_only_scientific_changes_are_allowed(key):
    descriptor = adapter.CONTROLS[key]
    path = adapter.RUNS / descriptor["run_id"]
    assert hashlib.sha256((path / "config.yaml").read_bytes()).hexdigest() == descriptor["config_sha256"]
    record = json.loads((path / "record.json").read_text())
    assert record["config"]["sha256"] == descriptor["config_sha256"]
    assert record["status"] == "complete"
    parent, config = adapter.control_config(key), variant(key)
    adapter.check_recipe(config)
    first, second = config_to_dict(parent), config_to_dict(config)
    changes = {f"{section}.{name}" for section in first for name in first[section]
               if first[section][name] != second[section][name]}
    expected = {"model.attention_coefficient_zero_constant"}
    if key[0] == "static":
        expected |= {"model.attention_score_transform", "train.gradient_event_capture"}
    assert changes == expected
    assert (config.model.num_layers, config.model.hidden_size, config.model.num_heads) == (15, 896, 14)
    assert (config.model.attention_degree, config.model.attention_coefficient_input_power) == (8, 0)
    assert (config.model.attention_backend, config.model.positional_encoding) == ("mosaic", "rope")
    assert config.model.attention_coefficient_input_normalization == "none"
    assert config.model.attention_coefficient_kernel_weight_decay == (key[0] == "query")
    assert (config.train.seed, config.optim.total_steps, config.data.batch_size) == (42, 7570, 256)
    assert (config.optim.weight_decay, config.optim.learning_rate) == (.1, .001)


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("section,name,value", (
    ("model", "attention_coefficient_zero_constant", False),
    ("model", "attention_degree", 12),
    ("model", "attention_coefficient_input_power", 1),
    ("model", "attention_score_transform", "sigmoid"),
    ("model", "attention_backend", "blocked"),
    ("model", "positional_encoding", "none"),
    ("model", "qk_norm_head_scale", False),
    ("optim", "weight_decay", 0.),
    ("optim", "learning_rate", .002),
    ("optim", "total_steps", 7571),
    ("train", "seed", 43),
    ("train", "gradient_event_capture", False),
    ("data", "batch_size", 128),
))
def test_unrelated_changes_and_missing_restrictions_are_refused(key, section, name, value):
    config = variant(key)
    setattr(getattr(config, section), name, value)
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


@pytest.mark.parametrize("key", KEYS)
def test_wd_policy_cannot_silently_change(key):
    config = variant(key)
    config.model.attention_coefficient_kernel_weight_decay = not config.model.attention_coefficient_kernel_weight_decay
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


@pytest.mark.parametrize("key", KEYS)
def test_execution_paths_and_monitoring_labels_may_change(key):
    config = variant(key)
    config.train.workdir = Path("/new/workdir")
    config.train.checkpoint_dir = Path("/new/checkpoints")
    config.train.source_commit = "another-source"
    config.train.wandb_run_name = "another-run"
    config.train.wandb_group = "another-group"
    config.train.wandb_dir = Path("/new/wandb")
    config.train.wandb_mode = "disabled"
    adapter.check_recipe(config)


@pytest.mark.parametrize("key", KEYS)
def test_changed_control_bytes_are_refused(key, tmp_path, monkeypatch):
    target = tmp_path / adapter.CONTROLS[key]["run_id"]
    target.mkdir()
    (target / "config.yaml").write_text("changed: true\n")
    monkeypatch.setattr(adapter, "RUNS", tmp_path)
    with pytest.raises(ValueError, match="Pinned.*changed"):
        adapter.control_config(key)


def test_unknown_parent_is_refused():
    with pytest.raises(ValueError, match="No zero-constant parent"):
        adapter.control_config(("static", "sum_of_squares"))


@pytest.mark.parametrize("key", KEYS)
def test_qualification_requires_new_feature_and_native_mosaic(key):
    config = variant(key)
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    cpu = acceptance.test_paths(spec, config_to_dict(config), resolved_backend="mosaic", cpu=True)
    gpu = acceptance.test_paths(spec, config_to_dict(config), resolved_backend="mosaic", cpu=False)
    assert set(adapter.CPU_TESTS) <= set(cpu)
    assert set(adapter.GPU_TESTS) <= set(gpu)
    assert "scaffolds/transformer/tests/test_mosaic_score_transform.py" in gpu
    assert "scaffolds/transformer/tests/test_mosaic_query.py" in gpu
    assert spec.required_backend == "mosaic" and spec.pilot_steps == 20
    assert spec.check_config is adapter.check_recipe
    assert spec.check_initialization is adapter.check_initialization


@pytest.mark.parametrize("gpu_type", ("H100", "B200"))
def test_supported_allocations_enter_full_acceptance(gpu_type, monkeypatch):
    calls = []
    monkeypatch.setattr(adapter, "run_acceptance", lambda spec, argv: calls.append((spec, argv)))
    adapter.main(["unused-config.yaml", "--gpu-type", gpu_type])
    assert len(calls) == 1


@pytest.mark.parametrize("gpu_type", ("B300", "A100", "A6000"))
def test_unqualified_allocations_are_refused(gpu_type, monkeypatch):
    monkeypatch.setattr(adapter, "run_acceptance", lambda *_: pytest.fail("must reject before acceptance"))
    with pytest.raises(ValueError, match="H100 or B200 only"):
        adapter.main(["unused-config.yaml", "--gpu-type", gpu_type])


@pytest.mark.parametrize("key", KEYS)
def test_surviving_initialization_matches_parent_coordinate_arrays(key):
    parent = adapter.control_config(key)
    parent.model = replace(parent.model, hidden_size=32, num_layers=2, num_heads=2, num_kv_heads=2,
                           qk_head_dim=16, value_head_dim=16, intermediate_size=64,
                           vocab_size=64, max_seq_len=16, attention_backend="blocked")
    config = deepcopy(parent)
    config.model.attention_coefficient_zero_constant = True
    config.model.attention_score_transform = "softplus"
    facts = adapter.check_surviving_initialization(config, parent)
    count = 2 if key[1] == "sum_of_squares" else 1
    removed = 2 * 2 * count * (32 + 2 if key[0] == "query" else 1)
    assert facts["removed_parameter_count"] == removed
    assert facts["all_surviving_mechanism_parameters_match_parent"]
    if key[0] == "query":
        config.train.seed += 1
        with pytest.raises(RuntimeError, match="Surviving initial parameter query_polynomial_kernel"):
            adapter.check_surviving_initialization(config, parent)


@pytest.mark.parametrize("key", KEYS)
def test_actual_model_backbone_and_surviving_arrays_match_parent(key):
    """Check built model arrays, including independent backbone draws, on a small shape."""
    import jax
    from flax import nnx
    from scaffolds.transformer.models import build_model, model_parameter_count

    parent = adapter.control_config(key)
    parent.model = replace(parent.model, hidden_size=32, num_layers=2, num_heads=2, num_kv_heads=2,
                           qk_head_dim=16, value_head_dim=16, intermediate_size=64,
                           vocab_size=64, max_seq_len=16, attention_backend="blocked")
    config = deepcopy(parent)
    config.model.attention_coefficient_zero_constant = True
    config.model.attention_score_transform = "softplus"
    def arrays(current):
        rng = jax.random.PRNGKey(current.train.seed)
        model = build_model(current.model, nnx.Rngs(params=rng, dropout=jax.random.fold_in(rng, 0)))
        return {path: np.asarray(value[...]) for path, value in nnx.to_flat_state(nnx.state(model, nnx.Param))}
    original, actual = arrays(parent), arrays(config)
    assert set(original) == set(actual)
    indices = (1, 2, 3, 4, 6, 7, 8) if key[1] == "sum_of_squares" else tuple(range(1, 9))
    for path, value in original.items():
        expected = np.take(value, indices, axis=-1) if path[0] in adapter.COMPACT_PARAMETERS else value
        np.testing.assert_array_equal(actual[path], expected, err_msg=str(path))
    assert sum(value.size for value in actual.values()) == model_parameter_count(config.model)


@pytest.mark.parametrize("key", KEYS)
def test_raw_score_variants_qualify_on_the_plain_query_kernels(key):
    config = variant(key, "identity")
    adapter.check_recipe(config)
    summary = adapter.check_initialization(config)
    assert summary["score_transform"] == "identity" and "(s)" in summary["initial_kernel"]
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    gpu = acceptance.test_paths(spec, config_to_dict(config), resolved_backend="mosaic", cpu=False)
    assert set(adapter.RAW_GPU_TESTS) <= set(gpu)
    assert "scaffolds/transformer/tests/test_mosaic_softplus_zero_constant.py" not in gpu


def test_other_transforms_are_refused():
    config = variant(KEYS[0], "sigmoid")
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


def test_raw_score_recipes_may_train_on_b300(tmp_path, monkeypatch):
    from scaffolds.transformer.configs import save_config

    calls = []
    monkeypatch.setattr(adapter, "run_acceptance", lambda spec, argv: calls.append((spec, argv)))
    for transform, allowed in (("identity", True), ("softplus", False)):
        path = tmp_path / f"{transform}.yaml"
        save_config(variant(KEYS[0], transform), path)
        if allowed:
            adapter.main([str(path), "--gpu-type", "B300"])
        else:
            with pytest.raises(ValueError, match="H100 or B200 only"):
                adapter.main([str(path), "--gpu-type", "B300"])
    assert len(calls) == 1
