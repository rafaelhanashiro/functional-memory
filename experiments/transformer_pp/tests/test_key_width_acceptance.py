"""Recipe and initialization controls for the narrow-key runs: each partner's recipe with Q/K width 4, 8 or 16."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import pytest

from experiments.transformer_pp import key_width_acceptance as adapter
from scaffolds.transformer.configs import config_to_dict

KEYS = tuple(adapter.PARTNERS)


def variant(key, width):
    config = adapter.partner_config(key)
    config.model.qk_head_dim = width
    for name, value in adapter.PARTNERS[key]["train"].items():
        setattr(config.train, name, value)
    return config


def scientific_differences(config, other):
    first, second = config_to_dict(config), config_to_dict(other)
    for recipe in (first, second):
        for name in adapter.EXECUTION_KEYS:
            recipe["train"].pop(name, None)
    return {f"{section}.{name}" for section in set(first) | set(second)
            for name in set(first.get(section, {})) | set(second.get(section, {}))
            if first.get(section, {}).get(name) != second.get(section, {}).get(name)}


@pytest.mark.parametrize("key", KEYS)
def test_pinned_partners_are_the_recorded_runs(key):
    partner = adapter.PARTNERS[key]
    path = adapter.RUNS / partner["run_id"] / "config.yaml"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == partner["config_sha256"]
    record = json.loads((adapter.RUNS / partner["run_id"] / "record.json").read_text())
    assert record["config"]["sha256"] == partner["config_sha256"]
    config = adapter.partner_config(key)
    assert (config.model.num_layers, config.model.hidden_size, config.optim.total_steps) == (15, 896, 7570)
    assert (config.model.qk_head_dim, config.model.value_head_dim, config.train.seed) == (64, 64, 42)
    assert config.model.attention_backend == partner["backend"] and adapter.partner_key(config) == key


@pytest.mark.parametrize("width", adapter.WIDTHS)
@pytest.mark.parametrize("key", KEYS)
def test_only_the_key_width_and_capture_change(key, width):
    config = variant(key, width)
    adapter.check_recipe(config)
    expected = {"model.qk_head_dim", *(("train.gradient_event_capture",) if key == "exponential" else ())}
    assert scientific_differences(config, adapter.partner_config(key)) == expected
    assert config.model.value_head_dim == 64


@pytest.mark.parametrize("width", (2, 32, 64, 128))
@pytest.mark.parametrize("key", KEYS)
def test_other_key_widths_are_refused(key, width):
    config = variant(key, 16)
    config.model.qk_head_dim = width
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


@pytest.mark.parametrize("section,name,value", (
    ("model", "value_head_dim", 16), ("model", "num_heads", 7), ("optim", "learning_rate", 0.002),
    ("train", "seed", 43), ("model", "attention_backend", "blocked"), ("data", "batch_size", 128),
))
@pytest.mark.parametrize("key", KEYS)
def test_unrelated_changes_are_refused(key, section, name, value):
    config = variant(key, 8)
    setattr(getattr(config, section), name, value)
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


def test_the_exponential_runs_take_the_capture_policy():
    config = variant("exponential", 8)
    config.train.gradient_event_capture = False
    with pytest.raises(ValueError, match="train.gradient_event_capture"):
        adapter.check_recipe(config)


@pytest.mark.parametrize("key", KEYS)
def test_initialization_differs_only_in_the_reshaped_arrays_on_a_small_backbone(key):
    """Every partner array of unchanged shape is bit-identical at width 4, and Q, K and their gains change."""
    def small(config):
        return replace(config, model=replace(config.model, num_layers=2, hidden_size=64, num_heads=2, num_kv_heads=2,
                                             intermediate_size=128, vocab_size=64))

    facts = adapter.compare_with_partner(small(variant(key, 4)), small(adapter.partner_config(key)))
    assert facts["all_other_partner_arrays_identical"]
    reshaped = facts["reshaped_parameters"]
    assert any("q_proj" in path for path in reshaped) and any("k_proj" in path for path in reshaped)
    assert any("qk_norm_q_gain" in path for path in reshaped) and any("qk_norm_k_gain" in path for path in reshaped)
    assert not any("v_proj" in path or "o_proj" in path for path in reshaped)


@pytest.mark.parametrize("width", adapter.WIDTHS)
@pytest.mark.parametrize("key", KEYS)
def test_initialization_gate_uses_the_pinned_fingerprint(monkeypatch, key, width):
    count, sha256 = adapter.PARTNERS[key]["initialization"][width]
    monkeypatch.setattr(adapter, "matches_softmax_backbone_initialization", lambda _: {})
    monkeypatch.setattr(adapter, "compare_with_partner", lambda *_: {"all_other_partner_arrays_identical": True})
    monkeypatch.setattr(adapter, "initialization_facts", lambda _: {"parameter_count": count,
                                                                    "parameter_sha256": sha256})
    facts = adapter.check_initialization(variant(key, width))
    assert facts["pinned_initialization_matches"] and facts["qk_head_dim"] == width
    monkeypatch.setattr(adapter, "initialization_facts", lambda _: {"parameter_count": count,
                                                                    "parameter_sha256": "changed"})
    with pytest.raises(RuntimeError, match="Initial parameters differ"):
        adapter.check_initialization(variant(key, width))


def test_pinned_counts_are_the_model_formula():
    from scaffolds.transformer.models import model_parameter_count

    for key in KEYS:
        for width, (count, _) in adapter.PARTNERS[key]["initialization"].items():
            assert model_parameter_count(variant(key, width).model) == count


@pytest.mark.parametrize("key", KEYS)
def test_acceptance_selects_each_partners_kernels(key):
    spec = adapter.experiment_spec(variant(key, 4), entry_module=adapter.__name__)
    assert spec.required_backend == adapter.PARTNERS[key]["backend"]
    assert set(adapter.GPU_TESTS[key]) <= set(spec.gpu_tests)
    assert set(adapter.CPU_TESTS) <= set(spec.cpu_tests)
    assert spec.check_initialization is adapter.check_initialization


def test_each_partners_gpus_are_required(tmp_path):
    for key, allowed, refused in (("softplus_sos", ("H100", "B200"), "B300"), ("exponential", ("H100", "B300"), "B200")):
        path = tmp_path / f"{key}.yaml"
        from scaffolds.transformer.configs import save_config

        save_config(variant(key, 8), path)
        for gpu in allowed:
            adapter.require_gpu_type([str(path), "--gpu-type", gpu])
        with pytest.raises(ValueError, match="only"):
            adapter.require_gpu_type([str(path), "--gpu-type", refused])
        adapter.require_gpu_type([str(path), "--runtime-env"])


def test_an_edited_partner_recipe_is_refused(monkeypatch, tmp_path):
    run = adapter.PARTNERS["exponential"]["run_id"]
    edited = tmp_path / run / "config.yaml"
    edited.parent.mkdir(parents=True)
    edited.write_bytes((adapter.RUNS / run / "config.yaml").read_bytes() + b"# edited\n")
    config = deepcopy(variant("exponential", 8))
    monkeypatch.setattr(adapter, "RUNS", tmp_path)
    with pytest.raises(ValueError, match=f"Pinned key-width partner recipe {run} changed"):
        adapter.check_recipe(config)
