"""Recipe and initialization controls for the narrow-key runs: each partner's recipe with Q/K width 4, 8 or 16."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from experiments.transformer_pp import key_width_acceptance as adapter
from scaffolds.transformer.configs import config_to_dict

KEYS = tuple(adapter.PARTNERS)


def variant(key, width, **model):
    config = adapter.partner_config(key)
    config.model.qk_head_dim = width
    for name, value in model.items():
        setattr(config.model, name, value)
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


@pytest.mark.parametrize("width", adapter.WIDTHS)
def test_the_exponential_may_shift_by_its_row_maximum(width):
    config = variant("exponential", width, attention_exp_shift="row_max")
    adapter.check_recipe(config)
    assert scientific_differences(config, adapter.partner_config("exponential")) == {
        "model.qk_head_dim", "train.gradient_event_capture", "model.attention_exp_shift"}


@pytest.mark.parametrize("key", ("softplus_sos", "softmax"))
def test_only_the_exponential_takes_the_shift(key):
    config = variant(key, 8)
    config.model.attention_exp_shift = "row_max"
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


def test_softmax_is_the_normalized_partner_on_cudnn():
    config = variant("softmax", 4)
    assert adapter.partner_key(config) == "softmax" and config.model.attention_normalization
    assert config.model.attention_backend == "cudnn" and config.model.attention_output_scaling == "none"
    normalized_exponential = variant("exponential", 4)
    normalized_exponential.model.attention_normalization = True
    with pytest.raises(ValueError):
        adapter.check_recipe(normalized_exponential)


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
    for key, allowed, refused in (("softplus_sos", ("H100", "B200"), "B300"), ("exponential", ("H100", "B300"), "B200"),
                                  ("softmax", ("H100", "B300"), "B200")):
        path = tmp_path / f"{key}.yaml"
        from scaffolds.transformer.configs import save_config

        save_config(variant(key, 8), path)
        for gpu in allowed:
            adapter.require_gpu_type([str(path), "--gpu-type", gpu])
        with pytest.raises(ValueError, match="only"):
            adapter.require_gpu_type([str(path), "--gpu-type", refused])
        adapter.require_gpu_type([str(path), "--runtime-env"])


def test_softmax_gpu_cases_exist_at_the_narrow_widths():
    """The selected cuDNN cases are GPU tests of test_softmax_backends.py at widths 4, 8 and 16 only."""
    path = Path(__file__).resolve().parents[3] / "scaffolds/transformer/tests/test_softmax_backends.py"
    spec = importlib.util.spec_from_file_location("softmax_backend_cases", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    widths = set()
    for test in adapter.GPU_TESTS["softmax"]:
        file, case = test.split("::")
        name, width = case.removesuffix("]").split("[")
        assert Path(file).name == path.name and int(width) in module.ATTENTION_DIMENSIONS
        assert any(mark.name == "gpu" for mark in getattr(module, name).pytestmark)
        widths.add(int(width))
    assert widths == set(adapter.WIDTHS)


def test_acceptance_stages_take_each_partners_cases():
    """The shift's own suite joins both stages of a shifted recipe; softmax's GPU stage takes the narrow widths only."""
    from scaffolds.transformer.acceptance import test_paths as acceptance_test_paths

    shift = "scaffolds/transformer/tests/test_exp_row_max_shift.py"
    shifted = variant("exponential", 4, attention_exp_shift="row_max")
    spec = adapter.experiment_spec(shifted, entry_module=adapter.__name__)
    for cpu in (True, False):
        assert shift in acceptance_test_paths(spec, config_to_dict(shifted), resolved_backend="fused", cpu=cpu)
    assert "scaffolds/transformer/tests/test_unnormalized_fused_attention.py" in acceptance_test_paths(
        spec, config_to_dict(shifted), resolved_backend="fused", cpu=False)
    softmax = variant("softmax", 4)
    spec = adapter.experiment_spec(softmax, entry_module=adapter.__name__)
    gpu = acceptance_test_paths(spec, config_to_dict(softmax), resolved_backend="cudnn", cpu=False)
    assert set(adapter.SOFTMAX_WIDTH_TESTS) <= set(gpu) and shift not in gpu
    assert "scaffolds/transformer/tests/test_softmax_backends.py" not in gpu


def test_an_edited_partner_recipe_is_refused(monkeypatch, tmp_path):
    run = adapter.PARTNERS["exponential"]["run_id"]
    edited = tmp_path / run / "config.yaml"
    edited.parent.mkdir(parents=True)
    edited.write_bytes((adapter.RUNS / run / "config.yaml").read_bytes() + b"# edited\n")
    config = deepcopy(variant("exponential", 8))
    monkeypatch.setattr(adapter, "RUNS", tmp_path)
    with pytest.raises(ValueError, match=f"Pinned key-width partner recipe {run} changed"):
        adapter.check_recipe(config)


# The allocated runs: label, partner, width and the model settings the run chooses.
SOS_LABEL = "query-sos-core-m8-softplus-times-input-total-m9-rope-w-decay"
SHIFTED = {"attention_exp_shift": "row_max"}
RUNS = {
    "r5a25f500bfa8": (f"{SOS_LABEL}-dk4", "softplus_sos", 4, {}),
    "r8d6683db38a8": (f"{SOS_LABEL}-dk8", "softplus_sos", 8, {}),
    "r79abb4d7292c": (f"{SOS_LABEL}-dk16", "softplus_sos", 16, {}),
    "ra28b295d81e6": ("exp-head-scale-dk4", "exponential", 4, {}),
    "r535bc84a7772": ("exp-head-scale-dk8", "exponential", 8, {}),
    "r6f84b1db6de6": ("exp-head-scale-dk16", "exponential", 16, {}),
    "r2739c80cc032": ("exp-head-scale-row-max-dk4", "exponential", 4, SHIFTED),
    "reaca8ac9420c": ("softmax-head-scale-dk4", "softmax", 4, {}),
    "r6f65dc73550a": ("softmax-head-scale-dk8", "softmax", 8, {}),
    "r328466965c2b": ("softmax-head-scale-dk16", "softmax", 16, {}),
}


@pytest.mark.parametrize("run_id", RUNS)
def test_allocated_runs_are_their_partner_at_the_key_width(run_id):
    from scaffolds.transformer.configs import load_config

    label, key, width, model = RUNS[run_id]
    path = adapter.RUNS / run_id / "config.yaml"
    config = load_config(path)
    adapter.check_recipe(config)
    assert adapter.partner_key(config) == key and config.model.qk_head_dim == width
    assert scientific_differences(config, variant(key, width, **model)) == set()
    record = json.loads((adapter.RUNS / run_id / "record.json").read_text())
    assert record["run_id"] == run_id and record["label"] == label == config.train.wandb_run_name
    assert record["config"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_shifted_width_4_exponential_changes_only_the_shift():
    """r2739c80cc032 is ra28b295d81e6's recipe, which overflowed FP32 at update 3,954, with the row-maximum shift."""
    from scaffolds.transformer.configs import load_config

    shifted = load_config(adapter.RUNS / "r2739c80cc032" / "config.yaml")
    unshifted = load_config(adapter.RUNS / "ra28b295d81e6" / "config.yaml")
    assert scientific_differences(shifted, unshifted) == {"model.attention_exp_shift"}
    assert (unshifted.model.attention_exp_shift, shifted.model.attention_exp_shift) == ("none", "row_max")
