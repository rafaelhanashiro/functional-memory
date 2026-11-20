"""Recipe and initialization controls for the direct softplus runs: p(z), z p(z) and z p_t(z), z = softplus(s)."""
from dataclasses import replace
import hashlib
import json

import pytest

from experiments.transformer_pp import softplus_direct_acceptance as adapter
from scaffolds.transformer.configs import config_to_dict

# The run kinds: partner, transform and prefactor.
KINDS = {"static p(z)": ("static", "softplus", "none"), "static z p(z)": ("static", "softplus", "softplus"),
         "query z p_t(z)": ("query", "softplus", "softplus")}


def variant(kind):
    key, transform, prefactor = KINDS[kind]
    config = adapter.partner_config(key)
    config.model.attention_score_transform = transform
    config.model.attention_score_prefactor = prefactor
    for name, value in adapter.CAPTURE_POLICY.items():
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


@pytest.mark.parametrize("key", tuple(adapter.PARTNERS))
def test_pinned_partners_are_the_recorded_runs(key):
    partner = adapter.PARTNERS[key]
    path = adapter.RUNS / partner["run_id"] / "config.yaml"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == partner["config_sha256"]
    assert json.loads((adapter.RUNS / partner["run_id"] / "record.json").read_text())["config"]["sha256"] == (
        partner["config_sha256"])
    config = adapter.partner_config(key)
    assert (config.model.attention_degree, config.model.attention_coefficient_parameterization) == (8, "direct")
    assert config.model.attention_backend == "mosaic" and not config.model.attention_coefficient_zero_constant
    assert (config.model.num_layers, config.model.hidden_size, config.optim.total_steps) == (15, 896, 7570)
    from scaffolds.transformer.models import model_parameter_count

    assert model_parameter_count(config.model) == partner["initialization"][0]


@pytest.mark.parametrize("kind", KINDS)
def test_runs_are_their_partner_with_the_score_functions_and_capture(kind):
    config = variant(kind)
    adapter.check_recipe(config)
    key, transform, prefactor = KINDS[kind]
    expected = {"model.attention_score_prefactor"} if prefactor != "none" else set()
    if key == "static":
        expected |= {"model.attention_score_transform", "train.gradient_event_capture"}
    assert scientific_differences(config, adapter.partner_config(key)) == expected


@pytest.mark.parametrize("kind,field,value", (
    ("static p(z)", "attention_score_prefactor", "exp"),
    ("static p(z)", "attention_score_transform", "identity"),
    ("static p(z)", "attention_coefficient_zero_constant", True),
    ("query z p_t(z)", "attention_score_prefactor", "none"),
    ("query z p_t(z)", "attention_score_prefactor", "sigmoid"),
    ("query z p_t(z)", "attention_coefficient_constant_initial", None),
    ("query z p_t(z)", "attention_degree", 9),
    ("static z p(z)", "attention_coefficient_scale", "none"),
))
def test_other_changes_are_refused(kind, field, value):
    config = variant(kind)
    setattr(config.model, field, value)
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


@pytest.mark.parametrize("kind", ("static p(z)", "static z p(z)"))
def test_static_runs_take_the_capture_policy(kind):
    config = variant(kind)
    config.train.gradient_event_capture = False
    with pytest.raises(ValueError, match="train.gradient_event_capture"):
        adapter.check_recipe(config)


@pytest.mark.parametrize("kind", KINDS)
def test_initial_parameters_are_the_partners_on_a_small_backbone(kind):
    """The transform and prefactor add no parameter: on a small backbone built as the full one, the run's initial
    arrays are its partner's (the full fingerprints are pinned and checked by acceptance)."""
    from scaffolds.transformer.acceptance import initialization_facts

    def small(config):
        return replace(config, model=replace(config.model, attention_backend="blocked", num_layers=1, hidden_size=64,
                                             num_heads=2, num_kv_heads=2, intermediate_size=128, vocab_size=64))

    key = KINDS[kind][0]
    assert initialization_facts(small(variant(kind))) == initialization_facts(small(adapter.partner_config(key)))


@pytest.mark.parametrize("kind", KINDS)
def test_initialization_gate_uses_the_pinned_fingerprint(monkeypatch, kind):
    key = KINDS[kind][0]
    count, sha256 = adapter.PARTNERS[key]["initialization"]
    monkeypatch.setattr(adapter, "matches_softmax_backbone_initialization", lambda _: {})
    monkeypatch.setattr(adapter, "initialization_facts", lambda _: {"parameter_count": count,
                                                                    "parameter_sha256": sha256})
    facts = adapter.check_initialization(variant(kind))
    assert facts["all_initial_parameters_match_partner"]
    assert facts["initial_kernel"] == {"static p(z)": "T_8(z)", "static z p(z)": "z*T_8(z)",
                                       "query z p_t(z)": "z*(T_8(z) - 1)"}[kind]
    monkeypatch.setattr(adapter, "initialization_facts", lambda _: {"parameter_count": count + 1,
                                                                    "parameter_sha256": sha256})
    with pytest.raises(RuntimeError, match="Initial parameters differ"):
        adapter.check_initialization(variant(kind))


@pytest.mark.parametrize("kind", KINDS)
def test_acceptance_selects_the_transformed_and_prefactor_kernels(kind):
    config = variant(kind)
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    assert spec.required_backend == "mosaic"
    assert set(adapter.GPU_TESTS) <= set(spec.gpu_tests)
    assert (set(adapter.PREFACTOR_GPU_TESTS) <= set(spec.gpu_tests)) is (KINDS[kind][2] != "none")


@pytest.mark.parametrize("gpu_type", ("B300", "A100"))
def test_other_gpu_types_are_refused(gpu_type):
    with pytest.raises(ValueError, match="H100 or B200 only"):
        adapter.require_gpu_type(["--gpu-type", gpu_type])
    adapter.require_gpu_type(["--gpu-type", "B200"])
    adapter.require_gpu_type(["--runtime-env"])


# The allocated runs: label and kind.
RUNS = {
    "r2026eaa99f9a": ("static-m8-softplus-input", "static p(z)"),
    "rd00848e00f1d": ("static-m8-softplus-input-times-input", "static z p(z)"),
    "r7f80334aafbd": ("query-m8-dk64-w-decay-softplus-input-times-input", "query z p_t(z)"),
}


@pytest.mark.parametrize("run_id", RUNS)
def test_allocated_runs_are_their_partner_with_the_score_functions(run_id):
    from scaffolds.transformer.configs import load_config

    label, kind = RUNS[run_id]
    path = adapter.RUNS / run_id / "config.yaml"
    config = load_config(path)
    adapter.check_recipe(config)
    assert scientific_differences(config, variant(kind)) == set()
    record = json.loads((adapter.RUNS / run_id / "record.json").read_text())
    assert record["run_id"] == run_id and record["label"] == label == config.train.wandb_run_name
    assert record["config"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
