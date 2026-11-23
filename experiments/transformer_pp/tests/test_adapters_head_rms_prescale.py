"""Every study adapter takes either evaluation of head RMSNorm.

Fresh recipes with head RMSNorm output scaling carry its max-abs prescale (``model.attention_head_rms_prescale:
max_abs``, which ``scaffolds.records create`` writes), and every recorded partner predates it. The prescale is the
normalization's own arithmetic: it adds no parameter and runs after any kernel, so no partner's qualification depends
on it. Adapters therefore compare recipes through ``scaffolds.transformer.acceptance.comparison_recipe``, which holds
such settings at their historical values on both sides; every other difference from the partner is still refused.
"""
from dataclasses import replace
import importlib
from pathlib import Path

import pytest

from scaffolds.transformer.acceptance import EVALUATION_SETTINGS, comparison_recipe, test_paths as acceptance_test_paths
from scaffolds.transformer.configs import config_to_dict, load_config

STUDY = Path(__file__).resolve().parents[1]
PRESCALE_TESTS = "scaffolds/transformer/tests/test_head_rms_prescale.py"
# Allocated runs each adapter accepts, all with head RMSNorm and recorded before the prescale existed: one per
# adapter, each partner kernel of the key-width adapter, and both branches of the static distance gate's check.
RUNS = {
    "coefficient_forms_acceptance": ("reeefbcc19abf",),
    "coefficient_gdn_gate_acceptance": ("r1a5d217f030d",),
    "coefficient_l2_acceptance": ("r25a5f78841b2",),
    "endpoint_decay_acceptance": ("r02f984133092",),
    "exponential_query_distance_acceptance": ("r17444ad0f60e",),
    "key_width_acceptance": ("r8d6683db38a8", "rcc9e0b03e692", "r161f07127f20", "rd5e3ec5a4f02", "r2739c80cc032"),
    "per_order_cumulative_acceptance": ("r319b83140040",),
    "per_order_decay_acceptance": ("r03be769d2069",),
    "pointwise_attention_acceptance": ("r266c5dd1f935",),
    "query_distance_decay_acceptance": ("r3e26412b2811",),
    "query_distance_degree_acceptance": ("r9320e5cd3071",),
    "score_bias_acceptance": ("r8dc6e147601c",),
    "score_transform_acceptance": ("r1d342ca6e58c",),
    "softplus_direct_acceptance": ("r2026eaa99f9a",),
    "softplus_sos_acceptance": ("r841f35a24241",),
    "softplus_zero_constant_acceptance": ("r7d5baf55104e",),
    "spline_affine_acceptance": ("r2de7f1f4c9ce",),
    "spline_binomial_acceptance": ("r78fa0d682f44",),
    "spline_quadratic_acceptance": ("r2c5883395415",),
    "static_distance_gate_acceptance": ("r0e76396bdee5", "r527a30ff7ed1"),
    "static_fox_acceptance": ("r17fd7f040bff",),
    "static_sos_acceptance": ("r54f771e54722",),
}
# The coefficient-gate runs read another cluster's dataset manifests, which the adapter hashes: its case is the
# pinned control with the gate, as in the adapter's own tests.
CONSTRUCTED = ("coefficient_gate_acceptance",)
# Adapters whose mechanism needs the prescale (the row-power shift is divided out through it), each with a run; of
# the learning-rate adapter's partners that is z h_t(z), and its own tests hold its exponential to the shared rule.
PRESCALED_ONLY = {"shifted_sos_acceptance": ("r2e7f5617fdf2",), "learning_rate_acceptance": ("r1934bca84bd4",)}
CASES = (*((name, run) for name, runs in RUNS.items() for run in runs), *((name, None) for name in CONSTRUCTED))


def adapter_module(name):
    return importlib.import_module(f"experiments.transformer_pp.{name}")


def recipe(name, run):
    if run is not None:
        return load_config(STUDY / "runs" / run / "config.yaml")
    config = adapter_module(name).control_config()
    config.model.attention_coefficient_gate = "sigmoid"
    return config


def with_prescale(config, prescale):
    return replace(config, model=replace(config.model, attention_head_rms_prescale=prescale))


def test_every_adapter_has_a_case():
    """A new adapter joins this module, so its comparison is held to the shared rule too."""
    adapters = {path.stem for path in STUDY.glob("*_acceptance.py")}
    assert adapters == set(RUNS) | set(CONSTRUCTED) | set(PRESCALED_ONLY)


@pytest.mark.parametrize("name, run", tuple((name, run) for name, runs in PRESCALED_ONLY.items() for run in runs))
def test_an_adapter_whose_mechanism_needs_the_prescale_refuses_the_plain_evaluation(name, run):
    adapter, config = adapter_module(name), recipe(name, run)
    assert config.model.attention_head_rms_prescale == "max_abs"
    adapter.check_recipe(config)
    with pytest.raises(ValueError):
        adapter.check_recipe(with_prescale(config, "none"))


@pytest.mark.parametrize("name, run", CASES)
def test_adapter_takes_either_evaluation_of_head_rms_norm_and_nothing_else(name, run):
    adapter, config = adapter_module(name), recipe(name, run)
    assert config.model.attention_output_scaling == "head_rms_norm"
    assert config.model.attention_head_rms_prescale == "none"
    adapter.check_recipe(config)
    prescaled = with_prescale(config, "max_abs")
    adapter.check_recipe(prescaled)
    # The prescale loosens nothing else: any other difference from the partner is refused as before.
    for changed in (config, prescaled):
        changed = replace(changed, optim=replace(changed.optim, weight_decay=changed.optim.weight_decay + 0.5))
        with pytest.raises(ValueError, match="optim.weight_decay"):
            adapter.check_recipe(changed)


def test_a_prescaled_run_selects_the_prescale_tests_through_its_adapter():
    """The comparison leaves the prescale out; the run's acceptance still qualifies it: the module's CPU cases and
    its two device cases join the adapter's stages."""
    adapter = adapter_module("key_width_acceptance")
    prescaled = with_prescale(recipe("key_width_acceptance", "r8d6683db38a8"), "max_abs")
    spec = adapter.experiment_spec(prescaled, entry_module=adapter.__name__)
    assert PRESCALE_TESTS in acceptance_test_paths(spec, config_to_dict(prescaled), resolved_backend="mosaic", cpu=True)
    device = acceptance_test_paths(spec, config_to_dict(prescaled), resolved_backend="mosaic", cpu=False)
    assert sum(path.startswith(f"{PRESCALE_TESTS}::") for path in device) == 2
    plain = with_prescale(prescaled, "none")
    spec = adapter.experiment_spec(plain, entry_module=adapter.__name__)
    for cpu in (True, False):
        assert not any(PRESCALE_TESTS in path for path in acceptance_test_paths(
            spec, config_to_dict(plain), resolved_backend="mosaic", cpu=cpu))


def test_a_partner_without_head_rms_norm_still_refuses_the_prescale():
    """Normalized softmax has no head RMSNorm, so the recipe itself is invalid whatever the adapter compares."""
    softmax = recipe("key_width_acceptance", "reaca8ac9420c")
    assert softmax.model.attention_output_scaling == "none"
    adapter_module("key_width_acceptance").check_recipe(softmax)
    with pytest.raises(ValueError, match="head RMSNorm output scaling"):
        with_prescale(softmax, "max_abs")


def test_the_comparison_recipe_is_the_recipe_but_for_its_evaluation_settings():
    assert EVALUATION_SETTINGS == {"attention_head_rms_prescale": "none"}
    config = recipe("key_width_acceptance", "r8d6683db38a8")
    assert comparison_recipe(config) == config_to_dict(config)
    prescaled = with_prescale(config, "max_abs")
    written, compared = config_to_dict(prescaled), comparison_recipe(prescaled)
    assert written["model"]["attention_head_rms_prescale"] == "max_abs"  # the recipe itself keeps its setting
    assert compared == comparison_recipe(config)
    assert {key for key in written["model"] if written["model"][key] != compared["model"][key]} == set(
        EVALUATION_SETTINGS)
