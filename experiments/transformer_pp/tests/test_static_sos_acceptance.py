"""Pinned static SoS recipes cannot change settings beyond removing the query generator."""
import pytest

from experiments.transformer_pp import static_sos_acceptance as adapter
from scaffolds.transformer.acceptance import test_paths as acceptance_paths
from scaffolds.transformer.configs import config_to_dict
from scaffolds.transformer.tests.test_static_sos import _config
from scaffolds.transformer.tests.test_softplus_sos import _config as query_config


@pytest.mark.parametrize("key", tuple(adapter.CONTROLS))
def test_six_static_recipes_preserve_query_partner(key):
    config = adapter.static_config(*key)
    adapter.check_recipe(config)
    assert config.model.attention_static_sos_parameterization == "relative_taylor"
    assert config.model.attention_coefficient_conditioning == "static"
    assert not config.model.attention_coefficient_kernel_weight_decay
    assert config.model.qk_norm_head_scale
    assert config.model.positional_encoding == "rope"
    assert config.optim.total_steps == 7570
    assert config.train.seed == 42


@pytest.mark.parametrize("section,key,value", (
    ("model", "attention_static_sos_parameterization", "direct"),
    ("model", "attention_coefficient_kernel_weight_decay", True),
    ("optim", "learning_rate", .002), ("optim", "total_steps", 19074),
    ("data", "batch_size", 128), ("model", "attention_backend", "blocked"),
    ("model", "positional_encoding", "none"), ("train", "seed", 0),
))
def test_unrelated_static_recipe_edits_are_rejected(section, key, value):
    config = adapter.static_config(8, "softplus", 1)
    setattr(getattr(config, section), key, value)
    with pytest.raises(ValueError):
        adapter.check_recipe(config)


@pytest.mark.parametrize("core", (8, 12))
def test_initial_parameter_check_matches_query_backbone_and_factor_start(core):
    static = _config(core, "softplus", 1)
    query = query_config(1, core_degree=core)
    facts = adapter._initial_parameter_check(static, query)
    assert facts["all_backbone_parameters_match_control"]
    assert facts["effective_factors_match_control"]
    assert facts["static_factor_shape"] == [2, 2, core + 1]
    assert not facts["factor_weight_decay"]
    query.train.seed += 1
    with pytest.raises(RuntimeError, match="backbone differs"):
        adapter._initial_parameter_check(static, query)


@pytest.mark.parametrize("key", tuple(adapter.CONTROLS))
def test_all_recipes_require_static_numerics_and_initialization(key):
    config = adapter.static_config(*key)
    spec = adapter.experiment_spec(config, entry_module=adapter.__name__)
    assert spec.required_backend == "mosaic"
    assert spec.check_initialization == adapter.check_initialization
    assert "scaffolds/transformer/tests/test_static_sos.py" in spec.cpu_tests
    assert "scaffolds/transformer/tests/test_mosaic_static_sos.py" in spec.gpu_tests
    assert "scaffolds/transformer/tests/test_mosaic_static_sos.py" in acceptance_paths(
        spec, config_to_dict(config), resolved_backend="mosaic", cpu=False)


def test_missing_control_fails_closed():
    with pytest.raises(ValueError, match="requires core degree"):
        adapter.control_config(4, "identity", 0)
