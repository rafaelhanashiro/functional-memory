"""Frozen recipe and requested comparison coverage without allocating a 200M model."""
import copy
import math
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import pytest
import yaml

from experiments.transformer_200m_4b import acceptance, pipeline
from scaffolds.transformer.acceptance import test_paths as acceptance_test_paths
from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.tests.test_qk_rms_attention_backends import CASES


RECIPE = Path(__file__).resolve().parents[1] / "recipe.yaml"
BASE_PARAMETERS = 198_439_936


def _load_variant(tmp_path, kind, degree):
    base = config_to_dict(load_config(RECIPE))
    original = copy.deepcopy(base)
    raw = pipeline.variant_config(base, kind, degree)
    assert base == original, "Making a comparison must not mutate the common recipe"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(path)


def _shape_parameter_count(config):
    # Trace initializers and model ownership; ShapeDtypeStructs retain no weight data.
    parameters = jax.eval_shape(lambda: nnx.state(build_model(config.model, config.train.seed), nnx.Param))
    leaves = jax.tree.leaves(parameters)
    assert all(leaf.dtype == jnp.float32 for leaf in leaves)
    return sum(math.prod(leaf.shape) for leaf in leaves)


def test_reference_recipe_matches_agreed_count_schedule_and_lr_gate():
    config = load_config(RECIPE)
    acceptance.check_config(config)
    assert _shape_parameter_count(config) == model_parameter_count(config.model) == BASE_PARAMETERS
    assert config.optim.total_steps * config.data.batch_size * (config.data.sequence_length - 1) == 3_968_860_160
    assert (config.train.eval_early_every, config.train.eval_early_until, config.train.eval_every) == (100, 1500, 500)
    assert config.train.eval_batches * config.data.batch_size * 2048 == 9_961_472
    assert len(pipeline.PEAKS) == 3 and pipeline.PILOT_STEPS == 1500
    selected = acceptance_test_paths(acceptance.spec(config), config_to_dict(config),
                                     resolved_backend="cudnn", cpu=False)
    assert f"{acceptance.BACKEND_TESTS}::test_full_context_qk_rms_backend[softmax-0-cudnn]" in selected


@pytest.mark.parametrize(("kind", "degree"), pipeline.COMPARISONS,
                         ids=[f"{kind}-{degree}" for kind, degree in pipeline.COMPARISONS])
def test_all_comparisons_preserve_recipe_and_have_matching_gpu_gates(tmp_path, kind, degree):
    config = _load_variant(tmp_path, kind, degree)
    acceptance.check_config(config)
    assert len(pipeline.COMPARISONS) == len(set(pipeline.COMPARISONS)) == 13
    model = config.model
    assert model.attention_normalization is False and model.attention_output_scaling == "head_rms_norm"
    extra = 15 * 14 * 64  # output RMSNorm gains for every layer and head
    if kind in ("static", "query", "query_key"):
        per_order = {"static": 1, "query": 896 + 2, "query_key": 2 * 896 + 3}[kind]
        extra += 15 * 14 * (degree + 1) * per_order
        assert model.attention_coefficient_scale == "factorial"
    if kind in ("query", "query_key"):
        assert model.attention_coefficient_amplitude_scale == "factorial"
    assert _shape_parameter_count(config) == model_parameter_count(model) == BASE_PARAMETERS + extra
    case = (kind, degree or 0, model.attention_backend)
    assert case in CASES
    node = f"{acceptance.BACKEND_TESTS}::test_full_context_qk_rms_backend[{kind}-{degree or 0}-{model.attention_backend}]"
    selected = acceptance_test_paths(acceptance.spec(config), config_to_dict(config),
                                     resolved_backend=model.attention_backend, cpu=False)
    assert node in selected


@pytest.mark.parametrize("kind", ("query", "query_key"))
@pytest.mark.parametrize("field", ("attention_coefficient_scale", "attention_coefficient_amplitude_scale"))
def test_partial_order_scaling_is_rejected_before_launch(tmp_path, kind, field):
    config = _load_variant(tmp_path, kind, 12)
    setattr(config.model, field, "none")
    with pytest.raises(ValueError, match="scal"):
        acceptance.check_config(config)
