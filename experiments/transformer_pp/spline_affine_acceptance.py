"""Qualify the affine-start spline comparison against the original spline recipes."""

from dataclasses import replace
from functools import partial
import hashlib
from pathlib import Path

from scaffolds.transformer.acceptance import (
    attention_spec, main as run_acceptance, matches_softmax_backbone_initialization,
)
from scaffolds.transformer.configs import Config, config_to_dict, load_config
from scaffolds.transformer.configs.base import validate_config


SOURCE_RUNS = {
    "static": ("r7e4193fcafbd", "250aa775cc7e7635512a7a1194e40a5f9830d3326880bed1842d7674f83be2f9"),
    "query": ("rf303a5f94487", "1c8ea9e57d7f7a6e4ce0308f12c0fa50a3fba3b8adaea8b6972a26da4f2b2e14"),
}
AFFINE_INITIALIZATION = {
    "attention_coefficient_initialization": "affine",
    "attention_spline_initial_intercept": 1.0,
    "attention_spline_initial_slope": 0.25,
}
INITIAL_COEFFICIENTS = (-7.0, -5.375, -3.375, -1.0, 3.0, 5.375, 7.375, 9.0)
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name", "wandb_group", "wandb_dir",
                  "wandb_mode")
ADAPTER_TESTS = "experiments/transformer_pp/tests/test_spline_affine_acceptance.py"


def source_config(conditioning: str) -> Config:
    if conditioning not in SOURCE_RUNS:
        raise ValueError("This comparison contains only static and query spline attention")
    run_id, expected_hash = SOURCE_RUNS[conditioning]
    path = Path(__file__).resolve().parent / "runs" / run_id / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError(f"The pinned source recipe {run_id} changed")
    return load_config(path)


def check_recipe(config: Config) -> None:
    """The only scientific change from each source is the initial spline function."""
    validate_config(config)
    source = source_config(config.model.attention_coefficient_conditioning)
    actual, expected = config_to_dict(config), config_to_dict(source)
    for field, value in AFFINE_INITIALIZATION.items():
        if actual["model"][field] != value:
            raise ValueError(f"This comparison requires model.{field}: {value}")
        actual["model"][field] = expected["model"][field]
    for recipe in (actual, expected):
        for key in EXECUTION_KEYS:
            recipe["train"].pop(key, None)
    if actual != expected:
        differences = [f"{section}.{key}" for section in set(actual) | set(expected)
                       for key in set(actual.get(section, {})) | set(expected.get(section, {}))
                       if actual.get(section, {}).get(key) != expected.get(section, {}).get(key)]
        raise ValueError(f"Scientific settings differ from the spline source: {sorted(differences)}")


def check_initialization(config: Config) -> dict:
    """Inspect initialized model arrays and the mask supplied to the actual AdamW optimizer."""
    import jax
    import numpy as np
    from flax import nnx

    from scaffolds.transformer.attention.parameters import attention_parameter_specs
    from scaffolds.transformer.models import build_model, model_parameter_count
    from scaffolds.transformer.training.checkpointing import state_sha256
    from scaffolds.transformer.training.optim import attention_weight_decay_mask

    check_recipe(config)
    facts = matches_softmax_backbone_initialization(config)
    conditioning = config.model.attention_coefficient_conditioning
    source = source_config(conditioning)
    rng = jax.random.PRNGKey(config.train.seed)
    states, masks = [], []
    for recipe in (config, source):
        model = build_model(recipe.model, nnx.Rngs(params=rng, dropout=jax.random.fold_in(rng, 0)))
        params = nnx.state(model, nnx.Param)
        excluded = frozenset(spec.name for spec in attention_parameter_specs(recipe.model) if not spec.weight_decay)
        mask = attention_weight_decay_mask(params, excluded_names=excluded)
        masks.append({path: bool(value.get_value()) for path, value in nnx.to_flat_state(mask)})
        states.append(params)
    actual, original = states
    model = config.model
    coefficient_shape = (model.num_layers, model.num_heads, len(INITIAL_COEFFICIENTS))
    coefficient_name = "query_spline_bias" if conditioning == "query" else "spline_coefficients"
    coefficients = np.asarray(actual[coefficient_name][...])
    if (coefficients.dtype != np.float32 or coefficients.shape != coefficient_shape
            or not np.array_equal(coefficients, np.broadcast_to(INITIAL_COEFFICIENTS, coefficient_shape))):
        raise RuntimeError("Initialized spline coefficients do not equal the exact affine coefficients in FP32")
    parameter_count = sum(leaf.size for leaf in jax.tree.leaves(actual))
    if parameter_count != model_parameter_count(model):
        raise RuntimeError("Actual parameter count differs from the model recipe")
    expected_decay = {coefficient_name: False}
    if conditioning == "query":
        amplitude = np.asarray(actual["query_spline_amplitude"][...])
        kernel = np.asarray(actual["query_spline_kernel"][...])
        if (amplitude.shape != coefficient_shape or amplitude.dtype != np.float32
                or not np.array_equal(amplitude, np.zeros(coefficient_shape, np.float32))):
            raise RuntimeError("Query spline amplitudes must initialize at zero in FP32")
        expected_shape = (model.num_layers, model.num_heads, model.hidden_size, len(INITIAL_COEFFICIENTS))
        if kernel.shape != expected_shape or kernel.dtype != np.float32 or not np.isfinite(kernel).all():
            raise RuntimeError("Query spline projection has the wrong shape, precision, or nonfinite values")
        expected_decay.update(query_spline_amplitude=False, query_spline_kernel=True)
    if masks[0] != masks[1]:
        raise RuntimeError("The affine initialization changed the source run's optimizer decay mask")
    for name, decay in expected_decay.items():
        if masks[0][(name,)] != decay:
            raise RuntimeError(f"Unexpected optimizer decay mask for {name}")
    del actual[coefficient_name]
    del original[coefficient_name]
    shared_hash = state_sha256(actual)
    if shared_hash != state_sha256(original):
        raise RuntimeError("Parameters other than the initial coefficient vector differ from the source")
    run_id, source_hash = SOURCE_RUNS[conditioning]
    return {**facts, "affine_spline_start": {
        "source_run_id": run_id,
        "source_recipe_sha256": source_hash,
        "initial_function": "1 + score / 4",
        "initial_coefficients": list(INITIAL_COEFFICIENTS),
        "coefficient_shape": list(coefficient_shape),
        "actual_parameter_count": parameter_count,
        "unchanged_parameter_sha256": shared_hash,
        "all_other_parameters_match_source_initialization": True,
        "query_projection_draw_matches_source": conditioning == "query",
        "optimizer_decay_mask_matches_source": True,
        "spline_parameter_decay_mask": expected_decay,
    }}


def experiment_spec(config: Config, *, entry_module: str):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, cpu_tests=(*shared.cpu_tests, ADAPTER_TESTS),
                   check_config=check_recipe, check_initialization=check_initialization)


def main(argv=None) -> None:
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()
