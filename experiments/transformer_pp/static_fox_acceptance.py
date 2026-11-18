"""Qualify static coefficients as a one-change ablation of the FoX NoPE control."""
from dataclasses import replace
from functools import partial
import math

from experiments.transformer_pp.query_distance_decay_acceptance import (
    INITIALIZATION_SHA256 as CONTROL_INITIALIZATION_SHA256,
    control_config,
)
from scaffolds.transformer.acceptance import attention_spec, main as run_acceptance
from scaffolds.transformer.configs import config_to_dict
from scaffolds.transformer.configs.base import validate_config


STATIC_SETTINGS = {
    "attention_coefficient_conditioning": "static",
    "attention_coefficient_activation": "identity",
    "attention_coefficient_kernel_weight_decay": False,
    "attention_coefficient_amplitude_scale": "none",
    "attention_coefficient_amplitude_initial": 1.0,
    "attention_coefficient_projection_initial_std": 0.0,
}
CONTROL_RUN_ID = "r4a80e28ce0a7"
QUERY_PARAMETERS = {"query_polynomial_bias", "query_polynomial_amplitude", "query_polynomial_kernel"}


def check_recipe(config) -> None:
    validate_config(config)
    actual, expected = config_to_dict(config), config_to_dict(control_config())
    expected["model"].update(STATIC_SETTINGS)
    # Only storage locations and monitoring labels are execution choices.
    for recipe in (actual, expected):
        for key in ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name", "wandb_group", "wandb_dir", "wandb_mode"):
            recipe["train"].pop(key, None)
    if actual != expected:
        differences = [f"{section}.{key}" for section in set(actual) | set(expected)
                       for key in set(actual.get(section, {})) | set(expected.get(section, {}))
                       if actual.get(section, {}).get(key) != expected.get(section, {}).get(key)]
        raise ValueError(f"Scientific settings differ from the static FoX ablation: {sorted(differences)}")


def check_initialization(config) -> dict:
    """Compare instantiated parameters and optimizer masks to the pinned control."""
    import jax
    import numpy as np
    from flax import nnx

    from scaffolds.transformer.attention.parameters import attention_parameter_specs, factorial_scale
    from scaffolds.transformer.models import build_model
    from scaffolds.transformer.training.checkpointing import state_sha256
    from scaffolds.transformer.training.optim import attention_weight_decay_mask

    check_recipe(config)
    control = control_config()

    def parameters(recipe):
        rng = jax.random.PRNGKey(recipe.train.seed)
        model = build_model(recipe.model, nnx.Rngs(params=rng, dropout=jax.random.fold_in(rng, 0)))
        return nnx.state(model, nnx.Param)

    actual, expected = parameters(config), parameters(control)
    control_hash = state_sha256(expected)
    if control_hash != CONTROL_INITIALIZATION_SHA256:
        raise RuntimeError("Recreated FoX control no longer matches its recorded initialization")
    actual_flat, expected_flat = dict(nnx.to_flat_state(actual)), dict(nnx.to_flat_state(expected))
    if set(actual_flat) - set(expected_flat) != {("polynomial_coefficients",)} or (
            set(expected_flat) - set(actual_flat) != {(name,) for name in QUERY_PARAMETERS}):
        raise RuntimeError("Static FoX changed parameters outside the coefficient generator")
    for path in set(actual_flat) & set(expected_flat):
        if not np.array_equal(np.asarray(actual_flat[path][...]), np.asarray(expected_flat[path][...])):
            raise RuntimeError(f"Initialization differs from FoX control for {path}")

    coefficients = actual["polynomial_coefficients"][...]
    expected_shape = (config.model.num_layers, config.model.num_heads, config.model.attention_degree + 1)
    if coefficients.shape != expected_shape or not np.all(np.asarray(coefficients) == 1.0):
        raise RuntimeError("Static coefficients must initialize to one per layer, head, and order")
    taylor = np.asarray([1.0 / math.factorial(r) for r in range(config.model.attention_degree + 1)], np.float32)
    if not np.array_equal(np.asarray(factorial_scale(coefficients)), np.broadcast_to(taylor, expected_shape)):
        raise RuntimeError("Effective static coefficients do not equal the Taylor initialization")

    def decay_masks(params, recipe):
        excluded = {spec.name for spec in attention_parameter_specs(recipe.model) if not spec.weight_decay}
        return dict(nnx.to_flat_state(attention_weight_decay_mask(params, excluded_names=excluded)))

    actual_masks, expected_masks = decay_masks(actual, config), decay_masks(expected, control)
    for path in set(actual_masks) & set(expected_masks):
        if bool(actual_masks[path].get_value()) != bool(expected_masks[path].get_value()):
            raise RuntimeError(f"Weight decay differs from FoX control for {path}")
    required_masks = {"polynomial_coefficients": False, "forget_gate_kernel": True,
                      "forget_gate_log_rate": False, "forget_gate_dt_bias": False}
    for name, required in required_masks.items():
        if bool(actual_masks[(name,)].get_value()) != required:
            raise RuntimeError(f"Unexpected weight decay on {name}")
    return {"control_run_id": CONTROL_RUN_ID,
            "control_parameter_sha256": control_hash,
            "shared_initial_parameters_match_control": True,
            "gate_initial_parameters_match_control": True,
            "static_coefficient_shape": list(coefficients.shape),
            "static_effective_initial_coefficients": taylor.tolist(),
            "shared_decay_masks_match_control": True,
            "attention_decay_masks": required_masks}


def experiment_spec(config, *, entry_module: str):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared,
                   cpu_tests=(*shared.cpu_tests, "experiments/transformer_pp/tests/test_static_fox_acceptance.py"),
                   check_config=check_recipe, check_initialization=check_initialization)


def main(argv=None) -> None:
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()
