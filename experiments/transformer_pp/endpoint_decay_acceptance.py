"""Qualify endpoint-gate ablations against the pinned query-distance control."""
from dataclasses import replace
from functools import partial
import hashlib
from pathlib import Path

from scaffolds.transformer.acceptance import attention_spec, main as run_acceptance
from scaffolds.transformer.configs import Config, config_to_dict, load_config
from scaffolds.transformer.configs.base import validate_config

CONTROL = Path(__file__).resolve().parent / "runs/r3e26412b2811/config.yaml"
CONTROL_SHA256 = "3fa5cddfc7d5633c6f5919b30bda0a1e9eda145e04e891ee6ad9668f03210a26"
# The query-distance control shares all initial parameters with its FoX parent;
# this fingerprint was verified by query_distance_decay_acceptance at launch.
INITIALIZATION_SHA256 = "41222e5822311b7c7828965e5f4e03b72e820cc14f0928d6af073c8649a7b7d7"
PARAMETER_COUNT = 199585066
VARIANTS = (("query_key", "softplus"), ("query", "sigmoid"))


def control_config() -> Config:
    if hashlib.sha256(CONTROL.read_bytes()).hexdigest() != CONTROL_SHA256:
        raise ValueError("The pinned query-distance control recipe changed")
    return load_config(CONTROL)


def check_recipe(config: Config) -> None:
    validate_config(config)
    variant = (config.model.attention_forget_gate_conditioning,
               config.model.attention_forget_gate_rate_activation)
    if variant not in VARIANTS:
        raise ValueError("Select exactly one endpoint ablation: query-key/softplus or query/sigmoid")
    actual, expected = config_to_dict(config), config_to_dict(control_config())
    actual["model"]["attention_forget_gate_conditioning"] = "query"
    actual["model"]["attention_forget_gate_rate_activation"] = "softplus"
    for recipe in (actual, expected):
        for key in ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name", "wandb_group", "wandb_dir", "wandb_mode"):
            recipe["train"].pop(key, None)
    if actual != expected:
        differences = [f"{section}.{key}" for section in set(actual) | set(expected)
                       for key in set(actual.get(section, {})) | set(expected.get(section, {}))
                       if actual.get(section, {}).get(key) != expected.get(section, {}).get(key)]
        raise ValueError(f"Scientific settings differ from the query-distance control: {sorted(differences)}")


def _gate_initialization_diagnostics(params, declared, config: Config) -> dict:
    """Compare rates on reproducible RMS-normalized synthetic features, not LM activations."""
    import numpy as np

    sample = np.random.default_rng(0).standard_normal((64, config.model.hidden_size))
    sample /= np.sqrt(np.mean(sample * sample, axis=-1, keepdims=True))
    projected = np.einsum("td,lhd->tlh", sample, np.asarray(declared["forget_gate_kernel"], np.float64))
    reference_bias = np.asarray(declared["forget_gate_dt_bias"], np.float64)[..., 0]
    actual_bias = np.asarray(params["forget_gate_dt_bias"][...], np.float64)[..., 0]
    scale = np.exp(np.asarray(declared["forget_gate_log_rate"], np.float64)[..., 0])
    reference = scale * np.logaddexp(0.0, projected + reference_bias)
    bounded = config.model.attention_forget_gate_rate_activation == "sigmoid"
    transformed = (1.0 / (1.0 + np.exp(-(projected + actual_bias))) if bounded
                   else np.logaddexp(0.0, projected + actual_bias))
    actual = scale * transformed
    zero_reference = scale * np.logaddexp(0.0, reference_bias)
    zero_actual = scale * (1.0 / (1.0 + np.exp(-actual_bias)) if bounded
                           else np.logaddexp(0.0, actual_bias))
    return {"sample": "64 standard-normal vectors, RMS-normalized across hidden channels; NumPy seed 0",
            "sample_is_model_activation": False,
            "zero_input_max_relative_rate_error": float(np.max(np.abs(zero_actual / zero_reference - 1))),
            "rate_ratio_quantiles_05_50_95": np.quantile(actual / reference, [.05, .5, .95]).tolist(),
            "half_life_ratio_quantiles_05_50_95": np.quantile(reference / actual, [.05, .5, .95]).tolist(),
            "control_median_half_life_tokens": float(np.median(np.log(2.0) / reference)),
            "variant_median_half_life_tokens": float(np.median(np.log(2.0) / actual))}


def _check_initialized_parameters(params, config: Config) -> dict:
    """Validate the variant, then restore its tiny change to fingerprint the full control."""
    import jax
    import numpy as np
    from scaffolds.transformer.attention.parameters import (
        attention_parameter_specs, initial_attention_parameter_values, mechanism_key,
    )
    from scaffolds.transformer.training.checkpointing import state_sha256

    control = control_config()
    declared = initial_attention_parameter_values(
        attention_parameter_specs(control.model), num_layers=control.model.num_layers,
        num_heads=control.model.num_heads, key=mechanism_key(jax.random.PRNGKey(config.train.seed)))
    key_conditioned = config.model.attention_forget_gate_conditioning == "query_key"
    bounded = config.model.attention_forget_gate_rate_activation == "sigmoid"
    count = sum(leaf.size for leaf in jax.tree.leaves(params))
    extra_count = (config.model.hidden_size * config.model.num_layers * config.model.num_heads
                   if key_conditioned else 0)
    if count != PARAMETER_COUNT + extra_count:
        raise RuntimeError(f"Unexpected endpoint parameter count: {count}")
    actual_sha256 = state_sha256(params)
    for name, expected in declared.items():
        actual = np.asarray(params[name][...])
        if bounded and name == "forget_gate_dt_bias":
            # Independent zero-input identity: sigmoid(new bias) = softplus(old bias).
            zero_rate = 1.0 / (1.0 + np.exp(-actual.astype(np.float64)))
            expected_rate = np.logaddexp(0.0, np.asarray(expected, np.float64))
            if not np.allclose(zero_rate, expected_rate, rtol=2e-6, atol=1e-9):
                raise RuntimeError("Bounded gate does not match the control's initial zero-input rates")
        elif not np.array_equal(actual, np.asarray(expected)):
            raise RuntimeError(f"Initial parameter {name} differs from the query-distance control")
    rate_diagnostics = _gate_initialization_diagnostics(params, declared, config)
    if key_conditioned:
        key = np.asarray(params["forget_gate_key_kernel"][...])
        if key.shape != (config.model.num_layers, config.model.num_heads, config.model.hidden_size) or np.any(key != 0):
            raise RuntimeError("The endpoint key projection must initialize at zero")
        del params["forget_gate_key_kernel"]
    if bounded:
        params["forget_gate_dt_bias"] = params["forget_gate_dt_bias"].replace(value=declared["forget_gate_dt_bias"])
    recovered = state_sha256(params)
    if recovered != INITIALIZATION_SHA256:
        raise RuntimeError(f"Restored initialization differs from the query-distance control: {recovered}")
    return {"parameter_count": count, "parameter_sha256": actual_sha256,
            "control_run_id": "r3e26412b2811", "control_parameter_sha256": recovered,
            "control_parameter_count": PARAMETER_COUNT,
            "unchanged_parameters_match_control": True,
            "gate_initialization": ("zero_key_projection_exact_control" if key_conditioned
                                    else "sigmoid_bias_matches_softplus_zero_input_rate"),
            "initial_zero_input_decay_rates_match_control": True,
            "initial_gate_rate_comparison": rate_diagnostics}


def check_initialization(config: Config) -> dict:
    import jax
    from flax import nnx
    from scaffolds.transformer.models import build_model

    check_recipe(config)
    rng = jax.random.PRNGKey(config.train.seed)
    model = build_model(config.model, nnx.Rngs(params=rng, dropout=jax.random.fold_in(rng, 0)))
    return _check_initialized_parameters(nnx.state(model, nnx.Param), config)


def experiment_spec(config: Config, *, entry_module: str):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, cpu_tests=(*shared.cpu_tests,
                   "experiments/transformer_pp/tests/test_endpoint_decay_acceptance.py"),
                   check_config=check_recipe, check_initialization=check_initialization)


def main(argv=None) -> None:
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()
