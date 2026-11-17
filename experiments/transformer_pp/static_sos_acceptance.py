"""Static SoS counterparts of the six pinned query SoS recipes, with relative-Taylor factors."""
from dataclasses import replace
from functools import partial
import argparse
import hashlib
from pathlib import Path
import sys

from scaffolds.transformer.acceptance import attention_spec, main as run_acceptance
from scaffolds.transformer.configs import config_to_dict, load_config, validate_config

RUNS = Path(__file__).resolve().parent / "runs"
CONTROLS = {
    (8, "identity", 0): ("reeefbcc19abf", "7d3fca30ff8fb22520f5bdb6df9bffc3a962b2dc5ab539436a17b4d2b0e3d326"),
    (12, "identity", 0): ("r3d52b7ab43d1", "9b76b2bf414442b851519a08f5bcf193598d96381f055bb1de31dbd238f6c0fa"),
    (8, "softplus", 0): ("r41a4c74c5f58", "851c389a0e00d5e595c5c02b30e09255249412fc3d4f9d11a5ba0551e70a7110"),
    (12, "softplus", 0): ("rc5aac8f37a2e", "4801b17dd1823c665595e42b0c7bd8dcf03385d6b6ca4de6a27fa7419afb7944"),
    (8, "softplus", 1): ("r841f35a24241", "1643696258e0267ecba603c8e11cbfe9228cfcc43d9a9af91341d9b9a094a62b"),
    (12, "softplus", 1): ("rf637de9b65b1", "22a94ee8cc46fa053cfe2d7789567552888882038d0eb666c705a2762a762903"),
}
STATIC_CHANGES = dict(
    attention_coefficient_conditioning="static", attention_coefficient_activation="identity",
    attention_coefficient_kernel_weight_decay=False, attention_coefficient_projection_initial_std=0.,
    attention_coefficient_amplitude_initial=1., attention_coefficient_tanh_bias=False,
    attention_coefficient_gate="none", attention_coefficient_readout="diagonal",
    attention_coefficient_feature_count=None, attention_coefficient_readout_scale=1.,
    attention_coefficient_input_normalization="none", attention_coefficient_input_source="post_rmsnorm",
    attention_coefficient_amplitude_scale="none", attention_coefficient_scale="none",
    attention_static_sos_parameterization="relative_taylor",
)
PATH_FIELDS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name",
               "wandb_group", "wandb_dir", "wandb_mode")
CAPTURE_POLICY = dict(gradient_event_capture=True, gradient_event_factor=5.0,
                      gradient_event_min_norm=1.0, gradient_event_history=100,
                      gradient_event_min_history=20, gradient_event_max_states=3)
CPU_TESTS = ("scaffolds/transformer/tests/test_static_sos.py",
             "experiments/transformer_pp/tests/test_static_sos_acceptance.py")
GPU_TESTS = ("scaffolds/transformer/tests/test_mosaic_static_sos.py",)


def control_config(core_degree, transform, power):
    descriptor = (core_degree, transform, power)
    if descriptor not in CONTROLS:
        raise ValueError("Static SoS requires core degree 8 or 12 and raw, softplus, or softplus-prefactor weights")
    run_id, digest = CONTROLS[descriptor]
    path = RUNS / run_id / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Pinned query SoS control {run_id} changed")
    return load_config(path)


def static_config(core_degree, transform, power):
    config = control_config(core_degree, transform, power)
    config.model = replace(config.model, **STATIC_CHANGES)
    config.train = replace(config.train, **CAPTURE_POLICY)
    return config


def check_recipe(config):
    validate_config(config)
    core = config.model.attention_degree - config.model.attention_coefficient_input_power
    expected = config_to_dict(static_config(core, config.model.attention_score_transform,
                                           config.model.attention_coefficient_input_power))
    actual = config_to_dict(config)
    for recipe in (expected, actual):
        for key in PATH_FIELDS:
            recipe["train"].pop(key, None)
    if actual != expected:
        differences = [f"{section}.{key}" for section in set(actual) | set(expected)
                       for key in set(actual.get(section, {})) | set(expected.get(section, {}))
                       if actual.get(section, {}).get(key) != expected.get(section, {}).get(key)]
        raise ValueError(f"Static SoS differs from its pinned query partner beyond conditioning: {sorted(differences)}")


def _initial_parameter_check(config, control):
    """Actual initialized arrays, including the optimizer's decay mask, not just declared shapes."""
    import jax
    import numpy as np
    from flax import nnx
    from scaffolds.transformer.attention.nonnegative_polynomial import taylor_square_factors
    from scaffolds.transformer.attention.parameters import attention_parameter_specs
    from scaffolds.transformer.models import build_model, model_parameter_count
    from scaffolds.transformer.training.optim import attention_weight_decay_mask

    def flat(current):
        model = build_model(replace(current.model, attention_backend="blocked"), nnx.Rngs(
            params=jax.random.PRNGKey(current.train.seed),
            dropout=jax.random.fold_in(jax.random.PRNGKey(current.train.seed), 0)))
        return {path: np.asarray(value[...]) for path, value in nnx.to_flat_state(nnx.state(model, nnx.Param))}

    actual, query = flat(config), flat(control)
    expected_removed = {(name,) for name in ("query_polynomial_bias", "query_polynomial_amplitude", "query_polynomial_kernel")}
    if set(query) - set(actual) != expected_removed or set(actual) - set(query) != {("polynomial_square_factors",)}:
        raise RuntimeError("Static SoS must replace exactly query W, a, b with one static factor array")
    common = set(actual) & set(query)
    for path in common:
        if not np.array_equal(actual[path], query[path]):
            raise RuntimeError(f"Static SoS backbone differs from query partner at {path}")
    core = config.model.attention_degree - config.model.attention_coefficient_input_power
    stored = actual[("polynomial_square_factors",)]
    expected_shape = (config.model.num_layers, config.model.num_heads, core + 1)
    if stored.shape != expected_shape or not np.all(stored == 1):
        raise RuntimeError("Relative-Taylor static factors must store b=1 per layer/head")
    if not np.array_equal(stored, query[("query_polynomial_bias",)]) or not np.all(query[("query_polynomial_amplitude",)] == 0):
        raise RuntimeError("Static b must equal the query bias with zero query amplitude")
    fixed = np.asarray(taylor_square_factors(core), np.float32)
    actual_factors = fixed * (1 + stored) / 2
    query_factors = fixed * (1 + query[("query_polynomial_bias",)]) / 2
    if not np.array_equal(actual_factors, query_factors):
        raise RuntimeError("Initial effective square factors differ from query partner")
    count = sum(value.size for value in actual.values())
    if count != model_parameter_count(config.model):
        raise RuntimeError("Static SoS initialized parameter count differs from model formula")
    specs = attention_parameter_specs(config.model)
    mask = attention_weight_decay_mask({name: jax.numpy.ones((2, 2)) for name in (s.name for s in specs)},
                                       excluded_names={s.name for s in specs if not s.weight_decay})
    if mask["polynomial_square_factors"]:
        raise RuntimeError("Static SoS factors must be exempt from weight decay")
    digest = hashlib.sha256()
    for path in sorted(common, key=str):
        digest.update(str(path).encode())
        digest.update(actual[path].tobytes())
    return dict(parameter_count=count, static_factor_shape=list(stored.shape), static_stored_value=1.,
                factor_mapping="F_j = f_j (1 + b_j) / 2", factor_weight_decay=False,
                backbone_sha256=digest.hexdigest(), all_backbone_parameters_match_control=True,
                effective_factors_match_control=True)


def check_initialization(config):
    check_recipe(config)
    core = config.model.attention_degree - config.model.attention_coefficient_input_power
    key = (core, config.model.attention_score_transform, config.model.attention_coefficient_input_power)
    facts = _initial_parameter_check(config, control_config(*key))
    return {**facts, "control_run_id": CONTROLS[key][0], "sos_core_degree": core,
            "polynomial_degree": config.model.attention_degree, "score_transform": key[1], "input_power": key[2]}


def experiment_spec(config, *, entry_module):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, cpu_tests=(*shared.cpu_tests, *CPU_TESTS),
                   gpu_tests=(*shared.gpu_tests, *GPU_TESTS), required_backend="mosaic",
                   check_config=check_recipe, check_initialization=check_initialization)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu-type", default="H100")
    parser.add_argument("--runtime-env", action="store_true")
    args = parser.parse_known_args(argv)[0]
    if not args.runtime_env and args.gpu_type.upper() not in ("H100", "B200"):
        raise ValueError("These static SoS runs qualify and train on H100 or B200 only")
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()
