"""Qualify sigmoid, softplus and swish attention against the pinned exponential control.

The fixed degree-one polynomial [0, 1] returns its transformed score unchanged (swish, s sigmoid(s), is signed).
It adds no parameters: the complete initialized model and AdamW decay mask must
equal the exponential run's. Production uses Mosaic, qualified in the allocation
before the full-recipe pilot and exact fresh-process resume.
"""
import argparse
from dataclasses import replace
from functools import partial
import hashlib
from pathlib import Path
import sys

from scaffolds.transformer.acceptance import attention_spec, main as run_acceptance
from scaffolds.transformer.configs import Config, config_to_dict, load_config
from scaffolds.transformer.configs.base import validate_config

RUNS = Path(__file__).resolve().parent / "runs"
CONTROL_RUN = "r6a76e7fbde45"
CONTROL_CONFIG_SHA256 = "599dc40917e12a8308ae41fcb39623e1871e43fbd6d35402692d25f02450eee6"
# Computed from the pinned recipe on CPU with the same seed/RNG derivation as training.
CONTROL_INITIALIZATION = {
    "parameter_count": 198_453_586,
    "parameter_sha256": "637cb8beb5ce9275aa2b312964a68bc519efdeb4169bc7025ee914b38662f0b4",
    "weight_decay_mask_sha256": "d10a59d00d4f4b78b1bf8e994b91f0d5d4678a19d02b221227085932d5c8f23e",
}
TRANSFORMS = ("sigmoid", "softplus", "swish")
MODEL_CHANGES = {"attention_backend": "mosaic", "attention_degree": 1,
                 "attention_fixed_coefficients": (0.0, 1.0)}
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name",
                  "wandb_group", "wandb_dir", "wandb_mode")
OWN_TESTS = "experiments/transformer_pp/tests/test_pointwise_attention_acceptance.py"
INTEGRATION_TESTS = "scaffolds/transformer/tests/test_pointwise_attention.py"
EMBEDDING_TESTS = "scaffolds/transformer/tests/test_embedding_precision.py"
KERNEL_TESTS = "scaffolds/transformer/tests/test_pointwise_score_transform.py"
GPU_TYPES = ("H100", "B200")


def control_config() -> Config:
    path = RUNS / CONTROL_RUN / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != CONTROL_CONFIG_SHA256:
        raise ValueError(f"The pinned exponential control recipe {CONTROL_RUN} changed")
    return load_config(path)


def check_recipe(config: Config) -> None:
    """Only the pointwise weight function and its backend differ scientifically."""
    validate_config(config)
    if config.model.attention_score_transform not in TRANSFORMS:
        raise ValueError(f"Pointwise baselines require one of {TRANSFORMS}")
    for key, value in MODEL_CHANGES.items():
        actual = getattr(config.model, key)
        if key == "attention_fixed_coefficients":
            actual = tuple(actual) if actual is not None else None
        if actual != value:
            raise ValueError(f"Pointwise baselines require model.{key}: {value}")
    actual, expected = config_to_dict(config), config_to_dict(control_config())
    for key in (*MODEL_CHANGES, "attention_score_transform"):
        actual["model"][key] = expected["model"][key]
    for recipe in (actual, expected):
        for key in EXECUTION_KEYS:
            recipe["train"].pop(key, None)
    if actual != expected:
        differences = sorted(f"{section}.{key}" for section in set(actual) | set(expected)
                             for key in set(actual.get(section, {})) | set(expected.get(section, {}))
                             if actual.get(section, {}).get(key) != expected.get(section, {}).get(key))
        raise ValueError(f"Scientific settings differ from exponential control {CONTROL_RUN}: {differences}")


def parameter_facts(config: Config) -> dict:
    """Fingerprint every actual initial array and the optimizer's actual decay mask."""
    import jax
    from flax import nnx
    from scaffolds.transformer.attention.parameters import attention_parameter_specs
    from scaffolds.transformer.models import build_model
    from scaffolds.transformer.training.checkpointing import state_sha256
    from scaffolds.transformer.training.optim import attention_weight_decay_mask

    rng = jax.random.PRNGKey(config.train.seed)
    model = build_model(replace(config.model, attention_backend="blocked"),
                        nnx.Rngs(params=rng, dropout=jax.random.fold_in(rng, 0)))
    params = nnx.state(model, nnx.Param)
    excluded = frozenset(spec.name for spec in attention_parameter_specs(config.model) if not spec.weight_decay)
    mask = attention_weight_decay_mask(params, excluded_names=excluded)
    return {"parameter_count": sum(leaf.size for leaf in jax.tree.leaves(params)),
            "parameter_sha256": state_sha256(params), "weight_decay_mask_sha256": state_sha256(mask)}


def check_initialization(config: Config) -> dict:
    check_recipe(config)
    facts = parameter_facts(config)
    if facts != CONTROL_INITIALIZATION:
        raise RuntimeError(f"Initial parameters or decay mask differ from exponential control {CONTROL_RUN}: {facts}")
    return {"control_run_id": CONTROL_RUN, **facts,
            "all_initial_parameters_match_control": True, "weight_decay_mask_matches_control": True}


def experiment_spec(config: Config, *, entry_module: str):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, cpu_tests=(*shared.cpu_tests, OWN_TESTS, INTEGRATION_TESTS, KERNEL_TESTS, EMBEDDING_TESTS),
                   gpu_tests=(*shared.gpu_tests, INTEGRATION_TESTS, KERNEL_TESTS, EMBEDDING_TESTS),
                   check_config=check_recipe,
                   required_backend="mosaic", check_initialization=check_initialization)


def require_gpu_type(argv) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu-type", default="H100")
    parser.add_argument("--runtime-env", action="store_true")
    args = parser.parse_known_args(argv)[0]
    if not args.runtime_env and args.gpu_type.upper() not in GPU_TYPES:
        raise ValueError(f"Pointwise Mosaic attention qualifies only on {', '.join(GPU_TYPES)}, not {args.gpu_type}")


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    require_gpu_type(argv)
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()
