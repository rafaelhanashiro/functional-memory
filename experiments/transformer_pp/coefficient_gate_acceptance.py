"""Qualify per-order coefficient-gate runs against the pinned ungated query + W decay control."""
from dataclasses import replace
from functools import partial
import hashlib
from pathlib import Path

from scaffolds.transformer.acceptance import (
    attention_spec, main as run_acceptance, matches_softmax_backbone_initialization,
)
from scaffolds.transformer.configs import Config, config_to_dict, load_config
from scaffolds.transformer.configs.base import validate_config

CONTROL = Path(__file__).resolve().parent / "runs/r25fffc16c2f3/config.yaml"
CONTROL_SHA256 = "ffa72e4d99f016bedfd07b7b858199734d71596b3af18a325813bd810e3620b2"
# r25fffc16c2f3 reads Kennel's manifests; another cluster's copies must hash identically.
MANIFEST_SHA256 = {
    "train_manifest": "9de775ed836ab7c61857207335a981c09630caa3226d2dc0ee0b799cdcb2f4f4",
    "eval_manifest": "0e8615538d9dccad17956d8d9839394d05a56f8ec210dc67716db829a543634f",
}
DEGREES = (8, 10)
GATE_NAMES = ("query_polynomial_gate_kernel", "query_polynomial_gate_bias")
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name", "wandb_group", "wandb_dir",
                  "wandb_mode")


def control_config() -> Config:
    if hashlib.sha256(CONTROL.read_bytes()).hexdigest() != CONTROL_SHA256:
        raise ValueError("The pinned ungated query + W decay control recipe changed")
    return load_config(CONTROL)


def check_recipe(config: Config) -> None:
    """Only the sigmoid coefficient gate and, for the second run, degree 10 differ from r25fffc16c2f3."""
    validate_config(config)
    if config.model.attention_coefficient_gate != "sigmoid" or config.model.attention_degree not in DEGREES:
        raise ValueError("Coefficient-gate runs use the sigmoid gate at degree 8 or 10")
    actual, expected = config_to_dict(config), config_to_dict(control_config())
    actual["model"]["attention_coefficient_gate"] = expected["model"]["attention_coefficient_gate"]
    actual["model"]["attention_degree"] = expected["model"]["attention_degree"]
    for key, digest in MANIFEST_SHA256.items():
        path, control_path = actual["data"].pop(key), expected["data"].pop(key)
        if path != control_path and hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"data.{key} is not a copy of the control's dataset manifest")
    for recipe in (actual, expected):
        for key in EXECUTION_KEYS:
            recipe["train"].pop(key, None)
    if actual != expected:
        differences = [f"{section}.{key}" for section in set(actual) | set(expected)
                       for key in set(actual.get(section, {})) | set(expected.get(section, {}))
                       if actual.get(section, {}).get(key) != expected.get(section, {}).get(key)]
        raise ValueError(f"Scientific settings differ from the ungated control: {sorted(differences)}")


def check_initialization(config: Config) -> dict:
    """The shared softmax-backbone check, then exact parity with the ungated recipe at the same seed."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from scaffolds.transformer.models import build_model
    from scaffolds.transformer.training.checkpointing import state_sha256

    check_recipe(config)
    facts = matches_softmax_backbone_initialization(config)
    rng = jax.random.PRNGKey(config.train.seed)
    # The blocked reference evaluates the same function on any device; the parameters do not depend on it.
    models = [build_model(replace(config.model, attention_backend="blocked", attention_coefficient_gate=gate),
                          nnx.Rngs(params=rng, dropout=jax.random.fold_in(rng, 0)))
              for gate in ("sigmoid", "none")]
    gated, ungated = (nnx.state(model, nnx.Param) for model in models)
    model = config.model
    shapes = {"query_polynomial_gate_kernel": (model.num_layers, model.num_heads, model.hidden_size,
                                               model.attention_degree + 1),
              "query_polynomial_gate_bias": (model.num_layers, model.num_heads, model.attention_degree + 1)}
    gate_count = 0
    for name in GATE_NAMES:
        values = np.asarray(gated[name][...])
        if values.shape != shapes[name] or values.dtype != np.float32 or np.any(values != 0):
            raise RuntimeError(f"{name} must start at exact FP32 zeros with shape {shapes[name]}")
        gate_count += values.size
        del gated[name]
    shared_sha256, ungated_sha256 = state_sha256(gated), state_sha256(ungated)
    if shared_sha256 != ungated_sha256:
        raise RuntimeError("Initial parameters other than the gates differ from the ungated recipe")
    length = min(64, model.max_seq_len)
    tokens = jnp.arange(length, dtype=jnp.int32).reshape(1, length) % model.vocab_size
    if not np.array_equal(np.asarray(models[0](tokens)), np.asarray(models[1](tokens))):
        raise RuntimeError("The zero-initialized gates change the initial function")
    ungated_count = sum(leaf.size for leaf in jax.tree.leaves(ungated))
    return {**facts, "ungated_parity": {
        "ungated_reference": "this recipe with attention_coefficient_gate: none, built at the same seed and source",
        "recipe_pinned_to_run_id": "r25fffc16c2f3", "pinned_recipe_sha256": CONTROL_SHA256,
        "ungated_parameter_sha256": ungated_sha256, "ungated_parameter_count": ungated_count,
        "gate_parameter_count": gate_count, "gate_parameters_exact_zero": True,
        "shared_parameters_match_ungated_recipe": True,
        "initial_logits_match_ungated_recipe": f"{length} tokens, blocked reference, bitwise",
    }}


def experiment_spec(config: Config, *, entry_module: str):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, cpu_tests=(*shared.cpu_tests,
                   "experiments/transformer_pp/tests/test_coefficient_gate_acceptance.py"),
                   check_config=check_recipe, check_initialization=check_initialization)


def main(argv=None) -> None:
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()
