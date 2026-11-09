"""Qualify the per-order GDN coefficient-gate run against the pinned ungated query + W decay control."""
from dataclasses import replace
from functools import partial
import hashlib
from pathlib import Path

from scaffolds.transformer.acceptance import (
    attention_spec, gdn_gate_start_parity, main as run_acceptance, matches_softmax_backbone_initialization,
)
from scaffolds.transformer.attention.input_dependent import GDN_GATE_PARAMETERS
from scaffolds.transformer.configs import Config, config_to_dict, load_config
from scaffolds.transformer.configs.base import validate_config

CONTROL_RUN = "rca97cc9db8f2"
CONTROL = Path(__file__).resolve().parent / f"runs/{CONTROL_RUN}/config.yaml"
CONTROL_SHA256 = "75fa1e3e30eaf98aa49d5b4ca833860a2664877e5892ecca11c008eff1978f58"
# The control's seed-42 initial parameters (199,396,486), identical when built at its launch
# commit 9c39f921 and at this branch's base.
CONTROL_INITIALIZATION_SHA256 = "a236d07dea53ba8c03ad013b6071d6259e19be1d4c473c0e3d86497d2cae30cf"
# rca97cc9db8f2 reads Kennel's manifests; another cluster's copies must hash identically.
MANIFEST_SHA256 = {
    "train_manifest": "9de775ed836ab7c61857207335a981c09630caa3226d2dc0ee0b799cdcb2f4f4",
    "eval_manifest": "0e8615538d9dccad17956d8d9839394d05a56f8ec210dc67716db829a543634f",
}
# The control's recipe was written under the study's earlier folder name, so its execution
# paths and W&B group differ from a recipe allocated now.
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name", "wandb_group", "wandb_dir",
                  "wandb_mode")
LOGIT_TOKENS = 64


def control_config() -> Config:
    if hashlib.sha256(CONTROL.read_bytes()).hexdigest() != CONTROL_SHA256:
        raise ValueError("The pinned ungated query + W decay control recipe changed")
    return load_config(CONTROL)


def check_recipe(config: Config) -> None:
    """Only attention_coefficient_gate: gdn differs from rca97cc9db8f2."""
    validate_config(config)
    if config.model.attention_coefficient_gate != "gdn":
        raise ValueError("This run uses the GDN coefficient gate")
    actual, expected = config_to_dict(config), config_to_dict(control_config())
    actual["model"]["attention_coefficient_gate"] = expected["model"]["attention_coefficient_gate"]
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
    """The shared softmax-backbone check; every other parameter equal to the control's start; each
    gate array one draw per layer and head repeated over the orders; and initial logits within
    the RMSNorm epsilon bound of the ungated recipe's (scaffolds.transformer.acceptance)."""
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
              for gate in ("gdn", "none")]
    gated, ungated = (nnx.state(model, nnx.Param) for model in models)
    model = config.model
    layers, heads, count = model.num_layers, model.num_heads, model.attention_degree + 1
    kernel, log_rate, dt_bias = GDN_GATE_PARAMETERS
    shapes = {kernel: (layers, heads, model.hidden_size, count), log_rate: (layers, heads, count),
              dt_bias: (layers, heads, count)}
    gate_count = 0
    for name in GDN_GATE_PARAMETERS:
        values = np.asarray(gated[name][...])
        if (values.shape != shapes[name] or values.dtype != np.float32 or not np.isfinite(values).all()
                or not np.array_equal(values, np.broadcast_to(values[..., :1], values.shape))):
            raise RuntimeError(f"{name} must start as one finite FP32 draw per layer and head, repeated over "
                               f"the orders, with shape {shapes[name]}")
        gate_count += values.size
        del gated[name]
    shared_sha256, ungated_sha256 = state_sha256(gated), state_sha256(ungated)
    if shared_sha256 != ungated_sha256:
        raise RuntimeError("Initial parameters other than the gates differ from the ungated recipe")
    if ungated_sha256 != CONTROL_INITIALIZATION_SHA256:
        raise RuntimeError(f"Initial parameters other than the gates differ from {CONTROL_RUN}'s start")
    length = min(LOGIT_TOKENS, model.max_seq_len)
    sequence = jnp.arange(length, dtype=jnp.int32).reshape(1, length) % model.vocab_size
    parity = gdn_gate_start_parity(config, sequence)
    return {**facts, "ungated_parity": {
        "ungated_reference": "this recipe with attention_coefficient_gate: none, built at the same seed and source",
        "recipe_pinned_to_run_id": CONTROL_RUN, "pinned_recipe_sha256": CONTROL_SHA256,
        "ungated_parameter_sha256": ungated_sha256,
        "ungated_parameter_count": sum(leaf.size for leaf in jax.tree.leaves(ungated)),
        "control_initialization_sha256": CONTROL_INITIALIZATION_SHA256,
        "gate_parameter_count": gate_count, "gate_parameters_repeat_one_draw_over_orders": True,
        "shared_parameters_match_ungated_recipe": True,
        "initial_logits": {**parity, "sequence": f"tokens 0..{length - 1}, one sequence"},
    }}


def experiment_spec(config: Config, *, entry_module: str):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, cpu_tests=(*shared.cpu_tests,
                   "experiments/transformer_pp/tests/test_coefficient_gdn_gate_acceptance.py"),
                   check_config=check_recipe, check_initialization=check_initialization)


def main(argv=None) -> None:
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()
