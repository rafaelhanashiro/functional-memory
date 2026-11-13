"""Qualify the score-transform runs against their pinned controls.

Each run is one control's recipe with ``model.attention_score_transform`` and gradient-event capture under the
default policy: every order of the polynomial reads z = phi(s) in place of the score s
(scaffolds/transformer/attention/score_transform.py). The controls are the degree-eight tanh query polynomial with
RoPE and coefficient-W decay at the 12-layer width-640 shape (``r95a5691d293f``) and at the reference shape
(``r25fffc16c2f3``); the recipe's depth and width select its control, and a recipe that matches neither is
refused. A softplus run starts its constant coefficient at zero (``model.attention_coefficient_constant_initial:
0``), so its kernel starts at sum_(r=1..8) z^r/r!, the Taylor polynomial of e^z - 1 = e^s; an abs run keeps the
control's Taylor start. No parameter is added and nothing is drawn differently: every initial array is the
control's, except b_0 = 0 in place of the control's stored 1 in a softplus run.

Both controls trained on B300's Mosaic query-polynomial kernels. The 12-layer width-640 runs train on the blocked
backend's causal strips (plain JAX, on B200, B300 or H100), the one execution setting beyond locations and labels
that differs from their control's. The reference-shape runs keep their control's attention settings and train on
the Mosaic transformed query-polynomial kernels (the control's kernels with phi(s) evaluated in their tiles), on
the GPU types those kernels are qualified on.
"""
import argparse
from dataclasses import replace
from functools import partial
import hashlib
from pathlib import Path
import sys

from scaffolds.transformer.acceptance import (
    attention_spec, main as run_acceptance, matches_softmax_backbone_initialization,
)
from scaffolds.transformer.configs import Config, config_to_dict, load_config
from scaffolds.transformer.configs.base import validate_config

RUNS = Path(__file__).resolve().parent / "runs"
# Each control: run ID, config SHA256, and its initial parameters' SHA256 and count at seed 42, computed on CPU
# with acceptance.initialization_facts on the control recipe (the count is also the one its completion holds).
CONTROLS = {
    "l12_d640": ("r95a5691d293f", "e27c5364bfb1a47c6fde45598a3c31da8ad6c8c4402d29b873d76b0a7571f458",
                 "72d75524ec107d4acdeb4f77d8cdea838d4c57d54f79aa9c924da4a154b03667", 99_678_056),
    "reference": ("r25fffc16c2f3", "ffa72e4d99f016bedfd07b7b858199734d71596b3af18a325813bd810e3620b2",
                  "bacacd7eeee7c67dd4d5deb55ebfeef925ff4631a188d50d2e45524b6618c74d", 200_150_806),
}
# The recipe settings that select each control: depth and width.
SELECTING_FIELDS = ("model.num_layers", "model.hidden_size")
CONTROL_SHAPES = {"l12_d640": (12, 640), "reference": (15, 896)}
# Each transform and the constant-coefficient start its runs require (null: the control's Taylor start).
STARTS = {"softplus": 0.0, "abs": None}
# Gradient-event capture, on with the default rule and retention cap (configs/base.py TrainConfig): after
# warmup, a finite pre-clip norm above max(1, 5 x the median of the previous 100 norms, once 20 exist)
# saves the pre-update state; at most three such states are kept.
CAPTURE_POLICY = {"gradient_event_capture": True, "gradient_event_factor": 5.0, "gradient_event_min_norm": 1.0,
                  "gradient_event_history": 100, "gradient_event_min_history": 20, "gradient_event_max_states": 3}
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name", "wandb_group", "wandb_dir",
                  "wandb_mode")
TRANSFORM_KEYS = ("attention_score_transform", "attention_coefficient_constant_initial")
# The attention implementation each control's runs require. The 12-layer width-640 runs train on the blocked
# strips, where their control trained on Mosaic (scan, inert there); the reference-shape runs keep their
# control's Mosaic setting (the blocked field inert at its default).
STRIPS = {"attention_backend": "blocked", "attention_blocked_implementation": "strips"}
MOSAIC = {"attention_backend": "mosaic", "attention_blocked_implementation": "scan"}
BACKENDS = {"l12_d640": STRIPS, "reference": MOSAIC}
TRAINS_ON = {"blocked": "the blocked backend's causal strips",
             "mosaic": "the Mosaic transformed query-polynomial kernels"}
# The GPU types each control's runs train on: the strips are plain JAX and their GPU tests ran on each of these;
# the transformed Mosaic kernels' GPU tests and pilots ran on these (scaffolds/transformer/validation).
GPU_TYPES = {"l12_d640": ("B200", "B300", "H100"), "reference": ("H100",)}
GPU_REASONS = {"l12_d640": "the causal strips' GPU tests run there",
               "reference": "the Mosaic transformed query-polynomial kernels are qualified there"}
BIAS = "query_polynomial_bias"
OWN_TESTS = "experiments/transformer_pp/tests/test_score_transform_acceptance.py"


def control_name(config: Config) -> str:
    """The control at the recipe's depth and width; refuse a recipe that matches neither."""
    selected = (config.model.num_layers, config.model.hidden_size)
    for name, shape in CONTROL_SHAPES.items():
        if shape == selected:
            return name
    controls = "; ".join(f"{CONTROLS[name][0]} {list(shape)}" for name, shape in CONTROL_SHAPES.items())
    raise ValueError(f"No score-transform control has this recipe's {list(SELECTING_FIELDS)} = {list(selected)}; "
                     f"the controls have {controls}")


def control_config(name: str) -> Config:
    if name not in CONTROLS:
        raise ValueError(f"No score-transform control {name!r}")
    run, digest = CONTROLS[name][:2]
    path = RUNS / run / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"The pinned control recipe {run} changed")
    return load_config(path)


def check_recipe(config: Config) -> None:
    """Only the transform, a softplus run's constant start, gradient-event capture, execution locations and labels,
    and (against the 12-layer width-640 control's Mosaic kernels) the blocked strips differ from the control."""
    validate_config(config)
    model = config.model
    transform = model.attention_score_transform
    if transform not in STARTS:
        raise ValueError(f"This experiment requires a score transform ({', '.join(STARTS)}), not {transform!r}")
    start = model.attention_coefficient_constant_initial
    if start != STARTS[transform] or (start is None) != (STARTS[transform] is None):
        raise ValueError(f"The {transform} run starts its constant coefficient at "
                         f"attention_coefficient_constant_initial: {STARTS[transform]}, not {start}")
    name = control_name(config)
    run, required = CONTROLS[name][0], BACKENDS[name]
    backend = {key: getattr(model, key) for key in required}
    if backend != required:
        raise ValueError(f"Score transforms against control {run} train on {TRAINS_ON[required['attention_backend']]}: "
                         f"the recipe must set {required}, not {backend}")
    policy = {key: getattr(config.train, key) for key in CAPTURE_POLICY}
    if policy != CAPTURE_POLICY:
        raise ValueError(f"Gradient-event capture must be on with the default policy {CAPTURE_POLICY}, not {policy}")
    actual, expected = config_to_dict(config), config_to_dict(control_config(name))
    actual["model"].update({key: expected["model"][key] for key in (*TRANSFORM_KEYS, *required)})
    for recipe in (actual, expected):
        for key in (*EXECUTION_KEYS, *CAPTURE_POLICY):
            recipe["train"].pop(key, None)
    if actual != expected:
        differences = [f"{section}.{key}" for section in set(actual) | set(expected)
                       for key in set(actual.get(section, {})) | set(expected.get(section, {}))
                       if actual.get(section, {}).get(key) != expected.get(section, {}).get(key)]
        raise ValueError(f"Scientific settings differ from control {run}: {sorted(differences)}")


def initial_parameters(config: Config):
    """The parameters training starts from, derived as acceptance.initialization_facts derives them."""
    import jax
    from flax import nnx
    from scaffolds.transformer.models import build_model

    rng = jax.random.PRNGKey(config.train.seed)  # Same derivation as training/state.py create_state.
    # Initialization does not depend on the attention backend; built on the blocked backend, as
    # acceptance's initialization check builds every recipe.
    model = build_model(replace(config.model, attention_backend="blocked"),
                        nnx.Rngs(params=rng, dropout=jax.random.fold_in(rng, 0)))
    return nnx.state(model, nnx.Param)


def control_bias_start(name: str) -> float:
    """A control's stored b_0 start (Taylor's 1/0! under its factorial storage scale), from its declared
    parameters."""
    from scaffolds.transformer.attention.parameters import attention_parameter_specs

    return next(spec.initial[0] for spec in attention_parameter_specs(control_config(name).model) if spec.name == BIAS)


def as_control(params, config: Config, name: str | None = None) -> dict:
    """Count every initial parameter; for a softplus run require b_0 at exactly its FP32 start in every layer and
    head and restore the start of control ``name`` (the recipe's own by default) there; then fingerprint: the
    control's initialization if nothing else differs."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scaffolds.transformer.training.checkpointing import state_sha256

    count = sum(leaf.size for leaf in jax.tree.leaves(params))
    start = STARTS[config.model.attention_score_transform]
    if start is not None:
        bias = np.asarray(params[BIAS][...])
        if (bias.dtype != np.float32 or bias.ndim != 3
                or bias[..., 0].tobytes() != np.full(bias.shape[:-1], start, np.float32).tobytes()):
            raise RuntimeError(f"{BIAS}'s order 0 must start at exactly {start} in FP32 in every layer and head, not "
                               f"{bias.dtype} {np.unique(bias[..., :1]).tolist()[:4]}")
        restored = bias.copy()
        restored[..., 0] = control_bias_start(control_name(config) if name is None else name)
        params[BIAS] = params[BIAS].replace(value=jnp.asarray(restored))
    return {"parameter_count": count, "control_parameter_sha256": state_sha256(params)}


def check_initialization(config: Config) -> dict:
    """Every initial parameter equals the control's at the same seed (its pinned fingerprint and count), except
    b_0 = 0 in a softplus run."""
    check_recipe(config)
    run, _, digest, count = CONTROLS[control_name(config)]
    facts = as_control(initial_parameters(config), config)
    if facts != {"parameter_count": count, "control_parameter_sha256": digest}:
        raise RuntimeError(f"Initialization differs from control {run} beyond the constant start: {facts}")
    transform = config.model.attention_score_transform
    return {**matches_softmax_backbone_initialization(config),
            "control_run_id": run, "control_parameter_sha256": digest, "control_parameter_count": count,
            "score_transform": transform, "constant_coefficient_start": STARTS[transform],
            "all_other_initial_parameters_match_control": True}


def gpu_types(config: Config) -> tuple[str, ...]:
    """The GPU types the recipe trains on (its control's runs')."""
    return GPU_TYPES[control_name(config)]


def experiment_spec(config: Config, *, entry_module: str):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    # The pilot must resolve the recipe's own backend: the blocked strips or the Mosaic transformed kernels. The
    # default allocator share suffices.
    return replace(shared, cpu_tests=(*shared.cpu_tests, OWN_TESTS), check_config=check_recipe,
                   required_backend=BACKENDS[control_name(config)]["attention_backend"],
                   check_initialization=check_initialization)


def require_gpu_type(argv) -> None:
    """Refuse an allocation of a GPU type the recipe does not train on before acceptance starts. The launcher's
    --runtime-env query allocates nothing and names no GPU type."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("config", nargs="?", type=Path)
    parser.add_argument("--gpu-type", default="H100")  # acceptance's own default
    parser.add_argument("--runtime-env", action="store_true")
    args = parser.parse_known_args(argv)[0]
    if args.runtime_env:
        return
    if args.config is None:  # the launcher and acceptance's workers always pass the recipe first
        raise ValueError("The GPU type check needs the run's recipe, the first argument")
    name = control_name(load_config(args.config))
    allowed, reason = GPU_TYPES[name], GPU_REASONS[name]
    if args.gpu_type.upper() not in allowed:
        raise ValueError(f"This recipe trains only on {', '.join(allowed)} ({reason}), not on {args.gpu_type}. "
                         f"Submit it with GPU types {','.join(allowed)}.")


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    require_gpu_type(argv)
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()
