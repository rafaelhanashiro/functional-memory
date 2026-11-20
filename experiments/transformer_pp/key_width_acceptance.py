"""Qualify the narrow-key runs: the softplus-SoS input-power polynomial and the exponential at Q/K widths 4, 8, 16.

Each run is a completed partner's recipe with ``model.qk_head_dim`` alone changed (``value_head_dim`` stays 64), at
the 15-layer width-896 reference shape and 1x budget: r841f35a24241, W = z h_t(z), z = softplus(s), with a query
sum-of-squares core of degree eight on the Mosaic transformed kernels (H100 or B200), and r6a76e7fbde45, the
unnormalized exponential on the fused Pallas-Triton kernel (H100 or B300), which also takes the gradient-event
capture the softplus-SoS run already has. A narrower key changes the shapes of the Q and K projections and of the
QK RMSNorm gains alone, so initialization is checked three ways: the softmax backbone's draws at that width
(``matches_softmax_backbone_initialization``); every partner array of unchanged shape bit-identical, with only
q_proj, k_proj and the Q/K RMSNorm gains differing; and each (partner, width) parameter count and SHA256, pinned
here from ``acceptance.initialization_facts`` on CPU.
"""
import argparse
from dataclasses import replace
from functools import partial
import hashlib
from pathlib import Path
import sys

from scaffolds.transformer.acceptance import (
    attention_spec, initialization_facts, main as run_acceptance, matches_softmax_backbone_initialization,
)
from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.configs.base import validate_config

RUNS = Path(__file__).resolve().parent / "runs"
WIDTHS = (4, 8, 16)
CAPTURE_POLICY = {
    "gradient_event_capture": True, "gradient_event_factor": 5.0,
    "gradient_event_min_norm": 1.0, "gradient_event_history": 100,
    "gradient_event_min_history": 20, "gradient_event_max_states": 3,
}
# Each partner, selected by its kernel: recipe SHA256, backend, the GPUs its kernels are qualified on, the training
# settings a run adds, and the seed-42 initial parameters (count, SHA256) at each width, computed on CPU with
# acceptance.initialization_facts on the partner's recipe with qk_head_dim changed.
PARTNERS = {
    "softplus_sos": {
        "run_id": "r841f35a24241",
        "config_sha256": "1643696258e0267ecba603c8e11cbfe9228cfcc43d9a9af91341d9b9a094a62b",
        "backend": "mosaic", "gpu_types": ("H100", "B200"), "train": {},
        "initialization": {
            4: (177_569_806, "98062beda15b818e558372eebfd952763cef53134d9691a5b0193826ad2ef3a0"),
            8: (179_075_206, "c7ac75ba66434f7b21762fd61b0f3bd791e5880844b2a8c0eb13e9707914f53e"),
            16: (182_086_006, "b92a7c4119d1f297efd65689f4961aac7b524a6a6503392148923b52f8d242cf"),
        },
    },
    "exponential": {
        "run_id": "r6a76e7fbde45",
        "config_sha256": "599dc40917e12a8308ae41fcb39623e1871e43fbd6d35402692d25f02450eee6",
        "backend": "fused", "gpu_types": ("H100", "B300"), "train": CAPTURE_POLICY,
        "initialization": {
            4: (175_872_586, "43c380f796e3c44a63a20ca96c1afd41db919f071e81fb942d811b9773b7becf"),
            8: (177_377_986, "3ae84675c937f136528edf83920beb189c6e80d7c8e4e239a0888b9717af0e92"),
            16: (180_388_786, "dfd19a74f9ddbd2fc7d9ef4cc53dc46dc5f3acbaa3938b43e47b84a2cd170b43"),
        },
    },
}
# The partner arrays a narrower key reshapes: the Q/K projections and the QK RMSNorm channel gains.
RESHAPED = ("q_proj", "k_proj", "qk_norm_q_gain", "qk_norm_k_gain")
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name",
                  "wandb_group", "wandb_dir", "wandb_mode")
CPU_TESTS = ("experiments/transformer_pp/tests/test_key_width_acceptance.py",)
GPU_TESTS = {"softplus_sos": ("scaffolds/transformer/tests/test_mosaic_score_transform.py",
                              "scaffolds/transformer/tests/test_mosaic_softplus_sos.py"),
             "exponential": ("scaffolds/transformer/tests/test_unnormalized_fused_attention.py",)}


def partner_key(config) -> str:
    """The partner of a recipe, by its kernel: the exponential has no degree."""
    return "exponential" if config.model.attention_degree is None else "softplus_sos"


def partner_config(key):
    if key not in PARTNERS:
        raise ValueError(f"No key-width partner {key!r}")
    partner = PARTNERS[key]
    path = RUNS / partner["run_id"] / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != partner["config_sha256"]:
        raise ValueError(f"Pinned key-width partner recipe {partner['run_id']} changed")
    return load_config(path)


def check_recipe(config):
    validate_config(config)
    key = partner_key(config)
    partner = PARTNERS[key]
    width = config.model.qk_head_dim
    if width not in WIDTHS:
        raise ValueError(f"Key-width runs take model.qk_head_dim in {WIDTHS}, not {width}")
    actual, expected = config_to_dict(config), config_to_dict(partner_config(key))
    expected["model"]["qk_head_dim"] = width
    expected["train"].update(partner["train"])
    for recipe in (actual, expected):
        for name in EXECUTION_KEYS:
            recipe["train"].pop(name, None)
    if actual != expected:
        differences = sorted(f"{section}.{name}" for section in set(actual) | set(expected)
                             for name in set(actual.get(section, {})) | set(expected.get(section, {}))
                             if actual.get(section, {}).get(name) != expected.get(section, {}).get(name))
        raise ValueError(f"Settings differ from key-width partner {partner['run_id']}: {differences}")


def flat_initial_parameters(config) -> dict:
    """Every initial array of the recipe on the host, by its flat path, built as training builds it."""
    import jax
    import numpy as np
    from flax import nnx

    from scaffolds.transformer.models import build_model

    rng = jax.random.PRNGKey(config.train.seed)
    model = build_model(replace(config.model, attention_backend="blocked"),
                        nnx.Rngs(params=rng, dropout=jax.random.fold_in(rng, 0)))
    return {"/".join(str(part) for part in path): np.asarray(value[...])
            for path, value in nnx.to_flat_state(nnx.state(model, nnx.Param))}


def compare_with_partner(config, partner) -> dict:
    """Every partner array of unchanged shape bit-identical; only RESHAPED arrays change, and all of them do."""
    import numpy as np

    actual, original = flat_initial_parameters(config), flat_initial_parameters(partner)
    if set(actual) != set(original):
        raise RuntimeError("A narrower key changed the partner's parameter names")
    reshaped = []
    for path, value in original.items():
        if any(part in path.split("/") for part in RESHAPED):
            if actual[path].shape == value.shape:
                raise RuntimeError(f"{path} kept its shape at the narrower key")
            reshaped.append(path)
        elif actual[path].shape != value.shape or not np.array_equal(actual[path], value):
            raise RuntimeError(f"Initial parameter {path} differs from the partner's")
    return {"reshaped_parameters": sorted(reshaped), "all_other_partner_arrays_identical": True}


def check_initialization(config):
    check_recipe(config)
    key = partner_key(config)
    partner = PARTNERS[key]
    width = config.model.qk_head_dim
    count, sha256 = partner["initialization"][width]
    facts = initialization_facts(replace(config, model=replace(config.model, attention_backend="blocked")))
    if facts != {"parameter_count": count, "parameter_sha256": sha256}:
        raise RuntimeError(f"Initial parameters differ from the pinned d_K={width} {partner['run_id']} recipe: {facts}")
    return {**matches_softmax_backbone_initialization(config), **compare_with_partner(config, partner_config(key)),
            "partner_run_id": partner["run_id"], "qk_head_dim": width, "parameter_count": count,
            "parameter_sha256": sha256, "pinned_initialization_matches": True}


def experiment_spec(config, *, entry_module):
    check_recipe(config)
    key = partner_key(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, cpu_tests=(*shared.cpu_tests, *CPU_TESTS), gpu_tests=(*shared.gpu_tests, *GPU_TESTS[key]),
                   required_backend=PARTNERS[key]["backend"], check_config=check_recipe,
                   check_initialization=check_initialization)


def require_gpu_type(argv) -> None:
    """Each partner's kernels train only on the GPUs they are qualified on; the recipe names the partner."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("config")
    parser.add_argument("--gpu-type", default="H100")
    parser.add_argument("--runtime-env", action="store_true")
    args = parser.parse_known_args(argv)[0]
    if args.runtime_env:
        return
    allowed = PARTNERS[partner_key(load_config(args.config))]["gpu_types"]
    if args.gpu_type.upper() not in allowed:
        raise ValueError(f"This key-width run qualifies and trains on {' or '.join(allowed)} only, not {args.gpu_type}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    require_gpu_type(argv)
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()
