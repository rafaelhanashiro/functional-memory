"""Qualify the narrow-key runs: softplus-SoS, the exponential and softmax at Q/K widths 4, 8, 16.

Each run is a completed partner's recipe with ``model.qk_head_dim`` changed (``value_head_dim`` stays 64), at the
15-layer width-896 reference shape and 1x budget: r841f35a24241, W = z h_t(z), z = softplus(s), with a query
sum-of-squares core of degree eight on the Mosaic transformed kernels (H100 or B200); r6a76e7fbde45, the
unnormalized exponential on the fused Pallas-Triton kernel (H100 or B300), which also takes the gradient-event
capture the softplus-SoS run already has and may shift the exponential by its causal row maximum
(``model.attention_exp_shift: row_max``: the same function under head RMSNorm, finite where e^s overflows FP32);
and r9040f12f3d80, normalized softmax on cuDNN (H100 or B300), which zero-pads Q and K to the value width 64, so the
scores are the narrow key's exactly. A narrower key changes the shapes of the Q and K projections and of the QK
RMSNorm gains alone, so initialization is checked three ways: the softmax backbone's draws at that width
(``matches_softmax_backbone_initialization``); every partner array of unchanged shape bit-identical, with only
q_proj, k_proj and the Q/K RMSNorm gains differing; and each (partner, width) parameter count and SHA256, pinned
here from ``acceptance.initialization_facts`` on CPU. The row-maximum shift adds no parameter, so it keeps the pins.
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
# settings a run adds, the model settings a run may choose (each with its allowed values, the partner's first), and
# the seed-42 initial parameters (count, SHA256) at each width, computed on CPU with acceptance.initialization_facts
# on the partner's recipe with qk_head_dim changed.
PARTNERS = {
    "softplus_sos": {
        "run_id": "r841f35a24241",
        "config_sha256": "1643696258e0267ecba603c8e11cbfe9228cfcc43d9a9af91341d9b9a094a62b",
        "backend": "mosaic", "gpu_types": ("H100", "B200"), "train": {}, "model_options": {},
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
        "model_options": {"attention_exp_shift": ("none", "row_max")},
        "initialization": {
            4: (175_872_586, "43c380f796e3c44a63a20ca96c1afd41db919f071e81fb942d811b9773b7becf"),
            8: (177_377_986, "3ae84675c937f136528edf83920beb189c6e80d7c8e4e239a0888b9717af0e92"),
            16: (180_388_786, "dfd19a74f9ddbd2fc7d9ef4cc53dc46dc5f3acbaa3938b43e47b84a2cd170b43"),
        },
    },
    "softmax": {
        "run_id": "r9040f12f3d80",
        "config_sha256": "92b7682b6c95c60ef9bb05ba067e9ceb87f6de0766af48bbbc8ca7e2bd4be897",
        "backend": "cudnn", "gpu_types": ("H100", "B300"), "train": {}, "model_options": {},
        "initialization": {
            4: (175_859_146, "37c0f693729a83ea02b8c88b0e901340ee2f7586f53fa780fee52752c9bde031"),
            8: (177_364_546, "4af23ea498e142b9ba3710e3f63c911f27f236f20055513ecc010ce9aa41e553"),
            16: (180_375_346, "ea6990e908af9739d41b6eea2b886d0701bb5aed9c5430977d446dd7f254c7ef"),
        },
    },
}
# The partner arrays a narrower key reshapes: the Q/K projections and the QK RMSNorm channel gains.
RESHAPED = ("q_proj", "k_proj", "qk_norm_q_gain", "qk_norm_k_gain")
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name",
                  "wandb_group", "wandb_dir", "wandb_mode")
CPU_TESTS = ("experiments/transformer_pp/tests/test_key_width_acceptance.py",)
# cuDNN softmax at the narrow widths, each zero-padded to the value width (the module's width-256 cases, which B300
# refuses for unequal Q/K and value widths, are not selected): parity with the manual softmax's outputs and
# gradients, and a finite forward and backward at the 2,048-token context. The shift's own GPU cases are among the
# fused module's and in test_exp_row_max_shift.py, which shared acceptance selects for shifted recipes.
SOFTMAX_WIDTH_TESTS = tuple(
    f"scaffolds/transformer/tests/test_softmax_backends.py::{test}[{width}]"
    for test in ("test_cudnn_matches_manual_outputs_input_and_parameter_gradients",
                 "test_cudnn_production_context_forward_and_backward_are_finite") for width in WIDTHS)
GPU_TESTS = {"softplus_sos": ("scaffolds/transformer/tests/test_mosaic_score_transform.py",
                              "scaffolds/transformer/tests/test_mosaic_softplus_sos.py"),
             "exponential": ("scaffolds/transformer/tests/test_unnormalized_fused_attention.py",),
             "softmax": SOFTMAX_WIDTH_TESTS}


def partner_key(config) -> str:
    """The partner of a recipe, by its kernel: softmax and the exponential have no degree."""
    if config.model.attention_degree is not None:
        return "softplus_sos"
    return "softmax" if config.model.attention_normalization else "exponential"


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
    for name, allowed in partner["model_options"].items():
        if actual["model"][name] in allowed:
            expected["model"][name] = actual["model"][name]
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
