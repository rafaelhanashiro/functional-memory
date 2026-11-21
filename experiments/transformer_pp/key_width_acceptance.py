"""Qualify the narrow-key runs: each partner's recipe at Q/K width 4, 8 or 16.

Each run is a completed partner's recipe with ``model.qk_head_dim`` changed (``value_head_dim`` stays 64), at the
15-layer width-896 reference shape and 1x budget:

- r841f35a24241, W = z h_t(z), z = softplus(s), with a query sum-of-squares core of degree eight, on the Mosaic
  transformed kernels;
- r6a76e7fbde45, the unnormalized exponential on the fused Pallas-Triton kernel, which also takes the gradient-event
  capture the softplus-SoS run already has and may shift the exponential by its causal row maximum
  (``model.attention_exp_shift: row_max``: the same function under head RMSNorm, finite where e^s overflows FP32);
- r9040f12f3d80, normalized softmax on cuDNN, which zero-pads Q and K to the value width 64, so the scores are the
  narrow key's exactly;
- the degree-eight polynomial ladder on the raw score, on the Mosaic query-coefficient kernel: fixed Taylor T_8(s)
  (ra7ee8a4b2544), static learned p(s) (r9c69dd7cfc8d), both of which run as every query's equal coefficients, and
  the tanh query polynomial p_t(s) with W decay (r16b23893884d, the key-width grid's d_K 8 run, which this adapter
  takes to width 4 alone).

Each partner's kernels train on the GPUs they are qualified on at the run's width (``gpu_types``). A narrower key
changes the shapes of the Q and K projections and of the QK RMSNorm gains alone, so initialization is checked three
ways: the softmax backbone's draws at that width (``matches_softmax_backbone_initialization``); every partner array of
unchanged shape bit-identical, with only q_proj, k_proj and the Q/K RMSNorm gains differing; and each (partner, width)
parameter count and SHA256, pinned here from ``acceptance.initialization_facts`` on CPU. The row-maximum shift adds
no parameter, so it keeps the pins.
"""
import argparse
from dataclasses import replace
from functools import partial
import hashlib
from pathlib import Path
import sys

from scaffolds.transformer.acceptance import (
    attention_spec, comparison_recipe, initialization_facts, main as run_acceptance,
    matches_softmax_backbone_initialization,
)
from scaffolds.transformer.configs import load_config
from scaffolds.transformer.configs.base import validate_config

RUNS = Path(__file__).resolve().parent / "runs"
WIDTHS = (4, 8, 16)
CAPTURE_POLICY = {
    "gradient_event_capture": True, "gradient_event_factor": 5.0,
    "gradient_event_min_norm": 1.0, "gradient_event_history": 100,
    "gradient_event_min_history": 20, "gradient_event_max_states": 3,
}


def every_width(gpus, widths=WIDTHS) -> dict:
    return {width: gpus for width in widths}


# The Mosaic query-coefficient kernel (the plain query family; static and fixed coefficients run on it with every
# query's coefficients equal): widths 8-128 are qualified on H100 and B300 (validation README, "Mosaic GPU polynomial
# attention kernels on H100 and B300"); width 4, added on 2026-11-20, on H100 (job 8217) and B200 (job 8251) only.
MOSAIC_QUERY_GPUS = {4: ("H100", "B200"), 8: ("H100", "B300"), 16: ("H100", "B300")}
# Each partner, selected by its kernel: recipe SHA256, backend, the GPUs its kernels are qualified on at each width,
# the training settings a run adds, the model settings a run may choose (each with its allowed values, the partner's
# first), and the seed-42 initial parameters (count, SHA256) at each width the partner takes, computed on CPU with
# acceptance.initialization_facts on the partner's recipe with qk_head_dim changed.
PARTNERS = {
    "softplus_sos": {
        "run_id": "r841f35a24241",
        "config_sha256": "1643696258e0267ecba603c8e11cbfe9228cfcc43d9a9af91341d9b9a094a62b",
        "backend": "mosaic", "gpu_types": every_width(("H100", "B200")), "train": {}, "model_options": {},
        "initialization": {
            4: (177_569_806, "98062beda15b818e558372eebfd952763cef53134d9691a5b0193826ad2ef3a0"),
            8: (179_075_206, "c7ac75ba66434f7b21762fd61b0f3bd791e5880844b2a8c0eb13e9707914f53e"),
            16: (182_086_006, "b92a7c4119d1f297efd65689f4961aac7b524a6a6503392148923b52f8d242cf"),
        },
    },
    "exponential": {
        "run_id": "r6a76e7fbde45",
        "config_sha256": "599dc40917e12a8308ae41fcb39623e1871e43fbd6d35402692d25f02450eee6",
        "backend": "fused", "gpu_types": every_width(("H100", "B300")), "train": CAPTURE_POLICY,
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
        "backend": "cudnn", "gpu_types": every_width(("H100", "B300")), "train": {}, "model_options": {},
        "initialization": {
            4: (175_859_146, "37c0f693729a83ea02b8c88b0e901340ee2f7586f53fa780fee52752c9bde031"),
            8: (177_364_546, "4af23ea498e142b9ba3710e3f63c911f27f236f20055513ecc010ce9aa41e553"),
            16: (180_375_346, "ea6990e908af9739d41b6eea2b886d0701bb5aed9c5430977d446dd7f254c7ef"),
        },
    },
    # Fixed Taylor T_8(s) adds no parameter to the exponential's backbone, so its pins are the exponential's.
    "taylor": {
        "run_id": "ra7ee8a4b2544",
        "config_sha256": "ad685b374e7caa940ebed20c5a696747e142b3ec252a6811cda322898c484a02",
        "backend": "mosaic", "gpu_types": MOSAIC_QUERY_GPUS, "train": {}, "model_options": {},
        "initialization": {
            4: (175_872_586, "43c380f796e3c44a63a20ca96c1afd41db919f071e81fb942d811b9773b7becf"),
            8: (177_377_986, "3ae84675c937f136528edf83920beb189c6e80d7c8e4e239a0888b9717af0e92"),
            16: (180_388_786, "dfd19a74f9ddbd2fc7d9ef4cc53dc46dc5f3acbaa3938b43e47b84a2cd170b43"),
        },
    },
    "static": {
        "run_id": "r9c69dd7cfc8d",
        "config_sha256": "f17be1f96b471b36d6c0369b4e4e7dc1ac93b0960376c920b8b40972760c9b8f",
        "backend": "mosaic", "gpu_types": MOSAIC_QUERY_GPUS, "train": {}, "model_options": {},
        "initialization": {
            4: (175_874_476, "7fe351aabe14b5cdd64dbc1ef98c38338ce8a566bb90c2119e78a3d772042f81"),
            8: (177_379_876, "d67d7873c0087e7482966873845bda863041979ff6808e61fd035b7884b6a634"),
            16: (180_390_676, "efc4d82ad5de6ca96c6aa79d220daa016a4d107e86e36b5a40b884f52a633044"),
        },
    },
    # The grid's d_K 8 run, taken to width 4 alone (the grid has 8, 16, 32 and 64). At width 4 it starts from the
    # same arrays (paths, shapes and values) as softplus-SoS, so the two pins agree.
    "query": {
        "run_id": "r16b23893884d",
        "config_sha256": "86c7df0a9f03354e46671838bb3f6432f49592ce1df1253c9ea1569caa0dfed7",
        "backend": "mosaic", "gpu_types": {4: MOSAIC_QUERY_GPUS[4]}, "train": {}, "model_options": {},
        "initialization": {
            4: (177_569_806, "98062beda15b818e558372eebfd952763cef53134d9691a5b0193826ad2ef3a0"),
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
             "softmax": SOFTMAX_WIDTH_TESTS, "query": ()}
# Fixed and static coefficients, whose shared Mosaic suite (test_mosaic_unnormalized.py) takes widths 16 and 64, add
# the Mosaic query-coefficient kernel's cases at the run's degree and width alone (a width the GPU is not qualified
# at stays out of its stage), against FP64 and the Pallas-Triton kernel; acceptance keeps the run's weight precision.
# The query polynomial's shared suite is that kernel's whole module.
MOSAIC_EQUAL_COEFFICIENT_KEYS = ("taylor", "static")


def mosaic_query_width_tests(degree, width) -> tuple:
    return tuple(f"scaffolds/transformer/tests/test_mosaic_query.py::test_outputs_and_vjp_match_fp64_and_triton"
                 f"[{degree}-{width}-{precision}]" for precision in ("bf16", "bf16x3"))


def gpu_tests(config) -> tuple:
    """The partner's own GPU cases for this recipe, beside the shared ones."""
    key = partner_key(config)
    if key in MOSAIC_EQUAL_COEFFICIENT_KEYS:
        return mosaic_query_width_tests(config.model.attention_degree, config.model.qk_head_dim)
    return GPU_TESTS[key]


def partner_key(config) -> str:
    """The partner of a recipe, by its kernel: softmax and the exponential have no degree; of the polynomials,
    softplus-SoS has square factors, fixed Taylor fixed coefficients, and the others static or query ones."""
    model = config.model
    if model.attention_degree is None:
        return "softmax" if model.attention_normalization else "exponential"
    if model.attention_coefficient_parameterization == "sum_of_squares":
        return "softplus_sos"
    if model.attention_coefficients == "fixed":
        return "taylor"
    return "query" if model.attention_coefficient_conditioning == "query" else "static"


def partner_widths(key) -> tuple:
    """The Q/K widths a partner's runs take: those with pinned initial parameters."""
    return tuple(PARTNERS[key]["initialization"])


def gpu_types(key, width) -> tuple:
    if width not in PARTNERS[key]["gpu_types"]:
        raise ValueError(f"Key-width partner {PARTNERS[key]['run_id']} takes model.qk_head_dim in "
                         f"{partner_widths(key)}, not {width}")
    return PARTNERS[key]["gpu_types"][width]


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
    if width not in partner_widths(key):
        raise ValueError(f"Key-width partner {partner['run_id']} takes model.qk_head_dim in {partner_widths(key)}, "
                         f"not {width}")
    actual, expected = comparison_recipe(config), comparison_recipe(partner_config(key))
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
    return replace(shared, cpu_tests=(*shared.cpu_tests, *CPU_TESTS), gpu_tests=(*shared.gpu_tests, *gpu_tests(config)),
                   required_backend=PARTNERS[key]["backend"], check_config=check_recipe,
                   check_initialization=check_initialization)


def require_gpu_type(argv) -> None:
    """Each partner's kernels train only on the GPUs they are qualified on at the recipe's width; the recipe names
    the partner and the width."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("config")
    parser.add_argument("--gpu-type", default="H100")
    parser.add_argument("--runtime-env", action="store_true")
    args = parser.parse_known_args(argv)[0]
    if args.runtime_env:
        return
    config = load_config(args.config)
    allowed = gpu_types(partner_key(config), config.model.qk_head_dim)
    if args.gpu_type.upper() not in allowed:
        raise ValueError(f"This key-width run qualifies and trains on {' or '.join(allowed)} only, not {args.gpu_type}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    require_gpu_type(argv)
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()
