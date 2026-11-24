"""Qualify pure tanh query coefficients c_r(x_t) = tanh(w_r^T x_hat_t + beta_r) / r! at Q/K width 64, degrees 8 and
16, on the row-power shifted kernel, with the degree-sixteen tanh query polynomial (b + a tanh), their affine
counterpart, on the same kernel.

Each run is the degree-eight tanh query + W decay run at Q/K width 64 (r25fffc16c2f3) at its degree, on the fused kernel
in units of each row's largest power (``attention_polynomial_shift: row_power``) with head RMSNorm's max-abs prescale:
the Mosaic kernels stop at degree twelve and take no inner bias. Pure tanh runs also take the Q/K width 4 runs'
coefficient settings (pure_tanh_prefactor_acceptance.CHANGES): every order 0..m from the raw attention-block input at
unit L2 norm, W at zero and every beta_r at atanh(1/2), so each coefficient starts at 0.5/r!. The optimizer, seed, data
order and schedule are the partner's; the dataset manifests may be another cluster's copies of the partner's.
"""
import argparse
from dataclasses import replace
from functools import partial
import hashlib
from pathlib import Path
import sys

from experiments.transformer_pp.pure_tanh_prefactor_acceptance import CHANGES as PURE_TANH
from scaffolds.transformer.acceptance import attention_spec, comparison_recipe, main as run_acceptance
from scaffolds.transformer.configs import Config, load_config
from scaffolds.transformer.configs.base import validate_config

PARTNER = Path(__file__).resolve().parent / "runs/r25fffc16c2f3/config.yaml"
PARTNER_SHA256 = "ffa72e4d99f016bedfd07b7b858199734d71596b3af18a325813bd810e3620b2"
# r25fffc16c2f3 reads Kennel's manifests; another cluster's copies must hash identically.
MANIFEST_SHA256 = {
    "train_manifest": "9de775ed836ab7c61857207335a981c09630caa3226d2dc0ee0b799cdcb2f4f4",
    "eval_manifest": "0e8615538d9dccad17956d8d9839394d05a56f8ec210dc67716db829a543634f",
}
# The kernel and evaluation of every run.
SHIFTED = {"attention_backend": "fused", "attention_polynomial_shift": "row_power",
           "attention_head_rms_prescale": "max_abs"}
# (coefficient activation, degree): pure tanh at 8 and 16, and tanh at 16, the partner's own generator.
CELLS = (("pure_tanh", 8), ("pure_tanh", 16), ("tanh", 16))
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name",
                  "wandb_group", "wandb_dir", "wandb_mode")
# The shifted kernel is qualified at Q/K width 4 on B200; on ORCD's Hopper GPUs each run's acceptance qualifies it at
# width 64 on the allocated GPU (WIDTH_TESTS) with the shifted module's own cases.
GPU_TYPES = ("H100", "H200")
WIDTH_TESTS = "scaffolds/transformer/tests/test_shifted_polynomial_key_width.py"
ADAPTER_TESTS = "experiments/transformer_pp/tests/test_pure_tanh_dk64_acceptance.py"


def partner_config() -> Config:
    if hashlib.sha256(PARTNER.read_bytes()).hexdigest() != PARTNER_SHA256:
        raise ValueError("Pinned tanh query partner r25fffc16c2f3 changed")
    return load_config(PARTNER)


def recipe(activation: str, degree: int) -> Config:
    if (activation, degree) not in CELLS:
        raise ValueError(f"The grid takes {CELLS}, not {(activation, degree)}")
    partner = partner_config()
    changes = {**(PURE_TANH if activation == "pure_tanh" else {}), **SHIFTED}
    return replace(partner, model=replace(partner.model, **changes, attention_degree=degree))


def recipe_key(config: Config) -> tuple[str, int]:
    return config.model.attention_coefficient_activation, config.model.attention_degree


def check_recipe(config: Config) -> None:
    validate_config(config)
    actual, expected = comparison_recipe(config), comparison_recipe(recipe(*recipe_key(config)))
    for key, digest in MANIFEST_SHA256.items():
        path, partner_path = actual["data"].pop(key), expected["data"].pop(key)
        if path != partner_path and hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"data.{key} is not a copy of the partner's dataset manifest")
    for entry in (actual, expected):
        for key in EXECUTION_KEYS:
            entry["train"].pop(key, None)
    if actual != expected:
        changed = sorted(f"{section}.{name}" for section in set(actual) | set(expected)
                         for name in set(actual.get(section, {})) | set(expected.get(section, {}))
                         if actual.get(section, {}).get(name) != expected.get(section, {}).get(name))
        raise ValueError(f"Settings differ from the {recipe_key(config)} recipe: {changed}")


def experiment_spec(config: Config, *, entry_module: str):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, check_config=check_recipe, required_backend="fused",
                   cpu_tests=(*shared.cpu_tests, ADAPTER_TESTS, WIDTH_TESTS),
                   gpu_tests=(*shared.gpu_tests, WIDTH_TESTS))


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu-type", default="H200")
    args, _ = parser.parse_known_args(argv)
    if args.gpu_type.upper() not in GPU_TYPES:
        raise ValueError(f"These runs are qualified on {', '.join(GPU_TYPES)}, not {args.gpu_type}")
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()
