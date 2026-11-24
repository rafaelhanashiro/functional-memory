"""Matched width-four unscaled-cosine controls for affine query and softplus-prefactor SoS."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from scaffolds.transformer.acceptance import attention_spec, main
from scaffolds.transformer.configs import load_config


CONTROLS = {"direct": "rd5e3ec5a4f02", "sum_of_squares": "rabfbab32a594"}
RUNS = Path(__file__).resolve().parent / "runs"


def check_recipe(config):
    control = load_config(RUNS / CONTROLS[config.model.attention_coefficient_parameterization] / "config.yaml")
    model = replace(control.model, qk_norm_type="rms_norm_cosine", qk_norm_head_scale=False,
                    attention_head_rms_prescale="max_abs")
    # Acceptance sets the launch commit before it checks the recipe.
    train = replace(control.train, workdir=config.train.workdir, checkpoint_dir=config.train.checkpoint_dir,
                    wandb_run_name=config.train.wandb_run_name, source_commit=config.train.source_commit)
    if config != replace(control, model=model, train=train):
        raise ValueError("The affine-query cosine recipe differs from its width-four control")


def check_pilot(config, outcome, validation):
    layers = validation["attention_diagnostics"]["layers"]
    low = np.asarray([row["score_min"] for row in layers])
    high = np.asarray([row["score_max"] for row in layers])
    # The final query and key directions are rounded to BF16 before the dot product.
    if len(layers) != config.model.num_layers or not np.all((-1.01 <= low) & (low <= high) & (high <= 1.01)):
        raise RuntimeError("Pilot raw scores exceed the unscaled cosine range or are missing")
    return {"score_min": float(low.min()), "score_max": float(high.max())}


def experiment_spec(config):
    check_recipe(config)
    shared = attention_spec(config, entry_module="experiments.transformer_pp.affine_query_cosine_acceptance")
    gpu_tests = shared.gpu_tests
    if config.model.attention_coefficient_parameterization == "sum_of_squares":
        gpu_tests += ("scaffolds/transformer/tests/test_mosaic_score_transform.py",
                      "scaffolds/transformer/tests/test_mosaic_softplus_sos.py")
    return replace(shared, check_config=check_recipe, check_pilot=check_pilot, gpu_tests=gpu_tests,
                   cpu_tests=(*shared.cpu_tests, "scaffolds/transformer/tests/test_qk_rms_cosine.py",
                              "experiments/transformer_pp/tests/test_affine_query_cosine_acceptance.py"))


if __name__ == "__main__":
    main(experiment_spec)
