"""Each unnormalized Taylor recipe changes only the kernel and output locations."""
from dataclasses import asdict, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from scaffolds.transformer.configs import Config, ModelConfig, load_config, save_config, validate_config
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.training.diagnostics import attention_identity


ROOT = Path(__file__).resolve().parents[4]
BASE_CONFIG = ROOT / "experiments/unnormalized_attention/configs/base.yaml"


@pytest.mark.parametrize("degree", (1, 2, 3, 4))
def test_unnormalized_taylor_recipes_change_only_the_kernel_and_locations(degree):
    from scaffolds.transformer.configs import config_to_dict
    run = f"200m-m{degree:03d}-seed0"
    path = ROOT / "experiments/unnormalized_taylor_attention/configs" / f"{run}.yaml"
    base, taylor = (config_to_dict(load_config(p)) for p in (BASE_CONFIG, path))
    assert taylor["model"].pop("attention_degree") == degree
    assert base["model"].pop("attention_degree") is None
    assert taylor["train"]["workdir"] == f"experiments/unnormalized_taylor_attention/runs/{run}"
    assert taylor["train"]["checkpoint_dir"] == f"experiments/unnormalized_taylor_attention/checkpoints/{run}"
    for section, key in (("data", "train_manifest"), ("data", "eval_manifest"), ("train", "workdir"),
                         ("train", "checkpoint_dir"), ("train", "wandb_project"), ("train", "wandb_run_name")):
        base[section].pop(key)
        taylor[section].pop(key)
    assert taylor == base
