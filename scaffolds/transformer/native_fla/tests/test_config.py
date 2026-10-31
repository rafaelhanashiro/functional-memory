"""Reference fidelity and rejection of accidental shared-recipe inheritance."""
from copy import deepcopy
import importlib.metadata
import json

import pytest
import yaml

from scaffolds.transformer.native_fla.config import (
    FLA_COMMIT, fla_model_kwargs, load_config, parameter_count, reference_recipe, validate_config,
)


def complete_recipe():
    recipe = reference_recipe()
    recipe["data"].update(train_manifest="/data/train/manifest.json", eval_manifest="/data/eval/manifest.json")
    recipe["train"].update(workdir="/runs/example", checkpoint_dir="/runs/example/checkpoints", wandb_mode="disabled")
    return recipe


def test_reference_is_explicit_and_survives_yaml(tmp_path):
    recipe = complete_recipe()
    validate_config(recipe)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(recipe))
    assert load_config(path) == recipe
    assert parameter_count(recipe) == 200_205_878
    assert recipe["data"]["batch_size"] * (recipe["data"]["sequence_length"] - 1) * recipe["optim"]["total_steps"] == 10_737_418_240


@pytest.mark.parametrize("section,key,value", [
    ("optim", "learning_rate", 0.001),
    ("optim", "eps", 1e-8),
    ("optim", "warmup_steps", 455),
    ("optim", "weight_decay_policy", "exclude_embeddings_norms_biases_v1"),
    ("model", "initializer_range", 0.01),
    ("model", "tie_word_embeddings", True),
    ("model", "use_short_conv", False),
    ("model", "allow_neg_eigval", True),
    ("model", "num_v_heads", 6),
    ("model", "intermediate_size", 3008),
    ("train", "dtype", "float32"),
])
def test_accidental_recipe_overrides_fail(section, key, value):
    recipe = complete_recipe()
    recipe[section][key] = value
    with pytest.raises(ValueError):
        validate_config(recipe)


@pytest.mark.parametrize("section,key", [("optim", "eps"), ("model", "initializer_range"), ("train", "dtype")])
def test_no_scientific_default_is_filled_implicitly(section, key):
    recipe = complete_recipe()
    del recipe[section][key]
    with pytest.raises(ValueError, match="missing"):
        validate_config(recipe)


def test_jax_model_settings_are_not_accepted():
    recipe = complete_recipe()
    recipe["model"]["initialization"] = "gated_deltanet_reference"
    with pytest.raises(ValueError, match="unknown"):
        validate_config(recipe)


def test_approved_dimension_scaling_preserves_reference_proportions():
    recipe = complete_recipe()
    recipe["model"].update(hidden_size=2048, num_hidden_layers=21, num_heads=6, intermediate_size=5632)
    validate_config(recipe)
    recipe["model"]["num_heads"] = 8
    with pytest.raises(ValueError, match="key projection"):
        validate_config(recipe)


def test_allocation_and_runtime_settings_must_be_explicit():
    with pytest.raises(ValueError, match="manifest"):
        validate_config(reference_recipe())
    recipe = complete_recipe()
    recipe["train"]["micro_batch_size"] = 7
    with pytest.raises(ValueError, match="divide"):
        validate_config(recipe)


def test_model_kwargs_are_an_independent_copy():
    recipe = complete_recipe()
    kwargs = fla_model_kwargs(recipe)
    kwargs["hidden_size"] = 8
    assert recipe["model"]["hidden_size"] == 1024


def test_pinned_upstream_defaults_and_native_parameter_count():
    """The independent upstream class owns model construction and initialization."""
    import torch
    from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetForCausalLM

    direct_url = json.loads(importlib.metadata.distribution("flash-linear-attention").read_text("direct_url.json"))
    assert direct_url["vcs_info"]["commit_id"] == FLA_COMMIT
    recipe = complete_recipe()
    defaults = GatedDeltaNetConfig()
    adjusted = {"hidden_size", "num_hidden_layers", "num_heads", "intermediate_size", "use_cache"}
    for name, value in recipe["model"].items():
        if name not in adjusted:
            assert getattr(defaults, name) == value, name

    # Meta construction verifies the complete proposed model without allocating
    # 200M weights, running a CUDA kernel or consuming a GPU on a login node.
    with torch.device("meta"):
        model = GatedDeltaNetForCausalLM(GatedDeltaNetConfig(**fla_model_kwargs(recipe)))
    assert sum(p.numel() for p in model.parameters()) == parameter_count(recipe)
    assert model.lm_head.weight is not model.model.embeddings.weight
    assert model.model.layers[0].attn.num_heads == model.model.layers[0].attn.num_v_heads == 3


def test_native_initialization_matches_the_full_model_policy():
    """Catch the easy mistake of retaining standalone convolution initialization."""
    import torch
    from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetForCausalLM

    recipe = deepcopy(complete_recipe())
    recipe["model"].update(hidden_size=128, num_hidden_layers=2, num_heads=3,
                           head_dim=32, intermediate_size=512, vocab_size=1024)
    torch.manual_seed(42)
    model = GatedDeltaNetForCausalLM(GatedDeltaNetConfig(**fla_model_kwargs(recipe)))
    assert sum(p.numel() for p in model.parameters()) == parameter_count(recipe)
    attention = model.model.layers[0].attn
    tensors = [attention.q_proj.weight, attention.o_proj.weight, attention.q_conv1d.weight,
               model.model.layers[0].mlp.down_proj.weight,
               model.model.embeddings.weight, model.lm_head.weight]
    for weight in tensors:
        assert weight.dtype == torch.float32
        assert 0.017 < float(weight.detach().std()) < 0.023
    assert torch.equal(model.model.norm.weight, torch.ones_like(model.model.norm.weight))
    assert attention.A_log._no_weight_decay and attention.dt_bias._no_weight_decay
    assert torch.all((attention.A_log.exp() > 0) & (attention.A_log.exp() < 16))
    dt = torch.nn.functional.softplus(attention.dt_bias)
    assert torch.all((dt >= 0.001 - 1e-7) & (dt <= 0.1 + 1e-7))


def test_external_shifted_targets_match_upstream_label_convention():
    """Supervise every prefix position without asking FLA to shift labels twice.

    The real FLA causal-LM wrapper owns logits and target alignment. A tokenwise
    CPU decoder replaces only its CUDA-only recurrent blocks for this test.
    """
    import torch
    from torch import nn
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetForCausalLM

    class TokenDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(16, 16)

        def forward(self, input_ids, **kwargs):
            return BaseModelOutputWithPast(last_hidden_state=self.embedding(input_ids))

    kwargs = fla_model_kwargs(complete_recipe())
    kwargs.update(hidden_size=16, num_hidden_layers=1, num_heads=1, head_dim=8,
                  intermediate_size=32, vocab_size=16, fuse_cross_entropy=False)
    torch.manual_seed(42)
    model = GatedDeltaNetForCausalLM(GatedDeltaNetConfig(**kwargs))
    model.model = TokenDecoder()
    tokens = torch.tensor([[1, 4, 2, 7, 9], [3, 5, 8, 6, 2]])
    prefix, targets = tokens[:, :-1], tokens[:, 1:]
    logits = model(input_ids=prefix, labels=None, use_cache=False).logits
    actual = nn.functional.cross_entropy(logits.reshape(-1, 16), targets.reshape(-1))
    expected = model(input_ids=tokens, labels=tokens, use_cache=False).loss
    torch.testing.assert_close(actual, expected)
    # Passing already-shifted targets to labels would shift again and discard
    # the final prediction. Establish that this is not an equivalent call.
    double_shifted = model(input_ids=prefix, labels=targets, use_cache=False).loss
    assert not torch.isclose(actual, double_shifted, rtol=1e-6, atol=1e-7)
