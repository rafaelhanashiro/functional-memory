"""CPU-only checks of the installed FLA reference model initialization."""

from copy import deepcopy
from functools import wraps
from inspect import signature
from unittest.mock import patch

import pytest
import torch


def _import_fla_reference():
    # Triton 3.3 asks for a GPU benchmarker when decorating FLA kernels at import
    # time, even though these tests only initialize CPU parameters. Defer that
    # lookup without replacing any model initializer or kernel implementation.
    # GPU test runs import FLA normally. The temporary constructor is restored as
    # soon as the CPU import finishes, and its callback uses the real driver if
    # a benchmark is ever requested. No CPU forward pass is attempted here.
    if torch.cuda.is_available():
        from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetModel
    else:
        from triton.runtime.autotuner import Autotuner
        from triton.runtime.driver import driver

        original_init = Autotuner.__init__
        init_signature = signature(original_init)

        def deferred_benchmark(*args, **kwargs):
            return driver.active.get_benchmarker()(*args, **kwargs)

        @wraps(original_init)
        def lazy_autotuner_init(self, *args, **kwargs):
            # FLA's CachedAutotuner subclasses Autotuner directly, so intercept
            # this constructor instead of only the public autotune decorator.
            bound = init_signature.bind_partial(self, *args, **kwargs)
            if bound.arguments.get("do_bench") is None:
                bound.arguments["do_bench"] = deferred_benchmark
            return original_init(*bound.args, **bound.kwargs)

        with patch.object(Autotuner, "__init__", lazy_autotuner_init):
            from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetModel
    return GatedDeltaNetConfig, GatedDeltaNetModel


GatedDeltaNetConfig, GatedDeltaNetModel = _import_fla_reference()
from transformers import AutoModel, LlamaConfig

from scaffolds.state_rank.random_models import build_random_gated_deltanet, parameter_fingerprint


@pytest.fixture
def config():
    return GatedDeltaNetConfig(
        vocab_size=127, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_heads=4, head_dim=8, expand_v=2,
        max_position_embeddings=64, use_gate=True, use_short_conv=True,
        conv_size=4, attn_mode="chunk", initializer_range=0.02,
    )


def test_reproducibility_and_exact_fla_reference_initialization(config):
    first = build_random_gated_deltanet(config, 17)
    second = build_random_gated_deltanet(config, 17)
    reference_config = deepcopy(config)
    reference_config.attn_mode = "fused_recurrent"
    previous_dtype = torch.get_default_dtype()
    try:
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.set_default_dtype(torch.float32)
            torch.random.default_generator.manual_seed(17)
            # Independent direct FLA construction executes its normal post_init.
            reference = GatedDeltaNetModel(reference_config)
    finally:
        torch.set_default_dtype(previous_dtype)
    assert isinstance(first, GatedDeltaNetModel)
    assert first.config.attn_mode == "fused_recurrent"
    assert not first.training
    assert set(dict(first.named_parameters())) == set(dict(reference.named_parameters()))
    for name, parameter in first.named_parameters():
        assert parameter.device.type == "cpu" and parameter.dtype == torch.float32
        assert torch.equal(parameter, second.get_parameter(name)), name
        assert torch.equal(parameter, reference.get_parameter(name)), name
    assert parameter_fingerprint(first) == parameter_fingerprint(second)
    assert parameter_fingerprint(first) != parameter_fingerprint(build_random_gated_deltanet(config, 18))


def test_fla_special_gates_convolutions_norms_and_independent_layers(config):
    model = build_random_gated_deltanet(config, 9)
    first, second = model.layers[0].attn, model.layers[1].attn
    for layer in model.layers:
        attention = layer.attn
        assert attention.mode == "fused_recurrent"
        assert not hasattr(attention, "D")
        decay = attention.A_log.exp()
        time_step = torch.nn.functional.softplus(attention.dt_bias)
        assert torch.all((decay > 0) & (decay < 16))
        assert torch.all((time_step >= 0.001) & (time_step <= 0.1))
        assert attention.A_log._no_weight_decay is True
        assert attention.dt_bias._no_weight_decay is True
        for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
            convolution = getattr(attention, name)
            assert convolution.weight.shape[-1] == config.conv_size
            assert torch.count_nonzero(convolution.weight) == convolution.weight.numel()
            # FLA post_init replaces Conv1d's default uniform with normal(0, .02).
            assert 0.012 < float(convolution.weight.detach().std()) < 0.028
        for norm in (layer.attn_norm, layer.mlp_norm, attention.o_norm):
            assert torch.equal(norm.weight, torch.ones_like(norm.weight))
    assert torch.equal(model.norm.weight, torch.ones_like(model.norm.weight))
    for name in ("q_proj", "k_proj", "v_proj", "a_proj", "b_proj", "g_proj", "o_proj",
                 "q_conv1d", "k_conv1d", "v_conv1d"):
        a, b = getattr(first, name).weight, getattr(second, name).weight
        assert a.data_ptr() != b.data_ptr()
        assert not torch.equal(a, b)
    assert not torch.equal(first.A_log, second.A_log)
    assert not torch.equal(first.dt_bias, second.dt_bias)
    assert not torch.equal(first.q_proj.weight[:config.head_dim], first.q_proj.weight[config.head_dim:2*config.head_dim])
    assert not torch.equal(first.a_proj.weight[0], first.a_proj.weight[1])
    assert not torch.equal(model.layers[0].mlp.gate_proj.weight, model.layers[1].mlp.gate_proj.weight)
    assert 0.018 < float(model.embeddings.weight.detach().std()) < 0.022


def test_caller_rng_config_default_dtype_and_device_are_preserved(config):
    config.dtype = torch.bfloat16
    before_config = deepcopy(config.to_dict())
    old_dtype = torch.get_default_dtype()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(71)
        before_rng = torch.random.get_rng_state().clone()
        try:
            torch.set_default_dtype(torch.float64)
            with torch.device("meta"):
                model = build_random_gated_deltanet(config, 3)
                assert torch.empty(0).device.type == "meta"
                assert torch.get_default_dtype() == torch.float64
            assert torch.equal(torch.random.get_rng_state(), before_rng)
            assert all(p.dtype == torch.float32 and p.device.type == "cpu" for p in model.parameters())
        finally:
            torch.set_default_dtype(old_dtype)
    assert config.to_dict() == before_config
    # Drawing directly in FP32 must give the same values regardless of caller defaults.
    plain = build_random_gated_deltanet(config, 3)
    assert parameter_fingerprint(model) == parameter_fingerprint(plain)


def test_defaults_and_rng_are_restored_after_constructor_failure(config, monkeypatch):
    def fail(*args, **kwargs):
        torch.rand(5)
        raise RuntimeError("construction failed")

    before_rng = torch.random.get_rng_state().clone()
    before_dtype = torch.get_default_dtype()
    monkeypatch.setattr(AutoModel, "from_config", fail)
    with pytest.raises(RuntimeError, match="construction failed"):
        build_random_gated_deltanet(config, 0)
    assert torch.equal(torch.random.get_rng_state(), before_rng)
    assert torch.get_default_dtype() == before_dtype


@pytest.mark.parametrize("seed", [-1, True, 1.5, "0", 2**64])
def test_invalid_seeds_rejected(config, seed):
    with pytest.raises(ValueError, match="seed"):
        build_random_gated_deltanet(config, seed)


def test_non_deltanet_configuration_is_rejected():
    with pytest.raises(ValueError, match="GatedDeltaNet configuration"):
        build_random_gated_deltanet(LlamaConfig(), 0)
