"""Recipe-driven acceptance follows the mechanism and resolved implementation."""
from pathlib import Path

import pytest

from scaffolds.transformer.acceptance import attention_spec, main, test_paths as selected_paths
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict, save_config

ROOT = Path(__file__).resolve().parents[3]
ENTRY = "experiments.attention_scaling.acceptance"


def recipe(**model):
    config = Config(model=ModelConfig(**model))
    config.train.dtype = "bfloat16"
    config.data.synthetic_vocab_size = config.model.vocab_size
    return config


def checks(config, backend, cpu):
    return selected_paths(attention_spec(config, entry_module=ENTRY), config_to_dict(config),
                          resolved_backend=backend, cpu=cpu)


@pytest.mark.parametrize("degree", (1, 2, 3, 4))
@pytest.mark.parametrize("sharing", ("global", "per_head"))
@pytest.mark.parametrize("backend", ("auto", "blocked", "fused"))
def test_normalized_learned_recipes_support_current_backends_and_sharing(degree, sharing, backend):
    config = recipe(attention_degree=degree, attention_coefficients="learned",
                    attention_coefficient_sharing=sharing, attention_backend=backend)
    spec = attention_spec(config, entry_module=ENTRY)
    assert spec.required_backend == (None if backend == "auto" else backend)
    resolved = "fused" if backend == "auto" else backend
    selected = checks(config, resolved, False)
    expected = ("test_normalized_polynomial_fused_attention.py" if resolved == "fused"
                else "test_learned_polynomial_normalized.py")
    assert any(path.endswith(expected) for path in selected)
    if resolved == "blocked":
        assert not any("fused" in path for path in selected)


@pytest.mark.parametrize("changes,backend,required", [
    ({"attention_degree": 2, "quadratic_mode": "unrestricted"}, "fused", "learned_fused_attention"),
    ({"attention_degree": 4, "attention_coefficients": "learned", "attention_normalization": False},
     "fused", "learned_polynomial_fused_attention"),
    ({"attention_degree": 4, "attention_coefficients": "learned",
      "attention_coefficient_parameterization": "sum_of_squares"}, "fused", "normalized_polynomial_fused_attention"),
    ({"attention_degree": 3}, "fused", "odd_attention"),
    ({"attention_degree": 4}, "fused", "taylor_kernel_widths"),
    ({"attention_normalization": False}, "fused", "unnormalized_fused_attention"),
    ({"attention_mechanism": "gla"}, "fused", "gla_kernel"),
    ({"attention_degree": 2, "attention_fixed_coefficients": (0.5, 1.0, 0.5)}, "fused", "fixed_polynomial_fused"),
    ({"attention_degree": 2, "attention_fixed_square_factors": (1.0, 1.0, 0.0),
      "attention_coefficient_parameterization": "sum_of_squares"}, "fused", "fixed_square_factors_fused"),
    ({}, "cudnn", "attention.py::test_gpu_exact_control_matches_original_cudnn"),
])
def test_qualification_covers_each_recipe_mechanism(changes, backend, required):
    config = recipe(**changes)
    selected = checks(config, backend, False)
    assert any(f"test_{required}" in path for path in selected)
    for cpu in (False, True):
        paths = checks(config, backend, cpu)
        assert all((ROOT / path.split("::")[0]).is_file() for path in paths)
        assert all(path.startswith("scaffolds/transformer/tests/") for path in paths)
        assert len(paths) == len(set(paths))


def test_reference_gla_does_not_require_a_fused_kernel():
    config = recipe(attention_mechanism="gla", attention_backend="blocked")
    assert checks(config, "blocked", False) == ()
    assert any(path.endswith("test_gla.py") for path in checks(config, "blocked", True))


@pytest.mark.parametrize("changes", [
    {}, {"attention_degree": 0},
    {"attention_degree": 2, "quadratic_mode": "unrestricted"},
    {"attention_degree": 4, "attention_coefficients": "learned", "attention_coefficient_sharing": "per_head"},
    {"attention_degree": 4, "attention_coefficients": "learned",
     "attention_coefficient_parameterization": "sum_of_squares"},
    {"attention_degree": 2, "attention_fixed_coefficients": (0.5, 1.0, 0.5)},
])
def test_shared_initialization_check_accepts_each_parameter_ownership(changes):
    config = recipe(vocab_size=16, hidden_size=16, intermediate_size=32,
                    num_layers=1, num_heads=2, num_kv_heads=2, qk_head_dim=8, value_head_dim=8,
                    attention_backend="blocked", **changes)
    check = attention_spec(config, entry_module=ENTRY).check_initialization
    facts = check(config)
    assert len(facts["softmax_counterpart_parameter_sha256"]) == 64


def test_gla_runtime_flags_are_recipe_specific_and_preserve_other_flags(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XLA_FLAGS", "--other=1 --xla_gpu_enable_command_buffer=FUSION")
    gla = recipe(attention_mechanism="gla")
    config = tmp_path / "config.yaml"
    save_config(gla, config)
    main(lambda cfg: attention_spec(cfg, entry_module=ENTRY), [str(config), "--runtime-env"])
    output = capsys.readouterr().out
    assert "export XLA_FLAGS='--other=1 --xla_gpu_enable_command_buffer='" in output
    assert "EJKERNEL_TRITON_CACHE_DIR" in output
    assert attention_spec(recipe(), entry_module=ENTRY).runtime_env == {}
    assert gla.model.attention_backend == "auto"


def test_runtime_env_requires_the_recipe_for_a_configuration_driven_adapter(capsys):
    with pytest.raises(SystemExit) as error:
        main(lambda cfg: attention_spec(cfg, entry_module=ENTRY), ["--runtime-env"])
    assert error.value.code == 2
    assert "config is required" in capsys.readouterr().err


@pytest.mark.parametrize("launcher", ("kennel.sbatch", "orcd.sbatch"))
def test_launchers_pass_the_allocated_recipe_to_runtime_selection(launcher):
    source = (ROOT / "scaffolds/transformer/launchers" / launcher).read_text()
    assert '"$ADAPTER" "$CONFIG" --runtime-env' in source
