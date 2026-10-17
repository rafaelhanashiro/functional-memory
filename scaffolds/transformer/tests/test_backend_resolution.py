"""Backend selection is explicit provenance, including across resumed processes."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field

import pytest

from scaffolds.transformer.attention.backends.selection import resolve_attention_backend
from scaffolds.transformer.configs import Config, ModelConfig, load_config
from scaffolds.transformer.configs.loader import config_dicts_equal, config_to_dict
from scaffolds.transformer.training.diagnostics import attention_identity
from scaffolds.transformer.training.loop import ensure_run_config


@pytest.mark.parametrize("platform,dtype,degree,width,mode,expected", [
    ("cpu", "float32", None, 32, "fixed", "blocked"),
    ("cpu", "bfloat16", 2, 32, "positive", "blocked"),
    ("gpu", "float32", None, 64, "fixed", "xla"),
    ("gpu", "bfloat16", None, 64, "fixed", "cudnn"),
    ("gpu", "bfloat16", None, 256, "fixed", "xla"),
    ("gpu", "float32", 3, 64, "fixed", "fused"),
    ("gpu", "bfloat16", 2, 128, "unrestricted", "fused"),
    ("gpu", "float32", 2, 256, "fixed", "blocked"),
])
def test_auto_selects_supported_implementation(platform, dtype, degree, width, mode, expected):
    model = ModelConfig(attention_degree=degree, qk_head_dim=width, quadratic_mode=mode)
    assert resolve_attention_backend(model, dtype=dtype, platform=platform) == expected
    assert model.attention_backend == "auto"


@pytest.mark.parametrize("backend,dtype,degree,width,platform,error", [
    ("fused", "float32", 2, 256, "gpu", "up to 128"),
    ("fused", "float32", 2, 64, "cpu", "NVIDIA GPU"),
    ("cudnn", "float32", None, 64, "gpu", "bfloat16"),
    ("cudnn", "bfloat16", None, 64, "cpu", "requires a GPU"),
    ("cudnn", "bfloat16", None, 256, "gpu", "up to 128"),
    ("xla", "float32", 2, 64, "gpu", "exact softmax only"),
])
def test_explicit_unsupported_choice_never_falls_back(backend, dtype, degree, width, platform, error):
    model = ModelConfig(attention_degree=degree, qk_head_dim=width)
    model.attention_backend = backend
    with pytest.raises(ValueError, match=error):
        resolve_attention_backend(model, dtype=dtype, platform=platform)


def test_resume_auto_uses_persisted_backend_without_reselection(tmp_path, monkeypatch):
    from scaffolds.transformer.attention.backends import selection

    config = Config()
    config.train.workdir = tmp_path
    # Save a concrete backend that differs from the CPU auto selection.
    config.model.attention_backend = "xla"
    ensure_run_config(config, resume=True)
    persisted = (tmp_path / "config.json").read_bytes()
    resumed = Config()
    resumed.train.workdir = tmp_path
    monkeypatch.setattr(selection, "_runtime_platform", lambda: "cpu")
    ensure_run_config(resumed, resume=True)
    assert resumed.model.attention_backend == "xla"
    assert (tmp_path / "config.json").read_bytes() == persisted
    identity = attention_identity(resumed)
    assert identity["attention_backend_requested"] == "auto"
    assert identity["attention_backend"] == "xla"
    assert identity["diagnostic_backend"] == "blocked"
    resumed.model.attention_backend = "manual"
    with pytest.raises(ValueError, match="does not match"):
        ensure_run_config(resumed, resume=True)


def test_auto_is_resolved_in_saved_config_and_request_kept_in_metadata(tmp_path):
    config = Config()
    config.train.workdir = tmp_path
    ensure_run_config(config, resume=True)
    assert json.loads((tmp_path / "config.json").read_text())["model"]["attention_backend"] == "blocked"
    assert attention_identity(config)["attention_backend_requested"] == "auto"


@pytest.mark.parametrize("parameterization", ("direct", "sum_of_squares"))
def test_learned_normalized_resume_keeps_recorded_blocked_backend(tmp_path, monkeypatch, parameterization):
    from scaffolds.transformer.attention.backends import selection

    config = Config()
    config.train.workdir = tmp_path
    config.model = ModelConfig(attention_degree=2, attention_coefficients="learned",
                               attention_coefficient_parameterization=parameterization,
                               attention_backend="blocked")
    ensure_run_config(config, resume=True)
    original = (tmp_path / "config.json").read_bytes()
    resumed = load_config(tmp_path / "config.json")
    resumed.model.attention_backend = "auto"
    monkeypatch.setattr(selection, "_runtime_platform", lambda: "gpu")
    monkeypatch.setattr(selection, "_cuda_compute_capability", lambda: 9)
    assert resolve_attention_backend(resumed.model, dtype="bfloat16") == "fused"
    ensure_run_config(resumed, resume=True)
    assert resumed.model.attention_backend == "blocked"
    assert (tmp_path / "config.json").read_bytes() == original


@pytest.mark.parametrize("parameterization", ("direct", "sum_of_squares"))
def test_fused_normalized_identity_records_reference_diagnostics(parameterization):
    config = Config()
    config.model = ModelConfig(attention_degree=2, attention_coefficients="learned",
                               attention_coefficient_parameterization=parameterization,
                               attention_backend="fused")
    identity = attention_identity(config)
    assert identity["attention_backend"] == "fused"
    assert identity["attention_implementation"] == "pallas_triton"
    assert identity["diagnostic_backend"] == "blocked"
    assert identity["attention_coefficient_parameterization"] == parameterization
    assert identity["query_block_size"] > 0 and identity["key_block_size"] > 0


def test_resuming_saved_resolved_config_preserves_original_request(tmp_path):
    from scaffolds.transformer.training.loop import _write_status

    config = Config()
    config.train.workdir = tmp_path
    ensure_run_config(config, resume=True)
    _write_status(config, status="initialized")
    path = tmp_path / "config.json"
    original = path.read_bytes()
    resumed = load_config(path)
    assert resumed.model.attention_backend == "blocked"
    ensure_run_config(resumed, resume=True)
    _write_status(resumed, status="paused")
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["attention_backend_requested"] == "auto"
    assert metadata["attention_backend_resume_requested"] == "blocked"
    assert metadata["attention_backend"] == "blocked"
    assert path.read_bytes() == original


def test_legacy_fixed_config_equivalence_does_not_rewrite_provenance(tmp_path):
    config = Config()
    config.train.workdir = tmp_path
    config.model.attention_backend = "manual"
    legacy = config_to_dict(config)
    for key in ("attention_degree", "quadratic_mode", "quadratic_sharing"):
        legacy["model"].pop(key)
    old = copy.deepcopy(legacy)
    assert config_dicts_equal(legacy, config_to_dict(config))
    assert legacy == old
    path = tmp_path / "config.json"
    path.write_text(json.dumps(legacy))
    persisted = path.read_bytes()
    ensure_run_config(config, resume=True)
    assert path.read_bytes() == persisted


def test_legacy_auto_without_resolved_provenance_is_rejected(tmp_path):
    config = Config()
    config.train.workdir = tmp_path
    (tmp_path / "config.json").write_text(json.dumps(config_to_dict(config)))
    with pytest.raises(ValueError, match="no recorded concrete"):
        ensure_run_config(config, resume=True)


def test_loader_supports_family_default_factory(tmp_path):
    @dataclass
    class FamilyModel(ModelConfig):
        attention_backend: str = "manual"

    @dataclass
    class FamilyConfig(Config):
        model: ModelConfig = field(default_factory=FamilyModel)

    path = tmp_path / "family.yaml"
    path.write_text("model:\n  qk_head_dim: 16\n")
    config = load_config(path, config_factory=FamilyConfig)
    assert config.model.qk_head_dim == 16
    assert config.model.attention_backend == "manual"


MOSAIC_TAYLOR = dict(attention_backend="mosaic", attention_degree=2, qk_head_dim=64, value_head_dim=64)
MOSAIC_QUERY = dict(MOSAIC_TAYLOR, attention_degree=3, attention_normalization=False, attention_coefficients="learned",
                    attention_coefficient_sharing="per_head", attention_coefficient_conditioning="query",
                    attention_coefficient_activation="tanh")


@pytest.mark.parametrize("settings", [
    MOSAIC_TAYLOR, dict(MOSAIC_TAYLOR, attention_degree=1), dict(MOSAIC_TAYLOR, attention_degree=16),
    dict(MOSAIC_TAYLOR, qk_head_dim=8), dict(MOSAIC_TAYLOR, qk_head_dim=128),
    MOSAIC_QUERY, dict(MOSAIC_QUERY, attention_degree=1), dict(MOSAIC_QUERY, attention_degree=4),
])
def test_mosaic_accepts_its_kernels(settings):
    model = ModelConfig(**settings)
    assert resolve_attention_backend(model, dtype="bfloat16", platform="gpu", sequence_length=2048) == "mosaic"


@pytest.mark.parametrize("settings", [
    dict(MOSAIC_TAYLOR, attention_degree=None),
    dict(MOSAIC_TAYLOR, attention_coefficients="learned"),
    dict(MOSAIC_TAYLOR, attention_fixed_coefficients=(0.5, 1.0, 0.5)),
    dict(MOSAIC_TAYLOR, attention_normalization=False),
    dict(MOSAIC_TAYLOR, attention_mechanism="gated_deltanet", attention_degree=None),
    dict(MOSAIC_QUERY, attention_degree=5),
    dict(MOSAIC_QUERY, attention_degree=1, attention_coefficient_activation="identity"),
    dict(MOSAIC_TAYLOR, qk_head_dim=48),
    dict(MOSAIC_TAYLOR, value_head_dim=128),
])
def test_mosaic_rejects_recipes_without_a_kernel(settings):
    with pytest.raises(ValueError, match="mosaic"):
        ModelConfig(**settings)


@pytest.mark.parametrize("dtype,platform,capability,length,error", [
    ("bfloat16", "cpu", 9, 2048, "requires a GPU"),
    ("float32", "gpu", 9, 2048, "bfloat16"),
    ("bfloat16", None, 8, 2048, "H100"),
    ("bfloat16", None, 9, 1920, None),
    ("bfloat16", None, 10, 1920, "divisible by 256"),
    ("bfloat16", None, 9, 2000, "divisible by 128"),
])
def test_mosaic_resolution_checks_the_allocation(monkeypatch, dtype, platform, capability, length, error):
    from scaffolds.transformer.attention.backends import selection

    monkeypatch.setattr(selection, "_runtime_platform", lambda: "gpu")
    monkeypatch.setattr(selection, "_cuda_compute_capability", lambda: capability)
    model = ModelConfig(**MOSAIC_TAYLOR)
    resolve = lambda: resolve_attention_backend(model, dtype=dtype, platform=platform, sequence_length=length)
    if error is None:
        assert resolve() == "mosaic"
    else:
        with pytest.raises(ValueError, match=error):
            resolve()


@pytest.mark.parametrize("settings,dtype,capability,length,expected", [
    (MOSAIC_TAYLOR, "bfloat16", 9, 2048, "mosaic"),
    (MOSAIC_QUERY, "bfloat16", 10, 2048, "mosaic"),
    (MOSAIC_TAYLOR, "float32", 9, 2048, "fused"),
    (MOSAIC_TAYLOR, "bfloat16", 8, 2048, "fused"),
    (MOSAIC_TAYLOR, "bfloat16", 10, 1920, "fused"),
    (dict(MOSAIC_TAYLOR, attention_coefficients="learned"), "bfloat16", 9, 2048, "fused"),
    (dict(MOSAIC_TAYLOR, qk_head_dim=48), "bfloat16", 9, 2048, "fused"),
    (dict(MOSAIC_TAYLOR, attention_degree=None), "bfloat16", 9, 2048, "cudnn"),
])
def test_auto_prefers_mosaic_where_a_kernel_and_the_gpu_support_it(monkeypatch, settings, dtype, capability, length,
                                                                    expected):
    from scaffolds.transformer.attention.backends import selection

    monkeypatch.setattr(selection, "_runtime_platform", lambda: "gpu")
    monkeypatch.setattr(selection, "_cuda_compute_capability", lambda: capability)
    model = ModelConfig(**dict(settings, attention_backend="auto"))
    assert resolve_attention_backend(model, dtype=dtype, sequence_length=length) == expected


def test_mosaic_requires_bfloat16_training():
    from scaffolds.transformer.configs import validate_config

    config = Config(model=ModelConfig(**MOSAIC_TAYLOR))
    config.train.dtype = "float32"
    with pytest.raises(ValueError, match="bfloat16"):
        validate_config(config)


@pytest.mark.parametrize("settings,precision", [
    (MOSAIC_QUERY, "bf16"), (dict(MOSAIC_QUERY, attention_degree=1), "bf16"),
    (MOSAIC_TAYLOR, "bf16"), (dict(MOSAIC_TAYLOR, attention_degree=16), "bf16"),
    (dict(MOSAIC_TAYLOR, attention_degree=1), "bf16x3"), (dict(MOSAIC_TAYLOR, attention_degree=3), "bf16x3"),
])
def test_mosaic_weight_precision_is_chosen_per_setup_and_recorded(settings, precision):
    config = Config(model=ModelConfig(**settings))
    config.train.dtype = "bfloat16"
    identity = attention_identity(config)
    assert identity["attention_implementation"].startswith("pallas_mosaic_gpu")
    assert identity["attention_weight_precision"] == precision
