"""Dense readout scaling preserves Taylor and trains with independent QK head scales."""
from dataclasses import replace
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer import acceptance
from scaffolds.transformer.attention.input_dependent import project_coefficients
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict
from scaffolds.transformer.models import build_model
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.checkpointing import (
    make_checkpoint_manager, restore_latest_checkpoint, save_checkpoint,
)
from scaffolds.transformer.training.diagnostics import (
    attention_identity, qk_norm_head_scale_report, query_coefficient_report,
)
from scaffolds.transformer.training.state import create_state


def _config(degree, **changes):
    model = dict(vocab_size=16, max_seq_len=8, num_layers=1, hidden_size=8,
                 num_heads=2, num_kv_heads=2, qk_head_dim=2, value_head_dim=4,
                 intermediate_size=12, attention_degree=degree, attention_backend="blocked",
                 attention_normalization=False, attention_coefficients="learned",
                 attention_coefficient_sharing="per_head", attention_coefficient_conditioning="query",
                 attention_coefficient_activation="tanh", attention_coefficient_readout="dense",
                 attention_coefficient_feature_count=9, attention_coefficient_readout_scale=1 / 3,
                 attention_coefficient_amplitude_initial=0.0,
                 attention_coefficient_projection_initial_std=8 ** -.5,
                 attention_coefficient_amplitude_scale="factorial", attention_coefficient_scale="factorial",
                 attention_output_scaling="head_rms_norm", qk_norm=True, qk_norm_type="rms_norm",
                 qk_norm_sharing="per_layer", qk_norm_head_scale=True,
                 initialization="transformer_truncated_reference")
    config = Config(model=ModelConfig(**{**model, **changes}))
    config.data.sequence_length = config.model.max_seq_len + 1
    config.data.synthetic_vocab_size = config.model.vocab_size
    config.optim.total_steps, config.optim.warmup_steps = 4, 0
    return config


@pytest.mark.parametrize("degree", (4, 8))
@pytest.mark.parametrize("factorial", (False, True))
def test_scaled_readout_and_all_derivatives_match_independent_float64(degree, factorial):
    rng = np.random.default_rng(13 + degree)
    count = degree + 1
    arrays = (rng.normal(size=(2, 7, 8)), rng.normal(size=(2, 8, 9)) * .3,
              rng.normal(size=(2, count)), rng.normal(size=(2, count, 9)) * .2)
    operands = tuple(jnp.asarray(x, jnp.float32) for x in arrays)
    cotangent = rng.normal(size=(2, 7, 2, count))
    scaling = "factorial" if factorial else "none"
    factors = np.array([math.factorial(r) for r in range(count)]) if factorial else np.ones(count)

    def reference(x, w, b, v):
        features = jnp.tanh(jnp.einsum("btd,hdj->bthj", x, w))
        return b + jnp.einsum("bthj,hrj->bthr", features, v / (3 * jnp.asarray(factors)[None, :, None]))

    def actual(x, w, b, v, scale=1 / 3):
        return project_coefficients(x, w, b, mixing=v, activation="tanh",
                                    amplitude_scale=scaling, readout_scale=scale)

    with jax.enable_x64():
        inputs64 = tuple(jnp.asarray(x, jnp.float64) for x in arrays)
        wanted = np.asarray(reference(*inputs64))
        gradients64 = jax.grad(lambda *x: jnp.sum(reference(*x) * jnp.asarray(cotangent)),
                               argnums=(0, 1, 2, 3))(*inputs64)
        gradients64 = [np.asarray(g) for g in gradients64]
    np.testing.assert_allclose(jax.jit(actual)(*operands), wanted, rtol=3e-5, atol=3e-6)
    gradient = lambda f: jax.grad(lambda *x: jnp.sum(f(*x) * jnp.asarray(cotangent, jnp.float32)),
                                 argnums=(0, 1, 2, 3))(*operands)
    gradients = gradient(actual)
    unscaled = gradient(lambda *x: actual(*x, scale=1.0))
    for name, got, expected, old in zip(("x", "W", "b", "V"), gradients, gradients64, unscaled):
        np.testing.assert_allclose(got, expected, rtol=8e-5, atol=3e-6, err_msg=name)
        np.testing.assert_allclose(got, old if name == "b" else old / 3,
                                   rtol=8e-5, atol=3e-6, err_msg=name)
    zero = jnp.zeros_like(operands[-1])
    np.testing.assert_array_equal(actual(*operands[:-1], zero), np.broadcast_to(arrays[2].astype(np.float32), wanted.shape))
    default = project_coefficients(*operands[:3], mixing=operands[3], activation="tanh", amplitude_scale=scaling)
    np.testing.assert_array_equal(actual(*operands, scale=1.0), default)


@pytest.mark.parametrize("bad", (0, -1, float("nan"), float("inf"), True))
def test_project_rejects_invalid_readout_scale(bad):
    with pytest.raises(ValueError, match="scale"):
        project_coefficients(jnp.ones((1, 2, 3)), jnp.ones((1, 3, 5)), jnp.ones((1, 5)),
                             mixing=jnp.ones((1, 5, 5)), activation="tanh", readout_scale=bad)


def test_scale_cannot_silently_apply_to_diagonal_or_identity():
    x, w, b = jnp.ones((1, 2, 3)), jnp.ones((1, 3, 5)), jnp.ones((1, 5))
    with pytest.raises(ValueError, match="dense"):
        project_coefficients(x, w, b, amplitude=b, activation="tanh", readout_scale=1 / 3)
    with pytest.raises(ValueError, match="dense"):
        project_coefficients(x, w, b, readout_scale=1 / 3)


@pytest.mark.parametrize("degree", (4, 8))
def test_scaled_dense_and_head_scale_start_exactly_at_taylor(degree):
    config = _config(degree)
    changed = build_model(config.model, 19)
    legacy = build_model(replace(config.model, attention_coefficient_readout_scale=1.0,
                                  qk_norm_head_scale=False), 19)
    fixed = build_model(replace(config.model, attention_coefficients="fixed",
                                 attention_coefficient_readout="diagonal", attention_coefficient_feature_count=None,
                                 attention_coefficient_readout_scale=1.0, attention_coefficient_conditioning="static",
                                 attention_coefficient_activation="identity", attention_coefficient_scale="none",
                                 attention_coefficient_amplitude_scale="none",
                                 attention_coefficient_amplitude_initial=1.0,
                                 attention_coefficient_projection_initial_std=0.0, qk_norm_head_scale=False), 19)
    tokens = jnp.arange(16).reshape(2, 8) % 16
    np.testing.assert_array_equal(changed(tokens), legacy(tokens))
    # The fixed-polynomial and general query-coefficient reference graphs have
    # different FP32 accumulation order; coefficient initialization is exact.
    np.testing.assert_allclose(changed(tokens), fixed(tokens), rtol=1e-6, atol=3e-8)
    np.testing.assert_array_equal(changed.query_polynomial_mixing[...], 0)
    np.testing.assert_array_equal(changed.qk_norm_head_log_scale[...], 0)


def test_diagnostics_describe_actual_scaled_mixing_and_head_temperatures():
    config = _config(4)
    state = create_state(config, jax.random.PRNGKey(3))
    state.params["query_polynomial_mixing"][...] = jnp.full((1, 2, 5, 9), 6.0)
    state.params["qk_norm_head_log_scale"][...] = jnp.log(jnp.asarray([[[2.0], [.5]]]))
    raw = [{f"query_coefficient_{stat}": np.ones((2, 5)) for stat in ("mean", "std", "min", "max")}]
    report = query_coefficient_report(state.params, raw, amplitude_scale="factorial",
                                      coefficient_scale="factorial", readout_scale=1 / 3)
    factors = np.array([1, 1, 2, 6, 24])
    np.testing.assert_allclose(report["effective_modulation_mixing"],
                               np.broadcast_to(2 / factors[:, None], (1, 2, 5, 9)), rtol=0, atol=0)
    np.testing.assert_allclose(report["effective_projection_bias"], np.broadcast_to(1 / factors, (1, 2, 5)))
    assert report["readout_scale"] == 1 / 3 and report["finite"]
    scales = qk_norm_head_scale_report(state.params)
    np.testing.assert_allclose(scales["effective"], [[[2.0], [.5]]], rtol=1e-7)
    identity = attention_identity(config)
    assert identity["attention_coefficient_readout_scale"] == 1 / 3
    assert "0.333333333 * sum_j" in identity["attention_formula"]
    assert "exp(gamma_h)" in identity["attention_score_formula"]
    assert identity["qk_norm_head_scale"] is True
    assert identity["qk_norm_parameter_count"] == 2 * 1 * 2 + 1 * 2
    assert identity["qk_norm_gain_shared_across_heads"] is True
    state.params["qk_norm_head_log_scale"][...] = jnp.full((1, 2, 1), jnp.inf)
    assert not qk_norm_head_scale_report(state.params)["finite"]


@pytest.mark.parametrize("degree", (4, 8))
def test_joint_parameters_train_and_checkpoint_preserves_next_update(degree, tmp_path):
    config = _config(degree)
    config.train.workdir = tmp_path
    config.optim.grad_clip_norm = .01
    initial = create_state(config, jax.random.PRNGKey(0))
    batch = jnp.arange(18).reshape(2, 9) % 16
    step = jax.jit(lambda state, rng: loop.train_step(state, batch, rng, check_all_gradients=True))
    first, loss, _ = step(initial, jax.random.PRNGKey(1))
    assert np.isfinite(loss)
    for name in ("query_polynomial_mixing", "qk_norm_head_log_scale"):
        assert np.linalg.norm(np.asarray(first.params[name][...] - initial.params[name][...])) > 0
    second, _, _ = step(first, jax.random.PRNGKey(2))
    assert np.linalg.norm(np.asarray(second.params["query_polynomial_kernel"][...]
                                    - first.params["query_polynomial_kernel"][...])) > 0
    manager = make_checkpoint_manager(config)
    try:
        save_checkpoint(manager, 1, first, metrics={"validation_nll": float(loss)})
        manager.wait_until_finished()
        restored, restored_step = restore_latest_checkpoint(manager, create_state(config, jax.random.PRNGKey(0)))
        assert restored_step == 1
        resumed, _, _ = step(restored, jax.random.PRNGKey(2))
    finally:
        manager.close()
    for actual, expected in zip(jax.tree.leaves(resumed), jax.tree.leaves(second), strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_acceptance_requires_joint_checks_on_cpu_and_gpu():
    config = config_to_dict(_config(4))
    spec = acceptance.AcceptanceSpec(entry_module="tests", cpu_tests=(), gpu_tests=())
    for cpu in (True, False):
        paths = acceptance.test_paths(spec, config, resolved_backend="mosaic", cpu=cpu)
        assert acceptance.DENSE_MIXING_SCALE_TESTS in paths
        assert (acceptance.QK_HEAD_SCALE_TESTS in paths) == cpu
    facts = acceptance.matches_softmax_backbone_initialization(_config(4))
    assert facts["mechanism_parameters"]["qk_norm_head_log_scale"]["shape"] == [1, 2, 1]
    assert facts["mechanism_parameters"]["query_polynomial_mixing"]["shape"] == [1, 2, 5, 9]


@pytest.mark.gpu
@pytest.mark.parametrize("degree", (4, 8))
@pytest.mark.parametrize("mixing_std", (0.0, .15))
def test_gpu_joint_scaled_dense_model_forward_and_gradients_match_reference(degree, mixing_std):
    """Production composition against FP32 and BF16 blocked references, including both new parameters."""
    if jax.default_backend() != "gpu" or not jax.devices()[0].device_kind.startswith(("NVIDIA H100", "NVIDIA B300")):
        pytest.skip("Requires an H100 or B300 allocation")
    results = {}
    rng = np.random.default_rng(21 + degree)
    mixing = rng.normal(size=(2, 2, degree + 1, 9)) * mixing_std
    batch = jnp.asarray(rng.integers(0, 64, size=(2, 257)), jnp.int32)
    head_scales = jnp.log(jnp.asarray([[[.8], [1.2]], [[1.1], [.9]]], jnp.float32))
    for backend, dtype in (("blocked", "float32"), ("blocked", "bfloat16"), ("mosaic", "bfloat16")):
        config = _config(degree, vocab_size=64, max_seq_len=256, num_layers=2, hidden_size=128,
                         qk_head_dim=64, value_head_dim=64, intermediate_size=256,
                         attention_backend=backend, attention_coefficient_projection_initial_std=128 ** -.5)
        config.train.dtype = dtype
        state = create_state(config, jax.random.PRNGKey(3))
        state.params["query_polynomial_mixing"][...] = jnp.asarray(mixing, jnp.float32)
        state.params["qk_norm_head_log_scale"][...] = head_scales
        with jax.default_matmul_precision("highest" if dtype == "float32" else "default"):
            loss, gradients = jax.jit(loop.loss_and_grads)(state, batch, jax.random.PRNGKey(0))
        results[backend, dtype] = float(loss), {path: np.asarray(value[...], np.float64)
                                               for path, value in nnx.to_flat_state(gradients)}
    (reference_loss, reference), (_, bf16), (loss, mosaic) = (
        results["blocked", "float32"], results["blocked", "bfloat16"], results["mosaic", "bfloat16"])
    assert math.isfinite(loss) and abs(loss - reference_loss) < .02
    for path, expected in reference.items():
        assert np.isfinite(mosaic[path]).all(), path
        scale = np.linalg.norm(expected)
        if scale == 0:
            np.testing.assert_array_equal(mosaic[path], 0, err_msg=str(path))
            continue
        error = np.linalg.norm(mosaic[path] - expected) / scale
        rounding = np.linalg.norm(bf16[path] - expected) / scale
        assert error <= max(2.5 * rounding, .03), (path, error, rounding)
    for name in ("query_polynomial_mixing", "qk_norm_head_log_scale"):
        assert np.linalg.norm(mosaic[(name,)]) > 0
    if mixing_std:
        assert np.linalg.norm(mosaic[("query_polynomial_kernel",)]) > 0
