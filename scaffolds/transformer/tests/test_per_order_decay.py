"""Per-order endpoint gates: explicit mathematics, gradients, and matched starts."""
from dataclasses import replace
from functools import partial
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention import forget_gate
from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention
from scaffolds.transformer.attention.parameters import attention_parameter_specs
from scaffolds.transformer.configs.base import validate_config
from scaffolds.transformer.configs import config_to_dict
from scaffolds.transformer.configs.loader import config_dicts_equal
from scaffolds.transformer.models import build_model
from scaffolds.transformer.tests.test_endpoint_decay import _config, _params
from scaffolds.transformer.training.optim import attention_weight_decay_mask
from scaffolds.transformer.training.diagnostics import attention_identity


def inputs(time=9, degree=4, *, width=3, value_width=2):
    rng = np.random.default_rng(12)
    q, k = (rng.normal(size=(2, time, 2, width)) * .4 for _ in range(2))
    v = rng.normal(size=(2, time, 2, value_width))
    c = rng.normal(size=(2, time, 2, degree + 1))
    x = rng.normal(size=(2, time, 7))
    wq, wk = (rng.normal(size=(2, degree + 1, 7)) * .2 for _ in range(2))
    rate = rng.uniform(-1, 1, size=(2, degree + 1, 1))
    bias = rng.uniform(-4, -1, size=rate.shape)
    return tuple(jnp.asarray(a, jnp.float32) for a in (q, k, v, c, x, wq, wk, rate, bias))


def operation(q, k, v, c, x, wq, wk, rate, bias, *, endpoint, diagnostics=False):
    if endpoint:
        query, key = forget_gate.project_gate_inputs(x, wq, bias, wk)
        gate = dict(query_gate_input=query, key_gate_input=key, gate_log_rate=rate)
    else:
        gate = dict(query_log_decay=forget_gate.log_decay(x, wq, rate, bias))
    return blocked_dot_product_attention(q, k, v, attention_degree=c.shape[-1]-1,
        attention_normalization=False, query_polynomial_coefficients=c,
        query_decay_mode="query_distance", query_block_size=128 if q.shape[1] > 128 else 4,
        return_diagnostics=diagnostics, **gate)


def dense(q, k, v, c, x, wq, wk, rate, bias, *, endpoint):
    # Explicit powers and all pairs, independently from the blocked Horner implementation.
    score = jnp.einsum("bthd,bihd->bhti", q, k) / math.sqrt(q.shape[-1])
    query = jnp.einsum("btd,hrd->bhtr", x, wq) + bias[..., 0][None, :, None, :]
    z = query[:, :, :, None, :]
    if endpoint:
        key = jnp.einsum("bid,hrd->bhir", x, wk)
        z = z + key[:, :, None, :, :]
    lam = jnp.exp(rate[..., 0])[None, :, None, None, :] * jnp.logaddexp(0., z)
    distance = jnp.arange(q.shape[1])[:, None] - jnp.arange(q.shape[1])[None, :]
    weights = sum(c[:, :, :, r].transpose(0, 2, 1)[..., None] * score**r
                  * jnp.exp(-jnp.maximum(distance, 0) * lam[..., r]) for r in range(c.shape[-1]))
    weights = jnp.where(distance >= 0, weights, 0.)
    return jnp.einsum("bhti,bihd->bthd", weights, v)


@pytest.mark.parametrize("endpoint", (False, True))
@pytest.mark.parametrize("time,degree", ((1, 1), (9, 4), (13, 6)))
def test_explicit_output_and_every_parameter_gradient(endpoint, time, degree):
    arrays = inputs(time, degree)
    cotangent = jnp.arange(arrays[2].size).reshape(arrays[2].shape) / arrays[2].size
    def evaluate(fn, args):
        out, backward = jax.vjp(fn, *args)
        return out, backward(cotangent.astype(out.dtype))
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        expected = evaluate(partial(dense, endpoint=endpoint), tuple(a.astype(jnp.float64) for a in arrays))
        expected = jax.tree.map(np.asarray, expected)
    with jax.default_matmul_precision("highest"):
        actual = jax.jit(lambda *a: evaluate(partial(operation, endpoint=endpoint), a))(*arrays)
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(got, want, rtol=8e-5, atol=2e-6)
    if time > 1:
        grad_wq = actual[1][5]
        assert np.all(np.linalg.norm(grad_wq, axis=-1) > 0)
        assert not np.allclose(grad_wq[:, 0], grad_wq[:, 1])
        if endpoint:
            assert np.all(np.linalg.norm(actual[1][6], axis=-1) > 0)


@pytest.mark.parametrize("conditioning", ("query", "query_key"))
def test_model_initialization_matches_control_and_decay_masks(conditioning):
    base = _config(attention_forget_gate_conditioning=conditioning)
    config = replace(base, model=replace(base.model, attention_forget_gate_per_order=True))
    validate_config(config)
    control, model = build_model(base.model, 42), build_model(config.model, 42)
    old, new = _params(control), _params(model)
    for path, value in old.items():
        if path[0].startswith("forget_gate_"):
            expected = np.broadcast_to(np.asarray(value)[:, :, None, :], new[path].shape)
            np.testing.assert_array_equal(new[path], expected)
        else:
            np.testing.assert_array_equal(new[path], value)
    tokens = jnp.arange(16).reshape(2, 8) % 32
    np.testing.assert_allclose(model(tokens), control(tokens), rtol=2e-5, atol=2e-6)
    excluded = frozenset(s.name for s in attention_parameter_specs(config.model) if not s.weight_decay)
    mask = {p: bool(v.get_value()) for p, v in nnx.to_flat_state(
        attention_weight_decay_mask(nnx.state(model, nnx.Param), excluded_names=excluded))}
    assert mask[("forget_gate_kernel",)] and mask[("query_polynomial_kernel",)]
    if conditioning == "query_key":
        assert mask[("forget_gate_key_kernel",)]
    for name in ("forget_gate_log_rate", "forget_gate_dt_bias", "query_polynomial_amplitude", "query_polynomial_bias"):
        assert not mask[(name,)]
    identity = attention_identity(config)
    assert identity["attention_forget_gate_per_order"] is True
    assert identity["forget_gate_parameter_count"] == sum(new[p].size for p in new if p[0].startswith("forget_gate_"))


def test_endpoint_diagnostics_keep_each_order_and_exclude_padding():
    args = inputs()
    _, stats = operation(*args, endpoint=True, diagnostics=True)
    q, k, v, c, x, wq, wk, rate, bias = map(np.asarray, args)
    z = np.einsum("btd,hrd->bthr", x, wq)[:, :, None] + np.einsum("bid,hrd->bihr", x, wk)[:, None]
    alpha = -np.exp(rate[..., 0]) * np.logaddexp(0., z + bias[..., 0])
    causal = np.arange(9)[:, None] >= np.arange(9)[None, :]
    for name, reduce in (("mean", np.mean), ("min", np.min), ("max", np.max)):
        np.testing.assert_allclose(stats[f"forget_gate_log_alpha_{name}"], reduce(alpha[:, causal], axis=(0, 1)), rtol=3e-6)


@pytest.mark.parametrize("endpoint", (False, True))
def test_degree_zero_decays_and_diagonal_is_unchanged(endpoint):
    args = list(inputs())
    args[0] = jnp.zeros_like(args[0])
    args[1] = jnp.zeros_like(args[1])
    args[2] = jnp.broadcast_to(jnp.eye(9)[None, :, None, :], (2, 9, 2, 9))
    args[3] = jnp.zeros_like(args[3]).at[..., 0].set(1.)
    weights = operation(*args, endpoint=endpoint)[0, :, 0]
    np.testing.assert_array_equal(np.diag(weights), np.ones(9))
    np.testing.assert_array_equal(np.triu(weights, 1), np.zeros((9, 9)))
    assert 0 < weights[8, 0] < 1


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
def test_invalid_order_gate_poisoning_is_causal(bad):
    q, k, v, c, x, wq, wk, rate, bias = inputs()
    query, key = forget_gate.project_gate_inputs(x, wq, bias, wk)
    key = key.at[0, 3, 0, 2].set(bad)
    out = blocked_dot_product_attention(q, k, v, attention_degree=4, attention_normalization=False,
        query_polynomial_coefficients=c, query_gate_input=query, key_gate_input=key, gate_log_rate=rate,
        query_decay_mode="query_distance", query_block_size=4)
    invalid = ~np.isfinite(out).all(axis=-1)
    expected = np.zeros(invalid.shape, bool)
    expected[0, 3:, 0] = True
    np.testing.assert_array_equal(invalid, expected)


@pytest.mark.parametrize("changes", ({"attention_backend": "fused"}, {"attention_forget_gate_mode": "cumulative"},
    {"attention_forget_gate": False}, {"attention_forget_gate_rate_activation": "sigmoid"}))
def test_unsupported_per_order_recipes_rejected(changes):
    with pytest.raises(ValueError):
        validate_config(_config(attention_forget_gate_per_order=True, **changes))


def test_mosaic_per_order_limits_fit_blackwell_forward_and_dq():
    """Largest degrees whose B300 kernels lower: the forward and dQ also hold every key's
    per-order vectors, which leaves the query-key gate little room at d_K = 128."""
    from scaffolds.transformer.attention.backends.mosaic.api import query_degree_supported

    def limit(gate, width, time):
        return max(m for m in range(1, 13) if query_degree_supported(m, width, time, gate=gate, per_order=True))
    assert (limit("query", 64, 2048), limit("query_key", 64, 2048)) == (7, 6)
    assert (limit("query", 128, 512), limit("query_key", 128, 2048)) == (10, 1)


def test_historical_checkpoint_defaults_keep_shared_gates():
    current = config_to_dict(_config())
    old = config_to_dict(_config())
    old["model"].pop("attention_forget_gate_per_order")
    assert config_dicts_equal(current, old)
    current["model"]["attention_forget_gate_per_order"] = True
    assert not config_dicts_equal(current, old)


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
def test_nonfinite_query_projection_is_not_hidden_by_softplus(bad):
    _, _, _, _, x, wq, _, rate, bias = inputs()
    alpha = forget_gate.log_decay(x, wq, rate, bias.at[0, 2, 0].set(bad))
    assert np.isnan(alpha[:, :, 0, 2]).all()


@pytest.mark.gpu
@pytest.mark.parametrize("endpoint", (False, True), ids=("query", "query_key"))
def test_gpu_per_order_forward_and_backward(endpoint):
    if jax.default_backend() != "gpu":
        pytest.skip("GPU qualification")
    arrays = list(inputs(2048, 4, width=64, value_width=64))
    arrays[:3] = [a.astype(jnp.bfloat16) for a in arrays[:3]]
    cotangent = jnp.asarray(np.random.default_rng(5).normal(size=arrays[2].shape), jnp.bfloat16)
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        out, backward = jax.vjp(partial(dense, endpoint=endpoint), *[a.astype(jnp.float64) for a in arrays])
        expected = jax.tree.map(np.asarray, (out, backward(cotangent.astype(jnp.float64))))
    def evaluate(*args):
        out, backward = jax.vjp(partial(operation, endpoint=endpoint), *args)
        return out, backward(cotangent)
    with jax.default_matmul_precision("highest"):
        actual = jax.jit(evaluate)(*arrays)
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        got = np.asarray(got, np.float64)
        assert np.isfinite(got).all()
        error = np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-12)
        assert error < .01, error
    test_model_initialization_matches_control_and_decay_masks("query_key" if endpoint else "query")


# Mosaic GPU kernels for per-order gates: the same mathematics fused into the causal tiles.
def _requires_mosaic_gpu():
    if jax.default_backend() != "gpu" or jax.devices()[0].compute_capability not in ("9.0", "10.0", "10.3"):
        pytest.skip("Requires an H100 or B300 allocation")


def mosaic_operation(q, k, v, c, x, wq, wk, rate, bias, *, endpoint, per_order=True):
    """The production dispatch: gate projections -> attention_core on the Mosaic backend."""
    from scaffolds.transformer.attention.core import attention_core

    if not per_order:  # the shared gate of the same parameters' order zero
        wq, wk, rate, bias = wq[:, 0], wk[:, 0], rate[:, 0], bias[:, 0]
    if endpoint:
        query, key = forget_gate.project_gate_inputs(x, wq, bias, wk)
        gate = dict(query_gate_input=query, key_gate_input=key, gate_log_rate=rate)
    else:
        gate = dict(query_log_decay=forget_gate.log_decay(x, wq, rate, bias))
    return attention_core(q, k, v, backend="mosaic", attention_degree=c.shape[-1] - 1, attention_normalization=False,
                          query_polynomial_coefficients=c, attention_coefficient_activation="tanh",
                          query_decay_mode="query_distance", **gate)


def _mosaic_inputs(time=2048, *, key=True):
    arrays = list(inputs(time, 4, width=64, value_width=64))
    arrays[3] = arrays[3] / jnp.asarray([math.factorial(r) for r in range(5)], jnp.float32)
    if not key:
        arrays[6] = jnp.zeros_like(arrays[6])
    arrays[:3] = [a.astype(jnp.bfloat16) for a in arrays[:3]]
    cotangent = jnp.asarray(np.random.default_rng(5).normal(size=arrays[2].shape), jnp.bfloat16)
    return arrays, cotangent


@pytest.mark.gpu
@pytest.mark.parametrize("endpoint", (False, True), ids=("query", "query_key"))
def test_mosaic_per_order_output_and_every_gradient_match_float64(endpoint):
    _requires_mosaic_gpu()
    arrays, cotangent = _mosaic_inputs(key=endpoint)
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        out, backward = jax.vjp(partial(dense, endpoint=endpoint), *[a.astype(jnp.float64) for a in arrays])
        expected = jax.tree.map(np.asarray, (out, backward(cotangent.astype(jnp.float64))))
    def evaluate(*args):
        out, backward = jax.vjp(partial(mosaic_operation, endpoint=endpoint), *args)
        return out, backward(cotangent)
    actual = jax.jit(evaluate)(*arrays)
    # output, dq, dk, dv at the Mosaic BF16 bounds; coefficients and gate parameters through FP32 sums.
    tolerances = (.006, .025, .025, .015, .01, .01, .01, .01, .01, .01)
    for i, (got, want) in enumerate(zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True)):
        got = np.asarray(got, np.float64)
        assert np.isfinite(got).all(), i
        if not endpoint and i == 7:  # wk is unused by query-only gates
            np.testing.assert_array_equal(got, 0)
            continue
        error = np.linalg.norm(got - want) / np.linalg.norm(want)
        assert error < tolerances[i], (endpoint, i, error)


@pytest.mark.gpu
@pytest.mark.parametrize("endpoint", (False, True), ids=("query", "query_key"))
def test_mosaic_duplicated_orders_reproduce_the_shared_gate_kernel(endpoint):
    """Every order a copy of one gate (the runs' initialization): the per-order kernels equal the
    shared-gate kernels, and each shared parameter's gradient is the sum over its copies."""
    _requires_mosaic_gpu()
    arrays, cotangent = _mosaic_inputs(1024, key=endpoint)
    q, k, v, c, x, wq, wk, rate, bias = arrays
    copies = [jnp.broadcast_to(a[:, :1], a.shape) for a in (wq, wk, rate, bias)]
    per_order = jax.jit(lambda *a: _vjp(partial(mosaic_operation, endpoint=endpoint), a, cotangent))(
        q, k, v, c, x, *copies)
    shared = jax.jit(lambda *a: _vjp(partial(mosaic_operation, endpoint=endpoint, per_order=False), a, cotangent))(
        q, k, v, c, x, *copies)
    observed = (per_order[0], *per_order[1][:5], *(g.sum(axis=1) for g in per_order[1][5:]))
    expected = (shared[0], *shared[1][:5], *(g[:, 0] for g in shared[1][5:]))
    for i, (got, want) in enumerate(zip(observed, expected, strict=True)):
        got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
        norm = np.linalg.norm(want)
        if norm == 0:
            np.testing.assert_array_equal(got, 0)
        else:
            assert np.linalg.norm(got - want) / norm < 2e-3, (endpoint, i)


@pytest.mark.gpu
def test_mosaic_per_order_zero_key_recovers_per_order_query_gates():
    _requires_mosaic_gpu()
    arrays, cotangent = _mosaic_inputs(1024, key=False)
    endpoint = jax.jit(lambda *a: _vjp(partial(mosaic_operation, endpoint=True), a, cotangent))(*arrays)
    query = jax.jit(lambda *a: _vjp(partial(mosaic_operation, endpoint=False), a, cotangent))(*arrays)
    for i, (got, want) in enumerate(zip(jax.tree.leaves(endpoint), jax.tree.leaves(query), strict=True)):
        got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
        if i == 7:  # the key projection's gradient exists only for endpoint gates
            continue
        assert np.linalg.norm(got - want) <= 2e-4 * np.linalg.norm(want), i


@pytest.mark.gpu
@pytest.mark.parametrize("operand", ("query", "key", "rate"))
def test_mosaic_invalid_order_gate_poisons_only_affected_causal_rows(operand):
    _requires_mosaic_gpu()
    from scaffolds.transformer.attention.backends.mosaic.api import per_order_endpoint_decay_attention

    q, k, v, c, x, wq, wk, rate, bias = _mosaic_inputs(512)[0]
    query, key = forget_gate.project_gate_inputs(x, wq, bias, wk)
    if operand == "query":
        query = query.at[0, 300, 0, 2].set(jnp.nan)
    elif operand == "key":
        key = key.at[0, 300, 0, 2].set(jnp.inf)
    else:
        rate = rate.at[0, 2, 0].set(jnp.nan)
    out = per_order_endpoint_decay_attention(q, k, v, c, query, key, rate)
    actual = ~np.isfinite(np.asarray(out, np.float32)).all(axis=-1)
    expected = np.zeros(actual.shape, bool)
    expected[:, {"query": slice(300, 301), "key": slice(300, None), "rate": slice(None)}[operand], 0] = True
    if operand != "rate":
        expected[1] = False  # the change is in batch 0 only
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.parametrize("endpoint", (False, True), ids=("query", "query_key"))
def test_mosaic_per_order_diagonal_is_exactly_unattenuated(endpoint):
    _requires_mosaic_gpu()
    q, k, v, c, x, wq, wk, rate, bias = _mosaic_inputs(512)[0]
    c = jnp.zeros_like(c).at[..., 0].set(1.)  # p = 1
    rate = jnp.full_like(rate, 80.)  # every earlier key underflows in every order
    output = mosaic_operation(q, k, v, c, x, wq, wk, rate, bias, endpoint=endpoint)
    np.testing.assert_array_equal(np.asarray(output, np.float32), np.asarray(v, np.float32))


def _vjp(function, operands, cotangent):
    output, backward = jax.vjp(function, *operands)
    return output, backward(cotangent)


@pytest.mark.gpu
@pytest.mark.parametrize("conditioning", ("query", "query_key"))
def test_mosaic_per_order_model_trains_like_the_float32_blocked_model(conditioning, monkeypatch):
    """End-to-end per-order model on Mosaic in BF16: loss and gradient direction of the FP32 blocked model
    (as test_endpoint_decay's model check, whose tolerances these share)."""
    _requires_mosaic_gpu()
    from scaffolds.transformer.attention.backends.mosaic import api
    from scaffolds.transformer.training import loop
    from scaffolds.transformer.training.state import create_state

    rng = np.random.default_rng(57)
    batch = jnp.asarray(rng.integers(0, 64, size=(2, 257)), jnp.int32)
    amplitude = jnp.asarray(rng.normal(size=(2, 2, 5)) * .5, jnp.float32)
    key_kernel = jnp.asarray(rng.normal(size=(2, 2, 5, 128)) * .1, jnp.float32)
    gate_kernel = jnp.asarray(rng.normal(size=(2, 2, 5, 128)) * .05, jnp.float32)  # orders differ
    results = {}
    for backend, dtype, precision in (("blocked", "float32", None), ("blocked", "bfloat16", None),
                                      ("mosaic", "bfloat16", "bf16"), ("mosaic", "bfloat16", "bf16x3")):
        monkeypatch.setattr(api, "default_precision", lambda family, degree, precision=precision: precision)
        base = _config(vocab_size=64, max_seq_len=256, hidden_size=128, qk_head_dim=64, value_head_dim=64,
                       intermediate_size=256, attention_backend=backend,
                       attention_coefficient_projection_initial_std=128 ** -.5,
                       attention_forget_gate_conditioning=conditioning)
        config = replace(base, model=replace(base.model, attention_forget_gate_per_order=True))
        config.train.dtype, config.data.sequence_length = dtype, batch.shape[1]
        state = create_state(config, jax.random.PRNGKey(3))
        state.params["query_polynomial_amplitude"][...] = amplitude
        state.params["forget_gate_kernel"][...] = state.params["forget_gate_kernel"][...] + gate_kernel
        if conditioning == "query_key":
            state.params["forget_gate_key_kernel"][...] = key_kernel
        with jax.default_matmul_precision("highest" if dtype == "float32" else "default"):
            loss, gradients = jax.jit(lambda *a: loop.loss_and_grads(*a))(state, batch, jax.random.PRNGKey(0))
        results[backend, dtype, precision] = float(loss), {path: np.asarray(value[...], np.float64)
                                                          for path, value in nnx.to_flat_state(gradients)}
    reference_loss, reference = results["blocked", "float32", None]
    bf16 = results["blocked", "bfloat16", None][1]
    loss, actual = results["mosaic", "bfloat16", "bf16"]
    assert math.isfinite(loss) and abs(loss - reference_loss) < .02
    assert all(np.isfinite(value).all() for value in actual.values())

    def cosine(gradients):
        a, b = (np.concatenate([g[path].ravel() for path in sorted(reference, key=str)]) for g in (gradients, reference))
        return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

    exact = results["mosaic", "bfloat16", "bf16x3"][1]
    assert cosine(exact) >= min(.995, cosine(bf16) - .01), (cosine(exact), cosine(bf16))
    assert cosine(actual) >= min(.995, cosine(bf16) - .03), (cosine(actual), cosine(bf16))
