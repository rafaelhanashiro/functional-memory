"""Transformed polynomial inputs (attention/score_transform.py): the blocked scan and causal strips against an
independent FP64 oracle with every gradient (through softplus and |s|, signed coefficients, very negative and
very positive scores, exact zeros, masked entries), the softplus start, the identity's unchanged programs,
dispatch and refusals, configuration and allocation, the model start, and diagnostics."""
from dataclasses import replace
from functools import partial
import math
from pathlib import Path
import re
from types import SimpleNamespace

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from scaffolds.transformer import acceptance
from scaffolds.transformer.attention import score_transform
from scaffolds.transformer.attention.backends import reference
from scaffolds.transformer.attention.backends.selection import resolve_attention_backend
from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.attention.input_dependent import project_coefficients, query_polynomial_attention
from scaffolds.transformer.attention.multihead import MultiHeadAttention
from scaffolds.transformer.attention.parameters import attention_parameter_count, attention_parameter_specs
from scaffolds.transformer.attention.score_bias import HISTOGRAM_EDGES
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict, load_config, validate_config
from scaffolds.transformer.configs.loader import config_dicts_equal
from scaffolds.transformer.models import build_model, model_parameter_count
from scaffolds.transformer.training.checkpointing import state_sha256
from scaffolds.transformer.training.diagnostics import attention_diagnostics, attention_identity, fixed_diagnostic_sample
from scaffolds.transformer.training.state import create_state
from scaffolds.transformer.training.tracking import run_tags

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "experiments" / "transformer_pp" / "runs"
# The query polynomial m = 8 with RoPE and coefficient-W decay at the 12-layer width-640 shape.
CONTROL, CONTROL_PARAMETERS = "r95a5691d293f", 99_678_056
TRANSFORMS = ("softplus", "abs")
KINDS = ("mixed", "negative", "positive")


def _phi(transform, scores):
    """The oracle's own forms: logaddexp (JAX's softplus) and s sign(s), whose derivative is sign(s)."""
    return jnp.logaddexp(0.0, scores) if transform == "softplus" else scores * jnp.sign(scores)


def _config(transform="softplus", start=None, **changes):
    model = dict(vocab_size=32, max_seq_len=16, num_layers=2, hidden_size=16, num_heads=2, num_kv_heads=2,
                 qk_head_dim=8, value_head_dim=8, intermediate_size=32, attention_backend="blocked",
                 attention_blocked_implementation="strips", attention_degree=4, attention_normalization=False,
                 attention_coefficients="learned", attention_coefficient_sharing="per_head",
                 attention_coefficient_conditioning="query", attention_coefficient_activation="tanh",
                 attention_coefficient_amplitude_initial=0., attention_coefficient_projection_initial_std=16 ** -.5,
                 attention_coefficient_amplitude_scale="factorial", attention_coefficient_scale="factorial",
                 attention_coefficient_kernel_weight_decay=True, attention_output_scaling="head_rms_norm",
                 qk_norm=True, qk_norm_type="rms_norm", qk_norm_sharing="per_layer", qk_norm_head_scale=True,
                 rms_norm_fp32=True, initialization="transformer_truncated_reference",
                 attention_score_transform=transform, attention_coefficient_constant_initial=start)
    config = Config(model=ModelConfig(**{**model, **changes}))
    config.data.sequence_length, config.data.synthetic_vocab_size = 17, 32
    config.data.batch_size = config.train.micro_batch_size = 2
    config.optim.total_steps, config.optim.warmup_steps = 4, 0
    config.optim.learning_rate = config.optim.min_learning_rate = 1e-3
    return config


def _control(**changes):
    """The control recipe at a small shape (its every other setting kept), on the blocked strips."""
    config = load_config(RUNS / CONTROL / "config.yaml")
    return replace(config, model=replace(config.model, vocab_size=64, max_seq_len=16, num_layers=2, hidden_size=32,
                                         num_heads=2, num_kv_heads=2, intermediate_size=64,
                                         attention_backend="blocked", attention_blocked_implementation="strips",
                                         **changes))


def _params(model):
    return {path: value[...] for path, value in nnx.to_flat_state(nnx.state(model, nnx.Param))}


def _vjp(function, operands, cotangent):
    output, backward = jax.vjp(function, *operands)
    return output, backward(cotangent)


def _operands(kind, *, time=13, degree=4, seed=5):
    """FP64 operands whose scores s = q.k / 2 (width 4) are exact in BF16, FP32 and FP64, with signed
    coefficients N(0, 1)/r!. "mixed": integer Q and K, each query scaled by 1, 1, 6 or 12, so the scores
    reach +-24 with exact zeros. "negative" and "positive": queries and keys along one direction, every
    causal score in [-30, -1/2] or [1/2, 30]."""
    rng = np.random.default_rng(seed + 100 * KINDS.index(kind) + degree)
    batch, heads = 2, 2
    if kind == "mixed":
        q, k = (rng.integers(-1, 2, size=(batch, time, heads, 4)).astype(np.float64) for _ in range(2))
        q = q * rng.choice([1., 1., 6., 12.], size=(batch, time, heads, 1))
    else:
        direction = np.full(4, .5)  # |u|^2 = 1, so s = a_t b_i / 2
        sign = -1. if kind == "negative" else 1.
        q = sign * rng.integers(1, 11, size=(batch, time, heads, 1)) * direction
        k = rng.integers(1, 7, size=(batch, time, heads, 1)) * direction
    v = rng.normal(size=(batch, time, heads, 3))
    c = rng.normal(size=(batch, time, heads, degree + 1)) / np.array([math.factorial(r) for r in range(degree + 1)])
    return q, k, v, c


def _causal_scores(q, k):
    scores = np.einsum("bthd,bihd->bhti", np.asarray(q, np.float64), np.asarray(k, np.float64)) / 2
    return scores[:, :, np.tril(np.ones((q.shape[1],) * 2, bool))]


def _dense(transform, q, k, v, c):
    """Independent oracle: every pair at once, W = sum_r c_r phi(s)^r with explicit powers, masked by where."""
    time = q.shape[1]
    scores = jnp.einsum("bthd,bihd->bhti", q, k) / math.sqrt(q.shape[-1])
    z = _phi(transform, scores)
    coefficient = jnp.swapaxes(c, 1, 2)[:, :, :, None, :]
    weights = sum(coefficient[..., r] * z ** r for r in range(c.shape[-1]))
    causal = jnp.arange(time)[:, None] >= jnp.arange(time)[None, :]
    return jnp.einsum("bhti,bihd->bthd", jnp.where(causal, weights, 0.), v)


def _reference(transform, q, k, v, c, *, implementation="scan", compute=jnp.float32, block=4, diagnostics=False):
    arguments = dict(attention_degree=c.shape[-1] - 1, attention_normalization=False,
                     query_polynomial_coefficients=c, score_transform=transform)
    if implementation == "strips":
        return reference.strip_dot_product_attention(q, k, v, strip_size=block, **arguments)
    if compute == jnp.float32:
        return reference.blocked_dot_product_attention(q, k, v, query_block_size=block,
                                                       return_diagnostics=diagnostics, **arguments)
    return reference._blocked_scan(q, k, v, compute=compute, query_block_size=block, return_diagnostics=diagnostics,
                                   **arguments)


def _assert_close(actual, expected, tolerance, label):
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    assert actual.shape == expected.shape and np.isfinite(actual).all(), label
    scale = max(np.abs(expected).max(), 1e-300)
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance * scale, err_msg=label)


def _assert_gradients(actual, expected, tolerance, label):
    """Output, dQ, dK and dV as whole arrays; the coefficient gradient order by order, so that no order's
    error hides behind a larger order's."""
    for got, want, name in zip(actual[:4], expected[:4], ("output", "dq", "dk", "dv"), strict=True):
        _assert_close(got, want, tolerance, f"{label} {name}")
    for order in range(expected[4].shape[-1]):
        _assert_close(actual[4][..., order], expected[4][..., order], tolerance, f"{label} dc order {order}")


@pytest.mark.parametrize("time,degree", ((1, 1), (13, 4), (11, 8)))
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("transform", TRANSFORMS)
def test_reference_matches_float64_oracle_with_every_gradient(transform, kind, time, degree):
    """The scan's code and the strips in FP64 (to 1e-11) and the public FP32 scan and strips (to 3e-5) against
    the oracle: the output, dQ, dK, dV and each order's coefficient gradient, in 4-query blocks and strips."""
    arrays = _operands(kind, time=time, degree=degree)
    scores = _causal_scores(arrays[0], arrays[1])
    if kind == "mixed" and time > 1:
        assert (scores == 0).any() and scores.min() <= -12 and scores.max() >= 12
    else:
        assert (scores < 0).all() if kind == "negative" else (scores > 0).all() if kind == "positive" else True
    cotangent = np.random.default_rng(7).normal(size=arrays[2].shape)
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        wide = [jnp.asarray(array, jnp.float64) for array in arrays]
        expected = jax.tree.map(np.asarray, _vjp(partial(_dense, transform), wide, jnp.asarray(cotangent)))
        expected = [expected[0], *expected[1]]
        for implementation in ("scan", "strips"):
            exact = jax.jit(lambda *a: _vjp(partial(_reference, transform, implementation=implementation,
                                                    compute=jnp.float64), a, jnp.asarray(cotangent)))(*wide)
            _assert_gradients([exact[0], *exact[1]], expected, 1e-11, f"{implementation} FP64")
    with jax.default_matmul_precision("highest"):
        narrow = [jnp.asarray(array, jnp.float32) for array in arrays]
        for implementation in ("scan", "strips"):
            actual = jax.jit(lambda *a: _vjp(partial(_reference, transform, implementation=implementation), a,
                                             jnp.asarray(cotangent, jnp.float32)))(*narrow)
            assert actual[0].dtype == jnp.float32
            _assert_gradients([actual[0], *actual[1]], expected, 3e-5, f"{implementation} FP32")


SPECIAL = np.array([-1e30, -1e4, -100., -88., -40., -16.5, -1., -1e-6, -0., 0., 1e-6, 1., 16.5, 40., 88., 100., 1e4,
                    1e30])


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_softplus_and_abs_values_and_derivatives(dtype):
    """Softplus as max(s, 0) + log1p(exp(-|s|)) equals log(1 + exp(s)) (logaddexp) to rounding and is never
    nonfinite for a finite score, with derivative sigmoid(s), 1/2 at zero; |s| has derivative sign(s), 0 at zero.
    Infinite and NaN scores propagate: +inf to +inf, -inf to 0 (a nonfinite score poisons its row anyway), NaN
    to NaN."""
    with jax.enable_x64(dtype == np.float64):
        s = jnp.asarray(SPECIAL, dtype)
        for transform, value, slope in (("softplus", jnp.logaddexp(0., s), jax.nn.sigmoid(s)),
                                        ("abs", jnp.abs(s), jnp.sign(s))):
            function = partial(score_transform.apply, transform=transform)
            observed = function(s)
            assert observed.dtype == dtype and bool(jnp.isfinite(observed).all())
            np.testing.assert_allclose(observed, value, rtol=2 * np.finfo(dtype).eps, atol=0)
            np.testing.assert_array_equal(jax.vmap(jax.grad(function))(s), slope)
            np.testing.assert_array_equal(jax.vjp(function, s)[1](jnp.ones_like(s))[0], slope)
        assert float(jax.grad(partial(score_transform.apply, transform="softplus"))(jnp.asarray(0., dtype))) == .5
        assert float(jax.grad(partial(score_transform.apply, transform="abs"))(jnp.asarray(0., dtype))) == 0.
        special = jnp.asarray([np.inf, -np.inf, np.nan], dtype)
        np.testing.assert_array_equal(score_transform.apply(special, "softplus"), [np.inf, 0., np.nan])
        np.testing.assert_array_equal(score_transform.apply(special, "abs"), [np.inf, np.inf, np.nan])
        assert score_transform.apply(s, "identity") is s
    for unknown in ("relu", None, 1):
        with pytest.raises(ValueError, match="Unknown attention score transform"):
            score_transform.apply(s, unknown)


@pytest.mark.parametrize("implementation", ("scan", "strips"))
@pytest.mark.parametrize("transform", TRANSFORMS)
def test_masked_entries_never_reach_earlier_rows(transform, implementation):
    """Future keys along a direction that only earlier queries read, so that every masked pair of an earlier
    row scores +-1e15 (phi(s)^8 would overflow FP32) while every causal pair stays moderate, and future values
    of 1e20: every earlier row's output and every gradient of the earlier tokens (cotangent on the earlier
    rows) are bitwise unchanged and finite. Masked entries evaluate phi at their zero score and are zeroed
    before any product; a transform of the raw scores would put 0 * inf into these gradients."""
    q, k, v, c = (jnp.asarray(array, jnp.float32) for array in _operands("mixed", time=12, degree=8))
    cut = 6
    q = q.at[:, :cut, :, :2].set(jnp.asarray([1., 0.])).at[:, cut:, :, :2].set(0.)
    sign = jnp.where(jnp.arange(q.shape[1] - cut) % 2 == 0, 1., -1.)[None, :, None]
    k_extreme = k.at[:, cut:, :, 0].set(2e15 * sign).at[:, cut:, :, 1].set(0.)  # +-1e15 with earlier queries
    v_extreme = v.at[:, cut:].set(1e20)
    attend = jax.jit(lambda *a: _vjp(partial(_reference, transform, implementation=implementation), a,
                                     jnp.ones_like(v).at[:, cut:].set(0.)))
    base, extreme = attend(q, k, v, c), attend(q, k_extreme, v_extreme, c)
    masked = np.einsum("bthd,bihd->bhti", np.asarray(q[:, :cut]), np.asarray(k_extreme[:, cut:])) / 2
    assert np.abs(masked).min() >= 1e15 - 1e9 and (masked < 0).any() and (masked > 0).any()
    assert np.isfinite(np.asarray(extreme[0])).all()
    np.testing.assert_array_equal(extreme[0][:, :cut], base[0][:, :cut])
    for got, want in zip(extreme[1], base[1], strict=True):
        assert np.isfinite(np.asarray(got[:, :cut])).all()
        np.testing.assert_array_equal(got[:, :cut], want[:, :cut])


def _counted_strips(monkeypatch, strip_size):
    calls, strips = [], reference.strip_dot_product_attention

    def counted(*args, **kwargs):
        calls.append(True)
        return strips(*args, **{**kwargs, "strip_size": strip_size})

    monkeypatch.setattr(reference, "strip_dot_product_attention", counted)
    return calls, strips


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_attention_core_trains_on_the_strips_and_diagnoses_with_the_scan(transform, monkeypatch):
    """attention_core gives the transform to the strips (counted; three 4-query strips of 9 tokens): output and
    every gradient equal a direct call of the strips bitwise and the scan's to FP32 rounding, and the
    diagnostics, from the scan, are the scan's own, with the transform's measurements."""
    arrays = [jnp.asarray(array, jnp.float32) for array in _operands("mixed", time=9)]
    calls, strips = _counted_strips(monkeypatch, 4)

    def direct(q, k, v, c):
        return strips(q, k, v, strip_size=4, attention_degree=4, attention_normalization=False,
                      query_polynomial_coefficients=c, score_transform=transform)

    def core(implementation, diagnostics=False):
        return lambda q, k, v, c: attention_core(
            q, k, v, backend="blocked", blocked_implementation=implementation, attention_degree=4,
            attention_normalization=False, query_polynomial_coefficients=c, attention_coefficient_activation="tanh",
            return_diagnostics=diagnostics, score_transform=transform)

    cotangent = jnp.asarray(np.random.default_rng(9).normal(size=arrays[2].shape), jnp.float32)
    actual = jax.jit(lambda *a: _vjp(core("strips"), a, cotangent))(*arrays)
    assert len(calls) == 1
    expected = jax.jit(lambda *a: _vjp(direct, a, cotangent))(*arrays)
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(got, want)
    scan = jax.jit(lambda *a: _vjp(core("scan"), a, cotangent))(*arrays)
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(scan), strict=True):
        _assert_close(got, want, 3e-5, "strips against the scan")
    untransformed = jax.jit(lambda *a: attention_core(*a[:3], backend="blocked", attention_degree=4,
                                                      attention_normalization=False, query_polynomial_coefficients=a[3],
                                                      attention_coefficient_activation="tanh"))(*arrays)
    assert np.abs(np.asarray(untransformed) - np.asarray(scan[0])).max() > 1e-2  # the transform is applied
    output, diagnostics = jax.jit(core("strips", diagnostics=True))(*arrays)
    _, expected = jax.jit(core("scan", diagnostics=True))(*arrays)
    np.testing.assert_array_equal(output, actual[0])
    assert "score_transform_input_sum" in expected and diagnostics.keys() == expected.keys()
    for name in expected:
        np.testing.assert_array_equal(diagnostics[name], expected[name], err_msg=name)
    # query_polynomial_attention, attention_core's tanh path, gives the same output.
    np.testing.assert_array_equal(jax.jit(lambda q, k, v, c: query_polynomial_attention(
        q, k, v, c, backend="blocked", score_transform=transform))(*arrays), scan[0])


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_model_layers_train_on_the_strips(transform, monkeypatch):
    """A two-layer model with nonzero amplitudes: on the strips (counted, 4-query strips of 16 tokens) every
    layer's attention runs there, and the loss and every parameter gradient match the scan's in FP32."""
    calls, _ = _counted_strips(monkeypatch, 4)
    tokens = jnp.asarray(np.random.default_rng(5).integers(0, 32, (2, 17)), jnp.int32)
    start = 0. if transform == "softplus" else None
    results = {}
    for implementation in ("scan", "strips"):
        model = build_model(_config(transform, start, attention_blocked_implementation=implementation).model, 0)
        amplitude = model.query_polynomial_amplitude
        amplitude[...] = jnp.asarray(np.random.default_rng(6).normal(size=amplitude[...].shape) * .5, jnp.float32)
        graphdef, params = nnx.split(model, nnx.Param)

        def loss(params):
            logits = nnx.merge(graphdef, params)(tokens[:, :-1]).astype(jnp.float32)
            return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits), tokens[:, 1:, None], axis=-1))

        results[implementation] = jax.jit(jax.value_and_grad(loss))(params)
    assert len(calls) == 2  # one trace, two layers
    (scan_loss, scan_gradients), (strips_loss, strips_gradients) = results["scan"], results["strips"]
    np.testing.assert_allclose(strips_loss, scan_loss, rtol=1e-6)
    for (path, got), want in zip(jax.tree_util.tree_leaves_with_path(strips_gradients),
                                 jax.tree.leaves(scan_gradients), strict=True):
        error = np.linalg.norm(np.asarray(got) - np.asarray(want)) / max(np.linalg.norm(np.asarray(want)), 1e-30)
        assert error < 5e-5, (jax.tree_util.keystr(path), error)
    for name in ("query_polynomial_bias", "query_polynomial_amplitude", "query_polynomial_kernel"):
        assert np.linalg.norm(np.asarray(strips_gradients[name][...])) > 0, name


def _taylor_tail(z, degree):
    """sum_(r > degree) z^r / r!, in FP64, by its series (z <= 2 here)."""
    return sum(z ** r / math.factorial(r) for r in range(degree + 1, degree + 40))


def test_softplus_start_is_the_exponential_up_to_its_taylor_tail():
    """With c_0 = 0 and c_r = 1/r! (the softplus run's start), every weight is sum_(r=1..8) z^r/r!, z =
    softplus(s): exactly e^s minus the Taylor tail sum_(r>=9) z^r/r! (e^z - 1 = e^s), within 1e-4 of e^s
    relative for |s| <= 1 and equal to e^s to rounding for very negative scores; for large scores it is
    T_8(s) - 1 to rounding. Each query's weights are read through one-hot values."""
    degree, time = 8, 64
    scores = np.concatenate([np.linspace(-1, 1, 21), [-3., -5., -8., -12., -20., -30., 2., 3., 30.]])
    count = len(scores)
    # Row 0 of key i holds score s_i through a query along e_0 and keys x e_0: s_ti = q_t[0] k_i[0] / 2.
    q = np.zeros((1, time, 1, 4))
    k = np.zeros((1, time, 1, 4))
    q[0, :, 0, 0] = 2.
    k[0, :count, 0, 0] = scores
    v = np.eye(time)[None, :, None, :]
    coefficients = np.asarray([0.] + [1 / math.factorial(r) for r in range(1, degree + 1)])
    c = np.broadcast_to(coefficients, (1, time, 1, degree + 1))
    with jax.enable_x64():
        weights = np.asarray(reference._blocked_scan(
            *(jnp.asarray(x, jnp.float64) for x in (q, k, v)), compute=jnp.float64, attention_degree=degree,
            attention_normalization=False, query_polynomial_coefficients=jnp.asarray(c, jnp.float64),
            score_transform="softplus", query_block_size=16))[0, -1, 0, :count]
    z = np.logaddexp(0., scores)
    exact = np.exp(scores) - np.array([_taylor_tail(x, degree) if x < 2 else np.nan for x in z])
    small = np.abs(scores) <= 1
    np.testing.assert_allclose(weights[small], exact[small], rtol=1e-14)
    assert (np.abs(weights[small] / np.exp(scores[small]) - 1) < 1e-4).all()
    negative = scores <= -3
    np.testing.assert_allclose(weights[negative], np.exp(scores[negative]), rtol=1e-6)
    np.testing.assert_allclose(weights[negative], exact[negative], rtol=1e-14)
    assert weights[scores == -30.][0] < 1e-12  # a very negative score removes its key
    large = scores >= 30
    taylor = sum(scores[large] ** r / math.factorial(r) for r in range(1, degree + 1))
    np.testing.assert_allclose(weights[large], taylor, rtol=1e-12)


def test_softplus_model_starts_at_zero_constant_and_taylor_orders():
    """The softplus recipe's coefficients at initialization, through the production generator: c_0 = 0 and
    c_r = 1/r! exactly (FP32) for r >= 1, at every token, layer and head (a_r = 0 at the start)."""
    config = _control(attention_score_transform="softplus", attention_coefficient_constant_initial=0.)
    model = build_model(config.model, 0)
    x = jnp.asarray(np.random.default_rng(1).normal(size=(2, 5, 32)), jnp.float32)
    factorial = jnp.asarray([math.factorial(r) for r in range(9)], jnp.float32)
    for layer in range(2):
        coefficients = project_coefficients(
            x, model.query_polynomial_kernel[layer], model.query_polynomial_bias[layer] / factorial,
            amplitude=model.query_polynomial_amplitude[layer], activation="tanh", amplitude_scale="factorial")
        expected = np.asarray([0.] + [np.float32(1) / np.float32(math.factorial(r)) for r in range(1, 9)], np.float32)
        np.testing.assert_array_equal(np.asarray(coefficients), np.broadcast_to(expected, coefficients.shape))


def _jaxpr(function, *arguments):
    return re.sub(r"0x[0-9a-f]+", "0x?", str(jax.make_jaxpr(function)(*arguments)))


def test_the_identity_is_the_untransformed_program():
    """An explicit identity transform traces the same programs as no transform at all (forward, VJP and
    diagnostics through the scan, the strips, attention_core and query_polynomial_attention, and a model's loss
    and gradients); a null constant start declares the same parameters. The change's own check against the
    previous commit's programs is in validation/README.md."""
    q, k, v, c = (jnp.asarray(array, jnp.float32) for array in _operands("mixed", time=9))
    common = dict(attention_degree=4, attention_normalization=False, query_polynomial_coefficients=c)
    functions = {
        "scan": lambda **extra: reference.blocked_dot_product_attention(q, k, v, query_block_size=4, **common, **extra),
        "scan diagnostics": lambda **extra: reference.blocked_dot_product_attention(
            q, k, v, query_block_size=4, return_diagnostics=True, **common, **extra),
        "strips": lambda **extra: reference.strip_dot_product_attention(q, k, v, strip_size=4, **common, **extra),
        "core": lambda **extra: attention_core(q, k, v, backend="blocked", blocked_implementation="strips",
                                               attention_coefficient_activation="tanh", **common, **extra),
        "query": lambda **extra: query_polynomial_attention(q, k, v, c, backend="blocked", **extra),
        "strips vjp": lambda **extra: jax.vjp(lambda q: reference.strip_dot_product_attention(
            q, k, v, strip_size=4, **common, **extra), q)[1](v),
    }
    for name, function in functions.items():
        assert _jaxpr(lambda: function()) == _jaxpr(lambda: function(score_transform="identity")), name
        assert _jaxpr(lambda: function()) != _jaxpr(lambda: function(score_transform="softplus")), name
    # A null start declares Taylor's b exactly; a start replaces b_0 alone.
    plain = attention_parameter_specs(_config("identity").model)
    started = attention_parameter_specs(_config("identity", -0.5).model)
    assert [spec for spec in plain if spec.name != "query_polynomial_bias"] == [
        spec for spec in started if spec.name != "query_polynomial_bias"]
    bias = {spec.name: spec for spec in plain}["query_polynomial_bias"]
    assert bias.initial == tuple(1.0 for _ in range(5))  # stored r! (1/r!) under the factorial scale
    assert {spec.name: spec for spec in started}["query_polynomial_bias"].initial == (-0.5, 1.0, 1.0, 1.0, 1.0)


def test_historical_recipes_keep_their_meaning():
    """Recipes that predate the fields load as the identity with a null start, compare equal to their saved
    configurations without the fields, and record no new identity entry or tag."""
    control = load_config(RUNS / CONTROL / "config.yaml")
    assert (control.model.attention_score_transform, control.model.attention_coefficient_constant_initial) == (
        "identity", None)
    current = config_to_dict(control)
    legacy = {**current, "model": {key: value for key, value in current["model"].items()
                                   if key not in ("attention_score_transform", "attention_coefficient_constant_initial")}}
    assert config_dicts_equal(current, legacy)
    for changes in ({"attention_score_transform": "abs"}, {"attention_coefficient_constant_initial": 0.0}):
        assert not config_dicts_equal({**current, "model": {**current["model"], **changes}}, legacy)
    identity = attention_identity(control)
    assert not {"attention_score_transform", "attention_coefficient_constant_initial"} & identity.keys()
    assert not any(tag.startswith(("score_transform", "coefficient_constant")) for tag in run_tags(control, {}))
    assert model_parameter_count(control.model) == CONTROL_PARAMETERS


@pytest.mark.parametrize("changes", (
    {"attention_score_transform": "relu"},
    {"attention_score_transform": None},
    {"attention_backend": "mosaic"},
    {"attention_backend": "fused"},
    {"attention_backend": "auto"},
    {"attention_coefficient_conditioning": "query_key"},
    {"attention_coefficient_activation": "sigmoid", "attention_coefficient_amplitude_initial": 1.,
     "attention_coefficient_scale": "none"},
    {"attention_coefficient_activation": "affine_sigmoid"},
    {"attention_coefficient_readout": "dense"},
    {"attention_coefficient_orders": (0, 2, 4)},
    {"attention_coefficient_gate": "sigmoid"},
    {"attention_coefficient_conditioning": "static", "attention_coefficient_activation": "identity",
     "attention_coefficient_amplitude_initial": 1., "attention_coefficient_projection_initial_std": 0.,
     "attention_coefficient_amplitude_scale": "none", "attention_coefficient_kernel_weight_decay": False},
    {"positional_encoding": "path", "attention_blocked_implementation": "scan"},
    {"attention_forget_gate": True},
    {"attention_forget_gate": True, "attention_forget_gate_mode": "query_distance", "positional_encoding": "none"},
    {"attention_score_bias": "static"},
))
def test_unqualified_recipes_are_rejected(changes):
    with pytest.raises(ValueError, match="score transform|Unknown attention score transform"):
        validate_config(_config("softplus", **changes))


@pytest.mark.parametrize("start", (float("nan"), float("inf"), True, "zero"))
def test_invalid_constant_starts_are_rejected(start):
    with pytest.raises(ValueError, match="attention_coefficient_constant_initial"):
        _config("identity", start)


@pytest.mark.parametrize("changes", (
    {"attention_coefficient_conditioning": "query_key"},
    {"attention_coefficient_activation": "sigmoid", "attention_coefficient_amplitude_initial": 1.,
     "attention_coefficient_scale": "none"},
    {"attention_coefficient_readout": "dense"},
    {"attention_coefficient_orders": (0, 2, 4)},
    {"attention_coefficient_gate": "sigmoid"},
    {"attention_coefficient_conditioning": "static", "attention_coefficient_activation": "identity",
     "attention_coefficient_amplitude_initial": 1., "attention_coefficient_projection_initial_std": 0.,
     "attention_coefficient_amplitude_scale": "none", "attention_coefficient_kernel_weight_decay": False},
))
def test_constant_starts_apply_to_diagonal_tanh_query_coefficients_alone(changes):
    with pytest.raises(ValueError, match="constant-coefficient start"):
        _config("identity", 0., **changes)


def test_accepted_recipes_and_rejected_training_options():
    for transform in ("identity", *TRANSFORMS):
        for start in (None, 0., -.5):
            for positional in ("rope", "none"):
                validate_config(_config(transform, start, positional_encoding=positional))
    # The constant start alone runs on any backend that runs the polynomial, Mosaic included.
    mosaic = _config("identity", 0., attention_backend="mosaic", qk_head_dim=64, value_head_dim=64)
    mosaic.train.dtype = "bfloat16"
    validate_config(mosaic)
    assert _config("identity", 0).model.attention_coefficient_constant_initial == 0. and isinstance(
        _config("identity", 0).model.attention_coefficient_constant_initial, float)
    for transform, start in (("softplus", None), ("identity", 0.)):
        config = _config(transform, start)
        for optim in (dict(coefficient_release_schedule="all", coefficient_release_step=1),
                      dict(query_coefficient_learning_rate_scale=2.0)):
            with pytest.raises(ValueError, match="score transform or a constant-coefficient start"):
                validate_config(replace(config, optim=replace(config.optim, **optim)))
        continuation = replace(config.continuation, parent_config="parent.yaml", parent_config_sha256="0" * 64,
                               parent_checkpoint="parent/7", parent_checkpoint_sha256="0" * 64, parent_step=1)
        with pytest.raises(ValueError, match="score transform or a constant-coefficient start"):
            validate_config(replace(config, continuation=continuation))
    arguments = dict(backend="blocked", attention_degree=4, attention_normalization=False,
                     attention_coefficient_conditioning="query", attention_coefficient_activation="tanh",
                     rngs=nnx.Rngs(0))
    MultiHeadAttention(16, 2, 2, 8, 8, **arguments, attention_score_transform="softplus")
    MultiHeadAttention(16, 2, 2, 8, 8, **{**arguments, "backend": "mosaic"}, attention_score_transform="softplus")
    for changes in ({"backend": "fused"}, {"backend": "mosaic", "positional_encoding": "none"},
                    {"attention_coefficient_activation": "sigmoid"}, {"attention_forget_gate": True},
                    {"positional_encoding": "path"}):
        with pytest.raises(ValueError, match="score transform"):
            MultiHeadAttention(16, 2, 2, 8, 8, **{**arguments, **changes}, attention_score_transform="softplus")
    with pytest.raises(ValueError, match="Unknown attention score transform"):
        MultiHeadAttention(16, 2, 2, 8, 8, **arguments, attention_score_transform="square")
    model = build_model(_config("abs").model, 0)
    for options in ({"return_coefficients": True}, {"query_coefficients": jnp.zeros((2, 2, 8, 2, 5))}):
        with pytest.raises(ValueError, match="score transform"):
            model(jnp.zeros((2, 8), jnp.int32), **options)


def test_backend_resolution_never_falls_back():
    model = _config("softplus").model
    assert resolve_attention_backend(model, platform="gpu") == resolve_attention_backend(model, platform="cpu") == "blocked"
    for request in ("auto", "fused", "xla", "cudnn", "manual"):
        with pytest.raises(ValueError, match="score transform"):
            resolve_attention_backend(SimpleNamespace(attention_backend=request, attention_score_transform="abs"),
                                      platform="gpu")
    # Only an explicit mosaic backend selects the transformed kernels: auto, which follows mosaic_supported, never
    # does, even at a shape the kernels hold.
    shaped = replace(model, qk_head_dim=64, value_head_dim=64, max_seq_len=128)
    assert not shaped.mosaic_supported() and not shaped.mosaic_decay_supported()
    assert shaped.mosaic_score_transform_supported() and not model.mosaic_score_transform_supported()
    assert replace(shaped, attention_score_transform="identity").mosaic_supported()
    assert not replace(shaped, attention_score_transform="identity").mosaic_score_transform_supported()
    mosaic = replace(shaped, attention_backend="mosaic")
    assert resolve_attention_backend(mosaic, dtype="bfloat16", platform="gpu") == "mosaic"
    for dtype, platform, message in (("bfloat16", "cpu", "requires a GPU"), ("float32", "gpu", "bfloat16")):
        with pytest.raises(ValueError, match=message):
            resolve_attention_backend(mosaic, dtype=dtype, platform=platform)
    with pytest.raises(ValueError, match="divisible by 128"):
        resolve_attention_backend(mosaic, dtype="bfloat16", platform="gpu", sequence_length=200)


@pytest.mark.parametrize("changes,problem", (
    ({}, None),
    ({"value_head_dim": 32}, "value width 32"),
    ({"qk_head_dim": 48}, "Q/K width 48"),
    ({"positional_encoding": "none"}, "RoPE"),
    ({"attention_degree": 13}, "degree 13"),
    ({"attention_degree": 12, "max_seq_len": 4096}, "degree 12"),
))
def test_mosaic_recipes_run_where_the_query_kernels_run(changes, problem):
    """On attention_backend mosaic a transform runs where the query polynomial's kernels run: RoPE, their widths,
    and the degrees they hold at the recipe's length; everything else is refused with the reason."""
    settings = {**dict(attention_backend="mosaic", qk_head_dim=64, value_head_dim=64, max_seq_len=128), **changes}
    for transform in TRANSFORMS:
        if problem is None:
            config = _config(transform, 0. if transform == "softplus" else None, **settings)
            config.train.dtype = "bfloat16"
            validate_config(config)
            assert config.model.mosaic_score_transform_problem() is None
            assert replace(config.model, attention_score_transform="identity").mosaic_supported()
            continue
        with pytest.raises(ValueError, match=f"Backend 'mosaic' cannot run the {transform} score transform: .*{problem}"):
            _config(transform, **settings)


def test_every_other_kernel_and_backend_refuses_a_transform():
    q, k, v, c = (jnp.asarray(array, jnp.float32) for array in _operands("mixed", time=5))
    transform = {"score_transform": "softplus"}
    refusals = (
        dict(attention_degree=4, attention_normalization=False, polynomial_coefficients=c[0, 0, 0]),
        dict(attention_degree=None, attention_normalization=True),
        dict(attention_degree=None, attention_normalization=False),
        dict(attention_degree=4, attention_normalization=True),
        dict(attention_degree=4, attention_normalization=False),
        dict(attention_degree=4, attention_normalization=False, query_polynomial_coefficients=c,
             key_polynomial_factors=jnp.ones_like(c), attention_coefficient_activation="tanh"),
        dict(attention_degree=4, attention_normalization=False, query_polynomial_coefficients=c,
             attention_coefficient_activation="sigmoid"),
        dict(attention_degree=4, attention_normalization=False, query_polynomial_coefficients=c,
             attention_coefficient_activation="tanh", query_log_decay=-jnp.ones(q.shape[:3]),
             query_decay_mode="query_distance"),
        dict(attention_degree=4, attention_normalization=False, query_polynomial_coefficients=c,
             attention_coefficient_activation="tanh", score_bias_row_shift=jnp.zeros((*q.shape[:3], 4))),
    )
    for arguments in refusals:
        with pytest.raises(ValueError, match="score transform"):
            attention_core(q, k, v, backend="blocked", **arguments, **transform)
    for backend in ("fused", "manual"):  # the Mosaic dispatch is in test_score_transform_lowering.py
        with pytest.raises(ValueError, match="score transform"):
            query_polynomial_attention(q, k, v, c, backend=backend, **transform)
    for decay in (dict(decay_mode="query_distance", log_decay=-jnp.ones(q.shape[:3])),
                  dict(decay_mode="query_distance", query_gate_input=q[..., 0], key_gate_input=k[..., 0],
                       gate_log_rate=jnp.zeros((2, 1)))):
        with pytest.raises(ValueError, match="score transform"):
            query_polynomial_attention(q, k, v, c, **decay, **transform)
    for arguments in (dict(key_polynomial_factors=jnp.ones_like(c)), dict(query_log_decay=jnp.cumsum(-jnp.ones(
            q.shape[:3]), axis=1)), dict(score_bias_row_shift=jnp.zeros((*q.shape[:3], 4)))):
        for attend in (reference.blocked_dot_product_attention, reference.strip_dot_product_attention):
            with pytest.raises(ValueError, match="score transform"):
                attend(q, k, v, attention_degree=4, attention_normalization=False, query_polynomial_coefficients=c,
                       **arguments, **transform)
    for attend in (reference.blocked_dot_product_attention, reference.strip_dot_product_attention):
        with pytest.raises(ValueError, match="score transform"):
            attend(q, k, v, attention_degree=4, attention_normalization=False, **transform)
        with pytest.raises(ValueError, match="Unknown attention score transform"):
            attend(q, k, v, attention_degree=4, attention_normalization=False, query_polynomial_coefficients=c,
                   score_transform="exp")


@pytest.mark.parametrize("backend,implementation", (
    ("blocked", "strips"), ("blocked", None), ("blocked", "scan"), ("auto", None), ("fused", None), ("mosaic", None),
))
def test_records_create_allocates_transforms_on_the_strips(tmp_path, monkeypatch, capsys, backend, implementation):
    """A fresh transformed recipe trains on the blocked backend's causal strips, which a blocked recipe that
    leaves the implementation unset receives, or on an explicit mosaic backend's transformed kernels, whose recipe
    keeps the inert scan default; the scan, the transform's reference, is refused even with the scan's opt-out
    flag, and validation refuses every other backend before allocation."""
    from scaffolds.records.__main__ import main

    recipe = yaml.safe_load((RUNS / CONTROL / "config.yaml").read_text())
    recipe["model"].update(attention_score_transform="softplus", attention_coefficient_constant_initial=0.0,
                           attention_backend=backend)
    recipe["model"].pop("attention_blocked_implementation")
    if implementation is not None:
        recipe["model"]["attention_blocked_implementation"] = implementation
    source = tmp_path / "recipe.yaml"
    source.write_text(yaml.safe_dump(recipe))
    study = tmp_path / "study"
    monkeypatch.setattr("sys.argv", ["records", "create", "--study", str(study), "--config", str(source)])
    if backend == "mosaic":
        main()
        run = Path(capsys.readouterr().out.strip())
        model = yaml.safe_load((run / "config.yaml").read_text())["model"]
        assert (model["attention_score_transform"], model["attention_coefficient_constant_initial"]) == ("softplus", 0.)
        assert (model["attention_backend"], model["attention_blocked_implementation"]) == ("mosaic", "scan")
    elif backend != "blocked":
        with pytest.raises(ValueError, match="score transform"):
            main()
        assert not study.exists()
    elif implementation == "scan":
        for flags in ((), ("--allow-scan-blocked-attention",)):
            monkeypatch.setattr("sys.argv", ["records", "create", "--study", str(study), "--config", str(source),
                                             *flags])
            with pytest.raises(SystemExit) as failure:
                main()
            error = capsys.readouterr().err
            assert failure.value.code == 2 and "the only blocked computation with a GPU qualification" in error
            assert not study.exists()
    else:
        main()
        run = Path(capsys.readouterr().out.strip())
        model = yaml.safe_load((run / "config.yaml").read_text())["model"]
        assert (model["attention_score_transform"], model["attention_coefficient_constant_initial"]) == ("softplus", 0.)
        assert (model["attention_backend"], model["attention_blocked_implementation"]) == ("blocked", "strips")


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_start_keeps_every_control_array_but_the_softplus_constant(transform):
    """At the control's settings and seed (small shape): no array is added; the abs recipe starts bitwise as
    the control, and the softplus recipe differs only in b_0 = 0 (the control's stored 1) in every layer and
    head. The full-size count is the control's 99,678,056."""
    start = 0. if transform == "softplus" else None
    control, changed = _control(), _control(attention_score_transform=transform,
                                            attention_coefficient_constant_initial=start)
    key = jax.random.PRNGKey(control.train.seed)

    def build(config):
        return build_model(config.model, nnx.Rngs(params=key, dropout=jax.random.fold_in(key, 0)))

    baseline, actual = _params(build(control)), _params(build(changed))
    assert actual.keys() == baseline.keys()
    bias = ("query_polynomial_bias",)
    for path, value in baseline.items():
        if path == bias and transform == "softplus":
            np.testing.assert_array_equal(actual[path][..., 1:], value[..., 1:])
            assert actual[path].dtype == jnp.float32 and (np.asarray(actual[path][..., 0]) == 0).all()
            assert (np.asarray(value[..., 0]) == 1).all()
        else:
            np.testing.assert_array_equal(actual[path], value, err_msg=str(path))
    model = load_config(RUNS / CONTROL / "config.yaml").model
    assert model_parameter_count(replace(model, attention_score_transform=transform, attention_backend="blocked",
                                         attention_blocked_implementation="strips",
                                         attention_coefficient_constant_initial=start)) == CONTROL_PARAMETERS
    facts = acceptance.matches_softmax_backbone_initialization(changed)
    assert "query_polynomial_bias" in facts["mechanism_parameters"]


def _numpy_bins(values):
    """Histogram bins by NumPy's half-open binning: 0 below the first edge, j covering [edge_(j-1), edge_j)."""
    return np.searchsorted(np.asarray(HISTOGRAM_EDGES, np.float64), values, side="right")


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_reference_statistics_are_exact(transform):
    """Raw statistics against NumPy on half-integer scores from -20 to 20, which put every input in the same
    histogram bin in FP32 as in FP64: per head the raw-score and input sums, square sums and extrema, the raw
    scores below zero, and the pairs where the polynomial decreases in z (queries alternating between
    c = (0.4, -1, 0.3, 0.05), which decreases for z below 1.27, and c = (0.4, 1, 0.3, 0.05)); histograms pooled
    over heads."""
    rng = np.random.default_rng(3)
    batch, time, heads, degree = 2, 13, 2, 3
    q, k = (np.zeros((batch, time, heads, 4)) for _ in range(2))
    # s_ti = q_t[0] k_i[0] / 2, multiples of 1/2 from -20 to 20, exact in BF16 and FP32.
    q[..., 0], k[..., 0] = rng.integers(-8, 9, q.shape[:3]), rng.integers(-5, 6, q.shape[:3])
    c = np.empty((batch, time, heads, degree + 1))
    c[...] = [.4, 1., .3, .05]
    c[:, ::2, :, 1] = -1.
    v = np.random.default_rng(4).normal(size=(batch, time, heads, 3))
    _, stats = reference.blocked_dot_product_attention(
        jnp.asarray(q, jnp.bfloat16), jnp.asarray(k, jnp.bfloat16), jnp.asarray(v, jnp.float32),
        attention_degree=degree, attention_normalization=False, query_polynomial_coefficients=jnp.asarray(c, jnp.float32),
        query_block_size=5, return_diagnostics=True, score_transform=transform)
    scores = np.einsum("bthd,bihd->bhti", q, k) / 2
    causal = np.tril(np.ones((time, time), bool))
    raw = scores[:, :, causal]  # [batch, heads, pairs]
    z = np.logaddexp(0., raw) if transform == "softplus" else np.abs(raw)
    assert (raw < 0).any() and (raw > 0).any() and (raw == 0).any() and np.abs(raw).max() >= 16
    # Every input's bin is the same in FP32 (an independent NumPy evaluation) as in FP64: z = |s| is exact, and
    # softplus(s) is s plus a nonnegative term for s >= 0 and never near an edge from below.
    raw32 = raw.astype(np.float32)
    z32 = (np.maximum(raw32, 0) + np.log1p(np.exp(-np.abs(raw32)))) if transform == "softplus" else np.abs(raw32)
    np.testing.assert_array_equal(np.floor(z32), np.floor(z))
    rows = np.broadcast_to(np.arange(time)[:, None], (time, time))[causal]
    coefficients = np.swapaxes(c, 1, 2)[:, :, rows]  # [batch, heads, pairs, m+1]
    slope = sum(r * coefficients[..., r] * z ** (r - 1) for r in range(1, degree + 1))
    assert (slope < 0).any() and (slope > 0).any() and np.abs(slope).min() > 1e-3
    axes = (0, 2)
    expected = {"score_transform_pair_count": batch * time * (time + 1) // 2,
                "score_transform_raw_negative_count": (raw < 0).sum(axis=axes),
                "score_transform_decreasing_count": (slope < 0).sum(axis=axes)}
    for name, values in (("raw", raw), ("input", z)):
        expected.update({f"score_transform_{name}_sum": values.sum(axis=axes),
                         f"score_transform_{name}_sumsq": (values ** 2).sum(axis=axes),
                         f"score_transform_{name}_min": values.min(axis=axes),
                         f"score_transform_{name}_max": values.max(axis=axes),
                         f"score_transform_{name}_histogram": np.bincount(_numpy_bins(values).ravel(),
                                                                         minlength=len(HISTOGRAM_EDGES) + 1)})
    actual = {name: np.asarray(value, np.float64) for name, value in stats.items() if name.startswith("score_transform_")}
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        np.testing.assert_allclose(actual[name], value, rtol=1e-6, atol=1e-6, err_msg=name)
    assert actual["score_transform_raw_histogram"].sum() == actual["score_transform_input_histogram"].sum() == (
        heads * expected["score_transform_pair_count"])
    # The kernel-level diagnostics are unchanged in kind: the extended signed-kernel entries remain.
    assert {"negative_kernel_count", "cancellation_ratio_mean", "kernel_min"} <= stats.keys()


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_diagnostics_report_the_raw_and_transformed_inputs(transform):
    start = 0. if transform == "softplus" else None
    config = _config(transform, start)
    state = create_state(config, jax.random.PRNGKey(0))
    amplitude = state.params["query_polynomial_amplitude"]
    amplitude[...] = jnp.asarray(np.random.default_rng(4).normal(size=amplitude.shape), jnp.float32)
    sample, provenance = fixed_diagnostic_sample(config)
    report = attention_diagnostics(state, config, sample, provenance)
    assert report["valid"], report["issues"]
    assert report["kernel_values"] == f"query_polynomial_of_{transform}_scores"
    assert report["attention_score_transform"] == transform
    assert ("attention_coefficient_constant_initial" in report) is (start is not None)
    section = report["coefficients"]["score_transform"]
    layers, heads, time = 2, 2, 16
    pairs = time * (time + 1) // 2
    assert section["transform"] == transform and section["causal_pair_count"] == pairs
    for group in ("raw_score", "polynomial_input"):
        for name in ("mean", "std", "min", "max"):
            values = np.asarray(section[group][name], np.float64)
            assert values.shape == (layers, heads) and np.isfinite(values).all(), (group, name)
        assert (np.asarray(section[group]["min"]) <= np.asarray(section[group]["max"])).all()
    if transform == "abs":
        np.testing.assert_allclose(section["polynomial_input"]["max"], np.maximum(
            np.abs(section["raw_score"]["min"]), np.abs(section["raw_score"]["max"])), rtol=1e-6)
    assert (np.asarray(section["polynomial_input"]["min"]) >= 0).all()
    for name in ("negative_fraction",):
        assert 0 < np.mean(section["raw_score"][name]) < 1
    decreasing = np.asarray(section["decreasing_fraction"], np.float64)
    assert decreasing.shape == (layers, heads) and ((0 <= decreasing) & (decreasing <= 1)).all()
    for name in ("raw_score_histogram", "polynomial_input_histogram"):
        counts = np.asarray(section[name]["counts"])
        assert counts.shape == (layers, len(HISTOGRAM_EDGES) + 1)
        np.testing.assert_array_equal(counts.sum(axis=-1), heads * pairs)
    # The coefficient statistics stay beside it, per head and order, for the coefficients' signs.
    assert np.asarray(report["coefficients"]["effective_summary"]["min"]).shape == (layers, heads, 5)
    # Per-head arrays never leak into the layer summaries.
    assert not any(name.startswith("score_transform_") for layer in report["layers"] for name in layer)


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_identity_formula_and_tags(transform):
    start = 0. if transform == "softplus" else None
    config = _config(transform, start)
    identity = attention_identity(config)
    formula = identity["attention_formula"]
    assert identity["attention_score_transform"] == transform
    assert "c_r(x_t) z_ti^r v_i" in formula and "s_ti^r" not in formula
    assert score_transform.FORMULAS[transform] in formula and "s_ti = Q_t K_i^T / sqrt(d_k)" in formula
    assert formula.endswith(", then per-head RMSNorm over value channels with a learned gain")
    assert identity["attention_implementation"] == "blocked_causal_strips"
    assert ("b_0 initially 0" in formula) is (start is not None)
    assert ("attention_coefficient_constant_initial" in identity) is (start is not None)
    tags = run_tags(config, {})
    assert f"score_transform_{transform}" in tags and ("coefficient_constant_start" in tags) is (start is not None)
    started = attention_identity(_config("identity", -0.5))
    assert "b_r initially 1/r! for r >= 1 and b_0 initially -0.5" in started["attention_formula"]
    assert "attention_score_transform" not in started and "s_ti^r" in started["attention_formula"]
