"""Mosaic GPU kernels for per-order cumulative forget gates (family ``per_order_cumulative_decay``).

W_ti = sum_r c_r(x_t) s_ti^r exp(L_tr - L_ir), with every order's own cumulative log decay L_r, inside
H100's causal tiles: the forward, the dQ kernel with the coefficient sums dc_r, and the dK/dV kernel with
every key's sum of G W_r over its queries, the key side of dL_r = c_r dc_r - sum_t' G W_r(t', i). B300
refuses the family before building a kernel. On CPU: the tile mathematics against FP64 (masked future
keys, very negative sums, invalid values), the kernels' lowering in exactly their counted shared memory,
the entry point's operands, refusals and gradient assembly, the model dispatch, and the recipe rules. On
the GPU (``-m gpu`` in an H100 allocation): every output and gradient against host FP64 and against the
blocked strips, gate extremes, invalid sums, bitwise repeatability, and a two-layer model against the
FP32 blocked model.
"""
from functools import partial
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer import acceptance
from scaffolds.transformer.attention import input_dependent
from scaffolds.transformer.attention.backends.mosaic import api, common, hopper
from scaffolds.transformer.attention.backends.reference import (
    blocked_dot_product_attention, strip_dot_product_attention,
)
from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.attention.forget_gate import log_decay
from scaffolds.transformer.attention.multihead import attention_from_config
from scaffolds.transformer.configs import config_to_dict
from scaffolds.transformer.models import build_model
from scaffolds.transformer.tests.test_per_order_cumulative import _config, _dense, _numpy_vjp
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.diagnostics import attention_identity
from scaffolds.transformer.training.state import create_state

FAMILY = common.PER_ORDER_CUMULATIVE_DECAY
# Output, dQ, dK, dV: the Mosaic query-polynomial kernels' BF16 bounds (test_mosaic_query, from
# test_fused_attention). The coefficient and log-decay gradients are FP32 sums of FP32 terms built from
# exact BF16 products, held to the strips' bound (test_per_order_cumulative), order by order.
BF16_TOLERANCES = {"output": .006, "dq": .025, "dk": .025, "dv": .015}
FP32_TOLERANCE = 1e-4
NAMES = ("output", "dq", "dk", "dv", "coefficients", "log_decay")
# The Transformer++ widths and context.
MOSAIC_SHAPE = dict(qk_head_dim=64, value_head_dim=64, max_seq_len=2048)


def _relative_l2(actual, expected):
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


# --- The tile mathematics (common.py) against FP64 ------------------------------------------------

def _tile(degree, queries, keys, *, offset=0.0, seed=0):
    """FP32 operands of one tile, as the kernels see them, and FP64 copies of the same values: scores s,
    G = dO V^T, query coefficients, and per-order cumulative sums whose per-token log alpha spans half-lives
    from under a token to about a thousand tokens, shifted by ``offset`` (sums near -3,000 are a fast gate's
    at the end of a 2,048-token context)."""
    rng = np.random.default_rng(seed + degree)
    orders = degree + 1
    end = max(queries[-1], keys[-1]) + 1
    sums = np.cumsum(-np.exp(rng.uniform(-7.0, 1.0, (end, orders))), axis=0) + offset
    coefficients = rng.standard_normal((len(queries), orders)) / np.asarray([math.factorial(r) for r in range(orders)])
    scores, gradient = (rng.standard_normal((len(queries), len(keys))) for _ in range(2))
    single = lambda x: np.asarray(x, np.float32).astype(np.float64)  # the FP32 values themselves
    return {"s": single(scores), "g": single(gradient), "c": single(coefficients), "lq": single(sums[queries]),
            "lk": single(sums[keys]), "causal": np.asarray(queries)[:, None] >= np.asarray(keys)[None, :]}


def _tile_reference(t):
    """FP64 W, dS, dc_r (per query) and sum_t G W_r (per key) of a tile."""
    causal, orders = t["causal"], t["c"].shape[-1]
    gap = np.where(causal[..., None], t["lq"][:, None, :] - t["lk"][None, :, :], 0.0)
    decay = np.where(causal[..., None], np.exp(gap), 0.0)
    powers = t["s"][..., None] ** np.arange(orders)
    g = np.where(causal, t["g"], 0.0)
    weights = (t["c"][:, None, :] * powers * decay).sum(-1)
    slopes = (t["c"][:, None, 1:] * np.arange(1, orders) * powers[..., :-1] * decay[..., 1:]).sum(-1)
    return {"w": weights, "ds": g * slopes, "dc": (g[..., None] * powers * decay).sum(1),
            "keys": (g[..., None] * t["c"][:, None, :] * powers * decay).sum(0)}


def _tile_terms(t, degree):
    """The kernels' use of common.py on a tile and on its transpose (dK/dV: keys in rows): masked scores
    and G enter as zero, a tile wholly below the diagonal passes no mask, and the returned tiles are masked
    as the kernels mask them. Vectors and keys are ``Lazy``, as the kernels pass them."""
    orders = degree + 1
    causal = jnp.asarray(t["causal"])
    masked = not t["causal"].all()
    rows = jnp.asarray(np.concatenate([t["c"].T, t["lq"].T]), jnp.float32)  # [c_0..c_m, L_0..L_m] per query
    keys = jnp.asarray(t["lk"].T, jnp.float32)
    s, g = (jnp.asarray(t[name], jnp.float32) for name in ("s", "g"))
    if masked:
        s, g = jnp.where(causal, s, 0.0), jnp.where(causal, g, 0.0)

    def operands(transposed):
        shape = (keys.shape[1], rows.shape[1]) if transposed else (rows.shape[1], keys.shape[1])
        query_axis, key_axis = (1, 0) if transposed else (0, 1)
        vectors = common.Lazy(2 * orders, lambda i: jax.lax.broadcast_in_dim(rows[i], shape, [query_axis]))
        key_vectors = common.Lazy(orders, lambda i: jax.lax.broadcast_in_dim(keys[i], shape, [key_axis]))
        mask = (causal.T if transposed else causal) if masked else None
        return vectors, common.Pair(None, key_vectors, mask)

    def where(x, mask):
        return jnp.where(mask, x, 0.0) if masked else x

    vectors, pair = operands(False)
    weights = where(common.forward_weights(FAMILY, degree, s, vectors, pair), causal)
    ds, row_sums = common.dq_terms(FAMILY, degree, s, g, vectors, pair)
    vectors, pair = operands(True)
    weights_t, ds_t, key_sums = common.dkv_terms(FAMILY, degree, s.T, g.T, vectors, pair)
    return {"w": weights, "ds": where(ds, causal), "dc": jnp.stack(row_sums, -1),
            "w_t": where(weights_t, causal.T), "ds_t": where(ds_t, causal.T), "keys": jnp.stack(key_sums, -1)}


SPANS = {"diagonal": (np.arange(16, 32), np.arange(8, 24)), "below": (np.arange(32, 48), np.arange(0, 16))}


@pytest.mark.parametrize("offset", (0.0, -3000.0))
@pytest.mark.parametrize("span", list(SPANS))
@pytest.mark.parametrize("degree", (1, 4, 12))
def test_tile_mathematics_match_float64(degree, span, offset):
    """Weights, dS, the dQ sums dc_r and the dK sums of G W_r, in a tile crossing the diagonal and one wholly
    below it, within FP32 rounding of FP64 (relative L2 2e-6, each order's sums on their own). Sums near
    -3,000 keep that accuracy only because the tile forms L_tr - L_ir before scaling it to base 2: scaling
    each sum first errs by up to 3e-4."""
    tile = _tile(degree, *SPANS[span], offset=offset)
    expected, actual = _tile_reference(tile), _tile_terms(tile, degree)
    for name, got, want in (("w", actual["w"], expected["w"]), ("ds", actual["ds"], expected["ds"]),
                            ("w_t", actual["w_t"], expected["w"].T), ("ds_t", actual["ds_t"], expected["ds"].T)):
        assert _relative_l2(got, want) < 2e-6, name
    for name in ("dc", "keys"):
        assert actual[name].shape == expected[name].shape, name
        for order in range(degree + 1):
            assert _relative_l2(actual[name][:, order], expected[name][:, order]) < 2e-6, (name, order)


def test_tile_gate_is_exact_on_the_diagonal_and_drops_future_keys():
    """Sums falling 1e30 per token: every earlier key's gate underflows and every future key's exponent
    overflows. The diagonal gate is exactly one (W is the identity for c = e_0); future keys whose sums are
    nonfinite, and a query whose every entry is masked carrying NaN coefficients, reach no weight, dS, or
    sum; dc_0 and the key sum of order 0 are each diagonal's G exactly, so their difference (L_0's
    gradient) is exactly zero."""
    degree, orders = 4, 5
    tile = _tile(degree, np.arange(16, 32), np.arange(16, 32))
    tile["lq"] = tile["lk"] = np.broadcast_to(np.float32(-1e30) * np.arange(16, 32)[:, None], (16, orders)).astype(np.float64)
    tile["c"] = np.zeros_like(tile["c"])
    tile["c"][:, 0] = 1.0
    actual = _tile_terms(tile, degree)
    np.testing.assert_array_equal(actual["w"], np.eye(16))
    np.testing.assert_array_equal(actual["w_t"], np.eye(16))
    np.testing.assert_array_equal(actual["dc"][:, 0], np.diag(tile["g"]).astype(np.float32))
    np.testing.assert_array_equal(actual["keys"][:, 0], actual["dc"][:, 0])
    # Keys 32..39 are future to every query (16..31), and query 16 precedes every key (24..39).
    tile = _tile(degree, np.arange(16, 32), np.arange(24, 40))
    tile["lk"][8:] = np.nan
    tile["lk"][9] = np.inf
    tile["c"][0] = np.nan
    expected, actual = _tile_reference({**tile, "lk": np.where(np.isfinite(tile["lk"]), tile["lk"], 0.0),
                                        "c": np.where(np.isfinite(tile["c"]), tile["c"], 0.0)}), _tile_terms(tile, degree)
    for name, value in actual.items():
        assert np.isfinite(np.asarray(value)).all(), name
    assert _relative_l2(actual["w"], expected["w"]) < 2e-6 and _relative_l2(actual["ds"], expected["ds"]) < 2e-6
    np.testing.assert_array_equal(actual["keys"][8:], 0.0)
    for name in ("dc", "keys"):
        for order in range(orders):
            assert _relative_l2(actual[name][:, order], expected[name][:, order]) < 2e-6, (name, order)


# --- H100 kernels and the entry point ---------------------------------------------------------------

def _lowers_for_cuda(function, *operands):
    """Trace and lower ``function`` for the cuda platform on any host, building every Mosaic GPU kernel
    (through a fresh wrapper: JAX would reuse an earlier lowering of the same function)."""
    jax.jit(lambda *arguments: function(*arguments)).trace(*operands).lower(lowering_platforms=("cuda",))


@pytest.mark.parametrize("degree,width", ((4, 64), (12, 64), (12, 128)), ids=("m4-dk64", "m12-dk64", "m12-dk128"))
def test_hopper_kernels_lower_in_exactly_their_counted_shared_memory(degree, width):
    """H100's forward, dQ and dK/dV lower at 2,048 tokens in exactly the bytes hopper_smem_bytes counts (one
    byte less fails), below H100's limit at the runs' degrees and up to the largest degree and width."""
    from jax.experimental.mosaic.gpu.core import artificial_shared_memory_limit

    n_vectors, n_keys = common.vector_counts(FAMILY, degree)[0], common.key_counts(FAMILY, degree)[0]
    assert (n_vectors, n_keys, common.head_count(FAMILY, degree)) == (2 * (degree + 1), degree + 1, 0)
    time = 2048
    q, v = (jax.ShapeDtypeStruct((1, time, 2, size), jnp.bfloat16) for size in (width, 64))
    vectors = jax.ShapeDtypeStruct((n_vectors, 1, 2, time), jnp.float32)
    keys = jax.ShapeDtypeStruct((n_keys, 1, 2, time), jnp.float32)
    options = dict(family=FAMILY, degree=degree, scale=width ** -.5, precision="bf16")
    kernels = {"forward": (lambda q, k, v, x, y: hopper.forward(q, k, v, x, y, **options), (q, q, v)),
               "dq": (lambda q, k, v, do, x, y: hopper.dq(q, k, v, do, x, y, **options), (q, q, v, v)),
               "dkv": (lambda q, k, v, do, x, y: hopper.dkv(q, k, v, do, x, y, **options), (q, q, v, v))}
    sizes = common.hopper_smem_bytes(FAMILY, degree, width, 64)
    assert max(sizes.values()) <= common.HOPPER_SMEM_BYTES
    for name, (kernel, arrays) in kernels.items():
        with artificial_shared_memory_limit(sizes[name]):
            _lowers_for_cuda(kernel, *arrays, vectors, keys)
        with artificial_shared_memory_limit(sizes[name] - 1), pytest.raises(ValueError, match="shared memory"):
            _lowers_for_cuda(kernel, *arrays, vectors, keys)


def _abstract_operands(degree, *, time=2048, width=64):
    q = jax.ShapeDtypeStruct((1, time, 2, width), jnp.bfloat16)
    v = jax.ShapeDtypeStruct((1, time, 2, 64), jnp.bfloat16)
    per_order = jax.ShapeDtypeStruct((1, time, 2, degree + 1), jnp.float32)
    return q, q, v, per_order, per_order


def _gradients(*operands):
    output, pullback = jax.vjp(api.per_order_cumulative_decay_attention, *operands)
    return output, pullback(jnp.ones(output.shape, output.dtype))


def test_the_entry_point_lowers_on_h100_and_b300_refuses_it(monkeypatch):
    """Through the production entry point (forward, dQ, dK and dV) at d_K = 64 and 2,048 tokens: H100's
    kernels lower at degrees 1 through 12, and B300 raises before any kernel is traced, at every degree
    (degree one would fit its shared memory)."""
    monkeypatch.setattr(api, "_kernels", lambda: ("hopper", hopper))
    for degree in (1, 4, 8, 12):
        _lowers_for_cuda(_gradients, *_abstract_operands(degree))
    _lowers_for_cuda(_gradients, *_abstract_operands(4, time=128, width=8))  # one CTA; width padded to 16
    monkeypatch.setattr(api, "_kernels", lambda: ("blackwell", None))  # no kernel may be reached
    for degree in (1, 12):
        with pytest.raises(ValueError, match="runs on H100 only"):
            _lowers_for_cuda(_gradients, *_abstract_operands(degree))


def test_support_predicates():
    """Cumulative gates are supported per order with query conditioning (H100's kernels fit every degree to
    12 at every width), never shared, query-key, above degree 12, or on B300; the query-distance
    predicates are unchanged."""
    for width in api.WIDTHS:
        assert [m for m in range(api.MAX_QUERY_DEGREE + 2)
                if api.query_degree_supported(m, width, 2048, gate="query", per_order=True, mode="cumulative")
                ] == list(range(1, 13))
        assert all(api.hopper_supported(FAMILY, m, width) and not api.blackwell_supported(FAMILY, m, width, 2048)
                   for m in range(1, 13))
        for gate, per_order in (("query", False), ("query_key", False), ("query_key", True)):
            assert not any(api.query_degree_supported(m, width, 2048, gate=gate, per_order=per_order,
                                                      mode="cumulative") for m in range(1, 13))
        for gate in ("query", "query_key"):
            for per_order in (False, True):
                assert all(api.query_degree_supported(m, width, 2048, gate=gate, per_order=per_order)
                           == api.query_degree_supported(m, width, 2048, gate=gate, per_order=per_order,
                                                         mode="query_distance") for m in range(14))
    with pytest.raises(ValueError, match="Unknown decay mode"):
        api.query_degree_supported(4, 64, 2048, gate="query", per_order=True, mode="distance")
    sizes = common.hopper_smem_bytes(FAMILY, 12, 64, 64)
    assert sizes == {"forward": 125040, "dq": 95312, "dkv": 95280}  # at degree twelve and d_K = 64


@pytest.mark.parametrize("change,message", (
    ({"log_decay": (1, 128, 2, 5), "log_decay_dtype": jnp.float32}, "cumulative log decays"),
    ({"log_decay": (1, 128, 2), "log_decay_dtype": jnp.float32}, "cumulative log decays"),
    ({"log_decay": (1, 128, 2, 4), "log_decay_dtype": jnp.bfloat16}, "cumulative log decays"),
    ({"coefficients": (1, 128, 2, 14)}, "coefficients"),
    ({"time": 192}, "multiple of 128"),
    ({"precision": "fp32"}, "precision"),
))
def test_the_entry_point_validates_its_operands(monkeypatch, change, message):
    monkeypatch.setattr(api, "_kernels", lambda: ("hopper", None))  # no kernel may be reached
    time = change.get("time", 128)
    q = jnp.zeros((1, time, 2, 64), jnp.bfloat16)
    c = jnp.zeros(change.get("coefficients", (1, time, 2, 4)), jnp.float32)
    sums = jnp.zeros(change.get("log_decay", (1, time, 2, 4)), change.get("log_decay_dtype", jnp.float32))
    with pytest.raises(ValueError, match=message):
        api.per_order_cumulative_decay_attention(q, q, q, c, sums, precision=change.get("precision"))


class _DenseKernels:
    """H100's kernel contract for this family, evaluated densely in FP32 on any host: per-query vectors
    [c_0..c_m, L_0..L_m] and per-key vectors [L_0..L_m], [n, batch, heads, time]; ``forward`` returns the
    FP32 output and a row statistic, ``dq`` dQ and the sums dc_r, ``dkv`` dK, dV and each key's sums
    sum_t G W_r. The entry point's operands and its assembly of the gradients run unchanged."""
    received = []

    @classmethod
    def _tiles(cls, q, k, vectors, key_vectors, degree, scale):
        orders = degree + 1
        cls.received.append((vectors, key_vectors))
        time = q.shape[1]
        causal = jnp.tril(jnp.ones((time, time), bool))
        scores = jnp.where(causal, jnp.einsum("bthd,bihd->bhti", q.astype(jnp.float32), k.astype(jnp.float32))
                           * scale, 0.0)
        c = vectors[:orders].transpose(1, 2, 3, 0)[..., :, None, :]  # [b, h, t, 1, r]
        query_sums = vectors[orders:].transpose(1, 2, 3, 0)[..., :, None, :]
        key_sums = key_vectors.transpose(1, 2, 3, 0)[..., None, :, :]  # [b, h, 1, i, r]
        gap = jnp.where(causal[..., None], query_sums - key_sums, 0.0)
        decay = jnp.where(causal[..., None], jnp.exp(gap), 0.0)
        powers = scores[..., None] ** jnp.arange(orders)
        return causal, scores, c, decay, powers

    @classmethod
    def forward(cls, q, k, v, vectors, key_vectors=None, head_constants=None, *, family, degree, scale, precision):
        assert family == FAMILY and head_constants is None
        _, _, c, decay, powers = cls._tiles(q, k, vectors, key_vectors, degree, scale)
        weights = (c * powers * decay).sum(-1)
        output = jnp.einsum("bhti,bihd->bthd", weights, v.astype(jnp.float32))
        return output, (weights - weights).sum(-1)

    @classmethod
    def _backward(cls, q, k, v, do, vectors, key_vectors, degree, scale):
        causal, scores, c, decay, powers = cls._tiles(q, k, vectors, key_vectors, degree, scale)
        g = jnp.where(causal, jnp.einsum("bthd,bihd->bhti", do.astype(jnp.float32), v.astype(jnp.float32)), 0.0)
        orders = jnp.arange(1, degree + 1)
        ds = g * (c[..., 1:] * orders * powers[..., :-1] * decay[..., 1:]).sum(-1)
        return g, ds, c, decay, powers

    @classmethod
    def dq(cls, q, k, v, do, vectors, key_vectors=None, head_constants=None, *, family, degree, scale, precision):
        g, ds, _, decay, powers = cls._backward(q, k, v, do, vectors, key_vectors, degree, scale)
        grad_q = jnp.einsum("bhti,bihd->bthd", ds, k.astype(jnp.float32)) * scale
        return grad_q.astype(q.dtype), (g[..., None] * powers * decay).sum(-2).transpose(3, 0, 1, 2)

    @classmethod
    def dkv(cls, q, k, v, do, vectors, key_vectors=None, head_constants=None, *, family, degree, scale, precision):
        g, ds, c, decay, powers = cls._backward(q, k, v, do, vectors, key_vectors, degree, scale)
        weights = (c * powers * decay).sum(-1)
        grad_k = jnp.einsum("bhti,bthd->bihd", ds, q.astype(jnp.float32)) * scale
        grad_v = jnp.einsum("bhti,bthd->bihd", weights, do.astype(jnp.float32))
        key_sums = (g[..., None] * c * powers * decay).sum(-3).transpose(3, 0, 1, 2)
        return grad_k.astype(k.dtype), grad_v.astype(v.dtype), key_sums


def _cpu_operands(time, degree, *, width=64, heads=2, seed=5):
    rng = np.random.default_rng(seed + degree)
    q, k = (jnp.asarray(rng.standard_normal((2, time, heads, width)), jnp.bfloat16) for _ in range(2))
    v, cotangent = (jnp.asarray(rng.standard_normal((2, time, heads, 64)), jnp.bfloat16) for _ in range(2))
    orders = degree + 1
    taylor = np.asarray([1 / math.factorial(r) for r in range(orders)])
    c = jnp.asarray(taylor * rng.standard_normal((2, time, heads, orders)), jnp.float32)  # signed
    sums = jnp.cumsum(jnp.asarray(-np.exp(rng.uniform(-7.0, 1.0, (2, time, heads, orders))), jnp.float32), axis=1)
    return (q, k, v, c, sums), cotangent


@pytest.mark.parametrize("width", (8, 64))
def test_the_entry_point_assembles_every_gradient_from_the_kernel_sums(monkeypatch, width):
    """With the kernels replaced by their dense FP32 contract, the entry point's output and its gradients
    of Q, K, V, the coefficients, and the cumulative sums (dL = c dc less the dK kernel's sums) match FP64
    autodiff of the all-pairs reference: the BF16 outputs within their rounding, the FP32 gradients order by
    order within 1e-5. Its kernels receive [c; L] and L with a nonfinite sum as NaN."""
    monkeypatch.setattr(api, "_kernels", lambda: ("hopper", _DenseKernels))
    _DenseKernels.received.clear()
    degree = 4
    operands, cotangent = _cpu_operands(128, degree, width=width)
    output, pullback = jax.vjp(api.per_order_cumulative_decay_attention, *operands)
    actual = (output, *pullback(cotangent))
    with jax.enable_x64():
        expected = jax.tree.map(np.asarray, (lambda o, p: (o, *p(jnp.asarray(cotangent, jnp.float64))))(
            *jax.vjp(_dense, *(jnp.asarray(x, jnp.float64) for x in operands))))
    for name, got, want in zip(NAMES, actual, expected, strict=True):
        assert got.shape == want.shape, name
        assert got.dtype == (jnp.float32 if name in ("coefficients", "log_decay") else jnp.bfloat16), name
        if name in ("coefficients", "log_decay"):
            for order in range(degree + 1):
                assert _relative_l2(np.asarray(got)[..., order], want[..., order]) < 1e-5, (name, order)
        else:
            assert _relative_l2(np.asarray(got.astype(jnp.float32)), want) < 4e-3, name
    vectors, key_vectors = _DenseKernels.received[0]
    np.testing.assert_array_equal(key_vectors, operands[4].transpose(3, 0, 2, 1))
    np.testing.assert_array_equal(vectors, jnp.concatenate([operands[3].transpose(3, 0, 2, 1), key_vectors]))
    _DenseKernels.received.clear()
    bad = operands[4].at[1, 70, 0, 2].set(jnp.inf).at[0, 9, 1, 0].set(-jnp.inf)
    output = api.per_order_cumulative_decay_attention(*operands[:4], bad)
    key_vectors = np.asarray(_DenseKernels.received[0][1])
    assert np.isnan(key_vectors[2, 1, 0, 70]) and np.isnan(key_vectors[0, 0, 1, 9])
    assert np.isfinite(np.delete(key_vectors.reshape(-1), [np.ravel_multi_index(i, key_vectors.shape)
                                                            for i in ((2, 1, 0, 70), (0, 0, 1, 9))])).all()
    invalid = ~np.isfinite(np.asarray(output, np.float32)).all(axis=-1)
    expected = np.zeros(invalid.shape, bool)
    expected[1, 70:, 0] = expected[0, 9:, 1] = True
    np.testing.assert_array_equal(invalid, expected)


# --- Dispatch, model, and recipes ----------------------------------------------------------------------

def test_the_dispatch_routes_per_order_cumulative_decay_to_the_kernels(monkeypatch):
    """query_polynomial_attention and attention_core send a per-order cumulative log decay on Mosaic to the
    kernels' entry point, with diagnostics from the blocked reference; a shared cumulative gate stays on the
    blocked backend, and the fused backend refuses both."""
    calls = []

    def kernels(q, k, v, coefficients, log_decay, *, precision=None):
        calls.append((q, k, v, coefficients, log_decay))
        return jnp.full(v.shape, 7.0, v.dtype)

    monkeypatch.setattr(api, "per_order_cumulative_decay_attention", kernels)
    (q, k, v, c, sums), _ = _cpu_operands(16, 4)
    output = input_dependent.query_polynomial_attention(q, k, v, c, backend="mosaic", log_decay=sums, decay_mode="cumulative")
    np.testing.assert_array_equal(output, jnp.full(v.shape, 7.0, v.dtype))
    assert len(calls) == 1 and all(a is b for a, b in zip(calls[0], (q, k, v, c, sums)))
    output, diagnostics = attention_core(q, k, v, backend="mosaic", attention_degree=4, attention_normalization=False,
                                        query_polynomial_coefficients=c, attention_coefficient_activation="tanh",
                                        query_log_decay=sums, query_decay_mode="cumulative", return_diagnostics=True)
    assert len(calls) == 2 and calls[1][4] is sums
    _, reference = blocked_dot_product_attention(q, k, v, attention_degree=4, attention_normalization=False,
                                                 query_polynomial_coefficients=c, query_log_decay=sums,
                                                 return_diagnostics=True)
    assert set(reference) < set(diagnostics) and "query_coefficient_mean" in diagnostics
    for name, value in reference.items():
        np.testing.assert_array_equal(diagnostics[name], value, err_msg=name)
    for backend, decay in (("mosaic", sums[..., 0]), ("fused", sums), ("fused", sums[..., 0])):
        with pytest.raises(ValueError, match="blocked reference backend"):
            input_dependent.query_polynomial_attention(q, k, v, c, backend=backend, log_decay=decay, decay_mode="cumulative")
    assert len(calls) == 2


def _mosaic_config(degree=4, **changes):
    """The run's recipe on Mosaic with BF16, at a tiny shape with value width 64 (the kernels') unless
    ``changes`` give the shape."""
    return _config(degree, **{"attention_backend": "mosaic", "dtype": "bfloat16", "value_head_dim": 64, **changes})


def test_the_mosaic_model_passes_each_orders_cumulative_sum_to_the_kernels(monkeypatch):
    """The model's route to the kernels: with the entry point evaluated by the causal strips (and the
    backend left unresolved, as no GPU is present), the Mosaic recipe's loss and every gradient equal the
    strips recipe's bitwise; the kernels receive the query coefficients and each order's FP32 cumulative
    sum L [B, T, H, m+1] of the gate's log alpha."""
    from scaffolds.transformer.attention.backends import selection

    monkeypatch.setattr(selection, "resolve_attention_backend", lambda model, **_: model.attention_backend)
    received = []

    def strips(q, k, v, coefficients, log_decay, *, precision=None):
        received.append((coefficients, log_decay))
        return strip_dot_product_attention(q, k, v, attention_degree=coefficients.shape[-1] - 1,
                                           attention_normalization=False, query_polynomial_coefficients=coefficients,
                                           query_log_decay=log_decay)

    monkeypatch.setattr(api, "per_order_cumulative_decay_attention", strips)
    batch = jnp.asarray(np.random.default_rng(8).integers(0, 32, (2, 9)), jnp.int32)
    results = {}
    for name, config in (("mosaic", _mosaic_config()), ("strips", _config(dtype="bfloat16", value_head_dim=64))):
        state = create_state(config, jax.random.PRNGKey(3))
        for parameter in ("forget_gate_kernel", "forget_gate_dt_bias"):  # let the orders differ
            value = state.params[parameter][...]
            state.params[parameter][...] = value + jnp.asarray(
                0.3 * np.random.default_rng(9).standard_normal(value.shape), value.dtype)
        loss, gradients = loop.loss_and_grads(state, batch, jax.random.PRNGKey(0))
        results[name] = float(loss), {path: np.asarray(value.get_value()) for path, value in nnx.to_flat_state(gradients)}
    assert len(received) == 2  # one call per layer, from the Mosaic model alone
    for coefficients, sums in received:
        assert coefficients.dtype == sums.dtype == jnp.float32 and coefficients.shape == sums.shape == (2, 8, 2, 5)
    assert results["mosaic"][0] == results["strips"][0]
    for path, value in results["strips"][1].items():
        np.testing.assert_array_equal(results["mosaic"][1][path], value, err_msg=str(path))


@pytest.mark.parametrize("degree", (1, 4, 8, 12))
def test_mosaic_accepts_the_run_recipe_and_names_its_kernels(degree):
    """The run's recipe validates on Mosaic at the Transformer++ widths and context with BF16 (RoPE or none,
    every nonlinear generator), the module builds, and its identity names the cumulative kernels at BF16
    weight precision. ``auto`` never selects these kernels."""
    for encoding in ("none", "rope"):
        config = _mosaic_config(degree, positional_encoding=encoding, **MOSAIC_SHAPE)
        assert config.model.mosaic_decay_supported() and not config.model.mosaic_supported()
        attention_from_config(config.model, rngs=nnx.Rngs(0))
    identity = attention_identity(_mosaic_config(degree, **MOSAIC_SHAPE))
    assert identity["attention_implementation"] == "pallas_mosaic_gpu_query_polynomial_per_order_cumulative_decay"
    assert identity["attention_weight_precision"] == "bf16"
    assert identity["attention_formula"] == attention_identity(_config(degree, **MOSAIC_SHAPE))["attention_formula"]
    for changes in ({"attention_coefficient_activation": "sigmoid", "attention_coefficient_amplitude_initial": 1.0,
                     "attention_coefficient_scale": "none"},
                    {"attention_coefficient_readout": "dense", "attention_coefficient_readout_scale": 5 ** -.5}):
        assert _mosaic_config(degree, **changes, **MOSAIC_SHAPE).model.mosaic_decay_supported()


REFUSED = (
    ({"attention_forget_gate_per_order": False}, "per-order cumulative query gates"),
    ({"attention_forget_gate_conditioning": "query_key"}, "Backend 'mosaic'"),
    ({"attention_degree": 13, "attention_coefficient_scale": "none", "attention_coefficient_amplitude_scale": "none"},
     "Backend 'mosaic'"),
    ({"qk_head_dim": 96}, "Backend 'mosaic'"),
    ({"value_head_dim": 128}, "Backend 'mosaic'"),
    ({"attention_forget_gate_rate_activation": "sigmoid"}, "Backend 'mosaic'|Per-order gates"),
    ({"dtype": "float32"}, "bfloat16"),
)


@pytest.mark.parametrize("change,message", REFUSED, ids=[",".join(map(str, c)) for c, _ in REFUSED])
def test_mosaic_refuses_what_its_cumulative_kernels_do_not_run(change, message):
    """Shared cumulative gates, query-key conditioning, degree 13, unsupported widths, bounded rates, and FP32
    training are refused on Mosaic."""
    change = dict(change)
    settings = {**MOSAIC_SHAPE, **{key: value for key, value in change.items() if key in MOSAIC_SHAPE}}
    change = {key: value for key, value in change.items() if key not in MOSAIC_SHAPE}
    dtype = change.pop("dtype", "bfloat16")
    with pytest.raises(ValueError, match=message):
        _config(change.pop("attention_degree", 4), attention_backend="mosaic", dtype=dtype, **settings, **change)


def test_a_shared_cumulative_gate_on_mosaic_is_refused_by_the_model():
    """The attention module cannot see whether its gates are per order (their parameters carry the order
    axis), so it builds a shared cumulative gate on Mosaic; the model re-validates its recipe, which refuses
    it, and the dispatch refuses its log decay [B, T, H] on Mosaic (above)."""
    model = _config(attention_forget_gate_per_order=False, value_head_dim=64).model
    model.attention_backend = "mosaic"  # bypass the recipe's validation
    attention_from_config(model, rngs=nnx.Rngs(0))
    with pytest.raises(ValueError, match="per-order cumulative query gates"):
        build_model(model, 42)


def test_acceptance_runs_the_kernel_checks_for_mosaic_recipes():
    """Mosaic per-order cumulative recipes run this module on CPU and GPU beside the cumulative gates' own
    checks, and not the query-distance kernels' qualification; blocked recipes keep their selection."""
    for degree in (4, 12):
        config = _mosaic_config(degree, **MOSAIC_SHAPE)
        spec = acceptance.attention_spec(config, entry_module="tests")
        recipe = config_to_dict(config)
        for cpu in (True, False):
            paths = acceptance.test_paths(spec, recipe, resolved_backend="mosaic", cpu=cpu)
            assert acceptance.PER_ORDER_CUMULATIVE_MOSAIC_TESTS in paths
            assert acceptance.PER_ORDER_CUMULATIVE_TESTS in paths
            assert acceptance.ENDPOINT_DECAY_TESTS not in paths
            assert (acceptance.PER_ORDER_DECAY_TESTS in paths) == cpu
        blocked = _config(degree, **MOSAIC_SHAPE)
        for cpu in (True, False):
            paths = acceptance.test_paths(acceptance.attention_spec(blocked, entry_module="tests"),
                                          config_to_dict(blocked), resolved_backend="blocked", cpu=cpu)
            assert acceptance.PER_ORDER_CUMULATIVE_MOSAIC_TESTS not in paths
    assert acceptance.mosaic_test_options(config_to_dict(_mosaic_config(12, **MOSAIC_SHAPE)), "mosaic") == (
        "--mosaic-precision", "bf16")


# --- GPU qualification (H100) ------------------------------------------------------------------------

def _requires_h100():
    if jax.default_backend() != "gpu" or jax.devices()[0].compute_capability not in ("9.0", "10.0", "10.3"):
        pytest.skip("Requires an H100 or B300 allocation")
    if jax.devices()[0].compute_capability != "9.0":
        pytest.skip("These kernels run on H100 only (test_the_gpu_refuses_b300 checks B300's refusal)")


def _gpu_inputs(time, degree, *, width=64, heads=2, batch=1, seed=41, rates=(3e-4, 2.0), variant="taylor"):
    """BF16 Q, K, V with unit-scale scores (as QK RMSNorm gives), FP32 coefficients near Taylor (``signed``:
    random signs; ``top``: orders above m - 4 alone), and each order's FP32 cumulative sum of per-token log
    alpha, rates log-uniform in ``rates`` per head and order and varying by token; a BF16 cotangent."""
    rng = np.random.default_rng(seed + 100 * degree + width + time)
    q, k = (jnp.asarray(rng.standard_normal((batch, time, heads, width)), jnp.bfloat16) for _ in range(2))
    v, cotangent = (jnp.asarray(rng.standard_normal((batch, time, heads, 64)), jnp.bfloat16) for _ in range(2))
    orders = degree + 1
    taylor = np.asarray([1 / math.factorial(r) for r in range(orders)])
    shape = (batch, time, heads, orders)
    c = taylor * (rng.standard_normal(shape) if variant == "signed" else 1 + 0.3 * np.tanh(rng.standard_normal(shape)))
    if variant == "top":
        c = c * (np.arange(orders) > degree - 4)
    rate = np.exp(rng.uniform(math.log(rates[0]), math.log(rates[1]), (heads, orders)))
    raw = -rate * np.exp(rng.uniform(-0.5, 0.5, shape))
    sums = jnp.cumsum(jnp.asarray(raw, jnp.float32), axis=1)
    return (q, k, v, jnp.asarray(c, jnp.float32), sums), cotangent


def _vjp(function, operands, cotangent):
    output, pullback = jax.vjp(function, *operands)
    return (output, *pullback(cotangent))


def _oracle(operands, cotangent):
    """Host FP64 output and gradients (test_per_order_cumulative's oracle, which matches FP64 autodiff)."""
    q, k, v, c, sums = operands
    output, gradients = _numpy_vjp(*(x.astype(jnp.float32) for x in (q, k, v)), c, sums, cotangent.astype(jnp.float32))
    return (output, *gradients)


def _check(actual, expected, *, scale=None):
    """Output, dQ, dK, dV within their BF16 bounds; the FP32 coefficient and log-decay gradients within
    1e-4 order by order (an order whose reference gradient is zero must be exactly zero). ``scale`` replaces
    the log-decay gradient's norm where a closed gate cancels it: c_r dc_r and the key sum agree to rounding
    while their difference vanishes."""
    for name, got, want in zip(NAMES, actual, expected, strict=True):
        got = np.asarray(got.astype(jnp.float32) if got.dtype == jnp.bfloat16 else got, np.float64)
        assert got.shape == want.shape and np.isfinite(got).all(), name
        if name in BF16_TOLERANCES:
            error = _relative_l2(got, want)
            assert error < BF16_TOLERANCES[name], f"{name}: relative L2 error {error:.3e}"
            continue
        for order in range(want.shape[-1]):
            part, reference = got[..., order], want[..., order]
            norm = np.linalg.norm(reference)
            if name == "log_decay" and scale is not None:
                norm = max(norm, scale[order])
            if norm == 0:
                np.testing.assert_array_equal(part, 0, err_msg=f"{name} order {order}")
                continue
            error = float(np.linalg.norm(part - reference) / norm)
            assert error < FP32_TOLERANCE, f"{name} order {order}: relative L2 error {error:.3e}"


GPU_CASES = (
    *((2048, degree, 64, "taylor") for degree in (1, 2, 4, 8, 12)),
    (2048, 12, 64, "top"),  # the top orders barely move the output and dQ, dK, dV at degree 12
    (2048, 12, 64, "signed"),
    *((512, 4, width, "taylor") for width in (8, 16, 32, 128)),
    (512, 12, 128, "taylor"),
    (128, 12, 64, "taylor"),  # one CTA
    (384, 12, 64, "signed"),  # an odd number of CTAs
)


@pytest.mark.gpu
@pytest.mark.parametrize("time,degree,width,variant", GPU_CASES,
                         ids=[f"t{t}-m{m}-dk{w}-{v}" for t, m, w, v in GPU_CASES])
def test_gpu_output_and_every_gradient_match_float64(time, degree, width, variant):
    _requires_h100()
    operands, cotangent = _gpu_inputs(time, degree, width=width, variant=variant)
    actual = jax.jit(lambda *x: _vjp(api.per_order_cumulative_decay_attention, x, cotangent))(*operands)
    _check(actual, _oracle(operands, cotangent))


@pytest.mark.gpu
@pytest.mark.parametrize("degree", (4, 12))
def test_gpu_gate_extremes_stay_finite_and_match_float64(degree):
    """Orders cycle through an open gate (log alpha 0), a 7,000-token half-life, a 14-token one, a gate
    closing within a token (log alpha -2), and gates that underflow at the next key (-100 per token, and
    -1e30, whose sums reach -2e33): every output and gradient is finite and matches FP64."""
    _requires_h100()
    operands, cotangent = _gpu_inputs(1024, degree)
    q, k, v, c, _ = operands
    rates = np.asarray([0.0, 1e-4, 0.05, 2.0, 100.0, 1e30])[np.arange(degree + 1) % 6]
    sums = jnp.cumsum(jnp.broadcast_to(jnp.asarray(-rates, jnp.float32), c.shape), axis=1)
    operands = (q, k, v, c, sums)
    actual = jax.jit(lambda *x: _vjp(api.per_order_cumulative_decay_attention, x, cotangent))(*operands)
    expected = _oracle(operands, cotangent)
    query_side = np.asarray(c, np.float64) * expected[4]  # c_r dc_r of the reference
    _check(actual, expected, scale=[np.linalg.norm(query_side[..., r]) for r in range(degree + 1)])


@pytest.mark.gpu
@pytest.mark.parametrize("degree", (4, 8, 12))
def test_gpu_production_dispatch_matches_the_strips_and_float64(degree):
    """BF16 Q/K/V at [1, 2048, 14, 64] through attention_core on Mosaic and on the causal strips (the
    run series' training path): the Mosaic output and gradients against host FP64 within the bounds above,
    and against the strips within the same bounds (the strips are within 1.2x the BF16 floor of FP64)."""
    _requires_h100()
    operands, cotangent = _gpu_inputs(2048, degree, heads=14, seed=31)

    def attend(backend, *arguments):
        q, k, v, c, sums = arguments
        return attention_core(q, k, v, backend=backend, attention_degree=degree, attention_normalization=False,
                              query_polynomial_coefficients=c, attention_coefficient_activation="tanh",
                              query_log_decay=sums, query_decay_mode="cumulative", blocked_implementation="strips")

    mosaic = jax.jit(lambda *x: _vjp(partial(attend, "mosaic"), x, cotangent))(*operands)
    strips = jax.jit(lambda *x: _vjp(partial(attend, "blocked"), x, cotangent))(*operands)
    _check(mosaic, _oracle(operands, cotangent))
    _check(mosaic, tuple(np.asarray(x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, np.float64)
                         for x in strips))


@pytest.mark.gpu
@pytest.mark.parametrize("degree", (4, 12))
def test_gpu_gate_projection_chain_rule_matches_float64(degree):
    """The model's route: forget_gate.log_decay of per-order parameters, the FP32 cumulative sum over time,
    and attention_core on Mosaic. The output and the gradients of Q, K, V, the coefficients, x, and every
    gate parameter (order by order) against an independent FP64 formula."""
    _requires_h100()
    rng = np.random.default_rng(degree)
    (q, k, v, c, _), cotangent = _gpu_inputs(512, degree)
    orders = degree + 1
    x = jnp.asarray(rng.standard_normal((1, 512, 32)), jnp.float32)
    kernel = jnp.asarray(rng.uniform(-1, 1, (2, orders, 32)) / math.sqrt(32), jnp.float32)
    log_rate = jnp.asarray(np.log(rng.uniform(0.2, 4.0, (2, orders, 1))), jnp.float32)
    dt_bias = jnp.asarray(rng.uniform(-6.0, -1.0, (2, orders, 1)), jnp.float32)
    operands = (q, k, v, c, x, kernel, log_rate, dt_bias)

    def oracle(q, k, v, c, x, w, a, b):
        alpha = -jnp.exp(a[..., 0]) * jnp.logaddexp(0.0, jnp.einsum("btd,hrd->bthr", x, w) + b[..., 0])
        return _dense(q, k, v, c, jnp.cumsum(alpha, axis=1))

    def production(q, k, v, c, x, w, a, b):
        return attention_core(q, k, v, backend="mosaic", attention_degree=degree, attention_normalization=False,
                              query_polynomial_coefficients=c, attention_coefficient_activation="tanh",
                              query_log_decay=jnp.cumsum(log_decay(x, w, a, b), axis=1),
                              query_decay_mode="cumulative")

    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        expected = jax.tree.map(np.asarray, _vjp(oracle, tuple(jnp.asarray(a, jnp.float64) for a in operands),
                                                 jnp.asarray(cotangent, jnp.float64)))
    actual = jax.jit(lambda *a: _vjp(production, a, cotangent))(*operands)
    names = ("output", "dq", "dk", "dv", "coefficients", "x", "kernel", "log_rate", "dt_bias")
    for name, got, want in zip(names, actual, expected, strict=True):
        got = np.asarray(got.astype(jnp.float32) if got.dtype == jnp.bfloat16 else got, np.float64)
        assert np.isfinite(got).all(), name
        if name in BF16_TOLERANCES:
            assert _relative_l2(got, want) < BF16_TOLERANCES[name], name
            continue
        axis = {"coefficients": -1, "kernel": 1, "log_rate": 1, "dt_bias": 1}.get(name)
        parts = ([(got, want)] if axis is None else
                 [(np.take(got, r, axis), np.take(want, r, axis)) for r in range(orders)])
        for order, (part, reference) in enumerate(parts):
            error = _relative_l2(part, reference)
            assert error < FP32_TOLERANCE, f"{name} order {order}: relative L2 error {error:.3e}"


@pytest.mark.gpu
@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
def test_gpu_invalid_sums_poison_exactly_their_causal_rows(bad):
    """A nonfinite sum of one order, head and token poisons that token's row and every later row of its
    batch and head (it enters their pairs as a key); every other row is bitwise unchanged. A nonfinite log
    alpha does the same through the cumulative sum."""
    _requires_h100()
    (q, k, v, c, sums), _ = _gpu_inputs(512, 4, batch=2)
    attend = jax.jit(api.per_order_cumulative_decay_attention)
    clean = np.asarray(attend(q, k, v, c, sums), np.float32)
    raw = jnp.diff(sums, axis=1, prepend=0.0)
    for changed in (sums.at[0, 300, 1, 2].set(bad), jnp.cumsum(raw.at[0, 300, 1, 2].set(bad), axis=1)):
        output = np.asarray(attend(q, k, v, c, changed), np.float32)
        invalid = ~np.isfinite(output).all(axis=-1)
        expected = np.zeros(invalid.shape, bool)
        expected[0, 300:, 1] = True
        np.testing.assert_array_equal(invalid, expected)
        np.testing.assert_array_equal(output[~expected], clean[~expected])


@pytest.mark.gpu
def test_gpu_diagonal_is_exactly_unattenuated():
    """With c = e_0 and every gate underflowing at the next key, each query keeps exactly its own value."""
    _requires_h100()
    (q, k, v, c, _), _ = _gpu_inputs(512, 4)
    c = jnp.zeros_like(c).at[..., 0].set(1.0)
    for rate in (100.0, 1e30):
        sums = jnp.cumsum(jnp.full(c.shape, -rate, jnp.float32), axis=1)
        output = api.per_order_cumulative_decay_attention(q, k, v, c, sums)
        np.testing.assert_array_equal(np.asarray(output, np.float32), np.asarray(v, np.float32))


@pytest.mark.gpu
@pytest.mark.parametrize("degree", (4, 12))
def test_gpu_constant_gates_match_the_per_order_query_distance_kernels(degree):
    """One log alpha per head and order at every token: each order's cumulative decay is exp((t - i) log
    alpha_r), so the cumulative kernels match H100's per-order query-distance kernels, the output and the
    gradients of Q, K, V, the coefficients, and the rates, within the FP32 cumulative sums' rounding (2e-3,
    the per-order kernels' duplicated-gate bound)."""
    _requires_h100()
    (q, k, v, c, _), cotangent = _gpu_inputs(2048, degree)
    rates = jnp.asarray(-np.exp(np.random.default_rng(3).uniform(math.log(3e-4), math.log(2.0), (2, degree + 1))),
                        jnp.float32)
    tokens = lambda a: jnp.broadcast_to(a, c.shape)
    cumulative = jax.jit(lambda *x: _vjp(lambda q, k, v, c, a: api.per_order_cumulative_decay_attention(
        q, k, v, c, jnp.cumsum(tokens(a), axis=1)), x, cotangent))(q, k, v, c, rates)
    distance = jax.jit(lambda *x: _vjp(lambda q, k, v, c, a: api.per_order_query_decay_attention(
        q, k, v, c, tokens(a)), x, cotangent))(q, k, v, c, rates)
    for name, got, want in zip(("output", "dq", "dk", "dv", "coefficients", "rates"), cumulative, distance, strict=True):
        got, want = (np.asarray(x.astype(jnp.float32), np.float64) for x in (got, want))
        assert np.isfinite(got).all() and _relative_l2(got, want) < 2e-3, name


@pytest.mark.gpu
def test_gpu_kernels_are_bitwise_repeatable():
    """No atomics and a fixed accumulation order: 20 repeats of the forward and every gradient at degree 12
    with several CTAs per SM (8 sequences of 2,048 tokens, 16 heads) are bitwise identical."""
    _requires_h100()
    operands, cotangent = _gpu_inputs(2048, 12, heads=16, batch=8, seed=98)
    run = jax.jit(lambda *x: _vjp(api.per_order_cumulative_decay_attention, x, cotangent))
    reference = jax.tree.map(np.asarray, run(*operands))
    for attempt in range(20):
        for index, (got, want) in enumerate(zip(jax.tree.map(np.asarray, run(*operands)), reference, strict=True)):
            assert np.array_equal(got, want, equal_nan=True), f"attempt {attempt}: leaf {index} differs"


@pytest.mark.gpu
def test_gpu_refuses_b300():
    """On B300 the entry point raises before building a kernel; on H100 it runs."""
    if jax.default_backend() != "gpu" or jax.devices()[0].compute_capability not in ("9.0", "10.0", "10.3"):
        pytest.skip("Requires an H100 or B300 allocation")
    operands, _ = _gpu_inputs(256, 12)
    if jax.devices()[0].compute_capability == "9.0":
        assert np.isfinite(np.asarray(api.per_order_cumulative_decay_attention(*operands), np.float32)).all()
    else:
        with pytest.raises(ValueError, match="runs on H100 only"):
            api.per_order_cumulative_decay_attention(*operands)


@pytest.mark.gpu
@pytest.mark.parametrize("degree", (4, 12))
def test_gpu_model_trains_like_the_float32_blocked_model(degree, monkeypatch):
    """End-to-end per-order cumulative model on Mosaic in BF16 against the FP32 blocked model: its loss and
    gradient direction, with the tolerances of the per-order query-distance kernels' model check. With
    FP32-precision weights (bf16x3) Mosaic is as faithful as the BF16 strips; its default rounds weights and
    score gradients to BF16 before the tensor-core products, as FlashAttention and cuDNN do."""
    _requires_h100()
    rng = np.random.default_rng(57)
    batch = jnp.asarray(rng.integers(0, 64, size=(2, 257)), jnp.int32)
    amplitude = jnp.asarray(rng.normal(size=(2, 2, degree + 1)) * .5, jnp.float32)
    gate_kernel = jnp.asarray(rng.normal(size=(2, 2, degree + 1, 128)) * .05, jnp.float32)  # orders differ
    shape = dict(vocab_size=64, max_seq_len=256, hidden_size=128, qk_head_dim=64, value_head_dim=64,
                 intermediate_size=256, attention_coefficient_projection_initial_std=128 ** -.5)
    results = {}
    for backend, dtype, precision in (("blocked", "float32", None), ("blocked", "bfloat16", None),
                                      ("mosaic", "bfloat16", "bf16"), ("mosaic", "bfloat16", "bf16x3")):
        monkeypatch.setattr(api, "default_precision", lambda family, degree, precision=precision: precision)
        config = _config(degree, attention_backend=backend, dtype=dtype, **shape)
        state = create_state(config, jax.random.PRNGKey(3))
        state.params["query_polynomial_amplitude"][...] = amplitude
        state.params["forget_gate_kernel"][...] = state.params["forget_gate_kernel"][...] + gate_kernel
        with jax.default_matmul_precision("highest" if dtype == "float32" else "default"):
            # A fresh function per case: the precision is read while tracing.
            loss, gradients = jax.jit(lambda *a: loop.loss_and_grads(*a))(state, batch, jax.random.PRNGKey(0))
        results[backend, dtype, precision] = float(loss), {path: np.asarray(value[...], np.float64)
                                                          for path, value in nnx.to_flat_state(gradients)}
    reference_loss, reference = results["blocked", "float32", None]
    bf16 = results["blocked", "bfloat16", None][1]
    loss, actual = results["mosaic", "bfloat16", "bf16"]
    assert math.isfinite(loss) and abs(loss - reference_loss) < .02
    assert all(np.isfinite(value).all() for value in actual.values())
    for name in ("forget_gate_kernel", "forget_gate_log_rate", "forget_gate_dt_bias"):
        assert np.all(np.linalg.norm(actual[(name,)].reshape(*actual[(name,)].shape[:3], -1), axis=-1) > 0), name

    def cosine(gradients):
        a, b = (np.concatenate([g[path].ravel() for path in sorted(reference, key=str)]) for g in (gradients, reference))
        return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

    exact = results["mosaic", "bfloat16", "bf16x3"][1]
    assert cosine(exact) >= min(.995, cosine(bf16) - .01), (cosine(exact), cosine(bf16))
    assert cosine(actual) >= min(.995, cosine(bf16) - .03), (cosine(actual), cosine(bf16))
