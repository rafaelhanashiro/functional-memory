"""CPU checks of the H100 Mosaic per-degree score-bias kernels (``mosaic/score_bias.py``).

Every kernel lowers for cuda on a CPU host in exactly the shared memory ``hopper_smem_bytes``
counts (one byte less fails); the production entry points lower forward and with every gradient;
B300 and other GPUs refuse before a kernel is traced; the support predicate and implementation
names; rejected operands; and the tile mathematics of ``common`` (the functions the kernels call
on each tile), driven on whole causal matrices in FP64 against an explicit all-pairs reference,
including exactly zero and negative shifted scores and the transposed dK tile.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.backends.mosaic import api, common, score_bias

FAMILIES = (("none", "row"), ("none", "pair"), ("endpoint", "row"), ("endpoint", "pair"))
FAMILY_IDS = ("row", "pair", "endpoint-row", "endpoint-pair")


def _lowers_for_cuda(function, *operands):
    """Trace and lower ``function`` for cuda on any host, building every Mosaic GPU kernel (through
    a fresh wrapper: JAX would reuse an earlier lowering of the same function)."""
    jax.jit(lambda *arguments: function(*arguments)).trace(*operands).lower(lowering_platforms=("cuda",))


def test_counts_follow_the_operand_layout():
    """Vectors: c (m+1) [+ gate u (m+1)] + shift (m); key vectors: [gate v (m+1)] + pair q (m);
    heads: [gate rates (m+1)] + pair b, a (2m); dQ sums: dc (m+1) [+ two gate kinds] + one (row)
    or three (pair) shift kinds of m; dK sums: [gate (m+1)] + pair (m)."""
    for degree in range(1, 13):
        orders = degree + 1
        for gate, shift in FAMILIES:
            family = score_bias.score_bias_family(gate, shift)
            gated, paired = gate == "endpoint", shift == "pair"
            vectors = orders * (2 if gated else 1) + degree
            sums = orders * (3 if gated else 1) + degree * (3 if paired else 1)
            keys = (orders if gated else 0) + (degree if paired else 0)
            assert common.vector_counts(family, degree) == (vectors, vectors, sums)
            assert common.key_counts(family, degree) == (keys, keys)
            assert common.head_count(family, degree) == (orders if gated else 0) + (2 * degree if paired else 0)
            assert common.approx_math(family) and common.has_pair(family) and common.accumulates(family)
            assert family in common.PER_ORDER_FAMILIES and (family in common.DECAY_FAMILIES) == gated
    # Head scalars pad to whole 128-byte rows: 32 through degree 10 with the gate, 64 above.
    assert [common.head_width(common.head_count(common.SCORE_BIAS_ENDPOINT_PAIR, m)) for m in (8, 10, 11, 12)] == [
        32, 32, 64, 64]


def test_support_predicate_names_and_families():
    for gate, shift in FAMILIES:
        assert [m for m in range(15) if score_bias.score_bias_supported(m, 64, 2048, gate=gate, shift=shift)] == list(
            range(1, 13))
        assert all(score_bias.score_bias_supported(12, width, 2048, gate=gate, shift=shift) for width in api.WIDTHS)
        for width, time in ((48, 2048), (64, 2000), (64, 0)):
            assert not score_bias.score_bias_supported(8, width, time, gate=gate, shift=shift)
        assert not score_bias.score_bias_supported(True, 64, 2048, gate=gate, shift=shift)
        name = score_bias.implementation_name(gate, shift)
        assert name.startswith("pallas_mosaic_gpu_query_polynomial") and name.endswith(f"score_bias_{shift}")
        assert ("per_order_endpoint_decay" in name) == (gate == "endpoint")
    assert len({score_bias.implementation_name(*f) for f in FAMILIES}) == 4
    assert score_bias.SHIFT_OF_MODE == {"static": "row", "query": "row", "query_key": "pair"}
    for gate, shift in (("query", "row"), ("none", "static"), ("endpoint", None)):
        with pytest.raises(ValueError):
            score_bias.score_bias_supported(8, 64, 2048, gate=gate, shift=shift)


@pytest.mark.parametrize("degree", (4, 8))
@pytest.mark.parametrize("gate,shift", FAMILIES, ids=FAMILY_IDS)
def test_hopper_kernels_lower_in_exactly_their_counted_shared_memory(gate, shift, degree):
    """H100's forward, dQ and dK/dV lower at 2,048 tokens and d_K = 64 in exactly the bytes
    hopper_smem_bytes counts, and fail one byte below."""
    from jax.experimental.mosaic.gpu.core import artificial_shared_memory_limit
    from scaffolds.transformer.attention.backends.mosaic import hopper

    family = score_bias.score_bias_family(gate, shift)
    n_vectors, n_keys, n_heads = (common.vector_counts(family, degree)[0], common.key_counts(family, degree)[0],
                                  common.head_count(family, degree))
    time, width = 2048, 64
    q, v = (jax.ShapeDtypeStruct((1, time, 2, size), jnp.bfloat16) for size in (width, 64))
    vectors = jax.ShapeDtypeStruct((n_vectors, 1, 2, time), jnp.float32)
    keys = jax.ShapeDtypeStruct((n_keys, 1, 2, time), jnp.float32) if n_keys else None
    heads = jax.ShapeDtypeStruct((2, common.head_width(n_heads)), jnp.float32) if n_heads else None
    options = dict(family=family, degree=degree, scale=width ** -.5, precision="bf16")
    kernels = {"forward": (lambda q, k, v, x, y, z: hopper.forward(q, k, v, x, y, z, **options), (q, q, v)),
               "dq": (lambda q, k, v, do, x, y, z: hopper.dq(q, k, v, do, x, y, z, **options), (q, q, v, v)),
               "dkv": (lambda q, k, v, do, x, y, z: hopper.dkv(q, k, v, do, x, y, z, **options), (q, q, v, v))}
    sizes = common.hopper_smem_bytes(family, degree, width, 64)
    assert max(sizes.values()) <= common.HOPPER_SMEM_BYTES
    for name, (kernel, arrays) in kernels.items():
        with artificial_shared_memory_limit(sizes[name]):
            _lowers_for_cuda(kernel, *arrays, vectors, keys, heads)
        with artificial_shared_memory_limit(sizes[name] - 1), pytest.raises(ValueError, match="shared memory"):
            _lowers_for_cuda(kernel, *arrays, vectors, keys, heads)


def _operands(gate, shift, degree, *, time=2048, heads=2):
    """Shape structs of (q, k, v, c, gate operands..., shift operands...) and the keyword names of
    the shift operands."""
    q = jax.ShapeDtypeStruct((1, time, heads, 64), jnp.bfloat16)
    orders = jax.ShapeDtypeStruct((1, time, heads, degree + 1), jnp.float32)
    shifts = jax.ShapeDtypeStruct((1, time, heads, degree), jnp.float32)
    head = jax.ShapeDtypeStruct((heads, degree), jnp.float32)
    gate_operands = (orders, orders, jax.ShapeDtypeStruct((heads, degree + 1, 1), jnp.float32)) if gate == "endpoint" else ()
    if shift == "row":
        return (q, q, q, orders, *gate_operands, shifts), ("row_shift",)
    return (q, q, q, orders, *gate_operands, shifts, shifts, head, head), (
        "query_shift_input", "key_shift_input", "shift_offset", "shift_amplitude")


def _attention(gate, names):
    def attend(q, k, v, c, *operands):
        gate_operands = operands[:3] if gate == "endpoint" else ()
        shift = dict(zip(names, operands[len(gate_operands):], strict=True))
        if gate == "endpoint":
            return score_bias.score_bias_per_order_endpoint_attention(q, k, v, c, *gate_operands, **shift)
        return score_bias.score_bias_query_polynomial_attention(q, k, v, c, **shift)
    return attend


@pytest.mark.parametrize("gate,shift", FAMILIES, ids=FAMILY_IDS)
def test_entry_points_lower_on_h100_and_refuse_every_other_gpu(monkeypatch, gate, shift):
    """At d_K = 64 and 2,048 tokens the production entry points lower forward and with every
    gradient for H100 at the runs' degree eight and the largest, twelve; B300's kernels (and any
    other GPU's) are never reached: the entry points raise before tracing one."""
    from scaffolds.transformer.attention.backends.mosaic import hopper

    monkeypatch.setattr(api, "_kernels", lambda: ("hopper", hopper))
    for degree in (8, 12):
        operands, names = _operands(gate, shift, degree)
        attend = _attention(gate, names)
        _lowers_for_cuda(attend, *operands)
        _lowers_for_cuda(lambda *a: jax.vjp(attend, *a)[1](jnp.ones(a[2].shape, a[2].dtype)), *operands)
    operands, names = _operands(gate, shift, 1, time=256)
    monkeypatch.setattr(api, "_kernels", lambda: ("blackwell", None))  # no kernel may be reached
    with pytest.raises(ValueError, match="H100 only"):
        _lowers_for_cuda(_attention(gate, names), *operands)

    def no_gpu():
        raise ValueError("Mosaic GPU attention supports compute capability 9.0 and 10.x, not 8.0")
    monkeypatch.setattr(api, "_kernels", no_gpu)
    with pytest.raises(ValueError, match="H100"):
        _lowers_for_cuda(_attention(gate, names), *operands)


def test_blackwell_kernels_refuse_score_bias_families():
    from scaffolds.transformer.attention.backends.mosaic import blackwell

    q = jax.ShapeDtypeStruct((1, 256, 2, 64), jnp.bfloat16)
    for gate, shift in FAMILIES:
        family = score_bias.score_bias_family(gate, shift)
        vectors = jax.ShapeDtypeStruct((common.vector_counts(family, 4)[0], 1, 2, 256), jnp.float32)
        options = dict(family=family, degree=4, scale=.125, precision="bf16")
        for kernel in (lambda q, x: blackwell.forward(q, q, q, x, **options),
                       lambda q, x: blackwell.dq(q, q, q, q, x, **options),
                       lambda q, x: blackwell.dkv(q, q, q, q, x, which="dk", **options)):
            with pytest.raises(ValueError, match="H100 only"):
                jax.eval_shape(kernel, q, vectors)


@pytest.mark.parametrize("change", ("both shifts", "no shift", "row dtype", "row shape", "pair offset shape",
                                    "pair amplitude dtype", "pair input shape", "degree", "width", "time",
                                    "precision", "gate input shape", "gate rate shape", "unsupported degree"))
def test_invalid_operands_raise_before_tracing(monkeypatch, change):
    from scaffolds.transformer.attention.backends.mosaic import hopper

    monkeypatch.setattr(api, "_kernels", lambda: ("hopper", hopper))
    rng = np.random.default_rng(3)
    shape = (1, 256, 2)
    f32 = lambda *s: jnp.asarray(rng.standard_normal(s), jnp.float32)
    q = k = v = jnp.zeros((*shape, 64), jnp.bfloat16)
    c, beta = f32(*shape, 5), f32(*shape, 4)
    pair = dict(query_shift_input=f32(*shape, 4), key_shift_input=f32(*shape, 4), shift_offset=f32(2, 4),
                shift_amplitude=f32(2, 4))
    gate = (f32(*shape, 5), f32(*shape, 5), f32(2, 5, 1))
    kwargs, gated = dict(row_shift=beta), change.startswith("gate")
    if change == "both shifts":
        kwargs |= pair
    elif change == "no shift":
        kwargs = {}
    elif change == "row dtype":
        kwargs["row_shift"] = beta.astype(jnp.bfloat16)
    elif change == "row shape":
        kwargs["row_shift"] = beta[..., :-1]
    elif change.startswith("pair"):
        kwargs = dict(pair)
        name = {"pair offset shape": "shift_offset", "pair amplitude dtype": "shift_amplitude",
                "pair input shape": "key_shift_input"}[change]
        kwargs[name] = kwargs[name].astype(jnp.bfloat16) if "dtype" in change else kwargs[name][..., :-1]
    elif change == "degree":
        c, kwargs["row_shift"] = f32(*shape, 14), f32(*shape, 13)
    elif change == "width":
        q = k = jnp.zeros((*shape, 48), jnp.bfloat16)
    elif change == "time":
        q, k, v, c, kwargs["row_shift"] = (x[:, :200] for x in (q, k, v, c, beta))
    elif change == "precision":
        kwargs["precision"] = "fp8"
    elif change == "gate input shape":
        gate = (gate[0][..., :-1], *gate[1:])
    elif change == "gate rate shape":
        gate = (*gate[:2], gate[2][:, :-1])
    elif change == "unsupported degree":  # within MAX_QUERY_DEGREE, but not in H100's shared memory
        monkeypatch.setattr(api, "hopper_supported", lambda family, degree, width: False)
    with pytest.raises(ValueError):
        if gated:
            score_bias.score_bias_per_order_endpoint_attention(q, k, v, c, *gate, **kwargs)
        else:
            score_bias.score_bias_query_polynomial_attention(q, k, v, c, **kwargs)


# Tile mathematics: the functions every kernel calls on its tiles, on one whole causal tile.

class _RefSums:
    """The kernels' accumulator protocol (``common.accumulates``) on a JAX ref."""

    def __init__(self, count, length):
        self.count, self.ref = count, jax.new_ref(jnp.zeros((max(count, 1), length), jnp.float64))

    def add(self, index, x):
        self.ref[index] = self.ref[index] + x

    def value(self):
        return np.asarray(self.ref[...])[:self.count]


def _tile_terms(family, degree, s, dp, vectors, keys, heads):
    """(W, dS, dQ sums, W^T, dS^T, dK sums) from ``common`` on the whole causal matrix as one tile
    crossing the diagonal, and its transpose as the dK pass's tile."""
    time = s.shape[0]
    rows, cols = np.arange(time)[:, None], np.arange(time)[None, :]
    causal = jnp.asarray(rows >= cols)
    shape = (time, time)

    def broadcast(stack, axis):
        return common.Lazy(stack.shape[0], lambda i: jnp.broadcast_to(jnp.expand_dims(stack[i], 1 - axis), shape))

    def pair(transposed):
        distance = None
        if family in common.DECAY_FAMILIES:
            distance = jnp.asarray((cols - rows) if transposed else (rows - cols), jnp.float64)
        return common.Pair(distance, broadcast(keys, 0 if transposed else 1) if len(keys) else (),
                           causal.T if transposed else causal,
                           common.Lazy(len(heads), lambda i: heads[i]) if len(heads) else ())

    _, _, n_sums = common.vector_counts(family, degree)
    n_key_sums = common.key_counts(family, degree)[1]
    s, dp = jnp.where(causal, s, 0.), jnp.where(causal, dp, 0.)
    w = jnp.where(causal, common.forward_weights(family, degree, s, broadcast(vectors, 0), pair(False)), 0.)
    sums, key_sums = _RefSums(n_sums, time), _RefSums(n_key_sums, time)
    ds, _ = common.dq_terms(family, degree, s, dp, broadcast(vectors, 0), pair(False), sums=sums)
    wt, dst, _ = common.dkv_terms(family, degree, s.T, dp.T, broadcast(vectors, 1), pair(True), sums=key_sums)
    return (w, jnp.where(causal, ds, 0.), sums.value(), jnp.where(causal.T, wt, 0.), jnp.where(causal.T, dst, 0.),
            key_sums.value())


def _tile_reference(degree, s, dp, c, gate, shift):
    """Explicit per-order W, dS and every dQ and dK sum, in the families' layouts."""
    time = s.shape[0]
    rows, cols = np.arange(time)[:, None], np.arange(time)[None, :]
    causal = rows >= cols
    weights, derivative = np.zeros_like(s), np.zeros_like(s)
    g = np.where(causal, dp, 0.)
    row, key = {}, {}
    for r in range(degree + 1):
        factor = np.ones_like(s)
        if gate is not None:
            u, w, rate = gate
            x = u[r][:, None] + w[r][None, :]
            factor = np.exp(rate[r] * np.maximum(rows - cols, 0) * np.logaddexp(0., x))
            gwd = np.where(causal, g * c[r][:, None] * factor * (rows - cols), 0.)
        power, slope, tau = np.ones_like(s), np.zeros_like(s), None
        if r:
            if shift[0] == "row":
                beta = shift[1][r - 1][:, None]
            else:
                p, q, b, a = shift[1:]
                tau = np.tanh(p[r - 1][:, None] + q[r - 1][None, :])
                beta = b[r - 1] + a[r - 1] * tau
            z = s + beta
            power, slope = z ** r, r * z ** (r - 1)
        weights += c[r][:, None] * factor * power
        row.setdefault("dc", []).append((g * factor * power).sum(1))
        if gate is not None:
            sigmoid, log2_softplus = 1 / (1 + np.exp(-x)), np.logaddexp(0., x) / math.log(2)
            row.setdefault("sigmoid", []).append((gwd * power * sigmoid).sum(1))
            row.setdefault("log2_softplus", []).append((gwd * power * log2_softplus).sum(1))
            key.setdefault("sigmoid", []).append((gwd * power * sigmoid).sum(0))
        if r:
            term = np.where(causal, c[r][:, None] * factor * slope, 0.)
            derivative += term
            dbeta = g * term
            if tau is not None:
                row.setdefault("sech2", []).append((dbeta * (1 - tau ** 2)).sum(1))
                row.setdefault("tanh", []).append((dbeta * tau).sum(1))
                key.setdefault("sech2", []).append((dbeta * (1 - tau ** 2)).sum(0))
            row.setdefault("dbeta", []).append(dbeta.sum(1))
    kinds = ["dc"] + (["sigmoid", "log2_softplus"] if gate is not None else []) + (
        ["sech2", "tanh", "dbeta"] if shift[0] == "pair" else ["dbeta"])
    key_kinds = (["sigmoid"] if gate is not None else []) + (["sech2"] if shift[0] == "pair" else [])
    sums = np.stack([x for kind in kinds for x in row[kind]])
    key_sums = np.stack([x for kind in key_kinds for x in key[kind]]) if key_kinds else np.zeros((0, time))
    weights, ds = np.where(causal, weights, 0.), np.where(causal, dp * derivative, 0.)
    return weights, ds, sums, weights.T, ds.T, key_sums


TILE_CASES = [pytest.param(gate, shift, degree, "random", id=f"{name}-m{degree}")
              for (gate, shift), name in zip(FAMILIES, FAMILY_IDS) for degree in (1, 4, 8)] + [
    pytest.param(gate, "row", 8, "exact zeros", id=f"{name}-m8-exact-zeros")
    for gate, name in (("none", "row"), ("endpoint", "endpoint-row"))]


@pytest.mark.parametrize("gate,shift,degree,case", TILE_CASES)
def test_tile_mathematics_match_explicit_float64(gate, shift, degree, case):
    """The forward weights, dS and every dQ and dK sum of a tile crossing the diagonal, against
    explicit (s + beta_r)^r over all pairs. ``exact zeros`` draws integer scores and sets the row
    shift beta_r(t) = -s(t, i_r) for a random causal key: many shifted scores are exactly zero,
    about half the others negative. Ungated tiles agree to rounding; gated ones to the gate's own
    approximation (log2(1 + y) by its series below 2^-6, relative error < 1e-6)."""
    family = score_bias.score_bias_family(gate, shift)
    rng = np.random.default_rng(degree)
    time, orders = 37, degree + 1
    if case == "exact zeros":
        qk = rng.integers(-1, 2, size=(2, time, 16)).astype(np.float64)
        s = qk[0] @ qk[1].T / 4
    else:
        s = rng.standard_normal((time, time))
    dp = rng.standard_normal((time, time))
    c = rng.standard_normal((orders, time)) / np.array([math.factorial(r) for r in range(orders)])[:, None]
    vectors, keys, heads, gate_operands = [c], [], [], None
    if gate == "endpoint":
        gate_operands = (rng.uniform(-4, -1, (orders, time)), .5 * rng.standard_normal((orders, time)),
                         -np.exp(rng.uniform(-1, 1, orders)))
        vectors.append(gate_operands[0])
        keys.append(gate_operands[1])
        heads.append(gate_operands[2])
    if shift == "pair":
        p, q = rng.standard_normal((degree, time)), rng.standard_normal((degree, time))
        b, a = .5 * rng.standard_normal(degree), 1 + .3 * rng.standard_normal(degree)
        shift_operands = ("pair", p, q, b, a)
        vectors.append(p)
        keys.append(q)
        heads += [b, a]
    else:
        beta = rng.standard_normal((degree, time))
        if case == "exact zeros":
            for r in range(degree):
                beta[r] = -s[np.arange(time), rng.integers(0, np.arange(time) + 1)]
            z = s[:, :, None] + beta.T[:, None, :]
            causal = np.tril(np.ones((time, time), bool))
            assert (z[causal] == 0).mean() > .05 and (z[causal] < 0).mean() > .3
        shift_operands = ("row", beta)
        vectors.append(beta)
    with jax.enable_x64():
        got = _tile_terms(family, degree, jnp.asarray(s), jnp.asarray(dp), jnp.asarray(np.concatenate(vectors)),
                          jnp.asarray(np.concatenate(keys)) if keys else (), jnp.asarray(np.concatenate(heads)) if heads
                          else ())
        got = [np.asarray(x) for x in got]
    want = _tile_reference(degree, s, dp, c, gate_operands, shift_operands)
    tolerance = 2e-6 if gate == "endpoint" else 1e-12
    for name, x, y in zip(("W", "dS", "dQ sums", "W^T", "dS^T", "dK sums"), got, want, strict=True):
        assert x.shape == y.shape, (name, x.shape, y.shape)
        if y.size:
            assert np.max(np.abs(x - y)) <= tolerance * np.max(np.abs(y)), (name, np.max(np.abs(x - y)))
