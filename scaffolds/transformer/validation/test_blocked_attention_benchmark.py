"""CPU checks of the blocked softmax candidates in blocked_attention_benchmark.py.

Run from the repository root:
  JAX_PLATFORMS=cpu PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m pytest -q \
    -c scaffolds/transformer/pyproject.toml --import-mode=importlib \
    scaffolds/transformer/validation/test_blocked_attention_benchmark.py
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention
from scaffolds.transformer.validation import blocked_attention_benchmark as bench


def lean(block, weights, *, checkpoint=True, causal_skip=True):
    return functools.partial(bench.lean_blocked_attention, block_size=block, weights=weights,
                             checkpoint=checkpoint, causal_skip=causal_skip)


def flash(block, weights, *, backward="columns"):
    return functools.partial(bench.flash_blocked_attention, block_size=block, weights=weights, backward=backward)


CANDIDATES = {
    "lean-skip-ckpt": lean,
    "lean-skip-nockpt": functools.partial(lean, checkpoint=False),
    "lean-scan-ckpt": functools.partial(lean, causal_skip=False),
    "lean-scan-nockpt": functools.partial(lean, checkpoint=False, causal_skip=False),
    "flash-columns": flash,
    "flash-rows": functools.partial(flash, backward="rows"),
    "flash-saved": functools.partial(flash, backward="saved"),
}


def inputs(shape, seed=0, sharp=2.0):
    rng = np.random.default_rng(seed)
    q, k, v, g = (rng.standard_normal(shape) for _ in range(4))
    return q * sharp, k, v, g


def run(attend, q, k, v, g):
    def program(q, k, v, g):
        output, pullback = jax.vjp(attend, q, k, v)
        return output, *pullback(g)

    return jax.jit(program)(*(jnp.asarray(x) for x in (q, k, v, g)))


def test_oracle_matches_float64_autodiff():
    check = bench.oracle_check()
    assert check["passed"], check
    assert check["worst_relative_l2"] < 1e-12


@pytest.mark.parametrize("length,block", [(1, 1), (1, 4), (37, 4), (37, 16), (32, 8), (37, 64)])
@pytest.mark.parametrize("weights", bench.WEIGHTS)
@pytest.mark.parametrize("candidate", sorted(CANDIDATES))
def test_float64_outputs_and_gradients_match_oracle(candidate, weights, length, block):
    q, k, v, g = inputs((2, length, 3, 8), seed=length + block)
    v = v[..., :5]
    g = g[..., :5]
    with jax.enable_x64():
        results = run(CANDIDATES[candidate](block, weights), q, k, v, g)
        assert all(x.dtype == jnp.float64 for x in results)
        for actual, expected in zip(results, bench.oracle(q, k, v, g), strict=True):
            np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("candidate", sorted(CANDIDATES))
def test_float32_matches_production_blocked(candidate):
    q, k, v, g = (x.astype(np.float32) for x in inputs((2, 45, 2, 16), seed=4))
    expected = run(functools.partial(blocked_dot_product_attention, query_block_size=8), q, k, v, g)
    actual = run(CANDIDATES[candidate](8, "float32"), q, k, v, g)
    reference = bench.oracle(q, k, v, g)
    for observed, production, exact in zip(actual, expected, reference, strict=True):
        assert observed.dtype == jnp.float32
        np.testing.assert_allclose(observed, production, rtol=2e-5, atol=2e-6)
        assert bench.relative_l2(observed, exact) < 1e-5


@pytest.mark.parametrize("candidate", sorted(CANDIDATES))
def test_bfloat16_dtypes_and_weight_rounding(candidate):
    q, k, v, g = (jnp.asarray(x, jnp.bfloat16) for x in inputs((1, 64, 2, 16), seed=5))
    exact = bench.oracle(*(np.asarray(x.astype(jnp.float32)) for x in (q, k, v, g)))
    results = {}
    for weights in bench.WEIGHTS:
        results[weights] = run(CANDIDATES[candidate](16, weights), q, k, v, g)
        assert all(x.dtype == jnp.bfloat16 for x in results[weights])
        for observed, reference in zip(results[weights], exact, strict=True):
            assert bench.relative_l2(np.asarray(observed.astype(jnp.float32)), reference) < 2e-2
    # Input-dtype weights round P (and the backward's weights) to BF16: results differ.
    assert any(not np.array_equal(np.asarray(a.astype(jnp.float32)), np.asarray(b.astype(jnp.float32)))
               for a, b in zip(results["float32"], results["input"], strict=True))


def test_gradient_sums_over_blocks_round_once_with_fp32_accumulation():
    """Autodiff over BF16 blocks sums dK/dV in BF16; the custom VJP and upcast operands do not."""
    q, k, v, g = (jnp.asarray(x, jnp.bfloat16) for x in inputs((1, 512, 2, 32), seed=8, sharp=1.0))
    exact = bench.oracle(*(np.asarray(x.astype(jnp.float32)) for x in (q, k, v, g)))
    floor = [bench.relative_l2(np.asarray(jnp.asarray(x, jnp.bfloat16).astype(jnp.float32)), x) for x in exact]

    def errors(attend):
        return [bench.relative_l2(np.asarray(x.astype(jnp.float32)), reference)
                for x, reference in zip(run(attend, q, k, v, g), exact, strict=True)]

    single = errors(functools.partial(blocked_dot_product_attention, query_block_size=512))
    blocked = errors(functools.partial(blocked_dot_product_attention, query_block_size=64))
    for attend in (flash(64, "float32"), flash(64, "float32", backward="rows"), flash(64, "float32", backward="saved"),
                   functools.partial(bench.lean_blocked_attention, block_size=64, upcast=True)):
        for observed, best in zip(errors(attend), floor, strict=True):
            assert observed < 1.05 * best
    for index in (2, 3):  # dK and dV: eight BF16 block sums in production
        assert single[index] < 1.05 * floor[index] < blocked[index] / 1.3


@pytest.mark.parametrize("backward", bench.BACKWARDS)
def test_custom_backward_starts_behind_a_barrier_with_the_cotangent(backward):
    """Keeps XLA from merging the backward's score products with the forward's across layers."""
    q = jnp.ones((1, 8, 1, 4))
    jaxpr = jax.make_jaxpr(lambda q, k, v, g: jax.vjp(flash(4, "float32", backward=backward), q, k, v)[1](g))(
        q, q, q, q)
    assert "optimization_barrier" in str(jaxpr)


@pytest.mark.parametrize("checkpoint", (True, False))
def test_upcast_lean_matches_float64_oracle(checkpoint):
    q, k, v, g = inputs((2, 37, 3, 8), seed=9)
    with jax.enable_x64():
        results = run(functools.partial(bench.lean_blocked_attention, block_size=8, checkpoint=checkpoint,
                                        upcast=True), q, k, v, g)
        for actual, expected in zip(results, bench.oracle(q, k, v, g), strict=True):
            np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-11, atol=1e-12)
    with pytest.raises(ValueError, match="upcast"):
        bench.lean_blocked_attention(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), weights="input", upcast=True)


@pytest.mark.parametrize("candidate", sorted(CANDIDATES))
def test_outputs_are_causal(candidate):
    q, k, v, _ = (x.astype(np.float32) for x in inputs((1, 37, 2, 8), seed=6))
    attend = jax.jit(CANDIDATES[candidate](8, "float32"))
    output = attend(q, k, v)
    changed = attend(np.concatenate([q[:, :19], -4 * q[:, 19:]], axis=1),
                     np.concatenate([k[:, :19], 3 * k[:, 19:]], axis=1),
                     np.concatenate([v[:, :19], v[:, 19:] + 2], axis=1))
    np.testing.assert_array_equal(np.asarray(changed[:, :19]), np.asarray(output[:, :19]))


def nonfinite_rows(attend, q, k, v, g, head=1):
    output, *grads = run(attend, q, k, v, g)
    output = np.asarray(output)
    bad = ~np.isfinite(output)
    others = bad.copy()
    others[0, :, head] = False
    return bad[0, :, head].any(-1), output, not others.any(), all(bool(np.isfinite(np.asarray(x)).all()) for x in grads)


@pytest.mark.parametrize("candidate", sorted(CANDIDATES))
def test_nonfinite_inputs_poison_rows_as_production(candidate):
    """Rows with a nonfinite causal score are NaN, as production; V and overflow propagate."""
    position, block = 17, 8
    base = [x.astype(np.float32) for x in inputs((1, 37, 2, 8), seed=7)]
    g = base[3]
    production = functools.partial(blocked_dot_product_attention, query_block_size=block)
    clean = np.asarray(jax.jit(production)(*base[:3]))
    for name, (q, k, v, expected) in bench.failure_cases(*base[:3], position=position).items():
        rows, output, others_finite, grads_finite = nonfinite_rows(CANDIDATES[candidate](block, "float32"), q, k, v, g)
        reference_rows, reference, _, _ = nonfinite_rows(production, q, k, v, g)
        assert others_finite, name
        assert rows[expected].all(), name
        assert not grads_finite, name
        first = position
        if name in ("v_nan", "v_inf"):
            # A masked zero weight times a nonfinite value is NaN: rows sharing the product with the
            # bad key may also be nonfinite (its strip with causal skipping; all rows in production).
            first = 0 if not candidate.startswith(("lean-skip", "flash")) else position // block * block
            assert reference_rows.all(), name
            # Rows before the bad key do not depend on it: compare with clean inputs.
            reference = clean
        elif name == "score_huge":
            # Huge finite scores in later rows may also overflow exp(s - max) after a fused
            # multiply-add; production behaves the same way, and earlier rows are unaffected.
            assert reference_rows[expected].all(), name
        else:
            np.testing.assert_array_equal(rows, reference_rows, err_msg=name)
            assert np.array_equal(rows, expected), name
        assert not rows[:first].any(), name
        finite = ~rows & np.isfinite(reference[0, :, 1]).all(-1)
        np.testing.assert_allclose(output[0, finite, 1], reference[0, finite, 1], rtol=2e-5, atol=2e-6)


def test_dense_manual_does_not_poison_negative_infinite_scores():
    """The textbook dense path (and cuDNN on GPU) treat a -inf score as a zero weight."""
    base = [x.astype(np.float32) for x in inputs((1, 37, 2, 8), seed=7)]
    q, k, v, _ = bench.failure_cases(*base[:3], position=17)["score_neg_inf"]
    output = np.asarray(jax.jit(bench.dense_manual_attention)(q, k, v))
    assert np.isfinite(output).all()
    assert np.isnan(np.asarray(jax.jit(flash(8, "float32"))(q, k, v))[0, 17:, 1]).all()


@pytest.mark.parametrize("arguments,message", [
    ({"block_size": 0}, "block_size"), ({"block_size": True}, "block_size"),
    ({"weights": "bfloat16"}, "weights"), ({"backward": "diagonal"}, "backward"),
])
def test_invalid_options_are_rejected(arguments, message):
    q = jnp.zeros((1, 4, 1, 2))
    with pytest.raises(ValueError, match=message):
        bench.flash_blocked_attention(q, q, q, **arguments)
    if "backward" not in arguments:
        with pytest.raises(ValueError, match=message):
            bench.lean_blocked_attention(q, q, q, **arguments)


def test_mismatched_shapes_are_rejected():
    q = jnp.zeros((1, 4, 2, 8))
    with pytest.raises(ValueError, match="shape"):
        bench.flash_blocked_attention(q, q[:, :3], q)
    with pytest.raises(ValueError, match="shape"):
        bench.lean_blocked_attention(q, q, q[..., 0])


def test_variant_table_builds_every_function():
    table = bench.variants()
    assert len(table) == len(set(table))
    q = jnp.zeros((1, 4, 1, 8), jnp.bfloat16)
    for name, variant in table.items():
        if variant.family != "cudnn":
            assert variant.function()(q, q, q).shape == q.shape, name


def test_routed_model_matches_production_blocked(monkeypatch):
    """The model benchmark's routing reproduces the scaffold's blocked loss and gradients."""
    import scaffolds.transformer.attention.softmax as softmax_module

    def tiny(backend):
        config = bench.load_model_config(bench.RECIPE, micro_batch=2, backend=backend)
        model = config.model
        model.num_layers, model.hidden_size, model.num_heads, model.num_kv_heads = 2, 32, 2, 2
        model.qk_head_dim, model.value_head_dim, model.intermediate_size = 16, 16, 64
        model.vocab_size = config.data.synthetic_vocab_size = 64
        model.max_seq_len, config.data.sequence_length = 32, 33
        config.train.dtype = "float32"
        return config

    batch = jnp.asarray(np.random.default_rng(0).integers(0, 64, (2, 33), dtype=np.int32))
    rng = jax.random.PRNGKey(1)
    state, program = bench.micro_step_program(tiny("blocked"), variant=None)
    expected_nll, expected_grads = jax.jit(program)(state.params, batch, rng)
    for attend in (flash(8, "float32"), flash(8, "input", backward="rows"), flash(8, "float32", backward="saved"),
                   lean(8, "float32")):
        monkeypatch.setattr(softmax_module, "softmax_attention", bench._routed_softmax(attend))
        state, program = bench.micro_step_program(tiny("blocked"), variant=None)
        nll, grads = jax.jit(program)(state.params, batch, rng)
        np.testing.assert_allclose(nll, expected_nll, rtol=1e-6)
        for observed, expected in zip(jax.tree.leaves(grads), jax.tree.leaves(expected_grads), strict=True):
            np.testing.assert_allclose(observed, expected, rtol=1e-4, atol=1e-6)
