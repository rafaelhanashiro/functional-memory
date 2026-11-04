"""Fused score-MLP attention: interpreted programs against float64, selection, and the compiled kernel.

CPU tests run the three Pallas programs in the interpreter with exact FP32 products and
interpreter-supported loads, against float64 autodiff of the dense definition. GPU tests
compile the same programs and compare BF16 production shapes with the host FP64 oracle (XLA
cannot compile the dense FP32 definition at 2,048 tokens on H100) and with the blocked path.
"""
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from scaffolds.transformer import acceptance
from scaffolds.transformer.attention import score_mlp
from scaffolds.transformer.attention.backends import fused_score_mlp as fused
from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention
from scaffolds.transformer.attention.backends.selection import resolve_attention_backend
from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.attention.multihead import MultiHeadAttention
from scaffolds.transformer.configs import ModelConfig, config_to_dict, save_config
from scaffolds.transformer.models import build_model
from scaffolds.transformer.tests.test_score_mlp_attention import (
    _config, _dense_units, _model, _numpy_dense_vjp, _units as _small_case, _vjp,
)
from scaffolds.transformer.training.diagnostics import attention_identity
from scaffolds.transformer.training.loop import cross_entropy_loss

WIDTHS = (1, 2, 3, 4, 8)
NAMES = ("output", "q", "k", "v", "a", "b", "w")


def _interpret(monkeypatch):
    """Run the three Triton programs in the Pallas CPU interpreter with exact FP32 products."""
    monkeypatch.setattr(fused, "_dot", lambda a, b: jnp.matmul(
        a.astype(jnp.float32), b.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST))

    def load(ref, rows, width, time):
        # Pallas' interpreter lacks mixed slice/array indexing; this load uses array indices only.
        columns = jnp.arange(ref.shape[-1])
        return fused.plt.load(ref.at[rows[:, None], columns[None, :]],
                              mask=(rows[:, None] < time) & (columns[None, :] < width), other=0)

    monkeypatch.setattr(fused, "_load", load)
    entry = fused.fused_score_mlp_attention
    monkeypatch.setattr(fused, "fused_score_mlp_attention",
                        lambda *operands, form: entry(*operands, form=form, interpret=True))


def _requires_gpu():
    if jax.default_backend() != "gpu":
        pytest.skip("Requires a CUDA allocation; the compiled fused kernel is under test")


def _inputs(width, *, time=67, batch=2, heads=2, qk=17, value=9, dtype=jnp.float32, seed=0):
    """Signed amplitudes and slopes, offsets spanning both SiLU tails; a, b, w are FP32."""
    rng = np.random.default_rng(seed + 31 * width + time)
    arrays = (rng.normal(size=(batch, time, heads, qk)), rng.normal(size=(batch, time, heads, qk)),
              rng.normal(size=(batch, time, heads, value)), rng.normal(size=(batch, time, heads, width)),
              2.0 * rng.normal(size=(batch, time, heads, width)), rng.uniform(-2.0, 2.0, size=(heads, width)))
    operands = tuple(jnp.asarray(x, dtype if index < 3 else jnp.float32) for index, x in enumerate(arrays))
    return operands, jnp.asarray(rng.normal(size=(batch, time, heads, value)), dtype)


def _production_inputs(form, width, *, batch=1, time=2048, heads=4, channels=64, seed=91):
    """BF16 Q/K/V and FP32 units around the form's recipe constants, as after some training."""
    rng = np.random.default_rng(seed + width)
    q, k, v = (rng.normal(size=(batch, time, heads, channels)) for _ in range(3))
    constants = score_mlp.initial_constants(width, form)
    alpha, beta, slope = (np.broadcast_to(np.asarray(constants[name]), (heads, width))
                          for name in ("amplitude", "offset", "slope"))
    a = alpha + (.1 + .1 * np.abs(alpha)) * rng.normal(size=(batch, time, heads, width))
    b = beta + .5 * rng.normal(size=(batch, time, heads, width))
    operands = tuple(jnp.asarray(x, jnp.bfloat16 if index < 3 else jnp.float32)
                     for index, x in enumerate((q, k, v, a, b, slope)))
    return operands, jnp.asarray(rng.normal(size=v.shape), jnp.bfloat16)


def _attend(form, backend="fused"):
    return lambda *operands: score_mlp.score_mlp_attention(*operands, form=form, backend=backend)


def _oracle(form, operands, cotangent):
    """float64 autodiff of the dense definition on the same (rounded) values."""
    with jax.enable_x64(), jax.default_matmul_precision("highest"):
        result = _vjp(lambda *x: _dense_units(*x, form=form), tuple(jnp.asarray(x, jnp.float64) for x in operands),
                      jnp.asarray(cotangent, jnp.float64))
        return jax.tree.map(lambda x: np.asarray(x, np.float64), result)


def _relative(observed, expected, label):
    observed, expected = np.asarray(observed, np.float64), np.asarray(expected, np.float64)
    assert observed.shape == expected.shape, label
    assert np.isfinite(observed).all() and np.isfinite(expected).all(), label
    scale = np.linalg.norm(expected)
    assert scale > 0, label
    return float(np.linalg.norm(observed - expected) / scale)


def _slope_magnitude(form, operands, cotangent):
    """sum_(b,t,i<=t) |D_tij s_ti| per head and unit in FP64, the scale of dw_hj's FP32 rounding.

    The terms of one unit's dw may cancel to a small fraction of this sum, so its error is
    measured against the sum rather than against dw itself.
    """
    q, k, v, a, b, w = (np.asarray(x, np.float64) for x in operands)
    qh, kh, vh, gh = (np.asarray(x, np.float64).transpose(0, 2, 1, 3) for x in (q, k, v, cotangent))
    ah, bh = a.transpose(0, 2, 1, 3), b.transpose(0, 2, 1, 3)
    causal = np.tril(np.ones((q.shape[1], q.shape[1]), dtype=bool))
    scores = np.where(causal, qh @ kh.swapaxes(-1, -2) / np.sqrt(q.shape[-1]), 0.0)
    cotangent = np.where(causal, gh @ vh.swapaxes(-1, -2), 0.0) * (np.exp(scores) if form == "exp_residual" else 1.0)
    magnitude = []
    for j in range(a.shape[-1]):
        pre = w[None, :, None, None, j] * scores + bh[..., j, None]
        gate = 1.0 / (1.0 + np.exp(-pre))
        terms = cotangent * ah[..., j, None] * gate * (1.0 + pre * (1.0 - gate)) * scores
        magnitude.append(np.sum(np.abs(terms), axis=(0, 2, 3)))
    return np.stack(magnitude, axis=-1)


def _compare(actual, expected, tolerances, slope_magnitude):
    """Output and every gradient: whole, per head, and per head and unit for a, b, w.

    A wrong reduction (a missing tile, unit, or head) cannot hide behind larger terms.
    Each dw_hj error is bounded by the tolerance times its terms' absolute sum.
    Returns each array's whole relative L2 error.
    """
    errors = {}
    for name, observed, wanted in zip(NAMES, (actual[0], *actual[1]), (expected[0], *expected[1]), strict=True):
        observed, wanted = np.asarray(observed, np.float64), np.asarray(wanted, np.float64)
        checks = {name: (observed, wanted)}
        if name == "w":
            assert np.isfinite(observed).all() and (slope_magnitude > 0).all()
            scaled = np.abs(observed - wanted) / slope_magnitude
            assert scaled.max() < tolerances[name], f"dw error over its terms' absolute sum: {scaled}"
        else:
            for h in range(wanted.shape[2]):
                checks[f"{name} head {h}"] = (observed[:, :, h], wanted[:, :, h])
                if name in ("a", "b"):
                    checks.update({f"{name} head {h} unit {j}": (observed[:, :, h, j], wanted[:, :, h, j])
                                   for j in range(wanted.shape[-1])})
        for label, (got, want) in checks.items():
            error = _relative(got, want, label)
            assert error < tolerances[name], f"{label}: relative L2 {error:.3g} >= {tolerances[name]}"
        errors[name] = _relative(observed, wanted, name)
    return errors


def _tolerances(dtype, *, rounded_bf16=5e-3, fp32=1e-5):
    """Rounding of BF16 outputs and Q/K/V gradients; FP32 for FP32 operands and the unit gradients."""
    rounded = rounded_bf16 if dtype == jnp.bfloat16 else fp32
    return {"output": rounded, "q": rounded, "k": rounded, "v": rounded, "a": fp32, "b": fp32, "w": fp32}


def _assert_dtypes(actual, dtype):
    assert actual[0].dtype == dtype and all(gradient.dtype == dtype for gradient in actual[1][:3])
    assert all(gradient.dtype == jnp.float32 for gradient in actual[1][3:])


@pytest.mark.parametrize(("time", "dtype"), ((67, jnp.float32), (130, jnp.float32), (130, jnp.bfloat16)),
                         ids=("t67-float32", "t130-float32", "t130-bfloat16"))
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("form", score_mlp.FORMS)
def test_interpreted_outputs_and_all_gradients_match_float64(monkeypatch, form, width, time, dtype):
    """Odd Q/K width 17, value width 9, two batches and heads; 67 and 130 tokens cross query and key tiles."""
    _interpret(monkeypatch)
    operands, cotangent = _inputs(width, time=time, dtype=dtype)
    actual = _vjp(_attend(form), operands, cotangent)
    _compare(actual, _oracle(form, operands, cotangent), _tolerances(dtype),
             _slope_magnitude(form, operands, cotangent))
    _assert_dtypes(actual, dtype)


@pytest.mark.parametrize(("time", "qk", "value"), ((1, 2, 1), (64, 64, 64), (65, 128, 128), (75, 72, 40)))
@pytest.mark.parametrize("form", score_mlp.FORMS)
def test_interpreted_edge_shapes_and_wide_heads_match_float64(monkeypatch, form, time, qk, value):
    """One token, exactly one tile, and widths above 64, which use 32-token tiles."""
    _interpret(monkeypatch)
    operands, cotangent = _inputs(3, time=time, batch=1, heads=2, qk=qk, value=value)
    actual = _vjp(_attend(form), operands, cotangent)
    _compare(actual, _oracle(form, operands, cotangent), _tolerances(jnp.float32),
             _slope_magnitude(form, operands, cotangent))
    assert fused._tile_sizes(qk, value) == ((64, 64) if max(qk, value) <= 64 else (32, 32))


@pytest.mark.parametrize("prefix", (37, 70))
@pytest.mark.parametrize("form", score_mlp.FORMS)
def test_interpreted_outputs_and_gradients_are_causal(monkeypatch, form, prefix):
    """Changing every later query, key, value, a, and b leaves earlier outputs and gradients exactly unchanged."""
    _interpret(monkeypatch)
    operands, cotangent = _inputs(4, time=130)
    cotangent = cotangent.at[:, prefix:].set(0)
    actual = _vjp(_attend(form), operands, cotangent)
    changed = tuple(x.at[:, prefix:].multiply(-3.0).at[:, prefix:].add(.5) if index < 5 else x
                    for index, x in enumerate(operands))
    altered = _vjp(_attend(form), changed, cotangent)
    np.testing.assert_array_equal(altered[0][:, :prefix], actual[0][:, :prefix])
    for name, left, right in zip(NAMES[1:], actual[1], altered[1], strict=True):
        if name == "w":
            np.testing.assert_array_equal(left, right)
            continue
        np.testing.assert_array_equal(left[:, :prefix], right[:, :prefix], err_msg=name)
        np.testing.assert_array_equal(left[:, prefix:], np.zeros_like(left[:, prefix:]), err_msg=name)
        assert np.abs(np.asarray(left[:, :prefix])).max() > 0, name


@pytest.mark.parametrize("width", (1, 4, 8))
def test_interpreted_exp_residual_start_is_the_unnormalized_exp_kernel(monkeypatch, width):
    """At a = 0, f = exp(s) exactly; b and w receive no gradient and a its exact one."""
    _interpret(monkeypatch)
    (q, k, v, a, b, w), cotangent = _inputs(width, time=67)
    a = jnp.zeros_like(a)
    actual = _vjp(_attend("exp_residual"), (q, k, v, a, b, w), cotangent)
    exponential = _vjp(lambda q, k, v: blocked_dot_product_attention(q, k, v, attention_normalization=False),
                       (q, k, v), cotangent)
    for observed, expected in zip((actual[0], *actual[1][:3]), (exponential[0], *exponential[1]), strict=True):
        assert _relative(observed, expected, "exp") < 2e-6
    np.testing.assert_array_equal(actual[1][4], np.zeros_like(actual[1][4]))
    np.testing.assert_array_equal(actual[1][5], np.zeros_like(actual[1][5]))
    oracle = _oracle("exp_residual", (q, k, v, a, b, w), cotangent)
    assert _relative(actual[1][3], oracle[1][3], "a") < 1e-5


@pytest.mark.parametrize("form", score_mlp.FORMS)
@pytest.mark.parametrize("case", ("score", "unit", "kernel", "output"))
def test_interpreted_nonfinite_scores_units_kernels_and_outputs_poison_only_their_rows(monkeypatch, form, case):
    _interpret(monkeypatch)
    q, k, v, a, b, w = _small_case()
    attend = _attend(form)
    if case == "output":
        # Large finite weights in head 1, so one huge value overflows every output that reads it.
        a, b = a.at[0, :, 1].set(1e3), b.at[0, :, 1].set(10.0)
    clean = attend(q, k, v, a, b, w)
    expected = np.zeros((1, 7, 2), dtype=bool)
    if case == "score":
        k = k.at[0, 4, 0].set(jnp.inf)        # every query that sees key 4 in head 0
        expected[0, 4:, 0] = True
    elif case == "unit":
        a = a.at[0, 5, 1, 2].set(jnp.nan)     # one query's unit amplitude in head 1
        expected[0, 5, 1] = True
    elif case == "kernel":
        a, b = a.at[0, 3, 0].set(1e37), b.at[0, 3, 0].set(100.0)   # finite units, infinite kernel
        expected[0, 3, 0] = True
    else:
        v = v.at[0, 2, 1, 0].set(3e38)        # finite weights and values, overflowing outputs from key 2 on
        expected[0, 2:, 1] = True
    output = np.asarray(attend(q, k, v, a, b, w))
    invalid = ~np.isfinite(output)
    np.testing.assert_array_equal(invalid.any(axis=-1), expected)
    np.testing.assert_array_equal(invalid.all(axis=-1), expected)   # a poisoned row is NaN in every channel
    np.testing.assert_array_equal(output[~expected], np.asarray(clean)[~expected])


@pytest.mark.parametrize("form", score_mlp.FORMS)
def test_interpreted_poisoning_crosses_tiles_and_nonfinite_weights_cannot_hide(monkeypatch, form):
    _interpret(monkeypatch)
    operands, _ = _inputs(3, time=130, batch=1)
    q, k, v, a, b, w = operands
    clean = np.asarray(_attend(form)(*operands))
    poisoned = np.asarray(_attend(form)(q, k.at[0, 70, 1].set(jnp.inf), v, a, b, w))
    assert np.isnan(poisoned[0, 70:, 1]).all() and np.isfinite(poisoned[0, :70, 1]).all()
    np.testing.assert_array_equal(poisoned[0, :70, 1], clean[0, :70, 1])
    np.testing.assert_array_equal(poisoned[:, :, 0], clean[:, :, 0])
    # An infinite weight poisons its row even where every value is zero.
    q = jnp.full((1, 3, 1, 1), 3.0, jnp.float32)
    a = jnp.ones((1, 3, 1, 2), jnp.float32).at[:, 1].set(1e38)
    output = _attend(form)(q, q, jnp.zeros_like(q), a, jnp.ones_like(a), jnp.ones((1, 2), jnp.float32))
    assert bool(jnp.isfinite(output[:, 0]).all()) and bool(jnp.isnan(output[:, 1]).all())
    assert bool(jnp.isfinite(output[:, 2]).all())


@pytest.mark.parametrize("form", score_mlp.FORMS)
def test_interpreted_masked_future_overflow_is_never_evaluated(monkeypatch, form):
    _interpret(monkeypatch)
    # Only the masked entry (query 0, key 1) has a huge score; evaluating it would overflow either form.
    q = jnp.array([1.0, 0.0]).reshape(1, 2, 1, 1)
    k = jnp.array([0.0, 1e30]).reshape(1, 2, 1, 1)
    v = jnp.array([3.0, 7.0]).reshape(1, 2, 1, 1)
    a, b, w = jnp.full((1, 2, 1, 2), 1e10), jnp.ones((1, 2, 1, 2)), jnp.ones((1, 2))
    output = _attend(form)(q, k, v, a, b, w)
    at_zero = 2e10 / (1.0 + np.exp(-1.0)) + (1.0 if form == "exp_residual" else 0.0)  # f(0)
    np.testing.assert_allclose(np.asarray(output).reshape(-1), [3 * at_zero, 10 * at_zero], rtol=1e-6)


def _valid_operands():
    (q, k, v, a, b, w), _ = _inputs(2, time=5, batch=1, heads=2, qk=4, value=3)
    return {"q": q, "k": k, "v": v, "amplitude": a, "offset": b, "slope": w, "form": "kernel"}


@pytest.mark.parametrize(("changes", "message"), (
    ({"form": "mixture"}, "Unknown score-MLP form"),
    ({"form": None}, "Unknown score-MLP form"),
    ({"q": jnp.zeros((5, 2, 4))}, r"\[batch,time,heads,channels\]"),
    ({"k": jnp.zeros((1, 6, 2, 4))}, "must match"),
    ({"v": jnp.zeros((1, 5, 3, 3))}, "must match"),
    ({"q": jnp.zeros((1, 5, 2, 129)), "k": jnp.zeros((1, 5, 2, 129))}, "up to 128"),
    ({"v": jnp.zeros((1, 5, 2, 129))}, "up to 128"),
    ({"v": jnp.zeros((1, 5, 2, 0))}, "positive"),
    ({"v": jnp.zeros((1, 5, 2, 3), jnp.bfloat16)}, "common float32 or bfloat16"),
    ({"q": jnp.zeros((1, 5, 2, 4), jnp.float16), "k": jnp.zeros((1, 5, 2, 4), jnp.float16),
      "v": jnp.zeros((1, 5, 2, 3), jnp.float16)}, "common float32 or bfloat16"),
    ({"amplitude": jnp.zeros((1, 5, 2, 2), jnp.bfloat16)}, "amplitudes and offsets must be float32"),
    ({"offset": jnp.zeros((1, 5, 2, 2), jnp.float16)}, "amplitudes and offsets must be float32"),
    ({"amplitude": jnp.zeros((1, 5, 2, 3)), "slope": jnp.zeros((2, 3))}, "amplitudes and offsets"),
    ({"amplitude": jnp.zeros((1, 4, 2, 2))}, "amplitudes and offsets"),
    ({"amplitude": jnp.zeros((1, 5, 2, 0)), "offset": jnp.zeros((1, 5, 2, 0)), "slope": jnp.zeros((2, 0))},
     "J>=1"),
    ({"slope": jnp.zeros((2, 3))}, "slopes must be float32 with shape"),
    ({"slope": jnp.zeros((1, 2))}, "slopes must be float32 with shape"),
    ({"slope": jnp.zeros((2, 2), jnp.bfloat16)}, "slopes must be float32 with shape"),
))
def test_fused_entry_validates_its_inputs(changes, message):
    with pytest.raises(ValueError, match=message):
        fused.fused_score_mlp_attention(**{**_valid_operands(), **changes})


@pytest.mark.parametrize("form", score_mlp.FORMS)
def test_dispatch_runs_the_fused_kernel_and_takes_diagnostics_from_the_blocked_reference(monkeypatch, form):
    _interpret(monkeypatch)
    q, k, v, a, b, w = _small_case()
    direct = fused.fused_score_mlp_attention(q, k, v, a, b, w, form=form)
    np.testing.assert_array_equal(_attend(form)(q, k, v, a, b, w), direct)
    core = attention_core(q, k, v, backend="fused", attention_normalization=False, score_mlp_amplitude=a,
                          score_mlp_offset=b, score_mlp_slope=w, score_mlp_form=form)
    np.testing.assert_array_equal(core, direct)
    output, diagnostics = score_mlp.score_mlp_attention(q, k, v, a, b, w, form=form, backend="fused",
                                                        return_diagnostics=True)
    _, blocked = score_mlp.score_mlp_attention(q, k, v, a, b, w, form=form, return_diagnostics=True)
    np.testing.assert_array_equal(output, direct)
    assert set(diagnostics) == set(blocked) and "score_mlp_amplitude_mean" in diagnostics
    for name, value in blocked.items():
        np.testing.assert_array_equal(diagnostics[name], value, err_msg=name)
    with pytest.raises(ValueError, match="forget gate runs on the blocked backend"):
        attention_core(q, k, v, backend="fused", attention_normalization=False, score_mlp_amplitude=a,
                       score_mlp_offset=b, score_mlp_slope=w, score_mlp_form=form,
                       query_log_decay=jnp.zeros(q.shape[:3]))


@pytest.mark.parametrize("width", (1, 4, 8))
@pytest.mark.parametrize("form", score_mlp.FORMS)
def test_configuration_accepts_fused_and_auto_only_without_a_gate(form, width):
    for backend in ("fused", "auto"):
        config = _config(form, width, attention_backend=backend)
        assert config.model.attention_backend == backend and not config.model.mosaic_supported()
        with pytest.raises(ValueError, match="gated score MLP runs on model.attention_backend blocked"):
            _config(form, width, attention_backend=backend, attention_forget_gate=True)
    assert _config(form, width, attention_forget_gate=True).model.attention_backend == "blocked"
    with pytest.raises(ValueError, match="up to 128"):
        _config(form, width, attention_backend="fused", qk_head_dim=130)


@pytest.mark.parametrize("dtype", ("float32", "bfloat16"))
@pytest.mark.parametrize("form", score_mlp.FORMS)
def test_backend_resolution_selects_fused_only_for_an_ungated_score_mlp_on_a_capable_gpu(form, dtype):
    resolve = lambda model, platform: resolve_attention_backend(model, dtype=dtype, platform=platform)
    fused_model, auto, blocked = (_config(form, attention_backend=backend).model
                                  for backend in ("fused", "auto", "blocked"))
    assert [resolve(model, "gpu") for model in (fused_model, auto, blocked)] == ["fused", "fused", "blocked"]
    assert [resolve(model, "cpu") for model in (auto, blocked)] == ["blocked", "blocked"]
    with pytest.raises(ValueError, match="Ampere-or-newer"):
        resolve(fused_model, "cpu")
    # Heads wider than 128 stay blocked under auto and reject an explicit fused request.
    wide = {"qk_head_dim": 130, "value_head_dim": 130}
    assert resolve(ModelConfig(**_model(form, attention_backend="auto", **wide)), "gpu") == "blocked"
    with pytest.raises(ValueError, match="up to 128"):
        resolve(ModelConfig(**_model(form, attention_backend="fused", **wide)), "gpu")
    # A gated score MLP never reaches the fused kernel, even if a request bypassed validation.
    gated = _config(form, attention_forget_gate=True).model
    assert [resolve(gated, platform) for platform in ("gpu", "cpu")] == ["blocked", "blocked"]
    gated.attention_backend = "auto"
    assert resolve(gated, "gpu") == "blocked"
    gated.attention_backend = "fused"
    with pytest.raises(ValueError, match="gated score MLP"):
        resolve(gated, "gpu")
    gated.attention_backend = "mosaic"
    with pytest.raises(ValueError, match="'auto', 'blocked' or 'fused'"):
        resolve(gated, "gpu")


def test_module_accepts_the_fused_backend_without_a_gate():
    settings = {"attention_normalization": False, "score_mlp_width": 4, "backend": "fused"}
    assert MultiHeadAttention(16, 2, 2, 4, 4, rngs=nnx.Rngs(0), **settings).backend == "fused"
    with pytest.raises(ValueError, match="gated score MLP runs on the blocked backend"):
        MultiHeadAttention(16, 2, 2, 4, 4, rngs=nnx.Rngs(0), attention_forget_gate=True, **settings)


@pytest.mark.parametrize("backend", ("fused", "auto"))
def test_records_create_accepts_a_fused_recipe_without_the_triton_opt_out(tmp_path, monkeypatch, capsys, backend):
    from scaffolds.records import validate_record
    from scaffolds.records.__main__ import main

    source = tmp_path / "score_mlp.yaml"
    save_config(_config("exp_residual", 8, attention_backend=backend), source)
    monkeypatch.setattr("sys.argv", ["records", "create", "--study", str(tmp_path / "study"), "--config", str(source)])
    main()
    run = Path(capsys.readouterr().out.strip())
    model = yaml.safe_load((run / "config.yaml").read_text())["model"]
    assert (model["attention_backend"], model["attention_score_mlp_width"]) == (backend, 8)
    validate_record(run)


def test_acceptance_runs_the_fused_tests_for_fused_recipes_only():
    spec = acceptance.AcceptanceSpec(entry_module="tests", cpu_tests=(), gpu_tests=())
    for form in score_mlp.FORMS:
        recipe, blocked = config_to_dict(_config(form, attention_backend="fused")), config_to_dict(_config(form))
        auto = config_to_dict(_config(form, attention_backend="auto"))
        for cpu in (True, False):
            for config, resolved in ((recipe, "fused"), (auto, "fused")):
                paths = acceptance.test_paths(spec, config, resolved_backend=resolved, cpu=cpu)
                assert {acceptance.SCORE_MLP_TESTS, acceptance.SCORE_MLP_FUSED_TESTS} <= set(paths)
            for config in (blocked, auto):
                paths = acceptance.test_paths(spec, config, resolved_backend="blocked", cpu=cpu)
                assert acceptance.SCORE_MLP_TESTS in paths and acceptance.SCORE_MLP_FUSED_TESTS not in paths
        production = acceptance.attention_spec(_config(form, attention_backend="fused"), entry_module="tests")
        assert production.required_backend == "fused"
        assert acceptance.SCORE_MLP_FUSED_TESTS in acceptance.test_paths(production, recipe, resolved_backend="fused",
                                                                         cpu=False)


def test_identity_names_the_fused_implementation_and_keeps_the_blocked_one():
    identity = attention_identity(_config("exp_residual", 4, attention_backend="fused"))
    assert identity["attention_kernel"] == "exp_score_mlp_residual_unnormalized"
    assert identity["attention_implementation"] == "pallas_triton_score_mlp"
    assert identity["attention_score_mlp_backward"] == "pallas_triton_query_and_key_tiles"
    assert identity["diagnostic_backend"] == "blocked"
    blocked = attention_identity(_config("exp_residual", 4))
    assert blocked["attention_implementation"] == "blocked"
    assert blocked["attention_score_mlp_backward"] == "custom_vjp_over_key_chunks"
    # Everything else, including the formula and the frozen constants, is the blocked run's.
    shared = set(identity) & set(blocked) - {"attention_backend", "attention_backend_requested",
                                             "attention_implementation", "query_block_size",
                                             "attention_score_mlp_backward"}
    assert {key: identity[key] for key in shared} == {key: blocked[key] for key in shared}


def _model_loss_and_gradients(form, backend, tokens, *, seed=5, **changes):
    """Loss and parameter gradients of a whole model with query-dependent units (A, B nonzero)."""
    model = build_model(_config(form, 4, attention_backend=backend, **changes).model, seed)
    graphdef, params = nnx.split(model, nnx.Param)
    rng = np.random.default_rng(seed)
    for name in score_mlp.PARAMETERS:
        params[name][...] = params[name][...] + jnp.asarray(.3 * rng.normal(size=params[name][...].shape), jnp.float32)

    def loss(params):
        return cross_entropy_loss(nnx.merge(graphdef, params)(tokens[:, :-1]), tokens[:, 1:])

    value, gradients = jax.jit(jax.value_and_grad(loss))(params)
    return float(value), {path: np.asarray(leaf[...]) for path, leaf in nnx.to_flat_state(gradients)}


def _compare_models(form, tokens, *, tolerance, **changes):
    """Both backends from identical parameters: an integration check, so a misrouted array fails by O(1).

    Signed weights f = exp(s) (1 + u) cancel within rows that head RMSNorm then normalizes, so
    FP32 agreement is about 2e-5 here; the kernels themselves are bounded against FP64 above.
    """
    fused_loss, fused_gradients = _model_loss_and_gradients(form, "fused", tokens, **changes)
    blocked_loss, blocked_gradients = _model_loss_and_gradients(form, "blocked", tokens, **changes)
    assert abs(fused_loss - blocked_loss) <= tolerance * abs(blocked_loss)
    assert set(fused_gradients) == set(blocked_gradients)
    for path, expected in blocked_gradients.items():
        assert _relative(fused_gradients[path], expected, str(path)) < tolerance, path
    return fused_loss, blocked_loss


@pytest.mark.parametrize("form", score_mlp.FORMS)
def test_interpreted_model_matches_the_blocked_model(monkeypatch, form):
    """Two layers, 70 tokens (two query tiles): loss and every parameter gradient."""
    _interpret(monkeypatch)
    tokens = jnp.asarray(np.random.default_rng(3).integers(0, 16, size=(2, 71)), jnp.int32)
    _compare_models(form, tokens, tolerance=1e-4, max_seq_len=70)


def _production_errors(form, width, **shape):
    """Relative L2 errors of the fused BF16 kernel against the host FP64 oracle on the same BF16 values."""
    operands, cotangent = _production_inputs(form, width, **shape)
    expected = _numpy_dense_vjp(operands, cotangent, form=form)
    actual = jax.jit(lambda *x: _vjp(_attend(form), x, cotangent))(*operands)
    _assert_dtypes(actual, jnp.bfloat16)
    return _compare(actual, expected, _tolerances(jnp.bfloat16, rounded_bf16=.012, fp32=1e-4),
                    _slope_magnitude(form, operands, cotangent))


@pytest.mark.parametrize("form", score_mlp.FORMS)
def test_interpreted_bfloat16_qualification_runs_at_a_small_shape(monkeypatch, form):
    """The GPU qualification's comparison and recipe-constant units, at a CPU-sized context."""
    _interpret(monkeypatch)
    errors = _production_errors(form, 4, time=160, heads=2, channels=16)
    assert max(errors[name] for name in ("a", "b", "w")) < 1e-5, errors


@pytest.mark.gpu
@pytest.mark.parametrize("width", (4, 8))
@pytest.mark.parametrize("form", score_mlp.FORMS)
def test_gpu_production_shape_bfloat16_matches_float64_oracle(form, width):
    """2,048 tokens, 4 heads, 64-channel BF16 Q/K/V: output and every gradient, each head and unit, below 0.012."""
    _requires_gpu()
    _production_errors(form, width)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16))
@pytest.mark.parametrize("form", score_mlp.FORMS)
def test_gpu_odd_shapes_match_float64_oracle(form, dtype):
    """131 tokens, Q/K width 33, value width 17, J = 3: partial tiles and padded channels on the device."""
    _requires_gpu()
    operands, cotangent = _inputs(3, time=131, batch=2, heads=3, qk=33, value=17, dtype=dtype)
    actual = jax.jit(lambda *x: _vjp(_attend(form), x, cotangent))(*operands)
    _compare(actual, _numpy_dense_vjp(operands, cotangent, form=form),
             _tolerances(dtype, rounded_bf16=.012, fp32=1e-4), _slope_magnitude(form, operands, cotangent))
    _assert_dtypes(actual, dtype)


@pytest.mark.gpu
@pytest.mark.parametrize("width", (4, 8))
@pytest.mark.parametrize("form", score_mlp.FORMS)
def test_gpu_production_shape_matches_blocked_and_repeats_bitwise(form, width):
    _requires_gpu()
    operands, cotangent = _production_inputs(form, width)
    attend = jax.jit(lambda *x: _vjp(_attend(form), x, cotangent))
    first, second = attend(*operands), attend(*operands)
    for left, right in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    blocked = jax.jit(lambda *x: _vjp(_attend(form, "blocked"), x, cotangent))(*operands)
    _compare(first, blocked, _tolerances(jnp.bfloat16, rounded_bf16=.012, fp32=1e-4),
             _slope_magnitude(form, operands, cotangent))


@pytest.mark.gpu
@pytest.mark.parametrize("width", (4, 8))
def test_gpu_exp_residual_start_matches_the_fused_exp_kernel(width):
    """At a = 0 the fused residual kernel computes the fused unnormalized exp kernel's function."""
    _requires_gpu()
    from scaffolds.transformer.attention.backends.fused_unnormalized import fused_unnormalized_attention

    (q, k, v, a, b, w), cotangent = _production_inputs("exp_residual", width)
    a = jnp.zeros_like(a)
    actual = jax.jit(lambda *x: _vjp(_attend("exp_residual"), x, cotangent))(q, k, v, a, b, w)
    exponential = jax.jit(lambda *x: _vjp(fused_unnormalized_attention, x, cotangent))(q, k, v)
    for name, observed, expected in zip(NAMES, (actual[0], *actual[1][:3]), (exponential[0], *exponential[1])):
        assert _relative(observed, expected, name) < 1e-3, name
    np.testing.assert_array_equal(np.asarray(actual[1][4]), 0)
    np.testing.assert_array_equal(np.asarray(actual[1][5]), 0)


@pytest.mark.gpu
@pytest.mark.parametrize("form", score_mlp.FORMS)
def test_gpu_outputs_and_gradients_are_causal(form):
    _requires_gpu()
    operands, cotangent = _production_inputs(form, 4)
    prefix = 1000
    cotangent = cotangent.at[:, prefix:].set(0)
    attend = jax.jit(lambda *x: _vjp(_attend(form), x, cotangent))
    actual = attend(*operands)
    altered = attend(*(x.at[:, prefix:].multiply(-3).at[:, prefix:].add(.5) if index < 5 else x
                       for index, x in enumerate(operands)))
    np.testing.assert_array_equal(np.asarray(altered[0][:, :prefix]), np.asarray(actual[0][:, :prefix]))
    for name, left, right in zip(NAMES[1:], actual[1], altered[1], strict=True):
        left, right = np.asarray(left), np.asarray(right)
        if name == "w":
            np.testing.assert_array_equal(left, right)
            continue
        np.testing.assert_array_equal(left[:, :prefix], right[:, :prefix], err_msg=name)
        np.testing.assert_array_equal(left[:, prefix:], 0, err_msg=name)


@pytest.mark.gpu
@pytest.mark.parametrize("form", score_mlp.FORMS)
def test_gpu_nonfinite_scores_and_weights_poison_only_their_rows(form):
    _requires_gpu()
    attend = jax.jit(_attend(form))
    operands, _ = _production_inputs(form, 4, time=256, heads=2)
    q, k, v, a, b, w = operands
    clean = np.asarray(attend(*operands), np.float32)
    poisoned = np.asarray(attend(q, k.at[0, 70, 1].set(jnp.inf), v, a, b, w), np.float32)
    assert np.isnan(poisoned[0, 70:, 1]).all() and np.isfinite(poisoned[0, :70, 1]).all()
    np.testing.assert_array_equal(poisoned[0, :70, 1], clean[0, :70, 1])
    np.testing.assert_array_equal(poisoned[:, :, 0], clean[:, :, 0])
    q = jnp.full((1, 3, 1, 1), 3.0, jnp.float32)
    a = jnp.ones((1, 3, 1, 2), jnp.float32).at[:, 1].set(1e38)
    output = attend(q, q, jnp.zeros_like(q), a, jnp.ones_like(a), jnp.ones((1, 2), jnp.float32))
    assert bool(jnp.isfinite(output[:, 0]).all()) and bool(jnp.isnan(output[:, 1]).all())
    assert bool(jnp.isfinite(output[:, 2]).all())
    q = jnp.array([1.0, 0.0]).reshape(1, 2, 1, 1)
    k = jnp.array([0.0, 1e30]).reshape(1, 2, 1, 1)
    output = attend(q, k, jnp.array([3.0, 7.0]).reshape(1, 2, 1, 1), jnp.full((1, 2, 1, 2), 1e10),
                    jnp.ones((1, 2, 1, 2)), jnp.ones((1, 2)))
    at_zero = 2e10 / (1.0 + np.exp(-1.0)) + (1.0 if form == "exp_residual" else 0.0)
    np.testing.assert_allclose(np.asarray(output).reshape(-1), [3 * at_zero, 10 * at_zero], rtol=1e-6)


@pytest.mark.gpu
@pytest.mark.parametrize("form", score_mlp.FORMS)
def test_gpu_model_shape_compiles_with_finite_outputs_and_gradients(form):
    """Micro-batch 16, 14 heads, 2,048 tokens, J = 8, against the blocked path on the same GPU."""
    _requires_gpu()
    operands, cotangent = _production_inputs(form, 8, batch=16, heads=14)
    actual = jax.jit(lambda *x: _vjp(_attend(form), x, cotangent))(*operands)
    for leaf in jax.tree.leaves(actual):
        assert bool(jnp.isfinite(leaf).all())
    blocked = jax.jit(lambda *x: _vjp(_attend(form, "blocked"), x, cotangent))(*operands)
    for name, observed, expected in zip(NAMES, (actual[0], *actual[1]), (blocked[0], *blocked[1]), strict=True):
        assert _relative(observed, expected, name) < (.012 if name in ("output", "q", "k", "v") else 1e-4), name


@pytest.mark.gpu
@pytest.mark.parametrize("form", score_mlp.FORMS)
def test_gpu_model_matches_the_blocked_model(form):
    """A whole two-layer model at 256 tokens: the same loss and parameter gradients as the blocked backend."""
    _requires_gpu()
    tokens = jnp.asarray(np.random.default_rng(3).integers(0, 16, size=(2, 257)), jnp.int32)
    with jax.default_matmul_precision("highest"):
        _compare_models(form, tokens, tolerance=3e-4, max_seq_len=256, hidden_size=32, num_heads=2, num_kv_heads=2,
                        qk_head_dim=16, value_head_dim=16, intermediate_size=48)
