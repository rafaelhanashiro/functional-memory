"""Query-key generators regressed onto free query-key pair coefficients."""
from dataclasses import replace
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest

from scaffolds.transformer.configs import Config
from scaffolds.transformer.models import build_model
from scaffolds.transformer.tests.test_free_query_key_fit import _taylor
from scaffolds.transformer.training.coefficient_distillation_model import make_student_losses
from scaffolds.transformer.training.query_key_distillation import (
    build_student, make_regression_functions, make_teacher_functions, pack_generator,
    predict_pair_factorial, query_key_table_chunks, replace_generator, sample_pairs,
)

FACTORIAL = np.array([math.factorial(r) for r in range(5)], np.float64)


@pytest.fixture(scope="module")
def parent():
    config = Config(model=_taylor(max_seq_len=8))
    config.train.dtype = "float32"
    model = build_model(config.model, 7)
    graph, params = nnx.split(model, nnx.Param)
    return config, graph, params


def _tokens():
    return jnp.arange(18).reshape(2, 9) % 16


def test_student_starts_at_its_taylor_parent_and_round_trips(parent):
    config, graph, params = parent
    student_config, student_graph, student_params, audit = build_student(config, params, backend="blocked")
    assert student_config.model.attention_coefficient_conditioning == "query_key"
    tokens = _tokens()
    np.testing.assert_allclose(make_student_losses(student_graph)(student_params, tokens),
                               make_student_losses(graph)(params, tokens), rtol=1e-5, atol=1e-6)
    packed = pack_generator(student_params)
    assert set(packed) == {"kernel", "bias", "amplitude", "key_kernel", "key_amplitude"}
    changed = {key: value + .1 for key, value in packed.items()}
    for key, value in pack_generator(replace_generator(student_params, changed)).items():
        np.testing.assert_array_equal(value, changed[key])


def test_pair_prediction_matches_the_generator_formula(parent):
    config, _, params = parent
    _, _, student_params, _ = build_student(config, params, backend="blocked")
    rng = np.random.default_rng(1)
    generator = {key: jnp.asarray(np.asarray(value) + .3 * rng.normal(size=value.shape), jnp.float32)
                 for key, value in pack_generator(student_params).items()}
    xq, xk = (jnp.asarray(rng.normal(size=(2, 6, 8)), jnp.float32) for _ in range(2))
    g = {key: np.asarray(value, np.float64) for key, value in generator.items()}
    query = (g["bias"][:, :, None] + g["amplitude"][:, :, None] * np.tanh(
        np.einsum("lpd,lhdr->lhpr", np.asarray(xq, np.float64), g["kernel"]))) / FACTORIAL
    key = 1 + g["key_amplitude"][:, :, None] * np.tanh(
        np.einsum("lpd,lhdr->lhpr", np.asarray(xk, np.float64), g["key_kernel"]))
    expected = np.swapaxes(query * key, 1, 2) * FACTORIAL
    np.testing.assert_allclose(predict_pair_factorial(generator, xq, xk), expected, rtol=2e-5, atol=2e-6)


def test_sampled_pairs_are_causal_and_uniform():
    queries, keys = sample_pairs(jax.random.key(0), sequence_count=3, length=6, size=200_000)
    queries, keys = np.asarray(queries), np.asarray(keys)
    assert np.array_equal(queries // 6, keys // 6) and np.all(keys % 6 <= queries % 6)
    counts = np.zeros((3, 6, 6))
    np.add.at(counts, (queries // 6, queries % 6, keys % 6), 1)
    causal = counts[:, np.tril_indices(6)[0], np.tril_indices(6)[1]]
    assert np.all(causal > 0) and abs(causal.std() / causal.mean()) < .05
    assert counts.sum() == causal.sum()


def test_teacher_capture_records_layer_inputs_and_replays(parent):
    config, graph, params = parent
    tokens = _tokens()
    _, coefficients = nnx.merge(graph, params)(tokens[:, :-1], return_coefficients=True)
    coefficients = jnp.swapaxes(coefficients, 0, 1)
    key_factors = jnp.ones_like(coefficients).at[:, :, 3].multiply(1.5)
    capture, native = make_teacher_functions(graph)
    observed = capture(params, tokens, coefficients, key_factors)
    assert observed["inputs"].shape == (2, 2, 8, 8)
    np.testing.assert_allclose(observed["nll"], native(params, tokens, coefficients, key_factors), rtol=1e-6)
    # Unit key factors reproduce the parent; the first layer's input is its normalized embedding.
    np.testing.assert_allclose(native(params, tokens, coefficients, jnp.ones_like(coefficients)),
                               make_student_losses(graph)(params, tokens), rtol=1e-5, atol=1e-6)
    model = nnx.merge(graph, params)
    first = model.layers_0.attention_norm(model.token_embedding(tokens[:, :-1]))
    np.testing.assert_allclose(observed["inputs"][:, 0], first, rtol=1e-6, atol=1e-6)


def test_regression_learns_pair_targets_and_moves_every_group(parent):
    config, _, params = parent
    _, _, student_params, _ = build_student(config, params, backend="blocked")
    generator = pack_generator(student_params)
    rng = np.random.default_rng(2)
    xq, xk = (jnp.asarray(rng.normal(size=(2, 64, 8)), jnp.float32) for _ in range(2))
    target_generator = {key: value + jnp.asarray(.3 * rng.normal(size=value.shape), jnp.float32)
                        for key, value in generator.items()}
    targets = predict_pair_factorial(target_generator, xq, xk)
    init, step, evaluate = make_regression_functions(clip_norm=1.0)
    optimizer, current = init(generator), generator
    start = float(evaluate(current, xq, xk, targets)["factorial_mse"])
    for _ in range(200):
        current, optimizer, metrics = step(current, optimizer, xq, xk, targets, 1e-2)
        assert bool(metrics["finite"])
    assert float(evaluate(current, xq, xk, targets)["factorial_mse"]) < .2 * start
    for key in generator:
        assert float(jnp.abs(current[key] - generator[key]).max()) > 0, key


def test_table_reader_returns_each_microbatch_pair(tmp_path):
    shape = (2, 2, 8, 2, 5)
    tables = [(np.full(shape, 1. + index, np.float32), np.full(shape, 10. + index, np.float32)) for index in range(3)]
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(tmp_path / "best", {"coefficients": tables})
    recipe = {"model": {"num_layers": 2, "num_heads": 2, "attention_degree": 4}, "data": {"sequence_length": 9},
              "oracle": {"sequence_count": 6}}
    chunks = list(query_key_table_chunks(tmp_path, recipe))
    assert [offset for offset, _, _ in chunks] == [0, 2, 4]
    for index, (_, coefficients, key_factors) in enumerate(chunks):
        np.testing.assert_array_equal(coefficients, tables[index][0])
        np.testing.assert_array_equal(key_factors, tables[index][1])
