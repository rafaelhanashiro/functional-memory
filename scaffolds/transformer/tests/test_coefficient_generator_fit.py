"""Generator adaptation updates only the original coefficient network."""
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from scaffolds.transformer.configs import Config, ModelConfig
from scaffolds.transformer.models import build_model
from scaffolds.transformer.training.checkpointing import state_sha256
from scaffolds.transformer.training.coefficient_fit import (
    GENERATOR_PARAMETER_NAMES, make_generator_functions, qualify, restore_state,
    save_state, split_generator_parameters, _generator_group_norms,
    convert_generator_readout, make_conversion_diagnostics, reference_backend_agreement,
)

KEY_PARAMETER_NAMES = {"key_polynomial_kernel", "key_polynomial_amplitude"}
DIAGONAL_PARAMETER_NAMES = GENERATOR_PARAMETER_NAMES - {"query_polynomial_mixing"} - KEY_PARAMETER_NAMES
QUERY_KEY_PARAMETER_NAMES = {"query_polynomial_bias", "query_polynomial_amplitude", "query_polynomial_kernel",
                             *KEY_PARAMETER_NAMES}


@pytest.fixture(scope="module")
def fitting():
    config = ModelConfig(
        vocab_size=16, max_seq_len=7, num_layers=2, hidden_size=8, num_heads=2,
        num_kv_heads=2, qk_head_dim=2, value_head_dim=4, intermediate_size=12,
        attention_degree=8, attention_normalization=False, attention_coefficients="learned",
        attention_coefficient_sharing="per_head", attention_coefficient_conditioning="query",
        attention_coefficient_activation="tanh", attention_coefficient_scale="factorial",
        attention_coefficient_amplitude_scale="factorial", attention_coefficient_tanh_bias=True,
        attention_coefficient_projection_initial_std=.2, attention_coefficient_amplitude_initial=.2,
        attention_backend="blocked", attention_output_scaling="head_rms_norm",
        qk_norm=True, qk_norm_type="rms_norm", qk_norm_sharing="per_layer",
        rms_norm_fp32=True, initialization="transformer_truncated_reference", remat=False,
    )
    graph, source = nnx.split(build_model(config, 7), nnx.Param)
    options = dict(clip_norm=1., beta1=.9, beta2=.95, epsilon=1e-8)
    functions = make_generator_functions(graph, options)
    return graph, source, jnp.arange(8)[None] % 16, options, functions


def _equal(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(left, right)


def test_split_replays_source_and_only_generator_parameters_change(fitting):
    graph, source, tokens, _, (capture, initialize, step, evaluate) = fitting
    generator, frozen = split_generator_parameters(source)
    assert set(generator.keys()) == DIAGONAL_PARAMETER_NAMES
    assert sum(value.size for value in jax.tree.leaves(generator)) == 2 * 2 * 9 * (8 + 3)
    frozen_hash = state_sha256(frozen)
    source_hash = state_sha256(source)
    initial, parent_nll, replay_nll, logit_error = capture(source, tokens)
    _equal(initial, generator)
    np.testing.assert_array_equal(parent_nll, replay_nll)
    assert float(logit_error) == 0.
    np.testing.assert_allclose(evaluate(initial, source, tokens), parent_nll, rtol=1e-5, atol=1e-6)
    optimizer = initialize(initial)
    current = initial
    for _ in range(5):
        current, optimizer, _, norm, finite, groups = step(current, optimizer, source, tokens, 1e-3)
        assert bool(finite) and float(norm[0]) > 0
        assert all(np.isfinite(value) and float(value) > 0 for value in groups.values())
        np.testing.assert_allclose(sum(value ** 2 for value in groups.values()), norm[0] ** 2,
                                   rtol=2e-5, atol=1e-9)
    assert float(evaluate(current, source, tokens)[0]) < float(parent_nll[0])
    updated = nnx.state(nnx.merge(graph, frozen, current), nnx.Param)
    actual_generator, actual_frozen = split_generator_parameters(updated)
    _equal(actual_generator, current)
    _equal(actual_frozen, frozen)
    assert state_sha256(source) == source_hash
    assert state_sha256(actual_frozen) == frozen_hash
    for key in DIAGONAL_PARAMETER_NAMES:
        assert not np.array_equal(current[key][...], initial[key][...])


def test_step_uses_stored_parameter_gradients_without_extra_factorials(fitting):
    _, source, tokens, options, (capture, initialize, step, evaluate) = fitting
    initial = capture(source, tokens)[0]
    gradient = jax.jit(jax.grad(lambda values: evaluate(values, source, tokens).mean()))(initial)
    expected_norm = optax.global_norm(gradient)
    clip = jnp.minimum(1., options["clip_norm"] / expected_norm)
    expected = jax.tree.map(
        lambda value, grad: value - 1e-3 * (grad * clip) / (jnp.abs(grad * clip) + options["epsilon"]),
        initial, gradient,
    )
    result = step(initial, initialize(initial), source, tokens, 1e-3)
    np.testing.assert_allclose(result[3][0], expected_norm, rtol=1e-5, atol=1e-8)
    for actual, desired in zip(jax.tree.leaves(result[0]), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(actual, desired, rtol=2e-5, atol=2e-7)


def test_later_generator_inputs_recompute_and_propagate_early_changes(fitting):
    graph, source, tokens, _, (capture, _, _, evaluate) = fitting
    generator = capture(source, tokens)[0]
    _, frozen = split_generator_parameters(source)
    # A first-layer coefficient change modifies later-layer coefficient inputs,
    # while the later generator's own parameters remain identical.
    shifted = jax.tree.map(lambda value: value, generator)
    shifted["query_polynomial_bias"][...] = shifted["query_polynomial_bias"][...].at[0, :, 0].add(.5)
    _, initial_coefficients = nnx.merge(graph, frozen, generator)(tokens[:, :-1], return_coefficients=True)
    _, changed_coefficients = nnx.merge(graph, frozen, shifted)(tokens[:, :-1], return_coefficients=True)
    assert not np.allclose(initial_coefficients[1], changed_coefficients[1], rtol=1e-6, atol=1e-9)
    # Finite differences include the changed downstream generator activations.
    direction = jax.tree.map(jnp.zeros_like, generator)
    direction["query_polynomial_bias"][...] = direction["query_polynomial_bias"][...].at[0, :, 0].set(1.)
    gradient = jax.jit(jax.grad(lambda values: evaluate(values, source, tokens).mean()))(generator)
    eps = 2e-3
    plus = jax.tree.map(lambda v, d: v + eps * d, generator, direction)
    minus = jax.tree.map(lambda v, d: v - eps * d, generator, direction)
    numerical = (evaluate(plus, source, tokens).mean() - evaluate(minus, source, tokens).mean()) / (2 * eps)
    analytic = sum(jnp.sum(g * d) for g, d in zip(jax.tree.leaves(gradient), jax.tree.leaves(direction)))
    np.testing.assert_allclose(numerical, analytic, rtol=.06, atol=5e-5)


def test_generator_checkpoint_resume_and_runtime_acceptance(fitting, tmp_path):
    graph, source, tokens, _, functions = fitting
    capture, initialize, step, _ = functions
    initial = capture(source, tokens)[0]
    optimizer = initialize(initial)
    updated, moments = step(initial, optimizer, source, tokens, 1e-3)[:2]
    controller = dict(step=1, best_step=1, best_loss=2., history=[2.1, 2.], learning_rate=1e-3)
    root = tmp_path / "checkpoints"
    root.mkdir()
    path = save_state(root, 1, [updated], [moments], [updated], controller)
    restored, restored_controller = restore_state(path, [updated], [moments])
    assert restored_controller == controller
    _equal(restored["tables"][0], updated)
    _equal(step(restored["tables"][0], restored["optimizers"][0], source, tokens, 1e-3),
           step(updated, moments, source, tokens, 1e-3))
    report = qualify(tmp_path, functions, source, tokens, initial, optimizer, 1e-3,
                     graph=graph, generator=True)
    assert report["checkpoint_resume_exact"]
    assert report["frozen_parameter_scope"] == "backbone"
    assert set(report["generator_gradient_norms"]) == DIAGONAL_PARAMETER_NAMES
    assert report["one_update_nll"] < report["separate_training_forward_nll"]


def test_generator_fit_rejects_invalid_input_axes(fitting):
    _, source, tokens, _, (capture, _, _, evaluate) = fitting
    initial = capture(source, tokens)[0]
    with pytest.raises(ValueError, match="axes"):
        evaluate(initial, source, tokens[None, None])


def _converted(fitting, readout):
    graph, source, tokens, options, functions = fitting
    if readout == "diagonal":
        return graph, source, tokens, options, functions
    config = Config(model=nnx.merge(graph, source).config)
    config.train.dtype = "float32"
    graph, source = convert_generator_readout(config, graph, source, readout)
    return graph, source, tokens, options, make_generator_functions(graph, options)


def test_dense_conversion_preserves_trained_generator_and_backbone(fitting):
    graph, source, tokens, options, (capture, initialize, step, _) = fitting
    generator = capture(source, tokens)[0]
    trained = step(generator, initialize(generator), source, tokens, 1e-3)[0]
    _, frozen = split_generator_parameters(source)
    source = nnx.state(nnx.merge(graph, frozen, trained), nnx.Param)
    source_hash = state_sha256(source)
    config = Config(model=nnx.merge(graph, source).config)
    config.train.dtype = "float32"
    dense_graph, dense_source = convert_generator_readout(config, graph, source, "dense")
    dense_generator, dense_frozen = split_generator_parameters(dense_source)
    _equal(dense_frozen, frozen)
    for name in ("query_polynomial_bias", "query_polynomial_kernel", "query_polynomial_tanh_bias"):
        np.testing.assert_array_equal(dense_generator[name][...], trained[name][...])
    amplitude = trained["query_polynomial_amplitude"][...]
    np.testing.assert_array_equal(dense_generator["query_polynomial_mixing"][...],
                                  amplitude[..., :, None] * np.eye(amplitude.shape[-1]))
    assert "query_polynomial_amplitude" not in dense_generator
    report = make_conversion_diagnostics(graph, dense_graph)(source, dense_source, tokens)
    assert bool(report["all_finite"])
    np.testing.assert_allclose(report["converted_nll"], report["original_nll"], rtol=1e-6, atol=1e-7)
    assert float(report["coefficient_relative_l2_error"]) < 1e-6
    assert float(report["logit_max_absolute_error"]) < 1e-5
    assert state_sha256(source) == source_hash


@pytest.mark.parametrize("readout", ["diagonal", "dense"])
def test_batch_gradient_accumulation_matches_one_full_batch_adam_update(fitting, readout):
    graph, source, tokens, options, _ = _converted(fitting, readout)
    # Trigger clipping to ensure that the mean gradient is clipped once, rather
    # than separately clipping each microbatch's contribution.
    options = dict(options, clip_norm=1e-4)
    capture, initialize, step, evaluate = make_generator_functions(graph, options)
    batch = jnp.concatenate([jnp.roll(tokens, shift, axis=1) for shift in range(4)])
    chunks = batch.reshape(2, 2, batch.shape[-1])
    initial = capture(source, batch[:2])[0]
    optimizer = initialize(initial)
    source_hash = state_sha256(source)
    direct = step(initial, optimizer, source, batch, 1e-3)
    accumulated = step(initial, optimizer, source, chunks, 1e-3)
    assert float(accumulated[3][0]) > options["clip_norm"]
    assert bool(accumulated[4])
    # Parameter updates, first/second moments, Adam count, gradient norm, and all
    # per-sequence losses must agree with differentiating the concatenated batch.
    for actual, expected in zip(jax.tree.leaves(accumulated), jax.tree.leaves(direct), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=8e-5, atol=3e-7)
    np.testing.assert_allclose(evaluate(initial, source, chunks), evaluate(initial, source, batch),
                               rtol=1e-5, atol=1e-6)
    assert state_sha256(source) == source_hash
    assert all(float(value) > 0 for value in accumulated[5].values())
    if readout == "dense":
        mixing = accumulated[0]["query_polynomial_mixing"][...]
        off_diagonal = 1 - jnp.eye(mixing.shape[-1])
        assert float(jnp.linalg.norm(mixing * off_diagonal)) > 0


@pytest.mark.parametrize("readout", ["diagonal", "dense"])
def test_batch_generator_acceptance_and_exact_resume(fitting, readout, tmp_path):
    graph, source, tokens, _, functions = _converted(fitting, readout)
    capture, initialize, _, _ = functions
    batch = jnp.concatenate([tokens, jnp.roll(tokens, 1, axis=1)])
    generator, parent_nll, _, error = capture(source, batch)
    report = qualify(tmp_path, functions, source, batch[:, None], generator, initialize(generator),
                     1e-3, graph=graph, generator=True,
                     parent_nll=np.asarray(parent_nll), capture_logit_error=float(error))
    assert report["checkpoint_resume_exact"]
    assert report["one_update_nll"] < report["separate_training_forward_nll"]
    assert set(report["generator_gradient_norms"]) == set(generator.keys())


def test_group_norms_include_all_layer_leaves():
    state = nnx.State({
        "first": {"query_polynomial_bias": nnx.Param(jnp.asarray([3.]))},
        "second": {"query_polynomial_bias": nnx.Param(jnp.asarray([4.]))},
    })
    assert float(_generator_group_norms(state)["query_polynomial_bias"]) == 5.


def query_key_model(backend="blocked"):
    """A trained-like native query-key parent: nonzero key amplitudes and per-head QK scales."""
    return ModelConfig(
        vocab_size=16, max_seq_len=13, num_layers=2, hidden_size=8, num_heads=2,
        num_kv_heads=2, qk_head_dim=2, value_head_dim=4, intermediate_size=12,
        attention_degree=4, attention_normalization=False, attention_coefficients="learned",
        attention_coefficient_sharing="per_head", attention_coefficient_conditioning="query_key",
        attention_coefficient_activation="tanh", attention_coefficient_scale="factorial",
        attention_coefficient_amplitude_scale="factorial",
        attention_coefficient_projection_initial_std=.2, attention_coefficient_amplitude_initial=.2,
        attention_backend=backend, attention_output_scaling="head_rms_norm",
        qk_norm=True, qk_norm_type="rms_norm", qk_norm_sharing="per_layer", qk_norm_head_scale=True,
        rms_norm_fp32=True, initialization="transformer_truncated_reference", remat=False,
    )


@pytest.fixture(scope="module")
def query_key_fitting():
    model = build_model(query_key_model(), 7)
    model.key_polynomial_amplitude[...] = jnp.full(model.key_polynomial_amplitude.shape, .3)
    model.qk_norm_head_log_scale[...] = jnp.full(model.qk_norm_head_log_scale.shape, .2)
    graph, source = nnx.split(model, nnx.Param)
    options = dict(clip_norm=1., beta1=.9, beta2=.95, epsilon=1e-8)
    return graph, source, jnp.arange(26).reshape(2, 13) % 16, make_generator_functions(graph, options)


def test_native_query_key_fit_trains_both_restored_generators_and_freezes_the_rest(query_key_fitting, tmp_path):
    graph, source, tokens, functions = query_key_fitting
    capture, initialize, step, _ = functions
    generator, frozen = split_generator_parameters(source)
    assert set(generator.keys()) == QUERY_KEY_PARAMETER_NAMES
    frozen_names = {str(path[-1]) for path, _ in nnx.to_flat_state(frozen)}
    assert {"qk_norm_head_log_scale", "qk_norm_q_gain", "qk_norm_k_gain", "output_norm_scale"} <= frozen_names
    initial = capture(source, tokens)[0]
    # The pretrained key generator is restored unchanged, not reset to identity factors.
    _equal(initial, generator)
    assert float(jnp.min(jnp.abs(initial["key_polynomial_amplitude"][...]))) > 0
    groups = step(initial, initialize(initial), source, tokens, 1e-3)[5]
    assert set(groups) == QUERY_KEY_PARAMETER_NAMES
    assert all(np.isfinite(value) and float(value) > 0 for value in groups.values())
    frozen_hash = state_sha256(frozen)
    report = qualify(tmp_path, functions, source, tokens, initial, initialize(initial), 1e-3,
                     graph=graph, generator=True)
    assert report["checkpoint_resume_exact"]
    assert set(report["trainable_parameter_names"]) == QUERY_KEY_PARAMETER_NAMES
    assert set(report["generator_gradient_norms_after_one_update"]) == QUERY_KEY_PARAMETER_NAMES
    assert all(value > 0 for value in report["generator_gradient_norms_after_one_update"].values())
    assert any(name.endswith("qk_norm_head_log_scale") for name in report["frozen_parameter_names"])
    assert report["frozen_parameter_sha256"] == frozen_hash == state_sha256(split_generator_parameters(source)[1])


def test_reference_backend_agreement_compares_fused_with_blocked(query_key_fitting, monkeypatch):
    from scaffolds.transformer.tests.test_query_key_polynomial import _interpret

    _interpret(monkeypatch)
    blocked, source, tokens, _ = query_key_fitting
    fused, _ = nnx.split(build_model(query_key_model("fused"), 7), nnx.Param)
    report = reference_backend_agreement(fused, blocked, source, tokens)
    assert report["all_logits_finite"]
    np.testing.assert_allclose(report["production_nll"], report["reference_nll"], rtol=0, atol=1e-5)
    # The same graph on both sides agrees exactly.
    assert reference_backend_agreement(blocked, blocked, source, tokens)["logit_max_absolute_error"] == 0


def test_capped_generator_fit_finalizes_its_best_generator(tmp_path):
    import json
    import orbax.checkpoint as ocp
    import yaml
    from scaffolds.transformer.configs.loader import config_to_dict
    from scaffolds.transformer.training.coefficient_fit import finalize_stopped

    config = Config(model=query_key_model())
    config.data.synthetic_vocab_size, config.data.sequence_length, config.data.batch_size = 16, 14, 2
    config.train.micro_batch_size = 2
    parent = tmp_path / "parent.yaml"
    parent.write_text(yaml.safe_dump(config_to_dict(config)))
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    recipe = {"model": config_to_dict(config)["model"],
              "oracle": {"parameterization": "query_coefficient_generator", "generator_readout": "diagonal",
                         "parent_config": str(parent), "parent_run_id": "rparent", "parent_checkpoint": "/parent"}}
    (run / "config.yaml").write_text(yaml.safe_dump(recipe))
    (run / "record.json").write_text(json.dumps({"run_id": "run", "status": "paused"}))
    generator = split_generator_parameters(nnx.split(build_model(config.model, 7), nnx.Param)[1])[0]
    best = jax.tree.map(lambda value: value + 1., generator)
    tx = optax.chain(optax.clip_by_global_norm(1.), optax.scale_by_adam())
    controller = {"step": 10, "best_step": 9, "best_loss": 1.5, "initial_loss": 2.7, "reduced_at": 4}
    save_state(run / "checkpoints", 10, [generator], [tx.init(generator)], [best], controller)
    (run / "metrics.jsonl").write_text("".join(json.dumps({"step": s, "nll": 2. - s / 10}) + "\n" for s in range(11)))
    outcome = finalize_stopped(run, "safety update budget exhausted")
    assert outcome["status"] == "complete" and outcome["best_step"] == 9 and outcome["step"] == 10
    assert outcome["frozen_parameter_scope"] == "backbone" and outcome["stopping"] == "safety update budget exhausted"
    assert [p.name for p in (run / "checkpoints").iterdir()] == ["9"]
    with ocp.StandardCheckpointer() as checkpointer:
        saved = checkpointer.restore(run / "checkpoints/9/best", target={"generator_parameters": [best]})
    _equal(saved["generator_parameters"][0], best)
    identity = json.loads((run / "checkpoints/9/coefficient_identity.json").read_text())
    assert identity["layout"] == "model_parameter_tree"
    assert json.loads((run / "record.json").read_text())["status"] == "complete"
