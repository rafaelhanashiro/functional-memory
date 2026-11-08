"""Coefficient release: frozen coefficient entries never move before their release update.

A schedule freezes learned static coefficients c_r, or the diagonal tanh query generator's
b_r, a_r, and column r of W, for optimizer updates with count k < s_r. Tiny CPU models;
acceptance selects this file for every recipe with a release schedule.
"""
from dataclasses import replace
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from scaffolds.transformer import acceptance
from scaffolds.transformer.attention.parameters import attention_parameter_specs
from scaffolds.transformer.configs import Config, ModelConfig, OptimConfig, load_config, save_config, validate_config
from scaffolds.transformer.configs.loader import config_dicts_equal, config_to_dict
from scaffolds.transformer.training import loop, tracking
from scaffolds.transformer.training.checkpointing import make_checkpoint_manager, restore_latest_checkpoint
from scaffolds.transformer.training.coefficient_release import (
    CoefficientReleaseOptimizer, coefficient_release, mask_frozen_gradients, release_counts,
)
from scaffolds.transformer.training.diagnostics import attention_identity
from scaffolds.transformer.training.distributed import make_data_parallel_functions, put_replicated, put_train_batches
from scaffolds.transformer.training.numerics import checked_update_loss, update_numerics
from scaffolds.transformer.training.optim import (
    attention_weight_decay_mask, make_learning_rate_schedule, make_optimizer,
)
from scaffolds.transformer.training.state import create_state

COEFFICIENTS = {
    "static": ("polynomial_coefficients",),
    "query": ("query_polynomial_bias", "query_polynomial_amplitude", "query_polynomial_kernel"),
}


def release_config(conditioning="query", schedule="all", start=3, end=None, *, degree=2, total_steps=8,
                   kernel_weight_decay=False, workdir=Path("unused")):
    """Unnormalized learned polynomial attention with QK RMSNorm, head scales, and head RMSNorm."""
    query = conditioning == "query"
    config = Config()
    config.model = ModelConfig(
        vocab_size=16, max_seq_len=4, num_layers=2, hidden_size=8, num_heads=2, num_kv_heads=2,
        qk_head_dim=2, value_head_dim=4, intermediate_size=12, attention_backend="blocked",
        attention_degree=degree, attention_normalization=False, attention_output_scaling="head_rms_norm",
        attention_coefficients="learned", attention_coefficient_sharing="per_head",
        attention_coefficient_scale="factorial",
        qk_norm=True, qk_norm_type="rms_norm", qk_norm_sharing="per_layer", qk_norm_head_scale=True,
        **({"attention_coefficient_conditioning": "query", "attention_coefficient_activation": "tanh",
            "attention_coefficient_amplitude_scale": "factorial",
            # A random W with a = 1 gives b, a, and W nonzero gradients from the first update.
            "attention_coefficient_projection_initial_std": 0.1,
            "attention_coefficient_kernel_weight_decay": kernel_weight_decay} if query else {}),
    )
    config.data.synthetic_vocab_size, config.data.sequence_length, config.data.batch_size = 16, 5, 4
    config.train.micro_batch_size = 2
    config.train.workdir = workdir
    config.train.eval_every, config.train.eval_batches, config.train.ckpt_every, config.train.log_every = 2, 1, 2, 1
    config.optim.total_steps, config.optim.warmup_steps = total_steps, 0
    config.optim.learning_rate, config.optim.min_learning_rate = 1e-2, 1e-3
    config.optim.grad_clip_norm = 0.01  # Every update clips.
    config.optim.coefficient_release_schedule = schedule
    config.optim.coefficient_release_step = start if schedule else None
    config.optim.coefficient_release_end_step = end
    validate_config(config)
    return config


def flat(tree) -> dict:
    return {path: np.asarray(value[...]) for path, value in nnx.to_flat_state(tree)}


def batch(step):
    return jnp.asarray((np.arange(20).reshape(4, 5) * (step + 1) + step) % 16)


def train(config, updates):
    """Parameters after each train step: ``history[k]`` follows the updates with counts 0..k-1."""
    state = create_state(config, jax.random.PRNGKey(0))
    history = [flat(state.params)]
    for step in range(1, updates + 1):
        state, loss, _ = loop.train_step(state, batch(step), jax.random.PRNGKey(step), check_all_gradients=True)
        assert np.isfinite(float(loss))
        history.append(flat(state.params))
    return state, history


def on_coefficients(tree, function, names):
    return jax.tree.map_with_path(lambda path, value: function(value) if path[0].key in names else value, tree)


def assert_bitwise_equal(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for first, second in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        first, second = np.asarray(first), np.asarray(second)
        assert first.dtype == second.dtype and first.shape == second.shape
        assert first.tobytes() == second.tobytes()


def test_release_counts_follow_the_schedule_formula():
    by_order = OptimConfig(total_steps=2000, coefficient_release_schedule="by_order",
                           coefficient_release_step=455, coefficient_release_end_step=1500)
    assert release_counts(by_order, 8) == (455, 585, 716, 846, 977, 1108, 1238, 1369, 1500)
    every = replace(by_order, coefficient_release_schedule="all", coefficient_release_end_step=None)
    assert release_counts(every, 8) == (455,) * 9
    assert release_counts(OptimConfig(), 8) is None
    assert coefficient_release(release_config("query", None)) is None
    release = coefficient_release(release_config("static", "by_order", 2, 6))
    assert (release.schedule, release.counts) == ("by_order", (2, 4, 6))
    assert [release.released_at(count) for count in range(8)] == [(), (), (0,), (), (1,), (), (2,), ()]
    assert [release.trainable_orders(count) for count in range(8)] == [0, 0, 1, 1, 2, 2, 3, 3]


@pytest.mark.parametrize("conditioning, kernel_weight_decay", [("static", False), ("query", False), ("query", True)])
def test_frozen_coefficients_stay_bitwise_then_move_after_release(conditioning, kernel_weight_decay):
    config = release_config(conditioning, "all", start=3, kernel_weight_decay=kernel_weight_decay)
    _, history = train(config, 5)
    initial, names = history[0], COEFFICIENTS[conditioning]
    assert initial[(names[0],)].shape[:2] == (2, 2)  # [layers, heads, ..., m + 1]
    for params in history[1:4]:  # After the updates with counts 0, 1, and 2.
        for name in names:
            assert params[(name,)].tobytes() == initial[(name,)].tobytes(), name
    for name in names:  # Update 3, logged as step 4, trains every order.
        for order in range(config.model.attention_degree + 1):
            assert not np.array_equal(history[4][(name,)][..., order], initial[(name,)][..., order]), (name, order)
    # Every other array, including QK gains, head score scales, and output norm gains, trains from update 0.
    others = [path for path in initial if path[0] not in names]
    assert {("qk_norm_q_gain",), ("qk_norm_k_gain",), ("qk_norm_head_log_scale",), ("output_norm_scale",)} <= set(others)
    for path in others:
        assert not np.array_equal(history[1][path], initial[path]), path


def test_weight_decay_cannot_move_a_frozen_query_projection():
    config = release_config("query", "all", start=2, kernel_weight_decay=True)
    state = create_state(config, jax.random.PRNGKey(0))
    assert isinstance(state.tx, CoefficientReleaseOptimizer)
    schedule, decay = make_learning_rate_schedule(config), config.optim.weight_decay
    kernel = ("query_polynomial_kernel",)
    zeros = jax.tree.map(jnp.zeros_like, state.params)
    # Without a schedule, decay alone moves W in the first update.
    plain = make_optimizer(release_config("query", None, kernel_weight_decay=True))
    plain_updates, _ = plain.update(zeros, state.opt_state, state.params)
    np.testing.assert_allclose(flat(plain_updates)[kernel], -float(schedule(0)) * decay * flat(state.params)[kernel],
                               rtol=1e-6)
    params, opt_state = state.params, state.opt_state
    for count in range(4):
        current = flat(params)
        updates, opt_state = state.tx.update(zeros, opt_state, params)
        values = flat(updates)
        rate = float(schedule(count))
        np.testing.assert_allclose(values[("lm_head", "kernel")], -rate * decay * current[("lm_head", "kernel")],
                                   rtol=1e-6)
        params = optax.apply_updates(params, updates)
        if count < 2:
            assert np.all(values[kernel] == 0) and np.all(np.signbit(values[kernel]))
            assert flat(params)[kernel].tobytes() == current[kernel].tobytes()
        else:
            np.testing.assert_allclose(values[kernel], -rate * decay * current[kernel], rtol=1e-6)


def test_frozen_gradients_are_excluded_from_the_clip_norm():
    config = release_config("query", "all", start=3, kernel_weight_decay=True)
    plain = make_optimizer(release_config("query", None, kernel_weight_decay=True))
    state = create_state(config, jax.random.PRNGKey(0))
    names = set(COEFFICIENTS["query"])
    _, grads = loop.loss_and_grads(state, batch(1), jax.random.PRNGKey(1))
    # Frozen gradients of 1,000 per entry would dominate a clip norm that counted them.
    loud = on_coefficients(grads, lambda value: jnp.full_like(value, 1e3), names)
    absent = on_coefficients(grads, jnp.zeros_like, names)
    released_updates, released_state = state.tx.update(loud, state.opt_state, state.params)
    absent_updates, absent_state = plain.update(absent, state.opt_state, state.params)
    # The rest of the model trains exactly as without coefficient gradients: the same clipping,
    # the same Adam moments (the coefficients' stay zero), and the same updates.
    assert_bitwise_equal(released_state, absent_state)
    released_values = flat(released_updates)
    for path, value in flat(absent_updates).items():
        if path[0] not in names:
            assert released_values[path].tobytes() == value.tobytes(), path
    # Counting the frozen gradients would have clipped every other gradient far harder.
    _, counted_state = plain.update(loud, state.opt_state, state.params)
    backbone = lambda tree: [value for path, value in flat(optax.tree_utils.tree_get(tree, "mu")).items()
                             if path[0] not in names]
    assert float(optax.tree.norm(backbone(counted_state))) < 0.01 * float(optax.tree.norm(backbone(released_state)))
    # The logged pre-clip norm is the masked gradient's, the one clipping sees.
    _, _, numerics = loop.train_step(state, batch(1), jax.random.PRNGKey(1), check_all_gradients=True)
    masked_norm, raw_norm = float(optax.tree.norm(absent)), float(optax.tree.norm(grads))
    assert raw_norm - masked_norm > 1e-3 * raw_norm  # The coefficient gradients are not negligible.
    np.testing.assert_allclose(float(numerics[1]), masked_norm, rtol=1e-5)


@pytest.mark.parametrize("conditioning", ["static", "query"])
def test_by_order_release_trains_each_order_from_its_listed_update(conditioning):
    config = release_config(conditioning, "by_order", start=2, end=8, degree=3, total_steps=10)
    counts = coefficient_release(config).counts
    assert counts == (2, 4, 6, 8)
    _, history = train(config, 9)
    initial = history[0]
    for name in COEFFICIENTS[conditioning]:
        first_moves = []
        for order, release in enumerate(counts):
            unchanged = [params[(name,)][..., order].tobytes() == initial[(name,)][..., order].tobytes()
                         for params in history]
            # history[k + 1] follows update k: orders are bitwise frozen through update s_r - 1.
            assert all(unchanged[:release + 1]), (name, order)
            first_moves.append(unchanged.index(False) - 1)
        assert tuple(first_moves) == counts, name


def test_masks_depend_only_on_the_update_count():
    config = release_config("query", "by_order", start=2, end=6, kernel_weight_decay=True)
    state = create_state(config, jax.random.PRNGKey(0))
    release, names = state.tx.coefficient_release, COEFFICIENTS["query"]

    def draw(tree, scale, seed):
        leaves, structure = jax.tree.flatten(tree)
        keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
        return jax.tree.unflatten(structure, [scale * jax.random.normal(key, leaf.shape, leaf.dtype)
                                              for key, leaf in zip(keys, leaves, strict=True)])

    for count in range(8):
        opt_state = optax.tree_utils.tree_set(state.opt_state, count=jnp.asarray(count, jnp.int32))
        trainable = np.asarray(release.counts) <= count
        np.testing.assert_array_equal(np.asarray(release.trainable(opt_state)), trainable)
        cases = ((state.params, jax.tree.map(jnp.zeros_like, state.params)),
                 (draw(state.params, 1.0, count), draw(state.params, 1e6, 100 + count)),
                 (draw(state.params, 1e-3, 200 + count), draw(state.params, 1e-6, 300 + count)))
        for params, grads in cases:
            masked = flat(mask_frozen_gradients(state.replace(opt_state=opt_state), grads))
            updates = flat(state.tx.update(grads, opt_state, params)[0])
            inner = flat(state.tx.inner.update(release.mask(grads, jnp.asarray(trainable), 0.0), opt_state, params)[0])
            raw = flat(grads)
            for name in names:
                path = (name,)
                assert not np.any(masked[path][..., ~trainable])
                assert masked[path][..., trainable].tobytes() == raw[path][..., trainable].tobytes()
                frozen = updates[path][..., ~trainable]
                assert np.all(frozen == 0) and np.all(np.signbit(frozen))
                assert updates[path][..., trainable].tobytes() == inner[path][..., trainable].tobytes()


@pytest.mark.parametrize("conditioning", ["static", "query"])
def test_a_missing_coefficient_array_fails_instead_of_training(conditioning):
    config = release_config(conditioning, "all", start=3)
    state = create_state(config, jax.random.PRNGKey(0))
    release, names = state.tx.coefficient_release, COEFFICIENTS[conditioning]
    assert release.parameters == names
    trainable = release.trainable(state.opt_state)
    values = flat(state.params)
    complete = {name: jnp.asarray(values[(name,)]) for name in names}
    assert not any(np.any(np.asarray(value)) for value in release.mask(complete, trainable, 0.0).values())
    with pytest.raises(ValueError, match="freezes"):
        release.mask({**{name: complete[name] for name in names[1:]}, "renamed": complete[names[0]]}, trainable, 0.0)


def test_released_entries_start_adam_from_zero_with_the_global_bias_correction():
    config = release_config("static", "all", start=3)
    state = create_state(config, jax.random.PRNGKey(0))
    schedule, (b1, b2) = make_learning_rate_schedule(config), (config.optim.beta1, config.optim.beta2)
    coefficients = ("polynomial_coefficients",)
    grads = jax.tree.map(lambda value: jnp.cos(jnp.arange(value.size, dtype=value.dtype).reshape(value.shape)) + 1.5,
                         state.params)
    params, opt_state = state.params, state.opt_state
    for count in range(4):
        updates, opt_state = state.tx.update(grads, opt_state, params)
        moments = [flat(optax.tree_utils.tree_get(opt_state, name))[coefficients] for name in ("mu", "nu")]
        if count < 3:
            assert not any(np.any(moment) for moment in moments)
        params = optax.apply_updates(params, updates)
    # Update 3 is the first for the coefficients: zero moments, global bias-correction count 4.
    gradient = flat(grads)[coefficients] * config.optim.grad_clip_norm / float(optax.tree.norm(grads))
    np.testing.assert_allclose(moments[0], (1 - b1) * gradient, rtol=1e-5)
    step = ((1 - b1) * gradient / (1 - b1 ** 4)) / (np.sqrt((1 - b2) * gradient ** 2 / (1 - b2 ** 4)) + 1e-8)
    np.testing.assert_allclose(flat(updates)[coefficients], -float(schedule(3)) * step, rtol=1e-5)
    # The first released step is about half the learning rate (0.56 here, (1 - b1) / sqrt(1 - b2) late).
    np.testing.assert_allclose(np.abs(step), 0.5601, rtol=1e-3)


def test_resume_across_a_release_matches_an_uninterrupted_run(tmp_path, monkeypatch, capsys):
    events = []
    monkeypatch.setattr(tracking, "log_coefficient_release",
                        lambda run, step, trainable_orders: events.append((step, trainable_orders)))
    full = release_config("query", "by_order", start=2, end=6, kernel_weight_decay=True, workdir=tmp_path / "full")
    loop.run_training(full)
    output = capsys.readouterr().out
    assert "coefficient_release_schedule=by_order release_updates=2,4,6 trainable_orders=0" in output
    lines = [line for line in output.splitlines() if line.startswith("coefficient_release update=")]
    assert lines == ["coefficient_release update=2 step=3 orders=0", "coefficient_release update=4 step=5 orders=1",
                     "coefficient_release update=6 step=7 orders=2"]
    assert events == [(3, 1), (5, 2), (7, 3)]
    resumed = replace(full, train=replace(full.train, workdir=tmp_path / "resumed"))
    events.clear()
    # Stop after update 3, so the first resumed update (count 4) releases order 1.
    interrupted = loop.run_training(resumed, stop_after_step=4, audit_checkpoint=True)
    manager = make_checkpoint_manager(resumed)
    try:
        paused, step = restore_latest_checkpoint(manager, create_state(resumed, jax.random.PRNGKey(0)))
    finally:
        manager.close()
    assert step == 4 and events == [(3, 1)]
    initial, paused = flat(create_state(resumed, jax.random.PRNGKey(0)).params), flat(paused.params)
    for name in COEFFICIENTS["query"]:
        assert not np.array_equal(paused[(name,)][..., 0], initial[(name,)][..., 0])
        assert paused[(name,)][..., 1:].tobytes() == initial[(name,)][..., 1:].tobytes()
    capsys.readouterr()
    finished = loop.run_training(resumed, audit_checkpoint=True)
    output = capsys.readouterr().out
    assert "trainable_orders=2" in output and events == [(3, 1), (5, 2), (7, 3)]
    assert [line for line in output.splitlines() if line.startswith("coefficient_release update=")] == lines[1:]
    first, second = interrupted["checkpoint_audit"], finished["checkpoint_audit"]
    assert first["final_state_sha256"] == first["checkpoint_state_sha256"] == second["start_state_sha256"]
    assert first["next_update"] == second["first_update"]
    states = []
    for current in (full, resumed):
        manager = make_checkpoint_manager(current)
        try:
            state, step = restore_latest_checkpoint(manager, create_state(current, jax.random.PRNGKey(0)))
        finally:
            manager.close()
        assert step == full.optim.total_steps
        states.append(state)
    assert_bitwise_equal(states[0].params, states[1].params)
    assert_bitwise_equal(states[0].opt_state, states[1].opt_state)


@nnx.jit
def train_step_without_release(state, tokens, rng):
    """The train step as it was before release schedules existed."""
    loss, grads = loop.loss_and_grads(state, tokens, rng)
    updated = state.apply_gradients(grads=grads)
    return (updated, checked_update_loss(state.params, grads, updated.params, loss, check_all_gradients=True),
            update_numerics(grads, updated.params, loss))


@pytest.mark.parametrize("conditioning", ["static", "query"])
def test_without_a_schedule_the_optimizer_and_train_step_are_unchanged(conditioning):
    config = release_config(conditioning, None)
    state = create_state(config, jax.random.PRNGKey(0))
    optimizer = make_optimizer(config)
    assert not hasattr(optimizer, "coefficient_release")
    excluded = frozenset(spec.name for spec in attention_parameter_specs(config.model) if not spec.weight_decay)
    unmodified = optax.chain(optax.clip_by_global_norm(config.optim.grad_clip_norm), optax.adamw(
        make_learning_rate_schedule(config), b1=config.optim.beta1, b2=config.optim.beta2,
        weight_decay=config.optim.weight_decay, mask=partial(attention_weight_decay_mask, excluded_names=excluded)))
    params = state.params
    expected_state, observed_state = unmodified.init(params), optimizer.init(params)
    assert_bitwise_equal(expected_state, observed_state)
    # A schedule adds no optimizer state either, so release runs keep this checkpoint layout.
    scheduled = make_optimizer(release_config(conditioning, "all", start=3))
    assert jax.tree.structure(scheduled.init(params)) == jax.tree.structure(observed_state)
    for step in range(3):
        grads = jax.tree.map(lambda value: jnp.sin(jnp.arange(value.size, dtype=value.dtype).reshape(value.shape)
                                                   + step), params)
        expected, expected_state = unmodified.update(grads, expected_state, params)
        observed, observed_state = optimizer.update(grads, observed_state, params)
        assert_bitwise_equal(expected, observed)
        assert_bitwise_equal(expected_state, observed_state)
        params = optax.apply_updates(params, observed)
    # The train step adds no operations: masking returns the gradient tree itself.
    _, grads = loop.loss_and_grads(state, batch(1), jax.random.PRNGKey(1))
    assert mask_frozen_gradients(state, grads) is grads
    assert_bitwise_equal(train_step_without_release(state, batch(1), jax.random.PRNGKey(1)),
                         loop.train_step(state, batch(1), jax.random.PRNGKey(1), check_all_gradients=True))


def test_identity_tags_and_recipes_record_the_schedule_only_when_set(tmp_path):
    plain = release_config("query", None)
    plain_identity = attention_identity(plain)
    assert not any("release" in key for key in plain_identity) and "s_r" not in plain_identity["attention_formula"]
    scheduled = release_config("query", "by_order", start=2, end=6)
    identity = attention_identity(scheduled)
    assert {key: identity[key] for key in identity if key.startswith("coefficient_release")} == {
        "coefficient_release_schedule": "by_order", "coefficient_release_step": 2,
        "coefficient_release_end_step": 6, "coefficient_release_counts": [2, 4, 6]}
    # The schedule only adds to the identity.
    assert {key: value for key, value in identity.items()
            if not key.startswith("coefficient_release") and key != "attention_formula"} == {
        key: value for key, value in plain_identity.items() if key != "attention_formula"}
    assert identity["attention_formula"] == plain_identity["attention_formula"] + (
        "; order r's stored coefficients (b_r, a_r, and column r of W) stay at their initial values for "
        "optimizer updates k < s_r (gradient zeroed before clipping, update zeroed after AdamW) and train for "
        "k >= s_r, the update logged as step k + 1, s_r = 2 + floor(r (6 - 2) / 2) = [2, 4, 6], lowest order first")
    static = attention_identity(release_config("static", "all", start=3))
    assert "coefficient_release_end_step" not in static and static["coefficient_release_counts"] == [3, 3, 3]
    assert static["attention_formula"].endswith("(c_r) stay at their initial values for optimizer updates k < s_r "
                                                "(gradient zeroed before clipping, update zeroed after AdamW) and "
                                                "train for k >= s_r, the update logged as step k + 1, "
                                                "s_r = 3 for every order")
    assert "coefficient_release_by_order" in tracking.run_tags(scheduled, {})
    assert "coefficient_release_all" in tracking.run_tags(release_config("static", "all", start=3), {})
    assert not any("release" in tag for tag in tracking.run_tags(plain, {}))
    path = tmp_path / "config.yaml"
    save_config(scheduled, path)
    loaded = load_config(path).optim
    assert (loaded.coefficient_release_schedule, loaded.coefficient_release_step,
            loaded.coefficient_release_end_step) == ("by_order", 2, 6)
    # A recorded run that predates the settings resumes as a run without a schedule.
    legacy = config_to_dict(plain)
    for key in ("coefficient_release_step", "coefficient_release_schedule", "coefficient_release_end_step"):
        legacy["optim"].pop(key)
    assert config_dicts_equal(legacy, config_to_dict(plain))
    assert not config_dicts_equal(legacy, config_to_dict(scheduled))


def _set(config, model=None, **optim):
    if model:
        config.model = replace(config.model, **model)
    for key, value in optim.items():
        setattr(config.optim, f"coefficient_release_{key}", value)
    return config


MECHANISM = "requires Taylor-initialized learned direct static coefficients or diagonal tanh query"


@pytest.mark.parametrize("conditioning, model, message", [
    ("static", {"attention_coefficients": "fixed", "attention_coefficient_scale": "none"}, MECHANISM),
    ("static", {"attention_coefficient_initialization": "linear"}, MECHANISM),
    ("static", {"attention_coefficient_orders": (0, 2)}, MECHANISM),
    ("static", {"attention_normalization": True, "attention_output_scaling": "none",
                "attention_coefficient_scale": "none", "attention_coefficient_parameterization": "rational"}, MECHANISM),
    ("static", {"attention_normalization": True, "attention_output_scaling": "none",
                "attention_coefficient_scale": "none",
                "attention_coefficient_parameterization": "sum_of_squares"}, MECHANISM),
    ("query", {"attention_coefficient_conditioning": "query_key"}, MECHANISM),
    ("query", {"attention_coefficient_readout": "dense"}, MECHANISM),
    ("query", {"attention_coefficient_activation": "affine_sigmoid"}, MECHANISM),
    ("query", {"attention_coefficient_activation": "sigmoid", "attention_coefficient_scale": "none"}, MECHANISM),
    ("query", {"attention_degree": 1, "attention_coefficient_activation": "identity",
               "attention_coefficient_scale": "none", "attention_coefficient_amplitude_scale": "none",
               "attention_coefficient_projection_initial_std": 0.0}, MECHANISM),
    ("query", {"attention_coefficient_tanh_bias": True}, MECHANISM),
    ("query", {"attention_coefficient_orders": (0, 2)}, MECHANISM),
])
def test_unsupported_coefficients_reject_a_schedule(conditioning, model, message):
    config = _set(release_config(conditioning, None), model, schedule="all", step=3)
    with pytest.raises(ValueError, match=message):
        validate_config(config)


@pytest.mark.parametrize("settings, message", [
    ({"schedule": "all", "step": 0}, "coefficient_release_step must be an integer with 0 <"),
    ({"schedule": "all", "step": -1}, "coefficient_release_step must be an integer with 0 <"),
    ({"schedule": "all", "step": 8}, "coefficient_release_step must be an integer with 0 <"),
    ({"schedule": "all", "step": 9}, "coefficient_release_step must be an integer with 0 <"),
    ({"schedule": "all", "step": 3.0}, "coefficient_release_step must be an integer with 0 <"),
    ({"schedule": "all", "step": True}, "coefficient_release_step must be an integer with 0 <"),
    ({"schedule": "all"}, "schedule requires optim.coefficient_release_step"),
    ({"step": 3}, "require optim.coefficient_release_schedule"),
    ({"end_step": 5}, "require optim.coefficient_release_schedule"),
    ({"step": 3, "end_step": 5}, "require optim.coefficient_release_schedule"),
    ({"schedule": "all", "step": 3, "end_step": 5}, "applies to the by_order schedule only"),
    ({"schedule": "by_order", "step": 3}, "by_order coefficient release requires optim.coefficient_release_end_step"),
    ({"schedule": "by_order", "step": 3, "end_step": 3}, "coefficient_release_end_step must be an integer with"),
    ({"schedule": "by_order", "step": 3, "end_step": 2}, "coefficient_release_end_step must be an integer with"),
    ({"schedule": "by_order", "step": 3, "end_step": 8}, "coefficient_release_end_step must be an integer with"),
    ({"schedule": "by_order", "step": 3, "end_step": 9}, "coefficient_release_end_step must be an integer with"),
    ({"schedule": "by_order", "step": 3, "end_step": 5.0}, "coefficient_release_end_step must be an integer with"),
    ({"schedule": "by_order", "step": 0, "end_step": 5}, "coefficient_release_step must be an integer with 0 <"),
    ({"schedule": "linear", "step": 3}, "Unknown optim.coefficient_release_schedule"),
])
@pytest.mark.parametrize("conditioning", ["static", "query"])
def test_invalid_release_steps_are_rejected(conditioning, settings, message):
    config = _set(release_config(conditioning, None), **settings)
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_a_continuation_rejects_a_schedule():
    config = release_config("static", None)
    config.optim.min_learning_rate = config.optim.learning_rate
    config.continuation.parent_config = "parent/config.yaml"
    config.continuation.parent_checkpoint = "parent/checkpoints/1"
    config.continuation.parent_config_sha256 = config.continuation.parent_checkpoint_sha256 = "0" * 64
    config.continuation.parent_step = 1
    validate_config(config)
    _set(config, schedule="all", step=3)
    with pytest.raises(ValueError, match="not supported for checkpoint continuations"):
        validate_config(config)


@pytest.mark.parametrize("conditioning, model", [
    ("static", {"attention_coefficient_sharing": "global"}),
    ("static", {"attention_normalization": True, "attention_output_scaling": "none",
                "attention_coefficient_scale": "none"}),
    ("query", {"attention_forget_gate": True}),
    ("query", {"attention_coefficient_amplitude_initial": 0.0}),
])
def test_supported_coefficients_accept_a_schedule_and_mask_every_coefficient(conditioning, model):
    config = _set(release_config(conditioning, None), model, schedule="by_order", step=2, end_step=6)
    validate_config(config)
    state = create_state(config, jax.random.PRNGKey(0))
    ones = jax.tree.map(jnp.ones_like, state.params)
    masked = flat(mask_frozen_gradients(state, ones))
    for path, value in masked.items():
        assert np.all(value == (0 if path[0] in COEFFICIENTS[conditioning] else 1)), path


def test_acceptance_selects_these_checks_for_every_recipe_with_a_schedule():
    spec = acceptance.AcceptanceSpec(entry_module="experiment", cpu_tests=())
    assert acceptance.ROOT / acceptance.COEFFICIENT_RELEASE_TESTS == Path(__file__).resolve()
    for config in (release_config("static", "all", 3), release_config("query", "by_order", 2, 6)):
        recipe = config_to_dict(config)
        assert acceptance.COEFFICIENT_RELEASE_TESTS in acceptance.test_paths(
            spec, recipe, resolved_backend="blocked", cpu=True)
        # CPU-only checks; the GPU stage would collect none of them.
        assert acceptance.COEFFICIENT_RELEASE_TESTS not in acceptance.test_paths(
            spec, recipe, resolved_backend="mosaic", cpu=False)
    for conditioning in ("static", "query"):
        assert acceptance.COEFFICIENT_RELEASE_TESTS not in acceptance.test_paths(
            spec, config_to_dict(release_config(conditioning, None)), resolved_backend="blocked", cpu=True)


def test_distributed_release_freezes_coefficients_and_reports_masked_gradients():
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip("Requires two host devices (XLA_FLAGS=--xla_force_host_platform_device_count=2)")
    config = release_config("query", "all", start=3, kernel_weight_decay=True)
    state = create_state(config, jax.random.PRNGKey(0))
    before = flat(state.params)
    functions = make_data_parallel_functions(state, devices[:2], check_all_gradients=True, collect_telemetry=True)
    replicated = put_replicated(state, functions.replicated_sharding)
    batches = put_train_batches(np.arange(40, dtype=np.int64).reshape(2, 4, 5) % 16, functions.train_batch_sharding)
    updated, loss, numerics, vector = functions.train_step(replicated, batches, jax.random.PRNGKey(3))
    assert np.isfinite(float(loss))
    after = flat(updated.params)
    for name in COEFFICIENTS["query"]:
        assert after[(name,)].tobytes() == before[(name,)].tobytes(), name
    assert not np.array_equal(after[("lm_head", "kernel")], before[("lm_head", "kernel")])
    statistics = dict(zip(functions.telemetry_keys, np.asarray(vector).tolist(), strict=True))
    assert statistics["update/attention_coefficients/gradient_norm"] == 0.0
    assert statistics["update/attention_coefficients/update_norm"] == 0.0
    group_norms = [value for key, value in statistics.items() if key.endswith("/gradient_norm")]
    np.testing.assert_allclose(np.sqrt(np.sum(np.square(group_norms))), float(np.asarray(numerics)[1]), rtol=1e-5)
