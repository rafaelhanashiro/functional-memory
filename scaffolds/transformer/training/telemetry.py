"""Opt-in training telemetry: cheap activation and per-group update statistics.

Activation statistics are scalar reductions recorded while a collector is active
during the training forward pass. Recording never changes model outputs: when no
collector is active (evaluation, diagnostics, and training without telemetry),
``record`` is a no-op at trace time and the compiled computation is unchanged.
Recording requires ``model.remat == False``; values recorded inside a rematerialized
block would escape its trace.

Each recorded scalar carries its reduction over microbatches and data-parallel
devices: ``rms`` (the recorded value is a mean square; means are taken before the
final square root), ``mean``, ``min``, or ``max``. Update statistics are computed
from the averaged gradient, the parameters before the update, and the parameters
after it, grouped by parameter role.
"""
from __future__ import annotations

import contextlib
import contextvars
import re

import jax
import jax.numpy as jnp

DATA_AXIS = "data"
REDUCTIONS = ("rms", "mean", "min", "max")

_COLLECTOR: contextvars.ContextVar = contextvars.ContextVar("transformer_telemetry", default=None)


class _Collector:
    def __init__(self):
        self.values: dict[str, jax.Array] = {}
        self.reductions: dict[str, str] = {}
        self.prefix = ""


@contextlib.contextmanager
def collect():
    """Collect recorded activation statistics; yields the collector."""
    collector = _Collector()
    token = _COLLECTOR.set(collector)
    try:
        yield collector
    finally:
        _COLLECTOR.reset(token)


def active() -> bool:
    return _COLLECTOR.get() is not None


@contextlib.contextmanager
def scope(name: str):
    collector = _COLLECTOR.get()
    if collector is None:
        yield
        return
    previous = collector.prefix
    collector.prefix = f"{previous}{name}/"
    try:
        yield
    finally:
        collector.prefix = previous


def record(name: str, value, reduction: str) -> None:
    collector = _COLLECTOR.get()
    if collector is None:
        return
    if reduction not in REDUCTIONS:
        raise ValueError(f"Unknown telemetry reduction: {reduction}")
    key = f"{collector.prefix}{name}"
    if key in collector.values:
        raise ValueError(f"Telemetry key recorded twice: {key}")
    collector.values[key] = jnp.asarray(value, dtype=jnp.float32)
    collector.reductions[key] = reduction


def record_residual(x: jax.Array) -> None:
    """RMS and maximum magnitude of a residual-stream tensor."""
    if not active():
        return
    value = x.astype(jnp.float32)
    record("residual_rms", jnp.mean(jnp.square(value)), "rms")
    record("residual_max_abs", jnp.max(jnp.abs(value)), "max")


def record_attention_output(y: jax.Array, *, head_rms_norm_eps: float | None = None) -> None:
    """Attention output ``[batch, positions, heads, value_dim]`` before output scaling.

    ``attention_output_head_rms_min`` is the smallest RMS of any single head at any
    single position. With head RMSNorm, ``head_rms_norm_gain_max`` is the largest
    factor ``1/sqrt(mean(y^2) + eps)`` that the normalization applies.
    """
    if not active():
        return
    head_mean_square = jnp.mean(jnp.square(y.astype(jnp.float32)), axis=-1)
    smallest = jnp.min(head_mean_square)
    record("attention_output_rms", jnp.mean(head_mean_square), "rms")
    record("attention_output_head_rms_min", jnp.sqrt(smallest), "min")
    if head_rms_norm_eps is not None:
        record("head_rms_norm_gain_max", jax.lax.rsqrt(smallest + head_rms_norm_eps), "max")


def reduce_microbatches(stacked: dict[str, jax.Array], reductions: dict[str, str]) -> dict[str, jax.Array]:
    """Reduce scan-stacked ``[microbatches]`` statistics within one device."""
    reduced = {}
    for key, values in stacked.items():
        kind = reductions[key]
        reduced[key] = (jnp.min(values) if kind == "min" else jnp.max(values) if kind == "max"
                        else jnp.mean(values))
    return reduced


def reduce_devices(values: dict[str, jax.Array], reductions: dict[str, str],
                   axis_name: str = DATA_AXIS) -> dict[str, jax.Array]:
    """Reduce across data-parallel devices and convert mean squares to RMS."""
    reduced = {}
    for key, value in values.items():
        kind = reductions[key]
        if kind == "min":
            reduced[key] = jax.lax.pmin(value, axis_name)
        elif kind == "max":
            reduced[key] = jax.lax.pmax(value, axis_name)
        else:
            mean = jax.lax.pmean(value, axis_name)
            reduced[key] = jnp.sqrt(mean) if kind == "rms" else mean
    return reduced


_GROUP_RULES = (
    ("embedding", re.compile(r"token_embedding")),
    ("output_head", re.compile(r"lm_head")),
    ("attention_qkv", re.compile(r"/(q_proj|k_proj|v_proj)/")),
    ("attention_output_projection", re.compile(r"/o_proj/")),
    ("mlp", re.compile(r"feed_forward")),
    ("qk_norm_scale", re.compile(r"qk_norm_log_scale")),
    ("attention_output_norm", re.compile(r"output_norm_scale")),
    ("attention_coefficients", re.compile(r"coefficient|polynomial|denominator")),
    ("norm_scales", re.compile(r"(attention_norm|ffn_norm|final_norm)/")),
)


def parameter_group(path) -> str:
    text = "/".join(str(getattr(entry, "key", getattr(entry, "idx", entry))) for entry in path)
    for group, pattern in _GROUP_RULES:
        if pattern.search(text):
            return group
    return "other"


def _group_squared_norms(tree) -> dict[str, jax.Array]:
    sums: dict[str, jax.Array] = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        group = parameter_group(path)
        value = jnp.sum(jnp.square(leaf.astype(jnp.float32)))
        sums[group] = sums[group] + value if group in sums else value
    return sums


def update_statistics(params, grads, updated_params) -> dict[str, jax.Array]:
    """Per-group gradient, parameter, and actual-update norms of one optimizer step."""
    gradient = _group_squared_norms(grads)
    parameter = _group_squared_norms(params)
    update = _group_squared_norms(jax.tree.map(lambda after, before: after - before, updated_params, params))
    statistics = {}
    for group in sorted(gradient):
        statistics[f"update/{group}/gradient_norm"] = jnp.sqrt(gradient[group])
        statistics[f"update/{group}/parameter_norm"] = jnp.sqrt(parameter[group])
        statistics[f"update/{group}/update_norm"] = jnp.sqrt(update[group])
    return statistics
