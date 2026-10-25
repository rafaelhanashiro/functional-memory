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

from collections import deque
import contextlib
import contextvars
import math
import re

import jax
import jax.numpy as jnp
import numpy as np

DATA_AXIS = "data"
REDUCTIONS = ("rms", "mean", "min", "max")
# A gradient spike is an update whose global gradient norm exceeds SPIKE_FACTOR times
# the median of the preceding SPIKE_HISTORY updates (once SPIKE_MIN_HISTORY exist).
SPIKE_FACTOR = 5.0
SPIKE_HISTORY = 100
SPIKE_MIN_HISTORY = 20
MAX_RECORDED_SPIKES = 20

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
    """Reduce across data-parallel devices and convert mean squares to RMS.

    Statistics are packed into one vector per collective (mean, min, max), so the
    number of cross-device reductions does not grow with the number of statistics.
    """
    collectives = {"mean": jax.lax.pmean, "min": jax.lax.pmin, "max": jax.lax.pmax}
    reduced = {}
    for collective, operation in collectives.items():
        keys = sorted(key for key in values
                      if (reductions[key] in ("rms", "mean") if collective == "mean" else reductions[key] == collective))
        if not keys:
            continue
        packed = operation(jnp.stack([values[key] for key in keys]), axis_name)
        for index, key in enumerate(keys):
            reduced[key] = jnp.sqrt(packed[index]) if reductions[key] == "rms" else packed[index]
    return reduced


_GROUP_RULES = (
    ("embedding", re.compile(r"token_embedding")),
    ("output_head", re.compile(r"lm_head")),
    ("attention_qkv", re.compile(r"/(q_proj|k_proj|v_proj)/")),
    ("attention_output_projection", re.compile(r"/o_proj/")),
    ("attention_erase_gate", re.compile(r"/lambda_proj/")),
    ("attention_write_gate", re.compile(r"/b_proj/")),
    ("attention_write_degree_0", re.compile(r"/b0_proj/")),
    ("attention_write_degree_2", re.compile(r"/b2_proj/")),
    ("attention_write_degree_3", re.compile(r"/b3_proj/")),
    ("attention_decay_gate", re.compile(r"/a_proj/|/(A_log|dt_bias)$")),
    ("attention_output_gate", re.compile(r"/g_proj/")),
    ("attention_short_convolution", re.compile(r"/(q_conv|k_conv|v_conv)/")),
    ("mlp", re.compile(r"feed_forward")),
    ("qk_norm_scale", re.compile(r"qk_norm_log_scale")),
    ("qk_norm_gains", re.compile(r"qk_norm_[qk]_gain")),
    ("attention_output_norm", re.compile(r"output_norm_scale|o_norm_scale")),
    ("attention_coefficients", re.compile(r"coefficient|polynomial|basis|denominator")),
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


def _number(value: float):
    """JSON-safe float: nonfinite values become strings, as in the diagnostics records."""
    value = float(value)
    return value if math.isfinite(value) else ("NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity"))


class TelemetryWindow:
    """Host-side record of every update's telemetry between validation events.

    ``summary`` gives each statistic's mean, minimum, and maximum over the window's
    updates. Gradient spikes keep their update's complete statistics (the first
    ``MAX_RECORDED_SPIKES`` per window; all are counted). The spike reference is the
    running median of preceding updates, which spans windows within one process.
    """

    def __init__(self):
        self.recent_gradient_norms: deque[float] = deque(maxlen=SPIKE_HISTORY)
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.total = self.minimum = self.maximum = None
        self.spikes: list[dict] = []
        self.spike_count = 0

    def add(self, step: int, keys: list[str], values: np.ndarray, gradient_norm: float | None) -> dict | None:
        """Accumulate one update; return its spike record when it is a gradient spike."""
        values = np.asarray(values, dtype=np.float64)
        if self.count == 0:
            self.total, self.minimum, self.maximum = values.copy(), values.copy(), values.copy()
        else:
            self.total += values
            np.fmin(self.minimum, values, out=self.minimum)
            np.fmax(self.maximum, values, out=self.maximum)
        self.count += 1
        spike = None
        if gradient_norm is not None and math.isfinite(gradient_norm):
            if len(self.recent_gradient_norms) >= SPIKE_MIN_HISTORY:
                median = float(np.median(self.recent_gradient_norms))
                if gradient_norm > SPIKE_FACTOR * median:
                    spike = {"step": step, "gradient_norm": gradient_norm, "running_median": median,
                             "statistics": dict(zip(keys, (_number(value) for value in values)))}
                    self.spike_count += 1
                    if len(self.spikes) < MAX_RECORDED_SPIKES:
                        self.spikes.append(spike)
            self.recent_gradient_norms.append(gradient_norm)
        return spike

    def summary(self, keys: list[str]) -> dict:
        if self.count == 0:
            return {}
        mean = self.total / self.count
        return {
            "train_telemetry_update_count": self.count,
            "train_telemetry": {key: {"mean": _number(mean[index]), "min": _number(self.minimum[index]),
                                      "max": _number(self.maximum[index])} for index, key in enumerate(keys)},
            "train_gradient_spike_count": self.spike_count,
            "train_gradient_spikes": self.spikes,
            "gradient_spike_rule": {"factor": SPIKE_FACTOR, "history": SPIKE_HISTORY,
                                    "minimum_history": SPIKE_MIN_HISTORY, "recorded_limit": MAX_RECORDED_SPIKES},
        }


def log_summary(values: dict[str, float]) -> dict[str, float]:
    """The few extremes printed on each training log line."""
    def extreme(suffix, function):
        matches = [value for key, value in values.items() if key.endswith(suffix)]
        return function(matches) if matches else None
    summary = {
        "attention_output_head_rms_min": extreme("/attention_output_head_rms_min", min),
        "head_rms_norm_gain_max": extreme("/head_rms_norm_gain_max", max),
        "residual_max_abs": extreme("/residual_max_abs", max),
    }
    return {key: value for key, value in summary.items() if value is not None}
