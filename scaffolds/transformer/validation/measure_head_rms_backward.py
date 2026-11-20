"""Measure, at a run's checkpoint, how far the FP32 head RMSNorm backward moves the gradient of the next-token NLL.

Head RMSNorm (attention/output_scaling.py) maps a head's output y (per token and head, its value channels) to
y / sqrt(mean(y^2) + eps). Its FP32 backward multiplies by rsqrt(m) / m = m^(-3/2), m = mean(y^2) + eps: a subnormal
number once m passes 1.9e25 (output RMS 4.4e12) and less than half the smallest subnormal past 1.3e30 (RMS 1.1e15).
The term it scales removes the radial component of the gradient, so an inexact or flushed m^(-3/2) leaves a gradient
along y that the exact Jacobian does not have.

For each run (its recipe, a checkpoint directory, and the commit it trained at) the gradient with respect to every
parameter is evaluated on the first held-out sequences of the run's own eval manifest, one sequence at a time, at the
run's precision (train.dtype) and with its recipe unchanged except ``model.attention_backend: blocked``, so every
kernel runs on any GPU. One compiled program evaluates the same model with five head RMSNorms, chosen at run time:

- ``production``: ``scale_attention_output`` exactly as committed at the run's source commit (read with ``git show``).
- ``stable``: the same function evaluated stably in FP32. For any constant c > 0,
  y / sqrt(mean(y^2) + eps) = (y / c) / sqrt(mean((y / c)^2) + eps / c^2) identically in y, so the gradient of the
  right-hand side with c held constant is the gradient of the left. c is the row's largest magnitude (at least one),
  which keeps mean((y / c)^2) between 1/channels and 1 and every factor of the backward in FP32's normal range.
- ``fp64_head``: the head RMSNorm and its VJP in NumPy FP64 on the host (a callback), everything else as the run:
  the reference that tells which of the two FP32 evaluations is right.
- ``stable_backward`` and ``fp64_backward``: production's forward, bit for bit, with the stable evaluation's VJP or
  the host FP64 VJP. A BF16 model's rounding changes with any last-bit change of an activation, so two evaluations
  whose forwards differ in the last FP32 bit differ by the model's rounding noise (about 1% of the gradient); with
  one forward, these two and ``production`` differ by the backward arithmetic of this operation alone.

``--fp64-model`` adds, once every run has these evaluations, the whole model in FP64 on the first sequence
(parameters upcast, every layer computing in FP64, the blocked attention's arithmetic included; the scaffold's
explicit FP32 steps, QK RMSNorm, RoPE tables and coefficient generators, keep their FP32 rounding): the gradient
without BF16 rounding or any FP32 range limit, and each evaluation's distance from it.

``--probe`` first measures the VJP of each production version and of the stable evaluation on synthetic FP32 rows of
given RMS against FP64, which gives the device's own threshold. The model and training code are only read: the script
replaces ``attention.multihead.scale_attention_output`` inside its own process. If the backend cannot run the host
callback, the measurement keeps its FP32 evaluations (production, stable, stable_backward) and says so.

    python -m scaffolds.transformer.validation.measure_head_rms_backward --runs runs.json --output result.json

``runs.json`` is a list of {"run_id", "config", "checkpoint", "source_commit", optional "label", "checkpoint_step",
"fp64_model"}. The result is rewritten after every run, so an interrupted job keeps what it finished;
``--deadline`` and ``--time-budget`` stop new work from starting late (measure_head_rms_backward.sbatch).
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
import math
import os
import re
import subprocess
import time
import types
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCALING_SOURCE = "scaffolds/transformer/attention/output_scaling.py"
# mean(y^2) past which FP32's m^(-3/2) is subnormal, and past which it is below the smallest subnormal.
SUBNORMAL_MEAN_SQUARE = float(np.finfo(np.float32).tiny) ** (-2.0 / 3.0)
VANISHING_MEAN_SQUARE = float(np.finfo(np.float32).smallest_subnormal) ** (-2.0 / 3.0)
VARIANTS = ("production", "stable", "fp64_head", "stable_backward", "fp64_backward")
HOST_VARIANTS = ("fp64_head", "fp64_backward")
# The gradient comparisons reported per run, each ||left - right|| / ||right||.
PAIRS = (("production", "stable"), ("production", "fp64_head"), ("stable", "fp64_head"),
         ("production", "stable_backward"), ("production", "fp64_backward"), ("stable_backward", "fp64_backward"))
GROUPS = (
    ("qk", "Q and K projections, QK gains and head scales", re.compile(r"/(q_proj|k_proj)/|/qk_norm_")),
    ("v", "V projections", re.compile(r"/v_proj/")),
    ("attention_output", "attention output projections", re.compile(r"/o_proj/")),
    ("head_rms_gain", "head RMSNorm gains", re.compile(r"/output_norm_scale")),
    ("kernel_coefficients", "kernel coefficient parameters",
     re.compile(r"coefficient|polynomial|denominator|spline|score_bias|score_mlp|forget_gate")),
)
GROUP_NAMES = (*(name for name, _, _ in GROUPS), "other", "all")
PROBE_RMS = (1e10, 1e11, 1e12, 3e12, 4.4e12, 6e12, 1e13, 3e13, 1e14, 2e14, 3e14, 4.5e14, 6e14, 9e14, 1.2e15, 2e15,
             1e16, 1e17, 1e18, 1e19, 3e19, 1e25)


def _number(value):
    """A JSON-safe float: nonfinite values become strings, as in the run records."""
    value = float(value)
    return value if math.isfinite(value) else ("NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity")


def _write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    os.replace(temporary, path)


def production_scaling(commit: str):
    """``scale_attention_output`` as committed at ``commit``, with its source's SHA256."""
    source = subprocess.run(["git", "show", f"{commit}:{SCALING_SOURCE}"], cwd=ROOT, capture_output=True,
                            check=True).stdout
    module = types.ModuleType(f"output_scaling_{commit[:12]}")
    exec(compile(source, f"{commit[:12]}:{SCALING_SOURCE}", "exec"), module.__dict__)
    return module.scale_attention_output, hashlib.sha256(source).hexdigest()


def stable_head_rms_norm(y, gain, eps):
    """Head RMSNorm in FP32 with each row divided by its largest magnitude c (at least one), a constant to
    differentiation, and eps by c^2: the same function of y, whose backward stays in FP32's normal range."""
    import jax
    import jax.numpy as jnp

    value = y.astype(jnp.float32)
    largest = jax.lax.stop_gradient(jnp.maximum(jnp.max(jnp.abs(value), axis=-1, keepdims=True), 1.0))
    scaled = value / largest
    # c^2 overflows past 1.8e19, where eps / c^2 is below FP32's range and zero is its nearest value.
    scaled = scaled * jax.lax.rsqrt(jnp.mean(jnp.square(scaled), axis=-1, keepdims=True) + eps / jnp.square(largest))
    return (scaled * gain.astype(jnp.float32)[None, None]).astype(y.dtype)


def plain_head_rms_norm(y, gain, eps):
    """Head RMSNorm as written, in y's own dtype: the FP64 model's."""
    import jax
    import jax.numpy as jnp

    return y * jax.lax.rsqrt(jnp.mean(jnp.square(y), axis=-1, keepdims=True) + eps) * gain.astype(y.dtype)[None, None]


def _host_forward(value, gain, *, eps):
    value, gain = np.asarray(value, np.float64), np.asarray(gain, np.float64)
    with np.errstate(all="ignore"):
        return (value / np.sqrt(np.mean(value * value, axis=-1, keepdims=True) + eps) * gain).astype(np.float32)


def _vjp_fp64(value, gain, cotangent, *, eps):
    """FP64 VJP of out = value * s * gain, s = (mean(value^2) + eps)^(-1/2), over [batch, tokens, heads, channels]:
    d/dvalue = s (c g) - value s^3 sum(c g value) / channels, d/dgain = sum over batch and tokens of c value s."""
    value, gain, cotangent = (np.asarray(array, np.float64) for array in (value, gain, cotangent))
    with np.errstate(all="ignore"):
        scale = 1.0 / np.sqrt(np.mean(value * value, axis=-1, keepdims=True) + eps)
        weighted = cotangent * gain
        radial = np.sum(weighted * value, axis=-1, keepdims=True) * scale ** 3 / value.shape[-1]
        return weighted * scale - value * radial, np.sum(cotangent * value * scale, axis=(0, 1))


def _host_vjp(value, gain, cotangent, *, eps):
    with np.errstate(all="ignore"):
        return tuple(array.astype(np.float32) for array in _vjp_fp64(value, gain, cotangent, eps=eps))


def host_fp64_head_rms_norm(y, gain, eps):
    """Head RMSNorm whose forward and VJP run in NumPy FP64 on the host; FP32 in and out, as production's casts."""
    import jax
    import jax.numpy as jnp

    @jax.custom_vjp
    def normalize(value, gain):
        return jax.pure_callback(functools.partial(_host_forward, eps=eps),
                                 jax.ShapeDtypeStruct(value.shape, jnp.float32), value, gain)

    def forward(value, gain):
        return normalize(value, gain), (value, gain)

    def backward(residual, cotangent):
        value, gain = residual
        return jax.pure_callback(functools.partial(_host_vjp, eps=eps),
                                 (jax.ShapeDtypeStruct(value.shape, jnp.float32),
                                  jax.ShapeDtypeStruct(gain.shape, jnp.float32)), value, gain, cotangent)

    normalize.defvjp(forward, backward)
    return normalize(y.astype(jnp.float32), gain.astype(jnp.float32)).astype(y.dtype)


def production_forward(production, y, gain, eps, backward: str):
    """Production's head RMSNorm forward, bit for bit, with another VJP: ``"stable"`` pulls the cotangent back
    through the stable evaluation at the same input, ``"fp64"`` through the host FP64 VJP. The activations, and so
    the loss, are the production evaluation's own; only the backward arithmetic of this one operation differs."""
    import jax
    import jax.numpy as jnp

    @jax.custom_vjp
    def normalize(y, gain):
        return production(y, "head_rms_norm", output_norm_scale=gain, eps=eps)

    def forward(y, gain):
        return normalize(y, gain), (y, gain)

    def pull_back(residual, cotangent):
        y, gain = residual
        if backward == "stable":
            return jax.vjp(lambda y, gain: stable_head_rms_norm(y, gain, eps), y, gain)[1](cotangent)
        value, scale = jax.pure_callback(
            functools.partial(_host_vjp, eps=eps),
            (jax.ShapeDtypeStruct(y.shape, jnp.float32), jax.ShapeDtypeStruct(gain.shape, jnp.float32)),
            y.astype(jnp.float32), gain.astype(jnp.float32), cotangent.astype(jnp.float32))
        return value.astype(y.dtype), scale.astype(gain.dtype)

    normalize.defvjp(forward, pull_back)
    return normalize(y, gain)


def row_statistics(y):
    """Counts over one layer's (token, head) rows of [batch, tokens, heads, channels], and their largest RMS.

    The mean square is the FP32 one production forms (FP64 in the FP64 model); the RMS is taken after dividing each
    row by its largest magnitude, so it stays finite where the FP32 mean square overflows."""
    import jax
    import jax.numpy as jnp

    value = jax.lax.stop_gradient(y.astype(jnp.promote_types(y.dtype, jnp.float32)))
    mean_square = jnp.mean(jnp.square(value), axis=-1)
    finite = jnp.all(jnp.isfinite(value), axis=-1)
    largest = jnp.max(jnp.abs(value), axis=-1)
    divisor = jnp.where(finite & (largest > 0), largest, 1.0)
    rms = jnp.where(finite, divisor * jnp.sqrt(jnp.mean(jnp.square(value / divisor[..., None]), axis=-1)), 0.0)
    subnormal = mean_square > SUBNORMAL_MEAN_SQUARE
    count = functools.partial(jnp.sum, dtype=jnp.int32)
    return {
        "rows": jnp.asarray(mean_square.size, jnp.int32),
        "rows_subnormal": count(subnormal),
        "rows_vanishing": count(mean_square > VANISHING_MEAN_SQUARE),
        "rows_mean_square_overflow": count(jnp.isinf(mean_square) & finite),
        "rows_nonfinite": count(~finite),
        "heads_with_subnormal_rows": count(jnp.any(subnormal, axis=(0, 1))),
        "largest_rms": jnp.max(rms),
    }


class HeadNorm:
    """Stands in for ``attention.multihead.scale_attention_output`` while a measurement traces the model.

    Inside ``tracing(variant)`` every head RMSNorm call records its rows' statistics and evaluates ``variant``: a
    name, or a traced index into ``variants`` that selects the evaluation at run time. Outside it, and for every
    other output scaling, the call goes to the production function unchanged."""

    def __init__(self, production, variants=VARIANTS):
        self.production = production
        self.variants = tuple(variants)
        self._active = None

    @contextlib.contextmanager
    def tracing(self, variant):
        rows = []
        self._active = (variant, rows)
        try:
            yield rows
        finally:
            self._active = None

    def __call__(self, y, scaling, *, output_norm_scale=None, eps=1e-6, **shifted):
        import jax

        if self._active is None or scaling != "head_rms_norm":
            return self.production(y, scaling, output_norm_scale=output_norm_scale, eps=eps, **shifted)
        if shifted:
            raise NotImplementedError("A row-shifted output already reaches head RMSNorm at unit scale; "
                                      "this measurement covers unshifted recipes")
        variant, rows = self._active
        rows.append(row_statistics(y))
        evaluations = {
            "production": lambda y, gain: self.production(y, scaling, output_norm_scale=gain, eps=eps),
            "stable": lambda y, gain: stable_head_rms_norm(y, gain, eps),
            "fp64_head": lambda y, gain: host_fp64_head_rms_norm(y, gain, eps),
            "stable_backward": lambda y, gain: production_forward(self.production, y, gain, eps, "stable"),
            "fp64_backward": lambda y, gain: production_forward(self.production, y, gain, eps, "fp64"),
            "fp64_model": lambda y, gain: plain_head_rms_norm(y, gain, eps),
        }
        if isinstance(variant, str):
            return evaluations[variant](y, output_norm_scale)
        return jax.lax.switch(variant, [evaluations[name] for name in self.variants], y, output_norm_scale)


def install(head_norm: HeadNorm) -> None:
    """Route the attention layers' output scaling through ``head_norm`` in this process."""
    from scaffolds.transformer.attention import multihead

    multihead.scale_attention_output = head_norm


def gradient_function(graphdef, head_norm: HeadNorm, *, variant=None, fp64=False):
    """Jitted (params, tokens[, index]) -> ((NLL, per-layer row statistics), gradient), the training loss.

    ``variant`` fixes the head RMSNorm at trace time; None takes a run-time index into ``head_norm.variants``."""
    import jax
    import optax
    from flax import nnx

    from scaffolds.transformer.training.loop import cross_entropy_loss
    from scaffolds.transformer.training.randomness import model_forward_rngs

    def loss(params, tokens, index=None):
        model = nnx.merge(graphdef, params)
        with head_norm.tracing(index if variant is None else variant) as rows:
            logits = model(tokens[:, :-1], deterministic=False, rngs=model_forward_rngs(jax.random.PRNGKey(0)))
        targets = tokens[:, 1:]
        # Training's loss casts the logits to FP32; the FP64 model keeps its own precision.
        nll = (optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean() if fp64
               else cross_entropy_loss(logits, targets))
        return nll, (nll, tuple(rows))

    def function(*arguments):
        (_, auxiliary), gradient = jax.value_and_grad(loss, has_aux=True)(*arguments)
        return auxiliary, gradient

    return jax.jit(function)


def _path_text(path) -> str:
    return "/" + "/".join(str(getattr(entry, "key", getattr(entry, "idx", getattr(entry, "name", entry))))
                          for entry in path) + "/"


def parameter_group(path) -> str:
    text = _path_text(path)
    for name, _, pattern in GROUPS:
        if pattern.search(text):
            return name
    return "other"


def leaf_sums(left, right):
    """Per leaf [scale, sum((a - b)^2), sum(a^2), sum(b^2), sum(a b)] of a = left / scale and b = right / scale,
    scale the leaf's largest magnitude in either tree, so no square leaves FP32's range."""
    import jax
    import jax.numpy as jnp

    rows = []
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        a, b = a.astype(jnp.float32), b.astype(jnp.float32)
        scale = jnp.maximum(jnp.maximum(jnp.max(jnp.abs(a)), jnp.max(jnp.abs(b))), jnp.finfo(jnp.float32).tiny)
        a, b = a / scale, b / scale
        rows.append(jnp.stack([scale, jnp.sum(jnp.square(a - b)), jnp.sum(jnp.square(a)), jnp.sum(jnp.square(b)),
                               jnp.sum(a * b)]))
    return jnp.stack(rows)


def group_metrics(groups: list[str], sizes: list[int], sums) -> dict:
    """Relative L2 difference ||left - right|| / ||right|| and cosine per parameter group, from ``leaf_sums``."""
    sums = np.asarray(sums, np.float64)
    squared = sums[:, :1] ** 2 * sums[:, 1:]
    metrics = {}
    for group in GROUP_NAMES:
        index = [leaf for leaf, name in enumerate(groups) if group in ("all", name)]
        if not index:
            metrics[group] = None
            continue
        difference, left, right, product = squared[index].sum(axis=0)
        with np.errstate(all="ignore"):
            metrics[group] = {
                "relative_l2": _number(np.sqrt(difference / right)),
                "cosine": _number(product / np.sqrt(left * right)),
                "norm_left": _number(np.sqrt(left)), "norm_right": _number(np.sqrt(right)),
                "parameters": int(sum(sizes[leaf] for leaf in index)),
            }
    return metrics


def layer_statistics(per_sequence: list[tuple]) -> list[dict]:
    """Sum the row counts over sequences and keep each layer's largest RMS."""
    layers = []
    for layer in range(len(per_sequence[0])):
        entries = [{key: np.asarray(value) for key, value in sequence[layer].items()} for sequence in per_sequence]
        rows = int(sum(entry["rows"] for entry in entries))
        total = {key: int(sum(entry[key] for entry in entries))
                 for key in ("rows_subnormal", "rows_vanishing", "rows_mean_square_overflow", "rows_nonfinite")}
        layers.append({
            "layer": layer, "rows": rows, **total,
            "fraction_subnormal": total["rows_subnormal"] / rows,
            "fraction_vanishing": total["rows_vanishing"] / rows,
            "fraction_mean_square_overflow": total["rows_mean_square_overflow"] / rows,
            "heads_with_subnormal_rows_max": int(max(entry["heads_with_subnormal_rows"] for entry in entries)),
            "largest_rms": _number(max(float(entry["largest_rms"]) for entry in entries)),
        })
    return layers


def restore_parameters(config, checkpoint: Path):
    """The checkpoint's parameters in the structure training builds, its graph definition, and its update count."""
    import jax

    from scaffolds.transformer.training.checkpointing import restore_state
    from scaffolds.transformer.training.state import create_state

    state = restore_state(checkpoint, create_state(config, jax.random.PRNGKey(config.train.seed)))
    return state.graphdef, state.params, int(state.step)


def heldout_sequences(config, count: int) -> np.ndarray:
    from scaffolds.transformer.data.loaders import make_eval_iterator

    return np.asarray(next(make_eval_iterator(replace(config.data, batch_size=count))), dtype=np.int32)


def fp64_model_gradient(graphdef, params, tokens, head_norm: HeadNorm):
    """Loss, row statistics and gradient (rounded to FP32 for comparison) of the model computing in FP64."""
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from scaffolds.transformer.attention.backends import reference
    from scaffolds.transformer.models.components.normalization import RMSNorm

    scan = reference.blocked_dot_product_attention
    with jax.enable_x64():
        model = nnx.merge(graphdef, params)
        for _, module in nnx.iter_graph(model):
            if isinstance(module, nnx.Module) and hasattr(module, "dtype"):
                module.dtype = jnp.float64
            if isinstance(module, RMSNorm):
                module.fp32 = False
        graphdef64, params64 = nnx.split(model, nnx.Param)
        params64 = jax.tree.map(lambda leaf: leaf.astype(jnp.float64), params64)
        # The scan's entry point fixes FP32 arithmetic; the strips already follow their operands' dtype.
        reference.blocked_dot_product_attention = functools.partial(reference._blocked_scan, compute=jnp.float64)
        try:
            (nll, rows), gradient = gradient_function(graphdef64, head_norm, variant="fp64_model", fp64=True)(
                params64, tokens)
            dtypes = sorted({str(leaf.dtype) for leaf in jax.tree.leaves(gradient)})
            gradient = jax.tree.map(lambda leaf: leaf.astype(jnp.float32), gradient)
            return float(nll), jax.device_get(rows), gradient, dtypes
        finally:
            reference.blocked_dot_product_attention = scan


def backward_probe(productions: dict, *, host_callback: bool, eps=1e-6, rows=2048, channels=64, heads=2,
                   seed=0) -> dict:
    """VJP of each head RMSNorm on synthetic FP32 rows of a given RMS against FP64 on the host.

    ``relative_l2`` is ||vjp - reference|| / ||reference||. ``radial_fraction`` is sum|y . vjp| / sum|y . g s| over
    rows, g the gain-weighted cotangent and s = 1/sqrt(mean(y^2) + eps): zero for the exact Jacobian, whose result
    is orthogonal to y, and one when the term that removes the radial component is lost entirely."""
    import jax
    import jax.numpy as jnp

    generator = np.random.default_rng(seed)
    direction = generator.standard_normal((1, rows, heads, channels))
    direction /= np.sqrt(np.mean(direction ** 2, axis=-1, keepdims=True))
    cotangent = (1e-4 * generator.standard_normal(direction.shape)).astype(np.float32)
    gain = (1.0 + 0.1 * generator.standard_normal((heads, channels))).astype(np.float32)
    evaluations = {name: (lambda y, gain, function=function: function(y, "head_rms_norm", output_norm_scale=gain,
                                                                      eps=eps))
                   for name, function in productions.items()}
    evaluations["stable"] = lambda y, gain: stable_head_rms_norm(y, gain, eps)
    if host_callback:
        evaluations["fp64_head"] = lambda y, gain: host_fp64_head_rms_norm(y, gain, eps)
    compiled = {name: jax.jit(lambda y, gain, cotangent, function=function:
                              jax.vjp(lambda value: function(value, gain), y)[1](cotangent)[0])
                for name, function in evaluations.items()}
    results = []
    for rms in PROBE_RMS:
        y = (direction * rms).astype(np.float32)
        expected, _ = _vjp_fp64(y, gain, cotangent, eps=eps)
        rounded, _ = _host_vjp(y, gain, cotangent, eps=eps)
        y64 = y.astype(np.float64)
        scale = 1.0 / np.sqrt(np.mean(y64 * y64, axis=-1, keepdims=True) + eps)
        radial_lost = np.sum(np.abs(np.sum(y64 * (cotangent.astype(np.float64) * gain) * scale, axis=-1)))
        entry = {"rms": rms, "mean_square": _number(float(np.mean(y64 * y64, axis=-1).max()))}
        for name, function in compiled.items():
            actual = np.asarray(function(jnp.asarray(y), jnp.asarray(gain), jnp.asarray(cotangent)), np.float64)
            with np.errstate(all="ignore"):
                entry[name] = {
                    "relative_l2": _number(np.linalg.norm(actual - expected) / np.linalg.norm(expected)),
                    "radial_fraction": _number(np.sum(np.abs(np.sum(y64 * actual, axis=-1))) / radial_lost),
                    "zero_rows": int(np.sum(~np.any(actual != 0, axis=-1))),
                }
        entry["fp64_rounded_to_fp32_relative_l2"] = _number(
            np.linalg.norm(rounded.astype(np.float64) - expected) / np.linalg.norm(expected))
        results.append(entry)
    return {"rows": rows * heads, "channels": channels, "eps": eps, "cotangent_scale": 1e-4, "dtype": "float32",
            "evaluations": list(compiled), "scales": results}


def host_callback_problem() -> str | None:
    """None when a jitted program on this backend can differentiate through the host FP64 head RMSNorm."""
    import jax
    import jax.numpy as jnp

    try:
        y = jnp.linspace(-2.0, 3.0, 2 * 3 * 2 * 8, dtype=jnp.float32).reshape(2, 3, 2, 8)
        gain = jnp.ones((2, 8), jnp.float32)

        weights = jnp.cos(jnp.arange(y.size, dtype=jnp.float32)).reshape(y.shape)

        def total(y, index):
            return jnp.sum(weights * jax.lax.switch(index, [lambda v: stable_head_rms_norm(v, gain, 1e-6),
                                                            lambda v: host_fp64_head_rms_norm(v, gain, 1e-6)], y))

        gradient = jax.jit(jax.grad(total))
        difference = float(jnp.max(jnp.abs(gradient(y, jnp.asarray(0)) - gradient(y, jnp.asarray(1)))))
        return None if difference < 1e-5 else f"host FP64 gradient differs from the stable one by {difference}"
    except Exception as error:  # the measurement then keeps its FP32 evaluations
        return f"{type(error).__name__}: {error}"[:1000]


def measure_run(spec: dict, *, sequences: int, variants: tuple, direct_check: bool, keep_state: bool, log):
    """One run's result and, with ``keep_state``, what the FP64 model pass needs later (host arrays)."""
    import jax
    import jax.numpy as jnp

    from scaffolds.transformer.configs import load_config
    from scaffolds.transformer.configs.base import validate_config

    started = time.time()
    config = load_config(spec["config"])
    trained_backend = config.model.attention_backend
    config.model.attention_backend = "blocked"
    config.model.remat = False        # the row statistics leave each layer's trace; rematerialization changes no value
    config.model.__post_init__()
    validate_config(config)
    production, source_sha256 = production_scaling(spec["source_commit"])
    head_norm = HeadNorm(production, variants)
    install(head_norm)
    graphdef, params, step = restore_parameters(config, Path(spec["checkpoint"]))
    tokens = heldout_sequences(config, sequences)
    paths = [path for path, _ in jax.tree_util.tree_flatten_with_path(params)[0]]
    groups = [parameter_group(path) for path in paths]
    sizes = [int(leaf.size) for leaf in jax.tree.leaves(params)]
    result = {
        "run_id": spec["run_id"], "label": spec.get("label"), "checkpoint": str(spec["checkpoint"]),
        "checkpoint_step": step, "source_commit": spec["source_commit"],
        "production_scaling_sha256": source_sha256, "variants": list(variants),
        "recipe": {"trained_backend": trained_backend, "measured_backend": "blocked",
                   "blocked_implementation": config.model.attention_blocked_implementation,
                   "dtype": config.train.dtype, "num_layers": config.model.num_layers,
                   "hidden_size": config.model.hidden_size, "num_heads": config.model.num_heads,
                   "qk_head_dim": config.model.qk_head_dim, "value_head_dim": config.model.value_head_dim,
                   "attention_degree": config.model.attention_degree,
                   "rms_norm_eps": config.model.rms_norm_eps},
        "sequences": int(tokens.shape[0]), "tokens_per_sequence": int(tokens.shape[1] - 1),
        "eval_manifest": config.data.eval_manifest,
        "sequence_sha256": [hashlib.sha256(row.astype("<i4").tobytes()).hexdigest() for row in tokens],
        "parameters": {group: int(sum(size for size, name in zip(sizes, groups) if group in ("all", name)))
                       for group in GROUP_NAMES},
    }
    if spec.get("checkpoint_step") is not None and int(spec["checkpoint_step"]) != step:
        raise ValueError(f"{spec['run_id']}: checkpoint holds update {step}, expected {spec['checkpoint_step']}")
    log(f"{spec['run_id']}: restored update {step} in {time.time() - started:.0f} s; "
        f"{result['parameters']['all']:,} parameters, {len(paths)} arrays")

    evaluate = gradient_function(graphdef, head_norm)
    compare = jax.jit(leaf_sums)

    def device(tree):
        return jax.tree.map(jnp.asarray, tree)

    def summary(left, right):
        metrics = group_metrics(groups, sizes, compare(left, right))
        return {group: metrics[group] and {key: metrics[group][key] for key in ("relative_l2", "cosine")}
                for group in GROUP_NAMES}

    # Gradients accumulate on the host, so the device holds two at a time whatever the number of evaluations.
    totals = {name: None for name in variants}
    losses = {name: [] for name in variants}
    statistics = {name: [] for name in variants}
    per_sequence = []
    first = {}
    for index in range(tokens.shape[0]):
        batch = jnp.asarray(tokens[index:index + 1])
        entry = {"sequence": index}
        reference = None
        for number, name in enumerate(variants):
            clock = time.time()
            (nll, rows), gradient = evaluate(params, batch, jnp.asarray(number, jnp.int32))
            losses[name].append(float(nll))
            statistics[name].append(jax.device_get(rows))
            if name == "production":
                reference = gradient
            else:
                entry[f"production_vs_{name}"] = summary(reference, gradient)
            if index == 0:
                first[name] = jax.tree.map(np.array, gradient)
                totals[name] = jax.tree.map(np.array, gradient)
                log(f"{spec['run_id']}: {name} on sequence 0 in {time.time() - clock:.0f} s, NLL {float(nll):.6f}")
            else:
                jax.tree.map(lambda total, leaf: np.add(total, np.asarray(leaf), out=total), totals[name], gradient)
            del gradient
        del reference
        entry["loss"] = {name: _number(losses[name][-1]) for name in variants}
        entry["largest_rms"] = _number(max(float(layer["largest_rms"]) for layer in statistics["production"][-1]))
        per_sequence.append(entry)
    count = tokens.shape[0]
    for total in totals.values():
        jax.tree.map(lambda leaf: np.divide(leaf, count, out=leaf), total)
    result["loss"] = {name: _number(float(np.mean(losses[name]))) for name in variants}
    result["layers"] = layer_statistics(statistics["production"])
    result["layers_stable"] = [{key: layer[key] for key in ("layer", "fraction_subnormal", "fraction_vanishing",
                                                            "fraction_mean_square_overflow", "largest_rms")}
                               for layer in layer_statistics(statistics["stable"])]
    result["gradients"] = {
        f"{left}_vs_{right}": group_metrics(groups, sizes, compare(device(totals[left]), device(totals[right])))
        for left, right in PAIRS if left in variants and right in variants}
    result["per_sequence"] = per_sequence
    result["nonfinite_gradient"] = {name: bool(any(not np.isfinite(leaf).all() for leaf in jax.tree.leaves(total)))
                                    for name, total in totals.items()}
    result["seconds"] = round(time.time() - started, 1)
    log(f"{spec['run_id']}: {len(variants)} evaluations of {count} sequences in {result['seconds']:.0f} s; "
        f"largest RMS {max(float(layer['largest_rms']) for layer in result['layers']):.3g}; production against stable "
        f"Q/K {result['gradients']['production_vs_stable']['qk']['relative_l2']}, "
        f"all {result['gradients']['production_vs_stable']['all']['relative_l2']}"
        + (f"; against stable_backward Q/K {result['gradients']['production_vs_stable_backward']['qk']['relative_l2']}"
           f", all {result['gradients']['production_vs_stable_backward']['all']['relative_l2']}"
           if "production_vs_stable_backward" in result["gradients"] else ""))
    del totals

    if direct_check:
        # The run-time selection must leave the production evaluation as it is without it: the same model traced
        # with the production function alone.
        clock = time.time()
        (nll, _), gradient = gradient_function(graphdef, head_norm, variant="production")(
            params, jnp.asarray(tokens[:1]))
        selected = device(first["production"])
        largest = max(float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))
                      for a, b in zip(jax.tree.leaves(gradient), jax.tree.leaves(selected), strict=True))
        result["direct_check"] = {
            "loss_equal": float(nll) == losses["production"][0],
            "gradient_bitwise_equal": bool(all(bool(jnp.array_equal(a, b)) for a, b in zip(
                jax.tree.leaves(gradient), jax.tree.leaves(selected), strict=True))),
            "largest_absolute_difference": _number(largest),
            "all": group_metrics(groups, sizes, compare(gradient, selected))["all"],
            "seconds": round(time.time() - clock, 1)}
        log(f"{spec['run_id']}: direct check {result['direct_check']}")
        del gradient, selected
    result["seconds_total"] = round(time.time() - started, 1)
    state = None
    if keep_state:
        state = {"graphdef": graphdef, "params": jax.device_get(params), "tokens": tokens[:1], "groups": groups,
                 "sizes": sizes, "first": first, "losses": {name: losses[name][0] for name in variants},
                 "production": production}
    return result, state


def measure_fp64_model(result: dict, state: dict, *, log) -> None:
    """Add the FP64 model's first-sequence gradient, and each evaluation's distance from it, to ``result``."""
    import jax
    import jax.numpy as jnp

    clock = time.time()
    head_norm = HeadNorm(state["production"])
    install(head_norm)
    groups, sizes = state["groups"], state["sizes"]
    compare = jax.jit(leaf_sums)
    try:
        params = jax.tree.map(jnp.asarray, state["params"])
        nll, rows, gradient, dtypes = fp64_model_gradient(state["graphdef"], params, jnp.asarray(state["tokens"]),
                                                          head_norm)
        del params
        result["fp64_model"] = {
            "sequence": 0, "loss": _number(nll), "gradient_dtypes": dtypes,
            "loss_by_evaluation": {name: _number(value) for name, value in state["losses"].items()},
            "largest_rms": _number(max(float(layer["largest_rms"]) for layer in rows)),
            "layers": [{key: layer[key] for key in ("layer", "fraction_subnormal", "fraction_vanishing",
                                                    "largest_rms")} for layer in layer_statistics([rows])],
            "nonfinite_gradient": bool(any(not bool(jnp.all(jnp.isfinite(leaf)))
                                           for leaf in jax.tree.leaves(gradient))),
        }
        for name, other in state["first"].items():
            result["fp64_model"][f"{name}_vs_fp64_model"] = group_metrics(
                groups, sizes, compare(jax.tree.map(jnp.asarray, other), gradient))
        result["fp64_model"]["seconds"] = round(time.time() - clock, 1)
        log(f"{result['run_id']}: FP64 model in {time.time() - clock:.0f} s, NLL {nll:.6f}; against it Q/K "
            f"production {result['fp64_model']['production_vs_fp64_model']['qk']['relative_l2']}, stable "
            f"{result['fp64_model']['stable_vs_fp64_model']['qk']['relative_l2']}")
    except Exception as error:  # a recipe the FP64 path cannot run is reported, not fatal
        result["fp64_model"] = {"error": f"{type(error).__name__}: {error}"[:2000]}
        log(f"{result['run_id']}: FP64 model failed: {result['fp64_model']['error'][:300]}")


def environment() -> dict:
    import jax

    def command(*arguments):
        try:
            return subprocess.run(arguments, capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    gpu = command("nvidia-smi", "--query-gpu=name,uuid,pci.bus_id,driver_version,compute_cap",
                  "--format=csv,noheader")
    # Of the job's GPUs, the one this process computes on, when CUDA_VISIBLE_DEVICES names it by UUID.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    lines = [line for line in (gpu.splitlines() if gpu else []) if not visible.startswith("GPU-") or visible in line]
    return {
        "backend": jax.default_backend(), "devices": [str(device) for device in jax.devices()],
        "gpu": lines or None, "jax": jax.__version__, "numpy": np.__version__,
        "host": os.uname().nodename, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_environment": os.environ.get("TRANSFORMER_CUDA_ENVIRONMENT"),
        "xla_flags": os.environ.get("XLA_FLAGS", ""), "source_commit": command("git", "rev-parse", "HEAD"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True, help="JSON list of runs (module docstring)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequences", type=int, default=4, help="held-out sequences, from the manifest's first")
    parser.add_argument("--probe", action="store_true", help="measure the synthetic VJP threshold first")
    parser.add_argument("--direct-check", type=int, default=1,
                        help="number of leading runs whose production evaluation is also traced without selection")
    parser.add_argument("--fp64-model", action="store_true",
                        help="after every run's evaluations, evaluate the whole model in FP64 on the first "
                             "sequence of each run (those whose entry does not set fp64_model false), in order")
    parser.add_argument("--time-budget", type=float, default=None,
                        help="seconds from start after which no further run or FP64 model begins")
    parser.add_argument("--deadline", type=float, default=None,
                        help="UNIX time by which a run or FP64 model must be expected to end in order to begin")
    arguments = parser.parse_args()

    import jax

    start = time.time()

    def log(message):
        print(f"[{time.time() - start:7.0f} s] {message}", flush=True)

    limits = [limit for limit in (arguments.deadline, arguments.time_budget and start + arguments.time_budget)
              if limit is not None]

    def out_of_time(needed):
        """Whether work expected to take ``needed`` seconds (the longest so far) would pass the earliest limit."""
        return bool(limits) and time.time() + needed > min(limits)

    specs = json.loads(arguments.runs.read_text())
    problem = host_callback_problem()
    variants = tuple(name for name in VARIANTS if problem is None or name not in HOST_VARIANTS)
    productions, sources = {}, {}
    for spec in specs:
        function, digest = production_scaling(spec["source_commit"])
        productions.setdefault(f"production_{digest[:12]}", function)
        sources.setdefault(f"production_{digest[:12]}", {"sha256": digest, "source_commits": []})
        if spec["source_commit"] not in sources[f"production_{digest[:12]}"]["source_commits"]:
            sources[f"production_{digest[:12]}"]["source_commits"].append(spec["source_commit"])
    payload = {
        "format": "head-rms-norm-backward-measurement", "schema_version": 1, "environment": environment(),
        "thresholds": {"mean_square_subnormal": SUBNORMAL_MEAN_SQUARE, "mean_square_vanishing": VANISHING_MEAN_SQUARE,
                       "rms_subnormal": math.sqrt(SUBNORMAL_MEAN_SQUARE), "rms_vanishing": math.sqrt(VANISHING_MEAN_SQUARE)},
        "variants": list(variants), "host_callback_problem": problem, "production_scalings": sources,
        "groups": {name: description for name, description, _ in GROUPS}, "sequences": arguments.sequences,
        "runs": [],
    }
    log(f"environment {payload['environment']}; host FP64 head RMSNorm: {problem or 'available'}")
    if arguments.probe:
        payload["backward_probe"] = backward_probe(productions, host_callback=problem is None)
        for entry in payload["backward_probe"]["scales"]:
            log("probe RMS %.3g: " % entry["rms"] + ", ".join(
                f"{name} error {entry[name]['relative_l2']} radial {entry[name]['radial_fraction']}"
                for name in payload["backward_probe"]["evaluations"]))
        _write_json(arguments.output, payload)
    longest, states = 0.0, {}
    for position, spec in enumerate(specs):
        if out_of_time(longest):
            payload["runs"].append({"run_id": spec["run_id"], "label": spec.get("label"),
                                    "skipped": f"time budget: {time.time() - start:.0f} s elapsed, a run took up to "
                                               f"{longest:.0f} s"})
            log(f"{spec['run_id']}: skipped for the time budget")
            _write_json(arguments.output, payload)
            continue
        clock = time.time()
        try:
            result, state = measure_run(
                spec, sequences=arguments.sequences, variants=variants, direct_check=position < arguments.direct_check,
                keep_state=bool(arguments.fp64_model and spec.get("fp64_model", True)), log=log)
            payload["runs"].append(result)
            if state is not None:
                states[len(payload["runs"]) - 1] = state
        except Exception as error:  # an unavailable checkpoint or a failing recipe is reported; the others go on
            payload["runs"].append({"run_id": spec["run_id"], "label": spec.get("label"),
                                    "error": f"{type(error).__name__}: {error}"[:4000]})
            log(f"{spec['run_id']}: failed: {type(error).__name__}: {str(error)[:500]}")
        longest = max(longest, time.time() - clock)
        _write_json(arguments.output, payload)
        jax.clear_caches()
    slowest = None      # of the FP64 models so far; a run's time stands in before the first
    for position, state in states.items():
        result = payload["runs"][position]
        if out_of_time(longest if slowest is None else slowest):
            result["fp64_model"] = {"skipped": f"time budget: {time.time() - start:.0f} s elapsed"}
            log(f"{result['run_id']}: FP64 model skipped for the time budget")
            continue
        clock = time.time()
        measure_fp64_model(result, state, log=log)
        states[position] = None
        slowest = max(slowest or 0.0, time.time() - clock)
        _write_json(arguments.output, payload)
        jax.clear_caches()
    payload["seconds"] = round(time.time() - start, 1)
    _write_json(arguments.output, payload)
    log(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
