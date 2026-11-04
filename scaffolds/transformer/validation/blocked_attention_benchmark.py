"""Blocked (plain JAX/XLA) causal softmax attention at 2,048 tokens: candidates and benchmarks.

Every candidate takes q, k, v [B, T, H, D] (BF16 in production) and returns [B, T, H, D] in
V's dtype. In the blocked candidates, scores and softmax arithmetic are FP32 (FP64 for FP64
inputs), and a row whose causal scores include any nonfinite value (+-inf or NaN) returns
NaN, as the production blocked path does for softmax; nonfinite values and cotangents
otherwise propagate through the products. The references (cuDNN, the dense textbook
path) only propagate: a -inf score is a zero weight there. ``weights`` chooses the precision of the softmax weights entering the products
(P x V forward; P and dS backward): "float32" keeps them FP32 with V/dO/Q/K upcast, as
production does; "input" rounds them to the input dtype (BF16) with FP32 accumulation, as
cuDNN and FlashAttention do.

Stages (run from the repository root with PYTHONPATH=.):
  oracle-check  CPU: NumPy FP64 oracle against float64 JAX autodiff at small shapes.
  kernel        One GPU: accuracy at [2,2048,4,64] against the oracle, nonfinite-input
                behaviour, then timing and compiled memory at [16,2048,14,64].
  model         One GPU: the Transformer++ softmax recipe, one forward+backward micro-step
                per (variant, micro-batch), each in a fresh process for its allocator peak.

Timings are medians of compiled calls after warmup, synchronized after every call;
compilation is excluded. Compiled memory is the compiler's estimate for one call.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import UTC, datetime
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

WEIGHTS = ("float32", "input")
BACKWARDS = ("columns", "rows")
ROOT = Path(__file__).resolve().parents[3]
RECIPE = ROOT / "experiments/transformer_pp/runs/r9040f12f3d80/config.yaml"
UPDATE_LAYER_MICRO_BATCHES = 15 * 16  # layers x micro-batches of 16 sequences in one 256-sequence update


# --------------------------------------------------------------------------------------------
# Candidate implementations
# --------------------------------------------------------------------------------------------

def _compute_dtype(dtype):
    return jnp.promote_types(dtype, jnp.float32)


def _validate(q, k, v, block_size, weights="float32"):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Q, K, V must have shape [batch, time, heads, channels]")
    if q.shape != k.shape or q.shape[:3] != v.shape[:3]:
        raise ValueError("Q/K shapes and Q/K/V batch, time, and head dimensions must match")
    if min(*q.shape, v.shape[-1]) <= 0:
        raise ValueError("Attention dimensions must be positive")
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if weights not in WEIGHTS:
        raise ValueError(f"weights must be one of {WEIGHTS}")


def _strips(length: int, block_size: int) -> list[tuple[int, int]]:
    """Static query (or key) blocks [start, end); the last one may be shorter."""
    return [(start, min(start + block_size, length)) for start in range(0, length, block_size)]


def _row_strip(q, k, v, start, weights):
    """Queries q [B,H,n,D] at positions start..start+n-1 against keys/values [B,H,end,D] at 0..end-1.

    Returns the strip output in the compute dtype and its log-sum-exp [B,H,n,1]. The output is
    (P V) / l with P = exp(S - max) and l = sum(P) over causal keys; rows with a nonfinite
    causal score are NaN in both. ``start`` may be traced (scan) or static (unrolled).
    """
    compute = _compute_dtype(q.dtype)
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k, preferred_element_type=compute) * (1.0 / math.sqrt(q.shape[-1]))
    causal = jnp.arange(k.shape[2])[None, :] <= (start + jnp.arange(q.shape[2]))[:, None]
    invalid = jnp.any(causal & ~jnp.isfinite(scores), axis=-1, keepdims=True)
    # The shift is an exact identity of softmax, so it carries no gradient.
    maximum = jax.lax.stop_gradient(jnp.max(jnp.where(causal, scores, -jnp.inf), axis=-1, keepdims=True))
    # Masked entries are -inf before exp, never exponentiated sentinels.
    weights_ = jnp.exp(jnp.where(causal, scores - maximum, -jnp.inf))
    total = jnp.sum(weights_, axis=-1, keepdims=True)
    if weights == "float32":
        unnormalized = jnp.einsum("bhqk,bhkd->bhqd", weights_, v.astype(compute), preferred_element_type=compute)
    else:
        unnormalized = jnp.einsum("bhqk,bhkd->bhqd", weights_.astype(v.dtype), v, preferred_element_type=compute)
    output = jnp.where(invalid, jnp.nan, unnormalized / total)
    log_sum_exp = jnp.where(invalid, jnp.nan, maximum + jnp.log(total))
    return output, log_sum_exp


def production_blocked_attention(q, k, v, *, block_size: int = 128):
    """The scaffold's blocked backend: a checkpointed scan of query blocks, each against all keys."""
    from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention

    return blocked_dot_product_attention(q, k, v, query_block_size=block_size)


def lean_blocked_attention(q, k, v, *, block_size: int = 128, checkpoint: bool = True,
                           causal_skip: bool = True, weights: str = "float32", upcast: bool = False):
    """Production's numerics with fewer passes; JAX autodiff backward.

    ``causal_skip`` unrolls the query blocks in Python (static T), so block b multiplies
    only keys [0, end_b); otherwise a scan runs equal (padded) blocks against all keys,
    as production. ``checkpoint`` recomputes each block's forward in the backward.
    As in production, autodiff returns each block's K/V cotangent in the input dtype and
    sums the blocks there (BF16 accumulation of dK and dV); ``upcast`` converts Q/K/V to
    FP32 before blocking, so the sum is FP32 and rounds once (the products then take FP32
    operands). With ``weights="input"`` the weights' cotangent dP is also rounded to the
    input dtype (the transpose of the rounded product), unlike cuDNN.
    """
    _validate(q, k, v, block_size, weights)
    if upcast and weights != "float32":
        raise ValueError("upcast requires float32 weights")
    batch, length, heads, _ = q.shape
    dtype = v.dtype
    if upcast:
        q, k, v = (x.astype(_compute_dtype(x.dtype)) for x in (q, k, v))
    qt, kt, vt = (jnp.swapaxes(x, 1, 2) for x in (q, k, v))
    if causal_skip:
        outputs = []
        for start, end in _strips(length, block_size):
            def strip(query, key, value, start=start):
                return _row_strip(query, key, value, start, weights)[0]

            if checkpoint:
                strip = jax.checkpoint(strip)
            outputs.append(strip(qt[:, :, start:end], kt[:, :, :end], vt[:, :, :end]))
        output = jnp.concatenate(outputs, axis=2)
    else:
        size = min(block_size, length)
        count = -(-length // size)
        padded = jnp.pad(qt, ((0, 0), (0, 0), (0, count * size - length), (0, 0)))
        blocks = jnp.moveaxis(padded.reshape(batch, heads, count, size, q.shape[-1]), 2, 0)

        def body(carry, inputs):
            query, start = inputs
            return carry, _row_strip(query, kt, vt, start, weights)[0]

        if checkpoint:
            body = jax.checkpoint(body, prevent_cse=False)
        _, result = jax.lax.scan(body, None, (blocks, jnp.arange(count, dtype=jnp.int32) * size))
        output = jnp.moveaxis(result, 0, 2).reshape(batch, heads, count * size, v.shape[-1])[:, :, :length]
    return jnp.swapaxes(output, 1, 2).astype(dtype)


def _flash_forward(q, k, v, block_size, weights):
    qt, kt, vt = (jnp.swapaxes(x, 1, 2) for x in (q, k, v))
    if weights == "float32":
        vt = vt.astype(_compute_dtype(v.dtype))  # once, not per strip
    outputs, log_sum_exps = [], []
    for start, end in _strips(q.shape[1], block_size):
        output, log_sum_exp = _row_strip(qt[:, :, start:end], kt[:, :, :end], vt[:, :, :end], start, weights)
        outputs.append(output)
        log_sum_exps.append(log_sum_exp)
    return jnp.concatenate(outputs, axis=2), jnp.concatenate(log_sum_exps, axis=2)[..., 0]


def _flash(q, k, v, block_size, weights, backward):
    return jnp.swapaxes(_flash_forward(q, k, v, block_size, weights)[0], 1, 2).astype(v.dtype)


def _flash_fwd(q, k, v, block_size, weights, backward):
    output, log_sum_exp = _flash_forward(q, k, v, block_size, weights)
    # delta = rowsum(dO * O) uses the FP32 output with FP32 weights, and the rounded
    # output with input-dtype weights (as FlashAttention and cuDNN).
    saved = output if weights == "float32" else output.astype(v.dtype)
    return jnp.swapaxes(output, 1, 2).astype(v.dtype), (q, k, v, saved, log_sum_exp)


def _flash_bwd(block_size, weights, backward, residuals, cotangent):
    q, k, v, saved, log_sum_exp = residuals
    compute = _compute_dtype(q.dtype)
    scale = 1.0 / math.sqrt(q.shape[-1])
    length = q.shape[1]
    qt, kt, vt, gt = (jnp.swapaxes(x, 1, 2) for x in (q, k, v, cotangent))
    delta = jnp.sum(gt.astype(compute) * saved.astype(compute), axis=-1)  # [B, H, T]
    # Operands of the weight products: FP32 weights multiply upcast copies (converted once),
    # input-dtype weights multiply them as stored. Score products always use the stored inputs.
    qo, ko, go = (x.astype(compute) for x in (qt, kt, gt)) if weights == "float32" else (qt, kt, gt)

    def rounded(x, like):
        return x if weights == "float32" else x.astype(like.dtype)

    def contract(spec, left, right):
        return jnp.einsum(spec, left, right, preferred_element_type=compute)

    if backward == "columns":
        # Key blocks j against queries [start_j, T): dK_j and dV_j are complete per block;
        # dQ sums one front-padded part per key block.
        dq_parts, dk_blocks, dv_blocks = [], [], []
        for start, end in _strips(length, block_size):
            key, value = kt[:, :, start:end], vt[:, :, start:end]
            query, grad = qt[:, :, start:], gt[:, :, start:]
            scores = contract("bhkd,bhqd->bhkq", key, query) * scale
            causal = jnp.arange(start, end)[:, None] <= jnp.arange(start, length)[None, :]
            weights_ = jnp.exp(jnp.where(causal, scores - log_sum_exp[:, :, None, start:], -jnp.inf))
            grad_weights = contract("bhkd,bhqd->bhkq", value, grad)
            grad_scores = jnp.where(causal, weights_ * (grad_weights - delta[:, :, None, start:]), 0.0)
            used_weights, used_scores = rounded(weights_, v), rounded(grad_scores, q)
            dv_blocks.append(contract("bhkq,bhqd->bhkd", used_weights, go[:, :, start:]))
            dk_blocks.append(contract("bhkq,bhqd->bhkd", used_scores, qo[:, :, start:]) * scale)
            part = contract("bhkq,bhkd->bhqd", used_scores, ko[:, :, start:end]) * scale
            dq_parts.append(jnp.pad(part, ((0, 0), (0, 0), (start, 0), (0, 0))))
        dq = functools.reduce(jnp.add, dq_parts)
        dk, dv = jnp.concatenate(dk_blocks, axis=2), jnp.concatenate(dv_blocks, axis=2)
    else:
        # Query blocks i against keys [0, end_i): dQ_i is complete per block; dK and dV
        # sum one back-padded part per query block.
        dq_blocks, dk_parts, dv_parts = [], [], []
        for start, end in _strips(length, block_size):
            query, grad = qt[:, :, start:end], gt[:, :, start:end]
            key, value = kt[:, :, :end], vt[:, :, :end]
            scores = contract("bhqd,bhkd->bhqk", query, key) * scale
            causal = jnp.arange(end)[None, :] <= jnp.arange(start, end)[:, None]
            weights_ = jnp.exp(jnp.where(causal, scores - log_sum_exp[:, :, start:end, None], -jnp.inf))
            grad_weights = contract("bhqd,bhkd->bhqk", grad, value)
            grad_scores = jnp.where(causal, weights_ * (grad_weights - delta[:, :, start:end, None]), 0.0)
            used_weights, used_scores = rounded(weights_, v), rounded(grad_scores, q)
            tail = ((0, 0), (0, 0), (0, length - end), (0, 0))
            dq_blocks.append(contract("bhqk,bhkd->bhqd", used_scores, ko[:, :, :end]) * scale)
            dk_parts.append(jnp.pad(contract("bhqk,bhqd->bhkd", used_scores, qo[:, :, start:end]) * scale, tail))
            dv_parts.append(jnp.pad(contract("bhqk,bhqd->bhkd", used_weights, go[:, :, start:end]), tail))
        dq = jnp.concatenate(dq_blocks, axis=2)
        dk, dv = functools.reduce(jnp.add, dk_parts), functools.reduce(jnp.add, dv_parts)
    return tuple(jnp.swapaxes(gradient, 1, 2).astype(x.dtype) for gradient, x in ((dq, q), (dk, k), (dv, v)))


_flash = jax.custom_vjp(_flash, nondiff_argnums=(3, 4, 5))
_flash.defvjp(_flash_fwd, _flash_bwd)


def flash_blocked_attention(q, k, v, *, block_size: int = 128, weights: str = "float32",
                            backward: str = "columns"):
    """Custom-VJP blocked softmax: the forward keeps only O and the per-row log-sum-exp.

    Query blocks are unrolled with causal key prefixes. The backward recomputes
    S = QK^T/sqrt(d) and P = exp(S - lse) per block and uses dS = P * (dO V^T - rowsum(dO * O)),
    over key blocks ("columns", accumulating dQ) or query blocks ("rows", accumulating dK, dV).
    Rows with a nonfinite causal score have NaN output and log-sum-exp, so their gradients
    are NaN too.
    """
    _validate(q, k, v, block_size, weights)
    if backward not in BACKWARDS:
        raise ValueError(f"backward must be one of {BACKWARDS}")
    return _flash(q, k, v, block_size, weights, backward)


def dense_manual_attention(q, k, v):
    """The scaffold's dense textbook path (backend="manual"): input-dtype scores, no blocks."""
    from scaffolds.transformer.attention.softmax import softmax_attention

    return softmax_attention(q, k, v, backend="manual")


def cudnn_attention(q, k, v):
    """The production softmax path (jax.nn.dot_product_attention, cuDNN)."""
    from scaffolds.transformer.attention.softmax import softmax_attention

    return softmax_attention(q, k, v, backend="cudnn")


@dataclass(frozen=True)
class Variant:
    name: str
    family: str
    description: str
    options: dict = field(default_factory=dict)
    failure: str = "explicit"  # explicit: NaN rows for nonfinite causal scores, as production

    def function(self):
        if self.family == "production":
            return functools.partial(production_blocked_attention, **self.options)
        if self.family == "lean":
            return functools.partial(lean_blocked_attention, **self.options)
        if self.family == "flash":
            return functools.partial(flash_blocked_attention, **self.options)
        if self.family == "dense-manual":
            return dense_manual_attention
        if self.family == "cudnn":
            return cudnn_attention
        raise ValueError(self.family)


def variants(length: int = 2048) -> dict[str, Variant]:
    """Benchmark variants, most important first. Block size ``length`` is a single dense block."""
    table = [Variant("cudnn", "cudnn", "cuDNN fused softmax (production softmax path); speed reference",
                     failure="natural")]
    for size in (128, 256, 512, 1024, 2048):
        table.append(Variant(f"production-b{size}", "production",
                             f"production blocked_dot_product_attention, {size}-query scan blocks x all keys, "
                             "checkpointed", {"block_size": size}))
    for weights, label in (("float32", "f32"), ("input", "bf16")):
        for backward in ("columns", "rows"):
            sizes = (128, 256, 512, 1024, 2048) if backward == "columns" else (128, 256)
            for size in sizes:
                table.append(Variant(
                    f"flash-{label}-{'cols' if backward == 'columns' else 'rows'}-b{size}", "flash",
                    f"custom VJP, {size}-query strips x causal key prefix, saves O+lse, "
                    f"{backward} backward, {'FP32' if weights == 'float32' else 'BF16'} weights",
                    {"block_size": size, "weights": weights, "backward": backward}))
    for weights, label in (("float32", "f32"), ("input", "bf16")):
        sizes = (128, 256, 512, 1024, 2048) if weights == "float32" else (128, 256, 512)
        for checkpoint in (True, False):
            for size in sizes:
                table.append(Variant(
                    f"lean-{label}-skip-{'ckpt' if checkpoint else 'nockpt'}-b{size}", "lean",
                    f"lean autodiff, {size}-query strips x causal key prefix, "
                    f"{'checkpointed' if checkpoint else 'no checkpoint'}, "
                    f"{'FP32' if weights == 'float32' else 'BF16'} weights",
                    {"block_size": size, "checkpoint": checkpoint, "causal_skip": True, "weights": weights}))
    for checkpoint in (True, False):
        table.append(Variant(
            f"lean-f32-scan-{'ckpt' if checkpoint else 'nockpt'}-b128", "lean",
            f"lean autodiff, scan of 128-query blocks x all keys, {'checkpointed' if checkpoint else 'no checkpoint'}, "
            "FP32 weights", {"block_size": 128, "checkpoint": checkpoint, "causal_skip": False, "weights": "float32"}))
    for checkpoint, size in ((True, 128), (True, 256), (False, 128)):
        table.append(Variant(
            f"lean-f32up-skip-{'ckpt' if checkpoint else 'nockpt'}-b{size}", "lean",
            f"lean autodiff on FP32-upcast Q/K/V (FP32 dK/dV sums), {size}-query strips x causal key prefix, "
            f"{'checkpointed' if checkpoint else 'no checkpoint'}, FP32 weights",
            {"block_size": size, "checkpoint": checkpoint, "causal_skip": True, "weights": "float32", "upcast": True}))
    table.append(Variant("dense-f32scores-bf16w", "lean",
                         "dense reference: FP32 scores/softmax, BF16 weights x V, one block, no checkpoint",
                         {"block_size": length, "checkpoint": False, "causal_skip": True, "weights": "input"}))
    table.append(Variant("dense-manual", "dense-manual",
                         "dense textbook softmax_attention(backend='manual'): BF16 scores, BF16 weights, no checkpoint",
                         failure="natural"))
    return {variant.name: variant for variant in table}


# --------------------------------------------------------------------------------------------
# FP64 oracle
# --------------------------------------------------------------------------------------------

def oracle(q, k, v, cotangent):
    """Host NumPy FP64 causal softmax attention and its analytic VJP; [B,T,H,D] arrays in, out."""
    q, k, v, g = (np.asarray(x).astype(np.float64).transpose(0, 2, 1, 3) for x in (q, k, v, cotangent))
    scale = 1.0 / math.sqrt(q.shape[-1])
    length = q.shape[2]
    scores = np.where(np.tril(np.ones((length, length), dtype=bool)), q @ np.swapaxes(k, -1, -2) * scale, -np.inf)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    output = weights @ v
    grad_weights = g @ np.swapaxes(v, -1, -2)
    grad_scores = weights * (grad_weights - np.sum(weights * grad_weights, axis=-1, keepdims=True))
    dq = grad_scores @ k * scale
    dk = np.swapaxes(grad_scores, -1, -2) @ q * scale
    dv = np.swapaxes(weights, -1, -2) @ g
    return tuple(x.transpose(0, 2, 1, 3) for x in (output, dq, dk, dv))


def dense_float64_reference(q, k, v):
    """Independent dense JAX formula for autodiff cross-checks (enable x64 first)."""
    scores = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(q.shape[-1])
    causal = jnp.tril(jnp.ones(scores.shape[-2:], dtype=bool))
    weights = jax.nn.softmax(jnp.where(causal, scores, -jnp.inf), axis=-1)
    return jnp.einsum("bhts,bshd->bthd", weights, v)


def relative_l2(actual, reference) -> float:
    actual, reference = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    norm = float(np.linalg.norm(reference))
    error = float(np.linalg.norm(actual - reference))
    return error / norm if norm else (0.0 if error == 0 else math.inf)


def oracle_check(shapes=((2, 37, 3, 16), (1, 1, 2, 8), (2, 130, 2, 32)), seed: int = 3) -> dict:
    """Compare the NumPy oracle with float64 JAX autodiff of an independent dense formula."""
    with jax.enable_x64():
        rng = np.random.default_rng(seed)
        cases = []
        for shape in shapes:
            q, k, v, g = (rng.standard_normal(shape) for _ in range(4))
            q *= 2.0
            expected_output, vjp = jax.vjp(dense_float64_reference, *(jnp.asarray(x) for x in (q, k, v)))
            expected = (expected_output, *vjp(jnp.asarray(g)))
            actual = oracle(q, k, v, g)
            cases.append({"shape": list(shape), **{
                f"{name}_relative_l2": relative_l2(a, np.asarray(b))
                for name, a, b in zip(("output", "dq", "dk", "dv"), actual, expected, strict=True)}})
    worst = max(value for case in cases for key, value in case.items() if key.endswith("relative_l2"))
    return {"reference": "float64 jax.vjp of jax.nn.softmax over -inf-masked scores", "cases": cases,
            "worst_relative_l2": worst, "passed": worst < 1e-12}


# --------------------------------------------------------------------------------------------
# Measurement helpers
# --------------------------------------------------------------------------------------------

def forward_backward(attend):
    def program(q, k, v, cotangent):
        output, pullback = jax.vjp(attend, q, k, v)
        return output, pullback(cotangent)

    return program


def kernel_count(compiled) -> int | None:
    """Fusions and custom calls outside fused computations in the optimized HLO.

    A diagnostic of launch count: loop bodies (scans) count once although they run per iteration.
    """
    try:
        text = compiled.as_text()
    except Exception:  # noqa: BLE001
        return None
    fused = set(re.findall(r"\bfusion\(.*?calls=%?([\w.\-]+)", text))
    count, current = 0, None
    for line in text.splitlines():
        if line and not line[0].isspace() and line.rstrip().endswith("{"):
            words = line.split()
            current = (words[1] if words[0] == "ENTRY" else words[0]).lstrip("%")
        elif current is not None and current not in fused and re.search(r"\b(fusion|custom-call)\(", line):
            count += 1
    return count


def compiled_memory(compiled) -> dict:
    try:
        stats = compiled.memory_analysis()
    except Exception as error:  # noqa: BLE001
        return {"available": False, "error": f"{type(error).__name__}: {error}"}
    if stats is None:
        return {"available": False}
    return {name: int(getattr(stats, name)) for name in (
        "argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes",
        "generated_code_size_in_bytes") if getattr(stats, name, None) is not None}


def measure(program, arguments, *, warmup: int, iterations: int):
    started = time.perf_counter()
    compiled = jax.jit(program).lower(*arguments).compile()
    compile_seconds = time.perf_counter() - started
    for _ in range(warmup):
        jax.block_until_ready(compiled(*arguments))
    samples, result = [], None
    for _ in range(iterations):
        started = time.perf_counter()
        result = compiled(*arguments)
        jax.block_until_ready(result)
        samples.append((time.perf_counter() - started) * 1e3)
    return {
        "median_ms": float(np.median(samples)), "p10_ms": float(np.quantile(samples, 0.1)),
        "p90_ms": float(np.quantile(samples, 0.9)), "samples_ms": samples,
        "warmup": warmup, "iterations": iterations, "compile_seconds": compile_seconds,
        "compiled_memory": compiled_memory(compiled), "kernels": kernel_count(compiled),
    }, result, compiled


def row_ranges(mask) -> list[list[int]]:
    """[start, end) ranges of True entries."""
    mask = np.asarray(mask, dtype=bool)
    edges = np.flatnonzero(np.diff(np.concatenate([[False], mask, [False]]).astype(np.int8)))
    return [[int(start), int(end)] for start, end in zip(edges[::2], edges[1::2])]


def failure_cases(q, k, v, *, position: int, head: int = 1) -> dict[str, tuple]:
    """Inputs with one injected nonfinite or overflowing value in (batch 0, ``head``).

    Returns name -> (q, k, v, causal rows that must be nonfinite under production's rule).
    """
    length = q.shape[1]
    q, k, v = (np.array(x) for x in (q, k, v))
    dtype = q.dtype
    cases = {}

    def put(array, index, value):
        changed = array.copy()
        changed[index] = np.asarray(value).astype(dtype)
        return changed

    single = np.zeros(length, dtype=bool)
    single[position] = True
    suffix = np.arange(length) >= position
    cases["q_nan"] = (put(q, (0, position, head, 0), np.nan), k, v, single)
    cases["k_nan"] = (q, put(k, (0, position, head, 0), np.nan), v, suffix)
    cases["v_nan"] = (q, k, put(v, (0, position, head, 0), np.nan), suffix)
    cases["v_inf"] = (q, k, put(v, (0, position, head, 0), np.inf), suffix)
    # Positive queries in this head make every score against a -inf key exactly -inf.
    positive = q.copy()
    positive[0, :, head] = (np.abs(q[0, :, head].astype(np.float32)) + 0.25).astype(dtype)
    cases["score_neg_inf"] = (positive, put(k, (0, position, head, slice(None)), -np.inf), v, suffix)
    # Finite inputs whose own score overflows FP32 (1e40 / sqrt(d)); channel 0 is zero elsewhere in
    # this head, so every other score is unchanged.
    zero_q, zero_k = put(q, (0, slice(None), head, 0), 0.0), put(k, (0, slice(None), head, 0), 0.0)
    cases["score_overflow"] = (put(zero_q, (0, position, head, 0), 1e20), put(zero_k, (0, position, head, 0), 1e20),
                               v, single)
    # The same token large in every channel: its own score overflows, and other rows get huge finite
    # scores (|s| ~ 1e20) against it. exp(s - max) is then sensitive to how s * scale is rounded
    # (e.g. fused multiply-add), so such rows may also become nonfinite.
    big_q, big_k = put(q, (0, position, head, slice(None)), 1e20), put(k, (0, position, head, slice(None)), 1e20)
    cases["score_huge"] = (big_q, big_k, v, single)
    return cases


def failure_report(compiled, cases: dict, cotangent, *, head: int = 1) -> dict:
    report = {}
    for name, (q, k, v, expected) in cases.items():
        output, grads = compiled(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), cotangent)
        bad = ~jnp.isfinite(output.astype(jnp.float32))
        rows, others, grads_finite = jax.device_get((
            jnp.any(bad[0, :, head], axis=-1),
            ~jnp.any(bad.at[0, :, head].set(False)),
            jnp.all(jnp.stack([jnp.all(jnp.isfinite(g.astype(jnp.float32))) for g in grads])),
        ))
        rows = np.asarray(rows)
        report[name] = {
            "nonfinite_rows": row_ranges(rows), "expected_rows": row_ranges(expected),
            "covers_expected": bool(np.all(rows[expected])), "extra_rows": row_ranges(rows & ~expected),
            "other_heads_finite": bool(others), "gradients_finite": bool(grads_finite),
        }
    return report


def matmul_precision_probe() -> dict:
    """Relative error of an FP32 x FP32 matmul at JAX's default and highest precision."""
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((512, 2048)).astype(np.float32), rng.standard_normal((2048, 64)).astype(np.float32)
    exact = a.astype(np.float64) @ b.astype(np.float64)
    result = {}
    for name, precision in (("default", None), ("highest", jax.lax.Precision.HIGHEST)):
        product = jax.jit(lambda x, y, p=precision: jnp.dot(x, y, precision=p))(jnp.asarray(a), jnp.asarray(b))
        result[f"{name}_relative_l2"] = relative_l2(np.asarray(product), exact)
    return result


def provenance() -> dict:
    root = Path(__file__).resolve().parents[3]

    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain", "--untracked-files=no")
    devices = jax.local_devices()
    versions = {}
    import importlib.metadata
    for name in ("jax", "jaxlib", "jax-cuda12-plugin", "flax", "numpy", "nvidia-cudnn-cu12", "nvidia-cublas-cu12"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "source_commit": git("rev-parse", "HEAD"), "tracked_files_dirty": None if status is None else bool(status),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "hostname": os.uname().nodename, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "device_kind": devices[0].device_kind, "platform": devices[0].platform, "device_count": len(devices),
        "package_versions": versions,
        "environment": {name: os.environ.get(name) for name in (
            "JAX_PLATFORMS", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION",
            "CUDA_VISIBLE_DEVICES")},
    }


def _json_safe(value):
    """Strict JSON: nonfinite floats become the strings "NaN", "Infinity" and "-Infinity"."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_json_safe(value), indent=1, allow_nan=False) + "\n")
    temporary.replace(path)


def _bf16_inputs(shape, seed, sharp):
    rng = np.random.default_rng(seed)
    q, k, v, g = (rng.standard_normal(shape, dtype=np.float32) for _ in range(4))
    if sharp:
        q *= 4.0
    return tuple(jnp.asarray(x, jnp.bfloat16) for x in (q, k, v, g))


# --------------------------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------------------------

def kernel_stage(args) -> dict:
    table = variants(args.time_shape[1])
    names = args.variants or list(table)
    report = {"format": "blocked-softmax-kernel-benchmark", "schema_version": 1, "status": "running",
              "created_at_utc": datetime.now(UTC).isoformat(), "provenance": provenance(),
              "configuration": {"accuracy_shape": args.accuracy_shape, "time_shape": args.time_shape,
                                "dtype": "bfloat16", "warmup": args.warmup, "iterations": args.iterations,
                                "failure_position": args.failure_position,
                                "update_layer_micro_batches": UPDATE_LAYER_MICRO_BATCHES,
                                "timing": "median of synchronized compiled calls after warmup; compile excluded",
                                "memory": "compiled.memory_analysis temp bytes (estimate, one call)",
                                "backward": "jax.vjp of the attention; outputs, dq, dk, dv"},
              "matmul_precision": matmul_precision_probe(), "variants": {}}
    write_json(args.output, report)
    deadline = time.monotonic() + 60 * args.budget_minutes

    accuracy_inputs = {name: _bf16_inputs(args.accuracy_shape, args.seed, name == "sharp") for name in ("normal", "sharp")}
    oracles, floors = {}, {}
    for name, (q, k, v, g) in accuracy_inputs.items():
        oracles[name] = oracle(*(np.asarray(x.astype(jnp.float32)) for x in (q, k, v, g)))
        floors[name] = {label: relative_l2(np.asarray(jnp.asarray(x, jnp.bfloat16).astype(jnp.float32)), x)
                        for label, x in zip(("output", "dq", "dk", "dv"), oracles[name], strict=True)}
    report["bf16_rounding_floor"] = floors
    float32_inputs = tuple(x.astype(jnp.float32) for x in accuracy_inputs["normal"])
    float32_oracle = oracle(*(np.asarray(x) for x in float32_inputs))
    cases = failure_cases(*(np.asarray(x) for x in accuracy_inputs["normal"][:3]), position=args.failure_position)
    time_inputs = _bf16_inputs(args.time_shape, args.seed + 1, False)

    for name in names:
        variant = table[name]
        entry = {"family": variant.family, "description": variant.description, "options": variant.options,
                 "failure_rule": variant.failure}
        report["variants"][name] = entry
        if time.monotonic() > deadline:
            entry["skipped"] = "time budget exhausted"
            write_json(args.output, report)
            continue
        attend = variant.function()
        program = forward_backward(attend)
        print(f"START {name}", flush=True)
        try:
            compiled = jax.jit(program).lower(*accuracy_inputs["normal"]).compile()
            entry["accuracy"] = {}
            for distribution, inputs in accuracy_inputs.items():
                output, grads = compiled(*inputs)
                entry["accuracy"][distribution] = {
                    label: relative_l2(np.asarray(x.astype(jnp.float32)), reference)
                    for label, x, reference in zip(("output", "dq", "dk", "dv"), (output, *grads),
                                                   oracles[distribution], strict=True)}
            entry["failure"] = failure_report(compiled, cases, accuracy_inputs["normal"][3])
            del compiled
            if name in (args.float32_input_variants or ()):
                output, grads = jax.jit(program)(*float32_inputs)
                entry["accuracy"]["float32_inputs"] = {
                    label: relative_l2(np.asarray(x), reference)
                    for label, x, reference in zip(("output", "dq", "dk", "dv"), (output, *grads),
                                                   float32_oracle, strict=True)}
            forward, result, _ = measure(attend, time_inputs[:3], warmup=args.warmup, iterations=args.iterations)
            del result
            both, result, _ = measure(program, time_inputs, warmup=args.warmup, iterations=args.iterations)
            finite = bool(jax.device_get(jnp.all(jnp.stack(
                [jnp.all(jnp.isfinite(x.astype(jnp.float32))) for x in jax.tree.leaves(result)]))))
            del result
            entry["timing"] = {"forward": forward, "forward_backward": both, "outputs_finite": finite,
                               "attention_seconds_per_update": both["median_ms"] * UPDATE_LAYER_MICRO_BATCHES / 1e3}
            print(f"DONE {name} forward={forward['median_ms']:.2f}ms forward_backward={both['median_ms']:.2f}ms "
                  f"temp={both['compiled_memory'].get('temp_size_in_bytes', 0) / 1e9:.2f}GB "
                  f"accuracy={json.dumps(entry['accuracy']['normal'])}", flush=True)
        except Exception as error:  # noqa: BLE001
            entry["error"] = f"{type(error).__name__}: {str(error)[:2000]}"
            print(f"FAILED {name}: {entry['error'][:500]}", flush=True)
        jax.clear_caches()
        write_json(args.output, report)

    # Repeat two reference rows to bound drift over the job.
    report["repeats"] = {}
    for name in ("cudnn", "production-b128"):
        if name in names and "timing" in report["variants"][name]:
            attend = table[name].function()
            both, result, _ = measure(forward_backward(attend), time_inputs, warmup=args.warmup,
                                      iterations=args.iterations)
            del result
            report["repeats"][name] = {"forward_backward_median_ms": both["median_ms"]}
    report["status"] = "complete"
    report["completed_at_utc"] = datetime.now(UTC).isoformat()
    write_json(args.output, report)
    return report


def _routed_softmax(attend):
    def softmax_attention(q, k, v, *, backend="blocked", return_diagnostics=False, dropout=None,
                          deterministic=True, rngs=None, weights_dtype=None):
        if return_diagnostics:
            raise ValueError("Benchmark routing computes no attention diagnostics")
        if dropout is not None and getattr(dropout, "rate", 0.0) != 0.0:
            raise ValueError("Benchmark routing requires dropout 0")
        return attend(q, k, v)

    return softmax_attention


def load_model_config(recipe: Path, *, micro_batch: int, backend: str):
    from scaffolds.transformer.configs import load_config

    config = load_config(recipe)
    config.data.source = "synthetic"
    config.data.synthetic_vocab_size = config.model.vocab_size
    config.train.micro_batch_size = micro_batch
    config.model.attention_backend = backend
    return config


def micro_step_program(config, *, variant: str | None):
    """(state, jit-able f(params, batch, rng) -> (nll, grads)) running the scaffold's loss_and_grads.

    ``variant`` None keeps the configured backend; otherwise softmax attention is routed to it.
    """
    from types import SimpleNamespace

    import scaffolds.transformer.attention.softmax as softmax_module
    from scaffolds.transformer.training import loop
    from scaffolds.transformer.training.state import create_state

    if variant is not None:
        softmax_module.softmax_attention = _routed_softmax(variants(config.model.max_seq_len)[variant].function())
    state = create_state(config, jax.random.PRNGKey(config.train.seed))

    def program(params, batch, rng):
        return loop.loss_and_grads(SimpleNamespace(graphdef=state.graphdef, params=params), batch, rng)

    return state, program


def model_case(args) -> dict:
    """One (variant, micro-batch) forward+backward micro-step; run in a fresh process."""
    started = time.perf_counter()
    result = {"variant": args.variant, "micro_batch": args.micro_batch, "status": "running",
              "provenance": provenance()}
    try:
        routed = args.variant not in ("production-b128", "cudnn")
        backend = "cudnn" if args.variant == "cudnn" else "blocked"
        config = load_model_config(args.recipe, micro_batch=args.micro_batch, backend=backend)
        state, program = micro_step_program(config, variant=args.variant if routed else None)
        result["routing"] = (f"model.attention_backend={backend}" if not routed else
                             f"model.attention_backend=blocked; softmax_attention routed to {args.variant}")
        batch = jnp.asarray(np.random.default_rng(config.train.seed).integers(
            0, config.model.vocab_size, (args.micro_batch, config.data.sequence_length), dtype=np.int32))
        rng = jax.random.PRNGKey(config.train.seed)
        timing, (nll, grads), _ = measure(program, (state.params, batch, rng), warmup=args.warmup,
                                          iterations=args.iterations)
        import optax
        nll, grad_norm = (float(x) for x in jax.device_get((nll, optax.tree.norm(grads))))
        del grads
        stats = jax.local_devices()[0].memory_stats() or {}
        result.update({
            "status": "complete", "timing": timing, "nll": nll, "grad_norm": grad_norm,
            "finite": math.isfinite(nll) and math.isfinite(grad_norm),
            "seconds_per_update": timing["median_ms"] * config.data.batch_size / args.micro_batch / 1e3,
            "peak_bytes_in_use": stats.get("peak_bytes_in_use"), "bytes_limit": stats.get("bytes_limit"),
            "global_batch": config.data.batch_size,
        })
    except Exception as error:  # noqa: BLE001
        message = f"{type(error).__name__}: {str(error)[:3000]}"
        result.update({"status": "failed", "error": message,
                       "out_of_memory": "RESOURCE_EXHAUSTED" in message or "out of memory" in message.lower()})
    result["wall_seconds"] = time.perf_counter() - started
    write_json(args.output, result)
    return result


def model_stage(args) -> dict:
    recipe = Path(args.recipe).resolve()
    report = {"format": "blocked-softmax-model-benchmark", "schema_version": 1, "status": "running",
              "created_at_utc": datetime.now(UTC).isoformat(),
              "recipe": str(recipe.relative_to(ROOT)) if recipe.is_relative_to(ROOT) else str(recipe),
              "recipe_sha256": hashlib.sha256(recipe.read_bytes()).hexdigest(),
              "configuration": {"micro_batches": args.micro_batches, "warmup": args.warmup,
                                "iterations": args.iterations,
                                "timing": "jit(training.loop.loss_and_grads): one forward+backward micro-step "
                                          "(no optimizer update), median of synchronized calls; compile excluded",
                                "memory": "allocator peak_bytes_in_use of a fresh process holding the full "
                                          "train state (params and AdamW moments) plus the micro-step"},
              "cases": []}
    scratch = args.output.parent
    for variant in args.variants:
        for micro_batch in args.micro_batches:
            path = scratch / f".model-case-{variant}-{micro_batch}-{os.getpid()}.json"
            command = [sys.executable, "-m", "scaffolds.transformer.validation.blocked_attention_benchmark",
                       "model-case", "--variant", variant, "--micro-batch", str(micro_batch),
                       "--recipe", str(args.recipe), "--output", str(path),
                       "--warmup", str(args.warmup), "--iterations", str(args.iterations)]
            print(f"START model {variant} micro_batch={micro_batch}", flush=True)
            completed = subprocess.run(command, capture_output=True, text=True, timeout=args.case_timeout)
            if path.exists():
                case = json.loads(path.read_text())
                path.unlink()
            else:
                case = {"variant": variant, "micro_batch": micro_batch, "status": "crashed",
                        "returncode": completed.returncode, "stderr_tail": completed.stderr[-3000:]}
            if "provenance" in case:
                # Every case runs the same commit on the same GPU; keep one copy.
                report.setdefault("provenance", case["provenance"])
                case["provenance_matches"] = case.pop("provenance") == report["provenance"]
            report["cases"].append(case)
            summary = {key: case.get(key) for key in ("status", "seconds_per_update", "peak_bytes_in_use", "nll")}
            if case.get("timing"):
                summary["median_ms"] = case["timing"]["median_ms"]
                summary["compile_seconds"] = case["timing"]["compile_seconds"]
            print(f"DONE model {variant} micro_batch={micro_batch} {json.dumps(summary)}", flush=True)
            write_json(args.output, report)
    report["status"] = "complete"
    report["completed_at_utc"] = datetime.now(UTC).isoformat()
    write_json(args.output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("oracle-check", "kernel", "model", "model-case"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--variant")
    parser.add_argument("--accuracy-shape", type=int, nargs=4, default=[2, 2048, 4, 64])
    parser.add_argument("--time-shape", type=int, nargs=4, default=[16, 2048, 14, 64])
    parser.add_argument("--failure-position", type=int, default=1000)
    parser.add_argument("--float32-input-variants", nargs="+",
                        help="Also measure these variants' accuracy with FP32 inputs (precision of the products)")
    parser.add_argument("--micro-batches", type=int, nargs="+", default=[16, 8, 4])
    parser.add_argument("--micro-batch", type=int)
    parser.add_argument("--recipe", type=Path, default=RECIPE)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget-minutes", type=float, default=75.0)
    parser.add_argument("--case-timeout", type=float, default=1200.0)
    args = parser.parse_args()
    if args.iterations < 10 or args.warmup < 1:
        parser.error("Require at least 10 timed iterations and one warmup call")
    if args.stage == "oracle-check":
        write_json(args.output, {"format": "blocked-softmax-oracle-check", "schema_version": 1,
                                 "created_at_utc": datetime.now(UTC).isoformat(), "provenance": provenance(),
                                 **oracle_check()})
    elif args.stage == "kernel":
        known = variants(args.time_shape[1])
        unknown = sorted(set(args.variants or ()) - set(known))
        if unknown:
            parser.error(f"Unknown variants: {unknown}")
        kernel_stage(args)
    elif args.stage == "model":
        if not args.variants:
            parser.error("The model stage requires --variants")
        model_stage(args)
    else:
        if args.variant is None or args.micro_batch is None:
            parser.error("model-case requires --variant and --micro-batch")
        model_case(args)


if __name__ == "__main__":
    main()
