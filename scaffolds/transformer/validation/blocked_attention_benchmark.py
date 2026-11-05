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
compilation is excluded. Compiled memory is the compiler's estimate for one call; it covers one
layer, not memory a model holds across layers (each layer's residuals until its backward).
Accuracy and failure programs compile at the accuracy shape without GEMM autotuning; the timed
programs (XLA defaults, timing shape) are also scored on their first sequence.
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
BACKWARDS = ("columns", "rows", "saved")
ROOT = Path(__file__).resolve().parents[3]
RECIPE = ROOT / "experiments/transformer_pp/runs/r9040f12f3d80/config.yaml"
UPDATE_LAYER_MICRO_BATCHES = 15 * 16  # layers x micro-batches of 16 sequences in one 256-sequence update
TIMED_ACCURACY_SEQUENCES = 1  # sequences of each timed program's result scored against the oracle


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


def _rounded_to(x, dtype):
    """``x`` rounded to ``dtype``'s precision and kept in x's dtype; ``x`` itself if dtype is not narrower.

    Write every intended rounding of a value that is upcast again this way. With the default
    xla_allow_excess_precision, XLA GPU removes a conversion pair f32 -> bf16 -> f32 when the
    second conversion directly consumes the first, so ``x.astype(bf16).astype(f32)`` may not
    round (test_xla_gpu_removes_bf16_round_trips); reduce_precision is kept.
    For BF16 (FP32's exponent range) it equals the conversion round trip; for narrower exponent
    ranges XLA flushes values below the normal range to zero.
    """
    source, target = jnp.finfo(x.dtype), jnp.finfo(dtype)
    if target.nmant >= source.nmant and target.nexp >= source.nexp:
        return x
    return jax.lax.reduce_precision(x, exponent_bits=target.nexp, mantissa_bits=target.nmant)


def _strips(length: int, block_size: int) -> list[tuple[int, int]]:
    """Static query (or key) blocks [start, end); the last one may be shorter."""
    return [(start, min(start + block_size, length)) for start in range(0, length, block_size)]


def _row_strip(q, k, v, start, weights, normalize_first=False):
    """Queries q [B,H,n,D] at positions start..start+n-1 against keys/values [B,H,end,D] at 0..end-1.

    Returns the strip output in the compute dtype and its log-sum-exp [B,H,n,1]. The output is
    (P V) / l with P = exp(S - max) and l = sum(P) over causal keys; rows with a nonfinite
    causal score are NaN in both. ``start`` may be traced (scan) or static (unrolled).
    ``normalize_first`` multiplies the normalized weights P / l instead and also returns them,
    in the dtype that entered the product (NaN rows where the output is NaN).
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
    if normalize_first:
        weights_ = weights_ / total
    used = weights_ if weights == "float32" else weights_.astype(v.dtype)
    product = jnp.einsum("bhqk,bhkd->bhqd", used, v.astype(compute) if weights == "float32" else v,
                         preferred_element_type=compute)
    output = jnp.where(invalid, jnp.nan, product if normalize_first else product / total)
    log_sum_exp = jnp.where(invalid, jnp.nan, maximum + jnp.log(total))
    if normalize_first:
        return output, log_sum_exp, jnp.where(invalid, jnp.nan, used).astype(used.dtype)
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


def _flash_forward(q, k, v, block_size, weights, backward):
    """Strip outputs [B,H,T,D] (compute dtype), log-sum-exp [B,H,T], and, for the "saved"
    backward, each strip's normalized weights [B,H,n,end] as they entered P x V."""
    qt, kt, vt = (jnp.swapaxes(x, 1, 2) for x in (q, k, v))
    if weights == "float32":
        vt = vt.astype(_compute_dtype(v.dtype))  # once, not per strip
    outputs, log_sum_exps, kept = [], [], []
    for start, end in _strips(q.shape[1], block_size):
        result = _row_strip(qt[:, :, start:end], kt[:, :, :end], vt[:, :, :end], start, weights,
                            normalize_first=backward == "saved")
        outputs.append(result[0])
        log_sum_exps.append(result[1])
        kept.extend(result[2:])
    return jnp.concatenate(outputs, axis=2), jnp.concatenate(log_sum_exps, axis=2)[..., 0], tuple(kept)


def _flash(q, k, v, block_size, weights, backward):
    return jnp.swapaxes(_flash_forward(q, k, v, block_size, weights, backward)[0], 1, 2).astype(v.dtype)


def _flash_fwd(q, k, v, block_size, weights, backward):
    output, log_sum_exp, kept = _flash_forward(q, k, v, block_size, weights, backward)
    # delta = rowsum(dO * O) uses the FP32 output with FP32 weights, and the output rounded to
    # the input dtype with input-dtype weights (as FlashAttention and cuDNN); the residual is
    # stored rounded.
    saved = output if weights == "float32" else output.astype(v.dtype)
    return jnp.swapaxes(output, 1, 2).astype(v.dtype), (q, k, v, saved, log_sum_exp, kept)


def _flash_bwd(block_size, weights, backward, residuals, cotangent):
    # Recompute only from values that pass this barrier with the cotangent. Without it XLA merges
    # the backward's score products with the forward's identical ones (or schedules them early)
    # and keeps every layer's scores alive between the passes, as checkpointing would prevent.
    residuals, cotangent = jax.lax.optimization_barrier((residuals, cotangent))
    q, k, v, saved, log_sum_exp, kept = residuals
    compute = _compute_dtype(q.dtype)
    scale = 1.0 / math.sqrt(q.shape[-1])
    length = q.shape[1]
    qt, kt, vt, gt = (jnp.swapaxes(x, 1, 2) for x in (q, k, v, cotangent))
    # Residuals stored in the input dtype enter FP32 arithmetic with their rounding restated, so it
    # does not depend on the barrier keeping the two conversions apart (see _rounded_to).
    output = _rounded_to(saved.astype(compute), saved.dtype)
    delta = jnp.sum(gt.astype(compute) * output, axis=-1)  # [B, H, T]
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
        # sum one back-padded part per query block. "saved" reuses the forward's weights.
        dq_blocks, dk_parts, dv_parts = [], [], []
        for index, (start, end) in enumerate(_strips(length, block_size)):
            query, grad = qt[:, :, start:end], gt[:, :, start:end]
            key, value = kt[:, :, :end], vt[:, :, :end]
            causal = jnp.arange(end)[None, :] <= jnp.arange(start, end)[:, None]
            if backward == "saved":
                weights_ = _rounded_to(kept[index].astype(compute), kept[index].dtype)
            else:
                scores = contract("bhqd,bhkd->bhqk", query, key) * scale
                weights_ = jnp.exp(jnp.where(causal, scores - log_sum_exp[:, :, start:end, None], -jnp.inf))
            grad_weights = contract("bhqd,bhkd->bhqk", grad, value)
            grad_scores = jnp.where(causal, weights_ * (grad_weights - delta[:, :, start:end, None]), 0.0)
            used_weights = kept[index] if backward == "saved" else rounded(weights_, v)
            used_scores = rounded(grad_scores, q)
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
    "saved" instead keeps every strip's normalized weights from the forward, in the weights
    dtype, and runs the rows backward without recomputing them (memory O(T^2) per layer; the
    forward multiplies P / l rather than dividing P V by l). Rows with a nonfinite causal score
    have NaN output and log-sum-exp (and saved weights), so their gradients are NaN too.
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
    """Benchmark variants: references first, then candidates by priority. Block ``length`` is dense."""
    precision = {"float32": ("f32", "FP32"), "input": ("bf16", "BF16")}
    table = [
        Variant("cudnn", "cudnn", "cuDNN fused softmax (the production softmax path); speed reference",
                failure="natural"),
        Variant("dense-manual", "dense-manual", "dense textbook softmax_attention(backend='manual'): BF16 scores "
                "and weights, no blocks, no checkpoint", failure="natural"),
        Variant("dense-f32scores-bf16w", "lean", "dense reference: FP32 scores and softmax, BF16 weights x V, "
                "one block, no checkpoint", {"block_size": length, "checkpoint": False, "causal_skip": True,
                                             "weights": "input"}),
    ]
    for size in (128, 256, 512, 1024, 2048):
        table.append(Variant(f"production-b{size}", "production", f"production blocked_dot_product_attention: "
                             f"checkpointed scan of {size}-query blocks x all keys", {"block_size": size}))
    for checkpoint in (True, False):
        table.append(Variant(f"lean-f32-scan-{'ckpt' if checkpoint else 'nockpt'}-b128", "lean",
                             f"lean autodiff, scan of 128-query blocks x all keys, "
                             f"{'checkpointed' if checkpoint else 'no checkpoint'}, FP32 weights",
                             {"block_size": 128, "checkpoint": checkpoint, "causal_skip": False, "weights": "float32"}))

    def flash(weights, backward, size):
        label, name = precision[weights]
        short = {"columns": "cols", "rows": "rows", "saved": "saved"}[backward]
        how = {"columns": "recomputes S and P over key blocks (accumulates dQ)",
               "rows": "recomputes S and P over query blocks (accumulates dK, dV)",
               "saved": "keeps the normalized weights from the forward (rows backward, no recompute)"}[backward]
        return Variant(f"flash-{label}-{short}-b{size}", "flash",
                       f"custom VJP, {size}-query strips x causal key prefix; backward {how}; {name} weights",
                       {"block_size": size, "weights": weights, "backward": backward})

    def lean(weights, checkpoint, size, upcast=False):
        label, name = precision[weights]
        return Variant(f"lean-{label}{'up' if upcast else ''}-skip-{'ckpt' if checkpoint else 'nockpt'}-b{size}",
                       "lean", f"lean autodiff{' on FP32-upcast Q/K/V (FP32 dK/dV sums)' if upcast else ''}, "
                       f"{size}-query strips x causal key prefix, {'checkpointed' if checkpoint else 'no checkpoint'}, "
                       f"{name} weights", {"block_size": size, "checkpoint": checkpoint, "causal_skip": True,
                                            "weights": weights, **({"upcast": True} if upcast else {})})

    for weights in ("float32", "input"):
        table += [flash(weights, "columns", size) for size in (128, 256, 512, 1024, 2048)]
    for backward in ("saved", "rows"):
        table += [flash(weights, backward, size) for weights in ("float32", "input") for size in (256, 512)]
    table += [lean("float32", True, 256, upcast=True), lean("float32", True, 512, upcast=True),
              lean("float32", False, 256, upcast=True)]
    table += [lean("float32", True, size) for size in (128, 256, 512, 1024, 2048)]
    table += [lean("float32", False, size) for size in (256, 512, 2048)]
    table += [lean("input", checkpoint, size) for checkpoint in (True, False) for size in (256, 512)]
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


KERNEL_CLASSES = ("gemm", "fused_attention", "other")
FUSED_ATTENTION_KERNEL = re.compile(r"cudnn", re.IGNORECASE)  # only the cuDNN attention calls cuDNN here
GEMM_KERNEL = re.compile(r"gemm|nvjet|cutlass|xmma|matmul|cublas|sm90_", re.IGNORECASE)


def kernel_class(name: str) -> str:
    """Class of a GPU kernel by name: cuDNN fused attention (all its kernels), GEMM (cuBLAS,
    CUTLASS, XLA's Triton GEMM fusions), or other (XLA's elementwise, reduction, transpose and
    copy kernels)."""
    if FUSED_ATTENTION_KERNEL.search(name):
        return "fused_attention"
    return "gemm" if GEMM_KERNEL.search(name) else "other"


_HLO_ARRAY = re.compile(r"\b(pred|[su](?:4|8|16|32|64)|f8e\w+|bf16|f16|f32|f64|c64|c128)\[([\d,]*)\]")
_HLO_ITEM_BYTES = {"pred": 1, "s4": 0.5, "u4": 0.5, "s8": 1, "u8": 1, "s16": 2, "u16": 2, "s32": 4, "u32": 4,
                   "s64": 8, "u64": 8, "bf16": 2, "f16": 2, "f32": 4, "f64": 8, "c64": 8, "c128": 16}
_HLO_INSTRUCTION = re.compile(r"^\s+(ROOT\s+)?%?([\w.\-]+)\s*=\s*")
_HLO_NOT_KERNELS = {"parameter", "constant", "get-tuple-element", "tuple", "bitcast", "opt-barrier",
                    "optimization-barrier", "after-all", "partition-id", "replica-id", "add-dependency"}


def _hlo_arrays(shape: str) -> list[tuple[str, list[int]]]:
    return [(dtype, [int(d) for d in dims.split(",") if d]) for dtype, dims in _HLO_ARRAY.findall(shape)]


def _hlo_bytes(shape: str) -> float:
    return sum(_HLO_ITEM_BYTES.get(dtype, 1) * math.prod(dims) for dtype, dims in _hlo_arrays(shape))


def _closing(text: str, start: int) -> int:
    """Index of the bracket closing text[start]."""
    depth = 0
    for index in range(start, len(text)):
        depth += text[index] in "([{"
        depth -= text[index] in ")]}"
        if depth == 0:
            return index
    raise ValueError(f"Unbalanced HLO text: {text[:200]}")


def _split_top_level(text: str) -> list[str]:
    parts, depth, start = [], 0, 0
    for index, char in enumerate(text):
        depth += char in "([{"
        depth -= char in ")]}"
        if char == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    return [part for part in [*parts, text[start:]] if part.strip()]


def hlo_computations(text: str) -> dict:
    """Optimized HLO text -> {computation: {"entry", "instructions": [...]}}; each instruction has
    name, shape, opcode, operand names and its attribute text."""
    computations, current = {}, None
    for line in text.splitlines():
        if line and not line[0].isspace():
            current = None
            words = line.split()
            if line.rstrip().endswith("{") and words and words[0] != "HloModule":
                name = (words[1] if words[0] == "ENTRY" else words[0]).lstrip("%")
                current = computations[name] = {"entry": words[0] == "ENTRY", "instructions": []}
            continue
        match = _HLO_INSTRUCTION.match(line) if current is not None else None
        if match is None:
            continue
        rest = line[match.end():]
        if rest.startswith("("):
            end = _closing(rest, 0)
            shape, rest = rest[:end + 1], rest[end + 1:].lstrip()
        else:
            shape, _, rest = rest.partition(" ")
        opcode, _, rest = rest.partition("(")
        end = _closing("(" + rest, 0) - 1
        operands = [part.split()[-1].lstrip("%") for part in _split_top_level(rest[:end])]
        current["instructions"].append({"root": bool(match.group(1)), "name": match.group(2), "shape": shape,
                                        "opcode": opcode.strip(), "operands": operands,
                                        "attributes": rest[end + 1:]})
    return computations


def _called(attributes: str, key: str) -> str | None:
    match = re.search(rf"\b{key}=%?([\w.\-]+)", attributes)
    return match.group(1) if match else None


def _dot_flops(computation: dict, computations: dict, seen=None) -> float:
    """2 x output elements x contraction length, summed over the dots of a (nested) computation."""
    seen = set() if seen is None else seen
    shapes = {instruction["name"]: instruction["shape"] for instruction in computation["instructions"]}
    flops = 0.0
    for instruction in computation["instructions"]:
        if instruction["opcode"] == "dot":
            dims = re.search(r"lhs_contracting_dims=\{([\d,]*)\}", instruction["attributes"])
            lhs = _hlo_arrays(shapes.get(instruction["operands"][0], ""))
            if dims and lhs:
                length = math.prod(lhs[0][1][int(d)] for d in dims.group(1).split(",") if d)
                flops += 2.0 * math.prod(_hlo_arrays(instruction["shape"])[0][1]) * length
        nested = _called(instruction["attributes"], "calls")
        if nested in computations and nested not in seen:
            seen.add(nested)
            flops += _dot_flops(computations[nested], computations, seen)
    return flops


def _fusion_bytes(instruction: dict, shapes: dict, fused: dict | None) -> float:
    """Operands read plus results written, counting operands used only through slices at the sliced
    size and in-place dynamic-update-slice buffers at the update's size."""
    if fused is None:
        return _hlo_bytes(instruction["shape"]) + sum(_hlo_bytes(shapes.get(name, "")) for name in instruction["operands"])
    inner = {item["name"]: item for item in fused["instructions"]}
    parameters = {int(item["operands"][0]): item["name"] for item in fused["instructions"]
                  if item["opcode"] == "parameter" and item["operands"] and item["operands"][0].isdigit()}
    users = {}
    for item in fused["instructions"]:
        for name in item["operands"]:
            users.setdefault(name, []).append(item)

    def through_bitcasts(item):
        while item is not None and item["opcode"] == "bitcast" and item["operands"]:
            item = inner.get(item["operands"][0])
        return item

    in_place, written = set(), 0.0
    roots = [item for item in fused["instructions"] if item["root"]]
    if roots and roots[0]["opcode"] == "tuple":
        roots = [inner[name] for name in roots[0]["operands"] if name in inner]
    for root in roots:
        update = through_bitcasts(root)
        if update is not None and update["opcode"] == "dynamic-update-slice" and len(update["operands"]) > 1:
            buffer = through_bitcasts(inner.get(update["operands"][0]))
            if buffer is not None:
                in_place.add(buffer["name"])
            written += _hlo_bytes(inner.get(update["operands"][1], {}).get("shape", ""))
        else:
            written += _hlo_bytes(root["shape"])
    read = 0.0
    for index, name in enumerate(instruction["operands"]):
        parameter = parameters.get(index)
        if parameter in in_place:
            continue
        uses = users.get(parameter, [])
        if uses and all(use["opcode"] in ("slice", "dynamic-slice") and use["operands"][0] == parameter for use in uses):
            read += min(sum(_hlo_bytes(use["shape"]) for use in uses), _hlo_bytes(shapes.get(name, "")))
        else:
            read += _hlo_bytes(shapes.get(name, ""))
    return read + written


def hlo_kernel_estimate(text: str) -> dict:
    """Per kernel class, launches, bytes moved (operands + results of each launch) and GEMM FLOPs
    of one execution of an optimized GPU HLO module, with known while-loop trip counts.

    An estimate of HBM traffic: caches, re-reads inside a kernel and launch-internal split-K
    passes are not modeled.
    """
    computations = hlo_computations(text)
    totals = {name: {"launches": 0, "bytes": 0.0, "flops": 0.0} for name in KERNEL_CLASSES}
    unresolved = []  # while loops counted once (unknown trip count), missing computations

    def walk(name, repeat):
        if name not in computations:
            unresolved.append(f"missing computation {name}")
            return
        computation = computations[name]
        shapes = {instruction["name"]: instruction["shape"] for instruction in computation["instructions"]}
        for instruction in computation["instructions"]:
            opcode, attributes = instruction["opcode"], instruction["attributes"]
            if opcode == "while":
                trips = re.search(r'"known_trip_count":\{"n":"(\d+)"', attributes)
                if trips is None:
                    unresolved.append(f"while {instruction['name']}: unknown trip count, counted once")
                count = int(trips.group(1)) if trips else 1
                walk(_called(attributes, "body"), repeat * count)
                walk(_called(attributes, "condition"), repeat * (count + 1))
                continue
            if opcode in ("call", "async-start"):
                target = _called(attributes, "to_apply") or _called(attributes, "calls")
                if target in computations:
                    walk(target, repeat)
                continue
            if opcode in _HLO_NOT_KERNELS or opcode.endswith("-done"):
                continue
            fused = computations.get(_called(attributes, "calls")) if opcode == "fusion" else None
            flops = 0.0
            if opcode == "custom-call":
                target = re.search(r'custom_call_target="([^"]+)"', attributes)
                target = target.group(1) if target else ""
                kind = ("fused_attention" if "cudnn$fmha" in target.lower() or "fmha" in target.lower()
                        else "gemm" if re.search(r"cublas|gemm|matmul", target, re.IGNORECASE) else "other")
                arrays = _hlo_arrays(instruction["shape"])
                result = arrays[0] if kind == "gemm" else None
                moved = sum(_hlo_bytes(shapes.get(name, "")) for name in instruction["operands"])
                moved += (_HLO_ITEM_BYTES.get(result[0], 1) * math.prod(result[1]) if result
                          else sum(_HLO_ITEM_BYTES.get(d, 1) * math.prod(s) for d, s in arrays if d != "s8" and d != "u8"))
                if kind == "gemm" and instruction["operands"]:
                    dims = re.search(r'"lhs_contracting_dimensions":\[([^\]]*)\]', attributes)
                    lhs = _hlo_arrays(shapes.get(instruction["operands"][0], ""))
                    if dims and lhs:
                        length = math.prod(lhs[0][1][int(d.strip('"'))] for d in dims.group(1).split(",") if d.strip('"'))
                        flops = 2.0 * math.prod(result[1]) * length
            else:
                kind = "gemm" if fused is not None and (
                    "gemm" in attributes or _dot_flops(fused, computations)) else "other"
                moved = _fusion_bytes(instruction, shapes, fused)
                if kind == "gemm":
                    flops = _dot_flops(fused, computations)
            totals[kind]["launches"] += repeat
            totals[kind]["bytes"] += repeat * moved
            totals[kind]["flops"] += repeat * flops

    entry = next(name for name, computation in computations.items() if computation["entry"])
    walk(entry, 1)
    return {"classes": totals, "unresolved": unresolved,
            "method": "per launch: operands read + results written at full size (slices and in-place "
                      "updates at their size); GEMM FLOPs 2 x output x contraction; while bodies x trip count"}


def profile(compiled, arguments, trace_dir: Path, *, calls: int = 3) -> dict:
    """GPU stream time of ``calls`` synchronized calls of an already compiled program, by kernel
    class over all kernels, with the optimized HLO's traffic and FLOP estimate per class."""
    from scaffolds.transformer.benchmarks.analyze_profile import analyze_file, discover_traces

    jax.block_until_ready(compiled(*arguments))
    with jax.profiler.trace(str(trace_dir), create_perfetto_trace=True):
        for index in range(calls):
            with jax.profiler.StepTraceAnnotation("attention", step_num=index + 1):
                jax.block_until_ready(compiled(*arguments))
    return {"calls": calls, **profile_summary(analyze_file(
        discover_traces([trace_dir])[-1], step_name="attention", top_kernels=1_000_000,
        max_bytes=400 * 1024 * 1024), compiled.as_text())}


def profile_summary(summary: dict, hlo_text: str) -> dict:
    """Per-call class times of an analyze_profile summary, with the HLO estimate and achieved rates."""
    device = summary["devices"][0]
    mean = device["mean_per_update"]
    names = {}
    for update in device["updates"]:
        for kernel in update["top_kernels"]:
            names[kernel["name"]] = names.get(kernel["name"], 0.0) + kernel["kernel_sum_ms"] / len(device["updates"])
    class_ms = {name: 0.0 for name in KERNEL_CLASSES}
    for name, duration in names.items():
        class_ms[kernel_class(name)] += duration
    estimate = hlo_kernel_estimate(hlo_text)
    achieved = {}
    for name in KERNEL_CLASSES:
        seconds = class_ms[name] / 1e3
        moved, flops = estimate["classes"][name]["bytes"], estimate["classes"][name]["flops"]
        achieved[name] = {"tb_per_s": moved / seconds / 1e12 if seconds else None,
                          "tflop_per_s": flops / seconds / 1e12 if seconds and flops else None}
    return {
        "wall_ms": mean["wall_ms"], "kernel_count": mean["kernel_count"],
        "kernel_sum_ms": mean["kernel_sum_ms"], "gpu_activity_union_ms": mean["gpu_activity_union_ms"],
        "no_recorded_gpu_activity_ms": mean["no_recorded_gpu_activity_ms"],
        "class_ms": class_ms, "hlo_estimate": estimate, "achieved": achieved,
        "top_kernels": sorted(({"name": name[:120], "ms": duration, "class": kernel_class(name)}
                               for name, duration in names.items()), key=lambda item: -item["ms"])[:12],
        "note": "per call, all kernels; class by kernel name (kernel_class); achieved = HLO estimate / class time",
    }


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


GRADIENT_LABELS = ("output", "dq", "dk", "dv")


def _errors(results, references, *, sequences=None) -> dict:
    """Relative L2 errors of (output, dq, dk, dv), optionally over the first ``sequences`` only."""
    return {label: relative_l2(np.asarray((x if sequences is None else x[:sequences]).astype(jnp.float32)), reference)
            for label, x, reference in zip(GRADIENT_LABELS, results, references, strict=True)}


def _rounding_floor(references) -> dict:
    return {label: relative_l2(np.asarray(jnp.asarray(x, jnp.bfloat16).astype(jnp.float32)), x)
            for label, x in zip(GRADIENT_LABELS, references, strict=True)}


def historical_blocked_kernels(shape, seed: int) -> dict:
    """Options of blocked_dot_product_attention for kernels that trained BF16 runs on the blocked
    backend (softmax for comparison): query coefficients (Taylor x (1 + 0.1 noise)) and a
    normalized learned polynomial."""
    rng = np.random.default_rng(seed)
    taylor = np.array([1.0, 1.0, 0.5, 1 / 6, 1 / 24], np.float32)
    coefficients = taylor * (1 + 0.1 * rng.standard_normal((*shape[:3], 5))).astype(np.float32)
    return {
        "softmax": {},
        "query polynomial m=4, unnormalized": {
            "attention_degree": 4, "attention_normalization": False,
            "query_polynomial_coefficients": jnp.asarray(coefficients)},
        "normalized learned polynomial m=2": {
            "attention_degree": 2, "polynomial_coefficients": jnp.asarray([1.0, 1.0, 0.5], jnp.float32)},
    }


def block_sum_check(shape, seed: int, block_sizes=(128, 2048), compile_options=None) -> dict:
    """BF16 production blocked attention at several query-block sizes against an FP32 reference
    of the same kernel (HIGHEST matmul precision, one block) on the same BF16 values."""
    from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention

    inputs = _bf16_inputs(shape, seed, False)
    upcast = tuple(x.astype(jnp.float32) for x in inputs)
    report = {"shape": list(shape), "reference": "same kernel, FP32-upcast inputs, one query block, "
              "jax.default_matmul_precision('highest')", "kernels": {}}
    for name, options in historical_blocked_kernels(shape, seed + 1).items():
        def attend(q, k, v, size, options=options):
            return blocked_dot_product_attention(q, k, v, query_block_size=size, **options)

        with jax.default_matmul_precision("highest"):
            program = forward_backward(functools.partial(attend, size=shape[1]))
            output, grads = jax.jit(program).lower(*upcast).compile(compile_options)(*upcast)
        reference = tuple(np.asarray(x) for x in (output, *grads))
        entry = {"bf16_rounding_floor": _rounding_floor(reference), "blocks": {}}
        for size in block_sizes:
            program = forward_backward(functools.partial(attend, size=size))
            output, grads = jax.jit(program).lower(*inputs).compile(compile_options)(*inputs)
            entry["blocks"][str(size)] = _errors((output, *grads), reference)
        report["kernels"][name] = entry
        jax.clear_caches()
    return report


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
                                "backward": "jax.vjp of the attention; outputs, dq, dk, dv",
                                "accuracy_compile": "accuracy and failure programs: accuracy shape, "
                                                    "xla_gpu_autotune_level=0; timed programs: timing shape, "
                                                    "XLA defaults, scored on their first sequence",
                                "timed_accuracy_sequences": TIMED_ACCURACY_SEQUENCES},
              "matmul_precision": matmul_precision_probe(), "variants": {}}
    write_json(args.output, report)
    deadline = time.monotonic() + 60 * args.budget_minutes

    accuracy_inputs = {name: _bf16_inputs(args.accuracy_shape, args.seed, name == "sharp") for name in ("normal", "sharp")}
    oracles, floors = {}, {}
    for name, (q, k, v, g) in accuracy_inputs.items():
        oracles[name] = oracle(*(np.asarray(x.astype(jnp.float32)) for x in (q, k, v, g)))
        floors[name] = _rounding_floor(oracles[name])
    report["bf16_rounding_floor"] = floors
    float32_inputs = tuple(x.astype(jnp.float32) for x in accuracy_inputs["normal"])
    float32_oracle = oracle(*(np.asarray(x) for x in float32_inputs))
    cases = failure_cases(*(np.asarray(x) for x in accuracy_inputs["normal"][:3]), position=args.failure_position)
    time_inputs = _bf16_inputs(args.time_shape, args.seed + 1, False)
    # The timed (default-compiled) programs are scored too, on their first sequence(s): attention is
    # independent per sequence, so the oracle of that slice is exact for it.
    timed_oracle = oracle(*(np.asarray(x[:TIMED_ACCURACY_SEQUENCES].astype(jnp.float32)) for x in time_inputs))
    report["bf16_rounding_floor_timed"] = _rounding_floor(timed_oracle)

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
            # Accuracy does not need tuned GEMM tiles; autotuning dominates compile time.
            untuned = {"xla_gpu_autotune_level": "0"}
            compiled = jax.jit(program).lower(*accuracy_inputs["normal"]).compile(untuned)
            entry["accuracy"] = {}
            for distribution, inputs in accuracy_inputs.items():
                output, grads = compiled(*inputs)
                entry["accuracy"][distribution] = _errors((output, *grads), oracles[distribution])
            entry["failure"] = failure_report(compiled, cases, accuracy_inputs["normal"][3])
            del compiled
            if name in (args.float32_input_variants or ()):
                output, grads = jax.jit(program).lower(*float32_inputs).compile(untuned)(*float32_inputs)
                entry["accuracy"]["float32_inputs"] = _errors((output, *grads), float32_oracle)
            forward, result, _ = measure(attend, time_inputs[:3], warmup=args.warmup, iterations=args.iterations)
            del result
            both, result, compiled = measure(program, time_inputs, warmup=args.warmup, iterations=args.iterations)
            finite = bool(jax.device_get(jnp.all(jnp.stack(
                [jnp.all(jnp.isfinite(x.astype(jnp.float32))) for x in jax.tree.leaves(result)]))))
            timed_accuracy = _errors((result[0], *result[1]), timed_oracle, sequences=TIMED_ACCURACY_SEQUENCES)
            del result
            entry["timing"] = {"forward": forward, "forward_backward": both, "outputs_finite": finite,
                               "accuracy_first_sequences": timed_accuracy,
                               "attention_seconds_per_update": both["median_ms"] * UPDATE_LAYER_MICRO_BATCHES / 1e3}
            if name in (args.profile_variants or ()):
                try:
                    trace_dir = args.output.parent / f"traces-{os.environ.get('SLURM_JOB_ID', os.getpid())}" / name
                    entry["profile"] = profile(compiled, time_inputs, trace_dir)
                except Exception as error:  # noqa: BLE001
                    entry["profile"] = {"error": f"{type(error).__name__}: {str(error)[:500]}"}
            del compiled
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
    if not args.skip_block_sums:
        # Production's BF16 dK/dV block sums for kernels that trained historical blocked runs.
        try:
            report["block_sums"] = block_sum_check(args.accuracy_shape, args.seed + 2,
                                                   compile_options={"xla_gpu_autotune_level": "0"})
        except Exception as error:  # noqa: BLE001
            report["block_sums"] = {"error": f"{type(error).__name__}: {str(error)[:2000]}"}
    report["status"] = "complete"
    report["completed_at_utc"] = datetime.now(UTC).isoformat()
    write_json(args.output, report)
    return report


class _RoutedSoftmax:
    """Stands in for attention.softmax.softmax_attention: sends attention to ``attend`` and counts
    the calls in ``calls`` (one per layer each time the model is traced)."""

    def __init__(self, attend):
        self.attend = attend
        self.calls = 0

    def __call__(self, q, k, v, *, backend="blocked", return_diagnostics=False, dropout=None,
                 deterministic=True, rngs=None, weights_dtype=None):
        if return_diagnostics:
            raise ValueError("Benchmark routing computes no attention diagnostics")
        if dropout is not None and getattr(dropout, "rate", 0.0) != 0.0:
            raise ValueError("Benchmark routing requires dropout 0")
        self.calls += 1
        return self.attend(q, k, v)


def _routed_softmax(attend) -> _RoutedSoftmax:
    return _RoutedSoftmax(attend)


def load_model_config(recipe: Path, *, micro_batch: int, backend: str):
    from scaffolds.transformer.configs import load_config

    config = load_config(recipe)
    config.data.source = "synthetic"
    config.data.synthetic_vocab_size = config.model.vocab_size
    config.train.micro_batch_size = micro_batch
    config.model.attention_backend = backend
    return config


def micro_step_program(config, *, variant: str | None):
    """(state, jit-able f(params, batch, rng) -> (nll, grads), router) running the scaffold's loss_and_grads.

    ``variant`` None keeps the configured backend (router None); otherwise softmax attention is
    routed to it for the rest of the process, and ``router.calls`` counts the routed calls.
    """
    from types import SimpleNamespace

    import scaffolds.transformer.attention.softmax as softmax_module
    from scaffolds.transformer.training import loop
    from scaffolds.transformer.training.state import create_state

    router = None
    if variant is not None:
        router = _routed_softmax(variants(config.model.max_seq_len)[variant].function())
        softmax_module.softmax_attention = router
    state = create_state(config, jax.random.PRNGKey(config.train.seed))

    def program(params, batch, rng):
        return loop.loss_and_grads(SimpleNamespace(graphdef=state.graphdef, params=params), batch, rng)

    return state, program, router


def model_case(args) -> dict:
    """One (variant, micro-batch) forward+backward micro-step; run in a fresh process."""
    started = time.perf_counter()
    result = {"variant": args.variant, "micro_batch": args.micro_batch, "status": "running",
              "provenance": provenance()}
    try:
        routed = args.variant not in ("production-b128", "cudnn")
        backend = "cudnn" if args.variant == "cudnn" else "blocked"
        config = load_model_config(args.recipe, micro_batch=args.micro_batch, backend=backend)
        state, program, router = micro_step_program(config, variant=args.variant if routed else None)
        result["routing"] = {"attention_backend": backend, "routed_to": args.variant if routed else None,
                             "layers": config.model.num_layers, "routed_calls_traced": None}
        batch = jnp.asarray(np.random.default_rng(config.train.seed).integers(
            0, config.model.vocab_size, (args.micro_batch, config.data.sequence_length), dtype=np.int32))
        rng = jax.random.PRNGKey(config.train.seed)
        if router is not None:
            router.calls = 0  # count the measured program's single trace only
        timing, (nll, grads), _ = measure(program, (state.params, batch, rng), warmup=args.warmup,
                                          iterations=args.iterations)
        if router is not None:
            # Observed, not configured: each layer's attention passed through the routed candidate.
            result["routing"]["routed_calls_traced"] = router.calls
            if router.calls != config.model.num_layers:
                raise RuntimeError(f"Routing traced {router.calls} attention calls for "
                                   f"{config.model.num_layers} layers")
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
            try:
                completed = subprocess.run(command, capture_output=True, text=True, timeout=args.case_timeout)
                returncode, stderr = completed.returncode, completed.stderr
            except subprocess.TimeoutExpired as error:
                returncode, stderr = "timeout", (error.stderr or b"").decode(errors="replace") \
                    if isinstance(error.stderr, bytes) else (error.stderr or "")
            if path.exists():
                case = json.loads(path.read_text())
                path.unlink()
            else:
                case = {"variant": variant, "micro_batch": micro_batch, "status": "crashed",
                        "returncode": returncode, "stderr_tail": stderr[-3000:]}
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


def _rows_text(ranges, length):
    if not ranges:
        return "none"
    parts = []
    for start, end in ranges:
        parts.append(str(start) if end - start == 1 else "all" if (start, end) == (0, length)
                     else f">= {start}" if end == length else f"{start}-{end - 1}")
    return ", ".join(parts)


def summarize(kernel_path: Path | None, model_path: Path | None) -> str:
    """Markdown tables of the kernel and model records, in their variant order."""
    lines = []
    if kernel_path is not None:
        kernel = json.loads(kernel_path.read_text())
        entries = kernel["variants"]
        length = kernel["configuration"]["accuracy_shape"][1]
        lines += ["| Variant | Forward ms | Fwd+bwd ms | Temp GB | Attention s/update | Kernels |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for name, entry in entries.items():
            timing = entry.get("timing")
            if timing is None:
                lines.append(f"| {name} | {entry.get('skipped') or entry.get('error', '')[:60]} | | | | |")
                continue
            both = timing["forward_backward"]
            lines.append(f"| {name} | {timing['forward']['median_ms']:.2f} | {both['median_ms']:.2f} | "
                         f"{both['compiled_memory'].get('temp_size_in_bytes', 0) / 1e9:.2f} | "
                         f"{timing['attention_seconds_per_update']:.2f} | {both['kernels']} |")
        lines += ["", "Relative L2 error x 1e3 against the FP64 oracle (normal; sharp; FP32 inputs where measured):", "",
                  "| Variant | Output | dQ | dK | dV | Sharp output | Sharp dQ | Sharp dK | Sharp dV |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]

        def errors(record):
            return [f"{1e3 * record[label]:.2f}" for label in ("output", "dq", "dk", "dv")]

        floor = kernel["bf16_rounding_floor"]
        lines.append("| BF16 rounding of the oracle | " + " | ".join(errors(floor["normal"]) + errors(floor["sharp"])) + " |")
        float32_rows = []
        for name, entry in entries.items():
            accuracy = entry.get("accuracy")
            if accuracy:
                lines.append(f"| {name} | " + " | ".join(errors(accuracy["normal"]) + errors(accuracy["sharp"])) + " |")
                if "float32_inputs" in accuracy:
                    float32_rows.append(f"| {name} | " + " | ".join(
                        f"{1e6 * accuracy['float32_inputs'][label]:.2f}" for label in ("output", "dq", "dk", "dv")) + " |")
        if float32_rows:
            lines += ["", "FP32 inputs, relative L2 error x 1e6:", "", "| Variant | Output | dQ | dK | dV |",
                      "| --- | ---: | ---: | ---: | ---: |", *float32_rows]
        timed_rows = [f"| {name} | " + " | ".join(errors(entry["timing"]["accuracy_first_sequences"])) + " |"
                      for name, entry in entries.items() if "accuracy_first_sequences" in entry.get("timing", {})]
        if timed_rows:
            sequences = kernel["configuration"].get("timed_accuracy_sequences", 1)
            lines += ["", f"Timed programs (XLA defaults, timing shape), first {sequences} sequence(s), relative L2 "
                          "error x 1e3:", "", "| Variant | Output | dQ | dK | dV |", "| --- | ---: | ---: | ---: | ---: |",
                      "| BF16 rounding of the oracle | " + " | ".join(errors(kernel["bf16_rounding_floor_timed"])) + " |",
                      *timed_rows]
        cases = ("q_nan", "k_nan", "v_nan", "v_inf", "score_neg_inf", "score_overflow", "score_huge")
        lines += ["", f"Nonfinite output rows (batch 0, head 1; injected at token "
                      f"{kernel['configuration']['failure_position']}); gradients finite in brackets:", "",
                  "| Variants | " + " | ".join(cases) + " |", "| --- |" + " --- |" * len(cases)]
        groups = {}
        for name, entry in entries.items():
            failure = entry.get("failure")
            if failure:
                key = tuple(_rows_text(failure[case]["nonfinite_rows"], length)
                            + ("" if failure[case]["other_heads_finite"] else " (other heads!)")
                            + (" [finite]" if failure[case]["gradients_finite"] else "") for case in cases)
                groups.setdefault(key, []).append(name)
        expected = next(iter(entries.values()), {}).get("failure")
        if expected:
            lines.append("| production rule (causal) | " + " | ".join(
                _rows_text(expected[case]["expected_rows"], length) for case in cases) + " |")
        for key, names in groups.items():
            lines.append(f"| {', '.join(names)} | " + " | ".join(key) + " |")
        profiled = [(name, entry["profile"]) for name, entry in entries.items()
                    if "profile" in entry and "error" not in entry["profile"]]
        if profiled:
            def rate(value):
                return "" if value is None else f"{value:.2f}"

            lines += ["", "Profiled forward+backward calls (GPU stream time per call, all kernels by class; "
                          "HLO-estimated traffic and GEMM FLOPs over class time):", "",
                      "| Variant | Wall ms | GPU busy ms | Idle ms | Kernels | GEMM ms | Other ms | cuDNN ms | "
                      "GEMM GB | Other GB | GEMM TB/s | GEMM TFLOP/s | Other TB/s |",
                      "| --- |" + " ---: |" * 12]
            for name, record in profiled:
                ms, estimate, achieved = record["class_ms"], record["hlo_estimate"]["classes"], record["achieved"]
                lines.append(f"| {name} | {record['wall_ms']:.2f} | {record['gpu_activity_union_ms']:.2f} | "
                             f"{record['no_recorded_gpu_activity_ms']:.2f} | {record['kernel_count']:.0f} | "
                             f"{ms['gemm']:.2f} | {ms['other']:.2f} | {ms['fused_attention']:.2f} | "
                             f"{estimate['gemm']['bytes'] / 1e9:.1f} | {estimate['other']['bytes'] / 1e9:.1f} | "
                             f"{rate(achieved['gemm']['tb_per_s'])} | {rate(achieved['gemm']['tflop_per_s'])} | "
                             f"{rate(achieved['other']['tb_per_s'])} |")
        repeats = kernel.get("repeats", {})
        if repeats:
            lines += ["", "Repeated at the end of the job (fwd+bwd ms): " + ", ".join(
                f"{name} {value['forward_backward_median_ms']:.2f} (first {entries[name]['timing']['forward_backward']['median_ms']:.2f})"
                for name, value in repeats.items())]
        block_sums = kernel.get("block_sums", {})
        if "kernels" in block_sums:
            sizes = list(next(iter(block_sums["kernels"].values()))["blocks"])
            lines += ["", f"Production blocked BF16 at {block_sums['shape']} against an FP32 reference of the same "
                          "kernel: dK and dV error x 1e3 (ratio to the BF16 rounding floor) per query-block size:", "",
                      "| Kernel | Floor dK / dV | " + " | ".join(f"{size}-query blocks dK / dV" for size in sizes) + " |",
                      "| --- | ---: |" + " ---: |" * len(sizes)]
            for name, record in block_sums["kernels"].items():
                floor = record["bf16_rounding_floor"]
                cells = [f"{1e3 * record['blocks'][size]['dk']:.2f} ({record['blocks'][size]['dk'] / floor['dk']:.2f}x) / "
                         f"{1e3 * record['blocks'][size]['dv']:.2f} ({record['blocks'][size]['dv'] / floor['dv']:.2f}x)"
                         for size in sizes]
                lines.append(f"| {name} | {1e3 * floor['dk']:.2f} / {1e3 * floor['dv']:.2f} | " + " | ".join(cells) + " |")
        elif block_sums:
            lines += ["", f"Block sums: {block_sums.get('error', '')[:200]}"]
    if model_path is not None:
        model = json.loads(model_path.read_text())
        lines += ["", "| Variant | Micro-batch | Micro-step ms | s/update | Peak GB | Compile s | NLL | Routed calls |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        steps, layers = {}, None
        for case in model["cases"]:
            routing = case.get("routing") if isinstance(case.get("routing"), dict) else {}
            calls = routing.get("routed_calls_traced")
            layers = routing.get("layers", layers)
            if case.get("status") != "complete":
                reason = "out of memory" if case.get("out_of_memory") else case.get("status")
                lines.append(f"| {case['variant']} | {case['micro_batch']} | {reason} | | | | | |")
                continue
            steps[(case["variant"], case["micro_batch"])] = case["timing"]["median_ms"]
            peak = case.get("peak_bytes_in_use")
            lines.append(f"| {case['variant']} | {case['micro_batch']} | {case['timing']['median_ms']:.1f} | "
                         f"{case['seconds_per_update']:.2f} | {'' if peak is None else f'{peak / 1e9:.1f}'} | "
                         f"{case['timing']['compile_seconds']:.0f} | {case['nll']:.4f} | "
                         f"{'-' if calls is None else calls} |")
        if kernel_path is not None and layers:
            # A routed model's micro-step differs from the cuDNN model's by its attention only, one call per
            # layer at the kernel stage's batch: compare with layers x the one-layer difference.
            batch = kernel["configuration"]["time_shape"][0]
            base_step = steps.get(("cudnn", batch))
            base_layer = entries.get("cudnn", {}).get("timing", {}).get("forward_backward", {}).get("median_ms")
            rows = []
            for (variant, micro_batch), step in steps.items():
                layer = entries.get(variant, {}).get("timing", {}).get("forward_backward", {}).get("median_ms")
                if micro_batch == batch and variant != "cudnn" and None not in (base_step, base_layer, layer):
                    rows.append(f"| {variant} | {step - base_step:.1f} | {layers * (layer - base_layer):.1f} |")
            if rows:
                lines += ["", f"Micro-step minus cuDNN's at micro-batch {batch} against {layers} x the one-layer "
                              "forward+backward difference (ms):", "", "| Variant | Model | Layers x kernel |",
                          "| --- | ---: | ---: |", *rows]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("oracle-check", "kernel", "model", "model-case", "summarize"))
    parser.add_argument("--kernel-record", type=Path, help="summarize: a kernel-stage JSON record")
    parser.add_argument("--model-record", type=Path, help="summarize: a model-stage JSON record")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--variant")
    parser.add_argument("--accuracy-shape", type=int, nargs=4, default=[2, 2048, 4, 64])
    parser.add_argument("--time-shape", type=int, nargs=4, default=[16, 2048, 14, 64])
    parser.add_argument("--failure-position", type=int, default=1000)
    parser.add_argument("--profile-variants", nargs="+",
                        help="Trace three forward+backward calls of these variants (GPU stream time breakdown)")
    parser.add_argument("--float32-input-variants", nargs="+",
                        help="Also measure these variants' accuracy with FP32 inputs (precision of the products)")
    parser.add_argument("--skip-block-sums", action="store_true",
                        help="Skip the BF16 block-sum check of the historical blocked kernels")
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
    elif args.stage == "summarize":
        args.output.write_text(summarize(args.kernel_record, args.model_record))
    else:
        if args.variant is None or args.micro_batch is None:
            parser.error("model-case requires --variant and --micro-batch")
        model_case(args)


if __name__ == "__main__":
    main()
