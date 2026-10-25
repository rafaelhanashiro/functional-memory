"""Diagnose reduced head-scale gradients independently of mixed-precision kernels."""
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.attention.output_scaling import scale_attention_output
from scaffolds.transformer.attention.positional import apply_rope
from scaffolds.transformer.attention.qk_norm import rms_score_operands, scale_rms_queries
from scaffolds.transformer.tests.test_qk_rms_attention_backends import _inputs, _operation

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


def raw_reference(family, q, k, v):
    """Independent dense causal exp/Taylor contraction, preserving its input precision."""
    score = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(jnp.asarray(q.shape[-1], q.dtype))
    causal = jnp.arange(q.shape[1])[:, None] >= jnp.arange(k.shape[1])[None, :]
    if family == "exp":
        weight = jnp.exp(jnp.where(causal, score, -jnp.inf))
    else:
        weight = sum(score ** r / math.factorial(r) for r in range(5))
        weight = jnp.where(causal, weight, 0)
    return jnp.einsum("bhts,bshd->bthd", weight, v)


def smooth_reference(family, *operands):
    """Independent continuous math, with no BF16 casts or implementation helpers."""
    q, k, v, qgain, kgain, gamma, output_gain = operands
    q = q / jnp.sqrt(jnp.mean(q * q, axis=-1, keepdims=True) + 1e-6) * qgain
    k = k / jnp.sqrt(jnp.mean(k * k, axis=-1, keepdims=True) + 1e-6) * kgain
    q = q * jnp.exp(gamma)[None, None]
    dim = q.shape[-1]
    frequency = jnp.arange(q.shape[1], dtype=q.dtype)[:, None] / (
        10000 ** (2 * jnp.arange(dim // 2, dtype=q.dtype) / dim))[None]
    cos, sin = jnp.cos(frequency)[None, :, None], jnp.sin(frequency)[None, :, None]

    def rotate(x):
        first, second = jnp.split(x, 2, axis=-1)
        return jnp.concatenate((first * cos - second * sin, second * cos + first * sin), axis=-1)

    y = raw_reference(family, rotate(q), rotate(k), v)
    return y / jnp.sqrt(jnp.mean(y * y, axis=-1, keepdims=True) + 1e-6) * output_gain


def vjp(operation, operands, cotangent):
    output, pullback = jax.vjp(operation, *operands)
    return output, pullback(cotangent)


def host(tree):
    return jax.tree.map(lambda x: np.asarray(x, np.float64), tree)


def relative_per_head(actual, expected):
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    if actual.ndim == 4:
        actual, expected = np.moveaxis(actual, 2, 0), np.moveaxis(expected, 2, 0)
    return [float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)) for a, b in zip(actual, expected)]


def summarize(actual, reference):
    return {name: relative_per_head(a, b) for name, a, b in zip(
        ("q", "k", "v", "q_gain", "k_gain", "head_log_scale", "output_gain"), actual, reference)}


def prepared(operands):
    q, k, v, qgain, kgain, gamma, output_gain = operands
    qn, kn = rms_score_operands(q, k, qgain, kgain, eps=1e-6)
    scaled = scale_rms_queries(qn, gamma)
    rotate = lambda x: apply_rope(x, jnp.arange(x.shape[1])).astype(jnp.bfloat16)
    return qn, scaled, rotate(scaled), rotate(kn), v, output_gain, rotate


def gamma_contraction(qn, scaled, rotate, dq, gamma):
    dscaled = jax.vjp(rotate, scaled)[1](dq.astype(jnp.bfloat16))[0]
    terms = dscaled.astype(jnp.float64) * qn.astype(jnp.float64) * jnp.exp(gamma.astype(jnp.float64))[None, None]
    signed = jnp.sum(terms, axis=(0, 1, 3))
    magnitude = jnp.sum(jnp.abs(terms), axis=(0, 1, 3))
    return {"gamma": np.asarray(signed).tolist(), "absolute_contribution_sum": np.asarray(magnitude).tolist(),
            "cancellation_ratio": np.asarray(magnitude / jnp.maximum(jnp.abs(signed), 1e-30)).tolist()}


def main():
    destination = Path(sys.argv[1])
    report = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "devices": [d.device_kind for d in jax.devices()],
        "families": {},
    }

    def save():
        destination.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")

    def message(text):
        print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), text, flush=True)

    for family, degree, backend in (("exp", 0, "fused"), ("fixed", 4, "mosaic")):
        message(f"Starting {family} full-chain references")
        operands, cotangent = _inputs(family, degree, head_scale=True)
        result = report["families"][family] = {}
        values = {}
        for precision in (jnp.float32, jnp.float64):
            arrays = tuple(x.astype(precision) for x in operands)
            values[str(precision)] = jax.jit(partial(vjp, partial(smooth_reference, family)))(arrays, cotangent.astype(precision))
            jax.block_until_ready(values[str(precision)])
        ref32, ref64 = values[str(jnp.float32)], values[str(jnp.float64)]
        result["smooth_fp32_vs_fp64"] = summarize(host(ref32[1]), host(ref64[1]))
        result["smooth_fp64_gamma"] = host(ref64[1][5]).reshape(-1).tolist()
        measured = {}
        for chosen in ("blocked", backend):
            operation = partial(_operation, family, degree, chosen, head_scale=True)
            measured[chosen] = jax.jit(partial(vjp, operation))(operands, cotangent)
            jax.block_until_ready(measured[chosen])
            result[f"{chosen}_gamma"] = host(measured[chosen][1][5]).reshape(-1).tolist()
            result[f"{chosen}_vs_smooth_fp64"] = summarize(host(measured[chosen][1]), host(ref64[1]))
        result["backend_vs_blocked"] = summarize(host(measured[backend][1]), host(measured["blocked"][1]))
        save()
        message(f"{family}: full-chain references complete; isolating kernel VJPs")

        qn, scaled, q, k, v, gain, rotate = prepared(operands)
        kernel = lambda name, q, k, v: attention_core(q, k, v, backend=name,
            attention_degree=degree or None, attention_normalization=False)
        kernel_outputs = {name: jax.jit(partial(kernel, name))(q, k, v) for name in ("blocked", backend)}
        # One exact same upstream gradient for all local VJPs, including the dense oracle.
        doutput = jax.vjp(lambda y: scale_attention_output(y, "head_rms_norm", output_norm_scale=gain),
                          kernel_outputs["blocked"])[1](cotangent)[0]
        local = {}
        for name in ("blocked", backend):
            local[name] = jax.jit(lambda q, k, v, dy: jax.vjp(partial(kernel, name), q, k, v)[1](dy))(q, k, v, doutput)
            jax.block_until_ready(local[name])
        exact_operands = tuple(x.astype(jnp.float64) for x in (q, k, v))
        local64 = jax.jit(lambda q, k, v, dy: jax.vjp(partial(raw_reference, family), q, k, v)[1](dy))(
            *exact_operands, doutput.astype(jnp.float64))
        jax.block_until_ready(local64)
        result["local_common_bf16_upstream"] = {
            name: {key: relative_per_head(a, b) for key, a, b in zip("qkv", host(local[name]), host(local64))}
            for name in local}
        result["local_gamma_contraction"] = {
            name: gamma_contraction(qn, scaled, rotate, value[0], operands[5])
            for name, value in {**local, "dense_fp64_rounded_dq": local64}.items()}
        save()

        # A smooth FP64 reference has a well-defined derivative; finite differences
        # of the quantized BF16 graph would instead measure quantization jumps.
        message(f"{family}: testing a tangent-aligned, nonvacuous scalar cotangent")
        arrays32 = tuple(x.astype(jnp.float32) for x in operands)
        tangent = tuple(jnp.ones_like(x) if index == 5 else jnp.zeros_like(x)
                        for index, x in enumerate(arrays32))
        aligned = jax.jit(lambda xs, ts: jax.jvp(partial(smooth_reference, family), xs, ts)[1])(
            arrays32, tangent).astype(jnp.bfloat16)
        aligned_values = {}
        for name in ("blocked", backend):
            operation = partial(_operation, family, degree, name, head_scale=True)
            aligned_values[name] = jax.jit(partial(vjp, operation))(operands, aligned)
            jax.block_until_ready(aligned_values[name])
        aligned64 = jax.jit(partial(vjp, partial(smooth_reference, family)))(
            tuple(x.astype(jnp.float64) for x in operands), aligned.astype(jnp.float64))
        jax.block_until_ready(aligned64)
        result["aligned_backend_vs_blocked"] = summarize(host(aligned_values[backend][1]), host(aligned_values["blocked"][1]))
        result["aligned_backend_vs_smooth_fp64"] = summarize(host(aligned_values[backend][1]), host(aligned64[1]))
        result["aligned_gamma"] = {name: host(value[1][5]).reshape(-1).tolist()
                                    for name, value in {**aligned_values, "smooth_fp64": aligned64}.items()}
        save()
        message(f"Finished {family}; random gamma max error={max(result['backend_vs_blocked']['head_log_scale']):.6g}, "
                f"aligned max error={max(result['aligned_backend_vs_blocked']['head_log_scale']):.6g}")
        del values, measured, aligned_values, aligned64, local, local64
        jax.clear_caches()
    message(f"Report saved to {destination}")


if __name__ == "__main__":
    main()
