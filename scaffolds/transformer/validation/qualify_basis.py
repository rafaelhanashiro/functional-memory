"""Check frozen RBF attention against analytic FP64 derivatives and time its kernel.

This is a scaffold qualification using synthetic inputs, not a language-model run.
It never opens a dataset or changes a run's scientific recipe or execution record.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from scaffolds.transformer.attention.basis import projection_from_config
from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.configs import load_config


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_record(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def oracle(q, k, v, cotangent, spec, normalized):
    """Independent NumPy FP64 forward and analytic VJP, including the causal mask."""
    q, k, v, cotangent = (np.asarray(x, dtype=np.float64) for x in (q, k, v, cotangent))
    scores = np.einsum('bthd,bshd->bhts', q, k) / math.sqrt(q.shape[-1])
    kernel, derivative = np.zeros_like(scores), np.zeros_like(scores)
    for function, coefficient in zip(spec['basis'], spec['coefficients'], strict=True):
        delta = scores - function['center']
        term = coefficient * np.exp(-0.5 * (delta / function['width'])**2)
        kernel += term
        derivative -= term * delta / function['width']**2
    causal = np.tril(np.ones(scores.shape[-2:], dtype=bool))
    kernel = np.where(causal, kernel, 0)
    derivative = np.where(causal, derivative, 0)
    normalizer = kernel.sum(axis=-1, keepdims=True)
    weights = kernel / normalizer if normalized else kernel
    output = np.einsum('bhts,bshv->bthv', weights, v)
    grad_weights = np.einsum('bthv,bshv->bhts', cotangent, v)
    if normalized:
        grad_weights = (grad_weights - np.einsum('bthv,bthv->bht', cotangent, output)[..., None]) / normalizer
    grad_scores = grad_weights * derivative
    grad_q = np.einsum('bhts,bshd->bthd', grad_scores, k) / math.sqrt(q.shape[-1])
    grad_k = np.einsum('bhts,bthd->bshd', grad_scores, q) / math.sqrt(q.shape[-1])
    grad_v = np.einsum('bhts,bthv->bshv', weights, cotangent)
    return output, grad_q, grad_k, grad_v


def operands(config, length, heads, dtype):
    rng = np.random.default_rng(27)
    q, k = (rng.normal(size=(1, length, heads, config.model.qk_head_dim)) for _ in range(2))
    if config.model.qk_norm:
        alpha = config.model.qk_norm_initial_scale or math.sqrt(config.model.qk_head_dim)
        q = q / np.sqrt((q*q).sum(-1, keepdims=True) + config.model.qk_norm_eps)
        q *= alpha * math.sqrt(config.model.qk_head_dim)
        k = k / np.sqrt((k*k).sum(-1, keepdims=True) + config.model.qk_norm_eps)
    v, cotangent = (rng.normal(size=(1, length, heads, config.model.value_head_dim)) for _ in range(2))
    return tuple(jnp.asarray(x, dtype=dtype) for x in (q, k, v, cotangent))


def operation(config):
    fit = projection_from_config(config.model.attention_basis)

    def forward(q, k, v):
        return attention_core(q, k, v, backend='blocked', score_kernel=fit,
                              attention_normalization=config.model.attention_normalization)

    def joint(q, k, v, cotangent):
        output, pullback = jax.vjp(forward, q, k, v)
        return (output, *pullback(cotangent))

    return jax.jit(forward), jax.jit(joint)


def check(config, dtype):
    forward, joint = operation(config)
    # Odd length exercises padding and causal block edges.
    arrays = operands(config, 17, 2, dtype)
    actual = jax.block_until_ready(joint(*arrays))
    expected = oracle(*arrays, config.model.attention_basis, config.model.attention_normalization)
    tolerance = 1e-3 if dtype == jnp.float32 else 4e-2
    errors = {}
    for name, observed, reference in zip(('output', 'dq', 'dk', 'dv'), actual, expected, strict=True):
        observed = np.asarray(observed, dtype=np.float64)
        error = float(np.linalg.norm(observed-reference) / max(np.linalg.norm(reference), 1e-12))
        if not np.isfinite(observed).all() or not np.isfinite(error) or error > tolerance:
            raise AssertionError(f'{dtype} {name} relative error {error} exceeds {tolerance}')
        errors[name] = error
    altered = arrays[2].at[:, -1].add(10)
    future_changed = forward(*arrays[:2], altered)
    np.testing.assert_array_equal(np.asarray(future_changed[:, :-1]), np.asarray(actual[0][:, :-1]))
    return {'dtype': str(jnp.dtype(dtype)), 'relative_errors': errors,
            'relative_tolerance': tolerance, 'causality': 'passed', 'length': 17, 'heads': 2}


def benchmark(config, repeats):
    _, joint = operation(config)
    arrays = operands(config, config.model.max_seq_len, config.model.num_heads, jnp.bfloat16)
    start = time.perf_counter()
    output = jax.block_until_ready(joint(*arrays))
    compile_and_first_seconds = time.perf_counter() - start
    if any(not np.isfinite(np.asarray(x, dtype=np.float32)).all() for x in output):
        raise AssertionError('Nonfinite output or gradient at production attention shape')
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = jax.block_until_ready(joint(*arrays))
        durations.append(time.perf_counter() - start)
    return {'dtype': 'bfloat16', 'batch': 1, 'length': config.model.max_seq_len,
            'heads': config.model.num_heads, 'qk_head_dim': config.model.qk_head_dim,
            'value_head_dim': config.model.value_head_dim,
            'compile_and_first_seconds': compile_and_first_seconds,
            'forward_backward_seconds': durations, 'median_seconds': statistics.median(durations),
            'scope': 'one attention layer, forward plus Q/K/V backward; excludes model and optimizer'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs', nargs='+', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--platform', choices=('gpu', 'cpu'), default='gpu')
    parser.add_argument('--skip-benchmark', action='store_true')
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    device = jax.devices()[0]
    if (args.platform == 'gpu') != (device.platform == 'gpu'):
        raise RuntimeError(f'Requested {args.platform}, but found {device}')
    record = {
        'format': 'shared-transformer-rbf-qualification', 'version': 1, 'status': 'running',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'source_commit': os.environ.get('SHARED_TRANSFORMER_SOURCE_COMMIT') or subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'module_sha256': sha256(__file__), 'dependency_lock_sha256': sha256('scaffolds/transformer/uv.lock'),
        'home_cluster': os.environ.get('TRANSFORMER_CLUSTER'),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'slurm_node': os.environ.get('SLURMD_NODENAME'),
        'device_kind': device.device_kind, 'platform': device.platform, 'jax_version': jax.__version__,
        'runtime': {key: os.environ.get(key) for key in ('TRANSFORMER_CUDA_ENVIRONMENT',
                    'XLA_FLAGS', 'XLA_PYTHON_CLIENT_PREALLOCATE', 'XLA_PYTHON_CLIENT_MEM_FRACTION',
                    'CUDA_HOME', 'LD_LIBRARY_PATH')},
        'cases': [],
    }
    write_record(args.output, record)
    try:
        if device.platform == 'gpu' and os.environ.get('TRANSFORMER_CUDA_ENVIRONMENT') == 'cuda13':
            from jax_plugins.xla_cuda13 import cuda_versions
            record['cublas_runtime_version'] = cuda_versions.cublas_get_version()
            if 'B300' in device.device_kind and record['cublas_runtime_version'] < 130200:
                raise RuntimeError('B300 qualification requires loaded cuBLAS >= 13.2')
            write_record(args.output, record)
        for path in args.configs:
            config = load_config(path)
            spec = config.model.attention_basis
            if not spec or any(b['kind'] != 'rbf' for b in spec['basis']):
                raise ValueError('Qualification requires a frozen RBF-only basis')
            case = {'config': str(path), 'config_sha256': sha256(path), 'basis_count': len(spec['basis']),
                    'normalization': config.model.attention_normalization, 'checks': []}
            record['cases'].append(case)
            for dtype in (jnp.float32, jnp.bfloat16):
                case['checks'].append(check(config, dtype))
                write_record(args.output, record)
                print(f'basis_count={len(spec["basis"])} check={case["checks"][-1]}', flush=True)
            if not args.skip_benchmark:
                case['benchmark'] = benchmark(config, args.repeats)
                print(f'basis_count={len(spec["basis"])} benchmark={case["benchmark"]}', flush=True)
            write_record(args.output, record)
            jax.clear_caches()
        record['status'] = 'passed'
    except BaseException as error:
        record['status'] = 'failed'
        record['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        record['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_record(args.output, record)


if __name__ == '__main__':
    main()
