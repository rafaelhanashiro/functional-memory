"""Synthetic GPU precision audit; does not change or qualify production kernels."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from scaffolds.transformer.attention.backends.fla_gated_delta_rule import (
    fla_gated_delta_rule, fla_independent_gated_delta_rule,
)
from scaffolds.transformer.tests.test_gated_deltanet_kernel import _operands


def oracle(args, cotangent):
    """Closed form at q=k=e1, g=0, erase=1, for independent raw q/k inputs."""
    q, k, v, g, beta, erase = (np.asarray(a, np.float64) for a in args)
    c = np.asarray(cotangent, np.float64)
    scale = k.shape[-1] ** -0.5
    a = beta[..., None] * v
    prev = np.concatenate((np.zeros_like(a[:, :1]), a[:, :-1]), axis=1)
    dq, dk = np.zeros_like(q), np.zeros_like(k)
    dq[..., 0] = scale * np.sum(c * a, axis=-1)
    dk[..., 0] = scale * np.sum(c * (a - 2 * prev), axis=-1)
    return (scale * a, dq, dk, scale * beta[..., None] * c, np.zeros_like(g),
            scale * np.sum(c * v, axis=-1), -scale * np.sum(c * prev, axis=-1))


def error(actual, expected):
    a, e = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    n = float(np.linalg.norm(e))
    d = float(np.linalg.norm(a - e))
    return {'relative_l2': d / n if n else None, 'absolute_l2': d,
            'max_absolute': float(np.max(np.abs(a - e))), 'reference_l2': n,
            'finite': bool(np.isfinite(a).all())}


def evaluate(function, args, cotangent, expected):
    output = function(*args)
    gradients = jax.grad(lambda *a: jnp.sum(function(*a).astype(jnp.float32)
                                           * cotangent.astype(jnp.float32)), argnums=range(6))(*args)
    names = ('output', 'dq', 'dk', 'dv', 'dg', 'dbeta', 'derase')
    return dict(zip(names, (error(a, e) for a, e in zip((output, *gradients), expected)), strict=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args_cli = parser.parse_args()
    rows = []
    scale = 64 ** -0.5
    independent = lambda q, k, v, g, b, e: fla_independent_gated_delta_rule(q, k, v, g, b, e, scale)
    # Diagnostic identity for nonzero erase only; no such division is used in production.
    original = lambda q, k, v, g, b, e: fla_gated_delta_rule(q, k, v * (b / e)[..., None], g, e, scale)
    for dtype in (jnp.float32, jnp.bfloat16):
        original_args, cot = _operands(64, dtype, batch=1, heads=2, length=128, seed=44)
        _, _, v, g, b = original_args
        q = k = jnp.zeros_like(original_args[0]).at[..., 0].set(1)
        args = (q, k, v, jnp.zeros_like(g), b, jnp.ones_like(b))
        expected = oracle(args, cot)
        for name, function, inputs in (
            ('independent_native', independent, args),
            ('original_reparameterized', original, args),
            ('independent_promoted_fp32', independent, tuple(a.astype(jnp.float32) for a in args)),
        ):
            print(f'START dtype={dtype.__name__} variant={name}', flush=True)
            metrics = evaluate(function, inputs, cot, expected)
            row = {'fixture_dtype': dtype.__name__, 'variant': name, 'metrics': metrics}
            rows.append(row)
            print(json.dumps(row), flush=True)
    record = {
        'source_commit': os.environ.get('GDN_DIAGNOSTIC_SOURCE') or subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'dependency_lock_sha256': hashlib.sha256(Path('scaffolds/transformer/uv.lock').read_bytes()).hexdigest(),
        'completed_utc': datetime.now(UTC).isoformat(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'node': os.environ.get('SLURMD_NODENAME'),
        'devices': [d.device_kind for d in jax.local_devices()],
        'runtime_environment': {k: os.environ.get(k) for k in ('TRITON_F32_DEFAULT', 'XLA_FLAGS')},
        'fixture': {'seed': 44, 'batch': 1, 'length': 128, 'heads': 2, 'key_dim': 64, 'value_dim': 64,
                    'keys_and_queries': 'e1', 'log_decay': 0, 'erase': 1},
        'measurements': rows,
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
