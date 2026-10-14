"""Compare Ampere polynomial-GDN occupancy hints at production layer shape.

Runs baseline four warps, four warps with one minimum CTA per SM, and eight
warps with the same hint. Every variant measures synchronized forward and full
six-input value-and-gradient operations, and compares against baseline outputs
and gradients. PTXAS reports describe the actual PTX handed to the JAX runtime.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np

from scaffolds.transformer.attention.backends import polynomial_gdn as backend
from scaffolds.transformer.attention.backends.ejkernel_bridge import _triton_call as bridge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--length', type=int, default=2048)
    parser.add_argument('--heads', type=int, default=16)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--degree', type=int, choices=(2, 3), default=3)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--sweep', action='store_true',
                        help='Compare additional register-occupancy hints against four-warps hint=1')
    args = parser.parse_args()
    if jax.default_backend() != 'gpu' or jax.local_device_count() != 1:
        raise RuntimeError('Occupancy benchmark requires exactly one allocated GPU')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'status': 'running',
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_files': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            Path(__file__), Path(backend.__file__), Path(bridge.__file__),
            Path(backend.__file__).with_name('polynomial_gdn_kernels.py'))},
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'device_kind': jax.devices()[0].device_kind,
        'shape': {'batch': args.batch, 'length': args.length, 'heads': args.heads,
                  'key_width': args.width, 'value_width': args.width, 'degree': args.degree},
        'dtype': 'bfloat16', 'variants': {}, 'compiled_resources': [],
    }
    context = {'variant': 'preparation', 'stage': 'preparation'}
    def save():
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')

    original_constructor = bridge.triton_kernel_call_lib.TritonKernel
    def inspect_kernel(*constructor_args):
        kernel = original_constructor(*constructor_args)
        name = constructor_args[0]
        if not name.startswith(('forward', 'backward_queries', 'backward_keys')):
            return kernel
        ptx_index = next(i for i, value in enumerate(constructor_args)
                         if isinstance(value, str) and '.version ' in value)
        ptx = constructor_args[ptx_index]
        resource = {**context, 'kernel': name, 'num_warps': constructor_args[1],
                    'shared_bytes': constructor_args[ptx_index - 1],
                    'ptx_sha256': hashlib.sha256(ptx.encode()).hexdigest(),
                    'min_ctas_directive': re.findall(r'\.minnctapersm\s+(\d+)', ptx)}
        assembler = os.environ.get('TRITON_PTXAS_PATH')
        if not assembler:
            raise RuntimeError('Set TRITON_PTXAS_PATH to the locked assembler')
        with tempfile.TemporaryDirectory(prefix='gdn-occupancy-ptx-') as temp:
            path = Path(temp) / 'kernel.ptx'
            path.write_text(ptx)
            target = re.search(r'\.target\s+(sm_\w+)', ptx).group(1)
            compiled = subprocess.run([assembler, '-v', '--gpu-name', target, str(path),
                                       '-o', str(Path(temp) / 'kernel.cubin')], capture_output=True, text=True)
        if compiled.returncode:
            raise RuntimeError(compiled.stderr)
        resource['ptxas_report'] = compiled.stderr.strip()
        for field, pattern in (('registers', r'Used (\d+) registers'),
                               ('spill_store_bytes', r'(\d+) bytes spill stores'),
                               ('spill_load_bytes', r'(\d+) bytes spill loads'),
                               ('stack_frame_bytes', r'(\d+) bytes stack frame')):
            found = re.search(pattern, compiled.stderr)
            resource[field] = int(found.group(1)) if found else None
        report['compiled_resources'].append(resource)
        save()
        return kernel
    bridge.triton_kernel_call_lib.TritonKernel = inspect_kernel

    rng = np.random.default_rng(127)
    shape = (args.batch, args.length, args.heads, args.width)
    def normal(shape):
        return rng.standard_normal(shape, dtype=np.float32)
    q, k = normal(shape), normal(shape)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    q, k, v, cotangent = (jnp.asarray(a, jnp.bfloat16) for a in (q, k, normal(shape), normal(shape)))
    gate_shape = shape[:3]
    g = jnp.asarray(-rng.uniform(.001, .1, gate_shape), jnp.float32)
    beta = jnp.asarray(normal((*gate_shape, args.degree + 1)) * .2, jnp.float32)
    erase = jnp.asarray(rng.uniform(.1, .9, gate_shape), jnp.float32)
    operands = (q, k, v, g, beta, erase)
    scale = args.width ** -.5
    baseline = None
    original_call = backend._polynomial_call

    def measure(function, stage):
        context['stage'] = stage
        start = time.perf_counter()
        executable = jax.jit(function).lower(*operands).compile()
        compile_seconds = time.perf_counter() - start
        for _ in range(2):
            result = jax.block_until_ready(executable(*operands))
        timings = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            result = jax.block_until_ready(executable(*operands))
            timings.append(time.perf_counter() - start)
        timing = {'seconds': timings, 'median_seconds': statistics.median(timings),
                  'minimum_seconds': min(timings), 'compile_seconds': compile_seconds}
        try:
            timing['memory_analysis'] = dataclasses.asdict(executable.memory_analysis())
        except TypeError:
            timing['memory_analysis'] = str(executable.memory_analysis())
        print(json.dumps({**context, **timing}), flush=True)
        return result, timing

    try:
        variants = [('baseline_w4', 4, None, {}), ('hint_w4', 4, 1, {}), ('hint_w8', 8, 1, {})]
        if args.sweep:
            variants = [('hint_w4', 4, 1, {}), ('hint2_w4', 4, 2, {}),
                        ('hint3_w4', 4, 3, {}), ('hint4_w4', 4, 4, {}),
                        ('query_hint3_w4', 4, 1, {'backward_queries': 3}),
                        ('hint_w2', 2, 1, {})]
        report['baseline_variant'] = variants[0][0]
        for variant, warps, hint, stage_hints in variants:
            context['variant'] = variant
            entry = {'num_warps': warps, 'min_ctas_per_sm': hint,
                     'stage_min_ctas_per_sm': stage_hints, 'timings': {}}
            report['variants'][variant] = entry
            def configured_call(kernel, *values, out, q, v, name, **extra):
                return bridge.triton_call(
                    *values, kernel=kernel, out_shape=out,
                    grid=(q.shape[1] // backend.CHUNK_SIZE, q.shape[0] * q.shape[2]),
                    name=f'polynomial_gdn_occupancy::{variant}::{name}',
                    default_dot_input_precision='tf32x3', num_warps=warps, num_stages=1,
                    min_ctas_per_sm=stage_hints.get(name, hint),
                    **backend._metadata(q, v), **extra)
            backend._polynomial_call = configured_call
            output, entry['timings']['forward'] = measure(
                lambda *a: backend.polynomial_gated_delta_rule(*a, scale), 'forward')
            def objective(*a):
                return jnp.sum(backend.polynomial_gated_delta_rule(*a, scale).astype(jnp.float32)
                               * cotangent.astype(jnp.float32))
            gradients, entry['timings']['value_and_grad'] = measure(
                jax.value_and_grad(objective, argnums=tuple(range(6))), 'value_and_grad')
            host = jax.tree.map(lambda a: np.asarray(a, np.float32), (output, gradients))
            if not all(np.all(np.isfinite(a)) for a in jax.tree.leaves(host)):
                raise AssertionError(f'{variant} produced a nonfinite output or gradient')
            if baseline is None:
                baseline = host
            else:
                errors = []
                for actual, expected in zip(jax.tree.leaves(host), jax.tree.leaves(baseline), strict=True):
                    delta = actual.astype(np.float64) - expected.astype(np.float64)
                    relative = float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(expected.ravel()), 1e-30))
                    errors.append(relative)
                    if not np.all(np.isfinite(actual)) or relative > 3e-3:
                        raise AssertionError(f'{variant} changed output/gradient: relative_error={relative}')
                entry['relative_errors_vs_baseline'] = errors
            del output, gradients, host
            save()
        report['status'] = 'passed'
    except BaseException as error:
        report['status'] = 'failed'
        report['error'] = repr(error)
        raise
    finally:
        backend._polynomial_call = original_call
        bridge.triton_kernel_call_lib.TritonKernel = original_constructor
        save()
    print(f'benchmark_result={args.output}', flush=True)


if __name__ == '__main__':
    main()
