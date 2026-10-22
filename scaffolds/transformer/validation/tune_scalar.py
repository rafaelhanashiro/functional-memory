"""Compare a small explicit tile grid after the scalar-attention H100 profile."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from scaffolds.transformer.benchmarks import benchmark_scalar as bench
from scaffolds.transformer.attention.backends.triton_scalar import (
    Tuning, scalar_attention, uniform_rbf_kernel,
)


TUNINGS = {
    'original': Tuning(),
    'small': Tuning(forward_q=64, forward_k=32, backward_q=64, kv_k=64),
    'square': Tuning(forward_q=64, forward_k=64, backward_q=64, backward_k=64,
                     kv_q=64, kv_k=64),
    'tiny': Tuning(forward_q=64, forward_k=32, backward_q=32, kv_k=32),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel', choices=('rbf3', 'rbf20'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    original_operations = bench.operations

    def operations(name, mode, precise=False, tune=False):
        implementations, recipe = original_operations(name, mode)
        kernel, reference_args, _ = bench.case(name)
        if name == 'rbf20':
            kernel = uniform_rbf_kernel(reference_args['score_kernel'])
        result = {'blocked': implementations['blocked']}
        for label, tuning in TUNINGS.items():
            def operation(q, k, v, gain, tuning=tuning):
                return scalar_attention(q, k, v, kernel=kernel, output_norm_scale=gain,
                                        streaming=True, tuning=tuning)
            result[label] = operation
        return result, recipe

    bench.operations = operations
    bench.benchmark(SimpleNamespace(kernel=args.kernel, batch=16, length=2048, mode='rms',
        iterations=15, output=args.output, profile=None, precise=False, tune=False))
    report = json.loads(args.output.read_text())
    report['format'] = 'scalar-attention-tuning'
    report['callback'] = 'basis' if args.kernel == 'rbf3' else 'rbf_recurrence'
    report['driver_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for case in report['cases']:
        if case['implementation'] in TUNINGS:
            case['tuning'] = asdict(TUNINGS[case['implementation']])
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')


if __name__ == '__main__':
    main()
