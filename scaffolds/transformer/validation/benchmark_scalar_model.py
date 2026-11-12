"""Match the model's selected scalar kernel against blocked JAX at training shapes."""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
from types import SimpleNamespace

from scaffolds.transformer.benchmarks import benchmark_scalar as bench
from scaffolds.transformer.attention.backends.triton_scalar import Tuning, basis_kernel, scalar_attention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel', choices=('rbf3', 'rbf5', 'rbf9', 'rbf20'), required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    original_operations, original_write = bench.operations, bench.write_report
    _, reference_args, _ = bench.case(args.kernel)
    selected = basis_kernel(reference_args['score_kernel'], implementation='auto')

    def operations(name, mode, precise=False, tune=False):
        implementations, recipe = original_operations(name, mode)
        return {'blocked': implementations['blocked'], 'triton_scalar': lambda q, k, v, gain:
                scalar_attention(q, k, v, kernel=selected, output_norm_scale=gain)}, recipe

    def write(path, report):
        report['format'] = 'scalar-model-kernel-comparison'
        report['scalar_execution'] = {'callback': selected.name, 'tuning': asdict(Tuning()),
                                      'streaming': True, 'fused_head_rms_norm': True}
        report['driver_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        original_write(path, report)

    bench.operations, bench.write_report = operations, write
    bench.benchmark(SimpleNamespace(kernel=args.kernel, batch=args.batch, length=2048, mode='rms',
        iterations=15, output=args.output, profile=None, precise=False, tune=False))


if __name__ == '__main__':
    main()
