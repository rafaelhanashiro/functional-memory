"""Summarize scalar benchmark GPU traces, including kernel register usage."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from scaffolds.transformer.benchmarks import analyze_profile as base


NATIVE_KERNELS = {'forward', 'preprocess', 'backward_q', 'backward_kv'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--step-name', choices=('train', 'attention'), default='train')
    parser.add_argument('--max-expanded-mib', type=int, default=100)
    args = parser.parse_args()
    original_category = base.kernel_category
    base.CATEGORIES = (*base.CATEGORIES, 'native_scalar_attention')
    base.kernel_category = lambda e: ('native_scalar_attention' if e.get('name') in NATIVE_KERNELS
                                      else original_category(e))
    report = {'format': 'scalar-attention-profile', 'version': 1,
              'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'method': 'GPU Stream # events inside synchronized host annotations; duration sums '
                        'can include overlap. Native kernel names are specific to these benchmarks. '
                        'Register counts and theoretical occupancy are profiler metadata, not measured SM utilization.',
              'traces': []}
    for path in base.discover_traces(args.traces):
        trace = base.analyze_file(path, step_name=args.step_name, top_kernels=20,
                                  max_bytes=args.max_expanded_mib*1024*1024)
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rb') as stream:
            events = json.load(stream)['traceEvents']
        _, selected, _, _ = base.select_events(events, args.step_name)
        resources = Counter()
        for device_events in selected.values():
            for event, is_kernel in device_events:
                if is_kernel and event['name'] in NATIVE_KERNELS:
                    resources[(event['name'], event['args']['kernel_details'])] += 1
        trace['native_kernel_resources'] = [
            {'kernel': name, 'details': details, 'recorded_launches': count}
            for (name, details), count in sorted(resources.items())]
        report['traces'].append(trace)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')


if __name__ == '__main__':
    main()
