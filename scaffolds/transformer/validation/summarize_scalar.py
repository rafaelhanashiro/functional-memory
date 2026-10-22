"""Collect explicit scalar-attention qualification jobs, retaining failed cases."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
RECORDS = ROOT / 'scaffolds/transformer/validation/runs/scalar-triton'


def artifact(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'size_bytes': path.stat().st_size}


def suite_record(path):
    suite = ET.parse(path).getroot().find('testsuite')
    return {'report': artifact(path), **{k: int(suite.get(k)) for k in ('tests', 'errors', 'failures', 'skipped')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, nargs='+', required=True)
    args = parser.parse_args()
    record = {'format': 'scalar-triton-qualification', 'version': 1, 'home_cluster': 'Kennel',
              'cpu': {'unit_tests': suite_record(RECORDS/'cpu.xml')},
              'gpu': {'jobs': args.jobs, 'allocations': [], 'qualification': [],
                      'kernel_comparisons': [], 'optimizer_updates': []}}
    for label, filename in [('interpreter', 'interpreter.json'), ('cuda_export', 'cuda-export.json'),
                            ('training_cuda_export', 'training-cuda-export.json')]:
        path = RECORDS / filename
        result = json.loads(path.read_text())
        item = {'status': result['status'], 'cases': len(result['cases']), 'report': artifact(path),
                'source_commit': result['source_commit'],
                'frozen_execution_sha256': result['frozen_execution_sha256']}
        if label == 'interpreter':
            item['max_relative_l2_error'] = max(max(c['relative_l2_errors'].values()) for c in result['cases'])
        record['cpu'][label] = item
    record['cpu']['source_commit'] = record['cpu']['interpreter']['source_commit']
    record['cpu']['frozen_execution_sha256'] = record['cpu']['interpreter']['frozen_execution_sha256']
    record['dependency_lock_sha256'] = hashlib.sha256((ROOT/'scaffolds/transformer/uv.lock').read_bytes()).hexdigest()
    accounting = subprocess.check_output(['sacct', '-n', '-P', '-j', ','.join(map(str, args.jobs)),
        '--format=JobIDRaw,State,Elapsed,ExitCode,NodeList'], text=True, timeout=15)
    for line in accounting.splitlines():
        fields = line.split('|')
        if fields[0] in {str(j) for j in args.jobs}:
            record['gpu']['allocations'].append(dict(zip(
                ('job_id', 'state', 'elapsed', 'exit_code', 'nodes'), fields[:5], strict=True)))
    seen = set()
    for job in args.jobs:
        directory = RECORDS / str(job)
        tests = directory / 'tests.xml'
        if tests.exists():
            record['gpu']['qualification'].append({'job_id': job, **suite_record(tests)})
        for path in sorted(directory.glob('*.json')):
            result = json.loads(path.read_text())
            if result.get('format') == 'scalar-attention-comparison':
                coordinate = (result['function'], result['mode'], tuple(result['shape']))
                if coordinate in seen:
                    raise ValueError(f'Duplicate comparison: {coordinate}; select explicit nonoverlapping jobs')
                seen.add(coordinate)
                comparison = {k: result[k] for k in ('function', 'mode', 'shape', 'dtype', 'status')}
                comparison['report'] = artifact(path)
                comparison['implementations'] = []
                for c in result['cases']:
                    item = {k: c[k] for k in ('implementation', 'status', 'error') if k in c}
                    for field in ('forward', 'forward_backward'):
                        if field in c:
                            item[field+'_median_ms'] = c[field]['median_ms']
                    if 'relative_l2_errors_vs_blocked' in c:
                        item['max_relative_l2_error_vs_blocked'] = max(c['relative_l2_errors_vs_blocked'].values())
                    comparison['implementations'].append(item)
                record['gpu']['kernel_comparisons'].append(comparison)
            elif result.get('format') == 'single-gpu-accumulation':
                cfg = result['configuration']
                item = {'report': artifact(path), 'status': result['status'],
                        'basis_count': len(cfg['model']['attention_basis']['basis']),
                        'backend': cfg['model']['attention_backend'],
                        'microbatch': cfg['train']['micro_batch_size'], 'global_batch': cfg['data']['batch_size']}
                for key in ('error', 'steady_compute', 'initial_parameter_sha256', 'gpu_memory_stats'):
                    if key in result:
                        item[key] = result[key]
                record['gpu']['optimizer_updates'].append(item)
    live = any(x['state'] in ('PENDING', 'RUNNING', 'COMPLETING') for x in record['gpu']['allocations'])
    record['gpu']['status'] = 'in_progress' if live else 'finished'
    (RECORDS/'record.json').write_text(json.dumps(record, indent=2, sort_keys=True)+'\n')
    for comparison in record['gpu']['kernel_comparisons']:
        print(comparison['function'], comparison['mode'], comparison['shape'])
        for c in comparison['implementations']:
            print(' ', c['implementation'], c['status'], c.get('forward_backward_median_ms', c.get('error')))
    for item in record['gpu']['optimizer_updates']:
        print('update',item['basis_count'],item['backend'],item['status'],item.get('steady_compute'))


if __name__ == '__main__':
    main()
