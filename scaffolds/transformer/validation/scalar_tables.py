"""Regenerate the scalar validation tables from explicitly selected completed jobs."""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
RECORDS = ROOT / 'scaffolds/transformer/validation/runs/scalar-triton'
RUNS = {3: 'r10b2cb22687c', 20: 'r873f76c0a542'}


def read(job, filename):
    data = json.loads((RECORDS / str(job) / filename).read_text())
    if data.get('status') != 'complete':
        raise ValueError(f'{job}/{filename} is not a completed comparison')
    return data


def cases(data):
    if any(c['status'] != 'complete' for c in data['cases']):
        raise ValueError('Cannot publish a table with an unqualified implementation')
    return {c['implementation']: c['forward_backward']['median_ms'] for c in data['cases']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ('initial', 'tuning', 'training', 'grid'):
        parser.add_argument(f'--{role}-job', type=int, required=True)
    parser.add_argument('--write-readme', action='store_true')
    args = parser.parse_args()
    rows = ['<!-- BEGIN SCALAR TABLES -->',
            f'Full optimizer updates: job `{args.training_job}`, seconds per global batch.', '',
            '| RBFs | Blocked | Selected Triton | Speedup | Peak allocator GiB, blocked / Triton |',
            '| ---: | ---: | ---: | ---: | ---: |']
    for count, run in RUNS.items():
        b, n = [read(args.training_job, f'{run}-{backend}.json') for backend in ('blocked', 'triton_scalar')]
        if b['initial_parameter_sha256'] != n['initial_parameter_sha256']:
            raise ValueError('Initial model parameters differ')
        for data in (b, n):
            cfg = data['configuration']
            assert cfg['data']['batch_size'] == 256 and cfg['data']['sequence_length'] == 2049
            assert cfg['train']['micro_batch_size'] == 16 and data['timed_updates'] == 5
        bt, nt = [d['steady_compute']['median_seconds'] for d in (b, n)]
        bm, nm = [d['gpu_memory_stats']['peak_bytes_in_use']/(1 << 30) for d in (b, n)]
        rows.append(f'| {count} | {bt:.3f} | {nt:.3f} | {bt/nt:.2f}x | {bm:.2f} / {nm:.2f} |')
    rows += ['', f'Selected RBF kernels: job `{args.grid_job}`. Each cell is blocked → selected Triton,',
             'forward plus backward in milliseconds, at `(B, 2048, 16, 64)`.', '',
             '| RBFs | B=1 | B=4 | B=16 | B=32 |', '| ---: | ---: | ---: | ---: | ---: |']
    for count in (3, 5, 9, 20):
        values = []
        for batch in (1, 4, 16, 32):
            data = read(args.grid_job, f'rbf{count}-b{batch}.json')
            assert data['shape'] == [batch, 2048, 16, 64]
            c = cases(data)
            values.append(f'{c["blocked"]:.3f} → {c["triton_scalar"]:.3f}')
        rows.append(f'| {count} | ' + ' | '.join(values) + ' |')
    rows += ['', f'Tile comparison: job `{args.tuning_job}`, microbatch 16, forward plus backward in milliseconds.',
             'The 3-RBF callback is direct; the 20-RBF callback uses the recurrence.', '',
             '| Tiles: forward; dQ; dK/dV | 3 RBFs | 20 RBFs |', '| --- | ---: | ---: |']
    left, right = [read(args.tuning_job, f'rbf{count}-tuning.json') for count in (3, 20)]
    lc, rc = cases(left), cases(right)
    for case in left['cases']:
        label = case['implementation']
        if label == 'blocked':
            continue
        t = case['tuning']
        description = f'{t["forward_q"]}×{t["forward_k"]}; {t["backward_q"]}×{t["backward_k"]}; {t["kv_q"]}×{t["kv_k"]}'
        if label == 'small':
            description += ' (selected)'
        rows.append(f'| {description} | {lc[label]:.3f} | {rc[label]:.3f} |')
    rows += ['', f'Other scalar functions: job `{args.initial_job}`, microbatch 16, forward plus backward',
             'in milliseconds. These native-Triton controls use the initial 128×64 forward,',
             '128×32 dQ, and 32×128 dK/dV tiles; they do not measure the selected smaller tiles.', '',
             '| Function / output | Blocked | Native Triton | Pallas | Mosaic | cuDNN |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for function, mode in (('sine', 'rms'), ('poly3', 'rms'), ('poly8', 'rms'), ('exp', 'rms'),
                           ('poly3', 'sum'), ('poly8', 'sum'), ('exp', 'sum')):
        c = cases(read(args.initial_job, f'{function}-{mode}-b16.json'))
        native = 'triton_streamed' if mode == 'rms' else 'triton_separate'
        values = [f'{c[key]:.3f}' if key in c else '—' for key in ('blocked', native, 'pallas', 'mosaic', 'cudnn')]
        label = 'head RMSNorm' if mode == 'rms' else 'normalized'
        rows.append(f'| {function} / {label} | ' + ' | '.join(values) + ' |')
    rows += ['<!-- END SCALAR TABLES -->']
    block = '\n'.join(rows)
    if args.write_readme:
        path = ROOT / 'scaffolds/transformer/validation/README.md'
        text, count = re.subn(r'<!-- BEGIN SCALAR TABLES -->.*?<!-- END SCALAR TABLES -->',
                              lambda match: block, path.read_text(), flags=re.S)
        if count != 1:
            raise ValueError('Expected one scalar-table marker block in the validation README')
        path.write_text(text)
    else:
        print(block)


if __name__ == '__main__':
    main()
