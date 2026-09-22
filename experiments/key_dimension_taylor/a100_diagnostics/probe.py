"""Temporary A100 transfer diagnosis; bounded pilots, no production submission."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT.parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def number(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def worker(args):
    import jax
    import numpy as np
    from experiments.polynomial_attention import configs
    from experiments.polynomial_attention.training import loop, distributed

    policy = json.loads((PROJECT / 'results/key_dimension_taylor/hardware_checks/policy.json').read_text())
    run = next(r for r in policy['runs'] if r['name'] == args.name)
    config = configs.load_config(run['config_path'])
    config.train.workdir = args.output / args.mode / args.name / 'workdir'
    config.train.source_commit = args.commit
    config.train.wandb_mode = 'disabled'
    config.train.eval_every, config.train.eval_batches = 32, 1
    assert config.optim.total_steps == 19074
    assert len(jax.local_devices()) == 4
    assert all('A100' in d.device_kind for d in jax.local_devices())
    record = {'mode': args.mode, 'stop': args.worker, 'name': args.name,
              'runtime_commit': args.commit, 'scientific_source': policy['source_commit'],
              'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'node': socket.gethostname(),
              'devices': [str(d) for d in jax.local_devices()],
              'environment': {k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'NCCL_P2P_DISABLE',
                  'POLYNOMIAL_ATTENTION_HOST_STAGING', 'XLA_PYTHON_CLIENT_MEM_FRACTION')},
              'config': configs.config_to_dict(config), 'events': [], 'status': 'running'}
    output = args.output / args.mode / args.name / f'probe-{args.worker}.json'

    def event(value):
        record['events'].append(value)
        write(output, record)
        print(json.dumps(value), flush=True)

    original_put = loop.put_replicated
    def checked_put(state, sharding):
        placed = original_put(state, sharding)
        by_device = {str(d): {'leaves': 0, 'mismatched_leaves': 0, 'examples': []} for d in jax.local_devices()}
        original = jax.tree_util.tree_flatten_with_path(state)[0]
        result = jax.tree.leaves(placed)
        for (path, source), value in zip(original, result):
            if not isinstance(source, jax.Array):
                continue
            expected = np.asarray(source)
            for shard in value.addressable_shards:
                actual = np.asarray(shard.data)
                item = by_device[str(shard.device)]
                item['leaves'] += 1
                equal = actual.dtype == expected.dtype and actual.shape == expected.shape and np.array_equal(
                    actual.ravel().view(np.uint8), expected.ravel().view(np.uint8))
                if not equal:
                    item['mismatched_leaves'] += 1
                    if len(item['examples']) < 8:
                        item['examples'].append({'path': jax.tree_util.keystr(path), 'shape': list(expected.shape),
                            'expected_first': [number(x) for x in expected.ravel()[:8]],
                            'actual_first': [number(x) for x in actual.ravel()[:8]],
                            'expected_sha256': hashlib.sha256(expected.tobytes()).hexdigest(),
                            'actual_sha256': hashlib.sha256(actual.tobytes()).hexdigest(),
                            'actual_nonfinite': int((~np.isfinite(actual)).sum()),
                            'actual_zero_fraction': number(np.mean(actual == 0))})
        event({'kind': 'initial_state_replicas', 'devices': by_device})
        return placed
    loop.put_replicated = checked_put

    def check_batch(host, placed, kind):
        items = []
        for shard in placed.addressable_shards:
            actual, expected = np.asarray(shard.data), np.asarray(host)[shard.index]
            items.append({'device': str(shard.device), 'shape': list(actual.shape),
                          'matches_host': bool(np.array_equal(actual, expected)),
                          'min': int(actual.min()), 'max': int(actual.max()),
                          'mismatch_elements': int(np.sum(actual != expected))})
        event({'kind': kind, 'shards': items})
        return placed

    original_train_put, original_eval_put = loop.put_train_batches, loop.put_eval_batch
    calls = 0
    def train_put(host, sharding):
        nonlocal calls
        calls += 1
        result = original_train_put(host, sharding)
        return check_batch(host, result, 'first_train_batch') if calls == 1 else result
    loop.put_train_batches = train_put
    loop.put_eval_batch = lambda host, sharding: check_batch(host, original_eval_put(host, sharding), 'eval_batch')
    original_functions = loop.make_data_parallel_functions
    def functions(state, devices):
        from dataclasses import replace
        functions = original_functions(state, devices)
        evaluate = functions.eval_step
        def eval_step(state, batch, rng):
            loss = evaluate(state, batch, rng)
            event({'kind': 'eval_loss', 'value': number(loss), 'step': int(state.step)})
            return loss
        return replace(functions, eval_step=eval_step)
    loop.make_data_parallel_functions = functions
    try:
        result = loop.run_training(config, resume=True, stop_after_step=args.worker, audit_checkpoint=True)
        record.update(status='passed', result=result)
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        traceback.print_exc()
    write(output, record)
    return record['status'] == 'passed'


def supervise(args):
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == args.commit
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
    args.output.mkdir(parents=True, exist_ok=True)
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid,driver_version', '--format=csv'], text=True)
    topology = subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True)
    write(args.output / 'allocation.json', {'job': os.environ.get('SLURM_JOB_ID'), 'node': socket.gethostname(),
        'inventory': inventory, 'topology': topology, 'source_commit': args.commit, 'home_cluster': 'csail-kennel',
        'artifact_root': str(args.output), 'production_submission': False})
    summary = []
    for mode in args.modes:
        env = dict(os.environ, JAX_PLATFORMS='cuda', PYTHONPATH=str(ROOT), PYTHONUNBUFFERED='1',
                   XLA_PYTHON_CLIENT_PREALLOCATE='false', JAX_COMPILATION_CACHE_DIR=str(args.output / 'cache'))
        for key in ('NCCL_P2P_DISABLE', 'POLYNOMIAL_ATTENTION_HOST_STAGING', 'XLA_PYTHON_CLIENT_MEM_FRACTION'):
            env.pop(key, None)
        if mode in ('nccl_only', 'both'):
            env['NCCL_P2P_DISABLE'] = '1'
        if mode in ('host_only', 'both'):
            env['POLYNOMIAL_ATTENTION_HOST_STAGING'] = '1'
        for step in (32, 33):
            target = args.output / mode / args.name
            target.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(Path(__file__).resolve()), '--output', str(args.output), '--commit', args.commit,
                       '--name', args.name, '--mode', mode, '--worker', str(step)]
            start = time.monotonic()
            with (target / f'probe-{step}.log').open('w') as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = process.wait(timeout=900)
                except BaseException:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    code = 124
            outcome = {'mode': mode, 'step': step, 'exit_code': code, 'seconds': time.monotonic()-start}
            summary.append(outcome)
            write(args.output / 'summary.json', summary)
            print(json.dumps(outcome), flush=True)
            if code:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--name', default='200m-dk008-m004-seed0')
    parser.add_argument('--worker', type=int, choices=(32, 33))
    parser.add_argument('--mode', choices=('native', 'nccl_only', 'host_only', 'both'))
    parser.add_argument('--modes', nargs='+', default=['native', 'nccl_only', 'host_only', 'both'])
    args = parser.parse_args()
    if args.worker:
        sys.exit(0 if worker(args) else 1)
    supervise(args)


if __name__ == '__main__':
    main()
