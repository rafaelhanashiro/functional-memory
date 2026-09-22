"""Isolated A6000 host-staging qualification; never submit production work."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import UTC, datetime

try:
    from . import common
except ImportError:
    import common

RUNTIME_ROOT = Path(__file__).resolve().parents[3]
TIMEOUT_SECONDS = 900
ENVIRONMENT_KEYS = ('NCCL_P2P_DISABLE', 'POLYNOMIAL_ATTENTION_HOST_STAGING',
                    'POLYNOMIAL_ATTENTION_SOURCE_COMMIT', 'CUDA_VISIBLE_DEVICES',
                    'JAX_PLATFORMS', 'JAX_COMPILATION_CACHE_DIR', 'XLA_FLAGS', 'PYTHONPATH',
                    'XLA_PYTHON_CLIENT_MEM_FRACTION', 'XLA_PYTHON_CLIENT_PREALLOCATE')


def verify_runtime(expected: str) -> None:
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=RUNTIME_ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=RUNTIME_ROOT, text=True)
    if actual != expected or dirty:
        raise ValueError('Host-staging runtime checkout must match its recorded clean commit')


def load_inputs(policy_path: Path) -> tuple[dict, dict]:
    # Unlike common.load_policy, this does not put the old training checkout on
    # sys.path. Scientific imports below must resolve to the fixed runtime.
    policy = json.loads(policy_path.read_text())
    manifest = json.loads(Path(policy['training_manifest']).read_text())
    if common.policy_identity(policy) != policy['identity'] or policy['production_submission'] is not False:
        raise ValueError('Original validation-only policy identity changed')
    if policy['source_commit'] != manifest['source_commit'] or policy['training_identity'] != manifest['identity']:
        raise ValueError('Original scientific provenance changed')
    return policy, manifest


def resolve_runs(policy: dict, requested: list[str]) -> list[dict]:
    names = [name for value in requested for name in value.split(',') if name]
    if not names or len(set(names)) != len(names):
        raise ValueError('Specify distinct explicit run names')
    by_name = {run['name']: run for run in policy['runs']}
    if any(name not in by_name for name in names):
        raise ValueError('Unknown run name')
    runs = [by_name[name] for name in names]
    for run in runs:
        if common.sha256(Path(run['config_path'])) != run['config_sha256']:
            raise ValueError(f"Original config changed: {run['name']}")
    return runs


def worker_environment(commit: str, output_root: Path) -> dict:
    env = dict(os.environ)
    env.update(NCCL_P2P_DISABLE='1', POLYNOMIAL_ATTENTION_HOST_STAGING='1',
               POLYNOMIAL_ATTENTION_SOURCE_COMMIT=commit, JAX_PLATFORMS='cuda',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', PYTHONUNBUFFERED='1',
               XLA_PYTHON_CLIENT_MEM_FRACTION='0.90',
               PYTHONPATH=str(RUNTIME_ROOT),
               JAX_COMPILATION_CACHE_DIR=str(output_root / 'cache'))
    return env


def terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_child(command: list[str], log_path: Path, env: dict, lock_fd: int) -> dict:
    started = time.monotonic()
    with log_path.open('w') as log:
        process = subprocess.Popen(command, cwd=RUNTIME_ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1,
                                   start_new_session=True, pass_fds=(lock_fd,))

        def stream():
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)

        reader = threading.Thread(target=stream, daemon=True)
        reader.start()
        timeout = False
        try:
            process.wait(timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            timeout = True
            terminate(process)
        except BaseException:
            terminate(process)
            raise
        finally:
            reader.join(timeout=5)
    return {'command': command, 'exit_code': process.returncode, 'timeout': timeout,
            'elapsed_seconds': time.monotonic() - started, 'timeout_seconds': TIMEOUT_SECONDS,
            'log': common.reference(log_path)}


def scientific_helpers():
    sys.path.insert(0, str(RUNTIME_ROOT))
    preflight = importlib.import_module('experiments.key_dimension_taylor.preflight')
    metrics = importlib.import_module('experiments.polynomial_attention.plot')
    runner = importlib.import_module('experiments.key_dimension_taylor.hardware_checks.runner')
    for module in (preflight, metrics, runner):
        if not Path(module.__file__).resolve().is_relative_to(RUNTIME_ROOT):
            raise RuntimeError('Scientific import escaped the fixed runtime')
    return preflight, metrics, runner


def compare_h100(initial: dict, run: dict, manifest: dict) -> dict:
    root = Path(manifest['project_root']) / 'results/key_dimension_taylor/preflight' / manifest['identity']
    path = root / 'checks' / f"{run['name']}.json"
    if not path.exists():
        return {'status': 'unavailable', 'reason': 'No matching H100 full-model pilot; historical dK=64 reference'}
    reference = json.loads(path.read_text())
    if reference.get('status') != 'passed' or reference.get('identity') != manifest['identity'] or reference.get('source_commit') != manifest['source_commit']:
        raise ValueError('H100 pilot reference identity is invalid')
    expected = reference['initial']['checkpoint_audit']
    observed = initial['checkpoint_audit']
    for key in ('start_state_sha256', 'first_update'):
        if observed[key] != expected[key]:
            raise ValueError(f'H100 parity failed: {key}')
    return {'status': 'passed', 'reference': common.reference(path),
            'start_state_sha256': observed['start_state_sha256'], 'first_update': observed['first_update']}


def validate_rows(rows: list[dict], run: dict, commit: str, step: int) -> dict:
    row = next((item for item in rows if item['step'] == step), {})
    expected = {'step': step, 'source_commit': commit, 'qk_head_dim': run['qk_head_dim'],
                'attention_degree': run['attention_degree'], 'parameter_count': run['parameter_count']}
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError('Pilot validation identity or parameter count changed')
    if not all(math.isfinite(row.get(key, float('nan'))) for key in ('validation_nll', 'validation_perplexity')):
        raise FloatingPointError('Nonfinite pilot evaluation')
    return row


def validate_worker_metrics(rows: list[dict], run: dict, commit: str, stop: int) -> dict:
    if stop not in (32, 33):
        raise ValueError('Qualification is limited to updates 32 and 33')
    # Update 33 is a finite, audited continuation, not an evaluation boundary.
    # The training loop checks every update loss; its durable step-32 validation
    # must remain intact after the fresh worker saves update 33.
    return validate_rows(rows, run, commit, 32)


def worker(args, policy, manifest, run):
    verify_runtime(args.runtime_commit)
    if any(os.environ.get(key) != '1' for key in ('NCCL_P2P_DISABLE', 'POLYNOMIAL_ATTENTION_HOST_STAGING')):
        raise ValueError('Host staging and NCCL P2P disable must be explicit')
    if args.workdir is None or not args.workdir.resolve().is_relative_to(args.output_root.resolve() / 'attempts') or args.workdir.is_symlink():
        raise ValueError('Worker workdir must be an isolated qualification attempt')
    sys.path.insert(0, str(RUNTIME_ROOT))
    configs = importlib.import_module('experiments.polynomial_attention.configs')
    loop = importlib.import_module('experiments.polynomial_attention.training.loop')
    if any(not Path(module.__file__).resolve().is_relative_to(RUNTIME_ROOT) for module in (configs, loop)):
        raise ValueError('Worker imported scientific code outside the fixed runtime')
    hardware = common.check_hardware(policy['profiles']['a6000'], count=4)
    hardware['gpu_inventory'] = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid,driver_version', '--format=csv'], text=True)
    config = configs.load_config(run['config_path'])
    config.train.source_commit = args.runtime_commit
    config.train.workdir, config.train.wandb_mode = args.workdir, 'disabled'
    config.train.eval_every, config.train.eval_batches = 32, 1
    expected = copy.deepcopy(run['config'])
    expected['train'].update(source_commit=args.runtime_commit, workdir=str(args.workdir), wandb_mode='disabled', eval_every=32, eval_batches=1)
    if configs.config_to_dict(config) != expected or config.optim.total_steps != 19074:
        raise ValueError('Pilot changed an unapproved scientific setting')
    # Preserve hardware and resolved-config evidence even when training fails.
    common.write_json(args.output.with_name(args.output.stem + '-setup.json'),
                      {'hardware': hardware, 'resolved_config': expected,
                       'environment': {key: os.environ.get(key) for key in ENVIRONMENT_KEYS}})
    result = loop.run_training(config, resume=True, stop_after_step=int(args.worker), audit_checkpoint=True)
    result['hardware'] = hardware
    result['resolved_config'] = expected
    return result


def supervise(args, policy, manifest, runs):
    verify_runtime(args.runtime_commit)
    canonical = Path(manifest['project_root']) / 'results/key_dimension_taylor/hardware_checks/a6000_host_staging'
    if not args.output_root.resolve().is_relative_to(canonical.resolve()):
        raise ValueError('Qualification outputs must remain in a6000_host_staging')
    if common.sha256(RUNTIME_ROOT / 'experiments/polynomial_attention/uv.lock') != policy['dependency_lock_sha256']:
        raise ValueError('Fixed runtime changed the pinned dependency lock')
    # Full legacy validation runs in a separate CPU-only process, preventing its
    # original-source imports from contaminating fixed-runtime workers/helpers.
    verification = ("import importlib.util,sys; from pathlib import Path; "
                    "spec=importlib.util.spec_from_file_location('original_hardware_common',sys.argv[2]); "
                    "common=importlib.util.module_from_spec(spec); spec.loader.exec_module(common); "
                    "p,m=common.load_policy(Path(sys.argv[1])); common.verify_policy(p,m)")
    original_common = Path(policy['orchestrator_root']) / 'experiments/key_dimension_taylor/hardware_checks/common.py'
    subprocess.run([sys.executable, '-c', verification, str(args.policy), str(original_common)], cwd=policy['training_root'],
                   env=dict(os.environ, JAX_PLATFORMS='cpu', PYTHONPATH=policy['orchestrator_root']), check=True)
    preflight, metrics, runner = scientific_helpers()
    kernel_path = Path(policy['output_root']) / 'a6000/kernels.json'
    kernel = json.loads(kernel_path.read_text())
    if kernel.get('status') != 'passed' or kernel.get('tests_passed') != 105 or kernel.get('source_commit') != policy['source_commit']:
        raise ValueError('A6000 requires the existing successful 105-case kernel gate')
    for relative in ('experiments/polynomial_attention/models/components/fused_attention.py',
                     'experiments/polynomial_attention/models/components/attention.py'):
        if common.sha256(RUNTIME_ROOT / relative) != common.sha256(Path(policy['training_root']) / relative):
            raise ValueError('Kernel changed: original GPU correctness evidence cannot be reused')
    args.output_root.mkdir(parents=True, exist_ok=True)
    env = worker_environment(args.runtime_commit, args.output_root)
    outcomes = {}
    with common.locked(args.output_root / '.qualification.lock') as lock:
        for run in runs:
            attempt = args.output_root / 'attempts' / f"{run['name']}-{os.environ.get('SLURM_JOB_ID', 'local')}-{uuid.uuid4().hex[:8]}"
            attempt.mkdir(parents=True)
            workdir = attempt / 'workdir'
            record = {'format': 'key-dimension-taylor-a6000-host-staging', 'validation_only': True,
                      'production_submission': False, 'original_policy': common.reference(args.policy),
                      'original_policy_identity': policy['identity'], 'original_scientific_source_commit': policy['source_commit'],
                      'runtime_commit': args.runtime_commit, 'runtime_root': str(RUNTIME_ROOT),
                      'name': run['name'], 'qk_head_dim': run['qk_head_dim'], 'attention_degree': run['attention_degree'],
                      'parameter_count': run['parameter_count'], 'original_config': common.reference(Path(run['config_path'])),
                      'resolved_original_config': run['config'], 'data_identity': policy['data_identity'],
                      'dependency_lock_sha256': policy['dependency_lock_sha256'], 'kernel_reference': common.reference(kernel_path),
                      'environment': {key: env.get(key) for key in ENVIRONMENT_KEYS},
                      'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'node': socket.gethostname(),
                      'partition': os.environ.get('SLURM_JOB_PARTITION'), 'attempt_path': str(attempt),
                      'started_utc': datetime.now(UTC).isoformat(), 'status': 'running', 'processes': []}
            try:
                for stop, key in ((32, 'initial'), (33, 'resumed')):
                    output = attempt / f'{key}.json'
                    command = [sys.executable, str(Path(__file__).resolve()), '--policy', str(args.policy),
                               '--runtime-commit', args.runtime_commit, '--output-root', str(args.output_root),
                               '--names', run['name'], '--worker', str(stop), '--workdir', str(workdir), '--output', str(output)]
                    common.write_json(attempt / 'result.json', record)
                    process = run_child(command, attempt / f'{key}.log', env, lock.fileno())
                    record['processes'].append(process)
                    if output.exists():
                        record[key] = json.loads(output.read_text())
                    if process['timeout']:
                        raise TimeoutError(f'{key} exceeded {TIMEOUT_SECONDS} seconds')
                    if process['exit_code'] != 0:
                        error = record.get(key, {}).get('error', f"{key} exited {process['exit_code']}")
                        raise runner.CheckFailure(record.get(key, {}).get('status', 'infrastructure_failure'), error)
                    if key not in record:
                        raise RuntimeError('Successful worker omitted its audit')
                    record[f'{key}_last_committed_validation'] = validate_worker_metrics(metrics.committed_records(workdir), run, args.runtime_commit, stop)
                    if stop == 32:
                        record['h100_initialization_and_input_parity'] = compare_h100(record['initial'], run, manifest)
                runner._validate_progress(run, policy['profiles']['a6000'], record['initial'], record['resumed'])
                preflight.validate_resume_audit(record['initial'], record['resumed'])
                record['runtime'] = preflight.projected_runtime(run, record['initial'], record['resumed'], policy['profiles']['a6000'])
                record['status'] = 'passed'
            except Exception as error:
                record['status'] = (error.status if isinstance(error, runner.CheckFailure) else 'runtime_ineligible'
                                    if isinstance(error, preflight.RuntimeLimitExceeded) else runner.failure_status(error))
                record['error'] = f'{type(error).__name__}: {error}'
                record['traceback'] = traceback.format_exc()
                record['numerical_failures'] = [json.loads(path.read_text()) for path in sorted(workdir.glob('numerical_failure_step_*.json'))]
            finally:
                # Persist the result and audits before deleting only temporary
                # checkpoints; metrics/config/failure records remain inspectable.
                record['completed_utc'] = datetime.now(UTC).isoformat()
                common.write_json(attempt / 'result.json', record)
                runner._cleanup_checkpoints(workdir)
                record['temporary_checkpoints_removed'] = not (workdir / 'checkpoints').exists()
                common.write_json(attempt / 'result.json', record)
            common.write_json(args.output_root / 'checks' / f"{run['name']}.json", record)
            outcomes[run['name']] = record['status']
            print(json.dumps({'name': run['name'], 'status': record['status']}), flush=True)
        summary = {'validation_only': True, 'production_submission': False, 'runtime_commit': args.runtime_commit,
                   'outcomes': outcomes, 'completed_utc': datetime.now(UTC).isoformat()}
        common.write_json(args.output_root / f"summary-{os.environ.get('SLURM_JOB_ID', 'local')}.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--runtime-commit', required=True)
    parser.add_argument('--names', nargs='+', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--worker', choices=('32', '33'))
    parser.add_argument('--workdir', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    policy, manifest = load_inputs(args.policy)
    runs = resolve_runs(policy, args.names)
    if args.worker:
        if len(runs) != 1 or args.output is None:
            parser.error('Worker requires exactly one name and --output')
        try:
            result = worker(args, policy, manifest, runs[0])
            common.write_json(args.output, result)
        except Exception as error:
            # Importing the classification helper itself does not create a GPU
            # context; the failed worker may already have one.
            _, _, runner = scientific_helpers()
            common.write_json(args.output, {'status': runner.failure_status(error), 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()})
            raise
    else:
        supervise(args, policy, manifest, runs)


if __name__ == '__main__':
    main()
