"""Preserve the fitting recipe and selected inputs while adding explicit degrees."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest
import yaml

from experiments.transformer_pp import launch_coefficient_fit as launcher
from experiments.transformer_pp.launch_coefficient_fit import fitting_scope


@pytest.mark.parametrize(('count', 'role'), [
    (1, 'static_coefficients_single_sequence'),
    (256, 'static_coefficients_full_batch'),
])
def test_static_scope_requires_explicit_single_or_parent_batch(count, role):
    assert fitting_scope('static_coefficients', 256, count) == (count, role)


@pytest.mark.parametrize(('parent_batch', 'count'), [(256, None), (256, 8), (128, 256)])
def test_static_scope_rejects_missing_partial_or_wrong_parent_batch(parent_batch, count):
    with pytest.raises(ValueError, match='requires --sequence-count'):
        fitting_scope('static_coefficients', parent_batch, count)


@pytest.mark.parametrize(('count', 'role'), [
    (1, 'free_coefficients_single_sequence'),
    (256, 'free_coefficients_full_batch'),
])
def test_free_scope_requires_explicit_single_or_parent_batch(count, role):
    assert fitting_scope('free_per_example_query_coefficients', 256, count) == (count, role)


@pytest.mark.parametrize(('parent_batch', 'count'), [(256, None), (256, 8), (128, 256)])
def test_free_scope_rejects_missing_partial_or_wrong_parent_batch(parent_batch, count):
    with pytest.raises(ValueError, match='requires --sequence-count'):
        fitting_scope('free_per_example_query_coefficients', parent_batch, count)


@pytest.mark.parametrize(('mode', 'expected'), [
    ('query_coefficient_generator', (1, 'query_generator_single_sequence')),
    ('shared_batch_query_coefficients', (256, 'shared_coefficients_full_batch')),
])
def test_existing_scopes_are_preserved(mode, expected):
    assert fitting_scope(mode, 256) == expected
    if mode == 'query_coefficient_generator':
        assert fitting_scope(mode, 256, 1) == expected
        assert fitting_scope(mode, 256, 256) == (256, 'query_generator_full_batch')
        for count in (0, 8, 128):
            with pytest.raises(ValueError, match='requires --sequence-count'):
                fitting_scope(mode, 256, count)
    else:
        with pytest.raises(ValueError, match='only supported'):
            fitting_scope(mode, 256, 1)


def test_degree_roles_preserve_existing_selection_names():
    role = 'query_generator_single_sequence'
    assert launcher.fitting_role(role, 8) == role
    assert launcher.fitting_role(role, 4) == role + '_m4'
    for invalid in (None, 0, -1, True, 4.0):
        with pytest.raises(ValueError, match='positive integer'):
            launcher.fitting_role(role, invalid)


def original_query_fit():
    # A complete query-generator fit recipe (degree 8, earlier QK setup) used as the test's starting point.
    return yaml.safe_load((Path(__file__).parent / 'query_fit_recipe.yaml').read_text())


def test_parent_degree_uses_shared_backend_validation():
    model = original_query_fit()['model']
    for degree in (4, 8, 12):
        assert launcher.parent_degree({**model, 'attention_degree': degree}) == degree
    with pytest.raises(ValueError, match="Backend 'mosaic'"):
        launcher.parent_degree({**model, 'attention_degree': 13})
    with pytest.raises(ValueError, match='query-only parent'):
        launcher.parent_degree({**model, 'attention_coefficient_scale': 'none'})


@pytest.mark.parametrize('parent_id,mode', [
    ('r2b22f2ff6461', 'query_coefficient_generator'),
    ('r4339a245820e', 'static_coefficients'),
    ('r816ab0fbe119', 'free_per_example_query_coefficients'),
])
def test_head_scaled_parent_families_accept_only_compatible_fits(parent_id, mode):
    source = yaml.safe_load((Path(__file__).parents[1] / 'runs' / parent_id / 'config.yaml').read_text())
    assert launcher.parent_degree(source['model'], mode) == 4
    assert source['model']['qk_norm_head_scale'] is True
    if mode != 'query_coefficient_generator':
        with pytest.raises(ValueError, match='query-only parent'):
            launcher.parent_degree(source['model'], 'query_coefficient_generator')
    if mode == 'free_per_example_query_coefficients':
        with pytest.raises(ValueError, match='compatible 4B'):
            launcher.parent_degree(source['model'], 'static_coefficients')


@pytest.mark.parametrize('parameterization,sequence_count,degree,readout', [
    ('query_coefficient_generator', None, 4, 'diagonal'),
    ('query_coefficient_generator', 256, 8, 'diagonal'),
    ('query_coefficient_generator', 256, 8, 'dense'),
    ('free_per_example_query_coefficients', 256, 8, 'diagonal'),
    ('free_per_example_query_coefficients', 1, 4, 'diagonal'),
    ('query_coefficient_generator', 256, 4, 'diagonal'),
    ('static_coefficients', 256, 4, 'diagonal'),
    ('free_per_example_query_coefficients', 256, 4, 'diagonal'),
])
def test_prepare_preserves_parent_and_allocates_the_selected_fit(
        tmp_path, monkeypatch, parameterization, sequence_count, degree, readout):
    import hashlib
    import numpy as np
    from scaffolds.transformer.data import packed

    prior = original_query_fit()
    source = deepcopy(prior)
    for key in ('oracle', 'experiment_kind', 'condition', 'schema_version'):
        source.pop(key, None)
    source['model']['attention_degree'] = degree
    source['model']['gdn_polynomial_degree'] = 1
    if degree == 4 and sequence_count == 256:
        parent_id = {'query_coefficient_generator': 'r2b22f2ff6461',
                     'static_coefficients': 'r4339a245820e',
                     'free_per_example_query_coefficients': 'r816ab0fbe119'}[parameterization]
        source['model'] = yaml.safe_load((Path(__file__).parents[1] / 'runs' / parent_id / 'config.yaml').read_text())['model']
    source['data']['batch_size'] = 256
    manifest = tmp_path / 'eval-manifest.json'
    manifest.write_text('{}\n')
    source['data']['eval_manifest'] = str(manifest)
    parent = tmp_path / 'source/experiments/transformer_pp/runs/r49b7943aee1b'
    parent.mkdir(parents=True)
    (parent / 'config.yaml').write_text(yaml.safe_dump(source))
    (parent / 'best_validation.json').write_text(json.dumps({'step': 7500}))
    checkpoint = parent / 'checkpoints/7500'
    checkpoint.mkdir(parents=True)
    descriptor = {'path': str(checkpoint), 'step': 7500, 'sha256': 'c' * 64}
    monkeypatch.setattr(launcher, 'load_run', lambda _: {
        'status': 'complete', 'home_cluster': 'kennel', 'selected_checkpoint': descriptor})
    monkeypatch.setattr(launcher, 'validate_record', lambda *a, **kw: None)
    study = tmp_path / 'child/experiments/transformer_pp'
    topic = study / 'results/diagnostics/overfitting/coefficient_oracle'
    topic.mkdir(parents=True)
    old_entry = {'run_id': 'rd7bf9bc37cb6', 'role': 'query_generator_single_sequence',
                 'parent_run_id': 'r98d5895c02d0', 'sources': {'config.yaml': {'sha256': 'a' * 64}}}
    (topic / 'selection.yaml').write_text(yaml.safe_dump({'settings': {}, 'runs': [old_entry]}))
    (study / 'sweep.yaml').write_text(yaml.safe_dump({'data_manifests': {'eval': {'sha256': launcher.sha256(manifest)}}}))
    monkeypatch.setattr(launcher, 'STUDY', study)
    monkeypatch.setattr(launcher, 'TOPIC', topic)
    captured = {}
    def create_run(_, recipe, label):
        captured.update(recipe=recipe, label=label)
        return study / 'runs/r000000000004'
    monkeypatch.setattr(launcher, 'create_run', create_run)
    monkeypatch.setattr(launcher, 'update_record', lambda *a, **kw: None)
    monkeypatch.setattr(launcher, 'refresh_selection', lambda *a: None)
    monkeypatch.setattr(launcher, 'regenerate_index', lambda *a: None)
    count = sequence_count or 1
    batch = np.arange(count * 5, dtype=np.int32).reshape(count, 5)
    def packed_batch_iterator(manifest_path, *, batch_size, sequence_length, start_sequence):
        assert manifest_path == str(manifest)
        assert batch_size == count
        assert sequence_length == source['data']['sequence_length']
        assert start_sequence == 0
        return iter([batch])
    monkeypatch.setattr(packed, 'PackedBatchIterator', packed_batch_iterator)
    launcher.prepare(parent, parameterization=parameterization, sequence_count=sequence_count,
                     generator_readout=readout)
    recipe = captured['recipe']
    expected_model = deepcopy(source['model'])
    if parameterization == 'query_coefficient_generator' and sequence_count is not None:
        expected_model['attention_coefficient_readout'] = readout
    assert recipe['model'] == expected_model
    assert recipe['data']['batch_size'] == count
    assert recipe['train']['seed'] == source['train']['seed']
    for key in ('data_start', 'max_updates', 'initial_updates', 'plateau_window',
                'plateau_tolerance', 'confirmation_updates',
                'learning_rate_decay', 'beta1', 'beta2', 'epsilon', 'clip_norm'):
        assert recipe['oracle'][key] == prior['oracle'][key]
    assert recipe['oracle']['parameterization'] == parameterization
    assert recipe['oracle']['sequence_count'] == count
    if parameterization == 'query_coefficient_generator':
        suffix = 'single_sequence' if count == 1 else 'full_batch'
        prefix = 'dense_query_generator' if readout == 'dense' else 'query_generator'
        expected_role = launcher.fitting_role(f'{prefix}_{suffix}', degree)
        label_prefix = 'query-generator-dense' if readout == 'dense' else 'query-generator'
        expected_label = f'{label_prefix}-m{degree}-' + suffix.replace('_', '-')
        expected_hash = hashlib.sha256(batch.tobytes()).hexdigest()
        assert recipe['oracle']['batch_sha256'] == expected_hash
        assert recipe['oracle']['microbatch_size'] == min(count, 8)
        assert recipe['oracle']['learning_rate'] == prior['oracle']['learning_rate']
        assert recipe['oracle']['optimizer_initialization'].startswith('fresh zero moments')
        assert recipe['oracle']['generator_readout'] == readout
        assert recipe['oracle']['trainable_parameters'] == [
            'query_polynomial_bias',
            'query_polynomial_mixing' if readout == 'dense' else 'query_polynomial_amplitude',
            'query_polynomial_kernel',
        ]
        if count > 1:
            assert 'before one clip and Adam update' in recipe['oracle']['gradient_scaling']
    else:
        suffix = 'single_sequence' if count == 1 else 'full_batch'
        prefix = 'static' if parameterization == 'static_coefficients' else 'free'
        expected_role = launcher.fitting_role(f'{prefix}_coefficients_{suffix}', degree)
        expected_label = f'{prefix}-coefficients-m{degree}-' + suffix.replace('_', '-')
        assert recipe['oracle']['batch_sha256'] == hashlib.sha256(batch.tobytes()).hexdigest()
        assert recipe['oracle']['microbatch_size'] == recipe['train']['micro_batch_size'] == min(count, 8)
        assert recipe['oracle']['learning_rate'] == recipe['optim']['learning_rate'] == 0.01
        if parameterization == 'static_coefficients':
            assert recipe['oracle']['trainable_parameters'] == ['static_effective_monomial_coefficients']
            assert recipe['oracle']['static_initialization'] == 'checkpoint'
            assert recipe['oracle']['initialization'] == "restore the parent's exact effective static coefficients without averaging"
        else:
            assert recipe['oracle']['trainable_parameters'] == ['per_example_effective_monomial_coefficients']
            assert recipe['oracle']['initialization'] == "capture actual coefficients on each sequence's original forward pass"
            assert recipe['oracle']['gradient_scaling'] == 'per-sequence mean NLL, clipped independently per sequence'
        assert recipe['oracle']['optimizer_coordinates'] == 'factorial(order)*effective_monomial_coefficient'
        assert recipe['oracle']['frozen_parameters'].startswith('all pretrained parameters;')
    if source['model'].get('qk_norm_head_scale', False):
        expected_role += '_head_scale'
        expected_label += '-head-scale'
    assert captured['label'] == expected_label
    assert recipe['condition'] == f'coefficient_oracle/{expected_role}'
    assert recipe['oracle']['parent_checkpoint_sha256'] == descriptor['sha256']
    assert recipe['oracle']['parent_config_sha256'] == launcher.sha256(parent / 'config.yaml')
    selection = yaml.safe_load((topic / 'selection.yaml').read_text())
    assert selection['runs'][0] == old_entry
    assert selection['runs'][1]['role'] == expected_role


@pytest.mark.parametrize('returncode,stdout,stderr,expected', [
    (0, 'RUNNING\n', '', 'RUNNING'),
    (0, '', '', ''),
    (1, '', 'slurm_load_jobs error: Invalid job id specified', ''),
])
def test_live_job_lookup_accepts_expired_ids(monkeypatch, returncode, stdout, stderr, expected):
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], returncode, stdout=stdout, stderr=stderr))
    assert launcher.live_job_state('224999') == expected


def test_live_job_lookup_preserves_scheduler_errors(monkeypatch):
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], 1, stdout='', stderr='scheduler unavailable'))
    with pytest.raises(RuntimeError, match='scheduler unavailable'):
        launcher.live_job_state('224999')


@pytest.mark.parametrize('gpu_type,gres', [(None, 'gpu:1'), ('B300', 'gpu:b300:1'), ('H100', 'gpu:h100:1')])
@pytest.mark.parametrize('qualify_only', [False, True])
def test_submit_selects_gpu_without_changing_run_recipe(tmp_path, monkeypatch, gpu_type, gres, qualify_only):
    monkeypatch.setattr(launcher, 'STUDY', tmp_path)
    monkeypatch.setattr(launcher, 'load_run', lambda _: {'status': 'planned', 'home_cluster': 'kennel'})
    monkeypatch.setattr(launcher, 'validate_record', lambda *a, **kw: None)
    monkeypatch.setattr(launcher, 'changes', lambda *a: [])
    monkeypatch.setattr(launcher, 'frozen_hash', lambda *a: 'f' * 64)
    monkeypatch.setattr(launcher, 'regenerate_index', lambda *a: None)
    captured = {}
    def check_output(command, **kwargs):
        if command[0] == 'git':
            return 'c' * 40 + '\n'
        captured['command'] = command
        return '225100\n'
    monkeypatch.setattr(launcher.subprocess, 'check_output', check_output)
    monkeypatch.setattr(launcher, 'update_record', lambda run, changes: captured.update(record=changes))
    launcher.submit('r000000000008', partition='long', time_limit='06:00:00', gpu_type=gpu_type,
                    qualify_only=qualify_only)
    assert f'--gres={gres}' in captured['command']
    assert '--partition=long' in captured['command']
    assert '--time=06:00:00' in captured['command']
    assert captured['record']['submission']['command'] == captured['command']
    assert captured['record']['submission']['slurm_job_id'] == '225100'
    assert ('--qualify-only' in captured['command']) == qualify_only
    assert captured['record']['submission']['phase'] == ('qualification' if qualify_only else 'training')


@pytest.mark.parametrize('count', (512, 1024))
def test_generator_fits_a_larger_fixed_batch_named_by_its_size(count):
    assert fitting_scope('query_coefficient_generator', 256, count) == (count, f'query_generator_{count}_sequences')


@pytest.mark.parametrize('count', (2, 255, 260, 1020))
def test_generator_rejects_other_batch_sizes(count):
    with pytest.raises(ValueError):
        fitting_scope('query_coefficient_generator', 256, count)


QUERY_KEY_PARENT = 'r501f5cca0d4c'
QUERY_KEY_TRAINABLE = ['query_polynomial_bias', 'query_polynomial_amplitude', 'query_polynomial_kernel',
                       'key_polynomial_kernel', 'key_polynomial_amplitude']


def query_key_parent():
    return yaml.safe_load((Path(__file__).parents[1] / 'runs' / QUERY_KEY_PARENT / 'config.yaml').read_text())


def test_query_key_parent_is_a_fused_generator_parent_only():
    model = query_key_parent()['model']
    assert model['attention_coefficient_conditioning'] == 'query_key' and model['qk_norm_head_scale'] is True
    assert launcher.parent_degree(model) == 4
    with pytest.raises(ValueError, match='compatible 4B'):
        launcher.parent_degree({**model, 'attention_backend': 'mosaic'})
    for mode in ('static_coefficients', 'free_per_example_query_coefficients'):
        with pytest.raises(ValueError, match='compatible 4B'):
            launcher.parent_degree(model, mode)


def test_prepare_keeps_a_native_query_key_parent_and_trains_both_generators(tmp_path, monkeypatch):
    import numpy as np
    from scaffolds.transformer.data import packed

    source = query_key_parent()
    manifest = tmp_path / 'eval-manifest.json'
    manifest.write_text('{}\n')
    source['data']['eval_manifest'] = str(manifest)
    parent = tmp_path / 'source/experiments/transformer_pp/runs' / QUERY_KEY_PARENT
    (parent / 'checkpoints/7500').mkdir(parents=True)
    (parent / 'config.yaml').write_text(yaml.safe_dump(source))
    (parent / 'best_validation.json').write_text(json.dumps({'step': 7500}))
    descriptor = {'path': str(parent / 'checkpoints/7500'), 'step': 7500, 'sha256': 'c' * 64}
    monkeypatch.setattr(launcher, 'load_run', lambda _: {
        'status': 'complete', 'home_cluster': 'orcd', 'selected_checkpoint': descriptor})
    monkeypatch.setattr(launcher, 'validate_record', lambda *a, **kw: None)
    study = tmp_path / 'child/experiments/transformer_pp'
    topic = study / 'results/diagnostics/overfitting/coefficient_oracle'
    topic.mkdir(parents=True)
    (topic / 'selection.yaml').write_text(yaml.safe_dump({'settings': {}, 'runs': []}))
    (study / 'sweep.yaml').write_text(yaml.safe_dump({'data_manifests': {'eval': {'sha256': launcher.sha256(manifest)}}}))
    monkeypatch.setattr(launcher, 'STUDY', study)
    monkeypatch.setattr(launcher, 'TOPIC', topic)
    captured = {}
    monkeypatch.setattr(launcher, 'create_run', lambda _, recipe, label: captured.update(recipe=recipe, label=label)
                        or study / 'runs/r000000000009')
    monkeypatch.setattr(launcher, 'update_record', lambda run, changes: captured.update(record=changes))
    monkeypatch.setattr(launcher, 'refresh_selection', lambda *a: None)
    monkeypatch.setattr(launcher, 'regenerate_index', lambda *a: None)
    batch = np.arange(256 * 5, dtype=np.int32).reshape(256, 5)
    monkeypatch.setattr(packed, 'PackedBatchIterator', lambda *a, **kw: iter([batch]))
    with pytest.raises(ValueError, match='completed kennel run'):
        launcher.prepare(parent, sequence_count=256)
    launcher.prepare(parent, sequence_count=256, home_cluster='orcd')
    recipe = captured['recipe']
    assert recipe['model'] == source['model']
    assert recipe['oracle']['trainable_parameters'] == QUERY_KEY_TRAINABLE
    assert 'per-head QK score scales' in recipe['oracle']['frozen_parameters']
    assert recipe['oracle']['initialization'].startswith('restore the trained query and key')
    assert recipe['oracle']['microbatch_size'] == 8 and recipe['oracle']['sequence_count'] == 256
    assert recipe['oracle']['learning_rate'] == 0.001 and recipe['oracle']['max_updates'] == 10000
    assert recipe['condition'] == 'coefficient_oracle/query_key_generator_full_batch_m4_head_scale'
    assert captured['label'] == 'query-key-generator-m4-full-batch-head-scale'
    assert captured['record']['home_cluster'] == 'orcd'
    assert 'key-conditioned' in captured['record']['coefficient_sharing']
    with pytest.raises(ValueError, match='diagonal readout'):
        launcher.prepare(parent, sequence_count=256, home_cluster='orcd', generator_readout='dense')


def test_orcd_submission_requests_a_typed_hopper_and_requeues(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, 'STUDY', tmp_path)
    monkeypatch.setattr(launcher, 'validate_record', lambda *a, **kw: None)
    monkeypatch.setattr(launcher, 'changes', lambda *a: [])
    monkeypatch.setattr(launcher, 'frozen_hash', lambda *a: 'f' * 64)
    monkeypatch.setattr(launcher, 'regenerate_index', lambda *a: None)
    captured = {}
    def check_output(command, **kwargs):
        if command[0] == 'git':
            return 'c' * 40 + '\n'
        captured['command'] = command
        return '27200000\n'
    monkeypatch.setattr(launcher.subprocess, 'check_output', check_output)
    monkeypatch.setattr(launcher, 'update_record', lambda run, changes: captured.update(record=changes))
    monkeypatch.setattr(launcher, 'load_run', lambda _: {'status': 'planned', 'home_cluster': 'kennel'})
    with pytest.raises(ValueError, match='belongs to kennel'):
        launcher.submit('r000000000009', cluster='orcd')
    monkeypatch.setattr(launcher, 'load_run', lambda _: {'status': 'planned', 'home_cluster': 'orcd'})
    with pytest.raises(ValueError, match='GPU type'):
        launcher.submit('r000000000009', cluster='orcd', gpu_type='B300')
    launcher.submit('r000000000009', cluster='orcd')
    command = captured['command']
    for argument in ('--partition=mit_preemptable,mit_normal_gpu', '--gres=gpu:h100:1', '--time=12:00:00',
                     '--time-min=06:00:00', '--cpus-per-task=16', '--mem=128G',
                     '--exclude=node3009,node3209,node1708,node1709,node4508,node2801,node3409'):
        assert argument in command
    assert command[-5].endswith('coefficient_fit_orcd.sbatch') and command[-1] == 'H100'
    submission = captured['record']['submission']
    assert submission['cluster'] == 'orcd' and submission['gpu_type'] == 'H100' and submission['gpu_count'] == 1
