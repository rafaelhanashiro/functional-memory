"""CPU-only driver/provenance tests; native CUDA recurrence is tested separately."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scaffolds.state_rank import compare_deltanet as driver, compare_random as common, spectrum


def metrics_from_eigenvalues(eigenvalues):
    return {key: value for key, value in spectrum.spectral_metrics(
        np.asarray(eigenvalues, dtype=float), np.zeros(np.asarray(eigenvalues).shape[:-1])
    ).items() if key in driver.METRICS}


def baseline_case(eigenvalues=None):
    if eigenvalues is None:
        eigenvalues = np.asarray([[[1., 0., 0.], [1., 0., 0.]],
                                  [[.8, .2, 0.], [.7, .3, 0.]],
                                  [[.6, .3, .1], [.5, .4, .1]]])
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    result = metrics_from_eigenvalues(eigenvalues)
    reference = {key: value[None].copy() for key, value in result.items()}
    reference['normalized_eigenvalues'] = eigenvalues[None].copy()
    return eigenvalues, result, reference


def test_baseline_exact_match_and_float_kernel_noise():
    eig, result, reference = baseline_case()
    exact = driver.compare_baseline(result, eig, reference, 0)
    assert exact['passed'] and exact['normalized_spectrum_max_absolute_error'] == 0
    for key in ('threshold_rank', 'energy_rank90', 'energy_rank99'):
        assert exact[key + '_mismatches'] == 0
    noisy = eig.copy()
    noisy[-1, 0, :2] += [1e-6, -1e-6]
    assert driver.compare_baseline(metrics_from_eigenvalues(noisy), noisy, reference, 0)['passed']


@pytest.mark.parametrize('metric', driver.METRICS)
def test_baseline_rejects_material_metric_drift(metric):
    eig, result, reference = baseline_case()
    result[metric] = result[metric].copy()
    result[metric][-1, 0] += 1
    assert not driver.compare_baseline(result, eig, reference, 0)['passed']


def test_baseline_rejects_material_spectrum_drift():
    eig, result, reference = baseline_case()
    eig[-1, 0, :2] += [.01, -.01]
    assert not driver.compare_baseline(result, eig, reference, 0)['passed']


@pytest.mark.parametrize('energy', [.9, .99])
def test_only_true_near_boundary_energy_rank_crossings_are_tolerated(energy):
    eig = np.array([[[energy - 1e-6, 1 - energy + 1e-6]]])
    _, _, reference = baseline_case(eig)
    changed = np.array([[[energy + 1e-6, 1 - energy - 1e-6]]])
    result = metrics_from_eigenvalues(changed)
    check = driver.compare_baseline(result, changed, reference, 0)
    metric = 'energy_rank90' if energy == .9 else 'energy_rank99'
    assert check['passed']
    assert check[metric + '_boundary_tolerated_mismatches'] == 1
    # Merely changing a count near the boundary without a spectral crossing fails.
    unchanged = metrics_from_eigenvalues(eig)
    unchanged[metric][0, 0] = result[metric][0, 0]
    assert not driver.compare_baseline(unchanged, eig, reference, 0)['passed']


def test_threshold_tolerance_requires_a_small_actual_singular_boundary_crossing():
    ratio = spectrum.RANK_TOL
    old = np.array([[[1., (ratio - 1e-7)**2]]]);old /= old.sum(axis=-1, keepdims=True)
    new = np.array([[[1., (ratio + 1e-7)**2]]]);new /= new.sum(axis=-1, keepdims=True)
    _, _, reference = baseline_case(old)
    check = driver.compare_baseline(metrics_from_eigenvalues(new), new, reference, 0)
    assert check['passed'] and check['threshold_rank_boundary_tolerated_mismatches'] == 1
    far = np.array([[[1., (ratio + 1e-5)**2]]]);far /= far.sum(axis=-1, keepdims=True)
    assert not driver.compare_baseline(metrics_from_eigenvalues(far), far, reference, 0)['passed']


def test_loading_info_allows_only_historical_D_and_omitted_readout():
    loading = {'missing_keys': [], 'mismatched_keys': [], 'unexpected_keys': ['lm_head.weight', 'layers.0.attn.D'], 'error_msgs': []}
    assert driver.validate_loading_info(loading, 2) == ['layers.0.attn.D', 'lm_head.weight']
    for key in ('layers.2.attn.D', 'layers.0.attn.q_proj.weight'):
        with pytest.raises(RuntimeError, match='exact loading'):
            driver.validate_loading_info(loading | {'unexpected_keys': [key]}, 2)
    with pytest.raises(RuntimeError, match='exact loading'):
        driver.validate_loading_info(loading | {'missing_keys': ['norm.weight']}, 2)


def test_analyze_model_stacks_input_axis_checks_full_baseline_then_discards_it(monkeypatch):
    eig, base_metrics, reference = baseline_case()
    reference = {key: np.repeat(value, 2, axis=0) for key, value in reference.items()}
    calls = []
    def measure(model, input_ids, layers, selected, *, full_spectrum_layers, svd_workers):
        calls.append((input_ids.tolist(), full_spectrum_layers, svd_workers))
        arrays = {key: np.stack([value, value]) for key, value in base_metrics.items()}
        arrays.update(layers=np.array(layers), head_indices=np.arange(2), prefixes=np.arange(1, 4),
                      spectrum_prefixes=np.array(selected), normalized_eigenvalues=np.stack([eig[np.array(selected)-1]]*2),
                      prefix_check_relative_error=np.zeros((2, len(selected), 2)))
        return arrays, {'full_spectra': {12: eig.copy()}, 'maximum_prefix_relative_error': 1e-7}
    monkeypatch.setattr(driver, '_measure_input', measure)
    inputs = [{'token_ids': [1, 2, 3]}, {'token_ids': [1, 4, 5]}]
    arrays, validation = driver.analyze_model(SimpleNamespace(), inputs, [0, 12], [1, 3],
                                             baseline=reference, baseline_layer=12, svd_workers=4)
    assert calls == [([[1, 2, 3]], (12,), 4), ([[1, 4, 5]], (12,), 4)]
    for key in driver.METRICS:
        assert arrays[key].shape == (2, 2, 3, 2)
        np.testing.assert_array_equal(arrays[key][1], reference[key])
    assert arrays['normalized_eigenvalues'].shape == (2, 2, 2, 2, 3)
    assert arrays['prefix_check_relative_error'].shape == (2, 2, 2, 2)
    assert not {'keys','values','key_norms','value_norms','full_spectra','singular_values'}.intersection(arrays)
    assert all(check['passed'] for check in validation['baseline_checks'])
    assert validation['maximum_prefix_relative_error'] == 1e-7
    json.dumps(validation, allow_nan=False)
    reference['participation_ratio'][0, -1, 0] *= 2
    with pytest.raises(ValueError, match='baseline reproduction failed'):
        driver.analyze_model(SimpleNamespace(), inputs, [0, 12], [1, 3], baseline=reference, baseline_layer=12)


@pytest.fixture
def project(tmp_path, monkeypatch):
    baseline = tmp_path / 'baseline';baseline.mkdir()
    (tmp_path/'uv.lock').write_text('version=1\n')
    metadata = {'model': 'test/gdn', 'model_revision': 'b'*40, 'layer': 0, 'tokens': 3,
                'model_dtype':'float32','analysis_dtype':'float64','attention_implementation':'fused_recurrent',
                'threshold_rank_singular_tolerance': spectrum.RANK_TOL}
    (baseline/'metadata.json').write_text(json.dumps(metadata))
    (baseline/'inputs.json').write_text(json.dumps([{'token_ids':[1,2,3]}]))
    np.savez_compressed(baseline/'spectra.npz', token_ids=np.array([[1,2,3]]))
    config = {'schema_version':1, 'model_id':'test/gdn','model_revision':'b'*40,
              'baseline_directory':'baseline','baseline_layer':0,'layers':[0,1], 'seeds':[0,1],
              'tokens':3,'input_count':1,'spectrum_prefixes':[1,3], 'svd_workers':4,
              'model_dtype':'float32','analysis_dtype':'float64','attention_implementation':'fused_recurrent',
              'allow_tf32':False, 'output_directory':'comparison'}
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    monkeypatch.setattr(common, 'source_commit', lambda:'a'*40)
    return tmp_path,path,config,baseline


def test_manifest_is_cpu_safe_idempotent_and_pins_inputs(project):
    root,config_path,config,baseline=project
    path=driver.prepare_manifest(config_path,root)
    before=path.read_bytes();manifest=json.loads(before)
    assert manifest['experiment']==driver.EXPERIMENT
    assert manifest['config']['attention_implementation']=='fused_recurrent'
    assert manifest['checkpoint_retention']=='none'
    assert (path.parent/'inputs.json').read_bytes()==(baseline/'inputs.json').read_bytes()
    assert manifest['baseline_tolerances']==driver.BASELINE_TOLERANCES
    driver.validate_manifest(manifest)
    assert driver.prepare_manifest(config_path,root)==path and path.read_bytes()==before
    config['seeds']=[2,3];config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='different source/configuration/input identity'):
        driver.prepare_manifest(config_path,root)


@pytest.mark.parametrize('drift',['inputs','baseline','lock','tolerances'])
def test_manifest_rejects_provenance_and_tolerance_drift(project,drift):
    root,config_path,_,baseline=project
    path=driver.prepare_manifest(config_path,root);manifest=json.loads(path.read_text())
    if drift=='tolerances':
        manifest['baseline_tolerances']['normalized_spectrum_max_absolute_error']=1.
    else:
        target={'inputs':path.parent/'inputs.json','baseline':baseline/'spectra.npz','lock':root/'uv.lock'}[drift]
        target.write_bytes(target.read_bytes()+b'changed')
    with pytest.raises(ValueError):driver.validate_manifest(manifest)


def test_completed_condition_checks_archive_before_cuda_or_model_loading(project):
    root,config_path,_,_=project
    manifest_path=driver.prepare_manifest(config_path,root);manifest=json.loads(manifest_path.read_text())
    output=manifest_path.parent/'random-seed0';output.mkdir()
    np.savez_compressed(output/'metrics.npz', value=np.array([1]))
    metadata={'identity':manifest['identity'],'status':'complete','metrics_sha256':common.sha256_file(output/'metrics.npz')}
    (output/'metadata.json').write_text(json.dumps(metadata))
    assert driver.run_condition(manifest_path,'random-seed0',require_cuda=True)==metadata
    with (output/'metrics.npz').open('ab') as handle:handle.write(b'changed')
    with pytest.raises(ValueError,match='Completed metrics were changed'):
        driver.run_condition(manifest_path,'random-seed0')


def test_baseline_zero_states_match_without_infinite_minus_infinite():
    zero = np.zeros((3, 2, 4))
    _, result, reference = baseline_case(zero)
    check = driver.compare_baseline(result, zero, reference, 0)
    assert check['passed'] and check['log_trace_max_absolute_error'] == 0.
    result['log_trace'][0, 0] = 0.
    mismatch = driver.compare_baseline(result, zero, reference, 0)
    assert not mismatch['passed'] and np.isinf(mismatch['log_trace_max_absolute_error'])
