"""Axis sentinels test aggregation; these fixtures are not simulated physical states."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.state_rank.plot_comparison import Condition
from experiments.state_rank.unified_metrics import (
    ModelAnalysis, RANK_KEYS, comparison_statistics, depth_curves, head_curves,
    load_analysis, summary_rows, write_summaries,
)


def sentinel_arrays(offset=0., layers=(0, 1, 2), tokens=4, heads=2, inputs=2, width=4):
    l, i, t, h = np.ogrid[:len(layers), :inputs, :tokens, :heads]
    values = 1 + offset + l*1000 + i*100 + t*10 + h
    arrays = {metric: np.broadcast_to(values + index*10000, (len(layers), inputs, tokens, heads)).copy().astype(float)
              for index, metric in enumerate((*RANK_KEYS, 'log_trace'))}
    arrays.update(layers=np.array(layers), head_indices=np.arange(heads)*2,
                  prefixes=np.arange(1, tokens+1), spectrum_prefixes=np.array([1, tokens]),
                  token_ids=np.arange(inputs*tokens).reshape(inputs, tokens))
    eigen = np.zeros((len(layers), inputs, 2, heads, width));eigen[..., 0] = 1.
    arrays['normalized_eigenvalues'] = eigen
    return arrays


def condition(name='pretrained', weights='pretrained', seed=None, offset=0., **kwargs):
    arrays = sentinel_arrays(offset, **kwargs)
    return Condition(name, weights, seed, {'model_config': {'num_hidden_layers': int(arrays['layers'][-1])+1}}, arrays)


def test_head_curves_reduce_sequences_only_and_use_actual_layer_ids():
    current = condition(layers=(1, 4, 9))
    curves = head_curves(current, 4)
    expected_mean = np.array([[1051.,1052.], [1061.,1062.], [1071.,1072.], [1081.,1082.]])
    np.testing.assert_array_equal(curves['participation_ratio']['mean'], expected_mean)
    np.testing.assert_array_equal(curves['participation_ratio']['min'], expected_mean-50)
    np.testing.assert_array_equal(curves['participation_ratio']['max'], expected_mean+50)
    assert curves['participation_ratio']['mean'].shape == (4,2)
    for index, metric in enumerate(RANK_KEYS):
        np.testing.assert_array_equal(curves[metric]['mean'], expected_mean + index*10000)
    # Saved normalized spectra are all rank-one, deliberately unlike sentinel
    # metrics: aggregation must never recompute metrics from an averaged spectrum.
    assert curves['participation_ratio']['mean'][0,0] == 1051.
    with pytest.raises(ValueError, match='not uniquely present'):
        head_curves(current, 0)


def test_optional_single_input_is_not_averaged_and_returned_arrays_are_independent():
    current = condition()
    curves = head_curves(current, 1, input_index=1)
    expected = np.array([[1101.,1102.], [1111.,1112.], [1121.,1122.], [1131.,1132.]])
    for statistic in ('mean','min','max'):
        np.testing.assert_array_equal(curves['participation_ratio'][statistic], expected)
    curves['participation_ratio']['mean'][0,0] = -99.
    assert current.arrays['participation_ratio'][1,1,0,0] == 1101.
    assert curves['participation_ratio']['min'][0,0] == 1101.
    for invalid in (-1,2,True):
        with pytest.raises(ValueError,match='input_index'):
            head_curves(current, 1, input_index=invalid)


def test_each_random_model_keeps_its_own_head_curves():
    first = condition('random-seed0','random',0,offset=0)
    second = condition('random-seed1','random',1,offset=700)
    np.testing.assert_array_equal(head_curves(first,0)['participation_ratio']['mean'][0], [51.,52.])
    np.testing.assert_array_equal(head_curves(second,0)['participation_ratio']['mean'][0], [751.,752.])


def test_depth_curves_use_final_token_then_input_head_average():
    curves = depth_curves(condition())
    np.testing.assert_array_equal(curves['participation_ratio'], [81.5,1081.5,2081.5])
    np.testing.assert_array_equal(curves['threshold_rank'], np.array([81.5,1081.5,2081.5])+10000)
    assert curves['participation_ratio'].shape == (3,)


def write_package(root, *, second_seed=True, model_depth=3, layers=(0,1,2)):
    root.mkdir()
    conditions = [condition(layers=layers),condition('random-seed0','random',0,offset=100,layers=layers)]
    if second_seed:
        conditions.append(condition('random-seed1','random',1,offset=200,layers=layers))
    inputs = [{'title':f'Input {i}','article_index':i,'body_sha256':f'article-{i}', 'token_ids':ids.tolist()}
              for i,ids in enumerate(conditions[0].arrays['token_ids'])]
    raw=json.dumps(inputs).encode();(root/'inputs.json').write_bytes(raw)
    manifest={'identity':'immutable-fixture','inputs_sha256':hashlib.sha256(raw).hexdigest(),
              'config':{'comparison_layer':layers[1],'layers':list(layers),'tokens':4,'input_count':2,
                        'seeds':[0,1] if second_seed else [0],'spectrum_prefixes':[1,4]},
              'conditions':[{'name':c.name,'weights':c.weights,'seed':c.seed} for c in conditions]}
    (root/'manifest.json').write_text(json.dumps(manifest))
    for c in conditions:
        path=root/c.name;path.mkdir()
        np.savez_compressed(path/'metrics.npz',**c.arrays)
        metadata={'status':'complete','identity':manifest['identity'],'weights':c.weights,'seed':c.seed,
                  'metrics_sha256':hashlib.sha256((path/'metrics.npz').read_bytes()).hexdigest(),
                  'model_config':{'num_hidden_layers':model_depth}}
        (path/'metadata.json').write_text(json.dumps(metadata))
    return manifest


def test_load_checks_identity_inputs_and_depth_and_never_mutates_source(tmp_path):
    source=tmp_path/'source';write_package(source)
    before={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    analysis=load_analysis(source,'llama','Llama')
    assert analysis.layer==1 and analysis.width==4 and analysis.model_depth==3
    assert analysis.input_count==2 and analysis.head_count==2
    np.testing.assert_array_equal(analysis.head_indices,[0,2])
    np.testing.assert_array_equal(analysis.relative_depth,[0,.5,1])
    assert not analysis.conditions[0].arrays['participation_ratio'].flags.writeable
    counts=write_summaries(tmp_path/'derived',[analysis])
    assert counts=={'per_head_terminal_rows':6,'depth_summary_rows':9}
    for path in (tmp_path/'derived').glob('*.csv'):
        assert b'\r\n' not in path.read_bytes()  # Git enforces LF for result CSVs.
    assert before=={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    with (tmp_path/'derived/depth_summary.csv').open() as stream:
        row=next(csv.DictReader(stream))
    assert float(row['participation_ratio'])==81.5
    assert float(row['participation_ratio_fraction'])==81.5/4
    with (tmp_path/'derived/per_head_terminal.csv').open() as stream:
        row=next(csv.DictReader(stream))
    assert int(row['head'])==0 and int(row['layer'])==1
    assert float(row['participation_ratio_mean'])==1081.
    assert float(row['participation_ratio_min'])==1031.
    assert float(row['participation_ratio_max'])==1131.
    with pytest.raises(ValueError,match='outside original'):
        write_summaries(source,[analysis])


@pytest.mark.parametrize('drift',['input_hash','input_tokens','partial','missing_layer','config_layers','checksum','seed','model_depth'])
def test_load_rejects_mismatched_or_incomplete_packages(tmp_path,drift):
    source=tmp_path/'source';manifest=write_package(source)
    metadata_path=source/'random-seed0/metadata.json'
    metadata=json.loads(metadata_path.read_text())
    kwargs={}
    if drift in ('input_hash','input_tokens'):
        records=json.loads((source/'inputs.json').read_text());records[0]['token_ids'][0]=99
        raw=json.dumps(records).encode();(source/'inputs.json').write_bytes(raw)
        if drift=='input_tokens':
            manifest['inputs_sha256']=hashlib.sha256(raw).hexdigest()
            (source/'manifest.json').write_text(json.dumps(manifest))
    elif drift=='partial':
        metadata['status']='running'
    elif drift=='missing_layer':
        kwargs['layer']=8
    elif drift=='config_layers':
        manifest['config']['layers']=[0,1]
        (source/'manifest.json').write_text(json.dumps(manifest))
    elif drift=='checksum':
        with (source/'random-seed0/metrics.npz').open('ab') as stream:stream.write(b'changed')
    elif drift=='seed':
        metadata['seed']=15
    elif drift=='model_depth':
        metadata['model_config']['num_hidden_layers']=4
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):load_analysis(source,'test','Test',**kwargs)


def test_relative_depth_uses_full_model_not_measured_subset(tmp_path):
    source=tmp_path/'source';write_package(source,model_depth=10,layers=(1,4,9))
    analysis=load_analysis(source,'subset','Subset')
    np.testing.assert_allclose(analysis.relative_depth,[1/9,4/9,1])
    assert analysis.layer==4


def test_comparison_statistics_compare_model_level_seed_means_without_head_pairing(tmp_path):
    conditions=[condition(),condition('random-seed0','random',0),condition('random-seed1','random',1)]
    for current,targets in zip(conditions,([3.,2.,7.],[1.,2.,8.],[3.,2.,10.]),strict=True):
        for metric in RANK_KEYS:
            for layer,value in enumerate(targets):current.arrays[metric][layer]=value
    analysis=ModelAnalysis('test','Test',tmp_path,{},conditions,1,[{},{}])
    stats=comparison_statistics(analysis)
    for metric in RANK_KEYS:
        assert stats[metric]['pretrained_above_random_mean']==[0]
        assert stats[metric]['pretrained_below_random_mean']==[2]
        assert stats[metric]['pretrained_equal_random_mean']==[1]
        assert stats[metric]['random_mean']==[2.,2.,9.]
        assert stats[metric]['random_by_condition']['random-seed0']==[1.,2.,8.]
    # Shuffling heads inside one random model has no effect on depth summaries.
    conditions[1].arrays['participation_ratio']=conditions[1].arrays['participation_ratio'][...,::-1]
    assert comparison_statistics(analysis)==stats


def test_summary_rows_keep_architectures_conditions_and_head_ids_separate(tmp_path):
    first=ModelAnalysis('llama','Llama',tmp_path/'llama',{},[condition()],1,[{},{}])
    second_condition=condition(width=8)
    second_condition.arrays['token_ids']+=100  # Native tokenizer may differ across architectures.
    second=ModelAnalysis('gdn','Gated DeltaNet',tmp_path/'gdn',{},[second_condition],1,[{},{}])
    rows=summary_rows([first,second])
    assert len(rows['per_head_terminal'])==4 and len(rows['depth_summary'])==6
    assert {row['head'] for row in rows['per_head_terminal']}=={0,2}
    assert rows['depth_summary'][0]['participation_ratio_fraction']==81.5/4
    assert rows['depth_summary'][3]['participation_ratio_fraction']==81.5/8
    with pytest.raises(ValueError,match='keys must be distinct'):
        summary_rows([first,first])


def grouped_fixture(tmp_path):
    """Seed-specific input spread separates seed bands from pooled sample bands."""
    models=[condition(layers=(1,4,9)), condition('random-seed0','random',0,layers=(1,4,9)),
            condition('random-seed1','random',1,layers=(1,4,9))]
    # For every metric use independent layer/time/head offsets, while choosing
    # sequence values whose seed range is narrower than their pooled range.
    for current, input_values in zip(models, ([20.,60.], [0.,100.], [75.,85.]), strict=True):
        for metric_index, metric in enumerate((*RANK_KEYS,'log_trace')):
            for layer_index in range(3):
                for input_index, base in enumerate(input_values):
                    for token in range(4):
                        current.arrays[metric][layer_index,input_index,token,:] = (
                            base + layer_index*1000 + token*10 + np.array([0.,100.]) + metric_index*10000
                        )
    return ModelAnalysis('test','Test',tmp_path,{},models,4,[{},{}])


def test_grouped_heads_average_sequences_then_seeds_and_keep_head_axis(tmp_path):
    from experiments.state_rank.unified_metrics import grouped_curves
    analysis=grouped_fixture(tmp_path)
    groups=grouped_curves(analysis,view='heads')
    base=np.array([[1000.,1100.],[1010.,1110.],[1020.,1120.],[1030.,1130.]])
    for metric_index,metric in enumerate(RANK_KEYS):
        expected=base+metric_index*10000
        pretrained,random=groups['pretrained'][metric],groups['random'][metric]
        np.testing.assert_array_equal(pretrained['mean'],expected+40)
        np.testing.assert_array_equal(pretrained['min'],expected+20)
        np.testing.assert_array_equal(pretrained['max'],expected+60)
        np.testing.assert_array_equal(random['mean'],expected+65)
        # Seed 0 averages inputs 0/100 to 50; seed 1 averages 75/85 to 80.
        np.testing.assert_array_equal(random['min'],expected+50)
        np.testing.assert_array_equal(random['max'],expected+80)
        assert random['mean'].shape==(4,2)
    assert groups['random']['participation_ratio']['min'][0,0] != 1000.  # Not pooled input minimum.
    assert groups['random']['participation_ratio']['max'][0,0] != 1100.  # Not pooled input maximum.
    # The stored spectra are rank one; these figures still average raw scalar
    # measurements instead of deriving fresh ranks from an averaged spectrum.
    assert groups['random']['participation_ratio']['mean'][0,0]==1065.


def test_grouped_layer_curves_average_heads_and_sequences_before_seed_range(tmp_path):
    from experiments.state_rank.unified_metrics import grouped_curves
    groups=grouped_curves(grouped_fixture(tmp_path),view='layers')
    base=np.array([[0.,10.,20.,30.],[1000.,1010.,1020.,1030.],[2000.,2010.,2020.,2030.]])
    pretrained,random=groups['pretrained']['participation_ratio'],groups['random']['participation_ratio']
    # Each head mean adds 50, then the pretrained sequence mean adds 40.
    np.testing.assert_array_equal(pretrained['mean'],base+90)
    np.testing.assert_array_equal(pretrained['min'],base+70)
    np.testing.assert_array_equal(pretrained['max'],base+110)
    np.testing.assert_array_equal(random['mean'],base+115)
    np.testing.assert_array_equal(random['min'],base+100)
    np.testing.assert_array_equal(random['max'],base+130)
    assert random['mean'].shape==(3,4)
    # Terminal depth statistics remain the identical mean of per-condition raw
    # aggregates; grouping has not averaged over token positions.
    analysis=grouped_fixture(tmp_path)
    expected=np.stack([depth_curves(c)['participation_ratio'] for c in analysis.conditions if c.weights=='random']).mean(axis=0)
    np.testing.assert_array_equal(random['mean'][:,-1],expected)


def test_grouped_curves_are_seed_order_invariant_and_leave_audit_data_unchanged(tmp_path):
    from experiments.state_rank.unified_metrics import grouped_curves
    analysis=grouped_fixture(tmp_path)
    audit_before=summary_rows([analysis])
    arrays_before=[{key:value.copy() for key,value in c.arrays.items()} for c in analysis.conditions]
    first=grouped_curves(analysis)
    analysis.conditions[1:]=analysis.conditions[:0:-1]
    reordered=grouped_curves(analysis)
    for group in first:
        for metric in first[group]:
            for statistic in first[group][metric]:
                np.testing.assert_array_equal(first[group][metric][statistic],reordered[group][metric][statistic])
    analysis.conditions[1:]=analysis.conditions[:0:-1]
    first['random']['participation_ratio']['mean'][:]=-99
    assert summary_rows([analysis])==audit_before
    for current,before in zip(analysis.conditions,arrays_before,strict=True):
        for key,value in current.arrays.items():np.testing.assert_array_equal(value,before[key])


def test_grouped_curves_reject_unknown_view_or_missing_weight_group(tmp_path):
    from experiments.state_rank.unified_metrics import grouped_curves
    analysis=grouped_fixture(tmp_path)
    with pytest.raises(ValueError,match='view must be'):
        grouped_curves(analysis,view='tokens')
    analysis.conditions=analysis.conditions[:1]
    with pytest.raises(ValueError,match='one pretrained model and random controls'):
        grouped_curves(analysis)
