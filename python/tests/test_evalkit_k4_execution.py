"""Execution amendments preserve the scientific protocol; live rows preserve denominators."""
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]

def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'python/scripts'/f'{name}.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

runner = module('evalkit_k4_full_regression')
live = module('evalkit_k4_live_report')

def test_execution_amendment_cannot_change_science():
    protocol = dict(workers=3, output='/run')
    identity = dict(protocol_sha256='protocol', runner_sha256='runner')
    amendment = dict(version='k4-execution-amendment-1', protocol_sha256='protocol',
                     original_runner_sha256='runner', output='/run', max_workers=8, authorization='user')
    assert runner.execution_workers(protocol, 'protocol', identity, None) == 3
    assert runner.execution_workers(protocol, 'protocol', identity, amendment) == 8
    for changes in ({'max_workers':9}, {'protocol_sha256':'other'}, {'output':'/other'},
                    {'original_runner_sha256':'other'}, {'fit_config':'other'}):
        with pytest.raises(ValueError):
            runner.execution_workers(protocol, 'protocol', identity, dict(amendment, **changes))

def test_live_denominators_and_record_whitelist(tmp_path):
    records = [dict(window=f'{n:03d}', method='kimodo') for n in range(3)]
    (tmp_path/'protocol.json').write_text(json.dumps(dict(records=records, workers=3)))
    task = tmp_path/'tasks/000_kimodo'; task.mkdir(parents=True)
    (task/'result.json').write_text(json.dumps(dict(status='failed', passed=False)))
    running = tmp_path/'tasks/001_kimodo'; running.mkdir()
    (running/'state.json').write_text('{}')
    api = live.Results(tmp_path)
    result = api.snapshot()
    assert (result['expected'], result['evaluated'], result['passed'], result['execution_failed'], result['pending'], result['running']) == (3,0,0,1,2,1)
    with pytest.raises(KeyError): api.detail('../../protocol')
    epochs = tmp_path/'execution_epochs'; epochs.mkdir()
    (epochs/'123.json').write_text(json.dumps(dict(effective_workers=8)))
    assert api.snapshot()['worker_limit'] == 8


def test_twelve_worker_amendment_reserves_other_numerical_jobs():
    protocol=dict(workers=6,output='/run')
    identity=dict(protocol_sha256='p',runner_sha256='r')
    amendment=dict(version='k4-execution-amendment-2',protocol_sha256='p',
        original_runner_sha256='r',output='/run',max_workers=11,
        total_worker_limit=12,reserved_workers=1,authorization='User authorized twelve total')
    assert runner.execution_workers(protocol,'p',identity,amendment)==11
    assert runner.execution_workers(protocol,'p',identity,dict(amendment,max_workers=12,reserved_workers=0))==12
    for change in ({'max_workers':12},{'total_worker_limit':13},{'reserved_workers':-1},
                   {'max_workers':True},{'reserved_workers':True},{'authorization':''},
                   {'fit_config':'changed'},{'protocol_sha256':'other'}):
        with pytest.raises(ValueError):
            runner.execution_workers(protocol,'p',identity,dict(amendment,**change))


def test_new_protocol_declares_eight_worker_budget_without_changing_legacy():
    protocol=dict(version='k4-full-frozen-regression-2',workers=8,resource_budget=dict(
        max_workers=8,numerical_threads=1,minimum_free_GiB=80,authorization='User authorized at most8 CPU workers'))
    assert runner.execution_workers(protocol,'new',None,None)==8
    with pytest.raises(ValueError):runner.execution_workers(dict(workers=8),'old',None,None)
    for change in ({'max_workers':9},{'numerical_threads':2},{'minimum_free_GiB':79},{'authorization':''}):
        candidate=dict(protocol,resource_budget=dict(protocol['resource_budget'],**change))
        with pytest.raises(ValueError):runner.execution_workers(candidate,'new',None,None)


def test_stages_are_disjoint_subsets_and_keep_unrun_denominator():
    records=[dict(window=f'{n:03d}',method=m) for n in range(3) for m in ('a','b')]
    protocol=dict(version='k4-full-frozen-regression-2',records=records,
                  validation_stages=[dict(name='panel',paths=['000_a','001_b']),dict(name='formats',paths=['002_a'])])
    batches=runner.validation_batches(protocol)
    assert [(n,len(rows),gate) for n,rows,gate in batches]==[('panel',2,True),('formats',1,True),('full_remaining',3,False)]
    assert sorted((r['window'],r['method']) for _,rows,_ in batches for r in rows)==sorted((r['window'],r['method']) for r in records)
    for stages in ([dict(name='bad',paths=['999_a'])],[dict(name='a',paths=['000_a']),dict(name='b',paths=['000_a'])]):
        with pytest.raises(ValueError):runner.validation_batches(dict(protocol,validation_stages=stages))


def test_retry_archives_execution_history_but_keeps_metric_failures(tmp_path):
    task=tmp_path/'tasks/001_a';task.mkdir(parents=True)
    fit=tmp_path/'fits/001/a';fit.mkdir(parents=True);(fit/'partial').write_text('preserve')
    score=tmp_path/'scores/native/001_a.json';score.parent.mkdir(parents=True);score.write_text('old-score')
    result=task/'result.json';result.write_text(json.dumps(dict(status='completed',passed=False)))
    assert not runner.archive_execution_failure(tmp_path,'001_a')
    result.write_text(json.dumps(dict(status='failed',error='interrupted')))
    assert runner.archive_execution_failure(tmp_path,'001_a')
    history=list((tmp_path/'attempt_history/001_a').iterdir());assert len(history)==1
    assert (history[0]/'fit/partial').read_text()=='preserve'
    assert (history[0]/'native_score.json').read_text()=='old-score'
    assert not task.exists() and not fit.exists() and not score.exists()


def test_interrupted_attempt_is_preserved_before_explicit_retry(tmp_path):
    task=tmp_path/'tasks/001_a';task.mkdir(parents=True);(task/'state.json').write_text('{"status":"running"}')
    fit=tmp_path/'fits/001/a';fit.mkdir(parents=True);(fit/'partial').write_text('unfinished-native')
    assert runner.archive_execution_failure(tmp_path,'001_a')
    history=next((tmp_path/'attempt_history/001_a').iterdir())
    assert (history/'fit/partial').read_text()=='unfinished-native'
    assert (history/'task/state.json').exists()


def test_readiness_keeps_pending_and_rejects_failed_or_changed_acceptance(tmp_path):
    prerequisite=dict(run=str(tmp_path),code_commit='frozen',expected=2,require_all_passed=True)
    protocol=dict(acceptance_prerequisites=[prerequisite])
    assert runner.acceptance_readiness(protocol,'policy')[0]==[str(tmp_path)]
    (tmp_path/'identity.json').write_text(json.dumps(dict(code_commit='frozen')))
    (tmp_path/'completion.json').write_text(json.dumps(dict(expected=2)))
    review=dict(expected=2,completed=2,execution_failed=0,passed=1,policy_sha256='policy')
    (tmp_path/'review.json').write_text(json.dumps(review))
    with pytest.raises(RuntimeError):runner.acceptance_readiness(protocol,'policy')
    review['passed']=2;(tmp_path/'review.json').write_text(json.dumps(review))
    waiting,evidence=runner.acceptance_readiness(protocol,'policy')
    assert not waiting and evidence[str(tmp_path)]['passed']==2
    with pytest.raises(ValueError):runner.acceptance_readiness(protocol,'different-policy')
