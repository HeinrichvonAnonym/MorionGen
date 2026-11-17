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
