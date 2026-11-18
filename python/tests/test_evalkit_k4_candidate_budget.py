import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.evalkit.k.candidate_budget import accepts,validate

POLICY={'version':'first_input_margin_v1','margin':.8}
def quality(value=4.,threshold=5.):
    return {'passed':True,'export_path_diagnostic':True,'metrics':{'direction':{'expected':3,'present':3,'missing':0,'value':value,'threshold':threshold}}}
def test_fixed_margin_and_no_reference_arguments():
    assert accepts(quality(),POLICY)
    assert not accepts(quality(4.000000001),POLICY)
    assert not accepts(quality(),None)
@pytest.mark.parametrize('key,value',[('value',float('nan')),('value',None),('missing',1),('present',2),('threshold',0),('value',-1)])
def test_missing_invalid_or_incomplete_never_early_passes(key,value):
    q=quality();q['metrics']['direction'][key]=value
    assert not accepts(q,POLICY)
def test_path_and_all_metrics_required():
    q=quality();q['export_path_diagnostic']=False
    assert not accepts(q,POLICY)
    q=quality();q['metrics']['interpolation_rate']={'expected':5,'present':5,'missing':0,'value':9.,'threshold':10.}
    assert not accepts(q,POLICY)
def test_disabled_observation_does_not_authorize_empty_result():
    q=quality();q['metrics']['direction']['expected']=0
    assert not accepts(q,POLICY)
@pytest.mark.parametrize('margin',[0,1,1.1,float('nan'),float('inf')])
def test_margin_requires_strict_reserve(margin):
    with pytest.raises(ValueError):validate(dict(POLICY,margin=margin))
