import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('ablation',ROOT/'utils/motiogen/evalkit/reporting/candidate_ablation.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def route(cells):return dict(window='001',method='example',cells=cells)


def test_selected_pass_does_not_complete_candidate_review():
    routes=[route({'a':{'status':'applicable'},'b':{'status':'applicable'}})]
    states={('001_example','a'):dict(status='evaluated',passed=True)}
    s=m.summarize(routes,states,{'001_example':dict(status='completed',passed=True,selected='a')})
    assert s['selector_passed_routes']==1 and s['fully_evaluated_routes']==0
    assert s['pending_cells']==1 and s['candidate_metric_failed_cells']==0


def test_failed_candidate_is_a_complete_ablation_not_missing_review():
    routes=[route({'a':{'status':'applicable'},'b':{'status':'applicable'}})]
    states={('001_example','a'):dict(status='evaluated',passed=True),('001_example','b'):dict(status='evaluated',passed=False)}
    s=m.summarize(routes,states,{'001_example':dict(status='completed',passed=False,selected='b')})
    assert s['fully_evaluated_routes']==1 and s['all_candidates_passed_routes']==0
    assert s['selector_missed_passing_routes']==1 and s['candidate_metric_failed_cells']==1


def test_inapplicable_and_proven_equivalence_keep_denominators_clear():
    routes=[route({'a':{'status':'applicable'},'b':{'status':'equivalent','alias':'a'},'c':{'status':'inapplicable','reason':'no_rotation'}})]
    s=m.summarize(routes,{('001_example','a'):dict(status='evaluated',passed=True)}, {})
    assert s['applicable_cells']==2 and s['independently_evaluated_cells']==2
    assert s['fully_evaluated_routes']==1 and s['all_candidates_passed_routes']==1
    assert s['rows'][0]['candidates']['b']['equivalent_to']=='a'


def test_execution_failure_never_inherits_accuracy_verdict():
    s=m.summarize([route({'a':{'status':'applicable'}})],{('001_example','a'):dict(status='execution_failed',passed=None)}, {})
    assert s['execution_failed_cells']==1 and s['independently_evaluated_cells']==0
    assert s['fully_evaluated_routes']==0 and s['rows'][0]['any_candidate_passed'] is None


def test_fit_cache_requires_same_numerics_inputs_assets_and_actual_variant():
    expected=dict(inputs={'D':'hash'},mass_kg=70,floor_m=0,model_sha256='model',input_config_sha256='input',
        biomech_sha256='biomech',implementation={'k4_fitting.py':'fit','candidate_budget.py':'new'})
    key=dict(expected,config={'solver':'x'},implementation={'k4_fitting.py':'fit','candidate_budget.py':'old'})
    assert m.compatible(key,expected,{'solver':'x'})[0]
    assert not m.compatible(dict(key,inputs={'R':'hash'}),expected,{'solver':'x'})[0]
    assert not m.compatible(dict(key,implementation={'k4_fitting.py':'other'}),expected,{'solver':'x'})[0]
    assert not m.compatible(key,expected,{'solver':'x'},policy={'version':'path'})[0]
    v=dict(key,export_variant={'policy':{'version':'path'},'implementation':expected['implementation']})
    assert m.compatible(v,expected,{'solver':'x'},policy={'version':'path'})[0]
