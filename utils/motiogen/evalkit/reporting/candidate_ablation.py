"""Accounting and cache contracts for independent, exhaustive candidate review.

Runtime selection is an observation, never the independent candidate verdict.
This module reads JSON and hashes only; it neither fits nor selects motion.
"""
import hashlib
import json
from pathlib import Path

ARTIFACTS=('model.osim','coordinates_refined_native.mot','fit.npz',
           'head_neck.npz','head_neck_model.json','coordinates.mot')
NON_NUMERICAL_BANK_FILES={'candidate_pool.py','candidate_budget.py','native_cache.py'}


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)


def compatible(key,expected,config,policy=None):
    """Single-fit execution ignores bank dispatch/budget, never numerical code."""
    for name in ('inputs','mass_kg','floor_m','model_sha256','input_config_sha256','biomech_sha256'):
        if key.get(name)!=expected[name]:return False,'different_'+name
    if key.get('config')!=config:return False,'different_config'
    strip=lambda x:{k:v for k,v in x.items() if k not in NON_NUMERICAL_BANK_FILES}
    if strip(key.get('implementation',{}))!=strip(expected['implementation']):
        return False,'different_numerical_implementation'
    variant=key.get('export_variant')
    if policy is None:
        if variant:return False,'unexpected_export_variant'
    elif (not variant or variant.get('policy')!=policy or
          strip(variant.get('implementation',{}))!=strip(expected['implementation'])):
        return False,'different_export_policy_or_implementation'
    return True,'same_input_config_assets_and_numerical_implementation'


def summarize(routes,states,selectors):
    per={};rows=[];totals=dict(expected_routes=len(routes),applicable_cells=0,
        independently_evaluated_cells=0,candidate_passed_cells=0,candidate_metric_failed_cells=0,
        execution_failed_cells=0,pending_cells=0,fully_evaluated_routes=0,all_candidates_passed_routes=0,
        selector_evaluated_routes=0,selector_passed_routes=0,selector_missed_passing_routes=0,
        selector_comparison_complete_routes=0)
    for route in routes:
        key=route['window']+'_'+route['method'];cells={}
        for name,contract in route['cells'].items():
            if contract['status']=='inapplicable':
                cells[name]=dict(status='inapplicable',reason=contract['reason']);continue
            state=states.get((key,name),dict(status='pending',passed=None))
            if contract.get('alias'):
                state=dict(states.get((key,contract['alias']),dict(status='pending',passed=None)),
                           equivalent_to=contract['alias'])
            cells[name]=state
            c=per.setdefault(name,dict(name=name,applicable=0,evaluated=0,passed=0,metric_failed=0,execution_failed=0,pending=0))
            c['applicable']+=1;totals['applicable_cells']+=1
            if state['status']=='evaluated':
                c['evaluated']+=1;totals['independently_evaluated_cells']+=1
                if state['passed']:c['passed']+=1;totals['candidate_passed_cells']+=1
                else:c['metric_failed']+=1;totals['candidate_metric_failed_cells']+=1
            elif state['status']=='execution_failed':
                c['execution_failed']+=1;totals['execution_failed_cells']+=1
            else:c['pending']+=1;totals['pending_cells']+=1
        applicable=[v for v in cells.values() if v['status']!='inapplicable']
        complete=bool(applicable) and all(v['status']=='evaluated' for v in applicable)
        all_pass=complete and all(v['passed'] for v in applicable)
        any_pass=any(v.get('passed') is True for v in applicable)
        selector=selectors.get(key,{})
        selected_complete=selector.get('status')=='completed'
        totals['fully_evaluated_routes']+=complete;totals['all_candidates_passed_routes']+=all_pass
        totals['selector_evaluated_routes']+=selected_complete
        totals['selector_passed_routes']+=selected_complete and selector.get('passed') is True
        totals['selector_comparison_complete_routes']+=complete and selected_complete
        # Positive counterexample needs one passing candidate; a negative
        # conclusion about the entire bank requires complete candidate review.
        missed=selected_complete and selector.get('passed') is False and any_pass
        totals['selector_missed_passing_routes']+=missed
        rows.append(dict(window=route['window'],method=route['method'],candidates=cells,
            fully_evaluated=complete,all_candidates_passed=all_pass,
            any_candidate_passed=True if any_pass else (False if complete else None),
            selector=selector,missed_passing_candidate=missed if selected_complete else None))
    return dict(totals,per_candidate=list(per.values()),rows=rows,
        scope='Every applicable fixed candidate reviewed against R independently; unknown is not failure. All-candidate pass is a diagnostic, not a requirement that weak ablations pass. Selector results are separate.')
