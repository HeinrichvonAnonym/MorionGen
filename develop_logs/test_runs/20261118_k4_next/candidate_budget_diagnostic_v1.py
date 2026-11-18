"""Reconstruct input-only budget choices, then join independent acceptance."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
B=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117')
HERE=Path(__file__).parent
cfg_path=HERE/'candidate_budget_preregistration_v1.json';cfg=json.loads(cfg_path.read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=[];sources={str(cfg_path):sha(cfg_path)}
for run,audit in zip(cfg['sources'],['pool_candidates','composed_pool_candidates']):
    auditpath=ROOT/'develop_logs/test_runs/20261118_k4_monitor'/audit/'review.json'
    independent={(r['window'],r['method']):r for r in json.loads(auditpath.read_text())['records']}
    sources[str(auditpath)]=sha(auditpath)
    for file in sorted((B/run/'fits/pool').glob('*/*/candidate_selection.json')):
        quality=json.loads(file.read_text());w,m=file.parts[-3:-1];sources[str(file)]=sha(file)
        # Choose before consulting independent R acceptance.
        ordered=list(quality['qualities']);chosen=None;position=None
        for i,name in enumerate(ordered):
            q=quality['qualities'][name];metrics=[v for v in q['metrics'].values() if v['expected']]
            if q.get('export_path_diagnostic') and metrics and all(v['missing']==0 and v['value'] is not None and v['value']<=cfg['margin']*v['threshold'] for v in metrics):
                chosen=name;position=i+1;break
        if chosen is None:chosen=quality['selected'];position=len(ordered);mode='complete_pool_fallback'
        else:mode='input_margin'
        item=independent.get((w,m),{}).get('candidates',{}).get(chosen)
        rows.append(dict(run=run,window=w,method=m,chosen=chosen,full_pool_selected=quality['selected'],mode=mode,
            inspected_candidates=position,total_materialized_candidates=len(ordered),
            omitted_materialized_candidates=len(ordered)-position,independent_pass=None if item is None else item['passed'],
            independent_failures=None if item is None else item['failures']))
out=dict(protocol=cfg,source_hashes=sources,rows=rows,summary=dict(records=len(rows),early_choices=sum(r['mode']=='input_margin' for r in rows),
    independently_passed=sum(r['independent_pass'] is True for r in rows),independently_failed=sum(r['independent_pass'] is False for r in rows),
    independently_unknown=sum(r['independent_pass'] is None for r in rows),
    omitted_materialized_candidates=sum(r['omitted_materialized_candidates'] for r in rows),total_materialized_candidates=sum(r['total_materialized_candidates'] for r in rows)),
    limitation='Saved-candidate counts are not wall-clock speedup; incomplete panels and candidates with no independent audit stay unknown; no method or656 changes')
(HERE/'candidate_budget_diagnostic_v1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out['summary'],indent=2))
for r in rows:
 if r['independent_pass'] is False:print('FAIL',r['run'],r['window'],r['method'],r['chosen'],r['independent_failures'])
