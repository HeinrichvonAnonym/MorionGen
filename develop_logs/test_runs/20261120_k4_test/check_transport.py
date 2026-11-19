import json, sys, hashlib
from pathlib import Path
ROOT=Path('/home/heinrich/MorionGen');B=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117');code=B/'closure_budget_v3_checkout'
sys.path.insert(0,str(ROOT/'python/scripts'))
import evalkit_k4_test_evaluate as test
test.setup(code)
import numpy as np
run=Path(json.loads((code/'config/k_degradation_20261117/freeze.json').read_text())['run'])
row=json.loads((code/'config/k_degradation_20261117/windows.json').read_text())['records'][0]
root=json.loads((code/'config/gold_k_20261115/evidence.json').read_text())['local_roots']['collaborate']
out=B/'test_transport_validation_20261120_v1'
prepared=test.prepare(row,root,out)
checks=[]
for method, item in prepared['records'].items():
 original=run/'inputs'/row['window_key']/method/'candidate'
 new=out/'runtime_inputs'/row['window_key']/method
 mismatches=[str(p.relative_to(original)) for p in original.rglob('*') if p.is_file() and test.sha(p)!=test.sha(new/p.relative_to(original))]
 assert not mismatches,(method,mismatches)
 checks.append({'method':method,'byte_identical_input_files':len(item['input_hashes'])})
records={(r['window'],r['method']):r for r in json.loads((run/'artifact_verification.json').read_text())['records']}
paired=[]
for w,m in [('000','kimodo'),('008','motionlcm'),('077','ardy')]:
 a=json.loads((B/'selector_replay_20261120_v1/routes'/f'{w}_{m}'/'audit.json').read_text())
 for path in a['budget']['score_files']:
  old=json.loads(Path(path).read_text());source=run/'inputs'/w/m/'candidate';source=source/'factorial/reference.npz' if m=='primal' else source
  new=test.score_view(run/'references'/f'{w}.npz',source,m,records[w,m],old['fitted_artifact'],code)
  for view in [None,'original_clock']:
   x,y=(new,old) if view is None else (new[view],old[view])
   for key in ['comparisons','head_neck_rotation_comparisons','head_neck_rotation_temporal','fixed_observation_support']:
    assert x[key]==y[key],(w,m,path,view,key)
  paired.append({'window':w,'method':m,'score':path,'full_metric_tree_identical':True})
receipt={'eight_format_export':checks,'score_pairs':paired,'scope':'Development-only equivalence of new path orchestration; no test result used'}
(ROOT/'develop_logs/test_runs/20261120_k4_test/transport_verification.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2))
