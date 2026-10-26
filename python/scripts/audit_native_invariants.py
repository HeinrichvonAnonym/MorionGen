"""Output-level negative controls and common-prefix checks on actual runs."""
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT
from motiogen.motion import load_motion
from motiogen.runtime import json_write

analysis=json.loads((ROOT/'develop_logs/test_runs/20261024_native_matrix_analysis/analysis.json').read_text())
rows=[r for r in analysis['rows'] if r['intervention_checks']['passed'] and r['weight_identity'] is not None]
checks=[]

def compare(rule,left,right,frames=None):
    a=load_motion(ROOT/left['run']/'motion.npz')['joints']
    b=load_motion(ROOT/right['run']/'motion.npz')['joints']
    if frames is not None:a=a[:frames];b=b[:frames]
    if not len(a):return
    shape=a.shape==b.shape
    delta=float(np.max(np.abs(a-b))) if shape else None
    checks.append({'rule':rule,'method':left['method'],'left_run':left['run'],'right_run':right['run'],
        'frames':len(a),'exact_equal':bool(shape and np.array_equal(a,b)),
        'max_abs_delta_m':delta,'passed':bool(shape and delta<=1e-6),
        'tolerance_m':1e-6,'scope':'Invariant before intended divergence or with all target-control paths disabled; not equality after a meaningful intervention.'})

offline=defaultdict(list)
for row in rows:
    if row['method'] not in ['motionlcm','omnicontrol'] or any(row['condition'].values()):continue
    scene=json.loads((ROOT/row['run']/'scenario.json').read_text())
    offline[(row['method'],row['seed'],scene['prompt'],row['weight_identity'])].append(row)
for group in offline.values():
    for row in group[1:]:compare('All target-control contributions off: changing target geometry does not change generated motion',group[0],row)

paired=defaultdict(dict)
for row in rows:
    if row['method'] not in ['ardy','dart','closd']:continue
    condition=row['condition']
    key=(row['method'],row['task'],row['seed'],row['clock'],condition.get('correction'))
    factor={'ardy':'history_long','dart':'goal_update','closd':'execution_context'}[row['method']]
    paired[key][condition[factor]]=row
for key,pair in paired.items():
    if set(pair)!={0,1}:continue
    method=key[0]
    if method=='ardy':compare('Short/long histories are identical until the short buffer truncates',pair[0],pair[1],40)
    else:
        traces=[json.loads((ROOT/pair[c]['run']/'trace.json').read_text()) for c in (0,1)]
        if method=='dart':
            limit=traces[1]['applied_update_frame']
            compare('Goal freeze/update share identical motion before the goal changes',pair[0],pair[1],limit)
        else:
            limit=min([60]+[t['goal_applied_frame'] for t in traces if t.get('goal_applied_frame') is not None])
            compare('Simulation/prediction context share native simulation bootstrap before their first divergence',pair[0],pair[1],limit)
summary=[]
for rule in sorted({c['rule'] for c in checks}):
    selected=[c for c in checks if c['rule']==rule]
    summary.append({'rule':rule,'checked':len(selected),'passed':sum(c['passed'] for c in selected),
        'bitwise_equal':sum(c['exact_equal'] for c in selected),'max_abs_delta_m':max(c['max_abs_delta_m'] or 0. for c in selected)})
output=ROOT/'develop_logs/test_runs/20261024_native_invariants/audit.json'
json_write(output,{'status':'passed' if all(c['passed'] for c in checks) else 'failed',
    'analysis_cutoff':analysis['updated_at'],'checks':checks,'summary':summary,
    'scope':'Output-level checks complement source hooks, weights and paired-input identities. They do not measure task capability.'})
print(json.dumps({'checks':len(checks),'failed':[c for c in checks if not c['passed']],'summary':summary,'output':str(output)}))
