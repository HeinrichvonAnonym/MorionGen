"""Recompute core metrics from persisted motions with original metric contracts."""
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT
from motiogen.motion import load_motion
from motiogen.metrics import evaluate
from motiogen.runtime import json_write

base=ROOT/'develop_logs/test_runs'
analysis=json.loads((base/'20261024_native_matrix_analysis/analysis.json').read_text())
checks=[];missing=Counter();reset_checks=[]
core=['jerk_rms','trunk_inclination','trunk_flexion','trunk_risk_fraction',
      'left_upper_arm_elevation','right_upper_arm_elevation','left_elbow_rom','right_elbow_rom',
      'hand_position_error','position_success_rate','contact_distance']
for row in analysis['rows']:
    run=ROOT/row['run'];variant=row.get('variant')
    motion_path=run/'factorial'/f'{variant}_canonical.npz' if variant else run/'motion.npz'
    metrics_path=run/'factorial'/f'{variant}_metrics.json' if variant else run/'metrics.json'
    motion=load_motion(motion_path)
    if row['method']=='closd':
        trace=json.loads((run/'trace.json').read_text())
        reset_frames=[step['frame'] for step in trace['steps'] if step['done']]
        goal=trace['requested_goals'][-1]
        start=next(step['frame'] for step in trace['steps'] if step['request_wall_seconds']>=5.) if row['clock']=='realtime' else goal['frame']
        hold=round(float(motion['fps'])*.5)
        xy=motion['joints'][:,0,:2]-np.asarray(goal['position'])[:2]
        distances=np.sqrt(np.sum(xy*xy,axis=1))
        windows=[i for i in range(start,len(distances)-hold+1) if max(distances[i:i+hold])<=.3]
        following=min([frame for frame in reset_frames if frame>=start],default=len(distances))
        safe=[i for i in windows if i+hold-1<following]
        expected={'new_goal_success_before_next_reset':int(bool(safe)),
                  'new_goal_success_reset_free_episode':int(bool(windows) and not reset_frames)}
        saved=row['reset_semantics']
        passed=all(row['values'][key]==value==saved[key] for key,value in expected.items())
        passed=passed and saved['reset_frames']==reset_frames and saved['goal_frame']==start and saved['hold_samples']==hold
        passed=passed and saved['first_eventual_hold_frame']==(min(windows) if windows else None)
        passed=passed and saved['first_hold_before_next_reset_frame']==(min(safe) if safe else None)
        passed=passed and int(bool(windows))==row['values']['new_goal_success_30cm_hold_05s']
        recovery=row['task_results'][-1]['recovery_seconds']
        passed=passed and (np.isclose((min(windows)-start)/float(motion['fps']),recovery,rtol=0,atol=1e-12) if windows else recovery is None)
        reset_checks.append({'run':row['run'],'passed':bool(passed),'reset_frames':reset_frames,
                             'goal_frame':start,'first_eventual_hold_frame':min(windows) if windows else None,**expected})
    raw=json.loads(metrics_path.read_text())['metrics']
    for name,metric in raw.items():
        if metric['status']=='measured':
            if metric['value'] is None or not np.isfinite(metric['value']) or not metric.get('unit') or metric.get('n',0)<=0:
                raise ValueError(f'Invalid measured value/unit/denominator: {metrics_path}/{name}')
        else:
            if metric['value'] is not None or not metric.get('reason'):raise ValueError('Missingness lost its explicit reason')
            missing[(row['method'],name,metric['reason'])]+=1
    contracts=json.loads((run/'contracts.json').read_text())
    policy=contracts['metrics.json']['content']
    recalculated=evaluate(motion,row['task_spec'],policy=policy)['metrics']
    compared=[]
    for name in core:
        if raw.get(name,{}).get('status')!='measured':continue
        actual=recalculated[name]
        valid=actual['status']=='measured' and actual['unit']==raw[name]['unit'] and actual['n']==raw[name]['n']
        delta=abs(actual['value']-raw[name]['value']) if actual['value'] is not None else None
        passed=bool(valid and np.isclose(actual['value'],raw[name]['value'],rtol=1e-6,atol=1e-8))
        compared.append({'metric':name,'passed':passed,'absolute_difference':delta,'unit':raw[name]['unit'],'denominator':raw[name]['n']})
    checks.append({'run':row['run'],'variant':variant,'method':row['method'],'checks':compared,
                   'passed':bool(compared) and all(c['passed'] for c in compared)})
receipt={'analysis_cutoff':analysis['updated_at'],'checks':checks,'reset_checks':reset_checks,
         'reset_scope':'Independent enumeration of all 15-sample, 30cm goal windows from saved CLoSD joints and raw trace events. Check eventual arrival, next reset after input, and absence of resets across the full episode separately; no call to the analysis helper.',
         'passed':all(c['passed'] for c in checks+reset_checks),
         'missingness':[{'method':m,'metric':k,'reason':reason,'condition_cells':n} for (m,k,reason),n in sorted(missing.items())],
         'scope':'Registered evaluator rerun on persisted canonical arrays using each run\'s captured metric policy; checks serialization, units and denominators. Historical CHOIS elbow-ROM fields are retained as legacy_hand_anchor proxies in the report; corrected anatomical wrist ROM has its own independent raw-joint audit. Not a claim of an independent biomechanical validation or a second implementation.'}
json_write(base/'20261024_native_metric_reconstruction/audit.json',receipt)
print(json.dumps({'condition_cells':len(checks),'metric_checks':sum(len(c['checks']) for c in checks),
                  'passed':receipt['passed'],'failures':[c for c in checks if not c['passed']]}))
assert receipt['passed']
