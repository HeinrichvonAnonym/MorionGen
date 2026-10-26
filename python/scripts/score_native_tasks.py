"""Recompute root-task, foot sensitivity and native CHOIS evidence from saved outputs."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT
from motiogen.runtime import json_write,checksum
from motiogen.motion import load_motion
from motiogen.feedback import score_root_goals
from motiogen.native_analysis import foot_sensitivity
contract=json.loads((ROOT/'config/native_metric_contract.json').read_text())
base=ROOT/'develop_logs/test_runs/20261024_native_pilots'
pilots=json.loads((base/'audited_pilots.json').read_text())
rows=[]
for entry in pilots['rows']:
    path=ROOT/entry['run']; motion=load_motion(path/'motion.npz')
    xyz=motion['joints']; fps=float(motion['fps'])
    result={'method':entry['method'],'run':entry['run'],'condition':entry['condition'],
            'motion_sha256':checksum(path/'motion.npz'),'contract':contract,
            'feet':foot_sensitivity(motion,contract)}
    if entry['method']=='dart':
        goals=[{'frame':0,'position':[1,0,0]},{'frame':150,'position':[-1,.5,0]}]
        result['goals']=score_root_goals(xyz,goals,fps=fps,radius=contract['root_goal_radius_m'],hold_seconds=contract['root_goal_hold_seconds'])
        trace=json.loads((path/'trace.json').read_text())
        result['input_response_seconds']=(trace['applied_update_frame']-150)/fps if trace['applied_update_frame'] is not None else None
        result['native_terminations']=sum(step['terminated'] for step in trace['steps'])
    elif entry['method']=='ardy':
        frames=np.arange(7,len(xyz),8); seconds=frames/fps
        distance=.4*np.minimum(seconds,5.)-.2*np.maximum(seconds-5.,0.)
        target=np.stack([np.zeros_like(distance),-distance],axis=-1)
        errors=np.linalg.norm(xyz[frames,0,:2]-target,axis=-1)
        result['native_root_waypoints']={'frames':frames.tolist(),'error_xy_m':errors.tolist(),
            'mean_error_m':float(errors.mean()),'success_at_5cm':float((errors<=.05).mean()),
            'denominator':len(frames),'axes':'canonical X,Y, converted from native X,Z',
            'scope':'skeletal root against supplied native Root2D waypoints; no hand constraint is implied'}
    elif entry['method']=='chois':
        native=list((path/'raw/chois_results/evaluation_metrics_json').rglob('evaluation_metrics_for_all_test_data.json'))
        if len(native)!=1: raise ValueError(f'Expected one native CHOIS aggregate in {path}')
        result['native_object_metrics']=json.loads(native[0].read_text())
        result['native_source']=str(native[0].relative_to(ROOT))
        result['native_metric_notes']='JPE/object position errors are upstream millimetres; contact F1 is a fraction. Penetration is mean absolute negative object-SDF distance over sampled human mesh vertices/time, not collision probability.'
    json_write(path/'native_task_metrics_v1.json',result)
    rows.append(result)
json_write(base/'native_task_scores.json',rows)
for row in rows:
    if row['method'] in ['ardy','dart']:
        print(json.dumps({k:row[k] for k in ['method','condition','goals','input_response_seconds','native_root_waypoints'] if k in row}))
