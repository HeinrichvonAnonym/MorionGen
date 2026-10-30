"""Second-layer anatomical recovery, keeping all position-only denominators."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT
from motiogen.runtime import json_write
from motiogen.supplement_inputs import accepted_inputs
from motiogen.posture_geometry import read_positions,observable_angles
from motiogen.posture_rotations import recovered_angles
from motiogen.motion import canonicalize,Y_TO_Z
from motiogen.ergonomics import score
parser=argparse.ArgumentParser();parser.add_argument('--methods',nargs='+',default=['ardy','kimodo','chois']);parser.add_argument('--output',type=Path);parser.add_argument('--limit',type=int);args=parser.parse_args()
base=ROOT/'develop_logs/test_runs/20261028_supplement';out=(args.output or base/'posture_rotations').resolve();out.mkdir(exist_ok=True)
cal=json.loads((base/'skeleton_calibration.json').read_text());chois=json.loads((base/'chois_skeleton_calibration.json').read_text())['windows']
primal_cal=json.loads((base/'primal_skeleton_calibration.json').read_text()) if 'primal' in args.methods else None
rows=[];start=time.monotonic()
for ordinal,row in enumerate(accepted_inputs()):
    if row['method'] not in args.methods:continue
    if args.limit is not None and len(rows)>=args.limit:break
    receipt={k:row[k] for k in ['method','task','seed','clock','condition','run','motion_path'] if k in row}
    key=hashlib.sha256((row['motion_path']+json.dumps(row.get('condition',{}),sort_keys=True)).encode()).hexdigest()[:20]
    try:
        xyz,names,fps=read_positions(row);angles=observable_angles(xyz,names)
        if row['method']=='primal':
            from scipy.spatial.transform import Rotation
            calibration=primal_cal
            folder=Path(row['motion_path']).parent
            with np.load(folder/'initial_history.npz') as d:
                if not np.all(d['betas']==0):raise ValueError('PRIMAL source body shape differs from neutral calibration')
            with np.load(folder/(row['variant']+'.npz')) as d:pose=d['pose'][0]
            local=Rotation.from_rotvec(pose[:,3:].reshape(-1,3)).as_matrix().reshape(len(pose),22,3,3)
            rot=np.empty_like(local)
            for j,parent in enumerate(calibration['parents'][:22]):
                rot[:,j]=Y_TO_Z@local[:,j] if parent<0 else rot[:,parent]@local[:,j]
        elif row['method']=='chois':
            with np.load(ROOT/row['run']/'raw/native_prediction.npz') as d:rot=d['global_rotations']
            calibration=chois[str(row['task_spec'].get('native_window_index',0))]
        else:
            calibration=cal[row['method']]
            with np.load(row['motion_path']) as d:
                if row['method']=='ardy':rot=d['global_rotations']
                else:_,rot=canonicalize(d['posed_joints'],'Y-up',d['global_rot_mats'])
        recovered,proof=recovered_angles(xyz,rot,calibration);angles.update(recovered)
        arrays=dict(angles);summaries={}
        for side in ['left','right']:
            for method in ['RULA','REBA']:
                bounds=[];levels=[];component_rows={};missing=set()
                for f in range(len(xyz)):
                    inputs={k:float(angles[side+'_'+k][f]) for k in ['elbow_flexion_deg','upper_arm_flexion_deg','wrist_flexion_deg'] if np.isfinite(angles[side+'_'+k][f])}
                    for k in ['neck_flexion_deg','trunk_flexion_deg']:
                        if np.isfinite(angles[k][f]):inputs[k]=float(angles[k][f])
                    if method=='REBA':inputs['knee_flexion_deg']=float(angles[side+'_knee_flexion_deg'][f])
                    result=score(inputs,method);missing.update(result['missing_inputs']);bounds.append(result['score_range']);levels.append([min(result['possible_action_levels']),max(result['possible_action_levels'])])
                    for k,values in result['components'].items():component_rows.setdefault(k,[]).append([min(values),max(values)])
                prefix=side+'_'+method
                arrays[prefix+'_bounds']=np.asarray(bounds,dtype=np.int8);arrays[prefix+'_action_level_bounds']=np.asarray(levels,dtype=np.int8)
                for k,values in component_rows.items():arrays[prefix+'_component_'+k]=np.asarray(values,dtype=np.int8)
                bounds=np.asarray(bounds);levels=np.asarray(levels)
                summaries[prefix]={'frames':len(xyz),'complete_total_frames':0,'mean_lower_bound':float(bounds[:,0].mean()),'mean_upper_bound':float(bounds[:,1].mean()),
                    'score_range':[int(bounds.min()),int(bounds.max())],'missing_inputs':sorted(missing),
                    'definite_high_action_seconds':float((levels[:,0]>=3).sum()/fps),'possible_high_action_seconds':float((levels[:,1]>=3).sum()/fps),
                    'scope':'Conservative unknown-input bounds; qualifying support/load/repetition/coupling and qualitative adjustment variables remain explicit, not neutral.'}
        arrays['fps']=fps;np.savez_compressed(out/(key+'.npz'),**arrays)
        receipt.update(status='rotation_components_computed',calibration=proof,ergonomic=summaries,arrays=str((out/(key+'.npz')).relative_to(ROOT)),
            angles={k:{'min':float(np.nanmin(v)) if np.isfinite(v).any() else None,'max':float(np.nanmax(v)) if np.isfinite(v).any() else None,'finite_frames':int(np.isfinite(v).sum())} for k,v in angles.items()},
            human_validation='pending_no_raters',task_conditions_source='Not registered; enumerated as missing variables')
    except Exception as exc:receipt.update(status='failed',reason=f'{type(exc).__name__}: {exc}')
    rows.append(receipt)
    if len(rows)%100==0:print(json.dumps({'completed':len(rows),'elapsed_seconds':time.monotonic()-start,'last_status':receipt['status']}),flush=True)
json_write(out/'analysis.json',{'rows':rows,'elapsed_seconds':time.monotonic()-start,
    'scope':f'{args.methods} native rotation recovery. Parent position-only analysis retains every method/cell. Automatic angle conventions and rules are distinct from pending human scoring validity.'})
