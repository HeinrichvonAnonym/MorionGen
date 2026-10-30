"""Bilateral observable components/bounds on every accepted historical cell."""
import sys,json,hashlib,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT
from motiogen.runtime import json_write
from motiogen.supplement_inputs import accepted_inputs
from motiogen.posture_geometry import read_positions,observable_angles,episodes
from motiogen.ergonomics import score

out=ROOT/'develop_logs/test_runs/20261028_supplement/posture';out.mkdir(parents=True,exist_ok=True)
rows=[];started=time.monotonic()
for ordinal,row in enumerate(accepted_inputs()):
    receipt={k:row[k] for k in ['method','task','seed','clock','condition','run','motion_path','parent_analysis'] if k in row}
    key=hashlib.sha256((row['motion_path']+json.dumps(row.get('condition',{}),sort_keys=True)).encode()).hexdigest()[:20]
    try:
        xyz,names,fps=read_positions(row);angles=observable_angles(xyz,names)
        arrays=dict(angles);sides={}
        for side in ['left','right']:
            outputs={}
            for method in ['RULA','REBA']:
                bounds=[];levels=[];identifiable=[]
                for frame in range(len(xyz)):
                    inputs={'elbow_flexion_deg':float(angles[side+'_elbow_flexion_deg'][frame])}
                    if method=='REBA':inputs['knee_flexion_deg']=float(angles[side+'_knee_flexion_deg'][frame])
                    # Native rotations are inventoried separately. No fitted
                    # hand posture or neutral task-condition substitution.
                    result=score(inputs,method);bounds.append(result['score_range']);levels.append([min(result['possible_action_levels']),max(result['possible_action_levels'])]);identifiable.append(result['components']['lower_arm'])
                arrays[side+'_'+method+'_bounds']=np.asarray(bounds,dtype=np.int8)
                arrays[side+'_'+method+'_action_level_bounds']=np.asarray(levels,dtype=np.int8)
                outputs[method]={'complete_total_frames':0,'frames':len(xyz),'missing_inputs':result['missing_inputs'],
                    'score_range':[int(np.min(bounds)),int(np.max(bounds))],'scope':'Position-identifiable elbow/knee stage; rotation-calibrated fields must be added before final acceptance.'}
            sides[side]=outputs
        summaries={k:{'min':float(np.nanmin(v)),'max':float(np.nanmax(v)),'median':float(np.nanmedian(v)),
                      'finite_frames':int(np.isfinite(v).sum())} for k,v in angles.items()}
        exposure={side+'_upper_arm_above90':episodes(angles[side+'_upper_arm_elevation_deg']>90,fps) for side in ['left','right']}
        arrays['fps']=fps;np.savez_compressed(out/(key+'.npz'),**arrays)
        receipt.update(status='position_components_computed',frames=len(xyz),fps=fps,angles=summaries,exposure=exposure,
            ergonomic=sides,arrays=str((out/(key+'.npz')).relative_to(ROOT)),
            source_metrics=row.get('values',{}),human_validation='pending_no_raters',
            rotation_stage='Required: native basis/calibration audit; position-only stage does not exhaust recoverable data')
    except Exception as exc:receipt.update(status='failed',reason=f'{type(exc).__name__}: {exc}')
    rows.append(receipt)
    if ordinal%100==0:print(json.dumps({'completed':len(rows),'elapsed_seconds':time.monotonic()-started,'last_status':receipt['status']}),flush=True)
json_write(out/'analysis.json',{'rows':rows,'elapsed_seconds':time.monotonic()-started,
    'scope':'Intermediate position-only layer, all denominators retained. Not final ergonomic acceptance or human validity.'})
