"""Paired eight-format conversion checks; distinct from external anatomy accuracy."""
import tempfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from .motion import Motion,load,PARENTS
from .export import retarget,serialize,target_template,resample,ALIASES
from .body import evaluate_body
from .posture import evaluate
from .geometry import tiers,distance,frame,valid_rotation,rotation_errors
from .config import CONFIG
try:
    from ...motion import HUMANML_NAMES
except ImportError:
    from motion import HUMANML_NAMES


def segment_fixture(times,head,thorax):
    """Measured head/thorax on a declared template body; other segments are priors."""
    names,rest,parents=target_template('primal');r=np.broadcast_to(np.eye(3),(len(times),22,3,3)).copy()
    for j in (3,6,9,12,13,14,16,17,18,19,20,21):r[:,j]=thorax
    r[:,15]=head;p=np.zeros((len(times),22,3))
    for j,parent in enumerate(parents):p[:,j]=rest[j] if parent<0 else p[:,parent]+np.einsum('tij,j->ti',r[:,parent],rest[j]-rest[parent])
    return Motion(p,names,times,'reference',r,names,{'segment_basis':np.eye(3).tolist(),'scope':'isolated_head_thorax_on_template_body'})


def profile_checks(reference,head_name,thorax_name,scratch):
    good=np.isfinite(reference.positions).all((1,2))&np.isfinite(reference.rotations).all((1,2,3))
    edges=np.diff(np.r_[False,good,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
    candidates=[(a,b) for a,b in zip(starts,ends) if reference.times[b-1]-reference.times[a]>=1.]
    if not candidates:return {'status':'position_precision_unreliable' if reference.provenance.get('coordinate_precision') else 'no_complete_one_second_window',
                              'coordinate_precision':reference.provenance.get('coordinate_precision')}
    a,b=candidates[0];b=min(b,a+int(round(10*reference.fps))+1)
    ref=Motion(reference.positions[a:b],reference.names,reference.times[a:b],reference.method,
               reference.rotations[a:b],reference.rotation_names,reference.provenance)
    h=ref.rotation_names.index(head_name);c=ref.rotation_names.index(thorax_name)
    basis=np.asarray(ref.provenance.get('segment_basis',np.eye(3)))
    result={'status':'computed','window_s':[float(ref.times[0]),float(ref.times[-1])],
            'scope':ref.provenance.get('scope','full_body_published_bvh'),'methods':{}}
    body_reference = evaluate_body(Motion(ref.positions,[ALIASES.get(n,n) for n in ref.names],ref.times,'reference'))
    Path(scratch).mkdir(parents=True,exist_ok=True)
    for method in CONFIG['profiles']:
        try:
            target,parents=retarget(ref,method)
            with tempfile.TemporaryDirectory(prefix=method+'_',dir=scratch) as folder:
                source=serialize(target,parents,folder);back=load(source,method)
                _,rest,_=target_template(method);ni={n:i for i,n in enumerate(target.names)}
                target_basis=frame(np.array([0.,1,0]),rest[ni['right_hip']]-rest[ni['left_hip']])
                if method=='closd':
                    from .calibration import UPRIGHT
                    output_basis=UPRIGHT@target_basis
                else:output_basis=target_basis
                chest_basis=output_basis
                if method in ('motionlcm','omnicontrol'):
                    rest_frame=frame(rest[ni['neck']]-rest[ni['spine3']],rest[ni['right_collar']]-rest[ni['left_collar']])
                    chest_basis=rest_frame.T@target_basis
                measured_result=evaluate(back,{'head':output_basis,'thorax':chest_basis})
                measured=measured_result['metrics']['neck']
                # Scoring-only complete target. D1/D2's predictor never receives this rotation.
                chest_name='Spine3' if method=='ardy' else 'spine3'
                control_r=(target.rotations[:,ni[chest_name]]@target_basis).swapaxes(-1,-2)@(target.rotations[:,ni['head']]@target_basis)
                idx=np.searchsorted(target.times,back.times);idx=np.clip(idx,0,len(target.times)-1)
                expected_head=Slerp(ref.times,Rotation.from_matrix(ref.rotations[:,h]))(back.times).as_matrix()@basis
                expected_chest=Slerp(ref.times,Rotation.from_matrix(ref.rotations[:,c]))(back.times).as_matrix()@basis
                expected=expected_chest.swapaxes(-1,-2)@expected_head
                valid=measured['anatomical_valid']
                result['methods'][method]={'status':measured['status'],'frames':len(back.times),'valid_frames':int(valid.sum()),
                    'retarget_and_temporal':rotation_errors(expected,control_r[idx]),
                    'codec':rotation_errors(control_r[idx][valid],measured['rotations'][valid],len(valid)),
                    'end_to_end':rotation_errors(expected[valid],measured['rotations'][valid],len(valid)),
                    'thorax_end_to_end':rotation_errors(expected_chest,measured_result['metrics']['thorax']['rotations']),
                    'position_codec_p95_m':float(np.percentile(np.linalg.norm(back.positions[:,:min(len(target.names),len(back.names))]-target.positions[idx,:min(len(target.names),len(back.names))],axis=-1),95)),
                    'prior_joints':target.provenance['prior_joints'],
                    'position_retarget_median_m':target.provenance['position_median_error_m'],
                    'runtime_model_calibration':back.provenance['calibration']['status']}
                if result['scope'].startswith('full_body'):
                    body_errors={}
                    for name, item in measured_result['body'].items():
                        original=body_reference[name]
                        good=item['valid'] & np.interp(back.times,ref.times,original['valid'].astype(float)).__ge__(.999)
                        expected_values=np.interp(back.times,ref.times,original['values_deg'])
                        delta=item['values_deg'][good]-expected_values[good]
                        body_errors[name]={**tiers(np.abs(delta),len(good)),
                                           'signed_bias_deg':float(delta.mean()) if len(delta) else None}
                    result['methods'][method]['body_end_to_end']=body_errors
        except Exception as e:result['methods'][method]={'status':'failed','reason':f'{type(e).__name__}: {e}'}
    return result
