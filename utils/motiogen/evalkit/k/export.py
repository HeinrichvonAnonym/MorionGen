"""Retarget a complete segment reference and serialize the eight registered formats.

Retargeting uses source rest offsets and target model geometry. Unmatched joints
remain template priors; receipts report these separately from observed segments.
"""
import json
import subprocess
import atexit
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from scipy.signal import butter,sosfiltfilt
from .config import ROOT,CONFIG
from .motion import Motion,PARENTS
from .geometry import frame
from .calibration import CLOSD,UPRIGHT
try:
    from ...motion import HUMANML_NAMES,Y_TO_Z
except ImportError:
    from motion import HUMANML_NAMES,Y_TO_Z

ALIASES={'Hips':'pelvis','Spine':'spine1','Spine1':'spine2','Spine2':'spine_mid','Spine3':'spine3',
         'Neck':'neck','Neck1':'neck','Head':'head','Chest':'spine3','LeftShoulder':'left_collar','RightShoulder':'right_collar',
         'LeftArm':'left_shoulder','RightArm':'right_shoulder','LeftForeArm':'left_elbow','RightForeArm':'right_elbow',
         'LeftHand':'left_wrist','RightHand':'right_wrist','LeftUpLeg':'left_hip','RightUpLeg':'right_hip',
         'LeftLeg':'left_knee','RightLeg':'right_knee','LeftFoot':'left_ankle','RightFoot':'right_ankle',
         'LeftToeBase':'left_foot','RightToeBase':'right_foot'}

_CODEC=None


def humanml_codec(file):
    global _CODEC
    if _CODEC is None or _CODEC.poll() is not None:
        evidence=json.loads((ROOT/'config/gold_k_20261115/evidence.json').read_text())
        root=Path(evidence['local_roots']['andy']).parents[2]
        _CODEC=subprocess.Popen([str(root/'envs/motiogen-omnicontrol/bin/python'),str(ROOT/'python/scripts/evalkit_k_humanml_codec.py'),'--server'],
                                stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        atexit.register(_CODEC.terminate)
    _CODEC.stdin.write(json.dumps({'motion':str(file.resolve())})+'\n');_CODEC.stdin.flush()
    for line in _CODEC.stdout:
        if line.startswith('K_CODEC '):
            result=json.loads(line[8:])
            if result['status']!='ok':raise ValueError(result['error'])
            return
    raise RuntimeError('HumanML codec worker exited')


def target_template(method):
    key='primal' if method in ('motionlcm','omnicontrol','chois') else method
    cal=json.loads((ROOT/'config/evalkit_k_calibration'/f'{key}.json').read_text())
    count=22 if method in ('motionlcm','omnicontrol','chois','dart','primal') else len(cal['names'])
    return cal['names'][:count],np.asarray(cal['neutral_joints'])[:count],cal['parents'][:count]


def retarget(reference,method):
    """Source/reference world is Y-up. Outputs target FK and explicit retarget losses."""
    reference=resample(reference,CONFIG['profiles'][method]['fps'])
    names,rest,parents=target_template(method)
    sn=[ALIASES.get(n,n) for n in reference.names]
    rn=[ALIASES.get(n,n) for n in reference.rotation_names]
    # Older BVH hierarchies terminate the torso at Spine2. Its native rotation
    # remains observed; choose this only when the explicit upper chest is absent.
    if 'spine3' not in sn and 'Spine2' in reference.names:sn[reference.names.index('Spine2')]='spine3'
    if 'spine3' not in rn and 'Spine2' in reference.rotation_names:rn[reference.rotation_names.index('Spine2')]='spine3'
    target_alias={'Spine':'spine1','Spine3':'spine3','RightShoulder':'right_collar','LeftShoulder':'left_collar'}
    indices={n:i for i,n in enumerate(sn)}
    rotation_indices={n:i for i,n in enumerate(rn)}
    source_for=[target_alias.get(n,n) for n in names]
    r=np.broadcast_to(np.eye(3),(len(reference.times),len(names),3,3)).copy()
    supported=np.zeros(len(names),bool);bases=np.broadcast_to(np.eye(3),(len(names),3,3)).copy()
    for j,name in enumerate(source_for):
        if name not in rotation_indices:
            if parents[j]>=0:r[:,j]=r[:,parents[j]]
            continue
        supported[j]=True;rr=reference.rotations[:,rotation_indices[name]]
        children=[c for c,p in enumerate(parents) if p==j and source_for[c] in indices and name in indices]
        if children:
            target=[];observed=[]
            for child in children:
                v=reference.positions[:,indices[source_for[child]]]-reference.positions[:,indices[name]]
                local=np.einsum('tji,tj->ti',rr,v)
                a=rest[child]-rest[j];b=np.nanmedian(local,axis=0)
                if np.linalg.norm(a)>1e-6 and np.linalg.norm(b)>1e-6:
                    target.append(a/np.linalg.norm(a));observed.append(b/np.linalg.norm(b))
            if target:
                bases[j]=Rotation.align_vectors(np.asarray(observed),np.asarray(target))[0].as_matrix()
        r[:,j]=rr@bases[j]
    # Head and upper torso are measured segments: preserve their declared frame
    # semantics instead of borrowing a single child bone's ambiguous axial frame.
    source_basis=np.asarray(reference.provenance.get('segment_basis',np.eye(3)))
    ni={n:i for i,n in enumerate(names)}
    target_basis=frame(np.array([0.,1,0]),rest[ni['right_hip']]-rest[ni['left_hip']])
    for name in ('head','spine3'):
        for j,s in enumerate(source_for):
            if s==name and s in rotation_indices:
                bases[j]=source_basis@target_basis.T
                r[:,j]=reference.rotations[:,rotation_indices[s]]@bases[j]
    p=np.zeros((len(reference.times),len(names),3))
    root=reference.positions[:,indices['pelvis']]
    for j,parent in enumerate(parents):
        p[:,j]=root if parent<0 else p[:,parent]+np.einsum('tij,j->ti',r[:,parent],rest[j]-rest[parent])
    discrepancies={n:float(np.nanmedian(np.linalg.norm(p[:,j]-reference.positions[:,indices[s]],axis=-1)))
                   for j,(n,s) in enumerate(zip(names,source_for)) if s in indices}
    if reference.provenance.get('scope') == 'isolated_head_thorax_on_template_body':
        supported &= np.asarray([s in ('head','spine3') for s in source_for])
    return Motion(p,names,reference.times,method,r,names,{'retarget_bases':bases.tolist(),
        'source_observed_joint_mask':supported.tolist(),'prior_joints':[n for n,v in zip(names,supported) if not v],
        'position_median_error_m':discrepancies,'shape':'fixed_target_template',
        'precision_scope':'target_rig_conversion; anatomical transfer accuracy separately measured'}),parents


def resample(motion,fps):
    if not np.isfinite(motion.positions).all() or motion.rotations is None or not np.isfinite(motion.rotations).all():
        raise ValueError('Reference export requires a complete contiguous window; keep missing windows explicitly unavailable')
    target=motion.times[0]+np.arange(int(np.floor((motion.times[-1]-motion.times[0])*fps))+1)/fps
    p=motion.positions.copy();r=motion.rotations.copy()
    if motion.fps>fps and len(p)>30:
        sos=butter(4,min(6.,.4*fps),fs=motion.fps,output='sos')
        p=sosfiltfilt(sos,p,axis=0);raw=sosfiltfilt(sos,r,axis=0)
        u,_,vh=np.linalg.svd(raw);d=np.broadcast_to(np.eye(3),raw.shape).copy();d[...,2,2]=np.linalg.det(u@vh)
        r=u@d@vh
    xyz=np.stack([np.interp(target,motion.times,column) for column in p.reshape(len(p),-1).T],1).reshape(len(target),-1,3)
    rot=np.stack([Slerp(motion.times,Rotation.from_matrix(r[:,j]))(target).as_matrix() for j in range(r.shape[1])],1)
    return Motion(xyz,motion.names,target,motion.method,rot,motion.rotation_names,motion.provenance)


def serialize(motion,parents,out):
    """No target/reference labels enter position-only outputs."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    method=motion.method;p=motion.positions;r=motion.rotations;n=len(p)
    # The predictor receives only validity metadata, never reference rotations or retarget bases.
    provenance = {'prior_joints':list(motion.provenance['prior_joints']), 'shape':'fixed_target_template',
                  'origin':'controlled_format_degradation', 'reference_scope':motion.provenance.get('precision_scope')}
    if method=='chois':
        rename={'left_wrist':'left_wrist_body','right_wrist':'right_wrist_body'}
        provenance['prior_joints']=[rename.get(n,n) for n in provenance['prior_joints']]+['left_wrist','right_wrist']
    if method=='closd':provenance['unavailable_fields']=['dof_positions','contact_forces']
    (out/'k_provenance.json').write_text(json.dumps(provenance))
    meta=json.dumps({'baseline':method,'units':'m','axes':'right-handed Z-up','origin':'reference_degradation'})
    canonical={'joints':p@Y_TO_Z.T,'joint_names':motion.names,'fps':motion.fps,'times':motion.times,'metadata_json':meta}
    if method in ('motionlcm','omnicontrol'):
        np.savez_compressed(out/'motion.npz',**canonical)
        if method=='omnicontrol':
            humanml_codec(out/'motion.npz')
    elif method=='ardy':
        np.savez_compressed(out/'motion.npz',**canonical,global_rotations=np.einsum('ij,tkjl->tkil',Y_TO_Z,r))
    elif method=='kimodo':
        local=np.stack([r[:,j] if par<0 else r[:,par].swapaxes(-1,-2)@r[:,j] for j,par in enumerate(parents)],1)
        np.savez_compressed(out/'motion.npz',posed_joints=p,global_rot_mats=r,local_rot_mats=local,fps=motion.fps,times=motion.times)
    elif method=='chois':
        cp=np.concatenate([canonical['joints'],canonical['joints'][:,[20,21]]],1)
        cn=list(HUMANML_NAMES);cn[20:]=['left_wrist_body','right_wrist_body'];cn+=['left_wrist','right_wrist']
        canonical.update(joints=cp,joint_names=cn)
        np.savez_compressed(out/'motion.npz',**canonical)
        (out/'raw').mkdir(exist_ok=True)
        np.savez_compressed(out/'raw/native_prediction.npz',joints=cp,global_rotations=np.einsum('ij,tkjl->tkil',Y_TO_Z,r))
    elif method=='closd':
        order=[CLOSD.index(name) for name in CLOSD]
        world=np.einsum('ij,tkjl->tkil',Y_TO_Z,r)@UPRIGHT.T
        np.savez_compressed(out/'native_physics.npz',joints=(p@Y_TO_Z.T)[:,order],
            rotations_xyzw=Rotation.from_matrix(world.reshape(-1,3,3)).as_quat().reshape(n,24,4),body_names=CLOSD)
        canonical.update(joints=canonical['joints'][:,:22],joint_names=HUMANML_NAMES)
        np.savez_compressed(out/'motion.npz',**canonical)
    else:
        local=np.stack([r[:,j] if par<0 else r[:,par].swapaxes(-1,-2)@r[:,j] for j,par in enumerate(parents)],1)
        if method=='dart':
            np.savez_compressed(out/'motion.npz',**canonical)
            rest=target_template(method)[1]
            np.savez_compressed(out/'smplx_params.npz',transl=canonical['joints'][:,0]-rest[0],
                global_orient=np.einsum('ij,tjk->tik',Y_TO_Z,local[:,0]),body_pose=local[:,1:],
                betas=np.zeros((n,10)),gender='male',pelvis_delta=np.zeros(3),fps=motion.fps)
        elif method=='primal':
            folder=out/'factorial';folder.mkdir(exist_ok=True)
            pose=np.concatenate([p[:,0]-target_template(method)[1][0],Rotation.from_matrix(local.reshape(-1,3,3)).as_rotvec().reshape(n,66)],1)
            # First two frames are context, and are excluded from the scored output.
            np.savez_compressed(folder/'reference.npz',pose=pose[None,2:],joints=p[None,2:],fps=motion.fps,times=motion.times[2:])
            np.savez_compressed(folder/'initial_history.npz',pose=pose[None,:2],betas=np.zeros((1,2,16)))
            (folder/'k_provenance.json').write_text(json.dumps({**provenance,'context_frames_excluded':2}))
            return folder/'reference.npz'
    return out
