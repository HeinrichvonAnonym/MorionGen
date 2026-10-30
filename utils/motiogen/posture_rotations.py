"""Recover continuous angles from source-calibrated SOMA/Core global frames.

Angles are an explicit computational convention awaiting the independent
rating exercise. Qualitative twist/abduction/support flags are not inferred
with undocumented thresholds. RULA end-range pronation needs individual ROM.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .motion import Y_TO_Z
from .posture_geometry import unit

def recovered_angles(xyz,rotations,calibration):
    names=calibration['names'];idx={n:i for i,n in enumerate(names)}
    neutral=np.asarray(calibration['neutral_joints'],float);rot=np.asarray(rotations,float)
    count=calibration.get('rotation_joint_count',len(names))
    if rot.shape!=(len(xyz),count,3,3):raise ValueError('Source rotation/joint dimensions differ')
    if not np.allclose(rot@rot.swapaxes(-1,-2),np.eye(3),atol=2e-3):raise ValueError('Non-orthogonal source rotations')
    errors=[]
    for child,parent in enumerate(calibration['parents']):
        if parent<0 or child>=count:continue
        predicted=xyz[:,parent]+np.einsum('tij,j->ti',rot[:,parent],neutral[child]-neutral[parent])
        errors.append(np.linalg.norm(predicted-xyz[:,child],axis=-1))
    fk_error=float(np.max(errors))
    if fk_error>2e-4:raise ValueError(f'Neutral frame FK does not reproduce saved positions: {fk_error} m')
    right=unit(neutral[idx['right_hip']]-neutral[idx['left_hip']]);up=np.array([0.,1.,0.]);right=unit(right-up*np.dot(up,right));forward=unit(np.cross(up,right))
    trunk=idx['Spine3'] if 'Spine3' in idx else idx['spine3']
    world_up=np.array([0.,0.,1.]);pelvis_right=np.einsum('tij,j->ti',rot[:,idx['pelvis']],right);pelvis_right[:,2]=0;pelvis_right=unit(pelvis_right)
    world_forward=unit(np.cross(world_up,pelvis_right))
    torso_up=np.einsum('tij,j->ti',rot[:,trunk],up);torso_forward=np.einsum('tij,j->ti',rot[:,trunk],forward);torso_right=np.einsum('tij,j->ti',rot[:,trunk],right)
    def projected(v,x,y):return np.rad2deg(np.arctan2(np.sum(v*x,axis=-1),np.sum(v*y,axis=-1)))
    result={'trunk_flexion_deg':projected(torso_up,world_forward,world_up),
            'trunk_side_bend_deg':projected(torso_up,pelvis_right,world_up),
            'trunk_yaw_relative_pelvis_deg':projected(torso_right,world_forward,pelvis_right)}
    head_up=np.einsum('tij,j->ti',rot[:,idx['head']],up)
    result['neck_flexion_deg']=projected(head_up,torso_forward,torso_up)
    result['neck_side_bend_deg']=projected(head_up,torso_right,torso_up)
    for side in ['left','right']:
        arm=unit(xyz[:,idx[side+'_elbow']]-xyz[:,idx[side+'_shoulder']])
        flex=projected(arm,torso_forward,-torso_up)
        singular=(np.sum(arm*torso_forward,axis=-1)**2+np.sum(arm*torso_up,axis=-1)**2)<1e-8
        flex[singular]=np.nan
        result[side+'_upper_arm_flexion_deg']=flex
        result[side+'_upper_arm_abduction_deg']=projected(arm,torso_right if side=='right' else -torso_right,-torso_up)
        hand=idx[side+'_wrist'];elbow=idx[side+'_elbow'];prefix=side.title()
        if 'hand_landmarks' in calibration:
            # Template mesh landmarks calibrate axes only; they are not observed
            # finger joints and are never added to the motion skeleton.
            distal=np.asarray(calibration['hand_landmarks'][side]['middle'],float)
            thumb=np.asarray(calibration['hand_landmarks'][side]['thumb'],float)
        else:
            distal=neutral[idx[prefix+'HandMiddle1'] if prefix+'HandMiddle1' in idx else idx[prefix+'HandEnd']]
            thumb=neutral[idx[prefix+'HandThumb1']]
        longitudinal=unit(distal-neutral[hand]);radial=thumb-neutral[hand]
        radial=unit(radial-longitudinal*np.dot(longitudinal,radial));normal=unit(np.cross(longitudinal,radial))
        basis=np.stack([longitudinal,radial,normal],axis=1)
        relative=np.einsum('tji,tjk->tik',rot[:,elbow],rot[:,hand])
        anatomical=np.einsum('ij,tjk,kl->til',basis.T,relative,basis)
        # Intrinsic ZYX: deviation, flexion, axial hand-relative-forearm twist.
        angles=Rotation.from_matrix(anatomical).as_euler('ZYX',degrees=True)
        result[side+'_wrist_deviation_deg']=angles[:,0]
        result[side+'_wrist_flexion_deg']=angles[:,1]
        result[side+'_hand_relative_forearm_twist_deg']=angles[:,2]
    return result,{'fk_max_error_m':fk_error,'neutral_source_sha256':calibration['sha256'],
        'angle_convention':'Source global rotations; neutral Y-up skeletal frames. Trunk sagittal/coronal projections relative to pelvis heading and gravity; head relative to upper torso. Wrist intrinsic ZYX in neutral hand longitudinal/radial/palm-normal basis.',
        'limits':'Projected arm flexion is undefined for pure lateral elevation. Hand-relative-forearm twist is not a substitute for RULA end-range forearm pronation. Qualitative correction flags stay unknown; no zero-angle threshold invented.',
        'human_validation':'pending_no_raters'}
