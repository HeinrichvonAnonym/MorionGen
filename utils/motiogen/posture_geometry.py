"""Position-observable joint angles, independent of ergonomic lookup tables.

Three-point elbow/knee angles are identifiable from joint centres. Hand axial
rotation and supported/load categories are not inferred from endpoint motion.
"""
import numpy as np
from .motion import load_motion,canonicalize,SOMA77_NAMES

def unit(v):
    norm=np.linalg.norm(v,axis=-1,keepdims=True)
    return np.divide(v,norm,out=np.full_like(v,np.nan,dtype=float),where=norm>1e-8)

def flexion(a,b,c):
    """Zero is straight; invariant to rigid transforms and scale."""
    dot=(unit(a-b)*unit(c-b)).sum(-1)
    return 180-np.rad2deg(np.arccos(np.clip(dot,-1,1)))

def read_positions(row):
    if row['input_format']=='kimodo_soma77':
        with np.load(row['motion_path'],allow_pickle=False) as d:
            xyz,_=canonicalize(d['posed_joints'],'Y-up')
        return xyz,SOMA77_NAMES,30.
    d=load_motion(row['motion_path'])
    names=d['joint_names'].tolist()
    if row['method']=='chois':
        # Historical adapter named middle-finger anchors wrist. Anatomical
        # centres are the body wrist channels 20/21, not those anchors.
        names=[n.replace('left_wrist','left_middle_anchor').replace('right_wrist','right_middle_anchor') if n in ['left_wrist','right_wrist'] else n.replace('_wrist_body','_wrist') for n in names]
    return d['joints'],names,float(d['fps'])

def observable_angles(xyz,names):
    xyz=np.asarray(xyz,float);index={n:i for i,n in enumerate(names)}
    get=lambda n:xyz[:,index[n]]
    result={}
    for side in ['left','right']:
        result[side+'_elbow_flexion_deg']=flexion(get(side+'_shoulder'),get(side+'_elbow'),get(side+'_wrist'))
        result[side+'_knee_flexion_deg']=flexion(get(side+'_hip'),get(side+'_knee'),get(side+'_ankle'))
    shoulders=(get('left_shoulder')+get('right_shoulder'))/2
    hips=(get('left_hip')+get('right_hip'))/2
    torso=unit(shoulders-hips)
    # Orientation to world vertical is observable; do not silently call this
    # sagittal trunk flexion: twist/sidebend must be resolved separately.
    result['torso_inclination_to_vertical_deg']=np.rad2deg(np.arccos(np.clip(torso[:,2],-1,1)))
    for side in ['left','right']:
        arm=unit(get(side+'_elbow')-get(side+'_shoulder'))
        result[side+'_upper_arm_elevation_deg']=np.rad2deg(np.arccos(np.clip((-torso*arm).sum(-1),-1,1)))
    return result

def episodes(mask,fps):
    """All contiguous exposure intervals, including intervals at clip edges."""
    edges=np.diff(np.r_[False,np.asarray(mask,bool),False].astype(int))
    starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
    return [{'start_seconds':int(a)/fps,'end_seconds':int(b)/fps,'duration_seconds':int(b-a)/fps} for a,b in zip(starts,ends)]
