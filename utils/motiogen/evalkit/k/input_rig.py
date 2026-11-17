"""Declared kinematic input geometry. Never reads a reference or scoring file."""
import json
from pathlib import Path
import numpy as np
from .config import digest
from .geometry import valid_rotation


def attach(motion, source):
    folder=Path(source) if Path(source).is_dir() else Path(source).parent
    declared=motion.provenance.get('degradation',{})
    name=declared.get('rig')
    if name is None:
        if (folder/'input_rig.json').exists():
            raise ValueError('Input rig exists without a declared hash binding')
        motion.provenance['input_geometry']={'status':'positions_only' if motion.rotations is None else 'legacy_asset',
            'rest_geometry':'unavailable','neutral_trunk_length':'unidentified'}
        return
    if motion.method in ('motionlcm','omnicontrol'):
        raise ValueError('Position-only profile must not acquire an undeclared rest rig')
    if name!='input_rig.json':raise ValueError('Only the local input_rig.json contract is supported')
    path=folder/name
    if path.resolve().parent!=folder.resolve():raise ValueError('Input rig escapes input directory')
    if digest(path)!=declared.get('rig_sha256'):raise ValueError('Input rig hash mismatch')
    rig=json.loads(path.read_text())
    if rig.get('version')!='source-rest-rig-1':raise ValueError('Unsupported input rig version')
    if rig.get('units')!='m' or rig.get('axes')!='right-handed Y-up':raise ValueError('Unsupported rig coordinates')
    if rig.get('geometry_policy')!='fixed_for_entire_clip':raise ValueError('Rig geometry must be fixed')
    names=rig['names'];parents=rig['parents'];rest=np.asarray(rig['neutral_joints'],float)
    n=len(names)
    if not n or len(set(names))!=n or len(parents)!=n:raise ValueError('Invalid rig node layout')
    if rest.shape!=(n,3) or not np.isfinite(rest).all():raise ValueError('Invalid fixed geometry')
    if parents[0]!=-1 or any(type(p) is not int or not 0<=p<j for j,p in enumerate(parents[1:],1)):
        raise ValueError('Rig must be one topologically ordered rooted tree')
    if any(np.linalg.norm(rest[j]-rest[p])<1e-9 for j,p in enumerate(parents) if p>=0):
        raise ValueError('Zero-length rig edge')
    basis=np.asarray(rig['segment_frame_basis'],float)
    right=np.asarray(rig['decoded_native_right_basis'],float)
    if basis.shape!=(n,3,3) or not valid_rotation(basis).all() or right.shape!=(3,3) or not valid_rotation(right):
        raise ValueError('Invalid rig rotation bases')
    observed=rig['observed_nodes']
    if len(observed)!=n or any(type(x) is not bool for x in observed):raise ValueError('Invalid node support')
    if motion.rotations is None or motion.rotations.shape[1]!=n:raise ValueError('Rig/rotation count mismatch')
    from .calibration import CLOSD
    expected=CLOSD if motion.method=='closd' else names
    if list(motion.rotation_names)!=expected:raise ValueError('Rig/native rotation order mismatch')
    motion.provenance['inputs'][str(path.resolve())]=digest(path)
    motion.provenance['input_rig']=rig
    motion.provenance['input_geometry']={'status':'declared_fixed_rig','sha256':digest(path),
        'rest_geometry':'declared_kinematic_configuration','anatomical_calibration':'unverified',
        'native_frames':'declared_coordinate_transport_only','stock_mesh_fk_compatible':False}


def geometry_targets(motion):
    """Separate fixed rest dimensions from observable positional segment lengths."""
    rig=motion.provenance.get('input_rig')
    if rig:
        points=np.asarray(rig['neutral_joints'])[None];names=rig['names'];source='declared_input_rig'
    else:
        points=motion.positions;names=motion.names;source='observed_segment_distance_median'
    rename={'left_wrist_body':'left_wrist','right_wrist_body':'right_wrist'}
    idx={rename.get(n,n):j for j,n in enumerate(names) if n not in ('left_wrist','right_wrist') or n+'_body' not in names}
    p=lambda n:points[:,idx[n]]
    distance=lambda a,b:float(np.nanmedian(np.linalg.norm(p(a)-p(b),axis=-1)))
    result={'source':source,'lengths_m':{},'scope':'kinematic geometry; not anatomical calibration'}
    for side in ('left','right'):
        for part,a,b in [('thigh','hip','knee'),('shank','knee','ankle'),('foot','ankle','foot'),
                         ('upper_arm','shoulder','elbow'),('forearm','elbow','wrist')]:
            result['lengths_m'][side+'_'+part]=distance(side+'_'+a,side+'_'+b)
    result['lengths_m']['hip_width']=distance('left_hip','right_hip')
    result['lengths_m']['shoulder_width']=distance('left_shoulder','right_shoulder')
    if rig:
        result['lengths_m']['rest_hip_to_shoulder_midpoint']=float(np.linalg.norm(
            (p('left_shoulder')[0]+p('right_shoulder')[0]-p('left_hip')[0]-p('right_hip')[0])/2))
    else:
        result['neutral_trunk_length']='unidentified; retain explicit model prior, never infer from max endpoint span'
    return result
