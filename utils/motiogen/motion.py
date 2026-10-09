import json
from pathlib import Path

import numpy as np

HUMANML_NAMES = ['pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2','left_ankle','right_ankle','spine3','left_foot','right_foot','neck','left_collar','right_collar','head','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist']
SOMA77_NAMES = ['pelvis','spine1','spine2','spine3','neck','Neck2','head','HeadEnd','Jaw','LeftEye','RightEye',
    'LeftShoulder','left_shoulder','left_elbow','left_wrist','LeftHandThumb1','LeftHandThumb2','LeftHandThumb3','LeftHandThumbEnd',
    'LeftHandIndex1','LeftHandIndex2','LeftHandIndex3','LeftHandIndex4','LeftHandIndexEnd','LeftHandMiddle1','LeftHandMiddle2',
    'LeftHandMiddle3','LeftHandMiddle4','LeftHandMiddleEnd','LeftHandRing1','LeftHandRing2','LeftHandRing3','LeftHandRing4',
    'LeftHandRingEnd','LeftHandPinky1','LeftHandPinky2','LeftHandPinky3','LeftHandPinky4','LeftHandPinkyEnd','RightShoulder',
    'right_shoulder','right_elbow','right_wrist','RightHandThumb1','RightHandThumb2','RightHandThumb3','RightHandThumbEnd',
    'RightHandIndex1','RightHandIndex2','RightHandIndex3','RightHandIndex4','RightHandIndexEnd','RightHandMiddle1','RightHandMiddle2',
    'RightHandMiddle3','RightHandMiddle4','RightHandMiddleEnd','RightHandRing1','RightHandRing2','RightHandRing3','RightHandRing4',
    'RightHandRingEnd','RightHandPinky1','RightHandPinky2','RightHandPinky3','RightHandPinky4','RightHandPinkyEnd',
    'left_hip','left_knee','left_ankle','left_foot','LeftToeEnd','right_hip','right_knee','right_ankle','right_foot','RightToeEnd']
# A proper rotation maps native (x,y,z) Y-up to (x,-z,y) Z-up.
Y_TO_Z = np.array([[1.,0,0],[0,0,-1],[0,1,0]])


def save_motion(path, joints, names, fps, metadata, rotations=None, **optional):
    joints = np.asarray(joints)
    if joints.ndim != 3 or joints.shape[1:] != (len(names), 3) or not np.isfinite(joints).all():
        raise ValueError('joints must be finite [T,J,3] matching joint_names')
    if not np.isfinite(fps) or fps <= 0 or len(joints) == 0 or len(set(names)) != len(names):
        raise ValueError('Invalid motion dimensions, names or FPS')
    arrays = {'joints': joints, 'joint_names': np.asarray(names, dtype=str), 'fps': np.array(float(fps)),
              'metadata_json': np.array(json.dumps(metadata, ensure_ascii=False))}
    if rotations is not None:
        rotations = np.asarray(rotations)
        if rotations.shape != (len(joints), len(names), 3, 3) or not np.isfinite(rotations).all():
            raise ValueError('global rotations must be finite [T,J,3,3]')
        if not np.allclose(rotations @ rotations.swapaxes(-1,-2), np.eye(3), atol=2e-3) or not np.allclose(np.linalg.det(rotations), 1, atol=2e-3):
            raise ValueError('rotation matrices are not SO(3)')
        arrays['global_rotations'] = rotations
    arrays.update({k: np.asarray(v) for k,v in optional.items() if v is not None})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_motion(path):
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key].copy() for key in data.files}
    result['metadata'] = json.loads(str(result.pop('metadata_json')))
    if result['metadata'].get('units') != 'm' or result['metadata'].get('axes') != 'right-handed Z-up':
        raise ValueError('Expected canonical metres and right-handed Z-up')
    joints=result['joints']; names=result['joint_names']; fps=float(result['fps'])
    if joints.ndim!=3 or joints.shape[1:]!=(len(names),3) or len(joints)==0 or not np.isfinite(joints).all() or not np.isfinite(fps) or fps<=0:
        raise ValueError('Invalid canonical dimensions, values or FPS')
    if len(set(names.tolist()))!=len(names):
        raise ValueError('Canonical joint names must be unique')
    return result


def canonicalize(joints, axes, rotations=None, units='m'):
    if axes not in ('Y-up','Z-up') or units not in ('m','cm','mm'):
        raise ValueError('Unsupported source axes/units; explicit conversion is required')
    transform = Y_TO_Z if axes == 'Y-up' else np.eye(3)
    xyz = (np.asarray(joints)*{'m':1.,'cm':0.01,'mm':0.001}[units]) @ transform.T
    # Rotate world basis; local anatomical basis stays unchanged.
    rot = None if rotations is None else np.einsum('ab,tjbc->tjac', transform, rotations)
    return xyz, rot


def native_positions(points, axes):
    return np.asarray(points) @ (Y_TO_Z if axes == 'Y-up' else np.eye(3))
