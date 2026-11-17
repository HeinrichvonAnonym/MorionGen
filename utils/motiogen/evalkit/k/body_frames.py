"""Declared native-frame targets for existing body coordinates, D only."""
from .native_frames import read
from .rig_geometry import transport


def targets(motion):
    native=read(motion)
    if native['status']!='declared_native_frames':return {},{'status':'unavailable','reason':native.get('reason')}
    basis=transport(motion.provenance['input_rig']);names=native['names']
    chest='Spine3' if 'Spine3' in names else 'spine3'
    result={body:native['frames'][:,names.index(node)]@basis.T for body,node in [('pelvis','pelvis'),('torso',chest)]}
    return result,{'status':'declared_source_frames','correspondence':{'pelvis':'pelvis','torso':chest},
                   'transport':basis.tolist(),'formula':'model body target = transported D source frame @ rest-rig-to-model basis transpose',
                   'scope':'Project kinematic body/source correspondence; no anatomical calibration; no R or clip pose estimation'}
