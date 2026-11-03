"""Motion sources for the cross-method OpenSim evaluation (E4, config/e4_cross_method_20261103.json).

biomech.load reads an ARDY run directory. This loader also reads the other methods' native outputs
and returns them in the same form (OpenSim Y-up metres, marker source names, fps):

- run directory with motion.npz (MotionLCM, OmniControl, CHOIS, DART, CLoSD, ARDY): canonical Z-up joints;
- a PRIMAL factorial file (<run>/factorial/ground*_reproject*_inertialize*_canonical.npz): canonical Z-up joints;
- a Kimodo case directory with motion.npz holding posed_joints: native Y-up SOMA77 joints, 30 fps.

The SMPL-family and SOMA77 skeletons name the lowest and highest spine joints spine1 / spine3; the
axial markers of biomech use the ARDY names Spine / Spine3, so those two are renamed. The axial marker
offsets are registered per run on the calibration pass, so the renamed joints need not coincide
anatomically with the ARDY ones. CHOIS stores the SMPL-X body wrists as *_wrist_body and hand points as
*_wrist; the body wrists are the joint centres, so they take the wrist names.
"""
from pathlib import Path

import numpy as np

import biomech
from motion import SOMA77_NAMES, Y_TO_Z

RENAME = {'spine1': 'Spine', 'spine3': 'Spine3'}
CHOIS_WRISTS = {'left_wrist_body': 'left_wrist', 'right_wrist_body': 'right_wrist', 'left_wrist': 'left_hand_point',
                'right_wrist': 'right_hand_point'}
KIMODO_FPS = 30.


def load(source):
    path = Path(source)
    file = path if path.suffix == '.npz' else path/'motion.npz'
    with np.load(file, allow_pickle=False) as m:
        if 'posed_joints' in m.files:
            joints = np.asarray(m['posed_joints'], float)
            joints = joints[0] if joints.ndim == 4 else joints
            joints, names, fps = joints @ Y_TO_Z.T, list(SOMA77_NAMES), KIMODO_FPS
        else:
            joints, names, fps = np.asarray(m['joints'], float), [str(n) for n in m['joint_names']], float(m['fps'])
    if 'left_wrist_body' in names:
        names = [CHOIS_WRISTS.get(n, n) for n in names]
    if 'Spine' not in names:
        names = [RENAME.get(n, n) for n in names]
    return biomech.to_opensim(joints), names, fps
