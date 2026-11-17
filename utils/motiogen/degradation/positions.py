"""Loss-preserving position profiles. No target-template FK in this path."""
import json
from pathlib import Path
import numpy as np
from motiogen.motion import HUMANML_NAMES, Y_TO_Z
from motiogen.evalkit.k.motion import Motion

SOURCE = {
    'pelvis': 'Hips', 'spine1': 'Spine', 'spine2': 'Spine1', 'spine3': 'Spine3',
    'neck': 'Neck', 'head': 'Head',
    **{f'{side}_{target}': prefix+source for side, prefix in [('left', 'Left'), ('right', 'Right')]
       for target, source in [('hip','UpLeg'), ('knee','Leg'), ('ankle','Foot'), ('foot','ToeBase'),
                              ('collar','Shoulder'), ('shoulder','Arm'), ('elbow','ForeArm'), ('wrist','Hand')]}
}


def project(reference, method):
    lookup = {n: i for i, n in enumerate(reference.names)}
    mapping = dict(SOURCE)
    if 'Spine3' not in lookup:
        mapping['spine3'] = 'Spine2'
    if 'Neck' not in lookup:
        mapping['neck'] = 'Neck1'
    missing = [mapping[n] for n in HUMANML_NAMES if mapping[n] not in lookup]
    if missing:
        raise ValueError(f'Unobserved source nodes: {missing}')
    p = reference.positions[:, [lookup[mapping[n]] for n in HUMANML_NAMES]]
    return Motion(p, list(HUMANML_NAMES), reference.times, method, provenance={
        'source_map': mapping, 'geometry': 'observed_joint_positions',
        'head_axial_rotation': 'unobservable', 'independent_rotations': False})


def write(motion, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/'motion.npz', joints=motion.positions@Y_TO_Z.T,
                        joint_names=motion.names, fps=motion.fps, times=motion.times,
                        metadata_json=json.dumps({'baseline':motion.method,'units':'m','axes':'right-handed Z-up'}))
    # No R, matrices, shape parameters, anatomical labels or mapping residuals.
    (folder/'k_provenance.json').write_text(json.dumps({
        'origin':'controlled_position_projection_v1','prior_joints':[],
        'shape':'observed_positions','independent_head_rotation':'unavailable'})+'\n')
    return folder
