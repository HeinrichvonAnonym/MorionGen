"""Declared native frame transport, independent of model-fitted FK outputs."""
import numpy as np
from .geometry import valid_rotation


def read(motion):
    rig=motion.provenance.get('input_rig')
    if motion.rotations is None:
        return {'status':'unavailable','reason':'Position profile has no independent segment rotations'}
    if rig is None:
        return {'status':'undeclared','reason':'Native rotations retained; no declared common-frame transport'}
    frames=motion.rotations@np.asarray(rig['decoded_native_right_basis'])@np.asarray(rig['segment_frame_basis']).swapaxes(-1,-2)
    valid=valid_rotation(frames)
    frames=frames.copy();frames[~valid]=np.nan
    names=rig['names'];head=names.index('head');chest=names.index('Spine3' if 'Spine3' in names else 'spine3')
    return {'status':'declared_native_frames','frames':frames,'names':names,'valid':valid,
            'head_relative_thorax':frames[:,chest].swapaxes(-1,-2)@frames[:,head],
            'head_relative_valid':valid[:,head]&valid[:,chest],
            'source':'D native rotations plus declared fixed bases',
            'scope':'Transported source-native segment frames; no anatomical axes or fitted head claim'}
