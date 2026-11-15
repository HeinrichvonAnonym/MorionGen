"""Format-specific arm fitting policy and source-fidelity diagnostics.

Joint-centre inputs constrain bone geometry. Registering their marker offsets
from an imperfect IK solution can absorb the very elbow error being measured.
These formats keep arm markers at the model joint centres instead.
"""
import numpy as np
from pathlib import Path
from .config import CONFIG


def policy(method):
    settings = CONFIG['arm_fit']
    return dict(settings) if method in settings['methods'] else {}


def configure_model(path, settings):
    if not settings:
        return
    import opensim as osim
    model = osim.Model(str(path))
    lo, hi = np.deg2rad(settings['shoulder_flexion_range_deg'])
    for side in 'rl':
        coordinate = model.updCoordinateSet().get('arm_flex_' + side)
        coordinate.setRangeMin(float(lo))
        coordinate.setRangeMax(float(hi))
    model.printToXML(str(path))


def inverse_kinematics(model_path, trc, out, t0, t1, weights, settings, lock_lumbar=False):
    """Continue from a bounded first-frame solution into the expanded range.

    Expanded Euler ranges can put neutral-start IK in an opposite shoulder
    branch, especially for abducted straight arms. Solve the first observed
    frame using the original range, then release that range for the full solve.
    Defaults in the delivered model remain unchanged (foot calibration uses them).
    """
    import biomech
    if not settings or settings['shoulder_flexion_range_deg'] == settings['initialization_range_deg']:
        return biomech.inverse_kinematics(model_path,trc,out,t0,t1,weights,lock_lumbar=lock_lumbar)
    import opensim as osim
    folder=Path(out).parent/'initialization';folder.mkdir(parents=True,exist_ok=True)
    model=osim.Model(str(model_path))
    lo,hi=np.deg2rad(settings['initialization_range_deg'])
    for side in 'lr':
        c=model.updCoordinateSet().get('arm_flex_'+side)
        c.setRangeMin(float(lo));c.setRangeMax(float(hi))
    model.printToXML(str(folder/'bounded.osim'))
    # TRC decimal serialization can move the boundary by a few ulps. A tiny
    # enclosing interval selects the same observed frame without end < start.
    biomech.inverse_kinematics(folder/'bounded.osim',trc,folder/'first.mot',t0-1e-6,t0+1e-6,weights,lock_lumbar=lock_lumbar)
    _,labels,data=biomech.read_storage(folder/'first.mot')
    model=osim.Model(str(model_path))
    for label,value in zip(labels,data[0]):
        coordinate=model.updCoordinateSet().get(label)
        coordinate.setDefaultValue(float(np.deg2rad(value) if coordinate.getMotionType()==1 else value))
    model.printToXML(str(folder/'warm.osim'))
    return biomech.inverse_kinematics(folder/'warm.osim',trc,out,t0,t1,weights,lock_lumbar=lock_lumbar)


def geometry_quality(source, fitted, angle_limit, position_limit):
    """Arrays [time, shoulder/elbow/wrist, xyz]; assess angle AND endpoints.

    Validity is fidelity to the observed input, not external anatomical accuracy.
    Degenerate or missing segments retain false masks and NaN angular errors.
    """
    source, fitted = np.asarray(source), np.asarray(fitted)
    a, b = np.diff(source, axis=1), np.diff(fitted, axis=1)
    an, bn = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    good = (np.isfinite(source).all(axis=(1, 2)) & np.isfinite(fitted).all(axis=(1, 2))
            & (an > 1e-6).all(axis=1) & (bn > 1e-6).all(axis=1))
    with np.errstate(divide='ignore', invalid='ignore'):
        a, b = a / an[..., None], b / bn[..., None]
    acos = lambda x: np.rad2deg(np.arccos(np.clip(x, -1, 1)))
    direction = acos((a * b).sum(-1))
    elbow = acos((b[:, 0] * b[:, 1]).sum(-1)) - acos((a[:, 0] * a[:, 1]).sum(-1))
    direction[~good] = np.nan
    elbow[~good] = np.nan
    position = np.linalg.norm(source - fitted, axis=-1)
    angle_valid = good & (np.abs(elbow) <= angle_limit) & (direction <= angle_limit).all(axis=1)
    position_valid = good & (position <= position_limit).all(axis=1)
    return {'direction_error_deg': direction, 'elbow_bias_deg': elbow,
            'position_error_m': position, 'observed': good,
            'angle_valid': angle_valid, 'position_valid': position_valid,
            'valid': angle_valid & position_valid}


def assess(model_path, points, names, fps, times, labels, coordinates, settings):
    """Check the final filtered coordinates consumed by downstream modules."""
    import opensim as osim
    import biomech
    model = osim.Model(str(model_path))
    state = model.initSystem()
    fitted = {side: [] for side in 'lr'}
    for row in coordinates:
        biomech._set(model, state, labels, row)
        for side in 'lr':
            fitted[side].append([biomech._child_origin(model, state, joint + '_' + side)
                                 for joint in ('acromial', 'elbow', 'radius_hand')])
    result = {}
    source_times = np.arange(len(points)) / fps
    for side, prefix in (('l', 'left'), ('r', 'right')):
        source = np.stack([np.stack([np.interp(times, source_times, points[:, names.index(prefix + '_' + joint), axis])
                                    for axis in range(3)], -1) for joint in ('shoulder', 'elbow', 'wrist')], 1)
        result[side] = geometry_quality(source, fitted[side], settings['quality_angle_deg'], settings['quality_position_m'])
    return result
