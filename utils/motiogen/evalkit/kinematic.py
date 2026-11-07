"""Kinematic dimension: reach, jerk, bone length, hover / penetration depth and foot skating.

Joints are canonical Z-up metres (motion.npz). Hover, penetration and the contact spans of foot skating use the
per-frame sole elevation and contact of the residual dimension (frames.npz), so they share its floor input and
calibrated reference. No OpenSim import; runs in any environment with numpy.
"""
import numpy as np

from .spec import SPEC, SIDES

C = SPEC.get('kinematic', {})
BONES = (('pelvis', 'left_hip'), ('pelvis', 'right_hip'), ('left_hip', 'left_knee'), ('right_hip', 'right_knee'),
         ('left_knee', 'left_ankle'), ('right_knee', 'right_ankle'), ('left_ankle', 'left_foot'),
         ('right_ankle', 'right_foot'), ('left_collar', 'left_shoulder'), ('right_collar', 'right_shoulder'),
         ('left_shoulder', 'left_elbow'), ('right_shoulder', 'right_elbow'), ('left_elbow', 'left_wrist'),
         ('right_elbow', 'right_wrist'), ('neck', 'head'))
FOOT_JOINT = {'heel': '{}_ankle', 'toe': '{}_foot'}
SIDE_NAME = {'r': 'right', 'l': 'left'}


def step_frames(fps, step_seconds=.1):
    if not np.isfinite(fps) or fps <= 0 or step_seconds <= 0:
        raise ValueError('Positive FPS and step required')
    return max(1, int(round(step_seconds*fps)))


def common_step_jerk(joints, fps, step_seconds=.1):
    """Third difference on a common time step, so 20 Hz and 30 Hz outputs compare.

    Same normalisation as the archived native metric (component RMS over all
    joints and axes); only the difference step changes from 1/FPS to ~0.1 s.
    """
    xyz = np.asarray(joints, float); k = step_frames(fps, step_seconds)
    if len(xyz) <= 3*k:
        raise ValueError('Motion shorter than three common steps')
    d = (xyz[3*k:]-3*xyz[2*k:-k]+3*xyz[k:-2*k]-xyz[:-3*k])/(k/fps)**3
    return float(np.sqrt(np.mean(d**2)))


def native_jerk(joints, fps):
    return float(np.sqrt(np.mean((np.diff(np.asarray(joints, float), n=3, axis=0)*fps**3)**2)))


def bone_length_cv(joints, names, bones=BONES):
    """Within-clip coefficient of variation of each named bone length."""
    xyz = np.asarray(joints, float); index = {str(n): i for i, n in enumerate(names)}; result = {}
    for a, b in bones:
        if a not in index or b not in index:
            continue
        length = np.linalg.norm(xyz[:, index[a]]-xyz[:, index[b]], axis=-1)
        result[f'{a}-{b}'] = {'mean_m': float(length.mean()), 'cv': float(length.std()/length.mean())}
    return result


def bone_ratio(joints, names, bones=BONES):
    """Per frame and bone: length over the clip median length {bone: [frame]}."""
    xyz = np.asarray(joints, float); index = {str(n): i for i, n in enumerate(names)}; out = {}
    for a, b in bones:
        if a in index and b in index:
            length = np.linalg.norm(xyz[:, index[a]]-xyz[:, index[b]], axis=-1)
            out[f'{a}-{b}'] = length/np.median(length)
    return out


def threshold_success(errors, thresholds=(.02, .05, .1)):
    errors = np.asarray(errors, float)
    return {f'success_{round(t*100)}cm': float((errors <= t).mean()) for t in thresholds}


def facing_heading(joints, names, frame=0):
    """Planar facing from hip and shoulder axes; degrees from +X in the Z-up report frame."""
    xyz = np.asarray(joints, float); ix = {str(n): i for i, n in enumerate(names)}
    across = xyz[frame, ix['right_hip']]-xyz[frame, ix['left_hip']]
    if 'right_shoulder' in ix and 'left_shoulder' in ix:
        across = across+xyz[frame, ix['right_shoulder']]-xyz[frame, ix['left_shoulder']]
    forward = np.cross([0., 0., 1.], across); forward[2] = 0; forward /= np.linalg.norm(forward)
    return forward, float(np.degrees(np.arctan2(forward[1], forward[0])))


def reach(joints, names, fps, target, time_seconds, joint):
    """Wrist reach at the constrained frame. Success counts the rigid refit error (shoulder plus median upper-arm and
    forearm lengths along the frame's directions), and fails whenever either arm bone at that frame exceeds its clip
    median by more than kinematic.reach_bone_stretch_max."""
    xyz = np.asarray(joints, float); ix = {str(n): i for i, n in enumerate(names)}
    frame = int(round(time_seconds*fps)); target = np.asarray(target, float)
    side = joint.split('_')[0]; other = 'left' if side == 'right' else 'right'
    sh, el, wr = (xyz[:, ix[f'{side}_{j}']] for j in ('shoulder', 'elbow', 'wrist'))
    upper = np.linalg.norm(el-sh, axis=-1); fore = np.linalg.norm(wr-el, axis=-1)
    unit = lambda v: v/np.linalg.norm(v)
    refit = sh[frame]+np.median(upper)*unit(el[frame]-sh[frame])+np.median(fore)*unit(wr[frame]-el[frame])
    error = float(np.linalg.norm(wr[frame]-target)); refit_error = float(np.linalg.norm(refit-target))
    stretch = float(max(upper[frame]/np.median(upper), fore[frame]/np.median(fore))-1)
    stretched = stretch > C['reach_bone_stretch_max']
    values = {'wrist_error_m': error, 'refit_wrist_error_m': refit_error, 'stretch_contribution_m': refit_error-error,
              'arm_bone_stretch_at_frame': stretch, 'stretch_fail': bool(stretched),
              'other_wrist_distance_m': float(np.linalg.norm(xyz[frame, ix[other+'_wrist']]-target)),
              'shoulder_distance_ratio': float(np.linalg.norm(target-sh[0])/(upper[0]+fore[0])),
              'arm_bone_cv': float(np.mean([upper.std()/upper.mean(), fore.std()/fore.mean()]))}
    values['other_hand_closer'] = float(values['other_wrist_distance_m'] < error)
    values.update({k: 0. if stretched else v for k, v in threshold_success([refit_error], C['reach_thresholds_m']).items()})
    return values


def _spans(mask, min_frames):
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    return [(a, b) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)) if b-a >= min_frames]


def skating(joints, names, fps, frames):
    """Horizontal slip of the foot joint over each contact span of the matching sole landmark (heel: ankle joint,
    toe: foot joint), contact from the residual dimension inside the scored window. Slip is the distance between
    the span's first and last kinematic.skate_end_seconds medians."""
    xyz = np.asarray(joints, float); ix = {str(n): i for i, n in enumerate(names)}
    t = np.arange(len(xyz))/fps; ft = frames['times']
    nearest = lambda x: np.asarray(x)[np.clip(np.searchsorted(ft, t), 0, len(ft)-1)]
    scored = nearest(frames['scored']).astype(bool)
    end = max(1, int(round(C['skate_end_seconds']*fps))); min_frames = max(2*end, int(round(C['skate_min_span_seconds']*fps)))
    spans = []
    for s in SIDES:
        for k, j in FOOT_JOINT.items():
            name = j.format(SIDE_NAME[s])
            if name not in ix:
                continue
            contact = nearest(frames[f'contact_{s}_{k}']).astype(bool) & scored
            p = xyz[:, ix[name], :2]
            for a, b in _spans(contact, min_frames):
                slip = float(np.linalg.norm(np.median(p[b-end:b], 0)-np.median(p[a:a+end], 0)))
                spans.append({'side': s, 'landmark': k, 'start_s': float(a/fps), 'seconds': float((b-a)/fps), 'slip_m': slip})
    slips = np.array([x['slip_m'] for x in spans])
    return {'spans': len(spans), 'max_slip_m': float(slips.max()) if len(spans) else None,
            'median_slip_m': float(np.median(slips)) if len(spans) else None,
            'max_slip_speed_m_s': float(max(x['slip_m']/x['seconds'] for x in spans)) if len(spans) else None,
            'skating': bool((slips > C['skate_slip_m']).any()) if (len(spans) and C.get('skate_slip_m')) else None,
            'per_span': spans}


def floor_depth(frames):
    """Hover and penetration depth of the lowest sole landmark relative to the input floor, scored window."""
    low = np.asarray(frames['lowest_elevation_m'])[np.asarray(frames['scored'], bool)]
    pen = np.maximum(-low, 0.)
    return {'penetration_depth_p95_m': float(np.percentile(pen, 95)), 'penetration_depth_max_m': float(pen.max()),
            'hover_height_median_m': float(np.median(np.maximum(low, 0.))),
            'hover_height_p95_m': float(np.percentile(np.maximum(low, 0.), 95))}


def evaluate(joints, names, fps, frames=None, reach_task=None):
    """Summary of the kinematic dimension. frames: residual-dimension frames (None: joint-only metrics).
    reach_task: {'position_m', 'time_seconds', 'joint'} of a wrist target."""
    names = [str(n) for n in names]
    cv = bone_length_cv(joints, names)
    ratio = bone_ratio(joints, names)
    out = {'jerk_0p1s': common_step_jerk(joints, fps), 'jerk_native': native_jerk(joints, fps),
           'bone_cv_mean': float(np.mean([v['cv'] for v in cv.values()])), 'bone_cv_max': float(max(v['cv'] for v in cv.values())),
           'bone_stretch_max': float(max(r.max() for r in ratio.values())-1), 'bone_cv': cv}
    if frames is not None:
        out['floor'] = floor_depth(frames)
        out['skating'] = skating(joints, names, fps, frames)
    if reach_task is not None:
        out['reach'] = reach(joints, names, fps, reach_task['position_m'], reach_task['time_seconds'], reach_task['joint'])
    return out
