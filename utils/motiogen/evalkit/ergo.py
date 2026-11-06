"""RULA/REBA frame classes (config/ergo_rula_reba_20261105.json, revision 3) from the shared IK coordinates.

Inputs the marker set cannot observe stay unknown, so each frame gets the admissible score range [lower, upper]; with
the action threshold (REBA 4, RULA 5) a frame is certain risk (lower >= threshold), indeterminate, or certain
acceptable (upper < threshold); frames failing the IK gate are unmeasurable. Legs and activity come from the
height-only contact of the residual dimension; no dimension gates another. Neck and shoulder elevation come from the
source joints minus the skeleton's neutral reading. OpenSim python (trunk angles use the model).
"""
import json
import math
from pathlib import Path

import numpy as np

from .spec import SPEC as KIT, ROOT, SIDES
import ergonomics

SPEC = json.loads((ROOT/'config/ergo_rula_reba_20261105.json').read_text())
CONV = SPEC['conventions_not_derived']
R3 = next(r['rules'] for r in SPEC['revisions'] if r.get('id') == 'r3')
NEUTRAL = json.loads((ROOT/'config/ergo_neutral_offsets_20261105.json').read_text())
CLASSES = ('unmeasurable', 'certain_risk', 'indeterminate', 'certain_acceptable')
RISK = {'REBA': CONV['risk_threshold_REBA'], 'RULA': CONV['risk_threshold_RULA']}
SMPL22_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
SMPL22_NAMES = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3',
                'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow',
                'right_elbow', 'left_wrist', 'right_wrist']
JOINT_ALIAS = {'LeftArm': 'left_shoulder', 'RightArm': 'right_shoulder', 'Hips': 'pelvis', 'Neck1': 'neck', 'Neck': 'neck',
               'LeftShoulder': 'left_collar', 'RightShoulder': 'right_collar'}
SOURCE_MIN = {'neck_flexion_deg': 10., 'neck_twisted': 0, 'neck_side_bent': 0, 'shoulder_raised': 0}
SIDE_NAME = {'r': 'right', 'l': 'left'}


def object_free(prompt):
    """Known zero load: the task text names no object. No text (e.g. unconditioned generation) is object-free."""
    if prompt is None:
        return True
    text = prompt.lower()
    return not any(k in text for k in SPEC['object_keywords'])


def trunk_angles(model_path, labels, rows):
    """Trunk line (hip-joint midpoint to the neck-base marker neck_c) against ground vertical, in the pelvis sagittal
    and frontal planes (degrees)."""
    import opensim as osim
    import biomech
    model = osim.Model(str(model_path)); state = model.initSystem()
    bodies = model.getBodySet(); pelvis = bodies.get('pelvis')
    hips = [bodies.get('femur_r'), bodies.get('femur_l')]; neck = model.getMarkerSet().get('neck_c')
    vec = lambda v: np.array([v.get(k) for k in range(3)])
    out = np.zeros((len(rows), 2))
    for i, row in enumerate(rows):
        biomech._set(model, state, labels, row)
        mid = sum(vec(h.getPositionInGround(state)) for h in hips)/2
        axis = vec(neck.getLocationInGround(state))-mid
        forward = vec(pelvis.expressVectorInGround(state, osim.Vec3(1, 0, 0)))
        forward[1] = 0.; forward /= np.linalg.norm(forward)
        lateral = np.cross(forward, [0., 1., 0.])
        out[i] = [math.degrees(math.atan2(axis @ forward, axis[1])), math.degrees(math.atan2(axis @ lateral, axis[1]))]
    return out


def neck_angles(j, idx):
    """Neck-to-head line against pelvis-to-neck: flexion (sagittal, forward positive) and side bend (degrees), (N, 2)."""
    up = j[:, idx['neck']]-j[:, idx['pelvis']]; up /= np.linalg.norm(up, axis=1, keepdims=True)
    lateral = j[:, idx['right_hip']]-j[:, idx['left_hip']]; lateral -= np.sum(lateral*up, 1, keepdims=True)*up
    lateral /= np.linalg.norm(lateral, axis=1, keepdims=True)
    forward = np.cross(up, lateral)
    head = j[:, idx['head']]-j[:, idx['neck']]
    h_up = np.sum(head*up, 1)
    return np.degrees(np.stack([np.arctan2(np.sum(head*forward, 1), h_up), np.arctan2(np.sum(head*lateral, 1), h_up)], 1))


def posture_index(names):
    """Indices of the joints of the source neck and shoulder inputs (SMPL names; SOMA/cskel clavicles are *Shoulder)."""
    n = [JOINT_ALIAS[x] if x in JOINT_ALIAS and JOINT_ALIAS[x] not in names else x for x in names]
    need = ('pelvis', 'neck', 'head', 'right_hip', 'left_hip', 'left_collar', 'left_shoulder', 'right_collar', 'right_shoulder')
    return {k: n.index(k) for k in need} if set(need) <= set(n) else None


def shoulder_elevation(j, idx):
    """Clavicle-to-glenohumeral line above the plane normal to pelvis-to-neck (degrees), columns left, right."""
    up = j[:, idx['neck']]-j[:, idx['pelvis']]; up = up/np.linalg.norm(up, axis=-1, keepdims=True)
    out = []
    for s in ('left', 'right'):
        v = j[:, idx[f'{s}_shoulder']]-j[:, idx[f'{s}_collar']]
        out.append(np.degrees(np.arcsin(np.clip(np.sum(v*up, -1)/np.linalg.norm(v, axis=-1), -1, 1))))
    return np.stack(out, 1)


def rest_reading(joints, names):
    """Neck [flexion, side bend] and shoulder elevation [left, right] of one pose."""
    idx = posture_index(names)
    return neck_angles(joints[None], idx)[0], shoulder_elevation(joints[None], idx)[0]


def fk_rest(run):
    """Rest pose of an FK skeleton rebuilt from its global rotations (bone offsets in the parent frame)."""
    z = np.load(Path(run)/'raw/native_prediction.npz')
    j, R = z['joints'][:, :22].astype(float), z['global_rotations'][:, :22].astype(float)
    rest = np.zeros((22, 3))
    for k in range(1, 22):
        rest[k] = rest[SMPL22_PARENTS[k]]+R[0, SMPL22_PARENTS[k]].T @ (j[0, k]-j[0, SMPL22_PARENTS[k]])
    return rest


def neutral(method, source_dir=None):
    """('measured', neck zero [flex, side], shoulder zero [l, r]) | ('excluded', None, None) | ('unknown', None, None)."""
    if method in NEUTRAL['excluded']:
        return 'excluded', None, None
    own = KIT['ergo']['neutral_calibrated'].get(method)
    if own is not None:
        return 'measured', np.array(own['neck_deg']), np.array(own['shoulder_elevation_deg'])
    skeleton = NEUTRAL['methods'].get(method)
    if skeleton is None:
        return 'unknown', None, None
    off = NEUTRAL['skeletons']['smplx_neutral' if skeleton == 'per_run_fk' else skeleton]
    neck0, shoulder0 = np.array([off['neck_flexion_deg'], off['neck_side_bend_deg']]), np.array(off['shoulder_elevation_deg'])
    if skeleton == 'per_run_fk' and source_dir is not None:
        try:
            neck0, shoulder0 = rest_reading(fk_rest(source_dir), SMPL22_NAMES)
        except (OSError, KeyError):
            pass
    return 'measured', neck0, shoulder0


def source_posture(source, times, zero):
    """source: (joints OpenSim Y-up, names, fps); zero: neutral(). Returns (mode, neck (N, 2), shoulder (N, 2))."""
    mode, neck0, shoulder0 = zero
    if mode != 'measured' or source is None:
        return ('unknown' if mode == 'measured' else mode), None, None
    joints, names, fps = source
    idx = posture_index(names)
    if idx is None:
        return 'unknown', None, None
    frames = np.clip(np.round(np.asarray(times)*fps).astype(int), 0, len(joints)-1)
    return 'measured', neck_angles(joints[frames], idx)-neck0, shoulder_elevation(joints[frames], idx)-shoulder0


def side_inputs(row, col, side, trunk, legs_both, knee, known):
    s = side
    inputs = {'trunk_flexion_deg': float(trunk[0]), 'trunk_side_bent': int(abs(trunk[1]) > CONV['twist_and_side_bend_threshold_deg']),
              'trunk_twisted': int(abs(row[col['lumbar_rotation']]) > CONV['twist_and_side_bend_threshold_deg']),
              'upper_arm_flexion_deg': float(row[col[f'arm_flex_{s}']]),
              'arm_abducted_or_rotated': int(-row[col[f'arm_add_{s}']] > CONV['abduction_threshold_deg']),
              'arm_supported': 0, 'elbow_flexion_deg': float(np.clip(row[col[f'elbow_flex_{s}']], 0, 180)),
              'seated': 0, 'trunk_supported_seated': 0, 'static_over_minute': 0, 'knee_flexion_deg': float(knee)}
    if legs_both:
        inputs['leg_support_score'] = 1; inputs['legs_score'] = 1
    inputs.update(known)
    return inputs


def r3_inputs(base, neck, shoulder, locomoting, flight, both, mode, repeated=0):
    """Revision 3: wrist excluded and repeated-action frequency fixed at 0 (job-cycle property); neck and shoulder
    elevation from source; walking counts as bilateral support, a flight frame as an unstable base. mode 'excluded'
    fixes the source inputs at their minimum."""
    inp = {**base, 'wrist_flexion_deg': 0., 'wrist_deviated': 0, 'wrist_deviated_or_twisted': 0, 'wrist_twist_score': 1,
           'rapid_change_or_unstable': int(flight), 'repeated_small_range_over4_per_minute': repeated}
    inp['leg_support_score'] = inp['legs_score'] = 1 if (locomoting or flight or both) else 2
    if mode == 'excluded':
        inp.update(SOURCE_MIN)
    elif mode == 'measured':
        flex = float(neck[0])
        if -R3['neck_extension_tolerance_deg'] <= flex < 0:
            flex = 0.
        inp['neck_flexion_deg'] = flex
        inp['neck_side_bent'] = int(abs(neck[1]) > CONV['twist_and_side_bend_threshold_deg'])
        inp['shoulder_raised'] = int(shoulder > R3['shoulder_raised_threshold_deg'])
    return inp


def score_union(inputs, method):
    variants = [inputs]
    if inputs['upper_arm_flexion_deg'] >= CONV['upper_arm_bound_deg']:
        variants = [{**inputs, 'upper_arm_flexion_deg': 90.}, {**inputs, 'upper_arm_flexion_deg': 91.}]
    scores = set()
    for v in variants:
        scores |= set(ergonomics.score(v, method)['possible_scores'])
    return min(scores), max(scores)


def frame_class(bounds, unmeasurable, risk):
    lo, hi = bounds[:, 0], bounds[:, 1]
    return np.select([unmeasurable, lo >= risk, hi >= risk], [0, 1, 2], 3)


def score(model_path, times, labels, coords, masks, sel, source=None, zero=('unknown', None, None), free=True, unmeasurable=None):
    """Frame classes on every frame_stride-th scored frame. masks: height-only contact {side: {landmark: (T,) bool}}."""
    keep = np.flatnonzero(sel)[::int(CONV['frame_stride'])]
    dt = float(np.median(np.diff(times))); col = {l: i for i, l in enumerate(labels)}
    rows = coords[keep]
    trunk = trunk_angles(model_path, labels, rows)
    contact = {s: (masks[s]['heel'] | masks[s]['toe'])[keep] for s in SIDES}
    speed = np.hypot(np.gradient(coords[:, col['pelvis_tx']], dt), np.gradient(coords[:, col['pelvis_tz']], dt))[keep]
    locomoting = speed >= R3['locomotion_speed_m_s']; flight = ~(contact['r'] | contact['l'])
    known = {'load_score': 0, 'coupling_score': 0, 'load_score_a': 0, 'load_score_b': 0} if free else {}
    mode, neck, shoulder = source_posture(source, times[keep], zero)
    bounds = {m: np.zeros((len(keep), 2), int) for m in RISK}
    for i, row in enumerate(rows):
        both = bool(contact['r'][i] and contact['l'][i])
        loaded = [s for s in SIDES if contact[s][i]] or list(SIDES)
        knee = max(row[col[f'knee_angle_{s}']] for s in loaded)
        for m in RISK:
            lo = hi = None
            for s in SIDES:
                inp = side_inputs(row, col, s, trunk[i], both, knee, known)
                src = (None if neck is None else neck[i], None if shoulder is None else shoulder[i, 0 if s == 'l' else 1])
                a, b = score_union(r3_inputs(inp, *src, locomoting[i], flight[i], both, mode), m)
                lo, hi = (a, b) if lo is None else (max(lo, a), max(hi, b))
            bounds[m][i] = lo, hi
    gate = np.zeros(len(keep), bool) if unmeasurable is None else unmeasurable[keep]
    out = {'times': times[keep], 'index': keep, 'trunk_flexion_deg': trunk[:, 0], 'trunk_side_deg': trunk[:, 1],
           'locomoting': locomoting, 'flight': flight, 'source_mode': mode}
    for m, risk in RISK.items():
        out[f'{m.lower()}_bounds'] = bounds[m]; out[f'{m.lower()}_class'] = frame_class(bounds[m], gate, risk)
    return out
