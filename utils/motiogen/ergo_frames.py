"""Frame-level RULA/REBA from E-pipeline outputs (config/ergo_rula_reba_20261105.json).

Inputs come from the run work directory of the E criterion: the scaled Rajagopal model, the 100 Hz
filtered IK coordinates and the residual force / contact masks on the same time base. Inputs the
marker set cannot observe stay unknown, so ergonomics.score returns the admissible score range.
"""
import json
import math
from pathlib import Path

import numpy as np

import ergonomics

ROOT = Path(__file__).resolve().parents[2]
SPEC = json.loads((ROOT/'config/ergo_rula_reba_20261105.json').read_text())
CONV = SPEC['conventions_not_derived']
SIDES = {'r': 'right', 'l': 'left'}
CLASSES = ('infeasible', 'certain_risk', 'indeterminate', 'certain_acceptable')
R3 = next(r['rules'] for r in SPEC['revisions'] if r.get('id') == 'r3')
NEUTRAL_PATH = ROOT/'config/ergo_neutral_offsets_20261105.json'


def read_mot(path):
    lines = Path(path).read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().lower() == 'endheader')+1
    labels = lines[start].split('\t')
    data = np.array([[float(v) for v in l.split('\t')] for l in lines[start+1:] if l.strip()])
    return labels, data


def object_free(prompt):
    if prompt is None:
        return False
    text = prompt.lower()
    return not any(k in text for k in SPEC['object_keywords'])


def trunk_angles(model_path, labels, rows):
    """Trunk line (hip-joint midpoint to the registered neck-base marker) against ground vertical,
    in the pelvis sagittal and frontal planes (degrees)."""
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
    """Neck-to-head line against pelvis-to-neck: flexion (sagittal, forward positive) and side bend (degrees), shape (N, 2)."""
    up = j[:, idx['neck']]-j[:, idx['pelvis']]; up /= np.linalg.norm(up, axis=1, keepdims=True)
    lateral = j[:, idx['right_hip']]-j[:, idx['left_hip']]; lateral -= np.sum(lateral*up, 1, keepdims=True)*up
    lateral /= np.linalg.norm(lateral, axis=1, keepdims=True)
    forward = np.cross(up, lateral)
    head = j[:, idx['head']]-j[:, idx['neck']]
    h_up = np.sum(head*up, 1)
    return np.degrees(np.stack([np.arctan2(np.sum(head*forward, 1), h_up), np.arctan2(np.sum(head*lateral, 1), h_up)], 1))


SMPL22_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
SMPL22_NAMES = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3',
                'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow',
                'right_elbow', 'left_wrist', 'right_wrist']
JOINT_ALIAS = {'LeftArm': 'left_shoulder', 'RightArm': 'right_shoulder', 'Hips': 'pelvis', 'Neck1': 'neck', 'Neck': 'neck',
               'LeftShoulder': 'left_collar', 'RightShoulder': 'right_collar'}


def posture_index(names):
    """Indices of the joints used for the source neck and shoulder inputs (SMPL names; SOMA/cskel clavicles are *Shoulder)."""
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


def source_posture(source, times, method):
    """Revision 3 source inputs: (mode, neck angles minus the skeleton zero, shoulder elevation minus the skeleton zero),
    mode 'measured', 'excluded' (no rigid skeleton) or 'unknown'."""
    if not NEUTRAL_PATH.exists():
        return 'unknown', None, None
    neutral = json.loads(NEUTRAL_PATH.read_text())
    if method in neutral['excluded']:
        return 'excluded', None, None
    skeleton = neutral['methods'].get(method)
    if skeleton is None:
        return 'unknown', None, None
    import biomech_sources
    try:
        joints, names, fps = biomech_sources.load(source)
    except Exception:
        return 'unknown', None, None
    idx = posture_index(names)
    if idx is None:
        return 'unknown', None, None
    off = neutral['skeletons']['smplx_neutral' if skeleton == 'per_run_fk' else skeleton]
    neck0, shoulder0 = np.array([off['neck_flexion_deg'], off['neck_side_bend_deg']]), np.array(off['shoulder_elevation_deg'])
    if skeleton == 'per_run_fk':
        try:
            neck0, shoulder0 = rest_reading(fk_rest(source), SMPL22_NAMES)
        except Exception:
            pass
    frames = np.clip(np.round(np.asarray(times)*fps).astype(int), 0, len(joints)-1)
    return 'measured', neck_angles(joints[frames], idx)-neck0, shoulder_elevation(joints[frames], idx)-shoulder0


SOURCE_MIN = {'neck_flexion_deg': 10., 'neck_twisted': 0, 'neck_side_bent': 0, 'shoulder_raised': 0}


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
    """Revision 3: wrist excluded and repeated-action frequency fixed at 0 (job-cycle property; minimum for every method);
    neck and shoulder elevation from source; legs and activity from the REBA definitions (walking counts as bilateral
    support; a flight frame counts as an unstable base). mode 'excluded' fixes the source inputs at their minimum;
    repeated=1 gives the upper-bound sensitivity of the repeated-action convention."""
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


def worse(a, b):
    return max(a, b)


def score_run(work, source, object_free_task, mass=75., method=None):
    work = Path(work)
    labels, data = read_mot(work/'coordinates.mot')
    dyn = np.load(work/'dynamics.npz')
    times = data[:, 0]
    if len(dyn['times']) != len(times) or not np.allclose(dyn['times'], times, atol=1e-6):
        raise ValueError('coordinates.mot and dynamics.npz time bases differ')
    dt = float(np.median(np.diff(times)))
    keep = np.arange(int(.5/dt), len(times)-int(.5/dt), int(CONV['frame_stride']))
    col = {l: i for i, l in enumerate(labels)}
    rows = data[keep]
    trunk = trunk_angles(work/'model.osim', labels, rows)
    residual = np.linalg.norm(dyn['residual_force'][keep], axis=1)/(mass*9.80665)
    contact = {s: (dyn[f'contact_{s}_heel'][keep] > 0) | (dyn[f'contact_{s}_toe'][keep] > 0) for s in SIDES}
    speed = np.hypot(np.gradient(data[:, col['pelvis_tx']], dt), np.gradient(data[:, col['pelvis_tz']], dt))[keep]
    locomoting = speed >= R3['locomotion_speed_m_s']
    flight = ~(contact['r'] | contact['l'])
    known = {'load_score': 0, 'coupling_score': 0, 'load_score_a': 0, 'load_score_b': 0} if object_free_task else {}
    neck_mode, neck, shoulder = source_posture(source, times[keep], method)
    out = {m: np.zeros((len(keep), 2), int) for m in ('REBA', 'RULA')}
    r3 = {m: np.zeros((len(keep), 2), int) for m in ('REBA', 'RULA')}
    r3c = {m: np.zeros((len(keep), 2), int) for m in ('REBA', 'RULA')}
    r3r = {m: np.zeros((len(keep), 2), int) for m in ('REBA', 'RULA')}
    for i, row in enumerate(rows):
        legs_both = bool(contact['r'][i] and contact['l'][i])
        loaded = [s for s in SIDES if contact[s][i]] or list(SIDES)
        knee = max(row[col[f'knee_angle_{s}']] for s in loaded)
        for m in out:
            best = {}
            for s in SIDES:
                inp = side_inputs(row, col, s, trunk[i], legs_both, knee, known)
                src = (None if neck is None else neck[i], None if shoulder is None else shoulder[i, 0 if s == 'l' else 1])
                cand = {'reg': score_union(inp, m),
                        'r3': score_union(r3_inputs(inp, *src, locomoting[i], flight[i], legs_both, neck_mode), m),
                        'r3r': score_union(r3_inputs(inp, *src, locomoting[i], flight[i], legs_both, neck_mode, repeated=1), m),
                        'r3c': score_union(r3_inputs(inp, None, None, locomoting[i], flight[i], legs_both, 'excluded'), m)}
                best = cand if not best else {k: (worse(best[k][0], v[0]), worse(best[k][1], v[1])) for k, v in cand.items()}
            out[m][i], r3[m][i], r3c[m][i], r3r[m][i] = best['reg'], best['r3'], best['r3c'], best['r3r']
    feasible = residual <= .05
    np.savez_compressed(work/'ergo_frames.npz', times=times[keep], residual_bw=residual, feasible=feasible,
                        trunk_flexion_deg=trunk[:, 0], trunk_side_deg=trunk[:, 1], pelvis_speed_m_s=speed,
                        locomoting=locomoting, flight=flight,
                        **{f'{m.lower()}_bounds': out[m] for m in out}, **{f'{m.lower()}_bounds_r3': r3[m] for m in r3},
                        **{f'{m.lower()}_bounds_r3_common': r3c[m] for m in r3c},
                        **{f'{m.lower()}_bounds_r3_repeat1': r3r[m] for m in r3r},
                        **({} if neck is None else {'neck_source_deg': neck, 'shoulder_elevation_source_deg': shoulder}))
    res = summarize(out, feasible, trunk)
    for m, risk in (('REBA', CONV['risk_threshold_REBA']), ('RULA', CONV['risk_threshold_RULA'])):
        res[f'{m.lower()}_r3'] = level(r3[m], feasible, risk)
        res[f'{m.lower()}_r3_common'] = level(r3c[m], feasible, risk)
        res[f'{m.lower()}_r3_repeat1'] = level(r3r[m], feasible, risk)
    res.update(source_mode=neck_mode, shoulder_raised_share=None if shoulder is None else float((shoulder > R3['shoulder_raised_threshold_deg']).mean()), locomoting_share=float(locomoting.mean()), flight_share=float(flight.mean()))
    return res


def classes(bounds, feasible, risk):
    lo, hi = bounds[:, 0], bounds[:, 1]
    return {'infeasible': ~feasible, 'certain_risk': feasible & (lo >= risk),
            'indeterminate': feasible & (lo < risk) & (hi >= risk), 'certain_acceptable': feasible & (hi < risk)}


def level(b, feasible, risk):
    c = classes(b, feasible, risk)
    return {**{f'{k}_share': float(v.mean()) for k, v in c.items()},
            'acceptable_lower': float((feasible & (b[:, 0] < risk)).mean()),
            'conditional_certain_risk': float(c['certain_risk'][feasible].mean()) if feasible.any() else None,
            'ungated_lower_acceptable': float((b[:, 0] < risk).mean()),
            'ungated_certain_acceptable': float((b[:, 1] < risk).mean()),
            'lower_median_feasible': float(np.median(b[feasible, 0])) if feasible.any() else None,
            'upper_median_feasible': float(np.median(b[feasible, 1])) if feasible.any() else None}


def summarize(out, feasible, trunk):
    res = {'frames': int(len(feasible)), 'feasible_share': float(feasible.mean()),
           'trunk_flexion_deg_median': float(np.median(trunk[:, 0]))}
    for m, risk in (('REBA', CONV['risk_threshold_REBA']), ('RULA', CONV['risk_threshold_RULA'])):
        res[m.lower()] = level(out[m], feasible, risk)
    return res
