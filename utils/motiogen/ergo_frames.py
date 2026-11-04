"""Frame-level RULA/REBA from E-pipeline outputs (config/ergo_rula_reba_20261105.json).

Inputs come from the run work directory of the E criterion: the scaled Rajagopal model, the 100 Hz
filtered IK coordinates and the residual force / contact masks on the same time base. Inputs the
marker set cannot observe stay unknown, so ergonomics.score returns the admissible score range.
"""
import json
import math
from pathlib import Path

import numpy as np

import biomech
import biomech_sources
import ergonomics

ROOT = Path(__file__).resolve().parents[2]
SPEC = json.loads((ROOT/'config/ergo_rula_reba_20261105.json').read_text())
CONV = SPEC['conventions_not_derived']
SIDES = {'r': 'right', 'l': 'left'}
CLASSES = ('infeasible', 'certain_risk', 'indeterminate', 'certain_acceptable')


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


def source_neck(source, times):
    """Neck flexion from source joints: neck-to-head against pelvis-to-neck, sagittal plane of the pelvis heading."""
    try:
        joints, names, fps = biomech_sources.load(source)
    except Exception:
        return None
    need = ('pelvis', 'neck', 'head', 'right_hip', 'left_hip')
    if not set(need) <= set(names):
        return None
    idx = {n: names.index(n) for n in need}
    frames = np.clip(np.round(np.asarray(times)*fps).astype(int), 0, len(joints)-1)
    j = joints[frames]
    out = np.zeros(len(frames))
    for i in range(len(frames)):
        up = j[i, idx['neck']]-j[i, idx['pelvis']]; up /= np.linalg.norm(up)
        lateral = j[i, idx['right_hip']]-j[i, idx['left_hip']]; lateral -= (lateral @ up)*up; lateral /= np.linalg.norm(lateral)
        forward = np.cross(up, lateral)
        head = j[i, idx['head']]-j[i, idx['neck']]
        out[i] = math.degrees(math.atan2(head @ forward, head @ up))
    return out


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


def score_run(work, source, object_free_task, mass=75.):
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
    known = {'load_score': 0, 'coupling_score': 0, 'load_score_a': 0, 'load_score_b': 0} if object_free_task else {}
    neck = source_neck(source, times[keep])
    out = {m: np.zeros((len(keep), 2), int) for m in ('REBA', 'RULA')}
    neck_out = {m: np.zeros((len(keep), 2), int) for m in ('REBA', 'RULA')} if neck is not None else None
    for i, row in enumerate(rows):
        legs_both = bool(contact['r'][i] and contact['l'][i])
        loaded = [s for s in SIDES if contact[s][i]] or list(SIDES)
        knee = max(row[col[f'knee_angle_{s}']] for s in loaded)
        for m in out:
            best = None; best_neck = None
            for s in SIDES:
                inp = side_inputs(row, col, s, trunk[i], legs_both, knee, known)
                lo, hi = score_union(inp, m)
                best = (lo, hi) if best is None else (worse(best[0], lo), worse(best[1], hi))
                if neck_out is not None:
                    lo2, hi2 = score_union({**inp, 'neck_flexion_deg': float(neck[i])}, m)
                    best_neck = (lo2, hi2) if best_neck is None else (worse(best_neck[0], lo2), worse(best_neck[1], hi2))
            out[m][i] = best
            if neck_out is not None:
                neck_out[m][i] = best_neck
    feasible = residual <= .05
    np.savez_compressed(work/'ergo_frames.npz', times=times[keep], residual_bw=residual, feasible=feasible,
                        trunk_flexion_deg=trunk[:, 0], trunk_side_deg=trunk[:, 1],
                        **({'neck_source_deg': neck} if neck is not None else {}),
                        **{f'{m.lower()}_bounds': out[m] for m in out},
                        **({f'{m.lower()}_bounds_neck': neck_out[m] for m in neck_out} if neck_out else {}))
    return summarize(out, neck_out, feasible, trunk)


def classes(bounds, feasible, risk):
    lo, hi = bounds[:, 0], bounds[:, 1]
    return {'infeasible': ~feasible, 'certain_risk': feasible & (lo >= risk),
            'indeterminate': feasible & (lo < risk) & (hi >= risk), 'certain_acceptable': feasible & (hi < risk)}


def summarize(out, neck_out, feasible, trunk):
    res = {'frames': int(len(feasible)), 'feasible_share': float(feasible.mean()),
           'trunk_flexion_deg_median': float(np.median(trunk[:, 0]))}
    for m, risk in (('REBA', CONV['risk_threshold_REBA']), ('RULA', CONV['risk_threshold_RULA'])):
        b = out[m]; c = classes(b, feasible, risk); key = m.lower()
        res[key] = {**{f'{k}_share': float(v.mean()) for k, v in c.items()},
                    'acceptable_lower': float((feasible & (b[:, 0] < risk)).mean()),
                    'conditional_certain_risk': float(c['certain_risk'][feasible].mean()) if feasible.any() else None,
                    'ungated_lower_acceptable': float((b[:, 0] < risk).mean()),
                    'lower_median_feasible': float(np.median(b[feasible, 0])) if feasible.any() else None,
                    'upper_median_feasible': float(np.median(b[feasible, 1])) if feasible.any() else None}
        if neck_out is not None:
            nb = neck_out[m]
            res[key]['neck_source_acceptable_lower'] = float((feasible & (nb[:, 0] < risk)).mean())
    return res
