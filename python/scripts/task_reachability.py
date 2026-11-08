#!/usr/bin/env python
"""Task reachability premise check (design notes section 14 item 2; inference registration).

Nominal body: the unscaled generic Rajagopal 2016 model (outputs/assets/rajagopal2016_generic). No method output is read.
For every reach target the wrist (hand body origin) is brought as close as possible to the target with both feet kept
at their default standing pose and every other coordinate inside the model's own range (multi-start SLSQP). A target is
reachable without stepping when the residual is below TOL_M.

  variants: flat (calcn bodies fixed: no heel rise) and heel_rise (toes bodies fixed, mtp unlocked; reported alongside)

Computed once: develop_logs/test_runs/20261109_task_reachability/reachability.json. Run with the OpenSim python.
  --overhead-limit      overhead_limit.json (Rajagopal and the HumanML3D reference body)
  --origin-shift M      flat variant with every target moved M forward (the methods' pelvis joint is not the hip-centre
                        midpoint): reachability_origin_shift_<M>.json
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import opensim as osim
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils'))
from motiogen.config import data_root, load_config  # noqa: E402

MODEL = data_root(load_config())/'outputs/assets/rajagopal2016_generic/Rajagopal2016.osim'
OUT = ROOT/'develop_logs/test_runs/20261109_task_reachability'
TOL_M = .01
FOOT_TOL_M = .001
RISK_HEIGHTS_M = {'floor': .15, 'knee': .5, 'overhead': 1.95}
REACH_RATIO = .8
LEGACY_WORLD = {'low': .9, 'mid': 1.2, 'high': 1.5}


def coords(model):
    cs = model.getCoordinateSet()
    return {cs.get(i).getName(): cs.get(i) for i in range(cs.getSize())}


def body_frame(p):
    """model ground (x forward, y up, z right) -> body frame [forward, right, up]."""
    return [float(p[0]), float(p[2]), float(p[1])]


def vec(v):
    return np.array([v.get(0), v.get(1), v.get(2)])


def rot(body, state):
    R = body.getTransformInGround(state).R()
    return np.array([[R.get(i, j) for j in range(3)] for i in range(3)])


def origin(model, state):
    """task origin: the initial pelvis joint of SMPL-family skeletons, taken on the nominal body as the midpoint of
    the two hip joint centres (convention; HumanML3D standing frames put the pelvis joint 0.012 m ahead of it)."""
    bs = model.getBodySet(); model.realizePosition(state)
    return (vec(bs.get('femur_r').getPositionInGround(state))+vec(bs.get('femur_l').getPositionInGround(state)))/2


def nominal(model, state):
    bs = model.getBodySet()
    model.realizePosition(state)
    o = origin(model, state)
    out = {}
    for side, s in (('right', 'r'), ('left', 'l')):
        sh = vec(bs.get(f'humerus_{s}').getPositionInGround(state))
        wr = vec(bs.get(f'hand_{s}').getPositionInGround(state))
        el = vec(bs.get(f'radius_{s}').getPositionInGround(state))
        out[side] = {'shoulder_body_m': body_frame(sh-o*[1, 0, 1]), 'arm_length_m': float(np.linalg.norm(wr-sh)),
                     'upper_arm_m': float(np.linalg.norm(el-sh)), 'forearm_m': float(np.linalg.norm(wr-el))}
    knee = vec(bs.get('tibia_r').getPositionInGround(state))
    return {'pelvis_height_m': float(vec(bs.get('pelvis').getPositionInGround(state))[1]), 'hip_centre_height_m': float(o[1]),
            'knee_joint_height_m': float(knee[1]), 'sides': out}


def tasks(nom, heights=RISK_HEIGHTS_M):
    """24 reach targets in the body frame [forward, right(+), up], metres, origin: initial pelvis on the floor (origin())."""
    rows = []
    for side in ('right', 'left'):
        sign = 1. if side == 'right' else -1.
        for height in ('low', 'mid', 'high'):
            for fwd in (.6, .4):
                # 20261024 world target (x, y, z) = (sign*0.35, fwd, height): body faces -Y, right is -X
                rows.append({'tier': 'legacy_behind', 'side': side, 'height': height, 'legacy_y_m': fwd,
                             'target_body_m': [-fwd, -sign*.35, LEGACY_WORLD[height]]})
    for side in ('right', 'left'):
        sign = 1. if side == 'right' else -1.
        sh = nom['sides'][side]['shoulder_body_m']; reach = REACH_RATIO*nom['sides'][side]['arm_length_m']
        for zone, z in heights.items():
            rows.append({'tier': 'risk', 'side': side, 'zone': zone, 'direction': 'front',
                         'target_body_m': [sh[0]+reach, sh[1], z]})
            rows.append({'tier': 'risk', 'side': side, 'zone': zone, 'direction': 'side90',
                         'target_body_m': [sh[0], sh[1]+sign*reach, z]})
    for i, r in enumerate(rows):
        r['target_body_m'] = [round(v, 4) for v in r['target_body_m']]
        r['id'] = i
    return rows


class Reach:
    def __init__(self, model, state, side, variant):
        self.m, self.s, self.side = model, state, side
        s = 'r' if side == 'right' else 'l'
        self.c = coords(model)
        self.bs = model.getBodySet()
        names = ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz']
        for k in ('r', 'l'):
            names += [f'hip_flexion_{k}', f'hip_adduction_{k}', f'hip_rotation_{k}', f'knee_angle_{k}', f'ankle_angle_{k}']
            if variant == 'heel_rise':
                names.append(f'mtp_angle_{k}')
        names += ['lumbar_extension', 'lumbar_bending', 'lumbar_rotation',
                  f'arm_flex_{s}', f'arm_add_{s}', f'arm_rot_{s}', f'elbow_flex_{s}', f'pro_sup_{s}']
        self.names = names
        if variant == 'heel_rise':
            for k in ('r', 'l'):
                self.c[f'mtp_angle_{k}'].setLocked(state, False)
        self.lo = np.array([self.c[n].getRangeMin() for n in names])
        self.hi = np.array([self.c[n].getRangeMax() for n in names])
        self.q0 = np.array([self.c[n].getValue(state) for n in names])
        self.hand = self.bs.get(f'hand_{s}')
        self.feet = [self.bs.get(f'{"toes" if variant == "heel_rise" else "calcn"}_{k}') for k in ('r', 'l')]
        model.realizePosition(state)
        self.feet0 = [(vec(b.getPositionInGround(state)), rot(b, state)) for b in self.feet]
        self.origin0 = origin(model, state)

    def set(self, q):
        for n, v in zip(self.names, q):
            self.c[n].setValue(self.s, float(v), False)
        self.m.realizePosition(self.s)

    def wrist(self, q):
        self.set(q)
        return vec(self.hand.getPositionInGround(self.s))

    def feet_err(self, q):
        self.set(q)
        e = []
        for b, (p0, R0) in zip(self.feet, self.feet0):
            e += list(vec(b.getPositionInGround(self.s))-p0)
            e += list((rot(b, self.s)-R0)[[0, 1, 2], [1, 2, 0]]*.1)
        return np.array(e)

    def solve(self, target_ground, starts):
        best = None
        for q_start in starts:
            res = minimize(lambda q: float(np.sum((self.wrist(q)-target_ground)**2)), q_start, method='SLSQP',
                           bounds=list(zip(self.lo, self.hi)), constraints=[{'type': 'eq', 'fun': self.feet_err}],
                           options={'maxiter': 400, 'ftol': 1e-10})
            d = float(np.linalg.norm(self.wrist(res.x)-target_ground)); fe = float(np.abs(self.feet_err(res.x)[[0, 1, 2, 6, 7, 8]]).max())
            if fe <= FOOT_TOL_M and (best is None or d < best[0]):
                best = (d, fe, res.x.copy())
        return best

    def starts(self, target_ground):
        q = self.q0.copy(); out = [q.copy()]
        idx = {n: i for i, n in enumerate(self.names)}
        s = 'r' if self.side == 'right' else 'l'
        low = target_ground[1] < .9
        for hip, knee, lum, arm_flex, arm_add, dy in ((90, 30, -40, 60, 0, -.25), (60, 100, -20, 60, 0, -.45),
                                                     (100, 110, -60, 40, -20, -.55), (0, 0, 0, 90, -100, 0),
                                                     (20, 10, 20, 0, -110, -.03), (0, 0, 0, -60, -10, 0)):
            if low != (dy < -.1):
                continue
            q = self.q0.copy()
            for k in ('r', 'l'):
                q[idx[f'hip_flexion_{k}']] = math.radians(hip); q[idx[f'knee_angle_{k}']] = math.radians(knee)
                q[idx[f'ankle_angle_{k}']] = math.radians(min(30., knee*.3))
            q[idx['pelvis_tilt']] = -math.radians(hip)*.5
            q[idx['lumbar_extension']] = math.radians(lum)
            q[idx['pelvis_ty']] = self.q0[idx['pelvis_ty']]+dy
            q[idx[f'arm_flex_{s}']] = math.radians(arm_flex); q[idx[f'arm_add_{s}']] = math.radians(arm_add)
            out.append(np.clip(q, self.lo, self.hi))
        return out


HUMANML_BODY = OUT/'humanml_reference_body.json'
OVERHEAD_MARGIN_M = .05


def body_points(model, state):
    """default-pose landmarks in ground: hip / knee / ankle centres, shoulder, elbow (ulna origin), wrist (hand origin)."""
    bs = model.getBodySet(); model.realizePosition(state)
    p = {}
    for s in 'rl':
        for key, body in (('hip', 'femur'), ('knee', 'tibia'), ('ankle', 'talus'), ('shoulder', 'humerus'), ('elbow', 'ulna'), ('wrist', 'hand')):
            p[f'{key}_{s}'] = vec(bs.get(f'{body}_{s}').getPositionInGround(state))
    return p


def scaled(factors, pelvis_ty=None):
    model = osim.Model(str(MODEL)); state = model.initSystem()
    ss = osim.ScaleSet()
    for body, f in factors.items():
        sc = osim.Scale(); sc.setSegmentName(body); sc.setScaleFactors(osim.Vec3(*f)); sc.setApply(True)
        ss.cloneAndAppend(sc)
    model.scale(state, ss, True)
    state = model.initSystem()
    if pelvis_ty is not None:
        coords(model)['pelvis_ty'].setDefaultValue(pelvis_ty)
        state = model.initSystem()
    return model, state


def humanml_body():
    """Rajagopal scaled segment-wise to the HumanML3D reference body; Rajagopal joint ranges kept (convention).

    legs: thigh / shank to the left-right mean bone lengths, foot (talus, calcn, toes) to the standing ankle height,
    torso: vertical so the standing shoulder height matches, lateral so the shoulder half-width matches,
    arms per side: humerus to the upper arm, ulna / radius / hand to the forearm. Standing: pelvis_ty puts the ankle
    centres at the HumanML3D standing ankle height (floor = dataset floor)."""
    ref = json.loads(HUMANML_BODY.read_text())
    b, st = ref['bone_lengths_m'], ref['standing']['joints']
    want = {'thigh': (b['thigh_l']+b['thigh_r'])/2, 'shank': (b['shank_l']+b['shank_r'])/2,
            'ankle_h': (st['ankle_l']['median_body_m'][2]+st['ankle_r']['median_body_m'][2])/2,
            'shoulder_h': (st['shoulder_l']['median_body_m'][2]+st['shoulder_r']['median_body_m'][2])/2,
            'shoulder_half_width': (abs(st['shoulder_l']['median_body_m'][1])+abs(st['shoulder_r']['median_body_m'][1]))/2}
    model = osim.Model(str(MODEL)); state = model.initSystem(); p = body_points(model, state)
    have = {'thigh': np.linalg.norm(p['hip_r']-p['knee_r']), 'shank': np.linalg.norm(p['knee_r']-p['ankle_r']), 'ankle_h': p['ankle_r'][1]}
    torso = vec(model.getJointSet().get('acromial_r').getParentFrame().findTransformInBaseFrame().p())
    factors = {}
    for s in 'rl':
        factors[f'femur_{s}'] = [want['thigh']/have['thigh']]*3
        factors[f'tibia_{s}'] = [want['shank']/have['shank']]*3
        for body in ('talus', 'calcn', 'toes'):
            factors[f'{body}_{s}'] = [want['ankle_h']/have['ankle_h']]*3
        up = b[f'upper_arm_{s}']/np.linalg.norm(p[f'shoulder_{s}']-p[f'elbow_{s}'])
        fo = b[f'forearm_{s}']/np.linalg.norm(p[f'elbow_{s}']-p[f'wrist_{s}'])
        factors[f'humerus_{s}'] = [up]*3
        for body in ('ulna', 'radius', 'hand'):
            factors[f'{body}_{s}'] = [fo]*3
    factors['torso'] = [1., 1., want['shoulder_half_width']/abs(torso[2])]
    m, st0 = scaled(factors); q = body_points(m, st0)
    ty = coords(m)['pelvis_ty'].getValue(st0)+want['ankle_h']-(q['ankle_r'][1]+q['ankle_l'][1])/2
    m, st0 = scaled(factors, ty); q = body_points(m, st0)
    factors['torso'][1] = (torso[1]+want['shoulder_h']-(q['shoulder_r'][1]+q['shoulder_l'][1])/2)/torso[1]
    m, st0 = scaled(factors, ty); q = body_points(m, st0)
    got = {'thigh_r': np.linalg.norm(q['hip_r']-q['knee_r']), 'shank_r': np.linalg.norm(q['knee_r']-q['ankle_r']),
           'ankle_h': (q['ankle_r'][1]+q['ankle_l'][1])/2, 'shoulder_h': (q['shoulder_r'][1]+q['shoulder_l'][1])/2,
           'shoulder_half_width': (abs(q['shoulder_r'][2])+abs(q['shoulder_l'][2]))/2}
    for s in 'rl':
        got[f'upper_arm_{s}'] = np.linalg.norm(q[f'shoulder_{s}']-q[f'elbow_{s}'])
        got[f'forearm_{s}'] = np.linalg.norm(q[f'elbow_{s}']-q[f'wrist_{s}'])
    target = {**{k: want[k] for k in ('ankle_h', 'shoulder_h', 'shoulder_half_width')}, 'thigh_r': want['thigh'], 'shank_r': want['shank'],
              **{f'{k}_{s}': b[f'{k}_{s}'] for k in ('upper_arm', 'forearm') for s in 'rl'}}
    check = {k: {'target_m': round(float(target[k]), 4), 'scaled_model_m': round(float(got[k]), 4)} for k in target}
    return factors, ty, check


def build(body, factors=None, ty=None):
    if body == 'rajagopal':
        model = osim.Model(str(MODEL)); return model, model.initSystem()
    return scaled(factors, ty)


def elevation(model, state, side, target, start):
    """smallest humeral elevation over all poses that put the wrist on the target with the feet fixed (flat):
    world (humerus long axis vs downward vertical) and humerothoracic (vs the torso's downward axis), degrees."""
    s = 'r' if side == 'right' else 'l'
    r = Reach(model, state, side, 'flat')
    hum, uln, tor = r.bs.get(f'humerus_{s}'), r.bs.get(f'ulna_{s}'), r.bs.get('torso')

    def angles(q):
        r.set(q)
        u = vec(uln.getPositionInGround(r.s))-vec(hum.getPositionInGround(r.s)); u /= np.linalg.norm(u)
        return (math.degrees(math.acos(np.clip(-u[1], -1, 1))), math.degrees(math.acos(np.clip(-u@rot(tor, r.s)[:, 1], -1, 1))))

    out = {}
    for k, name in enumerate(('world', 'humerothoracic')):
        best = None
        for q0 in [start]+r.starts(target):
            res = minimize(lambda q: angles(q)[k], q0, method='SLSQP', bounds=list(zip(r.lo, r.hi)),
                           constraints=[{'type': 'eq', 'fun': r.feet_err}, {'type': 'eq', 'fun': lambda q: r.wrist(q)-target}],
                           options={'maxiter': 400, 'ftol': 1e-10})
            d = float(np.linalg.norm(r.wrist(res.x)-target)); fe = float(np.abs(r.feet_err(res.x)[[0, 1, 2, 6, 7, 8]]).max())
            if d < 1e-3 and fe <= FOOT_TOL_M and (best is None or angles(res.x)[k] < best):
                best = angles(res.x)[k]
        out[f'min_{name}_elevation_deg'] = None if best is None else round(best, 1)
    return out


def overhead_limit():
    """highest reachable wrist height at each body's own registered horizontal offsets (front, side90), by bisection,
    for the nominal Rajagopal body and the HumanML3D reference body; overhead height = min over the flat limits - margin."""
    t0 = time.time()
    factors, ty, check = humanml_body()
    bodies, limits = {}, {}
    for body in ('rajagopal', 'humanml'):
        model, state = build(body, factors, ty)
        nom = nominal(model, state)
        bodies[body] = {'nominal': nom}
        for variant in ('flat', 'heel_rise'):
            for side in ('right', 'left'):
                model, state = build(body, factors, ty)
                r = Reach(model, state, side, variant)
                sign = 1. if side == 'right' else -1.
                sh = nom['sides'][side]['shoulder_body_m']; reach = REACH_RATIO*nom['sides'][side]['arm_length_m']
                for direction, (f, rt) in (('front', (sh[0]+reach, sh[1])), ('side90', (sh[0], sh[1]+sign*reach))):
                    lo, hi = 1.5, 1.95
                    for _ in range(10):
                        mid = (lo+hi)/2
                        target = np.array([r.origin0[0]+f, mid, r.origin0[2]+rt])
                        best = r.solve(target, r.starts(target))
                        if best is not None and best[0] < TOL_M:
                            lo = mid
                        else:
                            hi = mid
                    limits[f'{body}/{variant}/{side}/{direction}'] = {'max_reachable_height_m': round(lo, 4), 'bracket_m': [round(lo, 4), round(hi, 4)],
                                                                      'target_body_m': [round(f, 4), round(rt, 4)]}
                    print(body, variant, side, direction, limits[f'{body}/{variant}/{side}/{direction}'], flush=True)
    flat = {k: v['max_reachable_height_m'] for k, v in limits.items() if '/flat/' in k}
    height = math.floor((min(flat.values())-OVERHEAD_MARGIN_M)*1000)/1000
    checks = {}
    for body in ('rajagopal', 'humanml'):
        for side in ('right', 'left'):
            model, state = build(body, factors, ty)
            r = Reach(model, state, side, 'flat'); nom = bodies[body]['nominal']
            sign = 1. if side == 'right' else -1.
            sh = nom['sides'][side]['shoulder_body_m']; reach = REACH_RATIO*nom['sides'][side]['arm_length_m']
            for direction, (f, rt) in (('front', (sh[0]+reach, sh[1])), ('side90', (sh[0], sh[1]+sign*reach))):
                target = np.array([r.origin0[0]+f, height, r.origin0[2]+rt])
                best = r.solve(target, r.starts(target))
                row = {'residual_m': None if best is None else round(best[0], 5)}
                if body == 'rajagopal' and best is not None:
                    row.update(elevation(*build(body, factors, ty), side, target, best[2]))
                checks[f'{body}/{side}/{direction}'] = row
                print('at', height, body, side, direction, row, flush=True)
    out = {'created': time.strftime('%Y-%m-%d %H:%M'), 'tol_m': TOL_M, 'reach_ratio': REACH_RATIO,
           'standard': 'feet fixed at their default standing pose (flat: calcn; heel_rise: toes, reported only), no stepping; each body uses its own shoulder position and 0.8 x its own arm length',
           'humanml_body': {'source': str(HUMANML_BODY.relative_to(ROOT)), 'method_convention_not_derived': humanml_body.__doc__.split('\n\n')[0].strip(),
                            'scale_factors': {k: [round(v, 4) for v in f] for k, f in factors.items()}, 'pelvis_ty_m': round(ty, 4), 'match': check},
           'bodies': bodies, 'limits': limits,
           'overhead_height': {'rule': f'min over the flat limits (rajagopal and humanml, front and side90, both hands) - {OVERHEAD_MARGIN_M} m (convention, not derived), floored to mm',
                               'min_flat_limit_m': min(flat.values()), 'argmin': min(flat, key=flat.get), 'height_m': height},
           'checks_at_height': checks, 'seconds': round(time.time()-t0, 1)}
    (OUT/'overhead_limit.json').write_text(json.dumps(out, indent=1))
    print('height', height, 'seconds', out['seconds'])


def main():
    if '--overhead-limit' in sys.argv:
        return overhead_limit()
    shift = float(sys.argv[sys.argv.index('--origin-shift')+1]) if '--origin-shift' in sys.argv else 0.
    t0 = time.time()
    limit = json.loads((OUT/'overhead_limit.json').read_text())['overhead_height']
    heights = {**RISK_HEIGHTS_M, 'overhead': limit['height_m']}
    model = osim.Model(str(MODEL)); state = model.initSystem()
    nom = nominal(model, state)
    rows = tasks(nom, heights)
    o = origin(model, state)
    results = []
    for variant in ('flat', 'heel_rise') if not shift else ('flat',):
        for side in ('right', 'left'):
            model = osim.Model(str(MODEL)); state = model.initSystem()
            r = Reach(model, state, side, variant)
            for row in [x for x in rows if x['side'] == side]:
                f, rt, up = row['target_body_m']
                target = np.array([o[0]+f+shift, up, o[2]+rt])
                best = r.solve(target, r.starts(target))
                if best is None:
                    results.append({'id': row['id'], 'variant': variant, 'residual_m': None, 'reachable': None,
                                    'note': 'no start met the foot tolerance'})
                    continue
                d, fe, q = best
                pose = {n: round(float(v), 3) if n in ('pelvis_tx', 'pelvis_ty', 'pelvis_tz') else round(math.degrees(v), 1)
                        for n, v in zip(r.names, q)}
                at_limit = [n for n, v, lo, hi in zip(r.names, q, r.lo, r.hi) if 'pelvis' not in n and (v-lo < 1e-3 or hi-v < 1e-3)]
                results.append({'id': row['id'], 'variant': variant, 'residual_m': round(d, 4), 'reachable': d < TOL_M,
                                'foot_error_m': round(fe, 5), 'coordinates_at_limit': at_limit, 'pose': pose})
    for row in rows:
        for x in results:
            if x['id'] == row['id']:
                row[x['variant']] = {k: v for k, v in x.items() if k not in ('id', 'variant')}
    out = {'created': time.strftime('%Y-%m-%d %H:%M'), 'model': str(MODEL.relative_to(data_root(load_config()))),
           'model_sha256': __import__('hashlib').sha256(MODEL.read_bytes()).hexdigest(),
           'nominal_body': nom, 'frame': 'body frame [forward, right, up] in metres; origin: initial pelvis (nominal body: hip joint centre midpoint, see origin()) projected on the floor (model ground y = 0); forward is the model +X, right +Z',
           'criterion': {'tol_m': TOL_M, 'foot_tol_m': FOOT_TOL_M, 'flat': 'calcn_r and calcn_l keep their default pose (position and orientation)',
                         'heel_rise': 'toes_r and toes_l keep their default pose, mtp unlocked (range ±30°), heels may lift'},
           'conventions_not_derived': {'risk_heights_m': heights, 'reach_ratio': REACH_RATIO, 'tol_m': TOL_M,
                                       'overhead': f"design value {RISK_HEIGHTS_M['overhead']} m is not reachable without stepping; replaced by overhead_limit.json: {limit['rule']}"},
           'seconds': round(time.time()-t0, 1), 'tasks': rows}
    if shift:
        out['origin_shift_m'] = {'forward': shift, 'meaning': 'targets moved forward by this amount relative to the nominal origin (sensitivity check; flat only)'}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/(f'reachability_origin_shift_{shift:+.3f}.json' if shift else 'reachability.json')).write_text(json.dumps(out, indent=1))
    for row in rows:
        print(row['id'], row['tier'], row['side'], row.get('zone', row.get('height')), row.get('direction', row.get('legacy_y_m')),
              row['target_body_m'], 'flat', (row.get('flat') or {}).get('residual_m'), 'heel', (row.get('heel_rise') or {}).get('residual_m'),
              (row.get('flat') or {}).get('coordinates_at_limit'))
    print('seconds', out['seconds'])


if __name__ == '__main__':
    main()
