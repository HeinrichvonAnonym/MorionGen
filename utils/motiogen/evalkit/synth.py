"""Model-internal synthetic validation clips with computed truth (config/evalkit_synth_20261107.json). OpenSim python.

A clip is a truth motion of a Rajagopal model (100 Hz coordinates, contacts fixed by construction or by the force plate)
plus the generator-like input the tools see: sparse joints at 30 fps with landmark offsets and jitter, the floor height
and the mass. Constructed classes are keyframed in coordinate space and solved by IK with the planted foot stations
fixed; the demo walk is the Moco example3DWalking stride, tiled.
"""
import json
from pathlib import Path

import numpy as np
import opensim as osim

import biomech

ROOT = Path(__file__).resolve().parents[3]
SPEC = json.loads((ROOT/'config/evalkit_synth_20261107.json').read_text())
DEMO = Path(SPEC['demo_dir'])/'moco_example3DWalking'
FPS = SPEC['fps_truth']
SIDES = ('r', 'l')
STATIONS = ((0., 0., 0.), (.05, 0., 0.), (0., 0., .03))
TORSO = {'Spine': (0., .08, 0.), 'Spine3': (0., .28, 0.), 'neck': (0., .40, 0.), 'head': (.02, .55, 0.),
         'right_collar': (.01, .37, .07), 'left_collar': (.01, .37, -.07)}
LIMB = {'hip': 'femur', 'knee': 'tibia', 'ankle': 'talus', 'foot': 'toes', 'shoulder': 'humerus', 'elbow': 'ulna', 'wrist': 'hand'}
ACROMION_Y = .3715
osim.Logger.setLevelString('error')


def min_jerk(s):
    s = np.clip(s, 0, 1)
    return 10*s**3-15*s**4+6*s**5


def keyframed(times, keys):
    """keys: [(t, value)] -> minimum-jerk interpolation between consecutive keys, held outside."""
    out = np.full(len(times), float(keys[0][1]))
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        m = (times >= t0) & (times <= t1)
        out[m] = v0+(v1-v0)*min_jerk((times[m]-t0)/max(t1-t0, 1e-9))
    out[times > keys[-1][0]] = keys[-1][1]
    return out


def ramp(n, a, b, edge):
    w = np.zeros(n); w[a:b] = 1; k = max(1, edge)
    up = .5-.5*np.cos(np.pi*np.arange(1, k+1)/(k+1))
    w[a:a+k] = up[:max(0, min(k, b-a))]; w[max(a, b-k):b] = up[::-1][-min(k, b-a):]
    return w


# ---------------------------------------------------------------- models

def _vec(v):
    return np.array([v.get(k) for k in range(3)])


def add_stations(model):
    for s in SIDES:
        for body in ('calcn', 'toes'):
            for k, off in enumerate(STATIONS):
                model.addMarker(osim.Marker(f'{body}_{s}_{k}', model.getBodySet().get(f'{body}_{s}'), osim.Vec3(*off)))


def subject_model(seed, path):
    rng = np.random.default_rng(seed); sub = SPEC['subject']
    model = osim.Model(str(biomech.MODEL)); state = model.initSystem()
    k = rng.uniform(*sub['scale'])
    factors = {seg: k*rng.uniform(1-sub['segment_jitter'], 1+sub['segment_jitter']) for seg in biomech.SEGMENTS}
    scales = osim.ScaleSet()
    for seg, bodies in biomech.SEGMENTS.items():
        for b in bodies:
            s = osim.Scale(); s.setSegmentName(b); s.setScaleFactors(osim.Vec3(*[factors[seg]]*3)); s.setApply(True)
            scales.cloneAndAppend(s)
    mass = float(rng.uniform(*sub['mass_kg']))
    model.scale(state, scales, True, mass)
    unclamp(model)
    for s in SIDES:
        model.updCoordinateSet().get(f'subtalar_angle_{s}').setDefaultLocked(False)
    add_stations(model)
    model.finalizeConnections(); model.printToXML(str(path))
    return {'seed': seed, 'scale': k, 'segment_factors': factors, 'mass_kg': mass, 'foot_factor': factors['foot'], 'torso_factor': factors['torso']}


def unclamp(model):
    for name in ('pelvis_tx', 'pelvis_ty', 'pelvis_tz', 'pelvis_rotation'):
        c = model.updCoordinateSet().get(name); c.setRangeMin(-1e3); c.setRangeMax(1e3); c.setDefaultClamped(False)


def demo_model(path):
    model = osim.Model(str(DEMO/'subject_walk_scaled.osim')); model.initSystem()
    unclamp(model)
    add_stations(model)
    model.finalizeConnections(); model.printToXML(str(path))
    state = model.initSystem()
    acr = _vec(model.getJointSet().get('acromial_r').getParentFrame().findTransformInBaseFrame().p())
    return {'mass_kg': float(model.getTotalMass(state)), 'torso_factor': float(acr[1]/ACROMION_Y), 'foot_factor': 1.}


# ---------------------------------------------------------------- kinematics helpers

def coordinate_names(model):
    cs = model.getCoordinateSet()
    return [cs.get(i).getName() for i in range(cs.getSize())]


def set_row(model, state, labels, row):
    biomech._set(model, state, labels, row)


def stations(model, state, names):
    ms = model.getMarkerSet()
    return np.array([_vec(ms.get(n).getLocationInGround(state)) for n in names])


def station_names(side=None):
    return [f'{b}_{s}_{k}' for s in (SIDES if side is None else (side,)) for b in ('calcn', 'toes') for k in range(len(STATIONS))]


def landmark_heights(model, state):
    b = model.getBodySet()
    return {s: {'heel': float(b.get(f'calcn_{s}').getPositionInGround(state).get(1)),
                'toe': float(b.get(f'toes_{s}').getPositionInGround(state).get(1))} for s in SIDES}


def neutral(model_path, foot_factor):
    """Default-pose station targets per side, floor height and flat-foot landmark heights above it."""
    model = osim.Model(str(model_path)); state = model.initSystem()
    h = landmark_heights(model, state)
    floor = min(h[s]['toe'] for s in SIDES)-SPEC['landmark_floor_offset_m']*foot_factor
    ref = {s: {k: v-floor for k, v in h[s].items()} for s in SIDES}
    return {s: stations(model, state, station_names(s)) for s in SIDES}, float(floor), ref


def write_trc(points, names, fps, path):
    table = osim.TimeSeriesTableVec3(); table.setColumnLabels(names)
    for i in range(points.shape[0]):
        row = osim.RowVectorVec3(len(names))
        for k in range(len(names)):
            row[k] = osim.Vec3(*points[i, k])
        table.appendRow(i/fps, row)
    table.addTableMetaDataString('DataRate', f'{fps:g}'); table.addTableMetaDataString('Units', 'm')
    osim.TRCFileAdapter.write(table, str(path))


def write_mot(times, labels, data, path):
    table = osim.TimeSeriesTable(); table.setColumnLabels(labels)
    for t, row in zip(times, data):
        table.appendRow(float(t), osim.RowVector([float(v) for v in row]))
    table.addTableMetaDataString('inDegrees', 'yes')
    osim.STOFileAdapter.write(table, str(path))


# ---------------------------------------------------------------- constructed classes

def plans(cls, T):
    """(coordinate keyframes {name: [(t, deg)]}, foot offsets {side: (T, 3) m}, contact {side: (T,) bool}, support weight on r)."""
    t = np.arange(T)/FPS; D = t[-1]
    pose = lambda v: [(0., 0.), (.6, v), (2.2, v), (D, 0.)]
    keys, feet = {}, {s: np.zeros((T, 3)) for s in SIDES}
    contact = {s: np.ones(T, bool) for s in SIDES}; w_r = np.full(T, .5)
    if cls == 'stand':
        keys = {'lumbar_extension': [(0., 0.), (1.5, -3.), (D, 0.)], 'arm_flex_r': [(0., 0.), (1.5, 8.), (D, 0.)]}
    elif cls == 'squat':
        keys = {**{f'hip_flexion_{s}': pose(85.) for s in SIDES}, **{f'knee_angle_{s}': pose(105.) for s in SIDES},
                **{f'ankle_angle_{s}': pose(25.) for s in SIDES}, 'lumbar_extension': pose(-15.),
                **{f'arm_flex_{s}': pose(70.) for s in SIDES}}
    elif cls == 'stoop':
        keys = {**{f'hip_flexion_{s}': pose(70.) for s in SIDES}, **{f'knee_angle_{s}': pose(15.) for s in SIDES},
                'lumbar_extension': pose(-35.), **{f'arm_flex_{s}': pose(45.) for s in SIDES}}
    elif cls == 'reach':
        keys = {'arm_flex_r': pose(100.), 'elbow_flex_r': pose(10.), 'lumbar_extension': pose(-12.),
                **{f'hip_flexion_{s}': pose(15.) for s in SIDES}, 'lumbar_rotation': pose(10.)}
    elif cls == 'weight_shift':
        keys = {'lumbar_bending': pose(6.), 'arm_add_l': pose(-20.)}
        w_r = keyframed(t, [(0., .5), (.8, .8), (2.0, .8), (D, .5)])
    elif cls in ('single_leg', 'single_leg_unbalanced'):
        keys = {'hip_flexion_l': [(0., 0.), (.9, 0.), (1.5, 45.), (2.0, 45.), (2.6, 0.)],
                'knee_angle_l': [(0., 0.), (.9, 0.), (1.5, 70.), (2.0, 70.), (2.6, 0.)], 'arm_add_l': pose(-25.)}
        lift = keyframed(t, [(0., 0.), (.9, 0.), (1.5, .12), (2.0, .12), (2.6, 0.)])
        feet['l'][:, 1] = lift; feet['l'][:, 0] = .5*lift
        contact['l'] = lift <= 1e-6
        if cls == 'single_leg':
            w_r = keyframed(t, [(0., .5), (.8, 1.), (2.7, 1.), (D, .5)])
    elif cls == 'step':
        keys = {f'arm_flex_{s}': [(0., 0.), (1.2, 12. if s == 'l' else -8.), (2.1, -8. if s == 'l' else 12.), (D, 0.)] for s in SIDES}
        w_r = keyframed(t, [(0., .5), (.9, 0.), (1.5, 0.), (1.8, 1.), (2.4, 1.), (D, .5)])
        for side, (a, b) in (('r', (.9, 1.5)), ('l', (1.8, 2.4))):
            keys[f'hip_flexion_{side}'] = [(0., 0.), (.6, 5.), (a, 5.), ((a+b)/2, 30.), (b, 5.), (2.7, 5.), (D, 0.)]
            keys[f'knee_angle_{side}'] = [(0., 0.), (.6, 8.), (a, 8.), ((a+b)/2, 45.), (b, 8.), (2.7, 8.), (D, 0.)]
            s = np.clip((t-a)/(b-a), 0, 1)
            feet[side][:, 0] = .35*min_jerk(s); feet[side][:, 1] = .06*np.sin(np.pi*s)*((t > a) & (t < b))
            contact[side] = (t <= a) | (t >= b)
    else:
        raise ValueError(cls)
    return keys, feet, contact, w_r


def construct(model_path, info, cls, slide_m_s=0.):
    """Truth coordinates (degrees / metres, 100 Hz) of a constructed class and its contact plan."""
    c = SPEC['constructed']; T = int(round(c['duration_s']*FPS))+1; t = np.arange(T)/FPS
    model = osim.Model(str(model_path)); state = model.initSystem()
    labels = coordinate_names(model)
    keys, feet, contact, w_r = plans(cls, T)
    if slide_m_s:
        a, b = T//3, 2*T//3
        d = np.zeros(T); d[a:b] = slide_m_s*np.arange(b-a)/FPS; d[b:] = d[b-1]
        feet['r'][:, 0] += d
    ref_st, floor, ref = neutral(model_path, info['foot_factor'])
    names = station_names()
    targets = np.concatenate([ref_st[s][None]+feet[s][:, None] for s in SIDES], 1)
    cs = model.getCoordinateSet()
    defaults = np.array([cs.get(n).getDefaultValue()*(180/np.pi if cs.get(n).getMotionType() == 1 else 1.) for n in labels])
    coords = np.repeat(defaults[None], T, 0)
    for n, k in keys.items():
        coords[:, labels.index(n)] = keyframed(t, k)
    work = Path(model_path).parent
    write_trc(targets, names, FPS, work/'feet.trc')
    balance = cls != 'single_leg_unbalanced'
    free = [n for n in labels if not cs.get(n).get_locked() and not cs.get(n).isConstrained(state)]
    pelvis = [n for n in free if n.startswith('pelvis_')]
    leg = [n for n in free if n.startswith(('hip_', 'knee_', 'ankle_', 'subtalar_'))]
    upper = [n for n in free if n not in pelvis+leg]
    w = c['ik_weights']
    tasks = {**{n: w['leg_coordinate'] for n in leg}, **{n: w['coordinate'] for n in upper}}
    com_error = None
    for it in range(1+c['balance_iterations']*balance):
        write_mot(t, labels, coords, work/'keys.mot')
        labels_ik, data = ik(model_path, work, names, tasks, T)
        coords = np.stack([data[:, labels_ik.index(n)] if n in labels_ik else coords[:, k] for k, n in enumerate(labels)], 1)
        com_error = support_error(model, labels, coords, ref_st, feet, w_r)
        if not balance or it == c['balance_iterations']:
            break
        shift = com_shift(com_error, model, labels, coords, ref_st)/c['balance_gain']
        coords[:, labels.index('pelvis_tx')] += shift[:, 0]; coords[:, labels.index('pelvis_tz')] += shift[:, 1]
        tasks = {**{n: w['pelvis_balance'] for n in pelvis if n != 'pelvis_ty'}, **tasks}
    from scipy.signal import butter, sosfiltfilt
    coords = sosfiltfilt(butter(4, c['truth_cutoff_hz'], fs=FPS, output='sos'), coords, axis=0)
    if balance:
        com_error = support_error(model, labels, coords, ref_st, feet, w_r)
    slip = foot_slip(model, labels, coords, targets, names, contact)
    return {'times': t, 'labels': labels, 'coords': coords, 'contact': {s: {'heel': contact[s].copy(), 'toe': contact[s].copy()} for s in SIDES},
            'floor': floor, 'ref': ref, 'foot_slip_m': slip,
            'com_support_error_m': float(np.abs(com_error).max()) if balance else None}


def _com(model, labels, coords):
    from scipy.signal import butter, sosfiltfilt
    kin = biomech.body_kinematics(model, labels, coords)
    com = (kin['mass'][None, :, None]*kin['com']).sum(1)/kin['mass'].sum()
    return sosfiltfilt(butter(4, SPEC['constructed']['truth_cutoff_hz'], fs=FPS, output='sos'), com, axis=0)


def com_shift(error, model, labels, coords, ref_st):
    """COM shift y with y - h / g * y'' = ZMP error (zero velocity at the ends): the shift that moves the
    linear-inverted-pendulum ZMP onto the support point without feeding its own acceleration back."""
    T = len(error); h = float(np.mean(_com(model, labels, coords)[:, 1]-ref_st['r'][:, 1].mean()))
    D2 = (np.eye(T, k=1)-2*np.eye(T)+np.eye(T, k=-1))*FPS**2
    D2[0, 1] = D2[-1, -2] = 2*FPS**2
    return np.linalg.solve(np.eye(T)-h/9.80665*D2, error)


def support_error(model, labels, coords, ref_st, feet, w_r):
    """Horizontal (x, z) offset from the linear-inverted-pendulum ZMP (COM - h / g * COM acceleration) to the keyframed
    support point."""
    com = _com(model, labels, coords)
    acc = np.gradient(np.gradient(com, 1/FPS, axis=0), 1/FPS, axis=0)
    zmp = com-(com[:, 1:2]-ref_st['r'][:, 1].mean())/9.80665*acc
    centre = {s: ref_st[s].mean(0)[None]+feet[s] for s in SIDES}
    want = w_r[:, None]*centre['r']+(1-w_r[:, None])*centre['l']
    return (want-zmp)[:, [0, 2]]


def ik(model_path, work, names, tasks, T):
    """IK on the foot stations plus coordinate tasks {name: weight} tracked from keys.mot."""
    model = osim.Model(str(model_path)); model.initSystem()
    tool = osim.InverseKinematicsTool(); tool.setModel(model)
    tool.setMarkerDataFileName(str(work/'feet.trc')); tool.setCoordinateFileName(str(work/'keys.mot'))
    tool.setStartTime(0.); tool.setEndTime((T-1)/FPS); tool.setOutputMotionFileName(str(work/'ik.mot')); tool.setResultsDir(str(work))
    ts = tool.getIKTaskSet()
    for n in names:
        task = osim.IKMarkerTask(); task.setName(n); task.setWeight(SPEC['constructed']['ik_weights']['foot_station']); task.setApply(True)
        ts.cloneAndAppend(task)
    for n, weight in tasks.items():
        task = osim.IKCoordinateTask(); task.setName(n); task.setApply(True); task.setWeight(weight)
        task.setValueType(osim.IKCoordinateTask.StringToValueType('from_file')); ts.cloneAndAppend(task)
    tool.run()
    _, labels, data = biomech.read_storage(work/'ik.mot')
    return labels, data


def foot_slip(model, labels, coords, targets, names, contact):
    state = model.initSystem(); worst = 0.
    for i in range(len(coords)):
        set_row(model, state, labels, coords[i])
        p = stations(model, state, names)
        for k, n in enumerate(names):
            if contact[n.split('_')[1]][i]:
                worst = max(worst, float(np.linalg.norm(p[k]-targets[i, k])))
    return worst


# ---------------------------------------------------------------- demo walk

def demo_walk(model_path):
    """Tiled Moco example stride on the demo model: coordinates (deg / m, 100 Hz), force-plate contact, floor 0."""
    d = SPEC['demo_walk']
    tc, lc, c = biomech.read_storage(DEMO/'coordinates.sto')
    lc = [x.split('/')[-2] for x in lc]
    tg, lg, g = biomech.read_storage(DEMO/'grf_walk.mot')
    model = osim.Model(str(model_path)); state = model.initSystem(); cs = model.getCoordinateSet()
    labels = coordinate_names(model)
    t0, t1 = d['stride_s']; src = np.arange(t0, t1, 1/FPS)
    rows = np.stack([np.interp(src, tc, c[:, lc.index(n)]) if n in lc else np.zeros(len(src)) for n in labels], 1)
    for s in SIDES:
        if f'knee_angle_{s}_beta' in labels:
            rows[:, labels.index(f'knee_angle_{s}_beta')] = rows[:, labels.index(f'knee_angle_{s}')]
    deg = np.array([180/np.pi if cs.get(n).getMotionType() == 1 else 1. for n in labels])
    rows = rows*deg[None]
    flags = {s: np.interp(src, tg, g[:, lg.index(f'ground_force_{s}_vy')]) > d['grf_contact_n'] for s in SIDES}
    n, k = len(src), int(round(d['crossfade_s']*FPS)); step = n-k
    ix = labels.index('pelvis_tx'); speed = (rows[-1, ix]-rows[0, ix])/(src[-1]-src[0])
    T = step*(d['tiles']-1)+n
    out = np.zeros((T, len(labels))); wsum = np.zeros(T); fl = {s: np.zeros(T) for s in SIDES}
    w = np.ones(n); w[:k] = np.linspace(0, 1, k+2)[1:-1]; w[-k:] = np.linspace(1, 0, k+2)[1:-1]
    for tile in range(d['tiles']):
        r = rows.copy(); r[:, ix] = rows[0, ix]+speed*(np.arange(n)/FPS+tile*step/FPS)
        ww = w.copy()
        if tile == 0:
            ww[:k] = 1
        if tile == d['tiles']-1:
            ww[-k:] = 1
        out[tile*step:tile*step+n] += ww[:, None]*r; wsum[tile*step:tile*step+n] += ww
        for s in SIDES:
            fl[s][tile*step:tile*step+n] += ww*flags[s]
    out /= wsum[:, None]
    contact = {s: (fl[s]/wsum) > .5 for s in SIDES}
    edge = int(round(d['exclude_margin_s']*FPS)); exclude = np.zeros(T, bool); exclude[:edge] = exclude[-edge:] = True
    for tile in range(1, d['tiles']):
        exclude[max(0, tile*step-edge):tile*step+k+edge] = True
    return {'times': np.arange(T)/FPS, 'labels': labels, 'coords': out, 'contact_plate': contact, 'floor': 0., 'exclude': exclude}


def demo_reference(model_path, walk):
    """Flat-foot landmark heights of the demo model: median landmark height above the floor in plate-contact frames
    where both landmarks of the foot move vertically slower than 0.1 m/s (as the kin30 reference)."""
    model = osim.Model(str(model_path)); state = model.initSystem()
    h = {s: {'heel': [], 'toe': []} for s in SIDES}
    rows = []
    for i in range(len(walk['times'])):
        set_row(model, state, walk['labels'], walk['coords'][i])
        rows.append(landmark_heights(model, state))
    ref = {}
    for s in SIDES:
        H = np.array([[r[s]['heel'], r[s]['toe']] for r in rows])
        v = np.abs(np.gradient(H, 1/FPS, axis=0))
        sel = walk['contact_plate'][s] & (v < .1).all(1)
        ref[s] = {'heel': float(np.median(H[sel, 0])-walk['floor']), 'toe': float(np.median(H[sel, 1])-walk['floor'])}
    return ref


# ---------------------------------------------------------------- defects

def apply_defect(base, defect, size):
    """Truth motion of a defect clip (dict as construct / demo output with 'fps'); bone_stretch only marks the window."""
    x = {**base, 'coords': base['coords'].copy(), 'fps': FPS, 'window': None, 'stretch': None}
    T = len(x['coords']); a, b = T//3, 2*T//3; edge = int(round(.1*FPS)); L = x['labels']
    if defect in ('float_window_m', 'penetrate_window_m', 'float_whole_m', 'penetrate_whole_m'):
        sign = 1 if defect.startswith('float') else -1
        prof = ramp(T, a, b, edge) if 'window' in defect else np.ones(T)
        x['coords'][:, L.index('pelvis_ty')] += sign*size*prof
        x['window'] = (a, b) if 'window' in defect else None
    elif defect == 'lean_deg':
        x['coords'][:, L.index('lumbar_extension')] -= size*ramp(T, a, b, edge); x['window'] = (a, b)
    elif defect == 'time_compression':
        x['fps'] = FPS*size
    elif defect == 'bone_stretch':
        x['stretch'] = size*ramp(T, a, b, edge); x['window'] = (a, b)
    elif defect not in ('control', 'slide_m_s'):
        raise ValueError(defect)
    return x


# ---------------------------------------------------------------- truth

def elevations(model_path, labels, coords, floor, ref):
    model = osim.Model(str(model_path)); state = model.initSystem()
    out = {s: {'heel': np.zeros(len(coords)), 'toe': np.zeros(len(coords))} for s in SIDES}
    for i in range(len(coords)):
        set_row(model, state, labels, coords[i])
        h = landmark_heights(model, state)
        for s in SIDES:
            for k in ('heel', 'toe'):
                out[s][k][i] = h[s][k]-floor-ref[s][k]
    return out


def truth(model_path, clip, mass, source):
    """Per-frame truth of the clip: contact, elevation, realizability residual (BW), GRF, E label (1 positive,
    0 negative, -1 ambiguous) and the measurability label."""
    tr = SPEC['truth']; low, high = tr['contact_band_m'][source]; pen_lo, pen_hi = tr['penetration_m'][source]
    labels, coords, fps = clip['labels'], clip['coords'], clip['fps']
    elev = elevations(model_path, labels, coords, clip['floor'], clip['ref'])
    base = clip['contact']
    contact = {s: {k: base[s][k] & (elev[s][k] <= low) for k in elev[s]} for s in SIDES}
    off = {s: {k: (~base[s][k]) | (elev[s][k] >= high) for k in elev[s]} for s in SIDES}
    ambiguous_contact = np.any([~(contact[s][k] | off[s][k]) for s in SIDES for k in elev[s]], 0)
    lowest = np.min([elev[s][k] for s in SIDES for k in elev[s]], 0)
    model = osim.Model(str(model_path))
    kin = biomech.body_kinematics(model, labels, coords)
    F, M, C = biomech.required_wrench(kin, 1/fps)
    points = {s: biomech.sole_points(kin, s)[0] for s in SIDES}
    bw = mass*9.80665
    grf, rF, _ = biomech.estimate_grf(F, M, C, points, contact, clip['floor'], bw)
    r = np.linalg.norm(rF, axis=1)/bw
    rb = tr['residual_bw']
    positive = (lowest <= pen_hi) | ((r >= rb['positive_min']) & ~ambiguous_contact)
    negative = (lowest > pen_lo) & (r <= rb['negative_max']) & ~ambiguous_contact
    e = np.where(positive, 1, np.where(negative, 0, -1))
    if clip.get('exclude') is not None:
        e[clip['exclude']] = -1
    measurable = np.zeros(len(coords), int)
    if clip.get('stretch') is not None:
        st = clip['stretch']; measurable = np.where(st >= .1, 1, np.where(st <= .02, 0, -1))
    return {'contact': contact, 'elevation': elev, 'lowest_elevation': lowest, 'residual_bw': r, 'grf': grf,
            'e_label': e, 'unmeasurable_label': measurable, 'com': C}


# ---------------------------------------------------------------- export

def export(model_path, clip, info, seed):
    """Generator-like sparse joints (canonical Z-up, export fps) with landmark offsets and jitter; floor shifted."""
    ex = SPEC['export']; rng = np.random.default_rng(seed)
    model = osim.Model(str(model_path)); state = model.initSystem(); bodies = model.getBodySet()
    fps_in = clip['fps']; fps_out = float(SPEC['fps_export'])
    n_out = int(np.floor((len(clip['coords'])-1)/fps_in*fps_out))+1
    src = np.arange(n_out)/fps_out*fps_in
    rows = np.stack([np.interp(src, np.arange(len(clip['coords'])), clip['coords'][:, k]) for k in range(len(clip['labels']))], 1)
    names = ex['joints']; spec = {}
    for n in names:
        if n == 'pelvis':
            spec[n] = ('pelvis', np.zeros(3))
        elif n in TORSO:
            spec[n] = ('torso', np.array(TORSO[n])*info['torso_factor'])
        else:
            side, part = n.split('_', 1); spec[n] = (f"{LIMB[part]}_{side[0]}", np.zeros(3))
    offset = {n: rng.normal(0, ex['landmark_offset_sd_m'], 3) for n in names}
    shift = float(rng.uniform(*ex['floor_shift_m']))
    out = np.zeros((n_out, len(names), 3)); clean = np.zeros_like(out)
    for i in range(n_out):
        set_row(model, state, clip['labels'], rows[i])
        for k, n in enumerate(names):
            body, loc = spec[n]
            clean[i, k] = _vec(bodies.get(body).findStationLocationInGround(state, osim.Vec3(*loc)))
            out[i, k] = _vec(bodies.get(body).findStationLocationInGround(state, osim.Vec3(*(loc+offset[n]))))
    if clip.get('stretch') is not None:
        st = np.interp(src, np.arange(len(clip['coords'])), clip['stretch'])
        sh, el, wr = (names.index(f'right_{p}') for p in ('shoulder', 'elbow', 'wrist'))
        for arr in (out, clean):
            up = arr[:, el]-arr[:, sh]; fo = arr[:, wr]-arr[:, el]
            arr[:, el] = arr[:, sh]+up*(1+st[:, None]); arr[:, wr] = arr[:, el]+fo*(1+st[:, None])
    out += rng.normal(0, ex['jitter_sd_m'], out.shape)
    out[..., 1] += shift; clean[..., 1] += shift
    to_z = lambda p: p @ biomech.ROTATION
    return {'joints': to_z(out), 'clean_joints': to_z(clean), 'joint_names': names, 'fps': fps_out,
            'floor_height_m': clip['floor']+shift, 'floor_shift_m': shift}
