"""OpenSim musculoskeletal evaluation of generated ARDY motions (Rajagopal full-body model).

Runs in the OpenSim 4.6 Python environment. Generated joints (Z-up, metres) become
joint-centre markers; the model is scaled per segment from the generated skeleton,
axial markers are registered on a lumbar-locked calibration pass, then IK, inverse
dynamics without external loads (required external wrench), contact-constrained
ground reaction estimation, inverse dynamics with those loads and static optimization.
"""
import json
from pathlib import Path

import numpy as np
import opensim as osim

MODEL = Path.home()/'.local/share/opensimcreator/resources/OpenSimCreator/models/RajagopalModel/RajagopalLaiUhlrich2023.osim'
LIMB = {side: {f'{name}_{s}': (joint, body, f'{prefix}_{part}')
               for name, joint, body, part in (('hip', 'hip', 'femur', 'hip'), ('knee', 'walker_knee', 'tibia', 'knee'),
                                               ('ankle', 'ankle', 'talus', 'ankle'), ('toe', 'mtp', 'toes', 'foot'),
                                               ('shoulder', 'acromial', 'humerus', 'shoulder'), ('elbow', 'elbow', 'ulna', 'elbow'),
                                               ('wrist', 'radius_hand', 'hand', 'wrist'))
               for joint, body in [(f'{joint}_{s}', f'{body}_{s}')]}
        for side, s, prefix in (('right', 'r', 'right'), ('left', 'l', 'left'))}
MARKERS = {name: (joint, body, source) for side in LIMB.values() for name, (joint, body, source) in side.items()}
AXIAL = {'pelvis_c': ('pelvis', 'pelvis'), 'spine_c': ('torso', 'Spine'), 'chest_c': ('torso', 'Spine3'), 'neck_c': ('torso', 'neck')}
WEIGHTS = {**{name: 1. for name in MARKERS}, **{name: .3 for name in AXIAL}}
LUMBAR = ('lumbar_extension', 'lumbar_bending', 'lumbar_rotation')
ROTATION = np.array([[1., 0, 0], [0, 0, 1], [0, -1, 0]])


def to_opensim(joints):
    return np.asarray(joints, float) @ ROTATION.T


def load(run):
    motion = np.load(Path(run)/'motion.npz')
    names = [str(n) for n in motion['joint_names']]
    return to_opensim(motion['joints']), names, float(motion['fps'])


def _child_origin(model, state, joint):
    frame = model.getJointSet().get(joint).getChildFrame()
    return np.array([frame.getPositionInGround(state).get(k) for k in range(3)])


def segment_lengths(points, names, torso_percentile=50.):
    """Median joint distances; the torso span crosses the lumbar joint, whose flexion only shortens it, so a high
    torso_percentile approximates the zero-lumbar length the model scale needs."""
    p = {n: points[:, names.index(n)] for n in names}
    d = lambda a, b: float(np.median(np.linalg.norm(p[a]-p[b], axis=-1)))
    mid = lambda a, b: .5*(p[a]+p[b])
    torso = np.linalg.norm(mid('right_shoulder', 'left_shoulder')-mid('right_hip', 'left_hip'), axis=-1)
    return {'pelvis': d('right_hip', 'left_hip'),
            'femur': .5*(d('right_hip', 'right_knee')+d('left_hip', 'left_knee')),
            'tibia': .5*(d('right_knee', 'right_ankle')+d('left_knee', 'left_ankle')),
            'foot': .5*(d('right_ankle', 'right_foot')+d('left_ankle', 'left_foot')),
            'torso': float(np.percentile(torso, torso_percentile)),
            'humerus': .5*(d('right_shoulder', 'right_elbow')+d('left_shoulder', 'left_elbow')),
            'forearm': .5*(d('right_elbow', 'right_wrist')+d('left_elbow', 'left_wrist'))}


def model_lengths(model, state):
    c = lambda j: _child_origin(model, state, j)
    n = lambda a, b: float(np.linalg.norm(c(a)-c(b)))
    return {'pelvis': n('hip_r', 'hip_l'), 'femur': n('hip_r', 'walker_knee_r'), 'tibia': n('walker_knee_r', 'ankle_r'),
            'foot': n('ankle_r', 'mtp_r'), 'torso': float(np.linalg.norm(.5*(c('acromial_r')+c('acromial_l'))-.5*(c('hip_r')+c('hip_l')))),
            'humerus': n('acromial_r', 'elbow_r'), 'forearm': n('elbow_r', 'radius_hand_r')}


SEGMENTS = {'pelvis': ['pelvis'], 'femur': ['femur_r', 'femur_l', 'patella_r', 'patella_l'], 'tibia': ['tibia_r', 'tibia_l'],
            'foot': ['talus_r', 'talus_l', 'calcn_r', 'calcn_l', 'toes_r', 'toes_l'], 'torso': ['torso'],
            'humerus': ['humerus_r', 'humerus_l'], 'forearm': ['ulna_r', 'ulna_l', 'radius_r', 'radius_l', 'hand_r', 'hand_l']}


def scaled_model(points, names, mass, path, free_yaw=False, torso_percentile=50.):
    """free_yaw: also unclamp pelvis_rotation, the global heading (the model clamps it to +-90 degrees)."""
    model = osim.Model(str(MODEL)); state = model.initSystem()
    target, base = segment_lengths(points, names, torso_percentile), model_lengths(model, state)
    factors = {k: target[k]/base[k] for k in target}
    scales = osim.ScaleSet()
    for segment, bodies in SEGMENTS.items():
        for body in bodies:
            s = osim.Scale(); s.setSegmentName(body); s.setScaleFactors(osim.Vec3(*[factors[segment]]*3)); s.setApply(True)
            scales.cloneAndAppend(s)
    model.scale(state, scales, True, mass)
    for name in ('pelvis_tx', 'pelvis_ty', 'pelvis_tz', *(('pelvis_rotation',) if free_yaw else ())):
        coordinate = model.updCoordinateSet().get(name)
        coordinate.setRangeMin(-1e3); coordinate.setRangeMax(1e3); coordinate.setDefaultClamped(False)
    state = model.initSystem()
    for name, (joint, body, _) in MARKERS.items():
        frame = model.getJointSet().get(joint).getChildFrame()
        location = frame.findTransformInBaseFrame().p()
        model.addMarker(osim.Marker(name, model.getBodySet().get(body), location))
    for name, (body, _) in AXIAL.items():
        model.addMarker(osim.Marker(name, model.getBodySet().get(body), osim.Vec3(0, 0, 0)))
    model.finalizeConnections(); model.printToXML(str(path))
    return {'segment_factors': factors, 'target_lengths_m': target, 'model_lengths_m': base, 'mass_kg': mass}


def write_trc(points, names, fps, path, frames=None):
    labels = list(MARKERS)+list(AXIAL)
    sources = [MARKERS[n][2] for n in MARKERS]+[AXIAL[n][1] for n in AXIAL]
    index = [names.index(s) for s in sources]
    rows = range(points.shape[0]) if frames is None else frames
    table = osim.TimeSeriesTableVec3()
    table.setColumnLabels(labels)
    for i in rows:
        row = osim.RowVectorVec3(len(labels))
        for k, j in enumerate(index):
            row[k] = osim.Vec3(*points[i, j])
        table.appendRow(i/fps, row)
    table.addTableMetaDataString('DataRate', f'{fps:g}'); table.addTableMetaDataString('Units', 'm')
    osim.TRCFileAdapter.write(table, str(path))


def inverse_kinematics(model_path, trc, out, t0, t1, weights, lock_lumbar=False):
    model = osim.Model(str(model_path))
    if lock_lumbar:
        for name in LUMBAR:
            model.updCoordinateSet().get(name).setDefaultLocked(True)
    model.initSystem()
    tool = osim.InverseKinematicsTool()
    tool.setModel(model); tool.setMarkerDataFileName(str(trc)); tool.setStartTime(t0); tool.setEndTime(t1)
    tool.setOutputMotionFileName(str(out)); tool.set_report_errors(True); tool.setResultsDir(str(Path(out).parent))
    tasks = tool.getIKTaskSet()
    for name, w in weights.items():
        task = osim.IKMarkerTask(); task.setName(name); task.setWeight(w); task.setApply(w > 0); tasks.cloneAndAppend(task)
    tool.run()
    return out


def read_storage(path):
    table = osim.TimeSeriesTable(str(path))
    labels = list(table.getColumnLabels())
    data = np.array([[table.getRowAtIndex(i)[k] for k in range(len(labels))] for i in range(table.getNumRows())])
    return np.array(list(table.getIndependentColumn())), labels, data


def register_axial(model_path, ik_mot, points, names, fps, frames, out_path, markers=None):
    """Place markers ({name: (body, source joint)}, default the axial ones) at the mean body-frame location of their
    generated joints over the calibration frames."""
    markers = AXIAL if markers is None else markers
    model = osim.Model(str(model_path)); state = model.initSystem()
    times, labels, data = read_storage(ik_mot)
    coords = model.getCoordinateSet()
    sums = {name: np.zeros(3) for name in markers}
    for row in range(len(times)):
        for k, label in enumerate(labels):
            if coords.contains(label):
                c = coords.get(label)
                c.setValue(state, np.deg2rad(data[row, k]) if c.getMotionType() == 1 else data[row, k], False)
        model.realizePosition(state)
        frame = int(round(times[row]*fps))
        for name, (body, source) in markers.items():
            ground = osim.Vec3(*points[frame, names.index(source)])
            local = model.getGround().findStationLocationInAnotherFrame(state, ground, model.getBodySet().get(body))
            sums[name] += [local.get(k) for k in range(3)]
    marker_set = model.updMarkerSet()
    for name in markers:
        marker_set.get(name).set_location(osim.Vec3(*(sums[name]/len(times))))
    model.printToXML(str(out_path))
    return {name: (sums[name]/len(times)).tolist() for name in markers}


def marker_errors(ik_dir):
    path = next(Path(ik_dir).glob('*_ik_marker_errors.sto'))
    _, labels, data = read_storage(path)
    rms = data[:, labels.index('marker_error_RMS')]; peak = data[:, labels.index('marker_error_max')]
    return {'rms_mean_m': float(rms.mean()), 'rms_p95_m': float(np.percentile(rms, 95)), 'max_m': float(peak.max())}


GRAVITY = np.array([0., -9.80665, 0.])
FRICTION = .8
CONTACT_MARGIN_M = .03
SOLE = {'heel': ('calcn', (0., 0.), (.03, -.03)), 'toe': ('toes', (.04, .04), (.04, -.04))}
MOMENTS = ['hip_flexion', 'hip_adduction', 'knee_angle', 'ankle_angle']


def write_coordinates(ik_mot, out, rate=100., cutoff=6.):
    from scipy.interpolate import CubicSpline
    from scipy.signal import butter, sosfiltfilt
    times, labels, data = read_storage(ik_mot)
    fps = 1./np.median(np.diff(times))
    data = sosfiltfilt(butter(4, cutoff, fs=fps, output='sos'), data, axis=0)
    new = np.arange(times[0], times[-1]+.5/rate, 1./rate)
    data = CubicSpline(times, data, axis=0)(new)
    table = osim.TimeSeriesTable()
    table.setColumnLabels(labels)
    for t, row in zip(new, data):
        table.appendRow(float(t), osim.RowVector(row.tolist()))
    table.addTableMetaDataString('inDegrees', 'yes')
    osim.STOFileAdapter.write(table, str(out))
    return new, labels, data


def _set(model, state, labels, row):
    coords = model.getCoordinateSet()
    for k, label in enumerate(labels):
        if coords.contains(label):
            c = coords.get(label)
            c.setValue(state, np.deg2rad(row[k]) if c.getMotionType() == 1 else row[k], False)
    model.realizePosition(state)


def body_kinematics(model, labels, data):
    state = model.initSystem(); bodies = model.getBodySet()
    names = [bodies.get(i).getName() for i in range(bodies.getSize())]
    T = len(data); R = np.zeros((T, len(names), 3, 3)); c = np.zeros((T, len(names), 3)); origin = np.zeros((T, len(names), 3))
    for t in range(T):
        _set(model, state, labels, data[t])
        for i, n in enumerate(names):
            body = bodies.get(n); X = body.getTransformInGround(state)
            rot = X.R().asMat33(); R[t, i] = [[rot.get(a, b) for b in range(3)] for a in range(3)]
            p = X.p(); origin[t, i] = [p.get(k) for k in range(3)]
            com = body.findStationLocationInGround(state, body.getMassCenter()); c[t, i] = [com.get(k) for k in range(3)]
    mass = np.array([bodies.get(n).getMass() for n in names])
    inertia = []
    for n in names:
        I = bodies.get(n).getInertia(); m, p = I.getMoments(), I.getProducts()
        inertia.append([[m.get(0), p.get(0), p.get(1)], [p.get(0), m.get(1), p.get(2)], [p.get(1), p.get(2), m.get(2)]])
    return {'names': names, 'R': R, 'com': c, 'origin': origin, 'mass': mass, 'inertia': np.array(inertia)}


def required_wrench(kin, dt):
    """Non-gravitational external force and moment about the whole-body COM needed by the motion."""
    m, c, R = kin['mass'], kin['com'], kin['R']
    a = np.gradient(np.gradient(c, dt, axis=0), dt, axis=0)
    C = (m[None, :, None]*c).sum(1)/m.sum()
    F = (m[None, :, None]*(a-GRAVITY)).sum(1)
    Rdot = np.gradient(R, dt, axis=0)
    W = Rdot@np.swapaxes(R, -1, -2)
    omega = np.stack([W[..., 2, 1]-W[..., 1, 2], W[..., 0, 2]-W[..., 2, 0], W[..., 1, 0]-W[..., 0, 1]], -1)/2
    I_ground = R@kin['inertia'][None]@np.swapaxes(R, -1, -2)
    H = np.einsum('tbij,tbj->tbi', I_ground, omega)
    M = np.cross(c-C[:, None], m[None, :, None]*a).sum(1)+np.gradient(H, dt, axis=0).sum(1)
    return F, M, C


def sole_points(kin, side):
    """Heel and toe sole points (ground frame, two per landmark) and landmark heights."""
    idx = {k: kin['names'].index(f'{body}_{side}') for k, (body, _, _) in SOLE.items()}
    heel, mtp = kin['origin'][:, idx['heel']], kin['origin'][:, idx['toe']]
    forward = mtp-heel; forward[:, 1] = 0; forward /= np.linalg.norm(forward, axis=-1, keepdims=True)+1e-9
    lateral = np.cross(forward, np.array([0., 1., 0.]))
    points = {}
    for key, (_, (f0, f1), (l0, l1)) in SOLE.items():
        base = heel if key == 'heel' else mtp
        points[key] = np.stack([base+f0*forward+l0*lateral, base+f1*forward+l1*lateral], 1)
    return points, {'heel': heel[:, 1], 'toe': mtp[:, 1]}


def onset_contact(low, speed, von):
    """Within each run of low frames, contact from the first frame with speed <= von to the end of the run; runs that
    never reach von are swing."""
    out = np.zeros_like(low, dtype=bool)
    edges = np.diff(np.r_[0, low.astype(int), 0])
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        rest = np.flatnonzero(speed[a:b] <= von)
        if len(rest):
            out[a+rest[0]:b] = True
    return out


def contact_masks(heights, margin=CONTACT_MARGIN_M, dt=None, vmax=None, hmax=None, horizontal=None, onset=None):
    """Landmark within margin of its 5th-percentile height; with vmax, also vertical speed <= vmax (m/s); with hmax, also
    horizontal speed <= hmax (m/s); with onset, the onset rule of onset_contact on the horizontal speed (m/s);
    horizontal: {side: {landmark: (T, 2) ground-plane positions}}."""
    floor = min(np.percentile(h, 5) for side in heights.values() for h in side.values())

    def mask(side, k, h):
        m = h <= np.percentile(h, 5)+margin
        if vmax is not None:
            m = m & (np.abs(np.gradient(h, dt)) <= vmax)
        if hmax is not None or onset is not None:
            v = np.linalg.norm(np.gradient(horizontal[side][k], dt, axis=0), axis=1)
            if hmax is not None:
                m = m & (v <= hmax)
            if onset is not None:
                m = onset_contact(m, v, onset)
        return m
    return floor, {side: {k: mask(side, k, h) for k, h in hs.items()} for side, hs in heights.items()}


def landmark_ground(kin, side):
    """Heel (calcn origin) and toe (toes origin) ground-plane positions, the landmarks of contact_masks."""
    return {k: kin['origin'][:, kin['names'].index(f'{body}_{side}')][:, [0, 2]] for k, (body, _, _) in SOLE.items()}


def _edges(mu=FRICTION):
    e = np.array([[mu, 1, 0], [-mu, 1, 0], [0, 1, mu], [0, 1, -mu]], float)
    return e/np.linalg.norm(e, axis=1, keepdims=True)


def estimate_grf(F, M, C, points, masks, floor, body_weight, regularization=1e-4, moment_scale=10.):
    from scipy.optimize import nnls
    edges = _edges(); T = len(F)
    out = {side: {'force': np.zeros((T, 3)), 'moment': np.zeros((T, 3))} for side in points}
    residual_F, residual_M = np.zeros((T, 3)), np.zeros((T, 3))
    for t in range(T):
        columns, owner = [], []
        for side, pts in points.items():
            for key, p in pts.items():
                if not masks[side][key][t]:
                    continue
                for q in p[t]:
                    q = q.copy(); q[1] = floor
                    for e in edges:
                        columns.append(np.concatenate([e, moment_scale*np.cross(q-C[t], e)])); owner.append((side, q, e))
        target = np.concatenate([F[t], moment_scale*M[t]])
        if not columns:
            residual_F[t], residual_M[t] = F[t], M[t]; continue
        A = np.array(columns).T
        A_aug = np.vstack([A, np.sqrt(regularization)*np.eye(A.shape[1])])
        lam, _ = nnls(A_aug, np.concatenate([target, np.zeros(A.shape[1])]))
        achieved = A@lam
        residual_F[t] = F[t]-achieved[:3]; residual_M[t] = M[t]-achieved[3:]/moment_scale
        for (side, q, e), l in zip(owner, lam):
            out[side]['force'][t] += l*e; out[side]['moment'][t] += l*np.cross(q, e)
    return out, residual_F, residual_M


def write_loads(times, grf, floor, path, xml):
    labels, rows = [], []
    for side, w in grf.items():
        F, Mo = w['force'], w['moment']; Fy = np.where(F[:, 1] > 1e-6, F[:, 1], np.inf)
        px = np.where(np.isfinite(Fy), (Mo[:, 2]+floor*F[:, 0])/Fy, 0.)
        pz = np.where(np.isfinite(Fy), (floor*F[:, 2]-Mo[:, 0])/Fy, 0.)
        Ty = Mo[:, 1]-(pz*F[:, 0]-px*F[:, 2])
        cols = {f'ground_force_{side}_vx': F[:, 0], f'ground_force_{side}_vy': F[:, 1], f'ground_force_{side}_vz': F[:, 2],
                f'ground_force_{side}_px': px, f'ground_force_{side}_py': np.full_like(px, floor), f'ground_force_{side}_pz': pz,
                f'ground_torque_{side}_x': np.zeros_like(px), f'ground_torque_{side}_y': np.where(np.isfinite(Fy), Ty, 0.), f'ground_torque_{side}_z': np.zeros_like(px)}
        labels += list(cols); rows.append(np.stack(list(cols.values()), 1))
    data = np.concatenate(rows, 1)
    table = osim.TimeSeriesTable(); table.setColumnLabels(labels)
    for t, row in zip(times, data):
        table.appendRow(float(t), osim.RowVector(row.tolist()))
    osim.STOFileAdapter.write(table, str(path))
    loads = osim.ExternalLoads()
    for side in grf:
        f = osim.ExternalForce(); f.setName(f'grf_{side}'); f.set_applied_to_body(f'calcn_{side}')
        f.set_force_expressed_in_body('ground'); f.set_point_expressed_in_body('ground')
        f.set_force_identifier(f'ground_force_{side}_v'); f.set_point_identifier(f'ground_force_{side}_p'); f.set_torque_identifier(f'ground_torque_{side}_')
        loads.cloneAndAppend(f)
    loads.setDataFileName(str(path)); loads.printToXML(str(xml))


def inverse_dynamics(model_path, coordinates, loads_xml, out_dir, name, t0, t1):
    model = osim.Model(str(model_path)); model.initSystem()
    tool = osim.InverseDynamicsTool(); tool.setModel(model)
    tool.setCoordinatesFileName(str(coordinates)); tool.setLowpassCutoffFrequency(-1)
    tool.setStartTime(t0); tool.setEndTime(t1); tool.setResultsDir(str(out_dir)); tool.setOutputGenForceFileName(name)
    excluded = osim.ArrayStr(); excluded.append('Muscles'); tool.setExcludedForces(excluded)
    if loads_xml:
        tool.setExternalLoadsFileName(str(loads_xml))
    tool.run()
    return read_storage(Path(out_dir)/name)


def dynamics(out, mass, contact_vmax=None, contact_hmax=None, contact_onset=None):
    out = Path(out).resolve()
    model_path = out/'model.osim'
    times, labels, data = write_coordinates(out/'ik/ik.mot', out/'coordinates.mot')
    dt = float(np.median(np.diff(times)))
    model = osim.Model(str(model_path))
    kin = body_kinematics(model, labels, data)
    F, M, C = required_wrench(kin, dt)
    points, heights = {}, {}
    for side in ('r', 'l'):
        points[side], heights[side] = sole_points(kin, side)
    horizontal = {side: landmark_ground(kin, side) for side in ('r', 'l')} if contact_hmax is not None or contact_onset is not None else None
    floor, masks = contact_masks(heights, dt=dt, vmax=contact_vmax, hmax=contact_hmax, horizontal=horizontal, onset=contact_onset)
    bw = mass*9.80665
    grf, rF, rM = estimate_grf(F, M, C, points, masks, floor, bw)
    write_loads(times, grf, floor, out/'grf.mot', out/'grf.xml')
    _, id_labels, id_data = inverse_dynamics(model_path, out/'coordinates.mot', out/'grf.xml', out, 'id.sto', float(times[0]), float(times[-1]))
    np.savez_compressed(out/'dynamics.npz', times=times, required_force=F, required_moment=M, com=C, residual_force=rF, residual_moment=rM,
                        **{f'grf_{s}': grf[s]['force'] for s in grf}, **{f'contact_{s}_{k}': masks[s][k] for s in masks for k in masks[s]},
                        id=id_data, id_labels=np.array(id_labels))
    trim = slice(int(.5/dt), len(times)-int(.5/dt))
    rf = np.linalg.norm(rF, axis=1)[trim]/bw; rm = np.linalg.norm(rM, axis=1)[trim]/bw
    col = lambda n: id_data[trim, id_labels.index(n)]
    metrics = {'floor_m': float(floor), 'contact_fraction': {f'{s}_{k}': float(masks[s][k].mean()) for s in masks for k in masks[s]},
               'no_contact_fraction': float(np.mean(~np.any([masks[s][k] for s in masks for k in masks[s]], axis=0))),
               'residual_force_rms_bw': float(np.sqrt(np.mean(rf**2))), 'residual_force_p95_bw': float(np.percentile(rf, 95)),
               'unsupported_fraction_5pct_bw': float(np.mean(rf > .05)),
               'residual_moment_rms_m': float(np.sqrt(np.mean(rm**2))),
               'required_vertical_force_mean_bw': float(F[trim, 1].mean()/bw),
               'id_pelvis_residual_force_rms_bw': float(np.sqrt(np.mean(sum(col(f'pelvis_t{a}_force')**2 for a in 'xyz')))/bw),
               'lumbar_extension_moment_rms_nm_kg': float(np.sqrt(np.mean(col('lumbar_extension_moment')**2))/mass)}
    for name in MOMENTS:
        values = np.stack([col(f'{name}_{s}_moment') for s in ('r', 'l')])/mass
        metrics[f'{name}_moment_rms_nm_kg'] = float(np.sqrt(np.mean(values**2)))
        metrics[f'{name}_moment_peak_nm_kg'] = float(np.abs(values).max())
    return metrics


RESERVE_NM = 1.
RESIDUAL = {'pelvis_tx': 1e3, 'pelvis_ty': 1e3, 'pelvis_tz': 1e3, 'pelvis_tilt': 1e3, 'pelvis_list': 1e3, 'pelvis_rotation': 1e3}


def static_optimization(out, t0, t1, step=2, unlock=()):
    """Muscle activations (activation exponent 2) with weak lower-limb reserves and strong pelvis residual actuators.
    unlock: coordinates freed in the optimization model only (held at their coordinates-file value, default 0); names
    the model lacks are ignored (some synthetic base models have no subtalar joint)."""
    out = Path(out).resolve()
    model = osim.Model(str(out/'model.osim'))
    for name in unlock:
        if model.getCoordinateSet().contains(name):
            model.updCoordinateSet().get(name).setDefaultLocked(False)
    state = model.initSystem(); coords = model.getCoordinateSet()
    names = [coords.get(i).getName() for i in range(coords.getSize())
             if not coords.get(i).isConstrained(state) and not coords.get(i).getName().startswith(('lumbar', 'arm', 'elbow', 'pro_sup', 'wrist'))]
    for name in names:
        actuator = osim.CoordinateActuator(name); actuator.setName(f'reserve_{name}')
        actuator.setOptimalForce(RESIDUAL.get(name, RESERVE_NM)); actuator.setMinControl(-1e4); actuator.setMaxControl(1e4)
        model.addForce(actuator)
    model.finalizeConnections(); model.printToXML(str(out/'model_so.osim'))
    so = osim.StaticOptimization(); so.setName('so'); so.setStartTime(t0); so.setEndTime(t1)
    so.setUseModelForceSet(True); so.setActivationExponent(2); so.setUseMusclePhysiology(True); so.setStepInterval(step)
    tool = osim.AnalyzeTool(); tool.setName('so'); tool.setModelFilename(str(out/'model_so.osim'))
    tool.setCoordinatesFileName(str(out/'coordinates.mot')); tool.setExternalLoadsFileName(str(out/'grf.xml'))
    tool.setLowpassCutoffFrequency(-1); tool.setInitialTime(t0); tool.setFinalTime(t1); tool.setResultsDir(str(out/'so'))
    tool.getAnalysisSet().cloneAndAppend(so); tool.setReplaceForceSet(False)
    tool.printToXML(str(out/'so_setup.xml'))
    osim.AnalyzeTool(str(out/'so_setup.xml')).run()
    return so_metrics(out)


def so_metrics(out):
    out = Path(out).resolve()
    model = osim.Model(str(out/'model_so.osim')); model.initSystem()
    _, labels, act = read_storage(out/'so/so_so_activation.sto')
    muscles = [k for k, l in enumerate(labels) if model.getMuscles().contains(l)]
    reserves = [k for k, l in enumerate(labels) if l.startswith('reserve_') and not l.startswith('reserve_pelvis')]
    a = act[:, muscles]
    return {'muscle_effort_sum_a2_mean': float((a**2).sum(1).mean()), 'muscle_activation_peak': float(a.max()),
            'muscle_activation_mean': float(a.mean()), 'muscles': len(muscles),
            'reserve_rms_nm': float(np.sqrt(np.mean((act[:, reserves]*RESERVE_NM)**2))) if reserves else 0.,
            'reserve_peak_nm': float(np.abs(act[:, reserves]*RESERVE_NM).max()) if reserves else 0.}


def prepare(run, out, mass=75., calibration_seconds=1., free_yaw=False):
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
    points, names, fps = load(run)
    receipt = {'run': str(run), 'model_source': str(MODEL), 'fps': fps, 'frames': int(points.shape[0]), 'free_yaw': free_yaw}
    receipt['scaling'] = scaled_model(points, names, mass, out/'scaled.osim', free_yaw)
    calib = list(range(int(calibration_seconds*fps)))
    write_trc(points, names, fps, out/'calibration.trc', calib)
    limb_only = {n: (w if n in MARKERS else 0.) for n, w in WEIGHTS.items()}
    (out/'calib').mkdir(exist_ok=True)
    inverse_kinematics(out/'scaled.osim', out/'calibration.trc', out/'calib/ik.mot', 0., (len(calib)-1)/fps, limb_only, lock_lumbar=True)
    receipt['axial_markers_body_frame_m'] = register_axial(out/'scaled.osim', out/'calib/ik.mot', points, names, fps, calib, out/'model.osim')
    write_trc(points, names, fps, out/'markers.trc')
    (out/'ik').mkdir(exist_ok=True)
    inverse_kinematics(out/'model.osim', out/'markers.trc', out/'ik/ik.mot', 0., (points.shape[0]-1)/fps, WEIGHTS)
    receipt['ik_marker_error'] = marker_errors(out/'ik')
    (out/'receipt.json').write_text(json.dumps(receipt, indent=1))
    return receipt
