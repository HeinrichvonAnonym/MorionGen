"""Upper-body muscle load (config/evalkit_upper_muscle_20261108.json). OpenSim python.

lumbar: the E3 FBLS pipeline (utils/motiogen/upper_body.py, imported read-only): 324 muscles, static optimization,
reactions at the five lumbar intervertebral joints resolved along the model spine tangent, and the E3 frame
classification (type 1 run flags; type 2 unreachable frames; loads only on reachable frames).
neck, shoulder: reserved slots; they return status 'reserved' with what they need.

Modes of the lumbar region:
- evalkit: calibration on the most upright frames, the evaluation's contact decision and floor, its scored window;
- e3: the E3 front end unchanged (first-second calibration, E3 contact, 0.5 s trimmed), the cross-check;
- truth: as evalkit, with given ground reactions (grf.xml with centres of pressure) instead of the estimated ones.
"""
import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from .spec import ROOT, SIDES, LANDMARKS
from . import fit as fit_mod
import biomech
import biomech_sources
import upper_body as e3

PROTOCOL_PATH = ROOT/'config/evalkit_upper_muscle_20261108.json'
PROTOCOL = json.loads(PROTOCOL_PATH.read_text())
PREPARE = 'fbls-e3-helpers-axial-subset-1'
MODES = ('evalkit', 'e3', 'truth')
SO_STEP = 10
IK_RMS_M, REACTION_VALID = .03, .95
RESIDUAL_BW, RESERVE_NM, LIMIT_DEG = .05, 1., .5
SHEAR_N = (700., 1000.)
G = 9.80665


@lru_cache(None)
def geometry():
    return e3.geometry()


def write_trc(points, names, fps, path, labels, rows=None):
    """e3.write_trc restricted to the given marker labels."""
    import opensim as osim
    source = {**{n: v[2] for n, v in e3.MARKERS.items()}, **{n: v[1] for n, v in e3.AXIAL.items()}}
    index = [names.index(source[n]) for n in labels]
    table = osim.TimeSeriesTableVec3(); table.setColumnLabels(labels)
    for i in (range(points.shape[0]) if rows is None else rows):
        row = osim.RowVectorVec3(len(labels))
        for k, j in enumerate(index):
            row[k] = osim.Vec3(*points[i, j])
        table.appendRow(i/fps, row)
    table.addTableMetaDataString('DataRate', f'{fps:g}'); table.addTableMetaDataString('Units', 'm')
    osim.TRCFileAdapter.write(table, str(path))


def prepare(run, work, mass, mode):
    """Scaled FBLS model with registered axial markers and its IK (cached in work). Axial markers whose generated
    joint the skeleton lacks (E3 uses the ARDY spine; SMPL-family and SOMA77 skeletons have no 'spine1' after the
    biomech_sources renaming) are left out of the IK."""
    work.mkdir(parents=True, exist_ok=True)
    key = {'run': str(run), 'mass_kg': mass, 'calibration': 'first_second' if mode == 'e3' else 'upright', 'prepare': PREPARE}
    path = work/'receipt.json'
    if path.exists() and (work/'ik/ik.mot').exists():
        old = json.loads(path.read_text())
        if old.get('key') == key:
            return old
    points, names, fps = biomech_sources.load(run)
    missing = [n for n, v in e3.MARKERS.items() if v[2] not in names]
    if missing:
        raise ValueError(f'limb markers without a generated joint: {missing}')
    axial = {n: v for n, v in e3.AXIAL.items() if v[1] in names}
    weights = {**{n: e3.WEIGHTS[n] for n in e3.MARKERS}, **{n: e3.WEIGHTS[n] for n in axial}}
    labels = list(weights)
    receipt = {'key': key, 'model_source': str(e3.FBLS), 'fps': fps, 'frames': int(points.shape[0]),
               'axial_markers': list(axial), 'axial_markers_missing': [n for n in e3.AXIAL if n not in axial]}
    receipt['scaling'] = e3.scaled_model(points, names, mass, work/'scaled.osim')
    if mode == 'e3':
        calib = list(range(int(1.*fps)))
    else:
        calib, receipt['calibration'] = fit_mod.upright_frames(points, names, fps)
    write_trc(points, names, fps, work/'calibration.trc', labels, calib)
    (work/'calib').mkdir(exist_ok=True)
    e3.inverse_kinematics(work/'scaled.osim', work/'calibration.trc', work/'calib/ik.mot', calib[0]/fps, calib[-1]/fps,
                          {n: (w if n in e3.MARKERS else 0.) for n, w in weights.items()}, locked=e3.LUMBAR)
    receipt['axial_markers_body_frame_m'] = biomech.register_axial(work/'scaled.osim', work/'calib/ik.mot', points, names, fps,
                                                                   calib, work/'model.osim', markers=axial)
    write_trc(points, names, fps, work/'markers.trc', labels)
    (work/'ik').mkdir(exist_ok=True)
    e3.inverse_kinematics(work/'model.osim', work/'markers.trc', work/'ik/ik.mot', 0., (points.shape[0]-1)/fps, weights)
    receipt['ik_marker_error'] = biomech.marker_errors(work/'ik')
    receipt['frames_at_coordinate_limit'] = e3.limit_frames(work/'model.osim', work/'ik/ik.mot')
    path.write_text(json.dumps(receipt, indent=1))
    return receipt


def dynamics(work, mass, mode, floor=None, contact=None, truth=None):
    """Ground reactions and inverse dynamics of the FBLS kinematics; writes work/dynamics.npz in the E3 layout and
    returns the external-loads file."""
    import opensim as osim
    if mode == 'e3':
        e3.lumbar_dynamics(work, mass)
        return work/'grf.xml'
    times, labels, data = biomech.write_coordinates(work/'ik/ik.mot', work/'coordinates.mot')
    dt = float(np.median(np.diff(times)))
    kin = biomech.body_kinematics(osim.Model(str(work/'model.osim')), labels, data)
    F, M, C = biomech.required_wrench(kin, dt)
    if mode == 'evalkit':
        source_times, masks = contact
        near = np.clip(np.searchsorted(source_times, times-1e-9), 0, len(source_times)-1)
        near = np.where((near > 0) & (np.abs(source_times[near-1]-times) < np.abs(source_times[near]-times)), near-1, near)
        m = {s: {k: np.asarray(masks[s][k], bool)[near] for k in LANDMARKS} for s in SIDES}
        points = {s: biomech.sole_points(kin, s)[0] for s in SIDES}
        grf, rF, rM = biomech.estimate_grf(F, M, C, points, m, floor, mass*G)
        biomech.write_loads(times, grf, floor, work/'grf.mot', work/'grf.xml')
        loads = work/'grf.xml'
    else:
        import xml.etree.ElementTree as ET
        loads = Path(truth)
        tg, lg, g = biomech.read_storage(ET.parse(loads).getroot().find('.//datafile').text.strip())
        total = sum(np.stack([np.interp(times, tg, g[:, lg.index(f'ground_force_{s}_v{a}')]) for a in 'xyz'], 1) for s in SIDES)
        rF, rM = F-total, np.zeros_like(M)
    _, id_labels, id_data = biomech.inverse_dynamics(work/'model.osim', work/'coordinates.mot', loads, work, 'id.sto',
                                                     float(times[0]), float(times[-1]))
    np.savez_compressed(work/'dynamics.npz', times=times, required_force=F, required_moment=M, com=C, residual_force=rF,
                        residual_moment=rM, id=id_data, id_labels=np.array(id_labels))
    return loads


def solve(work, loads, t0, t1):
    import opensim as osim
    model = osim.Model(str(work/'model.osim')); state = model.initSystem()
    e3.add_reserves(model, state, e3.RESERVES)
    model.finalizeConnections(); model.printToXML(str(work/'model_so.osim'))
    e3.static_optimization(work, work/'model_so.osim', work/'coordinates.mot', loads, t0, t1, SO_STEP)
    e3.joint_reactions(work, work/'model_so.osim', work/'coordinates.mot', loads, t0, t1,
                       [j for j, _ in e3.LUMBAR_JOINTS.values()], SO_STEP, use_controls=False)


def project(labels, data, joint, child, axis):
    """Reaction on the child along the spine tangent: compression, anterior-posterior and medio-lateral shear."""
    F = np.stack([data[:, labels.index(f'{joint}_on_{child}_in_{child}_f{a}')] for a in 'xyz'], 1)
    u = np.asarray(axis); ap = np.cross(u, [0., 0., 1.]); ap /= np.linalg.norm(ap); ml = np.cross(ap, u)
    return F@u, F@ap, F@ml


def near_limit(values, spec):
    tol = np.deg2rad(LIMIT_DEG)
    radians = np.deg2rad(values) if spec['degrees_in_files'] else values
    return (radians <= spec['min']+tol) | (radians >= spec['max']-tol)


def classify(work, mass, t0, t1):
    """E3 classification (python/scripts/analyze_e3.py classify_lumbar) on the window [t0, t1]."""
    geo = geometry(); bw = mass*G
    receipt = json.loads((work/'receipt.json').read_text())
    t_ik, l_ik, d_ik = biomech.read_storage(next((work/'ik').glob('*_ik_marker_errors.sto')))
    window = (t_ik >= t0) & (t_ik <= t1)
    flags = {'ik_fit': bool(np.median(d_ik[window, l_ik.index('marker_error_RMS')]) > IK_RMS_M),
             'non_anatomical_limit': bool(set(receipt.get('frames_at_coordinate_limit', {})) & set(geo['pelvis']))}
    t, labels, jr = biomech.read_storage(next((work/'jr').glob('*ReactionLoads.sto')))
    finite = np.isfinite(jr).all(1) & (np.abs(jr) < 1e6).all(1)
    flags['reaction'] = bool(finite.mean() < REACTION_VALID)
    dyn = np.load(work/'dynamics.npz')
    external = np.interp(t, dyn['times'], np.linalg.norm(dyn['residual_force'], axis=1)/bw) > RESIDUAL_BW
    t_act, l_act, act = biomech.read_storage(work/'so/so_so_activation.sto')
    internal = np.zeros(len(t), bool)
    for c in geo['lumbar']['coordinates']:
        if f'reserve_{c}' in l_act:
            internal |= np.abs(np.interp(t, t_act, act[:, l_act.index(f'reserve_{c}')])) > RESERVE_NM
    t_q, l_q, q = biomech.read_storage(work/'coordinates.mot')
    kinematic = np.zeros(len(t), bool)
    for c, spec in geo['lumbar']['coordinates'].items():
        kinematic |= near_limit(np.interp(t, t_q, q[:, l_q.index(c)]), spec)
    type2 = external | internal | kinematic
    reachable = ~type2 & finite
    m = {'type2_fraction': float(type2.mean()), 'type2_external_fraction': float(external.mean()),
         'type2_internal_fraction': float(internal.mean()), 'type2_kinematic_fraction': float(kinematic.mean())}
    series = {'times': t, 'reachable': reachable, 'finite': finite, 'type2_external': external, 'type2_internal': internal,
              'type2_kinematic': kinematic}
    for key, (joint, child) in geo['joints']['lumbar'].items():
        comp, ap, ml = project(labels, jr, joint, child, geo['lumbar']['tangents'][key]['axis'])
        series[f'{key}_compression_n'] = comp
        if key == 'L5_S1':
            series['L5_S1_ap_shear_n'], series['L5_S1_ml_shear_n'] = ap, ml
        if not reachable.any():
            continue
        m[f'{key}_compression_p95_bw'] = float(np.percentile(comp[reachable], 95)/bw)
        m[f'{key}_compression_mean_bw'] = float(comp[reachable].mean()/bw)
        if key == 'L5_S1':
            m['L5_S1_over_niosh_fraction'] = float(np.mean(comp[reachable] > geo['niosh_n']))
            m['L5_S1_ap_shear_p95_n'] = float(np.percentile(np.abs(ap[reachable]), 95))
            m['L5_S1_ml_shear_p95_n'] = float(np.percentile(np.abs(ml[reachable]), 95))
            for limit in SHEAR_N:
                m[f'L5_S1_ap_shear_over_{limit:g}_fraction'] = float(np.mean(np.abs(ap[reachable]) > limit))
    rows = np.array([int(np.argmin(np.abs(t_act-x))) for x in t])
    for name, members in geo['groups']['lumbar'].items():
        cols = [k for k, l in enumerate(l_act) if not l.startswith('reserve_') and any(l.startswith(p) for p in members)]
        if cols:
            series[f'{name}_activation'] = act[np.ix_(rows, cols)].mean(1)
            if reachable.any():
                m[f'{name}_activation_mean'] = float(series[f'{name}_activation'][reachable].mean())
    status = 'type1' if any(flags.values()) else 'measured'
    return {'status': status, 'type1_flags': flags, 'frames': int(len(t)), 'reachable_frames': int(reachable.sum()), 'metrics': m}, series


def _digest(contact):
    if contact is None:
        return None
    h = hashlib.sha256(np.asarray(contact[0], float).tobytes())
    for s in SIDES:
        for k in LANDMARKS:
            h.update(np.asarray(contact[1][s][k], bool).tobytes())
    return h.hexdigest()[:16]


def lumbar_fbls(run, work, mass=75., mode='evalkit', floor=None, contact=None, truth=None):
    """work: region directory (FBLS files, frames.npz, lumbar.json). contact: (times, {side: {landmark: bool}}) of the
    evaluation and floor its floor_eval (evalkit and truth modes); truth: external-loads xml (truth mode)."""
    if mode not in MODES:
        raise ValueError(mode)
    start = time.time(); work = Path(work).resolve()
    key = {'mode': mode, 'mass_kg': mass, 'floor_m': floor, 'contact': _digest(contact), 'truth': str(truth) if truth else None,
           'so_step': SO_STEP, 'protocol': PROTOCOL['protocol']}
    if (work/'lumbar.json').exists():
        old = json.loads((work/'lumbar.json').read_text())
        if old.get('key') == key and old.get('prepare_key') == json.loads((work/'receipt.json').read_text()).get('key'):
            return old
    receipt = prepare(run, work, mass, mode)
    loads = dynamics(work, mass, mode, floor, contact, truth)
    if mode == 'e3':
        t0, t1 = e3.TRIM, (receipt['frames']-1)/receipt['fps']-e3.TRIM
    else:
        times = np.load(work/'dynamics.npz')['times']; sel = fit_mod.scored(times)
        t0, t1 = float(times[sel][0]), float(times[sel][-1])
    solve(work, loads, t0, t1)
    region, series = classify(work, mass, t0, t1)
    np.savez_compressed(work/'frames.npz', **series)
    region.update(key=key, prepare_key=receipt['key'], window_s=[t0, t1], ik_marker_error=receipt['ik_marker_error'],
                  axial_markers_missing=receipt['axial_markers_missing'], calibration=receipt.get('calibration'),
                  seconds=round(time.time()-start, 1))
    (work/'lumbar.json').write_text(json.dumps(region, indent=1))
    return region


REGIONS = {'lumbar': lumbar_fbls, 'neck': None, 'shoulder': None}


def evaluate(run, out, mass=75., mode='evalkit', floor=None, contact=None, truth=None):
    """Every region of the protocol under out/<region>; reserved regions report what they need."""
    result = {'protocol': PROTOCOL['protocol']}
    for region, fn in REGIONS.items():
        result[region] = (fn(run, Path(out)/region, mass, mode, floor, contact, truth) if fn else
                          {'status': 'reserved', 'needs': PROTOCOL['regions'][region]['needs']})
    return result
