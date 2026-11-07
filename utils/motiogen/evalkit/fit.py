"""Shared front end of the OpenSim dimensions: scaled Rajagopal model, IK of the sparse joints, 100 Hz low-passed
coordinates and the frame-level IK gate. OpenSim python.

Calibration (registration of the marker offsets that the generated skeleton does not define the way the model does):
the most upright frames of the clip (not its first second, which a generated motion need not start upright in)
carry a lumbar-locked limb-only IK; axial markers are registered there (biomech.register_axial), and on each foot that
is planted in those frames the ankle and toe markers are registered with the foot levelled on the given floor."""
import json
from pathlib import Path

import numpy as np

from .spec import SPEC
import biomech

PREPARE = 'evalkit-upright-flatfoot-torso95-3'
C = SPEC['fit']


def fit(run, work, mass, floor):
    """run: any motion source biomech_sources.load reads (motion.npz run directory, PRIMAL factorial file, Kimodo case).
    Writes work/model.osim,
    work/coordinates.mot and the IK files (reused when work already holds the fit of the same run, mass, floor and
    preparation); returns the 100 Hz coordinates and the per-frame gate."""
    work = Path(work).resolve(); work.mkdir(parents=True, exist_ok=True)
    old = json.loads((work/'receipt.json').read_text()) if (work/'receipt.json').exists() else None
    key = {'run': str(run), 'mass_kg': mass, 'floor_m': floor, 'prepare': PREPARE}
    if old and all(old.get(k) == v for k, v in key.items()) and (work/'coordinates.mot').exists():
        receipt = old
        times, labels, coords = biomech.read_storage(work/'coordinates.mot')
    else:
        receipt = {**prepare(run, work, mass, floor), **key}
        (work/'receipt.json').write_text(json.dumps(receipt, indent=1))
        times, labels, coords = biomech.write_coordinates(work/'ik/ik.mot', work/'coordinates.mot')
    _, el, err = biomech.read_storage(next((work/'ik').glob('*_ik_marker_errors.sto')))
    peak = err[:, el.index('marker_error_max')]
    idx = np.clip(np.round((times-times[0])*receipt['fps']).astype(int), 0, len(peak)-1)
    return {'times': times, 'labels': labels, 'coords': coords, 'ik_max_error_m': peak[idx],
            'unmeasurable': peak[idx] > SPEC['ik_gate']['max_marker_error_m'], 'receipt': receipt}


def prepare(run, out, mass, floor):
    import biomech_sources
    points, names, fps = biomech_sources.load(run)
    receipt = {'fps': fps, 'frames': int(points.shape[0]), 'free_yaw': True}
    receipt['scaling'] = biomech.scaled_model(points, names, mass, out/'scaled.osim', free_yaw=True,
                                              torso_percentile=C['torso_length_percentile'])
    calib, receipt['calibration'] = upright_frames(points, names, fps)
    biomech.write_trc(points, names, fps, out/'calibration.trc', calib)
    limb_only = {n: (w if n in biomech.MARKERS else 0.) for n, w in biomech.WEIGHTS.items()}
    (out/'calib').mkdir(exist_ok=True)
    biomech.inverse_kinematics(out/'scaled.osim', out/'calibration.trc', out/'calib/ik.mot', calib[0]/fps, calib[-1]/fps,
                               limb_only, lock_lumbar=True)
    receipt['axial_markers_body_frame_m'] = biomech.register_axial(out/'scaled.osim', out/'calib/ik.mot', points, names, fps,
                                                                    calib, out/'axial.osim')
    receipt['feet'] = register_feet(out/'axial.osim', out/'calib/ik.mot', points, names, fps, floor, out/'model.osim')
    biomech.write_trc(points, names, fps, out/'markers.trc')
    (out/'ik').mkdir(exist_ok=True)
    biomech.inverse_kinematics(out/'model.osim', out/'markers.trc', out/'ik/ik.mot', 0., (points.shape[0]-1)/fps, biomech.WEIGHTS)
    receipt['ik_marker_error'] = biomech.marker_errors(out/'ik')
    return receipt


def _incline(v):
    return np.degrees(np.arccos(np.clip(v[..., 1]/np.linalg.norm(v, axis=-1), -1, 1)))


def upright_frames(points, names, fps):
    """Frames by summed trunk, thigh and shank inclination from vertical (OpenSim Y-up): the most upright ones within
    upright_tolerance_deg of the clip's minimum, at most calibration_s and at least min_calibration_frames of them,
    in time order."""
    p = lambda n: points[:, names.index(n)]
    mid = lambda a, b: .5*(p(a)+p(b))
    score = _incline(mid('right_shoulder', 'left_shoulder')-mid('right_hip', 'left_hip'))
    for s in ('right', 'left'):
        score = score+.5*(_incline(p(f'{s}_hip')-p(f'{s}_knee'))+_incline(p(f'{s}_knee')-p(f'{s}_ankle')))
    order = np.argsort(score, kind='stable')
    k = int(np.clip((score <= score[order[0]]+C['upright_tolerance_deg']).sum(),
                    min(len(score), C['min_calibration_frames']), max(2, int(round(C['calibration_s']*fps)))))
    frames = np.sort(order[:k])
    return [int(f) for f in frames], {'frames': k, 'score_deg_min': float(score[order[0]]), 'score_deg_max': float(score[frames].max()),
                                      'score_deg_clip_median': float(np.median(score))}


def register_feet(model_path, ik_mot, points, names, fps, floor, out_path):
    """Per side, in the calibration frames where the generated foot joint is within planted_height_m of the floor and
    slower than planted_speed_m_s: level the foot (calcn up-axis as close as the ankle angle allows to its default-pose
    direction; subtalar and mtp are locked in the model), then place the ankle (talus) and toe (toes) markers at the mean body-frame location of their generated
    joints. A side with fewer than min_planted_frames such frames keeps the joint-centre markers."""
    import opensim as osim
    from scipy.optimize import least_squares
    model = osim.Model(str(model_path)); state = model.initSystem(); cs = model.getCoordinateSet()
    bodies = model.getBodySet()
    up = lambda body: np.array([bodies.get(body).getTransformInGround(state).R().get(1, k) for k in range(3)])
    model.realizePosition(state)
    level = {s: up(f'calcn_{s}') for s in 'rl'}
    times, labels, data = biomech.read_storage(ik_mot)
    frames = [int(round(t*fps)) for t in times]
    speed = np.linalg.norm(np.gradient(points, 1/fps, axis=0), axis=-1)
    out = {}
    for s, side in (('r', 'right'), ('l', 'left')):
        j = names.index(f'{side}_foot')
        planted = [r for r, f in enumerate(frames)
                   if points[f, j, 1]-floor <= C['planted_height_m'] and speed[f, j] <= C['planted_speed_m_s']]
        if len(planted) < C['min_planted_frames']:
            out[s] = {'registered': False, 'planted_frames': len(planted)}
            continue
        sums = {'ankle': np.zeros(3), 'toe': np.zeros(3)}; tilt = []
        for r in planted:
            biomech._set(model, state, labels, data[r])

            def residual(x):
                cs.get(f'ankle_angle_{s}').setValue(state, x[0], False); model.realizePosition(state)
                return up(f'calcn_{s}')-level[s]
            sol = least_squares(residual, [cs.get(f'ankle_angle_{s}').getValue(state)])
            residual(sol.x); tilt.append(float(np.degrees(np.arcsin(min(1., np.linalg.norm(sol.fun))))))
            for key, body, src in (('ankle', f'talus_{s}', f'{side}_ankle'), ('toe', f'toes_{s}', f'{side}_foot')):
                g = osim.Vec3(*points[frames[r], names.index(src)])
                loc = model.getGround().findStationLocationInAnotherFrame(state, g, bodies.get(body))
                sums[key] += [loc.get(k) for k in range(3)]
        markers = model.updMarkerSet()
        loc = {k: (v/len(planted)).tolist() for k, v in sums.items()}
        markers.get(f'ankle_{s}').set_location(osim.Vec3(*loc['ankle'])); markers.get(f'toe_{s}').set_location(osim.Vec3(*loc['toe']))
        out[s] = {'registered': True, 'planted_frames': len(planted), 'level_residual_deg_max': max(tilt), 'body_frame_m': loc}
    model.printToXML(str(out_path))
    return out


def scored(times):
    """Frames scored by every dimension: the registered trim at both ends (filter and differentiation edges)."""
    dt = float(np.median(np.diff(times))); k = int(round(SPEC['trim_s']/dt))
    sel = np.zeros(len(times), bool); sel[k:len(times)-k] = True
    return sel
