"""Shared front end of the OpenSim dimensions: scaled Rajagopal model, IK of the sparse joints, 100 Hz low-passed
coordinates and the frame-level IK gate. OpenSim python.

Calibration (registration of the marker offsets that the generated skeleton does not define the way the model does):
the most upright frames of the clip (not its first second, which a generated motion need not start upright in)
carry a lumbar-locked limb-only IK; axial markers are registered there (biomech.register_axial), on each foot that
is planted in those frames the ankle and toe markers are registered with the foot levelled on the given floor, and the
remaining limb markers (fit.registered_limb_markers) are registered like the axial ones. Unregistered limb markers sit
at the model joint centres with weight 1 against 0.3 for the axial ones, so the full IK moves the pelvis to absorb
their landmark offsets and the pelvis / lumbar split takes the error."""
import json
from pathlib import Path

import numpy as np

from ..spec import SPEC
from .config import CONFIG, implementation_key, digest
from .motion import load
import biomech

PREPARE = 'evalkit-k2-arm-joint-centres'
C = SPEC['fit']


def fit(run, work, mass, floor):
    """run: any motion source biomech_sources.load reads (motion.npz run directory, PRIMAL factorial file, Kimodo case).
    Writes work/model.osim,
    work/coordinates.mot and the IK files (reused when work already holds the fit of the same run, mass, floor and
    preparation); returns the 100 Hz coordinates and the per-frame gate."""
    work = Path(work).resolve(); work.mkdir(parents=True, exist_ok=True)
    old = json.loads((work/'receipt.json').read_text()) if (work/'receipt.json').exists() else None
    motion = load(run)
    key = {'run': str(run), 'mass_kg': mass, 'floor_m': floor, 'prepare': PREPARE,
           'k_inputs': motion.provenance['inputs'], 'k_implementation': implementation_key(),
           'biomech_sha256': digest(biomech.__file__), 'model_sha256': digest(biomech.MODEL), 'fit_config': C}
    if old and all(old.get(k) == v for k, v in key.items()) and (work/'coordinates.mot').exists() and (work/'arm_quality.npz').exists():
        receipt = old
        times, labels, coords = biomech.read_storage(work/'coordinates.mot')
    else:
        receipt = {**prepare(run, work, mass, floor), **key}
        times, labels, coords = biomech.write_coordinates(work/'ik/ik.mot', work/'coordinates.mot')
        from .arms import assess
        points, names, fps = motion.legacy()
        quality = assess(work/'model.osim', points, names, fps, times, labels, coords, CONFIG['arm_fit'])
        prior = set(motion.provenance.get('degradation', {}).get('prior_joints', []))
        for side, prefix in (('l','left'), ('r','right')):
            if any(prefix+'_'+joint in prior for joint in ('shoulder','elbow','wrist','wrist_body')):
                for field in ('observed','angle_valid','position_valid','valid'):
                    quality[side][field][:] = False
        np.savez_compressed(work/'arm_quality.npz', times=times,
                            **{side+'_'+k: v for side, values in quality.items() for k, v in values.items()})
        p95 = lambda values: float(np.percentile(values[np.isfinite(values)],95)) if np.isfinite(values).any() else None
        receipt['arm_quality'] = {side: {'valid_share':float(q['valid'].mean()),
            'angle_valid_share':float(q['angle_valid'].mean()), 'position_valid_share':float(q['position_valid'].mean()),
            'elbow_p95_deg':p95(abs(q['elbow_bias_deg'])),
            'wrist_position_p95_m':p95(q['position_error_m'][:,2])} for side,q in quality.items()}
        receipt['arm_quality_scope'] = 'final filtered coordinates versus source joint geometry; anatomical accuracy and axial twist remain separate'
        (work/'receipt.json').write_text(json.dumps(receipt, indent=1))
    _, el, err = biomech.read_storage(next((work/'ik').glob('*_ik_marker_errors.sto')))
    peak = err[:, el.index('marker_error_max')]
    idx = np.clip(np.round((times-times[0])*receipt['fps']).astype(int), 0, len(peak)-1)
    unmeasurable = peak[idx] > SPEC['ik_gate']['max_marker_error_m']
    calibration_valid = receipt['calibration'].get('qualified', False)
    with np.load(work/'arm_quality.npz') as quality:
        arm_valid = {side:quality[side+'_valid'].copy() & calibration_valid for side in 'lr'}
    return {'times': times, 'labels': labels, 'coords': coords, 'ik_max_error_m': peak[idx], 'arm_valid':arm_valid,
            'unmeasurable': unmeasurable | (not calibration_valid),
            'position_fit_valid': ~unmeasurable, 'calibration_valid': calibration_valid, 'receipt': receipt}


def prepare(run, out, mass, floor):
    """Compare stable and expanded shoulder charts using observed input fidelity."""
    import shutil
    from .arms import policy, assess
    motion=load(run);settings=policy(motion.method)
    if not settings:
        return _prepare_candidate(run,out,mass,floor,{})
    points,names,fps=motion.legacy();candidates=[];failures={}
    variants=[('original_range',settings['initialization_range_deg'],'joint_centres'),
              ('expanded_range',settings['shoulder_flexion_range_deg'],'joint_centres'),
              ('registered_range',settings['initialization_range_deg'],'registered_markers')]
    for name,angle_range,registration in variants:
        folder=out/'arm_candidates'/name;folder.mkdir(parents=True,exist_ok=True)
        try:
            receipt=_prepare_candidate(run,folder,mass,floor,{**settings,'shoulder_flexion_range_deg':angle_range,'registration':registration})
            times,labels,coordinates=biomech.write_coordinates(folder/'ik/ik.mot',folder/'coordinates.mot')
            quality=assess(folder/'model.osim',points,names,fps,times,labels,coordinates,settings)
            valid_share=float(np.mean([q['valid'].mean() for q in quality.values()]))
            rmse=float(np.sqrt(np.mean([np.mean(q['position_error_m']**2) for q in quality.values()])))
            score={'source_fidelity_share':valid_share,'arm_joint_rmse_m':rmse,
                   'elbow_p95_deg':{s:float(np.nanpercentile(abs(q['elbow_bias_deg']),95)) for s,q in quality.items()}}
            if not np.isfinite(rmse):raise ValueError('Nonfinite candidate geometry')
            candidates.append((valid_share,-rmse,name,folder,receipt,score))
            (folder/'candidate.json').write_text(json.dumps({'receipt':receipt,'score':score},indent=2))
        except (ValueError,RuntimeError) as error:
            failures[name]=str(error)
    if not candidates:
        raise RuntimeError('All arm fitting candidates failed: '+str(failures))
    selected=max(candidates,key=lambda item:(item[0],item[1]))
    _,_,name,folder,receipt,_=selected
    for source in folder.iterdir():
        destination=out/source.name
        if source.is_dir():shutil.copytree(source,destination,dirs_exist_ok=True)
        elif source.name!='candidate.json':shutil.copy2(source,destination)
    receipt['arm_selection']={'selected':name,'criterion':'maximum observed angle-and-position fidelity share; tie-break minimum arm joint RMSE; calibration validity remains separate',
                              'candidates':{c[2]:c[5] for c in candidates},'failures':failures}
    return receipt


def _prepare_candidate(run, out, mass, floor, arm_policy):
    from .arms import policy, configure_model, inverse_kinematics
    motion = load(run)
    points, names, fps = motion.legacy()
    if not np.isfinite(points).all():
        raise ValueError('Full-body IK requires finite observed inputs; missing joints remain unavailable in K')
    receipt = {'fps': fps, 'frames': int(points.shape[0]), 'free_yaw': True, 'arm_policy':arm_policy}
    receipt['scaling'] = biomech.scaled_model(points, names, mass, out/'scaled.osim', free_yaw=True,
                                              torso_percentile=C['torso_length_percentile'])
    configure_model(out/'scaled.osim', arm_policy)
    calib, receipt['calibration'] = upright_frames(points, names, fps)
    if not receipt['calibration']['qualified']:
        import shutil
        shutil.copyfile(out/'scaled.osim', out/'model.osim')
        receipt['calibration']['fallback'] = 'model_joint_centres_unverified'
        biomech.write_trc(points, names, fps, out/'markers.trc')
        (out/'ik').mkdir(exist_ok=True)
        inverse_kinematics(out/'model.osim', out/'markers.trc', out/'ik/ik.mot', 0., (len(points)-1)/fps, biomech.WEIGHTS, arm_policy)
        receipt['ik_marker_error'] = biomech.marker_errors(out/'ik')
        return receipt
    biomech.write_trc(points, names, fps, out/'calibration.trc', calib)
    limb_only = {n: (w if n in biomech.MARKERS else 0.) for n, w in biomech.WEIGHTS.items()}
    (out/'calib').mkdir(exist_ok=True)
    inverse_kinematics(out/'scaled.osim', out/'calibration.trc', out/'calib/ik.mot', calib[0]/fps, calib[-1]/fps,
                       limb_only, arm_policy, lock_lumbar=True)
    receipt['axial_markers_body_frame_m'] = biomech.register_axial(out/'scaled.osim', out/'calib/ik.mot', points, names, fps,
                                                                    calib, out/'axial.osim')
    receipt['feet'] = register_feet(out/'axial.osim', out/'calib/ik.mot', points, names, fps, floor, out/'feet.osim')
    limbs = {n: (body, src) for n, (_, body, src) in biomech.MARKERS.items()
             if n.rsplit('_', 1)[0] in C['registered_limb_markers'] and src in names
             and not (arm_policy.get('registration')=='joint_centres' and n.rsplit('_',1)[0] in ('shoulder','elbow','wrist'))}
    receipt['limb_markers_body_frame_m'] = biomech.register_axial(out/'feet.osim', out/'calib/ik.mot', points, names, fps,
                                                                  calib, out/'model.osim', markers=limbs)
    biomech.write_trc(points, names, fps, out/'markers.trc')
    (out/'ik').mkdir(exist_ok=True)
    inverse_kinematics(out/'model.osim', out/'markers.trc', out/'ik/ik.mot', 0., (points.shape[0]-1)/fps, biomech.WEIGHTS, arm_policy)
    receipt['ik_marker_error'] = biomech.marker_errors(out/'ik')
    return receipt


def _incline(v):
    return np.degrees(np.arccos(np.clip(v[..., 1]/np.linalg.norm(v, axis=-1), -1, 1)))


def upright_frames(points, names, fps):
    """Filter absolute trunk/thigh eligibility, then rank by trunk/thigh/shank tilt.

    Prefer the tolerance band around the best eligible frame, expanding within
    the eligible pool to the minimum duration. Insufficient eligibility retains
    diagnostic frames and an explicitly unqualified calibration status.
    """
    p = lambda n: points[:, names.index(n)]
    mid = lambda a, b: .5*(p(a)+p(b))
    score = _incline(mid('right_shoulder', 'left_shoulder')-mid('right_hip', 'left_hip'))
    for s in ('right', 'left'):
        score = score+.5*(_incline(p(f'{s}_hip')-p(f'{s}_knee'))+_incline(p(f'{s}_knee')-p(f'{s}_ankle')))
    order = np.argsort(score, kind='stable')
    k = int(np.clip((score <= score[order[0]]+C['upright_tolerance_deg']).sum(),
                    min(len(score), C['min_calibration_frames']), max(2, int(round(C['calibration_s']*fps)))))
    frames = np.sort(order[:k])
    limits = CONFIG['calibration']
    trunk = _incline(mid('right_shoulder','left_shoulder')-mid('right_hip','left_hip'))
    thighs = np.maximum(*[_incline(p(f'{s}_hip')-p(f'{s}_knee')) for s in ('right','left')])
    eligible = order[(trunk[order] <= limits['max_trunk_incline_deg']) & (thighs[order] <= limits['max_thigh_incline_deg'])]
    required = max(C['min_calibration_frames'], int(np.ceil(limits['minimum_seconds']*fps - 1e-9)))
    enough = len(eligible) >= required
    if enough:
        preferred = int((score[eligible] <= score[eligible[0]]+C['upright_tolerance_deg']).sum())
        count = min(len(eligible), max(required, min(preferred, int(round(C['calibration_s']*fps)))))
        frames = np.sort(eligible[:count])
    return [int(f) for f in frames], {'frames': len(frames), 'qualified': bool(enough),
        'eligible_frames':len(eligible), 'required_frames':required,
        'reason': '' if enough else 'insufficient_upright_calibration',
        'score_deg_min': float(score[order[0]]), 'score_deg_max': float(score[frames].max()),
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


def foot_offset(points, names, fps, kin, times):
    """Per side, median over the scored frames of the fitted talus origin height minus median of the generated ankle
    joint height over the same trimmed window (OpenSim Y, m); 'mean' averages the sides."""
    sel = scored(times); k = int(round(SPEC['trim_s']*fps)); window = slice(k, max(k+1, points.shape[0]-k))
    out = {}
    for s, side in (('r', 'right'), ('l', 'left')):
        talus = kin['origin'][sel, kin['names'].index(f'talus_{s}'), 1]
        out[s] = float(np.median(talus)-np.median(points[window, names.index(f'{side}_ankle'), 1]))
    out['mean'] = .5*(out['r']+out['l'])
    return out


def scored(times):
    """Frames scored by every dimension: the registered trim at both ends (filter and differentiation edges)."""
    dt = float(np.median(np.diff(times))); k = int(round(SPEC['trim_s']/dt))
    sel = np.zeros(len(times), bool); sel[k:len(times)-k] = True
    return sel
