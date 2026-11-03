"""Upper-body ergonomic evaluation (E3) of generated ARDY motions.

Runs in the OpenSim 4.6 Python environment; reuses the E1 pipeline helpers of biomech.py, which stays unchanged.
Three parts, each on its own model:
- posture: trunk, neck and upper-arm angles taken straight from the generated joint rotations (no model);
- lumbar: the E1 pipeline (per-segment scaling, joint-centre IK, contact-constrained ground reactions, inverse
  dynamics, static optimization) on the full-body lumbar spine model (FBLS, Raabe & Chaudhari 2016), plus joint
  reactions at the five lumbar intervertebral joints;
- neck: the HYOID head-neck model (Mortensen et al. 2018) posed frame by frame on the generated chest, neck and
  head rotations, inverse dynamics, static optimization of its 72 muscles and joint reactions at C7-T1 and C0-C1.
"""
import json
from pathlib import Path

import numpy as np
import opensim as osim

import biomech
from biomech import ROTATION, load, read_storage, segment_lengths

MODELS = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/third_party_models')
FBLS = MODELS/'fbls/FBLSmodel.osim'
HYOID = MODELS/'mortensen2018/Mortensen2018.osim'

# Generated joint frames: local x left, y up, z forward. OpenSim body axes: x forward, y up, z right.
BODY_AXES = np.array([[0., 0, -1], [0, 1, 0], [1, 0, 0]])
TRIM = .5

# ---------------------------------------------------------------- posture

# ISO 11226:2000 static posture zones; applying them frame by frame to dynamic motion is a convention (约定，未经推演).
ZONES = {'trunk_inclination_deg': (20., 60.), 'head_inclination_deg': (25., 85.), 'neck_flexion_deg': (25.,),
         'upper_arm_elevation_deg': (20., 60.), 'trunk_asymmetry_deg': (10.,), 'neck_asymmetry_deg': (10.,)}


def frames(run):
    """Joint positions and body-axis rotations in the OpenSim ground frame."""
    motion = np.load(Path(run)/'motion.npz')
    names = [str(n) for n in motion['joint_names']]
    rotations = ROTATION@motion['global_rotations']@BODY_AXES
    return motion['joints']@ROTATION.T, rotations, names, float(motion['fps'])


def zxy(relative):
    """Body-fixed Z-X-Y angles (extension about z, lateral bending about x, axial rotation about y), degrees."""
    a = np.arctan2(-relative[..., 0, 1], relative[..., 1, 1])
    b = np.arcsin(np.clip(relative[..., 2, 1], -1, 1))
    c = np.arctan2(-relative[..., 2, 0], relative[..., 2, 2])
    return np.rad2deg(a), np.rad2deg(b), np.rad2deg(c)


def inclination(vector, forward):
    """Angle of a segment from vertical, positive when it leans towards forward."""
    angle = np.rad2deg(np.arccos(np.clip(vector[..., 1]/np.linalg.norm(vector, axis=-1), -1, 1)))
    return np.where((vector*forward).sum(-1) >= 0, angle, -angle)


def posture_series(run):
    points, rotations, names, fps = frames(run)
    i = names.index
    pelvis, chest, head = rotations[:, i('pelvis')], rotations[:, i('Spine3')], rotations[:, i('head')]
    forward = pelvis[..., 0]
    trunk = {k: v for k, v in zip(('extension', 'bending', 'rotation'), zxy(np.swapaxes(pelvis, -1, -2)@chest))}
    neck = {k: v for k, v in zip(('extension', 'bending', 'rotation'), zxy(np.swapaxes(chest, -1, -2)@head))}
    arms = []
    for side in ('right', 'left'):
        humerus = points[:, i(f'{side}_elbow')]-points[:, i(f'{side}_shoulder')]
        local = np.einsum('tji,tj->ti', chest, humerus)
        arms.append(np.rad2deg(np.arccos(np.clip(-local[:, 1]/np.linalg.norm(local, axis=-1), -1, 1))))
    omega = np.gradient(head, 1/fps, axis=0)@np.swapaxes(head, -1, -2)
    head_speed = np.rad2deg(np.linalg.norm(np.stack([omega[:, 2, 1], omega[:, 0, 2], omega[:, 1, 0]], -1), axis=-1))
    series = {'trunk_inclination_deg': inclination(points[:, i('neck')]-points[:, i('pelvis')], forward),
              'trunk_flexion_deg': -trunk['extension'], 'trunk_bending_deg': trunk['bending'], 'trunk_rotation_deg': trunk['rotation'],
              'head_inclination_deg': inclination(head[..., 1], chest[..., 0]),
              'neck_flexion_deg': -neck['extension'], 'neck_bending_deg': neck['bending'], 'neck_rotation_deg': neck['rotation'],
              'upper_arm_elevation_r_deg': arms[0], 'upper_arm_elevation_l_deg': arms[1],
              'head_angular_speed_deg_s': head_speed}
    series['trunk_asymmetry_deg'] = np.maximum(np.abs(series['trunk_bending_deg']), np.abs(series['trunk_rotation_deg']))
    series['neck_asymmetry_deg'] = np.maximum(np.abs(series['neck_bending_deg']), np.abs(series['neck_rotation_deg']))
    series['upper_arm_elevation_deg'] = np.maximum(arms[0], arms[1])
    return series, fps


def posture_metrics(run):
    series, fps = posture_series(run)
    keep = slice(int(TRIM*fps), len(next(iter(series.values())))-int(TRIM*fps))
    metrics = {}
    for name, values in series.items():
        v = values[keep]
        metrics[f'{name}_mean'] = float(v.mean())
        metrics[f'{name}_abs_p95'] = float(np.percentile(np.abs(v), 95))
        metrics[f'{name}_rms'] = float(np.sqrt(np.mean(v**2)))
    for name, bounds in ZONES.items():
        v = series[name][keep]
        for bound in bounds:
            metrics[f'{name}_over_{bound:g}_fraction'] = float(np.mean(v > bound))
    return metrics


# ---------------------------------------------------------------- lumbar (FBLS)

LIMB = {'hip': ('hip', 'femur', 'hip'), 'knee': ('knee', 'tibia', 'knee'), 'ankle': ('ankle', 'talus', 'ankle'),
        'toe': ('mtp', 'toes', 'foot'), 'shoulder': ('acromial', 'humerus', 'shoulder'), 'elbow': ('elbow', 'ulna', 'elbow'),
        'wrist': ('radius_hand', 'hand', 'wrist')}
MARKERS = {f'{name}_{s}': (f'{joint}_{s}', f'{body}_{s}', f'{prefix}_{part}')
           for name, (joint, body, part) in LIMB.items() for s, prefix in (('r', 'right'), ('l', 'left'))}
AXIAL = {'pelvis_c': ('pelvis', 'pelvis'), 'spine_c': ('lumbar5', 'Spine'), 'spine1_c': ('lumbar2', 'spine1'),
         'spine2_c': ('torso', 'spine2'), 'chest_c': ('torso', 'Spine3'), 'neck_c': ('torso', 'neck')}
WEIGHTS = {**{name: 1. for name in MARKERS}, **{name: .3 for name in AXIAL}}
LUMBAR = ('flex_extension', 'lat_bending', 'axial_rotation')
LUMBAR_JOINTS = {'L5_S1': ('L5_S1_IVDjnt', 'lumbar5'), 'L4_L5': ('L4_L5_IVDjnt', 'lumbar4'), 'L3_L4': ('L3_L4_IVDjnt', 'lumbar3'),
                 'L2_L3': ('L2_L3_IVD_jnt', 'lumbar2'), 'L1_L2': ('L1_L2_IVD_jnt', 'lumbar1')}
SEGMENTS = {'pelvis': ['pelvis', 'sacrum'], 'femur': ['femur_r', 'femur_l', 'patella_r', 'patella_l'], 'tibia': ['tibia_r', 'tibia_l'],
            'foot': ['talus_r', 'talus_l', 'calcn_r', 'calcn_l', 'toes_r', 'toes_l'],
            'torso': ['lumbar5', 'lumbar4', 'lumbar3', 'lumbar2', 'lumbar1', 'torso', 'Abdomen', 'LB_wrap'],
            'humerus': ['humerus_r', 'humerus_l'], 'forearm': ['ulna_r', 'ulna_l', 'radius_r', 'radius_l', 'hand_r', 'hand_l']}
PELVIS = ('pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz')
ARM_COORDS = ('arm_flex', 'arm_add', 'arm_rot', 'elbow_flex', 'pro_sup')
# FBLS_SO_Actuators.xml of the model release: 1 N m reserves on legs and lumbar, 200 N m on the arms.
# Pelvis residual actuators as in E1 (1000) so that unsupported motions do not distort the muscle solution.
RESERVES = {**{c: 1e3 for c in PELVIS}, **{c: 1. for c in LUMBAR},
            **{f'{c}_{s}': 1. for c in ('hip_flexion', 'hip_adduction', 'hip_rotation', 'knee_angle', 'ankle_angle') for s in 'rl'},
            **{f'{c}_{s}': 200. for c in ARM_COORDS for s in 'rl'}}
NIOSH_N = 3400.


def model_lengths(model, state):
    c = lambda j: biomech._child_origin(model, state, j)
    n = lambda a, b: float(np.linalg.norm(c(a)-c(b)))
    return {'pelvis': n('hip_r', 'hip_l'), 'femur': n('hip_r', 'knee_r'), 'tibia': n('knee_r', 'ankle_r'), 'foot': n('ankle_r', 'mtp_r'),
            'torso': float(np.linalg.norm(.5*(c('acromial_r')+c('acromial_l'))-.5*(c('hip_r')+c('hip_l')))),
            'humerus': n('acromial_r', 'elbow_r'), 'forearm': n('elbow_r', 'radius_hand_r')}


def scaled_model(points, names, mass, path):
    model = osim.Model(str(FBLS)); state = model.initSystem()
    target, base = segment_lengths(points, names), model_lengths(model, state)
    factors = {k: target[k]/base[k] for k in target}
    scales = osim.ScaleSet()
    for segment, bodies in SEGMENTS.items():
        for body in bodies:
            s = osim.Scale(); s.setSegmentName(body); s.setScaleFactors(osim.Vec3(*[factors[segment]]*3)); s.setApply(True)
            scales.cloneAndAppend(s)
    model.scale(state, scales, True, mass)
    for name in PELVIS:
        coordinate = model.updCoordinateSet().get(name)
        coordinate.setRangeMin(-1e3); coordinate.setRangeMax(1e3); coordinate.setDefaultClamped(False)
    model.initSystem()
    for name, (joint, body, _) in MARKERS.items():
        location = model.getJointSet().get(joint).getChildFrame().findTransformInBaseFrame().p()
        model.addMarker(osim.Marker(name, model.getBodySet().get(body), location))
    for name, (body, _) in AXIAL.items():
        model.addMarker(osim.Marker(name, model.getBodySet().get(body), osim.Vec3(0, 0, 0)))
    model.finalizeConnections(); model.printToXML(str(path))
    return {'segment_factors': factors, 'target_lengths_m': target, 'model_lengths_m': base, 'mass_kg': mass}


def write_trc(points, names, fps, path, rows=None):
    labels = list(MARKERS)+list(AXIAL)
    index = [names.index(s) for s in [MARKERS[n][2] for n in MARKERS]+[AXIAL[n][1] for n in AXIAL]]
    table = osim.TimeSeriesTableVec3(); table.setColumnLabels(labels)
    for i in (range(points.shape[0]) if rows is None else rows):
        row = osim.RowVectorVec3(len(labels))
        for k, j in enumerate(index):
            row[k] = osim.Vec3(*points[i, j])
        table.appendRow(i/fps, row)
    table.addTableMetaDataString('DataRate', f'{fps:g}'); table.addTableMetaDataString('Units', 'm')
    osim.TRCFileAdapter.write(table, str(path))


def inverse_kinematics(model_path, trc, out, t0, t1, weights, locked=()):
    model = osim.Model(str(model_path))
    for name in locked:
        model.updCoordinateSet().get(name).setDefaultLocked(True)
    model.initSystem()
    tool = osim.InverseKinematicsTool()
    tool.setModel(model); tool.setMarkerDataFileName(str(trc)); tool.setStartTime(t0); tool.setEndTime(t1)
    tool.setOutputMotionFileName(str(out)); tool.set_report_errors(True); tool.setResultsDir(str(Path(out).parent))
    tasks = tool.getIKTaskSet()
    for name, w in weights.items():
        task = osim.IKMarkerTask(); task.setName(name); task.setWeight(w); task.setApply(w > 0); tasks.cloneAndAppend(task)
    tool.run()


def register_axial(model_path, ik_mot, points, names, fps, out_path):
    model = osim.Model(str(model_path)); state = model.initSystem()
    times, labels, data = read_storage(ik_mot)
    sums = {name: np.zeros(3) for name in AXIAL}
    for row in range(len(times)):
        biomech._set(model, state, labels, data[row])
        frame = int(round(times[row]*fps))
        for name, (body, source) in AXIAL.items():
            local = model.getGround().findStationLocationInAnotherFrame(state, osim.Vec3(*points[frame, names.index(source)]),
                                                                       model.getBodySet().get(body))
            sums[name] += [local.get(k) for k in range(3)]
    for name in AXIAL:
        model.updMarkerSet().get(name).set_location(osim.Vec3(*(sums[name]/len(times))))
    model.printToXML(str(out_path))
    return {name: (sums[name]/len(times)).tolist() for name in AXIAL}


def limit_frames(model_path, ik_mot, tolerance_deg=.5):
    """Frames at the range limit of each clamped, unconstrained coordinate."""
    model = osim.Model(str(model_path)); state = model.initSystem()
    _, labels, data = read_storage(ik_mot)
    coords, hits = model.getCoordinateSet(), {}
    for k, label in enumerate(labels):
        if not coords.contains(label):
            continue
        c = coords.get(label)
        if not c.getDefaultClamped() or c.isConstrained(state) or c.getDefaultLocked():
            continue
        values = np.deg2rad(data[:, k]) if c.getMotionType() == 1 else data[:, k]
        tol = np.deg2rad(tolerance_deg) if c.getMotionType() != 2 else 1e-3
        n = int(np.sum((values <= c.getRangeMin()+tol) | (values >= c.getRangeMax()-tol)))
        if n:
            hits[label] = n
    return hits


def prepare_lumbar(run, out, mass=75., calibration_seconds=1.):
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
    points, names, fps = load(run)
    receipt = {'run': str(run), 'model_source': str(FBLS), 'fps': fps, 'frames': int(points.shape[0])}
    receipt['scaling'] = scaled_model(points, names, mass, out/'scaled.osim')
    calib = list(range(int(calibration_seconds*fps)))
    write_trc(points, names, fps, out/'calibration.trc', calib)
    (out/'calib').mkdir(exist_ok=True)
    inverse_kinematics(out/'scaled.osim', out/'calibration.trc', out/'calib/ik.mot', 0., (len(calib)-1)/fps,
                       {n: (w if n in MARKERS else 0.) for n, w in WEIGHTS.items()}, locked=LUMBAR)
    receipt['axial_markers_body_frame_m'] = register_axial(out/'scaled.osim', out/'calib/ik.mot', points, names, fps, out/'model.osim')
    write_trc(points, names, fps, out/'markers.trc')
    (out/'ik').mkdir(exist_ok=True)
    inverse_kinematics(out/'model.osim', out/'markers.trc', out/'ik/ik.mot', 0., (points.shape[0]-1)/fps, WEIGHTS)
    receipt['ik_marker_error'] = biomech.marker_errors(out/'ik')
    receipt['frames_at_coordinate_limit'] = limit_frames(out/'model.osim', out/'ik/ik.mot')
    (out/'receipt.json').write_text(json.dumps(receipt, indent=1))
    return receipt


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x)**2)))


def lumbar_dynamics(out, mass):
    """E1 contact and inverse-dynamics pipeline on the FBLS model; trunk and shoulder moments from its inverse dynamics."""
    out = Path(out).resolve()
    times, labels, data = biomech.write_coordinates(out/'ik/ik.mot', out/'coordinates.mot')
    dt = float(np.median(np.diff(times)))
    kin = biomech.body_kinematics(osim.Model(str(out/'model.osim')), labels, data)
    F, M, C = biomech.required_wrench(kin, dt)
    points, heights = {}, {}
    for side in ('r', 'l'):
        points[side], heights[side] = biomech.sole_points(kin, side)
    floor, masks = biomech.contact_masks(heights)
    bw = mass*9.80665
    grf, rF, rM = biomech.estimate_grf(F, M, C, points, masks, floor, bw)
    biomech.write_loads(times, grf, floor, out/'grf.mot', out/'grf.xml')
    _, id_labels, id_data = biomech.inverse_dynamics(out/'model.osim', out/'coordinates.mot', out/'grf.xml', out, 'id.sto',
                                                     float(times[0]), float(times[-1]))
    np.savez_compressed(out/'dynamics.npz', times=times, required_force=F, required_moment=M, com=C, residual_force=rF,
                        residual_moment=rM, id=id_data, id_labels=np.array(id_labels))
    trim = slice(int(TRIM/dt), len(times)-int(TRIM/dt))
    col = lambda n: id_data[trim, id_labels.index(n)]
    moment = lambda c: next(l for l in id_labels if l.startswith(c+'_'))
    rf = np.linalg.norm(rF, axis=1)[trim]/bw
    metrics = {'residual_force_rms_bw': rms(rf), 'unsupported_fraction_5pct_bw': float(np.mean(rf > .05)),
               'required_vertical_force_mean_bw': float(F[trim, 1].mean()/bw)}
    for c in LUMBAR:
        values = col(moment(c))/mass
        metrics[f'lumbar_{c}_moment_rms_nm_kg'] = rms(values); metrics[f'lumbar_{c}_moment_peak_nm_kg'] = float(np.abs(values).max())
    shoulder = []
    for s in 'rl':
        vector = np.stack([col(moment(f'{c}_{s}')) for c in ('arm_flex', 'arm_add', 'arm_rot')], 1)/mass
        shoulder.append(np.linalg.norm(vector, axis=1))
        metrics[f'shoulder_{s}_moment_rms_nm_kg'] = rms(shoulder[-1])
        metrics[f'elbow_{s}_moment_rms_nm_kg'] = rms(col(moment(f'elbow_flex_{s}'))/mass)
    metrics['shoulder_moment_rms_nm_kg'] = .5*(metrics['shoulder_r_moment_rms_nm_kg']+metrics['shoulder_l_moment_rms_nm_kg'])
    metrics['shoulder_moment_peak_nm_kg'] = float(np.max(shoulder))
    return metrics


def add_reserves(model, state, reserves):
    """Reserve actuators on free coordinates. Millard2012 muscles (HYOID) become rigid-tendon without activation
    dynamics, so that a joint-reaction analysis driven by the static-optimization controls reproduces the
    static-optimization muscle forces; Thelen2003 muscles (FBLS) stay as released (their joint reactions use the
    forces file). Converting FBLS to Millard muscles adds passive forces of several hundred N per hip flexor."""
    coords = model.getCoordinateSet()
    eligible = [(name, force) for name, force in reserves.items()
                if coords.contains(name) and not coords.get(name).isConstrained(state) and not coords.get(name).getDefaultLocked()]
    muscles = model.updMuscles()
    for i in range(muscles.getSize()):
        if muscles.get(i).getConcreteClassName() == 'Millard2012EquilibriumMuscle':
            muscles.get(i).set_ignore_tendon_compliance(True); muscles.get(i).set_ignore_activation_dynamics(True)
    for name, force in eligible:
        actuator = osim.CoordinateActuator(name); actuator.setName(f'reserve_{name}')
        actuator.setOptimalForce(force); actuator.setMinControl(-1e4); actuator.setMaxControl(1e4)
        model.addForce(actuator)


def analyze(model_path, coordinates, loads_xml, results, analysis, t0, t1, name, controls=None):
    tool = osim.AnalyzeTool(); tool.setName(name); tool.setModelFilename(str(model_path))
    tool.setCoordinatesFileName(str(coordinates))
    if loads_xml:
        tool.setExternalLoadsFileName(str(loads_xml))
    if controls:
        tool.setControlsFileName(str(controls))
    tool.setLowpassCutoffFrequency(-1); tool.setInitialTime(t0); tool.setFinalTime(t1); tool.setResultsDir(str(results))
    tool.getAnalysisSet().cloneAndAppend(analysis); tool.setReplaceForceSet(False)
    setup = Path(results).parent/f'{name}_setup.xml'
    tool.printToXML(str(setup))
    osim.AnalyzeTool(str(setup)).run()


def static_optimization(work, model_so, coordinates, loads_xml, t0, t1, step):
    so = osim.StaticOptimization(); so.setName('so'); so.setStartTime(t0); so.setEndTime(t1)
    so.setUseModelForceSet(True); so.setActivationExponent(2); so.setUseMusclePhysiology(True); so.setStepInterval(step)
    analyze(model_so, coordinates, loads_xml, work/'so', so, t0, t1, 'so')


def joint_reactions(work, model_so, coordinates, loads_xml, t0, t1, joints, step, use_controls):
    """At the static-optimization frames. use_controls: driven by the static-optimization controls (HYOID, whose
    forces-file override returns NaN on most frames); otherwise the static-optimization forces file (FBLS)."""
    jr = osim.JointReaction(); jr.setName('jr'); jr.setStartTime(t0); jr.setEndTime(t1); jr.setStepInterval(step)
    if not use_controls:
        jr.setForcesFileName(str(work/'so/so_so_force.sto'))
    names, on, frame = osim.ArrayStr(), osim.ArrayStr(), osim.ArrayStr()
    for joint in joints:
        names.append(joint)
    on.append('child'); frame.append('child')
    jr.setJointNames(names); jr.setOnBody(on); jr.setInFrame(frame)
    analyze(model_so, coordinates, loads_xml, work/'jr', jr, t0, t1, 'jr',
            controls=work/'so/so_so_controls.xml' if use_controls else None)
    return read_storage(next((work/'jr').glob('*ReactionLoads.sto')))


def reaction_metrics(times, labels, data, joints, prefix, weight, t0, t1):
    keep = (times >= t0) & (times <= t1)
    finite = np.isfinite(data).all(1) & (np.abs(data) < 1e6).all(1)
    metrics = {f'{prefix}_reaction_valid_fraction': float(finite[keep].mean())}
    keep &= finite
    for key, (joint, child) in joints.items():
        f = lambda axis: data[keep, labels.index(f'{joint}_on_{child}_in_{child}_f{axis}')]
        compression, shear = f('y'), np.hypot(f('x'), f('z'))
        metrics[f'{prefix}_{key}_compression_rms_n'] = rms(compression)
        metrics[f'{prefix}_{key}_compression_peak_n'] = float(compression.max())
        metrics[f'{prefix}_{key}_compression_rms_bw'] = rms(compression)/weight
        metrics[f'{prefix}_{key}_shear_rms_n'] = rms(shear)
        metrics[f'{prefix}_{key}_shear_peak_n'] = float(shear.max())
    return metrics


def activation_metrics(work, model_so, prefix, groups=None):
    model = osim.Model(str(model_so)); model.initSystem()
    _, labels, act = read_storage(work/'so/so_so_activation.sto')
    muscles = [k for k, l in enumerate(labels) if model.getMuscles().contains(l)]
    reserves = {l: k for k, l in enumerate(labels) if l.startswith('reserve_')}
    a = act[:, muscles]
    metrics = {f'{prefix}_muscle_effort_sum_a2_mean': float((a**2).sum(1).mean()), f'{prefix}_muscle_activation_peak': float(a.max()),
               f'{prefix}_muscles': len(muscles)}
    for name, members in (groups or {}).items():
        cols = [k for k in muscles if any(labels[k].startswith(m) for m in members)]
        if cols:
            metrics[f'{prefix}_{name}_activation_mean'] = float(act[:, cols].mean())
            metrics[f'{prefix}_{name}_effort_sum_a2_mean'] = float((act[:, cols]**2).sum(1).mean())
    return metrics, labels, act, reserves


TRUNK_GROUPS = {'erector_spinae': ('IL_', 'LTpT_', 'LTpL_'), 'multifidus': ('MF_',), 'quadratus_lumborum': ('QL_',),
                'psoas': ('Ps_',), 'abdominals': ('rect_abd', 'EO', 'IO'), 'latissimus': ('LD_',)}


def lumbar(run, work, mass=75., step=10):
    """step 10 (10 Hz on the 6 Hz-filtered kinematics): the 324-muscle static optimization takes about 4 s per frame
    (约定，未经推演)."""
    work = Path(work).resolve()
    receipt = prepare_lumbar(run, work, mass)
    metrics = lumbar_dynamics(work, mass)
    t0, t1 = TRIM, (receipt['frames']-1)/receipt['fps']-TRIM
    model = osim.Model(str(work/'model.osim')); state = model.initSystem()
    add_reserves(model, state, RESERVES)
    model.finalizeConnections(); model.printToXML(str(work/'model_so.osim'))
    static_optimization(work, work/'model_so.osim', work/'coordinates.mot', work/'grf.xml', t0, t1, step)
    times, labels, data = joint_reactions(work, work/'model_so.osim', work/'coordinates.mot', work/'grf.xml', t0, t1,
                                          [j for j, _ in LUMBAR_JOINTS.values()], step, use_controls=False)
    weight = mass*9.80665
    metrics.update(reaction_metrics(times, labels, data, LUMBAR_JOINTS, 'lumbar', weight, t0, t1))
    keep = (times >= t0) & (times <= t1)
    l5 = data[keep, labels.index('L5_S1_IVDjnt_on_lumbar5_in_lumbar5_fy')]
    metrics['lumbar_L5_S1_over_niosh_fraction'] = float(np.mean(l5 > NIOSH_N))
    act_metrics, act_labels, act, reserves = activation_metrics(work, work/'model_so.osim', 'lumbar', TRUNK_GROUPS)
    metrics.update(act_metrics)
    lumbar_reserves = [reserves[f'reserve_{c}'] for c in LUMBAR if f'reserve_{c}' in reserves]
    metrics['lumbar_reserve_rms_nm'] = rms(act[:, lumbar_reserves]*RESERVES[LUMBAR[0]]) if lumbar_reserves else 0.
    for name in ('so/so_so_force.sto', 'so/so_so_controls.xml', 'calibration.trc', 'scaled.osim'):
        (work/name).unlink(missing_ok=True)
    return receipt, metrics


# ---------------------------------------------------------------- neck (HYOID)

BASE = ('gndpitch', 'gndroll', 'gndyaw', 'spine_tx', 'spine_ty', 'spine_tz')
NECK = ('pitch2', 'roll2', 'yaw2', 'pitch1', 'roll1', 'yaw1')
NECK_JOINTS = {'C7_T1': ('auxt1jnt', 'cerv7'), 'C0_C1': ('aux1jnt', 'skull')}
NECK_GROUPS = {'extensors': ('semi_', 'splen_', 'longissi_', 'iliocost_', 'trap', 'levator', 'obl_cap', 'rectcap'),
               'flexors': ('stern', 'cleid', 'scalenus', 'long_cap', 'long_col'),
               'hyoid': ('Genio', 'Mylo', 'Omo', 'Sterno', 'Stylo', 'digastric')}
REFERENCE_MASS = 79.54


def neck_model(path, mass):
    """HYOID with the impact actuators removed, the base unclamped and head-neck masses scaled to the subject (约定，未经推演)."""
    model = osim.Model(str(HYOID))
    forces = model.updForceSet()
    for name in ('Skull_Impact', 'Skull_Moment'):
        if forces.contains(name):
            forces.remove(forces.getIndex(name))
    for name in BASE:
        c = model.updCoordinateSet().get(name)
        c.setRangeMin(-1e4); c.setRangeMax(1e4); c.setDefaultClamped(False)
    ratio = mass/REFERENCE_MASS
    bodies = model.updBodySet()
    for i in range(bodies.getSize()):
        body = bodies.get(i)
        if body.getName() != 'spine':
            body.setMass(body.getMass()*ratio)
            inertia = body.getInertia(); m, p = inertia.getMoments(), inertia.getProducts()
            body.setInertia(osim.Inertia(m.get(0)*ratio, m.get(1)*ratio, m.get(2)*ratio, p.get(0)*ratio, p.get(1)*ratio, p.get(2)*ratio))
    model.finalizeConnections(); model.printToXML(str(path))


class Chain:
    """Forward kinematics of the HYOID model with its coupled vertebral coordinates evaluated explicitly."""

    def __init__(self, path):
        self.model = osim.Model(str(path)); self.state = self.model.initSystem()
        coords = self.model.getCoordinateSet()
        self.independent = [coords.get(n) for n in BASE+NECK]
        self.couplers = []
        constraints = self.model.getConstraintSet()
        for i in range(constraints.getSize()):
            c = osim.CoordinateCouplerConstraint.safeDownCast(constraints.get(i))
            if c is None:
                continue
            names = c.getIndependentCoordinateNames()
            self.couplers.append((coords.get(c.getDependentCoordinateName()), [coords.get(names.get(k)) for k in range(names.getSize())],
                                  c.getFunction()))
        self.bodies = {n: self.model.getBodySet().get(n) for n in ('spine', 'cerv2', 'skull')}
        joint = self.model.getJointSet().get('auxt1jnt')
        self.t1 = biomech_vec(joint.getParentFrame().findTransformInBaseFrame().p())
        self.bounds = (np.array([-np.inf]*len(BASE)+[c.getRangeMin() for c in self.independent[len(BASE):]]),
                       np.array([np.inf]*len(BASE)+[c.getRangeMax() for c in self.independent[len(BASE):]]))

    def set(self, q):
        for c, v in zip(self.independent, q):
            c.setValue(self.state, float(v), False)
        for _ in range(2):
            for dependent, inputs, function in self.couplers:
                x = osim.Vector(len(inputs), 0.)
                for k, c in enumerate(inputs):
                    x.set(k, c.getValue(self.state))
                dependent.setValue(self.state, function.calcValue(x), False)
        self.model.realizePosition(self.state)

    def pose(self, name):
        transform = self.bodies[name].getTransformInGround(self.state)
        return biomech_mat(transform.R()), biomech_vec(transform.p())

    def values(self):
        coords = self.model.getCoordinateSet()
        return {coords.get(i).getName(): coords.get(i).getValue(self.state) for i in range(coords.getSize())}


def biomech_vec(v):
    return np.array([v.get(k) for k in range(3)])


def biomech_mat(r):
    m = r.asMat33()
    return np.array([[m.get(a, b) for b in range(3)] for a in range(3)])


def fit_neck(chain, run):
    """Per-frame least squares: chest orientation, T1 at the generated neck joint, C2 and skull rotations relative to the
    chest equal to the generated neck and head rotations relative to the chest, both measured from the model's neutral."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    points, rotations, names, fps = frames(run)
    chest, neck, head = (rotations[:, names.index(n)] for n in ('Spine3', 'neck', 'head'))
    t1_target = points[:, names.index('neck')]
    chain.set(np.zeros(len(BASE)+len(NECK)))
    spine0 = chain.pose('spine')[0]
    rest = {n: spine0.T@chain.pose(n)[0] for n in ('cerv2', 'skull')}
    lower, upper = chain.bounds

    def residual(q, t):
        chain.set(q)
        Rs, ps = chain.pose('spine')
        out = [Rotation.from_matrix(chest[t].T@Rs).as_rotvec()]
        for name, frame in (('cerv2', neck[t]), ('skull', head[t])):
            out.append(Rotation.from_matrix((frame@rest[name]).T@chain.pose(name)[0]).as_rotvec())
        out.append((ps+Rs@chain.t1-t1_target[t])/.01)
        return np.concatenate(out)

    q = np.zeros(len(BASE)+len(NECK)); rows, errors = [], []
    for t in range(len(points)):
        q = np.clip(q, lower+1e-9, upper-1e-9)
        solution = least_squares(residual, q, args=(t,), bounds=(lower, upper), x_scale=1., xtol=1e-10, ftol=1e-10, max_nfev=400)
        q = solution.x
        r = solution.fun
        errors.append([np.rad2deg(np.linalg.norm(r[0:3])), np.rad2deg(np.linalg.norm(r[3:6])), np.rad2deg(np.linalg.norm(r[6:9])),
                       np.linalg.norm(r[9:12])*.01])
        chain.set(q); rows.append(chain.values())
    errors = np.array(errors)
    span = upper[len(BASE):]-lower[len(BASE):]
    neck_q = np.array([[row[n] for n in NECK] for row in rows])
    at_limit = (neck_q <= lower[len(BASE):]+1e-3*span) | (neck_q >= upper[len(BASE):]-1e-3*span)
    receipt = {'fps': fps, 'frames': len(rows),
               'fit_error': {'chest_deg_max': float(errors[:, 0].max()), 'c2_deg_rms': rms(errors[:, 1]), 'c2_deg_max': float(errors[:, 1].max()),
                             'skull_deg_rms': rms(errors[:, 2]), 'skull_deg_max': float(errors[:, 2].max()), 't1_m_max': float(errors[:, 3].max())},
               'frames_at_coordinate_limit': {n: int(at_limit[:, k].sum()) for k, n in enumerate(NECK) if at_limit[:, k].any()},
               'limit_fraction': float(at_limit.any(1).mean())}
    return rows, receipt, {'errors_chest_c2_skull_deg_t1_m': errors, 'at_limit': at_limit}


def write_motion(model, rows, fps, path):
    coords = model.getCoordinateSet()
    labels = list(rows[0])
    table = osim.TimeSeriesTable(); table.setColumnLabels(labels)
    for i, row in enumerate(rows):
        values = [np.rad2deg(row[n]) if coords.get(n).getMotionType() == 1 else row[n] for n in labels]
        table.appendRow(i/fps, osim.RowVector(values))
    table.addTableMetaDataString('inDegrees', 'yes')
    osim.STOFileAdapter.write(table, str(path))


def neck(run, work, mass=75., step=2):
    work = Path(work).resolve(); work.mkdir(parents=True, exist_ok=True)
    neck_model(work/'model.osim', mass)
    chain = Chain(work/'model.osim')
    rows, receipt, per_frame = fit_neck(chain, run)
    np.savez_compressed(work/'fit_frames.npz', fps=receipt['fps'], **per_frame)
    (work/'ik').mkdir(exist_ok=True)
    write_motion(chain.model, rows, receipt['fps'], work/'ik/ik.mot')
    times, labels, data = biomech.write_coordinates(work/'ik/ik.mot', work/'coordinates.mot')
    _, id_labels, id_data = biomech.inverse_dynamics(work/'model.osim', work/'coordinates.mot', None, work, 'id.sto',
                                                     float(times[0]), float(times[-1]))
    dt = float(np.median(np.diff(times)))
    trim = slice(int(TRIM/dt), len(times)-int(TRIM/dt))
    moment = lambda c: id_data[trim, id_labels.index(next(l for l in id_labels if l.startswith(c+'_')))]
    metrics = {}
    for c in NECK:
        metrics[f'neck_{c}_moment_rms_nm'] = rms(moment(c)); metrics[f'neck_{c}_moment_peak_nm'] = float(np.abs(moment(c)).max())
    t0, t1 = TRIM, (receipt['frames']-1)/receipt['fps']-TRIM
    model = osim.Model(str(work/'model.osim')); state = model.initSystem()
    add_reserves(model, state, {**{c: 1e4 for c in BASE}, **{c: 1. for c in NECK}})
    model.finalizeConnections(); model.printToXML(str(work/'model_so.osim'))
    static_optimization(work, work/'model_so.osim', work/'coordinates.mot', None, t0, t1, step)
    times, labels, data = joint_reactions(work, work/'model_so.osim', work/'coordinates.mot', None, t0, t1,
                                          [j for j, _ in NECK_JOINTS.values()], step, use_controls=True)
    metrics.update(reaction_metrics(times, labels, data, NECK_JOINTS, 'neck', mass*9.80665, t0, t1))
    act_metrics, act_labels, act, reserves = activation_metrics(work, work/'model_so.osim', 'neck', NECK_GROUPS)
    metrics.update(act_metrics)
    cols = [reserves[f'reserve_{c}'] for c in NECK if f'reserve_{c}' in reserves]
    metrics['neck_reserve_rms_nm'] = rms(act[:, cols]) if cols else 0.
    (work/'so/so_so_force.sto').unlink(missing_ok=True); (work/'so/so_so_controls.xml').unlink(missing_ok=True)
    (work/'receipt.json').write_text(json.dumps(receipt, indent=1))
    return receipt, metrics


# ---------------------------------------------------------------- validity

def validity(part, receipt, metrics):
    """Lumbar: the E1 criteria. Neck fit and reaction thresholds: 约定，未经推演. Reserves are outcomes, not validity:
    output-layer root corrections load the neck beyond its muscles at the corrected windows, and excluding those runs
    would remove the effect under test."""
    if part == 'lumbar':
        ik = receipt['ik_marker_error']
        return {'ik_rms_ok': ik['rms_mean_m'] <= .03, 'ik_max_ok': ik['max_m'] <= .10,
                'vertical_force_ok': abs(metrics['required_vertical_force_mean_bw']-1) <= .02,
                'reaction_ok': metrics['lumbar_reaction_valid_fraction'] >= .95}
    fit = receipt['fit_error']
    return {'chest_fit_ok': fit['chest_deg_max'] <= 1., 't1_fit_ok': fit['t1_m_max'] <= .005,
            'head_fit_ok': max(fit['c2_deg_rms'], fit['skull_deg_rms']) <= 2., 'limit_ok': receipt['limit_fraction'] <= .2,
            'reaction_ok': metrics['neck_reaction_valid_fraction'] >= .95}
