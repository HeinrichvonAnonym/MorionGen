#!/usr/bin/env python
"""Dynamic references (config/e_criterion_domain_20261105.json, 'moco'): a torque-driven, sphere-contact MocoTrack of a
kin30 clip in the E-pipeline scaled model; the solved motion's model markers then go through the full E pipeline.

  --solve --trial <id>     solve one clip, write develop_logs/test_runs/20261105_e_domain/moco/<id>/solution.sto
  --evaluate --trial <id>  E pipeline on the solved markers, compare with Moco's contact forces -> evaluation.json

Contact and actuator parameters below are conventions (约定，未经推演). OpenSim python with Moco.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import opensim as osim

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils/motiogen'))
import biomech  # noqa: E402

OUT = ROOT/'develop_logs/test_runs/20261105_e_domain/moco'
DISK = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/outputs/e_domain_20261105/moco')
KIN30 = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/addb_val_20261105/work/kin30')
VAL = ROOT/'develop_logs/test_runs/20261105_val_a'
WELD = ['mtp_r', 'mtp_l', 'subtalar_r', 'subtalar_l', 'radius_hand_r', 'radius_hand_l', 'patellofemoral_r', 'patellofemoral_l']
SPHERE_R = .03
CONTACT = dict(stiffness=1e6, dissipation=2., static_friction=.8, dynamic_friction=.8, viscous_friction=.5,
               transition_velocity=.2, hertz_smoothing=300., hunt_crossley_smoothing=50.)
TORQUE = {'lower': (250., 2.), 'lumbar': (250., 2.), 'arm': (80., 2.)}
CONTACT_N = 20.
RESIDUAL_BOUND, RESIDUAL_WEIGHT = 2000., .4
FPS_OUT = 30.


def spheres():
    """(name, body, location in body frame, landmark): E's sole points (biomech.SOLE)."""
    out = []
    for s in 'rl':
        for key, (body, (f0, f1), (l0, l1)) in biomech.SOLE.items():
            for tag, f, l in (('a', f0, l0), ('b', f1, l1)):
                out.append((f'{key}_{s}_{tag}', f'{body}_{s}', (f, 0., l), (s, key)))
    return out


def trial_floor(work):
    import e_rules
    times, _, _, kin = e_rules.kinematics(work)
    return float(e_rules.evaluate(times, kin, 75., 'R0')['floor'])


def build(work, floor, out, residuals=False):
    model = osim.Model(str(work/'model.osim'))
    model.updConstraintSet().clearAndDestroy()
    model.updForceSet().clearAndDestroy()
    model.finalizeConnections(); model.printToXML(str(out/'stripped.osim'))
    proc = osim.ModelProcessor(str(out/'stripped.osim'))
    proc.append(osim.ModOpReplaceJointsWithWelds(osim.StdVectorString(WELD)))
    model = proc.process()
    coords = model.getCoordinateSet()
    for i in range(coords.getSize()):
        c = coords.get(i); name = c.getName()
        if name.startswith('pelvis_'):
            if residuals:
                a = osim.CoordinateActuator(name); a.setName(f'res_{name}'); a.setOptimalForce(1.)
                a.setMinControl(-RESIDUAL_BOUND); a.setMaxControl(RESIDUAL_BOUND); model.addForce(a)
            continue
        group = 'lumbar' if name.startswith('lumbar') else 'arm' if any(k in name for k in ('arm_', 'elbow', 'pro_sup')) else 'lower'
        force, bound = TORQUE[group]
        a = osim.CoordinateActuator(name); a.setName(f'tau_{name}'); a.setOptimalForce(force)
        a.setMinControl(-bound); a.setMaxControl(bound); model.addForce(a)
    ground = osim.ContactHalfSpace(osim.Vec3(0, floor-SPHERE_R, 0), osim.Vec3(0, 0, -np.pi/2), model.getGround(), 'floor')
    model.addContactGeometry(ground)
    for name, body, loc, _ in spheres():
        geo = osim.ContactSphere(SPHERE_R, osim.Vec3(*loc), model.getBodySet().get(body), f'sphere_{name}')
        model.addContactGeometry(geo)
        f = osim.SmoothSphereHalfSpaceForce(f'contact_{name}', geo, ground)
        for k, v in CONTACT.items():
            getattr(f, f'set_{k}')(v)
        model.addForce(f)
    model.finalizeConnections()
    return model


def clip(trial, seconds):
    times, _, _ = biomech.read_storage(KIN30/trial/'coordinates.mot')
    mid = .5*(times[0]+times[-1]); half = min(seconds, times[-1]-times[0]-1.)/2
    return float(mid-half), float(mid+half)


KEEP = []


def problem(args, out, work, floor, t0, t1, stage):
    """MocoTrack study; stage 'residual' adds penalized pelvis residual actuators (continuation only)."""
    name = 'moco_model.osim' if stage == 'final' else 'moco_model_residual.osim'
    model = build(work, floor, out, residuals=stage == 'residual'); model.printToXML(str(out/name))
    track = osim.MocoTrack(); track.setName('e_domain')
    model_proc = osim.ModelProcessor(str(out/name))
    track.setModel(model_proc)
    table_proc = osim.TableProcessor(str(work/'coordinates.mot')); table_proc.append(osim.TabOpUseAbsoluteStateNames())
    track.setStatesReference(table_proc)
    track.set_states_global_tracking_weight(10.); track.set_allow_unused_references(True)
    track.set_track_reference_position_derivatives(True)
    weights = osim.MocoWeightSet()
    coords = model.getCoordinateSet()
    for i in range(coords.getSize()):
        weights.cloneAndAppend(osim.MocoWeight(f'{coords.get(i).getAbsolutePathString()}/speed', .01))
    track.set_states_weight_set(weights)
    track.set_apply_tracked_states_to_guess(bool(args.track_guess))
    track.set_control_effort_weight(args.effort)
    track.set_initial_time(t0); track.set_final_time(t1); track.set_mesh_interval(args.mesh)
    study = track.initialize()
    if stage == 'residual':
        effort = osim.MocoControlGoal.safeDownCast(study.updProblem().updGoal('control_effort'))
        for c in ('tilt', 'list', 'rotation', 'tx', 'ty', 'tz'):
            effort.setWeightForControl(f'/forceset/res_pelvis_{c}', RESIDUAL_WEIGHT)
    if args.track_guess:
        solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())
    else:
        solver = study.initCasADiSolver(); solver.set_num_mesh_intervals(int(round((t1-t0)/args.mesh)))
        solver.set_optim_sparsity_detection('random'); solver.set_optim_finite_difference_scheme('forward')
        solver.set_multibody_dynamics_mode('implicit' if args.implicit else 'explicit')
    solver.set_optim_convergence_tolerance(1e-3); solver.set_optim_constraint_tolerance(1e-3)
    solver.set_optim_max_iterations(args.iterations); solver.set_parallel(args.threads)
    if not args.track_guess:
        model.initSystem(); solver.setGuess(reference_guess(solver, model, table_proc))
    KEEP.append((track, model_proc, table_proc, model, study))
    return study, solver


def vec(x):
    return np.asarray(x.to_numpy() if hasattr(x, 'to_numpy') else x, dtype=float).ravel()


def reference_guess(solver, model, table_proc):
    """Guess on this solver's grid: tracked coordinate values (radians) and their finite-difference speeds."""
    ref = table_proc.processAndConvertToRadians(model)
    t = np.asarray(ref.getIndependentColumn()); guess = solver.createGuess(); tg = vec(guess.getTimeMat())
    states = set(guess.getStateNames())
    for label in ref.getColumnLabels():
        if label not in states:
            continue
        q = vec(ref.getDependentColumn(label))
        guess.setState(label, osim.Vector(np.interp(tg, t, q).tolist()))
        speed = label[:-len('/value')]+'/speed'
        if label.endswith('/value') and speed in states:
            guess.setState(speed, osim.Vector(np.interp(tg, t, np.gradient(q, t)).tolist()))
    return guess


def guess_from(solver, path):
    """Guess on this solver's grid from an earlier solution: its states, and the controls this problem has."""
    prev = osim.MocoTrajectory(str(path)); guess = solver.createGuess()
    guess.insertStatesTrajectory(prev.exportToStatesTable(), True)
    tp = vec(prev.getTimeMat()); tg = vec(guess.getTimeMat())
    for name in guess.getControlNames():
        guess.setControl(name, osim.Vector(np.interp(tg, tp, vec(prev.getControlMat(name))).tolist()))
    return guess


def pruned_guess(path):
    """Copy of a residual-stage solution without the pelvis residual controls, as a guess file for the final stage
    (createGuess on the solver of a MocoTrack-initialised study dereferences a freed problem)."""
    lines = Path(path).read_text().splitlines(); end = lines.index('endheader')
    header = lines[end+1].split('\t'); keep = [i for i, h in enumerate(header) if 'res_pelvis_' not in h]
    body = ['\t'.join(row.split('\t')[i] for i in keep) for row in lines[end+1:] if row.strip()]
    meta = [f'num_controls={sum(1 for h in (header[i] for i in keep) if h.startswith("/forceset/"))}' if l.startswith('num_controls=') else l
            for l in lines[:end]]
    out = Path(path).with_name('guess_final.sto'); out.write_text('\n'.join(meta+['endheader']+body)+'\n')
    return out


def outdir(args):
    return OUT/(args.trial if not args.tag else f'{args.trial}__{args.tag}')


def solve(args):
    out = outdir(args); out.mkdir(parents=True, exist_ok=True)
    work = KIN30/args.trial; floor = trial_floor(work); t0, t1 = clip(args.trial, args.seconds)
    receipt = {'trial': args.trial, 'floor_m': floor, 't0': t0, 't1': t1, 'mesh_interval_s': args.mesh, 'effort_weight': args.effort,
               'contact': CONTACT, 'sphere_radius_m': SPHERE_R, 'torque': TORQUE, 'welded': WELD, 'continuation': args.continuation,
               'implicit': args.implicit, 'track_guess': args.track_guess, 'stages': {}}
    stages = ['residual', 'final'] if args.continuation else ['final']
    for stage in stages:
        study, solver = problem(args, out, work, floor, t0, t1, stage)
        if stage == 'final' and args.continuation:
            if args.track_guess:
                solver.setGuessFile(str(pruned_guess(out/'solution_residual.sto')))
            else:
                solver.setGuess(guess_from(solver, out/'solution_residual.sto'))
        start = time.time()
        solution = study.solve()
        success = bool(solution.success())
        if not success:
            solution.unseal()
        rec = {'seconds': time.time()-start, 'success': success, 'status': solution.getStatus(),
               'iterations': int(solution.getNumIterations()), 'objective': float(solution.getObjective())}
        solution.write(str(out/('solution.sto' if stage == 'final' else 'solution_residual.sto')))
        if stage == 'residual':
            res = {c: float(np.sqrt(np.mean(vec(solution.getControlMat(f"/forceset/res_pelvis_{c}"))**2)))
                   for c in ('tilt', 'list', 'rotation', 'tx', 'ty', 'tz')}
            rec['pelvis_residual_rms'] = res
        receipt['stages'][stage] = rec
        print(json.dumps({stage: rec}), flush=True)
    receipt['success'] = receipt['stages']['final']['success']
    (out/'solve.json').write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt))


def moco_outputs(out):
    """Per-time markers (source names, Y-up), per-landmark vertical contact force and per-foot GRF of the solution."""
    model = osim.Model(str(out/'moco_model.osim')); model.initSystem()
    solution = osim.MocoTrajectory(str(out/'solution.sto'))
    states = solution.exportToStatesTrajectory(model)
    marker_names = list(biomech.MARKERS)+list(biomech.AXIAL)
    source = [biomech.MARKERS[n][2] for n in biomech.MARKERS]+[biomech.AXIAL[n][1] for n in biomech.AXIAL]
    markers = model.getMarkerSet(); forces = {n: osim.SmoothSphereHalfSpaceForce.safeDownCast(model.getForceSet().get(f'contact_{n}'))
                                              for n, *_ in spheres()}
    T = states.getSize(); times = np.zeros(T); P = np.zeros((T, len(marker_names), 3))
    land = {(s, k): np.zeros(T) for s in 'rl' for k in ('heel', 'toe')}; grf = {s: np.zeros((T, 3)) for s in 'rl'}
    for t in range(T):
        st = states.get(t); times[t] = st.getTime()
        model.realizeDynamics(st)
        for k, n in enumerate(marker_names):
            p = markers.get(n).getLocationInGround(st); P[t, k] = [p.get(i) for i in range(3)]
        for n, body, loc, (s, key) in spheres():
            v = forces[n].getRecordValues(st)
            f = -np.array([v.get(0), v.get(1), v.get(2)])
            land[(s, key)][t] += f[1]; grf[s][t] += f
    return times, P, source, land, grf


def evaluate(args):
    import e_rules
    out = outdir(args); receipt = json.loads((out/'solve.json').read_text())
    times, P, names, land, grf = moco_outputs(out)
    if np.median([g[:, 1].sum() for g in grf.values()]) < 0:
        land = {k: -v for k, v in land.items()}; grf = {k: -v for k, v in grf.items()}
    from scipy.interpolate import CubicSpline
    t_out = np.arange(times[0], times[-1]+1e-9, 1/FPS_OUT)
    joints = CubicSpline(times, P, axis=0)(t_out)
    row = next(r for r in json.loads((VAL/'selection.json').read_text())['trials'] if r['id'] == args.trial)
    mass = row['mass_kg']
    biomech.load = lambda _run: (joints, names, FPS_OUT)
    work = DISK/'work'/out.name
    rec = biomech.prepare(out, work, mass, free_yaw=True)
    biomech.write_coordinates(work/'ik/ik.mot', work/'coordinates.mot')
    et, _, _, kin = e_rules.kinematics(work)
    dt = float(np.median(np.diff(et))); sel = np.zeros(len(et), bool); sel[int(.5/dt):len(et)-int(.5/dt)] = True
    bw = mass*9.80665
    moco_t = et+times[0]
    vz = {s: np.interp(moco_t, times, grf[s][:, 1])/bw for s in 'rl'}
    moco_contact = {k: np.interp(moco_t, times, v) > CONTACT_N for k, v in land.items()}
    result = {'trial': args.trial, 'solve': receipt, 'ik': rec['ik_marker_error'], 'mass_kg': mass, 'frames': int(sel.sum()),
              'moco_vertical_grf_mean_bw': float(np.mean((vz['r']+vz['l'])[sel])), 'rules': {}}
    res0 = None
    for rule in e_rules.RULES:
        res = e_rules.evaluate(et, kin, mass, rule); res0 = res0 or res
        s = e_rules.summary(res, rule, sel, res0['masks'])
        ev = {k: res['grf'][k]['force'][:, 1]/bw for k in 'rl'}
        s['vertical_grf'] = {k: {'r': float(np.corrcoef(ev[k][sel], vz[k][sel])[0, 1]),
                                 'rmse_bw': float(np.sqrt(np.mean((ev[k][sel]-vz[k][sel])**2)))} for k in 'rl'}
        agree = [res['masks'][s_][k_][sel] == moco_contact[(s_, k_)][sel] for s_, k_ in moco_contact]
        s['contact_agreement'] = float(np.mean(agree))
        s['contact_confusion'] = {
            'e_contact_moco_free': float(np.mean([np.mean(res['masks'][s_][k_][sel] & ~moco_contact[(s_, k_)][sel]) for s_, k_ in moco_contact])),
            'e_free_moco_contact': float(np.mean([np.mean(~res['masks'][s_][k_][sel] & moco_contact[(s_, k_)][sel]) for s_, k_ in moco_contact]))}
        result['rules'][rule] = s
    reference = {s_: {k_: receipt['floor_m'] for k_ in ('heel', 'toe')} for s_ in 'rl'}
    frames = {'sel': sel, 'moco_vertical_grf_bw': vz['r']+vz['l']}
    result['v2'] = {'reference_m': receipt['floor_m'], 'world_floor_m': receipt['floor_m']-SPHERE_R, 'rules': {}}
    for rule in e_rules.RULES:
        res = e_rules.evaluate(et, kin, mass, rule, reference=reference, world_floor=receipt['floor_m']-SPHERE_R)
        st = e_rules.states(res['masks'], res['dt'])
        s = {k: v for k, v in e_rules.summary(res, rule, sel).items() if k != 'envelope_infeasible_share'}
        agree = [res['masks'][s_][k_][sel] == moco_contact[(s_, k_)][sel] for s_, k_ in moco_contact]
        s['contact_agreement'] = float(np.mean(agree))
        ev = {k: res['grf'][k]['force'][:, 1]/bw for k in 'rl'}
        s['vertical_grf'] = {k: {'r': float(np.corrcoef(ev[k][sel], vz[k][sel])[0, 1]),
                                 'rmse_bw': float(np.sqrt(np.mean((ev[k][sel]-vz[k][sel])**2)))} for k in 'rl'}
        result['v2']['rules'][rule] = s
        frames[f'{rule}_residual_bw'] = res['residual_bw']; frames[f'{rule}_lowest_elevation'] = res['lowest_elevation']
        frames[f'{rule}_vertical_grf_bw'] = ev['r']+ev['l']
        frames.update({f'{rule}_state_{k}': m for k, m in st.items()})
    np.savez_compressed(out/'frames_v2.npz', **frames)
    (out/'evaluation.json').write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ('trial', 'frames', 'moco_vertical_grf_mean_bw')} |
                     {r: {k: v[k] for k in ('envelope_infeasible_share', 'residual_force_rms_bw', 'contact_agreement')} for r, v in result['rules'].items()}))


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--trial', required=True); a.add_argument('--solve', action='store_true'); a.add_argument('--evaluate', action='store_true')
    a.add_argument('--seconds', type=float, default=2.); a.add_argument('--mesh', type=float, default=.04)
    a.add_argument('--effort', type=float, default=.01); a.add_argument('--iterations', type=int, default=3000)
    a.add_argument('--threads', type=int, default=4); a.add_argument('--continuation', action='store_true')
    a.add_argument('--implicit', action='store_true'); a.add_argument('--tag')
    a.add_argument('--track-guess', action='store_true', help="MocoTrack's own solver and tracked-state guess; final stage from a guess file")
    a.add_argument('--out', help='results directory relative to the repo'); a.add_argument('--disk', help='work directory for IK on the data disk')
    args = a.parse_args()
    global OUT, DISK
    OUT = ROOT/args.out if args.out else OUT; DISK = Path(args.disk) if args.disk else DISK
    os.nice(10)
    osim.Logger.setLevelString('info')
    if args.solve:
        solve(args)
    if args.evaluate:
        evaluate(args)


if __name__ == '__main__':
    main()
